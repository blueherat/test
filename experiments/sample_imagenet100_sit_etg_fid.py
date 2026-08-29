#!/usr/bin/env python3
"""Sample v800 with single-head, averaged, or error-triangulated guidance."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torchvision.utils import save_image

try:
    from experiments.imagenet100_sit_error_triangulated_guidance import (
        TARGETS,
        full_and_internal_predictions,
        guided_field,
        load_etg_model,
        predictions_to_velocity,
        time_bin_index,
    )
    from experiments.sample_imagenet100_sit_fid import (
        configure_cuda_allocator,
        decode_latents_in_chunks,
        load_rank_resource_usage,
        official_pixel_quantization,
        official_rank_seed,
        official_total_samples,
    )
    from experiments.sample_imagenet100_sit_flow import integrate_velocity
    from experiments.train_imagenet100_sit_flow import (
        DEFAULT_OFFICIAL_SIT_REPO,
        LATENT_SHAPE,
        NUM_CLASSES,
        SD_VAE_SCALING_FACTOR,
        atomic_json_dump,
        load_official_sit_module,
        sha256_file,
    )
except ModuleNotFoundError:
    from imagenet100_sit_error_triangulated_guidance import (
        TARGETS,
        full_and_internal_predictions,
        guided_field,
        load_etg_model,
        predictions_to_velocity,
        time_bin_index,
    )
    from sample_imagenet100_sit_fid import (
        configure_cuda_allocator,
        decode_latents_in_chunks,
        load_rank_resource_usage,
        official_pixel_quantization,
        official_rank_seed,
        official_total_samples,
    )
    from sample_imagenet100_sit_flow import integrate_velocity
    from train_imagenet100_sit_flow import (
        DEFAULT_OFFICIAL_SIT_REPO,
        LATENT_SHAPE,
        NUM_CLASSES,
        SD_VAE_SCALING_FACTOR,
        atomic_json_dump,
        load_official_sit_module,
        sha256_file,
    )


DEFAULT_CALIBRATION = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/"
    "error_triangulated_guidance_v800_depth8_v1/calibration.json"
)
DEFAULT_OUTPUT_DIR = DEFAULT_CALIBRATION.parent / "fid1k" / "baseline"
MODES = (
    "baseline",
    "single_velocity",
    "single_clean",
    "single_epsilon",
    "mean",
    "etg_scalar",
    "etg_channel",
    "private_velocity",
    "private_clean",
    "private_epsilon",
)


def load_calibration(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "eqvae_imagenet100_sit_etg_calibration_v1":
        raise ValueError(f"unsupported ETG calibration: {path}")
    return payload


def calibration_checkpoints(payload: dict) -> dict[str, Path]:
    heads = payload["model"]["heads"]
    return {target: Path(heads[target]["checkpoint"]) for target in TARGETS}


def conditional_etg_field(
    model,
    heads,
    labels: torch.Tensor,
    *,
    calibration: dict,
    mode: str,
    gamma: float,
    autocast_dtype: torch.dtype | None,
):
    counter = {
        "nfe": 0,
        "backbone_forwards": 0,
        "internal_head_forwards": 0,
    }
    source = calibration["model"]["source"]
    internal_depth = int(source["internal_depth"])
    denominator_floor = float(source["denominator_floor"])
    edges = calibration["config"]["time_bin_edges"]
    scalar_weights = torch.tensor(
        calibration["rollout_calibration"]["scalar"]["weights"],
        dtype=torch.float32,
        device=labels.device,
    )
    channel_weights = torch.tensor(
        calibration["rollout_calibration"]["channel"]["weights"],
        dtype=torch.float32,
        device=labels.device,
    )

    def velocity(time_value: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        counter["nfe"] += 1
        counter["backbone_forwards"] += 1
        counter["internal_head_forwards"] += len(TARGETS)
        times = time_value.expand(len(state))
        if autocast_dtype is None:
            full, native = full_and_internal_predictions(
                model,
                heads,
                state,
                times,
                labels,
                internal_depth=internal_depth,
            )
        else:
            with torch.autocast("cuda", dtype=autocast_dtype):
                full, native = full_and_internal_predictions(
                    model,
                    heads,
                    state,
                    times,
                    labels,
                    internal_depth=internal_depth,
                )
        weak = predictions_to_velocity(
            native,
            state=state,
            time_value=times,
            denominator_floor=denominator_floor,
        )
        if mode == "baseline":
            field_mode = "baseline"
            weights = None
            private_target = None
        elif mode.startswith("single_"):
            field_mode = mode
            weights = None
            private_target = None
        elif mode == "mean":
            field_mode = "mean"
            weights = None
            private_target = None
        else:
            index = time_bin_index(float(time_value.item()), edges)
            weights = (
                scalar_weights[index, :, 0]
                if mode in {"etg_scalar"}
                else channel_weights[index]
            )
            private_target = mode.removeprefix("private_") if mode.startswith("private_") else None
            field_mode = "private" if private_target is not None else "etg"
        return guided_field(
            full,
            weak,
            mode=field_mode,
            gamma=gamma,
            weights=weights,
            private_target=private_target,
        )

    return velocity, counter


@torch.inference_mode()
def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.mode == "baseline" and args.gamma != 0.0:
        raise ValueError("baseline requires gamma=0")
    if not np.isfinite(args.gamma):
        raise ValueError("gamma must be finite")

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    allocator = configure_cuda_allocator(device, limit_gib=args.cuda_allocator_limit_gib)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    rank_seed = official_rank_seed(args.global_seed, world_size, rank)
    torch.manual_seed(rank_seed)
    torch.cuda.manual_seed(rank_seed)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if args.allow_tf32 else "highest")

    calibration_path = args.calibration.expanduser().resolve()
    calibration = load_calibration(calibration_path)
    sit_module, source_metadata = load_official_sit_module(
        args.official_sit_repo.expanduser().resolve(),
        verify_source=args.verify_sit_source,
    )
    if calibration["official_sit"] != source_metadata:
        raise ValueError("calibration and sampler use different SiT revisions")
    model, heads, model_metadata = load_etg_model(
        checkpoint_paths=calibration_checkpoints(calibration),
        head_weights=str(calibration["model"]["head_weights"]),
        sit_module=sit_module,
        source_metadata=source_metadata,
        device=device,
    )
    if model_metadata != calibration["model"]:
        raise ValueError("current ETG checkpoints differ from calibration")

    from diffusers.models import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        local_files_only=True,
    )
    vae.to(device).eval().requires_grad_(False)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    total_samples = official_total_samples(
        args.num_samples,
        args.per_rank_batch_size,
        world_size,
    )
    samples_per_rank = total_samples // world_size
    iterations = samples_per_rank // args.per_rank_batch_size
    rank_images = np.empty((samples_per_rank, 256, 256, 3), dtype=np.uint8)
    rank_labels = np.empty(samples_per_rank, dtype=np.int16)
    rank_indices = np.empty(samples_per_rank, dtype=np.int64)
    autocast_dtype = None if args.precision == "fp32" else torch.bfloat16
    totals = {"nfe": 0, "backbone_forwards": 0, "internal_head_forwards": 0}
    noise_digest = hashlib.sha256()
    label_digest = hashlib.sha256()
    cursor = 0
    preview = None
    started = time.perf_counter()
    global_batch_size = args.per_rank_batch_size * world_size

    for iteration in range(iterations):
        batch_size = args.per_rank_batch_size
        noise = torch.randn(batch_size, *LATENT_SHAPE, device=device)
        labels = torch.randint(0, NUM_CLASSES, (batch_size,), device=device)
        velocity, counter = conditional_etg_field(
            model,
            heads,
            labels,
            calibration=calibration,
            mode=args.mode,
            gamma=args.gamma,
            autocast_dtype=autocast_dtype,
        )
        latents = integrate_velocity(
            noise,
            velocity,
            num_output_points=args.num_output_points,
            atol=args.atol,
            rtol=args.rtol,
        )
        if not torch.isfinite(latents).all():
            raise FloatingPointError(f"non-finite ETG latent on rank {rank}")
        decoded = decode_latents_in_chunks(
            vae,
            latents,
            scaling_factor=SD_VAE_SCALING_FACTOR,
            chunk_size=args.vae_decode_batch_size,
        )
        stop = cursor + batch_size
        noise_digest.update(noise.detach().cpu().contiguous().numpy().tobytes())
        label_digest.update(labels.detach().cpu().contiguous().numpy().tobytes())
        rank_images[cursor:stop] = official_pixel_quantization(decoded)
        rank_labels[cursor:stop] = labels.cpu().numpy().astype(np.int16, copy=False)
        base_index = iteration * global_batch_size
        rank_indices[cursor:stop] = (
            np.arange(batch_size, dtype=np.int64) * world_size + rank + base_index
        )
        cursor = stop
        for key in totals:
            totals[key] += int(counter[key])
        if preview is None:
            preview = decoded[: min(16, len(decoded))].detach().cpu()
        if iteration == 0 or cursor == samples_per_rank or cursor % args.log_every == 0:
            print(
                json.dumps(
                    {
                        "rank": rank,
                        "mode": args.mode,
                        "gamma": args.gamma,
                        "generated_on_rank": cursor,
                        "samples_per_rank": samples_per_rank,
                        "elapsed_seconds": time.perf_counter() - started,
                        "last_batch_nfe": int(counter["nfe"]),
                    }
                ),
                flush=True,
            )

    rank_path = output_dir / f"rank_{rank:02d}.npz"
    np.savez(rank_path, images=rank_images, labels=rank_labels, indices=rank_indices)
    assert preview is not None
    save_image(
        preview,
        output_dir / f"preview_rank_{rank:02d}.png",
        nrow=4,
        normalize=True,
        value_range=(-1, 1),
    )
    atomic_json_dump(
        {
            "rank": rank,
            "rank_seed": rank_seed,
            "mode": args.mode,
            "gamma": float(args.gamma),
            "sample_count": samples_per_rank,
            "noise_sha256": noise_digest.hexdigest(),
            "label_sha256": label_digest.hexdigest(),
            **totals,
            **allocator,
            "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
        output_dir / f"rank_{rank:02d}.json",
    )
    dist.barrier()

    if rank == 0:
        rank_payloads = [
            json.loads((output_dir / f"rank_{i:02d}.json").read_text(encoding="utf-8"))
            for i in range(world_size)
        ]
        rank_resource_usage = [
            load_rank_resource_usage(output_dir / f"rank_{i:02d}.json")
            for i in range(world_size)
        ]
        merged_images = np.empty((total_samples, 256, 256, 3), dtype=np.uint8)
        merged_labels = np.empty(total_samples, dtype=np.int16)
        seen = np.zeros(total_samples, dtype=np.bool_)
        for source_rank in range(world_size):
            with np.load(output_dir / f"rank_{source_rank:02d}.npz") as shard:
                indices = shard["indices"]
                if seen[indices].any():
                    raise ValueError("duplicate global sample indices")
                merged_images[indices] = shard["images"]
                merged_labels[indices] = shard["labels"]
                seen[indices] = True
        if not seen.all():
            raise ValueError("missing global sample indices")
        merged_images = merged_images[: args.num_samples]
        merged_labels = merged_labels[: args.num_samples]
        sample_path = output_dir / f"samples_unguided_n{args.num_samples}.npz"
        label_path = output_dir / f"sample_labels_unguided_n{args.num_samples}.npy"
        np.savez(sample_path, arr_0=merged_images)
        np.save(label_path, merged_labels, allow_pickle=False)
        histogram = np.bincount(merged_labels.astype(np.int64), minlength=NUM_CLASSES)
        manifest = {
            "format": "eqvae_imagenet100_sit_etg_samples_v1",
            "scope": "paired v800 depth-8 ETG FID screening",
            "calibration": str(calibration_path),
            "calibration_sha256": sha256_file(calibration_path),
            "model": model_metadata,
            "mode": args.mode,
            "gamma": float(args.gamma),
            "requested_samples": int(args.num_samples),
            "generated_for_ddp_divisibility": int(total_samples),
            "world_size": world_size,
            "per_rank_batch_size": int(args.per_rank_batch_size),
            "vae_decode_batch_size": int(args.vae_decode_batch_size),
            "cuda_allocator_limit_gib": float(args.cuda_allocator_limit_gib),
            "global_seed": int(args.global_seed),
            "rank_seeds": [official_rank_seed(args.global_seed, world_size, i) for i in range(world_size)],
            "rank_noise_sha256": [row["noise_sha256"] for row in rank_payloads],
            "rank_label_sha256": [row["label_sha256"] for row in rank_payloads],
            "label_histogram": histogram.tolist(),
            "cfg_scale": 1.0,
            "same_noise_and_labels_across_conditions": True,
            "single_shared_backbone_forward_per_nfe": True,
            "three_internal_head_forwards_per_nfe": True,
            "sampler": {
                "path": "linear",
                "method": "dopri5",
                "interval": [0.0, 1.0],
                "num_output_points": int(args.num_output_points),
                "atol": float(args.atol),
                "rtol": float(args.rtol),
                "precision": args.precision,
                "allow_tf32": bool(args.allow_tf32),
            },
            "rank_resource_usage": rank_resource_usage,
            "total_nfe": sum(int(row["nfe"]) for row in rank_payloads),
            "total_backbone_forwards": sum(int(row["backbone_forwards"]) for row in rank_payloads),
            "total_internal_head_forwards": sum(int(row["internal_head_forwards"]) for row in rank_payloads),
            "samples": str(sample_path),
            "labels": str(label_path),
        }
        atomic_json_dump(manifest, output_dir / "sampling_manifest.json")
        for source_rank in range(world_size):
            (output_dir / f"rank_{source_rank:02d}.npz").unlink(missing_ok=True)
        print(json.dumps({"event": "complete", "samples": str(sample_path)}), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    del model, heads, vae
    gc.collect()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--official-sit-repo", type=Path, default=DEFAULT_OFFICIAL_SIT_REPO)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--num-samples", type=int, default=1_000)
    parser.add_argument("--per-rank-batch-size", type=int, default=8)
    parser.add_argument("--vae-decode-batch-size", type=int, default=2)
    parser.add_argument("--cuda-allocator-limit-gib", type=float, default=6.0)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-output-points", type=int, default=250)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--log-every", type=int, default=256)
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verify-sit-source", action=argparse.BooleanOptionalAction, default=True)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
