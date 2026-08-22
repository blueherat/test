#!/usr/bin/env python3
"""Sample a frozen SiT with jointly trained cumulative depth readouts."""

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
    from experiments.imagenet100_sit_joint_cumulative_heads import (
        create_joint_cumulative_parts,
        select_joint_cumulative_field,
        source_velocity_from_final_features,
    )
    from experiments.imagenet100_sit_multiscale_models import load_sit_field_model
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
    from experiments.train_imagenet100_sit_joint_cumulative_heads import PROTOCOL
except ModuleNotFoundError:
    from imagenet100_sit_joint_cumulative_heads import (
        create_joint_cumulative_parts,
        select_joint_cumulative_field,
        source_velocity_from_final_features,
    )
    from imagenet100_sit_multiscale_models import load_sit_field_model
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
    from train_imagenet100_sit_joint_cumulative_heads import PROTOCOL


DEFAULT_CHECKPOINT = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/"
    "sit-s-2_v800-frozen-joint-cumulative-heads_seed0/"
    "checkpoints/step_00010000.pt"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/"
    "joint_cumulative_rollout_v1/step10000_ema/final"
)


def load_joint_cumulative_model(
    *,
    checkpoint_path: Path,
    readout_weights: str,
    sit_module,
    source_metadata: dict,
    device: torch.device,
):
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    if checkpoint.get("protocol") != PROTOCOL:
        raise ValueError("unsupported joint cumulative checkpoint protocol")
    if checkpoint.get("official_sit") != source_metadata:
        raise ValueError("joint checkpoint uses a different official SiT revision")
    config = checkpoint["config"]
    source_path = Path(config["source_checkpoint"]).expanduser().resolve()
    source_digest = sha256_file(source_path)
    if source_digest != config["source_checkpoint_sha256"]:
        raise ValueError("source checkpoint SHA256 changed after joint training")
    source, semantics, live_metadata = load_sit_field_model(
        checkpoint_path=source_path,
        weights=str(config["source_state_key"]),
        sit_module=sit_module,
        source_metadata=source_metadata,
        device=device,
    )
    if semantics.prediction_target != "velocity":
        raise ValueError("joint cumulative sampling requires a native velocity source")
    depths = tuple(int(depth) for depth in config["depths"])
    prefix, readouts = create_joint_cumulative_parts(
        sit_module, source, depths=depths, latent_channels=LATENT_SHAPE[0]
    )
    state_key = "readouts_ema" if readout_weights == "ema" else "readouts"
    readouts.load_state_dict(checkpoint[state_key], strict=True)
    prefix.to(device).eval().requires_grad_(False)
    readouts.to(device).eval().requires_grad_(False)
    metadata = {
        "joint_checkpoint": str(checkpoint_path),
        "joint_checkpoint_sha256": sha256_file(checkpoint_path),
        "joint_step": int(checkpoint["step"]),
        "readout_weights": state_key,
        "depths": list(depths),
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": source_digest,
        "source_step": int(config["source_step"]),
        "source_state_key": str(config["source_state_key"]),
        "source": live_metadata,
        "training": config,
    }
    del checkpoint
    gc.collect()
    return prefix, readouts, metadata


def conditional_joint_cumulative_field(
    prefix,
    readouts,
    labels: torch.Tensor,
    *,
    mode: str,
    gamma: float,
    stage_depth: int | None,
    autocast_dtype: torch.dtype | None,
):
    counter = {"nfe": 0, "backbone_forwards": 0, "readout_stack_forwards": 0}

    def velocity(time_value: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        counter["nfe"] += 1
        counter["backbone_forwards"] += 1
        counter["readout_stack_forwards"] += 1
        times = time_value.expand(len(state))
        with torch.autocast(
            device_type="cuda",
            dtype=autocast_dtype or torch.bfloat16,
            enabled=autocast_dtype is not None,
        ):
            features, conditioning = prefix(state, times, labels)
            outputs, innovations = readouts(features, conditioning)
            strong = source_velocity_from_final_features(
                prefix.source,
                features[-1],
                conditioning,
                latent_channels=LATENT_SHAPE[0],
            )
        stage_index = None
        if mode == "stage":
            if stage_depth not in readouts.depths:
                raise ValueError(f"stage depth {stage_depth} is not in {readouts.depths}")
            stage_index = readouts.depths.index(stage_depth)
        return select_joint_cumulative_field(
            strong.float(),
            tuple(value.float() for value in outputs),
            tuple(value.float() for value in innovations),
            mode=mode,
            gamma=gamma,
            stage_index=stage_index,
        )

    return velocity, counter


@torch.inference_mode()
def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.mode != "last_extrapolation" and args.gamma != 0.0:
        raise ValueError("--gamma is only meaningful for last_extrapolation")
    if args.mode == "last_extrapolation" and args.gamma < 0:
        raise ValueError("last-innovation extrapolation requires gamma >= 0")
    if (args.mode == "stage") != (args.stage_depth is not None):
        raise ValueError("--stage-depth is required exactly when --mode=stage")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    allocator = configure_cuda_allocator(
        device, limit_gib=args.cuda_allocator_limit_gib
    )
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

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    sit_module, source_metadata = load_official_sit_module(
        args.official_sit_repo.expanduser().resolve(),
        verify_source=args.verify_sit_source,
    )
    prefix, readouts, model_metadata = load_joint_cumulative_model(
        checkpoint_path=checkpoint_path,
        readout_weights=args.readout_weights,
        sit_module=sit_module,
        source_metadata=source_metadata,
        device=device,
    )

    from diffusers.models import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse", local_files_only=True
    )
    vae.to(device).eval().requires_grad_(False)
    total_samples = official_total_samples(
        args.num_samples, args.per_rank_batch_size, world_size
    )
    samples_per_rank = total_samples // world_size
    iterations = samples_per_rank // args.per_rank_batch_size
    rank_images = np.empty((samples_per_rank, 256, 256, 3), dtype=np.uint8)
    rank_labels = np.empty(samples_per_rank, dtype=np.int16)
    rank_indices = np.empty(samples_per_rank, dtype=np.int64)
    autocast_dtype = None if args.precision == "fp32" else torch.bfloat16
    totals = {"nfe": 0, "backbone_forwards": 0, "readout_stack_forwards": 0}
    noise_digest = hashlib.sha256()
    label_digest = hashlib.sha256()
    cursor = 0
    preview: torch.Tensor | None = None
    started = time.perf_counter()
    global_batch_size = args.per_rank_batch_size * world_size

    for iteration in range(iterations):
        batch_size = args.per_rank_batch_size
        noise = torch.randn(batch_size, *LATENT_SHAPE, device=device)
        labels = torch.randint(0, NUM_CLASSES, (batch_size,), device=device)
        velocity, counter = conditional_joint_cumulative_field(
            prefix,
            readouts,
            labels,
            mode=args.mode,
            gamma=args.gamma,
            stage_depth=args.stage_depth,
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
            raise FloatingPointError(
                f"non-finite latent on rank {rank}, iteration {iteration}"
            )
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
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "rank": rank,
                        "mode": args.mode,
                        "gamma": args.gamma,
                        "generated_on_rank": cursor,
                        "samples_per_rank": samples_per_rank,
                        "elapsed_seconds": elapsed,
                        "images_per_second": cursor / elapsed,
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
    elapsed = time.perf_counter() - started
    atomic_json_dump(
        {
            "rank": rank,
            "rank_seed": rank_seed,
            "mode": args.mode,
            "gamma": float(args.gamma),
            "sample_count": samples_per_rank,
            "noise_sha256": noise_digest.hexdigest(),
            "label_sha256": label_digest.hexdigest(),
            "rank_npz": str(rank_path),
            "elapsed_seconds": elapsed,
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
            json.loads((output_dir / f"rank_{index:02d}.json").read_text())
            for index in range(world_size)
        ]
        rank_resource_usage = [
            load_rank_resource_usage(output_dir / f"rank_{index:02d}.json")
            for index in range(world_size)
        ]
        merged_images = np.empty((total_samples, 256, 256, 3), dtype=np.uint8)
        merged_labels = np.empty(total_samples, dtype=np.int16)
        seen = np.zeros(total_samples, dtype=np.bool_)
        for source_rank in range(world_size):
            with np.load(output_dir / f"rank_{source_rank:02d}.npz") as shard:
                indices = shard["indices"]
                if seen[indices].any():
                    raise ValueError("duplicate global sample indices across ranks")
                merged_images[indices] = shard["images"]
                merged_labels[indices] = shard["labels"]
                seen[indices] = True
        if not seen.all():
            raise ValueError("missing global sample indices after DDP merge")
        merged_images = merged_images[: args.num_samples]
        merged_labels = merged_labels[: args.num_samples]
        sample_path = output_dir / f"samples_unguided_n{args.num_samples}.npz"
        label_path = output_dir / f"sample_labels_unguided_n{args.num_samples}.npy"
        np.savez(sample_path, arr_0=merged_images)
        np.save(label_path, merged_labels, allow_pickle=False)
        histogram = np.bincount(merged_labels.astype(np.int64), minlength=NUM_CLASSES)
        final_depth = int(model_metadata["depths"][-1])
        previous_depth = int(model_metadata["depths"][-2])
        if args.mode == "stage":
            assert args.stage_depth is not None
            formula = f"v_d{args.stage_depth}"
        else:
            formula = {
                "strong": "v_source",
                "final": f"v_d{final_depth}",
                "last_extrapolation": (
                    f"v_d{final_depth} + gamma * "
                    f"(v_d{final_depth} - v_d{previous_depth})"
                ),
            }[args.mode]
        manifest = {
            "format": "eqvae_imagenet100_sit_joint_cumulative_samples_v1",
            "scope": "paired closed-loop screening of joint cumulative readouts",
            "model": model_metadata,
            "mode": args.mode,
            "gamma": float(args.gamma),
            "stage_depth": args.stage_depth,
            "formula": formula,
            "requested_samples": int(args.num_samples),
            "generated_for_ddp_divisibility": int(total_samples),
            "discarded_samples": int(total_samples - args.num_samples),
            "world_size": world_size,
            "per_rank_batch_size": int(args.per_rank_batch_size),
            "vae_decode_batch_size": int(args.vae_decode_batch_size),
            "global_seed": int(args.global_seed),
            "rank_seeds": [
                official_rank_seed(args.global_seed, world_size, index)
                for index in range(world_size)
            ],
            "rank_noise_sha256": [row["noise_sha256"] for row in rank_payloads],
            "rank_label_sha256": [row["label_sha256"] for row in rank_payloads],
            "label_histogram": histogram.tolist(),
            "label_sampling": "torch.randint after noise, matching official SiT",
            "cfg_scale": 1.0,
            "same_noise_and_labels_across_conditions": True,
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
            "pixel_quantization": "clamp(127.5*x + 128, 0, 255), NHWC uint8",
            "rank_resource_usage": rank_resource_usage,
            "rank_sampling_stats": rank_payloads,
            "samples": str(sample_path),
            "labels": str(label_path),
        }
        atomic_json_dump(manifest, output_dir / "sampling_manifest.json")
        for source_rank in range(world_size):
            (output_dir / f"rank_{source_rank:02d}.npz").unlink(missing_ok=True)
        print(
            json.dumps(
                {
                    "event": "complete",
                    "mode": args.mode,
                    "gamma": args.gamma,
                    "samples": str(sample_path),
                    "label_count_min": int(histogram.min()),
                    "label_count_max": int(histogram.max()),
                }
            ),
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--official-sit-repo", type=Path, default=DEFAULT_OFFICIAL_SIT_REPO)
    parser.add_argument("--readout-weights", choices=("ema", "model"), default="ema")
    parser.add_argument(
        "--mode",
        choices=("strong", "stage", "final", "last_extrapolation"),
        required=True,
    )
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--stage-depth", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=1_000)
    parser.add_argument("--per-rank-batch-size", type=int, default=8)
    parser.add_argument("--vae-decode-batch-size", type=int, default=2)
    parser.add_argument("--cuda-allocator-limit-gib", type=float, default=8.0)
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
