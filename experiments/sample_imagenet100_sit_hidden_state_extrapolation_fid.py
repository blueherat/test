"""Sample a frozen SiT after extrapolating its depth-8 and final hidden states."""

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
    from experiments.imagenet100_sit_hidden_state_extrapolation import (
        frozen_hidden_state_field,
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
    from imagenet100_sit_hidden_state_extrapolation import (
        frozen_hidden_state_field,
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


PROTOCOL = "imagenet100_sit_hidden_state_extrapolation_v1"
DEFAULT_CHECKPOINT = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/"
    "sit-s-2_seed0/checkpoints/step_00800000.pt"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/"
    "fid1k_v800_hidden_state_depth8/final"
)


def load_frozen_velocity_model(
    *,
    checkpoint_path: Path,
    weights: str,
    sit_module,
    source_metadata: dict,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, object]]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("protocol") != "imagenet100_sit_linear_flow_v1":
        raise ValueError(f"unexpected source protocol: {checkpoint.get('protocol')!r}")
    if checkpoint.get("official_sit") != source_metadata:
        raise ValueError("checkpoint and sampler use different SiT revisions")
    config = checkpoint["config"]
    if str(config.get("prediction_target", "velocity")) != "velocity":
        raise ValueError("hidden-state extrapolation requires a native velocity model")
    state_key = "ema" if weights == "ema" else "model"
    model = sit_module.SiT_models[str(config["model_name"])](
        input_size=LATENT_SHAPE[-1],
        num_classes=NUM_CLASSES,
        class_dropout_prob=float(config["cfg_dropout"]),
    )
    model.load_state_dict(checkpoint[state_key], strict=True)
    model.to(device).eval().requires_grad_(False)
    metadata: dict[str, object] = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "weights": weights,
        "state_key": state_key,
        "model_name": str(config["model_name"]),
        "cfg_dropout": float(config["cfg_dropout"]),
        "source_protocol": str(checkpoint["protocol"]),
        "prediction_target": "velocity",
    }
    del checkpoint
    gc.collect()
    return model, metadata


def conditional_hidden_state_field(
    model: torch.nn.Module,
    labels: torch.Tensor,
    *,
    internal_depth: int,
    mode: str,
    gamma: float,
    alpha: float,
    extrapolation_space: str,
    autocast_dtype: torch.dtype | None,
) -> tuple[object, dict[str, int]]:
    counter = {"nfe": 0, "backbone_forwards": 0, "final_layer_forwards": 0}

    def velocity(time_value: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        counter["nfe"] += 1
        counter["backbone_forwards"] += 1
        counter["final_layer_forwards"] += (
            2
            if mode in {"extrapolation", "interpolation"}
            and extrapolation_space == "output"
            else 1
        )
        times = time_value.expand(len(state))
        if autocast_dtype is None:
            return frozen_hidden_state_field(
                model,
                state,
                times,
                labels,
                internal_depth=internal_depth,
                latent_channels=LATENT_SHAPE[0],
                mode=mode,
                gamma=gamma,
                alpha=alpha,
                extrapolation_space=extrapolation_space,
            )
        with torch.autocast("cuda", dtype=autocast_dtype):
            return frozen_hidden_state_field(
                model,
                state,
                times,
                labels,
                internal_depth=internal_depth,
                latent_channels=LATENT_SHAPE[0],
                mode=mode,
                gamma=gamma,
                alpha=alpha,
                extrapolation_space=extrapolation_space,
            )

    return velocity, counter


@torch.inference_mode()
def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.mode != "extrapolation" and args.gamma != 0.0:
        raise ValueError("--gamma is only meaningful for extrapolation mode")
    if args.mode != "interpolation" and args.alpha != 0.0:
        raise ValueError("--alpha is only meaningful for interpolation mode")
    if args.mode == "extrapolation" and args.gamma < 0:
        raise ValueError("extrapolation gamma must be nonnegative")
    if args.mode == "interpolation" and not 0 <= args.alpha <= 1:
        raise ValueError("interpolation alpha must lie in [0,1]")

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    allocator = configure_cuda_allocator(
        device,
        limit_gib=args.cuda_allocator_limit_gib,
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
    model, model_metadata = load_frozen_velocity_model(
        checkpoint_path=checkpoint_path,
        weights=args.weights,
        sit_module=sit_module,
        source_metadata=source_metadata,
        device=device,
    )
    if not 1 <= args.internal_depth < len(model.blocks):
        raise ValueError("internal depth must precede the final transformer block")

    from diffusers.models import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        local_files_only=True,
    )
    vae.to(device).eval().requires_grad_(False)

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
    totals = {"nfe": 0, "backbone_forwards": 0, "final_layer_forwards": 0}
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
        velocity, counter = conditional_hidden_state_field(
            model,
            labels,
            internal_depth=args.internal_depth,
            mode=args.mode,
            gamma=args.gamma,
            alpha=args.alpha,
            extrapolation_space=args.extrapolation_space,
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
        if args.inter_batch_sleep > 0:
            time.sleep(args.inter_batch_sleep)
        if iteration == 0 or cursor == samples_per_rank or cursor % args.log_every == 0:
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "rank": rank,
                        "mode": args.mode,
                        "space": args.extrapolation_space,
                        "gamma": args.gamma,
                        "alpha": args.alpha,
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
            "extrapolation_space": args.extrapolation_space,
            "gamma": float(args.gamma),
            "alpha": float(args.alpha),
            "sample_count": samples_per_rank,
            "noise_sha256": noise_digest.hexdigest(),
            "label_sha256": label_digest.hexdigest(),
            "rank_npz": str(rank_path),
            "elapsed_seconds": elapsed,
            "inter_batch_sleep": float(args.inter_batch_sleep),
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
                    raise ValueError("duplicate global sample indices across DDP ranks")
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
        formula = {
            "final": "FinalLayer(h_final, c)",
            "internal": f"FinalLayer(h_{args.internal_depth}, c)",
            "extrapolation": (
                f"FinalLayer(h_final + gamma * (h_final - h_{args.internal_depth}), c)"
                if args.extrapolation_space == "hidden"
                else (
                    "FinalLayer(h_final,c) + gamma * (FinalLayer(h_final,c) - "
                    f"FinalLayer(h_{args.internal_depth},c))"
                )
            ),
            "interpolation": (
                f"FinalLayer((1-alpha) * h_final + alpha * h_{args.internal_depth}, c)"
                if args.extrapolation_space == "hidden"
                else (
                    "(1-alpha) * FinalLayer(h_final,c) + alpha * "
                    f"FinalLayer(h_{args.internal_depth},c)"
                )
            ),
        }[args.mode]
        manifest = {
            "format": "eqvae_imagenet100_sit_hidden_state_extrapolation_samples_v1",
            "protocol": PROTOCOL,
            "scope": "paired frozen-v800 hidden-state mixing screening",
            "model": model_metadata,
            "mode": args.mode,
            "internal_depth": int(args.internal_depth),
            "extrapolation_space": args.extrapolation_space,
            "gamma": float(args.gamma),
            "alpha": float(args.alpha),
            "formula": formula,
            "official_sit": source_metadata,
            "requested_samples": int(args.num_samples),
            "generated_for_ddp_divisibility": int(total_samples),
            "discarded_samples": int(total_samples - args.num_samples),
            "world_size": world_size,
            "per_rank_batch_size": int(args.per_rank_batch_size),
            "vae_decode_batch_size": int(args.vae_decode_batch_size),
            "cuda_allocator_limit_gib": float(args.cuda_allocator_limit_gib),
            "inter_batch_sleep": float(args.inter_batch_sleep),
            "global_seed": int(args.global_seed),
            "rank_seeds": [
                official_rank_seed(args.global_seed, world_size, i)
                for i in range(world_size)
            ],
            "rank_noise_sha256": [row["noise_sha256"] for row in rank_payloads],
            "rank_label_sha256": [row["label_sha256"] for row in rank_payloads],
            "label_histogram": histogram.tolist(),
            "label_sampling": "torch.randint after noise, matching official SiT",
            "cfg_scale": 1.0,
            "guidance": False,
            "same_noise_and_labels_across_conditions": True,
            "single_shared_backbone_forward_per_nfe": True,
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
            "rank_sampling_stats": [
                {
                    "rank": int(row["rank"]),
                    "elapsed_seconds": float(row["elapsed_seconds"]),
                    "nfe": int(row["nfe"]),
                    "backbone_forwards": int(row["backbone_forwards"]),
                    "final_layer_forwards": int(row["final_layer_forwards"]),
                }
                for row in rank_payloads
            ],
            "total_nfe": sum(int(row["nfe"]) for row in rank_payloads),
            "total_backbone_forwards": sum(
                int(row["backbone_forwards"]) for row in rank_payloads
            ),
            "total_final_layer_forwards": sum(
                int(row["final_layer_forwards"]) for row in rank_payloads
            ),
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
                    "space": args.extrapolation_space,
                    "gamma": args.gamma,
                    "alpha": args.alpha,
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
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--internal-depth", type=int, default=8)
    parser.add_argument(
        "--mode",
        choices=("final", "internal", "extrapolation", "interpolation"),
        required=True,
    )
    parser.add_argument("--extrapolation-space", choices=("hidden", "output"), default="hidden")
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--num-samples", type=int, default=1_000)
    parser.add_argument("--per-rank-batch-size", type=int, default=8)
    parser.add_argument("--vae-decode-batch-size", type=int, default=2)
    parser.add_argument("--cuda-allocator-limit-gib", type=float, default=6.0)
    parser.add_argument("--inter-batch-sleep", type=float, default=0.0)
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
