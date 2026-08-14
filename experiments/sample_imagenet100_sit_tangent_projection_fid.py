#!/usr/bin/env python3
"""Sample endpoint projections of the SiT tangent and exact frozen response.

For each paired sample this script computes the baseline endpoint ``z_b``, the
gamma-zero transported tangent ``xi``, and the exact gamma=1 frozen endpoint
``z_f``. It then decomposes ``z_f-z_b`` into its least-squares projection onto
``xi`` and the orthogonal remainder before decoding five paired conditions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torchvision.utils import save_image

try:
    from experiments.finite_guidance_dynamics import (
        decompose_along_reference,
        integrate_baseline_tangent_frozen,
        sample_cosine,
        sample_rms,
    )
    from experiments.run_imagenet100_sit_finite_guidance import (
        DEFAULT_V500,
        DEFAULT_X800,
        PairedFields,
        _load_pair,
    )
    from experiments.sample_imagenet100_sit_fid import (
        configure_cuda_allocator,
        decode_latents_in_chunks,
        load_rank_resource_usage,
        official_pixel_quantization,
        official_rank_seed,
        official_total_samples,
    )
    from experiments.train_imagenet100_sit_flow import (
        DEFAULT_OFFICIAL_SIT_REPO,
        LATENT_SHAPE,
        NUM_CLASSES,
        SD_VAE_SCALING_FACTOR,
        atomic_json_dump,
    )
except ModuleNotFoundError:
    from finite_guidance_dynamics import (
        decompose_along_reference,
        integrate_baseline_tangent_frozen,
        sample_cosine,
        sample_rms,
    )
    from run_imagenet100_sit_finite_guidance import (
        DEFAULT_V500,
        DEFAULT_X800,
        PairedFields,
        _load_pair,
    )
    from sample_imagenet100_sit_fid import (
        configure_cuda_allocator,
        decode_latents_in_chunks,
        load_rank_resource_usage,
        official_pixel_quantization,
        official_rank_seed,
        official_total_samples,
    )
    from train_imagenet100_sit_flow import (
        DEFAULT_OFFICIAL_SIT_REPO,
        LATENT_SHAPE,
        NUM_CLASSES,
        SD_VAE_SCALING_FACTOR,
        atomic_json_dump,
    )


DEFAULT_ANCHOR = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/"
    "sit-s-2_seed0/checkpoints/step_00800000.pt"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/"
    "tangent_projection_800k_v1/x800"
)
CONDITIONS = (
    "baseline",
    "tangent_raw",
    "tangent_parallel",
    "tangent_orthogonal",
    "frozen",
)
GEOMETRY_KEYS = (
    "projection_coefficient",
    "response_tangent_cosine",
    "response_rms",
    "tangent_rms",
    "parallel_rms",
    "orthogonal_rms",
    "parallel_energy_fraction",
    "orthogonal_energy_fraction",
    "reconstruction_max_abs",
    "orthogonality_cosine",
)


def _digest_update(digest: hashlib._Hash, value: torch.Tensor) -> None:
    digest.update(value.detach().cpu().contiguous().numpy().tobytes())


def _condition_latents(
    baseline: torch.Tensor,
    tangent: torch.Tensor,
    frozen: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    response = frozen - baseline
    coefficient, parallel, orthogonal = decompose_along_reference(response, tangent)
    response_energy = response.flatten(1).square().sum(dim=1)
    tiny = torch.finfo(response.dtype).tiny
    reconstruction = parallel + orthogonal - response
    geometry = {
        "projection_coefficient": coefficient,
        "response_tangent_cosine": sample_cosine(response, tangent),
        "response_rms": sample_rms(response),
        "tangent_rms": sample_rms(tangent),
        "parallel_rms": sample_rms(parallel),
        "orthogonal_rms": sample_rms(orthogonal),
        "parallel_energy_fraction": parallel.flatten(1).square().sum(dim=1)
        / response_energy.clamp_min(tiny),
        "orthogonal_energy_fraction": orthogonal.flatten(1).square().sum(dim=1)
        / response_energy.clamp_min(tiny),
        "reconstruction_max_abs": reconstruction.flatten(1).abs().amax(dim=1),
        "orthogonality_cosine": sample_cosine(parallel, orthogonal),
    }
    conditions = {
        "baseline": baseline,
        "tangent_raw": baseline + tangent,
        "tangent_parallel": baseline + parallel,
        "tangent_orthogonal": baseline + orthogonal,
        "frozen": frozen,
    }
    return conditions, geometry


def _summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "q05": float(np.quantile(array, 0.05)),
        "median": float(np.quantile(array, 0.50)),
        "q95": float(np.quantile(array, 0.95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _validate_geometry(geometry: dict[str, torch.Tensor]) -> None:
    for key, value in geometry.items():
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"non-finite projection geometry: {key}")
    if float(geometry["reconstruction_max_abs"].max()) > 2e-5:
        raise RuntimeError("parallel and orthogonal responses do not reconstruct frozen")
    energy_sum = (
        geometry["parallel_energy_fraction"]
        + geometry["orthogonal_energy_fraction"]
    )
    if not torch.allclose(energy_sum, torch.ones_like(energy_sum), atol=2e-5, rtol=2e-5):
        raise RuntimeError("projection energy fractions do not sum to one")


def _rank_sample_path(output_dir: Path, rank: int, condition: str) -> Path:
    return output_dir / "rank_shards" / f"rank_{rank:02d}_{condition}.npz"


def _rank_geometry_path(output_dir: Path, rank: int) -> Path:
    return output_dir / "rank_shards" / f"rank_{rank:02d}_geometry.npz"


def _merge_outputs(
    output_dir: Path,
    *,
    world_size: int,
    total_samples: int,
    requested_samples: int,
) -> tuple[dict[str, str], Path, Path]:
    sample_paths: dict[str, str] = {}
    seen = np.zeros(total_samples, dtype=np.bool_)
    merged_labels = np.empty(total_samples, dtype=np.int16)
    for rank in range(world_size):
        with np.load(_rank_geometry_path(output_dir, rank)) as shard:
            indices = shard["indices"]
            if seen[indices].any():
                raise ValueError("duplicate global sample indices across DDP ranks")
            seen[indices] = True
            merged_labels[indices] = shard["labels"]
    if not seen.all():
        raise ValueError("missing global sample indices after DDP merge")

    for condition in CONDITIONS:
        merged = np.empty((total_samples, 256, 256, 3), dtype=np.uint8)
        for rank in range(world_size):
            with np.load(_rank_sample_path(output_dir, rank, condition)) as shard:
                merged[shard["indices"]] = shard["images"]
        condition_dir = output_dir / condition
        condition_dir.mkdir(parents=True, exist_ok=True)
        sample_path = condition_dir / f"samples_unguided_n{requested_samples}.npz"
        np.savez(sample_path, arr_0=merged[:requested_samples])
        sample_paths[condition] = str(sample_path)
        del merged

    geometry_arrays: dict[str, np.ndarray] = {
        key: np.empty(total_samples, dtype=np.float32) for key in GEOMETRY_KEYS
    }
    merged_indices = np.arange(requested_samples, dtype=np.int64)
    for rank in range(world_size):
        with np.load(_rank_geometry_path(output_dir, rank)) as shard:
            indices = shard["indices"]
            for key in GEOMETRY_KEYS:
                geometry_arrays[key][indices] = shard[key]
    geometry_path = output_dir / "projection_geometry.npz"
    np.savez(
        geometry_path,
        indices=merged_indices,
        labels=merged_labels[:requested_samples],
        **{key: value[:requested_samples] for key, value in geometry_arrays.items()},
    )
    labels_path = output_dir / f"sample_labels_n{requested_samples}.npy"
    np.save(labels_path, merged_labels[:requested_samples], allow_pickle=False)
    return sample_paths, labels_path, geometry_path


def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.num_samples < 1 or args.per_rank_batch_size < 1:
        raise ValueError("sample counts must be positive")
    if args.heun_steps < 1 or args.vae_decode_batch_size < 1:
        raise ValueError("integration and decode batch sizes must be positive")
    if args.direction not in {"x800", "v500"}:
        raise ValueError("this experiment only supports x800 and v500")
    if args.share_visible_gpus and args.process_group_backend != "gloo":
        raise ValueError("shared visible GPUs require the Gloo process group")

    local_rank = int(os.environ["LOCAL_RANK"])
    visible_device_count = torch.cuda.device_count()
    if args.share_visible_gpus:
        device_index = local_rank % visible_device_count
    else:
        if local_rank >= visible_device_count:
            raise RuntimeError(
                "LOCAL_RANK exceeds the visible GPU count; use "
                "--share-visible-gpus only for independent-rank sampling"
            )
        device_index = local_rank
    device = torch.device("cuda", device_index)
    torch.cuda.set_device(device)
    allocator = configure_cuda_allocator(
        device,
        limit_gib=args.cuda_allocator_limit_gib,
    )
    if args.process_group_backend == "nccl":
        dist.init_process_group("nccl", device_id=device)
    else:
        dist.init_process_group(args.process_group_backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    rank_seed = official_rank_seed(args.global_seed, world_size, rank)
    torch.manual_seed(rank_seed)
    torch.cuda.manual_seed(rank_seed)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
    torch.set_float32_matmul_precision("high" if args.allow_tf32 else "highest")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    total_samples = official_total_samples(
        args.num_samples,
        args.per_rank_batch_size,
        world_size,
    )
    samples_per_rank = total_samples // world_size
    iterations = samples_per_rank // args.per_rank_batch_size
    initial_labels = torch.zeros(args.per_rank_batch_size, dtype=torch.long, device=device)
    fields, pair_metadata = _load_pair(args, initial_labels, device)
    if not isinstance(fields, PairedFields):
        raise TypeError("unexpected paired-field implementation")

    from diffusers.models import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        local_files_only=True,
    )
    vae.to(device).eval().requires_grad_(False)
    time_grid = torch.linspace(0.0, 1.0, args.heun_steps + 1, device=device)
    rank_images = {
        condition: np.empty((samples_per_rank, 256, 256, 3), dtype=np.uint8)
        for condition in CONDITIONS
    }
    rank_geometry = {
        key: np.empty(samples_per_rank, dtype=np.float32) for key in GEOMETRY_KEYS
    }
    rank_labels = np.empty(samples_per_rank, dtype=np.int16)
    rank_indices = np.empty(samples_per_rank, dtype=np.int64)
    noise_digest = hashlib.sha256()
    label_digest = hashlib.sha256()
    previews: dict[str, torch.Tensor] = {}
    cursor = 0
    global_batch_size = args.per_rank_batch_size * world_size
    started = time.perf_counter()

    for iteration in range(iterations):
        batch_size = args.per_rank_batch_size
        noise = torch.randn(batch_size, *LATENT_SHAPE, device=device)
        labels = torch.randint(0, NUM_CLASSES, (batch_size,), device=device)
        fields.labels = labels
        baseline, tangent, frozen = integrate_baseline_tangent_frozen(
            fields.anchor,
            fields.direction,
            noise,
            time_grid,
            gamma=1.0,
        )
        conditions, geometry = _condition_latents(baseline, tangent, frozen)
        _validate_geometry(geometry)
        for condition, latents in conditions.items():
            if not torch.isfinite(latents).all():
                raise FloatingPointError(f"non-finite endpoint for {condition}")
            with torch.inference_mode():
                decoded = decode_latents_in_chunks(
                    vae,
                    latents,
                    scaling_factor=SD_VAE_SCALING_FACTOR,
                    chunk_size=args.vae_decode_batch_size,
                )
            stop = cursor + batch_size
            rank_images[condition][cursor:stop] = official_pixel_quantization(decoded)
            if condition not in previews:
                previews[condition] = decoded[: min(8, len(decoded))].detach().cpu()
        stop = cursor + batch_size
        _digest_update(noise_digest, noise)
        _digest_update(label_digest, labels)
        rank_labels[cursor:stop] = labels.detach().cpu().numpy().astype(np.int16)
        base_index = iteration * global_batch_size
        rank_indices[cursor:stop] = (
            np.arange(batch_size, dtype=np.int64) * world_size + rank + base_index
        )
        for key in GEOMETRY_KEYS:
            rank_geometry[key][cursor:stop] = (
                geometry[key].detach().float().cpu().numpy()
            )
        cursor = stop
        if iteration == 0 or cursor == samples_per_rank or cursor % args.log_every == 0:
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "rank": rank,
                        "direction": args.direction,
                        "generated_on_rank": cursor,
                        "samples_per_rank": samples_per_rank,
                        "elapsed_seconds": elapsed,
                        "images_per_second": cursor / elapsed,
                        "peak_memory_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
                    }
                ),
                flush=True,
            )

    shard_dir = output_dir / "rank_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    for condition in CONDITIONS:
        np.savez(
            _rank_sample_path(output_dir, rank, condition),
            images=rank_images[condition],
            indices=rank_indices,
        )
        save_image(
            previews[condition],
            output_dir / f"preview_{condition}_rank_{rank:02d}.png",
            nrow=max(1, int(math.sqrt(len(previews[condition])))),
            normalize=True,
            value_range=(-1, 1),
        )
    np.savez(
        _rank_geometry_path(output_dir, rank),
        indices=rank_indices,
        labels=rank_labels,
        **rank_geometry,
    )
    elapsed = time.perf_counter() - started
    atomic_json_dump(
        {
            "rank": rank,
            "local_rank": local_rank,
            "visible_device_index": device_index,
            "rank_seed": rank_seed,
            "sample_count": samples_per_rank,
            "noise_sha256": noise_digest.hexdigest(),
            "label_sha256": label_digest.hexdigest(),
            "elapsed_seconds": elapsed,
            "anchor_forwards": fields.anchor_forwards,
            "other_forwards": fields.other_forwards,
            **allocator,
            "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
        output_dir / f"rank_{rank:02d}.json",
    )
    dist.barrier()

    if rank == 0:
        sample_paths, labels_path, geometry_path = _merge_outputs(
            output_dir,
            world_size=world_size,
            total_samples=total_samples,
            requested_samples=args.num_samples,
        )
        rank_payloads = [
            json.loads((output_dir / f"rank_{source_rank:02d}.json").read_text())
            for source_rank in range(world_size)
        ]
        with np.load(geometry_path) as geometry_file:
            geometry_summary = {
                key: _summary(geometry_file[key]) for key in GEOMETRY_KEYS
            }
        manifest = {
            "format": "eqvae_imagenet100_sit_tangent_projection_samples_v1",
            "scope": "paired endpoint projection mechanism test on ImageNet-100",
            "direction": args.direction,
            "anchor": pair_metadata["anchor"],
            "other": pair_metadata["other"],
            "weights": args.weights,
            "requested_samples": int(args.num_samples),
            "generated_for_ddp_divisibility": int(total_samples),
            "discarded_samples": int(total_samples - args.num_samples),
            "world_size": world_size,
            "visible_gpu_count": visible_device_count,
            "process_group_backend": args.process_group_backend,
            "share_visible_gpus": bool(args.share_visible_gpus),
            "per_rank_batch_size": int(args.per_rank_batch_size),
            "vae_decode_batch_size": int(args.vae_decode_batch_size),
            "cuda_allocator_limit_gib": float(args.cuda_allocator_limit_gib),
            "global_seed": int(args.global_seed),
            "rank_seeds": [payload["rank_seed"] for payload in rank_payloads],
            "rank_noise_sha256": [payload["noise_sha256"] for payload in rank_payloads],
            "rank_label_sha256": [payload["label_sha256"] for payload in rank_payloads],
            "same_noise_labels_and_endpoints_across_conditions": True,
            "projection_scope": "one least-squares scalar per sample over all latent C,H,W values",
            "conditions": {
                "baseline": "z_b",
                "tangent_raw": "z_b + xi",
                "tangent_parallel": "z_b + Proj_xi(z_f - z_b)",
                "tangent_orthogonal": "z_b + (I - Proj_xi)(z_f - z_b)",
                "frozen": "z_f",
            },
            "sample_paths": sample_paths,
            "labels": str(labels_path),
            "projection_geometry": str(geometry_path),
            "geometry_summary": geometry_summary,
            "sampler": {
                "path": "linear",
                "method": "fixed_heun",
                "steps": int(args.heun_steps),
                "interval": [0.0, 1.0],
                "precision": "fp32",
                "allow_tf32": bool(args.allow_tf32),
                "attention_backend": "math",
            },
            "pixel_quantization": "clamp(127.5*x + 128, 0, 255), NHWC uint8",
            "rank_resource_usage": [
                load_rank_resource_usage(output_dir / f"rank_{source_rank:02d}.json")
                for source_rank in range(world_size)
            ],
            "rank_sampling_stats": [
                {
                    "rank": int(payload["rank"]),
                    "elapsed_seconds": float(payload["elapsed_seconds"]),
                    "anchor_forwards": int(payload["anchor_forwards"]),
                    "other_forwards": int(payload["other_forwards"]),
                }
                for payload in rank_payloads
            ],
        }
        atomic_json_dump(manifest, output_dir / "sampling_manifest.json")
        for source_rank in range(world_size):
            for condition in CONDITIONS:
                _rank_sample_path(output_dir, source_rank, condition).unlink(
                    missing_ok=True
                )
            _rank_geometry_path(output_dir, source_rank).unlink(missing_ok=True)
        print(
            json.dumps(
                {
                    "event": "complete",
                    "direction": args.direction,
                    "samples": sample_paths,
                    "geometry_summary": geometry_summary,
                },
                indent=2,
            ),
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", choices=("x800", "v500"), required=True)
    parser.add_argument("--anchor-checkpoint", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--x800-checkpoint", type=Path, default=DEFAULT_X800)
    parser.add_argument("--v500-checkpoint", type=Path, default=DEFAULT_V500)
    parser.add_argument("--x400-checkpoint", type=Path, default=DEFAULT_X800)
    parser.add_argument("--v270-checkpoint", type=Path, default=DEFAULT_V500)
    parser.add_argument("--official-sit-repo", type=Path, default=DEFAULT_OFFICIAL_SIT_REPO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--per-rank-batch-size", type=int, default=32)
    parser.add_argument("--vae-decode-batch-size", type=int, default=4)
    parser.add_argument("--heun-steps", type=int, default=100)
    parser.add_argument("--global-seed", type=int, default=20260814)
    parser.add_argument("--cuda-allocator-limit-gib", type=float, default=9.5)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--process-group-backend",
        choices=("nccl", "gloo"),
        default="nccl",
    )
    parser.add_argument(
        "--share-visible-gpus",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Map logical ranks round-robin onto visible GPUs; use with Gloo.",
    )
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--verify-sit-source", action=argparse.BooleanOptionalAction, default=True)
    parser.set_defaults(math_attention=True)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
