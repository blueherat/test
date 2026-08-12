"""Sample a fixed linear mixture of two 400K SiT velocity fields.

The two checkpoints are evaluated on the same ODE state and class label. Their
outputs are first converted to the common linear-flow velocity space and then
mixed as ``anchor + scale * (other - anchor)``. Separate invocations with the
same sampling protocol receive identical initial noise and labels.
"""

from __future__ import annotations

import argparse
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
    from experiments.imagenet100_sit_static_pair import (
        DUAL_OUTPUT_PROTOCOL,
        FieldSemantics,
        output_to_field_velocity,
        resolve_field_semantics,
        static_pair_velocity,
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
    from experiments.train_imagenet100_sit_dual_output import create_dual_output_sit
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
    from imagenet100_sit_static_pair import (
        DUAL_OUTPUT_PROTOCOL,
        FieldSemantics,
        output_to_field_velocity,
        resolve_field_semantics,
        static_pair_velocity,
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
    from train_imagenet100_sit_dual_output import create_dual_output_sit
    from train_imagenet100_sit_flow import (
        DEFAULT_OFFICIAL_SIT_REPO,
        LATENT_SHAPE,
        NUM_CLASSES,
        SD_VAE_SCALING_FACTOR,
        atomic_json_dump,
        load_official_sit_module,
        sha256_file,
    )


DEFAULT_ANCHOR_CHECKPOINT = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/"
    "sit-s-2_seed0/checkpoints/step_00400000.pt"
)
DEFAULT_OTHER_CHECKPOINT = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/"
    "sit-s-2_x-velocity-loss-floor0p05_seed0/checkpoints/step_00400000.pt"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/"
    "fid5k_static_pair_step400000_seed0/velocity_to_jit_x/static_s0"
)


def _load_field_model(
    *,
    checkpoint_path: Path,
    requested_field: str,
    weights: str,
    sit_module,
    source_metadata: dict,
    device: torch.device,
) -> tuple[torch.nn.Module, FieldSemantics, dict[str, object], dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    protocol = str(checkpoint.get("protocol"))
    config = checkpoint["config"]
    if checkpoint.get("official_sit") != source_metadata:
        raise ValueError(f"{checkpoint_path} uses a different official SiT revision")
    semantics = resolve_field_semantics(
        protocol=protocol,
        config=config,
        requested_path=requested_field,
    )
    model_name = str(config["model_name"])
    if protocol == DUAL_OUTPUT_PROTOCOL:
        model = create_dual_output_sit(
            sit_module,
            model_name=model_name,
            cfg_dropout=float(config["cfg_dropout"]),
        )
    else:
        model = sit_module.SiT_models[model_name](
            input_size=LATENT_SHAPE[-1],
            num_classes=NUM_CLASSES,
            class_dropout_prob=float(config["cfg_dropout"]),
        )
    state_key = "ema" if weights == "ema" else "model"
    model.load_state_dict(checkpoint[state_key], strict=True)
    model.to(device).eval().requires_grad_(False)
    metadata: dict[str, object] = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "protocol": protocol,
        "model_name": model_name,
        "field_path": semantics.field_path,
        "prediction_target": semantics.prediction_target,
        "denominator_floor": semantics.denominator_floor,
        "training_world_size": int(config.get("world_size", 1)),
    }
    return model, semantics, metadata, checkpoint


def validate_pair_compatibility(
    anchor_checkpoint: dict,
    other_checkpoint: dict,
    anchor_metadata: dict[str, object],
    other_metadata: dict[str, object],
) -> None:
    mismatches: list[str] = []
    for key in ("checkpoint_step", "model_name"):
        if anchor_metadata[key] != other_metadata[key]:
            mismatches.append(
                f"{key}: anchor={anchor_metadata[key]!r}, other={other_metadata[key]!r}"
            )
    for key in ("data_manifest_sha256", "official_sit"):
        if anchor_checkpoint.get(key) != other_checkpoint.get(key):
            mismatches.append(f"checkpoint {key} differs")
    for key in ("global_batch_size", "seed"):
        anchor_value = anchor_checkpoint["config"].get(key)
        other_value = other_checkpoint["config"].get(key)
        if anchor_value != other_value:
            mismatches.append(
                f"{key}: anchor={anchor_value!r}, other={other_value!r}"
            )
    if mismatches:
        raise ValueError("incompatible static-pair checkpoints:\n  " + "\n  ".join(mismatches))


def conditional_static_pair_velocity(
    anchor_model: torch.nn.Module,
    other_model: torch.nn.Module,
    labels: torch.Tensor,
    *,
    anchor_semantics: FieldSemantics,
    other_semantics: FieldSemantics,
    scale: float,
    autocast_dtype: torch.dtype | None,
) -> tuple[object, dict[str, int]]:
    counter = {"nfe": 0, "anchor_forwards": 0, "other_forwards": 0}

    def evaluate(
        model: torch.nn.Module,
        semantics: FieldSemantics,
        state: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        if autocast_dtype is None:
            output = model(state, times, labels)
        else:
            with torch.autocast("cuda", dtype=autocast_dtype):
                output = model(state, times, labels)
        return output_to_field_velocity(
            output,
            state=state,
            time_value=times,
            semantics=semantics,
        )

    def velocity(time_value: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        counter["nfe"] += 1
        times = time_value.expand(len(state))
        if scale == 0.0:
            counter["anchor_forwards"] += 1
            return evaluate(anchor_model, anchor_semantics, state, times)
        if scale == 1.0:
            counter["other_forwards"] += 1
            return evaluate(other_model, other_semantics, state, times)
        counter["anchor_forwards"] += 1
        anchor_velocity = evaluate(anchor_model, anchor_semantics, state, times)
        counter["other_forwards"] += 1
        other_velocity = evaluate(other_model, other_semantics, state, times)
        return static_pair_velocity(
            anchor_velocity,
            other_velocity,
            scale=scale,
        )

    return velocity, counter


@torch.inference_mode()
def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.anchor_checkpoint.resolve() == args.other_checkpoint.resolve():
        raise ValueError("use the dual-output sampler for two paths from one checkpoint")
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

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    anchor_path = args.anchor_checkpoint.expanduser().resolve()
    other_path = args.other_checkpoint.expanduser().resolve()
    sit_module, source_metadata = load_official_sit_module(
        args.official_sit_repo.expanduser().resolve(),
        verify_source=args.verify_sit_source,
    )
    anchor_model, anchor_semantics, anchor_metadata, anchor_checkpoint = (
        _load_field_model(
            checkpoint_path=anchor_path,
            requested_field=args.anchor_field,
            weights=args.weights,
            sit_module=sit_module,
            source_metadata=source_metadata,
            device=device,
        )
    )
    other_model, other_semantics, other_metadata, other_checkpoint = _load_field_model(
        checkpoint_path=other_path,
        requested_field=args.other_field,
        weights=args.weights,
        sit_module=sit_module,
        source_metadata=source_metadata,
        device=device,
    )
    validate_pair_compatibility(
        anchor_checkpoint,
        other_checkpoint,
        anchor_metadata,
        other_metadata,
    )
    del anchor_checkpoint, other_checkpoint

    from diffusers.models import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse", local_files_only=True
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
    totals = {"nfe": 0, "anchor_forwards": 0, "other_forwards": 0}
    noise_digest = hashlib.sha256()
    label_digest = hashlib.sha256()
    cursor = 0
    started = time.perf_counter()
    preview: torch.Tensor | None = None
    global_batch_size = args.per_rank_batch_size * world_size

    for iteration in range(iterations):
        batch_size = args.per_rank_batch_size
        noise = torch.randn(batch_size, *LATENT_SHAPE, device=device)
        labels = torch.randint(0, NUM_CLASSES, (batch_size,), device=device)
        velocity, counter = conditional_static_pair_velocity(
            anchor_model,
            other_model,
            labels,
            anchor_semantics=anchor_semantics,
            other_semantics=other_semantics,
            scale=args.static_scale,
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
                f"non-finite static-pair latent on rank {rank}, iteration {iteration}"
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
                        "scale": args.static_scale,
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
            "static_scale": float(args.static_scale),
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
            json.loads(
                (output_dir / f"rank_{source_rank:02d}.json").read_text(
                    encoding="utf-8"
                )
            )
            for source_rank in range(world_size)
        ]
        rank_resource_usage = [
            load_rank_resource_usage(output_dir / f"rank_{source_rank:02d}.json")
            for source_rank in range(world_size)
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
        manifest = {
            "format": "eqvae_imagenet100_sit_static_pair_samples_v1",
            "scope": "paired FID screening on ImageNet-100",
            "anchor": anchor_metadata,
            "other": other_metadata,
            "weights": args.weights,
            "static_scale": float(args.static_scale),
            "formula": "anchor + scale * (other - anchor)",
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
                official_rank_seed(args.global_seed, world_size, source_rank)
                for source_rank in range(world_size)
            ],
            "label_sampling": "torch.randint after noise, matching official SiT",
            "label_histogram": histogram.tolist(),
            "cfg_scale": 1.0,
            "guidance": False,
            "same_noise_and_labels_across_scales": True,
            "rank_noise_sha256": [
                payload["noise_sha256"] for payload in rank_payloads
            ],
            "rank_label_sha256": [
                payload["label_sha256"] for payload in rank_payloads
            ],
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
                    "scale": args.static_scale,
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
    parser.add_argument("--anchor-checkpoint", type=Path, default=DEFAULT_ANCHOR_CHECKPOINT)
    parser.add_argument("--anchor-field", choices=("auto", "x", "epsilon", "dynamic"), default="auto")
    parser.add_argument("--other-checkpoint", type=Path, default=DEFAULT_OTHER_CHECKPOINT)
    parser.add_argument("--other-field", choices=("auto", "x", "epsilon", "dynamic"), default="auto")
    parser.add_argument("--static-scale", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--official-sit-repo", type=Path, default=DEFAULT_OFFICIAL_SIT_REPO)
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--num-samples", type=int, default=5_000)
    parser.add_argument("--per-rank-batch-size", type=int, default=8)
    parser.add_argument("--vae-decode-batch-size", type=int, default=2)
    parser.add_argument("--cuda-allocator-limit-gib", type=float, default=4.0)
    parser.add_argument("--inter-batch-sleep", type=float, default=0.0)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-output-points", type=int, default=250)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--log-every", type=int, default=512)
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verify-sit-source", action=argparse.BooleanOptionalAction, default=True)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
