#!/usr/bin/env python3
"""Sample SiT with guidance frozen on the paired unguided trajectory.

The coupled ODE is

    base'   = v400(base, t)
    frozen' = v400(frozen, t) + gamma * [v400(base, t) - weak(base, t)].

The strong anchor remains state-aware on the frozen branch.  Only the guidance
increment is evaluated on the baseline trajectory.  This distinguishes open-
loop guidance injection from fully closed-loop weak/strong feedback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import save_image

try:
    from experiments.imagenet100_sit_static_pair import output_to_field_velocity
    from experiments.sample_imagenet100_sit_fid import (
        configure_cuda_allocator,
        decode_latents_in_chunks,
        official_pixel_quantization,
        official_rank_seed,
        official_total_samples,
    )
    from experiments.sample_imagenet100_sit_flow import integrate_velocity
    from experiments.sample_imagenet100_sit_static_pair_fid import (
        DEFAULT_ANCHOR_CHECKPOINT,
        DEFAULT_OTHER_CHECKPOINT,
        _load_field_model,
        validate_pair_compatibility,
    )
    from experiments.train_imagenet100_sit_flow import (
        DEFAULT_OFFICIAL_SIT_REPO,
        LATENT_SHAPE,
        NUM_CLASSES,
        SD_VAE_SCALING_FACTOR,
        atomic_json_dump,
        load_official_sit_module,
    )
except ModuleNotFoundError:
    from imagenet100_sit_static_pair import output_to_field_velocity
    from sample_imagenet100_sit_fid import (
        configure_cuda_allocator,
        decode_latents_in_chunks,
        official_pixel_quantization,
        official_rank_seed,
        official_total_samples,
    )
    from sample_imagenet100_sit_flow import integrate_velocity
    from sample_imagenet100_sit_static_pair_fid import (
        DEFAULT_ANCHOR_CHECKPOINT,
        DEFAULT_OTHER_CHECKPOINT,
        _load_field_model,
        validate_pair_compatibility,
    )
    from train_imagenet100_sit_flow import (
        DEFAULT_OFFICIAL_SIT_REPO,
        LATENT_SHAPE,
        NUM_CLASSES,
        SD_VAE_SCALING_FACTOR,
        atomic_json_dump,
        load_official_sit_module,
    )


DEFAULT_OUTPUT_DIR = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/"
    "finite_guidance_400k_mechanism/frozen_fid5k/x400_gamma1"
)


def frozen_derivative(
    anchor_baseline: torch.Tensor,
    anchor_frozen: torch.Tensor,
    other_baseline: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    if not (
        anchor_baseline.shape == anchor_frozen.shape == other_baseline.shape
    ):
        raise ValueError("all frozen-guidance fields must have identical shapes")
    return anchor_frozen + float(gamma) * (anchor_baseline - other_baseline)


def conditional_frozen_guidance_velocity(
    anchor_model: torch.nn.Module,
    other_model: torch.nn.Module,
    labels: torch.Tensor,
    *,
    anchor_semantics,
    other_semantics,
    gamma: float,
    autocast_dtype: torch.dtype | None,
):
    counter = {"nfe": 0, "anchor_forwards": 0, "other_forwards": 0}
    base_batch = len(labels)

    def evaluate(model, semantics, state, times, model_labels):
        if autocast_dtype is None:
            output = model(state, times, model_labels)
        else:
            with torch.autocast("cuda", dtype=autocast_dtype):
                output = model(state, times, model_labels)
        return output_to_field_velocity(
            output,
            state=state,
            time_value=times,
            semantics=semantics,
        )

    def velocity(time_value: torch.Tensor, combined_state: torch.Tensor) -> torch.Tensor:
        counter["nfe"] += 1
        if len(combined_state) != 2 * base_batch:
            raise ValueError("coupled state must contain baseline then frozen batches")
        baseline, frozen = combined_state.split(base_batch)
        times_pair = time_value.expand(2 * base_batch)
        counter["anchor_forwards"] += 1
        anchor_pair = evaluate(
            anchor_model,
            anchor_semantics,
            combined_state,
            times_pair,
            labels.repeat(2),
        )
        anchor_baseline, anchor_frozen = anchor_pair.split(base_batch)
        counter["other_forwards"] += 1
        other_baseline = evaluate(
            other_model,
            other_semantics,
            baseline,
            time_value.expand(base_batch),
            labels,
        )
        return torch.cat(
            (
                anchor_baseline,
                frozen_derivative(
                    anchor_baseline,
                    anchor_frozen,
                    other_baseline,
                    gamma,
                ),
            )
        )

    return velocity, counter


@torch.inference_mode()
def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.num_samples <= 0 or args.batch_size <= 0:
        raise ValueError("num-samples and batch-size must be positive")
    if not np.isfinite(args.gamma):
        raise ValueError("gamma must be finite")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    allocator = configure_cuda_allocator(device, limit_gib=args.cuda_allocator_limit_gib)
    seed = official_rank_seed(args.global_seed, 1, 0)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
    torch.set_float32_matmul_precision("high" if args.allow_tf32 else "highest")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sit_module, source_metadata = load_official_sit_module(
        args.official_sit_repo.expanduser().resolve(),
        verify_source=args.verify_sit_source,
    )
    anchor_model, anchor_semantics, anchor_metadata, anchor_checkpoint = _load_field_model(
        checkpoint_path=args.anchor_checkpoint.expanduser().resolve(),
        requested_field="auto",
        weights=args.weights,
        sit_module=sit_module,
        source_metadata=source_metadata,
        device=device,
    )
    other_model, other_semantics, other_metadata, other_checkpoint = _load_field_model(
        checkpoint_path=args.other_checkpoint.expanduser().resolve(),
        requested_field="auto",
        weights=args.weights,
        sit_module=sit_module,
        source_metadata=source_metadata,
        device=device,
    )
    assert anchor_model is not None and other_model is not None
    validate_pair_compatibility(
        anchor_checkpoint,
        other_checkpoint,
        anchor_metadata,
        other_metadata,
        allow_step_mismatch=args.allow_step_mismatch,
    )
    if anchor_semantics.prediction_target != "velocity":
        raise ValueError("anchor must be a native velocity checkpoint")
    del anchor_checkpoint, other_checkpoint

    from diffusers.models import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse", local_files_only=True
    )
    vae.to(device).eval().requires_grad_(False)
    total_samples = official_total_samples(args.num_samples, args.batch_size, 1)
    images = np.empty((total_samples, 256, 256, 3), dtype=np.uint8)
    labels_array = np.empty(total_samples, dtype=np.int16)
    noise_digest = hashlib.sha256()
    label_digest = hashlib.sha256()
    cursor = 0
    totals = {"nfe": 0, "anchor_forwards": 0, "other_forwards": 0}
    preview = None
    started = time.perf_counter()
    autocast_dtype = None if args.precision == "fp32" else torch.bfloat16
    while cursor < total_samples:
        noise = torch.randn(args.batch_size, *LATENT_SHAPE, device=device)
        labels = torch.randint(0, NUM_CLASSES, (args.batch_size,), device=device)
        velocity, counter = conditional_frozen_guidance_velocity(
            anchor_model,
            other_model,
            labels,
            anchor_semantics=anchor_semantics,
            other_semantics=other_semantics,
            gamma=args.gamma,
            autocast_dtype=autocast_dtype,
        )
        combined = integrate_velocity(
            torch.cat((noise, noise)),
            velocity,
            num_output_points=args.num_output_points,
            atol=args.atol,
            rtol=args.rtol,
        )
        baseline_latent, frozen_latent = combined.split(args.batch_size)
        if not torch.isfinite(frozen_latent).all():
            raise FloatingPointError("non-finite frozen-guidance endpoint")
        decoded = decode_latents_in_chunks(
            vae,
            frozen_latent,
            scaling_factor=SD_VAE_SCALING_FACTOR,
            chunk_size=args.vae_decode_batch_size,
        )
        stop = cursor + args.batch_size
        images[cursor:stop] = official_pixel_quantization(decoded)
        labels_array[cursor:stop] = labels.cpu().numpy().astype(np.int16, copy=False)
        noise_digest.update(noise.cpu().contiguous().numpy().tobytes())
        label_digest.update(labels.cpu().contiguous().numpy().tobytes())
        cursor = stop
        for key in totals:
            totals[key] += int(counter[key])
        if preview is None:
            preview = decoded[: min(16, len(decoded))].cpu()
        if cursor == args.batch_size or cursor == total_samples or cursor % args.log_every == 0:
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "generated": cursor,
                        "total": total_samples,
                        "elapsed_seconds": elapsed,
                        "images_per_second": cursor / elapsed,
                        "last_batch_nfe": counter["nfe"],
                        "baseline_latent_rms": float(baseline_latent.square().mean().sqrt()),
                    }
                ),
                flush=True,
            )

    sample_path = output_dir / f"samples_frozen_n{args.num_samples}.npz"
    label_path = output_dir / f"sample_labels_frozen_n{args.num_samples}.npy"
    np.savez(sample_path, arr_0=images[: args.num_samples])
    np.save(label_path, labels_array[: args.num_samples], allow_pickle=False)
    assert preview is not None
    save_image(
        preview,
        output_dir / "preview.png",
        nrow=4,
        normalize=True,
        value_range=(-1, 1),
    )
    histogram = np.bincount(labels_array[: args.num_samples].astype(np.int64), minlength=NUM_CLASSES)
    manifest = {
        "format": "eqvae_imagenet100_sit_frozen_guidance_samples_v1",
        "scope": "paired 5k finite-guidance feedback audit",
        "formula": (
            "base'=anchor(base,t); frozen'=anchor(frozen,t)+gamma*"
            "(anchor(base,t)-other(base,t))"
        ),
        "anchor": anchor_metadata,
        "other": other_metadata,
        "weights": args.weights,
        "gamma": float(args.gamma),
        "requested_samples": args.num_samples,
        "generated_for_batch_divisibility": total_samples,
        "batch_size": args.batch_size,
        "vae_decode_batch_size": args.vae_decode_batch_size,
        "cuda_allocator_limit_gib": args.cuda_allocator_limit_gib,
        "global_seed": args.global_seed,
        "rank_seed": seed,
        "noise_sha256": noise_digest.hexdigest(),
        "label_sha256": label_digest.hexdigest(),
        "label_histogram": histogram.tolist(),
        "sampler": {
            "path": "linear",
            "method": "dopri5",
            "interval": [0.0, 1.0],
            "num_output_points": args.num_output_points,
            "atol": args.atol,
            "rtol": args.rtol,
            "precision": args.precision,
            "allow_tf32": args.allow_tf32,
        },
        "samples": str(sample_path),
        "labels": str(label_path),
        "elapsed_seconds": time.perf_counter() - started,
        **totals,
        **allocator,
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    atomic_json_dump(manifest, output_dir / "sampling_manifest.json")
    print(json.dumps({"event": "complete", **manifest}, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-checkpoint", type=Path, default=DEFAULT_ANCHOR_CHECKPOINT)
    parser.add_argument("--other-checkpoint", type=Path, default=DEFAULT_OTHER_CHECKPOINT)
    parser.add_argument("--allow-step-mismatch", action="store_true")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--official-sit-repo", type=Path, default=DEFAULT_OFFICIAL_SIT_REPO)
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--vae-decode-batch-size", type=int, default=2)
    parser.add_argument("--cuda-allocator-limit-gib", type=float, default=6.0)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-output-points", type=int, default=250)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--log-every", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verify-sit-source", action=argparse.BooleanOptionalAction, default=True)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
