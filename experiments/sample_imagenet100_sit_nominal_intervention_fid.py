#!/usr/bin/env python3
"""Sample one paired nominal-guidance intervention for ImageNet-100 SiT."""

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
    from experiments.nominal_guidance_transfer import (
        INTERVENTION_MODES,
        intervention_guidance,
    )
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
    from nominal_guidance_transfer import INTERVENTION_MODES, intervention_guidance
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
    "nominal_guidance_transfer_800k_v1/fid5k/x800_gain_only_seed0"
)


def conditional_nominal_intervention_velocity(
    anchor_model: torch.nn.Module,
    other_model: torch.nn.Module,
    labels: torch.Tensor,
    *,
    anchor_semantics,
    other_semantics,
    mode: str,
    gamma: float,
    autocast_dtype: torch.dtype | None,
    nominal_scale: float = 1.0,
    orthogonal_scale: float = 1.0,
    response_scale: float = 1.0,
):
    """Return a coupled baseline/intervention field and accounting counters."""

    if mode not in INTERVENTION_MODES:
        raise ValueError(f"unsupported intervention mode: {mode}")
    counter = {
        "nfe": 0,
        "anchor_forwards": 0,
        "other_forwards": 0,
        "anchor_examples": 0,
        "other_examples": 0,
    }
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
        ).float()

    def velocity(time_value: torch.Tensor, combined_state: torch.Tensor) -> torch.Tensor:
        counter["nfe"] += 1
        if len(combined_state) != 2 * base_batch:
            raise ValueError("coupled state must contain baseline then intervention")
        baseline, current = combined_state.split(base_batch)
        base_times = time_value.expand(base_batch)
        if mode == "replay":
            counter["anchor_forwards"] += 1
            counter["anchor_examples"] += base_batch
            anchor_baseline = evaluate(
                anchor_model,
                anchor_semantics,
                baseline,
                base_times,
                labels,
            )
            counter["other_forwards"] += 1
            counter["other_examples"] += base_batch
            other_baseline = evaluate(
                other_model,
                other_semantics,
                baseline,
                base_times,
                labels,
            )
            nominal_gap = anchor_baseline - other_baseline
            return torch.cat(
                (
                    anchor_baseline,
                    anchor_baseline + float(gamma) * nominal_gap,
                )
            )

        counter["anchor_forwards"] += 1
        counter["anchor_examples"] += 2 * base_batch
        anchor_pair = evaluate(
            anchor_model,
            anchor_semantics,
            combined_state,
            time_value.expand(2 * base_batch),
            labels.repeat(2),
        )
        anchor_baseline, anchor_current = anchor_pair.split(base_batch)
        factorized_without_online_gap = (
            mode == "factorized" and float(orthogonal_scale) == 0.0
        )
        if mode == "frozen" or factorized_without_online_gap:
            counter["other_forwards"] += 1
            counter["other_examples"] += base_batch
            other_baseline = evaluate(
                other_model,
                other_semantics,
                baseline,
                base_times,
                labels,
            )
            nominal_gap = anchor_baseline - other_baseline
            guidance = intervention_guidance(
                nominal_gap,
                nominal_gap if mode == "factorized" else None,
                mode=mode,
                nominal_scale=nominal_scale,
                orthogonal_scale=orthogonal_scale,
            )
        else:
            counter["other_forwards"] += 1
            counter["other_examples"] += 2 * base_batch
            other_pair = evaluate(
                other_model,
                other_semantics,
                combined_state,
                time_value.expand(2 * base_batch),
                labels.repeat(2),
            )
            other_baseline, other_current = other_pair.split(base_batch)
            nominal_gap = anchor_baseline - other_baseline
            current_gap = anchor_current - other_current
            guidance = intervention_guidance(
                nominal_gap,
                current_gap,
                mode=mode,
                nominal_scale=nominal_scale,
                orthogonal_scale=orthogonal_scale,
            )
        current_anchor = anchor_current
        if mode == "factorized":
            current_anchor = anchor_baseline + float(response_scale) * (
                anchor_current - anchor_baseline
            )
        return torch.cat(
            (
                anchor_baseline,
                current_anchor + float(gamma) * guidance,
            )
        )

    return velocity, counter


def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.mode not in INTERVENTION_MODES:
        raise ValueError(f"unsupported intervention mode: {args.mode}")
    if args.batch_size <= 0 or args.num_samples <= 0:
        raise ValueError("batch-size and num-samples must be positive")
    if not all(
        np.isfinite(value)
        for value in (
            args.gamma,
            args.nominal_scale,
            args.orthogonal_scale,
            args.response_scale,
        )
    ):
        raise ValueError("guidance coefficients must be finite")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    allocator = configure_cuda_allocator(
        device,
        limit_gib=args.cuda_allocator_limit_gib,
    )
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
        "stabilityai/sd-vae-ft-mse",
        local_files_only=True,
    )
    vae.to(device).eval().requires_grad_(False)
    total_samples = official_total_samples(args.num_samples, args.batch_size, 1)
    images = np.empty((total_samples, 256, 256, 3), dtype=np.uint8)
    labels_array = np.empty(total_samples, dtype=np.int16)
    noise_digest = hashlib.sha256()
    label_digest = hashlib.sha256()
    cursor = 0
    totals = {
        "nfe": 0,
        "anchor_forwards": 0,
        "other_forwards": 0,
        "anchor_examples": 0,
        "other_examples": 0,
    }
    preview = None
    started = time.perf_counter()
    autocast_dtype = None if args.precision == "fp32" else torch.bfloat16
    while cursor < total_samples:
        noise = torch.randn(args.batch_size, *LATENT_SHAPE, device=device)
        labels = torch.randint(0, NUM_CLASSES, (args.batch_size,), device=device)
        velocity, counter = conditional_nominal_intervention_velocity(
            anchor_model,
            other_model,
            labels,
            anchor_semantics=anchor_semantics,
            other_semantics=other_semantics,
            mode=args.mode,
            gamma=args.gamma,
            autocast_dtype=autocast_dtype,
            nominal_scale=args.nominal_scale,
            orthogonal_scale=args.orthogonal_scale,
            response_scale=args.response_scale,
        )
        combined = integrate_velocity(
            torch.cat((noise, noise)),
            velocity,
            num_output_points=args.num_output_points,
            atol=args.atol,
            rtol=args.rtol,
        )
        _, endpoint = combined.split(args.batch_size)
        if not torch.isfinite(endpoint).all():
            raise FloatingPointError("non-finite nominal-intervention endpoint")
        decoded = decode_latents_in_chunks(
            vae,
            endpoint,
            scaling_factor=SD_VAE_SCALING_FACTOR,
            chunk_size=args.vae_decode_batch_size,
        )
        stop = cursor + args.batch_size
        images[cursor:stop] = official_pixel_quantization(decoded)
        labels_array[cursor:stop] = labels.cpu().numpy().astype(np.int16, copy=False)
        noise_digest.update(noise.cpu().contiguous().numpy().tobytes())
        label_digest.update(labels.cpu().contiguous().numpy().tobytes())
        for key in totals:
            totals[key] += int(counter[key])
        if preview is None:
            preview = decoded.detach().cpu()
        cursor = stop
        print(f"[{cursor}/{total_samples}] elapsed={time.perf_counter()-started:.1f}s", flush=True)

    sample_path = output_dir / f"samples_{args.mode}_n{args.num_samples}.npz"
    label_path = output_dir / f"sample_labels_{args.mode}_n{args.num_samples}.npy"
    np.savez(sample_path, images)
    np.save(label_path, labels_array)
    if preview is not None:
        save_image(
            (preview[: min(len(preview), 16)] + 1.0) / 2.0,
            output_dir / "preview.png",
            nrow=4,
        )
    manifest = {
        "format": "eqvae_imagenet100_sit_nominal_intervention_samples_v1",
        "mode": args.mode,
        "formula": {
            "baseline": "z'=S(z,t)",
            "frozen": "z'=S(z,t)+gamma*g(z_baseline,t)",
            "replay": "z'=S(z_baseline,t)+gamma*g(z_baseline,t)",
            "gain_only": "z'=S(z,t)+gamma*Proj_gbase(g(z,t))",
            "direction_only": (
                "z'=S(z,t)+gamma*[g(z_baseline,t)+"
                "Orth_gbase(g(z,t)-g(z_baseline,t))]"
            ),
            "factorized": (
                "z'=S(z_baseline,t)+response_scale*[S(z,t)-S(z_baseline,t)]+"
                "gamma*[nominal_scale*g(z_baseline,t)+"
                "orthogonal_scale*Orth_gbase(g(z,t))]"
            ),
            "closed": "z'=S(z,t)+gamma*g(z,t)",
            "gap": "g=S-W",
        },
        "anchor": anchor_metadata,
        "other": other_metadata,
        "weights": args.weights,
        "gamma": float(args.gamma),
        "nominal_scale": float(args.nominal_scale),
        "orthogonal_scale": float(args.orthogonal_scale),
        "response_scale": float(args.response_scale),
        "requested_samples": int(args.num_samples),
        "generated_for_batch_divisibility": int(total_samples),
        "batch_size": int(args.batch_size),
        "vae_decode_batch_size": int(args.vae_decode_batch_size),
        "global_seed": int(args.global_seed),
        "rank_seed": int(seed),
        "noise_sha256": noise_digest.hexdigest(),
        "label_sha256": label_digest.hexdigest(),
        "label_histogram": np.bincount(labels_array, minlength=NUM_CLASSES).tolist(),
        "sampler": {
            "method": "dopri5",
            "num_output_points": int(args.num_output_points),
            "atol": float(args.atol),
            "rtol": float(args.rtol),
            "precision": args.precision,
            "allow_tf32": bool(args.allow_tf32),
        },
        **totals,
        "elapsed_seconds": time.perf_counter() - started,
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        **allocator,
        "samples": str(sample_path),
        "labels": str(label_path),
    }
    atomic_json_dump(manifest, output_dir / "sampling_manifest.json")
    print(json.dumps(manifest, indent=2, default=str), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-checkpoint", type=Path, default=DEFAULT_ANCHOR_CHECKPOINT)
    parser.add_argument("--other-checkpoint", type=Path, default=DEFAULT_OTHER_CHECKPOINT)
    parser.add_argument("--allow-step-mismatch", action="store_true")
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--mode", choices=INTERVENTION_MODES, required=True)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--nominal-scale", type=float, default=1.0)
    parser.add_argument("--orthogonal-scale", type=float, default=1.0)
    parser.add_argument("--response-scale", type=float, default=1.0)
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--vae-decode-batch-size", type=int, default=2)
    parser.add_argument("--num-output-points", type=int, default=250)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cuda-allocator-limit-gib", type=float, default=8.0)
    parser.add_argument("--official-sit-repo", type=Path, default=DEFAULT_OFFICIAL_SIT_REPO)
    parser.add_argument(
        "--verify-sit-source",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
