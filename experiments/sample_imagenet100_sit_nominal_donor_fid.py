#!/usr/bin/env python3
"""Sample frozen guidance using controlled nominal-trajectory donor pairs."""

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
    from experiments.nominal_guidance_transfer import DONOR_MODES, donor_inputs
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
    from nominal_guidance_transfer import DONOR_MODES, donor_inputs
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
    "nominal_guidance_transfer_800k_v1/donor/x800_paired_seed0"
)


def conditional_donor_guidance_velocity(
    anchor_model: torch.nn.Module,
    other_model: torch.nn.Module,
    target_labels: torch.Tensor,
    donor_labels: torch.Tensor,
    *,
    anchor_semantics,
    other_semantics,
    gamma: float,
    autocast_dtype: torch.dtype | None,
):
    """Couple target baseline, donor baseline, and donor-guided target state."""

    if target_labels.shape != donor_labels.shape:
        raise ValueError("target and donor labels must have identical shapes")
    counter = {
        "nfe": 0,
        "anchor_forwards": 0,
        "other_forwards": 0,
        "anchor_examples": 0,
        "other_examples": 0,
    }
    batch_size = len(target_labels)

    def evaluate(model, semantics, state, times, labels):
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
        ).float()

    def velocity(time_value: torch.Tensor, combined: torch.Tensor) -> torch.Tensor:
        counter["nfe"] += 1
        if len(combined) != 3 * batch_size:
            raise ValueError("state must contain target baseline, donor, and guided target")
        target_baseline, donor, current = combined.split(batch_size)
        counter["anchor_forwards"] += 1
        counter["anchor_examples"] += 3 * batch_size
        anchor_values = evaluate(
            anchor_model,
            anchor_semantics,
            combined,
            time_value.expand(3 * batch_size),
            torch.cat((target_labels, donor_labels, target_labels)),
        )
        anchor_target, anchor_donor, anchor_current = anchor_values.split(batch_size)
        counter["other_forwards"] += 1
        counter["other_examples"] += batch_size
        other_donor = evaluate(
            other_model,
            other_semantics,
            donor,
            time_value.expand(batch_size),
            donor_labels,
        )
        donor_gap = anchor_donor - other_donor
        return torch.cat(
            (
                anchor_target,
                anchor_donor,
                anchor_current + float(gamma) * donor_gap,
            )
        )

    return velocity, counter


def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.donor_mode not in DONOR_MODES:
        raise ValueError(f"unsupported donor mode: {args.donor_mode}")
    if args.batch_size < 2 or args.num_samples <= 0:
        raise ValueError("batch-size must be at least two and num-samples positive")
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
    donor_labels_array = np.empty(total_samples, dtype=np.int16)
    target_noise_digest = hashlib.sha256()
    donor_noise_digest = hashlib.sha256()
    target_label_digest = hashlib.sha256()
    donor_label_digest = hashlib.sha256()
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
        target_noise = torch.randn(args.batch_size, *LATENT_SHAPE, device=device)
        target_labels = torch.randint(0, NUM_CLASSES, (args.batch_size,), device=device)
        donor_noise, donor_labels = donor_inputs(
            target_noise,
            target_labels,
            mode=args.donor_mode,
            num_classes=NUM_CLASSES,
            class_shift=args.class_shift,
        )
        velocity, counter = conditional_donor_guidance_velocity(
            anchor_model,
            other_model,
            target_labels,
            donor_labels,
            anchor_semantics=anchor_semantics,
            other_semantics=other_semantics,
            gamma=args.gamma,
            autocast_dtype=autocast_dtype,
        )
        initial = torch.cat((target_noise, donor_noise, target_noise))
        endpoint = integrate_velocity(
            initial,
            velocity,
            num_output_points=args.num_output_points,
            atol=args.atol,
            rtol=args.rtol,
        )[-args.batch_size :]
        if not torch.isfinite(endpoint).all():
            raise FloatingPointError("non-finite donor-guidance endpoint")
        decoded = decode_latents_in_chunks(
            vae,
            endpoint,
            scaling_factor=SD_VAE_SCALING_FACTOR,
            chunk_size=args.vae_decode_batch_size,
        )
        stop = cursor + args.batch_size
        images[cursor:stop] = official_pixel_quantization(decoded)
        labels_array[cursor:stop] = target_labels.cpu().numpy().astype(np.int16, copy=False)
        donor_labels_array[cursor:stop] = donor_labels.cpu().numpy().astype(
            np.int16,
            copy=False,
        )
        target_noise_digest.update(target_noise.cpu().contiguous().numpy().tobytes())
        donor_noise_digest.update(donor_noise.cpu().contiguous().numpy().tobytes())
        target_label_digest.update(target_labels.cpu().contiguous().numpy().tobytes())
        donor_label_digest.update(donor_labels.cpu().contiguous().numpy().tobytes())
        for key in totals:
            totals[key] += int(counter[key])
        if preview is None:
            preview = decoded.detach().cpu()
        cursor = stop
        print(f"[{cursor}/{total_samples}] elapsed={time.perf_counter()-started:.1f}s", flush=True)

    sample_path = output_dir / f"samples_{args.donor_mode}_n{args.num_samples}.npz"
    labels_path = output_dir / f"sample_labels_{args.donor_mode}_n{args.num_samples}.npy"
    donor_labels_path = output_dir / f"donor_labels_{args.donor_mode}_n{args.num_samples}.npy"
    np.savez(sample_path, images)
    np.save(labels_path, labels_array)
    np.save(donor_labels_path, donor_labels_array)
    if preview is not None:
        save_image(
            (preview[: min(len(preview), 16)] + 1.0) / 2.0,
            output_dir / "preview.png",
            nrow=4,
        )
    manifest = {
        "format": "eqvae_imagenet100_sit_nominal_donor_samples_v1",
        "formula": (
            "target'=S(target,current_class)+gamma*["
            "S(donor,donor_class)-W(donor,donor_class)]"
        ),
        "donor_mode": args.donor_mode,
        "class_shift": int(args.class_shift),
        "anchor": anchor_metadata,
        "other": other_metadata,
        "weights": args.weights,
        "gamma": float(args.gamma),
        "requested_samples": int(args.num_samples),
        "generated_for_batch_divisibility": int(total_samples),
        "batch_size": int(args.batch_size),
        "vae_decode_batch_size": int(args.vae_decode_batch_size),
        "global_seed": int(args.global_seed),
        "rank_seed": int(seed),
        "target_noise_sha256": target_noise_digest.hexdigest(),
        "donor_noise_sha256": donor_noise_digest.hexdigest(),
        "target_label_sha256": target_label_digest.hexdigest(),
        "donor_label_sha256": donor_label_digest.hexdigest(),
        "target_label_histogram": np.bincount(labels_array, minlength=NUM_CLASSES).tolist(),
        "donor_label_histogram": np.bincount(
            donor_labels_array,
            minlength=NUM_CLASSES,
        ).tolist(),
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
        "labels": str(labels_path),
        "donor_labels": str(donor_labels_path),
    }
    atomic_json_dump(manifest, output_dir / "sampling_manifest.json")
    print(json.dumps(manifest, indent=2, default=str), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-checkpoint", type=Path, default=DEFAULT_ANCHOR_CHECKPOINT)
    parser.add_argument("--other-checkpoint", type=Path, default=DEFAULT_OTHER_CHECKPOINT)
    parser.add_argument("--allow-step-mismatch", action="store_true")
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--donor-mode", choices=DONOR_MODES, required=True)
    parser.add_argument("--class-shift", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--num-samples", type=int, default=1000)
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
