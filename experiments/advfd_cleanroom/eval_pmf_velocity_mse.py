"""Paired continuation-unseen local-velocity MSE for pMF checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset


EQVAE_ROOT = Path(__file__).resolve().parents[2]


def parse_condition(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("condition must be NAME=CHECKPOINT")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("condition must be NAME=CHECKPOINT")
    return name, Path(path).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--packed-data", type=Path, required=True)
    parser.add_argument("--condition", action="append", type=parse_condition, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=260823)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--continuation-samples-seen",
        type=int,
        default=0,
        help=(
            "Global samples consumed from the first shuffled continuation epoch. "
            "When positive, the evaluation bank is selected from the unseen suffix "
            "of that exact DistributedSampler permutation."
        ),
    )
    parser.add_argument("--continuation-sampler-seed", type=int, default=0)
    parser.add_argument("--continuation-world-size", type=int, default=1)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    return checkpoint["model"] if "model" in checkpoint else checkpoint


def continuation_unseen_indices(
    *,
    dataset_size: int,
    samples: int,
    samples_seen: int,
    sampler_seed: int,
    world_size: int,
) -> list[int]:
    """Reconstruct the first DDP epoch and return indices not consumed by training."""

    if dataset_size <= 0 or samples <= 0 or world_size <= 0:
        raise ValueError("dataset size, samples, and world size must be positive")
    total_size = (dataset_size // world_size) * world_size
    if not 0 <= samples_seen <= total_size:
        raise ValueError(
            f"samples_seen={samples_seen} is outside the first DDP epoch [0,{total_size}]"
        )
    if samples_seen % world_size:
        raise ValueError("samples_seen must be divisible by world_size")
    generator = torch.Generator().manual_seed(int(sampler_seed))
    permutation = torch.randperm(dataset_size, generator=generator).tolist()[:total_size]
    unseen = permutation[samples_seen:]
    if samples > len(unseen):
        raise ValueError(
            f"requested {samples} unseen samples, but only {len(unseen)} remain"
        )
    return unseen[:samples]


def main() -> None:
    args = parse_args()
    if args.samples <= 0 or args.batch_size <= 0:
        raise ValueError("samples and batch size must be positive")

    official_root = args.official_root.expanduser().resolve()
    sys.path.insert(0, str(EQVAE_ROOT))
    sys.path.insert(0, str(official_root))
    from experiments.advfd_cleanroom.pmf_velocity_control import (  # noqa: PLC0415
        local_pmf_conditioning,
        pmf_velocity_target,
    )
    from experiments.raev2_training_core import (  # noqa: PLC0415
        DeterministicImageNetPacked,
    )
    from models.denoiser_pmf import (  # noqa: PLC0415
        convert_pmf_checkpoint,
        pMFDenoiser_models,
    )

    device = torch.device(args.device)
    dataset = DeterministicImageNetPacked(
        args.packed_data,
        split="train",
        image_size=256,
        augmentation_seed=args.seed,
        horizontal_flip=False,
    )
    if args.samples > len(dataset):
        raise ValueError("requested more held-out samples than the dataset contains")
    if args.continuation_samples_seen > 0:
        indices = continuation_unseen_indices(
            dataset_size=len(dataset),
            samples=args.samples,
            samples_seen=args.continuation_samples_seen,
            sampler_seed=args.continuation_sampler_seed,
            world_size=args.continuation_world_size,
        )
        bank_mode = "continuation_unseen_first_epoch_suffix"
    else:
        indices = list(range(len(dataset) - args.samples, len(dataset)))
        bank_mode = "fixed_train_tail_not_guaranteed_unseen"
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = pMFDenoiser_models["pMF_B"](
        img_size=256,
        patch_size=16,
        in_channels=3,
        tokenizer_patch_size=1,
        num_classes=1000,
        label_drop_prob=0.1,
        P_mean=0.8,
        P_std=0.8,
        ratio_r_neq_t=0.5,
        cfg_beta=1.0,
        cfg_omega_max=7.0,
        aux_head_depth=8,
        class_tokens=8,
        time_tokens=4,
        guidance_tokens=4,
        interval_tokens=2,
        rope_2d=True,
        learned_pe=True,
        disable_v_head=True,
        grad_checkpointing=False,
        t_eps=0.05,
        noise_scale=1.0,
    ).to(device).eval()
    bin_edges = (0.0, 0.2, 0.4, 0.6, 0.8, 1.000001)
    results = []

    for condition, checkpoint_path in args.condition:
        state = convert_pmf_checkpoint(load_model_state(checkpoint_path))
        incompatible = model.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                f"{condition} load mismatch: missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        generator = torch.Generator(device=device).manual_seed(args.seed)
        squared_error_sum = 0.0
        value_count = 0
        bin_sums = [0.0] * (len(bin_edges) - 1)
        bin_counts = [0] * (len(bin_edges) - 1)

        with torch.inference_mode():
            for images, labels, _ in loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                batch_size = images.shape[0]
                t = torch.sigmoid(
                    torch.randn(
                        (batch_size, 1, 1, 1),
                        device=device,
                        generator=generator,
                    ) * 0.8 + 0.8
                )
                noise = torch.randn(
                    images.shape,
                    device=device,
                    dtype=images.dtype,
                    generator=generator,
                )
                x0 = images.mul(2.0).sub(1.0)
                z_t, target = pmf_velocity_target(
                    x0,
                    noise,
                    t,
                    t_eps=float(model.t_eps),
                )
                cond = local_pmf_conditioning(t)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    prediction = model.u_fn(
                        z_t,
                        t,
                        cond["h"],
                        cond["omega"],
                        cond["t_min"],
                        cond["t_max"],
                        labels,
                    )[0]
                per_sample = (prediction.float() - target.float()).square().flatten(1).mean(1)
                squared_error_sum += float(per_sample.sum())
                value_count += batch_size
                flat_t = t.flatten()
                for index, (lower, upper) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
                    mask = (flat_t >= lower) & (flat_t < upper)
                    if mask.any():
                        bin_sums[index] += float(per_sample[mask].sum())
                        bin_counts[index] += int(mask.sum())
                if not torch.isfinite(per_sample).all():
                    raise RuntimeError(f"non-finite velocity MSE for {condition}")

        row = {
            "condition": condition,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "samples": value_count,
            "velocity_mse": squared_error_sum / value_count,
            "velocity_rmse": (squared_error_sum / value_count) ** 0.5,
            "time_bins": [
                {
                    "lower": lower,
                    "upper": min(upper, 1.0),
                    "samples": count,
                    "velocity_mse": total / count if count else None,
                }
                for lower, upper, total, count in zip(
                    bin_edges[:-1], bin_edges[1:], bin_sums, bin_counts
                )
            ],
        }
        results.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    payload = {
        "protocol": "paired pMF local velocity MSE v2",
        "seed": args.seed,
        "bank_mode": bank_mode,
        "sample_indices_sha256": hashlib.sha256(
            torch.tensor(indices, dtype=torch.int64).numpy().tobytes()
        ).hexdigest(),
        "sample_indices_preview": list(indices[:8]),
        "continuation_samples_seen": args.continuation_samples_seen,
        "continuation_sampler_seed": args.continuation_sampler_seed,
        "continuation_world_size": args.continuation_world_size,
        "label_dropout": False,
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("condition", "checkpoint", "samples", "velocity_mse", "velocity_rmse"),
        )
        writer.writeheader()
        for row in results:
            writer.writerow({key: row[key] for key in writer.fieldnames})


if __name__ == "__main__":
    main()
