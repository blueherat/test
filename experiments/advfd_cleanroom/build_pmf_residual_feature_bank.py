#!/usr/bin/env python3
"""Build paired-protocol real/fake Inception banks for frozen pMF audits."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from experiments.advfd_cleanroom.feature_extractors import (
    DifferentiableInception2048,
    generator_output_to_unit_interval,
)
from experiments.advfd_cleanroom.generators import load_pmf_b16, pmf_one_step
from experiments.advfd_cleanroom.run_pmf_pilot import autocast_context
from experiments.raev2_training_core import DeterministicImageNetPacked


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def whitening_from_warmstart(
    warmstart: dict,
    *,
    ridge_fraction: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    projection = warmstart["projection"].to(device=device, dtype=torch.float32)
    mean = warmstart["real_projected"]["mean"].to(
        device=device, dtype=torch.float64
    )
    covariance = warmstart["real_projected"]["covariance"].to(
        device=device, dtype=torch.float64
    )
    covariance = 0.5 * (covariance + covariance.mT)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    average_variance = eigenvalues.clamp_min(0).mean()
    ridge = float(ridge_fraction) * average_variance
    stabilized = eigenvalues.clamp_min(0) + ridge
    whitening = (
        eigenvectors
        @ torch.diag(stabilized.rsqrt())
        @ eigenvectors.mT
    ).to(torch.float32)
    diagnostics = {
        "minimum_eigenvalue": float(eigenvalues.min()),
        "median_eigenvalue": float(eigenvalues.median()),
        "maximum_eigenvalue": float(eigenvalues.max()),
        "average_variance": float(average_variance),
        "ridge": float(ridge),
        "condition_after_ridge": float(
            stabilized.max() / stabilized.min().clamp_min(torch.finfo(stabilized.dtype).tiny)
        ),
    }
    return projection, mean.float(), whitening, diagnostics


def transform_features(
    full_features: torch.Tensor,
    projection: torch.Tensor,
    mean: torch.Tensor,
    whitening: torch.Tensor,
) -> torch.Tensor:
    return ((full_features.float() @ projection) - mean) @ whitening


def summarize_bank(features: torch.Tensor) -> dict[str, float]:
    values = features.double()
    mean = values.mean(dim=0)
    centered = values - mean
    covariance = centered.mT @ centered / max(len(values) - 1, 1)
    eigenvalues = torch.linalg.eigvalsh(0.5 * (covariance + covariance.mT))
    return {
        "count": int(len(values)),
        "dimension": int(values.shape[1]),
        "mean_rms": float(mean.square().mean().sqrt()),
        "coordinate_std_mean": float(values.std(dim=0).mean()),
        "covariance_trace_per_dim": float(eigenvalues.sum() / values.shape[1]),
        "covariance_minimum_eigenvalue": float(eigenvalues.min()),
        "covariance_maximum_eigenvalue": float(eigenvalues.max()),
    }


def collect_real(
    *,
    encoder: DifferentiableInception2048,
    projection: torch.Tensor,
    mean: torch.Tensor,
    whitening: torch.Tensor,
    packed_data: Path,
    indices: range,
    batch_size: int,
    num_workers: int,
    augmentation_seed: int,
    horizontal_flip: bool,
    amp: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dataset = DeterministicImageNetPacked(
        packed_data,
        split="train",
        image_size=256,
        augmentation_seed=augmentation_seed,
        horizontal_flip=horizontal_flip,
    )
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    feature_parts = []
    label_parts = []
    index_parts = []
    for images, labels, source_indices in loader:
        images = images.to(device=device, non_blocking=True)
        with torch.inference_mode(), autocast_context(device, amp):
            full = encoder(images)
        transformed = transform_features(full, projection, mean, whitening)
        feature_parts.append(transformed.cpu())
        label_parts.append(labels.long().cpu())
        index_parts.append(source_indices.long().cpu())
    return (
        torch.cat(feature_parts),
        torch.cat(label_parts),
        torch.cat(index_parts),
    )


def collect_fake(
    *,
    model: torch.nn.Module,
    encoder: DifferentiableInception2048,
    projection: torch.Tensor,
    mean: torch.Tensor,
    whitening: torch.Tensor,
    count: int,
    batch_size: int,
    noise_seed: int,
    label_seed: int,
    amp: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    noise_generator = torch.Generator(device=device).manual_seed(noise_seed)
    label_generator = torch.Generator(device=device).manual_seed(label_seed)
    feature_parts = []
    label_parts = []
    noise_seed_parts = []
    completed = 0
    while completed < count:
        batch = min(batch_size, count - completed)
        noise = torch.randn(
            batch,
            model.img_channels,
            model.img_size,
            model.img_size,
            generator=noise_generator,
            device=device,
            dtype=torch.float32,
        ) * float(model.noise_scale)
        labels = torch.randint(
            0,
            1000,
            (batch,),
            generator=label_generator,
            device=device,
        )
        with torch.inference_mode(), autocast_context(device, amp):
            raw = pmf_one_step(model, noise, labels)
            images = generator_output_to_unit_interval(raw.float())
            full = encoder(images)
        transformed = transform_features(full, projection, mean, whitening)
        feature_parts.append(transformed.cpu())
        label_parts.append(labels.cpu())
        noise_seed_parts.append(torch.arange(completed, completed + batch))
        completed += batch
    return (
        torch.cat(feature_parts),
        torch.cat(label_parts),
        torch.cat(noise_seed_parts),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--warmstart",
        type=Path,
        default=Path(
            "/data/users/zhoushunyu/eqvae/experiments/"
            "advfd_cleanroom_pmf_projected_pilot_v1/warmstart.pt"
        ),
    )
    parser.add_argument(
        "--pmf-repo",
        type=Path,
        default=Path("/data/users/zhoushunyu/research_repos/pMF"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/data/users/zhoushunyu/research_repos/FD-Loss-assets/pMF-B_256.pth"
        ),
    )
    parser.add_argument(
        "--packed-data",
        type=Path,
        default=Path("/data/shared/imagenet-1k/random_access_v1"),
    )
    parser.add_argument("--train-samples", type=int, default=8192)
    parser.add_argument("--heldout-samples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--ridge-fraction", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--no-amp", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.train_samples <= 0 or args.heldout_samples <= 0:
        raise ValueError("sample counts must be positive")
    if args.batch_size <= 0 or args.ridge_fraction < 0:
        raise ValueError("invalid batch/whitening configuration")
    device = torch.device(args.device)
    started = time.perf_counter()
    warmstart = torch.load(args.warmstart, map_location="cpu", weights_only=False)
    if int(warmstart["feature_dim"]) != 64:
        raise ValueError("the frozen audit requires the established 64D projection")
    projection, mean, whitening, whitening_diagnostics = whitening_from_warmstart(
        warmstart,
        ridge_fraction=args.ridge_fraction,
        device=device,
    )
    encoder = DifferentiableInception2048(trainable=False).to(device).eval()
    model = load_pmf_b16(repo=args.pmf_repo, checkpoint=args.checkpoint, device=device)
    model.eval().requires_grad_(False)
    train_end = args.train_samples
    heldout_end = train_end + args.heldout_samples
    print("collecting real train bank", flush=True)
    real_train, real_train_labels, real_train_indices = collect_real(
        encoder=encoder,
        projection=projection,
        mean=mean,
        whitening=whitening,
        packed_data=args.packed_data,
        indices=range(0, train_end),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augmentation_seed=args.seed + 11,
        horizontal_flip=True,
        amp=not args.no_amp,
        device=device,
    )
    print("collecting real heldout bank", flush=True)
    real_heldout, real_heldout_labels, real_heldout_indices = collect_real(
        encoder=encoder,
        projection=projection,
        mean=mean,
        whitening=whitening,
        packed_data=args.packed_data,
        indices=range(train_end, heldout_end),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augmentation_seed=args.seed + 13,
        horizontal_flip=False,
        amp=not args.no_amp,
        device=device,
    )
    print("collecting fake train bank", flush=True)
    fake_train, fake_train_labels, fake_train_indices = collect_fake(
        model=model,
        encoder=encoder,
        projection=projection,
        mean=mean,
        whitening=whitening,
        count=args.train_samples,
        batch_size=args.batch_size,
        noise_seed=args.seed + 101,
        label_seed=args.seed + 103,
        amp=not args.no_amp,
        device=device,
    )
    print("collecting fake heldout bank", flush=True)
    fake_heldout, fake_heldout_labels, fake_heldout_indices = collect_fake(
        model=model,
        encoder=encoder,
        projection=projection,
        mean=mean,
        whitening=whitening,
        count=args.heldout_samples,
        batch_size=args.batch_size,
        noise_seed=args.seed + 201,
        label_seed=args.seed + 203,
        amp=not args.no_amp,
        device=device,
    )
    diagnostics = {
        "whitening": whitening_diagnostics,
        "real_train": summarize_bank(real_train),
        "real_heldout": summarize_bank(real_heldout),
        "fake_train": summarize_bank(fake_train),
        "fake_heldout": summarize_bank(fake_heldout),
    }
    payload = {
        "protocol": "frozen_pmf_b_inception64_residual_feature_bank_v1",
        "projection": projection.cpu(),
        "real_mean": mean.cpu(),
        "whitening": whitening.cpu(),
        "real_train": real_train,
        "real_train_labels": real_train_labels,
        "real_train_indices": real_train_indices,
        "real_heldout": real_heldout,
        "real_heldout_labels": real_heldout_labels,
        "real_heldout_indices": real_heldout_indices,
        "fake_train": fake_train,
        "fake_train_labels": fake_train_labels,
        "fake_train_indices": fake_train_indices,
        "fake_heldout": fake_heldout,
        "fake_heldout_labels": fake_heldout_labels,
        "fake_heldout_indices": fake_heldout_indices,
        "diagnostics": diagnostics,
        "config": {
            "train_samples": args.train_samples,
            "heldout_samples": args.heldout_samples,
            "batch_size": args.batch_size,
            "ridge_fraction": args.ridge_fraction,
            "seed": args.seed,
            "amp": not args.no_amp,
            "warmstart": str(args.warmstart),
            "warmstart_sha256": sha256(args.warmstart),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256(args.checkpoint),
            "packed_data": str(args.packed_data),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(
        json.dumps(
            {
                "protocol": payload["protocol"],
                "diagnostics": diagnostics,
                "config": payload["config"],
                "elapsed_seconds": payload["elapsed_seconds"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(diagnostics, indent=2), flush=True)
    print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
