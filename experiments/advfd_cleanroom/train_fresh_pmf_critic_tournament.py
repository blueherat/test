#!/usr/bin/env python3
"""Train a fresh AdvFD critic against one frozen pMF image bank.

The critic only sees the training split of the target generator.  Every saved
cross-play result is computed on disjoint real and generated images shared by
all generator conditions.  This distinguishes a co-trained critic being
satisfied from broad improvement across independently optimized witnesses.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


EQVAE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EQVAE_ROOT))

from experiments.advfd_cleanroom.audit_pmf_critic_generator_crossplay import (
    FlatImageDataset,
    calibrate_crossplay_rows,
    extract_features,
    moments,
    named_path,
    participation_rank,
)


DEFAULT_OFFICIAL_ROOT = Path("/data/users/zhoushunyu/research_repos/AdvFD")
DEFAULT_REFERENCE_STATS = Path(
    "/data/users/zhoushunyu/research_deps/advfd_reference_stats/"
    "guided_diffusion_stats.npz"
)


def parse_int_set(value: str) -> tuple[int, ...]:
    values = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    if any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("steps must be nonnegative")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-folder",
        action="append",
        type=named_path,
        required=True,
        help="LABEL=FLAT_PNG_FOLDER; all folders must share PNG filenames",
    )
    parser.add_argument("--target-generator", required=True)
    parser.add_argument("--train-count", type=int, default=2500)
    parser.add_argument("--eval-count", type=int, default=2500)
    parser.add_argument("--real-train-count", type=int, default=50000)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument(
        "--eval-steps",
        type=parse_int_set,
        default=(0, 100, 250, 500, 1000),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--stats-ema-beta", type=float, default=0.99)
    parser.add_argument("--whiten-eps", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument(
        "--packed-imagenet-root",
        type=Path,
        default=Path("/data/shared/imagenet-1k/random_access_v1"),
    )
    parser.add_argument(
        "--reference-stats", type=Path, default=DEFAULT_REFERENCE_STATS
    )
    parser.add_argument("--official-root", type=Path, default=DEFAULT_OFFICIAL_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--save-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def _validate_image_folders(
    folders: list[tuple[str, Path | None]],
) -> dict[str, dict[str, Path]]:
    labels = [label for label, _ in folders]
    if len(labels) != len(set(labels)):
        raise ValueError("duplicate image-folder label")
    mappings: dict[str, dict[str, Path]] = {}
    for label, folder in folders:
        if folder is None or not folder.is_dir():
            raise FileNotFoundError(f"image folder not found for {label}: {folder}")
        mapping = {path.name: path for path in folder.glob("*.png")}
        if not mapping:
            raise ValueError(f"image folder has no PNG files for {label}: {folder}")
        mappings[label] = mapping
    return mappings


def paired_train_eval_paths(
    folders: list[tuple[str, Path | None]],
    *,
    train_count: int,
    eval_count: int,
    seed: int,
) -> tuple[dict[str, list[Path]], dict[str, list[Path]], dict[str, Any]]:
    """Create filename-paired, disjoint train/evaluation generator splits."""

    if train_count < 2 or eval_count < 2:
        raise ValueError("train-count and eval-count must both be at least two")
    mappings = _validate_image_folders(folders)
    common = sorted(set.intersection(*(set(mapping) for mapping in mappings.values())))
    required = train_count + eval_count
    if len(common) < required:
        raise ValueError(f"only {len(common)} paired PNG names for {required} samples")
    rng = np.random.default_rng(seed)
    selected = rng.choice(np.asarray(common), size=required, replace=False).tolist()
    train_names = sorted(selected[:train_count])
    eval_names = sorted(selected[train_count:])
    if set(train_names) & set(eval_names):
        raise AssertionError("train/evaluation generator filenames overlap")
    train_paths = {
        label: [mapping[name] for name in train_names]
        for label, mapping in mappings.items()
    }
    eval_paths = {
        label: [mapping[name] for name in eval_names]
        for label, mapping in mappings.items()
    }
    manifest = {
        "common_count": len(common),
        "train_count": train_count,
        "eval_count": eval_count,
        "train_name_sha256": hashlib.sha256(
            "\n".join(train_names).encode("utf-8")
        ).hexdigest(),
        "eval_name_sha256": hashlib.sha256(
            "\n".join(eval_names).encode("utf-8")
        ).hexdigest(),
    }
    return train_paths, eval_paths, manifest


def infinite_batches(loader: DataLoader) -> Iterator[Any]:
    while True:
        yield from loader


def images_from_batch(batch: Any, device: torch.device) -> torch.Tensor:
    images = batch[0] if isinstance(batch, (tuple, list)) else batch
    return images.to(device, non_blocking=True)


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
        persistent_workers=num_workers > 0,
    )


def extract_primary(critic: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    primary, _ = critic(images)
    return primary


@torch.no_grad()
def initialize_fake_stats(
    critic: torch.nn.Module,
    loader: DataLoader,
    stats: torch.nn.Module,
    device: torch.device,
) -> dict[str, float]:
    features = extract_features(critic, loader, device)
    mean, covariance = moments(features, device)
    stats.initialize_from_mean_cov(mean, covariance)
    return {
        "feature_rms": float(features.double().square().mean().sqrt()),
        "feature_effective_rank": participation_rank(covariance),
    }


def covariance_contribution_summary(
    real_mean: torch.Tensor,
    real_covariance: torch.Tensor,
    fake_covariance: torch.Tensor,
    *,
    epsilon: float,
) -> dict[str, Any]:
    """Summarize the actual whitened FD covariance-mode contributions."""

    del real_mean
    dim = real_covariance.shape[0]
    eye = torch.eye(dim, device=real_covariance.device, dtype=torch.float64)
    real_regularized = 0.5 * (real_covariance + real_covariance.mT) + epsilon * eye
    values, vectors = torch.linalg.eigh(real_regularized)
    inverse_sqrt = values.clamp_min(epsilon).rsqrt()
    fake_regularized = 0.5 * (fake_covariance + fake_covariance.mT) + epsilon * eye
    fake_eigenbasis = vectors.mT @ fake_regularized @ vectors
    fake_white = (
        fake_eigenbasis * inverse_sqrt[:, None] * inverse_sqrt[None, :]
    )
    eigenvalues = torch.linalg.eigvalsh(0.5 * (fake_white + fake_white.mT)).clamp_min(0)
    contributions = (eigenvalues.sqrt() - 1.0).square()
    total = contributions.sum().clamp_min(1e-30)
    sorted_contributions = contributions.sort(descending=True).values
    probabilities = contributions / total
    participation = total.square() / contributions.square().sum().clamp_min(1e-30)
    entropy_rank = torch.exp(
        -(probabilities * probabilities.clamp_min(1e-30).log()).sum()
    )
    return {
        "participation_rank": float(participation),
        "entropy_effective_rank": float(entropy_rank),
        "top_k_share": {
            str(k): float(sorted_contributions[: min(k, dim)].sum() / total)
            for k in (1, 2, 4, 8, 16, 32, 64, 128, 256)
        },
    }


@torch.inference_mode()
def evaluate_crossplay(
    critic: torch.nn.Module,
    *,
    real_loader: DataLoader,
    real_null_loader: DataLoader,
    fake_loaders: dict[str, DataLoader],
    device: torch.device,
    epsilon: float,
    anchor_generator: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from experiments.advfd_cleanroom.temporal_gauge import (
        real_whitened_fd_components_from_stats,
    )

    real_features = extract_features(critic, real_loader, device)
    real_mean, real_covariance = moments(real_features, device)
    null_features = extract_features(critic, real_null_loader, device)
    null_mean, null_covariance = moments(null_features, device)
    null_mean_fd, null_covariance_fd, _ = real_whitened_fd_components_from_stats(
        real_mean,
        real_covariance,
        null_mean,
        null_covariance,
        epsilon=epsilon,
    )
    null = {
        "mean_fd": float(null_mean_fd),
        "covariance_fd": float(null_covariance_fd),
        "full_fd": float(null_mean_fd + null_covariance_fd),
    }
    rows: list[dict[str, Any]] = []
    contribution_summaries: dict[str, Any] = {}
    for label, loader in fake_loaders.items():
        fake_features = extract_features(critic, loader, device)
        fake_mean, fake_covariance = moments(fake_features, device)
        mean_fd, covariance_fd, _ = real_whitened_fd_components_from_stats(
            real_mean,
            real_covariance,
            fake_mean,
            fake_covariance,
            epsilon=epsilon,
        )
        rows.append(
            {
                "critic": "fresh",
                "generator": label,
                "sample_count": int(fake_features.shape[0]),
                "mean_fd": float(mean_fd),
                "covariance_fd": float(covariance_fd),
                "full_fd": float(mean_fd + covariance_fd),
                "covariance_fraction": float(
                    covariance_fd / (mean_fd + covariance_fd).clamp_min(1e-30)
                ),
                "real_feature_rms": float(
                    real_features.double().square().mean().sqrt()
                ),
                "fake_feature_rms": float(
                    fake_features.double().square().mean().sqrt()
                ),
                "real_effective_rank": participation_rank(real_covariance),
                "fake_effective_rank": participation_rank(fake_covariance),
            }
        )
        contribution_summaries[label] = covariance_contribution_summary(
            real_mean,
            real_covariance,
            fake_covariance,
            epsilon=epsilon,
        )
    calibrate_crossplay_rows(
        rows,
        anchor_generator=anchor_generator,
        real_null_by_critic={"fresh": null},
    )
    return rows, {
        "real_null": null,
        "covariance_contributions": contribution_summaries,
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(
    path: Path,
    *,
    step: int,
    critic: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    real_stats: torch.nn.Module,
    fake_stats: torch.nn.Module,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": int(step),
            "protocol": "fresh_pmf_advfd_critic_tournament_v1",
            "config": config,
            "fd_adv_states": [
                {
                    "name": "inception",
                    "model": critic.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "real_stats": real_stats.state_dict(),
                    "fake_stats": fake_stats.state_dict(),
                }
            ],
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if args.steps < 0 or args.batch_size < 2 or args.eval_batch_size < 1:
        raise ValueError("invalid steps or batch size")
    eval_steps = tuple(sorted(set(args.eval_steps) | {0, args.steps}))
    if eval_steps[-1] > args.steps:
        raise ValueError("eval-steps cannot exceed steps")
    labels = {label for label, _ in args.image_folder}
    if args.target_generator not in labels:
        raise ValueError("target-generator is not one of the image folders")
    if "static" not in labels:
        raise ValueError("a static image-folder anchor is required")

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    official_root = args.official_root.expanduser().resolve()
    sys.path.insert(0, str(official_root))
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    from experiments.raev2_training_core import DeterministicImageNetPacked
    from frechet_distance.adversarial import (
        FeatureStatsEMA,
        build_real_whitening,
        real_whitened_frechet_distance_from_stats,
    )
    from frechet_distance.losses import load_mu_and_sigma_reference
    from frechet_distance.repr_models import load_repr_model

    train_paths, eval_paths, split_manifest = paired_train_eval_paths(
        args.image_folder,
        train_count=args.train_count,
        eval_count=args.eval_count,
        seed=args.seed + 1,
    )
    fake_train_loader = make_loader(
        FlatImageDataset(train_paths[args.target_generator]),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        seed=args.seed + 10,
    )
    fake_init_loader = make_loader(
        FlatImageDataset(train_paths[args.target_generator]),
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        seed=args.seed + 11,
    )
    fake_eval_loaders = {
        label: make_loader(
            FlatImageDataset(paths),
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            seed=args.seed + 12,
        )
        for label, paths in eval_paths.items()
    }

    real_dataset = DeterministicImageNetPacked(
        args.packed_imagenet_root,
        split="train",
        image_size=256,
        augmentation_seed=args.seed,
        horizontal_flip=False,
    )
    real_required = args.real_train_count + 2 * args.eval_count
    if real_required > len(real_dataset):
        raise ValueError("requested more real images than the packed dataset contains")
    real_rng = np.random.default_rng(args.seed + 2)
    real_indices = real_rng.choice(len(real_dataset), size=real_required, replace=False)
    train_indices = real_indices[: args.real_train_count].tolist()
    eval_indices = real_indices[
        args.real_train_count : args.real_train_count + args.eval_count
    ].tolist()
    null_indices = real_indices[args.real_train_count + args.eval_count :].tolist()
    if set(train_indices) & (set(eval_indices) | set(null_indices)):
        raise AssertionError("real training and evaluation indices overlap")
    real_train_loader = make_loader(
        Subset(real_dataset, train_indices),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        seed=args.seed + 20,
    )
    real_eval_loader = make_loader(
        Subset(real_dataset, eval_indices),
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        seed=args.seed + 21,
    )
    real_null_loader = make_loader(
        Subset(real_dataset, null_indices),
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        seed=args.seed + 22,
    )

    critic, feature_dim, _, _ = load_repr_model("inception", device=str(device))
    critic.eval().requires_grad_(True)
    optimizer = torch.optim.AdamW(
        critic.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    real_stats = FeatureStatsEMA(feature_dim, beta=args.stats_ema_beta).to(device)
    fake_stats = FeatureStatsEMA(feature_dim, beta=args.stats_ema_beta).to(device)
    reference_mean, reference_covariance = load_mu_and_sigma_reference(
        str(args.reference_stats), pool_type="cls"
    )
    reference_mean = reference_mean.to(device)
    reference_covariance = reference_covariance.to(device)
    real_stats.initialize_from_mean_cov(reference_mean, reference_covariance)
    fake_init = initialize_fake_stats(
        critic, fake_init_loader, fake_stats, device
    )

    config = {
        "target_generator": args.target_generator,
        "seed": args.seed,
        "steps": args.steps,
        "eval_steps": eval_steps,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "betas": [args.beta1, args.beta2],
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "stats_ema_beta": args.stats_ema_beta,
        "whiten_eps": args.whiten_eps,
        "split": split_manifest,
        "fake_initialization": fake_init,
        "real_train_count": args.real_train_count,
        "real_eval_count": args.eval_count,
        "real_null_count": args.eval_count,
        "reference_stats": str(args.reference_stats),
        "official_root": str(official_root),
        "image_folders": {
            label: str(path) for label, path in args.image_folder
        },
    }
    (output_root / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    train_fake_batches = infinite_batches(fake_train_loader)
    train_real_batches = infinite_batches(real_train_loader)
    history: list[dict[str, Any]] = []
    all_eval_rows: list[dict[str, Any]] = []
    started = time.monotonic()

    def run_evaluation(step: int) -> None:
        critic.eval()
        rows, details = evaluate_crossplay(
            critic,
            real_loader=real_eval_loader,
            real_null_loader=real_null_loader,
            fake_loaders=fake_eval_loaders,
            device=device,
            epsilon=args.whiten_eps,
            anchor_generator="static",
        )
        for row in rows:
            row["step"] = step
            row["target_generator"] = args.target_generator
            row["seed"] = args.seed
        all_eval_rows.extend(rows)
        write_rows(output_root / f"crossplay_step_{step:06d}.csv", rows)
        (output_root / f"details_step_{step:06d}.json").write_text(
            json.dumps(details, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        write_rows(output_root / "crossplay_all.csv", all_eval_rows)
        target_row = next(row for row in rows if row["generator"] == args.target_generator)
        print(
            json.dumps(
                {
                    "event": "evaluation",
                    "step": step,
                    "target": args.target_generator,
                    "target_full_fd_over_static": target_row["full_fd_over_anchor"],
                    "target_full_fd_over_real_null": target_row["full_fd_over_real_null"],
                    "elapsed_seconds": time.monotonic() - started,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    if 0 in eval_steps:
        run_evaluation(0)

    for step in range(1, args.steps + 1):
        real_images = images_from_batch(next(train_real_batches), device)
        fake_images = images_from_batch(next(train_fake_batches), device)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            real_features = extract_primary(critic, real_images).detach()
        fake_features = extract_primary(critic, fake_images)
        real_mean, real_covariance = real_stats.build_stats(real_features)
        fake_mean, fake_covariance = fake_stats.build_stats(fake_features)
        real_whitening = build_real_whitening(
            real_mean, real_covariance, eps=args.whiten_eps
        )
        fd = real_whitened_frechet_distance_from_stats(
            real_mean,
            real_covariance,
            fake_mean,
            fake_covariance,
            eps=args.whiten_eps,
            real_whitening=real_whitening,
        )
        (-fd).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            critic.parameters(), args.grad_clip
        )
        optimizer.step()
        with torch.no_grad():
            real_update = extract_primary(critic, real_images).detach()
            fake_update = extract_primary(critic, fake_images).detach()
            real_stats.update(real_update)
            fake_stats.update(fake_update)
        history.append(
            {
                "step": step,
                "critic_fd": float(fd.detach()),
                "grad_norm": float(grad_norm),
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        if step == 1 or step % 20 == 0:
            print(json.dumps(history[-1]), flush=True)
        if step in eval_steps:
            run_evaluation(step)
        if args.save_checkpoints and step in {args.steps}:
            save_checkpoint(
                output_root / "checkpoints" / f"step_{step:06d}.pth",
                step=step,
                critic=critic,
                optimizer=optimizer,
                real_stats=real_stats,
                fake_stats=fake_stats,
                config=config,
            )
        if step % 20 == 0 or step == args.steps:
            write_rows(output_root / "training_history.csv", history)

    summary = {
        "protocol": "fresh_pmf_advfd_critic_tournament_v1",
        "config": config,
        "final_step": args.steps,
        "elapsed_seconds": time.monotonic() - started,
        "crossplay_rows": all_eval_rows,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
