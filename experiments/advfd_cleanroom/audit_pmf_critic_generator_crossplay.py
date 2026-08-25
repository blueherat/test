#!/usr/bin/env python3
"""Cross-evaluate pMF generators under pretrained and adaptive AdvFD critics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms.functional import pil_to_tensor


EQVAE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EQVAE_ROOT))

from experiments.advfd_cleanroom.audit_advfd_temporal_gauge_gradients import (
    select_adv_state,
)
from experiments.advfd_cleanroom.temporal_gauge import (
    real_whitened_fd_components_from_stats,
)


DEFAULT_OFFICIAL_ROOT = Path("/data/users/zhoushunyu/research_repos/AdvFD")


class FlatImageDataset(Dataset[torch.Tensor]):
    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            return pil_to_tensor(image.convert("RGB")).float().div_(255.0)


def named_path(value: str) -> tuple[str, Path | None]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=PATH or LABEL=pretrained")
    label, raw_path = value.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("label must be nonempty")
    if raw_path == "pretrained":
        return label, None
    return label, Path(raw_path).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--critic",
        action="append",
        type=named_path,
        required=True,
        help="LABEL=CHECKPOINT, or LABEL=pretrained",
    )
    parser.add_argument(
        "--image-folder",
        action="append",
        type=named_path,
        required=True,
        help="LABEL=FLAT_PNG_FOLDER",
    )
    parser.add_argument(
        "--packed-imagenet-root",
        type=Path,
        default=Path("/data/shared/imagenet-1k/random_access_v1"),
    )
    parser.add_argument("--real-split", choices=("train", "val"), default="val")
    parser.add_argument("--sample-count", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--whiten-eps", type=float, default=1e-3)
    parser.add_argument("--cka-samples", type=int, default=512)
    parser.add_argument(
        "--generator-anchor",
        default="static",
        help="Generator row used to remove critic-specific scale in cross-play tables.",
    )
    parser.add_argument("--official-root", type=Path, default=DEFAULT_OFFICIAL_ROOT)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def ensure_unique(entries: list[tuple[str, Path | None]], kind: str) -> None:
    labels = [label for label, _ in entries]
    if len(labels) != len(set(labels)):
        raise ValueError(f"duplicate {kind} label")


def common_image_paths(
    folders: list[tuple[str, Path | None]], count: int, seed: int
) -> dict[str, list[Path]]:
    by_label: dict[str, dict[str, Path]] = {}
    for label, folder in folders:
        if folder is None or not folder.is_dir():
            raise FileNotFoundError(f"image folder not found for {label}: {folder}")
        mapping = {path.name: path for path in folder.glob("*.png")}
        by_label[label] = mapping
    common = set.intersection(*(set(mapping) for mapping in by_label.values()))
    if len(common) < count:
        raise ValueError(f"only {len(common)} paired PNG names for {count} samples")
    names = np.asarray(sorted(common))
    rng = np.random.default_rng(seed)
    selected = sorted(rng.choice(names, size=count, replace=False).tolist())
    return {
        label: [mapping[name] for name in selected]
        for label, mapping in by_label.items()
    }


@torch.inference_mode()
def extract_features(
    critic: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    chunks = []
    for batch in loader:
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        images = images.to(device, non_blocking=True)
        features, _ = critic(images)
        chunks.append(features.detach().float().cpu())
    return torch.cat(chunks, dim=0)


def moments(features: torch.Tensor, device: torch.device):
    values = features.to(device=device, dtype=torch.float64)
    mean = values.mean(dim=0)
    second = values.mT @ values / values.shape[0]
    covariance = second - mean[:, None] * mean[None, :]
    covariance = 0.5 * (covariance + covariance.mT)
    return mean, covariance


def participation_rank(covariance: torch.Tensor) -> float:
    trace = torch.trace(covariance).clamp_min(0.0)
    squared = covariance.square().sum()
    return float(trace.square() / squared.clamp_min(1e-30))


def linear_cka(first: torch.Tensor, second: torch.Tensor, max_samples: int) -> float:
    count = min(first.shape[0], second.shape[0], max_samples)
    first = first[:count].double()
    second = second[:count].double()
    first = first - first.mean(dim=0, keepdim=True)
    second = second - second.mean(dim=0, keepdim=True)
    first_gram = first @ first.mT
    second_gram = second @ second.mT
    numerator = (first_gram * second_gram).sum()
    denominator = torch.linalg.vector_norm(first_gram) * torch.linalg.vector_norm(
        second_gram
    )
    return float(numerator / denominator.clamp_min(1e-30))


def interaction_summary(rows: list[dict[str, Any]], component: str) -> dict[str, Any]:
    critic_labels = sorted({str(row["critic"]) for row in rows})
    generator_labels = sorted({str(row["generator"]) for row in rows})
    lookup = {
        (str(row["critic"]), str(row["generator"])): float(row[component])
        for row in rows
    }
    matrix = np.asarray(
        [
            [lookup[(critic, generator)] for generator in generator_labels]
            for critic in critic_labels
        ],
        dtype=np.float64,
    )
    centered = matrix - matrix.mean()
    interaction = (
        matrix
        - matrix.mean(axis=1, keepdims=True)
        - matrix.mean(axis=0, keepdims=True)
        + matrix.mean()
    )
    return {
        "critic_labels": critic_labels,
        "generator_labels": generator_labels,
        "matrix": matrix.tolist(),
        "interaction_frobenius_fraction": float(
            np.linalg.norm(interaction) / max(np.linalg.norm(centered), 1e-30)
        ),
    }


def calibrate_crossplay_rows(
    rows: list[dict[str, Any]],
    *,
    anchor_generator: str,
    real_null_by_critic: dict[str, dict[str, float]],
) -> None:
    """Add within-critic ratios that are invariant to row-wise score scaling."""

    components = ("mean_fd", "covariance_fd", "full_fd")
    anchor_by_critic = {
        str(row["critic"]): row
        for row in rows
        if str(row["generator"]) == anchor_generator
    }
    critic_labels = {str(row["critic"]) for row in rows}
    missing = critic_labels - set(anchor_by_critic)
    if missing:
        raise ValueError(
            f"anchor generator {anchor_generator!r} missing for critics {sorted(missing)}"
        )
    if critic_labels != set(real_null_by_critic):
        raise ValueError("real-null calibration does not match critic labels")

    tiny = np.finfo(np.float64).tiny
    for row in rows:
        critic = str(row["critic"])
        anchor = anchor_by_critic[critic]
        null = real_null_by_critic[critic]
        for component in components:
            value = max(float(row[component]), tiny)
            anchor_value = max(float(anchor[component]), tiny)
            null_value = max(float(null[component]), tiny)
            row[f"{component}_over_anchor"] = value / anchor_value
            row[f"log_{component}_over_anchor"] = math.log(value / anchor_value)
            row[f"{component}_over_real_null"] = value / null_value


def load_critic(
    checkpoint_path: Path | None,
    *,
    device: torch.device,
):
    from frechet_distance.repr_models import load_repr_model

    critic, _, _, _ = load_repr_model("inception", device=str(device))
    metadata: dict[str, Any] = {"checkpoint": "pretrained"}
    if checkpoint_path is not None:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
        adv_state = select_adv_state(checkpoint)
        critic.load_state_dict(adv_state["model"], strict=True)
        metadata = {
            "checkpoint": str(checkpoint_path),
            "step": int(checkpoint.get("step", checkpoint.get("current_step", -1))),
        }
        del checkpoint, adv_state
    critic.eval().requires_grad_(False)
    return critic, metadata


def main() -> None:
    args = parse_args()
    if args.sample_count < 2 or args.batch_size < 1 or args.cka_samples < 2:
        raise ValueError("invalid sample or batch count")
    ensure_unique(args.critic, "critic")
    ensure_unique(args.image_folder, "image-folder")
    generator_labels = {label for label, _ in args.image_folder}
    if args.generator_anchor not in generator_labels:
        raise ValueError(
            f"generator anchor {args.generator_anchor!r} is not an image-folder label"
        )
    official_root = args.official_root.expanduser().resolve()
    sys.path.insert(0, str(official_root))
    device = torch.device(args.device)

    from experiments.raev2_training_core import DeterministicImageNetPacked

    selected_images = common_image_paths(
        args.image_folder, args.sample_count, args.seed + 1
    )
    image_loaders = {
        label: DataLoader(
            FlatImageDataset(paths),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        for label, paths in selected_images.items()
    }
    real_dataset = DeterministicImageNetPacked(
        args.packed_imagenet_root,
        split=args.real_split,
        image_size=256,
        augmentation_seed=1,
        horizontal_flip=False,
    )
    rng = np.random.default_rng(args.seed + 2)
    real_indices = rng.choice(
        len(real_dataset), size=2 * args.sample_count, replace=False
    ).tolist()
    real_loader = DataLoader(
        Subset(real_dataset, real_indices[: args.sample_count]),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    real_null_loader = DataLoader(
        Subset(real_dataset, real_indices[args.sample_count :]),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    rows: list[dict[str, Any]] = []
    real_features_by_critic: dict[str, torch.Tensor] = {}
    fake_features_by_critic: dict[str, dict[str, torch.Tensor]] = {}
    critic_metadata: dict[str, Any] = {}
    real_null_by_critic: dict[str, dict[str, float]] = {}
    for critic_label, checkpoint_path in args.critic:
        critic, metadata = load_critic(checkpoint_path, device=device)
        critic_metadata[critic_label] = metadata
        real_features = extract_features(critic, real_loader, device)
        real_features_by_critic[critic_label] = real_features
        real_mean, real_covariance = moments(real_features, device)
        real_null_features = extract_features(critic, real_null_loader, device)
        real_null_mean, real_null_covariance = moments(real_null_features, device)
        null_mean_term, null_covariance_term, _ = (
            real_whitened_fd_components_from_stats(
                real_mean,
                real_covariance,
                real_null_mean,
                real_null_covariance,
                epsilon=args.whiten_eps,
            )
        )
        real_null_by_critic[critic_label] = {
            "mean_fd": float(null_mean_term),
            "covariance_fd": float(null_covariance_term),
            "full_fd": float(null_mean_term + null_covariance_term),
        }
        fake_features_by_critic[critic_label] = {}
        for generator_label, loader in image_loaders.items():
            fake_features = extract_features(critic, loader, device)
            fake_features_by_critic[critic_label][generator_label] = fake_features
            fake_mean, fake_covariance = moments(fake_features, device)
            mean_term, covariance_term, _ = real_whitened_fd_components_from_stats(
                real_mean,
                real_covariance,
                fake_mean,
                fake_covariance,
                epsilon=args.whiten_eps,
            )
            rows.append(
                {
                    "critic": critic_label,
                    "generator": generator_label,
                    "sample_count": args.sample_count,
                    "mean_fd": float(mean_term),
                    "covariance_fd": float(covariance_term),
                    "full_fd": float(mean_term + covariance_term),
                    "covariance_fraction": float(
                        covariance_term
                        / (mean_term + covariance_term).clamp_min(1e-30)
                    ),
                    "real_feature_rms": float(real_features.double().square().mean().sqrt()),
                    "fake_feature_rms": float(fake_features.double().square().mean().sqrt()),
                    "real_effective_rank": participation_rank(real_covariance),
                    "fake_effective_rank": participation_rank(fake_covariance),
                }
            )
            del fake_mean, fake_covariance
        del (
            critic,
            real_mean,
            real_covariance,
            real_null_features,
            real_null_mean,
            real_null_covariance,
        )
        torch.cuda.empty_cache()
        print(f"completed critic {critic_label}", flush=True)

    calibrate_crossplay_rows(
        rows,
        anchor_generator=args.generator_anchor,
        real_null_by_critic=real_null_by_critic,
    )

    critic_labels = [label for label, _ in args.critic]
    cka_rows = []
    for first_index, first in enumerate(critic_labels):
        for second in critic_labels[first_index + 1 :]:
            entry: dict[str, Any] = {
                "first_critic": first,
                "second_critic": second,
                "real_cka": linear_cka(
                    real_features_by_critic[first],
                    real_features_by_critic[second],
                    args.cka_samples,
                ),
            }
            for generator_label in image_loaders:
                entry[f"{generator_label}_cka"] = linear_cka(
                    fake_features_by_critic[first][generator_label],
                    fake_features_by_critic[second][generator_label],
                    args.cka_samples,
                )
            cka_rows.append(entry)

    args.output_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_root / "crossplay.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cka_path = args.output_root / "critic_cka.csv"
    with cka_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cka_rows[0]))
        writer.writeheader()
        writer.writerows(cka_rows)
    result = {
        "protocol": "advfd_pmf_critic_generator_crossplay_v1",
        "sample_count": args.sample_count,
        "real_split": args.real_split,
        "whiten_epsilon": args.whiten_eps,
        "generator_anchor": args.generator_anchor,
        "critic_metadata": critic_metadata,
        "real_null_by_critic": real_null_by_critic,
        "interactions": {
            component: interaction_summary(rows, component)
            for component in ("mean_fd", "covariance_fd", "full_fd")
        },
        "anchor_relative_log_interactions": {
            component: interaction_summary(rows, f"log_{component}_over_anchor")
            for component in ("mean_fd", "covariance_fd", "full_fd")
        },
        "interpretation_boundary": (
            "Cross-play uses fresh real probes rather than checkpoint EMA moments, "
            "plus paired generated-image moments. "
            "A large row-by-column interaction or unstable critic CKA is evidence of "
            "critic-generator co-adaptation, not by itself evidence of generator mode collapse."
        ),
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["interactions"], indent=2), flush=True)


if __name__ == "__main__":
    main()
