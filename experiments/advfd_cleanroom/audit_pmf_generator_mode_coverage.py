#!/usr/bin/env python3
"""Audit pMF generator coverage in a fixed, non-adaptive Inception space."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import pil_to_tensor
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3

from experiments.advfd_cleanroom.audit_pmf_critic_generator_crossplay import (
    common_image_paths,
    ensure_unique,
    named_path,
)


DEFAULT_REAL_FEATURES = Path(
    "/home/zhoushunyu/.cache/torch/fidelity_cache/"
    "imagenet256_virtual_reference_10k-inception-v3-compat-features-2048.pt"
)
DEFAULT_REAL_LOGITS = Path(
    "/home/zhoushunyu/.cache/torch/fidelity_cache/"
    "imagenet256_virtual_reference_10k-inception-v3-compat-features-logits_unbiased.pt"
)


class Uint8ImageDataset(Dataset[torch.Tensor]):
    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            return pil_to_tensor(image.convert("RGB"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-folder", action="append", type=named_path, required=True
    )
    parser.add_argument("--sample-count", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--distance-batch", type=int, default=128)
    parser.add_argument("--neighborhood", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--real-features", type=Path, default=DEFAULT_REAL_FEATURES)
    parser.add_argument("--real-logits", type=Path, default=DEFAULT_REAL_LOGITS)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


@torch.inference_mode()
def extract_inception(
    paths: list[Path],
    *,
    extractor: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        Uint8ImageDataset(paths),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    feature_chunks = []
    logit_chunks = []
    for images in loader:
        outputs = extractor(images.to(device, non_blocking=True))
        feature_chunks.append(outputs[0].float().cpu())
        logit_chunks.append(outputs[1].float().cpu())
    return torch.cat(feature_chunks), torch.cat(logit_chunks)


@torch.inference_mode()
def manifold_radii(
    features: torch.Tensor, *, neighborhood: int, batch_size: int
) -> torch.Tensor:
    if features.ndim != 2 or features.shape[0] <= neighborhood:
        raise ValueError("not enough feature vectors for neighborhood estimate")
    chunks = []
    for start in range(0, features.shape[0], batch_size):
        distances = torch.cdist(features[start : start + batch_size], features)
        chunks.append(distances.kthvalue(neighborhood + 1, dim=1).values)
    return torch.cat(chunks)


@torch.inference_mode()
def fixed_space_coverage_metrics(
    generated: torch.Tensor,
    reference: torch.Tensor,
    *,
    generated_radii: torch.Tensor,
    reference_radii: torch.Tensor,
    neighborhood: int,
    batch_size: int,
) -> dict[str, float]:
    precision_hits = 0
    density_sum = 0.0
    generated_count = generated.shape[0]
    for start in range(0, generated_count, batch_size):
        distances = torch.cdist(generated[start : start + batch_size], reference)
        memberships = distances <= reference_radii.unsqueeze(0)
        precision_hits += int(memberships.any(dim=1).sum())
        density_sum += float(memberships.sum()) / neighborhood

    recall_hits = 0
    coverage_hits = 0
    reference_count = reference.shape[0]
    for start in range(0, reference_count, batch_size):
        distances = torch.cdist(reference[start : start + batch_size], generated)
        recall_hits += int(
            (distances <= generated_radii.unsqueeze(0)).any(dim=1).sum()
        )
        coverage_hits += int(
            (distances.min(dim=1).values <= reference_radii[start : start + batch_size])
            .sum()
        )

    return {
        "precision": precision_hits / generated_count,
        "recall": recall_hits / reference_count,
        "density": density_sum / generated_count,
        "coverage": coverage_hits / reference_count,
    }


def covariance_effective_rank(features: torch.Tensor) -> float:
    centered = features.double() - features.double().mean(dim=0, keepdim=True)
    covariance = centered.mT @ centered / centered.shape[0]
    trace = torch.trace(covariance)
    return float(trace.square() / covariance.square().sum().clamp_min(1e-30))


def class_occupancy(logits: torch.Tensor) -> dict[str, float]:
    probabilities = logits.double().softmax(dim=1)
    marginal = probabilities.mean(dim=0)
    marginal_entropy = float(
        -(marginal * marginal.clamp_min(1e-30).log()).sum()
        / math.log(marginal.numel())
    )
    predictions = logits.argmax(dim=1)
    counts = torch.bincount(predictions, minlength=logits.shape[1]).double()
    occupied = int((counts > 0).sum())
    empirical = counts / counts.sum()
    top1_entropy = float(
        -(empirical * empirical.clamp_min(1e-30).log()).sum()
        / math.log(empirical.numel())
    )
    return {
        "soft_class_entropy_normalized": marginal_entropy,
        "top1_class_entropy_normalized": top1_entropy,
        "occupied_top1_classes": occupied,
        "max_top1_class_fraction": float(counts.max() / counts.sum()),
    }


def exact_duplicate_fraction(paths: list[Path]) -> float:
    digests = [hashlib.sha256(path.read_bytes()).digest() for path in paths]
    return 1.0 - len(set(digests)) / len(digests)


def quantile_fields(prefix: str, values: torch.Tensor) -> dict[str, float]:
    probabilities = torch.tensor([0.01, 0.05, 0.5, 0.95], device=values.device)
    quantiles = torch.quantile(values, probabilities).cpu().tolist()
    return {
        f"{prefix}_q01": float(quantiles[0]),
        f"{prefix}_q05": float(quantiles[1]),
        f"{prefix}_q50": float(quantiles[2]),
        f"{prefix}_q95": float(quantiles[3]),
    }


def main() -> None:
    args = parse_args()
    if args.sample_count <= args.neighborhood:
        raise ValueError("sample count must exceed neighborhood")
    ensure_unique(args.image_folder, "image-folder")
    selected = common_image_paths(args.image_folder, args.sample_count, args.seed)
    output_root = args.output_root.expanduser().resolve()
    cache_root = output_root / "features"
    cache_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    real_features_all = torch.load(
        args.real_features.expanduser().resolve(), map_location="cpu", weights_only=False
    ).float()
    real_logits_all = torch.load(
        args.real_logits.expanduser().resolve(), map_location="cpu", weights_only=False
    ).float()
    if real_features_all.shape[0] != real_logits_all.shape[0]:
        raise ValueError("real feature and logit cache counts differ")
    rng = np.random.default_rng(args.seed + 1)
    real_indices = torch.from_numpy(
        rng.choice(real_features_all.shape[0], args.sample_count, replace=False)
    )
    real_features = real_features_all.index_select(0, real_indices).to(device)
    real_logits = real_logits_all.index_select(0, real_indices)
    real_radii = manifold_radii(
        real_features,
        neighborhood=args.neighborhood,
        batch_size=args.distance_batch,
    )
    real_effective_rank = covariance_effective_rank(real_features.cpu())
    real_radius_fields = quantile_fields("nn_radius", real_radii)

    extractor = FeatureExtractorInceptionV3(
        "inception-v3-compat", ["2048", "logits_unbiased"], verbose=False
    ).to(device).eval().requires_grad_(False)
    rows: list[dict[str, Any]] = []
    for label, paths in selected.items():
        feature_path = cache_root / f"{label}_features.pt"
        logit_path = cache_root / f"{label}_logits.pt"
        if feature_path.is_file() and logit_path.is_file():
            generated_cpu = torch.load(feature_path, map_location="cpu", weights_only=True)
            logits = torch.load(logit_path, map_location="cpu", weights_only=True)
        else:
            generated_cpu, logits = extract_inception(
                paths,
                extractor=extractor,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            torch.save(generated_cpu, feature_path)
            torch.save(logits, logit_path)
        generated = generated_cpu.float().to(device)
        generated_radii = manifold_radii(
            generated,
            neighborhood=args.neighborhood,
            batch_size=args.distance_batch,
        )
        row: dict[str, Any] = {
            "generator": label,
            "sample_count": args.sample_count,
            **fixed_space_coverage_metrics(
                generated,
                real_features,
                generated_radii=generated_radii,
                reference_radii=real_radii,
                neighborhood=args.neighborhood,
                batch_size=args.distance_batch,
            ),
            "feature_effective_rank": covariance_effective_rank(generated_cpu),
            "feature_effective_rank_over_real": covariance_effective_rank(generated_cpu)
            / real_effective_rank,
            "nn_radius_median_over_real": float(
                generated_radii.median() / real_radii.median().clamp_min(1e-30)
            ),
            "exact_duplicate_fraction": exact_duplicate_fraction(paths),
            **class_occupancy(logits),
            **quantile_fields("nn_radius", generated_radii),
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        del generated, generated_radii
        torch.cuda.empty_cache()

    with (output_root / "coverage.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "protocol": "advfd_pmf_fixed_inception_mode_coverage_v1",
        "sample_count": args.sample_count,
        "neighborhood": args.neighborhood,
        "seed": args.seed,
        "real_features": str(args.real_features.expanduser().resolve()),
        "real_logits": str(args.real_logits.expanduser().resolve()),
        "real_feature_effective_rank": real_effective_rank,
        "real_class_occupancy": class_occupancy(real_logits),
        "real_nn_radius": real_radius_fields,
        "interpretation_boundary": (
            "These fixed-representation precision/recall, density/coverage, class-occupancy, "
            "and duplicate diagnostics test generator coverage. Critic rank collapse or "
            "cross-play interaction alone is not generator mode collapse."
        ),
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
