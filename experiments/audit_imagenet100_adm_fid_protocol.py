#!/usr/bin/env python3
"""Audit finite-sample and class-prior effects in ImageNet-100 ADM FID.

This script consumes already extracted ADM ``pool_3`` activations. It reports:

* ordinary plug-in FID against the full validation reference;
* FID after reweighting generated activations to an exactly uniform class prior;
* exact unbiased polynomial-kernel KID on the same activations;
* a stratified real-vs-real split-half calibration value;
* consistency of cached reference statistics with the retained activations.

The calculations do not alter the canonical ADM metric. They expose when an
FID-1K screen is moving because of estimator bias or a noisy sampled class
prior rather than because of the compared guidance field.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import linalg


@dataclass(frozen=True)
class ConditionInput:
    name: str
    activations: Path
    labels: Path


def parse_condition(value: str) -> ConditionInput:
    fields = value.split("=", 2)
    if len(fields) != 3 or not all(fields):
        raise argparse.ArgumentTypeError(
            "condition must be NAME=ACTIVATIONS_NPZ=LABELS_NPY"
        )
    return ConditionInput(fields[0], Path(fields[1]), Path(fields[2]))


def mean_covariance(
    features: np.ndarray, weights: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("features must be a two-dimensional array with N >= 2")
    if weights is None:
        return values.mean(axis=0), np.cov(values, rowvar=False)

    normalized = np.asarray(weights, dtype=np.float64)
    if normalized.shape != (len(values),):
        raise ValueError("weights must have one entry per feature")
    if not np.isfinite(normalized).all() or np.any(normalized <= 0.0):
        raise ValueError("weights must be positive and finite")
    normalized = normalized / normalized.sum()
    mean = normalized @ values
    centered = values - mean
    denominator = 1.0 - float(np.square(normalized).sum())
    if denominator <= 0.0:
        raise ValueError("effective weighted sample count must exceed one")
    covariance = (centered * normalized[:, None]).T @ centered / denominator
    return mean, covariance


def uniform_class_weights(labels: np.ndarray, num_classes: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError("labels lie outside the declared class range")
    counts = np.bincount(labels, minlength=num_classes)
    if np.any(counts == 0):
        raise ValueError("uniform class reweighting requires every class to appear")
    return 1.0 / (num_classes * counts[labels])


def frechet_distance(
    first_mean: np.ndarray,
    first_covariance: np.ndarray,
    second_mean: np.ndarray,
    second_covariance: np.ndarray,
    *,
    eps: float = 1e-6,
) -> float:
    difference = np.atleast_1d(first_mean) - np.atleast_1d(second_mean)
    first_covariance = np.atleast_2d(first_covariance)
    second_covariance = np.atleast_2d(second_covariance)
    covariance_mean, _ = linalg.sqrtm(
        first_covariance.dot(second_covariance), disp=False
    )
    if not np.isfinite(covariance_mean).all():
        offset = np.eye(first_covariance.shape[0]) * eps
        covariance_mean = linalg.sqrtm(
            (first_covariance + offset).dot(second_covariance + offset)
        )
    if np.iscomplexobj(covariance_mean):
        if not np.allclose(np.diagonal(covariance_mean).imag, 0.0, atol=1e-3):
            raise ValueError("FID covariance square root has a large imaginary part")
        covariance_mean = covariance_mean.real
    return float(
        difference.dot(difference)
        + np.trace(first_covariance)
        + np.trace(second_covariance)
        - 2.0 * np.trace(covariance_mean)
    )


def _polynomial_kernel_sum(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    block_size: int,
    exclude_diagonal: bool,
) -> float:
    if exclude_diagonal and (
        first.data_ptr() != second.data_ptr() or first.shape != second.shape
    ):
        raise ValueError("diagonal exclusion requires the same tensor")
    feature_dim = first.shape[1]
    total = torch.zeros((), dtype=torch.float64, device=first.device)
    for row_start in range(0, len(first), block_size):
        row_stop = min(row_start + block_size, len(first))
        row = first[row_start:row_stop]
        for col_start in range(0, len(second), block_size):
            col_stop = min(col_start + block_size, len(second))
            col = second[col_start:col_stop]
            kernel = (row @ col.T / feature_dim + 1.0).pow_(3)
            block_sum = kernel.sum(dtype=torch.float64)
            if exclude_diagonal and row_start == col_start:
                block_sum = block_sum - kernel.diagonal().sum(dtype=torch.float64)
            total = total + block_sum
    return float(total.item())


def unbiased_kid(
    first: np.ndarray,
    second: np.ndarray,
    *,
    device: torch.device,
    block_size: int = 1024,
) -> float:
    """Exact unbiased KID using k(x,y)=(x^T y / d + 1)^3."""

    first_tensor = torch.as_tensor(first, dtype=torch.float32, device=device)
    second_tensor = torch.as_tensor(second, dtype=torch.float32, device=device)
    if len(first_tensor) < 2 or len(second_tensor) < 2:
        raise ValueError("KID requires at least two samples from each distribution")
    first_sum = _polynomial_kernel_sum(
        first_tensor,
        first_tensor,
        block_size=block_size,
        exclude_diagonal=True,
    )
    second_sum = _polynomial_kernel_sum(
        second_tensor,
        second_tensor,
        block_size=block_size,
        exclude_diagonal=True,
    )
    cross_sum = _polynomial_kernel_sum(
        first_tensor,
        second_tensor,
        block_size=block_size,
        exclude_diagonal=False,
    )
    return float(
        first_sum / (len(first_tensor) * (len(first_tensor) - 1))
        + second_sum / (len(second_tensor) * (len(second_tensor) - 1))
        - 2.0 * cross_sum / (len(first_tensor) * len(second_tensor))
    )


def stratified_split_indices(
    labels: np.ndarray, num_classes: int
) -> tuple[np.ndarray, np.ndarray]:
    first: list[int] = []
    second: list[int] = []
    for class_index in range(num_classes):
        indices = np.flatnonzero(labels == class_index)
        if len(indices) < 2:
            raise ValueError("each reference class needs at least two samples")
        midpoint = len(indices) // 2
        first.extend(indices[:midpoint].tolist())
        second.extend(indices[midpoint : 2 * midpoint].tolist())
    return np.asarray(first), np.asarray(second)


def load_pool(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        if "pool_3" not in payload.files:
            raise KeyError(f"{path} has no pool_3 activation array")
        values = np.asarray(payload["pool_3"], dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 2048:
        raise ValueError(f"unexpected ADM pool_3 shape: {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite activations in {path}")
    return values


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    reference = load_pool(args.reference_activations)
    reference_labels = np.load(args.reference_labels, allow_pickle=False).astype(
        np.int64, copy=False
    )
    if len(reference_labels) != len(reference):
        raise ValueError("reference labels and activations have different lengths")
    reference_mean, reference_covariance = mean_covariance(reference)

    with np.load(args.reference_stats, allow_pickle=False) as payload:
        cached_mean = np.asarray(payload["mu"])
        cached_covariance = np.asarray(payload["sigma"])
    cache_fid_delta_rows: list[dict[str, float]] = []
    rows: list[dict[str, Any]] = []
    for condition in args.conditions:
        features = load_pool(condition.activations)
        labels = np.load(condition.labels, allow_pickle=False).astype(
            np.int64, copy=False
        )
        if len(labels) != len(features):
            raise ValueError(f"label count mismatch for {condition.name}")
        raw_mean, raw_covariance = mean_covariance(features)
        weights = uniform_class_weights(labels, args.num_classes)
        balanced_mean, balanced_covariance = mean_covariance(features, weights)
        raw_fid = frechet_distance(
            raw_mean, raw_covariance, reference_mean, reference_covariance
        )
        cached_fid = frechet_distance(
            raw_mean, raw_covariance, cached_mean, cached_covariance
        )
        balanced_fid = frechet_distance(
            balanced_mean,
            balanced_covariance,
            reference_mean,
            reference_covariance,
        )
        counts = np.bincount(labels, minlength=args.num_classes)
        rows.append(
            {
                "condition": condition.name,
                "sample_count": len(features),
                "metric_name": f"ADM-FID-{len(features)}",
                "fid_recomputed_reference": raw_fid,
                "fid_cached_reference": cached_fid,
                "cached_reference_delta": cached_fid - raw_fid,
                "uniform_class_weighted_fid": balanced_fid,
                "class_prior_delta": balanced_fid - raw_fid,
                "unbiased_kid": unbiased_kid(
                    features,
                    reference,
                    device=device,
                    block_size=args.kid_block_size,
                ),
                "class_count_min": int(counts.min()),
                "class_count_max": int(counts.max()),
                "class_count_std": float(counts.std()),
            }
        )
        cache_fid_delta_rows.append(
            {
                "raw": raw_fid,
                "cached": cached_fid,
                "delta": cached_fid - raw_fid,
            }
        )

    first_indices, second_indices = stratified_split_indices(
        reference_labels, args.num_classes
    )
    first_mean, first_covariance = mean_covariance(reference[first_indices])
    second_mean, second_covariance = mean_covariance(reference[second_indices])
    real_split_fid = frechet_distance(
        first_mean, first_covariance, second_mean, second_covariance
    )
    real_split_kid = unbiased_kid(
        reference[first_indices],
        reference[second_indices],
        device=device,
        block_size=args.kid_block_size,
    )

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    table_path = output / "fid_protocol_audit.csv"
    write_csv(table_path, rows)
    summary = {
        "format": "eqvae_imagenet100_adm_fid_protocol_audit_v1",
        "reference": {
            "activations": str(args.reference_activations),
            "statistics": str(args.reference_stats),
            "labels": str(args.reference_labels),
            "count": len(reference),
            "class_count": args.num_classes,
            "cached_mean_max_abs_error": float(
                np.max(np.abs(reference_mean - cached_mean))
            ),
            "cached_covariance_max_abs_error": float(
                np.max(np.abs(reference_covariance - cached_covariance))
            ),
        },
        "real_stratified_split_half": {
            "count_per_half": len(first_indices),
            "fid": real_split_fid,
            "unbiased_kid": real_split_kid,
        },
        "conditions": rows,
        "interpretation_guardrails": [
            "FID-N values with different N are not directly comparable.",
            "The plug-in FID estimator is biased at finite sample count.",
            "Uniform class weighting diagnoses sampled class-prior noise; it is not the canonical ADM FID.",
            "Unbiased KID is reported as a secondary feature-distribution check, not a replacement benchmark.",
        ],
        "table": str(table_path),
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-activations", type=Path, required=True)
    parser.add_argument("--reference-stats", type=Path, required=True)
    parser.add_argument("--reference-labels", type=Path, required=True)
    parser.add_argument(
        "--condition", dest="conditions", action="append", type=parse_condition, required=True
    )
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--kid-block-size", type=int, default=1024)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
