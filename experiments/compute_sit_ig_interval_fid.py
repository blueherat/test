"""Compute exact FID from cached ADM Inception features with PyTorch/MKL."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROTOCOL = "sit_ig_interval_adm_fid_v2"
FEATURE_PROTOCOL = "sit_ig_interval_adm_features_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-threads", type=int, default=16)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def symmetric_psd_sqrt(matrix: torch.Tensor) -> torch.Tensor:
    matrix = 0.5 * (matrix + matrix.T)
    values, vectors = torch.linalg.eigh(matrix)
    values = values.clamp_min(0.0)
    return (vectors * values.sqrt().unsqueeze(0)) @ vectors.T


def frechet_from_activations(
    activations: np.ndarray | torch.Tensor,
    reference_mean: np.ndarray | torch.Tensor,
    reference_covariance: np.ndarray | torch.Tensor,
    *,
    device: torch.device | str = "cpu",
    reference_sqrt: torch.Tensor | None = None,
) -> float:
    device = torch.device(device)
    if isinstance(activations, torch.Tensor):
        features = activations.to(device=device, dtype=torch.float64)
    else:
        features = torch.from_numpy(
            np.array(activations, dtype=np.float64, copy=True)
        ).to(device=device)
    mean_ref = torch.as_tensor(reference_mean, dtype=torch.float64, device=device)
    covariance_ref = torch.as_tensor(
        reference_covariance, dtype=torch.float64, device=device
    )
    if features.ndim != 2 or features.shape[1] != mean_ref.numel():
        raise ValueError("activation and reference dimensions must align")
    sample_mean = features.mean(dim=0)
    centered = features - sample_mean
    denominator = max(features.shape[0] - 1, 1)
    sample_trace = centered.square().sum() / denominator
    mean_term = (sample_mean - mean_ref).square().sum()
    dimension = features.shape[1]
    if features.shape[0] < dimension:
        kernel = centered @ covariance_ref @ centered.T / denominator
        cross_values = torch.linalg.eigvalsh(0.5 * (kernel + kernel.T))
    else:
        sample_covariance = centered.T @ centered / denominator
        if reference_sqrt is None:
            reference_sqrt = symmetric_psd_sqrt(covariance_ref)
        middle = reference_sqrt @ sample_covariance @ reference_sqrt
        cross_values = torch.linalg.eigvalsh(0.5 * (middle + middle.T))
    cross_trace = cross_values.clamp_min(0.0).sqrt().sum()
    value = mean_term + sample_trace + torch.trace(covariance_ref) - 2 * cross_trace
    return float(value.clamp_min(0.0).cpu())


def main() -> None:
    args = parse_args()
    if args.num_threads <= 0:
        raise ValueError("num threads must be positive")
    torch.set_num_threads(args.num_threads)
    feature_dir = args.feature_dir.expanduser().resolve()
    feature_manifest = json.loads(
        (feature_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if feature_manifest.get("status") != "complete":
        raise RuntimeError("ADM feature extraction is incomplete")
    if feature_manifest.get("protocol") != FEATURE_PROTOCOL:
        raise RuntimeError("unexpected ADM feature protocol")
    conditions = [str(value) for value in feature_manifest["conditions"]]
    reference_cache = Path(feature_manifest["reference_stats_cache"])
    with np.load(reference_cache) as reference:
        reference_mean = np.asarray(reference["mu"], dtype=np.float64)
        reference_covariance = np.asarray(reference["sigma"], dtype=np.float64)
    device = torch.device(args.device)
    reference_covariance_tensor = torch.as_tensor(
        reference_covariance, dtype=torch.float64, device=device
    )
    sample_count = int(feature_manifest["sample_count"])
    reference_sqrt = (
        symmetric_psd_sqrt(reference_covariance_tensor)
        if sample_count >= reference_mean.size
        else None
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in conditions:
        record = json.loads((feature_dir / f"{name}.json").read_text(encoding="utf-8"))
        metric_path = output_dir / f"{name}.json"
        if metric_path.is_file():
            metric = json.loads(metric_path.read_text(encoding="utf-8"))
        else:
            activations = np.load(record["activations"], mmap_mode="r")
            metric = {
                "condition": name,
                "samples": record["samples"],
                "activations": record["activations"],
                "fid": frechet_from_activations(
                    activations,
                    reference_mean,
                    reference_covariance,
                    device=device,
                    reference_sqrt=reference_sqrt,
                ),
                "inception_score": float(record["inception_score"]),
            }
            atomic_json(metric_path, metric)
        rows.append(metric)
        print(json.dumps(metric, ensure_ascii=False), flush=True)
    rows.sort(key=lambda row: float(row["fid"]))
    with (output_dir / "interval_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(
        output_dir / "manifest.json",
        {
            "protocol": PROTOCOL,
            "status": "complete",
            "feature_dir": str(feature_dir),
            "reference_stats_cache": str(reference_cache),
            "sample_count": sample_count,
            "conditions": conditions,
            "device": str(device),
            "num_threads": args.num_threads,
            "fid_backend": "ADM Inception pool_3 + exact torch symmetric eigendecomposition",
            "scope": feature_manifest["scope"],
        },
    )
    print(json.dumps(rows, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
