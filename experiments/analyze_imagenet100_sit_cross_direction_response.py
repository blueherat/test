#!/usr/bin/env python3
"""Compare endpoint responses induced by x400 and v270 guidance directions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from experiments.analyze_imagenet100_sit_finite_guidance_features import (
        _gamma_key,
        load_trajectory_shards,
    )
except ModuleNotFoundError:
    from analyze_imagenet100_sit_finite_guidance_features import (
        _gamma_key,
        load_trajectory_shards,
    )


ROOT = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/finite_guidance_400k_mechanism"
)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_flat = left.double().flatten(1)
    right_flat = right.double().flatten(1)
    return (left_flat * right_flat).sum(1) / (
        left_flat.norm(dim=1) * right_flat.norm(dim=1)
    ).clamp_min(torch.finfo(torch.float64).tiny)


def _vector_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.double().flatten()
    right = right.double().flatten()
    return float(
        (left @ right)
        / (left.norm() * right.norm()).clamp_min(torch.finfo(torch.float64).tiny)
    )


def distribution_response_metrics(
    baseline: torch.Tensor,
    x_value: torch.Tensor,
    v_value: torch.Tensor,
) -> dict[str, float]:
    """Compare mean/covariance shifts from one paired baseline distribution."""

    if baseline.shape != x_value.shape or baseline.shape != v_value.shape:
        raise ValueError("all feature arrays must have the same shape")
    if baseline.ndim != 2 or len(baseline) < 2:
        raise ValueError("features must have shape [N,D] with N >= 2")

    def moments(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value = value.double()
        mean = value.mean(dim=0)
        centered = value - mean
        covariance = centered.T @ centered / (len(value) - 1)
        return mean, covariance

    baseline_mean, baseline_covariance = moments(baseline)
    x_mean, x_covariance = moments(x_value)
    v_mean, v_covariance = moments(v_value)
    x_mean_shift = x_mean - baseline_mean
    v_mean_shift = v_mean - baseline_mean
    x_covariance_shift = x_covariance - baseline_covariance
    v_covariance_shift = v_covariance - baseline_covariance
    mean_dot = x_mean_shift.flatten() @ v_mean_shift.flatten()
    covariance_dot = x_covariance_shift.flatten() @ v_covariance_shift.flatten()
    x_mean_norm = x_mean_shift.norm()
    v_mean_norm = v_mean_shift.norm()
    x_covariance_norm = x_covariance_shift.norm()
    v_covariance_norm = v_covariance_shift.norm()
    tiny = torch.finfo(torch.float64).tiny
    return {
        "mean_shift_cosine": float(
            mean_dot / (x_mean_norm * v_mean_norm).clamp_min(tiny)
        ),
        "covariance_shift_cosine": float(
            covariance_dot / (x_covariance_norm * v_covariance_norm).clamp_min(tiny)
        ),
        "joint_moment_shift_cosine": float(
            (mean_dot + covariance_dot)
            / (
                torch.sqrt(x_mean_norm.square() + x_covariance_norm.square())
                * torch.sqrt(v_mean_norm.square() + v_covariance_norm.square())
            ).clamp_min(tiny)
        ),
        "x_mean_shift_norm": float(x_mean_norm),
        "v_mean_shift_norm": float(v_mean_norm),
        "x_covariance_shift_norm": float(x_covariance_norm),
        "v_covariance_shift_norm": float(v_covariance_norm),
    }


def _load_feature_npz(path: Path) -> tuple[dict[str, torch.Tensor], np.ndarray]:
    with np.load(path) as payload:
        names = payload["names"].tolist()
        values = payload["features"]
        labels = payload["labels"].copy()
    if len(names) != len(set(names)) or len(names) != len(values):
        raise ValueError(f"invalid condition names in {path}")
    return {
        str(name): torch.from_numpy(value.copy())
        for name, value in zip(names, values, strict=True)
    }, labels


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(args: argparse.Namespace) -> None:
    x_payload, x_manifest = load_trajectory_shards(args.x_run)
    v_payload, v_manifest = load_trajectory_shards(args.v_run)
    if x_manifest["study"] != "linearity" or v_manifest["study"] != "linearity":
        raise ValueError("both inputs must be completed linearity runs")
    for key in ("num_samples", "seed", "noise_sha256", "labels_sha256", "heun_steps"):
        if x_manifest[key] != v_manifest[key]:
            raise ValueError(f"paired manifests differ at {key}")
    if not torch.equal(x_payload["labels"], v_payload["labels"]):
        raise ValueError("labels differ across directions")
    torch.testing.assert_close(x_payload["baseline"], v_payload["baseline"], rtol=0, atol=0)

    x_features, x_labels = _load_feature_npz(args.x_features)
    v_features, v_labels = _load_feature_npz(args.v_features)
    if not np.array_equal(x_labels, v_labels):
        raise ValueError("decoded feature labels differ")
    torch.testing.assert_close(x_features["baseline"], v_features["baseline"], rtol=0, atol=0)

    x_gammas = [float(value) for value in x_payload["gammas"].tolist()]
    v_gammas = [float(value) for value in v_payload["gammas"].tolist()]
    common = sorted(set(x_gammas).intersection(v_gammas))
    x_endpoints = {
        gamma: x_payload["endpoints"][x_gammas.index(gamma)] for gamma in common
    }
    v_endpoints = {
        gamma: v_payload["endpoints"][v_gammas.index(gamma)] for gamma in common
    }
    rows: list[dict[str, object]] = []
    for gamma in common:
        if gamma <= 0:
            continue
        x_latent_response = x_endpoints[gamma] - x_payload["baseline"]
        v_latent_response = v_endpoints[gamma] - v_payload["baseline"]
        feature_key = f"closed_{_gamma_key(gamma)}"
        x_feature_response = x_features[feature_key] - x_features["baseline"]
        v_feature_response = v_features[feature_key] - v_features["baseline"]
        latent_cosine = _cosine(x_latent_response, v_latent_response)
        feature_cosine = _cosine(x_feature_response, v_feature_response)
        moment_metrics = distribution_response_metrics(
            x_features["baseline"],
            x_features[feature_key],
            v_features[feature_key],
        )
        rows.append(
            {
                "gamma": gamma,
                "latent_paired_response_cosine_mean": float(latent_cosine.mean()),
                "latent_paired_response_cosine_median": float(latent_cosine.median()),
                "feature_paired_response_cosine_mean": float(feature_cosine.mean()),
                "feature_paired_response_cosine_median": float(feature_cosine.median()),
                "latent_mean_response_cosine": _vector_cosine(
                    x_latent_response.mean(0), v_latent_response.mean(0)
                ),
                "feature_mean_response_cosine": _vector_cosine(
                    x_feature_response.mean(0), v_feature_response.mean(0)
                ),
                **moment_metrics,
            }
        )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, output_dir / "cross_direction_response.csv")
    figure, axis = plt.subplots(figsize=(9.5, 5.8))
    gammas = [float(row["gamma"]) for row in rows]
    for key, label in (
        ("latent_paired_response_cosine_mean", "paired latent response"),
        ("feature_paired_response_cosine_mean", "paired decoded feature response"),
        ("mean_shift_cosine", "distribution mean shift"),
        ("covariance_shift_cosine", "distribution covariance shift"),
    ):
        axis.plot(gammas, [float(row[key]) for row in rows], marker="o", label=label)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("guidance scale gamma")
    axis.set_ylabel("cosine between x400 and v270 endpoint responses")
    axis.set_ylim(-1.05, 1.05)
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_dir / "cross_direction_response.png", dpi=180)
    plt.close(figure)
    summary = {
        "format": "eqvae_sit400_cross_direction_functional_response_v1",
        "scope": "paired N=128 mechanism features; not ADM FID",
        "x_run": str(args.x_run.resolve()),
        "v_run": str(args.v_run.resolve()),
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    x_run = ROOT / "linearity/x400/n128_seed20260814"
    v_run = ROOT / "linearity/v270/n128_seed20260814"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x-run", type=Path, default=x_run)
    parser.add_argument("--v-run", type=Path, default=v_run)
    parser.add_argument(
        "--x-features",
        type=Path,
        default=x_run / "decoded_feature_response/continuous_inception_features.npz",
    )
    parser.add_argument(
        "--v-features",
        type=Path,
        default=v_run / "decoded_feature_response/continuous_inception_features.npz",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "cross_direction_response")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
