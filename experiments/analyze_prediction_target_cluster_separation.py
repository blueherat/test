#!/usr/bin/env python3
"""Audit cluster separation without conflating it with global distribution fit."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from experiments.run_prediction_target_bayes_oracle_v5 import (
    TangentGaussianMixture,
)


EPS = 1e-12


def component_statistics(
    mixture: TangentGaussianMixture,
    samples: np.ndarray,
    *,
    batch_size: int = 512,
) -> dict[str, np.ndarray]:
    """Return exact mixture likelihood and component-assignment statistics."""
    nll_parts: list[np.ndarray] = []
    entropy_parts: list[np.ndarray] = []
    max_resp_parts: list[np.ndarray] = []
    margin_parts: list[np.ndarray] = []
    assignment_parts: list[np.ndarray] = []
    mahalanobis_parts: list[np.ndarray] = []

    tangent_var = mixture.sigma_tangent**2
    normal_var = mixture.sigma_normal**2
    if normal_var <= 0.0:
        raise ValueError("exact likelihood audit requires sigma_normal > 0")
    log_det = 2.0 * math.log(tangent_var) + float(mixture.D - 2) * math.log(
        normal_var
    )
    normalizer = float(mixture.D) * math.log(2.0 * math.pi)

    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            x = torch.from_numpy(samples[start : start + batch_size]).to(
                mixture.device, dtype=torch.float32
            )
            residual = x[:, None, :] - mixture.means[None]
            tangent_coeff = torch.einsum(
                "bkd,kdr->bkr", residual, mixture.bases
            )
            tangent_sq = tangent_coeff.square().sum(dim=2)
            residual_sq = residual.square().sum(dim=2)
            normal_sq = (residual_sq - tangent_sq).clamp_min(0.0)
            mahalanobis = tangent_sq / tangent_var + normal_sq / normal_var
            log_component = -0.5 * (mahalanobis + log_det + normalizer)
            log_joint = log_component + mixture.log_weights[None]
            log_prob = torch.logsumexp(log_joint, dim=1)
            responsibilities = torch.softmax(log_joint, dim=1)
            top2 = torch.topk(log_joint, k=2, dim=1).values
            assignment = responsibilities.argmax(dim=1)
            assigned_mahalanobis = mahalanobis.gather(
                1, assignment[:, None]
            ).squeeze(1)

            nll_parts.append((-log_prob).cpu().numpy())
            entropy_parts.append(
                (
                    -(
                        responsibilities
                        * responsibilities.clamp_min(EPS).log()
                    ).sum(dim=1)
                    / math.log(float(mixture.components))
                )
                .cpu()
                .numpy()
            )
            max_resp_parts.append(responsibilities.max(dim=1).values.cpu().numpy())
            margin_parts.append((top2[:, 0] - top2[:, 1]).cpu().numpy())
            assignment_parts.append(assignment.cpu().numpy())
            mahalanobis_parts.append(assigned_mahalanobis.cpu().numpy())

    return {
        "nll": np.concatenate(nll_parts),
        "normalized_entropy": np.concatenate(entropy_parts),
        "max_responsibility": np.concatenate(max_resp_parts),
        "top2_log_margin": np.concatenate(margin_parts),
        "assignment": np.concatenate(assignment_parts),
        "assigned_mahalanobis": np.concatenate(mahalanobis_parts),
    }


def intrinsic_projection(
    mixture: TangentGaussianMixture,
    samples: np.ndarray,
    *,
    batch_size: int = 1024,
) -> np.ndarray:
    parts = []
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            x = torch.from_numpy(samples[start : start + batch_size]).to(
                mixture.device, dtype=torch.float32
            )
            parts.append(mixture.intrinsic_readout(x).cpu().numpy())
    return np.concatenate(parts)


def kde_density(
    samples: np.ndarray, queries: np.ndarray, bandwidth: float
) -> np.ndarray:
    squared_distance = (
        (samples[:, None, :] - queries[None, :, :]) ** 2
    ).sum(axis=2)
    return np.exp(-0.5 * squared_distance / bandwidth**2).mean(axis=0)


def jensen_shannon_counts(a: np.ndarray, b: np.ndarray) -> float:
    p = a.astype(np.float64) / max(float(a.sum()), EPS)
    q = b.astype(np.float64) / max(float(b.sum()), EPS)
    midpoint = 0.5 * (p + q)

    def kl(left: np.ndarray, right: np.ndarray) -> float:
        mask = left > 0.0
        return float(np.sum(left[mask] * np.log(left[mask] / right[mask])))

    return 0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint)


def condition_sort_key(name: str) -> tuple[int, float]:
    if name == "reference":
        return (0, 0.0)
    if name == "bayes":
        return (1, 0.0)
    if name == "x":
        return (2, 0.0)
    if name == "v":
        return (3, 0.0)
    match = re.fullmatch(r"xv_g(.+)", name)
    if match:
        value = match.group(1).replace("m", "-").replace("p", ".")
        return (4, float(value))
    return (5, 0.0)


def display_name(name: str) -> str:
    if name.startswith("xv_g"):
        gamma = condition_sort_key(name)[1]
        return f"x+{gamma:g}(x-v)"
    return name


def load_samples(sample_dir: Path, source_seed_dir: Path) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "reference": np.load(source_seed_dir / "common" / "reference.npy"),
        "bayes": np.load(source_seed_dir / "common" / "bayes.npy"),
    }
    for path in sample_dir.glob("samples_*.npy"):
        name = path.stem.removeprefix("samples_")
        if name in {"x", "v"} or name.startswith("xv_g"):
            arrays[name] = np.load(path)
    return dict(sorted(arrays.items(), key=lambda item: condition_sort_key(item[0])))


def build_mixture(manifest: dict, device: torch.device) -> TangentGaussianMixture:
    return TangentGaussianMixture(
        D=int(manifest["D"]),
        components=int(manifest["components"]),
        curvature=float(manifest["curvature"]),
        frequency_scale=float(manifest["frequency_scale"]),
        center_rms=float(manifest["center_rms"]),
        sigma_tangent=float(manifest["sigma_tangent"]),
        sigma_normal=float(manifest["sigma_normal"]),
        seed=int(manifest["mixture_seed"]),
        device=device,
    )


def audit(
    arrays: dict[str, np.ndarray], mixture: TangentGaussianMixture
) -> tuple[pd.DataFrame, dict[str, float]]:
    raw = {
        name: component_statistics(mixture, samples)
        for name, samples in arrays.items()
    }
    intrinsic = {
        name: intrinsic_projection(mixture, samples)
        for name, samples in arrays.items()
    }
    centers = mixture.intrinsic_centers.float().cpu().numpy()
    intrinsic_distance = {
        name: np.linalg.norm(
            points[:, None, :] - centers[None, :, :], axis=2
        ).min(axis=1)
        for name, points in intrinsic.items()
    }
    kde_bandwidth = float(np.median(intrinsic_distance["reference"]))
    midpoints = 0.5 * (centers[:-1] + centers[1:])
    reference = raw["reference"]
    thresholds = {
        "reference_nll_p95": float(np.quantile(reference["nll"], 0.95)),
        "reference_max_responsibility_p05": float(
            np.quantile(reference["max_responsibility"], 0.05)
        ),
        "reference_mahalanobis_p95": float(
            np.quantile(reference["assigned_mahalanobis"], 0.95)
        ),
        "reference_intrinsic_center_distance_p95": float(
            np.quantile(intrinsic_distance["reference"], 0.95)
        ),
        "intrinsic_kde_bandwidth": kde_bandwidth,
    }
    reference_counts = np.bincount(
        reference["assignment"], minlength=mixture.components
    )
    rows = []
    for name, stats in raw.items():
        confident = (
            stats["max_responsibility"]
            >= thresholds["reference_max_responsibility_p05"]
        )
        inlier = stats["nll"] <= thresholds["reference_nll_p95"]
        mahalanobis_inlier = (
            stats["assigned_mahalanobis"]
            <= thresholds["reference_mahalanobis_p95"]
        )
        counts = np.bincount(stats["assignment"], minlength=mixture.components)
        center_density = kde_density(intrinsic[name], centers, kde_bandwidth)
        midpoint_density = kde_density(intrinsic[name], midpoints, kde_bandwidth)
        adjacent_peak_density = 0.5 * (
            center_density[:-1] + center_density[1:]
        )
        log_density_contrast = np.log(adjacent_peak_density + EPS) - np.log(
            midpoint_density + EPS
        )
        rows.append(
            {
                "condition": name,
                "label": display_name(name),
                "samples": len(stats["nll"]),
                "mean_nll": float(np.mean(stats["nll"])),
                "median_nll": float(np.median(stats["nll"])),
                "mean_normalized_assignment_entropy": float(
                    np.mean(stats["normalized_entropy"])
                ),
                "mean_max_responsibility": float(
                    np.mean(stats["max_responsibility"])
                ),
                "mean_top2_log_margin": float(
                    np.mean(stats["top2_log_margin"])
                ),
                "bridge_rate": float(np.mean(~confident)),
                "nll_outlier_rate": float(np.mean(~inlier)),
                "mahalanobis_outlier_rate": float(np.mean(~mahalanobis_inlier)),
                "mean_intrinsic_center_distance": float(
                    np.mean(intrinsic_distance[name])
                ),
                "intrinsic_bridge_rate": float(
                    np.mean(
                        intrinsic_distance[name]
                        > thresholds[
                            "reference_intrinsic_center_distance_p95"
                        ]
                    )
                ),
                "mean_adjacent_log_density_contrast": float(
                    np.mean(log_density_contrast)
                ),
                "confident_inlier_rate": float(np.mean(confident & inlier)),
                "component_jsd": jensen_shannon_counts(reference_counts, counts),
                "occupied_components": int(np.count_nonzero(counts)),
            }
        )
    frame = pd.DataFrame(rows)
    reference_row = frame.loc[frame.condition == "reference"].iloc[0]
    frame["nll_excess_vs_reference"] = frame.mean_nll - float(
        reference_row.mean_nll
    )
    frame["absolute_nll_gap_vs_reference"] = frame[
        "nll_excess_vs_reference"
    ].abs()
    frame["entropy_gap_vs_reference"] = (
        frame.mean_normalized_assignment_entropy
        - float(reference_row.mean_normalized_assignment_entropy)
    )
    frame["absolute_intrinsic_distance_gap_vs_reference"] = (
        frame.mean_intrinsic_center_distance
        - float(reference_row.mean_intrinsic_center_distance)
    ).abs()
    return frame, thresholds


def plot_audit(frame: pd.DataFrame, path: Path) -> None:
    labels = frame.label.tolist()
    x = np.arange(len(frame))
    metrics = [
        (
            "absolute_nll_gap_vs_reference",
            "Absolute mean-NLL gap vs reference",
            False,
        ),
        ("bridge_rate", "Low-confidence bridge rate", False),
        (
            "absolute_intrinsic_distance_gap_vs_reference",
            "Absolute 2-D center-distance gap",
            False,
        ),
        ("intrinsic_bridge_rate", "2-D bridge rate", False),
        ("nll_outlier_rate", "Reference-calibrated outlier rate", False),
        ("confident_inlier_rate", "Confident inlier rate", True),
        ("component_jsd", "Component occupancy JSD", False),
        (
            "mean_adjacent_log_density_contrast",
            "2-D adjacent peak/valley log-contrast",
            True,
        ),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(22, 9), constrained_layout=True)
    colors = ["#4c78a8", "#59a14f", "#f28e2b", "#e15759"] + [
        "#b07aa1"
    ] * max(0, len(frame) - 4)
    for axis, (column, title, higher_is_sharper) in zip(axes.flat, metrics):
        values = frame[column].to_numpy()
        axis.bar(x, values, color=colors[: len(frame)])
        axis.set_title(title)
        axis.set_xticks(x, labels, rotation=45, ha="right")
        axis.grid(axis="y", alpha=0.25)
        if higher_is_sharper:
            axis.set_ylabel("higher = sharper/more confident")
        else:
            axis.set_ylabel("lower = closer/cleaner")
    fig.suptitle(
        "Cluster separation and distribution fidelity are different axes",
        fontsize=16,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--source-seed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    manifest = json.loads(
        (args.source_seed_dir / "manifest.json").read_text(encoding="utf-8")
    )
    mixture = build_mixture(manifest, torch.device(args.device))
    arrays = load_samples(args.sample_dir, args.source_seed_dir)
    frame, thresholds = audit(arrays, mixture)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "cluster_separation_metrics.csv", index=False)
    (args.output_dir / "cluster_separation_thresholds.json").write_text(
        json.dumps(thresholds, indent=2) + "\n", encoding="utf-8"
    )
    plot_audit(frame, args.output_dir / "cluster_separation_audit.png")


if __name__ == "__main__":
    main()
