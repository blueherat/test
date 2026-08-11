#!/usr/bin/env python3
"""Evaluate guidance toy samples without rewarding oversharpening.

The target distribution is analytic, so a single image-quality-style score is
unnecessarily lossy.  This benchmark separates global distribution matching,
typical-set calibration, local width, mode coverage, and regional separation.
It also creates controlled failure cases to verify that the metrics react to
oversharpening, blur, mode dropping, and bridges in the expected way.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree, distance
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.analyze_guidance_sharpness_tradeoff import (
    adjacent_contrast_profile,
    summarize_contrast_regions,
)
from experiments.analyze_prediction_target_cluster_separation import (
    component_statistics,
    intrinsic_projection,
    jensen_shannon_counts,
)
from experiments.evaluate_prediction_target_autoguidance import (
    METHOD_COLORS,
    METHOD_LABELS,
    build_mixture,
    scale_sample_name,
)
from experiments.run_prediction_target_bayes_oracle_v5 import stable_seed


EPS = 1e-12


def parse_names(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        raise ValueError("at least one condition is required")
    return names


def deterministic_directions(count: int) -> np.ndarray:
    angles = np.linspace(0.0, math.pi, count, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=0)


def sliced_wasserstein_2d(
    reference: np.ndarray, sample: np.ndarray, *, projections: int = 512
) -> float:
    count = min(len(reference), len(sample))
    directions = deterministic_directions(projections)
    left = np.sort(reference[:count].astype(np.float64) @ directions, axis=0)
    right = np.sort(sample[:count].astype(np.float64) @ directions, axis=0)
    return float(np.mean(np.abs(left - right)))


def energy_distance_2d(
    reference: np.ndarray,
    sample: np.ndarray,
    *,
    limit: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    count = min(limit, len(reference), len(sample))
    left = reference[rng.choice(len(reference), count, replace=False)]
    right = sample[rng.choice(len(sample), count, replace=False)]
    # The V-statistic includes diagonal self-distances and is non-negative for
    # finite empirical measures.  Using ``pdist().mean()`` mixes in an
    # incompatible U-statistic and can make genuinely different samples look
    # exactly zero after clipping.
    cross = distance.cdist(left, right).mean()
    left_self = distance.cdist(left, left).mean()
    right_self = distance.cdist(right, right).mean()
    return float(max(0.0, 2.0 * cross - left_self - right_self))


def classifier_two_sample_auc(
    reference: np.ndarray,
    sample: np.ndarray,
    *,
    limit: int,
    seed: int,
) -> float:
    """Return held-out nonlinear C2ST AUC; 0.5 means indistinguishable."""
    rng = np.random.default_rng(seed)
    count = min(limit, len(reference), len(sample))
    left = reference[rng.choice(len(reference), count, replace=False)]
    right = sample[rng.choice(len(sample), count, replace=False)]
    features = np.concatenate([left, right], axis=0).astype(np.float64)
    labels = np.concatenate(
        [np.zeros(count, dtype=np.int64), np.ones(count, dtype=np.int64)]
    )
    # A local classifier is appropriate here because the important failures
    # are thin-manifold width changes and small bridges.  Calibrated controls
    # showed that a global random-feature probe missed both despite large NLL
    # and width errors.
    model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=1),
    )
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    scores = cross_val_predict(
        model,
        features,
        labels,
        cv=folds,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    auc = float(roc_auc_score(labels, scores))
    return max(auc, 1.0 - auc)


def manifold_metrics(
    reference: np.ndarray,
    sample: np.ndarray,
    *,
    k: int,
    limit: int,
    seed: int,
    chunk_size: int = 256,
) -> dict[str, float]:
    """Compute improved precision/recall and density/coverage in 2-D."""
    rng = np.random.default_rng(seed)
    count = min(limit, len(reference), len(sample))
    real = reference[rng.choice(len(reference), count, replace=False)]
    fake = sample[rng.choice(len(sample), count, replace=False)]
    real_tree = cKDTree(real)
    fake_tree = cKDTree(fake)
    real_radius = real_tree.query(real, k=k + 1)[0][:, -1]
    fake_radius = fake_tree.query(fake, k=k + 1)[0][:, -1]

    precision_hits = 0
    density_sum = 0.0
    for start in range(0, count, chunk_size):
        distances = distance.cdist(fake[start : start + chunk_size], real)
        inside = distances <= real_radius[None]
        precision_hits += int(inside.any(axis=1).sum())
        density_sum += float(inside.sum())

    recall_hits = 0
    for start in range(0, count, chunk_size):
        distances = distance.cdist(real[start : start + chunk_size], fake)
        recall_hits += int((distances <= fake_radius[None]).any(axis=1).sum())

    nearest_fake = fake_tree.query(real, k=1)[0]
    return {
        "precision": precision_hits / float(count),
        "recall": recall_hits / float(count),
        "density": density_sum / float(k * count),
        "coverage": float(np.mean(nearest_fake <= real_radius)),
    }


def component_widths(
    intrinsic: np.ndarray,
    assignment: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    values = np.full(len(centers), np.nan, dtype=np.float64)
    for component in range(len(centers)):
        selected = intrinsic[assignment == component]
        if len(selected) >= 4:
            residual = selected - centers[component]
            values[component] = math.sqrt(float(np.mean(np.square(residual).sum(axis=1))))
    return values


def region_means(values: np.ndarray) -> dict[str, float]:
    chunks = np.array_split(values, 3)
    return {
        "inner": float(np.nanmean(chunks[0])),
        "middle": float(np.nanmean(chunks[1])),
        "outer": float(np.nanmean(chunks[2])),
    }


def make_controls(
    mixture,
    *,
    count: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    generator = torch.Generator(device=mixture.device.type)
    generator.manual_seed(seed)
    base, labels = mixture.sample_clean(count, generator=generator)
    means = mixture.means[labels]
    residual = base - means
    tangent, normal = mixture.split_by_component(residual, labels)
    controls = {
        "control_sharp_0p7": means + 0.7 * residual,
        "control_normal_sharp_0p4": means + tangent + 0.4 * normal,
        "control_blur_1p3": means + 1.3 * residual,
    }

    bridge = base.clone()
    bridge_count = int(round(0.20 * count))
    bridge_index = torch.randperm(
        count, device=mixture.device, generator=generator
    )[:bridge_count]
    bridge_labels = labels[bridge_index]
    neighbor = torch.where(
        bridge_labels + 1 < mixture.components,
        bridge_labels + 1,
        bridge_labels - 1,
    )
    midpoint = 0.5 * (
        mixture.means[bridge_labels] + mixture.means[neighbor]
    )
    bridge[bridge_index] = midpoint + residual[bridge_index]
    controls["control_bridge_20pct"] = bridge

    pool, pool_labels = mixture.sample_clean(count * 3, generator=generator)
    retained = pool_labels.remainder(4) != 0
    controls["control_drop_8_modes"] = pool[retained][:count]
    if len(controls["control_drop_8_modes"]) != count:
        raise RuntimeError("mode-drop control did not produce enough samples")
    return base.float().cpu().numpy(), {
        name: value.float().cpu().numpy() for name, value in controls.items()
    }


def evaluate_arrays(
    arrays: dict[str, np.ndarray],
    *,
    mixture,
    reference_name: str,
    seed: int,
    energy_limit: int,
    c2st_limit: int,
    knn_limit: int,
) -> pd.DataFrame:
    reference = arrays[reference_name]
    intrinsic = {
        name: intrinsic_projection(mixture, values)
        for name, values in arrays.items()
    }
    statistics = {
        name: component_statistics(mixture, values)
        for name, values in arrays.items()
    }
    reference_stats = statistics[reference_name]
    reference_intrinsic = intrinsic[reference_name]
    centers = mixture.intrinsic_centers.float().cpu().numpy()
    nearest = np.linalg.norm(
        reference_intrinsic[:, None] - centers[None], axis=2
    ).min(axis=1)
    bandwidth = float(np.median(nearest))
    reference_contrast = adjacent_contrast_profile(
        reference, mixture, bandwidth
    )
    reference_regions = summarize_contrast_regions(reference_contrast)
    reference_widths = component_widths(
        reference_intrinsic,
        reference_stats["assignment"],
        centers,
    )
    reference_geometry = mixture.nearest_patch_geometry(
        torch.from_numpy(reference).to(mixture.device)
    )
    reference_counts = np.bincount(
        reference_stats["assignment"], minlength=mixture.components
    )
    low_nll = float(np.quantile(reference_stats["nll"], 0.05))
    high_nll = float(np.quantile(reference_stats["nll"], 0.95))

    rows = []
    for index, (name, values) in enumerate(arrays.items()):
        if name == reference_name:
            continue
        stats = statistics[name]
        points = intrinsic[name]
        geometry = mixture.nearest_patch_geometry(
            torch.from_numpy(values).to(mixture.device)
        )
        widths = component_widths(points, stats["assignment"], centers)
        width_regions = region_means(widths / reference_widths)
        contrast = adjacent_contrast_profile(values, mixture, bandwidth)
        contrast_regions = summarize_contrast_regions(contrast)
        counts = np.bincount(
            stats["assignment"], minlength=mixture.components
        )
        local_seed = stable_seed(seed, 3101, index)
        neighborhood = manifold_metrics(
            reference_intrinsic,
            points,
            k=5,
            limit=knn_limit,
            seed=local_seed,
        )
        rows.append(
            {
                "condition": name,
                "kind": (
                    "reference"
                    if name == "reference_replicate"
                    else "control"
                    if name.startswith("control_")
                    else "model"
                ),
                "intrinsic_swd": sliced_wasserstein_2d(
                    reference_intrinsic, points
                ),
                "energy_distance": energy_distance_2d(
                    reference_intrinsic,
                    points,
                    limit=energy_limit,
                    seed=local_seed,
                ),
                "c2st_auc": classifier_two_sample_auc(
                    reference_intrinsic,
                    points,
                    limit=c2st_limit,
                    seed=local_seed,
                ),
                "nll_ks": float(
                    ks_2samp(reference_stats["nll"], stats["nll"]).statistic
                ),
                "nll_w1": float(
                    wasserstein_distance(reference_stats["nll"], stats["nll"])
                ),
                "overconcentrated_rate": float(np.mean(stats["nll"] < low_nll)),
                "outlier_rate": float(np.mean(stats["nll"] > high_nll)),
                "tangent_width_ratio": (
                    geometry["nearest_tangent_rms"]
                    / reference_geometry["nearest_tangent_rms"]
                ),
                "normal_width_ratio": (
                    geometry["nearest_normal_rms"]
                    / reference_geometry["nearest_normal_rms"]
                ),
                **neighborhood,
                "component_jsd": jensen_shannon_counts(reference_counts, counts),
                "occupied_components": int(np.count_nonzero(counts)),
                "contrast_profile_rmse": float(
                    np.sqrt(np.mean(np.square(contrast - reference_contrast)))
                ),
                "inner_contrast_ratio": (
                    contrast_regions["inner_contrast"]
                    / max(reference_regions["inner_contrast"], EPS)
                ),
                "middle_contrast_ratio": (
                    contrast_regions["middle_contrast"]
                    / max(reference_regions["middle_contrast"], EPS)
                ),
                "outer_contrast_ratio": (
                    contrast_regions["outer_contrast"]
                    / max(reference_regions["outer_contrast"], EPS)
                ),
                "inner_local_width_ratio": width_regions["inner"],
                "middle_local_width_ratio": width_regions["middle"],
                "outer_local_width_ratio": width_regions["outer"],
            }
        )
    return pd.DataFrame(rows)


def plot_samples(
    arrays: dict[str, np.ndarray],
    *,
    mixture,
    order: list[str],
    path: Path,
) -> None:
    projected = {
        name: intrinsic_projection(mixture, arrays[name]) for name in order
    }
    reference = projected["reference"]
    lower = np.quantile(reference, 0.002, axis=0)
    upper = np.quantile(reference, 0.998, axis=0)
    padding = 0.06 * float(np.max(upper - lower))
    columns = 4
    rows = math.ceil(len(order) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.7 * columns, 4.7 * rows),
        constrained_layout=True,
    )
    for axis, name in zip(axes.flat, order):
        if name != "reference":
            axis.scatter(
                reference[:3000, 0],
                reference[:3000, 1],
                s=3,
                color="#8b8b8b",
                alpha=0.10,
                linewidths=0,
            )
        points = projected[name]
        axis.scatter(
            points[:3000, 0],
            points[:3000, 1],
            s=3,
            color="#2878b5" if name == "reference" else "#e66b1a",
            alpha=0.48,
            linewidths=0,
            rasterized=True,
        )
        axis.set_title(name.replace("_", " "), fontsize=10)
        axis.set_xlim(lower[0] - padding, upper[0] + padding)
        axis.set_ylim(lower[1] - padding, upper[1] + padding)
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in axes.flat[len(order) :]:
        axis.set_visible(False)
    figure.suptitle(
        "Guidance methods beside calibrated distribution failures",
        fontsize=15,
    )
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def plot_scorecard(frame: pd.DataFrame, path: Path) -> None:
    labels = frame.condition.str.replace("_", " ").tolist()
    y = np.arange(len(frame))
    figure, axes = plt.subplots(
        1, 4, figsize=(22, max(7.0, 0.43 * len(frame))), constrained_layout=True
    )

    axes[0].barh(y, frame.intrinsic_swd, color="#4c78a8", alpha=0.86)
    axes[0].set_title("Full distribution\nIntrinsic SWD (lower)")

    axes[1].scatter(frame.tangent_width_ratio, y, label="tangent", s=48)
    axes[1].scatter(frame.normal_width_ratio, y, label="normal", s=48)
    axes[1].axvline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Width calibration\n1.0 is correct")
    axes[1].legend(fontsize=8)

    axes[2].scatter(frame.precision, y, label="precision", s=48)
    axes[2].scatter(frame.recall, y, label="recall", s=48)
    axes[2].scatter(frame.coverage, y, label="coverage", s=48)
    axes[2].set_xlim(0.0, 1.03)
    axes[2].set_title("Fidelity and coverage\n(higher, interpreted together)")
    axes[2].legend(fontsize=8)

    axes[3].scatter(frame.inner_contrast_ratio, y, label="inner", s=48)
    axes[3].scatter(frame.middle_contrast_ratio, y, label="middle", s=48)
    axes[3].scatter(frame.outer_contrast_ratio, y, label="outer", s=48)
    axes[3].axvline(1.0, color="black", linestyle="--", linewidth=1)
    axes[3].set_title("Separation calibration\n1.0 matches target")
    axes[3].legend(fontsize=8)

    for axis in axes:
        axis.set_yticks(y, labels)
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.2)
    figure.suptitle(
        "No single sharpness score decides which distribution is better",
        fontsize=15,
    )
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def plot_typicality(frame: pd.DataFrame, path: Path) -> None:
    labels = frame.condition.str.replace("_", " ").tolist()
    y = np.arange(len(frame))
    figure, axes = plt.subplots(1, 3, figsize=(18, max(7.0, 0.43 * len(frame))))
    axes[0].scatter(frame.c2st_auc, y, color="#4c78a8", s=48)
    axes[0].axvline(0.5, color="black", linestyle="--", linewidth=1)
    axes[0].set_xlim(0.48, 1.01)
    axes[0].set_title("Nonlinear two-sample AUC\n0.5 is indistinguishable")
    axes[1].scatter(100.0 * frame.overconcentrated_rate, y, label="too dense", s=48)
    axes[1].scatter(100.0 * frame.outlier_rate, y, label="outlier", s=48)
    axes[1].axvline(5.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("True-density tails (%)\n5% / 5% is calibrated")
    axes[1].legend(fontsize=8)
    axes[2].scatter(frame.nll_ks, y, color="#f28e2b", s=48)
    axes[2].set_title("NLL-distribution KS\n(lower is more typical)")
    for axis in axes:
        axis.set_yticks(y, labels)
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def expand_scale_frame(
    frame: pd.DataFrame,
    *,
    methods: list[str],
    weights: list[float],
) -> pd.DataFrame:
    indexed = frame.set_index("condition")
    rows = []
    for method in methods:
        for weight in weights:
            name = scale_sample_name(method, weight)
            if name not in indexed.index:
                continue
            row = indexed.loc[name].to_dict()
            row.update(method=method, weight=weight, source_condition=name)
            rows.append(row)
    return pd.DataFrame(rows)


def plot_scale_benchmark(
    frame: pd.DataFrame,
    *,
    methods: list[str],
    weights: list[float],
    path: Path,
) -> None:
    scale = expand_scale_frame(frame, methods=methods, weights=weights)
    if scale.empty:
        return
    replicate = frame.loc[frame.condition == "reference_replicate"].iloc[0]
    v_baseline = frame.loc[frame.condition == "v"].iloc[0]
    figure, axes = plt.subplots(2, 3, figsize=(19, 11), constrained_layout=True)
    panels = [
        ("intrinsic_swd", "Intrinsic SWD", float(replicate.intrinsic_swd)),
        ("nll_ks", "True-NLL distribution KS", float(replicate.nll_ks)),
        ("normal_width_ratio", "Normal width ratio", 1.0),
        ("inner_contrast_ratio", "Inner separation ratio", 1.0),
        ("precision", "Precision", float(replicate.precision)),
    ]
    for axis, (column, title, target) in zip(axes.flat[:5], panels):
        for method, group in scale.groupby("method", sort=False):
            group = group.sort_values("weight")
            axis.plot(
                group.weight,
                group[column],
                marker="o",
                linewidth=2,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
        axis.axhline(target, color="black", linestyle="--", linewidth=1)
        axis.axhline(
            float(v_baseline[column]),
            color="#666666",
            linestyle=":",
            linewidth=1.4,
            label="standalone v baseline",
        )
        axis.set_xlabel("Guidance scale w")
        axis.set_title(title)
        axis.grid(alpha=0.22)
    coverage_axis = axes.flat[5]
    for method, group in scale.groupby("method", sort=False):
        group = group.sort_values("weight")
        coverage_axis.plot(
            group.weight,
            group.recall,
            marker="o",
            linewidth=2,
            color=METHOD_COLORS[method],
        )
        coverage_axis.plot(
            group.weight,
            group.coverage,
            marker="s",
            linestyle="--",
            linewidth=1.5,
            color=METHOD_COLORS[method],
        )
    coverage_axis.axhline(
        float(replicate.recall), color="black", linestyle="-", linewidth=1
    )
    coverage_axis.axhline(
        float(replicate.coverage), color="black", linestyle="--", linewidth=1
    )
    coverage_axis.axhline(
        float(v_baseline.recall), color="#666666", linestyle=":", linewidth=1.4
    )
    coverage_axis.set_xlabel("Guidance scale w")
    coverage_axis.set_title("Recall (circle) and coverage (square)")
    coverage_axis.grid(alpha=0.22)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes[0, 0].legend(unique.values(), unique.keys(), fontsize=8)
    figure.suptitle(
        "A useful guidance scale must improve separation without losing the target distribution",
        fontsize=16,
    )
    figure.savefig(path, dpi=185, bbox_inches="tight")
    plt.close(figure)


def plot_scale_pareto(
    frame: pd.DataFrame,
    *,
    methods: list[str],
    weights: list[float],
    path: Path,
) -> None:
    scale = expand_scale_frame(frame, methods=methods, weights=weights)
    if scale.empty:
        return
    figure, axes = plt.subplots(1, 3, figsize=(20, 6.3), constrained_layout=True)
    for method, group in scale.groupby("method", sort=False):
        group = group.sort_values("weight")
        color = METHOD_COLORS[method]
        label = METHOD_LABELS[method]
        axes[0].plot(
            group.intrinsic_swd,
            group.inner_contrast_ratio,
            marker="o",
            color=color,
            linewidth=2,
            label=label,
        )
        axes[1].plot(
            group.recall,
            group.precision,
            marker="o",
            color=color,
            linewidth=2,
        )
        axes[2].plot(
            group.normal_width_ratio,
            group.inner_contrast_ratio,
            marker="o",
            color=color,
            linewidth=2,
        )
        for _, row in group.iterrows():
            for axis, x, y in (
                (axes[0], row.intrinsic_swd, row.inner_contrast_ratio),
                (axes[1], row.recall, row.precision),
                (axes[2], row.normal_width_ratio, row.inner_contrast_ratio),
            ):
                axis.annotate(
                    f"{row.weight:g}",
                    (x, y),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                    color=color,
                )
    baseline_styles = {
        "reference_replicate": ("Reference replicate", "*", "#111111"),
        "x": ("Standalone x", "D", "#777777"),
        "v": ("Standalone v", "P", "#4c78a8"),
    }
    indexed = frame.set_index("condition")
    for name, (label, marker, color) in baseline_styles.items():
        row = indexed.loc[name]
        axes[0].scatter(
            row.intrinsic_swd,
            row.inner_contrast_ratio,
            marker=marker,
            s=110,
            color=color,
            edgecolors="white",
            linewidths=0.8,
            zorder=6,
            label=label,
        )
        axes[1].scatter(
            row.recall,
            row.precision,
            marker=marker,
            s=110,
            color=color,
            edgecolors="white",
            linewidths=0.8,
            zorder=6,
        )
        axes[2].scatter(
            row.normal_width_ratio,
            row.inner_contrast_ratio,
            marker=marker,
            s=110,
            color=color,
            edgecolors="white",
            linewidths=0.8,
            zorder=6,
        )
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0].set(
        xlabel="Intrinsic SWD (left is better)",
        ylabel="Inner separation ratio (1 is target)",
        title="Global fit versus inner-mode separation",
    )
    axes[1].set(
        xlabel="Recall",
        ylabel="Precision",
        title="Coverage versus fidelity",
    )
    axes[2].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[2].axvline(1.0, color="black", linestyle="--", linewidth=1)
    axes[2].set(
        xlabel="Normal width ratio (1 is target)",
        ylabel="Inner separation ratio (1 is target)",
        title="Does sharpening preserve local width?",
    )
    for axis in axes:
        axis.grid(alpha=0.22)
    axes[0].legend(fontsize=8)
    figure.suptitle("Guidance scale Pareto map", fontsize=16)
    figure.savefig(path, dpi=185, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--conditions",
        default=(
            "bayes,x,v,ig_w0,ig_w1,ig_w2_mid03_07,ig_w3_mid03_07,"
            "ag_early_w3_mid03_07"
        ),
    )
    parser.add_argument("--sample-count", type=int, default=5000)
    parser.add_argument("--energy-limit", type=int, default=1500)
    parser.add_argument("--c2st-limit", type=int, default=2500)
    parser.add_argument("--knn-limit", type=int, default=2500)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    sample_dir = args.sample_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_manifest = json.loads(
        (sample_dir / "manifest.json").read_text(encoding="utf-8")
    )
    seed = int(sample_manifest["seed"])
    source_dir = Path(str(sample_manifest["source_root"])) / f"seed{seed}"
    source_manifest = json.loads(
        (source_dir / "manifest.json").read_text(encoding="utf-8")
    )
    device = torch.device(args.device)
    mixture = build_mixture(source_manifest, device)
    cached_reference = np.load(source_dir / "common" / "reference.npy")
    if len(cached_reference) >= args.sample_count:
        reference = cached_reference[: args.sample_count]
        reference_source = "cached_prefix"
    else:
        reference_generator = torch.Generator(device=device.type)
        reference_generator.manual_seed(stable_seed(seed, 607))
        reference_tensor, _ = mixture.sample_clean(
            args.sample_count, generator=reference_generator
        )
        reference = reference_tensor.float().cpu().numpy()
        reference_source = "fresh_analytic_mixture_fixed_seed"
    count = len(reference)
    arrays: dict[str, np.ndarray] = {"reference": reference}
    replicate, controls = make_controls(
        mixture,
        count=count,
        seed=stable_seed(seed, 3001),
    )
    arrays["reference_replicate"] = replicate
    arrays.update(controls)
    condition_names = (
        sorted(
            path.stem.removeprefix("samples_")
            for path in sample_dir.glob("samples_*.npy")
        )
        if args.conditions.strip().lower() == "all"
        else parse_names(args.conditions)
    )
    missing = []
    for name in condition_names:
        path = sample_dir / f"samples_{name}.npy"
        if not path.is_file():
            missing.append(name)
            continue
        arrays[name] = np.load(path)[:count]
    if missing:
        raise FileNotFoundError(f"missing sample arrays: {missing}")

    frame = evaluate_arrays(
        arrays,
        mixture=mixture,
        reference_name="reference",
        seed=seed,
        energy_limit=args.energy_limit,
        c2st_limit=args.c2st_limit,
        knn_limit=args.knn_limit,
    )
    order = [name for name in arrays if name != "reference"]
    order_frame = pd.DataFrame({"condition": order, "display_order": range(len(order))})
    frame = order_frame.merge(frame, on="condition", how="left").sort_values(
        "display_order"
    )
    frame.to_csv(output_dir / "distribution_metrics.csv", index=False)
    plot_order = ["reference", *order]
    plot_samples(
        arrays,
        mixture=mixture,
        order=plot_order,
        path=output_dir / "calibrated_failure_atlas.png",
    )
    plot_scorecard(frame, output_dir / "distribution_scorecard.png")
    plot_typicality(frame, output_dir / "typicality_scorecard.png")
    methods = [str(value) for value in sample_manifest.get("methods", [])]
    weights = sorted(
        {1.0, *[float(value) for value in sample_manifest.get("weights", [])]}
    )
    plot_scale_benchmark(
        frame,
        methods=methods,
        weights=weights,
        path=output_dir / "scale_benchmark.png",
    )
    plot_scale_pareto(
        frame,
        methods=methods,
        weights=weights,
        path=output_dir / "scale_pareto.png",
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "protocol": "guidance_toy_distribution_benchmark_v1",
                "sample_dir": str(sample_dir),
                "source_dir": str(source_dir),
                "seed": seed,
                "sample_count": count,
                "reference_source": reference_source,
                "conditions": condition_names,
                "reference_replicate": "fresh analytic-mixture sample",
                "controls": list(controls),
                "metric_interpretation": {
                    "distances_and_c2st": "closer to zero or 0.5 AUC is better",
                    "width_and_contrast_ratios": "closer to 1 is better",
                    "precision_recall_density_coverage": "inspect jointly",
                    "overconcentrated_and_outlier_rates": "5% each is calibrated",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    columns = [
        "condition",
        "intrinsic_swd",
        "c2st_auc",
        "tangent_width_ratio",
        "normal_width_ratio",
        "precision",
        "recall",
        "coverage",
        "component_jsd",
        "inner_contrast_ratio",
        "middle_contrast_ratio",
        "outer_contrast_ratio",
    ]
    print(frame[columns].to_string(index=False))


if __name__ == "__main__":
    main()
