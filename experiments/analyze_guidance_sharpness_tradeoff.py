#!/usr/bin/env python3
"""Summarize sharpness versus distribution fidelity for guidance toy runs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from experiments.analyze_prediction_target_cluster_separation import (
    intrinsic_projection,
    kde_density,
)
from experiments.evaluate_prediction_target_autoguidance import (
    build_mixture,
    plot_intrinsic,
)
from experiments.run_prediction_target_bayes_oracle_v5 import stable_seed


FAMILY_COLORS = {
    "bayes": "#4c78a8",
    "x": "#4c4c4c",
    "v": "#9c755f",
    "ptg": "#b279a2",
    "ag_early": "#f28e2b",
    "ag_small": "#e15759",
    "ig": "#59a14f",
}
WINDOW_MARKERS = {"full": "o", "windowed": "s"}


def method_family(condition: str) -> str:
    if condition in {"bayes", "x", "v"}:
        return condition
    for prefix in ("ag_early", "ag_small", "ptg", "ig"):
        if condition.startswith(prefix + "_"):
            return prefix
    return "other"


def is_windowed(condition: str) -> bool:
    return bool(re.search(r"_(?:mid|high|low)\d", condition))


def candidate_table(
    summary: pd.DataFrame,
    reference: pd.Series,
) -> pd.DataFrame:
    """Select one fidelity-efficient sharp candidate per guidance family.

    A candidate must retain every component, reach at least 90% of the
    reference peak/valley contrast, and keep its bridge rate within three
    percentage points of the reference. Among qualifying settings we choose
    the lowest latent SWD. The thresholds are fixed before looking at a
    particular guidance family, so this is not a visual best-case selection.
    """
    frame = summary.copy()
    frame["family"] = frame.condition.map(method_family)
    frame["windowed"] = frame.condition.map(is_windowed)
    contrast_target = float(reference.mean_adjacent_log_density_contrast)
    bridge_target = float(reference.intrinsic_bridge_rate)
    component_target = int(reference.occupied_components)
    frame["passes_sharpness_gate"] = (
        (
            frame.mean_adjacent_log_density_contrast
            >= 0.90 * contrast_target
        )
        & (frame.intrinsic_bridge_rate <= bridge_target + 0.03)
        & (frame.occupied_components >= component_target)
    )
    selected = []
    for family in ("ptg", "ag_early", "ag_small", "ig"):
        group = frame[
            (frame.family == family) & frame.passes_sharpness_gate
        ]
        if group.empty:
            continue
        selected.append(group.sort_values("latent_swd").iloc[0])
    if not selected:
        return frame.iloc[:0].copy()
    return pd.DataFrame(selected).reset_index(drop=True)


def contrast_match_table(
    summary: pd.DataFrame,
    reference: pd.Series,
) -> pd.DataFrame:
    """Choose the setting whose mean contrast best matches the reference.

    Complete mode coverage and the same three-percentage-point bridge-rate
    tolerance are required first. This rule targets visual separation rather
    than minimum global SWD, and is reported separately from ``candidate_table``.
    """
    frame = summary.copy()
    frame["family"] = frame.condition.map(method_family)
    contrast_target = float(reference.mean_adjacent_log_density_contrast)
    bridge_target = float(reference.intrinsic_bridge_rate)
    component_target = int(reference.occupied_components)
    frame = frame[
        (frame.intrinsic_bridge_rate <= bridge_target + 0.03)
        & (frame.occupied_components >= component_target)
    ].copy()
    frame["absolute_contrast_gap"] = (
        frame.mean_adjacent_log_density_contrast - contrast_target
    ).abs()
    selected = []
    for family in ("ptg", "ag_early", "ag_small", "ig"):
        group = frame[frame.family == family]
        if group.empty:
            continue
        selected.append(
            group.sort_values(["absolute_contrast_gap", "latent_swd"]).iloc[0]
        )
    if not selected:
        return frame.iloc[:0].copy()
    return pd.DataFrame(selected).reset_index(drop=True)


def plot_tradeoff(
    summary: pd.DataFrame,
    reference: pd.Series,
    candidates: pd.DataFrame,
    path: Path,
) -> None:
    frame = summary.copy()
    frame["family"] = frame.condition.map(method_family)
    frame["window"] = np.where(
        frame.condition.map(is_windowed), "windowed", "full"
    )
    reference_contrast = float(reference.mean_adjacent_log_density_contrast)
    reference_bridge = float(reference.intrinsic_bridge_rate)

    figure, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    for family, group in frame.groupby("family"):
        if family not in FAMILY_COLORS:
            continue
        for window, subgroup in group.groupby("window"):
            axes[0].scatter(
                subgroup.latent_swd,
                subgroup.mean_adjacent_log_density_contrast,
                s=62,
                marker=WINDOW_MARKERS[window],
                color=FAMILY_COLORS[family],
                alpha=0.82,
                label=f"{family}, {window}",
            )
            axes[1].scatter(
                subgroup.latent_swd,
                100.0 * subgroup.intrinsic_bridge_rate,
                s=62,
                marker=WINDOW_MARKERS[window],
                color=FAMILY_COLORS[family],
                alpha=0.82,
                label=f"{family}, {window}",
            )

    axes[0].axhline(
        reference_contrast,
        color="#2574a9",
        linestyle="--",
        linewidth=1.5,
        label="reference",
    )
    axes[1].axhline(
        100.0 * reference_bridge,
        color="#2574a9",
        linestyle="--",
        linewidth=1.5,
        label="reference",
    )
    for _, row in candidates.iterrows():
        for axis, y in (
            (axes[0], row.mean_adjacent_log_density_contrast),
            (axes[1], 100.0 * row.intrinsic_bridge_rate),
        ):
            axis.annotate(
                row.condition,
                (row.latent_swd, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
            axis.scatter(
                [row.latent_swd],
                [y],
                s=150,
                facecolors="none",
                edgecolors="black",
                linewidths=1.4,
            )

    axes[0].set(
        xlabel="Latent SWD (lower = closer to full distribution)",
        ylabel="Adjacent peak/valley log contrast (higher = sharper)",
        title="Sharpness versus distribution fidelity",
    )
    axes[1].set(
        xlabel="Latent SWD (lower = closer to full distribution)",
        ylabel="Intrinsic bridge rate (%) (lower = better separated)",
        title="Cluster bridges versus distribution fidelity",
    )
    for axis in axes:
        axis.grid(alpha=0.22)
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes[0].legend(unique.values(), unique.keys(), fontsize=8, ncol=2)
    figure.suptitle(
        "Guidance is a sharpness/fidelity trade-off, not a single-score win",
        fontsize=15,
    )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def reconstruct_reference(
    output_dir: Path,
    manifest: dict[str, object],
) -> np.ndarray:
    source_root = Path(str(manifest["source_root"]))
    seed = int(manifest["seed"])
    cached = np.load(source_root / f"seed{seed}" / "common" / "reference.npy")
    count = int(manifest["sample_count"])
    rng = np.random.default_rng(stable_seed(seed, 2287))
    indices = rng.choice(len(cached), size=count, replace=False)
    return cached[indices]


def adjacent_contrast_profile(
    values: np.ndarray,
    mixture,
    bandwidth: float,
) -> np.ndarray:
    intrinsic = intrinsic_projection(mixture, values)
    centers = mixture.intrinsic_centers.float().cpu().numpy()
    midpoints = 0.5 * (centers[:-1] + centers[1:])
    center_density = kde_density(intrinsic, centers, bandwidth)
    midpoint_density = kde_density(intrinsic, midpoints, bandwidth)
    peaks = 0.5 * (center_density[:-1] + center_density[1:])
    return np.log(peaks + 1e-12) - np.log(midpoint_density + 1e-12)


def summarize_contrast_regions(profile: np.ndarray) -> dict[str, float]:
    """Split ordered spiral gaps into inner, middle, and outer thirds."""
    chunks = np.array_split(np.asarray(profile, dtype=np.float64), 3)
    return {
        "inner_contrast": float(chunks[0].mean()),
        "middle_contrast": float(chunks[1].mean()),
        "outer_contrast": float(chunks[2].mean()),
        "all_contrast": float(np.mean(profile)),
    }


def regional_contrast_table(
    *,
    output_dir: Path,
    manifest: dict[str, object],
    summary: pd.DataFrame,
    device: torch.device,
) -> pd.DataFrame:
    source_manifest = json.loads(
        (
            Path(str(manifest["source_root"]))
            / f"seed{int(manifest['seed'])}"
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    mixture = build_mixture(source_manifest, device)
    reference = reconstruct_reference(output_dir, manifest)
    reference_intrinsic = intrinsic_projection(mixture, reference)
    centers = mixture.intrinsic_centers.float().cpu().numpy()
    nearest = np.linalg.norm(
        reference_intrinsic[:, None, :] - centers[None, :, :], axis=2
    ).min(axis=1)
    bandwidth = float(np.median(nearest))
    arrays = {"reference": reference}
    for condition in summary.condition:
        path = output_dir / f"samples_{condition}.npy"
        if path.is_file():
            arrays[condition] = np.load(path)
    swd = dict(zip(summary.condition, summary.latent_swd))
    rows = []
    for condition, values in arrays.items():
        profile = adjacent_contrast_profile(values, mixture, bandwidth)
        rows.append(
            {
                "condition": condition,
                "family": method_family(condition),
                "windowed": is_windowed(condition),
                "latent_swd": 0.0 if condition == "reference" else swd[condition],
                **summarize_contrast_regions(profile),
            }
        )
    return pd.DataFrame(rows)


def plot_regional_tradeoff(frame: pd.DataFrame, path: Path) -> None:
    reference = frame.loc[frame.condition == "reference"].iloc[0]
    generated = frame[frame.condition != "reference"]
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    regions = [
        ("inner_contrast", "Inner gaps 1-11"),
        ("middle_contrast", "Middle gaps 12-21"),
        ("outer_contrast", "Outer gaps 22-31"),
    ]
    for axis, (column, title) in zip(axes, regions):
        for family, group in generated.groupby("family"):
            if family not in FAMILY_COLORS:
                continue
            for windowed, subgroup in group.groupby("windowed"):
                axis.scatter(
                    subgroup.latent_swd,
                    subgroup[column],
                    s=58,
                    color=FAMILY_COLORS[family],
                    marker="s" if windowed else "o",
                    alpha=0.82,
                    label=f"{family}, {'windowed' if windowed else 'full'}",
                )
        axis.axhline(
            float(reference[column]),
            color="#2574a9",
            linestyle="--",
            linewidth=1.5,
            label="reference",
        )
        axis.set(
            xlabel="Latent SWD (lower = closer)",
            ylabel="Mean peak/valley log contrast",
            title=title,
        )
        axis.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes[0].legend(unique.values(), unique.keys(), fontsize=8)
    figure.suptitle(
        "Where along the spiral does guidance sharpen the distribution?",
        fontsize=15,
    )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_contrast_profiles(
    *,
    output_dir: Path,
    manifest: dict[str, object],
    candidates: pd.DataFrame,
    path: Path,
    device: torch.device,
) -> None:
    mixture = build_mixture(
        json.loads(
            (
                Path(str(manifest["source_root"]))
                / f"seed{int(manifest['seed'])}"
                / "manifest.json"
            ).read_text(encoding="utf-8")
        ),
        device,
    )
    reference = reconstruct_reference(output_dir, manifest)
    reference_intrinsic = intrinsic_projection(mixture, reference)
    centers = mixture.intrinsic_centers.float().cpu().numpy()
    reference_distance = np.linalg.norm(
        reference_intrinsic[:, None, :] - centers[None, :, :], axis=2
    ).min(axis=1)
    bandwidth = float(np.median(reference_distance))

    names = ["reference", "x"]
    names.extend(candidates.condition.tolist())
    names = list(dict.fromkeys(names))
    arrays = {"reference": reference}
    for name in names[1:]:
        sample_path = output_dir / f"samples_{name}.npy"
        if sample_path.is_file():
            arrays[name] = np.load(sample_path)

    figure, axis = plt.subplots(figsize=(14, 6), constrained_layout=True)
    gap_index = np.arange(1, len(centers))
    for name, values in arrays.items():
        profile = adjacent_contrast_profile(values, mixture, bandwidth)
        family = method_family(name)
        axis.plot(
            gap_index,
            profile,
            marker="o" if name in {"reference", "x"} else None,
            markersize=3,
            linewidth=2.2 if name in {"reference", "x"} else 1.7,
            color="#2574a9" if name == "reference" else FAMILY_COLORS.get(
                family, "#777777"
            ),
            alpha=0.92,
            label=name,
        )
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    axis.set(
        xlabel="Gap between adjacent spiral components",
        ylabel="Peak/valley log-density contrast",
        title="Does guidance separate every adjacent pair, or only a few?",
        xticks=gap_index,
    )
    axis.grid(alpha=0.2)
    axis.legend(ncol=2)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_candidate_samples(
    *,
    output_dir: Path,
    manifest: dict[str, object],
    balanced: pd.DataFrame,
    contrast_matched: pd.DataFrame,
    device: torch.device,
    analysis_output_dir: Path,
) -> None:
    source_manifest = json.loads(
        (
            Path(str(manifest["source_root"]))
            / f"seed{int(manifest['seed'])}"
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    mixture = build_mixture(source_manifest, device)
    arrays = {"reference": reconstruct_reference(output_dir, manifest)}
    ordered = ["reference", "x"]
    ordered.extend(balanced.condition.tolist())
    ordered.extend(contrast_matched.condition.tolist())
    ordered = list(dict.fromkeys(ordered))
    available = ["reference"]
    for name in ordered[1:]:
        sample_path = output_dir / f"samples_{name}.npy"
        if sample_path.is_file():
            arrays[name] = np.load(sample_path)
            available.append(name)
    plot_intrinsic(
        path=analysis_output_dir / "candidate_samples.png",
        arrays=arrays,
        mixture=mixture,
        ordered_names=available,
        title="Reference, baseline, fidelity-efficient and contrast-matched guidance",
    )
    plot_intrinsic(
        path=analysis_output_dir / "candidate_samples_inner_zoom.png",
        arrays=arrays,
        mixture=mixture,
        ordered_names=available,
        title="Candidate guidance settings: inner spiral zoom",
        zoom_limit=0.85,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--analysis-output-dir",
        type=Path,
        help="Write figures here while reading samples from --output-dir.",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    analysis_output_dir = (
        args.analysis_output_dir.resolve()
        if args.analysis_output_dir is not None
        else output_dir
    )
    analysis_output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(output_dir / "summary.csv")
    cluster = pd.read_csv(output_dir / "cluster_metrics.csv")
    manifest = json.loads(
        (output_dir / "manifest.json").read_text(encoding="utf-8")
    )
    reference = cluster.loc[cluster.condition == "reference"].iloc[0]
    candidates = candidate_table(summary, reference)
    contrast_matched = contrast_match_table(summary, reference)
    candidates.to_csv(
        analysis_output_dir / "balanced_candidates.csv", index=False
    )
    contrast_matched.to_csv(
        analysis_output_dir / "reference_contrast_candidates.csv", index=False
    )
    regional = regional_contrast_table(
        output_dir=output_dir,
        manifest=manifest,
        summary=summary,
        device=torch.device(args.device),
    )
    regional.to_csv(
        analysis_output_dir / "regional_contrast_metrics.csv", index=False
    )
    plot_tradeoff(
        summary,
        reference,
        candidates,
        analysis_output_dir / "guidance_tradeoff.png",
    )
    plot_contrast_profiles(
        output_dir=output_dir,
        manifest=manifest,
        candidates=candidates,
        path=analysis_output_dir / "adjacent_contrast_profiles.png",
        device=torch.device(args.device),
    )
    plot_candidate_samples(
        output_dir=output_dir,
        manifest=manifest,
        balanced=candidates,
        contrast_matched=contrast_matched,
        device=torch.device(args.device),
        analysis_output_dir=analysis_output_dir,
    )
    plot_regional_tradeoff(
        regional,
        analysis_output_dir / "regional_sharpness_tradeoff.png",
    )
    if candidates.empty:
        print("No guidance condition passed the fixed sharpness gate.")
    else:
        columns = [
            "condition",
            "latent_swd",
            "intrinsic_bridge_rate",
            "mean_adjacent_log_density_contrast",
            "component_jsd_y",
            "occupied_components",
        ]
        print(candidates[columns].to_string(index=False))
    if not contrast_matched.empty:
        print("\nReference-contrast matches:")
        print(contrast_matched[columns].to_string(index=False))


if __name__ == "__main__":
    main()
