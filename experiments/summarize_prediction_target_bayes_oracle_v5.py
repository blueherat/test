#!/usr/bin/env python3
"""Aggregate exact-Bayes prediction-target v5 runs across seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GROUP = [
    "D",
    "components",
    "sigma_tangent",
    "sigma_normal",
    "architecture",
    "hidden",
    "loss_space",
]


def load_tables(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries = []
    generation = []
    teacher = []
    for path in root.rglob("setting_summary.json"):
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    for path in root.rglob("generation_metrics.csv"):
        generation.append(pd.read_csv(path))
    for path in root.rglob("teacher_metrics.csv"):
        teacher.append(pd.read_csv(path))
    if not summaries or not generation or not teacher:
        raise FileNotFoundError(f"incomplete v5 results under {root}")
    return (
        pd.DataFrame(summaries),
        pd.concat(generation, ignore_index=True),
        pd.concat(teacher, ignore_index=True),
    )


def build_baseline_aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in summary.groupby(GROUP, dropna=False):
        row = dict(zip(GROUP, key))
        row.update(
            {
                "seeds": int(group["seed"].nunique()),
                "mean_x_excess_over_bayes_risk": group[
                    "x_excess_over_bayes_risk"
                ].mean(),
                "mean_v_excess_over_bayes_risk": group[
                    "v_excess_over_bayes_risk"
                ].mean(),
                "mean_x_latent_swd": group["x_latent_swd"].mean(),
                "mean_v_latent_swd": group["v_latent_swd"].mean(),
                "mean_bayes_latent_swd": group["bayes_latent_swd"].mean(),
                "x_better_latent_seed_fraction": group[
                    "x_better_than_v_latent_swd"
                ].astype(float).mean(),
                "x_better_pushforward_seed_fraction": group[
                    "x_better_than_v_pushforward_swd"
                ].astype(float).mean(),
                "mean_gap_normal_energy_fraction": group[
                    "mean_gap_xv_normal_energy_fraction"
                ].mean(),
                "mean_gap_bayes_residual_cosine": group[
                    "mean_cos_xv_bayes_residual"
                ].mean(),
                "mean_gamma_star_bayes": group["mean_gamma_star_bayes"].mean(),
                "mean_x_plateau_relative_change": group[
                    "x_plateau_relative_change"
                ].mean(),
                "mean_v_plateau_relative_change": group[
                    "v_plateau_relative_change"
                ].mean(),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(GROUP).reset_index(drop=True)


def build_condition_contrasts(generation: pd.DataFrame) -> pd.DataFrame:
    keys = GROUP + ["seed"]
    baseline = generation[generation["condition"] == "x"].set_index(keys)
    rows = []
    candidates = generation[generation["condition"].str.startswith("xv_")]
    for _, candidate in candidates.iterrows():
        key = tuple(candidate[column] for column in keys)
        x = baseline.loc[key]
        if isinstance(x, pd.DataFrame):
            raise ValueError(f"duplicate x baseline for {key}")
        row = {column: candidate[column] for column in keys}
        row.update(
            {
                "condition": candidate["condition"],
                "kind": candidate["kind"],
                "strength": candidate["strength"],
                "delta_latent_swd": candidate["latent_swd"] - x["latent_swd"],
                "relative_latent_swd": candidate["latent_swd"]
                / max(x["latent_swd"], 1e-12)
                - 1.0,
                "delta_latent_mmd": candidate["latent_mmd_rff"]
                - x["latent_mmd_rff"],
                "delta_pushforward_swd": candidate["pushforward_swd"]
                - x["pushforward_swd"],
                "relative_pushforward_swd": candidate["pushforward_swd"]
                / max(x["pushforward_swd"], 1e-12)
                - 1.0,
                "delta_pushforward_mmd": candidate["pushforward_mmd_rff"]
                - x["pushforward_mmd_rff"],
                "delta_component_jsd": candidate["component_jsd"]
                - x["component_jsd"],
                "delta_nearest_normal_rms": candidate["nearest_normal_rms"]
                - x["nearest_normal_rms"],
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_contrasts(contrasts: pd.DataFrame) -> pd.DataFrame:
    grouping = GROUP + ["condition", "kind", "strength"]
    rows = []
    for key, group in contrasts.groupby(grouping, dropna=False):
        row = dict(zip(grouping, key))
        row.update(
            {
                "seeds": int(group["seed"].nunique()),
                "mean_relative_latent_swd": group["relative_latent_swd"].mean(),
                "latent_swd_improved_seed_fraction": (
                    group["delta_latent_swd"] < 0
                ).mean(),
                "latent_mmd_improved_seed_fraction": (
                    group["delta_latent_mmd"] < 0
                ).mean(),
                "mean_relative_pushforward_swd": group[
                    "relative_pushforward_swd"
                ].mean(),
                "pushforward_swd_improved_seed_fraction": (
                    group["delta_pushforward_swd"] < 0
                ).mean(),
                "pushforward_mmd_improved_seed_fraction": (
                    group["delta_pushforward_mmd"] < 0
                ).mean(),
                "mean_delta_component_jsd": group["delta_component_jsd"].mean(),
                "mean_delta_nearest_normal_rms": group[
                    "delta_nearest_normal_rms"
                ].mean(),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(grouping).reset_index(drop=True)


def plot_summary(path: Path, baseline: pd.DataFrame, contrasts: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    labels = [
        f"{row.architecture}\nH={int(row.hidden)}"
        for row in baseline.itertuples()
    ]
    x = np.arange(len(labels))
    axes[0].bar(x - 0.18, baseline["mean_x_latent_swd"], width=0.36, label="x")
    axes[0].bar(x + 0.18, baseline["mean_v_latent_swd"], width=0.36, label="v")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Ambient SWD")
    axes[0].set_title("Closed-loop baseline")
    axes[0].legend()

    axes[1].bar(x - 0.18, baseline["mean_x_excess_over_bayes_risk"], width=0.36, label="x")
    axes[1].bar(x + 0.18, baseline["mean_v_excess_over_bayes_risk"], width=0.36, label="v")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Excess / Bayes risk")
    axes[1].set_title("Teacher-forced capacity")
    axes[1].legend()

    full = contrasts[contrasts["kind"] == "xv"]
    for (architecture, hidden), group in full.groupby(["architecture", "hidden"]):
        group = group.sort_values("strength")
        axes[2].plot(
            group["strength"],
            100.0 * group["mean_relative_latent_swd"],
            marker="o",
            label=f"{architecture}, H={int(hidden)}",
        )
    axes[2].axhline(0.0, color="black", linewidth=1)
    axes[2].set_xlabel("gamma in x + gamma(x-v)")
    axes[2].set_ylabel("Latent SWD change (%)")
    axes[2].set_title("Prediction-target intervention")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def load_jit_projections(root: Path) -> list[tuple[dict, dict[str, np.ndarray]]]:
    entries = []
    for path in root.rglob("jit_projection.npz"):
        summary_path = path.with_name("setting_summary.json")
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        with np.load(path) as data:
            arrays = {key: data[key] for key in data.files}
        entries.append((summary, arrays))
    return entries


def plot_jit_comparisons(root: Path, output_dir: Path) -> None:
    entries = load_jit_projections(root)
    if not entries:
        return
    group_keys = sorted(
        {
            (
                int(summary["D"]),
                float(summary["sigma_tangent"]),
                float(summary["sigma_normal"]),
                int(summary["hidden"]),
                str(summary["loss_space"]),
            )
            for summary, _ in entries
        }
    )
    columns = ("reference", "bayes", "x", "eps", "v")
    column_titles = ("Reference", "Bayes oracle", "x-pred", "eps-pred", "v-pred")
    architecture_order = ("jit_relu", "plain", "residual", "residual_skip")
    for D, sigma_tangent, sigma_normal, hidden, loss_space in group_keys:
        selected = [
            (summary, arrays)
            for summary, arrays in entries
            if int(summary["D"]) == D
            and float(summary["sigma_tangent"]) == sigma_tangent
            and float(summary["sigma_normal"]) == sigma_normal
            and int(summary["hidden"]) == hidden
            and str(summary["loss_space"]) == loss_space
        ]
        architectures = [
            architecture
            for architecture in architecture_order
            if any(summary["architecture"] == architecture for summary, _ in selected)
        ]
        if not architectures:
            continue
        combined: dict[tuple[str, str], np.ndarray] = {}
        for architecture in architectures:
            architecture_entries = [
                arrays
                for summary, arrays in selected
                if summary["architecture"] == architecture
            ]
            for column in columns:
                combined[(architecture, column)] = np.concatenate(
                    [arrays[column][:1200] for arrays in architecture_entries], axis=0
                )
        reference_points = np.concatenate(
            [combined[(architecture, "reference")] for architecture in architectures],
            axis=0,
        )
        low = np.quantile(reference_points, 0.005, axis=0)
        high = np.quantile(reference_points, 0.995, axis=0)
        center = 0.5 * (low + high)
        radius = max(0.62 * float(np.max(high - low)), 0.25)

        fig, axes = plt.subplots(
            len(architectures),
            len(columns),
            figsize=(16.5, 3.25 * len(architectures)),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        for row_index, architecture in enumerate(architectures):
            reference = combined[(architecture, "reference")]
            for column_index, (column, title) in enumerate(zip(columns, column_titles)):
                axis = axes[row_index, column_index]
                if column != "reference":
                    axis.scatter(
                        reference[:, 0],
                        reference[:, 1],
                        s=2,
                        alpha=0.08,
                        color="#555555",
                        linewidths=0,
                        rasterized=True,
                    )
                points = combined[(architecture, column)]
                axis.scatter(
                    points[:, 0],
                    points[:, 1],
                    s=3,
                    alpha=0.35 if column == "reference" else 0.43,
                    color="#2878b5" if column == "reference" else "#d95f02",
                    linewidths=0,
                    rasterized=True,
                )
                if row_index == 0:
                    axis.set_title(title, fontsize=11)
                if column_index == 0:
                    axis.set_ylabel(architecture, fontsize=11)
                axis.set_aspect("equal", adjustable="box")
                axis.set_xlim(center[0] - radius, center[0] + radius)
                axis.set_ylim(center[1] - radius, center[1] + radius)
                axis.set_xticks([])
                axis.set_yticks([])
                outside = np.mean(
                    (np.abs(points[:, 0] - center[0]) > radius)
                    | (np.abs(points[:, 1] - center[1]) > radius)
                )
                if outside > 0.005:
                    axis.text(
                        0.03,
                        0.96,
                        f"outside: {100.0 * outside:.1f}%",
                        transform=axis.transAxes,
                        va="top",
                        fontsize=7,
                        color="#9c2f00",
                    )
        fig.suptitle(
            f"JiT-style exact-Bayes comparison: D={D}, H={hidden}, "
            f"sigma_normal={sigma_normal:g}, loss={loss_space}",
            fontsize=13,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        tag = str(sigma_normal).replace(".", "p")
        path = output_dir / f"jit_style_D{D}_H{hidden}_sn{tag}_{loss_space}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)


def write_report(path: Path, baseline: pd.DataFrame, aggregate: pd.DataFrame) -> None:
    lines = [
        "# Prediction-target v5 exact-Bayes summary",
        "",
        f"Completed baseline settings: {len(baseline)}",
        f"Completed intervention settings: {len(aggregate)}",
        "",
        "Interpretation guardrail: tangent/normal values are geometric diagnostics;",
        "success is judged independently by latent and pushforward distribution metrics.",
        "",
        "## Baselines",
        baseline.to_string(index=False),
        "",
        "## Interventions",
        aggregate.to_string(index=False),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary, generation, teacher = load_tables(args.input_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "all_setting_summaries.csv", index=False)
    generation.to_csv(args.output_dir / "all_generation_metrics.csv", index=False)
    teacher.to_csv(args.output_dir / "all_teacher_metrics.csv", index=False)
    baseline = build_baseline_aggregate(summary)
    contrasts = build_condition_contrasts(generation)
    aggregate = aggregate_contrasts(contrasts)
    baseline.to_csv(args.output_dir / "aggregate_baselines.csv", index=False)
    contrasts.to_csv(args.output_dir / "seed_condition_contrasts.csv", index=False)
    aggregate.to_csv(args.output_dir / "aggregate_condition_contrasts.csv", index=False)
    plot_summary(args.output_dir / "summary.png", baseline, aggregate)
    plot_jit_comparisons(args.input_root, args.output_dir)
    write_report(args.output_dir / "final_report.txt", baseline, aggregate)


if __name__ == "__main__":
    main()
