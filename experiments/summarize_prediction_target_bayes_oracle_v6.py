#!/usr/bin/env python3
"""Aggregate exact-Bayes trajectory runs without hiding failed settings."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.run_prediction_target_bayes_oracle_v5 import save_csv, save_json


METRICS = (
    "latent_swd",
    "latent_mmd_rff",
    "pushforward_swd",
    "pushforward_mmd_rff",
)
STRUCTURE_METRICS = (
    "component_jsd",
    "nearest_tangent_rms",
    "nearest_normal_rms",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def sem(values: list[float]) -> float:
    return (
        float(np.std(values, ddof=1) / np.sqrt(len(values)))
        if len(values) > 1
        else float("nan")
    )


def collect(root: Path) -> tuple[list[dict], list[dict]]:
    summaries: list[dict] = []
    generation: list[dict] = []
    for path in sorted(root.glob("seed*/*/H*/step*/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        summaries.append(summary)
        for row in read_csv(path.parent / "generation_metrics.csv"):
            generation.append(
                {
                    **row,
                    "seed": int(row["seed"]),
                    "step": int(row["step"]),
                    "hidden": int(row["hidden"]),
                    "strength": float(row["strength"]),
                    **{metric: float(row[metric]) for metric in METRICS},
                    **{
                        metric: float(row[metric]) for metric in STRUCTURE_METRICS
                    },
                    "reference_nearest_tangent_rms": float(
                        row["reference_nearest_tangent_rms"]
                    ),
                    "reference_nearest_normal_rms": float(
                        row["reference_nearest_normal_rms"]
                    ),
                }
            )
    return summaries, generation


def aggregate_summary(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["architecture"], int(row["hidden"]), int(row["step"]))
        groups[key].append(row)
    output = []
    numeric = (
        "x_excess_over_bayes_risk",
        "v_excess_over_bayes_risk",
        "x_recent_relative_change",
        "v_recent_relative_change",
        "x_recent_relative_span",
        "v_recent_relative_span",
        "bayes_latent_swd",
        "x_latent_swd",
        "v_latent_swd",
        "bayes_pushforward_swd",
        "x_pushforward_swd",
        "v_pushforward_swd",
        "x_over_bayes_latent_swd",
        "v_over_x_latent_swd",
        "x_over_bayes_pushforward_swd",
        "v_over_x_pushforward_swd",
        "x_tangent_over_reference",
        "x_normal_over_reference",
        "x_over_bayes_component_jsd",
    )
    for (architecture, hidden, step), group in sorted(groups.items()):
        quality_values = [
            float(
                float(item["x_over_bayes_latent_swd"]) <= 1.5
                and float(item["x_over_bayes_pushforward_swd"]) <= 1.5
                and 1.0 < float(item["v_over_x_latent_swd"]) <= 2.5
                and 1.0 < float(item["v_over_x_pushforward_swd"]) <= 2.5
                and 2.0 / 3.0
                <= float(item["x_tangent_over_reference"])
                <= 1.5
                and 2.0 / 3.0
                <= float(item["x_normal_over_reference"])
                <= 1.5
                and float(item["x_over_bayes_component_jsd"]) <= 2.5
                and abs(float(item["x_recent_relative_change"])) <= 0.1
                and abs(float(item["v_recent_relative_change"])) <= 0.1
                and float(item["x_recent_relative_span"]) <= 0.15
                and float(item["v_recent_relative_span"]) <= 0.15
            )
            for item in group
        ]
        row = {
            "architecture": architecture,
            "hidden": hidden,
            "step": step,
            "seeds": len(group),
            "quality_band_fraction": mean(quality_values),
            "candidate_success_fraction": mean(
                [float(item["candidate_success"]) for item in group]
            ),
        }
        for key in numeric:
            values = [float(item[key]) for item in group]
            row[f"mean_{key}"] = mean(values)
            row[f"sem_{key}"] = sem(values)
        output.append(row)
    return output


def aggregate_gammas(rows: list[dict]) -> list[dict]:
    by_setting_seed: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        key = (
            row["architecture"],
            row["hidden"],
            row["step"],
            row["seed"],
        )
        by_setting_seed[key][row["condition"]] = row

    paired: list[dict] = []
    for key, conditions in by_setting_seed.items():
        baseline = conditions.get("x")
        if baseline is None:
            continue
        for condition in conditions.values():
            if condition["kind"] != "xv" or condition["strength"] <= 0:
                continue
            row = {
                "architecture": key[0],
                "hidden": key[1],
                "step": key[2],
                "seed": key[3],
                "gamma": condition["strength"],
            }
            wins = 0
            for metric in METRICS:
                base = baseline[metric]
                value = condition[metric]
                improvement = (base - value) / max(abs(base), 1e-12)
                row[f"relative_improvement_{metric}"] = improvement
                wins += int(improvement > 0)
            row["all_four_improved"] = int(wins == len(METRICS))
            structure_improvements = {
                "component_jsd": (
                    baseline["component_jsd"] - condition["component_jsd"]
                )
                / max(abs(baseline["component_jsd"]), 1e-12),
                "nearest_tangent_rms": (
                    abs(
                        baseline["nearest_tangent_rms"]
                        - baseline["reference_nearest_tangent_rms"]
                    )
                    - abs(
                        condition["nearest_tangent_rms"]
                        - baseline["reference_nearest_tangent_rms"]
                    )
                )
                / max(
                    abs(
                        baseline["nearest_tangent_rms"]
                        - baseline["reference_nearest_tangent_rms"]
                    ),
                    1e-12,
                ),
                "nearest_normal_rms": (
                    abs(
                        baseline["nearest_normal_rms"]
                        - baseline["reference_nearest_normal_rms"]
                    )
                    - abs(
                        condition["nearest_normal_rms"]
                        - baseline["reference_nearest_normal_rms"]
                    )
                )
                / max(
                    abs(
                        baseline["nearest_normal_rms"]
                        - baseline["reference_nearest_normal_rms"]
                    ),
                    1e-12,
                ),
            }
            for metric, improvement in structure_improvements.items():
                row[f"relative_improvement_{metric}"] = improvement
            row["all_three_structure_improved"] = int(
                all(value > 0 for value in structure_improvements.values())
            )
            row["all_seven_improved"] = int(
                row["all_four_improved"] and row["all_three_structure_improved"]
            )
            paired.append(row)

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in paired:
        key = (
            row["architecture"],
            row["hidden"],
            row["step"],
            row["gamma"],
        )
        groups[key].append(row)
    output = []
    for (architecture, hidden, step, gamma), group in sorted(groups.items()):
        row = {
            "architecture": architecture,
            "hidden": hidden,
            "step": step,
            "gamma": gamma,
            "seeds": len(group),
            "all_four_improved_fraction": mean(
                [item["all_four_improved"] for item in group]
            ),
            "all_three_structure_improved_fraction": mean(
                [item["all_three_structure_improved"] for item in group]
            ),
            "all_seven_improved_fraction": mean(
                [item["all_seven_improved"] for item in group]
            ),
        }
        for metric in METRICS:
            key = f"relative_improvement_{metric}"
            values = [item[key] for item in group]
            row[f"mean_{key}"] = mean(values)
            row[f"sem_{key}"] = sem(values)
            row[f"improved_fraction_{metric}"] = mean(
                [float(value > 0) for value in values]
            )
        for metric in STRUCTURE_METRICS:
            key = f"relative_improvement_{metric}"
            values = [item[key] for item in group]
            row[f"mean_{key}"] = mean(values)
            row[f"sem_{key}"] = sem(values)
            row[f"improved_fraction_{metric}"] = mean(
                [float(value > 0) for value in values]
            )
        output.append(row)
    return output


def candidate_rows(summary: list[dict], gamma: list[dict]) -> list[dict]:
    quality = {
        (row["architecture"], row["hidden"], row["step"]): row
        for row in summary
    }
    output = []
    for row in gamma:
        key = (row["architecture"], row["hidden"], row["step"])
        base = quality[key]
        all_means_positive = all(
            row[f"mean_relative_improvement_{metric}"] > 0
            for metric in METRICS + STRUCTURE_METRICS
        )
        accepted = (
            row["seeds"] >= 4
            and base["quality_band_fraction"] >= 0.75
            and row["all_seven_improved_fraction"] >= 0.75
            and all_means_positive
        )
        if accepted:
            output.append(
                {
                    **row,
                    "quality_band_fraction": base["quality_band_fraction"],
                    "discovery_only": True,
                }
            )
    return output


def plot_trajectory(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), constrained_layout=True)
    colors = {"bayes": "#222222", "x": "#2a6fbb", "v": "#d0782a"}
    for (architecture, hidden), group in _groups(rows).items():
        group = sorted(group, key=lambda row: row["step"])
        label = f"{architecture}, H={hidden}"
        steps = [row["step"] for row in group]
        axes[0, 0].plot(
            steps,
            [row["mean_x_over_bayes_latent_swd"] for row in group],
            marker="o",
            label=label,
        )
        axes[0, 1].plot(
            steps,
            [row["mean_v_over_x_latent_swd"] for row in group],
            marker="o",
            label=label,
        )
        for target in ("bayes", "x", "v"):
            axes[1, 0].plot(
                steps,
                [row[f"mean_{target}_latent_swd"] for row in group],
                marker="o",
                color=colors[target],
                alpha=0.8,
                label=f"{label}: {target}",
            )
        axes[1, 1].plot(
            steps,
            [row["quality_band_fraction"] for row in group],
            marker="o",
            label=label,
        )
    axes[0, 0].axhline(1.5, color="#555555", linestyle="--", linewidth=1)
    axes[0, 0].set_title("x quality relative to exact-Bayes rollout")
    axes[0, 0].set_ylabel("x SWD / Bayes SWD")
    axes[0, 1].axhline(1.0, color="#555555", linestyle="--", linewidth=1)
    axes[0, 1].set_title("Weak/strong separation")
    axes[0, 1].set_ylabel("v SWD / x SWD")
    axes[1, 0].set_title("Absolute latent distribution error")
    axes[1, 0].set_ylabel("SWD")
    axes[1, 1].set_title("Fraction of seeds inside predeclared quality band")
    axes[1, 1].set_ylabel("fraction")
    axes[1, 1].set_ylim(-0.02, 1.02)
    for axis in axes.flat:
        axis.set_xlabel("training step")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    fig.suptitle("Prediction-target trajectory: convergence before extrapolation")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190)
    plt.close(fig)


def plot_gamma_heatmaps(output_dir: Path, rows: list[dict]) -> None:
    if not rows:
        return
    panels = (
        ("latent_swd", "Latent SWD"),
        ("latent_mmd_rff", "Latent MMD"),
        ("pushforward_swd", "Pushforward SWD"),
        ("pushforward_mmd_rff", "Pushforward MMD"),
        ("component_jsd", "Component JSD"),
        ("nearest_tangent_rms", "Tangent-width error"),
        ("nearest_normal_rms", "Normal-width error"),
    )
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["architecture"], row["hidden"])].append(row)
    for (architecture, hidden), group in grouped.items():
        steps = sorted({int(row["step"]) for row in group})
        gammas = sorted({float(row["gamma"]) for row in group})
        lookup = {
            (int(row["step"]), float(row["gamma"])): row for row in group
        }
        fig, axes = plt.subplots(2, 4, figsize=(19, 8.8), constrained_layout=True)
        for axis, (metric, title) in zip(axes.flat[:7], panels):
            matrix = np.full((len(steps), len(gammas)), np.nan)
            for row_index, step in enumerate(steps):
                for column_index, gamma in enumerate(gammas):
                    row = lookup.get((step, gamma))
                    if row is not None:
                        matrix[row_index, column_index] = 100.0 * float(
                            row[f"mean_relative_improvement_{metric}"]
                        )
            finite = np.abs(matrix[np.isfinite(matrix)])
            vmax = max(float(np.quantile(finite, 0.95)) if len(finite) else 1.0, 0.05)
            image = axis.imshow(
                matrix,
                aspect="auto",
                cmap="RdBu",
                vmin=-vmax,
                vmax=vmax,
                interpolation="nearest",
            )
            axis.set_title(title)
            axis.set_xticks(range(len(gammas)), [f"{gamma:g}" for gamma in gammas])
            axis.set_yticks(range(len(steps)), [str(step) for step in steps])
            axis.set_xlabel("gamma")
            axis.set_ylabel("training step")
            for row_index in range(len(steps)):
                for column_index in range(len(gammas)):
                    value = matrix[row_index, column_index]
                    if np.isfinite(value):
                        axis.text(
                            column_index,
                            row_index,
                            f"{value:+.1f}",
                            ha="center",
                            va="center",
                            fontsize=7,
                            color="white" if abs(value) > 0.62 * vmax else "black",
                        )
            fig.colorbar(image, ax=axis, shrink=0.78, label="improvement vs x (%)")

        axis = axes.flat[7]
        matrix = np.full((len(steps), len(gammas)), np.nan)
        for row_index, step in enumerate(steps):
            for column_index, gamma in enumerate(gammas):
                row = lookup.get((step, gamma))
                if row is not None:
                    matrix[row_index, column_index] = float(
                        row["all_seven_improved_fraction"]
                    )
        image = axis.imshow(
            matrix,
            aspect="auto",
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        axis.set_title("All seven metrics improve")
        axis.set_xticks(range(len(gammas)), [f"{gamma:g}" for gamma in gammas])
        axis.set_yticks(range(len(steps)), [str(step) for step in steps])
        axis.set_xlabel("gamma")
        axis.set_ylabel("training step")
        for row_index in range(len(steps)):
            for column_index in range(len(gammas)):
                value = matrix[row_index, column_index]
                if np.isfinite(value):
                    axis.text(
                        column_index,
                        row_index,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if value > 0.62 else "black",
                    )
        fig.colorbar(image, ax=axis, shrink=0.78, label="seed fraction")
        fig.suptitle(
            f"Wide-gamma extrapolation atlas: {architecture}, H={hidden}\n"
            "Positive values mean improvement over the x-prediction rollout",
            fontsize=14,
        )
        path = output_dir / f"gamma_heatmap_{architecture}_H{hidden}.png"
        fig.savefig(path, dpi=190)
        plt.close(fig)


def _groups(rows: list[dict]) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["architecture"], row["hidden"])].append(row)
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries, generation = collect(args.input_root)
    summary_rows = aggregate_summary(summaries)
    gamma_rows = aggregate_gammas(generation)
    candidates = candidate_rows(summary_rows, gamma_rows)
    save_csv(args.output_dir / "trajectory_summary.csv", summary_rows)
    save_csv(args.output_dir / "gamma_summary.csv", gamma_rows)
    save_csv(args.output_dir / "discovery_candidates.csv", candidates)
    save_json(
        args.output_dir / "report.json",
        {
            "settings": len(summary_rows),
            "milestone_runs": len(summaries),
            "discovery_candidates": len(candidates),
            "warning": (
                "Candidates are selected on these seeds and require a frozen "
                "gamma/configuration test on new seeds."
            ),
        },
    )
    plot_trajectory(args.output_dir / "trajectory.png", summary_rows)
    plot_gamma_heatmaps(args.output_dir, gamma_rows)


if __name__ == "__main__":
    main()
