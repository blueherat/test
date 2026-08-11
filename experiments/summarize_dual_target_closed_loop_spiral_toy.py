#!/usr/bin/env python3
"""Create spiral-specific cross-seed tables and figures.

The common closed-loop summarizer focuses on SWD and Bayes-field error.  This
companion keeps the v10 geometry diagnostics visible: ridge fidelity, ridge
width, arc coverage, and ambient-surface deviation.  It consumes only the CSV
files produced by ``run_dual_target_closed_loop_spiral_toy.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import summarize_dual_target_closed_loop_toy as common


CONDITION_ORDER = [
    "Reference_resample",
    "Bayes_exact",
    "B0_v_ind",
    "B1_x_ind",
    "B2_eps_ind",
    "D0_x_shared",
    "D0_eps_shared",
    "D0_fixed_x_eps",
    "D0_safe_schedule",
    "D1_scaled_gate",
    "D2_velocity_gate",
    "D3_oracle_bayes_gate",
    "D4_safe_velocity_gate",
    "S0_xv_switch",
    "S1_xv_consistency_switch",
]

CROSS_CONDITION_ORDER = [
    "D0_x_shared",
    "D3_oracle_bayes_gate",
    "D1_gate_on_D0",
    "D2_gate_on_D0",
    "D4_gate_on_D0",
]

METRICS = [
    ("swd_2d", "Intrinsic SWD", "log", None),
    ("swd_fullD", "Full-D SWD", "log", None),
    ("ridge_distance_mean", "Distance to spiral ridge", "log", None),
    ("ridge_width_ratio", "Ridge width / reference", "linear", 1.0),
    ("arc_hist_tv", "Arc coverage TV", "linear", 0.0),
    ("ambient_surface_rms", "Ambient surface RMS", "log", None),
]


def require_columns(frame: pd.DataFrame, columns: set[str], table_name: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{table_name} is missing columns: {missing}")


def summarize_endpoint(frame: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        frame,
        {
            "seed",
            "ambient_dim",
            "hidden_dim",
            "condition",
            *(metric for metric, _, _, _ in METRICS),
        },
        "endpoint",
    )
    return common.aggregate_numeric(
        frame, ["ambient_dim", "hidden_dim", "condition"]
    )


def build_mechanism_table(
    endpoint: pd.DataFrame,
    cross_endpoint: pd.DataFrame,
) -> pd.DataFrame:
    """Make one compact per-seed table for the central closed-loop claims."""
    require_columns(
        endpoint,
        {"seed", "ambient_dim", "condition", "swd_2d", "swd_fullD"},
        "endpoint",
    )
    rows: list[dict] = []
    for (seed, dimension), frame in endpoint.groupby(["seed", "ambient_dim"]):
        indexed = frame.set_index("condition")

        def get(condition: str, metric: str) -> float:
            if condition not in indexed.index:
                return float("nan")
            return float(indexed.loc[condition, metric])

        best_shared_full = min(
            get("D0_x_shared", "swd_fullD"),
            get("D0_eps_shared", "swd_fullD"),
        )
        best_shared_2d = min(
            get("D0_x_shared", "swd_2d"),
            get("D0_eps_shared", "swd_2d"),
        )
        row = {
            "seed": int(seed),
            "ambient_dim": int(dimension),
            "reference_fullD_swd": get("Reference_resample", "swd_fullD"),
            "reference_2d_swd": get("Reference_resample", "swd_2d"),
            "best_shared_fullD_swd": best_shared_full,
            "best_shared_2d_swd": best_shared_2d,
            "oracle_fullD_swd": get("D3_oracle_bayes_gate", "swd_fullD"),
            "oracle_2d_swd": get("D3_oracle_bayes_gate", "swd_2d"),
            "oracle_over_best_shared_fullD": (
                get("D3_oracle_bayes_gate", "swd_fullD")
                / max(best_shared_full, 1e-12)
            ),
            "oracle_over_best_shared_2d": (
                get("D3_oracle_bayes_gate", "swd_2d")
                / max(best_shared_2d, 1e-12)
            ),
        }

        cross = cross_endpoint[
            (cross_endpoint["seed"] == seed)
            & (cross_endpoint["ambient_dim"] == dimension)
        ]
        cross_indexed = cross.set_index("condition")
        for condition in ("D1_gate_on_D0", "D2_gate_on_D0", "D4_gate_on_D0"):
            for metric in ("swd_2d", "swd_fullD"):
                key = f"{condition}_{metric}"
                row[key] = (
                    float(cross_indexed.loc[condition, metric])
                    if condition in cross_indexed.index
                    else float("nan")
                )
        rows.append(row)
    return pd.DataFrame(rows)


def _ordered(frame: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    present = [condition for condition in order if condition in set(frame["condition"])]
    return frame.set_index("condition").reindex(present).reset_index()


def plot_geometry_atlas(path: Path, summary: pd.DataFrame) -> None:
    dimensions = sorted(summary["ambient_dim"].unique())
    figure, axes = plt.subplots(
        len(dimensions),
        len(METRICS),
        figsize=(5.2 * len(METRICS), 5.2 * len(dimensions)),
        squeeze=False,
    )
    for row_index, dimension in enumerate(dimensions):
        frame = _ordered(summary[summary["ambient_dim"] == dimension], CONDITION_ORDER)
        positions = np.arange(len(frame))
        for column_index, (metric, title, scale, target) in enumerate(METRICS):
            axis = axes[row_index, column_index]
            mean = frame[f"{metric}_mean"].to_numpy()
            std = frame[f"{metric}_std"].to_numpy()
            axis.errorbar(positions, mean, yerr=std, marker="o", capsize=2)
            if scale == "log" and np.all(mean > 0):
                axis.set_yscale("log")
            if target is not None:
                axis.axhline(target, color="black", linestyle="--", linewidth=1)
            axis.set_title(f"D={dimension}: {title}")
            axis.set_xticks(positions, frame["condition"], rotation=65, ha="right")
            axis.grid(alpha=0.25)
    figure.suptitle("Continuous-spiral endpoint geometry (mean +/- seed std)", y=1.002)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_common_head_gate_controls(path: Path, summary: pd.DataFrame) -> None:
    dimensions = sorted(summary["ambient_dim"].unique())
    metrics = [
        ("swd_2d", "Intrinsic SWD"),
        ("swd_fullD", "Full-D SWD"),
        ("ridge_distance_mean", "Distance to spiral ridge"),
        ("arc_hist_tv", "Arc coverage TV"),
    ]
    figure, axes = plt.subplots(
        len(dimensions), len(metrics), figsize=(5.5 * len(metrics), 5 * len(dimensions)),
        squeeze=False,
    )
    for row_index, dimension in enumerate(dimensions):
        frame = _ordered(
            summary[summary["ambient_dim"] == dimension], CROSS_CONDITION_ORDER
        )
        positions = np.arange(len(frame))
        for column_index, (metric, title) in enumerate(metrics):
            axis = axes[row_index, column_index]
            axis.errorbar(
                positions,
                frame[f"{metric}_mean"],
                yerr=frame[f"{metric}_std"],
                fmt="o",
                capsize=3,
            )
            axis.set_title(f"D={dimension}: {title}")
            axis.set_xticks(positions, frame["condition"], rotation=55, ha="right")
            if metric in {"swd_2d", "swd_fullD"}:
                axis.set_yscale("log")
            axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Gate transfer on the same D0 x/epsilon heads", y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_common_head_teacher_rollout(
    path: Path,
    teacher_summary: pd.DataFrame,
    rollout_summary: pd.DataFrame,
) -> None:
    conditions = [
        "D0_x_shared",
        "D1_gate_on_D0",
        "D2_gate_on_D0",
        "D3_oracle_bayes_gate",
        "D4_gate_on_D0",
    ]
    dimensions = sorted(teacher_summary["ambient_dim"].unique())
    figure, axes = plt.subplots(
        len(dimensions), 2, figsize=(15, 5.4 * len(dimensions)), squeeze=False
    )
    for row_index, dimension in enumerate(dimensions):
        for condition in conditions:
            teacher = teacher_summary[
                (teacher_summary["ambient_dim"] == dimension)
                & (teacher_summary["condition"] == condition)
            ].sort_values("time")
            rollout = rollout_summary[
                (rollout_summary["ambient_dim"] == dimension)
                & (rollout_summary["condition"] == condition)
            ].sort_values("time")
            axes[row_index, 0].plot(
                teacher["time"],
                teacher["bayes_velocity_mse_mean"],
                marker="o",
                label=condition,
            )
            axes[row_index, 1].plot(
                rollout["time"],
                rollout["rollout_bayes_velocity_mse_mean"],
                marker="o",
                label=condition,
            )
        axes[row_index, 0].set_title(f"D={dimension}: teacher states")
        axes[row_index, 1].set_title(f"D={dimension}: rollout states")
        for axis in axes[row_index]:
            axis.set_yscale("log")
            axis.set_xlabel("t")
            axis.set_ylabel("MSE to exact Bayes field")
            axis.grid(alpha=0.25)
    axes[-1, 1].legend(
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
    )
    figure.suptitle("Same-head gate errors: local fit versus closed-loop states", y=0.995)
    figure.tight_layout(rect=(0, 0.03, 1, 0.97))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    tables = common.load_tables(args.root)
    if tables["endpoint"].empty:
        raise RuntimeError("no endpoint_metrics.csv files found")
    if tables["cross_gate"].empty:
        raise RuntimeError("no cross_gate_endpoint_metrics.csv files found")

    output = args.root / "aggregate"
    output.mkdir(parents=True, exist_ok=True)
    endpoint_summary = summarize_endpoint(tables["endpoint"])
    cross_summary = summarize_endpoint(tables["cross_gate"])
    cross_teacher_summary = common.aggregate_numeric(
        tables["cross_gate_teacher"],
        ["ambient_dim", "hidden_dim", "time", "condition"],
    )
    cross_rollout_summary = common.aggregate_numeric(
        tables["cross_gate_rollout"],
        ["ambient_dim", "hidden_dim", "time", "condition"],
    )
    mechanism = build_mechanism_table(tables["endpoint"], tables["cross_gate"])
    mechanism_summary = common.aggregate_numeric(mechanism, ["ambient_dim"])

    endpoint_summary.to_csv(
        output / "spiral_geometry_seed_summary.csv", index=False, lineterminator="\n"
    )
    cross_summary.to_csv(
        output / "spiral_cross_gate_seed_summary.csv", index=False, lineterminator="\n"
    )
    mechanism.to_csv(
        output / "spiral_mechanism_by_seed.csv", index=False, lineterminator="\n"
    )
    mechanism_summary.to_csv(
        output / "spiral_mechanism_seed_summary.csv", index=False, lineterminator="\n"
    )
    plot_geometry_atlas(output / "spiral_endpoint_geometry_atlas.png", endpoint_summary)
    plot_common_head_gate_controls(
        output / "spiral_common_head_gate_controls.png", cross_summary
    )
    plot_common_head_teacher_rollout(
        output / "spiral_common_head_teacher_vs_rollout.png",
        cross_teacher_summary,
        cross_rollout_summary,
    )
    print(mechanism_summary.to_string(index=False))


if __name__ == "__main__":
    main()
