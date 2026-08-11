#!/usr/bin/env python3
"""Aggregate the dual-target closed-loop toy across dimensions and seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TABLE_FILES = {
    "endpoint": "endpoint_metrics.csv",
    "teacher": "teacher_metrics.csv",
    "rollout": "rollout_metrics.csv",
    "branch": "branch_pair_metrics.csv",
    "gradient": "gradient_audit.csv",
    "cross_gate": "cross_gate_endpoint_metrics.csv",
    "cross_gate_teacher": "cross_gate_teacher_metrics.csv",
    "cross_gate_rollout": "cross_gate_rollout_metrics.csv",
}


def setting_metadata(path: Path) -> tuple[int, int, int]:
    setting = path.parent.name
    seed = int(path.parent.parent.name.removeprefix("seed"))
    dimension = int(setting.split("_")[0].removeprefix("D"))
    hidden = int(setting.split("_")[1].removeprefix("H"))
    return seed, dimension, hidden


def load_tables(root: Path) -> dict[str, pd.DataFrame]:
    tables: dict[str, list[pd.DataFrame]] = {key: [] for key in TABLE_FILES}
    for table_name, filename in TABLE_FILES.items():
        for path in sorted(root.glob(f"seed*/D*_H*/{filename}")):
            seed, dimension, hidden = setting_metadata(path)
            frame = pd.read_csv(path)
            frame.insert(0, "hidden_dim", hidden)
            frame.insert(0, "ambient_dim", dimension)
            frame.insert(0, "seed", seed)
            tables[table_name].append(frame)
    return {
        key: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        for key, frames in tables.items()
    }


def aggregate_numeric(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    numeric = [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column not in set(group_columns + ["seed"])
    ]
    grouped = frame.groupby(group_columns, dropna=False)[numeric]
    mean = grouped.mean().add_suffix("_mean")
    std = grouped.std(ddof=1).fillna(0.0).add_suffix("_std")
    count = frame.groupby(group_columns, dropna=False)["seed"].nunique().rename("seeds")
    return pd.concat((count, mean, std), axis=1).reset_index()


def endpoint_contrasts(endpoint: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (seed, dimension), frame in endpoint.groupby(["seed", "ambient_dim"]):
        by_condition = frame.set_index("condition")

        def metric(condition: str, name: str) -> float:
            if condition not in by_condition.index:
                return float("nan")
            return float(by_condition.loc[condition, name])

        independent = ["B0_v_ind", "B1_x_ind", "B2_eps_ind"]
        shared_branches = ["D0_x_shared", "D0_eps_shared"]
        best_independent_ambient = min(
            metric(condition, "ambient_swd") for condition in independent
        )
        best_shared_ambient = min(
            metric(condition, "ambient_swd") for condition in shared_branches
        )
        oracle_ambient = metric("D3_oracle_bayes_gate", "ambient_swd")
        rows.append(
            {
                "seed": int(seed),
                "ambient_dim": int(dimension),
                "best_independent_ambient_swd": best_independent_ambient,
                "best_shared_branch_ambient_swd": best_shared_ambient,
                "oracle_ambient_swd": oracle_ambient,
                "oracle_delta_vs_best_independent": (
                    oracle_ambient - best_independent_ambient
                ),
                "oracle_delta_vs_best_shared_branch": (
                    oracle_ambient - best_shared_ambient
                ),
                "oracle_improves_best_shared_branch": bool(
                    oracle_ambient < best_shared_ambient
                ),
                "analytic_safe_schedule_ambient_swd": metric(
                    "D0_safe_schedule", "ambient_swd"
                ),
                "scaled_gate_ambient_swd": metric("D1_scaled_gate", "ambient_swd"),
                "velocity_gate_ambient_swd": metric("D2_velocity_gate", "ambient_swd"),
                "safe_gate_ambient_swd": metric(
                    "D4_safe_velocity_gate", "ambient_swd"
                ),
                "sc_no_consistency_ambient_swd": metric("S0_xv_switch", "ambient_swd"),
                "sc_consistency_ambient_swd": metric(
                    "S1_xv_consistency_switch", "ambient_swd"
                ),
                "oracle_intrinsic_swd": metric(
                    "D3_oracle_bayes_gate", "intrinsic_swd"
                ),
                "oracle_off_subspace_rms": metric(
                    "D3_oracle_bayes_gate", "off_subspace_rms"
                ),
            }
        )
    return pd.DataFrame(rows)


def teacher_oracle_contrasts(teacher: pd.DataFrame) -> pd.DataFrame:
    required = {"D0_x_shared", "D0_eps_shared", "D3_oracle_bayes_gate"}
    rows: list[dict] = []
    keys = ["seed", "ambient_dim", "time"]
    for key, frame in teacher.groupby(keys):
        by_condition = frame.set_index("condition")
        if not required.issubset(by_condition.index):
            continue
        x_error = float(by_condition.loc["D0_x_shared", "bayes_velocity_mse"])
        epsilon_error = float(
            by_condition.loc["D0_eps_shared", "bayes_velocity_mse"]
        )
        oracle_error = float(
            by_condition.loc["D3_oracle_bayes_gate", "bayes_velocity_mse"]
        )
        best = min(x_error, epsilon_error)
        rows.append(
            {
                "seed": int(key[0]),
                "ambient_dim": int(key[1]),
                "time": float(key[2]),
                "x_bayes_mse": x_error,
                "epsilon_bayes_mse": epsilon_error,
                "best_branch_bayes_mse": best,
                "oracle_bayes_mse": oracle_error,
                "oracle_over_best_branch": oracle_error / max(best, 1e-12),
                "oracle_improves_best_branch": bool(oracle_error <= best + 1e-8),
            }
        )
    return pd.DataFrame(rows)


def mechanism_classification(
    endpoint_contrast: pd.DataFrame,
    teacher_contrast: pd.DataFrame,
) -> pd.DataFrame:
    teacher_summary = (
        teacher_contrast.groupby(["seed", "ambient_dim"])
        .agg(
            teacher_oracle_over_best_mean=("oracle_over_best_branch", "mean"),
            teacher_oracle_wins_all_times=("oracle_improves_best_branch", "all"),
        )
        .reset_index()
    )
    merged = endpoint_contrast.merge(
        teacher_summary, on=["seed", "ambient_dim"], validate="one_to_one"
    )

    def classify(row: pd.Series) -> str:
        if not bool(row["teacher_oracle_wins_all_times"]):
            return "invalid_oracle_check"
        if bool(row["oracle_improves_best_shared_branch"]):
            return "A_teacher_and_closed_loop_improve"
        return "B_teacher_improves_closed_loop_worsens"

    merged["case"] = merged.apply(classify, axis=1)
    return merged


def plot_endpoint_summary(path: Path, endpoint_summary: pd.DataFrame) -> None:
    conditions = [
        "Reference_resample",
        "Bayes_exact",
        "B0_v_ind",
        "B1_x_ind",
        "B2_eps_ind",
        "D0_fixed_x_eps",
        "D0_safe_schedule",
        "D1_scaled_gate",
        "D2_velocity_gate",
        "D3_oracle_bayes_gate",
        "D4_safe_velocity_gate",
        "S0_xv_switch",
        "S1_xv_consistency_switch",
    ]
    dimensions = sorted(endpoint_summary["ambient_dim"].unique())
    figure, axes = plt.subplots(1, len(dimensions), figsize=(9 * len(dimensions), 6))
    if len(dimensions) == 1:
        axes = [axes]
    for axis, dimension in zip(axes, dimensions):
        frame = endpoint_summary[endpoint_summary["ambient_dim"] == dimension]
        frame = frame.set_index("condition").reindex(conditions).dropna(how="all")
        position = np.arange(len(frame))
        axis.bar(
            position,
            frame["ambient_swd_mean"],
            yerr=frame["ambient_swd_std"],
            capsize=3,
        )
        axis.set_xticks(position, frame.index, rotation=55, ha="right")
        axis.set_ylabel("Ambient SWD (lower is better)")
        axis.set_yscale("log")
        axis.set_title(f"D={dimension}")
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_teacher_rollout(
    path: Path,
    teacher_summary: pd.DataFrame,
    rollout_summary: pd.DataFrame,
) -> None:
    conditions = [
        "B0_v_ind",
        "B1_x_ind",
        "D1_scaled_gate",
        "D2_velocity_gate",
        "D3_oracle_bayes_gate",
        "D4_safe_velocity_gate",
        "S1_xv_consistency_switch",
    ]
    dimensions = sorted(teacher_summary["ambient_dim"].unique())
    figure, axes = plt.subplots(len(dimensions), 2, figsize=(15, 5.5 * len(dimensions)))
    axes = np.asarray(axes).reshape(len(dimensions), 2)
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
            if not teacher.empty:
                axes[row_index, 0].plot(
                    teacher["time"],
                    teacher["bayes_velocity_mse_mean"],
                    marker="o",
                    label=condition,
                )
            if not rollout.empty:
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
    axes[-1, 1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    output = args.root / "aggregate"
    output.mkdir(parents=True, exist_ok=True)
    tables = load_tables(args.root)
    required = ("endpoint", "teacher", "rollout")
    missing = [name for name in required if tables[name].empty]
    if missing:
        raise RuntimeError(f"missing result tables: {missing}")

    for name, frame in tables.items():
        if not frame.empty:
            frame.to_csv(output / f"{name}_all.csv", index=False, lineterminator="\n")

    endpoint_summary = aggregate_numeric(
        tables["endpoint"], ["ambient_dim", "hidden_dim", "condition"]
    )
    teacher_summary = aggregate_numeric(
        tables["teacher"], ["ambient_dim", "hidden_dim", "time", "condition"]
    )
    rollout_summary = aggregate_numeric(
        tables["rollout"], ["ambient_dim", "hidden_dim", "time", "condition"]
    )
    endpoint_summary.to_csv(
        output / "endpoint_seed_summary.csv", index=False, lineterminator="\n"
    )
    teacher_summary.to_csv(
        output / "teacher_seed_summary.csv", index=False, lineterminator="\n"
    )
    rollout_summary.to_csv(
        output / "rollout_seed_summary.csv", index=False, lineterminator="\n"
    )
    if not tables["branch"].empty:
        branch_summary = aggregate_numeric(
            tables["branch"],
            ["ambient_dim", "hidden_dim", "time", "model", "model_kind"],
        )
        branch_summary.to_csv(
            output / "branch_seed_summary.csv", index=False, lineterminator="\n"
        )
    if not tables["cross_gate"].empty:
        cross_gate_summary = aggregate_numeric(
            tables["cross_gate"], ["ambient_dim", "hidden_dim", "condition"]
        )
        cross_gate_summary.to_csv(
            output / "cross_gate_seed_summary.csv", index=False, lineterminator="\n"
        )
    for name, metric in (
        ("cross_gate_teacher", "cross_gate_teacher_seed_summary.csv"),
        ("cross_gate_rollout", "cross_gate_rollout_seed_summary.csv"),
    ):
        if tables[name].empty:
            continue
        summary = aggregate_numeric(
            tables[name], ["ambient_dim", "hidden_dim", "time", "condition"]
        )
        summary.to_csv(output / metric, index=False, lineterminator="\n")

    endpoint_contrast = endpoint_contrasts(tables["endpoint"])
    teacher_contrast = teacher_oracle_contrasts(tables["teacher"])
    classification = mechanism_classification(endpoint_contrast, teacher_contrast)
    endpoint_contrast.to_csv(
        output / "endpoint_contrasts.csv", index=False, lineterminator="\n"
    )
    teacher_contrast.to_csv(
        output / "teacher_oracle_contrasts.csv", index=False, lineterminator="\n"
    )
    classification.to_csv(
        output / "mechanism_classification.csv", index=False, lineterminator="\n"
    )
    (output / "mechanism_classification.json").write_text(
        json.dumps(classification.to_dict(orient="records"), indent=2) + "\n",
        encoding="utf-8",
    )
    plot_endpoint_summary(output / "endpoint_ambient_swd.png", endpoint_summary)
    plot_teacher_rollout(
        output / "teacher_vs_rollout_bayes_error.png",
        teacher_summary,
        rollout_summary,
    )
    print(classification.to_string(index=False))


if __name__ == "__main__":
    main()
