"""Aggregate multi-seed MNIST generation-time bottleneck experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODES = ("all_time", "high_noise", "low_noise", "none")


def _single_run(root: Path, mode: str) -> Path:
    matches = sorted(root.glob(f"{mode}_seed*"))
    if len(matches) != 1:
        raise ValueError(f"expected one {mode} run under {root}, found {len(matches)}")
    return matches[0]


def load_runs(roots: Sequence[Path]) -> dict[str, pd.DataFrame]:
    teacher, rollout, summaries = [], [], []
    for root in roots:
        for mode in MODES:
            run = _single_run(root, mode)
            config = json.loads((run / "config.json").read_text())
            seed = int(config["seed"])
            teacher_table = pd.read_csv(run / "teacher_profile.csv")
            teacher_table.insert(0, "seed", seed)
            teacher.append(teacher_table)
            rollout_table = pd.read_csv(run / "rollout.csv")
            rollout_table.insert(0, "seed", seed)
            rollout.append(rollout_table)
            summary = json.loads((run / "summary.json").read_text())
            summary["seed"] = seed
            summaries.append(summary)
    return {
        "teacher": pd.concat(teacher, ignore_index=True),
        "rollout": pd.concat(rollout, ignore_index=True),
        "summary": pd.DataFrame(summaries),
    }


def seed_gate_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    teacher = tables["teacher"]
    rollout = tables["rollout"]
    summary = tables["summary"]
    records: list[dict[str, float | int | bool]] = []
    for seed in sorted(summary["seed"].unique()):
        seed_teacher = teacher[teacher.seed == seed]
        seed_rollout = rollout[rollout.seed == seed]
        seed_summary = summary[summary.seed == seed]
        real = (
            seed_rollout[seed_rollout.branch == "real"]
            .set_index("mode")["source_class_match"]
        )
        shuffled = (
            seed_rollout[seed_rollout.branch == "shuffle"]
            .set_index("mode")["source_class_match"]
        )
        high_teacher = seed_teacher[seed_teacher["mode"] == "high_noise"]
        none_teacher = seed_teacher[seed_teacher["mode"] == "none"]
        high_active = high_teacher[high_teacher.timestep >= 0.85]
        high_inactive = high_teacher[high_teacher.timestep <= 0.55]
        identity_max = float(
            seed_summary[["identity_absolute_max", "identity_relative_max"]]
            .to_numpy(dtype=float)
            .max()
        )
        classifier_min = float(seed_summary["classifier_accuracy"].min())
        p1 = bool(
            (high_active["delta_shuffle_mean"] > 0.0).all()
            and (high_inactive["delta_shuffle_mean"] == 0.0).all()
        )
        high_advantage = float(real.high_noise - shuffled.high_noise)
        retention = float(real.high_noise / real.all_time)
        p2 = high_advantage >= 0.20 and retention >= 0.90
        p3 = bool(real.low_noise < real.high_noise)
        p4 = bool(
            identity_max == 0.0
            and (none_teacher["delta_shuffle_mean"] == 0.0).all()
            and classifier_min >= 0.98
        )
        records.append(
            {
                "seed": int(seed),
                "high_real_class_match": float(real.high_noise),
                "high_shuffle_class_match": float(shuffled.high_noise),
                "high_advantage": high_advantage,
                "all_real_class_match": float(real.all_time),
                "high_over_all": retention,
                "low_real_class_match": float(real.low_noise),
                "none_real_class_match": float(real.none),
                "classifier_accuracy_min": classifier_min,
                "identity_max": identity_max,
                "p1": p1,
                "p2": p2,
                "p3": p3,
                "p4": p4,
                "prior_gate_pass": bool(p1 and p2 and p3 and p4),
            }
        )
    return pd.DataFrame(records)


def plot_summary(tables: dict[str, pd.DataFrame], output: Path) -> None:
    rollout = tables["rollout"]
    teacher = tables["teacher"]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    branches = ("real", "shuffle")
    width = 0.34
    positions = np.arange(len(MODES))
    for offset, branch in enumerate(branches):
        selected = rollout[rollout.branch == branch]
        means = selected.groupby("mode")["source_class_match"].mean().reindex(MODES)
        stds = selected.groupby("mode")["source_class_match"].std().reindex(MODES).fillna(0.0)
        axes[0].bar(
            positions + (offset - 0.5) * width,
            means,
            width,
            yerr=stds,
            capsize=4,
            label=branch,
        )
    axes[0].set_xticks(positions, MODES, rotation=15)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Source-class match")
    axes[0].set_title("Conditional rollout")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    for mode in ("all_time", "high_noise", "low_noise"):
        selected = teacher[teacher["mode"] == mode]
        grouped = selected.groupby("timestep")["delta_shuffle_mean"].agg(["mean", "std"])
        grouped = grouped.sort_index(ascending=False)
        axes[1].plot(grouped.index, grouped["mean"], marker="o", label=mode)
        axes[1].fill_between(
            grouped.index,
            grouped["mean"] - grouped["std"].fillna(0.0),
            grouped["mean"] + grouped["std"].fillna(0.0),
            alpha=0.15,
        )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].invert_xaxis()
    axes[1].set_xlabel("Flow time t (noise -> image)")
    axes[1].set_ylabel("MSE(shuffle) - MSE(real)")
    axes[1].set_title("Teacher-path responsibility")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tables = load_runs(args.roots)
    gates = seed_gate_table(tables)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(args.output / f"{name}.csv", index=False)
    gates.to_csv(args.output / "seed_gates.csv", index=False)
    plot_summary(tables, args.output / "summary.png")
    print(gates.to_string(index=False))
    print(f"\nAll seeds pass prior gate: {bool(gates.prior_gate_pass.all())}")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
