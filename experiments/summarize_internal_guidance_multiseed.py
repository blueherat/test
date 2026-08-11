#!/usr/bin/env python3
"""Aggregate fixed Internal Guidance settings across independent toy seeds."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_CONDITIONS = (
    "ig_w1",
    "ig_w1.9",
    "ig_w1.9_mid03_07",
    "ig_w2.3",
    "ig_w2.3_high03",
    "ig_w2.3_mid03_07",
)
METRICS = (
    "latent_swd",
    "pushforward_swd",
    "intrinsic_bridge_rate",
    "mean_adjacent_log_density_contrast",
    "component_jsd_y",
    "occupied_components",
)


def load_runs(
    run_dirs: list[Path], conditions: tuple[str, ...] = DEFAULT_CONDITIONS
) -> pd.DataFrame:
    rows = []
    for run_dir in run_dirs:
        frame = pd.read_csv(run_dir / "summary.csv")
        selected = frame[frame.condition.isin(conditions)].copy()
        missing = sorted(set(conditions) - set(selected.condition))
        if missing:
            raise ValueError(f"{run_dir} is missing conditions: {missing}")
        selected["run_dir"] = str(run_dir.resolve())
        rows.append(selected)
    combined = pd.concat(rows, ignore_index=True)
    if combined.groupby("seed").size().nunique() != 1:
        raise ValueError("seeds do not contain the same number of conditions")
    return combined


def paired_deltas(frame: pd.DataFrame, baseline: str = "ig_w1") -> pd.DataFrame:
    baseline_frame = (
        frame[frame.condition == baseline]
        .set_index("seed")
        .loc[:, list(METRICS)]
    )
    rows = []
    for _, row in frame[frame.condition != baseline].iterrows():
        base = baseline_frame.loc[row.seed]
        record = {
            "seed": int(row.seed),
            "condition": row.condition,
        }
        for metric in METRICS:
            value = float(row[metric])
            baseline_value = float(base[metric])
            record[f"{metric}_baseline"] = baseline_value
            record[f"{metric}_guided"] = value
            record[f"{metric}_delta"] = value - baseline_value
            record[f"{metric}_ratio"] = (
                value / baseline_value
                if abs(baseline_value) > 1e-12
                else np.nan
            )
        rows.append(record)
    return pd.DataFrame(rows)


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby("condition", sort=False)
    rows = []
    for condition, group in grouped:
        row: dict[str, float | int | str] = {
            "condition": condition,
            "seeds": int(group.seed.nunique()),
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def consistency_table(deltas: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition, group in deltas.groupby("condition", sort=False):
        rows.append(
            {
                "condition": condition,
                "seeds": int(group.seed.nunique()),
                "bridge_improved_seeds": int(
                    (group.intrinsic_bridge_rate_delta < 0).sum()
                ),
                "contrast_improved_seeds": int(
                    (group.mean_adjacent_log_density_contrast_delta > 0).sum()
                ),
                "latent_swd_improved_seeds": int(
                    (group.latent_swd_delta < 0).sum()
                ),
                "latent_swd_within_10pct_seeds": int(
                    (group.latent_swd_ratio <= 1.10).sum()
                ),
                "all_modes_retained_seeds": int(
                    (
                        group.occupied_components_guided
                        >= group.occupied_components_baseline
                    ).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_paired(frame: pd.DataFrame, path: Path) -> None:
    conditions = [
        "ig_w1",
        "ig_w1.9_mid03_07",
        "ig_w2.3_mid03_07",
    ]
    labels = ["w=1", "w=1.9, [0.3,0.7]", "w=2.3, [0.3,0.7]"]
    metrics = [
        ("latent_swd", "Latent SWD", "lower is closer"),
        ("intrinsic_bridge_rate", "Bridge rate", "lower is sharper"),
        (
            "mean_adjacent_log_density_contrast",
            "Peak/valley contrast",
            "higher is sharper",
        ),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for axis, (metric, title, direction) in zip(axes, metrics):
        pivot = frame.pivot(index="seed", columns="condition", values=metric)
        for seed, row in pivot.iterrows():
            values = [float(row[condition]) for condition in conditions]
            axis.plot(
                range(len(conditions)),
                values,
                marker="o",
                linewidth=1.7,
                alpha=0.78,
                label=str(seed),
            )
        axis.set_xticks(range(len(labels)), labels, rotation=20, ha="right")
        axis.set_title(f"{title}\n({direction})")
        axis.grid(alpha=0.22)
    axes[0].legend(title="seed", fontsize=8)
    figure.suptitle(
        "Fixed Internal Guidance settings across independent training seeds",
        fontsize=15,
    )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, action="append", required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_runs(args.run_dir)
    deltas = paired_deltas(frame)
    aggregate_frame = aggregate(frame)
    consistency = consistency_table(deltas)
    frame.to_csv(output_dir / "per_seed_metrics.csv", index=False)
    deltas.to_csv(output_dir / "paired_deltas_vs_ig_w1.csv", index=False)
    aggregate_frame.to_csv(output_dir / "aggregate_metrics.csv", index=False)
    consistency.to_csv(output_dir / "direction_consistency.csv", index=False)
    plot_paired(frame, output_dir / "multiseed_paired_metrics.png")
    print(aggregate_frame.to_string(index=False))
    print("\nDirection consistency:")
    print(consistency.to_string(index=False))


if __name__ == "__main__":
    main()
