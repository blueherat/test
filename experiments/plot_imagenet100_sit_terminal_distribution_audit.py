#!/usr/bin/env python3
"""Plot the cross-seed SiT terminal-distribution mechanism summary."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ROOT = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/"
    "terminal_distribution_audit_800k_v1"
)
ORDER = (
    "factorized_g1_r1p5",
    "factorized_g1p5_r1p35",
    "factorized_g2_r1p35",
    "factorized_g2p5_r1p35",
    "factorized_g3_r1",
    "closed_g3",
)
LABELS = {
    "factorized_g1_r1p5": r"$(\gamma,\rho)=(1,1.5)$",
    "factorized_g1p5_r1p35": r"$(1.5,1.35)$",
    "factorized_g2_r1p35": r"$(2,1.35)$",
    "factorized_g2p5_r1p35": r"$(2.5,1.35)$",
    "factorized_g3_r1": r"frozen $\gamma=3$",
    "closed_g3": r"closed $\gamma=3$",
}
COLORS = {
    name: color
    for name, color in zip(
        ORDER,
        ("#2b6f9f", "#2f8f72", "#d28c28", "#bf5b45", "#6f5aa8", "#333333"),
    )
}


def _mean_std(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    return frame.groupby("condition", sort=False)[value].agg(["mean", "std"])


def main(args: argparse.Namespace) -> None:
    root = args.root.expanduser().resolve()
    quality = pd.read_csv(root / "combined_quality.csv")
    action = pd.read_csv(root / "combined_action.csv")
    latent = pd.read_csv(root / "combined_latent_pairwise.csv")
    feature = pd.read_csv(root / "combined_feature_pairwise.csv")
    merged = quality.merge(
        action[["seed", "condition", "control_action_mean"]],
        on=["seed", "condition"],
        how="inner",
    )

    figure, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    ax = axes[0, 0]
    for condition in ORDER:
        rows = merged[merged["condition"] == condition]
        ax.scatter(
            rows["control_action_mean"],
            rows["fid"],
            s=30,
            alpha=0.35,
            color=COLORS[condition],
        )
        action_mean = rows["control_action_mean"].mean()
        fid_mean = rows["fid"].mean()
        ax.scatter(action_mean, fid_mean, s=80, color=COLORS[condition], zorder=3)
        ax.annotate(
            LABELS[condition],
            (action_mean, fid_mean),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_xlabel("Control action")
    ax.set_ylabel("FID (1K, lower is better)")
    ax.set_title("A. Terminal quality versus control energy")
    ax.grid(alpha=0.2)

    ax = axes[0, 1]
    action_means = _mean_std(action, "forcing_action_mean").reindex(ORDER)
    response_means = _mean_std(action, "response_control_action_mean").reindex(ORDER)
    cross_means = _mean_std(action, "forcing_response_cross_action_mean").reindex(ORDER)
    positions = np.arange(len(ORDER))
    ax.bar(positions, action_means["mean"], color="#4c78a8", label="forcing")
    ax.bar(
        positions,
        response_means["mean"],
        bottom=action_means["mean"],
        color="#f2a541",
        label="strong response",
    )
    ax.scatter(
        positions,
        action_means["mean"] + response_means["mean"] + cross_means["mean"],
        marker="_",
        s=180,
        linewidths=2,
        color="#b33f40",
        label="including cross term",
    )
    ax.set_xticks(positions, [LABELS[name] for name in ORDER], rotation=24, ha="right")
    ax.set_ylabel("Integrated action")
    ax.set_title("B. Exact action decomposition")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1, 0]
    selected_pairs = (
        ("factorized_g1p5_r1p35", "factorized_g2_r1p35"),
        ("factorized_g2_r1p35", "factorized_g2p5_r1p35"),
        ("factorized_g2_r1p35", "closed_g3"),
    )
    for first, second in selected_pairs:
        rows = latent[
            (latent["comparison"] == "cross_condition")
            & (latent["condition_a"] == first)
            & (latent["condition_b"] == second)
        ]
        grouped = rows.groupby("time")["paired_rms"].agg(["mean", "std"])
        label = f"{LABELS[first]} vs {LABELS[second]}"
        ax.plot(grouped.index, grouped["mean"], linewidth=2, label=label)
        ax.fill_between(
            grouped.index,
            grouped["mean"] - grouped["std"].fillna(0),
            grouped["mean"] + grouped["std"].fillna(0),
            alpha=0.15,
        )
    ax.set_xlabel("Generation time")
    ax.set_ylabel("Paired latent RMS")
    ax.set_title("C. Different particle trajectories")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)

    ax = axes[1, 1]
    matrix = np.zeros((len(ORDER), len(ORDER)), dtype=np.float64)
    column = "c2st_auc_mean_excess_over_split_null"
    cross = feature[feature["comparison"] == "cross_condition"]
    for first_index, first in enumerate(ORDER):
        for second_index, second in enumerate(ORDER):
            if first_index == second_index:
                continue
            low, high = sorted((first_index, second_index))
            rows = cross[
                (cross["condition_a"] == ORDER[low])
                & (cross["condition_b"] == ORDER[high])
            ]
            matrix[first_index, second_index] = rows[column].mean()
    limit = max(float(np.nanmax(np.abs(matrix))), 0.01)
    image = ax.imshow(matrix, cmap="magma", vmin=0.0, vmax=limit)
    ax.set_xticks(
        range(len(ORDER)),
        [LABELS[name] for name in ORDER],
        rotation=35,
        ha="right",
        fontsize=8,
    )
    ax.set_yticks(range(len(ORDER)), [LABELS[name] for name in ORDER], fontsize=8)
    ax.set_title("D. Endpoint feature C2ST above split-null")
    figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    figure.suptitle("SiT-v800 / v500 terminal-distribution control audit", fontsize=16)
    png = root / "terminal_distribution_audit_summary.png"
    pdf = root / "terminal_distribution_audit_summary.pdf"
    figure.savefig(png, dpi=180)
    figure.savefig(pdf)
    print(png)
    print(pdf)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
