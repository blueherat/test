"""Summarize the post-hoc decoder-witness gap audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CAPACITIES = (16, 64, 256)
SEEDS = (0, 1, 2, 3, 4)
DEFAULT_ROOT = Path.home() / "data/eqvae/imagenette_latent_prior_tradeoff"


def load_runs(root: Path) -> pd.DataFrame:
    rows = []
    for capacity in CAPACITIES:
        for seed in SEEDS:
            path = root / f"d{capacity}_seed{seed}_p0/decoder_witness_gap_posthoc.json"
            if path.is_file():
                rows.append(json.loads(path.read_text()))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["frozen_seed", "latent_dim"]).reset_index(drop=True)


def paired_summary(table: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    metrics = (
        "condition_domain_mlp_auc",
        "decoded_feature_domain_mlp_auc",
        "output_argmax_class_tv",
    )
    pivots = {
        metric: table.pivot(index="frozen_seed", columns="latent_dim", values=metric)
        for metric in metrics
    }
    paired = pd.DataFrame(index=SEEDS)
    paired.index.name = "frozen_seed"
    paired["condition_auc_64_minus_256"] = (
        pivots["condition_domain_mlp_auc"][64]
        - pivots["condition_domain_mlp_auc"][256]
    )
    paired["decoded_auc_256_minus_64"] = (
        pivots["decoded_feature_domain_mlp_auc"][256]
        - pivots["decoded_feature_domain_mlp_auc"][64]
    )
    paired["class_tv_256_minus_64"] = (
        pivots["output_argmax_class_tv"][256]
        - pivots["output_argmax_class_tv"][64]
    )
    checks = {
        "complete_grid": bool(
            set(zip(table.latent_dim.astype(int), table.frozen_seed.astype(int)))
            == {(capacity, seed) for capacity in CAPACITIES for seed in SEEDS}
        ),
        "condition_auc_64_gt_256_seed_count": int(
            (paired.condition_auc_64_minus_256 > 0).sum()
        ),
        "condition_auc_64_minus_256_mean": float(
            paired.condition_auc_64_minus_256.mean()
        ),
        "decoded_auc_256_gt_64_seed_count": int(
            (paired.decoded_auc_256_minus_64 > 0).sum()
        ),
        "decoded_auc_256_minus_64_mean": float(
            paired.decoded_auc_256_minus_64.mean()
        ),
        "class_tv_256_gt_64_seed_count": int(
            (paired.class_tv_256_minus_64 > 0).sum()
        ),
        "class_tv_256_minus_64_mean": float(
            paired.class_tv_256_minus_64.mean()
        ),
    }
    return paired.reset_index(), checks


def capacity_summary(table: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "modeling_gap",
        "condition_domain_linear_auc",
        "condition_domain_mlp_auc",
        "decoded_feature_domain_linear_auc",
        "decoded_feature_domain_mlp_auc",
        "output_argmax_class_tv",
        "output_probability_class_tv",
        "output_empirical_class_entropy",
        "output_prior_class_entropy",
        "condition_to_output_linear_balanced_accuracy",
        "condition_probe_predicted_class_tv",
    )
    records = []
    for capacity, frame in table.groupby("latent_dim"):
        record = {"latent_dim": int(capacity), "seed_count": len(frame)}
        for metric in metrics:
            record[f"{metric}_mean"] = float(frame[metric].mean())
            record[f"{metric}_std"] = float(frame[metric].std())
            record[f"{metric}_sem"] = float(frame[metric].sem())
        records.append(record)
    return pd.DataFrame(records).sort_values("latent_dim").reset_index(drop=True)


def plot_summary(table: pd.DataFrame, output: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {16: "#4C78A8", 64: "#F2A541", 256: "#C44E52"}
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10), constrained_layout=True)
    x = np.arange(len(CAPACITIES))
    width = 0.34

    for offset, metric, label, hatch in (
        (-width / 2, "condition_domain_mlp_auc", "condition embedding", ""),
        (width / 2, "decoded_feature_domain_mlp_auc", "decoded ResNet feature", "//"),
    ):
        means = [table[table.latent_dim == value][metric].mean() for value in CAPACITIES]
        sems = [table[table.latent_dim == value][metric].sem() for value in CAPACITIES]
        axes[0, 0].bar(
            x + offset,
            means,
            width,
            yerr=sems,
            capsize=4,
            label=label,
            color="#4C78A8" if offset < 0 else "#F2A541",
            edgecolor="#333333",
            linewidth=0.8,
            hatch=hatch,
        )
    axes[0, 0].axhline(0.5, color="#333333", linestyle="--", linewidth=1)
    axes[0, 0].set_ylim(0.45, 0.9)
    axes[0, 0].set_title("Two-sample classifier AUC")
    axes[0, 0].set_ylabel("held-out AUC")
    axes[0, 0].set_xticks(x, [str(value) for value in CAPACITIES])
    axes[0, 0].set_xlabel("latent capacity")
    axes[0, 0].legend(frameon=False)

    for metric, label, color, marker in (
        ("output_argmax_class_tv", "argmax histogram", "#C44E52", "o"),
        ("output_probability_class_tv", "mean probability", "#4C78A8", "s"),
    ):
        means = [table[table.latent_dim == value][metric].mean() for value in CAPACITIES]
        sems = [table[table.latent_dim == value][metric].sem() for value in CAPACITIES]
        axes[0, 1].errorbar(
            x,
            means,
            yerr=sems,
            marker=marker,
            capsize=4,
            linewidth=2,
            color=color,
            label=label,
        )
    axes[0, 1].set_title("Decoded Imagenette class-distribution shift")
    axes[0, 1].set_ylabel("total variation")
    axes[0, 1].set_xticks(x, [str(value) for value in CAPACITIES])
    axes[0, 1].set_xlabel("latent capacity")
    axes[0, 1].legend(frameon=False)

    means = [table[table.latent_dim == value].modeling_gap.mean() for value in CAPACITIES]
    sems = [table[table.latent_dim == value].modeling_gap.sem() for value in CAPACITIES]
    axes[1, 0].bar(
        x,
        means,
        yerr=sems,
        capsize=4,
        color=[colors[value] for value in CAPACITIES],
        edgecolor="#333333",
        linewidth=0.8,
    )
    axes[1, 0].set_title("Decoded modeling gap")
    axes[1, 0].set_ylabel("prior FID - empirical FID")
    axes[1, 0].set_xticks(x, [str(value) for value in CAPACITIES])
    axes[1, 0].set_xlabel("latent capacity")

    for capacity in CAPACITIES:
        frame = table[table.latent_dim == capacity]
        axes[1, 1].scatter(
            frame.condition_domain_mlp_auc,
            frame.decoded_feature_domain_mlp_auc,
            s=70,
            color=colors[capacity],
            edgecolor="#333333",
            linewidth=0.7,
            label=f"{capacity}d",
        )
    axes[1, 1].plot([0.48, 0.88], [0.48, 0.88], color="#333333", linestyle="--", linewidth=1)
    axes[1, 1].set_xlim(0.48, 0.88)
    axes[1, 1].set_ylim(0.47, 0.68)
    axes[1, 1].set_title("Mismatch visibility before and after decoding")
    axes[1, 1].set_xlabel("condition-space MLP AUC")
    axes[1, 1].set_ylabel("decoded-feature MLP AUC")
    axes[1, 1].legend(frameon=False)

    fig.suptitle(
        "Imagenette-64 decoder witness gap (5 frozen seeds per capacity)",
        fontsize=16,
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)


def summarize(root: Path) -> dict:
    output = root / "comparison_p0"
    output.mkdir(exist_ok=True)
    table = load_runs(root)
    if table.empty:
        raise FileNotFoundError("no decoder witness audit files found")
    capacities = capacity_summary(table)
    paired, checks = paired_summary(table)
    correlations = (
        table.select_dtypes(include=[np.number])
        .corr()["modeling_gap"]
        .sort_values()
        .rename("pearson_with_modeling_gap")
        .reset_index()
        .rename(columns={"index": "metric"})
    )
    table.to_csv(output / "decoder_witness_runs.csv", index=False)
    capacities.to_csv(output / "decoder_witness_capacity_summary.csv", index=False)
    paired.to_csv(output / "decoder_witness_paired.csv", index=False)
    correlations.to_csv(output / "decoder_witness_correlations.csv", index=False)
    (output / "decoder_witness_posthoc_summary.json").write_text(
        json.dumps(checks, indent=2, ensure_ascii=False) + "\n"
    )
    if checks["complete_grid"]:
        plot_summary(table, output / "decoder_witness_gap.png")
    print(json.dumps(checks, indent=2, ensure_ascii=False))
    return checks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    return summarize(args.root)


if __name__ == "__main__":
    main()
