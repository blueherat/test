"""Audit and analyze paired generation metrics for the SPC multi-seed study."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_RESULTS = Path.home() / "data/eqvae/experiments/rae_spc_multiseed_v1"
METRICS = {
    "fid": "frechet_inception_distance",
    "kid": "kernel_inception_distance_mean",
    "is": "inception_score_mean",
}


def paired_table(table: pd.DataFrame) -> pd.DataFrame:
    required = {"seed", "condition", *METRICS.values()}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"metrics table lacks columns: {sorted(missing)}")
    if table.duplicated(["seed", "condition"]).any():
        raise ValueError("duplicate seed/condition rows")
    wide = table.pivot(index="seed", columns="condition", values=list(METRICS.values()))
    for condition in ("static", "spc"):
        if condition not in wide.columns.get_level_values(1):
            raise ValueError(f"missing condition: {condition}")
    rows = []
    for seed in wide.index:
        row: dict[str, float | int] = {"seed": int(seed)}
        for short, metric in METRICS.items():
            baseline = float(wide.loc[seed, (metric, "static")])
            treatment = float(wide.loc[seed, (metric, "spc")])
            row[f"static_{short}"] = baseline
            row[f"spc_{short}"] = treatment
            row[f"delta_{short}"] = treatment - baseline
            row[f"relative_delta_{short}"] = (treatment - baseline) / baseline
        rows.append(row)
    return pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)


def bootstrap_mean_ci(
    values: np.ndarray, *, seed: int = 20_260_719, draws: int = 50_000
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def exact_sign_flip_pvalue(values: np.ndarray) -> float:
    """One-sided paired randomization p-value for a negative mean effect."""

    values = np.asarray(values, dtype=np.float64)
    observed = float(values.mean())
    randomized = [
        float((values * np.asarray(signs, dtype=np.float64)).mean())
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    return float(np.mean(np.asarray(randomized) <= observed + 1e-15))


def summarize(paired: pd.DataFrame) -> dict[str, object]:
    both_better = (paired["delta_fid"] < 0) & (paired["delta_kid"] < 0)
    fid_mean, fid_low, fid_high = bootstrap_mean_ci(paired["delta_fid"].to_numpy())
    relative_mean, relative_low, relative_high = bootstrap_mean_ci(
        paired["relative_delta_fid"].to_numpy()
    )
    kid_mean, kid_low, kid_high = bootstrap_mean_ci(paired["delta_kid"].to_numpy())
    criteria = {
        "at_least_four_of_five_improve_fid_and_kid": int(both_better.sum()) >= 4,
        "mean_relative_fid_improvement_at_least_five_percent": relative_mean <= -0.05,
        "paired_fid_bootstrap_ci_excludes_zero": fid_high < 0.0,
    }
    return {
        "seed_count": int(len(paired)),
        "both_fid_kid_better_count": int(both_better.sum()),
        "fid_delta_mean_ci95": [fid_mean, fid_low, fid_high],
        "fid_relative_delta_mean_ci95": [relative_mean, relative_low, relative_high],
        "kid_delta_mean_ci95": [kid_mean, kid_low, kid_high],
        "fid_exact_one_sided_sign_flip_p": exact_sign_flip_pvalue(
            paired["delta_fid"].to_numpy()
        ),
        "kid_exact_one_sided_sign_flip_p": exact_sign_flip_pvalue(
            paired["delta_kid"].to_numpy()
        ),
        "criteria": criteria,
        "generation_gate_pass": bool(all(criteria.values())),
    }


def plot_pairs(paired: pd.DataFrame, output: Path, weight_source: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    colors = {"static": "#4C78A8", "spc": "#E45756"}
    for axis, short, title in zip(
        axes[:2], ("fid", "kid"), ("FID (lower is better)", "KID (lower is better)")
    ):
        for _, row in paired.iterrows():
            axis.plot(
                [0, 1],
                [row[f"static_{short}"], row[f"spc_{short}"]],
                color="#8A8A8A",
                alpha=0.7,
                linewidth=1.5,
                marker="o",
            )
        axis.scatter(
            [0] * len(paired), paired[f"static_{short}"], color=colors["static"], s=55, zorder=3
        )
        axis.scatter(
            [1] * len(paired), paired[f"spc_{short}"], color=colors["spc"], s=55, zorder=3
        )
        axis.set_xticks([0, 1], ["Static", "SPC"])
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[2].axhline(0, color="#222222", linewidth=1)
    axes[2].bar(
        paired["seed"].astype(str),
        100.0 * paired["relative_delta_fid"],
        color=["#59A14F" if value < 0 else "#E45756" for value in paired["relative_delta_fid"]],
    )
    axes[2].set_title("Paired SPC FID change")
    axes[2].set_ylabel("Relative change (%)")
    axes[2].set_xlabel("Training seed")
    axes[2].grid(axis="y", alpha=0.25)
    figure.suptitle(f"RAE Subspace Path Curriculum, {weight_source} weights", fontsize=16)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--weight-source", choices=("model", "ema"), default="model")
    parser.add_argument("--sample-count", type=int, default=5000)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()
    root = args.results.expanduser().resolve()
    evaluation = root / "evaluation"
    source = evaluation / f"spc_metrics_{args.weight_source}_n{args.sample_count}_{args.steps}steps.csv"
    table = pd.read_csv(source)
    paired = paired_table(table)
    summary = summarize(paired)
    paired_path = evaluation / f"spc_paired_{args.weight_source}_n{args.sample_count}.csv"
    summary_path = evaluation / f"spc_summary_{args.weight_source}_n{args.sample_count}.json"
    figure_path = evaluation / f"spc_pairs_{args.weight_source}_n{args.sample_count}.png"
    paired.to_csv(paired_path, index=False)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    plot_pairs(paired, figure_path, args.weight_source)
    print(paired.to_string(index=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
