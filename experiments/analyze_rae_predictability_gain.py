"""Explain stage-2 directional gain using latent variance and predictability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ROOT = (
    Path.home()
    / "data/eqvae/experiments/rae_spc_multiseed_v1/evaluation"
)
DEFAULT_METRICS = DEFAULT_ROOT / "predictability_basis_v1/basis_block_metrics.csv"
DEFAULT_SENSITIVITY = (
    DEFAULT_ROOT / "predictability_block_sensitivity_n128_v1/sensitivity_per_seed.csv"
)
DEFAULT_OUTPUT = DEFAULT_ROOT / "predictability_block_sensitivity_n128_v1"


MATCHED_PAIRS = (
    ("fractional_000_015", "absolute_032_047"),
    ("fractional_032_047", "absolute_080_095"),
    ("fractional_032_047", "pca_096_111"),
)


def _regression_score(target: np.ndarray, features: np.ndarray) -> tuple[float, np.ndarray]:
    features = np.asarray(features)
    if features.ndim == 1:
        features = features[:, None]
    design = np.column_stack([np.ones(len(features)), features])
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    prediction = design @ coefficients
    denominator = np.square(target - target.mean()).sum()
    score = 1.0 - np.square(target - prediction).sum() / max(denominator, 1e-20)
    return float(score), coefficients


def gain_explanation_per_seed(
    metrics: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> pd.DataFrame:
    metrics = metrics[metrics["basis_family"] != "random"]
    sensitivity = sensitivity[sensitivity["basis_family"] != "random"]
    merged = sensitivity.merge(
        metrics[["basis", "val_final_variance_per_dimension", "val_r2"]],
        on="basis",
        validate="many_to_one",
    )
    rows: list[dict[str, float | int]] = []
    for (seed, time_value), frame in merged.groupby(["seed", "time"]):
        target = np.log(frame["total_gain"].to_numpy())
        features = np.column_stack(
            [
                np.log(frame["val_final_variance_per_dimension"].to_numpy()),
                frame["val_r2"].to_numpy(),
            ]
        )
        features = (features - features.mean(axis=0)) / features.std(axis=0)
        target = (target - target.mean()) / target.std()
        variance_r2, _ = _regression_score(target, features[:, 0])
        predictability_r2, _ = _regression_score(target, features[:, 1])
        combined_r2, coefficients = _regression_score(target, features)
        rows.append(
            {
                "seed": int(seed),
                "time": float(time_value),
                "variance_only_r2": variance_r2,
                "predictability_only_r2": predictability_r2,
                "combined_r2": combined_r2,
                "variance_beta": float(coefficients[1]),
                "predictability_beta": float(coefficients[2]),
            }
        )
    return pd.DataFrame(rows)


def matched_pair_ratios(
    metrics: pd.DataFrame,
    sensitivity: pd.DataFrame,
    pairs: tuple[tuple[str, str], ...] = MATCHED_PAIRS,
) -> pd.DataFrame:
    indexed_metrics = metrics.set_index("basis")
    rows: list[dict[str, float | int | str]] = []
    for high, low in pairs:
        high_rows = sensitivity[sensitivity["basis"] == high].set_index(
            ["seed", "time"]
        )
        low_rows = sensitivity[sensitivity["basis"] == low].set_index(
            ["seed", "time"]
        )
        common = high_rows.index.intersection(low_rows.index)
        ratios = high_rows.loc[common, "total_gain"] / low_rows.loc[common, "total_gain"]
        for (seed, time_value), ratio in ratios.items():
            rows.append(
                {
                    "higher_predictability_basis": high,
                    "lower_predictability_basis": low,
                    "seed": int(seed),
                    "time": float(time_value),
                    "gain_ratio": float(ratio),
                    "higher_variance": float(
                        indexed_metrics.loc[high, "val_final_variance_per_dimension"]
                    ),
                    "lower_variance": float(
                        indexed_metrics.loc[low, "val_final_variance_per_dimension"]
                    ),
                    "higher_r2": float(indexed_metrics.loc[high, "val_r2"]),
                    "lower_r2": float(indexed_metrics.loc[low, "val_r2"]),
                }
            )
    return pd.DataFrame(rows)


def summarize_explanation(per_seed: pd.DataFrame) -> pd.DataFrame:
    values = [
        "variance_only_r2",
        "predictability_only_r2",
        "combined_r2",
        "variance_beta",
        "predictability_beta",
    ]
    summary = per_seed.groupby("time", as_index=False)[values].agg(["mean", "std"])
    summary.columns = [
        "_".join(value for value in column if value)
        for column in summary.columns.to_flat_index()
    ]
    return summary


def plot_results(
    explanation: pd.DataFrame,
    pairs: pd.DataFrame,
    output: Path,
) -> None:
    summary = summarize_explanation(explanation)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    for field, label, color in (
        ("variance_beta", "latent variance", "#c84c32"),
        ("predictability_beta", "cross-layer predictability", "#2678a8"),
    ):
        axes[0].errorbar(
            summary["time"],
            summary[f"{field}_mean"],
            yerr=summary[f"{field}_std"],
            marker="o",
            capsize=3,
            color=color,
            label=label,
        )
    axes[0].axhline(0.0, color="#999999", linewidth=1, linestyle="--")
    axes[0].set_xlabel("noise time t")
    axes[0].set_ylabel("standardized coefficient for log directional gain")
    axes[0].set_title("What controls the model's directional response")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)
    for field, label, color in (
        ("variance_only_r2", "variance only", "#c84c32"),
        ("predictability_only_r2", "predictability only", "#2678a8"),
        ("combined_r2", "combined", "#2f855a"),
    ):
        axes[1].errorbar(
            summary["time"],
            summary[f"{field}_mean"],
            yerr=summary[f"{field}_std"],
            marker="o",
            capsize=3,
            color=color,
            label=label,
        )
    axes[1].set_xlabel("noise time t")
    axes[1].set_ylabel("explained variance (R2)")
    axes[1].set_ylim(0, 1.03)
    axes[1].set_title("Variance and predictability become complementary at high noise")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)
    fig.savefig(output / "gain_explanation_over_time.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 6.5), constrained_layout=True)
    colors = ("#2678a8", "#2f855a", "#8b5a9f")
    for (high, low), color in zip(MATCHED_PAIRS, colors):
        values = pairs[
            (pairs["higher_predictability_basis"] == high)
            & (pairs["lower_predictability_basis"] == low)
        ]
        grouped = values.groupby("time")["gain_ratio"].agg(["mean", "std"]).reset_index()
        axis.errorbar(
            grouped["time"],
            grouped["mean"],
            yerr=grouped["std"],
            marker="o",
            capsize=3,
            color=color,
            label=f"{high} / {low}",
        )
    axis.axhline(1.0, color="#999999", linewidth=1, linestyle="--")
    axis.set_xlabel("noise time t")
    axis.set_ylabel("gain ratio at matched latent variance")
    axis.set_title("High-noise response crosses over toward predictable directions")
    axis.legend(frameon=False, fontsize=8)
    axis.grid(alpha=0.25)
    fig.savefig(output / "matched_variance_gain_crossover.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--sensitivity", type=Path, default=DEFAULT_SENSITIVITY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    metrics = pd.read_csv(args.metrics.expanduser())
    sensitivity = pd.read_csv(args.sensitivity.expanduser())
    explanation = gain_explanation_per_seed(metrics, sensitivity)
    pairs = matched_pair_ratios(metrics, sensitivity)
    summary = summarize_explanation(explanation)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    explanation.to_csv(output / "gain_explanation_per_seed.csv", index=False)
    summary.to_csv(output / "gain_explanation_summary.csv", index=False)
    pairs.to_csv(output / "matched_variance_pairs.csv", index=False)
    plot_results(explanation, pairs, output)

    indexed = summary.set_index("time")
    pair_summary = pairs.groupby(
        ["higher_predictability_basis", "lower_predictability_basis", "time"],
        as_index=False,
    )["gain_ratio"].agg(["mean", "std"])
    payload = {
        "sample_count": int(sensitivity["sample_index"].max() + 1)
        if "sample_index" in sensitivity
        else 128,
        "seed_count": int(sensitivity["seed"].nunique()),
        "high_noise_t095": indexed.loc[0.95].to_dict(),
        "high_noise_t085": indexed.loc[0.85].to_dict(),
        "low_noise_t010": indexed.loc[0.1].to_dict(),
        "matched_pairs": pair_summary.to_dict(orient="records"),
    }
    (output / "gain_explanation_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
