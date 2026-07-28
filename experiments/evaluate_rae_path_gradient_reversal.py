"""Evaluate the preregistered low-noise RAE gradient-reversal confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT = (
    Path.home()
    / "data/eqvae/experiments/rae_path_gradient_interference/confirm_n128_seed20260725"
)


def paired_mean_bootstrap(
    batches: pd.DataFrame,
    *,
    step: int,
    time: float,
    parameter_group: str,
    other: str,
    seed: int,
    repetitions: int = 20_000,
) -> dict[str, float]:
    subset = batches[
        (batches.checkpoint_step == int(step))
        & (batches.time == float(time))
        & (batches.parameter_group == parameter_group)
    ]
    pivot = subset.pivot(
        index=["split", "batch_index"],
        columns="condition",
        values="semantic_descent_ratio",
    )
    difference = (pivot["static"] - pivot[other]).to_numpy(dtype=np.float64)
    if len(difference) < 2:
        raise ValueError("paired bootstrap requires at least two batches")
    generator = np.random.default_rng(int(seed))
    indices = generator.integers(
        0, len(difference), size=(int(repetitions), len(difference))
    )
    bootstrap = difference[indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap, (0.025, 0.975))
    return {
        "mean": float(difference.mean()),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "static_win_fraction": float((difference > 0.0).mean()),
        "batch_count": int(len(difference)),
    }


def _value(
    table: pd.DataFrame,
    condition: str,
    step: int,
    time: float,
    group: str,
    metric: str,
) -> float:
    row = table[
        (table.condition == condition)
        & (table.checkpoint_step == int(step))
        & (table.time == float(time))
        & (table.parameter_group == group)
    ]
    if len(row) != 1:
        raise ValueError(
            f"expected one row for {condition}/{step}/{time}/{group}, got {len(row)}"
        )
    return float(row.iloc[0][metric])


def evaluate_confirmation(
    aggregate: pd.DataFrame,
    batches: pd.DataFrame,
    *,
    seed: int = 20_260_725,
) -> dict[str, object]:
    main_conditions = ("floor020_p2", "annealed")
    main_group = "last_block"
    main_time = 0.1
    c1_values = {}
    c1 = True
    for condition in main_conditions + ("static",):
        c1_values[condition] = {}
        for step in (2000, 5000):
            aggregate_ratio = _value(
                aggregate,
                condition,
                step,
                main_time,
                main_group,
                "semantic_descent_ratio",
            )
            cross_ratio = _value(
                aggregate,
                condition,
                step,
                main_time,
                main_group,
                "cross_split_semantic_descent_ratio",
            )
            c1_values[condition][str(step)] = {
                "aggregate": aggregate_ratio,
                "cross_split": cross_ratio,
            }
            expected_positive = condition == "static" or step == 2000
            if expected_positive:
                c1 &= aggregate_ratio > 1.0 and cross_ratio > 1.0
            else:
                c1 &= aggregate_ratio < 1.0 and cross_ratio < 1.0

    bootstrap = {}
    for step in (2000, 5000):
        for other_index, other in enumerate(main_conditions):
            bootstrap[f"step{step}_static_minus_{other}"] = paired_mean_bootstrap(
                batches,
                step=step,
                time=main_time,
                parameter_group=main_group,
                other=other,
                seed=seed + step + other_index,
            )
    c2 = (
        bootstrap["step5000_static_minus_floor020_p2"]["ci_lower"] > 0.0
        and bootstrap["step5000_static_minus_annealed"]["ci_lower"] > 0.0
        and bootstrap["step2000_static_minus_floor020_p2"]["ci_upper"] < 0.0
    )

    gaps = {}
    c3 = True
    c4 = True
    for condition in main_conditions:
        gap_t01 = _value(
            aggregate, "static", 5000, 0.1, main_group, "semantic_descent_ratio"
        ) - _value(
            aggregate, condition, 5000, 0.1, main_group, "semantic_descent_ratio"
        )
        gap_t03 = _value(
            aggregate, "static", 5000, 0.3, main_group, "semantic_descent_ratio"
        ) - _value(
            aggregate, condition, 5000, 0.3, main_group, "semantic_descent_ratio"
        )
        output_gap = _value(
            aggregate, "static", 5000, 0.1, "output_head", "semantic_descent_ratio"
        ) - _value(
            aggregate, condition, 5000, 0.1, "output_head", "semantic_descent_ratio"
        )
        localization_ratio = gap_t01 / max(output_gap, 1e-20)
        gaps[condition] = {
            "last_block_t01": gap_t01,
            "last_block_t03": gap_t03,
            "output_head_t01": output_gap,
            "last_over_output_t01": localization_ratio,
        }
        c3 &= gap_t01 > gap_t03 > 0.0
        c4 &= localization_ratio >= 3.0

    signs = []
    for condition in main_conditions + ("static",):
        for step in (2000, 5000):
            aggregate_cosine = _value(
                aggregate,
                condition,
                step,
                main_time,
                main_group,
                "semantic_basis_cosine",
            )
            cross_cosine = _value(
                aggregate,
                condition,
                step,
                main_time,
                main_group,
                "cross_split_semantic_basis_cosine",
            )
            signs.append(aggregate_cosine * cross_cosine > 0.0)
    sign_rate = float(np.mean(signs))
    c5 = sign_rate == 1.0
    predictions = {
        "c1_main_reversal": bool(c1),
        "c2_paired_batch_robustness": bool(c2),
        "c3_low_noise_localization": bool(c3),
        "c4_shared_block_localization": bool(c4),
        "c5_split_sign_stability": bool(c5),
    }
    return {
        "pass": bool(all(predictions.values())),
        "predictions": predictions,
        "details": {
            "main_ratios": c1_values,
            "paired_bootstrap": bootstrap,
            "localization_gaps": gaps,
            "main_sign_agreement": sign_rate,
        },
    }


def plot_checkpoint_trajectory(aggregate: pd.DataFrame, output: Path) -> None:
    colors = {"static": "#4C78A8", "annealed": "#E45756", "floor020_p2": "#54A24B"}
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    for row, group in enumerate(("last_block", "output_head")):
        for column, time in enumerate((0.3, 0.1)):
            axis = axes[row, column]
            subset = aggregate[
                (aggregate.parameter_group == group) & (aggregate.time == time)
            ]
            for condition, color in colors.items():
                values = subset[subset.condition == condition].sort_values(
                    "checkpoint_step"
                )
                axis.plot(
                    values.checkpoint_step,
                    values.semantic_descent_ratio,
                    marker="o",
                    linewidth=2.3,
                    color=color,
                    label=condition,
                )
            axis.axhline(1.0, color="#555555", linestyle="--", linewidth=1.2)
            axis.set_title(f"{group}, t={time}")
            axis.set_xlabel("training step")
            axis.set_ylabel("semantic descent ratio")
            axis.grid(alpha=0.25)
            axis.legend(frameon=False)
    figure.savefig(output / "low_noise_gradient_reversal.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    root = args.input.expanduser().resolve()
    aggregate = pd.read_csv(root / "aggregate_metrics.csv")
    batches = pd.read_csv(root / "batch_metrics.csv")
    result = evaluate_confirmation(aggregate, batches)
    (root / "reversal_confirmation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    plot_checkpoint_trajectory(aggregate, root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
