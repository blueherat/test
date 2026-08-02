from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.summarize_raev2_distribution_auc import (
    class_cluster_bootstrap_delta,
    conclusion,
    summarize,
)


def test_cross_seed_summary_requires_same_direction_in_every_seed() -> None:
    rows = []
    for run, deltas in {
        "seed_a": {0.2: (0.07, 0.03, 0.11), 0.8: (-0.03, -0.06, -0.01)},
        "seed_b": {0.2: (0.09, 0.05, 0.13), 0.8: (0.00, -0.03, 0.03)},
    }.items():
        for time, (delta, low, high) in deltas.items():
            rows.append(
                {
                    "run": run,
                    "requested_time": time,
                    "actual_time": time,
                    "auc_full": 0.6,
                    "auc_ig": 0.6 + delta,
                    "auc_delta_ig_minus_full": delta,
                    "delta_ci_low": low,
                    "delta_ci_high": high,
                    "q_ig_vs_full_relative_rms": 0.2,
                }
            )
    summary = summarize(pd.DataFrame(rows))

    late = summary[summary["requested_time"] == 0.2].iloc[0]
    early = summary[summary["requested_time"] == 0.8].iloc[0]
    assert bool(late["all_seeds_farther"])
    assert not bool(early["all_seeds_closer"])
    assert conclusion(summary).startswith("no global distribution correction")


def test_class_cluster_bootstrap_preserves_a_clear_paired_direction() -> None:
    classes = np.repeat(np.arange(20), 5)
    baseline = np.tile(np.linspace(-0.2, 0.2, 5), 20)
    low, high = class_cluster_bootstrap_delta(
        classes,
        baseline,
        baseline + 0.1,
        baseline,
        baseline + 1.0,
        repeats=200,
        seed=7,
    )
    assert low > 0
    assert high >= low
