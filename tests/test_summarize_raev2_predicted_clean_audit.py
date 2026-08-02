from __future__ import annotations

import pandas as pd

from experiments.summarize_raev2_predicted_clean_audit import aggregate_runs


def test_aggregate_runs_reports_cross_seed_direction_counts() -> None:
    summaries = pd.DataFrame(
        [
            {
                "seed": seed,
                "requested_time": 0.2,
                "actual_time": 0.198,
                "condition": condition,
                "head": condition.split("_on_")[0],
                "state_branch": condition.split("_on_")[1],
                "on_policy": condition in ("full_on_full", "ig_on_ig"),
                "auc": 0.6,
                "auc_separability": 0.1,
                "fid_real": 10.0,
                "fid_reconstruction": 5.0,
            }
            for seed in (1, 2)
            for condition in ("full_on_full", "ig_on_full", "full_on_ig", "ig_on_ig")
        ]
    )
    effects = pd.DataFrame(
        [
            {
                "seed": seed,
                "requested_time": 0.2,
                "actual_time": 0.198,
                "effect": "on_policy_total",
                "positive_condition": "ig_on_ig",
                "negative_condition": "full_on_full",
                "auc_delta": -0.03,
                "auc_separability_delta": -0.02 if seed == 1 else 0.01,
                "fid_real_delta": -2.0,
                "fid_reconstruction_delta": -1.0,
            }
            for seed in (1, 2)
        ]
    )
    summary_agg, effect_agg = aggregate_runs(summaries, effects)
    assert len(summary_agg) == 4
    assert effect_agg.loc[0, "fid_real_delta_negative_seed_count"] == 2
    assert effect_agg.loc[0, "auc_separability_delta_negative_seed_count"] == 1
