from __future__ import annotations

import pandas as pd

from experiments.summarize_raev2_decoded_distribution_audit import summarize


def test_summary_requires_cross_seed_decoder_reversal() -> None:
    rows = []
    for run, scale in (("a", 1.0), ("b", 1.1)):
        rows.append(
            {
                "run": run,
                "requested_time": 0.2,
                "actual_time": 0.198,
                "auc_delta_ig_minus_full": -0.04 * scale,
                "auc_delta_ci_high": -0.01,
                "fid_real_delta_ig_minus_full": -2.0 * scale,
                "fid_p_delta_ig_minus_full": -1.0 * scale,
            }
        )
    result = summarize(pd.DataFrame(rows)).iloc[0]
    assert result["runs"] == 2
    assert bool(result["all_seeds_auc_closer"])
    assert bool(result["all_seeds_real_fid_better"])
    assert bool(result["all_seeds_p_fid_closer"])
