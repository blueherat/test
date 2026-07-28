import pandas as pd

from experiments.small_image_time_seed_sweep import summarize_seed_sweep


def test_summarize_seed_sweep_reports_variance_and_correlations() -> None:
    conditions = pd.DataFrame(
        {
            "schedule": ["iid_seed0", "iid_seed1", "iid_seed2"],
            "metric": ["feature_fid"] * 3,
            "baseline_mean": [10.0, 20.0, 30.0],
            "weighted_mean": [12.0, 18.0, 24.0],
            "ratio_mean": [1.2, 0.9, 0.8],
        }
    )
    summary = summarize_seed_sweep(conditions).iloc[0]
    assert summary["time_seeds"] == 3
    assert summary["weighted_over_baseline_variance"] < 1.0
    assert summary["harm_rate"] == 1.0 / 3.0
    assert summary["baseline_weighted_correlation"] > 0.99
    assert summary["baseline_ratio_correlation"] < 0.0
