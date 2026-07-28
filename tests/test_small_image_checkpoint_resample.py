from __future__ import annotations

import pandas as pd
import pytest

from experiments.small_image_checkpoint_resample import (
    paired_endpoint_rows,
    summarize_resamples,
)


def test_paired_endpoint_rows_compute_ratios_and_deltas():
    rollout = pd.DataFrame(
        [
            {"variant": "baseline", "feature_fid": 10.0, "feature_swd": 2.0, "latent_swd": 1.0},
            {"variant": "weighted", "feature_fid": 12.0, "feature_swd": 1.0, "latent_swd": 1.5},
        ]
    )
    rows = paired_endpoint_rows(
        rollout,
        dataset="mnist",
        basis="dct",
        training_seed=4,
        evaluation_seed=1701,
    )
    fid = next(row for row in rows if row["metric"] == "feature_fid")
    assert fid["delta"] == pytest.approx(2.0)
    assert fid["ratio"] == pytest.approx(1.2)


def test_resample_summary_keeps_training_and_evaluation_seeds_separate():
    rows = []
    for evaluation_seed, ratio in ((10, 0.8), (11, 1.2)):
        rows.append(
            {
                "dataset": "mnist",
                "basis": "pca",
                "training_seed": 4,
                "evaluation_seed": evaluation_seed,
                "metric": "feature_fid",
                "baseline": 10.0,
                "weighted": 10.0 * ratio,
                "delta": 10.0 * ratio - 10.0,
                "ratio": ratio,
            }
        )
    summary = summarize_resamples(pd.DataFrame(rows)).iloc[0]
    assert summary["evaluation_seeds"] == 2
    assert summary["ratio_mean"] == pytest.approx(1.0)
    assert summary["harm_rate"] == pytest.approx(0.5)


def test_paired_rows_reject_incomplete_variants():
    rollout = pd.DataFrame(
        [{"variant": "baseline", "feature_fid": 1.0, "feature_swd": 1.0, "latent_swd": 1.0}]
    )
    with pytest.raises(ValueError, match="exactly baseline and weighted"):
        paired_endpoint_rows(
            rollout,
            dataset="mnist",
            basis="dct",
            training_seed=0,
            evaluation_seed=0,
        )
