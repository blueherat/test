from __future__ import annotations

import pandas as pd
import pytest

from experiments.small_image_training_path_probe import summarize_training_path


def test_training_path_summary_pairs_variants_at_each_checkpoint():
    rows = []
    for step in (0, 10):
        for variant, loss, gap in (
            ("baseline", 1.0, -0.2),
            ("weighted", 0.8, -0.1),
        ):
            rows.append(
                {
                    "dataset": "mnist",
                    "basis": "dct",
                    "training_seed": 4,
                    "step": step,
                    "variant": variant,
                    "endpoint_moment_loss": loss,
                    "band0_log_gap": gap,
                }
            )
    summary = summarize_training_path(pd.DataFrame(rows))
    assert len(summary) == 2
    assert summary["weighted_minus_baseline_endpoint_moment_loss"].tolist() == pytest.approx(
        [-0.2, -0.2]
    )
    assert summary["weighted_minus_baseline_band0_log_gap"].tolist() == pytest.approx(
        [0.1, 0.1]
    )


def test_training_path_summary_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing columns"):
        summarize_training_path(pd.DataFrame({"step": [0]}))
