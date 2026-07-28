from __future__ import annotations

import pandas as pd
import pytest

from experiments.summarize_rollout_checkpoint_selection import (
    proxy_fid_alignment,
    semantic_selection_diagnostic,
)


def test_proxy_fid_alignment_keeps_proxy_and_test_metrics_separate():
    history = pd.DataFrame(
        [
            {
                "dataset": "mnist",
                "training_seed": 5,
                "variant": "baseline",
                "step": 500,
                "proxy_loss": 0.5,
                "selected": True,
            },
            {
                "dataset": "mnist",
                "training_seed": 5,
                "variant": "baseline",
                "step": 1000,
                "proxy_loss": 1.0,
                "selected": False,
            },
        ]
    )
    seed_summary = pd.DataFrame(
        [
            {
                "dataset": "mnist",
                "training_seed": 5,
                "basis": "baseline",
                "comparison": "selected_vs_final",
                "metric": "feature_fid",
                "ratio": 1.2,
            }
        ]
    )
    rows, summary = proxy_fid_alignment(history, seed_summary)
    assert rows.iloc[0]["proxy_ratio"] == pytest.approx(0.5)
    assert rows.iloc[0]["fid_ratio"] == pytest.approx(1.2)
    assert summary.iloc[0]["conditions"] == 1


def test_proxy_alignment_rejects_multiple_selected_checkpoints():
    history = pd.DataFrame(
        [
            {
                "dataset": "mnist",
                "training_seed": 5,
                "variant": "baseline",
                "step": step,
                "proxy_loss": 0.5,
                "selected": True,
            }
            for step in (500, 600)
        ]
    )
    seed_summary = pd.DataFrame()
    with pytest.raises(ValueError, match="duplicate keys"):
        proxy_fid_alignment(history, seed_summary)


def test_semantic_diagnostic_reports_larger_selected_entropy_gap():
    rows = []
    for variant in ("baseline", "dct", "pca", "random"):
        for checkpoint, entropy in (("selected", 2.0), ("final", 2.2)):
            rows.append(
                {
                    "dataset": "mnist",
                    "training_seed": 5,
                    "evaluation_seed": 1,
                    "variant": f"{variant}_{checkpoint}",
                    "classifier_confidence": 0.8,
                    "class_entropy": entropy,
                }
            )
    summary = semantic_selection_diagnostic(pd.DataFrame(rows))
    assert len(summary) == 4
    assert summary["class_entropy_gap_change"].gt(0).all()
