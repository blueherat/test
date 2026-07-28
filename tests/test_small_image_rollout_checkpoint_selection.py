from __future__ import annotations

import pandas as pd
import pytest

from experiments.small_image_rollout_checkpoint_selection import (
    METRICS,
    evaluate_gates,
    paired_rollout_rows,
    select_checkpoint,
    summarize_paired_metrics,
)


def test_checkpoint_selection_uses_lowest_proxy_and_earliest_tie():
    history = [
        {"step": 400, "proxy_loss": 0.2},
        {"step": 200, "proxy_loss": 0.1},
        {"step": 300, "proxy_loss": 0.1},
    ]
    assert select_checkpoint(history) == (200, 0.1)


def test_paired_rows_use_fair_selected_and_final_baselines():
    rows = []
    values = {
        "baseline_final": 10.0,
        "baseline_selected": 8.0,
        "dct_final": 12.0,
        "dct_selected": 6.0,
    }
    for variant, value in values.items():
        rows.append({"variant": variant, **{metric: value for metric in METRICS}})
    paired = pd.DataFrame(
        paired_rollout_rows(
            pd.DataFrame(rows),
            dataset="mnist",
            training_seed=5,
            evaluation_seed=1,
            bases=("dct",),
        )
    )
    selected = paired[
        paired["comparison"].eq("selected_vs_selected_baseline")
        & paired["metric"].eq("feature_fid")
    ].iloc[0]
    final = paired[
        paired["comparison"].eq("final_vs_final_baseline")
        & paired["metric"].eq("feature_fid")
    ].iloc[0]
    assert selected["ratio"] == pytest.approx(0.75)
    assert final["ratio"] == pytest.approx(1.2)


def _synthetic_paired(ratio: float = 0.95) -> pd.DataFrame:
    rows = []
    for dataset in ("mnist", "fashion_mnist"):
        for seed in range(5, 10):
            for basis in ("baseline", "dct", "pca"):
                rows.append(
                    {
                        "dataset": dataset,
                        "training_seed": seed,
                        "evaluation_seed": 1,
                        "basis": basis,
                        "comparison": "selected_vs_final",
                        "metric": "feature_fid",
                        "delta": ratio - 1.0,
                        "ratio": ratio,
                    }
                )
            for basis in ("dct", "pca"):
                rows.append(
                    {
                        "dataset": dataset,
                        "training_seed": seed,
                        "evaluation_seed": 1,
                        "basis": basis,
                        "comparison": "selected_vs_selected_baseline",
                        "metric": "feature_fid",
                        "delta": ratio - 1.0,
                        "ratio": ratio,
                    }
                )
    return pd.DataFrame(rows)


def test_summary_and_gates_pass_only_when_every_preregistered_cell_passes():
    paired = _synthetic_paired()
    _, aggregate = summarize_paired_metrics(paired)
    gates = evaluate_gates(aggregate)
    assert gates["h1_rollout_selection_prevents_late_drift"] is True
    assert gates["h2_selection_makes_spectral_weighting_better_than_baseline"] is True

    paired.loc[
        paired["dataset"].eq("mnist")
        & paired["basis"].eq("pca")
        & paired["comparison"].eq("selected_vs_selected_baseline"),
        "ratio",
    ] = 1.1
    _, aggregate = summarize_paired_metrics(paired)
    gates = evaluate_gates(aggregate)
    assert gates["h2_selection_makes_spectral_weighting_better_than_baseline"] is False


def test_empty_selection_history_is_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        select_checkpoint([])
