from __future__ import annotations

import pandas as pd
import torch

from experiments.rae_layerwise_path import DetailSubspace
from experiments.run_rae_layerwise_path_gate import (
    GateConfig,
    acceptance_table,
    neighborhood_statistics,
)


def test_neighborhood_recall_is_one_for_identical_features() -> None:
    features = torch.randn((20, 12), generator=torch.Generator().manual_seed(3))
    labels = torch.arange(20) % 4
    result = neighborhood_statistics(features, features.clone(), labels, k=5)
    assert result["neighborhood_recall"] == 1.0
    assert result["label_purity_ratio"] == 1.0


def _metric_rows(rank: int, transform: str):
    values = {
        ("semantic", "direct_error"): 1.0,
        ("detail", "direct_error"): 0.8,
        ("random_detail", "direct_error"): 0.95,
        ("semantic", "mean_diag_cosine"): 0.2,
        ("detail", "mean_diag_cosine"): 0.3,
        ("random_detail", "mean_diag_cosine"): 0.25,
    }
    return [
        {
            "rank": rank,
            "transform": transform,
            "component": component,
            "metric": metric,
            "value": value,
        }
        for (component, metric), value in values.items()
    ]


def test_acceptance_requires_all_preregistered_controls() -> None:
    config = GateConfig(ranks=(4,), transforms=("flip_h",))
    metrics = pd.DataFrame(_metric_rows(4, "flip_h"))
    summary = pd.DataFrame(
        [
            {
                "rank": 4,
                "neighborhood_recall": 0.97,
                "full_l1": 0.1,
                "semantic_l1": 0.2,
                "full_lpips": 0.15,
                "semantic_lpips": 0.3,
            }
        ]
    )
    accepted = acceptance_table(metrics, summary, config)
    assert bool(accepted.iloc[0]["gate_pass"])

    summary.loc[0, "neighborhood_recall"] = 0.9
    rejected = acceptance_table(metrics, summary, config)
    assert not bool(rejected.iloc[0]["gate_pass"])


def test_detail_subspace_payload_shape() -> None:
    subspace = DetailSubspace(torch.eye(6)[:, :2], 0.5, 0.2, 1e-3, 100)
    assert subspace.rank == 2
    assert subspace.channels == 6
