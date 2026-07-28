from __future__ import annotations

import torch

from experiments.analyze_rae_predictability_bases import (
    analyze_moments,
    build_block_atlas,
)
from experiments.rae_layerwise_path import MiddleFinalCovariance, fit_detail_subspace


def synthetic_moments() -> MiddleFinalCovariance:
    return MiddleFinalCovariance(
        middle_gram=torch.eye(4, dtype=torch.float64),
        middle_final=torch.diag(
            torch.tensor([5.0, 2.0, 0.95, 0.65], dtype=torch.float64)
        ),
        final_gram=torch.diag(
            torch.tensor([100.0, 20.0, 1.0, 0.5], dtype=torch.float64)
        ),
        token_count=100,
    )


def test_analysis_distinguishes_absolute_and_fractional_bases() -> None:
    moments = synthetic_moments()
    reference = fit_detail_subspace(moments, rank=1, ridge=0.0).basis
    rows, stability, bases = analyze_moments(
        moments,
        moments,
        moments,
        moments,
        {1: reference},
        ranks=(1,),
        ridge=0.0,
        final_ridge=0.0,
        random_controls=2,
        random_seed=9,
    )
    absolute = rows[rows["basis"] == "absolute_refit"].iloc[0]
    fractional = rows[rows["basis"] == "fractional"].iloc[0]

    assert absolute["val_final_variance_per_dimension"] > 0.9
    assert 0.2 < absolute["val_r2"] < 0.3
    assert fractional["val_final_variance_per_dimension"] < 0.02
    assert fractional["val_r2"] > 0.85
    assert float(bases[1]["absolute_refit"][0].abs()) > 0.999
    assert float(bases[1]["fractional"][2].abs()) > 0.999
    assert len(rows) == 6
    assert len(stability) == 3
    assert float(stability["half_split_overlap"].min()) > 0.999


def test_block_atlas_builds_disjoint_ordered_families() -> None:
    moments = synthetic_moments()
    rows, bases = build_block_atlas(
        moments,
        moments,
        block_rank=1,
        max_rank=4,
        ridge=0.0,
        final_ridge=0.0,
        random_controls=2,
        random_seed=5,
    )
    assert len(rows) == 14
    assert len(bases) == 14
    for family in ("absolute", "fractional", "pca"):
        family_bases = [bases[f"{family}_{index:03d}_{index:03d}"] for index in range(4)]
        matrix = torch.cat(family_bases, dim=1)
        torch.testing.assert_close(
            matrix.T @ matrix,
            torch.eye(4),
            atol=2e-6,
            rtol=0,
        )
