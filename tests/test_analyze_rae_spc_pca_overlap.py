from __future__ import annotations

import torch

from experiments.analyze_rae_spc_pca_overlap import (
    pca_overlap_metrics,
    subspace_overlap,
)


def test_subspace_overlap_recovers_identical_and_orthogonal_spaces() -> None:
    identity = torch.eye(6, dtype=torch.float64)
    same = subspace_overlap(identity[:, :2], identity[:, :2])
    orthogonal = subspace_overlap(identity[:, :2], identity[:, 2:4])
    assert abs(same["mean_squared_principal_cosine"] - 1.0) < 1e-12
    assert orthogonal["mean_squared_principal_cosine"] < 1e-12


def test_pca_overlap_reports_top_variance_space() -> None:
    covariance = torch.diag(torch.tensor([9.0, 4.0, 2.0, 1.0]))
    identity = torch.eye(4)
    metrics = pca_overlap_metrics(
        covariance, identity[:, :2], identity[:, 2:4]
    )
    assert metrics["guided_vs_top_pca"]["mean_squared_principal_cosine"] > 0.999
    assert metrics["control_vs_top_pca"]["mean_squared_principal_cosine"] < 1e-12
    assert abs(
        metrics["guided_energy_fraction"] - metrics["top_pca_energy_fraction"]
    ) < 1e-12
