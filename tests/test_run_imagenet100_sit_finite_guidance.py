from __future__ import annotations

import torch

from experiments.run_imagenet100_sit_finite_guidance import (
    _aggregate_feedback,
    _aggregate_linearity,
    _aggregate_solver,
    _gamma_index,
)


def test_gamma_index_tolerates_fp32_rounding() -> None:
    values = torch.tensor([-0.01, 0.0, 0.01]).tolist()
    assert _gamma_index(values, -0.01) == 0
    assert _gamma_index(values, 0.01) == 2


def test_linearity_aggregation_recovers_exact_response() -> None:
    gammas = torch.tensor([-0.01, 0.0, 0.01, 0.1, 1.0])
    tangent = torch.tensor([[[1.0, 2.0]], [[-0.5, 0.25]]])
    baseline = torch.tensor([[[0.2, -0.1]], [[0.4, 0.3]]])
    endpoints = torch.stack([baseline + gamma * tangent for gamma in gammas])
    rows, summary = _aggregate_linearity(
        [
            {
                "gammas": gammas,
                "baseline": baseline,
                "tangent": tangent,
                "endpoints": endpoints,
            }
        ],
        0.01,
    )
    assert summary["central_difference_pass"]
    assert summary["largest_passing_positive_gamma"] == 1.0
    gamma_one = next(row for row in rows if abs(float(row["gamma"]) - 1.0) < 1e-6)
    assert float(gamma_one["relative_residual_mean"]) < 1e-6


def test_feedback_aggregation_detects_retained_frozen_response() -> None:
    gammas = torch.tensor([0.0, 0.1, 1.0])
    baseline = torch.zeros(2, 1, 2)
    direction = torch.tensor([[[1.0, -0.5]], [[0.2, 0.8]]])
    frozen = torch.stack([baseline + 0.8 * gamma * direction for gamma in gammas])
    closed = torch.stack([baseline + gamma * direction for gamma in gammas])
    rows, summary = _aggregate_feedback(
        [
            {
                "gammas": gammas,
                "baseline": baseline,
                "frozen": frozen,
                "closed": closed,
            }
        ]
    )
    assert len(rows) == 2
    assert summary["gamma_one_frozen_response_retained"]


def test_solver_aggregation_orders_fixed_step_rows() -> None:
    gammas = torch.tensor([0.0, 1.0])
    adaptive = torch.ones(2, 2, 1, 2)
    fixed = {4: adaptive + 0.1, 8: adaptive + 0.01}
    rows, summary = _aggregate_solver(
        [{"gammas": gammas, "adaptive": adaptive, "fixed": fixed}]
    )
    assert len(rows) == 4
    assert summary["highest_step_count"] == 8
    assert max(
        float(row["relative_endpoint_difference_mean"])
        for row in summary["highest_step_rows"]
    ) < 0.02

