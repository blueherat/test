from __future__ import annotations

import math

import torch

from experiments.audit_imagenet100_sit_depth_convergence import (
    fit_scalar_transition,
    fit_target_step,
    geometric_limit,
    transition_metrics,
)


def test_exact_contractive_sequence_is_recovered() -> None:
    torch.manual_seed(11)
    first = torch.randn(8, 2, 3, 3)
    contraction = 0.4
    second = contraction * first

    fitted = fit_scalar_transition(first[:4], second[:4])
    metrics = transition_metrics(first[4:], second[4:], fitted)

    assert math.isclose(fitted, contraction, rel_tol=1e-6, abs_tol=1e-6)
    assert metrics["cosine_mean"] > 0.999999
    assert metrics["sample_lambda_in_0_1_fraction"] == 1.0
    assert metrics["relative_transition_residual"] < 1e-6
    assert metrics["prediction_over_zero"] < 1e-10


def test_geometric_limit_recovers_known_limit() -> None:
    torch.manual_seed(12)
    limit = torch.randn(4, 1, 2, 2)
    initial_error = torch.randn_like(limit)
    contraction = 0.25
    previous = limit + contraction * initial_error
    current = limit + contraction**2 * initial_error
    latest_increment = current - previous

    estimate = geometric_limit(current, latest_increment, contraction)

    assert estimate is not None
    torch.testing.assert_close(estimate, limit, rtol=1e-6, atol=1e-6)
    oscillatory = geometric_limit(current, latest_increment, -0.1)
    assert oscillatory is not None
    assert geometric_limit(current, latest_increment, 0.99) is None


def test_negative_contraction_recovers_oscillatory_limit() -> None:
    torch.manual_seed(14)
    limit = torch.randn(4, 1, 2, 2)
    initial_error = torch.randn_like(limit)
    contraction = -0.3
    previous = limit + contraction * initial_error
    current = limit + contraction**2 * initial_error
    latest_increment = current - previous

    estimate = geometric_limit(current, latest_increment, contraction)

    assert estimate is not None
    torch.testing.assert_close(estimate, limit, rtol=1e-6, atol=1e-6)


def test_target_step_is_fit_without_sign_error() -> None:
    torch.manual_seed(13)
    base = torch.randn(6, 1, 2, 2)
    direction = torch.randn_like(base)
    target = base + 1.75 * direction

    gamma = fit_target_step(base, direction, target)

    assert math.isclose(gamma, 1.75, rel_tol=1e-6, abs_tol=1e-6)
