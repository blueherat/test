from __future__ import annotations

import sys
from pathlib import Path

import torch
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.audit_raev2_posterior_geometry import (
    centered_directional_derivative,
    covariance_action_metrics,
    unit_sample_rms,
)


def test_centered_directional_derivative_is_exact_for_linear_map() -> None:
    direction = torch.tensor([[[[1.0, -2.0]]]])
    matrix = torch.tensor([[2.0, 0.5], [0.5, 1.0]])

    def mapping(value: torch.Tensor) -> torch.Tensor:
        return value.flatten(1).matmul(matrix.T).reshape_as(value)

    center = torch.tensor([[[[0.3, -0.7]]]])
    epsilon = 1e-3
    observed = centered_directional_derivative(
        mapping(center + epsilon * direction),
        mapping(center - epsilon * direction),
        epsilon,
    )
    expected = mapping(direction)
    assert torch.allclose(observed, expected, atol=1e-4, rtol=1e-4)


def test_covariance_action_reports_effective_scale_matched_rotation() -> None:
    gap = torch.tensor([[[[2.0, 0.0]]]])
    direction = unit_sample_rms(gap)
    action = torch.cat((2.0 * direction[..., :1], direction[..., :1]), dim=-1)
    metrics = covariance_action_metrics(gap, action)

    assert metrics["positive_rayleigh"].item() == 1.0
    assert metrics["rayleigh"].item() == pytest.approx(2.0)
    assert metrics["matched_orthogonal_rms_over_gap"].item() == pytest.approx(0.5)
    assert metrics["matched_total_rms_over_gap"].item() == pytest.approx(1.25**0.5)


def test_isotropic_action_reduces_to_ordinary_internal_guidance() -> None:
    gap = torch.randn(3, 4, 2, 2)
    action = 0.7 * unit_sample_rms(gap)
    metrics = covariance_action_metrics(gap, action)
    assert torch.allclose(metrics["action_gap_cosine"], torch.ones(3), atol=1e-6)
    assert torch.allclose(
        metrics["matched_total_rms_over_gap"], torch.ones(3), atol=1e-6
    )
    assert torch.all(metrics["matched_orthogonal_rms_over_gap"] < 1e-4)
