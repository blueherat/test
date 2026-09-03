from __future__ import annotations

import sys
from pathlib import Path

import torch

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from path_evidence_pfr_bridge import (  # noqa: E402
    finite_horizon_nominal_evidence_gradient,
    match_sample_rms,
    normalized_flow_evidence_rate,
    pfr_revision,
    project_per_sample,
    project_to_forward_ray,
    sample_rms,
)


def test_flow_evidence_rate_matches_closed_form() -> None:
    gap = torch.tensor([[[[1.0, 2.0]]], [[[3.0, 4.0]]]])
    time = torch.tensor([0.2, 0.4])
    result = normalized_flow_evidence_rate(gap, time_value=time, beta=1.6)
    energy = gap.flatten(1).square().mean(dim=1)
    expected = 1.6 * 0.6 * time / (1.0 - time) * energy
    torch.testing.assert_close(result, expected)


def test_projection_is_exact_and_orthogonal() -> None:
    value = torch.tensor([[2.0, 3.0], [-1.0, 2.0]])
    direction = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    result = project_per_sample(value, direction)
    torch.testing.assert_close(result.parallel + result.orthogonal, value)
    dot = (result.orthogonal * direction).sum(dim=1)
    torch.testing.assert_close(dot, torch.zeros_like(dot))
    torch.testing.assert_close(result.coefficient, torch.tensor([2.0, 1.0]))


def test_forward_ray_clamps_negative_coefficient() -> None:
    value = torch.tensor([[-2.0, 1.0], [2.0, 1.0]])
    direction = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    result = project_to_forward_ray(value, direction)
    torch.testing.assert_close(result.coefficient, torch.tensor([0.0, 2.0]))
    torch.testing.assert_close(result.parallel, torch.tensor([[0.0, 0.0], [2.0, 0.0]]))


def test_rms_match_preserves_direction_and_matches_reference_norm() -> None:
    value = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
    reference = torch.tensor([[4.0, 0.0], [0.0, 2.0]])
    matched = match_sample_rms(value, reference)
    torch.testing.assert_close(sample_rms(matched), sample_rms(reference))
    assert torch.all((matched * value).sum(dim=1) > 0)


def test_finite_horizon_gradient_matches_linear_toy_autograd() -> None:
    state = torch.tensor([[1.0, -0.5], [0.25, 2.0]], requires_grad=True)
    time = torch.tensor(0.2)

    def evaluate_pair(t: torch.Tensor, z: torch.Tensor):
        strong = 2.0 * z + t
        weak = 0.5 * z - t
        return strong, weak

    result = finite_horizon_nominal_evidence_gradient(
        state,
        time,
        horizon=0.05,
        intervention_time=0.5,
        evaluate_pair=evaluate_pair,
        gamma_at=lambda value: 0.6 if value < 0.25 else 0.7,
        create_graph=True,
    )
    assert result.gradient.shape == state.shape
    assert torch.isfinite(result.gradient).all()
    assert torch.isfinite(result.value).all()

    expected = torch.autograd.grad(result.value.sum(), state)[0]
    torch.testing.assert_close(result.gradient, expected)


def test_pfr_revision_matches_projected_future_reference_formula() -> None:
    state = torch.tensor([[1.0, -1.0], [0.5, 2.0]])
    time = torch.tensor(0.2)
    strong = torch.tensor([[2.0, 0.0], [1.0, 3.0]])
    weak = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    gamma = 0.6
    guided = strong + gamma * (strong - weak)

    def evaluate_weak(future_time: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        return query + future_time

    revision, query, weak_query, alpha = pfr_revision(
        state,
        time,
        strong=strong,
        weak=weak,
        guided=guided,
        gamma=gamma,
        horizon=0.05,
        intervention_time=0.5,
        evaluate_weak=evaluate_weak,
    )
    calibration = (1.0 + gamma) * (strong - weak)
    expected_projection = project_to_forward_ray(calibration, guided)
    expected_query = state + 0.05 * expected_projection.parallel
    expected_weak_query = expected_query + 0.25
    expected_revision = (1.0 + gamma) * (weak - expected_weak_query)
    torch.testing.assert_close(alpha, expected_projection.coefficient)
    torch.testing.assert_close(query, expected_query)
    torch.testing.assert_close(weak_query, expected_weak_query)
    torch.testing.assert_close(revision, expected_revision)
