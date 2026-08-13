from __future__ import annotations

import pytest
import torch

from experiments.finite_guidance_dynamics import (
    central_difference_metrics,
    integrate_baseline_tangent,
    integrate_frozen_closed_sweep,
    integrate_guidance_sweep,
    jacobian_symmetry_probe,
    linearity_metrics,
    velocity_gap_to_score_gap,
)


def _linear_anchor(time_value: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    del time_value
    return 0.4 * state


def _constant_direction(
    time_value: torch.Tensor,
    state: torch.Tensor,
    anchor: torch.Tensor,
) -> torch.Tensor:
    del time_value, anchor
    return torch.full_like(state, 0.7)


def _state_direction(
    time_value: torch.Tensor,
    state: torch.Tensor,
    anchor: torch.Tensor,
) -> torch.Tensor:
    del time_value, anchor
    return 0.25 + 0.3 * state


def test_tangent_matches_paired_central_difference() -> None:
    initial = torch.tensor([[[-0.4, 0.2, 0.8]], [[0.5, -0.1, 0.3]]])
    times = torch.linspace(0.0, 1.0, 401)
    baseline, tangent = integrate_baseline_tangent(
        _linear_anchor,
        _state_direction,
        initial,
        times,
    )
    # fp32 endpoint subtraction becomes cancellation-limited around 1e-3.
    delta = 1e-2
    endpoints = integrate_guidance_sweep(
        _linear_anchor,
        _state_direction,
        initial,
        times,
        torch.tensor([-delta, 0.0, delta]),
    )
    assert torch.equal(endpoints[1], baseline)
    metrics = central_difference_metrics(
        endpoints[0],
        endpoints[2],
        tangent,
        delta=delta,
    )
    assert metrics["cosine"].min().item() > 0.99999
    assert metrics["relative_residual"].max().item() < 2e-4


def test_linearity_metrics_are_exact_for_constant_direction() -> None:
    initial = torch.tensor([[[0.2, -0.3]], [[0.7, 0.1]]])
    times = torch.linspace(0.0, 1.0, 101)
    baseline, tangent = integrate_baseline_tangent(
        _linear_anchor,
        _constant_direction,
        initial,
        times,
    )
    gamma = 0.75
    guided = integrate_guidance_sweep(
        _linear_anchor,
        _constant_direction,
        initial,
        times,
        torch.tensor([gamma]),
    )[0]
    metrics = linearity_metrics(baseline, guided, tangent, gamma=gamma)
    assert metrics["cosine"].min().item() > 0.999999
    assert metrics["relative_residual"].max().item() < 2e-6
    assert metrics["magnitude_ratio"].mean().item() == pytest.approx(1.0, abs=2e-6)


def test_frozen_and_closed_are_equal_for_state_independent_direction() -> None:
    initial = torch.tensor([[[0.2, -0.3]], [[0.7, 0.1]]])
    times = torch.linspace(0.0, 1.0, 101)
    gammas = torch.tensor([0.0, 0.2, 1.0])
    baseline, frozen, closed = integrate_frozen_closed_sweep(
        _linear_anchor,
        _constant_direction,
        initial,
        times,
        gammas,
    )
    assert torch.equal(frozen[0], baseline)
    assert torch.equal(closed[0], baseline)
    torch.testing.assert_close(frozen, closed, rtol=0.0, atol=0.0)


def test_state_dependent_direction_separates_frozen_and_closed() -> None:
    initial = torch.tensor([[[0.2, -0.3]], [[0.7, 0.1]]])
    times = torch.linspace(0.0, 1.0, 101)
    baseline, frozen, closed = integrate_frozen_closed_sweep(
        _linear_anchor,
        _state_direction,
        initial,
        times,
        torch.tensor([0.0, 1.0]),
    )
    assert torch.equal(frozen[0], baseline)
    assert torch.equal(closed[0], baseline)
    assert not torch.allclose(frozen[1], closed[1], rtol=1e-5, atol=1e-6)


def test_invalid_linearity_gamma_is_rejected() -> None:
    value = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="gamma=0"):
        linearity_metrics(value, value, value, gamma=0.0)


def test_velocity_to_score_gap_uses_linear_path_orientation() -> None:
    direction = torch.ones(2, 3)
    times = torch.tensor([0.2, 0.8])
    converted = velocity_gap_to_score_gap(direction, times)
    torch.testing.assert_close(
        converted,
        torch.tensor([[0.25, 0.25, 0.25], [4.0, 4.0, 4.0]]),
    )
    with pytest.raises(ValueError, match="strictly inside"):
        velocity_gap_to_score_gap(direction[:1], torch.tensor([1.0]))


def test_jacobian_symmetry_probe_accepts_gradient_field() -> None:
    matrix = torch.tensor([[2.0, -0.5], [-0.5, 1.0]])
    state = torch.randn(7, 2)
    probe = torch.sign(torch.randn_like(state))

    metrics = jacobian_symmetry_probe(lambda value: value @ matrix.T, state, probe)

    assert metrics["antisymmetric_rms"].max() < 1e-6
    assert metrics["antisymmetric_energy_fraction"].max() < 1e-6
    assert torch.allclose(metrics["jvp_vjp_cosine"], torch.ones(7), atol=1e-6)


def test_jacobian_symmetry_probe_detects_rotation_field() -> None:
    matrix = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    state = torch.randn(7, 2)
    probe = torch.sign(torch.randn_like(state))

    metrics = jacobian_symmetry_probe(lambda value: value @ matrix.T, state, probe)

    assert torch.allclose(
        metrics["antisymmetric_energy_fraction"], torch.ones(7), atol=1e-6
    )
    assert torch.allclose(metrics["jvp_vjp_cosine"], -torch.ones(7), atol=1e-6)
