from __future__ import annotations

import torch

from experiments.semigroup_guidance_value import (
    TokenPotentialHead,
    bellman_log_value_target,
    boundary_envelope,
    flow_time_from_heat_variance,
    flow_velocity_to_heat_score,
    heat_variance_from_flow_time,
    potential_gradient_to_velocity_correction,
    velocity_gap_to_heat_score_gap,
)


def test_heat_and_flow_time_are_inverse() -> None:
    time_value = torch.tensor([0.1, 0.25, 0.5, 1.0])
    recovered = flow_time_from_heat_variance(
        heat_variance_from_flow_time(time_value)
    )
    torch.testing.assert_close(recovered, time_value)


def test_velocity_score_conversion_and_gap_are_consistent() -> None:
    state = torch.randn(3, 2, 2, 2)
    first = torch.randn_like(state)
    second = torch.randn_like(state)
    time_value = torch.tensor([0.2, 0.3, 0.4])
    score_difference = flow_velocity_to_heat_score(
        first,
        state=state,
        time_value=time_value,
    ) - flow_velocity_to_heat_score(
        second,
        state=state,
        time_value=time_value,
    )
    expected = velocity_gap_to_heat_score_gap(
        first - second,
        time_value=time_value,
    )
    torch.testing.assert_close(score_difference, expected)


def test_potential_gradient_velocity_conversion_inverts_known_relation() -> None:
    gradient = torch.randn(3, 4)
    time_value = torch.tensor([0.2, 0.4, 0.8])
    correction = potential_gradient_to_velocity_correction(
        gradient,
        time_value=time_value,
    )
    expected = gradient * ((1.0 - time_value) / time_value)[:, None]
    torch.testing.assert_close(correction, expected)


def test_soft_bellman_target_uses_log_mean_exp() -> None:
    next_values = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    cost = torch.tensor([0.5, 0.25])
    step = torch.tensor([0.2, 0.4])
    result = bellman_log_value_target(
        next_values,
        running_cost=cost,
        heat_step=step,
    )
    expected = torch.log(torch.exp(next_values + cost.mul(step)[None]).mean(0))
    torch.testing.assert_close(result, expected)


def test_potential_head_fixes_both_boundary_gauges() -> None:
    head = TokenPotentialHead(16, intervention_time=0.5)
    with torch.no_grad():
        head.value[-1].bias.fill_(3.0)
        head.baseline[-1].bias.fill_(2.0)
    tokens = torch.randn(2, 4, 16)
    conditioning = torch.randn(2, 16)
    values = head(tokens, conditioning, torch.tensor([0.0, 0.5]))
    torch.testing.assert_close(values, torch.tensor([2.0, 0.0]))


def test_boundary_envelope_only_vanishes_at_intervention() -> None:
    values = boundary_envelope(
        torch.tensor([0.0, 0.25, 0.5]),
        intervention_time=0.5,
    )
    torch.testing.assert_close(values, torch.tensor([1.0, 0.5, 0.0]))
