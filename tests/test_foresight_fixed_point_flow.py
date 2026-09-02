from __future__ import annotations

import pytest
import torch

from experiments.foresight_fixed_point_flow import (
    ForesightEvent,
    anchored_foresight_step,
    conjugated_future_gap_step,
    cross_time_norm_matched_gap_step,
    euler_flow_map,
    foresight_round_trip,
    future_raw_gap_step,
    guided_field,
    implicit_autoguidance_euler_step,
    integrate_flow_map,
    iterate_foresight_operator,
    iterate_anchored_foresight_operator,
    iterate_anchored_gap_operator,
    iterate_conjugated_future_gap_operator,
    local_calibrated_autoguidance_euler_step,
    parse_foresight_schedule,
    schedule_by_step,
    scheduled_autoguidance_euler_step,
    split_guided_euler_step,
)


def test_anchored_operator_preserves_nonzero_constant_gap() -> None:
    state = torch.tensor([[1.0, -2.0]])

    def strong(_time: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return torch.full_like(value, 2.0)

    def weak(_time: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return torch.full_like(value, -1.0)

    endpoint, moves, discrepancies = iterate_anchored_foresight_operator(
        state,
        time_value=torch.tensor(0.2),
        future_time=torch.tensor(0.3),
        iterations=3,
        forward_field=strong,
        inverse_field=weak,
        strength=0.5,
    )
    # F_H(y)-y = 0.1 * (2 - (-1)) for these constant fields.  Anchoring
    # means every iteration returns the same nonzero solution instead of
    # repeatedly marching until the discrepancy is erased.
    expected = state + 0.5 * 0.1 * 3.0
    torch.testing.assert_close(endpoint, expected)
    assert len(moves) == len(discrepancies) == 3
    torch.testing.assert_close(discrepancies[-1], torch.full_like(state, 0.3))
    torch.testing.assert_close(moves[-1], torch.zeros_like(state))


def test_anchored_first_iteration_matches_local_ag_for_constant_fields() -> None:
    state = torch.tensor([[0.5, -0.25]])
    time_value = torch.tensor(0.2)
    future_time = torch.tensor(0.325)
    next_time = torch.tensor(0.225)
    gamma = 3.0

    def strong(_time: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return torch.full_like(value, 2.0)

    def weak(_time: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return torch.full_like(value, -1.0)

    lookahead_steps = 5
    calibrated, _ = anchored_foresight_step(
        state,
        state,
        time_value=time_value,
        future_time=future_time,
        forward_field=strong,
        inverse_field=weak,
        strength=gamma / lookahead_steps,
    )
    actual = calibrated + (next_time - time_value) * strong(time_value, calibrated)
    ag = strong(time_value, state) + gamma * (
        strong(time_value, state) - weak(time_value, state)
    )
    expected = state + (next_time - time_value) * ag
    torch.testing.assert_close(actual, expected)


def test_local_implicit_k1_is_exact_explicit_autoguidance() -> None:
    state = torch.tensor([[0.3, -0.8]])
    time_value = torch.tensor(0.4)
    next_time = torch.tensor(0.425)
    step = next_time - time_value
    gamma = 4.0

    def strong(_time: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return value.square() + 0.2

    def weak(_time: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return -0.3 * value + 0.1

    actual, moves, gaps = implicit_autoguidance_euler_step(
        state,
        time_value=time_value,
        next_time=next_time,
        iterations=1,
        strong_field=strong,
        weak_field=weak,
        gamma=gamma,
    )
    strong_value = strong(time_value, state)
    weak_value = weak(time_value, state)
    expected = state + step * (
        strong_value + gamma * (strong_value - weak_value)
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert len(moves) == len(gaps) == 1


def test_scheduled_ag_is_exact_local_gamma_rescaling() -> None:
    state = torch.tensor([[0.3, -0.8]])
    time_value = torch.tensor(0.4)
    next_time = torch.tensor(0.425)
    gamma = 4.0
    multiplier = 1.75

    def strong(_time: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return value.square() + 0.2

    def weak(_time: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return -0.3 * value + 0.1

    actual = scheduled_autoguidance_euler_step(
        state,
        time_value=time_value,
        next_time=next_time,
        strong_field=strong,
        weak_field=weak,
        gamma=gamma,
        multiplier=multiplier,
    )
    strong_value = strong(time_value, state)
    weak_value = weak(time_value, state)
    expected = state + (next_time - time_value) * (
        strong_value + gamma * multiplier * (strong_value - weak_value)
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_local_calibration_is_zero_horizon_split_control() -> None:
    state = torch.tensor([[0.3, -0.8]])
    time_value = torch.tensor(0.4)
    next_time = torch.tensor(0.425)
    gamma = 4.0
    multiplier = 1.5

    def strong(_time: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return value.square() + 0.2

    def weak(_time: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return -0.3 * value + 0.1

    actual, calibration = local_calibrated_autoguidance_euler_step(
        state,
        time_value=time_value,
        next_time=next_time,
        strong_field=strong,
        weak_field=weak,
        gamma=gamma,
        multiplier=multiplier,
    )
    step = next_time - time_value
    expected_calibration = (
        step
        * gamma
        * multiplier
        * (strong(time_value, state) - weak(time_value, state))
    )
    calibrated = state + expected_calibration
    expected = calibrated + step * strong(time_value, calibrated)
    torch.testing.assert_close(calibration, expected_calibration)
    torch.testing.assert_close(actual, expected)


def test_future_raw_gap_omits_inverse_flow_transport() -> None:
    state = torch.tensor([[0.4, -0.7]])
    times = (torch.tensor(0.1), torch.tensor(0.2), torch.tensor(0.3))

    def strong(_time: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return 0.5 * value + 1.0

    def weak(_time: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return -0.25 * value - 0.5

    strength = 0.075
    actual, gap, move = future_raw_gap_step(
        state,
        time_values=times,
        strong_field=strong,
        weak_field=weak,
        calibration_strength=strength,
        flow_integrator="rk4",
    )
    future = integrate_flow_map(
        state, time_values=times, field=strong, method="rk4"
    )
    expected_gap = strong(times[-1], future) - weak(times[-1], future)
    torch.testing.assert_close(gap, expected_gap)
    torch.testing.assert_close(move, strength * expected_gap)
    torch.testing.assert_close(actual, state + move)


@pytest.mark.parametrize(
    ("direction", "expected_source", "expected_norm"),
    (
        ("future_match_current", "future", "current"),
        ("current_match_future", "current", "future"),
    ),
)
def test_cross_time_gap_matching_separates_direction_and_norm(
    direction: str, expected_source: str, expected_norm: str
) -> None:
    state = torch.tensor([[0.4, -0.7], [1.2, 0.3]])
    times = (torch.tensor(0.1), torch.tensor(0.2), torch.tensor(0.3))

    def strong(time: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return (0.5 + time) * value + 1.0

    def weak(time: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return (-0.25 + 0.5 * time) * value - 0.5

    calibrated, current_gap, future_gap, move = cross_time_norm_matched_gap_step(
        state,
        time_values=times,
        strong_field=strong,
        weak_field=weak,
        calibration_strength=0.075,
        direction=direction,
        flow_integrator="rk4",
    )

    source = future_gap if expected_source == "future" else current_gap
    target = current_gap if expected_norm == "current" else future_gap
    move_direction = move.flatten(1) / move.flatten(1).norm(dim=1, keepdim=True)
    source_direction = source.flatten(1) / source.flatten(1).norm(
        dim=1, keepdim=True
    )
    torch.testing.assert_close(move_direction, source_direction)
    move_rms = move.square().mean(dim=1).sqrt() / 0.075
    target_rms = target.square().mean(dim=1).sqrt()
    torch.testing.assert_close(move_rms, target_rms)
    torch.testing.assert_close(calibrated, state + move)


def test_conjugated_future_gap_recovers_local_ag_for_constant_fields() -> None:
    state = torch.tensor([[0.4, -0.7]])
    times = (torch.tensor(0.1), torch.tensor(0.2), torch.tensor(0.3))

    def strong(_time: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return torch.full_like(value, 1.5)

    def weak(_time: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return torch.full_like(value, -0.5)

    strength = 0.075
    calibrated, gap, future_calibrated = conjugated_future_gap_step(
        state,
        time_values=times,
        strong_field=strong,
        weak_field=weak,
        calibration_strength=strength,
    )
    expected = state + strength * (1.5 - (-0.5))
    torch.testing.assert_close(calibrated, expected)
    torch.testing.assert_close(gap, torch.full_like(state, 2.0))

    future = euler_flow_map(state, time_values=times, field=strong)
    torch.testing.assert_close(future_calibrated, future + strength * gap)


def test_conjugated_future_gap_iterations_are_future_ag_iterations() -> None:
    state = torch.tensor([[0.4, -0.7]])
    times = (torch.tensor(0.1), torch.tensor(0.2), torch.tensor(0.3))

    def strong(_time: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return torch.full_like(value, 1.5)

    def weak(_time: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return torch.full_like(value, -0.5)

    endpoint, moves, gaps = iterate_conjugated_future_gap_operator(
        state,
        time_values=times,
        iterations=3,
        strong_field=strong,
        weak_field=weak,
        calibration_strength=0.075,
    )
    torch.testing.assert_close(endpoint, state + 3 * 0.075 * 2.0)
    assert len(moves) == len(gaps) == 3


def test_parse_and_validate_schedule() -> None:
    events = parse_foresight_schedule("15:5:1,0:5:2,5:5:2")
    assert events == (
        ForesightEvent(0, 5, 2),
        ForesightEvent(5, 5, 2),
        ForesightEvent(15, 5, 1),
    )
    assert set(schedule_by_step(events, num_steps=40)) == {0, 5, 15}
    with pytest.raises(ValueError, match="beyond"):
        schedule_by_step((ForesightEvent(38, 5, 1),), num_steps=40)


def test_split_step_matches_guided_euler_exactly() -> None:
    state = torch.tensor([[1.0, -2.0]])
    time_value = torch.tensor(0.2)
    next_time = torch.tensor(0.3)

    def reference(_time: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return 2.0 * value + 1.0

    def target(_time: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return -value + 0.25

    scale = 1.7
    actual, _ = split_guided_euler_step(
        state,
        time_value=time_value,
        next_time=next_time,
        reference_field=reference,
        target_field=target,
        scale=scale,
    )
    expected_velocity = guided_field(
        reference(time_value, state),
        target(time_value, state),
        scale=scale,
    )
    expected = state + (next_time - time_value) * expected_velocity
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_ag_strong_and_weak_decompositions_match_for_one_step() -> None:
    state = torch.tensor([[0.25, -0.5]])
    time_value = torch.tensor(0.4)
    next_time = torch.tensor(0.45)
    gamma = 3.0

    def strong(_time: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return value.square() + 0.4

    def weak(_time: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return -0.3 * value + 0.1

    weak_reference, _ = split_guided_euler_step(
        state,
        time_value=time_value,
        next_time=next_time,
        reference_field=weak,
        target_field=strong,
        scale=1.0 + gamma,
    )
    strong_reference, _ = split_guided_euler_step(
        state,
        time_value=time_value,
        next_time=next_time,
        reference_field=strong,
        target_field=weak,
        scale=-gamma,
    )
    torch.testing.assert_close(weak_reference, strong_reference, rtol=1e-6, atol=1e-7)


def test_long_horizon_operator_depends_on_reference_decomposition() -> None:
    state = torch.tensor([[1.0]])
    time_value = torch.tensor(0.2)
    future_time = torch.tensor(0.7)
    gamma = 2.0

    def strong(_time: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return 2.0 * value

    def weak(_time: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return 0.5 * value

    def ag(_time: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return guided_field(weak(_time, value), strong(_time, value), scale=1 + gamma)

    via_weak = foresight_round_trip(
        state,
        time_value=time_value,
        future_time=future_time,
        forward_field=ag,
        inverse_field=weak,
    )
    via_strong = foresight_round_trip(
        state,
        time_value=time_value,
        future_time=future_time,
        forward_field=ag,
        inverse_field=strong,
    )
    assert not torch.allclose(via_weak, via_strong)


def test_strong_forward_weak_inverse_moves_along_strong_minus_weak() -> None:
    state = torch.tensor([[1.0, -2.0]])
    time_value = torch.tensor(0.2)
    future_time = torch.tensor(0.25)

    def strong(_time: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return torch.full_like(value, 2.0)

    def weak(_time: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return torch.full_like(value, -1.0)

    actual = foresight_round_trip(
        state,
        time_value=time_value,
        future_time=future_time,
        forward_field=strong,
        inverse_field=weak,
    )
    expected = state + (future_time - time_value) * (2.0 - (-1.0))
    torch.testing.assert_close(actual, expected)

    relaxed = foresight_round_trip(
        state,
        time_value=time_value,
        future_time=future_time,
        forward_field=strong,
        inverse_field=weak,
        relaxation=0.25,
    )
    torch.testing.assert_close(relaxed, state + 0.25 * (expected - state))

    with pytest.raises(ValueError, match="relaxation"):
        foresight_round_trip(
            state,
            time_value=time_value,
            future_time=future_time,
            forward_field=strong,
            inverse_field=weak,
            relaxation=0.0,
        )


def test_relaxed_ag_forward_strong_inverse_matches_local_ag_to_first_order() -> None:
    state = torch.tensor([[0.25, -0.5]])
    time_value = torch.tensor(0.2)
    future_time = torch.tensor(0.325)
    next_time = torch.tensor(0.225)
    gamma = 3.0

    def strong(_time: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return torch.full_like(value, 2.0)

    def weak(_time: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return torch.full_like(value, -1.0)

    def ag(_time: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return strong(_time, value) + gamma * (
            strong(_time, value) - weak(_time, value)
        )

    horizon = future_time - time_value
    local_step = next_time - time_value
    calibrated = foresight_round_trip(
        state,
        time_value=time_value,
        future_time=future_time,
        forward_field=ag,
        inverse_field=strong,
        relaxation=float(local_step / horizon),
    )
    actual = calibrated + local_step * strong(time_value, calibrated)
    expected = state + local_step * ag(time_value, state)
    torch.testing.assert_close(actual, expected)


def test_iteration_records_each_round_trip_displacement() -> None:
    state = torch.ones(2, 3)

    def zero(_time: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(value)

    def constant(_time: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(value)

    endpoint, moves = iterate_foresight_operator(
        state,
        time_value=torch.tensor(0.0),
        future_time=torch.tensor(0.1),
        iterations=3,
        forward_field=constant,
        inverse_field=zero,
    )
    assert len(moves) == 3
    torch.testing.assert_close(endpoint, state + 0.3)
