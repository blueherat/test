from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.pfr_ou_semigroup_spectrum import (
    linear_velocity_to_ou_relative_score,
    ou_bridge_coordinates,
    ou_degree_retiming_velocity_defect,
    ou_degree1_retiming_velocity_defect,
    ou_future_posterior_mean_state,
    ou_mode_retiming_defect,
    ou_relative_score_consistency_velocity_defect,
    ou_relative_score_delta_to_linear_velocity_delta,
    ou_relative_score_to_linear_velocity,
    state_to_ou,
    transport_state_at_fixed_ou_coordinate,
)


@pytest.mark.parametrize("degree", [1.0, 2.0, 3.0])
def test_stable_degree_defect_matches_direct_relative_score_algebra(
    degree: float,
) -> None:
    generator = torch.Generator().manual_seed(91)
    state = torch.randn(3, 4, generator=generator, dtype=torch.float64)
    current = torch.randn(3, 4, generator=generator, dtype=torch.float64)
    future = torch.randn(3, 4, generator=generator, dtype=torch.float64)
    time = torch.tensor(0.17, dtype=torch.float64)
    future_time = torch.tensor(0.31, dtype=torch.float64)
    future_state = transport_state_at_fixed_ou_coordinate(
        state, time, future_time
    )
    current_score = linear_velocity_to_ou_relative_score(current, state, time)
    future_score = linear_velocity_to_ou_relative_score(
        future, future_state, future_time
    )
    direct_score_defect = ou_mode_retiming_defect(
        current_score,
        future_score,
        time,
        future_time,
        degree=degree,
    )
    direct = ou_relative_score_delta_to_linear_velocity_delta(
        direct_score_defect, state, time
    )
    stable = ou_degree_retiming_velocity_defect(
        current,
        future,
        state,
        time,
        future_time,
        degree=degree,
    )
    torch.testing.assert_close(stable, direct, atol=1e-11, rtol=1e-11)


def test_higher_degree_defect_has_finite_zero_time_limit() -> None:
    state = torch.tensor([[1.0, -2.0]], dtype=torch.float64)
    current = torch.tensor([[0.25, 0.5]], dtype=torch.float64)
    future = torch.tensor([[4.0, -3.0]], dtype=torch.float64)
    defect = ou_degree_retiming_velocity_defect(
        current,
        future,
        state,
        0.0,
        0.2,
        degree=2.0,
    )
    torch.testing.assert_close(defect, current + state)


def test_ou_coordinate_is_variance_preserving() -> None:
    reference = torch.zeros(3, 2, dtype=torch.float64)
    times = torch.tensor([0.1, 0.4, 0.8], dtype=torch.float64)
    coordinates = ou_bridge_coordinates(times, reference)
    torch.testing.assert_close(
        coordinates.signal.square() + coordinates.noise.square(),
        torch.ones_like(coordinates.signal),
    )


def test_relative_score_velocity_round_trip() -> None:
    generator = torch.Generator().manual_seed(190)
    state = torch.randn(4, 3, 2, 2, generator=generator, dtype=torch.float64)
    velocity = torch.randn(4, 3, 2, 2, generator=generator, dtype=torch.float64)
    times = torch.tensor([0.1, 0.25, 0.5, 0.8], dtype=torch.float64)
    relative_score = linear_velocity_to_ou_relative_score(velocity, state, times)
    reconstructed = ou_relative_score_to_linear_velocity(
        relative_score, state, times
    )
    torch.testing.assert_close(reconstructed, velocity, rtol=1e-12, atol=1e-12)


def test_standard_gaussian_bridge_has_zero_relative_score() -> None:
    generator = torch.Generator().manual_seed(191)
    state = torch.randn(4, 5, generator=generator, dtype=torch.float64)
    times = torch.tensor([0.1, 0.3, 0.6, 0.9], dtype=torch.float64)
    shaped = times[:, None]
    scale_squared = shaped.square() + (1.0 - shaped).square()
    gaussian_velocity = (2.0 * shaped - 1.0) / scale_squared * state
    relative_score = linear_velocity_to_ou_relative_score(
        gaussian_velocity, state, times
    )
    torch.testing.assert_close(
        relative_score, torch.zeros_like(relative_score), rtol=1e-12, atol=1e-12
    )


def test_fixed_ou_coordinate_transport_is_exact() -> None:
    generator = torch.Generator().manual_seed(192)
    state = torch.randn(3, 2, 4, 4, generator=generator, dtype=torch.float64)
    current = torch.tensor([0.1, 0.2, 0.4], dtype=torch.float64)
    future = current + 1.0 / 32.0
    transported = transport_state_at_fixed_ou_coordinate(state, current, future)
    torch.testing.assert_close(
        state_to_ou(state, current), state_to_ou(transported, future)
    )


def test_degree_matched_ou_mode_has_zero_defect() -> None:
    generator = torch.Generator().manual_seed(193)
    template = torch.randn(3, 7, generator=generator, dtype=torch.float64)
    current = torch.tensor([0.1, 0.25, 0.45], dtype=torch.float64)
    future = current + 1.0 / 32.0
    current_alpha = ou_bridge_coordinates(current, template).signal
    future_alpha = ou_bridge_coordinates(future, template).signal
    for degree in (1.0, 2.0, 3.0):
        current_mode = current_alpha.pow(degree) * template
        future_mode = future_alpha.pow(degree) * template
        defect = ou_mode_retiming_defect(
            current_mode,
            future_mode,
            current,
            future,
            degree=degree,
        )
        torch.testing.assert_close(
            defect, torch.zeros_like(defect), rtol=1e-12, atol=1e-12
        )


def test_relative_score_delta_conversion_matches_velocity_difference() -> None:
    generator = torch.Generator().manual_seed(194)
    state = torch.randn(3, 2, 3, 3, generator=generator, dtype=torch.float64)
    times = torch.tensor([0.1, 0.35, 0.7], dtype=torch.float64)
    first = torch.randn_like(state, generator=generator)
    second = torch.randn_like(state, generator=generator)
    first_velocity = ou_relative_score_to_linear_velocity(first, state, times)
    second_velocity = ou_relative_score_to_linear_velocity(second, state, times)
    converted = ou_relative_score_delta_to_linear_velocity_delta(
        first - second, state, times
    )
    torch.testing.assert_close(
        converted, first_velocity - second_velocity, rtol=1e-12, atol=1e-12
    )


def test_stable_degree1_velocity_defect_matches_score_space_formula() -> None:
    generator = torch.Generator().manual_seed(195)
    state = torch.randn(3, 4, 2, 2, generator=generator, dtype=torch.float64)
    current = torch.randn(3, 4, 2, 2, generator=generator, dtype=torch.float64)
    future = torch.randn(3, 4, 2, 2, generator=generator, dtype=torch.float64)
    time = torch.full((len(state),), 0.2, dtype=torch.float64)
    future_time = torch.full((len(state),), 0.27, dtype=torch.float64)
    future_state = transport_state_at_fixed_ou_coordinate(
        state, time, future_time
    )

    current_relative = linear_velocity_to_ou_relative_score(
        current, state, time
    )
    future_relative = linear_velocity_to_ou_relative_score(
        future, future_state, future_time
    )
    score_defect = ou_mode_retiming_defect(
        current_relative,
        future_relative,
        time,
        future_time,
        degree=1.0,
    )
    expected = ou_relative_score_delta_to_linear_velocity_delta(
        score_defect, state, time
    )
    actual = ou_degree1_retiming_velocity_defect(
        current, future, state, time, future_time
    )

    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)


def test_stable_degree1_velocity_defect_has_finite_zero_time_limit() -> None:
    generator = torch.Generator().manual_seed(196)
    state = torch.randn(2, 3, 2, 2, generator=generator, dtype=torch.float64)
    current = torch.randn(2, 3, 2, 2, generator=generator, dtype=torch.float64)
    future = torch.randn(2, 3, 2, 2, generator=generator, dtype=torch.float64)

    endpoint = ou_degree1_retiming_velocity_defect(
        current, future, state, 0.0, 0.04
    )
    nearby = ou_degree1_retiming_velocity_defect(
        current, future, state, 1e-9, 0.04
    )

    assert torch.isfinite(endpoint).all()
    torch.testing.assert_close(endpoint, nearby, atol=1e-7, rtol=1e-7)


def test_posterior_mean_state_matches_direct_ou_tweedie_formula() -> None:
    generator = torch.Generator().manual_seed(197)
    state = torch.randn(3, 4, generator=generator, dtype=torch.float64)
    velocity = torch.randn(3, 4, generator=generator, dtype=torch.float64)
    time = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    future_time = time + 0.07
    current = ou_bridge_coordinates(time, state)
    future = ou_bridge_coordinates(future_time, state)
    relative = linear_velocity_to_ou_relative_score(velocity, state, time)
    channel_signal = current.signal / future.signal
    expected_normalized = (
        channel_signal * state_to_ou(state, time)
        + (1.0 - channel_signal.square()) / channel_signal * relative
    )
    expected = future.scale * expected_normalized

    actual = ou_future_posterior_mean_state(
        velocity, state, time, future_time
    )

    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)


def test_posterior_mean_state_has_finite_zero_time_limit() -> None:
    generator = torch.Generator().manual_seed(198)
    state = torch.randn(2, 5, generator=generator, dtype=torch.float64)
    velocity = torch.randn(2, 5, generator=generator, dtype=torch.float64)

    endpoint = ou_future_posterior_mean_state(
        velocity, state, 0.0, 0.04
    )
    nearby = ou_future_posterior_mean_state(
        velocity, state, 1e-9, 0.04
    )

    assert torch.isfinite(endpoint).all()
    torch.testing.assert_close(endpoint, nearby, atol=1e-7, rtol=1e-7)


def test_relative_score_consistency_defect_vanishes_for_gaussian_path() -> None:
    state = torch.tensor(
        [[-1.2, 0.3, 1.7], [0.8, -0.4, 2.1]], dtype=torch.float64
    )
    time = torch.tensor([0.2, 0.35], dtype=torch.float64)
    future_time = time + 0.08
    data_mean = 0.7
    data_variance = 1.8

    def gaussian_relative_score(
        raw_state: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        coordinates = ou_bridge_coordinates(values, raw_state)
        normalized = raw_state / coordinates.scale
        variance = (
            coordinates.signal.square() * data_variance
            + coordinates.noise.square()
        )
        score = -(
            normalized - coordinates.signal * data_mean
        ) / variance
        return score + normalized

    current_relative = gaussian_relative_score(state, time)
    current_velocity = ou_relative_score_to_linear_velocity(
        current_relative, state, time
    )
    future_state = ou_future_posterior_mean_state(
        current_velocity, state, time, future_time
    )
    future_relative = gaussian_relative_score(future_state, future_time)
    future_velocity = ou_relative_score_to_linear_velocity(
        future_relative, future_state, future_time
    )
    defect = ou_relative_score_consistency_velocity_defect(
        current_velocity,
        future_velocity,
        state,
        future_state,
        time,
        future_time,
    )

    torch.testing.assert_close(
        defect, torch.zeros_like(defect), atol=2e-12, rtol=0.0
    )
