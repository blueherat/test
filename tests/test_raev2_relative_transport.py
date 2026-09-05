from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.raev2_relative_transport import (
    first_index_at_or_below,
    integrate_euler,
    invert_euler_map_fixed_point,
    invert_euler_step_fixed_point,
    relative_transport_iterate,
)
from experiments.sample_raev2_relative_transport import sample_batches


def _affine_velocity(scale: float, bias: float):
    return lambda state, time: scale * state + bias * time


def test_switch_uses_first_grid_point_at_or_below_boundary():
    assert first_index_at_or_below([1.0, 0.7, 0.49, 0.0], 0.5) == 2


def test_fixed_point_inverse_recovers_one_affine_euler_step():
    source = torch.tensor([[1.0, -2.0], [0.5, 3.0]])
    velocity = _affine_velocity(0.2, -0.1)
    current, following = 0.8, 0.6
    terminal = source + (following - current) * velocity(source, current)
    recovered, iterations, residual, converged = invert_euler_step_fixed_point(
        terminal,
        current,
        following,
        velocity,
        tolerance=1e-10,
        maximum_iterations=32,
    )
    assert converged
    assert iterations < 16
    assert residual <= 1e-10
    torch.testing.assert_close(recovered, source, atol=1e-7, rtol=1e-7)


def test_fixed_point_inverse_recovers_discrete_affine_map():
    source = torch.tensor([[1.0, -2.0], [0.5, 3.0]])
    grid = [1.0, 0.8, 0.55, 0.3]
    velocity = _affine_velocity(-0.15, 0.3)
    terminal = integrate_euler(source, grid, velocity)
    inverse = invert_euler_map_fixed_point(
        terminal,
        grid,
        velocity,
        tolerance=1e-10,
        maximum_iterations=32,
    )
    assert inverse.converged
    assert inverse.maximum_relative_residual <= 1e-10
    torch.testing.assert_close(inverse.state, source, atol=1e-7, rtol=1e-7)


def test_relative_map_iteration_matches_affine_composition():
    noise = torch.tensor([[0.5, -1.5]])
    reference = lambda value: 2.0 * value + 1.0
    guided = lambda value: 3.0 * value - 2.0
    guided_switch = guided(noise)
    inverse_reference_noise = (guided_switch - 1.0) / 2.0
    actual = relative_transport_iterate(
        guided_switch, inverse_reference_noise, guided
    )
    expected = guided((guided(noise) - 1.0) / 2.0)
    torch.testing.assert_close(actual, expected)


def test_relative_iteration_differs_from_linear_map_extrapolation_at_second_order():
    source = torch.tensor([[0.25, -0.5]])

    def reference(value):
        return 2.0 * value + 1.0

    def error(epsilon: float) -> float:
        def guided(value):
            return reference(value) + epsilon * value.square()

        guided_source = guided(source)
        inverse_reference = (guided_source - 1.0) / 2.0
        relative = relative_transport_iterate(
            guided_source, inverse_reference, guided
        )
        linear = 2.0 * guided_source - reference(source)
        return float((relative - linear).norm().item())

    coarse = error(0.2)
    fine = error(0.1)
    assert coarse > 0.0
    assert fine / coarse == pytest.approx(0.25, rel=0.08)


def test_whole_batch_sharding_preserves_global_rng_batch_boundaries():
    shard_zero = list(sample_batches(18, 8, 0, 2))
    shard_one = list(sample_batches(18, 8, 1, 2))
    assert shard_zero == [(0, 8, True), (8, 16, False), (16, 18, True)]
    assert shard_one == [(0, 8, False), (8, 16, True), (16, 18, False)]
