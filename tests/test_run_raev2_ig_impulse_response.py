from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.run_raev2_ig_impulse_response import (
    bootstrap_mean_interval,
    build_interventions,
    build_validation_labels,
    equal_step_ranges,
    euler_x_prediction_step,
    guided_clean_prediction,
    official_shifted_solver_grid,
)


def test_equal_step_ranges_cover_solver_without_overlap() -> None:
    ranges = equal_step_ranges(100, 5)
    assert ranges == ((0, 20), (20, 40), (40, 60), (60, 80), (80, 100))


def test_official_grid_uses_sampler_float32_parameterization() -> None:
    grid = official_shifted_solver_grid(100, 8.0)
    assert grid.dtype == torch.float32
    assert float(grid[67]) == pytest.approx(0.7975831)
    assert float(grid[-1]) == 0.0


def test_interventions_are_symmetric_and_use_one_step_pulses() -> None:
    values = build_interventions(
        num_steps=100,
        pulse_steps=(10, 98),
        window_count=5,
        gamma=0.05,
    )
    assert values[0].name == "baseline"
    pulse = [item for item in values if item.family == "pulse"]
    assert {(item.start_step, item.end_step) for item in pulse} == {(10, 11), (98, 99)}
    for pair in {item.pair_name for item in values if item.pair_name}:
        gammas = sorted(item.gamma for item in values if item.pair_name == pair)
        assert gammas == pytest.approx([-0.05, 0.05])


def test_window_specific_gammas_do_not_change_pulse_gamma() -> None:
    values = build_interventions(
        num_steps=10,
        pulse_steps=(2,),
        window_count=2,
        gamma=0.05,
        window_gammas=(0.01, 0.02),
    )
    pulse = [abs(item.gamma) for item in values if item.family == "pulse"]
    windows = [item.gamma for item in values if item.family == "window" and item.gamma > 0]
    assert pulse == pytest.approx([0.05, 0.05])
    assert windows == pytest.approx([0.01, 0.02])


def test_pulse_gamma_sweep_builds_distinct_signed_pairs() -> None:
    values = build_interventions(
        num_steps=10,
        pulse_steps=(2,),
        window_count=1,
        gamma=0.05,
        pulse_gammas=(0.01, 0.05),
    )
    pulse = [item for item in values if item.family == "pulse"]
    assert len({item.pair_name for item in pulse}) == 2
    assert sorted(abs(item.gamma) for item in pulse) == pytest.approx(
        [0.01, 0.01, 0.05, 0.05]
    )


def test_guided_clean_prediction_matches_official_scale_convention() -> None:
    full = torch.tensor([[2.0], [4.0]])
    base = torch.tensor([[1.0], [1.0]])
    gamma = torch.tensor([0.5, -0.5])
    result = guided_clean_prediction(full, base, gamma)
    assert torch.allclose(result, torch.tensor([[2.5], [2.5]]))


def test_x_prediction_euler_step_has_h_over_t_injection() -> None:
    state = torch.tensor([[3.0]])
    full = torch.tensor([[1.0]])
    guided = torch.tensor([[1.5]])
    baseline_next = euler_x_prediction_step(
        state, full, time=0.5, step_size=0.1, t_eps=0.05
    )
    guided_next = euler_x_prediction_step(
        state, guided, time=0.5, step_size=0.1, t_eps=0.05
    )
    assert torch.allclose(guided_next - baseline_next, torch.tensor([[0.1]]))


def test_bootstrap_interval_contains_constant_mean() -> None:
    low, high = bootstrap_mean_interval(
        np.ones(16, dtype=np.float64), repeats=100, seed=3
    )
    assert low == pytest.approx(1.0)
    assert high == pytest.approx(1.0)


def test_validation_labels_can_cover_random_classes_reproducibly() -> None:
    first = build_validation_labels(
        256, 1000, mode="random_without_replacement", seed=17
    )
    second = build_validation_labels(
        256, 1000, mode="random_without_replacement", seed=17
    )
    assert np.array_equal(first, second)
    assert len(np.unique(first)) == 256
    assert first.min() >= 0 and first.max() < 1000
    assert not np.array_equal(first, np.arange(256))


def test_without_replacement_labels_reject_too_many_samples() -> None:
    with pytest.raises(ValueError):
        build_validation_labels(1001, 1000, mode="random_without_replacement", seed=0)
