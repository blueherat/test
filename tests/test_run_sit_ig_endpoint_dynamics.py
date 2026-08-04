import numpy as np

from experiments.run_sit_ig_endpoint_dynamics import (
    build_schedules,
    interaction_metrics,
)
from experiments.run_raev2_ig_impulse_response import equal_step_ranges


def test_schedule_builder_has_pulses_windows_and_interaction_corners():
    schedules = build_schedules(
        num_steps=10,
        pulse_steps=(2,),
        pulse_gammas=(0.01, 0.05),
        windows=equal_step_ranges(10, 5),
        window_gamma=0.01,
        interaction_windows=(0, 2, 4),
        include_pulses=True,
        include_windows=True,
        include_interactions=True,
    )
    names = {item.name for item in schedules}
    assert len(schedules) == 27
    assert {"baseline", "interaction_w0_w4_pp", "interaction_w0_w4_mm"} <= names
    pulse = [item for item in schedules if item.family == "pulse"]
    assert sorted(abs(item.coefficient(2)) for item in pulse) == [0.01, 0.01, 0.05, 0.05]


def test_interaction_only_keeps_required_single_window_controls():
    schedules = build_schedules(
        num_steps=10,
        pulse_steps=(),
        pulse_gammas=(0.01,),
        windows=equal_step_ranges(10, 5),
        window_gamma=0.01,
        interaction_windows=(0, 4),
        include_pulses=False,
        include_windows=False,
        include_interactions=True,
    )
    assert {item.window_index for item in schedules if item.family == "window"} == {0, 4}
    assert len(schedules) == 9


def test_interaction_metrics_vanish_for_additive_map():
    baseline = np.zeros((3, 1, 2))
    a = np.array([[[1.0, 2.0]], [[2.0, 1.0]], [[1.0, -1.0]]])
    b = np.array([[[3.0, 1.0]], [[-1.0, 2.0]], [[2.0, 2.0]]])
    gamma = 0.05
    endpoint = lambda alpha, beta: baseline + alpha * a + beta * b
    result = interaction_metrics(
        baseline,
        endpoint(gamma, 0), endpoint(-gamma, 0),
        endpoint(0, gamma), endpoint(0, -gamma),
        endpoint(gamma, gamma), endpoint(gamma, -gamma),
        endpoint(-gamma, gamma), endpoint(-gamma, -gamma),
        gamma=gamma,
    )
    np.testing.assert_allclose(result["derivative_relative_error"], 0.0, atol=1e-12)
    np.testing.assert_allclose(result["derivative_cosine"], 1.0, atol=1e-12)
    np.testing.assert_allclose(result["mixed_over_joint"], 0.0, atol=1e-12)


def test_interaction_metrics_detect_bilinear_term():
    baseline = np.zeros((2, 1, 2))
    a, b, mixed = np.ones_like(baseline), np.full_like(baseline, 2.0), np.full_like(baseline, 10.0)
    gamma = 0.1
    endpoint = lambda alpha, beta: baseline + alpha * a + beta * b + alpha * beta * mixed
    result = interaction_metrics(
        baseline,
        endpoint(gamma, 0), endpoint(-gamma, 0),
        endpoint(0, gamma), endpoint(0, -gamma),
        endpoint(gamma, gamma), endpoint(gamma, -gamma),
        endpoint(-gamma, gamma), endpoint(-gamma, -gamma),
        gamma=gamma,
    )
    assert np.all(result["mixed_over_joint"] > 0)
