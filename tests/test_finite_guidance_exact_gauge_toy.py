from __future__ import annotations

import torch

from experiments.run_finite_guidance_exact_gauge_toy import (
    exact_density_action,
    mixture_path_score_velocity,
    ring_means,
)


def test_score_velocity_identity_for_linear_path() -> None:
    means = ring_means(6, 2.5)
    state = torch.tensor([[0.2, -0.5], [1.1, 0.7]], dtype=torch.float64)
    for time_value in (0.1, 0.4, 0.9):
        score, velocity = mixture_path_score_velocity(
            state, time_value, means, data_std=0.2
        )
        recovered = (time_value * velocity - state) / (1.0 - time_value)
        torch.testing.assert_close(score, recovered, rtol=1e-11, atol=1e-11)


def test_rotated_score_has_zero_exact_density_action() -> None:
    means = ring_means(8, 3.0)
    states = torch.tensor(
        [[0.2, -0.5], [1.1, 0.7], [-2.0, 0.25]], dtype=torch.float64
    )
    action = exact_density_action(states, 0.45, means, 0.18, "gauge")
    torch.testing.assert_close(action, torch.zeros_like(action), atol=1e-10, rtol=0.0)


def test_score_control_is_density_active() -> None:
    means = ring_means(8, 3.0)
    states = torch.tensor(
        [[0.2, -0.5], [1.1, 0.7], [-2.0, 0.25]], dtype=torch.float64
    )
    action = exact_density_action(states, 0.45, means, 0.18, "active")
    assert float(action.abs().max()) > 1e-2
