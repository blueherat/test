from __future__ import annotations

import torch

from experiments.run_imagenet100_sit_guidance_conservativity import (
    _collect_rollout_states,
    _teacher_states,
)


def test_teacher_states_follow_linear_bridge() -> None:
    clean = torch.full((2, 3), 2.0)
    noise = torch.full((2, 3), -1.0)
    states = _teacher_states(clean, noise, [0.25, 0.5])
    torch.testing.assert_close(states[0.25], torch.full((2, 3), -0.25))
    torch.testing.assert_close(states[0.5], torch.full((2, 3), 0.5))


def test_collect_rollout_states_matches_constant_velocity() -> None:
    initial = torch.zeros(2, 3)

    def field(time_value: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        del time_value
        return torch.full_like(state, 2.0)

    states = _collect_rollout_states(
        field,
        initial,
        steps=100,
        requested_times=[0.1, 0.5, 0.9],
    )
    for time_value, state in states.items():
        torch.testing.assert_close(state, torch.full_like(state, 2.0 * time_value))
