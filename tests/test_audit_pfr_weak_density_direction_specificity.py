from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.audit_pfr_weak_density_direction_specificity import (
    finite_score_work,
    implied_score,
    make_directions,
)


def test_implied_score_inverts_linear_flow_identity() -> None:
    state = torch.tensor([[[[2.0]]], [[[3.0]]]])
    velocity = torch.tensor([[[[4.0]]], [[[5.0]]]])
    time = torch.tensor(0.25)
    expected_noise = state - time * velocity
    torch.testing.assert_close(
        implied_score(state, time, velocity),
        -expected_noise / (1.0 - time),
    )


def test_five_point_work_is_exact_for_affine_score() -> None:
    state = torch.tensor([[[[1.0, 2.0]]]])
    displacement = torch.tensor([[[[0.5, -1.0]]]])
    time = torch.tensor(0.4)

    # Choose velocity so the implied score is exactly s(z)=2*z+3.
    def velocity(query_time: torch.Tensor, query_state: torch.Tensor) -> torch.Tensor:
        score = 2.0 * query_state + 3.0
        return (query_state + (1.0 - query_time) * score) / query_time

    work, trapezoid, directional = finite_score_work(
        state, displacement, time, velocity
    )
    score_start = 2.0 * state + 3.0
    score_end = 2.0 * (state + displacement) + 3.0
    expected = 0.5 * (
        (score_start + score_end) * displacement
    ).flatten(1).sum(1) / state[0].numel()
    expected_directional = (score_start * displacement).flatten(1).sum(1) / state[0].numel()
    torch.testing.assert_close(work, expected)
    torch.testing.assert_close(trapezoid, expected)
    torch.testing.assert_close(directional, expected_directional)


def test_alternative_directions_match_pfr_norm_per_sample() -> None:
    generator = torch.Generator().manual_seed(7)
    shape = (4, 3, 2, 2)
    values = [torch.randn(shape, generator=generator) for _ in range(7)]
    state, pfr, strong, weak, strong_future, weak_future, score = values
    directions = make_directions(
        state=state,
        pfr_shift=pfr,
        strong_now=strong,
        weak_now=weak,
        strong_future=strong_future,
        weak_future=weak_future,
        gamma=0.7,
        weak_score_future=score,
    )
    target_norm = pfr.flatten(1).norm(dim=1)
    for direction in directions.values():
        torch.testing.assert_close(direction.flatten(1).norm(dim=1), target_norm)
    torch.testing.assert_close(directions["anti_pfr"], -pfr)
    orthogonal_dot = (directions["orthogonal_pfr"] * pfr).flatten(1).sum(1)
    torch.testing.assert_close(orthogonal_dot, torch.zeros_like(orthogonal_dot), atol=2e-6, rtol=0)
