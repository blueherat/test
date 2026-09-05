from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.pfr_exponential_retiming import (
    compose_retiming_weights,
    exponential_retiming_defect,
    linear_velocity_to_score,
    reinterpret_future_velocity_score,
    retime_future_score,
    retiming_weight,
    score_to_linear_velocity,
    split_exponential_retiming_defect,
)


def test_velocity_score_round_trip() -> None:
    generator = torch.Generator().manual_seed(4)
    state = torch.randn(3, 2, 4, 4, generator=generator, dtype=torch.float64)
    score = torch.randn(3, 2, 4, 4, generator=generator, dtype=torch.float64)
    times = torch.tensor([0.1, 0.3, 0.7], dtype=torch.float64)
    velocity = score_to_linear_velocity(score, state, times)
    torch.testing.assert_close(
        linear_velocity_to_score(velocity, state, times), score
    )


def test_future_velocity_is_exact_exponential_retiming() -> None:
    generator = torch.Generator().manual_seed(5)
    state = torch.randn(4, 3, 2, 2, generator=generator, dtype=torch.float64)
    future_score = torch.randn(
        4, 3, 2, 2, generator=generator, dtype=torch.float64
    )
    current = torch.tensor([0.05, 0.1, 0.25, 0.4], dtype=torch.float64)
    future = current + 1.0 / 32.0
    future_velocity = score_to_linear_velocity(future_score, state, future)
    direct = reinterpret_future_velocity_score(future_velocity, state, current)
    geometric = retime_future_score(
        future_score, state, current, future
    )
    torch.testing.assert_close(direct, geometric, rtol=1e-12, atol=1e-12)
    weight = retiming_weight(current, future, state)
    assert torch.all((weight > 0.0) & (weight < 1.0))


def test_retiming_weights_and_scores_form_a_semigroup() -> None:
    generator = torch.Generator().manual_seed(6)
    state = torch.randn(2, 3, 2, 2, generator=generator, dtype=torch.float64)
    score = torch.randn(2, 3, 2, 2, generator=generator, dtype=torch.float64)
    direct_weight, two_hop_weight = compose_retiming_weights(
        0.1, 0.25, 0.4, state
    )
    torch.testing.assert_close(direct_weight, two_hop_weight)
    direct = retime_future_score(score, state, 0.1, 0.4)
    middle = retime_future_score(score, state, 0.25, 0.4)
    two_hop = retime_future_score(middle, state, 0.1, 0.25)
    torch.testing.assert_close(direct, two_hop)


def test_defect_split_and_cocycle_are_exact() -> None:
    generator = torch.Generator().manual_seed(7)
    state = torch.randn(2, 3, 2, 2, generator=generator, dtype=torch.float64)
    early = torch.randn_like(state, generator=generator)
    middle = torch.randn_like(state, generator=generator)
    late = torch.randn_like(state, generator=generator)
    evolution, gaussian = split_exponential_retiming_defect(
        early, middle, state, 0.1, 0.25
    )
    defect = exponential_retiming_defect(early, middle, state, 0.1, 0.25)
    torch.testing.assert_close(evolution + gaussian, defect)

    direct = exponential_retiming_defect(early, late, state, 0.1, 0.4)
    first = exponential_retiming_defect(early, middle, state, 0.1, 0.25)
    second = exponential_retiming_defect(middle, late, state, 0.25, 0.4)
    weight = retiming_weight(0.1, 0.25, state)
    torch.testing.assert_close(direct, first + weight * second)
