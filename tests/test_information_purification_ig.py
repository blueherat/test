from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.information_purification_ig import (
    four_corner_revision,
    interaction_residualized_guidance,
    lambda_residualized_guidance,
    projected_information_query,
)


def test_four_corner_identities_are_exact() -> None:
    strong_now = torch.tensor([[5.0, 7.0], [2.0, 4.0]])
    weak_now = torch.tensor([[1.0, 3.0], [0.0, 1.0]])
    strong_query = torch.tensor([[6.0, 6.0], [4.0, 3.0]])
    weak_query = torch.tensor([[2.5, 2.0], [1.0, 2.0]])
    parts = four_corner_revision(
        strong_now, weak_now, strong_query, weak_query
    )
    torch.testing.assert_close(
        parts.gap_query,
        parts.gap_now - parts.interaction_revision,
    )
    torch.testing.assert_close(
        parts.cross_corner_gap,
        parts.gap_now - parts.weak_revision,
    )
    torch.testing.assert_close(
        parts.gap_query - parts.gap_now,
        parts.strong_revision - parts.weak_revision,
    )


def test_lambda_endpoints_recover_ordinary_ig_and_pfr() -> None:
    weak = torch.tensor([[1.0, 2.0]])
    strong = torch.tensor([[3.0, 1.0]])
    weak_query = torch.tensor([[1.5, 1.25]])
    gamma = 0.6
    beta = 1.0 + gamma
    guided = weak + beta * (strong - weak)
    revision = weak_query - weak
    ordinary = lambda_residualized_guidance(
        guided, revision, beta=beta, residualization=0.0
    )
    pfr = lambda_residualized_guidance(
        guided, revision, beta=beta, residualization=1.0
    )
    torch.testing.assert_close(ordinary, guided)
    torch.testing.assert_close(pfr, weak + beta * (strong - weak_query))


def test_interaction_residualization_uses_aligned_query_gap() -> None:
    strong_now = torch.tensor([[5.0, 7.0]])
    weak_now = torch.tensor([[1.0, 3.0]])
    strong_query = torch.tensor([[6.0, 6.0]])
    weak_query = torch.tensor([[2.5, 2.0]])
    beta = 1.7
    parts = four_corner_revision(
        strong_now, weak_now, strong_query, weak_query
    )
    guided = weak_now + beta * parts.gap_now
    actual = interaction_residualized_guidance(guided, parts, beta=beta)
    torch.testing.assert_close(actual, weak_now + beta * parts.gap_query)


def test_projected_query_matches_closed_form_ray_projection() -> None:
    state = torch.tensor([[1.0, -1.0], [0.5, 2.0]])
    strong = torch.tensor([[2.0, 0.0], [1.0, 3.0]])
    weak = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    gamma = 0.6
    beta = 1.0 + gamma
    guided = weak + beta * (strong - weak)
    query = projected_information_query(
        state,
        torch.tensor(0.2),
        strong_now=strong,
        weak_now=weak,
        guided_now=guided,
        gamma=gamma,
        horizon=0.05,
        intervention_time=0.5,
    )
    calibration = beta * (strong - weak)
    numerator = (calibration * guided).sum(dim=1)
    denominator = guided.square().sum(dim=1)
    expected_alpha = (numerator / denominator).clamp_min(0.0)
    torch.testing.assert_close(query.projection.coefficient, expected_alpha)
    torch.testing.assert_close(
        query.state,
        state + 0.05 * expected_alpha[:, None] * guided,
    )
    torch.testing.assert_close(query.time, torch.tensor(0.25))
