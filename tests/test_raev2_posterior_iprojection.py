from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.raev2_posterior_iprojection import (
    first_crossing_bracket,
    reflect_same_progress_shift,
    regula_falsi_coordinate,
    same_progress_shift,
    sample_mean_product,
    update_crossing_bracket,
)


def test_first_crossing_and_regula_falsi_use_earliest_interval() -> None:
    coordinates = torch.tensor([0.0, 1.0, 2.0, 3.0])
    values = torch.tensor([[0.0, 0.4, 1.4, 0.2], [0.0, 0.5, 0.8, 1.1]])
    targets = torch.tensor([1.0, 1.0])
    bracket = first_crossing_bracket(coordinates, values, targets)
    assert torch.equal(bracket.found, torch.tensor([True, True]))
    assert torch.equal(bracket.monotone_to_crossing, torch.tensor([True, True]))
    assert torch.allclose(bracket.lower_coordinate, torch.tensor([1.0, 2.0]))
    assert torch.allclose(bracket.upper_coordinate, torch.tensor([2.0, 3.0]))
    assert torch.allclose(regula_falsi_coordinate(bracket, targets), torch.tensor([1.6, 2.6666667]))


def test_bracket_update_preserves_target_crossing() -> None:
    coordinates = torch.tensor([0.0, 1.0, 2.0])
    values = torch.tensor([[0.0, 0.5, 1.5], [0.0, 0.2, 1.2]])
    targets = torch.ones(2)
    bracket = first_crossing_bracket(coordinates, values, targets)
    coordinate = torch.tensor([1.5, 1.8])
    value = torch.tensor([0.9, 1.1])
    updated = update_crossing_bracket(bracket, coordinate, value, targets)
    assert torch.allclose(updated.lower_coordinate, torch.tensor([1.5, 1.0]))
    assert torch.allclose(updated.upper_coordinate, torch.tensor([2.0, 1.8]))
    assert torch.all(updated.lower_value < targets)
    assert torch.all(updated.upper_value >= targets)


def test_same_progress_and_reflection_keep_gap_projection() -> None:
    gap = torch.tensor([[[[1.0, 0.0]]], [[[1.0, 1.0]]]])
    response = torch.tensor([[[[1.0, 2.0]]], [[[2.0, 0.0]]]])
    gamma = 0.75
    candidate, valid = same_progress_shift(response, gap, gamma)
    reflected = reflect_same_progress_shift(candidate, gap)
    target = gamma * sample_mean_product(gap, gap)
    assert torch.equal(valid, torch.tensor([True, True]))
    assert torch.allclose(sample_mean_product(gap, candidate), target)
    assert torch.allclose(sample_mean_product(gap, reflected), target)
    ordinary = gamma * gap
    assert torch.allclose(reflected - ordinary, -(candidate - ordinary))


def test_reflection_preserves_imperfect_candidate_projection_and_norm() -> None:
    gap = torch.tensor([[[[1.0, 0.0]]]])
    candidate = torch.tensor([[[[0.7, 0.9]]]])
    reflected = reflect_same_progress_shift(candidate, gap)
    assert torch.allclose(
        sample_mean_product(gap, reflected), sample_mean_product(gap, candidate)
    )
    assert torch.allclose(
        sample_mean_product(reflected, reflected),
        sample_mean_product(candidate, candidate),
    )


def test_nonpositive_response_falls_back_to_ordinary() -> None:
    gap = torch.tensor([[[[1.0, 0.0]]]])
    response = torch.tensor([[[[-1.0, 2.0]]]])
    candidate, valid = same_progress_shift(response, gap, 0.5)
    assert not bool(valid.item())
    assert torch.equal(candidate, 0.5 * gap)
