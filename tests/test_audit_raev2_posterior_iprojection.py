from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.audit_raev2_posterior_iprojection import (
    finite_candidate_metrics,
    first_crossing_linear,
    jacobian_pair_metrics,
)


def test_first_crossing_interpolates_and_reports_missing_curve() -> None:
    coordinates = torch.tensor([0.0, 0.5, 1.0, 2.0])
    values = torch.tensor(
        [
            [0.0, 1.0, 2.0, 4.0],
            [0.0, 0.2, 0.3, 0.4],
        ]
    )
    roots, found, monotone = first_crossing_linear(
        coordinates, values, torch.tensor([3.0, 1.0])
    )
    assert roots[0].item() == pytest.approx(1.5)
    assert torch.isnan(roots[1])
    assert found.tolist() == [True, False]
    assert monotone.tolist() == [True, True]


def test_first_crossing_flags_nonmonotone_prefix() -> None:
    coordinates = torch.tensor([0.0, 1.0, 2.0, 3.0])
    values = torch.tensor([[0.0, 0.8, 0.7, 1.2]])
    roots, found, monotone = first_crossing_linear(
        coordinates, values, torch.tensor([1.0])
    )
    assert found.item()
    assert roots.item() == pytest.approx(2.6)
    assert not monotone.item()


def test_jacobian_pair_metrics_identifies_symmetric_action() -> None:
    direction = torch.tensor([[[[1.0, 0.0]]]])
    action = torch.tensor([[[[2.0, 1.0]]]])
    metrics = jacobian_pair_metrics(direction, action, action)
    assert metrics["jvp_vjp_cosine"].item() == pytest.approx(1.0)
    assert metrics["antisymmetric_rms"].item() == pytest.approx(0.0)
    assert metrics["symmetric_rayleigh"].item() == pytest.approx(1.0)


def test_finite_candidate_reports_same_progress_with_orthogonal_change() -> None:
    gap = torch.tensor([[[[2.0, 0.0]]]])
    gamma = 0.5
    target = gamma * gap.flatten(1).square().mean(dim=1)
    shift = torch.tensor([[[[1.0, 1.0]]]])
    metrics = finite_candidate_metrics(gap, shift, target, ordinary_scale=gamma)
    assert metrics["finite_progress_ratio"].item() == pytest.approx(1.0)
    assert metrics["finite_parallel_scale"].item() == pytest.approx(gamma)
    assert metrics["finite_orthogonal_over_gap"].item() == pytest.approx(0.5)
    assert metrics["finite_shift_over_ordinary"].item() == pytest.approx(2.0**0.5)
