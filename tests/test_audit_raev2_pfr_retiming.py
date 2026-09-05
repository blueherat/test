from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.audit_raev2_pfr_retiming import (  # noqa: E402
    _projection_fraction,
    nearest_indices,
)


def test_projection_fraction_has_expected_extremes() -> None:
    reference = torch.tensor([[1.0, 0.0]])
    assert float(_projection_fraction(reference, reference)) == 1.0
    orthogonal = torch.tensor([[0.0, 2.0]])
    assert float(_projection_fraction(orthogonal, reference)) == 0.0


def test_nearest_indices_uses_solver_states_not_terminal() -> None:
    grid = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
    indices = nearest_indices(grid)
    assert all(index < len(grid) - 1 for index in indices)
