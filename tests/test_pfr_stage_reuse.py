from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.pfr_stage_reuse import (
    StageStart,
    integrate_stage_reused_heun,
)


class AffineStageField:
    def __init__(self) -> None:
        self.nfe = 0
        self.query_nfe = 0

    def evaluate_start(
        self, time_value: torch.Tensor, state: torch.Tensor
    ) -> StageStart:
        self.nfe += 1
        self.query_nfe += 1
        return StageStart(2.0 * state + time_value, torch.ones_like(state))

    def evaluate_end(
        self,
        time_value: torch.Tensor,
        state: torch.Tensor,
        weak_revision: torch.Tensor | None,
    ) -> torch.Tensor:
        assert weak_revision is not None
        self.nfe += 1
        return 2.0 * state + time_value


def test_stage_reused_heun_uses_one_query_and_two_full_calls_per_step() -> None:
    field = AffineStageField()
    times = torch.linspace(0.0, 1.0, 5)

    result = integrate_stage_reused_heun(
        field, torch.zeros(2, 3), times
    )

    assert result.endpoint.shape == (2, 3)
    assert result.nfe == 8
    assert field.nfe == 8
    assert field.query_nfe == 4


def test_stage_reused_heun_requires_an_increasing_grid() -> None:
    field = AffineStageField()
    try:
        integrate_stage_reused_heun(
            field, torch.zeros(1, 1), torch.tensor([0.0, 0.5, 0.5])
        )
    except ValueError as error:
        assert "strictly increasing" in str(error)
    else:
        raise AssertionError("non-increasing grid was accepted")
