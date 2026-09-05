from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.audit_pfr_channel_consistency import (
    summarize_metrics,
    teacher_channel_query,
)


def test_teacher_channel_query_has_exact_endpoints() -> None:
    state = torch.tensor([[[[1.0, -2.0]]]])
    velocity = torch.tensor([[[[4.0, 6.0]]]])

    clock_only = teacher_channel_query(
        state, velocity, horizon=0.25, rho=0.0
    )
    exact_future = teacher_channel_query(
        state, velocity, horizon=0.25, rho=1.0
    )
    midpoint = teacher_channel_query(
        state, velocity, horizon=0.25, rho=0.5
    )

    torch.testing.assert_close(clock_only, state)
    torch.testing.assert_close(exact_future, state + 0.25 * velocity)
    torch.testing.assert_close(midpoint, 0.5 * (state + exact_future))


@pytest.mark.parametrize("rho", (-0.1, 1.1))
def test_teacher_channel_query_rejects_invalid_rho(rho: float) -> None:
    value = torch.zeros(1, 1, 1, 1)
    with pytest.raises(ValueError, match="rho"):
        teacher_channel_query(value, value, horizon=0.1, rho=rho)


def test_summarize_metrics_preserves_distribution_statistics() -> None:
    result = summarize_metrics(
        {"metric": [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]}
    )
    assert result["metric_mean"] == pytest.approx(2.5)
    assert result["metric_median"] == pytest.approx(2.0)
    assert result["metric_q10"] == pytest.approx(1.3)
    assert result["metric_q90"] == pytest.approx(3.7)
