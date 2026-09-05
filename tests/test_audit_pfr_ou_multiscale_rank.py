from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.audit_pfr_ou_multiscale_rank import (
    infer_effective_degree,
    ou_degree_response_ratio,
)


@pytest.mark.parametrize("degree", [2.0, 3.0, 5.5])
def test_effective_degree_inverts_exact_single_mode_ratio(degree: float) -> None:
    time = 0.1
    short_time = 0.13125
    long_time = 0.1625
    ratio = ou_degree_response_ratio(time, short_time, long_time, degree)

    inferred = infer_effective_degree(
        ratio, time, short_time, long_time, grid_size=40000
    )

    assert inferred == pytest.approx(degree, abs=4e-4)


def test_long_horizon_has_larger_single_mode_response() -> None:
    for degree in (2.0, 3.0, 8.0):
        assert ou_degree_response_ratio(0.2, 0.23125, 0.2625, degree) > 1.0


def test_invalid_degree_or_time_order_is_rejected() -> None:
    with pytest.raises(ValueError):
        ou_degree_response_ratio(0.1, 0.2, 0.3, 1.0)
    with pytest.raises(ValueError):
        ou_degree_response_ratio(0.2, 0.1, 0.3, 2.0)
