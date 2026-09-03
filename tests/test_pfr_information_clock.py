import math
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.pfr_information_clock import matched_information_horizon


COMMON = {
    "anchor_time": 0.25,
    "anchor_horizon": 1.0 / 32.0,
    "intervention_time": 0.5,
}


@pytest.mark.parametrize("clock", ("raw_t", "log_snr", "snr"))
def test_clocks_match_at_anchor(clock: str) -> None:
    step = matched_information_horizon(0.25, clock=clock, **COMMON)
    assert step == pytest.approx(1.0 / 32.0, abs=1e-12)


def test_raw_time_is_constant_until_taper() -> None:
    assert matched_information_horizon(0.1, clock="raw_t", **COMMON) == pytest.approx(
        1.0 / 32.0
    )
    assert matched_information_horizon(0.49, clock="raw_t", **COMMON) == pytest.approx(
        0.01
    )


def test_log_snr_uses_multiplicative_scale() -> None:
    low = matched_information_horizon(0.05, clock="log_snr", **COMMON)
    high = matched_information_horizon(0.4, clock="log_snr", **COMMON)
    assert 0.0 < low < 1.0 / 32.0 < high
    assert matched_information_horizon(0.0, clock="log_snr", **COMMON) == 0.0


def test_snr_uses_additive_channel_precision() -> None:
    low = matched_information_horizon(0.05, clock="snr", **COMMON)
    high = matched_information_horizon(0.4, clock="snr", **COMMON)
    assert low > 1.0 / 32.0 > high > 0.0


@pytest.mark.parametrize("clock", ("raw_t", "log_snr", "snr"))
def test_clocks_taper_at_intervention(clock: str) -> None:
    step = matched_information_horizon(0.49, clock=clock, **COMMON)
    assert 0.0 <= step <= 0.01 + 1e-12
    assert math.isfinite(step)
    assert matched_information_horizon(0.5, clock=clock, **COMMON) == 0.0


def test_rejects_unknown_clock() -> None:
    with pytest.raises(ValueError, match="unknown information clock"):
        matched_information_horizon(0.25, clock="banana", **COMMON)
