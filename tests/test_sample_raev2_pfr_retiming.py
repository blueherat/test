from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.sample_raev2_pfr_retiming import (  # noqa: E402
    SamplingCondition,
    dataward_future_time,
    parse_condition,
    parse_piecewise_condition,
    parse_window_condition,
    shifted_time_grid,
)


def test_parse_condition_round_trip() -> None:
    condition = parse_condition("pfr,1.78,0.03125,1")
    assert condition == SamplingCondition("pfr", 1.78, 0.03125, 1.0)
    assert condition.uses_retiming
    with pytest.raises(Exception):
        parse_condition("broken")


def test_scalar_guidance_schedule_preserves_historical_scale() -> None:
    condition = parse_condition("ordinary,1.78,0,0")
    assert not condition.uses_piecewise_guidance
    assert condition.guidance_scale_at(0.95) == pytest.approx(1.78)
    assert condition.guidance_scale_at(0.75) == pytest.approx(1.78)
    assert condition.guidance_scale_at(0.25) == pytest.approx(1.78)


def test_piecewise_guidance_uses_early_mid_and_strong_only_intervals() -> None:
    condition = parse_piecewise_condition("scheduled,1.6,1.7")
    assert condition.uses_piecewise_guidance
    assert condition.guidance_scale_at(0.95) == pytest.approx(1.6)
    assert condition.guidance_scale_at(0.750001) == pytest.approx(1.6)
    assert condition.guidance_scale_at(0.75) == pytest.approx(1.7)
    assert condition.guidance_scale_at(0.500001) == pytest.approx(1.7)
    assert condition.guidance_scale_at(0.5) == pytest.approx(1.0)
    assert condition.guidance_scale_at(0.1) == pytest.approx(1.0)


def test_piecewise_guidance_can_wrap_a_pathwise_revision() -> None:
    condition = parse_piecewise_condition(
        "scheduled_pfr,1.6,1.7,0.03125,1,pathwise_first_half_retiming"
    )
    assert condition.guidance_scale == pytest.approx(1.6)
    assert condition.mid_guidance_scale == pytest.approx(1.7)
    assert condition.uses_pathwise_retiming


def test_window_guidance_uses_scale_only_inside_requested_interval() -> None:
    condition = parse_window_condition("windowed,1.78,0.4,1")
    assert condition.uses_window_guidance
    assert not condition.uses_piecewise_guidance
    assert condition.guidance_scale_at(1.0) == pytest.approx(1.78)
    assert condition.guidance_scale_at(0.4) == pytest.approx(1.78)
    assert condition.guidance_scale_at(0.399999) == pytest.approx(1.0)


def test_window_guidance_can_wrap_a_pathwise_revision() -> None:
    condition = parse_window_condition(
        "windowed_pfr,1.6,0.4,1,0.03125,1,pathwise_first_half_retiming"
    )
    assert condition.guidance_min_time == pytest.approx(0.4)
    assert condition.guidance_max_time == pytest.approx(1.0)
    assert condition.uses_pathwise_retiming


def test_guidance_condition_rejects_partial_or_overlapping_schedules() -> None:
    partial = SamplingCondition("partial", 1.78, 0.0, 0.0, guidance_min_time=0.4)
    with pytest.raises(ValueError):
        partial.validate()
    overlapping = SamplingCondition(
        "overlapping",
        1.78,
        0.0,
        0.0,
        mid_guidance_scale=1.6,
        guidance_min_time=0.4,
        guidance_max_time=1.0,
    )
    with pytest.raises(ValueError):
        overlapping.validate()


def test_shifted_grid_matches_official_endpoints_and_is_decreasing() -> None:
    grid = shifted_time_grid(100, 8.0, torch.device("cpu"))
    assert float(grid[0]) == 1.0
    assert float(grid[-1]) == 0.0
    assert torch.all(grid[:-1] > grid[1:])


def test_shifted_grid_step_override_changes_only_grid_resolution() -> None:
    official = shifted_time_grid(100, 8.0, torch.device("cpu"))
    equal_compute = shifted_time_grid(124, 8.0, torch.device("cpu"))
    assert len(official) == 101
    assert len(equal_compute) == 125
    assert torch.equal(official[[0, -1]], equal_compute[[0, -1]])


def test_condition_rejects_revision_without_horizon() -> None:
    with pytest.raises(ValueError, match="positive horizon"):
        SamplingCondition("bad", 1.78, 0.0, 1.0).validate()


def test_shared_retiming_condition_requests_a_full_future_pair() -> None:
    condition = parse_condition("shared,1.78,0.03125,0.05,shared_retiming")
    assert condition.uses_retiming
    assert condition.uses_shared_retiming


def test_ou_polar_retiming_is_a_retiming_condition() -> None:
    condition = parse_condition("ou,1.78,0.03125,0.05,ou_polar_retiming")
    assert condition.uses_retiming
    assert condition.uses_ou_certificate
    assert not condition.uses_shared_retiming


def test_weak_ou_polar_uses_a_weak_certificate() -> None:
    condition = parse_condition(
        "ou_weak,1.78,0.03125,0.05,ou_weak_polar_retiming"
    )
    assert condition.uses_retiming
    assert condition.uses_ou_certificate
    assert condition.uses_weak_ou_certificate
    assert condition.uses_polar_ou_certificate
    assert not condition.uses_shared_retiming


def test_strong_ou_polar_does_not_report_a_weak_certificate() -> None:
    condition = parse_condition("ou,1.78,0.03125,0.05,ou_polar_retiming")
    assert condition.uses_polar_ou_certificate
    assert not condition.uses_weak_ou_certificate


def test_first_half_retiming_is_the_sit_window_in_rae_time() -> None:
    condition = parse_condition(
        "ou_first,1.78,0.03125,1,ou_polar_first_half_retiming"
    )
    assert condition.uses_retiming
    assert condition.uses_ou_certificate
    assert condition.uses_polar_ou_certificate
    assert condition.uses_first_half_revision
    assert condition.revision_is_active(0.75)
    assert not condition.revision_is_active(0.5)
    assert not condition.revision_is_active(0.25)
    assert condition.revision_future_floor(0.1) == 0.5
    assert dataward_future_time(
        0.51,
        condition.horizon,
        coordinate="raw_time",
        minimum_time=condition.revision_future_floor(0.1),
    ) == 0.5


@pytest.mark.parametrize(
    ("method", "uses_ou"),
    [
        ("pathwise_first_half_retiming", False),
        ("ou_polar_pathwise_first_half_retiming", True),
    ],
)
def test_pathwise_retiming_uses_a_bounded_dataward_ig_query(
    method: str, uses_ou: bool
) -> None:
    condition = parse_condition(f"pathwise,1.78,0.03125,1,{method}")
    assert condition.uses_pathwise_retiming
    assert condition.uses_first_half_revision
    assert condition.uses_ou_certificate is uses_ou
    assert condition.revision_is_active(0.75)
    assert not condition.revision_is_active(0.5)
    assert condition.revision_future_floor(0.1) == 0.5
