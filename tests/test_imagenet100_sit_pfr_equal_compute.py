from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.run_imagenet100_sit_pfr_equal_compute import (
    DEPTH10_PREFIX_FULL_RATIO,
    PREFIX_FULL_RATIO,
    Condition,
    balanced_labels,
    parse_condition,
    segment_step_counts,
)


def test_equal_compute_pair_has_nearly_identical_analytical_budget() -> None:
    ordinary = Condition("ordinary_ig", 32)
    pfr = Condition("projected", 27)

    assert ordinary.full_calls_per_batch == 64
    assert ordinary.query_calls_per_batch == 0
    assert pfr.full_calls_per_batch == 54
    assert pfr.query_calls_per_batch == 27
    assert ordinary.full_forward_equivalents(PREFIX_FULL_RATIO) == 64.0
    assert pfr.full_forward_equivalents(PREFIX_FULL_RATIO) == pytest.approx(64.557)
    assert (
        pfr.full_forward_equivalents(PREFIX_FULL_RATIO)
        / ordinary.full_forward_equivalents(PREFIX_FULL_RATIO)
        - 1.0
    ) == pytest.approx(0.008703125)


def test_pfr_heun26_is_strictly_under_the_ordinary_budget() -> None:
    ordinary = Condition("ordinary_ig", 32)
    pfr = Condition("projected", 26)

    assert segment_step_counts(26) == (6, 6, 14)
    assert pfr.query_calls_per_batch == 23
    assert pfr.full_forward_equivalents(PREFIX_FULL_RATIO) < (
        ordinary.full_forward_equivalents(PREFIX_FULL_RATIO)
    )


def test_stage_reused_pfr_heun29_matches_ordinary_heun32_budget() -> None:
    ordinary = Condition("ordinary_ig", 32)
    stage_reused = Condition("projected_stage_reuse", 29)

    assert segment_step_counts(29) == (7, 7, 15)
    assert stage_reused.full_calls_per_batch == 58
    assert stage_reused.query_calls_per_batch == 14
    assert stage_reused.full_forward_equivalents(PREFIX_FULL_RATIO) == pytest.approx(
        63.474
    )
    assert stage_reused.full_forward_equivalents(PREFIX_FULL_RATIO) < (
        ordinary.full_forward_equivalents(PREFIX_FULL_RATIO)
    )


def test_eulerian_decomposition_condition_accounting() -> None:
    time_only = Condition("time_only", 32)
    material = Condition("material_guided", 32)
    frame = Condition("frame_guided", 32)

    assert time_only.query_calls_per_batch == 31
    assert material.query_calls_per_batch == 62
    assert frame.query_calls_per_batch == 62


def test_strong_weak_retiming_controls_count_future_full_evaluations() -> None:
    condition = Condition("weak_common_strong", 32)

    assert condition.full_calls_per_batch == 64
    assert condition.query_calls_per_batch == 0
    assert condition.full_query_calls_per_batch == 31
    assert condition.full_forward_equivalents(PREFIX_FULL_RATIO) == 95.0

    multidepth = Condition("weak_common_depth10", 32)
    assert multidepth.full_calls_per_batch == 64
    assert multidepth.query_calls_per_batch == 31
    assert multidepth.full_query_calls_per_batch == 0
    assert multidepth.query_full_ratio(PREFIX_FULL_RATIO) == pytest.approx(
        DEPTH10_PREFIX_FULL_RATIO
    )
    assert multidepth.full_forward_equivalents(PREFIX_FULL_RATIO) == pytest.approx(
        89.327
    )


def test_ou_spectral_controls_count_two_prefix_queries_per_active_evaluation() -> None:
    condition = Condition("ou_d1_common", 32)

    assert condition.full_calls_per_batch == 64
    assert condition.query_calls_per_batch == 62
    assert condition.full_query_calls_per_batch == 0
    assert condition.full_forward_equivalents(PREFIX_FULL_RATIO) == pytest.approx(
        88.242
    )

    first_stage_only = Condition("ou_d1_common_first", 32)
    assert first_stage_only.query_calls_per_batch == 46
    assert first_stage_only.full_forward_equivalents(
        PREFIX_FULL_RATIO
    ) == pytest.approx(81.986)
    for kind in (
        "ou_d1_common_norm_raw_direction_first",
        "ou_d1_common_direction_raw_norm_first",
    ):
        control = Condition(kind, 32)
        assert control.query_calls_per_batch == 46
        assert control.full_forward_equivalents(
            PREFIX_FULL_RATIO
        ) == pytest.approx(81.986)

    then_projected = Condition("ou_d1_common_then_projected", 32)
    assert then_projected.query_calls_per_batch == 46
    plus_spatial = Condition("ou_d1_common_plus_spatial", 32)
    assert plus_spatial.query_calls_per_batch == 61
    assert plus_spatial.full_forward_equivalents(
        PREFIX_FULL_RATIO
    ) == pytest.approx(87.851)
    assert Condition("ou_d2_common_first", 32).query_calls_per_batch == 46
    assert Condition("ou_d2_unique_first", 32).query_calls_per_batch == 46
    adaptive = Condition("ou_d1_energy_adaptive", 23)
    assert adaptive.query_calls_per_batch == 46
    assert adaptive.full_forward_equivalents(PREFIX_FULL_RATIO) == pytest.approx(
        63.986
    )
    two_scale = Condition("ou_d1_two_scale_span_first", 23)
    assert two_scale.name == "pfr_ou_d1_two_scale_span_first_heun_n23"
    assert two_scale.query_calls_per_batch == 45
    assert two_scale.full_forward_equivalents(PREFIX_FULL_RATIO) == pytest.approx(
        63.595
    )
    strong_conditions = (
        Condition("ou_d1_strong_common_first", 22),
        Condition("ou_d1_strong_unique_first", 22),
        Condition("ou_d1_strong_common_norm_raw_direction_first", 22),
        Condition("ou_d1_strong_common_direction_raw_norm_first", 22),
        Condition("ou_d1_strong_anchored_common_direction_raw_norm_first", 22),
        Condition("ou_d1_strong_anchored_angular_first", 22),
        Condition("ou_d2_strong_common_first", 22),
        Condition("ou_d2_strong_common_direction_raw_norm_first", 22),
    )
    for condition in strong_conditions:
        assert condition.query_calls_per_batch == 23
        assert condition.full_query_calls_per_batch == 11
        assert condition.full_forward_equivalents(
            PREFIX_FULL_RATIO
        ) == pytest.approx(63.993)


def test_condition_parser_and_payload_are_canonical() -> None:
    condition = parse_condition("projected:27")
    assert condition == Condition("projected", 27)
    assert condition.payload(PREFIX_FULL_RATIO)["name"] == "pfr_heun_n27"
    with pytest.raises(Exception):
        parse_condition("projected")


def test_scaled_time_condition_round_trips_without_changing_defaults() -> None:
    default = Condition("time_only", 32)
    scaled = parse_condition("time_only:32:0.015625:2")

    assert default.name == "pfr_time_only_heun_n32"
    assert "anchor_horizon" not in default.payload(PREFIX_FULL_RATIO)
    assert scaled == Condition(
        "time_only", 32, anchor_horizon=1.0 / 64.0, revision_scale=2.0
    )
    assert scaled.cli_spec == "time_only:32:0.015625:2.0"
    assert scaled.name == "pfr_time_only_h0p015625_r2_heun_n32"
    assert scaled.payload(PREFIX_FULL_RATIO)["anchor_horizon"] == 1.0 / 64.0

    with pytest.raises(ValueError, match="only for time_only"):
        Condition("projected", 32, revision_scale=2.0).validate()


def test_balanced_labels_are_exact_and_deterministic() -> None:
    first = balanced_labels(
        num_samples=1000,
        num_classes=100,
        seed=17,
        device=torch.device("cpu"),
    )
    second = balanced_labels(
        num_samples=1000,
        num_classes=100,
        seed=17,
        device=torch.device("cpu"),
    )
    histogram = torch.bincount(first, minlength=100)

    assert torch.equal(first, second)
    assert torch.equal(histogram, torch.full((100,), 10, dtype=torch.int64))
    with pytest.raises(ValueError, match="divisible"):
        balanced_labels(
            num_samples=999,
            num_classes=100,
            seed=17,
            device=torch.device("cpu"),
        )
