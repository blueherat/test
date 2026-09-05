from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.raev2_pfr_retiming import (
    bridge_latentized_counterfactual_state,
    clean_to_velocity,
    dataward_future_time,
    euler_dataward_query_state,
    exponential_retiming_defect,
    ordinary_internal_guidance_clean,
    norm_preserving_certificate_revision,
    orthogonal_counterfactual_guidance_clean,
    pfr_velocity,
    project_revision_onto_certificate,
    raev2_ou_degree1_velocity_defect,
    retime_future_score_to_current,
    shared_retiming_revision,
    strong_anchored_counterfactual_guidance_clean,
    velocity_to_score,
)
from experiments.pfr_ou_semigroup_spectrum import (
    linear_velocity_to_ou_relative_score,
    ou_bridge_coordinates,
    ou_relative_score_delta_to_linear_velocity_delta,
    transport_state_at_fixed_ou_coordinate,
)


def test_log_odds_horizon_has_constant_information_distance() -> None:
    current = 0.5
    horizon = 0.2
    future = dataward_future_time(
        current, horizon, coordinate="log_odds", minimum_time=0.0
    )
    current_odds = (1.0 - current) / current
    future_odds = (1.0 - future) / future
    assert future < current
    assert future_odds / current_odds == pytest.approx(math.exp(horizon))
    assert dataward_future_time(
        1.0, horizon, coordinate="log_odds", minimum_time=0.0
    ) == pytest.approx(1.0)


def test_raw_time_horizon_preserves_existing_rule() -> None:
    assert dataward_future_time(
        0.6, 0.03125, coordinate="raw_time", minimum_time=0.1
    ) == pytest.approx(0.56875)
    assert dataward_future_time(
        0.11, 0.03125, coordinate="raw_time", minimum_time=0.1
    ) == pytest.approx(0.1)


def test_euler_dataward_query_uses_the_signed_rae_time_increment() -> None:
    state = torch.randn(3, 4, 2, 2)
    velocity = torch.randn_like(state)
    current = torch.tensor([0.9, 0.6, 0.51])
    future = torch.tensor([0.8, 0.56875, 0.5])
    actual = euler_dataward_query_state(state, velocity, current, future)
    delta = (future - current)[:, None, None, None]
    torch.testing.assert_close(actual, state + delta * velocity)


def test_raev2_clean_to_velocity_recovers_linear_bridge_velocity() -> None:
    clean = torch.randn(4, 3, 2, 2)
    noise = torch.randn_like(clean)
    time = torch.tensor([0.1, 0.25, 0.5, 0.9])
    expanded = time[:, None, None, None]
    state = (1.0 - expanded) * clean + expanded * noise

    actual = clean_to_velocity(
        clean, state, time, denominator_floor=0.05
    )
    torch.testing.assert_close(actual, noise - clean)


def test_raev2_ou_defect_matches_direct_relative_score_algebra() -> None:
    state = torch.randn(4, 3, 2, 2, dtype=torch.float64)
    current_velocity = torch.randn_like(state)
    future_velocity = torch.randn_like(state)
    time = torch.tensor([0.95, 0.8, 0.6, 0.3], dtype=torch.float64)
    future_time = time - 0.05
    data_time = 1.0 - time
    future_data_time = 1.0 - future_time
    future_state = transport_state_at_fixed_ou_coordinate(
        state, data_time, future_data_time
    )
    current_score = linear_velocity_to_ou_relative_score(
        -current_velocity, state, data_time
    )
    future_score = linear_velocity_to_ou_relative_score(
        -future_velocity, future_state, future_data_time
    )
    current_signal = ou_bridge_coordinates(data_time, state).signal
    future_signal = ou_bridge_coordinates(future_data_time, state).signal
    score_defect = current_score - current_signal / future_signal * future_score
    expected = -ou_relative_score_delta_to_linear_velocity_delta(
        score_defect, state, data_time
    )

    actual = raev2_ou_degree1_velocity_defect(
        current_velocity,
        future_velocity,
        state,
        time,
        future_time,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)


def test_norm_preserving_certificate_is_nearest_feasible_orientation() -> None:
    raw = torch.tensor([[3.0, 4.0], [-2.0, 1.0]])
    certificate = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    common = project_revision_onto_certificate(raw, certificate)
    polar = norm_preserving_certificate_revision(raw, certificate)

    torch.testing.assert_close(polar.norm(dim=1), raw.norm(dim=1))
    torch.testing.assert_close(
        torch.nn.functional.cosine_similarity(polar, common),
        torch.ones(2),
    )
    opposite = -polar
    assert torch.all((polar - raw).norm(dim=1) < (opposite - raw).norm(dim=1))


def test_bridge_latentization_keeps_the_weak_implied_noise_exactly_fixed() -> None:
    weak_clean = torch.randn(4, 3, 2, 2)
    strong_clean = torch.randn_like(weak_clean)
    weak_noise = torch.randn_like(weak_clean)
    time = torch.tensor([0.1, 0.25, 0.5, 0.9])
    expanded = time[:, None, None, None]
    state = (1.0 - expanded) * weak_clean + expanded * weak_noise
    scale = 1.78

    counterfactual = bridge_latentized_counterfactual_state(
        state,
        strong_clean,
        weak_clean,
        time,
        guidance_scale=scale,
    )
    guided_clean = ordinary_internal_guidance_clean(
        strong_clean,
        weak_clean,
        guidance_scale=scale,
    )
    implied_noise = (
        counterfactual - (1.0 - expanded) * guided_clean
    ) / expanded
    torch.testing.assert_close(implied_noise, weak_noise, rtol=1e-5, atol=1e-5)


def test_counterfactual_guidance_preserves_the_strong_anchor() -> None:
    strong = torch.randn(4, 8)
    counterfactual_weak = torch.randn_like(strong)
    actual = strong_anchored_counterfactual_guidance_clean(
        strong,
        counterfactual_weak,
        guidance_scale=1.0,
    )
    torch.testing.assert_close(actual, strong, rtol=0.0, atol=0.0)


def test_counterfactual_guidance_recovers_ordinary_ig_for_original_weak() -> None:
    strong = torch.randn(4, 8)
    weak = torch.randn_like(strong)
    scale = 1.78
    actual = strong_anchored_counterfactual_guidance_clean(
        strong,
        weak,
        guidance_scale=scale,
    )
    expected = ordinary_internal_guidance_clean(
        strong,
        weak,
        guidance_scale=scale,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_orthogonal_counterfactual_guidance_changes_only_direction() -> None:
    weak = torch.zeros(2, 4)
    strong = torch.tensor([[2.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0]])
    revision = torch.tensor([[4.0, 1.0, 0.0, 0.0], [1.0, -6.0, 0.0, 0.0]])
    counterfactual_weak = weak - revision
    scale = 1.78
    actual = orthogonal_counterfactual_guidance_clean(
        strong,
        weak,
        counterfactual_weak,
        guidance_scale=scale,
    )
    extra = actual - strong
    ordinary_extra = (scale - 1.0) * (strong - weak)
    expected_directions = torch.tensor(
        [[2.0, 1.0, 0.0, 0.0], [1.0, 3.0, 0.0, 0.0]]
    )
    cosine = torch.nn.functional.cosine_similarity(extra, expected_directions)
    torch.testing.assert_close(cosine, torch.ones_like(cosine))
    torch.testing.assert_close(extra.norm(dim=1), ordinary_extra.norm(dim=1))


def test_orthogonal_counterfactual_guidance_has_exact_strong_anchor() -> None:
    strong = torch.randn(3, 8)
    weak = torch.randn_like(strong)
    counterfactual_weak = torch.randn_like(strong)
    actual = orthogonal_counterfactual_guidance_clean(
        strong,
        weak,
        counterfactual_weak,
        guidance_scale=1.0,
    )
    torch.testing.assert_close(actual, strong, rtol=0.0, atol=0.0)


def test_reusing_future_velocity_equals_exponential_score_retiming() -> None:
    state = torch.randn(4, 3, 2, 2)
    future_velocity = torch.randn_like(state)
    current_time = torch.tensor([0.8, 0.6, 0.4, 0.2])
    future_time = torch.tensor([0.7, 0.5, 0.3, 0.1])

    future_score = velocity_to_score(
        future_velocity, state, future_time
    )
    retimed = retime_future_score_to_current(
        future_score, state, current_time, future_time
    )
    reused = velocity_to_score(future_velocity, state, current_time)
    torch.testing.assert_close(retimed, reused, rtol=1e-5, atol=1e-6)


def test_pfr_adds_the_exponential_retiming_defect_in_score_space() -> None:
    state = torch.randn(4, 3, 2, 2)
    strong = torch.randn_like(state)
    weak = torch.randn_like(state)
    weak_future = torch.randn_like(state)
    time = torch.tensor([0.8, 0.6, 0.4, 0.2])
    scale = 1.78

    ordinary_velocity = weak + scale * (strong - weak)
    guided_velocity = pfr_velocity(
        strong,
        weak,
        weak_future,
        guidance_scale=scale,
    )
    score_change = velocity_to_score(
        guided_velocity, state, time
    ) - velocity_to_score(ordinary_velocity, state, time)
    expected = scale * exponential_retiming_defect(
        weak, weak_future, time
    )
    torch.testing.assert_close(score_change, expected, rtol=1e-5, atol=1e-6)


def test_zero_revision_is_exactly_ordinary_internal_guidance() -> None:
    strong = torch.randn(2, 8)
    weak = torch.randn_like(strong)
    future = torch.randn_like(strong)
    scale = 1.78

    actual = pfr_velocity(
        strong,
        weak,
        future,
        guidance_scale=scale,
        revision_scale=0.0,
    )
    expected = weak + scale * (strong - weak)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@torch.no_grad()
def test_strong_anchored_additive_replaces_the_weak_reference() -> None:
    strong = torch.randn(4, 3, 5, 5)
    weak = torch.randn_like(strong)
    future = torch.randn_like(strong)
    scale = 1.78
    revision_scale = 0.4

    actual = pfr_velocity(
        strong,
        weak,
        future,
        guidance_scale=scale,
        revision_scale=revision_scale,
        composition="strong_anchored_additive",
    )
    counterfactual_weak = weak - revision_scale * (weak - future)
    expected = strong + (scale - 1.0) * (strong - counterfactual_weak)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


@torch.no_grad()
def test_strong_anchored_additive_has_exact_null_guidance_limit() -> None:
    strong = torch.randn(4, 3, 5, 5)
    weak = torch.randn_like(strong)
    future = torch.randn_like(strong)

    actual = pfr_velocity(
        strong,
        weak,
        future,
        guidance_scale=1.0,
        revision_scale=1.0,
        composition="strong_anchored_additive",
    )
    torch.testing.assert_close(actual, strong, rtol=0.0, atol=0.0)


@torch.no_grad()
def test_norm_preserving_revision_rotates_without_rescaling_guidance() -> None:
    strong = torch.randn(4, 3, 5, 5)
    weak = torch.randn_like(strong)
    future = torch.randn_like(strong)
    scale = 1.78
    actual = pfr_velocity(
        strong,
        weak,
        future,
        guidance_scale=scale,
        revision_scale=0.7,
        composition="norm_preserving",
    )
    baseline_gap = scale * (strong - weak)
    revised_gap = actual - weak
    torch.testing.assert_close(
        revised_gap.float().flatten(1).norm(dim=1),
        baseline_gap.float().flatten(1).norm(dim=1),
        rtol=1e-5,
        atol=1e-5,
    )


@torch.no_grad()
def test_strong_anchored_norm_preserving_rotates_only_extrapolation() -> None:
    strong = torch.randn(4, 3, 5, 5)
    weak = torch.randn_like(strong)
    future = torch.randn_like(strong)
    scale = 1.78
    actual = pfr_velocity(
        strong,
        weak,
        future,
        guidance_scale=scale,
        revision_scale=1.0,
        composition="strong_anchored_norm_preserving",
    )
    original_extrapolation = (scale - 1.0) * (strong - weak)
    revised_extrapolation = actual - strong
    torch.testing.assert_close(
        revised_extrapolation.float().flatten(1).norm(dim=1),
        original_extrapolation.float().flatten(1).norm(dim=1),
        rtol=1e-5,
        atol=1e-5,
    )


@torch.no_grad()
def test_strong_anchored_zero_revision_is_exactly_ordinary_guidance() -> None:
    strong = torch.randn(2, 8)
    weak = torch.randn_like(strong)
    future = torch.randn_like(strong)
    scale = 1.78
    actual = pfr_velocity(
        strong,
        weak,
        future,
        guidance_scale=scale,
        revision_scale=0.0,
        composition="strong_anchored_norm_preserving",
    )
    expected = weak + scale * (strong - weak)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@torch.no_grad()
def test_strong_anchored_angular_discards_parallel_revision() -> None:
    weak = torch.zeros(2, 4)
    strong = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]])
    revision = torch.tensor([[3.0, 1.0, 0.0, 0.0], [1.0, -4.0, 0.0, 0.0]])
    future = weak - revision
    scale = 1.78
    actual = pfr_velocity(
        strong,
        weak,
        future,
        guidance_scale=scale,
        revision_scale=1.0,
        composition="strong_anchored_angular",
    )
    expected_directions = torch.tensor(
        [[1.0, 1.0, 0.0, 0.0], [1.0, 2.0, 0.0, 0.0]]
    )
    revised_extrapolation = actual - strong
    cosine = torch.nn.functional.cosine_similarity(
        revised_extrapolation, expected_directions
    )
    torch.testing.assert_close(cosine, torch.ones_like(cosine))
    torch.testing.assert_close(
        revised_extrapolation.norm(dim=1),
        (scale - 1.0) * (strong - weak).norm(dim=1),
    )


@torch.no_grad()
def test_orthogonal_composition_removes_parallel_revision_before_rotation() -> None:
    weak = torch.zeros(2, 4)
    strong = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]])
    revision = torch.tensor([[3.0, 1.0, 0.0, 0.0], [1.0, -4.0, 0.0, 0.0]])
    future = weak - revision
    actual = pfr_velocity(
        strong,
        weak,
        future,
        guidance_scale=1.0,
        revision_scale=1.0,
        composition="orthogonal_norm_preserving",
    )
    expected_directions = torch.tensor(
        [[1.0, 1.0, 0.0, 0.0], [1.0, 2.0, 0.0, 0.0]]
    )
    cosine = torch.nn.functional.cosine_similarity(actual, expected_directions)
    torch.testing.assert_close(cosine, torch.ones_like(cosine))
    torch.testing.assert_close(
        actual.norm(dim=1), (strong - weak).norm(dim=1)
    )


def test_shared_retiming_revision_projects_weak_change_onto_strong_change() -> None:
    weak = torch.tensor([[3.0, 4.0], [4.0, 2.0]])
    weak_future = torch.tensor([[1.0, 3.0], [2.0, 1.0]])
    strong = torch.tensor([[2.0, 2.0], [3.0, 5.0]])
    strong_future = torch.tensor([[1.0, 2.0], [3.0, 3.0]])

    actual = shared_retiming_revision(
        weak, weak_future, strong, strong_future
    )
    expected = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
    torch.testing.assert_close(actual, expected)
    residual = (weak - weak_future) - actual
    strong_revision = strong - strong_future
    torch.testing.assert_close(
        (residual * strong_revision).sum(dim=1), torch.zeros(2)
    )


def test_shared_retiming_revision_is_zero_for_zero_strong_change() -> None:
    weak = torch.randn(3, 8)
    weak_future = torch.randn_like(weak)
    strong = torch.randn_like(weak)
    actual = shared_retiming_revision(
        weak, weak_future, strong, strong
    )
    torch.testing.assert_close(actual, torch.zeros_like(actual))
