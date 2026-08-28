from __future__ import annotations

import inspect
import math

import numpy as np

from experiments import calibrate_dit_blur_focused_eprocess_v2 as calibrate
from experiments import observe_dit_blur_focused_eprocess_v2 as observer
from experiments import replay_dit_blur_focused_eprocess_inputs_v2 as replay


def _track_fixture(batch: int = 4) -> dict[str, np.ndarray]:
    effective = np.asarray(observer.EFFECTIVE_NONIDENTITY, dtype=np.uint8)
    theta = np.ones(
        (batch, len(observer.HEAT_SHIFTS), len(observer.CHECKPOINTS), 4, 8, 8),
        dtype=np.float64,
    )
    theta[:, effective == 0] = 0.0
    return {
        "theta": theta,
        "p_standard_deviation": np.ones(
            (batch, len(observer.CHECKPOINTS), 4, 8, 8), dtype=np.float32
        ),
        "transition_innovation": np.zeros(
            (batch, len(observer.CHECKPOINTS), 4, 8, 8), dtype=np.float32
        ),
        "local_mask": np.ones(
            (batch, len(observer.CHECKPOINTS), 1, 8, 8), dtype=np.float64
        ),
        "blur_severity": np.ones((batch, len(observer.CHECKPOINTS)), dtype=np.float64),
        "blur_gate_threshold": np.zeros(
            (batch, len(observer.CHECKPOINTS)), dtype=np.float64
        ),
        "effective_nonidentity": effective,
    }


def test_v2_selftest() -> None:
    observer.self_test()


def test_predictable_constructor_has_no_innovation_argument() -> None:
    names = inspect.signature(observer.construct_predictable_shift).parameters
    assert "innovation" not in names
    assert "transition_innovation" not in names


def test_fixed_information_direction_and_floor() -> None:
    theta = np.ones((2, 4, 8, 8), dtype=np.float64)
    theta[1] *= 1e-12
    sigma = np.ones_like(theta, dtype=np.float32)
    mask = np.ones((2, 1, 8, 8), dtype=np.float64)
    gate = np.ones(2, dtype=np.bool_)
    shift, _, applied, valid = observer.construct_predictable_shift(
        theta, sigma, mask, gate, K_allowance=0.4
    )
    assert valid.tolist() == [True, False]
    assert math.isclose(applied[0], 0.4, rel_tol=2e-12, abs_tol=1e-14)
    assert applied[1] == 0.0
    assert np.array_equal(shift[1], np.zeros_like(shift[1]))


def test_late_start_freezes_h_three_and_kappa_two_thirds() -> None:
    fixture = _track_fixture(batch=1)
    fixture["blur_severity"][:] = -1.0
    fixture["blur_severity"][:, 6:] = 1.0
    tracks = observer.compute_eprocess_tracks(**fixture, use_state_gate=True)
    assert np.array_equal(
        tracks.start_remaining_effective_count, np.asarray([[3, 3]], dtype=np.int16)
    )
    assert np.allclose(tracks.frozen_K_per_step_after_start, 2.0 / 3.0)
    assert np.array_equal(np.sum(tracks.applied_K > 0.0, axis=2), [[3, 3]])
    assert np.allclose(np.sum(tracks.applied_K, axis=2), 2.0)


def test_missing_direction_reuses_last_unit_but_fails_stale_direction_gate() -> None:
    fixture = _track_fixture(batch=observer.PATH_MECHANICS_MINIMUM_SAMPLES)
    effective = fixture["effective_nonidentity"].astype(bool)
    missing_index = int(np.flatnonzero(effective[0])[1])
    fixture["theta"][:, 0, missing_index] = 0.0
    tracks = observer.compute_eprocess_tracks(**fixture, use_state_gate=True)
    expected_kappa = observer.FIXED_K_PER_EFFECTIVE_CHECKPOINT[0]
    next_index = int(np.flatnonzero(effective[0])[2])
    assert np.allclose(tracks.applied_K[:, 0, missing_index], expected_kappa)
    assert np.all(tracks.direction_reused[:, 0, missing_index])
    assert np.allclose(tracks.applied_K[:, 0, next_index], expected_kappa)
    classes = np.arange(observer.PATH_MECHANICS_MINIMUM_SAMPLES, dtype=np.int16) % 6
    audit = observer.label_free_path_mechanics_audit(
        applied_K=tracks.applied_K,
        start_time_index=tracks.start_time_index,
        start_remaining_effective_count=tracks.start_remaining_effective_count,
        direction_reused=tracks.direction_reused,
        class_id=classes,
        effective_nonidentity=fixture["effective_nonidentity"],
    )
    assert audit["passes"] is False


def test_innovation_poison_changes_only_likelihood_observation() -> None:
    fixture = _track_fixture(batch=5)
    rng = np.random.default_rng(77)
    first = dict(fixture)
    second = dict(fixture)
    first["transition_innovation"] = rng.normal(
        size=fixture["transition_innovation"].shape
    ).astype(np.float32)
    second["transition_innovation"] = rng.normal(
        size=fixture["transition_innovation"].shape
    ).astype(np.float32)
    left = observer.compute_eprocess_tracks(**first, use_state_gate=True)
    right = observer.compute_eprocess_tracks(**second, use_state_gate=True)
    for name in (
        "raw_K",
        "applied_K",
        "start_time_index",
        "start_remaining_effective_count",
        "frozen_K_per_step_after_start",
        "direction_reused",
    ):
        assert np.array_equal(getattr(left, name), getattr(right, name))
    assert not np.array_equal(left.component_increment, right.component_increment)
    assert np.array_equal(
        observer.gate_only_start_schedule_score(left),
        observer.gate_only_start_schedule_score(right),
    )


def test_gate_only_start_schedule_formula_uses_five_and_eight_denominators() -> None:
    fixture = _track_fixture(batch=1)
    tracks = observer.compute_eprocess_tracks(**fixture, use_state_gate=True)
    score = observer.gate_only_start_schedule_score(tracks)
    direct = observer.gate_only_start_schedule_score_from_metadata(
        tracks.start_time_index, tracks.start_remaining_effective_count
    )
    assert np.array_equal(tracks.start_remaining_effective_count, [[5, 8]])
    assert np.array_equal(score, [1.0])
    assert np.array_equal(score, direct)


def test_one_shot_is_distinct_equal_total_budget_ablation() -> None:
    fixture = _track_fixture(batch=3)
    tracks = observer.compute_eprocess_tracks(
        **fixture, use_state_gate=True, one_shot_full_budget=True
    )
    assert np.array_equal(np.sum(tracks.applied_K > 0.0, axis=2), np.ones((3, 2)))
    assert np.allclose(np.sum(tracks.applied_K, axis=2), 2.0)


def test_fixed_directional_e_and_mixture_have_null_mean_one() -> None:
    rng = np.random.default_rng(2026082819)
    draws = 300_000
    component_terminal = []
    for h in (3, 8):
        kappa = observer.TOTAL_K_PER_SCALE / h
        shift = math.sqrt(2.0 * kappa)
        z = rng.normal(size=(draws, h))
        log_e = np.sum(shift * z - kappa, axis=1)
        component_terminal.append(np.exp(log_e))
    component = np.stack(component_terminal, axis=1)
    mixture = component @ np.asarray(observer.MIXTURE_WEIGHTS)
    assert np.all(np.abs(np.mean(component, axis=0) - 1.0) < 0.04)
    assert abs(float(np.mean(mixture)) - 1.0) < 0.03


def test_all_h_conditional_power_lower_bound() -> None:
    result = observer.matched_q_power_reference(draws=100_000, seed=17)
    expected = 0.5 * math.erfc(
        (math.log(20.0) - 2.0) / (2.0 * math.sqrt(2.0))
    )
    assert math.isclose(
        result["dependence_robust_conditional_terminal_power_lower_bound"],
        expected,
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    assert len(result["component_results"]) == sum(
        count - observer.MINIMUM_REMAINING_EFFECTIVE_AT_START + 1
        for count in observer.EFFECTIVE_STEP_COUNT_PER_SCALE
    ) == 9
    assert result["passes"] is True


def test_adaptive_predictable_null_calibration_and_ville_witness() -> None:
    result = observer.adaptive_predictable_null_reference(draws=120_000, seed=11)
    assert result["passes"] is True


def test_calibration_and_replay_contracts_are_unchanged() -> None:
    calibrate.self_test()
    replay.self_test()
