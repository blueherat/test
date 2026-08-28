from __future__ import annotations

import inspect

import numpy as np

from experiments import observe_dit_blur_focused_eprocess as observer
from experiments import calibrate_dit_blur_focused_eprocess as calibrate
from experiments import replay_dit_blur_focused_eprocess_inputs as replay


def test_synthetic_contract() -> None:
    observer.self_test()


def test_predictable_constructor_cannot_receive_innovation() -> None:
    parameters = inspect.signature(observer.construct_predictable_shift).parameters
    assert "transition_innovation" not in parameters
    assert "innovation" not in parameters


def test_gate_off_is_identity_alternative() -> None:
    theta = np.ones((3, 4, 8, 8), dtype=np.float64)
    sigma = np.full(theta.shape, 0.25, dtype=np.float32)
    mask = np.ones((3, 1, 8, 8), dtype=np.float64)
    gate = np.zeros(3, dtype=np.bool_)
    shift, raw_K, applied_K = observer.construct_predictable_shift(
        theta, sigma, mask, gate, K_allowance=0.02
    )
    assert np.array_equal(shift, np.zeros_like(shift))
    assert np.array_equal(raw_K, np.zeros_like(raw_K))
    assert np.array_equal(applied_K, np.zeros_like(applied_K))


def test_fixed_mixture_not_posthoc_max() -> None:
    rng = np.random.default_rng(91)
    z = rng.normal(size=(150_000, 2))
    component = np.exp(0.3 * z - 0.5 * 0.3**2)
    assert abs(float(np.mean(np.mean(component, axis=1))) - 1.0) < 0.008
    assert float(np.mean(np.max(component, axis=1))) > 1.08


def test_matched_q_power_gate_rejects_old_budget_and_accepts_frozen_budget() -> None:
    weak = observer.matched_q_power_reference(total_K=0.5, draws=100_000, seed=4)
    frozen = observer.matched_q_power_reference(draws=100_000, seed=4)
    assert weak["minimum_anytime_power"] < 0.03
    assert frozen["minimum_anytime_power"] >= observer.MATCHED_Q_ANYTIME_POWER_MINIMUM


def test_replay_adapter_cpu_contract() -> None:
    replay.self_test()


def test_label_free_B_order_statistic_calibration() -> None:
    source = {
        "experiment": calibrate.SOURCE_EXPERIMENT,
        "manifest_identity_sha256": "1" * 64,
        "manifest_file_sha256": "2" * 64,
        "time_series_file_sha256": "3" * 64,
        "unused_archive_members_not_loaded": ["resnet18_target_log_odds"],
    }
    arrays = calibrate._synthetic_arrays()
    payload = calibrate.derive_calibration(arrays, source)
    calibrate.validate_calibration(payload)
    assert payload["state_gate_order_statistic"] == "17th ascending; strict greater"
    assert payload["pure_B_order_statistic"] == "19th ascending; strict greater"
    assert payload["calibration_seed_count_per_class"] == 20


def test_calibrator_rejects_nonwhitelisted_array() -> None:
    source = {
        "experiment": calibrate.SOURCE_EXPERIMENT,
        "manifest_identity_sha256": "1" * 64,
        "manifest_file_sha256": "2" * 64,
        "time_series_file_sha256": "3" * 64,
        "unused_archive_members_not_loaded": [],
    }
    arrays = calibrate._synthetic_arrays()
    arrays["human_label"] = np.zeros(len(arrays["sample_index"]), dtype=np.int8)
    try:
        calibrate.derive_calibration(arrays, source)
    except RuntimeError:
        pass
    else:
        raise AssertionError("calibrator accepted a supervised/non-whitelisted array")
