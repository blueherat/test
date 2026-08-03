from __future__ import annotations

import numpy as np

from experiments.run_raev2_scale_response_study import (
    diagonal_lda_metrics,
    normalized_scales,
    rbf_mmd_squared,
    scale_key,
    sketch_distance_metrics,
)


def test_scale_key_and_normalization_are_stable() -> None:
    assert scale_key(1.78) == "scale_s1p780000"
    assert normalized_scales([1.78, 1.0, 1.78]) == (1.0, 1.78)
    assert normalized_scales([1.78], require_unguided=False) == (1.78,)


def test_latent_metrics_are_null_for_identical_arrays() -> None:
    rng = np.random.default_rng(11)
    reference = rng.normal(size=(20, 3, 2, 2)).astype(np.float32)
    test_mask = np.zeros(20, dtype=bool)
    test_mask[-6:] = True
    metrics = diagonal_lda_metrics(
        reference, reference.copy(), test_mask, ridge_ratio=1e-4
    )
    assert metrics["auc"] == 0.5
    assert metrics["mean_shift_rms"] == 0.0
    assert metrics["diagonal_variance_relative_l2"] == 0.0


def test_latent_metrics_detect_heldout_global_shift() -> None:
    rng = np.random.default_rng(17)
    reference = rng.normal(size=(60, 8)).astype(np.float32)
    candidate = reference + 1.5
    test_mask = np.zeros(60, dtype=bool)
    test_mask[-20:] = True
    metrics = diagonal_lda_metrics(
        reference, candidate, test_mask, ridge_ratio=1e-4
    )
    assert metrics["auc_separability"] > 0.95
    assert metrics["mean_shift_rms"] > 1.0


def test_sketch_metrics_are_null_for_identical_arrays() -> None:
    rng = np.random.default_rng(23)
    reference = rng.normal(size=(24, 6)).astype(np.float32)
    metrics, bandwidth = sketch_distance_metrics(reference, reference.copy())
    assert bandwidth > 0
    assert metrics["sketch_sliced_wasserstein"] == 0.0
    assert metrics["sketch_rbf_mmd_squared"] < 1e-6
    assert metrics["sketch_covariance_relative_frobenius"] == 0.0


def test_rbf_mmd_rejects_a_large_shift() -> None:
    rng = np.random.default_rng(29)
    reference = rng.normal(size=(40, 5)).astype(np.float32)
    shifted = reference + 3.0
    assert rbf_mmd_squared(reference, shifted, bandwidth_sq=5.0) > 0.1
