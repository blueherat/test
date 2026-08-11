from __future__ import annotations

import numpy as np

from experiments.evaluate_guidance_toy_benchmark import (
    classifier_two_sample_auc,
    energy_distance_2d,
    manifold_metrics,
    sliced_wasserstein_2d,
)


def test_sliced_wasserstein_is_zero_for_identical_samples() -> None:
    rng = np.random.default_rng(7)
    values = rng.normal(size=(128, 2))
    assert sliced_wasserstein_2d(values, values) == 0.0


def test_sliced_wasserstein_detects_a_shift() -> None:
    rng = np.random.default_rng(11)
    values = rng.normal(size=(256, 2))
    shifted = values + np.array([0.8, -0.4])
    assert sliced_wasserstein_2d(values, shifted) > 0.3


def test_energy_distance_is_zero_for_identical_empirical_measure() -> None:
    rng = np.random.default_rng(12)
    values = rng.normal(size=(128, 2))
    assert energy_distance_2d(values, values, limit=128, seed=3) == 0.0


def test_local_c2st_detects_a_large_scale_change() -> None:
    rng = np.random.default_rng(15)
    values = rng.normal(size=(1200, 2))
    sharp = 0.35 * values
    auc = classifier_two_sample_auc(values, sharp, limit=1000, seed=21)
    assert auc > 0.75


def test_manifold_metrics_are_perfect_for_identical_samples() -> None:
    rng = np.random.default_rng(13)
    values = rng.normal(size=(128, 2))
    metrics = manifold_metrics(
        values,
        values,
        k=5,
        limit=128,
        seed=17,
        chunk_size=32,
    )
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["coverage"] == 1.0
    assert metrics["density"] >= 1.0
