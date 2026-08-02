from __future__ import annotations

import numpy as np

from experiments.analyze_raev2_decoder_pushforward_mechanism import (
    feature_bias_metrics,
    low_frequency_features,
    paired_increment_metrics,
)


def test_feature_bias_metrics_detect_exact_cancellation() -> None:
    rng = np.random.default_rng(11)
    source = rng.normal(size=(64, 5)).astype(np.float32)
    reconstruction = source + np.asarray([1.0, -0.5, 0.2, 0.0, 0.3], dtype=np.float32)
    candidate = source.copy()
    metrics = feature_bias_metrics(source, reconstruction, candidate)
    assert metrics["raw_ig_vs_reconstruction_bias_cosine"] < -0.999
    assert metrics["raw_ig_mean_error_ratio"] < 1e-12
    assert metrics["diag_white_ig_mean_error_ratio"] < 1e-12


def test_feature_bias_metrics_isolates_increment_from_full_control() -> None:
    rng = np.random.default_rng(13)
    source = rng.normal(size=(64, 4)).astype(np.float32)
    reconstruction = source + 0.4
    control = source + np.asarray([0.8, 0.0, 0.0, 0.0], dtype=np.float32)
    candidate = source.copy()
    metrics = feature_bias_metrics(
        source, reconstruction, candidate, control=control
    )
    assert metrics["raw_ig_vs_full_error_cosine"] < -0.999
    assert metrics["raw_ig_mean_error_ratio"] < 1e-12


def test_paired_increment_metrics_detect_contraction() -> None:
    rng = np.random.default_rng(17)
    control = rng.normal(size=(12, 3, 2, 2)).astype(np.float32)
    increment = rng.normal(size=control.shape).astype(np.float32)
    candidate = control + increment
    roundtrip_control = 0.7 * control
    roundtrip_candidate = roundtrip_control + 0.25 * increment
    weight = increment.mean(axis=0).reshape(-1)
    weight /= np.linalg.norm(weight)
    metrics = paired_increment_metrics(
        control,
        candidate,
        roundtrip_control,
        roundtrip_candidate,
        probe_weight=weight,
    )
    assert abs(metrics["roundtrip_over_raw_norm_mean"] - 0.25) < 1e-6
    assert metrics["raw_roundtrip_cosine_mean"] > 0.999
    assert metrics["roundtrip_probe_delta_sum"] > 0


def test_low_frequency_features_preserve_constant_block_statistics() -> None:
    images = np.zeros((2, 8, 8, 3), dtype=np.float32)
    images[0] = 0.25
    images[1] = 0.75
    features = low_frequency_features(images, grid_size=2)
    assert features.shape == (2, 2 * 2 * 3 * 2)
    np.testing.assert_allclose(features[0, :12], 0.25)
    np.testing.assert_allclose(features[1, :12], 0.75)
    np.testing.assert_allclose(features[:, 12:], 0.0)
