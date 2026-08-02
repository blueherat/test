from __future__ import annotations

import numpy as np

from experiments.analyze_raev2_scale_path_geometry import (
    chord_metrics,
    low_frequency_features,
)


def test_chord_metrics_is_exact_on_a_straight_path() -> None:
    rng = np.random.default_rng(3)
    control = rng.normal(size=(5, 3, 4)).astype(np.float32)
    anchor = rng.normal(size=(5, 3, 4)).astype(np.float32)
    interpolation = 0.5
    actual = control + interpolation * (anchor - control)

    result = chord_metrics(control, anchor, actual, interpolation)

    assert result["sample_count"] == 5
    assert float(result["residual_sq_sum"]) < 1e-10
    np.testing.assert_allclose(result["cosine"], 1.0, atol=1e-6)
    np.testing.assert_allclose(result["radial_gain"], 1.0, atol=1e-6)


def test_chord_metrics_detects_an_orthogonal_detour() -> None:
    control = np.zeros((2, 2), dtype=np.float32)
    anchor = np.asarray([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    actual = np.asarray([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)

    result = chord_metrics(control, anchor, actual, 0.5)

    np.testing.assert_allclose(result["relative_to_chord"], 1.0)
    np.testing.assert_allclose(result["cosine"], 1.0 / np.sqrt(2.0))


def test_low_frequency_features_has_expected_shape() -> None:
    images = np.zeros((3, 8, 8, 3), dtype=np.float32)
    images[:, :4] = 1.0

    features = low_frequency_features(images, grid_size=2)

    assert features.shape == (3, 2 * 2 * 3 * 2)
    assert np.isfinite(features).all()
