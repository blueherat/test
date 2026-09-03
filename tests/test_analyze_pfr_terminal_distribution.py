from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.analyze_pfr_terminal_distribution import (
    class_scatter,
    covariance_scatter,
    feature_scatter,
)


def test_feature_and_covariance_scatter_agree() -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(16, 5))
    covariance = np.cov(features, rowvar=False)
    feature_trace, feature_rank = feature_scatter(features)
    covariance_trace, covariance_rank = covariance_scatter(covariance)
    np.testing.assert_allclose(feature_trace, covariance_trace, rtol=1e-12)
    np.testing.assert_allclose(feature_rank, covariance_rank, rtol=1e-12)


def test_class_scatter_satisfies_expected_extremes() -> None:
    features = np.asarray(
        [[0.0, 0.0], [0.0, 0.0], [2.0, 0.0], [2.0, 0.0]],
        dtype=np.float64,
    )
    labels = np.asarray([0, 0, 1, 1])
    result = class_scatter(features, labels)
    assert result["within_class_trace"] == 0.0
    assert result["between_total_scatter_fraction"] == 1.0
    assert result["requested_class_count"] == 2.0
