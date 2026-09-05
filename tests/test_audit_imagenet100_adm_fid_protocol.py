from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.audit_imagenet100_adm_fid_protocol import (
    frechet_distance,
    mean_covariance,
    stratified_split_indices,
    unbiased_kid,
    uniform_class_weights,
)


def test_weighted_statistics_equal_explicit_balanced_duplication() -> None:
    features = np.asarray([[0.0], [2.0], [10.0], [12.0], [14.0]])
    labels = np.asarray([0, 0, 1, 1, 1])
    weights = uniform_class_weights(labels, 2)
    mean, _ = mean_covariance(features, weights)

    assert weights.sum() == pytest.approx(1.0)
    assert mean.item() == pytest.approx((1.0 + 12.0) / 2.0)


def test_frechet_distance_is_zero_for_identical_statistics() -> None:
    features = np.asarray([[0.0, 1.0], [2.0, 3.0], [1.0, -1.0]])
    mean, covariance = mean_covariance(features)
    assert frechet_distance(mean, covariance, mean, covariance) == pytest.approx(
        0.0, abs=1e-7
    )


def test_unbiased_kid_matches_direct_kernel_formula() -> None:
    first = np.asarray([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]], dtype=np.float32)
    second = np.asarray([[0.5, 1.5], [1.5, 0.5], [2.5, 1.5]], dtype=np.float32)

    def kernel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (x @ y.T / x.shape[1] + 1.0) ** 3

    xx = kernel(first, first)
    yy = kernel(second, second)
    xy = kernel(first, second)
    expected = (
        (xx.sum() - np.trace(xx)) / (len(first) * (len(first) - 1))
        + (yy.sum() - np.trace(yy)) / (len(second) * (len(second) - 1))
        - 2.0 * xy.mean()
    )
    actual = unbiased_kid(first, second, device=torch.device("cpu"), block_size=2)
    assert actual == pytest.approx(float(expected), abs=1e-6)


def test_stratified_split_is_balanced_and_disjoint() -> None:
    labels = np.repeat(np.arange(3), 4)
    first, second = stratified_split_indices(labels, 3)

    assert set(first).isdisjoint(set(second))
    assert np.array_equal(np.bincount(labels[first]), np.asarray([2, 2, 2]))
    assert np.array_equal(np.bincount(labels[second]), np.asarray([2, 2, 2]))
