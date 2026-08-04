import numpy as np
import torch
from scipy import linalg

from experiments.compute_sit_ig_interval_fid import (
    frechet_from_activations,
    symmetric_psd_sqrt,
)


def test_torch_fid_matches_standard_sqrtm_formula():
    rng = np.random.default_rng(7)
    activations = rng.normal(size=(9, 5))
    reference = rng.normal(size=(20, 5))
    reference_mean = reference.mean(axis=0)
    reference_covariance = np.cov(reference, rowvar=False)
    sample_mean = activations.mean(axis=0)
    sample_covariance = np.cov(activations, rowvar=False)
    covariance_mean = linalg.sqrtm(sample_covariance @ reference_covariance).real
    expected = (
        np.square(sample_mean - reference_mean).sum()
        + np.trace(sample_covariance)
        + np.trace(reference_covariance)
        - 2 * np.trace(covariance_mean)
    )
    reference_sqrt = symmetric_psd_sqrt(
        torch.as_tensor(reference_covariance, dtype=torch.float64)
    )
    actual = frechet_from_activations(
        activations,
        reference_mean,
        reference_covariance,
        reference_sqrt=reference_sqrt,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)


def test_low_rank_torch_fid_matches_standard_sqrtm_formula():
    rng = np.random.default_rng(11)
    activations = rng.normal(size=(4, 9))
    reference = rng.normal(size=(20, 9))
    reference_mean = reference.mean(axis=0)
    reference_covariance = np.cov(reference, rowvar=False)
    sample_mean = activations.mean(axis=0)
    sample_covariance = np.cov(activations, rowvar=False)
    covariance_mean = linalg.sqrtm(sample_covariance @ reference_covariance).real
    expected = (
        np.square(sample_mean - reference_mean).sum()
        + np.trace(sample_covariance)
        + np.trace(reference_covariance)
        - 2 * np.trace(covariance_mean)
    )
    actual = frechet_from_activations(
        activations,
        reference_mean,
        reference_covariance,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-7)
