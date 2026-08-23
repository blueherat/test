import numpy as np
import pytest
import torch

from experiments.advfd_cleanroom.audit_official_advfd_features import (
    equally_weighted_mixture_moments,
    scalar_distribution_summary,
    whiten_moment_pair,
)
from experiments.advfd_cleanroom.diagnose_official_advfd_checkpoints import (
    real_whitened_fd_from_stats,
)


def test_regularized_real_whitening_maps_anchor_covariance_to_identity():
    real_mu = np.array([2.0, -1.0])
    real_covariance = np.array([[4.0, 1.0], [1.0, 2.0]])
    fake_mu = np.array([3.0, 4.0])
    fake_covariance = np.array([[3.0, 0.2], [0.2, 1.0]])

    real, _ = whiten_moment_pair(
        real_mu,
        real_covariance,
        fake_mu,
        fake_covariance,
        anchor_mu=real_mu,
        anchor_covariance=real_covariance,
        epsilon=1e-3,
    )

    assert torch.allclose(real[0], torch.zeros(2, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(real[1], torch.eye(2, dtype=torch.float64), atol=1e-10)


def test_pooled_moments_include_between_distribution_mean_variance():
    covariance = np.zeros((1, 1))
    mean, mixture_covariance = equally_weighted_mixture_moments(
        np.array([-2.0]), covariance, np.array([2.0]), covariance
    )

    np.testing.assert_allclose(mean, np.array([0.0]))
    np.testing.assert_allclose(mixture_covariance, np.array([[4.0]]))


def test_real_whitened_fd_is_zero_for_matching_regularized_gaussians():
    mean = torch.tensor([1.0, -2.0], dtype=torch.float64)
    covariance = torch.tensor([[2.0, 0.3], [0.3, 0.5]], dtype=torch.float64)

    total, mean_term, covariance_term = real_whitened_fd_from_stats(
        mean,
        covariance,
        mean,
        covariance,
        epsilon=1e-3,
    )

    assert abs(total) < 1e-10
    assert abs(mean_term) < 1e-12
    assert abs(covariance_term) < 1e-10


def test_scalar_distribution_summary_reports_requested_quantiles():
    summary = scalar_distribution_summary(np.arange(1.0, 101.0))

    assert summary["count"] == 100
    assert summary["min"] == 1.0
    assert summary["q50"] == pytest.approx(50.5)
    assert summary["q90"] == pytest.approx(90.1)
    assert summary["q99"] == pytest.approx(99.01)
    assert summary["max"] == 100.0
