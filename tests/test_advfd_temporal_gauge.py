from __future__ import annotations

import numpy as np
import pytest

from experiments.advfd_cleanroom.temporal_gauge import (
    PopulationMoments,
    blend_anchor_with_batch,
    interpolate_population_moments,
    merge_population_moments,
    population_moments_from_sums,
    real_whitened_fd_components_from_stats,
    regularized_whitening_consistency,
)
import torch

from experiments.advfd_cleanroom.audit_advfd_rotating_gaussian import run_condition
from experiments.advfd_cleanroom.audit_advfd_temporal_gauge_gradients import (
    gradient_concentration_metrics,
)


def test_population_moments_match_direct_computation() -> None:
    values = np.asarray([[1.0, 2.0], [3.0, 1.0], [-1.0, 4.0]])
    moments = population_moments_from_sums(
        values.sum(axis=0), values.T @ values, len(values)
    )
    np.testing.assert_allclose(moments.mean, values.mean(axis=0))
    np.testing.assert_allclose(moments.covariance, np.cov(values, rowvar=False, ddof=0))


def test_merge_population_moments_matches_full_bank() -> None:
    rng = np.random.default_rng(7)
    values = rng.normal(size=(19, 5))
    first_values, second_values = values[:8], values[8:]
    first = population_moments_from_sums(
        first_values.sum(0), first_values.T @ first_values, len(first_values)
    )
    second = population_moments_from_sums(
        second_values.sum(0), second_values.T @ second_values, len(second_values)
    )
    merged = merge_population_moments(first, second)
    direct = population_moments_from_sums(
        values.sum(0), values.T @ values, len(values)
    )
    np.testing.assert_allclose(merged.mean, direct.mean, atol=1e-12)
    np.testing.assert_allclose(merged.covariance, direct.covariance, atol=1e-12)


def test_interpolate_population_moments_matches_repeated_expected_ema() -> None:
    historical = PopulationMoments(
        mean=np.asarray([1.0, -2.0]),
        covariance=np.asarray([[2.0, 0.3], [0.3, 1.0]]),
        count=-1,
    )
    current = PopulationMoments(
        mean=np.asarray([-0.5, 0.25]),
        covariance=np.asarray([[0.7, -0.1], [-0.1, 1.8]]),
        count=100,
    )
    beta = 0.9
    steps = 7
    expected_mean = historical.mean.copy()
    expected_covariance = historical.covariance.copy()
    current_values = torch.tensor(
        [[-1.0, -0.75], [0.0, 1.25]], dtype=torch.float64
    )
    assert np.allclose(current_values.mean(0).numpy(), current.mean)
    expected_current_covariance = (
        current_values.T @ current_values / current_values.shape[0]
        - current_values.mean(0)[:, None] * current_values.mean(0)[None, :]
    ).numpy()
    current = PopulationMoments(
        mean=current.mean,
        covariance=expected_current_covariance,
        count=2,
    )
    for _ in range(steps):
        mean_tensor, covariance_tensor = blend_anchor_with_batch(
            torch.from_numpy(expected_mean),
            torch.from_numpy(expected_covariance),
            current_values,
            beta=beta,
        )
        expected_mean = mean_tensor.numpy()
        expected_covariance = covariance_tensor.numpy()
    interpolated = interpolate_population_moments(
        historical,
        current,
        historical_weight=beta**steps,
    )
    np.testing.assert_allclose(interpolated.mean, expected_mean, atol=1e-12)
    np.testing.assert_allclose(
        interpolated.covariance, expected_covariance, atol=1e-12
    )


def test_regularized_self_whitening_is_exact_identity() -> None:
    mean = np.asarray([2.0, -1.0, 0.5])
    covariance = np.asarray(
        [[3.0, 0.4, 0.0], [0.4, 1.5, 0.2], [0.0, 0.2, 0.5]]
    )
    moments = PopulationMoments(mean=mean, covariance=covariance, count=100)
    metrics = regularized_whitening_consistency(moments, moments, epsilon=1e-3)
    assert metrics["mean_mahalanobis_sq"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["covariance_trace_per_dim"] == pytest.approx(1.0, abs=1e-10)
    assert metrics["covariance_identity_frobenius"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["regularized_whitened_fd"] == pytest.approx(0.0, abs=1e-9)


def test_omitting_probe_regularization_would_create_false_mismatch() -> None:
    moments = PopulationMoments(
        mean=np.zeros(2), covariance=np.diag([0.0, 2.0]), count=10
    )
    metrics = regularized_whitening_consistency(moments, moments, epsilon=0.1)
    assert metrics["covariance_eigenvalues"]["min"] == pytest.approx(1.0, abs=1e-10)
    assert metrics["covariance_eigenvalues"]["max"] == pytest.approx(1.0, abs=1e-10)


def test_blend_anchor_with_batch_matches_direct_second_moment_formula() -> None:
    anchor_values = torch.tensor([[1.0, 2.0], [3.0, -1.0]], dtype=torch.float64)
    batch_values = torch.tensor([[2.0, 0.0], [4.0, 2.0]], dtype=torch.float64)
    anchor_mean = anchor_values.mean(0)
    anchor_covariance = (
        anchor_values.T @ anchor_values / len(anchor_values)
        - anchor_mean[:, None] * anchor_mean[None, :]
    )
    mean, covariance = blend_anchor_with_batch(
        anchor_mean, anchor_covariance, batch_values, beta=0.75
    )
    expected_mean = 0.75 * anchor_values.mean(0) + 0.25 * batch_values.mean(0)
    expected_second = 0.75 * (
        anchor_values.T @ anchor_values / len(anchor_values)
    ) + 0.25 * (batch_values.T @ batch_values / len(batch_values))
    torch.testing.assert_close(mean, expected_mean)
    torch.testing.assert_close(
        covariance,
        expected_second - expected_mean[:, None] * expected_mean[None, :],
    )


def test_fd_mean_and_covariance_gradients_sum_to_total_gradient() -> None:
    real_values = torch.tensor(
        [[-1.0, 0.2, 0.5], [0.4, 1.2, -0.3], [1.1, -0.7, 0.8]],
        dtype=torch.float64,
    )
    fake_values = torch.tensor(
        [[-0.5, 0.7, 0.1], [0.9, 0.4, -0.8], [1.4, -0.1, 1.0]],
        dtype=torch.float64,
        requires_grad=True,
    )

    def moments(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = values.mean(0)
        covariance = values.T @ values / values.shape[0] - mean[:, None] * mean[None, :]
        return mean, covariance

    real_mean, real_covariance = moments(real_values)
    fake_mean, fake_covariance = moments(fake_values)
    mean_term, covariance_term, _ = real_whitened_fd_components_from_stats(
        real_mean,
        real_covariance,
        fake_mean,
        fake_covariance,
        epsilon=1e-3,
    )
    denominator = (mean_term + covariance_term).detach() + 0.01
    mean_gradient = torch.autograd.grad(
        mean_term / denominator,
        fake_values,
        retain_graph=True,
    )[0]
    covariance_gradient = torch.autograd.grad(
        covariance_term / denominator,
        fake_values,
        retain_graph=True,
    )[0]
    total_gradient = torch.autograd.grad(
        (mean_term + covariance_term) / denominator,
        fake_values,
    )[0]
    torch.testing.assert_close(
        total_gradient,
        mean_gradient + covariance_gradient,
        rtol=1e-10,
        atol=1e-10,
    )


def test_gradient_concentration_distinguishes_uniform_and_sparse_energy() -> None:
    uniform = torch.ones(4, 3, 2, 2)
    sparse = torch.zeros_like(uniform)
    sparse[0, 0, 0, 0] = 1.0
    uniform_metrics = gradient_concentration_metrics(uniform)
    sparse_metrics = gradient_concentration_metrics(sparse)
    assert uniform_metrics["sample_effective_fraction"] == pytest.approx(1.0)
    assert uniform_metrics["coordinate_effective_fraction"] == pytest.approx(1.0)
    assert sparse_metrics["sample_effective_fraction"] == pytest.approx(0.25)
    assert sparse_metrics["coordinate_effective_fraction"] == pytest.approx(
        1.0 / sparse.numel()
    )


def test_rotating_frame_breaks_direct_ema_but_not_transported_ema() -> None:
    static, _ = run_condition(
        beta=0.99,
        degrees_per_step=0.0,
        steps=500,
        anisotropy=0.8,
        mean_offset=1.0,
        epsilon=1e-3,
    )
    rotating, _ = run_condition(
        beta=0.99,
        degrees_per_step=1.0,
        steps=500,
        anisotropy=0.8,
        mean_offset=1.0,
        epsilon=1e-3,
    )

    assert static["naive_final_to_current"] == pytest.approx(1.0, abs=1e-12)
    assert rotating["naive_final_to_current"] < 0.3
    assert rotating["transported_final_to_current"] == pytest.approx(
        1.0, abs=1e-12
    )
    assert rotating["transported_max_relative_error"] < 1e-12
