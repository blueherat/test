"""Numerical helpers for auditing AdvFD moment-frame consistency."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class PopulationMoments:
    mean: np.ndarray
    covariance: np.ndarray
    count: int


@dataclass(frozen=True)
class RegularizedWhitener:
    anchor_mean: torch.Tensor
    transform: torch.Tensor
    anchor_eigenvalues: torch.Tensor
    epsilon: float


def population_moments_from_sums(
    feature_sum: np.ndarray,
    feature_outer_sum: np.ndarray,
    count: int,
) -> PopulationMoments:
    """Build population moments, matching AdvFD's ``E[ff^T] - E[f]E[f]^T``."""

    count = int(count)
    if count <= 0:
        raise ValueError("count must be positive")
    feature_sum = np.asarray(feature_sum, dtype=np.float64)
    feature_outer_sum = np.asarray(feature_outer_sum, dtype=np.float64)
    if feature_outer_sum.shape != (feature_sum.size, feature_sum.size):
        raise ValueError("outer-sum shape does not match feature dimension")
    mean = feature_sum / count
    covariance = feature_outer_sum / count - np.outer(mean, mean)
    covariance = 0.5 * (covariance + covariance.T)
    return PopulationMoments(mean=mean, covariance=covariance, count=count)


def merge_population_moments(
    first: PopulationMoments,
    second: PopulationMoments,
) -> PopulationMoments:
    """Merge two disjoint population-moment estimates without raw features."""

    if first.mean.shape != second.mean.shape:
        raise ValueError("moment dimensions differ")
    total = first.count + second.count
    mean = (first.count * first.mean + second.count * second.mean) / total
    first_m2 = first.covariance + np.outer(first.mean, first.mean)
    second_m2 = second.covariance + np.outer(second.mean, second.mean)
    m2 = (first.count * first_m2 + second.count * second_m2) / total
    covariance = m2 - np.outer(mean, mean)
    covariance = 0.5 * (covariance + covariance.T)
    return PopulationMoments(mean=mean, covariance=covariance, count=total)


def interpolate_population_moments(
    historical: PopulationMoments,
    current: PopulationMoments,
    *,
    historical_weight: float,
) -> PopulationMoments:
    """Interpolate uncentered moments, as repeated EMA updates would in expectation."""

    if historical.mean.shape != current.mean.shape:
        raise ValueError("moment dimensions differ")
    if not 0.0 <= historical_weight <= 1.0:
        raise ValueError("historical_weight must be in [0, 1]")
    current_weight = 1.0 - historical_weight
    mean = historical_weight * historical.mean + current_weight * current.mean
    historical_second = historical.covariance + np.outer(
        historical.mean, historical.mean
    )
    current_second = current.covariance + np.outer(current.mean, current.mean)
    second = historical_weight * historical_second + current_weight * current_second
    covariance = second - np.outer(mean, mean)
    covariance = 0.5 * (covariance + covariance.T)
    return PopulationMoments(mean=mean, covariance=covariance, count=-1)


def torch_population_moments(
    moments: PopulationMoments,
    *,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(moments.mean).double().to(device),
        torch.from_numpy(moments.covariance).double().to(device),
    )


def blend_anchor_with_batch(
    anchor_mean: torch.Tensor,
    anchor_covariance: torch.Tensor,
    batch_features: torch.Tensor,
    *,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match ``FeatureStatsEMA.build_stats`` for an arbitrary moment anchor."""

    if not 0.0 <= beta < 1.0:
        raise ValueError("beta must be in [0, 1)")
    features = batch_features.double()
    batch_mean = features.mean(dim=0)
    batch_second = features.T @ features / features.shape[0]
    anchor_second = anchor_covariance + anchor_mean[:, None] * anchor_mean[None, :]
    mean = beta * anchor_mean + (1.0 - beta) * batch_mean
    second = beta * anchor_second + (1.0 - beta) * batch_second
    covariance = second - mean[:, None] * mean[None, :]
    covariance = 0.5 * (covariance + covariance.T)
    return mean, covariance


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    probabilities = torch.tensor(
        [0.0, 0.01, 0.10, 0.50, 0.90, 0.99, 1.0],
        dtype=values.dtype,
        device=values.device,
    )
    quantiles = torch.quantile(values, probabilities).cpu().tolist()
    return {
        name: float(value)
        for name, value in zip(
            ("min", "q01", "q10", "q50", "q90", "q99", "max"),
            quantiles,
        )
    }


def build_regularized_whitener(
    anchor: PopulationMoments,
    *,
    epsilon: float,
    device: str | torch.device = "cpu",
) -> RegularizedWhitener:
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    anchor_mean = torch.from_numpy(anchor.mean).double().to(device)
    anchor_covariance = torch.from_numpy(anchor.covariance).double().to(device)
    dimension = int(anchor_mean.numel())
    identity = torch.eye(dimension, dtype=torch.float64, device=device)
    anchor_regularized = 0.5 * (anchor_covariance + anchor_covariance.T)
    anchor_regularized = anchor_regularized + epsilon * identity
    eigenvalues, eigenvectors = torch.linalg.eigh(anchor_regularized)
    inverse_roots = eigenvalues.clamp_min(epsilon).rsqrt()
    return RegularizedWhitener(
        anchor_mean=anchor_mean,
        transform=eigenvectors * inverse_roots.unsqueeze(0),
        anchor_eigenvalues=eigenvalues,
        epsilon=float(epsilon),
    )


def real_whitened_fd_components_from_stats(
    real_mean: torch.Tensor,
    real_covariance: torch.Tensor,
    fake_mean: torch.Tensor,
    fake_covariance: torch.Tensor,
    *,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the exact mean/covariance terms used by official AdvFD.

    AdvFD treats ``covariance + epsilon * I`` as the Gaussian covariance on
    both sides, whitens by the real side, and detaches that whitening frame.
    The returned eigenvalues are therefore the generalized regularized
    covariance spectrum, and ``(sqrt(lambda) - 1)^2`` sums exactly to the
    covariance term of this whitened objective.
    """

    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    real_mean = real_mean.double()
    real_covariance = real_covariance.double()
    fake_mean = fake_mean.double()
    fake_covariance = fake_covariance.double()
    if real_mean.shape != fake_mean.shape:
        raise ValueError("real and fake mean dimensions differ")
    dimension = int(real_mean.numel())
    expected_shape = (dimension, dimension)
    if real_covariance.shape != expected_shape or fake_covariance.shape != expected_shape:
        raise ValueError("covariance shape does not match mean dimension")

    identity = torch.eye(
        dimension,
        dtype=torch.float64,
        device=real_mean.device,
    )
    real_regularized = 0.5 * (real_covariance + real_covariance.T)
    real_regularized = real_regularized + float(epsilon) * identity
    real_eigenvalues, real_eigenvectors = torch.linalg.eigh(real_regularized)
    inverse_roots = real_eigenvalues.clamp_min(float(epsilon)).rsqrt()
    real_mean_detached = real_mean.detach()
    real_eigenvectors = real_eigenvectors.detach()
    inverse_roots = inverse_roots.detach()

    mean_white = ((fake_mean - real_mean_detached) @ real_eigenvectors) * inverse_roots
    fake_regularized = 0.5 * (fake_covariance + fake_covariance.T)
    fake_regularized = fake_regularized + float(epsilon) * identity
    fake_in_real_basis = real_eigenvectors.T @ fake_regularized @ real_eigenvectors
    fake_white = (
        fake_in_real_basis
        * inverse_roots[:, None]
        * inverse_roots[None, :]
    )
    fake_white = 0.5 * (fake_white + fake_white.T)
    generalized_eigenvalues = torch.linalg.eigvalsh(fake_white).clamp_min(0.0)

    mean_term = mean_white.square().sum()
    covariance_term = (generalized_eigenvalues.sqrt() - 1.0).square().sum()
    return mean_term.float(), covariance_term.float(), generalized_eigenvalues


def regularized_whitening_consistency(
    anchor: PopulationMoments,
    probe: PopulationMoments,
    *,
    epsilon: float,
    device: str | torch.device = "cpu",
    whitener: RegularizedWhitener | None = None,
) -> dict[str, float | int | dict[str, float]]:
    """Measure whether ``probe`` is zero-mean/unit-covariance in ``anchor`` frame.

    AdvFD adds ``epsilon * I`` to both the real covariance used to build the
    whitener and the covariance being evaluated.  Keeping that loading on both
    sides is required for the self-comparison to equal exactly zero/identity.
    """

    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if anchor.mean.shape != probe.mean.shape:
        raise ValueError("moment dimensions differ")

    if whitener is None:
        whitener = build_regularized_whitener(
            anchor, epsilon=epsilon, device=device
        )
    if whitener.epsilon != float(epsilon):
        raise ValueError("whitener epsilon differs from requested epsilon")
    anchor_mean = whitener.anchor_mean
    transform = whitener.transform
    eigenvalues = whitener.anchor_eigenvalues
    device = anchor_mean.device
    probe_mean = torch.from_numpy(probe.mean).double().to(device)
    probe_covariance = torch.from_numpy(probe.covariance).double().to(device)
    dimension = int(anchor_mean.numel())
    identity = torch.eye(dimension, dtype=torch.float64, device=device)

    mean_white = (probe_mean - anchor_mean) @ transform
    probe_regularized = 0.5 * (probe_covariance + probe_covariance.T)
    probe_regularized = probe_regularized + epsilon * identity
    covariance_white = transform.T @ probe_regularized @ transform
    covariance_white = 0.5 * (covariance_white + covariance_white.T)
    covariance_eigenvalues = torch.linalg.eigvalsh(covariance_white).clamp_min(0.0)

    mean_term = mean_white.square().sum()
    covariance_term = (
        torch.diagonal(covariance_white).sum()
        + dimension
        - 2.0 * covariance_eigenvalues.sqrt().sum()
    ).clamp_min(0.0)
    identity_residual = covariance_white - identity
    positive_eigenvalues = covariance_eigenvalues.clamp_min(torch.finfo(torch.float64).tiny)
    log_eigenvalues = positive_eigenvalues.log()

    return {
        "anchor_count": int(anchor.count),
        "probe_count": int(probe.count),
        "feature_dim": dimension,
        "mean_mahalanobis_sq": float(mean_term),
        "mean_mahalanobis_rms_per_dim": float((mean_term / dimension).sqrt()),
        "covariance_identity_frobenius": float(torch.linalg.matrix_norm(identity_residual)),
        "covariance_identity_frobenius_per_sqrt_dim": float(
            torch.linalg.matrix_norm(identity_residual) / math.sqrt(dimension)
        ),
        "covariance_log_eigen_rms": float(log_eigenvalues.square().mean().sqrt()),
        "covariance_trace_per_dim": float(torch.trace(covariance_white) / dimension),
        "covariance_bures_to_identity": float(covariance_term),
        "covariance_bures_to_identity_per_dim": float(covariance_term / dimension),
        "regularized_whitened_fd": float(mean_term + covariance_term),
        "regularized_whitened_fd_per_dim": float((mean_term + covariance_term) / dimension),
        "covariance_eigenvalues": _quantiles(covariance_eigenvalues),
        "anchor_regularized_eigenvalues": _quantiles(eigenvalues),
    }
