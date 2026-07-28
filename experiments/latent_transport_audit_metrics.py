"""Metrics and controlled maps for latent transport compatibility audits.

The module intentionally separates source-prior mismatch from path-curvature
mismatch.  A non-orthogonal linear map is the key control: it changes the
Gaussian source geometry while commuting exactly with straight chords.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityLatentTransform(nn.Module):
    is_linear = True

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value

    def inverse(self, value: torch.Tensor) -> torch.Tensor:
        return value


class SignedChannelOrthogonalTransform(nn.Module):
    """A deterministic signed channel permutation."""

    is_linear = True

    def __init__(self, channels: int, seed: int):
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        permutation = torch.randperm(int(channels), generator=generator)
        signs = torch.randint(0, 2, (int(channels),), generator=generator)
        signs = signs.to(torch.float32).mul_(2.0).sub_(1.0)
        inverse_permutation = torch.empty_like(permutation)
        inverse_permutation[permutation] = torch.arange(int(channels))
        self.register_buffer("permutation", permutation)
        self.register_buffer("inverse_permutation", inverse_permutation)
        self.register_buffer("signs", signs)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        signs = self.signs.to(device=value.device, dtype=value.dtype)
        return value[:, self.permutation] * signs.view(1, -1, 1, 1)

    def inverse(self, value: torch.Tensor) -> torch.Tensor:
        signs = self.signs.to(device=value.device, dtype=value.dtype)
        signed = value * signs.view(1, -1, 1, 1)
        return signed[:, self.inverse_permutation]


class AnisotropicChannelTransform(nn.Module):
    """Volume-preserving diagonal map with a prescribed condition number."""

    is_linear = True

    def __init__(self, channels: int, condition_number: float, seed: int):
        super().__init__()
        condition_number = float(condition_number)
        if condition_number < 1.0:
            raise ValueError("condition_number must be at least one")
        log_half = 0.5 * math.log(condition_number)
        log_scales = torch.linspace(-log_half, log_half, int(channels))
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        permutation = torch.randperm(int(channels), generator=generator)
        scales = torch.exp(log_scales[permutation])
        self.condition_number = condition_number
        self.register_buffer("scales", scales)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        scales = self.scales.to(device=value.device, dtype=value.dtype)
        return value * scales.view(1, -1, 1, 1)

    def inverse(self, value: torch.Tensor) -> torch.Tensor:
        scales = self.scales.to(device=value.device, dtype=value.dtype)
        return value / scales.view(1, -1, 1, 1)


class LatentSketch(nn.Module):
    """Fixed linear JL-style channel projection followed by spatial pooling."""

    def __init__(
        self,
        channels: int,
        projected_channels: int = 16,
        spatial_size: int = 4,
        seed: int = 0,
    ):
        super().__init__()
        channels = int(channels)
        projected_channels = int(projected_channels)
        spatial_size = int(spatial_size)
        if not 0 < projected_channels <= channels:
            raise ValueError("projected_channels must lie in [1, channels]")
        if spatial_size <= 0:
            raise ValueError("spatial_size must be positive")
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        matrix = torch.randn(
            (channels, projected_channels),
            generator=generator,
            dtype=torch.float64,
        )
        matrix = torch.linalg.qr(matrix, mode="reduced").Q.to(torch.float32)
        self.spatial_size = spatial_size
        self.register_buffer("matrix", matrix)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4:
            raise ValueError(f"expected BCHW latent, got shape {tuple(value.shape)}")
        if value.shape[1] != self.matrix.shape[0]:
            raise ValueError("latent channels do not match sketch matrix")
        matrix = self.matrix.to(device=value.device, dtype=value.dtype)
        projected = torch.einsum("bchw,ck->bkhw", value, matrix)
        pooled = F.adaptive_avg_pool2d(
            projected,
            (self.spatial_size, self.spatial_size),
        )
        # For divisible grids this makes an iid N(0,1) latent approximately
        # standard normal after each spatial average.
        area = (value.shape[-2] / self.spatial_size) * (
            value.shape[-1] / self.spatial_size
        )
        return pooled.flatten(1) * math.sqrt(float(area))


def relative_l2_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have equal shape")
    numerator = (prediction - target).flatten(1).square().sum(dim=1)
    denominator = target.flatten(1).square().sum(dim=1).clamp_min(float(eps))
    return torch.sqrt(numerator / denominator)


def covariance_mismatch(
    sample: torch.Tensor,
    reference: torch.Tensor,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Projected first/second-moment mismatch without Gaussian assumptions."""

    if sample.shape != reference.shape or sample.ndim != 2:
        raise ValueError("sample and reference must be equal [N, D] matrices")
    x = sample.float()
    y = reference.float()
    mean_gap = torch.linalg.vector_norm(x.mean(0) - y.mean(0))
    mean_scale = torch.linalg.vector_norm(y.mean(0)).clamp_min(float(eps))
    xc = x - x.mean(0, keepdim=True)
    yc = y - y.mean(0, keepdim=True)
    covariance_x = xc.T @ xc / max(1, x.shape[0] - 1)
    covariance_y = yc.T @ yc / max(1, y.shape[0] - 1)
    covariance_gap = torch.linalg.matrix_norm(covariance_x - covariance_y)
    covariance_scale = torch.linalg.matrix_norm(covariance_y).clamp_min(float(eps))
    variance_x = covariance_x.diagonal().clamp_min(float(eps))
    variance_y = covariance_y.diagonal().clamp_min(float(eps))
    return {
        "mean_gap": float(mean_gap),
        "mean_gap_over_reference_mean": float(mean_gap / mean_scale),
        "covariance_relative_frobenius": float(covariance_gap / covariance_scale),
        "log_variance_rms": float(
            torch.sqrt(torch.mean(torch.log(variance_x / variance_y).square()))
        ),
    }


def sliced_wasserstein_1(
    sample: torch.Tensor,
    reference: torch.Tensor,
    *,
    directions: int = 64,
    seed: int = 0,
) -> float:
    """Two-sample sliced W1 in a fixed projected latent space."""

    if sample.shape != reference.shape or sample.ndim != 2:
        raise ValueError("sample and reference must be equal [N, D] matrices")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    direction = torch.randn(
        (sample.shape[1], int(directions)),
        generator=generator,
        dtype=torch.float32,
    )
    direction = F.normalize(direction, dim=0).to(sample.device)
    x = torch.sort(sample.float() @ direction, dim=0).values
    y = torch.sort(reference.float() @ direction, dim=0).values
    return float((x - y).abs().mean())


def projected_shared_class_viv(
    values: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 1e-12,
) -> dict[str, float | int]:
    """VIV under a fixed projection and shared within-class covariance.

    This is not the full-dimensional class-specific VIV from the source paper.
    It estimates a pooled within-class covariance after removing empirical class
    means, then applies the uniform-time closed form pi/2 * sum sqrt(lambda).
    """

    if values.ndim != 2 or labels.ndim != 1 or values.shape[0] != labels.shape[0]:
        raise ValueError("values must be [N, D] and labels must be [N]")
    values = values.float()
    labels = labels.to(device=values.device)
    residual = torch.empty_like(values)
    valid_count = 0
    class_count = 0
    used = torch.zeros(len(values), dtype=torch.bool, device=values.device)
    for label in torch.unique(labels):
        index = torch.where(labels == label)[0]
        if len(index) < 2:
            continue
        centered = values[index] - values[index].mean(dim=0, keepdim=True)
        residual[index] = centered
        used[index] = True
        valid_count += len(index)
        class_count += 1
    degrees_of_freedom = valid_count - class_count
    if degrees_of_freedom <= 0:
        raise ValueError("at least one class with two samples is required")
    residual = residual[used]
    covariance = residual.T @ residual / degrees_of_freedom
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    viv = 0.5 * math.pi * torch.sqrt(eigenvalues + float(eps)).sum()
    return {
        "projected_viv": float(viv),
        "projected_viv_per_dim": float(viv / values.shape[1]),
        "within_class_trace_per_dim": float(eigenvalues.mean()),
        "effective_rank": float(
            eigenvalues.sum().square()
            / eigenvalues.square().sum().clamp_min(float(eps))
        ),
        "valid_samples": int(valid_count),
        "classes_with_repeats": int(class_count),
        "degrees_of_freedom": int(degrees_of_freedom),
    }


def local_velocity_ambiguity(
    states: torch.Tensor,
    velocities: torch.Tensor,
    *,
    neighbors: int = 8,
    eps: float = 1e-12,
) -> dict[str, float | int]:
    """kNN conditional-variance proxy; deliberately not named VIV."""

    if states.shape != velocities.shape or states.ndim != 2:
        raise ValueError("states and velocities must be equal [N, D] matrices")
    n = states.shape[0]
    if n < 3:
        raise ValueError("at least three samples are required")
    k = min(max(1, int(neighbors)), n - 1)
    states = states.float()
    velocities = velocities.float()
    distances = torch.cdist(states, states)
    distances.fill_diagonal_(float("inf"))
    indices = distances.topk(k=k, largest=False).indices
    local_velocities = velocities[indices]
    prediction = local_velocities.mean(dim=1)
    residual_mse = (velocities - prediction).square().mean()
    centered = velocities - velocities.mean(dim=0, keepdim=True)
    global_variance = centered.square().mean().clamp_min(float(eps))
    local_variance = (
        local_velocities - local_velocities.mean(dim=1, keepdim=True)
    ).square().mean()
    nearest_distance = distances.gather(1, indices[:, :1]).mean()
    return {
        "neighbors": int(k),
        "knn_prediction_mse": float(residual_mse),
        "global_velocity_variance": float(global_variance),
        "ambiguity_ratio": float(residual_mse / global_variance),
        "local_variance_ratio": float(local_variance / global_variance),
        "mean_nearest_state_distance": float(nearest_distance),
    }


def knn_overlap(
    reference: torch.Tensor,
    transformed: torch.Tensor,
    *,
    neighbors: int = 8,
) -> dict[str, float | int]:
    if reference.shape != transformed.shape or reference.ndim != 2:
        raise ValueError("reference and transformed must be equal [N, D] matrices")
    n = reference.shape[0]
    k = min(max(1, int(neighbors)), n - 1)

    def indices(value: torch.Tensor) -> torch.Tensor:
        distances = torch.cdist(value.float(), value.float())
        distances.fill_diagonal_(float("inf"))
        return distances.topk(k=k, largest=False).indices

    left = indices(reference)
    right = indices(transformed)
    matches = (left[:, :, None] == right[:, None, :]).any(dim=2).sum(dim=1)
    intersection = matches.float()
    union = 2 * k - intersection
    return {
        "neighbors": int(k),
        "recall": float((intersection / k).mean()),
        "jaccard": float((intersection / union.clamp_min(1.0)).mean()),
    }


def _rankdata(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("left and right need equal length >= 2")
    x = _rankdata(left)
    y = _rankdata(right)
    x = x - x.mean()
    y = y - y.mean()
    denominator = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denominator == 0:
        return float("nan")
    return float(np.sum(x * y) / denominator)


@dataclass(frozen=True)
class BootstrapCorrelation:
    correlation: float
    ci_low: float
    ci_high: float
    valid_bootstraps: int


def bootstrap_spearman(
    left: Sequence[float],
    right: Sequence[float],
    *,
    resamples: int = 2000,
    seed: int = 0,
) -> BootstrapCorrelation:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    correlation = spearman_correlation(left, right)
    generator = np.random.default_rng(int(seed))
    estimates = []
    for _ in range(int(resamples)):
        index = generator.integers(0, len(left), size=len(left))
        estimate = spearman_correlation(left[index], right[index])
        if np.isfinite(estimate):
            estimates.append(estimate)
    if not estimates:
        return BootstrapCorrelation(correlation, float("nan"), float("nan"), 0)
    low, high = np.quantile(np.asarray(estimates), [0.025, 0.975])
    return BootstrapCorrelation(correlation, float(low), float(high), len(estimates))


def apply_linear_or_jvp(
    transform: Callable[[torch.Tensor], torch.Tensor],
    point: torch.Tensor,
    direction: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use an exact linear shortcut or an autograd JVP for nonlinear maps."""

    if bool(getattr(transform, "is_linear", False)):
        return transform(point), transform(direction)
    return torch.func.jvp(transform, (point,), (direction,))
