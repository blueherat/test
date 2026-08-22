"""Train-only diagonal moment decomposition for ImageNet-100 SiT latents.

For the linear path ``z_t = (1 - t) * epsilon + t * x``, this module removes
the best affine velocity predictor in the coordinate-wise (diagonal) family.
The neural SiT then predicts the variance-normalized residual, while both
training and sampling recover an ordinary velocity field before using it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


MOMENT_STATS_FORMAT = "eqvae_imagenet100_sdvae_diagonal_moments_v1"
VELOCITY_DECOMPOSITIONS = ("native", "diagonal_lmmse")


def _time_image(time_value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if time_value.shape != (reference.shape[0],):
        raise ValueError("time_value must have shape [B]")
    return time_value.float().reshape(-1, *([1] * (reference.ndim - 1)))


@dataclass(frozen=True)
class DiagonalMomentStats:
    mean: torch.Tensor
    variance: torch.Tensor
    count: int
    cache_manifest_sha256: str
    scaling_factor: float
    source_path: str
    source_sha256: str

    def __post_init__(self) -> None:
        if self.mean.shape != self.variance.shape or self.mean.ndim != 3:
            raise ValueError("mean and variance must have identical [C,H,W] shapes")
        if self.count < 1:
            raise ValueError("moment count must be positive")
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.variance).all():
            raise ValueError("moment statistics must be finite")
        if (self.variance <= 0).any():
            raise ValueError("all diagonal variances must be positive")

    def to(self, device: torch.device | str) -> "DiagonalMomentStats":
        return DiagonalMomentStats(
            mean=self.mean.to(device=device, dtype=torch.float32),
            variance=self.variance.to(device=device, dtype=torch.float32),
            count=self.count,
            cache_manifest_sha256=self.cache_manifest_sha256,
            scaling_factor=self.scaling_factor,
            source_path=self.source_path,
            source_sha256=self.source_sha256,
        )

    def checkpoint_payload(self) -> dict:
        return {
            "format": MOMENT_STATS_FORMAT,
            "mean": self.mean.detach().cpu(),
            "variance": self.variance.detach().cpu(),
            "count": self.count,
            "cache_manifest_sha256": self.cache_manifest_sha256,
            "scaling_factor": self.scaling_factor,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
        }


def diagonal_stats_from_payload(payload: dict) -> DiagonalMomentStats:
    if payload.get("format") != MOMENT_STATS_FORMAT:
        raise ValueError(f"unsupported moment statistics format: {payload.get('format')!r}")
    return DiagonalMomentStats(
        mean=torch.as_tensor(payload["mean"], dtype=torch.float32),
        variance=torch.as_tensor(payload["variance"], dtype=torch.float32),
        count=int(payload["count"]),
        cache_manifest_sha256=str(payload["cache_manifest_sha256"]),
        scaling_factor=float(payload["scaling_factor"]),
        source_path=str(payload.get("source_path", "")),
        source_sha256=str(payload.get("source_sha256", "")),
    )


def load_diagonal_moment_stats(
    path: Path,
    *,
    expected_cache_manifest_sha256: str | None = None,
    expected_scaling_factor: float | None = None,
) -> DiagonalMomentStats:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    stats = diagonal_stats_from_payload(payload)
    if (
        expected_cache_manifest_sha256 is not None
        and stats.cache_manifest_sha256 != expected_cache_manifest_sha256
    ):
        raise ValueError("moment statistics were estimated from a different cache manifest")
    if (
        expected_scaling_factor is not None
        and abs(stats.scaling_factor - expected_scaling_factor) > 1e-12
    ):
        raise ValueError("moment statistics use a different VAE scaling factor")
    return stats


def diagonal_lmmse_terms(
    state: torch.Tensor,
    time_value: torch.Tensor,
    stats: DiagonalMomentStats,
    *,
    variance_floor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the analytic velocity and residual standard deviation.

    The covariance is restricted to a diagonal matrix. Within that affine
    family these are the exact LMMSE terms, not an approximation introduced by
    optimization.
    """

    if state.ndim != 4 or tuple(state.shape[1:]) != tuple(stats.mean.shape):
        raise ValueError("state shape does not match diagonal moment statistics")
    if variance_floor <= 0:
        raise ValueError("variance_floor must be positive")
    time_image = _time_image(time_value, state)
    one_minus_time = 1.0 - time_image
    mean = stats.mean.to(device=state.device, dtype=torch.float32).unsqueeze(0)
    variance = stats.variance.to(device=state.device, dtype=torch.float32).clamp_min(
        variance_floor
    ).unsqueeze(0)
    covariance_t = one_minus_time.square() + time_image.square() * variance
    cross_covariance = time_image * variance - one_minus_time
    analytic_velocity = mean + (cross_covariance / covariance_t) * (
        state.float() - time_image * mean
    )
    residual_std = (variance / covariance_t).sqrt()
    return analytic_velocity, residual_std


def moment_residual_to_velocity(
    residual_prediction: torch.Tensor,
    *,
    state: torch.Tensor,
    time_value: torch.Tensor,
    stats: DiagonalMomentStats,
    variance_floor: float,
) -> torch.Tensor:
    if residual_prediction.shape != state.shape:
        raise ValueError("residual prediction and state must have identical shapes")
    analytic_velocity, residual_std = diagonal_lmmse_terms(
        state,
        time_value,
        stats,
        variance_floor=variance_floor,
    )
    return analytic_velocity + residual_std * residual_prediction.float()


def moment_residual_losses(
    residual_prediction: torch.Tensor,
    *,
    state: torch.Tensor,
    velocity_target: torch.Tensor,
    time_value: torch.Tensor,
    stats: DiagonalMomentStats,
    variance_floor: float,
) -> dict[str, torch.Tensor]:
    """Optimize the same recovered velocity MSE as the native SiT baseline."""

    if not (residual_prediction.shape == state.shape == velocity_target.shape):
        raise ValueError("prediction, state, and velocity target shapes must match")
    analytic_velocity, residual_std = diagonal_lmmse_terms(
        state,
        time_value,
        stats,
        variance_floor=variance_floor,
    )
    normalized_target = (velocity_target.float() - analytic_velocity) / residual_std
    native_loss = F.mse_loss(
        residual_prediction.float(), normalized_target, reduction="mean"
    )
    velocity_prediction = (
        analytic_velocity + residual_std * residual_prediction.float()
    )
    velocity_loss = F.mse_loss(
        velocity_prediction, velocity_target.float(), reduction="mean"
    )
    return {
        "optimized": velocity_loss,
        "native": native_loss,
        "velocity": velocity_loss,
    }
