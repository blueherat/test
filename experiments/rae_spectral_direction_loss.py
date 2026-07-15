"""Fixed direction-only DCT loss for the tiny RAE transfer experiment."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn


def dct_matrix(size: int, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    n = torch.arange(int(size), dtype=dtype)[None, :]
    k = torch.arange(int(size), dtype=dtype)[:, None]
    matrix = torch.cos(math.pi * (n + 0.5) * k / int(size))
    matrix[0] *= math.sqrt(1.0 / int(size))
    if size > 1:
        matrix[1:] *= math.sqrt(2.0 / int(size))
    return matrix


def radial_band_index(size: int, band_count: int) -> torch.Tensor:
    coefficient_count = int(size) ** 2
    if not 1 <= int(band_count) <= coefficient_count:
        raise ValueError("band_count must lie in [1, size**2]")
    coordinates = torch.cartesian_prod(torch.arange(size), torch.arange(size))
    radius = coordinates[:, 0].float().square() + coordinates[:, 1].float().square()
    order = torch.argsort(radius, stable=True)
    index = torch.empty(coefficient_count, dtype=torch.long)
    for band, positions in enumerate(torch.tensor_split(order, int(band_count))):
        index[positions] = band
    return index.reshape(int(size), int(size))


def bounded_coefficient_mean_one(
    raw: torch.Tensor,
    counts: torch.Tensor,
    lower: float,
    upper: float,
    iterations: int = 40,
) -> torch.Tensor:
    """Scale each positive row into fixed bounds with weighted mean exactly one."""

    if not 0.0 < float(lower) <= 1.0 <= float(upper):
        raise ValueError("weight bounds must satisfy 0 < lower <= 1 <= upper")
    counts = counts.to(device=raw.device, dtype=raw.dtype)
    denominator = counts.sum().clamp_min(1.0)
    left = torch.zeros((raw.shape[0], 1), device=raw.device, dtype=raw.dtype)
    right = (float(upper) / raw.min(dim=1, keepdim=True).values.clamp_min(1e-20)).clamp_min(1.0)
    for _ in range(int(iterations)):
        middle = (left + right) * 0.5
        candidate = (raw * middle).clamp(float(lower), float(upper))
        mean = (candidate * counts[None]).sum(dim=1, keepdim=True) / denominator
        left = torch.where(mean < 1.0, middle, left)
        right = torch.where(mean >= 1.0, middle, right)
    return (raw * ((left + right) * 0.5)).clamp(float(lower), float(upper))


class DCTDirectionLoss(nn.Module):
    """Per-time mean-one weighting over fixed radial DCT frequency bands."""

    def __init__(
        self,
        spatial_size: int,
        second_moments: Sequence[float],
        *,
        gamma: float,
        damping: float = 1e-4,
        min_weight: float = 0.2,
        max_weight: float = 2.0,
    ) -> None:
        super().__init__()
        moments = torch.as_tensor(list(second_moments), dtype=torch.float32)
        if moments.ndim != 1 or moments.numel() < 1 or torch.any(moments <= 0):
            raise ValueError("second_moments must be a non-empty positive vector")
        if float(gamma) < 0:
            raise ValueError("gamma must be non-negative")
        if float(damping) < 0:
            raise ValueError("damping must be non-negative")
        index = radial_band_index(int(spatial_size), int(moments.numel()))
        counts = torch.bincount(index.flatten(), minlength=int(moments.numel())).float()
        self.spatial_size = int(spatial_size)
        self.band_count = int(moments.numel())
        self.gamma = float(gamma)
        self.damping = float(damping)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.register_buffer("dct", dct_matrix(self.spatial_size), persistent=False)
        self.register_buffer("band_index", index, persistent=False)
        self.register_buffer("band_counts", counts, persistent=False)
        self.register_buffer("second_moments", moments, persistent=True)

    def residual_variance(self, time: torch.Tensor) -> torch.Tensor:
        time = time.reshape(-1, 1)
        moments = self.second_moments.to(device=time.device, dtype=time.dtype)[None]
        denominator = (1.0 - time).square() * moments + time.square()
        return moments / denominator.clamp_min(1e-12)

    def weights(self, time: torch.Tensor) -> torch.Tensor:
        raw = (self.residual_variance(time) + self.damping).pow(-self.gamma)
        return bounded_coefficient_mean_one(
            raw,
            self.band_counts,
            self.min_weight,
            self.max_weight,
        )

    def transform(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[-2:] != (self.spatial_size, self.spatial_size):
            raise ValueError(
                f"expected [B,C,{self.spatial_size},{self.spatial_size}], got {tuple(value.shape)}"
            )
        matrix = self.dct.to(device=value.device, dtype=value.dtype)
        return torch.matmul(torch.matmul(matrix, value), matrix.T)

    def band_mse(self, error: torch.Tensor) -> torch.Tensor:
        coefficient_mse = self.transform(error).square().mean(dim=1).flatten(1)
        index = self.band_index.flatten().to(device=error.device)
        sums = torch.zeros(
            (error.shape[0], self.band_count), device=error.device, dtype=error.dtype
        )
        sums.scatter_add_(1, index[None].expand(error.shape[0], -1), coefficient_mse)
        return sums / self.band_counts.to(device=error.device, dtype=error.dtype)[None]

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        time: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if prediction.shape != target.shape:
            raise ValueError("prediction and target must have the same shape")
        error = prediction - target
        band_mse = self.band_mse(error)
        weights = self.weights(time).to(dtype=band_mse.dtype)
        counts = self.band_counts.to(device=band_mse.device, dtype=band_mse.dtype)
        loss = (band_mse * weights * counts[None]).sum(dim=1) / counts.sum()
        return loss, {
            "raw_mse": error.square().flatten(1).mean(dim=1),
            "band_mse": band_mse,
            "band_weights": weights,
            "residual_variance": self.residual_variance(time).to(dtype=band_mse.dtype),
        }


__all__ = [
    "DCTDirectionLoss",
    "bounded_coefficient_mean_one",
    "dct_matrix",
    "radial_band_index",
]
