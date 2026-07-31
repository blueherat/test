"""Shared statistics for decoder endpoint feature-distribution atlases."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


ATLAS_TENSOR_KEYS = (
    "reference_projection",
    "candidate_projection",
    "spatial_variance_ratio",
    "centered_cosine",
    "raw_mse",
)


def _stable_seed(name: str, base: int) -> int:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int(base) + int.from_bytes(digest[:4], "little") % 1_000_000


class FixedSpatialFeatureProjector:
    """Pool decoder activations and apply a deterministic Gaussian projection."""

    def __init__(
        self,
        output_dim: int = 32,
        spatial_size: int = 4,
        seed: int = 72_913,
    ) -> None:
        if output_dim <= 0 or spatial_size <= 0:
            raise ValueError("projection dimensions must be positive")
        self.output_dim = int(output_dim)
        self.spatial_size = int(spatial_size)
        self.seed = int(seed)
        self._matrices: dict[tuple[str, int, int], torch.Tensor] = {}

    def flatten(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim == 2:
            return value.float()
        if value.ndim != 4:
            raise ValueError(f"expected rank-two or rank-four features, got {value.shape}")
        pooled = F.adaptive_avg_pool2d(
            value.float(),
            (self.spatial_size, self.spatial_size),
        )
        return pooled.flatten(1)

    def __call__(self, layer: str, value: torch.Tensor) -> torch.Tensor:
        flattened = self.flatten(value)
        input_dim = int(flattened.shape[1])
        output_dim = min(self.output_dim, input_dim)
        if output_dim == input_dim:
            return flattened
        key = (str(layer), input_dim, output_dim)
        matrix = self._matrices.get(key)
        if matrix is None:
            generator = torch.Generator(device="cpu").manual_seed(
                _stable_seed(f"{layer}:{input_dim}:{output_dim}", self.seed)
            )
            matrix = torch.randn(
                input_dim,
                output_dim,
                generator=generator,
                dtype=torch.float32,
            ) / math.sqrt(output_dim)
            self._matrices[key] = matrix
        return flattened @ matrix.to(flattened.device)


def _spatial_center(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 2:
        return value - value.mean(dim=1, keepdim=True)
    if value.ndim != 4:
        raise ValueError(f"expected rank-two or rank-four features, got {value.shape}")
    return value - value.mean(dim=(-2, -1), keepdim=True)


def paired_feature_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """Return per-sample spatial calibration and paired-direction metrics."""

    if reference.shape != candidate.shape or reference.ndim not in (2, 4):
        raise ValueError("paired decoder features must have equal rank-two/four shapes")
    reference = reference.float()
    candidate = candidate.float()
    reference_centered = _spatial_center(reference)
    candidate_centered = _spatial_center(candidate)
    if reference.ndim == 4:
        spatial_dims = (-2, -1)
        reference_variance = reference_centered.square().mean(dim=spatial_dims)
        candidate_variance = candidate_centered.square().mean(dim=spatial_dims)
    else:
        reference_variance = reference_centered.square()
        candidate_variance = candidate_centered.square()
    variance_ratio = (
        (candidate_variance + float(eps))
        / (reference_variance + float(eps))
    ).clamp_min(float(eps))
    variance_ratio = variance_ratio.log().mean(dim=1).exp()
    reference_flat = reference_centered.flatten(1)
    candidate_flat = candidate_centered.flatten(1)
    cosine = F.cosine_similarity(
        reference_flat,
        candidate_flat,
        dim=1,
        eps=float(eps),
    )
    raw_mse = (candidate - reference).square().flatten(1).mean(dim=1)
    return {
        "spatial_variance_ratio": variance_ratio,
        "centered_cosine": cosine,
        "raw_mse": raw_mse,
    }


def make_feature_chunks(
    *,
    context: Mapping[str, Any],
    reference_features: Sequence[torch.Tensor],
    candidate_features: Sequence[torch.Tensor],
    layer_indices: Sequence[int],
    layer_fractions: Sequence[float],
    projector: FixedSpatialFeatureProjector,
) -> list[dict[str, Any]]:
    if not (
        len(reference_features)
        == len(candidate_features)
        == len(layer_indices)
        == len(layer_fractions)
    ):
        raise ValueError("feature, layer-index, and layer-fraction lengths differ")
    chunks = []
    for position, (
        reference,
        candidate,
        decoder_layer_index,
        layer_fraction,
    ) in enumerate(
        zip(
            reference_features,
            candidate_features,
            layer_indices,
            layer_fractions,
        )
    ):
        paired = paired_feature_metrics(reference, candidate)
        layer_name = f"layer_{position:02d}"
        chunks.append(
            {
                **dict(context),
                "layer_position": int(position),
                "decoder_layer_index": int(decoder_layer_index),
                "layer_fraction": float(layer_fraction),
                "reference_projection": projector(
                    layer_name,
                    reference,
                ).detach().cpu(),
                "candidate_projection": projector(
                    layer_name,
                    candidate,
                ).detach().cpu(),
                **{
                    name: value.detach().cpu()
                    for name, value in paired.items()
                },
            }
        )
    return chunks


def _covariance(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    value = value.double()
    mean = value.mean(dim=0)
    centered = value - mean
    covariance = centered.T @ centered / max(len(value) - 1, 1)
    return mean, 0.5 * (covariance + covariance.T)


def _psd_sqrt(value: torch.Tensor) -> torch.Tensor:
    eigenvalues, eigenvectors = torch.linalg.eigh(0.5 * (value + value.T))
    return (
        eigenvectors
        * eigenvalues.clamp_min(0.0).sqrt().unsqueeze(0)
    ) @ eigenvectors.T


def _effective_rank(covariance: torch.Tensor) -> float:
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    total = eigenvalues.sum()
    if float(total) <= 1e-18:
        return 0.0
    probabilities = eigenvalues / total
    entropy = -(
        probabilities * probabilities.clamp_min(1e-30).log()
    ).sum()
    return float(entropy.exp())


def _normalized_sliced_wasserstein(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    directions: int,
    seed: int,
) -> float:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    projection = torch.randn(
        reference.shape[1],
        int(directions),
        generator=generator,
        dtype=torch.float64,
    )
    projection = F.normalize(projection, dim=0)
    reference_projected = torch.sort(reference.double() @ projection, dim=0).values
    candidate_projected = torch.sort(candidate.double() @ projection, dim=0).values
    distance = (
        reference_projected - candidate_projected
    ).square().mean().sqrt()
    scale = (
        reference.double() - reference.double().mean(dim=0)
    ).square().mean().sqrt()
    return float(distance / scale.clamp_min(1e-12))


def projected_distribution_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    seed: int,
) -> dict[str, float]:
    """Compare two equally sized projected feature populations."""

    if (
        reference.ndim != 2
        or reference.shape != candidate.shape
        or len(reference) < 8
    ):
        raise ValueError("distribution metrics require equal [N,D] tensors with N>=8")
    reference_mean, reference_covariance = _covariance(reference)
    candidate_mean, candidate_covariance = _covariance(candidate)
    reference_trace = torch.trace(reference_covariance).clamp_min(1e-12)
    candidate_trace = torch.trace(candidate_covariance).clamp_min(0.0)
    mean_square = (reference_mean - candidate_mean).square().sum()
    covariance_difference = reference_covariance - candidate_covariance
    covariance_relative = (
        torch.linalg.matrix_norm(covariance_difference)
        / torch.linalg.matrix_norm(reference_covariance).clamp_min(1e-12)
    )
    reference_root = _psd_sqrt(reference_covariance)
    middle_root = _psd_sqrt(
        reference_root @ candidate_covariance @ reference_root
    )
    frechet = (
        mean_square
        + reference_trace
        + candidate_trace
        - 2.0 * torch.trace(middle_root)
    ).clamp_min(0.0)
    reference_variance = torch.diagonal(reference_covariance).clamp_min(1e-12)
    candidate_variance = torch.diagonal(candidate_covariance).clamp_min(1e-12)
    marginal_variance_ratio = (
        candidate_variance / reference_variance
    ).log().mean().exp()
    return {
        "projected_mean_relative_error": float(
            mean_square.sqrt() / reference_trace.sqrt()
        ),
        "projected_covariance_relative_error": float(covariance_relative),
        "projected_normalized_frechet": float(frechet / reference_trace),
        "projected_normalized_swd": _normalized_sliced_wasserstein(
            reference,
            candidate,
            directions=64,
            seed=int(seed),
        ),
        "projected_covariance_trace_ratio": float(
            candidate_trace / reference_trace
        ),
        "projected_marginal_variance_ratio_gmean": float(
            marginal_variance_ratio
        ),
        "reference_projected_effective_rank": _effective_rank(
            reference_covariance
        ),
        "candidate_projected_effective_rank": _effective_rank(
            candidate_covariance
        ),
    }


def summarize_feature_chunks(
    chunks: Iterable[Mapping[str, Any]],
    *,
    seed: int = 72_913,
) -> list[dict[str, Any]]:
    """Concatenate batch/rank chunks and return one row per atlas condition."""

    grouped: dict[
        tuple[tuple[str, Any], ...],
        dict[str, list[torch.Tensor]],
    ] = defaultdict(lambda: {name: [] for name in ATLAS_TENSOR_KEYS})
    for chunk in chunks:
        missing = [name for name in ATLAS_TENSOR_KEYS if name not in chunk]
        if missing:
            raise KeyError(f"atlas chunk is missing tensors: {missing}")
        context = tuple(
            sorted(
                (str(name), value)
                for name, value in chunk.items()
                if name not in ATLAS_TENSOR_KEYS
            )
        )
        for name in ATLAS_TENSOR_KEYS:
            value = chunk[name]
            if not torch.is_tensor(value):
                raise TypeError(f"atlas field {name!r} is not a tensor")
            grouped[context][name].append(value.detach().cpu())

    rows = []
    for group_index, (context_items, tensors) in enumerate(
        sorted(grouped.items(), key=lambda item: repr(item[0]))
    ):
        context = dict(context_items)
        values = {
            name: torch.cat(parts, dim=0)
            for name, parts in tensors.items()
        }
        sample_count = len(values["reference_projection"])
        if any(len(value) != sample_count for value in values.values()):
            raise RuntimeError("atlas tensor sample counts differ")
        distribution = projected_distribution_metrics(
            values["reference_projection"],
            values["candidate_projection"],
            seed=int(seed) + group_index,
        )
        variance_ratio = values["spatial_variance_ratio"].double().clamp_min(1e-30)
        rows.append(
            {
                **context,
                "sample_count": int(sample_count),
                "projection_dim": int(values["reference_projection"].shape[1]),
                "spatial_variance_ratio_gmean": float(
                    variance_ratio.log().mean().exp()
                ),
                "spatial_variance_log_abs_error": float(
                    variance_ratio.log().abs().mean()
                ),
                "centered_cosine_mean": float(
                    values["centered_cosine"].double().mean()
                ),
                "raw_mse_mean": float(values["raw_mse"].double().mean()),
                **distribution,
            }
        )
    return rows


def relative_log_distance(value: float, target: float = 1.0) -> float:
    if value <= 0 or target <= 0:
        return float("inf")
    return abs(math.log(float(value) / float(target)))


__all__ = [
    "FixedSpatialFeatureProjector",
    "make_feature_chunks",
    "paired_feature_metrics",
    "projected_distribution_metrics",
    "relative_log_distance",
    "summarize_feature_chunks",
]
