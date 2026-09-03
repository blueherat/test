"""Counterfactual query controls for Projected Future Reference guidance.

The deployed PFR query advances information time and adds a small, per-sample
spatial displacement.  This module changes only that displacement while
preserving its norm, so experiments can distinguish a sample-aligned response
from a generic time or perturbation effect.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from experiments.information_purification_ig import (
    InformationQuery,
    projected_information_query,
)
from experiments.path_evidence_pfr_bridge import SampleProjection
from experiments.path_evidence_pfr_bridge import project_per_sample


Tensor = torch.Tensor

QUERY_KINDS = (
    "projected",
    "time_only",
    "state_only",
    "anti_projected",
    "orthogonal_projected",
    "donor_projected",
)


@dataclass(frozen=True)
class QueryControl:
    state: Tensor
    time: Tensor
    spatial_shift: Tensor
    horizon: float
    projected: InformationQuery


@dataclass(frozen=True)
class ResponseSplit:
    temporal: Tensor
    spatial: Tensor
    spatial_parallel: Tensor
    spatial_orthogonal: Tensor
    coefficient: Tensor


def _sample_norm(value: Tensor) -> Tensor:
    return value.float().flatten(1).square().sum(dim=1).sqrt()


def _reshape_sample_scalars(values: Tensor, reference: Tensor) -> Tensor:
    return values.reshape(-1, *((1,) * (reference.ndim - 1)))


def rms_match_per_sample(value: Tensor, reference: Tensor) -> Tensor:
    """Match each sample's Euclidean norm, preserving exact zero references."""

    if value.shape != reference.shape:
        raise ValueError("value and reference must have identical shapes")
    value_norm = _sample_norm(value)
    reference_norm = _sample_norm(reference)
    scale = reference_norm / value_norm.clamp_min(1e-30)
    matched = value * _reshape_sample_scalars(scale.to(value.dtype), value)
    zero = _reshape_sample_scalars(reference_norm == 0, value)
    return torch.where(zero, torch.zeros_like(matched), matched)


def deterministic_scramble(value: Tensor) -> Tensor:
    """Apply an invertible within-sample signed permutation."""

    if value.ndim < 2:
        raise ValueError("expected a batched tensor")
    scrambled = value.flip(1)
    if value.ndim >= 4:
        scrambled = torch.roll(scrambled, shifts=(1, 1), dims=(-2, -1))
    elif value.ndim >= 3:
        scrambled = torch.roll(scrambled, shifts=1, dims=-1)
    signs = torch.ones(
        value.shape[1], device=value.device, dtype=value.dtype
    )
    signs[1::2] = -1
    return scrambled * signs.reshape(1, -1, *((1,) * (value.ndim - 2)))


def matched_orthogonal_scramble(reference: Tensor) -> Tensor:
    """Return a deterministic, norm-matched direction orthogonal per sample."""

    candidate = deterministic_scramble(reference)
    reference_flat = reference.float().flatten(1)
    candidate_flat = candidate.float().flatten(1)
    coefficient = (
        (candidate_flat * reference_flat).sum(dim=1)
        / reference_flat.square().sum(dim=1).clamp_min(1e-30)
    )
    orthogonal = candidate - _reshape_sample_scalars(
        coefficient.to(candidate.dtype), candidate
    ) * reference

    # A signed permutation can only become exactly collinear on a degenerate
    # tensor.  A second permutation supplies a deterministic fallback without
    # introducing an RNG stream into adaptive ODE evaluation.
    degenerate = _sample_norm(orthogonal) <= 1e-20
    if torch.any(degenerate):
        fallback = torch.roll(candidate, shifts=1, dims=1)
        fallback_flat = fallback.float().flatten(1)
        fallback_coefficient = (
            (fallback_flat * reference_flat).sum(dim=1)
            / reference_flat.square().sum(dim=1).clamp_min(1e-30)
        )
        fallback = fallback - _reshape_sample_scalars(
            fallback_coefficient.to(fallback.dtype), fallback
        ) * reference
        orthogonal = torch.where(
            _reshape_sample_scalars(degenerate, reference), fallback, orthogonal
        )
    return rms_match_per_sample(orthogonal, reference)


def matched_donor_shift(reference: Tensor) -> Tensor:
    """Use another sample's displacement and restore the recipient's norm."""

    if len(reference) < 2:
        raise ValueError("donor control requires a batch of at least two samples")
    return rms_match_per_sample(torch.roll(reference, shifts=1, dims=0), reference)


def controlled_information_query(
    state: Tensor,
    time_value: Tensor,
    *,
    strong_now: Tensor,
    weak_now: Tensor,
    guided_now: Tensor,
    gamma: float,
    horizon: float,
    intervention_time: float,
    kind: str,
) -> QueryControl:
    """Construct a PFR query or a norm-controlled counterfactual ablation."""

    if kind not in QUERY_KINDS:
        raise ValueError(f"unknown query kind: {kind}")
    projected = projected_information_query(
        state,
        time_value,
        strong_now=strong_now,
        weak_now=weak_now,
        guided_now=guided_now,
        gamma=gamma,
        horizon=horizon,
        intervention_time=intervention_time,
    )
    shift = projected.state - state
    query_time = projected.time
    if kind == "time_only":
        shift = torch.zeros_like(shift)
    elif kind == "state_only":
        query_time = time_value
    elif kind == "anti_projected":
        shift = -shift
    elif kind == "orthogonal_projected":
        shift = matched_orthogonal_scramble(shift)
    elif kind == "donor_projected":
        shift = matched_donor_shift(shift)
    return QueryControl(
        state=state + shift,
        time=query_time,
        spatial_shift=shift,
        horizon=projected.horizon,
        projected=projected,
    )


def response_odd_even(
    weak_center: Tensor,
    weak_positive: Tensor,
    weak_negative: Tensor,
) -> tuple[Tensor, Tensor]:
    """Split a symmetric finite response into odd and even spatial parts."""

    if not (
        weak_center.shape == weak_positive.shape == weak_negative.shape
    ):
        raise ValueError("weak response tensors must have identical shapes")
    odd = 0.5 * (weak_positive - weak_negative)
    even = 0.5 * (weak_positive + weak_negative) - weak_center
    return odd, even


def split_spatial_response(
    weak_now: Tensor,
    weak_time_only: Tensor,
    weak_projected: Tensor,
) -> ResponseSplit:
    """Split the projected query's spatial response around its time response."""

    if not (
        weak_now.shape == weak_time_only.shape == weak_projected.shape
    ):
        raise ValueError("weak response tensors must have identical shapes")
    temporal = weak_time_only - weak_now
    spatial = weak_projected - weak_time_only
    projection = project_per_sample(spatial, temporal)
    return ResponseSplit(
        temporal=temporal,
        spatial=spatial,
        spatial_parallel=projection.parallel,
        spatial_orthogonal=projection.orthogonal,
        coefficient=projection.coefficient,
    )
