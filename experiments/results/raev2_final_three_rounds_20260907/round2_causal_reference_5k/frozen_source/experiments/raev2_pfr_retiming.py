"""Exact retiming algebra and cheap base-head queries for RAEv2 PFR.

RAEv2 uses the linear bridge

    z_t = (1 - t) x + t epsilon,

with data at ``t=0`` and Gaussian noise at ``t=1``.  The model predicts the
clean endpoint and converts it to the sampling velocity ``epsilon - x``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from experiments.pfr_ou_semigroup_spectrum import (
    ou_degree1_retiming_velocity_defect,
    transport_state_at_fixed_ou_coordinate,
)


def _expand_time(time: Tensor, reference: Tensor) -> Tensor:
    return time.view(time.shape[0], *([1] * (reference.ndim - 1)))


def clean_to_velocity(
    clean: Tensor,
    state: Tensor,
    time: Tensor,
    *,
    denominator_floor: float,
) -> Tensor:
    """Convert a clean prediction to RAEv2's noise-minus-data velocity."""

    denominator = _expand_time(time, state).clamp_min(denominator_floor)
    return (state - clean) / denominator


def euler_dataward_query_state(
    state: Tensor,
    velocity: Tensor,
    current_time: Tensor,
    future_time: Tensor,
) -> Tensor:
    """Predict a dataward state using the current RAE velocity field.

    RAE time decreases from noise to data, so ``future_time < current_time``.
    Writing the signed increment explicitly keeps the formula independent of
    that time convention.
    """

    delta_time = _expand_time(future_time - current_time, state)
    return state + delta_time * velocity


def ordinary_internal_guidance_clean(
    strong_clean: Tensor,
    weak_clean: Tensor,
    *,
    guidance_scale: float,
) -> Tensor:
    """Return ordinary IG in the model's native clean-prediction space."""

    return weak_clean + guidance_scale * (strong_clean - weak_clean)


def bridge_latentized_counterfactual_state(
    state: Tensor,
    strong_clean: Tensor,
    weak_clean: Tensor,
    time: Tensor,
    *,
    guidance_scale: float,
) -> Tensor:
    """Write the ordinary-IG clean proposal into the current linear bridge.

    RAEv2 uses ``z_t = (1-t) x + t epsilon``.  This is the unique state at
    the same time that replaces the weak clean endpoint by the ordinary-IG
    endpoint while keeping the weak predictor's implied noise endpoint fixed.
    """

    guided_clean = ordinary_internal_guidance_clean(
        strong_clean,
        weak_clean,
        guidance_scale=guidance_scale,
    )
    signal = 1.0 - _expand_time(time, state)
    return state + signal * (guided_clean - weak_clean)


def strong_anchored_counterfactual_guidance_clean(
    strong_clean: Tensor,
    counterfactual_weak_clean: Tensor,
    *,
    guidance_scale: float,
) -> Tensor:
    """Replace only IG's negative reference while retaining the strong anchor."""

    extrapolation = guidance_scale - 1.0
    return strong_clean + extrapolation * (
        strong_clean - counterfactual_weak_clean
    )


def orthogonal_counterfactual_guidance_clean(
    strong_clean: Tensor,
    weak_clean: Tensor,
    counterfactual_weak_clean: Tensor,
    *,
    guidance_scale: float,
) -> Tensor:
    """Use only counterfactual direction innovation at the original IG budget.

    The response parallel to the ordinary strong-minus-weak gap is equivalent
    to changing the already calibrated IG scale.  Removing it and restoring
    the original gap norm isolates the only genuinely new direction supplied
    by the counterfactual query.
    """

    depth_gap = strong_clean.float() - weak_clean.float()
    reference_revision = weak_clean.float() - counterfactual_weak_clean.float()
    reduce_dims = tuple(range(1, depth_gap.ndim))
    depth_energy = depth_gap.square().sum(dim=reduce_dims, keepdim=True)
    parallel_coefficient = (reference_revision * depth_gap).sum(
        dim=reduce_dims, keepdim=True
    ) / depth_energy.clamp_min(1e-20)
    orthogonal_revision = reference_revision - parallel_coefficient * depth_gap
    candidate = depth_gap + orthogonal_revision
    candidate_energy = candidate.square().sum(dim=reduce_dims, keepdim=True)
    direction = candidate * torch.sqrt(
        depth_energy / candidate_energy.clamp_min(1e-20)
    )
    extrapolation = guidance_scale - 1.0
    return strong_clean + (extrapolation * direction).to(strong_clean.dtype)


def velocity_to_score(velocity: Tensor, state: Tensor, time: Tensor) -> Tensor:
    """Map a velocity to the score of the RAEv2 linear Gaussian bridge."""

    expanded = _expand_time(time, state)
    return -(state + (1.0 - expanded) * velocity) / expanded


def data_odds(time: Tensor) -> Tensor:
    """Return the data-to-noise odds ``(1-t)/t``."""

    return (1.0 - time) / time


def dataward_future_time(
    current_time: float,
    horizon: float,
    *,
    coordinate: str,
    minimum_time: float,
) -> float:
    """Move dataward in raw time or Gaussian data-odds information time."""

    if horizon < 0.0:
        raise ValueError("horizon must be non-negative")
    if coordinate == "raw_time":
        future = current_time - horizon
    elif coordinate == "log_odds":
        multiplier = math.exp(horizon)
        denominator = current_time + multiplier * (1.0 - current_time)
        future = current_time / denominator if denominator > 0.0 else 0.0
    else:
        raise ValueError(f"unknown horizon coordinate: {coordinate}")
    return max(float(future), float(minimum_time))


def retime_future_score_to_current(
    future_score: Tensor,
    state: Tensor,
    current_time: Tensor,
    future_time: Tensor,
) -> Tensor:
    """Move a future score toward the Gaussian prior on an exponential ray."""

    current_odds = _expand_time(data_odds(current_time), state)
    future_odds = _expand_time(data_odds(future_time), state)
    coefficient = current_odds / future_odds
    gaussian_score = -state
    return coefficient * future_score + (1.0 - coefficient) * gaussian_score


def exponential_retiming_defect(
    current_velocity: Tensor,
    future_velocity: Tensor,
    current_time: Tensor,
) -> Tensor:
    """Score defect from violating a Gaussian-to-data exponential ray."""

    odds = _expand_time(data_odds(current_time), current_velocity)
    return odds * (future_velocity - current_velocity)


def pfr_velocity(
    strong_velocity: Tensor,
    weak_velocity: Tensor,
    weak_future_velocity: Tensor,
    *,
    guidance_scale: float,
    revision_scale: float = 1.0,
    composition: str = "additive",
) -> Tensor:
    """Compose ordinary IG with a time-retiming correction in velocity space."""

    depth_gap = strong_velocity - weak_velocity
    revision = weak_velocity - weak_future_velocity
    if composition == "additive":
        return weak_velocity + guidance_scale * (
            depth_gap + revision_scale * revision
        )
    if composition == "strong_anchored_additive":
        return strong_velocity + (guidance_scale - 1.0) * (
            depth_gap + revision_scale * revision
        )
    if composition not in {
        "norm_preserving",
        "orthogonal_norm_preserving",
        "strong_anchored_additive",
        "strong_anchored_norm_preserving",
        "strong_anchored_angular",
    }:
        raise ValueError(f"unknown PFR composition: {composition}")
    if revision_scale == 0.0:
        return weak_velocity + guidance_scale * depth_gap

    depth_float = depth_gap.float()
    revision_float = revision.float()
    reduce_dims = tuple(range(1, depth_gap.ndim))
    depth_energy = depth_float.square().sum(dim=reduce_dims, keepdim=True)
    if composition in {
        "orthogonal_norm_preserving",
        "strong_anchored_angular",
    }:
        projection = (revision_float * depth_float).sum(
            dim=reduce_dims, keepdim=True
        ) / depth_energy.clamp_min(1e-20)
        revision_float = revision_float - projection * depth_float
    candidate = depth_float + revision_scale * revision_float
    candidate_energy = candidate.square().sum(dim=reduce_dims, keepdim=True)
    norm_ratio = torch.sqrt(
        depth_energy / candidate_energy.clamp_min(1e-20)
    )
    rotated_gap = (candidate * norm_ratio).to(depth_gap.dtype)
    if composition in {
        "strong_anchored_norm_preserving",
        "strong_anchored_angular",
    }:
        return strong_velocity + (guidance_scale - 1.0) * rotated_gap
    return weak_velocity + guidance_scale * rotated_gap


def shared_retiming_revision(
    weak_velocity: Tensor,
    weak_future_velocity: Tensor,
    strong_velocity: Tensor,
    strong_future_velocity: Tensor,
) -> Tensor:
    """Keep the weak retiming change reproduced by the strong predictor.

    The projection is computed independently per sample over the complete
    latent tensor.  It introduces no threshold: a head-specific temporal
    change is suppressed continuously as strong/weak agreement vanishes.
    """

    tensors = (
        weak_velocity,
        weak_future_velocity,
        strong_velocity,
        strong_future_velocity,
    )
    if any(value.shape != weak_velocity.shape for value in tensors[1:]):
        raise ValueError("all retiming velocities must have identical shapes")
    weak_revision = (weak_velocity - weak_future_velocity).float()
    strong_revision = (strong_velocity - strong_future_velocity).float()
    reduce_dims = tuple(range(1, weak_revision.ndim))
    strong_energy = strong_revision.square().sum(
        dim=reduce_dims, keepdim=True
    )
    coefficient = (weak_revision * strong_revision).sum(
        dim=reduce_dims, keepdim=True
    ) / strong_energy.clamp_min(1e-20)
    return (coefficient * strong_revision).to(weak_velocity.dtype)


def transport_raev2_state_at_fixed_ou_coordinate(
    state: Tensor,
    time: Tensor,
    future_time: Tensor,
) -> Tensor:
    """Represent one RAEv2 state at the same normalized OU coordinate.

    RAEv2 uses noise time ``t`` while the SiT OU utilities use data time
    ``u = 1 - t``. A dataward RAEv2 query therefore has
    ``1 - future_time > 1 - time``.
    """

    return transport_state_at_fixed_ou_coordinate(
        state,
        1.0 - time,
        1.0 - future_time,
    )


def raev2_ou_degree1_velocity_defect(
    current_velocity: Tensor,
    future_velocity: Tensor,
    state: Tensor,
    time: Tensor,
    future_time: Tensor,
) -> Tensor:
    """Return the OU degree-1 defect in RAEv2 velocity coordinates.

    Under ``u = 1 - t``, RAEv2 velocity ``epsilon - x`` is the negative of
    the SiT dataward velocity ``x - epsilon``. Reusing the exact, endpoint-safe
    SiT algebra with both sign changes avoids a separate singular formula.
    """

    return -ou_degree1_retiming_velocity_defect(
        -current_velocity,
        -future_velocity,
        state,
        1.0 - time,
        1.0 - future_time,
    )


def project_revision_onto_certificate(
    raw_revision: Tensor,
    certificate: Tensor,
) -> Tensor:
    """Project each sample's raw revision onto one certificate axis."""

    if raw_revision.shape != certificate.shape:
        raise ValueError("raw revision and certificate must have identical shapes")
    raw = raw_revision.float()
    axis = certificate.float()
    reduce_dims = tuple(range(1, raw.ndim))
    coefficient = (raw * axis).sum(dim=reduce_dims, keepdim=True) / axis.square().sum(
        dim=reduce_dims, keepdim=True
    ).clamp_min(1e-30)
    return (coefficient * axis).to(raw_revision.dtype)


def norm_preserving_certificate_revision(
    raw_revision: Tensor,
    certificate: Tensor,
) -> Tensor:
    """Keep raw per-sample RMS along the nearest certificate orientation."""

    common = project_revision_onto_certificate(raw_revision, certificate).float()
    raw = raw_revision.float()
    reduce_dims = tuple(range(1, raw.ndim))
    raw_norm = raw.square().sum(dim=reduce_dims, keepdim=True).sqrt()
    common_norm = common.square().sum(dim=reduce_dims, keepdim=True).sqrt()
    scale = torch.where(
        common_norm > torch.finfo(common_norm.dtype).tiny,
        raw_norm / common_norm.clamp_min(torch.finfo(common_norm.dtype).tiny),
        torch.zeros_like(common_norm),
    )
    return (common * scale).to(raw_revision.dtype)


def evaluate_base_head_only(
    model: nn.Module,
    state: Tensor,
    time: Tensor,
    **condition_kwargs: Tensor | None,
) -> Tensor:
    """Evaluate the official RAEv2 base head without the unused suffix."""

    if not hasattr(model, "base_model_depth") or not hasattr(
        model, "base_final_layer"
    ):
        raise TypeError("model does not expose the RAEv2 internal base head")
    sequence, time_embedding = model._build_sequence(  # noqa: SLF001
        state, time, condition_kwargs
    )
    attention_mask = model._build_attn_mask(  # noqa: SLF001
        sequence, condition_kwargs
    )
    for index in range(int(model.base_model_depth)):
        sequence = model.blocks[index](
            sequence, model.enc_rope, attention_mask
        )
    base = sequence[:, : model.s_embedder.num_patches, :]
    base = F.silu(time_embedding + base)
    base = model.base_final_layer(base, base)
    return model.unpatchify(base, model.s_patch_size)


@dataclass(frozen=True)
class PFRCondition:
    name: str
    guidance_scale: float
    horizon: float = 0.0
    revision_scale: float = 0.0

    @property
    def uses_guidance(self) -> bool:
        return self.guidance_scale != 1.0

    @property
    def uses_retiming(self) -> bool:
        return self.horizon > 0.0 and self.revision_scale != 0.0
