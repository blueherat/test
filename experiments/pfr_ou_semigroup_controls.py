"""Causal controls for the OU mean-mode-annihilating PFR component.

The linear flow bridge is exactly a variance-preserving Ornstein--Uhlenbeck
channel after a deterministic rescaling. Near the Gaussian fixed point, a
degree-1 density-ratio mode has relative-score amplitude proportional to
``alpha_t``. A two-time defect can therefore annihilate that universal mode
without fitting a coefficient.

This module does not assume that a finite neural score has an exact Hermite
decomposition. It asks a narrower causal question: how much of the deployed
time-only PFR correction lies along that theoretically defined defect, and
which exact component carries the endpoint quality change?
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from experiments.internal_guidance_path_extrapolation import project_per_sample
from experiments.pfr_ou_semigroup_spectrum import (
    ou_degree_retiming_velocity_defect,
    ou_degree1_retiming_velocity_defect,
    transport_state_at_fixed_ou_coordinate,
)
from experiments.pfr_query_controls import controlled_information_query
from experiments.pfr_retiming_controls import rms_match_per_sample
from experiments.run_imagenet100_sit_path_evidence_pfr_bridge import (
    HORIZON,
    INTERVENTION_TIME,
    gamma_at,
)


Tensor = torch.Tensor

OU_SPECTRAL_CONTROL_KINDS = (
    "ou_d1_common",
    "ou_d1_unique",
    "ou_d1_rms_matched",
    "ou_d1_common_first",
    "ou_d1_unique_first",
    "ou_d1_common_norm_raw_direction_first",
    "ou_d1_common_direction_raw_norm_first",
    "ou_d1_common_then_projected",
    "ou_d1_common_plus_spatial",
    "ou_d1_energy_adaptive",
    "ou_d1_two_scale_span_first",
    "ou_d1_strong_common_first",
    "ou_d1_strong_unique_first",
    "ou_d1_strong_common_norm_raw_direction_first",
    "ou_d1_strong_common_direction_raw_norm_first",
    "ou_d1_strong_anchored_common_direction_raw_norm_first",
    "ou_d1_strong_anchored_angular_first",
    "ou_d2_strong_common_first",
    "ou_d2_strong_common_direction_raw_norm_first",
    "ou_d2_common_first",
    "ou_d2_unique_first",
)


@dataclass(frozen=True)
class OUSpectralRevisionSplit:
    raw: Tensor
    spectral: Tensor
    common: Tensor
    unique: Tensor
    spectral_rms_matched: Tensor


def split_raw_revision_against_ou_degree1(
    raw_revision: Tensor,
    spectral_revision: Tensor,
) -> OUSpectralRevisionSplit:
    """Split a raw PFR revision along the OU defect, per sample."""

    if raw_revision.shape != spectral_revision.shape:
        raise ValueError("raw and spectral revisions must have identical shapes")
    projection = project_per_sample(raw_revision, spectral_revision)
    return OUSpectralRevisionSplit(
        raw=raw_revision,
        spectral=spectral_revision,
        common=projection.parallel,
        unique=projection.orthogonal,
        spectral_rms_matched=rms_match_per_sample(
            spectral_revision, raw_revision
        ),
    )


def energy_adaptive_ou_revision(split: OUSpectralRevisionSplit) -> Tensor:
    """Shrink unsupported energy in proportion to spectral explained energy.

    Let ``q = ||common||^2 / ||raw||^2``. Orthogonality guarantees
    ``q in [0, 1]`` up to floating-point error. The parameter-free revision

    ``common + (1 - q) * unique``

    uses the certified component when the OU defect explains the raw revision,
    but continuously returns to the complete raw revision when that explanation
    loses support.
    """

    raw_flat = split.raw.float().flatten(1)
    common_flat = split.common.float().flatten(1)
    explained = common_flat.square().sum(1) / raw_flat.square().sum(1).clamp_min(
        1e-30
    )
    explained = explained.clamp(0.0, 1.0)
    explained = explained.reshape(-1, *((1,) * (split.raw.ndim - 1)))
    return split.common + (1.0 - explained).to(split.unique.dtype) * split.unique


def project_onto_sample_span(
    value: Tensor,
    references: tuple[Tensor, ...],
) -> Tensor:
    """Project each sample onto the span of a small reference frame."""

    if not references:
        raise ValueError("at least one reference is required")
    if value.ndim < 2 or any(
        reference.shape != value.shape for reference in references
    ):
        raise ValueError("value and references must share batch and feature shapes")
    value_flat = value.float().flatten(1)
    frame = torch.stack(
        [reference.float().flatten(1) for reference in references], dim=-1
    )
    gram = frame.transpose(1, 2) @ frame
    rhs = frame.transpose(1, 2) @ value_flat.unsqueeze(-1)
    coefficients = torch.linalg.pinv(gram, hermitian=True) @ rhs
    projected = (frame @ coefficients).squeeze(-1).reshape_as(value)
    return projected.to(value.dtype)


def select_ou_spectral_revision(
    split: OUSpectralRevisionSplit,
    condition: str,
    *,
    time_value: float,
) -> Tensor:
    """Select one preregistered exact component without a fitted gain."""

    if condition == "ou_d1_common":
        return split.common
    if condition == "ou_d1_unique":
        return split.unique
    if condition == "ou_d1_rms_matched":
        return split.spectral_rms_matched
    if condition == "ou_d1_energy_adaptive":
        return energy_adaptive_ou_revision(split)
    if condition == "ou_d1_common_first":
        return split.common if time_value < 0.25 else split.raw
    if condition == "ou_d1_unique_first":
        return split.unique if time_value < 0.25 else split.raw
    if condition == "ou_d1_common_norm_raw_direction_first":
        return (
            rms_match_per_sample(split.raw, split.common)
            if time_value < 0.25
            else split.raw
        )
    if condition == "ou_d1_common_direction_raw_norm_first":
        return (
            rms_match_per_sample(split.common, split.raw)
            if time_value < 0.25
            else split.raw
        )
    if condition == "ou_d2_common_first":
        return split.common if time_value < 0.25 else split.raw
    if condition == "ou_d2_unique_first":
        return split.unique if time_value < 0.25 else split.raw
    raise ValueError(f"unknown OU spectral control: {condition}")


def strong_anchored_angular_guidance(
    strong: Tensor,
    weak: Tensor,
    revision: Tensor,
    *,
    gamma: float,
) -> Tensor:
    """Rotate the strong-anchored extrapolation without changing its norm."""

    gap = strong - weak
    directional_revision = project_per_sample(revision, gap).orthogonal
    rotated_gap = rms_match_per_sample(gap + directional_revision, gap)
    return strong + gamma * rotated_gap


class OUSpectralControlField:
    """Ordinary IG plus an OU-semigroup-filtered weak time revision."""

    def __init__(self, runtime: Any, labels: Tensor, condition: str) -> None:
        if condition not in OU_SPECTRAL_CONTROL_KINDS:
            raise ValueError(f"unknown OU spectral control: {condition}")
        self.runtime = runtime
        self.labels = labels
        self.condition = condition
        self.nfe = 0
        self.query_nfe = 0
        self.full_query_nfe = 0

    def __call__(self, time_value: Tensor, state: Tensor) -> Tensor:
        self.nfe += 1
        with torch.inference_mode():
            strong, weak = self.runtime.evaluate_pair(time_value, state, self.labels)
            scalar_time = float(time_value.detach().float().item())
            gamma = gamma_at(scalar_time)
            guided = strong + gamma * (strong - weak)
            if gamma == 0.0:
                return guided

            horizon = min(HORIZON, INTERVENTION_TIME - scalar_time)
            if horizon <= 0.0:
                return guided
            future_time = time_value + horizon

            if self.condition in {
                "ou_d1_common_then_projected",
                "ou_d1_common_plus_spatial",
            } and scalar_time >= 0.25:
                projected = controlled_information_query(
                    state,
                    time_value,
                    strong_now=strong,
                    weak_now=weak,
                    guided_now=guided,
                    gamma=gamma,
                    horizon=horizon,
                    intervention_time=INTERVENTION_TIME,
                    kind="projected",
                )
                weak_projected = self.runtime.evaluate_weak(
                    projected.time, projected.state, self.labels
                )
                self.query_nfe += 1
                return guided + (1.0 + gamma) * (weak - weak_projected)

            weak_future_raw = self.runtime.evaluate_weak(
                future_time, state, self.labels
            )
            self.query_nfe += 1
            raw_revision = weak - weak_future_raw
            anchored_revision = (
                self.condition
                == "ou_d1_strong_anchored_common_direction_raw_norm_first"
            )
            revision_multiplier = gamma if anchored_revision else 1.0 + gamma
            if self.condition in {
                "ou_d1_common_first",
                "ou_d1_unique_first",
                "ou_d1_common_norm_raw_direction_first",
                "ou_d1_common_direction_raw_norm_first",
                "ou_d1_two_scale_span_first",
                "ou_d1_strong_common_first",
                "ou_d1_strong_unique_first",
                "ou_d1_strong_common_norm_raw_direction_first",
                "ou_d1_strong_common_direction_raw_norm_first",
                "ou_d1_strong_anchored_common_direction_raw_norm_first",
                "ou_d1_strong_anchored_angular_first",
                "ou_d2_strong_common_first",
                "ou_d2_strong_common_direction_raw_norm_first",
                "ou_d2_common_first",
                "ou_d2_unique_first",
            } and scalar_time >= 0.25:
                if self.condition == "ou_d1_strong_anchored_angular_first":
                    return strong_anchored_angular_guidance(
                        strong,
                        weak,
                        raw_revision,
                        gamma=gamma,
                    )
                return guided + revision_multiplier * (weak - weak_future_raw)

            future_ou_state = transport_state_at_fixed_ou_coordinate(
                state, time_value, future_time
            )
            if self.condition in {
                "ou_d1_strong_common_first",
                "ou_d1_strong_unique_first",
                "ou_d1_strong_common_norm_raw_direction_first",
                "ou_d1_strong_common_direction_raw_norm_first",
                "ou_d1_strong_anchored_common_direction_raw_norm_first",
                "ou_d1_strong_anchored_angular_first",
                "ou_d2_strong_common_first",
                "ou_d2_strong_common_direction_raw_norm_first",
            }:
                strong_future_ou, _ = self.runtime.evaluate_pair(
                    future_time, future_ou_state, self.labels
                )
                self.full_query_nfe += 1
                if self.condition.startswith("ou_d2_"):
                    strong_spectral_revision = ou_degree_retiming_velocity_defect(
                        strong,
                        strong_future_ou,
                        state,
                        time_value,
                        future_time,
                        degree=2.0,
                    )
                else:
                    strong_spectral_revision = ou_degree1_retiming_velocity_defect(
                        strong,
                        strong_future_ou,
                        state,
                        time_value,
                        future_time,
                    )
                strong_split = split_raw_revision_against_ou_degree1(
                    raw_revision, strong_spectral_revision
                )
                if self.condition in {
                    "ou_d1_strong_common_first",
                    "ou_d2_strong_common_first",
                }:
                    revision = strong_split.common
                elif self.condition == "ou_d1_strong_unique_first":
                    revision = strong_split.unique
                elif self.condition == "ou_d1_strong_common_norm_raw_direction_first":
                    revision = rms_match_per_sample(
                        strong_split.raw, strong_split.common
                    )
                else:
                    revision = rms_match_per_sample(
                        strong_split.common, strong_split.raw
                    )
                if self.condition == "ou_d1_strong_anchored_angular_first":
                    return strong_anchored_angular_guidance(
                        strong,
                        weak,
                        revision,
                        gamma=gamma,
                    )
                return guided + revision_multiplier * revision
            weak_future_ou = self.runtime.evaluate_weak(
                future_time, future_ou_state, self.labels
            )
            self.query_nfe += 1

            if self.condition.startswith("ou_d2_"):
                spectral_revision = ou_degree_retiming_velocity_defect(
                    weak,
                    weak_future_ou,
                    state,
                    time_value,
                    future_time,
                    degree=2.0,
                )
            else:
                spectral_revision = ou_degree1_retiming_velocity_defect(
                    weak,
                    weak_future_ou,
                    state,
                    time_value,
                    future_time,
                )
            if self.condition == "ou_d1_two_scale_span_first":
                long_future_time = time_value + 2.0 * HORIZON
                long_future_ou_state = transport_state_at_fixed_ou_coordinate(
                    state, time_value, long_future_time
                )
                weak_long_future_ou = self.runtime.evaluate_weak(
                    long_future_time, long_future_ou_state, self.labels
                )
                self.query_nfe += 1
                long_spectral_revision = ou_degree1_retiming_velocity_defect(
                    weak,
                    weak_long_future_ou,
                    state,
                    time_value,
                    long_future_time,
                )
                revision = project_onto_sample_span(
                    raw_revision,
                    (spectral_revision, long_spectral_revision),
                )
                return guided + (1.0 + gamma) * revision
            split = split_raw_revision_against_ou_degree1(
                raw_revision, spectral_revision
            )
            revision = select_ou_spectral_revision(
                split,
                (
                    "ou_d1_common"
                    if self.condition in {
                        "ou_d1_common_then_projected",
                        "ou_d1_common_plus_spatial",
                    }
                    else self.condition
                ),
                time_value=scalar_time,
            )
            if self.condition == "ou_d1_common_plus_spatial":
                projected = controlled_information_query(
                    state,
                    time_value,
                    strong_now=strong,
                    weak_now=weak,
                    guided_now=guided,
                    gamma=gamma,
                    horizon=horizon,
                    intervention_time=INTERVENTION_TIME,
                    kind="projected",
                )
                weak_projected = self.runtime.evaluate_weak(
                    projected.time, projected.state, self.labels
                )
                self.query_nfe += 1
                revision = revision + weak_future_raw - weak_projected
            return guided + (1.0 + gamma) * revision
