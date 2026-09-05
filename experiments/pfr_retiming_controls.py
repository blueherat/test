"""Causal controls for strong/weak exponential-retiming defects.

For a fixed latent coordinate ``z`` and a future clock value ``tau=t+h``, the
time-only PFR correction is proportional to ``W(z,t)-W(z,tau)``.  This module
compares that weak-field secant with the corresponding strong-field secant
``S(z,t)-S(z,tau)`` and splits the weak secant into the component shared with
the strong secant plus a weak-specific orthogonal residual.

All projections are per sample over the complete latent tensor.  The split is
therefore an exact Euclidean control, not a claim about manifold tangency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from experiments.internal_guidance_path_extrapolation import project_per_sample
from experiments.pfr_information_clock import matched_information_horizon
from experiments.run_imagenet100_sit_path_evidence_pfr_bridge import (
    HORIZON,
    INTERVENTION_TIME,
    gamma_at,
)


Tensor = torch.Tensor

RETIMING_CONTROL_KINDS = (
    "weak_time_pair",
    "strong_time",
    "strong_time_rms_matched",
    "weak_common_strong",
    "weak_unique_strong",
)
MULTIDEPTH_RETIMING_KINDS = (
    "weak_common_depth10",
    "weak_unique_depth10",
)


@dataclass(frozen=True)
class RetimingRevisionSplit:
    weak: Tensor
    strong: Tensor
    weak_common: Tensor
    weak_unique: Tensor
    strong_rms_matched: Tensor


def _sample_rms(value: Tensor) -> Tensor:
    if value.ndim < 2:
        raise ValueError("sample RMS expects a batch and feature dimensions")
    return value.float().flatten(1).square().mean(1).sqrt()


def rms_match_per_sample(value: Tensor, reference: Tensor) -> Tensor:
    """Scale ``value`` to the per-sample RMS of ``reference``."""

    if value.shape != reference.shape:
        raise ValueError("value and reference must have identical shapes")
    value_rms = _sample_rms(value)
    reference_rms = _sample_rms(reference)
    tiny = torch.finfo(value_rms.dtype).tiny
    scale_flat = torch.where(
        value_rms > tiny,
        reference_rms / value_rms.clamp_min(tiny),
        torch.zeros_like(value_rms),
    )
    scale = scale_flat.reshape(len(value), *([1] * (value.ndim - 1)))
    return value * scale.to(dtype=value.dtype)


def split_weak_retiming_against_strong(
    weak_revision: Tensor,
    strong_revision: Tensor,
) -> RetimingRevisionSplit:
    """Split the weak time secant along the strong time-secant direction."""

    if weak_revision.shape != strong_revision.shape:
        raise ValueError("weak and strong revisions must have identical shapes")
    projection = project_per_sample(weak_revision, strong_revision)
    return RetimingRevisionSplit(
        weak=weak_revision,
        strong=strong_revision,
        weak_common=projection.parallel,
        weak_unique=projection.orthogonal,
        strong_rms_matched=rms_match_per_sample(strong_revision, weak_revision),
    )


def select_retiming_revision(
    split: RetimingRevisionSplit,
    condition: str,
) -> Tensor:
    choices = {
        "weak_time_pair": split.weak,
        "strong_time": split.strong,
        "strong_time_rms_matched": split.strong_rms_matched,
        "weak_common_strong": split.weak_common,
        "weak_unique_strong": split.weak_unique,
    }
    try:
        return choices[condition]
    except KeyError as error:
        raise ValueError(f"unknown retiming control: {condition}") from error


class RetimingControlField:
    """Ordinary IG plus one selected strong/weak time-retiming correction."""

    def __init__(
        self,
        runtime: Any,
        labels: Tensor,
        condition: str,
        *,
        query_clock: str = "raw_t",
        clock_anchor_time: float = 0.25,
        anchor_horizon: float = HORIZON,
    ) -> None:
        if condition not in RETIMING_CONTROL_KINDS:
            raise ValueError(f"unknown retiming control: {condition}")
        self.runtime = runtime
        self.labels = labels
        self.condition = condition
        self.query_clock = query_clock
        self.clock_anchor_time = float(clock_anchor_time)
        self.anchor_horizon = float(anchor_horizon)
        self.nfe = 0
        self.query_nfe = 0
        self.full_query_nfe = 0

    def __call__(self, time_value: Tensor, state: Tensor) -> Tensor:
        self.nfe += 1
        with torch.inference_mode():
            strong, weak = self.runtime.evaluate_pair(time_value, state, self.labels)
            gamma = gamma_at(float(time_value.detach().float().item()))
            guided = strong + gamma * (strong - weak)
            if gamma == 0.0:
                return guided

            horizon = matched_information_horizon(
                float(time_value.detach().float().item()),
                clock=self.query_clock,
                anchor_time=self.clock_anchor_time,
                anchor_horizon=self.anchor_horizon,
                intervention_time=INTERVENTION_TIME,
            )
            if horizon <= 0.0:
                return guided

            future_time = time_value + horizon
            strong_future, weak_future = self.runtime.evaluate_pair(
                future_time, state, self.labels
            )
            self.full_query_nfe += 1
            split = split_weak_retiming_against_strong(
                weak - weak_future,
                strong - strong_future,
            )
            revision = select_retiming_revision(split, self.condition)
            return guided + (1.0 + gamma) * revision


class MultiDepthRetimingField:
    """Filter the depth-4 retiming correction using a deeper internal head."""

    def __init__(
        self,
        runtime: Any,
        labels: Tensor,
        condition: str,
        *,
        deeper_head: Any,
        query_clock: str = "raw_t",
        clock_anchor_time: float = 0.25,
        anchor_horizon: float = HORIZON,
    ) -> None:
        if condition not in MULTIDEPTH_RETIMING_KINDS:
            raise ValueError(f"unknown multidepth retiming control: {condition}")
        self.runtime = runtime
        self.labels = labels
        self.condition = condition
        self.deeper_head = deeper_head
        self.query_clock = query_clock
        self.clock_anchor_time = float(clock_anchor_time)
        self.anchor_horizon = float(anchor_horizon)
        self.nfe = 0
        self.query_nfe = 0
        self.full_query_nfe = 0

    def _evaluate_current(
        self, time_value: Tensor, state: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        heads = {
            "depth4_v": self.runtime.head,
            self.deeper_head.name: self.deeper_head,
        }
        full, trained, _ = self.runtime.modules["evaluate_source_with_heads"](
            self.runtime.strong,
            state,
            time_value.expand(len(state)),
            self.labels,
            heads=heads,
        )
        return full, trained["depth4_v"], trained[self.deeper_head.name]

    def _evaluate_future(self, time_value: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        heads = {
            "depth4_v": self.runtime.head,
            self.deeper_head.name: self.deeper_head,
        }
        trained = self.runtime.modules["evaluate_internal_heads_only"](
            self.runtime.strong,
            state,
            time_value.expand(len(state)),
            self.labels,
            heads=heads,
        )
        return trained["depth4_v"], trained[self.deeper_head.name]

    def __call__(self, time_value: Tensor, state: Tensor) -> Tensor:
        self.nfe += 1
        with torch.inference_mode():
            strong, weak, deeper = self._evaluate_current(time_value, state)
            gamma = gamma_at(float(time_value.detach().float().item()))
            guided = strong + gamma * (strong - weak)
            if gamma == 0.0:
                return guided

            horizon = matched_information_horizon(
                float(time_value.detach().float().item()),
                clock=self.query_clock,
                anchor_time=self.clock_anchor_time,
                anchor_horizon=self.anchor_horizon,
                intervention_time=INTERVENTION_TIME,
            )
            if horizon <= 0.0:
                return guided

            weak_future, deeper_future = self._evaluate_future(
                time_value + horizon, state
            )
            self.query_nfe += 1
            projection = project_per_sample(
                weak - weak_future,
                deeper - deeper_future,
            )
            revision = (
                projection.parallel
                if self.condition == "weak_common_depth10"
                else projection.orthogonal
            )
            return guided + (1.0 + gamma) * revision
