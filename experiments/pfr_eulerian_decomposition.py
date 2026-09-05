"""Finite Eulerian/material decomposition of the PFR time query.

At a state ``(z, t)``, let ``W`` be the weak velocity and ``G`` the ordinary
guided velocity. For a finite horizon ``h`` define

    d_eulerian = W(z, t + h) - W(z, t)
    d_material  = W(z + h G, t + h) - W(z, t)
    d_frame     = W(z, t + h) - W(z + h G, t + h).

The identity ``d_eulerian = d_material + d_frame`` is exact. In the smooth
small-h limit the first two terms approach ``h partial_t W`` and
``h (partial_t + G dot grad) W``; the remainder approaches
``-h (G dot grad) W``. Sampling the terms separately therefore tests whether
PFR benefits from a Lagrangian future estimate or specifically from holding
the latent coordinate fixed while advancing information time.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from experiments.pfr_information_clock import matched_information_horizon
from experiments.run_imagenet100_sit_path_evidence_pfr_bridge import (
    HORIZON,
    INTERVENTION_TIME,
    gamma_at,
)


Tensor = torch.Tensor
DECOMPOSITION_KINDS = ("time_only", "material_guided", "frame_guided")


@dataclass(frozen=True)
class FiniteEulerianComponents:
    eulerian: Tensor
    material: Tensor
    frame: Tensor


def finite_eulerian_components(
    weak_now: Tensor,
    weak_time: Tensor,
    weak_material: Tensor,
) -> FiniteEulerianComponents:
    """Return the exact finite secants used by the causal decomposition."""

    if not (weak_now.shape == weak_time.shape == weak_material.shape):
        raise ValueError("weak outputs must have identical shapes")
    material = weak_material - weak_now
    frame = weak_time - weak_material
    return FiniteEulerianComponents(
        eulerian=weak_time - weak_now,
        material=material,
        frame=frame,
    )


class EulerianDecompositionField:
    """Ordinary IG plus one selected finite weak-response component."""

    def __init__(
        self,
        runtime: Any,
        labels: Tensor,
        condition: str,
        *,
        query_clock: str = "raw_t",
        clock_anchor_time: float = 0.25,
        anchor_horizon: float = HORIZON,
        revision_scale: float = 1.0,
    ) -> None:
        if condition not in DECOMPOSITION_KINDS:
            raise ValueError(f"unknown decomposition condition: {condition}")
        self.runtime = runtime
        self.labels = labels
        self.condition = condition
        self.query_clock = query_clock
        self.clock_anchor_time = float(clock_anchor_time)
        self.anchor_horizon = float(anchor_horizon)
        self.revision_scale = float(revision_scale)
        if not math.isfinite(self.anchor_horizon) or self.anchor_horizon <= 0.0:
            raise ValueError("anchor_horizon must be positive and finite")
        if not math.isfinite(self.revision_scale):
            raise ValueError("revision_scale must be finite")
        self.nfe = 0
        self.query_nfe = 0

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

            query_time = time_value + horizon
            weak_time = self.runtime.evaluate_weak(query_time, state, self.labels)
            self.query_nfe += 1
            if self.condition == "time_only":
                revision = weak_time - weak
            else:
                weak_material = self.runtime.evaluate_weak(
                    query_time,
                    state + horizon * guided,
                    self.labels,
                )
                self.query_nfe += 1
                components = finite_eulerian_components(
                    weak, weak_time, weak_material
                )
                revision = (
                    components.material
                    if self.condition == "material_guided"
                    else components.frame
                )
            return guided - (1.0 + gamma) * self.revision_scale * revision
