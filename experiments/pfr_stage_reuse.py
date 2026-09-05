"""One-query-per-step approximation to projected PFR under Heun sampling.

The ordinary strong/weak pair is still evaluated at both Heun stages.  Only
the extra depth-prefix counterfactual revision is evaluated at the beginning
of a step and reused at the corrector stage.  This isolates a practical way
to pay for PFR with fewer solver steps without freezing the primary model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from experiments.information_purification_ig import projected_information_query
from experiments.pfr_information_clock import matched_information_horizon
from experiments.run_imagenet100_sit_path_evidence_pfr_bridge import (
    HORIZON,
    INTERVENTION_TIME,
    gamma_at,
)


Tensor = torch.Tensor


@dataclass(frozen=True)
class StageStart:
    velocity: Tensor
    weak_revision: Tensor | None


@dataclass(frozen=True)
class StageReuseResult:
    endpoint: Tensor
    nfe: int


class StageReusedProjectedField:
    """Projected PFR whose finite weak revision is shared within one step."""

    def __init__(
        self,
        runtime: Any,
        labels: Tensor,
        *,
        query_clock: str = "raw_t",
        clock_anchor_time: float = 0.25,
    ) -> None:
        self.runtime = runtime
        self.labels = labels
        self.query_clock = query_clock
        self.clock_anchor_time = float(clock_anchor_time)
        self.nfe = 0
        self.query_nfe = 0

    def _ordinary_guided(
        self, time_value: Tensor, state: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, float]:
        self.nfe += 1
        strong, weak = self.runtime.evaluate_pair(time_value, state, self.labels)
        gamma = gamma_at(float(time_value.detach().float().item()))
        guided = strong + gamma * (strong - weak)
        return guided, strong, weak, gamma

    def evaluate_start(self, time_value: Tensor, state: Tensor) -> StageStart:
        with torch.inference_mode():
            guided, strong, weak, gamma = self._ordinary_guided(time_value, state)
            if gamma == 0.0:
                return StageStart(guided, None)
            horizon = matched_information_horizon(
                float(time_value.detach().float().item()),
                clock=self.query_clock,
                anchor_time=self.clock_anchor_time,
                anchor_horizon=HORIZON,
                intervention_time=INTERVENTION_TIME,
            )
            if horizon <= 0.0:
                return StageStart(guided, None)
            query = projected_information_query(
                state,
                time_value,
                strong_now=strong,
                weak_now=weak,
                guided_now=guided,
                gamma=gamma,
                horizon=horizon,
                intervention_time=INTERVENTION_TIME,
            )
            weak_query = self.runtime.evaluate_weak(
                query.time, query.state, self.labels
            )
            self.query_nfe += 1
            revision = weak_query - weak
            return StageStart(guided - (1.0 + gamma) * revision, revision)

    def evaluate_end(
        self,
        time_value: Tensor,
        state: Tensor,
        weak_revision: Tensor | None,
    ) -> Tensor:
        with torch.inference_mode():
            guided, _, _, gamma = self._ordinary_guided(time_value, state)
            if gamma == 0.0 or weak_revision is None:
                return guided
            return guided - (1.0 + gamma) * weak_revision


def integrate_stage_reused_heun(
    field: StageReusedProjectedField,
    initial: Tensor,
    times: Tensor,
) -> StageReuseResult:
    """Integrate one grid while sharing only the PFR query across stages."""

    if times.ndim != 1 or len(times) < 2:
        raise ValueError("times must be one-dimensional with at least two points")
    if not bool(torch.all(times[1:] > times[:-1])):
        raise ValueError("times must be strictly increasing")
    state = initial
    nfe = 0
    for index in range(len(times) - 1):
        time_value = times[index]
        next_time = times[index + 1]
        step = next_time - time_value
        start = field.evaluate_start(time_value, state)
        predictor = state + step * start.velocity
        end_velocity = field.evaluate_end(
            next_time, predictor, start.weak_revision
        )
        state = state + 0.5 * step * (start.velocity + end_velocity)
        nfe += 2
    return StageReuseResult(endpoint=state, nfe=nfe)
