"""Counterfactual information-revision probes for Internal Guidance.

This module keeps the algebra deliberately small.  At a current query ``p``
and a counterfactual query ``q`` it distinguishes the weak and strong changes

    I_w = W_q - W_p,    I_s = S_q - S_p,

instead of assuming that the weak change alone is a nuisance estimate.  The
decomposition is useful both for auditing that assumption and for constructing
an exact scalar sweep between ordinary IG and the deployed PFR controller.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from experiments.path_evidence_pfr_bridge import (
    SampleProjection,
    project_to_forward_ray,
)


Tensor = torch.Tensor


@dataclass(frozen=True)
class InformationQuery:
    state: Tensor
    time: Tensor
    projection: SampleProjection
    horizon: float


@dataclass(frozen=True)
class FourCornerRevision:
    strong_now: Tensor
    weak_now: Tensor
    strong_query: Tensor
    weak_query: Tensor
    gap_now: Tensor
    gap_query: Tensor
    weak_revision: Tensor
    strong_revision: Tensor
    cross_corner_gap: Tensor
    interaction_revision: Tensor


def projected_information_query(
    state: Tensor,
    time_value: Tensor,
    *,
    strong_now: Tensor,
    weak_now: Tensor,
    guided_now: Tensor,
    gamma: float,
    horizon: float,
    intervention_time: float,
) -> InformationQuery:
    """Construct the established minimum-spatial-intervention PFR query."""

    if not (
        state.shape == strong_now.shape == weak_now.shape == guided_now.shape
    ):
        raise ValueError("state and field tensors must have identical shapes")
    if horizon <= 0.0:
        raise ValueError("horizon must be positive")
    scalar_time = float(time_value.detach().float().item())
    step = min(float(horizon), float(intervention_time) - scalar_time)
    if step <= 0.0 or float(gamma) == 0.0:
        zeros = torch.zeros_like(state)
        coefficients = torch.zeros(len(state), device=state.device)
        projection = SampleProjection(
            parallel=zeros,
            orthogonal=zeros,
            coefficient=coefficients,
        )
        return InformationQuery(
            state=state,
            time=time_value,
            projection=projection,
            horizon=0.0,
        )

    beta = 1.0 + float(gamma)
    calibration = beta * (strong_now - weak_now)
    projection = project_to_forward_ray(calibration, guided_now)
    return InformationQuery(
        state=state + step * projection.parallel,
        time=time_value + time_value.new_tensor(step),
        projection=projection,
        horizon=step,
    )


def four_corner_revision(
    strong_now: Tensor,
    weak_now: Tensor,
    strong_query: Tensor,
    weak_query: Tensor,
) -> FourCornerRevision:
    """Return the exact 2x2 depth-by-query decomposition."""

    shapes = {
        strong_now.shape,
        weak_now.shape,
        strong_query.shape,
        weak_query.shape,
    }
    if len(shapes) != 1:
        raise ValueError("all four fields must have identical shapes")
    gap_now = strong_now - weak_now
    gap_query = strong_query - weak_query
    weak_revision = weak_query - weak_now
    strong_revision = strong_query - strong_now
    return FourCornerRevision(
        strong_now=strong_now,
        weak_now=weak_now,
        strong_query=strong_query,
        weak_query=weak_query,
        gap_now=gap_now,
        gap_query=gap_query,
        weak_revision=weak_revision,
        strong_revision=strong_revision,
        cross_corner_gap=strong_now - weak_query,
        interaction_revision=weak_revision - strong_revision,
    )


def lambda_residualized_guidance(
    guided_now: Tensor,
    weak_revision: Tensor,
    *,
    beta: float,
    residualization: float,
) -> Tensor:
    """Scale only the PFR weak-reference revision.

    ``residualization=0`` is ordinary IG and ``residualization=1`` is exactly
    the deployed PFR field:

        G_lambda = G - beta * lambda * (W_q - W_p).
    """

    if guided_now.shape != weak_revision.shape:
        raise ValueError("guided field and weak revision must have identical shapes")
    beta = float(beta)
    residualization = float(residualization)
    if not math.isfinite(beta) or beta < 1.0:
        raise ValueError("beta must be finite and at least one")
    if not math.isfinite(residualization):
        raise ValueError("residualization must be finite")
    return guided_now - beta * residualization * weak_revision


def interaction_residualized_guidance(
    guided_now: Tensor,
    parts: FourCornerRevision,
    *,
    beta: float,
) -> Tensor:
    """Remove only the depth-by-query interaction from the current gap."""

    return guided_now - float(beta) * parts.interaction_revision
