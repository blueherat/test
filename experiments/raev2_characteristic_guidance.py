"""One-step characteristic correction for RAEv2 Internal Guidance.

RAEv2 predicts the clean endpoint of the linear bridge

    z_t = (1 - t) * x + t * epsilon.

For a full predictor ``F`` and a base predictor ``B``, ordinary Internal
Guidance with complete coefficient ``beta`` is ``beta*F-(beta-1)*B``.
Characteristic Guidance replaces this same-state affine combination by two
shifted queries.  This module implements its first fixed-point iterate with
the theoretical identity projection and introduces no additional scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def _expand_batch(value: Tensor, reference: Tensor) -> Tensor:
    if value.shape != (reference.shape[0],):
        raise ValueError("time must contain one value per sample")
    return value.view(value.shape[0], *([1] * (reference.ndim - 1)))


@dataclass(frozen=True)
class CharacteristicQueries:
    """First-iterate displacement and the two characteristic query states."""

    displacement: Tensor
    full_query: Tensor
    base_query: Tensor


def first_characteristic_queries(
    state: Tensor,
    time: Tensor,
    full_clean: Tensor,
    base_clean: Tensor,
    *,
    guidance_scale: float,
) -> CharacteristicQueries:
    """Construct the first characteristic fixed-point iterate.

    Writing ``omega = beta - 1``, Characteristic Guidance queries the strong
    and weak predictors at ``z + omega*Delta`` and ``z + beta*Delta``.
    Starting its fixed-point solve from ``Delta=0`` gives exactly

        Delta_1 = (1 - t) * (F(z,t) - B(z,t))

    in RAEv2's clean-prediction coordinates.
    """

    if state.shape != full_clean.shape or state.shape != base_clean.shape:
        raise ValueError("state and clean predictions must have identical shapes")
    if guidance_scale < 1.0:
        raise ValueError("characteristic extrapolation requires guidance_scale >= 1")
    signal = 1.0 - _expand_batch(time, state)
    displacement = signal * (full_clean - base_clean)
    omega = guidance_scale - 1.0
    return CharacteristicQueries(
        displacement=displacement,
        full_query=state + omega * displacement,
        base_query=state + guidance_scale * displacement,
    )


def characteristic_clean_from_shifted_predictions(
    shifted_full_clean: Tensor,
    shifted_base_clean: Tensor,
    *,
    guidance_scale: float,
) -> Tensor:
    """Combine shifted predictions with the original IG coefficient."""

    if shifted_full_clean.shape != shifted_base_clean.shape:
        raise ValueError("shifted predictions must have identical shapes")
    if guidance_scale < 1.0:
        raise ValueError("characteristic extrapolation requires guidance_scale >= 1")
    return (
        guidance_scale * shifted_full_clean
        - (guidance_scale - 1.0) * shifted_base_clean
    )


def evaluate_first_characteristic_clean(
    model: torch.nn.Module,
    state: Tensor,
    time: Tensor,
    labels: Tensor,
    full_clean: Tensor,
    base_clean: Tensor,
    *,
    guidance_scale: float,
) -> tuple[Tensor, CharacteristicQueries]:
    """Evaluate both shifted queries in one batched model call."""

    if labels.shape != (state.shape[0],):
        raise ValueError("labels must contain one value per sample")
    queries = first_characteristic_queries(
        state,
        time,
        full_clean,
        base_clean,
        guidance_scale=guidance_scale,
    )
    output = model(
        torch.cat((queries.full_query, queries.base_query), dim=0),
        torch.cat((time, time), dim=0),
        context=torch.cat((labels, labels), dim=0),
        attn_mask=None,
    )
    if not isinstance(output, tuple) or len(output) != 2:
        raise TypeError("RAEv2 Internal Guidance model must return (full, base)")
    all_full, all_base = output
    batch_size = state.shape[0]
    if all_full.shape[0] != 2 * batch_size or all_base.shape != all_full.shape:
        raise ValueError("shifted model outputs have unexpected shapes")
    shifted_full = all_full[:batch_size]
    shifted_base = all_base[batch_size:]
    clean = characteristic_clean_from_shifted_predictions(
        shifted_full,
        shifted_base,
        guidance_scale=guidance_scale,
    )
    return clean, queries
