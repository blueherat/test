"""Component-wise exponential quadrature of RAEv2's guided clean prediction.

This is an ablation of temporal quadrature, not a claim that a new score or
new density has been constructed. Full correction is the data-prediction
second-order exponential multistep formula (DPM-Solver++ 2M, Heun variant).
The split arms interpolate only one of Full and native-guidance increment.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

MODES = ("official", "exponential_2m", "guidance_2m", "full_2m")


def log_snr(t: float) -> float:
    if not 0.0 < t < 1.0:
        raise ValueError("log SNR requires an interior bridge time")
    return math.log1p(-t) - math.log(t)


def extrapolation_weight(previous: float | None, current: float, following: float) -> float:
    """Exactly integrate a clean prediction linear in log((1-t)/t).

Bootstrap, final clean endpoint, and the original hard guidance event use
the original Euler update. We do not extrapolate through infinite log SNR
or across a discontinuous guidance window. No tuned cutoff is introduced.
"""
    if previous is None or not 0.0 < following < current < previous < 1.0:
        return 0.0
    if not (following >= 0.1) == (current >= 0.1) == (previous >= 0.1):
        return 0.0
    h = log_snr(following) - log_snr(current)
    old_h = log_snr(current) - log_snr(previous)
    # h/(1-exp(-h))-1 = h/2+h^2/12-h^4/720+...
    numerator = h / 2 + h * h / 12 - h ** 4 / 720 if h < 1e-3 else h / -math.expm1(-h) - 1.0
    return numerator / old_h


def native_clean(full: Tensor, base: Tensor, time: float) -> Tensor:
    """Retain the official three BF16 operations before FP32 conversion."""
    return (base + 1.78 * (full - base) if time >= 0.1 else full).float()


def selected_component(full: Tensor, clean: Tensor, mode: str) -> Tensor:
    if mode not in MODES:
        raise ValueError(mode)
    if mode == "full_2m":
        return full.float()
    if mode == "guidance_2m":
        # Includes the native rounding residual, so components sum to the
        # actual deployed G rather than a differently rounded approximation.
        return clean - full.float()
    return clean


def step(state: Tensor, clean: Tensor, component: Tensor, previous_component: Tensor | None,
         previous_time: float | None, current: float, following: float, *,
         mode: str, denominator_floor: float = 0.05) -> tuple[Tensor, float]:
    if mode not in MODES or not 0 <= following < current <= 1:
        raise ValueError("invalid method or dataward time interval")
    weight = 0.0 if mode == "official" else extrapolation_weight(previous_time, current, following)
    if weight and previous_component is None:
        raise ValueError("quadrature requires the previous prediction")
    effective = clean if not weight else clean + weight * (component - previous_component)
    return state - (current - following) * ((state - effective) / max(current, denominator_floor)), weight
