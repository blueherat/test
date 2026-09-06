"""Gaussian-split full-score refresh for RAEv2's dataward Euler sampler.

This is a first-order stochastic-interpolant adaptation, not a new invariance
theorem.  The frozen nonlinear score residual is approximate at finite steps.
The Gaussian part is integrated exactly to avoid Euler noise/decay imbalance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class RefreshCoefficients:
    active_step: float
    reference_variance: float
    clock_increment: float
    decay: float
    residual_weight: float
    noise_std: float


def gaussian_bridge_variance(time: float) -> float:
    """Variance of (1-t) X + t E for independent standard Gaussian X, E."""
    return (1.0 - time) ** 2 + time**2


def refresh_coefficients(
    time: float,
    next_time: float,
    *,
    eta: float,
    interval: tuple[float, float] = (0.5, 0.95),
) -> RefreshCoefficients:
    """Return stable coefficients for diffusion a(t)=2 eta t/(1-t).

    Times decrease during generation.  Only the overlap with ``interval``
    contributes clock time, so crossing a boundary does not refresh a whole
    inactive step.  The infinitesimal diffusion is time dependent, not state
    dependent: no omitted Ito divergence correction is required.
    """
    lower, upper = interval
    if not all(math.isfinite(v) for v in (time, next_time, eta, lower, upper)):
        raise ValueError("times, eta, and interval must be finite")
    if not 0.0 <= next_time <= time <= 1.0:
        raise ValueError("expected 0 <= next_time <= time <= 1")
    if not 0.0 < lower < upper < 1.0:
        raise ValueError("refresh interval must lie strictly inside (0, 1)")
    if eta < 0.0:
        raise ValueError("eta must be nonnegative")

    variance = gaussian_bridge_variance(next_time)
    active_step = max(0.0, min(time, upper) - max(next_time, lower))
    if eta == 0.0 or active_step == 0.0:
        return RefreshCoefficients(active_step, variance, 0.0, 1.0, 0.0, 0.0)

    # A boundary-crossing step uses the first time inside the active interval.
    clock_time = min(time, upper)
    diffusion_half = eta * clock_time / (1.0 - clock_time)
    clock_increment = diffusion_half * active_step / variance
    one_minus_decay = -math.expm1(-clock_increment)
    decay = math.exp(-clock_increment)
    residual_weight = variance * one_minus_decay
    noise_std = math.sqrt(variance * -math.expm1(-2.0 * clock_increment))
    return RefreshCoefficients(
        active_step,
        variance,
        clock_increment,
        decay,
        residual_weight,
        noise_std,
    )


def full_score_residual(state: Tensor, full_clean: Tensor, time: float) -> Tensor:
    """Return s_full(z,t) + z/c(t), computed in float32.

    RAEv2 uses z_t=(1-t)x+t epsilon, hence
    s_full=((1-t) full_clean-z)/t^2.  This conversion must not use a
    clean-to-velocity denominator clamp; refresh is disabled near t=0.
    """
    if state.shape != full_clean.shape:
        raise ValueError("state and full_clean shapes must agree")
    if not math.isfinite(time) or not 0.0 < time <= 1.0:
        raise ValueError("score conversion requires 0 < time <= 1")
    value = state.float()
    score = ((1.0 - time) * full_clean.float() - value) / time**2
    return score + value / gaussian_bridge_variance(time)


def refresh_after_euler(
    state: Tensor,
    euler_state: Tensor,
    full_clean: Tensor,
    time: float,
    next_time: float,
    *,
    eta: float,
    noise: Tensor | None = None,
    mode: str = "balanced",
    interval: tuple[float, float] = (0.5, 0.95),
) -> Tensor:
    """Refresh an already-computed ordinary IG Euler update.

    ``state`` and ``full_clean`` are from the OLD time. ``euler_state`` is the
    unchanged existing Euler implementation, evaluated at ``next_time``.
    ``noise`` must come from a dedicated per-sample refresh RNG stream.

    The balanced step is
      z_new = decay*z_euler + c_next*(1-decay)*r_old
              + sqrt(c_next*(1-decay**2))*noise.

    ``score_only`` removes diffusion; ``noise_only`` removes score drift.
    They are deliberately non-invariant causal controls, not alternatives
    claimed to preserve the full model's distribution.  With eta=0 or outside
    the active interval this function returns the exact input euler_state
    object without consuming noise or changing arithmetic.
    """
    if mode not in {"balanced", "score_only", "noise_only", "none"}:
        raise ValueError(f"unknown refresh mode: {mode}")
    if state.shape != euler_state.shape or state.shape != full_clean.shape:
        raise ValueError("state, euler_state, and full_clean shapes must agree")
    coefficients = refresh_coefficients(
        time, next_time, eta=eta, interval=interval
    )
    if mode == "none" or coefficients.clock_increment == 0.0:
        return euler_state

    if mode in {"balanced", "noise_only"}:
        if noise is None or noise.shape != state.shape:
            raise ValueError("active diffusion requires a matching noise tensor")
        if noise.device != state.device:
            raise ValueError("noise and state devices must agree")

    result = euler_state.float()
    if mode in {"balanced", "score_only"}:
        residual = full_score_residual(state, full_clean, time)
        result = (
            coefficients.decay * result
            + coefficients.residual_weight * residual
        )
    if mode in {"balanced", "noise_only"}:
        result = result + coefficients.noise_std * noise.float()
    return result.to(dtype=euler_state.dtype)
