"""Frozen-coefficient density decontamination ODE algebra and barrier solver.

This module only implements a candidate under the hypothesis
p_star = w*p_F - (w-1)*p_B >= 0. It does not establish that RAEv2 heads are
consistent densities, that this mixture hypothesis holds, or any FID gain.
All scalar integrations and root calculations use float64, without clipping
the density ratio or guidance coefficient. No sampler or model is included.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class FrozenDensityCoefficients:
    A: torch.Tensor
    Bcoef: torch.Tensor
    K: torch.Tensor
    inverse_cube_integral: torch.Tensor
    inverse_time_integral: torch.Tensor


@dataclass(frozen=True)
class ImplicitDensityStep:
    ell_new: torch.Tensor
    beta: torch.Tensor
    residual: torch.Tensor
    barrier_distance: torch.Tensor
    iterations: int


def _float64_broadcast(*values):
    devices = {value.device for value in values if isinstance(value, torch.Tensor)}
    if len(devices) > 1:
        raise ValueError("all tensor inputs must be on the same device")
    device = next(iter(devices), torch.device("cpu"))
    tensors = tuple(torch.as_tensor(value, device=device, dtype=torch.float64) for value in values)
    if any(not bool(torch.isfinite(value).all()) for value in tensors):
        raise ValueError("density inputs must be finite")
    return torch.broadcast_tensors(*tensors)


def _barrier(w: float) -> float:
    w = float(w)
    if not math.isfinite(w) or w < 1:
        raise ValueError("w must be finite and at least 1")
    # log1p preserves the barrier when w/(w-1) would round to exactly one.
    return math.inf if w == 1 else math.log1p(1.0/(w-1.0))


def log1mexp(value: torch.Tensor) -> torch.Tensor:
    """Stable log(1-exp(x)) for x <= 0, including its exact -inf at x=0."""
    value = torch.as_tensor(value, dtype=torch.float64)
    if bool(torch.isnan(value).any() or (value > 0).any()):
        raise ValueError("log1mexp requires nonpositive, non-NaN inputs")
    return torch.where(value < -math.log(2),
                       torch.log1p(-torch.exp(value)), torch.log(-torch.expm1(value)))


def _log_beta_from_distance(distance):
    return -distance-log1mexp(-distance)


def current_beta(ell, w: float = 1.78) -> torch.Tensor:
    """(w-1)*exp(ell)/(w-(w-1)*exp(ell)), without denominator clipping.

At w=1 guidance is exactly zero and there is no finite barrier. Near a finite
barrier, retain the implicit step's returned beta rather than recomputing it
from an ell whose subtraction from L may have lost relative precision.
"""
    limit = _barrier(w)
    (ell,) = _float64_broadcast(ell)
    if w == 1:
        return torch.zeros_like(ell)
    distance = limit-ell
    if bool((distance <= 0).any()):
        raise ValueError("ell must be strictly below the positivity barrier")
    beta = torch.exp(_log_beta_from_distance(distance))
    if not bool(torch.isfinite(beta).all()):
        raise FloatingPointError("the uncapped guidance coefficient is not representable in float64")
    return beta


def frozen_step_coefficients(divergence, dot_dz, dot_db, gap_squared_norm,
                             t, s) -> FrozenDensityCoefficients:
    """Integrate the frozen z,F,B,div(D) coefficients over t -> s > 0.

dot_dz=<F-B,z>, dot_db=<F-B,B>, and gap_squared_norm=||F-B||^2
are sums over every latent coordinate, not dimension-normalized means.
The s=0 endpoint is deliberately excluded: use the current beta there.
"""
    divergence, dot_dz, dot_db, gap_squared_norm, t, s = _float64_broadcast(
        divergence, dot_dz, dot_db, gap_squared_norm, t, s)
    if bool(((s <= 0) | (s >= t) | (t > 1)).any()):
        raise ValueError("density integration requires 0 < s < t <= 1; do not update density at s=0")
    if bool((gap_squared_norm < 0).any()):
        raise ValueError("gap_squared_norm must be nonnegative")
    # Factored formulas avoid subtracting nearly equal inverse powers.
    difference = t-s
    denominator = 2*s.square()*t.square()
    inverse_cube = difference*(t+s)/denominator
    K = difference*(t*(1-s)+s*(1-t))/denominator
    inverse_time = torch.log1p(difference/s)
    A = divergence*inverse_time-dot_dz*inverse_cube+dot_db*K
    Bcoef = gap_squared_norm*K
    if any(not bool(torch.isfinite(value).all()) for value in (A, Bcoef, K)):
        raise FloatingPointError("frozen density integrals overflowed float64")
    return FrozenDensityCoefficients(A, Bcoef, K, inverse_cube, inverse_time)


def frozen_coefficients_from_fields(state, full, base, divergence, t, s):
    """Compute the frozen dot products in FP64 from matching [batch,...] fields."""
    if not all(isinstance(value, torch.Tensor) for value in (state, full, base)):
        raise TypeError("state/full/base must be tensors")
    if state.ndim < 2 or state.shape != full.shape or state.shape != base.shape:
        raise ValueError("state/full/base must share a [batch,...] shape")
    if len({value.device for value in (state, full, base)}) != 1:
        raise ValueError("state/full/base must share a device")
    state, full, base = state.double(), full.double(), base.double()
    gap = full-base
    sum_coordinates = lambda value: value.flatten(1).sum(1)
    return frozen_step_coefficients(divergence, sum_coordinates(gap*state),
                                    sum_coordinates(gap*base), sum_coordinates(gap.square()), t, s)


@torch.no_grad()
def implicit_density_step(ell_old, A, Bcoef, w: float = 1.78) -> ImplicitDensityStep:
    """Solve ell_new+Bcoef*beta(ell_new)=ell_old+A in the positive-density domain.

For Bcoef>0 the left side strictly increases from -infinity to +infinity,
so the root is unique. Bisection uses log(delta), delta=L-ell, preserving
small barrier distances and large uncapped beta. It terminates at adjacent
float64 log-distance values (or an exact residual zero), not a manual cap.

ell_new is rounded to float64 while barrier_distance and beta retain their
stable coordinate. If the strict ell_new<L cannot be represented, raise;
never clamp/nextafter the value and pretend that it solves the equation.
"""
    limit = _barrier(w)
    ell_old, A, Bcoef = _float64_broadcast(ell_old, A, Bcoef)
    if bool((Bcoef < 0).any()):
        raise ValueError("Bcoef must be nonnegative")
    rhs = ell_old+A
    if not bool(torch.isfinite(rhs).all()):
        raise FloatingPointError("implicit density right hand side overflowed float64")
    if w == 1:
        return ImplicitDensityStep(rhs, torch.zeros_like(rhs), torch.zeros_like(rhs),
                                   torch.full_like(rhs, math.inf), 0)
    if bool((ell_old >= limit).any()):
        raise ValueError("ell_old must be strictly below the positivity barrier")
    positive = Bcoef > 0
    if bool((~positive & (rhs >= limit)).any()):
        raise ValueError("Bcoef=0 and rhs>=L has no admissible root; clipping is forbidden")
    ell_new = rhs.clone()
    beta = current_beta(torch.where(positive, ell_old, rhs), w)
    distance = limit-rhs
    residual = torch.zeros_like(rhs)
    if not bool(positive.any()):
        return ImplicitDensityStep(ell_new, beta, residual, distance, 0)

    # g(delta)=C-delta+B/(exp(delta)-1) strictly decreases. Analytic brackets:
    # B/(B+|C|+sqrt(B)+1) <= delta_root <= max(C,0)+sqrt(B).
    # Their logarithms avoid overflowing the bracket arithmetic itself.
    B = Bcoef[positive]
    target = rhs[positive]
    C = limit-target
    logB = B.log()
    log_abs_C = C.abs().log()
    log_positive_C = torch.where(C > 0, C.log(), torch.full_like(C, -math.inf))
    lo = logB-torch.logsumexp(torch.stack((logB, log_abs_C, .5*logB, torch.zeros_like(B))), dim=0)
    hi = torch.logaddexp(log_positive_C, .5*logB)

    def evaluate(log_distance):
        delta = log_distance.exp()
        log_beta = _log_beta_from_distance(delta)
        product = (logB+log_beta).exp()
        return C-delta+product, delta, log_beta

    finished = torch.zeros_like(B, dtype=torch.bool)
    for iteration in range(1, 129):
        midpoint = lo+(hi-lo)*.5
        adjacent = (midpoint == lo) | (midpoint == hi)
        value, _, _ = evaluate(midpoint)
        exact = value == 0
        active = ~finished & ~adjacent
        lo = torch.where(active & (value > 0), midpoint, lo)
        hi = torch.where(active & (value < 0), midpoint, hi)
        lo = torch.where(active & exact, midpoint, lo)
        hi = torch.where(active & exact, midpoint, hi)
        finished |= adjacent | exact
        if bool(finished.all()):
            break
    else:
        raise ArithmeticError("barrier root did not reach adjacent float64 log-distance values")
    lo_value, _, _ = evaluate(lo)
    hi_value, _, _ = evaluate(hi)
    chosen = torch.where(lo_value.abs() <= hi_value.abs(), lo, hi)
    _, solved_distance, log_beta = evaluate(chosen)
    solved_beta = log_beta.exp()
    solved_ell = limit-solved_distance
    solved_residual = (solved_ell-target)+B*solved_beta
    if (not bool(torch.isfinite(solved_ell).all() & torch.isfinite(solved_beta).all()
                 & torch.isfinite(solved_residual).all()) or bool((solved_ell >= limit).any())):
        raise FloatingPointError("the uncapped root is not representable with finite float64 ell<L and beta")
    scale = 1+target.abs()+solved_distance+B*solved_beta
    if bool((solved_residual.abs() > 64*torch.finfo(torch.float64).eps*scale).any()):
        raise ArithmeticError("barrier root residual exceeds float64 rounding accuracy")
    ell_new[positive], beta[positive] = solved_ell, solved_beta
    distance[positive], residual[positive] = solved_distance, solved_residual
    return ImplicitDensityStep(ell_new, beta, residual, distance, iteration)
