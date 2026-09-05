"""Exact score geometry induced by reusing a linear-flow velocity in time.

For the affine interpolant ``Z_t = t X + (1-t) E``, a velocity and its
same-time marginal score obey

    v_t(z) = z/t + (1-t)/t * s_t(z).

If a velocity evaluated at a later time ``tau`` is reused at the current
time ``t``, its current-time implied score is not simply ``s_tau``.  It is

    R_{t<-tau}(s_tau) = a s_tau + (1-a) s_phi,

where ``s_phi(z)=-z`` is the standard Gaussian score and
``a = odds(t)/odds(tau)``.  Thus, for conservative scores, raw velocity
reuse is exactly the score of the exponential-geodesic density

    q_tau(z)^a phi(z)^(1-a) / Z.

The affine maps ``R`` compose as a semigroup.  These identities are purely
algebraic and do not assume that a finite neural field is conservative.
"""

from __future__ import annotations

import torch


Tensor = torch.Tensor


def _time_like(value: Tensor | float, reference: Tensor, *, name: str) -> Tensor:
    result = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if result.ndim == 0:
        return result
    if result.ndim != 1 or len(result) != len(reference):
        raise ValueError(f"{name} must be scalar or have one entry per sample")
    return result.reshape(-1, *((1,) * (reference.ndim - 1)))


def linear_velocity_to_score(
    velocity: Tensor,
    state: Tensor,
    time: Tensor | float,
) -> Tensor:
    """Convert affine-flow velocity to its same-time implied score."""

    if velocity.shape != state.shape:
        raise ValueError("velocity and state must have identical shapes")
    time_value = _time_like(time, state, name="time")
    if torch.any(time_value <= 0.0) or torch.any(time_value >= 1.0):
        raise ValueError("time must lie strictly inside (0, 1)")
    return (time_value * velocity - state) / (1.0 - time_value)


def score_to_linear_velocity(
    score: Tensor,
    state: Tensor,
    time: Tensor | float,
) -> Tensor:
    """Convert a marginal score to affine-flow velocity."""

    if score.shape != state.shape:
        raise ValueError("score and state must have identical shapes")
    time_value = _time_like(time, state, name="time")
    if torch.any(time_value <= 0.0) or torch.any(time_value >= 1.0):
        raise ValueError("time must lie strictly inside (0, 1)")
    return (state + (1.0 - time_value) * score) / time_value


def retiming_weight(
    time: Tensor | float,
    future_time: Tensor | float,
    reference: Tensor,
) -> Tensor:
    """Return ``odds(time) / odds(future_time)`` with broadcast shape."""

    current = _time_like(time, reference, name="time")
    future = _time_like(future_time, reference, name="future_time")
    if torch.any(current <= 0.0) or torch.any(future >= 1.0):
        raise ValueError("times must lie strictly inside (0, 1)")
    if torch.any(future <= current):
        raise ValueError("future_time must be greater than time")
    return current * (1.0 - future) / ((1.0 - current) * future)


def retime_future_score(
    future_score: Tensor,
    state: Tensor,
    time: Tensor | float,
    future_time: Tensor | float,
) -> Tensor:
    """Pull a future score back through the raw-velocity representation."""

    if future_score.shape != state.shape:
        raise ValueError("future_score and state must have identical shapes")
    weight = retiming_weight(time, future_time, state)
    return weight * future_score + (1.0 - weight) * (-state)


def reinterpret_future_velocity_score(
    future_velocity: Tensor,
    state: Tensor,
    time: Tensor | float,
) -> Tensor:
    """Interpret a future-time raw velocity as a current-time velocity."""

    return linear_velocity_to_score(future_velocity, state, time)


def exponential_retiming_defect(
    current_score: Tensor,
    future_score: Tensor,
    state: Tensor,
    time: Tensor | float,
    future_time: Tensor | float,
) -> Tensor:
    """Return current score minus its future-score e-geodesic prediction."""

    if current_score.shape != state.shape:
        raise ValueError("current_score and state must have identical shapes")
    return current_score - retime_future_score(
        future_score,
        state,
        time,
        future_time,
    )


def split_exponential_retiming_defect(
    current_score: Tensor,
    future_score: Tensor,
    state: Tensor,
    time: Tensor | float,
    future_time: Tensor | float,
) -> tuple[Tensor, Tensor]:
    """Split the defect into score evolution and Gaussian retiming terms.

    The returned terms satisfy the exact identity

        s_t - R(s_tau)
        = (s_t - s_tau) + (1-a) * (s_tau - s_phi).
    """

    if current_score.shape != future_score.shape or current_score.shape != state.shape:
        raise ValueError("scores and state must have identical shapes")
    weight = retiming_weight(time, future_time, state)
    score_evolution = current_score - future_score
    gaussian_retiming = (1.0 - weight) * (future_score + state)
    return score_evolution, gaussian_retiming


def compose_retiming_weights(
    early_time: Tensor | float,
    middle_time: Tensor | float,
    late_time: Tensor | float,
    reference: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return direct and two-hop weights for a semigroup identity audit."""

    direct = retiming_weight(early_time, late_time, reference)
    two_hop = retiming_weight(
        early_time, middle_time, reference
    ) * retiming_weight(middle_time, late_time, reference)
    return direct, two_hop
