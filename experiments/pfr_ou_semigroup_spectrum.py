"""OU-semigroup coordinates for linear-flow velocity fields.

The linear bridge

    Z_t = t X + (1 - t) E

becomes a variance-preserving Gaussian channel after the deterministic change
of variables

    Y_t = Z_t / c_t,
    c_t = sqrt(t^2 + (1 - t)^2),
    Y_t = alpha_t X + sqrt(1 - alpha_t^2) E.

Writing ``s_t = -log(alpha_t)``, the channel is the Ornstein--Uhlenbeck
semigroup.  Density ratios to the standard Gaussian have Hermite degree-k
components that contract by ``exp(-k s_t) = alpha_t^k``.  This module only
implements the exact coordinate changes and the corresponding degree-wise
retiming algebra; interpreting a finite neural score through a Hermite
linearization remains an empirical hypothesis.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


Tensor = torch.Tensor


@dataclass(frozen=True)
class OUBridgeCoordinates:
    scale: Tensor
    signal: Tensor
    noise: Tensor
    semigroup_time: Tensor


def _time_like(value: Tensor | float, reference: Tensor, *, name: str) -> Tensor:
    result = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if result.ndim == 0:
        return result
    if result.ndim != 1 or len(result) != len(reference):
        raise ValueError(f"{name} must be scalar or have one entry per sample")
    return result.reshape(-1, *((1,) * (reference.ndim - 1)))


def ou_bridge_coordinates(
    time: Tensor | float,
    reference: Tensor,
) -> OUBridgeCoordinates:
    """Return the exact VP/OU coordinates of a linear interpolation time."""

    time_value = _time_like(time, reference, name="time")
    if torch.any(time_value <= 0.0) or torch.any(time_value >= 1.0):
        raise ValueError("time must lie strictly inside (0, 1)")
    scale = torch.sqrt(time_value.square() + (1.0 - time_value).square())
    signal = time_value / scale
    noise = (1.0 - time_value) / scale
    return OUBridgeCoordinates(
        scale=scale,
        signal=signal,
        noise=noise,
        semigroup_time=-torch.log(signal),
    )


def state_to_ou(state: Tensor, time: Tensor | float) -> Tensor:
    """Normalize a raw linear-bridge state into its VP/OU coordinate."""

    return state / ou_bridge_coordinates(time, state).scale


def transport_state_at_fixed_ou_coordinate(
    state: Tensor,
    time: Tensor | float,
    query_time: Tensor | float,
) -> Tensor:
    """Represent the same normalized OU point at another bridge time."""

    time_value = _time_like(time, state, name="time")
    query_value = _time_like(query_time, state, name="query_time")
    if torch.any(time_value < 0.0) or torch.any(time_value >= 1.0):
        raise ValueError("time must lie in [0, 1)")
    if torch.any(query_value <= time_value) or torch.any(query_value >= 1.0):
        raise ValueError("query_time must lie strictly between time and 1")
    current_scale = torch.sqrt(
        time_value.square() + (1.0 - time_value).square()
    )
    query_scale = torch.sqrt(
        query_value.square() + (1.0 - query_value).square()
    )
    return state * (query_scale / current_scale)


def linear_velocity_to_ou_relative_score(
    velocity: Tensor,
    state: Tensor,
    time: Tensor | float,
) -> Tensor:
    """Convert raw bridge velocity to ``score(Y_t) - score(N(0,I))``."""

    if velocity.shape != state.shape:
        raise ValueError("velocity and state must have identical shapes")
    time_value = _time_like(time, state, name="time")
    coordinates = ou_bridge_coordinates(time, state)
    normalized_state = state / coordinates.scale
    normalized_score = (
        coordinates.scale * (time_value * velocity - state) / (1.0 - time_value)
    )
    return normalized_score + normalized_state


def ou_relative_score_to_linear_velocity(
    relative_score: Tensor,
    state: Tensor,
    time: Tensor | float,
) -> Tensor:
    """Invert :func:`linear_velocity_to_ou_relative_score` at one time."""

    if relative_score.shape != state.shape:
        raise ValueError("relative_score and state must have identical shapes")
    time_value = _time_like(time, state, name="time")
    coordinates = ou_bridge_coordinates(time, state)
    normalized_state = state / coordinates.scale
    normalized_score = relative_score - normalized_state
    raw_score = normalized_score / coordinates.scale
    return (state + (1.0 - time_value) * raw_score) / time_value


def ou_mode_retiming_weight(
    time: Tensor | float,
    future_time: Tensor | float,
    reference: Tensor,
    *,
    degree: float = 1.0,
) -> Tensor:
    """Pull a degree-wise OU mode from a cleaner future channel to now."""

    if degree <= 0.0:
        raise ValueError("degree must be positive")
    current = ou_bridge_coordinates(time, reference)
    future = ou_bridge_coordinates(future_time, reference)
    if torch.any(future.signal <= current.signal):
        raise ValueError("future_time must be greater than time")
    return (current.signal / future.signal).pow(float(degree))


def ou_mode_retiming_defect(
    current_relative_score: Tensor,
    future_relative_score: Tensor,
    time: Tensor | float,
    future_time: Tensor | float,
    *,
    degree: float = 1.0,
) -> Tensor:
    """Return current relative score minus a retimed future OU mode."""

    if current_relative_score.shape != future_relative_score.shape:
        raise ValueError("relative scores must have identical shapes")
    weight = ou_mode_retiming_weight(
        time,
        future_time,
        current_relative_score,
        degree=degree,
    )
    return current_relative_score - weight * future_relative_score


def ou_relative_score_delta_to_linear_velocity_delta(
    relative_score_delta: Tensor,
    state: Tensor,
    time: Tensor | float,
) -> Tensor:
    """Map a normalized relative-score difference to raw velocity units."""

    if relative_score_delta.shape != state.shape:
        raise ValueError("relative_score_delta and state must have identical shapes")
    time_value = _time_like(time, state, name="time")
    scale = ou_bridge_coordinates(time, state).scale
    return (1.0 - time_value) / (time_value * scale) * relative_score_delta


def ou_degree1_retiming_velocity_defect(
    current_velocity: Tensor,
    future_velocity: Tensor,
    state: Tensor,
    time: Tensor | float,
    future_time: Tensor | float,
) -> Tensor:
    """Return the degree-1-annihilating OU defect in raw velocity units.

    ``future_velocity`` must be evaluated at the same normalized OU point as
    ``state``. In raw bridge coordinates that point is
    ``state * c(future_time) / c(time)``.

    This expanded expression is algebraically identical to forming
    ``r_t - (alpha_t / alpha_tau) r_tau`` in relative-score coordinates and
    mapping it back to velocity. Unlike that direct expression, it remains
    finite at ``time == 0``, where the apparent ``0 / 0`` is removable.
    """

    if current_velocity.shape != state.shape or future_velocity.shape != state.shape:
        raise ValueError("velocities and state must have identical shapes")
    future_state = transport_state_at_fixed_ou_coordinate(
        state, time, future_time
    )
    return ou_relative_score_consistency_velocity_defect(
        current_velocity,
        future_velocity,
        state,
        future_state,
        time,
        future_time,
    )


def ou_degree_retiming_velocity_defect(
    current_velocity: Tensor,
    future_velocity: Tensor,
    state: Tensor,
    time: Tensor | float,
    future_time: Tensor | float,
    *,
    degree: float,
) -> Tensor:
    """Return a degree-wise OU defect directly in raw velocity units.

    The represented relative-score residual is
    ``r_t - (alpha_t / alpha_tau) ** degree * r_tau``. The cleaner velocity
    must be evaluated at the same normalized OU coordinate as ``state``.
    This expanded form remains finite at ``time == 0`` for ``degree >= 1``.
    """

    if degree < 1.0:
        raise ValueError("degree must be at least one for a finite t=0 limit")
    if current_velocity.shape != state.shape or future_velocity.shape != state.shape:
        raise ValueError("velocities and state must have identical shapes")
    if degree == 1.0:
        return ou_degree1_retiming_velocity_defect(
            current_velocity,
            future_velocity,
            state,
            time,
            future_time,
        )

    time_value = _time_like(time, state, name="time")
    future_value = _time_like(future_time, state, name="future_time")
    if torch.any(time_value < 0.0) or torch.any(time_value >= 1.0):
        raise ValueError("time must lie in [0, 1)")
    if torch.any(future_value <= time_value) or torch.any(future_value >= 1.0):
        raise ValueError("future_time must lie strictly between time and 1")

    current_scale_sq = time_value.square() + (1.0 - time_value).square()
    current_scale = torch.sqrt(current_scale_sq)
    future_scale_sq = future_value.square() + (1.0 - future_value).square()
    future_scale = torch.sqrt(future_scale_sq)
    future_state = state * (future_scale / current_scale)

    current_term = current_velocity + (
        (1.0 - 2.0 * time_value) / current_scale_sq
    ) * state
    future_term = future_velocity + (
        (1.0 - 2.0 * future_value) / future_scale_sq
    ) * future_state
    future_multiplier = (
        (1.0 - time_value)
        / (1.0 - future_value)
        * time_value.pow(float(degree) - 1.0)
        * future_scale.pow(float(degree) + 1.0)
        / (
            future_value.pow(float(degree) - 1.0)
            * current_scale.pow(float(degree) + 1.0)
        )
    )
    return current_term - future_multiplier * future_term


def ou_future_posterior_mean_state(
    current_velocity: Tensor,
    state: Tensor,
    time: Tensor | float,
    future_time: Tensor | float,
) -> Tensor:
    """Estimate the cleaner raw state posterior mean from the current score.

    Let ``Y_t = Z_t / c_t`` and let ``a = alpha_t / alpha_tau`` for
    ``tau > t``. The exact OU Tweedie identity gives

    ``E[Y_tau | Y_t=y] = a*y + (1-a^2)/a * r_t(y)``,

    where ``r_t`` is the score relative to the standard Gaussian. The formula
    is expanded below so its removable limit at ``time == 0`` is finite.
    """

    if current_velocity.shape != state.shape:
        raise ValueError("current_velocity and state must have identical shapes")
    time_value = _time_like(time, state, name="time")
    future_value = _time_like(future_time, state, name="future_time")
    if torch.any(time_value < 0.0) or torch.any(time_value >= 1.0):
        raise ValueError("time must lie in [0, 1)")
    if torch.any(future_value <= time_value) or torch.any(future_value >= 1.0):
        raise ValueError("future_time must lie strictly between time and 1")

    current_scale_sq = time_value.square() + (1.0 - time_value).square()
    current_scale = torch.sqrt(current_scale_sq)
    future_scale = torch.sqrt(
        future_value.square() + (1.0 - future_value).square()
    )
    channel_signal = (
        time_value * future_scale / (current_scale * future_value)
    )
    normalized_mean = channel_signal * (state / current_scale)
    bracket = current_scale * current_velocity + (
        (1.0 - 2.0 * time_value) / current_scale
    ) * state
    stable_tweedie_factor = (
        (1.0 - channel_signal.square())
        * current_scale
        * future_value
        / (future_scale * (1.0 - time_value))
    )
    normalized_mean = normalized_mean + stable_tweedie_factor * bracket
    return future_scale * normalized_mean


def ou_relative_score_consistency_velocity_defect(
    current_velocity: Tensor,
    future_velocity: Tensor,
    state: Tensor,
    future_state: Tensor,
    time: Tensor | float,
    future_time: Tensor | float,
) -> Tensor:
    """Return ``r_t-a*r_tau`` in current raw-velocity units.

    Here ``a=alpha_t/alpha_tau`` and ``future_velocity`` is evaluated at the
    supplied ``future_state``. This is the stable raw-coordinate form of the
    OU relative-score consistency residual and is finite at ``time == 0``.
    """

    if not (
        current_velocity.shape
        == future_velocity.shape
        == state.shape
        == future_state.shape
    ):
        raise ValueError("velocities and states must have identical shapes")
    time_value = _time_like(time, state, name="time")
    future_value = _time_like(future_time, state, name="future_time")
    if torch.any(time_value < 0.0) or torch.any(time_value >= 1.0):
        raise ValueError("time must lie in [0, 1)")
    if torch.any(future_value <= time_value) or torch.any(future_value >= 1.0):
        raise ValueError("future_time must lie strictly between time and 1")

    current_scale_sq = time_value.square() + (1.0 - time_value).square()
    current_scale = torch.sqrt(current_scale_sq)
    future_scale = torch.sqrt(
        future_value.square() + (1.0 - future_value).square()
    )
    current_term = current_velocity + (
        (1.0 - 2.0 * time_value) / current_scale_sq
    ) * state
    future_multiplier = (
        (1.0 - time_value)
        * future_scale
        / (current_scale_sq * (1.0 - future_value))
    )
    future_term = future_scale * future_velocity + (
        (1.0 - 2.0 * future_value) / future_scale
    ) * future_state
    return current_term - future_multiplier * future_term
