"""Matrix-free posterior-response operators.

For z = alpha x + sigma epsilon and the Bayes clean estimator
m(z) = E[x | z], generalized Tweedie's formula gives

    alpha J_m(z) = alpha^2 Cov(x | z) / sigma^2.

Near a smooth data manifold in the low-noise limit this response approaches
the local tangent projector.  At finite noise it is generally a soft,
state-dependent response operator rather than an idempotent projector.  The
routines below compute only P a, which is all downstream methods need, and
therefore avoid constructing a dense ambient-dimensional Jacobian or choosing
an intrinsic rank.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


CleanEstimator = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def posterior_response_blend(
    anchor_velocity: torch.Tensor,
    other_velocity: torch.Tensor,
    response_action: torch.Tensor,
    *,
    strength: float = 1.0,
) -> torch.Tensor:
    """Blend two fields using ``P(anchor-other)`` without materializing P.

    The geometry-selected field is

        other + P(anchor - other).

    For a hard projector this takes the anchor field in the selected subspace
    and the other field in its orthogonal complement. ``strength`` interpolates
    from the anchor field (zero) to this geometry-selected field (one), which is
    useful for paired inference controls.
    """
    if (
        anchor_velocity.shape != other_velocity.shape
        or anchor_velocity.shape != response_action.shape
    ):
        raise ValueError("all velocity tensors must have identical shapes")
    selected = other_velocity + response_action
    if strength == 0.0:
        return anchor_velocity
    if strength == 1.0:
        return selected
    return anchor_velocity + float(strength) * (selected - anchor_velocity)


def relative_direction_step(
    state: torch.Tensor,
    direction: torch.Tensor,
    *,
    relative_step: float,
    state_rms_floor: float = 0.1,
    direction_rms_floor: float = 1e-8,
) -> torch.Tensor:
    """Return per-sample c so c*direction has controlled RMS."""
    if state.shape != direction.shape or state.ndim < 2:
        raise ValueError("state and direction must have matching [B,...] shapes")
    if relative_step <= 0:
        raise ValueError("relative_step must be positive")
    reduce_dims = tuple(range(1, state.ndim))
    state_rms = (
        state.float().square().mean(dim=reduce_dims).sqrt().clamp_min(state_rms_floor)
    )
    direction_rms = (
        direction.float()
        .square()
        .mean(dim=reduce_dims)
        .sqrt()
        .clamp_min(direction_rms_floor)
    )
    return relative_step * state_rms / direction_rms


@torch.no_grad()
def posterior_response_action(
    clean_estimator: CleanEstimator,
    *,
    state: torch.Tensor,
    time: torch.Tensor,
    direction: torch.Tensor,
    alpha: torch.Tensor,
    relative_step: float = 1e-2,
    central: bool = True,
) -> torch.Tensor:
    """Estimate alpha J_m(state,time) direction using model forwards only."""
    if state.shape != direction.shape or state.ndim < 2:
        raise ValueError("state and direction must have matching [B,...] shapes")
    if time.ndim != 1 or alpha.ndim != 1 or len(time) != len(state) or len(alpha) != len(state):
        raise ValueError("time and alpha must be [B]")
    step = relative_direction_step(
        state, direction, relative_step=relative_step
    )
    broadcast_shape = (len(step),) + (1,) * (state.ndim - 1)
    scaled_step = step.reshape(broadcast_shape).to(direction.dtype)
    delta = scaled_step * direction
    if central:
        plus = clean_estimator(state + delta, time)
        minus = clean_estimator(state - delta, time)
        derivative = (plus - minus) / (2.0 * scaled_step)
    else:
        base = clean_estimator(state, time)
        plus = clean_estimator(state + delta, time)
        derivative = (plus - base) / scaled_step
    scaled_alpha = alpha.reshape(broadcast_shape).to(derivative.dtype)
    return scaled_alpha * derivative


@torch.no_grad()
def posterior_response_basis(
    clean_estimator: CleanEstimator,
    *,
    state: torch.Tensor,
    time: torch.Tensor,
    alpha: torch.Tensor,
    probes: int,
    rank: int,
    relative_step: float = 1e-2,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomized range estimate of the leading response subspace.

    Returns a basis [B,D,rank] and singular values [B,probes].  This routine is
    intended for audits.  Practical methods should prefer direct action on the
    one vector they need, avoiding rank selection and extra probe forwards.
    """
    ambient_shape = state.shape[1:]
    D = int(state[0].numel())
    if not (1 <= rank <= probes <= D):
        raise ValueError("require 1 <= rank <= probes <= D")
    batch = len(state)
    random_directions = torch.randn(
        batch,
        probes,
        *ambient_shape,
        device=state.device,
        dtype=state.dtype,
        generator=generator,
    )
    random_directions = random_directions / torch.linalg.vector_norm(
        random_directions.float(),
        dim=tuple(range(2, random_directions.ndim)),
        keepdim=True,
    ).clamp_min(1e-12).to(random_directions.dtype)
    flat_state = state[:, None].expand(-1, probes, *([-1] * len(ambient_shape))).reshape(
        batch * probes, *ambient_shape
    )
    flat_time = time[:, None].expand(-1, probes).reshape(batch * probes)
    flat_alpha = alpha[:, None].expand(-1, probes).reshape(batch * probes)
    responses = posterior_response_action(
        clean_estimator,
        state=flat_state,
        time=flat_time,
        direction=random_directions.reshape(batch * probes, *ambient_shape),
        alpha=flat_alpha,
        relative_step=relative_step,
    ).reshape(batch, probes, D)
    _left, singular_values, right_t = torch.linalg.svd(
        responses.float(), full_matrices=False
    )
    basis = right_t[:, :rank].transpose(1, 2).to(state.dtype)
    return basis, singular_values
