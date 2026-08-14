"""Numerical tools for finite-strength guidance mechanism diagnostics.

The routines in this module deliberately use a fixed time grid.  This keeps
all guidance conditions on the same numerical budget and makes paired endpoint
differences interpretable without adaptive-solver NFE as a confound.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


Tensor = torch.Tensor
AnchorField = Callable[[Tensor, Tensor], Tensor]
DirectionField = Callable[[Tensor, Tensor, Tensor], Tensor]


def _validate_time_grid(time_grid: Tensor) -> None:
    if time_grid.ndim != 1 or len(time_grid) < 2:
        raise ValueError("time_grid must be one-dimensional with at least two points")
    if not bool(torch.all(time_grid[1:] > time_grid[:-1])):
        raise ValueError("time_grid must be strictly increasing")


def _expand_per_condition(values: Tensor, batch_size: int, ndim: int) -> Tensor:
    expanded = values.repeat_interleave(batch_size)
    return expanded.reshape(len(expanded), *([1] * (ndim - 1)))


def _repeat_conditions(value: Tensor, condition_count: int) -> Tensor:
    return value.unsqueeze(0).expand(condition_count, *value.shape).reshape(
        condition_count * len(value), *value.shape[1:]
    )


def _restore_conditions(value: Tensor, condition_count: int, batch_size: int) -> Tensor:
    return value.reshape(condition_count, batch_size, *value.shape[1:])


def _heun_step(
    state: Tensor,
    time_value: Tensor,
    next_time: Tensor,
    field: Callable[[Tensor, Tensor], Tensor],
) -> Tensor:
    step = next_time - time_value
    derivative = field(time_value, state)
    predicted = state + step * derivative
    corrected = field(next_time, predicted)
    return state + 0.5 * step * (derivative + corrected)


def integrate_heun(
    field: Callable[[Tensor, Tensor], Tensor],
    initial_state: Tensor,
    time_grid: Tensor,
) -> Tensor:
    """Integrate one field with explicit Heun steps on ``time_grid``."""

    _validate_time_grid(time_grid)
    state = initial_state.float()
    for time_value, next_time in zip(time_grid[:-1], time_grid[1:], strict=True):
        state = _heun_step(state, time_value, next_time, field)
    return state


def integrate_guidance_sweep(
    anchor_field: AnchorField,
    direction_field: DirectionField,
    initial_state: Tensor,
    time_grid: Tensor,
    gammas: Tensor,
) -> Tensor:
    """Integrate ``anchor + gamma * direction`` for paired gamma conditions.

    Conditions are folded into the batch dimension so every gamma is evaluated
    in the same model calls and on an identical fixed time grid.
    """

    _validate_time_grid(time_grid)
    if gammas.ndim != 1 or len(gammas) == 0:
        raise ValueError("gammas must be a non-empty one-dimensional tensor")
    condition_count = len(gammas)
    batch_size = len(initial_state)
    state = _repeat_conditions(initial_state.float(), condition_count)
    scales = _expand_per_condition(
        gammas.to(device=state.device, dtype=state.dtype),
        batch_size,
        state.ndim,
    )

    def guided_field(time_value: Tensor, current_state: Tensor) -> Tensor:
        anchor = anchor_field(time_value, current_state)
        direction = direction_field(time_value, current_state, anchor)
        return anchor + scales * direction

    endpoint = integrate_heun(guided_field, state, time_grid)
    return _restore_conditions(endpoint, condition_count, batch_size)


def _anchor_primal_and_jvp(
    anchor_field: AnchorField,
    time_value: Tensor,
    state: Tensor,
    tangent: Tensor,
) -> tuple[Tensor, Tensor]:
    def at_state(current_state: Tensor) -> Tensor:
        return anchor_field(time_value, current_state)

    primal, jacobian_vector = torch.func.jvp(
        at_state,
        (state,),
        (tangent,),
        strict=True,
    )
    return primal, jacobian_vector


def integrate_baseline_tangent(
    anchor_field: AnchorField,
    direction_field: DirectionField,
    initial_state: Tensor,
    time_grid: Tensor,
) -> tuple[Tensor, Tensor]:
    """Integrate the baseline and its exact ``gamma=0`` variational tangent.

    For ``dz_gamma/dt = v(z_gamma,t) + gamma*u(z_gamma,t)``, the tangent
    ``xi = d z_gamma / d gamma | gamma=0`` obeys
    ``dxi/dt = J_v(z,t) xi + u(z,t)`` with ``xi(0)=0``.
    """

    _validate_time_grid(time_grid)
    state = initial_state.float()
    tangent = torch.zeros_like(state)
    for time_value, next_time in zip(time_grid[:-1], time_grid[1:], strict=True):
        step = next_time - time_value
        anchor, anchor_jvp = _anchor_primal_and_jvp(
            anchor_field,
            time_value,
            state,
            tangent,
        )
        direction = direction_field(time_value, state, anchor)
        tangent_derivative = anchor_jvp + direction

        predicted_state = state + step * anchor
        predicted_tangent = tangent + step * tangent_derivative
        next_anchor, next_anchor_jvp = _anchor_primal_and_jvp(
            anchor_field,
            next_time,
            predicted_state,
            predicted_tangent,
        )
        next_direction = direction_field(next_time, predicted_state, next_anchor)
        next_tangent_derivative = next_anchor_jvp + next_direction

        state = state + 0.5 * step * (anchor + next_anchor)
        tangent = tangent + 0.5 * step * (
            tangent_derivative + next_tangent_derivative
        )
    return state, tangent


def integrate_baseline_tangent_frozen(
    anchor_field: AnchorField,
    direction_field: DirectionField,
    initial_state: Tensor,
    time_grid: Tensor,
    *,
    gamma: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Jointly integrate the baseline, its tangent, and exact frozen guidance.

    The frozen branch follows ``v(z_f,t) + gamma*u(z_b,t)`` while the tangent
    obeys ``d xi / dt = J_v(z_b,t) xi + u(z_b,t)``. Sharing the baseline
    evaluations makes this cheaper than running the two diagnostics separately
    without changing either numerical update.
    """

    _validate_time_grid(time_grid)
    if not torch.isfinite(torch.tensor(float(gamma))):
        raise ValueError("gamma must be finite")
    state = initial_state.float()
    tangent = torch.zeros_like(state)
    frozen = state.clone()
    scale = float(gamma)
    for time_value, next_time in zip(time_grid[:-1], time_grid[1:], strict=True):
        step = next_time - time_value
        anchor, anchor_jvp = _anchor_primal_and_jvp(
            anchor_field,
            time_value,
            state,
            tangent,
        )
        direction = direction_field(time_value, state, anchor)
        tangent_derivative = anchor_jvp + direction
        frozen_anchor = anchor_field(time_value, frozen)
        frozen_derivative = frozen_anchor + scale * direction

        predicted_state = state + step * anchor
        predicted_tangent = tangent + step * tangent_derivative
        predicted_frozen = frozen + step * frozen_derivative
        next_anchor, next_anchor_jvp = _anchor_primal_and_jvp(
            anchor_field,
            next_time,
            predicted_state,
            predicted_tangent,
        )
        next_direction = direction_field(next_time, predicted_state, next_anchor)
        next_tangent_derivative = next_anchor_jvp + next_direction
        next_frozen_anchor = anchor_field(next_time, predicted_frozen)
        next_frozen_derivative = next_frozen_anchor + scale * next_direction

        state = state + 0.5 * step * (anchor + next_anchor)
        tangent = tangent + 0.5 * step * (
            tangent_derivative + next_tangent_derivative
        )
        frozen = frozen + 0.5 * step * (
            frozen_derivative + next_frozen_derivative
        )
    return state, tangent, frozen


def decompose_along_reference(
    response: Tensor,
    reference: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Project each full sample response onto one paired reference direction.

    The returned coefficient has shape ``[batch]``. The projection uses one
    scalar over every non-batch dimension, matching the geometry diagnostics
    used elsewhere in this repository.
    """

    if response.shape != reference.shape or response.ndim < 2:
        raise ValueError("response and reference must have the same batched shape")
    response_flat = response.flatten(1)
    reference_flat = reference.flatten(1)
    denominator = reference_flat.square().sum(dim=1)
    tiny = torch.finfo(reference.dtype).tiny
    coefficient = (response_flat * reference_flat).sum(dim=1) / denominator.clamp_min(
        tiny
    )
    coefficient = torch.where(
        denominator > tiny,
        coefficient,
        torch.zeros_like(coefficient),
    )
    parallel = coefficient.reshape(-1, *([1] * (reference.ndim - 1))) * reference
    orthogonal = response - parallel
    return coefficient, parallel, orthogonal


def integrate_frozen_closed_sweep(
    anchor_field: AnchorField,
    direction_field: DirectionField,
    initial_state: Tensor,
    time_grid: Tensor,
    gammas: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compare frozen-guidance and fully closed-loop trajectories.

    The baseline follows ``v(z0,t)``.  Frozen guidance follows
    ``v(zf,t) + gamma*u(z0,t)`` while closed guidance follows
    ``v(zc,t) + gamma*u(zc,t)``.  Thus the strong field remains state-aware in
    both branches and only the weak/strong guidance feedback is frozen.
    """

    _validate_time_grid(time_grid)
    if gammas.ndim != 1 or len(gammas) == 0:
        raise ValueError("gammas must be a non-empty one-dimensional tensor")
    condition_count = len(gammas)
    batch_size = len(initial_state)
    baseline = initial_state.float()
    frozen = _repeat_conditions(baseline, condition_count)
    closed = frozen.clone()
    scales = _expand_per_condition(
        gammas.to(device=baseline.device, dtype=baseline.dtype),
        batch_size,
        frozen.ndim,
    )

    for time_value, next_time in zip(time_grid[:-1], time_grid[1:], strict=True):
        step = next_time - time_value
        base_anchor = anchor_field(time_value, baseline)
        base_direction = direction_field(time_value, baseline, base_anchor)

        frozen_anchor = anchor_field(time_value, frozen)
        repeated_base_direction = _repeat_conditions(base_direction, condition_count)
        frozen_derivative = frozen_anchor + scales * repeated_base_direction

        closed_anchor = anchor_field(time_value, closed)
        closed_direction = direction_field(time_value, closed, closed_anchor)
        closed_derivative = closed_anchor + scales * closed_direction

        predicted_baseline = baseline + step * base_anchor
        predicted_frozen = frozen + step * frozen_derivative
        predicted_closed = closed + step * closed_derivative

        next_base_anchor = anchor_field(next_time, predicted_baseline)
        next_base_direction = direction_field(
            next_time,
            predicted_baseline,
            next_base_anchor,
        )
        next_frozen_anchor = anchor_field(next_time, predicted_frozen)
        next_frozen_derivative = next_frozen_anchor + scales * _repeat_conditions(
            next_base_direction,
            condition_count,
        )
        next_closed_anchor = anchor_field(next_time, predicted_closed)
        next_closed_direction = direction_field(
            next_time,
            predicted_closed,
            next_closed_anchor,
        )
        next_closed_derivative = next_closed_anchor + scales * next_closed_direction

        baseline = baseline + 0.5 * step * (base_anchor + next_base_anchor)
        frozen = frozen + 0.5 * step * (
            frozen_derivative + next_frozen_derivative
        )
        closed = closed + 0.5 * step * (
            closed_derivative + next_closed_derivative
        )

    return (
        baseline,
        _restore_conditions(frozen, condition_count, batch_size),
        _restore_conditions(closed, condition_count, batch_size),
    )


def sample_rms(value: Tensor) -> Tensor:
    """Return one RMS value per sample over all non-batch dimensions."""

    return value.flatten(1).square().mean(dim=1).sqrt()


def sample_cosine(left: Tensor, right: Tensor) -> Tensor:
    """Return one cosine value per paired sample."""

    if left.shape != right.shape:
        raise ValueError("left and right must have identical shapes")
    left_flat = left.flatten(1)
    right_flat = right.flatten(1)
    denominator = left_flat.norm(dim=1) * right_flat.norm(dim=1)
    return (left_flat * right_flat).sum(dim=1) / denominator.clamp_min(
        torch.finfo(left.dtype).tiny
    )


def velocity_gap_to_score_gap(direction: Tensor, time_value: Tensor) -> Tensor:
    """Convert a linear-path velocity gap to its score-gap scaling.

    For ``z_t = (1-t) epsilon + t x``, the Bayes relation is
    ``score_t(z) = (t v_t(z) - z) / (1-t)``.  Hence two velocity fields differ
    in score space by ``t / (1-t)`` times their velocity difference.
    """

    time = time_value.to(device=direction.device, dtype=direction.dtype)
    if bool(torch.any(time <= 0.0)) or bool(torch.any(time >= 1.0)):
        raise ValueError("score-gap conversion requires time strictly inside (0, 1)")
    scale = time / (1.0 - time)
    while scale.ndim < direction.ndim:
        scale = scale.unsqueeze(-1)
    return scale * direction


def jacobian_symmetry_probe(
    field: Callable[[Tensor], Tensor],
    state: Tensor,
    probe: Tensor,
) -> dict[str, Tensor]:
    """Estimate whether a vector field has a symmetric state Jacobian.

    A differentiable score field is conservative and therefore has a symmetric
    Jacobian wherever the underlying potential is twice differentiable. For a
    Hutchinson probe ``q``, this compares ``J q`` from forward AD with ``J^T q``
    from reverse AD without materializing the full Jacobian.

    ``antisymmetric_energy_fraction`` is zero for an exactly symmetric Jacobian
    and one for an exactly antisymmetric linear map.
    """

    if state.shape != probe.shape:
        raise ValueError("state and probe must have identical shapes")

    output, jacobian_vector = torch.func.jvp(
        field,
        (state,),
        (probe,),
        strict=True,
    )
    transpose_output, transpose_function = torch.func.vjp(field, state)
    if not torch.allclose(output, transpose_output, rtol=1e-5, atol=1e-6):
        raise RuntimeError("JVP and VJP primal evaluations disagree")
    transpose_jacobian_vector = transpose_function(probe)[0]
    antisymmetric = jacobian_vector - transpose_jacobian_vector
    jvp_rms = sample_rms(jacobian_vector)
    vjp_rms = sample_rms(transpose_jacobian_vector)
    antisymmetric_rms = sample_rms(antisymmetric)
    tiny = torch.finfo(state.dtype).tiny
    energy_denominator = 2.0 * (
        jacobian_vector.flatten(1).square().mean(dim=1)
        + transpose_jacobian_vector.flatten(1).square().mean(dim=1)
    )
    return {
        "field_rms": sample_rms(output),
        "jvp_rms": jvp_rms,
        "vjp_rms": vjp_rms,
        "antisymmetric_rms": antisymmetric_rms,
        "antisymmetric_over_jvp_rms": antisymmetric_rms / jvp_rms.clamp_min(tiny),
        "antisymmetric_energy_fraction": antisymmetric.flatten(1)
        .square()
        .mean(dim=1)
        / energy_denominator.clamp_min(tiny),
        "jvp_vjp_cosine": sample_cosine(
            jacobian_vector, transpose_jacobian_vector
        ),
    }


def linearity_metrics(
    baseline: Tensor,
    guided: Tensor,
    tangent: Tensor,
    *,
    gamma: float,
) -> dict[str, Tensor]:
    """Measure a finite endpoint response against the variational prediction."""

    if gamma == 0.0:
        raise ValueError("linearity metrics are undefined for gamma=0")
    if baseline.shape != guided.shape or baseline.shape != tangent.shape:
        raise ValueError("baseline, guided, and tangent must have identical shapes")
    actual = guided - baseline
    linear = float(gamma) * tangent
    residual = actual - linear
    return {
        "actual_rms": sample_rms(actual),
        "linear_rms": sample_rms(linear),
        "residual_rms": sample_rms(residual),
        "relative_residual": sample_rms(residual)
        / sample_rms(actual).clamp_min(torch.finfo(actual.dtype).tiny),
        "cosine": sample_cosine(actual, linear),
        "magnitude_ratio": sample_rms(actual)
        / sample_rms(linear).clamp_min(torch.finfo(actual.dtype).tiny),
    }


def central_difference_metrics(
    minus_endpoint: Tensor,
    plus_endpoint: Tensor,
    tangent: Tensor,
    *,
    delta: float,
) -> dict[str, Tensor]:
    """Validate a tangent with paired ``+/- delta`` finite differences."""

    if delta <= 0:
        raise ValueError("delta must be positive")
    finite_difference = (plus_endpoint - minus_endpoint) / (2.0 * float(delta))
    residual = finite_difference - tangent
    return {
        "finite_difference_rms": sample_rms(finite_difference),
        "tangent_rms": sample_rms(tangent),
        "relative_residual": sample_rms(residual)
        / sample_rms(finite_difference).clamp_min(
            torch.finfo(finite_difference.dtype).tiny
        ),
        "cosine": sample_cosine(finite_difference, tangent),
    }
