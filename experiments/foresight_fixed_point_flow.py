"""Fixed-point calibration operators for noise-to-data flow matching.

The repository's SiT convention is ``t=0`` noise and ``t=1`` data.  A
foresight round trip therefore advances from ``t`` to ``t + delta`` with a
guided field and returns to ``t`` with a reference field.  This is the
time-reversed analogue of Foresight Guidance's denoise-then-invert operator.

The routines use explicit Euler updates on purpose: one forward and one
backward model evaluation match the two-evaluation operator studied by FSG,
and the one-step split below is algebraically identical to Euler integration
of ``reference + scale * (target - reference)``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch


Tensor = torch.Tensor
VectorField = Callable[[Tensor, Tensor], Tensor]


@dataclass(frozen=True)
class ForesightEvent:
    """Apply ``iterations`` round trips over ``lookahead_steps`` base steps."""

    step_index: int
    lookahead_steps: int
    iterations: int

    def validate(self, *, num_steps: int) -> None:
        if self.step_index < 0 or self.step_index >= num_steps:
            raise ValueError("foresight step index is outside the base grid")
        if self.lookahead_steps <= 0:
            raise ValueError("lookahead_steps must be positive")
        if self.step_index + self.lookahead_steps > num_steps:
            raise ValueError("foresight interval extends beyond t=1")
        if self.iterations <= 0:
            raise ValueError("foresight iterations must be positive")


def parse_foresight_schedule(value: str) -> tuple[ForesightEvent, ...]:
    """Parse ``step:lookahead:iterations`` entries separated by commas."""

    if not value.strip():
        return ()
    events: list[ForesightEvent] = []
    for raw_entry in value.split(","):
        fields = raw_entry.strip().split(":")
        if len(fields) != 3:
            raise ValueError(
                "schedule entries must have step:lookahead:iterations format"
            )
        events.append(ForesightEvent(*(int(field) for field in fields)))
    indices = [event.step_index for event in events]
    if len(indices) != len(set(indices)):
        raise ValueError("foresight schedule contains duplicate step indices")
    return tuple(sorted(events, key=lambda event: event.step_index))


def schedule_by_step(
    events: Sequence[ForesightEvent], *, num_steps: int
) -> dict[int, ForesightEvent]:
    result: dict[int, ForesightEvent] = {}
    for event in events:
        event.validate(num_steps=num_steps)
        if event.step_index in result:
            raise ValueError("foresight schedule contains duplicate step indices")
        result[event.step_index] = event
    return result


def guided_field(
    reference: Tensor,
    target: Tensor,
    *,
    scale: float,
) -> Tensor:
    """Return ``reference + scale * (target - reference)`` exactly at endpoints."""

    if reference.shape != target.shape:
        raise ValueError("reference and target fields must have identical shapes")
    if scale == 0.0:
        return reference
    if scale == 1.0:
        return target
    return reference + float(scale) * (target - reference)


def euler_update(
    state: Tensor,
    *,
    time_value: Tensor,
    next_time: Tensor,
    field: VectorField,
) -> Tensor:
    """Take one explicit Euler update in either time direction."""

    return state + (next_time - time_value) * field(time_value, state)


def heun_update(
    state: Tensor,
    *,
    time_value: Tensor,
    next_time: Tensor,
    field: VectorField,
) -> Tensor:
    """Take one explicit trapezoidal (Heun) update."""

    step = next_time - time_value
    first = field(time_value, state)
    predictor = state + step * first
    second = field(next_time, predictor)
    return state + 0.5 * step * (first + second)


def rk4_update(
    state: Tensor,
    *,
    time_value: Tensor,
    next_time: Tensor,
    field: VectorField,
) -> Tensor:
    """Take one classical fourth-order Runge--Kutta update."""

    step = next_time - time_value
    midpoint = time_value + 0.5 * step
    first = field(time_value, state)
    second = field(midpoint, state + 0.5 * step * first)
    third = field(midpoint, state + 0.5 * step * second)
    fourth = field(next_time, state + step * third)
    return state + (step / 6.0) * (first + 2.0 * second + 2.0 * third + fourth)


def euler_flow_map(
    state: Tensor,
    *,
    time_values: Sequence[Tensor],
    field: VectorField,
) -> Tensor:
    """Integrate a field across an ordered time grid with explicit Euler."""

    if len(time_values) < 2:
        raise ValueError("a flow map requires at least two time values")
    current = state
    for time_value, next_time in zip(time_values[:-1], time_values[1:], strict=True):
        current = euler_update(
            current,
            time_value=time_value,
            next_time=next_time,
            field=field,
        )
    return current


def integrate_flow_map(
    state: Tensor,
    *,
    time_values: Sequence[Tensor],
    field: VectorField,
    method: str,
) -> Tensor:
    """Integrate a field over an ordered grid with a selected fixed-step solver."""

    updates = {
        "euler": euler_update,
        "heun": heun_update,
        "rk4": rk4_update,
    }
    try:
        update = updates[method]
    except KeyError as error:
        raise ValueError(f"unsupported flow-map integrator: {method}") from error
    if len(time_values) < 2:
        raise ValueError("a flow map requires at least two time values")
    current = state
    for time_value, next_time in zip(time_values[:-1], time_values[1:], strict=True):
        current = update(
            current,
            time_value=time_value,
            next_time=next_time,
            field=field,
        )
    return current


def foresight_round_trip(
    state: Tensor,
    *,
    time_value: Tensor,
    future_time: Tensor,
    forward_field: VectorField,
    inverse_field: VectorField,
    relaxation: float = 1.0,
) -> Tensor:
    """Advance, return, then optionally relax toward the round-trip result."""

    if not bool(future_time > time_value):
        raise ValueError("future_time must be greater than time_value")
    if not 0.0 < float(relaxation) <= 1.0:
        raise ValueError("relaxation must be in (0, 1]")
    future = euler_update(
        state,
        time_value=time_value,
        next_time=future_time,
        field=forward_field,
    )
    round_trip = euler_update(
        future,
        time_value=future_time,
        next_time=time_value,
        field=inverse_field,
    )
    if relaxation == 1.0:
        return round_trip
    return state + float(relaxation) * (round_trip - state)


def iterate_foresight_operator(
    state: Tensor,
    *,
    time_value: Tensor,
    future_time: Tensor,
    iterations: int,
    forward_field: VectorField,
    inverse_field: VectorField,
    relaxation: float = 1.0,
) -> tuple[Tensor, list[Tensor]]:
    """Apply a foresight round trip repeatedly and return per-iteration moves."""

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    current = state
    displacements: list[Tensor] = []
    for _ in range(iterations):
        updated = foresight_round_trip(
            current,
            time_value=time_value,
            future_time=future_time,
            forward_field=forward_field,
            inverse_field=inverse_field,
            relaxation=relaxation,
        )
        displacements.append(updated - current)
        current = updated
    return current, displacements


def anchored_foresight_step(
    anchor: Tensor,
    current: Tensor,
    *,
    time_value: Tensor,
    future_time: Tensor,
    forward_field: VectorField,
    inverse_field: VectorField,
    strength: float,
) -> tuple[Tensor, Tensor]:
    """Take one input-anchored iteration using a future field discrepancy.

    Let ``F_H`` be the forward/inverse round trip.  Unlike direct iteration
    ``y <- F_H(y)``, whose fixed points erase the discrepancy, this update is

    ``y <- anchor + strength * (F_H(y) - y)``.

    Its fixed point retains a generally nonzero discrepancy balanced by the
    displacement from ``anchor``.  For a short interval ``H`` and a
    strong-forward/weak-inverse pair, ``F_H(y)-y = H(S-W)(y)+O(H^2)``.
    """

    if anchor.shape != current.shape:
        raise ValueError("anchor and current must have identical shapes")
    if float(strength) <= 0.0:
        raise ValueError("anchored strength must be positive")
    mapped = foresight_round_trip(
        current,
        time_value=time_value,
        future_time=future_time,
        forward_field=forward_field,
        inverse_field=inverse_field,
    )
    discrepancy = mapped - current
    return anchor + float(strength) * discrepancy, discrepancy


def iterate_anchored_foresight_operator(
    state: Tensor,
    *,
    time_value: Tensor,
    future_time: Tensor,
    iterations: int,
    forward_field: VectorField,
    inverse_field: VectorField,
    strength: float,
) -> tuple[Tensor, list[Tensor], list[Tensor]]:
    """Solve the anchored nonzero-discrepancy subproblem by Picard iteration."""

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    anchor = state
    current = state
    moves: list[Tensor] = []
    discrepancies: list[Tensor] = []
    for _ in range(iterations):
        updated, discrepancy = anchored_foresight_step(
            anchor,
            current,
            time_value=time_value,
            future_time=future_time,
            forward_field=forward_field,
            inverse_field=inverse_field,
            strength=strength,
        )
        moves.append(updated - current)
        discrepancies.append(discrepancy)
        current = updated
    return current, moves, discrepancies


def iterate_anchored_gap_operator(
    state: Tensor,
    *,
    time_value: Tensor,
    iterations: int,
    strong_field: VectorField,
    weak_field: VectorField,
    step_strength: float,
) -> tuple[Tensor, Tensor, list[Tensor], list[Tensor]]:
    """Solve ``y = x + eta * (S(y)-W(y))`` with an input anchor ``x``.

    The first iteration is the ordinary explicit AutoGuidance calibration.
    ``strong_at_anchor`` is returned so adding the base denoising update uses
    exactly the same strong velocity as explicit Euler, making ``K=1`` an
    exact numerical control rather than only a first-order approximation.
    """

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if float(step_strength) <= 0.0:
        raise ValueError("gap step strength must be positive")
    anchor = state
    current = state
    strong_at_anchor: Tensor | None = None
    moves: list[Tensor] = []
    gaps: list[Tensor] = []
    for iteration in range(iterations):
        strong = strong_field(time_value, current)
        weak = weak_field(time_value, current)
        if iteration == 0:
            strong_at_anchor = strong
        gap = strong - weak
        updated = anchor + float(step_strength) * gap
        moves.append(updated - current)
        gaps.append(gap)
        current = updated
    assert strong_at_anchor is not None
    return current, strong_at_anchor, moves, gaps


def implicit_autoguidance_euler_step(
    state: Tensor,
    *,
    time_value: Tensor,
    next_time: Tensor,
    iterations: int,
    strong_field: VectorField,
    weak_field: VectorField,
    gamma: float,
) -> tuple[Tensor, list[Tensor], list[Tensor]]:
    """Take one Euler AG step with an implicitly calibrated discrepancy.

    The inner Picard iteration solves

    ``y = x + h * gamma * (S(y)-W(y))``.

    The outer update is evaluated as

    ``x_next = x + h * (S(x) + gamma * (S(y)-W(y)))``.

    With one inner iteration, ``y`` is initialized at ``x`` and the returned
    expression is bitwise identical to the ordinary explicit Euler AG update.
    Additional iterations change only where the strong-minus-weak discrepancy
    is queried; the base strong velocity remains fixed at the input anchor.
    """

    step = next_time - time_value
    if not bool(step > 0):
        raise ValueError("next_time must be greater than time_value")
    if float(gamma) <= 0.0:
        raise ValueError("AG gamma must be positive")
    _, strong_at_anchor, moves, gaps = iterate_anchored_gap_operator(
        state,
        time_value=time_value,
        iterations=iterations,
        strong_field=strong_field,
        weak_field=weak_field,
        step_strength=float(step) * float(gamma),
    )
    endpoint = state + step * (strong_at_anchor + float(gamma) * gaps[-1])
    return endpoint, moves, gaps


def scheduled_autoguidance_euler_step(
    state: Tensor,
    *,
    time_value: Tensor,
    next_time: Tensor,
    strong_field: VectorField,
    weak_field: VectorField,
    gamma: float,
    multiplier: float,
) -> Tensor:
    """Take an ordinary AG Euler step with a locally rescaled gamma.

    This is the direct time-schedule control for foresight calibration.  It
    changes no query state and introduces no future information or flow-map
    transport.
    """

    step = next_time - time_value
    if not bool(step > 0):
        raise ValueError("next_time must be greater than time_value")
    if float(gamma) <= 0.0 or float(multiplier) <= 0.0:
        raise ValueError("AG gamma and multiplier must be positive")
    strong = strong_field(time_value, state)
    weak = weak_field(time_value, state)
    return state + step * (
        strong + float(gamma) * float(multiplier) * (strong - weak)
    )


def local_calibrated_autoguidance_euler_step(
    state: Tensor,
    *,
    time_value: Tensor,
    next_time: Tensor,
    strong_field: VectorField,
    weak_field: VectorField,
    gamma: float,
    multiplier: float,
) -> tuple[Tensor, Tensor]:
    """Calibrate at the current state, then let the strong field respond.

    This is the zero-lookahead counterpart of conjugated future calibration:

    ``y = x + h * gamma * multiplier * (S(x)-W(x))``
    ``x_next = y + h * S(y)``.

    Comparing it with direct scheduled AG isolates the effect of evaluating
    the strong field after calibration; comparing it with future calibration
    isolates the information supplied by a nonzero lookahead horizon.
    """

    step = next_time - time_value
    if not bool(step > 0):
        raise ValueError("next_time must be greater than time_value")
    if float(gamma) <= 0.0 or float(multiplier) <= 0.0:
        raise ValueError("AG gamma and multiplier must be positive")
    strong = strong_field(time_value, state)
    weak = weak_field(time_value, state)
    calibration = step * float(gamma) * float(multiplier) * (strong - weak)
    calibrated = state + calibration
    return calibrated + step * strong_field(time_value, calibrated), calibration


def future_raw_gap_step(
    state: Tensor,
    *,
    time_values: Sequence[Tensor],
    strong_field: VectorField,
    weak_field: VectorField,
    calibration_strength: float,
    flow_integrator: str = "rk4",
) -> tuple[Tensor, Tensor, Tensor]:
    """Query the future AG gap but inject it in current latent coordinates.

    This deliberately omits the inverse strong-flow pullback.  It is therefore
    a counterfactual control, not a proposed sampler: any difference from
    conjugated calibration measures the value of transporting the future
    correction through the strong flow geometry.
    """

    if float(calibration_strength) <= 0.0:
        raise ValueError("future calibration strength must be positive")
    future = integrate_flow_map(
        state,
        time_values=time_values,
        field=strong_field,
        method=flow_integrator,
    )
    future_gap = strong_field(time_values[-1], future) - weak_field(
        time_values[-1], future
    )
    move = float(calibration_strength) * future_gap
    return state + move, future_gap, move


def cross_time_norm_matched_gap_step(
    state: Tensor,
    *,
    time_values: Sequence[Tensor],
    strong_field: VectorField,
    weak_field: VectorField,
    calibration_strength: float,
    direction: str,
    flow_integrator: str = "rk4",
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Calibrate with current/future AG directions under matched sample RMS.

    ``future_match_current`` preserves the future-gap direction but gives it
    the current gap's per-sample RMS. ``current_match_future`` does the exact
    converse.  The pair isolates direction from magnitude while performing
    identical model queries.
    """

    if float(calibration_strength) <= 0.0:
        raise ValueError("future calibration strength must be positive")
    if direction not in {"future_match_current", "current_match_future"}:
        raise ValueError(f"unsupported matched-gap direction: {direction}")
    current_gap = strong_field(time_values[0], state) - weak_field(
        time_values[0], state
    )
    future = integrate_flow_map(
        state,
        time_values=time_values,
        field=strong_field,
        method=flow_integrator,
    )
    future_gap = strong_field(time_values[-1], future) - weak_field(
        time_values[-1], future
    )

    reduce_dims = tuple(range(1, state.ndim))
    tiny = torch.finfo(state.dtype).tiny
    current_rms = current_gap.square().mean(dim=reduce_dims, keepdim=True).sqrt()
    future_rms = future_gap.square().mean(dim=reduce_dims, keepdim=True).sqrt()
    if direction == "future_match_current":
        selected = future_gap * (current_rms / future_rms.clamp_min(tiny))
    else:
        selected = current_gap * (future_rms / current_rms.clamp_min(tiny))
    move = float(calibration_strength) * selected
    return state + move, current_gap, future_gap, move


def conjugated_future_gap_step(
    state: Tensor,
    *,
    time_values: Sequence[Tensor],
    strong_field: VectorField,
    weak_field: VectorField,
    calibration_strength: float,
    flow_integrator: str = "rk4",
) -> tuple[Tensor, Tensor, Tensor]:
    """Transport a future AG calibration back through the strong flow.

    For an exact strong flow map ``Phi`` and future calibration
    ``C(z)=z+eta*(S(z)-W(z))``, this computes the numerical analogue of

    ``T = Phi^{-1} o C o Phi``.

    The returned tensors are the calibrated current state, the future gap,
    and the future calibrated state.
    """

    if float(calibration_strength) <= 0.0:
        raise ValueError("future calibration strength must be positive")
    future = integrate_flow_map(
        state,
        time_values=time_values,
        field=strong_field,
        method=flow_integrator,
    )
    future_gap = strong_field(time_values[-1], future) - weak_field(
        time_values[-1], future
    )
    calibrated_future = future + float(calibration_strength) * future_gap
    reverse_times = tuple(reversed(time_values))
    calibrated_current = integrate_flow_map(
        calibrated_future,
        time_values=reverse_times,
        field=strong_field,
        method=flow_integrator,
    )
    return calibrated_current, future_gap, calibrated_future


def iterate_conjugated_future_gap_operator(
    state: Tensor,
    *,
    time_values: Sequence[Tensor],
    iterations: int,
    strong_field: VectorField,
    weak_field: VectorField,
    calibration_strength: float,
    flow_integrator: str = "rk4",
) -> tuple[Tensor, list[Tensor], list[Tensor]]:
    """Apply conjugated future AutoGuidance calibration repeatedly."""

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    current = state
    moves: list[Tensor] = []
    future_gaps: list[Tensor] = []
    for _ in range(iterations):
        calibrated, future_gap, _ = conjugated_future_gap_step(
            current,
            time_values=time_values,
            strong_field=strong_field,
            weak_field=weak_field,
            calibration_strength=calibration_strength,
            flow_integrator=flow_integrator,
        )
        moves.append(calibrated - current)
        future_gaps.append(future_gap)
        current = calibrated
    return current, moves, future_gaps


def split_guided_euler_step(
    state: Tensor,
    *,
    time_value: Tensor,
    next_time: Tensor,
    reference_field: VectorField,
    target_field: VectorField,
    scale: float,
    calibration_iterations: int = 1,
) -> tuple[Tensor, list[Tensor]]:
    """Calibrate repeatedly, then denoise once with the reference field.

    With one calibration iteration this is exactly

    ``state + h * [reference(state) + scale * (target(state)-reference(state))]``.

    For multiple iterations, the final reference velocity follows FSG's
    CFG/CFG++ algorithms: it is evaluated at the input of the final calibration
    and then applied to the calibrated state.
    """

    if calibration_iterations <= 0:
        raise ValueError("calibration_iterations must be positive")
    step = next_time - time_value
    calibrated = state
    displacements: list[Tensor] = []
    last_reference: Tensor | None = None
    for _ in range(calibration_iterations):
        reference = reference_field(time_value, calibrated)
        target = target_field(time_value, calibrated)
        displacement = step * float(scale) * (target - reference)
        last_reference = reference
        calibrated = calibrated + displacement
        displacements.append(displacement)
    assert last_reference is not None
    return calibrated + step * last_reference, displacements


def sample_rms(value: Tensor) -> Tensor:
    """Return RMS over all non-batch dimensions."""

    if value.ndim < 2:
        raise ValueError("expected a batched tensor")
    return value.float().flatten(1).square().mean(dim=1).sqrt()


def sample_cosine(left: Tensor, right: Tensor) -> Tensor:
    """Return one flattened cosine per sample."""

    if left.shape != right.shape or left.ndim < 2:
        raise ValueError("left and right must have the same batched shape")
    left_flat = left.float().flatten(1)
    right_flat = right.float().flatten(1)
    denominator = left_flat.norm(dim=1) * right_flat.norm(dim=1)
    tiny = torch.finfo(left_flat.dtype).tiny
    cosine = (left_flat * right_flat).sum(dim=1) / denominator.clamp_min(tiny)
    return torch.where(denominator > tiny, cosine, torch.zeros_like(cosine))
