"""Fixed-step explicit and implicit ODE solvers for mechanism experiments.

The implicit methods use an explicit-Euler predictor followed by a finite
number of relaxed Picard corrections.  They therefore expose the exact cost
of treating the next state as a fixed point instead of hiding nonlinear solves
inside a library routine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch


Tensor = torch.Tensor
Field = Callable[[Tensor, Tensor], Tensor]


@dataclass(frozen=True)
class FixedStepResult:
    endpoint: Tensor
    nfe: int
    mean_last_update_rms: float
    max_last_update_rms: float


def _tensor_rms(value: Tensor) -> Tensor:
    return value.float().square().mean().sqrt()


def _validate(
    times: Tensor,
    method: str,
    corrections: int,
    relaxation: float,
) -> None:
    if times.ndim != 1 or len(times) < 2:
        raise ValueError("times must be a one-dimensional grid with at least two points")
    if not bool(torch.all(times[1:] > times[:-1])):
        raise ValueError("times must be strictly increasing")
    allowed = {
        "euler",
        "heun",
        "backward_euler",
        "implicit_midpoint",
        "implicit_trapezoid",
    }
    if method not in allowed:
        raise ValueError(f"unsupported method: {method}")
    if corrections < 1:
        raise ValueError("corrections must be positive")
    if not math.isfinite(relaxation) or not 0.0 < relaxation <= 1.0:
        raise ValueError("relaxation must lie in (0, 1]")


def integrate_fixed_grid(
    field: Field,
    initial: Tensor,
    times: Tensor,
    *,
    method: str,
    corrections: int = 1,
    relaxation: float = 1.0,
) -> FixedStepResult:
    """Integrate ``dz/dt = field(t, z)`` over a prescribed time grid.

    ``implicit_trapezoid`` with one correction is exactly Heun's method.
    ``implicit_midpoint`` with one correction is the explicit midpoint method.
    For all implicit methods, each additional correction costs one NFE.
    ``mean_last_update_rms`` measures the final Picard update, not the exact
    post-update algebraic residual, which would require another model call.
    """

    _validate(times, method, corrections, relaxation)
    state = initial
    nfe = 0
    last_updates: list[float] = []

    for index in range(len(times) - 1):
        time = times[index]
        next_time = times[index + 1]
        step = next_time - time
        start_velocity = field(time, state)
        nfe += 1

        if method == "euler":
            state = state + step * start_velocity
            continue

        predictor = state + step * start_velocity
        if method == "heun":
            end_velocity = field(next_time, predictor)
            nfe += 1
            state = state + 0.5 * step * (start_velocity + end_velocity)
            continue

        candidate = predictor
        last_update = torch.zeros((), device=state.device, dtype=torch.float32)
        for _ in range(corrections):
            if method == "backward_euler":
                velocity = field(next_time, candidate)
                target = state + step * velocity
            elif method == "implicit_midpoint":
                midpoint = 0.5 * (state + candidate)
                velocity = field(0.5 * (time + next_time), midpoint)
                target = state + step * velocity
            else:
                velocity = field(next_time, candidate)
                target = state + 0.5 * step * (start_velocity + velocity)
            nfe += 1
            updated = candidate + relaxation * (target - candidate)
            last_update = _tensor_rms(updated - candidate)
            candidate = updated
        state = candidate
        last_updates.append(float(last_update.item()))

    return FixedStepResult(
        endpoint=state,
        nfe=nfe,
        mean_last_update_rms=(
            float(sum(last_updates) / len(last_updates)) if last_updates else 0.0
        ),
        max_last_update_rms=max(last_updates, default=0.0),
    )
