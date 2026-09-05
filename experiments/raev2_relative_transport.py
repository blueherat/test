"""Discrete relative transport operators for RAEv2 flow-map experiments.

The reference Euler map ``R`` and guided Euler map ``G`` share a time grid.
The candidate operator is ``T = G o R^{-1}``, applied once more to a guided
switch state.  This module contains only generic tensor/map numerics; model
loading and image sampling live in ``sample_raev2_relative_transport.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import Tensor


VelocityFunction = Callable[[Tensor, float], Tensor]


@dataclass(frozen=True)
class InverseStepAudit:
    index: int
    current_time: float
    following_time: float
    iterations: int
    relative_fixed_point_residual: float
    converged: bool


@dataclass(frozen=True)
class InverseMapResult:
    state: Tensor
    steps: tuple[InverseStepAudit, ...]

    @property
    def converged(self) -> bool:
        return all(step.converged for step in self.steps)

    @property
    def maximum_iterations(self) -> int:
        return max((step.iterations for step in self.steps), default=0)

    @property
    def maximum_relative_residual(self) -> float:
        return max(
            (step.relative_fixed_point_residual for step in self.steps),
            default=0.0,
        )


def _validate_state(state: Tensor, name: str) -> None:
    if not state.is_floating_point() or state.ndim < 2 or len(state) == 0:
        raise ValueError(f"{name} must be a nonempty floating batch tensor")


def _validate_grid(time_grid: Sequence[float]) -> tuple[float, ...]:
    grid = tuple(float(value) for value in time_grid)
    if len(grid) < 2 or any(not math.isfinite(value) for value in grid):
        raise ValueError("time_grid must contain at least two finite values")
    if any(left <= right for left, right in zip(grid[:-1], grid[1:])):
        raise ValueError("time_grid must be strictly decreasing")
    return grid


def sample_rms(value: Tensor) -> Tensor:
    _validate_state(value, "value")
    return value.float().flatten(1).square().mean(1).sqrt()


def first_index_at_or_below(time_grid: Sequence[float], switch_time: float) -> int:
    """Return the first grid index at or below a requested switch time."""

    grid = _validate_grid(time_grid)
    if not math.isfinite(switch_time) or not grid[-1] <= switch_time < grid[0]:
        raise ValueError("switch_time must lie inside the grid interval")
    for index, value in enumerate(grid):
        if value <= switch_time:
            if index == 0:
                raise AssertionError("validated switch unexpectedly selected index zero")
            return index
    raise AssertionError("validated switch was not found")


def euler_step(
    state: Tensor,
    current_time: float,
    following_time: float,
    velocity: VelocityFunction,
) -> Tensor:
    """Apply one explicit Euler step on a decreasing time grid."""

    _validate_state(state, "state")
    current = float(current_time)
    following = float(following_time)
    if not (math.isfinite(current) and math.isfinite(following)) or following >= current:
        raise ValueError("Euler times must be finite and strictly decreasing")
    field = velocity(state, current)
    if field.shape != state.shape or field.device != state.device:
        raise ValueError("velocity output must match the state")
    return state + (following - current) * field


def integrate_euler(
    state: Tensor,
    time_grid: Sequence[float],
    velocity: VelocityFunction,
) -> Tensor:
    """Integrate a field across an explicit decreasing grid."""

    grid = _validate_grid(time_grid)
    result = state
    for current, following in zip(grid[:-1], grid[1:]):
        result = euler_step(result, current, following, velocity)
    return result


def invert_euler_step_fixed_point(
    following_state: Tensor,
    current_time: float,
    following_time: float,
    velocity: VelocityFunction,
    *,
    tolerance: float = 1e-5,
    maximum_iterations: int = 16,
    tiny: float = 1e-8,
) -> tuple[Tensor, int, float, bool]:
    """Invert one explicit Euler update by fixed-point iteration.

    A forward step is ``y = x + (t_next - t) f(x, t)``.  Its predecessor
    therefore solves ``x = y + (t - t_next) f(x, t)``.  The returned residual
    is the maximum per-sample RMS of the fixed-point update divided by the
    predecessor RMS.  It is a numerical diagnostic, not a scientific knob.
    """

    _validate_state(following_state, "following_state")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if not isinstance(maximum_iterations, int) or maximum_iterations < 1:
        raise ValueError("maximum_iterations must be a positive integer")
    if not math.isfinite(tiny) or tiny <= 0.0:
        raise ValueError("tiny must be finite and positive")
    current = float(current_time)
    following = float(following_time)
    if not (math.isfinite(current) and math.isfinite(following)) or following >= current:
        raise ValueError("inverse Euler times must be finite and strictly decreasing")

    step = current - following
    estimate = following_state
    relative = math.inf
    for iteration in range(1, maximum_iterations + 1):
        revised = following_state + step * velocity(estimate, current)
        relative_per_sample = sample_rms(revised - estimate) / sample_rms(
            revised
        ).clamp_min(tiny)
        relative = float(relative_per_sample.max().item())
        estimate = revised
        if relative <= tolerance:
            return estimate, iteration, relative, True
    return estimate, maximum_iterations, relative, False


def invert_euler_map_fixed_point(
    terminal_state: Tensor,
    time_grid: Sequence[float],
    velocity: VelocityFunction,
    *,
    tolerance: float = 1e-5,
    maximum_iterations: int = 16,
) -> InverseMapResult:
    """Invert every step of a discrete Euler map in reverse order."""

    grid = _validate_grid(time_grid)
    state = terminal_state
    audits: list[InverseStepAudit] = []
    for index in reversed(range(len(grid) - 1)):
        state, iterations, residual, converged = invert_euler_step_fixed_point(
            state,
            grid[index],
            grid[index + 1],
            velocity,
            tolerance=tolerance,
            maximum_iterations=maximum_iterations,
        )
        audits.append(
            InverseStepAudit(
                index=index,
                current_time=grid[index],
                following_time=grid[index + 1],
                iterations=iterations,
                relative_fixed_point_residual=residual,
                converged=converged,
            )
        )
    return InverseMapResult(state=state, steps=tuple(audits))


def relative_transport_iterate(
    guided_switch: Tensor,
    inverse_reference_noise: Tensor,
    guided_map: Callable[[Tensor], Tensor],
) -> Tensor:
    """Apply ``G o R^{-1}`` once to a state already produced by ``G``."""

    _validate_state(guided_switch, "guided_switch")
    _validate_state(inverse_reference_noise, "inverse_reference_noise")
    if guided_switch.shape != inverse_reference_noise.shape:
        raise ValueError("guided switch and inverse noise must have matching shapes")
    result = guided_map(inverse_reference_noise)
    if result.shape != guided_switch.shape:
        raise ValueError("guided map output must match the switch state")
    return result
