"""Model-independent diagnostics for internal-guidance directions.

The module treats a dual-head model as producing a strong prediction ``full``
and a weaker prediction ``base``.  Internal guidance uses

    guided(scale) = base + scale * (full - base).

Relative to the full prediction, ``gamma = scale - 1`` controls the direction
``full - base``.  The functions below distinguish a merely worse base head
from a direction that actually points toward the remaining supervised error.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch


Tensor = torch.Tensor
ScalePolicy = Callable[[int, Tensor, Tensor, Tensor], float | Tensor]


def _validate_same_shape(*tensors: Tensor) -> None:
    if not tensors:
        raise ValueError("at least one tensor is required")
    shape = tensors[0].shape
    if tensors[0].ndim < 2:
        raise ValueError("predictions must have a batch dimension and features")
    if any(tensor.shape != shape for tensor in tensors[1:]):
        raise ValueError("all predictions and targets must have identical shapes")


def _flatten_batch(tensor: Tensor) -> Tensor:
    if tensor.ndim < 2:
        raise ValueError("tensor must have a batch dimension and features")
    return tensor.reshape(tensor.shape[0], -1)


def guided_prediction(full: Tensor, base: Tensor, scale: float | Tensor) -> Tensor:
    """Return ``base + scale * (full - base)`` with safe broadcasting."""

    _validate_same_shape(full, base)
    if isinstance(scale, Tensor):
        scale = scale.to(device=full.device, dtype=full.dtype)
        if scale.ndim == 0:
            pass
        elif scale.ndim == 1 and scale.shape[0] == full.shape[0]:
            scale = scale.reshape((full.shape[0],) + (1,) * (full.ndim - 1))
        else:
            raise ValueError("tensor scale must be scalar or contain one value per sample")
    return base + scale * (full - base)


def direction_metrics(
    full: Tensor,
    base: Tensor,
    target: Tensor,
    *,
    eps: float = 1e-12,
) -> dict[str, Tensor]:
    """Measure whether ``full - base`` corrects the residual of ``full``.

    ``gamma_star`` is the per-sample least-squares coefficient in
    ``full + gamma * (full - base)``.  The corresponding internal-guidance
    scale is ``scale_star = 1 + gamma_star``.
    """

    _validate_same_shape(full, base, target)
    if eps <= 0:
        raise ValueError("eps must be positive")

    full_flat = _flatten_batch(full.float())
    base_flat = _flatten_batch(base.float())
    target_flat = _flatten_batch(target.float())
    direction = full_flat - base_flat
    correction = target_flat - full_flat
    direction_sq = direction.square().sum(dim=1)
    correction_sq = correction.square().sum(dim=1)
    alignment = (direction * correction).sum(dim=1)
    cosine = alignment / (direction_sq * correction_sq).sqrt().clamp_min(eps)
    gamma_star = alignment / direction_sq.clamp_min(eps)
    oracle_error = correction - gamma_star[:, None] * direction

    dimension = float(full_flat.shape[1])
    full_mse = correction_sq / dimension
    base_mse = (target_flat - base_flat).square().mean(dim=1)
    oracle_mse = oracle_error.square().mean(dim=1)
    return {
        "full_mse": full_mse,
        "base_mse": base_mse,
        "direction_rms": (direction_sq / dimension).sqrt(),
        "residual_rms": full_mse.sqrt(),
        "alignment": alignment / dimension,
        "alignment_cosine": cosine,
        "positive_alignment": alignment.gt(0),
        "gamma_star": gamma_star,
        "scale_star": 1.0 + gamma_star,
        "oracle_mse": oracle_mse,
        "oracle_relative_gain": 1.0 - oracle_mse / full_mse.clamp_min(eps),
    }


def scale_sweep_metrics(
    full: Tensor,
    base: Tensor,
    target: Tensor,
    scales: Sequence[float],
    *,
    eps: float = 1e-12,
) -> dict[str, Tensor]:
    """Evaluate paired supervised error for a fixed set of IG scales."""

    _validate_same_shape(full, base, target)
    scale_values = tuple(float(value) for value in scales)
    if not scale_values:
        raise ValueError("at least one scale is required")
    predictions = torch.stack(
        [guided_prediction(full, base, value).float() for value in scale_values], dim=0
    )
    target_stack = target.float().unsqueeze(0)
    mse = (predictions - target_stack).square().flatten(2).mean(dim=2)
    full_mse = (full.float() - target.float()).square().flatten(1).mean(dim=1)
    return {
        "scales": torch.tensor(scale_values, dtype=torch.float32),
        "mse": mse,
        "gain_over_full": 1.0 - mse / full_mse.unsqueeze(0).clamp_min(eps),
    }


def direction_gram_metrics(
    directions: Sequence[Tensor],
    *,
    eps: float = 1e-12,
) -> dict[str, Tensor]:
    """Return the Gram spectrum and effective rank of several directions.

    Each direction is flattened over samples and features.  Directions are
    normalized to unit RMS before the Gram matrix is formed, so the spectrum
    measures redundancy rather than raw output-head scale.
    """

    values = tuple(directions)
    if not values:
        raise ValueError("at least one direction is required")
    _validate_same_shape(*values)
    columns = torch.stack([value.float().reshape(-1) for value in values], dim=1)
    rms = columns.square().mean(dim=0).sqrt().clamp_min(eps)
    normalized = columns / rms
    gram = normalized.T @ normalized / float(normalized.shape[0])
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0).flip(0)
    probabilities = eigenvalues / eigenvalues.sum().clamp_min(eps)
    nonzero = probabilities > eps
    entropy = -(probabilities[nonzero] * probabilities[nonzero].log()).sum()
    effective_rank = entropy.exp()
    participation_ratio = eigenvalues.sum().square() / eigenvalues.square().sum().clamp_min(eps)
    explained = eigenvalues.cumsum(0) / eigenvalues.sum().clamp_min(eps)
    return {
        "gram": gram,
        "eigenvalues": eigenvalues,
        "explained_variance": explained,
        "effective_rank": effective_rank,
        "participation_ratio": participation_ratio,
        "direction_rms": rms,
    }


def split_dual_output(output: object) -> tuple[Tensor, Tensor]:
    """Extract the first two tensors from an official dual-head model output."""

    if not isinstance(output, (tuple, list)) or len(output) < 2:
        raise TypeError("model must return at least (full, base)")
    full, base = output[:2]
    if not isinstance(full, Tensor) or not isinstance(base, Tensor):
        raise TypeError("full and base outputs must be tensors")
    _validate_same_shape(full, base)
    return full, base


@dataclass(frozen=True)
class RolloutResult:
    endpoint: Tensor
    step_rows: tuple[dict[str, float | int], ...]


@torch.no_grad()
def euler_ig_scale_sweep_rollout(
    model: torch.nn.Module,
    initial_state: Tensor,
    labels: Tensor,
    times: Tensor,
    scales: Sequence[float],
    *,
    mode: str = "persistent",
    active_interval: tuple[float, float] = (0.0, 1.0),
) -> Tensor:
    """Integrate several paired IG scales in one model batch.

    The returned shape is ``[num_scales, batch, ...]``.  ``persistent`` uses
    each scale at every active step; ``first_step_impulse`` uses it only at the
    first step and then restores the unmodified full prediction (scale 1).
    """

    scale_values = tuple(float(value) for value in scales)
    if not scale_values:
        raise ValueError("at least one scale is required")
    if mode not in {"persistent", "first_step_impulse"}:
        raise ValueError(f"unsupported rollout mode: {mode}")
    if times.ndim != 1 or len(times) < 2:
        raise ValueError("times must be a one-dimensional grid with at least two entries")
    if not bool(torch.all(times[:-1] > times[1:])):
        raise ValueError("times must be strictly descending")
    if labels.ndim != 1 or labels.shape[0] != initial_state.shape[0]:
        raise ValueError("labels must contain one class per sample")
    low, high = (float(value) for value in active_interval)
    if not 0.0 <= low <= high <= 1.0:
        raise ValueError("active_interval must lie in [0, 1]")

    scale_count = len(scale_values)
    batch_size = initial_state.shape[0]
    state = initial_state.repeat(scale_count, *([1] * (initial_state.ndim - 1)))
    repeated_labels = labels.repeat(scale_count)
    scale_tensor = torch.tensor(
        scale_values, device=initial_state.device, dtype=initial_state.dtype
    ).repeat_interleave(batch_size)
    grid = times.to(device=state.device, dtype=torch.float32)
    for step, (current, following) in enumerate(zip(grid[:-1], grid[1:])):
        batch_time = torch.full(
            (state.shape[0],),
            float(current),
            device=state.device,
            dtype=state.dtype,
        )
        full, base = split_dual_output(model(state, batch_time, repeated_labels))
        active = low <= float(current) <= high
        if mode == "first_step_impulse":
            active = active and step == 0
        current_scale = scale_tensor if active else torch.ones_like(scale_tensor)
        velocity = guided_prediction(full, base, current_scale)
        state = state + (following - current).to(state.dtype) * velocity
    return state.float().reshape(
        (scale_count, batch_size) + tuple(initial_state.shape[1:])
    )


@torch.no_grad()
def euler_ig_rollout(
    model: torch.nn.Module,
    initial_state: Tensor,
    labels: Tensor,
    times: Tensor,
    *,
    scale_policy: ScalePolicy,
) -> RolloutResult:
    """Integrate a dual-head velocity model over a descending Euler grid."""

    if times.ndim != 1 or len(times) < 2:
        raise ValueError("times must be a one-dimensional grid with at least two entries")
    if not bool(torch.all(times[:-1] > times[1:])):
        raise ValueError("times must be strictly descending")
    if labels.ndim != 1 or labels.shape[0] != initial_state.shape[0]:
        raise ValueError("labels must contain one class per sample")

    state = initial_state
    grid = times.to(device=state.device, dtype=torch.float32)
    rows: list[dict[str, float | int]] = []
    for step, (current, following) in enumerate(zip(grid[:-1], grid[1:])):
        batch_time = torch.full(
            (state.shape[0],),
            float(current),
            device=state.device,
            dtype=state.dtype,
        )
        full, base = split_dual_output(model(state, batch_time, labels))
        scale = scale_policy(step, current, full, base)
        velocity = guided_prediction(full, base, scale)
        direction = full.float() - base.float()
        rows.append(
            {
                "step": int(step),
                "time": float(current),
                "next_time": float(following),
                "direction_rms": float(direction.square().mean().sqrt().cpu()),
                "state_rms": float(state.float().square().mean().sqrt().cpu()),
            }
        )
        state = state + (following - current).to(state.dtype) * velocity
    return RolloutResult(endpoint=state.float(), step_rows=tuple(rows))


def fixed_scale_policy(
    scale: float,
    *,
    active_interval: tuple[float, float] = (0.0, 1.0),
) -> ScalePolicy:
    """Return a policy that applies one fixed IG scale in a time interval."""

    low, high = (float(value) for value in active_interval)
    if not 0.0 <= low <= high <= 1.0:
        raise ValueError("active_interval must lie in [0, 1]")

    def policy(step: int, time: Tensor, full: Tensor, base: Tensor) -> float:
        del step, full, base
        value = float(time)
        return float(scale) if low <= value <= high else 1.0

    return policy


def first_step_impulse_policy(scale: float) -> ScalePolicy:
    """Apply an IG scale only to the first Euler step, then return to full."""

    def policy(step: int, time: Tensor, full: Tensor, base: Tensor) -> float:
        del time, full, base
        return float(scale) if step == 0 else 1.0

    return policy
