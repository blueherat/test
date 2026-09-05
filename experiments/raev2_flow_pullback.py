"""Frozen finite-horizon covector pullbacks for RAEv2 guidance experiments.

The full predictor defines a differentiable Euler map ``Phi``.  The future
full-minus-base clean gap is evaluated at ``Phi(state)`` and then detached.
Its pullback is ``D Phi(state).T @ future_gap``.  At a fixed current Euclidean
norm this direction maximizes the *local linear future-gap work*.  That
identity is not a claim about image quality or FID, nor does it require the
network gap to be the gradient of a global probability density.

All model parameters and model training/evaluation flags are left untouched.
Callers should supply a deterministic model in evaluation mode.  Differentiation
targets only the detached current state and never accumulates parameter grads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class FlowPullbackResult:
    """Detached directions and unnormalized fields for independent audits."""

    direction: Tensor
    raw_future_direction: Tensor
    pullback: Tensor
    future_gap: Tensor
    telemetry: dict[str, Any]


def _check_batch_tensor(value: Tensor, name: str) -> None:
    if not value.is_floating_point() or value.ndim < 2 or value.shape[0] == 0:
        raise ValueError(f"{name} must be a nonempty floating batch tensor")


def _statistics_tensor(value: Tensor) -> Tensor:
    return value if value.dtype == torch.float64 else value.float()


def _sample_norm(value: Tensor) -> Tensor:
    return torch.linalg.vector_norm(_statistics_tensor(value).flatten(1), dim=1)


def _expand_sample(value: Tensor, reference: Tensor) -> Tensor:
    return value.reshape(len(reference), *([1] * (reference.ndim - 1)))


def normalize_like(direction: Tensor, reference: Tensor, *, tiny: float = 1e-12) -> Tensor:
    """Give each direction the corresponding reference's Euclidean norm.

    A direction of norm at most ``tiny`` falls back to the reference itself;
    this makes a degenerate pullback preserve ordinary guidance.  A zero
    reference remains zero.  Calculations use at least float32 precision and
    the result has the reference dtype.  No batch statistics are mixed.
    """

    _check_batch_tensor(direction, "direction")
    _check_batch_tensor(reference, "reference")
    if direction.shape != reference.shape or direction.device != reference.device:
        raise ValueError("direction and reference must have identical shapes and devices")
    if not math.isfinite(tiny) or tiny <= 0.0:
        raise ValueError("tiny must be finite and positive")
    dtype = torch.promote_types(direction.dtype, reference.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    working = direction.to(dtype=dtype)
    reference_working = reference.to(dtype=dtype)
    direction_norm = torch.linalg.vector_norm(working.flatten(1), dim=1)
    reference_norm = torch.linalg.vector_norm(reference_working.flatten(1), dim=1)
    scaled = working * _expand_sample(
        reference_norm / direction_norm.clamp_min(tiny), working
    )
    result = torch.where(
        _expand_sample(direction_norm > tiny, working), scaled, reference_working
    )
    return result.to(dtype=reference.dtype)


def frozen_covector_vjp(endpoint: Tensor, source: Tensor, covector: Tensor) -> Tensor:
    """Return ``D(endpoint)/D(source).T @ stopgrad(covector)`` once.

    ``endpoint`` must retain its graph with respect to ``source``.  This
    low-level helper rejects inference tensors; ``flow_pullback_direction``
    prepares ordinary detached clones even when called inside inference mode.
    The returned tensor has no derivative graph.  Existing parameter grads
    are neither cleared nor changed.
    """

    if torch.is_inference(source) or torch.is_inference(endpoint):
        raise ValueError("VJP requires ordinary tensors, not inference tensors")
    if endpoint.shape != covector.shape or endpoint.device != covector.device:
        raise ValueError("endpoint and covector must have identical shapes and devices")
    if not source.requires_grad or not endpoint.requires_grad:
        raise ValueError("source and endpoint must retain a differentiable flow graph")
    with torch.enable_grad():
        return torch.autograd.grad(
            outputs=endpoint,
            inputs=source,
            grad_outputs=covector.detach().to(dtype=endpoint.dtype),
            create_graph=False,
            retain_graph=False,
        )[0].detach()


def _dual_prediction(model: nn.Module, state: Tensor, times: Tensor, labels: Tensor):
    output = model(state, times, context=labels, attn_mask=None)
    if not isinstance(output, (tuple, list)) or len(output) != 2:
        raise TypeError("RAEv2 model must return (full_clean, base_clean)")
    full, base = output
    if full.shape != state.shape or base.shape != state.shape:
        raise ValueError("both model heads must match the state shape")
    return full, base


def _validate_flow(
    state: Tensor,
    labels: Tensor,
    start_time: float,
    end_time: float,
    substeps: int,
    denominator_floor: float,
) -> None:
    _check_batch_tensor(state, "state")
    if labels.shape != (len(state),) or labels.device != state.device:
        raise ValueError("labels must contain one entry per sample on the state device")
    if isinstance(substeps, bool) or not isinstance(substeps, int) or substeps < 1:
        raise ValueError("substeps must be a positive integer")
    if not (math.isfinite(start_time) and math.isfinite(end_time)):
        raise ValueError("flow times must be finite")
    if not 0.0 <= end_time <= start_time <= 1.0:
        raise ValueError("flow times must satisfy 0 <= end_time <= start_time <= 1")
    if not math.isfinite(denominator_floor) or denominator_floor <= 0.0:
        raise ValueError("denominator_floor must be finite and positive")


def full_euler_flow(
    model: nn.Module,
    state: Tensor,
    labels: Tensor,
    start_time: float,
    end_time: float,
    substeps: int,
    denominator_floor: float = 0.05,
    checkpoint_forward: bool = True,
) -> Tensor:
    """Integrate the full clean predictor dataward on a uniform raw-time grid.

    Each step uses ``z += dt * (z - full_clean) / max(t, floor)`` with
    negative ``dt``.  All samples receive the same fixed scalar time, stored
    as a float32 batch vector as in the RAEv2 sampler.  Non-reentrant
    activation checkpointing recomputes the full prediction during the VJP.
    Time and labels are explicit checkpoint arguments, avoiding late-bound
    loop variables during recomputation.  The caller owns the source graph.
    """

    _validate_flow(state, labels, start_time, end_time, substeps, denominator_floor)
    if torch.is_inference_mode_enabled() or torch.is_inference(state):
        raise ValueError("use flow_pullback_direction to prepare tensors outside inference mode")
    if start_time == end_time:
        return state
    delta = (float(end_time) - float(start_time)) / substeps

    def full_prediction(value: Tensor, times: Tensor, context: Tensor) -> Tensor:
        full, _ = _dual_prediction(model, value, times, context)
        return full.to(dtype=value.dtype)

    with torch.enable_grad():
        current_state = state
        for index in range(substeps):
            current = float(start_time) + index * delta
            times = torch.full(
                (len(state),), current, device=state.device, dtype=torch.float32
            )
            if checkpoint_forward:
                full = checkpoint(
                    full_prediction, current_state, times, labels, use_reentrant=False
                )
            else:
                full = full_prediction(current_state, times, labels)
            denominator = _expand_sample(
                times.to(dtype=current_state.dtype).clamp_min(denominator_floor),
                current_state,
            )
            current_state = current_state + delta * (current_state - full) / denominator
    return current_state


def _sample_cosine(left: Tensor, right: Tensor, tiny: float) -> Tensor:
    left = _statistics_tensor(left).flatten(1)
    right = _statistics_tensor(right).flatten(1)
    denominator = (
        torch.linalg.vector_norm(left, dim=1) * torch.linalg.vector_norm(right, dim=1)
    )
    return (left * right).sum(dim=1) / denominator.clamp_min(tiny)


def flow_pullback_direction(
    model: nn.Module,
    state: Tensor,
    labels: Tensor,
    current_gap: Tensor,
    start_time: float,
    end_time: float,
    substeps: int,
    denominator_floor: float = 0.05,
    checkpoint_forward: bool = True,
    *,
    tiny: float = 1e-12,
) -> FlowPullbackResult:
    """Return norm-matched pullback, norm-matched raw future gap, and telemetry.

    Both returned directions have each sample's ``current_gap`` norm.  Neither
    is multiplied by the guidance scale.  Telemetry contains detached per-
    sample tensors on the input device, so the caller can aggregate as needed.
    The future gap is frozen before the single input-only VJP: differentiating
    it would instead optimize a different quadratic gap objective.

    This wrapper supports outer ``no_grad`` and ``inference_mode`` contexts by
    cloning state, labels, and reference after locally disabling inference.
    Returned tensors are ordinary detached tensors, not live autograd graphs.
    Near-collinearity at a short horizon only rejects useful rotation at that
    horizon; it says nothing about the much longer remaining flow interval.
    """

    _validate_flow(state, labels, start_time, end_time, substeps, denominator_floor)
    if current_gap.shape != state.shape or current_gap.device != state.device:
        raise ValueError("current_gap must match the state shape and device")
    _check_batch_tensor(current_gap, "current_gap")
    with torch.inference_mode(False), torch.enable_grad():
        source = state.detach().clone().requires_grad_(True)
        context = labels.detach().clone()
        reference = current_gap.detach().clone()
        endpoint = full_euler_flow(
            model,
            source,
            context,
            start_time,
            end_time,
            substeps,
            denominator_floor,
            checkpoint_forward,
        )
        with torch.no_grad():
            future_times = torch.full(
                (len(source),), end_time, device=source.device, dtype=torch.float32
            )
            future_full, future_base = _dual_prediction(
                model, endpoint.detach(), future_times, context
            )
            # Cast before subtraction; bf16 subtraction can destroy small gaps.
            future_gap = future_full.to(source.dtype) - future_base.to(source.dtype)
        pulled = frozen_covector_vjp(endpoint, source, future_gap)
        with torch.no_grad():
            pullback_direction = normalize_like(pulled, reference, tiny=tiny)
            raw_future_direction = normalize_like(future_gap, reference, tiny=tiny)
            reference_norm = _sample_norm(reference)
            future_norm = _sample_norm(future_gap)
            pulled_norm = _sample_norm(pulled)
            dimension_root = math.sqrt(reference[0].numel())
            telemetry = {
                "start_time": float(start_time),
                "end_time": float(end_time),
                "substeps": substeps,
                "checkpoint_forward": bool(checkpoint_forward),
                "current_gap_rms": reference_norm / dimension_root,
                "future_gap_rms": future_norm / dimension_root,
                "pullback_rms": pulled_norm / dimension_root,
                "pullback_gain": pulled_norm / future_norm.clamp_min(tiny),
                "pullback_raw_cosine": _sample_cosine(pulled, future_gap, tiny),
                "pullback_current_cosine": _sample_cosine(pulled, reference, tiny),
                "future_current_cosine": _sample_cosine(future_gap, reference, tiny),
                "pullback_fallback": pulled_norm <= tiny,
                "future_fallback": future_norm <= tiny,
            }
    return FlowPullbackResult(
        direction=pullback_direction.detach(),
        raw_future_direction=raw_future_direction.detach(),
        pullback=pulled.detach(),
        future_gap=future_gap.detach(),
        telemetry=telemetry,
    )
