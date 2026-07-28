"""Coordinate-aware conditional paths for latent flow matching.

The convention matches the RAE transport implementation: ``t=0`` is the data
endpoint and ``t=1`` is the source endpoint.  The four branches intentionally
separate source-prior mismatch from the failure of nonlinear maps to commute
with straight interpolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import torch
import torch.nn as nn


TransportBranch = Literal[
    "base",
    "gaussian_straight",
    "matched_chord",
    "pushforward",
]
TensorMap = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class ConditionalPathSample:
    """One paired conditional path sample and its exact target velocity."""

    branch: TransportBranch
    time: torch.Tensor
    state: torch.Tensor
    velocity: torch.Tensor
    data_endpoint: torch.Tensor
    source_endpoint: torch.Tensor


def _expand_time(time: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if time.ndim == 0:
        time = time.expand(reference.shape[0])
    if time.ndim != 1 or time.shape[0] != reference.shape[0]:
        raise ValueError(
            "time must be scalar or shaped [batch], got "
            f"{tuple(time.shape)} for batch {reference.shape[0]}"
        )
    return time.to(device=reference.device, dtype=reference.dtype).view(
        reference.shape[0], *([1] * (reference.ndim - 1))
    )


def _check_pair(data: torch.Tensor, base_noise: torch.Tensor) -> None:
    if data.shape != base_noise.shape:
        raise ValueError(
            f"data and base_noise must have equal shape, got {data.shape} and {base_noise.shape}"
        )
    if not data.is_floating_point() or not base_noise.is_floating_point():
        raise TypeError("conditional paths require floating-point tensors")
    if data.device != base_noise.device:
        raise ValueError("data and base_noise must be on the same device")


def identity_map(value: torch.Tensor) -> torch.Tensor:
    return value


class ScaledAdditiveCouplingTransform(nn.Module):
    """Continuously scale additive coupling updates while preserving inversion."""

    def __init__(self, adapter: nn.Module, scale: float):
        super().__init__()
        if not 0.0 <= float(scale) <= 1.0:
            raise ValueError(f"scale must lie in [0, 1], got {scale}")
        if not hasattr(adapter, "blocks"):
            raise TypeError("adapter must expose additive coupling blocks")
        self.adapter = adapter
        self.scale = float(scale)

    @staticmethod
    def _forward_block(block: nn.Module, value: torch.Tensor, scale: float) -> torch.Tensor:
        if not all(hasattr(block, name) for name in ("_split", "_merge", "net")):
            raise TypeError("each adapter block must expose _split, _merge and net")
        first, second = block._split(value)
        return block._merge(first, second + float(scale) * block.net(first))

    @staticmethod
    def _inverse_block(block: nn.Module, value: torch.Tensor, scale: float) -> torch.Tensor:
        first, second = block._split(value)
        return block._merge(first, second - float(scale) * block.net(first))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output = value
        for block in self.adapter.blocks:
            output = self._forward_block(block, output, self.scale)
        return output

    def inverse(self, value: torch.Tensor) -> torch.Tensor:
        output = value
        for block in reversed(self.adapter.blocks):
            output = self._inverse_block(block, output, self.scale)
        return output


def conditional_path_sample(
    data: torch.Tensor,
    base_noise: torch.Tensor,
    time: torch.Tensor,
    *,
    branch: TransportBranch,
    transform: TensorMap | None = None,
    gaussian_noise: torch.Tensor | None = None,
) -> ConditionalPathSample:
    """Construct one of the four preregistered latent transport branches.

    ``base_noise`` is always the noise paired with ``data`` in the original
    latent coordinates.  ``gaussian_noise`` is used only by
    ``gaussian_straight``; when omitted, the same numerical tensor as
    ``base_noise`` is used to make paired comparisons deterministic.
    """

    _check_pair(data, base_noise)
    transform = identity_map if transform is None else transform
    expanded_time = _expand_time(time, data)

    if branch == "base":
        data_endpoint = data
        source_endpoint = base_noise
        state = (1.0 - expanded_time) * data_endpoint + expanded_time * source_endpoint
        velocity = source_endpoint - data_endpoint
    elif branch == "gaussian_straight":
        if gaussian_noise is None:
            gaussian_noise = base_noise
        _check_pair(data, gaussian_noise)
        data_endpoint = transform(data)
        source_endpoint = gaussian_noise
        state = (1.0 - expanded_time) * data_endpoint + expanded_time * source_endpoint
        velocity = source_endpoint - data_endpoint
    elif branch == "matched_chord":
        data_endpoint = transform(data)
        source_endpoint = transform(base_noise)
        state = (1.0 - expanded_time) * data_endpoint + expanded_time * source_endpoint
        velocity = source_endpoint - data_endpoint
    elif branch == "pushforward":
        base_state = (1.0 - expanded_time) * data + expanded_time * base_noise
        base_velocity = base_noise - data
        state, velocity = torch.func.jvp(
            transform,
            (base_state,),
            (base_velocity,),
        )
        data_endpoint = transform(data)
        source_endpoint = transform(base_noise)
    else:
        raise ValueError(f"unknown transport branch: {branch}")

    return ConditionalPathSample(
        branch=branch,
        time=time,
        state=state,
        velocity=velocity,
        data_endpoint=data_endpoint,
        source_endpoint=source_endpoint,
    )


def relative_l2_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have equal shape")
    numerator = (prediction - target).flatten(1).square().sum(dim=1)
    denominator = target.flatten(1).square().sum(dim=1).clamp_min(float(eps))
    return torch.sqrt(numerator / denominator)


def bridge_commutation_defect(
    data: torch.Tensor,
    base_noise: torch.Tensor,
    time: torch.Tensor,
    transform: TensorMap,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Return ``||f(z_t) - chord(f(z), f(eps))|| / ||f(z_t)||`` per sample."""

    _check_pair(data, base_noise)
    expanded_time = _expand_time(time, data)
    pushforward_state = transform(
        (1.0 - expanded_time) * data + expanded_time * base_noise
    )
    chord_state = (
        (1.0 - expanded_time) * transform(data)
        + expanded_time * transform(base_noise)
    )
    return relative_l2_per_sample(chord_state, pushforward_state, eps=eps)


def finite_difference_jvp(
    transform: TensorMap,
    point: torch.Tensor,
    direction: torch.Tensor,
    *,
    step: float = 1e-3,
) -> torch.Tensor:
    """Central finite-difference reference for a transform JVP."""

    if point.shape != direction.shape:
        raise ValueError("point and direction must have equal shape")
    if step <= 0:
        raise ValueError("step must be positive")
    return (
        transform(point + float(step) * direction)
        - transform(point - float(step) * direction)
    ) / (2.0 * float(step))


def jvp_relative_error(
    transform: TensorMap,
    point: torch.Tensor,
    direction: torch.Tensor,
    *,
    step: float = 1e-3,
    eps: float = 1e-12,
) -> torch.Tensor:
    _, exact = torch.func.jvp(transform, (point,), (direction,))
    reference = finite_difference_jvp(transform, point, direction, step=step)
    return relative_l2_per_sample(exact, reference, eps=eps)
