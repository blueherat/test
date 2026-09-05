"""Split RAEv2 depth guidance into radius and direction changes.

The reference radius is the full head's posterior-mean radius at the current
state.  It is a measurable null hypothesis, not a claim that clean RAE latents
lie on a sphere: the selected DINOv3 encoder averages seven normalized layer
outputs and adds the final layer's spatial token mean; RAE then applies
anisotropic normalization statistics.
All arithmetic is FP32, including subtraction of the two clean predictions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


MODES = ("ordinary", "tangent", "radial", "retracted")
GROUPINGS = ("token", "global")


def _group_dims(grouping: str) -> tuple[int, ...]:
    if grouping == "token":
        return (1,)
    if grouping == "global":
        return (1, 2, 3)
    raise ValueError(f"unknown radius grouping: {grouping}")


def _require_finite(value: Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


@dataclass(frozen=True)
class DepthDirectionDecomposition:
    full: Tensor
    direction: Tensor
    radial: Tensor
    tangent: Tensor
    reference_squared_norm: Tensor
    direction_squared_norm: Tensor
    degenerate_reference: Tensor


def decompose_depth_direction(
    full_clean: Tensor,
    base_clean: Tensor,
    *,
    grouping: str = "token",
    check_finite: bool = True,
) -> DepthDirectionDecomposition:
    """Orthogonally split ``F-B`` along ``F`` within each requested group.

    Inputs have shape ``[batch, channels, height, width]``. Token groups use
    channels only; global groups use all three non-batch dimensions. At an
    exactly zero reference the radial component is zero and the tangent is
    the complete depth difference. No epsilon changes nonzero projections.
    """

    dims = _group_dims(grouping)
    if full_clean.shape != base_clean.shape or full_clean.ndim != 4:
        raise ValueError("predictions must share [batch, channels, height, width]")
    if full_clean.device != base_clean.device:
        raise ValueError("predictions must be on the same device")
    if not full_clean.is_floating_point() or not base_clean.is_floating_point():
        raise ValueError("predictions must be floating-point tensors")
    if any(size == 0 for size in full_clean.shape):
        raise ValueError("prediction dimensions must be nonempty")
    full = full_clean.float()
    base = base_clean.float()
    if check_finite:
        _require_finite(full, "full prediction")
        _require_finite(base, "base prediction")
    direction = full - base
    reference_squared_norm = full.square().sum(dim=dims, keepdim=True)
    direction_squared_norm = direction.square().sum(dim=dims, keepdim=True)
    degenerate = reference_squared_norm == 0
    denominator = torch.where(
        degenerate, torch.ones_like(reference_squared_norm), reference_squared_norm
    )
    coefficient = (direction * full).sum(dim=dims, keepdim=True) / denominator
    radial = coefficient * full
    tangent = direction - radial
    if check_finite:
        for name, value in (
            ("reference squared norm", reference_squared_norm),
            ("direction squared norm", direction_squared_norm),
            ("radial direction", radial),
            ("tangent direction", tangent),
        ):
            _require_finite(value, name)
    return DepthDirectionDecomposition(
        full=full,
        direction=direction,
        radial=radial,
        tangent=tangent,
        reference_squared_norm=reference_squared_norm,
        direction_squared_norm=direction_squared_norm,
        degenerate_reference=degenerate,
    )


def radius_guided_clean(
    full_clean: Tensor,
    base_clean: Tensor,
    *,
    guidance_scale: float,
    mode: str = "ordinary",
    grouping: str = "token",
    return_telemetry: bool = False,
    check_finite: bool = True,
) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
    """Apply one radius intervention with the established IG coefficient.

    Write ``g = guidance_scale - 1`` and ``d = F-B``. Ordinary guidance is
    ``F + g*d``; tangent and radial use their respective parts of ``d``.
    Retraction is the closest point to ordinary guidance at radius ``||F||``.
    If ordinary guidance is exactly zero with nonzero ``F``, every point on
    that sphere is equally close; choose ``F``. Every mode falls back to
    ordinary guidance when ``F`` is zero and returns ``F`` exactly at scale 1.

    Optional telemetry consists of detached FP32 per-group tensors, retaining
    singleton reduction dimensions. Undefined zero-reference radius ratios
    use 1 and undefined cosines use 0; explicit masks identify those groups.
    ``check_finite=False`` omits synchronizing finite checks in an already
    validated sampling loop without changing any numerical operation.
    """

    if mode not in MODES:
        raise ValueError(f"unknown radius guidance mode: {mode}")
    if not math.isfinite(guidance_scale) or guidance_scale < 0:
        raise ValueError("guidance_scale must be finite and non-negative")
    split = decompose_depth_direction(
        full_clean, base_clean, grouping=grouping, check_finite=check_finite
    )
    dims = _group_dims(grouping)
    full = split.full
    gain = guidance_scale - 1.0
    ordinary = full + gain * split.direction
    ordinary_squared_norm = ordinary.square().sum(dim=dims, keepdim=True)
    zero_candidate = ordinary_squared_norm == 0
    if guidance_scale == 1.0:
        guided = full
    elif mode == "ordinary":
        guided = ordinary
    elif mode == "tangent":
        guided = full + gain * split.tangent
    elif mode == "radial":
        guided = full + gain * split.radial
    else:
        denominator = torch.where(
            zero_candidate, torch.ones_like(ordinary_squared_norm), ordinary_squared_norm
        )
        scale = split.reference_squared_norm.sqrt() / denominator.sqrt()
        guided = torch.where(zero_candidate, full, ordinary * scale)
    guided = torch.where(split.degenerate_reference, ordinary, guided)
    if check_finite:
        _require_finite(ordinary_squared_norm, "ordinary squared norm")
        _require_finite(guided, "guided prediction")
    if not return_telemetry:
        return guided

    reference_norm = split.reference_squared_norm.sqrt()
    guided_norm = guided.square().sum(dim=dims, keepdim=True).sqrt()
    safe_reference = torch.where(
        split.degenerate_reference, torch.ones_like(reference_norm), reference_norm
    )
    zero_direction = split.direction_squared_norm == 0
    safe_direction_energy = torch.where(
        zero_direction,
        torch.ones_like(split.direction_squared_norm),
        split.direction_squared_norm,
    )
    cosine_denominator = reference_norm * guided_norm
    zero_cosine = cosine_denominator == 0
    safe_cosine_denominator = torch.where(
        zero_cosine, torch.ones_like(cosine_denominator), cosine_denominator
    )
    telemetry = {
        "radial_energy_fraction": split.radial.square().sum(dim=dims, keepdim=True)
        / safe_direction_energy,
        "relative_radius_ratio": torch.where(
            split.degenerate_reference,
            torch.ones_like(reference_norm),
            guided_norm / safe_reference,
        ),
        "cosine": torch.where(
            zero_cosine,
            torch.zeros_like(reference_norm),
            (full * guided).sum(dim=dims, keepdim=True) / safe_cosine_denominator,
        ).clamp(-1.0, 1.0),
        "degenerate_reference": split.degenerate_reference.float(),
        "degenerate_ordinary": zero_candidate.float(),
        "zero_depth_direction": zero_direction.float(),
    }
    if check_finite:
        for name, value in telemetry.items():
            _require_finite(value, name)
    return guided, {name: value.detach() for name, value in telemetry.items()}
