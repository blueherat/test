"""FP32 controls for the condition and depth axes of RAEv2 guidance.

The additive and bilinear controls are established algebra, not a proposed
new method. ``semantic_orthogonal`` tests whether the part of the full CFG
direction orthogonal to conditional IG adds useful information. The projection
is per image, in the model's normalized latent Euclidean metric, with no norm
restoration. Calling the axes semantic/quality is a hypothesis to test, not an
assumption that they are independent or that IG only affects visual quality.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


MODES = ("ordinary", "full_cfg", "combined", "semantic_orthogonal", "bilinear")


def _validated_predictions(*predictions: Tensor, check_finite: bool) -> tuple[Tensor, ...]:
    reference = predictions[0]
    if reference.ndim < 2 or any(size == 0 for size in reference.shape):
        raise ValueError("predictions must have a nonempty batch and feature dimensions")
    for value in predictions:
        if value.shape != reference.shape or value.device != reference.device:
            raise ValueError("all four predictions must share shape and device")
        if not value.is_floating_point():
            raise ValueError("predictions must be floating-point tensors")
    converted = tuple(value.float() for value in predictions)
    if check_finite and any(not bool(torch.isfinite(value).all()) for value in converted):
        raise ValueError("predictions must be finite")
    return converted


def semantic_orthogonal_residual(ig_direction: Tensor, cfg_direction: Tensor) -> Tensor:
    """Remove the per-image CFG projection on IG, entirely in FP32.

    A zero IG direction imposes no constraint and leaves CFG intact. No epsilon
    shrinks a nonzero projection, and no norm rescaling magnifies the residual.
    """

    ig, cfg = _validated_predictions(ig_direction, cfg_direction, check_finite=False)
    dims = tuple(range(1, ig.ndim))
    energy = ig.square().sum(dim=dims, keepdim=True)
    denominator = torch.where(energy == 0, torch.ones_like(energy), energy)
    coefficient = (ig * cfg).sum(dim=dims, keepdim=True) / denominator
    return cfg - coefficient * ig


def semantic_quality_clean(
    full_conditional: Tensor,
    base_conditional: Tensor,
    full_unconditional: Tensor,
    base_unconditional: Tensor,
    *,
    ig_scale: float,
    cfg_scale: float,
    mode: str = "combined",
    return_telemetry: bool = False,
    check_finite: bool = True,
) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
    """Evaluate an intervention about the full conditional clean prediction.

    Write ``g=ig_scale-1``, ``s=cfg_scale-1``, ``D=F_c-B_c``,
    ``C=F_c-F_u`` and ``I=F_c-B_c-F_u+B_u``. The modes are:

    * ordinary: ``F_c + g D`` (CFG scale ignored);
    * full_cfg: ``F_c + s C`` (IG scale ignored);
    * combined: ``F_c + g D + s C``;
    * semantic_orthogonal: ``F_c + g D + s (C-Proj_D C)``;
    * bilinear: ``F_c + g D + s C + g s I``.

    Bilinear is the unique two-axis extension through the four predictions at
    (ig,cfg)=(0,0), (0,1), (1,0), (1,1). The anchored expression preserves the
    exact ordinary IG result at cfg=1 and the full CFG result at ig=1. An axis
    outside its sampling time window must be passed as scale 1.
    """

    if mode not in MODES:
        raise ValueError(f"unknown semantic/quality guidance mode: {mode}")
    for name, scale in (("ig_scale", ig_scale), ("cfg_scale", cfg_scale)):
        if not math.isfinite(scale) or scale < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    fc, bc, fu, bu = _validated_predictions(
        full_conditional, base_conditional, full_unconditional, base_unconditional,
        check_finite=check_finite,
    )
    depth = fc - bc
    semantic = fc - fu
    interaction = depth - (fu - bu)
    ig_gain = 0.0 if mode == "full_cfg" else ig_scale - 1.0
    cfg_gain = 0.0 if mode == "ordinary" else cfg_scale - 1.0
    orthogonal = None
    if mode == "semantic_orthogonal" or return_telemetry:
        orthogonal = semantic_orthogonal_residual(depth, semantic)
    semantic_used = orthogonal if mode == "semantic_orthogonal" else semantic
    clean = fc + ig_gain * depth + cfg_gain * semantic_used
    if mode == "bilinear":
        clean = clean + (ig_gain * cfg_gain) * interaction
    if check_finite and not bool(torch.isfinite(clean).all()):
        raise ValueError("guided clean prediction is not finite")
    if not return_telemetry:
        return clean

    assert orthogonal is not None
    dims = tuple(range(1, fc.ndim))
    depth_energy = depth.square().sum(dim=dims)
    semantic_energy = semantic.square().sum(dim=dims)
    orthogonal_energy = orthogonal.square().sum(dim=dims)
    norm_product = (depth_energy * semantic_energy).sqrt()
    safe_product = torch.where(norm_product == 0, torch.ones_like(norm_product), norm_product)
    safe_semantic = torch.where(semantic_energy == 0, torch.ones_like(semantic_energy), semantic_energy)
    telemetry = {
        "ig_rms": depth.square().mean(dim=dims).sqrt(),
        "cfg_rms": semantic.square().mean(dim=dims).sqrt(),
        "cfg_orthogonal_rms": orthogonal.square().mean(dim=dims).sqrt(),
        "interaction_rms": interaction.square().mean(dim=dims).sqrt(),
        "cos_ig_cfg": (depth * semantic).sum(dim=dims) / safe_product,
        "cfg_orthogonal_energy_fraction": orthogonal_energy / safe_semantic,
        "zero_ig_fraction": (depth_energy == 0).float(),
        "zero_cfg_fraction": (semantic_energy == 0).float(),
        "correction_rms": (clean - fc).square().mean(dim=dims).sqrt(),
    }
    return clean, {key: value.detach() for key, value in telemetry.items()}
