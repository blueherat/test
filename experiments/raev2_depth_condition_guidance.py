"""Depth-by-condition decompositions for RAEv2 internal guidance.

The RAEv2 IG checkpoint exposes two depth levels (base/full) and was trained
with conditional dropout, so the same state admits four clean predictions::

    base_uncond, base_cond, full_uncond, full_cond.

These four corners have a unique two-axis Mobius decomposition.  Keeping the
algebra in one small module makes the sampling experiment auditable and avoids
silently changing the established IG scale convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class DepthConditionComponents:
    """The four coefficients of the bilinear depth/condition table."""

    base_unconditional: Tensor
    depth_main: Tensor
    condition_main: Tensor
    interaction: Tensor


def minimum_norm_convex_consensus(
    first: Tensor,
    second: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return the minimum-norm point between two per-sample directions.

    The returned ``weight`` is the coefficient of ``second`` in
    ``(1 - weight) * first + weight * second``.  This is the closed-form
    two-objective MGDA solution and has no learned or swept coefficient.
    """

    if first.shape != second.shape:
        raise ValueError("consensus directions must have identical shapes")
    if first.ndim < 2:
        raise ValueError("consensus directions must include a batch dimension")
    first_flat = first.flatten(1).float()
    delta_flat = second.flatten(1).float() - first_flat
    denominator = delta_flat.square().sum(dim=1)
    numerator = -(first_flat * delta_flat).sum(dim=1)
    weight = torch.where(
        denominator > 0,
        (numerator / denominator).clamp(0.0, 1.0),
        torch.zeros_like(denominator),
    )
    view_shape = (first.shape[0],) + (1,) * (first.ndim - 1)
    consensus = first + weight.to(first.dtype).view(view_shape) * (second - first)
    return consensus, weight


def orthogonal_residual(reference: Tensor, candidate: Tensor) -> Tensor:
    """Return the per-sample part of ``candidate`` orthogonal to ``reference``."""

    if reference.shape != candidate.shape:
        raise ValueError("projection tensors must have identical shapes")
    if reference.ndim < 2:
        raise ValueError("projection tensors must include a batch dimension")
    reference_flat = reference.flatten(1).float()
    candidate_flat = candidate.flatten(1).float()
    denominator = reference_flat.square().sum(dim=1)
    coefficient = torch.where(
        denominator > 0,
        (reference_flat * candidate_flat).sum(dim=1) / denominator,
        torch.zeros_like(denominator),
    )
    view_shape = (reference.shape[0],) + (1,) * (reference.ndim - 1)
    return candidate - coefficient.to(candidate.dtype).view(view_shape) * reference


def matched_donor_orthogonal(reference: Tensor, residual: Tensor) -> Tensor:
    """Break sample association while preserving orthogonality and per-sample norm."""

    if reference.shape != residual.shape:
        raise ValueError("donor tensors must have identical shapes")
    if reference.ndim < 2:
        raise ValueError("donor tensors must include a batch dimension")
    if reference.shape[0] < 2:
        raise ValueError("donor control requires at least two samples per batch")
    donor = orthogonal_residual(reference, residual.roll(shifts=1, dims=0))
    residual_norm = residual.flatten(1).float().norm(dim=1)
    donor_norm = donor.flatten(1).float().norm(dim=1)
    scale = torch.where(
        donor_norm > 0,
        residual_norm / donor_norm,
        torch.zeros_like(donor_norm),
    )
    view_shape = (reference.shape[0],) + (1,) * (reference.ndim - 1)
    return donor * scale.to(donor.dtype).view(view_shape)


def mobius_components(
    *,
    full_conditional: Tensor,
    base_conditional: Tensor,
    full_unconditional: Tensor,
    base_unconditional: Tensor,
) -> DepthConditionComponents:
    """Return the unique Mobius decomposition of the four predictions.

    ``depth_main`` and ``condition_main`` are measured at the unconditional
    base corner.  ``interaction`` is the mixed finite difference and therefore
    exactly measures how much the conditioning effect changes with depth.
    """

    shape = full_conditional.shape
    if any(
        value.shape != shape
        for value in (base_conditional, full_unconditional, base_unconditional)
    ):
        raise ValueError("all four predictions must have identical shapes")

    depth_main = full_unconditional - base_unconditional
    condition_main = base_conditional - base_unconditional
    interaction = (
        full_conditional
        - full_unconditional
        - base_conditional
        + base_unconditional
    )
    return DepthConditionComponents(
        base_unconditional=base_unconditional,
        depth_main=depth_main,
        condition_main=condition_main,
        interaction=interaction,
    )


def reconstruct_corner(
    components: DepthConditionComponents,
    *,
    depth_coordinate: float,
    condition_coordinate: float,
) -> Tensor:
    """Evaluate the unique bilinear extension through the four corners."""

    return (
        components.base_unconditional
        + depth_coordinate * components.depth_main
        + condition_coordinate * components.condition_main
        + depth_coordinate * condition_coordinate * components.interaction
    )


def guidance_direction(
    *,
    full_conditional: Tensor,
    base_conditional: Tensor,
    full_unconditional: Tensor,
    base_unconditional: Tensor,
    mode: str,
) -> Tensor:
    """Return one explicitly named depth/condition guidance direction.

    ``conditional_depth`` is ordinary RAEv2 IG. ``marginal_depth`` removes
    the depth-by-condition interaction and uses only the class-independent
    strong-minus-weak correction. ``interaction`` isolates the removed term.
    ``conditional_marginal_midpoint`` is their fixed midpoint, while
    ``conditional_marginal_consensus`` is their per-sample minimum-norm
    convex combination. The three ``orthogonal`` modes preserve the ordinary
    conditional-depth component and add, subtract, or donor-swap the marginal
    information orthogonal to it.
    """

    components = mobius_components(
        full_conditional=full_conditional,
        base_conditional=base_conditional,
        full_unconditional=full_unconditional,
        base_unconditional=base_unconditional,
    )
    if mode == "conditional_depth":
        return full_conditional - base_conditional
    if mode == "marginal_depth":
        return components.depth_main
    if mode == "interaction":
        return components.interaction
    if mode == "conditional_marginal_midpoint":
        return components.depth_main + 0.5 * components.interaction
    if mode == "conditional_marginal_consensus":
        conditional_depth = full_conditional - base_conditional
        consensus, _ = minimum_norm_convex_consensus(
            conditional_depth,
            components.depth_main,
        )
        return consensus
    if mode.startswith("conditional_marginal_orthogonal"):
        conditional_depth = full_conditional - base_conditional
        residual = orthogonal_residual(
            conditional_depth,
            components.depth_main,
        )
        if mode == "conditional_marginal_orthogonal_positive":
            return conditional_depth + residual
        if mode == "conditional_marginal_orthogonal_negative":
            return conditional_depth - residual
        if mode == "conditional_marginal_orthogonal_donor":
            return conditional_depth + matched_donor_orthogonal(
                conditional_depth,
                residual,
            )
    raise ValueError(f"unknown guidance mode: {mode}")


def guided_clean_prediction(
    *,
    full_conditional: Tensor,
    base_conditional: Tensor,
    full_unconditional: Tensor,
    base_unconditional: Tensor,
    guidance_scale: float,
    mode: str,
) -> Tensor:
    """Apply an IG-style extrapolation while retaining the strong anchor.

    RAEv2 names the complete interpolation coefficient ``guidance_scale``:
    ordinary IG is ``B + beta(F-B) = F + (beta-1)(F-B)``.  All modes here
    preserve that convention and therefore use ``beta - 1`` beyond the full
    conditional prediction.
    """

    if guidance_scale < 0.0:
        raise ValueError("guidance_scale must be non-negative")
    if mode == "full_conditional":
        return full_conditional
    direction = guidance_direction(
        full_conditional=full_conditional,
        base_conditional=base_conditional,
        full_unconditional=full_unconditional,
        base_unconditional=base_unconditional,
        mode=mode,
    )
    return full_conditional + (guidance_scale - 1.0) * direction
