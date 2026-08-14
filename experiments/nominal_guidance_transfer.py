"""Core operators for nominal-trajectory guidance transfer diagnostics.

The projection scope is one scalar per sample over the complete latent tensor.
This makes the parallel component a literal gain change of the nominal
guidance direction; all remaining change is directional.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import torch


Tensor = torch.Tensor
InterventionMode = Literal[
    "frozen",
    "replay",
    "gain_only",
    "direction_only",
    "factorized",
    "closed",
]
INTERVENTION_MODES = (
    "frozen",
    "replay",
    "gain_only",
    "direction_only",
    "factorized",
    "closed",
)
DonorMode = Literal[
    "paired",
    "same_noise_other_class",
    "other_noise_same_class",
    "other_noise_other_class",
]
DONOR_MODES = (
    "paired",
    "same_noise_other_class",
    "other_noise_same_class",
    "other_noise_other_class",
)


class GapProjection(NamedTuple):
    """Decomposition of a current gap relative to a nominal gap."""

    coefficient: Tensor
    valid: Tensor
    current_parallel: Tensor
    current_orthogonal: Tensor
    delta_parallel: Tensor
    delta_orthogonal: Tensor


class NominalDerivatives(NamedTuple):
    """Coupled derivatives used by the transfer interventions."""

    baseline: Tensor
    frozen: Tensor
    replay: Tensor
    gain_only: Tensor
    direction_only: Tensor
    closed: Tensor


def _reduce_dims(value: Tensor) -> tuple[int, ...]:
    if value.ndim < 2:
        raise ValueError("expected a batch dimension and at least one feature dimension")
    return tuple(range(1, value.ndim))


def samplewise_gap_projection(
    nominal_gap: Tensor,
    current_gap: Tensor,
    *,
    relative_floor: float = 1e-12,
) -> GapProjection:
    """Project ``current_gap`` onto ``nominal_gap`` for every sample.

    The decomposition is

    ``current = coefficient * nominal + current_orthogonal``.

    It is also exposed as a correction to frozen guidance:

    ``current = nominal + delta_parallel + delta_orthogonal``.

    Samples whose nominal squared norm is below ``relative_floor`` times the
    current squared norm (or the dtype tiny value) are marked invalid. Their
    coefficient is set to zero so downstream summaries can mask them without
    propagating infinities.
    """

    if nominal_gap.shape != current_gap.shape:
        raise ValueError("nominal_gap and current_gap must have identical shapes")
    if not nominal_gap.is_floating_point() or not current_gap.is_floating_point():
        raise TypeError("guidance gaps must be floating-point tensors")
    if relative_floor < 0:
        raise ValueError("relative_floor must be non-negative")

    reduce_dims = _reduce_dims(nominal_gap)
    nominal_energy = nominal_gap.square().sum(dim=reduce_dims, keepdim=True)
    current_energy = current_gap.square().sum(dim=reduce_dims, keepdim=True)
    dtype_floor = torch.finfo(nominal_gap.dtype).tiny
    threshold = torch.maximum(
        torch.full_like(nominal_energy, dtype_floor),
        float(relative_floor) * current_energy,
    )
    valid_expanded = nominal_energy > threshold
    denominator = nominal_energy.clamp_min(dtype_floor)
    coefficient_expanded = (current_gap * nominal_gap).sum(
        dim=reduce_dims,
        keepdim=True,
    ) / denominator
    coefficient_expanded = torch.where(
        valid_expanded,
        coefficient_expanded,
        torch.zeros_like(coefficient_expanded),
    )
    current_parallel = coefficient_expanded * nominal_gap
    current_orthogonal = current_gap - current_parallel
    delta_parallel = current_parallel - nominal_gap
    delta_orthogonal = current_orthogonal
    return GapProjection(
        coefficient=coefficient_expanded.flatten(1)[:, 0],
        valid=valid_expanded.flatten(1)[:, 0],
        current_parallel=current_parallel,
        current_orthogonal=current_orthogonal,
        delta_parallel=delta_parallel,
        delta_orthogonal=delta_orthogonal,
    )


def nominal_transfer_derivatives(
    *,
    anchor_baseline: Tensor,
    other_baseline: Tensor,
    anchor_frozen: Tensor,
    anchor_gain: Tensor,
    other_gain: Tensor,
    anchor_direction: Tensor,
    other_direction: Tensor,
    anchor_closed: Tensor,
    other_closed: Tensor,
    gamma: float,
) -> NominalDerivatives:
    """Build all derivatives for one synchronized nominal-transfer step.

    ``replay`` freezes both the strong field and the gap on the nominal path.
    ``frozen`` freezes only the gap. ``gain_only`` and ``direction_only``
    restore the corresponding part of current-state gap reevaluation.
    """

    tensors = (
        anchor_baseline,
        other_baseline,
        anchor_frozen,
        anchor_gain,
        other_gain,
        anchor_direction,
        other_direction,
        anchor_closed,
        other_closed,
    )
    if any(value.shape != anchor_baseline.shape for value in tensors[1:]):
        raise ValueError("all field tensors must have identical shapes")

    nominal_gap = anchor_baseline - other_baseline
    gain_gap = anchor_gain - other_gain
    direction_gap = anchor_direction - other_direction
    closed_gap = anchor_closed - other_closed
    gain_projection = samplewise_gap_projection(nominal_gap, gain_gap)
    direction_projection = samplewise_gap_projection(nominal_gap, direction_gap)
    scale = float(gamma)
    return NominalDerivatives(
        baseline=anchor_baseline,
        frozen=anchor_frozen + scale * nominal_gap,
        replay=anchor_baseline + scale * nominal_gap,
        gain_only=anchor_gain
        + scale * (nominal_gap + gain_projection.delta_parallel),
        direction_only=anchor_direction
        + scale * (nominal_gap + direction_projection.delta_orthogonal),
        closed=anchor_closed + scale * closed_gap,
    )


def intervention_guidance(
    nominal_gap: Tensor,
    current_gap: Tensor | None,
    *,
    mode: InterventionMode,
    nominal_scale: float = 1.0,
    orthogonal_scale: float = 1.0,
) -> Tensor:
    """Return the guidance increment used by one intervention branch."""

    if mode not in INTERVENTION_MODES:
        raise ValueError(f"unsupported intervention mode: {mode}")
    if mode in ("frozen", "replay"):
        return nominal_gap
    if current_gap is None:
        raise ValueError(f"{mode} requires a current-state guidance gap")
    projection = samplewise_gap_projection(nominal_gap, current_gap)
    if mode == "gain_only":
        return nominal_gap + projection.delta_parallel
    if mode == "direction_only":
        return nominal_gap + projection.delta_orthogonal
    if mode == "factorized":
        return (
            float(nominal_scale) * nominal_gap
            + float(orthogonal_scale) * projection.current_orthogonal
        )
    return current_gap


def donor_inputs(
    target_noise: Tensor,
    target_labels: Tensor,
    *,
    mode: DonorMode,
    num_classes: int,
    class_shift: int = 1,
) -> tuple[Tensor, Tensor]:
    """Construct the controlled donor noise/class pair for one batch."""

    if mode not in DONOR_MODES:
        raise ValueError(f"unsupported donor mode: {mode}")
    if len(target_noise) != len(target_labels):
        raise ValueError("noise and labels must have the same batch size")
    if num_classes <= 1:
        raise ValueError("num_classes must exceed one")
    if class_shift % num_classes == 0:
        raise ValueError("class_shift must change every class")
    uses_other_noise = mode in ("other_noise_same_class", "other_noise_other_class")
    uses_other_class = mode in ("same_noise_other_class", "other_noise_other_class")
    if uses_other_noise and len(target_noise) < 2:
        raise ValueError("other-noise donor modes require batch size at least two")
    donor_noise = torch.roll(target_noise, shifts=1, dims=0) if uses_other_noise else target_noise
    donor_labels = (
        (target_labels + int(class_shift)) % int(num_classes)
        if uses_other_class
        else target_labels
    )
    return donor_noise, donor_labels


def sample_rms(value: Tensor) -> Tensor:
    """Return one RMS value per sample."""

    return value.flatten(1).square().mean(dim=1).sqrt()


def sample_cosine(left: Tensor, right: Tensor) -> Tensor:
    """Return one cosine per sample, with zero for undefined pairs."""

    if left.shape != right.shape:
        raise ValueError("left and right must have identical shapes")
    left_flat = left.flatten(1)
    right_flat = right.flatten(1)
    denominator = left_flat.norm(dim=1) * right_flat.norm(dim=1)
    valid = denominator > torch.finfo(left.dtype).tiny
    cosine = (left_flat * right_flat).sum(dim=1) / denominator.clamp_min(
        torch.finfo(left.dtype).tiny
    )
    return torch.where(valid, cosine, torch.zeros_like(cosine))


def nominal_transfer_metrics(
    nominal_gap: Tensor,
    current_gap: Tensor,
    *,
    state_shift: Tensor | None = None,
) -> dict[str, Tensor]:
    """Return per-sample geometry of nominal-to-current gap transfer."""

    projection = samplewise_gap_projection(nominal_gap, current_gap)
    nominal_rms = sample_rms(nominal_gap)
    current_rms = sample_rms(current_gap)
    change = current_gap - nominal_gap
    change_rms = sample_rms(change)
    tiny = torch.finfo(nominal_gap.dtype).tiny
    metrics = {
        "valid": projection.valid,
        "cosine": sample_cosine(nominal_gap, current_gap),
        "coefficient": projection.coefficient,
        "nominal_rms": nominal_rms,
        "current_rms": current_rms,
        "current_over_nominal_rms": current_rms / nominal_rms.clamp_min(tiny),
        "change_rms": change_rms,
        "change_over_nominal_rms": change_rms / nominal_rms.clamp_min(tiny),
        "delta_cosine_nominal": sample_cosine(change, nominal_gap),
        "delta_parallel_rms": sample_rms(projection.delta_parallel),
        "delta_orthogonal_rms": sample_rms(projection.delta_orthogonal),
        "orthogonal_over_current_rms": sample_rms(projection.current_orthogonal)
        / current_rms.clamp_min(tiny),
        "orthogonal_energy_fraction": projection.current_orthogonal.flatten(1)
        .square()
        .sum(dim=1)
        / current_gap.flatten(1).square().sum(dim=1).clamp_min(tiny),
    }
    if state_shift is not None:
        if state_shift.shape != nominal_gap.shape:
            raise ValueError("state_shift must match the gap shape")
        shift_rms = sample_rms(state_shift)
        shift_valid = shift_rms > 1e-12
        metrics["state_shift_rms"] = shift_rms
        metrics["state_shift_valid"] = shift_valid
        metrics["effective_secant_gain"] = torch.where(
            shift_valid,
            change_rms / shift_rms.clamp_min(tiny),
            torch.full_like(shift_rms, float("nan")),
        )
    return metrics
