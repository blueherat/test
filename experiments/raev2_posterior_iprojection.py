"""Pure helpers for posterior I-projected internal guidance."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def sample_mean_product(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape != right.shape or left.ndim < 2:
        raise ValueError("expected matching batched tensors")
    return (left.float() * right.float()).flatten(1).mean(dim=1)


def sample_rms(value: torch.Tensor) -> torch.Tensor:
    if value.ndim < 2:
        raise ValueError("expected a batched tensor")
    return value.float().flatten(1).square().mean(dim=1).sqrt()


def unit_sample_rms(value: torch.Tensor, tiny: float = 1e-12) -> torch.Tensor:
    scale = sample_rms(value).clamp_min(tiny)
    return value.float() / scale.reshape(len(value), *([1] * (value.ndim - 1)))


@dataclass(frozen=True)
class CrossingBracket:
    lower_coordinate: torch.Tensor
    upper_coordinate: torch.Tensor
    lower_value: torch.Tensor
    upper_value: torch.Tensor
    found: torch.Tensor
    monotone_to_crossing: torch.Tensor


def first_crossing_bracket(
    coordinates: torch.Tensor,
    values: torch.Tensor,
    targets: torch.Tensor,
    tiny: float = 1e-12,
) -> CrossingBracket:
    """Return the first target-crossing interval for each batched curve."""

    if coordinates.ndim != 1 or values.ndim != 2:
        raise ValueError("expected coordinates [grid] and values [batch, grid]")
    if values.shape[1] != len(coordinates) or targets.shape != (len(values),):
        raise ValueError("incompatible curve shapes")
    if len(coordinates) < 2 or not torch.all(coordinates[1:] > coordinates[:-1]):
        raise ValueError("coordinates must be strictly increasing")

    batch = len(values)
    zeros = torch.zeros_like(targets)
    lower_coordinate = zeros.clone()
    upper_coordinate = zeros.clone()
    lower_value = zeros.clone()
    upper_value = zeros.clone()
    found = torch.zeros(batch, dtype=torch.bool, device=values.device)
    monotone = torch.ones(batch, dtype=torch.bool, device=values.device)
    for index in range(1, len(coordinates)):
        delta = values[:, index] - values[:, index - 1]
        monotone = torch.where(~found, monotone & (delta >= -tiny), monotone)
        newly_found = (~found) & (values[:, index] >= targets)
        lower_coordinate = torch.where(
            newly_found, coordinates[index - 1], lower_coordinate
        )
        upper_coordinate = torch.where(
            newly_found, coordinates[index], upper_coordinate
        )
        lower_value = torch.where(newly_found, values[:, index - 1], lower_value)
        upper_value = torch.where(newly_found, values[:, index], upper_value)
        found = found | newly_found
    return CrossingBracket(
        lower_coordinate=lower_coordinate,
        upper_coordinate=upper_coordinate,
        lower_value=lower_value,
        upper_value=upper_value,
        found=found,
        monotone_to_crossing=monotone,
    )


def regula_falsi_coordinate(
    bracket: CrossingBracket,
    targets: torch.Tensor,
    tiny: float = 1e-12,
) -> torch.Tensor:
    denominator = bracket.upper_value - bracket.lower_value
    fraction = (targets - bracket.lower_value) / denominator.clamp_min(tiny)
    fraction = fraction.clamp(0.0, 1.0)
    return bracket.lower_coordinate + fraction * (
        bracket.upper_coordinate - bracket.lower_coordinate
    )


def update_crossing_bracket(
    bracket: CrossingBracket,
    coordinate: torch.Tensor,
    value: torch.Tensor,
    targets: torch.Tensor,
) -> CrossingBracket:
    if coordinate.shape != targets.shape or value.shape != targets.shape:
        raise ValueError("coordinate, value, and target shapes must match")
    use_lower = bracket.found & (value < targets)
    use_upper = bracket.found & ~use_lower
    return CrossingBracket(
        lower_coordinate=torch.where(
            use_lower, coordinate, bracket.lower_coordinate
        ),
        upper_coordinate=torch.where(
            use_upper, coordinate, bracket.upper_coordinate
        ),
        lower_value=torch.where(use_lower, value, bracket.lower_value),
        upper_value=torch.where(use_upper, value, bracket.upper_value),
        found=bracket.found,
        monotone_to_crossing=bracket.monotone_to_crossing,
    )


def same_progress_shift(
    response: torch.Tensor,
    gap: torch.Tensor,
    gamma: float,
    tiny: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize a response to match ordinary IG progress along ``gap``."""

    if response.shape != gap.shape:
        raise ValueError("response and gap must have matching shapes")
    target = float(gamma) * sample_mean_product(gap, gap)
    progress = sample_mean_product(gap, response)
    valid = torch.isfinite(progress) & (progress > tiny)
    scale = target / progress.clamp_min(tiny)
    scale_view = scale.reshape(len(scale), *([1] * (response.ndim - 1)))
    normalized = response.float() * scale_view
    ordinary = float(gamma) * gap.float()
    valid_view = valid.reshape(len(valid), *([1] * (response.ndim - 1)))
    return torch.where(valid_view, normalized, ordinary), valid


def reflect_same_progress_shift(
    candidate_shift: torch.Tensor,
    gap: torch.Tensor,
) -> torch.Tensor:
    """Flip only the component orthogonal to ``gap``.

    The reflection preserves both the candidate's projection onto ``gap`` and
    its Euclidean norm, including when a numerical root has a small residual.
    """

    if candidate_shift.shape != gap.shape:
        raise ValueError("candidate shift and gap must have matching shapes")
    coefficient = sample_mean_product(candidate_shift, gap) / sample_mean_product(
        gap, gap
    ).clamp_min(1e-12)
    projection = _sample_view(coefficient, candidate_shift) * gap.float()
    return 2.0 * projection - candidate_shift.float()


def _sample_view(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return value.reshape(len(value), *([1] * (reference.ndim - 1)))
