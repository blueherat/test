"""Pure utilities for swapping RAEv2 full and internal-guidance fields."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def combine_full_and_contrast(
    full_source: tuple[torch.Tensor, torch.Tensor],
    contrast_source: tuple[torch.Tensor, torch.Tensor],
    *,
    guidance_scale: float,
    active: torch.Tensor | None = None,
    identical_sources: bool = False,
) -> torch.Tensor:
    """Return ``F + (scale - 1) * D`` with independently sourced F and D.

    ``F`` is the full prediction from ``full_source`` and
    ``D = full - base`` is computed from ``contrast_source``.  With identical
    sources this is exactly ``base + scale * (full - base)``.
    """

    full_prediction, _ = full_source
    contrast_full, contrast_base = contrast_source
    if full_prediction.shape != contrast_full.shape:
        raise ValueError("full and contrast predictions must have identical shapes")
    if contrast_full.shape != contrast_base.shape:
        raise ValueError("contrast full/base predictions must have identical shapes")

    contrast = contrast_full - contrast_base
    if identical_sources:
        # Preserve the official RAEv2 operation order for bitwise parity under
        # reduced precision.
        guided = contrast_base + float(guidance_scale) * contrast
    else:
        guided = full_prediction + (float(guidance_scale) - 1.0) * contrast
    if active is None:
        return guided
    if active.ndim != 1 or active.shape[0] != full_prediction.shape[0]:
        raise ValueError("active must contain one boolean per sample")
    shape = (active.shape[0],) + (1,) * (full_prediction.ndim - 1)
    return torch.where(active.to(device=guided.device).reshape(shape), guided, full_prediction)


def slice_batch_kwargs(
    values: Mapping[str, torch.Tensor | None],
    batch_size: int,
) -> dict[str, torch.Tensor]:
    """Slice tensor-valued conditioning arguments to the active half-batch."""

    result = {}
    for key, value in values.items():
        if value is not None:
            result[key] = value[:batch_size]
    return result


def forward_with_head_swap(
    full_model: torch.nn.Module,
    contrast_model: torch.nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    *,
    guidance_scale: float,
    guidance_interval: tuple[float, float],
    **condition_kwargs: torch.Tensor | None,
) -> torch.Tensor:
    """Evaluate a mixed full/contrast field using RAEv2's doubled-batch API."""

    if x.shape[0] % 2:
        raise ValueError("head-swap guidance expects an even doubled batch")
    batch_size = x.shape[0] // 2
    active_x = x[:batch_size]
    active_t = t[:batch_size]
    active_kwargs = slice_batch_kwargs(condition_kwargs, batch_size)

    full_output = full_model(active_x, active_t, **active_kwargs)
    if full_model is contrast_model:
        contrast_output = full_output
    else:
        contrast_output = contrast_model(active_x, active_t, **active_kwargs)
    if not (
        isinstance(full_output, tuple)
        and len(full_output) == 2
        and isinstance(contrast_output, tuple)
        and len(contrast_output) == 2
    ):
        raise ValueError("head-swap models must return (full, base)")

    in_channels = int(full_model.in_channels)
    full_pair = (
        full_output[0][:, :in_channels],
        full_output[1][:, :in_channels],
    )
    contrast_pair = (
        contrast_output[0][:, :in_channels],
        contrast_output[1][:, :in_channels],
    )
    lower, upper = (float(value) for value in guidance_interval)
    if not lower < upper:
        raise ValueError("guidance_interval must satisfy lower < upper")
    active = (active_t >= lower) & (active_t <= upper)
    mixed = combine_full_and_contrast(
        full_pair,
        contrast_pair,
        guidance_scale=float(guidance_scale),
        active=active,
        identical_sources=full_model is contrast_model,
    )
    return torch.cat([mixed, mixed], dim=0)
