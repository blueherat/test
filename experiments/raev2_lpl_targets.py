"""Prediction targets for decoder-aware RAEv2 internal-guidance training."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from experiments.raev2_training_core import split_internal_guidance_output


LPL_TARGETS = (
    "full",
    "full_base",
    "guided",
    "guided_common",
    "guided_multiscale",
)

LPL_GRADIENT_MODES = (
    "direct",
    "flow_parallel",
)


def parse_guidance_scales(value: str | Sequence[float]) -> tuple[float, ...]:
    if isinstance(value, str):
        scales = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    else:
        scales = tuple(float(item) for item in value)
    if not scales:
        raise ValueError("at least one guidance scale is required")
    if any(scale < 0 for scale in scales):
        raise ValueError("guidance scales must be non-negative")
    return scales


def _guided_prediction(
    full: torch.Tensor,
    base: torch.Tensor,
    scale: float | torch.Tensor,
) -> torch.Tensor:
    if isinstance(scale, torch.Tensor):
        scale = scale.to(device=full.device, dtype=full.dtype)
        scale = scale.reshape((scale.shape[0],) + (1,) * (full.ndim - 1))
    return base + scale * (full - base)


def _guided_prediction_with_common_gradient(
    full: torch.Tensor,
    base: torch.Tensor,
    scale: float | torch.Tensor,
) -> torch.Tensor:
    """Use the true guided value while moving full/base together under LPL."""

    guided = _guided_prediction(full, base, scale)
    common = 0.5 * (full + base)
    return guided.detach() + common - common.detach()


def positive_parallel_projection(
    auxiliary_gradient: torch.Tensor,
    reference_gradient: torch.Tensor,
    *,
    eps: float = 1e-30,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Keep only the auxiliary gradient's positive reference-parallel part.

    A descent step using the returned gradient cannot increase the reference
    objective to first order at the prediction tensor. Projection is performed
    independently for each sample.
    """

    if auxiliary_gradient.shape != reference_gradient.shape:
        raise ValueError("gradient tensors must have identical shapes")
    if auxiliary_gradient.ndim < 2:
        raise ValueError("gradient tensors must include a batch dimension")

    auxiliary_flat = auxiliary_gradient.flatten(1)
    reference_flat = reference_gradient.flatten(1)
    dot = (auxiliary_flat * reference_flat).sum(dim=1)
    reference_norm_squared = reference_flat.square().sum(dim=1)
    coefficient = (dot / reference_norm_squared.clamp_min(float(eps))).clamp_min(0.0)
    shape = (coefficient.shape[0],) + (1,) * (auxiliary_gradient.ndim - 1)
    projected = coefficient.reshape(shape) * reference_gradient

    auxiliary_norm = auxiliary_flat.norm(dim=1)
    reference_norm = reference_flat.norm(dim=1)
    cosine = dot / (auxiliary_norm * reference_norm).clamp_min(float(eps))
    projected_fraction = projected.flatten(1).norm(dim=1) / auxiliary_norm.clamp_min(
        float(eps)
    )
    return projected, {
        "auxiliary_reference_cosine": cosine,
        "positive_parallel_coefficient": coefficient,
        "projected_gradient_fraction": projected_fraction,
        "conflict_fraction": (dot < 0).to(dtype=auxiliary_gradient.dtype),
    }


def substitute_prediction_gradient(
    forward_loss: torch.Tensor,
    prediction: torch.Tensor,
    replacement_gradient: torch.Tensor,
) -> torch.Tensor:
    """Preserve a scalar loss value while replacing its prediction gradient."""

    if forward_loss.ndim != 0:
        raise ValueError("forward_loss must be scalar")
    if prediction.shape != replacement_gradient.shape:
        raise ValueError("prediction and replacement gradient shapes must match")
    zero_value_surrogate = (
        (prediction - prediction.detach()) * replacement_gradient.detach()
    ).sum()
    return forward_loss.detach() + zero_value_surrogate


def lpl_prediction_targets(
    model_output: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    *,
    target: str,
    guidance_scale: float,
    multiscale_scales: Sequence[float],
    sample_indices: torch.Tensor,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor | None]:
    """Return predictions decoded by LPL and any per-sample guidance scales.

    ``full_base`` uses an index-deterministic, unbiased one-sample estimate of
    the equally weighted full/base objective.  This keeps only one frozen
    decoder graph per microbatch.  ``guided_multiscale`` similarly assigns a
    scale from the data index, so neither mode consumes training RNG.
    """

    if target not in LPL_TARGETS:
        raise ValueError(f"unsupported LPL target: {target!r}")
    full, base = split_internal_guidance_output(model_output)
    if target == "full":
        return (full,), None
    if base is None:
        raise ValueError(f"LPL target {target!r} requires a dual-output model")
    if target == "full_base":
        if sample_indices.ndim != 1 or sample_indices.shape[0] != full.shape[0]:
            raise ValueError("sample_indices must contain one index per prediction")
        choose_base = sample_indices.to(device=full.device, dtype=torch.long).remainder(2)
        choose_base = choose_base.bool().reshape(
            (full.shape[0],) + (1,) * (full.ndim - 1)
        )
        return (torch.where(choose_base, base, full),), None
    if target in {"guided", "guided_common"}:
        scales = torch.full(
            (full.shape[0],),
            float(guidance_scale),
            device=full.device,
            dtype=full.dtype,
        )
        if target == "guided_common":
            prediction = _guided_prediction_with_common_gradient(
                full,
                base,
                scales,
            )
        else:
            prediction = _guided_prediction(full, base, scales)
        return (prediction,), scales

    choices = parse_guidance_scales(multiscale_scales)
    if sample_indices.ndim != 1 or sample_indices.shape[0] != full.shape[0]:
        raise ValueError("sample_indices must contain one index per prediction")
    choice_indices = sample_indices.to(device="cpu", dtype=torch.long).remainder(
        len(choices)
    )
    scale_values = torch.tensor(
        choices,
        device=full.device,
        dtype=full.dtype,
    )
    scales = scale_values.index_select(0, choice_indices.to(full.device))
    return (_guided_prediction(full, base, scales),), scales
