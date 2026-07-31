"""Shared output-latent gradient metrics for RAE LPL mechanism audits."""

from __future__ import annotations

import math
from typing import Mapping

import torch

from experiments.rae_lpl_detach_audit import (
    cosine_per_sample,
    gradient_decomposition_metrics,
    tensor_rms,
)
from experiments.raev2_lpl_targets import positive_parallel_projection


def dot_per_sample(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape != right.shape:
        raise ValueError("dot-product inputs must have identical shapes")
    return (left.flatten(1) * right.flatten(1)).sum(dim=1)


def normalized_descent_effect(
    metric_gradient: torch.Tensor,
    step_gradient: torch.Tensor,
) -> torch.Tensor:
    """First-order metric change under a unit-RMS descent step."""

    dimensions = step_gradient[0].numel()
    normalized = step_gradient / tensor_rms(step_gradient).clamp_min(1e-30).view(
        -1, *([1] * (step_gradient.ndim - 1))
    )
    return -dot_per_sample(metric_gradient, normalized) / math.sqrt(dimensions)


def component_metrics(
    *,
    flow_gradient: torch.Tensor,
    raw_gradient: torch.Tensor,
    error_gradient: torch.Tensor,
    variance_gradient: torch.Tensor,
    full_gradient: torch.Tensor,
    log_variance_gradient: torch.Tensor,
) -> Mapping[str, torch.Tensor]:
    """Return interpretable magnitudes, alignments, and directional effects."""

    _, projection = positive_parallel_projection(full_gradient, flow_gradient)
    decomposition = gradient_decomposition_metrics(
        raw_gradient,
        error_gradient,
        full_gradient,
        log_variance_gradient,
    )
    consistency = tensor_rms(
        full_gradient - error_gradient - variance_gradient
    ) / tensor_rms(full_gradient).clamp_min(1e-30)
    return {
        **decomposition,
        "flow_gradient_rms": tensor_rms(flow_gradient),
        "variance_only_gradient_rms": tensor_rms(variance_gradient),
        "variance_over_error_gradient_rms": tensor_rms(variance_gradient)
        / tensor_rms(error_gradient).clamp_min(1e-30),
        "gradient_split_relative_residual": consistency,
        "flow_error_gradient_cosine": cosine_per_sample(
            flow_gradient, error_gradient
        ),
        "flow_variance_gradient_cosine": cosine_per_sample(
            flow_gradient, variance_gradient
        ),
        "flow_full_gradient_cosine": cosine_per_sample(
            flow_gradient, full_gradient
        ),
        "flow_positive_parallel_coefficient": projection[
            "positive_parallel_coefficient"
        ],
        "flow_projected_gradient_fraction": projection[
            "projected_gradient_fraction"
        ],
        "flow_projection_conflict": projection["conflict_fraction"],
        "error_descent_raw_change": normalized_descent_effect(
            raw_gradient, error_gradient
        ),
        "variance_descent_raw_change": normalized_descent_effect(
            raw_gradient, variance_gradient
        ),
        "full_descent_raw_change": normalized_descent_effect(
            raw_gradient, full_gradient
        ),
        "error_descent_log_variance_change": normalized_descent_effect(
            log_variance_gradient, error_gradient
        ),
        "variance_descent_log_variance_change": normalized_descent_effect(
            log_variance_gradient, variance_gradient
        ),
        "full_descent_log_variance_change": normalized_descent_effect(
            log_variance_gradient, full_gradient
        ),
    }


def gradient(
    loss: torch.Tensor,
    prediction: torch.Tensor,
    *,
    retain_graph: bool,
) -> torch.Tensor:
    return torch.autograd.grad(
        loss.sum(),
        prediction,
        retain_graph=retain_graph,
        create_graph=False,
    )[0]


def scalar(value: torch.Tensor) -> float:
    return float(value.detach().float().mean().cpu())
