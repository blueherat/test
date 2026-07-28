"""Loss and gradient utilities for auditing LPL prediction-stat detachment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from experiments.rae_strict_lpl import (
    decoder_outlier_mask,
    strict_lpl_per_sample,
)


def tensor_rms(value: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    """Return one RMS value for each batch element."""

    if value.ndim < 2:
        raise ValueError("tensor_rms expects a batch dimension")
    return value.square().flatten(1).mean(1).add(float(eps)).sqrt()


def cosine_per_sample(
    left: torch.Tensor,
    right: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Return flattened cosine similarity for each batch element."""

    if left.shape != right.shape:
        raise ValueError("cosine inputs must have identical shapes")
    left_flat = left.flatten(1)
    right_flat = right.flatten(1)
    numerator = (left_flat * right_flat).sum(1)
    denominator = left_flat.norm(dim=1) * right_flat.norm(dim=1)
    return numerator / denominator.clamp_min(float(eps))


def _masked_channel_moments(
    features: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = mask.to(dtype=features.dtype)
    count = weights.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
    mean = (features * weights).sum(dim=(-2, -1), keepdim=True) / count
    centered = (features - mean) * weights
    variance = centered.square().sum(dim=(-2, -1), keepdim=True) / count
    return mean, variance


def lpl_loss_variants_per_sample(
    target_features: Sequence[torch.Tensor],
    predicted_features: Sequence[torch.Tensor],
    *,
    layer_weights: Sequence[float] | None = None,
    outlier_quantile: float = 0.02,
    outlier_opening: int = 5,
    outlier_closing: int = 3,
    eps: float = 1e-6,
) -> tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]]:
    """Compute raw, detached-stat, and full prediction-normalized LPL.

    ``prediction_detach`` and ``prediction_full`` are exactly equal in the
    forward pass. Only the latter differentiates through prediction variance.
    """

    if not target_features or len(target_features) != len(predicted_features):
        raise ValueError("feature pyramids must be non-empty and aligned")
    if layer_weights is None:
        layer_weights = [1.0] * len(target_features)
    if len(layer_weights) != len(target_features):
        raise ValueError("layer_weights must match the feature pyramid")

    raw_layers = []
    detach_layers = []
    full_layers = []
    prediction_variances = []
    target_variances = []
    variance_ratios = []
    standard_deviation_ratios = []
    centered_cosines = []
    normalized_mean_errors = []
    keep_fractions = []

    for target, prediction, layer_weight in zip(
        target_features,
        predicted_features,
        layer_weights,
        strict=True,
    ):
        if target.shape != prediction.shape:
            raise ValueError("target and prediction feature shapes must match")
        if target.ndim != 4:
            raise ValueError("features must have BCHW shape")

        mask = decoder_outlier_mask(
            prediction,
            quantile=float(outlier_quantile),
            opening=int(outlier_opening),
            closing=int(outlier_closing),
        )
        weights = mask.to(dtype=prediction.dtype)
        prediction_mean, prediction_variance = _masked_channel_moments(
            prediction, mask
        )
        target_mean, target_variance = _masked_channel_moments(target, mask)
        denominator = prediction_variance + float(eps)
        residual_squared = (prediction - target).square() * weights

        raw_channel = residual_squared.sum(dim=(-2, -1))
        detach_channel = (
            residual_squared / denominator.detach()
        ).sum(dim=(-2, -1))
        full_channel = (residual_squared / denominator).sum(dim=(-2, -1))
        weight = float(layer_weight)
        raw_layers.append(raw_channel.mean(dim=1) * weight)
        detach_layers.append(detach_channel.mean(dim=1) * weight)
        full_layers.append(full_channel.mean(dim=1) * weight)

        prediction_centered = (prediction - prediction_mean) * weights
        target_centered = (target - target_mean) * weights
        covariance = (prediction_centered * target_centered).sum(
            dim=(-2, -1)
        )
        centered_norm = (
            prediction_centered.square().sum(dim=(-2, -1)).sqrt()
            * target_centered.square().sum(dim=(-2, -1)).sqrt()
        )
        centered_cosine = covariance / centered_norm.clamp_min(float(eps))

        prediction_variance_flat = prediction_variance[..., 0, 0]
        target_variance_flat = target_variance[..., 0, 0]
        variance_ratio = (prediction_variance_flat + float(eps)) / (
            target_variance_flat + float(eps)
        )
        mean_error = (prediction_mean - target_mean).square()[..., 0, 0] / (
            target_variance_flat + float(eps)
        )

        prediction_variances.append(
            prediction_variance_flat.mean(dim=1)
        )
        target_variances.append(target_variance_flat.mean(dim=1))
        variance_ratios.append(
            variance_ratio.clamp_min(float(eps)).log().mean(dim=1).exp()
        )
        standard_deviation_ratios.append(
            variance_ratio.clamp_min(float(eps)).sqrt().log().mean(dim=1).exp()
        )
        centered_cosines.append(centered_cosine.mean(dim=1))
        normalized_mean_errors.append(mean_error.mean(dim=1))
        keep_fractions.append(weights.mean(dim=(1, 2, 3)))

    raw_layer_tensor = torch.stack(raw_layers, dim=1)
    detach_layer_tensor = torch.stack(detach_layers, dim=1)
    full_layer_tensor = torch.stack(full_layers, dim=1)
    losses = {
        "raw": raw_layer_tensor.sum(dim=1),
        "prediction_detach": detach_layer_tensor.sum(dim=1),
        "prediction_full": full_layer_tensor.sum(dim=1),
    }
    details = {
        "raw_layers": raw_layer_tensor,
        "prediction_detach_layers": detach_layer_tensor,
        "prediction_full_layers": full_layer_tensor,
        "prediction_variance_layers": torch.stack(prediction_variances, dim=1),
        "target_variance_layers": torch.stack(target_variances, dim=1),
        "prediction_over_target_variance_layers": torch.stack(
            variance_ratios, dim=1
        ),
        "prediction_over_target_std_layers": torch.stack(
            standard_deviation_ratios, dim=1
        ),
        "centered_cosine_layers": torch.stack(centered_cosines, dim=1),
        "normalized_mean_error_layers": torch.stack(
            normalized_mean_errors, dim=1
        ),
        "mask_keep_fraction_layers": torch.stack(keep_fractions, dim=1),
        "mean_log_prediction_variance": torch.stack(
            [
                variance.clamp_min(float(eps)).log()
                for variance in prediction_variances
            ],
            dim=1,
        ).mean(dim=1),
    }
    return losses, details


def decoder_feature_objective_per_sample(
    mode: str,
    target_features: Sequence[torch.Tensor],
    predicted_features: Sequence[torch.Tensor],
    *,
    layer_weights: Sequence[float] | None = None,
    outlier_quantile: float = 0.02,
    outlier_opening: int = 5,
    outlier_closing: int = 3,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
    """Select a controlled raw, detached-stat, or full feature objective."""

    canonical_mode = {
        "raw": "raw",
        "detach": "prediction_detach",
        "prediction_detach": "prediction_detach",
        "full": "prediction_full",
        "lpl": "prediction_full",
        "prediction_full": "prediction_full",
    }.get(str(mode))
    if canonical_mode is None:
        raise ValueError(f"unknown decoder feature objective: {mode!r}")

    if canonical_mode == "prediction_full":
        return strict_lpl_per_sample(
            target_features,
            predicted_features,
            layer_weights=layer_weights,
            outlier_quantile=outlier_quantile,
            outlier_opening=outlier_opening,
            outlier_closing=outlier_closing,
            eps=eps,
        )

    losses, details = lpl_loss_variants_per_sample(
        target_features,
        predicted_features,
        layer_weights=layer_weights,
        outlier_quantile=outlier_quantile,
        outlier_opening=outlier_opening,
        outlier_closing=outlier_closing,
        eps=eps,
    )
    selected_details = dict(details)
    selected_details["mask_keep_fraction"] = details[
        "mask_keep_fraction_layers"
    ]
    return losses[canonical_mode], selected_details


def gradient_decomposition_metrics(
    raw_gradient: torch.Tensor,
    detach_gradient: torch.Tensor,
    full_gradient: torch.Tensor,
    log_variance_gradient: torch.Tensor,
) -> Mapping[str, torch.Tensor]:
    """Summarize the exact ``full = detach + stats`` gradient split."""

    shapes = {
        tuple(raw_gradient.shape),
        tuple(detach_gradient.shape),
        tuple(full_gradient.shape),
        tuple(log_variance_gradient.shape),
    }
    if len(shapes) != 1:
        raise ValueError("all gradients must have identical shapes")

    stats_gradient = full_gradient - detach_gradient
    raw_rms = tensor_rms(raw_gradient)
    detach_rms = tensor_rms(detach_gradient)
    full_rms = tensor_rms(full_gradient)
    stats_rms = tensor_rms(stats_gradient)
    return {
        "raw_gradient_rms": raw_rms,
        "detach_gradient_rms": detach_rms,
        "full_gradient_rms": full_rms,
        "stats_gradient_rms": stats_rms,
        "stats_over_full_gradient_rms": stats_rms / full_rms.clamp_min(1e-30),
        "detach_over_full_gradient_rms": detach_rms / full_rms.clamp_min(1e-30),
        "full_detach_gradient_cosine": cosine_per_sample(
            full_gradient, detach_gradient
        ),
        "stats_full_gradient_cosine": cosine_per_sample(
            stats_gradient, full_gradient
        ),
        "stats_detach_gradient_cosine": cosine_per_sample(
            stats_gradient, detach_gradient
        ),
        "raw_detach_gradient_cosine": cosine_per_sample(
            raw_gradient, detach_gradient
        ),
        "raw_full_gradient_cosine": cosine_per_sample(raw_gradient, full_gradient),
        "raw_descent_log_variance_cosine": cosine_per_sample(
            -raw_gradient, log_variance_gradient
        ),
        "detach_descent_log_variance_cosine": cosine_per_sample(
            -detach_gradient, log_variance_gradient
        ),
        "full_descent_log_variance_cosine": cosine_per_sample(
            -full_gradient, log_variance_gradient
        ),
        "stats_descent_log_variance_cosine": cosine_per_sample(
            -stats_gradient, log_variance_gradient
        ),
    }


__all__ = [
    "cosine_per_sample",
    "decoder_feature_objective_per_sample",
    "gradient_decomposition_metrics",
    "lpl_loss_variants_per_sample",
    "tensor_rms",
]
