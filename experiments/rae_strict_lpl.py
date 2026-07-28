"""Latent Perceptual Loss (LPL) for a frozen deterministic RAE decoder.

This module implements the objective from Berrada Ifriqi et al., ICLR 2025:
decoder features of a clean latent and a predicted clean latent are compared
after cross-normalization with the predicted feature statistics. Decoder
outliers are masked before normalization and feature matching.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F


def flow_clean_estimate(
    noisy: torch.Tensor,
    velocity: torch.Tensor,
    time: torch.Tensor,
) -> torch.Tensor:
    """Recover x_0 for the linear path x_t=(1-t)x_0+t*noise."""

    scale = time.reshape((time.shape[0],) + (1,) * (noisy.ndim - 1))
    return noisy - scale * velocity


def noise_to_signal_ratio(time: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return sigma/alpha=t/(1-t) for the linear optimal-transport path."""

    return time / (1.0 - time).clamp_min(eps)


def lpl_time_gate(time: torch.Tensor, max_noise_to_signal: float) -> torch.Tensor:
    """Select the high-SNR samples on which LPL is defined."""

    if max_noise_to_signal <= 0:
        raise ValueError("max_noise_to_signal must be positive")
    return noise_to_signal_ratio(time) <= float(max_noise_to_signal)


def _pool_kernel(value: int) -> int:
    value = max(int(value), 1)
    if value % 2 == 0:
        value += 1
    return value


def decoder_outlier_mask(
    features: torch.Tensor,
    *,
    quantile: float = 0.02,
    opening: int = 5,
    closing: int = 3,
) -> torch.Tensor:
    """Reproduce the paper's percentile and morphology outlier mask.

    Args:
        features: Decoder feature maps in ``[B, C, H, W]`` format.
        quantile: Fraction used for lower and upper nearest quantiles.
        opening: Kernel used by the erosion-like second morphology pass.
        closing: Kernel used by the dilation-like first morphology pass.
    """

    if features.ndim != 4:
        raise ValueError(f"expected BCHW features, got {tuple(features.shape)}")
    if not 0.0 < quantile < 0.5:
        raise ValueError("quantile must be in (0, 0.5)")

    flat = features.detach().flatten(-2)
    count = flat.shape[-1]
    lower_k = min(max(int(count * quantile), 1), count)
    upper_k = min(max(int(count * (1.0 - quantile)), 1), count)
    lower = flat.kthvalue(lower_k, dim=-1).values[..., None, None]
    upper = flat.kthvalue(upper_k, dim=-1).values[..., None, None]
    margin = 2.0 * flat.std(dim=-1, correction=1).nan_to_num(0.0)[..., None, None]
    mask = ((lower - margin < features.detach()) & (features.detach() < upper + margin)).float()

    closing = _pool_kernel(closing)
    opening = _pool_kernel(opening)
    mask = F.max_pool2d(mask, kernel_size=closing, stride=1, padding=closing // 2)
    mask = -F.max_pool2d(-mask, kernel_size=opening, stride=1, padding=opening // 2)
    return mask > 0.5


def cross_normalize_decoder_features(
    target: torch.Tensor,
    prediction: torch.Tensor,
    mask: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize both branches with prediction-branch channel statistics."""

    if target.shape != prediction.shape or target.shape != mask.shape:
        raise ValueError(
            "target, prediction and mask must have identical shapes; got "
            f"{tuple(target.shape)}, {tuple(prediction.shape)}, {tuple(mask.shape)}"
        )
    weights = mask.to(dtype=prediction.dtype)
    count = weights.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
    mean = (prediction * weights).sum(dim=(-2, -1), keepdim=True) / count
    centered = (prediction - mean) * weights
    variance = centered.square().sum(dim=(-2, -1), keepdim=True) / count
    inv_std = torch.rsqrt(variance + float(eps))
    return (target - mean) * inv_std, (prediction - mean) * inv_std


def strict_lpl_per_sample(
    target_features: Sequence[torch.Tensor],
    predicted_features: Sequence[torch.Tensor],
    *,
    layer_weights: Sequence[float] | None = None,
    outlier_quantile: float = 0.02,
    outlier_opening: int = 5,
    outlier_closing: int = 3,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the paper's channel-averaged, spatially summed LPL."""

    if not target_features or len(target_features) != len(predicted_features):
        raise ValueError("target and predicted feature pyramids must be non-empty and aligned")
    if layer_weights is None:
        layer_weights = [1.0] * len(target_features)
    if len(layer_weights) != len(target_features):
        raise ValueError("layer_weights must match the feature pyramid")

    layer_losses = []
    keep_fractions = []
    for target, prediction, weight in zip(
        target_features, predicted_features, layer_weights, strict=True
    ):
        if target.shape != prediction.shape:
            raise ValueError(
                f"feature shapes differ: {tuple(target.shape)} vs {tuple(prediction.shape)}"
            )
        mask = decoder_outlier_mask(
            prediction,
            quantile=outlier_quantile,
            opening=outlier_opening,
            closing=outlier_closing,
        )
        normalized_target, normalized_prediction = cross_normalize_decoder_features(
            target, prediction, mask, eps=eps
        )
        squared = (normalized_target - normalized_prediction).square()
        # Equation (3): spatial L2 squared, then average over channels.
        per_sample = (squared * mask).sum(dim=(-2, -1)).mean(dim=1)
        layer_losses.append(per_sample * float(weight))
        keep_fractions.append(mask.float().mean(dim=(1, 2, 3)))

    stacked_losses = torch.stack(layer_losses, dim=1)
    stacked_keep = torch.stack(keep_fractions, dim=1)
    return stacked_losses.sum(dim=1), {
        "layer_losses": stacked_losses,
        "mask_keep_fraction": stacked_keep,
    }


def decoder_hidden_indices(
    decoder_depth: int,
    fractions: Sequence[float] = (0.2, 0.4, 0.6, 0.8, 1.0),
) -> tuple[int, ...]:
    """Map relative decoder depths to hidden-state tuple indices."""

    if decoder_depth < 1:
        raise ValueError("decoder_depth must be positive")
    indices = []
    for fraction in fractions:
        if not 0.0 < float(fraction) <= 1.0:
            raise ValueError("decoder layer fractions must be in (0, 1]")
        index = min(max(int(round(float(fraction) * decoder_depth)), 1), decoder_depth)
        if not indices or index != indices[-1]:
            indices.append(index)
    return tuple(indices)


def _latent_to_decoder_tokens(rae: torch.nn.Module, latent: torch.Tensor) -> torch.Tensor:
    if getattr(rae, "do_normalization", False):
        mean = getattr(rae, "latent_mean", None)
        variance = getattr(rae, "latent_var", None)
        mean = mean.to(device=latent.device, dtype=latent.dtype) if mean is not None else 0.0
        variance = (
            variance.to(device=latent.device, dtype=latent.dtype)
            if variance is not None
            else 1.0
        )
        latent = latent * torch.sqrt(variance + float(getattr(rae, "eps", 1e-8))) + mean
    if latent.ndim == 4:
        batch, channels, height, width = latent.shape
        latent = latent.reshape(batch, channels, height * width).transpose(1, 2)
    elif latent.ndim != 3:
        raise ValueError(f"expected BCHW or BNC latent, got {tuple(latent.shape)}")
    return latent


def decoder_feature_pyramid(
    rae: torch.nn.Module,
    latent: torch.Tensor,
    *,
    layer_indices: Sequence[int] | None = None,
) -> tuple[torch.Tensor, ...]:
    """Return deterministic RAE decoder block features as BCHW maps."""

    decoder = rae.decoder
    depth = len(decoder.decoder_layers)
    indices = tuple(layer_indices) if layer_indices is not None else decoder_hidden_indices(depth)
    if not indices or min(indices) < 0 or max(indices) > depth:
        raise ValueError(f"invalid decoder hidden indices {indices} for depth {depth}")

    tokens = _latent_to_decoder_tokens(rae, latent)
    output = decoder(tokens, drop_cls_token=False, output_hidden_states=True)
    states = output.hidden_states
    features = []
    for index in indices:
        patch_tokens = states[int(index)][:, 1:, :]
        side = math.isqrt(patch_tokens.shape[1])
        if side * side != patch_tokens.shape[1]:
            raise ValueError(f"decoder patch count is not square: {patch_tokens.shape[1]}")
        features.append(
            patch_tokens.transpose(1, 2).reshape(
                patch_tokens.shape[0], patch_tokens.shape[2], side, side
            )
        )
    return tuple(features)
