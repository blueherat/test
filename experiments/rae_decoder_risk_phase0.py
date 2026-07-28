"""Core metrics for the decoder-aware RAE Phase-0 mechanism audit.

The module deliberately contains no training loop.  It compares losses on the
same static linear flow-matching path and supplies the structured quadratic
metrics used by the no-training proxy gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


DEFAULT_HIDDEN_FRACTIONS = (0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class Phase0GateThresholds:
    gradient_cosine: float = 0.80
    correction_advantage: float = 0.15
    correction_time_bins: int = 3
    proxy_spearman: float = 0.80
    proxy_gradient_cosine: float = 0.70
    proxy_split_gap: float = 0.15


def static_linear_state(
    clean: torch.Tensor,
    noise: torch.Tensor,
    time: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``z_t`` and the velocity target for the RAE linear FM path."""

    if clean.shape != noise.shape:
        raise ValueError("clean and noise must have the same shape")
    if isinstance(time, torch.Tensor):
        expanded = time.to(clean).reshape(-1, *([1] * (clean.ndim - 1)))
    else:
        expanded = clean.new_tensor(float(time))
    state = (1.0 - expanded) * clean + expanded * noise
    return state, noise - clean


def clean_from_velocity(
    state: torch.Tensor,
    velocity: torch.Tensor,
    time: torch.Tensor | float,
) -> torch.Tensor:
    """Recover the one-step clean estimate under the static linear FM path."""

    if isinstance(time, torch.Tensor):
        expanded = time.to(state).reshape(-1, *([1] * (state.ndim - 1)))
    else:
        expanded = state.new_tensor(float(time))
    return state - expanded * velocity


def velocity_and_clean_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    time: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-sample ``L_v`` and the exactly equivalent ``t^2 L_v``."""

    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    velocity_loss = (prediction - target).square().flatten(1).mean(dim=1)
    if isinstance(time, torch.Tensor):
        factor = time.to(velocity_loss).reshape(-1).square()
    else:
        factor = velocity_loss.new_full(velocity_loss.shape, float(time) ** 2)
    return velocity_loss, factor * velocity_loss


def _latent_to_decoder_tokens(rae: torch.nn.Module, latent: torch.Tensor) -> torch.Tensor:
    z = latent
    if bool(getattr(rae, "do_normalization", False)):
        latent_mean = getattr(rae, "latent_mean", None)
        latent_var = getattr(rae, "latent_var", None)
        mean = latent_mean.to(z) if latent_mean is not None else 0.0
        var = latent_var.to(z) if latent_var is not None else 1.0
        z = z * torch.sqrt(var + float(getattr(rae, "eps", 1e-5))) + mean
    if bool(getattr(rae, "reshape_to_2d", True)):
        batch, channels, height, width = z.shape
        z = z.reshape(batch, channels, height * width).transpose(1, 2)
    return z


def decoder_hidden_indices(
    hidden_count: int,
    fractions: Sequence[float] = DEFAULT_HIDDEN_FRACTIONS,
) -> tuple[int, ...]:
    """Choose unique middle/deep decoder states, including the final block."""

    if int(hidden_count) < 2:
        raise ValueError("decoder must expose at least two hidden states")
    last = int(hidden_count) - 1
    indices = {
        min(last, max(1, int(round(last * float(fraction)))))
        for fraction in fractions
    }
    return tuple(sorted(indices))


def decoder_hidden_features(
    rae: torch.nn.Module,
    latent: torch.Tensor,
    *,
    hidden_indices: Sequence[int] | None = None,
) -> tuple[torch.Tensor, ...]:
    """Differentiable official RAE decoder hidden states without image decoding."""

    tokens = _latent_to_decoder_tokens(rae, latent)
    output = rae.decoder(tokens, drop_cls_token=False, output_hidden_states=True)
    states = output.hidden_states
    if states is None:
        raise RuntimeError("RAE decoder did not return hidden states")
    selected = decoder_hidden_indices(len(states)) if hidden_indices is None else tuple(hidden_indices)
    if not selected or min(selected) < 0 or max(selected) >= len(states):
        raise ValueError(f"invalid decoder hidden indices {selected} for {len(states)} states")
    return tuple(states[index][:, 1:] for index in selected)


def decoder_hidden_loss(
    candidate: Sequence[torch.Tensor],
    reference: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Per-sample LPL: the mean of per-layer hidden-state mean square errors."""

    if len(candidate) != len(reference) or not candidate:
        raise ValueError("candidate and reference feature lists must be non-empty and equal")
    losses = []
    for candidate_layer, reference_layer in zip(candidate, reference):
        if candidate_layer.shape != reference_layer.shape:
            raise ValueError("candidate and reference decoder features must have equal shapes")
        losses.append((candidate_layer - reference_layer).square().flatten(1).mean(dim=1))
    return torch.stack(losses, dim=1).mean(dim=1)


def decoder_hidden_rms(features: Sequence[torch.Tensor]) -> torch.Tensor:
    """Return one RMS response per sample and selected decoder layer."""

    return torch.stack(
        [feature.square().flatten(1).mean(dim=1).sqrt() for feature in features], dim=1
    )


def dct_matrix(size: int, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if int(size) < 1:
        raise ValueError("DCT size must be positive")
    n = torch.arange(int(size), dtype=dtype)[None, :]
    k = torch.arange(int(size), dtype=dtype)[:, None]
    matrix = torch.cos(math.pi * (n + 0.5) * k / int(size))
    matrix[0] *= math.sqrt(1.0 / int(size))
    if int(size) > 1:
        matrix[1:] *= math.sqrt(2.0 / int(size))
    return matrix


def dct2(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] != (basis.shape[0], basis.shape[0]):
        raise ValueError("latent spatial shape and DCT basis disagree")
    basis = basis.to(x)
    return torch.einsum("ui,bcij,vj->bcuv", basis, x, basis)


def idct2(coefficients: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    if coefficients.shape[-2:] != (basis.shape[0], basis.shape[0]):
        raise ValueError("coefficient spatial shape and DCT basis disagree")
    basis = basis.to(coefficients)
    return torch.einsum("ui,bcuv,vj->bcij", basis, coefficients, basis)


def radial_dct_band_masks(
    size: int,
    band_count: int = 4,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Partition DCT coefficients into deterministic radial frequency bands."""

    if int(band_count) < 1:
        raise ValueError("band_count must be positive")
    axis = torch.arange(int(size), dtype=torch.float64)
    radius = torch.sqrt(axis[:, None].square() + axis[None, :].square())
    maximum = float(radius.max()) + 1e-12
    index = torch.floor(radius / maximum * int(band_count)).long().clamp_max(int(band_count) - 1)
    return torch.stack([index == band for band in range(int(band_count))]).to(device)


def trace_normalize_channel_metric(metric: torch.Tensor) -> torch.Tensor:
    if metric.ndim != 2 or metric.shape[0] != metric.shape[1]:
        raise ValueError("channel metric must be square")
    symmetric = 0.5 * (metric + metric.T)
    trace = torch.trace(symmetric)
    if not torch.isfinite(trace) or float(trace) <= 0.0:
        raise ValueError("channel metric must have positive finite trace")
    return symmetric * (metric.shape[0] / trace)


def trace_normalize_banded_metric(
    metrics: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    """Normalize the full separable spatial-channel metric to mean eigenvalue one."""

    if metrics.ndim != 3 or metrics.shape[1] != metrics.shape[2]:
        raise ValueError("banded metrics must have shape [bands,channels,channels]")
    if masks.ndim != 3 or masks.shape[0] != metrics.shape[0]:
        raise ValueError("band masks and metrics disagree")
    symmetric = 0.5 * (metrics + metrics.transpose(-1, -2))
    positions = masks.flatten(1).sum(dim=1).to(symmetric)
    total_trace = (positions * symmetric.diagonal(dim1=-2, dim2=-1).sum(dim=1)).sum()
    dimension = float(metrics.shape[1] * masks.shape[1] * masks.shape[2])
    if not torch.isfinite(total_trace) or float(total_trace) <= 0.0:
        raise ValueError("banded metric must have positive finite total trace")
    return symmetric * (dimension / total_trace)


def decoder_embed_metric(rae: torch.nn.Module) -> torch.Tensor:
    """Effective normalized-latent metric induced by decoder_embed."""

    weight = rae.decoder.decoder_embed.weight.detach().float()
    channel_metric = weight.T @ weight
    if bool(getattr(rae, "do_normalization", False)):
        latent_var = getattr(rae, "latent_var", None)
        if latent_var is not None:
            scale = torch.sqrt(
                latent_var.detach().to(
                    device=channel_metric.device, dtype=channel_metric.dtype
                )
                + float(getattr(rae, "eps", 1e-5))
            )
            scale = scale.reshape(scale.shape[0], -1)
            if scale.shape[0] != channel_metric.shape[0]:
                raise ValueError("RAE latent normalization does not match decoder channels")
            # Official RAE statistics are [C,H,W].  Averaging the exact
            # token-local embed metric gives this channel-shared proxy.
            scale_cross_moment = scale @ scale.T / float(scale.shape[1])
            channel_metric = channel_metric * scale_cross_moment
    return trace_normalize_channel_metric(channel_metric)


def channel_metric_loss(error: torch.Tensor, metric: torch.Tensor) -> torch.Tensor:
    if error.ndim != 4 or error.shape[1] != metric.shape[0]:
        raise ValueError("error and channel metric disagree")
    tokens = error.permute(0, 2, 3, 1)
    quadratic = torch.einsum("bhwc,cd,bhwd->bhw", tokens, metric.to(error), tokens)
    return quadratic.mean(dim=(1, 2)) / float(error.shape[1])


def banded_metric_loss(
    error: torch.Tensor,
    metrics: torch.Tensor,
    masks: torch.Tensor,
    basis: torch.Tensor,
) -> torch.Tensor:
    if error.ndim != 4 or error.shape[1] != metrics.shape[1]:
        raise ValueError("error and banded metric disagree")
    coefficients = dct2(error, basis)
    values = error.new_zeros(len(error))
    total_positions = error.shape[1] * error.shape[-2] * error.shape[-1]
    for band, mask in enumerate(masks):
        selected = coefficients[:, :, mask.to(coefficients.device)].transpose(1, 2)
        values += torch.einsum(
            "bnc,cd,bnd->b", selected, metrics[band].to(error), selected
        ) / float(total_positions)
    return values


def gradient_cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape != right.shape:
        raise ValueError("gradient tensors must have equal shapes")
    return F.cosine_similarity(left.flatten(1), right.flatten(1), dim=1, eps=1e-12)


def gradient_energy_distributions(
    gradient: torch.Tensor,
    *,
    basis: torch.Tensor,
    masks: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Normalized channel, token and DCT-band squared-gradient distributions."""

    if gradient.ndim != 4:
        raise ValueError("gradient must have shape [B,C,H,W]")
    energy = gradient.square()
    channel = energy.sum(dim=(-2, -1))
    token = energy.sum(dim=1).flatten(1)
    coefficients = dct2(gradient, basis).square()
    bands = torch.stack(
        [coefficients[:, :, mask.to(gradient.device)].sum(dim=(1, 2)) for mask in masks],
        dim=1,
    )

    def normalize(value: torch.Tensor) -> torch.Tensor:
        return value / value.sum(dim=1, keepdim=True).clamp_min(1e-30)

    return {"channel": normalize(channel), "token": normalize(token), "dct": normalize(bands)}


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape or left_array.ndim != 1:
        raise ValueError("Spearman inputs must be equal one-dimensional arrays")
    if len(left_array) < 3:
        raise ValueError("Spearman correlation needs at least three observations")
    left_rank = pd.Series(left_array).rank(method="average").to_numpy()
    right_rank = pd.Series(right_array).rank(method="average").to_numpy()
    if left_rank.std() == 0.0 or right_rank.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def loss_space_gate(
    exact_rows: pd.DataFrame,
    thresholds: Phase0GateThresholds = Phase0GateThresholds(),
) -> dict[str, object]:
    required = {
        "time_bin",
        "gradient_cosine_x0_dec",
        "decoder_reduction_x0",
        "decoder_reduction_dec",
    }
    missing = required.difference(exact_rows.columns)
    if missing:
        raise KeyError(f"exact rows are missing {sorted(missing)}")
    per_time = (
        exact_rows.groupby("time_bin", as_index=False)
        .agg(
            median_gradient_cosine=("gradient_cosine_x0_dec", "median"),
            mean_reduction_x0=("decoder_reduction_x0", "mean"),
            mean_reduction_dec=("decoder_reduction_dec", "mean"),
        )
        .sort_values("time_bin")
    )
    per_time["correction_ratio"] = per_time["mean_reduction_dec"] / per_time[
        "mean_reduction_x0"
    ].clip(lower=1e-12)
    distinct_bins = int(
        (per_time["median_gradient_cosine"] <= thresholds.gradient_cosine).sum()
    )
    better_bins = int(
        (per_time["correction_ratio"] >= 1.0 + thresholds.correction_advantage).sum()
    )
    total_x0 = float(exact_rows["decoder_reduction_x0"].mean())
    total_dec = float(exact_rows["decoder_reduction_dec"].mean())
    total_ratio = total_dec / max(total_x0, 1e-12)
    gates = {
        "gradient_signal_is_distinct": distinct_bins >= 2,
        "decoder_correction_advantage": total_ratio >= 1.0 + thresholds.correction_advantage,
        "correction_advantage_consistent": better_bins >= thresholds.correction_time_bins,
    }
    return {
        "pass": bool(all(gates.values())),
        "gates": gates,
        "distinct_time_bins": distinct_bins,
        "better_time_bins": better_bins,
        "mean_reduction_x0": total_x0,
        "mean_reduction_dec": total_dec,
        "correction_ratio": total_ratio,
        "per_time": per_time.to_dict(orient="records"),
    }


def proxy_gate(
    score_rows: pd.DataFrame,
    gradient_rows: pd.DataFrame,
    thresholds: Phase0GateThresholds = Phase0GateThresholds(),
) -> dict[str, object]:
    required_scores = {"proxy", "split", "l_dec", "l_proxy"}
    required_gradients = {"proxy", "gradient_cosine_proxy_dec"}
    if missing := required_scores.difference(score_rows.columns):
        raise KeyError(f"proxy score rows are missing {sorted(missing)}")
    if missing := required_gradients.difference(gradient_rows.columns):
        raise KeyError(f"proxy gradient rows are missing {sorted(missing)}")
    summaries = []
    for proxy in sorted(score_rows["proxy"].unique()):
        subset = score_rows[score_rows["proxy"] == proxy]
        correlations: dict[str, float] = {}
        per_time_correlations: dict[str, list[dict[str, float]]] = {}
        for split, rows in subset.groupby("split"):
            if "time_bin" in rows.columns:
                split_rows = [
                    {
                        "time_bin": int(time_bin),
                        "spearman": spearman_correlation(time_rows["l_proxy"], time_rows["l_dec"]),
                    }
                    for time_bin, time_rows in rows.groupby("time_bin")
                ]
                per_time_correlations[str(split)] = split_rows
                correlations[str(split)] = float(
                    np.median([entry["spearman"] for entry in split_rows])
                )
            else:
                correlations[str(split)] = spearman_correlation(rows["l_proxy"], rows["l_dec"])
        calibration = float(correlations.get("calibration", float("nan")))
        test = float(correlations.get("test", float("nan")))
        gap = abs(calibration - test) / max(abs(calibration), 0.05)
        gradient_subset = gradient_rows[gradient_rows["proxy"] == proxy]
        gradient_median = float(gradient_subset["gradient_cosine_proxy_dec"].median())
        gates = {
            "heldout_spearman": math.isfinite(test) and test >= thresholds.proxy_spearman,
            "gradient_alignment": gradient_median >= thresholds.proxy_gradient_cosine,
            "calibration_test_gap": math.isfinite(gap) and gap <= thresholds.proxy_split_gap,
        }
        summaries.append(
            {
                "proxy": proxy,
                "calibration_spearman": calibration,
                "test_spearman": test,
                "relative_split_gap": gap,
                "median_gradient_cosine": gradient_median,
                "per_time_spearman": per_time_correlations,
                **gates,
                "pass": bool(all(gates.values())),
            }
        )
    return {
        "pass": any(bool(row["pass"]) for row in summaries),
        "selected_proxy": next((row["proxy"] for row in summaries if row["pass"]), None),
        "proxies": summaries,
    }


def summarize_quadratic_metric(metric: torch.Tensor) -> Mapping[str, float]:
    eigenvalues = torch.linalg.eigvalsh(0.5 * (metric.double() + metric.double().T)).clamp_min(0)
    total = eigenvalues.sum().clamp_min(1e-30)
    participation = total.square() / eigenvalues.square().sum().clamp_min(1e-30)
    return {
        "trace": float(total),
        "minimum_eigenvalue": float(eigenvalues.min()),
        "maximum_eigenvalue": float(eigenvalues.max()),
        "condition_number_positive": float(
            eigenvalues.max() / eigenvalues[eigenvalues > 1e-12].min()
        ) if bool((eigenvalues > 1e-12).any()) else float("inf"),
        "effective_rank": float(participation),
    }


__all__ = [
    "DEFAULT_HIDDEN_FRACTIONS",
    "Phase0GateThresholds",
    "banded_metric_loss",
    "channel_metric_loss",
    "clean_from_velocity",
    "decoder_embed_metric",
    "decoder_hidden_features",
    "decoder_hidden_indices",
    "decoder_hidden_loss",
    "decoder_hidden_rms",
    "dct2",
    "dct_matrix",
    "gradient_cosine",
    "gradient_energy_distributions",
    "idct2",
    "loss_space_gate",
    "proxy_gate",
    "radial_dct_band_masks",
    "spearman_correlation",
    "static_linear_state",
    "summarize_quadratic_metric",
    "trace_normalize_banded_metric",
    "trace_normalize_channel_metric",
    "velocity_and_clean_losses",
]
