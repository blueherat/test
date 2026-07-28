"""Paired geometry metrics for Flow and LPL errors in an RAE decoder.

The functions in this module are intentionally model-agnostic.  The expensive
decoder and stage-2 forwards live in the runner; this file defines the
equal-norm controls, finite-difference pullback metric, and paired summaries.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def sample_rms(value: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    """Return one root-mean-square value per sample."""

    if value.ndim < 2:
        raise ValueError("sample_rms expects a batch dimension and at least one feature dimension")
    squared_mean = value.square().flatten(1).mean(dim=1)
    return (squared_mean + float(eps)).sqrt()


def expand_sample_scalar(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if value.ndim != 1 or len(value) != len(target):
        raise ValueError("expected one scalar per target sample")
    return value.reshape(len(value), *([1] * (target.ndim - 1)))


def unit_rms_direction(error: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize each error to unit RMS without changing its direction."""

    scale = sample_rms(error).clamp_min(float(eps))
    return error / expand_sample_scalar(scale, error)


def scale_direction_to_rms(
    direction: torch.Tensor,
    target_rms: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Scale arbitrary directions to an exact per-sample RMS."""

    if target_rms.ndim != 1 or len(target_rms) != len(direction):
        raise ValueError("target_rms must contain one value per direction sample")
    return unit_rms_direction(direction, eps=eps) * expand_sample_scalar(target_rms, direction)


def paired_amplitudes(
    clean: torch.Tensor,
    flow_error: torch.Tensor,
    lpl_error: torch.Tensor,
    *,
    local_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return realistic matched RMS and a locally capped matched RMS."""

    if clean.shape != flow_error.shape or clean.shape != lpl_error.shape:
        raise ValueError("clean and both errors must have identical shapes")
    if not 0.0 < float(local_fraction) <= 1.0:
        raise ValueError("local_fraction must be in (0, 1]")
    realistic = torch.minimum(sample_rms(flow_error), sample_rms(lpl_error))
    local_cap = float(local_fraction) * sample_rms(clean)
    return realistic, torch.minimum(realistic, local_cap)


def shuffled_direction(
    direction: torch.Tensor,
    *,
    channel_shift: int = 1,
    row_shift: int = 1,
    column_shift: int = 1,
) -> torch.Tensor:
    """Deterministically disrupt channel/token alignment while preserving RMS."""

    if direction.ndim != 4:
        raise ValueError("shuffled_direction expects BCHW tensors")
    return torch.roll(
        direction,
        shifts=(int(channel_shift), int(row_shift), int(column_shift)),
        dims=(1, 2, 3),
    )


def raw_feature_layer_losses(
    candidate: Sequence[torch.Tensor],
    reference: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Return per-sample, per-layer decoder hidden-state MSE."""

    if not candidate or len(candidate) != len(reference):
        raise ValueError("candidate and reference feature pyramids must align")
    losses = []
    for candidate_layer, reference_layer in zip(candidate, reference, strict=True):
        if candidate_layer.shape != reference_layer.shape:
            raise ValueError("candidate and reference layer shapes must match")
        losses.append((candidate_layer - reference_layer).square().flatten(1).mean(dim=1))
    return torch.stack(losses, dim=1)


def raw_feature_loss(
    candidate: Sequence[torch.Tensor],
    reference: Sequence[torch.Tensor],
) -> torch.Tensor:
    return raw_feature_layer_losses(candidate, reference).mean(dim=1)


def _feature_channels_by_position(feature: torch.Tensor) -> torch.Tensor:
    if feature.ndim == 3:
        return feature.transpose(1, 2)
    if feature.ndim == 4:
        return feature.flatten(2)
    raise ValueError("decoder features must have shape [B,N,C] or [B,C,H,W]")


def feature_normalization_decomposition(
    candidate: Sequence[torch.Tensor],
    reference: Sequence[torch.Tensor],
    *,
    eps: float = 1e-6,
) -> Mapping[str, torch.Tensor]:
    """Separate clean-, prediction-, and symmetrically normalized feature errors.

    Values use the same spatial-sum/channel-mean scale as strict LPL, but omit
    its outlier mask.  ``prediction_normalized`` is therefore the mask-free
    counterpart of strict LPL.  All returned tensors have shape ``[B, layers]``.
    """

    if not candidate or len(candidate) != len(reference):
        raise ValueError("candidate and reference feature pyramids must align")
    metrics: dict[str, list[torch.Tensor]] = {
        "target_normalized": [],
        "prediction_normalized": [],
        "symmetric_normalized": [],
        "prediction_over_target_variance_gmean": [],
        "centered_channel_cosine": [],
    }
    for candidate_layer, reference_layer in zip(candidate, reference, strict=True):
        candidate_flat = _feature_channels_by_position(candidate_layer)
        reference_flat = _feature_channels_by_position(reference_layer)
        if candidate_flat.shape != reference_flat.shape:
            raise ValueError("candidate and reference layer shapes must match")
        positions = candidate_flat.shape[-1]
        difference_mse = (candidate_flat - reference_flat).square().mean(dim=-1)
        candidate_centered = candidate_flat - candidate_flat.mean(dim=-1, keepdim=True)
        reference_centered = reference_flat - reference_flat.mean(dim=-1, keepdim=True)
        candidate_variance = candidate_centered.square().mean(dim=-1)
        reference_variance = reference_centered.square().mean(dim=-1)
        symmetric_variance = 0.5 * (candidate_variance + reference_variance)
        metrics["target_normalized"].append(
            positions
            * (difference_mse / (reference_variance + float(eps))).mean(dim=1)
        )
        metrics["prediction_normalized"].append(
            positions
            * (difference_mse / (candidate_variance + float(eps))).mean(dim=1)
        )
        metrics["symmetric_normalized"].append(
            positions
            * (difference_mse / (symmetric_variance + float(eps))).mean(dim=1)
        )
        log_variance_ratio = torch.log(candidate_variance + float(eps)) - torch.log(
            reference_variance + float(eps)
        )
        metrics["prediction_over_target_variance_gmean"].append(
            log_variance_ratio.mean(dim=1).exp()
        )
        numerator = (candidate_centered * reference_centered).sum(dim=-1)
        denominator = (
            candidate_centered.square().sum(dim=-1)
            * reference_centered.square().sum(dim=-1)
        ).clamp_min(float(eps) ** 2).sqrt()
        metrics["centered_channel_cosine"].append(
            (numerator / denominator).mean(dim=1)
        )
    return {
        name: torch.stack(layer_values, dim=1)
        for name, layer_values in metrics.items()
    }


def finite_difference_feature_gain(
    plus_features: Sequence[torch.Tensor],
    minus_features: Sequence[torch.Tensor],
    step_rms: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Approximate ``||J_phi u||^2`` for unit-RMS latent directions.

    ``plus_features`` and ``minus_features`` must be evaluated at
    ``z +/- h*u``, where ``u`` has unit latent RMS and ``h=step_rms``.
    The result is the local decoder feature amplification per layer and its
    layer mean.
    """

    if step_rms.ndim != 1:
        raise ValueError("step_rms must contain one value per sample")
    layer_squared = raw_feature_layer_losses(plus_features, minus_features)
    denominator = (2.0 * step_rms).square().clamp_min(1e-30)
    layer_gain = layer_squared / denominator[:, None]
    return layer_gain, layer_gain.mean(dim=1)


def cosine_per_sample(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape != right.shape:
        raise ValueError("cosine inputs must have identical shapes")
    return F.cosine_similarity(left.flatten(1), right.flatten(1), dim=1, eps=1e-12)


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(left_array) & np.isfinite(right_array)
    left_array = left_array[finite]
    right_array = right_array[finite]
    if len(left_array) < 3:
        return float("nan")
    left_rank = pd.Series(left_array).rank(method="average").to_numpy()
    right_rank = pd.Series(right_array).rank(method="average").to_numpy()
    if left_rank.std() == 0.0 or right_rank.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def geometric_mean(values: Sequence[float], eps: float = 1e-30) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array) & (array >= 0.0)]
    if not len(array):
        return float("nan")
    return float(np.exp(np.log(np.clip(array, float(eps), None)).mean()))


def _long_prediction_table(rows: pd.DataFrame) -> pd.DataFrame:
    values = []
    for _, row in rows.iterrows():
        for branch in ("flow", "lpl"):
            values.append(
                {
                    "training_seed": int(row["training_seed"]),
                    "sample_index": int(row["sample_index"]),
                    "noise_to_signal_ratio": float(row["noise_to_signal_ratio"]),
                    "branch": branch,
                    "latent_mse": float(row[f"{branch}_latent_mse"]),
                    "actual_raw_loss": float(row[f"{branch}_actual_raw_loss"]),
                    "quadratic_prediction": float(row[f"{branch}_quadratic_prediction"]),
                    "actual_strict_lpl": float(row[f"{branch}_actual_strict_lpl"]),
                    "strict_quadratic_prediction": float(
                        row[f"{branch}_strict_quadratic_prediction"]
                    ),
                }
            )
    return pd.DataFrame(values)


def summarize_geometry_rows(
    rows: pd.DataFrame,
    *,
    required_seed_count: int = 4,
    minimum_seed_improvement: float = 0.10,
    minimum_quadratic_spearman: float = 0.70,
    minimum_control_degradation: float = 0.05,
) -> tuple[pd.DataFrame, Mapping[str, object]]:
    """Summarize paired rows and apply the preregistered mechanism gate."""

    required = {
        "training_seed",
        "sample_index",
        "noise_to_signal_ratio",
        "flow_latent_mse",
        "lpl_latent_mse",
        "flow_actual_raw_loss",
        "lpl_actual_raw_loss",
        "flow_local_gain",
        "lpl_local_gain",
        "random_local_gain",
        "shuffled_lpl_local_gain",
        "flow_fd_gain",
        "lpl_fd_gain",
        "random_fd_gain",
        "flow_quadratic_prediction",
        "lpl_quadratic_prediction",
        "flow_local_strict_gain",
        "lpl_local_strict_gain",
        "flow_fd_strict_gain",
        "lpl_fd_strict_gain",
        "flow_strict_quadratic_prediction",
        "lpl_strict_quadratic_prediction",
    }
    missing = required.difference(rows.columns)
    if missing:
        raise KeyError(f"geometry rows are missing columns: {sorted(missing)}")

    seed_rows = []
    for training_seed, group in rows.groupby("training_seed", sort=True):
        local_ratio = group["lpl_local_gain"] / group["flow_local_gain"].clip(lower=1e-30)
        fd_ratio = group["lpl_fd_gain"] / group["flow_fd_gain"].clip(lower=1e-30)
        random_over_lpl = group["random_local_gain"] / group["lpl_local_gain"].clip(
            lower=1e-30
        )
        shuffle_over_lpl = group["shuffled_lpl_local_gain"] / group[
            "lpl_local_gain"
        ].clip(lower=1e-30)
        actual_ratio = group["lpl_actual_raw_loss"] / group["flow_actual_raw_loss"].clip(
            lower=1e-30
        )
        local_strict_ratio = group["lpl_local_strict_gain"] / group[
            "flow_local_strict_gain"
        ].clip(lower=1e-30)
        fd_strict_ratio = group["lpl_fd_strict_gain"] / group[
            "flow_fd_strict_gain"
        ].clip(lower=1e-30)
        seed_rows.append(
            {
                "training_seed": int(training_seed),
                "observations": int(len(group)),
                "actual_raw_lpl_over_flow_gmean": geometric_mean(actual_ratio),
                "local_gain_lpl_over_flow_gmean": geometric_mean(local_ratio),
                "local_lpl_better_fraction": float((local_ratio < 1.0).mean()),
                "fd_gain_lpl_over_flow_gmean": geometric_mean(fd_ratio),
                "local_strict_gain_lpl_over_flow_gmean": geometric_mean(
                    local_strict_ratio
                ),
                "fd_strict_gain_lpl_over_flow_gmean": geometric_mean(fd_strict_ratio),
                "random_over_lpl_local_gain_gmean": geometric_mean(random_over_lpl),
                "shuffle_over_lpl_local_gain_gmean": geometric_mean(shuffle_over_lpl),
            }
        )
    seed_table = pd.DataFrame(seed_rows)

    predictions = _long_prediction_table(rows)
    latent_spearman = spearman(predictions["latent_mse"], predictions["actual_raw_loss"])
    quadratic_spearman = spearman(
        predictions["quadratic_prediction"], predictions["actual_raw_loss"]
    )
    strict_quadratic_spearman = spearman(
        predictions["strict_quadratic_prediction"],
        predictions["actual_strict_lpl"],
    )
    seed_threshold = 1.0 - float(minimum_seed_improvement)
    improved_seed_count = int(
        (seed_table["local_gain_lpl_over_flow_gmean"] <= seed_threshold).sum()
    )
    required_improved_seeds = max(1, int(required_seed_count) - 1)
    random_control = geometric_mean(
        rows["random_local_gain"] / rows["lpl_local_gain"].clip(lower=1e-30)
    )
    shuffle_control = geometric_mean(
        rows["shuffled_lpl_local_gain"] / rows["lpl_local_gain"].clip(lower=1e-30)
    )
    controls_pass = bool(
        random_control >= 1.0 + float(minimum_control_degradation)
        and shuffle_control >= 1.0 + float(minimum_control_degradation)
    )
    gate = {
        "seed_count": int(seed_table["training_seed"].nunique()),
        "required_seed_count": int(required_seed_count),
        "improved_seed_count": improved_seed_count,
        "required_improved_seeds": required_improved_seeds,
        "minimum_seed_improvement": float(minimum_seed_improvement),
        "latent_mse_to_actual_raw_spearman": latent_spearman,
        "quadratic_prediction_to_actual_raw_spearman": quadratic_spearman,
        "strict_quadratic_prediction_to_actual_strict_spearman": (
            strict_quadratic_spearman
        ),
        "minimum_quadratic_spearman": float(minimum_quadratic_spearman),
        "random_over_lpl_local_gain_gmean": random_control,
        "shuffle_over_lpl_local_gain_gmean": shuffle_control,
        "minimum_control_degradation": float(minimum_control_degradation),
        "controls_pass": controls_pass,
    }
    gate["mechanism_supported"] = bool(
        gate["seed_count"] >= int(required_seed_count)
        and improved_seed_count >= required_improved_seeds
        and np.isfinite(quadratic_spearman)
        and quadratic_spearman >= float(minimum_quadratic_spearman)
        and quadratic_spearman > latent_spearman
        and controls_pass
    )
    return seed_table, gate
