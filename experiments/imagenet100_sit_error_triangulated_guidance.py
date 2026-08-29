"""Core utilities for Error-Triangulated Guidance (ETG) experiments.

ETG treats velocity-, clean-, and epsilon-prediction readouts attached to the
same frozen SiT feature tensor as three measurements of one latent weak field.
This module deliberately keeps the statistical estimate separate from rollout
quality: the three-cornered-hat estimate can be audited before it is used by a
sampler.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn as nn

try:
    from experiments.imagenet100_sit_internal_v_head import (
        create_internal_velocity_head,
        extract_internal_features,
        full_velocity_from_features,
        internal_velocity_from_features,
    )
    from experiments.imagenet100_sit_prediction_targets import prediction_to_velocity
    from experiments.imagenet100_sit_vx_dual_head import clean_prediction_to_velocity
    from experiments.train_imagenet100_sit_flow import LATENT_SHAPE, sha256_file
    from experiments.train_imagenet100_sit_frozen_internal_v_head import (
        CLEAN_PROTOCOL,
        EPSILON_PROTOCOL,
        PROTOCOL,
        create_frozen_internal_probe,
    )
except ModuleNotFoundError:
    from imagenet100_sit_internal_v_head import (
        create_internal_velocity_head,
        extract_internal_features,
        full_velocity_from_features,
        internal_velocity_from_features,
    )
    from imagenet100_sit_prediction_targets import prediction_to_velocity
    from imagenet100_sit_vx_dual_head import clean_prediction_to_velocity
    from train_imagenet100_sit_flow import LATENT_SHAPE, sha256_file
    from train_imagenet100_sit_frozen_internal_v_head import (
        CLEAN_PROTOCOL,
        EPSILON_PROTOCOL,
        PROTOCOL,
        create_frozen_internal_probe,
    )


TARGETS = ("velocity", "clean", "epsilon")
PAIR_NAMES = ("velocity_clean", "velocity_epsilon", "clean_epsilon")
PROTOCOLS = {
    "velocity": PROTOCOL,
    "clean": CLEAN_PROTOCOL,
    "epsilon": EPSILON_PROTOCOL,
}


def normalize_head_checkpoint(checkpoint: Mapping[str, object]) -> dict[str, object]:
    """Return the immutable metadata needed to compare three head checkpoints."""

    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("head checkpoint is missing config")
    prediction_target = str(config.get("prediction_target", "velocity"))
    if prediction_target not in TARGETS:
        raise ValueError(f"unsupported head target: {prediction_target!r}")
    if checkpoint.get("protocol") != PROTOCOLS[prediction_target]:
        raise ValueError(
            f"checkpoint protocol does not match {prediction_target!r} target"
        )
    return {
        "prediction_target": prediction_target,
        "step": int(checkpoint["step"]),
        "source_checkpoint": str(Path(str(config["source_checkpoint"])).resolve()),
        "source_checkpoint_sha256": str(config["source_checkpoint_sha256"]),
        "source_state_key": str(config["source_state_key"]),
        "source_step": int(config["source_step"]),
        "model_name": str(config["model_name"]),
        "cfg_dropout": float(config["cfg_dropout"]),
        "internal_depth": int(config["internal_depth"]),
        "denominator_floor": float(
            config.get("clean_velocity_denominator_floor", 0.05)
        ),
        "data_manifest_sha256": checkpoint.get("data_manifest_sha256"),
        "official_sit": checkpoint.get("official_sit"),
    }


def _shared_source_fields(metadata: Mapping[str, object]) -> dict[str, object]:
    keys = (
        "source_checkpoint",
        "source_checkpoint_sha256",
        "source_state_key",
        "source_step",
        "model_name",
        "cfg_dropout",
        "internal_depth",
        "denominator_floor",
        "data_manifest_sha256",
        "official_sit",
    )
    return {key: metadata[key] for key in keys}


def load_etg_model(
    *,
    checkpoint_paths: Mapping[str, Path],
    head_weights: str,
    sit_module,
    source_metadata: Mapping[str, object],
    device: torch.device,
) -> tuple[nn.Module, nn.ModuleDict, dict[str, object]]:
    """Load one frozen source model and three compatible internal readouts."""

    if set(checkpoint_paths) != set(TARGETS):
        raise ValueError(f"ETG requires exactly these heads: {TARGETS}")
    if head_weights not in {"ema", "model"}:
        raise ValueError("head_weights must be 'ema' or 'model'")

    loaded: dict[str, dict[str, object]] = {}
    metadata: dict[str, dict[str, object]] = {}
    for target in TARGETS:
        path = Path(checkpoint_paths[target]).expanduser().resolve()
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"invalid checkpoint payload: {path}")
        current = normalize_head_checkpoint(checkpoint)
        if current["prediction_target"] != target:
            raise ValueError(f"{path} is not a {target!r} head checkpoint")
        if current["official_sit"] != dict(source_metadata):
            raise ValueError(f"{target} head uses a different SiT revision")
        loaded[target] = checkpoint
        metadata[target] = {
            **current,
            "checkpoint": str(path),
            "checkpoint_sha256": sha256_file(path),
            "head_weights": head_weights,
        }

    source_reference = _shared_source_fields(metadata["velocity"])
    for target in TARGETS[1:]:
        if _shared_source_fields(metadata[target]) != source_reference:
            raise ValueError(f"{target} head does not share the v-head source protocol")

    source_path = Path(str(source_reference["source_checkpoint"]))
    if sha256_file(source_path) != source_reference["source_checkpoint_sha256"]:
        raise ValueError("source checkpoint SHA256 no longer matches head training")
    source_checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    source_state_key = str(source_reference["source_state_key"])
    if source_checkpoint.get("official_sit") != dict(source_metadata):
        raise ValueError("source checkpoint uses a different SiT revision")
    if int(source_checkpoint["step"]) != int(source_reference["source_step"]):
        raise ValueError("source checkpoint step differs from head training")

    model, velocity_head, probe_metadata = create_frozen_internal_probe(
        sit_module,
        model_name=str(source_reference["model_name"]),
        cfg_dropout=float(source_reference["cfg_dropout"]),
        source_state=source_checkpoint[source_state_key],
        internal_depth=int(source_reference["internal_depth"]),
    )
    heads = nn.ModuleDict({"velocity": velocity_head})
    for target in TARGETS[1:]:
        heads[target] = create_internal_velocity_head(
            sit_module,
            model,
            latent_channels=LATENT_SHAPE[0],
        )
    state_key = "internal_head_ema" if head_weights == "ema" else "internal_head"
    for target in TARGETS:
        heads[target].load_state_dict(loaded[target][state_key], strict=True)

    model.to(device).eval().requires_grad_(False)
    heads.to(device).eval().requires_grad_(False)
    result_metadata: dict[str, object] = {
        "source": source_reference,
        "heads": metadata,
        "probe": probe_metadata,
        "head_weights": head_weights,
        "targets": list(TARGETS),
    }
    del source_checkpoint, loaded
    gc.collect()
    return model, heads, result_metadata


def full_and_internal_predictions(
    model: nn.Module,
    heads: Mapping[str, nn.Module],
    state: torch.Tensor,
    time_value: torch.Tensor,
    labels: torch.Tensor,
    *,
    internal_depth: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return the final field and all three native head predictions in one pass."""

    if set(heads) != set(TARGETS):
        raise ValueError(f"expected heads {TARGETS}, found {tuple(heads)}")
    features, conditioning = extract_internal_features(
        model,
        state,
        time_value,
        labels,
        internal_depth=internal_depth,
    )
    predictions = {
        target: internal_velocity_from_features(
            model,
            heads[target],
            features,
            conditioning,
            latent_channels=LATENT_SHAPE[0],
        )
        for target in TARGETS
    }
    full = full_velocity_from_features(
        model,
        features,
        conditioning,
        internal_depth=internal_depth,
        latent_channels=LATENT_SHAPE[0],
    )
    return full, predictions


def predictions_to_velocity(
    predictions: Mapping[str, torch.Tensor],
    *,
    state: torch.Tensor,
    time_value: torch.Tensor,
    denominator_floor: float,
) -> dict[str, torch.Tensor]:
    """Convert the three native outputs to the common velocity space."""

    if set(predictions) != set(TARGETS):
        raise ValueError(f"expected predictions {TARGETS}, found {tuple(predictions)}")
    return {
        "velocity": predictions["velocity"].float(),
        "clean": clean_prediction_to_velocity(
            predictions["clean"],
            state=state,
            time_value=time_value,
            denominator_floor=denominator_floor,
        ),
        "epsilon": prediction_to_velocity(
            predictions["epsilon"],
            state=state,
            time_value=time_value,
            prediction_target="epsilon",
            denominator_floor=denominator_floor,
        ),
    }


def pairwise_squared_differences(
    predictions: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return elementwise squared differences in canonical target order."""

    return {
        "velocity_clean": (predictions["velocity"] - predictions["clean"]).square(),
        "velocity_epsilon": (
            predictions["velocity"] - predictions["epsilon"]
        ).square(),
        "clean_epsilon": (predictions["clean"] - predictions["epsilon"]).square(),
    }


def three_cornered_hat(
    pairwise_variances: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Recover raw per-view private variances under uncorrelated errors.

    The returned target axis is ordered as ``(velocity, clean, epsilon)``.
    No clipping is performed here; negative values are an important audit signal.
    """

    if set(pairwise_variances) != set(PAIR_NAMES):
        raise ValueError(f"expected pairwise entries {PAIR_NAMES}")
    d_vx = pairwise_variances["velocity_clean"]
    d_ve = pairwise_variances["velocity_epsilon"]
    d_xe = pairwise_variances["clean_epsilon"]
    if d_vx.shape != d_ve.shape or d_vx.shape != d_xe.shape:
        raise ValueError("pairwise variance tensors must have matching shapes")
    return 0.5 * torch.stack(
        (
            d_vx + d_ve - d_xe,
            d_vx + d_xe - d_ve,
            d_ve + d_xe - d_vx,
        ),
        dim=-2,
    )


def regularize_private_variances(
    raw: torch.Tensor,
    *,
    shrinkage: float,
    ridge_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clip invalid variances, lightly shrink them, and return inverse weights."""

    if raw.ndim < 2 or raw.shape[-2] != len(TARGETS):
        raise ValueError("raw variances must have a three-element target axis at -2")
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must lie in [0,1]")
    if ridge_fraction < 0.0:
        raise ValueError("ridge_fraction must be non-negative")
    clipped = raw.clamp_min(0.0)
    target_mean = clipped.mean(dim=-2, keepdim=True)
    regularized = (1.0 - shrinkage) * clipped + shrinkage * target_mean
    scale = target_mean.clamp_min(torch.finfo(raw.dtype).eps)
    regularized = regularized + ridge_fraction * scale
    precision = regularized.reciprocal()
    weights = precision / precision.sum(dim=-2, keepdim=True)
    return regularized, weights


def fuse_predictions(
    predictions: Mapping[str, torch.Tensor],
    weights: torch.Tensor,
) -> torch.Tensor:
    """Fuse three [B,C,H,W] fields using global or per-channel weights."""

    stacked = torch.stack([predictions[target].float() for target in TARGETS], dim=1)
    if weights.ndim == 1:
        if weights.shape != (len(TARGETS),):
            raise ValueError("global weights must have shape [3]")
        broadcast = weights.reshape(1, len(TARGETS), 1, 1, 1)
    elif weights.ndim == 2:
        if weights.shape != (len(TARGETS), stacked.shape[2]):
            raise ValueError("channel weights must have shape [3,C]")
        broadcast = weights.reshape(1, len(TARGETS), stacked.shape[2], 1, 1)
    else:
        raise ValueError("weights must have shape [3] or [3,C]")
    return (stacked * broadcast.to(stacked)).sum(dim=1)


def time_bin_index(time_value: float, edges: Sequence[float]) -> int:
    if len(edges) < 2:
        raise ValueError("time-bin edges require at least two values")
    if any(right <= left for left, right in zip(edges[:-1], edges[1:], strict=True)):
        raise ValueError("time-bin edges must be strictly increasing")
    value = min(max(float(time_value), float(edges[0])), float(edges[-1]))
    for index, right in enumerate(edges[1:]):
        if value < right or index == len(edges) - 2:
            return index
    raise AssertionError("unreachable time-bin lookup")


def guided_field(
    full: torch.Tensor,
    weak_predictions: Mapping[str, torch.Tensor],
    *,
    mode: str,
    gamma: float,
    weights: torch.Tensor | None = None,
    private_target: str | None = None,
) -> torch.Tensor:
    """Construct baseline, single-head, fused, or private-residual fields."""

    if mode == "baseline":
        return full.float()
    if mode.startswith("single_"):
        target = mode.removeprefix("single_")
        if target not in TARGETS:
            raise ValueError(f"unsupported single-head mode: {mode!r}")
        common = weak_predictions[target].float()
        return full.float() + float(gamma) * (full.float() - common)
    if mode == "mean":
        common = fuse_predictions(
            weak_predictions,
            torch.full((len(TARGETS),), 1.0 / len(TARGETS), device=full.device),
        )
        return full.float() + float(gamma) * (full.float() - common)
    if mode not in {"etg", "private"}:
        raise ValueError(f"unsupported ETG field mode: {mode!r}")
    if weights is None:
        raise ValueError(f"{mode} mode requires calibrated weights")
    common = fuse_predictions(weak_predictions, weights)
    if mode == "etg":
        direction = full.float() - common
    else:
        if private_target not in TARGETS:
            raise ValueError("private mode requires one valid private_target")
        direction = weak_predictions[str(private_target)].float() - common
    return full.float() + float(gamma) * direction
