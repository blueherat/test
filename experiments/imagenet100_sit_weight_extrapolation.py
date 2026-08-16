"""Weight-space extrapolation utilities for matched SiT checkpoints."""

from __future__ import annotations

from collections.abc import Mapping

import torch


SUPPORTED_PROTOCOLS = frozenset(
    (
        "imagenet100_sit_linear_flow_v1",
        "imagenet100_sit_single_target_linear_flow_v2",
    )
)


def checkpoint_prediction_target(checkpoint: Mapping[str, object]) -> str:
    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint is missing a mapping config")
    return str(config.get("prediction_target", "velocity"))


def validate_weight_extrapolation_pair(
    strong: Mapping[str, object],
    weak: Mapping[str, object],
    *,
    weights: str,
) -> None:
    """Require two checkpoints to differ only by optimization progress."""

    if weights not in {"ema", "model"}:
        raise ValueError("weights must be 'ema' or 'model'")
    strong_protocol = str(strong.get("protocol"))
    weak_protocol = str(weak.get("protocol"))
    if strong_protocol != weak_protocol or strong_protocol not in SUPPORTED_PROTOCOLS:
        raise ValueError(
            "weight extrapolation requires one supported checkpoint protocol"
        )
    strong_config = strong.get("config")
    weak_config = weak.get("config")
    if not isinstance(strong_config, Mapping) or not isinstance(weak_config, Mapping):
        raise ValueError("both checkpoints must contain mapping configs")

    required_config_keys = (
        "model_name",
        "cfg_dropout",
        "global_batch_size",
        "seed",
    )
    optional_config_defaults = {
        "prediction_target": "velocity",
        "loss_space": "velocity",
        "denominator_floor": 1e-3,
        "time_sampler": "uniform",
        "time_logit_mean": -0.8,
        "time_logit_std": 0.8,
    }
    mismatches: list[str] = []
    for key in required_config_keys:
        if strong_config.get(key) != weak_config.get(key):
            mismatches.append(
                f"config.{key}: strong={strong_config.get(key)!r}, "
                f"weak={weak_config.get(key)!r}"
            )
    for key, default in optional_config_defaults.items():
        strong_value = strong_config.get(key, default)
        weak_value = weak_config.get(key, default)
        if strong_value != weak_value:
            mismatches.append(
                f"config.{key}: strong={strong_value!r}, weak={weak_value!r}"
            )
    for key in ("data_manifest_sha256", "official_sit"):
        if strong.get(key) != weak.get(key):
            mismatches.append(f"checkpoint {key} differs")
    if checkpoint_prediction_target(strong) != "velocity":
        mismatches.append("checkpoints are not native velocity predictors")
    if int(strong.get("step", -1)) <= int(weak.get("step", -1)):
        mismatches.append(
            f"strong step {strong.get('step')!r} must exceed weak step "
            f"{weak.get('step')!r}"
        )
    if weights not in strong or weights not in weak:
        mismatches.append(f"checkpoint is missing {weights!r} state dict")
    if mismatches:
        raise ValueError(
            "incompatible weight-extrapolation checkpoints:\n  "
            + "\n  ".join(mismatches)
        )

    strong_state = strong[weights]
    weak_state = weak[weights]
    if not isinstance(strong_state, Mapping) or not isinstance(weak_state, Mapping):
        raise ValueError(f"checkpoint {weights!r} entries must be mappings")
    validate_state_dict_pair(strong_state, weak_state)


def validate_state_dict_pair(
    strong: Mapping[str, torch.Tensor],
    weak: Mapping[str, torch.Tensor],
) -> None:
    if tuple(strong) != tuple(weak):
        strong_only = sorted(set(strong) - set(weak))
        weak_only = sorted(set(weak) - set(strong))
        raise ValueError(
            "state-dict keys/order differ: "
            f"strong_only={strong_only}, weak_only={weak_only}"
        )
    for name in strong:
        strong_value = strong[name]
        weak_value = weak[name]
        if not torch.is_tensor(strong_value) or not torch.is_tensor(weak_value):
            raise TypeError(f"state-dict entry {name!r} is not a tensor")
        if strong_value.shape != weak_value.shape:
            raise ValueError(f"state-dict shape differs for {name!r}")
        if strong_value.dtype != weak_value.dtype:
            raise ValueError(f"state-dict dtype differs for {name!r}")
        if not (strong_value.is_floating_point() or strong_value.is_complex()):
            if not torch.equal(strong_value, weak_value):
                raise ValueError(f"non-floating state differs for {name!r}")


def extrapolate_state_dict(
    strong: Mapping[str, torch.Tensor],
    weak: Mapping[str, torch.Tensor],
    *,
    scale: float,
) -> dict[str, torch.Tensor]:
    """Return ``strong + scale * (strong - weak)`` for every model tensor."""

    validate_state_dict_pair(strong, weak)
    result: dict[str, torch.Tensor] = {}
    for name in strong:
        strong_value = strong[name]
        weak_value = weak[name]
        if strong_value.is_floating_point() or strong_value.is_complex():
            if scale == 0.0:
                result[name] = strong_value.clone()
            else:
                result[name] = torch.add(
                    strong_value,
                    strong_value - weak_value,
                    alpha=float(scale),
                )
        else:
            result[name] = strong_value.clone()
    return result


def velocity_extrapolation(
    strong_velocity: torch.Tensor,
    weak_velocity: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    """Return the standard AutoGuidance field at one paired state/time."""

    if strong_velocity.shape != weak_velocity.shape:
        raise ValueError("strong and weak velocities must have identical shapes")
    if scale == 0.0:
        return strong_velocity
    return strong_velocity + float(scale) * (strong_velocity - weak_velocity)


def format_scale(scale: float) -> str:
    value = format(float(scale), ".8g")
    return value.replace("-", "m").replace(".", "p").replace("+", "")
