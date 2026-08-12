"""Shared semantics for mixing velocity fields from two SiT checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

try:
    from experiments.imagenet100_sit_dual_output import dual_output_velocities
    from experiments.imagenet100_sit_prediction_targets import prediction_to_velocity
except ModuleNotFoundError:
    from imagenet100_sit_dual_output import dual_output_velocities
    from imagenet100_sit_prediction_targets import prediction_to_velocity


LEGACY_PROTOCOL = "imagenet100_sit_linear_flow_v1"
SINGLE_TARGET_PROTOCOL = "imagenet100_sit_single_target_linear_flow_v2"
DUAL_OUTPUT_PROTOCOL = "imagenet100_sit_dual_output_linear_flow_v1"
SUPPORTED_PROTOCOLS = frozenset(
    (LEGACY_PROTOCOL, SINGLE_TARGET_PROTOCOL, DUAL_OUTPUT_PROTOCOL)
)
FieldPath = Literal["auto", "x", "epsilon", "dynamic"]


@dataclass(frozen=True)
class FieldSemantics:
    protocol: str
    field_path: str
    prediction_target: str | None
    denominator_floor: float
    gate_activation: str | None


def resolve_field_semantics(
    *,
    protocol: str,
    config: dict,
    requested_path: str,
) -> FieldSemantics:
    if protocol not in SUPPORTED_PROTOCOLS:
        raise ValueError(f"unsupported SiT checkpoint protocol: {protocol!r}")
    if protocol == DUAL_OUTPUT_PROTOCOL:
        if requested_path not in {"x", "epsilon", "dynamic"}:
            raise ValueError("dual-output checkpoints require x, epsilon, or dynamic")
        return FieldSemantics(
            protocol=protocol,
            field_path=requested_path,
            prediction_target=None,
            denominator_floor=float(config["denominator_floor"]),
            gate_activation=str(config["gate_activation"]),
        )
    if requested_path != "auto":
        raise ValueError("single-output checkpoints require field path 'auto'")
    return FieldSemantics(
        protocol=protocol,
        field_path="auto",
        prediction_target=str(config.get("prediction_target", "velocity")),
        denominator_floor=float(config.get("denominator_floor", 1e-3)),
        gate_activation=None,
    )


def output_to_field_velocity(
    output: torch.Tensor,
    *,
    state: torch.Tensor,
    time_value: torch.Tensor,
    semantics: FieldSemantics,
) -> torch.Tensor:
    if semantics.protocol == DUAL_OUTPUT_PROTOCOL:
        assert semantics.gate_activation is not None
        return dual_output_velocities(
            output,
            state=state,
            time_value=time_value,
            gate_activation=semantics.gate_activation,
            denominator_floor=semantics.denominator_floor,
        )[semantics.field_path].float()
    assert semantics.prediction_target is not None
    return prediction_to_velocity(
        output,
        state=state,
        time_value=time_value,
        prediction_target=semantics.prediction_target,
        denominator_floor=semantics.denominator_floor,
    ).float()


def static_pair_velocity(
    anchor_velocity: torch.Tensor,
    other_velocity: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    """Return ``anchor + scale * (other - anchor)`` with exact endpoints."""

    if anchor_velocity.shape != other_velocity.shape:
        raise ValueError("anchor and other velocities must have identical shapes")
    if scale == 0.0:
        return anchor_velocity
    if scale == 1.0:
        return other_velocity
    return anchor_velocity + float(scale) * (other_velocity - anchor_velocity)
