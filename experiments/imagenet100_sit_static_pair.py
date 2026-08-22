"""Shared semantics for mixing velocity fields from two SiT checkpoints."""

from __future__ import annotations

from dataclasses import dataclass, replace
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
ControlMode = Literal[
    "full_pair",
    "floor_only",
    "floor_residual",
    "pre_floor_pair",
    "post_floor_pair",
    "parallel_pair",
    "orthogonal_pair",
    "posterior_response",
]
CommonUniqueComponent = Literal[
    "x_common_on_v",
    "x_unique_to_v",
    "v_common_on_x",
    "v_unique_to_x",
]
CONTROL_MODES = (
    "full_pair",
    "floor_only",
    "floor_residual",
    "pre_floor_pair",
    "post_floor_pair",
    "parallel_pair",
    "orthogonal_pair",
    "posterior_response",
)
X_FLOOR_CONTROL_MODES = frozenset(
    ("floor_only", "floor_residual", "pre_floor_pair", "post_floor_pair")
)
WINDOW_CONTROL_MODES = frozenset(("pre_floor_pair", "post_floor_pair"))
COMMON_UNIQUE_COMPONENTS = (
    "x_common_on_v",
    "x_unique_to_v",
    "v_common_on_x",
    "v_unique_to_x",
)


@dataclass(frozen=True)
class FieldSemantics:
    protocol: str
    field_path: str
    prediction_target: str | None
    denominator_floor: float
    gate_activation: str | None


def with_inference_denominator_floor(
    semantics: FieldSemantics,
    inference_floor: float | None,
) -> FieldSemantics:
    """Override an x-field conversion floor without changing checkpoint metadata."""

    if inference_floor is None:
        return semantics
    if inference_floor <= 0 or inference_floor >= 0.5:
        raise ValueError("inference denominator floor must be in (0, 0.5)")
    is_x_field = semantics.prediction_target == "x" or (
        semantics.protocol == DUAL_OUTPUT_PROTOCOL and semantics.field_path == "x"
    )
    if not is_x_field:
        raise ValueError("an inference denominator override requires an x field")
    return replace(semantics, denominator_floor=float(inference_floor))


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


def x_floor_coefficient(
    time_value: torch.Tensor,
    *,
    denominator_floor: float,
) -> torch.Tensor:
    """Return the ideal JiT-x velocity attenuation caused by its denominator floor."""

    if denominator_floor <= 0 or denominator_floor >= 0.5:
        raise ValueError("denominator_floor must be in (0, 0.5)")
    remaining = (1.0 - time_value.float()).clamp_min(0.0)
    return remaining / remaining.clamp_min(float(denominator_floor))


def post_floor_window(
    time_value: torch.Tensor,
    *,
    denominator_floor: float,
    transition_width: float,
) -> torch.Tensor:
    """Smoothly partition the trajectory around ``t = 1 - denominator_floor``."""

    if transition_width <= 0 or transition_width >= denominator_floor:
        raise ValueError("transition_width must lie in (0, denominator_floor)")
    boundary = 1.0 - float(denominator_floor)
    start = boundary - 0.5 * float(transition_width)
    unit = ((time_value.float() - start) / float(transition_width)).clamp(0.0, 1.0)
    return unit.square() * (3.0 - 2.0 * unit)


def decompose_relative_to_anchor(
    anchor_velocity: torch.Tensor,
    direction: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a per-sample direction into components parallel/orthogonal to anchor."""

    if anchor_velocity.shape != direction.shape:
        raise ValueError("anchor velocity and direction must have identical shapes")
    reduce_dims = tuple(range(1, anchor_velocity.ndim))
    denominator = anchor_velocity.square().sum(
        dim=reduce_dims, keepdim=True
    ).clamp_min(torch.finfo(anchor_velocity.dtype).tiny)
    coefficient = (direction * anchor_velocity).sum(
        dim=reduce_dims, keepdim=True
    ) / denominator
    parallel = coefficient * anchor_velocity
    return parallel, direction - parallel


def project_onto_direction(
    value: torch.Tensor,
    direction: torch.Tensor,
) -> torch.Tensor:
    """Project each sample in ``value`` onto its paired ``direction`` vector."""

    if value.shape != direction.shape:
        raise ValueError("value and direction must have identical shapes")
    reduce_dims = tuple(range(1, value.ndim))
    denominator = direction.square().sum(
        dim=reduce_dims, keepdim=True
    ).clamp_min(torch.finfo(direction.dtype).tiny)
    coefficient = (value * direction).sum(
        dim=reduce_dims, keepdim=True
    ) / denominator
    return coefficient * direction


def common_unique_orthogonal_directions(
    anchor_velocity: torch.Tensor,
    x_velocity: torch.Tensor,
    v_velocity: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Split x-target/same-target guidance into reciprocal common/unique parts.

    The two full guidance directions use the empirically beneficial orientation
    ``anchor - other``. Each is first made orthogonal to the anchor field. The
    projection is intentionally reciprocal rather than pretending that two
    non-collinear vectors have one canonical shared component.
    """

    if (
        anchor_velocity.shape != x_velocity.shape
        or anchor_velocity.shape != v_velocity.shape
    ):
        raise ValueError("anchor, x, and v velocities must have identical shapes")
    _, x_orthogonal = decompose_relative_to_anchor(
        anchor_velocity,
        anchor_velocity - x_velocity,
    )
    _, v_orthogonal = decompose_relative_to_anchor(
        anchor_velocity,
        anchor_velocity - v_velocity,
    )
    x_common = project_onto_direction(x_orthogonal, v_orthogonal)
    v_common = project_onto_direction(v_orthogonal, x_orthogonal)
    return {
        "x_orthogonal": x_orthogonal,
        "v_orthogonal": v_orthogonal,
        "x_common_on_v": x_common,
        "x_unique_to_v": x_orthogonal - x_common,
        "v_common_on_x": v_common,
        "v_unique_to_x": v_orthogonal - v_common,
    }


def common_unique_guided_velocity(
    anchor_velocity: torch.Tensor,
    x_velocity: torch.Tensor,
    v_velocity: torch.Tensor,
    *,
    scale: float,
    component: CommonUniqueComponent,
) -> torch.Tensor:
    """Add one reciprocal common/unique guidance component to the anchor."""

    if component not in COMMON_UNIQUE_COMPONENTS:
        raise ValueError(f"unsupported common/unique component: {component}")
    if scale == 0.0:
        return anchor_velocity
    directions = common_unique_orthogonal_directions(
        anchor_velocity,
        x_velocity,
        v_velocity,
    )
    return anchor_velocity + float(scale) * directions[component]


def controlled_pair_velocity(
    anchor_velocity: torch.Tensor,
    other_velocity: torch.Tensor | None,
    *,
    time_value: torch.Tensor,
    scale: float,
    mode: ControlMode,
    other_prediction_target: str | None,
    other_denominator_floor: float,
    window_transition_width: float,
) -> torch.Tensor:
    """Apply a full or mechanism-isolating field extrapolation.

    ``floor_only`` and ``floor_residual`` decompose the full x/v field gap as

    ``v_x - v_v = (c(t) - 1) v_v + (v_x - c(t) v_v)``,

    where ``c(t)`` is the deterministic attenuation introduced by the JiT-x
    denominator floor. Consequently, their perturbations sum exactly to the
    full-pair perturbation at every state and time.
    """

    if mode not in CONTROL_MODES:
        raise ValueError(f"unsupported control mode: {mode}")
    if mode == "posterior_response":
        raise ValueError(
            "posterior_response requires a clean-estimator callback and must "
            "be evaluated by conditional_static_pair_velocity"
        )
    if scale == 0.0:
        return anchor_velocity
    if mode == "full_pair":
        if other_velocity is None:
            raise ValueError("full_pair requires other_velocity")
        return static_pair_velocity(anchor_velocity, other_velocity, scale=scale)

    if mode in X_FLOOR_CONTROL_MODES and other_prediction_target != "x":
        raise ValueError(f"{mode} requires an x-prediction other field")
    if mode in {"parallel_pair", "orthogonal_pair"}:
        if other_velocity is None:
            raise ValueError(f"{mode} requires other_velocity")
        full_perturbation = other_velocity - anchor_velocity
        parallel, orthogonal = decompose_relative_to_anchor(
            anchor_velocity,
            full_perturbation,
        )
        perturbation = parallel if mode == "parallel_pair" else orthogonal
        return anchor_velocity + float(scale) * perturbation

    coefficient = x_floor_coefficient(
        time_value,
        denominator_floor=other_denominator_floor,
    ).to(device=anchor_velocity.device, dtype=anchor_velocity.dtype)
    while coefficient.ndim < anchor_velocity.ndim:
        coefficient = coefficient.unsqueeze(-1)

    if mode == "floor_only":
        perturbation = (coefficient - 1.0) * anchor_velocity
    else:
        if other_velocity is None:
            raise ValueError(f"{mode} requires other_velocity")
        full_perturbation = other_velocity - anchor_velocity
        if mode == "floor_residual":
            perturbation = other_velocity - coefficient * anchor_velocity
        else:
            post_weight = post_floor_window(
                time_value,
                denominator_floor=other_denominator_floor,
                transition_width=window_transition_width,
            ).to(device=anchor_velocity.device, dtype=anchor_velocity.dtype)
            while post_weight.ndim < anchor_velocity.ndim:
                post_weight = post_weight.unsqueeze(-1)
            if mode == "pre_floor_pair":
                perturbation = (1.0 - post_weight) * full_perturbation
            elif mode == "post_floor_pair":
                perturbation = post_weight * full_perturbation
            else:  # pragma: no cover - guarded by CONTROL_MODES above
                raise AssertionError(mode)
    return anchor_velocity + float(scale) * perturbation
