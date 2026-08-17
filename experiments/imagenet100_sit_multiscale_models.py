"""Model loading and shared-backbone evaluation for multiscale guidance."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

try:
    from experiments.imagenet100_sit_internal_v_head import (
        create_internal_velocity_head,
        embed_sit_inputs,
        internal_velocity_from_features,
    )
    from experiments.imagenet100_sit_prediction_targets import prediction_to_velocity
    from experiments.imagenet100_sit_static_pair import (
        FieldSemantics,
        output_to_field_velocity,
        resolve_field_semantics,
    )
    from experiments.imagenet100_sit_vx_dual_head import clean_prediction_to_velocity
    from experiments.train_imagenet100_sit_flow import (
        LATENT_SHAPE,
        NUM_CLASSES,
        sha256_file,
    )
    from experiments.train_imagenet100_sit_frozen_internal_v_head import (
        CLEAN_PROTOCOL,
        EPSILON_PROTOCOL,
        PROTOCOL,
    )
except ModuleNotFoundError:
    from imagenet100_sit_internal_v_head import (
        create_internal_velocity_head,
        embed_sit_inputs,
        internal_velocity_from_features,
    )
    from imagenet100_sit_prediction_targets import prediction_to_velocity
    from imagenet100_sit_static_pair import (
        FieldSemantics,
        output_to_field_velocity,
        resolve_field_semantics,
    )
    from imagenet100_sit_vx_dual_head import clean_prediction_to_velocity
    from train_imagenet100_sit_flow import LATENT_SHAPE, NUM_CLASSES, sha256_file
    from train_imagenet100_sit_frozen_internal_v_head import (
        CLEAN_PROTOCOL,
        EPSILON_PROTOCOL,
        PROTOCOL,
    )


@dataclass(frozen=True)
class InternalHeadSpec:
    name: str
    depth: int
    prediction_target: str
    denominator_floor: float
    module: nn.Module
    checkpoint: str
    checkpoint_sha256: str
    source_checkpoint_sha256: str


def load_sit_field_model(
    *,
    checkpoint_path: Path,
    weights: str,
    sit_module,
    source_metadata: dict,
    device: torch.device,
) -> tuple[nn.Module, FieldSemantics, dict[str, object]]:
    """Load one single-output SiT field with strict metadata validation."""

    checkpoint_path = checkpoint_path.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("official_sit") != source_metadata:
        raise ValueError(f"{checkpoint_path} uses a different official SiT revision")
    protocol = str(checkpoint.get("protocol"))
    config = checkpoint["config"]
    semantics = resolve_field_semantics(
        protocol=protocol,
        config=config,
        requested_path="auto",
    )
    model = sit_module.SiT_models[str(config["model_name"])](
        input_size=LATENT_SHAPE[-1],
        num_classes=NUM_CLASSES,
        class_dropout_prob=float(config["cfg_dropout"]),
    )
    state_key = "ema" if weights == "ema" else "model"
    model.load_state_dict(checkpoint[state_key], strict=True)
    model.to(device).eval().requires_grad_(False)
    metadata: dict[str, object] = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "weights": weights,
        "protocol": protocol,
        "model_name": str(config["model_name"]),
        "prediction_target": semantics.prediction_target,
        "denominator_floor": semantics.denominator_floor,
        "data_manifest_sha256": checkpoint.get("data_manifest_sha256"),
    }
    del checkpoint
    gc.collect()
    return model, semantics, metadata


def evaluate_sit_field(
    model: nn.Module,
    semantics: FieldSemantics,
    state: torch.Tensor,
    time_value: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    output = model(state, time_value, labels)
    return output_to_field_velocity(
        output,
        state=state,
        time_value=time_value,
        semantics=semantics,
    )


def load_internal_head_for_source(
    *,
    checkpoint_path: Path,
    name: str,
    head_weights: str,
    model: nn.Module,
    sit_module,
    source_checkpoint_path: Path,
    source_metadata: dict,
    device: torch.device,
) -> InternalHeadSpec:
    """Load only an auxiliary head and prove that it belongs to ``model``."""

    checkpoint_path = checkpoint_path.expanduser().resolve()
    source_checkpoint_path = source_checkpoint_path.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    protocols = {
        "velocity": PROTOCOL,
        "clean": CLEAN_PROTOCOL,
        "epsilon": EPSILON_PROTOCOL,
    }
    config = checkpoint["config"]
    prediction_target = str(config.get("prediction_target", "velocity"))
    if checkpoint.get("protocol") != protocols.get(prediction_target):
        raise ValueError(f"invalid protocol/target pair in {checkpoint_path}")
    if checkpoint.get("official_sit") != source_metadata:
        raise ValueError(f"{checkpoint_path} uses a different official SiT revision")
    configured_source = Path(config["source_checkpoint"]).expanduser().resolve()
    if configured_source != source_checkpoint_path:
        raise ValueError(
            f"head {checkpoint_path} was trained from {configured_source}, "
            f"not {source_checkpoint_path}"
        )
    source_digest = sha256_file(source_checkpoint_path)
    if source_digest != config["source_checkpoint_sha256"]:
        raise ValueError("head source SHA256 does not match the live source checkpoint")
    if str(config["source_state_key"]) != "ema":
        raise ValueError("the multiscale study requires heads trained from v800 EMA")
    depth = int(config["internal_depth"])
    head = create_internal_velocity_head(
        sit_module,
        model,
        latent_channels=LATENT_SHAPE[0],
    )
    state_key = "internal_head_ema" if head_weights == "ema" else "internal_head"
    head.load_state_dict(checkpoint[state_key], strict=True)
    head.to(device).eval().requires_grad_(False)
    result = InternalHeadSpec(
        name=name,
        depth=depth,
        prediction_target=prediction_target,
        denominator_floor=float(config.get("clean_velocity_denominator_floor", 0.05)),
        module=head,
        checkpoint=str(checkpoint_path),
        checkpoint_sha256=sha256_file(checkpoint_path),
        source_checkpoint_sha256=source_digest,
    )
    del checkpoint
    gc.collect()
    return result


def internal_prediction_to_velocity(
    prediction: torch.Tensor,
    *,
    state: torch.Tensor,
    time_value: torch.Tensor,
    spec: InternalHeadSpec,
) -> torch.Tensor:
    if spec.prediction_target == "velocity":
        return prediction.float()
    if spec.prediction_target == "clean":
        return clean_prediction_to_velocity(
            prediction,
            state=state,
            time_value=time_value,
            denominator_floor=spec.denominator_floor,
        )
    if spec.prediction_target == "epsilon":
        return prediction_to_velocity(
            prediction,
            state=state,
            time_value=time_value,
            prediction_target="epsilon",
            denominator_floor=spec.denominator_floor,
        ).float()
    raise ValueError(f"unsupported internal prediction target: {spec.prediction_target}")


def _source_velocity_from_tokens(
    model: nn.Module,
    tokens: torch.Tensor,
    conditioning: torch.Tensor,
) -> torch.Tensor:
    projected = model.final_layer(tokens, conditioning)
    output = model.unpatchify(projected)
    if output.shape[1] < LATENT_SHAPE[0]:
        raise ValueError("source model emitted too few channels")
    return output[:, : LATENT_SHAPE[0]].float()


def evaluate_source_with_heads(
    model: nn.Module,
    state: torch.Tensor,
    time_value: torch.Tensor,
    labels: torch.Tensor,
    *,
    heads: dict[str, InternalHeadSpec],
    raw_depths: tuple[int, ...] = (),
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[int, torch.Tensor]]:
    """Run one shared backbone and evaluate all requested trained/raw readouts."""

    required_depths = {spec.depth for spec in heads.values()} | set(raw_depths)
    block_count = len(model.blocks)
    if any(depth < 1 or depth > block_count for depth in required_depths):
        raise ValueError("requested readout depth lies outside the source backbone")
    tokens, conditioning = embed_sit_inputs(model, state, time_value, labels)
    trained: dict[str, torch.Tensor] = {}
    raw: dict[int, torch.Tensor] = {}
    heads_by_depth: dict[int, list[InternalHeadSpec]] = {}
    for spec in heads.values():
        heads_by_depth.setdefault(spec.depth, []).append(spec)

    for depth, block in enumerate(model.blocks, start=1):
        tokens = block(tokens, conditioning)
        for spec in heads_by_depth.get(depth, []):
            prediction = internal_velocity_from_features(
                model,
                spec.module,
                tokens,
                conditioning,
                latent_channels=LATENT_SHAPE[0],
            )
            trained[spec.name] = internal_prediction_to_velocity(
                prediction,
                state=state,
                time_value=time_value,
                spec=spec,
            )
        if depth in raw_depths:
            raw[depth] = _source_velocity_from_tokens(model, tokens, conditioning)

    full = _source_velocity_from_tokens(model, tokens, conditioning)
    if set(trained) != set(heads):
        raise RuntimeError("not all trained heads were evaluated")
    if set(raw) != set(raw_depths):
        raise RuntimeError("not all raw depths were evaluated")
    return full, trained, raw
