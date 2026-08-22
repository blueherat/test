"""Two-stage additive innovation parameterization for ImageNet-100 SiT.

The model is deliberately stricter than an ordinary auxiliary-head setup:

* stage 1 trains a shallow cumulative predictor ``weak``;
* stage 2 freezes that predictor and trains only the remaining blocks to emit
  an additive ``innovation``;
* the sampling field is always ``weak + innovation``.

This module contains only architecture and phase-isolation logic.  The
training protocol lives in ``train_imagenet100_sit_progressive_innovation``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from experiments.imagenet100_sit_internal_v_head import (
        create_internal_velocity_head,
        unpatchify_channels,
        validate_internal_depth,
    )
except ModuleNotFoundError:
    from imagenet100_sit_internal_v_head import (
        create_internal_velocity_head,
        unpatchify_channels,
        validate_internal_depth,
    )


PHASES = ("weak", "innovation")


class ProgressiveInnovationSiT(nn.Module):
    """Split a SiT into a frozen cumulative stage and an additive refinement."""

    def __init__(
        self,
        source: nn.Module,
        weak_head: nn.Module,
        *,
        split_depth: int,
        latent_channels: int,
    ) -> None:
        super().__init__()
        if latent_channels < 1:
            raise ValueError("latent_channels must be positive")
        self.source = source
        self.weak_head = weak_head
        self.split_depth = validate_internal_depth(source, split_depth)
        if self.split_depth >= len(source.blocks):
            raise ValueError("split depth must leave at least one innovation block")
        self.latent_channels = int(latent_channels)

    def _embed(
        self,
        state: torch.Tensor,
        time_value: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state.ndim != 4:
            raise ValueError("state must have shape [B,C,H,W]")
        if time_value.shape != (len(state),) or labels.shape != (len(state),):
            raise ValueError("time and labels must both have shape [B]")
        tokens = self.source.x_embedder(state) + self.source.pos_embed
        time_embedding = self.source.t_embedder(time_value)
        label_embedding = self.source.y_embedder(labels, self.training)
        return tokens, time_embedding + label_embedding

    def _weak_from_tokens(
        self,
        tokens: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        projected = self.weak_head(tokens, conditioning)
        return unpatchify_channels(
            self.source,
            projected,
            channels=self.latent_channels,
        )

    def _innovation_from_tokens(
        self,
        tokens: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        projected = self.source.final_layer(tokens, conditioning)
        output = self.source.unpatchify(projected)
        if output.shape[1] < self.latent_channels:
            raise ValueError("source final layer returned too few channels")
        return output[:, : self.latent_channels]

    def forward_weak(
        self,
        state: torch.Tensor,
        time_value: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        tokens, conditioning = self._embed(state, time_value, labels)
        for block in self.source.blocks[: self.split_depth]:
            tokens = block(tokens, conditioning)
        return self._weak_from_tokens(tokens, conditioning)

    def forward_components(
        self,
        state: torch.Tensor,
        time_value: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens, conditioning = self._embed(state, time_value, labels)
        for block in self.source.blocks[: self.split_depth]:
            tokens = block(tokens, conditioning)
        weak = self._weak_from_tokens(tokens, conditioning)
        for block in self.source.blocks[self.split_depth :]:
            tokens = block(tokens, conditioning)
        innovation = self._innovation_from_tokens(tokens, conditioning)
        return weak, innovation, weak + innovation

    def forward(
        self,
        state: torch.Tensor,
        time_value: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_components(state, time_value, labels)[2]


class WeakStage(nn.Module):
    """Compile-friendly stage-1 view of a progressive model."""

    def __init__(self, model: ProgressiveInnovationSiT) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        state: torch.Tensor,
        time_value: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.forward_weak(state, time_value, labels)


class InnovationStage(nn.Module):
    """Compile-friendly stage-2 view returning all cumulative components."""

    def __init__(self, model: ProgressiveInnovationSiT) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        state: torch.Tensor,
        time_value: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model.forward_components(state, time_value, labels)


def create_progressive_innovation_sit(
    sit_module,
    *,
    model_name: str,
    num_classes: int,
    input_size: int,
    cfg_dropout: float,
    split_depth: int,
    latent_channels: int,
) -> ProgressiveInnovationSiT:
    """Create a baseline-identical SiT plus one zero-initialized weak head."""

    source = sit_module.SiT_models[model_name](
        input_size=input_size,
        num_classes=num_classes,
        class_dropout_prob=cfg_dropout,
    )
    weak_head = create_internal_velocity_head(
        sit_module,
        source,
        latent_channels=latent_channels,
    )
    return ProgressiveInnovationSiT(
        source,
        weak_head,
        split_depth=split_depth,
        latent_channels=latent_channels,
    )


def configure_training_phase(
    model: ProgressiveInnovationSiT,
    phase: str,
) -> ProgressiveInnovationSiT:
    """Expose exactly one disjoint parameter group for a training stage."""

    if phase not in PHASES:
        raise ValueError(f"unsupported innovation phase: {phase!r}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    if phase == "weak":
        modules = (
            model.source.x_embedder,
            model.source.t_embedder,
            model.source.y_embedder,
            *model.source.blocks[: model.split_depth],
            model.weak_head,
        )
    else:
        modules = (
            *model.source.blocks[model.split_depth :],
            model.source.final_layer,
        )
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    return model


def trainable_parameter_names(model: nn.Module) -> tuple[str, ...]:
    return tuple(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )


def innovation_losses(
    weak: torch.Tensor,
    innovation: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return equivalent residual and cumulative stage-2 objectives."""

    if weak.shape != innovation.shape or weak.shape != target.shape:
        raise ValueError("weak, innovation and target must have identical shapes")
    weak_float = weak.detach().float()
    innovation_float = innovation.float()
    target_float = target.float()
    residual_target = target_float - weak_float
    innovation_loss = (innovation_float - residual_target).square().mean()
    cumulative = weak_float + innovation_float
    cumulative_loss = (cumulative - target_float).square().mean()
    return {
        "optimized": innovation_loss,
        "cumulative": cumulative_loss,
        "weak": (weak_float - target_float).square().mean(),
        "innovation_rms": innovation_float.square().mean().sqrt(),
        "residual_rms": residual_target.square().mean().sqrt(),
    }
