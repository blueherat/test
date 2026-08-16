"""Frozen-SiT intermediate velocity head used for Internal Guidance."""

from __future__ import annotations

import torch
import torch.nn as nn


def validate_internal_depth(model: nn.Module, internal_depth: int) -> int:
    depth = int(internal_depth)
    block_count = len(model.blocks)
    if depth < 1 or depth >= block_count:
        raise ValueError(
            f"internal depth must lie in [1, {block_count - 1}], found {depth}"
        )
    return depth


def create_internal_velocity_head(
    sit_module,
    model: nn.Module,
    *,
    latent_channels: int,
) -> nn.Module:
    """Create the same independent FinalLayer used by official IG."""

    if latent_channels < 1:
        raise ValueError("latent_channels must be positive")
    hidden_size = int(model.pos_embed.shape[-1])
    patch_size = int(model.x_embedder.patch_size[0])
    with torch.random.fork_rng(devices=[]):
        head = sit_module.FinalLayer(hidden_size, patch_size, latent_channels)
    with torch.no_grad():
        nn.init.zeros_(head.adaLN_modulation[-1].weight)
        nn.init.zeros_(head.adaLN_modulation[-1].bias)
        nn.init.zeros_(head.linear.weight)
        nn.init.zeros_(head.linear.bias)
    return head


def freeze_source_model(model: nn.Module) -> nn.Module:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model


def embed_sit_inputs(
    model: nn.Module,
    state: torch.Tensor,
    time_value: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if state.ndim != 4:
        raise ValueError("state must have shape [B,C,H,W]")
    if time_value.shape != (len(state),) or labels.shape != (len(state),):
        raise ValueError("time and labels must both have shape [B]")
    tokens = model.x_embedder(state) + model.pos_embed
    time_embedding = model.t_embedder(time_value)
    label_embedding = model.y_embedder(labels, False)
    return tokens, time_embedding + label_embedding


def extract_internal_features(
    model: nn.Module,
    state: torch.Tensor,
    time_value: torch.Tensor,
    labels: torch.Tensor,
    *,
    internal_depth: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the frozen prefix and return post-block tokens plus conditioning."""

    depth = validate_internal_depth(model, internal_depth)
    tokens, conditioning = embed_sit_inputs(model, state, time_value, labels)
    for block in model.blocks[:depth]:
        tokens = block(tokens, conditioning)
    return tokens, conditioning


def unpatchify_channels(
    model: nn.Module,
    tokens: torch.Tensor,
    *,
    channels: int,
) -> torch.Tensor:
    if tokens.ndim != 3 or channels < 1:
        raise ValueError("tokens must be [B,T,F] and channels must be positive")
    patch_size = int(model.x_embedder.patch_size[0])
    side = int(tokens.shape[1] ** 0.5)
    if side * side != tokens.shape[1]:
        raise ValueError("token count must form a square grid")
    expected = patch_size**2 * channels
    if tokens.shape[-1] != expected:
        raise ValueError(
            f"expected {expected} output features per token, found {tokens.shape[-1]}"
        )
    values = tokens.reshape(
        len(tokens),
        side,
        side,
        patch_size,
        patch_size,
        channels,
    )
    values = torch.einsum("nhwpqc->nchpwq", values)
    return values.reshape(
        len(tokens),
        channels,
        side * patch_size,
        side * patch_size,
    )


def internal_velocity_from_features(
    model: nn.Module,
    head: nn.Module,
    features: torch.Tensor,
    conditioning: torch.Tensor,
    *,
    latent_channels: int,
) -> torch.Tensor:
    projected = head(features, conditioning)
    return unpatchify_channels(model, projected, channels=latent_channels)


def full_velocity_from_features(
    model: nn.Module,
    features: torch.Tensor,
    conditioning: torch.Tensor,
    *,
    internal_depth: int,
    latent_channels: int,
) -> torch.Tensor:
    depth = validate_internal_depth(model, internal_depth)
    tokens = features
    for block in model.blocks[depth:]:
        tokens = block(tokens, conditioning)
    projected = model.final_layer(tokens, conditioning)
    full_output = model.unpatchify(projected)
    if full_output.shape[1] < latent_channels:
        raise ValueError("source model returned too few output channels")
    return full_output[:, :latent_channels]


def full_and_internal_velocity(
    model: nn.Module,
    head: nn.Module,
    state: torch.Tensor,
    time_value: torch.Tensor,
    labels: torch.Tensor,
    *,
    internal_depth: int,
    latent_channels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    features, conditioning = extract_internal_features(
        model,
        state,
        time_value,
        labels,
        internal_depth=internal_depth,
    )
    internal = internal_velocity_from_features(
        model,
        head,
        features,
        conditioning,
        latent_channels=latent_channels,
    )
    full = full_velocity_from_features(
        model,
        features,
        conditioning,
        internal_depth=internal_depth,
        latent_channels=latent_channels,
    )
    return full, internal


def select_internal_guidance_field(
    full: torch.Tensor,
    internal: torch.Tensor,
    *,
    mode: str,
    gamma: float = 0.0,
) -> torch.Tensor:
    """Select full/base or use ``full + gamma * (full - base)``."""

    if full.shape != internal.shape:
        raise ValueError("full and internal predictions must have identical shapes")
    if mode == "full":
        return full.float()
    if mode == "internal":
        return internal.float()
    if mode != "extrapolation":
        raise ValueError(f"unsupported Internal Guidance mode: {mode!r}")
    gamma_tensor = torch.tensor(float(gamma))
    if not torch.isfinite(gamma_tensor):
        raise ValueError("gamma must be finite")
    if float(gamma) == 0.0:
        return full.float()
    full_float = full.float()
    return full_float + float(gamma) * (full_float - internal.float())


class FrozenPrefix(nn.Module):
    """Compile-friendly frozen prefix used only while training the head."""

    def __init__(self, model: nn.Module, internal_depth: int) -> None:
        super().__init__()
        self.model = model
        self.internal_depth = validate_internal_depth(model, internal_depth)

    def forward(
        self,
        state: torch.Tensor,
        time_value: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return extract_internal_features(
            self.model,
            state,
            time_value,
            labels,
            internal_depth=self.internal_depth,
        )
