"""Hidden-state interpolation and extrapolation for a frozen SiT model."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

try:
    from experiments.imagenet100_sit_internal_v_head import (
        embed_sit_inputs,
        validate_internal_depth,
    )
except ModuleNotFoundError:
    from imagenet100_sit_internal_v_head import (
        embed_sit_inputs,
        validate_internal_depth,
    )


def internal_and_final_hidden_states(
    model: nn.Module,
    state: torch.Tensor,
    time_value: torch.Tensor,
    labels: torch.Tensor,
    *,
    internal_depth: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return post-depth tokens, final-block tokens, and conditioning."""

    depth = validate_internal_depth(model, internal_depth)
    tokens, conditioning = embed_sit_inputs(model, state, time_value, labels)
    internal: torch.Tensor | None = None
    for block_index, block in enumerate(model.blocks, start=1):
        tokens = block(tokens, conditioning)
        if block_index == depth:
            internal = tokens
    if internal is None:
        raise RuntimeError("failed to capture the requested internal hidden state")
    return internal, tokens, conditioning


def velocity_from_hidden_state(
    model: nn.Module,
    hidden_state: torch.Tensor,
    conditioning: torch.Tensor,
    *,
    latent_channels: int,
) -> torch.Tensor:
    """Apply the frozen source FinalLayer to one token hidden state."""

    projected = model.final_layer(hidden_state, conditioning)
    full_output = model.unpatchify(projected)
    if full_output.shape[1] < latent_channels:
        raise ValueError("source FinalLayer returned too few output channels")
    return full_output[:, :latent_channels]


def select_hidden_state_field(
    model: nn.Module,
    internal_hidden: torch.Tensor,
    final_hidden: torch.Tensor,
    conditioning: torch.Tensor,
    *,
    latent_channels: int,
    mode: str,
    gamma: float = 0.0,
    alpha: float = 0.0,
    extrapolation_space: str = "hidden",
) -> torch.Tensor:
    """Read out h_final, h_internal, or a frozen-model mixture of the two.

    ``hidden`` extrapolates before the nonlinear conditional FinalLayer:

        FinalLayer(h_final + gamma * (h_final - h_internal), c)

    ``output`` first applies the same frozen FinalLayer to both states and then
    extrapolates their velocity outputs.  It is a diagnostic control because
    FinalLayer contains conditional LayerNorm/AdaLN and the two operations are
    therefore not equivalent.

    Interpolation uses the same two spaces with ``alpha`` in ``[0, 1]`` and
    moves from the final state (zero) toward the internal state (one).
    """

    if internal_hidden.shape != final_hidden.shape:
        raise ValueError("internal and final hidden states must have identical shapes")
    if mode not in {"final", "internal", "extrapolation", "interpolation"}:
        raise ValueError(f"unsupported hidden-state mode: {mode!r}")
    if extrapolation_space not in {"hidden", "output"}:
        raise ValueError(
            f"unsupported extrapolation space: {extrapolation_space!r}"
        )
    if not math.isfinite(float(gamma)):
        raise ValueError("gamma must be finite")
    if not math.isfinite(float(alpha)):
        raise ValueError("alpha must be finite")
    if mode != "extrapolation" and float(gamma) != 0.0:
        raise ValueError("gamma is only meaningful in extrapolation mode")
    if mode != "interpolation" and float(alpha) != 0.0:
        raise ValueError("alpha is only meaningful in interpolation mode")
    if mode == "extrapolation" and float(gamma) < 0.0:
        raise ValueError("extrapolation gamma must be nonnegative")
    if mode == "interpolation" and not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("interpolation alpha must lie in [0,1]")

    if mode == "final" or (
        mode in {"extrapolation", "interpolation"}
        and float(gamma) == 0.0
        and float(alpha) == 0.0
    ):
        return velocity_from_hidden_state(
            model,
            final_hidden,
            conditioning,
            latent_channels=latent_channels,
        ).float()

    if mode == "internal":
        return velocity_from_hidden_state(
            model,
            internal_hidden,
            conditioning,
            latent_channels=latent_channels,
        ).float()

    signed_distance = float(gamma) if mode == "extrapolation" else -float(alpha)
    extrapolated_hidden = final_hidden + signed_distance * (
        final_hidden - internal_hidden
    )
    if extrapolation_space == "hidden":
        return velocity_from_hidden_state(
            model,
            extrapolated_hidden,
            conditioning,
            latent_channels=latent_channels,
        ).float()

    final_velocity = velocity_from_hidden_state(
        model,
        final_hidden,
        conditioning,
        latent_channels=latent_channels,
    ).float()
    internal_velocity = velocity_from_hidden_state(
        model,
        internal_hidden,
        conditioning,
        latent_channels=latent_channels,
    ).float()
    return final_velocity + signed_distance * (final_velocity - internal_velocity)


def frozen_hidden_state_field(
    model: nn.Module,
    state: torch.Tensor,
    time_value: torch.Tensor,
    labels: torch.Tensor,
    *,
    internal_depth: int,
    latent_channels: int,
    mode: str,
    gamma: float = 0.0,
    alpha: float = 0.0,
    extrapolation_space: str = "hidden",
) -> torch.Tensor:
    """Evaluate one frozen SiT backbone and select the requested field."""

    internal, final, conditioning = internal_and_final_hidden_states(
        model,
        state,
        time_value,
        labels,
        internal_depth=internal_depth,
    )
    return select_hidden_state_field(
        model,
        internal,
        final,
        conditioning,
        latent_channels=latent_channels,
        mode=mode,
        gamma=gamma,
        alpha=alpha,
        extrapolation_space=extrapolation_space,
    )
