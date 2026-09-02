"""Amortized soft-Bellman value for semigroup-consistent guidance."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

try:
    from experiments.imagenet100_sit_internal_v_head import (
        embed_sit_inputs,
        internal_velocity_from_features,
        unpatchify_channels,
        validate_internal_depth,
    )
except ModuleNotFoundError:
    from imagenet100_sit_internal_v_head import (
        embed_sit_inputs,
        internal_velocity_from_features,
        unpatchify_channels,
        validate_internal_depth,
    )


def heat_variance_from_flow_time(time_value: torch.Tensor) -> torch.Tensor:
    """Map ``z_t=t*x+(1-t)*eps`` to VE heat variance."""

    if torch.any((time_value <= 0.0) | (time_value > 1.0)):
        raise ValueError("flow time must lie in (0, 1]")
    return ((1.0 - time_value) / time_value).square()


def flow_time_from_heat_variance(heat_variance: torch.Tensor) -> torch.Tensor:
    if torch.any(heat_variance < 0.0):
        raise ValueError("heat variance must be non-negative")
    return 1.0 / (1.0 + heat_variance.sqrt())


def _batch_scale(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if value.ndim != 1 or value.shape[0] != target.shape[0]:
        raise ValueError("coefficient must have one value per batch element")
    while value.ndim < target.ndim:
        value = value.unsqueeze(-1)
    return value


def flow_velocity_to_heat_score(
    velocity: torch.Tensor,
    *,
    state: torch.Tensor,
    time_value: torch.Tensor,
) -> torch.Tensor:
    """Convert a linear-flow velocity to the score of ``y=x+sigma*eps``."""

    if velocity.shape != state.shape:
        raise ValueError("velocity and state must have identical shapes")
    factor = _batch_scale(time_value.square() / (1.0 - time_value), state)
    inverse_time = _batch_scale(time_value.reciprocal(), state)
    return factor * (velocity - inverse_time * state)


def velocity_gap_to_heat_score_gap(
    velocity_gap: torch.Tensor,
    *,
    time_value: torch.Tensor,
) -> torch.Tensor:
    factor = _batch_scale(
        time_value.square() / (1.0 - time_value), velocity_gap
    )
    return factor * velocity_gap


def potential_gradient_to_velocity_correction(
    potential_gradient: torch.Tensor,
    *,
    time_value: torch.Tensor,
) -> torch.Tensor:
    """Convert ``grad_z delta`` to the corresponding linear-flow velocity."""

    factor = _batch_scale((1.0 - time_value) / time_value, potential_gradient)
    return factor * potential_gradient


def bellman_log_value_target(
    next_values: torch.Tensor,
    *,
    running_cost: torch.Tensor,
    heat_step: torch.Tensor,
) -> torch.Tensor:
    """Monte Carlo soft-Bellman target in stable log space.

    ``next_values`` has shape ``[K,B]``.  The remaining arguments have shape
    ``[B]`` and represent the Feynman--Kac running cost at the current state.
    """

    if next_values.ndim != 2:
        raise ValueError("next_values must have shape [particles, batch]")
    if running_cost.shape != next_values.shape[1:] or heat_step.shape != running_cost.shape:
        raise ValueError("running cost and heat step must have shape [batch]")
    if torch.any(heat_step <= 0.0):
        raise ValueError("heat steps must be positive")
    values = next_values + (running_cost * heat_step).unsqueeze(0)
    return torch.logsumexp(values, dim=0) - math.log(next_values.shape[0])


def potential_envelope(
    time_value: torch.Tensor,
    *,
    intervention_time: float,
) -> torch.Tensor:
    """Gauge fixing: zero spatial potential at pure noise and intervention."""

    if not 0.0 < intervention_time < 1.0:
        raise ValueError("intervention time must lie in (0,1)")
    normalized = time_value / float(intervention_time)
    return 4.0 * normalized * (1.0 - normalized)


def boundary_envelope(
    time_value: torch.Tensor,
    *,
    intervention_time: float,
) -> torch.Tensor:
    """Enforce only the Bellman boundary at the intervention plane."""

    if not 0.0 < intervention_time < 1.0:
        raise ValueError("intervention time must lie in (0,1)")
    return 1.0 - time_value / float(intervention_time)


class TokenPotentialHead(nn.Module):
    """Small scalar value head on frozen final SiT tokens."""

    def __init__(self, hidden_size: int, *, intervention_time: float = 0.5) -> None:
        super().__init__()
        if hidden_size < 4:
            raise ValueError("hidden size is too small")
        self.intervention_time = float(intervention_time)
        inner_size = max(64, hidden_size // 2)
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_size, inner_size),
            nn.SiLU(),
            nn.Linear(inner_size, 1),
        )
        self.baseline = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, inner_size),
            nn.SiLU(),
            nn.Linear(inner_size, 1),
        )
        nn.init.zeros_(self.value[-1].weight)
        nn.init.zeros_(self.value[-1].bias)
        nn.init.zeros_(self.baseline[-1].weight)
        nn.init.zeros_(self.baseline[-1].bias)

    def forward(
        self,
        tokens: torch.Tensor,
        conditioning: torch.Tensor,
        time_value: torch.Tensor,
    ) -> torch.Tensor:
        if tokens.ndim != 3 or conditioning.shape != (len(tokens), tokens.shape[-1]):
            raise ValueError("invalid token or conditioning shape")
        if time_value.shape != (len(tokens),):
            raise ValueError("time must have shape [batch]")
        shift, scale = self.modulation(conditioning).chunk(2, dim=1)
        normalized = self.norm(tokens)
        normalized = normalized * (1.0 + scale[:, None]) + shift[:, None]
        spatial_value = self.value(normalized).mean(dim=1).squeeze(-1)
        baseline_value = self.baseline(conditioning).squeeze(-1)
        return (
            baseline_value
            * boundary_envelope(
                time_value,
                intervention_time=self.intervention_time,
            )
            + spatial_value
            * potential_envelope(
                time_value,
                intervention_time=self.intervention_time,
            )
        )


def source_weak_and_final_features(
    source: nn.Module,
    weak_head: nn.Module,
    state: torch.Tensor,
    time_value: torch.Tensor,
    labels: torch.Tensor,
    *,
    internal_depth: int,
    latent_channels: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate strong/weak velocities and expose final hidden tokens."""

    depth = validate_internal_depth(source, internal_depth)
    tokens, conditioning = embed_sit_inputs(source, state, time_value, labels)
    internal_features = None
    for index, block in enumerate(source.blocks, start=1):
        tokens = block(tokens, conditioning)
        if index == depth:
            internal_features = tokens
    if internal_features is None:
        raise AssertionError("internal feature extraction failed")
    weak = internal_velocity_from_features(
        source,
        weak_head,
        internal_features,
        conditioning,
        latent_channels=latent_channels,
    )
    projected = source.final_layer(tokens, conditioning)
    strong = source.unpatchify(projected)[:, :latent_channels]
    return strong, weak, tokens, conditioning


def final_features(
    source: nn.Module,
    state: torch.Tensor,
    time_value: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens, conditioning = embed_sit_inputs(source, state, time_value, labels)
    for block in source.blocks:
        tokens = block(tokens, conditioning)
    return tokens, conditioning
