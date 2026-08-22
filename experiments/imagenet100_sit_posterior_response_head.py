"""Amortized diagonal posterior-response head for a frozen SiT.

The head predicts a state-dependent diagonal operator ``diag(d)`` and applies
it to an arbitrary direction.  With isotropic random probe directions, MSE to
the matrix-free teacher action has the diagonal of the teacher Jacobian as its
population optimum.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from experiments.imagenet100_sit_internal_v_head import (
    create_internal_velocity_head,
    internal_velocity_from_features,
)
from experiments.posterior_response_projector import relative_direction_step


def create_diagonal_response_head(
    sit_module,
    model: nn.Module,
    *,
    latent_channels: int,
) -> nn.Module:
    """Create a zero-logit SiT FinalLayer, corresponding to initial gain 0.5."""
    return create_internal_velocity_head(
        sit_module,
        model,
        latent_channels=latent_channels,
    )


def diagonal_response_gain(
    model: nn.Module,
    head: nn.Module,
    features: torch.Tensor,
    conditioning: torch.Tensor,
    *,
    latent_channels: int,
) -> torch.Tensor:
    logits = internal_velocity_from_features(
        model,
        head,
        features,
        conditioning,
        latent_channels=latent_channels,
    )
    return torch.sigmoid(logits.float())


def diagonal_response_action(
    gain: torch.Tensor,
    direction: torch.Tensor,
) -> torch.Tensor:
    if gain.shape != direction.shape:
        raise ValueError("gain and direction must have identical shapes")
    return gain.float() * direction.float()


@torch.no_grad()
def finite_difference_clean_response_action(
    clean_model: nn.Module,
    *,
    state: torch.Tensor,
    time_value: torch.Tensor,
    labels: torch.Tensor,
    direction: torch.Tensor,
    alpha: torch.Tensor,
    relative_step: float,
) -> torch.Tensor:
    """Compute ``alpha J_clean direction`` with one paired model forward."""
    if state.shape != direction.shape:
        raise ValueError("state and direction must have identical shapes")
    batch = len(state)
    if time_value.shape != (batch,) or labels.shape != (batch,) or alpha.shape != (batch,):
        raise ValueError("time, labels, and alpha must all have shape [B]")
    step = relative_direction_step(
        state,
        direction,
        relative_step=relative_step,
    )
    broadcast_shape = (batch,) + (1,) * (state.ndim - 1)
    scaled_step = step.reshape(broadcast_shape).to(direction.dtype)
    delta = scaled_step * direction
    paired_state = torch.cat((state + delta, state - delta), dim=0)
    paired_time = torch.cat((time_value, time_value), dim=0)
    paired_labels = torch.cat((labels, labels), dim=0)
    paired_clean = clean_model(paired_state, paired_time, paired_labels).float()
    plus, minus = paired_clean[:batch], paired_clean[batch:]
    derivative = (plus - minus) / (2.0 * scaled_step.float())
    scaled_alpha = alpha.reshape(broadcast_shape).float()
    return scaled_alpha * derivative


def rademacher_probe_like(value: torch.Tensor) -> torch.Tensor:
    """Sample an isotropic unit-RMS probe with the same shape and device."""
    return torch.empty_like(value).bernoulli_(0.5).mul_(2.0).sub_(1.0)
