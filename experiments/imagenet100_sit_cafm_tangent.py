"""CAFM tangent-critic primitives for the local ImageNet-100 SiT models.

The public CAFM implementation parameterizes the interpolation from data to
noise and flips time inside its SiT discriminator.  This repository uses the
equivalent native SiT convention, noise at ``t=0`` to data at ``t=1``.  The
critic below therefore consumes native time directly and trains on

    x_t = (1 - t) * noise + t * data,   V = data - noise.

Apart from this coordinate change, the architecture and JVP objective match
the public implementation: a pretrained SiT trunk with RMSNorm, a learned
discriminator token, and a scalar output head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn
from torch.func import jvp, vmap


CAFM_REPOSITORY = "https://github.com/ByteDance-Seed/Adversarial-Flow-Models"
CAFM_REVISION = "9b84a478d523ff8f58b4c930ed43fca087282d31"


class NativeSiTTangentCritic(nn.Module):
    """Scalar SiT critic matching CAFM in the repository's native time axis."""

    def __init__(self, pretrained_sit: nn.Module):
        super().__init__()
        hidden_size = int(pretrained_sit.pos_embed.shape[-1])

        self.x_embedder = pretrained_sit.x_embedder
        self.t_embedder = pretrained_sit.t_embedder
        self.y_embedder = pretrained_sit.y_embedder
        self.y_embedder.dropout_prob = 0.0
        self.pos_embed = pretrained_sit.pos_embed
        self.blocks = pretrained_sit.blocks
        for block in self.blocks:
            block.norm1 = nn.RMSNorm(
                hidden_size, elementwise_affine=False, eps=1e-6
            )
            block.norm2 = nn.RMSNorm(
                hidden_size, elementwise_affine=False, eps=1e-6
            )

        self.dis_embed = nn.Parameter(torch.randn(hidden_size) * 0.02)
        self.final_layer = nn.Sequential(
            nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(
        self,
        state: torch.Tensor,
        time_value: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self.x_embedder(state) + self.pos_embed.to(dtype=state.dtype)
        conditioning = self.t_embedder(time_value)
        conditioning = conditioning + self.y_embedder(labels, self.training)
        conditioning = conditioning.to(dtype=tokens.dtype)

        dis_token = self.dis_embed.to(dtype=tokens.dtype)
        dis_token = dis_token.view(1, 1, -1).expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((dis_token, tokens), dim=1)
        for block in self.blocks:
            tokens = block(tokens, conditioning)
        return self.final_layer(tokens[:, :1]).reshape(-1)


class TangentJVP(nn.Module):
    """Evaluate one or several material derivatives of a scalar critic."""

    def __init__(self, critic: nn.Module):
        super().__init__()
        self.critic = critic

    def forward(
        self,
        state: torch.Tensor,
        time_value: torch.Tensor,
        labels: torch.Tensor,
        velocity: torch.Tensor,
        time_velocity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        def scalar_critic(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return self.critic(x, t, labels)

        def evaluate(dx: torch.Tensor, dt: torch.Tensor):
            return jvp(
                scalar_critic,
                (state, time_value),
                (dx, dt),
            )

        if velocity.ndim == state.ndim:
            if time_velocity.shape != time_value.shape:
                raise ValueError("time tangent does not match time values")
            return evaluate(velocity, time_velocity)
        if velocity.ndim != state.ndim + 1:
            raise ValueError("velocity must contain one tangent or a tangent bank")
        if time_velocity.shape != (velocity.shape[0], *time_value.shape):
            raise ValueError("batched time tangents do not match velocity tangents")
        return vmap(evaluate)(velocity, time_velocity)


def critic_from_sit_state(
    *,
    sit_module,
    model_name: str,
    state_dict: Mapping[str, torch.Tensor],
    input_size: int,
    num_classes: int,
    class_dropout_prob: float,
) -> NativeSiTTangentCritic:
    """Load a strict SiT checkpoint before replacing its output with a critic."""

    pretrained = sit_module.SiT_models[model_name](
        input_size=int(input_size),
        num_classes=int(num_classes),
        class_dropout_prob=float(class_dropout_prob),
    )
    pretrained.load_state_dict(state_dict, strict=True)
    return NativeSiTTangentCritic(pretrained)


def per_sample_dot(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape != right.shape or left.ndim < 2:
        raise ValueError("dot-product tensors must have the same batched shape")
    return (left * right).flatten(1).sum(dim=1)


def explicit_spatial_time_jvp(
    critic: nn.Module,
    state: torch.Tensor,
    time_value: torch.Tensor,
    labels: torch.Tensor,
    velocity: torch.Tensor,
    time_velocity: torch.Tensor,
    *,
    create_graph: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reverse-mode reference used for audits with many candidate directions."""

    state = state.requires_grad_(True)
    time_value = time_value.requires_grad_(True)
    value = critic(state, time_value, labels)
    gradient_state, gradient_time = torch.autograd.grad(
        value.sum(),
        (state, time_value),
        create_graph=create_graph,
    )
    directional = per_sample_dot(gradient_state, velocity)
    directional = directional + gradient_time * time_velocity
    return value, directional, gradient_state


@dataclass(frozen=True)
class LocalLSGANOptimum:
    gradient: torch.Tensor
    offset: torch.Tensor
    mahalanobis_residual: torch.Tensor
    discriminator_loss: torch.Tensor


def local_lsgan_optimum(
    mean_velocity: torch.Tensor,
    covariance: torch.Tensor,
    generator_velocity: torch.Tensor,
    *,
    ridge: float = 0.0,
) -> LocalLSGANOptimum:
    """Closed-form local CAFM critic for a positive-definite covariance.

    ``ridge`` makes the inverse explicit in nearly singular controls.  A raw
    Moore-Penrose expression is not generally sufficient when the residual has
    a component in the covariance null space, so the experiment records the
    regularization instead of silently calling that case solved.
    """

    if mean_velocity.ndim != 1 or generator_velocity.shape != mean_velocity.shape:
        raise ValueError("velocities must be vectors with matching shapes")
    dimension = mean_velocity.numel()
    if covariance.shape != (dimension, dimension):
        raise ValueError("covariance has the wrong shape")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    covariance = covariance.to(dtype=mean_velocity.dtype)
    regularized = covariance + float(ridge) * torch.eye(
        dimension, device=covariance.device, dtype=covariance.dtype
    )
    residual = mean_velocity - generator_velocity
    precision_residual = torch.linalg.solve(regularized, residual)
    mahalanobis = residual.dot(precision_residual)
    gradient = 2.0 * precision_residual / (2.0 + mahalanobis)
    offset = -0.5 * gradient.dot(mean_velocity + generator_velocity)
    loss = 4.0 / (2.0 + mahalanobis)
    return LocalLSGANOptimum(gradient, offset, mahalanobis, loss)


def lsgan_tangent_losses(
    critic_value: torch.Tensor,
    real_logit: torch.Tensor,
    fake_logit: torch.Tensor,
    *,
    centering_scale: float,
) -> dict[str, torch.Tensor]:
    if not (
        critic_value.shape == real_logit.shape == fake_logit.shape
        and critic_value.ndim == 1
    ):
        raise ValueError("critic values and tangent logits must be vectors")
    real_loss = (real_logit - 1.0).square().mean()
    fake_loss = (fake_logit + 1.0).square().mean()
    centering = critic_value.square().mean()
    total = real_loss + fake_loss + float(centering_scale) * centering
    return {
        "total": total,
        "real": real_loss,
        "fake": fake_loss,
        "centering": centering,
    }
