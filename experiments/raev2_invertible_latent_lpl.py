"""Invertible latent reparameterization utilities for frozen RAEv2 models.

The adapter defines a new clean coordinate ``u = A(z)`` while the frozen
RAEv2 Stage-2 model operates numerically in ``u`` space.  Decoder-facing
predictions are mapped back with the exact inverse ``A^{-1}``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from experiments.latent_equiv_adapter import InvertibleLatentAdapter


INVERTIBLE_LATENT_LPL_FORMAT = "raev2_invertible_latent_lpl_v1"


@dataclass(frozen=True)
class ReparameterizedPath:
    clean_latent: torch.Tensor
    transformed_clean: torch.Tensor
    noise: torch.Tensor
    time: torch.Tensor
    noisy_transformed: torch.Tensor
    target_velocity: torch.Tensor


def expand_time(time: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if time.shape != (reference.shape[0],):
        raise ValueError(
            f"time must have shape [{reference.shape[0]}], got {tuple(time.shape)}"
        )
    return time.to(device=reference.device, dtype=reference.dtype).view(
        reference.shape[0], *([1] * (reference.ndim - 1))
    )


def make_reparameterized_path(
    adapter: InvertibleLatentAdapter,
    clean_latent: torch.Tensor,
    noise: torch.Tensor,
    time: torch.Tensor,
    *,
    t_eps: float,
) -> ReparameterizedPath:
    """Build RAEv2's Gaussian-straight path in the learned coordinates."""

    if clean_latent.shape != noise.shape:
        raise ValueError("clean_latent and noise must have identical shapes")
    if clean_latent.ndim != 4:
        raise ValueError("RAEv2 latents must have shape [B, C, H, W]")
    if t_eps <= 0:
        raise ValueError("t_eps must be positive")

    transformed_clean = adapter(clean_latent)
    time_scale = expand_time(time, transformed_clean)
    noisy_transformed = (
        (1.0 - time_scale) * transformed_clean + time_scale * noise
    )
    target_velocity = (
        (noisy_transformed - transformed_clean) / time_scale.clamp_min(float(t_eps))
    )
    return ReparameterizedPath(
        clean_latent=clean_latent,
        transformed_clean=transformed_clean,
        noise=noise,
        time=time,
        noisy_transformed=noisy_transformed,
        target_velocity=target_velocity,
    )


def inverse_prediction(
    adapter: InvertibleLatentAdapter,
    transformed_prediction: torch.Tensor,
) -> torch.Tensor:
    return adapter.inverse(transformed_prediction)


def normalized_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    denominator = target.float().square().mean().clamp_min(float(eps))
    return (prediction.float() - target.float()).square().mean() / denominator


def cycle_metrics(
    adapter: InvertibleLatentAdapter,
    latent: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    transformed = adapter(latent)
    recovered = adapter.inverse(transformed)
    return {
        "cycle_max_abs": (recovered - latent).abs().max(),
        "cycle_relative_mse": normalized_mse(recovered, latent, eps=eps),
        "forward_relative_mse": normalized_mse(transformed, latent, eps=eps),
    }


def adapter_config(adapter: InvertibleLatentAdapter) -> dict[str, int]:
    blocks = list(adapter.blocks)
    if not blocks:
        raise ValueError("invertible adapter must contain at least one block")
    first = blocks[0]
    hidden_channels = int(first.net.net[0].out_channels)
    return {
        "channels": int(first.channels),
        "hidden_channels": hidden_channels,
        "blocks": len(blocks),
    }


def trainable_parameter_boundary(
    adapter: nn.Module,
    frozen_modules: Iterable[nn.Module],
    optimizer: torch.optim.Optimizer,
) -> dict[str, int]:
    adapter_ids = {id(parameter) for parameter in adapter.parameters()}
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if optimizer_ids != adapter_ids:
        raise RuntimeError("optimizer must contain exactly the adapter parameters")

    frozen_parameters = 0
    for module in frozen_modules:
        parameters = tuple(module.parameters())
        if any(parameter.requires_grad for parameter in parameters):
            raise RuntimeError("a frozen module still has trainable parameters")
        frozen_parameters += sum(parameter.numel() for parameter in parameters)
    return {
        "trainable_parameters": sum(
            parameter.numel() for parameter in adapter.parameters()
        ),
        "optimizer_parameter_tensors": len(optimizer_ids),
        "frozen_parameters": frozen_parameters,
    }


def all_reduce_adapter_gradients(
    adapter: nn.Module,
    *,
    world_size: int,
) -> None:
    """Average adapter gradients without wrapping the frozen Stage-2 model in DDP."""

    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if world_size == 1:
        return
    if not torch.distributed.is_initialized():
        raise RuntimeError("distributed process group is not initialized")
    for parameter in adapter.parameters():
        if parameter.grad is None:
            raise RuntimeError("adapter parameter has no gradient")
        torch.distributed.all_reduce(parameter.grad, op=torch.distributed.ReduceOp.SUM)
        parameter.grad.div_(float(world_size))
