"""Dual-output heads and objectives for the ImageNet-100 SiT flow baseline.

The architecture follows Dynamic Dual-Output Diffusion Models (CVPR 2022):
one shared backbone emits epsilon, clean endpoint, and a spatial gate.  The
paper mixes DDPM reverse means; for the repository's linear flow we mix the
equivalent endpoint-derived velocities instead.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


GateActivation = Literal["sigmoid", "identity", "clamp"]
StaticEndpointMode = Literal["raw", "override"]


def retrofit_dual_output_head(
    model: nn.Module,
    *,
    latent_channels: int,
) -> nn.Module:
    """Replace only SiT's last projection with the paper's ``2C+1`` output."""

    if latent_channels < 1:
        raise ValueError("latent_channels must be positive")
    linear = model.final_layer.linear
    if linear.weight.device.type != "cpu":
        raise ValueError("retrofit the dual-output head before moving SiT to CUDA")
    patch_size = int(model.x_embedder.patch_size[0])
    output_channels = 2 * latent_channels + 1
    # Constructing nn.Linear normally advances the global RNG. Preserve it so
    # the training noise/data stream remains aligned with the v-only baseline.
    with torch.random.fork_rng(devices=[]):
        replacement = nn.Linear(
            linear.in_features,
            patch_size * patch_size * output_channels,
            bias=linear.bias is not None,
        )
    nn.init.zeros_(replacement.weight)
    if replacement.bias is not None:
        nn.init.zeros_(replacement.bias)
    model.final_layer.linear = replacement
    model.out_channels = output_channels
    model.learn_sigma = False
    return model


def activate_gate(logits: torch.Tensor, activation: GateActivation) -> torch.Tensor:
    if activation == "sigmoid":
        return logits.sigmoid()
    if activation == "identity":
        return logits
    if activation == "clamp":
        return logits.clamp(0.0, 1.0)
    raise ValueError(f"unsupported gate activation: {activation}")


def split_dual_output(
    output: torch.Tensor,
    *,
    latent_channels: int,
    gate_activation: GateActivation,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    expected_channels = 2 * latent_channels + 1
    if output.ndim != 4 or output.shape[1] != expected_channels:
        raise ValueError(
            f"expected dual output [B,{expected_channels},H,W], found {tuple(output.shape)}"
        )
    epsilon, clean, gate_logits = torch.split(
        output,
        (latent_channels, latent_channels, 1),
        dim=1,
    )
    gate = activate_gate(gate_logits, gate_activation)
    return epsilon, clean, gate_logits, gate


def _time_image(time_value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if time_value.shape != (reference.shape[0],):
        raise ValueError("time_value must have shape [B]")
    return time_value.reshape(-1, *([1] * (reference.ndim - 1)))


def dual_output_flow_losses(
    output: torch.Tensor,
    *,
    clean_target: torch.Tensor,
    epsilon_target: torch.Tensor,
    time_value: torch.Tensor,
    gate_activation: GateActivation = "sigmoid",
    epsilon_weight: float = 1.0,
    clean_weight: float = 1.0,
    gate_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Paper-style three-term loss adapted to a linear interpolant.

    The two native heads use direct MSE.  The gate mixes the two implied
    velocity errors. Multiplying that error by ``t(1-t)`` removes endpoint
    divisions without changing the per-sample optimal gate. Both branch terms
    are detached exactly as in Eq. 19 of the paper.
    """

    if clean_target.shape != epsilon_target.shape:
        raise ValueError("clean and epsilon targets must have identical shapes")
    epsilon, clean, gate_logits, gate = split_dual_output(
        output,
        latent_channels=clean_target.shape[1],
        gate_activation=gate_activation,
    )
    time_image = _time_image(time_value, clean_target)
    epsilon_loss = F.mse_loss(epsilon.float(), epsilon_target.float())
    clean_loss = F.mse_loss(clean.float(), clean_target.float())

    # v_x - v = (x_hat - x)/(1-t)
    # v_eps - v = (eps - eps_hat)/t
    # Multiplication by t(1-t) gives this finite residual.
    clean_error = time_image * (clean.detach().float() - clean_target.float())
    epsilon_error = (1.0 - time_image) * (
        epsilon_target.float() - epsilon.detach().float()
    )
    scaled_velocity_residual = gate.float() * clean_error + (
        1.0 - gate.float()
    ) * epsilon_error
    gate_loss = scaled_velocity_residual.square().mean()
    total = (
        float(epsilon_weight) * epsilon_loss
        + float(clean_weight) * clean_loss
        + float(gate_weight) * gate_loss
    )
    return {
        "total": total,
        "epsilon": epsilon_loss,
        "clean": clean_loss,
        "gate": gate_loss,
        "gate_mean": gate.float().mean(),
        "gate_std": gate.float().std(unbiased=False),
        "gate_logit_mean": gate_logits.float().mean(),
    }


def dual_output_velocities(
    output: torch.Tensor,
    *,
    state: torch.Tensor,
    time_value: torch.Tensor,
    gate_activation: GateActivation = "sigmoid",
    denominator_floor: float = 1e-3,
) -> dict[str, torch.Tensor]:
    """Convert both native predictions to x-/epsilon-/dynamic-flow velocities."""

    if denominator_floor <= 0 or denominator_floor >= 0.5:
        raise ValueError("denominator_floor must be in (0, 0.5)")
    epsilon, clean, _, gate = split_dual_output(
        output,
        latent_channels=state.shape[1],
        gate_activation=gate_activation,
    )
    time_image = _time_image(time_value, state).to(dtype=state.dtype)
    velocity_x = (clean.float() - state.float()) / (1.0 - time_image).clamp_min(
        denominator_floor
    )
    velocity_epsilon = (state.float() - epsilon.float()) / time_image.clamp_min(
        denominator_floor
    )
    velocity_dynamic = gate.float() * velocity_x + (1.0 - gate.float()) * velocity_epsilon

    # At the exact flow endpoints one parameterization is undefined. The DDO
    # gate should select the well-defined branch; enforce that limiting value
    # explicitly so adaptive ODE solvers never evaluate 0/0 numerically.
    near_noise = time_image <= denominator_floor
    near_data = time_image >= 1.0 - denominator_floor
    velocity_dynamic = torch.where(near_noise, velocity_x, velocity_dynamic)
    velocity_dynamic = torch.where(near_data, velocity_epsilon, velocity_dynamic)
    return {
        "x": velocity_x,
        "epsilon": velocity_epsilon,
        "dynamic": velocity_dynamic,
        "gate": gate.float(),
        "clean": clean.float(),
        "epsilon_prediction": epsilon.float(),
    }


def static_dual_velocity(
    velocity_x: torch.Tensor,
    velocity_epsilon: torch.Tensor,
    *,
    time_value: torch.Tensor,
    scale: float,
    denominator_floor: float = 1e-3,
    endpoint_mode: StaticEndpointMode = "raw",
) -> torch.Tensor:
    """Mix the two endpoint-derived fields with one fixed scalar.

    ``scale=0`` selects epsilon and ``scale=1`` selects x.  Values outside
    that interval extrapolate along the same prediction-target direction.
    ``raw`` preserves those endpoint paths exactly.  ``override`` uses the
    well-defined x branch near noise and epsilon branch near data, matching
    the numerical endpoint convention of the learned dynamic path.
    """

    if velocity_x.shape != velocity_epsilon.shape:
        raise ValueError("x and epsilon velocities must have identical shapes")
    if not math.isfinite(scale):
        raise ValueError("static scale must be finite")
    if denominator_floor <= 0 or denominator_floor >= 0.5:
        raise ValueError("denominator_floor must be in (0, 0.5)")
    if endpoint_mode not in ("raw", "override"):
        raise ValueError(f"unsupported static endpoint mode: {endpoint_mode}")

    # Keep the two boundary scales bitwise identical to their native paths.
    if scale == 0.0:
        mixed = velocity_epsilon
    elif scale == 1.0:
        mixed = velocity_x
    else:
        mixed = velocity_epsilon + float(scale) * (velocity_x - velocity_epsilon)

    if endpoint_mode == "raw":
        return mixed

    time_image = _time_image(time_value, velocity_x).to(dtype=velocity_x.dtype)
    mixed = torch.where(time_image <= denominator_floor, velocity_x, mixed)
    mixed = torch.where(
        time_image >= 1.0 - denominator_floor,
        velocity_epsilon,
        mixed,
    )
    return mixed
