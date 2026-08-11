#!/usr/bin/env python3
"""Diagnose dual prediction targets under closed-loop flow sampling.

The experiment follows the SiT time convention exactly:

    x_t = (1 - t) * epsilon + t * x,  t: 0 -> 1
    u   = x - epsilon

The clean distribution is a Gaussian mixture in a known 2-D spiral coordinate
system, linearly embedded in R^D. This retains the high-dimensional ambient
effect while making the Bayes conditional vector field available in closed
form at arbitrary teacher-forced and rollout states.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-12
MODEL_IDS = (
    "B0_v",
    "B1_x",
    "B2_eps",
    "D0_xeps",
    "D1_scaled",
    "D2_velocity",
    "D4_safe",
    "S0_xv",
    "S1_xv",
)


def parse_int_list(text: str) -> list[int]:
    values = [int(value.strip()) for value in str(text).split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return values


def parse_float_list(text: str) -> list[float]:
    values = [float(value.strip()) for value in str(text).split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated floats")
    return values


def parse_str_list(text: str) -> list[str]:
    values = [value.strip() for value in str(text).split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated strings")
    return values


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_seed(*values: int) -> int:
    result = 2166136261
    for value in values:
        result ^= int(value) & 0xFFFFFFFF
        result = (result * 16777619) & 0xFFFFFFFF
    return int(result)


def save_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                columns.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def row_mse(value: torch.Tensor) -> torch.Tensor:
    return value.float().square().mean(dim=1)


def row_rms(value: torch.Tensor) -> torch.Tensor:
    return row_mse(value).sqrt()


class SpiralGaussianMixture:
    """Equal-weight Gaussian spiral mixture in a known linear 2-D subspace."""

    def __init__(
        self,
        ambient_dim: int,
        *,
        components: int,
        component_std: float,
        seed: int,
        device: torch.device,
    ) -> None:
        if ambient_dim < 2:
            raise ValueError("ambient_dim must be at least 2")
        if components < 2:
            raise ValueError("components must be at least 2")
        if component_std <= 0:
            raise ValueError("component_std must be positive")
        self.ambient_dim = int(ambient_dim)
        self.components = int(components)
        self.component_std = float(component_std)
        self.device = device

        position = (torch.arange(components, dtype=torch.float64) + 0.5) / components
        angle = 4.0 * math.pi * position
        radius = 0.18 + 0.82 * position
        centers = 1.6 * torch.stack(
            (radius * torch.cos(angle), radius * torch.sin(angle)), dim=1
        )
        expected_intrinsic_energy = (
            centers.square().sum(dim=1).mean() + 2.0 * component_std**2
        )
        self.scale = math.sqrt(ambient_dim / float(expected_intrinsic_energy))
        self.centers = centers.to(device=device, dtype=torch.float32)

        generator = torch.Generator(device="cpu").manual_seed(seed)
        matrix = torch.randn(ambient_dim, 2, generator=generator, dtype=torch.float64)
        basis, _ = torch.linalg.qr(matrix, mode="reduced")
        self.basis = basis.to(device=device, dtype=torch.float32)

    def sample_intrinsic(
        self, n: int, *, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        component = torch.randint(
            self.components, (n,), device=self.device, generator=generator
        )
        noise = torch.randn(n, 2, device=self.device, generator=generator)
        intrinsic = self.centers[component] + self.component_std * noise
        return intrinsic, component

    def embed(self, intrinsic: torch.Tensor) -> torch.Tensor:
        return self.scale * (intrinsic @ self.basis.T)

    def sample(
        self, n: int, *, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        intrinsic, component = self.sample_intrinsic(n, generator=generator)
        return self.embed(intrinsic), intrinsic, component

    def decode_intrinsic(self, value: torch.Tensor) -> torch.Tensor:
        return (value @ self.basis) / self.scale

    def off_subspace_rms(self, value: torch.Tensor) -> torch.Tensor:
        projection = (value @ self.basis) @ self.basis.T
        return row_rms(value - projection)

    def bayes_clean(self, state: torch.Tensor, time_value: torch.Tensor) -> torch.Tensor:
        """Return E[x | x_t=state] for the linear Gaussian mixture bridge."""
        if time_value.shape != (len(state),):
            raise ValueError("time_value must have shape [B]")
        projected = state @ self.basis
        time_column = time_value.float()[:, None]
        observation_gain = time_column * self.scale
        noise_variance = (1.0 - time_column).square()
        total_variance = (
            observation_gain.square() * self.component_std**2 + noise_variance
        ).clamp_min(1e-12)

        residual = projected[:, None, :] - observation_gain[:, None, :] * self.centers
        logits = -0.5 * residual.square().sum(dim=2) / total_variance
        weights = torch.softmax(logits, dim=1)

        kalman_gain = self.component_std**2 * observation_gain / total_variance
        posterior_means = self.centers[None, :, :] + kalman_gain[:, None, :] * residual
        intrinsic_mean = (weights[:, :, None] * posterior_means).sum(dim=1)
        return self.embed(intrinsic_mean)

    def bayes_velocity(
        self,
        state: torch.Tensor,
        time_value: torch.Tensor,
        *,
        denominator_floor: float,
    ) -> torch.Tensor:
        effective_time = time_value.float().clamp(max=1.0 - denominator_floor)
        clean = self.bayes_clean(state, effective_time)
        denominator = (1.0 - effective_time)[:, None]
        return (clean - state) / denominator

    def intrinsic_nll(self, intrinsic: torch.Tensor) -> torch.Tensor:
        residual = intrinsic[:, None, :] - self.centers[None, :, :]
        variance = self.component_std**2
        component_log_prob = (
            -0.5 * residual.square().sum(dim=2) / variance
            - math.log(2.0 * math.pi * variance)
            - math.log(self.components)
        )
        return -torch.logsumexp(component_log_prob, dim=1)

    def component_histogram(self, intrinsic: torch.Tensor) -> torch.Tensor:
        distance = torch.cdist(intrinsic.float(), self.centers.float()).square()
        assignment = distance.argmin(dim=1)
        return torch.bincount(assignment, minlength=self.components).float() / len(intrinsic)


class TimeEmbedding(nn.Module):
    def __init__(self, dimension: int, max_frequency: float = 32.0) -> None:
        super().__init__()
        if dimension % 2:
            raise ValueError("time embedding dimension must be even")
        frequencies = torch.exp(
            torch.linspace(0.0, math.log(max_frequency), dimension // 2)
        )
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, time_value: torch.Tensor) -> torch.Tensor:
        phase = 2.0 * math.pi * time_value[:, None] * self.frequencies[None]
        return torch.cat((phase.sin(), phase.cos()), dim=1)


class FeatureTrunk(nn.Module):
    def __init__(
        self,
        ambient_dim: int,
        hidden_dim: int,
        depth: int,
        time_dim: int,
        extra_dim: int = 0,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be at least 2")
        self.time = TimeEmbedding(time_dim)
        layers: list[nn.Module] = []
        input_dim = ambient_dim + time_dim + extra_dim
        for _ in range(depth):
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.SiLU()))
            input_dim = hidden_dim
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        state: torch.Tensor,
        time_value: torch.Tensor,
        extra: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pieces = [state, self.time(time_value)]
        if extra is not None:
            pieces.append(extra)
        return self.net(torch.cat(pieces, dim=1))


class MultiHeadPredictor(nn.Module):
    def __init__(
        self,
        ambient_dim: int,
        hidden_dim: int,
        depth: int,
        time_dim: int,
        heads: Sequence[str],
    ) -> None:
        super().__init__()
        self.trunk = FeatureTrunk(ambient_dim, hidden_dim, depth, time_dim)
        self.heads = nn.ModuleDict()
        for name in heads:
            output_dim = 1 if name == "gate" else ambient_dim
            head = nn.Linear(hidden_dim, output_dim)
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            self.heads[name] = head

    def forward(self, state: torch.Tensor, time_value: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.trunk(state, time_value)
        return {name: head(features) for name, head in self.heads.items()}


class ModeConditionedPredictor(nn.Module):
    """One shared network called with binary velocity/clean modes."""

    def __init__(
        self,
        ambient_dim: int,
        hidden_dim: int,
        depth: int,
        time_dim: int,
        mode_dim: int,
    ) -> None:
        super().__init__()
        self.mode_embedding = nn.Embedding(2, mode_dim)
        self.trunk = FeatureTrunk(
            ambient_dim, hidden_dim, depth, time_dim, extra_dim=mode_dim
        )
        self.head = nn.Linear(hidden_dim, ambient_dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self, state: torch.Tensor, time_value: torch.Tensor, mode: int
    ) -> torch.Tensor:
        mode_index = torch.full(
            (len(state),), int(mode), device=state.device, dtype=torch.long
        )
        extra = self.mode_embedding(mode_index)
        return self.head(self.trunk(state, time_value, extra))


@dataclass
class ModelSuite:
    models: dict[str, nn.Module]
    optimizers: dict[str, torch.optim.Optimizer]


def _copy_common_trunk_state(
    models: dict[str, nn.Module], model_ids: Sequence[str]
) -> None:
    multi_ids = [model_id for model_id in model_ids if model_id not in {"S0_xv", "S1_xv"}]
    if multi_ids:
        state = copy.deepcopy(models[multi_ids[0]].trunk.state_dict())
        for model_id in multi_ids[1:]:
            models[model_id].trunk.load_state_dict(state)
    mode_ids = [model_id for model_id in model_ids if model_id in {"S0_xv", "S1_xv"}]
    if mode_ids:
        state = copy.deepcopy(models[mode_ids[0]].state_dict())
        for model_id in mode_ids[1:]:
            models[model_id].load_state_dict(state)


def build_model_suite(
    *,
    ambient_dim: int,
    hidden_dim: int,
    depth: int,
    time_dim: int,
    mode_dim: int,
    model_ids: Sequence[str],
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> ModelSuite:
    unknown = sorted(set(model_ids) - set(MODEL_IDS))
    if unknown:
        raise ValueError(f"unknown model ids: {unknown}")
    torch.manual_seed(seed)
    models: dict[str, nn.Module] = {}
    head_specs = {
        "B0_v": ("v",),
        "B1_x": ("x",),
        "B2_eps": ("eps",),
        "D0_xeps": ("x", "eps"),
        "D1_scaled": ("x", "eps", "gate"),
        "D2_velocity": ("x", "eps", "gate"),
        "D4_safe": ("x", "eps", "gate"),
    }
    for model_id in model_ids:
        if model_id in head_specs:
            models[model_id] = MultiHeadPredictor(
                ambient_dim, hidden_dim, depth, time_dim, head_specs[model_id]
            ).to(device)
        else:
            models[model_id] = ModeConditionedPredictor(
                ambient_dim, hidden_dim, depth, time_dim, mode_dim
            ).to(device)
    _copy_common_trunk_state(models, model_ids)
    optimizers = {
        model_id: torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        for model_id, model in models.items()
    }
    return ModelSuite(models=models, optimizers=optimizers)


def endpoint_velocities(
    *,
    state: torch.Tensor,
    time_value: torch.Tensor,
    clean_prediction: torch.Tensor,
    epsilon_prediction: torch.Tensor,
    denominator_floor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    time_column = time_value[:, None]
    velocity_x = (clean_prediction - state) / (1.0 - time_column).clamp_min(
        denominator_floor
    )
    velocity_epsilon = (state - epsilon_prediction) / time_column.clamp_min(
        denominator_floor
    )
    return velocity_x, velocity_epsilon


def gate_value(
    gate_logit: torch.Tensor,
    time_value: torch.Tensor,
    *,
    asymptotic_safe: bool,
    denominator_floor: float,
) -> torch.Tensor:
    if not asymptotic_safe:
        return gate_logit.sigmoid()
    effective_time = time_value.clamp(denominator_floor, 1.0 - denominator_floor)
    prior_logit = torch.log1p(-effective_time) - torch.log(effective_time)
    return (gate_logit + prior_logit[:, None]).sigmoid()


def analytic_scalar_gate(
    velocity_x: torch.Tensor,
    velocity_epsilon: torch.Tensor,
    target_velocity: torch.Tensor,
    *,
    clip: bool,
) -> torch.Tensor:
    direction = velocity_x - velocity_epsilon
    numerator = ((target_velocity - velocity_epsilon) * direction).sum(dim=1, keepdim=True)
    denominator = direction.square().sum(dim=1, keepdim=True)
    gate = torch.where(
        denominator > EPS,
        numerator / denominator.clamp_min(EPS),
        torch.full_like(denominator, 0.5),
    )
    return gate.clamp(0.0, 1.0) if clip else gate


def scaled_gate_residual(
    *,
    gate: torch.Tensor,
    clean_prediction: torch.Tensor,
    epsilon_prediction: torch.Tensor,
    clean_target: torch.Tensor,
    epsilon_target: torch.Tensor,
    time_value: torch.Tensor,
) -> torch.Tensor:
    time_column = time_value[:, None]
    clean_error = clean_prediction - clean_target
    epsilon_error = epsilon_target - epsilon_prediction
    return (
        gate * time_column * clean_error
        + (1.0 - gate) * (1.0 - time_column) * epsilon_error
    )


def velocity_gate_residual(
    *,
    gate: torch.Tensor,
    clean_prediction: torch.Tensor,
    epsilon_prediction: torch.Tensor,
    clean_target: torch.Tensor,
    epsilon_target: torch.Tensor,
    time_value: torch.Tensor,
    denominator_floor: float,
) -> torch.Tensor:
    time_column = time_value[:, None]
    clean_error = (clean_prediction - clean_target) / (1.0 - time_column).clamp_min(
        denominator_floor
    )
    epsilon_error = (epsilon_target - epsilon_prediction) / time_column.clamp_min(
        denominator_floor
    )
    return gate * clean_error + (1.0 - gate) * epsilon_error


def model_loss(
    model_id: str,
    model: nn.Module,
    *,
    state: torch.Tensor,
    time_value: torch.Tensor,
    clean: torch.Tensor,
    epsilon: torch.Tensor,
    velocity: torch.Tensor,
    denominator_floor: float,
    consistency_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if model_id in {"S0_xv", "S1_xv"}:
        velocity_prediction = model(state, time_value, 0)
        clean_prediction = model(state, time_value, 1)
        velocity_loss = F.mse_loss(velocity_prediction, velocity)
        clean_loss = F.mse_loss(clean_prediction, clean)
        implied_clean = state + (1.0 - time_value[:, None]) * velocity_prediction
        consistency = F.mse_loss(clean_prediction, implied_clean)
        weight = consistency_weight if model_id == "S1_xv" else 0.0
        total = velocity_loss + clean_loss + weight * consistency
        return total, {
            "velocity": velocity_loss.detach(),
            "clean": clean_loss.detach(),
            "consistency": consistency.detach(),
        }

    outputs = model(state, time_value)
    if model_id == "B0_v":
        loss = F.mse_loss(outputs["v"], velocity)
        return loss, {"velocity": loss.detach()}
    if model_id == "B1_x":
        loss = F.mse_loss(outputs["x"], clean)
        return loss, {"clean": loss.detach()}
    if model_id == "B2_eps":
        loss = F.mse_loss(outputs["eps"], epsilon)
        return loss, {"epsilon": loss.detach()}

    clean_loss = F.mse_loss(outputs["x"], clean)
    epsilon_loss = F.mse_loss(outputs["eps"], epsilon)
    total = clean_loss + epsilon_loss
    details = {
        "clean": clean_loss.detach(),
        "epsilon": epsilon_loss.detach(),
    }
    if model_id == "D0_xeps":
        return total, details

    safe = model_id == "D4_safe"
    gate = gate_value(
        outputs["gate"],
        time_value,
        asymptotic_safe=safe,
        denominator_floor=denominator_floor,
    )
    if model_id == "D1_scaled":
        residual = scaled_gate_residual(
            gate=gate,
            clean_prediction=outputs["x"].detach(),
            epsilon_prediction=outputs["eps"].detach(),
            clean_target=clean,
            epsilon_target=epsilon,
            time_value=time_value,
        )
    else:
        residual = velocity_gate_residual(
            gate=gate,
            clean_prediction=outputs["x"].detach(),
            epsilon_prediction=outputs["eps"].detach(),
            clean_target=clean,
            epsilon_target=epsilon,
            time_value=time_value,
            denominator_floor=denominator_floor,
        )
    gate_loss = residual.square().mean()
    details.update({"gate": gate_loss.detach(), "gate_mean": gate.mean().detach()})
    return total + gate_loss, details


def train_models(
    *,
    suite: ModelSuite,
    distribution: SpiralGaussianMixture,
    steps: int,
    batch_size: int,
    t_min: float,
    t_max: float,
    denominator_floor: float,
    consistency_weight: float,
    grad_clip: float,
    log_every: int,
    seed: int,
    checkpoint_path: Path,
) -> list[dict]:
    generator = torch.Generator(device=distribution.device.type)
    generator.manual_seed(seed)
    history: list[dict] = []
    started = time.monotonic()
    for step in range(1, steps + 1):
        clean, _, _ = distribution.sample(batch_size, generator=generator)
        epsilon = torch.randn(clean.shape, device=clean.device, generator=generator)
        time_value = torch.empty(batch_size, device=clean.device).uniform_(
            t_min, t_max, generator=generator
        )
        state = (1.0 - time_value[:, None]) * epsilon + time_value[:, None] * clean
        velocity = clean - epsilon

        step_details: dict[str, dict[str, torch.Tensor]] = {}
        for model_id, model in suite.models.items():
            optimizer = suite.optimizers[model_id]
            optimizer.zero_grad(set_to_none=True)
            loss, details = model_loss(
                model_id,
                model,
                state=state,
                time_value=time_value,
                clean=clean,
                epsilon=epsilon,
                velocity=velocity,
                denominator_floor=denominator_floor,
                consistency_weight=consistency_weight,
            )
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            details["total"] = loss.detach()
            step_details[model_id] = details

        if step == 1 or step % log_every == 0 or step == steps:
            elapsed = time.monotonic() - started
            for model_id, details in step_details.items():
                numeric_details = {
                    key: float(value) for key, value in details.items()
                }
                history.append(
                    {
                        "step": step,
                        "model": model_id,
                        "elapsed_seconds": elapsed,
                        **numeric_details,
                    }
                )
            short = " ".join(
                f"{model_id}={float(details['total']):.4f}"
                for model_id, details in step_details.items()
            )
            print(f"step={step}/{steps} elapsed={elapsed:.1f}s {short}", flush=True)

    atomic_torch_save(
        {
            "steps": steps,
            "models": {key: model.state_dict() for key, model in suite.models.items()},
            "optimizers": {
                key: optimizer.state_dict() for key, optimizer in suite.optimizers.items()
            },
        },
        checkpoint_path,
    )
    return history


def _endpoint_override(
    velocity: torch.Tensor,
    *,
    velocity_x: torch.Tensor,
    velocity_epsilon: torch.Tensor,
    time_value: torch.Tensor,
    denominator_floor: float,
) -> torch.Tensor:
    near_noise = time_value[:, None] <= denominator_floor
    near_data = time_value[:, None] >= 1.0 - denominator_floor
    velocity = torch.where(near_noise, velocity_x, velocity)
    return torch.where(near_data, velocity_epsilon, velocity)


@torch.no_grad()
def condition_field(
    condition: str,
    *,
    suite: ModelSuite,
    distribution: SpiralGaussianMixture,
    state: torch.Tensor,
    time_value: torch.Tensor,
    denominator_floor: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if condition == "Bayes_exact":
        return (
            distribution.bayes_velocity(
                state, time_value, denominator_floor=denominator_floor
            ),
            None,
        )
    if condition == "B0_v_ind":
        return suite.models["B0_v"](state, time_value)["v"], None
    if condition == "B1_x_ind":
        output = suite.models["B1_x"](state, time_value)
        zero = torch.zeros_like(output["x"])
        velocity_x, _ = endpoint_velocities(
            state=state,
            time_value=time_value,
            clean_prediction=output["x"],
            epsilon_prediction=zero,
            denominator_floor=denominator_floor,
        )
        return velocity_x, None
    if condition == "B2_eps_ind":
        output = suite.models["B2_eps"](state, time_value)
        zero = torch.zeros_like(output["eps"])
        _, velocity_epsilon = endpoint_velocities(
            state=state,
            time_value=time_value,
            clean_prediction=zero,
            epsilon_prediction=output["eps"],
            denominator_floor=denominator_floor,
        )
        return velocity_epsilon, None

    if condition.startswith("D0_") or condition == "D3_oracle_bayes_gate":
        output = suite.models["D0_xeps"](state, time_value)
        velocity_x, velocity_epsilon = endpoint_velocities(
            state=state,
            time_value=time_value,
            clean_prediction=output["x"],
            epsilon_prediction=output["eps"],
            denominator_floor=denominator_floor,
        )
        if condition == "D0_x_shared":
            return velocity_x, None
        if condition == "D0_eps_shared":
            return velocity_epsilon, None
        if condition == "D0_fixed_x_eps":
            gate = (time_value <= 0.5).float()[:, None]
        elif condition == "D0_safe_schedule":
            gate = 1.0 - time_value[:, None]
        elif condition == "D3_oracle_bayes_gate":
            target = distribution.bayes_velocity(
                state, time_value, denominator_floor=denominator_floor
            )
            gate = analytic_scalar_gate(
                velocity_x, velocity_epsilon, target, clip=True
            )
        else:
            raise ValueError(condition)
        velocity = gate * velocity_x + (1.0 - gate) * velocity_epsilon
        return (
            _endpoint_override(
                velocity,
                velocity_x=velocity_x,
                velocity_epsilon=velocity_epsilon,
                time_value=time_value,
                denominator_floor=denominator_floor,
            ),
            gate,
        )

    gate_models = {
        "D1_scaled_gate": ("D1_scaled", False),
        "D2_velocity_gate": ("D2_velocity", False),
        "D4_safe_velocity_gate": ("D4_safe", True),
    }
    if condition in gate_models:
        model_id, safe = gate_models[condition]
        output = suite.models[model_id](state, time_value)
        velocity_x, velocity_epsilon = endpoint_velocities(
            state=state,
            time_value=time_value,
            clean_prediction=output["x"],
            epsilon_prediction=output["eps"],
            denominator_floor=denominator_floor,
        )
        gate = gate_value(
            output["gate"],
            time_value,
            asymptotic_safe=safe,
            denominator_floor=denominator_floor,
        )
        velocity = gate * velocity_x + (1.0 - gate) * velocity_epsilon
        return (
            _endpoint_override(
                velocity,
                velocity_x=velocity_x,
                velocity_epsilon=velocity_epsilon,
                time_value=time_value,
                denominator_floor=denominator_floor,
            ),
            gate,
        )

    own_branch_conditions = {
        "D1_x_own": ("D1_scaled", "x"),
        "D1_eps_own": ("D1_scaled", "epsilon"),
        "D2_x_own": ("D2_velocity", "x"),
        "D2_eps_own": ("D2_velocity", "epsilon"),
        "D4_x_own": ("D4_safe", "x"),
        "D4_eps_own": ("D4_safe", "epsilon"),
    }
    if condition in own_branch_conditions:
        model_id, branch = own_branch_conditions[condition]
        output = suite.models[model_id](state, time_value)
        velocity_x, velocity_epsilon = endpoint_velocities(
            state=state,
            time_value=time_value,
            clean_prediction=output["x"],
            epsilon_prediction=output["eps"],
            denominator_floor=denominator_floor,
        )
        velocity = velocity_x if branch == "x" else velocity_epsilon
        return _endpoint_override(
            velocity,
            velocity_x=velocity_x,
            velocity_epsilon=velocity_epsilon,
            time_value=time_value,
            denominator_floor=denominator_floor,
        ), None

    gate_on_d0_conditions = {
        "D1_gate_on_D0": ("D1_scaled", False),
        "D2_gate_on_D0": ("D2_velocity", False),
        "D4_gate_on_D0": ("D4_safe", True),
    }
    if condition in gate_on_d0_conditions:
        gate_model_id, safe = gate_on_d0_conditions[condition]
        base_output = suite.models["D0_xeps"](state, time_value)
        gate_output = suite.models[gate_model_id](state, time_value)
        velocity_x, velocity_epsilon = endpoint_velocities(
            state=state,
            time_value=time_value,
            clean_prediction=base_output["x"],
            epsilon_prediction=base_output["eps"],
            denominator_floor=denominator_floor,
        )
        gate = gate_value(
            gate_output["gate"],
            time_value,
            asymptotic_safe=safe,
            denominator_floor=denominator_floor,
        )
        velocity = gate * velocity_x + (1.0 - gate) * velocity_epsilon
        return (
            _endpoint_override(
                velocity,
                velocity_x=velocity_x,
                velocity_epsilon=velocity_epsilon,
                time_value=time_value,
                denominator_floor=denominator_floor,
            ),
            gate,
        )

    sc_conditions = {
        "S0_xv_switch": "S0_xv",
        "S1_xv_consistency_switch": "S1_xv",
    }
    if condition in sc_conditions:
        model = suite.models[sc_conditions[condition]]
        velocity_native = model(state, time_value, 0)
        clean_prediction = model(state, time_value, 1)
        velocity_x = (clean_prediction - state) / (1.0 - time_value[:, None]).clamp_min(
            denominator_floor
        )
        use_x = time_value[:, None] <= 0.5
        return torch.where(use_x, velocity_x, velocity_native), use_x.float()
    raise ValueError(condition)


def available_conditions(model_ids: Sequence[str]) -> list[str]:
    present = set(model_ids)
    conditions: list[str] = ["Bayes_exact"]
    if "B0_v" in present:
        conditions.append("B0_v_ind")
    if "B1_x" in present:
        conditions.append("B1_x_ind")
    if "B2_eps" in present:
        conditions.append("B2_eps_ind")
    if "D0_xeps" in present:
        conditions.extend(
            (
                "D0_x_shared",
                "D0_eps_shared",
                "D0_fixed_x_eps",
                "D0_safe_schedule",
                "D3_oracle_bayes_gate",
            )
        )
    if "D1_scaled" in present:
        conditions.append("D1_scaled_gate")
    if "D2_velocity" in present:
        conditions.append("D2_velocity_gate")
    if "D4_safe" in present:
        conditions.append("D4_safe_velocity_gate")
    if "S0_xv" in present:
        conditions.append("S0_xv_switch")
    if "S1_xv" in present:
        conditions.append("S1_xv_consistency_switch")
    return conditions


def available_cross_gate_conditions(model_ids: Sequence[str]) -> list[str]:
    """Endpoint-only controls that separate gate choice from branch quality."""
    present = set(model_ids)
    conditions: list[str] = []
    for model_id, prefix in (
        ("D1_scaled", "D1"),
        ("D2_velocity", "D2"),
        ("D4_safe", "D4"),
    ):
        if model_id not in present:
            continue
        conditions.extend(
            (
                f"{prefix}_x_own",
                f"{prefix}_eps_own",
                f"{prefix}_gate_on_D0",
            )
        )
    return conditions


def fixed_swd(
    sample: np.ndarray,
    reference: np.ndarray,
    *,
    directions: np.ndarray,
) -> float:
    n = min(len(sample), len(reference))
    sample_projection = np.sort(sample[:n] @ directions.T, axis=0)
    reference_projection = np.sort(reference[:n] @ directions.T, axis=0)
    return float(np.mean(np.abs(sample_projection - reference_projection)))


def fixed_directions(dimension: int, count: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    directions = generator.normal(size=(count, dimension))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True).clip(1e-12)
    return directions.astype(np.float32)


def reference_rbf_bandwidth(reference: np.ndarray, *, max_points: int) -> float:
    n = min(len(reference), max_points)
    value = torch.from_numpy(reference[:n]).float()
    return float(torch.pdist(value).square().median().clamp_min(1e-8))


def rbf_mmd(
    sample: np.ndarray,
    reference: np.ndarray,
    *,
    max_points: int,
    bandwidth: float,
) -> float:
    n = min(len(sample), len(reference), max_points)
    x = torch.from_numpy(sample[:n]).float()
    y = torch.from_numpy(reference[:n]).float()
    xx = torch.cdist(x, x).square()
    yy = torch.cdist(y, y).square()
    xy = torch.cdist(x, y).square()
    kernel_xx = torch.exp(-xx / (2.0 * bandwidth))
    kernel_yy = torch.exp(-yy / (2.0 * bandwidth))
    kernel_xy = torch.exp(-xy / (2.0 * bandwidth))
    off_x = (kernel_xx.sum() - kernel_xx.diagonal().sum()) / max(n * (n - 1), 1)
    off_y = (kernel_yy.sum() - kernel_yy.diagonal().sum()) / max(n * (n - 1), 1)
    return float(off_x + off_y - 2.0 * kernel_xy.mean())


@torch.no_grad()
def sample_heun(
    condition: str,
    *,
    suite: ModelSuite,
    distribution: SpiralGaussianMixture,
    initial_noise: torch.Tensor,
    steps: int,
    denominator_floor: float,
    snapshot_times: Sequence[float],
) -> tuple[torch.Tensor, dict[float, torch.Tensor]]:
    state = initial_noise.clone()
    snapshots: dict[float, torch.Tensor] = {}
    snapshot_steps = {int(round(value * steps)): float(value) for value in snapshot_times}
    if 0 in snapshot_steps:
        snapshots[snapshot_steps[0]] = state.detach().cpu()
    step_size = 1.0 / steps
    for step in range(steps):
        t0 = float(step) / steps
        t1 = float(step + 1) / steps
        time0 = torch.full((len(state),), t0, device=state.device)
        velocity0, _ = condition_field(
            condition,
            suite=suite,
            distribution=distribution,
            state=state,
            time_value=time0,
            denominator_floor=denominator_floor,
        )
        proposal = state + step_size * velocity0
        time1 = torch.full((len(state),), t1, device=state.device)
        velocity1, _ = condition_field(
            condition,
            suite=suite,
            distribution=distribution,
            state=proposal,
            time_value=time1,
            denominator_floor=denominator_floor,
        )
        state = state + 0.5 * step_size * (velocity0 + velocity1)
        completed = step + 1
        if completed in snapshot_steps:
            snapshots[snapshot_steps[completed]] = state.detach().cpu()
    return state, snapshots


def teacher_diagnostics(
    *,
    suite: ModelSuite,
    distribution: SpiralGaussianMixture,
    conditions: Sequence[str],
    times: Sequence[float],
    samples: int,
    denominator_floor: float,
    seed: int,
) -> list[dict]:
    rows: list[dict] = []
    generator = torch.Generator(device=distribution.device.type).manual_seed(seed)
    with torch.no_grad():
        for time_point in times:
            clean, _, _ = distribution.sample(samples, generator=generator)
            epsilon = torch.randn(clean.shape, device=clean.device, generator=generator)
            time_value = torch.full((samples,), time_point, device=clean.device)
            state = (1.0 - time_value[:, None]) * epsilon + time_value[:, None] * clean
            paired_velocity = clean - epsilon
            bayes_velocity = distribution.bayes_velocity(
                state, time_value, denominator_floor=denominator_floor
            )
            irreducible = float(row_mse(paired_velocity - bayes_velocity).mean())
            for condition in conditions:
                prediction, gate = condition_field(
                    condition,
                    suite=suite,
                    distribution=distribution,
                    state=state,
                    time_value=time_value,
                    denominator_floor=denominator_floor,
                )
                row = {
                    "time": time_point,
                    "condition": condition,
                    "paired_velocity_mse": float(
                        row_mse(prediction - paired_velocity).mean()
                    ),
                    "bayes_velocity_mse": float(
                        row_mse(prediction - bayes_velocity).mean()
                    ),
                    "irreducible_velocity_mse": irreducible,
                }
                if gate is not None:
                    row.update(
                        {
                            "gate_mean": float(gate.mean()),
                            "gate_std": float(gate.std(unbiased=False)),
                            "x_amplification_mean": float(
                                (gate / max(1.0 - time_point, denominator_floor)).mean()
                            ),
                            "epsilon_amplification_mean": float(
                                ((1.0 - gate) / max(time_point, denominator_floor)).mean()
                            ),
                        }
                    )
                rows.append(row)

            if "D0_xeps" in suite.models:
                output = suite.models["D0_xeps"](state, time_value)
                velocity_x, velocity_epsilon = endpoint_velocities(
                    state=state,
                    time_value=time_value,
                    clean_prediction=output["x"],
                    epsilon_prediction=output["eps"],
                    denominator_floor=denominator_floor,
                )
                pair_gate = analytic_scalar_gate(
                    velocity_x, velocity_epsilon, paired_velocity, clip=True
                )
                pair_prediction = (
                    pair_gate * velocity_x + (1.0 - pair_gate) * velocity_epsilon
                )
                rows.append(
                    {
                        "time": time_point,
                        "condition": "D3_oracle_pair_teacher_only",
                        "paired_velocity_mse": float(
                            row_mse(pair_prediction - paired_velocity).mean()
                        ),
                        "bayes_velocity_mse": float(
                            row_mse(pair_prediction - bayes_velocity).mean()
                        ),
                        "irreducible_velocity_mse": irreducible,
                        "gate_mean": float(pair_gate.mean()),
                        "gate_std": float(pair_gate.std(unbiased=False)),
                        "x_amplification_mean": float(
                            (pair_gate / max(1.0 - time_point, denominator_floor)).mean()
                        ),
                        "epsilon_amplification_mean": float(
                            ((1.0 - pair_gate) / max(time_point, denominator_floor)).mean()
                        ),
                    }
                )
    return rows


def gradient_audit(
    *,
    suite: ModelSuite,
    distribution: SpiralGaussianMixture,
    samples: int,
    seed: int,
) -> list[dict]:
    if "D0_xeps" not in suite.models:
        return []
    generator = torch.Generator(device=distribution.device.type).manual_seed(seed)
    clean, _, _ = distribution.sample(samples, generator=generator)
    epsilon = torch.randn(clean.shape, device=clean.device, generator=generator)
    time_value = torch.rand(samples, device=clean.device, generator=generator)
    state = (1.0 - time_value[:, None]) * epsilon + time_value[:, None] * clean
    model = suite.models["D0_xeps"]
    output = model(state, time_value)
    clean_loss = F.mse_loss(output["x"], clean)
    epsilon_loss = F.mse_loss(output["eps"], epsilon)
    named_parameters = list(model.trunk.named_parameters())
    parameters = [parameter for _, parameter in named_parameters]
    clean_gradients = torch.autograd.grad(clean_loss, parameters, retain_graph=True)
    epsilon_gradients = torch.autograd.grad(epsilon_loss, parameters)

    rows: list[dict] = []
    all_clean: list[torch.Tensor] = []
    all_epsilon: list[torch.Tensor] = []
    for (name, _), clean_gradient, epsilon_gradient in zip(
        named_parameters, clean_gradients, epsilon_gradients
    ):
        clean_flat = clean_gradient.detach().float().flatten()
        epsilon_flat = epsilon_gradient.detach().float().flatten()
        all_clean.append(clean_flat)
        all_epsilon.append(epsilon_flat)
        rows.append(
            {
                "parameter": name,
                "clean_grad_norm": float(clean_flat.norm()),
                "epsilon_grad_norm": float(epsilon_flat.norm()),
                "cosine": float(F.cosine_similarity(clean_flat, epsilon_flat, dim=0)),
            }
        )
    clean_flat = torch.cat(all_clean)
    epsilon_flat = torch.cat(all_epsilon)
    rows.append(
        {
            "parameter": "ALL_TRUNK",
            "clean_grad_norm": float(clean_flat.norm()),
            "epsilon_grad_norm": float(epsilon_flat.norm()),
            "cosine": float(F.cosine_similarity(clean_flat, epsilon_flat, dim=0)),
        }
    )
    return rows


@torch.no_grad()
def branch_pair_diagnostics(
    *,
    suite: ModelSuite,
    distribution: SpiralGaussianMixture,
    times: Sequence[float],
    samples: int,
    denominator_floor: float,
    seed: int,
) -> list[dict]:
    rows: list[dict] = []
    generator = torch.Generator(device=distribution.device.type).manual_seed(seed)
    for time_point in times:
        clean, _, _ = distribution.sample(samples, generator=generator)
        epsilon = torch.randn(clean.shape, device=clean.device, generator=generator)
        time_value = torch.full((samples,), time_point, device=clean.device)
        state = (1.0 - time_value[:, None]) * epsilon + time_value[:, None] * clean
        paired_target = clean - epsilon
        bayes_target = distribution.bayes_velocity(
            state, time_value, denominator_floor=denominator_floor
        )

        for model_id in ("D0_xeps", "D1_scaled", "D2_velocity", "D4_safe"):
            if model_id not in suite.models:
                continue
            output = suite.models[model_id](state, time_value)
            first, second = endpoint_velocities(
                state=state,
                time_value=time_value,
                clean_prediction=output["x"],
                epsilon_prediction=output["eps"],
                denominator_floor=denominator_floor,
            )
            first_name, second_name = "x", "epsilon"
            model_kind = "x_plus_epsilon"
            prediction_gate = None
            if "gate" in output:
                prediction_gate = gate_value(
                    output["gate"],
                    time_value,
                    asymptotic_safe=model_id == "D4_safe",
                    denominator_floor=denominator_floor,
                )

            rows.append(
                _branch_pair_row(
                    model_id=model_id,
                    model_kind=model_kind,
                    first_name=first_name,
                    second_name=second_name,
                    first=first,
                    second=second,
                    paired_target=paired_target,
                    bayes_target=bayes_target,
                    prediction_gate=prediction_gate,
                    time_point=time_point,
                )
            )

        for model_id in ("S0_xv", "S1_xv"):
            if model_id not in suite.models:
                continue
            model = suite.models[model_id]
            second = model(state, time_value, 0)
            clean_prediction = model(state, time_value, 1)
            first = (clean_prediction - state) / (
                1.0 - time_value[:, None]
            ).clamp_min(denominator_floor)
            rows.append(
                _branch_pair_row(
                    model_id=model_id,
                    model_kind="x_plus_v",
                    first_name="x",
                    second_name="v",
                    first=first,
                    second=second,
                    paired_target=paired_target,
                    bayes_target=bayes_target,
                    prediction_gate=None,
                    time_point=time_point,
                )
            )
    return rows


def _branch_pair_row(
    *,
    model_id: str,
    model_kind: str,
    first_name: str,
    second_name: str,
    first: torch.Tensor,
    second: torch.Tensor,
    paired_target: torch.Tensor,
    bayes_target: torch.Tensor,
    prediction_gate: torch.Tensor | None,
    time_point: float,
) -> dict:
    first_error = first - bayes_target
    second_error = second - bayes_target
    cosine = F.cosine_similarity(first_error, second_error, dim=1, eps=EPS)
    oracle_gate = analytic_scalar_gate(first, second, bayes_target, clip=True)
    oracle_prediction = oracle_gate * first + (1.0 - oracle_gate) * second
    first_bayes_mse = float(row_mse(first_error).mean())
    second_bayes_mse = float(row_mse(second_error).mean())
    oracle_bayes_mse = float(row_mse(oracle_prediction - bayes_target).mean())
    row = {
        "time": time_point,
        "model": model_id,
        "model_kind": model_kind,
        "first_branch": first_name,
        "second_branch": second_name,
        "first_bayes_mse": first_bayes_mse,
        "second_bayes_mse": second_bayes_mse,
        "first_paired_mse": float(row_mse(first - paired_target).mean()),
        "second_paired_mse": float(row_mse(second - paired_target).mean()),
        "branch_error_cosine_mean": float(cosine.mean()),
        "branch_error_cosine_negative_fraction": float((cosine < 0).float().mean()),
        "oracle_gate_mean": float(oracle_gate.mean()),
        "oracle_bayes_mse": oracle_bayes_mse,
        "oracle_gain_over_best_branch": 1.0
        - oracle_bayes_mse / max(min(first_bayes_mse, second_bayes_mse), EPS),
    }
    if prediction_gate is not None:
        prediction = prediction_gate * first + (1.0 - prediction_gate) * second
        row.update(
            {
                "learned_gate_mean": float(prediction_gate.mean()),
                "learned_bayes_mse": float(row_mse(prediction - bayes_target).mean()),
                "learned_gate_mae_to_oracle": float(
                    (prediction_gate - oracle_gate).abs().mean()
                ),
            }
        )
    return row


def endpoint_metrics(
    *,
    generated: dict[str, torch.Tensor],
    reference: torch.Tensor,
    distribution: SpiralGaussianMixture,
    seed: int,
    swd_projections: int,
    ambient_swd_projections: int,
    mmd_max_points: int,
) -> list[dict]:
    reference_device = reference.to(distribution.device)
    reference_intrinsic = distribution.decode_intrinsic(reference_device).cpu().numpy()
    mmd_bandwidth = reference_rbf_bandwidth(
        reference_intrinsic, max_points=mmd_max_points
    )
    intrinsic_directions = fixed_directions(2, swd_projections, seed)
    ambient_directions = fixed_directions(
        distribution.ambient_dim, ambient_swd_projections, seed + 1
    )
    reference_ambient = reference.cpu().numpy()
    uniform = torch.full(
        (distribution.components,),
        1.0 / distribution.components,
        device=distribution.device,
    )
    rows = []
    for condition, value_cpu in generated.items():
        value = value_cpu.to(distribution.device)
        intrinsic = distribution.decode_intrinsic(value)
        histogram = distribution.component_histogram(intrinsic)
        rows.append(
            {
                "condition": condition,
                "intrinsic_swd": fixed_swd(
                    intrinsic.cpu().numpy(),
                    reference_intrinsic,
                    directions=intrinsic_directions,
                ),
                "ambient_swd": fixed_swd(
                    value_cpu.numpy(), reference_ambient, directions=ambient_directions
                ),
                "intrinsic_mmd": rbf_mmd(
                    intrinsic.cpu().numpy(),
                    reference_intrinsic,
                    max_points=mmd_max_points,
                    bandwidth=mmd_bandwidth,
                ),
                "intrinsic_nll": float(distribution.intrinsic_nll(intrinsic).mean()),
                "component_tv": float(0.5 * (histogram - uniform).abs().sum()),
                "component_coverage": float((histogram > 0.0025).float().mean()),
                "off_subspace_rms": float(distribution.off_subspace_rms(value).mean()),
            }
        )
    return rows


def rollout_diagnostics(
    *,
    snapshots: dict[str, dict[float, torch.Tensor]],
    suite: ModelSuite,
    distribution: SpiralGaussianMixture,
    denominator_floor: float,
    seed: int,
) -> list[dict]:
    rows: list[dict] = []
    true_cache: dict[tuple[float, int], torch.Tensor] = {}
    for condition, condition_snapshots in snapshots.items():
        for time_point, state_cpu in sorted(condition_snapshots.items()):
            state = state_cpu.to(distribution.device)
            time_value = torch.full((len(state),), time_point, device=state.device)
            prediction, gate = condition_field(
                condition,
                suite=suite,
                distribution=distribution,
                state=state,
                time_value=time_value,
                denominator_floor=denominator_floor,
            )
            bayes = distribution.bayes_velocity(
                state, time_value, denominator_floor=denominator_floor
            )
            cache_key = (time_point, len(state))
            if cache_key not in true_cache:
                generator = torch.Generator(device=distribution.device.type).manual_seed(
                    stable_seed(seed, int(round(time_point * 10000)), len(state))
                )
                clean, _, _ = distribution.sample(len(state), generator=generator)
                epsilon = torch.randn(clean.shape, device=clean.device, generator=generator)
                true_cache[cache_key] = (
                    (1.0 - time_point) * epsilon + time_point * clean
                ).detach()
            true_state = true_cache[cache_key]
            state_intrinsic = distribution.decode_intrinsic(state).cpu().numpy()
            true_intrinsic = distribution.decode_intrinsic(true_state).cpu().numpy()
            directions = fixed_directions(2, 128, stable_seed(seed, 991))
            row = {
                "condition": condition,
                "time": time_point,
                "rollout_bayes_velocity_mse": float(row_mse(prediction - bayes).mean()),
                "state_intrinsic_swd": fixed_swd(
                    state_intrinsic, true_intrinsic, directions=directions
                ),
                "state_off_subspace_rms": float(
                    distribution.off_subspace_rms(state).mean()
                ),
                "true_off_subspace_rms": float(
                    distribution.off_subspace_rms(true_state).mean()
                ),
            }
            if gate is not None:
                row.update(
                    {
                        "gate_mean": float(gate.mean()),
                        "gate_std": float(gate.std(unbiased=False)),
                    }
                )
            rows.append(row)
    return rows


def plot_endpoint_scatter(
    path: Path,
    *,
    generated: dict[str, torch.Tensor],
    reference: torch.Tensor,
    distribution: SpiralGaussianMixture,
    limit: int,
) -> None:
    preferred = [
        "Bayes_exact",
        "B0_v_ind",
        "B1_x_ind",
        "B2_eps_ind",
        "D0_fixed_x_eps",
        "D0_safe_schedule",
        "D1_scaled_gate",
        "D2_velocity_gate",
        "D3_oracle_bayes_gate",
        "D4_safe_velocity_gate",
        "S0_xv_switch",
        "S1_xv_consistency_switch",
    ]
    names = [name for name in preferred if name in generated]
    columns = 4
    rows = math.ceil((len(names) + 1) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(18, 4.4 * rows), squeeze=False)
    panels = [("Reference", reference)] + [(name, generated[name]) for name in names]
    for axis, (name, value) in zip(axes.flat, panels):
        intrinsic = distribution.decode_intrinsic(value.to(distribution.device))
        points = intrinsic[:limit].cpu().numpy()
        axis.scatter(points[:, 0], points[:, 1], s=4, alpha=0.35, rasterized=True)
        axis.set_title(name)
        axis.set_aspect("equal")
        axis.set_xlim(-1.8, 1.8)
        axis.set_ylim(-1.8, 1.8)
    for axis in axes.flat[len(panels) :]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_mechanism(
    path: Path,
    *,
    endpoint_rows: Sequence[dict],
    teacher_rows: Sequence[dict],
    rollout_rows: Sequence[dict],
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    endpoint_sorted = sorted(endpoint_rows, key=lambda row: row["intrinsic_swd"])
    axes[0].barh(
        [row["condition"] for row in endpoint_sorted],
        [row["intrinsic_swd"] for row in endpoint_sorted],
    )
    axes[0].set_xlabel("Endpoint intrinsic SWD (lower is better)")
    axes[0].invert_yaxis()

    selected = {
        "B0_v_ind",
        "B1_x_ind",
        "B2_eps_ind",
        "D1_scaled_gate",
        "D2_velocity_gate",
        "D3_oracle_bayes_gate",
        "D4_safe_velocity_gate",
    }
    for condition in sorted(selected):
        teacher = sorted(
            (row for row in teacher_rows if row["condition"] == condition),
            key=lambda row: row["time"],
        )
        if teacher:
            axes[1].plot(
                [row["time"] for row in teacher],
                [row["bayes_velocity_mse"] for row in teacher],
                marker="o",
                label=condition,
            )
        rollout = sorted(
            (row for row in rollout_rows if row["condition"] == condition),
            key=lambda row: row["time"],
        )
        if rollout:
            axes[2].plot(
                [row["time"] for row in rollout],
                [row["rollout_bayes_velocity_mse"] for row in rollout],
                marker="o",
                label=condition,
            )
    axes[1].set_title("Teacher states")
    axes[2].set_title("Rollout states")
    for axis in axes[1:]:
        axis.set_xlabel("t")
        axis.set_ylabel("MSE to exact Bayes field")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
    axes[2].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_setting(args: argparse.Namespace, ambient_dim: int, seed: int) -> Path:
    output_dir = args.output_root / f"seed{seed}" / f"D{ambient_dim}_H{args.hidden_dim}"
    output_dir.mkdir(parents=True, exist_ok=True)
    complete_path = output_dir / "complete.json"
    if complete_path.exists() and not args.overwrite and not args.evaluation_only:
        print(f"skip complete setting: {output_dir}", flush=True)
        return output_dir

    set_seed(seed)
    device = torch.device(args.device)
    distribution = SpiralGaussianMixture(
        ambient_dim,
        components=args.components,
        component_std=args.component_std,
        seed=stable_seed(seed, ambient_dim, 71),
        device=device,
    )
    suite = build_model_suite(
        ambient_dim=ambient_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        time_dim=args.time_dim,
        mode_dim=args.mode_dim,
        model_ids=args.model_ids,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=stable_seed(seed, ambient_dim, 113),
        device=device,
    )
    if args.evaluation_only:
        checkpoint = torch.load(
            output_dir / "checkpoint.pt", map_location=device, weights_only=False
        )
        for model_id, model in suite.models.items():
            model.load_state_dict(checkpoint["models"][model_id])
    else:
        config = {
            **vars(args),
            "output_root": str(args.output_root),
            "ambient_dim": ambient_dim,
            "seed": seed,
            "device_resolved": str(device),
            "time_convention": "state=(1-t)*epsilon+t*clean; velocity=clean-epsilon",
            "scale": distribution.scale,
        }
        save_json(output_dir / "config.json", config)
        history = train_models(
            suite=suite,
            distribution=distribution,
            steps=args.train_steps,
            batch_size=args.batch_size,
            t_min=args.train_t_min,
            t_max=args.train_t_max,
            denominator_floor=args.denominator_floor,
            consistency_weight=args.consistency_weight,
            grad_clip=args.grad_clip,
            log_every=args.log_every,
            seed=stable_seed(seed, ambient_dim, 127),
            checkpoint_path=output_dir / "checkpoint.pt",
        )
        save_csv(output_dir / "train_history.csv", history)
    for model in suite.models.values():
        model.eval()

    conditions = available_conditions(args.model_ids)
    teacher_rows = teacher_diagnostics(
        suite=suite,
        distribution=distribution,
        conditions=conditions,
        times=args.diagnostic_times,
        samples=args.teacher_samples,
        denominator_floor=args.denominator_floor,
        seed=stable_seed(seed, ambient_dim, 131),
    )
    save_csv(output_dir / "teacher_metrics.csv", teacher_rows)
    gradient_rows = gradient_audit(
        suite=suite,
        distribution=distribution,
        samples=args.gradient_samples,
        seed=stable_seed(seed, ambient_dim, 137),
    )
    save_csv(output_dir / "gradient_audit.csv", gradient_rows)
    branch_rows = branch_pair_diagnostics(
        suite=suite,
        distribution=distribution,
        times=args.diagnostic_times,
        samples=args.teacher_samples,
        denominator_floor=args.denominator_floor,
        seed=stable_seed(seed, ambient_dim, 139),
    )
    save_csv(output_dir / "branch_pair_metrics.csv", branch_rows)

    generator = torch.Generator(device=device.type).manual_seed(
        stable_seed(seed, ambient_dim, 149)
    )
    initial_noise = torch.randn(
        args.sample_count, ambient_dim, device=device, generator=generator
    )
    generated: dict[str, torch.Tensor] = {}
    all_snapshots: dict[str, dict[float, torch.Tensor]] = {}
    for condition in conditions:
        print(f"sampling {condition}", flush=True)
        endpoint, snapshots = sample_heun(
            condition,
            suite=suite,
            distribution=distribution,
            initial_noise=initial_noise,
            steps=args.sample_steps,
            denominator_floor=args.denominator_floor,
            snapshot_times=args.diagnostic_times,
        )
        generated[condition] = endpoint.detach().cpu()
        all_snapshots[condition] = snapshots

    cross_conditions = available_cross_gate_conditions(args.model_ids)
    cross_teacher_rows: list[dict] = []
    if cross_conditions:
        cross_teacher_rows = teacher_diagnostics(
            suite=suite,
            distribution=distribution,
            conditions=cross_conditions,
            times=args.diagnostic_times,
            samples=args.teacher_samples,
            denominator_floor=args.denominator_floor,
            seed=stable_seed(seed, ambient_dim, 131),
        )
        cross_teacher_rows = [
            row
            for row in cross_teacher_rows
            if row["condition"] != "D3_oracle_pair_teacher_only"
        ]

    cross_generated: dict[str, torch.Tensor] = {}
    cross_snapshots: dict[str, dict[float, torch.Tensor]] = {}
    for condition in cross_conditions:
        print(f"sampling cross-control {condition}", flush=True)
        endpoint, snapshots = sample_heun(
            condition,
            suite=suite,
            distribution=distribution,
            initial_noise=initial_noise,
            steps=args.sample_steps,
            denominator_floor=args.denominator_floor,
            snapshot_times=args.diagnostic_times,
        )
        cross_generated[condition] = endpoint.detach().cpu()
        cross_snapshots[condition] = snapshots

    reference_generator = torch.Generator(device=device.type).manual_seed(
        stable_seed(seed, ambient_dim, 151)
    )
    reference, _, _ = distribution.sample(args.reference_count, generator=reference_generator)
    reference_cpu = reference.detach().cpu()
    endpoint_rows = endpoint_metrics(
        generated=generated,
        reference=reference_cpu,
        distribution=distribution,
        seed=stable_seed(seed, ambient_dim, 157),
        swd_projections=args.swd_projections,
        ambient_swd_projections=args.ambient_swd_projections,
        mmd_max_points=args.mmd_max_points,
    )
    save_csv(output_dir / "endpoint_metrics.csv", endpoint_rows)
    if cross_generated:
        cross_rows = endpoint_metrics(
            generated=cross_generated,
            reference=reference_cpu,
            distribution=distribution,
            seed=stable_seed(seed, ambient_dim, 157),
            swd_projections=args.swd_projections,
            ambient_swd_projections=args.ambient_swd_projections,
            mmd_max_points=args.mmd_max_points,
        )
        common_rows = [
            row
            for row in endpoint_rows
            if row["condition"]
            in {"D0_x_shared", "D0_eps_shared", "D3_oracle_bayes_gate"}
        ]
        save_csv(output_dir / "cross_gate_endpoint_metrics.csv", common_rows + cross_rows)
    rollout_rows = rollout_diagnostics(
        snapshots=all_snapshots,
        suite=suite,
        distribution=distribution,
        denominator_floor=args.denominator_floor,
        seed=stable_seed(seed, ambient_dim, 163),
    )
    save_csv(output_dir / "rollout_metrics.csv", rollout_rows)
    if cross_conditions:
        common = {"D0_x_shared", "D0_eps_shared", "D3_oracle_bayes_gate"}
        save_csv(
            output_dir / "cross_gate_teacher_metrics.csv",
            [row for row in teacher_rows if row["condition"] in common]
            + cross_teacher_rows,
        )
        cross_rollout_rows = rollout_diagnostics(
            snapshots=cross_snapshots,
            suite=suite,
            distribution=distribution,
            denominator_floor=args.denominator_floor,
            seed=stable_seed(seed, ambient_dim, 163),
        )
        save_csv(
            output_dir / "cross_gate_rollout_metrics.csv",
            [row for row in rollout_rows if row["condition"] in common]
            + cross_rollout_rows,
        )
    plot_endpoint_scatter(
        output_dir / "endpoint_scatter.png",
        generated=generated,
        reference=reference_cpu,
        distribution=distribution,
        limit=args.plot_points,
    )
    plot_mechanism(
        output_dir / "mechanism_summary.png",
        endpoint_rows=endpoint_rows,
        teacher_rows=teacher_rows,
        rollout_rows=rollout_rows,
    )
    save_json(
        complete_path,
        {
            "conditions": conditions,
            "endpoint_rows": endpoint_rows,
            "best_intrinsic_swd": min(endpoint_rows, key=lambda row: row["intrinsic_swd"]),
        },
    )
    return output_dir


def aggregate_results(output_root: Path) -> tuple[Path, Path]:
    endpoint_rows: list[dict] = []
    mechanism_rows: list[dict] = []
    for path in sorted(output_root.glob("seed*/D*_H*/endpoint_metrics.csv")):
        seed = int(path.parts[-3].removeprefix("seed"))
        setting = path.parts[-2]
        ambient_dim = int(setting.split("_")[0].removeprefix("D"))
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                endpoint_rows.append(
                    {
                        "seed": seed,
                        "ambient_dim": ambient_dim,
                        "condition": row["condition"],
                        **{
                            key: float(value)
                            for key, value in row.items()
                            if key != "condition" and value != ""
                        },
                    }
                )
    grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in endpoint_rows:
        grouped[(row["ambient_dim"], row["condition"])].append(row)
    for (ambient_dim, condition), rows in sorted(grouped.items()):
        summary = {
            "ambient_dim": ambient_dim,
            "condition": condition,
            "seeds": len(rows),
        }
        numeric_keys = [
            key for key in rows[0] if key not in {"seed", "ambient_dim", "condition"}
        ]
        for key in numeric_keys:
            values = np.asarray([row[key] for row in rows], dtype=np.float64)
            summary[f"{key}_mean"] = float(values.mean())
            summary[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        mechanism_rows.append(summary)
    endpoint_path = output_root / "endpoint_metrics_all_seeds.csv"
    summary_path = output_root / "endpoint_metrics_seed_summary.csv"
    save_csv(endpoint_path, endpoint_rows)
    save_csv(summary_path, mechanism_rows)
    return endpoint_path, summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--skip-aggregate", action="store_true")
    parser.add_argument("--evaluation-only", action="store_true")
    parser.add_argument("--dims", type=parse_int_list, default=parse_int_list("2,512"))
    parser.add_argument(
        "--seeds", type=parse_int_list, default=parse_int_list("20260821,20260822,20260823")
    )
    parser.add_argument("--model-ids", type=parse_str_list, default=list(MODEL_IDS))
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--time-dim", type=int, default=32)
    parser.add_argument("--mode-dim", type=int, default=8)
    parser.add_argument("--components", type=int, default=32)
    parser.add_argument("--component-std", type=float, default=0.035)
    parser.add_argument("--train-steps", type=int, default=15000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--train-t-min", type=float, default=0.001)
    parser.add_argument("--train-t-max", type=float, default=0.999)
    parser.add_argument("--denominator-floor", type=float, default=1e-3)
    parser.add_argument("--consistency-weight", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument(
        "--diagnostic-times",
        type=parse_float_list,
        default=parse_float_list("0.01,0.03,0.1,0.3,0.5,0.7,0.9,0.97,0.99"),
    )
    parser.add_argument("--teacher-samples", type=int, default=4096)
    parser.add_argument("--gradient-samples", type=int, default=2048)
    parser.add_argument("--sample-count", type=int, default=4096)
    parser.add_argument("--reference-count", type=int, default=8192)
    parser.add_argument("--sample-steps", type=int, default=200)
    parser.add_argument("--swd-projections", type=int, default=256)
    parser.add_argument("--ambient-swd-projections", type=int, default=64)
    parser.add_argument("--mmd-max-points", type=int, default=2048)
    parser.add_argument("--plot-points", type=int, default=3000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0 <= args.train_t_min < args.train_t_max <= 1:
        raise ValueError("invalid training time range")
    if not 0 < args.denominator_floor < 0.5:
        raise ValueError("denominator_floor must be in (0, 0.5)")
    if args.sample_steps < 2:
        raise ValueError("sample_steps must be at least 2")
    for time_point in args.diagnostic_times:
        scaled = time_point * args.sample_steps
        if abs(scaled - round(scaled)) > 1e-8:
            raise ValueError(
                f"diagnostic time {time_point} is not aligned with sample steps"
            )


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        aggregate_results(args.output_root)
        return
    for seed in args.seeds:
        for ambient_dim in args.dims:
            run_setting(args, ambient_dim, seed)
    if not args.skip_aggregate:
        aggregate_results(args.output_root)


if __name__ == "__main__":
    main()
