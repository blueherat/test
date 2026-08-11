#!/usr/bin/env python3
"""Exact-Bayes prediction-target capacity experiment (v5).

This experiment removes two confounds from the earlier v3/v4 toy:

1. The "oracle" is analytic.  Clean data follow a finite mixture of Gaussian
   tangent patches, so E[x | x_t, t] is available in closed form.
2. Residual architectures are explicit controls.  ``residual_skip`` adds a
   time-gated full-rank state path, removing the H < D output-rank bottleneck
   for identity-like target components.

Normal energy is reported as geometry, not labelled as error.  Final quality is
measured independently in the ambient latent distribution and after a fixed
anisotropic nonlinear pushforward, mirroring the fact that a decoder can
rotate, suppress, or amplify latent directions.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.run_prediction_target_extrapolation_toy_v3 import (
    DenoiseMLP as LegacyJitReLUMlp,
)
from experiments.run_prediction_target_extrapolation_toy_v4 import (
    CurvedEmbedding,
    DenoiseMLP,
    TimeEmbedding,
    clean_from_output,
    direct_target,
    parse_float_list,
    parse_int_list,
    parse_str_list,
    stable_seed,
    tag_float,
    velocity_from_output,
)

EPS = 1e-12


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def save_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def row_rms(x: torch.Tensor) -> torch.Tensor:
    return x.double().square().mean(dim=1).sqrt().float()


def row_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a.double(), b.double(), dim=1, eps=EPS).float()


def spiral_centers(k: int, device: torch.device) -> torch.Tensor:
    s = (torch.arange(k, device=device, dtype=torch.float32) + 0.5) / float(k)
    # A fixed offset avoids degenerate small-K visualizations on one axis.
    angle = 4.0 * math.pi * s + 0.37
    radius = 0.15 + 0.85 * s
    return 1.6 * torch.stack([radius * angle.cos(), radius * angle.sin()], dim=1)


class TangentGaussianMixture:
    """A curved mixture with an exact posterior clean mean.

    Component k has

        x | k ~ N(mu_k, sigma_tangent^2 P_k + sigma_normal^2 (I - P_k)),

    where P_k is the rank-2 local tangent projector.  The linear interpolation
    path x_t=(1-t)x+t*eps leaves every component Gaussian, so posterior weights
    and E[x | x_t,t] are analytic.
    """

    def __init__(
        self,
        *,
        D: int,
        components: int,
        curvature: float,
        frequency_scale: float,
        center_rms: float,
        sigma_tangent: float,
        sigma_normal: float,
        seed: int,
        device: torch.device,
    ) -> None:
        if D < 3:
            raise ValueError("D must be >= 3")
        if components < 2:
            raise ValueError("components must be >= 2")
        if sigma_tangent <= 0 or sigma_normal < 0:
            raise ValueError("invalid component scales")
        self.D = int(D)
        self.components = int(components)
        self.sigma_tangent = float(sigma_tangent)
        self.sigma_normal = float(sigma_normal)
        self.device = device

        embedding = CurvedEmbedding(
            D,
            curvature=curvature,
            frequency_scale=frequency_scale,
            seed=seed,
            device=device,
            scale_mode="unit_rms",
        )
        u = spiral_centers(components, device)
        raw_means = embedding.embed(u)
        self.embedding = embedding
        self.center_shift = raw_means.mean(dim=0, keepdim=True)
        centered_means = raw_means - self.center_shift
        rms = centered_means.square().mean().sqrt().clamp_min(1e-8)
        self.center_scale = float(center_rms) / float(rms)
        self.means = centered_means * self.center_scale
        self.intrinsic_centers = u
        self.bases = embedding.tangent_basis(u)
        self.log_weights = torch.full(
            (components,), -math.log(float(components)), device=device
        )

    def sample_clean(
        self, n: int, *, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        component = torch.randint(
            self.components, (n,), device=self.device, generator=generator
        )
        basis = self.bases[component]
        tangent_coeff = torch.randn(
            n, 2, device=self.device, generator=generator
        )
        tangent = torch.einsum("bdr,br->bd", basis, tangent_coeff)
        normal_raw = torch.randn(
            n, self.D, device=self.device, generator=generator
        )
        normal_tangent = torch.einsum(
            "bdr,br->bd",
            basis,
            torch.einsum("bdr,bd->br", basis, normal_raw),
        )
        normal = normal_raw - normal_tangent
        x = (
            self.means[component]
            + self.sigma_tangent * tangent
            + self.sigma_normal * normal
        )
        return x, component

    def noised_batch(
        self,
        n: int,
        *,
        t_min: float,
        t_max: float,
        time_sampler: str = "uniform",
        time_logit_mean: float = 0.8,
        time_logit_std: float = 0.8,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x, component = self.sample_clean(n, generator=generator)
        eps = torch.randn(x.shape, device=self.device, generator=generator)
        if time_sampler == "uniform":
            t = torch.empty(n, device=self.device).uniform_(
                t_min, t_max, generator=generator
            )
        elif time_sampler == "logit_normal":
            if time_logit_std <= 0.0:
                raise ValueError("time_logit_std must be positive")
            logits = torch.randn(n, device=self.device, generator=generator)
            logits.mul_(time_logit_std).add_(time_logit_mean)
            t = logits.sigmoid().clamp_(min=t_min, max=t_max)
        else:
            raise ValueError(f"unsupported time_sampler: {time_sampler}")
        x_t = (1.0 - t[:, None]) * x + t[:, None] * eps
        return x, eps, t, x_t, component

    def posterior_clean(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        *,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Compute the exact conditional mean E[x | x_t,t]."""
        if x_t.ndim != 2 or x_t.shape[1] != self.D:
            raise ValueError("x_t must be [B,D]")
        if t.ndim != 1 or len(t) != len(x_t):
            raise ValueError("t must be [B]")
        a = 1.0 - t
        residual = x_t[:, None, :] - a[:, None, None] * self.means[None]
        tangent_coeff = torch.einsum("bkd,kdr->bkr", residual, self.bases)
        tangent_sq = tangent_coeff.square().sum(dim=2)
        residual_sq = residual.square().sum(dim=2)
        normal_sq = (residual_sq - tangent_sq).clamp_min(0.0)

        var_tangent = a.square() * self.sigma_tangent**2 + t.square()
        var_normal = a.square() * self.sigma_normal**2 + t.square()
        log_det = (
            2.0 * var_tangent.log()
            + float(self.D - 2) * var_normal.log()
        )
        log_likelihood = -0.5 * (
            tangent_sq / var_tangent[:, None]
            + normal_sq / var_normal[:, None]
            + log_det[:, None]
            + float(self.D) * math.log(2.0 * math.pi)
        )
        weights = torch.softmax(log_likelihood + self.log_weights[None], dim=1)

        tangent_vector = torch.einsum(
            "bkr,kdr->bkd", tangent_coeff, self.bases
        )
        coeff_tangent = (
            a * self.sigma_tangent**2 / var_tangent
        )[:, None, None]
        coeff_normal = (
            a * self.sigma_normal**2 / var_normal
        )[:, None, None]
        correction = (
            coeff_normal * residual
            + (coeff_tangent - coeff_normal) * tangent_vector
        )
        component_mean = self.means[None] + correction
        posterior_mean = torch.einsum("bk,bkd->bd", weights, component_mean)
        if return_weights:
            return posterior_mean, weights
        return posterior_mean

    def split_by_component(
        self, vector: torch.Tensor, component: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        basis = self.bases[component]
        tangent = torch.einsum(
            "bdr,br->bd", basis, torch.einsum("bdr,bd->br", basis, vector)
        )
        return tangent, vector - tangent

    def nearest_components(self, x: torch.Tensor, batch_size: int = 2048) -> torch.Tensor:
        out = []
        means_sq = self.means.square().sum(dim=1)
        for start in range(0, len(x), batch_size):
            xb = x[start : start + batch_size]
            dist = (
                xb.square().sum(dim=1, keepdim=True)
                + means_sq[None]
                - 2.0 * xb @ self.means.T
            )
            out.append(dist.argmin(dim=1))
        return torch.cat(out)

    def nearest_patch_geometry(self, x: torch.Tensor) -> dict[str, float]:
        component = self.nearest_components(x)
        residual = x - self.means[component]
        tangent, normal = self.split_by_component(residual, component)
        return {
            "nearest_tangent_rms": float(row_rms(tangent).mean().cpu()),
            "nearest_normal_rms": float(row_rms(normal).mean().cpu()),
            "nearest_normal_energy_fraction": float(
                normal.double().square().sum()
                / residual.double().square().sum().clamp_min(EPS)
            ),
        }

    def intrinsic_readout(self, x: torch.Tensor) -> torch.Tensor:
        """Return the fixed 2-D construction coordinates used for visualization."""
        unscaled = x / self.center_scale + self.center_shift
        return self.embedding.decode_intrinsic(unscaled)


class ResidualBlock(nn.Module):
    def __init__(self, hidden: int, expansion: int = 2) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, expansion * hidden)
        self.fc2 = nn.Linear(expansion * hidden, hidden)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(F.silu(self.fc1(self.norm(x))))


class ResidualDenoiseMLP(nn.Module):
    """Residual MLP with an optional full-rank, time-gated state path."""

    def __init__(
        self,
        D: int,
        hidden: int,
        depth: int,
        time_dim: int,
        *,
        state_skip: bool,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be >= 2")
        self.time = TimeEmbedding(time_dim)
        self.in_proj = nn.Linear(D + time_dim, hidden)
        self.blocks = nn.ModuleList(
            ResidualBlock(hidden) for _ in range(max(depth - 2, 0))
        )
        self.out_norm = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, D)
        self.state_skip = bool(state_skip)
        if self.state_skip:
            gate_hidden = max(16, time_dim)
            self.skip_gate = nn.Sequential(
                nn.Linear(time_dim, gate_hidden),
                nn.SiLU(),
                nn.Linear(gate_hidden, 1),
            )
            nn.init.zeros_(self.skip_gate[-1].weight)
            nn.init.ones_(self.skip_gate[-1].bias)
        else:
            self.skip_gate = None

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        time = self.time(t)
        h = F.silu(self.in_proj(torch.cat([x, time], dim=1)))
        for block in self.blocks:
            h = block(h)
        output = self.out_proj(self.out_norm(h))
        if self.skip_gate is not None:
            output = output + self.skip_gate(time) * x
        return output


def build_model(
    architecture: str,
    *,
    D: int,
    hidden: int,
    depth: int,
    time_dim: int,
) -> nn.Module:
    if architecture == "jit_relu":
        return LegacyJitReLUMlp(D, hidden, depth, time_dim)
    if architecture == "plain":
        return DenoiseMLP(D, hidden, depth, time_dim)
    if architecture == "residual":
        return ResidualDenoiseMLP(
            D, hidden, depth, time_dim, state_skip=False
        )
    if architecture == "residual_skip":
        return ResidualDenoiseMLP(
            D, hidden, depth, time_dim, state_skip=True
        )
    raise ValueError(architecture)


def build_same_init_models(
    architecture: str,
    *,
    D: int,
    hidden: int,
    depth: int,
    time_dim: int,
    device: torch.device,
    seed: int,
) -> dict[str, nn.Module]:
    torch.manual_seed(seed)
    base = build_model(
        architecture, D=D, hidden=hidden, depth=depth, time_dim=time_dim
    ).to(device)
    state = copy.deepcopy(base.state_dict())
    models = {}
    for target in ("x", "v", "eps"):
        model = build_model(
            architecture, D=D, hidden=hidden, depth=depth, time_dim=time_dim
        ).to(device)
        model.load_state_dict(state)
        models[target] = model
    return models


def loss_for_output(
    output: torch.Tensor,
    *,
    x_t: torch.Tensor,
    t: torch.Tensor,
    x: torch.Tensor,
    eps: torch.Tensor,
    target: str,
    loss_space: str,
    conversion_clip: float,
) -> torch.Tensor:
    if loss_space == "v":
        prediction = velocity_from_output(
            output, x_t, t, target, conversion_clip
        )
        return F.mse_loss(prediction, eps - x)
    if loss_space == "direct":
        return F.mse_loss(output, direct_target(x, eps, target))
    raise ValueError(loss_space)


@torch.inference_mode()
def validation_metrics(
    models: dict[str, nn.Module],
    *,
    x: torch.Tensor,
    eps: torch.Tensor,
    t: torch.Tensor,
    x_t: torch.Tensor,
    bayes: torch.Tensor,
    conversion_clip: float,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    bayes_risk = (x - bayes).double().square().mean()
    metrics["bayes_risk_mse"] = float(bayes_risk.cpu())
    for target, model in models.items():
        clean = clean_from_output(
            model(x_t, t), x_t, t, target, conversion_clip
        )
        paired = (clean - x).double().square().mean()
        excess = (clean - bayes).double().square().mean()
        metrics[f"{target}_paired_mse"] = float(paired.cpu())
        metrics[f"{target}_excess_mse"] = float(excess.cpu())
        metrics[f"{target}_risk_closure"] = float(
            (paired - bayes_risk - excess).cpu()
        )
    return metrics


@dataclass
class TrainResult:
    models: dict[str, nn.Module]
    history: list[dict[str, float]]


def train_models(
    *,
    mixture: TangentGaussianMixture,
    architecture: str,
    hidden: int,
    depth: int,
    time_dim: int,
    steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    loss_space: str,
    t_min: float,
    t_max: float,
    conversion_clip: float,
    log_every: int,
    validation_samples: int,
    seed: int,
    device: torch.device,
) -> TrainResult:
    models = build_same_init_models(
        architecture,
        D=mixture.D,
        hidden=hidden,
        depth=depth,
        time_dim=time_dim,
        device=device,
        seed=stable_seed(seed, 101),
    )
    optimizers = {
        key: torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        for key, model in models.items()
    }
    generator = torch.Generator(device=device.type)
    generator.manual_seed(stable_seed(seed, 211))
    val_generator = torch.Generator(device=device.type)
    val_generator.manual_seed(stable_seed(seed, 223))
    val_x, val_eps, val_t, val_x_t, _ = mixture.noised_batch(
        validation_samples,
        t_min=t_min,
        t_max=t_max,
        generator=val_generator,
    )
    with torch.inference_mode():
        val_bayes = mixture.posterior_clean(val_x_t, val_t)

    history: list[dict[str, float]] = []
    for step in range(1, steps + 1):
        x, eps, t, x_t, _ = mixture.noised_batch(
            batch_size, t_min=t_min, t_max=t_max, generator=generator
        )
        train_losses = {}
        for target, model in models.items():
            model.train()
            optimizers[target].zero_grad(set_to_none=True)
            output = model(x_t, t)
            loss = loss_for_output(
                output,
                x_t=x_t,
                t=t,
                x=x,
                eps=eps,
                target=target,
                loss_space=loss_space,
                conversion_clip=conversion_clip,
            )
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizers[target].step()
            train_losses[target] = float(loss.detach().cpu())

        if step == 1 or step % log_every == 0 or step == steps:
            for model in models.values():
                model.eval()
            val = validation_metrics(
                models,
                x=val_x,
                eps=val_eps,
                t=val_t,
                x_t=val_x_t,
                bayes=val_bayes,
                conversion_clip=conversion_clip,
            )
            row = {
                "step": step,
                **{f"train_{key}_loss": value for key, value in train_losses.items()},
                **val,
            }
            history.append(row)
            print(
                f"[{architecture} H={hidden}] {step}/{steps} "
                + " ".join(
                    f"{key}:loss={train_losses[key]:.4g},"
                    f"excess={val[f'{key}_excess_mse']:.4g}"
                    for key in ("x", "v", "eps")
                ),
                flush=True,
            )
    return TrainResult(models=models, history=history)


@torch.inference_mode()
def evaluate_teacher(
    *,
    models: dict[str, nn.Module],
    mixture: TangentGaussianMixture,
    eval_times: Sequence[float],
    samples: int,
    batch_size: int,
    conversion_clip: float,
    gammas: Sequence[float],
    seed: int,
    setting: dict,
) -> list[dict[str, float]]:
    rows = []
    generator = torch.Generator(device=mixture.device.type)
    generator.manual_seed(stable_seed(seed, 307))
    for time in eval_times:
        sums: dict[str, float] = {}
        total = 0
        for start in range(0, samples, batch_size):
            n = min(batch_size, samples - start)
            x, component = mixture.sample_clean(n, generator=generator)
            eps = torch.randn(x.shape, device=mixture.device, generator=generator)
            t = torch.full((n,), float(time), device=mixture.device)
            x_t = (1.0 - t[:, None]) * x + t[:, None] * eps
            bayes = mixture.posterior_clean(x_t, t)
            clean = {
                target: clean_from_output(
                    model(x_t, t), x_t, t, target, conversion_clip
                )
                for target, model in models.items()
            }
            bayes_residual = bayes - clean["x"]
            paired_residual = x - clean["x"]
            gap = clean["x"] - clean["v"]
            gap_tangent, gap_normal = mixture.split_by_component(gap, component)
            bayes_tangent, bayes_normal = mixture.split_by_component(
                bayes_residual, component
            )
            gap_energy = gap.double().square().sum(dim=1).clamp_min(EPS)
            bayes_risk = (x - bayes).square().mean(dim=1)
            metrics: dict[str, torch.Tensor] = {
                "bayes_risk_mse": bayes_risk,
                "gap_xv_rms": row_rms(gap),
                "gap_xv_tangent_rms": row_rms(gap_tangent),
                "gap_xv_normal_rms": row_rms(gap_normal),
                "gap_xv_normal_energy_fraction": (
                    gap_normal.double().square().sum(dim=1) / gap_energy
                ).float(),
                "cos_xv_bayes_residual": row_cosine(gap, bayes_residual),
                "cos_xv_paired_residual": row_cosine(gap, paired_residual),
                "bayes_residual_tangent_rms": row_rms(bayes_tangent),
                "bayes_residual_normal_rms": row_rms(bayes_normal),
            }
            gamma_star = (
                (gap.double() * bayes_residual.double()).sum(dim=1)
                / gap_energy
            )
            metrics["gamma_star_bayes"] = gamma_star.float()
            metrics["positive_gamma_star_bayes_fraction"] = (
                gamma_star > 0
            ).float()
            for target in ("x", "v", "eps"):
                error = clean[target] - bayes
                tangent, normal = mixture.split_by_component(error, component)
                paired = (clean[target] - x).square().mean(dim=1)
                excess = error.square().mean(dim=1)
                metrics[f"{target}_paired_mse"] = paired
                metrics[f"{target}_excess_mse"] = excess
                metrics[f"{target}_excess_tangent_mse"] = tangent.square().mean(dim=1)
                metrics[f"{target}_excess_normal_mse"] = normal.square().mean(dim=1)
                metrics[f"{target}_risk_closure"] = paired - bayes_risk - excess
            for gamma in gammas:
                candidate = clean["x"] + float(gamma) * gap
                metrics[f"xv_g{tag_float(gamma)}_bayes_excess_mse"] = (
                    candidate - bayes
                ).square().mean(dim=1)

            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + float(value.double().sum().cpu())
            total += n
        rows.append(
            {
                **setting,
                "time": float(time),
                "samples": total,
                **{key: value / total for key, value in sums.items()},
            }
        )
    return rows


def prediction_clean(
    models: dict[str, nn.Module],
    state: torch.Tensor,
    t: torch.Tensor,
    target: str,
    clip: float,
) -> torch.Tensor:
    return clean_from_output(models[target](state, t), state, t, target, clip)


@torch.inference_mode()
def guided_clean(
    *,
    models: dict[str, nn.Module],
    mixture: TangentGaussianMixture,
    state: torch.Tensor,
    t: torch.Tensor,
    kind: str,
    strength: float,
    clip: float,
) -> torch.Tensor:
    if kind == "bayes":
        return mixture.posterior_clean(state, t)
    if kind in {"x", "v", "eps"}:
        return prediction_clean(models, state, t, kind, clip)
    clean_x = prediction_clean(models, state, t, "x", clip)
    clean_v = prediction_clean(models, state, t, "v", clip)
    gap = clean_x - clean_v
    if kind == "xv":
        chosen = gap
    elif kind in {"xv_tangent", "xv_normal"}:
        component = mixture.nearest_components(clean_x)
        tangent, normal = mixture.split_by_component(gap, component)
        chosen = tangent if kind == "xv_tangent" else normal
    else:
        raise ValueError(kind)
    return clean_x + float(strength) * chosen


@torch.inference_mode()
def sample_condition(
    *,
    models: dict[str, nn.Module],
    mixture: TangentGaussianMixture,
    kind: str,
    strength: float,
    sample_count: int,
    batch_size: int,
    steps: int,
    t_max: float,
    t_min: float,
    clip: float,
    seed: int,
) -> np.ndarray:
    """Roll out from an exact p_t_max marginal shared by every condition."""
    outputs = []
    grid = torch.linspace(t_max, t_min, steps + 1, device=mixture.device)
    for start in range(0, sample_count, batch_size):
        n = min(batch_size, sample_count - start)
        generator = torch.Generator(device=mixture.device.type)
        generator.manual_seed(seed + start)
        clean_start, _ = mixture.sample_clean(n, generator=generator)
        eps = torch.randn(
            clean_start.shape, device=mixture.device, generator=generator
        )
        state = (1.0 - float(t_max)) * clean_start + float(t_max) * eps
        for index in range(steps):
            t_now = grid[index]
            t_next = grid[index + 1]
            t = torch.full((n,), float(t_now), device=mixture.device)
            x_hat = guided_clean(
                models=models,
                mixture=mixture,
                state=state,
                t=t,
                kind=kind,
                strength=strength,
                clip=clip,
            )
            velocity = (state - x_hat) / t[:, None].clamp_min(clip)
            state = state + (t_next - t_now) * velocity
        t = torch.full((n,), float(grid[-1]), device=mixture.device)
        final = guided_clean(
            models=models,
            mixture=mixture,
            state=state,
            t=t,
            kind=kind,
            strength=strength,
            clip=clip,
        )
        outputs.append(final.float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


class FrozenPushforward:
    """Fixed anisotropic nonlinear observation map used as a decoder proxy."""

    def __init__(self, D: int, width: int, seed: int) -> None:
        width = min(int(width), int(D))
        rng = np.random.default_rng(seed)
        raw = rng.standard_normal((D, width))
        q, _ = np.linalg.qr(raw)
        self.matrix = q[:, :width].astype(np.float64)
        self.gains = np.exp(np.linspace(math.log(0.5), math.log(2.0), width))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h = (x.astype(np.float64) @ self.matrix) * self.gains[None]
        return np.concatenate([h, np.tanh(h), np.sin(0.5 * h)], axis=1)


def projection_directions(dim: int, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    directions = rng.standard_normal((dim, count))
    directions /= np.linalg.norm(directions, axis=0, keepdims=True).clip(1e-12)
    return directions


def projected_swd(a: np.ndarray, b: np.ndarray, directions: np.ndarray) -> float:
    if len(a) != len(b):
        raise ValueError("SWD inputs must have equal sample counts")
    pa = np.sort(a.astype(np.float64) @ directions, axis=0)
    pb = np.sort(b.astype(np.float64) @ directions, axis=0)
    return float(np.mean(np.abs(pa - pb)))


def make_rff(
    reference: np.ndarray, *, features: int, seed: int
) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    n = min(len(reference), 512)
    subset = reference[rng.choice(len(reference), size=n, replace=False)].astype(np.float64)
    left = subset[: n // 2]
    right = subset[n // 2 : n // 2 + len(left)]
    bandwidth = float(np.median(np.linalg.norm(left - right, axis=1)))
    bandwidth = max(bandwidth, 1e-6)
    weight = rng.standard_normal((reference.shape[1], features)) / bandwidth
    phase = rng.uniform(0.0, 2.0 * math.pi, size=(features,))
    return weight, phase, bandwidth


def rff_mmd_squared(
    a: np.ndarray,
    b: np.ndarray,
    weight: np.ndarray,
    phase: np.ndarray,
) -> float:
    scale = math.sqrt(2.0 / len(phase))
    mean_a = scale * np.cos(a.astype(np.float64) @ weight + phase).mean(axis=0)
    mean_b = scale * np.cos(b.astype(np.float64) @ weight + phase).mean(axis=0)
    return float(np.square(mean_a - mean_b).sum())


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    p = p.astype(np.float64) / max(float(p.sum()), EPS)
    q = q.astype(np.float64) / max(float(q.sum()), EPS)
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask].clip(EPS))))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


@torch.inference_mode()
def component_histogram(
    mixture: TangentGaussianMixture, samples: np.ndarray
) -> np.ndarray:
    tensor = torch.from_numpy(samples).to(mixture.device)
    component = mixture.nearest_components(tensor)
    return torch.bincount(component, minlength=mixture.components).cpu().numpy()


def evaluate_generation(
    *,
    samples: dict[str, np.ndarray],
    conditions: Sequence[tuple[str, str, float]],
    reference: np.ndarray,
    mixture: TangentGaussianMixture,
    pushforward: FrozenPushforward,
    projections: int,
    rff_features: int,
    seed: int,
    setting: dict,
) -> list[dict[str, float]]:
    latent_directions = projection_directions(
        mixture.D, projections, stable_seed(seed, 401)
    )
    ref_observed = pushforward(reference)
    observed_directions = projection_directions(
        ref_observed.shape[1], projections, stable_seed(seed, 409)
    )
    latent_rff = make_rff(
        reference, features=rff_features, seed=stable_seed(seed, 419)
    )
    observed_rff = make_rff(
        ref_observed, features=rff_features, seed=stable_seed(seed, 421)
    )
    ref_hist = component_histogram(mixture, reference)
    oracle = samples["bayes"]
    ref_tensor = torch.from_numpy(reference).to(mixture.device)
    ref_geometry = mixture.nearest_patch_geometry(ref_tensor)
    rows = []
    for name, kind, strength in conditions:
        sample = samples[name]
        observed = pushforward(sample)
        geometry = mixture.nearest_patch_geometry(
            torch.from_numpy(sample).to(mixture.device)
        )
        hist = component_histogram(mixture, sample)
        row = {
            **setting,
            "condition": name,
            "kind": kind,
            "strength": float(strength),
            "latent_swd": projected_swd(reference, sample, latent_directions),
            "latent_mmd_rff": rff_mmd_squared(
                reference, sample, latent_rff[0], latent_rff[1]
            ),
            "pushforward_swd": projected_swd(
                ref_observed, observed, observed_directions
            ),
            "pushforward_mmd_rff": rff_mmd_squared(
                ref_observed, observed, observed_rff[0], observed_rff[1]
            ),
            "component_jsd": jensen_shannon(ref_hist, hist),
            "endpoint_rms_to_bayes_rollout": float(
                np.sqrt(np.mean(np.square(sample.astype(np.float64) - oracle)))
            ),
            **geometry,
            "reference_nearest_tangent_rms": ref_geometry["nearest_tangent_rms"],
            "reference_nearest_normal_rms": ref_geometry["nearest_normal_rms"],
            "latent_rff_bandwidth": latent_rff[2],
            "pushforward_rff_bandwidth": observed_rff[2],
        }
        rows.append(row)
    return rows


def plot_setting(
    path: Path,
    teacher: list[dict[str, float]],
    generation: list[dict[str, float]],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    times = [row["time"] for row in teacher]
    for target in ("x", "v", "eps"):
        axes[0].plot(
            times,
            [row[f"{target}_excess_mse"] for row in teacher],
            marker="o",
            label=target,
        )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("Bayes excess MSE")
    axes[0].set_title("Teacher-forced Bayes excess")
    axes[0].legend()

    labels = [row["condition"] for row in generation]
    x = np.arange(len(labels))
    axes[1].bar(x, [row["latent_swd"] for row in generation])
    axes[1].set_xticks(x, labels, rotation=55, ha="right")
    axes[1].set_ylabel("SWD")
    axes[1].set_title("Ambient latent distribution")
    axes[2].bar(x, [row["pushforward_swd"] for row in generation])
    axes[2].set_xticks(x, labels, rotation=55, ha="right")
    axes[2].set_ylabel("SWD")
    axes[2].set_title("Fixed nonlinear pushforward")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_jit_style(
    path: Path,
    *,
    reference: np.ndarray,
    samples: dict[str, np.ndarray],
    mixture: TangentGaussianMixture,
    title: str,
    limit: int = 3500,
) -> None:
    names = ("reference", "bayes", "x", "eps", "v")
    labels = ("Reference", "Bayes oracle", "x-pred", "eps-pred", "v-pred")
    arrays = {"reference": reference, **{key: samples[key] for key in names[1:]}}
    projected = {}
    for name, values in arrays.items():
        tensor = torch.from_numpy(values[:limit]).to(mixture.device)
        projected[name] = mixture.intrinsic_readout(tensor).float().cpu().numpy()
    low = np.quantile(projected["reference"], 0.005, axis=0)
    high = np.quantile(projected["reference"], 0.995, axis=0)
    center = 0.5 * (low + high)
    radius = 0.62 * float(np.max(high - low))
    radius = max(radius, 0.25)

    fig, axes = plt.subplots(1, len(names), figsize=(17.5, 3.8), sharex=True, sharey=True)
    reference_points = projected["reference"]
    for axis, name, label in zip(axes, names, labels):
        if name != "reference":
            axis.scatter(
                reference_points[:, 0],
                reference_points[:, 1],
                s=3,
                alpha=0.10,
                color="#555555",
                linewidths=0,
                rasterized=True,
            )
        points = projected[name]
        axis.scatter(
            points[:, 0],
            points[:, 1],
            s=4,
            alpha=0.42 if name == "reference" else 0.50,
            color="#2878b5" if name == "reference" else "#d95f02",
            linewidths=0,
            rasterized=True,
        )
        axis.set_title(label, fontsize=12)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_xticks([])
        axis.set_yticks([])
        outside = np.mean(
            (np.abs(points[:, 0] - center[0]) > radius)
            | (np.abs(points[:, 1] - center[1]) > radius)
        )
        if outside > 0.005:
            axis.text(
                0.03,
                0.96,
                f"outside: {100.0 * outside:.1f}%",
                transform=axis.transAxes,
                va="top",
                fontsize=8,
                color="#9c2f00",
            )
        for spine in axis.spines.values():
            spine.set_color("#b8b8b8")
            spine.set_linewidth(0.8)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def history_plateau_ratio(history: list[dict[str, float]], target: str) -> float:
    if len(history) < 3:
        return float("nan")
    values = np.array([row[f"{target}_excess_mse"] for row in history], dtype=float)
    tail = values[max(0, len(values) // 2) :]
    if len(tail) < 2:
        return float("nan")
    return float((tail[-1] - tail[0]) / max(abs(tail[0]), EPS))


def run_setting(
    *,
    args: argparse.Namespace,
    mixture: TangentGaussianMixture,
    architecture: str,
    hidden: int,
    seed: int,
    output_dir: Path,
    device: torch.device,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    complete = output_dir / "complete.json"
    if args.resume and complete.is_file():
        print(f"[resume] {output_dir} complete", flush=True)
        return
    setting = {
        "seed": seed,
        "D": mixture.D,
        "components": mixture.components,
        "sigma_tangent": mixture.sigma_tangent,
        "sigma_normal": mixture.sigma_normal,
        "architecture": architecture,
        "hidden": hidden,
        "loss_space": args.loss_space,
    }
    train = train_models(
        mixture=mixture,
        architecture=architecture,
        hidden=hidden,
        depth=args.depth,
        time_dim=args.time_dim,
        steps=args.train_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        loss_space=args.loss_space,
        t_min=args.t_min,
        t_max=args.t_max,
        conversion_clip=args.conversion_clip,
        log_every=args.log_every,
        validation_samples=args.validation_samples,
        seed=seed,
        device=device,
    )
    save_csv(output_dir / "train_history.csv", train.history)
    teacher = evaluate_teacher(
        models=train.models,
        mixture=mixture,
        eval_times=args.eval_times,
        samples=args.teacher_samples,
        batch_size=args.eval_batch_size,
        conversion_clip=args.conversion_clip,
        gammas=args.gammas,
        seed=stable_seed(seed, 503),
        setting=setting,
    )
    save_csv(output_dir / "teacher_metrics.csv", teacher)

    conditions: list[tuple[str, str, float]] = [
        ("bayes", "bayes", 0.0),
        ("x", "x", 0.0),
        ("v", "v", 0.0),
        ("eps", "eps", 0.0),
    ]
    for gamma in args.gammas:
        conditions.append((f"xv_g{tag_float(gamma)}", "xv", float(gamma)))
    for gamma in args.geometry_gammas:
        conditions.extend(
            [
                (
                    f"xv_tangent_g{tag_float(gamma)}",
                    "xv_tangent",
                    float(gamma),
                ),
                (
                    f"xv_normal_g{tag_float(gamma)}",
                    "xv_normal",
                    float(gamma),
                ),
            ]
        )
    sample_seed = stable_seed(seed, mixture.D, 601)
    samples = {}
    for name, kind, strength in conditions:
        print(f"[sample] {architecture} H={hidden} {name}", flush=True)
        samples[name] = sample_condition(
            models=train.models,
            mixture=mixture,
            kind=kind,
            strength=strength,
            sample_count=args.sample_count,
            batch_size=args.sample_batch_size,
            steps=args.sample_steps,
            t_max=args.sample_t_max,
            t_min=args.sample_t_min,
            clip=args.conversion_clip,
            seed=sample_seed,
        )
        if args.save_samples:
            np.save(output_dir / f"samples_{name}.npy", samples[name])

    reference_generator = torch.Generator(device=device.type)
    reference_generator.manual_seed(stable_seed(seed, 607))
    reference, _ = mixture.sample_clean(
        args.sample_count, generator=reference_generator
    )
    reference_np = reference.float().cpu().numpy()
    jit_projection = {
        "reference": mixture.intrinsic_readout(reference).float().cpu().numpy(),
        **{
            key: mixture.intrinsic_readout(
                torch.from_numpy(samples[key]).to(device)
            ).float().cpu().numpy()
            for key in ("bayes", "x", "eps", "v")
        },
    }
    np.savez_compressed(output_dir / "jit_projection.npz", **jit_projection)
    plot_jit_style(
        output_dir / "jit_style.png",
        reference=reference_np,
        samples=samples,
        mixture=mixture,
        title=(
            f"Exact-Bayes toy: {architecture}, H={hidden}, D={mixture.D}, "
            f"sigma_normal={mixture.sigma_normal:g}"
        ),
    )
    pushforward = FrozenPushforward(
        mixture.D, args.pushforward_width, args.pushforward_seed
    )
    generation = evaluate_generation(
        samples=samples,
        conditions=conditions,
        reference=reference_np,
        mixture=mixture,
        pushforward=pushforward,
        projections=args.swd_projections,
        rff_features=args.rff_features,
        seed=stable_seed(seed, 613),
        setting=setting,
    )
    save_csv(output_dir / "generation_metrics.csv", generation)

    generation_by_name = {row["condition"]: row for row in generation}
    mean_teacher = {
        key: float(np.mean([row[key] for row in teacher]))
        for key in (
            "bayes_risk_mse",
            "x_excess_mse",
            "v_excess_mse",
            "eps_excess_mse",
            "gap_xv_normal_energy_fraction",
            "cos_xv_bayes_residual",
            "gamma_star_bayes",
        )
    }
    summary = {
        **setting,
        "parameters_per_model": sum(
            parameter.numel() for parameter in train.models["x"].parameters()
        ),
        **{f"mean_{key}": value for key, value in mean_teacher.items()},
        "x_excess_over_bayes_risk": mean_teacher["x_excess_mse"]
        / max(mean_teacher["bayes_risk_mse"], EPS),
        "v_excess_over_bayes_risk": mean_teacher["v_excess_mse"]
        / max(mean_teacher["bayes_risk_mse"], EPS),
        "x_plateau_relative_change": history_plateau_ratio(train.history, "x"),
        "v_plateau_relative_change": history_plateau_ratio(train.history, "v"),
        "x_latent_swd": generation_by_name["x"]["latent_swd"],
        "v_latent_swd": generation_by_name["v"]["latent_swd"],
        "bayes_latent_swd": generation_by_name["bayes"]["latent_swd"],
        "x_pushforward_swd": generation_by_name["x"]["pushforward_swd"],
        "v_pushforward_swd": generation_by_name["v"]["pushforward_swd"],
        "bayes_pushforward_swd": generation_by_name["bayes"]["pushforward_swd"],
        "x_better_than_v_latent_swd": bool(
            generation_by_name["x"]["latent_swd"]
            < generation_by_name["v"]["latent_swd"]
        ),
        "x_better_than_v_pushforward_swd": bool(
            generation_by_name["x"]["pushforward_swd"]
            < generation_by_name["v"]["pushforward_swd"]
        ),
    }
    save_json(output_dir / "setting_summary.json", summary)
    plot_setting(output_dir / "diagnostic.png", teacher, generation)
    if args.save_checkpoints:
        for target, model in train.models.items():
            torch.save(model.state_dict(), output_dir / f"model_{target}.pt")
    save_json(complete, {"status": "complete", "summary": summary})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--D", type=int, default=64)
    parser.add_argument("--components", type=int, default=32)
    parser.add_argument("--curvature", type=float, default=0.5)
    parser.add_argument("--frequency-scale", type=float, default=6.0)
    parser.add_argument("--center-rms", type=float, default=0.8)
    parser.add_argument("--sigma-tangent", type=float, default=0.35)
    parser.add_argument("--sigma-normal", type=float, default=0.03)
    parser.add_argument("--mixture-seed", type=int, default=1701)
    parser.add_argument(
        "--architectures",
        type=parse_str_list,
        default=parse_str_list("jit_relu,plain,residual,residual_skip"),
    )
    parser.add_argument(
        "--hidden-dims", type=parse_int_list, default=parse_int_list("64,128")
    )
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--time-dim", type=int, default=32)
    parser.add_argument("--train-steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--loss-space", choices=("v", "direct"), default="v")
    parser.add_argument("--t-min", type=float, default=0.02)
    parser.add_argument("--t-max", type=float, default=0.98)
    parser.add_argument("--conversion-clip", type=float, default=0.02)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--validation-samples", type=int, default=4096)
    parser.add_argument(
        "--eval-times",
        type=parse_float_list,
        default=parse_float_list("0.1,0.3,0.5,0.7,0.9"),
    )
    parser.add_argument("--teacher-samples", type=int, default=8192)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument(
        "--gammas",
        type=parse_float_list,
        default=parse_float_list("-0.1,-0.03,-0.01,0.01,0.03,0.1"),
    )
    parser.add_argument(
        "--geometry-gammas",
        type=parse_float_list,
        default=parse_float_list("-0.03,0.03"),
    )
    parser.add_argument("--sample-count", type=int, default=5000)
    parser.add_argument("--sample-batch-size", type=int, default=1000)
    parser.add_argument("--sample-steps", type=int, default=100)
    parser.add_argument("--sample-t-max", type=float, default=0.98)
    parser.add_argument("--sample-t-min", type=float, default=0.02)
    parser.add_argument("--swd-projections", type=int, default=256)
    parser.add_argument("--rff-features", type=int, default=1024)
    parser.add_argument("--pushforward-width", type=int, default=24)
    parser.add_argument("--pushforward-seed", type=int, default=1801)
    parser.add_argument(
        "--seeds", type=parse_int_list, default=parse_int_list("20260821")
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-samples", action="store_true")
    parser.add_argument("--save-checkpoints", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.D < 3 or args.components < 2:
        raise ValueError("invalid distribution dimensions")
    if any(
        name not in {"jit_relu", "plain", "residual", "residual_skip"}
        for name in args.architectures
    ):
        raise ValueError(f"unknown architectures: {args.architectures}")
    if any(hidden <= 0 for hidden in args.hidden_dims):
        raise ValueError("hidden dimensions must be positive")
    device = torch.device(args.device)
    args.output_root.mkdir(parents=True, exist_ok=True)
    save_json(
        args.output_root / "manifest.json",
        {
            **vars(args),
            "output_root": str(args.output_root),
            "device": str(device),
            "oracle": "analytic E[x|x_t,t] for tangent Gaussian mixture",
            "normal_interpretation": (
                "geometric component only; quality is evaluated separately in "
                "latent and pushforward distributions"
            ),
        },
    )
    for seed in args.seeds:
        set_seed(seed)
        mixture = TangentGaussianMixture(
            D=args.D,
            components=args.components,
            curvature=args.curvature,
            frequency_scale=args.frequency_scale,
            center_rms=args.center_rms,
            sigma_tangent=args.sigma_tangent,
            sigma_normal=args.sigma_normal,
            seed=args.mixture_seed,
            device=device,
        )
        for architecture in args.architectures:
            for hidden in args.hidden_dims:
                output_dir = (
                    args.output_root
                    / f"seed{seed}"
                    / architecture
                    / f"H{hidden}"
                )
                run_setting(
                    args=args,
                    mixture=mixture,
                    architecture=architecture,
                    hidden=hidden,
                    seed=seed,
                    output_dir=output_dir,
                    device=device,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
