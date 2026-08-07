#!/usr/bin/env python3
"""Prediction-target extrapolation toy experiment (v3).

Purpose
-------
A self-contained synthetic experiment inspired by the high-dimensional manifold
setup discussed in JiT. A 2-D spiral is embedded into a D-dimensional ambient
space with a fixed column-orthonormal matrix P. Three identical MLPs are trained
on the same batches/noise/times, differing only in what they directly predict:

    x-prediction       : network output is clean x
    v-prediction       : network output is v = eps - x
    epsilon-prediction : network output is eps

All three are optimized with the SAME v-space loss so the comparison isolates
direct prediction space as much as possible. For x_t = (1-t)x + t eps,
outputs are converted to the common velocity target v = eps - x as

    v_from_x   = (x_t - x_hat) / t
    v_from_v   = v_hat
    v_from_eps = (eps_hat - x_t) / (1-t)

The script then:
  1. measures denoising quality and exact off-subspace error;
  2. checks whether clean-equivalent estimates form an ordered axis
         x_hat^eps -> x_hat^v -> x_hat^x;
  3. samples each baseline with an Euler probability-flow/rectified-flow ODE;
  4. samples extrapolated estimators
         x_ext = x_hat^x + gamma (x_hat^x - x_hat^v)
         x_ext = x_hat^x + gamma (x_hat^x - x_hat^eps)
     and an optional three-point Richardson-style extrapolation;
  5. saves metrics and figures for each ambient dimension.

This is a mechanism toy, not a claim that the synthetic setup exactly reproduces
JiT's unpublished/implementation-specific details. Future revisions should use a
new versioned filename instead of overwriting this file.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-8


def parse_int_list(text: str) -> list[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def parse_float_list(text: str) -> list[float]:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one float")
    return values


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def orthonormal_embedding(D: int, d: int, seed: int, device: torch.device) -> torch.Tensor:
    if D < d:
        raise ValueError("ambient D must be >= intrinsic d")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    matrix = torch.randn(D, d, generator=generator, dtype=torch.float64)
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    return q.to(device=device, dtype=torch.float32)


def sample_spiral_2d(
    n: int,
    *,
    device: torch.device,
    jitter: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Two-turn spiral with small intrinsic noise, roughly unit scale."""
    u = torch.rand(n, device=device, generator=generator)
    theta = 4.0 * math.pi * u
    radius = 0.15 + 0.85 * u
    x = radius * torch.cos(theta)
    y = radius * torch.sin(theta)
    points = torch.stack([x, y], dim=1)
    if jitter > 0:
        points = points + jitter * torch.randn(
            points.shape, device=device, generator=generator
        )
    # Fixed scaling: keep typical ambient signal variance O(1).
    return points * 1.6


def embed(points_2d: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    return points_2d @ P.T


def project_to_intrinsic(points_D: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    return points_D @ P


def project_to_subspace(points_D: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    return project_to_intrinsic(points_D, P) @ P.T


def off_subspace_rms(points_D: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    residual = points_D - project_to_subspace(points_D, P)
    return residual.square().mean(dim=1).sqrt()


def row_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a.double(), b.double(), dim=1, eps=EPS).float()


def clean_from_output(
    output: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
    target: str,
    clip: float,
) -> torch.Tensor:
    t_col = t[:, None]
    if target == "x":
        return output
    if target == "v":
        return x_t - t_col * output
    if target == "eps":
        denom = (1.0 - t_col).clamp_min(clip)
        return (x_t - t_col * output) / denom
    raise ValueError(target)


def velocity_from_output(
    output: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
    target: str,
    clip: float,
) -> torch.Tensor:
    t_col = t[:, None]
    if target == "x":
        return (x_t - output) / t_col.clamp_min(clip)
    if target == "v":
        return output
    if target == "eps":
        return (output - x_t) / (1.0 - t_col).clamp_min(clip)
    raise ValueError(target)


def direct_training_target(x: torch.Tensor, eps: torch.Tensor, target: str) -> torch.Tensor:
    if target == "x":
        return x
    if target == "v":
        return eps - x
    if target == "eps":
        return eps
    raise ValueError(target)


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int = 32, max_freq: float = 32.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("time embedding dim must be even")
        half = dim // 2
        freqs = torch.exp(torch.linspace(0.0, math.log(max_freq), half))
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        phase = 2.0 * math.pi * t[:, None] * self.freqs[None]
        return torch.cat([torch.sin(phase), torch.cos(phase)], dim=1)


class DenoiseMLP(nn.Module):
    def __init__(self, ambient_dim: int, hidden_dim: int, depth: int, time_dim: int):
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be >= 2")
        self.time = TimeEmbedding(time_dim)
        layers: list[nn.Module] = []
        in_dim = ambient_dim + time_dim
        for _ in range(depth - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, ambient_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x_t, self.time(t)], dim=1))


@dataclass
class TrainResult:
    models: dict[str, DenoiseMLP]
    history: list[dict[str, float]]


def build_models(
    D: int,
    hidden: int,
    depth: int,
    time_dim: int,
    device: torch.device,
    seed: int,
) -> dict[str, DenoiseMLP]:
    torch.manual_seed(seed)
    base = DenoiseMLP(D, hidden, depth, time_dim).to(device)
    state = copy.deepcopy(base.state_dict())
    models: dict[str, DenoiseMLP] = {}
    for target in ("x", "v", "eps"):
        model = DenoiseMLP(D, hidden, depth, time_dim).to(device)
        model.load_state_dict(state)
        models[target] = model
    return models


def train_models(
    *,
    D: int,
    P: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> TrainResult:
    models = build_models(D, args.hidden_dim, args.depth, args.time_dim, device, args.seed + D)
    optimizers = {
        key: torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for key, model in models.items()
    }
    generator = torch.Generator(device=device.type)
    generator.manual_seed(args.seed * 1000 + D)
    history: list[dict[str, float]] = []

    for step in range(1, args.train_steps + 1):
        x2 = sample_spiral_2d(
            args.batch_size, device=device, jitter=args.data_jitter, generator=generator
        )
        x = embed(x2, P)
        eps = torch.randn(x.shape, device=device, generator=generator)
        t = torch.empty(args.batch_size, device=device).uniform_(
            args.t_min, args.t_max, generator=generator
        )
        x_t = (1.0 - t[:, None]) * x + t[:, None] * eps
        true_v = eps - x

        losses: dict[str, float] = {}
        for target, model in models.items():
            optimizers[target].zero_grad(set_to_none=True)
            output = model(x_t, t)
            if args.loss_space == "v":
                pred_for_loss = velocity_from_output(
                    output, x_t, t, target, args.conversion_clip
                )
                loss = F.mse_loss(pred_for_loss, true_v)
            elif args.loss_space == "direct":
                loss = F.mse_loss(output, direct_training_target(x, eps, target))
            else:
                raise ValueError(args.loss_space)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizers[target].step()
            losses[target] = float(loss.detach().cpu())

        if step == 1 or step % args.log_every == 0 or step == args.train_steps:
            row = {"step": float(step), **{f"loss_{k}": v for k, v in losses.items()}}
            history.append(row)
            print(
                f"[D={D}] step {step}/{args.train_steps} "
                + " ".join(f"{k}={losses[k]:.5g}" for k in ("x", "v", "eps")),
                flush=True,
            )

    return TrainResult(models=models, history=history)


@torch.inference_mode()
def evaluate_teacher_forced(
    *,
    models: dict[str, DenoiseMLP],
    D: int,
    P: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, float]], dict[str, np.ndarray]]:
    generator = torch.Generator(device=device.type)
    generator.manual_seed(args.seed * 2000 + D)
    rows: list[dict[str, float]] = []
    cached: dict[str, list[np.ndarray]] = {"t": [], "cos_ev_vx": [], "rho": []}

    for time in args.eval_times:
        total = args.eval_samples
        sums: dict[str, float] = {}
        count = 0
        for start in range(0, total, args.eval_batch_size):
            n = min(args.eval_batch_size, total - start)
            x2 = sample_spiral_2d(n, device=device, jitter=args.data_jitter, generator=generator)
            x = embed(x2, P)
            eps = torch.randn(x.shape, device=device, generator=generator)
            t = torch.full((n,), float(time), device=device)
            x_t = (1.0 - t[:, None]) * x + t[:, None] * eps
            clean: dict[str, torch.Tensor] = {}
            for target, model in models.items():
                output = model(x_t, t)
                clean[target] = clean_from_output(
                    output, x_t, t, target, args.conversion_clip
                )

            d_ev = clean["v"] - clean["eps"]
            d_vx = clean["x"] - clean["v"]
            cos_axis = row_cosine(d_ev, d_vx)
            rho = (
                (d_ev.double() * d_vx.double()).sum(dim=1)
                / d_ev.double().square().sum(dim=1).clamp_min(EPS)
            ).float()
            residual_x = x - clean["x"]
            cos_xv_residual = row_cosine(clean["x"] - clean["v"], residual_x)
            cos_xeps_residual = row_cosine(clean["x"] - clean["eps"], residual_x)

            metrics: dict[str, torch.Tensor] = {
                "cos_ev_vx": cos_axis,
                "rho": rho,
                "cos_xv_to_true_residual": cos_xv_residual,
                "cos_xeps_to_true_residual": cos_xeps_residual,
            }
            for target in ("x", "v", "eps"):
                error = clean[target] - x
                metrics[f"mse_{target}"] = error.square().mean(dim=1)
                metrics[f"off_subspace_rms_{target}"] = off_subspace_rms(clean[target], P)
                intrinsic_error = project_to_intrinsic(clean[target] - x, P)
                metrics[f"intrinsic_mse_{target}"] = intrinsic_error.square().mean(dim=1)

            # Fixed extrapolation diagnostics, teacher-forced only.
            for gamma in args.gammas:
                xv = clean["x"] + gamma * (clean["x"] - clean["v"])
                xe = clean["x"] + gamma * (clean["x"] - clean["eps"])
                metrics[f"mse_xv_g{gamma:g}"] = (xv - x).square().mean(dim=1)
                metrics[f"mse_xeps_g{gamma:g}"] = (xe - x).square().mean(dim=1)

            # Richardson-like estimate when local geometric ordering is plausible.
            rho_clip = rho.clamp(args.richardson_rho_min, args.richardson_rho_max)
            rich = clean["x"] + (rho_clip / (1.0 - rho_clip))[:, None] * d_vx
            metrics["mse_richardson"] = (rich - x).square().mean(dim=1)

            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + float(value.double().sum().cpu())
            count += n
            cached["t"].append(np.full(n, time, dtype=np.float32))
            cached["cos_ev_vx"].append(cos_axis.cpu().numpy())
            cached["rho"].append(rho.cpu().numpy())

        row: dict[str, float] = {"D": float(D), "time": float(time), "samples": float(count)}
        row.update({key: val / count for key, val in sums.items()})
        rows.append(row)

    return rows, {k: np.concatenate(v) if v else np.empty(0) for k, v in cached.items()}


@torch.inference_mode()
def velocity_for_condition(
    *,
    models: dict[str, DenoiseMLP],
    state: torch.Tensor,
    t: torch.Tensor,
    condition: str,
    gamma: float,
    args: argparse.Namespace,
) -> torch.Tensor:
    if condition in ("x", "v", "eps"):
        output = models[condition](state, t)
        return velocity_from_output(output, state, t, condition, args.conversion_clip)

    outputs = {target: models[target](state, t) for target in ("x", "v", "eps")}
    clean = {
        target: clean_from_output(outputs[target], state, t, target, args.conversion_clip)
        for target in outputs
    }
    if condition == "xv":
        guided_x = clean["x"] + gamma * (clean["x"] - clean["v"])
    elif condition == "xeps":
        guided_x = clean["x"] + gamma * (clean["x"] - clean["eps"])
    elif condition == "richardson":
        d_ev = clean["v"] - clean["eps"]
        d_vx = clean["x"] - clean["v"]
        rho = (
            (d_ev.double() * d_vx.double()).sum(dim=1)
            / d_ev.double().square().sum(dim=1).clamp_min(EPS)
        ).float()
        rho = rho.clamp(args.richardson_rho_min, args.richardson_rho_max)
        guided_x = clean["x"] + (rho / (1.0 - rho))[:, None] * d_vx
    else:
        raise ValueError(condition)
    return (state - guided_x) / t[:, None].clamp_min(args.conversion_clip)


@torch.inference_mode()
def sample_condition(
    *,
    models: dict[str, DenoiseMLP],
    D: int,
    condition: str,
    gamma: float,
    n: int,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=device.type)
    generator.manual_seed(seed)
    state = float(args.sample_t_max) * torch.randn((n, D), device=device, generator=generator)
    grid = torch.linspace(
        args.sample_t_max,
        args.sample_t_min,
        args.sample_steps + 1,
        device=device,
        dtype=torch.float32,
    )
    for i in range(args.sample_steps):
        t_now = grid[i]
        t_next = grid[i + 1]
        t = torch.full((n,), float(t_now), device=device)
        vel = velocity_for_condition(
            models=models,
            state=state,
            t=t,
            condition=condition,
            gamma=gamma,
            args=args,
        )
        state = state + (t_next - t_now) * vel
    # Final clean readout: x-model for extrapolated cases, target model otherwise.
    t = torch.full((n,), float(grid[-1]), device=device)
    if condition in ("x", "v", "eps"):
        output = models[condition](state, t)
        return clean_from_output(output, state, t, condition, args.conversion_clip)
    # One last guided clean estimate.
    outputs = {target: models[target](state, t) for target in ("x", "v", "eps")}
    clean = {
        target: clean_from_output(outputs[target], state, t, target, args.conversion_clip)
        for target in outputs
    }
    if condition == "xv":
        return clean["x"] + gamma * (clean["x"] - clean["v"])
    if condition == "xeps":
        return clean["x"] + gamma * (clean["x"] - clean["eps"])
    if condition == "richardson":
        d_ev = clean["v"] - clean["eps"]
        d_vx = clean["x"] - clean["v"]
        rho = (
            (d_ev.double() * d_vx.double()).sum(dim=1)
            / d_ev.double().square().sum(dim=1).clamp_min(EPS)
        ).float()
        rho = rho.clamp(args.richardson_rho_min, args.richardson_rho_max)
        return clean["x"] + (rho / (1.0 - rho))[:, None] * d_vx
    raise ValueError(condition)


def sliced_wasserstein_2d(
    a: np.ndarray,
    b: np.ndarray,
    *,
    projections: int,
    seed: int,
) -> float:
    if a.shape[1] != 2 or b.shape[1] != 2:
        raise ValueError("expected 2-D intrinsic samples")
    n = min(len(a), len(b))
    rng = np.random.default_rng(seed)
    idx_a = rng.choice(len(a), n, replace=False)
    idx_b = rng.choice(len(b), n, replace=False)
    aa = a[idx_a]
    bb = b[idx_b]
    theta = rng.normal(size=(projections, 2))
    theta /= np.linalg.norm(theta, axis=1, keepdims=True) + 1e-12
    pa = np.sort(aa @ theta.T, axis=0)
    pb = np.sort(bb @ theta.T, axis=0)
    return float(np.mean(np.sqrt(np.mean((pa - pb) ** 2, axis=0))))


def rbf_mmd_2d(a: np.ndarray, b: np.ndarray, *, max_points: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    n = min(len(a), len(b), max_points)
    aa = a[rng.choice(len(a), n, replace=False)]
    bb = b[rng.choice(len(b), n, replace=False)]
    joined = np.concatenate([aa, bb], axis=0)
    # Median heuristic on a bounded subset.
    m = min(len(joined), 1024)
    subset = joined[rng.choice(len(joined), m, replace=False)]
    d2 = ((subset[:, None, :] - subset[None, :, :]) ** 2).sum(axis=2)
    positive = d2[d2 > 0]
    sigma2 = float(np.median(positive)) if len(positive) else 1.0
    sigma2 = max(sigma2, 1e-8)

    def kernel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        dist2 = ((x[:, None, :] - y[None, :, :]) ** 2).sum(axis=2)
        return np.exp(-dist2 / (2.0 * sigma2))

    kaa = kernel(aa, aa)
    kbb = kernel(bb, bb)
    kab = kernel(aa, bb)
    np.fill_diagonal(kaa, 0.0)
    np.fill_diagonal(kbb, 0.0)
    term_aa = kaa.sum() / max(n * (n - 1), 1)
    term_bb = kbb.sum() / max(n * (n - 1), 1)
    term_ab = kab.mean()
    return float(term_aa + term_bb - 2.0 * term_ab)


def save_csv(path: Path, rows: Iterable[dict[str, float | str]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_scatter_grid(
    path: Path,
    reference: np.ndarray,
    samples: list[tuple[str, np.ndarray]],
    *,
    limit: int,
) -> None:
    count = 1 + len(samples)
    cols = min(4, count)
    rows = math.ceil(count / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows), squeeze=False)
    panels = [("reference", reference)] + samples
    for ax, (name, data) in zip(axes.flat, panels):
        arr = data[:limit]
        ax.scatter(arr[:, 0], arr[:, 1], s=3, alpha=0.45)
        ax.set_title(name)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-2.0, 2.0)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_axis_plot(path: Path, rows: list[dict[str, float]]) -> None:
    by_D: dict[int, list[dict[str, float]]] = {}
    for row in rows:
        by_D.setdefault(int(row["D"]), []).append(row)
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    for D, subset in sorted(by_D.items()):
        subset = sorted(subset, key=lambda r: r["time"])
        ax.plot(
            [r["time"] for r in subset],
            [r["cos_ev_vx"] for r in subset],
            marker="o",
            label=f"D={D}",
        )
    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("t (0=clean, 1=noise)")
    ax.set_ylabel("cos(x_v - x_eps, x_x - x_v)")
    ax.set_title("Prediction-target axis alignment")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--dims", type=parse_int_list, default=parse_int_list("2,8,16,512"))
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=5, help="number of Linear layers including output")
    p.add_argument("--time-dim", type=int, default=32)
    p.add_argument("--train-steps", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--loss-space", choices=("v", "direct"), default="v")
    p.add_argument("--t-min", type=float, default=0.02)
    p.add_argument("--t-max", type=float, default=0.98)
    p.add_argument("--conversion-clip", type=float, default=0.02)
    p.add_argument("--data-jitter", type=float, default=0.015)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--eval-times", type=parse_float_list, default=parse_float_list("0.1,0.3,0.5,0.7,0.9"))
    p.add_argument("--eval-samples", type=int, default=8192)
    p.add_argument("--eval-batch-size", type=int, default=2048)
    p.add_argument("--sample-count", type=int, default=10000)
    p.add_argument("--sample-steps", type=int, default=200)
    p.add_argument("--sample-t-max", type=float, default=0.98)
    p.add_argument("--sample-t-min", type=float, default=0.02)
    p.add_argument("--gammas", type=parse_float_list, default=parse_float_list("0.1,0.25,0.5,1.0"))
    p.add_argument("--swd-projections", type=int, default=256)
    p.add_argument("--mmd-max-points", type=int, default=4096)
    p.add_argument("--plot-points", type=int, default=4000)
    p.add_argument("--richardson-rho-min", type=float, default=0.0)
    p.add_argument("--richardson-rho-max", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=20260807)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-checkpoints", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if any(D < 2 for D in args.dims):
        raise ValueError("all D must be >= 2")
    if not (0 < args.t_min < args.t_max < 1):
        raise ValueError("training t range must be inside (0,1)")
    if not (0 < args.sample_t_min < args.sample_t_max < 1):
        raise ValueError("sampling t range must be inside (0,1)")
    if args.richardson_rho_max >= 1:
        raise ValueError("Richardson rho max must be < 1")

    set_seed(args.seed)
    device = torch.device(args.device)
    out = args.output_root.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "script": "run_prediction_target_extrapolation_toy_v3.py",
        "definition": "x_t=(1-t)x+t*eps; v=eps-x",
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    all_teacher_rows: list[dict[str, float]] = []
    all_generation_rows: list[dict[str, float | str]] = []
    reference_generator = torch.Generator(device=device.type)
    reference_generator.manual_seed(args.seed + 999)
    reference_2d = sample_spiral_2d(
        max(args.sample_count, 20000),
        device=device,
        jitter=args.data_jitter,
        generator=reference_generator,
    ).cpu().numpy()

    for D in args.dims:
        print(f"\n===== Ambient dimension D={D} =====", flush=True)
        P = orthonormal_embedding(D, 2, args.seed + 17 * D, device)
        result = train_models(D=D, P=P, args=args, device=device)
        dim_dir = out / f"D{D}"
        dim_dir.mkdir(parents=True, exist_ok=True)
        save_csv(dim_dir / "train_history.csv", result.history)
        if args.save_checkpoints:
            torch.save(
                {k: v.state_dict() for k, v in result.models.items()},
                dim_dir / "models.pt",
            )

        teacher_rows, _ = evaluate_teacher_forced(
            models=result.models,
            D=D,
            P=P,
            args=args,
            device=device,
        )
        save_csv(dim_dir / "teacher_metrics.csv", teacher_rows)
        all_teacher_rows.extend(teacher_rows)

        conditions: list[tuple[str, str, float]] = [
            ("x", "x", 0.0),
            ("v", "v", 0.0),
            ("eps", "eps", 0.0),
        ]
        for gamma in args.gammas:
            conditions.append((f"xv_g{gamma:g}", "xv", gamma))
            conditions.append((f"xeps_g{gamma:g}", "xeps", gamma))
        conditions.append(("richardson", "richardson", 0.0))

        sample_panels: list[tuple[str, np.ndarray]] = []
        for index, (name, kind, gamma) in enumerate(conditions):
            samples_D = sample_condition(
                models=result.models,
                D=D,
                condition=kind,
                gamma=gamma,
                n=args.sample_count,
                args=args,
                device=device,
                seed=args.seed * 10000 + D * 101,
            )
            intrinsic = project_to_intrinsic(samples_D, P).cpu().numpy()
            off = off_subspace_rms(samples_D, P).cpu().numpy()
            swd = sliced_wasserstein_2d(
                intrinsic,
                reference_2d,
                projections=args.swd_projections,
                seed=args.seed + D + index,
            )
            mmd = rbf_mmd_2d(
                intrinsic,
                reference_2d,
                max_points=args.mmd_max_points,
                seed=args.seed + 3 * D + index,
            )
            row = {
                "D": float(D),
                "condition": name,
                "kind": kind,
                "gamma": float(gamma),
                "samples": float(args.sample_count),
                "swd_2d": float(swd),
                "mmd_2d": float(mmd),
                "off_subspace_rms_mean": float(off.mean()),
                "off_subspace_rms_median": float(np.median(off)),
                "intrinsic_radius_mean": float(np.linalg.norm(intrinsic, axis=1).mean()),
            }
            all_generation_rows.append(row)
            sample_panels.append((name, intrinsic))
            np.savez_compressed(dim_dir / f"samples_{name}.npz", intrinsic=intrinsic)
            print(
                f"[D={D}] {name:>14s}: SWD={swd:.5f} MMD={mmd:.5g} "
                f"off-subspace={off.mean():.5g}",
                flush=True,
            )

        save_scatter_grid(
            dim_dir / "generation_scatter.png",
            reference_2d,
            sample_panels,
            limit=args.plot_points,
        )

        # Free GPU memory before next D.
        del result
        if device.type == "cuda":
            torch.cuda.empty_cache()

    save_csv(out / "teacher_metrics_all.csv", all_teacher_rows)
    save_csv(out / "generation_metrics_all.csv", all_generation_rows)
    save_axis_plot(out / "prediction_target_axis_alignment.png", all_teacher_rows)

    # Compact summary: best generation condition per D by SWD.
    summary: list[dict[str, float | str]] = []
    for D in args.dims:
        subset = [r for r in all_generation_rows if int(float(r["D"])) == D]
        best = min(subset, key=lambda r: float(r["swd_2d"]))
        x_base = next(r for r in subset if r["condition"] == "x")
        summary.append(
            {
                "D": float(D),
                "best_condition": str(best["condition"]),
                "best_swd": float(best["swd_2d"]),
                "x_swd": float(x_base["swd_2d"]),
                "relative_swd_vs_x": float(best["swd_2d"]) / max(float(x_base["swd_2d"]), EPS) - 1.0,
            }
        )
    save_csv(out / "summary.csv", summary)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nDone. Results written to {out}", flush=True)


if __name__ == "__main__":
    main()