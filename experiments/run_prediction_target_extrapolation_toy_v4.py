#!/usr/bin/env python3
"""Prediction-target extrapolation toy experiment (v4).

This version extends v3 to test the regime in which direct x-prediction is
better than v/epsilon prediction but is itself still capacity-limited.

Main additions
--------------
1. Curved high-dimensional manifold
   Clean intrinsic points u in R^2 are mapped by a known nonlinear embedding

       x = phi(u) in R^D.

   curvature=0 recovers a linear 2-D subspace. curvature>0 adds random Fourier
   features before a fixed orthogonal rotation, so intrinsic dimension stays
   <=2 while the linear span can approach D.

2. Explicit scale control
   --scale-mode constant_norm reproduces the old orthonormal-embedding scaling.
   --scale-mode unit_rms multiplies phi by sqrt(D), so clean signal RMS per
   ambient coordinate does not vanish as D grows.

3. Large x-oracle
   A much wider/longer-trained x-predictor approximates the Bayes/large-capacity
   x denoiser. Finite-model approximation residual is measured against this
   oracle, not only against the paired clean sample.

4. Capacity sweep
   Multiple hidden widths are trained for x/v/epsilon on exactly the same
   batches/noise/times and the same initial parameters within each width.

5. Tangent/normal geometry
   Because phi is known, its Jacobian gives a local tangent basis. The x-v and
   x-epsilon gaps are decomposed into tangent and normal components.

6. Generation extrapolation
   Tests full-gap, tangent-only, normal-only and RMS-normalized x-v guidance.
   Generation metrics use identical random projections/subsets across
   conditions within one setting, plus bootstrap confidence intervals for
   SWD differences relative to the x baseline.

The script remains self-contained and uses new versioned outputs; do not
overwrite v3 results.
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

EPS = 1e-10


# ---------------------------------------------------------------------------
# Parsing / reproducibility
# ---------------------------------------------------------------------------

def parse_int_list(text: str) -> list[int]:
    vals = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return vals


def parse_float_list(text: str) -> list[float]:
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("expected comma-separated floats")
    return vals


def parse_str_list(text: str) -> list[str]:
    vals = [x.strip() for x in str(text).split(",") if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("expected comma-separated strings")
    return vals


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_seed(*values: int) -> int:
    x = 2166136261
    for value in values:
        x ^= int(value) & 0xFFFFFFFF
        x = (x * 16777619) & 0xFFFFFFFF
    return int(x)


def tag_float(x: float) -> str:
    s = f"{float(x):.4g}"
    return s.replace("-", "m").replace(".", "p")


# ---------------------------------------------------------------------------
# Intrinsic data and known nonlinear manifold
# ---------------------------------------------------------------------------

def sample_spiral_2d(
    n: int,
    *,
    device: torch.device,
    jitter: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Two-turn spiral in canonical 2-D coordinates."""
    u = torch.rand(n, device=device, generator=generator)
    theta = 4.0 * math.pi * u
    radius = 0.15 + 0.85 * u
    p = torch.stack([radius * torch.cos(theta), radius * torch.sin(theta)], dim=1)
    if jitter > 0:
        p = p + jitter * torch.randn(p.shape, device=device, generator=generator)
    return 1.6 * p


class CurvedEmbedding:
    """Known R^2 -> R^D nonlinear embedding with exact intrinsic readout.

    Before the final orthogonal rotation the first two coordinates are exactly
    the intrinsic coordinates. Remaining coordinates are random Fourier
    features. Therefore decode_intrinsic() is exact for on-manifold points and
    still provides a canonical coordinate readout for off-manifold points.
    """

    def __init__(
        self,
        D: int,
        *,
        curvature: float,
        frequency_scale: float,
        seed: int,
        device: torch.device,
        scale_mode: str,
    ) -> None:
        if D < 2:
            raise ValueError("D must be >= 2")
        if curvature < 0:
            raise ValueError("curvature must be non-negative")
        if scale_mode not in {"constant_norm", "unit_rms"}:
            raise ValueError(scale_mode)
        self.D = int(D)
        self.curvature = float(curvature)
        self.frequency_scale = float(frequency_scale)
        self.scale_mode = str(scale_mode)
        self.device = device

        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))
        raw_q = torch.randn(D, D, generator=g, dtype=torch.float64)
        q, _ = torch.linalg.qr(raw_q)
        self.Q = q.to(device=device, dtype=torch.float32)

        m = D - 2
        if m:
            w = torch.randn(m, 2, generator=g, dtype=torch.float64)
            w = w / w.norm(dim=1, keepdim=True).clamp_min(1e-12)
            # Spread frequencies so curvature can create genuinely high-rank span.
            radii = torch.exp(
                torch.linspace(
                    math.log(max(0.5, frequency_scale / 4.0)),
                    math.log(max(0.5001, frequency_scale)),
                    m,
                    dtype=torch.float64,
                )
            )
            self.W = (w * radii[:, None]).to(device=device, dtype=torch.float32)
            self.phase = (
                2.0 * math.pi * torch.rand(m, generator=g, dtype=torch.float64)
            ).to(device=device, dtype=torch.float32)
        else:
            self.W = torch.empty(0, 2, device=device)
            self.phase = torch.empty(0, device=device)

        # Nonlinear block has expected total squared energy about curvature^2.
        self.fourier_amp = (
            float(curvature) * math.sqrt(2.0 / max(m, 1)) if m else 0.0
        )
        self.global_scale = math.sqrt(float(D)) if scale_mode == "unit_rms" else 1.0

    def feature_coordinates(self, u: torch.Tensor) -> torch.Tensor:
        if u.ndim != 2 or u.shape[1] != 2:
            raise ValueError("u must be [B,2]")
        if self.D == 2:
            feat = u
        else:
            phase = u @ self.W.T + self.phase[None]
            nonlinear = self.fourier_amp * torch.sin(phase)
            feat = torch.cat([u, nonlinear], dim=1)
        return self.global_scale * feat

    def embed(self, u: torch.Tensor) -> torch.Tensor:
        return self.feature_coordinates(u) @ self.Q.T

    def decode_intrinsic(self, x: torch.Tensor) -> torch.Tensor:
        feature = (x @ self.Q) / self.global_scale
        return feature[:, :2]

    def manifold_consistency_rms(self, x: torch.Tensor) -> torch.Tensor:
        u = self.decode_intrinsic(x)
        recon = self.embed(u)
        return (x - recon).square().mean(dim=1).sqrt()

    def jacobian(self, u: torch.Tensor) -> torch.Tensor:
        """Return J_phi(u) with shape [B,D,2]."""
        b = len(u)
        j_feat = torch.zeros(b, self.D, 2, device=u.device, dtype=u.dtype)
        j_feat[:, 0, 0] = 1.0
        j_feat[:, 1, 1] = 1.0
        if self.D > 2:
            phase = u @ self.W.T + self.phase[None]
            coeff = self.fourier_amp * torch.cos(phase)  # [B,m]
            j_feat[:, 2:, :] = coeff[:, :, None] * self.W[None, :, :]
        j_feat = self.global_scale * j_feat
        # x = feature @ Q^T -> column Jacobian is Q @ J_feature.
        return torch.einsum("ij,bjk->bik", self.Q, j_feat)

    def tangent_basis(self, u: torch.Tensor) -> torch.Tensor:
        j = self.jacobian(u).double()
        q, _ = torch.linalg.qr(j, mode="reduced")
        return q.float()  # [B,D,2]

    def split_tangent_normal(
        self, vec: torch.Tensor, u: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.tangent_basis(u)
        tangent = torch.einsum("bdi,bi->bd", q, torch.einsum("bdi,bd->bi", q, vec))
        return tangent, vec - tangent


# ---------------------------------------------------------------------------
# Diffusion parameterization
# ---------------------------------------------------------------------------

def row_inner(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.double() * b.double()).sum(dim=1)


def row_rms(a: torch.Tensor) -> torch.Tensor:
    return a.double().square().mean(dim=1).sqrt().float()


def row_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a.double(), b.double(), dim=1, eps=EPS).float()


def clean_from_output(
    output: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
    target: str,
    clip: float,
) -> torch.Tensor:
    tc = t[:, None]
    if target == "x":
        return output
    if target == "v":
        return x_t - tc * output
    if target == "eps":
        return (x_t - tc * output) / (1.0 - tc).clamp_min(clip)
    raise ValueError(target)


def velocity_from_output(
    output: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
    target: str,
    clip: float,
) -> torch.Tensor:
    tc = t[:, None]
    if target == "x":
        return (x_t - output) / tc.clamp_min(clip)
    if target == "v":
        return output
    if target == "eps":
        return (output - x_t) / (1.0 - tc).clamp_min(clip)
    raise ValueError(target)


def direct_target(x: torch.Tensor, eps: torch.Tensor, target: str) -> torch.Tensor:
    if target == "x":
        return x
    if target == "v":
        return eps - x
    if target == "eps":
        return eps
    raise ValueError(target)


# ---------------------------------------------------------------------------
# Models / training
# ---------------------------------------------------------------------------

class TimeEmbedding(nn.Module):
    def __init__(self, dim: int, max_freq: float = 32.0):
        super().__init__()
        if dim % 2:
            raise ValueError("time_dim must be even")
        half = dim // 2
        self.register_buffer(
            "freqs",
            torch.exp(torch.linspace(0.0, math.log(max_freq), half)),
            persistent=False,
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        p = 2.0 * math.pi * t[:, None] * self.freqs[None]
        return torch.cat([torch.sin(p), torch.cos(p)], dim=1)


class DenoiseMLP(nn.Module):
    def __init__(self, D: int, hidden: int, depth: int, time_dim: int):
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be >=2")
        self.time = TimeEmbedding(time_dim)
        layers: list[nn.Module] = []
        d = D + time_dim
        for _ in range(depth - 1):
            layers.extend([nn.Linear(d, hidden), nn.SiLU()])
            d = hidden
        layers.append(nn.Linear(d, D))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, self.time(t)], dim=1))


@dataclass
class TrainBundle:
    models: dict[str, DenoiseMLP]
    history: list[dict[str, float]]


def build_same_init_models(
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
    out = {}
    for target in ("x", "v", "eps"):
        m = DenoiseMLP(D, hidden, depth, time_dim).to(device)
        m.load_state_dict(state)
        out[target] = m
    return out


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
        pred = velocity_from_output(output, x_t, t, target, conversion_clip)
        return F.mse_loss(pred, eps - x)
    if loss_space == "direct":
        return F.mse_loss(output, direct_target(x, eps, target))
    raise ValueError(loss_space)


def train_triplet(
    *,
    embedding: CurvedEmbedding,
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
    data_jitter: float,
    log_every: int,
    seed: int,
    device: torch.device,
) -> TrainBundle:
    D = embedding.D
    models = build_same_init_models(D, hidden, depth, time_dim, device, seed + hidden)
    opts = {
        k: torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=weight_decay)
        for k, m in models.items()
    }
    g = torch.Generator(device=device.type)
    g.manual_seed(seed * 1009 + hidden)
    history = []

    for step in range(1, steps + 1):
        u = sample_spiral_2d(batch_size, device=device, jitter=data_jitter, generator=g)
        x = embedding.embed(u)
        eps = torch.randn(x.shape, device=device, generator=g)
        t = torch.empty(batch_size, device=device).uniform_(t_min, t_max, generator=g)
        x_t = (1.0 - t[:, None]) * x + t[:, None] * eps
        losses = {}
        for target, model in models.items():
            opts[target].zero_grad(set_to_none=True)
            out = model(x_t, t)
            loss = loss_for_output(
                out,
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
            opts[target].step()
            losses[target] = float(loss.detach().cpu())

        if step == 1 or step % log_every == 0 or step == steps:
            history.append({"step": step, **{f"loss_{k}": v for k, v in losses.items()}})
            print(
                f"[H={hidden} {loss_space}] {step}/{steps} "
                + " ".join(f"{k}={losses[k]:.5g}" for k in ("x", "v", "eps")),
                flush=True,
            )
    return TrainBundle(models=models, history=history)


def train_oracle_x(
    *,
    embedding: CurvedEmbedding,
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
    data_jitter: float,
    log_every: int,
    seed: int,
    device: torch.device,
) -> tuple[DenoiseMLP, list[dict[str, float]]]:
    D = embedding.D
    torch.manual_seed(seed + 900001)
    model = DenoiseMLP(D, hidden, depth, time_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    g = torch.Generator(device=device.type)
    g.manual_seed(seed * 2029 + 17)
    hist = []
    for step in range(1, steps + 1):
        u = sample_spiral_2d(batch_size, device=device, jitter=data_jitter, generator=g)
        x = embedding.embed(u)
        eps = torch.randn(x.shape, device=device, generator=g)
        t = torch.empty(batch_size, device=device).uniform_(t_min, t_max, generator=g)
        x_t = (1.0 - t[:, None]) * x + t[:, None] * eps
        opt.zero_grad(set_to_none=True)
        out = model(x_t, t)
        loss = loss_for_output(
            out,
            x_t=x_t,
            t=t,
            x=x,
            eps=eps,
            target="x",
            loss_space=loss_space,
            conversion_clip=conversion_clip,
        )
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        if step == 1 or step % log_every == 0 or step == steps:
            hist.append({"step": step, "loss_x_oracle": float(loss.detach().cpu())})
            print(
                f"[ORACLE H={hidden} {loss_space}] {step}/{steps} "
                f"x={float(loss.detach().cpu()):.5g}",
                flush=True,
            )
    return model, hist


# ---------------------------------------------------------------------------
# Teacher-forced mechanism audit
# ---------------------------------------------------------------------------

@torch.inference_mode()
def evaluate_teacher(
    *,
    models: dict[str, DenoiseMLP],
    oracle: DenoiseMLP,
    embedding: CurvedEmbedding,
    hidden: int,
    loss_space: str,
    eval_times: Sequence[float],
    eval_samples: int,
    eval_batch_size: int,
    data_jitter: float,
    conversion_clip: float,
    gammas: Sequence[float],
    seed: int,
    device: torch.device,
) -> list[dict[str, float]]:
    rows = []
    g = torch.Generator(device=device.type)
    g.manual_seed(seed * 3001 + hidden)

    for time in eval_times:
        sums: dict[str, float] = {}
        n_total = 0
        for start in range(0, eval_samples, eval_batch_size):
            n = min(eval_batch_size, eval_samples - start)
            u = sample_spiral_2d(n, device=device, jitter=data_jitter, generator=g)
            x = embedding.embed(u)
            eps = torch.randn(x.shape, device=device, generator=g)
            t = torch.full((n,), float(time), device=device)
            x_t = (1.0 - t[:, None]) * x + t[:, None] * eps

            clean = {}
            for target, m in models.items():
                clean[target] = clean_from_output(
                    m(x_t, t), x_t, t, target, conversion_clip
                )
            x_oracle = oracle(x_t, t)

            d_ev = clean["v"] - clean["eps"]
            g_xv = clean["x"] - clean["v"]
            g_xeps = clean["x"] - clean["eps"]
            r_oracle = x_oracle - clean["x"]
            r_true = x - clean["x"]

            txv, nxv = embedding.split_tangent_normal(g_xv, u)
            txe, nxe = embedding.split_tangent_normal(g_xeps, u)
            tr, nr = embedding.split_tangent_normal(r_oracle, u)

            gamma_star_xv = row_inner(r_oracle, g_xv) / row_inner(g_xv, g_xv).clamp_min(EPS)
            gamma_star_xeps = row_inner(r_oracle, g_xeps) / row_inner(g_xeps, g_xeps).clamp_min(EPS)

            metrics: dict[str, torch.Tensor] = {
                "cos_ev_vx": row_cosine(d_ev, g_xv),
                "cos_xv_oracle_residual": row_cosine(g_xv, r_oracle),
                "cos_xeps_oracle_residual": row_cosine(g_xeps, r_oracle),
                "cos_xv_true_residual": row_cosine(g_xv, r_true),
                "gamma_star_xv_oracle": gamma_star_xv.float(),
                "gamma_star_xeps_oracle": gamma_star_xeps.float(),
                "positive_gamma_star_xv_fraction": (gamma_star_xv > 0).float(),
                "positive_gamma_star_xeps_fraction": (gamma_star_xeps > 0).float(),
                "gap_xv_rms": row_rms(g_xv),
                "gap_xeps_rms": row_rms(g_xeps),
                "gap_xv_tangent_rms": row_rms(txv),
                "gap_xv_normal_rms": row_rms(nxv),
                "gap_xeps_tangent_rms": row_rms(txe),
                "gap_xeps_normal_rms": row_rms(nxe),
                "oracle_residual_rms": row_rms(r_oracle),
                "oracle_residual_tangent_rms": row_rms(tr),
                "oracle_residual_normal_rms": row_rms(nr),
                "oracle_true_mse": (x_oracle - x).square().mean(dim=1),
                "x_oracle_approx_mse": (clean["x"] - x_oracle).square().mean(dim=1),
                "v_oracle_approx_mse": (clean["v"] - x_oracle).square().mean(dim=1),
                "eps_oracle_approx_mse": (clean["eps"] - x_oracle).square().mean(dim=1),
                "x_true_mse": (clean["x"] - x).square().mean(dim=1),
                "v_true_mse": (clean["v"] - x).square().mean(dim=1),
                "eps_true_mse": (clean["eps"] - x).square().mean(dim=1),
                "x_manifold_consistency": embedding.manifold_consistency_rms(clean["x"]),
                "v_manifold_consistency": embedding.manifold_consistency_rms(clean["v"]),
                "eps_manifold_consistency": embedding.manifold_consistency_rms(clean["eps"]),
            }

            for gamma in gammas:
                xv = clean["x"] + float(gamma) * g_xv
                xe = clean["x"] + float(gamma) * g_xeps
                metrics[f"xv_g{tag_float(gamma)}_oracle_mse"] = (
                    xv - x_oracle
                ).square().mean(dim=1)
                metrics[f"xeps_g{tag_float(gamma)}_oracle_mse"] = (
                    xe - x_oracle
                ).square().mean(dim=1)

            for k, v in metrics.items():
                sums[k] = sums.get(k, 0.0) + float(v.double().sum().cpu())
            n_total += n

        row = {
            "D": embedding.D,
            "curvature": embedding.curvature,
            "scale_mode": embedding.scale_mode,
            "hidden": hidden,
            "loss_space": loss_space,
            "time": float(time),
            "samples": n_total,
        }
        row.update({k: v / n_total for k, v in sums.items()})
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Sampling conditions
# ---------------------------------------------------------------------------

@torch.inference_mode()
def clean_predictions(
    models: dict[str, DenoiseMLP],
    state: torch.Tensor,
    t: torch.Tensor,
    clip: float,
    targets: Sequence[str] | None = None,
) -> dict[str, torch.Tensor]:
    requested = tuple(models) if targets is None else tuple(targets)
    return {
        target: clean_from_output(models[target](state, t), state, t, target, clip)
        for target in requested
    }


def normalize_gap_rms(gap: torch.Tensor) -> torch.Tensor:
    return gap / row_rms(gap)[:, None].clamp_min(1e-8)


@torch.inference_mode()
def guided_clean(
    *,
    models: dict[str, DenoiseMLP],
    embedding: CurvedEmbedding,
    state: torch.Tensor,
    t: torch.Tensor,
    kind: str,
    strength: float,
    clip: float,
) -> torch.Tensor:
    if kind == "x":
        return clean_predictions(models, state, t, clip, ("x",))["x"]
    if kind == "v":
        return clean_predictions(models, state, t, clip, ("v",))["v"]
    if kind == "eps":
        return clean_predictions(models, state, t, clip, ("eps",))["eps"]

    targets = ("x", "v") if kind.startswith("xv") else ("x", "eps")
    c = clean_predictions(models, state, t, clip, targets)
    gap = c["x"] - c[targets[1]]

    if kind in {"xv", "xeps"}:
        return c["x"] + float(strength) * gap
    if kind == "xv_norm":
        return c["x"] + float(strength) * normalize_gap_rms(gap)
    if kind in {"xv_tangent", "xv_normal"}:
        u_hat = embedding.decode_intrinsic(c["x"])
        tangent, normal = embedding.split_tangent_normal(gap, u_hat)
        chosen = tangent if kind == "xv_tangent" else normal
        return c["x"] + float(strength) * chosen
    raise ValueError(kind)


@torch.inference_mode()
def sample_condition(
    *,
    models: dict[str, DenoiseMLP],
    oracle: DenoiseMLP,
    embedding: CurvedEmbedding,
    kind: str,
    strength: float,
    sample_count: int,
    sample_batch_size: int,
    sample_steps: int,
    t_max: float,
    t_min: float,
    clip: float,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    all_samples = []
    for start in range(0, sample_count, sample_batch_size):
        n = min(sample_batch_size, sample_count - start)
        g = torch.Generator(device=device.type)
        g.manual_seed(seed + start)
        state = float(t_max) * torch.randn((n, embedding.D), device=device, generator=g)
        grid = torch.linspace(t_max, t_min, sample_steps + 1, device=device)
        for i in range(sample_steps):
            t_now, t_next = grid[i], grid[i + 1]
            t = torch.full((n,), float(t_now), device=device)
            if kind == "oracle":
                xhat = oracle(state, t)
            else:
                xhat = guided_clean(
                    models=models,
                    embedding=embedding,
                    state=state,
                    t=t,
                    kind=kind,
                    strength=strength,
                    clip=clip,
                )
            vel = (state - xhat) / t[:, None].clamp_min(clip)
            state = state + (t_next - t_now) * vel

        t = torch.full((n,), float(grid[-1]), device=device)
        if kind == "oracle":
            final = oracle(state, t)
        else:
            final = guided_clean(
                models=models,
                embedding=embedding,
                state=state,
                t=t,
                kind=kind,
                strength=strength,
                clip=clip,
            )
        all_samples.append(final.cpu().numpy())
    return np.concatenate(all_samples, axis=0)


@torch.inference_mode()
def sample_mixture_conditions(
    *,
    models: dict[str, DenoiseMLP],
    embedding: CurvedEmbedding,
    kind: str,
    strengths: Sequence[float],
    sample_count: int,
    sample_batch_size: int,
    sample_steps: int,
    t_max: float,
    t_min: float,
    clip: float,
    seed: int,
    device: torch.device,
) -> list[np.ndarray]:
    """Integrate several raw x-v or x-epsilon mixtures in one GPU batch.

    The trajectories share initial noise but keep independent states after the
    first update. This is algebraically the same computation as repeatedly
    calling ``sample_condition`` and substantially reduces small-kernel launch
    overhead in the low-dimensional screen.
    """
    if kind not in {"xv", "xeps"}:
        raise ValueError(f"unsupported grouped mixture kind: {kind}")
    if not strengths:
        return []

    target = "v" if kind == "xv" else "eps"
    strengths_tensor = torch.as_tensor(
        strengths, device=device, dtype=torch.float32
    )[:, None, None]
    collected: list[list[np.ndarray]] = [[] for _ in strengths]
    grid = torch.linspace(t_max, t_min, sample_steps + 1, device=device)

    for start in range(0, sample_count, sample_batch_size):
        n = min(sample_batch_size, sample_count - start)
        generator = torch.Generator(device=device.type)
        generator.manual_seed(seed + start)
        initial = float(t_max) * torch.randn(
            (n, embedding.D), device=device, generator=generator
        )
        states = initial[None].expand(len(strengths), -1, -1).clone()

        for i in range(sample_steps):
            t_now, t_next = grid[i], grid[i + 1]
            flat_state = states.reshape(-1, embedding.D)
            flat_t = torch.full(
                (len(strengths) * n,), float(t_now), device=device
            )
            clean = clean_predictions(
                models, flat_state, flat_t, clip, ("x", target)
            )
            x_clean = clean["x"].reshape(len(strengths), n, embedding.D)
            other_clean = clean[target].reshape(
                len(strengths), n, embedding.D
            )
            guided = x_clean + strengths_tensor * (x_clean - other_clean)
            velocity = (states - guided) / flat_t[0].clamp_min(clip)
            states = states + (t_next - t_now) * velocity

        flat_state = states.reshape(-1, embedding.D)
        flat_t = torch.full(
            (len(strengths) * n,), float(grid[-1]), device=device
        )
        clean = clean_predictions(models, flat_state, flat_t, clip, ("x", target))
        x_clean = clean["x"].reshape(len(strengths), n, embedding.D)
        other_clean = clean[target].reshape(len(strengths), n, embedding.D)
        final = x_clean + strengths_tensor * (x_clean - other_clean)
        for index, value in enumerate(final):
            collected[index].append(value.cpu().numpy())

    return [np.concatenate(parts, axis=0) for parts in collected]


# ---------------------------------------------------------------------------
# Paired generation metrics
# ---------------------------------------------------------------------------

def fixed_projection_matrix(projections: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=(projections, 2))
    theta /= np.linalg.norm(theta, axis=1, keepdims=True) + 1e-12
    return theta


def swd_2d_fixed(
    a: np.ndarray,
    b: np.ndarray,
    *,
    theta: np.ndarray,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
) -> float:
    aa = a[idx_a]
    bb = b[idx_b]
    pa = np.sort(aa @ theta.T, axis=0)
    pb = np.sort(bb @ theta.T, axis=0)
    return float(np.mean(np.sqrt(np.mean((pa - pb) ** 2, axis=0))))


def rbf_bandwidth_2d_fixed(
    a: np.ndarray,
    b: np.ndarray,
    *,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    bandwidth_subset: np.ndarray,
) -> float:
    aa = a[idx_a]
    bb = b[idx_b]
    joined = np.concatenate([aa, bb], axis=0)
    sub = joined[bandwidth_subset % len(joined)]
    d2 = ((sub[:, None, :] - sub[None, :, :]) ** 2).sum(axis=2)
    positive = d2[d2 > 0]
    return max(float(np.median(positive)) if len(positive) else 1.0, 1e-8)


def mmd_2d_fixed(
    a: np.ndarray,
    b: np.ndarray,
    *,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    sigma2: float,
) -> float:
    aa = a[idx_a]
    bb = b[idx_b]
    sigma2 = max(float(sigma2), 1e-8)

    def k(x, y):
        z = ((x[:, None, :] - y[None, :, :]) ** 2).sum(axis=2)
        return np.exp(-z / (2.0 * sigma2))

    kaa, kbb, kab = k(aa, aa), k(bb, bb), k(aa, bb)
    np.fill_diagonal(kaa, 0.0)
    np.fill_diagonal(kbb, 0.0)
    n = len(aa)
    return float(
        kaa.sum() / max(n * (n - 1), 1)
        + kbb.sum() / max(n * (n - 1), 1)
        - 2.0 * kab.mean()
    )


def bootstrap_swd_delta(
    candidate: np.ndarray,
    baseline: np.ndarray,
    reference: np.ndarray,
    *,
    theta: np.ndarray,
    reps: int,
    seed: int,
    max_points: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    paired_count = min(len(candidate), len(baseline))
    bootstrap_count = min(paired_count, len(reference), max_points)
    candidate_projection = candidate[:paired_count] @ theta.T
    baseline_projection = baseline[:paired_count] @ theta.T
    reference_projection = reference @ theta.T
    deltas = []
    for _ in range(reps):
        ids = rng.integers(0, paired_count, size=bootstrap_count)
        rid = rng.integers(0, len(reference), size=bootstrap_count)
        # Conditions are paired by initial noise, so use the same ids for
        # baseline and candidate.
        ca = np.sort(candidate_projection[ids], axis=0)
        ba = np.sort(baseline_projection[ids], axis=0)
        rr = np.sort(reference_projection[rid], axis=0)
        c = float(np.mean(np.sqrt(np.mean((ca - rr) ** 2, axis=0))))
        b = float(np.mean(np.sqrt(np.mean((ba - rr) ** 2, axis=0))))
        deltas.append(c - b)
    arr = np.asarray(deltas)
    return float(arr.mean()), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


# ---------------------------------------------------------------------------
# IO / plotting
# ---------------------------------------------------------------------------

def save_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def load_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    typed = []
    for row in rows:
        converted = {}
        for key, value in row.items():
            try:
                converted[key] = float(value)
            except (TypeError, ValueError):
                converted[key] = value
        typed.append(converted)
    return typed


def plot_generation_grid(path: Path, ref: np.ndarray, panels: list[tuple[str, np.ndarray]], limit: int):
    all_panels = [("reference", ref)] + panels
    cols = 4
    rows = math.ceil(len(all_panels) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.1 * cols, 4.1 * rows), squeeze=False)
    for ax, (name, pts) in zip(axes.flat, all_panels):
        p = pts[:limit]
        ax.scatter(p[:, 0], p[:, 1], s=3, alpha=0.4)
        ax.set_title(name)
        ax.set_aspect("equal", adjustable="box")
    for ax in axes.flat[len(all_panels):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_phase(path: Path, rows: list[dict]):
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for curv in sorted({float(r["curvature"]) for r in rows}):
        sub = sorted(
            [r for r in rows if float(r["curvature"]) == curv],
            key=lambda r: float(r["x_oracle_error"]),
        )
        ax.plot(
            [r["x_oracle_error"] for r in sub],
            [r["best_relative_swd_vs_x"] for r in sub],
            marker="o",
            label=f"curvature={curv:g}",
        )
    ax.axhline(0.0, linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("x predictor approximation MSE to large x-oracle")
    ax.set_ylabel("best relative SWD vs x baseline")
    ax.set_title("Where does prediction-target extrapolation help?")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--dims", type=parse_int_list, default=parse_int_list("512"))
    p.add_argument("--curvatures", type=parse_float_list, default=parse_float_list("0,0.25,0.5,1.0"))
    p.add_argument("--hidden-dims", type=parse_int_list, default=parse_int_list("64,128,256,512,1024"))
    p.add_argument("--loss-spaces", type=parse_str_list, default=parse_str_list("v"))
    p.add_argument("--scale-mode", choices=("constant_norm", "unit_rms"), default="unit_rms")
    p.add_argument("--frequency-scale", type=float, default=6.0)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--time-dim", type=int, default=32)
    p.add_argument("--train-steps", type=int, default=30000)
    p.add_argument("--oracle-hidden-dim", type=int, default=2048)
    p.add_argument("--oracle-depth", type=int, default=6)
    p.add_argument("--oracle-train-steps", type=int, default=60000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--t-min", type=float, default=0.02)
    p.add_argument("--t-max", type=float, default=0.98)
    p.add_argument("--conversion-clip", type=float, default=0.02)
    p.add_argument("--data-jitter", type=float, default=0.015)
    p.add_argument("--log-every", type=int, default=1000)
    p.add_argument("--eval-times", type=parse_float_list, default=parse_float_list("0.1,0.3,0.5,0.7,0.9"))
    p.add_argument("--eval-samples", type=int, default=8192)
    p.add_argument("--eval-batch-size", type=int, default=1024)
    p.add_argument("--sample-count", type=int, default=10000)
    p.add_argument("--sample-batch-size", type=int, default=1000)
    p.add_argument("--sample-steps", type=int, default=200)
    p.add_argument("--sample-t-max", type=float, default=0.98)
    p.add_argument("--sample-t-min", type=float, default=0.02)
    p.add_argument(
        "--gammas",
        type=parse_float_list,
        default=parse_float_list("-0.1,-0.03,-0.01,0.003,0.01,0.03,0.1,0.3"),
    )
    p.add_argument(
        "--normalized-etas",
        type=parse_float_list,
        default=parse_float_list("-0.03,-0.01,0.01,0.03"),
    )
    p.add_argument("--swd-projections", type=int, default=256)
    p.add_argument("--metric-max-points", type=int, default=4096)
    p.add_argument("--bootstrap-reps", type=int, default=100)
    p.add_argument("--bootstrap-max-points", type=int, default=1024)
    p.add_argument("--bootstrap-projections", type=int, default=64)
    p.add_argument("--plot-points", type=int, default=4000)
    p.add_argument("--seeds", type=parse_int_list, default=parse_int_list("20260807"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--generation-profile",
        choices=("core", "full"),
        default="full",
        help=(
            "core samples x/v/eps/oracle and raw x-v/x-eps mixtures only; "
            "full additionally samples tangent, normal, and RMS-normalized variants"
        ),
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="reuse saved oracle/model checkpoints and completed condition files",
    )
    p.add_argument("--save-checkpoints", action="store_true")
    return p.parse_args()


def run_setting(
    *,
    args: argparse.Namespace,
    embedding: CurvedEmbedding,
    oracle: DenoiseMLP,
    models: dict[str, DenoiseMLP],
    hidden: int,
    loss_space: str,
    seed: int,
    out: Path,
    reference_u: np.ndarray,
    device: torch.device,
) -> tuple[list[dict], list[dict], dict]:
    teacher = evaluate_teacher(
        models=models,
        oracle=oracle,
        embedding=embedding,
        hidden=hidden,
        loss_space=loss_space,
        eval_times=args.eval_times,
        eval_samples=args.eval_samples,
        eval_batch_size=args.eval_batch_size,
        data_jitter=args.data_jitter,
        conversion_clip=args.conversion_clip,
        gammas=args.gammas,
        seed=seed,
        device=device,
    )
    save_csv(out / "teacher_metrics.csv", teacher)

    conditions: list[tuple[str, str, float]] = [
        ("x", "x", 0.0),
        ("v", "v", 0.0),
        ("eps", "eps", 0.0),
        ("oracle", "oracle", 0.0),
    ]
    for gamma in args.gammas:
        conditions.extend(
            [
                (f"xv_g{tag_float(gamma)}", "xv", gamma),
                (f"xeps_g{tag_float(gamma)}", "xeps", gamma),
            ]
        )
        if args.generation_profile == "full":
            conditions.extend(
                [
                    (f"xv_tan_g{tag_float(gamma)}", "xv_tangent", gamma),
                    (f"xv_normcomp_g{tag_float(gamma)}", "xv_normal", gamma),
                ]
            )
    if args.generation_profile == "full":
        for eta in args.normalized_etas:
            conditions.append((f"xv_rms_eta{tag_float(eta)}", "xv_norm", eta))

    # Identical projections/subsets for all conditions.
    n_metric = min(args.sample_count, len(reference_u), args.metric_max_points)
    rng = np.random.default_rng(stable_seed(seed, embedding.D, hidden, 991))
    idx_s = rng.choice(args.sample_count, n_metric, replace=False)
    idx_r = rng.choice(len(reference_u), n_metric, replace=False)
    theta = fixed_projection_matrix(
        args.swd_projections, stable_seed(seed, embedding.D, hidden, 992)
    )
    bw_subset = rng.choice(2 * n_metric, min(1024, 2 * n_metric), replace=False)

    partial_path = out / "generation_metrics.partial.csv"
    completed_rows = load_csv(partial_path) if args.resume else []
    completed_by_name = {str(row["condition"]): row for row in completed_rows}
    grouped_ambient: dict[str, np.ndarray] = {}
    gen_rows: list[dict] = []
    panels: list[tuple[str, np.ndarray]] = []
    x_intrinsic: np.ndarray | None = None
    mmd_sigma2: float | None = None
    sample_seed = stable_seed(seed, embedding.D, hidden, int(1000 * embedding.curvature), 77)

    for name, kind, strength in conditions:
        sample_path = out / f"samples_{name}.npz"
        if name in completed_by_name and sample_path.is_file():
            uu = np.load(sample_path)["intrinsic"]
            row = completed_by_name[name]
            if name == "x":
                mmd_sigma2 = float(
                    row.get(
                        "mmd_sigma2",
                        rbf_bandwidth_2d_fixed(
                            uu,
                            reference_u,
                            idx_a=idx_s,
                            idx_b=idx_r,
                            bandwidth_subset=bw_subset,
                        ),
                    )
                )
            print(f"[sample] {name}: complete; reusing", flush=True)
        else:
            print(f"[sample] {name}: generating {args.sample_count}", flush=True)
            if kind in {"xv", "xeps"}:
                if name not in grouped_ambient:
                    pending = [
                        (item_name, item_strength)
                        for item_name, item_kind, item_strength in conditions
                        if item_kind == kind
                        and not (
                            item_name in completed_by_name
                            and (out / f"samples_{item_name}.npz").is_file()
                        )
                    ]
                    print(
                        f"[sample] batching {len(pending)} {kind} trajectories",
                        flush=True,
                    )
                    values = sample_mixture_conditions(
                        models=models,
                        embedding=embedding,
                        kind=kind,
                        strengths=[item[1] for item in pending],
                        sample_count=args.sample_count,
                        sample_batch_size=args.sample_batch_size,
                        sample_steps=args.sample_steps,
                        t_max=args.sample_t_max,
                        t_min=args.sample_t_min,
                        clip=args.conversion_clip,
                        seed=sample_seed,
                        device=device,
                    )
                    grouped_ambient.update(
                        {item[0]: value for item, value in zip(pending, values)}
                    )
                ambient = grouped_ambient.pop(name)
            else:
                ambient = sample_condition(
                    models=models,
                    oracle=oracle,
                    embedding=embedding,
                    kind=kind,
                    strength=strength,
                    sample_count=args.sample_count,
                    sample_batch_size=args.sample_batch_size,
                    sample_steps=args.sample_steps,
                    t_max=args.sample_t_max,
                    t_min=args.sample_t_min,
                    clip=args.conversion_clip,
                    seed=sample_seed,
                    device=device,
                )
            with torch.inference_mode():
                a = torch.from_numpy(ambient).to(device)
                uu = embedding.decode_intrinsic(a).cpu().numpy()
                manifold_rms = embedding.manifold_consistency_rms(a).mean().item()
            np.savez_compressed(sample_path, intrinsic=uu)
            swd = swd_2d_fixed(
                uu, reference_u, theta=theta, idx_a=idx_s, idx_b=idx_r
            )
            if name == "x":
                mmd_sigma2 = rbf_bandwidth_2d_fixed(
                    uu,
                    reference_u,
                    idx_a=idx_s,
                    idx_b=idx_r,
                    bandwidth_subset=bw_subset,
                )
            if mmd_sigma2 is None:
                raise RuntimeError("x baseline bandwidth must be available")
            mmd = mmd_2d_fixed(
                uu,
                reference_u,
                idx_a=idx_s,
                idx_b=idx_r,
                sigma2=mmd_sigma2,
            )
            if name == "x":
                dmean = dlo = dhi = 0.0
            else:
                if x_intrinsic is None:
                    raise RuntimeError("x baseline must be evaluated before other conditions")
                dmean, dlo, dhi = bootstrap_swd_delta(
                    uu,
                    x_intrinsic,
                    reference_u,
                    theta=theta[: min(args.bootstrap_projections, len(theta))],
                    reps=args.bootstrap_reps,
                    seed=stable_seed(seed, embedding.D, hidden, 993),
                    max_points=min(args.bootstrap_max_points, args.sample_count),
                )
            row = {
                "D": embedding.D,
                "curvature": embedding.curvature,
                "scale_mode": embedding.scale_mode,
                "hidden": hidden,
                "loss_space": loss_space,
                "condition": name,
                "kind": kind,
                "strength": float(strength),
                "swd_2d": swd,
                "mmd_2d": mmd,
                "mmd_sigma2": mmd_sigma2,
                "manifold_consistency_rms": manifold_rms,
                "swd_delta_vs_x_boot_mean": dmean,
                "swd_delta_vs_x_ci_low": dlo,
                "swd_delta_vs_x_ci_high": dhi,
            }
            completed_by_name[name] = row
            ordered_partial = [
                completed_by_name[item_name]
                for item_name, _item_kind, _item_strength in conditions
                if item_name in completed_by_name
            ]
            save_csv(partial_path, ordered_partial)
            del ambient

        if name == "x":
            x_intrinsic = uu
        gen_rows.append(row)
        panels.append((name, uu))
    save_csv(out / "generation_metrics.csv", gen_rows)
    plot_generation_grid(out / "generation_scatter.png", reference_u, panels, args.plot_points)

    # Use time-averaged oracle approximation error as x quality.
    x_oracle_error = float(np.mean([r["x_oracle_approx_mse"] for r in teacher]))
    extrap_candidates = [
        r for r in gen_rows
        if r["condition"] not in {"x", "v", "eps", "oracle"}
    ]
    x_row = next(r for r in gen_rows if r["condition"] == "x")
    best = min(extrap_candidates, key=lambda r: r["swd_2d"])
    setting_summary = {
        "D": embedding.D,
        "curvature": embedding.curvature,
        "scale_mode": embedding.scale_mode,
        "hidden": hidden,
        "loss_space": loss_space,
        "seed": seed,
        "x_oracle_error": x_oracle_error,
        "x_swd": x_row["swd_2d"],
        "best_condition": best["condition"],
        "best_swd": best["swd_2d"],
        "best_relative_swd_vs_x": best["swd_2d"] / max(x_row["swd_2d"], EPS) - 1.0,
        "best_swd_delta_ci_low": best["swd_delta_vs_x_ci_low"],
        "best_swd_delta_ci_high": best["swd_delta_vs_x_ci_high"],
        "mean_cos_xv_oracle_residual": float(
            np.mean([r["cos_xv_oracle_residual"] for r in teacher])
        ),
        "mean_gamma_star_xv_oracle": float(
            np.mean([r["gamma_star_xv_oracle"] for r in teacher])
        ),
        "mean_gap_xv_normal_fraction": float(
            np.mean([
                r["gap_xv_normal_rms"] / max(r["gap_xv_rms"], EPS)
                for r in teacher
            ])
        ),
    }
    (out / "setting_summary.json").write_text(
        json.dumps(setting_summary, indent=2), encoding="utf-8"
    )
    return teacher, gen_rows, setting_summary


def main() -> None:
    args = parse_args()
    if any(D < 2 for D in args.dims):
        raise ValueError("all dims must be >=2")
    if any(h <= 0 for h in args.hidden_dims):
        raise ValueError("hidden dims must be positive")
    if any(c < 0 for c in args.curvatures):
        raise ValueError("curvatures must be non-negative")
    for ls in args.loss_spaces:
        if ls not in {"v", "direct"}:
            raise ValueError(f"unknown loss space {ls}")
    if not (0 < args.t_min < args.t_max < 1):
        raise ValueError("training t range must be inside (0,1)")
    if not (0 < args.sample_t_min < args.sample_t_max < 1):
        raise ValueError("sampling t range must be inside (0,1)")

    set_seed(args.seeds[0])
    device = torch.device(args.device)
    root = args.output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "script": "run_prediction_target_extrapolation_toy_v4.py",
                "definition": "x_t=(1-t)x+t eps; curved known manifold; x-oracle capacity reference",
                "args": {
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(args).items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    all_teacher, all_generation, all_summary = [], [], []

    for seed in args.seeds:
        for D in args.dims:
            for curvature in args.curvatures:
                embedding = CurvedEmbedding(
                    D,
                    curvature=curvature,
                    frequency_scale=args.frequency_scale,
                    seed=stable_seed(seed, D, int(curvature * 10000), 41),
                    device=device,
                    scale_mode=args.scale_mode,
                )

                ref_g = torch.Generator(device=device.type)
                ref_g.manual_seed(stable_seed(seed, D, int(curvature * 10000), 42))
                reference_u = sample_spiral_2d(
                    max(args.sample_count, 20000),
                    device=device,
                    jitter=args.data_jitter,
                    generator=ref_g,
                ).cpu().numpy()

                for loss_space in args.loss_spaces:
                    base_dir = (
                        root
                        / f"seed{seed}"
                        / f"D{D}"
                        / f"curv{tag_float(curvature)}"
                        / f"scale_{args.scale_mode}"
                        / f"loss_{loss_space}"
                    )
                    base_dir.mkdir(parents=True, exist_ok=True)

                    if args.resume:
                        completed_hiddens = {
                            hidden
                            for hidden in args.hidden_dims
                            if (base_dir / f"H{hidden}" / "setting_summary.json").is_file()
                            and (base_dir / f"H{hidden}" / "teacher_metrics.csv").is_file()
                            and (base_dir / f"H{hidden}" / "generation_metrics.csv").is_file()
                        }
                    else:
                        completed_hiddens = set()

                    for hidden in sorted(completed_hiddens):
                        setting_dir = base_dir / f"H{hidden}"
                        all_teacher.extend(load_csv(setting_dir / "teacher_metrics.csv"))
                        all_generation.extend(load_csv(setting_dir / "generation_metrics.csv"))
                        all_summary.append(
                            json.loads(
                                (setting_dir / "setting_summary.json").read_text(
                                    encoding="utf-8"
                                )
                            )
                        )
                        print(
                            f"[resume] seed={seed} D={D} curvature={curvature:g} "
                            f"H={hidden} is complete",
                            flush=True,
                        )

                    pending_hiddens = [
                        hidden for hidden in args.hidden_dims if hidden not in completed_hiddens
                    ]
                    if not pending_hiddens:
                        continue

                    oracle_path = base_dir / "oracle_x.pt"
                    if args.resume and oracle_path.is_file():
                        oracle = DenoiseMLP(
                            D, args.oracle_hidden_dim, args.oracle_depth, args.time_dim
                        ).to(device)
                        oracle.load_state_dict(
                            torch.load(oracle_path, map_location=device, weights_only=True)
                        )
                        print(f"[resume] loaded {oracle_path}", flush=True)
                    else:
                        oracle, oracle_hist = train_oracle_x(
                            embedding=embedding,
                            hidden=args.oracle_hidden_dim,
                            depth=args.oracle_depth,
                            time_dim=args.time_dim,
                            steps=args.oracle_train_steps,
                            batch_size=args.batch_size,
                            lr=args.lr,
                            weight_decay=args.weight_decay,
                            grad_clip=args.grad_clip,
                            loss_space=loss_space,
                            t_min=args.t_min,
                            t_max=args.t_max,
                            conversion_clip=args.conversion_clip,
                            data_jitter=args.data_jitter,
                            log_every=args.log_every,
                            seed=stable_seed(seed, D, int(curvature * 10000), 43),
                            device=device,
                        )
                        save_csv(base_dir / "oracle_train_history.csv", oracle_hist)
                        if args.save_checkpoints:
                            torch.save(oracle.state_dict(), oracle_path)

                    for hidden in pending_hiddens:
                        print(
                            f"\n=== seed={seed} D={D} curvature={curvature:g} "
                            f"H={hidden} loss={loss_space} scale={args.scale_mode} ===",
                            flush=True,
                        )
                        setting_dir = base_dir / f"H{hidden}"
                        setting_dir.mkdir(parents=True, exist_ok=True)
                        model_path = setting_dir / "models.pt"
                        if args.resume and model_path.is_file():
                            states = torch.load(
                                model_path, map_location=device, weights_only=True
                            )
                            models = {
                                target: DenoiseMLP(
                                    D, hidden, args.depth, args.time_dim
                                ).to(device)
                                for target in ("x", "v", "eps")
                            }
                            for target, model in models.items():
                                model.load_state_dict(states[target])
                            bundle = TrainBundle(models=models, history=[])
                            print(f"[resume] loaded {model_path}", flush=True)
                        else:
                            bundle = train_triplet(
                                embedding=embedding,
                                hidden=hidden,
                                depth=args.depth,
                                time_dim=args.time_dim,
                                steps=args.train_steps,
                                batch_size=args.batch_size,
                                lr=args.lr,
                                weight_decay=args.weight_decay,
                                grad_clip=args.grad_clip,
                                loss_space=loss_space,
                                t_min=args.t_min,
                                t_max=args.t_max,
                                conversion_clip=args.conversion_clip,
                                data_jitter=args.data_jitter,
                                log_every=args.log_every,
                                seed=stable_seed(
                                    seed, D, hidden, int(curvature * 10000), 44
                                ),
                                device=device,
                            )
                            save_csv(setting_dir / "train_history.csv", bundle.history)
                            if args.save_checkpoints:
                                torch.save(
                                    {
                                        key: model.state_dict()
                                        for key, model in bundle.models.items()
                                    },
                                    model_path,
                                )

                        teacher, generation, summary = run_setting(
                            args=args,
                            embedding=embedding,
                            oracle=oracle,
                            models=bundle.models,
                            hidden=hidden,
                            loss_space=loss_space,
                            seed=seed,
                            out=setting_dir,
                            reference_u=reference_u,
                            device=device,
                        )
                        all_teacher.extend(teacher)
                        all_generation.extend(generation)
                        all_summary.append(summary)

                        del bundle
                        if device.type == "cuda":
                            torch.cuda.empty_cache()

                    del oracle
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

    save_csv(root / "teacher_metrics_all.csv", all_teacher)
    save_csv(root / "generation_metrics_all.csv", all_generation)
    save_csv(root / "summary_all.csv", all_summary)
    (root / "summary_all.json").write_text(
        json.dumps(all_summary, indent=2), encoding="utf-8"
    )
    plot_phase(root / "extrapolation_phase_diagram.png", all_summary)
    print(f"\nDone. Results written to {root}", flush=True)


if __name__ == "__main__":
    main()
