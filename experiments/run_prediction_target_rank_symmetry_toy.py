#!/usr/bin/env python3
"""Causal output-rank and analytic-skip test on the v4 spiral toy.

This experiment keeps the MLP trunk fixed and changes only an explicit rank-r
output bottleneck.  Native x/v/epsilon heads are trained on identical batches
with a common velocity-space loss.  A fourth model uses the same rank-r network
to predict the clean residual F and analytically composes any native target:

    x_hat       = F
    v_hat       = (z_t - F) / t
    epsilon_hat = z_t + (1 - t) v_hat

All three analytic native outputs recover the same velocity field.  The control
therefore asks whether target differences disappear when the known identity
part is represented by a non-trainable skip instead of the rank-r head.

The time convention matches toy v4:

    z_t = (1 - t) x + t epsilon,  v = epsilon - x,

and generation integrates from t close to 1 back to t close to 0.
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
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.run_prediction_target_extrapolation_toy_v4 import (
    CurvedEmbedding,
    TimeEmbedding,
    clean_from_output,
    direct_target,
    fixed_projection_matrix,
    mmd_2d_fixed,
    parse_float_list,
    parse_int_list,
    rbf_bandwidth_2d_fixed,
    sample_spiral_2d,
    stable_seed,
    swd_2d_fixed,
    tag_float,
    velocity_from_output,
)


TARGETS = ("x", "v", "eps")
CONDITIONS = ("native_x", "native_v", "native_eps", "analytic_skip")


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot save an empty table")
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def stamp_seed_metadata(
    rows: list[dict], *, experiment_seed: int, setting_seed: int
) -> None:
    """Keep the user-facing seed distinct from the derived RNG seed."""
    for row in rows:
        row["seed"] = int(experiment_seed)
        row["setting_seed"] = int(setting_seed)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RankOutputMLP(nn.Module):
    """MLP with fixed trunk capacity and an explicit affine output rank."""

    def __init__(
        self,
        D: int,
        *,
        hidden: int,
        output_rank: int,
        depth: int,
        time_dim: int,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be >= 2")
        if not (1 <= output_rank <= min(hidden, D)):
            raise ValueError("output_rank must be in [1, min(hidden,D)]")
        self.D = int(D)
        self.hidden = int(hidden)
        self.output_rank = int(output_rank)
        self.time = TimeEmbedding(time_dim)
        layers: list[nn.Module] = [nn.Linear(D + time_dim, hidden), nn.SiLU()]
        for _ in range(depth - 2):
            layers.extend([nn.Linear(hidden, hidden), nn.SiLU()])
        self.trunk = nn.Sequential(*layers)
        self.to_rank = nn.Linear(hidden, output_rank, bias=False)
        self.output = nn.Linear(output_rank, D, bias=True)

    def forward(self, state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        hidden = self.trunk(torch.cat([state, self.time(time)], dim=1))
        return self.output(self.to_rank(hidden))


def build_matched_models(
    *,
    D: int,
    hidden: int,
    output_rank: int,
    depth: int,
    time_dim: int,
    seed: int,
    device: torch.device,
) -> dict[str, RankOutputMLP]:
    torch.manual_seed(seed)
    base = RankOutputMLP(
        D,
        hidden=hidden,
        output_rank=output_rank,
        depth=depth,
        time_dim=time_dim,
    ).to(device)
    state = copy.deepcopy(base.state_dict())
    models = {}
    for condition in CONDITIONS:
        model = RankOutputMLP(
            D,
            hidden=hidden,
            output_rank=output_rank,
            depth=depth,
            time_dim=time_dim,
        ).to(device)
        model.load_state_dict(state)
        models[condition] = model
    return models


def native_target_for_condition(condition: str) -> str:
    if condition.startswith("native_"):
        return condition.removeprefix("native_")
    if condition == "analytic_skip":
        return "x"
    raise ValueError(condition)


def native_to_velocity_gain(condition: str, time: float) -> float:
    """Jacobian norm multiplier from native output error to velocity error."""
    if condition in {"native_x", "analytic_skip"}:
        return 1.0 / time
    if condition == "native_v":
        return 1.0
    if condition == "native_eps":
        return 1.0 / (1.0 - time)
    raise ValueError(condition)


def analytic_native_output(
    clean_residual: torch.Tensor,
    state: torch.Tensor,
    time: torch.Tensor,
    target: str,
    clip: float,
) -> torch.Tensor:
    """Compose a requested native target from a rank-limited clean residual."""
    tc = time[:, None]
    velocity = (state - clean_residual) / tc.clamp_min(clip)
    if target == "x":
        return clean_residual
    if target == "v":
        return velocity
    if target == "eps":
        return state + (1.0 - tc) * velocity
    raise ValueError(target)


def condition_predictions(
    *,
    model: RankOutputMLP,
    condition: str,
    state: torch.Tensor,
    time: torch.Tensor,
    clip: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return native output, recovered velocity, and recovered clean point."""
    raw = model(state, time)
    if condition == "analytic_skip":
        native = analytic_native_output(raw, state, time, "x", clip)
        velocity = (state - raw) / time[:, None].clamp_min(clip)
        clean = raw
        return native, velocity, clean

    target = native_target_for_condition(condition)
    velocity = velocity_from_output(raw, state, time, target, clip)
    clean = clean_from_output(raw, state, time, target, clip)
    return raw, velocity, clean


def analytic_skip_velocity_via_target(
    clean_residual: torch.Tensor,
    state: torch.Tensor,
    time: torch.Tensor,
    target: str,
    clip: float,
) -> torch.Tensor:
    native = analytic_native_output(clean_residual, state, time, target, clip)
    return velocity_from_output(native, state, time, target, clip)


@dataclass
class TrainResult:
    models: dict[str, RankOutputMLP]
    history: list[dict]
    x_skip_parameter_max_abs: float


def train_models(
    *,
    embedding: CurvedEmbedding,
    output_rank: int,
    hidden: int,
    depth: int,
    time_dim: int,
    steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    t_min: float,
    t_max: float,
    conversion_clip: float,
    data_jitter: float,
    log_every: int,
    seed: int,
    device: torch.device,
    amp_dtype: str,
    rank_dependent_randomness: bool = True,
) -> TrainResult:
    models = build_matched_models(
        D=embedding.D,
        hidden=hidden,
        output_rank=output_rank,
        depth=depth,
        time_dim=time_dim,
        seed=seed,
        device=device,
    )
    optimizers = {
        name: torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        for name, model in models.items()
    }
    generator = torch.Generator(device=device.type)
    rank_seed = output_rank if rank_dependent_randomness else 0
    generator.manual_seed(stable_seed(seed, embedding.D, rank_seed, 701))
    use_amp = amp_dtype != "fp32" and device.type == "cuda"
    autocast_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    history: list[dict] = []

    for step in range(1, steps + 1):
        intrinsic = sample_spiral_2d(
            batch_size,
            device=device,
            jitter=data_jitter,
            generator=generator,
        )
        clean = embedding.embed(intrinsic)
        eps = torch.randn(clean.shape, device=device, generator=generator)
        time = torch.empty(batch_size, device=device).uniform_(t_min, t_max, generator=generator)
        state = (1.0 - time[:, None]) * clean + time[:, None] * eps
        true_velocity = eps - clean
        losses: dict[str, float] = {}

        for condition in CONDITIONS:
            model = models[condition]
            optimizer = optimizers[condition]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=use_amp,
            ):
                _native, predicted_velocity, _predicted_clean = condition_predictions(
                    model=model,
                    condition=condition,
                    state=state,
                    time=time,
                    clip=conversion_clip,
                )
                loss = F.mse_loss(predicted_velocity.float(), true_velocity.float())
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            losses[condition] = float(loss.detach().cpu())

        if step == 1 or step % log_every == 0 or step == steps:
            row = {"step": step, **{f"loss_{name}": value for name, value in losses.items()}}
            history.append(row)
            print(
                f"[train D={embedding.D} C={embedding.curvature:g} R={output_rank}] "
                f"{step}/{steps} "
                + " ".join(f"{name}={losses[name]:.5g}" for name in CONDITIONS),
                flush=True,
            )

    max_abs = 0.0
    for x_value, skip_value in zip(
        models["native_x"].parameters(), models["analytic_skip"].parameters()
    ):
        max_abs = max(max_abs, float((x_value - skip_value).abs().max().detach().cpu()))
    return TrainResult(models=models, history=history, x_skip_parameter_max_abs=max_abs)


def covariance_effective_rank(values: torch.Tensor) -> tuple[float, int, float]:
    centered = values.double() - values.double().mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    total = eigenvalues.sum().clamp_min(torch.finfo(eigenvalues.dtype).tiny)
    effective = total.square() / eigenvalues.square().sum().clamp_min(torch.finfo(eigenvalues.dtype).tiny)
    numerical = int((eigenvalues > eigenvalues.max() * 1e-6).sum().item())
    return float(effective), numerical, float(total / len(eigenvalues))


@torch.inference_mode()
def evaluate_teacher(
    *,
    models: dict[str, RankOutputMLP],
    embedding: CurvedEmbedding,
    output_rank: int,
    times: Iterable[float],
    samples: int,
    batch_size: int,
    data_jitter: float,
    conversion_clip: float,
    seed: int,
    device: torch.device,
    rank_dependent_randomness: bool = True,
) -> list[dict]:
    rows: list[dict] = []
    for time_index, time_value in enumerate(times):
        accumulator = {
            condition: {
                "velocity_squared": 0.0,
                "clean_squared": 0.0,
                "native_squared": 0.0,
                "tangent_squared": 0.0,
                "normal_squared": 0.0,
                "native_outputs": [],
            }
            for condition in CONDITIONS
        }
        generator = torch.Generator(device=device.type)
        rank_seed = output_rank if rank_dependent_randomness else 0
        generator.manual_seed(
            stable_seed(seed, embedding.D, rank_seed, time_index, 809)
        )
        n_total = 0
        skip_conversion_max_abs = 0.0
        for start in range(0, samples, batch_size):
            n = min(batch_size, samples - start)
            intrinsic = sample_spiral_2d(
                n,
                device=device,
                jitter=data_jitter,
                generator=generator,
            )
            clean = embedding.embed(intrinsic)
            eps = torch.randn(clean.shape, device=device, generator=generator)
            time = torch.full((n,), float(time_value), device=device)
            state = (1.0 - time[:, None]) * clean + time[:, None] * eps
            true_velocity = eps - clean
            tangent_basis = embedding.tangent_basis(intrinsic)

            skip_raw = models["analytic_skip"](state, time)
            skip_velocities = [
                analytic_skip_velocity_via_target(
                    skip_raw, state, time, target, conversion_clip
                )
                for target in TARGETS
            ]
            for value in skip_velocities[1:]:
                skip_conversion_max_abs = max(
                    skip_conversion_max_abs,
                    float((value - skip_velocities[0]).abs().max().cpu()),
                )

            for condition, model in models.items():
                native, velocity, predicted_clean = condition_predictions(
                    model=model,
                    condition=condition,
                    state=state,
                    time=time,
                    clip=conversion_clip,
                )
                error = velocity - true_velocity
                tangent = torch.einsum(
                    "bdi,bi->bd",
                    tangent_basis,
                    torch.einsum("bdi,bd->bi", tangent_basis, error),
                )
                normal = error - tangent
                target = native_target_for_condition(condition)
                native_truth = direct_target(clean, eps, target)
                values = accumulator[condition]
                values["velocity_squared"] += float(error.square().sum().cpu())
                values["clean_squared"] += float((predicted_clean - clean).square().sum().cpu())
                values["native_squared"] += float((native - native_truth).square().sum().cpu())
                values["tangent_squared"] += float(tangent.square().sum().cpu())
                values["normal_squared"] += float(normal.square().sum().cpu())
                values["native_outputs"].append(native.cpu())
            n_total += n

        denominator = n_total * embedding.D
        for condition in CONDITIONS:
            values = accumulator[condition]
            effective_rank, numerical_rank, output_variance = covariance_effective_rank(
                torch.cat(values.pop("native_outputs"), dim=0)
            )
            rows.append(
                {
                    "seed": seed,
                    "D": embedding.D,
                    "curvature": embedding.curvature,
                    "output_rank": output_rank,
                    "time": float(time_value),
                    "condition": condition,
                    "native_to_velocity_gain": native_to_velocity_gain(
                        condition, float(time_value)
                    ),
                    "native_to_velocity_squared_gain": native_to_velocity_gain(
                        condition, float(time_value)
                    )
                    ** 2,
                    "velocity_mse": values["velocity_squared"] / denominator,
                    "clean_mse": values["clean_squared"] / denominator,
                    "native_target_mse": values["native_squared"] / denominator,
                    "velocity_error_tangent_mse": values["tangent_squared"] / denominator,
                    "velocity_error_normal_mse": values["normal_squared"] / denominator,
                    "native_output_effective_rank": effective_rank,
                    "native_output_numerical_rank": numerical_rank,
                    "native_output_variance_per_dim": output_variance,
                    "skip_conversion_max_abs": skip_conversion_max_abs,
                }
            )
    return rows


@torch.inference_mode()
def sample_models(
    *,
    models: dict[str, RankOutputMLP],
    embedding: CurvedEmbedding,
    count: int,
    batch_size: int,
    steps: int,
    t_max: float,
    t_min: float,
    conversion_clip: float,
    seed: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    collected = {condition: [] for condition in CONDITIONS}
    grid = torch.linspace(t_max, t_min, steps + 1, device=device)
    for start in range(0, count, batch_size):
        n = min(batch_size, count - start)
        generator = torch.Generator(device=device.type)
        generator.manual_seed(seed + start)
        initial = float(t_max) * torch.randn((n, embedding.D), device=device, generator=generator)
        states = {condition: initial.clone() for condition in CONDITIONS}
        for index in range(steps):
            t_now, t_next = grid[index], grid[index + 1]
            time = t_now.expand(n)
            for condition in CONDITIONS:
                _native, velocity, _clean = condition_predictions(
                    model=models[condition],
                    condition=condition,
                    state=states[condition],
                    time=time,
                    clip=conversion_clip,
                )
                states[condition] = states[condition] + (t_next - t_now) * velocity

        final_time = grid[-1].expand(n)
        for condition in CONDITIONS:
            _native, _velocity, clean = condition_predictions(
                model=models[condition],
                condition=condition,
                state=states[condition],
                time=final_time,
                clip=conversion_clip,
            )
            collected[condition].append(clean.cpu().numpy())
    return {condition: np.concatenate(parts) for condition, parts in collected.items()}


def evaluate_generation(
    *,
    samples: dict[str, np.ndarray],
    reference_intrinsic: np.ndarray,
    embedding: CurvedEmbedding,
    output_rank: int,
    seed: int,
    device: torch.device,
    metric_max_points: int,
    projections: int,
    rank_dependent_randomness: bool = True,
) -> list[dict]:
    count = len(next(iter(samples.values())))
    n_metric = min(count, len(reference_intrinsic) // 2, metric_max_points)
    rank_seed = output_rank if rank_dependent_randomness else 0
    rng = np.random.default_rng(stable_seed(seed, embedding.D, rank_seed, 907))
    idx_sample = rng.choice(count, n_metric, replace=False)
    idx_reference = rng.choice(len(reference_intrinsic) // 2, n_metric, replace=False)
    reference_a = reference_intrinsic[: len(reference_intrinsic) // 2]
    reference_b = reference_intrinsic[len(reference_intrinsic) // 2 :]
    with torch.inference_mode():
        reference_ambient = (
            embedding.embed(torch.from_numpy(reference_a).to(device)).cpu().numpy()
        )
    idx_reference_b = rng.choice(len(reference_b), n_metric, replace=False)
    theta = fixed_projection_matrix(
        projections, stable_seed(seed, embedding.D, rank_seed, 911)
    )
    ambient_rng = np.random.default_rng(
        stable_seed(seed, embedding.D, rank_seed, 912)
    )
    ambient_theta = ambient_rng.normal(size=(embedding.D, projections))
    ambient_theta /= np.linalg.norm(ambient_theta, axis=0, keepdims=True) + 1e-12
    bandwidth_subset = rng.choice(2 * n_metric, min(1024, 2 * n_metric), replace=False)
    sigma2 = rbf_bandwidth_2d_fixed(
        reference_a,
        reference_b,
        idx_a=idx_reference,
        idx_b=idx_reference_b,
        bandwidth_subset=bandwidth_subset,
    )

    rows = []
    for condition, ambient in samples.items():
        tensor = torch.from_numpy(ambient).to(device)
        intrinsic = embedding.decode_intrinsic(tensor).cpu().numpy()
        manifold_rms = float(embedding.manifold_consistency_rms(tensor).mean().cpu())
        rows.append(
            {
                "seed": seed,
                "D": embedding.D,
                "curvature": embedding.curvature,
                "output_rank": output_rank,
                "condition": condition,
                "swd_2d": swd_2d_fixed(
                    intrinsic,
                    reference_a,
                    theta=theta,
                    idx_a=idx_sample,
                    idx_b=idx_reference,
                ),
                "swd_ambient": float(
                    np.mean(
                        np.sqrt(
                            np.mean(
                                (
                                    np.sort(ambient[idx_sample] @ ambient_theta, axis=0)
                                    - np.sort(
                                        reference_ambient[idx_reference] @ ambient_theta,
                                        axis=0,
                                    )
                                )
                                ** 2,
                                axis=0,
                            )
                        )
                    )
                ),
                "mmd_2d": mmd_2d_fixed(
                    intrinsic,
                    reference_a,
                    idx_a=idx_sample,
                    idx_b=idx_reference,
                    sigma2=sigma2,
                ),
                "mmd_sigma2": sigma2,
                "manifold_consistency_rms": manifold_rms,
            }
        )
    return rows


def plot_generation(
    path: Path,
    samples: dict[str, np.ndarray],
    reference_intrinsic: np.ndarray,
    embedding: CurvedEmbedding,
    max_points: int,
) -> None:
    panels = [("reference", reference_intrinsic)]
    with torch.inference_mode():
        for condition, ambient in samples.items():
            tensor = torch.from_numpy(ambient).to(embedding.device)
            panels.append((condition, embedding.decode_intrinsic(tensor).cpu().numpy()))
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4), squeeze=False)
    for axis, (name, values) in zip(axes[0], panels):
        values = values[:max_points]
        axis.scatter(values[:, 0], values[:, 1], s=2, alpha=0.45)
        axis.set_title(name)
        axis.set_aspect("equal")
        axis.set_xlim(-1.9, 1.9)
        axis.set_ylim(-1.9, 1.9)
        axis.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def output_head_rank(model: RankOutputMLP) -> int:
    return int(torch.linalg.matrix_rank(model.output.weight.float()).item())


def run_setting(
    *,
    args: argparse.Namespace,
    D: int,
    curvature: float,
    output_rank: int,
    seed: int,
    device: torch.device,
) -> tuple[list[dict], list[dict], dict]:
    rank_seed = output_rank if args.rank_dependent_randomness else 0
    setting_seed = stable_seed(seed, D, int(curvature * 10000), rank_seed, 1009)
    embedding = CurvedEmbedding(
        D,
        curvature=curvature,
        frequency_scale=args.frequency_scale,
        seed=stable_seed(seed, D, int(curvature * 10000), 41),
        device=device,
        scale_mode=args.scale_mode,
    )
    setting_dir = (
        args.output_root
        / f"seed{seed}"
        / f"D{D}"
        / f"curv{tag_float(curvature)}"
        / f"rank{output_rank}"
    )
    if args.resume and (setting_dir / "summary.json").is_file():
        print(f"[resume] {setting_dir} already complete", flush=True)
        with (setting_dir / "teacher_metrics.csv").open(newline="", encoding="utf-8") as handle:
            teacher = list(csv.DictReader(handle))
        with (setting_dir / "generation_metrics.csv").open(newline="", encoding="utf-8") as handle:
            generation = list(csv.DictReader(handle))
        stamp_seed_metadata(
            teacher, experiment_seed=seed, setting_seed=setting_seed
        )
        stamp_seed_metadata(
            generation, experiment_seed=seed, setting_seed=setting_seed
        )
        summary = json.loads((setting_dir / "summary.json").read_text(encoding="utf-8"))
        return teacher, generation, summary
    setting_dir.mkdir(parents=True, exist_ok=True)

    result = train_models(
        embedding=embedding,
        output_rank=output_rank,
        hidden=args.hidden,
        depth=args.depth,
        time_dim=args.time_dim,
        steps=args.train_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        t_min=args.t_min,
        t_max=args.t_max,
        conversion_clip=args.conversion_clip,
        data_jitter=args.data_jitter,
        log_every=args.log_every,
        seed=setting_seed,
        device=device,
        amp_dtype=args.amp_dtype,
        rank_dependent_randomness=args.rank_dependent_randomness,
    )
    save_csv(setting_dir / "train_history.csv", result.history)

    teacher = evaluate_teacher(
        models=result.models,
        embedding=embedding,
        output_rank=output_rank,
        times=args.eval_times,
        samples=args.eval_samples,
        batch_size=args.eval_batch_size,
        data_jitter=args.data_jitter,
        conversion_clip=args.conversion_clip,
        seed=setting_seed,
        device=device,
        rank_dependent_randomness=args.rank_dependent_randomness,
    )
    stamp_seed_metadata(teacher, experiment_seed=seed, setting_seed=setting_seed)
    save_csv(setting_dir / "teacher_metrics.csv", teacher)

    reference_generator = torch.Generator(device=device.type)
    reference_generator.manual_seed(stable_seed(setting_seed, 1201))
    reference_intrinsic = sample_spiral_2d(
        max(2 * args.sample_count, 8192),
        device=device,
        jitter=args.data_jitter,
        generator=reference_generator,
    ).cpu().numpy()
    generated = sample_models(
        models=result.models,
        embedding=embedding,
        count=args.sample_count,
        batch_size=args.sample_batch_size,
        steps=args.sample_steps,
        t_max=args.sample_t_max,
        t_min=args.sample_t_min,
        conversion_clip=args.conversion_clip,
        seed=stable_seed(setting_seed, 1213),
        device=device,
    )
    generation = evaluate_generation(
        samples=generated,
        reference_intrinsic=reference_intrinsic,
        embedding=embedding,
        output_rank=output_rank,
        seed=setting_seed,
        device=device,
        metric_max_points=args.metric_max_points,
        projections=args.swd_projections,
        rank_dependent_randomness=args.rank_dependent_randomness,
    )
    stamp_seed_metadata(generation, experiment_seed=seed, setting_seed=setting_seed)
    save_csv(setting_dir / "generation_metrics.csv", generation)
    plot_generation(
        setting_dir / "generation_scatter.png",
        generated,
        reference_intrinsic,
        embedding,
        args.plot_points,
    )
    if args.save_checkpoints:
        torch.save(
            {condition: model.state_dict() for condition, model in result.models.items()},
            setting_dir / "models.pt",
        )

    native_rows = [row for row in generation if row["condition"].startswith("native_")]
    best_native = min(native_rows, key=lambda row: float(row["swd_2d"]))
    mean_teacher = {
        condition: float(
            np.mean([float(row["velocity_mse"]) for row in teacher if row["condition"] == condition])
        )
        for condition in CONDITIONS
    }
    summary = {
        "seed": seed,
        "D": D,
        "curvature": curvature,
        "output_rank": output_rank,
        "hidden": args.hidden,
        "train_steps": args.train_steps,
        "rank_dependent_randomness": args.rank_dependent_randomness,
        "x_skip_parameter_max_abs": result.x_skip_parameter_max_abs,
        "actual_output_head_rank": {
            condition: output_head_rank(model) for condition, model in result.models.items()
        },
        "mean_teacher_velocity_mse": mean_teacher,
        "best_native_generation_target": best_native["condition"],
        "best_native_swd": float(best_native["swd_2d"]),
        "generation_swd": {
            row["condition"]: float(row["swd_2d"]) for row in generation
        },
    }
    (setting_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return teacher, generation, summary


def plot_phase(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    conditions = CONDITIONS
    colors = {
        "native_x": "#1f77b4",
        "native_v": "#2ca02c",
        "native_eps": "#d62728",
        "analytic_skip": "#9467bd",
    }
    settings = sorted({(int(row["D"]), float(row["curvature"])) for row in rows})
    fig, axes = plt.subplots(len(settings), 1, figsize=(8, 4 * len(settings)), squeeze=False)
    for axis, (D, curvature) in zip(axes[:, 0], settings):
        subset = [row for row in rows if int(row["D"]) == D and float(row["curvature"]) == curvature]
        ranks = sorted({int(row["output_rank"]) for row in subset})
        for condition in conditions:
            values = []
            for rank in ranks:
                matches = [
                    float(row["swd_2d"])
                    for row in subset
                    if int(row["output_rank"]) == rank and row["condition"] == condition
                ]
                values.append(float(np.mean(matches)))
            axis.plot(ranks, values, marker="o", label=condition, color=colors[condition])
        axis.set_xscale("log", base=2)
        axis.set_xlabel("explicit output rank r")
        axis.set_ylabel("intrinsic SWD")
        axis.set_title(f"D={D}, curvature={curvature}")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dims", type=parse_int_list, default=parse_int_list("64,512"))
    parser.add_argument("--curvatures", type=parse_float_list, default=parse_float_list("0,0.5"))
    parser.add_argument("--output-ranks", type=parse_int_list, default=parse_int_list("4,16,64,256"))
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--time-dim", type=int, default=32)
    parser.add_argument("--frequency-scale", type=float, default=6.0)
    parser.add_argument("--scale-mode", choices=("constant_norm", "unit_rms"), default="unit_rms")
    parser.add_argument("--train-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--amp-dtype", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--t-min", type=float, default=0.02)
    parser.add_argument("--t-max", type=float, default=0.98)
    parser.add_argument("--conversion-clip", type=float, default=0.02)
    parser.add_argument("--data-jitter", type=float, default=0.015)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--eval-times", type=parse_float_list, default=parse_float_list("0.1,0.3,0.5,0.7,0.9"))
    parser.add_argument("--eval-samples", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--sample-count", type=int, default=4096)
    parser.add_argument("--sample-batch-size", type=int, default=512)
    parser.add_argument("--sample-steps", type=int, default=100)
    parser.add_argument("--sample-t-max", type=float, default=0.98)
    parser.add_argument("--sample-t-min", type=float, default=0.02)
    parser.add_argument("--metric-max-points", type=int, default=4096)
    parser.add_argument("--swd-projections", type=int, default=256)
    parser.add_argument("--plot-points", type=int, default=3000)
    parser.add_argument("--seeds", type=parse_int_list, default=parse_int_list("20260821"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument(
        "--rank-dependent-randomness",
        action="store_true",
        help=(
            "use the legacy v1 protocol in which rank changes initialization, "
            "training batches, sampling noise, and metric projections; the default "
            "pairs all of these across ranks"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if not (0 < args.t_min < args.t_max < 1):
        raise ValueError("training times must be strictly inside (0,1)")
    if not (0 < args.sample_t_min < args.sample_t_max < 1):
        raise ValueError("sampling times must be strictly inside (0,1)")
    device = torch.device(args.device)
    set_seed(args.seeds[0])
    manifest = {
        "definition": "fixed-trunk explicit-output-rank target symmetry test",
        "path": "z_t=(1-t)x+t epsilon; v=epsilon-x",
        "loss": "common recovered-velocity MSE",
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    all_teacher: list[dict] = []
    all_generation: list[dict] = []
    all_summaries: list[dict] = []
    for seed in args.seeds:
        for D in args.dims:
            for curvature in args.curvatures:
                for output_rank in args.output_ranks:
                    if output_rank > min(args.hidden, D):
                        continue
                    teacher, generation, summary = run_setting(
                        args=args,
                        D=D,
                        curvature=curvature,
                        output_rank=output_rank,
                        seed=seed,
                        device=device,
                    )
                    all_teacher.extend(teacher)
                    all_generation.extend(generation)
                    all_summaries.append(summary)

    save_csv(args.output_root / "teacher_metrics.csv", all_teacher)
    save_csv(args.output_root / "generation_metrics.csv", all_generation)
    save_csv(args.output_root / "setting_summaries.csv", all_summaries)
    plot_phase(all_generation, args.output_root / "rank_phase_swd.png")
    print(f"[done] completed {len(all_summaries)} settings at {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
