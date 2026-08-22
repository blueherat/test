#!/usr/bin/env python3
"""Audit spectral target formulas under an explicit output-rank bottleneck.

The experiment follows the prediction-target toy convention

    z_t = (1 - t) x + t epsilon,    v = epsilon - x,

and compares every condition with the same recovered-velocity MSE.  It keeps
the forward path fixed and changes only the affine output parameterization.

Two analytically motivated conditions are included:

* k-Diff spectrum: k_i = 1 / (1 + lambda_i), applied per covariance mode.
* LMMSE residual: remove the best affine velocity predictor from z_t and let
  the rank-limited network predict a unit-variance residual in active modes.

The latter can optionally whiten the noisy input.  Keeping both variants
separates output residualization from input preconditioning.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.run_prediction_target_extrapolation_toy_v4 import (
    CurvedEmbedding,
    parse_float_list,
    parse_int_list,
    sample_spiral_2d,
    stable_seed,
)
from experiments.run_prediction_target_rank_symmetry_toy import (
    RankOutputMLP,
    covariance_effective_rank,
    evaluate_generation,
    save_csv,
    set_seed,
)


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    method: str
    input_mode: str = "raw"

    def __post_init__(self) -> None:
        if self.method not in {
            "native_x",
            "native_v",
            "legacy_scalar",
            "centered_scalar",
            "fixed_operator",
            "kdiff_spectrum",
            "subspace_velocity",
            "lmmse_residual",
        }:
            raise ValueError(f"unknown method: {self.method}")
        if self.input_mode not in {
            "raw",
            "full_whitened",
            "active_whitened",
            "projected_raw",
            "projected_whitened",
        }:
            raise ValueError(f"unknown input mode: {self.input_mode}")
        if self.input_mode != "raw" and self.method != "lmmse_residual":
            raise ValueError("input transforms are only defined for LMMSE residuals")


@dataclass(frozen=True)
class SpectralStats:
    mean: torch.Tensor
    basis: torch.Tensor
    eigenvalues: torch.Tensor
    active: torch.Tensor
    threshold: float
    samples: int

    @property
    def D(self) -> int:
        return int(self.mean.numel())


@dataclass(frozen=True)
class Prediction:
    velocity: torch.Tensor
    native_output: torch.Tensor
    velocity_residual: torch.Tensor


CONDITIONS = (
    ConditionSpec("native_x", "native_x"),
    ConditionSpec("native_v", "native_v"),
    ConditionSpec("legacy_scalar_k0p9", "legacy_scalar"),
    ConditionSpec("centered_scalar_k0p9", "centered_scalar"),
    ConditionSpec("fixed_operator_t050_n0p9", "fixed_operator"),
    ConditionSpec("kdiff_spectrum", "kdiff_spectrum"),
    ConditionSpec("subspace_velocity_raw_input", "subspace_velocity"),
    ConditionSpec("lmmse_residual_raw_input", "lmmse_residual"),
    ConditionSpec(
        "lmmse_residual_whitened_input",
        "lmmse_residual",
        input_mode="full_whitened",
    ),
    ConditionSpec(
        "lmmse_residual_active_whitened_input",
        "lmmse_residual",
        input_mode="active_whitened",
    ),
    ConditionSpec(
        "lmmse_residual_projected_raw_input",
        "lmmse_residual",
        input_mode="projected_raw",
    ),
    ConditionSpec(
        "lmmse_residual_projected_whitened_input",
        "lmmse_residual",
        input_mode="projected_whitened",
    ),
)


def to_modes(value: torch.Tensor, stats: SpectralStats) -> torch.Tensor:
    return value @ stats.basis


def from_modes(value: torch.Tensor, stats: SpectralStats) -> torch.Tensor:
    return value @ stats.basis.T


def centered_state(
    state: torch.Tensor, time: torch.Tensor, stats: SpectralStats
) -> torch.Tensor:
    return state - (1.0 - time[:, None]) * stats.mean[None]


def estimate_spectral_stats(
    *,
    embedding: CurvedEmbedding,
    samples: int,
    batch_size: int,
    data_jitter: float,
    seed: int,
    relative_floor: float,
    absolute_floor: float,
) -> SpectralStats:
    if samples < 2:
        raise ValueError("covariance estimation requires at least two samples")
    generator = torch.Generator(device=embedding.device.type)
    generator.manual_seed(seed)
    total = torch.zeros(embedding.D, device=embedding.device, dtype=torch.float64)
    second = torch.zeros(
        embedding.D, embedding.D, device=embedding.device, dtype=torch.float64
    )
    seen = 0
    for start in range(0, samples, batch_size):
        count = min(batch_size, samples - start)
        intrinsic = sample_spiral_2d(
            count,
            device=embedding.device,
            jitter=data_jitter,
            generator=generator,
        )
        clean = embedding.embed(intrinsic).double()
        total += clean.sum(dim=0)
        second += clean.T @ clean
        seen += count
    mean64 = total / seen
    covariance = (second - seen * torch.outer(mean64, mean64)) / (seen - 1)
    eigenvalues64, basis64 = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues64, descending=True)
    eigenvalues64 = eigenvalues64[order].clamp_min(0.0)
    basis64 = basis64[:, order]
    threshold = max(
        float(absolute_floor),
        float(relative_floor) * float(eigenvalues64[0].item()),
    )
    active = eigenvalues64 > threshold
    eigenvalues64 = torch.where(active, eigenvalues64, torch.zeros_like(eigenvalues64))
    return SpectralStats(
        mean=mean64.to(dtype=torch.float32),
        basis=basis64.to(dtype=torch.float32),
        eigenvalues=eigenvalues64.to(dtype=torch.float32),
        active=active,
        threshold=threshold,
        samples=seen,
    )


def constant_k_values(spec: ConditionSpec, stats: SpectralStats) -> torch.Tensor:
    if spec.method in {"legacy_scalar", "centered_scalar"}:
        return torch.full_like(stats.eigenvalues, 0.9)
    if spec.method == "fixed_operator":
        return torch.where(
            stats.active,
            torch.full_like(stats.eigenvalues, 0.5),
            torch.full_like(stats.eigenvalues, 0.9),
        )
    if spec.method == "kdiff_spectrum":
        return 1.0 / (1.0 + stats.eigenvalues)
    raise ValueError(f"{spec.name} does not use a constant K")


def constant_denominator(
    time: torch.Tensor, k_values: torch.Tensor, clip: float
) -> torch.Tensor:
    denominator = (
        1.0
        - time[:, None]
        + (2.0 * time[:, None] - 1.0) * k_values[None]
    )
    return denominator.clamp_min(clip)


def centered_constant_target(
    clean: torch.Tensor,
    epsilon: torch.Tensor,
    stats: SpectralStats,
    k_values: torch.Tensor,
) -> torch.Tensor:
    clean_modes = to_modes(clean - stats.mean[None], stats)
    epsilon_modes = to_modes(epsilon, stats)
    target_modes = (
        k_values[None] * clean_modes
        - (1.0 - k_values[None]) * epsilon_modes
    )
    return from_modes(target_modes, stats)


def velocity_from_centered_constant_output(
    output: torch.Tensor,
    state: torch.Tensor,
    time: torch.Tensor,
    stats: SpectralStats,
    k_values: torch.Tensor,
    clip: float,
) -> torch.Tensor:
    state_modes = to_modes(centered_state(state, time, stats), stats)
    output_modes = to_modes(output, stats)
    denominator = constant_denominator(time, k_values, clip)
    centered_velocity_modes = (
        (2.0 * k_values[None] - 1.0) * state_modes - output_modes
    ) / denominator
    return from_modes(centered_velocity_modes, stats) - stats.mean[None]


def linear_path_moments(
    time: torch.Tensor,
    stats: SpectralStats,
    clip: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return state variance, affine velocity coefficient, residual scale."""
    eigenvalues = stats.eigenvalues[None]
    one_minus_t = 1.0 - time[:, None]
    state_variance = one_minus_t.square() * eigenvalues + time[:, None].square()
    state_variance = state_variance.clamp_min(clip * clip)
    velocity_coefficient = (
        time[:, None] - one_minus_t * eigenvalues
    ) / state_variance
    residual_scale = torch.sqrt(eigenvalues / state_variance)
    residual_scale = torch.where(
        stats.active[None], residual_scale, torch.zeros_like(residual_scale)
    )
    return state_variance, velocity_coefficient, residual_scale


def lmmse_model_input(
    state: torch.Tensor,
    time: torch.Tensor,
    stats: SpectralStats,
    *,
    input_mode: str,
    clip: float,
) -> torch.Tensor:
    value = centered_state(state, time, stats)
    if input_mode == "raw":
        return value
    state_variance, _, _ = linear_path_moments(time, stats, clip)
    modes = to_modes(value, stats)
    whitened = modes / state_variance.sqrt()
    if input_mode == "full_whitened":
        transformed = whitened
    elif input_mode == "active_whitened":
        transformed = torch.where(stats.active[None], whitened, modes)
    elif input_mode == "projected_raw":
        transformed = torch.where(stats.active[None], modes, torch.zeros_like(modes))
    elif input_mode == "projected_whitened":
        transformed = torch.where(
            stats.active[None], whitened, torch.zeros_like(modes)
        )
    else:
        raise ValueError(f"unknown input mode: {input_mode}")
    return from_modes(transformed, stats)


def subspace_velocity_target(
    clean: torch.Tensor,
    epsilon: torch.Tensor,
    stats: SpectralStats,
) -> torch.Tensor:
    centered_velocity_modes = to_modes(
        epsilon - clean + stats.mean[None], stats
    )
    active_velocity_modes = torch.where(
        stats.active[None],
        centered_velocity_modes,
        torch.zeros_like(centered_velocity_modes),
    )
    return from_modes(active_velocity_modes, stats)


def velocity_from_subspace_output(
    output: torch.Tensor,
    state: torch.Tensor,
    time: torch.Tensor,
    stats: SpectralStats,
    clip: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    output_modes = to_modes(output, stats)
    active_output_modes = torch.where(
        stats.active[None], output_modes, torch.zeros_like(output_modes)
    )
    state_modes = to_modes(centered_state(state, time, stats), stats)
    normal_velocity_modes = state_modes / time[:, None].clamp_min(clip)
    centered_velocity_modes = torch.where(
        stats.active[None], active_output_modes, normal_velocity_modes
    )
    effective_output = from_modes(active_output_modes, stats)
    return from_modes(centered_velocity_modes, stats) - stats.mean[None], effective_output


def lmmse_residual_target(
    clean: torch.Tensor,
    epsilon: torch.Tensor,
    state: torch.Tensor,
    time: torch.Tensor,
    stats: SpectralStats,
    clip: float,
) -> torch.Tensor:
    state_modes = to_modes(centered_state(state, time, stats), stats)
    centered_velocity_modes = to_modes(
        epsilon - clean + stats.mean[None], stats
    )
    _, velocity_coefficient, residual_scale = linear_path_moments(time, stats, clip)
    residual_modes = centered_velocity_modes - velocity_coefficient * state_modes
    normalized = torch.where(
        stats.active[None],
        residual_modes / residual_scale.clamp_min(clip),
        torch.zeros_like(residual_modes),
    )
    return from_modes(normalized, stats)


def velocity_from_lmmse_output(
    output: torch.Tensor,
    state: torch.Tensor,
    time: torch.Tensor,
    stats: SpectralStats,
    clip: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state_modes = to_modes(centered_state(state, time, stats), stats)
    output_modes = to_modes(output, stats)
    _, velocity_coefficient, residual_scale = linear_path_moments(time, stats, clip)
    linear_modes = velocity_coefficient * state_modes
    residual_modes = residual_scale * output_modes
    centered_velocity = from_modes(linear_modes + residual_modes, stats)
    effective_output = from_modes(
        torch.where(stats.active[None], output_modes, torch.zeros_like(output_modes)),
        stats,
    )
    velocity_residual = from_modes(residual_modes, stats)
    return centered_velocity - stats.mean[None], effective_output, velocity_residual


def condition_prediction(
    *,
    model: RankOutputMLP,
    spec: ConditionSpec,
    state: torch.Tensor,
    time: torch.Tensor,
    stats: SpectralStats,
    clip: float,
) -> Prediction:
    if spec.method == "native_x":
        output = model(state, time)
        velocity = (state - output) / time[:, None].clamp_min(clip)
        return Prediction(velocity, output, velocity)

    if spec.method == "native_v":
        output = model(state, time)
        return Prediction(output, output, output)

    if spec.method == "legacy_scalar":
        output = model(state, time)
        k = 0.9
        denominator = ((1.0 - time) + (2.0 * time - 1.0) * k).clamp_min(
            clip
        )[:, None]
        velocity = ((2.0 * k - 1.0) * state - output) / denominator
        return Prediction(
            velocity=velocity,
            native_output=output,
            velocity_residual=velocity,
        )

    if spec.method == "lmmse_residual":
        model_input = lmmse_model_input(
            state,
            time,
            stats,
            input_mode=spec.input_mode,
            clip=clip,
        )
        output = model(model_input, time)
        velocity, effective_output, velocity_residual = velocity_from_lmmse_output(
            output, state, time, stats, clip
        )
        return Prediction(velocity, effective_output, velocity_residual)

    if spec.method == "subspace_velocity":
        output = model(centered_state(state, time, stats), time)
        velocity, effective_output = velocity_from_subspace_output(
            output, state, time, stats, clip
        )
        return Prediction(velocity, effective_output, effective_output)

    model_input = centered_state(state, time, stats)
    output = model(model_input, time)
    k_values = constant_k_values(spec, stats)
    velocity = velocity_from_centered_constant_output(
        output, state, time, stats, k_values, clip
    )
    return Prediction(
        velocity=velocity,
        native_output=output,
        velocity_residual=velocity,
    )


def condition_native_target(
    *,
    spec: ConditionSpec,
    clean: torch.Tensor,
    epsilon: torch.Tensor,
    state: torch.Tensor,
    time: torch.Tensor,
    stats: SpectralStats,
    clip: float,
) -> torch.Tensor:
    if spec.method == "native_x":
        return clean
    if spec.method == "native_v":
        return epsilon - clean
    if spec.method == "legacy_scalar":
        return 0.9 * clean - 0.1 * epsilon
    if spec.method == "lmmse_residual":
        return lmmse_residual_target(clean, epsilon, state, time, stats, clip)
    if spec.method == "subspace_velocity":
        return subspace_velocity_target(clean, epsilon, stats)
    return centered_constant_target(
        clean, epsilon, stats, constant_k_values(spec, stats)
    )


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
    initial = copy.deepcopy(base.state_dict())
    models: dict[str, RankOutputMLP] = {}
    for spec in CONDITIONS:
        model = RankOutputMLP(
            D,
            hidden=hidden,
            output_rank=output_rank,
            depth=depth,
            time_dim=time_dim,
        ).to(device)
        model.load_state_dict(initial)
        models[spec.name] = model
    return models


def train_models(
    *,
    models: dict[str, RankOutputMLP],
    embedding: CurvedEmbedding,
    stats: SpectralStats,
    args: argparse.Namespace,
    setting_seed: int,
    device: torch.device,
) -> list[dict]:
    optimizers = {
        name: torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        for name, model in models.items()
    }
    generator = torch.Generator(device=device.type)
    generator.manual_seed(stable_seed(setting_seed, 701))
    use_amp = args.amp_dtype == "bf16" and device.type == "cuda"
    history: list[dict] = []
    for step in range(1, args.train_steps + 1):
        intrinsic = sample_spiral_2d(
            args.batch_size,
            device=device,
            jitter=args.data_jitter,
            generator=generator,
        )
        clean = embedding.embed(intrinsic)
        epsilon = torch.randn(clean.shape, device=device, generator=generator)
        time = torch.empty(args.batch_size, device=device).uniform_(
            args.t_min, args.t_max, generator=generator
        )
        state = (1.0 - time[:, None]) * clean + time[:, None] * epsilon
        truth = epsilon - clean
        losses: dict[str, float] = {}
        for spec in CONDITIONS:
            optimizer = optimizers[spec.name]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                prediction = condition_prediction(
                    model=models[spec.name],
                    spec=spec,
                    state=state,
                    time=time,
                    stats=stats,
                    clip=args.conversion_clip,
                )
                loss = F.mse_loss(prediction.velocity.float(), truth.float())
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(models[spec.name].parameters(), args.grad_clip)
            optimizer.step()
            losses[spec.name] = float(loss.detach().cpu())
        if step == 1 or step % args.log_every == 0 or step == args.train_steps:
            row = {"step": step}
            row.update({f"loss_{name}": value for name, value in losses.items()})
            history.append(row)
            compact = " ".join(f"{name}={losses[name]:.4g}" for name in losses)
            print(f"[train] {step}/{args.train_steps} {compact}", flush=True)
    return history


@torch.inference_mode()
def evaluate_teacher(
    *,
    models: dict[str, RankOutputMLP],
    embedding: CurvedEmbedding,
    stats: SpectralStats,
    args: argparse.Namespace,
    setting_seed: int,
    experiment_seed: int,
    device: torch.device,
) -> list[dict]:
    rows: list[dict] = []
    for time_index, time_value in enumerate(args.eval_times):
        sums = {
            spec.name: {
                "velocity": 0.0,
                "tangent": 0.0,
                "normal": 0.0,
                "native": 0.0,
                "residual": 0.0,
                "outputs": [],
            }
            for spec in CONDITIONS
        }
        generator = torch.Generator(device=device.type)
        generator.manual_seed(stable_seed(setting_seed, time_index, 809))
        for start in range(0, args.eval_samples, args.eval_batch_size):
            count = min(args.eval_batch_size, args.eval_samples - start)
            intrinsic = sample_spiral_2d(
                count,
                device=device,
                jitter=args.data_jitter,
                generator=generator,
            )
            clean = embedding.embed(intrinsic)
            epsilon = torch.randn(clean.shape, device=device, generator=generator)
            time = torch.full((count,), float(time_value), device=device)
            state = (1.0 - time[:, None]) * clean + time[:, None] * epsilon
            truth = epsilon - clean
            tangent_basis = embedding.tangent_basis(intrinsic)
            for spec in CONDITIONS:
                prediction = condition_prediction(
                    model=models[spec.name],
                    spec=spec,
                    state=state,
                    time=time,
                    stats=stats,
                    clip=args.conversion_clip,
                )
                native_target = condition_native_target(
                    spec=spec,
                    clean=clean,
                    epsilon=epsilon,
                    state=state,
                    time=time,
                    stats=stats,
                    clip=args.conversion_clip,
                )
                error = prediction.velocity - truth
                tangent_coordinates = torch.einsum(
                    "bd,bdk->bk", error, tangent_basis
                )
                tangent_error = torch.einsum(
                    "bk,bdk->bd", tangent_coordinates, tangent_basis
                )
                normal_error = error - tangent_error
                values = sums[spec.name]
                values["velocity"] += float(error.square().sum().cpu())
                values["tangent"] += float(tangent_error.square().sum().cpu())
                values["normal"] += float(normal_error.square().sum().cpu())
                values["native"] += float(
                    (prediction.native_output - native_target).square().sum().cpu()
                )
                values["residual"] += float(
                    prediction.velocity_residual.square().sum().cpu()
                )
                values["outputs"].append(prediction.native_output.cpu())
        denominator = args.eval_samples * embedding.D
        for spec in CONDITIONS:
            values = sums[spec.name]
            effective_rank, numerical_rank, output_variance = covariance_effective_rank(
                torch.cat(values.pop("outputs"), dim=0)
            )
            rows.append(
                {
                    "seed": experiment_seed,
                    "setting_seed": setting_seed,
                    "D": embedding.D,
                    "output_rank": args.output_rank,
                    "time": float(time_value),
                    "condition": spec.name,
                    "velocity_mse": values["velocity"] / denominator,
                    "velocity_tangent_mse": values["tangent"] / denominator,
                    "velocity_normal_mse": values["normal"] / denominator,
                    "native_target_mse": values["native"] / denominator,
                    "predicted_velocity_residual_energy": values["residual"]
                    / denominator,
                    "native_output_effective_rank": effective_rank,
                    "native_output_numerical_rank": numerical_rank,
                    "native_output_variance_per_dim": output_variance,
                }
            )
    return rows


@torch.inference_mode()
def sample_models(
    *,
    models: dict[str, RankOutputMLP],
    embedding: CurvedEmbedding,
    stats: SpectralStats,
    args: argparse.Namespace,
    setting_seed: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    collected: dict[str, list[np.ndarray]] = {spec.name: [] for spec in CONDITIONS}
    grid = torch.linspace(
        args.sample_t_max,
        args.sample_t_min,
        args.sample_steps + 1,
        device=device,
    )
    sample_seed = stable_seed(setting_seed, 1213)
    for start in range(0, args.sample_count, args.sample_batch_size):
        count = min(args.sample_batch_size, args.sample_count - start)
        generator = torch.Generator(device=device.type)
        generator.manual_seed(sample_seed + start)
        initial = args.sample_t_max * torch.randn(
            count, embedding.D, device=device, generator=generator
        )
        states = {spec.name: initial.clone() for spec in CONDITIONS}
        for index in range(args.sample_steps):
            time_now, time_next = grid[index], grid[index + 1]
            time = time_now.expand(count)
            for spec in CONDITIONS:
                prediction = condition_prediction(
                    model=models[spec.name],
                    spec=spec,
                    state=states[spec.name],
                    time=time,
                    stats=stats,
                    clip=args.conversion_clip,
                )
                states[spec.name] = states[spec.name] + (
                    time_next - time_now
                ) * prediction.velocity
        final_time = grid[-1].expand(count)
        for spec in CONDITIONS:
            prediction = condition_prediction(
                model=models[spec.name],
                spec=spec,
                state=states[spec.name],
                time=final_time,
                stats=stats,
                clip=args.conversion_clip,
            )
            clean = states[spec.name] - final_time[:, None] * prediction.velocity
            collected[spec.name].append(clean.cpu().numpy())
    return {name: np.concatenate(parts) for name, parts in collected.items()}


def plot_generation(
    path: Path,
    samples: dict[str, np.ndarray],
    reference_intrinsic: np.ndarray,
    embedding: CurvedEmbedding,
    max_points: int,
) -> None:
    panels: list[tuple[str, np.ndarray]] = [("reference", reference_intrinsic)]
    with torch.inference_mode():
        for condition, ambient in samples.items():
            intrinsic = embedding.decode_intrinsic(
                torch.from_numpy(ambient).to(embedding.device)
            ).cpu().numpy()
            panels.append((condition, intrinsic))
    columns = 4
    rows = int(np.ceil(len(panels) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, (name, values) in zip(axes.flat, panels):
        axis.axis("on")
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


def run_seed(
    *,
    args: argparse.Namespace,
    experiment_seed: int,
    device: torch.device,
) -> tuple[list[dict], list[dict], dict]:
    setting_seed = stable_seed(
        experiment_seed,
        args.D,
        args.output_rank,
        int(round(10_000 * args.curvature)),
        2027,
    )
    setting_dir = args.output_root / f"seed{experiment_seed}"
    if args.resume and (setting_dir / "summary.json").is_file():
        with (setting_dir / "teacher_metrics.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            teacher = list(csv.DictReader(handle))
        with (setting_dir / "generation_metrics.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            generation = list(csv.DictReader(handle))
        summary = json.loads((setting_dir / "summary.json").read_text(encoding="utf-8"))
        print(f"[resume] {setting_dir}", flush=True)
        return teacher, generation, summary

    setting_dir.mkdir(parents=True, exist_ok=True)
    set_seed(setting_seed)
    embedding = CurvedEmbedding(
        args.D,
        curvature=args.curvature,
        frequency_scale=args.frequency_scale,
        seed=stable_seed(
            experiment_seed,
            args.D,
            int(round(10_000 * args.curvature)),
            41,
        ),
        device=device,
        scale_mode=args.scale_mode,
    )
    stats = estimate_spectral_stats(
        embedding=embedding,
        samples=args.covariance_samples,
        batch_size=args.covariance_batch_size,
        data_jitter=args.data_jitter,
        seed=stable_seed(setting_seed, 101),
        relative_floor=args.eigen_relative_floor,
        absolute_floor=args.eigen_absolute_floor,
    )
    models = build_matched_models(
        D=args.D,
        hidden=args.hidden,
        output_rank=args.output_rank,
        depth=args.depth,
        time_dim=args.time_dim,
        seed=setting_seed,
        device=device,
    )
    history = train_models(
        models=models,
        embedding=embedding,
        stats=stats,
        args=args,
        setting_seed=setting_seed,
        device=device,
    )
    for model in models.values():
        model.eval()
    teacher = evaluate_teacher(
        models=models,
        embedding=embedding,
        stats=stats,
        args=args,
        setting_seed=setting_seed,
        experiment_seed=experiment_seed,
        device=device,
    )
    reference_generator = torch.Generator(device=device.type)
    reference_generator.manual_seed(stable_seed(setting_seed, 1201))
    reference_intrinsic = sample_spiral_2d(
        max(2 * args.sample_count, 8192),
        device=device,
        jitter=args.data_jitter,
        generator=reference_generator,
    ).cpu().numpy()
    generated = sample_models(
        models=models,
        embedding=embedding,
        stats=stats,
        args=args,
        setting_seed=setting_seed,
        device=device,
    )
    generation = evaluate_generation(
        samples=generated,
        reference_intrinsic=reference_intrinsic,
        embedding=embedding,
        output_rank=args.output_rank,
        seed=setting_seed,
        device=device,
        metric_max_points=args.metric_max_points,
        projections=args.swd_projections,
        rank_dependent_randomness=False,
    )
    for row in generation:
        row["seed"] = experiment_seed
        row["setting_seed"] = setting_seed
    save_csv(setting_dir / "train_history.csv", history)
    save_csv(setting_dir / "teacher_metrics.csv", teacher)
    save_csv(setting_dir / "generation_metrics.csv", generation)
    plot_generation(
        setting_dir / "generation_scatter.png",
        generated,
        reference_intrinsic,
        embedding,
        args.plot_points,
    )
    summary = {
        "seed": experiment_seed,
        "setting_seed": setting_seed,
        "D": args.D,
        "curvature": args.curvature,
        "output_rank": args.output_rank,
        "mean_norm": float(stats.mean.norm().cpu()),
        "covariance_trace": float(stats.eigenvalues.sum().cpu()),
        "covariance_active_rank": int(stats.active.sum().cpu()),
        "covariance_threshold": stats.threshold,
        "top_eigenvalues": [float(value) for value in stats.eigenvalues[:8].cpu()],
        "kdiff_global_k": float(
            args.D / (args.D + float(stats.eigenvalues.sum().cpu()))
        ),
        "generation": {
            row["condition"]: {
                "swd_2d": float(row["swd_2d"]),
                "swd_ambient": float(row["swd_ambient"]),
                "mmd_2d": float(row["mmd_2d"]),
                "manifold_consistency_rms": float(row["manifold_consistency_rms"]),
            }
            for row in generation
        },
    }
    (setting_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return teacher, generation, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--D", type=int, default=64)
    parser.add_argument("--curvature", type=float, default=0.0)
    parser.add_argument("--output-rank", type=int, default=16)
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
    parser.add_argument("--covariance-samples", type=int, default=131072)
    parser.add_argument("--covariance-batch-size", type=int, default=8192)
    parser.add_argument("--eigen-relative-floor", type=float, default=1e-6)
    parser.add_argument("--eigen-absolute-floor", type=float, default=1e-7)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.output_rank > min(args.hidden, args.D):
        raise ValueError("output rank exceeds model dimensions")
    if args.curvature < 0:
        raise ValueError("curvature must be non-negative")
    if not (0.0 < args.t_min < args.t_max < 1.0):
        raise ValueError("training times must be inside (0,1)")
    if not (0.0 < args.sample_t_min < args.sample_t_max < 1.0):
        raise ValueError("sampling times must be inside (0,1)")
    device = torch.device(args.device)
    manifest = {
        "definition": "capacity-aware spectral output preconditioning audit",
        "path": "z_t=(1-t)x+t epsilon; velocity=epsilon-x",
        "loss": "common recovered-velocity MSE",
        "conditions": [spec.__dict__ for spec in CONDITIONS],
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    all_teacher: list[dict] = []
    all_generation: list[dict] = []
    summaries: list[dict] = []
    for seed in args.seeds:
        teacher, generation, summary = run_seed(
            args=args,
            experiment_seed=seed,
            device=device,
        )
        all_teacher.extend(teacher)
        all_generation.extend(generation)
        summaries.append(summary)
    save_csv(args.output_root / "teacher_metrics.csv", all_teacher)
    save_csv(args.output_root / "generation_metrics.csv", all_generation)
    (args.output_root / "summaries.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    print(f"[done] {len(summaries)} seeds at {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
