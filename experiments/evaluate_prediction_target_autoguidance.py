#!/usr/bin/env python3
"""Compare faithful AutoGuidance with prediction-target extrapolation.

The strong and weak AutoGuidance branches predict the same clean target.  The
weak branch is degraded either by early stopping or by a smaller hidden width.
The existing x-v extrapolation remains explicitly labelled as a different
mechanism because its two predictors use different training targets.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from experiments.analyze_prediction_target_cluster_separation import (
    audit,
    intrinsic_projection,
    kde_density,
)
from experiments.run_prediction_target_bayes_oracle_v5 import (
    FrozenPushforward,
    TangentGaussianMixture,
    build_model,
    evaluate_generation,
    save_csv,
    save_json,
    stable_seed,
)
from experiments.run_prediction_target_extrapolation_toy_v4 import (
    clean_from_output,
)
from experiments.train_prediction_target_internal_guidance import (
    InternalResidualDenoiseMLP,
)


def parse_float_list(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(not math.isfinite(item) for item in values):
        raise ValueError("weights must be a non-empty list of finite values")
    return values


def parse_methods(value: str) -> list[str]:
    methods = [item.strip() for item in value.split(",") if item.strip()]
    allowed = {
        "ptg",
        "ptg_eps",
        "ptg_reverse",
        "ag_early",
        "ag_small",
        "ig",
        "ig_v",
        "ctig_v",
    }
    if not methods or any(method not in allowed for method in methods):
        raise ValueError(f"methods must be chosen from {sorted(allowed)}")
    return methods


def parse_windows(value: str) -> list[tuple[str, float, float]]:
    windows = []
    for item in value.split(","):
        if not item.strip():
            continue
        fields = item.strip().split(":")
        if len(fields) != 3:
            raise ValueError("window format must be name:t_min:t_max")
        name, lower, upper = fields
        lower_value = float(lower)
        upper_value = float(upper)
        if not 0.0 <= lower_value < upper_value <= 1.0:
            raise ValueError(f"invalid guidance window: {item}")
        windows.append((name, lower_value, upper_value))
    if not windows:
        raise ValueError("at least one guidance window is required")
    return windows


def build_mixture(manifest: dict[str, object], device: torch.device) -> TangentGaussianMixture:
    return TangentGaussianMixture(
        D=int(manifest["D"]),
        components=int(manifest["components"]),
        curvature=float(manifest["curvature"]),
        frequency_scale=float(manifest["frequency_scale"]),
        center_rms=float(manifest["center_rms"]),
        sigma_tangent=float(manifest["sigma_tangent"]),
        sigma_normal=float(manifest["sigma_normal"]),
        seed=int(manifest["mixture_seed"]),
        device=device,
    )


def load_target_model(
    *,
    checkpoint: Path,
    target: str,
    architecture: str,
    hidden: int,
    manifest: dict[str, object],
    device: torch.device,
) -> nn.Module:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_model(
        architecture,
        D=int(manifest["D"]),
        hidden=hidden,
        depth=int(manifest["depth"]),
        time_dim=int(manifest["time_dim"]),
    ).to(device)
    model.load_state_dict(payload["models"][target])
    return model.eval().requires_grad_(False)


def load_internal_model(
    *,
    checkpoint: Path,
    expected_intermediate_target: str,
    manifest: dict[str, object],
    device: torch.device,
) -> InternalResidualDenoiseMLP:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    config = payload["config"]
    actual_target = str(config.get("intermediate_target", "x"))
    if actual_target != expected_intermediate_target:
        raise ValueError(
            f"{checkpoint} has intermediate_target={actual_target!r}, expected "
            f"{expected_intermediate_target!r}"
        )
    model = InternalResidualDenoiseMLP(
        D=int(manifest["D"]),
        hidden=int(config["hidden"]),
        depth=int(manifest["depth"]),
        time_dim=int(manifest["time_dim"]),
        intermediate_after=int(config["intermediate_after"]),
    ).to(device)
    model.load_state_dict(payload["model"])
    return model.eval().requires_grad_(False)


@torch.inference_mode()
def clean_prediction(
    model: nn.Module,
    state: torch.Tensor,
    time: torch.Tensor,
    *,
    target: str,
    conversion_clip: float,
) -> torch.Tensor:
    return clean_from_output(
        model(state, time), state, time, target, conversion_clip
    )


@torch.inference_mode()
def condition_clean(
    *,
    condition: str,
    weight: float,
    state: torch.Tensor,
    time: torch.Tensor,
    mixture: TangentGaussianMixture,
    strong_x: nn.Module,
    strong_v: nn.Module,
    strong_eps: nn.Module,
    weak_early_x: nn.Module,
    weak_small_x: nn.Module,
    internal_model: InternalResidualDenoiseMLP | None,
    internal_v_model: InternalResidualDenoiseMLP | None,
    conversion_clip: float,
    guidance_t_min: float = 0.0,
    guidance_t_max: float = 1.0,
) -> torch.Tensor:
    if condition == "bayes":
        return mixture.posterior_clean(state, time)
    if condition in {"ig", "ig_v", "ctig_v"}:
        model = internal_model if condition == "ig" else internal_v_model
        if model is None:
            raise ValueError(f"{condition} requested without its checkpoint")
        intermediate_target = "x" if condition == "ig" else "v"
        intermediate_raw, final_raw = model(state, time)
        intermediate = clean_from_output(
            intermediate_raw,
            state,
            time,
            intermediate_target,
            conversion_clip,
        )
        final = clean_from_output(
            final_raw, state, time, "x", conversion_clip
        )
        guided = intermediate + float(weight) * (final - intermediate)
        active = (time >= guidance_t_min) & (time <= guidance_t_max)
        return torch.where(active[:, None], guided, final)
    strong = clean_prediction(
        strong_x,
        state,
        time,
        target="x",
        conversion_clip=conversion_clip,
    )
    if condition == "x":
        return strong
    if condition in {"v", "ptg", "ptg_reverse"}:
        velocity_target = clean_prediction(
            strong_v,
            state,
            time,
            target="v",
            conversion_clip=conversion_clip,
        )
    if condition == "v":
        return velocity_target
    if condition in {"eps", "ptg_eps"}:
        noise_target = clean_prediction(
            strong_eps,
            state,
            time,
            target="eps",
            conversion_clip=conversion_clip,
        )
    if condition == "eps":
        return noise_target
    if condition == "ptg":
        weak = velocity_target
        guidance_strong = strong
    elif condition == "ptg_eps":
        weak = noise_target
        guidance_strong = strong
    elif condition == "ptg_reverse":
        weak = strong
        guidance_strong = velocity_target
    elif condition == "ag_early":
        weak = clean_prediction(
            weak_early_x,
            state,
            time,
            target="x",
            conversion_clip=conversion_clip,
        )
        guidance_strong = strong
    elif condition == "ag_small":
        weak = clean_prediction(
            weak_small_x,
            state,
            time,
            target="x",
            conversion_clip=conversion_clip,
        )
        guidance_strong = strong
    else:
        raise ValueError(condition)
    guided = weak + float(weight) * (guidance_strong - weak)
    active = (time >= guidance_t_min) & (time <= guidance_t_max)
    return torch.where(active[:, None], guided, guidance_strong)


@torch.inference_mode()
def sample_condition(
    *,
    condition: str,
    weight: float,
    sample_count: int,
    batch_size: int,
    steps: int,
    t_max: float,
    t_min: float,
    seed: int,
    mixture: TangentGaussianMixture,
    strong_x: nn.Module,
    strong_v: nn.Module,
    strong_eps: nn.Module,
    weak_early_x: nn.Module,
    weak_small_x: nn.Module,
    internal_model: InternalResidualDenoiseMLP | None,
    internal_v_model: InternalResidualDenoiseMLP | None,
    conversion_clip: float,
    guidance_t_min: float = 0.0,
    guidance_t_max: float = 1.0,
    sampler: str = "legacy_euler_clean",
    schedule_rho: float = 1.0,
    initial_state: str = "forward_noised",
) -> np.ndarray:
    if sampler not in {"legacy_euler_clean", "euler_state", "heun_state"}:
        raise ValueError(f"unsupported sampler: {sampler}")
    if schedule_rho <= 0.0:
        raise ValueError("schedule_rho must be positive")
    if initial_state not in {"forward_noised", "gaussian_approx"}:
        raise ValueError(f"unsupported initial state: {initial_state}")
    if t_min < 0.0:
        raise ValueError("t_min must be non-negative")
    if sampler == "legacy_euler_clean" and t_min <= 0.0:
        raise ValueError("legacy clean readout requires t_min > 0")
    outputs = []
    grid = torch.linspace(
        t_max ** (1.0 / schedule_rho),
        t_min ** (1.0 / schedule_rho),
        steps + 1,
        device=mixture.device,
    ).pow(schedule_rho)
    for start in range(0, sample_count, batch_size):
        count = min(batch_size, sample_count - start)
        generator = torch.Generator(device=mixture.device.type)
        generator.manual_seed(seed + start)
        if initial_state == "forward_noised":
            clean_start, _ = mixture.sample_clean(count, generator=generator)
            noise = torch.randn(
                clean_start.shape,
                device=mixture.device,
                generator=generator,
            )
            state = (1.0 - t_max) * clean_start + t_max * noise
        else:
            # At t close to one, omitting the tiny clean component gives a
            # data-independent approximation to the true noisy marginal.
            noise = torch.randn(
                (count, mixture.D),
                device=mixture.device,
                generator=generator,
            )
            state = t_max * noise
        for index in range(steps):
            current = grid[index]
            following = grid[index + 1]
            time = torch.full((count,), float(current), device=mixture.device)
            predicted_clean = condition_clean(
                condition=condition,
                weight=weight,
                state=state,
                time=time,
                mixture=mixture,
                strong_x=strong_x,
                strong_v=strong_v,
                strong_eps=strong_eps,
                weak_early_x=weak_early_x,
                weak_small_x=weak_small_x,
                internal_model=internal_model,
                internal_v_model=internal_v_model,
                conversion_clip=conversion_clip,
                guidance_t_min=guidance_t_min,
                guidance_t_max=guidance_t_max,
            )
            velocity = (state - predicted_clean) / time[:, None].clamp_min(
                1e-8
            )
            step_size = following - current
            # The clean parameterization is singular at the data endpoint.
            # Match JiT's sampler: use Heun for interior steps and one final
            # Euler step, so the network is never evaluated at t=0.
            if sampler == "heun_state" and index < steps - 1:
                predicted_state = state + step_size * velocity
                next_time = torch.full(
                    (count,), float(following), device=mixture.device
                )
                next_clean = condition_clean(
                    condition=condition,
                    weight=weight,
                    state=predicted_state,
                    time=next_time,
                    mixture=mixture,
                    strong_x=strong_x,
                    strong_v=strong_v,
                    strong_eps=strong_eps,
                    weak_early_x=weak_early_x,
                    weak_small_x=weak_small_x,
                    internal_model=internal_model,
                    internal_v_model=internal_v_model,
                    conversion_clip=conversion_clip,
                    guidance_t_min=guidance_t_min,
                    guidance_t_max=guidance_t_max,
                )
                next_velocity = (
                    (predicted_state - next_clean)
                    / next_time[:, None].clamp_min(1e-8)
                )
                state = state + 0.5 * step_size * (
                    velocity + next_velocity
                )
            else:
                state = state + step_size * velocity
        if sampler == "legacy_euler_clean":
            final_time = torch.full(
                (count,), float(grid[-1]), device=mixture.device
            )
            output = condition_clean(
                condition=condition,
                weight=weight,
                state=state,
                time=final_time,
                mixture=mixture,
                strong_x=strong_x,
                strong_v=strong_v,
                strong_eps=strong_eps,
                weak_early_x=weak_early_x,
                weak_small_x=weak_small_x,
                internal_model=internal_model,
                internal_v_model=internal_v_model,
                conversion_clip=conversion_clip,
                guidance_t_min=guidance_t_min,
                guidance_t_max=guidance_t_max,
            )
        else:
            output = state
        outputs.append(output.float().cpu().numpy())
    result = np.concatenate(outputs, axis=0)
    if not np.isfinite(result).all():
        raise FloatingPointError(f"non-finite samples for {condition}, w={weight}")
    return result


def condition_name(kind: str, weight: float, window_name: str = "full") -> str:
    if kind in {"bayes", "x", "v"}:
        return kind
    suffix = "" if window_name == "full" else f"_{window_name}"
    return f"{kind}_w{weight:g}{suffix}"


def plot_intrinsic(
    *,
    path: Path,
    arrays: dict[str, np.ndarray],
    mixture: TangentGaussianMixture,
    ordered_names: list[str] | None = None,
    title: str = "Prediction-target guidance vs same-target AutoGuidance",
    zoom_limit: float | None = None,
) -> None:
    names = ordered_names if ordered_names is not None else list(arrays)
    columns = 4
    rows = math.ceil(len(names) / columns)
    figure, axes = plt.subplots(
        rows, columns, figsize=(5.2 * columns, 5.2 * rows), squeeze=False
    )
    intrinsic = {}
    with torch.no_grad():
        for name, values in arrays.items():
            intrinsic[name] = (
                mixture.intrinsic_readout(
                    torch.from_numpy(values).to(mixture.device)
                )
                .cpu()
                .numpy()
            )
    reference = intrinsic["reference"]
    lower = np.quantile(reference, 0.002, axis=0)
    upper = np.quantile(reference, 0.998, axis=0)
    if zoom_limit is not None:
        lower = np.full(2, -float(zoom_limit))
        upper = np.full(2, float(zoom_limit))
    padding = 0.08 * float(np.max(upper - lower))
    for axis, name in zip(axes.flat, names):
        values = intrinsic[name]
        axis.scatter(
            reference[:3000, 0],
            reference[:3000, 1],
            s=3,
            color="#b8b8b8",
            alpha=0.14,
            linewidths=0,
        )
        axis.scatter(
            values[:3000, 0],
            values[:3000, 1],
            s=3,
            color="#e66b1a" if name != "reference" else "#2574a9",
            alpha=0.40,
            linewidths=0,
        )
        axis.set_title(name.replace("_", " "), fontsize=12)
        axis.set_xlim(lower[0] - padding, upper[0] + padding)
        axis.set_ylim(lower[1] - padding, upper[1] + padding)
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in axes.flat[len(names) :]:
        axis.set_visible(False)
    figure.suptitle(title, fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


METHOD_LABELS = {
    "ptg": "Prediction target: v -> x",
    "ptg_eps": "Prediction target: noise -> x",
    "ptg_reverse": "Prediction target: x -> v",
    "ag_early": "AutoGuidance: early x -> x",
    "ag_small": "AutoGuidance: small x -> x",
    "ig": "Internal Guidance: x head -> x head",
    "ig_v": "Cross-target internal (ours): v head -> x head",
    "ctig_v": "Cross-target internal (ours): v head -> x head",
}
METHOD_COLORS = {
    "ptg": "#b279a2",
    "ptg_eps": "#4e79a7",
    "ptg_reverse": "#6f4e9c",
    "ag_early": "#f28e2b",
    "ag_small": "#e15759",
    "ig": "#59a14f",
    "ig_v": "#2f8f9d",
    "ctig_v": "#2f8f9d",
}


def scale_sample_name(method: str, weight: float) -> str:
    if weight == 1.0 and method == "ptg_reverse":
        return "v"
    if weight == 1.0 and method not in {"ig", "ig_v", "ctig_v"}:
        return "x"
    return condition_name(method, weight)


def plot_scale_grid(
    *,
    path: Path,
    arrays: dict[str, np.ndarray],
    mixture: TangentGaussianMixture,
    methods: list[str],
    weights: list[float],
    zoom_limit: float | None = None,
) -> None:
    """Plot every method and scale with identical axes and reference overlay."""
    available_methods = [
        method
        for method in methods
        if all(scale_sample_name(method, weight) in arrays for weight in weights)
    ]
    if not available_methods:
        return
    needed = {"reference"}
    for method in available_methods:
        needed.update(scale_sample_name(method, weight) for weight in weights)
    projected = {
        name: intrinsic_projection(mixture, arrays[name]) for name in needed
    }
    reference = projected["reference"]
    if zoom_limit is None:
        lower = np.quantile(reference, 0.002, axis=0)
        upper = np.quantile(reference, 0.998, axis=0)
    else:
        lower = np.full(2, -float(zoom_limit))
        upper = np.full(2, float(zoom_limit))
    padding = 0.06 * float(np.max(upper - lower))
    figure, axes = plt.subplots(
        len(available_methods),
        len(weights),
        figsize=(4.15 * len(weights), 4.15 * len(available_methods)),
        squeeze=False,
        constrained_layout=True,
    )
    for row, method in enumerate(available_methods):
        for column, weight in enumerate(weights):
            axis = axes[row, column]
            name = scale_sample_name(method, weight)
            values = projected[name]
            axis.scatter(
                reference[:3000, 0],
                reference[:3000, 1],
                s=3,
                color="#8d8d8d",
                alpha=0.10,
                linewidths=0,
                rasterized=True,
            )
            axis.scatter(
                values[:3000, 0],
                values[:3000, 1],
                s=3,
                color=METHOD_COLORS[method],
                alpha=0.46,
                linewidths=0,
                rasterized=True,
            )
            if row == 0:
                axis.set_title(f"scale w = {weight:g}", fontsize=12)
            if column == 0:
                axis.set_ylabel(METHOD_LABELS[method], fontsize=11)
            axis.set_xlim(lower[0] - padding, upper[0] + padding)
            axis.set_ylim(lower[1] - padding, upper[1] + padding)
            axis.set_aspect("equal")
            axis.set_xticks([])
            axis.set_yticks([])
    suffix = "inner spiral" if zoom_limit is not None else "full distribution"
    figure.suptitle(
        f"Guidance methods across a shared scale sweep: {suffix}", fontsize=16
    )
    figure.savefig(path, dpi=185, bbox_inches="tight")
    plt.close(figure)


def inner_contrast_ratios(
    arrays: dict[str, np.ndarray], mixture: TangentGaussianMixture
) -> dict[str, float]:
    reference = intrinsic_projection(mixture, arrays["reference"])
    centers = mixture.intrinsic_centers.float().cpu().numpy()
    nearest = np.linalg.norm(reference[:, None] - centers[None], axis=2).min(axis=1)
    bandwidth = float(np.median(nearest))

    def inner_contrast(values: np.ndarray) -> float:
        points = intrinsic_projection(mixture, values)
        midpoints = 0.5 * (centers[:-1] + centers[1:])
        center_density = kde_density(points, centers, bandwidth)
        midpoint_density = kde_density(points, midpoints, bandwidth)
        contrast = np.log(
            0.5 * (center_density[:-1] + center_density[1:]) + 1e-12
        ) - np.log(midpoint_density + 1e-12)
        return float(np.array_split(contrast, 3)[0].mean())

    reference_value = inner_contrast(arrays["reference"])
    return {
        name: inner_contrast(values) / max(reference_value, 1e-12)
        for name, values in arrays.items()
    }


def plot_scale_metrics(
    *,
    path: Path,
    summary: pd.DataFrame,
    cluster: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    mixture: TangentGaussianMixture,
    methods: list[str],
    weights: list[float],
) -> None:
    inner_ratio = inner_contrast_ratios(arrays, mixture)
    summary_by_name = summary.set_index("condition")
    cluster_by_name = cluster.set_index("condition")
    reference_bridge = float(cluster_by_name.loc["reference", "intrinsic_bridge_rate"])
    rows = []
    for method in methods:
        for weight in weights:
            name = scale_sample_name(method, weight)
            if name not in summary_by_name.index:
                continue
            row = summary_by_name.loc[name]
            rows.append(
                {
                    "method": method,
                    "weight": weight,
                    "latent_swd": float(row.latent_swd),
                    "normal_width_ratio": float(row.nearest_normal_rms)
                    / float(row.reference_nearest_normal_rms),
                    "inner_contrast_ratio": inner_ratio[name],
                    "bridge_rate": float(
                        cluster_by_name.loc[name, "intrinsic_bridge_rate"]
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    figure, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    panels = [
        ("latent_swd", "Intrinsic SWD", None),
        ("normal_width_ratio", "Normal width ratio", 1.0),
        ("inner_contrast_ratio", "Inner separation ratio", 1.0),
        ("bridge_rate", "Intrinsic bridge rate", reference_bridge),
    ]
    for axis, (column, title, target) in zip(axes.flat, panels):
        for method, group in frame.groupby("method", sort=False):
            group = group.sort_values("weight")
            axis.plot(
                group.weight,
                group[column],
                marker="o",
                linewidth=2,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
        if target is not None:
            axis.axhline(target, color="black", linestyle="--", linewidth=1)
        axis.set_xlabel("Guidance scale w (w=1 is no extrapolation)")
        axis.set_title(title)
        axis.grid(alpha=0.22)
    axes[0, 0].legend(fontsize=9)
    figure.suptitle(
        "Scale changes separation and distribution fidelity differently",
        fontsize=16,
    )
    figure.savefig(path, dpi=185, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument(
        "--architecture",
        choices=("jit_relu", "plain", "residual", "residual_skip"),
        default="residual",
    )
    parser.add_argument("--strong-hidden", type=int, default=128)
    parser.add_argument("--weak-hidden", type=int, default=64)
    parser.add_argument("--strong-step", type=int, default=30000)
    parser.add_argument("--early-step", type=int, default=6000)
    parser.add_argument(
        "--weak-step",
        type=int,
        help="Checkpoint step for the smaller AG model (default: strong-step).",
    )
    parser.add_argument("--weights", default="1.1,1.3,1.5,2,3")
    parser.add_argument("--methods", default="ptg,ag_early,ag_small")
    parser.add_argument("--windows", default="full:0:1")
    parser.add_argument("--sample-count", type=int, default=5000)
    parser.add_argument("--sample-batch-size", type=int, default=1000)
    parser.add_argument("--sample-steps", type=int, default=100)
    parser.add_argument(
        "--sampler",
        choices=("legacy_euler_clean", "euler_state", "heun_state"),
        default="legacy_euler_clean",
        help=(
            "legacy_euler_clean reproduces historical runs; state-output "
            "samplers avoid the finite-t posterior-mean endpoint collapse"
        ),
    )
    parser.add_argument("--schedule-rho", type=float, default=1.0)
    parser.add_argument(
        "--initial-state",
        choices=("forward_noised", "gaussian_approx"),
        default="forward_noised",
        help=(
            "forward_noised reproduces paired diagnostic runs; "
            "gaussian_approx avoids using a real clean sample at initialization"
        ),
    )
    parser.add_argument("--sample-t-min", type=float)
    parser.add_argument("--sample-t-max", type=float)
    parser.add_argument("--ig-checkpoint", type=Path)
    parser.add_argument(
        "--ig-v-checkpoint",
        type=Path,
        help="IG checkpoint whose intermediate head directly predicts v.",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    weak_step = args.strong_step if args.weak_step is None else args.weak_step

    device = torch.device(args.device)
    seed_dir = args.source_root / f"seed{args.seed}"
    manifest = json.loads((seed_dir / "manifest.json").read_text())
    mixture = build_mixture(manifest, device)
    sample_t_min = (
        float(manifest["sample_t_min"])
        if args.sample_t_min is None
        else float(args.sample_t_min)
    )
    sample_t_max = (
        float(manifest["sample_t_max"])
        if args.sample_t_max is None
        else float(args.sample_t_max)
    )
    if not 0.0 <= sample_t_min < sample_t_max <= 1.0:
        raise ValueError("sample times must satisfy 0 <= t_min < t_max <= 1")
    if args.sampler == "legacy_euler_clean" and sample_t_min == 0.0:
        raise ValueError("legacy clean readout requires sample_t_min > 0")
    strong_checkpoint = (
        seed_dir
        / args.architecture
        / f"H{args.strong_hidden}"
        / "checkpoints"
        / f"step{args.strong_step:06d}.pt"
    )
    early_checkpoint = (
        seed_dir
        / args.architecture
        / f"H{args.strong_hidden}"
        / "checkpoints"
        / f"step{args.early_step:06d}.pt"
    )
    small_checkpoint = (
        seed_dir
        / args.architecture
        / f"H{args.weak_hidden}"
        / "checkpoints"
        / f"step{weak_step:06d}.pt"
    )
    for path in (strong_checkpoint, early_checkpoint, small_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)

    strong_x = load_target_model(
        checkpoint=strong_checkpoint,
        target="x",
        architecture=args.architecture,
        hidden=args.strong_hidden,
        manifest=manifest,
        device=device,
    )
    strong_v = load_target_model(
        checkpoint=strong_checkpoint,
        target="v",
        architecture=args.architecture,
        hidden=args.strong_hidden,
        manifest=manifest,
        device=device,
    )
    strong_eps = load_target_model(
        checkpoint=strong_checkpoint,
        target="eps",
        architecture=args.architecture,
        hidden=args.strong_hidden,
        manifest=manifest,
        device=device,
    )
    weak_early_x = load_target_model(
        checkpoint=early_checkpoint,
        target="x",
        architecture=args.architecture,
        hidden=args.strong_hidden,
        manifest=manifest,
        device=device,
    )
    weak_small_x = load_target_model(
        checkpoint=small_checkpoint,
        target="x",
        architecture=args.architecture,
        hidden=args.weak_hidden,
        manifest=manifest,
        device=device,
    )
    internal_model = None
    if args.ig_checkpoint is not None:
        if args.architecture != "residual":
            raise ValueError("current IG checkpoints require residual architecture")
        internal_model = load_internal_model(
            checkpoint=args.ig_checkpoint,
            expected_intermediate_target="x",
            manifest=manifest,
            device=device,
        )
    internal_v_model = None
    if args.ig_v_checkpoint is not None:
        if args.architecture != "residual":
            raise ValueError("current IG checkpoints require residual architecture")
        internal_v_model = load_internal_model(
            checkpoint=args.ig_v_checkpoint,
            expected_intermediate_target="v",
            manifest=manifest,
            device=device,
        )
    weights = parse_float_list(args.weights)
    methods = parse_methods(args.methods)
    windows = parse_windows(args.windows)
    if "ig" in methods and internal_model is None:
        raise ValueError("--methods includes ig but --ig-checkpoint is absent")
    if ({"ig_v", "ctig_v"} & set(methods)) and internal_v_model is None:
        raise ValueError(
            "--methods includes a cross-target v head but "
            "--ig-v-checkpoint is absent"
        )
    specs: list[tuple[str, str, float, float, float]] = [
        ("bayes", "bayes", 1.0, 0.0, 1.0),
        ("x", "x", 1.0, 0.0, 1.0),
        ("v", "v", 1.0, 0.0, 1.0),
        ("eps", "eps", 1.0, 0.0, 1.0),
    ]
    for kind in methods:
        if kind in {"ig", "ig_v", "ctig_v"}:
            specs.extend(
                [
                    (condition_name(kind, 0.0), kind, 0.0, 0.0, 1.0),
                    (condition_name(kind, 1.0), kind, 1.0, 0.0, 1.0),
                ]
            )
        for weight in weights:
            if kind in {"ig", "ig_v", "ctig_v"} and weight in {0.0, 1.0}:
                continue
            for window_name, lower, upper in windows:
                specs.append(
                    (
                        condition_name(kind, weight, window_name),
                        kind,
                        weight,
                        lower,
                        upper,
                    )
                )
    deduplicated = []
    seen = set()
    for spec in specs:
        if spec[0] not in seen:
            deduplicated.append(spec)
            seen.add(spec[0])
    specs = deduplicated
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cached_reference = np.load(seed_dir / "common" / "reference.npy")
    if len(cached_reference) >= args.sample_count:
        reference_rng = np.random.default_rng(stable_seed(args.seed, 2287))
        reference_indices = reference_rng.choice(
            len(cached_reference), size=args.sample_count, replace=False
        )
        reference = cached_reference[reference_indices]
        reference_source = "cached_without_replacement"
    else:
        reference_generator = torch.Generator(device=device.type)
        reference_generator.manual_seed(stable_seed(args.seed, 607))
        reference_tensor, _ = mixture.sample_clean(
            args.sample_count, generator=reference_generator
        )
        reference = reference_tensor.float().cpu().numpy()
        reference_source = "fresh_analytic_mixture_fixed_seed"
    arrays: dict[str, np.ndarray] = {"reference": reference}
    rollout_seed = stable_seed(args.seed, 64, 2309)
    for name, kind, weight, guidance_t_min, guidance_t_max in tqdm(
        specs, desc="Sampling guidance conditions"
    ):
        arrays[name] = sample_condition(
            condition=kind,
            weight=weight,
            sample_count=args.sample_count,
            batch_size=args.sample_batch_size,
            steps=args.sample_steps,
            t_max=sample_t_max,
            t_min=sample_t_min,
            seed=rollout_seed,
            mixture=mixture,
            strong_x=strong_x,
            strong_v=strong_v,
            strong_eps=strong_eps,
            weak_early_x=weak_early_x,
            weak_small_x=weak_small_x,
            internal_model=internal_model,
            internal_v_model=internal_v_model,
            conversion_clip=float(manifest["conversion_clip"]),
            guidance_t_min=guidance_t_min,
            guidance_t_max=guidance_t_max,
            sampler=args.sampler,
            schedule_rho=args.schedule_rho,
            initial_state=args.initial_state,
        )
        np.save(output_dir / f"samples_{name}.npy", arrays[name])

    pushforward = FrozenPushforward(
        D=int(manifest["D"]),
        width=int(manifest["pushforward_width"]),
        seed=int(manifest["pushforward_seed"]),
    )
    metric_conditions = [
        (name, kind, weight) for name, kind, weight, _, _ in specs
    ]
    generation_rows = evaluate_generation(
        samples=arrays,
        conditions=metric_conditions,
        reference=arrays["reference"],
        mixture=mixture,
        pushforward=pushforward,
        projections=int(manifest["swd_projections"]),
        rff_features=int(manifest["rff_features"]),
        seed=args.seed,
        setting={"seed": args.seed},
    )
    save_csv(output_dir / "generation_metrics.csv", generation_rows)
    cluster_frame, thresholds = audit(arrays, mixture)
    cluster_frame.to_csv(output_dir / "cluster_metrics.csv", index=False)
    plot_intrinsic(
        path=output_dir / "comparison.png", arrays=arrays, mixture=mixture
    )
    scale_weights = sorted({1.0, *weights})
    plot_scale_grid(
        path=output_dir / "all_methods_all_scales.png",
        arrays=arrays,
        mixture=mixture,
        methods=methods,
        weights=scale_weights,
    )
    plot_scale_grid(
        path=output_dir / "all_methods_all_scales_inner_zoom.png",
        arrays=arrays,
        mixture=mixture,
        methods=methods,
        weights=scale_weights,
        zoom_limit=0.85,
    )
    selected = [
        name
        for name in (
            "reference",
            "bayes",
            "x",
            "v",
            "eps",
            "ptg_w1.5",
            "ptg_w2",
            "ptg_eps_w1.5",
            "ptg_eps_w2",
            "ag_early_w1.3",
            "ag_early_w1.5",
            "ag_early_w2",
            "ag_early_w1.5_high03",
            "ag_early_w2_high03",
            "ag_early_w3_mid03_07",
            "ag_early_w4_mid03_07",
            "ag_early_w5_mid03_07",
            "ig_w0",
            "ig_w1",
            "ig_w1.5",
            "ig_w2",
            "ig_v_w0",
            "ig_v_w1",
            "ig_v_w1.5",
            "ig_v_w2",
            "ctig_v_w0",
            "ctig_v_w1",
            "ctig_v_w1.5",
            "ctig_v_w2",
        )
        if name in arrays
    ]
    plot_intrinsic(
        path=output_dir / "comparison_selected.png",
        arrays=arrays,
        mixture=mixture,
        ordered_names=selected,
        title="Selected guidance conditions",
    )
    plot_intrinsic(
        path=output_dir / "comparison_selected_inner_zoom.png",
        arrays=arrays,
        mixture=mixture,
        ordered_names=selected,
        title="Selected guidance conditions: inner spiral zoom",
        zoom_limit=0.85,
    )
    save_json(
        output_dir / "manifest.json",
        {
            "protocol": "same_target_autoguidance_vs_prediction_target_v1",
            "seed": args.seed,
            "architecture": args.architecture,
            "source_root": str(args.source_root),
            "strong_checkpoint": str(strong_checkpoint),
            "weak_early_checkpoint": str(early_checkpoint),
            "weak_small_checkpoint": str(small_checkpoint),
            "weak_small_step": weak_step,
            "ig_checkpoint": (
                str(args.ig_checkpoint.resolve())
                if args.ig_checkpoint is not None
                else None
            ),
            "ig_v_checkpoint": (
                str(args.ig_v_checkpoint.resolve())
                if args.ig_v_checkpoint is not None
                else None
            ),
            "weights": weights,
            "methods": methods,
            "windows": windows,
            "sample_count": args.sample_count,
            "sample_steps": args.sample_steps,
            "sampler": args.sampler,
            "schedule_rho": args.schedule_rho,
            "initial_state": args.initial_state,
            "sample_t_min": sample_t_min,
            "sample_t_max": sample_t_max,
            "same_initial_noise": True,
            "reference_source": reference_source,
            "cluster_thresholds": thresholds,
        },
    )
    summary = pd.DataFrame(generation_rows).merge(
        cluster_frame, on="condition", how="left"
    )
    summary.to_csv(output_dir / "summary.csv", index=False)
    plot_scale_metrics(
        path=output_dir / "all_methods_scale_metrics.png",
        summary=summary,
        cluster=cluster_frame,
        arrays=arrays,
        mixture=mixture,
        methods=methods,
        weights=scale_weights,
    )
    print(
        summary[
            [
                "condition",
                "latent_swd",
                "pushforward_swd",
                "mean_nll",
                "intrinsic_bridge_rate",
                "mean_adjacent_log_density_contrast",
                "component_jsd_x",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
