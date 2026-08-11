#!/usr/bin/env python3
"""Train exact-Bayes prediction targets once and evaluate fixed milestones.

The v5 sweep evaluates one endpoint per run.  This trajectory runner is the
follow-up for the regime-selection question: when does x-prediction become a
good but still imperfect generator while v-prediction remains only moderately
worse?  Every requested milestone uses the same held-out examples, rollout
initial states, reference samples, SWD projections, and RFF features.

Milestones are discovery data.  A successful step/gamma must subsequently be
frozen and checked on new seeds; the runner deliberately preserves all failed
milestones as well as successful ones.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from experiments.run_prediction_target_bayes_oracle_v5 import (
    EPS,
    FrozenPushforward,
    TangentGaussianMixture,
    build_same_init_models,
    evaluate_generation,
    evaluate_teacher,
    loss_for_output,
    parse_float_list,
    parse_int_list,
    parse_str_list,
    plot_jit_style,
    plot_setting,
    prediction_clean,
    sample_condition,
    save_csv,
    save_json,
    set_seed,
    stable_seed,
    tag_float,
    validation_metrics,
)


def recent_relative_change(
    history: Sequence[dict[str, float]], target: str, points: int = 3
) -> float:
    """Relative validation-excess change over the latest logged interval."""
    if len(history) < 2:
        return float("nan")
    values = np.asarray(
        [row[f"{target}_excess_mse"] for row in history[-points:]], dtype=float
    )
    return float((values[-1] - values[0]) / max(abs(values[0]), EPS))


def recent_relative_span(
    history: Sequence[dict[str, float]], target: str, points: int = 3
) -> float:
    """Relative range of recent validation excess; catches hidden oscillation."""
    if len(history) < 2:
        return float("nan")
    values = np.asarray(
        [row[f"{target}_excess_mse"] for row in history[-points:]], dtype=float
    )
    return float((values.max() - values.min()) / max(abs(values.mean()), EPS))


def checkpoint_path(setting_dir: Path, step: int) -> Path:
    return setting_dir / "checkpoints" / f"step{step:06d}.pt"


def save_checkpoint(
    *,
    path: Path,
    step: int,
    models: dict[str, nn.Module],
    optimizers: dict[str, torch.optim.Optimizer],
    generator: torch.Generator,
    history: list[dict[str, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "models": {name: model.state_dict() for name, model in models.items()},
        "optimizers": {
            name: optimizer.state_dict() for name, optimizer in optimizers.items()
        },
        "generator_state": generator.get_state(),
        "history": history,
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def latest_checkpoint(setting_dir: Path, max_step: int) -> Path | None:
    paths = sorted((setting_dir / "checkpoints").glob("step*.pt"))
    eligible = [path for path in paths if int(path.stem[4:]) <= max_step]
    return eligible[-1] if eligible else None


def load_checkpoint(
    *,
    path: Path,
    models: dict[str, nn.Module],
    optimizers: dict[str, torch.optim.Optimizer],
    generator: torch.Generator,
    device: torch.device,
) -> tuple[int, list[dict[str, float]]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    for name, model in models.items():
        model.load_state_dict(payload["models"][name])
    for name, optimizer in optimizers.items():
        optimizer.load_state_dict(payload["optimizers"][name])
    # ``map_location=device`` is appropriate for model and optimizer tensors,
    # but it also moves the serialized RNG state.  Generator.set_state expects
    # a CPU ByteTensor even when the generator itself targets CUDA.
    generator.set_state(payload["generator_state"].cpu())
    return int(payload["step"]), list(payload["history"])


def load_or_create_common_samples(
    *,
    common_dir: Path,
    mixture: TangentGaussianMixture,
    sample_count: int,
    sample_batch_size: int,
    sample_steps: int,
    sample_t_max: float,
    sample_t_min: float,
    conversion_clip: float,
    seed: int,
    distribution_signature: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    """Cache model-independent reference and exact-Bayes rollout samples."""
    common_dir.mkdir(parents=True, exist_ok=True)
    reference_path = common_dir / "reference.npy"
    bayes_path = common_dir / "bayes.npy"
    manifest_path = common_dir / "manifest.json"
    expected_manifest = {
        "distribution": distribution_signature,
        "sample_count": sample_count,
        "sample_steps": sample_steps,
        "sample_t_max": sample_t_max,
        "sample_t_min": sample_t_min,
        "conversion_clip": conversion_clip,
        "seed": seed,
    }
    cache_matches = False
    if manifest_path.is_file():
        cache_matches = json.loads(manifest_path.read_text()) == expected_manifest
    if cache_matches and reference_path.is_file() and bayes_path.is_file():
        reference = np.load(reference_path)
        bayes = np.load(bayes_path)
        if len(reference) == sample_count and len(bayes) == sample_count:
            return reference, bayes

    reference_generator = torch.Generator(device=mixture.device.type)
    reference_generator.manual_seed(stable_seed(seed, 607))
    reference_tensor, _ = mixture.sample_clean(
        sample_count, generator=reference_generator
    )
    reference = reference_tensor.float().cpu().numpy()
    bayes = sample_condition(
        models={},
        mixture=mixture,
        kind="bayes",
        strength=0.0,
        sample_count=sample_count,
        batch_size=sample_batch_size,
        steps=sample_steps,
        t_max=sample_t_max,
        t_min=sample_t_min,
        clip=conversion_clip,
        seed=stable_seed(seed, mixture.D, 601),
    )
    np.save(reference_path, reference)
    np.save(bayes_path, bayes)
    save_json(manifest_path, expected_manifest)
    return reference, bayes


def validate_setting_manifest(
    *,
    path: Path,
    expected: dict[str, object],
    resume: bool,
) -> None:
    """Refuse to resume a checkpoint under changed training semantics."""
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != expected:
            raise ValueError(
                f"setting manifest mismatch at {path}; use a new output directory"
            )
        return
    if resume and any(path.parent.glob("checkpoints/step*.pt")):
        raise ValueError(f"checkpoints exist without a setting manifest: {path}")
    save_json(path, expected)


def condition_spec(
    gammas: Sequence[float], geometry_gammas: Sequence[float]
) -> list[tuple[str, str, float]]:
    conditions: list[tuple[str, str, float]] = [
        ("bayes", "bayes", 0.0),
        ("x", "x", 0.0),
        ("v", "v", 0.0),
        ("eps", "eps", 0.0),
    ]
    conditions.extend(
        (f"xv_g{tag_float(gamma)}", "xv", float(gamma)) for gamma in gammas
    )
    for gamma in geometry_gammas:
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
    return conditions


def plot_extrapolation_style(
    path: Path,
    *,
    reference: np.ndarray,
    samples: dict[str, np.ndarray],
    mixture: TangentGaussianMixture,
    positive_conditions: Sequence[tuple[str, float]],
    title: str,
    limit: int = 3000,
) -> None:
    """JiT-like visual with the actual positive extrapolation candidates."""
    columns = [("reference", "Reference"), ("bayes", "Bayes oracle")]
    columns.extend([("x", "x-pred"), ("v", "v-pred")])
    columns.extend(
        (name, f"x + {gamma:g}(x-v)") for name, gamma in positive_conditions
    )
    ambient = {"reference": reference, **samples}
    projected = {
        name: mixture.intrinsic_readout(
            torch.from_numpy(ambient[name][:limit]).to(mixture.device)
        )
        .float()
        .cpu()
        .numpy()
        for name, _ in columns
    }
    low = np.quantile(projected["reference"], 0.005, axis=0)
    high = np.quantile(projected["reference"], 0.995, axis=0)
    center = 0.5 * (low + high)
    radius = max(0.62 * float(np.max(high - low)), 0.25)
    fig, axes = plt.subplots(
        1, len(columns), figsize=(3.45 * len(columns), 3.8), sharex=True, sharey=True
    )
    reference_points = projected["reference"]
    for axis, (name, label) in zip(np.atleast_1d(axes), columns):
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
            alpha=0.45,
            color="#2878b5" if name == "reference" else "#d95f02",
            linewidths=0,
            rasterized=True,
        )
        axis.set_title(label, fontsize=11)
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
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


@torch.inference_mode()
def sample_conditions_joint(
    *,
    models: dict[str, nn.Module],
    mixture: TangentGaussianMixture,
    conditions: Sequence[tuple[str, str, float]],
    sample_count: int,
    batch_size: int,
    steps: int,
    t_max: float,
    t_min: float,
    clip: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """Roll out many guidance conditions in one concatenated model batch.

    Each condition keeps an independent state trajectory, but all begin from
    the exact same sampled p_t state.  Concatenation only removes repeated
    model launches; it does not couple condition updates.
    """
    active = [condition for condition in conditions if condition[1] != "bayes"]
    if not active:
        return {}
    outputs: dict[str, list[np.ndarray]] = {name: [] for name, _, _ in active}
    grid = torch.linspace(t_max, t_min, steps + 1, device=mixture.device)
    condition_count = len(active)

    def predict(states: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        n = states.shape[1]
        flat_state = states.reshape(condition_count * n, mixture.D)
        flat_time = time.expand(condition_count * n)
        clean = {
            target: prediction_clean(models, flat_state, flat_time, target, clip)
            .reshape(condition_count, n, mixture.D)
            for target in ("x", "v", "eps")
        }
        chosen = []
        for index, (_, kind, strength) in enumerate(active):
            if kind in {"x", "v", "eps"}:
                chosen.append(clean[kind][index])
                continue
            if kind not in {"xv", "xv_tangent", "xv_normal"}:
                raise ValueError(f"unsupported joint condition: {kind}")
            clean_x = clean["x"][index]
            gap = clean_x - clean["v"][index]
            if kind == "xv":
                direction = gap
            else:
                component = mixture.nearest_components(clean_x)
                tangent, normal = mixture.split_by_component(gap, component)
                direction = tangent if kind == "xv_tangent" else normal
            chosen.append(clean_x + float(strength) * direction)
        return torch.stack(chosen, dim=0)

    for start in range(0, sample_count, batch_size):
        n = min(batch_size, sample_count - start)
        generator = torch.Generator(device=mixture.device.type)
        generator.manual_seed(seed + start)
        clean_start, _ = mixture.sample_clean(n, generator=generator)
        eps = torch.randn(
            clean_start.shape, device=mixture.device, generator=generator
        )
        initial = (1.0 - float(t_max)) * clean_start + float(t_max) * eps
        states = initial.unsqueeze(0).expand(condition_count, -1, -1).clone()
        for index in range(steps):
            t_now = grid[index]
            t_next = grid[index + 1]
            time = torch.full((1,), float(t_now), device=mixture.device)
            clean = predict(states, time)
            velocity = (states - clean) / time.clamp_min(clip)
            states = states + (t_next - t_now) * velocity
        time = torch.full((1,), float(grid[-1]), device=mixture.device)
        final = predict(states, time)
        for index, (name, _, _) in enumerate(active):
            outputs[name].append(final[index].float().cpu().numpy())
    return {name: np.concatenate(parts, axis=0) for name, parts in outputs.items()}


def evaluate_milestone(
    *,
    args: argparse.Namespace,
    models: dict[str, nn.Module],
    mixture: TangentGaussianMixture,
    architecture: str,
    hidden: int,
    seed: int,
    step: int,
    history: list[dict[str, float]],
    output_dir: Path,
    reference: np.ndarray,
    bayes: np.ndarray,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    complete_path = output_dir / "complete.json"
    if args.resume and complete_path.is_file():
        return json.loads((output_dir / "summary.json").read_text())

    for model in models.values():
        model.eval()
    setting = {
        "seed": seed,
        "step": step,
        "D": mixture.D,
        "components": mixture.components,
        "sigma_tangent": mixture.sigma_tangent,
        "sigma_normal": mixture.sigma_normal,
        "architecture": architecture,
        "hidden": hidden,
        "loss_space": args.loss_space,
    }
    teacher = evaluate_teacher(
        models=models,
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

    conditions = condition_spec(args.gammas, args.geometry_gammas)
    samples: dict[str, np.ndarray] = {"bayes": bayes}
    sample_seed = stable_seed(seed, mixture.D, 601)
    print(
        f"[sample-joint] step={step} {architecture} H={hidden} "
        f"conditions={len(conditions) - 1}",
        flush=True,
    )
    joint_samples = sample_conditions_joint(
        models=models,
        mixture=mixture,
        conditions=conditions,
        sample_count=args.sample_count,
        batch_size=args.sample_batch_size,
        steps=args.sample_steps,
        t_max=args.sample_t_max,
        t_min=args.sample_t_min,
        clip=args.conversion_clip,
        seed=sample_seed,
    )
    samples.update(joint_samples)
    for name, _, _ in conditions:
        if name == "bayes":
            continue
        if args.save_samples:
            np.save(output_dir / f"samples_{name}.npy", samples[name])

    np.savez_compressed(
        output_dir / "jit_projection.npz",
        reference=mixture.intrinsic_readout(
            torch.from_numpy(reference).to(mixture.device)
        )
        .float()
        .cpu()
        .numpy(),
        **{
            key: mixture.intrinsic_readout(
                torch.from_numpy(samples[key]).to(mixture.device)
            )
            .float()
            .cpu()
            .numpy()
            for key in ("bayes", "x", "eps", "v")
        },
    )
    plot_jit_style(
        output_dir / "jit_style.png",
        reference=reference,
        samples=samples,
        mixture=mixture,
        title=(
            f"Prediction-target trajectory: {architecture}, H={hidden}, "
            f"step={step}, D={mixture.D}"
        ),
    )
    positive_conditions = [
        (name, strength)
        for name, kind, strength in conditions
        if kind == "xv" and strength > 0
    ]
    if len(positive_conditions) > 3:
        positive_conditions = [
            positive_conditions[0],
            positive_conditions[len(positive_conditions) // 2],
            positive_conditions[-1],
        ]
    plot_extrapolation_style(
        output_dir / "extrapolation_style.png",
        reference=reference,
        samples=samples,
        mixture=mixture,
        positive_conditions=positive_conditions,
        title=(
            f"Positive x-away-from-v candidates: {architecture}, H={hidden}, "
            f"step={step}"
        ),
    )
    pushforward = FrozenPushforward(
        mixture.D, args.pushforward_width, args.pushforward_seed
    )
    generation = evaluate_generation(
        samples=samples,
        conditions=conditions,
        reference=reference,
        mixture=mixture,
        pushforward=pushforward,
        projections=args.swd_projections,
        rff_features=args.rff_features,
        seed=stable_seed(seed, 613),
        setting=setting,
    )
    save_csv(output_dir / "generation_metrics.csv", generation)
    plot_setting(output_dir / "diagnostic.png", teacher, generation)

    by_name = {row["condition"]: row for row in generation}
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
    positive_rows = [
        row
        for row in generation
        if row["kind"] == "xv" and float(row["strength"]) > 0
    ]
    metric_names = (
        "latent_swd",
        "latent_mmd_rff",
        "pushforward_swd",
        "pushforward_mmd_rff",
    )
    reference_tangent = float(by_name["x"]["reference_nearest_tangent_rms"])
    reference_normal = float(by_name["x"]["reference_nearest_normal_rms"])
    for row in positive_rows:
        distribution_wins = sum(
            float(row[metric]) < float(by_name["x"][metric])
            for metric in metric_names
        )
        structure_wins = sum(
            (
                float(row["component_jsd"])
                < float(by_name["x"]["component_jsd"])
            ,
                abs(float(row["nearest_tangent_rms"]) - reference_tangent)
                < abs(
                    float(by_name["x"]["nearest_tangent_rms"])
                    - reference_tangent
                )
            ,
                abs(float(row["nearest_normal_rms"]) - reference_normal)
                < abs(
                    float(by_name["x"]["nearest_normal_rms"])
                    - reference_normal
                )
            )
        )
        row["distribution_metrics_better"] = distribution_wins
        row["structure_metrics_better"] = structure_wins
        row["metrics_better_than_x"] = distribution_wins + structure_wins
    best_joint = max(
        positive_rows,
        key=lambda row: (
            int(row["metrics_better_than_x"]),
            -float(row["latent_swd"]),
        ),
        default=None,
    )
    x_swd = float(by_name["x"]["latent_swd"])
    v_swd = float(by_name["v"]["latent_swd"])
    bayes_swd = float(by_name["bayes"]["latent_swd"])
    x_push = float(by_name["x"]["pushforward_swd"])
    v_push = float(by_name["v"]["pushforward_swd"])
    bayes_push = float(by_name["bayes"]["pushforward_swd"])
    x_tangent = float(by_name["x"]["nearest_tangent_rms"])
    x_normal = float(by_name["x"]["nearest_normal_rms"])
    x_component_jsd = float(by_name["x"]["component_jsd"])
    bayes_component_jsd = float(by_name["bayes"]["component_jsd"])
    x_recent_change = recent_relative_change(history, "x")
    v_recent_change = recent_relative_change(history, "v")
    x_recent_span = recent_relative_span(history, "x")
    v_recent_span = recent_relative_span(history, "v")
    quality_band = (
        x_swd <= 1.5 * bayes_swd
        and x_push <= 1.5 * bayes_push
        and x_swd < v_swd <= 2.5 * x_swd
        and x_push < v_push <= 2.5 * x_push
        and (2.0 / 3.0) * reference_tangent <= x_tangent <= 1.5 * reference_tangent
        and (2.0 / 3.0) * reference_normal <= x_normal <= 1.5 * reference_normal
        and x_component_jsd <= 2.5 * max(bayes_component_jsd, EPS)
        and abs(x_recent_change) <= 0.1
        and abs(v_recent_change) <= 0.1
        and x_recent_span <= 0.15
        and v_recent_span <= 0.15
    )
    summary: dict[str, object] = {
        **setting,
        **{f"mean_{key}": value for key, value in mean_teacher.items()},
        "x_excess_over_bayes_risk": mean_teacher["x_excess_mse"]
        / max(mean_teacher["bayes_risk_mse"], EPS),
        "v_excess_over_bayes_risk": mean_teacher["v_excess_mse"]
        / max(mean_teacher["bayes_risk_mse"], EPS),
        "x_recent_relative_change": x_recent_change,
        "v_recent_relative_change": v_recent_change,
        "x_recent_relative_span": x_recent_span,
        "v_recent_relative_span": v_recent_span,
        "bayes_latent_swd": bayes_swd,
        "x_latent_swd": x_swd,
        "v_latent_swd": v_swd,
        "bayes_pushforward_swd": bayes_push,
        "x_pushforward_swd": x_push,
        "v_pushforward_swd": v_push,
        "x_over_bayes_latent_swd": x_swd / max(bayes_swd, EPS),
        "v_over_x_latent_swd": v_swd / max(x_swd, EPS),
        "x_over_bayes_pushforward_swd": x_push / max(bayes_push, EPS),
        "v_over_x_pushforward_swd": v_push / max(x_push, EPS),
        "reference_nearest_tangent_rms": reference_tangent,
        "reference_nearest_normal_rms": reference_normal,
        "x_nearest_tangent_rms": x_tangent,
        "x_nearest_normal_rms": x_normal,
        "x_tangent_over_reference": x_tangent / max(reference_tangent, EPS),
        "x_normal_over_reference": x_normal / max(reference_normal, EPS),
        "bayes_component_jsd": bayes_component_jsd,
        "x_component_jsd": x_component_jsd,
        "x_over_bayes_component_jsd": x_component_jsd
        / max(bayes_component_jsd, EPS),
        "quality_band": bool(quality_band),
        "best_positive_gamma": (
            float(best_joint["strength"]) if best_joint is not None else None
        ),
        "best_positive_metrics_better": (
            int(best_joint["metrics_better_than_x"])
            if best_joint is not None
            else 0
        ),
        "candidate_success": bool(
            quality_band
            and best_joint is not None
            and int(best_joint["metrics_better_than_x"]) == len(metric_names) + 3
        ),
    }
    save_json(output_dir / "summary.json", summary)
    save_json(complete_path, {"status": "complete"})
    return summary


def train_trajectory(
    *,
    args: argparse.Namespace,
    mixture: TangentGaussianMixture,
    architecture: str,
    hidden: int,
    seed: int,
    setting_dir: Path,
    reference: np.ndarray,
    bayes: np.ndarray,
    device: torch.device,
) -> None:
    setting_manifest = {
        "D": mixture.D,
        "components": mixture.components,
        "curvature": args.curvature,
        "frequency_scale": args.frequency_scale,
        "center_rms": args.center_rms,
        "sigma_tangent": mixture.sigma_tangent,
        "sigma_normal": mixture.sigma_normal,
        "mixture_seed": args.mixture_seed,
        "architecture": architecture,
        "hidden": hidden,
        "depth": args.depth,
        "time_dim": args.time_dim,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "loss_space": args.loss_space,
        "t_min": args.t_min,
        "t_max": args.t_max,
        "time_sampler": args.time_sampler,
        "time_logit_mean": args.time_logit_mean,
        "time_logit_std": args.time_logit_std,
        "conversion_clip": args.conversion_clip,
        "log_every": args.log_every,
        "validation_samples": args.validation_samples,
        "eval_times": list(args.eval_times),
        "teacher_samples": args.teacher_samples,
        "eval_batch_size": args.eval_batch_size,
        "gammas": list(args.gammas),
        "geometry_gammas": list(args.geometry_gammas),
        "sample_count": args.sample_count,
        "sample_batch_size": args.sample_batch_size,
        "sample_steps": args.sample_steps,
        "sample_t_max": args.sample_t_max,
        "sample_t_min": args.sample_t_min,
        "swd_projections": args.swd_projections,
        "rff_features": args.rff_features,
        "pushforward_width": args.pushforward_width,
        "pushforward_seed": args.pushforward_seed,
        "seed": seed,
    }
    setting_dir.mkdir(parents=True, exist_ok=True)
    validate_setting_manifest(
        path=setting_dir / "training_manifest.json",
        expected=setting_manifest,
        resume=args.resume,
    )
    models = build_same_init_models(
        architecture,
        D=mixture.D,
        hidden=hidden,
        depth=args.depth,
        time_dim=args.time_dim,
        device=device,
        seed=stable_seed(seed, 101),
    )
    optimizers = {
        name: torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        for name, model in models.items()
    }
    generator = torch.Generator(device=device.type)
    generator.manual_seed(stable_seed(seed, 211))

    val_generator = torch.Generator(device=device.type)
    val_generator.manual_seed(stable_seed(seed, 223))
    val_x, val_eps, val_t, val_x_t, _ = mixture.noised_batch(
        args.validation_samples,
        t_min=args.t_min,
        t_max=args.t_max,
        time_sampler=args.time_sampler,
        time_logit_mean=args.time_logit_mean,
        time_logit_std=args.time_logit_std,
        generator=val_generator,
    )
    with torch.inference_mode():
        val_bayes = mixture.posterior_clean(val_x_t, val_t)

    start_step = 0
    history: list[dict[str, float]] = []
    checkpoint = latest_checkpoint(setting_dir, max(args.milestones))
    if args.resume and checkpoint is not None:
        start_step, history = load_checkpoint(
            path=checkpoint,
            models=models,
            optimizers=optimizers,
            generator=generator,
            device=device,
        )
        print(f"[resume] {architecture} H={hidden} from step {start_step}")

    def run_evaluation(step: int) -> None:
        evaluate_milestone(
            args=args,
            models=models,
            mixture=mixture,
            architecture=architecture,
            hidden=hidden,
            seed=seed,
            step=step,
            history=history,
            output_dir=setting_dir / f"step{step:06d}",
            reference=reference,
            bayes=bayes,
        )

    if start_step in args.milestones:
        run_evaluation(start_step)

    milestone_set = set(args.milestones)
    final_step = max(args.milestones)
    for step in range(start_step + 1, final_step + 1):
        x, eps, t, x_t, _ = mixture.noised_batch(
            args.batch_size,
            t_min=args.t_min,
            t_max=args.t_max,
            time_sampler=args.time_sampler,
            time_logit_mean=args.time_logit_mean,
            time_logit_std=args.time_logit_std,
            generator=generator,
        )
        train_losses: dict[str, float] = {}
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
                loss_space=args.loss_space,
                conversion_clip=args.conversion_clip,
            )
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizers[target].step()
            train_losses[target] = float(loss.detach().cpu())

        if step == 1 or step % args.log_every == 0 or step in milestone_set:
            for model in models.values():
                model.eval()
            val = validation_metrics(
                models,
                x=val_x,
                eps=val_eps,
                t=val_t,
                x_t=val_x_t,
                bayes=val_bayes,
                conversion_clip=args.conversion_clip,
            )
            row = {
                "step": step,
                **{
                    f"train_{name}_loss": value
                    for name, value in train_losses.items()
                },
                **val,
            }
            history.append(row)
            save_csv(setting_dir / "train_history.csv", history)
            print(
                f"[{architecture} H={hidden}] {step}/{final_step} "
                + " ".join(
                    f"{name}:loss={train_losses[name]:.4g},"
                    f"excess={val[f'{name}_excess_mse']:.4g}"
                    for name in ("x", "v", "eps")
                ),
                flush=True,
            )

        if step in milestone_set:
            save_checkpoint(
                path=checkpoint_path(setting_dir, step),
                step=step,
                models=models,
                optimizers=optimizers,
                generator=generator,
                history=copy.deepcopy(history),
            )
            run_evaluation(step)


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
        "--architectures", type=parse_str_list, default=parse_str_list("residual")
    )
    parser.add_argument(
        "--hidden-dims", type=parse_int_list, default=parse_int_list("64,128")
    )
    parser.add_argument(
        "--milestones",
        type=parse_int_list,
        default=parse_int_list("3000,6000,10000,15000,20000,30000"),
    )
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--time-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--loss-space", choices=("v", "direct"), default="v")
    parser.add_argument("--t-min", type=float, default=0.02)
    parser.add_argument("--t-max", type=float, default=0.98)
    parser.add_argument(
        "--time-sampler",
        choices=("uniform", "logit_normal"),
        default="uniform",
    )
    parser.add_argument(
        "--time-logit-mean",
        type=float,
        default=0.8,
        help=(
            "Mean in the noise-time coordinate. JiT's data-time mean -0.8 "
            "maps to +0.8 because this experiment uses t=1 at the noise end."
        ),
    )
    parser.add_argument("--time-logit-std", type=float, default=0.8)
    parser.add_argument("--conversion-clip", type=float, default=0.02)
    parser.add_argument("--log-every", type=int, default=500)
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
        default=parse_float_list("-0.03,0.01,0.03,0.1"),
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
        "--seeds", type=parse_int_list, default=parse_int_list("20260901")
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-samples", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.milestones = sorted(set(args.milestones))
    if not args.milestones or min(args.milestones) <= 0:
        raise ValueError("milestones must be positive")
    if any(
        name not in {"jit_relu", "plain", "residual", "residual_skip"}
        for name in args.architectures
    ):
        raise ValueError(f"unknown architectures: {args.architectures}")
    device = torch.device(args.device)
    args.output_root.mkdir(parents=True, exist_ok=True)
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
        seed_dir = args.output_root / f"seed{seed}"
        save_json(
            seed_dir / "manifest.json",
            {
                **vars(args),
                "output_root": str(args.output_root),
                "device": str(device),
                "seed": seed,
                "role": "discovery trajectory; confirmation requires new seeds",
                "quality_band": (
                    "x latent/pushforward SWD <=1.5x Bayes; x tangent/normal "
                    "width <=1.5x reference; x component JSD <=2.5x Bayes; "
                    "x better than v; v no worse than 2.5x x; recent x/v "
                    "Bayes-excess change <=10% and span <=15%"
                ),
            },
        )
        reference, bayes = load_or_create_common_samples(
            common_dir=seed_dir / "common",
            mixture=mixture,
            sample_count=args.sample_count,
            sample_batch_size=args.sample_batch_size,
            sample_steps=args.sample_steps,
            sample_t_max=args.sample_t_max,
            sample_t_min=args.sample_t_min,
            conversion_clip=args.conversion_clip,
            seed=seed,
            distribution_signature={
                "D": args.D,
                "components": args.components,
                "curvature": args.curvature,
                "frequency_scale": args.frequency_scale,
                "center_rms": args.center_rms,
                "sigma_tangent": args.sigma_tangent,
                "sigma_normal": args.sigma_normal,
                "mixture_seed": args.mixture_seed,
            },
        )
        for architecture in args.architectures:
            for hidden in args.hidden_dims:
                train_trajectory(
                    args=args,
                    mixture=mixture,
                    architecture=architecture,
                    hidden=hidden,
                    seed=seed,
                    setting_dir=seed_dir / architecture / f"H{hidden}",
                    reference=reference,
                    bayes=bayes,
                    device=device,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
