#!/usr/bin/env python3
"""Compare scalar and operator-valued prediction targets on the rank toy.

The experiment uses the same convention as prediction-target toy v4:

    z_t = (1 - t) x + t epsilon,    velocity = epsilon - x.

For a symmetric operator K with eigenvalues in [0, 1], define

    u_K = K x - (I - K) epsilon.

When K has one eigenvalue k_parallel on the known data subspace and another
eigenvalue k_normal on its orthogonal complement, the velocity is recovered
exactly from a network prediction u_hat as

    v_hat = D_K(t)^-1 ((2 K - I) z_t - u_hat),
    D_K(t) = (1 - t) I + (2 t - 1) K.

Scalar k-Diff is the special case k_parallel == k_normal.  The operator cases
keep k_parallel=1/2 (velocity prediction up to scale) and vary k_normal toward
1 (clean prediction).  A separate projected-oracle condition forces the known
normal clean target to zero; keeping it separate avoids crediting manifold
knowledge to target parameterization alone.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
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
    tag_float,
)
from experiments.run_prediction_target_rank_symmetry_toy import (
    RankOutputMLP,
    covariance_effective_rank,
    evaluate_generation,
    save_csv,
    set_seed,
)


@dataclass(frozen=True)
class TargetSpec:
    name: str
    tangent_k: float
    normal_k: float
    project_normal_output: bool = False

    def __post_init__(self) -> None:
        if not (0.0 <= self.tangent_k <= 1.0):
            raise ValueError("tangent_k must be in [0,1]")
        if not (0.0 <= self.normal_k <= 1.0):
            raise ValueError("normal_k must be in [0,1]")
        if self.project_normal_output and self.normal_k != 1.0:
            raise ValueError("normal output can only be projected for the exact x target")


def project_data_subspace(value: torch.Tensor, embedding: CurvedEmbedding) -> torch.Tensor:
    basis = embedding.Q[:, :2].to(dtype=value.dtype)
    return (value @ basis) @ basis.T


def split_subspaces(
    value: torch.Tensor, embedding: CurvedEmbedding
) -> tuple[torch.Tensor, torch.Tensor]:
    tangent = project_data_subspace(value, embedding)
    return tangent, value - tangent


def scalar_denominator(time: torch.Tensor, k: float) -> torch.Tensor:
    return (1.0 - time) + (2.0 * time - 1.0) * float(k)


def generalized_target(
    clean: torch.Tensor,
    epsilon: torch.Tensor,
    embedding: CurvedEmbedding,
    spec: TargetSpec,
) -> torch.Tensor:
    clean_tangent, clean_normal = split_subspaces(clean, embedding)
    eps_tangent, eps_normal = split_subspaces(epsilon, embedding)
    tangent = spec.tangent_k * clean_tangent - (1.0 - spec.tangent_k) * eps_tangent
    normal = spec.normal_k * clean_normal - (1.0 - spec.normal_k) * eps_normal
    if spec.project_normal_output:
        normal = torch.zeros_like(normal)
    return tangent + normal


def velocity_from_operator_output(
    output: torch.Tensor,
    state: torch.Tensor,
    time: torch.Tensor,
    embedding: CurvedEmbedding,
    spec: TargetSpec,
    conversion_clip: float,
) -> torch.Tensor:
    state_tangent, state_normal = split_subspaces(state, embedding)
    output_tangent, output_normal = split_subspaces(output, embedding)
    if spec.project_normal_output:
        output_normal = torch.zeros_like(output_normal)

    tangent_denominator = scalar_denominator(time, spec.tangent_k).clamp_min(
        conversion_clip
    )[:, None]
    normal_denominator = scalar_denominator(time, spec.normal_k).clamp_min(
        conversion_clip
    )[:, None]
    tangent_velocity = (
        (2.0 * spec.tangent_k - 1.0) * state_tangent - output_tangent
    ) / tangent_denominator
    normal_velocity = (
        (2.0 * spec.normal_k - 1.0) * state_normal - output_normal
    ) / normal_denominator
    return tangent_velocity + normal_velocity


def clean_from_operator_output(
    output: torch.Tensor,
    state: torch.Tensor,
    time: torch.Tensor,
    embedding: CurvedEmbedding,
    spec: TargetSpec,
    conversion_clip: float,
) -> torch.Tensor:
    velocity = velocity_from_operator_output(
        output, state, time, embedding, spec, conversion_clip
    )
    return state - time[:, None] * velocity


def effective_output(
    output: torch.Tensor, embedding: CurvedEmbedding, spec: TargetSpec
) -> torch.Tensor:
    if not spec.project_normal_output:
        return output
    return project_data_subspace(output, embedding)


def build_specs(args: argparse.Namespace) -> list[TargetSpec]:
    specs: list[TargetSpec] = []
    for k in args.scalar_ks:
        specs.append(TargetSpec(f"scalar_k{tag_float(k)}", k, k))
    for normal_k in args.operator_normal_ks:
        specs.append(
            TargetSpec(
                f"operator_t050_n{tag_float(normal_k)}",
                tangent_k=0.5,
                normal_k=normal_k,
            )
        )
    if args.include_projected_oracle:
        specs.append(
            TargetSpec(
                "operator_t050_n100_projected",
                tangent_k=0.5,
                normal_k=1.0,
                project_normal_output=True,
            )
        )
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("target specification names must be unique")
    return specs


def build_matched_models(
    *,
    specs: list[TargetSpec],
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
    models: dict[str, RankOutputMLP] = {}
    for spec in specs:
        model = RankOutputMLP(
            D,
            hidden=hidden,
            output_rank=output_rank,
            depth=depth,
            time_dim=time_dim,
        ).to(device)
        model.load_state_dict(state)
        models[spec.name] = model
    return models


def train_models(
    *,
    models: dict[str, RankOutputMLP],
    specs: list[TargetSpec],
    embedding: CurvedEmbedding,
    output_rank: int,
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
    generator.manual_seed(stable_seed(setting_seed, embedding.D, output_rank, 701))
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
        true_velocity = epsilon - clean
        losses: dict[str, float] = {}

        for spec in specs:
            model = models[spec.name]
            optimizer = optimizers[spec.name]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                output = model(state, time)
                velocity = velocity_from_operator_output(
                    output,
                    state,
                    time,
                    embedding,
                    spec,
                    args.conversion_clip,
                )
                loss = F.mse_loss(velocity.float(), true_velocity.float())
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses[spec.name] = float(loss.detach().cpu())

        if step == 1 or step % args.log_every == 0 or step == args.train_steps:
            history.append(
                {"step": step, **{f"loss_{name}": value for name, value in losses.items()}}
            )
            compact = " ".join(f"{name}={losses[name]:.4g}" for name in losses)
            print(
                f"[train D={embedding.D} R={output_rank}] "
                f"{step}/{args.train_steps} {compact}",
                flush=True,
            )
    return history


@torch.inference_mode()
def evaluate_teacher(
    *,
    models: dict[str, RankOutputMLP],
    specs: list[TargetSpec],
    embedding: CurvedEmbedding,
    output_rank: int,
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
                "clean": 0.0,
                "native": 0.0,
                "outputs": [],
            }
            for spec in specs
        }
        generator = torch.Generator(device=device.type)
        generator.manual_seed(
            stable_seed(setting_seed, embedding.D, output_rank, time_index, 809)
        )
        for start in range(0, args.eval_samples, args.eval_batch_size):
            n = min(args.eval_batch_size, args.eval_samples - start)
            intrinsic = sample_spiral_2d(
                n,
                device=device,
                jitter=args.data_jitter,
                generator=generator,
            )
            clean = embedding.embed(intrinsic)
            epsilon = torch.randn(clean.shape, device=device, generator=generator)
            time = torch.full((n,), float(time_value), device=device)
            state = (1.0 - time[:, None]) * clean + time[:, None] * epsilon
            truth = epsilon - clean
            for spec in specs:
                raw_output = models[spec.name](state, time)
                used_output = effective_output(raw_output, embedding, spec)
                velocity = velocity_from_operator_output(
                    raw_output,
                    state,
                    time,
                    embedding,
                    spec,
                    args.conversion_clip,
                )
                predicted_clean = state - time[:, None] * velocity
                native_truth = generalized_target(clean, epsilon, embedding, spec)
                error = velocity - truth
                tangent_error, normal_error = split_subspaces(error, embedding)
                values = sums[spec.name]
                values["velocity"] += float(error.square().sum().cpu())
                values["tangent"] += float(tangent_error.square().sum().cpu())
                values["normal"] += float(normal_error.square().sum().cpu())
                values["clean"] += float((predicted_clean - clean).square().sum().cpu())
                values["native"] += float((used_output - native_truth).square().sum().cpu())
                values["outputs"].append(used_output.cpu())

        denominator = args.eval_samples * embedding.D
        for spec in specs:
            values = sums[spec.name]
            effective_rank, numerical_rank, output_variance = covariance_effective_rank(
                torch.cat(values.pop("outputs"), dim=0)
            )
            rows.append(
                {
                    "seed": experiment_seed,
                    "setting_seed": setting_seed,
                    "D": embedding.D,
                    "output_rank": output_rank,
                    "time": float(time_value),
                    "condition": spec.name,
                    "tangent_k": spec.tangent_k,
                    "normal_k": spec.normal_k,
                    "project_normal_output": spec.project_normal_output,
                    "velocity_mse": values["velocity"] / denominator,
                    "velocity_tangent_mse": values["tangent"] / denominator,
                    "velocity_normal_mse": values["normal"] / denominator,
                    "clean_mse": values["clean"] / denominator,
                    "native_target_mse": values["native"] / denominator,
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
    specs: list[TargetSpec],
    embedding: CurvedEmbedding,
    args: argparse.Namespace,
    setting_seed: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    collected: dict[str, list[np.ndarray]] = {spec.name: [] for spec in specs}
    grid = torch.linspace(
        args.sample_t_max,
        args.sample_t_min,
        args.sample_steps + 1,
        device=device,
    )
    sample_seed = stable_seed(setting_seed, 1213)
    for start in range(0, args.sample_count, args.sample_batch_size):
        n = min(args.sample_batch_size, args.sample_count - start)
        generator = torch.Generator(device=device.type)
        generator.manual_seed(sample_seed + start)
        initial = args.sample_t_max * torch.randn(
            (n, embedding.D), device=device, generator=generator
        )
        states = {spec.name: initial.clone() for spec in specs}
        for index in range(args.sample_steps):
            t_now, t_next = grid[index], grid[index + 1]
            time = t_now.expand(n)
            for spec in specs:
                output = models[spec.name](states[spec.name], time)
                velocity = velocity_from_operator_output(
                    output,
                    states[spec.name],
                    time,
                    embedding,
                    spec,
                    args.conversion_clip,
                )
                states[spec.name] = states[spec.name] + (t_next - t_now) * velocity
        final_time = grid[-1].expand(n)
        for spec in specs:
            output = models[spec.name](states[spec.name], final_time)
            clean = clean_from_operator_output(
                output,
                states[spec.name],
                final_time,
                embedding,
                spec,
                args.conversion_clip,
            )
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
    columns = min(4, len(panels))
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


def run_setting(
    *,
    args: argparse.Namespace,
    specs: list[TargetSpec],
    D: int,
    output_rank: int,
    experiment_seed: int,
    device: torch.device,
) -> tuple[list[dict], list[dict], dict]:
    setting_seed = stable_seed(experiment_seed, D, output_rank, 2027)
    setting_dir = args.output_root / f"seed{experiment_seed}" / f"D{D}" / f"rank{output_rank}"
    if args.resume and (setting_dir / "summary.json").is_file():
        print(f"[resume] {setting_dir}", flush=True)
        with (setting_dir / "teacher_metrics.csv").open(newline="", encoding="utf-8") as handle:
            teacher = list(csv.DictReader(handle))
        with (setting_dir / "generation_metrics.csv").open(newline="", encoding="utf-8") as handle:
            generation = list(csv.DictReader(handle))
        summary = json.loads((setting_dir / "summary.json").read_text(encoding="utf-8"))
        return teacher, generation, summary

    setting_dir.mkdir(parents=True, exist_ok=True)
    set_seed(setting_seed)
    embedding = CurvedEmbedding(
        D,
        curvature=0.0,
        frequency_scale=args.frequency_scale,
        seed=stable_seed(experiment_seed, D, 0, 41),
        device=device,
        scale_mode=args.scale_mode,
    )
    models = build_matched_models(
        specs=specs,
        D=D,
        hidden=args.hidden,
        output_rank=output_rank,
        depth=args.depth,
        time_dim=args.time_dim,
        seed=setting_seed,
        device=device,
    )
    history = train_models(
        models=models,
        specs=specs,
        embedding=embedding,
        output_rank=output_rank,
        args=args,
        setting_seed=setting_seed,
        device=device,
    )
    for model in models.values():
        model.eval()
    teacher = evaluate_teacher(
        models=models,
        specs=specs,
        embedding=embedding,
        output_rank=output_rank,
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
        specs=specs,
        embedding=embedding,
        args=args,
        setting_seed=setting_seed,
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
        rank_dependent_randomness=False,
    )
    for row in generation:
        row["seed"] = experiment_seed
        row["setting_seed"] = setting_seed
        spec = next(value for value in specs if value.name == row["condition"])
        row["tangent_k"] = spec.tangent_k
        row["normal_k"] = spec.normal_k
        row["project_normal_output"] = spec.project_normal_output

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
    if args.save_checkpoints:
        torch.save(
            {name: model.state_dict() for name, model in models.items()},
            setting_dir / "models.pt",
        )

    best_scalar = min(
        (row for row in generation if row["condition"].startswith("scalar_")),
        key=lambda row: float(row["swd_ambient"]),
    )
    best_unprojected_operator = min(
        (
            row
            for row in generation
            if row["condition"].startswith("operator_")
            and not bool(row["project_normal_output"])
        ),
        key=lambda row: float(row["swd_ambient"]),
    )
    summary = {
        "seed": experiment_seed,
        "setting_seed": setting_seed,
        "D": D,
        "output_rank": output_rank,
        "best_scalar_condition": best_scalar["condition"],
        "best_scalar_swd_ambient": float(best_scalar["swd_ambient"]),
        "best_operator_condition": best_unprojected_operator["condition"],
        "best_operator_swd_ambient": float(best_unprojected_operator["swd_ambient"]),
        "operator_minus_scalar_swd_ambient": float(best_unprojected_operator["swd_ambient"])
        - float(best_scalar["swd_ambient"]),
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
    parser.add_argument("--dims", type=parse_int_list, default=parse_int_list("64,512"))
    parser.add_argument("--output-ranks", type=parse_int_list, default=parse_int_list("4,16,64"))
    parser.add_argument("--scalar-ks", type=parse_float_list, default=parse_float_list("0.5,0.65,0.8,0.9,1.0"))
    parser.add_argument("--operator-normal-ks", type=parse_float_list, default=parse_float_list("0.65,0.8,0.9,1.0"))
    parser.add_argument("--include-projected-oracle", action="store_true")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if not (0.0 < args.t_min < args.t_max < 1.0):
        raise ValueError("training times must be strictly inside (0,1)")
    if not (0.0 < args.sample_t_min < args.sample_t_max < 1.0):
        raise ValueError("sampling times must be strictly inside (0,1)")
    specs = build_specs(args)
    if not any(spec.name.startswith("scalar_") for spec in specs):
        raise ValueError("at least one scalar target is required")
    if not any(
        spec.name.startswith("operator_") and not spec.project_normal_output
        for spec in specs
    ):
        raise ValueError("at least one unprojected operator target is required")
    device = torch.device(args.device)
    manifest = {
        "definition": "rank-controlled scalar-k versus operator-K prediction target",
        "path": "z_t=(1-t)x+t epsilon; velocity=epsilon-x",
        "loss": "common recovered-velocity MSE",
        "operator": "K=k_tangent P+k_normal(I-P)",
        "specs": [spec.__dict__ for spec in specs],
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
    all_summaries: list[dict] = []
    for seed in args.seeds:
        for D in args.dims:
            for output_rank in args.output_ranks:
                if output_rank > min(args.hidden, D):
                    continue
                teacher, generation, summary = run_setting(
                    args=args,
                    specs=specs,
                    D=D,
                    output_rank=output_rank,
                    experiment_seed=seed,
                    device=device,
                )
                all_teacher.extend(teacher)
                all_generation.extend(generation)
                all_summaries.append(summary)
    save_csv(args.output_root / "teacher_metrics.csv", all_teacher)
    save_csv(args.output_root / "generation_metrics.csv", all_generation)
    save_csv(args.output_root / "setting_summaries.csv", all_summaries)
    print(
        f"[done] completed {len(all_summaries)} settings at {args.output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
