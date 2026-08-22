#!/usr/bin/env python3
"""Estimate prediction-target projectors without access to the true subspace.

This is the no-leakage counterpart of ``run_prediction_target_operator_k_toy``.
The predictor never sees the embedding matrix used to create the toy data.
Projectors are estimated once from an independent clean sample bank, detached,
and then frozen.  The true subspace is retained only for an oracle upper bound
and post-hoc alignment diagnostics.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.prediction_target_spectral_projector import (
    SpectralProjector,
    SpectralTarget,
    estimate_pca_projector,
    estimate_soft_spectral_projector,
    hard_projector,
    projector_alignment,
    zero_projector,
)
from experiments.run_prediction_target_extrapolation_toy_v4 import (
    CurvedEmbedding,
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


def build_matched_models(
    targets: list[SpectralTarget],
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
    initial_state = copy.deepcopy(base.state_dict())
    models: dict[str, RankOutputMLP] = {}
    for target in targets:
        model = RankOutputMLP(
            D,
            hidden=hidden,
            output_rank=output_rank,
            depth=depth,
            time_dim=time_dim,
        ).to(device)
        model.load_state_dict(initial_state)
        models[target.name] = model
    return models


def random_projector(
    D: int,
    rank: int,
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> SpectralProjector:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.randn(D, rank, generator=generator, dtype=torch.float64)
    basis, _ = torch.linalg.qr(raw, mode="reduced")
    return hard_projector(
        basis.to(device=device, dtype=dtype), source="random_negative_control"
    )


@torch.no_grad()
def estimate_projectors(
    *,
    embedding: CurvedEmbedding,
    args: argparse.Namespace,
    setting_seed: int,
    device: torch.device,
) -> tuple[list[SpectralTarget], list[dict], list[dict]]:
    generator = torch.Generator(device=device.type)
    generator.manual_seed(stable_seed(setting_seed, 3001))
    intrinsic = sample_spiral_2d(
        args.projector_samples,
        device=device,
        jitter=args.data_jitter,
        generator=generator,
    )
    clean_bank = embedding.embed(intrinsic).detach()
    noisy_bank = clean_bank + args.projector_noise_std * torch.randn(
        clean_bank.shape, device=device, generator=generator
    )

    true_basis = embedding.Q[:, : args.intrinsic_dim].to(clean_bank.dtype)
    exact = hard_projector(true_basis, source="oracle_embedding_basis")
    pca_clean, clean_eigenvalues = estimate_pca_projector(
        clean_bank,
        rank=args.intrinsic_dim,
        source="independent_clean_bank_pca",
    )
    pca_noisy, noisy_eigenvalues = estimate_pca_projector(
        noisy_bank,
        rank=args.intrinsic_dim,
        source="independent_noisy_bank_pca",
    )
    spectral_clean, spectral_eigenvalues, spectral_tau = (
        estimate_soft_spectral_projector(
            clean_bank,
            tau_ratio=args.spectral_tau_ratio,
            max_rank=args.spectral_max_rank,
            min_weight=args.spectral_min_weight,
            source="independent_clean_bank_soft_spectrum",
        )
    )
    random = random_projector(
        embedding.D,
        args.intrinsic_dim,
        seed=stable_seed(setting_seed, 3007),
        device=device,
        dtype=clean_bank.dtype,
    )
    zero = zero_projector(
        embedding.D,
        device=device,
        dtype=clean_bank.dtype,
        source="scalar_identity_operator",
    )

    target_projectors = [
        ("scalar_k090", zero, 0.9, 0.9),
        ("exact_operator", exact, 0.5, 0.9),
        ("pca_clean_operator", pca_clean, 0.5, 0.9),
        ("spectral_clean_operator", spectral_clean, 0.5, 0.9),
        ("pca_noisy_operator", pca_noisy, 0.5, 0.9),
        ("random_operator", random, 0.5, 0.9),
    ]
    targets = [
        SpectralTarget(name, projector, tangent_k, normal_k)
        for name, projector, tangent_k, normal_k in target_projectors
    ]

    diagnostics: list[dict] = []
    for target in targets:
        alignment = projector_alignment(target.projector, true_basis)
        diagnostics.append(
            {
                "condition": target.name,
                "source": target.projector.source,
                "stored_rank": target.projector.rank,
                "projector_trace": float(target.projector.weights.sum().cpu()),
                "projector_weight_min": float(
                    target.projector.weights.min().cpu()
                    if target.projector.rank
                    else 0.0
                ),
                "projector_weight_max": float(
                    target.projector.weights.max().cpu()
                    if target.projector.rank
                    else 0.0
                ),
                "tangent_k": target.tangent_k,
                "normal_k": target.normal_k,
                **alignment,
            }
        )

    spectra: list[dict] = []
    spectrum_map = {
        "pca_clean": clean_eigenvalues,
        "pca_noisy": noisy_eigenvalues,
        "spectral_clean": spectral_eigenvalues,
    }
    for source, eigenvalues in spectrum_map.items():
        for index, eigenvalue in enumerate(
            eigenvalues[: args.spectrum_report_rank].detach().cpu().tolist()
        ):
            row = {"source": source, "index": index, "eigenvalue": eigenvalue}
            if source == "spectral_clean":
                row["tau"] = spectral_tau
                row["shrinkage_weight"] = eigenvalue / (eigenvalue + spectral_tau)
            spectra.append(row)
    return targets, diagnostics, spectra


def train_models(
    *,
    models: dict[str, RankOutputMLP],
    targets: list[SpectralTarget],
    embedding: CurvedEmbedding,
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
    generator.manual_seed(stable_seed(setting_seed, 4001))
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
        for target in targets:
            model = models[target.name]
            optimizer = optimizers[target.name]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                output = model(state, time)
                velocity = target.velocity(
                    output, state, time, args.conversion_clip
                )
                loss = F.mse_loss(velocity.float(), true_velocity.float())
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses[target.name] = float(loss.detach().cpu())
        if step == 1 or step % args.log_every == 0 or step == args.train_steps:
            row = {"step": step}
            row.update({f"loss_{name}": value for name, value in losses.items()})
            history.append(row)
            print(
                f"[train D={embedding.D} R={args.output_rank}] "
                f"{step}/{args.train_steps} "
                + " ".join(f"{name}={value:.4g}" for name, value in losses.items()),
                flush=True,
            )
    return history


@torch.inference_mode()
def evaluate_teacher(
    *,
    models: dict[str, RankOutputMLP],
    targets: list[SpectralTarget],
    embedding: CurvedEmbedding,
    args: argparse.Namespace,
    setting_seed: int,
    experiment_seed: int,
    device: torch.device,
) -> list[dict]:
    true_basis = embedding.Q[:, : args.intrinsic_dim].to(device=device)
    rows: list[dict] = []
    for time_index, time_value in enumerate(args.eval_times):
        generator = torch.Generator(device=device.type)
        generator.manual_seed(stable_seed(setting_seed, time_index, 5003))
        accumulators = {
            target.name: {
                "velocity": 0.0,
                "tangent": 0.0,
                "normal": 0.0,
                "native": 0.0,
                "outputs": [],
            }
            for target in targets
        }
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
            for target in targets:
                output = models[target.name](state, time)
                velocity = target.velocity(
                    output, state, time, args.conversion_clip
                )
                native_truth = target.target(clean, epsilon)
                error = velocity - truth
                tangent_error = (error @ true_basis) @ true_basis.T
                normal_error = error - tangent_error
                values = accumulators[target.name]
                values["velocity"] += float(error.square().sum().cpu())
                values["tangent"] += float(tangent_error.square().sum().cpu())
                values["normal"] += float(normal_error.square().sum().cpu())
                values["native"] += float((output - native_truth).square().sum().cpu())
                values["outputs"].append(output.cpu())
        denominator = args.eval_samples * embedding.D
        for target in targets:
            values = accumulators[target.name]
            effective_rank, numerical_rank, variance = covariance_effective_rank(
                torch.cat(values.pop("outputs"), dim=0)
            )
            rows.append(
                {
                    "seed": experiment_seed,
                    "setting_seed": setting_seed,
                    "D": embedding.D,
                    "output_rank": args.output_rank,
                    "time": float(time_value),
                    "condition": target.name,
                    "velocity_mse": values["velocity"] / denominator,
                    "velocity_tangent_mse": values["tangent"] / denominator,
                    "velocity_normal_mse": values["normal"] / denominator,
                    "native_target_mse": values["native"] / denominator,
                    "native_output_effective_rank": effective_rank,
                    "native_output_numerical_rank": numerical_rank,
                    "native_output_variance_per_dim": variance,
                }
            )
    return rows


@torch.inference_mode()
def sample_models(
    *,
    models: dict[str, RankOutputMLP],
    targets: list[SpectralTarget],
    embedding: CurvedEmbedding,
    args: argparse.Namespace,
    setting_seed: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    collected: dict[str, list[np.ndarray]] = {target.name: [] for target in targets}
    grid = torch.linspace(
        args.sample_t_max,
        args.sample_t_min,
        args.sample_steps + 1,
        device=device,
    )
    sample_seed = stable_seed(setting_seed, 6007)
    for start in range(0, args.sample_count, args.sample_batch_size):
        n = min(args.sample_batch_size, args.sample_count - start)
        generator = torch.Generator(device=device.type).manual_seed(sample_seed + start)
        initial = args.sample_t_max * torch.randn(
            n, embedding.D, device=device, generator=generator
        )
        states = {target.name: initial.clone() for target in targets}
        for index in range(args.sample_steps):
            t_now, t_next = grid[index], grid[index + 1]
            time = t_now.expand(n)
            for target in targets:
                output = models[target.name](states[target.name], time)
                velocity = target.velocity(
                    output, states[target.name], time, args.conversion_clip
                )
                states[target.name] = states[target.name] + (t_next - t_now) * velocity
        final_time = grid[-1].expand(n)
        for target in targets:
            output = models[target.name](states[target.name], final_time)
            clean = target.clean(
                output, states[target.name], final_time, args.conversion_clip
            )
            collected[target.name].append(clean.cpu().numpy())
    return {name: np.concatenate(parts) for name, parts in collected.items()}


def plot_generation(
    path: Path,
    *,
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
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, (name, values) in zip(axes.flat, panels):
        axis.axis("on")
        axis.scatter(values[:max_points, 0], values[:max_points, 1], s=2, alpha=0.45)
        axis.set_title(name)
        axis.set_aspect("equal")
        axis.set_xlim(-1.9, 1.9)
        axis.set_ylim(-1.9, 1.9)
        axis.grid(alpha=0.15)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run_seed(
    *,
    args: argparse.Namespace,
    experiment_seed: int,
    device: torch.device,
) -> tuple[list[dict], list[dict], list[dict]]:
    setting_seed = stable_seed(experiment_seed, args.D, args.output_rank, 7001)
    output_dir = args.output_root / f"seed{experiment_seed}"
    if args.resume and (output_dir / "summary.json").is_file():
        print(f"[resume] {output_dir}", flush=True)
        with (output_dir / "generation_metrics.csv").open(newline="", encoding="utf-8") as handle:
            generation = list(csv.DictReader(handle))
        with (output_dir / "teacher_metrics.csv").open(newline="", encoding="utf-8") as handle:
            teacher = list(csv.DictReader(handle))
        with (output_dir / "projector_diagnostics.csv").open(newline="", encoding="utf-8") as handle:
            diagnostics = list(csv.DictReader(handle))
        return teacher, generation, diagnostics

    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(setting_seed)
    embedding = CurvedEmbedding(
        args.D,
        curvature=0.0,
        frequency_scale=args.frequency_scale,
        seed=stable_seed(experiment_seed, args.D, 41),
        device=device,
        scale_mode=args.scale_mode,
    )
    targets, diagnostics, spectra = estimate_projectors(
        embedding=embedding,
        args=args,
        setting_seed=setting_seed,
        device=device,
    )
    for row in diagnostics:
        row.update(
            {
                "seed": experiment_seed,
                "setting_seed": setting_seed,
                "D": args.D,
                "output_rank": args.output_rank,
                "projector_samples": args.projector_samples,
                "projector_noise_std": args.projector_noise_std,
            }
        )
    for row in spectra:
        row.update({"seed": experiment_seed, "D": args.D})
    save_csv(output_dir / "projector_diagnostics.csv", diagnostics)
    save_csv(output_dir / "projector_spectra.csv", spectra)

    models = build_matched_models(
        targets,
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
        targets=targets,
        embedding=embedding,
        args=args,
        setting_seed=setting_seed,
        device=device,
    )
    save_csv(output_dir / "train_history.csv", history)
    for model in models.values():
        model.eval()
    teacher = evaluate_teacher(
        models=models,
        targets=targets,
        embedding=embedding,
        args=args,
        setting_seed=setting_seed,
        experiment_seed=experiment_seed,
        device=device,
    )
    save_csv(output_dir / "teacher_metrics.csv", teacher)

    reference_generator = torch.Generator(device=device.type).manual_seed(
        stable_seed(setting_seed, 7013)
    )
    reference_intrinsic = sample_spiral_2d(
        max(2 * args.sample_count, 8192),
        device=device,
        jitter=args.data_jitter,
        generator=reference_generator,
    ).cpu().numpy()
    generated = sample_models(
        models=models,
        targets=targets,
        embedding=embedding,
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
    save_csv(output_dir / "generation_metrics.csv", generation)
    plot_generation(
        output_dir / "generation_scatter.png",
        samples=generated,
        reference_intrinsic=reference_intrinsic,
        embedding=embedding,
        max_points=args.plot_points,
    )
    summary = {
        "seed": experiment_seed,
        "setting_seed": setting_seed,
        "D": args.D,
        "output_rank": args.output_rank,
        "generation": {
            row["condition"]: {
                "swd_2d": float(row["swd_2d"]),
                "swd_ambient": float(row["swd_ambient"]),
                "mmd_2d": float(row["mmd_2d"]),
                "manifold_consistency_rms": float(row["manifold_consistency_rms"]),
            }
            for row in generation
        },
        "projectors": {row["condition"]: row for row in diagnostics},
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return teacher, generation, diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--D", type=int, default=512)
    parser.add_argument("--intrinsic-dim", type=int, default=2)
    parser.add_argument("--output-rank", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--time-dim", type=int, default=32)
    parser.add_argument("--frequency-scale", type=float, default=6.0)
    parser.add_argument("--scale-mode", choices=("constant_norm", "unit_rms"), default="unit_rms")
    parser.add_argument("--projector-samples", type=int, default=1024)
    parser.add_argument("--projector-noise-std", type=float, default=0.25)
    parser.add_argument("--spectral-tau-ratio", type=float, default=1e-3)
    parser.add_argument("--spectral-max-rank", type=int, default=64)
    parser.add_argument("--spectral-min-weight", type=float, default=1e-3)
    parser.add_argument("--spectrum-report-rank", type=int, default=32)
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
    parser.add_argument("--eval-times", type=lambda value: [float(x) for x in value.split(",")], default=[0.1, 0.3, 0.5, 0.7, 0.9])
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
    parser.add_argument("--seeds", type=parse_int_list, default=parse_int_list("20260821,20260822,20260823"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.intrinsic_dim != 2:
        raise ValueError("the current spiral generator has intrinsic_dim=2")
    if args.spectral_max_rank > min(args.projector_samples, args.D):
        raise ValueError("spectral_max_rank exceeds the available sample matrix rank")
    device = torch.device(args.device)
    manifest = {
        "definition": "unknown-projector prediction-target audit",
        "projector_protocol": "estimate once from an independent clean bank, detach, freeze",
        "truth_usage": "exact-P upper bound and post-hoc diagnostics only",
        "path": "z_t=(1-t)x+t epsilon; velocity=epsilon-x",
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
    all_diagnostics: list[dict] = []
    for seed in args.seeds:
        teacher, generation, diagnostics = run_seed(
            args=args, experiment_seed=seed, device=device
        )
        all_teacher.extend(teacher)
        all_generation.extend(generation)
        all_diagnostics.extend(diagnostics)
    save_csv(args.output_root / "teacher_metrics.csv", all_teacher)
    save_csv(args.output_root / "generation_metrics.csv", all_generation)
    save_csv(args.output_root / "projector_diagnostics.csv", all_diagnostics)
    print(f"[done] {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
