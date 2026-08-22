#!/usr/bin/env python3
"""Audit whether two SiT depth differences follow one shared bias direction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchdiffeq import odeint

try:
    from experiments.imagenet100_sit_multiscale_guidance import (
        decompose_weak_head_difference,
    )
    from experiments.imagenet100_sit_multiscale_models import (
        evaluate_source_with_heads,
        load_internal_head_for_source,
        load_sit_field_model,
    )
    from experiments.train_imagenet100_sit_flow import (
        DEFAULT_OFFICIAL_SIT_REPO,
        LATENT_SHAPE,
        NpyMomentsDataset,
        SD_VAE_SCALING_FACTOR,
        atomic_json_dump,
        load_official_sit_module,
        sample_sdvae_posterior,
    )
except ModuleNotFoundError:
    from imagenet100_sit_multiscale_guidance import decompose_weak_head_difference
    from imagenet100_sit_multiscale_models import (
        evaluate_source_with_heads,
        load_internal_head_for_source,
        load_sit_field_model,
    )
    from train_imagenet100_sit_flow import (
        DEFAULT_OFFICIAL_SIT_REPO,
        LATENT_SHAPE,
        NpyMomentsDataset,
        SD_VAE_SCALING_FACTOR,
        atomic_json_dump,
        load_official_sit_module,
        sample_sdvae_posterior,
    )


BASE = Path("/data/users/zhoushunyu/eqvae/imagenet_sit_flow")
DEFAULT_STRONG = BASE / "runs/sit-s-2_seed0/checkpoints/step_00800000.pt"
DEFAULT_DEPTH4 = (
    BASE
    / "multiscale_guidance_study_v1/runs/depth4_v/checkpoints/step_00050000.pt"
)
DEFAULT_DEPTH8 = (
    BASE
    / "runs/sit-s-2_v800-ema_frozen-internal-v-depth8_seed0/"
    "checkpoints/step_00050000.pt"
)
DEFAULT_CACHE = BASE / "imagenet100_cmc_sdvae"
DEFAULT_OUTPUT = BASE / "depth_difference_mechanism_v1/geometry_v800"
DEFAULT_TIMES = (0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95)


def sample_inner(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left.double() * right.double()).flatten(1).sum(1)


def sample_energy(value: torch.Tensor) -> torch.Tensor:
    return sample_inner(value, value)


def geometry_metrics(
    strong: torch.Tensor,
    depth8: torch.Tensor,
    depth4: torch.Tensor,
    *,
    predicted_coefficient: float,
) -> dict[str, torch.Tensor]:
    difference, parallel, orthogonal, coefficient = decompose_weak_head_difference(
        strong,
        depth8,
        depth4,
    )
    reference = strong - depth4
    dot = sample_inner(difference, reference)
    reference_energy = sample_energy(reference)
    difference_energy = sample_energy(difference)
    parallel_energy = sample_energy(parallel)
    orthogonal_energy = sample_energy(orthogonal)
    predicted_residual = difference - float(predicted_coefficient) * reference
    tiny = torch.finfo(torch.float64).tiny
    return {
        "cosine": dot
        / (reference_energy.sqrt() * difference_energy.sqrt()).clamp_min(tiny),
        "projection_coefficient": coefficient,
        "positive_coefficient": (coefficient > 0).double(),
        "reference_rms": (reference_energy / reference[0].numel()).sqrt(),
        "difference_rms": (difference_energy / difference[0].numel()).sqrt(),
        "difference_over_reference_rms": difference_energy.sqrt()
        / reference_energy.sqrt().clamp_min(tiny),
        "parallel_energy_fraction": parallel_energy
        / difference_energy.clamp_min(tiny),
        "orthogonal_energy_fraction": orthogonal_energy
        / difference_energy.clamp_min(tiny),
        "predicted_residual_energy_fraction": sample_energy(predicted_residual)
        / difference_energy.clamp_min(tiny),
        "dot": dot,
        "reference_energy": reference_energy,
        "difference_energy": difference_energy,
    }


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    descriptive = [
        "cosine",
        "projection_coefficient",
        "positive_coefficient",
        "reference_rms",
        "difference_rms",
        "difference_over_reference_rms",
        "parallel_energy_fraction",
        "orthogonal_energy_fraction",
        "predicted_residual_energy_fraction",
    ]
    for (context, time_value), frame in raw.groupby(["context", "time"], sort=True):
        dot = float(frame["dot"].sum())
        reference_energy = float(frame["reference_energy"].sum())
        difference_energy = float(frame["difference_energy"].sum())
        beta = dot / max(reference_energy, np.finfo(np.float64).tiny)
        residual_energy = (
            difference_energy - 2.0 * beta * dot + beta * beta * reference_energy
        )
        row: dict[str, float | int | str] = {
            "context": str(context),
            "time": float(time_value),
            "samples": int(len(frame)),
            "global_projection_coefficient": beta,
            "global_cosine": dot
            / max(
                np.sqrt(reference_energy * difference_energy),
                np.finfo(np.float64).tiny,
            ),
            "global_residual_energy_fraction": residual_energy
            / max(difference_energy, np.finfo(np.float64).tiny),
        }
        for metric in descriptive:
            values = frame[metric].to_numpy(dtype=np.float64)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_q10"] = float(np.quantile(values, 0.1))
            row[f"{metric}_q90"] = float(np.quantile(values, 0.9))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["context", "time"]).reset_index(drop=True)


def plot_summary(summary: pd.DataFrame, output: Path, predicted: float) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharex=True)
    colors = {"teacher": "#2864a5", "strong_rollout": "#c44e38"}
    for context, frame in summary.groupby("context", sort=False):
        color = colors.get(str(context))
        axes[0].plot(frame.time, frame.global_cosine, "o-", color=color, label=context)
        axes[1].plot(
            frame.time,
            frame.global_projection_coefficient,
            "o-",
            color=color,
            label=context,
        )
        axes[2].plot(
            frame.time,
            frame.global_residual_energy_fraction,
            "o-",
            color=color,
            label=context,
        )
    axes[1].axhline(predicted, color="#333333", linestyle="--", label="FID-predicted")
    axes[0].set(title="Global direction cosine", ylabel="cosine")
    axes[1].set(title="Weak/full regression", ylabel="beta")
    axes[2].set(title="Residual after best global projection", ylabel="energy fraction")
    for axis in axes:
        axis.set_xlabel("flow time t")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strong-checkpoint", type=Path, default=DEFAULT_STRONG)
    parser.add_argument("--depth4-head", type=Path, default=DEFAULT_DEPTH4)
    parser.add_argument("--depth8-head", type=Path, default=DEFAULT_DEPTH8)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--official-sit-repo", type=Path, default=DEFAULT_OFFICIAL_SIT_REPO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--times", nargs="+", type=float, default=list(DEFAULT_TIMES))
    parser.add_argument("--full-best-gamma", type=float, default=0.25)
    parser.add_argument("--difference-best-gamma", type=float, default=0.65)
    parser.add_argument("--device", default="cuda:0")
    return parser


@torch.inference_mode()
def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.samples <= 0 or args.batch_size <= 0 or args.samples % args.batch_size:
        raise ValueError("samples must be positive and divisible by batch size")
    times = tuple(sorted(float(value) for value in args.times))
    if len(times) != len(set(times)) or any(value <= 0.0 or value >= 1.0 for value in times):
        raise ValueError("times must be unique and strictly inside (0, 1)")
    if args.difference_best_gamma == 0.0:
        raise ValueError("difference-best-gamma must be nonzero")
    predicted = float(args.full_best_gamma / args.difference_best_gamma)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    strong_path = args.strong_checkpoint.expanduser().resolve()
    sit_module, source_metadata = load_official_sit_module(
        args.official_sit_repo.expanduser().resolve(), verify_source=True
    )
    strong, semantics, strong_metadata = load_sit_field_model(
        checkpoint_path=strong_path,
        weights="ema",
        sit_module=sit_module,
        source_metadata=source_metadata,
        device=device,
    )
    heads = {
        name: load_internal_head_for_source(
            checkpoint_path=path.expanduser().resolve(),
            name=name,
            head_weights="ema",
            model=strong,
            sit_module=sit_module,
            source_checkpoint_path=strong_path,
            source_metadata=source_metadata,
            device=device,
        )
        for name, path in {
            "depth4_v": args.depth4_head,
            "depth8_v": args.depth8_head,
        }.items()
    }

    dataset = NpyMomentsDataset(args.cache_dir.expanduser().resolve(), "validation")
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(dataset), size=args.samples, replace=False)
    moments_array = np.load(dataset.moments_path, mmap_mode="r", allow_pickle=False)
    labels_array = np.load(dataset.labels_path, mmap_mode="r", allow_pickle=False)
    moments = torch.from_numpy(np.asarray(moments_array[indices]).copy())
    labels_all = torch.from_numpy(np.asarray(labels_array[indices]).copy()).long()
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    posterior_noise = torch.randn(args.samples, *LATENT_SHAPE, generator=generator)
    path_noise = torch.randn(args.samples, *LATENT_SHAPE, generator=generator)
    clean = sample_sdvae_posterior(
        moments,
        posterior_noise,
        scaling_factor=SD_VAE_SCALING_FACTOR,
    )

    raw_rows: list[dict[str, float | int | str]] = []
    integration_times = torch.tensor((0.0, *times), device=device, dtype=torch.float32)
    for start in range(0, args.samples, args.batch_size):
        stop = start + args.batch_size
        labels = labels_all[start:stop].to(device)
        noise = path_noise[start:stop].to(device)
        target = clean[start:stop].to(device)

        def source_velocity(time_value: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
            time_batch = time_value.expand(len(state))
            full, _, _ = evaluate_source_with_heads(
                strong,
                state,
                time_batch,
                labels,
                heads={},
                source_semantics=semantics,
            )
            return full

        rollout = odeint(
            source_velocity,
            noise.float(),
            integration_times,
            method="dopri5",
            atol=1e-6,
            rtol=1e-3,
        )[1:]
        for time_index, time_value in enumerate(times):
            teacher = (1.0 - time_value) * noise + time_value * target
            for context, state in (
                ("teacher", teacher),
                ("strong_rollout", rollout[time_index]),
            ):
                time_batch = torch.full(
                    (len(state),), time_value, device=device, dtype=torch.float32
                )
                full, trained, _ = evaluate_source_with_heads(
                    strong,
                    state,
                    time_batch,
                    labels,
                    heads=heads,
                    source_semantics=semantics,
                )
                metrics = geometry_metrics(
                    full,
                    trained["depth8_v"],
                    trained["depth4_v"],
                    predicted_coefficient=predicted,
                )
                arrays = {key: value.cpu().numpy() for key, value in metrics.items()}
                for local_index in range(len(state)):
                    row: dict[str, float | int | str] = {
                        "context": context,
                        "time": time_value,
                        "sample_id": int(indices[start + local_index]),
                    }
                    row.update(
                        {key: float(value[local_index]) for key, value in arrays.items()}
                    )
                    raw_rows.append(row)
        print(json.dumps({"processed": stop, "samples": args.samples}), flush=True)

    raw = pd.DataFrame(raw_rows)
    summary = summarize(raw)
    raw_path = output_dir / "depth_difference_geometry_per_sample.csv"
    summary_path = output_dir / "depth_difference_geometry_by_time.csv"
    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    figure_path = output_dir / "depth_difference_geometry.png"
    plot_summary(summary, figure_path, predicted)

    overall = summarize(raw.assign(time=-1.0)).drop(columns="time").to_dict(orient="records")
    payload = {
        "format": "eqvae_imagenet100_sit_depth_difference_geometry_v1",
        "definitions": {
            "full_gap": "strong - depth4_v",
            "weak_difference": "depth8_v - depth4_v",
            "regression": "weak_difference = beta * full_gap + residual",
            "projection_grain": "one scalar per sample over C,H,W",
        },
        "samples": args.samples,
        "seed": args.seed,
        "times": list(times),
        "full_best_gamma": args.full_best_gamma,
        "difference_best_gamma": args.difference_best_gamma,
        "fid_predicted_coefficient": predicted,
        "strong": strong_metadata,
        "heads": {
            name: {
                "depth": spec.depth,
                "checkpoint": spec.checkpoint,
                "checkpoint_sha256": spec.checkpoint_sha256,
            }
            for name, spec in heads.items()
        },
        "overall": overall,
        "raw_csv": str(raw_path),
        "summary_csv": str(summary_path),
        "figure": str(figure_path),
    }
    atomic_json_dump(payload, output_dir / "depth_difference_geometry_summary.json")
    print(json.dumps({"predicted": predicted, "overall": overall}, indent=2), flush=True)


if __name__ == "__main__":
    main(build_parser().parse_args())
