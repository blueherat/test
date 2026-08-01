"""Audit the official Internal-Guidance SiT dual-head direction.

This experiment does not train or modify the checkpoint.  It measures:

1. local supervised alignment of ``full - base`` with the remaining velocity
   error on held-out ImageNet validation latents;
2. paired fixed-scale error sweeps at the same states;
3. teacher-path recovery rollouts with either persistent guidance or a
   first-step-only impulse.

The rollout target is an individual VAE latent retained in a partially noised
state.  It is a mechanism probe, not an unconditional-generation metric and
must not be reported as FID or sample quality.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from experiments.internal_guidance_direction import (
    direction_metrics,
    euler_ig_scale_sweep_rollout,
    scale_sweep_metrics,
    split_dual_output,
)
from experiments.raev2_training_core import DeterministicImageNetParquet


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IG_REPO = ROOT / "research_repos" / "internal_guidance_study" / "Internal-Guidance" / "SiT"
DEFAULT_OUTPUT = Path.home() / "data" / "eqvae" / "internal_guidance_sit_audit"


def parse_float_list(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated floats")
    if any(not np.isfinite(item) for item in values):
        raise argparse.ArgumentTypeError("all values must be finite")
    return values


def configure_fp32(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def validate_protocol(
    *,
    times: tuple[float, ...],
    rollout_times: tuple[float, ...],
    scales: tuple[float, ...],
    samples: int,
    batch_size: int,
    rollout_samples: int,
    rollout_batch_size: int,
    rollout_steps: int,
) -> None:
    if samples <= 0 or batch_size <= 0:
        raise ValueError("samples and batch_size must be positive")
    if rollout_samples < 0 or rollout_samples > samples:
        raise ValueError("rollout_samples must lie between zero and samples")
    if rollout_batch_size <= 0:
        raise ValueError("rollout_batch_size must be positive")
    if rollout_samples and rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive when rollouts are enabled")
    if any(not 0.0 < value < 1.0 for value in times):
        raise ValueError("audit times must lie strictly inside (0, 1)")
    if any(value not in times for value in rollout_times):
        raise ValueError("rollout_times must be a subset of audit times")
    if any(value < 0 for value in scales):
        raise ValueError("IG scales must be non-negative")
    if 1.0 not in scales:
        raise ValueError("scales must include 1.0 as the unmodified full baseline")
    if len(set(times)) != len(times) or len(set(scales)) != len(scales):
        raise ValueError("times and scales must not contain duplicates")


def import_sit_models(repo: Path) -> dict[str, Any]:
    repo = repo.expanduser().resolve()
    model_file = repo / "models" / "sit.py"
    if not model_file.is_file():
        raise FileNotFoundError(f"official SiT model file not found: {model_file}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from models.sit import SiT_models

    return SiT_models


def checkpoint_state_dict(
    checkpoint: object,
    *,
    state_key: str,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    metadata: dict[str, object] = {"state_key": state_key}
    if isinstance(checkpoint, dict) and state_key in checkpoint:
        state = checkpoint[state_key]
        for key in ("epoch", "steps", "step"):
            if key in checkpoint and isinstance(checkpoint[key], (int, float)):
                metadata[key] = int(checkpoint[key])
    else:
        state = checkpoint
        metadata["state_key"] = "root"
    if not isinstance(state, dict) or not state:
        raise TypeError("checkpoint does not contain a non-empty state dictionary")
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()):
        raise TypeError("checkpoint state dictionary must map strings to tensors")
    if all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
        metadata["stripped_module_prefix"] = True
    else:
        metadata["stripped_module_prefix"] = False
    return state, metadata


def load_model(
    *,
    repo: Path,
    checkpoint_path: Path,
    model_name: str,
    encoder_depth: int,
    state_key: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, object]]:
    models = import_sit_models(repo)
    if model_name not in models:
        raise ValueError(f"unknown SiT model {model_name!r}; available: {sorted(models)}")
    model = models[model_name](
        input_size=32,
        num_classes=1000,
        use_cfg=True,
        encoder_depth=int(encoder_depth),
        fused_attn=False,
        qk_norm=False,
    )
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
    except RuntimeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    state, metadata = checkpoint_state_dict(checkpoint, state_key=state_key)
    incompatibility = model.load_state_dict(state, strict=True)
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise RuntimeError(f"strict checkpoint load failed: {incompatibility}")
    del checkpoint, state
    model.requires_grad_(False).eval().to(device=device, dtype=torch.float32)
    metadata.update(
        {
            "checkpoint": str(checkpoint_path),
            "checkpoint_bytes": checkpoint_path.stat().st_size,
            "model_name": model_name,
            "encoder_depth": int(encoder_depth),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        }
    )
    return model, metadata


@torch.no_grad()
def encode_validation_latents(
    *,
    dataset_root: Path,
    split: str,
    samples: int,
    vae_batch_size: int,
    seed: int,
    device: torch.device,
    vae_name: str,
    posterior_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    from diffusers.models import AutoencoderKL

    dataset = DeterministicImageNetParquet(
        dataset_root,
        split=split,
        image_size=256,
        horizontal_flip=False,
    )
    if samples > len(dataset):
        raise ValueError(f"requested {samples} samples from a dataset of size {len(dataset)}")
    index_generator = torch.Generator(device="cpu").manual_seed(int(seed))
    indices = torch.randperm(len(dataset), generator=index_generator)[:samples].tolist()

    vae = AutoencoderKL.from_pretrained(vae_name, local_files_only=True)
    vae.requires_grad_(False).eval().to(device=device, dtype=torch.float32)
    posterior_generator = torch.Generator(device="cpu").manual_seed(int(seed) + 1)
    latent_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    for start in range(0, samples, vae_batch_size):
        batch_indices = indices[start : start + vae_batch_size]
        items = [dataset[index] for index in batch_indices]
        images = torch.stack([item[0] for item in items]).to(device)
        labels = torch.tensor([item[1] for item in items], dtype=torch.long)
        distribution = vae.encode(images.mul(2.0).sub(1.0)).latent_dist
        mean = distribution.mean.float().cpu()
        if posterior_mode == "sample":
            std = distribution.std.float().cpu()
            posterior_noise = torch.randn(
                mean.shape, generator=posterior_generator, dtype=torch.float32
            )
            latent = mean + std * posterior_noise
        elif posterior_mode == "mean":
            latent = mean
        else:
            raise ValueError(f"unsupported posterior mode: {posterior_mode}")
        latent_chunks.append(latent.mul(0.18215))
        label_chunks.append(labels)
    del vae
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(latent_chunks), torch.cat(label_chunks), [int(value) for value in indices]


def _append_local_rows(
    rows: list[dict[str, object]],
    metrics: dict[str, torch.Tensor],
    *,
    indices: list[int],
    labels: torch.Tensor,
    time: float,
) -> None:
    for batch_index, dataset_index in enumerate(indices):
        row: dict[str, object] = {
            "dataset_index": int(dataset_index),
            "label": int(labels[batch_index]),
            "time": float(time),
        }
        for name, values in metrics.items():
            value = values[batch_index].detach().cpu()
            row[name] = bool(value) if value.dtype == torch.bool else float(value)
        rows.append(row)


@torch.no_grad()
def run_local_audit(
    model: torch.nn.Module,
    clean_all: torch.Tensor,
    noise_all: torch.Tensor,
    labels_all: torch.Tensor,
    indices: list[int],
    *,
    times: tuple[float, ...],
    scales: tuple[float, ...],
    batch_size: int,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    local_rows: list[dict[str, object]] = []
    sweep_rows: list[dict[str, object]] = []
    for start in range(0, len(clean_all), batch_size):
        stop = min(start + batch_size, len(clean_all))
        clean = clean_all[start:stop].to(device)
        noise = noise_all[start:stop].to(device)
        labels = labels_all[start:stop].to(device)
        batch_indices = indices[start:stop]
        for time_value in times:
            time = torch.full(
                (len(clean),), float(time_value), device=device, dtype=torch.float32
            )
            state = (1.0 - float(time_value)) * clean + float(time_value) * noise
            target_velocity = noise - clean
            full, base = split_dual_output(model(state, time, labels))
            metrics = direction_metrics(full, base, target_velocity)
            _append_local_rows(
                local_rows,
                metrics,
                indices=batch_indices,
                labels=labels,
                time=time_value,
            )
            sweep = scale_sweep_metrics(full, base, target_velocity, scales)
            for scale_index, scale in enumerate(scales):
                for batch_index, dataset_index in enumerate(batch_indices):
                    sweep_rows.append(
                        {
                            "dataset_index": int(dataset_index),
                            "label": int(labels[batch_index]),
                            "time": float(time_value),
                            "scale": float(scale),
                            "mse": float(sweep["mse"][scale_index, batch_index].cpu()),
                            "gain_over_full": float(
                                sweep["gain_over_full"][scale_index, batch_index].cpu()
                            ),
                        }
                    )
    return pd.DataFrame(local_rows), pd.DataFrame(sweep_rows)


@torch.no_grad()
def run_rollout_audit(
    model: torch.nn.Module,
    clean_all: torch.Tensor,
    noise_all: torch.Tensor,
    labels_all: torch.Tensor,
    indices: list[int],
    *,
    rollout_samples: int,
    rollout_batch_size: int,
    rollout_times: tuple[float, ...],
    rollout_steps: int,
    scales: tuple[float, ...],
    device: torch.device,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if rollout_samples == 0:
        return pd.DataFrame(rows)
    baseline_scale_index = scales.index(1.0)
    for start in range(0, rollout_samples, rollout_batch_size):
        stop = min(start + rollout_batch_size, rollout_samples)
        clean = clean_all[start:stop].to(device)
        noise = noise_all[start:stop].to(device)
        labels = labels_all[start:stop].to(device)
        batch_indices = indices[start:stop]
        for start_time in rollout_times:
            initial = (1.0 - float(start_time)) * clean + float(start_time) * noise
            grid = torch.linspace(
                float(start_time), 0.0, rollout_steps + 1, device=device, dtype=torch.float32
            )
            for mode in ("persistent", "first_step_impulse"):
                endpoints = euler_ig_scale_sweep_rollout(
                    model,
                    initial,
                    labels,
                    grid,
                    scales,
                    mode=mode,
                )
                endpoint_mse = (endpoints - clean.unsqueeze(0)).square().flatten(2).mean(2)
                baseline_mse = endpoint_mse[baseline_scale_index]
                baseline_endpoint = endpoints[baseline_scale_index]
                endpoint_delta = (
                    (endpoints - baseline_endpoint.unsqueeze(0)).square().flatten(2).mean(2).sqrt()
                )
                for scale_index, scale in enumerate(scales):
                    for batch_index, dataset_index in enumerate(batch_indices):
                        rows.append(
                            {
                                "dataset_index": int(dataset_index),
                                "label": int(labels[batch_index]),
                                "start_time": float(start_time),
                                "rollout_steps": int(rollout_steps),
                                "mode": mode,
                                "scale": float(scale),
                                "endpoint_mse": float(endpoint_mse[scale_index, batch_index].cpu()),
                                "gain_over_full": float(
                                    (
                                        1.0
                                        - endpoint_mse[scale_index, batch_index]
                                        / baseline_mse[batch_index].clamp_min(1e-12)
                                    ).cpu()
                                ),
                                "endpoint_delta_rms": float(
                                    endpoint_delta[scale_index, batch_index].cpu()
                                ),
                            }
                        )
    return pd.DataFrame(rows)


def summarize_results(
    local: pd.DataFrame,
    sweep: pd.DataFrame,
    rollout: pd.DataFrame,
) -> dict[str, object]:
    local_summary = (
        local.groupby("time", as_index=False)
        .agg(
            alignment_cosine_mean=("alignment_cosine", "mean"),
            alignment_cosine_median=("alignment_cosine", "median"),
            positive_alignment_fraction=("positive_alignment", "mean"),
            scale_star_median=("scale_star", "median"),
            oracle_relative_gain_mean=("oracle_relative_gain", "mean"),
            full_mse_mean=("full_mse", "mean"),
            base_mse_mean=("base_mse", "mean"),
        )
        .to_dict(orient="records")
    )
    sweep_summary = (
        sweep.groupby(["time", "scale"], as_index=False)
        .agg(gain_mean=("gain_over_full", "mean"), gain_median=("gain_over_full", "median"))
        .to_dict(orient="records")
    )

    rollout_summary: list[dict[str, object]] = []
    correlations: list[dict[str, object]] = []
    if not rollout.empty:
        rollout_summary = (
            rollout.groupby(["mode", "start_time", "scale"], as_index=False)
            .agg(
                endpoint_gain_mean=("gain_over_full", "mean"),
                endpoint_gain_median=("gain_over_full", "median"),
                endpoint_delta_rms_mean=("endpoint_delta_rms", "mean"),
            )
            .to_dict(orient="records")
        )
        paired = rollout.merge(
            sweep[["dataset_index", "time", "scale", "gain_over_full"]].rename(
                columns={"time": "start_time", "gain_over_full": "local_gain"}
            ),
            on=["dataset_index", "start_time", "scale"],
            how="inner",
        )
        for (mode, scale), group in paired.groupby(["mode", "scale"]):
            if (
                float(scale) == 1.0
                or len(group) < 3
                or group["local_gain"].nunique() < 2
                or group["gain_over_full"].nunique() < 2
            ):
                continue
            correlation = float(
                group["local_gain"].corr(group["gain_over_full"], method="spearman")
            )
            if np.isfinite(correlation):
                correlations.append(
                    {
                        "mode": str(mode),
                        "scale": float(scale),
                        "samples": int(len(group)),
                        "spearman_local_vs_endpoint_gain": correlation,
                    }
                )
    return {
        "local_summary": local_summary,
        "scale_sweep_summary": sweep_summary,
        "rollout_summary": rollout_summary,
        "local_endpoint_correlations": correlations,
    }


def plot_results(
    local: pd.DataFrame,
    sweep: pd.DataFrame,
    rollout: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    local_group = local.groupby("time")
    times = np.array(sorted(local["time"].unique()))
    cosine_mean = local_group["alignment_cosine"].mean().reindex(times)
    cosine_low = local_group["alignment_cosine"].quantile(0.25).reindex(times)
    cosine_high = local_group["alignment_cosine"].quantile(0.75).reindex(times)
    axes[0, 0].plot(times, cosine_mean, marker="o", color="#2563EB", label="mean cosine")
    axes[0, 0].fill_between(times, cosine_low, cosine_high, color="#93C5FD", alpha=0.45, label="IQR")
    axes[0, 0].axhline(0.0, color="#111827", linewidth=1)
    axes[0, 0].set(title="Local residual alignment", xlabel="flow time t", ylabel="cos(full-base, target-full)")
    axes[0, 0].legend()

    positive = local_group["positive_alignment"].mean().reindex(times)
    axes[0, 1].plot(times, positive, marker="o", color="#0F766E")
    axes[0, 1].axhline(0.5, color="#111827", linewidth=1, linestyle="--")
    axes[0, 1].set(title="Fraction with a useful positive direction", xlabel="flow time t", ylabel="fraction")
    axes[0, 1].set_ylim(0.0, 1.0)

    local_gain = sweep.groupby(["time", "scale"])["gain_over_full"].mean().unstack("scale")
    gain_values = local_gain.to_numpy()
    gain_bound = max(float(np.abs(gain_values).max()), 1e-12)
    image = axes[1, 0].imshow(
        gain_values,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-gain_bound,
        vmax=gain_bound,
    )
    axes[1, 0].set_xticks(range(len(local_gain.columns)), [f"{value:g}" for value in local_gain.columns])
    axes[1, 0].set_yticks(range(len(local_gain.index)), [f"{value:g}" for value in local_gain.index])
    axes[1, 0].set(title="Local MSE gain over full", xlabel="IG scale", ylabel="flow time t")
    figure.colorbar(image, ax=axes[1, 0], label="relative gain")

    if rollout.empty:
        axes[1, 1].axis("off")
        axes[1, 1].text(0.5, 0.5, "Rollout disabled", ha="center", va="center")
    else:
        rollout_group = rollout.groupby(["mode", "start_time", "scale"])["gain_over_full"].mean()
        colors = {"persistent": "#DC2626", "first_step_impulse": "#7C3AED"}
        for (mode, start_time), values in rollout_group.groupby(level=[0, 1]):
            series = values.droplevel([0, 1])
            axes[1, 1].plot(
                series.index,
                series.values,
                marker="o",
                color=colors[str(mode)],
                linestyle="-" if float(start_time) == min(rollout["start_time"]) else "--",
                label=f"{mode}, t={float(start_time):g}",
            )
        axes[1, 1].axhline(0.0, color="#111827", linewidth=1)
        axes[1, 1].set(title="Teacher-path endpoint gain", xlabel="IG scale", ylabel="relative gain over full")
        axes[1, 1].legend(fontsize=8)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ig-repo", type=Path, default=DEFAULT_IG_REPO)
    parser.add_argument("--model", default="SiT-XL/2")
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--state-key", default="ema")
    parser.add_argument("--dataset", type=Path, default=Path("/data/shared/imagenet-1k"))
    parser.add_argument("--split", default="validation")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--vae-batch-size", type=int, default=4)
    parser.add_argument("--rollout-samples", type=int, default=8)
    parser.add_argument("--rollout-batch-size", type=int, default=1)
    parser.add_argument("--rollout-steps", type=int, default=20)
    parser.add_argument("--times", type=parse_float_list, default=(0.2, 0.5, 0.8))
    parser.add_argument("--rollout-times", type=parse_float_list, default=(0.5, 0.8))
    parser.add_argument("--scales", type=parse_float_list, default=(0.0, 0.5, 1.0, 1.2, 1.4, 1.8))
    parser.add_argument("--vae", default="stabilityai/sd-vae-ft-ema")
    parser.add_argument("--posterior-mode", choices=("sample", "mean"), default="sample")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_protocol(
        times=args.times,
        rollout_times=args.rollout_times,
        scales=args.scales,
        samples=args.samples,
        batch_size=args.batch_size,
        rollout_samples=args.rollout_samples,
        rollout_batch_size=args.rollout_batch_size,
        rollout_steps=args.rollout_steps,
    )
    configure_fp32(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    clean, labels, indices = encode_validation_latents(
        dataset_root=args.dataset,
        split=args.split,
        samples=args.samples,
        vae_batch_size=args.vae_batch_size,
        seed=args.seed,
        device=device,
        vae_name=args.vae,
        posterior_mode=args.posterior_mode,
    )
    noise_generator = torch.Generator(device="cpu").manual_seed(int(args.seed) + 2)
    noise = torch.randn(clean.shape, generator=noise_generator, dtype=torch.float32)
    model, checkpoint_metadata = load_model(
        repo=args.ig_repo,
        checkpoint_path=args.checkpoint,
        model_name=args.model,
        encoder_depth=args.encoder_depth,
        state_key=args.state_key,
        device=device,
    )

    local, sweep = run_local_audit(
        model,
        clean,
        noise,
        labels,
        indices,
        times=args.times,
        scales=args.scales,
        batch_size=args.batch_size,
        device=device,
    )
    rollout = run_rollout_audit(
        model,
        clean,
        noise,
        labels,
        indices,
        rollout_samples=args.rollout_samples,
        rollout_batch_size=args.rollout_batch_size,
        rollout_times=args.rollout_times,
        rollout_steps=args.rollout_steps,
        scales=args.scales,
        device=device,
    )
    summary = summarize_results(local, sweep, rollout)
    metadata = {
        "experiment": "official_internal_guidance_sit_direction_audit_v1",
        "scope": "held-out local velocity and teacher-path recovery; not FID",
        "training": False,
        "split": args.split,
        "dataset": str(args.dataset.expanduser().resolve()),
        "dataset_indices": indices,
        "samples": int(args.samples),
        "rollout_samples": int(args.rollout_samples),
        "times": list(args.times),
        "rollout_times": list(args.rollout_times),
        "rollout_steps": int(args.rollout_steps),
        "scales": list(args.scales),
        "seed": int(args.seed),
        "precision": "fp32",
        "tf32": False,
        "vae": args.vae,
        "posterior_mode": args.posterior_mode,
        "checkpoint": checkpoint_metadata,
        "summary": summary,
        "caveat": "Individual endpoint MSE is meaningful only as partially-noised teacher-path recovery, not unconditional sample quality.",
    }
    local.to_csv(output_dir / "local_direction.csv", index=False)
    sweep.to_csv(output_dir / "local_scale_sweep.csv", index=False)
    rollout.to_csv(output_dir / "teacher_path_rollout.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    plot_results(local, sweep, rollout, output_dir / "mechanism_atlas.png")
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    print(f"saved audit to {output_dir}")


if __name__ == "__main__":
    main()
