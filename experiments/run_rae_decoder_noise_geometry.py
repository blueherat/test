"""Run the no-training RAE decoder noise-coordinate geometry experiment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.visual_adapters import load_rae_adapter  # noqa: E402
from experiments.rae_decoder_noise_geometry import (  # noqa: E402
    hidden_deviation_profile,
    matched_noise_geometry,
    relative_cycle_error,
    sample_rms,
)
from experiments.rae_decoder_risk_phase0 import (  # noqa: E402
    _latent_to_decoder_tokens,
    decoder_hidden_rms,
)
from experiments.rae_latent_cache import (  # noqa: E402
    CachedRAELatentDataset,
    load_cache_manifest,
    split_range,
)


DEFAULT_CACHE = (
    Path.home()
    / "data/eqvae/cache/rae_decoder_risk_phase0/seed20260718_cal1024_test2048_fp32"
)
DEFAULT_OUTPUT = Path.home() / "data/eqvae/experiments/rae_decoder_noise_geometry"


@dataclass(frozen=True)
class NoiseGeometryConfig:
    cache: Path = DEFAULT_CACHE
    calibration_count: int = 256
    test_count: int = 512
    batch_size: int = 2
    tau: float = 0.8
    seed: int = 20260718
    model_key: str = "rae_dinov2"
    rae_repo_path: str = "external/RAE"
    output_root: Path = DEFAULT_OUTPUT
    run_name: str = "dinov2_cal256_test512_tau08_seed20260718"


def _distributed(seed: int) -> tuple[int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full decoder geometry audit")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if world_size > 1:
        dist.init_process_group("nccl", device_id=device)
        rank = dist.get_rank()
    else:
        rank = 0
    torch.manual_seed(int(seed) * world_size + rank)
    np.random.seed(int(seed) * world_size + rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return rank, world_size, device


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _all_reduce(tensor: torch.Tensor) -> None:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)


@torch.no_grad()
def _decode(
    rae: torch.nn.Module,
    latent: torch.Tensor,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    output = rae.decoder(
        _latent_to_decoder_tokens(rae, latent),
        drop_cls_token=False,
        output_hidden_states=True,
    )
    if output.hidden_states is None:
        raise RuntimeError("RAE decoder did not expose hidden states")
    image = rae.decoder.unpatchify(output.logits)
    image = image * rae.encoder_std.to(image) + rae.encoder_mean.to(image)
    hidden = tuple(state[:, 1:].float() for state in output.hidden_states)
    return image.float(), hidden


def _sample_noise(
    shape: tuple[int, ...],
    indices: list[int],
    seed: int,
    tau: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    noises = []
    severities = []
    for index in indices:
        generator = torch.Generator(device="cpu").manual_seed(
            int(seed) + int(index) * 1_000_003
        )
        noises.append(torch.randn(shape, generator=generator, dtype=torch.float32))
        severities.append(float(tau) * float(torch.rand((), generator=generator)))
    return torch.stack(noises), torch.tensor(severities, dtype=torch.float32)


@torch.no_grad()
def _hidden_reference(
    rae: torch.nn.Module,
    dataset: CachedRAELatentDataset,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    total = torch.zeros(29, dtype=torch.float64, device=device)
    square = torch.zeros_like(total)
    count = torch.zeros((), dtype=torch.float64, device=device)
    for latent, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        _, hidden = _decode(rae, latent.to(device))
        values = decoder_hidden_rms(hidden).double()
        total.add_(values.sum(dim=0))
        square.add_(values.square().sum(dim=0))
        count.add_(len(values))
    for value in (total, square, count):
        _all_reduce(value)
    mean = total / count.clamp_min(1)
    variance = square / count.clamp_min(1) - mean.square()
    return mean.float(), variance.clamp_min(0).sqrt().clamp_min(1e-6).float()


def _condition_payloads(
    latent: torch.Tensor,
    latent_std: torch.Tensor,
    noise: torch.Tensor,
    severity: torch.Tensor,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    perturbations = matched_noise_geometry(noise, latent_std, severity)
    return {
        "raw_sphere": (
            latent + perturbations.raw_sphere_normalized,
            perturbations.raw_sphere_raw,
        ),
        "stage2_sphere_raw_rms_matched": (
            latent + perturbations.stage2_sphere_raw_matched_normalized,
            perturbations.stage2_sphere_raw_matched_raw,
        ),
        "stage2_sphere_u_rms_matched": (
            latent + perturbations.stage2_sphere_normalized_matched,
            perturbations.stage2_sphere_normalized_matched_raw,
        ),
    }


@torch.no_grad()
def _evaluate(
    rae: torch.nn.Module,
    dataset: CachedRAELatentDataset,
    *,
    local_start: int,
    batch_size: int,
    latent_std: torch.Tensor,
    hidden_mean: torch.Tensor,
    hidden_std: torch.Tensor,
    config: NoiseGeometryConfig,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_rows: list[dict[str, object]] = []
    layer_rows: list[dict[str, object]] = []
    consumed = 0
    for latent_cpu, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        batch = len(latent_cpu)
        indices = [local_start + consumed + offset for offset in range(batch)]
        consumed += batch
        noise_cpu, severity_cpu = _sample_noise(
            tuple(latent_cpu.shape[1:]), indices, config.seed + 91, config.tau
        )
        latent = latent_cpu.to(device)
        noise = noise_cpu.to(device)
        severity = severity_cpu.to(device)
        clean_image, clean_hidden = _decode(rae, latent)
        clean_cycle = rae.encode(clean_image.clamp(0.0, 1.0))
        clean_cycle_error = relative_cycle_error(clean_cycle, latent)
        payloads = _condition_payloads(latent, latent_std, noise, severity)

        for condition, (candidate, raw_delta) in payloads.items():
            image, hidden = _decode(rae, candidate)
            cycle = rae.encode(image.clamp(0.0, 1.0))
            hidden_rms = decoder_hidden_rms(hidden)
            hidden_z = (hidden_rms - hidden_mean) / hidden_std
            deviation, gain = hidden_deviation_profile(hidden, clean_hidden)
            normalized_delta = candidate - latent
            raw_delta_rms = sample_rms(raw_delta)
            normalized_delta_rms = sample_rms(normalized_delta)
            image_delta = sample_rms(image - clean_image)
            cycle_error = relative_cycle_error(cycle, candidate)
            clipping = ((image < 0.0) | (image > 1.0)).float().flatten(1).mean(dim=1)
            sensitivity = deviation.mean(dim=1) / raw_delta_rms.clamp_min(1e-12)

            for offset, sample_index in enumerate(indices):
                sample_rows.append(
                    {
                        "condition": condition,
                        "sample_index": sample_index,
                        "severity_raw_rms": float(severity[offset]),
                        "raw_delta_rms": float(raw_delta_rms[offset]),
                        "normalized_delta_rms": float(normalized_delta_rms[offset]),
                        "image_delta_rms": float(image_delta[offset]),
                        "cycle_relative_rms": float(cycle_error[offset]),
                        "clean_cycle_relative_rms": float(clean_cycle_error[offset]),
                        "cycle_excess": float(cycle_error[offset] - clean_cycle_error[offset]),
                        "mean_hidden_sensitivity": float(sensitivity[offset]),
                        "peak_hidden_rms_z": float(hidden_z[offset].max()),
                        "decoded_pixel_clipping_fraction": float(clipping[offset]),
                    }
                )
                for layer in range(len(hidden)):
                    layer_rows.append(
                        {
                            "condition": condition,
                            "sample_index": sample_index,
                            "decoder_layer": layer,
                            "hidden_rms_z": float(hidden_z[offset, layer]),
                            "hidden_deviation_rms": float(deviation[offset, layer]),
                            "hidden_sensitivity": float(
                                deviation[offset, layer]
                                / raw_delta_rms[offset].clamp_min(1e-12)
                            ),
                            "deviation_gain": (
                                float(gain[offset, layer - 1]) if layer > 0 else float("nan")
                            ),
                        }
                    )
    return pd.DataFrame(sample_rows), pd.DataFrame(layer_rows)


def _bootstrap_effect(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    seed: int,
    mode: str,
    draws: int = 4000,
) -> dict[str, float | str]:
    if mode == "ratio":
        effect = numerator / np.clip(denominator, 1e-12, None)
    elif mode == "difference":
        effect = numerator - denominator
    else:
        raise ValueError(f"unknown paired effect mode {mode}")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(effect), size=(draws, len(effect)))
    means = effect[indices].mean(axis=1)
    return {
        "effect_type": mode,
        "mean_effect": float(effect.mean()),
        "median_effect": float(np.median(effect)),
        "mean_effect_ci_low": float(np.quantile(means, 0.025)),
        "mean_effect_ci_high": float(np.quantile(means, 0.975)),
        "fraction_stage2_larger": float(np.mean(numerator > denominator)),
    }


def _summarize(
    samples: pd.DataFrame,
    layers: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scalar_columns = [
        "severity_raw_rms",
        "raw_delta_rms",
        "normalized_delta_rms",
        "image_delta_rms",
        "cycle_relative_rms",
        "cycle_excess",
        "mean_hidden_sensitivity",
        "peak_hidden_rms_z",
        "decoded_pixel_clipping_fraction",
    ]
    scalar = samples.groupby("condition", as_index=False)[scalar_columns].agg(
        ["mean", "median", "std"]
    )
    scalar.columns = [
        "condition" if column[0] == "condition" else f"{column[0]}_{column[1]}"
        for column in scalar.columns
    ]
    layer = layers.groupby(["condition", "decoder_layer"], as_index=False).agg(
        hidden_rms_z_median=("hidden_rms_z", "median"),
        hidden_rms_z_mean=("hidden_rms_z", "mean"),
        hidden_sensitivity_median=("hidden_sensitivity", "median"),
        hidden_sensitivity_mean=("hidden_sensitivity", "mean"),
        deviation_gain_median=("deviation_gain", "median"),
    )

    pivot = samples.pivot(index="sample_index", columns="condition")
    rows = []
    for condition in (
        "stage2_sphere_raw_rms_matched",
        "stage2_sphere_u_rms_matched",
    ):
        for metric in (
            "image_delta_rms",
            "mean_hidden_sensitivity",
            "cycle_relative_rms",
            "cycle_excess",
            "peak_hidden_rms_z",
        ):
            numerator = pivot[metric][condition].to_numpy(dtype=np.float64)
            denominator = pivot[metric]["raw_sphere"].to_numpy(dtype=np.float64)
            mode = "difference" if metric in {"cycle_excess", "peak_hidden_rms_z"} else "ratio"
            rows.append(
                {
                    "condition": condition,
                    "metric": metric,
                    **_bootstrap_effect(
                        numerator,
                        denominator,
                        seed=seed + len(rows) * 101,
                        mode=mode,
                    ),
                }
            )
    return scalar, layer, pd.DataFrame(rows)


def _plot(layer: pd.DataFrame, scalar: pd.DataFrame, path: Path) -> None:
    labels = {
        "raw_sphere": "raw sphere (official geometry)",
        "stage2_sphere_raw_rms_matched": "Stage-2 sphere, raw RMS matched",
        "stage2_sphere_u_rms_matched": "Stage-2 sphere, u RMS matched",
    }
    colors = {
        "raw_sphere": "#2878B5",
        "stage2_sphere_raw_rms_matched": "#D95F02",
        "stage2_sphere_u_rms_matched": "#5A9E45",
    }
    figure, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    for condition, values in layer.groupby("condition"):
        axes[0, 0].plot(
            values.decoder_layer,
            values.hidden_rms_z_median,
            marker="o",
            markersize=3,
            label=labels[condition],
            color=colors[condition],
        )
        axes[0, 1].plot(
            values.decoder_layer,
            values.hidden_sensitivity_median,
            marker="o",
            markersize=3,
            label=labels[condition],
            color=colors[condition],
        )
        axes[1, 0].plot(
            values.decoder_layer,
            values.deviation_gain_median,
            marker="o",
            markersize=3,
            label=labels[condition],
            color=colors[condition],
        )
    axes[0, 0].set_title("Absolute hidden RMS response (clean z-score)")
    axes[0, 1].set_title("Hidden deviation sensitivity per raw-latent RMS")
    axes[1, 0].set_title("Consecutive hidden-deviation gain")
    for axis in axes.flat[:3]:
        axis.set_xlabel("Decoder hidden state")
        axis.grid(alpha=0.25)
    axes[0, 0].set_ylabel("Median z-score")
    axes[0, 1].set_ylabel("Median sensitivity")
    axes[1, 0].set_ylabel("Median gain")
    axes[0, 0].legend(loc="best", frameon=False)

    positions = np.arange(len(scalar))
    axes[1, 1].bar(
        positions - 0.18,
        scalar.image_delta_rms_median,
        width=0.36,
        label="image delta RMS",
        color="#4C78A8",
    )
    axes[1, 1].bar(
        positions + 0.18,
        scalar.cycle_relative_rms_median,
        width=0.36,
        label="cycle relative RMS",
        color="#F58518",
    )
    axes[1, 1].set_xticks(positions, [labels[value] for value in scalar.condition], rotation=12)
    axes[1, 1].set_title("Decoder output and cycle response")
    axes[1, 1].grid(axis="y", alpha=0.25)
    axes[1, 1].legend(frameon=False)
    figure.suptitle("RAE decoder noise-coordinate geometry audit", fontsize=18)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(config: NoiseGeometryConfig) -> Path | None:
    rank, world_size, device = _distributed(config.seed)
    cache = config.cache.expanduser().resolve()
    manifest = load_cache_manifest(cache)
    available_calibration = int(manifest["calibration_count"])
    available_test = int(manifest["test_count"])
    if not 0 < config.calibration_count <= available_calibration:
        raise ValueError("invalid calibration_count")
    if not 0 < config.test_count <= available_test:
        raise ValueError("invalid test_count")

    adapter = load_rae_adapter(
        config.model_key,
        repo_path=ROOT / config.rae_repo_path,
        device=device,
        dtype=torch.float32,
    )
    rae = adapter.model.eval()
    rae.noise_tau = 0.0
    latent_var = getattr(rae, "latent_var", None)
    if latent_var is None:
        raise RuntimeError("noise geometry audit requires RAE normalization statistics")
    latent_std = torch.sqrt(
        latent_var.to(device).float() + float(getattr(rae, "eps", 1e-5))
    )

    cal_start, cal_stop = split_range(config.calibration_count, rank, world_size)
    calibration = CachedRAELatentDataset(cache, start=cal_start, stop=cal_stop)
    hidden_mean, hidden_std = _hidden_reference(
        rae, calibration, config.batch_size, device
    )

    test_start, test_stop = split_range(config.test_count, rank, world_size)
    test = CachedRAELatentDataset(
        cache,
        start=available_calibration + test_start,
        stop=available_calibration + test_stop,
    )
    samples, layers = _evaluate(
        rae,
        test,
        local_start=test_start,
        batch_size=config.batch_size,
        latent_std=latent_std,
        hidden_mean=hidden_mean,
        hidden_std=hidden_std,
        config=config,
        device=device,
    )

    output = config.output_root.expanduser().resolve() / config.run_name
    output.mkdir(parents=True, exist_ok=True)
    samples.to_csv(output / f"sample_metrics_rank{rank:02d}.csv", index=False)
    layers.to_csv(output / f"layer_metrics_rank{rank:02d}.csv", index=False)
    _barrier()
    if rank != 0:
        if dist.is_initialized():
            dist.destroy_process_group()
        return None

    sample_table = pd.concat(
        [pd.read_csv(output / f"sample_metrics_rank{r:02d}.csv") for r in range(world_size)],
        ignore_index=True,
    )
    layer_table = pd.concat(
        [pd.read_csv(output / f"layer_metrics_rank{r:02d}.csv") for r in range(world_size)],
        ignore_index=True,
    )
    sample_table.to_csv(output / "sample_metrics.csv", index=False)
    layer_table.to_csv(output / "layer_metrics.csv", index=False)
    scalar, layer, paired = _summarize(sample_table, layer_table, config.seed)
    scalar.to_csv(output / "scalar_summary.csv", index=False)
    layer.to_csv(output / "layer_summary.csv", index=False)
    paired.to_csv(output / "paired_effects.csv", index=False)
    _plot(layer, scalar, output / "noise_geometry_audit.png")
    payload = {
        "config": {**asdict(config), "cache": str(config.cache), "output_root": str(config.output_root)},
        "world_size": world_size,
        "latent_std_quantiles": {
            str(q): float(torch.quantile(latent_std.cpu().flatten(), q))
            for q in (0.0, 0.01, 0.5, 0.99, 1.0)
        },
        "conditions": list(sample_table.condition.drop_duplicates()),
    }
    (output / "result.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(scalar.to_string(index=False))
    print("\nPaired effects relative to raw sphere:\n", paired.to_string(index=False))
    if dist.is_initialized():
        dist.destroy_process_group()
    return output


def parse_args() -> NoiseGeometryConfig:
    defaults = NoiseGeometryConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=defaults.cache)
    parser.add_argument("--calibration-count", type=int, default=defaults.calibration_count)
    parser.add_argument("--test-count", type=int, default=defaults.test_count)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--tau", type=float, default=defaults.tau)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--output-root", type=Path, default=defaults.output_root)
    parser.add_argument("--run-name", default=defaults.run_name)
    args = parser.parse_args()
    return NoiseGeometryConfig(
        cache=args.cache,
        calibration_count=args.calibration_count,
        test_count=args.test_count,
        batch_size=args.batch_size,
        tau=args.tau,
        seed=args.seed,
        output_root=args.output_root,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    run(parse_args())
