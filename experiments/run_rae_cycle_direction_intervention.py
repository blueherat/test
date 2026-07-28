"""Run the paired no-training RAE cycle-direction causal intervention."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
RAE_ROOT = ROOT / "external/RAE"
RAE_SRC = RAE_ROOT / "src"
for path in (ROOT, RAE_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.rae_cycle_direction_intervention import (  # noqa: E402
    CycleDirectionThresholds,
    GENERATED_SOURCES,
    INTERVENTION_CONDITIONS,
    cycle_direction_gate,
    interpolate_direction,
    matched_intervention_directions,
    sample_rms,
    select_global_alpha,
)
from experiments.rae_decoder_risk_phase0 import decoder_hidden_rms  # noqa: E402
from experiments.rae_latent_cache import (  # noqa: E402
    CachedRAELatentDataset,
    load_cache_manifest,
    split_range,
)
from experiments.rae_teacher_rollout_gap import (  # noqa: E402
    configure_fp32,
    inception_features,
    load_inception,
)
from experiments.train_rae_layerwise_path import resolve_stage1_paths  # noqa: E402
from utils.model_utils import instantiate_from_config  # noqa: E402


DEFAULT_PHASE0 = Path.home() / "data/eqvae/experiments/rae_decoder_risk_phase0"
DEFAULT_CACHE = (
    Path.home()
    / "data/eqvae/cache/rae_decoder_risk_phase0/seed20260718_cal1024_test2048_fp32"
)
DEFAULT_OUTPUT = Path.home() / "data/eqvae/experiments/rae_cycle_direction"
PRIMARY_HIDDEN_STATE = 2


@dataclass(frozen=True)
class CycleDirectionConfig:
    phase0_root: Path = DEFAULT_PHASE0
    audit_cache: Path = DEFAULT_CACHE
    output_root: Path = DEFAULT_OUTPUT
    run_name: str = "generated_cal128_test128_clean128_seed20260718"
    calibration_count: int = 128
    test_count: int = 128
    clean_reference_count: int = 256
    clean_guardrail_count: int = 128
    clean_guardrail_offset: int = 256
    batch_size: int = 2
    alphas: tuple[float, ...] = (0.025, 0.05, 0.1, 0.25)
    feature_cosine_floor: float = 0.98
    seed: int = 20_260_718


def _distributed(seed: int) -> tuple[int, int, torch.device]:
    if not torch.cuda.is_available() or "RANK" not in os.environ:
        raise RuntimeError("launch the cycle-direction study with torchrun")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    configure_fp32(int(seed) * world_size + rank)
    torch.use_deterministic_algorithms(True, warn_only=True)
    return rank, world_size, device


def _finish() -> None:
    dist.barrier()
    dist.destroy_process_group()


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _merge_rank_csv(output: Path, stem: str, world_size: int) -> pd.DataFrame:
    table = pd.concat(
        [pd.read_csv(output / f"{stem}_rank{rank:02d}.csv") for rank in range(world_size)],
        ignore_index=True,
    )
    table.to_csv(output / f"{stem}.csv", index=False)
    return table


def _load_full_rae(config: OmegaConf, device: torch.device) -> torch.nn.Module:
    stage_1 = OmegaConf.create(OmegaConf.to_container(config.stage_1, resolve=True))
    wrapper = OmegaConf.create({"stage_1": stage_1})
    resolve_stage1_paths(wrapper)
    return (
        instantiate_from_config(wrapper.stage_1)
        .to(device=device, dtype=torch.float32)
        .requires_grad_(False)
        .eval()
    )


@torch.no_grad()
def _decode(
    rae: torch.nn.Module,
    latent: torch.Tensor,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    z = latent
    if bool(getattr(rae, "do_normalization", False)):
        mean = rae.latent_mean.to(z) if rae.latent_mean is not None else 0.0
        var = rae.latent_var.to(z) if rae.latent_var is not None else 1.0
        z = z * torch.sqrt(var + float(rae.eps)) + mean
    batch, channels, height, width = z.shape
    tokens = z.reshape(batch, channels, height * width).transpose(1, 2)
    output = rae.decoder(tokens, drop_cls_token=False, output_hidden_states=True)
    if output.hidden_states is None or len(output.hidden_states) != 29:
        raise RuntimeError(
            f"expected 29 decoder hidden states, got "
            f"{None if output.hidden_states is None else len(output.hidden_states)}"
        )
    image = rae.decoder.unpatchify(output.logits)
    image = image * rae.encoder_std.to(image) + rae.encoder_mean.to(image)
    hidden = tuple(state[:, 1:].float() for state in output.hidden_states)
    return image.float(), hidden


@torch.no_grad()
def _hidden_reference(
    rae: torch.nn.Module,
    dataset: CachedRAELatentDataset,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    total = torch.zeros(29, dtype=torch.float64, device=device)
    square = torch.zeros_like(total)
    count = torch.zeros((), dtype=torch.float64, device=device)
    for latent_cpu, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        _, hidden = _decode(rae, latent_cpu.to(device))
        values = decoder_hidden_rms(hidden).double()
        total.add_(values.sum(dim=0))
        square.add_(values.square().sum(dim=0))
        count.add_(len(values))
    for value in (total, square, count):
        dist.all_reduce(value)
    mean = total / count.clamp_min(1)
    variance = square / count.clamp_min(1) - mean.square()
    return mean.float(), variance.clamp_min(0).sqrt().clamp_min(1e-6).float()


def _cache_slice(cache: Path, start: int, count: int) -> torch.Tensor:
    dataset = CachedRAELatentDataset(cache, start=int(start), stop=int(start) + int(count))
    return torch.stack([dataset[index][0] for index in range(len(dataset))])


def _endpoint_path(root: Path, source: str) -> Path:
    path = root / f"0b_generated_latents_{source}_n256_s50.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _check_paired_labels(labels: torch.Tensor, world_size: int, device: torch.device) -> None:
    local = labels.to(device=device, dtype=torch.long)
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    if any(not torch.equal(gathered[0], value) for value in gathered[1:]):
        raise RuntimeError("generated paths do not share identical labels")


def _layer_metrics(
    hidden: tuple[torch.Tensor, ...],
    base_hidden: tuple[torch.Tensor, ...],
    hidden_mean: torch.Tensor,
    hidden_std: torch.Tensor,
    delta_rms: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden_rms = decoder_hidden_rms(hidden)
    hidden_z = (hidden_rms - hidden_mean) / hidden_std
    deviation = torch.stack(
        [sample_rms(value - base) for value, base in zip(hidden, base_hidden)], dim=1
    )
    sensitivity = deviation / delta_rms[:, None].clamp_min(1e-12)
    return hidden_z, deviation, sensitivity


@torch.no_grad()
def _append_condition_rows(
    *,
    sample_rows: list[dict[str, object]],
    layer_rows: list[dict[str, object]],
    source: str,
    split: str,
    sample_indices: list[int],
    condition: str,
    alpha: float,
    base_latent: torch.Tensor,
    candidate: torch.Tensor,
    base_image: torch.Tensor,
    candidate_image: torch.Tensor,
    base_hidden: tuple[torch.Tensor, ...],
    candidate_hidden: tuple[torch.Tensor, ...],
    base_cycle_error: torch.Tensor,
    candidate_cycle_error: torch.Tensor,
    base_features: torch.Tensor,
    candidate_features: torch.Tensor,
    hidden_mean: torch.Tensor,
    hidden_std: torch.Tensor,
) -> None:
    delta = candidate - base_latent
    delta_rms = sample_rms(delta)
    latent_rms = sample_rms(base_latent, 1e-12)
    hidden_z, deviation, sensitivity = _layer_metrics(
        candidate_hidden,
        base_hidden,
        hidden_mean,
        hidden_std,
        delta_rms,
    )
    image_delta = sample_rms(candidate_image - base_image)
    clipped_delta = sample_rms(
        candidate_image.clamp(0, 1) - base_image.clamp(0, 1)
    )
    feature_cosine = F.cosine_similarity(base_features, candidate_features, dim=1)
    clipping = ((candidate_image < 0) | (candidate_image > 1)).float().flatten(1).mean(dim=1)
    cycle_ratio = candidate_cycle_error / base_cycle_error.clamp_min(1e-12)
    hidden_cpu = hidden_z.cpu()
    deviation_cpu = deviation.cpu()
    sensitivity_cpu = sensitivity.cpu()
    for offset, sample_index in enumerate(sample_indices):
        sample_rows.append(
            {
                "source": source,
                "split": split,
                "sample_index": int(sample_index),
                "condition": condition,
                "alpha": float(alpha),
                "step_relative_rms": float(delta_rms[offset] / latent_rms[offset]),
                "cycle_relative_rms": float(candidate_cycle_error[offset]),
                "baseline_cycle_relative_rms": float(base_cycle_error[offset]),
                "cycle_ratio_to_base": float(cycle_ratio[offset]),
                "image_delta_rms": float(image_delta[offset]),
                "clipped_image_delta_rms": float(clipped_delta[offset]),
                "inception_cosine_to_base": float(feature_cosine[offset]),
                "decoded_pixel_clipping_fraction": float(clipping[offset]),
                "primary_hidden_rms_z": float(hidden_cpu[offset, PRIMARY_HIDDEN_STATE]),
                "early_hidden_abs_peak_z": float(hidden_cpu[offset, :9].abs().max()),
                "all_hidden_z_rms": float(hidden_cpu[offset].square().mean().sqrt()),
            }
        )
        for layer in range(hidden_cpu.shape[1]):
            layer_rows.append(
                {
                    "source": source,
                    "split": split,
                    "sample_index": int(sample_index),
                    "condition": condition,
                    "alpha": float(alpha),
                    "decoder_layer": layer,
                    "hidden_rms_z": float(hidden_cpu[offset, layer]),
                    "hidden_deviation_rms": float(deviation_cpu[offset, layer]),
                    "hidden_sensitivity": (
                        0.0
                        if condition == "baseline"
                        else float(sensitivity_cpu[offset, layer])
                    ),
                }
            )


@torch.no_grad()
def _evaluate_candidates(
    rae: torch.nn.Module,
    inception: torch.nn.Module,
    latents: torch.Tensor,
    *,
    source: str,
    split: str,
    first_index: int,
    batch_size: int,
    hidden_mean: torch.Tensor,
    hidden_std: torch.Tensor,
    device: torch.device,
    alpha_directions: list[tuple[str, float, torch.Tensor]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_rows: list[dict[str, object]] = []
    layer_rows: list[dict[str, object]] = []
    for start in range(0, len(latents), batch_size):
        stop = min(start + batch_size, len(latents))
        indices = list(range(int(first_index) + start, int(first_index) + stop))
        base = latents[start:stop].to(device)
        base_image, base_hidden = _decode(rae, base)
        base_cycle = rae.encode(base_image.clamp(0, 1))
        base_cycle_error = sample_rms(base_cycle - base) / sample_rms(base, 1e-12)
        base_features = inception_features(inception, base_image.clamp(0, 1))
        _append_condition_rows(
            sample_rows=sample_rows,
            layer_rows=layer_rows,
            source=source,
            split=split,
            sample_indices=indices,
            condition="baseline",
            alpha=0.0,
            base_latent=base,
            candidate=base,
            base_image=base_image,
            candidate_image=base_image,
            base_hidden=base_hidden,
            candidate_hidden=base_hidden,
            base_cycle_error=base_cycle_error,
            candidate_cycle_error=base_cycle_error,
            base_features=base_features,
            candidate_features=base_features,
            hidden_mean=hidden_mean,
            hidden_std=hidden_std,
        )
        if alpha_directions is None:
            directions = [("own", alpha, base_cycle - base) for alpha in ()]
        else:
            directions = [
                (condition, alpha, direction[start:stop].to(device))
                for condition, alpha, direction in alpha_directions
            ]
        for condition, alpha, direction in directions:
            candidate = interpolate_direction(base, direction, alpha)
            image, hidden = _decode(rae, candidate)
            cycle = rae.encode(image.clamp(0, 1))
            cycle_error = sample_rms(cycle - candidate) / sample_rms(candidate, 1e-12)
            features = inception_features(inception, image.clamp(0, 1))
            _append_condition_rows(
                sample_rows=sample_rows,
                layer_rows=layer_rows,
                source=source,
                split=split,
                sample_indices=indices,
                condition=condition,
                alpha=alpha,
                base_latent=base,
                candidate=candidate,
                base_image=base_image,
                candidate_image=image,
                base_hidden=base_hidden,
                candidate_hidden=hidden,
                base_cycle_error=base_cycle_error,
                candidate_cycle_error=cycle_error,
                base_features=base_features,
                candidate_features=features,
                hidden_mean=hidden_mean,
                hidden_std=hidden_std,
            )
    return pd.DataFrame(sample_rows), pd.DataFrame(layer_rows)


@torch.no_grad()
def _cycle_residuals(
    rae: torch.nn.Module,
    latents: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    values = []
    for start in range(0, len(latents), batch_size):
        latent = latents[start : start + batch_size].to(device)
        image, _ = _decode(rae, latent)
        values.append((rae.encode(image.clamp(0, 1)) - latent).cpu())
    return torch.cat(values)


@torch.no_grad()
def _evaluate_own_sweep(
    rae: torch.nn.Module,
    inception: torch.nn.Module,
    latents: torch.Tensor,
    *,
    source: str,
    alphas: tuple[float, ...],
    batch_size: int,
    hidden_mean: torch.Tensor,
    hidden_std: torch.Tensor,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_frames = []
    layer_frames = []
    for start in range(0, len(latents), batch_size):
        batch = latents[start : start + batch_size]
        residual = _cycle_residuals(rae, batch, batch_size=batch_size, device=device)
        directions = [("own", alpha, residual) for alpha in alphas]
        samples, layers = _evaluate_candidates(
            rae,
            inception,
            batch,
            source=source,
            split="calibration",
            first_index=start,
            batch_size=batch_size,
            hidden_mean=hidden_mean,
            hidden_std=hidden_std,
            device=device,
            alpha_directions=directions,
        )
        sample_frames.append(samples)
        layer_frames.append(layers)
    return pd.concat(sample_frames, ignore_index=True), pd.concat(layer_frames, ignore_index=True)


def _summaries(
    samples: pd.DataFrame,
    layers: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_summary = (
        samples.groupby(["source", "condition", "alpha"], as_index=False)
        .agg(
            sample_count=("sample_index", "count"),
            step_relative_rms_median=("step_relative_rms", "median"),
            cycle_relative_rms_median=("cycle_relative_rms", "median"),
            cycle_ratio_median=("cycle_ratio_to_base", "median"),
            image_delta_rms_median=("image_delta_rms", "median"),
            inception_cosine_median=("inception_cosine_to_base", "median"),
            primary_hidden_z_median=("primary_hidden_rms_z", "median"),
            early_hidden_abs_peak_z_median=("early_hidden_abs_peak_z", "median"),
        )
    )
    layer_summary = (
        layers.groupby(["source", "condition", "alpha", "decoder_layer"], as_index=False)
        .agg(
            hidden_rms_z_median=("hidden_rms_z", "median"),
            hidden_deviation_rms_median=("hidden_deviation_rms", "median"),
            hidden_sensitivity_median=("hidden_sensitivity", "median"),
        )
    )
    return sample_summary, layer_summary


def _plot(
    calibration: pd.DataFrame,
    samples: pd.DataFrame,
    layers: pd.DataFrame,
    selected_alpha: float,
    path: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(17, 12), constrained_layout=True)
    condition_colors = {
        "baseline": "#4c78a8",
        "own": "#59a14f",
        "shuffled": "#b279a2",
        "random": "#e15759",
        "opposite": "#f28e2b",
    }
    own_cal = calibration[calibration.condition == "own"]
    for source, values in own_cal.groupby("source"):
        curve = values.groupby("alpha").cycle_ratio_to_base.median()
        axes[0, 0].plot(curve.index, curve.values, marker="o", label=source)
    axes[0, 0].axvline(selected_alpha, color="black", linestyle="--", linewidth=1)
    axes[0, 0].set_title("Calibration: own cycle direction")
    axes[0, 0].set_xlabel("alpha")
    axes[0, 0].set_ylabel("Median cycle error / baseline")
    axes[0, 0].legend(frameon=False)

    generated = samples[samples.source.isin(GENERATED_SOURCES)]
    cycle = generated.groupby(["source", "condition"]).cycle_ratio_to_base.median().unstack()
    cycle = cycle.reindex(index=GENERATED_SOURCES)
    cycle = 100.0 * (
        cycle.reindex(columns=("own", "shuffled", "random", "opposite")) - 1.0
    )
    cycle.plot.bar(
        ax=axes[0, 1],
        width=0.82,
        color=[condition_colors[column] for column in cycle.columns],
    )
    axes[0, 1].axhline(0, color="black", linewidth=1)
    axes[0, 1].set_title("Held-out cycle change vs baseline")
    axes[0, 1].set_ylabel("Median cycle error change (%)")
    axes[0, 1].tick_params(axis="x", rotation=0)
    axes[0, 1].legend(frameon=False, ncol=2)

    reverse = layers[layers.source.eq("reverse")]
    condition_order = ("baseline", "own", "shuffled", "random", "opposite")
    for condition in condition_order:
        values = reverse[reverse.condition.eq(condition)]
        curve = values.groupby("decoder_layer").hidden_rms_z.median()
        axes[1, 0].plot(
            curve.index,
            curve.values,
            marker="o",
            markersize=3,
            color=condition_colors[condition],
            label=condition,
        )
    axes[1, 0].axvline(PRIMARY_HIDDEN_STATE, color="black", linestyle="--", linewidth=1)
    axes[1, 0].set_title("Reverse path: full decoder hidden response")
    axes[1, 0].set_xlabel("Decoder hidden state")
    axes[1, 0].set_ylabel("Median clean z-score")
    axes[1, 0].legend(frameon=False, ncol=2)

    primary = generated.groupby(["source", "condition"]).primary_hidden_rms_z.median().unstack()
    primary = primary.reindex(
        index=GENERATED_SOURCES,
        columns=("baseline", "own", "shuffled", "random", "opposite"),
    )
    primary.plot.bar(
        ax=axes[1, 1],
        width=0.82,
        color=[condition_colors[column] for column in primary.columns],
    )
    axes[1, 1].set_title(f"Primary early anomaly at D{PRIMARY_HIDDEN_STATE}")
    axes[1, 1].set_ylabel("Median clean z-score")
    axes[1, 1].tick_params(axis="x", rotation=0)
    axes[1, 1].legend(frameon=False, ncol=2)
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    figure.suptitle("RAE cycle-direction causal intervention", fontsize=18)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(config: CycleDirectionConfig) -> Path | None:
    rank, world_size, device = _distributed(config.seed)
    if world_size != len(GENERATED_SOURCES):
        raise ValueError(f"expected four GPUs, got {world_size}")
    source = GENERATED_SOURCES[rank]
    phase0 = config.phase0_root.expanduser().resolve()
    output = config.output_root.expanduser().resolve() / config.run_name
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    payload = torch.load(_endpoint_path(phase0, source), map_location="cpu", weights_only=True)
    endpoints = payload["latents"].float()
    labels = payload["labels"].long()
    total_generated = config.calibration_count + config.test_count
    if total_generated > len(endpoints):
        raise ValueError("generated calibration and test counts exceed cached endpoints")
    _check_paired_labels(labels[:total_generated], world_size, device)
    branch = Path(str(payload["branch"]))
    model_config = OmegaConf.load(branch / "config.yaml")
    rae = _load_full_rae(model_config, device)
    rae.noise_tau = 0.0
    inception = load_inception(device)

    cache = config.audit_cache.expanduser().resolve()
    manifest = load_cache_manifest(cache)
    reference_start, reference_stop = split_range(
        config.clean_reference_count, rank, world_size
    )
    reference = CachedRAELatentDataset(
        cache, start=reference_start, stop=reference_stop
    )
    hidden_mean, hidden_std = _hidden_reference(
        rae, reference, batch_size=config.batch_size, device=device
    )

    started = perf_counter()
    calibration_latents = endpoints[: config.calibration_count]
    calibration_samples, calibration_layers = _evaluate_own_sweep(
        rae,
        inception,
        calibration_latents,
        source=source,
        alphas=config.alphas,
        batch_size=config.batch_size,
        hidden_mean=hidden_mean,
        hidden_std=hidden_std,
        device=device,
    )
    calibration_samples.to_csv(
        output / f"calibration_samples_rank{rank:02d}.csv", index=False
    )
    calibration_layers.to_csv(
        output / f"calibration_layers_rank{rank:02d}.csv", index=False
    )
    print(
        f"rank{rank} {source} calibration complete in {(perf_counter() - started) / 60:.1f}m",
        flush=True,
    )
    dist.barrier()

    selected_tensor = torch.zeros((), device=device, dtype=torch.float64)
    if rank == 0:
        merged_calibration = _merge_rank_csv(
            output, "calibration_samples", world_size
        )
        _merge_rank_csv(output, "calibration_layers", world_size)
        selected_alpha, alpha_summary = select_global_alpha(
            merged_calibration, feature_cosine_floor=config.feature_cosine_floor
        )
        alpha_summary.to_csv(output / "alpha_selection.csv", index=False)
        selected_tensor.fill_(selected_alpha)
    dist.broadcast(selected_tensor, src=0)
    selected_alpha = float(selected_tensor.item())

    test_latents = endpoints[
        config.calibration_count : config.calibration_count + config.test_count
    ]
    residuals = _cycle_residuals(
        rae, test_latents, batch_size=config.batch_size, device=device
    )
    directions = matched_intervention_directions(residuals, seed=config.seed + 911)
    alpha_directions = [
        (condition, selected_alpha, directions[condition])
        for condition in INTERVENTION_CONDITIONS
    ]
    test_samples, test_layers = _evaluate_candidates(
        rae,
        inception,
        test_latents,
        source=source,
        split="test",
        first_index=config.calibration_count,
        batch_size=config.batch_size,
        hidden_mean=hidden_mean,
        hidden_std=hidden_std,
        device=device,
        alpha_directions=alpha_directions,
    )

    calibration_count = int(manifest["calibration_count"])
    guard_start, guard_stop = split_range(config.clean_guardrail_count, rank, world_size)
    clean_start = calibration_count + config.clean_guardrail_offset + guard_start
    clean_count = guard_stop - guard_start
    clean_latents = _cache_slice(cache, clean_start, clean_count)
    clean_residuals = _cycle_residuals(
        rae, clean_latents, batch_size=config.batch_size, device=device
    )
    clean_samples, clean_layers = _evaluate_candidates(
        rae,
        inception,
        clean_latents,
        source="clean_test",
        split="test",
        first_index=config.clean_guardrail_offset + guard_start,
        batch_size=config.batch_size,
        hidden_mean=hidden_mean,
        hidden_std=hidden_std,
        device=device,
        alpha_directions=[("own", selected_alpha, clean_residuals)],
    )
    pd.concat([test_samples, clean_samples], ignore_index=True).to_csv(
        output / f"test_samples_rank{rank:02d}.csv", index=False
    )
    pd.concat([test_layers, clean_layers], ignore_index=True).to_csv(
        output / f"test_layers_rank{rank:02d}.csv", index=False
    )
    print(
        f"rank{rank} {source} held-out complete in {(perf_counter() - started) / 60:.1f}m",
        flush=True,
    )
    dist.barrier()

    if rank != 0:
        _finish()
        return None

    samples = _merge_rank_csv(output, "test_samples", world_size)
    layers = _merge_rank_csv(output, "test_layers", world_size)
    sample_summary, layer_summary = _summaries(samples, layers)
    sample_summary.to_csv(output / "test_sample_summary.csv", index=False)
    layer_summary.to_csv(output / "test_layer_summary.csv", index=False)
    gate = cycle_direction_gate(samples)
    calibration = pd.read_csv(output / "calibration_samples.csv")
    _plot(
        calibration,
        samples,
        layers,
        selected_alpha,
        output / "cycle_direction_intervention.png",
    )
    result = {
        "config": {
            **asdict(config),
            "phase0_root": str(config.phase0_root),
            "audit_cache": str(config.audit_cache),
            "output_root": str(config.output_root),
        },
        "selected_alpha": selected_alpha,
        "primary_hidden_state": PRIMARY_HIDDEN_STATE,
        "old_0b_hidden_state_correction": (
            "the prior +6.615 value was output.hidden_states[2], not decoder state 7"
        ),
        "gate": gate,
        "protocol": {
            "generated_calibration_indices": [0, config.calibration_count],
            "generated_test_indices": [
                config.calibration_count,
                config.calibration_count + config.test_count,
            ],
            "alpha_selection": (
                "own direction only; minimize mean per-path median cycle ratio while every "
                "path has median Inception cosine >= feature floor"
            ),
            "heldout_controls": list(INTERVENTION_CONDITIONS),
            "direction_norm_control": "all controls match each sample's cycle-residual RMS",
            "clean_guardrail": (
                f"ImageNet validation logical indices {config.clean_guardrail_offset}:"
                f"{config.clean_guardrail_offset + config.clean_guardrail_count}"
            ),
            "cycle_operator_warning": (
                "E(clamp(D(z))) is a closure map, not assumed to be an exact manifold projection"
            ),
            "numerics": "fp32, TF32 disabled",
        },
        "thresholds": asdict(CycleDirectionThresholds()),
    }
    _atomic_json(output / "result.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    print("\nHeld-out summary:\n", sample_summary.to_string(index=False), flush=True)
    _finish()
    return output


def parse_args() -> CycleDirectionConfig:
    defaults = CycleDirectionConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase0-root", type=Path, default=defaults.phase0_root)
    parser.add_argument("--audit-cache", type=Path, default=defaults.audit_cache)
    parser.add_argument("--output-root", type=Path, default=defaults.output_root)
    parser.add_argument("--run-name", default=defaults.run_name)
    parser.add_argument("--calibration-count", type=int, default=defaults.calibration_count)
    parser.add_argument("--test-count", type=int, default=defaults.test_count)
    parser.add_argument("--clean-reference-count", type=int, default=defaults.clean_reference_count)
    parser.add_argument("--clean-guardrail-count", type=int, default=defaults.clean_guardrail_count)
    parser.add_argument("--clean-guardrail-offset", type=int, default=defaults.clean_guardrail_offset)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--alphas", default=",".join(str(value) for value in defaults.alphas))
    parser.add_argument("--feature-cosine-floor", type=float, default=defaults.feature_cosine_floor)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    args = parser.parse_args()
    alphas = tuple(float(value) for value in args.alphas.split(",") if value.strip())
    if not alphas or min(alphas) <= 0 or max(alphas) > 1:
        raise ValueError("alphas must lie in (0, 1]")
    return CycleDirectionConfig(
        phase0_root=args.phase0_root,
        audit_cache=args.audit_cache,
        output_root=args.output_root,
        run_name=args.run_name,
        calibration_count=args.calibration_count,
        test_count=args.test_count,
        clean_reference_count=args.clean_reference_count,
        clean_guardrail_count=args.clean_guardrail_count,
        clean_guardrail_offset=args.clean_guardrail_offset,
        batch_size=args.batch_size,
        alphas=alphas,
        feature_cosine_floor=args.feature_cosine_floor,
        seed=args.seed,
    )


if __name__ == "__main__":
    run(parse_args())
