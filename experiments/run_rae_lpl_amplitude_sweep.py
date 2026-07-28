"""Trace Flow and LPL error directions through the frozen RAE decoder by radius."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
RAE_ROOT = ROOT / "external" / "RAE"
RAE_SRC = RAE_ROOT / "src"
for path in (ROOT, RAE_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.rae_decoder_risk_phase0 import decoder_hidden_features  # noqa: E402
from experiments.rae_latent_cache import CachedRAELatentDataset  # noqa: E402
from experiments.rae_lpl_error_geometry import (  # noqa: E402
    feature_normalization_decomposition,
    raw_feature_layer_losses,
    sample_rms,
    scale_direction_to_rms,
    unit_rms_direction,
)
from experiments.rae_strict_lpl import (  # noqa: E402
    decoder_hidden_indices,
    flow_clean_estimate,
    strict_lpl_per_sample,
)
from experiments.rae_teacher_rollout_gap import load_frozen_decoder  # noqa: E402
from experiments.run_rae_lpl_error_geometry import (  # noqa: E402
    DEFAULT_CACHE,
    DEFAULT_CONFIG,
    CheckpointPair,
    atomic_json,
    batched_decoder_features,
    deterministic_tensor,
    distributed_context,
    finish_distributed,
    load_manifest,
    load_stage2,
    parse_pair,
    repeated_features,
    spatial_features,
)


DEFAULT_OUTPUT = Path.home() / "data/eqvae/experiments/rae_lpl_amplitude_sweep"
DEFAULT_FRACTIONS = (0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0)


@torch.no_grad()
def evaluate_amplitude_sweep(
    *,
    rae: torch.nn.Module,
    flow_model: torch.nn.Module,
    lpl_model: torch.nn.Module,
    clean: torch.Tensor,
    label: torch.Tensor,
    noise: torch.Tensor,
    ratio: float,
    amplitude_fractions: Sequence[float],
    hidden_indices: Sequence[int],
    decoder_batch_size: int,
    strict_lpl_config: dict[str, object],
) -> list[dict[str, float | str]]:
    time_value = float(ratio) / (1.0 + float(ratio))
    time = clean.new_full((1,), time_value)
    noisy = (1.0 - time_value) * clean + time_value * noise
    flow_clean = flow_clean_estimate(noisy, flow_model(noisy, time, y=label), time)
    lpl_clean = flow_clean_estimate(noisy, lpl_model(noisy, time, y=label), time)
    flow_error = flow_clean - clean
    lpl_error = lpl_clean - clean
    common_rms = torch.minimum(sample_rms(flow_error), sample_rms(lpl_error))
    directions = {
        "flow": unit_rms_direction(flow_error),
        "lpl": unit_rms_direction(lpl_error),
    }

    names = []
    candidates = []
    amplitudes = []
    for fraction in amplitude_fractions:
        amplitude = float(fraction) * common_rms
        for branch, direction in directions.items():
            names.append((branch, float(fraction)))
            amplitudes.append(float(amplitude.item()))
            candidates.append(clean + scale_direction_to_rms(direction, amplitude))
    candidate_latents = torch.cat(candidates, dim=0)

    reference = decoder_hidden_features(rae, clean, hidden_indices=hidden_indices)
    candidate_features = batched_decoder_features(
        rae,
        candidate_latents,
        hidden_indices=hidden_indices,
        batch_size=int(decoder_batch_size),
    )
    repeated_reference = repeated_features(reference, len(candidate_latents))
    raw_layers = raw_feature_layer_losses(candidate_features, repeated_reference)
    raw_loss = raw_layers.mean(dim=1)
    normalization = feature_normalization_decomposition(
        candidate_features,
        repeated_reference,
        eps=float(strict_lpl_config["normalization_eps"]),
    )
    strict_loss, strict_details = strict_lpl_per_sample(
        repeated_features(spatial_features(reference), len(candidate_latents)),
        spatial_features(candidate_features),
        layer_weights=[1.0] * len(hidden_indices),
        outlier_quantile=float(strict_lpl_config["outlier_quantile"]),
        outlier_opening=int(strict_lpl_config["outlier_opening"]),
        outlier_closing=int(strict_lpl_config["outlier_closing"]),
        eps=float(strict_lpl_config["normalization_eps"]),
    )

    rows = []
    for index, ((branch, fraction), amplitude) in enumerate(
        zip(names, amplitudes, strict=True)
    ):
        denominator = max(amplitude**2, 1e-30)
        row: dict[str, float | str] = {
            "branch": branch,
            "time": time_value,
            "noise_to_signal_ratio": float(ratio),
            "amplitude_fraction": fraction,
            "amplitude_rms": amplitude,
            "common_realistic_rms": float(common_rms.item()),
            "clean_rms": float(sample_rms(clean).item()),
            "raw_loss": float(raw_loss[index].item()),
            "raw_gain": float(raw_loss[index].item()) / denominator,
            "strict_lpl": float(strict_loss[index].item()),
            "strict_gain": float(strict_loss[index].item()) / denominator,
        }
        for metric_name, metric_values in normalization.items():
            metric_mean = float(metric_values[index].mean().item())
            row[metric_name] = metric_mean
            if metric_name.endswith("normalized"):
                row[f"{metric_name}_gain"] = metric_mean / denominator
        for layer_index in range(len(hidden_indices)):
            layer_raw = float(raw_layers[index, layer_index].item())
            layer_strict = float(
                strict_details["layer_losses"][index, layer_index].item()
            )
            row[f"raw_loss_layer{layer_index}"] = layer_raw
            row[f"raw_gain_layer{layer_index}"] = layer_raw / denominator
            row[f"strict_lpl_layer{layer_index}"] = layer_strict
            row[f"strict_gain_layer{layer_index}"] = layer_strict / denominator
            for metric_name, metric_values in normalization.items():
                row[f"{metric_name}_layer{layer_index}"] = float(
                    metric_values[index, layer_index].item()
                )
        rows.append(row)
    return rows


def pair_sweep_rows(rows: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "training_seed",
        "sample_index",
        "label",
        "noise_to_signal_ratio",
        "time",
        "amplitude_fraction",
        "amplitude_rms",
        "common_realistic_rms",
        "clean_rms",
    ]
    value_columns = [
        column
        for column in rows.columns
        if column.startswith(
            (
                "raw_",
                "strict_",
                "target_normalized",
                "prediction_normalized",
                "symmetric_normalized",
                "prediction_over_target_variance",
                "centered_channel_cosine",
            )
        )
        and column not in keys
    ]
    flow = rows[rows["branch"] == "flow"][keys + value_columns].rename(
        columns={column: f"flow_{column}" for column in value_columns}
    )
    lpl = rows[rows["branch"] == "lpl"][keys + value_columns].rename(
        columns={column: f"lpl_{column}" for column in value_columns}
    )
    paired = flow.merge(lpl, on=keys, how="inner", validate="one_to_one")
    if len(paired) * 2 != len(rows):
        raise RuntimeError("amplitude sweep did not form complete Flow/LPL pairs")
    return paired


def summarize_sweep(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (seed, fraction), group in paired.groupby(
        ["training_seed", "amplitude_fraction"], sort=True
    ):
        raw_ratio = group["lpl_raw_loss"] / group["flow_raw_loss"].clip(lower=1e-30)
        strict_ratio = group["lpl_strict_lpl"] / group["flow_strict_lpl"].clip(
            lower=1e-30
        )
        row = {
            "training_seed": int(seed),
            "amplitude_fraction": float(fraction),
            "observations": int(len(group)),
            "raw_lpl_over_flow_gmean": float(
                np.exp(np.log(raw_ratio.clip(lower=1e-30)).mean())
            ),
            "raw_lpl_better_fraction": float((raw_ratio < 1.0).mean()),
            "strict_lpl_over_flow_gmean": float(
                np.exp(np.log(strict_ratio.clip(lower=1e-30)).mean())
            ),
            "strict_lpl_better_fraction": float((strict_ratio < 1.0).mean()),
        }
        for metric_name in (
            "target_normalized",
            "prediction_normalized",
            "symmetric_normalized",
            "prediction_over_target_variance_gmean",
            "centered_channel_cosine",
        ):
            if metric_name == "centered_channel_cosine":
                row[f"flow_{metric_name}_mean"] = float(
                    group[f"flow_{metric_name}"].mean()
                )
                row[f"lpl_{metric_name}_mean"] = float(
                    group[f"lpl_{metric_name}"].mean()
                )
                continue
            metric_ratio = group[f"lpl_{metric_name}"] / group[
                f"flow_{metric_name}"
            ].clip(lower=1e-30)
            row[f"{metric_name}_lpl_over_flow_gmean"] = float(
                np.exp(np.log(metric_ratio.clip(lower=1e-30)).mean())
            )
        for layer_index in range(5):
            layer_ratio = group[f"lpl_strict_lpl_layer{layer_index}"] / group[
                f"flow_strict_lpl_layer{layer_index}"
            ].clip(lower=1e-30)
            row[f"strict_layer{layer_index}_lpl_over_flow_gmean"] = float(
                np.exp(np.log(layer_ratio.clip(lower=1e-30)).mean())
            )
        rows.append(row)
    return pd.DataFrame(rows)


def crossover_table(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["training_seed", "sample_index", "noise_to_signal_ratio"]
    for key, group in paired.groupby(keys, sort=True):
        group = group.sort_values("amplitude_fraction")
        ratio = group["lpl_strict_lpl"] / group["flow_strict_lpl"].clip(lower=1e-30)
        better = group.loc[ratio < 1.0, "amplitude_fraction"]
        rows.append(
            {
                "training_seed": int(key[0]),
                "sample_index": int(key[1]),
                "noise_to_signal_ratio": float(key[2]),
                "first_sampled_lpl_better_fraction": (
                    float(better.iloc[0]) if len(better) else float("nan")
                ),
                "strict_ratio_at_min_fraction": float(ratio.iloc[0]),
                "strict_ratio_at_full_fraction": float(ratio.iloc[-1]),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--pair", type=parse_pair, action="append", required=True)
    parser.add_argument("--state-key", choices=("model", "ema"), default="ema")
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument(
        "--noise-ratios", type=float, nargs="+", default=(0.5, 1.0, 2.0, 3.0)
    )
    parser.add_argument(
        "--amplitude-fractions", type=float, nargs="+", default=DEFAULT_FRACTIONS
    )
    parser.add_argument("--decoder-batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank, world_size, device = distributed_context(args.seed)
    if len(args.pair) != world_size:
        finish_distributed()
        raise ValueError(
            f"received {len(args.pair)} checkpoint pairs for world size {world_size}"
        )
    fractions = tuple(float(value) for value in args.amplitude_fractions)
    if not fractions or min(fractions) <= 0.0 or max(fractions) > 1.0:
        finish_distributed()
        raise ValueError("amplitude fractions must be in (0, 1]")
    if tuple(sorted(set(fractions))) != fractions:
        finish_distributed()
        raise ValueError("amplitude fractions must be unique and increasing")

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = args.cache.expanduser().resolve()
    manifest = load_manifest(cache)
    calibration_count = int(manifest["calibration_count"])
    test_count = int(manifest["test_count"])
    if not 0 < int(args.sample_count) <= test_count:
        finish_distributed()
        raise ValueError("invalid held-out sample count")
    dataset = CachedRAELatentDataset(
        cache,
        start=calibration_count,
        stop=calibration_count + int(args.sample_count),
    )

    config = OmegaConf.load(args.config.expanduser().resolve())
    stage1_config = OmegaConf.create(OmegaConf.to_container(config.stage_1, resolve=True))
    stage2_config = OmegaConf.create(OmegaConf.to_container(config.stage_2, resolve=True))
    strict_config = dict(
        OmegaConf.to_container(config.training.strict_lpl, resolve=True)
    )
    rae = load_frozen_decoder(stage1_config)
    rae = rae.to(device=device, dtype=torch.float32).requires_grad_(False).eval()
    hidden_indices = decoder_hidden_indices(
        len(rae.decoder.decoder_layers),
        tuple(float(value) for value in strict_config["layer_fractions"]),
    )
    pair: CheckpointPair = args.pair[rank]
    flow_model, flow_step = load_stage2(
        stage2_config, pair.flow, args.state_key, device
    )
    lpl_model, lpl_step = load_stage2(stage2_config, pair.lpl, args.state_key, device)
    if flow_step != lpl_step:
        finish_distributed()
        raise ValueError("paired checkpoints have different update steps")

    started = perf_counter()
    rows = []
    for sample_index in range(int(args.sample_count)):
        clean_cpu, label_value = dataset[sample_index]
        clean = clean_cpu[None].to(device=device, dtype=torch.float32)
        label = torch.tensor([label_value], device=device, dtype=torch.long)
        noise = deterministic_tensor(
            clean.shape,
            seed=int(args.seed),
            sample_index=sample_index,
            offset=10_000_019,
        ).to(device)
        for ratio in args.noise_ratios:
            observation_rows = evaluate_amplitude_sweep(
                rae=rae,
                flow_model=flow_model,
                lpl_model=lpl_model,
                clean=clean,
                label=label,
                noise=noise,
                ratio=float(ratio),
                amplitude_fractions=fractions,
                hidden_indices=hidden_indices,
                decoder_batch_size=int(args.decoder_batch_size),
                strict_lpl_config=strict_config,
            )
            for row in observation_rows:
                row.update(
                    {
                        "checkpoint_pair": pair.name,
                        "training_seed": int(pair.training_seed),
                        "checkpoint_step": int(flow_step),
                        "sample_index": int(sample_index),
                        "label": int(label_value),
                    }
                )
            rows.extend(observation_rows)
        print(
            f"[rank {rank}] {pair.name}: sample {sample_index + 1}/{args.sample_count}",
            flush=True,
        )

    pd.DataFrame(rows).to_csv(output / f"rows_rank{rank:02d}.csv", index=False)
    atomic_json(
        output / f"manifest_rank{rank:02d}.json",
        {
            "checkpoint_pair": pair.name,
            "training_seed": int(pair.training_seed),
            "flow_checkpoint": str(pair.flow),
            "lpl_checkpoint": str(pair.lpl),
            "checkpoint_step": int(flow_step),
            "state_key": args.state_key,
            "sample_count": int(args.sample_count),
            "noise_ratios": [float(value) for value in args.noise_ratios],
            "amplitude_fractions": list(fractions),
            "hidden_indices": list(hidden_indices),
            "elapsed_seconds": perf_counter() - started,
        },
    )
    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        table = pd.concat(
            [pd.read_csv(output / f"rows_rank{index:02d}.csv") for index in range(world_size)],
            ignore_index=True,
        )
        table.to_csv(output / "rows.csv", index=False)
        paired = pair_sweep_rows(table)
        paired.to_csv(output / "paired.csv", index=False)
        summary = summarize_sweep(paired)
        summary.to_csv(output / "summary.csv", index=False)
        crossovers = crossover_table(paired)
        crossovers.to_csv(output / "crossovers.csv", index=False)
        atomic_json(
            output / "summary.json",
            {
                "experiment": "RAE Flow-vs-LPL matched-direction amplitude sweep",
                "dataset": "ImageNet-1k validation latent cache",
                "precision": "fp32",
                "state_key": args.state_key,
                "sample_count_per_seed": int(args.sample_count),
                "training_seed_count": int(world_size),
                "observation_count": int(
                    len(crossovers)
                ),
                "amplitude_fractions": list(fractions),
                "fraction_with_observed_crossover": float(
                    crossovers["first_sampled_lpl_better_fraction"].notna().mean()
                ),
                "median_first_sampled_lpl_better_fraction": float(
                    crossovers["first_sampled_lpl_better_fraction"].median()
                ),
            },
        )
        print("\nAmplitude sweep summary")
        print(summary.to_string(index=False))
        print("\nCrossover summary")
        print(crossovers.describe(include="all").to_string())
        print(output)

    del flow_model, lpl_model, rae
    gc.collect()
    torch.cuda.empty_cache()
    finish_distributed()


if __name__ == "__main__":
    main()
