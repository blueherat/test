"""Audit raw, detached-stat, and full LPL gradients on frozen RAE checkpoints."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from time import perf_counter

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

from experiments.rae_latent_cache import CachedRAELatentDataset  # noqa: E402
from experiments.rae_lpl_detach_audit import (  # noqa: E402
    gradient_decomposition_metrics,
    lpl_loss_variants_per_sample,
)
from experiments.rae_strict_lpl import (  # noqa: E402
    decoder_feature_pyramid,
    decoder_hidden_indices,
    flow_clean_estimate,
)
from experiments.rae_teacher_rollout_gap import (  # noqa: E402
    configure_fp32,
    load_frozen_decoder,
)
from utils.model_utils import instantiate_from_config  # noqa: E402


DEFAULT_CONFIG = ROOT / "experiments/configs/rae_strict_lpl_ditdh_s_dinov2.yaml"
DEFAULT_CACHE = (
    Path.home()
    / "data/eqvae/cache/rae_decoder_risk_phase0/seed20260718_cal1024_test2048_fp32"
)
DEFAULT_OUTPUT = Path.home() / "data/eqvae/experiments/rae_lpl_detach_audit"


def parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must be NAME=PATH")
    name, path = value.split("=", 1)
    checkpoint = Path(path).expanduser().resolve()
    if not name:
        raise argparse.ArgumentTypeError("checkpoint name cannot be empty")
    if not checkpoint.exists():
        raise argparse.ArgumentTypeError(f"checkpoint does not exist: {checkpoint}")
    return name, checkpoint


def distributed_context(seed: int) -> tuple[int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("detach audit requires CUDA")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if "RANK" in os.environ:
        dist.init_process_group("nccl", device_id=device)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank, world_size = 0, 1
    configure_fp32(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)
    return rank, world_size, device


def finish_distributed() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def deterministic_noise(
    shape: tuple[int, ...],
    *,
    seed: int,
    sample_index: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(
        int(seed) + int(sample_index) * 1_000_003 + 10_000_019
    )
    return torch.randn(shape, generator=generator, dtype=torch.float32)


def load_stage2(
    stage2_config: OmegaConf,
    checkpoint: Path,
    state_key: str,
    device: torch.device,
) -> tuple[torch.nn.Module, int]:
    model = instantiate_from_config(stage2_config).to(
        device=device, dtype=torch.float32
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state_key not in payload:
        raise KeyError(f"{checkpoint} lacks state key {state_key!r}")
    model.load_state_dict(payload[state_key], strict=True)
    model.requires_grad_(False).eval()
    step = int(payload.get("step", -1))
    del payload
    gc.collect()
    return model, step


def _gradient(
    loss: torch.Tensor,
    predicted_clean: torch.Tensor,
    *,
    retain_graph: bool,
) -> torch.Tensor:
    return torch.autograd.grad(
        loss.sum(),
        predicted_clean,
        retain_graph=retain_graph,
        create_graph=False,
    )[0]


def evaluate_observation(
    *,
    rae: torch.nn.Module,
    model: torch.nn.Module,
    clean: torch.Tensor,
    label: torch.Tensor,
    noise: torch.Tensor,
    ratio: float,
    hidden_indices: tuple[int, ...],
    strict_config: dict[str, object],
) -> dict[str, float]:
    """Evaluate one sample, time, and Stage-2 checkpoint."""

    time_value = float(ratio) / (1.0 + float(ratio))
    time = clean.new_full((1,), time_value)
    noisy = (1.0 - time_value) * clean + time_value * noise
    target_velocity = noise - clean
    with torch.no_grad():
        prediction = model(noisy, time, y=label)
        predicted_clean_value = flow_clean_estimate(noisy, prediction, time)
        target_features = decoder_feature_pyramid(
            rae, clean, layer_indices=hidden_indices
        )

    predicted_clean = predicted_clean_value.detach().requires_grad_(True)
    predicted_features = decoder_feature_pyramid(
        rae, predicted_clean, layer_indices=hidden_indices
    )
    losses, details = lpl_loss_variants_per_sample(
        target_features,
        predicted_features,
        layer_weights=[1.0] * len(hidden_indices),
        outlier_quantile=float(strict_config["outlier_quantile"]),
        outlier_opening=int(strict_config["outlier_opening"]),
        outlier_closing=int(strict_config["outlier_closing"]),
        eps=float(strict_config["normalization_eps"]),
    )
    forward_relative_difference = (
        (losses["prediction_full"] - losses["prediction_detach"]).abs()
        / losses["prediction_full"].abs().clamp_min(1e-30)
    )
    raw_gradient = _gradient(losses["raw"], predicted_clean, retain_graph=True)
    detach_gradient = _gradient(
        losses["prediction_detach"], predicted_clean, retain_graph=True
    )
    full_gradient = _gradient(
        losses["prediction_full"], predicted_clean, retain_graph=True
    )
    log_variance_gradient = _gradient(
        details["mean_log_prediction_variance"],
        predicted_clean,
        retain_graph=False,
    )
    gradient_metrics = gradient_decomposition_metrics(
        raw_gradient,
        detach_gradient,
        full_gradient,
        log_variance_gradient,
    )

    row = {
        "noise_to_signal_ratio": float(ratio),
        "time": time_value,
        "flow_loss": float(
            (prediction - target_velocity).square().flatten(1).mean().item()
        ),
        "clean_latent_mse": float(
            (predicted_clean_value - clean).square().flatten(1).mean().item()
        ),
        "raw_feature_loss": float(losses["raw"].item()),
        "prediction_detach_loss": float(losses["prediction_detach"].item()),
        "prediction_full_loss": float(losses["prediction_full"].item()),
        "full_detach_forward_relative_difference": float(
            forward_relative_difference.item()
        ),
        "prediction_over_target_variance_gmean": float(
            details["prediction_over_target_variance_layers"]
            .clamp_min(1e-30)
            .log()
            .mean()
            .exp()
            .item()
        ),
        "prediction_over_target_std_gmean": float(
            details["prediction_over_target_std_layers"]
            .clamp_min(1e-30)
            .log()
            .mean()
            .exp()
            .item()
        ),
        "centered_cosine_mean": float(
            details["centered_cosine_layers"].mean().item()
        ),
        "normalized_mean_error_mean": float(
            details["normalized_mean_error_layers"].mean().item()
        ),
        "mask_keep_fraction_mean": float(
            details["mask_keep_fraction_layers"].mean().item()
        ),
    }
    for name, value in gradient_metrics.items():
        row[name] = float(value.item())
    for layer_index in range(len(hidden_indices)):
        row[f"raw_layer{layer_index}"] = float(
            details["raw_layers"][0, layer_index].item()
        )
        row[f"detach_layer{layer_index}"] = float(
            details["prediction_detach_layers"][0, layer_index].item()
        )
        row[f"full_layer{layer_index}"] = float(
            details["prediction_full_layers"][0, layer_index].item()
        )
        row[f"variance_ratio_layer{layer_index}"] = float(
            details["prediction_over_target_variance_layers"][
                0, layer_index
            ].item()
        )
        row[f"std_ratio_layer{layer_index}"] = float(
            details["prediction_over_target_std_layers"][0, layer_index].item()
        )
        row[f"centered_cosine_layer{layer_index}"] = float(
            details["centered_cosine_layers"][0, layer_index].item()
        )
        row[f"normalized_mean_error_layer{layer_index}"] = float(
            details["normalized_mean_error_layers"][0, layer_index].item()
        )
    return row


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "checkpoint",
        "state_key",
        "checkpoint_step",
        "noise_to_signal_ratio",
    ]
    metric_columns = [
        column
        for column in rows.columns
        if column
        not in {
            *group_columns,
            "checkpoint_path",
            "sample_index",
            "label",
            "rank",
        }
    ]
    return (
        rows.groupby(group_columns, sort=True)[metric_columns]
        .mean(numeric_only=True)
        .reset_index()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--checkpoint", action="append", type=parse_checkpoint, required=True
    )
    parser.add_argument("--state-key", choices=("model", "ema"), default="model")
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument(
        "--noise-ratios", type=float, nargs="+", default=(0.5, 1.0, 2.0, 3.0)
    )
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank, world_size, device = distributed_context(args.seed)
    if int(args.sample_count) < 1:
        finish_distributed()
        raise ValueError("sample-count must be positive")

    cache = args.cache.expanduser().resolve()
    manifest_path = cache / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not bool(manifest.get("complete")):
        finish_distributed()
        raise RuntimeError(f"cache is incomplete: {cache}")
    calibration_count = int(manifest["calibration_count"])
    if int(args.sample_count) > int(manifest["test_count"]):
        finish_distributed()
        raise ValueError("sample-count exceeds held-out cache")
    dataset = CachedRAELatentDataset(
        cache,
        start=calibration_count,
        stop=calibration_count + int(args.sample_count),
    )

    config = OmegaConf.load(args.config.expanduser().resolve())
    stage1_config = OmegaConf.create(
        OmegaConf.to_container(config.stage_1, resolve=True)
    )
    stage2_config = OmegaConf.create(
        OmegaConf.to_container(config.stage_2, resolve=True)
    )
    if OmegaConf.select(stage1_config, "params.encoder_params.hidden_size") is None:
        OmegaConf.update(
            stage1_config,
            "params.encoder_params.hidden_size",
            int(stage2_config.params.in_channels),
            merge=True,
        )
    strict_config = dict(
        OmegaConf.to_container(config.training.strict_lpl, resolve=True)
    )
    rae = load_frozen_decoder(stage1_config)
    rae = rae.to(device=device, dtype=torch.float32).requires_grad_(False).eval()
    hidden_indices = decoder_hidden_indices(
        len(rae.decoder.decoder_layers),
        tuple(float(value) for value in strict_config["layer_fractions"]),
    )

    local_indices = list(range(rank, int(args.sample_count), world_size))
    local_samples = []
    for sample_index in local_indices:
        clean_cpu, label_value = dataset[sample_index]
        local_samples.append(
            (
                sample_index,
                clean_cpu[None].to(device=device, dtype=torch.float32),
                torch.tensor([label_value], device=device, dtype=torch.long),
                int(label_value),
            )
        )

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    rows = []
    checkpoint_records = []
    for checkpoint_name, checkpoint_path in args.checkpoint:
        model, checkpoint_step = load_stage2(
            stage2_config, checkpoint_path, args.state_key, device
        )
        checkpoint_records.append(
            {
                "name": checkpoint_name,
                "path": str(checkpoint_path),
                "step": int(checkpoint_step),
            }
        )
        for local_offset, (sample_index, clean, label, label_value) in enumerate(
            local_samples
        ):
            noise = deterministic_noise(
                tuple(clean.shape),
                seed=int(args.seed),
                sample_index=int(sample_index),
            ).to(device=device)
            for ratio in args.noise_ratios:
                row = evaluate_observation(
                    rae=rae,
                    model=model,
                    clean=clean,
                    label=label,
                    noise=noise,
                    ratio=float(ratio),
                    hidden_indices=hidden_indices,
                    strict_config=strict_config,
                )
                row.update(
                    {
                        "checkpoint": checkpoint_name,
                        "checkpoint_path": str(checkpoint_path),
                        "state_key": args.state_key,
                        "checkpoint_step": int(checkpoint_step),
                        "sample_index": int(sample_index),
                        "label": int(label_value),
                        "rank": int(rank),
                    }
                )
                rows.append(row)
            print(
                f"[rank {rank}] {checkpoint_name}: "
                f"{local_offset + 1}/{len(local_samples)} samples",
                flush=True,
            )
        del model
        gc.collect()
        torch.cuda.empty_cache()

    rank_rows = pd.DataFrame(rows)
    rank_rows.to_csv(output / f"rows_rank{rank:02d}.csv", index=False)
    (output / f"manifest_rank{rank:02d}.json").write_text(
        json.dumps(
            {
                "rank": int(rank),
                "world_size": int(world_size),
                "sample_indices": local_indices,
                "rows": int(len(rank_rows)),
                "elapsed_seconds": perf_counter() - started,
                "checkpoints": checkpoint_records,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    barrier()

    if rank == 0:
        table = pd.concat(
            [
                pd.read_csv(output / f"rows_rank{index:02d}.csv")
                for index in range(world_size)
            ],
            ignore_index=True,
        )
        table.sort_values(
            ["checkpoint", "sample_index", "noise_to_signal_ratio"],
            inplace=True,
        )
        table.to_csv(output / "rows.csv", index=False)
        summary = summarize(table)
        summary.to_csv(output / "summary.csv", index=False)
        maximum_forward_difference = float(
            table["full_detach_forward_relative_difference"].max()
        )
        payload = {
            "experiment": "RAE LPL prediction-stat detach audit",
            "dataset": "ImageNet-1k validation latent cache",
            "cache": str(cache),
            "precision": "fp32",
            "tf32": False,
            "state_key": args.state_key,
            "sample_count": int(args.sample_count),
            "observation_count": int(len(table)),
            "noise_ratios": [float(value) for value in args.noise_ratios],
            "hidden_indices": list(hidden_indices),
            "checkpoints": checkpoint_records,
            "maximum_full_detach_forward_relative_difference": (
                maximum_forward_difference
            ),
            "elapsed_seconds_max_rank": max(
                json.loads(
                    (output / f"manifest_rank{index:02d}.json").read_text(
                        encoding="utf-8"
                    )
                )["elapsed_seconds"]
                for index in range(world_size)
            ),
        }
        (output / "summary.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print("\nDetach audit summary")
        display_columns = [
            "checkpoint",
            "noise_to_signal_ratio",
            "prediction_over_target_variance_gmean",
            "prediction_over_target_std_gmean",
            "centered_cosine_mean",
            "stats_over_full_gradient_rms",
            "full_detach_gradient_cosine",
            "raw_detach_gradient_cosine",
            "stats_descent_log_variance_cosine",
            "detach_descent_log_variance_cosine",
            "full_descent_log_variance_cosine",
        ]
        print(summary[display_columns].to_string(index=False))
        print(output)

    del rae
    gc.collect()
    torch.cuda.empty_cache()
    finish_distributed()


if __name__ == "__main__":
    main()
