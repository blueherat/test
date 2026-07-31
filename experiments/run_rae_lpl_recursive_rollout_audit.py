"""Compare old-RAE Flow/LPL checkpoints under recursive latent rollouts.

The first model query starts from an exact clean/noise interpolation used by
flow-matching training. Later queries consume states produced by the model's
own Euler updates. This isolates whether a paired decoder-feature advantage
survives the state shift induced by iterative sampling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
RAE_ROOT = ROOT / "external" / "RAE"
RAE_SRC = RAE_ROOT / "src"
for path in (ROOT, RAE_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.rae_lpl_detach_audit import (  # noqa: E402
    decoder_feature_objective_per_sample,
    tensor_rms,
)
from experiments.rae_strict_lpl import (  # noqa: E402
    decoder_feature_pyramid,
    decoder_hidden_indices,
    flow_clean_estimate,
)
from stage1 import RAE  # noqa: E402
from utils.model_utils import instantiate_from_config  # noqa: E402
from utils.train_utils import (  # noqa: E402
    ParquetImageNetDataset,
    center_crop_arr,
)


LPL_LAYER_FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)
PAIR_METRICS = (
    "final_raw_decoder_feature_loss",
    "final_latent_error_rms",
    "mean_state_path_error_rms",
    "max_state_path_error_rms",
    "mean_endpoint_error_rms",
    "last_endpoint_error_rms",
    "feature_variance_ratio_gmean",
    "feature_centered_cosine_mean",
)


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must be NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("checkpoint name cannot be empty")
    return name, Path(raw_path).expanduser()


def parse_positive_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("rollout step counts must be unique")
    return values


def shifted_time_grid(
    start_time: float,
    *,
    num_steps: int,
    shift: float,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the official shifted Euler grid restricted to ``start_time -> 0``."""

    if not 0.0 < float(start_time) <= 1.0:
        raise ValueError("start_time must be in (0, 1]")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if shift <= 0:
        raise ValueError("shift must be positive")
    start_raw = float(start_time) / (
        float(shift) - (float(shift) - 1.0) * float(start_time)
    )
    raw = torch.linspace(
        start_raw,
        0.0,
        num_steps + 1,
        device=device,
        dtype=dtype,
    )
    return float(shift) * raw / (1.0 + (float(shift) - 1.0) * raw)


@torch.no_grad()
def velocity_endpoint_rollout(
    *,
    model: torch.nn.Module,
    initial_state: torch.Tensor,
    clean: torch.Tensor,
    noise: torch.Tensor,
    labels: torch.Tensor,
    start_time: float,
    num_steps: int,
    time_shift: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Integrate an old-RAE velocity model from a paired state to time zero."""

    state = initial_state.clone()
    grid = shifted_time_grid(
        start_time,
        num_steps=num_steps,
        shift=time_shift,
        device=state.device,
        dtype=torch.float32,
    )
    state_path_errors = []
    endpoint_errors = []
    for step_index in range(num_steps):
        time_value = grid[step_index]
        next_time = grid[step_index + 1]
        time = torch.full(
            (state.shape[0],),
            float(time_value),
            device=state.device,
            dtype=torch.float32,
        )
        exact_state = (1.0 - time_value) * clean + time_value * noise
        velocity = model(state, time, y=labels).float()
        endpoint = flow_clean_estimate(state.float(), velocity, time)
        state_path_errors.append(tensor_rms(state.float() - exact_state.float()))
        endpoint_errors.append(tensor_rms(endpoint - clean.float()))
        state = state + (next_time - time_value) * velocity
    return state.float(), {
        "state_path_error_rms": torch.stack(state_path_errors),
        "endpoint_error_rms": torch.stack(endpoint_errors),
    }


def resolve_rae_paths(config: Any) -> None:
    params = config.stage_1.params
    for name in (
        "decoder_config_path",
        "pretrained_decoder_path",
        "normalization_stat_path",
    ):
        value = params.get(name)
        if value is not None and not Path(str(value)).is_absolute():
            params[name] = str(RAE_ROOT / str(value))


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def model_state_from_checkpoint(
    state: Any,
    *,
    state_key: str,
) -> tuple[dict[str, torch.Tensor], str]:
    """Extract wrapped model/EMA weights or an already materialized state dict."""

    if not isinstance(state, dict) or not state:
        raise TypeError("checkpoint must be a non-empty mapping")
    if state_key == "auto":
        for candidate_key in ("ema", "model", "state_dict"):
            candidate = state.get(candidate_key)
            if isinstance(candidate, dict) and candidate:
                state = candidate
                state_key = candidate_key
                break
        else:
            state_key = "raw"
    elif state_key != "raw":
        candidate = state.get(state_key)
        if not isinstance(candidate, dict) or not candidate:
            raise KeyError(f"checkpoint lacks non-empty {state_key!r} weights")
        state = candidate
    if not all(isinstance(key, str) for key in state):
        raise ValueError("model state dict contains non-string keys")
    if not all(torch.is_tensor(value) for value in state.values()):
        raise ValueError("model state dict contains non-tensor values")
    if all(key.startswith("module.") for key in state):
        state = {
            key.removeprefix("module."): value
            for key, value in state.items()
        }
    return state, state_key


def _fixed_noise(
    clean: torch.Tensor,
    *,
    sample_indices: list[int],
    ratio_index: int,
    seed: int,
) -> torch.Tensor:
    values = []
    for sample_index in sample_indices:
        generator = torch.Generator(device="cpu").manual_seed(
            int(seed) + 10_000 * int(sample_index) + int(ratio_index)
        )
        values.append(
            torch.randn(
                clean.shape[1:],
                generator=generator,
                dtype=torch.float32,
            )
        )
    return torch.stack(values).to(clean.device)


def paired_summary(
    raw: pd.DataFrame,
    *,
    reference: str,
) -> pd.DataFrame:
    """Summarize paired checkpoint differences with normal-approximation CIs."""

    if reference not in set(raw["checkpoint"]):
        raise ValueError(f"reference checkpoint {reference!r} is absent")
    keys = (
        "sample_index",
        "data_index",
        "label",
        "noise_to_signal_ratio",
        "start_time",
        "num_steps",
    )
    reference_frame = raw[raw["checkpoint"] == reference]
    rows = []
    for treatment in sorted(set(raw["checkpoint"]) - {reference}):
        treatment_frame = raw[raw["checkpoint"] == treatment]
        merged = reference_frame.merge(
            treatment_frame,
            on=list(keys),
            suffixes=("_reference", "_treatment"),
            validate="one_to_one",
        )
        if len(merged) != len(reference_frame):
            raise RuntimeError("paired checkpoint rows are incomplete")
        for (ratio, num_steps), group in merged.groupby(
            ["noise_to_signal_ratio", "num_steps"],
            sort=True,
        ):
            for metric in PAIR_METRICS:
                reference_values = group[f"{metric}_reference"].to_numpy(dtype=float)
                treatment_values = group[f"{metric}_treatment"].to_numpy(dtype=float)
                delta = treatment_values - reference_values
                count = len(delta)
                standard_error = (
                    float(np.std(delta, ddof=1) / math.sqrt(count))
                    if count > 1
                    else float("nan")
                )
                mean_delta = float(np.mean(delta))
                reference_mean = float(np.mean(reference_values))
                rows.append(
                    {
                        "reference": reference,
                        "treatment": treatment,
                        "noise_to_signal_ratio": float(ratio),
                        "num_steps": int(num_steps),
                        "metric": metric,
                        "sample_count": count,
                        "reference_mean": reference_mean,
                        "treatment_mean": float(np.mean(treatment_values)),
                        "mean_delta": mean_delta,
                        "relative_change": (
                            mean_delta / reference_mean
                            if abs(reference_mean) > 1e-30
                            else float("nan")
                        ),
                        "standard_error": standard_error,
                        "ci95_low": mean_delta - 1.96 * standard_error,
                        "ci95_high": mean_delta + 1.96 * standard_error,
                        "fraction_treatment_lower": float(np.mean(delta < 0)),
                    }
                )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=parse_named_path,
        required=True,
    )
    parser.add_argument("--reference-checkpoint")
    parser.add_argument(
        "--state-key",
        choices=("auto", "model", "ema", "raw"),
        default="auto",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--rollout-steps",
        type=parse_positive_ints,
        default=(1, 4, 16),
    )
    parser.add_argument(
        "--noise-ratio",
        action="append",
        type=float,
        dest="noise_ratios",
    )
    parser.add_argument("--seed", type=int, default=20260731)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples <= 0 or args.batch_size <= 0:
        raise ValueError("--samples and --batch-size must be positive")
    noise_ratios = tuple(args.noise_ratios or (1.0, 3.0))
    if any(ratio <= 0 for ratio in noise_ratios):
        raise ValueError("noise ratios must be positive")
    checkpoint_names = [name for name, _ in args.checkpoint]
    if len(checkpoint_names) != len(set(checkpoint_names)):
        raise ValueError("checkpoint names must be unique")
    reference = args.reference_checkpoint or checkpoint_names[0]
    if reference not in checkpoint_names:
        raise ValueError("--reference-checkpoint is not one of the checkpoints")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.manual_seed(int(args.seed) + rank)
    torch.cuda.manual_seed(int(args.seed) + rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.set_float32_matmul_precision("highest")

    config_path = args.config.expanduser().resolve()
    config = OmegaConf.load(config_path)
    resolve_rae_paths(config)
    lpl_config = OmegaConf.to_container(config.training.strict_lpl, resolve=True)
    transport_config = OmegaConf.to_container(config.transport.params, resolve=True)
    if (
        str(transport_config["path_type"]) != "Linear"
        or str(transport_config["prediction"]) != "velocity"
    ):
        raise ValueError("recursive audit requires linear velocity flow matching")

    rae: RAE = instantiate_from_config(config.stage_1).to(
        device=device,
        dtype=torch.float32,
    )
    rae.requires_grad_(False).eval()
    model = instantiate_from_config(config.stage_2).to(
        device=device,
        dtype=torch.float32,
    )
    model.requires_grad_(False).eval()
    layer_indices = decoder_hidden_indices(
        len(rae.decoder.decoder_layers),
        tuple(float(value) for value in lpl_config["layer_fractions"]),
    )
    layer_weights = (1.0,) * len(layer_indices)
    time_shift = math.sqrt(
        float(config.misc.time_dist_shift_dim)
        / float(config.misc.time_dist_shift_base)
    )

    transform = transforms.Compose(
        [
            transforms.Lambda(lambda image: center_crop_arr(image, 256)),
            transforms.ToTensor(),
        ]
    )
    dataset = ParquetImageNetDataset(
        args.data_path.expanduser().resolve(),
        split="validation",
        transform=transform,
    )
    if args.samples > len(dataset):
        raise ValueError("--samples exceeds validation dataset size")
    local_indices = list(range(rank, int(args.samples), world_size))
    clean_chunks = []
    label_values = []
    with torch.inference_mode():
        for start in range(0, len(local_indices), int(args.batch_size)):
            batch_indices = local_indices[start : start + int(args.batch_size)]
            samples = [dataset[index] for index in batch_indices]
            images = torch.stack([image for image, _ in samples]).to(device)
            clean_chunks.append(rae.encode(images).float().cpu())
            label_values.extend(int(label) for _, label in samples)
    clean_cache = torch.cat(clean_chunks)
    label_cache = torch.tensor(label_values, dtype=torch.long)

    checkpoint_metadata = []
    local_rows = []
    for checkpoint_name, raw_checkpoint_path in args.checkpoint:
        checkpoint_path = raw_checkpoint_path.expanduser().resolve()
        state = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        model_state, resolved_state_key = model_state_from_checkpoint(
            state,
            state_key=args.state_key,
        )
        model.load_state_dict(model_state, strict=True)
        model.eval()
        if rank == 0:
            checkpoint_metadata.append(
                {
                    "name": checkpoint_name,
                    "path": str(checkpoint_path),
                    "sha256": file_sha256(checkpoint_path),
                    "state_key": resolved_state_key,
                    "step": int(state.get("step", -1)),
                    "objective": state.get("objective"),
                }
            )
        del state

        for cache_start in range(0, len(local_indices), int(args.batch_size)):
            cache_stop = min(
                cache_start + int(args.batch_size),
                len(local_indices),
            )
            batch_indices = local_indices[cache_start:cache_stop]
            clean = clean_cache[cache_start:cache_stop].to(device)
            labels = label_cache[cache_start:cache_stop].to(device)
            with torch.inference_mode():
                target_features = tuple(
                    feature.float()
                    for feature in decoder_feature_pyramid(
                        rae,
                        clean,
                        layer_indices=layer_indices,
                    )
                )

            for ratio_index, ratio in enumerate(noise_ratios):
                noise = _fixed_noise(
                    clean,
                    sample_indices=batch_indices,
                    ratio_index=ratio_index,
                    seed=int(args.seed),
                )
                start_time = float(ratio / (1.0 + ratio))
                initial = (1.0 - start_time) * clean + start_time * noise
                for num_steps in args.rollout_steps:
                    final_state, path = velocity_endpoint_rollout(
                        model=model,
                        initial_state=initial,
                        clean=clean,
                        noise=noise,
                        labels=labels,
                        start_time=start_time,
                        num_steps=int(num_steps),
                        time_shift=time_shift,
                    )
                    with torch.inference_mode():
                        final_features = tuple(
                            feature.float()
                            for feature in decoder_feature_pyramid(
                                rae,
                                final_state,
                                layer_indices=layer_indices,
                            )
                        )
                    final_raw, details = decoder_feature_objective_per_sample(
                        "raw",
                        target_features,
                        final_features,
                        layer_weights=layer_weights,
                        outlier_quantile=float(lpl_config["outlier_quantile"]),
                        outlier_opening=int(lpl_config["outlier_opening"]),
                        outlier_closing=int(lpl_config["outlier_closing"]),
                        eps=float(lpl_config["normalization_eps"]),
                    )
                    variance_ratio = details[
                        "prediction_over_target_variance_layers"
                    ].clamp_min(1e-30).log().mean(1).exp()
                    centered_cosine = details["centered_cosine_layers"].mean(1)
                    latent_error = tensor_rms(final_state - clean)
                    state_path = path["state_path_error_rms"]
                    endpoint_path = path["endpoint_error_rms"]
                    for local_offset, sample_index in enumerate(batch_indices):
                        local_rows.append(
                            {
                                "checkpoint": checkpoint_name,
                                "sample_index": int(sample_index),
                                "data_index": int(sample_index),
                                "label": int(labels[local_offset]),
                                "noise_to_signal_ratio": float(ratio),
                                "start_time": start_time,
                                "num_steps": int(num_steps),
                                "final_raw_decoder_feature_loss": float(
                                    final_raw[local_offset]
                                ),
                                "final_latent_error_rms": float(
                                    latent_error[local_offset]
                                ),
                                "mean_state_path_error_rms": float(
                                    state_path[:, local_offset].mean()
                                ),
                                "max_state_path_error_rms": float(
                                    state_path[:, local_offset].max()
                                ),
                                "mean_endpoint_error_rms": float(
                                    endpoint_path[:, local_offset].mean()
                                ),
                                "last_endpoint_error_rms": float(
                                    endpoint_path[-1, local_offset]
                                ),
                                "feature_variance_ratio_gmean": float(
                                    variance_ratio[local_offset]
                                ),
                                "feature_centered_cosine_mean": float(
                                    centered_cosine[local_offset]
                                ),
                            }
                        )

    gathered_rows = [None] * world_size if rank == 0 else None
    dist.gather_object(local_rows, gathered_rows, dst=0)
    if rank == 0:
        rows = [row for rank_rows in gathered_rows for row in rank_rows]
        raw = pd.DataFrame(rows).sort_values(
            [
                "checkpoint",
                "noise_to_signal_ratio",
                "num_steps",
                "sample_index",
            ]
        )
        summary = (
            raw.groupby(
                [
                    "checkpoint",
                    "noise_to_signal_ratio",
                    "start_time",
                    "num_steps",
                ],
                as_index=False,
            )
            .mean(numeric_only=True)
            .sort_values(
                ["checkpoint", "noise_to_signal_ratio", "num_steps"]
            )
        )
        comparisons = paired_summary(raw, reference=reference)
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        raw.to_csv(output_dir / "rollout_raw.csv", index=False)
        summary.to_csv(output_dir / "rollout_summary.csv", index=False)
        comparisons.to_csv(output_dir / "paired_comparisons.csv", index=False)

        figure, axes = plt.subplots(
            1,
            3,
            figsize=(18, 5.5),
            constrained_layout=True,
        )
        plotted_metrics = (
            ("final_raw_decoder_feature_loss", "Final raw decoder-feature loss"),
            ("final_latent_error_rms", "Final latent RMS error"),
            ("feature_variance_ratio_gmean", "Decoder-feature variance ratio"),
        )
        for (checkpoint, ratio), frame in summary.groupby(
            ["checkpoint", "noise_to_signal_ratio"]
        ):
            label = f"{checkpoint}, ratio={ratio:g}"
            for axis, (metric, _) in zip(axes, plotted_metrics, strict=True):
                axis.plot(
                    frame["num_steps"],
                    frame[metric],
                    marker="o",
                    label=label,
                )
        for axis, (_, title) in zip(axes, plotted_metrics, strict=True):
            axis.set_xscale("log", base=2)
            axis.set_xticks(list(args.rollout_steps))
            axis.set_xticklabels([str(value) for value in args.rollout_steps])
            axis.set_xlabel("recursive endpoint queries")
            axis.set_title(title)
            axis.grid(alpha=0.25)
        axes[-1].legend(
            fontsize=8,
            ncol=1,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )
        figure.savefig(output_dir / "rollout_curves.png", dpi=180)
        plt.close(figure)

        manifest = {
            "format_version": 1,
            "scope": "old_rae_lpl_recursive_rollout_audit",
            "config": str(config_path),
            "config_sha256": file_sha256(config_path),
            "checkpoints": checkpoint_metadata,
            "reference_checkpoint": reference,
            "data_path": str(args.data_path.expanduser().resolve()),
            "split": "validation",
            "samples": int(args.samples),
            "world_size": world_size,
            "batch_size_per_rank": int(args.batch_size),
            "sample_indices": list(range(int(args.samples))),
            "noise_ratios": list(noise_ratios),
            "rollout_steps": list(args.rollout_steps),
            "time_shift": time_shift,
            "seed": int(args.seed),
            "precision": "fp32",
            "tf32": False,
            "validation_used_for_training": False,
            "interpretation_limit": (
                "Each rollout starts from a paired validation latent/noise "
                "interpolation and integrates only start_time->0. It is a "
                "controlled recursive state-shift audit, not unconditional FID."
            ),
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(summary.to_string(index=False))
        print("\nPaired comparisons")
        print(comparisons.to_string(index=False))
        print(output_dir)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
