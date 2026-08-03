"""Audit whether the official RAEv2 internal direction is locally corrective.

The frozen dual-head model predicts a clean latent with a full head ``F`` and
a base head ``B``.  At held-out forward-noised ImageNet latents this script
measures the exact quadratic geometry of

    F + gamma * (F - B)

against the clean target ``Y``.  It never trains a model and never uses the
validation split to alter a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist


ROOT = Path(__file__).resolve().parents[1]
RAEV2_SRC = ROOT / "external" / "RAEv2" / "src"
for path in (RAEV2_SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.internal_guidance_direction import (  # noqa: E402
    direction_metrics,
    scale_sweep_metrics,
)
from experiments.raev2_stage1_compat import (  # noqa: E402
    install_raev2_decoder_config_compat,
)
from experiments.raev2_training_core import (  # noqa: E402
    DeterministicImageNetParquet,
    file_sha256,
    split_internal_guidance_output,
    validate_full_stage2_checkpoint,
)
from experiments.run_raev2_distribution_auc import (  # noqa: E402
    load_config,
    shifted_solver_grid,
)
from utils.model_utils import instantiate_from_config  # noqa: E402


DEFAULT_TIMES = (0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.95)
DEFAULT_SCALES = (1.0, 1.25, 1.50, 1.78, 2.0)


def parse_float_list(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(not math.isfinite(item) for item in values):
        raise argparse.ArgumentTypeError("expected finite comma-separated floats")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/configs/raev2_strict_lpl_dinov3l_k7.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/home/zhoushunyu/data/eqvae/models/RAEv2/stage2/imagenet/"
            "dinov3l-k7/checkpoint.pt"
        ),
    )
    parser.add_argument("--data-path", type=Path, default=Path("/data/shared/imagenet-1k"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--times", type=parse_float_list, default=DEFAULT_TIMES)
    parser.add_argument("--scales", type=parse_float_list, default=DEFAULT_SCALES)
    parser.add_argument(
        "--state-key",
        action="append",
        choices=("ema", "model"),
        dest="state_keys",
    )
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--log-every-batches", type=int, default=20)
    parser.add_argument(
        "--dino-ckpt-dir",
        type=Path,
        default=Path("/home/zhoushunyu/data/eqvae/models/RAEv2/encoders/dinov3"),
    )
    parser.add_argument("--dino-repo-dir", type=Path)
    return parser.parse_args()


def validate_protocol(args: argparse.Namespace, world_size: int) -> tuple[str, ...]:
    positive = (
        args.samples,
        args.batch_size,
        args.bootstrap_repeats,
        args.log_every_batches,
    )
    if any(int(value) <= 0 for value in positive):
        raise ValueError("sample, batch, bootstrap, and logging counts must be positive")
    if args.samples < world_size:
        raise ValueError("samples must be at least the distributed world size")
    if any(not 0.0 < float(value) < 1.0 for value in args.times):
        raise ValueError("audit times must lie strictly inside (0, 1)")
    if len(set(args.times)) != len(args.times):
        raise ValueError("audit times must not contain duplicates")
    if any(float(value) < 0.0 for value in args.scales):
        raise ValueError("IG scales must be non-negative")
    if 1.0 not in args.scales or len(set(args.scales)) != len(args.scales):
        raise ValueError("scales must be unique and include the full-head scale 1.0")
    state_keys = tuple(args.state_keys or ("ema",))
    if len(set(state_keys)) != len(state_keys):
        raise ValueError("state keys must not contain duplicates")
    return state_keys


def autocast_context(precision: str):
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def nearest_solver_step(
    grid: torch.Tensor,
    time: float,
    *,
    t_eps: float,
) -> dict[str, float | int]:
    values = grid.detach().cpu().double()
    index = int(torch.argmin((values[:-1] - float(time)).abs()).item())
    solver_time = float(values[index])
    next_time = float(values[index + 1])
    step_size = solver_time - next_time
    return {
        "nearest_solver_step": index,
        "nearest_solver_time": solver_time,
        "nearest_next_time": next_time,
        "nearest_step_size": step_size,
        "nearest_h_over_t": step_size / max(solver_time, float(t_eps)),
    }


def _bootstrap_interval(
    values: np.ndarray,
    statistic: Any,
    *,
    repeats: int,
    seed: int,
) -> tuple[float, float]:
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("bootstrap values must be a two-dimensional sample matrix")
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(int(repeats), dtype=np.float64)
    for repeat in range(int(repeats)):
        indices = rng.integers(0, values.shape[0], size=values.shape[0])
        estimates[repeat] = float(statistic(values[indices]))
    low, high = np.quantile(estimates, (0.025, 0.975))
    return float(low), float(high)


def summarize_direction_rows(
    raw: pd.DataFrame,
    *,
    bootstrap_repeats: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_index, ((state_key, time), frame) in enumerate(
        raw.groupby(["state_key", "time"], sort=True)
    ):
        frame = frame.sort_values("sample_id")
        alignment = frame["alignment"].to_numpy(dtype=np.float64)
        direction_sq = frame["direction_mean_square"].to_numpy(dtype=np.float64)
        paired = np.column_stack((alignment, direction_sq))

        def population_gamma(values: np.ndarray) -> float:
            return float(values[:, 0].mean() / max(values[:, 1].mean(), 1e-30))

        gamma_low, gamma_high = _bootstrap_interval(
            paired,
            population_gamma,
            repeats=bootstrap_repeats,
            seed=seed + 1009 * group_index,
        )
        positive = frame["positive_alignment"].to_numpy(dtype=np.float64)[:, None]
        positive_low, positive_high = _bootstrap_interval(
            positive,
            lambda values: float(values.mean()),
            repeats=bootstrap_repeats,
            seed=seed + 2003 * group_index,
        )
        gamma_population = population_gamma(paired)
        row: dict[str, Any] = {
            "state_key": state_key,
            "time": float(time),
            "samples": int(len(frame)),
            "a_mean": float(frame["a_term"].mean()),
            "b_mean": float(frame["direction_mean_square"].mean()),
            "gamma_population": gamma_population,
            "scale_population": 1.0 + gamma_population,
            "gamma_population_ci_low": gamma_low,
            "gamma_population_ci_high": gamma_high,
            "gamma_sample_median": float(frame["gamma_star"].median()),
            "gamma_sample_q25": float(frame["gamma_star"].quantile(0.25)),
            "gamma_sample_q75": float(frame["gamma_star"].quantile(0.75)),
            "positive_alignment_fraction": float(frame["positive_alignment"].mean()),
            "positive_alignment_ci_low": positive_low,
            "positive_alignment_ci_high": positive_high,
            "alignment_cosine_mean": float(frame["alignment_cosine"].mean()),
            "alignment_cosine_median": float(frame["alignment_cosine"].median()),
            "full_mse_mean": float(frame["full_mse"].mean()),
            "base_mse_mean": float(frame["base_mse"].mean()),
            "base_over_full_mse": float(frame["base_mse"].mean() / frame["full_mse"].mean()),
            "oracle_relative_gain_mean": float(frame["oracle_relative_gain"].mean()),
            "direction_over_residual_rms": float(
                frame["direction_rms"].mean() / frame["residual_rms"].mean()
            ),
            "nearest_solver_step": int(frame["nearest_solver_step"].iloc[0]),
            "nearest_solver_time": float(frame["nearest_solver_time"].iloc[0]),
            "nearest_h_over_t": float(frame["nearest_h_over_t"].iloc[0]),
            "direction_velocity_rms_mean": float(frame["direction_velocity_rms"].mean()),
            "direction_step_rms_mean": float(frame["direction_step_rms"].mean()),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["state_key", "time"])


def summarize_scale_rows(
    raw: pd.DataFrame,
    *,
    bootstrap_repeats: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_index, ((state_key, time, scale), frame) in enumerate(
        raw.groupby(["state_key", "time", "scale"], sort=True)
    ):
        gains = frame["gain_over_full"].to_numpy(dtype=np.float64)[:, None]
        low, high = _bootstrap_interval(
            gains,
            lambda values: float(values.mean()),
            repeats=bootstrap_repeats,
            seed=seed + 3001 * group_index,
        )
        rows.append(
            {
                "state_key": state_key,
                "time": float(time),
                "scale": float(scale),
                "gamma": float(scale) - 1.0,
                "samples": int(len(frame)),
                "mse_mean": float(frame["mse"].mean()),
                "gain_over_full_mean": float(frame["gain_over_full"].mean()),
                "gain_over_full_median": float(frame["gain_over_full"].median()),
                "gain_over_full_ci_low": low,
                "gain_over_full_ci_high": high,
                "positive_gain_fraction": float(frame["gain_over_full"].gt(0).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["state_key", "time", "scale"])


def plot_results(direction: pd.DataFrame, scales: pd.DataFrame, output: Path) -> None:
    state_keys = tuple(direction["state_key"].drop_duplicates())
    figure, axes = plt.subplots(2, len(state_keys), figsize=(8 * len(state_keys), 10), squeeze=False)
    for column, state_key in enumerate(state_keys):
        local = direction[direction["state_key"].eq(state_key)].sort_values("time")
        axis = axes[0, column]
        axis.plot(local["time"], local["gamma_population"], "o-", label="population gamma*")
        axis.fill_between(
            local["time"],
            local["gamma_population_ci_low"],
            local["gamma_population_ci_high"],
            alpha=0.25,
        )
        axis.axhline(0.0, color="#222222", linewidth=1)
        axis.axhline(0.78, color="#b42318", linestyle="--", label="official gamma=0.78")
        axis.set(title=f"{state_key}: local optimal extrapolation", xlabel="noise time t", ylabel="gamma*")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)

        sweep = scales[scales["state_key"].eq(state_key)].pivot(
            index="time", columns="scale", values="gain_over_full_mean"
        )
        bound = max(float(np.abs(sweep.to_numpy()).max()), 1e-12)
        image = axes[1, column].imshow(
            sweep.to_numpy(), aspect="auto", origin="lower", cmap="RdBu", vmin=-bound, vmax=bound
        )
        axes[1, column].set_xticks(range(len(sweep.columns)), [f"{value:g}" for value in sweep.columns])
        axes[1, column].set_yticks(range(len(sweep.index)), [f"{value:g}" for value in sweep.index])
        axes[1, column].set(title=f"{state_key}: local MSE gain", xlabel="code scale s", ylabel="noise time t")
        figure.colorbar(image, ax=axes[1, column], label="gain over full")
    figure.suptitle("RAEv2 Dual-Head Local Guideability Audit", fontsize=16)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def selected_indices(dataset_size: int, samples: int, seed: int) -> np.ndarray:
    if samples > dataset_size:
        raise ValueError("requested samples exceed the validation split")
    rng = np.random.default_rng(int(seed))
    return rng.choice(dataset_size, size=int(samples), replace=False).astype(np.int64)


def main() -> None:
    install_raev2_decoder_config_compat()
    args = parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    state_keys = validate_protocol(args, world_size)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    os.environ["DINOV3_CKPT_DIR"] = str(args.dino_ckpt_dir.expanduser().resolve())
    if args.dino_repo_dir is not None:
        os.environ["DINOV3_REPO_DIR"] = str(args.dino_repo_dir.expanduser().resolve())

    output_dir = args.output_dir.expanduser().resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    config = load_config(args.config.expanduser().resolve())
    dataset = DeterministicImageNetParquet(
        args.data_path,
        split="validation",
        image_size=int(config.training.image_size),
        horizontal_flip=False,
    )
    indices = selected_indices(len(dataset), int(args.samples), int(args.seed))
    local_sample_ids = np.arange(rank, int(args.samples), world_size, dtype=np.int64)
    local_indices = indices[local_sample_ids]

    rae = instantiate_from_config(config.stage_1)
    del rae.decoder
    rae = rae.to(device).eval().requires_grad_(False)
    clean_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    data_indices: list[int] = []
    for start in range(0, len(local_indices), int(args.batch_size)):
        items = [dataset[int(index)] for index in local_indices[start : start + int(args.batch_size)]]
        images = torch.stack([item[0] for item in items]).to(device)
        with torch.inference_mode(), autocast_context(args.precision):
            clean_parts.append(rae.encode(images).float().cpu())
        label_parts.append(torch.tensor([item[1] for item in items], dtype=torch.long))
        data_indices.extend(int(item[2]) for item in items)
    clean_all = torch.cat(clean_parts)
    labels_all = torch.cat(label_parts)
    del rae, clean_parts, label_parts
    torch.cuda.empty_cache()

    config.prepare_model_params()
    model = instantiate_from_config(config.stage_2).to(device).eval().requires_grad_(False)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint_sha256 = file_sha256(checkpoint_path) if rank == 0 else ""
    hashes = [checkpoint_sha256]
    dist.broadcast_object_list(hashes, src=0)
    checkpoint_sha256 = hashes[0]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    validate_full_stage2_checkpoint(checkpoint)
    checkpoint_step = int(checkpoint["step"])
    checkpoint_epoch = int(checkpoint["epoch"])

    latent_size = tuple(int(value) for value in config.misc.latent_size)
    shift = math.sqrt(
        (config.misc.time_dist_shift_dim or math.prod(latent_size))
        / config.misc.time_dist_shift_base
    )
    grid = shifted_solver_grid(int(config.sampler.num_steps), shift)
    solver_metadata = {
        float(time): nearest_solver_step(grid, float(time), t_eps=float(config.transport.t_eps))
        for time in args.times
    }

    noise_parts = []
    for sample_id in local_sample_ids.tolist():
        generator = torch.Generator(device="cpu").manual_seed(int(args.seed) + 1_000_003 * sample_id)
        noise_parts.append(torch.randn(latent_size, generator=generator, dtype=torch.float32))
    noise_all = torch.stack(noise_parts)

    direction_rows: list[dict[str, Any]] = []
    scale_rows: list[dict[str, Any]] = []
    total_batches = math.ceil(len(local_sample_ids) / int(args.batch_size))
    for state_key in state_keys:
        model.load_state_dict(checkpoint[state_key], strict=True)
        for batch_index, start in enumerate(range(0, len(local_sample_ids), int(args.batch_size))):
            stop = min(start + int(args.batch_size), len(local_sample_ids))
            clean = clean_all[start:stop].to(device)
            noise = noise_all[start:stop].to(device)
            labels = labels_all[start:stop].to(device)
            batch_sample_ids = local_sample_ids[start:stop]
            for time_value in args.times:
                time = torch.full((len(clean),), float(time_value), device=device)
                time_map = time.reshape((len(clean),) + (1,) * (clean.ndim - 1))
                noisy = (1.0 - time_map) * clean + time_map * noise
                with torch.inference_mode(), autocast_context(args.precision):
                    output = model(noisy, time, context=labels, attn_mask=None)
                full, base = split_internal_guidance_output(output)
                if base is None:
                    raise RuntimeError("configured RAEv2 checkpoint has no base head")
                metrics = direction_metrics(full.float(), base.float(), clean.float())
                sweep = scale_sweep_metrics(full.float(), base.float(), clean.float(), args.scales)
                solver = solver_metadata[float(time_value)]
                for batch_offset, sample_id in enumerate(batch_sample_ids.tolist()):
                    direction_rms = float(metrics["direction_rms"][batch_offset].cpu())
                    direction_rows.append(
                        {
                            "state_key": state_key,
                            "sample_id": int(sample_id),
                            "dataset_index": int(local_indices[start + batch_offset]),
                            "data_index": int(data_indices[start + batch_offset]),
                            "label": int(labels[batch_offset].cpu()),
                            "time": float(time_value),
                            "a_term": -float(metrics["alignment"][batch_offset].cpu()),
                            "alignment": float(metrics["alignment"][batch_offset].cpu()),
                            "direction_mean_square": direction_rms**2,
                            "full_mse": float(metrics["full_mse"][batch_offset].cpu()),
                            "base_mse": float(metrics["base_mse"][batch_offset].cpu()),
                            "direction_rms": direction_rms,
                            "residual_rms": float(metrics["residual_rms"][batch_offset].cpu()),
                            "alignment_cosine": float(metrics["alignment_cosine"][batch_offset].cpu()),
                            "positive_alignment": bool(metrics["positive_alignment"][batch_offset].cpu()),
                            "gamma_star": float(metrics["gamma_star"][batch_offset].cpu()),
                            "scale_star": float(metrics["scale_star"][batch_offset].cpu()),
                            "oracle_relative_gain": float(metrics["oracle_relative_gain"][batch_offset].cpu()),
                            **solver,
                            "direction_velocity_rms": direction_rms / max(float(time_value), float(config.transport.t_eps)),
                            "direction_step_rms": direction_rms * float(solver["nearest_h_over_t"]),
                        }
                    )
                    for scale_index, scale in enumerate(args.scales):
                        scale_rows.append(
                            {
                                "state_key": state_key,
                                "sample_id": int(sample_id),
                                "dataset_index": int(local_indices[start + batch_offset]),
                                "time": float(time_value),
                                "scale": float(scale),
                                "mse": float(sweep["mse"][scale_index, batch_offset].cpu()),
                                "gain_over_full": float(sweep["gain_over_full"][scale_index, batch_offset].cpu()),
                            }
                        )
            if rank == 0 and (
                (batch_index + 1) % int(args.log_every_batches) == 0
                or batch_index + 1 == total_batches
            ):
                print(f"[{state_key}] batches {batch_index + 1}/{total_batches}", flush=True)
        dist.barrier()
    del checkpoint

    gathered_direction = [None] * world_size if rank == 0 else None
    gathered_scales = [None] * world_size if rank == 0 else None
    dist.gather_object(direction_rows, gathered_direction, dst=0)
    dist.gather_object(scale_rows, gathered_scales, dst=0)
    if rank == 0:
        raw_direction = pd.DataFrame([row for rows in gathered_direction for row in rows])
        raw_scales = pd.DataFrame([row for rows in gathered_scales for row in rows])
        direction_summary = summarize_direction_rows(
            raw_direction,
            bootstrap_repeats=int(args.bootstrap_repeats),
            seed=int(args.seed) + 71,
        )
        scale_summary = summarize_scale_rows(
            raw_scales,
            bootstrap_repeats=int(args.bootstrap_repeats),
            seed=int(args.seed) + 97,
        )
        raw_direction.to_csv(output_dir / "direction_raw.csv", index=False)
        raw_scales.to_csv(output_dir / "scale_sweep_raw.csv", index=False)
        direction_summary.to_csv(output_dir / "direction_summary.csv", index=False)
        scale_summary.to_csv(output_dir / "scale_sweep_summary.csv", index=False)
        plot_results(direction_summary, scale_summary, output_dir / "direction_audit.png")
        manifest = {
            "format_version": 1,
            "scope": "heldout_forward_noised_raev2_dual_head_local_guideability",
            "training": False,
            "validation_used_to_modify_checkpoint": False,
            "config": str(args.config.expanduser().resolve()),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_step": checkpoint_step,
            "checkpoint_epoch": checkpoint_epoch,
            "state_keys": list(state_keys),
            "data_path": str(args.data_path.expanduser().resolve()),
            "split": "validation",
            "samples": int(args.samples),
            "dataset_indices": indices.tolist(),
            "times": list(args.times),
            "scales": list(args.scales),
            "scale_convention": "base + s * (full - base); gamma = s - 1",
            "official_gamma": float(config.guidance.ig.scale) - 1.0,
            "official_interval": [float(config.guidance.ig.t_min), float(config.guidance.ig.t_max)],
            "precision": args.precision,
            "seed": int(args.seed),
            "bootstrap_repeats": int(args.bootstrap_repeats),
            "world_size": int(world_size),
            "same_images_and_noise_across_times_and_state_keys": True,
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(direction_summary.to_string(index=False), flush=True)
        print(scale_summary.to_string(index=False), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
