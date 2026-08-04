"""Separate baseline-flow propagation from state-dependent IG feedback.

``recursive`` recomputes ``full - base`` on every guided state.  ``replay``
reuses the gap recorded on the unguided trajectory while still recomputing the
full field on the current state.  Their paired difference isolates feedback
through the state dependence of the internal-guidance direction.
"""

from __future__ import annotations

import argparse
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


ROOT = Path(__file__).resolve().parents[1]
RAEV2_SRC = ROOT / "external" / "RAEv2" / "src"
for path in (RAEV2_SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.raev2_stage1_compat import install_raev2_decoder_config_compat  # noqa: E402
from experiments.raev2_training_core import (  # noqa: E402
    file_sha256,
    split_internal_guidance_output,
    validate_full_stage2_checkpoint,
)
from experiments.run_raev2_distribution_auc import build_requested_labels, load_config  # noqa: E402
from experiments.run_raev2_ig_impulse_response import (  # noqa: E402
    _atomic_json,
    _open_memmap,
    autocast_context,
    bootstrap_mean_interval,
    deterministic_noise,
    euler_x_prediction_step,
    official_baseline_endpoint,
    official_shifted_solver_grid,
)
from utils.model_utils import instantiate_from_config  # noqa: E402


PROTOCOL = "raev2_ig_replay_response_v1"
CONDITIONS = ("baseline", "recursive_pos", "recursive_neg", "replay_pos", "replay_neg")
STAT_FIELDS = ("unit_injected_energy", "actual_injected_energy", "unit_coherent_rms")


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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--per-rank-batch", type=int, default=1)
    parser.add_argument("--start-step", type=int, required=True)
    parser.add_argument("--end-step", type=int, required=True)
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--state-key", choices=("ema", "model"), default="ema")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--log-every-samples", type=int, default=2)
    parser.add_argument(
        "--dino-ckpt-dir",
        type=Path,
        default=Path("/home/zhoushunyu/data/eqvae/models/RAEv2/encoders/dinov3"),
    )
    parser.add_argument(
        "--dino-repo-dir",
        type=Path,
        default=Path("/home/zhoushunyu/data/eqvae/models/RAEv2/dinov3_repo"),
    )
    return parser.parse_args()


def choose_gap(
    current_gap: torch.Tensor,
    replay_gap: torch.Tensor,
    modes: tuple[str, ...],
) -> torch.Tensor:
    if current_gap.shape != replay_gap.shape or len(current_gap) != len(modes):
        raise ValueError("gap tensors and modes do not align")
    replay_mask = torch.tensor(
        [mode == "replay" for mode in modes],
        device=current_gap.device,
        dtype=torch.bool,
    ).reshape((len(modes),) + (1,) * (current_gap.ndim - 1))
    if any(mode not in ("recursive", "replay") for mode in modes):
        raise ValueError("modes must be recursive or replay")
    return torch.where(replay_mask, replay_gap, current_gap)


def _baseline_rollout(
    *,
    model: torch.nn.Module,
    noise: torch.Tensor,
    labels: torch.Tensor,
    grid: torch.Tensor,
    t_eps: float,
    precision: str,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    state = noise.float()
    gaps: list[torch.Tensor] = []
    with torch.inference_mode():
        for step in range(len(grid) - 1):
            time = float(grid[step])
            h = time - float(grid[step + 1])
            time_batch = torch.full((len(state),), time, device=state.device)
            with autocast_context(precision):
                output = model(state, time_batch, context=labels, attn_mask=None)
            full, base = split_internal_guidance_output(output)
            if base is None:
                raise RuntimeError("configured checkpoint has no IG base head")
            full = full.float()
            gaps.append((full - base.float()).detach())
            state = euler_x_prediction_step(
                state,
                full,
                time=time,
                step_size=h,
                t_eps=t_eps,
            )
    return state.float(), tuple(gaps)


def simulate_replay_recursive(
    *,
    model: torch.nn.Module,
    noise: torch.Tensor,
    labels: torch.Tensor,
    baseline_gaps: tuple[torch.Tensor, ...],
    grid: torch.Tensor,
    start_step: int,
    end_step: int,
    gamma: float,
    t_eps: float,
    precision: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    modes = ("recursive", "recursive", "replay", "replay")
    signs = torch.tensor((gamma, -gamma, gamma, -gamma), device=noise.device)
    branch_count = len(modes)
    batch_size = len(noise)
    state = (
        noise.unsqueeze(0)
        .expand(branch_count, *noise.shape)
        .reshape(branch_count * batch_size, *noise.shape[1:])
        .contiguous()
    )
    contexts = labels.unsqueeze(0).expand(branch_count, batch_size).reshape(-1).contiguous()
    unit_energy = torch.zeros(branch_count, batch_size, device=noise.device, dtype=torch.float64)
    coherent = torch.zeros(
        branch_count, batch_size, *noise.shape[1:], device=noise.device, dtype=torch.float32
    )
    with torch.inference_mode():
        for step in range(len(grid) - 1):
            time = float(grid[step])
            h = time - float(grid[step + 1])
            time_batch = torch.full((len(state),), time, device=state.device)
            with autocast_context(precision):
                output = model(state, time_batch, context=contexts, attn_mask=None)
            full, base = split_internal_guidance_output(output)
            if base is None:
                raise RuntimeError("configured checkpoint has no IG base head")
            full = full.float().reshape(branch_count, batch_size, *state.shape[1:])
            current_gap = full - base.float().reshape_as(full)
            recorded = baseline_gaps[step].unsqueeze(0).expand_as(current_gap)
            selected = choose_gap(current_gap, recorded, modes)
            active = start_step <= step < end_step
            coefficients = signs if active else torch.zeros_like(signs)
            guided = full + coefficients.reshape(
                (branch_count, 1) + (1,) * (full.ndim - 2)
            ) * selected
            state = euler_x_prediction_step(
                state.reshape_as(full),
                guided,
                time=time,
                step_size=h,
                t_eps=t_eps,
            ).reshape(branch_count * batch_size, *state.shape[1:])
            if active:
                unit_impulse = (h / max(time, t_eps)) * selected
                unit_energy += unit_impulse.flatten(2).square().mean(2).double()
                coherent += unit_impulse
    coherent_rms = coherent.flatten(2).square().mean(2).sqrt().double()
    actual_energy = signs.abs().double().square()[:, None] * unit_energy
    stats = torch.stack((unit_energy, actual_energy, coherent_rms), dim=2)
    return state.reshape(branch_count, batch_size, *state.shape[1:]).float(), stats


def feedback_metrics(
    baseline: np.ndarray,
    recursive_pos: np.ndarray,
    recursive_neg: np.ndarray,
    replay_pos: np.ndarray,
    replay_neg: np.ndarray,
    *,
    gamma: float,
) -> dict[str, np.ndarray]:
    arrays = tuple(np.asarray(value, dtype=np.float64) for value in (
        baseline, recursive_pos, recursive_neg, replay_pos, replay_neg
    ))
    if len({value.shape for value in arrays}) != 1:
        raise ValueError("all paired endpoint arrays must align")
    flat = tuple(value.reshape(len(value), -1) for value in arrays)
    base, rec_pos, rec_neg, rep_pos, rep_neg = flat
    rec_odd = 0.5 * (rec_pos - rec_neg)
    rep_odd = 0.5 * (rep_pos - rep_neg)
    difference = rec_odd - rep_odd

    def rms(value: np.ndarray) -> np.ndarray:
        return np.sqrt(np.mean(np.square(value), axis=1))

    rec_norm = rms(rec_odd)
    rep_norm = rms(rep_odd)
    diff_norm = rms(difference)
    cosine = np.sum(rec_odd * rep_odd, axis=1) / np.maximum(
        np.linalg.norm(rec_odd, axis=1) * np.linalg.norm(rep_odd, axis=1), 1e-30
    )
    return {
        "recursive_response_per_gamma": rec_norm / gamma,
        "replay_response_per_gamma": rep_norm / gamma,
        "feedback_difference_per_gamma": diff_norm / gamma,
        "feedback_fraction": diff_norm / np.maximum(rec_norm, 1e-30),
        "recursive_replay_cosine": cosine,
        "recursive_even_over_odd_sample": rms(0.5 * (rec_pos + rec_neg) - base)
        / np.maximum(rec_norm, 1e-30),
        "replay_even_over_odd_sample": rms(0.5 * (rep_pos + rep_neg) - base)
        / np.maximum(rep_norm, 1e-30),
    }


def _load_global(
    output_dir: Path,
    *,
    condition_index: int,
    samples: int,
    world_size: int,
) -> np.ndarray:
    result: np.ndarray | None = None
    for rank in range(world_size):
        ids = np.arange(rank, samples, world_size, dtype=np.int64)
        shard = np.load(output_dir / f"endpoints_rank{rank:02d}.npy", mmap_mode="r")
        local = np.asarray(shard[condition_index], dtype=np.float32)
        if result is None:
            result = np.empty((samples, *local.shape[1:]), dtype=np.float32)
        result[ids] = local
    if result is None:
        raise RuntimeError("no endpoint shards")
    return result


def analyze(
    output_dir: Path,
    *,
    samples: int,
    world_size: int,
    gamma: float,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    values = [
        _load_global(output_dir, condition_index=index, samples=samples, world_size=world_size)
        for index in range(len(CONDITIONS))
    ]
    metrics = feedback_metrics(*values, gamma=gamma)
    rows = []
    for index, (name, sample_values) in enumerate(metrics.items()):
        low, high = bootstrap_mean_interval(
            sample_values, repeats=repeats, seed=seed + 101 * index
        )
        rows.append(
            {
                "metric": name,
                "mean": float(sample_values.mean()),
                "std": float(sample_values.std(ddof=1)),
                "ci_low": low,
                "ci_high": high,
                "median": float(np.median(sample_values)),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "feedback_response.csv", index=False)
    figure, axis = plt.subplots(figsize=(10, 5.5))
    selected = frame[frame.metric.isin((
        "recursive_response_per_gamma",
        "replay_response_per_gamma",
        "feedback_difference_per_gamma",
    ))]
    positions = np.arange(len(selected))
    axis.bar(positions, selected["mean"], color=("#2563EB", "#0F766E", "#B42318"))
    axis.set_xticks(positions, selected["metric"], rotation=20, ha="right")
    axis.set_ylabel("endpoint RMS / gamma")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "feedback_response.png", dpi=180)
    plt.close(figure)
    print(frame.to_string(index=False), flush=True)
    return frame


def main() -> None:
    install_raev2_decoder_config_compat()
    args = parse_args()
    if args.samples <= 0 or args.per_rank_batch <= 0 or args.bootstrap_repeats <= 0:
        raise ValueError("sample, batch and bootstrap counts must be positive")
    if not math.isfinite(args.gamma) or args.gamma <= 0:
        raise ValueError("gamma must be finite and positive")
    os.environ["DINOV3_CKPT_DIR"] = str(args.dino_ckpt_dir.expanduser().resolve())
    os.environ["DINOV3_REPO_DIR"] = str(args.dino_repo_dir.expanduser().resolve())
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.backends.cuda.matmul.allow_tf32 = args.precision != "fp32"
    torch.backends.cudnn.allow_tf32 = args.precision != "fp32"

    output_dir = args.output_dir.expanduser().resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    config = load_config(args.config.expanduser().resolve())
    latent_size = tuple(int(value) for value in config.misc.latent_size)
    shift = math.sqrt(
        (config.misc.time_dist_shift_dim or math.prod(latent_size))
        / config.misc.time_dist_shift_base
    )
    grid = official_shifted_solver_grid(int(config.sampler.num_steps), shift)
    if not 0 <= args.start_step < args.end_step <= len(grid) - 1:
        raise ValueError("step range must be a nonempty subset of the solver")
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint_hash = file_sha256(checkpoint_path) if rank == 0 else ""
    objects = [checkpoint_hash]
    dist.broadcast_object_list(objects, src=0)
    manifest = {
        "protocol": PROTOCOL,
        "status": "running",
        "training": False,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": objects[0],
        "state_key": args.state_key,
        "samples": args.samples,
        "seed": args.seed,
        "world_size": world_size,
        "precision": args.precision,
        "start_step": args.start_step,
        "end_step": args.end_step,
        "start_time": float(grid[args.start_step]),
        "end_time": float(grid[args.end_step]),
        "gamma": args.gamma,
        "conditions": list(CONDITIONS),
        "latent_size": list(latent_size),
        "same_noise_and_labels": True,
        "replay_definition": "baseline gap replayed; full field recomputed on current state",
    }
    manifest_path = output_dir / "manifest.json"
    if rank == 0:
        if manifest_path.is_file():
            current = json.loads(manifest_path.read_text(encoding="utf-8"))
            keys = tuple(key for key in manifest if key != "status")
            changed = [key for key in keys if current.get(key) != manifest.get(key)]
            if changed:
                raise RuntimeError(f"cannot resume changed replay protocol: {changed}")
        else:
            _atomic_json(manifest_path, manifest)
            labels = build_requested_labels(args.samples, int(config.misc.num_classes))
            np.savez_compressed(
                output_dir / "sample_protocol.npz",
                sample_ids=np.arange(args.samples),
                labels=labels,
            )
    dist.barrier()
    labels = np.load(output_dir / "sample_protocol.npz")["labels"].astype(np.int64)
    local_ids = np.arange(rank, args.samples, world_size, dtype=np.int64)
    model = instantiate_from_config(config.stage_2).to(device).eval().requires_grad_(False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    validate_full_stage2_checkpoint(checkpoint)
    model.load_state_dict(checkpoint[args.state_key], strict=True)
    del checkpoint

    endpoints = _open_memmap(
        output_dir / f"endpoints_rank{rank:02d}.npy",
        shape=(len(CONDITIONS), len(local_ids), *latent_size),
        dtype=np.float32,
    )
    stats = _open_memmap(
        output_dir / f"injection_stats_rank{rank:02d}.npy",
        shape=(4, len(local_ids), len(STAT_FIELDS)),
        dtype=np.float64,
    )
    progress_path = output_dir / f"progress_rank{rank:02d}.npy"
    existed = progress_path.is_file()
    progress = _open_memmap(progress_path, shape=(len(local_ids),), dtype=np.bool_)
    if not existed:
        progress.fill(False)
        progress.flush()
    for start in range(0, len(local_ids), args.per_rank_batch):
        stop = min(start + args.per_rank_batch, len(local_ids))
        if bool(np.asarray(progress[start:stop]).all()):
            continue
        ids = local_ids[start:stop]
        noise = deterministic_noise(ids, latent_size, seed=args.seed).to(device)
        batch_labels = torch.from_numpy(labels[ids]).to(device=device, dtype=torch.long)
        baseline, baseline_gaps = _baseline_rollout(
            model=model,
            noise=noise,
            labels=batch_labels,
            grid=grid,
            t_eps=float(config.transport.t_eps),
            precision=args.precision,
        )
        branches, branch_stats = simulate_replay_recursive(
            model=model,
            noise=noise,
            labels=batch_labels,
            baseline_gaps=baseline_gaps,
            grid=grid,
            start_step=args.start_step,
            end_step=args.end_step,
            gamma=args.gamma,
            t_eps=float(config.transport.t_eps),
            precision=args.precision,
        )
        endpoints[0, start:stop] = baseline.cpu().numpy()
        endpoints[1:, start:stop] = branches.cpu().numpy()
        stats[:, start:stop] = branch_stats.cpu().numpy()
        verification_path = output_dir / f"official_baseline_check_rank{rank:02d}.json"
        if not verification_path.is_file():
            official = official_baseline_endpoint(
                model=model,
                noise=noise,
                labels=batch_labels,
                config=config,
                shift=shift,
                precision=args.precision,
            )
            delta = baseline.double() - official.double()
            check = {
                "rms": float(delta.square().mean().sqrt().cpu()),
                "maximum_absolute": float(delta.abs().max().cpu()),
            }
            _atomic_json(verification_path, check)
            if check["rms"] > 1e-5 or check["maximum_absolute"] > 1e-3:
                raise RuntimeError(f"baseline mismatch: {check}")
        endpoints.flush()
        stats.flush()
        progress[start:stop] = True
        progress.flush()
        if rank == 0 and (stop % args.log_every_samples == 0 or stop == len(local_ids)):
            print(f"[rank 0] local samples {stop}/{len(local_ids)}", flush=True)
    dist.barrier()
    if rank == 0:
        analyze(
            output_dir,
            samples=args.samples,
            world_size=world_size,
            gamma=args.gamma,
            repeats=args.bootstrap_repeats,
            seed=args.seed + 23,
        )
        manifest["status"] = "complete"
        _atomic_json(manifest_path, manifest)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
