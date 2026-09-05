#!/usr/bin/env python3
"""Audit RAEv2 strong/base exponential-retiming defects on an IG rollout."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.raev2_pfr_retiming import (
    clean_to_velocity,
    data_odds,
    shared_retiming_revision,
)
from experiments.sample_raev2_pfr_retiming import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    load_config,
    shifted_time_grid,
)
from experiments.raev2_stage1_compat import install_raev2_decoder_config_compat
from experiments.raev2_training_core import file_sha256
from utils.model_utils import instantiate_from_config


REQUESTED_TIMES = (0.95, 0.8, 0.6, 0.4, 0.2, 0.14)


def _flatten(value: torch.Tensor) -> torch.Tensor:
    return value.float().flatten(1)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_flat = _flatten(left)
    right_flat = _flatten(right)
    denominator = left_flat.norm(dim=1) * right_flat.norm(dim=1)
    return (left_flat * right_flat).sum(dim=1) / denominator.clamp_min(1e-12)


def _rms(value: torch.Tensor) -> torch.Tensor:
    return _flatten(value).square().mean(dim=1).sqrt()


def _projection_fraction(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    source_flat = _flatten(source)
    reference_flat = _flatten(reference)
    coefficient = (source_flat * reference_flat).sum(dim=1) / reference_flat.square().sum(
        dim=1
    ).clamp_min(1e-12)
    projected = coefficient[:, None] * reference_flat
    return projected.square().sum(dim=1) / source_flat.square().sum(dim=1).clamp_min(
        1e-12
    )


def _remove_parallel(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    source_flat = _flatten(source)
    reference_flat = _flatten(reference)
    coefficient = (source_flat * reference_flat).sum(
        dim=1, keepdim=True
    ) / reference_flat.square().sum(dim=1, keepdim=True).clamp_min(1e-12)
    return (source_flat - coefficient * reference_flat).reshape_as(source)


def nearest_indices(grid: torch.Tensor) -> dict[int, float]:
    result = {}
    for requested in REQUESTED_TIMES:
        index = int(torch.argmin((grid[:-1] - requested).abs()).item())
        result[index] = requested
    return result


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--horizon", type=float, default=1.0 / 32.0)
    parser.add_argument("--guidance-scale", type=float, default=1.78)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    args = parser.parse_args()
    if args.samples <= 0 or args.batch_size <= 0 or args.horizon <= 0.0:
        raise ValueError("samples, batch size, and horizon must be positive")
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    install_raev2_decoder_config_compat()
    config = load_config(args.config.expanduser().resolve())
    checkpoint_path = args.checkpoint.expanduser().resolve()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    allow_tf32 = args.precision != "fp32"
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    model = instantiate_from_config(config.stage_2).to(device).eval().requires_grad_(False)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    model.load_state_dict(checkpoint["ema"], strict=True)
    checkpoint_step = int(checkpoint.get("step", 0))
    del checkpoint

    shift = math.sqrt(
        (config.misc.time_dist_shift_dim or math.prod(config.misc.latent_size))
        / config.misc.time_dist_shift_base
    )
    grid = shifted_time_grid(config.sampler.num_steps, shift, device)
    selected = nearest_indices(grid)
    collected: dict[int, dict[str, list[float]]] = {
        index: {} for index in selected
    }
    generator = torch.Generator(device=device).manual_seed(args.seed)
    t_floor = float(config.transport.t_eps)
    ig_min = float(config.guidance.ig.t_min)
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.precision == "bf16"
        else __import__("contextlib").nullcontext()
    )
    with torch.inference_mode(), autocast:
        for start in range(0, args.samples, args.batch_size):
            batch = min(args.batch_size, args.samples - start)
            state = torch.randn(
                batch,
                *config.misc.latent_size,
                generator=generator,
                device=device,
            )
            labels = torch.arange(start, start + batch, device=device) % 1000
            kwargs = {"context": labels, "attn_mask": None}
            for index in range(len(grid) - 1):
                current = float(grid[index].item())
                following = float(grid[index + 1].item())
                times = torch.full((batch,), current, device=device)
                strong_clean, weak_clean = model(state, times, **kwargs)
                strong = clean_to_velocity(
                    strong_clean, state, times, denominator_floor=t_floor
                )
                weak = clean_to_velocity(
                    weak_clean, state, times, denominator_floor=t_floor
                )
                if index in selected:
                    future = max(current - args.horizon, ig_min)
                    future_times = torch.full_like(times, future)
                    strong_future_clean, weak_future_clean = model(
                        state, future_times, **kwargs
                    )
                    strong_future = clean_to_velocity(
                        strong_future_clean,
                        state,
                        future_times,
                        denominator_floor=t_floor,
                    )
                    weak_future = clean_to_velocity(
                        weak_future_clean,
                        state,
                        future_times,
                        denominator_floor=t_floor,
                    )
                    weak_defect = weak_future - weak
                    strong_defect = strong_future - strong
                    depth_gap = strong - weak
                    weak_revision = -weak_defect
                    strong_revision = -strong_defect
                    shared_revision = shared_retiming_revision(
                        weak,
                        weak_future,
                        strong,
                        strong_future,
                    )
                    plain_angular_gap = depth_gap + _remove_parallel(
                        weak_revision, depth_gap
                    )
                    shared_angular_gap = depth_gap + _remove_parallel(
                        shared_revision, depth_gap
                    )
                    metrics = {
                        "weak_defect_rms": _rms(weak_defect),
                        "strong_defect_rms": _rms(strong_defect),
                        "depth_gap_rms": _rms(depth_gap),
                        "revision_to_gap_rms": _rms(weak_revision)
                        / _rms(depth_gap).clamp_min(1e-12),
                        "weak_strong_defect_cosine": _cosine(
                            weak_defect, strong_defect
                        ),
                        "weak_defect_depth_gap_cosine": _cosine(
                            weak_defect, depth_gap
                        ),
                        "weak_common_strong_energy_fraction": _projection_fraction(
                            weak_defect, strong_defect
                        ),
                        "shared_revision_rms": _rms(shared_revision),
                        "shared_revision_to_gap_rms": _rms(shared_revision)
                        / _rms(depth_gap).clamp_min(1e-12),
                        "shared_revision_depth_gap_cosine": _cosine(
                            shared_revision, depth_gap
                        ),
                        "plain_angular_cosine_to_gap": _cosine(
                            plain_angular_gap, depth_gap
                        ),
                        "shared_angular_cosine_to_gap": _cosine(
                            shared_angular_gap, depth_gap
                        ),
                    }
                    bucket = collected[index]
                    for name, values in metrics.items():
                        bucket.setdefault(name, []).extend(values.cpu().tolist())

                if ig_min <= current <= float(config.guidance.ig.t_max):
                    drift = weak + args.guidance_scale * (strong - weak)
                else:
                    drift = strong
                state = state - (current - following) * drift

    rows = []
    for index, requested in selected.items():
        current = float(grid[index].item())
        future = max(current - args.horizon, ig_min)
        bucket = collected[index]
        row = {
            "requested_time": requested,
            "actual_time": current,
            "future_time": future,
            "data_odds": float(data_odds(torch.tensor(current))),
            **{name: mean(values) for name, values in bucket.items()},
        }
        rows.append(row)
    rows.sort(key=lambda row: float(row["actual_time"]), reverse=True)
    csv_path = args.output_dir / "retiming_geometry.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "protocol": "raev2_pfr_retiming_geometry_v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "state_key": "ema",
        "samples": args.samples,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "precision": args.precision,
        "allow_tf32": allow_tf32,
        "guidance_scale": args.guidance_scale,
        "horizon": args.horizon,
        "trajectory": "ordinary_internal_guidance",
        "rows": rows,
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
