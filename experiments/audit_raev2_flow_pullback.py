#!/usr/bin/env python3
"""Small FP32 audit of finite-flow cotangent guidance before a quality screen."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "external/RAEv2/src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.raev2_pfr_retiming import clean_to_velocity
from experiments.raev2_training_core import file_sha256
from experiments.sample_raev2_pfr_retiming import DEFAULT_CHECKPOINT, DEFAULT_CONFIG, load_config, shifted_time_grid
from utils.model_utils import instantiate_from_config


def rms(x):
    return x.float().flatten(1).square().mean(1).sqrt()


def cosine(x, y):
    return (x.double().flatten(1) * y.double().flatten(1)).sum(1) / (
        x.double().flatten(1).norm(dim=1) * y.double().flatten(1).norm(dim=1)
    ).clamp_min(1e-30)


def main():
    # Import delayed to allow the independent implementation to finish first.
    from experiments.raev2_flow_pullback import full_euler_flow, flow_pullback_direction
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--seed", type=int, default=202609053)
    p.add_argument("--horizon", type=float, default=1 / 32)
    p.add_argument("--epsilon", type=float, default=0.001)
    p.add_argument("--times", default="0.97,0.9,0.75")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = p.parse_args()
    out = args.output_dir.expanduser().resolve()
    if (out / "request.json").exists():
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = load_config(args.config)
    model = instantiate_from_config(config.stage_2).to(device).eval().requires_grad_(False)
    ckpt = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    model.load_state_dict(ckpt["ema"], strict=True)
    del ckpt
    shift = math.sqrt((config.misc.time_dist_shift_dim or math.prod(config.misc.latent_size)) / config.misc.time_dist_shift_base)
    grid = shifted_time_grid(100, shift, device).cpu().tolist()
    requested_times = tuple(float(value) for value in args.times.split(","))
    if not requested_times or any(not 0.1 <= value <= 1 for value in requested_times):
        raise ValueError("audit times must be between 0.1 and 1")
    indices = {min(range(100), key=lambda i: abs(grid[i] - t)) for t in requested_times}
    request = {"protocol": "raev2_flow_pullback_audit_v1", "seed": args.seed,
               "samples": args.samples, "horizon": args.horizon, "epsilon": args.epsilon,
               "precision": "fp32_no_tf32", "current_trajectory": "ordinary_ig_beta1.78",
               "future_flow": "full", "indices": sorted(indices), "requested_times": requested_times, "substeps": [4, 8],
               "checkpoint_sha256": file_sha256(args.checkpoint),
               "source_sha256": {f.name: file_sha256(f) for f in (Path(__file__), ROOT / "experiments/raev2_flow_pullback.py")},
               "provisional_gate": "median pullback/raw cosine < .99; K4/K8 cosine > .99; finite-work relative error < .05"}
    (out / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    rng = torch.Generator(device=device).manual_seed(args.seed)
    rows = []
    started = time.perf_counter()
    for sample_id in range(args.samples):
        with torch.no_grad():
            state = torch.randn(1, *config.misc.latent_size, device=device, generator=rng)
            labels = torch.tensor([(sample_id * 137) % 1000], device=device)
            snapshots = []
            for index in range(max(indices) + 1):
                current, following = grid[index:index + 2]
                times = torch.full((1,), current, device=device)
                full, base = model(state, times, context=labels, attn_mask=None)
                gap = full.float() - base.float()
                if index in indices:
                    snapshots.append((index, current, state.clone(), gap.clone()))
                clean = full.float() + 0.78 * gap
                state = state - (current - following) * clean_to_velocity(clean, state, times, denominator_floor=0.05)
        for index, current, state, gap in snapshots:
            future = max(0.1, current - args.horizon)
            results = []
            for substeps in (4, 8):
                result = flow_pullback_direction(
                    model, state, labels, gap, start_time=current, end_time=future,
                    substeps=substeps, denominator_floor=0.05, checkpoint_forward=True,
                )
                results.append(result)
            # API returns a structured result with the actual frozen future gap.
            first, refined = results
            unit = first.direction / rms(first.direction).view(-1, 1, 1, 1).clamp_min(1e-30)
            with torch.no_grad():
                plus = full_euler_flow(model, state + args.epsilon * unit, labels,
                                      start_time=current, end_time=future, substeps=4,
                                      denominator_floor=0.05, checkpoint_forward=False)
                minus = full_euler_flow(model, state - args.epsilon * unit, labels,
                                       start_time=current, end_time=future, substeps=4,
                                       denominator_floor=0.05, checkpoint_forward=False)
                numerical = ((plus.double() - minus.double()) * first.future_gap.double()).mean() / (2 * args.epsilon)
                expected = (first.pullback.double() * unit.double()).mean()
                work_current = (first.pullback.double() * gap.double()).mean()
                work_raw = (first.pullback.double() * first.raw_future_direction.double()).mean()
                work_pullback = (first.pullback.double() * first.direction.double()).mean()
            row = {"sample_id": sample_id, "label": int(labels.item()), "index": index,
                   "noise_time": current, "future_time": future,
                   "pullback_raw_cosine": float(cosine(first.direction, first.raw_future_direction).item()),
                   "pullback_current_cosine": float(cosine(first.direction, gap).item()),
                   "raw_current_cosine": float(cosine(first.raw_future_direction, gap).item()),
                   "k4_k8_cosine": float(cosine(first.direction, refined.direction).item()),
                   "norm_ratio": float((rms(first.direction) / rms(gap)).item()),
                   "finite_work_relative_error": float(((numerical - expected).abs() / expected.abs().clamp_min(1e-20)).item()),
                   "work_current": float(work_current.item()), "work_raw_future": float(work_raw.item()),
                   "work_pullback": float(work_pullback.item()),
                   "pullback_rms": float(rms(first.pullback).item()), "future_gap_rms": float(rms(first.future_gap).item())}
            rows.append(row)
            print(json.dumps(row), flush=True)
        with (out / "rows.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    numeric_keys = [k for k, v in rows[0].items() if isinstance(v, float)]
    summary = {"complete": True, "rows": len(rows), "elapsed_seconds": time.perf_counter() - started,
               "medians": {k: float(torch.tensor([r[k] for r in rows], dtype=torch.float64).median()) for k in numeric_keys},
               "max_memory_allocated_bytes": torch.cuda.max_memory_allocated()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
