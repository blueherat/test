#!/usr/bin/env python3
"""Paired RAEv2 screen separating clean-prediction radius and direction.

Single-process jobs deliberately keep batch size, random-number call shapes,
forward layout, and FP32 guidance arithmetic identical across conditions.
This is an APG-related mechanism control, not a claim of a new SOTA method.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "external/RAEv2/src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.raev2_radius_guidance import radius_guided_clean
from experiments.raev2_pfr_retiming import clean_to_velocity
from experiments.raev2_stage1_compat import install_raev2_decoder_config_compat
from experiments.raev2_training_core import file_sha256
from experiments.sample_raev2_pfr_retiming import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, load_config, shifted_time_grid,
)
from utils.model_utils import instantiate_from_config

PROTOCOL = "raev2_radius_direction_v1"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("ordinary", "radial", "tangent", "retracted"), required=True)
    parser.add_argument("--grouping", choices=("token", "global"), default="token")
    parser.add_argument("--sample-count", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=202609051)
    parser.add_argument("--guidance-scale", type=float, default=1.78)
    parser.add_argument("--guidance-min-time", type=float, default=0.1)
    parser.add_argument("--guidance-max-time", type=float, default=1.0)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    if min(args.sample_count, args.batch_size, args.num_steps) <= 0:
        parser.error("sample count, batch size and steps must be positive")
    if not 0 <= args.guidance_min_time <= args.guidance_max_time <= 1:
        parser.error("invalid guidance interval")
    if not math.isfinite(args.guidance_scale) or args.guidance_scale < 1:
        parser.error("guidance scale must be finite and >= 1")
    return args


def main():
    args = parse_args()
    out = args.output_dir.expanduser().resolve()
    if (out / "request.json").exists():
        raise FileExistsError(f"refusing to overwrite an existing run: {out}")
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("DINOV3_CKPT_DIR", "/home/zhoushunyu/data/eqvae/models/RAEv2/encoders/dinov3")
    install_raev2_decoder_config_compat()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    tf32 = args.precision == "bf16"
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    config = load_config(args.config.resolve())
    decoder = instantiate_from_config(config.stage_1).to(device).eval().requires_grad_(False)
    del decoder.encoder
    torch.cuda.empty_cache()
    model = instantiate_from_config(config.stage_2).to(device).eval().requires_grad_(False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    model.load_state_dict(checkpoint["ema"], strict=True)
    checkpoint_step = int(checkpoint.get("step", 0))
    del checkpoint
    shift = math.sqrt((config.misc.time_dist_shift_dim or math.prod(config.misc.latent_size)) / config.misc.time_dist_shift_base)
    grid = shifted_time_grid(args.num_steps, shift, device)
    grid_cpu = grid.cpu().tolist()
    request = {
        "protocol": PROTOCOL, **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "checkpoint_step": checkpoint_step, "checkpoint_sha256": file_sha256(args.checkpoint),
        "config_sha256": file_sha256(args.config), "state_key": "ema", "world_size": 1,
        "guidance_arithmetic": "fp32_clean_full_plus_beta_minus_one_gap",
        "forward_layout": "single_conditional_batch", "tf32": tf32,
        "time_grid": grid_cpu, "torch_version": torch.__version__,
        "source_sha256": {p.name: file_sha256(p) for p in (
            Path(__file__), ROOT / "experiments/raev2_radius_guidance.py",
            ROOT / "experiments/raev2_pfr_retiming.py",
        )},
    }
    (out / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    rng = torch.Generator(device=device).manual_seed(args.seed)
    noise_hash, label_hash = hashlib.sha256(), hashlib.sha256()
    images = []
    aggregate = [dict() for _ in range(args.num_steps)]
    aggregate_counts = [dict() for _ in range(args.num_steps)]
    started = time.perf_counter()
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if args.precision == "bf16" else nullcontext()
    with torch.inference_mode(), autocast:
        for start in range(0, args.sample_count, args.batch_size):
            stop = min(start + args.batch_size, args.sample_count)
            state = torch.randn(stop - start, *config.misc.latent_size, device=device, generator=rng, dtype=torch.float32)
            labels = torch.arange(start, stop, device=device) % int(config.misc.num_classes)
            noise_hash.update(state.cpu().contiguous().numpy().tobytes())
            label_hash.update(labels.cpu().numpy().tobytes())
            for index, (current, following) in enumerate(zip(grid_cpu[:-1], grid_cpu[1:])):
                times = torch.full((len(state),), current, device=device, dtype=torch.float32)
                full, base = model(state, times, context=labels, attn_mask=None)
                beta = args.guidance_scale if args.guidance_min_time <= current <= args.guidance_max_time else 1.0
                clean, telemetry = radius_guided_clean(
                    full, base, guidance_scale=beta, mode=args.mode,
                    grouping=args.grouping, return_telemetry=True, check_finite=False,
                )
                # Accumulate on device; transfer compact statistics only at the end.
                for key, value in telemetry.items():
                    values = value.detach().double()
                    aggregate[index][key] = aggregate[index].get(key, 0.0) + values.sum()
                    aggregate_counts[index][key] = aggregate_counts[index].get(key, 0) + values.numel()
                drift = clean_to_velocity(clean, state, times, denominator_floor=float(config.transport.t_eps))
                state = state - (current - following) * drift
            if not torch.isfinite(state).all():
                raise FloatingPointError(f"nonfinite endpoint in batch {start}")
            decoded = decoder.decode(state).clamp(0, 1)
            if start == 0:
                save_image(decoded.float().cpu(), out / "preview.png", nrow=4)
            images.append(decoded.mul(255).permute(0, 2, 3, 1).to(device="cpu", dtype=torch.uint8).numpy())
            if start == 0 or stop % (args.batch_size * 8) == 0 or stop == args.sample_count:
                elapsed = time.perf_counter() - started
                progress = {"completed": stop, "total": args.sample_count, "elapsed_seconds": elapsed,
                            "estimated_remaining_seconds": elapsed * (args.sample_count - stop) / stop}
                print(json.dumps(progress), flush=True)
                (out / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
    archive = out / "samples.npz"
    np.savez(archive, np.concatenate(images))
    rows = []
    for index, stats in enumerate(aggregate):
        rows.append({"index": index, "noise_time": grid_cpu[index], **{
            key: float(value.item()) / aggregate_counts[index][key] for key, value in stats.items()
        }})
    with (out / "geometry.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"protocol": PROTOCOL, "mode": args.mode, "grouping": args.grouping,
               "samples": args.sample_count, "seed": args.seed,
               "noise_sha256": noise_hash.hexdigest(), "labels_sha256": label_hash.hexdigest(),
               "archive_sha256": file_sha256(archive), "elapsed_seconds": time.perf_counter() - started,
               "full_model_calls": math.ceil(args.sample_count / args.batch_size) * args.num_steps,
               "full_sample_evaluations": args.sample_count * args.num_steps,
               "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(), "complete": True}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
