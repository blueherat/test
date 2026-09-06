#!/usr/bin/env python3
"""Paired RAEv2 IG with a balanced stochastic full-score corrector.

The deterministic predictor is the existing single-conditional FP32 Euler
path. An independent RNG supplies refresh noise and never advances the
initial-noise stream. Full/base predictions are shared with the corrector.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "external/RAEv2/src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.raev2_stochastic_guidance import refresh_after_euler, refresh_coefficients
from experiments.sample_raev2_semantic_quality_guidance import atomic_json


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--mode", choices=("none", "balanced", "score_only", "noise_only"), default="balanced")
    p.add_argument("--eta", type=float, default=0.25)
    p.add_argument("--sample-count", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=202609061)
    p.add_argument("--num-steps", type=int, default=100)
    p.add_argument("--ig-scale", type=float, default=1.78)
    p.add_argument("--ig-min-time", type=float, default=0.5)
    p.add_argument("--ig-max-time", type=float, default=1.0)
    p.add_argument("--refresh-min-time", type=float, default=0.5)
    p.add_argument("--refresh-max-time", type=float, default=0.95)
    p.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--config", type=Path, default=ROOT / "experiments/configs/raev2_strict_lpl_dinov3l_k7.yaml")
    p.add_argument("--checkpoint", type=Path, default=Path("/home/zhoushunyu/data/eqvae/models/RAEv2/stage2/imagenet/dinov3l-k7/checkpoint.pt"))
    a = p.parse_args()
    if min(a.sample_count, a.batch_size, a.num_steps) < 1:
        p.error("positive sample count, batch size and steps required")
    if not 0 <= a.seed < 2**63 or not math.isfinite(a.eta) or a.eta < 0:
        p.error("invalid seed or eta")
    if not math.isfinite(a.ig_scale) or a.ig_scale < 0:
        p.error("IG scale must be finite and nonnegative")
    if not 0 <= a.ig_min_time <= a.ig_max_time <= 1:
        p.error("invalid IG interval")
    if not 0 < a.refresh_min_time < a.refresh_max_time < 1:
        p.error("refresh interval must lie strictly inside (0,1)")
    return a


def main():
    args = parse_args()
    out = args.output_dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "batches").mkdir()
    from torchvision.utils import save_image
    from experiments.raev2_pfr_retiming import clean_to_velocity
    from experiments.raev2_stage1_compat import install_raev2_decoder_config_compat
    from experiments.raev2_training_core import file_sha256
    from experiments.sample_raev2_pfr_retiming import load_config, shifted_time_grid
    from utils.model_utils import instantiate_from_config

    os.environ.setdefault("DINOV3_CKPT_DIR", "/home/zhoushunyu/data/eqvae/models/RAEv2/encoders/dinov3")
    install_raev2_decoder_config_compat()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    tf32 = args.precision == "bf16"
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    config = load_config(args.config)
    decoder = instantiate_from_config(config.stage_1).to(device).eval().requires_grad_(False)
    del decoder.encoder
    torch.cuda.empty_cache()
    model = instantiate_from_config(config.stage_2).to(device).eval().requires_grad_(False)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    model.load_state_dict(ckpt["ema"], strict=True)
    checkpoint_step = int(ckpt.get("step", 0))
    del ckpt
    shift = math.sqrt((config.misc.time_dist_shift_dim or math.prod(config.misc.latent_size)) / config.misc.time_dist_shift_base)
    grid = shifted_time_grid(args.num_steps, shift, device).cpu().tolist()
    interval = (args.refresh_min_time, args.refresh_max_time)
    refresh_seed = int.from_bytes(hashlib.sha256(f"raev2-refresh-v1:{args.seed}".encode()).digest()[:8], "little") % (2**63)
    rng = torch.Generator(device=device).manual_seed(args.seed)
    refresh_rng = torch.Generator(device=device).manual_seed(refresh_seed)
    request = {
        "protocol": "raev2_stochastic_guidance_v1",
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "checkpoint_step": checkpoint_step, "checkpoint_sha256": file_sha256(args.checkpoint),
        "config_sha256": file_sha256(args.config), "time_grid": grid, "tf32": tf32,
        "state_key": "ema", "world_size": 1, "torch_version": torch.__version__,
        "forward_layout": "single_conditional_batch", "guidance_arithmetic": "fp32_clean_full_plus_beta_minus_one_gap",
        "pixel_arithmetic": "native_decode_clamp_mul255_uint8", "transport_t_eps": float(config.transport.t_eps),
        "score_reference": "full conditional head at pre-Euler state and time",
        "refresh_noise_schema": "independent torch.Generator, SHA256(raev2-refresh-v1:seed), first 8 little-endian bytes mod 2**63",
        "refresh_seed": refresh_seed, "refresh_draws": "every grid step and batch, regardless of mode/window",
        "source_sha256": {str(p.relative_to(ROOT)): file_sha256(p) for p in (
            Path(__file__), ROOT / "experiments/raev2_stochastic_guidance.py",
            ROOT / "experiments/raev2_pfr_retiming.py", ROOT / "experiments/sample_raev2_pfr_retiming.py",
            ROOT / "external/RAEv2/src/stage2/models/DDT.py",
        )},
        "decoder_artifacts": {key: {"path": str(config.stage_1.params[key]), "sha256": file_sha256(Path(config.stage_1.params[key]))}
                              for key in ("pretrained_decoder_path", "normalization_stat_path")},
    }
    atomic_json(out / "request.json", request)
    noise_hash, labels_hash, refresh_first_hash = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    aggregates = [torch.zeros(3, device=device, dtype=torch.float64) for _ in grid[:-1]]
    rows = []
    for t, s in zip(grid[:-1], grid[1:]):
        c = refresh_coefficients(t, s, eta=args.eta, interval=interval)
        rows.append({"time": t, "next_time": s, "clock_increment": c.clock_increment,
                     "noise_std": c.noise_std, "decay": c.decay})
    started = time.perf_counter()
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if args.precision == "bf16" else nullcontext()
    with torch.no_grad(), autocast:
        for start in range(0, args.sample_count, args.batch_size):
            stop = min(start + args.batch_size, args.sample_count)
            state = torch.randn(stop - start, *config.misc.latent_size, device=device, generator=rng, dtype=torch.float32)
            labels = torch.arange(start, stop, device=device) % int(config.misc.num_classes)
            noise_hash.update(state.cpu().numpy().tobytes())
            labels_hash.update(labels.cpu().numpy().tobytes())
            for i, (current, following) in enumerate(zip(grid[:-1], grid[1:])):
                times = torch.full((len(state),), current, device=device, dtype=torch.float32)
                full, base = model(state, times, context=labels, attn_mask=None)
                full, base = full.float(), base.float()
                beta = args.ig_scale if args.ig_min_time <= current <= args.ig_max_time else 1.0
                clean = full if beta == 1.0 else full + (beta - 1.0) * (full - base)
                euler = state - (current - following) * clean_to_velocity(clean, state, times, denominator_floor=float(config.transport.t_eps))
                noise = torch.randn(state.shape, device=device, generator=refresh_rng, dtype=torch.float32)
                if start == 0:
                    refresh_first_hash.update(noise.cpu().numpy().tobytes())
                refreshed = refresh_after_euler(state, euler, full, current, following,
                                               eta=args.eta, noise=noise, mode=args.mode, interval=interval)
                # Per-sample RMS; no all-image direction or terminal metric is used online.
                values = torch.stack([(refreshed - euler).flatten(1).square().mean(1).sqrt().sum(),
                                      euler.flatten(1).square().mean(1).sqrt().sum(),
                                      (clean-full).flatten(1).square().mean(1).sqrt().sum()])
                aggregates[i] += values.detach().double()
                state = refreshed
            if not bool(torch.isfinite(state).all()):
                raise FloatingPointError(f"nonfinite endpoint at {start}")
            decoded = decoder.decode(state)
            if not bool(torch.isfinite(decoded).all()):
                raise FloatingPointError(f"nonfinite decode at {start}")
            decoded = decoded.clamp(0, 1)
            if start == 0:
                save_image(decoded.float().cpu(), out / "preview.png", nrow=4)
            images = decoded.mul(255).permute(0, 2, 3, 1).to(device="cpu", dtype=torch.uint8).numpy()
            np.savez(out / "batches" / f"{start:06d}_{stop:06d}.npz", images)
            elapsed = time.perf_counter() - started
            progress = {"completed": stop, "total": args.sample_count, "elapsed_seconds": elapsed,
                        "estimated_remaining_seconds": elapsed * (args.sample_count-stop)/stop}
            atomic_json(out / "progress.json", progress)
            if start == 0 or stop % (8*args.batch_size) == 0 or stop == args.sample_count:
                print(json.dumps(progress), flush=True)
    arrays = []
    for p in sorted((out / "batches").glob("*.npz")):
        with np.load(p) as f:
            arrays.append(f["arr_0"])
    np.savez(out / "samples.npz", np.concatenate(arrays))
    for row, total in zip(rows, aggregates):
        row.update(dict(zip(("refresh_rms", "euler_state_rms", "guidance_rms"), (total/args.sample_count).cpu().tolist())))
    with (out / "geometry.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {"complete": True, "samples": args.sample_count, "seed": args.seed,
               "noise_sha256": noise_hash.hexdigest(), "labels_sha256": labels_hash.hexdigest(),
               "first_batch_all_step_refresh_noise_sha256": refresh_first_hash.hexdigest(),
               "archive_sha256": file_sha256(out / "samples.npz"),
               "full_model_calls": args.num_steps*math.ceil(args.sample_count/args.batch_size),
               "full_sample_evaluations": args.num_steps*args.sample_count, "extra_model_calls": 0,
               "elapsed_seconds": time.perf_counter()-started,
               "max_memory_allocated_bytes": torch.cuda.max_memory_allocated()}
    atomic_json(out / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
