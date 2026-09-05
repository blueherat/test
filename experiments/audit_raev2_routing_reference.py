#!/usr/bin/env python3
"""Real-checkpoint parity and kernel-residual audit for routing references."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "external/RAEv2/src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.raev2_routing_reference import evaluate_routing_reference
from experiments.raev2_pfr_retiming import clean_to_velocity
from experiments.raev2_training_core import file_sha256
from experiments.sample_raev2_pfr_retiming import DEFAULT_CHECKPOINT, DEFAULT_CONFIG, load_config, shifted_time_grid
from utils.model_utils import instantiate_from_config


def rms(value):
    return value.float().flatten(1).square().mean(1).sqrt()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=202609054)
    p.add_argument("--decoder-block", type=int, default=1)
    p.add_argument("--indices", default="0,20,47,73")
    args = p.parse_args()
    out = args.output_dir.resolve()
    if (out / "request.json").exists():
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    config = load_config(DEFAULT_CONFIG)
    model = instantiate_from_config(config.stage_2).to(device).eval().requires_grad_(False)
    ckpt = torch.load(DEFAULT_CHECKPOINT, map_location="cpu", mmap=True, weights_only=False)
    model.load_state_dict(ckpt["ema"], strict=True)
    del ckpt
    indices = sorted({int(value) for value in args.indices.split(",")})
    if not indices or min(indices) < 0 or max(indices) >= 100:
        raise ValueError("indices must be in [0,99]")
    request = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               "precision": "bf16_autocast_with_fp32_guidance", "checkpoint_sha256": file_sha256(DEFAULT_CHECKPOINT),
               "source_sha256": file_sha256(ROOT / "experiments/raev2_routing_reference.py"),
               "indices": indices, "modes": ["native_explicit", "identity", "preserve_self", "uniform"]}
    (out / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    shift = math.sqrt((config.misc.time_dist_shift_dim or math.prod(config.misc.latent_size)) / config.misc.time_dist_shift_base)
    grid = shifted_time_grid(100, shift, device).cpu().tolist()
    rng = torch.Generator(device=device).manual_seed(args.seed)
    rows = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = torch.randn(args.batch_size, *config.misc.latent_size, device=device, generator=rng)
        labels = (torch.arange(args.batch_size, device=device) * 137) % 1000
        for index in range(max(indices) + 1):
            current, following = grid[index:index + 2]
            times = torch.full((len(state),), current, device=device)
            full, base = model(state, times, context=labels, attn_mask=None)
            gap = full.float() - base.float()
            if index in request["indices"]:
                kernel_rms = None
                for mode in request["modes"]:
                    result = evaluate_routing_reference(model, state, times, labels, mode=mode, decoder_block=args.decoder_block)
                    if not torch.equal(result.full, full) or not torch.equal(result.base, base):
                        raise AssertionError(f"native full/base parity failed at {index}/{mode}")
                    negative_gap_rms = rms(result.full.float() - result.negative.float())
                    if mode == "native_explicit":
                        kernel_rms = negative_gap_rms
                    if not torch.isfinite(result.negative).all():
                        raise FloatingPointError(f"nonfinite negative at {index}/{mode}")
                    for sample in range(len(state)):
                        rows.append({"sample_id": sample, "label": int(labels[sample]), "index": index,
                                     "noise_time": current, "mode": mode, "full_base_bitwise_equal": True,
                                     "ig_gap_rms": float(rms(gap)[sample]), "negative_gap_rms": float(negative_gap_rms[sample]),
                                     "gap_over_kernel_rms": float(negative_gap_rms[sample] / kernel_rms[sample].clamp_min(1e-20)),
                                     "normalization_gain": float(rms(gap)[sample] / negative_gap_rms[sample].clamp_min(1e-20)),
                                     "mean_self_mass": float(result.telemetry["self_mass"][sample].mean()),
                                     "mean_negative_self_mass": float(result.telemetry["negative_self_mass"][sample].mean()),
                                     "mean_routing_information": float(result.telemetry["routing_information"][sample].mean())})
                print(json.dumps({"index": index, "rows": len(rows), "parity": True}), flush=True)
            clean = full.float() + 0.78 * gap
            state = state - (current - following) * clean_to_velocity(clean, state, times, denominator_floor=0.05)
    with (out / "rows.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary = {"complete": True, "native_full_base_parity": True, "rows": len(rows), "by_mode": {}}
    for mode in request["modes"]:
        subset = [r for r in rows if r["mode"] == mode]
        summary["by_mode"][mode] = {key: float(torch.tensor([r[key] for r in subset]).median())
                                      for key in ("ig_gap_rms", "negative_gap_rms", "gap_over_kernel_rms", "normalization_gain", "mean_self_mass", "mean_routing_information")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
