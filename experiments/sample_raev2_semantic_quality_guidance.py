#!/usr/bin/env python3
"""Paired single-process RAEv2 condition/depth guidance screen.

Every mode uses the same doubled [conditional, null] forward layout, complete
initial-noise/label hashes, FP32 clean-space guidance, and shifted Euler grid.
Use identical seed, batch size, sample count, config and precision to pair runs.
Durable batch archives and atomic progress survive interruption; partial runs
are deliberately not resumed into a potentially different numerical protocol.
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

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "external/RAEv2/src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.raev2_semantic_quality_guidance import MODES, semantic_quality_clean

PROTOCOL = "raev2_semantic_quality_guidance_v1"
DEFAULT_CONFIG = ROOT / "experiments/configs/raev2_strict_lpl_dinov3l_k7.yaml"
DEFAULT_CHECKPOINT = Path("/home/zhoushunyu/data/eqvae/models/RAEv2/stage2/imagenet/dinov3l-k7/checkpoint.pt")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--sample-count", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=202609051)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--ig-scale", type=float, default=1.78)
    for axis in ("cfg", "ig"):
        parser.add_argument(f"--{axis}-min-time", type=float, default=0.1)
        parser.add_argument(f"--{axis}-max-time", type=float, default=1.0)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args(argv)
    if min(args.sample_count, args.batch_size, args.num_steps) <= 0:
        parser.error("sample count, batch size and steps must be positive")
    if not 0 <= args.seed < 2**63:
        parser.error("seed must satisfy 0 <= seed < 2**63")
    for axis in ("cfg", "ig"):
        scale = getattr(args, f"{axis}_scale")
        minimum, maximum = (getattr(args, f"{axis}_{edge}_time") for edge in ("min", "max"))
        if not math.isfinite(scale) or scale < 0:
            parser.error(f"{axis} scale must be finite and non-negative")
        if not 0 <= minimum <= maximum <= 1:
            parser.error(f"invalid {axis} guidance interval")
    return args


def scale_at_time(scale: float, minimum: float, maximum: float, current: float) -> float:
    """Inclusive time window, matching the paired radius sampler convention."""
    return scale if minimum <= current <= maximum else 1.0


def evaluate_four_corners(model, state, times, labels, *, null_label: int):
    """One forward with layout [all conditional images, all null images]."""
    n = len(state)
    if times.shape != (n,) or labels.shape != (n,):
        raise ValueError("times and labels must match the state batch")
    full, base = model(
        torch.cat((state, state)), torch.cat((times, times)),
        context=torch.cat((labels, torch.full_like(labels, null_label))), attn_mask=None,
    )
    if full.shape != (2 * n, *state.shape[1:]) or base.shape != full.shape:
        raise ValueError("model predictions must preserve the doubled latent shape")
    fc, fu = full.chunk(2)
    bc, bu = base.chunk(2)
    return fc, bc, fu, bu


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    out = args.output_dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite a nonempty run directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "batches").mkdir()
    args.config = args.config.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_dir = out
    started = time.perf_counter()
    request = {"protocol": PROTOCOL, **{
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }, "world_size": 1, "state_key": "ema", "forward_layout": "concatenated_conditional_then_null",
        "guidance_arithmetic": "fp32_inputs_subtractions_projections_and_clean_sum",
        "state_arithmetic": "fp32_euler", "projection_grouping": "global_per_image",
        "pixel_arithmetic": "native_decode_clamp_mul255_uint8",
        "projection_norm_restoration": False, "time_window_endpoints": "inclusive",
        "label_schedule": "global_sample_index_mod_num_classes", "resume_supported": False,
        "mode_formulas": {
            "ordinary": "Fc+(ig-1)*(Fc-Bc)",
            "full_cfg": "Fc+(cfg-1)*(Fc-Fu)",
            "combined": "Fc+(ig-1)*(Fc-Bc)+(cfg-1)*(Fc-Fu)",
            "semantic_orthogonal": "Fc+(ig-1)*D+(cfg-1)*(C-Proj_D(C)); D=Fc-Bc,C=Fc-Fu",
            "bilinear": "Fc+(ig-1)*(Fc-Bc)+(cfg-1)*(Fc-Fu)+(ig-1)*(cfg-1)*(Fc-Bc-Fu+Bu)",
        }, "torch_version": torch.__version__, "status": "initializing"}
    atomic_json(out / "request.json", request)
    atomic_json(out / "progress.json", {"completed": 0, "total": args.sample_count, "status": "initializing"})

    # Keep GPU/model dependencies out of imports so algebra/layout tests run on CPU.
    from torchvision.utils import save_image
    from experiments.raev2_pfr_retiming import clean_to_velocity
    from experiments.raev2_stage1_compat import install_raev2_decoder_config_compat
    from experiments.raev2_training_core import file_sha256
    from experiments.sample_raev2_pfr_retiming import load_config, shifted_time_grid
    from utils.model_utils import instantiate_from_config

    source_paths = (
        Path(__file__), ROOT / "experiments/raev2_semantic_quality_guidance.py",
        ROOT / "experiments/raev2_pfr_retiming.py", ROOT / "experiments/sample_raev2_pfr_retiming.py",
        ROOT / "experiments/raev2_stage1_compat.py", ROOT / "external/RAEv2/src/stage2/models/DDT.py",
        ROOT / "external/RAEv2/src/stage2/models/model_utils.py",
    )
    request.update({"checkpoint_sha256": file_sha256(args.checkpoint),
                    "config_sha256": file_sha256(args.config),
                    "source_sha256": {str(path.relative_to(ROOT)): file_sha256(path) for path in source_paths}})
    atomic_json(out / "request.json", request)
    os.environ.setdefault("DINOV3_CKPT_DIR", "/home/zhoushunyu/data/eqvae/models/RAEv2/encoders/dinov3")
    install_raev2_decoder_config_compat()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    tf32 = args.precision == "bf16"
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    config = load_config(args.config)
    if config.conditioning.type != "label" or config.transport.prediction != "x":
        raise ValueError("sampler requires label conditioning and clean endpoint prediction")
    decoder_path = Path(config.stage_1.params["pretrained_decoder_path"]).expanduser().resolve()
    stats_path = Path(config.stage_1.params["normalization_stat_path"]).expanduser().resolve()
    request.update({"decoder_checkpoint": str(decoder_path), "decoder_sha256": file_sha256(decoder_path),
                    "normalization_stats": str(stats_path), "normalization_stats_sha256": file_sha256(stats_path)})
    atomic_json(out / "request.json", request)
    decoder = instantiate_from_config(config.stage_1).to(device).eval().requires_grad_(False)
    del decoder.encoder
    torch.cuda.empty_cache()
    model = instantiate_from_config(config.stage_2).to(device).eval().requires_grad_(False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    model.load_state_dict(checkpoint["ema"], strict=True)
    request["checkpoint_step"] = int(checkpoint.get("step", 0))
    del checkpoint
    shift = math.sqrt((config.misc.time_dist_shift_dim or math.prod(config.misc.latent_size)) / config.misc.time_dist_shift_base)
    grid = shifted_time_grid(args.num_steps, shift, device).cpu().tolist()
    request.update({"time_shift": shift, "time_grid": grid, "tf32": tf32,
                    "cuda_device": torch.cuda.get_device_name(device), "status": "sampling",
                    "latent_size": list(config.misc.latent_size), "num_classes": int(config.misc.num_classes),
                    "transport_t_eps": float(config.transport.t_eps)})
    atomic_json(out / "request.json", request)
    rng = torch.Generator(device=device).manual_seed(args.seed)
    noise_hash, label_hash = hashlib.sha256(), hashlib.sha256()
    aggregate = [dict() for _ in range(args.num_steps)]
    completed = 0
    model_calls = 0
    batch_manifest = []
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if args.precision == "bf16" else nullcontext()
    try:
        with torch.inference_mode(), autocast:
            for start in range(0, args.sample_count, args.batch_size):
                stop = min(start + args.batch_size, args.sample_count)
                state = torch.randn(stop - start, *config.misc.latent_size, device=device, generator=rng, dtype=torch.float32)
                labels = torch.arange(start, stop, device=device) % int(config.misc.num_classes)
                noise_bytes = state.cpu().contiguous().numpy().tobytes()
                label_bytes = labels.cpu().contiguous().numpy().tobytes()
                batch_noise_sha = hashlib.sha256(noise_bytes).hexdigest()
                batch_label_sha = hashlib.sha256(label_bytes).hexdigest()
                for index, (current, following) in enumerate(zip(grid[:-1], grid[1:])):
                    times = torch.full((len(state),), current, device=device, dtype=torch.float32)
                    corners = evaluate_four_corners(model, state, times, labels, null_label=int(config.misc.num_classes))
                    model_calls += 1
                    ig = scale_at_time(args.ig_scale, args.ig_min_time, args.ig_max_time, current)
                    cfg = scale_at_time(args.cfg_scale, args.cfg_min_time, args.cfg_max_time, current)
                    clean, telemetry = semantic_quality_clean(
                        *corners, ig_scale=ig, cfg_scale=cfg, mode=args.mode,
                        return_telemetry=True, check_finite=False,
                    )
                    for key, values in telemetry.items():
                        aggregate[index][key] = aggregate[index].get(key, 0.0) + values.double().sum()
                    drift = clean_to_velocity(clean, state, times, denominator_floor=float(config.transport.t_eps))
                    state = state - (current - following) * drift
                if state.dtype != torch.float32 or not bool(torch.isfinite(state).all()):
                    raise FloatingPointError(f"invalid FP32 endpoint in batch {start}")
                decoded = decoder.decode(state)
                if not bool(torch.isfinite(decoded).all()):
                    raise FloatingPointError(f"nonfinite decoded image in batch {start}")
                decoded = decoded.clamp(0, 1)
                if start == 0:
                    save_image(decoded.float().cpu(), out / "preview.png", nrow=4)
                images = decoded.mul(255).permute(0, 2, 3, 1).to(device="cpu", dtype=torch.uint8).numpy()
                batch_path = out / "batches" / f"{start:06d}_{stop:06d}.npz"
                temporary = batch_path.with_suffix(".tmp")
                with temporary.open("wb") as file:
                    np.savez(file, images)
                temporary.replace(batch_path)
                noise_hash.update(noise_bytes)
                label_hash.update(label_bytes)
                completed = stop
                batch_manifest.append({"start": start, "stop": stop, "path": str(batch_path.relative_to(out)),
                                       "noise_sha256": batch_noise_sha, "labels_sha256": batch_label_sha,
                                       "archive_sha256": file_sha256(batch_path)})
                atomic_json(out / "batch_manifest.json", {"batches": batch_manifest})
                elapsed = time.perf_counter() - started
                progress = {"completed": stop, "total": args.sample_count, "elapsed_seconds": elapsed,
                            "estimated_remaining_seconds": elapsed * (args.sample_count - stop) / stop,
                            "noise_sha256": noise_hash.hexdigest(), "labels_sha256": label_hash.hexdigest(),
                            "model_calls": model_calls, "status": "sampling"}
                atomic_json(out / "progress.json", progress)
                if start == 0 or stop % (args.batch_size * 8) == 0 or stop == args.sample_count:
                    print(json.dumps(progress), flush=True)
        images = []
        for batch in batch_manifest:
            with np.load(out / batch["path"]) as archive:
                images.append(archive["arr_0"])
        archive = out / "samples.npz"
        temporary = archive.with_suffix(".tmp")
        with temporary.open("wb") as file:
            np.savez(file, np.concatenate(images))
        temporary.replace(archive)
        rows = [{"index": index, "noise_time": grid[index], **{
            key: float(value.item()) / args.sample_count for key, value in stats.items()
        }} for index, stats in enumerate(aggregate)]
        with (out / "geometry.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = {"protocol": PROTOCOL, "mode": args.mode, "samples": args.sample_count,
                   "seed": args.seed, "cfg_scale": args.cfg_scale, "ig_scale": args.ig_scale,
                   "noise_sha256": noise_hash.hexdigest(), "labels_sha256": label_hash.hexdigest(),
                   "archive_sha256": file_sha256(archive), "elapsed_seconds": time.perf_counter() - started,
                   "full_model_calls": model_calls, "full_sample_evaluations": 2 * args.sample_count * args.num_steps,
                   "full_model_nfe_per_sample": 2 * args.num_steps,
                   "base_head_note": "base head is returned by each full forward; no extra prefix query",
                   "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(), "complete": True}
        atomic_json(out / "summary.json", summary)
        atomic_json(out / "progress.json", {**progress, "status": "complete", "complete": True})
        print(json.dumps(summary), flush=True)
    except BaseException as error:
        atomic_json(out / "progress.json", {"completed": completed, "total": args.sample_count,
                    "status": "failed", "error": f"{type(error).__name__}: {error}",
                    "elapsed_seconds": time.perf_counter() - started, "model_calls": model_calls,
                    "noise_sha256": noise_hash.hexdigest(), "labels_sha256": label_hash.hexdigest()})
        raise


if __name__ == "__main__":
    main()
