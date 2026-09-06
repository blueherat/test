#!/usr/bin/env python3
"""Whole-cohort energy-ball correction of the actual frozen RAEv2 rollout.

Every official Euler step is evaluated for the entire 1,000-image cohort before
one shared shrinkage factor is applied. The target radius uses only bank A's
saved FP64 clean second moment. It is an empirical training estimate, not a
known population bound. No strength, time window, or expansion is available.
"""
from __future__ import annotations

import argparse
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

from experiments.audit_raev2_proximal_calibration import sha256_file, write_csv
from experiments.sample_raev2_proximal_calibration import (
    CALIBRATION_PROTOCOL, CALIBRATION_SOURCE_FILES, DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG, NUM_STEPS, atomic_json, official_euler_step,
)

PROTOCOL = "raev2_actual_cohort_energy_ball_guidance_v1"
DEFAULT_PARENT = Path("/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/proximal_calibration_seed202609062/merged/calibration.pt")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--mode", choices=("official", "energy_ball"), required=True)
    p.add_argument("--calibration-parent", type=Path, default=DEFAULT_PARENT)
    p.add_argument("--paired-official", type=Path)
    p.add_argument("--sample-count", type=int, default=1000,
                   help="full experiment uses 1000; smaller cohorts are allowed for arithmetic smoke tests")
    p.add_argument("--batch-size", type=int, choices=(8,), default=8)
    p.add_argument("--seed", type=int, default=202609066)
    p.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = p.parse_args(argv)
    if not 0 <= args.seed < 2**63:
        p.error("seed must satisfy 0 <= seed < 2**63")
    if args.sample_count <= 0:
        p.error("sample count must be positive")
    return args


def clean_second_moment_from_parent(archive: dict) -> float:
    """Recover E_A X² from the last original population-moment fit (s=0)."""
    if set(archive.get("steps", {})) != set(range(NUM_STEPS)):
        raise ValueError("parent must contain all 100 original calibration steps")
    requests = archive.get("source_requests")
    if not isinstance(requests, list) or not requests:
        raise ValueError("parent requires original source_requests")
    for request in requests:
        if (request.get("protocol") != CALIBRATION_PROTOCOL
                or request.get("fit_input") != "true_next_state"
                or request.get("fit_error") != "frozen_euler_next_minus_true_next"
                or request.get("samples") != 1000
                or len(request.get("time_grid", [])) != 101
                or request["time_grid"][-1] != 0.0):
            raise ValueError("clean energy requires the original bank A fit at next_time zero")
    for key in ("train_bank_metadata", "time_grid", "source_sha256"):
        if any(r.get(key) != requests[0].get(key) for r in requests[1:]):
            raise ValueError(f"parent requests disagree on {key}")
    fit = archive["steps"][99]
    center, variance = fit["center"], fit["variance"]
    if (center.dtype != torch.float64 or variance.dtype != torch.float64
            or center.shape != variance.shape or fit["count"] != 1000):
        raise ValueError("expected FP64 population moments from all 1,000 A images")
    if not bool(torch.isfinite(center).all() & torch.isfinite(variance).all()) or bool((variance < 0).any()):
        raise ValueError("invalid clean population moments")
    result = float((variance + center.square()).mean())
    if not math.isfinite(result) or result <= 0:
        raise ValueError("clean second moment must be finite and positive")
    return result


def target_second_moment(next_time: float, clean_second_moment: float) -> float:
    if not 0 <= next_time <= 1 or not math.isfinite(clean_second_moment) or clean_second_moment < 0:
        raise ValueError("invalid bridge time or clean second moment")
    return (1.0 - next_time)**2 * clean_second_moment + next_time**2


def initialize_cohort(*, sample_count, latent_shape, batch_size, seed, device):
    """Preserve the official generator's sequence and individual draw shapes."""
    state = torch.empty(sample_count, *latent_shape, dtype=torch.float32, device=device)
    labels = torch.arange(sample_count, device=device) % 1000
    generator = torch.Generator(device=device).manual_seed(seed)
    noise_digest, label_digest = hashlib.sha256(), hashlib.sha256()
    batches = []
    for start in range(0, sample_count, batch_size):
        stop = min(start + batch_size, sample_count)
        noise = torch.randn(stop-start, *latent_shape, device=device, generator=generator, dtype=torch.float32)
        state[start:stop].copy_(noise)
        noise_bytes = noise.cpu().contiguous().numpy().tobytes()
        label_bytes = labels[start:stop].cpu().contiguous().numpy().tobytes()
        noise_digest.update(noise_bytes)
        label_digest.update(label_bytes)
        batches.append({"start": start, "stop": stop,
                        "noise_sha256": hashlib.sha256(noise_bytes).hexdigest(),
                        "labels_sha256": hashlib.sha256(label_bytes).hexdigest()})
    return state, labels, {"noise_sha256": noise_digest.hexdigest(),
                           "labels_sha256": label_digest.hexdigest(), "batches": batches}


def coordinate_second_moment(value: torch.Tensor, batch_size: int) -> float:
    total = torch.zeros((), dtype=torch.float64, device=value.device)
    for start in range(0, len(value), batch_size):
        total += value[start:start+batch_size].double().square().sum()
    return float(total / value.numel())


def project_cohort_(state, bound: float, *, mode: str, batch_size: int) -> dict:
    """Project one full empirical law; report FP32 rounding explicitly."""
    if state.dtype != torch.float32 or not math.isfinite(bound) or bound < 0:
        raise ValueError("expected FP32 states and a finite nonnegative energy bound")
    if mode not in ("official", "energy_ball"):
        raise ValueError("unsupported correction mode")
    before = coordinate_second_moment(state, batch_size)
    if not math.isfinite(before):
        raise FloatingPointError("nonfinite cohort energy")
    ideal = min(1.0, math.sqrt(bound / before)) if before > 0 else 1.0
    applied = float(torch.tensor(ideal, dtype=torch.float32)) if mode == "energy_ball" else 1.0
    after, correction = before, 0.0
    if applied < 1.0:
        after_sum = torch.zeros((), dtype=torch.float64, device=state.device)
        correction_sum = torch.zeros_like(after_sum)
        for start in range(0, len(state), batch_size):
            original = state[start:start+batch_size]
            corrected = original * applied
            after_sum += corrected.double().square().sum()
            correction_sum += (corrected.double() - original.double()).square().sum()
            original.copy_(corrected)
        after = float(after_sum / state.numel())
        correction = math.sqrt(float(correction_sum / state.numel()))
    return {"cohort_second_moment_before": before, "target_second_moment": bound,
            "lambda_ideal_fp64": ideal, "lambda_applied_fp32": applied,
            "active": applied < 1.0, "cohort_second_moment_after": after,
            "cohort_correction_rms": correction,
            "energy_bound_excess_after": max(after-bound, 0.0),
            "lambda_roundoff": applied-ideal if mode == "energy_ball" else 0.0}


def cohort_euler_step_(model, state, labels, current, following, *, batch_size):
    for start in range(0, len(state), batch_size):
        stop = min(start+batch_size, len(state))
        predicted = official_euler_step(model, state[start:stop], labels[start:stop], current, following)
        state[start:stop].copy_(predicted)


def validate_pair(request, evidence, directory: Path):
    official = json.loads((directory / "request.json").read_text())
    if official.get("mode") != "official":
        raise ValueError("paired reference must be official mode")
    for key in ("seed", "sample_count", "batch_size", "precision", "config_sha256",
                "checkpoint_sha256", "normalization_stats_sha256", "time_grid", "tf32",
                "guidance_and_step_arithmetic", "pixel_arithmetic", "forward_layout"):
        if request[key] != official[key]:
            raise ValueError(f"paired official request differs in {key}")
    if request["decoder_artifacts"] != official["decoder_artifacts"]:
        raise ValueError("paired official decoder artifacts differ")
    for path, digest in official["source_sha256"].items():
        if request["source_sha256"].get(path) != digest:
            raise ValueError(f"paired official source hash mismatch: {path}")
    summary = json.loads((directory / "summary.json").read_text())
    if not summary.get("complete") or summary.get("mode") != "official":
        raise ValueError("paired official run is not complete")
    for key in ("noise_sha256", "labels_sha256"):
        if summary[key] != evidence[key]:
            raise ValueError(f"paired official {key} mismatch")
    return {"path": str(directory), "request_sha256": sha256_file(directory / "request.json"),
            "summary_sha256": sha256_file(directory / "summary.json"), "noise_and_labels_match": True}


def main():
    args = parse_args()
    for key, value in vars(args).copy().items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "batches").mkdir()
    started = time.perf_counter()
    snapshot = out / "sampler_source.py"
    snapshot.write_bytes(Path(__file__).read_bytes())
    request = {"protocol": PROTOCOL, **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               "num_steps": 100, "world_size": 1, "state_key": "ema", "ig_scale": 1.78, "ig_interval": [0.1, 1.0],
               "forward_layout": "single_conditional_batch", "guidance_and_step_arithmetic": "fp32",
               "pixel_arithmetic": "native_decode_clamp_mul255_uint8", "loop_layout": "time_then_all_batches_then_one_cohort_projection",
               "energy_reduction_arithmetic": "fp64", "correction_arithmetic": "fp32",
               "radius_formula": "B_s=(1-s)^2*m2_clean+s^2; lambda=min(1,sqrt(B_s/A_actual_cohort))",
               "target_energy_source": "bank A step99 FP64 population variance+mean^2; no C fit",
               "target_energy_claim": "finite training estimate, not a known upper bound on true population energy",
               "law_claim": "exact radial law projection in real arithmetic for the full empirical cohort; FP32 rounding recorded",
               "correction_time_selection": "none; all 100 official steps", "expansion_allowed": False,
               "noise_schema": "one device torch.Generator(seed), sequential fixed-shape FP32 randn batches",
               "label_schema": "global_sample_index_mod_num_classes", "torch_version": str(torch.__version__),
               "partial_outputs": "step evidence and completed decoded batches durable; no mid-trajectory resume",
               "sampler_source_snapshot": {"path": str(snapshot), "sha256": sha256_file(snapshot)}, "status": "initializing"}
    atomic_json(out / "request.json", request)
    atomic_json(out / "progress.json", {"status": "initializing", "completed_steps": 0, "decoded": 0})
    from experiments.sample_raev2_pfr_retiming import load_config, shifted_time_grid
    from experiments.raev2_stage1_compat import install_raev2_decoder_config_compat
    from utils.model_utils import instantiate_from_config
    from torchvision.utils import save_image
    config = load_config(args.config)
    if (config.conditioning.type != "label" or config.transport.prediction != "x"
            or config.guidance.ig.scale != 1.78 or config.guidance.ig.t_min != 0.1
            or config.guidance.ig.t_max != 1.0 or config.guidance.cfg.scale != 1.0
            or config.sampler.num_steps != 100 or config.misc.num_classes != 1000):
        raise ValueError("requires the fixed official 100-step IG configuration")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = args.precision == "bf16"
    torch.backends.cudnn.allow_tf32 = args.precision == "bf16"
    shift = math.sqrt((config.misc.time_dist_shift_dim or math.prod(config.misc.latent_size)) / config.misc.time_dist_shift_base)
    grid = shifted_time_grid(100, shift, device).cpu().tolist()
    if float(config.transport.t_eps) >= grid[-2]:
        raise ValueError("transport denominator floor changes the fixed Euler map")
    source_files = (*CALIBRATION_SOURCE_FILES, "experiments/sample_raev2_proximal_calibration.py",
                    "experiments/raev2_stage1_compat.py", str(Path(__file__).relative_to(ROOT)))
    request.update(config_sha256=sha256_file(args.config), checkpoint_sha256=sha256_file(args.checkpoint),
                   normalization_stats_sha256=sha256_file(Path(config.stage_1.params["normalization_stat_path"])),
                   source_sha256={path: sha256_file(ROOT / path) for path in source_files},
                   time_grid=grid, time_shift=shift, tf32=args.precision == "bf16",
                   cuda_device=torch.cuda.get_device_name(device), latent_shape=list(config.misc.latent_size),
                   num_classes=1000, transport_t_eps=float(config.transport.t_eps),
                   decoder_artifacts={key: {"path": str(config.stage_1.params[key]), "sha256": sha256_file(Path(config.stage_1.params[key]))}
                                      for key in ("pretrained_decoder_path", "normalization_stat_path")})
    parent = torch.load(args.calibration_parent, map_location="cpu", weights_only=True, mmap=True)
    m2 = clean_second_moment_from_parent(parent)
    for original in parent["source_requests"]:
        for key in ("config_sha256", "checkpoint_sha256", "normalization_stats_sha256", "time_grid"):
            if original[key] != request[key]:
                raise ValueError(f"parent differs in {key}")
        for path, digest in original["source_sha256"].items():
            if sha256_file(ROOT / path) != digest:
                raise ValueError(f"parent source changed: {path}")
    request.update(calibration_parent_sha256=sha256_file(args.calibration_parent),
                   calibration_parent_source_requests=parent["source_requests"],
                   clean_second_moment=m2, train_bank_metadata=parent["source_requests"][0]["train_bank_metadata"])
    del parent
    state, labels, evidence = initialize_cohort(sample_count=args.sample_count, latent_shape=config.misc.latent_size,
                                               batch_size=args.batch_size, seed=args.seed, device=device)
    request.update(noise_sha256=evidence["noise_sha256"], labels_sha256=evidence["labels_sha256"])
    atomic_json(out / "noise_manifest.json", evidence)
    if args.paired_official:
        request["paired_official_verification"] = validate_pair(request, evidence, args.paired_official)
    atomic_json(out / "request.json", request)
    os.environ.setdefault("DINOV3_CKPT_DIR", "/home/zhoushunyu/data/eqvae/models/RAEv2/encoders/dinov3")
    install_raev2_decoder_config_compat()
    decoder = instantiate_from_config(config.stage_1).to(device).eval().requires_grad_(False)
    del decoder.encoder
    torch.cuda.empty_cache()
    model = instantiate_from_config(config.stage_2).to(device).eval().requires_grad_(False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    model.load_state_dict(checkpoint["ema"], strict=True)
    request.update(checkpoint_step=int(checkpoint.get("step", 0)), status="sampling")
    del checkpoint
    atomic_json(out / "request.json", request)
    rows, batches, model_calls, decoded_count = [], [], 0, 0
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if args.precision == "bf16" else nullcontext()
    try:
        with torch.inference_mode(), autocast:
            for index, (current, following) in enumerate(zip(grid[:-1], grid[1:])):
                cohort_euler_step_(model, state, labels, current, following, batch_size=args.batch_size)
                model_calls += math.ceil(args.sample_count / args.batch_size)
                metrics = project_cohort_(state, target_second_moment(following, m2), mode=args.mode, batch_size=args.batch_size)
                rows.append({"step_index": index, "time": current, "next_time": following, **metrics})
                write_csv(out / "step_metrics.csv", rows)
                progress = {"status": "sampling", "completed_steps": index+1, "total_steps": 100,
                            "decoded": 0, "elapsed_seconds": time.perf_counter()-started, "last_step": rows[-1]}
                atomic_json(out / "progress.json", progress)
                print(json.dumps(progress), flush=True)
            endpoint_digest = hashlib.sha256()
            for batch in evidence["batches"]:
                start, stop = batch["start"], batch["stop"]
                endpoint_bytes = state[start:stop].cpu().contiguous().numpy().tobytes()
                endpoint_digest.update(endpoint_bytes)
                decoded = decoder.decode(state[start:stop])
                if not bool(torch.isfinite(decoded).all()):
                    raise FloatingPointError("nonfinite decoded pixels")
                decoded = decoded.clamp(0, 1)
                if start == 0:
                    save_image(decoded.float().cpu(), out / "preview.png", nrow=4)
                pixels = decoded.mul(255).permute(0, 2, 3, 1).to(device="cpu", dtype=torch.uint8).numpy()
                path = out / "batches" / f"{start:06d}_{stop:06d}.npz"
                temporary = path.with_suffix(".tmp")
                with temporary.open("wb") as file:
                    np.savez(file, pixels)
                temporary.replace(path)
                batches.append({**batch, "path": str(path.relative_to(out)),
                                "endpoint_sha256": hashlib.sha256(endpoint_bytes).hexdigest(),
                                "pixels_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(), "archive_sha256": sha256_file(path)})
                decoded_count = stop
                atomic_json(out / "batch_manifest.json", {"batches": batches})
                atomic_json(out / "progress.json", {"status": "decoding", "completed_steps": 100,
                            "decoded": stop, "total": args.sample_count, "elapsed_seconds": time.perf_counter()-started})
        images = []
        for batch in batches:
            with np.load(out / batch["path"]) as archive:
                images.append(archive["arr_0"])
        image_array = np.concatenate(images)
        path = out / "samples.npz"
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as file:
            np.savez(file, image_array)
        temporary.replace(path)
        summary = {"protocol": PROTOCOL, "mode": args.mode, "complete": True, "samples": args.sample_count,
                   "seed": args.seed, "noise_sha256": evidence["noise_sha256"], "labels_sha256": evidence["labels_sha256"],
                   "endpoint_sha256": endpoint_digest.hexdigest(), "pixels_sha256": hashlib.sha256(image_array.tobytes()).hexdigest(),
                   "archive_sha256": sha256_file(path), "full_model_calls": model_calls,
                   "full_sample_evaluations": 100*args.sample_count, "full_model_nfe_per_sample": 100,
                   "clean_second_moment": m2, "active_steps": sum(row["active"] for row in rows),
                   "minimum_lambda": min(row["lambda_applied_fp32"] for row in rows),
                   "elapsed_seconds": time.perf_counter()-started,
                   "max_memory_allocated_bytes": torch.cuda.max_memory_allocated()}
        atomic_json(out / "summary.json", summary)
        atomic_json(out / "progress.json", {**summary, "status": "complete", "completed_steps": 100, "decoded": decoded_count})
        print(json.dumps(summary), flush=True)
    except BaseException as error:
        atomic_json(out / "progress.json", {"status": "failed", "completed_steps": len(rows), "decoded": decoded_count,
                    "model_calls": model_calls, "error": f"{type(error).__name__}: {error}"})
        raise


if __name__ == "__main__":
    main()
