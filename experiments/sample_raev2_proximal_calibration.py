#!/usr/bin/env python3
"""Paired frozen RAEv2 sampling with an audited full-grid proximal correction.

Both modes run the official 100-step shifted Euler grid, conditional-only
full/base forward, and FP32 IG=1.78 on the checkpoint's [.1,1] interval.
Proximal mode applies the merged fixed correction after every Euler step;
there is no scale, time window, null branch, or extra model evaluation.
Use the same seed/batch size/precision to pair independent mode runs.
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

from experiments.raev2_proximal_calibration import DiagonalFit, apply_proximal

PROTOCOL = "raev2_paired_official_proximal_sampling_v1"
CALIBRATION_PROTOCOL = "raev2_diagonal_proximal_real_coupling_audit_v1"
SHARED_CALIBRATION_PROTOCOL = "raev2_channel_shared_proximal_real_coupling_audit_v1"
SHARED_SOURCE_FILES = (
    "experiments/raev2_proximal_shared_calibration.py",
    "experiments/audit_raev2_proximal_shared_calibration.py",
)
GLOBAL_CALIBRATION_PROTOCOL = "raev2_global_proximal_real_coupling_audit_v1"
GLOBAL_SOURCE_FILES = (
    "experiments/raev2_proximal_global_calibration.py",
    "experiments/audit_raev2_proximal_global_calibration.py",
)
DERIVED_CALIBRATIONS = {
    SHARED_CALIBRATION_PROTOCOL: ("channel_shared_affine", SHARED_SOURCE_FILES),
    GLOBAL_CALIBRATION_PROTOCOL: ("global_zero_anchor", GLOBAL_SOURCE_FILES),
}
DEFAULT_CONFIG = ROOT / "experiments/configs/raev2_strict_lpl_dinov3l_k7.yaml"
DEFAULT_CHECKPOINT = Path("/home/zhoushunyu/data/eqvae/models/RAEv2/stage2/imagenet/dinov3l-k7/checkpoint.pt")
NUM_STEPS = 100
CALIBRATION_SOURCE_FILES = (
    "experiments/raev2_proximal_calibration.py",
    "experiments/audit_raev2_proximal_calibration.py",
    "experiments/sample_raev2_pfr_retiming.py",
    "external/RAEv2/src/stage2/models/DDT.py",
    "external/RAEv2/src/stage2/models/model_utils.py",
)


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("official", "proximal"), required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--sample-count", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=202609063)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args(argv)
    if args.mode == "proximal" and args.calibration is None:
        parser.error("--calibration is required in proximal mode")
    if min(args.sample_count, args.batch_size) <= 0:
        parser.error("sample count and batch size must be positive")
    if not 0 <= args.seed < 2**63:
        parser.error("seed must satisfy 0 <= seed < 2**63")
    return args


def validate_calibration_archive(
    archive: dict, *, config_sha256: str, checkpoint_sha256: str,
    normalization_stats_sha256: str, time_grid: list[float],
    latent_shape: tuple[int, ...], precision: str, batch_size: int,
    expected_source_sha256: dict[str, str],
) -> dict[int, DiagonalFit]:
    """Reject incomplete grids, incompatible checkpoints, or changed calibration semantics."""
    requests = archive.get("source_requests")
    if not isinstance(requests, list) or not requests:
        raise ValueError("expected merged calibration with source_requests")
    expected = {
        "num_steps": NUM_STEPS,
        "config_sha256": config_sha256, "checkpoint_sha256": checkpoint_sha256,
        "normalization_stats_sha256": normalization_stats_sha256,
        "time_grid": time_grid, "precision": precision, "batch_size": batch_size,
        "ig_scale": 1.78, "ig_interval": [0.1, 1.0],
        "fit_input": "true_next_state", "fit_error": "frozen_euler_next_minus_true_next",
        "forward_layout": "single_conditional_batch",
        "guidance_and_step_arithmetic": "fp32", "applied_coefficient_arithmetic": "fp32",
        "correction_multiplier": 1.0, "correction_time_selection": "none",
    }
    for request in requests:
        protocol = request.get("protocol")
        if protocol in DERIVED_CALIBRATIONS:
            structure, source_files = DERIVED_CALIBRATIONS[protocol]
            if request.get("fit_structure") != structure:
                raise ValueError(f"derived calibration requires its fixed {structure} structure")
            if not request.get("parent_calibration_path") or not request.get("parent_calibration_sha256"):
                raise ValueError("derived calibration must identify the original training calibration")
            for path in source_files:
                if path not in expected_source_sha256:
                    raise ValueError(f"derived calibration source hash is required: {path}")
        elif protocol != CALIBRATION_PROTOCOL or request.get("fit_structure", "diagonal_affine") != "diagonal_affine":
            raise ValueError("unsupported calibration protocol or fit structure")
        for key, value in expected.items():
            if request.get(key) != value:
                raise ValueError(f"calibration mismatch in {key}")
        for path, digest in expected_source_sha256.items():
            if request.get("source_sha256", {}).get(path) != digest:
                raise ValueError(f"calibration source hash mismatch: {path}")
    for key in ("protocol", "fit_structure", "samples", "seed", "train_bank_metadata", "parent_calibration_sha256"):
        if any(request.get(key) != requests[0].get(key) for request in requests[1:]):
            raise ValueError(f"calibration source requests differ in {key}")
    steps = archive.get("applied_fp32_steps")
    if not isinstance(steps, dict) or set(steps) != set(range(NUM_STEPS)):
        raise ValueError("applied_fp32_steps must contain all 100 official steps exactly once")
    if set(archive.get("steps", {})) != set(range(NUM_STEPS)):
        raise ValueError("source calibration must contain all 100 official steps")
    fits = {}
    for index in range(NUM_STEPS):
        fit = DiagonalFit.from_state_dict(steps[index])
        for field in ("center", "offset", "slope", "variance", "covariance"):
            value = getattr(fit, field)
            if value.shape != latent_shape or value.dtype != torch.float32:
                raise ValueError(f"step {index} {field} must match latent shape in FP32")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"step {index} {field} is nonfinite")
        if bool((fit.slope < 0).any()) or fit.count < 2:
            raise ValueError(f"step {index} is not a valid convex calibration")
        if requests[0]["protocol"] == GLOBAL_CALIBRATION_PROTOCOL:
            if bool((fit.center != 0).any()) or bool((fit.offset != 0).any()):
                raise ValueError(f"global_zero_anchor step {index} requires zero center and offset")
            if not bool((fit.slope == fit.slope.flatten()[0]).all()):
                raise ValueError(f"global_zero_anchor step {index} requires a single global slope")
        fits[index] = fit
    return fits


def official_euler_step(model, state, labels, current: float, following: float):
    """One conditional forward, matching the calibration's frozen Euler map."""
    if not 0 <= following < current <= 1:
        raise ValueError("Euler step must satisfy 0 <= next < current <= 1")
    if state.dtype != torch.float32 or labels.shape != (len(state),):
        raise ValueError("expected FP32 state and one label per image")
    times = torch.full((len(state),), current, device=state.device, dtype=torch.float32)
    full, base = model(state, times, context=labels, attn_mask=None)
    full, base = full.float(), base.float()
    if full.shape != state.shape or base.shape != state.shape:
        raise ValueError("model predictions must preserve the latent shape")
    clean = full + 0.78 * (full - base) if 0.1 <= current <= 1.0 else full
    return state - (current - following) * ((state - clean) / current)


def finish_step(predicted_next: torch.Tensor, *, mode: str, fit: DiagonalFit | None):
    if mode == "official":
        return predicted_next
    if mode != "proximal" or fit is None:
        raise ValueError("proximal mode requires a calibrated fit at every step")
    return apply_proximal(predicted_next, fit)


def main():
    args = parse_args()
    for key in ("output_dir", "config", "checkpoint", "calibration"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.expanduser().resolve())
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing run: {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "batches").mkdir()
    source_snapshot = out / "sampler_source.py"
    source_snapshot.write_bytes(Path(__file__).read_bytes())
    source_snapshot_sha256 = hashlib.sha256(source_snapshot.read_bytes()).hexdigest()
    started = time.perf_counter()
    request = {"protocol": PROTOCOL, **{
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }, "world_size": 1, "state_key": "ema", "num_steps": NUM_STEPS,
        "ig_scale": 1.78, "ig_interval": [0.1, 1.0],
        "correction_multiplier": 1.0, "correction_time_selection": "none; every official step",
        "correction_enabled": args.mode == "proximal",
        "forward_layout": "single_conditional_batch", "guidance_and_step_arithmetic": "fp32",
        "pixel_arithmetic": "native_decode_clamp_mul255_uint8",
        "noise_schema": "one device torch.Generator(seed), sequential fixed-shape FP32 randn batches",
        "label_schema": "global_sample_index_mod_num_classes",
        "partial_outputs": "completed batch archives are durable; no mid-trajectory resume",
        "sampler_source_snapshot": {"path": str(source_snapshot), "sha256": source_snapshot_sha256},
        "torch_version": str(torch.__version__), "status": "initializing"}
    atomic_json(out / "request.json", request)
    atomic_json(out / "progress.json", {"status": "initializing", "completed": 0, "total": args.sample_count})
    from torchvision.utils import save_image
    from experiments.raev2_stage1_compat import install_raev2_decoder_config_compat
    from experiments.raev2_training_core import file_sha256
    from experiments.sample_raev2_pfr_retiming import load_config, shifted_time_grid
    from utils.model_utils import instantiate_from_config
    config = load_config(args.config)
    if (config.conditioning.type != "label" or config.transport.prediction != "x"
            or config.guidance.ig.scale != 1.78 or config.guidance.ig.t_min != 0.1
            or config.guidance.ig.t_max != 1.0 or config.guidance.cfg.scale != 1.0
            or config.sampler.num_steps != NUM_STEPS):
        raise ValueError("configuration is not the fixed official 100-step IG protocol")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    tf32 = args.precision == "bf16"
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    shift = math.sqrt((config.misc.time_dist_shift_dim or math.prod(config.misc.latent_size)) / config.misc.time_dist_shift_base)
    grid = shifted_time_grid(NUM_STEPS, shift, device).cpu().tolist()
    if float(config.transport.t_eps) >= grid[-2]:
        raise ValueError("transport denominator floor changes the calibrated Euler map")
    model_source_sha256 = {path: file_sha256(ROOT / path) for path in CALIBRATION_SOURCE_FILES}
    request.update(config_sha256=file_sha256(args.config), checkpoint_sha256=file_sha256(args.checkpoint),
                   normalization_stats_sha256=file_sha256(Path(config.stage_1.params["normalization_stat_path"])),
                   source_sha256={**model_source_sha256, str(Path(__file__).relative_to(ROOT)): source_snapshot_sha256,
                                  "experiments/raev2_stage1_compat.py": file_sha256(ROOT / "experiments/raev2_stage1_compat.py")},
                   time_grid=grid, time_shift=shift, tf32=tf32, cuda_device=torch.cuda.get_device_name(device),
                   latent_shape=list(config.misc.latent_size), num_classes=int(config.misc.num_classes),
                   transport_t_eps=float(config.transport.t_eps),
                   decoder_artifacts={key: {"path": str(config.stage_1.params[key]), "sha256": file_sha256(Path(config.stage_1.params[key]))}
                                      for key in ("pretrained_decoder_path", "normalization_stat_path")})
    fits = {}
    if args.calibration is not None:
        archive = torch.load(args.calibration, map_location="cpu", weights_only=True)
        validation_source_sha256 = dict(model_source_sha256)
        derived_sources = set()
        for item in archive.get("source_requests", []):
            if item.get("protocol") in DERIVED_CALIBRATIONS:
                derived_sources.update(DERIVED_CALIBRATIONS[item["protocol"]][1])
        if derived_sources:
            validation_source_sha256.update({path: file_sha256(ROOT / path) for path in sorted(derived_sources)})
            request["source_sha256"].update({path: validation_source_sha256[path] for path in sorted(derived_sources)})
            verified_parents = {}
            for item in archive["source_requests"]:
                raw_path = item.get("parent_calibration_path")
                if not raw_path:
                    raise ValueError("derived calibration must identify its parent calibration path")
                parent = Path(raw_path).expanduser().resolve()
                if str(parent) not in verified_parents:
                    verified_parents[str(parent)] = file_sha256(parent)
                if verified_parents[str(parent)] != item.get("parent_calibration_sha256"):
                    raise ValueError("derived parent calibration hash mismatch")
            request["verified_parent_calibrations"] = verified_parents
        fits = validate_calibration_archive(
            archive, config_sha256=request["config_sha256"], checkpoint_sha256=request["checkpoint_sha256"],
            normalization_stats_sha256=request["normalization_stats_sha256"], time_grid=grid,
            latent_shape=tuple(config.misc.latent_size), precision=args.precision, batch_size=args.batch_size,
            expected_source_sha256=validation_source_sha256,
        )
        request.update(calibration_sha256=file_sha256(args.calibration),
                       calibration_source_requests=archive["source_requests"],
                       calibration_source_shards=archive.get("source_shards"),
                       calibration_steps=list(range(NUM_STEPS)), calibration_key="applied_fp32_steps")
        del archive
        fits = {index: fit.to(device=device, dtype=torch.float32) for index, fit in fits.items()} if args.mode == "proximal" else {}
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
    rng = torch.Generator(device=device).manual_seed(args.seed)
    noise_hash, label_hash = hashlib.sha256(), hashlib.sha256()
    batch_manifest = []
    correction_sum = torch.zeros(NUM_STEPS, dtype=torch.float64, device=device)
    completed = 0
    model_calls = 0
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if args.precision == "bf16" else nullcontext()
    try:
        with torch.inference_mode(), autocast:
            for start in range(0, args.sample_count, args.batch_size):
                stop = min(start + args.batch_size, args.sample_count)
                state = torch.randn(stop - start, *config.misc.latent_size, device=device, generator=rng, dtype=torch.float32)
                labels = torch.arange(start, stop, device=device) % int(config.misc.num_classes)
                noise_bytes = state.cpu().contiguous().numpy().tobytes()
                label_bytes = labels.cpu().contiguous().numpy().tobytes()
                for index, (current, following) in enumerate(zip(grid[:-1], grid[1:])):
                    predicted = official_euler_step(model, state, labels, current, following)
                    model_calls += 1
                    state = finish_step(predicted, mode=args.mode, fit=fits.get(index))
                    correction_sum[index] += (state - predicted).flatten(1).square().mean(1).sqrt().double().sum()
                if state.dtype != torch.float32 or not bool(torch.isfinite(state).all()):
                    raise FloatingPointError(f"invalid FP32 endpoint at batch {start}")
                endpoint_sha256 = hashlib.sha256(state.cpu().contiguous().numpy().tobytes()).hexdigest()
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
                                       "noise_sha256": hashlib.sha256(noise_bytes).hexdigest(),
                                       "labels_sha256": hashlib.sha256(label_bytes).hexdigest(),
                                       "endpoint_sha256": endpoint_sha256, "archive_sha256": file_sha256(batch_path)})
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
            with np.load(out / batch["path"]) as batch_archive:
                images.append(batch_archive["arr_0"])
        sample_path = out / "samples.npz"
        temporary = sample_path.with_suffix(".tmp")
        with temporary.open("wb") as file:
            np.savez(file, np.concatenate(images))
        temporary.replace(sample_path)
        rows = [{"step_index": index, "time": grid[index], "next_time": grid[index + 1],
                 "applied_correction_rms": float(correction_sum[index]) / args.sample_count} for index in range(NUM_STEPS)]
        with (out / "step_metrics.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = {"protocol": PROTOCOL, "mode": args.mode, "samples": args.sample_count, "seed": args.seed,
                   "noise_sha256": noise_hash.hexdigest(), "labels_sha256": label_hash.hexdigest(),
                   "archive_sha256": file_sha256(sample_path), "full_model_calls": model_calls,
                   "full_sample_evaluations": args.sample_count * NUM_STEPS, "full_model_nfe_per_sample": NUM_STEPS,
                   "base_head_note": "returned by each conditional full forward; no extra prefix or null query",
                   "elapsed_seconds": time.perf_counter() - started,
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
