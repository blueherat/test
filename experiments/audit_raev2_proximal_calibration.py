#!/usr/bin/env python3
"""Fit and audit every official RAEv2 Euler step on disjoint real-image banks.

This runner never generates endpoint samples or invokes a terminal evaluator.
For each official step t -> s, form z_t=(1-t)X+t*epsilon, the frozen Euler
prediction T(z_t), and the coupled true next state y_s=(1-s)X+s*epsilon. Fit a
convex diagonal quadratic gradient to (y_s, T(z_t)-y_s), and evaluate its exact
proximal map on independent images. All 100 steps are retained, with no learned
time selection or guidance multiplier. Time indices may be partitioned among
independent GPU workers; the parent audit combines all completed shards.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "external/RAEv2/src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.raev2_proximal_calibration import (
    DiagonalMoments, heldout_step_metrics,
)

PROTOCOL = "raev2_diagonal_proximal_real_coupling_audit_v1"
DATA_ROOT = Path("/home/zhoushunyu/data/eqvae/experiments")
DEFAULT_TRAIN = DATA_ROOT / "spectral_ig_mechanism_v3_formal_1k/01_clean_latents"
DEFAULT_HELDOUT = DATA_ROOT / "raev2_guidance_restart_20260906/heldout_clean_current_fp32"
DEFAULT_PROVENANCE = DATA_ROOT / "raev2_guidance_restart_20260906/cache_reencode_audit/summary.json"
DEFAULT_CONFIG = ROOT / "experiments/configs/raev2_strict_lpl_dinov3l_k7.yaml"
DEFAULT_CHECKPOINT = Path("/home/zhoushunyu/data/eqvae/models/RAEv2/stage2/imagenet/dinov3l-k7/checkpoint.pt")
NUM_STEPS = 100


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    for start in range(0, len(array), 8):
        digest.update(np.ascontiguousarray(array[start:start + 8]).tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_torch_save(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot save an empty audit table")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


@dataclass(frozen=True)
class CleanBank:
    latents: np.ndarray
    ids: np.ndarray
    rows: np.ndarray
    labels: np.ndarray
    metadata: dict


def load_clean_bank(directory: Path, samples: int = 1000) -> CleanBank:
    """Read exactly ids 0..samples-1; ``rows`` identify the actual images."""
    directory = directory.expanduser().resolve()
    paths = sorted(directory.glob("clean_rank*.npz"))
    if not paths:
        raise FileNotFoundError(f"no clean_rank*.npz files in {directory}")
    selected = {}
    records = []
    for path in paths:
        with np.load(path) as archive:
            ids, rows, labels = (archive[key] for key in ("ids", "rows", "labels"))
            if ids.ndim != 1 or rows.shape != ids.shape or labels.shape != ids.shape:
                raise ValueError(f"invalid bank identities in {path}")
            take = np.flatnonzero((ids >= 0) & (ids < samples))
            for index in take:
                sample_id = int(ids[index])
                if sample_id in selected:
                    raise ValueError(f"duplicate sample id {sample_id}")
                selected[sample_id] = (int(rows[index]), int(labels[index]))
            records.append((path, take, ids[take].astype(np.int64)))
    if sorted(selected) != list(range(samples)):
        raise ValueError("bank must contain every requested id exactly once")
    latents = None
    source_hashes = {}
    for path, take, ids in records:
        with np.load(path) as archive:
            source = archive["latents"]
            if source.ndim != 4 or not np.issubdtype(source.dtype, np.floating):
                raise ValueError("clean latents must be floating [N,C,H,W]")
            if latents is None:
                latents = np.empty((samples, *source.shape[1:]), dtype=source.dtype)
            if source.shape[1:] != latents.shape[1:] or source.dtype != latents.dtype:
                raise ValueError("bank shards have inconsistent latent shapes or dtypes")
            latents[ids] = source[take]
        source_hashes[str(path)] = sha256_file(path)
    assert latents is not None
    ids = np.arange(samples, dtype=np.int64)
    rows = np.asarray([selected[int(i)][0] for i in ids], dtype=np.int64)
    labels = np.asarray([selected[int(i)][1] for i in ids], dtype=np.int64)
    if len(np.unique(rows)) != samples:
        raise ValueError("bank repeats real ImageNet rows")
    metadata = {"directory": str(directory), "samples": samples,
                "selection": "0 <= experiment_id < samples, ascending id",
                "normalization": "already normalized RAE encoder latents; no renormalization",
                "latent_shape": list(latents.shape[1:]), "stored_dtype": str(latents.dtype),
                "source_sha256": source_hashes,
                "ids_sha256": sha256_array(ids), "rows_sha256": sha256_array(rows),
                "labels_sha256": sha256_array(labels), "latents_sha256": sha256_array(latents)}
    manifest = directory.parent / "manifest.json"
    if manifest.exists():
        metadata["manifest"] = str(manifest)
        metadata["manifest_sha256"] = sha256_file(manifest)
    metadata["provenance_files"] = {}
    for name in ("request.json", "summary.json"):
        path = directory / name
        if path.exists():
            metadata["provenance_files"][name] = {
                "path": str(path), "sha256": sha256_file(path),
                "contents": json.loads(path.read_text()),
            }
    return CleanBank(latents, ids, rows, labels, metadata)


def validate_bank_pair(train: CleanBank, heldout: CleanBank, *, num_classes: int = 1000) -> None:
    if np.intersect1d(train.rows, heldout.rows).size:
        raise ValueError("training and held-out banks overlap in actual ImageNet rows")
    for bank in (train, heldout):
        if len(bank.ids) != num_classes or not np.array_equal(bank.labels, bank.ids % num_classes):
            raise ValueError("audit requires exactly one image per class, labels == ids % num_classes")
        if not np.array_equal(np.sort(bank.labels), np.arange(num_classes)):
            raise ValueError("audit bank does not cover every class exactly once")
    if train.latents.shape != heldout.latents.shape:
        raise ValueError("training and held-out latent shapes differ")


def time_indices(shard_index: int, num_shards: int) -> list[int]:
    if not 1 <= num_shards <= NUM_STEPS or not 0 <= shard_index < num_shards:
        raise ValueError("invalid time shard")
    return [index for index in range(NUM_STEPS) if index % num_shards == shard_index]


def coupled_step_predictions(model, clean, noise, labels, current: float, following: float):
    """Official IG clean arithmetic and Euler step, with a coupled true target."""
    if not 0 <= following < current <= 1:
        raise ValueError("step must satisfy 0 <= next < current <= 1")
    if clean.shape != noise.shape:
        raise ValueError("clean and noise tensors must share shape")
    clean, noise = clean.float(), noise.float()
    state = (1.0 - current) * clean + current * noise
    target = (1.0 - following) * clean + following * noise
    times = torch.full((len(clean),), current, device=state.device, dtype=torch.float32)
    full, base = model(state, times, context=labels, attn_mask=None)
    full, base = full.float(), base.float()
    if full.shape != state.shape or base.shape != state.shape:
        raise ValueError("RAEv2 heads do not match the latent shape")
    guided = full + 0.78 * (full - base) if 0.1 <= current <= 1.0 else full
    predicted = state - (current - following) * ((state - guided) / current)
    return predicted, target


def bank_batches(bank: CleanBank, batch_size: int, device: torch.device, seed: int, digest):
    # Reset at each time: every image reuses its own epsilon along the whole bridge.
    generator = torch.Generator(device=device).manual_seed(seed)
    for start in range(0, len(bank.ids), batch_size):
        stop = min(start + batch_size, len(bank.ids))
        clean = torch.from_numpy(bank.latents[start:stop]).to(device=device, dtype=torch.float32)
        labels = torch.from_numpy(bank.labels[start:stop]).to(device=device)
        noise = torch.randn(clean.shape, generator=generator, device=device, dtype=torch.float32)
        digest.update(noise.cpu().contiguous().numpy().tobytes())
        yield start, stop, clean, noise, labels


def summarize_step(rows: list[dict], fit, current: float, following: float, index: int) -> dict:
    summary = {"step_index": index, "time": current, "next_time": following,
               "train_count": fit.count, "test_count": len(rows),
               "positive_slope_fraction": float((fit.slope > 0).double().mean()),
               "slope_mean": float(fit.slope.mean()), "slope_max": float(fit.slope.max()),
               "offset_rms": float(fit.offset.square().mean().sqrt()),
               "train_J": float((fit.offset.square() + 2 * fit.slope * fit.covariance
                                  - fit.slope.square() * fit.variance).mean())}
    for key in ("risk_before", "risk_after", "risk_gain", "J", "nonexpansive_slack",
                "h_target_mse", "applied_correction_mse"):
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        summary[key] = float(values.mean())
        summary[key + "_se"] = float(values.std(ddof=1) / math.sqrt(len(values)))
    summary.update(uncorrected_rms=math.sqrt(summary["risk_before"]),
                   corrected_rms=math.sqrt(summary["risk_after"]),
                   positive_J_image_fraction=float(np.mean([row["J"] > 0 for row in rows])),
                   positive_gain_image_fraction=float(np.mean([row["risk_gain"] > 0 for row in rows])))
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-bank", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--heldout-bank", type=Path, default=DEFAULT_HELDOUT)
    parser.add_argument("--bank-provenance-audit", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=202609062)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args(argv)
    if args.samples != 1000 or args.batch_size <= 0:
        parser.error("use exactly 1000 images (one per class), with positive batch size")
    if not 0 <= args.seed < 2**63 - 1:
        parser.error("seed must satisfy 0 <= seed < 2**63-1")
    try:
        time_indices(args.shard_index, args.num_shards)
    except ValueError as error:
        parser.error(str(error))
    return args


def main():
    args = parse_args()
    for key in ("output_dir", "train_bank", "heldout_bank", "config", "checkpoint", "bank_provenance_audit"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    root = args.output_dir.expanduser().resolve()
    out = root / f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing audit: {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "coefficients").mkdir()
    indices = time_indices(args.shard_index, args.num_shards)
    request = {"protocol": PROTOCOL, **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
               "time_indices": indices, "num_steps": NUM_STEPS, "ig_scale": 1.78, "ig_interval": [0.1, 1.0],
               "correction_multiplier": 1.0, "correction_time_selection": "none",
               "fit_input": "true_next_state", "fit_error": "frozen_euler_next_minus_true_next",
               "moment_arithmetic": "fp64_centered_merge", "applied_coefficient_arithmetic": "fp32",
               "metric_arithmetic": "fp32_products_then_fp64_per_image_reduction",
               "guidance_and_step_arithmetic": "fp32", "forward_layout": "single_conditional_batch",
               "noise_schema": "independent split generators seed and seed+1, reset at each time; fixed batch shape",
               "noise_seed_train": args.seed, "noise_seed_heldout": args.seed + 1,
               "torch_version": str(torch.__version__), "status": "initializing"}
    atomic_json(out / "request.json", request)
    started = time.perf_counter()
    atomic_json(out / "progress.json", {"status": "loading_banks", "completed_time_indices": [], "total_time_indices": indices})
    train = load_clean_bank(args.train_bank, args.samples)
    heldout = load_clean_bank(args.heldout_bank, args.samples)
    validate_bank_pair(train, heldout)
    if args.bank_provenance_audit.exists():
        request["bank_provenance"] = {
            "audit_path": str(args.bank_provenance_audit),
            "audit_sha256": sha256_file(args.bank_provenance_audit),
            "audit_contents": json.loads(args.bank_provenance_audit.read_text()),
            "interpretation": "Historical A passed current encoder checks; original B differed. The separately identified heldout_bank must be the replacement bank, not original B.",
        }
    from experiments.sample_raev2_pfr_retiming import load_config, shifted_time_grid
    from utils.model_utils import instantiate_from_config
    config = load_config(args.config.expanduser().resolve())
    if (config.conditioning.type != "label" or config.transport.prediction != "x"
            or config.guidance.ig.scale != 1.78 or config.guidance.ig.t_min != 0.1
            or config.guidance.ig.t_max != 1.0 or config.guidance.cfg.scale != 1.0
            or config.sampler.num_steps != NUM_STEPS):
        raise ValueError("configuration is not the locked official 100-step IG protocol")
    if tuple(config.misc.latent_size) != train.latents.shape[1:]:
        raise ValueError("bank shape does not match official latent coordinates")
    request.update(train_bank_metadata=train.metadata, heldout_bank_metadata=heldout.metadata,
                   config_sha256=sha256_file(args.config), checkpoint_sha256=sha256_file(args.checkpoint),
                   normalization_stats_sha256=sha256_file(Path(config.stage_1.params["normalization_stat_path"])),
                   source_sha256={str(path.relative_to(ROOT)): sha256_file(path) for path in (
                       Path(__file__), ROOT / "experiments/raev2_proximal_calibration.py",
                       ROOT / "experiments/sample_raev2_pfr_retiming.py",
                       ROOT / "external/RAEv2/src/stage2/models/DDT.py",
                       ROOT / "external/RAEv2/src/stage2/models/model_utils.py")})
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = args.precision == "bf16"
    torch.backends.cudnn.allow_tf32 = args.precision == "bf16"
    request.update(tf32=args.precision == "bf16", cuda_device=torch.cuda.get_device_name(device))
    shift = math.sqrt((config.misc.time_dist_shift_dim or math.prod(config.misc.latent_size)) / config.misc.time_dist_shift_base)
    grid = shifted_time_grid(NUM_STEPS, shift, device).cpu().tolist()
    request.update(time_grid=grid, time_shift=shift, status="loading_checkpoint")
    atomic_json(out / "request.json", request)
    model = instantiate_from_config(config.stage_2).to(device).eval().requires_grad_(False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    model.load_state_dict(checkpoint["ema"], strict=True)
    request.update(checkpoint_step=int(checkpoint.get("step", 0)), state_key="ema", status="auditing")
    del checkpoint
    atomic_json(out / "request.json", request)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if args.precision == "bf16" else nullcontext()
    step_rows, all_rows, coefficient_records = [], [], []
    noise_fingerprints = {}
    model_calls = 0
    with torch.inference_mode(), autocast:
        for index in indices:
            current, following = grid[index], grid[index + 1]
            atomic_json(out / "progress.json", {"status": "training_moments", "step_index": index,
                        "completed_time_indices": [row["step_index"] for row in step_rows], "total_time_indices": indices})
            moments = DiagonalMoments()
            train_digest, heldout_digest = hashlib.sha256(), hashlib.sha256()
            for _, _, clean, noise, labels in bank_batches(train, args.batch_size, device, args.seed, train_digest):
                predicted, target = coupled_step_predictions(model, clean, noise, labels, current, following)
                model_calls += 1
                moments.update(target, predicted - target)
            fit64 = moments.fit()
            fit32 = fit64.to(device=device, dtype=torch.float32)
            record = {"step_index": index, "time": current, "next_time": following,
                      "source_fp64": fit64.to(device="cpu", dtype=torch.float64).state_dict(),
                      "applied_fp32": fit32.to(device="cpu", dtype=torch.float32).state_dict()}
            atomic_torch_save(out / "coefficients" / f"step_{index:03d}.pt", record)
            coefficient_records.append(record)
            rows = []
            atomic_json(out / "progress.json", {"status": "heldout_validation", "step_index": index,
                        "completed_time_indices": [row["step_index"] for row in step_rows], "total_time_indices": indices})
            for start, stop, clean, noise, labels in bank_batches(heldout, args.batch_size, device, args.seed + 1, heldout_digest):
                predicted, target = coupled_step_predictions(model, clean, noise, labels, current, following)
                model_calls += 1
                metrics = {key: value.cpu().numpy() for key, value in heldout_step_metrics(predicted, target, fit32).items()}
                if any(not np.isfinite(value).all() for value in metrics.values()):
                    raise FloatingPointError(f"nonfinite held-out metric at time index {index}")
                for local, sample in enumerate(range(start, stop)):
                    rows.append({"step_index": index, "sample_id": int(heldout.ids[sample]), "source_row": int(heldout.rows[sample]),
                                 "label": int(heldout.labels[sample]), **{key: float(value[local]) for key, value in metrics.items()}})
            fingerprint = {"train_noise_sha256": train_digest.hexdigest(), "heldout_noise_sha256": heldout_digest.hexdigest()}
            if noise_fingerprints and fingerprint != noise_fingerprints:
                raise RuntimeError("noise coupling changed across official time indices")
            noise_fingerprints = fingerprint
            step_rows.append(summarize_step(rows, fit64, current, following, index))
            all_rows.extend(rows)
            write_csv(out / "step_metrics.csv", step_rows)
            write_csv(out / "heldout_per_image.csv", all_rows)
            elapsed = time.perf_counter() - started
            progress = {"status": "auditing", "completed_time_indices": [row["step_index"] for row in step_rows],
                        "total_time_indices": indices, "elapsed_seconds": elapsed, **noise_fingerprints,
                        "last_step": step_rows[-1]}
            atomic_json(out / "progress.json", progress)
            print(json.dumps(progress), flush=True)
            del moments, fit64, fit32
    atomic_torch_save(out / "calibration.pt", {"protocol": PROTOCOL,
                "steps": {record["step_index"]: record["source_fp64"] for record in coefficient_records},
                "applied_fp32_steps": {record["step_index"]: record["applied_fp32"] for record in coefficient_records},
                "request": request})
    summary = {"protocol": PROTOCOL, "complete": True, "time_indices": indices, **noise_fingerprints,
               "samples_per_split": args.samples, "full_model_calls": model_calls,
               "full_sample_evaluations": 2 * args.samples * len(indices),
               "calibration_sha256": sha256_file(out / "calibration.pt"), "elapsed_seconds": time.perf_counter() - started,
               "max_memory_allocated_bytes": torch.cuda.max_memory_allocated()}
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "progress.json", {**progress, "status": "complete", "complete": True})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
