#!/usr/bin/env python3
"""Measure RAEv2 guidance sensitivity to the encoder's known normal noise.

Diagnostic only: the rollout always uses untouched official IG predictions.
Teacher-forced errors use their true paired clean target; rollout distances to
that same source image are only reference-pair distances, not denoising risk.
No decoding, FID, fitting, time selection, or sampler intervention is performed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "external/RAEv2/src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.audit_raev2_proximal_calibration import (
    DEFAULT_CONFIG, DEFAULT_CHECKPOINT, atomic_json, atomic_torch_save, load_clean_bank,
    sha256_file, write_csv,
)

PROTOCOL = "raev2_affine_normal_noise_mechanism_audit_v1"
DEFAULT_BANK = Path("/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/heldout_clean_c_current_fp32")
STEPS, BATCH_SIZE, SAMPLES = 100, 8, 32


def affine_geometry(mean: torch.Tensor, variance: torch.Tensor):
    """Per-position unit normal u and closest affine point c, computed in FP64."""
    if mean.ndim != 3 or mean.shape != variance.shape:
        raise ValueError("mean and variance must be matching [C,H,W] tensors")
    if not bool(torch.isfinite(mean).all() and torch.isfinite(variance).all()):
        raise ValueError("nonfinite normalization statistics")
    if bool((variance < 0).any()):
        raise ValueError("normalization variance must be nonnegative")
    normal = (variance.double() + 1e-5).sqrt()
    norm = normal.square().sum(0, keepdim=True).sqrt()
    unit = normal / norm
    anchor = -mean.double().sum(0, keepdim=True) / norm * unit
    return unit, anchor


def normal_part(value: torch.Tensor, unit: torch.Tensor):
    return (value * unit).sum(1, keepdim=True) * unit


def project_clean(value: torch.Tensor, unit: torch.Tensor, anchor: torch.Tensor):
    return value - normal_part(value - anchor, unit)


def mirror_normal_noise(state: torch.Tensor, current: float,
                        unit: torch.Tensor, anchor: torch.Tensor):
    """Reflect around (1-t)c, preserving all tangential state components."""
    return state - 2 * normal_part(state - (1 - current) * anchor, unit)


def official_guided(full: torch.Tensor, base: torch.Tensor, current: float):
    if full.dtype != torch.float32 or base.dtype != torch.float32:
        raise ValueError("raw model heads must first be promoted to FP32")
    return full + 0.78 * (full - base) if 0.1 <= current <= 1 else full


def energy(value: torch.Tensor):
    return value.square().flatten(1).mean(1)


def fraction(numerator: torch.Tensor, denominator: torch.Tensor):
    # Exact zero denominators indicate absent energy, with fraction defined as 0.
    return torch.where(denominator > 0, numerator / denominator.clamp_min(1e-300), 0)


def domain_metrics(state, clean, full, base, reflected_full, reflected_base,
                   current: float, unit: torch.Tensor, anchor: torch.Tensor):
    """All reductions and derived diagnostic predictions are evaluated in FP64."""
    guided = official_guided(full, base, current).double()
    reflected_guided = official_guided(reflected_full, reflected_base, current).double()
    state, clean, full, base, reflected_full, reflected_base = (
        value.double() for value in (state, clean, full, base, reflected_full, reflected_base)
    )
    gap = full - base
    gap_energy = energy(gap)
    metrics = {
        "clean_energy": energy(clean),
        "clean_affine_normal_energy": energy(normal_part(clean-anchor, unit)),
        "state_normal_noise_energy": energy(normal_part(state-(1-current)*anchor, unit)),
        "gap_energy": gap_energy,
        "gap_normal_energy": energy(normal_part(gap, unit)),
    }
    metrics["gap_normal_fraction"] = fraction(metrics["gap_normal_energy"], gap_energy)
    metrics["clean_affine_normal_fraction"] = fraction(
        metrics["clean_affine_normal_energy"], energy(clean-anchor))
    for name, prediction, reflected in (("full", full, reflected_full),
                                         ("base", base, reflected_base),
                                         ("ig", guided, reflected_guided)):
        error = prediction-clean
        normal_error = normal_part(error, unit)
        normal_prediction = normal_part(prediction-anchor, unit)
        change = reflected-prediction
        tangent_change = change-normal_part(change, unit)
        average = (prediction+reflected)*0.5
        average_projected = project_clean(average, unit, anchor)
        original_projected = project_clean(prediction, unit, anchor)
        before = energy(error)
        values = {
            "energy": energy(prediction),
            "affine_normal_energy": energy(normal_prediction),
            "affine_normal_fraction": fraction(energy(normal_prediction), energy(prediction-anchor)),
            "reference_error_energy": before,
            "reference_error_normal_energy": energy(normal_error),
            "reference_error_normal_fraction": fraction(energy(normal_error), before),
            "mirror_change_energy": energy(change),
            "mirror_tangent_change_energy": energy(tangent_change),
            "mirror_tangent_change_over_gap_energy": fraction(energy(tangent_change), gap_energy),
            "projected_reference_error_energy": energy(original_projected-clean),
            "average_reference_error_energy": energy(average-clean),
            "average_projected_reference_error_energy": energy(average_projected-clean),
            "average_projected_reference_error_gain": before-energy(average_projected-clean),
            "average_projected_reference_error_relative_gain": fraction(
                before-energy(average_projected-clean), before),
        }
        metrics.update({f"{name}_{key}": value for key, value in values.items()})
    return metrics


def summarize_rows(rows):
    identifiers = {"domain", "sample_id", "source_row", "label", "step_index", "time", "next_time"}
    output = []
    for domain in ("teacher", "rollout"):
        for step in range(STEPS):
            selected = [row for row in rows if row["domain"] == domain and row["step_index"] == step]
            if not selected:
                continue
            record = {"domain": domain, "step_index": step, "samples": len(selected),
                      "time": selected[0]["time"], "next_time": selected[0]["next_time"]}
            record.update({key: float(np.mean([r[key] for r in selected]))
                           for key in selected[0] if key not in identifiers})
            record["pooled_gap_normal_fraction"] = (
                record["gap_normal_energy"] / record["gap_energy"] if record["gap_energy"] > 0 else 0)
            for head in ("full", "ig"):
                denominator = record[f"{head}_reference_error_energy"]
                record[f"{head}_pooled_average_projected_reference_error_relative_gain"] = (
                    record[f"{head}_average_projected_reference_error_gain"] / denominator
                    if denominator > 0 else 0)
            output.append(record)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--seed", type=int, default=202609071)
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**63:
        parser.error("seed must satisfy 0 <= seed < 2**63")
    for key in ("output_dir", "bank", "config", "checkpoint"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    request = {"protocol": PROTOCOL,
               **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               "sample_count": SAMPLES, "batch_size": BATCH_SIZE, "num_steps": STEPS,
               "state_key": "ema", "precision": "bf16", "tf32": True,
               "model_parameters_and_states": "fp32", "diagnostic_reductions": "fp64",
               "reflection_arithmetic": "fp32 with FP64-derived per-position geometry cast to FP32",
               "ig_scale": 1.78, "ig_interval": [0.1, 1.0],
               "noise_schema": "one CUDA Generator(seed), sequential B8 FP32 batches; same epsilon for teacher bridge and original rollout initialization",
               "rollout_update": "untouched original official IG Euler only; mirror forwards are diagnostic",
               "domain_interpretation": {"teacher": "paired denoising squared error against the true clean source",
                    "rollout": "distance to original paired source only; NOT posterior denoising risk"},
               "no_decode": True, "no_fid": True, "no_fitting": True,
               "torch_version": str(torch.__version__), "status": "initializing"}
    source_paths = [Path(__file__), ROOT / "experiments/audit_raev2_proximal_calibration.py",
                    ROOT / "experiments/sample_raev2_pfr_retiming.py",
                    ROOT / "external/RAEv2/src/stage2/models/DDT.py",
                    ROOT / "external/RAEv2/src/stage2/models/model_utils.py",
                    ROOT / "external/RAEv2/src/encoders/vision_encoder.py",
                    ROOT / "external/RAEv2/src/stage1/rae.py"]
    request["source_sha256"] = {str(p.relative_to(ROOT)): sha256_file(p) for p in source_paths}
    (out / "runner_source.py").write_bytes(Path(__file__).read_bytes())
    atomic_json(out / "request.json", request)
    atomic_json(out / "progress.json", {"status": "loading", "completed_samples": 0})
    from experiments.sample_raev2_pfr_retiming import load_config, shifted_time_grid
    from utils.model_utils import instantiate_from_config
    config = load_config(args.config)
    if (config.conditioning.type != "label" or config.transport.prediction != "x"
            or config.guidance.ig.scale != 1.78 or config.guidance.ig.t_min != 0.1
            or config.guidance.ig.t_max != 1.0 or config.guidance.cfg.scale != 1.0
            or config.sampler.num_steps != STEPS):
        raise ValueError("configuration is not the frozen official IG protocol")
    bank = load_clean_bank(args.bank, samples=SAMPLES)
    if tuple(bank.latents.shape[1:]) != tuple(config.misc.latent_size):
        raise ValueError("clean bank latent shape differs from the model")
    if not np.array_equal(bank.labels, bank.ids):
        raise ValueError("expected the first 32 class-balanced bank identities")
    stats_path = Path(config.stage_1.params["normalization_stat_path"])
    stats = torch.load(stats_path, map_location="cpu", weights_only=True)
    unit64, anchor64 = affine_geometry(stats["mean"], stats["var"])
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    unit64, anchor64 = unit64.to(device), anchor64.to(device)
    unit32, anchor32 = unit64.float(), anchor64.float()
    shift = math.sqrt((config.misc.time_dist_shift_dim or math.prod(config.misc.latent_size))
                      / config.misc.time_dist_shift_base)
    grid = shifted_time_grid(STEPS, shift, device).cpu().tolist()
    snapshot_indices = sorted({0} | {int(np.argmin(np.abs(np.asarray(grid[:-1])-q)))
                                    for q in np.linspace(.1, .9, 9)})
    if float(config.transport.t_eps) >= grid[-2]:
        raise ValueError("transport denominator floor changes the frozen Euler update")
    request.update(config_sha256=sha256_file(args.config), checkpoint_sha256=sha256_file(args.checkpoint),
                   normalization_stats_path=str(stats_path), normalization_stats_sha256=sha256_file(stats_path),
                   bank_metadata=bank.metadata, sample_ids=bank.ids.tolist(), source_rows=bank.rows.tolist(),
                   labels=bank.labels.tolist(), time_grid=grid, time_shift=shift,
                   snapshot_indices=snapshot_indices, snapshot_samples="first batch of 8 only",
                   snapshot_contents="untouched original teacher/rollout states and full/base heads, all CPU FP32",
                   normalization_shape=list(stats["mean"].shape), normalization_eps=1e-5,
                   cuda_device=torch.cuda.get_device_name(device), latent_shape=list(config.misc.latent_size))
    atomic_json(out / "request.json", request)
    model = instantiate_from_config(config.stage_2).to(device).eval().requires_grad_(False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    model.load_state_dict(checkpoint["ema"], strict=True)
    request.update(checkpoint_step=int(checkpoint.get("step", 0)), status="auditing")
    del checkpoint
    atomic_json(out / "request.json", request)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    noise_digest = hashlib.sha256()
    rows, model_calls, completed = [], 0, 0
    snapshots = []
    (out / "states").mkdir()
    csv_file = (out / "per_sample_step.csv").open("w", newline="")
    writer = None

    def predict(state, labels, current):
        times = torch.full((len(state),), current, device=device, dtype=torch.float32)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            full, base = model(state, times, context=labels, attn_mask=None)
        if full.shape != state.shape or base.shape != state.shape:
            raise ValueError("model heads changed shape")
        return full.float(), base.float()

    try:
        with torch.inference_mode():
            for start in range(0, SAMPLES, BATCH_SIZE):
                stop = start + BATCH_SIZE
                clean = torch.from_numpy(bank.latents[start:stop]).to(device=device, dtype=torch.float32)
                labels = torch.from_numpy(bank.labels[start:stop]).to(device)
                noise = torch.randn(clean.shape, device=device, dtype=torch.float32, generator=generator)
                noise_digest.update(noise.cpu().contiguous().numpy().tobytes())
                rollout = noise.clone()
                for step, (current, following) in enumerate(zip(grid[:-1], grid[1:])):
                    teacher = (1-current)*clean + current*noise
                    snapshot = ({"step_index": step, "t": current,
                                 "sample_ids": torch.from_numpy(bank.ids[start:stop].copy()),
                                 "labels": labels.cpu()}
                                if start == 0 and step in snapshot_indices else None)
                    for domain, state in (("teacher", teacher), ("rollout", rollout)):
                        full, base = predict(state, labels, current)
                        if snapshot is not None:
                            snapshot[domain] = {"state": state.cpu(), "full": full.cpu(), "base": base.cpu()}
                        reflected_state = mirror_normal_noise(state, current, unit32, anchor32)
                        reflected_full, reflected_base = predict(reflected_state, labels, current)
                        model_calls += 2
                        values = domain_metrics(state, clean, full, base, reflected_full, reflected_base,
                                                current, unit64, anchor64)
                        arrays = {key: value.cpu().numpy() for key, value in values.items()}
                        if any(not np.isfinite(value).all() for value in arrays.values()):
                            raise FloatingPointError("nonfinite diagnostic statistic")
                        batch_rows = [{"domain": domain, "sample_id": int(bank.ids[index]),
                                       "source_row": int(bank.rows[index]), "label": int(bank.labels[index]),
                                       "step_index": step, "time": current, "next_time": following,
                                       **{key: float(value[local]) for key, value in arrays.items()}}
                                      for local, index in enumerate(range(start, stop))]
                        if writer is None:
                            writer = csv.DictWriter(csv_file, fieldnames=list(batch_rows[0]))
                            writer.writeheader()
                        writer.writerows(batch_rows)
                        rows.extend(batch_rows)
                        if domain == "rollout":
                            guided = official_guided(full, base, current)
                            rollout = state - (current-following)*((state-guided)/current)
                    if snapshot is not None:
                        snapshot_path = out / "states" / f"step_{step:03d}.pt"
                        atomic_torch_save(snapshot_path, snapshot)
                        snapshots.append({"step_index": step, "t": current,
                                          "path": str(snapshot_path), "sha256": sha256_file(snapshot_path)})
                    if (step+1) % 10 == 0:
                        csv_file.flush()
                        progress = {"status": "auditing", "completed_samples": completed,
                                    "batch_start": start, "completed_steps_in_batch": step+1,
                                    "model_batch_calls": model_calls,
                                    "elapsed_seconds": time.perf_counter()-started}
                        atomic_json(out / "progress.json", progress)
                        print(json.dumps(progress), flush=True)
                completed = stop
                write_csv(out / "per_step_summary.csv", summarize_rows(rows))
        csv_file.close()
        summary = {"protocol": PROTOCOL, "complete": True, "sample_count": SAMPLES,
                   "per_sample_step_rows": len(rows), "model_batch_calls": model_calls,
                   "model_sample_evaluations": model_calls*BATCH_SIZE,
                   "noise_sha256": noise_digest.hexdigest(),
                   "snapshots": snapshots,
                   "per_sample_step_sha256": sha256_file(out / "per_sample_step.csv"),
                   "elapsed_seconds": time.perf_counter()-started,
                   "no_decode": True, "no_fid": True, "no_fitting": True,
                   "rollout_intervention": False,
                   "interpretation": "teacher MSE and encoder constraint diagnostics only; no terminal quality guarantee"}
        if model_calls != 4*STEPS*(SAMPLES//BATCH_SIZE) or len(rows) != 2*STEPS*SAMPLES:
            raise RuntimeError("incomplete audit counts")
        atomic_json(out / "summary.json", summary)
        atomic_json(out / "progress.json", {**summary, "status": "complete"})
        print(json.dumps(summary), flush=True)
    except BaseException as exc:
        csv_file.close()
        atomic_json(out / "progress.json", {"status": "failed", "completed_samples": completed,
                    "model_batch_calls": model_calls, "error": repr(exc),
                    "elapsed_seconds": time.perf_counter()-started})
        raise


if __name__ == "__main__":
    main()
