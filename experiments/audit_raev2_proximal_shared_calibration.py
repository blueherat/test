#!/usr/bin/env python3
"""Audit the fixed channel-shared cone on a fresh real-image bank C.

Training sufficient statistics come exclusively from the original bank A.
The structural revision was motivated by diagonal failure on bank B; C has
not been used to choose coefficients, times, or image identities. The original
diagonal correction is also recorded as a diagnostic using the same forward.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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

from experiments.audit_raev2_proximal_calibration import (
    DEFAULT_TRAIN, atomic_json, atomic_torch_save, bank_batches,
    coupled_step_predictions, load_clean_bank, sha256_file, summarize_step,
    time_indices, validate_bank_pair, write_csv,
)
from experiments.raev2_proximal_calibration import DiagonalFit, heldout_step_metrics
from experiments.raev2_proximal_shared_calibration import pool_channel_fit

PROTOCOL = "raev2_channel_shared_proximal_real_coupling_audit_v1"
DEFAULT_C = Path("/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/heldout_clean_c_current_fp32")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--heldout-bank", type=Path, default=DEFAULT_C)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--seed", type=int, default=202609065)
    args = parser.parse_args()
    indices = time_indices(args.shard_index, args.num_shards)
    parent_path = args.calibration.resolve()
    parent = torch.load(parent_path, map_location="cpu", weights_only=True)
    if set(parent["steps"]) != set(range(100)):
        raise ValueError("parent must contain all 100 official steps")
    original = parent["source_requests"][0]
    for relative, expected in original["source_sha256"].items():
        if sha256_file(ROOT / relative) != expected:
            raise ValueError(f"source changed since parent audit: {relative}")
    for key, hash_key in (("config", "config_sha256"), ("checkpoint", "checkpoint_sha256")):
        if sha256_file(Path(original[key])) != original[hash_key]:
            raise ValueError(f"changed {key}")
    out = args.output_dir.resolve() / f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    # Fit the one fixed function class before reading any C latent.
    fits64 = {i: pool_channel_fit(DiagonalFit.from_state_dict(parent["steps"][i])) for i in indices}
    diagonal32 = {i: DiagonalFit.from_state_dict(parent["applied_fp32_steps"][i]) for i in indices}
    request = dict(original)
    request.update(protocol=PROTOCOL, output_dir=str(args.output_dir.resolve()),
                   heldout_bank=str(args.heldout_bank.resolve()),
                   shard_index=args.shard_index, num_shards=args.num_shards, time_indices=indices,
                   fit_structure="channel_shared_affine", seed=args.seed,
                   noise_seed_heldout=args.seed,
                   noise_schema="fresh C generator reset at each time; fixed batch shape; A training noise inherited",
                   parent_calibration_path=str(parent_path), parent_calibration_sha256=sha256_file(parent_path),
                   original_train_request=original, status="loading_fresh_bank",
                   selection_history="channel sharing chosen after diagonal failure on B; C is fresh confirmation",
                   source_sha256={**original["source_sha256"], **{
                       str(path.relative_to(ROOT)): sha256_file(path) for path in (
                           Path(__file__), ROOT / "experiments/raev2_proximal_shared_calibration.py")}})
    atomic_json(out / "request.json", request)
    atomic_json(out / "progress.json", {"status": "loading_fresh_bank", "completed_time_indices": []})
    train = load_clean_bank(Path(original["train_bank"]))
    heldout = load_clean_bank(args.heldout_bank)
    validate_bank_pair(train, heldout)
    if train.metadata != original["train_bank_metadata"]:
        raise ValueError("training bank changed")
    del train
    request["heldout_bank_metadata"] = heldout.metadata
    request["bank_provenance"] = heldout.metadata["provenance_files"]
    from experiments.sample_raev2_pfr_retiming import load_config, shifted_time_grid
    from utils.model_utils import instantiate_from_config
    config = load_config(Path(original["config"]))
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = original["tf32"]
    torch.backends.cudnn.allow_tf32 = original["tf32"]
    request["cuda_device"] = torch.cuda.get_device_name(device)
    shift = math.sqrt((config.misc.time_dist_shift_dim or math.prod(config.misc.latent_size)) / config.misc.time_dist_shift_base)
    grid = shifted_time_grid(100, shift, device).cpu().tolist()
    if grid != original["time_grid"]:
        raise ValueError("official time grid changed")
    model = instantiate_from_config(config.stage_2).to(device).eval().requires_grad_(False)
    checkpoint = torch.load(original["checkpoint"], map_location="cpu", weights_only=False, mmap=True)
    model.load_state_dict(checkpoint["ema"], strict=True)
    del checkpoint
    original_train_noise = json.loads((Path(parent["source_shards"][0]["path"]) / "summary.json").read_text())["train_noise_sha256"]
    del parent
    request["status"] = "auditing"
    atomic_json(out / "request.json", request)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if original["precision"] == "bf16" else nullcontext()
    all_rows, step_rows, applied = [], [], {}
    fingerprint = None
    with torch.inference_mode(), autocast:
        for index in indices:
            current, following = grid[index], grid[index+1]
            fit64 = fits64[index]
            fit32 = fit64.to(device=device, dtype=torch.float32)
            diag = diagonal32[index].to(device=device, dtype=torch.float32)
            applied[index] = fit32.to(device="cpu").state_dict()
            digest, rows = hashlib.sha256(), []
            atomic_json(out / "progress.json", {"status": "heldout_validation", "step_index": index,
                        "completed_time_indices": [r["step_index"] for r in step_rows]})
            for start, stop, clean, noise, labels in bank_batches(heldout, original["batch_size"], device, args.seed, digest):
                predicted, target = coupled_step_predictions(model, clean, noise, labels, current, following)
                metrics = {k: v.cpu().numpy() for k, v in heldout_step_metrics(predicted, target, fit32).items()}
                diag_metrics = heldout_step_metrics(predicted, target, diag)
                for key in ("risk_after", "risk_gain", "J"):
                    metrics["diagonal_"+key] = diag_metrics[key].cpu().numpy()
                if any(not np.isfinite(value).all() for value in metrics.values()):
                    raise FloatingPointError("nonfinite C metrics")
                for local, sample in enumerate(range(start, stop)):
                    rows.append({"step_index": index, "sample_id": int(heldout.ids[sample]),
                                 "source_row": int(heldout.rows[sample]), "label": int(heldout.labels[sample]),
                                 **{k: float(v[local]) for k, v in metrics.items()}})
            if fingerprint is not None and fingerprint != digest.hexdigest():
                raise ValueError("noise coupling changed across time")
            fingerprint = digest.hexdigest()
            all_rows.extend(rows)
            summary = summarize_step(rows, fit64, current, following, index)
            summary["diagonal_risk_gain"] = float(np.mean([r["diagonal_risk_gain"] for r in rows]))
            step_rows.append(summary)
            write_csv(out / "heldout_per_image.csv", all_rows)
            write_csv(out / "step_metrics.csv", step_rows)
            progress = {"status": "auditing", "completed_time_indices": [r["step_index"] for r in step_rows],
                        "elapsed_seconds": time.perf_counter()-started, "last_step": summary}
            atomic_json(out / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    atomic_torch_save(out / "calibration.pt", {"protocol": PROTOCOL,
                      "steps": {i: fit.state_dict() for i, fit in fits64.items()},
                      "applied_fp32_steps": applied, "request": request})
    summary = {"protocol": PROTOCOL, "complete": True, "time_indices": indices,
               "heldout_noise_sha256": fingerprint, "train_noise_sha256": original_train_noise,
               "calibration_sha256": sha256_file(out / "calibration.pt"),
               "elapsed_seconds": time.perf_counter()-started,
               "full_sample_evaluations": len(indices)*len(heldout.ids),
               "samples_per_split": 1000, "no_fid_evaluated": True}
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "progress.json", {**progress, "status": "complete", "complete": True})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
