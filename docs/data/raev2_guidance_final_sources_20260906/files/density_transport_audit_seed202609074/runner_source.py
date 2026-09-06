#!/usr/bin/env python3
"""Test linear-decontamination prerequisites along eight unmodified IG paths.

Two independent fresh Hutchinson probes transport two estimates of the same
log marginal density ratio. An exact directional VJP checks a necessary
posterior-covariance inequality. Neither estimate controls sampling. Every
inner product and trace is a full coordinate sum; no density temperature,
clipping, sample rejection, or temporal guidance tuning is used.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "external/RAEv2/src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.audit_raev2_proximal_calibration import atomic_json, sha256_file, write_csv
from experiments.sample_raev2_proximal_calibration import DEFAULT_CONFIG, DEFAULT_CHECKPOINT
from experiments.sample_raev2_pfr_retiming import load_config, shifted_time_grid
from experiments.raev2_density_decontamination import frozen_coefficients_from_fields


def inner(first, second):
    return (first.double()*second.double()).flatten(1).sum(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=202609074)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config.resolve())
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source_paths = (Path(__file__), ROOT / "experiments/raev2_density_decontamination.py",
                    ROOT / "experiments/sample_raev2_pfr_retiming.py",
                    ROOT / "external/RAEv2/src/stage2/models/DDT.py",
                    ROOT / "external/RAEv2/src/stage2/models/model_utils.py")
    request = {"protocol": "raev2_density_transport_mixture_prerequisite_v1",
               "seed": args.seed, "samples": 8, "num_steps": 100, "w": 1.78,
               "classes": list(range(8)), "precision": "FP32_no_autocast_no_TF32_B8",
               "trajectory": "unmodified official IG formula and shifted Euler grid; FP32 forward",
               "config": str(args.config.resolve()), "config_sha256": sha256_file(args.config),
               "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256_file(args.checkpoint),
               "source_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in source_paths},
               "probe_seeds": [args.seed+1000000007, args.seed+2000000011],
               "probe_distribution": "two independent fresh Rademacher probes per step; full coordinate SUM",
               "density_update": "FP64 frozen-coefficient integral along actual official guidance; two estimates; no feedback",
               "covariance_test": "v^T(JF-u*JB)v >= (1-t)/t^2*u/(1-u)*||F-B||^2; v=detached unit gap",
               "no_candidate_sampling": True, "no_decode": True, "no_fid": True,
               "no_sample_selection": True, "torch_version": str(torch.__version__)}
    atomic_json(out / "request.json", request)
    (out / "runner_source.py").write_bytes(Path(__file__).read_bytes())
    atomic_json(out / "progress.json", {"status": "initializing", "rows": 0})
    from utils.model_utils import instantiate_from_config
    model = instantiate_from_config(config.stage_2).to(device).eval().requires_grad_(False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    model.load_state_dict(checkpoint["ema"], strict=True)
    request["checkpoint_step"] = int(checkpoint.get("step", 0))
    del checkpoint
    shift = math.sqrt((config.misc.time_dist_shift_dim or math.prod(config.misc.latent_size))
                      / config.misc.time_dist_shift_base)
    grid = shifted_time_grid(100, shift, device).cpu().tolist()
    request.update(time_shift=shift, time_grid=grid, cuda_device=torch.cuda.get_device_name(device))
    initial_generator = torch.Generator(device=device).manual_seed(args.seed)
    probe_generators = [torch.Generator(device=device).manual_seed(seed) for seed in request["probe_seeds"]]
    state = torch.randn((8, *config.misc.latent_size), device=device, generator=initial_generator)
    labels = torch.arange(8, device=device)
    import hashlib
    request["initial_noise_sha256"] = hashlib.sha256(state.cpu().numpy().tobytes()).hexdigest()
    atomic_json(out / "request.json", request)
    ell = torch.zeros((2, 8), device=device, dtype=torch.float64)
    limit = math.log(1.78/.78)
    rows = []
    started = time.perf_counter()
    try:
        for index, (t, s) in enumerate(zip(grid[:-1], grid[1:])):
            state = state.detach().requires_grad_(True)
            times = torch.full((8,), t, device=device, dtype=torch.float32)
            full, base = model(state, times, context=labels, attn_mask=None)
            gap = full-base
            gap_squared = inner(gap.detach(), gap.detach())
            if bool((gap_squared == 0).any()):
                raise ValueError("zero gap prevents the preregistered directional covariance test")
            direction = (gap.detach().double()/gap_squared.sqrt().view(8, 1, 1, 1)).float()
            divs = []
            for generator in probe_generators:
                probe = torch.empty_like(state).bernoulli_(.5, generator=generator).mul_(2).sub_(1)
                vjp = torch.autograd.grad((gap*probe).sum(), state, retain_graph=True)[0]
                divs.append(inner(vjp, probe).detach())
            full_vjp = torch.autograd.grad((full*direction).sum(), state, retain_graph=True)[0]
            base_vjp = torch.autograd.grad((base*direction).sum(), state)[0]
            qfull, qbase = inner(full_vjp, direction).detach(), inner(base_vjp, direction).detach()
            gamma = .78 if .1 <= t <= 1 else 0.
            ell_before = ell.clone()
            drift_increments = []
            with torch.no_grad():
                for probe_index, divergence in enumerate(divs):
                    if s > 0:
                        coef = frozen_coefficients_from_fields(state, full, base, divergence, t, s)
                        increment = coef.A-coef.Bcoef*gamma
                        ell[probe_index] += increment
                    else:
                        increment = torch.zeros_like(ell[probe_index])
                    drift_increments.append(increment)
                for slot in range(8):
                    row = {"step_index": index, "t": t, "next_t": s, "sample_id": slot,
                           "label": slot, "official_extra_gain": gamma,
                           "gap_squared_norm": float(gap_squared[slot]),
                           "q_full_gap_direction": float(qfull[slot]),
                           "q_base_gap_direction": float(qbase[slot])}
                    for probe_index in range(2):
                        value = float(ell_before[probe_index, slot])
                        valid = value < limit
                        prefix = f"probe{probe_index}"
                        row.update({f"{prefix}_ell_before": value,
                                    f"{prefix}_ell_after": float(ell[probe_index, slot]),
                                    f"{prefix}_divergence": float(divs[probe_index][slot]),
                                    f"{prefix}_ell_increment": float(drift_increments[probe_index][slot]),
                                    f"{prefix}_positive_density": int(valid)})
                        if valid:
                            u = math.exp(value-limit)
                            beta = u/(-math.expm1(value-limit))
                            lhs = float(qfull[slot])-u*float(qbase[slot])
                            rhs = (1-t)/t**2*beta*float(gap_squared[slot])
                            margin = lhs-rhs
                            row.update({f"{prefix}_beta": beta, f"{prefix}_covariance_lhs": lhs,
                                        f"{prefix}_covariance_rhs": rhs,
                                        f"{prefix}_covariance_margin": margin,
                                        f"{prefix}_covariance_test_available": 1})
                        else:
                            # Missing quantities remain empty in CSV. They are
                            # never repaired, clipped, or used by the sampler.
                            row.update({f"{prefix}_beta": None, f"{prefix}_covariance_lhs": None,
                                        f"{prefix}_covariance_rhs": None,
                                        f"{prefix}_covariance_margin": None,
                                        f"{prefix}_covariance_test_available": 0})
                    rows.append(row)
                guided = full.detach()+gamma*gap.detach()
                state = state.detach()-(t-s)*((state.detach()-guided)/t)
            if not bool(torch.isfinite(state).all() and torch.isfinite(ell).all()):
                raise FloatingPointError("nonfinite state or transported log ratio")
            write_csv(out / "per_sample_step.csv", rows)
            progress = {"status": "auditing", "step_index": index, "rows": len(rows),
                        "elapsed_seconds": time.perf_counter()-started}
            atomic_json(out / "progress.json", progress)
            if index % 10 == 0:
                print(json.dumps(progress), flush=True)
        torch.save({"state": state.cpu(), "labels": labels.cpu(), "ell": ell.cpu()}, out / "endpoint.pt")
        summary = {"complete": True, "rows": len(rows), "steps": 100,
                   "elapsed_seconds": time.perf_counter()-started,
                   "model_forward_calls_B8": 100, "VJP_calls_B8": 400,
                   "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "per_sample_step_sha256": sha256_file(out / "per_sample_step.csv"),
                   "no_decode": True, "no_fid": True, "no_candidate_sampling": True}
        for probe_index in range(2):
            prefix = f"probe{probe_index}"
            eligible = [row for row in rows if row[f"{prefix}_covariance_test_available"]]
            summary[prefix] = {
                "positive_density_rows": len(eligible),
                "negative_covariance_margin_rows": sum(row[f"{prefix}_covariance_margin"] < 0 for row in eligible),
                "ell_min": min(row[f"{prefix}_ell_before"] for row in rows),
                "ell_max": max(row[f"{prefix}_ell_before"] for row in rows)}
        atomic_json(out / "summary.json", summary)
        atomic_json(out / "progress.json", {**summary, "status": "complete"})
        print(json.dumps(summary), flush=True)
    except BaseException as exc:
        atomic_json(out / "progress.json", {"status": "failed", "rows": len(rows),
                    "elapsed_seconds": time.perf_counter()-started, "error": repr(exc)})
        raise


if __name__ == "__main__":
    main()
