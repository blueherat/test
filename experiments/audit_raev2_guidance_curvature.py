#!/usr/bin/env python3
"""Measure learned score curvature on frozen teacher and IG rollout states.

For z_t=(1-t)X+t*epsilon, exact posterior means obey
  t^2 H(log p_t) = (1-t) J_F - I.
We measure directional quadratic forms using FP32 autograd, without finite
differences, intervention, score selection, guidance tuning, or FID. For a
nonconservative learned field these are symmetric-Jacobian diagnostics, not a
proof that the network defines a density. All query states were saved before
this diagnostic and the denoising trajectories are never changed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "external/RAEv2/src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.audit_raev2_proximal_calibration import atomic_json, sha256_file
from experiments.sample_raev2_proximal_calibration import DEFAULT_CONFIG, DEFAULT_CHECKPOINT
from experiments.sample_raev2_pfr_retiming import load_config
from experiments.audit_raev2_guidance_normal_noise import affine_geometry


def dot(x, y):
    return float((x.double()*y.double()).sum())


def norm(x):
    return float(torch.linalg.vector_norm(x.double()))


def normal_part(value, unit_normal):
    return (value*unit_normal).sum(dim=1, keepdim=True)*unit_normal


def unit(value):
    length = norm(value)
    if not np.isfinite(length) or length == 0:
        raise ValueError("nonfinite/zero diagnostic direction")
    return value/length


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.state_dir.resolve().glob("step_*.pt"))
    if not paths:
        raise FileNotFoundError("no frozen state files")
    parent = args.state_dir.resolve().parent
    source_request = json.loads((parent / "request.json").read_text())
    source_summary = json.loads((parent / "summary.json").read_text())
    if not source_summary.get("complete"):
        raise ValueError("source audit must finish before curvature analysis")
    if [int(path.stem.split("_")[-1]) for path in paths] != source_request["snapshot_indices"]:
        raise ValueError("frozen state set is incomplete or differs from preregistered snapshots")
    config = load_config(args.config.resolve())
    stats_path = Path(config.stage_1.params["normalization_stat_path"])
    stats = torch.load(stats_path, map_location="cpu", weights_only=True)
    normal, _ = affine_geometry(stats["mean"], stats["var"])
    normal = normal.float().unsqueeze(0)
    state_artifacts = [{"path": str(path), "sha256": sha256_file(path)} for path in paths]
    request = {"protocol": "raev2_frozen_state_directional_score_curvature_v1",
               "states": state_artifacts, "state_source_request_sha256": sha256_file(parent / "request.json"),
               "state_source_summary_sha256": sha256_file(parent / "summary.json"),
               "config": str(args.config.resolve()), "config_sha256": sha256_file(args.config),
               "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256_file(args.checkpoint),
               "normalization_stats_sha256": sha256_file(stats_path),
               "source_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in (
                   Path(__file__), ROOT / "experiments/audit_raev2_guidance_normal_noise.py",
                   ROOT / "experiments/sample_raev2_pfr_retiming.py",
                   ROOT / "external/RAEv2/src/stage2/models/DDT.py",
                   ROOT / "external/RAEv2/src/stage2/models/model_utils.py")}, "state_key": "ema",
               "precision": "FP32_no_autocast_no_TF32; autograd VJP; microbatch1",
               "directions": ["gap_tangent", "independent_random_tangent", "known_normal"],
               "direction_seed": 202609072, "random_directions_reused_across_steps_and_domains": True,
               "gap_direction": "recomputed and detached separately at every frozen state",
               "quadratic_form": "rho=(1-t)*<v,J_head*v>-1 for unit v; exact-score interpretation requires conservative accurate head",
               "intervention": False, "fid": False, "hyperparameter_selection": False,
               "torch_version": str(torch.__version__)}
    atomic_json(out / "request.json", request)
    (out / "audit_source.py").write_bytes(Path(__file__).read_bytes())
    atomic_json(out / "progress.json", {"status": "initializing", "rows": 0})
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from utils.model_utils import instantiate_from_config
    model = instantiate_from_config(config.stage_2).to(device).eval().requires_grad_(False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    model.load_state_dict(checkpoint["ema"], strict=True)
    del checkpoint
    normal = normal.to(device)
    first = torch.load(paths[0], map_location="cpu", weights_only=True)
    ids = torch.as_tensor(first["sample_ids"]).tolist()
    generator = torch.Generator(device="cpu").manual_seed(request["direction_seed"])
    random_directions = {}
    for index in ids:
        raw = torch.randn((1,1024,16,16), generator=generator).to(device)
        random_directions[index] = (unit(raw-normal_part(raw, normal)), unit(normal_part(raw, normal)))
    del first
    rows = []
    started = time.perf_counter()
    for path, identity in zip(paths, state_artifacts):
        if sha256_file(path) != identity["sha256"]:
            raise ValueError("frozen state file changed during the audit")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        index, t = int(payload["step_index"]), float(payload["t"])
        sample_ids = torch.as_tensor(payload["sample_ids"]).tolist()
        labels = torch.as_tensor(payload["labels"]).long()
        if sample_ids != ids or len(labels) != len(ids) or not 0 < t <= 1:
            raise ValueError("inconsistent frozen state cohort")
        for domain in ("teacher", "rollout"):
            for slot, sample_id in enumerate(ids):
                state = payload[domain]["state"][slot:slot+1].float().to(device).requires_grad_(True)
                label = labels[slot:slot+1].to(device)
                full, base = model(state, torch.full((1,), t, device=device), context=label, attn_mask=None)
                gap = (full-base).detach()
                tangent = gap-normal_part(gap, normal)
                directions = [("gap_tangent", unit(tangent)),
                              ("independent_random_tangent", random_directions[sample_id][0]),
                              ("known_normal", random_directions[sample_id][1])]
                for direction_index, (name, vector) in enumerate(directions):
                    # The direction is detached: this is v^T J v, without any
                    # derivative through the rule that chose v at this state.
                    grad_full = torch.autograd.grad((full*vector).sum(), state, retain_graph=True)[0]
                    grad_base = torch.autograd.grad((base*vector).sum(), state,
                                                    retain_graph=direction_index != len(directions)-1)[0]
                    q_full, q_base = dot(vector, grad_full), dot(vector, grad_base)
                    rho_full, rho_base = (1-t)*q_full-1, (1-t)*q_base-1
                    ig_extra = .78 if .1 <= t <= 1 else 0.0
                    row = {"step_index": index, "t": t, "domain": domain, "sample_id": sample_id,
                           "label": int(labels[slot]), "direction": name,
                           "direction_norm": norm(vector), "rho_full": rho_full, "rho_base": rho_base,
                           "rho_official_ig": rho_full+ig_extra*(rho_full-rho_base),
                           "full_vjp_norm": norm(grad_full), "base_vjp_norm": norm(grad_base),
                           "full_vjp_input_normal_norm": norm(normal_part(grad_full, normal)),
                           "base_vjp_input_normal_norm": norm(normal_part(grad_base, normal)),
                           "raw_gap_norm": norm(gap), "gap_normal_norm": norm(normal_part(gap, normal)),
                           "full_fp32_vs_saved_bf16_rms": norm(full.detach().cpu()-payload[domain]["full"][slot:slot+1])/512,
                           "base_fp32_vs_saved_bf16_rms": norm(base.detach().cpu()-payload[domain]["base"][slot:slot+1])/512}
                    if not all(np.isfinite(value) for value in row.values() if isinstance(value, float)):
                        raise FloatingPointError("nonfinite curvature statistic")
                    rows.append(row)
                del state, full, base, gap, tangent, grad_full, grad_base
        with (out / "directional_curvature.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        progress = {"status": "auditing", "last_step_index": index, "rows": len(rows),
                    "elapsed_seconds": time.perf_counter()-started}
        atomic_json(out / "progress.json", progress)
        print(json.dumps(progress), flush=True)
    groups = []
    for domain in ("teacher", "rollout"):
        for name in request["directions"]:
            group = [row for row in rows if row["domain"] == domain and row["direction"] == name]
            groups.append({"domain": domain, "direction": name, "rows": len(group),
                           **{f"{head}_positive_fraction": float(np.mean([row[f"rho_{head}"] > 0 for row in group]))
                              for head in ("full", "base", "official_ig")},
                           "rho_full_median": float(np.median([row["rho_full"] for row in group]))})
    atomic_json(out / "summary.json", {"protocol": request["protocol"], "complete": True, "rows": len(rows),
                "groups": groups, "elapsed_seconds": time.perf_counter()-started,
                "model_forward_calls_microbatch1": len(rows)//3, "VJP_calls": len(rows)*2,
                "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
                "conclusion_boundary": "learned directional symmetric-Jacobian diagnostics; no largest-eigenvalue, model-error, density, or FID guarantee"})
    atomic_json(out / "progress.json", {**progress, "status": "complete", "complete": True})


if __name__ == "__main__":
    main()
