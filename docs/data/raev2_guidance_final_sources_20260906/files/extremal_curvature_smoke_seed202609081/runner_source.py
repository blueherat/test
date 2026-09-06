#!/usr/bin/env python3
"""Extremal symmetric-Jacobian diagnostic on previously frozen RAEv2 states.

The earlier three-direction Rayleigh audit did not estimate the spectral edge.
This audit searches for a positive direction of H=(1-t)*sym(J_F)-I. It makes
no density, saddle-error, quality, or full-spectrum-negative claim. No sampler
state is changed and no decoder, optimizer, selection or FID is used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "external/RAEv2/src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.audit_raev2_proximal_calibration import atomic_json, sha256_file, write_csv
from experiments.audit_raev2_guidance_normal_noise import affine_geometry
from experiments.sample_raev2_proximal_calibration import DEFAULT_CONFIG, DEFAULT_CHECKPOINT
from experiments.sample_raev2_pfr_retiming import load_config
from experiments.raev2_symmetric_krylov import symmetric_krylov


def inner(a, b):
    return (a.double()*b.double()).sum()


def length(a):
    return torch.linalg.vector_norm(a.double())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, default=Path(
        "/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/"
        "normal_noise_audit_seed202609071/states"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--smoke", action="store_true", help="one rollout state, 8 Krylov steps")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    config = load_config(args.config)
    source_dir = args.state_dir.resolve().parent
    parent_request = json.loads((source_dir / "request.json").read_text())
    parent_summary = json.loads((source_dir / "summary.json").read_text())
    if not parent_summary.get("complete"):
        raise ValueError("source states must come from a completed audit")
    indices = [47] if args.smoke else [47, 67, 84, 92, 97]
    sample_ids = [0] if args.smoke else [0, 1, 2, 3]
    domains = ["rollout"] if args.smoke else ["teacher", "rollout"]
    iterations = 8 if args.smoke else 24
    paths = []
    for index in indices:
        matching = [p for p in args.state_dir.glob("step_*.pt") if int(p.stem.split("_")[-1]) == index]
        if len(matching) != 1:
            raise ValueError(f"missing or ambiguous state {index}")
        paths.append(matching[0].resolve())
    if sha256_file(args.config) != parent_request["config_sha256"]:
        raise ValueError("config identity differs from saved states")
    checkpoint_hash = sha256_file(args.checkpoint)
    if checkpoint_hash != parent_request["checkpoint_sha256"]:
        raise ValueError("checkpoint identity differs from saved states")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    stats_path = Path(config.stage_1.params["normalization_stat_path"])
    stats = torch.load(stats_path, map_location="cpu", weights_only=True)
    normal, _ = affine_geometry(stats["mean"], stats["var"])
    normal = normal.to(device=device, dtype=torch.float64).unsqueeze(0)
    request = {
        "protocol": "raev2_frozen_state_extremal_symmetric_curvature_v1",
        "smoke": args.smoke, "sample_ids": sample_ids, "step_indices": indices,
        "domains": domains, "max_krylov_iterations": iterations,
        "krylov_checkpoints": [8] if args.smoke else [8, 16, 24],
        "operator": "H_F=(1-t)*(J_F+J_F^T)/2-I; full latent space",
        "start": "independent isotropic Gaussian, seed 202609081+sample_id; reused across times/domains",
        "precision": "FP32 parameters/states/JVP/VJP; FP64 Krylov basis/reductions; no autocast/TF32; math SDPA",
        "interpretation": "positive measured Rayleigh certifies a numerically observed expansion direction; negative largest Ritz value is NOT proof of a negative full spectrum",
        "nonconservative_boundary": "H_F is a symmetric Jacobian diagnostic; no assertion that F is the score of its endpoint law",
        "drift_relation": "for dataward clock tau=1-t, b=(F-z)/t, sym(J_b)=(H_F+t*I)/((1-t)*t); drift expansion threshold is rho>-t, not rho>0",
        "source_states": [{"path": str(p), "sha256": sha256_file(p)} for p in paths],
        "source_request_sha256": sha256_file(source_dir / "request.json"),
        "source_summary_sha256": sha256_file(source_dir / "summary.json"),
        "config_sha256": sha256_file(args.config), "checkpoint_sha256": checkpoint_hash,
        "stats_sha256": sha256_file(stats_path),
        "sources": {str(p.relative_to(ROOT)): sha256_file(p) for p in (
            Path(__file__), ROOT / "experiments/raev2_symmetric_krylov.py",
            ROOT / "external/RAEv2/src/stage2/models/DDT.py",
            ROOT / "external/RAEv2/src/stage2/models/model_utils.py")},
        "finite_difference": "first fixed state only; central steps h=sqrt(d)*[0.001,0.0003] on the final unit Ritz direction; both retained",
        "sampling": False, "decoder": False, "training": False, "fid": False,
        "torch_version": str(torch.__version__), "cuda_device": torch.cuda.get_device_name(device)}
    atomic_json(out / "request.json", request)
    (out / "runner_source.py").write_bytes(Path(__file__).read_bytes())
    atomic_json(out / "progress.json", {"status": "initializing"})
    from utils.model_utils import instantiate_from_config
    model = instantiate_from_config(config.stage_2).to(device).eval().requires_grad_(False)
    weights = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    model.load_state_dict(weights["ema"], strict=True)
    request["checkpoint_step"] = int(weights.get("step", 0))
    del weights
    atomic_json(out / "request.json", request)
    ledger = {"model_forward_calls": 0, "jvp_calls": 0, "vjp_calls": 0}
    rows, histories, differences = [], [], []
    try:
        with sdpa_kernel(SDPBackend.MATH):
            for path, identity in zip(paths, request["source_states"]):
                if sha256_file(path) != identity["sha256"]:
                    raise ValueError("state artifact changed")
                payload = torch.load(path, map_location="cpu", weights_only=True)
                t = float(payload["t"])
                step = int(payload["step_index"])
                for domain in domains:
                    for sample_id in sample_ids:
                        tick = time.perf_counter()
                        slots = torch.where(torch.as_tensor(payload["sample_ids"]) == sample_id)[0]
                        if len(slots) != 1:
                            raise ValueError("sample identity mismatch")
                        slot = int(slots[0])
                        state = payload[domain]["state"][slot:slot+1].to(device).float().requires_grad_(True)
                        labels = torch.as_tensor(payload["labels"])[slot:slot+1].to(device).long()
                        times = torch.full((1,), t, device=device)

                        def fields(z):
                            ledger["model_forward_calls"] += 1
                            return model(z, times, context=labels, attn_mask=None)

                        full, base = fields(state)

                        def jvp(vector):
                            ledger["jvp_calls"] += 1
                            with torch.no_grad():
                                _, tangents = torch.func.jvp(fields, (state.detach(),), (vector.float(),))
                            return tuple(v.detach().double() for v in tangents)

                        def vjp(output, vector):
                            ledger["vjp_calls"] += 1
                            return torch.autograd.grad(output, state, grad_outputs=vector.float(),
                                                       retain_graph=True)[0].detach().double()

                        def operator(vector):
                            forward, _ = jvp(vector)
                            transpose = vjp(full, vector)
                            return (1-t)*(forward+transpose)/2-vector.double()

                        rng = torch.Generator(device="cpu").manual_seed(202609081+sample_id)
                        start = torch.randn(state.shape, generator=rng, dtype=torch.float64).to(device)
                        result = symmetric_krylov(operator, start, max_iterations=iterations,
                                                  checkpoints=tuple(request["krylov_checkpoints"]))
                        vector = result["vector"].reshape_as(state).double()
                        vector = vector/length(vector)
                        jf, jb = jvp(vector)
                        jtf, jtb = vjp(full, vector), vjp(base, vector)
                        hf = (1-t)*(jf+jtf)/2-vector
                        hb = (1-t)*(jb+jtb)/2-vector
                        skew = (1-t)*(jf-jtf)/2
                        qf, qb = float(inner(vector, hf)), float(inner(vector, hb))
                        residual = float(length(hf-qf*vector))
                        gap = (full-base).detach().double()
                        random_unit = start/length(start)
                        random_h = operator(random_unit)
                        row = {"step_index": step, "t": t, "domain": domain, "sample_id": sample_id,
                               "label": int(labels[0]), "iterations": result["iterations"],
                               "ritz_value": result["ritz_value"], "direct_rayleigh_full": qf,
                               "direct_rayleigh_base_on_full_vector": qb,
                               "direct_rayleigh_ig_on_full_vector": qf+.78*(qf-qb),
                               "dataward_full_drift_rayleigh": (qf+t)/((1-t)*t),
                               "direct_full_residual_norm": residual,
                               "krylov_reported_residual_norm": result["residual_norm"],
                               "random_rayleigh_full": float(inner(random_unit, random_h)),
                               "skew_action_norm": float(length(skew)), "symmetric_action_norm": float(length(hf)),
                               "gap_cosine_with_full_vector": float(inner(vector, gap)/length(gap)),
                               "known_normal_vector_energy": float(((vector*normal).sum(1)**2).sum()),
                               "full_jvp_vjp_quadratic_discrepancy": float(abs(inner(vector, jf-jtf))),
                               "full_math_fp32_vs_saved_bf16_rms": float(length(full.detach().cpu()-payload[domain]["full"][slot:slot+1])/512),
                               "wall_seconds": time.perf_counter()-tick}
                        if not all(np.isfinite(v) for v in row.values() if isinstance(v, float)):
                            raise FloatingPointError("nonfinite spectral diagnostic")
                        if not rows:
                            for rms in (.001, .0003):
                                h = rms*state.numel()**.5
                                with torch.no_grad():
                                    plus, _ = fields(state.detach()+h*vector.float())
                                    minus, _ = fields(state.detach()-h*vector.float())
                                fd = (plus.double()-minus.double())/(2*h)
                                differences.append({"perturbation_rms": rms, "h": h,
                                    "relative_error": float(length(fd-jf)/length(jf)),
                                    "cosine": float(inner(fd, jf)/(length(fd)*length(jf))),
                                    "fd_rayleigh": float((1-t)*inner(vector, fd)-1),
                                    "autograd_rayleigh": qf})
                            torch.save({"state": state.detach().cpu(), "label": labels.cpu(), "t": t,
                                        "ritz_vector": vector.cpu(), "jvp_full": jf.cpu(), "vjp_full": jtf.cpu()},
                                       out / "first_state_numerics.pt")
                            atomic_json(out / "finite_difference.json", {"records": differences})
                        histories.append({"step_index": step, "domain": domain, "sample_id": sample_id,
                                          "history": result["history"]})
                        rows.append(row)
                        write_csv(out / "extremal_curvature.csv", rows)
                        atomic_json(out / "krylov_history.json", histories)
                        progress = {"status": "running", "rows": len(rows), "last": row,
                                    "ledger": ledger, "wall_seconds": time.perf_counter()-started}
                        atomic_json(out / "progress.json", progress)
                        print(json.dumps(progress), flush=True)
                        del state, full, base, jf, jb, jtf, jtb, hf, hb, result
        torch.cuda.synchronize(device)
        summary = {"complete": True, "protocol": request["protocol"], "rows": len(rows),
                   "positive_full_rayleigh_count": sum(r["direct_rayleigh_full"] > 0 for r in rows),
                   "negative_random_but_positive_full_ritz_count": sum(r["random_rayleigh_full"] < 0 < r["direct_rayleigh_full"] for r in rows),
                   "max_jvp_vjp_quadratic_discrepancy": max(r["full_jvp_vjp_quadratic_discrepancy"] for r in rows),
                   "ledger": ledger, "finite_difference": differences,
                   "total_wall_seconds": time.perf_counter()-started,
                   "peak_memory_bytes": torch.cuda.max_memory_allocated(),
                   "sampling": False, "training": False, "fid": False,
                   "boundary": "Limited saved-state FP32 local expansion diagnostic; no model-error, density, saddle or quality conclusion."}
        atomic_json(out / "summary.json", summary)
        atomic_json(out / "progress.json", {"status": "complete", "complete": True})
        print(json.dumps(summary), flush=True)
    except BaseException as exc:
        atomic_json(out / "progress.json", {"status": "failed", "error": repr(exc),
                    "rows": len(rows), "ledger": ledger, "wall_seconds": time.perf_counter()-started})
        raise


if __name__ == "__main__":
    main()
