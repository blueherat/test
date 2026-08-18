#!/usr/bin/env python3
"""Paired FID-1K temporal-utility sweep for external-checkpoint AutoGuidance.

V2 fixes ADM-FID interpreter handling: preserve the virtualenv symlink and
preflight TensorFlow before any sampling.

Scientific question
-------------------
For one fixed external weak checkpoint W, is

    S + gamma * (S - W)

more useful in the high-noise half or the low-noise half of the SiT trajectory?
The repository convention is t=0 noise -> t=1 data, so with switch=0.5:

    high_noise: guidance on for t < 0.5, off afterward
    low_noise : guidance off for t < 0.5, on afterward

Default pair:
    strong = v800
    weak   = v180   (best static checkpoint reference in the long study)

Protocol
--------
* Uses the repository's existing checkpoint-reference schedule sampler v2 and
  long-study runner. No new sampling semantics are introduced.
* 1000 samples, batch=8, seed=0, Dopri5, atol=1e-6, rtol=1e-3.
* GPUs 1 and 3. Conditions are ordered high/low by gamma, so one GPU naturally
  runs the high-noise strip and the other the low-noise strip.
* Every condition is exactly paired in initial noise and labels.
* First reproduces two anchors: unguided v800 and static v180 gamma=3.05.
* Coarse gamma sweep is common to high/low; refinement uses the UNION of local
  neighborhoods around both coarse optima, again evaluated for BOTH regions.

Known historical paired 1K fingerprints from checkpoint_reference_long_study_v1:
    noise = b693d3cc...02c7b8
    label = 76fcd0fc...c0758
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

EXPECTED_NOISE = "b693d3cc2f28249d84942f74586d1afda2df10879225cd69bbf5d6a2d602c7b8"
EXPECTED_LABEL = "76fcd0fce6808c069a79ee8fd795edf2a1785d73758dc62306e51700c44c0758"
HIST_STATIC_V180_G305_FID1K = 54.7064333488791
ANCHOR_FID_TOL = 0.15
DEFAULT_COARSE = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0)


def detect_repo() -> Path:
    candidates = [Path.cwd().resolve(), Path(__file__).resolve().parent.parent]
    for q in candidates:
        if (q / "experiments/run_imagenet100_sit_checkpoint_reference_long_study_v1.py").is_file():
            return q
    raise FileNotFoundError("Run from /home/zhoushunyu/eqvae or place this file under <repo>/experiments/.")


def detect_data() -> Path:
    candidates = (
        Path("/data/users/zhoushunyu/eqvae/imagenet_sit_flow"),
        Path("/home/zhoushunyu/data/eqvae/imagenet_sit_flow"),
    )
    marker = Path("runs/sit-s-2_seed0/checkpoints/step_00800000.pt")
    for q in candidates:
        if (q / marker).is_file():
            return q
    raise FileNotFoundError("Cannot find ImageNet-100 SiT data root")


def parse_gpus(s: str) -> tuple[int, ...]:
    try:
        out = tuple(int(x.strip()) for x in s.split(",") if x.strip())
    except ValueError as e:
        raise argparse.ArgumentTypeError("--gpus expects comma-separated integers") from e
    if not out or len(out) != len(set(out)) or any(x < 0 for x in out):
        raise argparse.ArgumentTypeError("GPU ids must be unique non-negative integers")
    return out


def parse_gammas(s: str) -> tuple[float, ...]:
    try:
        xs = tuple(float(x.strip()) for x in s.split(",") if x.strip())
    except ValueError as e:
        raise argparse.ArgumentTypeError("expected comma-separated gamma values") from e
    if not xs or any((not math.isfinite(x)) or x <= 0 for x in xs):
        raise argparse.ArgumentTypeError("gammas must be finite and > 0")
    return tuple(sorted(set(round(x, 8) for x in xs)))


def gtag(x: float) -> str:
    s = f"{float(x):.6f}".rstrip("0").rstrip(".") or "0"
    return s.replace("-", "m").replace(".", "p")


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpus", type=parse_gpus, default=parse_gpus("1,3"))
    ap.add_argument("--weak-step", type=int, default=180000)
    ap.add_argument("--switch-time", type=float, default=0.5)
    ap.add_argument("--coarse-gammas", type=parse_gammas,
                    default=DEFAULT_COARSE)
    ap.add_argument("--num-samples", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-output-points", type=int, default=250)
    ap.add_argument("--atol", type=float, default=1e-6)
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--vae-decode-batch-size", type=int, default=2)
    ap.add_argument("--cuda-allocator-limit-gib", type=float, default=4.0)
    ap.add_argument("--gpu-memory-ceiling-mib", type=int, default=15 * 1024)
    ap.add_argument("--memory-poll-interval", type=float, default=0.5)
    ap.add_argument("--fid-batch-size", type=int, default=8)
    ap.add_argument("--fid-gpu-memory-fraction", type=float, default=0.25)
    ap.add_argument("--output-root", type=Path)
    ap.add_argument("--adm-python", type=Path)
    ap.add_argument("--refine", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--keep-samples", action="store_true")
    ap.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if not 0 < a.switch_time < 1:
        raise ValueError("--switch-time must be in (0,1)")
    if a.weak_step <= 0 or a.weak_step >= 800000:
        raise ValueError("--weak-step must be in (0,800000)")
    if a.num_samples <= 0 or a.batch_size <= 0:
        raise ValueError("sample count and batch size must be positive")

    repo = detect_repo(); data = detect_data()
    sys.path.insert(0, str(repo))
    from experiments.run_imagenet100_sit_checkpoint_reference_long_study_v1 import (
        checkpoint_meta, validate_family, condition, stage, run_conditions
    )
    from experiments.run_imagenet100_sit_fid_curve import DEFAULT_ADM_PYTHON

    ckpt_dir = data / "runs/sit-s-2_seed0/checkpoints"
    strong_path = ckpt_dir / "step_00800000.pt"
    weak_path = ckpt_dir / f"step_{a.weak_step:08d}.pt"
    reference = data / "adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz"
    if not strong_path.is_file() or not weak_path.is_file() or not reference.is_file():
        raise FileNotFoundError(
            f"missing strong/weak/reference:\n  {strong_path}\n  {weak_path}\n  {reference}"
        )

    if a.adm_python is None:
        candidates = [Path(DEFAULT_ADM_PYTHON), Path("/data/shared/envs/adm-fid/bin/python")]
        a.adm_python = next((p for p in candidates if p.is_file()), candidates[0])
    # IMPORTANT: do NOT use Path.resolve() here.  The shared ADM-FID Python
    # entry point is a virtual-environment symlink; resolving it can bypass the
    # environment's pyvenv.cfg and fall back to the underlying myenv interpreter.
    a.adm_python = a.adm_python.expanduser().absolute()
    if not a.adm_python.is_file():
        raise FileNotFoundError(f"ADM Python not found: {a.adm_python}")

    # Fail before any sampling if the selected evaluator interpreter is not the
    # TensorFlow-enabled ADM environment.
    probe = subprocess.run(
        [
            str(a.adm_python),
            "-c",
            "import sys; import tensorflow.compat.v1 as tf; "
            "print(sys.executable); print(tf.__version__)",
        ],
        text=True, capture_output=True, check=False,
    )
    if probe.returncode != 0:
        raise RuntimeError(
            "ADM evaluator Python cannot import tensorflow.compat.v1:\n"
            f"  interpreter: {a.adm_python}\n"
            f"  stdout: {probe.stdout.strip()}\n"
            f"  stderr: {probe.stderr.strip()}"
        )
    print("ADM evaluator preflight: OK")
    print(probe.stdout.strip())

    weak_name = f"v{a.weak_step // 1000}"
    a.output_root = (
        a.output_root.expanduser().resolve() if a.output_root
        else data / f"external_{weak_name}_temporal_utility_fid1k_v1"
    )
    a.output_root.mkdir(parents=True, exist_ok=True)

    # Fields expected by the repository long-study run_condition().
    a.strong_checkpoint = strong_path.resolve()
    a.reference = reference.resolve()
    a.gpu_indices = list(a.gpus)

    strong = checkpoint_meta(a.strong_checkpoint)
    refs = {weak_name: checkpoint_meta(weak_path.resolve())}
    validate_family(strong, refs)

    def decorate(c: dict, region: str, gamma: float) -> dict:
        c = dict(c)
        c["temporal_region"] = region
        c["screen_gamma"] = float(gamma)
        c["switch_time"] = float(a.switch_time)
        return c

    def baseline_condition() -> dict:
        c = condition(
            "baseline_v800", [stage("strong", 0)], group="anchor",
            note="unguided v800 anchor"
        )
        return decorate(c, "baseline", 0.0)

    def static_anchor_condition() -> dict:
        c = condition(
            f"static_{weak_name}_g3p05", [stage(weak_name, 3.05)],
            group="anchor", note="historical full-time AG anchor"
        )
        return decorate(c, "full", 3.05)

    def half_condition(region: str, gamma: float) -> dict:
        if region == "high_noise":
            stages = [stage(weak_name, gamma), stage("strong", 0)]
        elif region == "low_noise":
            stages = [stage("strong", 0), stage(weak_name, gamma)]
        else:
            raise ValueError(region)
        c = condition(
            f"{region}_{weak_name}_g{gtag(gamma)}",
            stages,
            boundaries=[a.switch_time],
            mode="hard",
            group="temporal_half_sweep",
            note=(
                "t=0 is noise and t=1 is data; guidance active only in " + region
            ),
        )
        return decorate(c, region, gamma)

    anchors = [baseline_condition(), static_anchor_condition()]
    coarse = [
        half_condition(region, g)
        for g in a.coarse_gammas
        for region in ("high_noise", "low_noise")
    ]

    request = {
        "format": "eqvae_external_checkpoint_temporal_utility_request_v1",
        "strong": str(a.strong_checkpoint),
        "weak": str(weak_path.resolve()),
        "weak_name": weak_name,
        "switch_time": a.switch_time,
        "time_convention": "t=0 noise -> t=1 data",
        "coarse_gammas": list(a.coarse_gammas),
        "num_samples": a.num_samples,
        "batch_size": a.batch_size,
        "seed": a.seed,
        "gpus": list(a.gpus),
        "refine": a.refine,
    }
    atomic_json(a.output_root / "request.json", request)

    print("=== EXTERNAL CHECKPOINT TEMPORAL UTILITY ===")
    print(f"strong: v800")
    print(f"weak:   {weak_name}")
    print(f"switch: t={a.switch_time:g} (t=0 noise -> t=1 data)")
    print(f"GPUs:   {a.gpus}")
    print(f"coarse: {a.coarse_gammas}")
    print(f"output: {a.output_root}")
    print(f"coarse conditions: {len(coarse)} + 2 anchors")
    if a.dry_run:
        for c in anchors + coarse[:8]:
            print(" ", c["name"], c["stages"], c["boundaries"])
        if len(coarse) > 8:
            print(f"  ... +{len(coarse)-8} coarse")
        return

    anchor_results, anchor_fail = run_conditions(anchors, strong, refs, a, "00_anchor")
    if anchor_fail:
        raise RuntimeError(f"anchor failures: {anchor_fail}")

    # Pairing + historical static anchor check before spending the full sweep.
    def fingerprints(results):
        return (
            {x["sampling_manifest"]["noise_sha256"] for x in results},
            {x["sampling_manifest"]["label_sha256"] for x in results},
        )
    nset, lset = fingerprints(anchor_results)
    if nset != {EXPECTED_NOISE} or lset != {EXPECTED_LABEL}:
        raise RuntimeError(
            "anchor RNG does not match historical checkpoint study:\n"
            f"noise={nset}\nlabel={lset}"
        )
    static_result = next(x for x in anchor_results if x["condition"]["temporal_region"] == "full")
    static_fid = float(static_result["metrics"]["fid"])
    if abs(static_fid - HIST_STATIC_V180_G305_FID1K) > ANCHOR_FID_TOL and a.weak_step == 180000:
        raise RuntimeError(
            f"static v180 gamma3.05 anchor mismatch: observed={static_fid:.6f}, "
            f"historical={HIST_STATIC_V180_G305_FID1K:.6f}"
        )
    print(f"anchor pairing: OK; static {weak_name} g3.05 FID={static_fid:.4f}")

    coarse_results, coarse_fail = run_conditions(coarse, strong, refs, a, "01_half_coarse")
    if coarse_fail:
        raise RuntimeError(f"coarse failures: {coarse_fail}")

    def rows(results):
        out = []
        for x in results:
            c=x["condition"]; q=x["metrics"]; m=x["sampling_manifest"]
            out.append({
                "phase": x["phase"],
                "region": c.get("temporal_region", ""),
                "gamma": float(c.get("screen_gamma", 0.0)),
                "fid": float(q["fid"]),
                "sfid": float(q["sfid"]),
                "inception_score": float(q["inception_score"]),
                "total_nfe": int(m["total_nfe"]),
                "noise_sha256": m["noise_sha256"],
                "label_sha256": m["label_sha256"],
                "condition": c["name"],
            })
        return out

    coarse_rows = rows(coarse_results)
    best_coarse = {}
    for region in ("high_noise", "low_noise"):
        subset = [r for r in coarse_rows if r["region"] == region]
        best_coarse[region] = min(subset, key=lambda r:r["fid"])
        r=best_coarse[region]
        print(f"coarse best {region}: gamma={r['gamma']:.3f}, FID={r['fid']:.4f}")

    refine_results = []
    if a.refine:
        # Common refinement grid = union around both regional optima.
        rg = set()
        lo, hi = min(a.coarse_gammas), max(a.coarse_gammas)
        for region in ("high_noise", "low_noise"):
            center = float(best_coarse[region]["gamma"])
            for k in range(-3, 4):
                g = round(center + 0.25*k, 8)
                if 0.25 <= g <= 8.0:
                    rg.add(g)
            if abs(center-hi) < 1e-9:
                rg.update(g for g in (hi+0.5, hi+1.0, hi+1.5) if g <= 8.0)
            if abs(center-lo) < 1e-9 and lo > 0.25:
                rg.add(max(0.25, lo-0.25))
        rg -= set(a.coarse_gammas)
        refine_conditions = [
            half_condition(region, g)
            for g in sorted(rg)
            for region in ("high_noise", "low_noise")
        ]
        print(f"common refinement gammas: {tuple(sorted(rg))}; conditions={len(refine_conditions)}")
        refine_results, refine_fail = run_conditions(
            refine_conditions, strong, refs, a, "02_half_refine"
        )
        if refine_fail:
            raise RuntimeError(f"refinement failures: {refine_fail}")

    all_results = anchor_results + coarse_results + refine_results
    all_rows = rows(all_results)
    nset, lset = fingerprints(all_results)
    if nset != {EXPECTED_NOISE} or lset != {EXPECTED_LABEL}:
        raise RuntimeError(f"not exactly paired across sweep: noise={nset}, label={lset}")

    half_rows = [r for r in all_rows if r["region"] in ("high_noise", "low_noise")]
    best = {
        region: min((r for r in half_rows if r["region"] == region), key=lambda r:r["fid"])
        for region in ("high_noise", "low_noise")
    }
    baseline = next(r for r in all_rows if r["region"] == "baseline")
    static = next(r for r in all_rows if r["region"] == "full")

    # Same-gamma paired comparison table.
    maps = {
        region: {round(r["gamma"],8):r for r in half_rows if r["region"]==region}
        for region in ("high_noise", "low_noise")
    }
    common = sorted(set(maps["high_noise"]) & set(maps["low_noise"]))
    paired = []
    for g in common:
        h, l = maps["high_noise"][g], maps["low_noise"][g]
        paired.append({
            "gamma":g,
            "fid_high_noise":h["fid"],
            "fid_low_noise":l["fid"],
            "high_noise_advantage_fid":l["fid"]-h["fid"],
            "sfid_high_noise":h["sfid"],
            "sfid_low_noise":l["sfid"],
            "is_high_noise":h["inception_score"],
            "is_low_noise":l["inception_score"],
        })

    sd = a.output_root / "summary"; sd.mkdir(parents=True, exist_ok=True)
    write_csv(sd / "all_conditions.csv", all_rows)
    write_csv(sd / "paired_high_vs_low.csv", paired)
    write_csv(sd / "best_by_region.csv", [best["high_noise"], best["low_noise"]])

    summary = {
        "format":"eqvae_external_checkpoint_temporal_utility_summary_v1",
        "strong":"v800", "weak":weak_name,
        "time_convention":"t=0 noise -> t=1 data",
        "switch_time":a.switch_time,
        "best_high_noise":best["high_noise"],
        "best_low_noise":best["low_noise"],
        "high_noise_advantage_fid":best["low_noise"]["fid"]-best["high_noise"]["fid"],
        "baseline":baseline,
        "static_anchor":static,
        "high_noise_gain_vs_baseline":baseline["fid"]-best["high_noise"]["fid"],
        "low_noise_gain_vs_baseline":baseline["fid"]-best["low_noise"]["fid"],
        "static_gain_vs_baseline":baseline["fid"]-static["fid"],
        "pairing":{"noise_sha256":EXPECTED_NOISE,"label_sha256":EXPECTED_LABEL,"exact":True},
        "files":{
            "all_conditions":str(sd/"all_conditions.csv"),
            "paired_high_vs_low":str(sd/"paired_high_vs_low.csv"),
            "best_by_region":str(sd/"best_by_region.csv"),
        },
    }
    atomic_json(sd / "summary.json", summary)

    H,L=best["high_noise"],best["low_noise"]
    print("\n=== FINAL TEMPORAL UTILITY ===")
    print(f"baseline v800:     FID={baseline['fid']:.4f}")
    print(f"static {weak_name}:       gamma=3.05, FID={static['fid']:.4f}")
    print(f"high-noise only:   gamma={H['gamma']:.3f}, FID={H['fid']:.4f}")
    print(f"low-noise only:    gamma={L['gamma']:.3f}, FID={L['fid']:.4f}")
    print(f"high-noise advantage: {L['fid']-H['fid']:+.4f} FID (positive => high-noise wins)")
    print(f"summary: {sd/'summary.json'}")
    print(f"paired table: {sd/'paired_high_vs_low.csv'}")


if __name__ == "__main__":
    main()