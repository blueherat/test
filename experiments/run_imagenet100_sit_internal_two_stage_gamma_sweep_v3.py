#!/usr/bin/env python3
"""Paired FID-1K 2-D stage-gamma sweep for internal-head guidance.

Compare two symmetric families with the same hard switch time and exactly the
same (gamma_early, gamma_late) candidates:

  A: depth4(gamma_early) -> depth4(gamma_late)
  B: depth4(gamma_early) -> depth10(gamma_late)

At a fixed gamma pair, the only semantic difference is the late weak-head
identity.  Both families load and evaluate BOTH d4/d10 heads at every ODE
evaluation so that their strong-backbone execution path is also matched.

The script reports:
  (1) same-pair late-d10 identity effect:
        FID(A; ge,gl) - FID(B; ge,gl)
  (2) optimized-family effect:
        min FID(A) - min FID(B)

Default:
  switch t=0.5
  coarse gamma grid {.15, .25, .35, .45, .55, .65} on BOTH stages/families
  common refinement = union of 5x5 local grids (step .025, radius .05)
                      around both coarse optima, evaluated for BOTH families
  1000 samples, batch=8, seed=0, Dopri5, ADM FID
  GPUs 1 and 3, one sequential lane per GPU
  plus gamma_late=0 with gamma_early=0.45..0.75 step 0.025
  generated sample NPZ deleted after FID

The default RNG protocol exactly matches the repository v4 internal-head sweep.
Two anchor conditions are checked:
  d4->d4  (.25,.25) ~= existing static d4 gamma=.25
  d4->d10 (.45,.45) ~= existing 4->10 gamma=.45
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXPECTED_NOISE = "ab8419c7fdfd5b15dacbf4d37a3d567158e4332f25fd94580d3df73bac87e2c2"
EXPECTED_LABEL = "7c3ae6894e7ebab5c9b6524606f03b6a56b38dccbe472ff40edde26e48654fe6"
OLD_STATIC_D4_G025 = 69.73868613129753
OLD_D4_D10_G045 = 67.17630291386695
ANCHOR_TOL = 0.15

FAMILIES = ("d4_to_d4", "d4_to_d10")
DEFAULT_COARSE = (0.15, 0.25, 0.35, 0.45, 0.55, 0.65)
DEFAULT_LATE0_EARLY = tuple(round(0.45 + 0.025 * i, 8) for i in range(13))  # 0.45..0.75


def detect_repo() -> Path:
    here = Path.cwd().resolve()
    for q in (here, here.parent, Path(__file__).resolve().parent.parent):
        if (q / "experiments/compute_adm_fid.py").is_file():
            return q
    raise FileNotFoundError("Cannot find eqvae repo; run from /home/zhoushunyu/eqvae.")


def detect_data() -> Path:
    candidates = (
        Path("/data/users/zhoushunyu/eqvae/imagenet_sit_flow"),
        Path("/home/zhoushunyu/data/eqvae/imagenet_sit_flow"),
    )
    marker = Path("runs/sit-s-2_seed0/checkpoints/step_00800000.pt")
    for q in candidates:
        if (q / marker).is_file():
            return q
    raise FileNotFoundError("Cannot find ImageNet-100 SiT data root.")


def detect_adm_python() -> Path:
    for q in (
        Path("/data/shared/envs/adm-fid/bin/python"),
        Path("/home/zhoushunyu/data/shared/envs/adm-fid/bin/python"),
    ):
        if q.is_file():
            return q
    raise FileNotFoundError("Cannot find adm-fid Python environment.")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def parse_gpus(text: str) -> tuple[int, ...]:
    try:
        out = tuple(int(x.strip()) for x in text.split(",") if x.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated GPU indices") from exc
    if not out or len(set(out)) != len(out) or any(x < 0 for x in out):
        raise argparse.ArgumentTypeError("GPUs must be unique non-negative integers")
    return out


def parse_gammas(text: str) -> tuple[float, ...]:
    try:
        out = tuple(float(x.strip()) for x in text.split(",") if x.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated gammas") from exc
    if not out or any((not math.isfinite(x)) or x < 0 for x in out):
        raise argparse.ArgumentTypeError("Gammas must be finite and non-negative")
    return tuple(sorted(set(round(x, 8) for x in out)))


def gtag(x: float) -> str:
    s = f"{x:.6f}".rstrip("0").rstrip(".") or "0"
    return s.replace(".", "p")


@dataclass(frozen=True)
class Condition:
    family: str
    gamma_early: float
    gamma_late: float
    switch_time: float = 0.5

    @property
    def name(self) -> str:
        return f"{self.family}_ge{gtag(self.gamma_early)}_gl{gtag(self.gamma_late)}"

    def payload(self) -> dict[str, Any]:
        if self.family not in FAMILIES:
            raise ValueError(self.family)
        return {
            "format": "eqvae_internal_two_stage_gamma_condition_v1",
            "name": self.name,
            "family": self.family,
            "early_depth": 4,
            "late_depth": 4 if self.family == "d4_to_d4" else 10,
            "gamma_early": float(self.gamma_early),
            "gamma_late": float(self.gamma_late),
            "switch_time": float(self.switch_time),
            "formula": "S + gamma_stage*(S-W_stage); early t<switch, late t>=switch",
        }


def runtime_paths(repo: Path, data: Path, adm_python: Path) -> dict[str, Path]:
    out = {
        "strong": data / "runs/sit-s-2_seed0/checkpoints/step_00800000.pt",
        "d4": data / "multiscale_guidance_study_v1/runs/depth4_v/checkpoints/step_00050000.pt",
        "d10": data / "multiscale_guidance_study_v1/runs/depth10_v/checkpoints/step_00050000.pt",
        "reference": data / "adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz",
        "compute_fid": repo / "experiments/compute_adm_fid.py",
        "adm_python": adm_python,
    }
    missing = [f"{k}: {v}" for k, v in out.items() if not v.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files:\n  " + "\n  ".join(missing))
    return out


def reusable(path: Path, cond: Condition, n: int, batch: int, seed: int) -> bool:
    if not path.is_file():
        return False
    try:
        r = read_json(path)
        if r.get("condition") != cond.payload():
            return False
        m, metrics = r["sampling_manifest"], r["metrics"]
        s = m["sampling"]
        return (
            int(s["num_samples"]) == n
            and int(s["batch_size"]) == batch
            and int(s["seed"]) == seed
            and bool(m["noise_sha256"])
            and bool(m["label_sha256"])
            and all(
                isinstance(metrics.get(k), (int, float))
                and math.isfinite(float(metrics[k]))
                for k in ("fid", "sfid", "inception_score")
            )
        )
    except Exception:
        return False


def load_repo_modules(repo: Path):
    sys.path.insert(0, str(repo))
    from experiments.imagenet100_sit_multiscale_models import (
        evaluate_source_with_heads,
        load_internal_head_for_source,
        load_sit_field_model,
    )
    from experiments.sample_imagenet100_sit_fid import (
        configure_cuda_allocator,
        decode_latents_in_chunks,
        official_pixel_quantization,
    )
    from experiments.train_imagenet100_sit_flow import (
        DEFAULT_OFFICIAL_SIT_REPO,
        LATENT_SHAPE,
        NUM_CLASSES,
        SD_VAE_SCALING_FACTOR,
        atomic_json_dump,
        load_official_sit_module,
    )
    return locals()


def worker(args: argparse.Namespace) -> None:
    import numpy as np
    import torch
    from torchdiffeq import odeint
    from torchvision.utils import save_image
    from diffusers.models import AutoencoderKL

    repo = Path(args.repo).resolve()
    data = Path(args.data).resolve()
    paths = runtime_paths(repo, data, Path(args.adm_python))
    payload = read_json(Path(args.condition_json))
    cond = Condition(
        str(payload["family"]),
        float(payload["gamma_early"]),
        float(payload["gamma_late"]),
        float(payload["switch_time"]),
    )
    if payload != cond.payload():
        raise ValueError("Non-canonical condition JSON")

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "condition_result.json"
    if reusable(result_path, cond, args.num_samples, args.batch_size, args.seed):
        print(json.dumps({"event": "reuse", "condition": cond.name}), flush=True)
        return

    M = load_repo_modules(repo)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    alloc = M["configure_cuda_allocator"](device, limit_gib=args.cuda_allocator_limit_gib)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    sit_module, source_meta = M["load_official_sit_module"](
        Path(M["DEFAULT_OFFICIAL_SIT_REPO"]).expanduser().resolve(), verify_source=True
    )
    strong, strong_sem, strong_meta = M["load_sit_field_model"](
        checkpoint_path=paths["strong"], weights="ema",
        sit_module=sit_module, source_metadata=source_meta, device=device
    )
    if strong_sem.prediction_target != "velocity":
        raise ValueError("Strong must be native velocity")

    # IMPORTANT: both families load/evaluate BOTH heads at every NFE.
    heads = {}
    for name, hp in (("depth4_v", paths["d4"]), ("depth10_v", paths["d10"])):
        heads[name] = M["load_internal_head_for_source"](
            checkpoint_path=hp, name=name, head_weights="ema", model=strong,
            sit_module=sit_module, source_checkpoint_path=paths["strong"],
            source_metadata=source_meta, device=device
        )

    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse", local_files_only=True
    ).to(device).eval().requires_grad_(False)

    class Field:
        def __init__(self, labels):
            self.labels = labels
            self.nfe = 0

        def __call__(self, t, z):
            self.nfe += 1
            times = t.expand(len(z))
            full, trained, _ = M["evaluate_source_with_heads"](
                strong, z, times, self.labels, heads=heads
            )
            if float(t.detach().float().item()) < cond.switch_time:
                head_name, gamma = "depth4_v", cond.gamma_early
            else:
                head_name = "depth4_v" if cond.family == "d4_to_d4" else "depth10_v"
                gamma = cond.gamma_late
            if gamma == 0.0:
                return full
            return full + gamma * (full - trained[head_name])

    images = np.empty((args.num_samples, 256, 256, 3), dtype=np.uint8)
    labels_np = np.empty(args.num_samples, dtype=np.int16)
    noise_hash, label_hash = hashlib.sha256(), hashlib.sha256()
    cursor = total_nfe = 0
    preview = None

    with torch.inference_mode():
        while cursor < args.num_samples:
            bs = min(args.batch_size, args.num_samples - cursor)
            bi = cursor // args.batch_size
            gen = torch.Generator(device=device).manual_seed(args.seed + bi)
            noise = torch.randn(bs, *M["LATENT_SHAPE"], generator=gen, device=device)
            labels = torch.randint(
                0, M["NUM_CLASSES"], (bs,), generator=gen, device=device
            )
            f = Field(labels)
            endpoint = odeint(
                f, noise.float(), torch.tensor([0.0, 1.0], device=device),
                method="dopri5", atol=args.atol, rtol=args.rtol
            )[-1]
            if not torch.isfinite(endpoint).all():
                raise FloatingPointError(cond.name)
            decoded = M["decode_latents_in_chunks"](
                vae, endpoint, scaling_factor=M["SD_VAE_SCALING_FACTOR"],
                chunk_size=args.vae_decode_batch_size
            )
            stop = cursor + bs
            images[cursor:stop] = M["official_pixel_quantization"](decoded)
            labels_np[cursor:stop] = labels.cpu().numpy().astype(np.int16, copy=False)
            noise_hash.update(noise.cpu().contiguous().numpy().tobytes())
            label_hash.update(labels.cpu().contiguous().numpy().tobytes())
            if preview is None:
                preview = decoded[: min(16, len(decoded))].cpu()
            total_nfe += f.nfe
            cursor = stop
            if cursor == bs or cursor == args.num_samples or cursor % 256 == 0:
                print(json.dumps(
                    {"condition": cond.name, "generated": cursor,
                     "total": args.num_samples, "last_batch_nfe": f.nfe}
                ), flush=True)

    sample_path = out / f"samples_n{args.num_samples}.npz"
    label_path = out / f"labels_n{args.num_samples}.npy"
    np.savez(sample_path, arr_0=images)
    np.save(label_path, labels_np, allow_pickle=False)
    save_image(preview, out / "preview.png", nrow=4, normalize=True, value_range=(-1, 1))

    manifest = {
        "format": "eqvae_internal_two_stage_gamma_samples_v1",
        "condition": cond.payload(),
        "sampling": {
            "num_samples": args.num_samples, "batch_size": args.batch_size,
            "seed": args.seed, "integrator": "dopri5",
            "atol": args.atol, "rtol": args.rtol
        },
        "strong": strong_meta,
        "heads": {
            n: {"depth": s.depth, "prediction_target": s.prediction_target,
                "checkpoint": s.checkpoint, "checkpoint_sha256": s.checkpoint_sha256}
            for n, s in heads.items()
        },
        "noise_sha256": noise_hash.hexdigest(),
        "label_sha256": label_hash.hexdigest(),
        "total_nfe": total_nfe,
        "samples": str(sample_path),
        "labels": str(label_path),
        **alloc,
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    M["atomic_json_dump"](manifest, out / "sampling_manifest.json")

    fid_path = out / "adm_metrics.json"
    env = os.environ.copy()
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    subprocess.run([
        str(paths["adm_python"]), str(paths["compute_fid"]),
        "--reference", str(paths["reference"]),
        "--samples", str(sample_path),
        "--batch-size", str(args.fid_batch_size),
        "--gpu-memory-fraction", str(args.fid_gpu_memory_fraction),
        "--output", str(fid_path),
    ], cwd=repo, env=env, check=True)

    metrics = read_json(fid_path)
    result = {
        "format": "eqvae_internal_two_stage_gamma_result_v1",
        "condition": cond.payload(),
        "sampling_manifest": manifest,
        "metrics": metrics,
        "sample_retained": bool(args.keep_samples),
    }
    M["atomic_json_dump"](result, result_path)
    if not args.keep_samples:
        sample_path.unlink(missing_ok=True)
    print(json.dumps(
        {"event": "complete", "condition": cond.name, "fid": metrics["fid"]}
    ), flush=True)


def run_one(
    *, script: Path, repo: Path, data: Path, adm_python: Path, gpu: int,
    root: Path, phase: str, cond: Condition, args: argparse.Namespace
) -> dict[str, Any]:
    out = root / phase / cond.family / cond.name
    out.mkdir(parents=True, exist_ok=True)
    cpath = out / "condition.json"
    atomic_json(cpath, cond.payload())
    rpath = out / "condition_result.json"
    if reusable(rpath, cond, args.num_samples, args.batch_size, args.seed):
        r = read_json(rpath)
        print(f"[reuse] {cond.name}: FID={float(r['metrics']['fid']):.4f}", flush=True)
        return r

    cmd = [
        sys.executable, str(script), "worker",
        "--repo", str(repo),
        "--data", str(data),
        "--adm-python", str(adm_python),
        "--condition-json", str(cpath),
        "--output-dir", str(out),
        "--num-samples", str(args.num_samples),
        "--batch-size", str(args.batch_size),
        "--vae-decode-batch-size", "2",
        "--seed", str(args.seed),
        "--atol", str(args.atol),
        "--rtol", str(args.rtol),
        "--cuda-allocator-limit-gib", str(args.cuda_allocator_limit_gib),
        "--fid-batch-size", str(args.fid_batch_size),
        "--fid-gpu-memory-fraction", str(args.fid_gpu_memory_fraction),
    ]
    if args.keep_samples:
        cmd.append("--keep-samples")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    log = out / "run.log"
    with log.open("w", encoding="utf-8") as h:
        proc = subprocess.run(
            cmd, cwd=repo, env=env, stdout=h, stderr=subprocess.STDOUT, text=True
        )
    if proc.returncode:
        tail = "\n".join(log.read_text(encoding="utf-8").splitlines()[-40:])
        raise RuntimeError(f"{cond.name} failed on GPU {gpu}\n{tail}")

    r = read_json(rpath)
    print(f"[GPU {gpu}] {cond.name}: FID={float(r['metrics']['fid']):.4f}", flush=True)
    return r


def run_jobs(
    jobs: list[tuple[str, Condition]], *, gpus: tuple[int, ...], script: Path,
    repo: Path, data: Path, adm_python: Path, root: Path, args: argparse.Namespace
) -> list[dict[str, Any]]:
    lanes = [[] for _ in gpus]
    for i, job in enumerate(jobs):
        lanes[i % len(gpus)].append(job)

    def lane(gpu: int, items):
        out = []
        for phase, cond in items:
            out.append(run_one(
                script=script, repo=repo, data=data, adm_python=adm_python,
                gpu=gpu, root=root, phase=phase, cond=cond, args=args
            ))
        return out

    all_results = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [
            pool.submit(lane, gpu, items)
            for gpu, items in zip(gpus, lanes)
            if items
        ]
        for fut in as_completed(futures):
            all_results.extend(fut.result())
    return all_results


def rows_from_phase(root: Path, phase: str) -> list[dict[str, Any]]:
    rows = []
    phase_dir = root / phase
    if not phase_dir.is_dir():
        return rows
    for rp in sorted(phase_dir.glob("*/*/condition_result.json")):
        r = read_json(rp)
        c, m, met = r["condition"], r["sampling_manifest"], r["metrics"]
        rows.append({
            "phase": phase,
            "family": c["family"],
            "gamma_early": float(c["gamma_early"]),
            "gamma_late": float(c["gamma_late"]),
            "switch_time": float(c["switch_time"]),
            "fid": float(met["fid"]),
            "sfid": float(met["sfid"]),
            "inception_score": float(met["inception_score"]),
            "total_nfe": int(m["total_nfe"]),
            "noise_sha256": m["noise_sha256"],
            "label_sha256": m["label_sha256"],
        })
    return rows


def dedup(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chosen = {}
    prio = {"coarse": 0, "refine": 1}
    for r in rows:
        k = (
            r["family"],
            round(float(r["gamma_early"]), 8),
            round(float(r["gamma_late"]), 8),
        )
        old = chosen.get(k)
        if old is None or prio.get(r["phase"], 0) >= prio.get(old["phase"], 0):
            chosen[k] = r
    return list(chosen.values())


def best_by_family(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for fam in FAMILIES:
        subset = [r for r in rows if r["family"] == fam]
        if not subset:
            raise RuntimeError(f"No rows for {fam}")
        out[fam] = min(subset, key=lambda r: r["fid"])
    return out


def verify_default_anchors(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    noises = {r["noise_sha256"] for r in rows}
    labels = {r["label_sha256"] for r in rows}
    if len(noises) != 1 or len(labels) != 1:
        raise RuntimeError(
            f"Not fully paired: noise groups={len(noises)}, label groups={len(labels)}"
        )

    if args.num_samples == 1000 and args.batch_size == 8 and args.seed == 0:
        noise, label = next(iter(noises)), next(iter(labels))
        if noise != EXPECTED_NOISE or label != EXPECTED_LABEL:
            raise RuntimeError(
                "Default sweep does not match historical v4 1K RNG set:\n"
                f"noise {noise} != {EXPECTED_NOISE}\nlabel {label} != {EXPECTED_LABEL}"
            )
        lookup = {
            (r["family"], round(r["gamma_early"], 6), round(r["gamma_late"], 6)): r
            for r in rows
        }
        a = lookup.get(("d4_to_d4", 0.25, 0.25))
        b = lookup.get(("d4_to_d10", 0.45, 0.45))
        if a is None or b is None:
            raise RuntimeError("Historical anchor conditions missing from tested grid")
        da = abs(a["fid"] - OLD_STATIC_D4_G025)
        db = abs(b["fid"] - OLD_D4_D10_G045)
        if da > ANCHOR_TOL or db > ANCHOR_TOL:
            raise RuntimeError(
                "Anchor reproduction failed:\n"
                f"d4->d4(.25,.25): {a['fid']:.6f} vs {OLD_STATIC_D4_G025:.6f}\n"
                f"d4->d10(.45,.45): {b['fid']:.6f} vs {OLD_D4_D10_G045:.6f}"
            )


def local_pairs(
    ge: float,
    gl: float,
    *,
    radius: float,
    step: float,
) -> set[tuple[float, float]]:
    if radius < 0 or step <= 0:
        raise ValueError("refine radius must be non-negative and step must be positive")
    count = int(round(radius / step))
    if abs(count * step - radius) > 1e-8:
        raise ValueError("--refine-radius must be an integer multiple of --refine-step")
    offsets = tuple(i * step for i in range(-count, count + 1))
    es = {round(max(0.0, ge + d), 8) for d in offsets}
    ls = {round(max(0.0, gl + d), 8) for d in offsets}
    return {(e, l) for e in es for l in ls}


def common_refine_pairs(
    coarse_best: dict[str, dict[str, Any]],
    *,
    radius: float,
    step: float,
) -> set[tuple[float, float]]:
    out = set()
    for fam in FAMILIES:
        r = coarse_best[fam]
        out |= local_pairs(
            r["gamma_early"],
            r["gamma_late"],
            radius=radius,
            step=step,
        )
    return out


def write_summary(root: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    rows = dedup(rows)
    verify_default_anchors(rows, args)
    rows.sort(key=lambda r: (r["family"], r["gamma_early"], r["gamma_late"]))

    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    all_csv = summary_dir / "all_conditions.csv"
    with all_csv.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    best = best_by_family(rows)
    best_csv = summary_dir / "best_by_family.csv"
    with best_csv.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(best["d4_to_d4"].keys()))
        w.writeheader()
        w.writerow(best["d4_to_d4"])
        w.writerow(best["d4_to_d10"])

    maps = {
        fam: {
            (round(r["gamma_early"], 8), round(r["gamma_late"], 8)): r
            for r in rows if r["family"] == fam
        }
        for fam in FAMILIES
    }
    common = sorted(set(maps["d4_to_d4"]) & set(maps["d4_to_d10"]))
    id_rows = []
    for ge, gl in common:
        a = maps["d4_to_d4"][(ge, gl)]
        b = maps["d4_to_d10"][(ge, gl)]
        id_rows.append({
            "gamma_early": ge,
            "gamma_late": gl,
            "fid_d4_to_d4": a["fid"],
            "fid_d4_to_d10": b["fid"],
            "late_d10_identity_benefit_fid": a["fid"] - b["fid"],
            "sfid_d4_to_d4": a["sfid"],
            "sfid_d4_to_d10": b["sfid"],
            "is_d4_to_d4": a["inception_score"],
            "is_d4_to_d10": b["inception_score"],
        })

    identity_csv = summary_dir / "paired_identity_delta.csv"
    with identity_csv.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(id_rows[0].keys()))
        w.writeheader()
        w.writerows(id_rows)

    A, B = best["d4_to_d4"], best["d4_to_d10"]
    pair_A = (round(A["gamma_early"], 8), round(A["gamma_late"], 8))
    pair_B = (round(B["gamma_early"], 8), round(B["gamma_late"], 8))
    B_at_A = maps["d4_to_d10"][pair_A]
    A_at_B = maps["d4_to_d4"][pair_B]

    summary = {
        "format": "eqvae_internal_two_stage_gamma_sweep_summary_v1",
        "scientific_question": (
            "Does a late d10 head beat an optimized two-stage d4 amplitude schedule?"
        ),
        "best_d4_to_d4": A,
        "best_d4_to_d10": B,
        "optimized_late_d10_benefit_fid": A["fid"] - B["fid"],
        "same_pair_controls": {
            "at_d4_to_d4_optimum": {
                "gamma_early": pair_A[0],
                "gamma_late": pair_A[1],
                "fid_d4_to_d4": A["fid"],
                "fid_d4_to_d10": B_at_A["fid"],
                "late_d10_identity_benefit_fid": A["fid"] - B_at_A["fid"],
            },
            "at_d4_to_d10_optimum": {
                "gamma_early": pair_B[0],
                "gamma_late": pair_B[1],
                "fid_d4_to_d4": A_at_B["fid"],
                "fid_d4_to_d10": B["fid"],
                "late_d10_identity_benefit_fid": A_at_B["fid"] - B["fid"],
            },
            "max_identity_benefit": max(
                id_rows, key=lambda r: r["late_d10_identity_benefit_fid"]
            ),
            "min_identity_benefit": min(
                id_rows, key=lambda r: r["late_d10_identity_benefit_fid"]
            ),
        },
        "protocol": {
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "switch_time": args.switch_time,
            "coarse_gammas": list(args.coarse_gammas),
            "same_candidate_pairs_for_both_families": True,
            "refine": args.refine,
            "refine_step": args.refine_step,
            "refine_radius": args.refine_radius,
            "late0_enabled": args.late0,
            "late0_early_gammas": list(args.late0_early_gammas),
        },
        "pairing": {
            "noise_sha256": rows[0]["noise_sha256"],
            "label_sha256": rows[0]["label_sha256"],
            "verified": True,
        },
        "files": {
            "all_conditions": str(all_csv),
            "best_by_family": str(best_csv),
            "paired_identity_delta": str(identity_csv),
        },
    }
    summary_json = summary_dir / "summary.json"
    atomic_json(summary_json, summary)

    print("\n=== FINAL ===")
    print(
        f"best d4->d4 : ge={A['gamma_early']:.3f}, gl={A['gamma_late']:.3f}, "
        f"FID={A['fid']:.4f}"
    )
    print(
        f"best d4->d10: ge={B['gamma_early']:.3f}, gl={B['gamma_late']:.3f}, "
        f"FID={B['fid']:.4f}"
    )
    print(
        f"optimized late-d10 benefit: {A['fid'] - B['fid']:+.4f} FID "
        "(positive means d10 still wins after both families are tuned)"
    )
    print(
        "same-pair d10 benefit at d4->d4 optimum: "
        f"{A['fid'] - B_at_A['fid']:+.4f} FID"
    )
    print(
        "same-pair d10 benefit at d4->d10 optimum: "
        f"{A_at_B['fid'] - B['fid']:+.4f} FID"
    )
    print(f"summary: {summary_json}")
    print(f"identity map: {identity_csv}")


def sweep(args: argparse.Namespace) -> None:
    repo, data, adm = detect_repo(), detect_data(), detect_adm_python()
    runtime_paths(repo, data, adm)
    root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root else data / "internal_head_two_stage_gamma_sweep_v1"
    )
    root.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()

    coarse_pairs = list(itertools.product(args.coarse_gammas, repeat=2))
    coarse_jobs = [
        ("coarse", Condition(fam, ge, gl, args.switch_time))
        for ge, gl in coarse_pairs
        for fam in FAMILIES
    ]

    atomic_json(root / "request.json", {
        "format": "eqvae_internal_two_stage_gamma_sweep_request_v1",
        "repo": str(repo), "data": str(data), "gpus": list(args.gpus),
        "num_samples": args.num_samples, "batch_size": args.batch_size,
        "seed": args.seed, "switch_time": args.switch_time,
        "coarse_gammas": list(args.coarse_gammas),
        "coarse_conditions": len(coarse_jobs), "refine": args.refine,
        "late0": args.late0, "late0_early_gammas": list(args.late0_early_gammas),
    })

    print("=== TWO-STAGE GAMMA SWEEP ===")
    print(f"GPUs: {args.gpus}")
    print(f"output: {root}")
    print(f"switch: {args.switch_time}")
    print(f"coarse gammas: {args.coarse_gammas}")
    print(f"coarse conditions: {len(coarse_jobs)}")

    if args.dry_run:
        for phase, c in coarse_jobs[:12]:
            print(f"  {phase}: {c.name}")
        if len(coarse_jobs) > 12:
            print(f"  ... +{len(coarse_jobs)-12}")
        if args.late0:
            print(
                f"late=0 strip: early={args.late0_early_gammas}, "
                f"conditions={2 * len(args.late0_early_gammas)}"
            )
        return

    # Fail-fast numerical preflight: reproduce the two historical anchors
    # before spending compute on the full 2-D grid.
    anchor_conditions = [
        Condition("d4_to_d4", 0.25, 0.25, args.switch_time),
        Condition("d4_to_d10", 0.45, 0.45, args.switch_time),
    ]
    anchor_jobs = [("coarse", c) for c in anchor_conditions]
    print("\npreflight anchors: d4->d4(.25,.25), d4->d10(.45,.45)")
    run_jobs(
        anchor_jobs, gpus=args.gpus, script=script, repo=repo, data=data,
        adm_python=adm, root=root, args=args
    )
    anchor_rows = rows_from_phase(root, "coarse")
    # The verifier accepts additional rows on resume, but at minimum the anchors
    # must reproduce the old v4 sweep under the default 1K protocol.
    verify_default_anchors(anchor_rows, args)
    print("anchor reproduction: OK")

    anchor_names = {c.name for c in anchor_conditions}
    remaining_coarse = [
        job for job in coarse_jobs if job[1].name not in anchor_names
    ]
    run_jobs(
        remaining_coarse, gpus=args.gpus, script=script, repo=repo, data=data,
        adm_python=adm, root=root, args=args
    )
    coarse_rows = rows_from_phase(root, "coarse")
    verify_default_anchors(coarse_rows, args)
    cbest = best_by_family(coarse_rows)

    print("\n=== COARSE BEST ===")
    for fam in FAMILIES:
        r = cbest[fam]
        print(
            f"{fam}: ge={r['gamma_early']:.3f}, gl={r['gamma_late']:.3f}, "
            f"FID={r['fid']:.4f}"
        )

    if args.refine:
        rpairs = common_refine_pairs(cbest, radius=args.refine_radius, step=args.refine_step)
        coarse_set = {(round(a, 8), round(b, 8)) for a, b in coarse_pairs}
        rpairs = {p for p in rpairs if p not in coarse_set}
        refine_jobs = [
            ("refine", Condition(fam, ge, gl, args.switch_time))
            for ge, gl in sorted(rpairs)
            for fam in FAMILIES
        ]
        print(
            f"common refinement pairs/family: {len(rpairs)}, "
            f"refine conditions: {len(refine_jobs)}"
        )
        run_jobs(
            refine_jobs, gpus=args.gpus, script=script, repo=repo, data=data,
            adm_python=adm, root=root, args=args
        )

    # Targeted late-guidance shutdown strip. When gamma_late=0, the two
    # families are mathematically identical after t=switch. We still evaluate
    # both as an implementation-level equality check and to keep tables symmetric.
    late0_jobs = [
        ("late0", Condition(fam, ge, 0.0, args.switch_time))
        for ge in args.late0_early_gammas
        for fam in FAMILIES
    ]
    if args.late0:
        print(
            f"\nlate=0 strip: early gammas={args.late0_early_gammas}; "
            f"conditions={len(late0_jobs)}"
        )
        run_jobs(
            late0_jobs, gpus=args.gpus, script=script, repo=repo, data=data,
            adm_python=adm, root=root, args=args
        )

        # Exact field equality is expected pairwise when gamma_late=0.
        late0_rows = rows_from_phase(root, "late0")
        late0_map = {
            (r["family"], round(r["gamma_early"], 8)): r
            for r in late0_rows
        }
        for ge in args.late0_early_gammas:
            a = late0_map.get(("d4_to_d4", round(ge, 8)))
            b = late0_map.get(("d4_to_d10", round(ge, 8)))
            if a is None or b is None:
                raise RuntimeError(f"missing late0 pair for gamma_early={ge}")
            if abs(float(a["fid"]) - float(b["fid"])) > 1e-9:
                raise RuntimeError(
                    "late=0 families should be exactly equivalent but FID differs: "
                    f"ge={ge}, d4->d4={a['fid']}, d4->d10={b['fid']}"
                )

    rows = (
        rows_from_phase(root, "coarse")
        + rows_from_phase(root, "refine")
        + rows_from_phase(root, "late0")
    )
    write_summary(root, rows, args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")

    s = sub.add_parser("sweep")
    s.add_argument("--gpus", type=parse_gpus, default=parse_gpus("1,3"))
    s.add_argument(
        "--coarse-gammas",
        type=parse_gammas,
        default=DEFAULT_COARSE,
        help="common comma-separated gamma grid for BOTH stages and BOTH families",
    )
    s.add_argument("--switch-time", type=float, default=0.5)
    s.add_argument("--num-samples", type=int, default=1000)
    s.add_argument("--batch-size", type=int, default=8)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--atol", type=float, default=1e-6)
    s.add_argument("--rtol", type=float, default=1e-3)
    s.add_argument("--cuda-allocator-limit-gib", type=float, default=6.0)
    s.add_argument("--fid-batch-size", type=int, default=8)
    s.add_argument("--fid-gpu-memory-fraction", type=float, default=0.25)
    s.add_argument("--output-root", type=Path)
    s.add_argument("--refine", action=argparse.BooleanOptionalAction, default=True)
    s.add_argument(
        "--refine-step",
        type=float,
        default=0.025,
        help="local refinement gamma spacing; default 0.025",
    )
    s.add_argument(
        "--refine-radius",
        type=float,
        default=0.05,
        help="local refinement radius around each coarse optimum; default 0.05",
    )
    s.add_argument(
        "--late0",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also evaluate a targeted gamma_late=0 strip; default enabled",
    )
    s.add_argument(
        "--late0-early-gammas",
        type=parse_gammas,
        default=DEFAULT_LATE0_EARLY,
        help="early-gamma values for the gamma_late=0 strip",
    )
    s.add_argument("--keep-samples", action="store_true")
    s.add_argument("--dry-run", action="store_true")

    w = sub.add_parser("worker")
    w.add_argument("--repo", required=True)
    w.add_argument("--data", required=True)
    w.add_argument("--adm-python", required=True)
    w.add_argument("--condition-json", required=True)
    w.add_argument("--output-dir", required=True)
    w.add_argument("--num-samples", type=int, required=True)
    w.add_argument("--batch-size", type=int, required=True)
    w.add_argument("--vae-decode-batch-size", type=int, default=2)
    w.add_argument("--seed", type=int, required=True)
    w.add_argument("--atol", type=float, default=1e-6)
    w.add_argument("--rtol", type=float, default=1e-3)
    w.add_argument("--cuda-allocator-limit-gib", type=float, default=6.0)
    w.add_argument("--fid-batch-size", type=int, default=8)
    w.add_argument("--fid-gpu-memory-fraction", type=float, default=0.25)
    w.add_argument("--keep-samples", action="store_true")
    return parser


def validate_sweep(args: argparse.Namespace) -> None:
    if not 0 < args.switch_time < 1:
        raise ValueError("--switch-time must be in (0,1)")
    if args.num_samples <= 0 or args.batch_size <= 0:
        raise ValueError("sample count and batch size must be positive")
    if args.cuda_allocator_limit_gib <= 0:
        raise ValueError("allocator limit must be positive")
    if args.fid_batch_size <= 0:
        raise ValueError("FID batch size must be positive")
    if args.refine_step <= 0 or args.refine_radius < 0:
        raise ValueError("refine step must be positive and refine radius non-negative")
    q = args.refine_radius / args.refine_step
    if abs(q - round(q)) > 1e-8:
        raise ValueError("--refine-radius must be an integer multiple of --refine-step")
    if not 0 < args.fid_gpu_memory_fraction <= 1:
        raise ValueError("FID GPU memory fraction must be in (0,1]")


def self_test() -> None:
    a = Condition("d4_to_d4", 0.25, 0.45, 0.5)
    b = Condition("d4_to_d10", 0.25, 0.45, 0.5)
    assert a.payload()["late_depth"] == 4
    assert b.payload()["late_depth"] == 10
    coarse = set(itertools.product(DEFAULT_COARSE, repeat=2))
    assert (0.25, 0.25) in coarse
    assert (0.45, 0.45) in coarse

    fake = {
        "d4_to_d4": {"gamma_early": 0.45, "gamma_late": 0.25, "fid": 1.0},
        "d4_to_d10": {"gamma_early": 0.65, "gamma_late": 0.45, "fid": 0.9},
    }
    rp = common_refine_pairs(fake, radius=0.05, step=0.025)
    assert (0.45, 0.25) in rp
    assert (0.65, 0.45) in rp
    assert all(x >= 0 and y >= 0 for x, y in rp)
    assert DEFAULT_LATE0_EARLY[0] == 0.45
    assert DEFAULT_LATE0_EARLY[-1] == 0.75
    assert len(DEFAULT_LATE0_EARLY) == 13
    print("SELF_TEST_OK")
    print("families:", FAMILIES)
    print("coarse:", DEFAULT_COARSE)
    print("coarse conditions:", len(coarse) * 2)
    print("anchors included: yes")


def main() -> None:
    argv = sys.argv[1:]
    if argv == ["--self-test"]:
        self_test()
        return
    if not argv:
        argv = ["sweep"]
    elif argv[0] not in {"sweep", "worker"}:
        argv = ["sweep", *argv]

    args = build_parser().parse_args(argv)
    if args.command == "worker":
        worker(args)
    else:
        validate_sweep(args)
        sweep(args)


if __name__ == "__main__":
    main()