#!/usr/bin/env python3
"""Small paired FID-1K gamma search for v800 strong with v270/v400 weak checkpoints.

Thin orchestrator around the repository's existing
`run_imagenet100_sit_static_pair_fid5k.py` pipeline.

Guidance convention here:
    V_guided = V800 + gamma * (V800 - Vweak)

The existing static-pair runner uses:
    V = anchor + scale * (other - anchor)

therefore:
    scale = -gamma

Default:
    v270 on GPU 1
    v400 on GPU 3
    gamma in {1.5, 2.0, 2.5, 3.0, 3.5, 4.0}
    1,000 paired samples per condition
    GPU memory safety ceiling = 10,240 MiB
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
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
INNER_RUNNER = REPO_ROOT / "experiments/run_imagenet100_sit_static_pair_fid5k.py"

DEFAULT_DATA_ROOT = Path("/home/zhoushunyu/data/eqvae/imagenet_sit_flow")
DEFAULT_RUN_DIR = DEFAULT_DATA_ROOT / "runs/sit-s-2_seed0/checkpoints"
DEFAULT_V270 = DEFAULT_RUN_DIR / "step_00270000.pt"
DEFAULT_V400 = DEFAULT_RUN_DIR / "step_00400000.pt"
DEFAULT_V800 = DEFAULT_RUN_DIR / "step_00800000.pt"
DEFAULT_REFERENCE = (
    DEFAULT_DATA_ROOT
    / "adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz"
)
DEFAULT_OUTPUT_ROOT = (
    DEFAULT_DATA_ROOT / "v800_v270_v400_gamma_search_fid1k_v2"
)
DEFAULT_GAMMAS = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0)


def parse_gpu_pair(value: str) -> tuple[str, str]:
    parts = [x.strip() for x in value.split(",") if x.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "--gpus must contain exactly two GPU ids, e.g. 1,3"
        )
    if parts[0] == parts[1]:
        raise argparse.ArgumentTypeError("--gpus must contain two different GPU ids")
    if any(not x.isdigit() for x in parts):
        raise argparse.ArgumentTypeError("GPU ids must be non-negative integers")
    return parts[0], parts[1]


def validate_gammas(values: Iterable[float]) -> tuple[float, ...]:
    gammas = tuple(float(v) for v in values)
    if not gammas:
        raise ValueError("gamma list must be non-empty")
    if any((not math.isfinite(v)) or v <= 0 for v in gammas):
        raise ValueError("all gammas must be finite and > 0")
    if len(set(gammas)) != len(gammas):
        raise ValueError("gammas must not contain duplicates")
    return gammas


def build_command(
    *,
    weak_name: str,
    weak_checkpoint: Path,
    gpu: str,
    args: argparse.Namespace,
) -> tuple[list[str], Path]:
    out = args.output_root / weak_name
    scales = [-float(g) for g in args.gammas]

    cmd = [
        sys.executable,
        str(INNER_RUNNER),
        "--anchor-checkpoint",
        str(args.strong_checkpoint),
        "--anchor-field",
        "auto",
        "--other-checkpoint",
        str(weak_checkpoint),
        "--other-field",
        "auto",
        "--allow-step-mismatch",
        "--control-mode",
        "full_pair",
        "--output-root",
        str(out),
        "--reference",
        str(args.reference),
        "--num-samples",
        str(args.num_samples),
        "--global-seed",
        str(args.seed),
        "--cuda-allocator-limit-gib",
        str(args.cuda_allocator_limit_gib),
        "--gpu-memory-ceiling-mib",
        str(args.gpu_memory_ceiling_mib),
        "--sampling-cuda-visible-devices",
        gpu,
        "--fid-cuda-visible-devices",
        gpu,
        "--scales",
        *[repr(s) for s in scales],
    ]

    if args.per_rank_batch_size is not None:
        cmd += ["--per-rank-batch-size", str(args.per_rank_batch_size)]
    if args.vae_decode_batch_size is not None:
        cmd += ["--vae-decode-batch-size", str(args.vae_decode_batch_size)]
    if args.fid_batch_size is not None:
        cmd += ["--fid-batch-size", str(args.fid_batch_size)]

    return cmd, out


def read_result_csv(output_root: Path, weak_name: str) -> list[dict[str, object]]:
    csv_path = output_root / "field_control_fid5k.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"missing result CSV for {weak_name}: {csv_path}")

    rows: list[dict[str, object]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            scale = float(row["scale"])
            rows.append(
                {
                    "weak": weak_name,
                    "gamma": -scale,
                    "scale_internal": scale,
                    "fid": float(row["fid"]),
                    "sfid": float(row["sfid"]),
                    "inception_score": float(row["inception_score"]),
                    "num_samples": int(row["num_samples"]),
                    "noise_fingerprint": row["noise_fingerprint"],
                    "label_fingerprint": row["label_fingerprint"],
                    "total_nfe": int(row["total_nfe"]),
                    "total_model_forwards": int(row["total_model_forwards"]),
                }
            )
    rows.sort(key=lambda r: float(r["gamma"]))
    return rows


def write_summary(args: argparse.Namespace) -> None:
    all_rows: list[dict[str, object]] = []
    for weak_name in ("v270", "v400"):
        all_rows.extend(read_result_csv(args.output_root / weak_name, weak_name))

    expected = {
        (weak, float(gamma))
        for weak in ("v270", "v400")
        for gamma in args.gammas
    }
    observed = {(str(r["weak"]), float(r["gamma"])) for r in all_rows}
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise RuntimeError(
            f"incomplete result grid; missing={missing}, extra={extra}"
        )

    noise = {str(r["noise_fingerprint"]) for r in all_rows}
    labels = {str(r["label_fingerprint"]) for r in all_rows}
    if len(noise) != 1 or len(labels) != 1:
        raise RuntimeError(
            "v270/v400 sweeps are not fully paired: noise/label fingerprints differ"
        )

    summary_dir = args.output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    csv_path = summary_dir / "v800_v270_v400_gamma_search_fid1k.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    best_rows: list[dict[str, object]] = []
    for weak in ("v270", "v400"):
        subset = [r for r in all_rows if r["weak"] == weak]
        best = min(subset, key=lambda r: float(r["fid"]))
        tested = sorted(float(r["gamma"]) for r in subset)
        best_gamma = float(best["gamma"])
        best_rows.append(
            {
                "weak": weak,
                "best_gamma": best_gamma,
                "best_fid": float(best["fid"]),
                "sfid_at_best": float(best["sfid"]),
                "is_at_best": float(best["inception_score"]),
                "at_search_boundary": best_gamma in (tested[0], tested[-1]),
            }
        )

    best_csv = summary_dir / "best_gamma_by_weak.csv"
    with best_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(best_rows[0].keys()))
        writer.writeheader()
        writer.writerows(best_rows)

    summary = {
        "protocol": "v800_v270_v400_static_closed_guidance_gamma_search_fid1k_v2",
        "formula": "V_guided = V800 + gamma * (V800 - Vweak)",
        "strong": str(args.strong_checkpoint),
        "weak_checkpoints": {
            "v270": str(args.v270_checkpoint),
            "v400": str(args.v400_checkpoint),
        },
        "gammas": list(args.gammas),
        "num_samples": args.num_samples,
        "seed": args.seed,
        "comparison_is_paired": True,
        "noise_fingerprint": next(iter(noise)),
        "label_fingerprint": next(iter(labels)),
        "best": best_rows,
        "csv": str(csv_path),
        "best_csv": str(best_csv),
    }
    (summary_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\n=== BEST GAMMA ===")
    for row in best_rows:
        boundary = "  [BOUNDARY -> extend search]" if row["at_search_boundary"] else ""
        print(
            f"{row['weak']}: gamma={row['best_gamma']:.3g}, "
            f"FID-1K={row['best_fid']:.4f}{boundary}"
        )
    print(f"\ncombined CSV: {csv_path}")
    print(f"best CSV:     {best_csv}")


def check_inputs(args: argparse.Namespace) -> None:
    required = {
        "inner runner": INNER_RUNNER,
        "v800": args.strong_checkpoint,
        "v270": args.v270_checkpoint,
        "v400": args.v400_checkpoint,
        "ADM reference": args.reference,
    }
    missing = [
        f"{name}: {path}"
        for name, path in required.items()
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "missing required files:\n  " + "\n  ".join(missing)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=parse_gpu_pair, default=parse_gpu_pair("1,3"))
    parser.add_argument("--gammas", nargs="+", type=float, default=list(DEFAULT_GAMMAS))
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strong-checkpoint", type=Path, default=DEFAULT_V800)
    parser.add_argument("--v270-checkpoint", type=Path, default=DEFAULT_V270)
    parser.add_argument("--v400-checkpoint", type=Path, default=DEFAULT_V400)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--per-rank-batch-size", type=int)
    parser.add_argument("--vae-decode-batch-size", type=int)
    parser.add_argument("--fid-batch-size", type=int)
    parser.add_argument(
        "--cuda-allocator-limit-gib",
        type=float,
        default=4.0,
        help="PyTorch allocator limit forwarded to the inner sampler.",
    )
    parser.add_argument(
        "--gpu-memory-ceiling-mib",
        type=int,
        default=15 * 1024,
        help=(
            "Total physical-GPU memory safety ceiling used by the repository "
            "resource guard. Default 10240 MiB; v1 used the inner default "
            "8192 MiB and was killed at an observed 8653 MiB."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.gammas = validate_gammas(args.gammas)
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.cuda_allocator_limit_gib <= 0:
        raise ValueError("--cuda-allocator-limit-gib must be positive")
    if args.gpu_memory_ceiling_mib <= 0:
        raise ValueError("--gpu-memory-ceiling-mib must be positive")
    if args.cuda_allocator_limit_gib * 1024 >= args.gpu_memory_ceiling_mib:
        raise ValueError(
            "allocator limit must leave headroom below GPU memory ceiling"
        )

    args.strong_checkpoint = args.strong_checkpoint.expanduser().resolve()
    args.v270_checkpoint = args.v270_checkpoint.expanduser().resolve()
    args.v400_checkpoint = args.v400_checkpoint.expanduser().resolve()
    args.reference = args.reference.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()

    check_inputs(args)
    args.output_root.mkdir(parents=True, exist_ok=True)

    jobs = [
        ("v270", args.v270_checkpoint, args.gpus[0]),
        ("v400", args.v400_checkpoint, args.gpus[1]),
    ]

    print("Guidance: V800 + gamma * (V800 - Vweak)")
    print("gammas:", ", ".join(f"{g:g}" for g in args.gammas))
    print(f"num_samples: {args.num_samples}")
    print(
        f"memory guard: ceiling={args.gpu_memory_ceiling_mib} MiB, "
        f"allocator_limit={args.cuda_allocator_limit_gib:g} GiB"
    )
    for name, ckpt, gpu in jobs:
        print(f"{name}: GPU {gpu}, checkpoint={ckpt}")

    commands: list[tuple[str, list[str], Path]] = []
    for name, ckpt, gpu in jobs:
        cmd, out = build_command(
            weak_name=name,
            weak_checkpoint=ckpt,
            gpu=gpu,
            args=args,
        )
        commands.append((name, cmd, out))
        if args.dry_run:
            print(f"\n[{name}]")
            print(" ".join(cmd))

    if args.dry_run:
        print(f"\nconditions: {2 * len(args.gammas)}")
        return

    processes: list[tuple[str, subprocess.Popen[str], object]] = []
    try:
        for name, cmd, out in commands:
            log_path = args.output_root / f"{name}_runner.log"
            log_handle = log_path.open("a", encoding="utf-8")
            print(f"[launch] {name}; log={log_path}", flush=True)
            process = subprocess.Popen(
                cmd,
                cwd=REPO_ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                env=os.environ.copy(),
            )
            processes.append((name, process, log_handle))

        failures: list[tuple[str, int]] = []
        for name, process, log_handle in processes:
            returncode = process.wait()
            log_handle.close()
            if returncode != 0:
                failures.append((name, returncode))
            else:
                print(f"[complete] {name}", flush=True)

        if failures:
            details = ", ".join(
                f"{name}: exit {code}" for name, code in failures
            )
            raise RuntimeError(
                f"one or more sweeps failed ({details}). "
                f"Inspect {args.output_root}/*_runner.log; rerunning is resumable."
            )

        write_summary(args)

    finally:
        for _, process, log_handle in processes:
            if process.poll() is None:
                process.terminate()
            if not getattr(log_handle, "closed", True):
                log_handle.close()


if __name__ == "__main__":
    main()