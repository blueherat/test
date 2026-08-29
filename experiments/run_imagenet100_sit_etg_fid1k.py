#!/usr/bin/env python3
"""Run the paired v800 Error-Triangulated Guidance FID-1K screen."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from experiments.run_imagenet100_sit_fid_curve import DEFAULT_ADM_PYTHON
    from experiments.train_imagenet100_sit_flow import atomic_json_dump, sha256_file
except ModuleNotFoundError:
    from run_imagenet100_sit_fid_curve import DEFAULT_ADM_PYTHON
    from train_imagenet100_sit_flow import atomic_json_dump, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/"
    "error_triangulated_guidance_v800_depth8_v1/calibration.json"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_CALIBRATION.parent / "fid1k"
DEFAULT_REFERENCE = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/adm_reference_stats/"
    "imagenet100_validation_n5000_adm_stats.npz"
)


def parse_gpu_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or len(values) != len(set(values)) or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("GPU list must contain unique non-negative indices")
    return values


def format_float(value: float) -> str:
    return format(float(value), ".8g").replace("-", "m").replace(".", "p")


def condition_name(mode: str, gamma: float) -> str:
    return f"{mode}_g{format_float(gamma)}"


def formal_conditions() -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = [("baseline", 0.0)]
    sweeps = {
        "single_velocity": (0.25, 0.30, 0.35, 0.40, 0.45, 0.50),
        "single_clean": (0.08, 0.12, 0.15, 0.18, 0.22),
        "single_epsilon": (0.10, 0.14, 0.18, 0.22, 0.26),
        "mean": (0.10, 0.20, 0.30, 0.40, 0.50),
        "etg_scalar": (0.25, 0.30, 0.35, 0.40, 0.45, 0.50),
        "etg_channel": (0.25, 0.30, 0.35, 0.40, 0.45, 0.50),
        "private_velocity": (-2.0, -1.0, 1.0, 2.0),
        "private_clean": (-2.0, -1.0, 1.0, 2.0),
        "private_epsilon": (-2.0, -1.0, 1.0, 2.0),
    }
    for mode, gammas in sweeps.items():
        rows.extend((mode, gamma) for gamma in gammas)
    return rows


def smoke_conditions() -> list[tuple[str, float]]:
    return [
        ("baseline", 0.0),
        ("single_velocity", 0.4),
        ("mean", 0.4),
        ("etg_scalar", 0.4),
        ("etg_channel", 0.4),
        ("private_velocity", 1.0),
    ]


def run_command(command: list[str], *, env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n$ " + " ".join(command) + "\n")
        handle.flush()
        subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def valid_sampling(output_dir: Path, *, mode: str, gamma: float, args) -> bool:
    manifest_path = output_dir / "sampling_manifest.json"
    sample_path = output_dir / f"samples_unguided_n{args.num_samples}.npz"
    if not manifest_path.is_file() or not sample_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "format": "eqvae_imagenet100_sit_etg_samples_v1",
        "calibration_sha256": sha256_file(args.calibration),
        "mode": mode,
        "gamma": float(gamma),
        "requested_samples": args.num_samples,
        "world_size": 1,
        "per_rank_batch_size": args.per_rank_batch_size,
        "global_seed": args.global_seed,
    }
    mismatch = {key: (manifest.get(key), value) for key, value in expected.items() if manifest.get(key) != value}
    if mismatch:
        raise ValueError(f"incompatible sampling artifact at {output_dir}: {mismatch}")
    return True


def sample_one(
    condition: tuple[str, float],
    *,
    gpu: int,
    args,
) -> tuple[str, float]:
    mode, gamma = condition
    name = condition_name(mode, gamma)
    output_dir = args.output_root / name
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("PYTHONUNBUFFERED", "1")
    if not valid_sampling(output_dir, mode=mode, gamma=gamma, args=args):
        run_command(
            [
                args.torchrun,
                "--standalone",
                "--nproc_per_node=1",
                str(REPO_ROOT / "experiments/sample_imagenet100_sit_etg_fid.py"),
                "--calibration",
                str(args.calibration),
                "--output-dir",
                str(output_dir),
                "--mode",
                mode,
                "--gamma",
                str(gamma),
                "--num-samples",
                str(args.num_samples),
                "--per-rank-batch-size",
                str(args.per_rank_batch_size),
                "--vae-decode-batch-size",
                str(args.vae_decode_batch_size),
                "--cuda-allocator-limit-gib",
                str(args.cuda_allocator_limit_gib),
                "--global-seed",
                str(args.global_seed),
                "--precision",
                "fp32",
                "--allow-tf32",
            ],
            env=env,
            log_path=output_dir / "sampling.log",
        )
    print(
        json.dumps(
            {"event": "sampling_complete", "condition": name, "gpu": gpu}
        ),
        flush=True,
    )
    return condition


def evaluate_one(
    condition: tuple[str, float],
    *,
    gpu: int,
    args,
) -> dict[str, object]:
    mode, gamma = condition
    name = condition_name(mode, gamma)
    output_dir = args.output_root / name
    if not valid_sampling(output_dir, mode=mode, gamma=gamma, args=args):
        raise RuntimeError(f"evaluation requested before valid sampling: {output_dir}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("PYTHONUNBUFFERED", "1")
    metric_path = output_dir / "fid5k_adm_results.json"
    if not metric_path.is_file():
        run_command(
            [
                str(args.adm_python),
                str(REPO_ROOT / "experiments/compute_adm_fid.py"),
                "--reference",
                str(args.reference),
                "--samples",
                str(output_dir / f"samples_unguided_n{args.num_samples}.npz"),
                "--batch-size",
                str(args.fid_batch_size),
                "--gpu-memory-fraction",
                str(args.fid_gpu_memory_fraction),
                "--output",
                str(metric_path),
            ],
            env=env,
            log_path=output_dir / "evaluation.log",
        )
    metric = json.loads(metric_path.read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "sampling_manifest.json").read_text(encoding="utf-8"))
    row = {
        "condition": name,
        "mode": mode,
        "gamma": float(gamma),
        "num_samples": args.num_samples,
        "global_seed": args.global_seed,
        "fid": float(metric["fid"]),
        "sfid": float(metric["sfid"]),
        "inception_score": float(metric["inception_score"]),
        "total_nfe": int(manifest["total_nfe"]),
        "noise_fingerprint": ";".join(manifest["rank_noise_sha256"]),
        "label_fingerprint": ";".join(manifest["rank_label_sha256"]),
        "sample_sha256": sha256_file(output_dir / f"samples_unguided_n{args.num_samples}.npz"),
        "gpu": gpu,
    }
    print(json.dumps({"event": "condition_complete", **row}), flush=True)
    return row


def summarize(rows: list[dict[str, object]], output_root: Path, calibration: Path) -> None:
    rows.sort(key=lambda row: (str(row["mode"]), float(row["gamma"])))
    fingerprints = {(row["noise_fingerprint"], row["label_fingerprint"]) for row in rows}
    if len(fingerprints) != 1:
        raise ValueError("ETG conditions do not share identical noise and labels")
    csv_path = output_root / "etg_fid1k.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    best_by_mode = {}
    for mode in sorted({str(row["mode"]) for row in rows}):
        candidates = [row for row in rows if row["mode"] == mode]
        best_by_mode[mode] = min(candidates, key=lambda row: float(row["fid"]))
    baseline = best_by_mode["baseline"]
    summary = {
        "format": "eqvae_imagenet100_sit_etg_fid1k_summary_v1",
        "calibration": str(calibration),
        "calibration_sha256": sha256_file(calibration),
        "condition_count": len(rows),
        "paired_noise_and_labels": True,
        "baseline": baseline,
        "best_by_mode": best_by_mode,
        "best_overall": min(rows, key=lambda row: float(row["fid"])),
        "csv": str(csv_path),
    }
    atomic_json_dump(summary, output_root / "summary.json")
    print(json.dumps(summary, indent=2), flush=True)


def sample_gpu_queue(gpu: int, conditions: list[tuple[str, float]], args) -> None:
    for condition in conditions:
        sample_one(condition, gpu=gpu, args=args)


def evaluate_gpu_queue(gpu: int, conditions: list[tuple[str, float]], args) -> list[dict[str, object]]:
    return [evaluate_one(condition, gpu=gpu, args=args) for condition in conditions]


def main(args: argparse.Namespace) -> None:
    args.calibration = args.calibration.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.reference = args.reference.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    conditions = smoke_conditions() if args.profile == "smoke" else formal_conditions()
    queues = {gpu: [] for gpu in args.gpus}
    for index, condition in enumerate(conditions):
        queues[args.gpus[index % len(args.gpus)]].append(condition)
    # Sampling and ADM FID have very different bottlenecks.  Finish the GPU-heavy
    # sampling stage first, then limit concurrent covariance sqrtm evaluations.
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = {
            executor.submit(sample_gpu_queue, gpu, queue, args): gpu
            for gpu, queue in queues.items()
            if queue
        }
        for future in as_completed(futures):
            future.result()

    fid_gpus = args.gpus[: min(args.fid_workers, len(args.gpus))]
    fid_queues = {gpu: [] for gpu in fid_gpus}
    for index, condition in enumerate(conditions):
        fid_queues[fid_gpus[index % len(fid_gpus)]].append(condition)
    rows = []
    with ThreadPoolExecutor(max_workers=len(fid_gpus)) as executor:
        futures = {
            executor.submit(evaluate_gpu_queue, gpu, queue, args): gpu
            for gpu, queue in fid_queues.items()
            if queue
        }
        for future in as_completed(futures):
            rows.extend(future.result())
    summarize(rows, args.output_root, args.calibration)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--adm-python", type=Path, default=DEFAULT_ADM_PYTHON)
    parser.add_argument("--torchrun", default="torchrun")
    parser.add_argument("--gpus", type=parse_gpu_list, default=[0, 1, 2, 3])
    parser.add_argument("--profile", choices=("smoke", "formal"), default="formal")
    parser.add_argument("--num-samples", type=int, default=1_000)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--per-rank-batch-size", type=int, default=8)
    parser.add_argument("--vae-decode-batch-size", type=int, default=2)
    parser.add_argument("--cuda-allocator-limit-gib", type=float, default=6.0)
    parser.add_argument("--fid-batch-size", type=int, default=8)
    parser.add_argument("--fid-gpu-memory-fraction", type=float, default=0.35)
    parser.add_argument("--fid-workers", type=int, default=2)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
