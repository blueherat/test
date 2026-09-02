#!/usr/bin/env python3
"""Run the paired compute-matched foresight screen on strong Internal Guidance."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLER = REPO_ROOT / "experiments/sample_imagenet100_sit_foresight_fixed_point.py"
FID_SCRIPT = REPO_ROOT / "experiments/compute_adm_fid.py"
DEFAULT_OUTPUT_ROOT = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/ig_best_depth4_foresight_v2/fid1k"
)
DEFAULT_STRONG_CHECKPOINT = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/"
    "sit-s-2_seed0/checkpoints/step_00800000.pt"
)
DEFAULT_DEPTH4_HEAD = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/"
    "multiscale_guidance_study_v1/runs/depth4_v/checkpoints/step_00050000.pt"
)
DEFAULT_REFERENCE_STATS = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/adm_reference_stats/"
    "imagenet100_validation_n5000_adm_stats.npz"
)
DEFAULT_ADM_PYTHON = Path("/data/shared/envs/adm-fid/bin/python")


@dataclass(frozen=True)
class Condition:
    name: str
    method: str
    num_steps: int
    rho: float = 1.0
    schedule: str = ""
    forwards_per_batch: int = 0


CONDITIONS = (
    Condition("ig_best_d4_closed40", "closed", 40, forwards_per_batch=40),
    Condition("ig_best_d4_closed88_matched", "closed", 88, forwards_per_batch=88),
    *(
        Condition(
            f"ig_best_d4_future_h5_rho{str(rho).replace('.', 'p')}_matched88",
            "future_dir_current_norm_ag",
            40,
            rho=rho,
            schedule="0:5:1,5:5:1",
            forwards_per_batch=88,
        )
        for rho in (0.5, 1.0, 2.0, 4.0)
    ),
)


def parse_int_list(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not values or len(values) != len(set(values)) or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("values must be unique non-negative integers")
    return values


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_stem(num_samples: int) -> str:
    if num_samples > 0 and num_samples % 1000 == 0:
        return f"fid{num_samples // 1000}k"
    return f"fid_n{num_samples}"


def selected_conditions(pattern_text: str) -> tuple[Condition, ...]:
    pattern = re.compile(pattern_text)
    result = tuple(condition for condition in CONDITIONS if pattern.search(condition.name))
    if not result:
        raise ValueError(f"condition regex matched nothing: {pattern_text!r}")
    return result


def output_dir(args: argparse.Namespace, seed: int, condition: Condition) -> Path:
    return args.output_root / f"seed{seed}" / condition.name


def valid_samples(path: Path, *, num_samples: int, seed: int) -> bool:
    manifest_path = path / "sampling_manifest.json"
    samples_path = path / f"samples_n{num_samples}.npz"
    if not manifest_path.is_file() or not samples_path.is_file():
        return False
    manifest = load_json(manifest_path)
    return (
        manifest.get("format")
        == "eqvae_imagenet100_sit_foresight_fixed_point_samples_v1"
        and manifest.get("family") == "ig"
        and int(manifest.get("requested_samples", -1)) == num_samples
        and int(manifest.get("global_seed", -1)) == seed
        and manifest.get("sample_rng_mode") == "per_batch"
    )


def _run(command: list[str], *, gpu: int, log_path: Path) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    for key, value in {
        "OMP_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "NUMEXPR_NUM_THREADS": "8",
        "TF_NUM_INTRAOP_THREADS": "8",
        "TF_NUM_INTEROP_THREADS": "2",
    }.items():
        env[key] = value
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if process.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-6000:]
        raise RuntimeError(f"command failed on GPU {gpu}: {' '.join(command)}\n{tail}")


def run_parallel(
    jobs: list[tuple[str, list[str], Path]], *, gpu_indices: tuple[int, ...]
) -> None:
    if not jobs:
        return
    with ThreadPoolExecutor(max_workers=len(gpu_indices)) as executor:
        pending = iter(jobs)
        active = {}
        for gpu in gpu_indices:
            try:
                name, command, log_path = next(pending)
            except StopIteration:
                break
            future = executor.submit(_run, command, gpu=gpu, log_path=log_path)
            active[future] = (gpu, name)
        while active:
            for future in as_completed(tuple(active)):
                gpu, name = active.pop(future)
                future.result()
                print(
                    json.dumps({"event": "job_complete", "gpu": gpu, "name": name}),
                    flush=True,
                )
                try:
                    next_name, command, log_path = next(pending)
                except StopIteration:
                    continue
                next_future = executor.submit(_run, command, gpu=gpu, log_path=log_path)
                active[next_future] = (gpu, next_name)
                break


def sample_jobs(args: argparse.Namespace) -> list[tuple[str, list[str], Path]]:
    jobs = []
    for seed in args.global_seeds:
        for condition in selected_conditions(args.condition_regex):
            target_dir = output_dir(args, seed, condition)
            name = f"seed{seed}/{condition.name}"
            if valid_samples(target_dir, num_samples=args.num_samples, seed=seed):
                print(json.dumps({"event": "reuse_samples", "name": name}), flush=True)
                continue
            command = [
                sys.executable,
                str(SAMPLER),
                "--family",
                "ig",
                "--method",
                condition.method,
                "--strong-checkpoint",
                str(args.strong_checkpoint),
                "--weights",
                "ema",
                "--internal-head",
                f"depth4_v={args.depth4_head}",
                "--internal-head-weights",
                "ema",
                "--ig-depths",
                "4",
                "--ag-gamma",
                "1",
                "--ig-gamma-segments",
                args.ig_gamma_segments,
                "--num-steps",
                str(condition.num_steps),
                "--foresight-schedule",
                condition.schedule,
                "--anchored-strength-multiplier",
                str(condition.rho),
                "--conjugate-flow-integrator",
                "rk4",
                "--num-samples",
                str(args.num_samples),
                "--batch-size",
                str(args.batch_size),
                "--vae-decode-batch-size",
                str(args.vae_decode_batch_size),
                "--diagnostic-samples",
                str(args.diagnostic_samples),
                "--global-seed",
                str(seed),
                "--sample-rng-mode",
                "per_batch",
                "--cuda-allocator-limit-gib",
                str(args.cuda_allocator_limit_gib),
                "--output-dir",
                str(target_dir),
                "--device",
                "cuda:0",
            ]
            jobs.append((name, command, target_dir / "sampling.log"))
    return jobs


def evaluation_jobs(args: argparse.Namespace) -> list[tuple[str, list[str], Path]]:
    jobs = []
    stem = metric_stem(args.num_samples)
    for seed in args.global_seeds:
        for condition in selected_conditions(args.condition_regex):
            target_dir = output_dir(args, seed, condition)
            name = f"seed{seed}/{condition.name}"
            result_path = target_dir / f"{stem}_adm_results.json"
            samples_path = target_dir / f"samples_n{args.num_samples}.npz"
            if result_path.is_file():
                result = load_json(result_path)
                if Path(result.get("samples", "")).resolve() == samples_path.resolve():
                    print(json.dumps({"event": "reuse_fid", "name": name}), flush=True)
                    continue
            if not samples_path.is_file():
                raise FileNotFoundError(samples_path)
            command = [
                str(args.adm_python),
                str(FID_SCRIPT),
                "--reference",
                str(args.reference_stats),
                "--samples",
                str(samples_path),
                "--batch-size",
                str(args.fid_batch_size),
                "--gpu-memory-fraction",
                str(args.fid_gpu_memory_fraction),
                "--output",
                str(result_path),
            ]
            jobs.append((name, command, target_dir / "evaluation.log"))
    return jobs


def validate_forward_count(
    manifest: dict, *, condition: Condition, num_samples: int, batch_size: int
) -> int:
    actual = sum(int(value) for value in manifest["model_forward_totals"].values())
    batches = math.ceil(num_samples / batch_size)
    expected = condition.forwards_per_batch * batches
    if actual != expected:
        raise ValueError(
            f"{condition.name}: expected {expected} model forwards, observed {actual}"
        )
    return actual


def write_summary(args: argparse.Namespace) -> None:
    conditions = selected_conditions(args.condition_regex)
    stem = metric_stem(args.num_samples)
    rows: list[dict[str, object]] = []
    for seed in args.global_seeds:
        seed_rows = []
        noise_hashes: set[str] = set()
        label_hashes: set[str] = set()
        for condition in conditions:
            target_dir = output_dir(args, seed, condition)
            manifest = load_json(target_dir / "sampling_manifest.json")
            metrics = load_json(target_dir / f"{stem}_adm_results.json")
            noise_hashes.add(str(manifest["noise_sha256"]))
            label_hashes.add(str(manifest["label_sha256"]))
            row = {
                "seed": seed,
                "condition": condition.name,
                "method": condition.method,
                "num_steps": condition.num_steps,
                "rho": condition.rho,
                "fid": float(metrics["fid"]),
                "sfid": float(metrics["sfid"]),
                "inception_score": float(metrics["inception_score"]),
                "model_forwards": validate_forward_count(
                    manifest,
                    condition=condition,
                    num_samples=args.num_samples,
                    batch_size=args.batch_size,
                ),
                "noise_sha256": manifest["noise_sha256"],
                "label_sha256": manifest["label_sha256"],
                "output_dir": str(target_dir),
            }
            seed_rows.append(row)
        if len(noise_hashes) != 1 or len(label_hashes) != 1:
            raise ValueError(f"seed {seed} did not use paired noise and labels")
        by_name = {str(row["condition"]): row for row in seed_rows}
        closed40 = by_name.get("ig_best_d4_closed40")
        closed88 = by_name.get("ig_best_d4_closed88_matched")
        for row in seed_rows:
            row["fid_delta_vs_closed40"] = (
                float(row["fid"]) - float(closed40["fid"])
                if closed40 is not None
                else None
            )
            row["fid_delta_vs_closed88"] = (
                float(row["fid"]) - float(closed88["fid"])
                if closed88 is not None
                else None
            )
            rows.append(row)

    aggregates = []
    for condition in conditions:
        condition_rows = [row for row in rows if row["condition"] == condition.name]
        aggregate = {"condition": condition.name, "sample_seeds": len(condition_rows)}
        for metric in (
            "fid",
            "sfid",
            "inception_score",
            "fid_delta_vs_closed40",
            "fid_delta_vs_closed88",
        ):
            values = [float(row[metric]) for row in condition_rows if row[metric] is not None]
            aggregate[f"{metric}_mean"] = statistics.mean(values) if values else None
            aggregate[f"{metric}_pstdev"] = statistics.pstdev(values) if values else None
        aggregates.append(aggregate)

    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / f"{stem}_paired_rows.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_root / f"{stem}_aggregate.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)
    payload = {
        "format": "eqvae_imagenet100_sit_internal_foresight_study_v1",
        "scope": (
            f"paired ADM FID with {args.num_samples} samples; "
            "fixed-step mechanism result"
        ),
        "ig": {
            "depth_schedule": [4],
            "gamma_segments": args.ig_gamma_segments,
        },
        "compute_match": (
            "future H5 conditions and closed88 use 88 shared-backbone-equivalent "
            "forwards per sample batch"
        ),
        "rows": rows,
        "aggregates": aggregates,
    }
    (args.output_root / f"{stem}_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


def main(args: argparse.Namespace) -> None:
    args.output_root = args.output_root.expanduser().resolve()
    for path in (
        args.strong_checkpoint,
        args.depth4_head,
        args.reference_stats,
        args.adm_python,
    ):
        if not path.expanduser().resolve().is_file():
            raise FileNotFoundError(path)
    if not args.skip_sampling:
        run_parallel(sample_jobs(args), gpu_indices=args.gpu_indices)
    if not args.skip_evaluation:
        run_parallel(evaluation_jobs(args), gpu_indices=args.gpu_indices)
        write_summary(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--strong-checkpoint", type=Path, default=DEFAULT_STRONG_CHECKPOINT)
    parser.add_argument("--depth4-head", type=Path, default=DEFAULT_DEPTH4_HEAD)
    parser.add_argument("--reference-stats", type=Path, default=DEFAULT_REFERENCE_STATS)
    parser.add_argument("--adm-python", type=Path, default=DEFAULT_ADM_PYTHON)
    parser.add_argument("--gpu-indices", type=parse_int_list, default=(0, 1, 2, 3))
    parser.add_argument("--global-seeds", type=parse_int_list, default=(0, 1))
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch 8 reproduces the validated depth4 sweep's paired sample bank.",
    )
    parser.add_argument("--vae-decode-batch-size", type=int, default=4)
    parser.add_argument("--diagnostic-samples", type=int, default=16)
    parser.add_argument(
        "--ig-gamma-segments",
        default=".25:.6,.5:.7,1:0",
        help="Validated depth4 schedule: [0,.25)=.6, [.25,.5)=.7, [.5,1]=0.",
    )
    parser.add_argument("--cuda-allocator-limit-gib", type=float, default=6.0)
    parser.add_argument("--fid-batch-size", type=int, default=16)
    parser.add_argument("--fid-gpu-memory-fraction", type=float, default=0.25)
    parser.add_argument("--condition-regex", default=".*")
    parser.add_argument("--skip-sampling", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
