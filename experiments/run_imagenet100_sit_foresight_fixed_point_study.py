#!/usr/bin/env python3
"""Run the paired CFG/FSG and AutoGuidance fixed-point ImageNet-100 study."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/foresight_fixed_point_v1/fid1k"
)
DEFAULT_REFERENCE_STATS = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/adm_reference_stats/"
    "imagenet100_validation_n5000_adm_stats.npz"
)
DEFAULT_ADM_PYTHON = Path("/data/shared/envs/adm-fid/bin/python")
PAPER_NFE50_SCHEDULE = "0:5:2,5:5:2,15:5:1"


@dataclass(frozen=True)
class Condition:
    name: str
    family: str
    method: str
    num_steps: int
    extra: tuple[str, ...] = ()
    schedule: str | None = None


CONDITIONS = (
    Condition("cfg_w1p5_closed40", "cfg", "closed", 40, ("--cfg-scale", "1.5")),
    Condition("cfg_w1p5_closed50", "cfg", "closed", 50, ("--cfg-scale", "1.5")),
    Condition("cfg_w1p5_fsg40", "cfg", "foresight", 40, ("--cfg-scale", "1.5")),
    Condition(
        "cfg_w1p5_fsg40_matched50",
        "cfg",
        "foresight",
        40,
        ("--cfg-scale", "1.5"),
    ),
    Condition("ag_g1_closed40", "ag", "closed", 40, ("--ag-gamma", "1")),
    Condition("ag_g1_closed50", "ag", "closed", 50, ("--ag-gamma", "1")),
    Condition(
        "ag_g1_fsg40_weakref",
        "ag",
        "foresight",
        40,
        ("--ag-gamma", "1", "--ag-reference", "weak"),
    ),
    Condition(
        "ag_g1_fsg40_strongref",
        "ag",
        "foresight",
        40,
        ("--ag-gamma", "1", "--ag-reference", "strong"),
    ),
    Condition("ag_g3_closed40", "ag", "closed", 40, ("--ag-gamma", "3")),
    Condition("ag_g3_closed50", "ag", "closed", 50, ("--ag-gamma", "3")),
    Condition(
        "ag_g3_fsg40_weakref",
        "ag",
        "foresight",
        40,
        ("--ag-gamma", "3", "--ag-reference", "weak"),
    ),
    Condition(
        "ag_g3_fsg40_strongref",
        "ag",
        "foresight",
        40,
        ("--ag-gamma", "3", "--ag-reference", "strong"),
    ),
    Condition("ag_g0_closed40", "ag", "closed", 40, ("--ag-gamma", "0")),
    Condition(
        "ag_g0_fsg40_strongforward_weakinverse",
        "ag",
        "foresight",
        40,
        (
            "--ag-gamma",
            "0",
            "--foresight-ag-gamma",
            "0",
            "--ag-reference",
            "weak",
        ),
    ),
    Condition(
        "ag_g1_fsg40_strongforward_weakinverse",
        "ag",
        "foresight",
        40,
        (
            "--ag-gamma",
            "1",
            "--foresight-ag-gamma",
            "0",
            "--ag-reference",
            "weak",
        ),
    ),
    Condition(
        "ag_g3_fsg40_strongforward_weakinverse",
        "ag",
        "foresight",
        40,
        (
            "--ag-gamma",
            "3",
            "--foresight-ag-gamma",
            "0",
            "--ag-reference",
            "weak",
        ),
    ),
    *(
        Condition(
            f"ag_g3_fsg40_strongforward_weakinverse_rho{str(rho).replace('.', 'p')}",
            "ag",
            "foresight",
            40,
            (
                "--ag-gamma",
                "3",
                "--foresight-ag-gamma",
                "0",
                "--ag-reference",
                "weak",
                "--foresight-relaxation",
                str(rho),
            ),
        )
        for rho in (0.05, 0.1, 0.25, 0.5)
    ),
    Condition("ag_g4_closed40", "ag", "closed", 40, ("--ag-gamma", "4")),
    Condition("ag_g5_closed40", "ag", "closed", 40, ("--ag-gamma", "5")),
    *(
        Condition(
            f"fag_stack_g3_rho{str(rho).replace('.', 'p')}",
            "ag",
            "foresight",
            40,
            (
                "--ag-gamma",
                "3",
                "--foresight-ag-gamma",
                "3",
                "--ag-reference",
                "strong",
                "--foresight-relaxation",
                str(rho),
            ),
        )
        for rho in (0.02, 0.05, 0.1, 0.2)
    ),
    Condition(
        "fag_replace_g3_dt_over_h",
        "ag",
        "foresight",
        40,
        (
            "--ag-gamma",
            "3",
            "--foresight-ag-gamma",
            "3",
            "--ag-reference",
            "strong",
            "--foresight-relaxation",
            "0.2",
            "--foresight-event-local-mode",
            "reference",
        ),
        schedule="0:5:1,5:5:1,15:5:1",
    ),
    *(
        Condition(
            f"fag_sparse_g3_rho{str(rho).replace('.', 'p')}",
            "ag",
            "foresight",
            40,
            (
                "--ag-gamma",
                "0",
                "--foresight-ag-gamma",
                "3",
                "--ag-reference",
                "strong",
                "--foresight-relaxation",
                str(rho),
            ),
        )
        for rho in (0.25, 0.5, 1.0)
    ),
    *(
        condition
        for gamma in (3.0, 4.0)
        for condition in (
            Condition(
                f"ag_g{int(gamma)}_closed42",
                "ag",
                "closed",
                42,
                ("--ag-gamma", str(gamma)),
            ),
            Condition(
                f"ag_g{int(gamma)}_closed44",
                "ag",
                "closed",
                44,
                ("--ag-gamma", str(gamma)),
            ),
            Condition(
                f"aag_g{int(gamma)}_h5_k111",
                "ag",
                "anchored",
                40,
                (
                    "--ag-gamma",
                    str(gamma),
                    "--foresight-ag-gamma",
                    "0",
                    "--ag-reference",
                    "weak",
                    "--foresight-event-local-mode",
                    "target",
                ),
                schedule="0:5:1,5:5:1,15:5:1",
            ),
            Condition(
                f"aag_g{int(gamma)}_h5_k221",
                "ag",
                "anchored",
                40,
                (
                    "--ag-gamma",
                    str(gamma),
                    "--foresight-ag-gamma",
                    "0",
                    "--ag-reference",
                    "weak",
                    "--foresight-event-local-mode",
                    "target",
                ),
            ),
        )
    ),
    Condition("ag_g5_closed42", "ag", "closed", 42, ("--ag-gamma", "5")),
    Condition("ag_g5_closed44", "ag", "closed", 44, ("--ag-gamma", "5")),
    *(
        condition
        for gamma in (3.0, 4.0, 5.0)
        for condition in (
            Condition(
                f"iag_g{int(gamma)}_k2_early2_matched42",
                "ag",
                "implicit_ag",
                40,
                ("--ag-gamma", str(gamma), "--ag-reference", "weak"),
                schedule="0:1:2,5:1:2,15:1:1",
            ),
            Condition(
                f"iag_g{int(gamma)}_k3_early2_matched44",
                "ag",
                "implicit_ag",
                40,
                ("--ag-gamma", str(gamma), "--ag-reference", "weak"),
                schedule="0:1:3,5:1:3,15:1:1",
            ),
        )
    ),
    Condition("ag_g4_closed41", "ag", "closed", 41, ("--ag-gamma", "4")),
    Condition("ag_g4_closed47", "ag", "closed", 47, ("--ag-gamma", "4")),
    Condition("ag_g4_closed51", "ag", "closed", 51, ("--ag-gamma", "4")),
    Condition("ag_g4_closed61", "ag", "closed", 61, ("--ag-gamma", "4")),
    Condition("ag_g4_closed63", "ag", "closed", 63, ("--ag-gamma", "4")),
    Condition("ag_g3_closed81", "ag", "closed", 81, ("--ag-gamma", "3")),
    Condition("ag_g4_closed81", "ag", "closed", 81, ("--ag-gamma", "4")),
    Condition("ag_g4_closed83", "ag", "closed", 83, ("--ag-gamma", "4")),
    *(
        Condition(
            f"tag_g4_early2_rho{str(rho).replace('.', 'p')}_matched40",
            "ag",
            "scheduled_ag",
            40,
            (
                "--ag-gamma",
                "4",
                "--ag-reference",
                "weak",
                "--anchored-strength-multiplier",
                str(rho),
            ),
            schedule="0:1:1,5:1:1",
        )
        for rho in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
    ),
    *(
        Condition(
            f"lag_g4_h0_early2_rho{str(rho).replace('.', 'p')}_matched41",
            "ag",
            "local_calibration_ag",
            40,
            (
                "--ag-gamma",
                "4",
                "--ag-reference",
                "weak",
                "--anchored-strength-multiplier",
                str(rho),
            ),
            schedule="0:1:1,5:1:1",
        )
        for rho in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
    ),
    *(
        Condition(
            f"rag_g4_h5_early2_rho{str(rho).replace('.', 'p')}_matched61",
            "ag",
            "future_raw_ag",
            40,
            (
                "--ag-gamma",
                "4",
                "--ag-reference",
                "weak",
                "--anchored-strength-multiplier",
                str(rho),
                "--conjugate-flow-integrator",
                "rk4",
            ),
            schedule="0:5:1,5:5:1",
        )
        for rho in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
    ),
    *(
        condition
        for rho in (1.0, 2.0, 4.0)
        for condition in (
            Condition(
                (
                    f"fmc_g4_h5_early2_rho{str(rho).replace('.', 'p')}_"
                    "matched63"
                ),
                "ag",
                "future_dir_current_norm_ag",
                40,
                (
                    "--ag-gamma",
                    "4",
                    "--ag-reference",
                    "weak",
                    "--anchored-strength-multiplier",
                    str(rho),
                    "--conjugate-flow-integrator",
                    "rk4",
                ),
                schedule="0:5:1,5:5:1",
            ),
            Condition(
                (
                    f"cmf_g4_h5_early2_rho{str(rho).replace('.', 'p')}_"
                    "matched63"
                ),
                "ag",
                "current_dir_future_norm_ag",
                40,
                (
                    "--ag-gamma",
                    "4",
                    "--ag-reference",
                    "weak",
                    "--anchored-strength-multiplier",
                    str(rho),
                    "--conjugate-flow-integrator",
                    "rk4",
                ),
                schedule="0:5:1,5:5:1",
            ),
        )
    ),
    *(
        Condition(
            f"fmc_g4_h{horizon}_early2_rho4p0_matched{matched_steps}",
            "ag",
            "future_dir_current_norm_ag",
            40,
            (
                "--ag-gamma",
                "4",
                "--ag-reference",
                "weak",
                "--anchored-strength-multiplier",
                "4",
                "--conjugate-flow-integrator",
                "rk4",
            ),
            schedule=f"0:{horizon}:1,5:{horizon}:1",
        )
        for horizon, matched_steps in ((1, 47), (2, 51), (10, 83))
    ),
    *(
        Condition(
            (
                f"rag_g4_h5_k2_early2_rho{str(rho).replace('.', 'p')}_"
                "matched83"
            ),
            "ag",
            "future_raw_ag",
            40,
            (
                "--ag-gamma",
                "4",
                "--ag-reference",
                "weak",
                "--anchored-strength-multiplier",
                str(rho),
                "--conjugate-flow-integrator",
                "rk4",
            ),
            schedule="0:5:2,5:5:2",
        )
        for rho in (1.0, 2.0, 4.0)
    ),
    *(
        Condition(
            (
                f"cag_g{int(gamma)}_h5_k1_early2_"
                f"rho{str(rho).replace('.', 'p')}_matched81"
            ),
            "ag",
            "conjugate_ag",
            40,
            (
                "--ag-gamma",
                str(gamma),
                "--ag-reference",
                "weak",
                "--anchored-strength-multiplier",
                str(rho),
                "--conjugate-flow-integrator",
                "rk4",
            ),
            schedule="0:5:1,5:5:1",
        )
        for gamma in (3.0, 4.0)
        for rho in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
    ),
)


def selected_conditions(args: argparse.Namespace) -> tuple[Condition, ...]:
    pattern = re.compile(args.condition_regex)
    conditions = tuple(
        condition for condition in CONDITIONS if pattern.search(condition.name)
    )
    if not conditions:
        raise ValueError(f"condition regex matched nothing: {args.condition_regex!r}")
    names = [condition.name for condition in conditions]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"selected conditions contain duplicate names: {duplicates}")
    return conditions


def parse_gpu_list(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or len(result) != len(set(result)) or any(index < 0 for index in result):
        raise argparse.ArgumentTypeError("GPU list must contain unique non-negative indices")
    return result


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def valid_samples(output_dir: Path, *, num_samples: int) -> bool:
    manifest_path = output_dir / "sampling_manifest.json"
    sample_path = output_dir / f"samples_n{num_samples}.npz"
    if not manifest_path.is_file() or not sample_path.is_file():
        return False
    manifest = load_json(manifest_path)
    return (
        manifest.get("format")
        == "eqvae_imagenet100_sit_foresight_fixed_point_samples_v1"
        and int(manifest.get("requested_samples", -1)) == num_samples
    )


def _run(command: list[str], *, gpu: int, log_path: Path) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # ADM FID's TensorFlow + SciPy stack otherwise creates hundreds of BLAS
    # threads per process and four concurrent evaluations oversubscribe this
    # host by more than an order of magnitude.
    env["OMP_NUM_THREADS"] = "8"
    env["OPENBLAS_NUM_THREADS"] = "8"
    env["MKL_NUM_THREADS"] = "8"
    env["NUMEXPR_NUM_THREADS"] = "8"
    env["TF_NUM_INTRAOP_THREADS"] = "8"
    env["TF_NUM_INTEROP_THREADS"] = "2"
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
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"command failed on GPU {gpu}: {' '.join(command)}\n{tail}")


def run_parallel(
    jobs: list[tuple[str, list[str], Path]], *, gpu_indices: list[int]
) -> None:
    if not jobs:
        return
    with ThreadPoolExecutor(max_workers=len(gpu_indices)) as executor:
        active = {}
        pending = iter(jobs)
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
                print(json.dumps({"event": "job_complete", "gpu": gpu, "name": name}), flush=True)
                try:
                    next_name, command, log_path = next(pending)
                except StopIteration:
                    continue
                next_future = executor.submit(
                    _run, command, gpu=gpu, log_path=log_path
                )
                active[next_future] = (gpu, next_name)
                break


def sample_jobs(args: argparse.Namespace) -> list[tuple[str, list[str], Path]]:
    jobs = []
    for condition in selected_conditions(args):
        output_dir = args.output_root / condition.name
        if valid_samples(output_dir, num_samples=args.num_samples):
            print(json.dumps({"event": "reuse_samples", "name": condition.name}), flush=True)
            continue
        schedule = (
            condition.schedule
            if condition.schedule is not None
            else PAPER_NFE50_SCHEDULE
            if condition.method in {
                "foresight",
                "anchored",
                "implicit_ag",
                "scheduled_ag",
                "local_calibration_ag",
                "future_dir_current_norm_ag",
                "current_dir_future_norm_ag",
                "future_raw_ag",
                "conjugate_ag",
            }
            else ""
        )
        command = [
            sys.executable,
            str(REPO_ROOT / "experiments/sample_imagenet100_sit_foresight_fixed_point.py"),
            "--family",
            condition.family,
            "--method",
            condition.method,
            "--num-steps",
            str(condition.num_steps),
            "--foresight-schedule",
            schedule,
            "--num-samples",
            str(args.num_samples),
            "--batch-size",
            str(args.batch_size),
            "--vae-decode-batch-size",
            str(args.vae_decode_batch_size),
            "--diagnostic-samples",
            str(args.diagnostic_samples),
            "--global-seed",
            str(args.global_seed),
            "--cuda-allocator-limit-gib",
            str(args.cuda_allocator_limit_gib),
            "--output-dir",
            str(output_dir),
            "--device",
            "cuda:0",
            *condition.extra,
        ]
        jobs.append((condition.name, command, output_dir / "sampling.log"))
    return jobs


def evaluation_jobs(args: argparse.Namespace) -> list[tuple[str, list[str], Path]]:
    jobs = []
    for condition in selected_conditions(args):
        output_dir = args.output_root / condition.name
        result_path = output_dir / "fid1k_adm_results.json"
        sample_path = output_dir / f"samples_n{args.num_samples}.npz"
        if result_path.is_file():
            result = load_json(result_path)
            if Path(result.get("samples", "")).resolve() == sample_path.resolve():
                print(json.dumps({"event": "reuse_fid", "name": condition.name}), flush=True)
                continue
        if not sample_path.is_file():
            raise FileNotFoundError(f"missing samples for {condition.name}: {sample_path}")
        command = [
            str(args.adm_python),
            str(REPO_ROOT / "experiments/compute_adm_fid.py"),
            "--reference",
            str(args.reference_stats),
            "--samples",
            str(sample_path),
            "--batch-size",
            str(args.fid_batch_size),
            "--gpu-memory-fraction",
            str(args.fid_gpu_memory_fraction),
            "--output",
            str(result_path),
        ]
        jobs.append((condition.name, command, output_dir / "evaluation.log"))
    return jobs


def write_summary(args: argparse.Namespace) -> None:
    rows: list[dict[str, object]] = []
    noise_hashes: set[str] = set()
    label_hashes: set[str] = set()
    for condition in selected_conditions(args):
        output_dir = args.output_root / condition.name
        manifest = load_json(output_dir / "sampling_manifest.json")
        metrics = load_json(output_dir / "fid1k_adm_results.json")
        noise_hashes.add(str(manifest["noise_sha256"]))
        label_hashes.add(str(manifest["label_sha256"]))
        forwards = manifest["model_forward_totals"]
        rows.append(
            {
                "condition": condition.name,
                "family": condition.family,
                "method": condition.method,
                "num_steps": condition.num_steps,
                "fid": float(metrics["fid"]),
                "sfid": float(metrics["sfid"]),
                "inception_score": float(metrics["inception_score"]),
                "num_samples": args.num_samples,
                "global_seed": args.global_seed,
                "model_forwards": sum(int(value) for value in forwards.values()),
                "noise_sha256": manifest["noise_sha256"],
                "label_sha256": manifest["label_sha256"],
                "output_dir": str(output_dir),
            }
        )
    if len(noise_hashes) != 1 or len(label_hashes) != 1:
        raise ValueError("conditions did not use identical noise and labels")
    args.output_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_root / f"{args.summary_stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "format": "eqvae_imagenet100_sit_foresight_fixed_point_study_v1",
        "scope": "paired FID-1K mechanism screen; not a formal FID-50K result",
        "paper_nfe50_schedule": PAPER_NFE50_SCHEDULE,
        "reference_stats": str(args.reference_stats),
        "paired_noise_sha256": next(iter(noise_hashes)),
        "paired_label_sha256": next(iter(label_hashes)),
        "rows": rows,
    }
    (args.output_root / f"{args.summary_stem}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


def main(args: argparse.Namespace) -> None:
    args.output_root = args.output_root.expanduser().resolve()
    args.reference_stats = args.reference_stats.expanduser().resolve()
    if not args.reference_stats.is_file():
        raise FileNotFoundError(args.reference_stats)
    if not args.adm_python.is_file():
        raise FileNotFoundError(args.adm_python)
    if not args.skip_sampling:
        run_parallel(sample_jobs(args), gpu_indices=args.gpu_indices)
    if not args.skip_evaluation:
        run_parallel(evaluation_jobs(args), gpu_indices=args.gpu_indices)
        write_summary(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--reference-stats", type=Path, default=DEFAULT_REFERENCE_STATS)
    parser.add_argument("--adm-python", type=Path, default=DEFAULT_ADM_PYTHON)
    parser.add_argument("--gpu-indices", type=parse_gpu_list, default=[0, 1, 2, 3])
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--vae-decode-batch-size", type=int, default=4)
    parser.add_argument("--diagnostic-samples", type=int, default=16)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--cuda-allocator-limit-gib", type=float, default=10.0)
    parser.add_argument("--fid-batch-size", type=int, default=16)
    parser.add_argument("--fid-gpu-memory-fraction", type=float, default=0.30)
    parser.add_argument("--skip-sampling", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--condition-regex", default=".*")
    parser.add_argument("--summary-stem", default="fid1k_summary")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
