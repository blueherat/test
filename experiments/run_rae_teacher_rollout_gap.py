"""Run the teacher-forced versus rollout gap audit on four GPUs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = Path.home() / "data/eqvae/experiments/rae_spectral_tiny"


def completed_branches(results: Path) -> list[Path]:
    return sorted(
        branch
        for branch in results.glob("seed*_*_from_s5000")
        if (branch / "manifest.json").exists()
        and (branch / "generation" / "ema_step-0010000.pt").exists()
    )


def aggregate(results: Path, branches: list[Path]) -> dict[str, Path]:
    specifications = {
        "teacher": ("teacher_metrics.csv", "teacher_rollout_gap_teacher.csv"),
        "teacher_bands": ("teacher_bands.csv", "teacher_rollout_gap_teacher_bands.csv"),
        "rollout": ("rollout_metrics.csv", "teacher_rollout_gap_rollout.csv"),
        "bands": ("rollout_bands.csv", "teacher_rollout_gap_bands.csv"),
        "steps": ("step_metrics.csv", "teacher_rollout_gap_steps.csv"),
    }
    outputs: dict[str, Path] = {}
    for key, (source_name, output_name) in specifications.items():
        frames = []
        for branch in branches:
            source = branch / "gap_study" / source_name
            if not source.exists():
                raise FileNotFoundError(source)
            frames.append(pd.read_csv(source))
        table = pd.concat(frames, ignore_index=True)
        sort_columns = [column for column in ("seed", "treatment", "time_index", "metric", "band") if column in table]
        table = table.sort_values(sort_columns).reset_index(drop=True)
        output = results / output_name
        table.to_csv(output, index=False)
        outputs[key] = output

    metadata = {
        "status": "complete",
        "branches": [branch.name for branch in branches],
        "branch_count": len(branches),
        "outputs": {key: str(path) for key, path in outputs.items()},
        "interpretation_boundary": {
            "teacher": "paired clean-estimate errors on known linear interpolation states",
            "rollout": "distributional gaps; no arbitrary validation-image pairing",
            "endpoint": "trajectory consistency proxy, not FID or sample quality",
        },
    }
    metadata_path = results / "teacher_rollout_gap_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    outputs["metadata"] = metadata_path
    return outputs


def launch(
    branches: list[Path],
    *,
    devices: list[int],
    count: int,
    batch_size: int,
    perceptual_count: int,
    perceptual_batch_size: int,
    evaluation_seed: int,
    overwrite: bool,
) -> None:
    pending = list(branches)
    running: dict[int, tuple[subprocess.Popen, Path, object]] = {}
    while pending or running:
        for device in devices:
            if device in running or not pending:
                continue
            branch = pending.pop(0)
            output = branch / "gap_study"
            expected = output / "metadata.json"
            if expected.exists() and not overwrite:
                print(f"skip complete {branch.name}", flush=True)
                continue
            output.mkdir(parents=True, exist_ok=True)
            log_handle = (output / "run.log").open("w", encoding="utf-8")
            command = [
                sys.executable,
                str(ROOT / "experiments/rae_teacher_rollout_gap.py"),
                "--branch",
                str(branch),
                "--device",
                "cuda:0",
                "--count",
                str(int(count)),
                "--batch-size",
                str(int(batch_size)),
                "--perceptual-count",
                str(int(perceptual_count)),
                "--perceptual-batch-size",
                str(int(perceptual_batch_size)),
                "--evaluation-seed",
                str(int(evaluation_seed)),
            ]
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(int(device))
            environment["PYTHONUNBUFFERED"] = "1"
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            running[device] = (process, branch, log_handle)
            print(f"gpu {device}: start {branch.name} (pid {process.pid})", flush=True)

        if not running:
            continue
        time.sleep(2.0)
        for device, (process, branch, log_handle) in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            log_handle.close()
            del running[device]
            if return_code != 0:
                log_path = branch / "gap_study" / "run.log"
                tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-60:])
                for other, (_, _, other_log) in list(running.items()):
                    running[other][0].terminate()
                    other_log.close()
                raise RuntimeError(f"{branch.name} failed on gpu {device}:\n{tail}")
            print(f"gpu {device}: complete {branch.name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("run", "aggregate", "all"), default="all")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--perceptual-count", type=int, default=12)
    parser.add_argument("--perceptual-batch-size", type=int, default=2)
    parser.add_argument("--evaluation-seed", type=int, default=104729)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    results = args.results.expanduser().resolve()
    branches = completed_branches(results)
    if len(branches) != 6:
        raise RuntimeError(f"expected six completed paired branches, found {len(branches)}")
    if args.mode in {"run", "all"}:
        devices = [int(value.strip()) for value in args.devices.split(",") if value.strip()]
        if not devices:
            raise ValueError("at least one GPU device is required")
        launch(
            branches,
            devices=devices,
            count=args.count,
            batch_size=args.batch_size,
            perceptual_count=args.perceptual_count,
            perceptual_batch_size=args.perceptual_batch_size,
            evaluation_seed=args.evaluation_seed,
            overwrite=args.overwrite,
        )
    if args.mode in {"aggregate", "all"}:
        outputs = aggregate(results, branches)
        print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
