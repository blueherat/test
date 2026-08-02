"""Resumable runner for the replicated RAEv2 IG endpoint mechanism audit."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/home/zhoushunyu/data/eqvae/experiments")
SEEDS = (20260801, 20260802)


@dataclass(frozen=True)
class Stage:
    name: str
    command: tuple[str, ...]
    output_dir: Path
    required_files: tuple[str, ...]
    wait_for_external: bool = False

    @property
    def exit_code_path(self) -> Path:
        return self.output_dir / "exit_code"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipeline-dir",
        type=Path,
        default=DATA_ROOT / "raev2_ig_endpoint_pipeline_v1",
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--external-grace-seconds", type=float, default=180.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _torchrun() -> str:
    candidate = Path(sys.executable).resolve().parent / "torchrun"
    return str(candidate if candidate.exists() else "torchrun")


def endpoint_dir(seed: int) -> Path:
    return DATA_ROOT / "raev2_distribution_auc" / f"endpoint_n5000_seed{seed}_v2"


def predicted_clean_dir(seed: int) -> Path:
    return DATA_ROOT / "raev2_predicted_clean_audit" / f"n5000_seed{seed}_v1"


def decoded_reference_dir(seed: int) -> Path:
    return DATA_ROOT / "raev2_decoded_distribution_audit" / f"n5000_seed{seed}_v1"


def build_stages() -> list[Stage]:
    torchrun = _torchrun()
    endpoint_common = (
        torchrun,
        "--standalone",
        "--nproc_per_node=4",
        "experiments/run_raev2_distribution_auc.py",
        "--samples",
        "5000",
        "--per-rank-batch",
        "2",
        "--bootstrap-repeats",
        "2000",
        "--log-every-batches",
        "25",
        "--time",
        "0",
        "--time",
        "1",
    )
    endpoint_required = (
        "manifest.json",
        "auc_results.csv",
        "auc_delta_ig_minus_full.csv",
        "latent_moment_distances.csv",
        "heldout_probe_scores.npz",
    )
    stages = [
        Stage(
            name="endpoint_seed_20260801_external",
            command=(),
            output_dir=endpoint_dir(SEEDS[0]),
            required_files=endpoint_required,
            wait_for_external=True,
        ),
        Stage(
            name="endpoint_seed_20260802",
            command=endpoint_common
            + (
                "--output-dir",
                str(endpoint_dir(SEEDS[1])),
                "--seed",
                str(SEEDS[1]),
            ),
            output_dir=endpoint_dir(SEEDS[1]),
            required_files=endpoint_required,
        ),
    ]

    endpoint_summary = DATA_ROOT / "raev2_distribution_auc" / "endpoint_cross_seed_n5000_v2"
    stages.append(
        Stage(
            name="summarize_endpoint_seeds",
            command=(
                sys.executable,
                "experiments/summarize_raev2_distribution_auc.py",
                "--run",
                f"seed_{SEEDS[0]}={endpoint_dir(SEEDS[0])}",
                "--run",
                f"seed_{SEEDS[1]}={endpoint_dir(SEEDS[1])}",
                "--output-dir",
                str(endpoint_summary),
            ),
            output_dir=endpoint_summary,
            required_files=("summary.json", "cross_seed_summary.csv", "cross_seed_auc_delta.png"),
        )
    )

    predicted_required = (
        "manifest.json",
        "predicted_clean_summary.csv",
        "predicted_clean_effects.csv",
        "predicted_clean_heldout_scores.npz",
        "predicted_clean_features_rank00.npz",
        "predicted_clean_features_rank01.npz",
        "predicted_clean_features_rank02.npz",
        "predicted_clean_features_rank03.npz",
    )
    stages.append(
        Stage(
            name="predicted_clean_seed_20260802",
            command=(
                torchrun,
                "--standalone",
                "--nproc_per_node=4",
                "experiments/run_raev2_predicted_clean_audit.py",
                "--decoded-reference-run",
                str(decoded_reference_dir(SEEDS[1])),
                "--output-dir",
                str(predicted_clean_dir(SEEDS[1])),
                "--samples",
                "5000",
                "--per-rank-batch",
                "2",
                "--bootstrap-repeats",
                "2000",
                "--log-every-batches",
                "25",
                "--precision",
                "bf16",
                "--seed",
                str(SEEDS[1]),
            ),
            output_dir=predicted_clean_dir(SEEDS[1]),
            required_files=predicted_required,
        )
    )

    predicted_summary = (
        DATA_ROOT / "raev2_predicted_clean_audit" / "cross_seed_n5000_v1"
    )
    stages.append(
        Stage(
            name="summarize_predicted_clean_seeds",
            command=(
                sys.executable,
                "experiments/summarize_raev2_predicted_clean_audit.py",
                "--run-dir",
                str(predicted_clean_dir(SEEDS[0])),
                "--run-dir",
                str(predicted_clean_dir(SEEDS[1])),
                "--output-dir",
                str(predicted_summary),
            ),
            output_dir=predicted_summary,
            required_files=(
                "manifest.json",
                "cross_seed_predicted_clean_summary.csv",
                "cross_seed_predicted_clean_effects.csv",
                "cross_seed_predicted_clean_curves.png",
            ),
        )
    )

    for seed in SEEDS:
        run_dir = predicted_clean_dir(seed)
        stages.extend(
            (
                Stage(
                    name=f"kid_seed_{seed}",
                    command=(
                        sys.executable,
                        "experiments/compute_raev2_predicted_clean_kid.py",
                        "--run-dir",
                        str(run_dir),
                        "--device",
                        "cuda:0",
                        "--seed",
                        str(seed + 7341),
                    ),
                    output_dir=run_dir / "kid",
                    required_files=(
                        "manifest.json",
                        "predicted_clean_kid.csv",
                        "predicted_clean_kid_effects.csv",
                    ),
                ),
                Stage(
                    name=f"precision_recall_seed_{seed}",
                    command=(
                        sys.executable,
                        "experiments/compute_raev2_predicted_clean_precision_recall.py",
                        "--run-dir",
                        str(run_dir),
                        "--device",
                        "cuda:0",
                        "--seed",
                        str(seed + 9187),
                    ),
                    output_dir=run_dir / "precision_recall",
                    required_files=(
                        "manifest.json",
                        "predicted_clean_precision_recall.csv",
                        "predicted_clean_precision_recall_effects.csv",
                    ),
                ),
            )
        )
    return stages


def stage_succeeded(stage: Stage) -> bool:
    if not stage.exit_code_path.exists():
        return False
    try:
        exit_code = int(stage.exit_code_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return False
    return exit_code == 0 and all(
        (stage.output_dir / relative).is_file() for relative in stage.required_files
    )


def matching_process_exists(output_dir: Path) -> bool:
    needle = str(output_dir.resolve()).encode()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if needle in command:
            return True
    return False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(pipeline_dir: Path, payload: dict[str, object]) -> None:
    path = pipeline_dir / "pipeline_status.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def wait_for_external_stage(
    stage: Stage,
    *,
    poll_seconds: float,
    grace_seconds: float,
    pipeline_dir: Path,
) -> None:
    missing_since: float | None = None
    while not stage_succeeded(stage):
        if stage.exit_code_path.exists():
            value = stage.exit_code_path.read_text(encoding="utf-8").strip()
            if value != "0":
                raise RuntimeError(f"external stage {stage.name} exited with {value}")
        if matching_process_exists(stage.output_dir):
            missing_since = None
        elif missing_since is None:
            missing_since = time.monotonic()
        elif time.monotonic() - missing_since > grace_seconds:
            raise RuntimeError(
                f"external stage {stage.name} has no process and no valid outputs"
            )
        write_status(
            pipeline_dir,
            {
                "updated_at": _now(),
                "state": "waiting_for_external_stage",
                "stage": stage.name,
                "output_dir": str(stage.output_dir),
            },
        )
        time.sleep(poll_seconds)


def run_stage(stage: Stage, pipeline_dir: Path) -> None:
    stage.output_dir.mkdir(parents=True, exist_ok=True)
    if stage.exit_code_path.exists():
        stage.exit_code_path.unlink()
    log_path = stage.output_dir / "pipeline_run.log"
    write_status(
        pipeline_dir,
        {
            "updated_at": _now(),
            "state": "running",
            "stage": stage.name,
            "command": list(stage.command),
            "output_dir": str(stage.output_dir),
        },
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    environment["MPLCONFIGDIR"] = str(pipeline_dir / "matplotlib")
    (pipeline_dir / "matplotlib").mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{_now()}] START {stage.name}\n")
        log.write("COMMAND " + " ".join(stage.command) + "\n")
        log.flush()
        completed = subprocess.run(
            stage.command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log.write(f"[{_now()}] EXIT {completed.returncode}\n")
    stage.exit_code_path.write_text(str(completed.returncode) + "\n", encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"stage {stage.name} failed with {completed.returncode}")
    if not stage_succeeded(stage):
        missing = [
            relative
            for relative in stage.required_files
            if not (stage.output_dir / relative).is_file()
        ]
        raise RuntimeError(f"stage {stage.name} is missing outputs: {missing}")


def run_pipeline(
    stages: Sequence[Stage],
    *,
    pipeline_dir: Path,
    poll_seconds: float,
    grace_seconds: float,
    dry_run: bool,
) -> None:
    if dry_run:
        for stage in stages:
            state = "skip" if stage_succeeded(stage) else "dry-run"
            print(f"[{state}] {stage.name}", flush=True)
            print(
                " ".join(stage.command) if stage.command else "WAIT EXTERNAL",
                flush=True,
            )
        return
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    for stage in stages:
        if stage_succeeded(stage):
            print(f"[skip] {stage.name}", flush=True)
            continue
        print(f"[start] {stage.name}", flush=True)
        if stage.wait_for_external:
            wait_for_external_stage(
                stage,
                poll_seconds=poll_seconds,
                grace_seconds=grace_seconds,
                pipeline_dir=pipeline_dir,
            )
        else:
            run_stage(stage, pipeline_dir)
        print(f"[done] {stage.name}", flush=True)
    write_status(
        pipeline_dir,
        {
            "updated_at": _now(),
            "state": "complete",
            "completed_stages": [stage.name for stage in stages],
        },
    )


def main() -> None:
    args = parse_args()
    if args.poll_seconds <= 0 or args.external_grace_seconds <= 0:
        raise ValueError("poll and grace intervals must be positive")
    run_pipeline(
        build_stages(),
        pipeline_dir=args.pipeline_dir.expanduser().resolve(),
        poll_seconds=args.poll_seconds,
        grace_seconds=args.external_grace_seconds,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
