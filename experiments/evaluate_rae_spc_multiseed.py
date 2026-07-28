"""Sample and score the preregistered paired SPC multi-seed study."""

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
DEFAULT_RESULTS = Path.home() / "data/eqvae/experiments/rae_spc_multiseed_v1"
DEFAULT_REFERENCE = Path("/data/shared/adm_refs/VIRTUAL_imagenet256_labeled.npz")
DEFAULT_SEEDS = (1201, 2309, 3413, 4517, 5623)
SAMPLING_SEED = 20_260_718


def branch_name(seed: int, condition: str, endpoint: int, switch_step: int) -> str:
    if condition == "static":
        return f"seed{seed}_static_s0_to{endpoint}"
    if condition == "spc":
        return (
            f"seed{seed}_spc_floor020_p2_rank16_"
            f"switch{switch_step}_s0_to{endpoint}"
        )
    raise ValueError(f"unknown condition: {condition}")


def planned_branches(
    seeds: tuple[int, ...], endpoint: int, switch_step: int
) -> list[tuple[int, str, str]]:
    return [
        (seed, condition, branch_name(seed, condition, endpoint, switch_step))
        for seed in seeds
        for condition in ("static", "spc")
    ]


def sample_command(
    results: Path,
    name: str,
    *,
    endpoint: int,
    sample_count: int,
    steps: int,
    device: int,
    weight_source: str,
    per_process_batch: int,
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "experiments/evaluate_rae_layerwise_path_generation.py"),
        "--mode",
        "sample",
        "--results",
        str(results),
        "--branch-name",
        name,
        "--endpoint",
        str(endpoint),
        "--sample-count",
        str(sample_count),
        "--steps",
        str(steps),
        "--devices",
        str(device),
        "--processes",
        "1",
        "--per-process-batch",
        str(per_process_batch),
        "--weight-source",
        weight_source,
    ]


def metric_command(
    results: Path,
    name: str,
    *,
    endpoint: int,
    sample_count: int,
    steps: int,
    weight_source: str,
    batch_size: int,
    reference: Path,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-metrics",
        "--results",
        str(results),
        "--branch-name",
        name,
        "--endpoint",
        str(endpoint),
        "--sample-count",
        str(sample_count),
        "--steps",
        str(steps),
        "--weight-source",
        weight_source,
        "--metric-batch-size",
        str(batch_size),
        "--reference",
        str(reference),
    ]


def run_jobs(
    commands: list[tuple[str, list[str]]],
    *,
    devices: tuple[int, ...],
    log_root: Path,
) -> None:
    log_root.mkdir(parents=True, exist_ok=True)
    pending = list(commands)
    active: dict[int, tuple[str, subprocess.Popen, object]] = {}
    failures: list[tuple[str, int]] = []
    while pending or active:
        for device in devices:
            if not pending or device in active:
                continue
            name, command = pending.pop(0)
            command = list(command)
            if "--devices" in command:
                command[command.index("--devices") + 1] = str(device)
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(device)
            environment["PYTHONUNBUFFERED"] = "1"
            handle = (log_root / f"{name}.log").open("a", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            active[device] = (name, process, handle)
            print(f"started {name} cuda={device}", flush=True)
        if not active:
            break
        time.sleep(2)
        for device, (name, process, handle) in list(active.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            handle.close()
            del active[device]
            print(f"finished {name} cuda={device} exit={return_code}", flush=True)
            if return_code:
                failures.append((name, return_code))
    if failures:
        raise RuntimeError(f"evaluation failures: {failures}")


def metrics_path(
    branch: Path, weight_source: str, sample_count: int, steps: int
) -> Path:
    return (
        branch
        / "generation"
        / f"generation_metrics_{weight_source}_n{sample_count}_{steps}steps.json"
    )


def compute_worker_metrics(args: argparse.Namespace) -> None:
    from experiments.evaluate_rae_layerwise_path_generation import (
        NumpyRGBDataset,
        fidelity_metrics,
        sample_folder_name,
    )

    branch = args.results.expanduser().resolve() / args.branch_name
    manifest = json.loads((branch / "manifest.json").read_text(encoding="utf-8"))
    sample_npz = (
        branch
        / "generation"
        / sample_folder_name(
            args.sample_count, args.endpoint, args.steps, args.weight_source
        )
    ).with_suffix(".npz")
    samples = NumpyRGBDataset(sample_npz)
    if len(samples) != args.sample_count:
        raise ValueError(f"expected {args.sample_count} samples in {sample_npz}")
    reference = NumpyRGBDataset(args.reference)
    condition = "spc" if manifest.get("path_switch_step") is not None else "static"
    row = {
        "branch": branch.name,
        "seed": int(manifest["global_seed"]),
        "condition": condition,
        "endpoint": int(args.endpoint),
        "sample_count": int(args.sample_count),
        "sampling_seed": SAMPLING_SEED,
        "sampling_steps": int(args.steps),
        "weight_source": args.weight_source,
        **fidelity_metrics(samples, reference, batch_size=args.metric_batch_size),
    }
    output = metrics_path(branch, args.weight_source, args.sample_count, args.steps)
    output.write_text(
        json.dumps(row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(row, ensure_ascii=False))


def collect_metrics(
    results: Path,
    branches: list[tuple[int, str, str]],
    *,
    weight_source: str,
    sample_count: int,
    steps: int,
) -> pd.DataFrame:
    rows = []
    for _, _, name in branches:
        path = metrics_path(results / name, weight_source, sample_count, steps)
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    table = pd.DataFrame(rows).sort_values(["seed", "condition"])
    output = (
        results
        / "evaluation"
        / f"spc_metrics_{weight_source}_n{sample_count}_{steps}steps.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(table.to_string(index=False))
    print(f"saved {output}")
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--endpoint", type=int, default=5000)
    parser.add_argument("--switch-step", type=int, default=2000)
    parser.add_argument("--sample-count", type=int, default=5000)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--weight-source", choices=("model", "ema"), default="model")
    parser.add_argument("--per-process-batch", type=int, default=8)
    parser.add_argument("--metric-batch-size", type=int, default=64)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--mode", choices=("sample", "metrics", "all"), default="all")
    parser.add_argument("--worker-metrics", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--branch-name", default="", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker_metrics:
        compute_worker_metrics(args)
        return
    if args.sample_count % 1000:
        raise ValueError("class-balanced ImageNet sampling requires a multiple of 1000")
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    devices = tuple(int(value) for value in args.devices.split(",") if value.strip())
    results = args.results.expanduser().resolve()
    branches = planned_branches(seeds, args.endpoint, args.switch_step)
    for _, _, name in branches:
        checkpoint = results / name / "checkpoints" / f"step-{args.endpoint:07d}.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)

    if args.mode in {"sample", "all"}:
        commands = [
            (
                name,
                sample_command(
                    results,
                    name,
                    endpoint=args.endpoint,
                    sample_count=args.sample_count,
                    steps=args.steps,
                    device=device,
                    weight_source=args.weight_source,
                    per_process_batch=args.per_process_batch,
                ),
            )
            for device, (_, _, name) in zip(
                (devices[index % len(devices)] for index in range(len(branches))),
                branches,
            )
        ]
        run_jobs(
            commands,
            devices=devices,
            log_root=results / "evaluation" / f"sampling_{args.weight_source}_logs",
        )
    if args.mode in {"metrics", "all"}:
        commands = [
            (
                name,
                metric_command(
                    results,
                    name,
                    endpoint=args.endpoint,
                    sample_count=args.sample_count,
                    steps=args.steps,
                    weight_source=args.weight_source,
                    batch_size=args.metric_batch_size,
                    reference=args.reference,
                ),
            )
            for _, _, name in branches
        ]
        run_jobs(
            commands,
            devices=devices,
            log_root=results / "evaluation" / f"metrics_{args.weight_source}_logs",
        )
        collect_metrics(
            results,
            branches,
            weight_source=args.weight_source,
            sample_count=args.sample_count,
            steps=args.steps,
        )


if __name__ == "__main__":
    main()
