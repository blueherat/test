"""Verify and evaluate the preregistered floor/static crossover branches."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_RESULTS = Path.home() / "data/eqvae/experiments/rae_path_crossover_train_v2"
BASELINE_ROOT = Path.home() / "data/eqvae/experiments/rae_layerwise_path_train"
SCHEDULE_ROOT = Path.home() / "data/eqvae/experiments/rae_path_schedule_train"
REFERENCE = Path("/data/shared/adm_refs/VIRTUAL_imagenet256_labeled.npz")
SAMPLING_SEED = 20_260_718
BRANCH_NAMES = (
    "floor_to_floor",
    "floor_to_static",
    "static_to_static",
    "static_to_floor",
)
OLD_REPLAYS = {
    "floor_to_floor": SCHEDULE_ROOT / "seed3407_floor020_p2_rank16_s0_to_2000",
    "static_to_static": BASELINE_ROOT / "seed3407_static_rank16_s0_to_10000",
}


def branch_path(results: Path, condition: str, endpoint: int = 5000) -> Path:
    return results / f"seed3407_{condition}_rank16_s2000_to_{int(endpoint)}"


def exactly_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(left, right)
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        return left.keys() == right.keys() and all(
            exactly_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
            return False
        return len(left) == len(right) and all(
            exactly_equal(a, b) for a, b in zip(left, right)
        )
    return bool(left == right)


def verify_replays(results: Path, endpoint: int) -> dict[str, object]:
    rows = {}
    for condition, old_branch in OLD_REPLAYS.items():
        new_checkpoint = branch_path(results, condition, endpoint) / "checkpoints" / f"step-{endpoint:07d}.pt"
        old_checkpoint = old_branch / "checkpoints" / f"step-{endpoint:07d}.pt"
        new = torch.load(new_checkpoint, map_location="cpu", weights_only=False)
        old = torch.load(old_checkpoint, map_location="cpu", weights_only=False)
        checks = {
            key: exactly_equal(new.get(key), old.get(key))
            for key in (
                "model",
                "ema",
                "optimizer",
                "scheduler",
                "step",
                "branch_start_step",
                "epoch",
                "rng_cpu",
                "rng_cuda",
            )
        }
        rows[condition] = {
            "new_checkpoint": str(new_checkpoint),
            "old_checkpoint": str(old_checkpoint),
            "checks": checks,
            "exact": bool(all(checks.values())),
        }
    return {
        "pass": bool(all(row["exact"] for row in rows.values())),
        "replays": rows,
    }


def _sampling_command(
    results: Path,
    condition: str,
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
        branch_path(results, condition, endpoint).name,
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


def sample_branches(
    results: Path,
    *,
    endpoint: int,
    sample_count: int,
    steps: int,
    devices: Sequence[int],
    weight_source: str,
    per_process_batch: int,
) -> None:
    if len(devices) != len(BRANCH_NAMES):
        raise ValueError("sampling requires exactly four devices")
    log_root = results / "crossover_evaluation" / "sampling_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    processes = []
    for condition, device in zip(BRANCH_NAMES, devices):
        command = _sampling_command(
            results,
            condition,
            endpoint=endpoint,
            sample_count=sample_count,
            steps=steps,
            device=device,
            weight_source=weight_source,
            per_process_batch=per_process_batch,
        )
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        handle = (log_root / f"{condition}.log").open("a", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((condition, process, handle))
    failures = []
    for condition, process, handle in processes:
        return_code = process.wait()
        handle.close()
        print(f"[{condition}] sampling exit={return_code}", flush=True)
        if return_code:
            failures.append((condition, return_code))
    if failures:
        raise RuntimeError(f"sampling failures: {failures}")


def _sample_npz(
    results: Path,
    condition: str,
    endpoint: int,
    sample_count: int,
    steps: int,
    weight_source: str,
) -> Path:
    from experiments.evaluate_rae_layerwise_path_generation import sample_folder_name

    return (
        branch_path(results, condition, endpoint)
        / "generation"
        / sample_folder_name(sample_count, endpoint, steps, weight_source)
    ).with_suffix(".npz")


def compute_generation_metrics(
    results: Path,
    *,
    endpoint: int,
    sample_count: int,
    steps: int,
    batch_size: int,
    weight_source: str,
) -> pd.DataFrame:
    from experiments.evaluate_rae_layerwise_path_generation import (
        NumpyRGBDataset,
        fidelity_metrics,
    )

    reference = NumpyRGBDataset(REFERENCE)
    rows = []
    for condition in BRANCH_NAMES:
        sample_path = _sample_npz(
            results, condition, endpoint, sample_count, steps, weight_source
        )
        samples = NumpyRGBDataset(sample_path)
        if len(samples) != sample_count:
            raise ValueError(f"expected {sample_count} samples in {sample_path}")
        rows.append(
            {
                "condition": condition,
                "endpoint": endpoint,
                "sample_count": sample_count,
                "sampling_steps": steps,
                "sampling_seed": SAMPLING_SEED,
                "weight_source": weight_source,
                **fidelity_metrics(samples, reference, batch_size=batch_size),
            }
        )
    return pd.DataFrame(rows)


def summarize_generation(table: pd.DataFrame) -> dict[str, object]:
    indexed = table.set_index("condition")
    metrics = (
        "frechet_inception_distance",
        "kernel_inception_distance_mean",
    )
    effects = {}
    movements = {}
    for metric in metrics:
        ff = float(indexed.loc["floor_to_floor", metric])
        fs = float(indexed.loc["floor_to_static", metric])
        ss = float(indexed.loc["static_to_static", metric])
        sf = float(indexed.loc["static_to_floor", metric])
        effects[metric] = {
            "late_floor_with_early_floor": ff - fs,
            "late_floor_with_early_static": sf - ss,
            "difference_in_differences": (sf - ss) - (ff - fs),
        }
        movements[metric] = {
            "floor_to_static_closer_to_late_control": abs(fs - ss) < abs(fs - ff),
            "static_to_floor_closer_to_late_control": abs(sf - ff) < abs(sf - ss),
        }
    directions = {
        "floor_to_static_improves_both": all(
            indexed.loc["floor_to_static", metric]
            < indexed.loc["floor_to_floor", metric]
            for metric in metrics
        ),
        "static_to_floor_worsens_both": all(
            indexed.loc["static_to_floor", metric]
            > indexed.loc["static_to_static", metric]
            for metric in metrics
        ),
    }
    return {
        "directions": {key: bool(value) for key, value in directions.items()},
        "late_path_effects": effects,
        "movement_toward_late_control": movements,
    }


def plot_generation(table: pd.DataFrame, output: Path, weight_source: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#E45756", "#72B7B2", "#4C78A8", "#F2CF5B"]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    count = int(table.sample_count.iloc[0])
    for axis, metric, title in (
        (axes[0], "frechet_inception_distance", f"Fixed-seed n={count} FID"),
        (axes[1], "kernel_inception_distance_mean", f"Fixed-seed n={count} KID"),
    ):
        axis.bar(table.condition, table[metric], color=colors)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(
        output / f"crossover_generation_{weight_source}_n{count}.png", dpi=180
    )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("verify", "sample", "metrics", "all"), default="all"
    )
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--endpoint", type=int, default=5000)
    parser.add_argument("--sample-count", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--metric-batch-size", type=int, default=64)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--weight-source", choices=("ema", "model"), default="ema")
    parser.add_argument("--per-process-batch", type=int, default=4)
    args = parser.parse_args()
    results = args.results.expanduser().resolve()
    output = results / "crossover_evaluation"
    output.mkdir(parents=True, exist_ok=True)
    if args.mode in {"verify", "all"}:
        replay = verify_replays(results, args.endpoint)
        (output / "replay_verification.json").write_text(
            json.dumps(replay, indent=2), encoding="utf-8"
        )
        print(json.dumps(replay, indent=2), flush=True)
        if not replay["pass"]:
            raise RuntimeError("replay controls are not exact; crossover is uninterpretable")
    if args.mode in {"sample", "all"}:
        devices = [int(value) for value in args.devices.split(",") if value.strip()]
        sample_branches(
            results,
            endpoint=args.endpoint,
            sample_count=args.sample_count,
            steps=args.steps,
            devices=devices,
            weight_source=args.weight_source,
            per_process_batch=args.per_process_batch,
        )
    if args.mode in {"metrics", "all"}:
        table = compute_generation_metrics(
            results,
            endpoint=args.endpoint,
            sample_count=args.sample_count,
            steps=args.steps,
            batch_size=args.metric_batch_size,
            weight_source=args.weight_source,
        )
        summary = summarize_generation(table)
        stem = f"{args.weight_source}_n{args.sample_count}"
        table.to_csv(output / f"generation_metrics_{stem}.csv", index=False)
        (output / f"generation_decision_{stem}.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        plot_generation(table, output, args.weight_source)
        print(table.to_string(index=False), flush=True)
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
