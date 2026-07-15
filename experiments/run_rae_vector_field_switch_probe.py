"""Run paired vector-field time-switch probes on available GPUs."""

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


def paired_branches(results: Path) -> list[tuple[int, Path, Path]]:
    pairs = []
    for baseline in sorted(results.glob("seed*_baseline_from_s5000")):
        seed = int(baseline.name.split("_")[0].removeprefix("seed"))
        partial = results / f"seed{seed}_partial_from_s5000"
        if (baseline / "manifest.json").exists() and (partial / "manifest.json").exists():
            pairs.append((seed, baseline, partial))
    return pairs


def aggregate(results: Path, pairs: list[tuple[int, Path, Path]]) -> tuple[Path, Path]:
    root = results / "vector_field_switch"
    metrics = pd.concat(
        [pd.read_csv(root / f"seed{seed}" / "metrics.csv") for seed, _, _ in pairs],
        ignore_index=True,
    ).sort_values(["seed", "schedule", "metric"])
    bands = pd.concat(
        [pd.read_csv(root / f"seed{seed}" / "bands.csv") for seed, _, _ in pairs],
        ignore_index=True,
    ).sort_values(["seed", "schedule", "band"])
    metric_path = results / "vector_field_switch_metrics.csv"
    band_path = results / "vector_field_switch_bands.csv"
    metrics.to_csv(metric_path, index=False)
    bands.to_csv(band_path, index=False)
    return metric_path, band_path


def launch(
    results: Path,
    pairs: list[tuple[int, Path, Path]],
    devices: list[int],
    count: int,
    batch_size: int,
    evaluation_seed: int,
    overwrite: bool,
) -> None:
    pending = list(pairs)
    running: dict[int, tuple[subprocess.Popen, int, object, Path]] = {}
    output_root = results / "vector_field_switch"
    while pending or running:
        for device in devices:
            if device in running or not pending:
                continue
            seed, baseline, partial = pending.pop(0)
            output = output_root / f"seed{seed}"
            if (output / "metadata.json").exists() and not overwrite:
                print(f"skip seed {seed}", flush=True)
                continue
            output.mkdir(parents=True, exist_ok=True)
            log = (output / "run.log").open("w", encoding="utf-8")
            command = [
                sys.executable,
                str(ROOT / "experiments/rae_vector_field_switch_probe.py"),
                "--baseline",
                str(baseline),
                "--partial",
                str(partial),
                "--output",
                str(output),
                "--device",
                "cuda:0",
                "--count",
                str(int(count)),
                "--batch-size",
                str(int(batch_size)),
                "--evaluation-seed",
                str(int(evaluation_seed)),
            ]
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(device)
            environment["PYTHONUNBUFFERED"] = "1"
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            running[device] = (process, seed, log, output)
            print(f"gpu {device}: start seed {seed}", flush=True)

        if not running:
            continue
        time.sleep(2)
        for device, (process, seed, log, output) in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            log.close()
            del running[device]
            if return_code != 0:
                tail = "\n".join((output / "run.log").read_text().splitlines()[-80:])
                for other_process, _, other_log, _ in running.values():
                    other_process.terminate()
                    other_log.close()
                raise RuntimeError(f"seed {seed} failed on gpu {device}:\n{tail}")
            print(f"gpu {device}: complete seed {seed}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("run", "aggregate", "all"), default="all")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--devices", default="0,1,2")
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--evaluation-seed", type=int, default=161803)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    results = args.results.expanduser().resolve()
    pairs = paired_branches(results)
    if len(pairs) != 3:
        raise RuntimeError(f"expected three paired seeds, found {len(pairs)}")
    if args.mode in {"run", "all"}:
        devices = [int(value) for value in args.devices.split(",") if value.strip()]
        launch(
            results,
            pairs,
            devices,
            args.count,
            args.batch_size,
            args.evaluation_seed,
            args.overwrite,
        )
    if args.mode in {"aggregate", "all"}:
        outputs = aggregate(results, pairs)
        print(json.dumps([str(path) for path in outputs], indent=2))


if __name__ == "__main__":
    main()
