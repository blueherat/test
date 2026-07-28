"""Run a standardized static-objective gradient probe across SPC training seeds."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
RAE_SRC = ROOT / "external/RAE/src"
for import_path in (ROOT, RAE_SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from experiments.evaluate_rae_spc_multiseed import (  # noqa: E402
    DEFAULT_SEEDS,
    branch_name,
    planned_branches,
)
from experiments.rae_latent_cache import CachedRAELatentDataset  # noqa: E402
from experiments.run_rae_path_gradient_interference import (  # noqa: E402
    _load_basis,
    audit_checkpoint,
)
from experiments.train_rae_layerwise_path import configure_determinism  # noqa: E402


DEFAULT_RESULTS = Path.home() / "data/eqvae/experiments/rae_spc_multiseed_v1"
DEFAULT_OUTPUT = Path.home() / "data/eqvae/experiments/rae_spc_multiseed_v1/gradient_probe"


def worker_command(
    results: Path,
    output: Path,
    *,
    seed: int,
    condition: str,
    endpoint: int,
    switch_step: int,
    cache_start: int,
    count: int,
    batch_size: int,
    probe_seed: int,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--results",
        str(results),
        "--output",
        str(output),
        "--seeds",
        str(seed),
        "--condition",
        condition,
        "--endpoint",
        str(endpoint),
        "--switch-step",
        str(switch_step),
        "--cache-start",
        str(cache_start),
        "--count",
        str(count),
        "--batch-size",
        str(batch_size),
        "--probe-seed",
        str(probe_seed),
    ]


def run_worker(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    seed = int(args.seeds)
    name = branch_name(seed, args.condition, args.endpoint, args.switch_step)
    branch = args.results.expanduser().resolve() / name
    manifest = json.loads((branch / "manifest.json").read_text(encoding="utf-8"))
    dataset = CachedRAELatentDataset(
        Path(str(manifest["latent_cache"])),
        start=args.cache_start,
        stop=args.cache_start + args.count,
    )
    samples = [dataset[index] for index in range(len(dataset))]
    clean = torch.stack([sample[0] for sample in samples])
    labels = torch.tensor([sample[1] for sample in samples], dtype=torch.long)
    generator = torch.Generator(device="cpu").manual_seed(args.probe_seed)
    noise = torch.randn(clean.shape, generator=generator, dtype=torch.float32)
    basis = _load_basis(manifest).to(device)
    configure_determinism(args.probe_seed)
    batch_frames = []
    aggregate_frames = []
    for step in (args.switch_step, args.endpoint):
        batch, aggregate, _ = audit_checkpoint(
            args.condition,
            branch,
            step,
            clean,
            labels,
            noise,
            basis,
            manifest,
            batch_size=args.batch_size,
            times=(0.3, 0.1),
            device=device,
            path_mode_override="static",
        )
        batch.insert(0, "training_seed", seed)
        aggregate.insert(0, "training_seed", seed)
        batch_frames.append(batch)
        aggregate_frames.append(aggregate)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    pd.concat(batch_frames, ignore_index=True).to_csv(
        output / f"batch_{name}.csv", index=False
    )
    pd.concat(aggregate_frames, ignore_index=True).to_csv(
        output / f"aggregate_{name}.csv", index=False
    )
    print(f"completed {name}")


def launch(args: argparse.Namespace) -> None:
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    devices = tuple(int(value) for value in args.devices.split(",") if value.strip())
    results = args.results.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    pending = [
        (seed, condition)
        for seed, condition, _ in planned_branches(
            seeds, args.endpoint, args.switch_step
        )
    ]
    active: dict[int, tuple[int, str, subprocess.Popen, object]] = {}
    failures = []
    while pending or active:
        for device in devices:
            if not pending or device in active:
                continue
            seed, condition = pending.pop(0)
            command = worker_command(
                results,
                output,
                seed=seed,
                condition=condition,
                endpoint=args.endpoint,
                switch_step=args.switch_step,
                cache_start=args.cache_start,
                count=args.count,
                batch_size=args.batch_size,
                probe_seed=args.probe_seed,
            )
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(device)
            environment["PYTHONUNBUFFERED"] = "1"
            handle = (output / f"worker_seed{seed}_{condition}.log").open(
                "a", encoding="utf-8"
            )
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            active[device] = (seed, condition, process, handle)
            print(f"started seed={seed} {condition} cuda={device}", flush=True)
        time.sleep(2)
        for device, (seed, condition, process, handle) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            del active[device]
            print(f"finished seed={seed} {condition} exit={code}", flush=True)
            if code:
                failures.append((seed, condition, code))
    if failures:
        raise RuntimeError(f"gradient probe failures: {failures}")
    batches = pd.concat(
        [pd.read_csv(path) for path in sorted(output.glob("batch_seed*.csv"))],
        ignore_index=True,
    )
    aggregate = pd.concat(
        [pd.read_csv(path) for path in sorted(output.glob("aggregate_seed*.csv"))],
        ignore_index=True,
    )
    batches.to_csv(output / "batch_metrics.csv", index=False)
    aggregate.to_csv(output / "aggregate_metrics.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--condition", choices=("static", "spc"), default="static")
    parser.add_argument("--endpoint", type=int, default=5000)
    parser.add_argument("--switch-step", type=int, default=2000)
    parser.add_argument("--cache-start", type=int, default=100_288)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--probe-seed", type=int, default=20_260_730)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count % (2 * args.batch_size):
        raise ValueError("count must divide into equal calibration/test batches")
    if args.worker:
        run_worker(args)
    else:
        launch(args)


if __name__ == "__main__":
    main()
