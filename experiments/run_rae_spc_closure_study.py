"""Evaluate decoder closure for paired SPC models across training seeds."""

from __future__ import annotations

import argparse
import gc
import json
import math
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
from experiments.rae_teacher_rollout_gap import (  # noqa: E402
    configure_fp32,
    official_time_grid,
)
from experiments.run_rae_decoder_risk_phase0 import (  # noqa: E402
    DEFAULT_AUDIT_CACHE,
    _closure_rows,
    _load_cache_tensor,
    _load_full_rae,
    _sample_endpoints,
)
from experiments.run_rae_path_schedule_closure import _load_stage2_model  # noqa: E402


DEFAULT_RESULTS = Path.home() / "data/eqvae/experiments/rae_spc_multiseed_v1"
DEFAULT_OUTPUT = DEFAULT_RESULTS / "closure_probe"
METRICS = ("cycle_relative_rms", "local_decoder_sensitivity")


def worker_command(
    results: Path,
    output: Path,
    *,
    seed: int,
    condition: str,
    endpoint: int,
    switch_step: int,
    count: int,
    sampling_batch_size: int,
    closure_batch_size: int,
    sampling_steps: int,
    probe_seed: int,
    perturb_fraction: float,
    audit_cache: Path,
    weight_source: str,
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
        "--count",
        str(count),
        "--sampling-batch-size",
        str(sampling_batch_size),
        "--closure-batch-size",
        str(closure_batch_size),
        "--sampling-steps",
        str(sampling_steps),
        "--probe-seed",
        str(probe_seed),
        "--perturb-fraction",
        str(perturb_fraction),
        "--audit-cache",
        str(audit_cache),
        "--weight-source",
        weight_source,
    ]


def run_worker(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    configure_fp32(args.probe_seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    seed = int(args.seeds)
    name = branch_name(seed, args.condition, args.endpoint, args.switch_step)
    branch = args.results.expanduser().resolve() / name
    model, config = _load_stage2_model(
        branch, args.endpoint, device, weight_source=args.weight_source
    )
    shift = math.sqrt(
        float(config.misc.time_dist_shift_dim) / float(config.misc.time_dist_shift_base)
    )
    times = official_time_grid(args.sampling_steps, time_shift=shift).to(device)
    endpoints, _ = _sample_endpoints(
        model,
        count=args.count,
        batch_size=args.sampling_batch_size,
        times=times,
        seed=args.probe_seed,
        device=device,
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    rae = _load_full_rae(config, device)
    clean_reference = _load_cache_tensor(
        args.audit_cache.expanduser(), "calibration", args.reference_count
    )
    rows = _closure_rows(
        rae,
        endpoints,
        clean_reference,
        source=args.condition,
        batch_size=args.closure_batch_size,
        perturb_fraction=args.perturb_fraction,
        seed=args.probe_seed,
        device=device,
    )
    table = pd.DataFrame(rows)
    table.insert(0, "training_seed", seed)
    table.insert(1, "condition", args.condition)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / f"closure_{name}.csv", index=False)
    print(f"completed {name}")


def summarize_closure(table: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    summary = (
        table.groupby(["training_seed", "condition"])[list(METRICS)]
        .median()
        .reset_index()
    )
    wide = summary.pivot(index="training_seed", columns="condition", values=list(METRICS))
    rows = []
    for seed in wide.index:
        row: dict[str, float | int] = {"training_seed": int(seed)}
        for metric in METRICS:
            static = float(wide.loc[seed, (metric, "static")])
            spc = float(wide.loc[seed, (metric, "spc")])
            row[f"static_{metric}"] = static
            row[f"spc_{metric}"] = spc
            row[f"delta_{metric}"] = spc - static
        rows.append(row)
    paired = pd.DataFrame(rows).sort_values("training_seed")
    decision = {
        "seed_count": int(len(paired)),
        "spc_cycle_worse_count": int((paired["delta_cycle_relative_rms"] > 0).sum()),
        "spc_sensitivity_worse_count": int(
            (paired["delta_local_decoder_sensitivity"] > 0).sum()
        ),
        "closure_not_systematically_worse": bool(
            (paired["delta_cycle_relative_rms"] > 0).sum() < 4
            and (paired["delta_local_decoder_sensitivity"] > 0).sum() < 4
        ),
    }
    return paired, decision


def launch(args: argparse.Namespace) -> None:
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    devices = tuple(int(value) for value in args.devices.split(",") if value.strip())
    results = args.results.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    pending = [
        (seed, condition)
        for seed, condition, _ in planned_branches(seeds, args.endpoint, args.switch_step)
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
                count=args.count,
                sampling_batch_size=args.sampling_batch_size,
                closure_batch_size=args.closure_batch_size,
                sampling_steps=args.sampling_steps,
                probe_seed=args.probe_seed,
                perturb_fraction=args.perturb_fraction,
                audit_cache=args.audit_cache,
                weight_source=args.weight_source,
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
        raise RuntimeError(f"closure failures: {failures}")
    table = pd.concat(
        [pd.read_csv(path) for path in sorted(output.glob("closure_seed*.csv"))],
        ignore_index=True,
    )
    table.to_csv(output / "closure_metrics.csv", index=False)
    paired, decision = summarize_closure(table)
    paired.to_csv(output / "closure_paired.csv", index=False)
    (output / "decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(paired.to_string(index=False))
    print(json.dumps(decision, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--condition", choices=("static", "spc"), default="static")
    parser.add_argument("--endpoint", type=int, default=5000)
    parser.add_argument("--switch-step", type=int, default=2000)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--sampling-batch-size", type=int, default=8)
    parser.add_argument("--closure-batch-size", type=int, default=4)
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--reference-count", type=int, default=128)
    parser.add_argument("--probe-seed", type=int, default=20_260_731)
    parser.add_argument("--perturb-fraction", type=float, default=1e-3)
    parser.add_argument("--audit-cache", type=Path, default=DEFAULT_AUDIT_CACHE)
    parser.add_argument("--weight-source", choices=("model", "ema"), default="model")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker:
        run_worker(args)
    else:
        launch(args)


if __name__ == "__main__":
    main()
