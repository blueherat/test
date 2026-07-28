"""Measure how changing only the detail path perturbs semantic prediction."""

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
from experiments.rae_layerwise_path import plan_layerwise_path  # noqa: E402
from experiments.run_rae_path_gradient_interference import (  # noqa: E402
    _load_basis,
    _load_online_model,
    basis_projection,
)
from experiments.train_rae_layerwise_path import configure_determinism  # noqa: E402


DEFAULT_RESULTS = Path.home() / "data/eqvae/experiments/rae_spc_multiseed_v1"
DEFAULT_OUTPUT = DEFAULT_RESULTS / "cross_path_probe"


def component_losses_per_sample(
    prediction: torch.Tensor, target: torch.Tensor, basis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    error = prediction.float() - target.float()
    basis_error = basis_projection(error, basis)
    semantic_error = error - basis_error
    return (
        semantic_error.square().flatten(1).mean(1),
        basis_error.square().flatten(1).mean(1),
    )


def component_shift_per_sample(
    left: torch.Tensor, right: torch.Tensor, basis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    difference = left.float() - right.float()
    basis_difference = basis_projection(difference, basis)
    semantic_difference = difference - basis_difference
    return (
        semantic_difference.square().flatten(1).mean(1),
        basis_difference.square().flatten(1).mean(1),
    )


def evaluation_path_kwargs(spc_manifest: dict[str, object]) -> dict[str, dict[str, object]]:
    shared = {
        "power": float(spc_manifest["path_power"]),
        "family": str(spc_manifest.get("path_family", "power")),
        "floor": float(spc_manifest.get("path_floor", 0.0)),
        "alpha": float(spc_manifest.get("path_alpha", 1.0)),
        "detail_scale": float(spc_manifest["detail_scale"]),
    }
    return {
        "static": {**shared, "mode": "static"},
        "spc": {**shared, "mode": "annealed"},
    }


@torch.no_grad()
def evaluate_checkpoint(
    model: torch.nn.Module,
    clean: torch.Tensor,
    labels: torch.Tensor,
    noise: torch.Tensor,
    basis: torch.Tensor,
    path_kwargs: dict[str, dict[str, object]],
    *,
    seed: int,
    condition: str,
    step: int,
    times: tuple[float, ...],
    batch_size: int,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    loss_rows: list[dict[str, float | int | str]] = []
    shift_rows: list[dict[str, float | int | str]] = []
    for time_value in times:
        for start in range(0, len(clean), batch_size):
            end = min(start + batch_size, len(clean))
            data = clean[start:end].to(device)
            batch_noise = noise[start:end].to(device)
            batch_labels = labels[start:end].to(device)
            time_tensor = torch.full(
                (len(data),), float(time_value), dtype=torch.float32, device=device
            )
            plans = {
                name: plan_layerwise_path(
                    data, batch_noise, time_tensor, basis, **kwargs
                )
                for name, kwargs in path_kwargs.items()
            }
            predictions = {
                name: model(plan.state, time_tensor, y=batch_labels)
                for name, plan in plans.items()
            }
            for path_name in ("static", "spc"):
                semantic_loss, basis_loss = component_losses_per_sample(
                    predictions[path_name], plans[path_name].target, basis
                )
                for offset in range(len(data)):
                    loss_rows.append(
                        {
                            "seed": int(seed),
                            "condition": condition,
                            "checkpoint_step": int(step),
                            "time": float(time_value),
                            "sample_index": int(start + offset),
                            "evaluation_path": path_name,
                            "semantic_loss": float(semantic_loss[offset]),
                            "basis_loss": float(basis_loss[offset]),
                            "total_loss": float(
                                semantic_loss[offset] + basis_loss[offset]
                            ),
                        }
                    )
            semantic_prediction_shift, basis_prediction_shift = component_shift_per_sample(
                predictions["spc"], predictions["static"], basis
            )
            semantic_target_shift, basis_target_shift = component_shift_per_sample(
                plans["spc"].target, plans["static"].target, basis
            )
            for offset in range(len(data)):
                shift_rows.append(
                    {
                        "seed": int(seed),
                        "condition": condition,
                        "checkpoint_step": int(step),
                        "time": float(time_value),
                        "sample_index": int(start + offset),
                        "semantic_prediction_shift": float(
                            semantic_prediction_shift[offset]
                        ),
                        "basis_prediction_shift": float(basis_prediction_shift[offset]),
                        "semantic_target_shift": float(semantic_target_shift[offset]),
                        "basis_target_shift": float(basis_target_shift[offset]),
                    }
                )
    return pd.DataFrame(loss_rows), pd.DataFrame(shift_rows)


def worker_command(
    args: argparse.Namespace, *, seed: int, condition: str
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--results",
        str(args.results),
        "--output",
        str(args.output),
        "--seeds",
        str(seed),
        "--condition",
        condition,
        "--endpoint",
        str(args.endpoint),
        "--switch-step",
        str(args.switch_step),
        "--cache-start",
        str(args.cache_start),
        "--count",
        str(args.count),
        "--batch-size",
        str(args.batch_size),
        "--probe-seed",
        str(args.probe_seed),
        "--times",
        args.times,
    ]


def run_worker(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    seed = int(args.seeds)
    root = args.results.expanduser().resolve()
    name = branch_name(seed, args.condition, args.endpoint, args.switch_step)
    branch = root / name
    manifest = json.loads((branch / "manifest.json").read_text(encoding="utf-8"))
    spc_branch = root / branch_name(seed, "spc", args.endpoint, args.switch_step)
    spc_manifest = json.loads(
        (spc_branch / "manifest.json").read_text(encoding="utf-8")
    )
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
    paths = evaluation_path_kwargs(spc_manifest)
    configure_determinism(args.probe_seed)
    loss_frames = []
    shift_frames = []
    times = tuple(float(value) for value in args.times.split(",") if value.strip())
    for step in (args.switch_step, args.endpoint):
        model = _load_online_model(branch, step, device)
        losses, shifts = evaluate_checkpoint(
            model,
            clean,
            labels,
            noise,
            basis,
            paths,
            seed=seed,
            condition=args.condition,
            step=step,
            times=times,
            batch_size=args.batch_size,
            device=device,
        )
        loss_frames.append(losses)
        shift_frames.append(shifts)
        del model
        torch.cuda.empty_cache()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    pd.concat(loss_frames, ignore_index=True).to_csv(
        output / f"loss_{name}.csv", index=False
    )
    pd.concat(shift_frames, ignore_index=True).to_csv(
        output / f"shift_{name}.csv", index=False
    )
    print(f"completed {name}")


def launch(args: argparse.Namespace) -> None:
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    devices = tuple(int(value) for value in args.devices.split(",") if value.strip())
    pending = [
        (seed, condition)
        for seed, condition, _ in planned_branches(
            seeds, args.endpoint, args.switch_step
        )
    ]
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    active: dict[int, tuple[int, str, subprocess.Popen, object]] = {}
    failures = []
    while pending or active:
        for device in devices:
            if device in active or not pending:
                continue
            seed, condition = pending.pop(0)
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(device)
            environment["PYTHONUNBUFFERED"] = "1"
            handle = (output / f"worker_seed{seed}_{condition}.log").open(
                "a", encoding="utf-8"
            )
            process = subprocess.Popen(
                worker_command(args, seed=seed, condition=condition),
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
        raise RuntimeError(f"cross-path probe failures: {failures}")
    losses = pd.concat(
        [pd.read_csv(path) for path in sorted(output.glob("loss_seed*.csv"))],
        ignore_index=True,
    )
    shifts = pd.concat(
        [pd.read_csv(path) for path in sorted(output.glob("shift_seed*.csv"))],
        ignore_index=True,
    )
    losses.to_csv(output / "loss_samples.csv", index=False)
    shifts.to_csv(output / "shift_samples.csv", index=False)
    losses.groupby(
        ["seed", "condition", "checkpoint_step", "time", "evaluation_path"],
        as_index=False,
    )[["semantic_loss", "basis_loss", "total_loss"]].mean().to_csv(
        output / "loss_aggregate.csv", index=False
    )
    shifts.groupby(
        ["seed", "condition", "checkpoint_step", "time"], as_index=False
    )[
        [
            "semantic_prediction_shift",
            "basis_prediction_shift",
            "semantic_target_shift",
            "basis_target_shift",
        ]
    ].mean().to_csv(output / "shift_aggregate.csv", index=False)


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
    parser.add_argument("--times", default="0.85,0.3,0.1")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count % args.batch_size:
        raise ValueError("count must be divisible by batch size")
    if args.worker:
        run_worker(args)
    else:
        launch(args)


if __name__ == "__main__":
    main()
