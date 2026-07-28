"""Compare guided-detail and matched control sensitivity of SPC checkpoints."""

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
from experiments.rae_layerwise_path import (  # noqa: E402
    plan_layerwise_path,
    spatial_center,
)
from experiments.run_rae_path_gradient_interference import (  # noqa: E402
    _load_basis,
    _load_online_model,
    basis_projection,
)
from experiments.run_rae_spc_cross_path_study import (  # noqa: E402
    component_shift_per_sample,
    evaluation_path_kwargs,
)
from experiments.train_rae_layerwise_path import configure_determinism  # noqa: E402


DEFAULT_RESULTS = Path.home() / "data/eqvae/experiments/rae_spc_multiseed_v1"
DEFAULT_OUTPUT = DEFAULT_RESULTS / "directional_sensitivity"


def orthogonal_control_basis(
    guided_basis: torch.Tensor, *, seed: int
) -> torch.Tensor:
    channels, rank = guided_basis.shape
    if 2 * rank > channels:
        raise ValueError("control rank does not fit in the guided orthogonal complement")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    random = torch.randn(
        channels, rank, generator=generator, dtype=torch.float64
    ).to(guided_basis.device)
    guided, _ = torch.linalg.qr(guided_basis.double(), mode="reduced")
    random = random - guided @ (guided.transpose(0, 1) @ random)
    control, _ = torch.linalg.qr(random, mode="reduced")
    overlap = torch.linalg.matrix_norm(guided.transpose(0, 1) @ control)
    if float(overlap) > 1e-10:
        raise RuntimeError("failed to construct an orthogonal control basis")
    return control.to(dtype=guided_basis.dtype)


def projected_residual(value: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    _, residual = spatial_center(value)
    rows = residual.permute(0, 2, 3, 1).reshape(-1, value.shape[1])
    projected = (rows @ basis) @ basis.transpose(0, 1)
    return projected.reshape(
        value.shape[0], value.shape[2], value.shape[3], value.shape[1]
    ).permute(0, 3, 1, 2).contiguous()


def match_per_sample_norm(
    control: torch.Tensor, reference: torch.Tensor, *, eps: float = 1e-12
) -> torch.Tensor:
    control_norm = torch.linalg.vector_norm(control.flatten(1), dim=1).clamp_min(eps)
    reference_norm = torch.linalg.vector_norm(reference.flatten(1), dim=1)
    scale = (reference_norm / control_norm).reshape(
        (-1,) + (1,) * (control.ndim - 1)
    )
    return control * scale


@torch.no_grad()
def evaluate_directional_sensitivity(
    model: torch.nn.Module,
    clean: torch.Tensor,
    labels: torch.Tensor,
    noise: torch.Tensor,
    guided_basis: torch.Tensor,
    control_basis: torch.Tensor,
    path_kwargs: dict[str, dict[str, object]],
    *,
    seed: int,
    condition: str,
    step: int,
    times: tuple[float, ...],
    batch_size: int,
    device: torch.device,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for time_value in times:
        for start in range(0, len(clean), batch_size):
            end = min(start + batch_size, len(clean))
            data = clean[start:end].to(device)
            batch_noise = noise[start:end].to(device)
            batch_labels = labels[start:end].to(device)
            time_tensor = torch.full(
                (len(data),), float(time_value), dtype=torch.float32, device=device
            )
            static = plan_layerwise_path(
                data, batch_noise, time_tensor, guided_basis, **path_kwargs["static"]
            )
            spc = plan_layerwise_path(
                data, batch_noise, time_tensor, guided_basis, **path_kwargs["spc"]
            )
            guided_delta = spc.state - static.state
            control_component = projected_residual(data, control_basis)
            control_delta = -match_per_sample_norm(control_component, guided_delta)
            states = {
                "base": static.state,
                "guided": static.state + guided_delta,
                "control": static.state + control_delta,
            }
            predictions = {
                name: model(state, time_tensor, y=batch_labels)
                for name, state in states.items()
            }
            input_energy = guided_delta.square().flatten(1).mean(1)
            for direction in ("guided", "control"):
                semantic_shift, basis_shift = component_shift_per_sample(
                    predictions[direction], predictions["base"], guided_basis
                )
                total_shift = semantic_shift + basis_shift
                actual_input = states[direction] - states["base"]
                actual_input_energy = actual_input.square().flatten(1).mean(1)
                for offset in range(len(data)):
                    denominator = float(actual_input_energy[offset].clamp_min(1e-20))
                    rows.append(
                        {
                            "seed": int(seed),
                            "condition": condition,
                            "checkpoint_step": int(step),
                            "time": float(time_value),
                            "sample_index": int(start + offset),
                            "direction": direction,
                            "input_energy": float(actual_input_energy[offset]),
                            "reference_input_energy": float(input_energy[offset]),
                            "semantic_prediction_shift": float(
                                semantic_shift[offset]
                            ),
                            "basis_prediction_shift": float(basis_shift[offset]),
                            "total_prediction_shift": float(total_shift[offset]),
                            "semantic_gain": float(semantic_shift[offset]) / denominator,
                            "basis_gain": float(basis_shift[offset]) / denominator,
                            "total_gain": float(total_shift[offset]) / denominator,
                        }
                    )
    return pd.DataFrame(rows)


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
        "--control-seed",
        str(args.control_seed),
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
    guided_basis = _load_basis(manifest).to(device)
    control_basis = orthogonal_control_basis(
        guided_basis, seed=args.control_seed
    ).to(device)
    paths = evaluation_path_kwargs(spc_manifest)
    configure_determinism(args.probe_seed)
    frames = []
    times = tuple(float(value) for value in args.times.split(",") if value.strip())
    for step in (args.switch_step, args.endpoint):
        model = _load_online_model(branch, step, device)
        frames.append(
            evaluate_directional_sensitivity(
                model,
                clean,
                labels,
                noise,
                guided_basis,
                control_basis,
                paths,
                seed=seed,
                condition=args.condition,
                step=step,
                times=times,
                batch_size=args.batch_size,
                device=device,
            )
        )
        del model
        torch.cuda.empty_cache()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(
        output / f"samples_{name}.csv", index=False
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
        raise RuntimeError(f"directional sensitivity failures: {failures}")
    samples = pd.concat(
        [pd.read_csv(path) for path in sorted(output.glob("samples_seed*.csv"))],
        ignore_index=True,
    )
    samples.to_csv(output / "sensitivity_samples.csv", index=False)
    samples.groupby(
        ["seed", "condition", "checkpoint_step", "time", "direction"],
        as_index=False,
    )[
        [
            "input_energy",
            "reference_input_energy",
            "semantic_prediction_shift",
            "basis_prediction_shift",
            "total_prediction_shift",
            "semantic_gain",
            "basis_gain",
            "total_gain",
        ]
    ].mean().to_csv(output / "sensitivity_aggregate.csv", index=False)


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
    parser.add_argument("--control-seed", type=int, default=20_260_731)
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
