"""Measure stage-2 sensitivity to PCA and predictability-defined RAE subspaces."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
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
)
from experiments.rae_latent_cache import CachedRAELatentDataset  # noqa: E402
from experiments.rae_layerwise_path import plan_layerwise_path, spatial_center  # noqa: E402
from experiments.run_rae_path_gradient_interference import (  # noqa: E402
    _load_basis,
    _load_online_model,
)
from experiments.run_rae_spc_cross_path_study import evaluation_path_kwargs  # noqa: E402
from experiments.run_rae_spc_directional_sensitivity import (  # noqa: E402
    match_per_sample_norm,
    projected_residual,
)
from experiments.train_rae_layerwise_path import configure_determinism  # noqa: E402


DEFAULT_RESULTS = Path.home() / "data/eqvae/experiments/rae_spc_multiseed_v1"
DEFAULT_BASES = (
    DEFAULT_RESULTS / "evaluation/predictability_basis_v1/bases.pt"
)
DEFAULT_OUTPUT = (
    DEFAULT_RESULTS / "evaluation/predictability_basis_sensitivity_v1"
)


def component_shift(
    value: torch.Tensor,
    basis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean, residual = spatial_center(value)
    rows = residual.permute(0, 2, 3, 1).reshape(-1, value.shape[1])
    basis_rows = (rows @ basis) @ basis.transpose(0, 1)
    basis_value = basis_rows.reshape(
        value.shape[0], value.shape[2], value.shape[3], value.shape[1]
    ).permute(0, 3, 1, 2)
    complement = mean + residual - basis_value
    return (
        complement.square().flatten(1).mean(1),
        basis_value.square().flatten(1).mean(1),
    )


@torch.no_grad()
def evaluate_basis_sensitivity(
    model: torch.nn.Module,
    clean: torch.Tensor,
    labels: torch.Tensor,
    noise: torch.Tensor,
    reference_basis: torch.Tensor,
    bases: dict[str, torch.Tensor],
    path_kwargs: dict[str, dict[str, object]],
    *,
    seed: int,
    step: int,
    times: tuple[float, ...],
    batch_size: int,
    device: torch.device,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    device_bases = {
        name: basis.to(device=device, dtype=torch.float32)
        for name, basis in bases.items()
    }
    reference_basis = reference_basis.to(device=device, dtype=torch.float32)
    for time_value in times:
        for start in range(0, len(clean), int(batch_size)):
            end = min(start + int(batch_size), len(clean))
            data = clean[start:end].to(device)
            batch_noise = noise[start:end].to(device)
            batch_labels = labels[start:end].to(device)
            time_tensor = torch.full(
                (len(data),), float(time_value), dtype=torch.float32, device=device
            )
            static = plan_layerwise_path(
                data,
                batch_noise,
                time_tensor,
                reference_basis,
                **path_kwargs["static"],
            )
            spc = plan_layerwise_path(
                data,
                batch_noise,
                time_tensor,
                reference_basis,
                **path_kwargs["spc"],
            )
            reference_delta = spc.state - static.state
            reference_energy = reference_delta.square().flatten(1).mean(1)
            base_prediction = model(static.state, time_tensor, y=batch_labels)
            for name, basis in device_bases.items():
                if name == "reference_guided":
                    delta = reference_delta
                else:
                    component = projected_residual(data, basis)
                    delta = -match_per_sample_norm(component, reference_delta)
                prediction = model(static.state + delta, time_tensor, y=batch_labels)
                prediction_delta = prediction - base_prediction
                complement_shift, basis_shift = component_shift(
                    prediction_delta, basis
                )
                total_shift = prediction_delta.square().flatten(1).mean(1)
                input_energy = delta.square().flatten(1).mean(1)
                clean_component_energy = projected_residual(data, basis).square().flatten(1).mean(1)
                for offset in range(len(data)):
                    denominator = float(input_energy[offset].clamp_min(1e-20))
                    rows.append(
                        {
                            "seed": int(seed),
                            "checkpoint_step": int(step),
                            "time": float(time_value),
                            "sample_index": int(start + offset),
                            "basis": name,
                            "basis_family": "random" if name.startswith("random_") else name,
                            "input_energy": float(input_energy[offset]),
                            "reference_input_energy": float(reference_energy[offset]),
                            "clean_component_energy": float(clean_component_energy[offset]),
                            "total_gain": float(total_shift[offset]) / denominator,
                            "within_basis_gain": float(basis_shift[offset]) / denominator,
                            "complement_gain": float(complement_shift[offset]) / denominator,
                        }
                    )
    return pd.DataFrame(rows)


def worker_command(args: argparse.Namespace, seed: int) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--results",
        str(args.results),
        "--bases",
        str(args.bases),
        "--output",
        str(args.output),
        "--seeds",
        str(seed),
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
        "--rank",
        str(args.rank),
        "--basis-set",
        args.basis_set,
        "--steps",
        args.steps,
    ]


def run_worker(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    seed = int(args.seeds)
    root = args.results.expanduser().resolve()
    static_name = branch_name(seed, "static", args.endpoint, args.switch_step)
    static_branch = root / static_name
    manifest = json.loads(
        (static_branch / "manifest.json").read_text(encoding="utf-8")
    )
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
    reference_basis = _load_basis(manifest).to(device)
    payload = torch.load(args.bases.expanduser(), map_location="cpu", weights_only=False)
    if args.basis_set == "core":
        rank_bases = payload["bases"].get(
            args.rank, payload["bases"].get(str(args.rank))
        )
        if rank_bases is None:
            raise KeyError(f"rank {args.rank} is absent from {args.bases}")
        selected = {
            name: basis
            for name, basis in rank_bases.items()
            if name in {"reference_guided", "fractional", "top_pca"}
            or name.startswith("random_")
        }
    else:
        selected = dict(payload["blocks"])
    paths = evaluation_path_kwargs(spc_manifest)
    configure_determinism(args.probe_seed)
    times = tuple(float(value) for value in args.times.split(",") if value.strip())
    frames = []
    steps = tuple(int(value) for value in args.steps.split(",") if value.strip())
    for step in steps:
        model = _load_online_model(static_branch, step, device)
        frames.append(
            evaluate_basis_sensitivity(
                model,
                clean,
                labels,
                noise,
                reference_basis,
                selected,
                paths,
                seed=seed,
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
        output / f"samples_seed{seed}.csv", index=False
    )
    print(f"completed seed={seed}", flush=True)


def summarize(samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_seed = samples.groupby(
        ["seed", "checkpoint_step", "time", "basis", "basis_family"],
        as_index=False,
    )[
        [
            "input_energy",
            "clean_component_energy",
            "total_gain",
            "within_basis_gain",
            "complement_gain",
        ]
    ].mean()
    nonrandom = per_seed[per_seed["basis_family"] != "random"].copy()
    random = per_seed[per_seed["basis_family"] == "random"].groupby(
        ["seed", "checkpoint_step", "time"], as_index=False
    )[
        ["input_energy", "clean_component_energy", "total_gain", "within_basis_gain", "complement_gain"]
    ].mean()
    random["basis"] = "random_mean"
    random["basis_family"] = "random"
    display = pd.concat([nonrandom, random], ignore_index=True)
    random_gain = random.rename(columns={"total_gain": "random_total_gain"})[
        ["seed", "checkpoint_step", "time", "random_total_gain"]
    ]
    display = display.merge(
        random_gain, on=["seed", "checkpoint_step", "time"], how="left"
    )
    display["gain_over_random"] = display["total_gain"] / display[
        "random_total_gain"
    ].clip(lower=1e-20)
    aggregate = display.groupby(
        ["checkpoint_step", "time", "basis", "basis_family"], as_index=False
    )[
        [
            "input_energy",
            "clean_component_energy",
            "total_gain",
            "within_basis_gain",
            "complement_gain",
            "gain_over_random",
        ]
    ].agg(["mean", "std"])
    aggregate.columns = [
        "_".join(value for value in column if value)
        for column in aggregate.columns.to_flat_index()
    ]
    return display, aggregate


def plot_summary(display: pd.DataFrame, output: Path) -> None:
    names = ["reference_guided", "fractional", "top_pca", "random_mean"]
    labels = {
        "reference_guided": "original guided",
        "fractional": "whitened predictability",
        "top_pca": "top PCA",
        "random_mean": "random mean",
    }
    colors = {
        "reference_guided": "#c84c32",
        "fractional": "#2678a8",
        "top_pca": "#2f855a",
        "random_mean": "#777777",
    }
    steps = sorted(int(value) for value in display["checkpoint_step"].unique())
    fig, axes = plt.subplots(1, len(steps), figsize=(7.5 * len(steps), 5.8), constrained_layout=True)
    if len(steps) == 1:
        axes = [axes]
    for axis, step in zip(axes, steps):
        selected_step = display[display["checkpoint_step"] == step]
        for name in names:
            selected = selected_step[selected_step["basis"] == name]
            grouped = selected.groupby("time")["gain_over_random"].agg(["mean", "std"]).reset_index()
            axis.errorbar(
                grouped["time"],
                grouped["mean"],
                yerr=grouped["std"],
                marker="o",
                capsize=3,
                label=labels[name],
                color=colors[name],
            )
        axis.axhline(1.0, color="#999999", linewidth=1, linestyle="--")
        axis.set_title(f"Static model at step {step}")
        axis.set_xlabel("noise time t")
        axis.set_ylabel("directional gain / random-direction gain")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    fig.savefig(output / "basis_sensitivity_over_time.png", dpi=180)
    plt.close(fig)


def plot_block_summary(display: pd.DataFrame, output: Path) -> None:
    selected = display[
        display["basis"].str.startswith(("absolute_", "fractional_", "pca_"))
    ].copy()
    parts = selected["basis"].str.split("_", expand=True)
    selected["family"] = parts[0]
    selected["block_index"] = parts[1].astype(int) // 16
    colors = {"absolute": "#c84c32", "fractional": "#2678a8", "pca": "#2f855a"}
    labels = {
        "absolute": "absolute predictable energy",
        "fractional": "whitened predictability",
        "pca": "PCA",
    }
    panels = [
        (int(step), float(time_value))
        for step in sorted(selected["checkpoint_step"].unique())
        for time_value in sorted(selected["time"].unique())
    ]
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(7 * len(panels), 5.6),
        constrained_layout=True,
    )
    if len(panels) == 1:
        axes = [axes]
    for axis, (step, time_value) in zip(axes, panels):
        panel = selected[
            (selected["checkpoint_step"] == step) & (selected["time"] == time_value)
        ]
        for family in ("absolute", "fractional", "pca"):
            values = panel[panel["family"] == family]
            grouped = values.groupby("block_index")["gain_over_random"].agg(["mean", "std"]).reset_index()
            axis.errorbar(
                grouped["block_index"],
                grouped["mean"],
                yerr=grouped["std"],
                marker="o",
                capsize=3,
                color=colors[family],
                label=labels[family],
            )
        axis.axhline(1.0, color="#999999", linewidth=1, linestyle="--")
        axis.set_title(f"step {step}, t={time_value:g}")
        axis.set_xlabel("rank-16 block index")
        axis.set_ylabel("directional gain / random-direction gain")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    fig.savefig(output / "block_sensitivity.png", dpi=180)
    plt.close(fig)


def launch(args: argparse.Namespace) -> None:
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    devices = tuple(int(value) for value in args.devices.split(",") if value.strip())
    pending = list(seeds)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    active: dict[int, tuple[int, subprocess.Popen, object]] = {}
    failures: list[tuple[int, int]] = []
    while pending or active:
        for device in devices:
            if device in active or not pending:
                continue
            seed = pending.pop(0)
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(device)
            environment["PYTHONUNBUFFERED"] = "1"
            handle = (output / f"worker_seed{seed}.log").open("a", encoding="utf-8")
            process = subprocess.Popen(
                worker_command(args, seed),
                cwd=ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            active[device] = (seed, process, handle)
            print(f"started seed={seed} cuda={device}", flush=True)
        time.sleep(2)
        for device, (seed, process, handle) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            del active[device]
            print(f"finished seed={seed} exit={code}", flush=True)
            if code:
                failures.append((seed, code))
    if failures:
        raise RuntimeError(f"basis sensitivity failures: {failures}")
    samples = pd.concat(
        [pd.read_csv(output / f"samples_seed{seed}.csv") for seed in seeds],
        ignore_index=True,
    )
    samples.to_csv(output / "sensitivity_samples.csv", index=False)
    display, aggregate = summarize(samples)
    display.to_csv(output / "sensitivity_per_seed.csv", index=False)
    aggregate.to_csv(output / "sensitivity_aggregate.csv", index=False)
    if args.basis_set == "core":
        plot_summary(display, output)
    else:
        plot_block_summary(display, output)
    primary = display[
        (display["checkpoint_step"] == args.endpoint)
        & (display["time"] == min(float(value) for value in args.times.split(",") if float(value) >= 0.8))
    ]
    summary = {
        name: {
            "gain_over_random_mean": float(values["gain_over_random"].mean()),
            "gain_over_random_std": float(values["gain_over_random"].std()),
            "seed_count": int(values["seed"].nunique()),
        }
        for name, values in primary.groupby("basis")
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--bases", type=Path, default=DEFAULT_BASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--endpoint", type=int, default=5000)
    parser.add_argument("--switch-step", type=int, default=2000)
    parser.add_argument("--cache-start", type=int, default=100_288)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--probe-seed", type=int, default=20_260_730)
    parser.add_argument("--times", default="0.95,0.85,0.7,0.5,0.3,0.1")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--basis-set", choices=("core", "blocks"), default="core")
    parser.add_argument("--steps", default="2000,5000")
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
