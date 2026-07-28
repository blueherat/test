"""Test whether the latent trust-spectrum gain persists through ODE rollout."""

from __future__ import annotations

import argparse
import json
import math
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

from experiments.analyze_rae_predictability_gain import MATCHED_PAIRS  # noqa: E402
from experiments.evaluate_rae_spc_multiseed import DEFAULT_SEEDS, branch_name  # noqa: E402
from experiments.rae_latent_cache import CachedRAELatentDataset  # noqa: E402
from experiments.rae_layerwise_path import plan_layerwise_path  # noqa: E402
from experiments.rae_teacher_rollout_gap import official_time_grid  # noqa: E402
from experiments.run_rae_path_gradient_interference import _load_basis  # noqa: E402
from experiments.run_rae_path_schedule_closure import _load_stage2_model  # noqa: E402
from experiments.run_rae_spc_cross_path_study import evaluation_path_kwargs  # noqa: E402
from experiments.run_rae_spc_directional_sensitivity import (  # noqa: E402
    match_per_sample_norm,
    projected_residual,
)
from experiments.train_rae_layerwise_path import configure_determinism  # noqa: E402


DEFAULT_RESULTS = Path.home() / "data/eqvae/experiments/rae_spc_multiseed_v1"
DEFAULT_BASES = DEFAULT_RESULTS / "evaluation/predictability_basis_v1/bases.pt"
DEFAULT_SENSITIVITY = (
    DEFAULT_RESULTS
    / "evaluation/predictability_block_sensitivity_n128_v1/sensitivity_per_seed.csv"
)
DEFAULT_OUTPUT = DEFAULT_RESULTS / "evaluation/latent_trust_rollout_v1"


def selected_block_names() -> set[str]:
    return {name for pair in MATCHED_PAIRS for name in pair}


def perturbation_conditions(
    clean: torch.Tensor,
    reference_delta: torch.Tensor,
    bases: dict[str, torch.Tensor],
    amplitudes: tuple[float, ...],
) -> tuple[list[dict[str, object]], list[torch.Tensor]]:
    conditions: list[dict[str, object]] = []
    deltas: list[torch.Tensor] = []
    matched = selected_block_names()
    for amplitude in amplitudes:
        for name, basis in bases.items():
            if amplitude != max(amplitudes) and name not in matched:
                continue
            component = projected_residual(clean, basis.to(clean))
            delta = -float(amplitude) * match_per_sample_norm(
                component, reference_delta
            )
            conditions.append({"basis": name, "amplitude": float(amplitude)})
            deltas.append(delta)
    return conditions, deltas


@torch.no_grad()
def rollout_final(
    model: torch.nn.Module,
    initial: torch.Tensor,
    labels: torch.Tensor,
    times: torch.Tensor,
    *,
    model_batch_size: int,
) -> torch.Tensor:
    state = initial
    for current, following in zip(times[:-1], times[1:]):
        outputs = []
        for start in range(0, len(state), int(model_batch_size)):
            end = min(start + int(model_batch_size), len(state))
            batch_time = torch.full(
                (end - start,),
                float(current),
                dtype=state.dtype,
                device=state.device,
            )
            outputs.append(
                model(state[start:end], batch_time, y=labels[start:end])
            )
        velocity = torch.cat(outputs)
        state = state + (following.to(state) - current.to(state)) * velocity
    return state


@torch.no_grad()
def rollout_condition_endpoints(
    model: torch.nn.Module,
    clean: torch.Tensor,
    labels: torch.Tensor,
    noise: torch.Tensor,
    reference_basis: torch.Tensor,
    bases: dict[str, torch.Tensor],
    path_kwargs: dict[str, dict[str, object]],
    full_times: torch.Tensor,
    *,
    target_time: float,
    amplitudes: tuple[float, ...],
    model_batch_size: int,
) -> tuple[
    float,
    list[dict[str, object]],
    list[torch.Tensor],
    torch.Tensor,
]:
    """Roll out the unperturbed path and all matched perturbation conditions."""

    index = int(torch.argmin((full_times - float(target_time)).abs()).item())
    times = full_times[index:]
    actual_time = float(times[0])
    time_tensor = torch.full(
        (len(clean),), actual_time, dtype=clean.dtype, device=clean.device
    )
    static = plan_layerwise_path(
        clean, noise, time_tensor, reference_basis, **path_kwargs["static"]
    )
    spc = plan_layerwise_path(
        clean, noise, time_tensor, reference_basis, **path_kwargs["spc"]
    )
    reference_delta = spc.state - static.state
    conditions, deltas = perturbation_conditions(
        clean, reference_delta, bases, amplitudes
    )
    condition_count = 1 + len(conditions)
    initial = torch.cat([static.state] + [static.state + delta for delta in deltas])
    repeated_labels = labels.repeat(condition_count)
    endpoints = rollout_final(
        model,
        initial,
        repeated_labels,
        times,
        model_batch_size=model_batch_size,
    ).reshape(condition_count, len(clean), *clean.shape[1:])
    return actual_time, conditions, deltas, endpoints


@torch.no_grad()
def evaluate_rollout_batch(
    model: torch.nn.Module,
    clean: torch.Tensor,
    labels: torch.Tensor,
    noise: torch.Tensor,
    reference_basis: torch.Tensor,
    bases: dict[str, torch.Tensor],
    path_kwargs: dict[str, dict[str, object]],
    full_times: torch.Tensor,
    *,
    target_time: float,
    amplitudes: tuple[float, ...],
    model_batch_size: int,
) -> list[dict[str, float | str]]:
    actual_time, conditions, deltas, endpoints = rollout_condition_endpoints(
        model,
        clean,
        labels,
        noise,
        reference_basis,
        bases,
        path_kwargs,
        full_times,
        target_time=target_time,
        amplitudes=amplitudes,
        model_batch_size=model_batch_size,
    )
    base = endpoints[0]
    base_error = (base - clean).square().flatten(1).mean(1)
    rows: list[dict[str, float | str]] = []
    for condition_index, (condition, delta) in enumerate(
        zip(conditions, deltas), start=1
    ):
        endpoint = endpoints[condition_index]
        input_energy = delta.square().flatten(1).mean(1)
        endpoint_shift = (endpoint - base).square().flatten(1).mean(1)
        clean_error = (endpoint - clean).square().flatten(1).mean(1)
        for sample_index in range(len(clean)):
            denominator = float(input_energy[sample_index].clamp_min(1e-20))
            rows.append(
                {
                    "target_time": float(target_time),
                    "actual_time": actual_time,
                    "basis": str(condition["basis"]),
                    "amplitude": float(condition["amplitude"]),
                    "batch_sample_index": int(sample_index),
                    "input_energy": float(input_energy[sample_index]),
                    "endpoint_shift_energy": float(endpoint_shift[sample_index]),
                    "endpoint_shift_gain": float(endpoint_shift[sample_index])
                    / denominator,
                    "base_clean_mse": float(base_error[sample_index]),
                    "perturbed_clean_mse": float(clean_error[sample_index]),
                    "clean_mse_increase": float(
                        clean_error[sample_index] - base_error[sample_index]
                    ),
                    "clean_mse_ratio": float(
                        clean_error[sample_index]
                        / base_error[sample_index].clamp_min(1e-20)
                    ),
                }
            )
    return rows


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
        "--data-batch-size",
        str(args.data_batch_size),
        "--model-batch-size",
        str(args.model_batch_size),
        "--probe-seed",
        str(args.probe_seed),
        "--times",
        args.times,
        "--amplitudes",
        args.amplitudes,
        "--sampling-steps",
        str(args.sampling_steps),
    ]


def run_worker(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    seed = int(args.seeds)
    configure_determinism(args.probe_seed)
    root = args.results.expanduser().resolve()
    static_branch = root / branch_name(
        seed, "static", args.endpoint, args.switch_step
    )
    spc_branch = root / branch_name(seed, "spc", args.endpoint, args.switch_step)
    manifest = json.loads(
        (static_branch / "manifest.json").read_text(encoding="utf-8")
    )
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
    model, config = _load_stage2_model(
        static_branch, args.endpoint, device, weight_source="model"
    )
    shift = math.sqrt(
        float(config.misc.time_dist_shift_dim)
        / float(config.misc.time_dist_shift_base)
    )
    full_times = official_time_grid(args.sampling_steps, time_shift=shift).to(device)
    reference_basis = _load_basis(manifest).to(device)
    payload = torch.load(args.bases.expanduser(), map_location="cpu", weights_only=False)
    bases = {
        name: basis.to(device=device, dtype=torch.float32)
        for name, basis in payload["blocks"].items()
        if not name.startswith("random_")
    }
    paths = evaluation_path_kwargs(spc_manifest)
    target_times = tuple(float(value) for value in args.times.split(",") if value.strip())
    amplitudes = tuple(
        float(value) for value in args.amplitudes.split(",") if value.strip()
    )
    rows: list[dict[str, float | int | str]] = []
    for start in range(0, len(clean), args.data_batch_size):
        end = min(start + args.data_batch_size, len(clean))
        batch_clean = clean[start:end].to(device)
        batch_labels = labels[start:end].to(device)
        batch_noise = noise[start:end].to(device)
        for target_time in target_times:
            batch_rows = evaluate_rollout_batch(
                model,
                batch_clean,
                batch_labels,
                batch_noise,
                reference_basis,
                bases,
                paths,
                full_times,
                target_time=target_time,
                amplitudes=amplitudes,
                model_batch_size=args.model_batch_size,
            )
            for row in batch_rows:
                row["seed"] = seed
                row["sample_index"] = start + int(row.pop("batch_sample_index"))
            rows.extend(batch_rows)
        print(f"seed={seed} rollout {end}/{len(clean)}", flush=True)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / f"rollout_samples_seed{seed}.csv", index=False)
    print(f"completed seed={seed}", flush=True)


def pair_rollout_ratios(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for high, low in MATCHED_PAIRS:
        high_rows = per_seed[per_seed["basis"] == high].set_index(
            ["seed", "target_time", "amplitude"]
        )
        low_rows = per_seed[per_seed["basis"] == low].set_index(
            ["seed", "target_time", "amplitude"]
        )
        common = high_rows.index.intersection(low_rows.index)
        for seed, target_time, amplitude in common:
            high_row = high_rows.loc[(seed, target_time, amplitude)]
            low_row = low_rows.loc[(seed, target_time, amplitude)]
            rows.append(
                {
                    "higher_predictability_basis": high,
                    "lower_predictability_basis": low,
                    "seed": int(seed),
                    "target_time": float(target_time),
                    "amplitude": float(amplitude),
                    "endpoint_shift_gain_ratio": float(
                        high_row["endpoint_shift_gain"]
                        / max(float(low_row["endpoint_shift_gain"]), 1e-20)
                    ),
                    "endpoint_shift_energy_ratio": float(
                        high_row["endpoint_shift_energy"]
                        / max(float(low_row["endpoint_shift_energy"]), 1e-20)
                    ),
                    "clean_mse_increase_difference": float(
                        high_row["clean_mse_increase"]
                        - low_row["clean_mse_increase"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def gain_rollout_correlations(
    per_seed: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> pd.DataFrame:
    one_step = sensitivity.groupby(
        ["seed", "time", "basis"], as_index=False
    )["total_gain"].mean()
    largest_amplitude = float(per_seed["amplitude"].max())
    rollout = per_seed[per_seed["amplitude"] == largest_amplitude]
    merged = rollout.merge(
        one_step,
        left_on=["seed", "target_time", "basis"],
        right_on=["seed", "time", "basis"],
        validate="one_to_one",
    )
    rows = []
    for (seed, target_time), frame in merged.groupby(["seed", "target_time"]):
        rows.append(
            {
                "seed": int(seed),
                "target_time": float(target_time),
                "pearson_log_gain": float(
                    frame[["total_gain", "endpoint_shift_gain"]]
                    .apply(lambda value: value.clip(lower=1e-20).map(math.log))
                    .corr(method="pearson")
                    .iloc[0, 1]
                ),
                "spearman_gain": float(
                    frame[["total_gain", "endpoint_shift_gain"]]
                    .corr(method="spearman")
                    .iloc[0, 1]
                ),
                "basis_count": int(len(frame)),
            }
        )
    return pd.DataFrame(rows)


def plot_summary(pairs: pd.DataFrame, correlations: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), constrained_layout=True)
    colors = ("#2678a8", "#2f855a", "#8b5a9f")
    largest_amplitude = float(pairs["amplitude"].max())
    selected = pairs[pairs["amplitude"] == largest_amplitude]
    for (high, low), color in zip(MATCHED_PAIRS, colors):
        values = selected[
            (selected["higher_predictability_basis"] == high)
            & (selected["lower_predictability_basis"] == low)
        ]
        grouped = values.groupby("target_time")["endpoint_shift_gain_ratio"].agg(["mean", "std"]).reset_index()
        axes[0].errorbar(
            grouped["target_time"],
            grouped["mean"],
            yerr=grouped["std"],
            marker="o",
            capsize=3,
            color=color,
            label=f"{high} / {low}",
        )
    axes[0].axhline(1.0, color="#999999", linestyle="--", linewidth=1)
    axes[0].set_xlabel("teacher-path start time")
    axes[0].set_ylabel("endpoint shift-gain ratio")
    axes[0].set_title("Does the matched-variance crossover survive rollout?")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=0.25)
    for field, label, color in (
        ("spearman_gain", "Spearman", "#2678a8"),
        ("pearson_log_gain", "Pearson(log gain)", "#c84c32"),
    ):
        grouped = correlations.groupby("target_time")[field].agg(["mean", "std"]).reset_index()
        axes[1].errorbar(
            grouped["target_time"], grouped["mean"], yerr=grouped["std"], marker="o", capsize=3, color=color, label=label
        )
    axes[1].set_xlabel("teacher-path start time")
    axes[1].set_ylabel("one-step versus endpoint correlation")
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].set_title("Does local directional gain predict endpoint leverage?")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)
    fig.savefig(output / "latent_trust_rollout_summary.png", dpi=180)
    plt.close(fig)


def launch(args: argparse.Namespace) -> None:
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    devices = tuple(int(value) for value in args.devices.split(",") if value.strip())
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    pending = list(seeds)
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
        raise RuntimeError(f"latent trust rollout failures: {failures}")

    samples = pd.concat(
        [pd.read_csv(output / f"rollout_samples_seed{seed}.csv") for seed in seeds],
        ignore_index=True,
    )
    samples.to_csv(output / "rollout_samples.csv", index=False)
    per_seed = samples.groupby(
        ["seed", "target_time", "actual_time", "basis", "amplitude"],
        as_index=False,
    )[
        [
            "input_energy",
            "endpoint_shift_energy",
            "endpoint_shift_gain",
            "base_clean_mse",
            "perturbed_clean_mse",
            "clean_mse_increase",
            "clean_mse_ratio",
        ]
    ].mean()
    per_seed.to_csv(output / "rollout_per_seed.csv", index=False)
    pairs = pair_rollout_ratios(per_seed)
    pairs.to_csv(output / "matched_pair_rollout_ratios.csv", index=False)
    sensitivity = pd.read_csv(args.sensitivity.expanduser())
    correlations = gain_rollout_correlations(per_seed, sensitivity)
    correlations.to_csv(output / "gain_rollout_correlations.csv", index=False)
    plot_summary(pairs, correlations, output)

    largest_amplitude = float(pairs["amplitude"].max())
    primary_pairs = pairs[pairs["amplitude"] == largest_amplitude]
    pair_summary = (
        primary_pairs.groupby(
            ["higher_predictability_basis", "lower_predictability_basis", "target_time"]
        )["endpoint_shift_gain_ratio"]
        .agg(["mean", "std"])
        .reset_index()
    )
    correlation_summary = (
        correlations.groupby("target_time")[["pearson_log_gain", "spearman_gain"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    correlation_summary.columns = [
        "_".join(value for value in column if value)
        for column in correlation_summary.columns.to_flat_index()
    ]
    summary = {
        "sample_count": args.count,
        "seed_count": len(seeds),
        "pair_ratios": pair_summary.to_dict(orient="records"),
        "correlations": correlation_summary.to_dict(orient="records"),
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
    parser.add_argument("--sensitivity", type=Path, default=DEFAULT_SENSITIVITY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--endpoint", type=int, default=5000)
    parser.add_argument("--switch-step", type=int, default=2000)
    parser.add_argument("--cache-start", type=int, default=100_288)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--data-batch-size", type=int, default=4)
    parser.add_argument("--model-batch-size", type=int, default=8)
    parser.add_argument("--probe-seed", type=int, default=20_260_730)
    parser.add_argument("--times", default="0.95,0.85,0.3")
    parser.add_argument("--amplitudes", default="0.5,1.0")
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count % args.data_batch_size:
        raise ValueError("count must be divisible by data_batch_size")
    if args.worker:
        run_worker(args)
    else:
        launch(args)


if __name__ == "__main__":
    main()
