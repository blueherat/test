"""Summarize multi-seed, multi-gamma RAEv2 single-step pulse audits.

The scalar endpoint response can look linear even when the response direction
changes with intervention size.  This script therefore compares the complete
central-difference endpoint vectors obtained at two gamma values, in addition
to reporting response magnitude, nonlinearity, and sample-level tail ratios.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_raev2_ig_impulse_response import (
    Intervention,
    _load_condition,
    _load_small_shards,
    bootstrap_mean_interval,
)


PROTOCOL = "raev2_ig_pulse_validation_summary_v1"


@dataclass(frozen=True)
class PairScalars:
    derivative_rms: np.ndarray
    even_over_odd: np.ndarray
    propagation_gain: np.ndarray
    derivative: np.ndarray


def build_summary_manifest(
    run_dirs: list[Path],
    *,
    bootstrap_repeats: int,
    seed: int,
    rows: int,
) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "status": "complete",
        "run_dirs": [str(path.expanduser().resolve()) for path in run_dirs],
        "bootstrap_repeats": bootstrap_repeats,
        "seed": seed,
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260814)
    return parser.parse_args()


def sample_rms(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).reshape(len(values), -1)
    return np.sqrt(np.mean(np.square(flat), axis=1))


def sample_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_flat = np.asarray(left, dtype=np.float64).reshape(len(left), -1)
    right_flat = np.asarray(right, dtype=np.float64).reshape(len(right), -1)
    numerator = np.sum(left_flat * right_flat, axis=1)
    denominator = np.linalg.norm(left_flat, axis=1) * np.linalg.norm(right_flat, axis=1)
    return numerator / np.maximum(denominator, 1e-30)


def central_derivative(positive: np.ndarray, negative: np.ndarray, gamma: float) -> np.ndarray:
    if gamma <= 0 or positive.shape != negative.shape:
        raise ValueError("positive gamma and matching endpoint arrays are required")
    return (np.asarray(positive, dtype=np.float32) - np.asarray(negative, dtype=np.float32)) / (
        2.0 * float(gamma)
    )


def pair_scalars(
    baseline: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    *,
    gamma: float,
    unit_injected_norm: np.ndarray,
) -> PairScalars:
    derivative = central_derivative(positive, negative, gamma)
    derivative_rms = sample_rms(derivative)
    odd_rms = float(gamma) * derivative_rms
    even = 0.5 * (
        np.asarray(positive, dtype=np.float64) + np.asarray(negative, dtype=np.float64)
    ) - np.asarray(baseline, dtype=np.float64)
    even_rms = sample_rms(even)
    unit_norm = np.asarray(unit_injected_norm, dtype=np.float64).reshape(-1)
    if len(unit_norm) != len(derivative_rms):
        raise ValueError("injection statistics do not match endpoint rows")
    return PairScalars(
        derivative_rms=derivative_rms,
        even_over_odd=even_rms / np.maximum(odd_rms, 1e-30),
        propagation_gain=derivative_rms / np.maximum(unit_norm, 1e-30),
        derivative=derivative,
    )


def summarize_values(values: np.ndarray, *, repeats: int, seed: int) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    low, high = bootstrap_mean_interval(values, repeats=repeats, seed=seed)
    median = float(np.median(values))
    q95 = float(np.quantile(values, 0.95))
    return {
        "mean": float(values.mean()),
        "mean_ci_low": low,
        "mean_ci_high": high,
        "median": median,
        "q90": float(np.quantile(values, 0.90)),
        "q95": q95,
        "maximum": float(values.max()),
        "q95_over_median": q95 / max(median, 1e-30),
    }


def load_run(run_dir: Path) -> tuple[dict[str, Any], tuple[Intervention, ...]]:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError(f"run is not complete: {run_dir}")
    interventions = tuple(Intervention(**row) for row in manifest["interventions"])
    return manifest, interventions


def find_pair_indices(
    interventions: tuple[Intervention, ...], step: int, gamma: float
) -> tuple[int, int]:
    matches = [
        (index, item)
        for index, item in enumerate(interventions)
        if item.family == "pulse"
        and item.start_step == int(step)
        and np.isclose(abs(item.gamma), float(gamma), rtol=0.0, atol=1e-12)
    ]
    positive = [index for index, item in matches if item.gamma > 0]
    negative = [index for index, item in matches if item.gamma < 0]
    if len(positive) != 1 or len(negative) != 1:
        raise RuntimeError(f"cannot resolve pulse pair for step={step}, gamma={gamma}")
    return positive[0], negative[0]


def analyze_run(
    run_dir: Path,
    *,
    repeats: int,
    bootstrap_seed: int,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, np.ndarray]]]:
    manifest, interventions = load_run(run_dir)
    samples = int(manifest["samples"])
    world_size = int(manifest["world_size"])
    run_seed = int(manifest["seed"])
    baseline = _load_condition(run_dir, condition_index=0, samples=samples, world_size=world_size)
    injection = _load_small_shards(
        run_dir,
        filename="injection_stats_rank{rank:02d}.npy",
        samples=samples,
        world_size=world_size,
    )
    pulse_items = [item for item in interventions if item.family == "pulse"]
    steps = sorted({item.start_step for item in pulse_items})
    gammas = sorted({abs(float(item.gamma)) for item in pulse_items})
    if len(gammas) != 2:
        raise RuntimeError(f"expected exactly two pulse gammas, got {gammas}")

    rows: list[dict[str, Any]] = []
    pooled: dict[int, dict[str, np.ndarray]] = {}
    for step_index, step in enumerate(steps):
        by_gamma: dict[float, PairScalars] = {}
        for gamma in gammas:
            positive_index, negative_index = find_pair_indices(interventions, step, gamma)
            positive = _load_condition(
                run_dir, condition_index=positive_index, samples=samples, world_size=world_size
            )
            negative = _load_condition(
                run_dir, condition_index=negative_index, samples=samples, world_size=world_size
            )
            unit_energy = 0.5 * (
                injection[:, positive_index, 0] + injection[:, negative_index, 0]
            )
            by_gamma[gamma] = pair_scalars(
                baseline,
                positive,
                negative,
                gamma=gamma,
                unit_injected_norm=np.sqrt(np.maximum(unit_energy, 0.0)),
            )

        small, large = (by_gamma[gammas[0]], by_gamma[gammas[1]])
        scale = 0.5 * (small.derivative_rms + large.derivative_rms)
        linearity_error = sample_rms(small.derivative - large.derivative) / np.maximum(scale, 1e-30)
        derivative_cosine = sample_cosine(small.derivative, large.derivative)
        amplitude_ratio = small.derivative_rms / np.maximum(large.derivative_rms, 1e-30)
        pooled[step] = {
            "linearity_error": linearity_error,
            "derivative_cosine": derivative_cosine,
            "amplitude_ratio": amplitude_ratio,
        }
        for gamma_index, gamma in enumerate(gammas):
            scalars = by_gamma[gamma]
            pooled[step][f"response_g{gamma}"] = scalars.derivative_rms
            pooled[step][f"gain_g{gamma}"] = scalars.propagation_gain
            pooled[step][f"even_g{gamma}"] = scalars.even_over_odd
            response_summary = summarize_values(
                scalars.derivative_rms,
                repeats=repeats,
                seed=bootstrap_seed + 1009 * step_index + gamma_index,
            )
            gain_summary = summarize_values(
                scalars.propagation_gain,
                repeats=repeats,
                seed=bootstrap_seed + 2003 * step_index + gamma_index,
            )
            rows.append(
                {
                    "scope": "run",
                    "run_dir": str(run_dir),
                    "run_seed": run_seed,
                    "samples": samples,
                    "step": step,
                    "time": float(manifest["solver_grid"][step]),
                    "gamma": gamma,
                    **{f"response_{key}": value for key, value in response_summary.items()},
                    **{f"gain_{key}": value for key, value in gain_summary.items()},
                    "even_over_odd_mean": float(scalars.even_over_odd.mean()),
                    "cross_gamma_linearity_error_mean": float(linearity_error.mean()),
                    "cross_gamma_derivative_cosine_mean": float(derivative_cosine.mean()),
                    "cross_gamma_amplitude_ratio_mean": float(amplitude_ratio.mean()),
                }
            )
    return rows, pooled


def main() -> None:
    args = parse_args()
    if args.bootstrap_repeats <= 0:
        raise ValueError("bootstrap repeats must be positive")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    pooled_runs: list[dict[int, dict[str, np.ndarray]]] = []
    manifests = []
    for index, path in enumerate(args.run_dirs):
        run_dir = path.expanduser().resolve()
        manifest, _ = load_run(run_dir)
        manifests.append(manifest)
        rows, pooled = analyze_run(
            run_dir,
            repeats=args.bootstrap_repeats,
            bootstrap_seed=args.seed + 100_003 * index,
        )
        all_rows.extend(rows)
        pooled_runs.append(pooled)

    steps = sorted(set.intersection(*(set(run) for run in pooled_runs)))
    gammas = sorted(
        {
            float(row["gamma"])
            for row in all_rows
            if row["scope"] == "run"
        }
    )
    for step_index, step in enumerate(steps):
        merged = {
            key: np.concatenate([run[step][key] for run in pooled_runs])
            for key in pooled_runs[0][step]
        }
        for gamma_index, gamma in enumerate(gammas):
            response = merged[f"response_g{gamma}"]
            gain = merged[f"gain_g{gamma}"]
            response_summary = summarize_values(
                response,
                repeats=args.bootstrap_repeats,
                seed=args.seed + 3001 * step_index + gamma_index,
            )
            gain_summary = summarize_values(
                gain,
                repeats=args.bootstrap_repeats,
                seed=args.seed + 4001 * step_index + gamma_index,
            )
            all_rows.append(
                {
                    "scope": "pooled",
                    "run_dir": "",
                    "run_seed": -1,
                    "samples": len(response),
                    "step": step,
                    "time": float(manifests[0]["solver_grid"][step]),
                    "gamma": gamma,
                    **{f"response_{key}": value for key, value in response_summary.items()},
                    **{f"gain_{key}": value for key, value in gain_summary.items()},
                    "even_over_odd_mean": float(merged[f"even_g{gamma}"].mean()),
                    "cross_gamma_linearity_error_mean": float(merged["linearity_error"].mean()),
                    "cross_gamma_derivative_cosine_mean": float(merged["derivative_cosine"].mean()),
                    "cross_gamma_amplitude_ratio_mean": float(merged["amplitude_ratio"].mean()),
                }
            )

    frame = pd.DataFrame(all_rows).sort_values(["scope", "run_seed", "step", "gamma"])
    frame.to_csv(output_dir / "pulse_validation_summary.csv", index=False)
    pooled_frame = frame[frame["scope"].eq("pooled")]
    figure, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    for gamma, part in pooled_frame.groupby("gamma"):
        axes[0].plot(part["time"], part["gain_mean"], "o-", label=f"gamma={gamma:g}")
        axes[1].plot(
            part["time"], part["response_q95_over_median"], "o-", label=f"gamma={gamma:g}"
        )
    linearity = pooled_frame.groupby("step", as_index=False).first()
    axes[2].plot(
        linearity["time"], linearity["cross_gamma_linearity_error_mean"], "o-",
        label="relative derivative error",
    )
    axes[2].plot(
        linearity["time"], 1.0 - linearity["cross_gamma_derivative_cosine_mean"], "s-",
        label="1 - derivative cosine",
    )
    titles = ("Propagation gain", "Tail ratio", "Cross-gamma nonlinearity")
    ylabels = ("endpoint derivative / injected norm", "q95 / median", "relative value")
    for axis, title, ylabel in zip(axes, titles, ylabels):
        axis.invert_xaxis()
        axis.set(title=title, xlabel="solver time t", ylabel=ylabel)
        axis.grid(alpha=0.2)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "pulse_validation_summary.png", dpi=180)
    plt.close(figure)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            build_summary_manifest(
                args.run_dirs,
                bootstrap_repeats=args.bootstrap_repeats,
                seed=args.seed,
                rows=len(frame),
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(pooled_frame.to_string(index=False))


if __name__ == "__main__":
    main()
