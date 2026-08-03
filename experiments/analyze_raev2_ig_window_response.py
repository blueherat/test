"""Analyze paired RAEv2 endpoint responses to small IG time-window interventions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_condition(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("conditions must use NAME=/path/to/run")
    return name.strip(), Path(path.strip())


def parse_signed_pair(value: str) -> tuple[str, Path, Path]:
    name, separator, paths = value.partition("=")
    positive, comma, negative = paths.partition(",")
    if not separator or not comma or not name.strip() or not positive.strip() or not negative.strip():
        raise argparse.ArgumentTypeError("signed pairs must use NAME=/positive/run,/negative/run")
    return name.strip(), Path(positive.strip()), Path(negative.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--condition", type=parse_condition, action="append", default=[])
    parser.add_argument("--signed-pair", type=parse_signed_pair, action="append", default=[])
    parser.add_argument("--parameterization-curve", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.condition and not args.signed_pair:
        parser.error("at least one --condition or --signed-pair is required")
    return args


def load_manifest(directory: Path) -> dict[str, Any]:
    return json.loads((directory / "manifest.json").read_text(encoding="utf-8"))


def load_single_scale_endpoints(directory: Path) -> tuple[np.ndarray, dict[str, Any], float]:
    directory = directory.expanduser().resolve()
    manifest = load_manifest(directory)
    scales = tuple(float(value) for value in manifest["scales"])
    if len(scales) != 1:
        raise ValueError(f"expected one scale in {directory}, found {scales}")
    scale = scales[0]
    scale_name = f"scale_s{scale:.6f}".replace(".", "p")
    samples = int(manifest["samples"])
    world_size = int(manifest["world_size"])
    endpoint: np.ndarray | None = None
    for rank in range(world_size):
        ids = np.arange(rank, samples, world_size, dtype=np.int64)
        path = directory / "latents" / f"{scale_name}_rank{rank:02d}.npy"
        local = np.load(path, allow_pickle=False).astype(np.float32)
        if len(local) != len(ids):
            raise RuntimeError(f"rank shard length mismatch in {path}")
        if endpoint is None:
            endpoint = np.empty((samples, *local.shape[1:]), dtype=np.float32)
        endpoint[ids] = local
    if endpoint is None:
        raise RuntimeError(f"no endpoint shards found in {directory}")
    return endpoint, manifest, scale


def validate_paired_protocol(reference: dict[str, Any], candidate: dict[str, Any]) -> None:
    keys = (
        "protocol",
        "samples",
        "seed",
        "world_size",
        "latent_size",
        "sampler_steps",
        "precision",
        "state_key",
        "checkpoint",
        "checkpoint_size",
    )
    mismatched = [key for key in keys if reference.get(key) != candidate.get(key)]
    if mismatched:
        raise ValueError(f"window runs are not paired; mismatched fields: {mismatched}")


def mean_full_path_curve(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path.expanduser().resolve())
        frames.append(frame[frame["branch"].eq("full_path")])
    combined = pd.concat(frames, ignore_index=True)
    return (
        combined.groupby("solver_index", as_index=False)
        .agg(
            time=("time", "first"),
            h_over_t_safe=("h_over_t_safe", "first"),
            raw_head_gap_rms=("raw_head_gap_rms", "mean"),
        )
        .sort_values("solver_index")
    )


def predicted_injected_energy(
    curve: pd.DataFrame,
    *,
    interval: tuple[float, float],
    gamma: float,
) -> tuple[int, float]:
    low, high = interval
    active = curve[curve["time"].between(low, high, inclusive="both")]
    unit_impulse = active["raw_head_gap_rms"] * active["h_over_t_safe"]
    return int(len(active)), float(gamma**2 * np.square(unit_impulse).sum())


def endpoint_response_metrics(candidate: np.ndarray, baseline: np.ndarray) -> dict[str, float]:
    if candidate.shape != baseline.shape:
        raise ValueError("candidate and baseline endpoint shapes differ")
    delta = (candidate.astype(np.float64) - baseline.astype(np.float64)).reshape(len(baseline), -1)
    rms = np.sqrt(np.mean(np.square(delta), axis=1))
    return {
        "endpoint_delta_rms_mean": float(rms.mean()),
        "endpoint_delta_rms_std": float(rms.std(ddof=1)),
        "endpoint_delta_rms_median": float(np.median(rms)),
        "endpoint_delta_rms_q25": float(np.quantile(rms, 0.25)),
        "endpoint_delta_rms_q75": float(np.quantile(rms, 0.75)),
    }


def signed_pair_metrics(
    positive: np.ndarray,
    negative: np.ndarray,
    baseline: np.ndarray,
    *,
    gamma_abs: float,
) -> dict[str, float]:
    if gamma_abs <= 0:
        raise ValueError("gamma_abs must be positive")
    if positive.shape != negative.shape or positive.shape != baseline.shape:
        raise ValueError("signed endpoints and baseline must have identical shapes")
    positive64 = positive.astype(np.float64)
    negative64 = negative.astype(np.float64)
    baseline64 = baseline.astype(np.float64)
    odd = 0.5 * (positive64 - negative64)
    even = 0.5 * (positive64 + negative64) - baseline64
    odd_rms = np.sqrt(np.mean(np.square(odd).reshape(len(odd), -1), axis=1))
    even_rms = np.sqrt(np.mean(np.square(even).reshape(len(even), -1), axis=1))
    return {
        "odd_endpoint_rms_mean": float(odd_rms.mean()),
        "even_endpoint_rms_mean": float(even_rms.mean()),
        "even_over_odd_mean": float(even_rms.mean() / max(odd_rms.mean(), 1e-30)),
        "central_response_per_gamma": float(odd_rms.mean() / gamma_abs),
    }


def analyze(
    baseline_dir: Path,
    conditions: list[tuple[str, Path]],
    parameterization_curves: list[Path],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    baseline, reference, baseline_scale = load_single_scale_endpoints(baseline_dir)
    if not np.isclose(baseline_scale, 1.0):
        raise ValueError("baseline directory must contain scale 1.0")
    curve = mean_full_path_curve(parameterization_curves)
    rows: list[dict[str, Any]] = []
    for name, directory in conditions:
        candidate, manifest, scale = load_single_scale_endpoints(directory)
        validate_paired_protocol(reference, manifest)
        interval = tuple(float(value) for value in manifest["ig_interval"])
        gamma = float(scale) - 1.0
        steps, injected_energy = predicted_injected_energy(
            curve, interval=interval, gamma=gamma
        )
        response = endpoint_response_metrics(candidate, baseline)
        rows.append(
            {
                "condition": name,
                "scale": float(scale),
                "gamma": gamma,
                "t_min": interval[0],
                "t_max": interval[1],
                "active_steps": steps,
                "predicted_injected_energy": injected_energy,
                "predicted_injected_norm": float(np.sqrt(injected_energy)),
                **response,
                "response_per_abs_gamma": response["endpoint_delta_rms_mean"]
                / max(abs(gamma), 1e-30),
                "response_per_injected_norm": response["endpoint_delta_rms_mean"]
                / max(float(np.sqrt(injected_energy)), 1e-30),
            }
        )
    metadata = {
        "format_version": 1,
        "scope": "paired_raev2_ig_time_window_endpoint_response",
        "baseline_dir": str(baseline_dir.expanduser().resolve()),
        "conditions": {
            name: str(path.expanduser().resolve()) for name, path in conditions
        },
        "parameterization_curves": [
            str(path.expanduser().resolve()) for path in parameterization_curves
        ],
        "samples": int(reference["samples"]),
        "seed": int(reference["seed"]),
        "world_size": int(reference["world_size"]),
        "checkpoint": reference["checkpoint"],
        "same_noise_and_labels": True,
    }
    return pd.DataFrame(rows), metadata


def analyze_signed_pairs(
    baseline_dir: Path,
    pairs: list[tuple[str, Path, Path]],
    parameterization_curves: list[Path],
) -> pd.DataFrame:
    baseline, reference, baseline_scale = load_single_scale_endpoints(baseline_dir)
    if not np.isclose(baseline_scale, 1.0):
        raise ValueError("baseline directory must contain scale 1.0")
    curve = mean_full_path_curve(parameterization_curves)
    rows: list[dict[str, Any]] = []
    for name, positive_dir, negative_dir in pairs:
        positive, positive_manifest, positive_scale = load_single_scale_endpoints(positive_dir)
        negative, negative_manifest, negative_scale = load_single_scale_endpoints(negative_dir)
        validate_paired_protocol(reference, positive_manifest)
        validate_paired_protocol(reference, negative_manifest)
        positive_interval = tuple(float(value) for value in positive_manifest["ig_interval"])
        negative_interval = tuple(float(value) for value in negative_manifest["ig_interval"])
        if positive_interval != negative_interval:
            raise ValueError(f"signed pair {name!r} uses different time intervals")
        positive_gamma = float(positive_scale) - 1.0
        negative_gamma = float(negative_scale) - 1.0
        if positive_gamma <= 0 or negative_gamma >= 0 or not np.isclose(
            positive_gamma, -negative_gamma, rtol=1e-5, atol=1e-8
        ):
            raise ValueError(f"signed pair {name!r} must use symmetric nonzero gamma")
        steps, energy = predicted_injected_energy(
            curve, interval=positive_interval, gamma=positive_gamma
        )
        rows.append(
            {
                "condition": name,
                "gamma_abs": positive_gamma,
                "t_min": positive_interval[0],
                "t_max": positive_interval[1],
                "active_steps": steps,
                "predicted_injected_energy": energy,
                "predicted_injected_norm": float(np.sqrt(energy)),
                **signed_pair_metrics(
                    positive, negative, baseline, gamma_abs=positive_gamma
                ),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["t_min", "gamma_abs"], ascending=[False, False])


def plot_results(frame: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    ordered = frame.sort_values(["t_min", "condition"], ascending=[False, True])
    labels = ordered["condition"].tolist()
    positions = np.arange(len(ordered))
    axes[0].bar(positions, ordered["endpoint_delta_rms_mean"], color="#2563EB")
    axes[0].set_xticks(positions, labels, rotation=35, ha="right")
    axes[0].set(title="Terminal endpoint response", ylabel="RMS(endpoint - full)")
    axes[1].bar(positions, ordered["response_per_injected_norm"], color="#0F766E")
    axes[1].set_xticks(positions, labels, rotation=35, ha="right")
    axes[1].set(title="Propagation after energy normalization", ylabel="endpoint RMS / injected norm")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    frame, metadata = analyze(
        args.baseline_dir, args.condition, args.parameterization_curve
    )
    signed = analyze_signed_pairs(
        args.baseline_dir, args.signed_pair, args.parameterization_curve
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "window_endpoint_response.csv", index=False)
    signed.to_csv(output_dir / "signed_window_response.csv", index=False)
    (output_dir / "manifest.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if not frame.empty:
        plot_results(frame, output_dir / "window_endpoint_response.png")
        print(frame.to_string(index=False))
    if not signed.empty:
        print(signed.to_string(index=False))


if __name__ == "__main__":
    main()
