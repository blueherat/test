"""Combine independent RAEv2 local-guideability audits across random seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_audits(input_dirs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    directions: list[pd.DataFrame] = []
    scales: list[pd.DataFrame] = []
    manifests: list[dict[str, Any]] = []
    for input_dir in input_dirs:
        path = input_dir.expanduser().resolve()
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        direction = pd.read_csv(path / "direction_summary.csv")
        scale = pd.read_csv(path / "scale_sweep_summary.csv")
        seed = int(manifest["seed"])
        direction.insert(0, "seed", seed)
        scale.insert(0, "seed", seed)
        directions.append(direction)
        scales.append(scale)
        manifests.append(manifest)
    return pd.concat(directions, ignore_index=True), pd.concat(scales, ignore_index=True), manifests


def validate_manifests(manifests: list[dict[str, Any]]) -> None:
    if len(manifests) < 2:
        raise ValueError("at least two independent seed audits are required")
    seeds = [int(item["seed"]) for item in manifests]
    if len(set(seeds)) != len(seeds):
        raise ValueError("audit seeds must be unique")
    keys = (
        "checkpoint_sha256",
        "state_keys",
        "data_path",
        "split",
        "samples",
        "times",
        "scales",
        "precision",
        "world_size",
    )
    reference = manifests[0]
    for manifest in manifests[1:]:
        mismatched = [key for key in keys if manifest.get(key) != reference.get(key)]
        if mismatched:
            raise ValueError(f"audit manifests differ in required fields: {mismatched}")


def summarize_directions(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (state_key, time), frame in per_seed.groupby(["state_key", "time"], sort=True):
        gamma = frame["gamma_population"].to_numpy(dtype=np.float64)
        gain = frame["oracle_relative_gain_mean"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "state_key": state_key,
                "time": float(time),
                "seeds": int(len(frame)),
                "gamma_mean": float(gamma.mean()),
                "gamma_std": float(gamma.std(ddof=1)),
                "gamma_min": float(gamma.min()),
                "gamma_max": float(gamma.max()),
                "negative_gamma_seeds": int((gamma < 0).sum()),
                "positive_alignment_fraction_mean": float(
                    frame["positive_alignment_fraction"].mean()
                ),
                "full_mse_mean": float(frame["full_mse_mean"].mean()),
                "base_mse_mean": float(frame["base_mse_mean"].mean()),
                "base_over_full_mse_mean": float(frame["base_over_full_mse"].mean()),
                "oracle_relative_gain_mean": float(gain.mean()),
                "nearest_solver_step": int(frame["nearest_solver_step"].iloc[0]),
                "nearest_solver_time": float(frame["nearest_solver_time"].iloc[0]),
                "nearest_h_over_t": float(frame["nearest_h_over_t"].iloc[0]),
            }
        )
    return pd.DataFrame(rows).sort_values(["state_key", "time"])


def summarize_scales(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (state_key, time, scale), frame in per_seed.groupby(
        ["state_key", "time", "scale"], sort=True
    ):
        gain = frame["gain_over_full_mean"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "state_key": state_key,
                "time": float(time),
                "scale": float(scale),
                "seeds": int(len(frame)),
                "gain_over_full_mean": float(gain.mean()),
                "gain_over_full_std": float(gain.std(ddof=1)),
                "gain_over_full_min": float(gain.min()),
                "gain_over_full_max": float(gain.max()),
                "negative_gain_seeds": int((gain < 0).sum()),
                "positive_gain_fraction_mean": float(frame["positive_gain_fraction"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["state_key", "time", "scale"])


def plot_cross_seed(direction: pd.DataFrame, scales: pd.DataFrame, output: Path) -> None:
    state_keys = tuple(direction["state_key"].drop_duplicates())
    figure, axes = plt.subplots(2, len(state_keys), figsize=(8 * len(state_keys), 10), squeeze=False)
    for column, state_key in enumerate(state_keys):
        local = direction[direction["state_key"].eq(state_key)].sort_values("time")
        axis = axes[0, column]
        axis.errorbar(
            local["time"],
            local["gamma_mean"],
            yerr=local["gamma_std"],
            fmt="o-",
            capsize=4,
            label="seed mean +/- std",
        )
        axis.axhline(0.0, color="#222222", linewidth=1)
        axis.axhline(0.78, color="#b42318", linestyle="--", label="official gamma=0.78")
        axis.set(title=f"{state_key}: local optimal gamma", xlabel="noise time t", ylabel="gamma*")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)

        sweep = scales[scales["state_key"].eq(state_key)].pivot(
            index="time", columns="scale", values="gain_over_full_mean"
        )
        bound = max(float(np.abs(sweep.to_numpy()).max()), 1e-12)
        image = axes[1, column].imshow(
            sweep.to_numpy(), aspect="auto", origin="lower", cmap="RdBu", vmin=-bound, vmax=bound
        )
        axes[1, column].set_xticks(range(len(sweep.columns)), [f"{value:g}" for value in sweep.columns])
        axes[1, column].set_yticks(range(len(sweep.index)), [f"{value:g}" for value in sweep.index])
        axes[1, column].set(title=f"{state_key}: local MSE gain", xlabel="code scale s", ylabel="noise time t")
        figure.colorbar(image, ax=axes[1, column], label="gain over full")
    figure.suptitle("RAEv2 Dual-Head Local Guideability: Cross-Seed Audit", fontsize=16)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    per_seed_direction, per_seed_scale, manifests = load_audits(args.input_dir)
    validate_manifests(manifests)
    direction = summarize_directions(per_seed_direction)
    scales = summarize_scales(per_seed_scale)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed_direction.to_csv(output_dir / "direction_per_seed.csv", index=False)
    per_seed_scale.to_csv(output_dir / "scale_per_seed.csv", index=False)
    direction.to_csv(output_dir / "direction_cross_seed.csv", index=False)
    scales.to_csv(output_dir / "scale_cross_seed.csv", index=False)
    plot_cross_seed(direction, scales, output_dir / "direction_cross_seed.png")
    summary = {
        "format_version": 1,
        "scope": "cross_seed_raev2_dual_head_local_guideability",
        "input_dirs": [str(path.expanduser().resolve()) for path in args.input_dir],
        "seeds": [int(item["seed"]) for item in manifests],
        "checkpoint_sha256": manifests[0]["checkpoint_sha256"],
        "samples_per_seed": int(manifests[0]["samples"]),
        "times": manifests[0]["times"],
        "scales": manifests[0]["scales"],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(direction.to_string(index=False))
    print(scales.to_string(index=False))


if __name__ == "__main__":
    main()
