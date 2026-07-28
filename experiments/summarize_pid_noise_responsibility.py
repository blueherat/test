"""Summarize fixed-step latent responsibility screens across PiD checkpoints.

The script consumes only the per-sample CSV files written by
``run_pid_noise_responsibility.py``.  It deliberately keeps shuffled-latent
ablation as the primary signal; null-condition results are not used for gates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_FILES = (
    "paired_rows.csv",
    "identity_controls.csv",
    "batch_order_controls.csv",
    "provenance.json",
)


def load_screen(root: Path, models: Iterable[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load model screens and return paired rows plus numerical controls."""

    paired_tables: list[pd.DataFrame] = []
    controls: list[dict[str, float | int | str | bool]] = []
    for model in models:
        model_root = root / model
        missing = [name for name in REQUIRED_FILES if not (model_root / name).is_file()]
        if missing:
            raise FileNotFoundError(f"{model_root} is missing {missing}")

        paired = pd.read_csv(model_root / "paired_rows.csv")
        identity = pd.read_csv(model_root / "identity_controls.csv")
        order = pd.read_csv(model_root / "batch_order_controls.csv")
        provenance = json.loads((model_root / "provenance.json").read_text())
        paired.insert(0, "model", model)
        paired_tables.append(paired)
        identity_max = float(
            identity[["absolute_rms_max", "relative_rms_max"]].to_numpy().max()
        )
        order_max = float(order["max_absolute_difference"].max())
        controls.append(
            {
                "model": model,
                "images": int(paired["sample_index"].nunique()),
                "seeds": int(paired["seed"].nunique()),
                "checkpoint_sha256": provenance["checkpoint_sha256"],
                "identity_max": identity_max,
                "batch_order_max": order_max,
                "controls_exact": identity_max == 0.0 and order_max == 0.0,
            }
        )
    if not paired_tables:
        raise ValueError("at least one model is required")
    return pd.concat(paired_tables, ignore_index=True), pd.DataFrame(controls)


def profile_table(paired: pd.DataFrame) -> pd.DataFrame:
    """Aggregate shuffled-condition effects without hiding paired sign changes."""

    required = {"model", "mode", "seed", "sample_index", "timestep", "delta_shuffle"}
    missing = required - set(paired)
    if missing:
        raise ValueError(f"paired rows are missing {sorted(missing)}")
    table = paired.copy()
    table["timestep"] = table["timestep"].round(6)
    records: list[dict[str, float | int | str | bool]] = []
    for (model, mode, timestep), frame in table.groupby(
        ["model", "mode", "timestep"], sort=True
    ):
        values = frame["delta_shuffle"].to_numpy(dtype=np.float64)
        seed_means = frame.groupby("seed")["delta_shuffle"].mean()
        records.append(
            {
                "model": model,
                "mode": mode,
                "timestep": float(timestep),
                "count": int(len(frame)),
                "delta_shuffle_mean": float(values.mean()),
                "delta_shuffle_median": float(np.median(values)),
                "delta_shuffle_positive_rate": float(np.mean(values > 0.0)),
                "all_seed_means_positive": bool((seed_means > 0.0).all()),
            }
        )
    result = pd.DataFrame(records)
    result["delta_relative_to_first"] = np.nan
    for (_, _), index in result.groupby(["model", "mode"]).groups.items():
        group = result.loc[index]
        first = float(group.loc[group["timestep"].idxmax(), "delta_shuffle_mean"])
        if first != 0.0:
            result.loc[index, "delta_relative_to_first"] = (
                group["delta_shuffle_mean"] / first
            )
    return result


def model_gate_table(paired: pd.DataFrame, controls: pd.DataFrame) -> pd.DataFrame:
    """Evaluate only pre-registered, non-null automatic gates per model."""

    records: list[dict[str, float | str | bool]] = []
    for (model, mode), frame in paired.groupby(["model", "mode"], sort=True):
        seed_means = frame.groupby("seed")["delta_shuffle"].mean()
        time_means = frame.groupby(frame["timestep"].round(6))["delta_shuffle"].mean()
        highest_t = float(time_means.index.max())
        lower_times = time_means.index[time_means.index < highest_t]
        second_t = float(lower_times.max()) if len(lower_times) else float("nan")
        first = float(time_means.loc[highest_t])
        second = float(time_means.loc[second_t]) if len(lower_times) else float("nan")
        control_row = controls.loc[controls.model == model].iloc[0]
        records.append(
            {
                "model": model,
                "mode": mode,
                "overall_delta_shuffle": float(frame["delta_shuffle"].mean()),
                "all_seed_means_positive": bool((seed_means > 0.0).all()),
                "highest_t": highest_t,
                "highest_t_delta": first,
                "second_t": second_t,
                "second_t_delta": second,
                "second_to_first_ratio": second / first if first != 0.0 else float("nan"),
                "controls_exact": bool(control_row.controls_exact),
                "automatic_gate_pass": bool((seed_means > 0.0).all() and control_row.controls_exact),
            }
        )
    return pd.DataFrame(records)


def plot_profiles(
    profile: pd.DataFrame,
    output: Path,
    *,
    metric: str = "delta_shuffle_mean",
) -> None:
    """Plot raw or within-checkpoint-normalized responsibility profiles."""

    if metric not in profile:
        raise ValueError(f"profile does not contain {metric}")

    modes = ("teacher_forced", "real_rollout")
    models = list(dict.fromkeys(profile["model"].tolist()))
    figure, axes = plt.subplots(1, len(modes), figsize=(13, 5), sharey=False)
    if len(modes) == 1:
        axes = [axes]
    for axis, mode in zip(axes, modes):
        subset = profile[profile["mode"] == mode]
        for model in models:
            rows = subset[subset["model"] == model].sort_values("timestep", ascending=False)
            axis.plot(
                np.arange(len(rows)),
                rows[metric],
                marker="o",
                linewidth=2,
                label=model,
            )
        labels = (
            subset[["timestep"]]
            .drop_duplicates()
            .sort_values("timestep", ascending=False)["timestep"]
            .map(lambda value: f"{value:.3f}")
            .tolist()
        )
        axis.set_xticks(np.arange(len(labels)), labels)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xlabel("PiD fixed timestep t (noise -> image)")
        axis.set_ylabel(
            "Relative condition responsibility"
            if metric == "delta_relative_to_first"
            else "MSE(shuffled latent) - MSE(real latent)"
        )
        axis.set_title(mode.replace("_", " "))
        axis.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)), frameon=False)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paired, controls = load_screen(args.root, args.models)
    profile = profile_table(paired)
    gates = model_gate_table(paired, controls)
    args.output.mkdir(parents=True, exist_ok=True)
    controls.to_csv(args.output / "controls.csv", index=False)
    profile.to_csv(args.output / "profiles.csv", index=False)
    gates.to_csv(args.output / "model_gates.csv", index=False)
    plot_profiles(profile, args.output / "responsibility_profiles.png")
    plot_profiles(
        profile,
        args.output / "responsibility_profiles_normalized.png",
        metric="delta_relative_to_first",
    )
    print("\nControls")
    print(controls.to_string(index=False))
    print("\nModel gates")
    print(gates.to_string(index=False))
    print(f"\nResults: {args.output}")


if __name__ == "__main__":
    main()
