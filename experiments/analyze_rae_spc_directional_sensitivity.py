"""Summarize the exploratory SPC directional-sensitivity control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_RESULTS = Path.home() / "data/eqvae/experiments/rae_spc_multiseed_v1"


def build_paired_sensitivity(table: pd.DataFrame) -> pd.DataFrame:
    required = {
        "seed",
        "condition",
        "checkpoint_step",
        "time",
        "direction",
        "total_gain",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"sensitivity table lacks columns: {sorted(missing)}")
    keys = ["seed", "checkpoint_step", "time", "direction"]
    if table.duplicated(keys + ["condition"]).any():
        raise ValueError("duplicate sensitivity rows")
    wide = table.pivot(index=keys, columns="condition", values="total_gain")
    for condition in ("static", "spc"):
        if condition not in wide:
            raise ValueError(f"missing condition: {condition}")
    paired = wide.reset_index()
    paired["spc_over_static"] = paired["spc"] / paired["static"]
    direction = paired.pivot(
        index=["seed", "checkpoint_step", "time"],
        columns="direction",
        values=["static", "spc", "spc_over_static"],
    )
    rows = []
    for (seed, step, time), row in direction.iterrows():
        result: dict[str, float | int] = {
            "seed": int(seed),
            "checkpoint_step": int(step),
            "time": float(time),
        }
        for metric in ("static", "spc", "spc_over_static"):
            for direction_name in ("guided", "control"):
                result[f"{direction_name}_{metric}"] = float(
                    row[(metric, direction_name)]
                )
        result["guided_over_control_static"] = (
            result["guided_static"] / result["control_static"]
        )
        result["guided_over_control_spc"] = (
            result["guided_spc"] / result["control_spc"]
        )
        result["selective_suppression_ratio"] = (
            result["guided_spc_over_static"]
            / result["control_spc_over_static"]
        )
        rows.append(result)
    return pd.DataFrame(rows).sort_values(
        ["seed", "checkpoint_step", "time"]
    ).reset_index(drop=True)


def summarize_exploratory(paired: pd.DataFrame) -> dict[str, object]:
    rows = []
    for (step, time), group in paired.groupby(["checkpoint_step", "time"]):
        rows.append(
            {
                "checkpoint_step": int(step),
                "time": float(time),
                "seed_count": int(len(group)),
                "mean_guided_spc_over_static": float(
                    group["guided_spc_over_static"].mean()
                ),
                "mean_control_spc_over_static": float(
                    group["control_spc_over_static"].mean()
                ),
                "mean_selective_suppression_ratio": float(
                    group["selective_suppression_ratio"].mean()
                ),
                "guided_ratio_below_control_count": int(
                    (
                        group["guided_spc_over_static"]
                        < group["control_spc_over_static"]
                    ).sum()
                ),
                "guided_ratio_at_most_half_count": int(
                    (group["guided_spc_over_static"] <= 0.5).sum()
                ),
                "control_ratio_within_25pct_count": int(
                    group["control_spc_over_static"].between(0.75, 1.25).sum()
                ),
            }
        )
    return {
        "status": "exploratory_not_preregistered",
        "rows": rows,
    }


def plot_sensitivity(paired: pd.DataFrame, output: Path, time: float = 0.85) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = paired[np.isclose(paired["time"], time)]
    steps = sorted(selected["checkpoint_step"].unique())
    figure, axes = plt.subplots(1, len(steps), figsize=(7 * len(steps), 5.5), squeeze=False)
    colors = {"static": "#4C78A8", "spc": "#E45756"}
    for axis, step in zip(axes[0], steps):
        group = selected[selected["checkpoint_step"] == step]
        positions = {"control": 0, "guided": 1}
        offsets = {"static": -0.10, "spc": 0.10}
        for direction in ("control", "guided"):
            for condition in ("static", "spc"):
                values = group[f"{direction}_{condition}"]
                x = positions[direction] + offsets[condition]
                axis.scatter(
                    np.full(len(values), x),
                    values,
                    color=colors[condition],
                    s=55,
                    alpha=0.85,
                    label=condition.upper()
                    if direction == "control"
                    else None,
                )
                axis.plot(
                    [x - 0.06, x + 0.06],
                    [values.mean(), values.mean()],
                    color="#222222",
                    linewidth=2,
                )
        axis.set_xticks([0, 1], ["Matched control", "Guided rank-16"])
        axis.set_ylabel("Prediction shift / input shift")
        axis.set_title(f"Step {step}, t={time:g}")
        axis.set_yscale("log")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)
    figure.suptitle("SPC selectively suppresses sensitivity to the guided subspace", fontsize=15)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()
    root = args.results.expanduser().resolve()
    source = (
        args.input.expanduser().resolve()
        if args.input
        else root / "directional_sensitivity/sensitivity_aggregate.csv"
    )
    output = root / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    paired = build_paired_sensitivity(pd.read_csv(source))
    summary = summarize_exploratory(paired)
    paired.to_csv(output / "spc_directional_sensitivity_pairs.csv", index=False)
    (output / "spc_directional_sensitivity_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    plot_sensitivity(
        paired, output / "spc_directional_sensitivity_t085.png", time=0.85
    )
    print(paired.to_string(index=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
