"""Merge fixed-time strict-LPL mechanism audits for RAE and RAEv2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt


PANELS = (
    ("latent_relative_error_rms", "Relative clean-latent error"),
    ("flow_full_gradient_cosine", "cos(LPL gradient, Flow gradient)"),
    ("flow_projected_gradient_fraction", "Positive Flow-parallel fraction"),
    ("flow_projection_conflict", "Flow-conflict rate"),
    ("prediction_over_target_variance", "Prediction / target feature variance"),
    ("full_descent_raw_change", "Raw-feature first-order change under LPL"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rae", type=Path, required=True)
    parser.add_argument("--raev2", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_raw(path: Path, system: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["system"] = system
    if "prediction_target" not in frame:
        frame["prediction_target"] = "single" if system == "rae" else "full"
    return frame


def time_region(time: float) -> str:
    if time <= 0.5:
        return "low_t_le_0.5"
    if time <= 0.75:
        return "mid_0.5_to_0.75"
    return "high_t_gt_0.75"


def metric_uncertainty(
    raw: pd.DataFrame,
    group_keys: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, subset in raw.groupby(group_keys, sort=False):
        row = dict(zip(group_keys, keys, strict=True))
        row["sample_count"] = int(len(subset))
        for column, _ in PANELS:
            values = subset[column].dropna().to_numpy(dtype=np.float64)
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_sem"] = (
                float(values.std(ddof=1) / np.sqrt(len(values)))
                if len(values) > 1
                else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    raw = pd.concat(
        [
            load_raw(args.rae, "rae"),
            load_raw(args.raev2, "raev2"),
        ],
        ignore_index=True,
    )
    raw["time_region"] = raw["time"].map(time_region)
    group_keys = [
        "system",
        "prediction_target",
        "checkpoint",
        "state_key",
        "time",
        "noise_to_signal_ratio",
        "time_region",
    ]
    summary = (
        raw.groupby(group_keys, as_index=False)
        .mean(numeric_only=True)
        .sort_values(["system", "prediction_target", "time"])
    )
    uncertainty = metric_uncertainty(raw, group_keys).sort_values(
        ["system", "prediction_target", "time"]
    )
    region = (
        raw.groupby(
            [
                "system",
                "prediction_target",
                "checkpoint",
                "state_key",
                "time_region",
            ],
            as_index=False,
        )
        .mean(numeric_only=True)
        .sort_values(["system", "prediction_target", "time_region"])
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output_dir / "cross_system_raw.csv", index=False)
    summary.to_csv(output_dir / "cross_system_by_time.csv", index=False)
    uncertainty.to_csv(
        output_dir / "cross_system_by_time_uncertainty.csv",
        index=False,
    )
    region.to_csv(output_dir / "cross_system_by_region.csv", index=False)

    figure, axes = plt.subplots(2, 3, figsize=(19, 11))
    for axis, (column, title) in zip(axes.flat, PANELS, strict=True):
        for keys, subset in summary.groupby(
            ["system", "prediction_target", "checkpoint"], sort=False
        ):
            system, target, checkpoint = keys
            uncertainty_subset = uncertainty[
                (uncertainty["system"] == system)
                & (uncertainty["prediction_target"] == target)
                & (uncertainty["checkpoint"] == checkpoint)
            ]
            axis.errorbar(
                subset["time"],
                subset[column],
                yerr=uncertainty_subset[f"{column}_sem"],
                marker="o",
                linewidth=2.2,
                capsize=3,
                label=f"{system}:{target}:{checkpoint}",
            )
        axis.axvline(0.5, color="#6b7280", linestyle="--", linewidth=1)
        axis.axhline(0.0, color="#111827", linewidth=1, alpha=0.35)
        axis.set_xlabel("t (0 = clean, 1 = noise)")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=max(len(labels), 1))
    figure.suptitle("RAE vs RAEv2 fixed-time LPL mechanism")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(
        output_dir / "cross_system_mechanism.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    manifest = {
        "format_version": 1,
        "rae_input": str(args.rae.resolve()),
        "raev2_input": str(args.raev2.resolve()),
        "row_count": len(raw),
        "systems": sorted(raw["system"].unique().tolist()),
        "prediction_targets": sorted(
            raw["prediction_target"].unique().tolist()
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    display_columns = [
        "system",
        "prediction_target",
        "time_region",
        "latent_relative_error_rms",
        "flow_full_gradient_cosine",
        "flow_projected_gradient_fraction",
        "flow_projection_conflict",
        "prediction_over_target_variance",
        "full_descent_raw_change",
    ]
    print(region[display_columns].to_string(index=False))


if __name__ == "__main__":
    main()
