"""Combine and visualize endpoint decoder-feature distribution atlases."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PANEL_METRICS = (
    ("spatial_variance_ratio_gmean", "Within-image spatial variance ratio"),
    ("projected_covariance_trace_ratio", "Population covariance trace ratio"),
    ("projected_covariance_relative_error", "Population covariance relative error"),
    ("projected_normalized_frechet", "Projected normalized Frechet"),
)


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("atlas must be NAME=CSV")
    name, raw_path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("atlas name cannot be empty")
    return name, Path(raw_path).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas", action="append", type=parse_named_path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def branch_label(frame: pd.DataFrame) -> pd.Series:
    if "branch" not in frame and "checkpoint" not in frame:
        raise ValueError("atlas has neither branch nor checkpoint")
    branch = pd.Series(pd.NA, index=frame.index, dtype="object")
    if "branch" in frame:
        branch = branch.fillna(frame["branch"])
    if "checkpoint" in frame:
        branch = branch.fillna(frame["checkpoint"])
    branch = branch.fillna("unknown").astype(str)

    labels = frame["atlas_name"].astype(str) + ":" + branch
    if "branch_update" in frame:
        has_update = frame["branch_update"].notna()
        labels.loc[has_update] += (
            "@"
            + frame.loc[has_update, "branch_update"].astype(int).astype(str)
        )
    if "guidance_scale" in frame:
        has_guidance = frame["guidance_scale"].notna()
        labels.loc[has_guidance] += frame.loc[has_guidance, "guidance_scale"].map(
            lambda value: f",ig={float(value):g}"
        )
    return labels


def load_atlases(named_paths: list[tuple[str, Path]]) -> pd.DataFrame:
    names = [name for name, _ in named_paths]
    if len(names) != len(set(names)):
        raise ValueError("atlas names must be unique")
    frames = []
    for name, raw_path in named_paths:
        path = raw_path.resolve()
        frame = pd.read_csv(path)
        missing = {
            "noise_to_signal_ratio",
            "num_steps",
            "layer_fraction",
            *[metric for metric, _ in PANEL_METRICS],
        } - set(frame)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frame.insert(0, "atlas_path", str(path))
        frame.insert(0, "atlas_name", name)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["display_branch"] = branch_label(combined)
    return combined


def aggregate_atlas(
    combined: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = [metric for metric, _ in PANEL_METRICS]
    condition = (
        combined.groupby(
            [
                "display_branch",
                "noise_to_signal_ratio",
                "num_steps",
                "layer_fraction",
            ],
            as_index=False,
        )[metrics]
        .mean()
        .sort_values(
            [
                "display_branch",
                "noise_to_signal_ratio",
                "num_steps",
                "layer_fraction",
            ]
        )
    )
    layer = (
        condition.groupby(
            ["display_branch", "layer_fraction"],
            as_index=False,
        )[metrics]
        .mean()
        .sort_values(["display_branch", "layer_fraction"])
    )
    return condition, layer


def _heatmap_table(
    layer: pd.DataFrame,
    metric: str,
    branches: list[str],
    fractions: list[float],
) -> np.ndarray:
    pivot = layer.pivot(
        index="display_branch",
        columns="layer_fraction",
        values=metric,
    )
    return pivot.reindex(index=branches, columns=fractions).to_numpy(dtype=float)


def plot_layer_heatmaps(layer: pd.DataFrame, output: Path) -> None:
    branches = list(dict.fromkeys(layer["display_branch"]))
    fractions = sorted(layer["layer_fraction"].unique())
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(max(13, 1.45 * len(fractions)), max(9, 0.7 * len(branches) + 5)),
        constrained_layout=True,
    )
    for axis, (metric, title) in zip(axes.flat, PANEL_METRICS, strict=True):
        values = _heatmap_table(layer, metric, branches, fractions)
        image = axis.imshow(values, aspect="auto", cmap="viridis")
        axis.set_xticks(range(len(fractions)))
        axis.set_xticklabels([f"{value:g}" for value in fractions])
        axis.set_yticks(range(len(branches)))
        axis.set_yticklabels(branches, fontsize=8)
        axis.set_xlabel("decoder layer fraction")
        axis.set_title(title)
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = values[row, column]
                if np.isfinite(value):
                    axis.text(
                        column,
                        row,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if value > np.nanmedian(values) else "black",
                    )
        figure.colorbar(image, ax=axis, shrink=0.78)
    figure.savefig(output, dpi=190)
    plt.close(figure)


def plot_rollout_curves(condition: pd.DataFrame, output: Path) -> None:
    averaged = (
        condition.groupby(
            ["display_branch", "noise_to_signal_ratio", "num_steps"],
            as_index=False,
        )[
            [
                "spatial_variance_ratio_gmean",
                "projected_covariance_trace_ratio",
            ]
        ]
        .mean()
        .sort_values(["display_branch", "noise_to_signal_ratio", "num_steps"])
    )
    ratios = sorted(averaged["noise_to_signal_ratio"].unique())
    figure, axes = plt.subplots(
        len(ratios),
        2,
        figsize=(15, 4.5 * len(ratios)),
        squeeze=False,
        constrained_layout=True,
    )
    for row, ratio in enumerate(ratios):
        local = averaged[averaged["noise_to_signal_ratio"] == ratio]
        for branch, frame in local.groupby("display_branch"):
            for column, metric in enumerate(
                (
                    "spatial_variance_ratio_gmean",
                    "projected_covariance_trace_ratio",
                )
            ):
                axes[row, column].plot(
                    frame["num_steps"],
                    frame[metric],
                    marker="o",
                    label=branch,
                )
        for column, title in enumerate(
            ("Spatial variance ratio", "Population covariance trace ratio")
        ):
            axis = axes[row, column]
            axis.axhline(1.0, color="black", linewidth=1, linestyle="--")
            axis.set_xscale("log", base=2)
            axis.set_xticks(sorted(local["num_steps"].unique()))
            axis.set_xticklabels(
                [str(int(value)) for value in sorted(local["num_steps"].unique())]
            )
            axis.set_title(f"{title}, noise ratio={ratio:g}")
            axis.set_xlabel("recursive endpoint queries")
            axis.grid(alpha=0.25)
    axes[0, -1].legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.savefig(output, dpi=190)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    combined = load_atlases(args.atlas)
    condition, layer = aggregate_atlas(combined)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_dir / "endpoint_atlas_combined.csv", index=False)
    condition.to_csv(output_dir / "endpoint_atlas_condition_summary.csv", index=False)
    layer.to_csv(output_dir / "endpoint_atlas_layer_summary.csv", index=False)
    plot_layer_heatmaps(layer, output_dir / "endpoint_atlas_layer_heatmaps.png")
    plot_rollout_curves(condition, output_dir / "endpoint_atlas_rollout_curves.png")
    print(layer.to_string(index=False))
    print(output_dir)


if __name__ == "__main__":
    main()
