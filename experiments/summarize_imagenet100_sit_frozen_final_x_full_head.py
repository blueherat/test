#!/usr/bin/env python3
"""Package the frozen-v800 final-block full x-head experiment for Git."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

try:
    from experiments.summarize_imagenet100_sit_frozen_internal_v_head import (
        main as package_auxiliary_head,
    )
except ModuleNotFoundError:
    from summarize_imagenet100_sit_frozen_internal_v_head import (
        main as package_auxiliary_head,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_ROOT = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/"
    "sit-s-2_v800-ema_frozen-final-x-fullhead-depth12_seed0"
)
DEFAULT_FID_ROOT = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/"
    "fid1k_v800_frozen_final_x_fullhead_depth12_step50000_ema"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "docs/data/imagenet100_sit_frozen_final_x_full_head_50k"
)
TINY_FINAL_SUMMARY = (
    REPO_ROOT / "docs/data/imagenet100_sit_frozen_v_clean_head_50k/summary.json"
)
DEPTH8_FULL_SUMMARY = (
    REPO_ROOT / "docs/data/imagenet100_sit_frozen_internal_x_head_50k/summary.json"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def comparison_rows(
    tiny: dict[str, Any],
    depth8: dict[str, Any],
    depth12: dict[str, Any],
) -> list[dict[str, Any]]:
    tiny_best = tiny["fid1k"]["best_positive_extrapolation"]
    depth8_best = depth8["fid1k"]["best_positive_extrapolation"]
    depth12_best = depth12["fid1k"]["best_positive_extrapolation"]
    tiny_baseline = tiny["fid1k"]["baseline"]
    depth8_baseline = depth8["fid1k"]["baseline"]
    depth12_baseline = depth12["fid1k"]["baseline"]
    return [
        {
            "variant": "final_tiny_linear",
            "readout_depth": 12,
            "head_architecture": "shared source AdaLN + clean linear projection",
            "trainable_parameters": tiny["experiment"]["trainable_parameter_count"],
            "ema_native_x_mse": tiny["final_validation"]["ema"]["clean_mse"],
            "ema_velocity_mse": tiny["final_validation"]["ema"][
                "clean_derived_velocity_mse"
            ],
            "full_auxiliary_gap_rms": "",
            "auxiliary_only_fid1k": tiny["fid1k"]["clean_head_only"]["fid"],
            "baseline_fid1k": tiny_baseline["fid"],
            "best_gamma": tiny_best["gamma"],
            "best_extrapolation_fid1k": tiny_best["fid"],
            "fid_improvement": float(tiny_baseline["fid"]) - float(tiny_best["fid"]),
        },
        {
            "variant": "depth8_full_finallayer",
            "readout_depth": depth8["experiment"]["internal_depth"],
            "head_architecture": "independent AdaLN FinalLayer",
            "trainable_parameters": depth8["experiment"]["trainable_parameter_count"],
            "ema_native_x_mse": depth8["final_validation"]["ema"][
                "internal_native_mse"
            ],
            "ema_velocity_mse": depth8["final_validation"]["ema"][
                "internal_velocity_mse"
            ],
            "full_auxiliary_gap_rms": depth8["final_validation"]["ema"][
                "full_internal_gap_rms"
            ],
            "auxiliary_only_fid1k": depth8["fid1k"]["internal_head_only"]["fid"],
            "baseline_fid1k": depth8_baseline["fid"],
            "best_gamma": depth8_best["gamma"],
            "best_extrapolation_fid1k": depth8_best["fid"],
            "fid_improvement": float(depth8_baseline["fid"])
            - float(depth8_best["fid"]),
        },
        {
            "variant": "depth12_full_finallayer",
            "readout_depth": depth12["experiment"]["internal_depth"],
            "head_architecture": "independent AdaLN FinalLayer",
            "trainable_parameters": depth12["experiment"]["trainable_parameter_count"],
            "ema_native_x_mse": depth12["final_validation"]["ema"][
                "internal_native_mse"
            ],
            "ema_velocity_mse": depth12["final_validation"]["ema"][
                "internal_velocity_mse"
            ],
            "full_auxiliary_gap_rms": depth12["final_validation"]["ema"][
                "full_internal_gap_rms"
            ],
            "auxiliary_only_fid1k": depth12["fid1k"]["internal_head_only"]["fid"],
            "baseline_fid1k": depth12_baseline["fid"],
            "best_gamma": depth12_best["gamma"],
            "best_extrapolation_fid1k": depth12_best["fid"],
            "fid_improvement": float(depth12_baseline["fid"])
            - float(depth12_best["fid"]),
        },
    ]


def plot_comparison(rows: list[dict[str, Any]], output_path: Path) -> None:
    labels = ("Final tiny", "Depth-8 full", "Depth-12 full")
    colors = ("#8c8c8c", "#d36b3d", "#3178a8")
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
    panels = (
        ("ema_velocity_mse", "Converted velocity MSE", "Lower is better"),
        ("auxiliary_only_fid1k", "Auxiliary-only ADM FID-1K", "Lower is better"),
        (
            "best_extrapolation_fid1k",
            "Best extrapolation ADM FID-1K",
            "Dashed: paired v800 baseline",
        ),
    )
    for axis, (key, title, subtitle) in zip(axes, panels, strict=True):
        values = [float(row[key]) for row in rows]
        bars = axis.bar(
            labels,
            values,
            color=colors,
            edgecolor="#303030",
            linewidth=0.8,
        )
        axis.bar_label(bars, fmt="%.2f", padding=3)
        axis.set_title(title)
        axis.set_ylabel(subtitle)
        axis.grid(axis="y", alpha=0.22)
        axis.tick_params(axis="x", rotation=18)
    baseline = float(rows[-1]["baseline_fid1k"])
    axes[2].axhline(
        baseline,
        color="#202020",
        linestyle="--",
        linewidth=1.6,
        label=f"v800 baseline {baseline:.2f}",
    )
    axes[2].legend(fontsize=8)
    figure.suptitle("Frozen v800 clean-x auxiliary-head comparison")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main(args: argparse.Namespace) -> None:
    train_root = args.train_root.expanduser().resolve()
    fid_root = args.fid_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    package_auxiliary_head(
        argparse.Namespace(
            prediction_target="clean",
            train_root=train_root,
            fid_root=fid_root,
            output_root=output_root,
        )
    )
    depth12 = read_json(output_root / "summary.json")
    rows = comparison_rows(
        read_json(TINY_FINAL_SUMMARY),
        read_json(DEPTH8_FULL_SUMMARY),
        depth12,
    )
    write_csv(output_root / "head_architecture_comparison.csv", rows)
    plot_comparison(rows, output_root / "head_architecture_comparison.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--fid-root", type=Path, default=DEFAULT_FID_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
