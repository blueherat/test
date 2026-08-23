#!/usr/bin/env python3
"""Plot official AdvFD training and saved-critic diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-jsonl", type=Path, required=True)
    parser.add_argument("--checkpoint-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, float]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def values(rows, key: str) -> list[float]:
    return [float(row[key]) for row in rows]


def series(
    rows: list[dict[str, object]], x_key: str, y_key: str
) -> tuple[list[float], list[float]]:
    selected = [row for row in rows if x_key in row and y_key in row]
    return values(selected, x_key), values(selected, y_key)


def main() -> None:
    args = parse_args()
    metrics = read_jsonl(args.metrics_jsonl)
    checkpoints = read_csv(args.checkpoint_csv)
    if not metrics or not checkpoints:
        raise ValueError("both metric sources must contain data")

    iterations = values(metrics, "iteration")
    checkpoint_steps = values(checkpoints, "filename_step")
    prefix = "critic_0_inception"

    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)

    static_axis = axes[0, 0]
    for key, label in (
        ("fid_siglip", "SigLIP2"),
        ("fid_mae", "MAE"),
        ("fid_inception", "Inception"),
    ):
        static_axis.plot(iterations, values(metrics, key), label=label)
    static_axis.set_yscale("log")
    static_axis.set_title("Static SIM training FD")
    static_axis.set_xlabel("Generator step")
    static_axis.set_ylabel("FD (log scale)")
    static_axis.legend()
    static_axis.grid(alpha=0.25)

    adaptive_axis = axes[0, 1]
    for key, label, alpha in (
        ("fd_adv_inception", "Generator adaptive FD", 1.0),
        ("fd_adv_critic_inception", "Critic adaptive FD", 0.8),
        (
            "fd_adv_critic_grad_norm_inception",
            "Critic pre-clip grad norm",
            0.65,
        ),
    ):
        x_values, y_values = series(metrics, "iteration", key)
        adaptive_axis.plot(x_values, y_values, label=label, alpha=alpha)
    adaptive_axis.set_yscale("log")
    adaptive_axis.set_title("Adaptive branch scale")
    adaptive_axis.set_xlabel("Generator step")
    adaptive_axis.set_ylabel("Value (log scale)")
    adaptive_axis.legend()
    adaptive_axis.grid(alpha=0.25)

    rms_axis = axes[1, 0]
    rms_axis.plot(
        checkpoint_steps,
        values(checkpoints, f"{prefix}_real_feature_rms_ratio_to_reference"),
        marker="o",
        label="Real / reference RMS",
    )
    rms_axis.plot(
        checkpoint_steps,
        values(checkpoints, f"{prefix}_fake_feature_rms_ratio_to_reference"),
        marker="o",
        label="Fake / reference RMS",
    )
    rms_axis.set_yscale("log")
    rms_axis.set_title("Saved adaptive feature scale")
    rms_axis.set_xlabel("Generator step")
    rms_axis.set_ylabel("RMS ratio (log scale)")
    rms_axis.legend()
    rms_axis.grid(alpha=0.25)

    rank_axis = axes[1, 1]
    rank_axis.plot(
        checkpoint_steps,
        values(checkpoints, f"{prefix}_real_covariance_participation_rank"),
        marker="o",
        label="Real effective rank",
    )
    rank_axis.plot(
        checkpoint_steps,
        values(checkpoints, f"{prefix}_fake_covariance_participation_rank"),
        marker="o",
        label="Fake effective rank",
    )
    rank_axis.set_title("Saved adaptive covariance spectrum")
    rank_axis.set_xlabel("Generator step")
    rank_axis.set_ylabel("Participation rank (of 2048)")
    rank_axis.legend()
    rank_axis.grid(alpha=0.25)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.suptitle("Official-code-faithful pMF-B AdvFD scaled reproduction")
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
