#!/usr/bin/env python3
"""Exact one-dimensional selective-amplification counterexample."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def selective_amplification_distances(
    amplification: np.ndarray, artifact_mass: float
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < artifact_mass < 1.0:
        raise ValueError("artifact_mass must lie in (0, 1)")
    values = np.asarray(amplification, dtype=np.float64)
    if np.any(values < 0.0):
        raise ValueError("amplification must be nonnegative")

    epsilon = float(artifact_mass)
    fake_mean = epsilon * values
    fake_variance = (
        1.0
        - epsilon
        + epsilon * (1.0 - epsilon) * values**2
    )
    real_whitened = fake_mean**2 + (1.0 - np.sqrt(fake_variance)) ** 2

    # The equal real/fake mixture has mean fake_mean/2. Its variance includes
    # both within-distribution covariance and the between-distribution term.
    pooled_variance = (
        0.5 * (1.0 + fake_variance) + 0.25 * fake_mean**2
    )
    pooled_whitened = real_whitened / pooled_variance
    return real_whitened, pooled_whitened


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--artifact-mass", type=float, default=0.05)
    parser.add_argument("--points", type=int, default=241)
    parser.add_argument("--maximum", type=float, default=1e4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    amplification = np.concatenate(
        ([0.0], np.geomspace(1e-3, args.maximum, args.points - 1))
    )
    real_fd, pooled_fd = selective_amplification_distances(
        amplification, args.artifact_mass
    )

    csv_path = args.output_root / "selective_amplification.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("amplification", "real_whitened_fd", "pooled_whitened_fd"))
        writer.writerows(zip(amplification, real_fd, pooled_fd))

    positive = amplification > 0
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.plot(amplification[positive], real_fd[positive], label="real whitening")
    axis.plot(amplification[positive], pooled_fd[positive], label="pooled whitening")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("fake-only feature amplitude M")
    axis.set_ylabel("calibrated Fréchet distance")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output_root / "selective_amplification.png", dpi=180)
    plt.close(figure)

    epsilon = float(args.artifact_mass)
    pooled_limit = 1.0 / (0.5 - 0.25 * epsilon)
    summary = {
        "artifact_mass": epsilon,
        "maximum_amplification": float(amplification[-1]),
        "real_whitened_fd_at_maximum": float(real_fd[-1]),
        "pooled_whitened_fd_at_maximum": float(pooled_fd[-1]),
        "pooled_asymptotic_limit": pooled_limit,
        "scalar_full_fd_upper_bound": 4.0,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
