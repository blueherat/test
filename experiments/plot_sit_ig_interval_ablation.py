"""Plot the completed SiT interval ADM-FID table."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.metrics.expanduser().resolve()).sort_values("fid")
    figure, axis = plt.subplots(figsize=(12, 6))
    colors = ["#F58518" if "t0p7_1p0" in name else "#4C78A8" for name in frame.condition]
    axis.bar(frame.condition, frame.fid, color=colors)
    axis.set(
        title="SiT-XL/2 interval ablation (diagnostic sample count)",
        ylabel="ADM FID",
    )
    axis.tick_params(axis="x", rotation=35)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
