#!/usr/bin/env python3
"""Plot the paired FID-1K v500 gamma-rho response sweep."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


COLORS = {
    1.0: "#2F6B9A",
    1.5: "#C58A1E",
    2.0: "#D96C32",
    2.5: "#6F7F3B",
    3.0: "#B04A78",
}
MARKERS = {1.0: "o", 1.5: "s", 2.0: "^", 2.5: "D", 3.0: "P"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--closed-fid", type=float, default=74.4795421124332)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rho = float(row["response_scale"])
            if 1.0 <= rho <= 1.5:
                rows.append(
                    {
                        "gamma": float(row["gamma"]),
                        "rho": rho,
                        "fid": float(row["fid"]),
                    }
                )
    if not rows:
        raise ValueError(f"no fine-grid rows found in {path}")
    return rows


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    gammas = sorted({row["gamma"] for row in rows})

    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=False)
    for gamma in gammas:
        points = sorted(
            (row for row in rows if row["gamma"] == gamma),
            key=lambda row: row["rho"],
        )
        ax.plot(
            [row["rho"] for row in points],
            [row["fid"] for row in points],
            color=COLORS[gamma],
            marker=MARKERS[gamma],
            linewidth=2.2,
            markersize=6,
            label=f"gamma={gamma:g}",
        )

    best = min(rows, key=lambda row: row["fid"])
    ax.scatter(
        [best["rho"]],
        [best["fid"]],
        s=115,
        facecolor="white",
        edgecolor="#202124",
        linewidth=1.8,
        zorder=6,
    )
    ax.annotate(
        f"best: gamma={best['gamma']:g}, rho={best['rho']:.2f}\nFID={best['fid']:.3f}",
        xy=(best["rho"], best["fid"]),
        xytext=(24, 38),
        textcoords="offset points",
        fontsize=10,
        color="#202124",
        arrowprops={"arrowstyle": "-", "color": "#60646C", "linewidth": 1.0},
        bbox={
            "boxstyle": "square,pad=0.25",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.9,
        },
    )

    ax.axhline(
        args.closed_fid,
        color="#30343B",
        linestyle="--",
        linewidth=1.6,
        label=f"closed AG gamma=3 ({args.closed_fid:.3f})",
    )
    fig.suptitle(
        "v500 Gamma-Rho FID-1K Sweep",
        fontsize=16,
        y=0.965,
        color="#202124",
    )
    fig.text(
        0.5,
        0.925,
        "ImageNet-100, SiT-S/2, 1,000 paired samples, seed 0; focused FID scale",
        fontsize=10.5,
        color="#60646C",
        ha="center",
    )
    ax.set_xlabel("Strong-response scale (rho)", fontsize=11)
    ax.set_ylabel("FID-1K (lower is better)", fontsize=11)
    ax.set_xticks([1.0 + 0.05 * index for index in range(11)])
    ax.set_xlim(0.99, 1.51)
    ax.set_ylim(74.0, 83.0)
    ax.grid(axis="y", color="#D8DADD", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#8B8F97")
    ax.spines["bottom"].set_color("#8B8F97")
    ax.tick_params(colors="#50545C")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    fig.subplots_adjust(left=0.11, right=0.97, top=0.865, bottom=0.22)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
