#!/usr/bin/env python3
"""Summarize paired weight-space and velocity-space AutoGuidance FID-1K."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

try:
    from experiments.imagenet100_sit_weight_extrapolation import format_scale
    from experiments.train_imagenet100_sit_flow import atomic_json_dump
except ModuleNotFoundError:
    from imagenet100_sit_weight_extrapolation import format_scale
    from train_imagenet100_sit_flow import atomic_json_dump


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/"
    "weight_extrapolation_v800_v500_v1/fid1k_seed0"
)
DEFAULT_VELOCITY_CSVS = (
    REPO_ROOT
    / "docs/data/imagenet100_sit_800k_response_amplification/"
    "factorized_screen_fid1k.csv",
    REPO_ROOT
    / "docs/data/imagenet100_sit_800k_response_amplification/"
    "response_screen_fid1k.csv",
    REPO_ROOT
    / "docs/data/imagenet100_sit_800k_response_amplification/"
    "response_refinement_fid1k.csv",
)
DEFAULT_SCALES = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)


def read_velocity_rows(paths: tuple[Path, ...]) -> dict[float, dict[str, str]]:
    rows: dict[float, dict[str, str]] = {}
    for path in paths:
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("family") != "v500" or row.get("mode") != "closed":
                    continue
                gamma = float(row["gamma"])
                existing = rows.get(gamma)
                if existing is not None and float(existing["fid"]) != float(row["fid"]):
                    raise ValueError(f"conflicting velocity FID for gamma={gamma:g}")
                rows[gamma] = row
    return rows


def main(args: argparse.Namespace) -> None:
    root = args.root.expanduser().resolve()
    velocity = read_velocity_rows(tuple(path.resolve() for path in args.velocity_csvs))
    rows: list[dict[str, object]] = []
    noise_fingerprint = None
    label_fingerprint = None
    for gamma in args.scales:
        result_path = (
            root
            / f"g{format_scale(gamma)}_n{args.num_samples}_seed{args.global_seed}"
            / "weight_extrapolation_fid1k.json"
        )
        weight = json.loads(result_path.read_text(encoding="utf-8"))
        direct = velocity.get(float(gamma))
        if direct is None:
            raise ValueError(f"missing paired velocity FID for gamma={gamma:g}")
        pair = (weight["noise_fingerprint"], weight["label_fingerprint"])
        if noise_fingerprint is None:
            noise_fingerprint, label_fingerprint = pair
        if pair != (noise_fingerprint, label_fingerprint):
            raise ValueError("weight-space conditions do not share noise/labels")
        if pair != (direct["noise_fingerprint"], direct["label_fingerprint"]):
            raise ValueError(f"weight/velocity pairing mismatch at gamma={gamma:g}")
        rows.append(
            {
                "gamma": float(gamma),
                "weight_fid": float(weight["fid"]),
                "velocity_fid": float(direct["fid"]),
                "weight_minus_velocity_fid": float(weight["fid"])
                - float(direct["fid"]),
                "weight_sfid": float(weight["sfid"]),
                "velocity_sfid": float(direct["sfid"]),
                "weight_inception_score": float(weight["inception_score"]),
                "velocity_inception_score": float(direct["inception_score"]),
                "noise_fingerprint": pair[0],
                "label_fingerprint": pair[1],
                "weight_result": str(result_path),
                "velocity_result": direct["result"],
            }
        )

    csv_path = root / "weight_vs_velocity_fid1k.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    best_weight = min(rows, key=lambda row: float(row["weight_fid"]))
    best_velocity = min(rows, key=lambda row: float(row["velocity_fid"]))
    summary = {
        "protocol": "imagenet100_sit_weight_vs_velocity_fid1k_v1",
        "formula_weight": "theta800 + gamma * (theta800 - theta500)",
        "formula_velocity": "v800(z,t) + gamma * (v800(z,t) - v500(z,t))",
        "strictly_paired": True,
        "num_samples": args.num_samples,
        "global_seed": args.global_seed,
        "best_weight": best_weight,
        "best_velocity": best_velocity,
        "rows": rows,
        "csv": str(csv_path),
    }
    atomic_json_dump(summary, root / "weight_vs_velocity_fid1k.json")

    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    gamma = [float(row["gamma"]) for row in rows]
    axis.plot(
        gamma,
        [float(row["weight_fid"]) for row in rows],
        "o-",
        linewidth=2,
        label="Weight extrapolation (one model)",
    )
    axis.plot(
        gamma,
        [float(row["velocity_fid"]) for row in rows],
        "s-",
        linewidth=2,
        label="Velocity extrapolation (two models)",
    )
    axis.set(
        xlabel="Extrapolation gamma",
        ylabel="ADM FID-1K (lower is better)",
        title="v800 strong / v500 weak: weight vs velocity extrapolation",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(root / "weight_vs_velocity_fid1k.png", dpi=180)
    plt.close(figure)
    print(json.dumps(summary, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--velocity-csvs",
        type=Path,
        nargs="+",
        default=DEFAULT_VELOCITY_CSVS,
    )
    parser.add_argument("--scales", type=float, nargs="+", default=DEFAULT_SCALES)
    parser.add_argument("--num-samples", type=int, default=1_000)
    parser.add_argument("--global-seed", type=int, default=0)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
