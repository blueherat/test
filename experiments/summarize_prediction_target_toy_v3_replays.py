#!/usr/bin/env python3
"""Aggregate exact v3 replay runs after matched-randomness re-evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.input_root.expanduser().resolve()
    output = (args.output_dir or root / "aggregate").expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    frames = []
    for path in sorted(root.glob("run_seed*/reanalysis/matched_randomness_metrics.csv")):
        seed = int(path.parents[1].name.removeprefix("run_seed"))
        frame = pd.read_csv(path)
        frame["seed"] = seed
        frames.append(frame)
    if not frames:
        raise RuntimeError(f"no complete exact-v3 replays found below {root}")

    metrics = pd.concat(frames, ignore_index=True)
    contrasts = []
    for (seed, dim), frame in metrics.groupby(["seed", "D"]):
        baseline = frame.loc[frame["condition"] == "x"]
        if len(baseline) != 1:
            raise RuntimeError(f"expected one x baseline for seed={seed}, D={dim}")
        x = baseline.iloc[0]
        for _, row in frame[frame["kind"].isin(["xv", "xeps"])].iterrows():
            contrasts.append(
                {
                    "seed": int(seed),
                    "D": int(dim),
                    "condition": row["condition"],
                    "kind": row["kind"],
                    "strength": float(row["strength"]),
                    "operation": row["operation"],
                    "delta_swd_vs_x": float(
                        row["swd_2d_matched_randomness"]
                        - x["swd_2d_matched_randomness"]
                    ),
                    "relative_swd_vs_x": float(
                        row["swd_2d_matched_randomness"]
                        / max(float(x["swd_2d_matched_randomness"]), 1e-12)
                        - 1.0
                    ),
                    "delta_mmd_vs_x": float(
                        row["mmd_2d_fixed_bandwidth"]
                        - x["mmd_2d_fixed_bandwidth"]
                    ),
                    "swd_delta_ci_low": float(row["swd_delta_vs_x_ci_low"]),
                    "swd_delta_ci_high": float(row["swd_delta_vs_x_ci_high"]),
                }
            )
    contrast_frame = pd.DataFrame(contrasts)

    rows = []
    keys = ["D", "condition", "kind", "strength", "operation"]
    for values, frame in contrast_frame.groupby(keys, dropna=False):
        rows.append(
            dict(zip(keys, values))
            | {
                "seeds": len(frame),
                "mean_relative_swd_vs_x": float(frame["relative_swd_vs_x"].mean()),
                "swd_improved_seed_fraction": float(
                    (frame["delta_swd_vs_x"] < 0).mean()
                ),
                "swd_bootstrap_improved_seed_fraction": float(
                    (frame["swd_delta_ci_high"] < 0).mean()
                ),
                "swd_bootstrap_worsened_seed_fraction": float(
                    (frame["swd_delta_ci_low"] > 0).mean()
                ),
                "mean_delta_mmd_vs_x": float(frame["delta_mmd_vs_x"].mean()),
                "mmd_improved_seed_fraction": float(
                    (frame["delta_mmd_vs_x"] < 0).mean()
                ),
            }
        )
    aggregate = pd.DataFrame(rows).sort_values(keys)

    metrics.to_csv(output / "matched_randomness_metrics.csv", index=False)
    contrast_frame.to_csv(output / "matched_randomness_contrasts.csv", index=False)
    aggregate.to_csv(output / "aggregate_contrasts.csv", index=False)
    report = [
        "Exact v3 multi-seed replay",
        "==========================",
        "",
        f"Seeds: {sorted(metrics['seed'].unique().tolist())}",
        "gamma > 0 is extrapolation from x away from v/epsilon.",
        "",
        aggregate.to_string(index=False),
    ]
    (output / "final_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote exact-v3 replay aggregate to {output}")


if __name__ == "__main__":
    main()
