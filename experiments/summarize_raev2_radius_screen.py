#!/usr/bin/env python3
"""Validate paired RAEv2 radius experiments and summarize official FID."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PAIR_REQUEST_KEYS = (
    "protocol", "sample_count", "batch_size", "seed", "guidance_scale",
    "guidance_min_time", "guidance_max_time", "num_steps", "precision",
    "checkpoint_sha256", "config_sha256", "state_key", "world_size",
    "guidance_arithmetic", "forward_layout", "tf32", "time_grid", "torch_version",
)
PAIR_SUMMARY_KEYS = ("samples", "seed", "noise_sha256", "labels_sha256", "full_sample_evaluations")


def read_run(path: Path):
    request = json.loads((path / "request.json").read_text())
    summary = json.loads((path / "summary.json").read_text())
    if summary.get("complete") is not True:
        raise ValueError(f"incomplete run: {path}")
    with (path / "official_metrics.csv").open(newline="") as f:
        metrics = list(csv.DictReader(f))
    if len(metrics) != 1 or metrics[0]["sample_sha256"] != summary["archive_sha256"]:
        raise ValueError(f"metric/sample mismatch: {path}")
    return request, summary, metrics[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank-dir", type=Path, required=True)
    p.add_argument("--baseline", default="ordinary")
    args = p.parse_args()
    root = args.bank_dir.resolve()
    baseline = read_run(root / args.baseline)
    baseline_fid = float(baseline[2]["fid"])
    rows = []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or not (directory / "request.json").exists():
            continue
        request, summary, metrics = read_run(directory)
        for keys, reference, candidate in (
            (PAIR_REQUEST_KEYS, baseline[0], request),
            (PAIR_SUMMARY_KEYS, baseline[1], summary),
            (("fid_reference", "evaluator_commit"), baseline[2], metrics),
        ):
            for key in keys:
                if reference[key] != candidate[key]:
                    raise ValueError(f"unpaired {directory.name}: {key}")
        fid = float(metrics["fid"])
        with (directory / "geometry.csv").open(newline="") as f:
            geometry = list(csv.DictReader(f))
        active = [g for g in geometry if request["guidance_min_time"] <= float(g["noise_time"]) <= request["guidance_max_time"]]
        rows.append({
            "condition": directory.name, "mode": request["mode"], "grouping": request["grouping"],
            "seed": request["seed"], "samples": summary["samples"], "fid": fid,
            "delta_fid": fid - baseline_fid, "relative_improvement_percent": 100 * (baseline_fid - fid) / baseline_fid,
            "inception_score": float(metrics["inception_score"]),
            "elapsed_seconds": summary["elapsed_seconds"],
            "time_mean_active_radius_ratio": sum(float(g["relative_radius_ratio"]) for g in active) / len(active),
            "time_mean_active_radial_energy_fraction": sum(float(g["radial_energy_fraction"]) for g in active) / len(active),
            "archive_sha256": summary["archive_sha256"], "paired": True,
        })
    with (root / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result = {"pairing_validated": True, "baseline": args.baseline,
              "interpretation": "FID-1K screen only; an independent bank is required for candidate confirmation",
              "noise_sha256": baseline[1]["noise_sha256"], "labels_sha256": baseline[1]["labels_sha256"], "rows": rows}
    (root / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    for row in rows:
        print(f"{row['condition']:12s} FID {row['fid']:.6f} delta {row['delta_fid']:+.6f} improvement {row['relative_improvement_percent']:+.3f}% IS {row['inception_score']:.4f}")


if __name__ == "__main__":
    main()
