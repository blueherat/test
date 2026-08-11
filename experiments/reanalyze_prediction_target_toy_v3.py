#!/usr/bin/env python3
"""Re-evaluate saved v3 toy samples with matched metric randomness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_prediction_target_extrapolation_toy_v4 import (
    bootstrap_swd_delta,
    fixed_projection_matrix,
    mmd_2d_fixed,
    rbf_bandwidth_2d_fixed,
    sample_spiral_2d,
    stable_seed,
    swd_2d_fixed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("prediction_target_toy_v3"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dims", default="16")
    parser.add_argument("--reference-count", type=int, default=10000)
    parser.add_argument("--swd-projections", type=int, default=512)
    parser.add_argument("--mmd-points", type=int, default=2048)
    parser.add_argument("--bootstrap-points", type=int, default=2048)
    parser.add_argument("--bootstrap-projections", type=int, default=64)
    parser.add_argument("--bootstrap-reps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260807)
    return parser.parse_args()


def condition_metadata(name: str) -> tuple[str, float, str]:
    for kind in ("xv", "xeps"):
        prefix = f"{kind}_g"
        if name.startswith(prefix):
            strength = float(name[len(prefix) :])
            return kind, strength, "extrapolation" if strength > 0 else "interpolation"
    return name, 0.0, "baseline"


def main() -> None:
    args = parse_args()
    dims = [int(value.strip()) for value in args.dims.split(",") if value.strip()]
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + 999)
    reference = sample_spiral_2d(
        args.reference_count,
        device=torch.device("cpu"),
        jitter=0.015,
        generator=generator,
    ).numpy()

    rows: list[dict[str, float | int | str]] = []
    for D in dims:
        sample_files = sorted((args.input_root / f"D{D}").glob("samples_*.npz"))
        if not sample_files:
            raise FileNotFoundError(f"no v3 samples found for D={D}")
        samples = {
            path.stem.removeprefix("samples_"): np.load(path)["intrinsic"]
            for path in sample_files
        }
        baseline = samples["x"]
        n = min(len(baseline), len(reference))
        idx_sample = np.arange(n)
        idx_reference = np.arange(n)
        theta = fixed_projection_matrix(
            args.swd_projections, stable_seed(args.seed, D, 4101)
        )

        rng = np.random.default_rng(stable_seed(args.seed, D, 4102))
        mmd_n = min(args.mmd_points, n)
        mmd_sample_ids = rng.choice(n, mmd_n, replace=False)
        mmd_reference_ids = rng.choice(n, mmd_n, replace=False)
        bandwidth_ids = rng.choice(2 * mmd_n, min(1024, 2 * mmd_n), replace=False)
        sigma2 = rbf_bandwidth_2d_fixed(
            baseline,
            reference,
            idx_a=mmd_sample_ids,
            idx_b=mmd_reference_ids,
            bandwidth_subset=bandwidth_ids,
        )

        for name, sample in samples.items():
            kind, strength, operation = condition_metadata(name)
            swd = swd_2d_fixed(
                sample,
                reference,
                theta=theta,
                idx_a=idx_sample,
                idx_b=idx_reference,
            )
            mmd = mmd_2d_fixed(
                sample,
                reference,
                idx_a=mmd_sample_ids,
                idx_b=mmd_reference_ids,
                sigma2=sigma2,
            )
            if name == "x":
                delta, low, high = 0.0, 0.0, 0.0
            else:
                delta, low, high = bootstrap_swd_delta(
                    sample,
                    baseline,
                    reference,
                    theta=theta[: min(args.bootstrap_projections, len(theta))],
                    reps=args.bootstrap_reps,
                    seed=stable_seed(args.seed, D, 4103),
                    max_points=args.bootstrap_points,
                )
            rows.append(
                {
                    "D": D,
                    "condition": name,
                    "kind": kind,
                    "strength": strength,
                    "operation": operation,
                    "swd_2d_matched_randomness": swd,
                    "mmd_2d_fixed_bandwidth": mmd,
                    "swd_delta_vs_x_boot_mean": delta,
                    "swd_delta_vs_x_ci_low": low,
                    "swd_delta_vs_x_ci_high": high,
                    "mmd_sigma2": sigma2,
                }
            )

    frame = pd.DataFrame(rows).sort_values(["D", "swd_2d_matched_randomness"])
    frame.to_csv(output / "matched_randomness_metrics.csv", index=False)
    best = frame.groupby("D", as_index=False).first()
    report = {
        "protocol": "prediction_target_toy_v3_matched_randomness_reanalysis_v1",
        "args": vars(args) | {"input_root": str(args.input_root), "output_dir": str(output)},
        "best_by_dimension": best.to_dict(orient="records"),
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(frame.to_string(index=False))
    print(f"Wrote matched-randomness reanalysis to {output}")


if __name__ == "__main__":
    main()
