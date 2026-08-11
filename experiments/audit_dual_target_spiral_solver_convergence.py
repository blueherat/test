#!/usr/bin/env python3
"""Check whether the exact-Bayes spiral baseline converges by 200 Heun steps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_dual_target_closed_loop_spiral_toy as spiral
import run_dual_target_closed_loop_toy as core


@torch.no_grad()
def run_dimension(args: argparse.Namespace, dimension: int) -> list[dict]:
    device = torch.device(args.device)
    distribution = spiral.ContinuousSpiralDistribution(
        dimension,
        data_jitter=args.data_jitter,
        quadrature_points=args.quadrature_points,
        locator_points=args.locator_points,
        frequency_scale=args.frequency_scale,
        embedding_seed=core.stable_seed(args.seed, dimension, 71),
        device=device,
        scale_mode="unit_rms",
        curvature=0.0,
        bayes_batch_chunk=args.bayes_batch_chunk,
    )
    suite = core.ModelSuite(models={}, optimizers={})
    noise_generator = torch.Generator(device=device.type).manual_seed(
        core.stable_seed(args.seed, dimension, 149)
    )
    initial_noise = torch.randn(
        args.sample_count, dimension, device=device, generator=noise_generator
    )
    reference_generator = torch.Generator(device=device.type).manual_seed(
        core.stable_seed(args.seed, dimension, 151)
    )
    reference, reference_u, _ = distribution.sample(
        args.reference_count, generator=reference_generator
    )
    resample_generator = torch.Generator(device=device.type).manual_seed(
        core.stable_seed(args.seed, dimension, 153)
    )
    reference_resample, _, _ = distribution.sample(
        args.sample_count, generator=resample_generator
    )

    rows: list[dict] = []
    for steps in args.steps:
        print(f"D={dimension}: exact Bayes with {steps} Heun steps", flush=True)
        endpoint, _ = core.sample_heun(
            "Bayes_exact",
            suite=suite,
            distribution=distribution,
            initial_noise=initial_noise,
            steps=steps,
            denominator_floor=args.denominator_floor,
            snapshot_times=(),
        )
        metrics = spiral.endpoint_metrics_spiral(
            generated={
                "Reference_resample": reference_resample.cpu(),
                "Bayes_exact": endpoint.cpu(),
            },
            reference=reference.cpu(),
            reference_intrinsic=reference_u.cpu(),
            distribution=distribution,
            seed=core.stable_seed(args.seed, dimension, 157),
            swd_projections=args.swd_projections,
            swd_max_points=args.sample_count,
            full_swd_projections=args.full_swd_projections,
            full_swd_max_points=args.sample_count,
            mmd_max_points=args.mmd_max_points,
            coverage_bins=args.coverage_bins,
            conditional_ridge_bins=args.conditional_ridge_bins,
            conditional_ridge_min_count=args.conditional_ridge_min_count,
        )
        for row in metrics:
            rows.append({"ambient_dim": dimension, "heun_steps": steps, **row})
    return rows


def plot_convergence(path: Path, frame: pd.DataFrame) -> None:
    metrics = [
        ("swd_2d", "Intrinsic SWD"),
        ("swd_fullD", "Full-D SWD"),
        ("ridge_width_ratio", "Ridge width / reference"),
        ("arc_hist_tv", "Arc coverage TV"),
    ]
    dimensions = sorted(frame["ambient_dim"].unique())
    figure, axes = plt.subplots(
        len(dimensions), len(metrics), figsize=(5.2 * len(metrics), 4.8 * len(dimensions)),
        squeeze=False,
    )
    for row_index, dimension in enumerate(dimensions):
        subset = frame[frame["ambient_dim"] == dimension]
        for column_index, (metric, title) in enumerate(metrics):
            axis = axes[row_index, column_index]
            for condition, style in (
                ("Bayes_exact", "o-"),
                ("Reference_resample", "--"),
            ):
                values = subset[subset["condition"] == condition].sort_values(
                    "heun_steps"
                )
                axis.plot(values["heun_steps"], values[metric], style, label=condition)
            axis.set_xscale("log", base=2)
            if metric in {"swd_2d", "swd_fullD", "arc_hist_tv"}:
                axis.set_yscale("log")
            if metric == "ridge_width_ratio":
                axis.axhline(1.0, color="black", linewidth=1, linestyle=":")
            axis.set_xlabel("Heun steps")
            axis.set_title(f"D={dimension}: {title}")
            axis.grid(alpha=0.25)
    axes[0, -1].legend()
    figure.suptitle("Exact-Bayes solver convergence on the continuous spiral", y=1.01)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dims", type=core.parse_int_list, default=core.parse_int_list("2,512"))
    parser.add_argument(
        "--steps", type=core.parse_int_list, default=core.parse_int_list("50,100,200,400,800")
    )
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--sample-count", type=int, default=4096)
    parser.add_argument("--reference-count", type=int, default=8192)
    parser.add_argument("--data-jitter", type=float, default=0.015)
    parser.add_argument("--frequency-scale", type=float, default=6.0)
    parser.add_argument("--quadrature-points", type=int, default=1024)
    parser.add_argument("--locator-points", type=int, default=4096)
    parser.add_argument("--bayes-batch-chunk", type=int, default=4096)
    parser.add_argument("--denominator-floor", type=float, default=1e-3)
    parser.add_argument("--swd-projections", type=int, default=256)
    parser.add_argument("--full-swd-projections", type=int, default=64)
    parser.add_argument("--mmd-max-points", type=int, default=2048)
    parser.add_argument("--coverage-bins", type=int, default=32)
    parser.add_argument("--conditional-ridge-bins", type=int, default=16)
    parser.add_argument("--conditional-ridge-min-count", type=int, default=24)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output = args.output_root / "aggregate"
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for dimension in args.dims:
        rows.extend(run_dimension(args, dimension))
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "solver_step_convergence.csv", index=False, lineterminator="\n")
    plot_convergence(output / "solver_step_convergence.png", frame)
    (output / "solver_step_convergence_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "output_root": str(args.output_root),
                "device": str(args.device),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
