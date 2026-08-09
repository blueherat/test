#!/usr/bin/env python3
"""Evaluate extrapolation beyond a stronger v predictor using saved v4 models.

The existing x-v convention is

    clean_gamma = x + gamma * (x - v).

Writing lambda = -gamma gives interpolation from x to v. lambda=1 is v and
lambda>1 is true extrapolation beyond v:

    clean_lambda = x + lambda * (v - x)
                 = v + (lambda - 1) * (v - x).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from experiments.run_prediction_target_extrapolation_toy_v4 import (
        CurvedEmbedding,
        DenoiseMLP,
        bootstrap_swd_delta,
        fixed_projection_matrix,
        mmd_2d_fixed,
        rbf_bandwidth_2d_fixed,
        sample_mixture_conditions,
        sample_spiral_2d,
        stable_seed,
        swd_2d_fixed,
    )
except ModuleNotFoundError:
    from run_prediction_target_extrapolation_toy_v4 import (
        CurvedEmbedding,
        DenoiseMLP,
        bootstrap_swd_delta,
        fixed_projection_matrix,
        mmd_2d_fixed,
        rbf_bandwidth_2d_fixed,
        sample_mixture_conditions,
        sample_spiral_2d,
        stable_seed,
        swd_2d_fixed,
    )


DEFAULT_ROOT = Path(
    "/data/users/zhoushunyu/eqvae/experiments/"
    "prediction_target_toy_v4_reverse_from_v"
)
SOURCE_ROOT = Path(
    "/data/users/zhoushunyu/eqvae/experiments/"
    "prediction_target_toy_v4_multiregime_screen"
)
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.003, 1.01, 1.03, 1.1, 1.25)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--sample-count", type=int, default=10000)
    parser.add_argument("--sample-batch-size", type=int, default=500)
    parser.add_argument("--sample-steps", type=int, default=200)
    parser.add_argument("--bootstrap-reps", type=int, default=300)
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args()


def setting_dir(root: Path, seed: int) -> Path:
    return (
        root
        / f"run_seed{seed}"
        / "mid_linear"
        / f"seed{seed}"
        / "D16"
        / "curv0"
        / "scale_constant_norm"
        / "loss_v"
        / "H256"
    )


def evaluate_seed(args: argparse.Namespace) -> Path:
    if args.seed is None:
        raise ValueError("--seed is required unless --aggregate-only is used")
    seed = int(args.seed)
    device = torch.device(args.device)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    source = setting_dir(args.source_root, seed)
    checkpoint = source / "models.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    states = torch.load(checkpoint, map_location=device, weights_only=True)
    models = {
        target: DenoiseMLP(16, hidden=256, depth=5, time_dim=32).to(device).eval()
        for target in ("x", "v", "eps")
    }
    for target, model in models.items():
        model.load_state_dict(states[target])

    embedding = CurvedEmbedding(
        16,
        curvature=0.0,
        frequency_scale=6.0,
        seed=stable_seed(seed, 16, 0, 41),
        device=device,
        scale_mode="constant_norm",
    )
    reference_generator = torch.Generator(device=device.type)
    reference_generator.manual_seed(stable_seed(seed, 16, 0, 42))
    reference = sample_spiral_2d(
        max(args.sample_count, 20000),
        device=device,
        jitter=0.015,
        generator=reference_generator,
    ).cpu().numpy()

    gammas = [-value for value in LAMBDAS]
    sample_seed = stable_seed(seed, 16, 256, 0, 77)
    ambient_samples = sample_mixture_conditions(
        models=models,
        embedding=embedding,
        kind="xv",
        strengths=gammas,
        sample_count=args.sample_count,
        sample_batch_size=args.sample_batch_size,
        sample_steps=args.sample_steps,
        t_max=0.98,
        t_min=0.02,
        clip=0.02,
        seed=sample_seed,
        device=device,
    )

    metric_count = min(args.sample_count, len(reference), 4096)
    rng = np.random.default_rng(stable_seed(seed, 16, 256, 1991))
    sample_ids = rng.choice(args.sample_count, metric_count, replace=False)
    reference_ids = rng.choice(len(reference), metric_count, replace=False)
    theta = fixed_projection_matrix(512, stable_seed(seed, 16, 256, 1992))
    bandwidth_subset = rng.choice(
        2 * metric_count, min(1024, 2 * metric_count), replace=False
    )

    intrinsic_samples = []
    manifold_values = []
    with torch.inference_mode():
        for ambient in ambient_samples:
            tensor = torch.from_numpy(ambient).to(device)
            intrinsic_samples.append(embedding.decode_intrinsic(tensor).cpu().numpy())
            manifold_values.append(
                float(embedding.manifold_consistency_rms(tensor).mean().cpu())
            )

    v_index = LAMBDAS.index(1.0)
    v_intrinsic = intrinsic_samples[v_index]
    sigma2 = rbf_bandwidth_2d_fixed(
        v_intrinsic,
        reference,
        idx_a=sample_ids,
        idx_b=reference_ids,
        bandwidth_subset=bandwidth_subset,
    )

    rows = []
    for lam, gamma, intrinsic, manifold_rms in zip(
        LAMBDAS, gammas, intrinsic_samples, manifold_values
    ):
        swd = swd_2d_fixed(
            intrinsic,
            reference,
            theta=theta,
            idx_a=sample_ids,
            idx_b=reference_ids,
        )
        mmd = mmd_2d_fixed(
            intrinsic,
            reference,
            idx_a=sample_ids,
            idx_b=reference_ids,
            sigma2=sigma2,
        )
        if lam == 1.0:
            boot_mean = ci_low = ci_high = 0.0
        else:
            boot_mean, ci_low, ci_high = bootstrap_swd_delta(
                intrinsic,
                v_intrinsic,
                reference,
                theta=theta[:64],
                reps=args.bootstrap_reps,
                seed=stable_seed(seed, 16, 256, int(round(lam * 10000)), 1993),
                max_points=1024,
            )
        rows.append(
            {
                "seed": seed,
                "lambda_x_to_v": lam,
                "gamma_x_minus_v": gamma,
                "alpha_beyond_v": max(lam - 1.0, 0.0),
                "operation": (
                    "x" if lam == 0.0 else
                    "interpolation" if lam < 1.0 else
                    "v" if lam == 1.0 else
                    "extrapolation_beyond_v"
                ),
                "swd_2d": swd,
                "mmd_2d": mmd,
                "manifold_consistency_rms": manifold_rms,
                "swd_delta_vs_v_boot_mean": boot_mean,
                "swd_delta_vs_v_ci_low": ci_low,
                "swd_delta_vs_v_ci_high": ci_high,
                "mmd_bandwidth_sigma2": sigma2,
            }
        )

    output = args.output_root / f"seed{seed}"
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "metrics.csv", index=False)
    (output / "config.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "source_checkpoint": str(checkpoint),
                "sample_count": args.sample_count,
                "sample_steps": args.sample_steps,
                "lambdas": list(LAMBDAS),
                "shared_initial_noise": True,
                "comparison_baseline": "v (lambda=1)",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(frame.to_string(index=False), flush=True)
    return output / "metrics.csv"


def aggregate(root: Path) -> Path:
    paths = sorted(root.glob("seed*/metrics.csv"))
    if not paths:
        raise RuntimeError(f"no per-seed metrics found below {root}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    v = frame[np.isclose(frame["lambda_x_to_v"], 1.0)][
        ["seed", "swd_2d", "mmd_2d", "manifold_consistency_rms"]
    ].rename(
        columns={
            "swd_2d": "v_swd",
            "mmd_2d": "v_mmd",
            "manifold_consistency_rms": "v_manifold_rms",
        }
    )
    joined = frame.merge(v, on="seed", how="left")
    joined["delta_swd_vs_v"] = joined["swd_2d"] - joined["v_swd"]
    joined["relative_swd_vs_v"] = joined["swd_2d"] / joined["v_swd"] - 1.0
    joined["delta_mmd_vs_v"] = joined["mmd_2d"] - joined["v_mmd"]
    joined["delta_manifold_vs_v"] = (
        joined["manifold_consistency_rms"] - joined["v_manifold_rms"]
    )

    group_keys = [
        "lambda_x_to_v",
        "gamma_x_minus_v",
        "alpha_beyond_v",
        "operation",
    ]
    rows = []
    for keys, group in joined.groupby(group_keys, dropna=False):
        row = dict(zip(group_keys, keys))
        row.update(
            {
                "seeds": len(group),
                "mean_swd": float(group["swd_2d"].mean()),
                "mean_relative_swd_vs_v": float(group["relative_swd_vs_v"].mean()),
                "swd_improved_seed_fraction": float(
                    (group["delta_swd_vs_v"] < 0).mean()
                ),
                "swd_bootstrap_improved_seed_fraction": float(
                    (group["swd_delta_vs_v_ci_high"] < 0).mean()
                ),
                "mean_delta_mmd_vs_v": float(group["delta_mmd_vs_v"].mean()),
                "mmd_improved_seed_fraction": float(
                    (group["delta_mmd_vs_v"] < 0).mean()
                ),
                "mean_delta_manifold_vs_v": float(
                    group["delta_manifold_vs_v"].mean()
                ),
            }
        )
        rows.append(row)
    aggregate_frame = pd.DataFrame(rows).sort_values("lambda_x_to_v")
    output = root / "aggregate"
    output.mkdir(parents=True, exist_ok=True)
    joined.to_csv(output / "all_seed_metrics.csv", index=False)
    aggregate_frame.to_csv(output / "aggregate_metrics.csv", index=False)
    (output / "final_report.txt").write_text(
        "Reverse extrapolation from x through stronger v\n"
        "===============================================\n\n"
        + aggregate_frame.to_string(index=False)
        + "\n",
        encoding="utf-8",
    )
    print(aggregate_frame.to_string(index=False), flush=True)
    return output / "final_report.txt"


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.expanduser().resolve()
    args.source_root = args.source_root.expanduser().resolve()
    if args.aggregate_only:
        aggregate(args.output_root)
    else:
        evaluate_seed(args)


if __name__ == "__main__":
    main()
