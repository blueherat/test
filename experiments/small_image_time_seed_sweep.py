"""Sweep flow-matching time seeds with every other random source fixed."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.small_image_time_order_study import (  # noqa: E402
    TimeOrderStudyConfig,
    run_study,
)


DEFAULT_OUTPUT_ROOT = Path.home() / "data/eqvae/experiments/small_image_time_seed_sweep"


def summarize_seed_sweep(
    conditions: pd.DataFrame,
    *,
    metric: str = "feature_fid",
) -> pd.DataFrame:
    selected = conditions[conditions["metric"].eq(metric)].copy()
    if len(selected) < 2:
        raise ValueError("seed sweep summary requires at least two conditions")
    baseline = selected["baseline_mean"]
    weighted = selected["weighted_mean"]
    ratio = selected["ratio_mean"]
    difference = weighted - baseline
    difference_std = float(difference.std(ddof=1))
    difference_se = difference_std / math.sqrt(len(difference))
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "time_seeds": int(len(selected)),
                "baseline_mean": float(baseline.mean()),
                "baseline_std": float(baseline.std(ddof=1)),
                "baseline_cv": float(baseline.std(ddof=1) / baseline.mean()),
                "baseline_min": float(baseline.min()),
                "baseline_max": float(baseline.max()),
                "weighted_mean": float(weighted.mean()),
                "weighted_std": float(weighted.std(ddof=1)),
                "weighted_cv": float(weighted.std(ddof=1) / weighted.mean()),
                "weighted_min": float(weighted.min()),
                "weighted_max": float(weighted.max()),
                "weighted_over_baseline_variance": float(
                    weighted.var(ddof=1) / baseline.var(ddof=1)
                ),
                "mean_paired_difference": float(difference.mean()),
                "paired_difference_std": difference_std,
                "paired_difference_ci95_normal_low": float(
                    difference.mean() - 1.96 * difference_se
                ),
                "paired_difference_ci95_normal_high": float(
                    difference.mean() + 1.96 * difference_se
                ),
                "median_ratio": float(ratio.median()),
                "harm_rate": float((ratio > 1.0).mean()),
                "baseline_weighted_correlation": float(baseline.corr(weighted)),
                "baseline_ratio_correlation": float(baseline.corr(ratio)),
            }
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9,10,11")
    parser.add_argument("--dataset", choices=("mnist", "fashion_mnist"), default="mnist")
    parser.add_argument("--sampling", choices=("iid", "stratified"), default="iid")
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    if len(set(seeds)) < 2:
        raise ValueError("provide at least two distinct seeds")
    devices = tuple(value.strip() for value in args.devices.split(",") if value.strip())
    data_root = Path(
        "/data/shared/mnist"
        if args.dataset == "mnist"
        else "/data/shared/fashion_mnist"
    )
    config = TimeOrderStudyConfig(
        dataset=args.dataset,
        data_root=data_root,
        output_root=DEFAULT_OUTPUT_ROOT / args.dataset / args.sampling,
        schedules=tuple(f"{args.sampling}_seed{seed}" for seed in seeds),
        devices=devices or ("cpu",),
        save=not args.no_save,
    )
    if args.quick:
        config = replace(
            config,
            devices=(devices[0],) if devices else ("cpu",),
            train_size=128,
            test_size=64,
            sample_count=64,
            batch_size=16,
            steps=4,
            width=8,
            depth=1,
            ode_steps=5,
            classifier_epochs=1,
            classifier_batch_size=32,
            rollout_seeds=(1701,),
        )
    _, conditions, result_dir = run_study(config)
    summary = summarize_seed_sweep(conditions)
    if result_dir is not None:
        summary.to_csv(result_dir / "sweep_summary.csv", index=False)
    print(f"result_dir={result_dir}")
    print(conditions[conditions["metric"].eq("feature_fid")].to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
