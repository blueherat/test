"""Sweep step permutations of one fixed flow-matching time multiset."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.small_image_time_order_study import (  # noqa: E402
    TimeOrderStudyConfig,
    run_study,
)
from experiments.small_image_time_seed_sweep import summarize_seed_sweep  # noqa: E402


DEFAULT_OUTPUT_ROOT = (
    Path.home() / "data/eqvae/experiments/small_image_time_permutation_sweep"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-seed", type=int, default=3)
    parser.add_argument("--permutations", default="0,1,2,3,4,5,6,7,8,9,10,11")
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    permutations = tuple(
        int(value.strip()) for value in args.permutations.split(",") if value.strip()
    )
    if len(set(permutations)) < 2:
        raise ValueError("provide at least two distinct permutation seeds")
    devices = tuple(value.strip() for value in args.devices.split(",") if value.strip())
    schedules = tuple(
        f"step_permuted_seed{args.source_seed}_perm{seed}"
        for seed in permutations
    )
    config = TimeOrderStudyConfig(
        output_root=DEFAULT_OUTPUT_ROOT,
        schedules=schedules,
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
        summary.to_csv(result_dir / "permutation_summary.csv", index=False)
    print(f"result_dir={result_dir}")
    print(conditions[conditions["metric"].eq("feature_fid")].to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
