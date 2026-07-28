"""Evaluate the preregistered 5k-checkpoint RAE floor-path persistence gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


BASELINE_ROOT = Path.home() / "data/eqvae/experiments/rae_layerwise_path_train"
CANDIDATE_ROOT = Path.home() / "data/eqvae/experiments/rae_path_schedule_train"
DEFAULT_OUTPUT = CANDIDATE_ROOT / "checkpoint5k_gate"
REFERENCE = Path("/data/shared/adm_refs/VIRTUAL_imagenet256_labeled.npz")
SAMPLING_SEED = 20_260_718
BRANCHES = {
    "static": BASELINE_ROOT / "seed3407_static_rank16_s0_to_10000",
    "annealed": BASELINE_ROOT / "seed3407_annealed_rank16_s0_to_10000",
    "floor020_p2": CANDIDATE_ROOT / "seed3407_floor020_p2_rank16_s0_to_2000",
}


def summarize(table: pd.DataFrame) -> dict[str, object]:
    fid = "frechet_inception_distance"
    kid = "kernel_inception_distance_mean"
    indexed = table.set_index("condition")
    static = indexed.loc["static"]
    annealed = indexed.loc["annealed"]
    candidate = indexed.loc["floor020_p2"]
    relative_fid = float((candidate[fid] - static[fid]) / static[fid])
    relative_kid = float((candidate[kid] - static[kid]) / static[kid])
    mean_relative_degradation = 0.5 * (relative_fid + relative_kid)
    predictions = {
        "p1_static_beats_annealed_both": bool(
            static[fid] < annealed[fid] and static[kid] < annealed[kid]
        ),
        "p2_candidate_beats_annealed_both": bool(
            candidate[fid] < annealed[fid] and candidate[kid] < annealed[kid]
        ),
        "p3_candidate_mean_degradation_vs_static_le_2pct": bool(
            mean_relative_degradation <= 0.02
        ),
    }
    return {
        "candidate_relative_fid_vs_static": relative_fid,
        "candidate_relative_kid_vs_static": relative_kid,
        "candidate_mean_relative_degradation_vs_static": mean_relative_degradation,
        "predictions": predictions,
        "gate_pass": bool(all(predictions.values())),
        "scope": "single training seed, fixed 1k samples at checkpoint step 5000",
    }


def main() -> None:
    from experiments.evaluate_rae_layerwise_path_generation import (
        NumpyRGBDataset,
        fidelity_metrics,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-count", type=int, default=1000)
    parser.add_argument("--endpoint", type=int, default=5000)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--metric-batch-size", type=int, default=64)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    reference = NumpyRGBDataset(REFERENCE)
    rows = []
    for condition, branch in BRANCHES.items():
        folder_name = (
            f"fixed_seed{SAMPLING_SEED}_n{args.sample_count}_"
            f"step{args.endpoint}_{args.steps}steps"
        )
        sample_npz = (branch / "generation" / folder_name).with_suffix(".npz")
        metrics = fidelity_metrics(
            NumpyRGBDataset(sample_npz),
            reference,
            batch_size=args.metric_batch_size,
        )
        rows.append(
            {
                "condition": condition,
                "branch": branch.name,
                "endpoint": args.endpoint,
                "sample_count": args.sample_count,
                "sampling_seed": SAMPLING_SEED,
                "sampling_steps": args.steps,
                **metrics,
            }
        )
    table = pd.DataFrame(rows)
    result = summarize(table)
    table.to_csv(output / "checkpoint5k_metrics.csv", index=False)
    (output / "checkpoint5k_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(table.to_string(index=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
