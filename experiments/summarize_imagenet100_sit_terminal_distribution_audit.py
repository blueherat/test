#!/usr/bin/env python3
"""Aggregate the two-seed SiT terminal-distribution audit."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from experiments.train_imagenet100_sit_flow import atomic_json_dump
except ModuleNotFoundError:
    from train_imagenet100_sit_flow import atomic_json_dump


DEFAULT_ROOT = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/"
    "terminal_distribution_audit_800k_v1"
)
TABLES = {
    "quality": ("endpoint_quality.csv", ["condition"]),
    "path": ("trajectory_path_summary.csv", ["condition"]),
    "action": (
        "integrated_action_summary.csv",
        ["condition", "mode", "gamma", "response_scale"],
    ),
    "diagnostics": (
        "diagnostics_by_time.csv",
        ["condition", "mode", "gamma", "response_scale", "time"],
    ),
    "latent_pairwise": (
        "latent_distribution_pairwise.csv",
        ["time", "condition_a", "condition_b", "comparison"],
    ),
    "feature_pairwise": (
        "endpoint_feature_pairwise.csv",
        ["condition_a", "condition_b", "comparison"],
    ),
    "equivalence": (
        "grouped_individual_equivalence.csv",
        ["condition", "branch_role"],
    ),
}


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item) for item in value.split(",") if item.strip())
    if not seeds or len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("seeds must be a non-empty unique list")
    return seeds


def _load_table(root: Path, seeds: tuple[int, ...], filename: str) -> pd.DataFrame:
    frames = []
    for seed in seeds:
        path = root / f"seed{seed}" / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame.insert(0, "seed", seed)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _aggregate(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    missing = [key for key in keys if key not in frame]
    if missing:
        raise ValueError(f"missing grouping columns: {missing}")
    numeric = [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column not in {*keys, "seed"}
    ]
    grouped = frame.groupby(keys, dropna=False, sort=False)
    if not numeric:
        return grouped.size().rename("row_count").reset_index()
    output = grouped[numeric].agg(["mean", "std"]).reset_index()
    output.columns = [
        column if isinstance(column, str) else column[0]
        if not column[1]
        else f"{column[0]}_{column[1]}"
        for column in output.columns
    ]
    output["seed_count"] = grouped["seed"].nunique().to_numpy()
    return output


def _add_quality_improvements(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    baseline = output[output["condition"] == "baseline"].set_index("seed")
    if len(baseline) != output["seed"].nunique():
        raise ValueError("each seed must contain exactly one baseline quality row")
    for metric, higher_is_better in (
        ("fid", False),
        ("sfid", False),
        ("inception_score", True),
    ):
        reference = output["seed"].map(baseline[metric])
        output[f"{metric}_improvement_vs_baseline"] = (
            output[metric] - reference if higher_is_better else reference - output[metric]
        )
    return output


def _add_null_calibration(frame: pd.DataFrame, *, has_time: bool) -> pd.DataFrame:
    output = frame.copy()
    identity_keys = (["seed"] if "seed" in output else []) + (
        ["time"] if has_time else []
    ) + ["condition_a"]
    within = output[output["comparison"] == "within_condition_split"].copy()
    within = within.set_index(identity_keys)
    metric_columns = [
        column
        for column in output.columns
        if column.endswith("_mean")
        and column.startswith(
            ("swd_", "linear_mmd2_", "feature_frechet_", "c2st_auc_")
        )
    ]
    for metric in metric_columns:
        calibrated = np.full(len(output), np.nan, dtype=np.float64)
        for row_index, row in output.iterrows():
            if row["comparison"] != "cross_condition":
                continue
            prefix = ((row["seed"],) if "seed" in output else ()) + (
                (row["time"],) if has_time else ()
            )
            key_a = prefix + (row["condition_a"],)
            key_b = prefix + (row["condition_b"],)
            null = 0.5 * (float(within.loc[key_a, metric]) + float(within.loc[key_b, metric]))
            calibrated[row_index] = float(row[metric]) - null
        output[f"{metric}_excess_over_split_null"] = calibrated
    return output


def _condition_summary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    quality = tables["quality"]
    action = tables["action"]
    path = tables["path"]
    selected_action = [
        "seed",
        "condition",
        *[column for column in action if column.endswith("_mean")],
    ]
    selected_path = [
        "seed",
        "condition",
        *[column for column in path if column.endswith("_mean")],
    ]
    return (
        quality.merge(action[selected_action], on=["seed", "condition"], how="left")
        .merge(path[selected_path], on=["seed", "condition"], how="left")
        .sort_values(["seed", "fid"])
        .reset_index(drop=True)
    )


def _paired_feature_rows(root: Path, seeds: tuple[int, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed in seeds:
        seed_root = root / f"seed{seed}"
        manifest = json.loads((seed_root / "manifest.json").read_text(encoding="utf-8"))
        branches = tuple(manifest["branches"])
        activations = {}
        for branch in branches:
            with np.load(seed_root / "adm_activations" / f"{branch}.npz") as payload:
                activations[branch] = np.asarray(payload["pool_3"], dtype=np.float32)
        for first, second in itertools.combinations(branches, 2):
            if activations[first].shape != activations[second].shape:
                raise ValueError("paired ADM activation shapes differ")
            difference = activations[first] - activations[second]
            rms = np.sqrt(np.square(difference, dtype=np.float64).mean(axis=1))
            rows.append(
                {
                    "seed": seed,
                    "condition_a": first,
                    "condition_b": second,
                    "feature_paired_rms_mean": float(rms.mean()),
                    "feature_paired_rms_std": float(rms.std(ddof=1)),
                    "feature_paired_rms_q05": float(np.quantile(rms, 0.05)),
                    "feature_paired_rms_median": float(np.quantile(rms, 0.5)),
                    "feature_paired_rms_q95": float(np.quantile(rms, 0.95)),
                }
            )
    return pd.DataFrame(rows)


def main(args: argparse.Namespace) -> None:
    root = args.root.expanduser().resolve()
    seeds = tuple(args.seeds)
    tables: dict[str, pd.DataFrame] = {}
    aggregates: dict[str, pd.DataFrame] = {}
    for name, (filename, keys) in TABLES.items():
        frame = _load_table(root, seeds, filename)
        if name == "quality":
            frame = _add_quality_improvements(frame)
        elif name == "latent_pairwise":
            frame = _add_null_calibration(frame, has_time=True)
        elif name == "feature_pairwise":
            frame = _add_null_calibration(frame, has_time=False)
        tables[name] = frame
        aggregates[name] = _aggregate(frame, keys)
        frame.to_csv(root / f"combined_{name}.csv", index=False)
        aggregates[name].to_csv(root / f"aggregate_{name}.csv", index=False)

    paired_feature = _paired_feature_rows(root, seeds)
    tables["paired_feature"] = paired_feature
    aggregates["paired_feature"] = _aggregate(
        paired_feature,
        ["condition_a", "condition_b"],
    )
    paired_feature.to_csv(root / "combined_paired_feature.csv", index=False)
    aggregates["paired_feature"].to_csv(
        root / "aggregate_paired_feature.csv",
        index=False,
    )

    condition_summary = _condition_summary(tables)
    condition_summary.to_csv(root / "combined_condition_summary.csv", index=False)
    condition_aggregate = _aggregate(condition_summary, ["condition"])
    condition_aggregate.to_csv(root / "aggregate_condition_summary.csv", index=False)
    manifest = {
        "format": "eqvae_imagenet100_sit_terminal_distribution_summary_v1",
        "root": str(root),
        "seeds": list(seeds),
        "tables": {
            name: {
                "combined_rows": int(len(tables[name])),
                "aggregate_rows": int(len(aggregates[name])),
            }
            for name in tables
        },
        "condition_rows": int(len(condition_summary)),
    }
    atomic_json_dump(manifest, root / "summary_manifest.json")
    (root / "SUMMARY_COMPLETE").touch()
    print(json.dumps(manifest, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--seeds", type=_parse_seeds, default=(0, 1))
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
