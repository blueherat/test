"""Combine prospective rollout-selection runs and audit their evidence."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.small_image_rollout_checkpoint_selection import (  # noqa: E402
    evaluate_gates,
    summarize_paired_metrics,
)


DEFAULT_OUTPUT_ROOT = (
    Path.home() / "data/eqvae/experiments/small_image_rollout_checkpoint_selection"
)


def _require_unique(frame: pd.DataFrame, keys: Sequence[str], name: str) -> None:
    duplicated = frame.duplicated(list(keys), keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, list(keys)].head(5).to_dict("records")
        raise ValueError(f"{name} has duplicate keys: {examples}")


def proxy_fid_alignment(
    history: pd.DataFrame,
    seed_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = history[history["selected"]].copy()
    _require_unique(
        selected,
        ("dataset", "training_seed", "variant"),
        "selected proxy rows",
    )
    final = history[history["step"].eq(1000)].copy()
    _require_unique(
        final,
        ("dataset", "training_seed", "variant"),
        "final proxy rows",
    )
    proxy = selected[
        ["dataset", "training_seed", "variant", "step", "proxy_loss"]
    ].rename(columns={"step": "selected_step", "proxy_loss": "selected_proxy"})
    proxy = proxy.merge(
        final[["dataset", "training_seed", "variant", "proxy_loss"]].rename(
            columns={"proxy_loss": "final_proxy"}
        ),
        on=["dataset", "training_seed", "variant"],
        validate="one_to_one",
    )
    proxy["proxy_ratio"] = proxy["selected_proxy"] / proxy["final_proxy"].clip(
        lower=1e-12
    )
    fid = seed_summary[
        seed_summary["metric"].eq("feature_fid")
        & seed_summary["comparison"].eq("selected_vs_final")
    ][["dataset", "training_seed", "basis", "ratio"]].rename(
        columns={"basis": "variant", "ratio": "fid_ratio"}
    )
    rows = proxy.merge(
        fid,
        on=["dataset", "training_seed", "variant"],
        validate="one_to_one",
    )

    def spearman(group: pd.DataFrame) -> float:
        if group["proxy_ratio"].nunique() < 2 or group["fid_ratio"].nunique() < 2:
            return float("nan")
        return float(group["proxy_ratio"].corr(group["fid_ratio"], method="spearman"))

    summary_rows = []
    for dataset, group in rows.groupby("dataset", sort=True):
        summary_rows.append(
            {
                "dataset": dataset,
                "conditions": len(group),
                "proxy_ratio_mean": float(group["proxy_ratio"].mean()),
                "fid_ratio_mean": float(group["fid_ratio"].mean()),
                "proxy_fid_spearman": spearman(group),
            }
        )
    summary_rows.append(
        {
            "dataset": "all",
            "conditions": len(rows),
            "proxy_ratio_mean": float(rows["proxy_ratio"].mean()),
            "fid_ratio_mean": float(rows["fid_ratio"].mean()),
            "proxy_fid_spearman": spearman(rows),
        }
    )
    return rows, pd.DataFrame(summary_rows)


def semantic_selection_diagnostic(rollout: pd.DataFrame) -> pd.DataFrame:
    required = {
        "dataset",
        "training_seed",
        "evaluation_seed",
        "variant",
        "classifier_confidence",
        "class_entropy",
    }
    missing = required - set(rollout.columns)
    if missing:
        raise ValueError(f"rollout is missing semantic columns: {sorted(missing)}")
    rows = []
    target_entropy = math.log(10.0)
    for dataset, dataset_rows in rollout.groupby("dataset", sort=True):
        for variant in ("baseline", "dct", "pca", "random"):
            selected = dataset_rows[dataset_rows["variant"].eq(f"{variant}_selected")]
            final = dataset_rows[dataset_rows["variant"].eq(f"{variant}_final")]
            if len(selected) != len(final) or selected.empty:
                raise ValueError(f"unpaired semantic rows for {dataset}/{variant}")
            selected_confidence = float(selected["classifier_confidence"].mean())
            final_confidence = float(final["classifier_confidence"].mean())
            selected_entropy = float(selected["class_entropy"].mean())
            final_entropy = float(final["class_entropy"].mean())
            rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "classifier_confidence_selected": selected_confidence,
                    "classifier_confidence_final": final_confidence,
                    "classifier_confidence_delta": (
                        selected_confidence - final_confidence
                    ),
                    "class_entropy_gap_selected": abs(
                        target_entropy - selected_entropy
                    ),
                    "class_entropy_gap_final": abs(target_entropy - final_entropy),
                    "class_entropy_gap_change": abs(
                        target_entropy - selected_entropy
                    )
                    - abs(target_entropy - final_entropy),
                }
            )
    return pd.DataFrame(rows)


def combine_results(
    result_dirs: Sequence[Path],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, object],
    dict[str, object],
]:
    if len(result_dirs) < 2:
        raise ValueError("at least two result directories are required")
    paired_frames = []
    history_frames = []
    rollout_frames = []
    metadata = []
    configs = []
    for raw_dir in result_dirs:
        result_dir = raw_dir.expanduser().resolve()
        paired_frames.append(pd.read_csv(result_dir / "paired_metrics.csv"))
        history_frames.append(pd.read_csv(result_dir / "selection_history.csv"))
        rollout_frames.append(pd.read_csv(result_dir / "rollout_metrics.csv"))
        metadata.extend(json.loads((result_dir / "metadata.json").read_text()))
        configs.append(json.loads((result_dir / "config.json").read_text()))
    paired = pd.concat(paired_frames, ignore_index=True)
    history = pd.concat(history_frames, ignore_index=True)
    rollout = pd.concat(rollout_frames, ignore_index=True)
    paired_keys = (
        "dataset",
        "training_seed",
        "evaluation_seed",
        "basis",
        "comparison",
        "metric",
    )
    _require_unique(paired, paired_keys, "combined paired metrics")
    _require_unique(
        history,
        ("dataset", "training_seed", "variant", "step"),
        "combined selection history",
    )
    _require_unique(
        rollout,
        ("dataset", "training_seed", "evaluation_seed", "variant"),
        "combined rollout metrics",
    )
    seed_summary, aggregate = summarize_paired_metrics(paired)
    gates = evaluate_gates(aggregate)
    proxy_rows, proxy_summary = proxy_fid_alignment(history, seed_summary)
    semantic_summary = semantic_selection_diagnostic(rollout)
    expected_seeds = set(range(5, 10))
    integrity = {
        "datasets": sorted(paired["dataset"].unique().tolist()),
        "training_seeds_by_dataset": {
            dataset: sorted(group["training_seed"].unique().tolist())
            for dataset, group in paired.groupby("dataset")
        },
        "expected_training_seeds_present": all(
            set(group["training_seed"].unique()) == expected_seeds
            for _, group in paired.groupby("dataset")
        ),
        "evaluation_seed_count": int(paired["evaluation_seed"].nunique()),
        "metadata_rows": len(metadata),
        "all_update_selection_overlaps_zero": all(
            int(item["update_selection_overlap"]) == 0 for item in metadata
        ),
        "classifier_accuracy": {
            dataset: {
                "mean": float(group["classifier_accuracy"].mean()),
                "min": float(group["classifier_accuracy"].min()),
                "max": float(group["classifier_accuracy"].max()),
            }
            for dataset, group in pd.DataFrame(metadata).groupby("dataset")
        },
        "source_configs": configs,
    }
    if sorted(integrity["datasets"]) != ["fashion_mnist", "mnist"]:
        raise ValueError(f"expected both datasets, got {integrity['datasets']}")
    if not integrity["expected_training_seeds_present"]:
        raise ValueError("combined evidence does not contain exactly seeds 5--9")
    if integrity["evaluation_seed_count"] != 3:
        raise ValueError("combined evidence must contain three evaluation seeds")
    if not integrity["all_update_selection_overlaps_zero"]:
        raise ValueError("data leakage detected in metadata")
    return (
        seed_summary,
        aggregate,
        proxy_rows,
        proxy_summary,
        semantic_summary,
        gates,
        integrity,
    )


def save_combined(
    result_dirs: Sequence[Path],
    output_root: Path,
) -> Path:
    (
        seed_summary,
        aggregate,
        proxy_rows,
        proxy_summary,
        semantic_summary,
        gates,
        integrity,
    ) = combine_results(result_dirs)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root.expanduser() / f"combined_prospective_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    seed_summary.to_csv(output_dir / "seed_summary.csv", index=False)
    aggregate.to_csv(output_dir / "aggregate_summary.csv", index=False)
    proxy_rows.to_csv(output_dir / "proxy_fid_rows.csv", index=False)
    proxy_summary.to_csv(output_dir / "proxy_fid_summary.csv", index=False)
    semantic_summary.to_csv(output_dir / "semantic_summary.csv", index=False)
    (output_dir / "gates.json").write_text(
        json.dumps(gates, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "integrity.json").write_text(
        json.dumps(integrity, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "sources.json").write_text(
        json.dumps([str(path.expanduser().resolve()) for path in result_dirs], indent=2)
        + "\n",
        encoding="utf-8",
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dirs", nargs="+", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    result_dir = save_combined(args.result_dirs, args.output_root)
    aggregate = pd.read_csv(result_dir / "aggregate_summary.csv")
    print(
        aggregate[
            aggregate["metric"].eq("feature_fid")
            & aggregate["comparison"].isin(
                ["selected_vs_final", "selected_vs_selected_baseline"]
            )
        ].round(4).to_string(index=False)
    )
    print((result_dir / "gates.json").read_text(encoding="utf-8"))
    print(f"result_dir={result_dir}")


if __name__ == "__main__":
    main()
