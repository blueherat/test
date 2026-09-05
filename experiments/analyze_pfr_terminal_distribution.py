#!/usr/bin/env python3
"""Summarize how PFR query controls change the ADM feature distribution.

The audit deliberately separates the two exact FID terms and adds only
descriptive quantities that can be computed from the already fixed, paired
sample bank.  Generated labels are used to split generated scatter into
within-class and between-class parts; no class-conditioned claim is made for
the unlabeled ImageNet-100 reference archive.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.pfr_counterfactual_residual_theory import terminal_mean_witness


DEFAULT_CONDITIONS = (
    "ordinary_ig",
    "time_only",
    "projected_temporal_parallel",
    "projected_temporal_orthogonal",
    "projected",
)


def parse_conditions(value: str) -> tuple[str, ...]:
    conditions = tuple(item.strip() for item in value.split(",") if item.strip())
    if not conditions or len(conditions) != len(set(conditions)):
        raise argparse.ArgumentTypeError("conditions must be a non-empty unique list")
    return conditions


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def feature_scatter(features: np.ndarray) -> tuple[float, float]:
    """Return covariance trace and participation-ratio effective rank."""

    if features.ndim != 2 or len(features) < 2:
        raise ValueError("features must have shape [N,D] with N >= 2")
    centered = np.asarray(features, dtype=np.float64) - np.mean(features, axis=0)
    denominator = float(len(centered) - 1)
    trace = float(np.square(centered).sum() / denominator)
    gram = centered @ centered.T / denominator
    squared_trace = float(np.square(gram).sum())
    effective_rank = trace * trace / max(squared_trace, 1e-30)
    return trace, effective_rank


def covariance_scatter(covariance: np.ndarray) -> tuple[float, float]:
    covariance = np.asarray(covariance, dtype=np.float64)
    trace = float(np.trace(covariance))
    squared_trace = float(np.square(covariance).sum())
    return trace, trace * trace / max(squared_trace, 1e-30)


def class_scatter(features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """ANOVA scatter of generated features under their requested labels."""

    features64 = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    if features64.ndim != 2 or labels.shape != (len(features64),):
        raise ValueError("features/labels have incompatible shapes")
    total_center = features64.mean(axis=0)
    within_sum = 0.0
    between_sum = 0.0
    nonempty = 0
    within_degrees = 0
    counts = []
    for label in np.unique(labels):
        group = features64[labels == label]
        if not len(group):
            continue
        nonempty += 1
        counts.append(len(group))
        center = group.mean(axis=0)
        within_sum += float(np.square(group - center).sum())
        between_sum += float(len(group) * np.square(center - total_center).sum())
        within_degrees += max(len(group) - 1, 0)
    total_sum = float(np.square(features64 - total_center).sum())
    if not np.isclose(total_sum, within_sum + between_sum, rtol=1e-9, atol=1e-6):
        raise RuntimeError("within/between scatter identity failed")
    return {
        "requested_class_count": float(nonempty),
        "requested_class_min_count": float(min(counts)),
        "requested_class_max_count": float(max(counts)),
        "within_class_trace": within_sum / max(within_degrees, 1),
        "between_class_trace_per_sample": between_sum / max(len(features64) - 1, 1),
        "between_total_scatter_fraction": between_sum / max(total_sum, 1e-30),
    }


def sample_cosine(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first64 = np.asarray(first, dtype=np.float64)
    second64 = np.asarray(second, dtype=np.float64)
    numerator = np.sum(first64 * second64, axis=1)
    denominator = np.linalg.norm(first64, axis=1) * np.linalg.norm(second64, axis=1)
    return numerator / np.maximum(denominator, 1e-30)


def main(args: argparse.Namespace) -> None:
    root = args.root.expanduser().resolve()
    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root
    )
    with np.load(args.reference_stats, allow_pickle=False) as payload:
        reference_trace, reference_effective_rank = covariance_scatter(payload["sigma"])
    with np.load(args.reference_activations, allow_pickle=False) as payload:
        reference_features = np.asarray(payload["pool_3"], dtype=np.float32)
        reference_count = int(len(reference_features))

    activations: dict[str, np.ndarray] = {}
    labels_by_condition: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for condition in args.conditions:
        directory = root / condition
        activation_path = directory / args.activation_name
        metric_path = directory / args.metric_name
        label_path = directory / f"labels_n{args.num_samples}.npy"
        for path in (activation_path, metric_path, label_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        with np.load(activation_path, allow_pickle=False) as payload:
            features = np.asarray(payload["pool_3"], dtype=np.float32)
        labels = np.load(label_path, allow_pickle=False)
        if len(features) != args.num_samples or len(labels) != args.num_samples:
            raise ValueError(f"unexpected sample count for {condition}")
        metrics = load_json(metric_path)
        trace, effective_rank = feature_scatter(features)
        row = {
            "condition": condition,
            "sample_count": len(features),
            "fid": float(metrics["fid"]),
            "fid_mean_component": float(metrics["fid_mean_component"]),
            "fid_covariance_component": float(metrics["fid_covariance_component"]),
            "sfid": float(metrics["sfid"]),
            "inception_score": float(metrics["inception_score"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "feature_covariance_trace": trace,
            "reference_covariance_trace": reference_trace,
            "feature_covariance_trace_ratio": trace / reference_trace,
            "feature_effective_rank": effective_rank,
            "reference_effective_rank": reference_effective_rank,
            **class_scatter(features, labels),
        }
        rows.append(row)
        activations[condition] = features
        labels_by_condition[condition] = labels

    protocol_labels = labels_by_condition[args.conditions[0]]
    for condition in args.conditions[1:]:
        if not np.array_equal(labels_by_condition[condition], protocol_labels):
            raise RuntimeError("conditions do not use identical requested labels")

    baseline = next(row for row in rows if row["condition"] == args.baseline)
    for row in rows:
        row["fid_improvement_vs_baseline"] = baseline["fid"] - row["fid"]
        row["fid_mean_improvement_vs_baseline"] = (
            baseline["fid_mean_component"] - row["fid_mean_component"]
        )
        row["fid_covariance_improvement_vs_baseline"] = (
            baseline["fid_covariance_component"]
            - row["fid_covariance_component"]
        )
        row["precision_change_vs_baseline"] = row["precision"] - baseline["precision"]
        row["recall_change_vs_baseline"] = row["recall"] - baseline["recall"]

    pair_rows: list[dict[str, Any]] = []
    baseline_features = activations[args.baseline]
    shifts = {
        condition: activations[condition] - baseline_features
        for condition in args.conditions
        if condition != args.baseline
    }
    for condition, shift in shifts.items():
        rms = np.sqrt(np.square(shift, dtype=np.float64).mean(axis=1))
        pair_rows.append(
            {
                "condition": condition,
                "reference_condition": args.baseline,
                "paired_feature_shift_rms_mean": float(rms.mean()),
                "paired_feature_shift_rms_median": float(np.median(rms)),
                "paired_feature_shift_rms_q05": float(np.quantile(rms, 0.05)),
                "paired_feature_shift_rms_q95": float(np.quantile(rms, 0.95)),
                "shift_cosine_with_time_only_mean": (
                    1.0
                    if condition == "time_only"
                    else float(np.mean(sample_cosine(shift, shifts["time_only"])))
                ),
                "shift_cosine_with_projected_mean": (
                    1.0
                    if condition == "projected"
                    else float(np.mean(sample_cosine(shift, shifts["projected"])))
                ),
            }
        )

    witness_rows: list[dict[str, Any]] = []
    split_indices = {
        "full": (
            np.arange(args.num_samples),
            np.arange(reference_count),
        ),
        "even_split_half": (
            np.arange(0, args.num_samples, 2),
            np.arange(0, reference_count, 2),
        ),
        "odd_split_half": (
            np.arange(1, args.num_samples, 2),
            np.arange(1, reference_count, 2),
        ),
    }
    for split, (generated_indices, reference_indices) in split_indices.items():
        for condition in args.conditions:
            if condition == args.baseline:
                continue
            witness = terminal_mean_witness(
                reference_features[reference_indices],
                baseline_features[generated_indices],
                activations[condition][generated_indices],
            )
            witness_rows.append(
                {
                    "split": split,
                    "condition": condition,
                    "generated_count": len(generated_indices),
                    "reference_count": len(reference_indices),
                    **witness,
                }
            )

    full_witness = {
        str(row["condition"]): row
        for row in witness_rows
        if row["split"] == "full"
    }
    for row in rows:
        condition = str(row["condition"])
        if condition == args.baseline:
            continue
        if not np.isclose(
            float(row["fid_mean_improvement_vs_baseline"]),
            float(full_witness[condition]["mean_error_improvement"]),
            rtol=1e-5,
            atol=1e-5,
        ):
            raise RuntimeError("feature-mean witness does not match the FID mean term")

    row_by_condition = {str(row["condition"]): row for row in rows}
    fid_component_comparisons: dict[str, Any] | None = None
    if {args.baseline, "time_only", "projected"} <= set(row_by_condition):
        ordinary = row_by_condition[args.baseline]
        time_only = row_by_condition["time_only"]
        projected = row_by_condition["projected"]

        def improvement(first: dict[str, Any], second: dict[str, Any]) -> dict[str, float]:
            total = float(first["fid"] - second["fid"])
            mean = float(first["fid_mean_component"] - second["fid_mean_component"])
            covariance = float(
                first["fid_covariance_component"]
                - second["fid_covariance_component"]
            )
            return {
                "fid_improvement": total,
                "mean_component_improvement": mean,
                "covariance_component_improvement": covariance,
                "mean_fraction": mean / total,
                "covariance_fraction": covariance / total,
            }

        fid_component_comparisons = {
            "time_query_vs_ordinary": improvement(ordinary, time_only),
            "spatial_increment_projected_vs_time_only": improvement(
                time_only, projected
            ),
            "full_projected_vs_ordinary": improvement(ordinary, projected),
        }

    write_csv(output / "terminal_distribution_summary.csv", rows)
    write_csv(output / "paired_feature_response.csv", pair_rows)
    write_csv(output / "terminal_mean_witness.csv", witness_rows)
    summary = {
        "format": "eqvae_pfr_terminal_distribution_audit_v2",
        "source_root": str(root),
        "output_root": str(output),
        "conditions": list(args.conditions),
        "baseline": args.baseline,
        "sample_count": args.num_samples,
        "reference_count": reference_count,
        "reference_has_class_labels": False,
        "quality_scope": (
            "FID terms and ADM precision/recall are reference-backed; the "
            "within/between-class split is descriptive for generated requested labels only."
        ),
        "best_fid": min(rows, key=lambda row: float(row["fid"])),
        "fid_component_comparisons": fid_component_comparisons,
        "terminal_mean_witness": {
            "identity": (
                "||mu_ref-mu_base||^2-||mu_ref-mu_candidate||^2 "
                "=2<mu_ref-mu_base,mu_candidate-mu_base>"
                "-||mu_candidate-mu_base||^2"
            ),
            "positive_mean_witness_on_both_disjoint_halves": all(
                float(row["mean_error_improvement"]) > 0.0
                for row in witness_rows
                if row["split"] != "full"
            ),
        },
        "tables": {
            "distribution": str(output / "terminal_distribution_summary.csv"),
            "paired_response": str(output / "paired_feature_response.csv"),
            "terminal_mean_witness": str(output / "terminal_mean_witness.csv"),
        },
    }
    atomic_json(output / "terminal_distribution_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reference-stats", type=Path, required=True)
    parser.add_argument("--reference-activations", type=Path, required=True)
    parser.add_argument("--conditions", type=parse_conditions, default=DEFAULT_CONDITIONS)
    parser.add_argument("--baseline", default="ordinary_ig")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--activation-name", default="adm_activations.npz")
    parser.add_argument("--metric-name", default="adm_distribution_metrics.json")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
