#!/usr/bin/env python3
"""Conservative secondary statistics for CFG-Rejection discovery runs.

The input CSV is produced by ``score_cfg_rejection_edm2_imagenet.py``.  Tail
membership is treated as fixed by that upstream run: for every ASD metric,
``low`` is the CFG-Rejection-suspicious tail and ``high`` is the comparison
tail.  The estimand gives every ImageNet class equal weight.  Uncertainty is
estimated with a reproducible nonparametric bootstrap over seed clusters, so
the same initial-noise seed remains coupled across all classes.

This accepts only the locked ``pilot`` and ``paper-10k`` reproduction
protocols.  It is deliberately a secondary, exploratory semantic-consistency analysis.
An ImageNet classifier is not an artifact label, and this script reports no
p-values, significance declarations, or post-hoc winner selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


EVIDENCE_METRICS = (
    "official_notebook_metric_tau5",
    "denoiser_asd_tau5",
    "denoiser_asd_full",
    "score_asd_tau5",
    "score_asd_full",
)
TAIL_VALUES = {"low", "middle", "high"}
DEFAULT_BOOTSTRAP_SEED = 20_260_826
DEFAULT_BOOTSTRAP_REPLICATES = 10_000


@dataclass(frozen=True)
class ClassifierRecord:
    class_id: int
    seed: int
    target_probability: float
    top1_error: float
    tails: dict[str, str]

    @property
    def key(self) -> tuple[int, int]:
        return self.class_id, self.seed


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
    os.replace(temporary, path)


def atomic_csv_dump(rows: Sequence[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError("cannot write an empty statistics CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_manifest(path: Path) -> tuple[dict[str, Any], tuple[int, ...], tuple[int, ...]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing pilot manifest: {path}")
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"manifest must contain a JSON object: {path}")
    if manifest.get("experiment") != "cfg_rejection_edm2_reproduction":
        raise ValueError(
            "unexpected manifest experiment: "
            f"{manifest.get('experiment')!r}; expected 'cfg_rejection_edm2_reproduction'"
        )
    allowed_protocols = {"pilot", "paper-10k"}
    if manifest.get("protocol") not in allowed_protocols:
        raise ValueError(
            "this analysis is intentionally restricted to locked discovery protocols "
            f"{sorted(allowed_protocols)}; "
            f"manifest protocol is {manifest.get('protocol')!r}"
        )

    raw_classes = manifest.get("class_ids")
    raw_seeds = manifest.get("seeds")
    if not isinstance(raw_classes, list) or not raw_classes:
        raise ValueError("manifest class_ids must be a non-empty list")
    if not isinstance(raw_seeds, list) or not raw_seeds:
        raise ValueError("manifest seeds must be a non-empty list")
    class_ids = tuple(int(value) for value in raw_classes)
    seeds = tuple(int(value) for value in raw_seeds)
    if len(set(class_ids)) != len(class_ids):
        raise ValueError("manifest class_ids contains duplicates")
    if len(set(seeds)) != len(seeds):
        raise ValueError("manifest seeds contains duplicates")
    if len(class_ids) < 2:
        raise ValueError("leave-one-class-out analysis requires at least two classes")

    expected_count = len(class_ids) * len(seeds)
    declared_count = manifest.get("sample_count")
    if declared_count is not None and int(declared_count) != expected_count:
        raise ValueError(
            "manifest sample_count does not match its class/seed Cartesian product: "
            f"declared={declared_count}, expected={expected_count}"
        )
    return manifest, class_ids, seeds


def parse_probability(raw: str, *, field: str, line: int) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"CSV line {line}: {field} is not numeric: {raw!r}") from exc
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"CSV line {line}: {field} must be finite and in [0, 1], got {value}")
    return value


def parse_binary(raw: str, *, field: str, line: int) -> bool:
    normalized = raw.strip().lower()
    if normalized in {"1", "true"}:
        return True
    if normalized in {"0", "false"}:
        return False
    raise ValueError(f"CSV line {line}: {field} must be binary, got {raw!r}")


def load_classifier_csv(
    path: Path,
    *,
    class_ids: Sequence[int],
    seeds: Sequence[int],
) -> list[ClassifierRecord]:
    if not path.is_file():
        raise FileNotFoundError(f"missing classifier CSV: {path}")
    required_fields = {
        "class_id",
        "seed",
        "target_probability",
        "top1_correct",
        *EVIDENCE_METRICS,
        *(f"tail_{metric}" for metric in EVIDENCE_METRICS),
    }
    records: list[ClassifierRecord] = []
    seen: set[tuple[int, int]] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_fields = sorted(required_fields - set(reader.fieldnames or ()))
        if missing_fields:
            raise ValueError(f"classifier CSV is missing fields: {missing_fields}")
        for line, row in enumerate(reader, start=2):
            try:
                class_id = int(row["class_id"])
                seed = int(row["seed"])
            except ValueError as exc:
                raise ValueError(f"CSV line {line}: class_id and seed must be integers") from exc
            key = (class_id, seed)
            if key in seen:
                raise ValueError(f"CSV line {line}: duplicate class/seed key {key}")
            seen.add(key)

            target_probability = parse_probability(
                row["target_probability"], field="target_probability", line=line
            )
            top1_correct = parse_binary(row["top1_correct"], field="top1_correct", line=line)
            tails: dict[str, str] = {}
            for metric in EVIDENCE_METRICS:
                try:
                    evidence = float(row[metric])
                except ValueError as exc:
                    raise ValueError(
                        f"CSV line {line}: evidence {metric} is not numeric"
                    ) from exc
                if not math.isfinite(evidence):
                    raise ValueError(f"CSV line {line}: evidence {metric} is non-finite")
                tail = row[f"tail_{metric}"].strip().lower()
                if tail not in TAIL_VALUES:
                    raise ValueError(
                        f"CSV line {line}: tail_{metric} must be one of "
                        f"{sorted(TAIL_VALUES)}, got {tail!r}"
                    )
                tails[metric] = tail
            records.append(
                ClassifierRecord(
                    class_id=class_id,
                    seed=seed,
                    target_probability=target_probability,
                    top1_error=float(not top1_correct),
                    tails=tails,
                )
            )

    expected_keys = {(class_id, seed) for class_id in class_ids for seed in seeds}
    observed_keys = {record.key for record in records}
    missing = sorted(expected_keys - observed_keys)
    unexpected = sorted(observed_keys - expected_keys)
    if missing or unexpected:
        raise ValueError(
            "classifier CSV does not exactly match the pilot manifest: "
            f"missing={len(missing)} (first {missing[:5]}), "
            f"unexpected={len(unexpected)} (first {unexpected[:5]})"
        )
    records.sort(key=lambda record: (record.class_id, record.seed))
    return records


def build_matrices(
    records: Sequence[ClassifierRecord],
    *,
    class_ids: Sequence[int],
    seeds: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    class_to_index = {class_id: index for index, class_id in enumerate(class_ids)}
    seed_to_index = {seed: index for index, seed in enumerate(seeds)}
    shape = (len(class_ids), len(seeds))
    target_probability = np.full(shape, np.nan, dtype=np.float64)
    top1_error = np.full(shape, np.nan, dtype=np.float64)
    tails = {
        metric: np.full(shape, -1, dtype=np.int8)
        for metric in EVIDENCE_METRICS
    }
    tail_code = {"low": 0, "middle": 1, "high": 2}
    for record in records:
        row = class_to_index[record.class_id]
        column = seed_to_index[record.seed]
        target_probability[row, column] = record.target_probability
        top1_error[row, column] = record.top1_error
        for metric in EVIDENCE_METRICS:
            tails[metric][row, column] = tail_code[record.tails[metric]]
    if not np.isfinite(target_probability).all() or not np.isfinite(top1_error).all():
        raise RuntimeError("internal matrix construction left missing classifier values")
    if any((assignment < 0).any() for assignment in tails.values()):
        raise RuntimeError("internal matrix construction left missing tail assignments")
    return target_probability, top1_error, tails


def validate_symmetric_tails(
    assignments: np.ndarray,
    *,
    metric: str,
    class_ids: Sequence[int],
) -> dict[str, Any]:
    low_counts = (assignments == 0).sum(axis=1)
    high_counts = (assignments == 2).sum(axis=1)
    if np.any(low_counts == 0) or np.any(high_counts == 0):
        raise ValueError(f"metric {metric} has an empty low or high class tail")
    if not np.array_equal(low_counts, high_counts):
        details = {
            str(class_id): {"low": int(low), "high": int(high)}
            for class_id, low, high in zip(class_ids, low_counts, high_counts, strict=True)
        }
        raise ValueError(f"metric {metric} does not have symmetric class tails: {details}")
    return {
        "low_count_total": int(low_counts.sum()),
        "high_count_total": int(high_counts.sum()),
        "per_class": {
            str(class_id): {"low": int(low), "high": int(high)}
            for class_id, low, high in zip(class_ids, low_counts, high_counts, strict=True)
        },
    }


def class_point_statistics(
    target_probability: np.ndarray,
    top1_error: np.ndarray,
    assignments: np.ndarray,
) -> np.ndarray:
    """Return class x [target-low, target-high, error-low, error-high]."""

    class_count = target_probability.shape[0]
    output = np.empty((class_count, 4), dtype=np.float64)
    for class_index in range(class_count):
        low = assignments[class_index] == 0
        high = assignments[class_index] == 2
        output[class_index] = (
            target_probability[class_index, low].mean(),
            target_probability[class_index, high].mean(),
            top1_error[class_index, low].mean(),
            top1_error[class_index, high].mean(),
        )
    return output


def bootstrap_class_statistics(
    target_probability: np.ndarray,
    top1_error: np.ndarray,
    assignments: np.ndarray,
    *,
    repetitions: int,
    seed: int,
    batch_size: int = 2_048,
) -> tuple[np.ndarray, int]:
    """Bootstrap seed clusters while retaining fixed class strata and tails."""

    class_count, seed_count = target_probability.shape
    low = assignments == 0
    high = assignments == 2
    low_float = low.astype(np.float64)
    high_float = high.astype(np.float64)
    target_low = np.where(low, target_probability, 0.0)
    target_high = np.where(high, target_probability, 0.0)
    error_low = np.where(low, top1_error, 0.0)
    error_high = np.where(high, top1_error, 0.0)

    rng = np.random.default_rng(seed)
    accepted: list[np.ndarray] = []
    accepted_count = 0
    rejected_count = 0
    maximum_draws = max(repetitions * 20, repetitions + 1_000)
    total_draws = 0
    probabilities = np.full(seed_count, 1.0 / seed_count, dtype=np.float64)
    while accepted_count < repetitions:
        requested = min(batch_size, repetitions - accepted_count + 64)
        weights = rng.multinomial(seed_count, probabilities, size=requested).astype(
            np.float64, copy=False
        )
        total_draws += requested
        low_denominator = weights @ low_float.T
        high_denominator = weights @ high_float.T
        valid = (low_denominator > 0).all(axis=1) & (high_denominator > 0).all(axis=1)
        rejected_count += int((~valid).sum())
        if valid.any():
            valid_weights = weights[valid]
            low_denominator = low_denominator[valid]
            high_denominator = high_denominator[valid]
            values = np.stack(
                (
                    (valid_weights @ target_low.T) / low_denominator,
                    (valid_weights @ target_high.T) / high_denominator,
                    (valid_weights @ error_low.T) / low_denominator,
                    (valid_weights @ error_high.T) / high_denominator,
                ),
                axis=2,
            )
            remaining = repetitions - accepted_count
            values = values[:remaining]
            accepted.append(values)
            accepted_count += values.shape[0]
        if total_draws > maximum_draws and accepted_count < repetitions:
            raise RuntimeError(
                "too many cluster-bootstrap draws had empty class-tail cells; "
                "increase tail size or inspect the pilot design"
            )
    result = np.concatenate(accepted, axis=0)
    if result.shape != (repetitions, class_count, 4):
        raise RuntimeError(f"unexpected bootstrap shape: {result.shape}")
    return result, rejected_count


def interval_summary(
    estimate: float,
    replicates: np.ndarray,
    *,
    confidence_level: float,
) -> dict[str, float]:
    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(replicates, (tail, 1.0 - tail))
    return {
        "estimate": float(estimate),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
    }


def summarize_scope(
    point_by_class: np.ndarray,
    bootstrap_by_class: np.ndarray,
    *,
    included_class_indices: Sequence[int],
    assignments: np.ndarray,
    confidence_level: float,
) -> dict[str, Any]:
    indices = np.asarray(included_class_indices, dtype=np.int64)
    point = point_by_class[indices].mean(axis=0)
    bootstrap = bootstrap_by_class[:, indices, :].mean(axis=1)

    target_probability_contrast = point[1] - point[0]
    target_probability_contrast_bootstrap = bootstrap[:, 1] - bootstrap[:, 0]
    top1_error_contrast = point[2] - point[3]
    top1_error_contrast_bootstrap = bootstrap[:, 2] - bootstrap[:, 3]
    selected_assignments = assignments[indices]
    return {
        "class_count": int(indices.size),
        "low_tail_count": int((selected_assignments == 0).sum()),
        "high_tail_count": int((selected_assignments == 2).sum()),
        "target_probability": {
            "low": interval_summary(
                point[0], bootstrap[:, 0], confidence_level=confidence_level
            ),
            "high": interval_summary(
                point[1], bootstrap[:, 1], confidence_level=confidence_level
            ),
            "high_minus_low": interval_summary(
                target_probability_contrast,
                target_probability_contrast_bootstrap,
                confidence_level=confidence_level,
            ),
            "direction_if_low_asd_is_worse": "high_minus_low should be positive",
        },
        "top1_error_rate": {
            "low": interval_summary(
                point[2], bootstrap[:, 2], confidence_level=confidence_level
            ),
            "high": interval_summary(
                point[3], bootstrap[:, 3], confidence_level=confidence_level
            ),
            "low_minus_high": interval_summary(
                top1_error_contrast,
                top1_error_contrast_bootstrap,
                confidence_level=confidence_level,
            ),
            "direction_if_low_asd_is_worse": "low_minus_high should be positive",
        },
    }


def flatten_result_row(
    *,
    metric: str,
    scope: str,
    omitted_class_id: int | None,
    summary: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "metric": metric,
        "scope": scope,
        "omitted_class_id": "" if omitted_class_id is None else omitted_class_id,
        "class_count": summary["class_count"],
        "low_tail_count": summary["low_tail_count"],
        "high_tail_count": summary["high_tail_count"],
    }
    for outcome, contrast in (
        ("target_probability", "high_minus_low"),
        ("top1_error_rate", "low_minus_high"),
    ):
        for component in ("low", "high", contrast):
            for field in ("estimate", "ci_lower", "ci_upper"):
                row[f"{outcome}_{component}_{field}"] = summary[outcome][component][field]
    return row


def analyze(
    *,
    classifier_csv: Path,
    manifest_path: Path,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest, class_ids, seeds = load_manifest(manifest_path)
    records = load_classifier_csv(classifier_csv, class_ids=class_ids, seeds=seeds)
    target_probability, top1_error, tail_assignments = build_matrices(
        records, class_ids=class_ids, seeds=seeds
    )

    results: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    for metric in EVIDENCE_METRICS:
        assignments = tail_assignments[metric]
        tail_audit = validate_symmetric_tails(
            assignments, metric=metric, class_ids=class_ids
        )
        point_by_class = class_point_statistics(
            target_probability, top1_error, assignments
        )
        bootstrap_by_class, rejected_bootstrap_draws = bootstrap_class_statistics(
            target_probability,
            top1_error,
            assignments,
            repetitions=bootstrap_repetitions,
            seed=bootstrap_seed,
        )
        all_indices = tuple(range(len(class_ids)))
        full = summarize_scope(
            point_by_class,
            bootstrap_by_class,
            included_class_indices=all_indices,
            assignments=assignments,
            confidence_level=confidence_level,
        )
        leave_one_out: dict[str, Any] = {}
        csv_rows.append(
            flatten_result_row(
                metric=metric,
                scope="all_classes_equal_weight",
                omitted_class_id=None,
                summary=full,
            )
        )
        for omitted_index, omitted_class_id in enumerate(class_ids):
            retained = tuple(index for index in all_indices if index != omitted_index)
            omitted_summary = summarize_scope(
                point_by_class,
                bootstrap_by_class,
                included_class_indices=retained,
                assignments=assignments,
                confidence_level=confidence_level,
            )
            leave_one_out[str(omitted_class_id)] = omitted_summary
            csv_rows.append(
                flatten_result_row(
                    metric=metric,
                    scope="leave_one_class_out",
                    omitted_class_id=omitted_class_id,
                    summary=omitted_summary,
                )
            )

        target_contrasts = [
            summary["target_probability"]["high_minus_low"]["estimate"]
            for summary in leave_one_out.values()
        ]
        error_contrasts = [
            summary["top1_error_rate"]["low_minus_high"]["estimate"]
            for summary in leave_one_out.values()
        ]
        results[metric] = {
            "tail_audit": tail_audit,
            "bootstrap_rejected_draw_count": rejected_bootstrap_draws,
            "all_classes_equal_weight": full,
            "leave_one_class_out": leave_one_out,
            "leave_one_class_out_directional_contrast_range": {
                "target_probability_high_minus_low_min": float(min(target_contrasts)),
                "target_probability_high_minus_low_max": float(max(target_contrasts)),
                "top1_error_rate_low_minus_high_min": float(min(error_contrasts)),
                "top1_error_rate_low_minus_high_max": float(max(error_contrasts)),
            },
        }

    script_path = Path(__file__).resolve()
    payload = {
        "schema_version": 1,
        "analysis": "cfg_rejection_classifier_discovery_secondary_statistics",
        "analysis_status": {
            "role": "secondary exploratory semantic-consistency diagnostic",
            "not_primary_evidence": True,
            "no_p_values_or_significance_calls": True,
            "statement": (
                "ConvNeXt ImageNet target probability and top-1 error are automatic "
                "semantic proxies. They are not human artifact labels and cannot by "
                "themselves validate the proposed cross-scale e-process method."
            ),
        },
        "inputs": {
            "classifier_csv": str(classifier_csv),
            "classifier_csv_sha256": sha256_file(classifier_csv),
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "manifest_protocol": manifest.get("protocol"),
            "sample_count": len(records),
            "class_ids": list(class_ids),
            "seed_count": len(seeds),
        },
        "script": {"path": str(script_path), "sha256": sha256_file(script_path)},
        "estimand": {
            "tail_direction": (
                "CFG-Rejection convention: LOW ASD is the suspicious/worse tail; "
                "HIGH ASD is the comparison tail."
            ),
            "class_stratification": (
                "Compute low and high means separately inside every class, then give "
                "each manifest class equal weight."
            ),
            "seed_clustering": (
                "Resample manifest seed IDs as whole clusters; every selected seed "
                "retains all of its class-conditioned samples and tail memberships."
            ),
            "target_probability_directional_contrast": "high minus low",
            "top1_error_rate_directional_contrast": "low minus high",
        },
        "bootstrap": {
            "method": "nonparametric percentile bootstrap over seed clusters",
            "repetitions": bootstrap_repetitions,
            "rng": "numpy.random.Generator(PCG64)",
            "seed": bootstrap_seed,
            "confidence_level": confidence_level,
            "classes": "fixed strata, equal weight",
            "tail_membership": (
                "fixed from the upstream classifier CSV; the intervals are conditional "
                "on the observed ranking and do not include tail-cutpoint selection uncertainty"
            ),
        },
        "limitations": [
            "The protocol and classes come from a CFG-Rejection discovery/replication run.",
            "Images and ASD tails may already have been inspected; this is not a confirmation set.",
            "The classifier measures ImageNet semantic consistency, not structural or perceptual artifact quality.",
            "Five prespecified ASD representations are reported side by side; no metric is declared a winner.",
            "Percentile intervals are descriptive uncertainty summaries, not preregistered confirmatory tests.",
        ],
        "results": results,
    }
    return payload, csv_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classifier-csv",
        type=Path,
        required=True,
        help="Per-image classifier CSV from score_cfg_rejection_edm2_imagenet.py.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Matching CFG-Rejection manifest.json (protocol pilot or paper-10k).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Defaults to CLASSIFIER_CSV_DIR/classifier_discovery_statistics.json.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Defaults to CLASSIFIER_CSV_DIR/classifier_discovery_statistics.csv.",
    )
    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    args = parser.parse_args()
    if args.bootstrap_repetitions < 100:
        parser.error("--bootstrap-repetitions must be at least 100")
    if not 0.0 < args.confidence_level < 1.0:
        parser.error("--confidence-level must be in (0, 1)")
    return args


def main() -> None:
    args = parse_args()
    classifier_csv = args.classifier_csv.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    output_json = (
        args.output_json.expanduser().resolve()
        if args.output_json is not None
        else classifier_csv.parent / "classifier_discovery_statistics.json"
    )
    output_csv = (
        args.output_csv.expanduser().resolve()
        if args.output_csv is not None
        else classifier_csv.parent / "classifier_discovery_statistics.csv"
    )
    if output_json == output_csv:
        raise ValueError("--output-json and --output-csv must be different paths")

    payload, rows = analyze(
        classifier_csv=classifier_csv,
        manifest_path=manifest_path,
        bootstrap_repetitions=args.bootstrap_repetitions,
        bootstrap_seed=args.bootstrap_seed,
        confidence_level=args.confidence_level,
    )
    atomic_json_dump(payload, output_json)
    atomic_csv_dump(rows, output_csv)
    directional = {
        metric: {
            "target_probability_high_minus_low": details[
                "all_classes_equal_weight"
            ]["target_probability"]["high_minus_low"],
            "top1_error_rate_low_minus_high": details[
                "all_classes_equal_weight"
            ]["top1_error_rate"]["low_minus_high"],
        }
        for metric, details in payload["results"].items()
    }
    print(
        json.dumps(
            {
                "analysis_role": payload["analysis_status"]["role"],
                "output_json": str(output_json),
                "output_csv": str(output_csv),
                "directional_secondary_results": directional,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
