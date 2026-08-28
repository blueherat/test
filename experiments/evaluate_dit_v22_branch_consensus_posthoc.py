#!/usr/bin/env python3
"""Evaluate the sealed DiT branch-consensus product with external blind labels.

The feature product is validated and bound before this program opens the
protocol-conformance comparison table.  Blind visual labels are evaluation-only
targets.  FID, Inception, DINO, CLIP, decoded images, and embeddings are neither
read nor accepted as arguments.  This selected pilot remains posthoc discovery;
no p-value or interval produced here is confirmatory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


sys.dont_write_bytecode = True

EXPERIMENT = "dit_v22_branch_consensus_posthoc_evaluation_v1"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRODUCT_LOCK = (
    ROOT
    / "experiments"
    / "locks"
    / "dit_v22_branch_consensus_label_free_product_v1"
    / "lock.json"
)
DEFAULT_CONFIG = (
    ROOT
    / "experiments"
    / "configs"
    / "dit_v22_branch_consensus_posthoc_evaluation_v1.json"
)
DEFAULT_JUDGE_ROOT = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "dit_v22_repairability_pilot_v1_2_protocol_conformance_v2"
)
DEFAULT_OUTPUT = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "dit_v22_branch_consensus_posthoc_evaluation_v1"
)

EXPECTED_JUDGE_RESULT_ID = (
    "f608e77afd72bbb6921720eaf92840c519528697ea0dd1712bac8dd147d6a358"
)
EXPECTED_JUDGE_FILES = {
    "analyzer_source.py": "64d2f9b8b788509dc55ccb3ed7855af3c81884150ea5182b01fb12fbe7544100",
    "protocol_conformance_comparisons.csv": "8870d6151dd9646ce3acc12c54dbef29701df6b1d489f0cf1b3445db79190a07",
    "results.json": "45d00cce22e21fd334cd2d878bd6562dfd8770207de121665ef57d1639379b85",
}
EXPECTED_JUDGE_MANIFEST_SHA256 = (
    "cc511868a6d2cf15ae12f199b69314c31f8b03cd3277c13804f00d676d27ef06"
)
EXPECTED_JUDGE_COMPLETION_PAYLOAD_SHA256 = (
    "e60bc01fe8a6c5bb7b4c54bfc7500f4acb6489715a6b09e93bfe3d2760c24174"
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def require_regular(path: Path, label: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{label} must be a regular non-symlink file: {path}")
    return path


def require_directory(path: Path, label: str) -> Path:
    if not path.is_dir() or path.is_symlink():
        raise RuntimeError(f"{label} must be a real directory: {path}")
    return path


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def validate_self_identity(value: Mapping[str, Any], field: str, label: str) -> str:
    expected = value.get(field)
    if not isinstance(expected, str):
        raise RuntimeError(f"{label} lacks {field}")
    payload = dict(value)
    payload.pop(field)
    actual = canonical_sha256(payload)
    if actual != expected:
        raise RuntimeError(f"{label} canonical identity mismatch")
    return expected


def validate_feature_product(lock_path: Path) -> tuple[dict[str, Any], Path]:
    lock_path = require_regular(lock_path.resolve(), "feature product lock")
    lock = load_json(lock_path)
    lock_identity = validate_self_identity(
        lock, "lock_identity_sha256", "feature product lock"
    )
    product_root = require_directory(Path(lock["product_root"]), "feature product")
    completion = load_json(
        require_regular(product_root / "completion.json", "feature completion")
    )
    product_identity = validate_self_identity(
        completion, "product_identity_sha256", "feature completion"
    )
    manifest = load_json(require_regular(product_root / "manifest.json", "feature manifest"))
    manifest_identity = validate_self_identity(
        manifest, "identity_sha256", "feature manifest"
    )
    if product_identity != lock.get("product_identity_sha256"):
        raise RuntimeError("feature product identity differs from frozen lock")
    if manifest_identity != lock.get("product_manifest_identity_sha256"):
        raise RuntimeError("feature manifest identity differs from frozen lock")
    if completion.get("source_sha256") != lock.get("source_sha256"):
        raise RuntimeError("feature source identity differs from frozen lock")
    if completion.get("config_sha256") != lock.get("config_sha256"):
        raise RuntimeError("feature config identity differs from frozen lock")
    if completion.get("input_inventory_identity_sha256") != lock.get(
        "input_inventory_identity_sha256"
    ):
        raise RuntimeError("feature input inventory differs from frozen lock")
    file_records = lock.get("product_files")
    if not isinstance(file_records, dict) or file_records != completion.get("files"):
        raise RuntimeError("feature product file inventory differs from frozen lock")
    for name, record in sorted(file_records.items()):
        path = require_regular(product_root / name, f"feature product {name}")
        if path.stat().st_size != record.get("bytes") or sha256_file(path) != record.get(
            "sha256"
        ):
            raise RuntimeError(f"feature product file changed: {name}")
    if manifest.get("feature_selection_auc_thresholding_or_label_join_performed") is not False:
        raise RuntimeError("feature product was not label-free")
    if manifest.get("png_decoding_or_external_model_used") is not False:
        raise RuntimeError("feature product used forbidden external inputs")
    return {
        "lock_identity_sha256": lock_identity,
        "product_identity_sha256": product_identity,
        "manifest_identity_sha256": manifest_identity,
        "source_sha256": completion["source_sha256"],
        "config_sha256": completion["config_sha256"],
        "features_csv_sha256": completion["files"]["features.csv"]["sha256"],
    }, product_root


def validate_judge_product(judge_root: Path) -> dict[str, Any]:
    judge_root = require_directory(judge_root.resolve(), "external judge product")
    manifest_path = require_regular(judge_root / "manifest.json", "judge manifest")
    completion_path = require_regular(judge_root / "completion.json", "judge completion")
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    if sha256_file(manifest_path) != EXPECTED_JUDGE_MANIFEST_SHA256:
        raise RuntimeError("external judge manifest changed")
    if completion.get("payload_sha256") != EXPECTED_JUDGE_COMPLETION_PAYLOAD_SHA256:
        raise RuntimeError("external judge completion changed")
    if completion.get("result_identity_sha256") != EXPECTED_JUDGE_RESULT_ID:
        raise RuntimeError("external judge result identity changed")
    if manifest.get("result_identity_sha256") != EXPECTED_JUDGE_RESULT_ID:
        raise RuntimeError("external judge manifest result identity changed")
    manifest_files = {
        str(record["name"]): str(record["sha256"])
        for record in manifest.get("files", [])
        if isinstance(record, dict)
    }
    if manifest_files != EXPECTED_JUDGE_FILES:
        raise RuntimeError("external judge file inventory changed")
    for name, expected in EXPECTED_JUDGE_FILES.items():
        if sha256_file(require_regular(judge_root / name, f"judge file {name}")) != expected:
            raise RuntimeError(f"external judge file changed: {name}")
    results = load_json(judge_root / "results.json")
    if validate_self_identity(results, "identity_sha256", "judge results") != EXPECTED_JUDGE_RESULT_ID:
        raise RuntimeError("external judge results canonical identity changed")
    return {
        "result_identity_sha256": EXPECTED_JUDGE_RESULT_ID,
        "manifest_sha256": EXPECTED_JUDGE_MANIFEST_SHA256,
        "comparisons_sha256": EXPECTED_JUDGE_FILES[
            "protocol_conformance_comparisons.csv"
        ],
        "results_sha256": EXPECTED_JUDGE_FILES["results.json"],
    }


def parse_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise RuntimeError(f"expected canonical boolean, got {value!r}")


def read_feature_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for source in csv.DictReader(handle):
            row = dict(source)
            for key in (
                "pair_index",
                "global_seed",
                "class_id",
                "target_slot",
                "rollback_step",
                "rollback_internal_timestep",
                "horizon",
                "horizon_internal_timestep",
                "branch_count",
                "fresh_branch_count",
                "fresh_medoid_attempt",
            ):
                row[key] = int(row[key])
            for key in (
                "primary_horizon",
                "attempt0_exchangeable_with_fresh",
                "attempt0_rank_calibrated",
                "fresh_only_rank_exchangeability_eligible",
                "fresh_medoid_tie",
            ):
                if row[key] not in {"true", "false"}:
                    raise RuntimeError(f"invalid feature boolean {key}={row[key]!r}")
                row[key] = row[key] == "true"
            for key in (
                "fresh_dispersion_D",
                "attempt0_to_fresh_mean_A",
                "attempt0_outlier_ratio_O",
                "attempt0_all5_rank_p_descriptive",
                "fresh_nonconformity_min",
                "fresh_nonconformity_mean",
                "fresh_nonconformity_max",
            ):
                row[key] = float(row[key])
            for attempt in range(1, 5):
                row[f"fresh_attempt{attempt}_nonconformity"] = float(
                    row[f"fresh_attempt{attempt}_nonconformity"]
                )
                row[f"fresh_attempt{attempt}_rank_p"] = float(
                    row[f"fresh_attempt{attempt}_rank_p"]
                )
            rows.append(row)
    if len(rows) != 96:
        raise RuntimeError(f"expected 96 feature rows, found {len(rows)}")
    if any(row["attempt0_rank_calibrated"] for row in rows):
        raise RuntimeError("attempt0 rank was incorrectly marked calibrated")
    return rows


def read_comparison_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for source in csv.DictReader(handle):
            row: dict[str, Any] = dict(source)
            for key in (
                "pair_index",
                "global_seed",
                "rollback_sampling_step",
                "fresh_attempt",
            ):
                row[key] = int(row[key])
            for key in (
                "baseline_repair_opportunity_2of3",
                "blur_reduced_within_reviewer_2of3",
                "semantic_preservation_2of3_yes",
                "fresh_failure_2of3_clear_bad",
                "fresh_topology_2of3",
                "protocol_literal_success",
            ):
                row[key] = parse_bool(row[key])
            rows.append(row)
    if len(rows) != 128:
        raise RuntimeError(f"expected 128 comparison rows, found {len(rows)}")
    return rows


def mean(values: Sequence[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def median(values: Sequence[float]) -> float | None:
    return float(np.median(np.asarray(values, dtype=np.float64))) if values else None


def quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability))


def auc_higher_positive(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    positives = [score for score, label in zip(scores, labels) if label]
    negatives = [score for score, label in zip(scores, labels) if not label]
    if not positives or not negatives:
        return None
    wins = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives
        for negative in negatives
    )
    return float(wins / (len(positives) * len(negatives)))


def average_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="stable")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def spearman(values_x: Sequence[float], values_y: Sequence[float]) -> float | None:
    if len(values_x) != len(values_y) or len(values_x) < 2:
        return None
    x = average_ranks(values_x)
    y = average_ranks(values_y)
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denominator = float(np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2)))
    if denominator == 0.0:
        return None
    return float(np.sum(x_centered * y_centered) / denominator)


def poisson_binomial_upper_tail(probabilities: Sequence[float], observed: int) -> float:
    distribution = np.asarray([1.0], dtype=np.float64)
    for probability in probabilities:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("Poisson-binomial probability outside [0,1]")
        updated = np.zeros(len(distribution) + 1, dtype=np.float64)
        updated[:-1] += distribution * (1.0 - probability)
        updated[1:] += distribution * probability
        distribution = updated
    return float(np.sum(distribution[observed:], dtype=np.float64))


def centrality_concordance(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_job: dict[tuple[str, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_job[(row["role"], row["pair_index"], row["rollback_step"])].append(row)
    job_values: list[float] = []
    wins = 0.0
    comparisons = 0
    for block in by_job.values():
        successes = [row for row in block if row["protocol_literal_success"]]
        failures = [row for row in block if not row["protocol_literal_success"]]
        if not successes or not failures:
            continue
        local_wins = 0.0
        local_count = 0
        for success in successes:
            for failure in failures:
                left = float(success["fresh_nonconformity"])
                right = float(failure["fresh_nonconformity"])
                local_wins += 1.0 if left < right else 0.5 if left == right else 0.0
                local_count += 1
        job_values.append(local_wins / local_count)
        wins += local_wins
        comparisons += local_count
    return {
        "informative_job_count": len(job_values),
        "success_failure_pair_count": comparisons,
        "micro_concordance_lower_nonconformity_is_better": (
            wins / comparisons if comparisons else None
        ),
        "mean_within_job_concordance": mean(job_values),
        "job_concordances": job_values,
    }


def intervention_summary(
    branch_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    by_job: dict[tuple[str, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in branch_rows:
        if row["path_has_baseline_opportunity"]:
            by_job[(row["role"], row["pair_index"], row["rollback_step"])].append(row)
    if any(len(block) != 4 for block in by_job.values()):
        raise RuntimeError("eligible job does not contain exactly four fresh branches")

    jobs: list[dict[str, Any]] = []
    for (role, pair_index, step), block in sorted(by_job.items()):
        medoids = [row for row in block if row["is_fresh_medoid"]]
        if len(medoids) != 1:
            raise RuntimeError("each job must have exactly one frozen fresh medoid")
        success_count = sum(row["protocol_literal_success"] for row in block)
        opportunity_count = sum(row["baseline_repair_opportunity_2of3"] for row in block)
        medoid_success = int(bool(medoids[0]["protocol_literal_success"]))
        random_rate = success_count / 4.0
        jobs.append(
            {
                "role": role,
                "pair_index": pair_index,
                "rollback_step": step,
                "path_key": f"{role}|{pair_index}",
                "success_count": success_count,
                "random_scout_expected_success_rate": random_rate,
                "medoid_attempt": medoids[0]["fresh_attempt"],
                "medoid_success": medoid_success,
                "medoid_minus_random": medoid_success - random_rate,
                "baseline_opportunity_comparisons": opportunity_count,
                "all_four_comparisons_have_opportunity": opportunity_count == 4,
                "fresh_dispersion_D": float(block[0]["fresh_dispersion_D"]),
            }
        )

    eligible_path_keys = sorted({row["path_key"] for row in jobs})
    if any(sum(row["path_key"] == key for row in jobs) != 2 for key in eligible_path_keys):
        raise RuntimeError("each eligible path must contribute exactly two rollback jobs")
    medoid_success_count = sum(row["medoid_success"] for row in jobs)
    probabilities = [row["random_scout_expected_success_rate"] for row in jobs]
    differences = [float(row["medoid_minus_random"]) for row in jobs]

    path_values = {
        key: mean(
            [float(row["medoid_minus_random"]) for row in jobs if row["path_key"] == key]
        )
        for key in eligible_path_keys
    }
    if any(value is None for value in path_values.values()):
        raise RuntimeError("empty path cluster")
    ordered_path_values = np.asarray(
        [float(path_values[key]) for key in eligible_path_keys], dtype=np.float64
    )
    rng = np.random.default_rng(bootstrap_seed)
    draws = np.mean(
        ordered_path_values[
            rng.integers(
                0,
                len(ordered_path_values),
                size=(bootstrap_replicates, len(ordered_path_values)),
            )
        ],
        axis=1,
        dtype=np.float64,
    )
    leave_one_out = [
        float(np.mean(np.delete(ordered_path_values, index), dtype=np.float64))
        for index in range(len(ordered_path_values))
    ]

    by_step = {}
    for step in (109, 149):
        block = [row for row in jobs if row["rollback_step"] == step]
        by_step[str(step)] = {
            "job_count": len(block),
            "medoid_success_count": sum(row["medoid_success"] for row in block),
            "medoid_success_rate": mean([float(row["medoid_success"]) for row in block]),
            "uniform_random_expected_success_rate": mean(
                [float(row["random_scout_expected_success_rate"]) for row in block]
            ),
            "medoid_minus_random": mean(
                [float(row["medoid_minus_random"]) for row in block]
            ),
        }
    by_role = {}
    for role in sorted({row["role"] for row in jobs}):
        block = [row for row in jobs if row["role"] == role]
        by_role[role] = {
            "job_count": len(block),
            "medoid_success_rate": mean([float(row["medoid_success"]) for row in block]),
            "uniform_random_expected_success_rate": mean(
                [float(row["random_scout_expected_success_rate"]) for row in block]
            ),
            "medoid_minus_random": mean(
                [float(row["medoid_minus_random"]) for row in block]
            ),
        }

    fully_visible = [row for row in jobs if row["all_four_comparisons_have_opportunity"]]
    return {
        "eligible_path_count": len(eligible_path_keys),
        "eligible_job_count": len(jobs),
        "eligible_branch_comparison_count": len(jobs) * 4,
        "strict_success_count": sum(row["success_count"] for row in jobs),
        "medoid_success_count": medoid_success_count,
        "medoid_success_rate": medoid_success_count / len(jobs),
        "uniform_random_expected_success_count": sum(probabilities),
        "uniform_random_expected_success_rate": mean(probabilities),
        "medoid_minus_random": mean(differences),
        "conditional_poisson_binomial_upper_tail_posthoc": poisson_binomial_upper_tail(
            probabilities, medoid_success_count
        ),
        "path_cluster_bootstrap_95_interval_posthoc": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "leave_one_path_out_range": [min(leave_one_out), max(leave_one_out)],
        "by_rollback_step": by_step,
        "by_legacy_selection_role": by_role,
        "fully_visible_opportunity_sensitivity": {
            "job_count": len(fully_visible),
            "medoid_success_rate": mean(
                [float(row["medoid_success"]) for row in fully_visible]
            ),
            "uniform_random_expected_success_rate": mean(
                [float(row["random_scout_expected_success_rate"]) for row in fully_visible]
            ),
            "medoid_minus_random": mean(
                [float(row["medoid_minus_random"]) for row in fully_visible]
            ),
        },
        "fresh_dispersion_D_vs_job_success_fraction_spearman": spearman(
            [float(row["fresh_dispersion_D"]) for row in jobs],
            [float(row["random_scout_expected_success_rate"]) for row in jobs],
        ),
        "jobs": jobs,
    }


def detection_summary(
    feature_rows: Sequence[Mapping[str, Any]],
    path_opportunities: Mapping[tuple[str, int], bool],
) -> dict[str, Any]:
    by_path: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in feature_rows:
        by_path[(row["legacy_selection_role"], row["pair_index"])].append(row)
    if len(by_path) != 16 or any(len(block) != 2 for block in by_path.values()):
        raise RuntimeError("detection readout requires sixteen paths with two rollback jobs")
    path_rows = []
    for key, block in sorted(by_path.items()):
        opportunity = path_opportunities[key]
        path_rows.append(
            {
                "role": key[0],
                "pair_index": key[1],
                "path_has_baseline_opportunity": opportunity,
                "mean_attempt0_outlier_ratio_O": mean(
                    [float(row["attempt0_outlier_ratio_O"]) for row in block]
                ),
                "step109_attempt0_outlier_ratio_O": next(
                    float(row["attempt0_outlier_ratio_O"])
                    for row in block
                    if row["rollback_step"] == 109
                ),
                "step149_attempt0_outlier_ratio_O": next(
                    float(row["attempt0_outlier_ratio_O"])
                    for row in block
                    if row["rollback_step"] == 149
                ),
                "mean_fresh_dispersion_D": mean(
                    [float(row["fresh_dispersion_D"]) for row in block]
                ),
            }
        )
    labels = [bool(row["path_has_baseline_opportunity"]) for row in path_rows]
    scores = [float(row["mean_attempt0_outlier_ratio_O"]) for row in path_rows]
    positive_scores = [score for score, label in zip(scores, labels) if label]
    negative_scores = [score for score, label in zip(scores, labels) if not label]
    leave_one_out_auc = [
        auc_higher_positive(
            [score for position, score in enumerate(scores) if position != omitted],
            [label for position, label in enumerate(labels) if position != omitted],
        )
        for omitted in range(len(path_rows))
    ]
    valid_loo = [value for value in leave_one_out_auc if value is not None]
    by_step = {}
    for step in (109, 149):
        field = f"step{step}_attempt0_outlier_ratio_O"
        by_step[str(step)] = {
            "auc_higher_O_for_opportunity": auc_higher_positive(
                [float(row[field]) for row in path_rows], labels
            ),
            "median_O_opportunity": median(
                [float(row[field]) for row in path_rows if row["path_has_baseline_opportunity"]]
            ),
            "median_O_no_opportunity": median(
                [float(row[field]) for row in path_rows if not row["path_has_baseline_opportunity"]]
            ),
        }
    return {
        "path_count": len(path_rows),
        "opportunity_path_count": sum(labels),
        "non_opportunity_path_count": len(labels) - sum(labels),
        "auc_higher_mean_O_for_opportunity": auc_higher_positive(scores, labels),
        "median_mean_O_opportunity": median(positive_scores),
        "median_mean_O_no_opportunity": median(negative_scores),
        "median_difference_opportunity_minus_no_opportunity": (
            median(positive_scores) - median(negative_scores)
            if positive_scores and negative_scores
            else None
        ),
        "leave_one_path_out_auc_range": [min(valid_loo), max(valid_loo)],
        "by_rollback_step": by_step,
        "mean_D_auc_higher_for_opportunity": auc_higher_positive(
            [float(row["mean_fresh_dispersion_D"]) for row in path_rows], labels
        ),
        "calibration_status": "descriptive_not_conformal_attempt0_was_future_selected",
        "paths": path_rows,
    }


def rank_success_table(branch_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for rank_p in (1.0, 0.75, 0.5, 0.25):
        block = [
            row
            for row in branch_rows
            if row["path_has_baseline_opportunity"]
            and math.isclose(float(row["fresh_rank_p"]), rank_p, abs_tol=1e-12)
        ]
        output.append(
            {
                "fresh_rank_p": rank_p,
                "centrality_order": {1.0: 1, 0.75: 2, 0.5: 3, 0.25: 4}[rank_p],
                "branch_count": len(block),
                "strict_success_count": sum(
                    row["protocol_literal_success"] for row in block
                ),
                "strict_success_rate": mean(
                    [float(row["protocol_literal_success"]) for row in block]
                ),
            }
        )
    return output


def joined_rows_to_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        raise RuntimeError("joined evaluation rows are empty")
    columns = list(rows[0])
    if any(list(row) != columns for row in rows):
        raise RuntimeError("joined rows have inconsistent columns")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        serialized = {}
        for key, value in row.items():
            if isinstance(value, bool):
                serialized[key] = "True" if value else "False"
            elif isinstance(value, float):
                serialized[key] = format(value, ".17g")
            else:
                serialized[key] = value
        writer.writerow(serialized)
    return buffer.getvalue()


def build_evaluation(
    *,
    feature_rows: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    judge_results: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path_records = judge_results.get("per_path")
    if not isinstance(path_records, list) or len(path_records) != 16:
        raise RuntimeError("judge results path summaries changed")
    path_opportunities = {
        (str(row["role"]), int(row["pair_index"])): bool(
            row["path_has_baseline_opportunity"]
        )
        for row in path_records
    }
    if len(path_opportunities) != 16 or sum(path_opportunities.values()) != 9:
        raise RuntimeError("judge path opportunity inventory changed")

    features_by_key = {
        (
            row["legacy_selection_role"],
            row["pair_index"],
            row["rollback_step"],
            row["horizon"],
        ): row
        for row in feature_rows
    }
    if len(features_by_key) != 96:
        raise RuntimeError("duplicate feature identity")
    comparison_by_key = {
        (
            row["role"],
            row["pair_index"],
            row["rollback_sampling_step"],
            row["fresh_attempt"],
        ): row
        for row in comparisons
    }
    if len(comparison_by_key) != 128:
        raise RuntimeError("duplicate external judge comparison identity")

    joined: list[dict[str, Any]] = []
    for horizon in config["secondary_horizons"] + [config["primary_horizon"]]:
        for key, comparison in sorted(comparison_by_key.items()):
            role, pair_index, step, attempt = key
            feature = features_by_key[(role, pair_index, step, int(horizon))]
            if int(comparison["global_seed"]) != int(feature["global_seed"]):
                raise RuntimeError("feature/judge seed identity mismatch")
            joined.append(
                {
                    "role": role,
                    "pair_index": pair_index,
                    "global_seed": int(comparison["global_seed"]),
                    "class_id": int(feature["class_id"]),
                    "rollback_step": step,
                    "horizon": int(horizon),
                    "primary_horizon": int(horizon) == int(config["primary_horizon"]),
                    "fresh_attempt": attempt,
                    "fresh_nonconformity": float(
                        feature[f"fresh_attempt{attempt}_nonconformity"]
                    ),
                    "fresh_rank_p": float(feature[f"fresh_attempt{attempt}_rank_p"]),
                    "fresh_medoid_attempt": int(feature["fresh_medoid_attempt"]),
                    "is_fresh_medoid": attempt == int(feature["fresh_medoid_attempt"]),
                    "fresh_dispersion_D": float(feature["fresh_dispersion_D"]),
                    "attempt0_outlier_ratio_O": float(
                        feature["attempt0_outlier_ratio_O"]
                    ),
                    "path_has_baseline_opportunity": path_opportunities[
                        (role, pair_index)
                    ],
                    "baseline_repair_opportunity_2of3": bool(
                        comparison["baseline_repair_opportunity_2of3"]
                    ),
                    "blur_reduced_within_reviewer_2of3": bool(
                        comparison["blur_reduced_within_reviewer_2of3"]
                    ),
                    "semantic_preservation_2of3_yes": bool(
                        comparison["semantic_preservation_2of3_yes"]
                    ),
                    "fresh_failure_2of3_clear_bad": bool(
                        comparison["fresh_failure_2of3_clear_bad"]
                    ),
                    "protocol_literal_success": bool(
                        comparison["protocol_literal_success"]
                    ),
                }
            )
    if len(joined) != 384:
        raise RuntimeError(f"expected 384 joined rows, found {len(joined)}")

    by_horizon: dict[str, Any] = {}
    rank_tables: dict[str, Any] = {}
    for horizon in (5, 10, 20):
        block = [row for row in joined if row["horizon"] == horizon]
        intervention = intervention_summary(
            block,
            bootstrap_replicates=int(config["bootstrap"]["replicates"]),
            bootstrap_seed=int(config["bootstrap"]["seed"]) + horizon,
        )
        concordance = centrality_concordance(
            [row for row in block if row["path_has_baseline_opportunity"]]
        )
        intervention["centrality_concordance"] = concordance
        intervention["jobs"] = intervention.pop("jobs")
        by_horizon[str(horizon)] = intervention
        rank_tables[str(horizon)] = rank_success_table(block)

    feature_primary_rows = [
        row for row in feature_rows if row["horizon"] == int(config["primary_horizon"])
    ]
    detection = detection_summary(feature_primary_rows, path_opportunities)

    primary = by_horizon[str(config["primary_horizon"])]
    gate = config["prioritization_gate"]
    step_differences = [
        float(primary["by_rollback_step"][str(step)]["medoid_minus_random"])
        for step in (109, 149)
    ]
    intervention_gate = all(
        (
            primary["eligible_path_count"] >= int(gate["minimum_eligible_paths"]),
            float(primary["medoid_minus_random"]) > 0.0,
            float(
                primary["centrality_concordance"][
                    "micro_concordance_lower_nonconformity_is_better"
                ]
            )
            > 0.5,
            all(value >= 0.0 for value in step_differences),
        )
    )
    detection_step_aucs = [
        float(detection["by_rollback_step"][str(step)]["auc_higher_O_for_opportunity"])
        for step in (109, 149)
    ]
    detection_gate = (
        float(detection["auc_higher_mean_O_for_opportunity"])
        >= float(gate["attempt0_detection_auc_minimum"])
        and all(value > 0.5 for value in detection_step_aucs)
    )

    results_payload = {
        "schema_version": 1,
        "artifact_kind": "DIT_V22_BRANCH_CONSENSUS_POSTHOC_EVALUATION_V1",
        "scientific_role": config["scientific_role"],
        "primary_horizon": config["primary_horizon"],
        "secondary_horizons": config["secondary_horizons"],
        "external_judge_boundary": {
            "blind_visual_labels_used_only_as_outcomes": True,
            "FID_Inception_DINO_CLIP_embeddings_read": False,
            "external_metric_used_as_method_feature_or_trigger": False,
        },
        "by_horizon": by_horizon,
        "fresh_rank_success_tables": rank_tables,
        "attempt0_opportunity_detection_primary_h10": detection,
        "prioritization_decision": {
            "intervention_component_gate_passed": intervention_gate,
            "attempt0_detection_component_gate_passed": detection_gate,
            "new_gpu_symmetric_prefix_experiment_priority": (
                "CONDITIONAL_GO" if intervention_gate or detection_gate else "STOP_CURRENT_FORM"
            ),
            "gate_is_confirmatory": False,
            "reason": (
                "This gate only prioritizes whether a fully symmetric, unselected prefix experiment "
                "is worth new GPU budget; the current selected pilot cannot establish a method claim."
            ),
        },
        "claim_limits": [
            "The feature formula was sealed before this label join, but the research program had already seen these pilot labels; all associations are posthoc discovery.",
            "Attempt0 ranks are not conformal because attempt0 future information participated in retrospective B/E path selection.",
            "Fresh-only attempts are mutually exchangeable, but four branches give a minimum rank p of 0.25.",
            "Medoid is compared with uniform random selection among the same four already-computed fresh branches, not with a one-sample compute budget.",
            "External blind labels evaluate the method and are not an online signal.",
        ],
    }
    results_payload["identity_sha256"] = canonical_sha256(results_payload)
    return results_payload, joined


def run(
    *,
    product_lock: Path,
    judge_root: Path,
    config_path: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise RuntimeError(f"output exists; refusing overwrite: {output}")

    feature_lineage, product_root = validate_feature_product(product_lock)
    config_path = require_regular(config_path.resolve(), "evaluation config")
    config = load_json(config_path)
    config_sha256 = sha256_file(config_path)
    if config.get("schema_version") != 1 or config.get("evaluation_version") != EXPERIMENT:
        raise RuntimeError("evaluation config identity changed")
    if config.get("feature_product_lock_identity_sha256") != feature_lineage[
        "lock_identity_sha256"
    ]:
        raise RuntimeError("evaluation config binds a different feature-product lock")
    if config.get("feature_product_identity_sha256") != feature_lineage[
        "product_identity_sha256"
    ]:
        raise RuntimeError("evaluation config binds a different feature product")

    # The external judge is deliberately opened only after the exact feature
    # product, product lock, and evaluation configuration have passed.
    judge_lineage = validate_judge_product(judge_root)
    if config.get("external_judge_result_identity_sha256") != judge_lineage[
        "result_identity_sha256"
    ]:
        raise RuntimeError("evaluation config binds a different judge result")

    feature_rows = read_feature_rows(product_root / "features.csv")
    comparisons = read_comparison_rows(
        judge_root / "protocol_conformance_comparisons.csv"
    )
    judge_results = load_json(judge_root / "results.json")
    results_payload, joined = build_evaluation(
        feature_rows=feature_rows,
        comparisons=comparisons,
        judge_results=judge_results,
        config=config,
    )

    source_path = require_regular(Path(__file__).resolve(), "evaluator source")
    source_sha256 = sha256_file(source_path)
    manifest_payload = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "scientific_role": config["scientific_role"],
        "source_sha256": source_sha256,
        "config_sha256": config_sha256,
        "feature_lineage": feature_lineage,
        "external_judge_lineage": judge_lineage,
        "results_identity_sha256": results_payload["identity_sha256"],
        "counts": {
            "feature_rows": len(feature_rows),
            "judge_comparisons": len(comparisons),
            "joined_rows": len(joined),
        },
        "labels_opened_only_after_feature_product_validation": True,
        "FID_Inception_DINO_CLIP_embeddings_opened": False,
        "png_files_opened": False,
        "method_uses_external_evaluation_outcome": False,
    }
    manifest_payload["identity_sha256"] = canonical_sha256(manifest_payload)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        (temporary / "results.json").write_text(
            json.dumps(results_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (temporary / "joined_branch_evaluation.csv").write_text(
            joined_rows_to_csv(joined), encoding="utf-8"
        )
        (temporary / "manifest.json").write_text(
            json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        shutil.copyfile(source_path, temporary / "evaluator_source.py")
        shutil.copyfile(config_path, temporary / "frozen_evaluation_config.json")
        files = {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(temporary.iterdir())
            if path.is_file()
        }
        completion_payload = {
            "schema_version": 1,
            "experiment": EXPERIMENT,
            "manifest_identity_sha256": manifest_payload["identity_sha256"],
            "results_identity_sha256": results_payload["identity_sha256"],
            "source_sha256": source_sha256,
            "config_sha256": config_sha256,
            "files": files,
            "posthoc_only": True,
        }
        completion_payload["product_identity_sha256"] = canonical_sha256(
            completion_payload
        )
        (temporary / "completion.json").write_text(
            json.dumps(completion_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    primary = results_payload["by_horizon"][str(config["primary_horizon"])]
    return {
        "output": str(output),
        "product_identity_sha256": completion_payload["product_identity_sha256"],
        "results_identity_sha256": results_payload["identity_sha256"],
        "primary_horizon": config["primary_horizon"],
        "eligible_paths": primary["eligible_path_count"],
        "eligible_jobs": primary["eligible_job_count"],
        "medoid_success_rate": primary["medoid_success_rate"],
        "uniform_random_expected_success_rate": primary[
            "uniform_random_expected_success_rate"
        ],
        "medoid_minus_random": primary["medoid_minus_random"],
        "attempt0_detection_auc": results_payload[
            "attempt0_opportunity_detection_primary_h10"
        ]["auc_higher_mean_O_for_opportunity"],
        "decision": results_payload["prioritization_decision"],
    }


def self_test() -> None:
    assert math.isclose(
        poisson_binomial_upper_tail([0.5, 0.5], 2), 0.25, abs_tol=1e-15
    )
    assert math.isclose(auc_higher_positive([2.0, 1.0], [True, False]), 1.0)
    assert math.isclose(auc_higher_positive([1.0, 1.0], [True, False]), 0.5)
    assert math.isclose(spearman([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]), -1.0)
    rows = [
        {
            "role": "r",
            "pair_index": 0,
            "rollback_step": 109,
            "fresh_nonconformity": 0.1,
            "protocol_literal_success": True,
        },
        {
            "role": "r",
            "pair_index": 0,
            "rollback_step": 109,
            "fresh_nonconformity": 0.2,
            "protocol_literal_success": False,
        },
    ]
    assert centrality_concordance(rows)[
        "micro_concordance_lower_nonconformity_is_better"
    ] == 1.0
    print(
        json.dumps(
            {
                "self_test": "passed",
                "poisson_binomial": True,
                "auc_ties": True,
                "spearman": True,
                "centrality_concordance": True,
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--product-lock", type=Path, default=DEFAULT_PRODUCT_LOCK)
    parser.add_argument("--judge-root", type=Path, default=DEFAULT_JUDGE_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    result = run(
        product_lock=args.product_lock,
        judge_root=args.judge_root,
        config_path=args.config,
        output=args.output.resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
