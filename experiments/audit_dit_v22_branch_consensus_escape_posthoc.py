#!/usr/bin/env python3
"""Audit the posthoc reverse-direction "consensus trap / escape" hypothesis.

The medoid hypothesis failed on the sealed primary h=10 readout.  Inspection of
the already-emitted rank table suggested the opposite direction: among paths
with visible repair opportunity, an early branch that leaves the conditional
consensus may repair more often.  Because this direction was noticed after the
same blind labels were opened, every result here is hypothesis-generating only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


sys.dont_write_bytecode = True

EXPERIMENT = "dit_v22_branch_consensus_escape_posthoc_v1"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_EVALUATION = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "dit_v22_branch_consensus_posthoc_evaluation_v1"
)
DEFAULT_CONFIG = (
    ROOT
    / "experiments"
    / "configs"
    / "dit_v22_branch_consensus_escape_posthoc_v1.json"
)
DEFAULT_OUTPUT = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "dit_v22_branch_consensus_escape_posthoc_audit_v1"
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


def validate_identity(value: Mapping[str, Any], field: str, label: str) -> str:
    expected = value.get(field)
    if not isinstance(expected, str):
        raise RuntimeError(f"{label} lacks {field}")
    payload = dict(value)
    payload.pop(field)
    if canonical_sha256(payload) != expected:
        raise RuntimeError(f"{label} canonical identity mismatch")
    return expected


def validate_source_evaluation(
    root: Path, expected_product: str, expected_results: str
) -> dict[str, Any]:
    root = require_directory(root.resolve(), "source evaluation")
    completion = load_json(
        require_regular(root / "completion.json", "source evaluation completion")
    )
    product_identity = validate_identity(
        completion, "product_identity_sha256", "source evaluation completion"
    )
    if product_identity != expected_product:
        raise RuntimeError("source evaluation product identity changed")
    if completion.get("results_identity_sha256") != expected_results:
        raise RuntimeError("source evaluation result identity changed")
    files = completion.get("files")
    if not isinstance(files, dict):
        raise RuntimeError("source evaluation lacks file inventory")
    for name, record in sorted(files.items()):
        path = require_regular(root / name, f"source evaluation {name}")
        if path.stat().st_size != record.get("bytes") or sha256_file(path) != record.get(
            "sha256"
        ):
            raise RuntimeError(f"source evaluation file changed: {name}")
    results = load_json(root / "results.json")
    if validate_identity(results, "identity_sha256", "source results") != expected_results:
        raise RuntimeError("source results canonical identity changed")
    if completion.get("posthoc_only") is not True:
        raise RuntimeError("source evaluation lost its posthoc boundary")
    return {
        "root": str(root),
        "product_identity_sha256": product_identity,
        "results_identity_sha256": expected_results,
        "joined_csv_sha256": files["joined_branch_evaluation.csv"]["sha256"],
    }


def parse_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise RuntimeError(f"invalid canonical boolean {value!r}")


def load_joined(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for source in csv.DictReader(handle):
            row: dict[str, Any] = dict(source)
            for key in (
                "pair_index",
                "global_seed",
                "class_id",
                "rollback_step",
                "horizon",
                "fresh_attempt",
                "fresh_medoid_attempt",
            ):
                row[key] = int(row[key])
            for key in (
                "primary_horizon",
                "is_fresh_medoid",
                "path_has_baseline_opportunity",
                "baseline_repair_opportunity_2of3",
                "blur_reduced_within_reviewer_2of3",
                "semantic_preservation_2of3_yes",
                "fresh_failure_2of3_clear_bad",
                "protocol_literal_success",
            ):
                row[key] = parse_bool(row[key])
            for key in (
                "fresh_nonconformity",
                "fresh_rank_p",
                "fresh_dispersion_D",
                "attempt0_outlier_ratio_O",
            ):
                row[key] = float(row[key])
            rows.append(row)
    if len(rows) != 384:
        raise RuntimeError(f"expected 384 joined rows, found {len(rows)}")
    return rows


def mean(values: Sequence[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


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


def poisson_binomial_upper_tail(probabilities: Sequence[float], observed: int) -> float:
    distribution = np.asarray([1.0], dtype=np.float64)
    for probability in probabilities:
        updated = np.zeros(len(distribution) + 1, dtype=np.float64)
        updated[:-1] += distribution * (1.0 - probability)
        updated[1:] += distribution * probability
        distribution = updated
    return float(np.sum(distribution[observed:], dtype=np.float64))


def choose_extreme(
    block: Sequence[Mapping[str, Any]], *, maximum: bool
) -> Mapping[str, Any]:
    ordered = sorted(
        block,
        key=lambda row: (
            -float(row["fresh_nonconformity"])
            if maximum
            else float(row["fresh_nonconformity"]),
            int(row["fresh_attempt"]),
        ),
    )
    return ordered[0]


def path_cluster_diagnostics(
    jobs: Sequence[Mapping[str, Any]], *, replicates: int, seed: int
) -> dict[str, Any]:
    by_path: dict[str, list[float]] = defaultdict(list)
    for row in jobs:
        by_path[str(row["path_key"])].append(float(row["escape_minus_random"]))
    if not by_path or any(len(values) != 2 for values in by_path.values()):
        raise RuntimeError("path cluster must contain exactly two rollback jobs")
    path_values = np.asarray(
        [float(np.mean(values, dtype=np.float64)) for _, values in sorted(by_path.items())],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    draws = np.mean(
        path_values[
            rng.integers(0, len(path_values), size=(replicates, len(path_values)))
        ],
        axis=1,
        dtype=np.float64,
    )
    leave_one_out = [
        float(np.mean(np.delete(path_values, index), dtype=np.float64))
        for index in range(len(path_values))
    ]
    return {
        "path_count": len(path_values),
        "path_cluster_bootstrap_95_interval_posthoc": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "leave_one_path_out_range": [min(leave_one_out), max(leave_one_out)],
        "path_mean_differences": path_values.tolist(),
    }


def summarize_group(jobs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "job_count": len(jobs),
        "escape_success_count": sum(row["escape_success"] for row in jobs),
        "escape_success_rate": mean([float(row["escape_success"]) for row in jobs]),
        "uniform_random_expected_success_rate": mean(
            [float(row["random_success_rate"]) for row in jobs]
        ),
        "escape_minus_random": mean(
            [float(row["escape_minus_random"]) for row in jobs]
        ),
        "escape_minus_medoid": mean(
            [float(row["escape_success"] - row["medoid_success"]) for row in jobs]
        ),
    }


def escape_horizon_summary(
    rows: Sequence[Mapping[str, Any]], *, replicates: int, seed: int
) -> dict[str, Any]:
    by_job: dict[tuple[str, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["path_has_baseline_opportunity"]:
            by_job[(row["role"], row["pair_index"], row["rollback_step"])].append(row)
    jobs: list[dict[str, Any]] = []
    for (role, pair_index, step), block in sorted(by_job.items()):
        if len(block) != 4:
            raise RuntimeError("eligible escape job does not contain four branches")
        escape = choose_extreme(block, maximum=True)
        medoid = choose_extreme(block, maximum=False)
        successes = sum(row["protocol_literal_success"] for row in block)
        jobs.append(
            {
                "role": role,
                "pair_index": pair_index,
                "class_id": int(block[0]["class_id"]),
                "rollback_step": step,
                "path_key": f"{role}|{pair_index}",
                "escape_attempt": int(escape["fresh_attempt"]),
                "escape_success": int(bool(escape["protocol_literal_success"])),
                "medoid_attempt": int(medoid["fresh_attempt"]),
                "medoid_success": int(bool(medoid["protocol_literal_success"])),
                "success_count": successes,
                "random_success_rate": successes / 4.0,
                "escape_minus_random": int(bool(escape["protocol_literal_success"]))
                - successes / 4.0,
                "escape_nonconformity": float(escape["fresh_nonconformity"]),
                "medoid_nonconformity": float(medoid["fresh_nonconformity"]),
            }
        )
    summary = summarize_group(jobs)
    summary.update(
        {
            "eligible_path_count": len({row["path_key"] for row in jobs}),
            "eligible_branch_comparison_count": len(jobs) * 4,
            "strict_success_count": sum(row["success_count"] for row in jobs),
            "uniform_random_expected_success_count": sum(
                row["random_success_rate"] for row in jobs
            ),
            "conditional_poisson_binomial_upper_tail_posthoc": poisson_binomial_upper_tail(
                [float(row["random_success_rate"]) for row in jobs],
                sum(row["escape_success"] for row in jobs),
            ),
            "cluster_diagnostics": path_cluster_diagnostics(
                jobs, replicates=replicates, seed=seed
            ),
            "by_rollback_step": {
                str(step): summarize_group(
                    [row for row in jobs if row["rollback_step"] == step]
                )
                for step in (109, 149)
            },
            "by_legacy_selection_role": {
                role: summarize_group([row for row in jobs if row["role"] == role])
                for role in sorted({row["role"] for row in jobs})
            },
            "by_class_id": {
                str(class_id): summarize_group(
                    [row for row in jobs if row["class_id"] == class_id]
                )
                for class_id in sorted({row["class_id"] for row in jobs})
            },
            "jobs": jobs,
        }
    )
    return summary


def safety_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_job: dict[tuple[str, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if not row["path_has_baseline_opportunity"]:
            by_job[(row["role"], row["pair_index"], row["rollback_step"])].append(row)
    jobs = []
    for (role, pair_index, step), block in sorted(by_job.items()):
        if len(block) != 4:
            raise RuntimeError("non-opportunity safety job lacks four branches")
        escape = choose_extreme(block, maximum=True)
        jobs.append(
            {
                "role": role,
                "pair_index": pair_index,
                "class_id": int(block[0]["class_id"]),
                "rollback_step": step,
                "escape_clear_bad": int(bool(escape["fresh_failure_2of3_clear_bad"])),
                "random_clear_bad_rate": mean(
                    [float(row["fresh_failure_2of3_clear_bad"]) for row in block]
                ),
                "escape_preservation": int(
                    bool(escape["semantic_preservation_2of3_yes"])
                ),
                "random_preservation_rate": mean(
                    [float(row["semantic_preservation_2of3_yes"]) for row in block]
                ),
            }
        )
    return {
        "non_opportunity_path_count": len({(row["role"], row["pair_index"]) for row in jobs}),
        "non_opportunity_job_count": len(jobs),
        "escape_clear_bad_rate": mean([float(row["escape_clear_bad"]) for row in jobs]),
        "uniform_random_expected_clear_bad_rate": mean(
            [float(row["random_clear_bad_rate"]) for row in jobs]
        ),
        "escape_minus_random_clear_bad": mean(
            [float(row["escape_clear_bad"] - row["random_clear_bad_rate"]) for row in jobs]
        ),
        "escape_preservation_rate": mean(
            [float(row["escape_preservation"]) for row in jobs]
        ),
        "uniform_random_expected_preservation_rate": mean(
            [float(row["random_preservation_rate"]) for row in jobs]
        ),
        "escape_minus_random_preservation": mean(
            [
                float(row["escape_preservation"] - row["random_preservation_rate"])
                for row in jobs
            ]
        ),
        "jobs": jobs,
    }


def reverse_detection_summary(source_results: Mapping[str, Any]) -> dict[str, Any]:
    detection = source_results.get("attempt0_opportunity_detection_primary_h10")
    if not isinstance(detection, dict) or not isinstance(detection.get("paths"), list):
        raise RuntimeError("source detection rows missing")
    paths = detection["paths"]
    labels = [bool(row["path_has_baseline_opportunity"]) for row in paths]
    mean_scores = [-float(row["mean_attempt0_outlier_ratio_O"]) for row in paths]
    by_step = {}
    for step in (109, 149):
        field = f"step{step}_attempt0_outlier_ratio_O"
        by_step[str(step)] = {
            "auc_lower_O_for_opportunity": auc_higher_positive(
                [-float(row[field]) for row in paths], labels
            )
        }
    leave_one_out = [
        auc_higher_positive(
            [score for index, score in enumerate(mean_scores) if index != omitted],
            [label for index, label in enumerate(labels) if index != omitted],
        )
        for omitted in range(len(paths))
    ]
    valid = [value for value in leave_one_out if value is not None]
    return {
        "path_count": len(paths),
        "opportunity_path_count": sum(labels),
        "auc_lower_mean_O_for_opportunity": auc_higher_positive(mean_scores, labels),
        "by_rollback_step": by_step,
        "leave_one_path_out_auc_range": [min(valid), max(valid)],
        "attempt0_calibration_status": "not_exchangeable_posthoc_direction_only",
    }


def selected_attempt_stability(
    all_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    selections: dict[tuple[str, int, int, int], int] = {}
    for horizon in (5, 10, 20):
        block = [
            row
            for row in all_rows
            if row["horizon"] == horizon and row["path_has_baseline_opportunity"]
        ]
        by_job: dict[tuple[str, int, int], list[Mapping[str, Any]]] = defaultdict(list)
        for row in block:
            by_job[(row["role"], row["pair_index"], row["rollback_step"])].append(row)
        for key, rows in by_job.items():
            selections[(*key, horizon)] = int(
                choose_extreme(rows, maximum=True)["fresh_attempt"]
            )
    job_keys = sorted({key[:3] for key in selections})
    return {
        "eligible_job_count": len(job_keys),
        "same_escape_attempt_h5_h10": mean(
            [
                float(selections[(*key, 5)] == selections[(*key, 10)])
                for key in job_keys
            ]
        ),
        "same_escape_attempt_h10_h20": mean(
            [
                float(selections[(*key, 10)] == selections[(*key, 20)])
                for key in job_keys
            ]
        ),
        "same_escape_attempt_all_horizons": mean(
            [
                float(
                    selections[(*key, 5)]
                    == selections[(*key, 10)]
                    == selections[(*key, 20)]
                )
                for key in job_keys
            ]
        ),
    }


def run(source_root: Path, config_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise RuntimeError(f"output exists; refusing overwrite: {output}")
    config_path = require_regular(config_path.resolve(), "escape audit config")
    config = load_json(config_path)
    if config.get("schema_version") != 1 or config.get("audit_version") != EXPERIMENT:
        raise RuntimeError("escape audit config changed")
    if not config.get("claim_boundary", {}).get(
        "direction_was_discovered_after_opening_same_labels"
    ):
        raise RuntimeError("posthoc claim boundary missing")
    lineage = validate_source_evaluation(
        source_root,
        str(config["source_evaluation_product_identity_sha256"]),
        str(config["source_evaluation_results_identity_sha256"]),
    )
    source_results = load_json(source_root / "results.json")
    rows = load_joined(source_root / "joined_branch_evaluation.csv")

    by_horizon = {}
    for horizon in (5, 10, 20):
        block = [row for row in rows if row["horizon"] == horizon]
        by_horizon[str(horizon)] = escape_horizon_summary(
            block,
            replicates=int(config["bootstrap"]["replicates"]),
            seed=int(config["bootstrap"]["seed"]) + horizon,
        )
    primary_rows = [
        row for row in rows if row["horizon"] == int(config["primary_horizon"])
    ]
    safety = safety_summary(primary_rows)
    reverse_detection = reverse_detection_summary(source_results)
    stability = selected_attempt_stability(rows)

    primary = by_horizon[str(config["primary_horizon"])]
    checks = config["future_experiment_priority_checks"]
    step_differences = [
        float(primary["by_rollback_step"][str(step)]["escape_minus_random"])
        for step in (109, 149)
    ]
    role_differences = [
        float(summary["escape_minus_random"])
        for summary in primary["by_legacy_selection_role"].values()
    ]
    low_o_step_aucs = [
        float(reverse_detection["by_rollback_step"][str(step)]["auc_lower_O_for_opportunity"])
        for step in (109, 149)
    ]
    check_values = {
        "primary_escape_minus_random_strictly_positive": float(
            primary["escape_minus_random"]
        )
        > 0.0,
        "both_rollback_steps_nonnegative": all(value >= 0.0 for value in step_differences),
        "both_legacy_roles_nonnegative": all(value >= 0.0 for value in role_differences),
        "leave_one_path_out_lower_bound_strictly_positive": float(
            primary["cluster_diagnostics"]["leave_one_path_out_range"][0]
        )
        > 0.0,
        "low_O_auc_minimum": float(
            reverse_detection["auc_lower_mean_O_for_opportunity"]
        )
        >= float(checks["low_O_auc_minimum"]),
        "both_rollback_step_low_O_auc_strictly_above_half": all(
            value > 0.5 for value in low_o_step_aucs
        ),
        "non_opportunity_clear_bad_not_above_random": float(
            safety["escape_minus_random_clear_bad"]
        )
        <= 0.0,
        "non_opportunity_preservation_not_below_random": float(
            safety["escape_minus_random_preservation"]
        )
        >= 0.0,
    }

    results_payload = {
        "schema_version": 1,
        "artifact_kind": "DIT_V22_BRANCH_CONSENSUS_ESCAPE_POSTHOC_AUDIT_V1",
        "scientific_role": config["scientific_role"],
        "hypothesis": (
            "A blurred/fused prefix may be a conditional consensus trap rather than an isolated "
            "sampling accident; a successful fresh suffix may need to escape that consensus early."
        ),
        "primary_horizon": config["primary_horizon"],
        "by_horizon": by_horizon,
        "reverse_attempt0_detection": reverse_detection,
        "non_opportunity_safety_primary_h10": safety,
        "escape_attempt_stability": stability,
        "future_experiment_priority_checks": check_values,
        "all_priority_checks_passed": all(check_values.values()),
        "decision": (
            "PREREGISTER_NEW_SYMMETRIC_ESCAPE_TEST"
            if all(check_values.values())
            else "DO_NOT_PROMOTE_WITHOUT_RESOLVING_FAILED_CHECKS"
        ),
        "claim_limits": [
            "The reverse direction was discovered after inspecting the same h5/h10/h20 rank-success table.",
            "This audit cannot confirm escape selection, low-O detection, an online threshold, or population-level quality improvement.",
            "The 16 source paths were selected from an internal B/E-enriched population and do not represent ordinary baseline sampling.",
            "Attempt0 is not exchangeable with fresh suffixes because its future participated in retrospective selection.",
            "Blind visual outcomes are external judges only and are not part of the selector.",
        ],
    }
    results_payload["identity_sha256"] = canonical_sha256(results_payload)

    source_path = require_regular(Path(__file__).resolve(), "escape auditor source")
    source_sha256 = sha256_file(source_path)
    config_sha256 = sha256_file(config_path)
    manifest_payload = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "source_sha256": source_sha256,
        "config_sha256": config_sha256,
        "source_evaluation_lineage": lineage,
        "results_identity_sha256": results_payload["identity_sha256"],
        "posthoc_reverse_direction_declared": True,
        "external_metrics_used_as_method_inputs": False,
        "FID_Inception_DINO_CLIP_embeddings_opened": False,
        "png_files_opened": False,
    }
    manifest_payload["identity_sha256"] = canonical_sha256(manifest_payload)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        (temporary / "results.json").write_text(
            json.dumps(results_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (temporary / "manifest.json").write_text(
            json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        shutil.copyfile(source_path, temporary / "auditor_source.py")
        shutil.copyfile(config_path, temporary / "frozen_audit_config.json")
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

    return {
        "output": str(output),
        "product_identity_sha256": completion_payload["product_identity_sha256"],
        "results_identity_sha256": results_payload["identity_sha256"],
        "primary_horizon": config["primary_horizon"],
        "escape_success_rate": primary["escape_success_rate"],
        "uniform_random_expected_success_rate": primary[
            "uniform_random_expected_success_rate"
        ],
        "escape_minus_random": primary["escape_minus_random"],
        "low_O_opportunity_auc": reverse_detection[
            "auc_lower_mean_O_for_opportunity"
        ],
        "all_priority_checks_passed": results_payload["all_priority_checks_passed"],
        "decision": results_payload["decision"],
    }


def self_test() -> None:
    rows = [
        {"fresh_nonconformity": 0.2, "fresh_attempt": 2},
        {"fresh_nonconformity": 0.7, "fresh_attempt": 3},
        {"fresh_nonconformity": 0.7, "fresh_attempt": 1},
    ]
    assert choose_extreme(rows, maximum=True)["fresh_attempt"] == 1
    assert choose_extreme(rows, maximum=False)["fresh_attempt"] == 2
    assert math.isclose(
        poisson_binomial_upper_tail([0.5, 0.5], 2), 0.25, abs_tol=1e-15
    )
    assert auc_higher_positive([-1.0, -2.0], [True, False]) == 1.0
    print(
        json.dumps(
            {
                "self_test": "passed",
                "extreme_tie_break": True,
                "poisson_binomial": True,
                "reverse_auc": True,
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-evaluation", type=Path, default=DEFAULT_SOURCE_EVALUATION
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    result = run(
        source_root=args.source_evaluation,
        config_path=args.config,
        output=args.output.resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
