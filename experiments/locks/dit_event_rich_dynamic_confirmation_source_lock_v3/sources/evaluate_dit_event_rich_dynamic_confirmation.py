#!/usr/bin/env python3
"""Two-stage, candidate-separable evaluation for event-rich confirmation.

Stage A accepts no score-product argument.  It opens only the immutable
post-anchor plan plus the final blind-consensus manifest/completion and
aggregate counts, then independently gates B and C.  It neither opens, hashes,
stats, nor resolves a score product or the row-level consensus table.

Stage B requires a completed Stage-A receipt.  It opens the consensus rows and
only the score product(s) whose own gate passed.  A gated-off candidate gets
raw p=1 without its product being touched.  The frozen statistics are
directional within-class pair-count-weighted tie-aware AUC, one common
global-seed-block permutation stream, Holm over exactly B/C, and the original
20-sample class-specific strict order-statistic thresholds.  Output is
aggregate only: no row score, rank, permutation draw, image, or trace is
emitted.  Inception/DINO/FID/embedding inputs are forbidden throughout.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.dont_write_bytecode = True

import numpy as np

try:
    from .dit_event_rich_dynamic_contract import (
        B_CANDIDATE,
        B_FEATURE,
        C_CANDIDATE,
        C_FEATURE,
        CANDIDATES,
        CONFIRMATION_SEEDS,
        DEFAULT_DYNAMIC_SOURCE_LOCK,
        candidate_classes,
        canonical_sha256,
        exact_pairs,
        load_json,
        manifest_map,
        publish_artifact,
        reject_forbidden_external_name,
        require_directory,
        require_hex64,
        require_regular,
        sha256_file,
        validate_anchor_plan,
        validate_event_protocol,
        validate_manifest_tree,
        validate_score_columns,
        without_identity,
    )
    from .sample_dit_event_rich_dynamic_traces import load_source_lock
except ImportError:
    from dit_event_rich_dynamic_contract import (  # type: ignore
        B_CANDIDATE,
        B_FEATURE,
        C_CANDIDATE,
        C_FEATURE,
        CANDIDATES,
        CONFIRMATION_SEEDS,
        DEFAULT_DYNAMIC_SOURCE_LOCK,
        candidate_classes,
        canonical_sha256,
        exact_pairs,
        load_json,
        manifest_map,
        publish_artifact,
        reject_forbidden_external_name,
        require_directory,
        require_hex64,
        require_regular,
        sha256_file,
        validate_anchor_plan,
        validate_event_protocol,
        validate_manifest_tree,
        validate_score_columns,
        without_identity,
    )
    from sample_dit_event_rich_dynamic_traces import load_source_lock  # type: ignore


EVALUATOR = "evaluate_dit_event_rich_dynamic_confirmation"
CONSENSUS_KIND = "EVENT_RICH_FINAL_CONSENSUS_LABEL_LOCK_V1"
CONSENSUS_AGGREGATE = "aggregate_counts.json"
CONSENSUS_ROWS = "evaluation_labels.csv"
CONSENSUS_COLUMNS = (
    "phase",
    "global_seed",
    "class_id",
    "final_severity",
    "blur_component",
)
SEVERITIES = ("clean_good", "mild_or_disputed", "clear_bad")
COUNT_FIELDS = (
    "trajectory_count",
    "clean_good",
    "mild_or_disputed",
    "clear_bad",
    "blur_or_soft_fusion_clear_bad",
)
PERMUTATION_DRAWS = 100_000
PERMUTATION_SEED = 2026082801
PERMUTATION_BATCH = 256


def feature_for(candidate: str) -> str:
    return B_FEATURE if candidate == B_CANDIDATE else C_FEATURE


def endpoint_for(candidate: str) -> str:
    return "blur_or_soft_fusion_clear_bad" if candidate == B_CANDIDATE else "all_clear_bad"


def verify_evaluator_source(source_lock: Path, manifest: Mapping[str, Any]) -> None:
    rows = manifest.get("files")
    if not isinstance(rows, list):
        raise RuntimeError("dynamic source-lock file list is malformed")
    by_name = {row.get("name"): row for row in rows if isinstance(row, dict)}
    expected = "sources/evaluate_dit_event_rich_dynamic_confirmation.py"
    if by_name.get(expected, {}).get("sha256") != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("running evaluator differs from frozen source snapshot")


def _validate_count_row(value: Any, expected_n: int, description: str) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != set(COUNT_FIELDS):
        raise RuntimeError(f"{description} aggregate-count schema changed")
    if any(type(value[field]) is not int for field in COUNT_FIELDS):
        raise RuntimeError(f"{description} counts must be integers")
    result = {field: int(value[field]) for field in COUNT_FIELDS}
    if (
        result["trajectory_count"] != expected_n
        or any(number < 0 for number in result.values())
        or result["clean_good"] + result["mild_or_disputed"] + result["clear_bad"]
        != expected_n
        or result["blur_or_soft_fusion_clear_bad"] > result["clear_bad"]
    ):
        raise RuntimeError(f"{description} aggregate counts are impossible")
    return result


def validate_counts(
    counts: Any, active_classes: Sequence[int]
) -> dict[str, Any]:
    if not isinstance(counts, dict) or set(counts) != {"overall", "per_class"}:
        raise RuntimeError("aggregate counts must contain only overall/per_class")
    total = len(active_classes) * len(CONFIRMATION_SEEDS)
    overall = _validate_count_row(counts["overall"], total, "overall")
    rows = counts["per_class"]
    if not isinstance(rows, list) or len(rows) != len(active_classes):
        raise RuntimeError("per-class aggregate row count changed")
    normalized: list[dict[str, Any]] = []
    for expected_class, row in zip(active_classes, rows, strict=True):
        if not isinstance(row, dict) or set(row) != {"class_id", *COUNT_FIELDS}:
            raise RuntimeError("per-class aggregate schema changed")
        if row.get("class_id") != expected_class:
            raise RuntimeError("per-class aggregate order changed")
        values = _validate_count_row(
            {field: row[field] for field in COUNT_FIELDS},
            len(CONFIRMATION_SEEDS),
            f"class {expected_class}",
        )
        normalized.append({"class_id": expected_class, **values})
    for field in COUNT_FIELDS:
        if sum(row[field] for row in normalized) != overall[field]:
            raise RuntimeError(f"per-class {field} does not add to overall")
    return {"overall": overall, "per_class": normalized}


def _consensus_member_map(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = manifest.get("files")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise RuntimeError("consensus manifest member list is malformed")
    by_name = {str(row.get("name")): dict(row) for row in rows}
    expected = {
        "aggregate_counts.json",
        "consensus_rows.csv",
        "evaluation_labels.csv",
        "reviewer_agreement.json",
        "label_access_receipt.json",
    }
    if len(by_name) != len(rows) or set(by_name) != expected:
        raise RuntimeError("final consensus lock payload set changed")
    for name, row in by_name.items():
        if set(row) != {"name", "bytes", "sha256"}:
            raise RuntimeError("consensus member record schema changed")
        require_hex64(row.get("sha256"), f"consensus {name} hash")
        if type(row.get("bytes")) is not int or row["bytes"] <= 0:
            raise RuntimeError("consensus member byte count is invalid")
    return by_name


def load_consensus_aggregate_only(
    root: Path,
    *,
    protocol: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Stage-A loader: never add access to consensus rows or score products."""

    root = require_directory(root, "confirmation consensus export")
    # Directory names are checked, but row/audit payloads are not opened, hashed,
    # statted, resolved, or passed to require_regular in Stage A.
    if {path.name for path in root.iterdir()} != {
        "manifest.json",
        "completion.json",
        "aggregate_counts.json",
        "consensus_rows.csv",
        "evaluation_labels.csv",
        "reviewer_agreement.json",
        "label_access_receipt.json",
    }:
        raise RuntimeError("confirmation consensus export member set changed")
    manifest_path = require_regular(root / "manifest.json", "consensus manifest")
    completion_path = require_regular(root / "completion.json", "consensus completion")
    aggregate_path = require_regular(root / CONSENSUS_AGGREGATE, "consensus aggregate")
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    aggregate = load_json(aggregate_path)
    identity = require_hex64(manifest.get("identity_sha256"), "consensus manifest identity")
    identity_record = manifest.get("identity")
    if not isinstance(identity_record, dict):
        raise RuntimeError("final consensus manifest lacks identity record")
    manifest_identity = require_hex64(
        manifest.get("manifest_identity_sha256"), "consensus manifest envelope identity"
    )
    identity_without = without_identity(identity_record)
    manifest_without = dict(manifest)
    manifest_without.pop("manifest_identity_sha256", None)
    if (
        canonical_sha256(identity_without) != identity
        or identity_record.get("identity_sha256") != identity
        or canonical_sha256(manifest_without) != manifest_identity
        or manifest.get("schema_version") != 1
        or manifest.get("status") != "complete"
        or identity_record.get("artifact_kind") != CONSENSUS_KIND
        or identity_record.get("status")
        != "FINAL_ENDPOINT_LABELS_LOCKED_BEFORE_ANY_CANDIDATE_SCORE_PRODUCT"
        or identity_record.get("phase") != "confirmation"
        or identity_record.get("event_protocol_identity_sha256")
        != protocol["identity_sha256"]
        or identity_record.get("anchor_plan_identity_sha256") != plan["identity_sha256"]
        or identity_record.get("row_count")
        != len(plan["active_union_classes"]) * len(CONFIRMATION_SEEDS)
        or identity_record.get(
            "candidate_scores_features_trajectories_embeddings_thresholds_or_ranks_opened"
        )
        is not False
        or identity_record.get("three_of_three_never_downgraded") is not True
        or identity_record.get("single_adjudicator_changed_final_label") is not False
        or completion.get("complete") is not True
        or completion.get("identity_sha256") != identity
        or completion.get("manifest_identity_sha256") != manifest_identity
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
        or completion.get("file_count") != len(manifest.get("files", []))
    ):
        raise RuntimeError("confirmation consensus aggregate envelope changed")
    members = _consensus_member_map(manifest)
    aggregate_member = members[CONSENSUS_AGGREGATE]
    if (
        aggregate_member["bytes"] != aggregate_path.stat().st_size
        or aggregate_member["sha256"] != sha256_file(aggregate_path)
    ):
        raise RuntimeError("consensus aggregate member changed")
    active = tuple(plan["active_union_classes"])
    if (
        set(aggregate) != {"phase", "overall", "per_class"}
        or aggregate.get("phase") != "confirmation"
    ):
        raise RuntimeError("consensus aggregate scientific fields changed")
    raw_overall = aggregate.get("overall")
    raw_per_class = aggregate.get("per_class")
    required_count_keys = {
        "endpoint_count",
        "raw_clear_bad",
        "final_clean_good",
        "final_mild_or_disputed",
        "final_clear_bad",
        "final_blur_or_soft_fusion",
        "final_structural_non_blur",
        "union_any_positive",
        "random_decoys",
        "promoted_union_minority",
        "promoted_zero_positive_decoys",
        "downgraded_raw_2of3",
        "unanimous_3of3_retained",
    }
    per_class_keys = {
        "endpoint_count",
        "raw_clear_bad",
        "final_clean_good",
        "final_mild_or_disputed",
        "final_clear_bad",
        "final_blur_or_soft_fusion",
        "final_structural_non_blur",
    }
    if (
        not isinstance(raw_overall, dict)
        or set(raw_overall) != required_count_keys
        or not isinstance(raw_per_class, dict)
        or set(raw_per_class) != {str(class_id) for class_id in active}
        or any(
            not isinstance(raw_per_class[str(class_id)], dict)
            or set(raw_per_class[str(class_id)]) != per_class_keys
            for class_id in active
        )
    ):
        raise RuntimeError("review-pipeline aggregate count schema changed")
    normalized = {
        "overall": {
            "trajectory_count": raw_overall["endpoint_count"],
            "clean_good": raw_overall["final_clean_good"],
            "mild_or_disputed": raw_overall["final_mild_or_disputed"],
            "clear_bad": raw_overall["final_clear_bad"],
            "blur_or_soft_fusion_clear_bad": raw_overall[
                "final_blur_or_soft_fusion"
            ],
        },
        "per_class": [
            {
                "class_id": class_id,
                "trajectory_count": raw_per_class[str(class_id)]["endpoint_count"],
                "clean_good": raw_per_class[str(class_id)]["final_clean_good"],
                "mild_or_disputed": raw_per_class[str(class_id)][
                    "final_mild_or_disputed"
                ],
                "clear_bad": raw_per_class[str(class_id)]["final_clear_bad"],
                "blur_or_soft_fusion_clear_bad": raw_per_class[str(class_id)][
                    "final_blur_or_soft_fusion"
                ],
            }
            for class_id in active
        ],
    }
    counts = validate_counts(normalized, active)
    return {
        "root": str(root),
        "manifest_identity_sha256": identity,
        "manifest_file_sha256": sha256_file(manifest_path),
        "aggregate_file_sha256": sha256_file(aggregate_path),
        "row_member_declared_sha256": members[CONSENSUS_ROWS]["sha256"],
        "row_member_declared_bytes": members[CONSENSUS_ROWS]["bytes"],
        "counts": counts,
    }


def candidate_gate(
    counts: Mapping[str, Any],
    plan: Mapping[str, Any],
    candidate: str,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    classes = candidate_classes(plan, candidate)
    if not classes:
        return {
            "candidate": candidate,
            "anchor_go": False,
            "selected_classes": [],
            "event_endpoint": endpoint_for(candidate),
            "observed_events": 0,
            "observed_clean_good": 0,
            "event_bearing_classes": 0,
            "comparable_classes_with_event_and_clean_good": 0,
            "all_minima_met": False,
            "stage_B_score_open_authorized": False,
            "reason": "anchor_STOP",
        }
    by_class = {row["class_id"]: row for row in counts["per_class"]}
    event_field = (
        "blur_or_soft_fusion_clear_bad" if candidate == B_CANDIDATE else "clear_bad"
    )
    events = sum(by_class[class_id][event_field] for class_id in classes)
    clean = sum(by_class[class_id]["clean_good"] for class_id in classes)
    bearing = sum(by_class[class_id][event_field] > 0 for class_id in classes)
    comparable = sum(
        by_class[class_id][event_field] > 0 and by_class[class_id]["clean_good"] > 0
        for class_id in classes
    )
    rule = protocol["separate_score_label_unlock_gates"][candidate]
    event_minimum = int(
        rule[
            "minimum_blur_or_soft_fusion_clear_bad"
            if candidate == B_CANDIDATE
            else "minimum_total_clear_bad"
        ]
    )
    clean_minimum = int(rule["minimum_clean_good"])
    bearing_minimum = int(rule["minimum_event_bearing_classes"])
    comparable_minimum = int(rule["minimum_comparable_classes_with_event_and_clean_good"])
    passed = bool(
        events >= event_minimum
        and clean >= clean_minimum
        and bearing >= bearing_minimum
        and comparable >= comparable_minimum
    )
    return {
        "candidate": candidate,
        "anchor_go": True,
        "selected_classes": list(classes),
        "event_endpoint": endpoint_for(candidate),
        "minimum_events": event_minimum,
        "observed_events": events,
        "minimum_clean_good": clean_minimum,
        "observed_clean_good": clean,
        "minimum_event_bearing_classes": bearing_minimum,
        "event_bearing_classes": bearing,
        "minimum_comparable_classes_with_event_and_clean_good": comparable_minimum,
        "comparable_classes_with_event_and_clean_good": comparable,
        "all_minima_met": passed,
        "stage_B_score_open_authorized": passed,
        "reason": "all_minima_met" if passed else "candidate_specific_event_gate_failed",
    }


def run_stage_a(args: argparse.Namespace) -> None:
    source_lock = require_directory(args.source_lock, "dynamic source lock")
    source_contract, source_manifest, _ = load_source_lock(source_lock)
    verify_evaluator_source(source_lock, source_manifest)
    protocol = validate_event_protocol(Path(source_contract["event_protocol"]["path"]))
    plan = validate_anchor_plan(args.anchor_plan, protocol)
    consensus = load_consensus_aggregate_only(
        args.consensus_root, protocol=protocol, plan=plan
    )
    gates = {
        candidate: candidate_gate(consensus["counts"], plan, candidate, protocol)
        for candidate in CANDIDATES
    }
    record = {
        "schema_version": 1,
        "status": "CANDIDATE_SEPARATE_EVENT_GATES_COMPLETE_SCORES_UNOPENED",
        "event_protocol_identity_sha256": protocol["identity_sha256"],
        "anchor_plan_identity_sha256": plan["identity_sha256"],
        "dynamic_source_contract_identity_sha256": source_contract["identity_sha256"],
        "consensus_receipt": {
            key: consensus[key]
            for key in (
                "manifest_identity_sha256",
                "manifest_file_sha256",
                "aggregate_file_sha256",
                "row_member_declared_sha256",
                "row_member_declared_bytes",
            )
        },
        "aggregate_counts": consensus["counts"],
        "candidate_gates": gates,
        "access_audit": {
            "consensus_manifest_opened": True,
            "consensus_completion_opened": True,
            "consensus_aggregate_counts_opened": True,
            "consensus_rows_opened_hashed_statted_or_resolved": False,
            "B_product_argument_accepted": False,
            "C_product_argument_accepted": False,
            "B_product_opened_hashed_statted_or_resolved": False,
            "C_product_opened_hashed_statted_or_resolved": False,
            "score_csv_opened": False,
            "image_trace_or_external_representation_opened": False,
            "stage_B_invoked_in_same_process": False,
        },
        "implementation_source_sha256": sha256_file(Path(__file__).resolve()),
    }
    record["identity_sha256"] = canonical_sha256(record)
    publish_artifact(
        args.output,
        artifact_kind="dit_event_rich_candidate_separate_stage_a_gate_v1",
        payloads={
            "stage_a_gate_receipt.json": json.dumps(record, indent=2, sort_keys=True) + "\n",
            "evaluator_source.py": Path(__file__).read_text(encoding="utf-8"),
        },
        manifest_fields={
            "stage_a_receipt_identity_sha256": record["identity_sha256"],
            "anchor_plan_identity_sha256": plan["identity_sha256"],
        },
    )
    print(
        json.dumps(
            {
                "status": record["status"],
                "authorized": {
                    name: gate["stage_B_score_open_authorized"]
                    for name, gate in gates.items()
                },
            },
            sort_keys=True,
        )
    )


def validate_stage_a(
    root: Path,
    *,
    source_contract: Mapping[str, Any],
    protocol: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    manifest, _ = validate_manifest_tree(root)
    if manifest.get("artifact_kind") != "dit_event_rich_candidate_separate_stage_a_gate_v1":
        raise RuntimeError("not an event-rich Stage-A receipt")
    record_path = require_regular(root / "stage_a_gate_receipt.json", "Stage-A receipt")
    record = load_json(record_path)
    if (
        canonical_sha256(without_identity(record)) != record.get("identity_sha256")
        or manifest.get("stage_a_receipt_identity_sha256") != record["identity_sha256"]
        or record.get("status")
        != "CANDIDATE_SEPARATE_EVENT_GATES_COMPLETE_SCORES_UNOPENED"
        or record.get("event_protocol_identity_sha256") != protocol["identity_sha256"]
        or record.get("anchor_plan_identity_sha256") != plan["identity_sha256"]
        or record.get("dynamic_source_contract_identity_sha256")
        != source_contract["identity_sha256"]
        or record.get("implementation_source_sha256") != sha256_file(Path(__file__).resolve())
    ):
        raise RuntimeError("Stage-A receipt lineage changed")
    counts = validate_counts(record.get("aggregate_counts"), plan["active_union_classes"])
    expected_gates = {
        candidate: candidate_gate(counts, plan, candidate, protocol)
        for candidate in CANDIDATES
    }
    if record.get("candidate_gates") != expected_gates:
        raise RuntimeError("Stage-A candidate gates do not replay aggregate counts")
    expected_audit = {
        "consensus_manifest_opened": True,
        "consensus_completion_opened": True,
        "consensus_aggregate_counts_opened": True,
        "consensus_rows_opened_hashed_statted_or_resolved": False,
        "B_product_argument_accepted": False,
        "C_product_argument_accepted": False,
        "B_product_opened_hashed_statted_or_resolved": False,
        "C_product_opened_hashed_statted_or_resolved": False,
        "score_csv_opened": False,
        "image_trace_or_external_representation_opened": False,
        "stage_B_invoked_in_same_process": False,
    }
    if record.get("access_audit") != expected_audit:
        raise RuntimeError("Stage-A no-score access audit changed")
    return record


def parse_bool(value: str, description: str) -> bool:
    if value == "0":
        return False
    if value == "1":
        return True
    raise RuntimeError(f"{description} must be literal 0 or 1")


def load_consensus_rows(
    root: Path,
    *,
    aggregate_receipt: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    root = require_directory(root, "confirmation consensus export")
    path = require_regular(root / CONSENSUS_ROWS, "confirmation consensus rows")
    receipt = aggregate_receipt["consensus_receipt"]
    if (
        path.stat().st_size != receipt["row_member_declared_bytes"]
        or sha256_file(path) != receipt["row_member_declared_sha256"]
    ):
        raise RuntimeError("consensus rows differ from Stage-A declaration")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CONSENSUS_COLUMNS:
            raise RuntimeError("confirmation consensus row columns changed")
        rows: list[dict[str, Any]] = []
        for raw in reader:
            severity = raw["final_severity"]
            if severity not in SEVERITIES:
                raise RuntimeError("invalid final severity")
            blur = parse_bool(raw["blur_component"], "blur_component")
            if blur and severity != "clear_bad":
                raise RuntimeError("blur_component=1 requires clear_bad")
            rows.append(
                {
                    "phase": raw["phase"],
                    "global_seed": int(raw["global_seed"]),
                    "class_id": int(raw["class_id"]),
                    "final_severity": severity,
                    "blur_component": blur,
                }
            )
    expected = exact_pairs(plan, phases=("confirmation",))
    observed = tuple((row["phase"], row["global_seed"], row["class_id"]) for row in rows)
    if len(observed) != len(expected) or len(set(observed)) != len(observed) or set(observed) != set(expected):
        raise RuntimeError("confirmation evaluation-label axis is incomplete, duplicated, or unexpected")
    rows.sort(key=lambda row: (row["global_seed"], tuple(plan["active_union_classes"]).index(row["class_id"])))
    replay = count_rows(rows, plan["active_union_classes"])
    if replay != aggregate_receipt["aggregate_counts"]:
        raise RuntimeError("consensus rows do not replay Stage-A aggregate counts")
    return rows


def count_rows(rows: Sequence[Mapping[str, Any]], classes: Sequence[int]) -> dict[str, Any]:
    def one(subset: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        return {
            "trajectory_count": len(subset),
            "clean_good": sum(row["final_severity"] == "clean_good" for row in subset),
            "mild_or_disputed": sum(
                row["final_severity"] == "mild_or_disputed" for row in subset
            ),
            "clear_bad": sum(row["final_severity"] == "clear_bad" for row in subset),
            "blur_or_soft_fusion_clear_bad": sum(
                row["final_severity"] == "clear_bad" and row["blur_component"]
                for row in subset
            ),
        }

    return {
        "overall": one(rows),
        "per_class": [
            {"class_id": class_id, **one([row for row in rows if row["class_id"] == class_id])}
            for class_id in classes
        ],
    }


def load_candidate_product(
    root: Path,
    *,
    candidate: str,
    plan: Mapping[str, Any],
    protocol: Mapping[str, Any],
    source_contract: Mapping[str, Any],
) -> tuple[dict[tuple[str, int, int], float], dict[str, Any]]:
    reject_forbidden_external_name(str(root), f"{candidate} product path")
    root = require_directory(root, f"{candidate} product")
    manifest, _ = validate_manifest_tree(root)
    expected_kind = f"dit_event_rich_{candidate}_single_score_product_v1"
    if (
        manifest.get("artifact_kind") != expected_kind
        or manifest.get("candidate") != candidate
        or manifest.get("anchor_plan_identity_sha256") != plan["identity_sha256"]
        or manifest.get("dynamic_source_contract_identity_sha256")
        != source_contract["identity_sha256"]
    ):
        raise RuntimeError(f"{candidate} product manifest lineage changed")
    names = set(manifest_map(manifest, f"{candidate} product"))
    if names != {"scores.csv", "column_catalog.json", "product_record.json", "extractor_source.py"}:
        raise RuntimeError(f"{candidate} product payload set changed")
    record = load_json(require_regular(root / "product_record.json", "candidate product record"))
    catalog = load_json(require_regular(root / "column_catalog.json", "candidate column catalog"))
    if (
        canonical_sha256(without_identity(record)) != record.get("identity_sha256")
        or canonical_sha256(without_identity(catalog)) != catalog.get("identity_sha256")
        or manifest.get("product_record_identity_sha256") != record["identity_sha256"]
        or record.get("candidate") != candidate
        or record.get("feature") != feature_for(candidate)
        or record.get("event_protocol_identity_sha256") != protocol["identity_sha256"]
        or record.get("anchor_plan_identity_sha256") != plan["identity_sha256"]
        or record.get("dynamic_source_contract_identity_sha256")
        != source_contract["identity_sha256"]
        or record.get("implementation_source_sha256")
        != source_contract.get("source_snapshots", {}).get(
            "extract_dit_event_rich_candidate_product.py", {}
        ).get("sha256")
        or record.get("labels_reviews_consensus_or_endpoint_judgments_opened") is not False
        or record.get("external_representation_or_distance_opened") is not False
        or record.get("other_candidate_score_computed_or_emitted") is not False
        or catalog.get("candidate") != candidate
        or catalog.get("contains_labels_reviews_consensus_or_endpoint_judgments") is not False
        or catalog.get("contains_inception_dino_fid_embeddings_or_external_distances") is not False
        or catalog.get("contains_other_candidate_score") is not False
    ):
        raise RuntimeError(f"{candidate} product scientific contract changed")
    score_path = require_regular(root / "scores.csv", f"{candidate} scores")
    if record.get("scores_file_sha256") != sha256_file(score_path):
        raise RuntimeError(f"{candidate} score CSV hash changed")
    with score_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = validate_score_columns(reader.fieldnames or (), candidate)
        values: dict[tuple[str, int, int], float] = {}
        for row in reader:
            key = (row["phase"], int(row["global_seed"]), int(row["class_id"]))
            try:
                score = float(row[columns[-1]])
            except ValueError as exc:
                raise RuntimeError("candidate score is not numeric") from exc
            if not math.isfinite(score) or key in values:
                raise RuntimeError("candidate score is non-finite or duplicated")
            values[key] = score
    expected = exact_pairs(plan, candidate=candidate)
    if tuple(values) != expected:
        raise RuntimeError(f"{candidate} score row axis changed")
    if record.get("row_count") != len(values):
        raise RuntimeError(f"{candidate} product row count changed")
    return values, {
        "manifest_identity_sha256": manifest["identity_sha256"],
        "manifest_file_sha256": sha256_file(root / "manifest.json"),
        "product_record_identity_sha256": record["identity_sha256"],
        "scores_file_sha256": sha256_file(score_path),
        "column_catalog_identity_sha256": catalog["identity_sha256"],
        "candidate": candidate,
        "feature": feature_for(candidate),
    }


def positive(row: Mapping[str, Any], candidate: str) -> bool:
    if row["final_severity"] != "clear_bad":
        return False
    return bool(row["blur_component"]) if candidate == B_CANDIDATE else True


def oriented(value: float, candidate: str) -> float:
    return value if candidate == B_CANDIDATE else -value


def auc_summary(
    labels: Sequence[Mapping[str, Any]],
    scores: Mapping[tuple[str, int, int], float],
    *,
    candidate: str,
    classes: Sequence[int],
) -> dict[str, Any]:
    numerator = 0.0
    denominator = 0
    per_class: list[dict[str, Any]] = []
    for class_id in classes:
        positives = np.asarray(
            [
                oriented(scores[("confirmation", row["global_seed"], class_id)], candidate)
                for row in labels
                if row["class_id"] == class_id and positive(row, candidate)
            ],
            dtype=np.float64,
        )
        clean = np.asarray(
            [
                oriented(scores[("confirmation", row["global_seed"], class_id)], candidate)
                for row in labels
                if row["class_id"] == class_id and row["final_severity"] == "clean_good"
            ],
            dtype=np.float64,
        )
        pairs = int(len(positives) * len(clean))
        credit = 0.0
        if pairs:
            delta = positives[:, None] - clean[None, :]
            credit = float(np.sum(delta > 0) + 0.5 * np.sum(delta == 0))
        numerator += credit
        denominator += pairs
        per_class.append(
            {
                "class_id": class_id,
                "positive_count": int(len(positives)),
                "clean_good_count": int(len(clean)),
                "pair_count": pairs,
                "auc": float(credit / pairs) if pairs else None,
            }
        )
    if denominator == 0:
        raise RuntimeError(f"{candidate} has zero primary AUC pair denominator")
    return {
        "candidate": candidate,
        "endpoint": endpoint_for(candidate) + "_vs_clean_good",
        "positive_count": sum(row["positive_count"] for row in per_class),
        "clean_good_count": sum(row["clean_good_count"] for row in per_class),
        "pair_count": denominator,
        "auc": float(numerator / denominator),
        "per_class": per_class,
    }


def _tie_order(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    starts = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1,
        )
    )
    return order, starts


def _batch_credit(codes: np.ndarray, order: np.ndarray, starts: np.ndarray) -> np.ndarray:
    ordered = codes[:, order]
    pos = ordered == 1
    neg = ordered == 0
    pos_group = np.add.reduceat(pos, starts, axis=1).astype(np.float64)
    neg_group = np.add.reduceat(neg, starts, axis=1).astype(np.float64)
    neg_before = np.cumsum(neg_group, axis=1) - neg_group
    return np.sum(pos_group * (neg_before + 0.5 * neg_group), axis=1)


def permutation_p_values(
    labels: Sequence[Mapping[str, Any]],
    score_sets: Mapping[str, Mapping[tuple[str, int, int], float]],
    plans: Mapping[str, Sequence[int]],
    *,
    draws: int,
) -> dict[str, dict[str, Any]]:
    active = tuple(score_sets)
    matrices: dict[str, dict[str, Any]] = {}
    for candidate in active:
        classes = tuple(plans[candidate])
        codes = np.full((len(CONFIRMATION_SEEDS), len(classes)), -1, dtype=np.int8)
        scores = np.empty_like(codes, dtype=np.float64)
        by_key = {(row["global_seed"], row["class_id"]): row for row in labels}
        for seed_index, seed in enumerate(CONFIRMATION_SEEDS):
            for slot, class_id in enumerate(classes):
                row = by_key[(seed, class_id)]
                if row["final_severity"] == "clean_good":
                    codes[seed_index, slot] = 0
                elif positive(row, candidate):
                    codes[seed_index, slot] = 1
                scores[seed_index, slot] = oriented(
                    score_sets[candidate][("confirmation", seed, class_id)], candidate
                )
        orders: list[np.ndarray] = []
        starts: list[np.ndarray] = []
        observed = 0.0
        denominator = 0
        for slot in range(len(classes)):
            order, start = _tie_order(scores[:, slot])
            orders.append(order)
            starts.append(start)
            observed += float(_batch_credit(codes[:, slot][None, :], order, start)[0])
            denominator += int(np.sum(codes[:, slot] == 1) * np.sum(codes[:, slot] == 0))
        if denominator == 0:
            raise RuntimeError(f"{candidate} permutation denominator is zero")
        matrices[candidate] = {
            "codes": codes,
            "orders": orders,
            "starts": starts,
            "observed": observed,
            "denominator": denominator,
        }
    rng = np.random.default_rng(PERMUTATION_SEED)
    exceedances = {candidate: 0 for candidate in active}
    remaining = draws
    while remaining:
        size = min(PERMUTATION_BATCH, remaining)
        permutations = np.stack(
            [rng.permutation(len(CONFIRMATION_SEEDS)) for _ in range(size)]
        )
        for candidate, matrix in matrices.items():
            total = np.zeros(size, dtype=np.float64)
            for slot in range(len(plans[candidate])):
                total += _batch_credit(
                    matrix["codes"][permutations, slot],
                    matrix["orders"][slot],
                    matrix["starts"][slot],
                )
            exceedances[candidate] += int(np.sum(total >= matrix["observed"]))
        remaining -= size
    return {
        candidate: {
            "draws": draws,
            "exceedances": exceedances[candidate],
            "raw_p_value": float((1 + exceedances[candidate]) / (1 + draws)),
            "observed_auc": float(matrix["observed"] / matrix["denominator"]),
            "pair_count": matrix["denominator"],
            "same_global_seed_block_permutation_stream_for_all_open_candidates": True,
        }
        for candidate, matrix in matrices.items()
    }


def holm(raw: Mapping[str, float]) -> dict[str, float]:
    if set(raw) != set(CANDIDATES):
        raise RuntimeError("Holm family must contain exactly B and C")
    order = sorted(CANDIDATES, key=lambda candidate: (raw[candidate], candidate))
    result: dict[str, float] = {}
    running = 0.0
    for index, candidate in enumerate(order):
        adjusted = min(1.0, (len(order) - index) * float(raw[candidate]))
        running = max(running, adjusted)
        result[candidate] = running
    return result


def class_thresholds(
    scores: Mapping[tuple[str, int, int], float],
    *,
    candidate: str,
    classes: Sequence[int],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for class_id in classes:
        values = np.asarray(
            [scores[("calibration", seed, class_id)] for seed in range(1100, 1120)],
            dtype=np.float64,
        )
        if len(values) != 20 or not np.isfinite(values).all():
            raise RuntimeError("class calibration scores are incomplete/non-finite")
        ordered = np.sort(values)
        if candidate == B_CANDIDATE:
            index_010, index_005, tail = 18, 19, "upper"
            comparison = "confirmation_raw_score > threshold"
        else:
            index_010, index_005, tail = 1, 0, "lower"
            comparison = "confirmation_raw_score < threshold"
        result[str(class_id)] = {
            "calibration_count": 20,
            "calibration_values_ordered_by_seed_sha256": canonical_sha256(
                [
                    {
                        "global_seed": seed,
                        "class_id": class_id,
                        "raw_score": float(scores[("calibration", seed, class_id)]),
                    }
                    for seed in range(1100, 1120)
                ]
            ),
            "alpha_0p10": {
                "threshold": float(ordered[index_010]),
                "calibration_order_statistic_1_based": index_010 + 1,
                "finite_sample_bound_fraction": "2/21",
                "tail": tail,
                "strict_comparison": comparison,
            },
            "alpha_0p05": {
                "threshold": float(ordered[index_005]),
                "calibration_order_statistic_1_based": index_005 + 1,
                "finite_sample_bound_fraction": "1/21",
                "tail": tail,
                "strict_comparison": comparison,
            },
        }
    return result


def operating_point(
    labels: Sequence[Mapping[str, Any]],
    scores: Mapping[tuple[str, int, int], float],
    thresholds: Mapping[str, Any],
    *,
    candidate: str,
    classes: Sequence[int],
    alpha: str,
) -> dict[str, Any]:
    tp = fp = positive_n = clean_n = alert_n = excluded = 0
    per_class: list[dict[str, int]] = []
    for class_id in classes:
        threshold = float(thresholds[str(class_id)][alpha]["threshold"])
        class_tp = class_fp = class_alert = 0
        for row in labels:
            if row["class_id"] != class_id:
                continue
            score = scores[("confirmation", row["global_seed"], class_id)]
            alert = score > threshold if candidate == B_CANDIDATE else score < threshold
            is_positive = positive(row, candidate)
            is_clean = row["final_severity"] == "clean_good"
            positive_n += int(is_positive)
            clean_n += int(is_clean)
            if alert:
                alert_n += 1
                class_alert += 1
                if is_positive:
                    tp += 1
                    class_tp += 1
                elif is_clean:
                    fp += 1
                    class_fp += 1
                else:
                    excluded += 1
        per_class.append(
            {
                "class_id": class_id,
                "alert_count": class_alert,
                "true_positive_count": class_tp,
                "false_positive_count": class_fp,
            }
        )
    if positive_n == 0 or clean_n == 0:
        raise RuntimeError("operating point has zero TPR/FPR denominator")
    return {
        "candidate": candidate,
        "alpha": alpha,
        "positive_count": positive_n,
        "clean_good_count": clean_n,
        "alert_count_all_candidate_scope_rows": alert_n,
        "excluded_mild_or_nonendpoint_alert_count": excluded,
        "true_positive_count": tp,
        "false_positive_count": fp,
        "micro_TPR": float(tp / positive_n),
        "micro_FPR": float(fp / clean_n),
        "per_class_counts": per_class,
    }


def evaluate_open_candidates(
    labels: Sequence[Mapping[str, Any]],
    score_sets: Mapping[str, Mapping[tuple[str, int, int], float]],
    plan: Mapping[str, Any],
    protocol: Mapping[str, Any],
    gates: Mapping[str, Any],
    *,
    draws: int,
) -> dict[str, Any]:
    classes = {candidate: candidate_classes(plan, candidate) for candidate in score_sets}
    aucs = {
        candidate: auc_summary(
            labels,
            scores,
            candidate=candidate,
            classes=classes[candidate],
        )
        for candidate, scores in score_sets.items()
    }
    permutations = permutation_p_values(labels, score_sets, classes, draws=draws) if score_sets else {}
    raw_p = {
        candidate: (
            permutations[candidate]["raw_p_value"]
            if candidate in permutations
            else 1.0
        )
        for candidate in CANDIDATES
    }
    adjusted = holm(raw_p)
    results: dict[str, Any] = {}
    for candidate in CANDIDATES:
        if candidate not in score_sets:
            results[candidate] = {
                "status": "EVENT_GATED_OFF_SCORE_PRODUCT_UNOPENED",
                "event_gate": gates[candidate],
                "raw_permutation_p_value": 1.0,
                "Holm_adjusted_p_value": adjusted[candidate],
                "passes_all_candidate_gates": False,
            }
            continue
        candidate_thresholds = class_thresholds(
            score_sets[candidate], candidate=candidate, classes=classes[candidate]
        )
        operating = {
            alpha: operating_point(
                labels,
                score_sets[candidate],
                candidate_thresholds,
                candidate=candidate,
                classes=classes[candidate],
                alpha=alpha,
            )
            for alpha in ("alpha_0p10", "alpha_0p05")
        }
        auc_gate = float(protocol["candidates"][candidate]["auc_gate"])
        p_gate = adjusted[candidate] < 0.05
        pass_auc = aucs[candidate]["auc"] >= auc_gate
        if candidate == B_CANDIDATE:
            alpha10 = operating["alpha_0p10"]
            pass_operating = bool(
                alpha10["true_positive_count"] >= 3
                and alpha10["micro_TPR"] > alpha10["micro_FPR"]
            )
        else:
            pass_operating = True
        results[candidate] = {
            "status": "SCORE_OPENED_AND_FROZEN_TEST_COMPLETED",
            "event_gate": gates[candidate],
            "auc": aucs[candidate],
            "permutation": permutations[candidate],
            "raw_permutation_p_value": raw_p[candidate],
            "Holm_adjusted_p_value": adjusted[candidate],
            "class_specific_calibration_thresholds": candidate_thresholds,
            "operating_points": operating,
            "gate_replay": {
                "auc_at_least": auc_gate,
                "auc_pass": pass_auc,
                "Holm_adjusted_p_strictly_below": 0.05,
                "Holm_pass": p_gate,
                "B_alpha_0p10_TP_at_least_3_and_TPR_strictly_above_FPR_pass": (
                    pass_operating if candidate == B_CANDIDATE else None
                ),
            },
            "passes_all_candidate_gates": bool(pass_auc and p_gate and pass_operating),
        }
    return {
        "candidate_results": results,
        "multiple_testing": {
            "family": list(CANDIDATES),
            "raw_p_values": raw_p,
            "Holm_adjusted_p_values": adjusted,
            "gated_off_candidate_raw_p_fixed_to_1": True,
        },
    }


def run_stage_b(args: argparse.Namespace) -> None:
    source_lock = require_directory(args.source_lock, "dynamic source lock")
    source_contract, source_manifest, _ = load_source_lock(source_lock)
    verify_evaluator_source(source_lock, source_manifest)
    protocol = validate_event_protocol(Path(source_contract["event_protocol"]["path"]))
    plan = validate_anchor_plan(args.anchor_plan, protocol)
    stage_a = validate_stage_a(
        args.stage_a_receipt,
        source_contract=source_contract,
        protocol=protocol,
        plan=plan,
    )
    # First row-level access occurs here, after both candidate gate decisions
    # have already been immutably published by a separate process.
    labels = load_consensus_rows(
        args.consensus_root, aggregate_receipt=stage_a, plan=plan
    )
    score_paths = {B_CANDIDATE: args.B_product, C_CANDIDATE: args.C_product}
    score_sets: dict[str, dict[tuple[str, int, int], float]] = {}
    product_receipts: dict[str, Any] = {}
    opened = {candidate: False for candidate in CANDIDATES}
    for candidate in CANDIDATES:
        authorized = stage_a["candidate_gates"][candidate][
            "stage_B_score_open_authorized"
        ]
        path = score_paths[candidate]
        if not authorized:
            # Deliberately do not inspect truthiness, exists, stat, resolve, or
            # string content of the candidate product path in this branch.
            continue
        if path is None:
            raise RuntimeError(f"{candidate} gate passed but its product path was omitted")
        values, receipt = load_candidate_product(
            path,
            candidate=candidate,
            plan=plan,
            protocol=protocol,
            source_contract=source_contract,
        )
        score_sets[candidate] = values
        product_receipts[candidate] = receipt
        opened[candidate] = True
    analysis = evaluate_open_candidates(
        labels,
        score_sets,
        plan,
        protocol,
        stage_a["candidate_gates"],
        draws=args.permutation_draws,
    )
    record = {
        "schema_version": 1,
        "status": "EVENT_RICH_DYNAMIC_CONFIRMATION_EVALUATION_COMPLETE",
        "event_protocol_identity_sha256": protocol["identity_sha256"],
        "anchor_plan_identity_sha256": plan["identity_sha256"],
        "dynamic_source_contract_identity_sha256": source_contract["identity_sha256"],
        "stage_a_receipt_identity_sha256": stage_a["identity_sha256"],
        "consensus_receipt": stage_a["consensus_receipt"],
        "candidate_product_receipts_opened": product_receipts,
        **analysis,
        "intervention_authority": {
            "B_only": True,
            "B_authorized": analysis["candidate_results"][B_CANDIDATE][
                "passes_all_candidate_gates"
            ],
            "C_never_authorizes_intervention_by_itself": True,
        },
        "access_audit": {
            "consensus_rows_opened_after_stage_a": True,
            "B_product_opened": opened[B_CANDIDATE],
            "C_product_opened": opened[C_CANDIDATE],
            "B_product_unopened_if_gate_failed": bool(
                opened[B_CANDIDATE]
                or not stage_a["candidate_gates"][B_CANDIDATE][
                    "stage_B_score_open_authorized"
                ]
            ),
            "C_product_unopened_if_gate_failed": bool(
                opened[C_CANDIDATE]
                or not stage_a["candidate_gates"][C_CANDIDATE][
                    "stage_B_score_open_authorized"
                ]
            ),
            "image_or_trace_opened": False,
            "inception_dino_fid_embedding_or_external_distance_opened": False,
            "row_score_rank_or_permutation_draw_emitted": False,
        },
        "implementation_source_sha256": sha256_file(Path(__file__).resolve()),
    }
    record["identity_sha256"] = canonical_sha256(record)
    publish_artifact(
        args.output,
        artifact_kind="dit_event_rich_dynamic_confirmation_stage_b_v1",
        payloads={
            "aggregate_evaluation.json": json.dumps(record, indent=2, sort_keys=True) + "\n",
            "evaluator_source.py": Path(__file__).read_text(encoding="utf-8"),
        },
        manifest_fields={
            "evaluation_identity_sha256": record["identity_sha256"],
            "anchor_plan_identity_sha256": plan["identity_sha256"],
        },
    )
    print(
        json.dumps(
            {
                "status": record["status"],
                "products_opened": opened,
                "passes": {
                    candidate: analysis["candidate_results"][candidate][
                        "passes_all_candidate_gates"
                    ]
                    for candidate in CANDIDATES
                },
            },
            sort_keys=True,
        )
    )


def run_self_test() -> None:
    classes_b = (1, 2, 3, 4, 5, 6)
    classes_c = (4, 5, 6, 7, 8, 9)
    plan = {
        "B_decision": {"go": True, "selected_classes": list(classes_b)},
        "C_decision": {"go": True, "selected_classes": list(classes_c)},
    }
    labels: list[dict[str, Any]] = []
    union = tuple(range(1, 10))
    for seed in CONFIRMATION_SEEDS:
        for class_id in union:
            bad = (seed - CONFIRMATION_SEEDS[0]) % 16 == class_id % 16
            labels.append(
                {
                    "phase": "confirmation",
                    "global_seed": seed,
                    "class_id": class_id,
                    "final_severity": "clear_bad" if bad else "clean_good",
                    "blur_component": bool(bad and class_id in classes_b),
                }
            )
    scores: dict[str, dict[tuple[str, int, int], float]] = {}
    for candidate, classes in ((B_CANDIDATE, classes_b), (C_CANDIDATE, classes_c)):
        values: dict[tuple[str, int, int], float] = {}
        for seed in range(1100, 1120):
            for class_id in classes:
                values[("calibration", seed, class_id)] = float(seed - 1100)
        for row in labels:
            if row["class_id"] not in classes:
                continue
            signal = 100.0 if positive(row, candidate) else 0.0
            raw = signal if candidate == B_CANDIDATE else -signal
            values[("confirmation", row["global_seed"], row["class_id"])] = raw
        scores[candidate] = values
    for candidate, classes in ((B_CANDIDATE, classes_b), (C_CANDIDATE, classes_c)):
        summary = auc_summary(labels, scores[candidate], candidate=candidate, classes=classes)
        if summary["auc"] != 1.0:
            raise AssertionError("directional class-matched AUC changed")
        thresholds = class_thresholds(scores[candidate], candidate=candidate, classes=classes)
        first = thresholds[str(classes[0])]
        expected_010 = 18.0 if candidate == B_CANDIDATE else 1.0
        if first["alpha_0p10"]["threshold"] != expected_010:
            raise AssertionError("20-sample order-statistic threshold changed")
    permutation = permutation_p_values(
        labels,
        scores,
        {B_CANDIDATE: classes_b, C_CANDIDATE: classes_c},
        draws=31,
    )
    if set(permutation) != set(CANDIDATES) or any(row["draws"] != 31 for row in permutation.values()):
        raise AssertionError("shared permutation family changed")
    adjusted = holm({B_CANDIDATE: 0.01, C_CANDIDATE: 1.0})
    if adjusted[B_CANDIDATE] != 0.02 or adjusted[C_CANDIDATE] != 1.0:
        raise AssertionError("Holm with event-gated p=1 changed")
    poison_header = ("phase", "global_seed", "class_id", B_FEATURE, "label")
    try:
        validate_score_columns(poison_header, B_CANDIDATE)
    except RuntimeError:
        pass
    else:
        raise AssertionError("label-column poison escaped Stage B")
    for path in ("fid_product", "DINO_scores", "inception_distance"):
        try:
            reject_forbidden_external_name(path, "synthetic product")
        except RuntimeError:
            pass
        else:
            raise AssertionError("external representation path escaped Stage B")
    print(
        "self-test passed: separate gates, directional AUC, shared block permutation, "
        "Holm p=1, fixed thresholds, and score/external poisons"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-lock", type=Path, default=DEFAULT_DYNAMIC_SOURCE_LOCK)
    sub = parser.add_subparsers(dest="command", required=True)
    stage_a = sub.add_parser("stage-a")
    stage_a.add_argument("--anchor-plan", type=Path, required=True)
    stage_a.add_argument("--consensus-root", type=Path, required=True)
    stage_a.add_argument("--output", type=Path, required=True)
    stage_a.set_defaults(func=run_stage_a)
    stage_b = sub.add_parser("stage-b")
    stage_b.add_argument("--anchor-plan", type=Path, required=True)
    stage_b.add_argument("--consensus-root", type=Path, required=True)
    stage_b.add_argument("--stage-a-receipt", type=Path, required=True)
    stage_b.add_argument("--B-product", type=Path)
    stage_b.add_argument("--C-product", type=Path)
    stage_b.add_argument("--output", type=Path, required=True)
    stage_b.add_argument("--permutation-draws", type=int, default=PERMUTATION_DRAWS)
    stage_b.set_defaults(func=run_stage_b)
    test = sub.add_parser("self-test")
    test.set_defaults(func=lambda _args: run_self_test())
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if getattr(args, "permutation_draws", 1) <= 0:
        raise SystemExit("--permutation-draws must be positive")
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
