#!/usr/bin/env python3
"""Post-failed-gate exploratory label sensitivity audit for third-pool B/C.

This audit is deliberately non-confirmatory.  Its only question is whether the
already-frozen B and C summaries change materially when the endpoint labels are
defined by final adjudication, raw reviewer majority, or one reviewer at a time.
It exists only to inform design of a future pool after the formal third-pool
event gate failed. It cannot authorize an intervention or override that gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from . import evaluate_dit_bad_good_third_pool_confirmation as primary
    from . import prepare_dit_bad_good_third_pool_blind_reviews as reviews
except ImportError:
    import evaluate_dit_bad_good_third_pool_confirmation as primary
    import prepare_dit_bad_good_third_pool_blind_reviews as reviews


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_LOCK = (
    ROOT / "experiments/locks/dit_third_pool_bc_label_sensitivity_source_lock_v4"
)

EXPERIMENT = "dit_third_pool_bc_label_sensitivity_v4"
SOURCE_ARTIFACT_KIND = "dit_third_pool_bc_label_sensitivity_source_lock_v4"
RESULT_ARTIFACT_KIND = "dit_third_pool_bc_label_sensitivity_result_v4"
PROTOCOL_NAME = "protocol.json"
RESULT_NAME = "label_sensitivity_result.json"
SOURCE_NAMES = (
    "audit_dit_third_pool_bc_label_sensitivity.py",
    "freeze_dit_third_pool_bc_label_sensitivity.py",
    "evaluate_dit_bad_good_third_pool_confirmation.py",
    "prepare_dit_bad_good_third_pool_blind_reviews.py",
)

CLASSES = primary.CLASSES
SEEDS = primary.SEEDS
TRAJECTORY_COUNT = primary.TRAJECTORY_COUNT
REVIEWERS = reviews.REVIEWERS
LABEL_DEFINITIONS = (
    "final_adjudicated",
    "raw_majority",
    "reviewer_1",
    "reviewer_2",
    "reviewer_3",
)
CANDIDATES = ("B_blur_mean", "C_c3_low_jump")
FEATURES = {
    "B_blur_mean": primary.VISUAL_FEATURE,
    "C_c3_low_jump": primary.PRIMARY_FEATURE,
}
DIRECTIONS = {"B_blur_mean": "high_is_bad", "C_c3_low_jump": "low_is_bad"}
ALPHA = "alpha_0p10"
EXPECTED_FAILED_TOTAL_BAD = 6
EXPECTED_FAILED_BLUR_BAD = 4
SUPERSEDED_V1_MANIFEST_IDENTITY = (
    "b3e4fd24d471b7fc5dc71067262abe1b4029d6cf675fec2672261633f3a822f8"
)
SUPERSEDED_V2_MANIFEST_IDENTITY = (
    "088c54a76cf7055c6946f8437804ef3e440d0249dc272a6b91022916aaf8d127"
)
SUPERSEDED_V3_MANIFEST_IDENTITY = (
    "d4a59bc7e9f0dc2939b5b0714e072b85a375a6742ffb532750f92ef49dfe5bdb"
)
LEGACY_PLACEHOLDER_COLUMNS = ("label", "raw_consensus_label")
RAW_CONSENSUS_EMPTY_SENTINELS = ("", "nan", "NaN")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def without_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("identity_sha256", None)
    return result


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def require_hex64(value: Any, description: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise RuntimeError(f"{description} must be a sha256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise RuntimeError(f"{description} must be a sha256 hex digest") from exc
    return value


def source_paths() -> dict[str, Path]:
    return {
        "audit_dit_third_pool_bc_label_sensitivity.py": Path(__file__).resolve(),
        "freeze_dit_third_pool_bc_label_sensitivity.py": (
            ROOT / "experiments/freeze_dit_third_pool_bc_label_sensitivity.py"
        ),
        "evaluate_dit_bad_good_third_pool_confirmation.py": (
            ROOT / "experiments/evaluate_dit_bad_good_third_pool_confirmation.py"
        ),
        "prepare_dit_bad_good_third_pool_blind_reviews.py": (
            ROOT / "experiments/prepare_dit_bad_good_third_pool_blind_reviews.py"
        ),
    }


def scientific_protocol(source_records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    protocol: dict[str, Any] = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "designation": "POST_FAILED_GATE_EXPLORATORY_ONLY_FOR_NEXT_POOL_DESIGN",
        "question": (
            "How sensitive are the two already-frozen B/C associations to five "
            "pre-enumerated blind-label definitions after the formal third-pool "
            "event gate failed?"
        ),
        "supersession": {
            "v1_source_lock_manifest_identity_sha256": SUPERSEDED_V1_MANIFEST_IDENTITY,
            "v1_status": "SUPERSEDED_NO_RUN",
            "v1_reason": (
                "v1 incorrectly required a passed Stage-A receipt and therefore "
                "could not answer the post-failure design question"
            ),
            "v2_source_lock_manifest_identity_sha256": SUPERSEDED_V2_MANIFEST_IDENTITY,
            "v2_status": "SUPERSEDED_AFTER_FAIL_CLOSED_FIRST_REAL_ATTEMPT",
            "v2_failure_fact": (
                "the first real v2 attempt rejected the upstream primary CSV header "
                "before opening any C score value or producing an output artifact"
            ),
            "v2_failure_cause": (
                "the otherwise label-free upstream product retained fixed placeholder "
                "columns label and raw_consensus_label; V5's generic label-like-header "
                "guard rejects raw_consensus_label"
            ),
            "v3_source_lock_manifest_identity_sha256": SUPERSEDED_V3_MANIFEST_IDENTITY,
            "v3_status": "SUPERSEDED_AFTER_FAIL_CLOSED_FIRST_REAL_ATTEMPT",
            "v3_failure_fact": (
                "the first real v3 attempt rejected the visual B header before "
                "producing an output artifact"
            ),
            "v3_failed_output_target": (
                "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
                "bad_good_metric_confirmation_third_pool_v1/"
                "bc_label_sensitivity_exploratory_v3"
            ),
            "v3_failed_output_target_confirmed_absent_before_v4_freeze": True,
            "v3_failure_cause": (
                "V5's generic label-like substring guard interpreted the legitimate "
                "frozen feature decoded_local_blur_severity__maximum as label metadata"
            ),
            "v1_snapshot_must_remain_unchanged": True,
            "v2_snapshot_must_remain_unchanged": True,
            "v3_snapshot_must_remain_unchanged": True,
        },
        "temporal_firewall": {
            "protocol_and_all_executable_sources_frozen_before_any_third_pool_primary_or_visual_score_value_is_opened": True,
            "allowed_before_freeze": [
                "locked label/reviewer/consensus schemas",
                "aggregate event counts",
                "artifact paths and identities without score payload access",
            ],
            "production_join_requires_valid_replayed_stage_A_failure_receipt": True,
            "required_stage_A_status": "EVENT_GATE_FAILED_NO_SCORE_ACCESS",
            "required_replayed_counts": {
                "total_clear_bad": EXPECTED_FAILED_TOTAL_BAD,
                "blur_or_soft_fusion_clear_bad": EXPECTED_FAILED_BLUR_BAD,
            },
            "receipt_must_attest_primary_and_visual_feature_paths_unopened": True,
            "failed_stage_A_remains_failed_and_cannot_be_overridden": True,
        },
        "upstream_schema_defect_exception": {
            "decision_basis_only": [
                "upstream CSV schema/header",
                "fixed placeholder sentinel semantics",
            ],
            "candidate_score_values_used_to_define_exception": False,
            "label_values_or_outcomes_used_to_define_exception": False,
            "required_placeholder_columns_exact": list(LEGACY_PLACEHOLDER_COLUMNS),
            "required_label_value_every_row": "unlabeled",
            "allowed_raw_consensus_label_values_exact": list(
                RAW_CONSENSUS_EMPTY_SENTINELS
            ),
            "all_other_label_like_columns_forbidden": True,
            "only_fixed_C_feature_is_numeric_parsed": FEATURES["C_c3_low_jump"],
            "exception_changes_candidate_label_threshold_or_direction": False,
            "formal_evaluator_V5_would_also_reject_if_Stage_B_were_reached": True,
            "formal_evaluator_V5_never_reached_Stage_B_because_Stage_A_already_failed": True,
            "formal_failed_gate_result_affected": False,
            "future_pipeline_required_fix": (
                "label-free products must omit label and raw_consensus_label entirely"
            ),
        },
        "visual_B_explicit_role_contract": {
            "decision_basis_only": [
                "frozen visual CSV header",
                "frozen visual feature catalog",
                "pinned visual product manifest and extractor identity",
            ],
            "candidate_score_values_used_to_define_loader": False,
            "fixed_numeric_feature": FEATURES["B_blur_mean"],
            "numeric_parsed_feature_columns_exact": [FEATURES["B_blur_mean"]],
            "fixed_metadata_columns_exact": [
                "sample_index",
                "run_index",
                "global_seed",
                "class_slot",
                "class_id",
                "trace_dir",
                "endpoint_png_path",
            ],
            "all_other_columns_must_be_registered_in_frozen_feature_catalog": True,
            "catalog_registered_feature_names_may_contain_blur_or_severity": True,
            "uncatalogued_or_label_metadata_columns_forbidden": True,
            "duplicate_missing_extra_or_renamed_columns_forbidden": True,
            "candidate_label_threshold_or_direction_changed": False,
            "future_formal_schema_rule": (
                "label-free products must classify every column by an explicit role "
                "and frozen catalog, not by substring matching on feature names"
            ),
        },
        "cohort": {
            "classes_ordered": list(CLASSES),
            "global_seeds": [SEEDS[0], SEEDS[-1]],
            "global_seed_semantics": "inclusive endpoints; exact frozen set range(250,850)",
            "trajectory_count": TRAJECTORY_COUNT,
            "complete_class_matched_cohort_required": True,
        },
        "candidates": {
            "B_blur_mean": {
                "feature": FEATURES["B_blur_mean"],
                "direction": DIRECTIONS["B_blur_mean"],
                "positive_endpoint_for_every_label_definition": "clear-bad AND blur-or-soft-fusion phenotype",
                "negative_endpoint_for_every_label_definition": "clean-good under the same label definition",
            },
            "C_c3_low_jump": {
                "feature": FEATURES["C_c3_low_jump"],
                "direction": DIRECTIONS["C_c3_low_jump"],
                "positive_endpoint_for_every_label_definition": "all clear-bad",
                "negative_endpoint_for_every_label_definition": "clean-good under the same label definition",
            },
        },
        "label_definitions_ordered": {
            "final_adjudicated": {
                "bad": "final_severity == clear_bad",
                "good": "final_severity == clean_good",
                "blur": "frozen two-of-three blur_component_consensus",
            },
            "raw_majority": {
                "bad": "at least two of three reviewer severities are 2 or 3",
                "good": "at least two of three reviewer severities are 0",
                "blur": "at least two reviewers mark any frozen blur-group flag",
            },
            "reviewer_1": {
                "bad": "reviewer_1 severity is 2 or 3",
                "good": "reviewer_1 severity is 0",
                "blur": "reviewer_1 marks any frozen blur-group flag",
            },
            "reviewer_2": {
                "bad": "reviewer_2 severity is 2 or 3",
                "good": "reviewer_2 severity is 0",
                "blur": "reviewer_2 marks any frozen blur-group flag",
            },
            "reviewer_3": {
                "bad": "reviewer_3 severity is 2 or 3",
                "good": "reviewer_3 severity is 0",
                "blur": "reviewer_3 marks any frozen blur-group flag",
            },
        },
        "statistics": {
            "auc": (
                "class-matched pair-weighted Mann-Whitney AUC; ties receive 0.5; "
                "orientation fixed by candidate"
            ),
            "per_class_reporting": (
                "counts, pair counts, and AUC; numeric class AUC suppressed when "
                "either endpoint has fewer than five samples"
            ),
            "fixed_operating_point": {
                "threshold_source": "frozen old-data third-pool threshold lock",
                "alpha": ALPHA,
                "comparison_B": "raw B > class-specific frozen threshold",
                "comparison_C": "raw C < class-specific frozen threshold",
                "reported": [
                    "positive and clean-good counts",
                    "TP and FP counts",
                    "micro TPR and FPR",
                    "all-trajectory alert count and rate",
                    "excluded non-endpoint alert count",
                ],
            },
            "label_definition_sensitivity": {
                "for_each_candidate": [
                    "AUC min, max, range across the five definitions",
                    "micro TPR min, max, range across the five definitions",
                    "micro FPR min, max, range across the five definitions",
                    "each non-final AUC minus final-adjudicated AUC",
                ],
                "missing_value_rule": "a range or contrast is null if a required endpoint denominator is zero",
            },
            "confirmatory_p_values": False,
            "confidence_intervals": False,
            "new_feature_direction_threshold_or_label_selection": False,
            "candidate_combination": False,
        },
        "output_contract": {
            "aggregate_only": True,
            "forbidden": [
                "row-level score",
                "row-level label",
                "row-level rank",
                "sample identifier joined to any score or label",
                "confirmatory p-value",
                "pass/fail claim",
            ],
            "cannot_call_candidate_passed": True,
            "cannot_rescue_or_override_failed_stage_A_or_primary_stage_B": True,
            "intervention_experiment_authorized": False,
        },
        "foundation_identity_pins": {
            "phase1_protocol_identity_sha256": primary.EXPECTED_PHASE1_PROTOCOL_IDENTITY,
            "phase1_threshold_identity_sha256": primary.EXPECTED_PHASE1_THRESHOLD_IDENTITY,
            "sampling_protocol_identity_sha256": primary.EXPECTED_SAMPLING_PROTOCOL_IDENTITY,
            "primary_evaluator_source_sha256": source_records[
                "evaluate_dit_bad_good_third_pool_confirmation.py"
            ]["sha256"],
            "blind_review_pipeline_source_sha256": source_records[
                "prepare_dit_bad_good_third_pool_blind_reviews.py"
            ]["sha256"],
        },
        "sources": {name: dict(source_records[name]) for name in SOURCE_NAMES},
    }
    protocol["identity_sha256"] = canonical_sha256(protocol)
    return protocol


def _manifest_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {"manifest.json", "completion.json"}:
            records.append(
                {
                    "name": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return records


def freeze_source_lock(output: Path) -> Path:
    if output.exists():
        raise RuntimeError(f"refusing to overwrite source lock: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        source_dir = staging / "sources"
        source_dir.mkdir()
        records: dict[str, dict[str, Any]] = {}
        for name, source in source_paths().items():
            if not source.is_file() or source.is_symlink():
                raise RuntimeError(f"source is unavailable or unsafe: {source}")
            destination = source_dir / name
            shutil.copy2(source, destination)
            records[name] = {
                "live_path_at_freeze": str(source),
                "sha256": sha256_file(destination),
            }
        protocol = scientific_protocol(records)
        write_json(staging / PROTOCOL_NAME, protocol)
        files = _manifest_records(staging)
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "complete",
            "experiment": EXPERIMENT,
            "artifact_kind": SOURCE_ARTIFACT_KIND,
            "protocol_identity_sha256": protocol["identity_sha256"],
            "files": files,
        }
        manifest["identity_sha256"] = canonical_sha256(manifest)
        write_json(staging / "manifest.json", manifest)
        write_json(
            staging / "completion.json",
            {
                "complete": True,
                "third_pool_primary_or_visual_score_values_opened": False,
                "score_label_join_performed": False,
                "manifest_file_sha256": sha256_file(staging / "manifest.json"),
                "manifest_identity_sha256": manifest["identity_sha256"],
                "protocol_file_sha256": sha256_file(staging / PROTOCOL_NAME),
                "protocol_identity_sha256": protocol["identity_sha256"],
            },
        )
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    validate_source_lock(output)
    return output


def validate_source_lock(
    root: Path, expected_manifest_identity: str | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError("source lock must be a real directory")
    expected_names = {
        PROTOCOL_NAME,
        "manifest.json",
        "completion.json",
        *{f"sources/{name}" for name in SOURCE_NAMES},
    }
    observed: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise RuntimeError("source lock contains an unsafe filesystem entry")
        if path.is_file():
            observed.add(path.relative_to(root).as_posix())
    if observed != expected_names:
        raise RuntimeError("source-lock exact member set changed")
    protocol = load_json(root / PROTOCOL_NAME)
    manifest = load_json(root / "manifest.json")
    completion = load_json(root / "completion.json")
    manifest_identity = require_hex64(
        manifest.get("identity_sha256"), "source-lock manifest identity"
    )
    if expected_manifest_identity is not None and manifest_identity != require_hex64(
        expected_manifest_identity, "expected source-lock manifest identity"
    ):
        raise RuntimeError("source-lock manifest identity differs from the pinned value")
    records = protocol.get("sources")
    if not isinstance(records, dict) or set(records) != set(SOURCE_NAMES):
        raise RuntimeError("source records changed")
    for name in SOURCE_NAMES:
        copied = root / "sources" / name
        live = source_paths()[name]
        expected_hash = require_hex64(records[name].get("sha256"), f"{name} source hash")
        if sha256_file(copied) != expected_hash or sha256_file(live) != expected_hash:
            raise RuntimeError(f"frozen/live source mismatch: {name}")
    if (
        protocol.get("experiment") != EXPERIMENT
        or protocol.get("designation")
        != "POST_FAILED_GATE_EXPLORATORY_ONLY_FOR_NEXT_POOL_DESIGN"
        or canonical_sha256(without_identity(protocol))
        != require_hex64(protocol.get("identity_sha256"), "protocol identity")
        or manifest.get("status") != "complete"
        or manifest.get("experiment") != EXPERIMENT
        or manifest.get("artifact_kind") != SOURCE_ARTIFACT_KIND
        or manifest.get("protocol_identity_sha256") != protocol["identity_sha256"]
        or manifest.get("files") != _manifest_records(root)
        or canonical_sha256(without_identity(manifest)) != manifest_identity
        or completion.get("complete") is not True
        or completion.get("third_pool_primary_or_visual_score_values_opened") is not False
        or completion.get("score_label_join_performed") is not False
        or completion.get("manifest_file_sha256") != sha256_file(root / "manifest.json")
        or completion.get("manifest_identity_sha256") != manifest_identity
        or completion.get("protocol_file_sha256") != sha256_file(root / PROTOCOL_NAME)
        or completion.get("protocol_identity_sha256") != protocol["identity_sha256"]
        or scientific_protocol(records) != protocol
    ):
        raise RuntimeError("source-lock scientific contract or receipt changed")
    return protocol, manifest


def _oriented(raw: float, candidate: str) -> float:
    return raw if candidate == "B_blur_mean" else -raw


def _labels(row: Mapping[str, Any], definition: str, candidate: str) -> tuple[bool, bool]:
    bad = bool(row[f"{definition}_bad"])
    good = bool(row[f"{definition}_good"])
    if bad and good:
        raise RuntimeError("a row cannot be both bad and good")
    if candidate == "B_blur_mean":
        bad = bad and bool(row[f"{definition}_blur"])
    return bad, good


def auc_summary(
    rows: Sequence[Mapping[str, Any]], *, candidate: str, definition: str
) -> dict[str, Any]:
    total_credit = 0.0
    total_pairs = 0
    per_class: list[dict[str, Any]] = []
    for class_id in CLASSES:
        positives = np.asarray(
            [
                _oriented(float(row[f"{candidate}_score"]), candidate)
                for row in rows
                if row["class_id"] == class_id
                and _labels(row, definition, candidate)[0]
            ],
            dtype=np.float64,
        )
        negatives = np.asarray(
            [
                _oriented(float(row[f"{candidate}_score"]), candidate)
                for row in rows
                if row["class_id"] == class_id
                and _labels(row, definition, candidate)[1]
            ],
            dtype=np.float64,
        )
        pairs = int(len(positives) * len(negatives))
        credit = 0.0
        if pairs:
            delta = positives[:, None] - negatives[None, :]
            credit = float(np.sum(delta > 0.0) + 0.5 * np.sum(delta == 0.0))
        total_credit += credit
        total_pairs += pairs
        reportable = len(positives) >= 5 and len(negatives) >= 5
        per_class.append(
            {
                "class_id": class_id,
                "positive_count": int(len(positives)),
                "clean_good_count": int(len(negatives)),
                "pair_count": pairs,
                "auc": float(credit / pairs) if pairs and reportable else None,
                "numeric_auc_suppressed": bool(pairs and not reportable),
            }
        )
    positive_count = sum(item["positive_count"] for item in per_class)
    good_count = sum(item["clean_good_count"] for item in per_class)
    return {
        "candidate": candidate,
        "label_definition": definition,
        "endpoint": (
            "blur_or_soft_fusion_bad_vs_same_definition_clean_good"
            if candidate == "B_blur_mean"
            else "all_bad_vs_same_definition_clean_good"
        ),
        "positive_count": positive_count,
        "clean_good_count": good_count,
        "pair_count": total_pairs,
        "auc": float(total_credit / total_pairs) if total_pairs else None,
        "zero_total_pair_denominator": total_pairs == 0,
        "per_class": per_class,
    }


def operating_point(
    rows: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
    *,
    candidate: str,
    definition: str,
) -> dict[str, Any]:
    tp = fp = alerts = excluded = positives = negatives = 0
    per_class: list[dict[str, int]] = []
    for class_id in CLASSES:
        threshold = float(
            thresholds[candidate]["classes"][str(class_id)][ALPHA]["threshold"]
        )
        class_alert = class_tp = class_fp = 0
        for row in rows:
            if row["class_id"] != class_id:
                continue
            raw = float(row[f"{candidate}_score"])
            alert = raw > threshold if candidate == "B_blur_mean" else raw < threshold
            positive, negative = _labels(row, definition, candidate)
            positives += int(positive)
            negatives += int(negative)
            if alert:
                alerts += 1
                class_alert += 1
                if positive:
                    tp += 1
                    class_tp += 1
                elif negative:
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
    return {
        "candidate": candidate,
        "label_definition": definition,
        "alpha": ALPHA,
        "comparison": (
            "raw_score > frozen class-specific threshold"
            if candidate == "B_blur_mean"
            else "raw_score < frozen class-specific threshold"
        ),
        "positive_count": positives,
        "clean_good_count": negatives,
        "alert_count_all_trajectories": alerts,
        "alert_rate_all_trajectories": float(alerts / len(rows)),
        "excluded_nonendpoint_alert_count": excluded,
        "true_positive_count": tp,
        "false_positive_count": fp,
        "micro_TPR": float(tp / positives) if positives else None,
        "micro_FPR": float(fp / negatives) if negatives else None,
        "per_class_counts": per_class,
    }


def _range(values: Sequence[float | None]) -> dict[str, float | None]:
    if any(value is None for value in values):
        return {"min": None, "max": None, "range": None}
    concrete = [float(value) for value in values if value is not None]
    return {
        "min": min(concrete),
        "max": max(concrete),
        "range": max(concrete) - min(concrete),
    }


def evaluate_rows(
    rows: Sequence[Mapping[str, Any]], thresholds: Mapping[str, Any]
) -> dict[str, Any]:
    if len(rows) != TRAJECTORY_COUNT:
        raise RuntimeError("joined cohort must contain exactly 1800 trajectories")
    if set(thresholds) != set(CANDIDATES):
        raise RuntimeError("threshold family must contain exactly frozen B and C")
    definitions: dict[str, Any] = {}
    for definition in LABEL_DEFINITIONS:
        definitions[definition] = {}
        for candidate in CANDIDATES:
            definitions[definition][candidate] = {
                "auc": auc_summary(rows, candidate=candidate, definition=definition),
                "fixed_alpha_0p10": operating_point(
                    rows, thresholds, candidate=candidate, definition=definition
                ),
            }
    sensitivity: dict[str, Any] = {}
    for candidate in CANDIDATES:
        aucs = [definitions[d][candidate]["auc"]["auc"] for d in LABEL_DEFINITIONS]
        tprs = [
            definitions[d][candidate]["fixed_alpha_0p10"]["micro_TPR"]
            for d in LABEL_DEFINITIONS
        ]
        fprs = [
            definitions[d][candidate]["fixed_alpha_0p10"]["micro_FPR"]
            for d in LABEL_DEFINITIONS
        ]
        final_auc = aucs[0]
        contrasts = {
            definition: (
                None
                if final_auc is None or definitions[definition][candidate]["auc"]["auc"] is None
                else float(definitions[definition][candidate]["auc"]["auc"] - final_auc)
            )
            for definition in LABEL_DEFINITIONS[1:]
        }
        sensitivity[candidate] = {
            "auc_across_definitions": _range(aucs),
            "micro_TPR_across_definitions": _range(tprs),
            "micro_FPR_across_definitions": _range(fprs),
            "auc_minus_final_adjudicated": contrasts,
        }
    return {
        "definitions_in_frozen_order": list(LABEL_DEFINITIONS),
        "by_definition": definitions,
        "label_definition_sensitivity": sensitivity,
        "confirmatory_p_values_computed": False,
        "candidate_combination_performed": False,
        "new_feature_direction_threshold_or_label_selection_performed": False,
    }


def _reviewer_maps(
    orders: Mapping[str, Sequence[Mapping[str, Any]]],
    review_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, dict[int, Mapping[str, Any]]]:
    result: dict[str, dict[int, Mapping[str, Any]]] = {}
    for reviewer in REVIEWERS:
        by_id = {str(row["review_id"]): row for row in review_rows[reviewer]}
        mapped: dict[int, Mapping[str, Any]] = {}
        for order_row in orders[reviewer]:
            sample_index = int(
                Path(str(order_row["native_image_relative_path"])).stem.split("_")[-1]
            )
            value = by_id.get(str(order_row["review_id"]))
            if value is None or sample_index in mapped:
                raise RuntimeError("reviewer-to-sample mapping failed")
            mapped[sample_index] = value
        if set(mapped) != set(range(TRAJECTORY_COUNT)):
            raise RuntimeError("reviewer mapping does not cover the exact cohort")
        result[reviewer] = mapped
    return result


def validate_failed_stage_a_receipt(
    root: Path,
    binding: Mapping[str, Any],
    binding_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replay and require the exact failed, no-score-access Stage-A boundary."""

    receipt, manifest = primary.validate_record_lock(
        root,
        artifact_kind="dit_bad_good_third_pool_stage_a_event_gate_v1",
        record_name="stage_a_gate_receipt.json",
    )
    if (
        receipt.get("input_binding_identity_sha256") != binding["identity_sha256"]
        or receipt.get("input_binding_manifest_identity_sha256")
        != binding_manifest["identity_sha256"]
        or receipt.get("scientific_contract_identity_sha256")
        != binding["scientific_contract_identity_sha256"]
        or receipt.get("implementation_source_sha256")
        != sha256_file(ROOT / "experiments/evaluate_dit_bad_good_third_pool_confirmation.py")
    ):
        raise RuntimeError("failed Stage-A receipt lineage differs from the bound evaluator")

    # Re-open only the already-authorized aggregate metadata path. This proves
    # the receipt is complete and replayable without touching consensus rows,
    # feature products, images, or score payloads.
    replay = primary.load_consensus_aggregate_only(
        binding["inputs"]["consensus"], binding["inputs"]["sampling_pool"]
    )
    expected_consensus_receipt = {
        key: replay[key]
        for key in (
            "manifest_identity_sha256",
            "manifest_file_sha256",
            "aggregate_identity_sha256",
            "aggregate_file_sha256",
            "row_member_declared_sha256",
        )
    }
    counts = primary.validate_aggregate_counts(receipt.get("aggregate_counts"))
    if counts != replay["counts"] or receipt.get("consensus_receipt") != expected_consensus_receipt:
        raise RuntimeError("failed Stage-A receipt does not replay the frozen aggregate consensus")
    overall = counts["overall"]
    blur_ok = overall["blur_or_soft_fusion_clear_bad"] >= primary.EVENT_MIN_BLUR
    total_ok = overall["clear_bad"] >= primary.EVENT_MIN_TOTAL_BAD
    expected_gate = {
        "minimum_blur_or_soft_fusion_clear_bad": primary.EVENT_MIN_BLUR,
        "observed_blur_or_soft_fusion_clear_bad": overall[
            "blur_or_soft_fusion_clear_bad"
        ],
        "blur_minimum_met": blur_ok,
        "minimum_total_clear_bad": primary.EVENT_MIN_TOTAL_BAD,
        "observed_total_clear_bad": overall["clear_bad"],
        "total_bad_minimum_met": total_ok,
        "both_minima_met": bool(blur_ok and total_ok),
        "stage_B_authorized": bool(blur_ok and total_ok),
    }
    expected_access_audit = {
        "consensus_manifest_opened": True,
        "consensus_completion_opened": True,
        "consensus_aggregate_counts_opened": True,
        "consensus_rows_opened_or_hashed": False,
        "sampling_pool_path_opened_statted_or_hashed": False,
        "primary_feature_product_path_opened_statted_or_hashed": False,
        "visual_feature_product_path_opened_statted_or_hashed": False,
        "score_csv_or_npz_opened": False,
        "image_opened": False,
        "screen_result_opened": False,
        "stage_B_invoked_in_same_process": False,
    }
    if (
        receipt.get("status") != "EVENT_GATE_FAILED_NO_SCORE_ACCESS"
        or receipt.get("event_gate") != expected_gate
        or expected_gate["both_minima_met"] is not False
        or expected_gate["stage_B_authorized"] is not False
        or overall["clear_bad"] != EXPECTED_FAILED_TOTAL_BAD
        or overall["blur_or_soft_fusion_clear_bad"] != EXPECTED_FAILED_BLUR_BAD
        or receipt.get("access_audit") != expected_access_audit
        or receipt.get("output_scope")
        != "aggregate label/phenotype counts and gate decision only"
    ):
        raise RuntimeError(
            "requires the exact replayed 6-total/4-blur failed Stage-A no-score receipt"
        )
    return receipt, manifest


def validate_legacy_primary_header(
    fields: Sequence[str], registered_features: set[str]
) -> None:
    """Allow only the two exact, sentinel-only legacy label placeholders."""

    ordered = tuple(fields)
    if len(ordered) != len(set(ordered)):
        raise RuntimeError("primary score CSV contains duplicate column names")
    if not set(LEGACY_PLACEHOLDER_COLUMNS).issubset(ordered):
        raise RuntimeError("primary score CSV lacks the two required legacy placeholders")
    metadata = {
        "sample_index",
        "run_index",
        "global_seed",
        "class_slot",
        "class_id",
        "trace_dir",
        "endpoint_png_path",
    }
    unexpected = set(ordered) - metadata - registered_features - set(
        LEGACY_PLACEHOLDER_COLUMNS
    )
    if unexpected:
        raise RuntimeError(f"uncatalogued primary score columns are forbidden: {sorted(unexpected)}")
    for field in ordered:
        if field in LEGACY_PLACEHOLDER_COLUMNS or field == FEATURES["C_c3_low_jump"]:
            continue
        lower = field.lower()
        registered_label_free_feature = (
            field in registered_features and "label_free" in lower
        )
        if not registered_label_free_feature and (
            "label" in lower or any(
            token in lower for token in primary.FORBIDDEN_FEATURE_HEADER_TOKENS
            )
        ):
            raise RuntimeError(f"non-placeholder label-like column is forbidden: {field}")


def validate_legacy_placeholder_values(raw: Mapping[str, Any]) -> str:
    if raw.get("label") != "unlabeled":
        raise RuntimeError("legacy label placeholder must be exactly 'unlabeled' on every row")
    consensus = raw.get("raw_consensus_label")
    if consensus not in RAW_CONSENSUS_EMPTY_SENTINELS:
        raise RuntimeError(
            "legacy raw_consensus_label placeholder must be exactly empty, nan, or NaN"
        )
    return str(consensus)


VISUAL_B_METADATA_COLUMNS = (
    "sample_index",
    "run_index",
    "global_seed",
    "class_slot",
    "class_id",
    "trace_dir",
    "endpoint_png_path",
)


def validate_visual_b_header(
    fields: Sequence[str], registered_features: set[str]
) -> None:
    """Validate visual columns by explicit frozen role, never name substrings."""

    ordered = tuple(fields)
    if len(ordered) != len(set(ordered)):
        raise RuntimeError("visual B score CSV contains duplicate column names")
    if FEATURES["B_blur_mean"] not in registered_features:
        raise RuntimeError("fixed B feature is absent from the frozen feature catalog")
    expected = set(VISUAL_B_METADATA_COLUMNS) | registered_features
    observed = set(ordered)
    missing = expected - observed
    extra = observed - expected
    if missing or extra:
        raise RuntimeError(
            f"visual B explicit-role header mismatch; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )


def validate_directory_matches_manifest(
    root: Path, manifest: Mapping[str, Any], description: str
) -> None:
    if manifest.get("files") != primary.artifact_records(root):
        raise RuntimeError(f"{description} directory schema differs from frozen manifest")


def _registered_catalog_features(catalog_path: Path) -> set[str]:
    registered: set[str] = set()
    with catalog_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        if "feature" not in fields or len(fields) != len(set(fields)):
            raise RuntimeError("feature catalog header changed")
        for raw in reader:
            if None in raw or set(raw) != set(fields):
                raise RuntimeError("feature catalog row has missing or extra cells")
            feature = raw.get("feature")
            if not feature or feature in registered:
                raise RuntimeError("feature catalog contains an empty or duplicate feature")
            registered.add(feature)
    return registered


def load_visual_b_by_explicit_roles(
    ref: Mapping[str, Any],
) -> tuple[
    dict[tuple[int, int, int], float],
    dict[str, Any],
    dict[int, dict[str, str]],
]:
    """Strict B loader: parse only frozen B plus identity/lineage metadata."""

    candidate = "B_blur_mean"
    feature = FEATURES[candidate]
    source_sha = primary.VISUAL_EXTRACTOR_SHA256
    root = primary.require_real_directory(
        Path(ref["path"]), f"{candidate} label-free product"
    )
    manifest_path = primary.require_regular(root / "manifest.json", "product manifest")
    completion_path = primary.require_regular(
        root / "completion.json", "product completion"
    )
    summary_path = primary.require_regular(root / "summary.json", "product summary")
    catalog_path = primary.require_regular(
        root / "feature_catalog.csv", "feature catalog"
    )
    score_path = primary.require_regular(
        root / "sample_features.csv", "sample feature CSV"
    )
    inventory_path = primary.require_regular(
        root / "source_inventory.json", "source inventory"
    )
    manifest = primary.load_json(manifest_path)
    completion = primary.load_json(completion_path)
    summary = primary.load_json(summary_path)
    inventory = primary.load_json(inventory_path)
    identity = primary.require_hex64(
        manifest.get("identity_sha256"), "visual product manifest identity"
    )
    by_name = primary._manifest_map(manifest, "visual label-free product")
    required = {
        "analysis_source.py",
        "feature_catalog.csv",
        "sample_features.csv",
        "source_inventory.json",
        "summary.json",
    }
    validate_directory_matches_manifest(root, manifest, "visual B product")
    if (
        not required.issubset(by_name)
        or identity != ref.get("manifest_identity_sha256")
        or primary.canonical_sha256(primary.without_identity(manifest)) != identity
        or manifest.get("status") != "complete"
        or manifest.get("experiment")
        != "dit_predxstart_preterminal_visual_tracks_label_free"
        or manifest.get("analysis_source_sha256") != source_sha
        or by_name["analysis_source.py"].get("sha256") != source_sha
        or completion.get("complete") is not True
        or completion.get("manifest_file_sha256") != primary.sha256_file(manifest_path)
        or completion.get("manifest_identity_sha256") != identity
        or summary.get("status") != "COMPLETE_LABEL_FREE_VISUAL_TRACK_EXTRACTION"
        or summary.get("labels_read_or_emitted") is not False
        or tuple(summary.get("ordered_classes", ())) != CLASSES
        or tuple(summary.get("ordered_seeds", ())) != SEEDS
        or summary.get("sample_count") != TRAJECTORY_COUNT
        or tuple(inventory.get("ordered_classes", ())) != CLASSES
        or tuple(inventory.get("ordered_seeds", ())) != SEEDS
        or inventory.get("analysis_source", {}).get("sha256") != source_sha
    ):
        raise RuntimeError("B_blur_mean label-free product contract failed")
    catalog = primary._catalog_feature_row(catalog_path, feature)
    if (
        catalog.get("latest_required_sampling_step") != "149"
        or catalog.get("latest_required_internal_timestep") != "100"
        or catalog.get("preterminal_actionable") != "True"
        or catalog.get("uses_realized_innovation") != "False"
    ):
        raise RuntimeError("B_blur_mean preterminal timing contract changed")
    registered_features = _registered_catalog_features(catalog_path)

    values: dict[tuple[int, int, int], float] = {}
    with score_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        validate_visual_b_header(fields, registered_features)
        for raw in reader:
            if None in raw or set(raw) != set(fields):
                raise RuntimeError("B_blur_mean score row has missing or extra cells")
            try:
                sample_index = int(raw["sample_index"])
                run_index = int(raw["run_index"])
                seed = int(raw["global_seed"])
                slot = int(raw["class_slot"])
                class_id = int(raw["class_id"])
                value = float(raw[feature])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError("invalid B_blur_mean score row") from exc
            key = (seed, slot, class_id)
            if (
                key in values
                or seed not in SEEDS
                or slot < 0
                or slot >= len(CLASSES)
                or class_id != CLASSES[slot]
                or run_index != seed - SEEDS[0]
                or sample_index != (seed - SEEDS[0]) * len(CLASSES) + slot
                or not math.isfinite(value)
            ):
                raise RuntimeError(f"B_blur_mean score row contract failed: {key}")
            values[key] = value
    if set(values) != primary._expected_keys() or len(values) != TRAJECTORY_COUNT:
        raise RuntimeError("B_blur_mean score cohort is incomplete")

    trace_runs = inventory.get("trace_runs")
    if not isinstance(trace_runs, list) or len(trace_runs) != len(SEEDS):
        raise RuntimeError("B_blur_mean trace lineage count changed")
    trace_lineage: dict[int, dict[str, str]] = {}
    for expected_seed, item in zip(SEEDS, trace_runs, strict=True):
        if (
            not isinstance(item, dict)
            or item.get("global_seed") != expected_seed
            or tuple(item.get("classes", ())) != CLASSES
        ):
            raise RuntimeError("B_blur_mean trace lineage order changed")
        lineage_fields = {
            "identity_sha256": item.get("identity_sha256"),
            "manifest_sha256": item.get("manifest_sha256"),
            "completion_sha256": item.get("completion_sha256"),
            "trace_sha256": item.get("trace_sha256"),
        }
        for name, digest in lineage_fields.items():
            primary.require_hex64(digest, f"B_blur_mean seed {expected_seed} {name}")
        trace_lineage[expected_seed] = lineage_fields
    lineage_hash = primary.canonical_sha256(
        [{"seed": seed, **trace_lineage[seed]} for seed in SEEDS]
    )
    return values, {
        "path": str(root),
        "manifest_identity_sha256": identity,
        "manifest_file_sha256": primary.sha256_file(manifest_path),
        "sample_features_file_sha256": primary.sha256_file(score_path),
        "feature_catalog_file_sha256": primary.sha256_file(catalog_path),
        "analysis_source_sha256": source_sha,
        "feature": feature,
        "trace_run_lineage_sha256": lineage_hash,
        "explicit_role_validation": {
            "metadata_columns_exact": list(VISUAL_B_METADATA_COLUMNS),
            "registered_feature_count": len(registered_features),
            "numeric_parsed_feature_columns_exact": [feature],
            "catalog_feature_names_may_contain_blur_or_severity": True,
            "uncatalogued_columns_accepted": False,
        },
    }, trace_lineage


def load_primary_c_with_legacy_placeholders(
    ref: Mapping[str, Any],
) -> tuple[
    dict[tuple[int, int, int], float],
    dict[str, Any],
    dict[int, dict[str, str]],
]:
    """Strict C loader with a sentinel-only exception for an upstream schema defect."""

    candidate = "C_c3_low_jump"
    feature = FEATURES[candidate]
    source_sha = primary.PRIMARY_EXTRACTOR_SHA256
    root = primary.require_real_directory(
        Path(ref["path"]), f"{candidate} label-free product"
    )
    manifest_path = primary.require_regular(root / "manifest.json", "product manifest")
    completion_path = primary.require_regular(
        root / "completion.json", "product completion"
    )
    summary_path = primary.require_regular(root / "summary.json", "product summary")
    catalog_path = primary.require_regular(
        root / "feature_catalog.csv", "feature catalog"
    )
    score_path = primary.require_regular(
        root / "sample_features.csv", "sample feature CSV"
    )
    inventory_path = primary.require_regular(
        root / "source_inventory.json", "source inventory"
    )
    manifest = primary.load_json(manifest_path)
    completion = primary.load_json(completion_path)
    summary = primary.load_json(summary_path)
    inventory = primary.load_json(inventory_path)
    identity = primary.require_hex64(
        manifest.get("identity_sha256"), "product manifest identity"
    )
    by_name = primary._manifest_map(manifest, "label-free product")
    required = {
        "analysis_source.py",
        "feature_catalog.csv",
        "sample_features.csv",
        "source_inventory.json",
        "summary.json",
    }
    if (
        not required.issubset(by_name)
        or identity != ref.get("manifest_identity_sha256")
        or primary.canonical_sha256(primary.without_identity(manifest)) != identity
        or manifest.get("status") != "complete"
        or manifest.get("experiment") != "dit_bad_good_custom_trace_metric_discovery"
        or manifest.get("analysis_source_sha256") != source_sha
        or by_name["analysis_source.py"].get("sha256") != source_sha
        or completion.get("complete") is not True
        or completion.get("manifest_file_sha256") != primary.sha256_file(manifest_path)
        or completion.get("manifest_identity_sha256") != identity
        or manifest.get("files") != primary.artifact_records(root)
        or summary.get("status") != "DISCOVERY_ONLY_NOT_AN_INTERVENTION_TRIGGER"
        or summary.get("labels_joined") is not False
        or tuple(summary.get("ordered_classes", ())) != CLASSES
        or tuple(summary.get("ordered_seeds", ())) != SEEDS
        or summary.get("sample_count") != TRAJECTORY_COUNT
        or tuple(inventory.get("ordered_classes", ())) != CLASSES
        or tuple(inventory.get("ordered_seeds", ())) != SEEDS
        or inventory.get("analysis_source", {}).get("sha256") != source_sha
    ):
        raise RuntimeError("C_c3_low_jump label-free product contract failed")
    catalog = primary._catalog_feature_row(catalog_path, feature)
    if (
        catalog.get("latest_required_sampling_step") != "149"
        or catalog.get("latest_required_internal_timestep") != "100"
        or catalog.get("preterminal_actionable") != "True"
        or catalog.get("uses_realized_innovation") != "False"
    ):
        raise RuntimeError("C_c3_low_jump preterminal timing contract changed")
    registered_features: set[str] = set()
    with catalog_path.open("r", encoding="utf-8", newline="") as handle:
        catalog_reader = csv.DictReader(handle)
        catalog_fields = tuple(catalog_reader.fieldnames or ())
        if "feature" not in catalog_fields or len(catalog_fields) != len(set(catalog_fields)):
            raise RuntimeError("feature catalog header changed")
        for raw_catalog in catalog_reader:
            if None in raw_catalog or set(raw_catalog) != set(catalog_fields):
                raise RuntimeError("feature catalog row has missing or extra cells")
            catalog_feature = raw_catalog.get("feature")
            if not catalog_feature or catalog_feature in registered_features:
                raise RuntimeError("feature catalog contains an empty or duplicate feature")
            registered_features.add(catalog_feature)
    if feature not in registered_features:
        raise RuntimeError("fixed C feature is absent from the feature catalog")

    values: dict[tuple[int, int, int], float] = {}
    sentinel_counts = {value: 0 for value in RAW_CONSENSUS_EMPTY_SENTINELS}
    with score_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        required_fields = {
            "sample_index",
            "run_index",
            "global_seed",
            "class_slot",
            "class_id",
            feature,
            *LEGACY_PLACEHOLDER_COLUMNS,
        }
        if not required_fields.issubset(fields):
            raise RuntimeError("C_c3_low_jump score CSV lacks required columns")
        validate_legacy_primary_header(fields, registered_features)
        for raw in reader:
            if None in raw or set(raw) != set(fields):
                raise RuntimeError("C_c3_low_jump score row has missing or extra cells")
            sentinel = validate_legacy_placeholder_values(raw)
            sentinel_counts[sentinel] += 1
            try:
                sample_index = int(raw["sample_index"])
                run_index = int(raw["run_index"])
                seed = int(raw["global_seed"])
                slot = int(raw["class_slot"])
                class_id = int(raw["class_id"])
                value = float(raw[feature])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError("invalid C_c3_low_jump score row") from exc
            key = (seed, slot, class_id)
            if (
                key in values
                or seed not in SEEDS
                or slot < 0
                or slot >= len(CLASSES)
                or class_id != CLASSES[slot]
                or run_index != seed - SEEDS[0]
                or sample_index != (seed - SEEDS[0]) * len(CLASSES) + slot
                or not math.isfinite(value)
            ):
                raise RuntimeError(f"C_c3_low_jump score row contract failed: {key}")
            values[key] = value
    if set(values) != primary._expected_keys() or len(values) != TRAJECTORY_COUNT:
        raise RuntimeError("C_c3_low_jump score cohort is incomplete")
    if sum(sentinel_counts.values()) != TRAJECTORY_COUNT:
        raise RuntimeError("legacy sentinel counts do not cover the exact cohort")

    trace_runs = inventory.get("trace_runs")
    if not isinstance(trace_runs, list) or len(trace_runs) != len(SEEDS):
        raise RuntimeError("C_c3_low_jump trace lineage count changed")
    trace_lineage: dict[int, dict[str, str]] = {}
    for expected_seed, item in zip(SEEDS, trace_runs, strict=True):
        if (
            not isinstance(item, dict)
            or item.get("global_seed") != expected_seed
            or tuple(item.get("classes", ())) != CLASSES
        ):
            raise RuntimeError("C_c3_low_jump trace lineage order changed")
        lineage_fields = {
            "identity_sha256": item.get("identity_sha256"),
            "manifest_sha256": item.get("manifest_sha256"),
            "completion_sha256": item.get("completion_sha256"),
            "trace_sha256": item.get("trace_sha256"),
        }
        for name, digest in lineage_fields.items():
            primary.require_hex64(
                digest, f"C_c3_low_jump seed {expected_seed} {name}"
            )
        trace_lineage[expected_seed] = lineage_fields
    lineage_hash = primary.canonical_sha256(
        [{"seed": seed, **trace_lineage[seed]} for seed in SEEDS]
    )
    return values, {
        "path": str(root),
        "manifest_identity_sha256": identity,
        "manifest_file_sha256": primary.sha256_file(manifest_path),
        "sample_features_file_sha256": primary.sha256_file(score_path),
        "feature_catalog_file_sha256": primary.sha256_file(catalog_path),
        "analysis_source_sha256": source_sha,
        "feature": feature,
        "trace_run_lineage_sha256": lineage_hash,
        "upstream_schema_defect": {
            "legacy_placeholder_columns_exact": list(LEGACY_PLACEHOLDER_COLUMNS),
            "label_unlabeled_count": TRAJECTORY_COUNT,
            "raw_consensus_label_sentinel_counts": sentinel_counts,
            "candidate_values_used_to_define_exception": False,
            "future_pipeline_fix": (
                "omit label and raw_consensus_label from label-free products"
            ),
        },
    }, trace_lineage


def run_audit(
    *,
    source_lock: Path,
    source_lock_manifest_identity: str,
    input_lock: Path,
    stage_a_receipt: Path,
    review_source_lock: Path,
    review_pack: Path,
    review_pack_manifest_identity: str,
    review_locks: Mapping[str, Path],
    review_lock_manifest_identities: Mapping[str, str],
    output: Path,
) -> Path:
    protocol, source_manifest = validate_source_lock(
        source_lock, source_lock_manifest_identity
    )
    binding, binding_manifest = primary.validate_input_binding(input_lock)
    stage_a, stage_a_manifest = validate_failed_stage_a_receipt(
        stage_a_receipt, binding, binding_manifest
    )
    # The preceding call requires and replays the formal failure before any
    # row-level label or score product is opened. The join below is exploratory.
    pool_trace_lineage, pool_lineage = primary.validate_sampling_pool(
        binding["inputs"]["sampling_pool"]
    )
    final_rows, consensus_lineage = primary.load_full_consensus(
        binding["inputs"]["consensus"],
        {**stage_a["consensus_receipt"], "counts": stage_a["aggregate_counts"]},
    )
    review_source_contract, review_source_manifest = reviews.validate_source_lock(
        review_source_lock
    )
    pack_record, pack_manifest, _catalog, orders, review_lineage, raw_rows = (
        reviews.load_review_chain(
            source_lock=review_source_lock,
            review_pack=review_pack,
            review_pack_manifest_identity=review_pack_manifest_identity,
            review_locks=review_locks,
            review_lock_manifest_identities=review_lock_manifest_identities,
        )
    )
    aggregate = load_json(Path(binding["inputs"]["consensus"]["path"]) / primary.CONSENSUS_AGGREGATE_NAME)
    audit_lineage = aggregate.get("blind_review_audit_lineage", {})
    if (
        audit_lineage.get("source_contract_identity_sha256")
        != review_source_contract["identity_sha256"]
        or audit_lineage.get("source_manifest_identity_sha256")
        != review_source_manifest["identity_sha256"]
        or audit_lineage.get("review_pack_record_identity_sha256")
        != pack_record["identity_sha256"]
        or audit_lineage.get("review_pack_manifest_identity_sha256")
        != pack_manifest["identity_sha256"]
        or audit_lineage.get("review_lock_manifest_identities")
        != {
            reviewer: review_lineage[reviewer]["manifest_identity_sha256"]
            for reviewer in REVIEWERS
        }
        or pack_record.get("sampling_pool", {}).get("manifest_identity_sha256")
        != pool_lineage["pool_identity_sha256"]
    ):
        raise RuntimeError("review chain differs from the chain used by final consensus")
    c_values, c_lineage, c_trace_lineage = load_primary_c_with_legacy_placeholders(
        binding["inputs"]["primary_label_free_product"]
    )
    b_values, b_lineage, b_trace_lineage = load_visual_b_by_explicit_roles(
        binding["inputs"]["visual_label_free_product"]
    )
    if c_trace_lineage != pool_trace_lineage or b_trace_lineage != pool_trace_lineage:
        raise RuntimeError("score products are not derived from the exact sampling pool")
    by_final = {int(row["sample_index"]): row for row in final_rows}
    by_raw = {int(row["sample_index"]): row for row in raw_rows}
    reviewer_maps = _reviewer_maps(orders, {
            reviewer: reviews.validate_review_lock(
            review_locks[reviewer],
            reviewer=reviewer,
            expected_manifest_identity=review_lock_manifest_identities[reviewer],
            source_contract=review_source_contract,
            source_manifest=review_source_manifest,
            pack_record=pack_record,
            pack_manifest=pack_manifest,
            expected_order=orders[reviewer],
        )[2]
        for reviewer in REVIEWERS
    })
    joined: list[dict[str, Any]] = []
    for sample_index in range(TRAJECTORY_COUNT):
        final = by_final[sample_index]
        raw = by_raw[sample_index]
        key = (final["global_seed"], final["class_slot"], final["class_id"])
        if key != (raw["global_seed"], raw["class_slot"], raw["class_id"]):
            raise RuntimeError("final/raw label identity mismatch")
        row: dict[str, Any] = {
            "sample_index": sample_index,
            "global_seed": final["global_seed"],
            "class_slot": final["class_slot"],
            "class_id": final["class_id"],
            "B_blur_mean_score": b_values[key],
            "C_c3_low_jump_score": c_values[key],
            "final_adjudicated_bad": final["final_severity"] == "clear_bad",
            "final_adjudicated_good": final["final_severity"] == "clean_good",
            "final_adjudicated_blur": bool(final["blur_component_consensus"]),
            "raw_majority_bad": raw["raw_severity"] == "clear_bad",
            "raw_majority_good": raw["raw_severity"] == "clean_good",
            "raw_majority_blur": bool(raw["blur_component_consensus"]),
        }
        for reviewer in REVIEWERS:
            review = reviewer_maps[reviewer][sample_index]
            row[f"{reviewer}_bad"] = int(review["severity"]) >= 2
            row[f"{reviewer}_good"] = int(review["severity"]) == 0
            row[f"{reviewer}_blur"] = any(
                bool(review[flag]) for flag in reviews.BLUR_COMPONENTS
            )
        joined.append(row)
    foundations = primary.validate_foundation_locks()
    thresholds = foundations["thresholds"].get("thresholds", {})
    statistics = evaluate_rows(joined, thresholds)
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE_POST_FAILED_GATE_EXPLORATORY_FOR_NEXT_POOL_NO_AUTHORIZATION",
        "experiment": EXPERIMENT,
        "designation": protocol["designation"],
        "source_protocol_identity_sha256": protocol["identity_sha256"],
        "source_manifest_identity_sha256": source_manifest["identity_sha256"],
        "primary_input_binding_identity_sha256": binding["identity_sha256"],
        "primary_input_binding_manifest_identity_sha256": binding_manifest["identity_sha256"],
        "stage_a_receipt_identity_sha256": stage_a["identity_sha256"],
        "stage_a_manifest_identity_sha256": stage_a_manifest["identity_sha256"],
        "event_gate": stage_a["event_gate"],
        "formal_evaluation_status": {
            "stage_A_status": "EVENT_GATE_FAILED_NO_SCORE_ACCESS",
            "stage_B_reached": False,
            "formal_result_affected_by_upstream_schema_defect": False,
            "formal_evaluator_V5_hypothetical_stage_B_behavior": (
                "would reject raw_consensus_label as a forbidden label-like header"
            ),
        },
        "input_lineage": {
            "sampling_pool": pool_lineage,
            "final_consensus": consensus_lineage,
            "review_pack_record_identity_sha256": pack_record["identity_sha256"],
            "review_pack_manifest_identity_sha256": pack_manifest["identity_sha256"],
            "review_locks": review_lineage,
            "primary_C_product": c_lineage,
            "visual_B_product": b_lineage,
            "threshold_lock_identity_sha256": foundations["thresholds"]["identity_sha256"],
        },
        "statistics": statistics,
        "interpretation_constraints": {
            "post_failed_gate_exploratory_only": True,
            "purpose_limited_to_next_pool_design": True,
            "confirmatory_p_values_computed": False,
            "may_call_any_candidate_passed": False,
            "formal_stage_A_remains_failed": True,
            "may_override_failed_stage_A_or_primary_stage_B": False,
            "intervention_experiment_authorized": False,
        },
        "upstream_schema_defect": {
            "legacy_exception_limited_to_columns": list(LEGACY_PLACEHOLDER_COLUMNS),
            "label_required_value_every_row": "unlabeled",
            "raw_consensus_label_allowed_values": list(
                RAW_CONSENSUS_EMPTY_SENTINELS
            ),
            "all_other_label_like_columns_rejected": True,
            "source_decision_used_only_schema_header_and_sentinel_semantics": True,
            "candidate_label_threshold_or_direction_changed": False,
            "future_pipeline_fix": (
                "remove label and raw_consensus_label from every label-free product"
            ),
            "visual_generic_guard_defect": (
                "substring matching misclassifies legitimate frozen catalog features "
                "such as decoded_local_blur_severity__maximum"
            ),
            "visual_B_validation_rule": (
                "fixed metadata roles plus exact frozen catalog membership; only the "
                "fixed B feature is numeric parsed"
            ),
            "future_formal_schema_fix": (
                "use explicit column roles and a frozen feature catalog instead of "
                "name-substring rejection"
            ),
        },
        "access_audit": {
            "source_lock_validated_before_score_open": True,
            "exact_failed_stage_A_receipt_replayed_before_score_or_row_level_review_open": True,
            "failed_receipt_attested_primary_and_visual_feature_paths_unopened": True,
            "legacy_placeholder_schema_and_all_sentinel_values_validated_before_join": True,
            "only_fixed_C_column_numeric_parsed_from_primary_product": True,
            "visual_B_header_validated_by_explicit_roles_and_frozen_catalog": True,
            "only_fixed_B_column_numeric_parsed_from_visual_product": True,
            "only_frozen_B_C_scores_opened": True,
            "new_feature_direction_threshold_or_label_selected": False,
            "individual_rows_scores_labels_or_ranks_emitted": False,
        },
        "output_scope": (
            "aggregate-only post-failure design evidence; no row-level payload, "
            "no formal-gate rescue, and no intervention authorization"
        ),
        "implementation_source_sha256": sha256_file(Path(__file__).resolve()),
    }
    result["identity_sha256"] = canonical_sha256(result)
    return primary.publish_record_lock(
        output,
        artifact_kind=RESULT_ARTIFACT_KIND,
        record_name=RESULT_NAME,
        record=result,
        source_copies={"auditor_source.py": Path(__file__).resolve()},
    )


def _synthetic_rows() -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    for seed in SEEDS:
        for slot, class_id in enumerate(CLASSES):
            sample_index = (seed - SEEDS[0]) * len(CLASSES) + slot
            local = seed - SEEDS[0]
            row: dict[str, Any] = {
                "sample_index": sample_index,
                "global_seed": seed,
                "class_slot": slot,
                "class_id": class_id,
                "B_blur_mean_score": float(local % 23),
                "C_c3_low_jump_score": float(-(local % 19)),
            }
            for index, definition in enumerate(LABEL_DEFINITIONS):
                row[f"{definition}_bad"] = local < 10 + index and slot == 0
                row[f"{definition}_good"] = local >= 20
                row[f"{definition}_blur"] = local < 8 + index and slot == 0
            rows_out.append(row)
    return rows_out


def _synthetic_failed_consensus_rows() -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    for seed in SEEDS:
        for slot, class_id in enumerate(CLASSES):
            sample_index = (seed - SEEDS[0]) * len(CLASSES) + slot
            is_bad = sample_index < EXPECTED_FAILED_TOTAL_BAD
            rows_out.append(
                {
                    "sample_index": sample_index,
                    "global_seed": seed,
                    "class_slot": slot,
                    "class_id": class_id,
                    "final_severity": "clear_bad" if is_bad else "clean_good",
                    "blur_component_consensus": (
                        sample_index < EXPECTED_FAILED_BLUR_BAD
                    ),
                    "discrete_structure_component_consensus": False,
                }
            )
    return rows_out


def self_test() -> None:
    fake_thresholds = {
        candidate: {
            "classes": {
                str(class_id): {ALPHA: {"threshold": 10.0 if candidate == "B_blur_mean" else -10.0}}
                for class_id in CLASSES
            }
        }
        for candidate in CANDIDATES
    }
    result = evaluate_rows(_synthetic_rows(), fake_thresholds)
    assert tuple(result["definitions_in_frozen_order"]) == LABEL_DEFINITIONS
    assert result["confirmatory_p_values_computed"] is False
    assert result["candidate_combination_performed"] is False
    for definition in LABEL_DEFINITIONS:
        assert set(result["by_definition"][definition]) == set(CANDIDATES)
    # No row-level key may appear anywhere in the serializable aggregate result.
    serialized = json.dumps(result, sort_keys=True)
    for forbidden in ('"sample_index"', '"global_seed"', '"score"', '"rank"'):
        assert forbidden not in serialized

    metadata = (
        "sample_index",
        "run_index",
        "global_seed",
        "class_slot",
        "class_id",
        "trace_dir",
        "endpoint_png_path",
    )
    registered = {
        FEATURES["C_c3_low_jump"],
        "label_free_reference_q1_pred_roughness_z_mean",
    }
    good_fields = (
        *metadata,
        *LEGACY_PLACEHOLDER_COLUMNS,
        *sorted(registered),
    )
    validate_legacy_primary_header(good_fields, registered)
    assert validate_legacy_placeholder_values(
        {"label": "unlabeled", "raw_consensus_label": ""}
    ) == ""
    assert validate_legacy_placeholder_values(
        {"label": "unlabeled", "raw_consensus_label": "NaN"}
    ) == "NaN"
    poison_rows = (
        {"label": "clear_bad", "raw_consensus_label": ""},
        {"label": "unlabeled", "raw_consensus_label": "clear_bad"},
        {"label": "unlabeled", "raw_consensus_label": " "},
    )
    for poison in poison_rows:
        try:
            validate_legacy_placeholder_values(poison)
        except RuntimeError:
            pass
        else:
            raise AssertionError("poisoned legacy placeholder value was accepted")
    try:
        validate_legacy_primary_header(
            (*good_fields, "quality_label"), {*registered, "quality_label"}
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("an additional label-like column was accepted")
    try:
        validate_legacy_primary_header(
            tuple(field for field in good_fields if field != "raw_consensus_label"),
            registered,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("a missing required placeholder column was accepted")

    registered_visual = {
        FEATURES["B_blur_mean"],
        "decoded_local_blur_severity__maximum",
        "decoded_edge_tangle__mean",
    }
    valid_visual_header = (*VISUAL_B_METADATA_COLUMNS, *sorted(registered_visual))
    validate_visual_b_header(valid_visual_header, registered_visual)
    for poison_column in ("blur_label", "quality_label"):
        try:
            validate_visual_b_header(
                (*valid_visual_header, poison_column), registered_visual
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"uncatalogued {poison_column} was accepted")
    try:
        validate_visual_b_header(
            tuple(
                field
                for field in valid_visual_header
                if field != FEATURES["B_blur_mean"]
            ),
            registered_visual - {FEATURES["B_blur_mean"]},
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("a renamed or missing frozen B feature was accepted")
    try:
        validate_visual_b_header(
            (*valid_visual_header, FEATURES["B_blur_mean"]), registered_visual
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("a duplicate frozen B feature was accepted")

    with tempfile.TemporaryDirectory(prefix="bc-label-sensitivity-selftest-") as tmp:
        temporary = Path(tmp)
        schema_root = temporary / "visual_schema"
        schema_root.mkdir()
        (schema_root / "expected.txt").write_text("frozen\n", encoding="utf-8")
        schema_manifest = {"files": primary.artifact_records(schema_root)}
        validate_directory_matches_manifest(
            schema_root, schema_manifest, "synthetic visual product"
        )
        (schema_root / "injected.txt").write_text("poison\n", encoding="utf-8")
        try:
            validate_directory_matches_manifest(
                schema_root, schema_manifest, "synthetic visual product"
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("a tampered visual product directory was accepted")

        consensus = temporary / "consensus"
        pool_identity = "c" * 64
        consensus_identity = primary._write_synthetic_consensus_lock(
            consensus,
            _synthetic_failed_consensus_rows(),
            sampling_pool_identity_sha256=pool_identity,
        )
        poison_pool = temporary / "MUST_NOT_OPEN_pool"
        poison_primary = temporary / "MUST_NOT_OPEN_primary_scores"
        poison_visual = temporary / "MUST_NOT_OPEN_visual_scores"
        binding_root = primary.bind_inputs(
            source_lock=primary.DEFAULT_SOURCE_LOCK,
            sampling_pool_path=poison_pool,
            sampling_pool_manifest_identity=pool_identity,
            consensus_path=consensus,
            consensus_manifest_identity=consensus_identity,
            primary_path=poison_primary,
            primary_manifest_identity="a" * 64,
            visual_path=poison_visual,
            visual_manifest_identity="b" * 64,
            output=temporary / "binding",
        )
        receipt_root = primary.run_stage_a(
            input_lock=binding_root, output=temporary / "failed_stage_a"
        )
        binding, binding_manifest = primary.validate_input_binding(binding_root)
        receipt, _ = validate_failed_stage_a_receipt(
            receipt_root, binding, binding_manifest
        )
        assert receipt["status"] == "EVENT_GATE_FAILED_NO_SCORE_ACCESS"
        assert receipt["aggregate_counts"]["overall"]["clear_bad"] == 6
        assert receipt["aggregate_counts"]["overall"][
            "blur_or_soft_fusion_clear_bad"
        ] == 4
        assert not poison_pool.exists()
        assert not poison_primary.exists()
        assert not poison_visual.exists()

        lock = freeze_source_lock(temporary / "source_lock")
        validate_source_lock(lock)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-source-lock", action="store_true")
    parser.add_argument("--validate-source-lock", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--source-lock", type=Path, default=DEFAULT_SOURCE_LOCK)
    parser.add_argument("--source-lock-manifest-identity")
    parser.add_argument("--input-lock", type=Path)
    parser.add_argument("--stage-a-receipt", type=Path)
    parser.add_argument("--review-source-lock", type=Path)
    parser.add_argument("--review-pack", type=Path)
    parser.add_argument("--review-pack-manifest-identity")
    for reviewer in REVIEWERS:
        parser.add_argument(f"--{reviewer.replace('_', '-')}-lock", type=Path)
        parser.add_argument(f"--{reviewer.replace('_', '-')}-manifest-identity")
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    modes = [args.freeze_source_lock, args.validate_source_lock, args.self_test, args.run]
    if sum(bool(mode) for mode in modes) != 1:
        raise SystemExit("choose exactly one mode")
    if args.freeze_source_lock:
        path = freeze_source_lock(args.source_lock)
        protocol, manifest = validate_source_lock(path)
        print(json.dumps({"path": str(path), "protocol_identity_sha256": protocol["identity_sha256"], "manifest_identity_sha256": manifest["identity_sha256"]}, indent=2))
        return
    if args.validate_source_lock:
        protocol, manifest = validate_source_lock(
            args.source_lock, args.source_lock_manifest_identity
        )
        print(json.dumps({"valid": True, "protocol_identity_sha256": protocol["identity_sha256"], "manifest_identity_sha256": manifest["identity_sha256"]}, indent=2))
        return
    if args.self_test:
        self_test()
        print("self-test passed")
        return
    required = {
        "source_lock_manifest_identity": args.source_lock_manifest_identity,
        "input_lock": args.input_lock,
        "stage_a_receipt": args.stage_a_receipt,
        "review_source_lock": args.review_source_lock,
        "review_pack": args.review_pack,
        "review_pack_manifest_identity": args.review_pack_manifest_identity,
        "output": args.output,
    }
    for reviewer in REVIEWERS:
        required[f"{reviewer}_lock"] = getattr(args, f"{reviewer}_lock")
        required[f"{reviewer}_manifest_identity"] = getattr(
            args, f"{reviewer}_manifest_identity"
        )
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise SystemExit(f"missing --run arguments: {', '.join(missing)}")
    result = run_audit(
        source_lock=args.source_lock,
        source_lock_manifest_identity=args.source_lock_manifest_identity,
        input_lock=args.input_lock,
        stage_a_receipt=args.stage_a_receipt,
        review_source_lock=args.review_source_lock,
        review_pack=args.review_pack,
        review_pack_manifest_identity=args.review_pack_manifest_identity,
        review_locks={reviewer: getattr(args, f"{reviewer}_lock") for reviewer in REVIEWERS},
        review_lock_manifest_identities={reviewer: getattr(args, f"{reviewer}_manifest_identity") for reviewer in REVIEWERS},
        output=args.output,
    )
    print(result)


if __name__ == "__main__":
    main()
