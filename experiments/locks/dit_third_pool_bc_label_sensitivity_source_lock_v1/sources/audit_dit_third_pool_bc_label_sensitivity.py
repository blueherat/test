#!/usr/bin/env python3
"""Post-gate exploratory label-definition sensitivity audit for third-pool B/C.

This audit is deliberately non-confirmatory.  Its only question is whether the
already-frozen B and C summaries change materially when the endpoint labels are
defined by final adjudication, raw reviewer majority, or one reviewer at a time.
It cannot authorize an intervention or override either third-pool gate.
"""

from __future__ import annotations

import argparse
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
    ROOT / "experiments/locks/dit_third_pool_bc_label_sensitivity_source_lock_v1"
)

EXPERIMENT = "dit_third_pool_bc_label_sensitivity_v1"
SOURCE_ARTIFACT_KIND = "dit_third_pool_bc_label_sensitivity_source_lock_v1"
RESULT_ARTIFACT_KIND = "dit_third_pool_bc_label_sensitivity_result_v1"
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
        "designation": "POST_GATE_EXPLORATORY_ONLY_NEVER_AUTHORIZES_INTERVENTION",
        "question": (
            "How sensitive are the two already-frozen B/C associations to five "
            "pre-enumerated blind-label definitions?"
        ),
        "temporal_firewall": {
            "protocol_and_all_executable_sources_frozen_before_any_third_pool_primary_or_visual_score_value_is_opened": True,
            "allowed_before_freeze": [
                "locked label/reviewer/consensus schemas",
                "aggregate event counts",
                "artifact paths and identities without score payload access",
            ],
            "production_join_requires_valid_replayed_stage_A_pass_receipt": True,
            "stage_A_failure_cannot_be_overridden": True,
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
            "cannot_rescue_or_override_stage_A_or_primary_stage_B": True,
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
        != "POST_GATE_EXPLORATORY_ONLY_NEVER_AUTHORIZES_INTERVENTION"
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
    stage_a, stage_a_manifest = primary.validate_stage_a_receipt(
        stage_a_receipt, binding, binding_manifest
    )
    # The preceding call replays the count gate and rejects failure before any
    # row-level label or score product is opened.
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
    c_values, c_lineage, c_trace_lineage = primary.load_feature_product(
        binding["inputs"]["primary_label_free_product"], candidate="C_c3_low_jump"
    )
    b_values, b_lineage, b_trace_lineage = primary.load_feature_product(
        binding["inputs"]["visual_label_free_product"], candidate="B_blur_mean"
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
        "status": "COMPLETE_POST_GATE_EXPLORATORY_LABEL_SENSITIVITY_NO_AUTHORIZATION",
        "experiment": EXPERIMENT,
        "designation": protocol["designation"],
        "source_protocol_identity_sha256": protocol["identity_sha256"],
        "source_manifest_identity_sha256": source_manifest["identity_sha256"],
        "primary_input_binding_identity_sha256": binding["identity_sha256"],
        "primary_input_binding_manifest_identity_sha256": binding_manifest["identity_sha256"],
        "stage_a_receipt_identity_sha256": stage_a["identity_sha256"],
        "stage_a_manifest_identity_sha256": stage_a_manifest["identity_sha256"],
        "event_gate": stage_a["event_gate"],
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
            "post_gate_exploratory_only": True,
            "confirmatory_p_values_computed": False,
            "may_call_any_candidate_passed": False,
            "may_override_stage_A_or_primary_stage_B": False,
            "intervention_experiment_authorized": False,
        },
        "access_audit": {
            "source_lock_validated_before_score_open": True,
            "stage_A_pass_replayed_before_score_or_row_level_review_open": True,
            "only_frozen_B_C_scores_opened": True,
            "new_feature_direction_threshold_or_label_selected": False,
            "individual_rows_scores_labels_or_ranks_emitted": False,
        },
        "output_scope": "aggregate-only; no row-level payload and no intervention authorization",
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
    with tempfile.TemporaryDirectory(prefix="bc-label-sensitivity-selftest-") as tmp:
        lock = freeze_source_lock(Path(tmp) / "source_lock")
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
