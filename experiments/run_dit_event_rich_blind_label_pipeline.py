#!/usr/bin/env python3
"""Immutable endpoint-only review, dual-adjudication, and consensus pipeline.

Stages are explicit and one-way:

1. lock an actual endpoint cohort and its exact phase axis;
2. build three isolated reviewer deliveries after a five-role panel passes;
3. lock each completed reviewer form independently;
4. build two isolated adjudicator deliveries from the union of any positive
   reviewer vote plus an equal frozen random sample of zero-positive decoys;
5. lock both adjudicator forms and publish final consensus.

Review/adjudication deliveries never expose candidate values, class enrichment ranks,
metrics, trajectory features, embeddings, thresholds, alerts, reviewer
identity, vote counts, or trigger/decoy membership.  This script computes no
model score and cannot run any stage without its complete upstream artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image

import dit_event_rich_review_contract as contract


RUNNER = "run_dit_event_rich_blind_label_pipeline"
ATTESTATION = "I independently reviewed endpoint pixels only under the frozen rubric"
COHORT_INDEX_FIELDS = (
    "class_id",
    "global_seed",
    "class_name",
    "image_path",
    "image_sha256",
    "image_pixel_sha256",
    "width",
    "height",
    "mode",
    "source_pair_identity_sha256",
    "source_manifest_sha256",
)
COHORT_LOCK_FIELDS = (
    "class_id",
    "global_seed",
    "class_name",
    "source_image_path",
    "image_sha256",
    "image_pixel_sha256",
    "source_pair_identity_sha256",
    "source_manifest_sha256",
)
REVIEW_DELIVERY_FIELDS = (
    "blind_id",
    "class_id",
    "class_name",
    "image_relative_path",
    "image_sha256",
    "image_pixel_sha256",
    "mode",
    "width",
    "height",
)
REVIEW_LOCK_FIELDS = (
    "class_id",
    "global_seed",
    "image_sha256",
    "image_pixel_sha256",
    "severity",
    "components",
    "localization_reason",
)
ADJUDICATION_DELIVERY_FIELDS = REVIEW_DELIVERY_FIELDS
ADJUDICATION_MAPPING_FIELDS = (
    "adjudication_id",
    "class_id",
    "global_seed",
    "image_sha256",
    "image_pixel_sha256",
    "selection_kind",
    "positive_vote_count",
)
ADJUDICATION_LOCK_FIELDS = (
    "class_id",
    "global_seed",
    "image_sha256",
    "image_pixel_sha256",
    "decision",
    "components",
    "localization_reason",
)
CONSENSUS_FIELDS = (
    "phase",
    "class_id",
    "global_seed",
    "image_sha256",
    "image_pixel_sha256",
    "reviewer_positive_count",
    "reviewer_zero_count",
    "raw_label",
    "audit_selection",
    "adjudicator_1_decision",
    "adjudicator_2_decision",
    "final_label",
    "final_clear_bad",
    "blur_component_consensus",
    "structure_component_consensus",
    "other_component_consensus",
    "blur_or_soft_fusion_positive",
    "structural_non_blur",
    "phenotype_disputed",
    "consensus_rule",
)
EVALUATION_LABEL_FIELDS = (
    "phase",
    "global_seed",
    "class_id",
    "final_severity",
    "blur_component",
)


def source_binding() -> dict[str, Any]:
    manifest, _ = contract.validate_source_lock(invoked_source=Path(__file__).resolve())
    return {
        "review_source_lock": str(contract.REVIEW_SOURCE_LOCK),
        "review_source_lock_identity_sha256": manifest["identity_sha256"],
        "runner_source_sha256": contract.sha256_file(Path(__file__).resolve()),
    }


def roster() -> tuple[dict[str, Any], ...]:
    protocol, _ = contract.validate_event_protocol_lock()
    rows = protocol["endpoint_screen"]["class_roster"]
    if len(rows) != 84:
        raise RuntimeError("scientific-v4 class roster changed")
    return tuple(rows)


def load_frozen_selector() -> Any:
    path = contract.require_regular(
        contract.EVENT_PROTOCOL_LOCK / "sources/select_dit_event_rich_blur_classes_v4.py",
        "frozen scientific-v4 blur selector",
    )
    spec = importlib.util.spec_from_file_location("_event_review_frozen_selector", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen scientific-v4 blur selector")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_anchor_trace_plan(
    value: Mapping[str, Any], protocol: Mapping[str, Any]
) -> tuple[int, ...]:
    exact_fields = {
        "schema_version",
        "artifact_kind",
        "status",
        "protocol_identity_sha256",
        "selection_identity_sha256",
        "anchor_consensus_file_sha256",
        "selected_classes",
        "aggregate_counts",
        "decision",
        "descriptive_only",
        "calibration_seeds",
        "confirmation_seeds",
        "calibration_trace_rows",
        "confirmation_trace_rows",
        "B_and_E_share_exact_selected_class_set",
        "external_visual_labels_used_only_for_cohort_enrichment_and_go",
        "method_score_threshold_intervention_or_external_representation_input_used",
        "identity_sha256",
    }
    decision_fields = {
        "anchor_rows",
        "blur_clear_bad",
        "event_bearing_classes",
        "clean_good",
        "gates",
        "go",
        "failed_gates",
    }
    if set(value) != exact_fields:
        raise RuntimeError("authoritative anchor trace-plan schema changed")
    selected = value.get("selected_classes")
    roster_ids = tuple(row["class_id"] for row in protocol["endpoint_screen"]["class_roster"])
    counts = value.get("aggregate_counts")
    decision = value.get("decision")
    rule = protocol["anchor_go_rule"]
    if (
        value.get("schema_version") != 1
        or value.get("artifact_kind") != "EVENT_RICH_BLUR_ANCHOR_PLAN_LOCK_V1"
        or value.get("status") != "BLUR_ANCHOR_GO_DECISION_LOCKED_BEFORE_INTERNAL_TRACES"
        or value.get("protocol_identity_sha256") != contract.EVENT_PROTOCOL_IDENTITY
        or contract.canonical_sha256(contract.without_identity(value))
        != value.get("identity_sha256")
        or not isinstance(value.get("selection_identity_sha256"), str)
        or len(value["selection_identity_sha256"]) != 64
        or not isinstance(value.get("anchor_consensus_file_sha256"), str)
        or len(value["anchor_consensus_file_sha256"]) != 64
        or not isinstance(selected, list)
        or len(selected) != 6
        or len(set(selected)) != 6
        or any(type(class_id) is not int or class_id not in roster_ids for class_id in selected)
        or not isinstance(counts, list)
        or len(counts) != 6
        or not isinstance(decision, dict)
        or set(decision) != decision_fields
        or value.get("calibration_seeds") != protocol["dynamic_axis"]["calibration_seeds"]
        or value.get("confirmation_seeds") != protocol["dynamic_axis"]["confirmation_seeds"]
        or value.get("calibration_trace_rows") != 6 * len(protocol["dynamic_axis"]["calibration_seeds"])
        or value.get("confirmation_trace_rows") != 6 * len(protocol["dynamic_axis"]["confirmation_seeds"])
        or value.get("B_and_E_share_exact_selected_class_set") is not True
        or value.get("external_visual_labels_used_only_for_cohort_enrichment_and_go") is not True
        or value.get("method_score_threshold_intervention_or_external_representation_input_used") is not False
    ):
        raise RuntimeError("authoritative anchor trace-plan identity/contract failed")
    for class_id, row in zip(selected, counts):
        if (
            not isinstance(row, dict)
            or set(row) != {"class_id", "n", "blur_clear_bad", "clean_good"}
            or row.get("class_id") != class_id
            or row.get("n") != len(protocol["endpoint_screen"]["anchor_seeds"])
            or any(type(row.get(key)) is not int for key in ("class_id", "n", "blur_clear_bad", "clean_good"))
            or not 0 <= row["blur_clear_bad"] <= row["n"]
            or not 0 <= row["clean_good"] <= row["n"] - row["blur_clear_bad"]
        ):
            raise RuntimeError("anchor aggregate counts changed")
    events = sum(row["blur_clear_bad"] for row in counts)
    clean = sum(row["clean_good"] for row in counts)
    bearing = sum(row["blur_clear_bad"] > 0 for row in counts)
    expected_gates = {
        "blur_events": events >= int(rule["minimum_blur_clear_bad"]),
        "event_bearing_classes": bearing >= int(rule["minimum_event_bearing_classes"]),
        "clean_good": clean >= int(rule["minimum_clean_good"]),
    }
    if (
        decision.get("anchor_rows") != 6 * len(protocol["endpoint_screen"]["anchor_seeds"])
        or decision.get("blur_clear_bad") != events
        or decision.get("event_bearing_classes") != bearing
        or decision.get("clean_good") != clean
        or decision.get("gates") != expected_gates
        or decision.get("go") is not True
        or decision.get("failed_gates") != []
    ):
        raise RuntimeError("confirmation requires the exact single blur-set anchor GO")
    return tuple(selected)


def validate_phase_plan(path: Path, phase: str) -> tuple[dict[str, Any], tuple[int, ...], tuple[int, ...]]:
    value = contract.load_json(path)
    protocol, _ = contract.validate_event_protocol_lock()
    roster_ids = tuple(row["class_id"] for row in roster())
    if phase == "discovery":
        exact_fields = {
            "schema_version",
            "status",
            "phase",
            "event_protocol_identity_sha256",
            "class_ids_ordered",
            "global_seeds_ordered",
            "upstream_plan_identity_sha256",
            "labels_locked_before_plan",
            "candidate_scores_features_trajectories_embeddings_or_ranks_used",
            "identity_sha256",
        }
        if set(value) != exact_fields:
            raise RuntimeError("discovery phase-plan schema changed")
        classes = tuple(value.get("class_ids_ordered", ()))
        seeds = tuple(value.get("global_seeds_ordered", ()))
        if (
            value.get("schema_version") != 1
            or value.get("status") != "FROZEN_ENDPOINT_PHASE_PLAN"
            or value.get("phase") != phase
            or value.get("event_protocol_identity_sha256")
            != contract.EVENT_PROTOCOL_IDENTITY
            or value.get("identity_sha256")
            != contract.canonical_sha256(contract.without_identity(value))
            or seeds != contract.PHASE_SEEDS[phase]
            or classes != roster_ids
            or value.get("labels_locked_before_plan") is not False
            or value.get("upstream_plan_identity_sha256") is not None
            or value.get(
                "candidate_scores_features_trajectories_embeddings_or_ranks_used"
            )
            is not False
        ):
            raise RuntimeError("discovery phase must be the exact frozen 84-class roster and precede labels")
        return value, classes, seeds
    if phase == "anchor":
        selector = load_frozen_selector()
        selector.validate_selection(protocol, value)
        classes = tuple(value["selected_classes"])
        if len(classes) != 6:
            raise RuntimeError("anchor must contain the one exact six-class blur set")
        return value, classes, contract.PHASE_SEEDS[phase]
    if phase == "confirmation":
        classes = validate_anchor_trace_plan(value, protocol)
        return value, classes, contract.PHASE_SEEDS[phase]
    raise RuntimeError(f"unknown phase: {phase}")


def inspect_image(path: Path) -> dict[str, Any]:
    path = contract.require_regular(path, "endpoint cohort image")
    with Image.open(path) as image:
        image.load()
        mode = image.mode
        width, height = image.size
        pixel_sha256 = hashlib.sha256(image.tobytes()).hexdigest()
    if mode != "RGB" or (width, height) != (256, 256):
        raise RuntimeError(f"endpoint must be RGB 256x256: {path}")
    return {
        "path": str(path),
        "image_sha256": contract.sha256_file(path),
        "image_pixel_sha256": pixel_sha256,
        "width": width,
        "height": height,
        "mode": mode,
    }


FORBIDDEN_REVIEW_TREE_NAME_FRAGMENTS = (
    "trace",
    "method",
    "candidate",
    "score",
    "metric",
    "feature",
    "embedding",
    "inception",
    "dino",
    "fid",
    "clip",
    "calibration",
    "observer",
    "eprocess",
    "b_persistence",
    "e_blur_gated",
)
FORBIDDEN_REVIEW_TREE_SUFFIXES = (
    ".npz",
    ".npy",
    ".pt",
    ".pth",
    ".safetensors",
    ".parquet",
    ".arrow",
    ".csv",
)


def _reject_material_forbidden_json_fields(value: Any, location: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower()
            forbidden = any(
                fragment in normalized
                for fragment in FORBIDDEN_REVIEW_TREE_NAME_FRAGMENTS
            )
            # Negative/empty audit fields such as trace_saved=false and
            # quality_score=null are allowed; any material payload is not.
            if forbidden and child not in (False, None, "", [], {}):
                raise RuntimeError(
                    f"review endpoint tree contains material forbidden field {location}:{key}"
                )
            _reject_material_forbidden_json_fields(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_material_forbidden_json_fields(child, f"{location}[{index}]")


def validate_external_review_endpoint_tree(root: Path) -> None:
    """Reject any internal-method payload from the external review tree."""

    root = contract.require_directory(root, "external review endpoint artifact")
    endpoint_count = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"review endpoint tree contains a symlink: {path}")
        relative = path.relative_to(root).as_posix()
        lowered = relative.lower()
        if any(fragment in lowered for fragment in FORBIDDEN_REVIEW_TREE_NAME_FRAGMENTS):
            raise RuntimeError(f"review endpoint tree contains forbidden member name: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise RuntimeError(f"review endpoint tree contains a special member: {relative}")
        if path.suffix.lower() in FORBIDDEN_REVIEW_TREE_SUFFIXES:
            raise RuntimeError(f"review endpoint tree contains forbidden payload type: {relative}")
        if path.suffix.lower() == ".png":
            endpoint_count += 1
        elif path.suffix.lower() == ".json":
            _reject_material_forbidden_json_fields(contract.load_json(path), relative)
    if endpoint_count == 0:
        raise RuntimeError("external review endpoint tree contains no endpoint PNG")


def validate_source_receipt(
    path: Path,
    phase: str,
    phase_plan_identity: str,
    expected_count: int,
    cohort_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    value = contract.load_json(path)
    fields = {
        "schema_version",
        "status",
        "phase",
        "event_protocol_identity_sha256",
        "phase_plan_identity_sha256",
        "model",
        "sampler",
        "cfg_scale",
        "endpoint_count",
        "endpoint_only_review_payload",
        "source_artifact_path",
        "source_artifact_identity_sha256",
        "source_manifest_sha256",
        "source_payload_role",
        "method_traces_sibling_path",
        "physical_tree_separation_verified",
        "labels_reviews_metrics_features_embeddings_or_scores_opened_for_sampling",
    }
    if set(value) != fields:
        raise RuntimeError(f"source receipt schema changed: extra={sorted(set(value)-fields)}, missing={sorted(fields-set(value))}")
    source_artifact = contract.require_directory(Path(value["source_artifact_path"]), "source endpoint artifact")
    validate_external_review_endpoint_tree(source_artifact)
    payload_role = value.get("source_payload_role")
    method_sibling = value.get("method_traces_sibling_path")
    if phase == "confirmation":
        if payload_role != "external_review_endpoints_only" or source_artifact.name != "review_endpoints":
            raise RuntimeError("confirmation review source must be the dedicated review_endpoints root")
        if not isinstance(method_sibling, str):
            raise RuntimeError("confirmation receipt must bind the physically separate method_traces sibling")
        method_root = contract.require_directory(Path(method_sibling), "method_traces sibling")
        if (
            method_root.name != "method_traces"
            or method_root.parent != source_artifact.parent
            or method_root == source_artifact
        ):
            raise RuntimeError("method_traces and review_endpoints are not exact sibling trees")
    elif payload_role != "endpoint_only_screen_root" or method_sibling is not None:
        raise RuntimeError("discovery/anchor review must use an endpoint-only screen root")
    source_manifest = contract.require_regular(source_artifact / "pool_manifest.json", "source pool manifest")
    source_manifest_value = contract.load_json(source_manifest)
    source_identity = source_manifest_value.get("identity_sha256")
    source_identity_record = source_manifest_value.get("identity")
    manifest_event_identity = source_manifest_value.get(
        "event_protocol_identity_sha256"
    )
    if isinstance(source_identity_record, dict):
        manifest_event_identity = source_identity_record.get(
            "event_protocol_identity_sha256", manifest_event_identity
        )
    if (
        value.get("schema_version") != 1
        or value.get("status") != "complete"
        or value.get("phase") != phase
        or value.get("event_protocol_identity_sha256") != contract.EVENT_PROTOCOL_IDENTITY
        or value.get("phase_plan_identity_sha256") != phase_plan_identity
        or value.get("model") != "DiT-XL/2 ImageNet-256"
        or value.get("sampler") != "official 250-step ancestral DDPM"
        or value.get("cfg_scale") != 4.0
        or value.get("endpoint_count") != expected_count
        or value.get("endpoint_only_review_payload") is not True
        or value.get("physical_tree_separation_verified") is not True
        or value.get("labels_reviews_metrics_features_embeddings_or_scores_opened_for_sampling") is not False
        or value.get("source_manifest_sha256") != contract.sha256_file(source_manifest)
        or not isinstance(source_identity, str)
        or len(source_identity) != 64
        or value.get("source_artifact_identity_sha256") != source_identity
        or source_manifest_value.get("status") != "complete"
        or manifest_event_identity != contract.EVENT_PROTOCOL_IDENTITY
    ):
        raise RuntimeError("endpoint source receipt failed scientific/provenance validation")
    if phase == "confirmation":
        nested_anchor = (
            source_identity_record.get("anchor_plan_identity_sha256")
            if isinstance(source_identity_record, dict)
            else None
        )
        top_anchor = source_manifest_value.get("anchor_plan_identity_sha256")
        observed_anchor = nested_anchor if nested_anchor is not None else top_anchor
        if (
            observed_anchor != phase_plan_identity
            or (nested_anchor is not None and top_anchor is not None and nested_anchor != top_anchor)
        ):
            raise RuntimeError(
                "confirmation review_endpoints pool is not bound to the authoritative anchor plan"
            )
    if not isinstance(source_manifest_value.get("pair_outputs"), list):
        raise RuntimeError("review source manifest must expose endpoint-only pair_outputs")
    if "pairs" in source_manifest_value:
        raise RuntimeError("review source manifest must not expose dynamic trace pairs")
    raw_pairs = source_manifest_value["pair_outputs"]
    pair_records: dict[tuple[int, int], Mapping[str, Any]] = {}
    for row in raw_pairs:
        if not isinstance(row, dict):
            raise RuntimeError("source pool pair receipt is malformed")
        class_id = row.get("class_id")
        seed = row.get("global_seed")
        if type(class_id) is not int or type(seed) is not int:
            raise RuntimeError("source pool pair receipt key types changed")
        key = (class_id, seed)
        if key in pair_records:
            raise RuntimeError("source pool manifest duplicates a phase pair key")
        pair_records[key] = row
    expected_keys = {
        (int(row["class_id"]), int(row["global_seed"])) for row in cohort_rows
    }
    if set(pair_records) != expected_keys:
        raise RuntimeError("source pool phase pair receipts differ from the exact review cohort axis")
    for row in cohort_rows:
        key = (int(row["class_id"]), int(row["global_seed"]))
        source = pair_records[key]
        pair_identity = source.get("identity_sha256")
        if (
            pair_identity != row["source_pair_identity_sha256"]
            or source.get("manifest_sha256") != row["source_manifest_sha256"]
            or source.get("endpoint_sha256") != row["image_sha256"]
            or source.get("endpoint_pixel_sha256") != row["image_pixel_sha256"]
        ):
            raise RuntimeError(f"source pool pair receipt does not bind cohort endpoint {key}")
        image_path = contract.require_regular(
            Path(row["source_image_path"]), "review endpoint cohort image"
        )
        try:
            image_path.relative_to(source_artifact)
        except ValueError as exc:
            raise RuntimeError("review endpoint index points outside the isolated source tree") from exc
    return value


def lock_cohort(args: argparse.Namespace) -> Path:
    phase_plan, classes, seeds = validate_phase_plan(args.phase_plan, args.phase)
    expected_axis = tuple((str(class_id), str(seed)) for seed in seeds for class_id in classes)
    # Scientific-v4 uses the sampler's exact global-seed-major, class-minor axis.
    rows = contract.read_csv_exact(args.endpoint_index, COHORT_INDEX_FIELDS)
    contract.reject_forbidden_columns(COHORT_INDEX_FIELDS)
    contract.validate_unique_axis(rows, ("class_id", "global_seed"), expected_axis)
    if len(rows) != len(classes) * len(seeds):
        raise RuntimeError("endpoint cohort is not the exact Cartesian phase axis")
    name_by_id = {str(row["class_id"]): row["class_name"] for row in roster()}
    normalized: list[dict[str, Any]] = []
    pixel_hashes: set[str] = set()
    for row in rows:
        if row["class_name"] != name_by_id[row["class_id"]]:
            raise RuntimeError("endpoint class name differs from frozen roster")
        inspected = inspect_image(Path(row["image_path"]))
        checks = {
            "image_sha256": inspected["image_sha256"],
            "image_pixel_sha256": inspected["image_pixel_sha256"],
            "width": "256",
            "height": "256",
            "mode": "RGB",
        }
        if any(row[key] != value for key, value in checks.items()):
            raise RuntimeError(f"endpoint image provenance mismatch: class={row['class_id']} seed={row['global_seed']}")
        if inspected["image_pixel_sha256"] in pixel_hashes:
            raise RuntimeError("endpoint cohort contains duplicate pixels")
        pixel_hashes.add(inspected["image_pixel_sha256"])
        if len(row["source_pair_identity_sha256"]) != 64 or len(row["source_manifest_sha256"]) != 64:
            raise RuntimeError("endpoint source identity/hash is malformed")
        normalized.append(
            {
                "class_id": int(row["class_id"]),
                "global_seed": int(row["global_seed"]),
                "class_name": row["class_name"],
                "source_image_path": inspected["path"],
                "image_sha256": inspected["image_sha256"],
                "image_pixel_sha256": inspected["image_pixel_sha256"],
                "source_pair_identity_sha256": row["source_pair_identity_sha256"],
                "source_manifest_sha256": row["source_manifest_sha256"],
            }
        )
    receipt = validate_source_receipt(
        args.source_receipt,
        args.phase,
        phase_plan["identity_sha256"],
        len(normalized),
        normalized,
    )
    identity = {
        "schema_version": 1,
        "artifact_kind": "EVENT_RICH_ENDPOINT_COHORT_LOCK_V1",
        "status": "EXACT_ENDPOINT_PHASE_AXIS_LOCKED",
        "phase": args.phase,
        "event_protocol_identity_sha256": contract.EVENT_PROTOCOL_IDENTITY,
        "phase_plan_identity_sha256": phase_plan["identity_sha256"],
        "class_ids_ordered": list(classes),
        "global_seeds_ordered": list(seeds),
        "axis_order": "global-seed-major, class-minor",
        "endpoint_count": len(normalized),
        "source_artifact_identity_sha256": receipt["source_artifact_identity_sha256"],
        "source_manifest_sha256": receipt["source_manifest_sha256"],
        "labels_scores_features_trajectories_embeddings_or_ranks_read": False,
        **source_binding(),
    }

    def builder(root: Path) -> None:
        shutil.copyfile(args.phase_plan, root / "phase_plan.json")
        shutil.copyfile(args.source_receipt, root / "source_receipt.json")
        contract.write_csv(root / "cohort_rows.csv", COHORT_LOCK_FIELDS, normalized)

    return contract.publish_artifact(args.output, identity=identity, builder=builder)


def validate_qualification(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    manifest, _ = contract.validate_artifact(
        path, expected_kind="EVENT_RICH_PANEL_QUALIFICATION_EVALUATION_V1"
    )
    identity = manifest["identity"]
    tokens = identity.get("qualified_role_tokens")
    if (
        identity.get("event_protocol_identity_sha256") != contract.EVENT_PROTOCOL_IDENTITY
        or identity.get("panel_passed") is not True
        or identity.get("formal_screen_release_authorized") is not True
        or not contract.qualified_role_tokens_valid(tokens)
    ):
        raise RuntimeError("formal screen cannot be released: five-role qualification did not pass")
    return manifest, tokens


def validate_cohort(path: Path, phase: str | None = None) -> tuple[dict[str, Any], list[dict[str, str]]]:
    manifest, _ = contract.validate_artifact(path, expected_kind="EVENT_RICH_ENDPOINT_COHORT_LOCK_V1")
    identity = manifest["identity"]
    if identity.get("event_protocol_identity_sha256") != contract.EVENT_PROTOCOL_IDENTITY:
        raise RuntimeError("endpoint cohort event-protocol binding changed")
    if phase is not None and identity.get("phase") != phase:
        raise RuntimeError("endpoint cohort phase mismatch")
    rows = contract.read_csv_exact(path / "cohort_rows.csv", COHORT_LOCK_FIELDS)
    expected = [
        (str(class_id), str(seed))
        for seed in identity["global_seeds_ordered"]
        for class_id in identity["class_ids_ordered"]
    ]
    contract.validate_unique_axis(rows, ("class_id", "global_seed"), expected)
    if len(rows) != identity.get("endpoint_count"):
        raise RuntimeError("cohort row count changed")
    for row in rows:
        image = inspect_image(Path(row["source_image_path"]))
        if image["image_sha256"] != row["image_sha256"] or image["image_pixel_sha256"] != row["image_pixel_sha256"]:
            raise RuntimeError("cohort source endpoint changed after lock")
    return manifest, rows


def review_mapping_rows(pack_identity: str, slot: str, rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        blind_id = "r_" + contract.stable_blind_id(
            "eqvae.dit.event-rich.production-review.v1",
            pack_identity,
            slot,
            row["class_id"],
            row["global_seed"],
        )
        result.append(
            {
                "blind_id": blind_id,
                "class_id": row["class_id"],
                "global_seed": row["global_seed"],
                "class_name": row["class_name"],
                "source_image_path": row["source_image_path"],
                "image_sha256": row["image_sha256"],
                "image_pixel_sha256": row["image_pixel_sha256"],
            }
        )
    result.sort(key=lambda row: contract.stable_blind_id(pack_identity, slot, row["blind_id"]))
    return result


def build_review_pack(args: argparse.Namespace) -> Path:
    qualification_manifest, tokens = validate_qualification(args.qualification_evaluation)
    cohort_manifest, cohort_rows = validate_cohort(args.cohort_lock, args.phase)
    pack_seed_identity = contract.canonical_sha256(
        {
            "phase": args.phase,
            "cohort": cohort_manifest["identity_sha256"],
            "qualification": qualification_manifest["identity_sha256"],
        }
    )
    mappings = {
        slot: review_mapping_rows(pack_seed_identity, slot, cohort_rows)
        for slot in contract.REVIEWER_SLOTS
    }
    identity = {
        "schema_version": 1,
        "artifact_kind": "EVENT_RICH_THREE_REVIEWER_BLIND_INPUT_LOCK_V1",
        "status": "THREE_ISOLATED_ENDPOINT_ONLY_REVIEWER_DELIVERIES_FROZEN",
        "phase": args.phase,
        "event_protocol_identity_sha256": contract.EVENT_PROTOCOL_IDENTITY,
        "cohort_identity_sha256": cohort_manifest["identity_sha256"],
        "qualification_evaluation_identity_sha256": qualification_manifest["identity_sha256"],
        "class_ids_ordered": list(cohort_manifest["identity"]["class_ids_ordered"]),
        "global_seeds_ordered": list(cohort_manifest["identity"]["global_seeds_ordered"]),
        "reviewer_slots": list(contract.REVIEWER_SLOTS),
        "endpoint_count_per_reviewer": len(cohort_rows),
        "delivery_rule": "give each reviewer only delivery/<their role slot>",
        "class_comparison_band_visible": True,
        "native_resolution_review_required_for_every_suspicious_grid_item": True,
        "candidate_class_rank_scores_features_trajectories_embeddings_thresholds_alerts_other_votes_visible": False,
        **source_binding(),
    }

    def builder(root: Path) -> None:
        custody = root / "custody"
        delivery = root / "delivery"
        custody.mkdir()
        delivery.mkdir()
        shutil.copyfile(args.cohort_lock / "phase_plan.json", custody / "phase_plan.json")
        for slot in contract.REVIEWER_SLOTS:
            slot_delivery = delivery / slot
            images = slot_delivery / "images"
            slot_delivery.mkdir()
            images.mkdir()
            mapping_fields = (
                "blind_id",
                "class_id",
                "global_seed",
                "class_name",
                "source_image_path",
                "image_sha256",
                "image_pixel_sha256",
            )
            contract.write_csv(custody / f"{slot}_mapping.csv", mapping_fields, mappings[slot])
            index_rows = []
            form_rows = []
            for row in mappings[slot]:
                relative = f"images/{row['blind_id']}.png"
                shutil.copyfile(row["source_image_path"], slot_delivery / relative)
                if contract.sha256_file(slot_delivery / relative) != row["image_sha256"]:
                    raise RuntimeError("review delivery copy changed endpoint bytes")
                index_rows.append(
                    {
                        "blind_id": row["blind_id"],
                        "class_id": row["class_id"],
                        "class_name": row["class_name"],
                        "image_relative_path": relative,
                        "image_sha256": row["image_sha256"],
                        "image_pixel_sha256": row["image_pixel_sha256"],
                        "mode": "RGB",
                        "width": 256,
                        "height": 256,
                    }
                )
                form_rows.append(
                    {
                        "blind_id": row["blind_id"],
                        "severity": "",
                        "components": "",
                        "localization_reason": "",
                        "role_slot": slot,
                        "role_token": tokens[slot],
                        "independence_attestation": "",
                    }
                )
            contract.write_csv(slot_delivery / "item_index.csv", REVIEW_DELIVERY_FIELDS, index_rows)
            contract.write_csv(slot_delivery / "response_template.csv", contract.REVIEW_RESPONSE_FIELDS, form_rows)
            contract.write_json(
                slot_delivery / "review_instructions.json",
                {
                    "role_slot": slot,
                    "endpoint_only": True,
                    "review_every_row": True,
                    "inspect_native_resolution_if_suspicious": True,
                    "recognizable_subject_does_not_imply_quality_normal": True,
                    "severity_2_or_3_requires_component_and_localization_reason": True,
                    "do_not_access": [
                        "another role's delivery or response",
                        "candidate B/E values, class enrichment rank or trigger stratum",
                        "trajectory, metric, threshold, alert or embedding/FID/DINO/Inception data",
                    ],
                },
            )

    return contract.publish_artifact(args.output, identity=identity, builder=builder)


def validate_review_pack(path: Path, phase: str | None = None) -> tuple[dict[str, Any], dict[str, str]]:
    manifest, _ = contract.validate_artifact(
        path, expected_kind="EVENT_RICH_THREE_REVIEWER_BLIND_INPUT_LOCK_V1"
    )
    identity = manifest["identity"]
    if identity.get("event_protocol_identity_sha256") != contract.EVENT_PROTOCOL_IDENTITY:
        raise RuntimeError("review pack event-protocol binding changed")
    if phase is not None and identity.get("phase") != phase:
        raise RuntimeError("review pack phase changed")
    # Tokens are recovered from immutable templates rather than by path search.
    recovered: dict[str, str] = {}
    for slot in contract.REVIEWER_SLOTS:
        rows = contract.read_csv_exact(path / f"delivery/{slot}/response_template.csv", contract.REVIEW_RESPONSE_FIELDS)
        values = {row["role_token"] for row in rows}
        if len(values) != 1 or not next(iter(values)):
            raise RuntimeError("review template role-token binding changed")
        recovered[slot] = next(iter(values))
    return manifest, recovered
def lock_review(args: argparse.Namespace) -> Path:
    pack_manifest, tokens = validate_review_pack(args.review_pack, args.phase)
    if args.slot not in contract.REVIEWER_SLOTS:
        raise RuntimeError("invalid reviewer slot")
    mapping_fields = (
        "blind_id",
        "class_id",
        "global_seed",
        "class_name",
        "source_image_path",
        "image_sha256",
        "image_pixel_sha256",
    )
    mapping = contract.read_csv_exact(args.review_pack / f"custody/{args.slot}_mapping.csv", mapping_fields)
    responses = contract.read_csv_exact(args.completed_form, contract.REVIEW_RESPONSE_FIELDS)
    contract.validate_unique_axis(
        responses,
        ("blind_id",),
        [(row["blind_id"],) for row in mapping],
    )
    if {row["role_slot"] for row in responses} != {args.slot} or {row["role_token"] for row in responses} != {tokens[args.slot]}:
        raise RuntimeError("completed review role slot/token differs from passed panel")
    parsed: dict[str, dict[str, Any]] = {}
    for row in responses:
        contract.validate_attestation(row["independence_attestation"])
        parsed[row["blind_id"]] = contract.validate_severity_row(row)
    locked = []
    for row in mapping:
        value = parsed[row["blind_id"]]
        locked.append(
            {
                "class_id": row["class_id"],
                "global_seed": row["global_seed"],
                "image_sha256": row["image_sha256"],
                "image_pixel_sha256": row["image_pixel_sha256"],
                "severity": value["severity"],
                "components": ";".join(value["components"]),
                "localization_reason": value["localization_reason"],
            }
        )
    class_rank = {
        str(class_id): index
        for index, class_id in enumerate(pack_manifest["identity"]["class_ids_ordered"])
    }
    seed_rank = {
        str(seed): index
        for index, seed in enumerate(pack_manifest["identity"]["global_seeds_ordered"])
    }
    locked.sort(
        key=lambda row: (seed_rank[row["global_seed"]], class_rank[row["class_id"]])
    )
    identity = {
        "schema_version": 1,
        "artifact_kind": "EVENT_RICH_SINGLE_REVIEWER_LABEL_LOCK_V1",
        "status": "ONE_COMPLETE_ENDPOINT_ONLY_REVIEW_LOCKED",
        "phase": args.phase,
        "event_protocol_identity_sha256": contract.EVENT_PROTOCOL_IDENTITY,
        "review_pack_identity_sha256": pack_manifest["identity_sha256"],
        "role_slot": args.slot,
        "role_token": tokens[args.slot],
        "row_count": len(locked),
        "reviewer_independent_role_attested": True,
        "actual_personhood_or_organizational_independence_verified_by_software": False,
        "other_votes_candidate_scores_features_trajectories_embeddings_ranks_visible": False,
        **source_binding(),
    }

    def builder(root: Path) -> None:
        contract.write_csv(root / "review_rows.csv", REVIEW_LOCK_FIELDS, locked)
        shutil.copyfile(args.completed_form, root / "submitted_blind_form.csv")

    return contract.publish_artifact(args.output, identity=identity, builder=builder)


def validate_review_lock(path: Path, pack_identity: str, slot: str, phase: str) -> tuple[dict[str, Any], list[dict[str, str]]]:
    manifest, _ = contract.validate_artifact(path, expected_kind="EVENT_RICH_SINGLE_REVIEWER_LABEL_LOCK_V1")
    identity = manifest["identity"]
    if (
        identity.get("event_protocol_identity_sha256") != contract.EVENT_PROTOCOL_IDENTITY
        or identity.get("review_pack_identity_sha256") != pack_identity
        or identity.get("role_slot") != slot
        or identity.get("phase") != phase
    ):
        raise RuntimeError(f"review lock binding failed for {slot}")
    rows = contract.read_csv_exact(path / "review_rows.csv", REVIEW_LOCK_FIELDS)
    contract.validate_unique_axis(rows, ("class_id", "global_seed"))
    if len(rows) != identity.get("row_count"):
        raise RuntimeError("review-lock row count changed")
    return manifest, rows


def build_adjudication_pack(args: argparse.Namespace) -> Path:
    pack_manifest, _ = validate_review_pack(args.review_pack, args.phase)
    reviews: dict[str, tuple[dict[str, Any], list[dict[str, str]]]] = {}
    for slot in contract.REVIEWER_SLOTS:
        reviews[slot] = validate_review_lock(
            getattr(args, slot), pack_manifest["identity_sha256"], slot, args.phase
        )
    axes = [tuple((row["class_id"], row["global_seed"], row["image_sha256"], row["image_pixel_sha256"]) for row in value[1]) for value in reviews.values()]
    if any(axis != axes[0] for axis in axes[1:]):
        raise RuntimeError("three review locks do not cover the identical endpoint axis/order")
    vote_rows = []
    for index, key in enumerate(axes[0]):
        votes = [int(reviews[slot][1][index]["severity"]) >= 2 for slot in contract.REVIEWER_SLOTS]
        vote_rows.append({"key": key, "positive_vote_count": sum(votes)})
    positives = [row for row in vote_rows if row["positive_vote_count"] >= 1]
    negatives = [row for row in vote_rows if row["positive_vote_count"] == 0]
    if len(positives) > len(negatives):
        raise RuntimeError("not enough zero-positive rows for an equal-count decoy sample")
    random_domain = contract.canonical_sha256(
        {
            "domain": "eqvae.dit.event-rich.adjudication-decoy.v1",
            "review_lock_identities": [reviews[slot][0]["identity_sha256"] for slot in contract.REVIEWER_SLOTS],
        }
    )
    negatives.sort(key=lambda row: contract.stable_blind_id(random_domain, *row["key"], length=64))
    decoys = negatives[: len(positives)]
    selected = [
        {**row, "selection_kind": "any_reviewer_positive"} for row in positives
    ] + [{**row, "selection_kind": "zero_positive_random_decoy"} for row in decoys]
    selected.sort(key=lambda row: contract.stable_blind_id(random_domain, "pack", *row["key"], length=64))
    cohort_manifest, cohort_rows = validate_cohort(args.cohort_lock, args.phase)
    if cohort_manifest["identity_sha256"] != pack_manifest["identity"].get("cohort_identity_sha256"):
        raise RuntimeError("adjudication cohort differs from review pack cohort")
    cohort_by_key = {(row["class_id"], row["global_seed"]): row for row in cohort_rows}
    qual_manifest, tokens = validate_qualification(args.qualification_evaluation)
    if qual_manifest["identity_sha256"] != pack_manifest["identity"].get("qualification_evaluation_identity_sha256"):
        raise RuntimeError("adjudication panel qualification differs from review pack")
    identity_seed = contract.canonical_sha256(
        {"phase": args.phase, "review_locks": [reviews[slot][0]["identity_sha256"] for slot in contract.REVIEWER_SLOTS], "random_domain": random_domain}
    )
    mappings: dict[str, list[dict[str, Any]]] = {}
    for slot in contract.ADJUDICATOR_SLOTS:
        rows = []
        for item in selected:
            class_id, seed, image_sha, pixel_sha = item["key"]
            adjudication_id = "a_" + contract.stable_blind_id(
                "eqvae.dit.event-rich.adjudication.v1", identity_seed, slot, class_id, seed
            )
            rows.append(
                {
                    "adjudication_id": adjudication_id,
                    "class_id": class_id,
                    "global_seed": seed,
                    "image_sha256": image_sha,
                    "image_pixel_sha256": pixel_sha,
                    "selection_kind": item["selection_kind"],
                    "positive_vote_count": item["positive_vote_count"],
                }
            )
        rows.sort(key=lambda row: contract.stable_blind_id(identity_seed, slot, row["adjudication_id"], length=64))
        mappings[slot] = rows
    identity = {
        "schema_version": 1,
        "artifact_kind": "EVENT_RICH_DUAL_ADJUDICATOR_BLIND_INPUT_LOCK_V1",
        "status": "TWO_ISOLATED_ADJUDICATOR_DELIVERIES_FROZEN",
        "phase": args.phase,
        "event_protocol_identity_sha256": contract.EVENT_PROTOCOL_IDENTITY,
        "review_pack_identity_sha256": pack_manifest["identity_sha256"],
        "review_lock_identities": {slot: reviews[slot][0]["identity_sha256"] for slot in contract.REVIEWER_SLOTS},
        "qualification_evaluation_identity_sha256": qual_manifest["identity_sha256"],
        "union_any_positive_count": len(positives),
        "equal_random_decoy_count": len(decoys),
        "adjudication_item_count": len(selected),
        "decoy_rng": "SHA256 sort; domain and full review-lock identities frozen",
        "delivery_rule": "give each adjudicator only delivery/<their role slot>",
        "candidate_rank_scores_features_trajectories_embeddings_votes_reviewer_identity_trigger_membership_visible": False,
        **source_binding(),
    }

    def builder(root: Path) -> None:
        custody = root / "custody"
        delivery = root / "delivery"
        custody.mkdir()
        delivery.mkdir()
        for slot in contract.ADJUDICATOR_SLOTS:
            slot_root = delivery / slot
            images = slot_root / "images"
            slot_root.mkdir()
            images.mkdir()
            contract.write_csv(custody / f"{slot}_mapping.csv", ADJUDICATION_MAPPING_FIELDS, mappings[slot])
            index_rows = []
            forms = []
            for row in mappings[slot]:
                cohort = cohort_by_key[(row["class_id"], row["global_seed"])]
                relative = f"images/{row['adjudication_id']}.png"
                shutil.copyfile(cohort["source_image_path"], slot_root / relative)
                if contract.sha256_file(slot_root / relative) != row["image_sha256"]:
                    raise RuntimeError("adjudication delivery copy changed endpoint bytes")
                index_rows.append(
                    {
                        "blind_id": row["adjudication_id"],
                        "class_id": row["class_id"],
                        "class_name": cohort["class_name"],
                        "image_relative_path": relative,
                        "image_sha256": row["image_sha256"],
                        "image_pixel_sha256": row["image_pixel_sha256"],
                        "mode": "RGB",
                        "width": 256,
                        "height": 256,
                    }
                )
                forms.append(
                    {
                        "adjudication_id": row["adjudication_id"],
                        "decision": "",
                        "components": "",
                        "localization_reason": "",
                        "role_slot": slot,
                        "role_token": tokens[slot],
                        "independence_attestation": "",
                    }
                )
            contract.write_csv(slot_root / "item_index.csv", ADJUDICATION_DELIVERY_FIELDS, index_rows)
            contract.write_csv(slot_root / "response_template.csv", contract.ADJUDICATION_RESPONSE_FIELDS, forms)
            contract.write_json(
                slot_root / "adjudication_instructions.json",
                {
                    "role_slot": slot,
                    "independent_endpoint_only_decision": True,
                    "allowed_decisions": ["clear_bad", "mild_or_not_clear_bad"],
                    "specific_component_and_written_reason_required": True,
                    "do_not_access": [
                        "other adjudicator decision",
                        "reviewer identities, votes, vote counts or trigger/decoy membership",
                        "candidate B/E values, class enrichment ranks, trajectory/metric/embedding data",
                    ],
                },
            )

    return contract.publish_artifact(args.output, identity=identity, builder=builder)


def validate_adjudication_pack(path: Path, phase: str) -> tuple[dict[str, Any], dict[str, str]]:
    manifest, _ = contract.validate_artifact(
        path, expected_kind="EVENT_RICH_DUAL_ADJUDICATOR_BLIND_INPUT_LOCK_V1"
    )
    identity = manifest["identity"]
    if identity.get("phase") != phase or identity.get("event_protocol_identity_sha256") != contract.EVENT_PROTOCOL_IDENTITY:
        raise RuntimeError("adjudication pack phase/protocol binding changed")
    tokens = {}
    for slot in contract.ADJUDICATOR_SLOTS:
        rows = contract.read_csv_exact(path / f"delivery/{slot}/response_template.csv", contract.ADJUDICATION_RESPONSE_FIELDS, allow_empty=True)
        if rows:
            values = {row["role_token"] for row in rows}
            if len(values) != 1 or not next(iter(values)):
                raise RuntimeError("adjudication template role token changed")
            tokens[slot] = next(iter(values))
        else:
            tokens[slot] = "EMPTY_AUDIT_NO_RESPONSE_ROWS"
    return manifest, tokens


def lock_adjudication(args: argparse.Namespace) -> Path:
    pack_manifest, tokens = validate_adjudication_pack(args.adjudication_pack, args.phase)
    if args.slot not in contract.ADJUDICATOR_SLOTS:
        raise RuntimeError("invalid adjudicator slot")
    mapping = contract.read_csv_exact(
        args.adjudication_pack / f"custody/{args.slot}_mapping.csv",
        ADJUDICATION_MAPPING_FIELDS,
        allow_empty=True,
    )
    responses = contract.read_csv_exact(
        args.completed_form, contract.ADJUDICATION_RESPONSE_FIELDS, allow_empty=not mapping
    )
    contract.validate_unique_axis(
        responses,
        ("adjudication_id",),
        [(row["adjudication_id"],) for row in mapping],
    )
    locked = []
    if mapping:
        if {row["role_slot"] for row in responses} != {args.slot} or {row["role_token"] for row in responses} != {tokens[args.slot]}:
            raise RuntimeError("completed adjudication role slot/token differs from passed panel")
        by_id = {}
        for row in responses:
            contract.validate_attestation(row["independence_attestation"])
            if row["decision"] not in {"clear_bad", "mild_or_not_clear_bad"}:
                raise RuntimeError("invalid adjudication decision")
            proxy_severity = "2" if row["decision"] == "clear_bad" else "1"
            parsed = contract.validate_severity_row(
                {**row, "severity": proxy_severity}, always_reason=True
            )
            by_id[row["adjudication_id"]] = {"decision": row["decision"], **parsed}
        for row in mapping:
            value = by_id[row["adjudication_id"]]
            locked.append(
                {
                    "class_id": row["class_id"],
                    "global_seed": row["global_seed"],
                    "image_sha256": row["image_sha256"],
                    "image_pixel_sha256": row["image_pixel_sha256"],
                    "decision": value["decision"],
                    "components": ";".join(value["components"]),
                    "localization_reason": value["localization_reason"],
                }
            )
    identity = {
        "schema_version": 1,
        "artifact_kind": "EVENT_RICH_SINGLE_ADJUDICATOR_LOCK_V1",
        "status": "ONE_COMPLETE_BLIND_ADJUDICATION_LOCKED",
        "phase": args.phase,
        "event_protocol_identity_sha256": contract.EVENT_PROTOCOL_IDENTITY,
        "adjudication_pack_identity_sha256": pack_manifest["identity_sha256"],
        "role_slot": args.slot,
        "role_token": None if not mapping else tokens[args.slot],
        "row_count": len(locked),
        "reviewer_votes_identity_trigger_decoy_candidate_scores_features_trajectories_embeddings_ranks_visible": False,
        "actual_personhood_or_organizational_independence_verified_by_software": False,
        **source_binding(),
    }

    def builder(root: Path) -> None:
        contract.write_csv(root / "adjudication_rows.csv", ADJUDICATION_LOCK_FIELDS, locked)
        shutil.copyfile(args.completed_form, root / "submitted_blind_form.csv")

    return contract.publish_artifact(args.output, identity=identity, builder=builder)


def validate_adjudication_lock(path: Path, pack_identity: str, slot: str, phase: str) -> tuple[dict[str, Any], list[dict[str, str]]]:
    manifest, _ = contract.validate_artifact(path, expected_kind="EVENT_RICH_SINGLE_ADJUDICATOR_LOCK_V1")
    identity = manifest["identity"]
    if identity.get("adjudication_pack_identity_sha256") != pack_identity or identity.get("role_slot") != slot or identity.get("phase") != phase:
        raise RuntimeError("adjudication lock binding changed")
    rows = contract.read_csv_exact(path / "adjudication_rows.csv", ADJUDICATION_LOCK_FIELDS, allow_empty=True)
    contract.validate_unique_axis(rows, ("class_id", "global_seed"))
    if len(rows) != identity.get("row_count"):
        raise RuntimeError("adjudication lock row count changed")
    return manifest, rows


def component_groups(component_lists: Sequence[set[str]], minimum: int) -> tuple[bool, bool, bool]:
    blur = sum(bool(values & contract.BLUR_COMPONENTS) for values in component_lists) >= minimum
    structure = sum(bool(values & contract.STRUCTURE_COMPONENTS) for values in component_lists) >= minimum
    other = sum(bool(values & {"texture_break", "other"}) for values in component_lists) >= minimum
    return blur, structure, other


def consensus_policy(
    positive_count: int,
    zero_count: int,
    adjudicator_decisions: Sequence[str],
) -> tuple[str, str, bool]:
    """Apply the frozen asymmetric label-change policy.

    ``changed`` means adjudication changed the raw three-reviewer label and is
    subsequently used to require two written component/reason records.
    """

    if positive_count not in range(4) or zero_count not in range(4):
        raise RuntimeError("reviewer vote counts must lie in [0,3]")
    if len(adjudicator_decisions) != 2:
        raise RuntimeError("exactly two adjudicator decisions are required")
    allowed = {"clear_bad", "mild_or_not_clear_bad", "not_audited"}
    if any(value not in allowed for value in adjudicator_decisions):
        raise RuntimeError("invalid adjudicator decision in consensus policy")
    raw_label = (
        "clear_bad"
        if positive_count >= 2
        else ("clean_good" if zero_count >= 2 else "mild_or_disputed")
    )
    decisions = list(adjudicator_decisions)
    if positive_count == 3:
        return "clear_bad", "unanimous_3of3_never_downgradable", False
    if positive_count == 2:
        if decisions == ["mild_or_not_clear_bad", "mild_or_not_clear_bad"]:
            return (
                "mild_or_disputed",
                "raw_2of3_downgraded_by_two_unanimous_adjudicators",
                True,
            )
        return (
            "clear_bad",
            "raw_2of3_retained_without_unanimous_dual_downgrade",
            False,
        )
    if decisions == ["clear_bad", "clear_bad"]:
        return (
            "clear_bad",
            "raw_nonmajority_promoted_by_two_unanimous_adjudicators",
            True,
        )
    return raw_label, "raw_consensus_preserved", False


def pair_diagnostics(left: Sequence[int], right: Sequence[int]) -> dict[str, Any]:
    n11 = sum(a == 1 and b == 1 for a, b in zip(left, right))
    n10 = sum(a == 1 and b == 0 for a, b in zip(left, right))
    n01 = sum(a == 0 and b == 1 for a, b in zip(left, right))
    n00 = sum(a == 0 and b == 0 for a, b in zip(left, right))
    return {
        "n11": n11,
        "n10": n10,
        "n01": n01,
        "n00": n00,
        "positive_agreement": contract.positive_agreement(left, right),
        "binary_cohen_kappa": contract.binary_cohen_kappa(left, right),
    }


def lock_consensus(args: argparse.Namespace) -> Path:
    review_pack_manifest, _ = validate_review_pack(args.review_pack, args.phase)
    adjudication_pack_manifest, _ = validate_adjudication_pack(args.adjudication_pack, args.phase)
    if adjudication_pack_manifest["identity"].get("review_pack_identity_sha256") != review_pack_manifest["identity_sha256"]:
        raise RuntimeError("review and adjudication packs are not bound")
    reviews = {
        slot: validate_review_lock(
            getattr(args, slot), review_pack_manifest["identity_sha256"], slot, args.phase
        )
        for slot in contract.REVIEWER_SLOTS
    }
    adjudications = {
        slot: validate_adjudication_lock(
            getattr(args, slot), adjudication_pack_manifest["identity_sha256"], slot, args.phase
        )
        for slot in contract.ADJUDICATOR_SLOTS
    }
    review_axes = [tuple((row["class_id"], row["global_seed"], row["image_sha256"], row["image_pixel_sha256"]) for row in value[1]) for value in reviews.values()]
    if any(axis != review_axes[0] for axis in review_axes[1:]):
        raise RuntimeError("review axes differ at consensus")
    selection = contract.read_csv_exact(
        args.adjudication_pack / "custody/adjudicator_1_mapping.csv",
        ADJUDICATION_MAPPING_FIELDS,
        allow_empty=True,
    )
    selection_by_key = {(row["class_id"], row["global_seed"]): row for row in selection}
    adj_by_slot = {
        slot: {(row["class_id"], row["global_seed"]): row for row in value[1]}
        for slot, value in adjudications.items()
    }
    if set(adj_by_slot["adjudicator_1"]) != set(adj_by_slot["adjudicator_2"]) or set(adj_by_slot["adjudicator_1"]) != set(selection_by_key):
        raise RuntimeError("two adjudication locks and frozen adjudication selection differ")
    final_rows = []
    per_reviewer_binary = {slot: [] for slot in contract.REVIEWER_SLOTS}
    for index, axis in enumerate(review_axes[0]):
        class_id, seed, image_sha, pixel_sha = axis
        key = (class_id, seed)
        reviewer_rows = [reviews[slot][1][index] for slot in contract.REVIEWER_SLOTS]
        binaries = [int(row["severity"]) >= 2 for row in reviewer_rows]
        for slot, value in zip(contract.REVIEWER_SLOTS, binaries):
            per_reviewer_binary[slot].append(int(value))
        positive_count = sum(binaries)
        zero_count = sum(int(row["severity"]) == 0 for row in reviewer_rows)
        raw_label = "clear_bad" if positive_count >= 2 else ("clean_good" if zero_count >= 2 else "mild_or_disputed")
        selected = selection_by_key.get(key)
        adj_rows = [adj_by_slot[slot].get(key) for slot in contract.ADJUDICATOR_SLOTS]
        decisions = [row["decision"] if row else "not_audited" for row in adj_rows]
        audit_kind = selected["selection_kind"] if selected else "not_audited"
        final_label, rule, changed = consensus_policy(
            positive_count, zero_count, decisions
        )
        # A changed label requires both independently written reasons and explicit components.
        if changed:
            if any(row is None or len(row["localization_reason"].strip()) < 12 or not row["components"].strip() for row in adj_rows):
                raise RuntimeError("adjudication changed a final label without two component/reason records")
        reviewer_components = [set(row["components"].split(";")) for row in reviewer_rows]
        if final_label == "clear_bad" and positive_count < 2:
            phenotype_inputs = [set(row["components"].split(";")) for row in adj_rows if row]
            blur, structure, other = component_groups(phenotype_inputs, 2)
        else:
            blur, structure, other = component_groups(reviewer_components, 2)
        clear = final_label == "clear_bad"
        phenotype_disputed = clear and not (blur or structure or other)
        final_rows.append(
            {
                "phase": args.phase,
                "class_id": class_id,
                "global_seed": seed,
                "image_sha256": image_sha,
                "image_pixel_sha256": pixel_sha,
                "reviewer_positive_count": positive_count,
                "reviewer_zero_count": zero_count,
                "raw_label": raw_label,
                "audit_selection": audit_kind,
                "adjudicator_1_decision": decisions[0],
                "adjudicator_2_decision": decisions[1],
                "final_label": final_label,
                "final_clear_bad": str(clear).lower(),
                "blur_component_consensus": str(blur).lower(),
                "structure_component_consensus": str(structure).lower(),
                "other_component_consensus": str(other).lower(),
                "blur_or_soft_fusion_positive": str(clear and blur).lower(),
                "structural_non_blur": str(clear and structure and not blur).lower(),
                "phenotype_disputed": str(phenotype_disputed).lower(),
                "consensus_rule": rule,
            }
        )
    pairwise = []
    for left_index, left in enumerate(contract.REVIEWER_SLOTS):
        for right in contract.REVIEWER_SLOTS[left_index + 1 :]:
            pairwise.append({"left": left, "right": right, **pair_diagnostics(per_reviewer_binary[left], per_reviewer_binary[right])})
    overall_counts = {
        "endpoint_count": len(final_rows),
        "raw_clear_bad": sum(row["raw_label"] == "clear_bad" for row in final_rows),
        "final_clean_good": sum(row["final_label"] == "clean_good" for row in final_rows),
        "final_mild_or_disputed": sum(row["final_label"] == "mild_or_disputed" for row in final_rows),
        "final_clear_bad": sum(row["final_label"] == "clear_bad" for row in final_rows),
        "final_blur_or_soft_fusion": sum(row["blur_or_soft_fusion_positive"] == "true" for row in final_rows),
        "final_structural_non_blur": sum(row["structural_non_blur"] == "true" for row in final_rows),
        "union_any_positive": adjudication_pack_manifest["identity"]["union_any_positive_count"],
        "random_decoys": adjudication_pack_manifest["identity"]["equal_random_decoy_count"],
        "promoted_union_minority": sum(row["consensus_rule"] == "raw_nonmajority_promoted_by_two_unanimous_adjudicators" and row["audit_selection"] == "any_reviewer_positive" for row in final_rows),
        "promoted_zero_positive_decoys": sum(row["consensus_rule"] == "raw_nonmajority_promoted_by_two_unanimous_adjudicators" and row["audit_selection"] == "zero_positive_random_decoy" for row in final_rows),
        "downgraded_raw_2of3": sum(row["consensus_rule"] == "raw_2of3_downgraded_by_two_unanimous_adjudicators" for row in final_rows),
        "unanimous_3of3_retained": sum(row["consensus_rule"] == "unanimous_3of3_never_downgradable" for row in final_rows),
    }
    per_class: dict[str, dict[str, int]] = {}
    for class_id in review_pack_manifest["identity"].get("class_ids_ordered", []):
        # Older pack identities did not expose this convenience field; rows
        # below remain authoritative and fill any selected class regardless.
        per_class[str(class_id)] = {
            "endpoint_count": 0,
            "raw_clear_bad": 0,
            "final_clean_good": 0,
            "final_mild_or_disputed": 0,
            "final_clear_bad": 0,
            "final_blur_or_soft_fusion": 0,
            "final_structural_non_blur": 0,
        }
    for row in final_rows:
        values = per_class.setdefault(
            str(row["class_id"]),
            {
                "endpoint_count": 0,
                "raw_clear_bad": 0,
                "final_clean_good": 0,
                "final_mild_or_disputed": 0,
                "final_clear_bad": 0,
                "final_blur_or_soft_fusion": 0,
                "final_structural_non_blur": 0,
            },
        )
        values["endpoint_count"] += 1
        values["raw_clear_bad"] += int(row["raw_label"] == "clear_bad")
        values["final_clean_good"] += int(row["final_label"] == "clean_good")
        values["final_mild_or_disputed"] += int(row["final_label"] == "mild_or_disputed")
        values["final_clear_bad"] += int(row["final_label"] == "clear_bad")
        values["final_blur_or_soft_fusion"] += int(row["blur_or_soft_fusion_positive"] == "true")
        values["final_structural_non_blur"] += int(row["structural_non_blur"] == "true")
    if (
        overall_counts["final_clean_good"]
        + overall_counts["final_mild_or_disputed"]
        + overall_counts["final_clear_bad"]
        != overall_counts["endpoint_count"]
        or any(
            values["final_clean_good"]
            + values["final_mild_or_disputed"]
            + values["final_clear_bad"]
            != values["endpoint_count"]
            for values in per_class.values()
        )
    ):
        raise RuntimeError("final clean/mild/clear-bad counts do not partition the exact endpoint axis")
    counts = {"phase": args.phase, "overall": overall_counts, "per_class": per_class}
    phase_plan = contract.load_json(args.review_pack / "custody/phase_plan.json")
    identity = {
        "schema_version": 1,
        "artifact_kind": "EVENT_RICH_FINAL_CONSENSUS_LABEL_LOCK_V1",
        "status": "FINAL_EXTERNAL_ENDPOINT_LABELS_LOCKED_BEFORE_ANY_B_E_INTERNAL_PRODUCT",
        "phase": args.phase,
        "event_protocol_identity_sha256": contract.EVENT_PROTOCOL_IDENTITY,
        "phase_plan_identity_sha256": phase_plan["identity_sha256"],
        "anchor_plan_identity_sha256": (
            phase_plan["identity_sha256"]
            if args.phase == "confirmation"
            else None
        ),
        "review_pack_identity_sha256": review_pack_manifest["identity_sha256"],
        "review_lock_identities": {slot: reviews[slot][0]["identity_sha256"] for slot in contract.REVIEWER_SLOTS},
        "adjudication_pack_identity_sha256": adjudication_pack_manifest["identity_sha256"],
        "adjudication_lock_identities": {slot: adjudications[slot][0]["identity_sha256"] for slot in contract.ADJUDICATOR_SLOTS},
        "row_count": len(final_rows),
        "candidate_scores_features_trajectories_embeddings_thresholds_or_ranks_opened": False,
        "external_visual_evaluation_only": True,
        "class_selection_role": "evaluation cohort event enrichment, not a deployed method or per-sample detector",
        "endpoint_review_or_consensus_used_as_B_E_method_input": False,
        "three_of_three_never_downgraded": True,
        "single_adjudicator_changed_final_label": False,
        **source_binding(),
    }

    def builder(root: Path) -> None:
        contract.write_csv(root / "consensus_rows.csv", CONSENSUS_FIELDS, final_rows)
        contract.write_csv(
            root / "evaluation_labels.csv",
            EVALUATION_LABEL_FIELDS,
            (
                {
                    "phase": row["phase"],
                    "global_seed": row["global_seed"],
                    "class_id": row["class_id"],
                    "final_severity": row["final_label"],
                    "blur_component": (
                        "1" if row["blur_or_soft_fusion_positive"] == "true" else "0"
                    ),
                }
                for row in final_rows
            ),
        )
        contract.write_json(root / "aggregate_counts.json", counts)
        contract.write_json(
            root / "reviewer_agreement.json",
            {
                "phase": args.phase,
                "binary_definition": "severity >=2",
                "per_reviewer_prevalence": {
                    slot: sum(values) / len(values) if values else 0.0
                    for slot, values in per_reviewer_binary.items()
                },
                "pairwise": pairwise,
            },
        )
        contract.write_json(
            root / "label_access_receipt.json",
            {
                "labels_locked_before_any_B_or_E_product_is_opened": True,
                "inputs_opened": ["endpoint cohort", "three review locks", "two adjudication locks"],
                "inputs_not_opened": ["B/E internal products", "trajectory features", "thresholds/alerts", "embeddings", "Inception/DINO/FID/CLIP", "class enrichment ranks"],
                "visual_labels_used_for_evaluation_cohort_enrichment": True,
                "endpoint_review_or_consensus_used_as_B_E_method_input": False,
            },
        )

    return contract.publish_artifact(args.output, identity=identity, builder=builder)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)

    cohort = sub.add_parser("lock-cohort")
    cohort.add_argument("--phase", choices=contract.PHASES, required=True)
    cohort.add_argument("--phase-plan", type=Path, required=True)
    cohort.add_argument("--endpoint-index", type=Path, required=True)
    cohort.add_argument("--source-receipt", type=Path, required=True)
    cohort.add_argument("--output", type=Path, required=True)
    cohort.set_defaults(func=lock_cohort)

    review_pack = sub.add_parser("build-review-pack")
    review_pack.add_argument("--phase", choices=contract.PHASES, required=True)
    review_pack.add_argument("--cohort-lock", type=Path, required=True)
    review_pack.add_argument("--qualification-evaluation", type=Path, required=True)
    review_pack.add_argument("--output", type=Path, required=True)
    review_pack.set_defaults(func=build_review_pack)

    review = sub.add_parser("lock-review")
    review.add_argument("--phase", choices=contract.PHASES, required=True)
    review.add_argument("--review-pack", type=Path, required=True)
    review.add_argument("--slot", choices=contract.REVIEWER_SLOTS, required=True)
    review.add_argument("--completed-form", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True)
    review.set_defaults(func=lock_review)

    adjudication_pack = sub.add_parser("build-adjudication-pack")
    adjudication_pack.add_argument("--phase", choices=contract.PHASES, required=True)
    adjudication_pack.add_argument("--review-pack", type=Path, required=True)
    adjudication_pack.add_argument("--cohort-lock", type=Path, required=True)
    adjudication_pack.add_argument("--qualification-evaluation", type=Path, required=True)
    for slot in contract.REVIEWER_SLOTS:
        adjudication_pack.add_argument(f"--{slot.replace('_', '-')}", dest=slot, type=Path, required=True)
    adjudication_pack.add_argument("--output", type=Path, required=True)
    adjudication_pack.set_defaults(func=build_adjudication_pack)

    adjudication = sub.add_parser("lock-adjudication")
    adjudication.add_argument("--phase", choices=contract.PHASES, required=True)
    adjudication.add_argument("--adjudication-pack", type=Path, required=True)
    adjudication.add_argument("--slot", choices=contract.ADJUDICATOR_SLOTS, required=True)
    adjudication.add_argument("--completed-form", type=Path, required=True)
    adjudication.add_argument("--output", type=Path, required=True)
    adjudication.set_defaults(func=lock_adjudication)

    consensus = sub.add_parser("lock-consensus")
    consensus.add_argument("--phase", choices=contract.PHASES, required=True)
    consensus.add_argument("--review-pack", type=Path, required=True)
    consensus.add_argument("--adjudication-pack", type=Path, required=True)
    for slot in (*contract.REVIEWER_SLOTS, *contract.ADJUDICATOR_SLOTS):
        consensus.add_argument(f"--{slot.replace('_', '-')}", dest=slot, type=Path, required=True)
    consensus.add_argument("--output", type=Path, required=True)
    consensus.set_defaults(func=lock_consensus)
    return root


def main() -> None:
    args = parser().parse_args()
    path = args.func(args)
    print(json.dumps({"output": str(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
