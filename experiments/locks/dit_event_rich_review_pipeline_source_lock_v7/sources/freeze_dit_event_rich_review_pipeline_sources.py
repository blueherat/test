#!/usr/bin/env python3
"""Freeze event-rich visual-label pipeline sources, schemas, and blank forms.

The resulting artifact contains infrastructure only.  It deliberately does
not claim that any expert, reviewer, or adjudicator has participated, and it
keeps ``ready_for_real_sampling=false`` until the non-automatable ratification
and hidden qualification inputs have actually been supplied and passed.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import dit_event_rich_review_contract as contract
import selftest_dit_event_rich_review_pipeline as selftest


SOURCE_PATHS = {
    "dit_event_rich_review_contract.py": Path(contract.__file__).resolve(),
    "prepare_dit_event_rich_label_quality.py": contract.ROOT / "experiments/prepare_dit_event_rich_label_quality.py",
    "run_dit_event_rich_blind_label_pipeline.py": contract.ROOT / "experiments/run_dit_event_rich_blind_label_pipeline.py",
    "selftest_dit_event_rich_review_pipeline.py": contract.ROOT / "experiments/selftest_dit_event_rich_review_pipeline.py",
    "freeze_dit_event_rich_review_pipeline_sources.py": Path(__file__).resolve(),
}
ENDPOINT_SOURCE_LOCK = contract.ROOT / "experiments/locks/dit_event_rich_endpoint_sampling_source_lock_v1"


def validate_endpoint_source_lock() -> dict[str, Any]:
    root = contract.require_directory(ENDPOINT_SOURCE_LOCK, "endpoint sampling source lock")
    sampling_path = contract.require_regular(root / "sampling_protocol.json", "endpoint sampling protocol")
    manifest_path = contract.require_regular(root / "manifest.json", "endpoint source-lock manifest")
    completion_path = contract.require_regular(root / "completion.json", "endpoint source-lock completion")
    sampling = contract.load_json(sampling_path)
    manifest = contract.load_json(manifest_path)
    completion = contract.load_json(completion_path)
    sampling_identity = contract.canonical_sha256(contract.without_identity(sampling))
    if (
        sampling.get("identity_sha256") != sampling_identity
        or sampling.get("status") != "FROZEN_BEFORE_EVENT_RICH_ENDPOINT_GPU_SAMPLING"
        or sampling.get("event_protocol", {}).get("identity_sha256") != contract.EVENT_PROTOCOL_IDENTITY
        or manifest.get("status") != "complete"
        or manifest.get("sampling_protocol_identity_sha256") != sampling_identity
        or manifest.get("event_protocol_identity_sha256") != contract.EVENT_PROTOCOL_IDENTITY
        or completion.get("complete") is not True
        or completion.get("sampling_protocol_identity_sha256") != sampling_identity
        or completion.get("event_protocol_identity_sha256") != contract.EVENT_PROTOCOL_IDENTITY
        or completion.get("sampling_protocol_file_sha256") != contract.sha256_file(sampling_path)
        or completion.get("manifest_file_sha256") != contract.sha256_file(manifest_path)
    ):
        raise RuntimeError("endpoint sampling source lock failed identity validation")
    return {
        "path": str(root),
        "sampling_protocol_identity_sha256": sampling_identity,
        "sampling_protocol_file_sha256": contract.sha256_file(sampling_path),
        "manifest_identity_sha256": manifest.get("identity_sha256"),
        "manifest_file_sha256": contract.sha256_file(manifest_path),
        "completion_file_sha256": contract.sha256_file(completion_path),
    }


def source_snapshots() -> dict[str, Any]:
    records = {}
    for basename, path in SOURCE_PATHS.items():
        path = contract.require_regular(path, f"review pipeline source {basename}")
        records[basename] = {"live_path_at_freeze": str(path), "sha256": contract.sha256_file(path)}
    return records


def schemas() -> dict[str, Any]:
    return {
        "visible_anchor_ratification": {
            "format": "CSV",
            "exact_columns_ordered": list(contract.VISIBLE_RATIFICATION_FIELDS),
            "exact_rows": 20,
            "completed_rule": "both distinct expert role tokens independently ratify every frozen proposal; any rejection stops and requires a protocol revision",
        },
        "hidden_item_index": {
            "format": "CSV",
            "exact_columns_ordered": list(contract.HIDDEN_ITEM_FIELDS),
            "exact_rows_per_form": 60,
            "image_contract": "60 unique actual RGB 256x256 endpoint PNGs with byte and pixel hashes",
            "forbidden": "any label, model score, metric, feature, trajectory, embedding, rank, threshold or alert column",
        },
        "hidden_expert_label": {
            "format": "CSV",
            "exact_columns_ordered": list(contract.EXPERT_LABEL_FIELDS),
            "curation": "two independent expert forms cover all 60; third independent resolver covers exactly curator disagreements",
        },
        "qualification_response": {
            "format": "CSV",
            "exact_columns_ordered": list(contract.QUALIFICATION_RESPONSE_FIELDS),
            "role_slots": list(contract.ROLE_SLOTS),
            "exact_rows_per_role": 60,
        },
        "endpoint_cohort_input": {
            "format": "CSV",
            "exact_columns_ordered": [
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
            ],
            "axis": "exact frozen phase Cartesian product; no missing, duplicate, extra, or reordered row",
        },
        "production_review_response": {
            "format": "CSV",
            "exact_columns_ordered": list(contract.REVIEW_RESPONSE_FIELDS),
            "roles": list(contract.REVIEWER_SLOTS),
        },
        "blind_adjudication_response": {
            "format": "CSV",
            "exact_columns_ordered": list(contract.ADJUDICATION_RESPONSE_FIELDS),
            "roles": list(contract.ADJUDICATOR_SLOTS),
            "decisions": ["clear_bad", "mild_or_not_clear_bad"],
        },
        "final_evaluation_labels": {
            "format": "CSV",
            "exact_columns_ordered": [
                "phase",
                "global_seed",
                "class_id",
                "final_severity",
                "blur_component",
            ],
            "final_severity": ["clean_good", "mild_or_disputed", "clear_bad"],
            "blur_component": ["0", "1"],
            "compatibility": "exact frozen select_dit_event_rich_classes.py input; blur_component=1 only for final clear_bad",
        },
    }


def review_contract(event_binding: dict[str, Any], endpoint_binding: dict[str, Any]) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": 1,
        "status": "INFRASTRUCTURE_FROZEN_NO_REAL_LABEL_ROLES_COMPLETED",
        "canonical_source_lock_version": 7,
        "superseded_source_locks": [
            {
                "version": 1,
                "identity_sha256": "d684a6a2a1b128bf043d5d98fd12e6f93f2ff5cd0d77d6d5ab698650ebdfdbf2",
                "real_results_present": False,
                "reason": "initial infrastructure draft lacked the final Stage-A/Stage-B evaluation-label interface",
            },
            {
                "version": 2,
                "identity_sha256": "afdae9dad49e3da78dc1cb19424d3b88afb053d505ae59d76554d1ae5089ba04",
                "real_results_present": False,
                "reason": "per-reviewer shuffled mappings were not canonicalized before the three review axes were compared",
            },
            {
                "version": 3,
                "identity_sha256": "c3131fb4a549a45b9640af784252d1397e3e1af7344d5782d13b4d85d696dd9e",
                "real_results_present": False,
                "reason": "qualified-role validation depended on JSON dictionary order and confirmation aggregate/anchor-plan bindings were incomplete",
            },
            {
                "version": 4,
                "identity_sha256": "8be1446b48831f2716d3afdc6cb3b0b591f9073ca64fa11ae4ea7f2affe51dfc",
                "real_results_present": False,
                "reason": "anchor/confirmation review axes accepted a self-declared wrapper rather than the frozen selector output and exported blur_component with an incompatible boolean encoding",
            },
            {
                "version": 5,
                "identity_sha256": "9b01cab483bb7b0fc46b761f4b5631d6467aecc5fc4f796fb4ab557587db888d",
                "real_results_present": False,
                "reason": "the normalized cohort retained two provenance fields not declared by the isolated reviewer custody-mapping schema, so pack construction failed closed before any delivery",
            },
            {
                "version": 6,
                "identity_sha256": "0eff7deba4d20d79d125321e952bcce6ba1788a37896d303c63ace06c3eee6ed",
                "real_results_present": False,
                "reason": "cohort PNG hashes were checked but the per-pair endpoint hashes and identities were not yet reconciled against the immutable source pool receipt list",
            },
        ],
        "event_protocol_identity_sha256": contract.EVENT_PROTOCOL_IDENTITY,
        "event_protocol": event_binding,
        "endpoint_sampling_source_lock": endpoint_binding,
        "real_expert_or_reviewer_results_present": False,
        "role_language": {
            "allowed": ["expert", "reviewer", "adjudicator", "qualified external blind-review service"],
            "does_not_assert_human_participation": True,
            "execution_options": "a genuinely qualified independent human team or a qualified external blind-review service may fill the forms",
            "current_execution_completed": False,
        },
        "qualification": {
            "roles_qualified_together": list(contract.ROLE_SLOTS),
            "primary_and_reserve_hidden_forms": "each 60 items; disjoint endpoint pixels; 15 clean, 15 mild, 15 clear blur/fusion, 15 clear topology/attachment",
            "every_role_gold_recall_minimum": 0.8,
            "every_role_gold_specificity_minimum": 0.8,
            "every_of_10_role_pairs_positive_agreement_minimum": 0.6,
            "every_of_10_role_pairs_binary_cohen_kappa_minimum": 0.5,
            "failure": "STOP before formal screen release; consumed form cannot be reused; replace/retrain failed roles and run complete panel on pre-frozen disjoint reserve",
        },
        "production": {
            "phases": list(contract.PHASES),
            "reviewers": 3,
            "adjudicators": 2,
            "audit_population": "union of every endpoint with any reviewer severity>=2 plus an equal frozen SHA256-sort simple-random sample of zero-positive decoys",
            "unanimous_3of3": "never downgradable",
            "raw_2of3": "downgrade only if both adjudicators independently choose mild_or_not_clear_bad",
            "raw_nonmajority_and_decoy": "promote only if both adjudicators independently choose clear_bad",
            "single_adjudicator_change": False,
            "all_applied_changes_require_two_component_and_written_reason_records": True,
        },
        "blinding": {
            "reviewers_cannot_see": ["candidate B/C", "class rank", "trajectory", "score", "metric", "threshold", "alert", "embedding", "other votes"],
            "adjudicators_cannot_see": ["reviewer identity", "reviewer votes/counts", "trigger versus decoy", "candidate B/C", "class rank", "trajectory", "score", "metric", "embedding", "each other"],
            "delivery_isolation": "send only the named role's delivery subdirectory",
        },
        "immutability": {
            "no_overwrite": True,
            "file_and_artifact_hashes": True,
            "exact_axes": True,
            "duplicate_missing_extra_rows_fail": True,
            "unexpected_or_poison_columns_fail": True,
        },
        "schemas": schemas(),
        "ready_for_real_sampling": False,
        "remaining_non_automatable_inputs": [
            "two genuinely independent qualified experts must ratify all 20 visible anchors",
            "independent expert curators/resolver must create actual disjoint primary and reserve hidden-gold forms",
            "five genuinely independent role holders must complete a hidden qualification form and all gates must pass",
            "no formal endpoint image may be released before the qualification PASS lock exists",
        ],
    }
    value["identity_sha256"] = contract.canonical_sha256(value)
    return value


def write_blank_templates(root: Path, protocol: dict[str, Any]) -> None:
    templates = root / "templates"
    visible = templates / "visible_anchor_ratification"
    hidden = templates / "hidden_qualification_and_reserve"
    production = templates / "production"
    visible.mkdir(parents=True)
    hidden.mkdir()
    production.mkdir()
    catalog = protocol["label_system"]["instructional_anchor_catalog"]
    base_rows = [
        {
            "anchor_id": row["anchor_id"],
            "image_sha256": row["sha256"],
            "proposed_severity": row["severity"],
            "proposed_component_group": row["component_group"],
            "proposed_reason": row["reason"],
            "decision": "",
            "correction_severity": "",
            "correction_component_group": "",
            "correction_reason": "",
            "expert_role_token": "",
            "independence_attestation": "",
        }
        for row in catalog
    ]
    for index in (1, 2):
        contract.write_csv(
            visible / f"expert_{index}_ratification_template.csv",
            contract.VISIBLE_RATIFICATION_FIELDS,
            base_rows,
        )
    contract.write_csv(hidden / "item_index_template.csv", contract.HIDDEN_ITEM_FIELDS, ())
    for name in ("curator_1", "curator_2", "resolver"):
        contract.write_csv(hidden / f"{name}_template.csv", contract.EXPERT_LABEL_FIELDS, ())
    for slot in contract.ROLE_SLOTS:
        contract.write_csv(
            hidden / f"{slot}_qualification_response_template.csv",
            contract.QUALIFICATION_RESPONSE_FIELDS,
            (),
        )
    contract.write_csv(
        production / "endpoint_cohort_index_template.csv",
        schemas()["endpoint_cohort_input"]["exact_columns_ordered"],
        (),
    )
    contract.write_csv(production / "review_response_template.csv", contract.REVIEW_RESPONSE_FIELDS, ())
    contract.write_csv(production / "adjudication_response_template.csv", contract.ADJUDICATION_RESPONSE_FIELDS, ())
    contract.write_json(
        production / "phase_plan_schemas.json",
        {
            "discovery": {
                "kind": "static exact 84-class roster x seeds 1000..1011 wrapper",
                "exact_fields": [
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
                ],
            },
            "anchor": {
                "kind": "authoritative select_dit_event_rich_classes.py rank output",
                "status": "SCREEN_DISCOVERY_CLASSES_SELECTED_BEFORE_ANCHOR_SAMPLING",
                "axis": "union_selected_classes (6..12, frozen roster order) x seeds 1012..1035",
            },
            "confirmation": {
                "kind": "authoritative select_dit_event_rich_classes.py anchor output",
                "status": "PROSPECTIVE_TRACE_PLAN_LOCKED_AFTER_INDEPENDENT_ENDPOINT_ANCHOR",
                "axis": "nonempty active_union_classes, exactly the roster-ordered union of GO candidate six-class scopes, x seeds 1200..1327",
                "manifest_binding": "consensus identity records this file's identity_sha256 as anchor_plan_identity_sha256",
            },
            "identity_rule": "all plan identities are canonical SHA256 over all other exact fields",
            "no_example_values": "intentionally omitted to avoid creating a fake phase plan",
        },
    )
    contract.write_json(
        production / "endpoint_source_receipt_schema.json",
        {
            "exact_fields": [
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
                "labels_reviews_metrics_features_embeddings_or_scores_opened_for_sampling",
            ],
            "no_example_values": "intentionally omitted to avoid fabricating a completed sampler receipt",
        },
    )


def freeze(args: argparse.Namespace) -> Path:
    protocol, event_binding = contract.validate_event_protocol_lock(args.event_protocol_lock)
    endpoint_binding = validate_endpoint_source_lock()
    snapshots = source_snapshots()
    test_receipt = selftest.run_tests()
    frozen_contract = review_contract(event_binding, endpoint_binding)
    identity = {
        "schema_version": 1,
        "artifact_kind": contract.REVIEW_SOURCE_LOCK_KIND,
        "status": "SOURCE_SCHEMAS_AND_BLANK_FORMS_FROZEN_NO_REAL_REVIEWS",
        "canonical_source_lock_version": 7,
        "supersedes_source_lock": "dit_event_rich_review_pipeline_source_lock_v6",
        "superseded_source_lock_identities": [
            "d684a6a2a1b128bf043d5d98fd12e6f93f2ff5cd0d77d6d5ab698650ebdfdbf2",
            "afdae9dad49e3da78dc1cb19424d3b88afb053d505ae59d76554d1ae5089ba04",
            "c3131fb4a549a45b9640af784252d1397e3e1af7344d5782d13b4d85d696dd9e",
            "8be1446b48831f2716d3afdc6cb3b0b591f9073ca64fa11ae4ea7f2affe51dfc",
            "9b01cab483bb7b0fc46b761f4b5631d6467aecc5fc4f796fb4ab557587db888d",
            "0eff7deba4d20d79d125321e952bcce6ba1788a37896d303c63ace06c3eee6ed",
        ],
        "event_protocol_identity_sha256": contract.EVENT_PROTOCOL_IDENTITY,
        "endpoint_sampling_protocol_identity_sha256": endpoint_binding["sampling_protocol_identity_sha256"],
        "review_contract_identity_sha256": frozen_contract["identity_sha256"],
        "source_snapshots": snapshots,
        "selftest_status": test_receipt["status"],
        "selftest_count": test_receipt["test_count"],
        "real_expert_reviewer_or_adjudicator_results_present": False,
        "ready_for_real_sampling": False,
    }

    def builder(root: Path) -> None:
        sources = root / "sources"
        sources.mkdir()
        for basename, path in SOURCE_PATHS.items():
            shutil.copyfile(path, sources / basename)
        shutil.copyfile(args.event_protocol_lock / "protocol.json", root / "event_protocol.json")
        contract.write_json(root / "review_contract.json", frozen_contract)
        contract.write_json(root / "schemas.json", schemas())
        contract.write_json(root / "selftest_receipt.json", test_receipt)
        write_blank_templates(root, protocol)
        contract.write_json(
            root / "CURRENT_EXECUTION_STATUS.json",
            {
                "infrastructure_frozen": True,
                "real_expert_ratification_completed": False,
                "hidden_gold_primary_completed": False,
                "hidden_gold_reserve_completed": False,
                "panel_qualification_completed": False,
                "production_review_or_adjudication_completed": False,
                "real_sampling_authorized_by_this_artifact": False,
            },
        )
    return contract.publish_artifact(args.output, identity=identity, builder=builder)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-protocol-lock", type=Path, default=contract.EVENT_PROTOCOL_LOCK)
    parser.add_argument("--output", type=Path, default=contract.REVIEW_SOURCE_LOCK)
    args = parser.parse_args()
    output = freeze(args)
    print(json.dumps({"output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
