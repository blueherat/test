#!/usr/bin/env python3
"""Freeze corrected scientific v4.1: one blur-enriched population, B/E.

The lock is deliberately non-executable.  It supersedes both the unused B/C v3
scientific design and the immutable-but-superseded first v4 lock.  The latter
inherited one legacy C-endpoint sentence and audited only the old endpoint
output prefix.  v4.1 corrects both issues before any real event-screen sample
exists, binds the independent blur-focused operational e-process method lock,
and freezes the selection, axes, gates, statistics, falsification rules, and
authorization boundary.  It never opens an endpoint image, review row, trace,
score, feature, embedding, or external quality metric.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/locks/dit_event_rich_confirmation_protocol_lock_v4_1"
V1_LOCK = ROOT / "experiments/locks/dit_event_rich_confirmation_protocol_lock_v1"
V2_LOCK = ROOT / "experiments/locks/dit_event_rich_confirmation_protocol_lock_v2"
V3_LOCK = ROOT / "experiments/locks/dit_event_rich_confirmation_protocol_lock_v3"
V4_SUPERSEDED_LOCK = ROOT / "experiments/locks/dit_event_rich_confirmation_protocol_lock_v4"
METHOD_LOCK = ROOT / "experiments/locks/dit_blur_focused_eprocess_protocol_lock_v1"
SELECTOR = ROOT / "experiments/select_dit_event_rich_blur_classes_v4.py"
SELFTEST = ROOT / "experiments/selftest_dit_event_rich_scientific_v4.py"

EXPECTED_HISTORY = {
    1: {
        "protocol": "c98b66841f0c31c695558da2121a706381ffb05f9fc6557046e44bbd86a9305e",
        "manifest": "c9d138ee0cff33d661000e264e8c0f09419990063656950de854c17eaffa3dbf",
    },
    2: {
        "protocol": "7955fdc2b4a83fe14e738899ae941e9b27a4cf5c009d171e3eacae9cb8390ac6",
        "manifest": "72bf5f31fbd523d99a6bb625d9789c7470604819b0f93d42e7567efb7fbefec6",
    },
    3: {
        "protocol": "04e933793992e2a7ce62aa4ac66836412f3c4f221cce731f2e072da97e892dd7",
        "manifest": "0778e0ad2732256a1377d61ba7f04c6ad4f1fdca3a7fd9dec00d0b89e0247e36",
    },
}
EXPECTED_METHOD_IDENTITY = "facef0f59d1f10cde339440db3bc47dc26ca7fcef012faca01f7f2dfbb31b985"
EXPECTED_MATCHED_Q_IDENTITY = "226da1360ff5beed3b8441c3c0c147d78cf0356fe379e07f1d161364b836336c"
EXPECTED_V4_SUPERSEDED = {
    "protocol": "9d7b03278b9a87ad3436d015e0d3e723afc8c19e181e551b0760d9d341746b7a",
    "manifest": "b30e4d2ee14c7891c762a8d4db614cb28f19d8f4a2aa883e0e362e68551ee0cd",
}
SCREEN_PARENT = Path("/data/users/zhoushunyu/eqvae/cross_scale_evidence")
SCREEN_GLOBS = ["dit_event_rich_endpoint_screen*", "dit_scientific_v4_endpoint_screen*"]
CANONICAL_SCREEN_PATHS = [
    SCREEN_PARENT / "dit_event_rich_endpoint_screen_v1",
    SCREEN_PARENT / "dit_event_rich_endpoint_screen_v2",
    SCREEN_PARENT / "dit_event_rich_endpoint_screen_v4",
    SCREEN_PARENT / "dit_scientific_v4_endpoint_screen_v1",
]


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def without_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(value)
    output.pop("identity_sha256", None)
    return output


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    encoded = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def copy_regular(source: Path, destination: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise RuntimeError(f"source must be regular and non-symlink: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    with source.open("rb") as source_handle, temporary.open("xb") as target_handle:
        shutil.copyfileobj(source_handle, target_handle)
        target_handle.flush()
        os.fsync(target_handle.fileno())
    os.replace(temporary, destination)


def records(root: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {"manifest.json", "completion.json"}:
            continue
        result.append(
            {
                "name": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return result


def validate_event_protocol_lock(
    root: Path, *, version: int, exact_tree: bool
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"event protocol v{version} lock is missing or indirect")
    protocol_path = root / "protocol.json"
    manifest_path = root / "manifest.json"
    completion_path = root / "completion.json"
    protocol = load_json(protocol_path)
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    expected = EXPECTED_HISTORY[version]
    if (
        canonical_sha256(without_identity(protocol)) != protocol.get("identity_sha256")
        or protocol.get("identity_sha256") != expected["protocol"]
        or canonical_sha256(without_identity(manifest)) != manifest.get("identity_sha256")
        or manifest.get("identity_sha256") != expected["manifest"]
        or manifest.get("status") != "complete"
        or manifest.get("protocol_identity_sha256") != expected["protocol"]
        or completion.get("complete") is not True
        or completion.get("protocol_identity_sha256") != expected["protocol"]
        or completion.get("protocol_file_sha256") != sha256_file(protocol_path)
        or completion.get("manifest_identity_sha256") != expected["manifest"]
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
    ):
        raise RuntimeError(f"event protocol v{version} identity/receipt mismatch")
    if exact_tree:
        listed = manifest.get("files")
        if not isinstance(listed, list) or not all(isinstance(row, dict) for row in listed):
            raise RuntimeError("v3 manifest file list is malformed")
        by_name = {row.get("name"): row for row in listed}
        if len(by_name) != len(listed):
            raise RuntimeError("v3 manifest contains duplicate members")
        actual = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path.name not in {"manifest.json", "completion.json"}
        }
        if actual != set(by_name):
            raise RuntimeError("v3 exact member tree changed")
        for relative, record in by_name.items():
            path = root / str(relative)
            if path.is_symlink() or record.get("bytes") != path.stat().st_size:
                raise RuntimeError(f"v3 member changed: {relative}")
            if record.get("sha256") != sha256_file(path):
                raise RuntimeError(f"v3 member digest changed: {relative}")
    return protocol, manifest, completion


def validate_superseded_v4_lock() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate, but never mutate or execute, the first immutable v4 lock."""
    root = V4_SUPERSEDED_LOCK
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError("superseded v4 lock is missing or indirect")
    protocol_path = root / "protocol.json"
    manifest_path = root / "manifest.json"
    completion_path = root / "completion.json"
    protocol = load_json(protocol_path)
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    if (
        canonical_sha256(without_identity(protocol)) != protocol.get("identity_sha256")
        or protocol.get("identity_sha256") != EXPECTED_V4_SUPERSEDED["protocol"]
        or canonical_sha256(without_identity(manifest)) != manifest.get("identity_sha256")
        or manifest.get("identity_sha256") != EXPECTED_V4_SUPERSEDED["manifest"]
        or manifest.get("protocol_identity_sha256") != EXPECTED_V4_SUPERSEDED["protocol"]
        or manifest.get("ready_for_real_sampling") is not False
        or completion.get("complete") is not True
        or completion.get("ready_for_real_sampling") is not False
        or completion.get("protocol_identity_sha256") != EXPECTED_V4_SUPERSEDED["protocol"]
        or completion.get("protocol_file_sha256") != sha256_file(protocol_path)
        or completion.get("manifest_identity_sha256") != EXPECTED_V4_SUPERSEDED["manifest"]
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
    ):
        raise RuntimeError("superseded v4 identity/receipt mismatch")
    listed = manifest.get("files")
    if not isinstance(listed, list) or not all(isinstance(row, dict) for row in listed):
        raise RuntimeError("superseded v4 manifest files are malformed")
    actual = records(root)
    if actual != listed:
        raise RuntimeError("superseded v4 exact member tree changed")
    return protocol, manifest, completion


def validate_method_lock() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not METHOD_LOCK.is_dir() or METHOD_LOCK.is_symlink():
        raise RuntimeError("blur-focused method lock is missing or indirect")
    manifest_path = METHOD_LOCK / "manifest.json"
    completion_path = METHOD_LOCK / "completion.json"
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    protocol = load_json(METHOD_LOCK / "protocol.json")
    power = load_json(METHOD_LOCK / "matched_q_power_gate.json")
    observed_identity = manifest.get("identity_sha256")
    if (
        canonical_sha256(without_identity(manifest)) != observed_identity
        or observed_identity != EXPECTED_METHOD_IDENTITY
        or manifest.get("status") != "METHOD_PROTOCOL_FROZEN_EXECUTION_BLOCKED"
        or manifest.get("execution_ready") is not False
        or manifest.get("matched_q_power_gate_identity") != EXPECTED_MATCHED_Q_IDENTITY
        or canonical_sha256(power) != EXPECTED_MATCHED_Q_IDENTITY
        or power.get("passes") is not True
        or power.get("minimum_anytime_power", 0.0)
        < power.get("minimum_required_anytime_power", 1.0)
        or power.get("minimum_required_anytime_power") != 0.3
        or power.get("total_K_per_component") != 2.0
    ):
        raise RuntimeError("blur-focused method identity or matched-Q gate mismatch")
    listed = manifest.get("files")
    if not isinstance(listed, list) or not all(isinstance(row, dict) for row in listed):
        raise RuntimeError("method manifest files are malformed")
    expected_rows = []
    for path in sorted(METHOD_LOCK.rglob("*")):
        if path.is_file() and path.name not in {"manifest.json", "completion.json"}:
            expected_rows.append(
                {
                    "relative_path": path.relative_to(METHOD_LOCK).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    if listed != expected_rows or canonical_sha256(expected_rows) != manifest.get("files_sha256"):
        raise RuntimeError("method lock exact tree changed")
    expected_completion = {
        "schema_version": 1,
        "identity_sha256": observed_identity,
        "manifest_sha256": sha256_file(manifest_path),
        "files_sha256": manifest["files_sha256"],
        "file_count": len(expected_rows),
        "execution_ready": False,
    }
    if completion != expected_completion:
        raise RuntimeError("method lock completion mismatch")
    return protocol, manifest, completion, power


def audit_zero_real_screen() -> dict[str, Any]:
    matches = sorted(
        {
            path
            for pattern in SCREEN_GLOBS
            for path in (SCREEN_PARENT.glob(pattern) if SCREEN_PARENT.is_dir() else ())
        },
        key=str,
    )
    files: list[Path] = []
    for match in matches:
        if match.is_symlink():
            raise RuntimeError(f"event-screen path may not be a symlink: {match}")
        if match.is_file():
            files.append(match)
        elif match.is_dir():
            files.extend(path for path in match.rglob("*") if path.is_file())
    pngs = [path for path in files if path.suffix.lower() == ".png"]
    if files or pngs:
        raise RuntimeError(
            "real event-screen artifacts already exist; v4 may not supersede v3 on this pool"
        )
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "FROZEN_PRE_V4_1_OLD_AND_NEW_PREFIX_REAL_EVENT_SCREEN_COUNT_ZERO",
        "audit_date_utc": "2026-08-28",
        "filesystem_parent": str(SCREEN_PARENT),
        "globs": SCREEN_GLOBS,
        "old_and_new_launcher_prefixes_both_audited": True,
        "matched_paths": [str(path) for path in matches],
        "canonical_paths": [
            {"path": str(path), "exists": path.exists()} for path in CANONICAL_SCREEN_PATHS
        ],
        "regular_file_count": len(files),
        "endpoint_png_count": len(pngs),
        "real_event_screen_sample_count": 0,
        "endpoint_image_review_trace_score_embedding_or_label_opened": False,
        "filesystem_metadata_only": True,
    }
    result["identity_sha256"] = canonical_sha256(result)
    return result


def build_protocol(
    *,
    v3: Mapping[str, Any],
    v3_manifest: Mapping[str, Any],
    histories: Mapping[int, tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]],
    superseded_v4: tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]],
    method: Mapping[str, Any],
    method_manifest: Mapping[str, Any],
    method_completion: Mapping[str, Any],
    power: Mapping[str, Any],
    zero_audit: Mapping[str, Any],
) -> dict[str, Any]:
    old_screen = copy.deepcopy(v3["endpoint_screen"])
    for key in (
        "classes_selected_per_candidate",
        "ranking_B",
        "ranking_C",
        "maximum_selected_union_classes",
        "maximum_anchor_endpoint_count",
        "maximum_total_screen_endpoints",
    ):
        old_screen.pop(key, None)
    old_screen.update(
        {
            "axis_order": "global_seed outer; frozen class_roster order inner",
            "selected_class_count": 6,
            "one_authoritative_selected_class_set_shared_by_B_and_E": True,
            "ranking_rule": (
                "descending retained blind blur/soft-fusion clear-bad count over the 12 "
                "discovery endpoints; exact ties use frozen class_roster order; no total-bad, "
                "model score, embedding, or external representation tie-break"
            ),
            "ranking_role": (
                "construct one prospectively blur-enriched external evaluation population only; "
                "not a deployed method, sample detector, method ranking, threshold, or intervention"
            ),
            "anchor_samples_per_selected_class": 24,
            "anchor_endpoint_count": 144,
            "total_endpoint_screen_count_if_anchor_runs": 1152,
            "selection_and_anchor_exact_axes": {
                "discovery": "84 class_roster entries x global seeds 1000..1011 = 1008",
                "anchor": "the one six-class selected_classes x global seeds 1012..1035 = 144",
            },
        }
    )
    histories_json: list[dict[str, Any]] = []
    for version in (1, 2, 3):
        protocol, manifest, completion = histories[version]
        histories_json.append(
            {
                "version": version,
                "path": str((V1_LOCK, V2_LOCK, V3_LOCK)[version - 1]),
                "protocol_identity_sha256": protocol["identity_sha256"],
                "manifest_identity_sha256": manifest["identity_sha256"],
                "protocol_file_sha256": completion["protocol_file_sha256"],
                "manifest_file_sha256": completion["manifest_file_sha256"],
                "preserved_immutable": True,
            }
        )
    corrected_label_system = copy.deepcopy(v3["label_system"])
    inherited_rubric = corrected_label_system["core_rubric"]["rubric"]
    inherited_rubric["phenotype_disputed"] = (
        "final retained clear-bad without required blur/soft-fusion component consensus; "
        "excluded from the B/E blur endpoint and reported descriptively only"
    )
    old_lineage_flag = inherited_rubric.pop(
        "frozen_before_third_pool_images_are_reviewed", None
    )
    if old_lineage_flag is not True:
        raise RuntimeError("expected v3 rubric-lineage flag is missing")
    corrected_label_system["rubric_lineage_clarification_v4_1"] = {
        "source_v3_field": "frozen_before_third_pool_images_are_reviewed=true",
        "meaning": (
            "the inherited phenotype wording was frozen before the historical third-pool "
            "reviews; this records rubric lineage only and does not say that any v4/v4.1 "
            "discovery, anchor, calibration, or confirmation image exists or was reviewed"
        ),
        "v4_or_v4_1_image_or_label_opened_by_this_locker": False,
    }
    old_v4_protocol, old_v4_manifest, old_v4_completion = superseded_v4
    protocol: dict[str, Any] = {
        "schema_version": 4,
        "scientific_revision": "v4.1",
        "status": "SCIENTIFIC_V4_1_CORRECTED_FROZEN_EXECUTION_NOT_READY",
        "objective": (
            "On one prospectively blur-enriched but external-evaluation-only ImageNet class set, "
            "test whether the frozen exact B-gated operational cross-scale e-process E detects "
            "clear blur/soft-fusion failures before completion and adds information beyond the "
            "frozen internal B_persistence heuristic."
        ),
        "supersession": {
            "statement": (
                "v4.1 honestly supersedes the unused event-rich v3 B/C design and immutable first "
                "v4 lock before any real event-screen sample was generated; v1-v4 remain "
                "immutable audit history"
            ),
            "history": histories_json,
            "superseded_v4": {
                "path": str(V4_SUPERSEDED_LOCK),
                "protocol_identity_sha256": old_v4_protocol["identity_sha256"],
                "manifest_identity_sha256": old_v4_manifest["identity_sha256"],
                "protocol_file_sha256": old_v4_completion["protocol_file_sha256"],
                "manifest_file_sha256": old_v4_completion["manifest_file_sha256"],
                "ready_for_real_sampling": False,
                "real_sampling_authorized_or_executed": False,
                "corrections": [
                    "remove inherited candidate-C endpoint semantics from phenotype_disputed",
                    "clarify that the third-pool freeze field is rubric lineage only",
                    "audit both old dit_event_rich and new dit_scientific_v4 endpoint prefixes",
                ],
                "preserved_immutable": True,
            },
            "v3_B_C_execution_under_this_pool_forbidden": True,
            "v3_endpoint_source_scientific_binding_superseded": True,
            "v3_endpoint_model_axis_and_pair_keyed_rng_semantics_retained": True,
            "zero_real_screen_audit_identity_sha256": zero_audit["identity_sha256"],
            "real_event_screen_samples_at_v4_freeze": 0,
            "limitation": (
                "the six-class scope is selected by external blind visual outcomes and therefore "
                "supports only a conditional high-risk-population claim, never a random-ImageNet "
                "or deployment-wide claim"
            ),
        },
        "method_boundary": {
            "deployable_method_inputs_are_internal_only": True,
            "internal_inputs": [
                "frozen DiT latent and current/shifted model predictions",
                "frozen VAE decoded pred_xstart drafts at nine preterminal checkpoints for B",
                "predictable B-derived gates and masks frozen by the method lock",
                "the sampler transition innovation only after the predictable Q* shift is fixed",
            ],
            "forbidden_method_ranking_threshold_trigger_or_intervention_inputs": [
                "endpoint images",
                "human, AI-model, or adjudicated visual labels",
                "FID or any batch endpoint metric",
                "Inception features or distances",
                "DINO features or distances",
                "CLIP features or distances",
                "any endpoint representation distance",
                "any learned quality posterior",
            ],
            "external_visual_labels_allowed_roles": [
                "define the one six-class external evaluation population by discovery blur-event counts",
                "make the independent blur-event anchor GO decision",
                "provide locked confirmation outcomes for external falsification of B/E",
            ],
            "external_visual_labels_never_enter": [
                "a B or E value",
                "label-free calibration",
                "a per-sample method ranking or trigger",
                "a deployed class ranking or class-conditioned method behavior",
                "a threshold",
                "Q*",
                "rollback or any intervention decision",
            ],
            "external_representation_metrics_allowed_role": (
                "optional aggregate endpoint acceptance audit only after all labels and internal "
                "analyses are locked; never discovery selection, class/sample ranking, gate, "
                "threshold, candidate, ablation, trigger, or intervention"
            ),
            "screen_is_external_population_design_not_a_method": True,
        },
        "method_lock": {
            "path": str(METHOD_LOCK),
            "identity_sha256": method_manifest["identity_sha256"],
            "manifest_file_sha256": method_completion["manifest_sha256"],
            "protocol_file_sha256": sha256_file(METHOD_LOCK / "protocol.json"),
            "completion_file_sha256": sha256_file(METHOD_LOCK / "completion.json"),
            "matched_q_power_gate_identity_sha256": canonical_sha256(power),
            "matched_q_power_gate_file_sha256": sha256_file(
                METHOD_LOCK / "matched_q_power_gate.json"
            ),
            "exact_identity_required": EXPECTED_METHOD_IDENTITY,
            "method_definition_may_not_be_modified_by_v4": True,
        },
        "endpoint_screen": old_screen,
        "label_system": corrected_label_system,
        "selector_contract": {
            "source_path": str(SELECTOR),
            "source_sha256": sha256_file(SELECTOR),
            "selection_artifact_kind": "EVENT_RICH_BLUR_SCREEN_SELECTION_LOCK_V1",
            "selection_status": "BLUR_ENRICHED_CLASSES_SELECTED_BEFORE_ANCHOR",
            "authoritative_selected_class_field": "selected_classes",
            "selected_class_count": 6,
            "anchor_artifact_kind": "EVENT_RICH_BLUR_ANCHOR_PLAN_LOCK_V1",
            "anchor_status": "BLUR_ANCHOR_GO_DECISION_LOCKED_BEFORE_INTERNAL_TRACES",
            "authoritative_anchor_go_field": "decision.go",
            "selection_and_anchor_must_bind_protocol_identity": True,
            "exact_consensus_columns": [
                "phase", "class_id", "global_seed", "final_severity", "blur_component"
            ],
            "extra_columns_including_scores_or_embeddings_fail_closed": True,
        },
        "anchor_go_rule": {
            "endpoint": "retained blur_or_soft_fusion clear_bad",
            "minimum_blur_clear_bad": 6,
            "minimum_event_bearing_classes": 3,
            "minimum_clean_good": 60,
            "go_is_exact_conjunction_of_only_these_three_gates": True,
            "descriptive_wilson_z": 0.8416212335729143,
            "wilson_is_not_a_go_input_or_auc_power_guarantee": True,
            "if_fail": "STOP this pool before calibration or confirmation internal traces",
        },
        "dynamic_axis": {
            "selected_classes_source": "anchor_plan.selected_classes; exactly the same six for B and E",
            "calibration_seeds": list(range(1100, 1120)),
            "calibration_rows": 120,
            "confirmation_seeds": list(range(1200, 1328)),
            "confirmation_rows": 768,
            "axis_order": "global_seed outer; selected_classes order inner",
            "discovery_anchor_calibration_confirmation_seed_sets_pairwise_disjoint": True,
            "same_frozen_model_sampler_cfg_and_pair_keyed_singleton_rng": True,
            "confirmation_labels_or_scores_never_used_in_calibration": True,
        },
        "co_primary_family": {
            "family_size": 2,
            "members": [
                {
                    "id": "B_persistence",
                    "kind": "predictable_internal_heuristic_not_eprocess",
                    "definition": method["B_internal_statistic"]["pure_B_score"],
                    "orientation": "bad_high",
                    "endpoint": "blur_or_soft_fusion_clear_bad_vs_clean_good",
                    "minimum_auc": 0.75,
                },
                {
                    "id": "E_blur_gated_running_max_log",
                    "kind": "running_max_log_of_exact_operational_eprocess",
                    "definition": (
                        "max over observed checkpoints of log(0.5*E_Delta1+0.5*E_Delta4), "
                        "with E components and Q* exactly as frozen in the bound method lock"
                    ),
                    "orientation": "bad_high",
                    "endpoint": "blur_or_soft_fusion_clear_bad_vs_clean_good",
                    "minimum_auc": 0.70,
                },
            ],
            "Holm_family_exactly": ["B_persistence", "E_blur_gated_running_max_log"],
            "candidate_combination_substitution_or_refit_forbidden": True,
            "C_c3_low_jump": {
                "status": "EXCLUDED_FROM_V4_CONFIRMATORY_FAMILY",
                "may_not_enter": [
                    "class selection", "Holm", "event gate", "threshold", "claim", "intervention"
                ],
                "if_ever_computed": (
                    "separately named post-confirmation diagnostic only, with no rescue or "
                    "confirmatory interpretation"
                ),
            },
        },
        "frozen_method_details": {
            "observation_window": copy.deepcopy(method["observation_window"]),
            "B_internal_statistic": copy.deepcopy(method["B_internal_statistic"]),
            "label_free_calibration": copy.deepcopy(method["label_free_calibration"]),
            "cross_scale_components": copy.deepcopy(method["cross_scale_components"]),
            "blur_gated_operational_Q_star": copy.deepcopy(
                method["blur_gated_operational_Q_star"]
            ),
            "fixed_diagnostics_and_ablations": copy.deepcopy(
                method["fixed_diagnostics_and_ablations"]
            ),
        },
        "pre_label_E_gates": {
            "matched_Q_power_gate": {
                "identity_sha256": canonical_sha256(power),
                "draws": power["draws"],
                "minimum_required": power["minimum_required_anytime_power"],
                "observed_minimum": power["minimum_anytime_power"],
                "passes": power["passes"],
                "scope": power["scope"],
                "must_remain_passed_without_retuning": True,
            },
            "label_free_real_gate": {
                "input": (
                    "all 120 label-free calibration paths, before any confirmation endpoint "
                    "label/review is opened; identifiers, B tracks, Q* accounting only"
                ),
                "minimum_path_count": 60,
                "actual_required_complete_axis_count": 120,
                "per_each_heat_scale_fraction_paths_with_any_B_state_gate_open_at_least": 0.50,
                "per_each_heat_scale_fraction_gate_open_paths_using_total_K_at_least_1p5_at_least": 0.50,
                "K_utilization_denominator": "paths with at least one eligible B state gate open for that scale",
                "exact_scales": method["cross_scale_components"]["additive_heat_shifts"],
                "decision_is_aggregate_and_label_free": True,
                "post_label_tuning_forbidden": True,
                "if_fail": (
                    "STOP E before confirmation-label access; set E confirmatory p=1 and do not "
                    "open/join an E confirmation score product; B may continue independently"
                ),
            },
        },
        "stage_A_label_only_event_gate": {
            "positive": "retained blur_or_soft_fusion clear_bad",
            "negative": "clean_good",
            "mild_or_disputed_and_nonblur_clearbad_excluded_from_AUC": True,
            "minimum_blur_clear_bad": 15,
            "minimum_clean_good": 60,
            "minimum_comparable_classes": 3,
            "comparable_class_definition": "at least one positive and at least one clean_good",
            "same_gate_for_B_and_E": True,
            "labels_must_be_immutable_before_gate": True,
            "candidate_values_products_thresholds_and_external_representations_not_opened": True,
            "if_fail": "set both co-primary p-values to 1 and do not open either score product",
        },
        "stage_B_statistics": {
            "primary_statistic": "within-class pair-count-weighted tie-aware ROC AUC",
            "auc_formula": (
                "sum_c sum_{positive i, clean j in c}[1(score_i>score_j)+0.5*1(score_i=score_j)] "
                "/ sum_c(n_positive_c*n_clean_c)"
            ),
            "permutation": {
                "draws": 100000,
                "rng": "numpy.default_rng(PCG64(seed=2026082801))",
                "alternative": "frozen bad-high class-matched AUC greater than chance",
                "unit": "one complete six-class label/phenotype block per global seed",
                "rule": (
                    "apply one common random permutation of the 128 ordered global-seed blocks "
                    "to labels for both B and E, preserving every class count and cross-class "
                    "within-seed dependence"
                ),
                "p_value": "(1+number(permuted AUC>=observed AUC))/(1+100000)",
            },
            "multiple_testing": (
                "Holm step-down at family alpha 0.05 over exactly the two co-primary raw p-values; "
                "a pre-label-gated-off or Stage-A-gated-off candidate receives p=1"
            ),
            "B_quality_gates": [
                "class-matched AUC >= 0.75",
                "Holm-adjusted one-sided permutation p < 0.05",
                "at the frozen per-class 19th-of-20 label-free B threshold, TPR > FPR",
            ],
            "E_quality_gates": [
                "matched-Q power gate remains passed",
                "label-free real gate-open/K-utilization gate passes for each scale",
                "Stage-A event gate passes",
                "class-matched AUC >= 0.70",
                "Holm-adjusted one-sided permutation p < 0.05",
                "at the fixed alpha_e=0.10 threshold E>=10, TPR > FPR",
                "at least three blur/fusion positives cross E>=10",
                "all operational exactness/predictability/K-budget/RNG non-perturbation audits pass",
            ],
            "threshold_semantics": {
                "B": (
                    "strict B_persistence greater than the per-class 19th ascending value among "
                    "20 label-free calibration paths; heuristic overall trigger budget only"
                ),
                "E": "ever E_mix>=10, equivalently running_max_log>=log(10); alpha_e=0.10",
                "alpha_e": (
                    "overall anytime P-trigger budget for the actually implemented sampler/Q* "
                    "accounting, not clean-good conditional FPR and not an ideal heat-ratio claim"
                ),
            },
        },
        "E_incremental_and_ablation_gates": {
            "E_beyond_B": {
                "statistic": "DeltaAUC=AUC(E_blur_gated_running_max_log)-AUC(B_persistence)",
                "randomization": (
                    "100000 paired complete-global-seed-block swaps of B and E score identities, "
                    "one Bernoulli(0.5) swap shared across all six classes within each seed block"
                ),
                "rng": "numpy.default_rng(PCG64(seed=2026082811))",
                "p_value": "(1+number(permuted DeltaAUC>=observed DeltaAUC))/(1+100000)",
                "pass": "DeltaAUC>0 and one-sided paired seed-block permutation p<0.05",
                "hierarchical": "evaluate only after E passes every pre-label, event, and primary quality gate",
                "no_posthoc_combination": True,
            },
            "B_gate_ablation": {
                "ablation": "E_no_state_gate exactly as frozen in the method lock",
                "statistic": "DeltaAUC_gate=AUC(E_blur_gated)-AUC(E_no_state_gate)",
                "pass_for_B_gate_mechanism_claim": "DeltaAUC_gate>0",
                "not_co_primary_not_in_Holm": True,
                "cannot_rescue_failed_E_or_authorize_intervention_alone": True,
            },
            "fixed_operating_point_report": (
                "report B-alone and E TPR/FPR at their separately frozen approximately ten-percent "
                "overall trigger budgets; do not reinterpret either as clean-good FPR control"
            ),
        },
        "authorization_and_stop_rules": {
            "E_evidence_driven_rollback_authorization": (
                "only if every E pre-label gate, Stage-A gate, E primary quality gate, E-beyond-B "
                "incremental gate, and B-gate ablation sign gate passes may researchers proceed "
                "to a separately frozen prospective evidence-driven rollback experiment"
            ),
            "this_protocol_executes_or_validates_rollback": False,
            "B_only_pass": (
                "at most authorizes separately labeled B-specific heuristic exploration; it has "
                "no martingale, Ville, anytime-valid, likelihood-ratio, or distribution guarantee"
            ),
            "E_exactness_without_quality_or_incremental_pass": (
                "does not authorize rollback and may not be presented as image-quality evidence"
            ),
            "no_candidate_may_be_rescued_by_C_external_metric_or_posthoc_variant": True,
            "frozen_method_falsification_rules": copy.deepcopy(method["falsification_and_stop_rules"]),
        },
        "source_lock_requirements": {
            "endpoint_sampler": (
                "new non-overwriting v4-compatible source lock retaining the exact 84x12 axis, "
                "pair-keyed singleton RNG domain, frozen model/assets, no-overwrite/resume, and "
                "endpoint-only payload; it must bind this v4 identity"
            ),
            "review_pipeline": (
                "new non-overwriting v4-compatible qualification/blind-review/dual-adjudication "
                "source lock consuming exactly the one selected_classes artifact; it must bind v4"
            ),
            "dynamic_pipeline": (
                "new non-overwriting v4-compatible B/E trace, calibration, isolated-product, "
                "Stage-A/Stage-B evaluator source lock; it must bind v4 and the method lock"
            ),
            "old_v3_endpoint_review_dynamic_source_locks_do_not_authorize_v4": True,
        },
        "execution_readiness": {
            "ready_for_real_sampling": False,
            "reason": (
                "corrected scientific v4.1 is frozen first; adapted endpoint, review, and B/E dynamic source "
                "locks plus real independent reviewer qualification inputs are not yet all frozen"
            ),
            "required_before_sampling": [
                "v4-compatible endpoint source lock",
                "v4-compatible blind-review and consensus source lock",
                "v4-compatible B/E dynamic/evaluation source lock",
                "real independent visible-anchor ratification and hidden qualification/reserve inputs",
                "a separate immutable execution-authorization receipt binding all preceding identities",
            ],
            "this_lock_can_never_be_mutated_to_ready": True,
            "future_execution_authorization_must_be_new_non_overwriting_artifact": True,
        },
        "independence_and_scope": {
            "prior_seed_max": 849,
            "new_seed_min": 1000,
            "prior_classes_excluded": [207, 602, 795],
            "confirmation_claim": (
                "conditional on the external blind-label-selected six-class blur-risk population "
                "under the frozen DiT sampler"
            ),
            "universal_random_ImageNet_or_deployment_claim_forbidden": True,
            "locker_opened_endpoint_image_review_trace_score_embedding_or_label": False,
        },
    }
    protocol["identity_sha256"] = canonical_sha256(protocol)
    return protocol


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if canonical_sha256(without_identity(protocol)) != protocol.get("identity_sha256"):
        raise RuntimeError("v4.1 protocol identity mismatch")
    if (
        protocol.get("schema_version") != 4
        or protocol.get("scientific_revision") != "v4.1"
        or protocol.get("status")
        != "SCIENTIFIC_V4_1_CORRECTED_FROZEN_EXECUTION_NOT_READY"
        or protocol.get("method_lock", {}).get("identity_sha256") != EXPECTED_METHOD_IDENTITY
        or protocol.get("execution_readiness", {}).get("ready_for_real_sampling") is not False
    ):
        raise RuntimeError("v4.1 status/method/readiness changed")
    screen = protocol["endpoint_screen"]
    roster = [int(row["class_id"]) for row in screen["class_roster"]]
    if (
        len(roster) != 84
        or len(set(roster)) != 84
        or set(roster) & {207, 602, 795}
        or screen["discovery_seeds"] != list(range(1000, 1012))
        or screen["anchor_seeds"] != list(range(1012, 1036))
        or screen["discovery_endpoint_count"] != 1008
        or screen["anchor_endpoint_count"] != 144
        or screen["selected_class_count"] != 6
    ):
        raise RuntimeError("v4 endpoint/selection exact axis changed")
    dynamic = protocol["dynamic_axis"]
    if (
        dynamic["calibration_seeds"] != list(range(1100, 1120))
        or dynamic["confirmation_seeds"] != list(range(1200, 1328))
        or dynamic["calibration_rows"] != 120
        or dynamic["confirmation_rows"] != 768
    ):
        raise RuntimeError("v4 dynamic exact axis changed")
    if protocol["co_primary_family"]["Holm_family_exactly"] != [
        "B_persistence", "E_blur_gated_running_max_log"
    ]:
        raise RuntimeError("v4 co-primary family changed")
    anchor = protocol["anchor_go_rule"]
    event = protocol["stage_A_label_only_event_gate"]
    if (
        (anchor["minimum_blur_clear_bad"], anchor["minimum_event_bearing_classes"], anchor["minimum_clean_good"])
        != (6, 3, 60)
        or (event["minimum_blur_clear_bad"], event["minimum_clean_good"], event["minimum_comparable_classes"])
        != (15, 60, 3)
    ):
        raise RuntimeError("v4 event gates changed")
    forbidden = " ".join(
        protocol["method_boundary"][
            "forbidden_method_ranking_threshold_trigger_or_intervention_inputs"
        ]
    ).lower()
    for token in ("fid", "inception", "dino", "human", "visual labels"):
        if token not in forbidden:
            raise RuntimeError(f"v4 forbidden-method boundary omits {token}")


def validate_lock(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError("v4 lock is missing or indirect")
    protocol_path = root / "protocol.json"
    manifest_path = root / "manifest.json"
    completion_path = root / "completion.json"
    protocol = load_json(protocol_path)
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    validate_protocol(protocol)
    if (
        canonical_sha256(without_identity(manifest)) != manifest.get("identity_sha256")
        or manifest.get("status") != "complete"
        or manifest.get("experiment") != "dit_event_rich_confirmation_protocol_lock_v4_1"
        or manifest.get("protocol_identity_sha256") != protocol["identity_sha256"]
        or completion
        != {
            "complete": True,
            "manifest_identity_sha256": manifest.get("identity_sha256"),
            "manifest_file_sha256": sha256_file(manifest_path),
            "protocol_identity_sha256": protocol["identity_sha256"],
            "protocol_file_sha256": sha256_file(protocol_path),
            "ready_for_real_sampling": False,
        }
    ):
        raise RuntimeError("v4 lock manifest/completion mismatch")
    actual = records(root)
    if actual != manifest.get("files"):
        raise RuntimeError("v4 lock exact member tree changed")
    by_name = {row["name"]: row for row in actual}
    expected_names = {
        "pre_sampling_zero_audit.json",
        "protocol.json",
        "sources/freeze_dit_event_rich_confirmation_protocol_v4.py",
        "sources/select_dit_event_rich_blur_classes_v4.py",
        "sources/selftest_dit_event_rich_scientific_v4.py",
        "upstream/event_completion_v3.json",
        "upstream/event_manifest_v3.json",
        "upstream/event_protocol_v3.json",
        "upstream/event_completion_v4_superseded.json",
        "upstream/event_manifest_v4_superseded.json",
        "upstream/event_protocol_v4_superseded.json",
        "upstream/matched_q_power_gate.json",
        "upstream/method_completion.json",
        "upstream/method_manifest.json",
        "upstream/method_protocol.json",
    }
    if set(by_name) != expected_names:
        raise RuntimeError("v4 lock member names changed")
    zero = load_json(root / "pre_sampling_zero_audit.json")
    if (
        canonical_sha256(without_identity(zero)) != zero.get("identity_sha256")
        or zero.get("real_event_screen_sample_count") != 0
        or zero.get("endpoint_image_review_trace_score_embedding_or_label_opened") is not False
        or protocol["supersession"]["zero_real_screen_audit_identity_sha256"]
        != zero["identity_sha256"]
    ):
        raise RuntimeError("v4 pre-sampling-zero audit changed")
    if sha256_file(root / "sources/select_dit_event_rich_blur_classes_v4.py") != protocol[
        "selector_contract"
    ]["source_sha256"]:
        raise RuntimeError("v4 frozen selector binding changed")
    return protocol, manifest


def freeze(output: Path) -> Path:
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"refusing to overwrite existing v4 lock: {output}")
    for source in (Path(__file__).resolve(), SELECTOR, SELFTEST):
        if not source.is_file() or source.is_symlink():
            raise RuntimeError(f"missing regular v4 source: {source}")
    h1 = validate_event_protocol_lock(V1_LOCK, version=1, exact_tree=False)
    h2 = validate_event_protocol_lock(V2_LOCK, version=2, exact_tree=False)
    h3 = validate_event_protocol_lock(V3_LOCK, version=3, exact_tree=True)
    v4_superseded = validate_superseded_v4_lock()
    method, method_manifest, method_completion, power = validate_method_lock()
    zero = audit_zero_real_screen()
    protocol = build_protocol(
        v3=h3[0],
        v3_manifest=h3[1],
        histories={1: h1, 2: h2, 3: h3},
        superseded_v4=v4_superseded,
        method=method,
        method_manifest=method_manifest,
        method_completion=method_completion,
        power=power,
        zero_audit=zero,
    )
    validate_protocol(protocol)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        write_json(staging / "protocol.json", protocol)
        write_json(staging / "pre_sampling_zero_audit.json", zero)
        copy_regular(Path(__file__).resolve(), staging / "sources" / Path(__file__).name)
        copy_regular(SELECTOR, staging / "sources" / SELECTOR.name)
        copy_regular(SELFTEST, staging / "sources" / SELFTEST.name)
        copy_regular(V3_LOCK / "protocol.json", staging / "upstream/event_protocol_v3.json")
        copy_regular(V3_LOCK / "manifest.json", staging / "upstream/event_manifest_v3.json")
        copy_regular(V3_LOCK / "completion.json", staging / "upstream/event_completion_v3.json")
        copy_regular(
            V4_SUPERSEDED_LOCK / "protocol.json",
            staging / "upstream/event_protocol_v4_superseded.json",
        )
        copy_regular(
            V4_SUPERSEDED_LOCK / "manifest.json",
            staging / "upstream/event_manifest_v4_superseded.json",
        )
        copy_regular(
            V4_SUPERSEDED_LOCK / "completion.json",
            staging / "upstream/event_completion_v4_superseded.json",
        )
        copy_regular(METHOD_LOCK / "protocol.json", staging / "upstream/method_protocol.json")
        copy_regular(METHOD_LOCK / "manifest.json", staging / "upstream/method_manifest.json")
        copy_regular(METHOD_LOCK / "completion.json", staging / "upstream/method_completion.json")
        copy_regular(
            METHOD_LOCK / "matched_q_power_gate.json",
            staging / "upstream/matched_q_power_gate.json",
        )
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "complete",
            "experiment": "dit_event_rich_confirmation_protocol_lock_v4_1",
            "protocol_identity_sha256": protocol["identity_sha256"],
            "ready_for_real_sampling": False,
            "files": records(staging),
        }
        manifest["identity_sha256"] = canonical_sha256(manifest)
        write_json(staging / "manifest.json", manifest)
        completion = {
            "complete": True,
            "manifest_identity_sha256": manifest["identity_sha256"],
            "manifest_file_sha256": sha256_file(staging / "manifest.json"),
            "protocol_identity_sha256": protocol["identity_sha256"],
            "protocol_file_sha256": sha256_file(staging / "protocol.json"),
            "ready_for_real_sampling": False,
        }
        write_json(staging / "completion.json", completion)
        os.replace(staging, output)
        validate_lock(output)
        return output
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="dit-event-v4-lock-test-") as temporary:
        lock = freeze(Path(temporary) / "lock")
        protocol, _ = validate_lock(lock)
        assert protocol["execution_readiness"]["ready_for_real_sampling"] is False
        assert protocol["co_primary_family"]["Holm_family_exactly"] == [
            "B_persistence", "E_blur_gated_running_max_log"
        ]
    print("self-test passed: v4.1 exact identities/axes/method boundary; execution remains blocked")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    output = freeze(args.output.expanduser().absolute())
    protocol, manifest = validate_lock(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "protocol_identity_sha256": protocol["identity_sha256"],
                "manifest_identity_sha256": manifest["identity_sha256"],
                "ready_for_real_sampling": False,
                "real_event_screen_samples_at_freeze": 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
