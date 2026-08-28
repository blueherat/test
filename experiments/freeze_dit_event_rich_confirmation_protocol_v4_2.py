#!/usr/bin/env python3
"""Freeze scientific v4.2 bound to method v2.2 before any real screen.

v4.2 preserves v4.1's endpoint population, blind-label system, exact axes,
event gates, B/E co-primary family, class-matched AUC, primary permutation,
and Holm correction.  It replaces only the bound E mechanism/pre-label audit
and invalid score-identity swap comparisons.  Incremental comparisons now use
paired 128-seed cluster bootstrap lower bounds.  Existing v4.1 source locks
remain immutable but cannot execute v4.2 until independently rebound.
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
LOCK_NAME = "dit_event_rich_confirmation_protocol_lock_v4_2"
OUTPUT = ROOT / "experiments/locks" / LOCK_NAME
V4_1_LOCK = ROOT / "experiments/locks/dit_event_rich_confirmation_protocol_lock_v4_1"
METHOD_LOCK = ROOT / "experiments/locks/dit_blur_focused_eprocess_protocol_lock_v2_2"
DOC = ROOT / "docs/DIT_EVENT_RICH_SCIENTIFIC_V4_2_ZH.md"
EXPECTED_V4_1_PROTOCOL_ID = "0998e0a9def75fa26fa3403f589c6d86bdeb4b96747b1ee6d035d4e92a07d5b9"
EXPECTED_V4_1_MANIFEST_ID = "17251d8d7e4f69abb88c00ba795424b4d0747352f6145fff5ac426b5368c4430"
EXPECTED_METHOD_ID = "cc4dc5e7c06c25f4d8567a42fb4f0387097a6296c587543830bfeaa4771f6921"
EXPECTED_METHOD_POWER_ID = "ae284448a324349488ab1be3962502d5450d006a64722bb717f5199903c6e6b2"
EXPECTED_METHOD_NULL_ID = "4b69c132d39a70e615fc60ec12709daff670f15409a61c4f12e543f43fb7162c"
SCREEN_PARENT = Path("/data/users/zhoushunyu/eqvae/cross_scale_evidence")
SCREEN_GLOBS = ("dit_event_rich_endpoint_screen*", "dit_scientific_v4_endpoint_screen*")


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"expected regular JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def without_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("identity_sha256", None)
    return result


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
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
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.parent == root and path.name in {"manifest.json", "completion.json"}:
            continue
        if path.is_symlink():
            raise RuntimeError(f"lock member may not be a symlink: {path}")
        rows.append(
            {
                "name": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def validate_v4_1() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol_path = V4_1_LOCK / "protocol.json"
    manifest_path = V4_1_LOCK / "manifest.json"
    completion_path = V4_1_LOCK / "completion.json"
    protocol = load_json(protocol_path)
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    if (
        canonical_sha256(without_identity(protocol)) != protocol.get("identity_sha256")
        or protocol.get("identity_sha256") != EXPECTED_V4_1_PROTOCOL_ID
        or canonical_sha256(without_identity(manifest)) != manifest.get("identity_sha256")
        or manifest.get("identity_sha256") != EXPECTED_V4_1_MANIFEST_ID
        or manifest.get("protocol_identity_sha256") != EXPECTED_V4_1_PROTOCOL_ID
        or completion.get("manifest_identity_sha256") != EXPECTED_V4_1_MANIFEST_ID
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
        or completion.get("protocol_identity_sha256") != EXPECTED_V4_1_PROTOCOL_ID
        or completion.get("protocol_file_sha256") != sha256_file(protocol_path)
        or completion.get("ready_for_real_sampling") is not False
        or records(V4_1_LOCK) != manifest.get("files")
    ):
        raise RuntimeError("immutable scientific v4.1 lock changed")
    return protocol, manifest, completion


def validate_method() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol_path = METHOD_LOCK / "protocol.json"
    manifest_path = METHOD_LOCK / "manifest.json"
    completion_path = METHOD_LOCK / "completion.json"
    protocol = load_json(protocol_path)
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    power = load_json(METHOD_LOCK / "matched_q_conditional_power_gate.json")
    null = load_json(METHOD_LOCK / "adaptive_predictable_null_audit.json")
    identity = dict(manifest)
    observed = identity.pop("identity_sha256", None)
    if (
        canonical_sha256(identity) != observed
        or observed != EXPECTED_METHOD_ID
        or manifest.get("execution_ready") is not False
        or manifest.get("matched_q_power_gate_identity") != EXPECTED_METHOD_POWER_ID
        or manifest.get("adaptive_null_audit_identity") != EXPECTED_METHOD_NULL_ID
        or canonical_sha256(power) != EXPECTED_METHOD_POWER_ID
        or canonical_sha256(null) != EXPECTED_METHOD_NULL_ID
        or power.get("passes") is not True
        or power.get("dependence_robust_conditional_terminal_power_lower_bound", 0.0) < 0.30
        or null.get("passes") is not True
        or protocol.get("scientific_revision") != "v2.2"
        or completion.get("identity_sha256") != EXPECTED_METHOD_ID
        or completion.get("manifest_sha256") != sha256_file(manifest_path)
    ):
        raise RuntimeError("method v2.2 identity/gates changed")
    listed = manifest.get("files")
    expected = [
        {
            "relative_path": row["name"],
            "bytes": row["bytes"],
            "sha256": row["sha256"],
        }
        for row in records(METHOD_LOCK)
    ]
    if listed != expected or canonical_sha256(expected) != manifest.get("files_sha256"):
        raise RuntimeError("method v2.2 exact member tree changed")
    return protocol, manifest, completion, power, null


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
            raise RuntimeError(f"event-screen path may not be symlink: {match}")
        if match.is_file():
            files.append(match)
        elif match.is_dir():
            files.extend(path for path in match.rglob("*") if path.is_file())
    if files:
        raise RuntimeError("real screen artifacts exist; v4.2 cannot supersede v4.1")
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "FROZEN_PRE_V4_2_REAL_EVENT_SCREEN_COUNT_ZERO",
        "audit_date_utc": "2026-08-28",
        "filesystem_parent": str(SCREEN_PARENT),
        "globs": list(SCREEN_GLOBS),
        "matched_paths": [str(path) for path in matches],
        "regular_file_count": 0,
        "endpoint_png_count": 0,
        "real_event_screen_sample_count": 0,
        "endpoint_image_review_trace_score_embedding_or_label_opened": False,
        "filesystem_metadata_only": True,
    }
    result["identity_sha256"] = canonical_sha256(result)
    return result


def build_rebind_requirements(v4_2_protocol_id: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "ALL_V4_1_BOUND_EXECUTION_SOURCES_INCOMPATIBLE_WITH_V4_2",
        "v4_1_protocol_identity_sha256": EXPECTED_V4_1_PROTOCOL_ID,
        "v4_2_protocol_identity_sha256": v4_2_protocol_id,
        "incompatible_existing_locks": [
            {
                "role": "endpoint_sampler",
                "path": "experiments/locks/dit_scientific_v4_endpoint_sampling_source_lock_v1",
                "source_lock_identity": "48924d64b0e24caf02ea2458ef837be843e46dff2fe33731d585e0c91f67ec7d",
                "sampling_protocol_identity": "acfd7345d350a67f8974396b5799a892263609aa42e19633f2dd68e9fe33e92c",
            },
            {
                "role": "review_pipeline",
                "path": "experiments/locks/dit_scientific_v4_review_pipeline_source_lock_v2",
                "source_lock_identity": "90b0411415d00fb47be574858c0c292dbe0e61c522aaf9193bda90f721f47084",
                "inner_manifest_identity": "2b1f20e63544da77c8b4de20b84ae6f8e7abfccb172ef72bd3058887a919efa5",
                "review_contract_identity": "1f3936a0065753633fb42108cd58b5a4335b6464a27af77087db0410c09af528",
            },
            {
                "role": "dynamic_B_E_pipeline",
                "path": "any current v4.1-bound dynamic source lock",
                "reason": "must bind method v2.2 tracks, mechanics schema, G_start, and cluster bootstrap",
            },
        ],
        "source_files_requiring_new_frozen_hashes_or_contracts": [
            "select_dit_event_rich_blur_classes_v4.py output artifacts must bind v4.2",
            "selftest_dit_event_rich_scientific_v4.py must validate v4.2 method/gates",
            "run_dit_event_rich_endpoint_screen.py source lock must embed v4.2",
            "dit_event_rich_review_contract.py and review source lock must embed v4.2",
            "all DiT B/E trace, extraction, isolated-product, evaluator, and bootstrap sources must bind v4.2 and method v2.2",
        ],
        "reuse_without_new_lock_forbidden": True,
        "real_sampling_authorized": False,
    }


def _bootstrap_contract(seed: int, left: str, right: str) -> dict[str, Any]:
    return {
        "statistic": f"DeltaAUC=AUC({left})-AUC({right})",
        "draws": 100000,
        "rng": f"numpy.default_rng(PCG64(seed={seed}))",
        "cluster_unit": "one complete six-class score/outcome block for one of the 128 confirmation global seeds",
        "resampling": "sample 128 global-seed blocks with replacement; retain all six classes and paired score vectors within every selected block",
        "AUC_each_replicate": "same frozen within-class pair-count-weighted tie-aware ROC AUC",
        "zero_comparable_pair_replicate": "assign DeltaAUC=-1 conservatively rather than redraw or drop",
        "one_sided_95_percent_lower_bound": "sort 100000 DeltaAUC replicates and take zero-based index 4999 (the 5000th order statistic; no interpolation)",
        "hard_pass": "observed DeltaAUC>0 and one-sided 95% lower bound>0",
        "score_identity_swap_forbidden": "B, E, G, and ablation scores are not exchangeable treatments; swapping score identities is invalid",
    }


def build_protocol(
    *,
    old: Mapping[str, Any],
    old_manifest: Mapping[str, Any],
    old_completion: Mapping[str, Any],
    method: Mapping[str, Any],
    method_manifest: Mapping[str, Any],
    method_completion: Mapping[str, Any],
    power: Mapping[str, Any],
    null: Mapping[str, Any],
    zero: Mapping[str, Any],
) -> dict[str, Any]:
    protocol = copy.deepcopy(dict(old))
    protocol.pop("identity_sha256", None)
    protocol["scientific_revision"] = "v4.2"
    protocol["status"] = "SCIENTIFIC_V4_2_METHOD_CORRECTED_FROZEN_EXECUTION_NOT_READY"
    protocol["objective"] = (
        "On the unchanged prospectively blur-enriched external evaluation population, test "
        "whether method-v2.2's exact B-started, latched, fixed-information directional "
        "cross-scale e-process detects clear blur/soft-fusion failures before completion, "
        "adds beyond B, and adds beyond its innovation-free start schedule."
    )
    supersession = protocol["supersession"]
    supersession["statement"] = (
        "v4.2 supersedes immutable, unused v4.1 before any real screen sample; v4.1's "
        "population/label/evaluation design is retained, while method v2.2 and valid paired "
        "cluster-bootstrap incremental tests replace the v1 E mechanism and score swaps"
    )
    supersession["superseded_v4_1"] = {
        "path": str(V4_1_LOCK),
        "protocol_identity_sha256": old["identity_sha256"],
        "manifest_identity_sha256": old_manifest["identity_sha256"],
        "protocol_file_sha256": old_completion["protocol_file_sha256"],
        "manifest_file_sha256": old_completion["manifest_file_sha256"],
        "ready_for_real_sampling": False,
        "real_sampling_authorized_or_executed": False,
        "preserved_immutable": True,
        "corrections": [
            "bind blur-latched fixed-information directional method v2.2",
            "move the label-free mechanics audit from 120 threshold-fitting traces to all 768 disjoint confirmation traces before label access",
            "replace K-utilization target with start/coverage/stale-direction mechanics",
            "replace invalid score-identity swaps by paired global-seed cluster bootstrap",
            "add innovation-free G_start and exact one-shot diagnostics",
        ],
    }
    supersession["zero_real_screen_audit_identity_sha256"] = zero["identity_sha256"]
    supersession["real_event_screen_samples_at_v4_freeze"] = 0

    protocol["method_lock"] = {
        "path": str(METHOD_LOCK),
        "identity_sha256": method_manifest["identity_sha256"],
        "manifest_file_sha256": method_completion["manifest_sha256"],
        "protocol_file_sha256": sha256_file(METHOD_LOCK / "protocol.json"),
        "completion_file_sha256": sha256_file(METHOD_LOCK / "completion.json"),
        "matched_q_power_gate_identity_sha256": canonical_sha256(power),
        "matched_q_power_gate_file_sha256": sha256_file(
            METHOD_LOCK / "matched_q_conditional_power_gate.json"
        ),
        "adaptive_null_audit_identity_sha256": canonical_sha256(null),
        "adaptive_null_audit_file_sha256": sha256_file(
            METHOD_LOCK / "adaptive_predictable_null_audit.json"
        ),
        "exact_identity_required": EXPECTED_METHOD_ID,
        "method_definition_may_not_be_modified_by_v4_2": True,
    }
    members = protocol["co_primary_family"]["members"]
    members[1]["kind"] = "running_max_log_of_exact_blur_latched_directional_operational_eprocess"
    members[1]["definition"] = (
        "legacy field E_blur_gated_running_max_log, now semantically method-v2.2's "
        "B-started latch with h>=3, kappa=2/h, fixed-norm direction, last-valid fallback, "
        "and fixed 0.5/0.5 scale mixture"
    )
    protocol["co_primary_family"]["legacy_E_field_name_retained_for_schema_stability"] = True

    details = protocol["frozen_method_details"]
    details["observation_window"] = copy.deepcopy(method["observation_window"])
    details["B_internal_statistic"] = copy.deepcopy(method["B_internal_statistic"])
    details["label_free_calibration"] = copy.deepcopy(method["label_free_calibration"])
    details["cross_scale_components"] = copy.deepcopy(method["cross_scale_components"])
    details.pop("blur_gated_operational_Q_star", None)
    details["blur_latched_directional_operational_Q_star"] = copy.deepcopy(
        method["blur_latched_directional_operational_Q_star"]
    )
    details["fixed_diagnostics_and_ablations"] = copy.deepcopy(method["fixed_ablations"])

    protocol["pre_label_E_gates"] = {
        "matched_Q_conditional_power_gate": {
            "identity_sha256": canonical_sha256(power),
            "draws_per_scale_h": power["draws"],
            "allowed_h_by_scale": power["allowed_start_remaining_counts_by_scale"],
            "dependence_robust_terminal_lower_bound": power[
                "dependence_robust_conditional_terminal_power_lower_bound"
            ],
            "minimum_required": power["minimum_required_anytime_power"],
            "observed_minimum_sufficient_event_anytime_power": power["minimum_anytime_power"],
            "passes": power["passes"],
            "scope": power["scope"],
        },
        "adaptive_predictable_null_audit": {
            "identity_sha256": canonical_sha256(null),
            "component_terminal_e_means": null["component_terminal_e_means"],
            "fixed_mixture_terminal_e_mean": null["fixed_mixture_terminal_e_mean"],
            "anytime_trigger_fraction_under_P": null["anytime_trigger_fraction_under_P"],
            "passes": null["passes"],
            "quality_interpretation": False,
        },
        "confirmation_path_mechanics_gate": {
            "input": (
                "all 768 confirmation method traces, after thresholds are frozen on disjoint "
                "calibration seeds 1100..1119 and before any confirmation label/review is opened"
            ),
            "threshold_fitting_calibration_E": (
                "optional in-sample diagnostic only; operational LR exactness may remain if "
                "pre-innovation predictable, but it cannot enter fresh-rank, mechanics GO, "
                "candidate evaluation, or confirmation claims"
            ),
            "exact_confirmation_axis_count": 768,
            "per_scale_minimum_qualifying_started_paths": 12,
            "per_scale_minimum_qualifying_started_classes": 3,
            "per_scale_started_path_complete_h_step_coverage_fraction": 1.0,
            "per_scale_maximum_last_valid_fallback_fraction_among_started_steps": 0.01,
            "required_reports": [
                "positive-K step-count histogram",
                "KL max-share quantiles",
                "KL participation-ratio effective-step-count quantiles",
                "total-K quantiles",
                "fallback step/path counts and maximum consecutive fallback run",
            ],
            "labels_endpoints_external_representations_read": False,
            "quality_interpretation": False,
            "if_fail": (
                "STOP E before confirmation-label access; set E confirmatory p=1 and do not "
                "open/join E, G_start, or E-ablation confirmation products; B may continue"
            ),
            "post_label_tuning_forbidden": True,
        },
    }

    E_gates = protocol["stage_B_statistics"]["E_quality_gates"]
    protocol["stage_B_statistics"]["E_quality_gates"] = [
        "method-v2.2 matched-Q conditional power and adaptive predictable-null audits remain passed",
        "all-768 pre-label confirmation path-mechanics gate passes for each scale",
        "Stage-A event gate passes",
        "class-matched AUC >= 0.70",
        "Holm-adjusted one-sided primary permutation p < 0.05",
        "at fixed alpha_e=0.10, TPR > FPR",
        "at least three blur/fusion positives cross E_mix>=10",
        "all predictability, latch, fixed-kappa, fallback, exactness, source, and RNG non-perturbation audits pass",
    ]
    if not E_gates:
        raise RuntimeError("v4.1 E gates unexpectedly absent")
    protocol["stage_B_statistics"]["threshold_semantics"]["E"] = (
        "ever method-v2.2 E_mix>=10; P/Q* anytime budget under the implemented latch and "
        "fixed directional shifts, not clean-good FPR or ideal heat-marginal evidence"
    )

    protocol["E_incremental_and_ablation_gates"] = {
        "paired_cluster_bootstrap_common_rule": {
            "why_score_swap_removed": (
                "B, E, G_start, E_no_state_gate, and one-shot scores are nonexchangeable "
                "measurements on the same paths; score-identity permutation is invalid"
            ),
            "confirmation_global_seed_blocks": 128,
            "classes_per_block": 6,
            "draws_each_comparison": 100000,
            "hard_gate_form": "observed DeltaAUC>0 and one-sided percentile lower95>0",
        },
        "E_beyond_B": {
            **_bootstrap_contract(2026082811, "E_blur_gated_running_max_log", "B_persistence"),
            "hierarchical": "after E passes every pre-label, event, and primary quality gate",
            "required_for_E_incremental_claim_and_rollback": True,
        },
        "E_beyond_G_start_schedule": {
            **_bootstrap_contract(
                2026082813,
                "E_blur_gated_running_max_log",
                "G_start_schedule_diagnostic",
            ),
            "G_formula": (
                "0.5*1[T_Delta1 finite]*h_Delta1/5 + "
                "0.5*1[T_Delta4 finite]*h_Delta4/8"
            ),
            "meaning": "isolates innovation alignment from whether/how early the B/direction schedule starts",
            "required_for_path_LR_incremental_claim_and_rollback": True,
        },
        "B_start_beyond_no_state_gate": {
            **_bootstrap_contract(
                2026082812,
                "E_blur_gated_running_max_log",
                "E_no_state_gate_running_max_log_ablation",
            ),
            "required_for_B_start_mechanism_claim_and_rollback": True,
            "not_co_primary_not_in_Holm": True,
        },
        "multi_step_vs_one_shot": {
            **_bootstrap_contract(
                2026082814,
                "E_blur_gated_running_max_log",
                "E_first_hit_full_budget_running_max_log_ablation",
            ),
            "role": "diagnostic claim boundary; failure drops multi-step-superiority wording but cannot rescue or replace E",
            "required_for_rollback": False,
        },
        "fixed_operating_point_report": (
            "report B and E TPR/FPR at frozen thresholds; do not call alpha a clean-good FPR"
        ),
        "no_posthoc_combination_or_ablation_substitution": True,
    }

    authorization = protocol["authorization_and_stop_rules"]
    authorization["E_evidence_driven_rollback_authorization"] = (
        "only if E passes every v2.2 pre-label gate, Stage-A gate, primary quality gate, "
        "paired cluster-bootstrap E-beyond-B lower bound, E-beyond-G_start lower bound, "
        "and E-beyond-no-state-gate lower bound may a separately frozen rollback study begin"
    )
    authorization["G_start_or_ablation_can_rescue_failed_E"] = False
    authorization["frozen_method_falsification_rules"] = copy.deepcopy(
        method["falsification_and_stop_rules"]
    )

    protocol["source_lock_requirements"] = {
        "all_existing_v4_1_bound_locks_are_incompatible": True,
        "endpoint_sampler": "new non-overwriting source lock binding v4.2 while preserving the unchanged endpoint axis/RNG/payload",
        "review_pipeline": "new non-overwriting source lock binding v4.2 while preserving the unchanged physical endpoint/method firewall and reviewer design",
        "selector_and_scientific_selftest": "new frozen hashes/artifacts must bind v4.2 identity",
        "dynamic_pipeline": (
            "new lock binding method v2.2; physical B/E/G/ablation products; all-768 pre-label "
            "mechanics; and paired 128-seed cluster bootstrap evaluator"
        ),
        "execution_authorization": "new immutable receipt must bind all newly rebound identities",
        "reuse_or_manifest_relabeling_of_v4_1_locks_forbidden": True,
    }
    protocol["execution_readiness"] = {
        "ready_for_real_sampling": False,
        "reason": "scientific v4.2 is frozen, but every v4.1-bound endpoint/review/dynamic/selector/selftest source requires a new non-overwriting v4.2 lock",
        "required_before_sampling": [
            "v4.2-bound endpoint source lock",
            "v4.2-bound blind-review source lock",
            "v4.2/method-v2.2-bound dynamic and evaluator source lock",
            "real independent reviewer qualification and reserve inputs",
            "new execution-authorization receipt binding all v4.2 identities",
        ],
        "this_lock_can_never_be_mutated_to_ready": True,
        "future_execution_authorization_must_be_new_non_overwriting_artifact": True,
    }
    protocol["identity_sha256"] = canonical_sha256(protocol)
    return protocol


def validate_protocol(protocol: Mapping[str, Any], old: Mapping[str, Any]) -> None:
    if canonical_sha256(without_identity(protocol)) != protocol.get("identity_sha256"):
        raise RuntimeError("v4.2 protocol identity mismatch")
    if (
        protocol.get("scientific_revision") != "v4.2"
        or protocol.get("status")
        != "SCIENTIFIC_V4_2_METHOD_CORRECTED_FROZEN_EXECUTION_NOT_READY"
        or protocol.get("method_lock", {}).get("identity_sha256") != EXPECTED_METHOD_ID
        or protocol.get("execution_readiness", {}).get("ready_for_real_sampling") is not False
    ):
        raise RuntimeError("v4.2 method/status/readiness changed")
    # These blocks are intentionally byte-identical to v4.1.
    for key in (
        "endpoint_screen",
        "label_system",
        "selector_contract",
        "anchor_go_rule",
        "dynamic_axis",
        "stage_A_label_only_event_gate",
        "independence_and_scope",
    ):
        if protocol.get(key) != old.get(key):
            raise RuntimeError(f"v4.2 illegally changed retained evaluation block: {key}")
    if protocol["co_primary_family"]["Holm_family_exactly"] != [
        "B_persistence",
        "E_blur_gated_running_max_log",
    ]:
        raise RuntimeError("v4.2 B/E Holm family changed")
    old_family = old["co_primary_family"]
    new_family = protocol["co_primary_family"]
    for key in (
        "C_c3_low_jump",
        "Holm_family_exactly",
        "candidate_combination_substitution_or_refit_forbidden",
        "family_size",
    ):
        if new_family.get(key) != old_family.get(key):
            raise RuntimeError(f"v4.2 illegally changed retained co-primary field: {key}")
    if new_family["members"][0] != old_family["members"][0]:
        raise RuntimeError("v4.2 illegally changed the frozen B co-primary")
    for key in ("id", "endpoint", "minimum_auc", "orientation"):
        if new_family["members"][1].get(key) != old_family["members"][1].get(key):
            raise RuntimeError(f"v4.2 illegally changed retained E field: {key}")
    old_stage_b = old["stage_B_statistics"]
    new_stage_b = protocol["stage_B_statistics"]
    for key in (
        "B_quality_gates",
        "auc_formula",
        "multiple_testing",
        "permutation",
        "primary_statistic",
    ):
        if new_stage_b.get(key) != old_stage_b.get(key):
            raise RuntimeError(f"v4.2 illegally changed retained Stage-B field: {key}")
    for key in ("B", "alpha_e"):
        if (
            new_stage_b["threshold_semantics"].get(key)
            != old_stage_b["threshold_semantics"].get(key)
        ):
            raise RuntimeError(f"v4.2 illegally changed retained threshold field: {key}")
    mechanics = protocol["pre_label_E_gates"]["confirmation_path_mechanics_gate"]
    if (
        mechanics["exact_confirmation_axis_count"] != 768
        or mechanics["per_scale_minimum_qualifying_started_paths"] != 12
        or mechanics["per_scale_minimum_qualifying_started_classes"] != 3
        or mechanics["per_scale_started_path_complete_h_step_coverage_fraction"] != 1.0
        or mechanics["per_scale_maximum_last_valid_fallback_fraction_among_started_steps"]
        != 0.01
    ):
        raise RuntimeError("v4.2 mechanics gate changed")
    incremental = protocol["E_incremental_and_ablation_gates"]
    for key in (
        "E_beyond_B",
        "E_beyond_G_start_schedule",
        "B_start_beyond_no_state_gate",
        "multi_step_vs_one_shot",
    ):
        block = incremental[key]
        if (
            block.get("draws") != 100000
            or block.get("one_sided_95_percent_lower_bound")
            != "sort 100000 DeltaAUC replicates and take zero-based index 4999 (the 5000th order statistic; no interpolation)"
            or "swap" not in block.get("score_identity_swap_forbidden", "")
        ):
            raise RuntimeError(f"v4.2 bootstrap contract changed: {key}")


def validate_lock(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_path = root / "protocol.json"
    manifest_path = root / "manifest.json"
    completion_path = root / "completion.json"
    protocol = load_json(protocol_path)
    old = load_json(root / "upstream/v4_1_protocol.json")
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    validate_protocol(protocol, old)
    if (
        canonical_sha256(without_identity(manifest)) != manifest.get("identity_sha256")
        or manifest.get("experiment") != LOCK_NAME
        or manifest.get("protocol_identity_sha256") != protocol["identity_sha256"]
        or manifest.get("ready_for_real_sampling") is not False
        or completion
        != {
            "complete": True,
            "manifest_identity_sha256": manifest.get("identity_sha256"),
            "manifest_file_sha256": sha256_file(manifest_path),
            "protocol_identity_sha256": protocol["identity_sha256"],
            "protocol_file_sha256": sha256_file(protocol_path),
            "ready_for_real_sampling": False,
        }
        or records(root) != manifest.get("files")
    ):
        raise RuntimeError("v4.2 lock manifest/completion/tree mismatch")
    rebind = load_json(root / "downstream_rebind_requirements.json")
    if (
        rebind.get("v4_2_protocol_identity_sha256") != protocol["identity_sha256"]
        or rebind.get("reuse_without_new_lock_forbidden") is not True
        or rebind.get("real_sampling_authorized") is not False
    ):
        raise RuntimeError("v4.2 downstream rebind receipt changed")
    upstream_old_protocol = root / "upstream/v4_1_protocol.json"
    upstream_old_manifest = root / "upstream/v4_1_manifest.json"
    upstream_old_completion = root / "upstream/v4_1_completion.json"
    embedded_old_manifest = load_json(upstream_old_manifest)
    embedded_old_completion = load_json(upstream_old_completion)
    if (
        old.get("identity_sha256") != EXPECTED_V4_1_PROTOCOL_ID
        or embedded_old_manifest.get("identity_sha256") != EXPECTED_V4_1_MANIFEST_ID
        or embedded_old_completion.get("protocol_identity_sha256")
        != EXPECTED_V4_1_PROTOCOL_ID
        or embedded_old_completion.get("manifest_identity_sha256")
        != EXPECTED_V4_1_MANIFEST_ID
        or embedded_old_completion.get("protocol_file_sha256")
        != sha256_file(upstream_old_protocol)
        or embedded_old_completion.get("manifest_file_sha256")
        != sha256_file(upstream_old_manifest)
    ):
        raise RuntimeError("v4.2 embedded v4.1 envelope changed")
    upstream_method_protocol = root / "upstream/method_v2_2_protocol.json"
    upstream_method_manifest = root / "upstream/method_v2_2_manifest.json"
    upstream_method_completion = root / "upstream/method_v2_2_completion.json"
    embedded_method_manifest = load_json(upstream_method_manifest)
    embedded_method_completion = load_json(upstream_method_completion)
    method_binding = protocol["method_lock"]
    if (
        embedded_method_manifest.get("identity_sha256") != EXPECTED_METHOD_ID
        or embedded_method_completion.get("identity_sha256") != EXPECTED_METHOD_ID
        or embedded_method_completion.get("manifest_sha256")
        != sha256_file(upstream_method_manifest)
        or sha256_file(upstream_method_protocol)
        != method_binding["protocol_file_sha256"]
        or sha256_file(upstream_method_completion)
        != method_binding["completion_file_sha256"]
        or sha256_file(root / "upstream/method_v2_2_matched_q_power.json")
        != method_binding["matched_q_power_gate_file_sha256"]
        or sha256_file(root / "upstream/method_v2_2_adaptive_null.json")
        != method_binding["adaptive_null_audit_file_sha256"]
    ):
        raise RuntimeError("v4.2 embedded method-v2.2 envelope changed")
    return protocol, manifest


def freeze(output: Path) -> Path:
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"refusing to overwrite v4.2 lock: {output}")
    for source in (Path(__file__).resolve(), DOC):
        if not source.is_file() or source.is_symlink():
            raise RuntimeError(f"missing regular v4.2 source: {source}")
    old, old_manifest, old_completion = validate_v4_1()
    method, method_manifest, method_completion, power, null = validate_method()
    zero = audit_zero_real_screen()
    protocol = build_protocol(
        old=old,
        old_manifest=old_manifest,
        old_completion=old_completion,
        method=method,
        method_manifest=method_manifest,
        method_completion=method_completion,
        power=power,
        null=null,
        zero=zero,
    )
    validate_protocol(protocol, old)
    rebind = build_rebind_requirements(protocol["identity_sha256"])
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        write_json(staging / "protocol.json", protocol)
        write_json(staging / "pre_sampling_zero_audit.json", zero)
        write_json(staging / "downstream_rebind_requirements.json", rebind)
        copy_regular(DOC, staging / "scientific_amendment_zh.md")
        copy_regular(Path(__file__).resolve(), staging / "sources" / Path(__file__).name)
        for name in ("protocol.json", "manifest.json", "completion.json"):
            copy_regular(V4_1_LOCK / name, staging / "upstream" / f"v4_1_{name}")
            copy_regular(METHOD_LOCK / name, staging / "upstream" / f"method_v2_2_{name}")
        copy_regular(
            METHOD_LOCK / "matched_q_conditional_power_gate.json",
            staging / "upstream/method_v2_2_matched_q_power.json",
        )
        copy_regular(
            METHOD_LOCK / "adaptive_predictable_null_audit.json",
            staging / "upstream/method_v2_2_adaptive_null.json",
        )
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "complete",
            "experiment": LOCK_NAME,
            "protocol_identity_sha256": protocol["identity_sha256"],
            "method_identity_sha256": EXPECTED_METHOD_ID,
            "superseded_v4_1_protocol_identity_sha256": EXPECTED_V4_1_PROTOCOL_ID,
            "real_event_screen_samples_at_freeze": 0,
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
    with tempfile.TemporaryDirectory(prefix="dit-scientific-v4-2-test-") as temporary:
        lock = freeze(Path(temporary) / "lock")
        protocol, _ = validate_lock(lock)
        if protocol["execution_readiness"]["ready_for_real_sampling"] is not False:
            raise AssertionError("v4.2 self-test authorized execution")
    print("v4.2 self-test passed: v4.1 evaluation preserved, method v2.2/bootstrap rebound, execution blocked")


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
                "method_identity_sha256": EXPECTED_METHOD_ID,
                "ready_for_real_sampling": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
