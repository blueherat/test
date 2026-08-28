#!/usr/bin/env python3
"""Freeze scientific v4.2.1 as a claim-limited supersession of v4.2.

The immutable, unused v4.2 lock is preserved.  This correction says that the
E-vs-G gate establishes improvement only over one prespecified scalar start
summary.  It also preregisters a non-gating exact-schedule conditional
concordance diagnostic.  No real event-screen artifact may exist at freeze.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    from . import freeze_dit_event_rich_confirmation_protocol_v4_2 as v4_2
except ImportError:  # pragma: no cover
    import freeze_dit_event_rich_confirmation_protocol_v4_2 as v4_2


ROOT = Path(__file__).resolve().parents[1]
LOCK_NAME = "dit_event_rich_confirmation_protocol_lock_v4_2_1"
OUTPUT = ROOT / "experiments/locks" / LOCK_NAME
V4_2_LOCK = ROOT / "experiments/locks/dit_event_rich_confirmation_protocol_lock_v4_2"
METHOD_LOCK = ROOT / "experiments/locks/dit_blur_focused_eprocess_protocol_lock_v2_2"
DOC = ROOT / "docs/DIT_EVENT_RICH_SCIENTIFIC_V4_2_1_ZH.md"
EXPECTED_V4_2_PROTOCOL_ID = "03a82123980c63d91029ebeb146240123f5cfae5ca90eff8f8342f203b3b8e9f"
EXPECTED_V4_2_MANIFEST_ID = "2fcfcf5abf42c6e4d8c3bcd11be60423b6c1f87b379d5832c77262b92ce145a6"
EXPECTED_METHOD_ID = "cc4dc5e7c06c25f4d8567a42fb4f0387097a6296c587543830bfeaa4771f6921"


def validate_v4_2() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol, manifest = v4_2.validate_lock(V4_2_LOCK)
    completion_path = V4_2_LOCK / "completion.json"
    completion = v4_2.load_json(completion_path)
    if (
        protocol.get("identity_sha256") != EXPECTED_V4_2_PROTOCOL_ID
        or manifest.get("identity_sha256") != EXPECTED_V4_2_MANIFEST_ID
        or completion.get("protocol_identity_sha256") != EXPECTED_V4_2_PROTOCOL_ID
        or completion.get("manifest_identity_sha256") != EXPECTED_V4_2_MANIFEST_ID
        or completion.get("protocol_file_sha256")
        != v4_2.sha256_file(V4_2_LOCK / "protocol.json")
        or completion.get("manifest_file_sha256")
        != v4_2.sha256_file(V4_2_LOCK / "manifest.json")
        or completion.get("ready_for_real_sampling") is not False
    ):
        raise RuntimeError("immutable scientific v4.2 lock changed")
    return protocol, manifest, completion


def corrected_zero_audit() -> dict[str, Any]:
    result = v4_2.audit_zero_real_screen()
    result.pop("identity_sha256", None)
    result["status"] = "FROZEN_PRE_V4_2_1_REAL_EVENT_SCREEN_COUNT_ZERO"
    result["superseded_v4_2_protocol_identity_sha256"] = EXPECTED_V4_2_PROTOCOL_ID
    result["identity_sha256"] = v4_2.canonical_sha256(result)
    return result


def conditional_concordance_contract() -> dict[str, Any]:
    return {
        "status": "PREREGISTERED_DESCRIPTIVE_NON_GATING_DIAGNOSTIC",
        "score": "E_blur_gated_running_max_log",
        "orientation": "bad_high",
        "positive": "retained blur_or_soft_fusion clear_bad",
        "negative": "clean_good",
        "execution_precondition": (
            "run only if the E confirmation product is lawfully opened after all pre-label "
            "mechanics gates and Stage A; otherwise record NOT_RUN"
        ),
        "eligible_rows": "the unchanged confirmation binary endpoint subset after immutable labels and Stage-A handling",
        "exact_stratum": (
            "(class_id, scale1_start, scale4_start), where each scale start is the exact "
            "(start_time_index,start_remaining_effective_count) pair or bottom for no start"
        ),
        "no_start_encoding": {
            "symbol": "bottom",
            "required_metadata": "start_time_index=-1 and start_remaining_effective_count=0",
        },
        "statistic": (
            "C=sum_s sum_{positive i,negative j in s}[1(E_i>E_j)+0.5*1(E_i=E_j)] "
            "/ sum_s(n_positive_s*n_negative_s)"
        ),
        "aggregation": "micro pooled by comparable positive-negative pair count across exact strata",
        "macro_average_of_cell_AUCs_forbidden": True,
        "empty_or_single_outcome_strata": (
            "contribute zero numerator and zero denominator; never impute 0.5"
        ),
        "mandatory_reports": {
            "exact_schedule_comparable_pair_denominator": "sum_s n_positive_s*n_negative_s",
            "class_only_comparable_pair_denominator": (
                "sum_c n_positive_c*n_negative_c on the same eligible rows"
            ),
            "class_only_pair_coverage": (
                "exact_schedule_comparable_pair_denominator divided by "
                "class_only_comparable_pair_denominator; null if the latter is zero"
            ),
            "informative_exact_strata": "count of strata with both outcomes",
            "informative_classes": "distinct class_ids among informative exact strata",
            "informative_distinct_global_seeds": (
                "distinct original confirmation global seeds contributing an eligible "
                "positive or negative row to at least one informative exact stratum; "
                "mild/disputed or otherwise ineligible rows never count"
            ),
            "tie_rate": (
                "number of tied E positive-negative pairs within exact strata divided by "
                "the exact-schedule comparable-pair denominator"
            ),
        },
        "descriptive_bootstrap": {
            "draws": 100000,
            "rng": "numpy.default_rng(PCG64(seed=2026082815))",
            "cluster_unit": "one complete six-class confirmation global-seed block",
            "resampling": (
                "sample 128 global-seed blocks with replacement, retain all six classes and "
                "paired outcome/score/start metadata, and recompute exact strata and C"
            ),
            "zero_denominator_replicate": (
                "record C as undefined; do not impute 0.5, do not redraw, and report the "
                "zero-denominator replicate fraction"
            ),
            "descriptive_two_sided_95_percent_interval": (
                "among M defined replicates only, sort C and take indices "
                "max(0,ceil(0.025*M)-1) and max(0,ceil(0.975*M)-1); if M=0 report null"
            ),
            "p_value_or_decision_not_produced": True,
        },
        "has_pass_fail_gate": False,
        "enters_Holm_family": False,
        "enters_rollback_authorization": False,
        "claim_limit": (
            "At most describes within-exact-schedule ranking association on the subset with "
            "comparable pairs.  It is not a population-wide estimand, a causal innovation-"
            "alignment test, or evidence that every start-schedule explanation is excluded."
        ),
    }


def build_protocol(
    *,
    old: Mapping[str, Any],
    old_manifest: Mapping[str, Any],
    old_completion: Mapping[str, Any],
    zero: Mapping[str, Any],
) -> dict[str, Any]:
    protocol = copy.deepcopy(dict(old))
    protocol.pop("identity_sha256", None)
    protocol["scientific_revision"] = "v4.2.1"
    protocol["status"] = "SCIENTIFIC_V4_2_1_CLAIM_LIMITED_FROZEN_EXECUTION_NOT_READY"
    protocol["objective"] = (
        "On the unchanged prospectively blur-enriched external evaluation population, test "
        "method-v2.2 E against the frozen quality endpoint, B, and one prespecified scalar "
        "G_start summary.  Any exact-schedule conditional concordance is descriptive only."
    )
    supersession = protocol["supersession"]
    supersession["statement"] = (
        "v4.2.1 supersedes immutable, unused v4.2 before any real screen sample.  It leaves "
        "the method, population, labels, B/E primary family and all gates unchanged, but "
        "limits the E-vs-G interpretation and preregisters a non-gating exact-schedule "
        "conditional concordance diagnostic."
    )
    supersession["superseded_v4_2"] = {
        "path": str(V4_2_LOCK),
        "protocol_identity_sha256": old["identity_sha256"],
        "manifest_identity_sha256": old_manifest["identity_sha256"],
        "protocol_file_sha256": old_completion["protocol_file_sha256"],
        "manifest_file_sha256": old_completion["manifest_file_sha256"],
        "downstream_rebind_file_sha256": v4_2.sha256_file(
            V4_2_LOCK / "downstream_rebind_requirements.json"
        ),
        "ready_for_real_sampling": False,
        "real_sampling_authorized_or_executed": False,
        "preserved_immutable": True,
        "correction": (
            "E>G_start exceeds only one prespecified scalar summary; it does not exclude all "
            "start-schedule information and cannot causally establish innovation alignment"
        ),
    }
    supersession["zero_real_screen_audit_identity_sha256"] = zero["identity_sha256"]
    supersession["real_event_screen_samples_at_v4_freeze"] = 0

    incremental = protocol["E_incremental_and_ablation_gates"]
    e_vs_g = incremental["E_beyond_G_start_schedule"]
    e_vs_g["meaning"] = (
        "tests incremental discrimination beyond this one prespecified innovation-free scalar "
        "summary only"
    )
    e_vs_g.pop("required_for_path_LR_incremental_claim_and_rollback", None)
    e_vs_g["required_for_increment_over_prespecified_scalar_G_and_rollback"] = True
    e_vs_g["claim_limit"] = (
        "G is not an unrestricted schedule-only predictor and is not a causal control.  "
        "Although it encodes almost all allowed (h1,h4) pairs, it is a fixed ordinal scalar "
        "with a collision and cannot rule out every nonmonotone or interacting start-schedule "
        "explanation, nor prove innovation alignment."
    )
    incremental["schedule_exact_conditional_concordance_diagnostic"] = (
        conditional_concordance_contract()
    )

    authorization = protocol["authorization_and_stop_rules"]
    authorization["E_evidence_driven_rollback_authorization"] = (
        "only if E passes every v2.2 pre-label gate, Stage-A gate, primary quality gate, "
        "paired cluster-bootstrap E-beyond-B lower bound, E-beyond-prespecified-scalar-G_start "
        "lower bound, and E-beyond-no-state-gate lower bound may a separately frozen rollback "
        "study begin; the descriptive exact-schedule diagnostic neither gates nor authorizes it"
    )
    authorization["E_beyond_G_claim_limit"] = (
        "passing E>G supports only increment over the prespecified scalar G score, not full "
        "schedule adjustment or causal innovation-alignment language"
    )

    source_requirements = protocol["source_lock_requirements"]
    source_requirements["all_existing_v4_1_bound_locks_are_incompatible"] = True
    source_requirements["all_existing_v4_2_bound_locks_are_incompatible"] = True
    source_requirements["selector_and_scientific_selftest"] = (
        "new frozen hashes/artifacts must bind v4.2.1 identity"
    )
    source_requirements["dynamic_pipeline"] = (
        "new lock binding method v2.2 and scientific v4.2.1; physical B/E/G/ablation "
        "products; all-768 pre-label mechanics; paired hard-gate bootstraps; and the "
        "non-gating exact-schedule concordance diagnostic"
    )

    protocol["execution_readiness"] = {
        "ready_for_real_sampling": False,
        "reason": (
            "scientific v4.2.1 is frozen, but all v4.1/v4.2-bound endpoint, review, dynamic, "
            "selector and self-test sources require new non-overwriting v4.2.1 bindings"
        ),
        "required_before_sampling": [
            "v4.2.1-bound endpoint source lock",
            "v4.2.1-bound blind-review source lock",
            "v4.2.1/method-v2.2-bound dynamic and evaluator source lock",
            "v4.2.1-bound selector and scientific self-test artifacts",
            "real independent reviewer qualification and reserve inputs",
            "new execution-authorization receipt binding all v4.2.1 identities",
        ],
        "this_lock_can_never_be_mutated_to_ready": True,
        "future_execution_authorization_must_be_new_non_overwriting_artifact": True,
    }
    protocol["identity_sha256"] = v4_2.canonical_sha256(protocol)
    return protocol


def validate_protocol(protocol: Mapping[str, Any], old: Mapping[str, Any]) -> None:
    if (
        v4_2.canonical_sha256(v4_2.without_identity(protocol))
        != protocol.get("identity_sha256")
        or protocol.get("scientific_revision") != "v4.2.1"
        or protocol.get("status")
        != "SCIENTIFIC_V4_2_1_CLAIM_LIMITED_FROZEN_EXECUTION_NOT_READY"
        or protocol.get("method_lock", {}).get("identity_sha256") != EXPECTED_METHOD_ID
        or protocol.get("execution_readiness", {}).get("ready_for_real_sampling") is not False
    ):
        raise RuntimeError("v4.2.1 identity/method/status/readiness changed")
    for key in (
        "endpoint_screen",
        "label_system",
        "selector_contract",
        "anchor_go_rule",
        "dynamic_axis",
        "stage_A_label_only_event_gate",
        "co_primary_family",
        "frozen_method_details",
        "pre_label_E_gates",
        "stage_B_statistics",
        "independence_and_scope",
        "method_lock",
    ):
        if protocol.get(key) != old.get(key):
            raise RuntimeError(f"v4.2.1 illegally changed retained block: {key}")
    old_incremental = old["E_incremental_and_ablation_gates"]
    new_incremental = protocol["E_incremental_and_ablation_gates"]
    for key in (
        "paired_cluster_bootstrap_common_rule",
        "E_beyond_B",
        "B_start_beyond_no_state_gate",
        "multi_step_vs_one_shot",
        "fixed_operating_point_report",
        "no_posthoc_combination_or_ablation_substitution",
    ):
        if new_incremental.get(key) != old_incremental.get(key):
            raise RuntimeError(f"v4.2.1 illegally changed retained incremental block: {key}")
    e_vs_g = new_incremental["E_beyond_G_start_schedule"]
    if (
        e_vs_g.get("draws") != 100000
        or e_vs_g.get("rng") != "numpy.default_rng(PCG64(seed=2026082813))"
        or e_vs_g.get("required_for_increment_over_prespecified_scalar_G_and_rollback")
        is not True
        or "cannot rule out every" not in e_vs_g.get("claim_limit", "")
    ):
        raise RuntimeError("v4.2.1 E-vs-G claim boundary changed")
    diagnostic = new_incremental.get("schedule_exact_conditional_concordance_diagnostic", {})
    bootstrap = diagnostic.get("descriptive_bootstrap", {})
    if (
        diagnostic.get("has_pass_fail_gate") is not False
        or diagnostic.get("enters_rollback_authorization") is not False
        or diagnostic.get("macro_average_of_cell_AUCs_forbidden") is not True
        or bootstrap.get("draws") != 100000
        or bootstrap.get("rng") != "numpy.default_rng(PCG64(seed=2026082815))"
        or "do not impute 0.5" not in bootstrap.get("zero_denominator_replicate", "")
    ):
        raise RuntimeError("v4.2.1 exact-schedule diagnostic contract changed")


def build_rebind_requirements(protocol_id: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "ALL_V4_1_OR_V4_2_BOUND_EXECUTION_SOURCES_INCOMPATIBLE_WITH_V4_2_1",
        "v4_2_protocol_identity_sha256": EXPECTED_V4_2_PROTOCOL_ID,
        "v4_2_1_protocol_identity_sha256": protocol_id,
        "method_v2_2_identity_sha256": EXPECTED_METHOD_ID,
        "known_incompatible_existing_locks": [
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
                "role": "dynamic_selector_selftest_or_evaluator",
                "path": "any artifact bound to scientific v4.1 or v4.2",
            },
        ],
        "roles_requiring_new_non_overwriting_binding": [
            "endpoint_sampler",
            "review_pipeline",
            "class_selector_output",
            "scientific_selftest",
            "dynamic_B_E_G_and_ablation_pipeline",
            "primary_and_incremental_evaluator",
            "schedule_exact_conditional_concordance_diagnostic",
            "execution_authorization_receipt",
        ],
        "reuse_or_manifest_relabeling_forbidden": True,
        "real_sampling_authorized": False,
    }


def validate_lock(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_path = root / "protocol.json"
    manifest_path = root / "manifest.json"
    completion_path = root / "completion.json"
    protocol = v4_2.load_json(protocol_path)
    old = v4_2.load_json(root / "upstream/v4_2_protocol.json")
    manifest = v4_2.load_json(manifest_path)
    completion = v4_2.load_json(completion_path)
    validate_protocol(protocol, old)
    if (
        v4_2.canonical_sha256(v4_2.without_identity(manifest))
        != manifest.get("identity_sha256")
        or manifest.get("experiment") != LOCK_NAME
        or manifest.get("protocol_identity_sha256") != protocol["identity_sha256"]
        or manifest.get("ready_for_real_sampling") is not False
        or completion
        != {
            "complete": True,
            "manifest_identity_sha256": manifest.get("identity_sha256"),
            "manifest_file_sha256": v4_2.sha256_file(manifest_path),
            "protocol_identity_sha256": protocol["identity_sha256"],
            "protocol_file_sha256": v4_2.sha256_file(protocol_path),
            "ready_for_real_sampling": False,
        }
        or v4_2.records(root) != manifest.get("files")
    ):
        raise RuntimeError("v4.2.1 lock manifest/completion/tree mismatch")
    embedded_old_manifest = v4_2.load_json(root / "upstream/v4_2_manifest.json")
    embedded_old_completion = v4_2.load_json(root / "upstream/v4_2_completion.json")
    if (
        old.get("identity_sha256") != EXPECTED_V4_2_PROTOCOL_ID
        or embedded_old_manifest.get("identity_sha256") != EXPECTED_V4_2_MANIFEST_ID
        or embedded_old_completion.get("protocol_identity_sha256")
        != EXPECTED_V4_2_PROTOCOL_ID
        or embedded_old_completion.get("manifest_identity_sha256")
        != EXPECTED_V4_2_MANIFEST_ID
        or embedded_old_completion.get("protocol_file_sha256")
        != v4_2.sha256_file(root / "upstream/v4_2_protocol.json")
        or embedded_old_completion.get("manifest_file_sha256")
        != v4_2.sha256_file(root / "upstream/v4_2_manifest.json")
    ):
        raise RuntimeError("v4.2.1 embedded v4.2 envelope changed")
    if (
        v4_2.sha256_file(root / "upstream/v4_2_downstream_rebind_requirements.json")
        != protocol["supersession"]["superseded_v4_2"][
            "downstream_rebind_file_sha256"
        ]
    ):
        raise RuntimeError("v4.2.1 embedded v4.2 rebind receipt changed")
    method_binding = protocol["method_lock"]
    embedded_method_manifest = v4_2.load_json(
        root / "upstream/method_v2_2_manifest.json"
    )
    embedded_method_completion = v4_2.load_json(
        root / "upstream/method_v2_2_completion.json"
    )
    if (
        embedded_method_manifest.get("identity_sha256") != EXPECTED_METHOD_ID
        or embedded_method_completion.get("identity_sha256") != EXPECTED_METHOD_ID
        or embedded_method_completion.get("manifest_sha256")
        != v4_2.sha256_file(root / "upstream/method_v2_2_manifest.json")
        or v4_2.sha256_file(root / "upstream/method_v2_2_protocol.json")
        != method_binding["protocol_file_sha256"]
        or v4_2.sha256_file(root / "upstream/method_v2_2_completion.json")
        != method_binding["completion_file_sha256"]
        or v4_2.sha256_file(root / "upstream/method_v2_2_matched_q_power.json")
        != method_binding["matched_q_power_gate_file_sha256"]
        or v4_2.sha256_file(root / "upstream/method_v2_2_adaptive_null.json")
        != method_binding["adaptive_null_audit_file_sha256"]
    ):
        raise RuntimeError("v4.2.1 embedded method-v2.2 envelope changed")
    rebind = v4_2.load_json(root / "downstream_rebind_requirements.json")
    if (
        rebind.get("v4_2_1_protocol_identity_sha256") != protocol["identity_sha256"]
        or rebind.get("reuse_or_manifest_relabeling_forbidden") is not True
        or rebind.get("real_sampling_authorized") is not False
    ):
        raise RuntimeError("v4.2.1 downstream rebind receipt changed")
    return protocol, manifest


def freeze(output: Path) -> Path:
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"refusing to overwrite v4.2.1 lock: {output}")
    for source in (Path(__file__).resolve(), DOC):
        if not source.is_file() or source.is_symlink():
            raise RuntimeError(f"missing regular v4.2.1 source: {source}")
    old, old_manifest, old_completion = validate_v4_2()
    method, method_manifest, method_completion, power, null = v4_2.validate_method()
    if method_manifest.get("identity_sha256") != EXPECTED_METHOD_ID:
        raise RuntimeError("v4.2.1 method-v2.2 dependency changed")
    zero = corrected_zero_audit()
    protocol = build_protocol(
        old=old,
        old_manifest=old_manifest,
        old_completion=old_completion,
        zero=zero,
    )
    validate_protocol(protocol, old)
    rebind = build_rebind_requirements(protocol["identity_sha256"])
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        v4_2.write_json(staging / "protocol.json", protocol)
        v4_2.write_json(staging / "pre_sampling_zero_audit.json", zero)
        v4_2.write_json(staging / "downstream_rebind_requirements.json", rebind)
        v4_2.copy_regular(DOC, staging / "scientific_amendment_zh.md")
        v4_2.copy_regular(
            Path(__file__).resolve(), staging / "sources" / Path(__file__).name
        )
        v4_2.copy_regular(
            Path(v4_2.__file__).resolve(),
            staging / "sources/freeze_dit_event_rich_confirmation_protocol_v4_2.py",
        )
        for name in ("protocol.json", "manifest.json", "completion.json"):
            v4_2.copy_regular(V4_2_LOCK / name, staging / "upstream" / f"v4_2_{name}")
            v4_2.copy_regular(METHOD_LOCK / name, staging / "upstream" / f"method_v2_2_{name}")
        v4_2.copy_regular(
            V4_2_LOCK / "downstream_rebind_requirements.json",
            staging / "upstream/v4_2_downstream_rebind_requirements.json",
        )
        v4_2.copy_regular(
            METHOD_LOCK / "matched_q_conditional_power_gate.json",
            staging / "upstream/method_v2_2_matched_q_power.json",
        )
        v4_2.copy_regular(
            METHOD_LOCK / "adaptive_predictable_null_audit.json",
            staging / "upstream/method_v2_2_adaptive_null.json",
        )
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "complete",
            "experiment": LOCK_NAME,
            "protocol_identity_sha256": protocol["identity_sha256"],
            "method_identity_sha256": EXPECTED_METHOD_ID,
            "superseded_v4_2_protocol_identity_sha256": EXPECTED_V4_2_PROTOCOL_ID,
            "real_event_screen_samples_at_freeze": 0,
            "ready_for_real_sampling": False,
            "files": v4_2.records(staging),
        }
        manifest["identity_sha256"] = v4_2.canonical_sha256(manifest)
        v4_2.write_json(staging / "manifest.json", manifest)
        completion = {
            "complete": True,
            "manifest_identity_sha256": manifest["identity_sha256"],
            "manifest_file_sha256": v4_2.sha256_file(staging / "manifest.json"),
            "protocol_identity_sha256": protocol["identity_sha256"],
            "protocol_file_sha256": v4_2.sha256_file(staging / "protocol.json"),
            "ready_for_real_sampling": False,
        }
        v4_2.write_json(staging / "completion.json", completion)
        os.replace(staging, output)
        validate_lock(output)
        return output
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="dit-scientific-v4-2-1-test-") as temporary:
        lock = freeze(Path(temporary) / "lock")
        protocol, _ = validate_lock(lock)
        diagnostic = protocol["E_incremental_and_ablation_gates"][
            "schedule_exact_conditional_concordance_diagnostic"
        ]
        if (
            diagnostic["has_pass_fail_gate"] is not False
            or diagnostic["enters_rollback_authorization"] is not False
            or protocol["execution_readiness"]["ready_for_real_sampling"] is not False
        ):
            raise AssertionError("v4.2.1 self-test widened claims or authorized execution")
    print(
        "v4.2.1 self-test passed: v4.2 immutable, G claim limited, exact-schedule "
        "diagnostic non-gating, execution blocked"
    )


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
