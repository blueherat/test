#!/usr/bin/env python3
"""CPU/synthetic contract tests for the prospective v4 B/E dynamic pipeline."""

from __future__ import annotations

import csv
import io
import json
import math
import sys
import tempfile
from types import ModuleType
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    from . import observe_dit_blur_focused_eprocess_v2 as core
    from .dit_scientific_v4_be_contract import (
        B_CANDIDATE,
        CALIBRATION_SEEDS,
        CANDIDATES,
        CONFIRMATION_SEEDS,
        E_CANDIDATE,
        METHOD_LOCK_ID,
        canonical_sha256,
        derive_pair_seed,
        exact_pairs,
        publish_artifact,
        reject_forbidden_method_name,
        sha256_array,
        sha256_file,
        validate_trace_plan,
        write_json,
    )
    from .evaluate_dit_scientific_v4_be import (
        CONSENSUS_KIND,
        CONSENSUS_STATUS,
        LABEL_COLUMNS,
        load_consensus_aggregate_only,
        load_diagnostic_product,
        load_label_rows,
        load_prelable_receipt,
        load_score_product,
        paired_seed_cluster_bootstrap,
        rollback_mechanism_authorized,
        schedule_exact_conditional_concordance,
        validate_common_product_provenance,
        validate_E_mechanics_product_envelope,
        validate_no_touch_receipt,
    )
    from .extract_dit_scientific_v4_be_products import (
        reject_preexisting_upstream_dit_modules as reject_extractor_dit_modules,
    )
    from .sample_dit_scientific_v4_be_traces import (
        ENDPOINT_NAME,
        METHOD_TREE,
        REVIEW_TREE,
        TRACE_ARRAYS,
        _load_frozen_method_v2,
        pair_relative_directory,
        publish_pair,
        reject_preexisting_upstream_dit_modules as reject_sampler_dit_modules,
        validate_endpoint_pair,
        validate_method_tree_firewall,
        validate_review_tree_firewall,
        validate_trace_pair,
    )
except ImportError:
    import observe_dit_blur_focused_eprocess_v2 as core  # type: ignore
    from dit_scientific_v4_be_contract import (  # type: ignore
        B_CANDIDATE,
        CALIBRATION_SEEDS,
        CANDIDATES,
        CONFIRMATION_SEEDS,
        E_CANDIDATE,
        METHOD_LOCK_ID,
        canonical_sha256,
        derive_pair_seed,
        exact_pairs,
        publish_artifact,
        reject_forbidden_method_name,
        sha256_array,
        sha256_file,
        validate_trace_plan,
        write_json,
    )
    from evaluate_dit_scientific_v4_be import (  # type: ignore
        CONSENSUS_KIND,
        CONSENSUS_STATUS,
        LABEL_COLUMNS,
        load_consensus_aggregate_only,
        load_diagnostic_product,
        load_label_rows,
        load_prelable_receipt,
        load_score_product,
        paired_seed_cluster_bootstrap,
        rollback_mechanism_authorized,
        schedule_exact_conditional_concordance,
        validate_common_product_provenance,
        validate_E_mechanics_product_envelope,
        validate_no_touch_receipt,
    )
    from extract_dit_scientific_v4_be_products import (  # type: ignore
        reject_preexisting_upstream_dit_modules as reject_extractor_dit_modules,
    )
    from sample_dit_scientific_v4_be_traces import (  # type: ignore
        ENDPOINT_NAME,
        METHOD_TREE,
        REVIEW_TREE,
        TRACE_ARRAYS,
        _load_frozen_method_v2,
        pair_relative_directory,
        publish_pair,
        reject_preexisting_upstream_dit_modules as reject_sampler_dit_modules,
        validate_endpoint_pair,
        validate_method_tree_firewall,
        validate_review_tree_firewall,
        validate_trace_pair,
    )


def synthetic_plan(protocol_identity: str) -> dict[str, Any]:
    classes = [11, 22, 33, 44, 55, 66]
    value: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "EVENT_RICH_BLUR_ANCHOR_PLAN_LOCK_V1",
        "status": "BLUR_ANCHOR_GO_DECISION_LOCKED_BEFORE_INTERNAL_TRACES",
        "protocol_identity_sha256": protocol_identity,
        "selection_identity_sha256": "1" * 64,
        "anchor_consensus_file_sha256": "2" * 64,
        "selected_classes": classes,
        "aggregate_counts": [],
        "decision": {
            "anchor_rows": 144,
            "blur_clear_bad": 10,
            "event_bearing_classes": 4,
            "clean_good": 100,
            "gates": {"blur_events": True, "event_bearing_classes": True, "clean_good": True},
            "go": True,
            "failed_gates": [],
        },
        "descriptive_only": {
            "one_sided_80pct_wilson_lower": 0.01,
            "not_a_go_input_or_auc_power_guarantee": True,
        },
        "calibration_seeds": list(CALIBRATION_SEEDS),
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "calibration_trace_rows": 6 * len(CALIBRATION_SEEDS),
        "confirmation_trace_rows": 6 * len(CONFIRMATION_SEEDS),
        "B_and_E_share_exact_selected_class_set": True,
        "external_visual_labels_used_only_for_cohort_enrichment_and_go": True,
        "method_score_threshold_intervention_or_external_representation_input_used": False,
    }
    value["identity_sha256"] = canonical_sha256(value)
    return value


def synthetic_arrays() -> dict[str, np.ndarray]:
    state_shape = (9, 4, 32, 32)
    return {
        "state_before": np.zeros(state_shape, dtype=np.float32),
        "pred_xstart": np.zeros(state_shape, dtype=np.float32),
        "p_standard_deviation": np.ones(state_shape, dtype=np.float32),
        "transition_innovation": np.zeros(state_shape, dtype=np.float32),
        "sampling_step": np.asarray((69, 79, 89, 99, 109, 119, 129, 139, 149), dtype=np.int16),
        "internal_timestep": np.asarray((180, 170, 160, 150, 140, 130, 120, 110, 100), dtype=np.int16),
        "alpha_bar": np.full(9, 0.5, dtype=np.float64),
    }


def write_synthetic_consensus(
    root: Path,
    *,
    protocol: dict[str, Any],
    plan: dict[str, Any],
    blur_seed_count: int = 3,
    inconsistent_aggregate: bool = False,
) -> None:
    root.mkdir()
    rows = []
    for seed in CONFIRMATION_SEEDS:
        for class_id in plan["selected_classes"]:
            blur = int(seed < CONFIRMATION_SEEDS[0] + blur_seed_count)
            rows.append(
                {
                    "phase": "confirmation",
                    "global_seed": seed,
                    "class_id": class_id,
                    "final_severity": "clear_bad" if blur else "clean_good",
                    "blur_component": blur,
                }
            )
    with (root / "evaluation_labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(LABEL_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    per_class = {}
    for index, class_id in enumerate(plan["selected_classes"]):
        bad = blur_seed_count + int(inconsistent_aggregate and index == 0)
        per_class[str(class_id)] = {
            "endpoint_count": len(CONFIRMATION_SEEDS),
            "raw_clear_bad": bad,
            "final_clean_good": len(CONFIRMATION_SEEDS) - bad,
            "final_mild_or_disputed": 0,
            "final_clear_bad": bad,
            "final_blur_or_soft_fusion": blur_seed_count,
            "final_structural_non_blur": 0,
        }
    overall = {
        name: sum(row[name] for row in per_class.values())
        for name in (
            "endpoint_count",
            "raw_clear_bad",
            "final_clean_good",
            "final_mild_or_disputed",
            "final_clear_bad",
            "final_blur_or_soft_fusion",
            "final_structural_non_blur",
        )
    }
    overall.update(
        {
            "union_any_positive": 0,
            "random_decoys": 0,
            "promoted_union_minority": 0,
            "promoted_zero_positive_decoys": 0,
            "downgraded_raw_2of3": 0,
            "unanimous_3of3_retained": 0,
        }
    )
    write_json(
        root / "aggregate_counts.json",
        {"phase": "confirmation", "overall": overall, "per_class": per_class},
    )
    identity = {
        "schema_version": 1,
        "artifact_kind": CONSENSUS_KIND,
        "status": CONSENSUS_STATUS,
        "phase": "confirmation",
        "event_protocol_identity_sha256": protocol["identity_sha256"],
        "anchor_plan_identity_sha256": plan["identity_sha256"],
        "row_count": len(rows),
        "candidate_scores_features_trajectories_embeddings_thresholds_or_ranks_opened": False,
        "endpoint_review_or_consensus_used_as_B_E_method_input": False,
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    records = [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(root.iterdir())
    ]
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "identity": identity,
        "identity_sha256": identity["identity_sha256"],
        "files": records,
    }
    manifest["manifest_identity_sha256"] = canonical_sha256(manifest)
    write_json(root / "manifest.json", manifest)
    write_json(
        root / "completion.json",
        {
            "complete": True,
            "identity_sha256": identity["identity_sha256"],
            "manifest_identity_sha256": manifest["manifest_identity_sha256"],
            "manifest_file_sha256": sha256_file(root / "manifest.json"),
            "file_count": len(records),
        },
    )


def run() -> dict[str, Any]:
    protocol = {
        "identity_sha256": "a" * 64,
        "pre_label_E_gates": {
            "matched_Q_conditional_power_gate": {
                "identity_sha256": "ae284448a324349488ab1be3962502d5450d006a64722bb717f5199903c6e6b2"
            },
            "adaptive_predictable_null_audit": {
                "identity_sha256": "4b69c132d39a70e615fc60ec12709daff670f15409a61c4f12e543f43fb7162c"
            },
        },
    }
    plan = synthetic_plan(protocol["identity_sha256"])
    contract = {
        "identity_sha256": "b" * 64,
        "assets": {"synthetic_asset_identity_sha256": "9" * 64},
        "sampler_contract": {
            "model": "DiT-XL/2 ImageNet-256",
            "sampler": "official 250-step ancestral DDPM",
            "sampling_steps": 250,
            "classes_per_invocation": 1,
            "full_2B_randn_like_each_of_250_transitions_including_t0": True,
        },
    }
    tests: list[str] = []
    source_manifest = {"identity_sha256": "c" * 64}
    with tempfile.TemporaryDirectory(prefix="v4-be-selftest-") as raw:
        root = Path(raw)
        plan_path = root / "anchor_plan.json"
        write_json(plan_path, plan)
        validate_trace_plan(plan_path, protocol)
        tests.append("native_anchor_plan_with_external_label_enrichment_and_internal_method_firewall")

        def save_image(_: Any, path: Path, **__: Any) -> None:
            Image.new("RGB", (256, 256), color=(17, 31, 47)).save(path)

        execution = {
            "derived_torch_seed": 0,  # overwritten below
            "rng_state_sha256": {
                "after_pair_seed_reset": "3" * 64,
                "after_initial_noise": "4" * 64,
                "after_250_full_2B_transition_draws": "5" * 64,
            },
            "tensor_sha256": {},
            "transition_randn_like_calls": 250,
            "transition_draw_shape": [2, 4, 32, 32],
            "terminal_t0_draw_consumed_then_masked": True,
            "preinnovation_observation_enabled": True,
            "preinnovation_observation_count": 9,
            "all_observations_rng_neutral": True,
        }
        for phase, seed in (("calibration", 1100), ("confirmation", 1200)):
            execution["derived_torch_seed"] = derive_pair_seed(seed, 11)
            publish_pair(
                root,
                decoded=object(),  # consumed only by the synthetic save_image callback
                arrays=synthetic_arrays(),
                execution=execution,
                save_image=save_image,
                contract=contract,
                plan=plan,
                phase=phase,
                global_seed=seed,
                class_id=11,
            )
        method_root = root / METHOD_TREE
        review_root = root / REVIEW_TREE
        validate_trace_pair(
            method_root,
            contract=contract,
            plan=plan,
            phase="calibration",
            global_seed=1100,
            class_id=11,
        )
        validate_trace_pair(
            method_root,
            contract=contract,
            plan=plan,
            phase="confirmation",
            global_seed=1200,
            class_id=11,
        )
        validate_endpoint_pair(
            review_root,
            contract=contract,
            plan=plan,
            phase="confirmation",
            global_seed=1200,
            class_id=11,
        )
        calibration_endpoint = (
            review_root / pair_relative_directory("calibration", 1100, 11) / ENDPOINT_NAME
        )
        if calibration_endpoint.exists():
            raise AssertionError("calibration endpoint leaked into review-only tree")
        if any(path.suffix == ".png" for path in method_root.rglob("*")):
            raise AssertionError("endpoint PNG leaked into method-only tree")
        if any(path.suffix == ".npz" for path in review_root.rglob("*")):
            raise AssertionError("minimum internal array leaked into review-only tree")
        validate_method_tree_firewall(method_root)
        validate_review_tree_firewall(review_root)
        tests.append("physical_method_review_tree_separation")

        poison = method_root / "labels.csv"
        poison.write_text("label\n1\n", encoding="utf-8")
        try:
            validate_method_tree_firewall(method_root)
        except RuntimeError:
            pass
        else:
            raise AssertionError("label poison escaped method tree firewall")
        poison.unlink()
        review_poison = review_root / "internal_trace.npz"
        review_poison.write_bytes(b"poison")
        try:
            validate_review_tree_firewall(review_root)
        except RuntimeError:
            pass
        else:
            raise AssertionError("trace poison escaped review tree firewall")
        review_poison.unlink()
        tests.append("cross_tree_poison_rejection")

        valid = {
            "schema_version": 1,
            "status": "PASS_OBSERVATION_NO_TOUCH",
            "dynamic_source_lock_manifest_identity_sha256": source_manifest[
                "identity_sha256"
            ],
            "dynamic_contract_identity_sha256": contract["identity_sha256"],
            "scientific_protocol_identity_sha256": protocol["identity_sha256"],
            "method_lock_identity_sha256": METHOD_LOCK_ID,
            "trace_plan_identity_sha256": plan["identity_sha256"],
            "trace_plan_file_sha256": sha256_file(plan_path),
            "trace_pool_identity_sha256": "8" * 64,
            "trace_pool_ordered_pair_axis_sha256": canonical_sha256(
                [
                    {"phase": phase, "global_seed": seed, "class_id": class_id}
                    for phase, seed, class_id in exact_pairs(plan)
                ]
            ),
            "confirmation_ordered_pair_axis_sha256": canonical_sha256(
                [
                    {"phase": phase, "global_seed": seed, "class_id": class_id}
                    for phase, seed, class_id in exact_pairs(
                        plan, phases=("confirmation",)
                    )
                ]
            ),
            "asset_identities": contract["assets"],
            "asset_identities_sha256": canonical_sha256(contract["assets"]),
            "pair": {"phase": "calibration", "global_seed": 1100, "class_id": 11},
            "derived_torch_seed": derive_pair_seed(1100, 11),
            "baseline_trace_array_sha256": {name: "6" * 64 for name in TRACE_ARRAYS},
            "observed_trace_array_sha256": {name: "6" * 64 for name in TRACE_ARRAYS},
            "baseline_rng_boundary_sha256": {
                "after_pair_seed_reset": "3" * 64,
                "after_initial_noise": "4" * 64,
                "after_250_full_2B_transition_draws": "5" * 64,
            },
            "observed_rng_boundary_sha256": {
                "after_pair_seed_reset": "3" * 64,
                "after_initial_noise": "4" * 64,
                "after_250_full_2B_transition_draws": "5" * 64,
            },
            "all_trace_arrays_bitwise_equal": True,
            "endpoint_tensor_sha256_equal": True,
            "rng_boundaries_equal": True,
            "baseline_endpoint_tensor_sha256": "7" * 64,
            "observed_endpoint_tensor_sha256": "7" * 64,
            "labels_reviews_external_representations_opened": False,
        }
        valid["identity_sha256"] = canonical_sha256(valid)
        valid_root = root / "valid_no_touch"
        publish_artifact(
            valid_root,
            artifact_kind="SCIENTIFIC_V4_OBSERVATION_NO_TOUCH_AUDIT",
            payloads={"no_touch_receipt.json": json.dumps(valid, sort_keys=True) + "\n"},
            manifest_fields={
                "dynamic_source_lock_manifest_identity_sha256": source_manifest[
                    "identity_sha256"
                ],
                "dynamic_contract_identity_sha256": contract["identity_sha256"],
                "scientific_protocol_identity_sha256": protocol["identity_sha256"],
                "method_lock_identity_sha256": METHOD_LOCK_ID,
                "trace_plan_identity_sha256": plan["identity_sha256"],
                "trace_pool_identity_sha256": valid["trace_pool_identity_sha256"],
                "confirmation_ordered_pair_axis_sha256": valid[
                    "confirmation_ordered_pair_axis_sha256"
                ],
                "pair": valid["pair"],
                "receipt_identity_sha256": valid["identity_sha256"],
            },
        )
        validate_no_touch_receipt(
            valid_root,
            contract=contract,
            source_manifest=source_manifest,
            protocol=protocol,
            plan=plan,
            trace_plan_file_sha256=sha256_file(plan_path),
        )
        wrong_pair = dict(valid)
        wrong_pair["pair"] = {
            "phase": "calibration",
            "global_seed": 1100,
            "class_id": 22,
        }
        wrong_pair["derived_torch_seed"] = derive_pair_seed(1100, 22)
        wrong_pair["identity_sha256"] = canonical_sha256(
            {key: value for key, value in wrong_pair.items() if key != "identity_sha256"}
        )
        wrong_pair_root = root / "wrong_but_in_axis_no_touch_pair"
        publish_artifact(
            wrong_pair_root,
            artifact_kind="SCIENTIFIC_V4_OBSERVATION_NO_TOUCH_AUDIT",
            payloads={
                "no_touch_receipt.json": json.dumps(wrong_pair, sort_keys=True) + "\n"
            },
            manifest_fields={
                "dynamic_source_lock_manifest_identity_sha256": source_manifest[
                    "identity_sha256"
                ],
                "dynamic_contract_identity_sha256": contract["identity_sha256"],
                "scientific_protocol_identity_sha256": protocol["identity_sha256"],
                "method_lock_identity_sha256": METHOD_LOCK_ID,
                "trace_plan_identity_sha256": plan["identity_sha256"],
                "trace_pool_identity_sha256": wrong_pair[
                    "trace_pool_identity_sha256"
                ],
                "confirmation_ordered_pair_axis_sha256": wrong_pair[
                    "confirmation_ordered_pair_axis_sha256"
                ],
                "pair": wrong_pair["pair"],
                "receipt_identity_sha256": wrong_pair["identity_sha256"],
            },
        )
        try:
            validate_no_touch_receipt(
                wrong_pair_root,
                contract=contract,
                source_manifest=source_manifest,
                protocol=protocol,
                plan=plan,
                trace_plan_file_sha256=sha256_file(plan_path),
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("a post-hoc alternate valid-axis no-touch pair was accepted")
        failed = dict(valid)
        failed["status"] = "FAIL_OBSERVATION_NO_TOUCH"
        failed["all_trace_arrays_bitwise_equal"] = False
        failed["identity_sha256"] = canonical_sha256(
            {key: value for key, value in failed.items() if key != "identity_sha256"}
        )
        failed_root = root / "failed_no_touch"
        publish_artifact(
            failed_root,
            artifact_kind="SCIENTIFIC_V4_OBSERVATION_NO_TOUCH_AUDIT",
            payloads={"no_touch_receipt.json": json.dumps(failed, sort_keys=True) + "\n"},
            manifest_fields={
                "dynamic_source_lock_manifest_identity_sha256": source_manifest[
                    "identity_sha256"
                ],
                "dynamic_contract_identity_sha256": contract["identity_sha256"],
                "scientific_protocol_identity_sha256": protocol["identity_sha256"],
                "method_lock_identity_sha256": METHOD_LOCK_ID,
                "trace_plan_identity_sha256": plan["identity_sha256"],
                "trace_pool_identity_sha256": failed["trace_pool_identity_sha256"],
                "confirmation_ordered_pair_axis_sha256": failed[
                    "confirmation_ordered_pair_axis_sha256"
                ],
                "pair": failed["pair"],
                "receipt_identity_sha256": failed["identity_sha256"],
            },
        )
        try:
            validate_no_touch_receipt(
                failed_root,
                contract=contract,
                source_manifest=source_manifest,
                protocol=protocol,
                plan=plan,
                trace_plan_file_sha256=sha256_file(plan_path),
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("failed no-touch gate was accepted")
        bare_path = root / "bare_self_attested_no_touch.json"
        write_json(bare_path, valid)
        try:
            validate_no_touch_receipt(
                bare_path,
                contract=contract,
                source_manifest=source_manifest,
                protocol=protocol,
                plan=plan,
                trace_plan_file_sha256=sha256_file(plan_path),
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("bare self-attested no-touch JSON was accepted")
        tests.append(
            "fixed_pair_manifest_bound_no_touch_and_failed_or_bare_receipt_stop"
        )

        prelabel_payload = {
            "schema_version": 1,
            "status": "E_PRELABEL_GATE_PASSED",
            "dynamic_contract_identity_sha256": contract["identity_sha256"],
            "scientific_protocol_identity_sha256": protocol["identity_sha256"],
            "method_lock_identity_sha256": METHOD_LOCK_ID,
            "trace_plan_identity_sha256": plan["identity_sha256"],
            "matched_Q_power_gate_pass": True,
            "matched_Q_power_gate_identity_sha256": protocol["pre_label_E_gates"]
            ["matched_Q_conditional_power_gate"]["identity_sha256"],
            "adaptive_predictable_null_audit_pass": True,
            "adaptive_predictable_null_audit_identity_sha256": protocol[
                "pre_label_E_gates"
            ]["adaptive_predictable_null_audit"]["identity_sha256"],
            "label_free_real_gate_pass": True,
            "E_confirmation_label_join_authorized": True,
            "labels_reviews_consensus_endpoint_or_external_representations_opened": False,
            "E_scores_csv_opened_hashed_statted_or_resolved": False,
        }
        prelabel_payload["identity_sha256"] = canonical_sha256(prelabel_payload)
        prelabel_root = root / "valid_prelable_receipt"
        publish_artifact(
            prelabel_root,
            artifact_kind="SCIENTIFIC_V4_E_PRELABEL_GATE_RECEIPT",
            payloads={
                "prelabel_gate_receipt.json": json.dumps(
                    prelabel_payload, sort_keys=True
                )
                + "\n"
            },
            manifest_fields={
                "dynamic_contract_identity_sha256": contract["identity_sha256"],
                "scientific_protocol_identity_sha256": protocol["identity_sha256"],
                "trace_plan_identity_sha256": plan["identity_sha256"],
                "receipt_identity_sha256": prelabel_payload["identity_sha256"],
                "passed": True,
            },
        )
        replayed_prelabel = load_prelable_receipt(
            prelabel_root, contract=contract, protocol=protocol, plan=plan
        )
        if replayed_prelabel != prelabel_payload:
            raise AssertionError("valid prelabel receipt did not replay exactly")
        tests.append("actual_prelabel_receipt_loader_replays_frozen_gate_conjunction")

        confirmation_axis = exact_pairs(plan, phases=("confirmation",))
        confirmation_axis_hash = canonical_sha256(
            [
                {"phase": phase, "global_seed": seed, "class_id": class_id}
                for phase, seed, class_id in confirmation_axis
            ]
        )

        def publish_synthetic_score_product(
            output: Path,
            *,
            product: str,
            artifact_kind: str,
            score_name: str,
            alert_name: str | None,
            trace_pool_identity: str = "8" * 64,
        ) -> None:
            buffer = io.StringIO()
            columns = ["phase", "global_seed", "class_id", score_name]
            if alert_name is not None:
                columns.append(alert_name)
            writer = csv.DictWriter(buffer, fieldnames=columns)
            writer.writeheader()
            for phase, seed, class_id in confirmation_axis:
                row: dict[str, Any] = {
                    "phase": phase,
                    "global_seed": seed,
                    "class_id": class_id,
                    score_name: 0.0,
                }
                if alert_name is not None:
                    row[alert_name] = 0
                writer.writerow(row)
            fields: dict[str, Any] = {
                "product": product,
                "dynamic_source_lock_manifest_identity_sha256": source_manifest[
                    "identity_sha256"
                ],
                "dynamic_contract_identity_sha256": contract["identity_sha256"],
                "scientific_protocol_identity_sha256": protocol["identity_sha256"],
                "method_lock_identity_sha256": METHOD_LOCK_ID,
                "trace_plan_identity_sha256": plan["identity_sha256"],
                "trace_pool_identity_sha256": trace_pool_identity,
                "calibration_artifact_identity_sha256": "7" * 64,
                "row_count": len(confirmation_axis),
                "ordered_pair_axis_sha256": confirmation_axis_hash,
                "endpoint_images_or_envelopes_opened": False,
                "labels_reviews_consensus_opened": False,
                "FID_Inception_DINO_CLIP_embeddings_or_external_distances_opened": False,
            }
            if product == "G_start":
                fields.update(
                    {
                        "mechanics_product_manifest_identity_sha256": "6" * 64,
                        "mechanics_product_manifest_file_sha256": "5" * 64,
                        "transition_innovation_or_eprocess_increment_opened": False,
                    }
                )
            payloads: dict[str, bytes | str] = {"scores.csv": buffer.getvalue()}
            if product != "G_start":
                tracks_buffer = io.BytesIO()
                np.savez(
                    tracks_buffer,
                    synthetic_internal_track=np.zeros((len(confirmation_axis), 1)),
                )
                payloads["internal_tracks.npz"] = tracks_buffer.getvalue()
            publish_artifact(
                output,
                artifact_kind=artifact_kind,
                payloads=payloads,
                manifest_fields=fields,
            )

        score_specs = {
            "B": (
                "SCIENTIFIC_V4_B_LABEL_FREE_PRODUCT",
                "B_persistence",
                "B_alarm",
            ),
            "E": (
                "SCIENTIFIC_V4_E_LABEL_FREE_PRODUCT",
                "E_blur_gated_running_max_log",
                "E_blur_gated_alarm",
            ),
            "E_no_gate": (
                "SCIENTIFIC_V4_E_no_gate_LABEL_FREE_PRODUCT",
                "E_no_state_gate_running_max_log",
                "E_no_state_gate_alarm",
            ),
            "E_first_hit_full_budget": (
                "SCIENTIFIC_V4_E_first_hit_full_budget_LABEL_FREE_PRODUCT",
                "E_first_hit_full_budget_running_max_log",
                "E_first_hit_full_budget_alarm",
            ),
            "G_start": (
                "SCIENTIFIC_V4_G_START_SCHEDULE_LABEL_FREE_PRODUCT",
                "G_start_schedule_diagnostic",
                None,
            ),
        }
        score_roots: dict[str, Path] = {}
        for product, (kind, score_name, alert_name) in score_specs.items():
            score_roots[product] = root / f"score_{product}"
            publish_synthetic_score_product(
                score_roots[product],
                product=product,
                artifact_kind=kind,
                score_name=score_name,
                alert_name=alert_name,
            )
        _, B_receipt = load_score_product(
            score_roots["B"],
            candidate=B_CANDIDATE,
            contract=contract,
            source_manifest=source_manifest,
            protocol=protocol,
            plan=plan,
        )
        _, E_receipt = load_score_product(
            score_roots["E"],
            candidate=E_CANDIDATE,
            contract=contract,
            source_manifest=source_manifest,
            protocol=protocol,
            plan=plan,
        )
        diagnostic_receipts: dict[str, dict[str, Any]] = {}
        for product in ("E_no_gate", "E_first_hit_full_budget", "G_start"):
            _, diagnostic_receipts[product] = load_diagnostic_product(
                score_roots[product],
                product=product,
                contract=contract,
                source_manifest=source_manifest,
                protocol=protocol,
                plan=plan,
            )
        validate_common_product_provenance(
            {"B": B_receipt, "E": E_receipt, **diagnostic_receipts}
        )
        substituted_root = root / "score_E_no_gate_substituted_pool"
        publish_synthetic_score_product(
            substituted_root,
            product="E_no_gate",
            artifact_kind=score_specs["E_no_gate"][0],
            score_name=score_specs["E_no_gate"][1],
            alert_name=score_specs["E_no_gate"][2],
            trace_pool_identity="4" * 64,
        )
        _, substituted_receipt = load_diagnostic_product(
            substituted_root,
            product="E_no_gate",
            contract=contract,
            source_manifest=source_manifest,
            protocol=protocol,
            plan=plan,
        )
        try:
            validate_common_product_provenance(
                {"E": E_receipt, "substituted": substituted_receipt}
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("cross-trace-pool diagnostic substitution was accepted")
        tests.append("actual_B_E_three_diagnostic_loaders_and_cross_product_provenance")

        consensus = root / "consensus_valid"
        write_synthetic_consensus(consensus, protocol=protocol, plan=plan)
        counts, aggregate_receipt = load_consensus_aggregate_only(
            consensus, contract=contract, protocol=protocol, plan=plan
        )
        stage_A = {
            "consensus_aggregate_receipt": aggregate_receipt,
            "aggregate_counts": counts,
        }
        label_rows, label_join = load_label_rows(
            consensus,
            stage_a_receipt=stage_A,
            protocol=protocol,
            plan=plan,
        )
        if len(label_rows) != 768 or label_join["row_labels_reproduce_stage_A_aggregate"] is not True:
            raise AssertionError("valid consensus aggregate/row replay failed")
        substituted = root / "consensus_substituted"
        write_synthetic_consensus(
            substituted, protocol=protocol, plan=plan, blur_seed_count=4
        )
        try:
            load_label_rows(
                substituted,
                stage_a_receipt=stage_A,
                protocol=protocol,
                plan=plan,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("Stage-B accepted a consensus root substituted after Stage A")
        inconsistent = root / "consensus_inconsistent"
        write_synthetic_consensus(
            inconsistent,
            protocol=protocol,
            plan=plan,
            inconsistent_aggregate=True,
        )
        inconsistent_counts, inconsistent_receipt = load_consensus_aggregate_only(
            inconsistent, contract=contract, protocol=protocol, plan=plan
        )
        try:
            load_label_rows(
                inconsistent,
                stage_a_receipt={
                    "consensus_aggregate_receipt": inconsistent_receipt,
                    "aggregate_counts": inconsistent_counts,
                },
                protocol=protocol,
                plan=plan,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("row labels inconsistent with Stage-A aggregate were accepted")
        tests.append("stage_A_B_consensus_lineage_and_row_count_reconciliation")

        frozen_loader_root = root / "synthetic_source_lock"
        frozen_sources = frozen_loader_root / "sources"
        frozen_sources.mkdir(parents=True)
        (frozen_sources / "observe_dit_blur_focused_eprocess_v1.py").write_text(
            "SCHEMA_VERSION = 1\nTOKEN = 'frozen-v1'\n", encoding="utf-8"
        )
        (frozen_sources / "observe_dit_blur_focused_eprocess_v2.py").write_text(
            "import observe_dit_blur_focused_eprocess as v1\n"
            "SCHEMA_VERSION = 2\nTOKEN = v1.TOKEN\n",
            encoding="utf-8",
        )
        import_name = "observe_dit_blur_focused_eprocess"
        previous_module = sys.modules.get(import_name)
        poison_module = ModuleType(import_name)
        poison_module.TOKEN = "LIVE_POISON"
        sys.modules[import_name] = poison_module
        try:
            frozen_v2 = _load_frozen_method_v2(frozen_loader_root)
            if frozen_v2.TOKEN != "frozen-v1" or sys.modules[import_name] is not poison_module:
                raise AssertionError("frozen v2 dependency was hijacked or alias not restored")
        finally:
            if previous_module is None:
                sys.modules.pop(import_name, None)
            else:
                sys.modules[import_name] = previous_module
        tests.append("frozen_v2_dependency_rejects_live_module_hijack")

        relevant = {
            name: module
            for name, module in list(sys.modules.items())
            if name in {"models", "download", "diffusion"}
            or name.startswith("diffusion.")
        }
        for name in relevant:
            sys.modules.pop(name, None)
        sys.modules["models"] = ModuleType("models")
        try:
            for guard in (
                reject_sampler_dit_modules,
                reject_extractor_dit_modules,
            ):
                try:
                    guard()
                except RuntimeError:
                    pass
                else:
                    raise AssertionError(
                        "malicious preloaded upstream model module escaped import guard"
                    )
        finally:
            sys.modules.pop("models", None)
            sys.modules.update(relevant)
        tests.append("sampler_and_extractor_preloaded_DiT_module_hijack_rejection")

    if len(exact_pairs(plan)) != 888:
        raise AssertionError("method trace axis must be 120 calibration + 768 confirmation")
    if len(exact_pairs(plan, phases=("confirmation",))) != 768:
        raise AssertionError("review endpoint axis must contain only 768 confirmation rows")
    tests.append("exact_888_method_and_768_review_axes")

    rng = np.random.default_rng(2026082819)
    draws = 250_000
    K = 0.2
    u = math.sqrt(2.0 * K)
    log_lr = u * rng.normal(size=draws) - K
    mean_e = float(np.mean(np.exp(log_lr)))
    if not 0.99 <= mean_e <= 1.01:
        raise AssertionError(f"synthetic P e-value calibration failed: {mean_e}")
    tests.append("synthetic_P_calibration")

    power = core.matched_q_power_reference(draws=100_000, seed=2026082808)
    if power["minimum_anytime_power"] < 0.30 or power["passes"] is not True:
        raise AssertionError("synthetic matched-Q power gate failed")
    tests.append("synthetic_matched_Q_power")

    confirmation_pairs = exact_pairs(plan, phases=("confirmation",))
    class_ids = np.asarray([row[2] for row in confirmation_pairs], dtype=np.int16)
    effective = np.asarray(core.EFFECTIVE_NONIDENTITY, dtype=np.uint8)
    applied = np.zeros((len(confirmation_pairs), 2, 9), dtype=np.float64)
    start = np.full((len(confirmation_pairs), 2), -1, dtype=np.int16)
    remaining = np.zeros((len(confirmation_pairs), 2), dtype=np.int16)
    for scale_index in range(2):
        eligible_indices = np.flatnonzero(effective[scale_index])
        first = int(eligible_indices[0])
        h = len(eligible_indices)
        start[:18, scale_index] = first
        remaining[:18, scale_index] = h
        applied[:18, scale_index, eligible_indices] = 2.0 / h
    mechanics = core.label_free_path_mechanics_audit(
        applied_K=applied,
        direction_reused=np.zeros_like(applied, dtype=np.bool_),
        start_time_index=start,
        start_remaining_effective_count=remaining,
        class_id=class_ids,
        effective_nonidentity=effective,
    )
    if mechanics["passes"] is not True or mechanics["sample_count"] != 768:
        raise AssertionError("complete multi-step confirmation mechanics should pass")
    collapsed = applied.copy()
    for scale_index in range(2):
        eligible_indices = np.flatnonzero(effective[scale_index])
        collapsed[:18, scale_index] = 0.0
        collapsed[:18, scale_index, eligible_indices[0]] = 2.0
    collapsed_result = core.label_free_path_mechanics_audit(
        applied_K=collapsed,
        direction_reused=np.zeros_like(collapsed, dtype=np.bool_),
        start_time_index=start,
        start_remaining_effective_count=remaining,
        class_id=class_ids,
        effective_nonidentity=effective,
    )
    if collapsed_result["passes"] is not False:
        raise AssertionError("one-step KL collapse escaped the v2 mechanics gate")
    tests.append("confirmation_768_v2_mechanics_pass_and_one_step_collapse_stop")

    bootstrap_codes = np.zeros((128, 6), dtype=np.int8)
    bootstrap_codes[64:] = 1
    bootstrap_reference = np.zeros((128, 6), dtype=np.float64)
    bootstrap_proposed = bootstrap_codes.astype(np.float64)
    bootstrap_result = paired_seed_cluster_bootstrap(
        bootstrap_codes,
        bootstrap_reference,
        bootstrap_proposed,
        plan["selected_classes"],
        draws=100_000,
        seed=2026082811,
        contrast_name="synthetic_frozen_order_statistic",
    )
    if (
        bootstrap_result["draws"] != 100_000
        or bootstrap_result["seed"] != 2026082811
        or bootstrap_result["lower_order_statistic_zero_based_index"] != 4_999
        or "no interpolation" not in bootstrap_result["quantile_method"]
        or bootstrap_result["passes"] is not True
    ):
        raise AssertionError("frozen paired cluster bootstrap contract changed")
    try:
        paired_seed_cluster_bootstrap(
            bootstrap_codes,
            bootstrap_reference,
            bootstrap_proposed,
            plan["selected_classes"],
            draws=99_999,
            seed=2026082811,
            contrast_name="forbidden_draw_override",
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("paired cluster bootstrap accepted a draw override")
    if (
        rollback_mechanism_authorized(
            E_minus_no_state_gate_pass=True,
            E_minus_prespecified_scalar_G_pass=True,
            E_minus_one_shot_pass=False,
        )
        is not True
        or rollback_mechanism_authorized(
            E_minus_no_state_gate_pass=True,
            E_minus_prespecified_scalar_G_pass=False,
            E_minus_one_shot_pass=True,
        )
        is not False
    ):
        raise AssertionError("one-shot incorrectly entered or rescued rollback conjunction")
    tests.append("bootstrap_100k_index4999_and_one_shot_non_authorizing_conjunction")

    conditional_codes = np.full((128, 6), -1, dtype=np.int8)
    conditional_codes[0, 0] = 1
    conditional_codes[1, 0] = 0
    # This mild row shares the exact schedule stratum but is not an eligible
    # bad/good observation and must not inflate informative seed support.
    conditional_codes[2, 0] = -1
    conditional_scores = np.zeros((128, 6), dtype=np.float64)
    conditional_scores[0, 0] = 1.0
    conditional_scores[1, 0] = 0.0
    conditional_scores[2, 0] = 999.0
    bottom_start = np.full((768, 2), -1, dtype=np.int16)
    bottom_h = np.zeros((768, 2), dtype=np.int16)
    conditional = schedule_exact_conditional_concordance(
        conditional_codes,
        conditional_scores,
        bottom_start,
        bottom_h,
        plan["selected_classes"],
        draws=100_000,
        seed=2026082815,
    )
    if (
        conditional["concordance_C"] != 1.0
        or conditional["exact_schedule_comparable_pair_denominator"] != 1
        or conditional["informative_distinct_global_seed_count"] != 2
        or conditional["informative_distinct_global_seeds"] != [1200, 1201]
        or conditional["has_pass_fail_gate"] is not False
        or conditional["used_for_rollback_authorization"] is not False
        or conditional["bootstrap"]["zero_denominator_replicates_imputed"] is not False
    ):
        raise AssertionError("schedule-exact descriptive diagnostic/mild exclusion changed")
    tests.append("schedule_exact_descriptive_concordance_excludes_mild_seed_support")

    frozen_K = np.zeros((len(confirmation_pairs), 2), dtype=np.float64)
    for scale_index in range(2):
        frozen_K[:18, scale_index] = 2.0 / int(
            np.sum(effective[scale_index])
        )
    mechanics_arrays = {
        "applied_K": applied,
        "start_time_index": start,
        "start_remaining_effective_count": remaining,
        "frozen_K_per_step_after_start": frozen_K,
        "direction_reused": np.zeros_like(applied, dtype=np.bool_),
        "class_id": class_ids,
        "effective_nonidentity": effective,
    }
    mechanics_axis = canonical_sha256(
        [
            {"phase": phase, "global_seed": seed, "class_id": class_id}
            for phase, seed, class_id in confirmation_pairs
        ]
    )
    mechanics_json = {
        **mechanics,
        "phase": "confirmation",
        "ordered_pair_axis_sha256": mechanics_axis,
        "calibration_thresholds_fitted_on_these_paths": False,
        "confirmation_labels_scores_endpoints_or_external_representations_opened": False,
        "decision_not_made_by_product_extractor": True,
    }
    mechanics_json["identity_sha256"] = canonical_sha256(mechanics_json)

    def publish_mechanics_artifact(
        output: Path, arrays: dict[str, np.ndarray]
    ) -> None:
        buffer = io.BytesIO()
        np.savez(buffer, **arrays)
        publish_artifact(
            output,
            artifact_kind="SCIENTIFIC_V4_E_MECHANICS_LABEL_FREE_PRODUCT",
            payloads={
                "internal_tracks.npz": buffer.getvalue(),
                "label_free_mechanics_audit.json": json.dumps(
                    mechanics_json, sort_keys=True
                )
                + "\n",
            },
            manifest_fields={
                "product": "E_mechanics",
                "dynamic_source_lock_manifest_identity_sha256": source_manifest[
                    "identity_sha256"
                ],
                "dynamic_contract_identity_sha256": contract["identity_sha256"],
                "scientific_protocol_identity_sha256": protocol["identity_sha256"],
                "method_lock_identity_sha256": METHOD_LOCK_ID,
                "trace_plan_identity_sha256": plan["identity_sha256"],
                "trace_pool_identity_sha256": "8" * 64,
                "calibration_artifact_identity_sha256": "7" * 64,
                "row_count": len(confirmation_pairs),
                "ordered_pair_axis_sha256": mechanics_axis,
                "endpoint_images_or_envelopes_opened": False,
                "labels_reviews_consensus_opened": False,
                "FID_Inception_DINO_CLIP_embeddings_or_external_distances_opened": False,
                "mechanics_track_array_records": {
                    name: {
                        "shape": list(value.shape),
                        "dtype": value.dtype.str,
                        "raw_sha256": sha256_array(value),
                    }
                    for name, value in mechanics_arrays.items()
                },
                "label_free_mechanics_audit_identity_sha256": mechanics_json[
                    "identity_sha256"
                ],
            },
        )

    with tempfile.TemporaryDirectory(prefix="v4-be-mechanics-firewall-") as raw:
        mechanics_root = Path(raw) / "valid"
        publish_mechanics_artifact(mechanics_root, mechanics_arrays)
        validate_E_mechanics_product_envelope(
            mechanics_root,
            contract=contract,
            source_manifest=source_manifest,
            protocol=protocol,
            plan=plan,
            method_core=core,
        )
        # An unlisted score file must fail even though the original manifest and
        # both listed payloads remain byte-identical.
        (mechanics_root / "scores.csv").write_text("score\n999\n", encoding="utf-8")
        try:
            validate_E_mechanics_product_envelope(
                mechanics_root,
                contract=contract,
                source_manifest=source_manifest,
                protocol=protocol,
                plan=plan,
                method_core=core,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("hidden unlisted score escaped E mechanics exact tree")

        poison_arrays = dict(mechanics_arrays)
        poison_arrays["component_log_e"] = np.zeros(
            (len(confirmation_pairs), 2, 9), dtype=np.float64
        )
        poison_root = Path(raw) / "poison_npz_member"
        publish_mechanics_artifact(poison_root, poison_arrays)
        try:
            validate_E_mechanics_product_envelope(
                poison_root,
                contract=contract,
                source_manifest=source_manifest,
                protocol=protocol,
                plan=plan,
                method_core=core,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("forbidden log-E array escaped E mechanics NPZ whitelist")
    tests.append("E_mechanics_exact_tree_array_hash_and_score_poison_firewall")

    for poison_name in (
        "endpoint.png",
        "labels.csv",
        "Inception_features.npz",
        "DINO.json",
        "FID.json",
        "CLIP_embedding.npy",
    ):
        try:
            reject_forbidden_method_name(poison_name, "synthetic input")
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"external method poison escaped: {poison_name}")
    tests.append("external_input_name_poison_rejection")

    if CANDIDATES != (B_CANDIDATE, E_CANDIDATE):
        raise AssertionError("co-primary family is not exactly B/E")
    return {
        "status": "PASS",
        "test_count": len(tests),
        "tests": tests,
        "synthetic_P_mean_e": mean_e,
        "synthetic_Q_minimum_anytime_power": power["minimum_anytime_power"],
        "method_lock_identity_sha256": METHOD_LOCK_ID,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
