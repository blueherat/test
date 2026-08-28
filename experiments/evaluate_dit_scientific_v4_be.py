#!/usr/bin/env python3
"""Three-barrier scientific-v4 evaluation for the frozen B/E family.

``prelabel`` opens only label-free E accounting and a no-touch audit.  It locks
the per-scale gate-open/K-utilization decision before any confirmation label.
``stage-a`` opens only final blind-label aggregates and never accepts a score
product argument.  ``stage-b`` first validates both receipts, then opens row
labels and only the authorized physically isolated B/E score products.

The co-primary tests are class-matched tie-aware AUC, one common complete
global-seed-block label-permutation stream, and Holm over exactly B/E.  The
hierarchical E-beyond-B and post-primary mechanism contrasts use paired
complete-seed cluster bootstrap lower confidence bounds; score identities are
never treated as exchangeable.
No endpoint image or FID/Inception/DINO/CLIP/embedding input is accepted.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

sys.dont_write_bytecode = True

import numpy as np

try:
    from .calibrate_dit_scientific_v4_be import validate_calibration
    from .dit_scientific_v4_be_contract import (
        B_CANDIDATE,
        B_SCORE,
        CALIBRATION_SEEDS,
        CANDIDATES,
        CHECKPOINTS,
        CONFIRMATION_SEEDS,
        DEFAULT_DYNAMIC_SOURCE_LOCK,
        E_ALERT,
        E_CANDIDATE,
        E_SCORE,
        METHOD_LOCK_ID,
        canonical_sha256,
        derive_pair_seed,
        exact_pairs,
        fixed_no_touch_pair,
        load_json,
        manifest_map,
        publish_artifact,
        require_directory,
        require_hex64,
        require_regular,
        sha256_array,
        sha256_file,
        validate_manifest_tree,
        validate_method_lock,
        validate_scientific_protocol,
        validate_trace_plan,
        without_identity,
    )
    from .sample_dit_scientific_v4_be_traces import TRACE_ARRAYS, load_source_lock
except ImportError:
    from calibrate_dit_scientific_v4_be import validate_calibration  # type: ignore
    from dit_scientific_v4_be_contract import (  # type: ignore
        B_CANDIDATE,
        B_SCORE,
        CALIBRATION_SEEDS,
        CANDIDATES,
        CHECKPOINTS,
        CONFIRMATION_SEEDS,
        DEFAULT_DYNAMIC_SOURCE_LOCK,
        E_ALERT,
        E_CANDIDATE,
        E_SCORE,
        METHOD_LOCK_ID,
        canonical_sha256,
        derive_pair_seed,
        exact_pairs,
        fixed_no_touch_pair,
        load_json,
        manifest_map,
        publish_artifact,
        require_directory,
        require_hex64,
        require_regular,
        sha256_array,
        sha256_file,
        validate_manifest_tree,
        validate_method_lock,
        validate_scientific_protocol,
        validate_trace_plan,
        without_identity,
    )
    from sample_dit_scientific_v4_be_traces import (  # type: ignore
        TRACE_ARRAYS,
        load_source_lock,
    )


EVALUATOR = "evaluate_dit_scientific_v4_be"
PRELABEL_KIND = "SCIENTIFIC_V4_E_PRELABEL_GATE_RECEIPT"
STAGE_A_KIND = "SCIENTIFIC_V4_B_E_STAGE_A_RECEIPT"
CONSENSUS_KIND = "EVENT_RICH_FINAL_CONSENSUS_LABEL_LOCK_V1"
CONSENSUS_STATUS = "FINAL_EXTERNAL_ENDPOINT_LABELS_LOCKED_BEFORE_ANY_B_E_INTERNAL_PRODUCT"
LABEL_COLUMNS = (
    "phase",
    "global_seed",
    "class_id",
    "final_severity",
    "blur_component",
)
SEVERITIES = ("clean_good", "mild_or_disputed", "clear_bad")
PRIMARY_PERMUTATION_DRAWS = 100_000
PRIMARY_PERMUTATION_SEED = 2026082801
INCREMENTAL_DRAWS = 100_000
INCREMENTAL_SEED = 2026082811
ABLATION_BOOTSTRAP_SEED = 2026082812
G_START_BOOTSTRAP_SEED = 2026082813
ONE_SHOT_BOOTSTRAP_SEED = 2026082814
SCHEDULE_EXACT_BOOTSTRAP_SEED = 2026082815
BOOTSTRAP_LOWER_ORDER_INDEX = 4_999
PERMUTATION_BATCH = 256
LOG10 = math.log(10.0)

# The score-free mechanics envelope is intentionally tiny.  These are the
# only predictable arrays needed to establish that the frozen E construction
# starts and spends its fixed information budget as specified.  In
# particular, no innovation, likelihood increment, running maximum, alarm, or
# endpoint-derived quantity is allowed into this pre-label artifact.
E_MECHANICS_ARRAYS = (
    "applied_K",
    "start_time_index",
    "start_remaining_effective_count",
    "frozen_K_per_step_after_start",
    "direction_reused",
    "class_id",
    "effective_nonidentity",
)
E_MECHANICS_EFFECTIVE_NONIDENTITY = np.asarray(
    ((0, 0, 0, 0, 1, 1, 1, 1, 1), (0, 1, 1, 1, 1, 1, 1, 1, 1)),
    dtype=np.uint8,
)

OVERALL_COUNT_FIELDS = (
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
)
PER_CLASS_COUNT_FIELDS = OVERALL_COUNT_FIELDS[:7]
LABEL_RECOMPUTABLE_COUNT_FIELDS = (
    "endpoint_count",
    "final_clean_good",
    "final_mild_or_disputed",
    "final_clear_bad",
    "final_blur_or_soft_fusion",
)


def verify_source(manifest: Mapping[str, Any]) -> None:
    by_name = {row["name"]: row for row in manifest["files"]}
    expected = "sources/evaluate_dit_scientific_v4_be.py"
    if by_name.get(expected, {}).get("sha256") != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("running evaluator differs from frozen source snapshot")


def validate_no_touch_receipt(
    root: Path,
    *,
    contract: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    protocol: Mapping[str, Any],
    plan: Mapping[str, Any],
    trace_plan_file_sha256: str,
) -> dict[str, Any]:
    """Validate a manifest-bound no-touch replay, not a self-asserted JSON file."""

    manifest, _ = validate_manifest_tree(root)
    if set(manifest_map(manifest)) != {"no_touch_receipt.json"}:
        raise RuntimeError("no-touch audit exact tree changed")
    payload = load_json(
        require_regular(root / "no_touch_receipt.json", "GPU no-touch receipt")
    )
    identity = payload.get("identity_sha256")
    pair = payload.get("pair")
    pair_key = None
    if isinstance(pair, dict):
        pair_key = (pair.get("phase"), pair.get("global_seed"), pair.get("class_id"))
    baseline_arrays = payload.get("baseline_trace_array_sha256")
    observed_arrays = payload.get("observed_trace_array_sha256")
    baseline_rng = payload.get("baseline_rng_boundary_sha256")
    observed_rng = payload.get("observed_rng_boundary_sha256")
    rng_keys = {
        "after_pair_seed_reset",
        "after_initial_noise",
        "after_250_full_2B_transition_draws",
    }
    expected_trace_axis = canonical_sha256(
        [
            {"phase": phase, "global_seed": seed, "class_id": class_id}
            for phase, seed, class_id in exact_pairs(plan)
        ]
    )
    expected_confirmation_axis = canonical_sha256(
        [
            {"phase": phase, "global_seed": seed, "class_id": class_id}
            for phase, seed, class_id in exact_pairs(plan, phases=("confirmation",))
        ]
    )
    if (
        manifest.get("artifact_kind") != "SCIENTIFIC_V4_OBSERVATION_NO_TOUCH_AUDIT"
        or manifest.get("dynamic_source_lock_manifest_identity_sha256")
        != source_manifest["identity_sha256"]
        or manifest.get("dynamic_contract_identity_sha256") != contract["identity_sha256"]
        or manifest.get("scientific_protocol_identity_sha256") != protocol["identity_sha256"]
        or manifest.get("method_lock_identity_sha256") != METHOD_LOCK_ID
        or manifest.get("trace_plan_identity_sha256") != plan["identity_sha256"]
        or manifest.get("trace_pool_identity_sha256")
        != payload.get("trace_pool_identity_sha256")
        or manifest.get("confirmation_ordered_pair_axis_sha256")
        != expected_confirmation_axis
        or manifest.get("receipt_identity_sha256") != identity
        or manifest.get("pair") != pair
        or
        canonical_sha256(without_identity(payload)) != identity
        or payload.get("dynamic_source_lock_manifest_identity_sha256")
        != source_manifest["identity_sha256"]
        or payload.get("dynamic_contract_identity_sha256") != contract["identity_sha256"]
        or payload.get("scientific_protocol_identity_sha256") != protocol["identity_sha256"]
        or payload.get("method_lock_identity_sha256") != METHOD_LOCK_ID
        or payload.get("trace_plan_identity_sha256") != plan["identity_sha256"]
        or payload.get("trace_plan_file_sha256") != trace_plan_file_sha256
        or require_hex64(
            payload.get("trace_pool_identity_sha256"), "no-touch trace-pool identity"
        )
        != payload.get("trace_pool_identity_sha256")
        or payload.get("trace_pool_ordered_pair_axis_sha256") != expected_trace_axis
        or payload.get("confirmation_ordered_pair_axis_sha256")
        != expected_confirmation_axis
        or payload.get("asset_identities") != contract["assets"]
        or payload.get("asset_identities_sha256") != canonical_sha256(contract["assets"])
        or pair_key != fixed_no_touch_pair(plan)
        or payload.get("derived_torch_seed") != derive_pair_seed(pair_key[1], pair_key[2])
        or not isinstance(baseline_arrays, dict)
        or set(baseline_arrays) != set(TRACE_ARRAYS)
        or baseline_arrays != observed_arrays
        or not all(
            require_hex64(value, f"baseline no-touch array hash {name}") == value
            for name, value in baseline_arrays.items()
        )
        or not isinstance(baseline_rng, dict)
        or set(baseline_rng) != rng_keys
        or baseline_rng != observed_rng
        or not all(
            require_hex64(value, f"baseline no-touch RNG hash {name}") == value
            for name, value in baseline_rng.items()
        )
        or payload.get("status") != "PASS_OBSERVATION_NO_TOUCH"
        or payload.get("all_trace_arrays_bitwise_equal") is not True
        or payload.get("endpoint_tensor_sha256_equal") is not True
        or payload.get("rng_boundaries_equal") is not True
        or payload.get("labels_reviews_external_representations_opened") is not False
    ):
        raise RuntimeError("operational no-touch audit did not pass")
    require_hex64(payload.get("baseline_endpoint_tensor_sha256"), "baseline endpoint hash")
    if payload.get("baseline_endpoint_tensor_sha256") != payload.get(
        "observed_endpoint_tensor_sha256"
    ):
        raise RuntimeError("no-touch endpoint hashes differ")
    return payload


def validate_E_mechanics_product_envelope(
    root: Path,
    *,
    contract: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    protocol: Mapping[str, Any],
    plan: Mapping[str, Any],
    method_core: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    root = require_directory(root, "score-free confirmation E mechanics product")
    expected_root_members = {
        "manifest.json",
        "completion.json",
        "internal_tracks.npz",
        "label_free_mechanics_audit.json",
    }
    root_members = list(root.iterdir())
    if (
        {path.name for path in root_members} != expected_root_members
        or len(root_members) != len(expected_root_members)
        or any(path.is_symlink() or not path.is_file() for path in root_members)
    ):
        raise RuntimeError("E mechanics physical root is not the exact flat score-free tree")
    # This checks the disk tree, every payload byte hash, the manifest identity,
    # and its completion envelope.  It therefore rejects an unlisted hidden
    # score file as well as a modified NPZ.
    manifest, completion = validate_manifest_tree(root)
    manifest_path = require_regular(root / "manifest.json", "E mechanics manifest")
    identity = manifest.get("identity_sha256")
    records = manifest_map(manifest)
    expected_axis = canonical_sha256(
        [
            {"phase": phase, "global_seed": seed, "class_id": class_id}
            for phase, seed, class_id in exact_pairs(plan, phases=("confirmation",))
        ]
    )
    if (
        canonical_sha256(without_identity(manifest)) != identity
        or completion
        != {
            "complete": True,
            "artifact_kind": "SCIENTIFIC_V4_E_MECHANICS_LABEL_FREE_PRODUCT",
            "manifest_identity_sha256": identity,
            "manifest_file_sha256": sha256_file(manifest_path),
        }
        or manifest.get("status") != "complete"
        or manifest.get("artifact_kind") != "SCIENTIFIC_V4_E_MECHANICS_LABEL_FREE_PRODUCT"
        or manifest.get("product") != "E_mechanics"
        or manifest.get("dynamic_source_lock_manifest_identity_sha256")
        != source_manifest["identity_sha256"]
        or manifest.get("dynamic_contract_identity_sha256") != contract["identity_sha256"]
        or manifest.get("scientific_protocol_identity_sha256") != protocol["identity_sha256"]
        or manifest.get("method_lock_identity_sha256") != METHOD_LOCK_ID
        or manifest.get("trace_plan_identity_sha256") != plan["identity_sha256"]
        or manifest.get("row_count") != len(exact_pairs(plan, phases=("confirmation",)))
        or manifest.get("ordered_pair_axis_sha256") != expected_axis
        or manifest.get("endpoint_images_or_envelopes_opened") is not False
        or manifest.get("labels_reviews_consensus_opened") is not False
        or manifest.get("FID_Inception_DINO_CLIP_embeddings_or_external_distances_opened")
        is not False
    ):
        raise RuntimeError("score-free confirmation E mechanics-product boundary changed")
    if set(records) != {
        "internal_tracks.npz",
        "label_free_mechanics_audit.json",
    }:
        raise RuntimeError("E mechanics product is not physically score-free")

    track_path = require_regular(root / "internal_tracks.npz", "E mechanics tracks")
    with zipfile.ZipFile(track_path, "r") as archive_zip:
        zip_names = archive_zip.namelist()
    expected_zip_names = [f"{name}.npy" for name in E_MECHANICS_ARRAYS]
    if zip_names != expected_zip_names or len(set(zip_names)) != len(zip_names):
        raise RuntimeError("E mechanics NPZ member whitelist/order changed")
    with np.load(track_path, allow_pickle=False) as archive:
        if tuple(archive.files) != E_MECHANICS_ARRAYS:
            raise RuntimeError("E mechanics NPZ exposes a forbidden array")
        arrays = {name: np.asarray(archive[name]) for name in E_MECHANICS_ARRAYS}
    row_count = len(exact_pairs(plan, phases=("confirmation",)))
    expected_specs = {
        "applied_K": ((row_count, 2, len(CHECKPOINTS)), np.dtype(np.float64)),
        "start_time_index": ((row_count, 2), np.dtype(np.int16)),
        "start_remaining_effective_count": ((row_count, 2), np.dtype(np.int16)),
        "frozen_K_per_step_after_start": ((row_count, 2), np.dtype(np.float64)),
        "direction_reused": ((row_count, 2, len(CHECKPOINTS)), np.dtype(np.bool_)),
        "class_id": ((row_count,), np.dtype(np.int16)),
        "effective_nonidentity": ((2, len(CHECKPOINTS)), np.dtype(np.uint8)),
    }
    array_records = manifest.get("mechanics_track_array_records")
    if not isinstance(array_records, dict) or set(array_records) != set(E_MECHANICS_ARRAYS):
        raise RuntimeError("E mechanics per-array records changed")
    for name in E_MECHANICS_ARRAYS:
        value = arrays[name]
        expected_shape, expected_dtype = expected_specs[name]
        record = array_records[name]
        if (
            value.shape != expected_shape
            or value.dtype != expected_dtype
            or not isinstance(record, dict)
            or set(record) != {"shape", "dtype", "raw_sha256"}
            or record.get("shape") != list(value.shape)
            or record.get("dtype") != value.dtype.str
            or require_hex64(record.get("raw_sha256"), f"E mechanics {name} raw hash")
            != sha256_array(value)
        ):
            raise RuntimeError(f"E mechanics array record/content changed: {name}")
    expected_classes = np.asarray(
        [class_id for _, _, class_id in exact_pairs(plan, phases=("confirmation",))],
        dtype=np.int16,
    )
    if (
        not np.array_equal(arrays["class_id"], expected_classes)
        or not np.array_equal(
            arrays["effective_nonidentity"], E_MECHANICS_EFFECTIVE_NONIDENTITY
        )
        or not np.isfinite(arrays["applied_K"]).all()
        or not np.isfinite(arrays["frozen_K_per_step_after_start"]).all()
    ):
        raise RuntimeError("E mechanics predictable axis/content changed")

    mechanics_path = require_regular(
        root / "label_free_mechanics_audit.json", "E mechanics audit"
    )
    mechanics_record = records["label_free_mechanics_audit.json"]
    if (
        mechanics_record.get("sha256") != sha256_file(mechanics_path)
        or mechanics_record.get("bytes") != mechanics_path.stat().st_size
    ):
        raise RuntimeError("E mechanics aggregate differs from its product manifest")
    mechanics = load_json(mechanics_path)
    mechanics_identity = mechanics.get("identity_sha256")
    if (
        canonical_sha256(without_identity(mechanics)) != mechanics_identity
        or manifest.get("label_free_mechanics_audit_identity_sha256")
        != mechanics_identity
        or mechanics.get("phase") != "confirmation"
        or mechanics.get("sample_count")
        != len(exact_pairs(plan, phases=("confirmation",)))
        or mechanics.get("ordered_pair_axis_sha256") != expected_axis
        or mechanics.get("calibration_thresholds_fitted_on_these_paths") is not False
        or mechanics.get(
            "confirmation_labels_scores_endpoints_or_external_representations_opened"
        )
        is not False
        or mechanics.get("labels_endpoint_images_external_representations_used") is not False
        or mechanics.get("quality_or_power_interpretation") is not False
    ):
        raise RuntimeError("label-free confirmation E mechanics audit changed")

    recomputed = method_core.label_free_path_mechanics_audit(
        applied_K=arrays["applied_K"],
        direction_reused=arrays["direction_reused"],
        start_time_index=arrays["start_time_index"],
        start_remaining_effective_count=arrays[
            "start_remaining_effective_count"
        ],
        class_id=arrays["class_id"],
        effective_nonidentity=arrays["effective_nonidentity"],
    )
    recomputed = {
        **recomputed,
        "phase": "confirmation",
        "ordered_pair_axis_sha256": expected_axis,
        "calibration_thresholds_fitted_on_these_paths": False,
        "confirmation_labels_scores_endpoints_or_external_representations_opened": False,
        "decision_not_made_by_product_extractor": True,
    }
    recomputed["identity_sha256"] = canonical_sha256(recomputed)
    if mechanics != recomputed:
        raise RuntimeError("E mechanics JSON does not exactly replay from predictable tracks")
    return manifest, mechanics, arrays


def prelabel(args: argparse.Namespace) -> None:
    source_lock = require_directory(args.source_lock, "v4 dynamic source lock")
    contract, source_manifest, _, method_core = load_source_lock(source_lock)
    verify_source(source_manifest)
    if contract.get("execution_ready") is not True:
        raise RuntimeError("dynamic source lock is not execution-ready")
    method_manifest, _ = validate_method_lock(Path(contract["method_lock"]["path"]))
    _, protocol = validate_scientific_protocol(Path(contract["scientific_protocol"]["path"]))
    plan = validate_trace_plan(args.trace_plan, protocol)
    E_mechanics_manifest, mechanics, _ = validate_E_mechanics_product_envelope(
        args.E_mechanics_product,
        contract=contract,
        source_manifest=source_manifest,
        protocol=protocol,
        plan=plan,
        method_core=method_core,
    )
    no_touch = validate_no_touch_receipt(
        args.no_touch_receipt,
        contract=contract,
        source_manifest=source_manifest,
        protocol=protocol,
        plan=plan,
        trace_plan_file_sha256=sha256_file(require_regular(args.trace_plan, "trace plan")),
    )
    if (
        E_mechanics_manifest.get("trace_pool_identity_sha256")
        != no_touch["trace_pool_identity_sha256"]
        or E_mechanics_manifest.get("ordered_pair_axis_sha256")
        != no_touch["confirmation_ordered_pair_axis_sha256"]
    ):
        raise RuntimeError("E mechanics product differs from the no-touch trace pool/axis")
    method_root = Path(contract["method_lock"]["path"])
    matched = load_json(
        require_regular(
            method_root / "matched_q_conditional_power_gate.json",
            "frozen matched-Q conditional power artifact",
        )
    )
    adaptive_null = load_json(
        require_regular(
            method_root / "adaptive_predictable_null_audit.json",
            "frozen adaptive predictable-null artifact",
        )
    )
    protocol_E_gates = protocol.get("pre_label_E_gates")
    if not isinstance(protocol_E_gates, dict):
        raise RuntimeError("scientific protocol lacks frozen pre-label E gates")
    expected_matched = protocol_E_gates.get("matched_Q_conditional_power_gate")
    expected_adaptive = protocol_E_gates.get("adaptive_predictable_null_audit")
    if not isinstance(expected_matched, dict) or not isinstance(expected_adaptive, dict):
        raise RuntimeError("scientific protocol E audit identities are malformed")
    matched_identity = canonical_sha256(matched)
    adaptive_null_identity = canonical_sha256(adaptive_null)
    matched_pass = bool(
        matched.get("passes") is True
        and float(matched.get("minimum_anytime_power", -1.0)) >= 0.30
        and float(
            matched.get(
                "dependence_robust_conditional_terminal_power_lower_bound", -1.0
            )
        )
        >= 0.30
        and matched.get("draws") == 400_000
        and matched.get("seed") == 2026082808
        and method_manifest.get("matched_q_power_gate_identity") == matched_identity
        and expected_matched.get("identity_sha256") == matched_identity
        and expected_matched.get("passes") is True
    )
    adaptive_null_pass = bool(
        adaptive_null.get("passes") is True
        and adaptive_null.get("draws") == 250_000
        and adaptive_null.get("seed") == 2026082821
        and float(adaptive_null.get("anytime_trigger_fraction_under_P", 1.0))
        <= 0.105
        and method_manifest.get("adaptive_null_audit_identity")
        == adaptive_null_identity
        and expected_adaptive.get("identity_sha256") == adaptive_null_identity
        and expected_adaptive.get("passes") is True
    )
    path_count = mechanics.get("sample_count")
    scale_rows = mechanics.get("scale_results")
    if type(path_count) is not int or not isinstance(scale_rows, list) or len(scale_rows) != 2:
        raise RuntimeError("E confirmation mechanics aggregate schema changed")
    expected_shifts = (1.0, 4.0)
    decisions = []
    for expected, row in zip(expected_shifts, scale_rows, strict=True):
        if not isinstance(row, dict) or float(row.get("heat_shift", -1)) != float(expected):
            raise RuntimeError("E mechanics heat-scale axis changed")
        starts = int(row["qualifying_started_path_count"])
        started_classes = int(row["qualifying_started_class_count"])
        complete_fraction = float(
            row[
                "fraction_started_paths_with_exact_complete_fixed_information_coverage"
            ]
        )
        reused_fraction = float(
            row["reused_direction_fraction_among_started_steps"]
        )
        decisions.append(
            {
                "heat_shift": float(expected),
                "confirmation_path_count": path_count,
                "qualifying_started_path_count": starts,
                "qualifying_started_class_count": started_classes,
                "complete_fixed_information_coverage_fraction": complete_fraction,
                "reused_direction_fraction_among_started_steps": reused_fraction,
                "started_paths_at_least_12": starts >= 12,
                "started_classes_at_least_3": started_classes >= 3,
                "complete_coverage_exactly_1": complete_fraction == 1.0,
                "reused_direction_fraction_at_most_0p01": reused_fraction <= 0.01,
                "core_mechanics_scale_pass": row.get(
                    "passes_non_degenerate_multi_step_conditions"
                )
                is True,
            }
        )
    complete_axis = path_count == 6 * len(CONFIRMATION_SEEDS)
    real_gate_pass = bool(
        complete_axis
        and mechanics.get("passes") is True
        and all(
            row["confirmation_path_count"] == path_count
            and row["started_paths_at_least_12"]
            and row["started_classes_at_least_3"]
            and row["complete_coverage_exactly_1"]
            and row["reused_direction_fraction_at_most_0p01"]
            and row["core_mechanics_scale_pass"]
            for row in decisions
        )
    )
    passed = bool(matched_pass and adaptive_null_pass and real_gate_pass)
    receipt = {
        "schema_version": 1,
        "status": "E_PRELABEL_GATE_PASSED" if passed else "E_PRELABEL_GATE_FAILED_STOP_E",
        "dynamic_contract_identity_sha256": contract["identity_sha256"],
        "scientific_protocol_identity_sha256": protocol["identity_sha256"],
        "method_lock_identity_sha256": METHOD_LOCK_ID,
        "trace_plan_identity_sha256": plan["identity_sha256"],
        "E_mechanics_product_manifest_identity_sha256": E_mechanics_manifest[
            "identity_sha256"
        ],
        "E_mechanics_product_manifest_file_sha256": sha256_file(
            require_regular(
                args.E_mechanics_product / "manifest.json", "E mechanics manifest"
            )
        ),
        "calibration_artifact_identity_sha256": E_mechanics_manifest[
            "calibration_artifact_identity_sha256"
        ],
        "E_mechanics_audit_identity_sha256": mechanics["identity_sha256"],
        "no_touch_receipt_identity_sha256": no_touch["identity_sha256"],
        "trace_pool_identity_sha256": no_touch["trace_pool_identity_sha256"],
        "confirmation_ordered_pair_axis_sha256": no_touch[
            "confirmation_ordered_pair_axis_sha256"
        ],
        "matched_Q_power_gate_pass": matched_pass,
        "matched_Q_power_gate_identity_sha256": matched_identity,
        "adaptive_predictable_null_audit_pass": adaptive_null_pass,
        "adaptive_predictable_null_audit_identity_sha256": adaptive_null_identity,
        "complete_768_path_confirmation_axis": complete_axis,
        "per_scale_decisions": decisions,
        "label_free_real_gate_pass": real_gate_pass,
        "E_confirmation_label_join_authorized": passed,
        "B_may_continue_independently_if_E_fails": True,
        "labels_reviews_consensus_endpoint_or_external_representations_opened": False,
        "E_scores_csv_opened_hashed_statted_or_resolved": False,
    }
    receipt["identity_sha256"] = canonical_sha256(receipt)
    publish_artifact(
        args.output,
        artifact_kind=PRELABEL_KIND,
        payloads={"prelabel_gate_receipt.json": json.dumps(receipt, indent=2, sort_keys=True) + "\n"},
        manifest_fields={
            "dynamic_contract_identity_sha256": contract["identity_sha256"],
            "scientific_protocol_identity_sha256": protocol["identity_sha256"],
            "trace_plan_identity_sha256": plan["identity_sha256"],
            "receipt_identity_sha256": receipt["identity_sha256"],
            "passed": passed,
        },
    )
    print(json.dumps({"status": receipt["status"], "passed": passed}, sort_keys=True))


def load_prelable_receipt(
    root: Path,
    *,
    contract: Mapping[str, Any],
    protocol: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    manifest, _ = validate_manifest_tree(root)
    receipt = load_json(require_regular(root / "prelabel_gate_receipt.json", "prelabel receipt"))
    protocol_E_gates = protocol.get("pre_label_E_gates", {})
    matched_expected = protocol_E_gates.get("matched_Q_conditional_power_gate", {})
    adaptive_expected = protocol_E_gates.get("adaptive_predictable_null_audit", {})
    replayed_pass = bool(
        receipt.get("matched_Q_power_gate_pass") is True
        and receipt.get("adaptive_predictable_null_audit_pass") is True
        and receipt.get("label_free_real_gate_pass") is True
    )
    if (
        manifest.get("artifact_kind") != PRELABEL_KIND
        or manifest.get("dynamic_contract_identity_sha256") != contract["identity_sha256"]
        or manifest.get("scientific_protocol_identity_sha256") != protocol["identity_sha256"]
        or manifest.get("trace_plan_identity_sha256") != plan["identity_sha256"]
    ):
        raise RuntimeError("prelabel receipt lineage changed")
    if (
        set(manifest_map(manifest)) != {"prelabel_gate_receipt.json"}
        or manifest.get("receipt_identity_sha256") != receipt.get("identity_sha256")
        or manifest.get("passed")
        is not (receipt.get("E_confirmation_label_join_authorized") is True)
        or canonical_sha256(without_identity(receipt)) != receipt.get("identity_sha256")
        or manifest.get("receipt_identity_sha256") != receipt["identity_sha256"]
        or receipt.get("dynamic_contract_identity_sha256") != contract["identity_sha256"]
        or receipt.get("scientific_protocol_identity_sha256") != protocol["identity_sha256"]
        or receipt.get("method_lock_identity_sha256") != METHOD_LOCK_ID
        or receipt.get("trace_plan_identity_sha256") != plan["identity_sha256"]
        or receipt.get("matched_Q_power_gate_identity_sha256")
        != matched_expected.get("identity_sha256")
        or receipt.get("adaptive_predictable_null_audit_identity_sha256")
        != adaptive_expected.get("identity_sha256")
        or receipt.get("E_confirmation_label_join_authorized") is not replayed_pass
        or receipt.get("status")
        != ("E_PRELABEL_GATE_PASSED" if replayed_pass else "E_PRELABEL_GATE_FAILED_STOP_E")
        or receipt.get("labels_reviews_consensus_endpoint_or_external_representations_opened")
        is not False
        or receipt.get("E_scores_csv_opened_hashed_statted_or_resolved") is not False
    ):
        raise RuntimeError("prelabel gate receipt changed")
    return receipt


def validate_aggregate(counts: Any, classes: Sequence[int]) -> dict[str, Any]:
    if not isinstance(counts, dict) or set(counts) != {"phase", "overall", "per_class"}:
        raise RuntimeError("consensus aggregate schema changed")
    if counts.get("phase") != "confirmation":
        raise RuntimeError("consensus aggregate is not confirmation phase")
    overall = counts.get("overall")
    per_class = counts.get("per_class")
    if (
        not isinstance(overall, dict)
        or set(overall) != set(OVERALL_COUNT_FIELDS)
        or any(type(overall[name]) is not int or overall[name] < 0 for name in OVERALL_COUNT_FIELDS)
        or not isinstance(per_class, dict)
        or set(per_class) != {str(value) for value in classes}
    ):
        raise RuntimeError("consensus aggregate counts are malformed")
    normalized = []
    for class_id in classes:
        row = per_class[str(class_id)]
        if (
            not isinstance(row, dict)
            or set(row) != set(PER_CLASS_COUNT_FIELDS)
            or any(type(row[name]) is not int or row[name] < 0 for name in PER_CLASS_COUNT_FIELDS)
        ):
            raise RuntimeError("per-class aggregate schema/order changed")
        if (
            row["final_clean_good"]
            + row["final_mild_or_disputed"]
            + row["final_clear_bad"]
            != row["endpoint_count"]
            or row["final_blur_or_soft_fusion"] > row["final_clear_bad"]
            or row["final_structural_non_blur"] > row["final_clear_bad"]
        ):
            raise RuntimeError("per-class aggregate counts are impossible")
        normalized.append({"class_id": class_id, **dict(row)})
    for field in PER_CLASS_COUNT_FIELDS:
        if sum(row[field] for row in normalized) != overall[field]:
            raise RuntimeError(f"per-class {field} does not sum to overall")
    if overall["endpoint_count"] != len(classes) * len(CONFIRMATION_SEEDS):
        raise RuntimeError("confirmation aggregate does not cover exact axis")
    return {"phase": "confirmation", "overall": dict(overall), "per_class": normalized}


def validate_consensus_envelope(
    root: Path,
    *,
    protocol: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    root = require_directory(root, "v4 final consensus export")
    manifest_path = require_regular(root / "manifest.json", "consensus manifest")
    completion_path = require_regular(root / "completion.json", "consensus completion")
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    identity_record = manifest.get("identity")
    if not isinstance(identity_record, dict):
        raise RuntimeError("consensus manifest lacks immutable identity record")
    identity = identity_record.get("identity_sha256")
    manifest_without = dict(manifest)
    manifest_envelope_identity = manifest_without.pop("manifest_identity_sha256", None)
    if (
        canonical_sha256(without_identity(identity_record)) != identity
        or manifest.get("identity_sha256") != identity
        or canonical_sha256(manifest_without) != manifest_envelope_identity
        or manifest.get("status") != "complete"
        or identity_record.get("artifact_kind") != CONSENSUS_KIND
        or identity_record.get("status") != CONSENSUS_STATUS
        or identity_record.get("phase") != "confirmation"
        or identity_record.get("event_protocol_identity_sha256") != protocol["identity_sha256"]
        or identity_record.get("anchor_plan_identity_sha256") != plan["identity_sha256"]
        or identity_record.get("row_count") != len(exact_pairs(plan, phases=("confirmation",)))
        or identity_record.get("endpoint_review_or_consensus_used_as_B_E_method_input")
        is not False
        or identity_record.get(
            "candidate_scores_features_trajectories_embeddings_thresholds_or_ranks_opened"
        )
        is not False
        or completion.get("complete") is not True
        or completion.get("identity_sha256") != identity
        or completion.get("manifest_identity_sha256") != manifest_envelope_identity
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
        or completion.get("file_count") != len(manifest.get("files", ()))
    ):
        raise RuntimeError("consensus export lineage/status changed")
    records = manifest_map(manifest)
    return root, manifest, identity_record, records


def load_consensus_aggregate_only(
    root: Path, *, contract: Mapping[str, Any], protocol: Mapping[str, Any], plan: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    # Only manifest/completion and this aggregate are opened. evaluation_labels.csv
    # is not statted, resolved, hashed, or passed to require_regular in Stage A.
    root, manifest, identity_record, records = validate_consensus_envelope(
        root, protocol=protocol, plan=plan
    )
    manifest_path = root / "manifest.json"
    aggregate_path = require_regular(root / "aggregate_counts.json", "consensus aggregate")
    counts = validate_aggregate(load_json(aggregate_path), tuple(plan["selected_classes"]))
    aggregate_record = records.get("aggregate_counts.json")
    if (
        not isinstance(aggregate_record, dict)
        or aggregate_record.get("sha256") != sha256_file(aggregate_path)
        or aggregate_record.get("bytes") != aggregate_path.stat().st_size
    ):
        raise RuntimeError("consensus manifest does not bind aggregate counts")
    receipt = {
        "manifest_identity_sha256": identity_record["identity_sha256"],
        "manifest_envelope_identity_sha256": manifest["manifest_identity_sha256"],
        "manifest_file_sha256": sha256_file(manifest_path),
        "aggregate_file_sha256": sha256_file(aggregate_path),
        "aggregate_record": aggregate_record,
        "row_count": identity_record.get("row_count"),
    }
    return counts, receipt


def stage_a(args: argparse.Namespace) -> None:
    source_lock = require_directory(args.source_lock, "v4 dynamic source lock")
    contract, source_manifest, _, _ = load_source_lock(source_lock)
    verify_source(source_manifest)
    if contract.get("execution_ready") is not True:
        raise RuntimeError("dynamic source lock is not execution-ready")
    _, protocol = validate_scientific_protocol(Path(contract["scientific_protocol"]["path"]))
    plan = validate_trace_plan(args.trace_plan, protocol)
    pre = load_prelable_receipt(
        args.prelabel_receipt, contract=contract, protocol=protocol, plan=plan
    )
    counts, consensus = load_consensus_aggregate_only(
        args.consensus_root, contract=contract, protocol=protocol, plan=plan
    )
    overall = counts["overall"]
    comparable = sum(
        row["final_blur_or_soft_fusion"] > 0 and row["final_clean_good"] > 0
        for row in counts["per_class"]
    )
    gate = {
        "blur_or_soft_fusion_clear_bad": overall["final_blur_or_soft_fusion"],
        "clean_good": overall["final_clean_good"],
        "comparable_classes": comparable,
        "minimum_blur_or_soft_fusion_clear_bad": 15,
        "minimum_clean_good": 60,
        "minimum_comparable_classes": 3,
    }
    event_pass = bool(
        gate["blur_or_soft_fusion_clear_bad"] >= 15
        and gate["clean_good"] >= 60
        and gate["comparable_classes"] >= 3
    )
    E_prepass = pre.get("E_confirmation_label_join_authorized") is True
    receipt = {
        "schema_version": 1,
        "status": "STAGE_A_EVENT_GATE_PASSED" if event_pass else "STAGE_A_EVENT_GATE_FAILED",
        "dynamic_contract_identity_sha256": contract["identity_sha256"],
        "scientific_protocol_identity_sha256": protocol["identity_sha256"],
        "trace_plan_identity_sha256": plan["identity_sha256"],
        "prelabel_receipt_identity_sha256": pre["identity_sha256"],
        "E_prelabel_mechanics_manifest_identity_sha256": pre[
            "E_mechanics_product_manifest_identity_sha256"
        ],
        "E_prelabel_mechanics_manifest_file_sha256": pre[
            "E_mechanics_product_manifest_file_sha256"
        ],
        "calibration_artifact_identity_sha256": pre[
            "calibration_artifact_identity_sha256"
        ],
        "trace_pool_identity_sha256": pre["trace_pool_identity_sha256"],
        "confirmation_ordered_pair_axis_sha256": pre[
            "confirmation_ordered_pair_axis_sha256"
        ],
        "consensus_aggregate_receipt": consensus,
        "aggregate_counts": counts,
        "event_gate": gate,
        "event_gate_pass": event_pass,
        "E_prelabel_gate_pass": E_prepass,
        "stage_B_authorization": {
            B_CANDIDATE: event_pass,
            E_CANDIDATE: bool(event_pass and E_prepass),
        },
        "E_prelabel_failure_does_not_block_independent_B_evaluation": True,
        "candidate_score_products_thresholds_or_external_representations_opened": False,
        "evaluation_labels_csv_opened_hashed_statted_or_resolved": False,
    }
    receipt["identity_sha256"] = canonical_sha256(receipt)
    publish_artifact(
        args.output,
        artifact_kind=STAGE_A_KIND,
        payloads={"stage_a_gate_receipt.json": json.dumps(receipt, indent=2, sort_keys=True) + "\n"},
        manifest_fields={
            "dynamic_contract_identity_sha256": contract["identity_sha256"],
            "scientific_protocol_identity_sha256": protocol["identity_sha256"],
            "trace_plan_identity_sha256": plan["identity_sha256"],
            "receipt_identity_sha256": receipt["identity_sha256"],
            "event_gate_pass": event_pass,
        },
    )
    print(json.dumps({"status": receipt["status"], "authorizations": receipt["stage_B_authorization"]}, sort_keys=True))


def load_stage_a(
    root: Path,
    *,
    contract: Mapping[str, Any],
    protocol: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    manifest, _ = validate_manifest_tree(root)
    receipt = load_json(require_regular(root / "stage_a_gate_receipt.json", "Stage-A receipt"))
    if (
        manifest.get("artifact_kind") != STAGE_A_KIND
        or set(manifest_map(manifest)) != {"stage_a_gate_receipt.json"}
        or manifest.get("dynamic_contract_identity_sha256") != contract["identity_sha256"]
        or manifest.get("scientific_protocol_identity_sha256") != protocol["identity_sha256"]
        or manifest.get("trace_plan_identity_sha256") != plan["identity_sha256"]
        or manifest.get("receipt_identity_sha256") != receipt.get("identity_sha256")
        or canonical_sha256(without_identity(receipt)) != receipt.get("identity_sha256")
        or receipt.get("dynamic_contract_identity_sha256") != contract["identity_sha256"]
        or receipt.get("scientific_protocol_identity_sha256") != protocol["identity_sha256"]
        or receipt.get("trace_plan_identity_sha256") != plan["identity_sha256"]
        or receipt.get("candidate_score_products_thresholds_or_external_representations_opened")
        is not False
        or receipt.get("evaluation_labels_csv_opened_hashed_statted_or_resolved") is not False
    ):
        raise RuntimeError("Stage-A receipt changed")
    counts = receipt.get("aggregate_counts")
    if not isinstance(counts, dict) or not isinstance(counts.get("per_class"), list):
        raise RuntimeError("Stage-A normalized aggregate is missing")
    raw_per_class: dict[str, Any] = {}
    for row in counts["per_class"]:
        if not isinstance(row, dict) or "class_id" not in row:
            raise RuntimeError("Stage-A per-class aggregate is malformed")
        class_id = row["class_id"]
        raw_per_class[str(class_id)] = {
            name: row.get(name) for name in PER_CLASS_COUNT_FIELDS
        }
    replayed = validate_aggregate(
        {
            "phase": counts.get("phase"),
            "overall": counts.get("overall"),
            "per_class": raw_per_class,
        },
        tuple(plan["selected_classes"]),
    )
    if replayed != counts:
        raise RuntimeError("Stage-A aggregate cannot be replayed exactly")
    overall = replayed["overall"]
    comparable = sum(
        row["final_blur_or_soft_fusion"] > 0 and row["final_clean_good"] > 0
        for row in replayed["per_class"]
    )
    expected_gate = {
        "blur_or_soft_fusion_clear_bad": overall["final_blur_or_soft_fusion"],
        "clean_good": overall["final_clean_good"],
        "comparable_classes": comparable,
        "minimum_blur_or_soft_fusion_clear_bad": 15,
        "minimum_clean_good": 60,
        "minimum_comparable_classes": 3,
    }
    event_pass = bool(
        expected_gate["blur_or_soft_fusion_clear_bad"] >= 15
        and expected_gate["clean_good"] >= 60
        and expected_gate["comparable_classes"] >= 3
    )
    expected_authorization = {
        B_CANDIDATE: event_pass,
        E_CANDIDATE: bool(event_pass and receipt.get("E_prelabel_gate_pass") is True),
    }
    if (
        receipt.get("event_gate") != expected_gate
        or receipt.get("event_gate_pass") is not event_pass
        or manifest.get("event_gate_pass") is not event_pass
        or receipt.get("stage_B_authorization") != expected_authorization
        or receipt.get("status")
        != ("STAGE_A_EVENT_GATE_PASSED" if event_pass else "STAGE_A_EVENT_GATE_FAILED")
    ):
        raise RuntimeError("Stage-A gate or authorization cannot be replayed")
    return receipt


def load_label_rows(
    root: Path,
    *,
    stage_a_receipt: Mapping[str, Any],
    protocol: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root, manifest, identity_record, records = validate_consensus_envelope(
        root, protocol=protocol, plan=plan
    )
    manifest_path = root / "manifest.json"
    stage_consensus = stage_a_receipt.get("consensus_aggregate_receipt")
    if not isinstance(stage_consensus, dict):
        raise RuntimeError("Stage-A receipt lacks a bound consensus aggregate")
    expected_consensus = {
        "manifest_identity_sha256": identity_record["identity_sha256"],
        "manifest_envelope_identity_sha256": manifest["manifest_identity_sha256"],
        "manifest_file_sha256": sha256_file(manifest_path),
    }
    if any(stage_consensus.get(name) != value for name, value in expected_consensus.items()):
        raise RuntimeError("row-level consensus root differs from Stage-A aggregate root")
    aggregate_record = records.get("aggregate_counts.json")
    if (
        not isinstance(aggregate_record, dict)
        or stage_consensus.get("aggregate_record") != aggregate_record
        or stage_consensus.get("aggregate_file_sha256") != aggregate_record.get("sha256")
        or stage_consensus.get("row_count") != identity_record["row_count"]
    ):
        raise RuntimeError("Stage-A aggregate receipt is not bound to this consensus manifest")
    path = require_regular(root / "evaluation_labels.csv", "locked evaluation labels")
    label_record = records.get("evaluation_labels.csv")
    if (
        not isinstance(label_record, dict)
        or sha256_file(path) != label_record.get("sha256")
        or path.stat().st_size != label_record.get("bytes")
    ):
        raise RuntimeError("evaluation label rows differ from consensus manifest")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != LABEL_COLUMNS:
            raise RuntimeError("evaluation label columns changed")
        for raw in reader:
            severity = raw["final_severity"]
            blur = int(raw["blur_component"])
            row = {
                "phase": raw["phase"],
                "global_seed": int(raw["global_seed"]),
                "class_id": int(raw["class_id"]),
                "final_severity": severity,
                "blur_component": blur,
            }
            if (
                severity not in SEVERITIES
                or blur not in (0, 1)
                or (blur == 1 and severity != "clear_bad")
            ):
                raise RuntimeError("evaluation label value is invalid")
            rows.append(row)
    expected = [
        ("confirmation", seed, class_id)
        for seed in CONFIRMATION_SEEDS
        for class_id in plan["selected_classes"]
    ]
    if [(row["phase"], row["global_seed"], row["class_id"]) for row in rows] != expected:
        raise RuntimeError("evaluation label axis/order changed")
    if len(rows) != identity_record["row_count"]:
        raise RuntimeError("evaluation label count differs from locked consensus identity")

    recomputed_overall = {name: 0 for name in LABEL_RECOMPUTABLE_COUNT_FIELDS}
    recomputed_by_class = {
        int(class_id): {name: 0 for name in LABEL_RECOMPUTABLE_COUNT_FIELDS}
        for class_id in plan["selected_classes"]
    }
    for row in rows:
        values = recomputed_by_class[row["class_id"]]
        for target in (recomputed_overall, values):
            target["endpoint_count"] += 1
            target[f"final_{row['final_severity']}"] += 1
            target["final_blur_or_soft_fusion"] += row["blur_component"]
    locked_counts = stage_a_receipt.get("aggregate_counts")
    if not isinstance(locked_counts, dict):
        raise RuntimeError("Stage-A receipt lacks locked aggregate counts")
    locked_per_class = {
        int(row["class_id"]): row for row in locked_counts.get("per_class", ())
    }
    if any(
        recomputed_overall[name] != locked_counts.get("overall", {}).get(name)
        for name in LABEL_RECOMPUTABLE_COUNT_FIELDS
    ) or any(
        recomputed_by_class[class_id][name] != locked_per_class.get(class_id, {}).get(name)
        for class_id in recomputed_by_class
        for name in LABEL_RECOMPUTABLE_COUNT_FIELDS
    ):
        raise RuntimeError("row labels do not reproduce the Stage-A aggregate counts")
    join_receipt = {
        **expected_consensus,
        "evaluation_labels_record": label_record,
        "evaluation_labels_file_sha256": label_record["sha256"],
        "row_count": len(rows),
        "recomputed_count_fields": list(LABEL_RECOMPUTABLE_COUNT_FIELDS),
        "row_labels_reproduce_stage_A_aggregate": True,
    }
    join_receipt["identity_sha256"] = canonical_sha256(join_receipt)
    return rows, join_receipt


def load_score_product(
    root: Path,
    *,
    candidate: str,
    contract: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    protocol: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> tuple[dict[tuple[str, int, int], tuple[float, int]], dict[str, Any]]:
    manifest, _ = validate_manifest_tree(root)
    if set(manifest_map(manifest)) != {"scores.csv", "internal_tracks.npz"}:
        raise RuntimeError(f"{candidate} score product physical payload set changed")
    product = "B" if candidate == B_CANDIDATE else "E"
    kind = f"SCIENTIFIC_V4_{product}_LABEL_FREE_PRODUCT"
    if (
        manifest.get("artifact_kind") != kind
        or manifest.get("product") != product
        or manifest.get("dynamic_source_lock_manifest_identity_sha256")
        != source_manifest["identity_sha256"]
        or manifest.get("dynamic_contract_identity_sha256") != contract["identity_sha256"]
        or manifest.get("scientific_protocol_identity_sha256") != protocol["identity_sha256"]
        or manifest.get("trace_plan_identity_sha256") != plan["identity_sha256"]
        or manifest.get("method_lock_identity_sha256") != METHOD_LOCK_ID
        or manifest.get("row_count")
        != len(exact_pairs(plan, phases=("confirmation",)))
        or manifest.get("ordered_pair_axis_sha256")
        != canonical_sha256(
            [
                {"phase": phase, "global_seed": seed, "class_id": class_id}
                for phase, seed, class_id in exact_pairs(
                    plan, phases=("confirmation",)
                )
            ]
        )
        or manifest.get("endpoint_images_or_envelopes_opened") is not False
        or manifest.get("labels_reviews_consensus_opened") is not False
        or manifest.get("FID_Inception_DINO_CLIP_embeddings_or_external_distances_opened")
        is not False
    ):
        raise RuntimeError(f"{candidate} product lineage/boundary changed")
    score_name = B_SCORE if candidate == B_CANDIDATE else E_SCORE
    alert_name = "B_alarm" if candidate == B_CANDIDATE else E_ALERT
    expected_columns = ("phase", "global_seed", "class_id", score_name, alert_name)
    path = require_regular(root / "scores.csv", f"{candidate} scores")
    rows: dict[tuple[str, int, int], tuple[float, int]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != expected_columns:
            raise RuntimeError(f"{candidate} score columns changed")
        for raw in reader:
            key = (raw["phase"], int(raw["global_seed"]), int(raw["class_id"]))
            value = float(raw[score_name])
            alert = int(raw[alert_name])
            if key in rows or not math.isfinite(value) or alert not in (0, 1):
                raise RuntimeError(f"{candidate} score row invalid")
            if candidate == E_CANDIDATE and alert != int(value >= LOG10):
                raise RuntimeError("E alert differs from fixed E>=10 anytime threshold")
            rows[key] = (value, alert)
    if tuple(rows) != exact_pairs(plan, phases=("confirmation",)):
        raise RuntimeError(f"{candidate} score axis/order changed")
    receipt = {
        "manifest_identity_sha256": manifest["identity_sha256"],
        "manifest_file_sha256": sha256_file(root / "manifest.json"),
        "scores_file_sha256": sha256_file(path),
        "product": product,
        "calibration_artifact_identity_sha256": manifest[
            "calibration_artifact_identity_sha256"
        ],
        "trace_pool_identity_sha256": manifest["trace_pool_identity_sha256"],
        "ordered_pair_axis_sha256": manifest["ordered_pair_axis_sha256"],
        "dynamic_source_lock_manifest_identity_sha256": manifest[
            "dynamic_source_lock_manifest_identity_sha256"
        ],
    }
    return rows, receipt


def positive(row: Mapping[str, Any]) -> bool:
    return row["final_severity"] == "clear_bad" and row["blur_component"] == 1


PRODUCT_PROVENANCE_FIELDS = (
    "dynamic_source_lock_manifest_identity_sha256",
    "trace_pool_identity_sha256",
    "calibration_artifact_identity_sha256",
    "ordered_pair_axis_sha256",
)


def validate_common_product_provenance(
    receipts: Mapping[str, Mapping[str, Any]],
    *,
    expected: Mapping[str, Any] | None = None,
) -> None:
    if not receipts:
        raise RuntimeError("no product receipts supplied for provenance validation")
    for field in PRODUCT_PROVENANCE_FIELDS:
        values = {row.get(field) for row in receipts.values()}
        if None in values or len(values) != 1:
            raise RuntimeError(f"products do not share provenance field {field}")
        if expected is not None and field in expected and next(iter(values)) != expected[field]:
            raise RuntimeError(f"product provenance differs from frozen {field}")


def auc_from_vectors(
    codes: np.ndarray, scores: np.ndarray, classes: Sequence[int]
) -> dict[str, Any]:
    numerator = 0.0
    denominator = 0
    per_class = []
    for slot, class_id in enumerate(classes):
        pos = scores[codes[:, slot] == 1, slot]
        neg = scores[codes[:, slot] == 0, slot]
        pairs = len(pos) * len(neg)
        credit = float(np.sum(pos[:, None] > neg[None, :]) + 0.5 * np.sum(pos[:, None] == neg[None, :]))
        numerator += credit
        denominator += pairs
        per_class.append(
            {
                "class_id": class_id,
                "positive_count": int(len(pos)),
                "clean_good_count": int(len(neg)),
                "pair_count": int(pairs),
                "auc": float(credit / pairs) if pairs else None,
            }
        )
    if denominator == 0:
        raise RuntimeError("class-matched AUC denominator is zero")
    return {"auc": float(numerator / denominator), "pair_count": int(denominator), "per_class": per_class}


def matrices(
    labels: Sequence[Mapping[str, Any]],
    products: Mapping[str, Mapping[tuple[str, int, int], tuple[float, int]]],
    classes: Sequence[int],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    by_key = {(row["global_seed"], row["class_id"]): row for row in labels}
    codes = np.full((len(CONFIRMATION_SEEDS), len(classes)), -1, dtype=np.int8)
    scores = {candidate: np.empty(codes.shape, dtype=np.float64) for candidate in products}
    alerts = {candidate: np.empty(codes.shape, dtype=np.uint8) for candidate in products}
    for seed_slot, seed in enumerate(CONFIRMATION_SEEDS):
        for class_slot, class_id in enumerate(classes):
            label = by_key[(seed, class_id)]
            if label["final_severity"] == "clean_good":
                codes[seed_slot, class_slot] = 0
            elif positive(label):
                codes[seed_slot, class_slot] = 1
            for candidate, rows in products.items():
                scores[candidate][seed_slot, class_slot], alerts[candidate][seed_slot, class_slot] = rows[
                    ("confirmation", seed, class_id)
                ]
    return codes, scores, alerts


def primary_permutations(
    codes: np.ndarray,
    score_matrices: Mapping[str, np.ndarray],
    classes: Sequence[int],
    *,
    draws: int,
) -> dict[str, Any]:
    if draws != PRIMARY_PERMUTATION_DRAWS:
        raise RuntimeError("primary permutation draws must equal the frozen 100000")
    def tie_order(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        order = np.argsort(values, kind="stable")
        sorted_values = values[order]
        starts = np.concatenate(
            (np.asarray([0], dtype=np.int64), np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1)
        )
        return order, starts

    def batch_credit(batch_codes: np.ndarray, order: np.ndarray, starts: np.ndarray) -> np.ndarray:
        ordered = batch_codes[:, order]
        pos = ordered == 1
        neg = ordered == 0
        pos_group = np.add.reduceat(pos, starts, axis=1).astype(np.float64)
        neg_group = np.add.reduceat(neg, starts, axis=1).astype(np.float64)
        neg_before = np.cumsum(neg_group, axis=1) - neg_group
        return np.sum(pos_group * (neg_before + 0.5 * neg_group), axis=1)

    prepared: dict[str, Any] = {}
    observed: dict[str, float] = {}
    for candidate, score in score_matrices.items():
        orders = []
        starts = []
        credit = 0.0
        denominator = 0
        for slot in range(len(classes)):
            order, start = tie_order(score[:, slot])
            orders.append(order)
            starts.append(start)
            credit += float(batch_credit(codes[:, slot][None, :], order, start)[0])
            denominator += int(np.sum(codes[:, slot] == 1) * np.sum(codes[:, slot] == 0))
        observed[candidate] = credit / denominator
        prepared[candidate] = {
            "orders": orders,
            "starts": starts,
            "observed_credit": credit,
            "denominator": denominator,
        }
    exceed = {candidate: 0 for candidate in score_matrices}
    rng = np.random.default_rng(PRIMARY_PERMUTATION_SEED)
    remaining = draws
    while remaining:
        count = min(PERMUTATION_BATCH, remaining)
        permutations = np.stack(
            [rng.permutation(len(CONFIRMATION_SEEDS)) for _ in range(count)]
        )
        for candidate, item in prepared.items():
            total = np.zeros(count, dtype=np.float64)
            for slot in range(len(classes)):
                total += batch_credit(
                    codes[permutations, slot], item["orders"][slot], item["starts"][slot]
                )
            exceed[candidate] += int(np.sum(total >= item["observed_credit"]))
        remaining -= count
    return {
        candidate: {
            "draws": draws,
            "exceedances": exceed[candidate],
            "raw_p_value": float((1 + exceed[candidate]) / (1 + draws)),
            "observed_auc": observed[candidate],
            "common_global_seed_block_permutation_stream": True,
        }
        for candidate in score_matrices
    }


def holm(raw: Mapping[str, float]) -> dict[str, float]:
    if set(raw) != set(CANDIDATES):
        raise RuntimeError("Holm family must be exactly B/E")
    order = sorted(CANDIDATES, key=lambda name: (raw[name], name))
    result: dict[str, float] = {}
    running = 0.0
    for index, name in enumerate(order):
        running = max(running, min(1.0, (len(order) - index) * raw[name]))
        result[name] = running
    return result


def operating(codes: np.ndarray, alerts: np.ndarray) -> dict[str, Any]:
    valid_positive = codes == 1
    valid_clean = codes == 0
    tp = int(np.sum((alerts == 1) & valid_positive))
    fp = int(np.sum((alerts == 1) & valid_clean))
    positives = int(np.sum(valid_positive))
    clean = int(np.sum(valid_clean))
    if positives == 0 or clean == 0:
        raise RuntimeError("operating point denominator is zero")
    return {
        "positive_count": positives,
        "clean_good_count": clean,
        "true_positive_count": tp,
        "false_positive_count": fp,
        "TPR": float(tp / positives),
        "FPR": float(fp / clean),
        "TPR_strictly_above_FPR": tp / positives > fp / clean,
    }


def paired_seed_cluster_bootstrap(
    codes: np.ndarray,
    reference: np.ndarray,
    proposed: np.ndarray,
    classes: Sequence[int],
    *,
    draws: int,
    seed: int,
    contrast_name: str,
) -> dict[str, Any]:
    """Paired nonparametric bootstrap over all complete six-class seed blocks."""

    if draws != INCREMENTAL_DRAWS:
        raise RuntimeError("paired cluster bootstrap draws must equal the frozen 100000")
    seed_count = len(CONFIRMATION_SEEDS)
    class_count = len(classes)
    expected_shape = (seed_count, class_count)
    if codes.shape != expected_shape or reference.shape != expected_shape or proposed.shape != expected_shape:
        raise RuntimeError("paired cluster bootstrap requires the exact 128x6 axis")
    observed_reference = auc_from_vectors(codes, reference, classes)["auc"]
    observed_proposed = auc_from_vectors(codes, proposed, classes)["auc"]
    observed_delta = observed_proposed - observed_reference

    def tie_groups(score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        order = np.argsort(score, kind="stable")
        ordered = score[order]
        starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
        return order, starts

    prepared: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for slot in range(class_count):
        reference_order, reference_starts = tie_groups(reference[:, slot])
        proposed_order, proposed_starts = tie_groups(proposed[:, slot])
        prepared.append(
            (reference_order, reference_starts, proposed_order, proposed_starts)
        )

    def weighted_credit(
        weights: np.ndarray, labels: np.ndarray, order: np.ndarray, starts: np.ndarray
    ) -> np.ndarray:
        ordered_weights = weights[:, order]
        ordered_labels = labels[order]
        positive = np.add.reduceat(
            ordered_weights * (ordered_labels == 1)[None, :], starts, axis=1
        )
        negative = np.add.reduceat(
            ordered_weights * (ordered_labels == 0)[None, :], starts, axis=1
        )
        negative_before = np.cumsum(negative, axis=1) - negative
        return np.sum(positive * (negative_before + 0.5 * negative), axis=1)

    rng = np.random.default_rng(seed)
    deltas = np.empty(draws, dtype=np.float64)
    invalid = 0
    offset = 0
    probabilities = np.full(seed_count, 1.0 / seed_count, dtype=np.float64)
    while offset < draws:
        count = min(PERMUTATION_BATCH, draws - offset)
        weights = rng.multinomial(seed_count, probabilities, size=count).astype(
            np.float64, copy=False
        )
        denominator = np.zeros(count, dtype=np.float64)
        reference_credit = np.zeros(count, dtype=np.float64)
        proposed_credit = np.zeros(count, dtype=np.float64)
        for slot, (
            reference_order,
            reference_starts,
            proposed_order,
            proposed_starts,
        ) in enumerate(prepared):
            labels = codes[:, slot]
            positive_weight = weights @ (labels == 1).astype(np.float64)
            negative_weight = weights @ (labels == 0).astype(np.float64)
            denominator += positive_weight * negative_weight
            reference_credit += weighted_credit(
                weights, labels, reference_order, reference_starts
            )
            proposed_credit += weighted_credit(
                weights, labels, proposed_order, proposed_starts
            )
        valid = denominator > 0.0
        batch_delta = np.full(count, -1.0, dtype=np.float64)
        batch_delta[valid] = (
            proposed_credit[valid] - reference_credit[valid]
        ) / denominator[valid]
        invalid += int(np.sum(~valid))
        deltas[offset : offset + count] = batch_delta
        offset += count
    # Frozen v4.2.1 convention: for exactly 100,000 replicates, the one-sided
    # 95% lower bound is the zero-based order statistic 4,999, with no
    # interpolation.
    lower = float(
        np.partition(deltas, BOOTSTRAP_LOWER_ORDER_INDEX)[
            BOOTSTRAP_LOWER_ORDER_INDEX
        ]
    )
    return {
        "contrast": contrast_name,
        "observed_reference_auc": observed_reference,
        "observed_proposed_auc": observed_proposed,
        "observed_DeltaAUC_proposed_minus_reference": observed_delta,
        "draws": draws,
        "seed": seed,
        "resampling_unit": "one complete global-seed block containing all six classes",
        "seed_block_count": seed_count,
        "one_sided_confidence_level": 0.95,
        "lower_confidence_bound_DeltaAUC": lower,
        "quantile_method": (
            "zero-based sorted order statistic index 4999 of 100000; "
            "no interpolation"
        ),
        "lower_order_statistic_zero_based_index": BOOTSTRAP_LOWER_ORDER_INDEX,
        "invalid_zero_denominator_replicates": invalid,
        "invalid_replicate_conservative_DeltaAUC": -1.0,
        "score_identity_swaps_used": False,
        "passes": bool(observed_delta > 0.0 and lower > 0.0),
    }


def schedule_exact_conditional_concordance(
    codes: np.ndarray,
    E_scores: np.ndarray,
    start_time_index: np.ndarray,
    start_remaining_effective_count: np.ndarray,
    classes: Sequence[int],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    """Descriptive bad-good concordance conditional on the exact start state.

    A stratum is ``(class,(T_1,h_1) or bottom,(T_4,h_4) or bottom)``.  The
    statistic pair-count-weights all comparable blur/fusion-bad versus
    clean-good pairs inside those strata.  It is descriptive only: it neither
    enters a gate nor proves a causal innovation effect.
    """

    if draws != INCREMENTAL_DRAWS:
        raise RuntimeError("schedule-exact bootstrap draws must equal frozen 100000")
    seed_count = len(CONFIRMATION_SEEDS)
    class_count = len(classes)
    if (
        codes.shape != (seed_count, class_count)
        or E_scores.shape != codes.shape
        or start_time_index.shape != (seed_count * class_count, 2)
        or start_remaining_effective_count.shape != (seed_count * class_count, 2)
        or start_time_index.dtype != np.int16
        or start_remaining_effective_count.dtype != np.int16
    ):
        raise RuntimeError("schedule-exact concordance requires exact 128x6 metadata")
    flat_codes = codes.reshape(-1)
    flat_scores = E_scores.reshape(-1)
    flat_classes = np.tile(np.asarray(classes, dtype=np.int16), seed_count)
    flat_seed_slots = np.repeat(np.arange(seed_count, dtype=np.int16), class_count)
    groups: dict[tuple[Any, ...], list[int]] = {}
    for index in range(len(flat_codes)):
        components: list[tuple[int, int] | None] = []
        for scale_index in range(2):
            start = int(start_time_index[index, scale_index])
            remaining = int(start_remaining_effective_count[index, scale_index])
            if start < 0:
                if remaining != 0:
                    raise RuntimeError("unstarted schedule component has nonzero h")
                components.append(None)
            else:
                components.append((start, remaining))
        key = (int(flat_classes[index]), components[0], components[1])
        groups.setdefault(key, []).append(index)

    informative: list[dict[str, Any]] = []
    informative_classes: set[int] = set()
    informative_seed_slots: set[int] = set()
    for key, raw_indices in groups.items():
        indices = np.asarray(raw_indices, dtype=np.int64)
        local_codes = flat_codes[indices]
        if not np.any(local_codes == 1) or not np.any(local_codes == 0):
            continue
        order = np.argsort(flat_scores[indices], kind="stable")
        sorted_scores = flat_scores[indices][order]
        starts = np.r_[
            0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1
        ]
        informative.append(
            {
                "key": key,
                "seed_slots": flat_seed_slots[indices],
                "codes": local_codes,
                "order": order,
                "tie_starts": starts,
            }
        )
        informative_classes.add(int(key[0]))
        eligible_indices = indices[(local_codes == 0) | (local_codes == 1)]
        informative_seed_slots.update(
            int(value) for value in flat_seed_slots[eligible_indices]
        )

    class_only_denominator = int(
        sum(
            int(np.sum(codes[:, slot] == 1))
            * int(np.sum(codes[:, slot] == 0))
            for slot in range(class_count)
        )
    )

    def statistics(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        denominator = np.zeros(weights.shape[0], dtype=np.float64)
        credit = np.zeros(weights.shape[0], dtype=np.float64)
        ties = np.zeros(weights.shape[0], dtype=np.float64)
        for group in informative:
            local_weights = weights[:, group["seed_slots"]]
            ordered_weights = local_weights[:, group["order"]]
            ordered_codes = group["codes"][group["order"]]
            tie_starts = group["tie_starts"]
            positive = np.add.reduceat(
                ordered_weights * (ordered_codes == 1)[None, :],
                tie_starts,
                axis=1,
            )
            negative = np.add.reduceat(
                ordered_weights * (ordered_codes == 0)[None, :],
                tie_starts,
                axis=1,
            )
            negative_before = np.cumsum(negative, axis=1) - negative
            denominator += np.sum(positive, axis=1) * np.sum(negative, axis=1)
            credit += np.sum(
                positive * (negative_before + 0.5 * negative), axis=1
            )
            ties += np.sum(positive * negative, axis=1)
        return credit, denominator, ties

    point_weights = np.ones((1, seed_count), dtype=np.float64)
    point_credit, point_denominator, point_ties = statistics(point_weights)
    denominator_value = int(point_denominator[0])
    if denominator_value == 0:
        point_concordance: float | None = None
        tie_rate: float | None = None
    else:
        point_concordance = float(point_credit[0] / point_denominator[0])
        tie_rate = float(point_ties[0] / point_denominator[0])

    rng = np.random.default_rng(seed)
    probabilities = np.full(seed_count, 1.0 / seed_count, dtype=np.float64)
    bootstrap = np.empty(draws, dtype=np.float64)
    bootstrap.fill(np.nan)
    offset = 0
    while offset < draws:
        count = min(PERMUTATION_BATCH, draws - offset)
        weights = rng.multinomial(seed_count, probabilities, size=count).astype(
            np.float64, copy=False
        )
        credit, denominator, _ = statistics(weights)
        valid = denominator > 0.0
        values = np.full(count, np.nan, dtype=np.float64)
        values[valid] = credit[valid] / denominator[valid]
        bootstrap[offset : offset + count] = values
        offset += count
    valid_bootstrap = np.sort(bootstrap[np.isfinite(bootstrap)])
    invalid_count = int(draws - len(valid_bootstrap))
    if len(valid_bootstrap):
        lower_index = max(0, math.ceil(0.025 * len(valid_bootstrap)) - 1)
        upper_index = max(0, math.ceil(0.975 * len(valid_bootstrap)) - 1)
        interval: dict[str, float | None] = {
            "lower_2p5_percent": float(valid_bootstrap[lower_index]),
            "upper_97p5_percent": float(valid_bootstrap[upper_index]),
        }
    else:
        lower_index = None
        upper_index = None
        interval = {"lower_2p5_percent": None, "upper_97p5_percent": None}
    return {
        "role": "descriptive_non_gating_schedule_exact_conditional_concordance",
        "stratum_definition": (
            "(class_id,(T_Delta1,h_Delta1) or bottom,"
            "(T_Delta4,h_Delta4) or bottom)"
        ),
        "bad_definition": "clear_bad with blur_or_soft_fusion component",
        "good_definition": "clean_good",
        "pair_weighting": "all within-stratum bad-good pairs, pair-count weighted",
        "tie_credit": 0.5,
        "concordance_C": point_concordance,
        "exact_schedule_comparable_pair_denominator": denominator_value,
        "class_only_comparable_pair_denominator": class_only_denominator,
        "class_only_pair_coverage": (
            float(denominator_value / class_only_denominator)
            if class_only_denominator
            else None
        ),
        "informative_exact_strata": len(informative),
        "informative_class_count": len(informative_classes),
        "informative_classes": sorted(informative_classes),
        "informative_distinct_global_seed_count": len(informative_seed_slots),
        "informative_distinct_global_seeds": [
            int(CONFIRMATION_SEEDS[slot]) for slot in sorted(informative_seed_slots)
        ],
        "tie_rate": tie_rate,
        "bootstrap": {
            "draws": draws,
            "seed": seed,
            "rng": f"numpy.default_rng(PCG64(seed={seed}))",
            "cluster_unit": "one complete global-seed block with all six classes",
            "confidence_level": 0.95,
            "interval": interval,
            "valid_replicate_count": int(len(valid_bootstrap)),
            "zero_denominator_replicate_count": invalid_count,
            "zero_denominator_replicate_fraction": float(invalid_count / draws),
            "zero_denominator_replicates_imputed": False,
            "quantile_rule": (
                "sort only finite positive-denominator replicates; bounds use "
                "ceil(p*n)-1 order indices without interpolation"
            ),
            "lower_order_index": lower_index,
            "upper_order_index": upper_index,
        },
        "has_pass_fail_gate": False,
        "used_for_rollback_authorization": False,
        "causal_innovation_alignment_claimed": False,
    }


def rollback_mechanism_authorized(
    *,
    E_minus_no_state_gate_pass: bool,
    E_minus_prespecified_scalar_G_pass: bool,
    E_minus_one_shot_pass: bool,
) -> bool:
    """Frozen v4.2.1 post-primary mechanism conjunction.

    The one-shot contrast is supplied so the non-authorizing role is explicit,
    but by protocol it cannot veto or rescue rollback authorization.
    """

    if any(
        type(value) is not bool
        for value in (
            E_minus_no_state_gate_pass,
            E_minus_prespecified_scalar_G_pass,
            E_minus_one_shot_pass,
        )
    ):
        raise TypeError("rollback mechanism decisions must be booleans")
    return E_minus_no_state_gate_pass and E_minus_prespecified_scalar_G_pass


def stage_b(args: argparse.Namespace) -> None:
    source_lock = require_directory(args.source_lock, "v4 dynamic source lock")
    contract, source_manifest, _, _ = load_source_lock(source_lock)
    verify_source(source_manifest)
    if contract.get("execution_ready") is not True:
        raise RuntimeError("dynamic source lock is not execution-ready")
    _, protocol = validate_scientific_protocol(Path(contract["scientific_protocol"]["path"]))
    plan = validate_trace_plan(args.trace_plan, protocol)
    stage = load_stage_a(args.stage_a_receipt, contract=contract, protocol=protocol, plan=plan)
    authorizations = stage["stage_B_authorization"]
    if not any(authorizations.values()):
        result = {
            "schema_version": 1,
            "status": "STAGE_A_FAILED_ALL_SCORE_PRODUCTS_UNOPENED",
            "dynamic_contract_identity_sha256": contract["identity_sha256"],
            "scientific_protocol_identity_sha256": protocol["identity_sha256"],
            "trace_plan_identity_sha256": plan["identity_sha256"],
            "stage_A_locked_receipt": stage,
            "candidate_results": {
                candidate: {"raw_p_value": 1.0, "Holm_adjusted_p_value": 1.0, "passed": False}
                for candidate in CANDIDATES
            },
            "score_products_opened": [],
            "evaluation_labels_rows_opened": False,
        }
        result["identity_sha256"] = canonical_sha256(result)
        publish_artifact(
            args.output,
            artifact_kind="SCIENTIFIC_V4_B_E_STAGE_B_RESULT",
            payloads={"evaluation.json": json.dumps(result, indent=2, sort_keys=True) + "\n"},
            manifest_fields={
                "dynamic_contract_identity_sha256": contract["identity_sha256"],
                "scientific_protocol_identity_sha256": protocol["identity_sha256"],
                "trace_plan_identity_sha256": plan["identity_sha256"],
                "result_identity_sha256": result["identity_sha256"],
            },
        )
        return
    labels, consensus_join = load_label_rows(
        args.consensus_root,
        stage_a_receipt=stage,
        protocol=protocol,
        plan=plan,
    )
    product_paths = {B_CANDIDATE: args.B_product, E_CANDIDATE: args.E_product}
    products: dict[str, Any] = {}
    product_receipts: dict[str, Any] = {}
    opened = []
    for candidate in CANDIDATES:
        if not authorizations[candidate]:
            # Do not inspect, resolve, stat, or stringify this path.
            continue
        products[candidate], product_receipts[candidate] = load_score_product(
            product_paths[candidate],
            candidate=candidate,
            contract=contract,
            source_manifest=source_manifest,
            protocol=protocol,
            plan=plan,
        )
        opened.append(candidate)
    classes = tuple(plan["selected_classes"])
    validate_common_product_provenance(
        product_receipts,
        expected={
            "dynamic_source_lock_manifest_identity_sha256": source_manifest[
                "identity_sha256"
            ],
            "trace_pool_identity_sha256": stage["trace_pool_identity_sha256"],
            "calibration_artifact_identity_sha256": stage[
                "calibration_artifact_identity_sha256"
            ],
            "ordered_pair_axis_sha256": stage[
                "confirmation_ordered_pair_axis_sha256"
            ],
        },
    )
    codes, scores, alerts = matrices(labels, products, classes)
    aucs = {candidate: auc_from_vectors(codes, value, classes) for candidate, value in scores.items()}
    permutation = primary_permutations(
        codes, scores, classes, draws=PRIMARY_PERMUTATION_DRAWS
    )
    raw_p = {
        candidate: permutation[candidate]["raw_p_value"] if candidate in permutation else 1.0
        for candidate in CANDIDATES
    }
    adjusted = holm(raw_p)
    results: dict[str, Any] = {}
    for candidate in CANDIDATES:
        if candidate not in products:
            results[candidate] = {
                "status": "GATED_OFF_SCORE_PRODUCT_UNOPENED",
                "raw_p_value": 1.0,
                "Holm_adjusted_p_value": adjusted[candidate],
                "passes_all_primary_quality_gates": False,
            }
            continue
        op = operating(codes, alerts[candidate])
        minimum_auc = 0.75 if candidate == B_CANDIDATE else 0.70
        alert_positive_gate = (
            True if candidate == B_CANDIDATE else op["true_positive_count"] >= 3
        )
        passed = bool(
            aucs[candidate]["auc"] >= minimum_auc
            and adjusted[candidate] < 0.05
            and op["TPR_strictly_above_FPR"]
            and alert_positive_gate
        )
        results[candidate] = {
            "status": "PRIMARY_TEST_COMPLETED",
            "auc": aucs[candidate],
            "permutation": permutation[candidate],
            "raw_p_value": raw_p[candidate],
            "Holm_adjusted_p_value": adjusted[candidate],
            "operating_point": op,
            "minimum_auc": minimum_auc,
            "E_positive_alert_count_at_least_3": (
                alert_positive_gate if candidate == E_CANDIDATE else None
            ),
            "passes_all_primary_quality_gates": passed,
        }
    incremental: dict[str, Any]
    if results[E_CANDIDATE]["passes_all_primary_quality_gates"]:
        incremental = paired_seed_cluster_bootstrap(
            codes,
            scores[B_CANDIDATE],
            scores[E_CANDIDATE],
            classes,
            draws=INCREMENTAL_DRAWS,
            seed=INCREMENTAL_SEED,
            contrast_name="E_minus_B",
        )
        incremental["hierarchical_test_authorized_by_E_primary_pass"] = True
    else:
        incremental = {
            "status": "NOT_EVALUATED_BECAUSE_E_DID_NOT_PASS_ALL_PRIMARY_GATES",
            "hierarchical_test_authorized_by_E_primary_pass": False,
            "passes": False,
        }
    result = {
        "schema_version": 1,
        "status": "STAGE_B_COMPLETE",
        "dynamic_contract_identity_sha256": contract["identity_sha256"],
        "scientific_protocol_identity_sha256": protocol["identity_sha256"],
        "trace_plan_identity_sha256": plan["identity_sha256"],
        "stage_A_locked_receipt": stage,
        "consensus_row_join_receipt": consensus_join,
        "candidate_results": results,
        "multiple_testing": {
            "family_exactly": list(CANDIDATES),
            "raw_p_values": raw_p,
            "Holm_adjusted_p_values": adjusted,
            "gated_off_candidate_raw_p_fixed_to_1": True,
        },
        "E_beyond_B_incremental": incremental,
        "score_products_opened": opened,
        "score_product_receipts": product_receipts,
        "E_no_gate_product_opened": False,
        "external_FID_Inception_DINO_CLIP_embedding_inputs_opened": False,
        "alpha_0p10_semantics": "overall anytime P-trigger budget, not clean-good conditional FPR",
        "scope": "conditional six-class blur-enriched external evaluation population",
    }
    result.update(
        {
            "dynamic_contract_identity_sha256": contract["identity_sha256"],
            "scientific_protocol_identity_sha256": protocol["identity_sha256"],
            "trace_plan_identity_sha256": plan["identity_sha256"],
        }
    )
    result["identity_sha256"] = canonical_sha256(result)
    publish_artifact(
        args.output,
        artifact_kind="SCIENTIFIC_V4_B_E_STAGE_B_RESULT",
        payloads={"evaluation.json": json.dumps(result, indent=2, sort_keys=True) + "\n"},
        manifest_fields={
            "dynamic_contract_identity_sha256": contract["identity_sha256"],
            "scientific_protocol_identity_sha256": protocol["identity_sha256"],
            "trace_plan_identity_sha256": plan["identity_sha256"],
            "result_identity_sha256": result["identity_sha256"],
        },
    )
    print(json.dumps({"status": result["status"], "opened": opened}, sort_keys=True))


def load_stage_b_result(
    root: Path, *, contract: Mapping[str, Any], protocol: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, Any]:
    manifest, _ = validate_manifest_tree(root)
    result = load_json(require_regular(root / "evaluation.json", "Stage-B result"))
    if (
        manifest.get("artifact_kind") != "SCIENTIFIC_V4_B_E_STAGE_B_RESULT"
        or manifest.get("dynamic_contract_identity_sha256") != contract["identity_sha256"]
        or manifest.get("scientific_protocol_identity_sha256") != protocol["identity_sha256"]
        or manifest.get("trace_plan_identity_sha256") != plan["identity_sha256"]
        or manifest.get("result_identity_sha256") != result.get("identity_sha256")
        or canonical_sha256(without_identity(result)) != result.get("identity_sha256")
        or result.get("dynamic_contract_identity_sha256") != contract["identity_sha256"]
        or result.get("scientific_protocol_identity_sha256") != protocol["identity_sha256"]
        or result.get("trace_plan_identity_sha256") != plan["identity_sha256"]
    ):
        raise RuntimeError("Stage-B result lineage changed")
    return result


DIAGNOSTIC_PRODUCTS = {
    "E_no_gate": {
        "artifact_kind": "SCIENTIFIC_V4_E_no_gate_LABEL_FREE_PRODUCT",
        "score": "E_no_state_gate_running_max_log",
        "alert": "E_no_state_gate_alarm",
    },
    "E_first_hit_full_budget": {
        "artifact_kind": "SCIENTIFIC_V4_E_first_hit_full_budget_LABEL_FREE_PRODUCT",
        "score": "E_first_hit_full_budget_running_max_log",
        "alert": "E_first_hit_full_budget_alarm",
    },
    "G_start": {
        "artifact_kind": "SCIENTIFIC_V4_G_START_SCHEDULE_LABEL_FREE_PRODUCT",
        "score": "G_start_schedule_diagnostic",
        "alert": None,
    },
}


def load_diagnostic_product(
    root: Path,
    *,
    product: str,
    contract: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    protocol: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> tuple[dict[tuple[str, int, int], tuple[float, int]], dict[str, Any]]:
    spec = DIAGNOSTIC_PRODUCTS.get(product)
    if spec is None:
        raise RuntimeError("unknown frozen diagnostic product")
    manifest, _ = validate_manifest_tree(root)
    expected_payloads = (
        {"scores.csv"}
        if product == "G_start"
        else {"scores.csv", "internal_tracks.npz"}
    )
    if set(manifest_map(manifest)) != expected_payloads:
        raise RuntimeError(f"{product} diagnostic physical payload set changed")
    if (
        manifest.get("artifact_kind") != spec["artifact_kind"]
        or manifest.get("product") != product
        or manifest.get("dynamic_source_lock_manifest_identity_sha256")
        != source_manifest["identity_sha256"]
        or manifest.get("dynamic_contract_identity_sha256") != contract["identity_sha256"]
        or manifest.get("scientific_protocol_identity_sha256") != protocol["identity_sha256"]
        or manifest.get("trace_plan_identity_sha256") != plan["identity_sha256"]
        or manifest.get("method_lock_identity_sha256") != METHOD_LOCK_ID
        or manifest.get("row_count")
        != len(exact_pairs(plan, phases=("confirmation",)))
        or manifest.get("ordered_pair_axis_sha256")
        != canonical_sha256(
            [
                {"phase": phase, "global_seed": seed, "class_id": class_id}
                for phase, seed, class_id in exact_pairs(
                    plan, phases=("confirmation",)
                )
            ]
        )
        or manifest.get("endpoint_images_or_envelopes_opened") is not False
        or manifest.get("labels_reviews_consensus_opened") is not False
        or manifest.get("FID_Inception_DINO_CLIP_embeddings_or_external_distances_opened")
        is not False
    ):
        raise RuntimeError(f"{product} diagnostic product lineage/boundary changed")
    path = require_regular(root / "scores.csv", f"{product} diagnostic scores")
    columns = ("phase", "global_seed", "class_id", str(spec["score"]))
    if spec["alert"] is not None:
        columns += (str(spec["alert"]),)
    rows: dict[tuple[str, int, int], tuple[float, int]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != columns:
            raise RuntimeError(f"{product} diagnostic columns changed")
        for raw in reader:
            key = (raw["phase"], int(raw["global_seed"]), int(raw["class_id"]))
            score = float(raw[columns[3]])
            alert = int(raw[columns[4]]) if len(columns) == 5 else 0
            if key in rows or not math.isfinite(score) or alert not in (0, 1):
                raise RuntimeError(f"{product} diagnostic row invalid")
            if spec["alert"] is not None and alert != int(score >= LOG10):
                raise RuntimeError(f"{product} alert differs from E>=10")
            rows[key] = (score, alert)
    expected = exact_pairs(plan, phases=("confirmation",))
    if tuple(rows) != expected:
        raise RuntimeError(f"{product} diagnostic axis/order changed")
    receipt = {
        "manifest_identity_sha256": manifest["identity_sha256"],
        "manifest_file_sha256": sha256_file(root / "manifest.json"),
        "scores_file_sha256": sha256_file(path),
        "product": product,
        "dynamic_source_lock_manifest_identity_sha256": manifest[
            "dynamic_source_lock_manifest_identity_sha256"
        ],
        "trace_pool_identity_sha256": manifest["trace_pool_identity_sha256"],
        "calibration_artifact_identity_sha256": manifest[
            "calibration_artifact_identity_sha256"
        ],
        "ordered_pair_axis_sha256": manifest["ordered_pair_axis_sha256"],
    }
    if product == "G_start":
        receipt["mechanics_product_manifest_identity_sha256"] = manifest.get(
            "mechanics_product_manifest_identity_sha256"
        )
        receipt["mechanics_product_manifest_file_sha256"] = manifest.get(
            "mechanics_product_manifest_file_sha256"
        )
        if (
            receipt["mechanics_product_manifest_identity_sha256"] is None
            or receipt["mechanics_product_manifest_file_sha256"] is None
            or manifest.get("transition_innovation_or_eprocess_increment_opened")
            is not False
        ):
            raise RuntimeError("G_start is not bound to score-free frozen start metadata")
    return rows, receipt


def ablation(args: argparse.Namespace) -> None:
    """Post-primary isolated mechanism suite; no ablation can rescue E.

    This stage is hierarchical and cannot rescue E.  It is the only evaluator
    stage allowed to open no-gate, one-shot, and innovation-free G_start.
    """

    source_lock = require_directory(args.source_lock, "v4 dynamic source lock")
    contract, source_manifest, _, method_core = load_source_lock(source_lock)
    verify_source(source_manifest)
    if contract.get("execution_ready") is not True:
        raise RuntimeError("dynamic source lock is not execution-ready")
    _, protocol = validate_scientific_protocol(Path(contract["scientific_protocol"]["path"]))
    plan = validate_trace_plan(args.trace_plan, protocol)
    stage_result = load_stage_b_result(
        args.stage_b_result, contract=contract, protocol=protocol, plan=plan
    )
    E_primary = stage_result["candidate_results"][E_CANDIDATE][
        "passes_all_primary_quality_gates"
    ]
    incremental = stage_result["E_beyond_B_incremental"].get("passes") is True
    if not (E_primary and incremental):
        result = {
            "schema_version": 1,
            "status": "ABLATION_NOT_OPENED_HIERARCHICAL_PREREQUISITE_FAILED",
            "E_primary_pass": bool(E_primary),
            "E_beyond_B_incremental_pass": bool(incremental),
            "E_no_gate_product_opened": False,
            "E_first_hit_full_budget_product_opened": False,
            "G_start_product_opened": False,
            "B_gate_mechanism_claim_pass": False,
            "distributed_path_claim_pass": False,
            "E_outperforms_preregistered_one_dimensional_start_summary_pass": False,
            "G_start_is_not_a_schedule_exact_or_causal_control": True,
            "rollback_authorized": False,
        }
    else:
        stage_A_locked = stage_result.get("stage_A_locked_receipt")
        if not isinstance(stage_A_locked, dict):
            raise RuntimeError("Stage-B result lacks its exact Stage-A lineage")
        labels, consensus_join = load_label_rows(
            args.consensus_root,
            stage_a_receipt=stage_A_locked,
            protocol=protocol,
            plan=plan,
        )
        if consensus_join != stage_result.get("consensus_row_join_receipt"):
            raise RuntimeError("ablation consensus rows differ from Stage-B rows")
        E_rows, E_receipt = load_score_product(
            args.E_product,
            candidate=E_CANDIDATE,
            contract=contract,
            source_manifest=source_manifest,
            protocol=protocol,
            plan=plan,
        )
        expected_E = stage_result["score_product_receipts"][E_CANDIDATE][
            "manifest_identity_sha256"
        ]
        if E_receipt["manifest_identity_sha256"] != expected_E:
            raise RuntimeError("ablation E product differs from Stage-B E product")
        no_gate_rows, no_gate_receipt = load_diagnostic_product(
            args.E_no_gate_product,
            product="E_no_gate",
            contract=contract,
            source_manifest=source_manifest,
            protocol=protocol,
            plan=plan,
        )
        one_shot_rows, one_shot_receipt = load_diagnostic_product(
            args.E_first_hit_full_budget_product,
            product="E_first_hit_full_budget",
            contract=contract,
            source_manifest=source_manifest,
            protocol=protocol,
            plan=plan,
        )
        G_start_rows, G_start_receipt = load_diagnostic_product(
            args.G_start_product,
            product="G_start",
            contract=contract,
            source_manifest=source_manifest,
            protocol=protocol,
            plan=plan,
        )
        diagnostics = {
            "E_no_gate": no_gate_receipt,
            "E_first_hit_full_budget": one_shot_receipt,
            "G_start": G_start_receipt,
        }
        validate_common_product_provenance(
            {E_CANDIDATE: E_receipt, **diagnostics},
            expected={
                "dynamic_source_lock_manifest_identity_sha256": source_manifest[
                    "identity_sha256"
                ],
                "trace_pool_identity_sha256": stage_A_locked[
                    "trace_pool_identity_sha256"
                ],
                "calibration_artifact_identity_sha256": stage_A_locked[
                    "calibration_artifact_identity_sha256"
                ],
                "ordered_pair_axis_sha256": stage_A_locked[
                    "confirmation_ordered_pair_axis_sha256"
                ],
            },
        )
        if (
            G_start_receipt["mechanics_product_manifest_identity_sha256"]
            != stage_A_locked["E_prelabel_mechanics_manifest_identity_sha256"]
            or G_start_receipt["mechanics_product_manifest_file_sha256"]
            != stage_A_locked["E_prelabel_mechanics_manifest_file_sha256"]
        ):
            raise RuntimeError("G_start differs from the prelabel mechanics start metadata")
        mechanics_manifest, _, mechanics_arrays = validate_E_mechanics_product_envelope(
            args.E_mechanics_product,
            contract=contract,
            source_manifest=source_manifest,
            protocol=protocol,
            plan=plan,
            method_core=method_core,
        )
        if (
            mechanics_manifest["identity_sha256"]
            != stage_A_locked["E_prelabel_mechanics_manifest_identity_sha256"]
            or sha256_file(args.E_mechanics_product / "manifest.json")
            != stage_A_locked["E_prelabel_mechanics_manifest_file_sha256"]
        ):
            raise RuntimeError("schedule-exact diagnostic substituted E mechanics metadata")
        products = {
            E_CANDIDATE: E_rows,
            "E_no_gate": no_gate_rows,
            "E_first_hit_full_budget": one_shot_rows,
            "G_start": G_start_rows,
        }
        codes, score_matrices, _ = matrices(labels, products, tuple(plan["selected_classes"]))
        E_auc = auc_from_vectors(codes, score_matrices[E_CANDIDATE], plan["selected_classes"])[
            "auc"
        ]
        no_gate_auc = auc_from_vectors(
            codes, score_matrices["E_no_gate"], plan["selected_classes"]
        )["auc"]
        one_shot_auc = auc_from_vectors(
            codes,
            score_matrices["E_first_hit_full_budget"],
            plan["selected_classes"],
        )["auc"]
        G_start_auc = auc_from_vectors(
            codes, score_matrices["G_start"], plan["selected_classes"]
        )["auc"]
        schedule_exact = schedule_exact_conditional_concordance(
            codes,
            score_matrices[E_CANDIDATE],
            mechanics_arrays["start_time_index"],
            mechanics_arrays["start_remaining_effective_count"],
            tuple(plan["selected_classes"]),
            draws=INCREMENTAL_DRAWS,
            seed=SCHEDULE_EXACT_BOOTSTRAP_SEED,
        )
        gate_bootstrap = paired_seed_cluster_bootstrap(
            codes,
            score_matrices["E_no_gate"],
            score_matrices[E_CANDIDATE],
            tuple(plan["selected_classes"]),
            draws=INCREMENTAL_DRAWS,
            seed=ABLATION_BOOTSTRAP_SEED,
            contrast_name="E_blur_gated_minus_E_no_state_gate",
        )
        one_shot_bootstrap = paired_seed_cluster_bootstrap(
            codes,
            score_matrices["E_first_hit_full_budget"],
            score_matrices[E_CANDIDATE],
            tuple(plan["selected_classes"]),
            draws=INCREMENTAL_DRAWS,
            seed=ONE_SHOT_BOOTSTRAP_SEED,
            contrast_name="E_distributed_minus_E_first_hit_full_budget",
        )
        G_start_bootstrap = paired_seed_cluster_bootstrap(
            codes,
            score_matrices["G_start"],
            score_matrices[E_CANDIDATE],
            tuple(plan["selected_classes"]),
            draws=INCREMENTAL_DRAWS,
            seed=G_START_BOOTSTRAP_SEED,
            contrast_name="E_path_LR_minus_G_start_schedule",
        )
        all_three_descriptive_contrasts_pass = bool(
            gate_bootstrap["passes"]
            and one_shot_bootstrap["passes"]
            and G_start_bootstrap["passes"]
        )
        # The frozen authorization contract requires the B-gate ablation and
        # the pre-registered one-dimensional G_start comparison.  The
        # equal-budget one-shot comparison supports only the separate
        # distributed-path claim and cannot veto or rescue rollback.
        rollback_mechanism_gates = rollback_mechanism_authorized(
            E_minus_no_state_gate_pass=bool(gate_bootstrap["passes"]),
            E_minus_prespecified_scalar_G_pass=bool(G_start_bootstrap["passes"]),
            E_minus_one_shot_pass=bool(one_shot_bootstrap["passes"]),
        )
        result = {
            "schema_version": 1,
            "status": "POST_PRIMARY_ABLATION_COMPLETE",
            "E_primary_pass": True,
            "E_beyond_B_incremental_pass": True,
            "E_no_gate_product_opened": True,
            "E_first_hit_full_budget_product_opened": True,
            "G_start_product_opened": True,
            "E_product_receipt": E_receipt,
            "E_no_gate_product_receipt": no_gate_receipt,
            "E_first_hit_full_budget_product_receipt": one_shot_receipt,
            "G_start_product_receipt": G_start_receipt,
            "E_blur_gated_auc": E_auc,
            "E_no_state_gate_auc": no_gate_auc,
            "E_first_hit_full_budget_auc": one_shot_auc,
            "G_start_schedule_auc": G_start_auc,
            "DeltaAUC_gate": E_auc - no_gate_auc,
            "DeltaAUC_distributed_minus_one_shot": E_auc - one_shot_auc,
            "DeltaAUC_path_LR_minus_start_schedule": E_auc - G_start_auc,
            "paired_seed_cluster_bootstrap_E_minus_no_gate": gate_bootstrap,
            "paired_seed_cluster_bootstrap_E_minus_one_shot": one_shot_bootstrap,
            "paired_seed_cluster_bootstrap_E_minus_G_start": G_start_bootstrap,
            "schedule_exact_conditional_concordance_descriptive": schedule_exact,
            "B_gate_mechanism_claim_pass": gate_bootstrap["passes"],
            "distributed_path_claim_pass": one_shot_bootstrap["passes"],
            "E_outperforms_preregistered_one_dimensional_start_summary_pass": (
                G_start_bootstrap["passes"]
            ),
            "G_start_is_not_a_schedule_exact_or_causal_control": True,
            "G_start_claim_limit": (
                "E exceeds the pre-registered one-dimensional start summary; "
                "this does not isolate or causally prove innovation alignment"
            ),
            "all_three_descriptive_post_primary_contrasts_pass": (
                all_three_descriptive_contrasts_pass
            ),
            "one_shot_contrast_required_for_rollback": False,
            "rollback_required_mechanism_gates_pass": rollback_mechanism_gates,
            "rollback_authorized": rollback_mechanism_gates,
            "ablation_cannot_rescue_failed_E": True,
        }
    result.update(
        {
            "dynamic_contract_identity_sha256": contract["identity_sha256"],
            "scientific_protocol_identity_sha256": protocol["identity_sha256"],
            "trace_plan_identity_sha256": plan["identity_sha256"],
        }
    )
    result["identity_sha256"] = canonical_sha256(result)
    publish_artifact(
        args.output,
        artifact_kind="SCIENTIFIC_V4_E_POST_PRIMARY_MECHANISM_SUITE",
        payloads={"ablation.json": json.dumps(result, indent=2, sort_keys=True) + "\n"},
        manifest_fields={
            "dynamic_contract_identity_sha256": contract["identity_sha256"],
            "scientific_protocol_identity_sha256": protocol["identity_sha256"],
            "trace_plan_identity_sha256": plan["identity_sha256"],
            "result_identity_sha256": result["identity_sha256"],
        },
    )
    print(json.dumps({"status": result["status"], "rollback_authorized": result["rollback_authorized"]}, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subs = result.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--source-lock", type=Path, default=DEFAULT_DYNAMIC_SOURCE_LOCK)
    common.add_argument("--trace-plan", type=Path, required=True)
    pre = subs.add_parser("prelabel", parents=[common])
    pre.add_argument("--E-mechanics-product", type=Path, required=True)
    pre.add_argument("--no-touch-receipt", type=Path, required=True)
    pre.add_argument("--output", type=Path, required=True)
    a = subs.add_parser("stage-a", parents=[common])
    a.add_argument("--prelabel-receipt", type=Path, required=True)
    a.add_argument("--consensus-root", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    b = subs.add_parser("stage-b", parents=[common])
    b.add_argument("--stage-a-receipt", type=Path, required=True)
    b.add_argument("--consensus-root", type=Path, required=True)
    b.add_argument("--B-product", type=Path)
    b.add_argument("--E-product", type=Path)
    b.add_argument("--output", type=Path, required=True)
    abl = subs.add_parser("ablation", parents=[common])
    abl.add_argument("--stage-b-result", type=Path, required=True)
    abl.add_argument("--consensus-root", type=Path, required=True)
    abl.add_argument("--E-product", type=Path, required=True)
    abl.add_argument("--E-mechanics-product", type=Path, required=True)
    abl.add_argument("--E-no-gate-product", type=Path, required=True)
    abl.add_argument("--E-first-hit-full-budget-product", type=Path, required=True)
    abl.add_argument("--G-start-product", type=Path, required=True)
    abl.add_argument("--output", type=Path, required=True)
    pre.set_defaults(func=prelabel)
    a.set_defaults(func=stage_a)
    b.set_defaults(func=stage_b)
    abl.set_defaults(func=ablation)
    return result


def main(argv: Iterable[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "stage-b":
        # Paths are intentionally optional so a failed Stage A can complete
        # without ever receiving or inspecting a score location.
        if args.B_product is None and args.E_product is None:
            pass
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
