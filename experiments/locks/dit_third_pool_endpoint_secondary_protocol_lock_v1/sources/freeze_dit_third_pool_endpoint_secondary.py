#!/usr/bin/env python3
"""Freeze the third-pool terminal representation secondary before labels open.

This phase-0 freezer reads only the completed *old* endpoint-embedding product,
the old discovery consensus, the old aggregate endpoint audit, and frozen
source/protocol files.  It fits the pre-declared clean reference, snapshots all
code needed for future label-free extraction and aggregate evaluation, and
publishes an immutable hash lock.  It never opens or stats a third-pool image,
trace, embedding, label, review, score, or screening result.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import evaluate_dit_third_pool_endpoint_secondary as evaluator


ROOT = evaluator.ROOT
DEFAULT_OUTPUT = (
    ROOT / "experiments/locks/dit_third_pool_endpoint_secondary_protocol_lock_v1"
)
OLD_PROTOCOL = (
    ROOT / "experiments/locks/dit_endpoint_representation_distance_protocol_v1/protocol.json"
)
OLD_EMBEDDINGS = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "bad_good_metric_confirmation_expansion_v1/endpoint_embeddings_label_free_v1"
)
OLD_DISCOVERY_LABELS = (
    ROOT / "experiments/annotations/dit_fresh_eval240_adjudicated_consensus_lock_v2"
)
OLD_AUDIT = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "bad_good_metric_confirmation_expansion_v1/"
    "endpoint_representation_distance_audit_v1"
)

DISCOVERY_SEEDS = tuple(range(50, 130))
DISCOVERY_STATUS = "FINAL_VISUAL_LABELS_LOCKED_BEFORE_ANY_LABEL_SCORE_JOIN"
EXPECTED_CLEAN_COUNTS = {207: 71, 602: 70, 795: 75}

SOURCE_PATHS = {
    "evaluate_dit_third_pool_endpoint_secondary.py": (
        ROOT / "experiments/evaluate_dit_third_pool_endpoint_secondary.py"
    ),
    "freeze_dit_third_pool_endpoint_secondary.py": Path(__file__).resolve(),
    "extract_dit_endpoint_embeddings_label_free.py": (
        ROOT / "experiments/extract_dit_endpoint_embeddings_label_free.py"
    ),
    "audit_dit_endpoint_representation_distances.py": (
        ROOT / "experiments/audit_dit_endpoint_representation_distances.py"
    ),
    "evaluate_dit_bad_good_third_pool_confirmation.py": (
        ROOT / "experiments/evaluate_dit_bad_good_third_pool_confirmation.py"
    ),
}


def source_records() -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for basename, raw_path in SOURCE_PATHS.items():
        path = evaluator.require_regular(raw_path, f"source {basename}")
        records[basename] = {
            "live_path_at_freeze": str(path),
            "bytes": path.stat().st_size,
            "sha256": evaluator.sha256_file(path),
        }
    if set(records) != set(evaluator.SOURCE_BASENAMES):
        raise RuntimeError("endpoint secondary source set changed")
    if (
        records["extract_dit_endpoint_embeddings_label_free.py"]["sha256"]
        != evaluator.EXPECTED_ENDPOINT_EXTRACTOR_SHA256
        or records["audit_dit_endpoint_representation_distances.py"]["sha256"]
        != evaluator.EXPECTED_OLD_DISTANCE_HELPER_SHA256
        or records["evaluate_dit_bad_good_third_pool_confirmation.py"]["sha256"]
        != evaluator.EXPECTED_PRIMARY_EVALUATOR_SHA256
    ):
        raise RuntimeError("an imported helper differs from its frozen SHA pin")
    return records


def load_old_helper() -> Any:
    path = SOURCE_PATHS["audit_dit_endpoint_representation_distances.py"]
    if evaluator.sha256_file(path) != evaluator.EXPECTED_OLD_DISTANCE_HELPER_SHA256:
        raise RuntimeError("old endpoint-distance helper changed")
    return evaluator.load_module(path, "_third_pool_endpoint_reference_helper")


def validate_old_audit(helper: Any) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    root = evaluator.require_directory(OLD_AUDIT, "old aggregate endpoint audit")
    manifest_path = evaluator.require_regular(root / "manifest.json", "old audit manifest")
    completion_path = evaluator.require_regular(root / "completion.json", "old audit completion")
    metric_path = evaluator.require_regular(root / "metric_results.csv", "old metric results")
    fits_path = evaluator.require_regular(
        root / "reference_fit_summaries.json", "old reference-fit summaries"
    )
    manifest = evaluator.load_json(manifest_path)
    completion = evaluator.load_json(completion_path)
    if (
        manifest.get("identity_sha256")
        != evaluator.EXPECTED_OLD_AUDIT_MANIFEST_IDENTITY
        or evaluator.canonical_sha256(evaluator.without_identity(manifest))
        != evaluator.EXPECTED_OLD_AUDIT_MANIFEST_IDENTITY
        or evaluator.sha256_file(manifest_path)
        != evaluator.EXPECTED_OLD_AUDIT_MANIFEST_FILE_SHA256
        or evaluator.sha256_file(metric_path)
        != evaluator.EXPECTED_OLD_AUDIT_METRIC_RESULTS_FILE_SHA256
        or manifest.get("protocol_identity_sha256")
        != evaluator.EXPECTED_OLD_ENDPOINT_PROTOCOL_IDENTITY
        or manifest.get("status") != "complete"
        or completion.get("complete") is not True
        or completion.get("manifest_identity_sha256")
        != evaluator.EXPECTED_OLD_AUDIT_MANIFEST_IDENTITY
        or completion.get("manifest_file_sha256")
        != evaluator.EXPECTED_OLD_AUDIT_MANIFEST_FILE_SHA256
        or completion.get("row_level_payload_emitted") is not False
        or completion.get("image_payload_emitted") is not False
    ):
        raise RuntimeError("old aggregate endpoint audit identity changed")
    helper.validate_manifest_members(root, manifest)
    actual = {path.name for path in root.iterdir()}
    expected = {item["name"] for item in manifest["files"]} | {
        "manifest.json",
        "completion.json",
    }
    if actual != expected or any(not path.is_file() or path.is_symlink() for path in root.iterdir()):
        raise RuntimeError("old endpoint audit has an unexpected member")

    rows: list[dict[str, Any]] = []
    with metric_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "metric",
            "direction",
            "expansion_primary_bad_vs_good_pair_weighted_auc",
            "expansion_block_permutation_p_one_sided",
            "expansion_holm_p_across_six",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise RuntimeError("old aggregate metric table columns changed")
        for raw in reader:
            if raw["metric"] not in evaluator.METRICS or raw["direction"] != "distance_high_is_bad":
                raise RuntimeError("old endpoint metric family/direction changed")
            row = {
                "metric": raw["metric"],
                "expansion_bad_vs_good_pair_weighted_auc": float(
                    raw["expansion_primary_bad_vs_good_pair_weighted_auc"]
                ),
                "old_one_sided_block_permutation_p": float(
                    raw["expansion_block_permutation_p_one_sided"]
                ),
                "old_Holm_p_across_six": float(raw["expansion_holm_p_across_six"]),
            }
            if not all(np.isfinite(value) for key, value in row.items() if key != "metric"):
                raise RuntimeError("old aggregate endpoint result is non-finite")
            rows.append(row)
    if len(rows) != len(evaluator.METRICS) or {row["metric"] for row in rows} != set(evaluator.METRICS):
        raise RuntimeError("old aggregate metric row set changed")
    rows.sort(key=lambda row: evaluator.METRICS.index(row["metric"]))

    fit_payload = evaluator.load_json(fits_path)
    fits = fit_payload.get("fits")
    if not isinstance(fits, list):
        raise RuntimeError("old reference-fit summaries changed")
    final_fits = [
        dict(item)
        for item in fits
        if isinstance(item, dict)
        and item.get("reference") == "all_discovery_clean_for_expansion"
        and item.get("fold") is None
    ]
    if {item.get("representation") for item in final_fits} != set(evaluator.REPRESENTATIONS):
        raise RuntimeError("old final reference-fit summaries changed")
    return (
        {
            "manifest_identity_sha256": manifest["identity_sha256"],
            "manifest_file_sha256": evaluator.sha256_file(manifest_path),
            "completion_file_sha256": evaluator.sha256_file(completion_path),
            "metric_results_file_sha256": evaluator.sha256_file(metric_path),
            "reference_fit_summaries_file_sha256": evaluator.sha256_file(fits_path),
        },
        rows,
        final_fits,
    )


def reference_array_records(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "shape": list(np.asarray(value).shape),
            "dtype": np.asarray(value).dtype.str,
            "raw_sha256": evaluator.sha256_array(np.asarray(value)),
        }
        for name, value in sorted(arrays.items())
    }


def build_reference() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    helper = load_old_helper()
    old_protocol, old_protocol_identity = helper.validate_protocol(OLD_PROTOCOL)
    if (
        old_protocol_identity != evaluator.EXPECTED_OLD_ENDPOINT_PROTOCOL_IDENTITY
        or evaluator.sha256_file(OLD_PROTOCOL)
        != evaluator.EXPECTED_OLD_ENDPOINT_PROTOCOL_FILE_SHA256
    ):
        raise RuntimeError("old endpoint protocol identity changed")
    index, embeddings, embedding_lineage = helper.validate_embedding_product(
        OLD_EMBEDDINGS, old_protocol
    )
    if (
        embedding_lineage.get("manifest_identity_sha256")
        != evaluator.EXPECTED_OLD_EMBEDDING_MANIFEST_IDENTITY
        or embedding_lineage.get("manifest_file_sha256")
        != evaluator.EXPECTED_OLD_EMBEDDING_MANIFEST_FILE_SHA256
        or evaluator.sha256_file(OLD_EMBEDDINGS / "embeddings.npz")
        != evaluator.EXPECTED_OLD_EMBEDDINGS_FILE_SHA256
        or embedding_lineage.get("embedding_array_sha256")
        != evaluator.EXPECTED_OLD_EMBEDDING_ARRAY_SHA256
    ):
        raise RuntimeError("old endpoint embedding identity changed")
    labels, label_lineage = helper.validate_label_lock(
        OLD_DISCOVERY_LABELS, DISCOVERY_SEEDS, DISCOVERY_STATUS
    )
    if (
        label_lineage.get("consensus_identity_sha256")
        != evaluator.EXPECTED_OLD_LABEL_CONSENSUS_IDENTITY
        or label_lineage.get("manifest_identity_sha256")
        != evaluator.EXPECTED_OLD_LABEL_MANIFEST_IDENTITY
        or evaluator.sha256_file(OLD_DISCOVERY_LABELS / "manifest.json")
        != evaluator.EXPECTED_OLD_LABEL_MANIFEST_FILE_SHA256
        or evaluator.sha256_file(OLD_DISCOVERY_LABELS / "consensus_locked.json")
        != evaluator.EXPECTED_OLD_LABEL_CONSENSUS_FILE_SHA256
    ):
        raise RuntimeError("old discovery label identity changed")
    audit_lineage, old_results, old_final_fits = validate_old_audit(helper)

    frame = index[["sample_index", "global_seed", "class_id"]].merge(
        labels, on=["global_seed", "class_id"], how="left", validate="one_to_one"
    )
    discovery = frame.global_seed.isin(DISCOVERY_SEEDS)
    if frame.loc[discovery, "label"].isna().any() or int(discovery.sum()) != 240:
        raise RuntimeError("old discovery embedding/label join is incomplete")
    discovery_positions = frame.loc[discovery, "sample_index"].to_numpy(dtype=int)
    discovery_classes = frame.loc[discovery, "class_id"].to_numpy(dtype=int)
    discovery_labels = frame.loc[discovery, "label"].to_numpy(dtype=str)
    clean_counts = {
        class_id: int(
            np.sum(
                (discovery_classes == class_id)
                & (discovery_labels == helper.GOOD)
            )
        )
        for class_id in evaluator.CLASSES
    }
    if clean_counts != EXPECTED_CLEAN_COUNTS:
        raise RuntimeError(f"old discovery clean counts changed: {clean_counts}")

    arrays: dict[str, np.ndarray] = {}
    fits_by_representation = {
        item["representation"]: item for item in old_final_fits
    }
    fit_replay: list[dict[str, Any]] = []
    score_hashes: dict[str, str] = {}
    all_classes = frame.class_id.to_numpy(dtype=int)
    for representation, dimension in evaluator.REPRESENTATIONS.items():
        values = helper.normalize_rows(
            np.asarray(embeddings[representation], dtype=np.float64)
        )
        model, fit = helper.fit_reference(
            values[discovery_positions], discovery_classes, discovery_labels
        )
        old_fit = fits_by_representation[representation]
        expected_fit = {
            "clean_counts": {str(key): value for key, value in fit["clean_counts"].items()},
            "pooled_residual_count": fit["pooled_residual_count"],
            "dimension": fit["dimension"],
            "ledoit_wolf_shrinkage": fit["ledoit_wolf_shrinkage"],
            "isotropic_variance_component": fit["isotropic_variance_component"],
            "effective_rank_bound": fit["effective_rank_bound"],
        }
        for key, value in expected_fit.items():
            observed = old_fit.get(key)
            if isinstance(value, float):
                if not np.isclose(value, observed, rtol=0.0, atol=1e-15):
                    raise RuntimeError(f"old {representation} reference fit no longer replays: {key}")
            elif value != observed:
                raise RuntimeError(f"old {representation} reference fit no longer replays: {key}")
        prefix = representation
        for class_id in evaluator.CLASSES:
            arrays[evaluator._reference_key(prefix, "center", class_id)] = np.asarray(
                model.centers[class_id], dtype=np.float64
            )
            arrays[evaluator._reference_key(prefix, "unit_center", class_id)] = np.asarray(
                model.unit_centers[class_id], dtype=np.float64
            )
            arrays[evaluator._reference_key(prefix, "clean_points", class_id)] = np.asarray(
                model.clean_points[class_id], dtype=np.float64
            )
        arrays[evaluator._reference_key(prefix, "covariance_basis")] = np.asarray(
            model.covariance_basis, dtype=np.float64
        )
        arrays[evaluator._reference_key(prefix, "covariance_eigenvalues")] = np.asarray(
            model.covariance_eigenvalues, dtype=np.float64
        )
        arrays[evaluator._reference_key(prefix, "shrinkage")] = np.asarray(
            [model.shrinkage], dtype=np.float64
        )
        arrays[evaluator._reference_key(prefix, "isotropic")] = np.asarray(
            [model.isotropic], dtype=np.float64
        )
        scores = helper.score_reference(model, values, all_classes)
        for distance, score in scores.items():
            score_hashes[f"{representation}__{distance}"] = evaluator.sha256_array(
                np.asarray(score, dtype=np.float64)
            )
        fit_replay.append(
            {
                "representation": representation,
                **expected_fit,
                "matches_old_aggregate_audit_fit_at_absolute_tolerance_1e-15": True,
            }
        )

    records = reference_array_records(arrays)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE_OLD_DISCOVERY_CLEAN_REFERENCE_FROZEN_BEFORE_THIRD_LABEL_ACCESS",
        "reference_role": (
            "terminal supervised typicality reference; not a label definition, "
            "candidate selector, online trigger, or intervention authority"
        ),
        "cohort": {
            "old_discovery_seeds_inclusive": [DISCOVERY_SEEDS[0], DISCOVERY_SEEDS[-1]],
            "ordered_classes": list(evaluator.CLASSES),
            "old_discovery_trajectory_count": 240,
            "clean_good_count_by_class": {
                str(key): value for key, value in clean_counts.items()
            },
            "clean_good_total": sum(clean_counts.values()),
        },
        "construction": {
            "row_normalization": "L2",
            "centroids": "one arithmetic clean-good centroid per class and representation",
            "knn": "five nearest same-class clean-good points by cosine distance",
            "covariance": (
                "one pooled within-class Ledoit-Wolf covariance per representation, "
                "represented exactly by shrinkage, isotropic component, residual "
                "right-singular basis, and residual covariance eigenvalues"
            ),
            "future_reference_refitting_allowed": False,
            "third_pool_endpoint_used_in_reference_fit": False,
        },
        "fit_replay": fit_replay,
        "old_aggregate_audit_results_context_only": old_results,
        "all_old_600_endpoint_score_vector_hashes_for_replay": score_hashes,
        "array_records": records,
        "array_bundle_logical_identity_sha256": evaluator.canonical_sha256(records),
        "input_lineage": {
            "old_endpoint_protocol": {
                "identity_sha256": old_protocol_identity,
                "file_sha256": evaluator.sha256_file(OLD_PROTOCOL),
            },
            "old_endpoint_embeddings": {
                **embedding_lineage,
                "embeddings_file_sha256": evaluator.sha256_file(
                    OLD_EMBEDDINGS / "embeddings.npz"
                ),
            },
            "old_discovery_consensus": {
                **label_lineage,
                "manifest_file_sha256": evaluator.sha256_file(
                    OLD_DISCOVERY_LABELS / "manifest.json"
                ),
                "consensus_file_sha256": evaluator.sha256_file(
                    OLD_DISCOVERY_LABELS / "consensus_locked.json"
                ),
                "completion_file_sha256": evaluator.sha256_file(
                    OLD_DISCOVERY_LABELS / "completion.json"
                ),
            },
            "old_aggregate_endpoint_audit": audit_lineage,
        },
        "evidence_access_audit": {
            "old_label_free_endpoint_embeddings_opened": True,
            "old_discovery_final_consensus_rows_opened_for_clean_reference": True,
            "old_aggregate_endpoint_audit_opened": True,
            "old_expansion_consensus_rows_opened": False,
            "third_pool_path_opened_statted_or_hashed": False,
            "third_pool_image_trace_embedding_label_review_score_or_screen_opened": False,
            "third_pool_score_label_join_performed": False,
        },
    }
    summary["identity_sha256"] = evaluator.canonical_sha256(summary)
    return arrays, summary


def build_protocol(
    summary: Mapping[str, Any], sources: Mapping[str, Any]
) -> dict[str, Any]:
    inception_weights = (
        "/home/zhoushunyu/.cache/torch/hub/checkpoints/"
        "pt_inception-2015-12-05-6726825d.pth"
    )
    dino_snapshot = (
        "/home/zhoushunyu/.cache/huggingface/hub/"
        "models--facebook--dinov2-with-registers-large/snapshots/"
        "e4c89a4e05589de9b3e188688a303d0f3c04d0f3"
    )
    protocol: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "dit_third_pool_endpoint_secondary_v1",
        "status": "FROZEN_BEFORE_ANY_THIRD_POOL_LABEL_IMAGE_SCORE_OR_EMBEDDING_ACCESS",
        "implementation_source_sha256": sources[
            "evaluate_dit_third_pool_endpoint_secondary.py"
        ]["sha256"],
        "foundation_identity_pins": {
            "sampling_protocol_identity_sha256": evaluator.EXPECTED_SAMPLING_PROTOCOL_IDENTITY,
            "sampling_manifest_identity_sha256": evaluator.EXPECTED_SAMPLING_MANIFEST_IDENTITY,
            "primary_evaluation_contract_identity_sha256": evaluator.EXPECTED_PRIMARY_CONTRACT_IDENTITY,
            "primary_evaluation_source_manifest_identity_sha256": evaluator.EXPECTED_PRIMARY_SOURCE_MANIFEST_IDENTITY,
            "primary_evaluator_source_sha256": evaluator.EXPECTED_PRIMARY_EVALUATOR_SHA256,
        },
        "reference": {
            "summary_identity_sha256": summary["identity_sha256"],
            "array_bundle_logical_identity_sha256": summary[
                "array_bundle_logical_identity_sha256"
            ],
            "old_discovery_seeds_inclusive": [50, 129],
            "old_discovery_clean_good_only": True,
            "clean_good_count_by_class": {"207": 71, "602": 70, "795": 75},
            "third_pool_refit_or_adaptation_allowed": False,
            "normalization": "rowwise L2",
            "knn_k": 5,
            "pooled_covariance": "within-class residual Ledoit-Wolf, one per representation",
        },
        "secondary_hypotheses": {
            "E1_inception_blur_centroid": {
                "metric": evaluator.E1_METRIC,
                "direction": "distance_high_is_bad",
                "endpoint": "blur_or_soft_fusion_clear_bad_vs_clean_good",
                "pair_weighted_auc_at_least": 0.75,
                "Holm_adjusted_p_strictly_below": 0.05,
                "role": "secondary_corroboration_only",
            },
            "E2_dino_allbad_mahalanobis": {
                "metric": evaluator.E2_METRIC,
                "direction": "distance_high_is_bad",
                "endpoint": "all_clear_bad_vs_clean_good",
                "pair_weighted_auc_at_least": 0.70,
                "Holm_adjusted_p_strictly_below": 0.05,
                "role": "secondary_corroboration_only",
            },
        },
        "secondary_family": {
            "family_size": 2,
            "ordered_ids": list(evaluator.SECONDARY_IDS),
            "method": "Holm step-down across exactly E1 and E2",
            "strict_alpha": 0.05,
            "candidate_combination_allowed": False,
            "each_secondary_reported_independently": True,
        },
        "stage_a_gate_contract": {
            "authority": (
                "validate_stage_a_receipt from the exact frozen primary evaluator v5"
            ),
            "minimum_blur_or_soft_fusion_clear_bad": evaluator.EVENT_MIN_BLUR,
            "minimum_total_clear_bad": evaluator.EVENT_MIN_TOTAL_BAD,
            "both_minima_required": True,
            "failure_action": (
                "fail before endpoint product, consensus rows, sampling pool, or reference access"
            ),
            "only_after_pass": (
                "open endpoint embeddings, full consensus rows, sampling lineage, and frozen reference"
            ),
        },
        "statistics": {
            "primary_auc": (
                "sum over classes of distance-high concordant bad-good pairs, ties=0.5, "
                "divided by sum over classes n_positive_class*n_clean_good_class; "
                "zero total pair denominator fails closed"
            ),
            "permutation_draws": evaluator.PERMUTATION_DRAWS,
            "permutation_seed": evaluator.PERMUTATION_SEED,
            "permutation_unit": (
                "one intact ordered three-class severity/phenotype block per global seed"
            ),
            "permutation_alternative": "distance-high one-sided",
            "other_four_metrics_inferential_p_values_allowed": False,
            "subtype_and_per_class_inferential_p_values_allowed": False,
            "descriptive_zero_pair_endpoint": "report non-evaluable with null AUC",
            "minimum_group_size_for_per_class_publication": evaluator.MINIMUM_GROUP_SIZE,
        },
        "all_fixed_metrics": list(evaluator.METRICS),
        "descriptive_guardrails": {
            "all_six_metrics_by_four_frozen_endpoints": True,
            "inferential_family_members": list(evaluator.SECONDARY_IDS),
            "all_other_metrics_and_cuts": "descriptive only",
            "no_post_hoc_combination_threshold_or_direction_change": True,
        },
        "label_free_extraction": {
            "existing_extractor_compatible": True,
            "inventory_adapter_required": False,
            "inventory_source": (
                "the future primary label-free product source_inventory.json already bound "
                "by the primary evaluator input lock"
            ),
            "required_seed_axis_half_open": "250:850",
            "required_classes_ordered": list(evaluator.CLASSES),
            "expected_trajectory_count": evaluator.TRAJECTORY_COUNT,
            "representations": dict(evaluator.REPRESENTATIONS),
            "inception_weights": {
                "path": inception_weights,
                "sha256": "6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2",
            },
            "dino_snapshot": {
                "path": dino_snapshot,
                "revision": "e4c89a4e05589de9b3e188688a303d0f3c04d0f3",
                "file_sha256": {
                    "config.json": "03eee42f646659a9480f8911a81fdd81efeedd7ff39083c8e36398068daf72f5",
                    "model.safetensors": "edccedab2c4e164e80833096de89a32a6e8d7365870499a066a61dbc8894b42b",
                    "preprocessor_config.json": "14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828",
                },
            },
            "future_command_template": [
                "python",
                "experiments/extract_dit_endpoint_embeddings_label_free.py",
                "--source-inventory",
                "<PRIMARY_LABEL_FREE_PRODUCT>/source_inventory.json",
                "--expected-seeds",
                "250:850",
                "--expected-classes",
                "207,602,795",
                "--inception-weights",
                inception_weights,
                "--dino-snapshot",
                dino_snapshot,
                "--device",
                "cuda:<GPU_ID>",
                "--batch-size",
                "16",
                "--output-dir",
                "<THIRD_POOL_ENDPOINT_EMBEDDING_OUTPUT>",
            ],
            "labels_reviews_scores_or_distances_read_by_extractor": False,
        },
        "publication": {
            "aggregate_only": True,
            "sample_rows_scores_distances_ranks_or_permutation_draws_emitted": False,
            "secondary_cannot_rescue_B_or_C": True,
            "secondary_cannot_authorize_intervention": True,
            "no_overwrite": True,
        },
        "threat_model": {
            "assumption": (
                "hash locks live on a controlled, static, non-concurrently-rewritten local "
                "filesystem and chronology is established by Git or an append-only record"
            ),
            "not_claimed": (
                "cryptographic authentication against malicious directory replacement or "
                "manual re-signing of a self-consistent JSON tree"
            ),
        },
        "source_snapshots": dict(sources),
        "evidence_access_audit": {
            "old_reference_inputs_opened": True,
            "third_pool_path_opened_statted_or_hashed": False,
            "third_pool_image_trace_embedding_label_review_score_or_screen_opened": False,
            "third_pool_score_label_join_performed": False,
            "scientific_override_interface_exists": False,
        },
    }
    protocol["identity_sha256"] = evaluator.canonical_sha256(protocol)
    return protocol


def publish(output: Path) -> Path:
    output = output.expanduser().absolute()
    if os.path.lexists(output):
        raise RuntimeError(f"refusing to overwrite endpoint protocol lock: {output}")
    sources = source_records()
    arrays, summary = build_reference()
    protocol = build_protocol(summary, sources)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        evaluator.write_json(staging / "protocol.json", protocol)
        evaluator.write_json(staging / "reference_summary.json", summary)
        np.savez_compressed(staging / "reference_models.npz", **arrays)
        source_root = staging / "sources"
        source_root.mkdir()
        for basename, path in SOURCE_PATHS.items():
            shutil.copy2(path, source_root / basename)
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "experiment": "dit_third_pool_endpoint_secondary_protocol_lock_v1",
            "status": "complete",
            "protocol_identity_sha256": protocol["identity_sha256"],
            "reference_summary_identity_sha256": summary["identity_sha256"],
            "files": evaluator.artifact_records(staging),
        }
        manifest["identity_sha256"] = evaluator.canonical_sha256(manifest)
        evaluator.write_json(staging / "manifest.json", manifest)
        evaluator.write_json(
            staging / "completion.json",
            {
                "complete": True,
                "protocol_identity_sha256": protocol["identity_sha256"],
                "protocol_file_sha256": evaluator.sha256_file(staging / "protocol.json"),
                "reference_summary_identity_sha256": summary["identity_sha256"],
                "reference_summary_file_sha256": evaluator.sha256_file(
                    staging / "reference_summary.json"
                ),
                "reference_models_file_sha256": evaluator.sha256_file(
                    staging / "reference_models.npz"
                ),
                "manifest_identity_sha256": manifest["identity_sha256"],
                "manifest_file_sha256": evaluator.sha256_file(staging / "manifest.json"),
                "third_pool_data_opened": False,
            },
        )
        evaluator.validate_source_lock(staging)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def dry_run() -> dict[str, Any]:
    sources = source_records()
    arrays, summary = build_reference()
    protocol = build_protocol(summary, sources)
    return {
        "status": "DRY_RUN_NO_OUTPUT_WRITTEN",
        "protocol_identity_sha256": protocol["identity_sha256"],
        "reference_summary_identity_sha256": summary["identity_sha256"],
        "reference_array_count": len(arrays),
        "reference_array_logical_identity_sha256": summary[
            "array_bundle_logical_identity_sha256"
        ],
        "third_pool_data_opened": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--freeze", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        evaluator.synthetic_self_test()
        print("phase-0 freezer synthetic self-test passed; no third-pool data opened")
        return 0
    if args.dry_run:
        print(json.dumps(dry_run(), indent=2, sort_keys=True))
        return 0
    path = publish(args.output)
    protocol, manifest = evaluator.validate_source_lock(path)
    print(
        json.dumps(
            {
                "path": str(path),
                "protocol_identity_sha256": protocol["identity_sha256"],
                "manifest_identity_sha256": manifest["identity_sha256"],
                "third_pool_data_opened": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
