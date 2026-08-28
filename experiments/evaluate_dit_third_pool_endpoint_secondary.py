#!/usr/bin/env python3
"""Gate-first aggregate evaluation of frozen third-pool endpoint distances.

This source is frozen before any third-pool label or endpoint embedding is
opened.  A small future input lock may bind paths and caller-supplied manifest
identities, but it does not inspect those paths.  Evaluation first validates
the already-frozen primary evaluator's Stage-A receipt.  Unless that receipt
replays both event minima as passed, this program fails before opening or
hashing the endpoint product, consensus rows, sampling pool, or reference NPZ.

After authorization, exactly two secondary hypotheses are tested: E1 is the
Inception clean-centroid distance on blur/soft-fusion clear-bad versus clean;
E2 is the DINO shared Ledoit-Wolf Mahalanobis distance on all clear-bad versus
clean.  Four other fixed distances and subtype cuts are aggregate descriptive
guardrails only.  No sample row, distance, rank, image, or permutation draw is
written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_LOCK = (
    ROOT / "experiments/locks/dit_third_pool_endpoint_secondary_protocol_lock_v1"
)
PRIMARY_EVALUATION_SOURCE_LOCK = (
    ROOT / "experiments/locks/dit_bad_good_third_pool_evaluation_source_lock_v5"
)

EXPECTED_PRIMARY_CONTRACT_IDENTITY = (
    "6638f75eef792fa313fa14ebb0b6c65a696dab881c193f2bf8fa83615e1475e2"
)
EXPECTED_PRIMARY_SOURCE_MANIFEST_IDENTITY = (
    "d7467275fab416a5eddadf528fd24b98ffb6bfeed499711c3e8ba6b6f72cd6e8"
)
EXPECTED_PRIMARY_EVALUATOR_SHA256 = (
    "006a9337295d1a3f27ad8626fdff21d227038cf11f291a769dde1af8c41aff5c"
)
EXPECTED_SAMPLING_PROTOCOL_IDENTITY = (
    "330661e87de7846e1f590660f03ecef6270fa45e2f39c4fc54d992e3260950d8"
)
EXPECTED_SAMPLING_MANIFEST_IDENTITY = (
    "eae86d48c1c1b9c732fbeea4838b2418b9b7261b61db0355fd7306469f5b6df3"
)
EXPECTED_ENDPOINT_EXTRACTOR_SHA256 = (
    "5f3d52cb24e89e2c92639e70200f72f1e739344906550385c46ea2dfff343f8b"
)
EXPECTED_OLD_DISTANCE_HELPER_SHA256 = (
    "fa1db6520cbc3f5fb828c7bcc8404dd907ffe2041693863e48f0f79bf0b5e317"
)
EXPECTED_OLD_ENDPOINT_PROTOCOL_IDENTITY = (
    "1489f4d5f9844977fe3ec5950e9cca1e08b2dc4b734d6d1ef6786a585df5299f"
)
EXPECTED_OLD_ENDPOINT_PROTOCOL_FILE_SHA256 = (
    "16cb1f188cd939bd6c56f14a5e6797011a9350f6ad0cb9e6f76166c632c405ac"
)
EXPECTED_OLD_EMBEDDING_MANIFEST_IDENTITY = (
    "3116d850421cae89245ff50e84d915e87c74fb1218f12de5c1dda10f77f04912"
)
EXPECTED_OLD_EMBEDDING_MANIFEST_FILE_SHA256 = (
    "67f7cbb13db7918360affa95f799ce98f6b120c560e4785c8009f355e2f9d056"
)
EXPECTED_OLD_EMBEDDINGS_FILE_SHA256 = (
    "4a1bed660e19dbd5429ce8e4e7ca012f069172ae22e0e3a6d501f011fec42952"
)
EXPECTED_OLD_EMBEDDING_ARRAY_SHA256 = {
    "inception_fid_pool2048": (
        "749470b8da96eeb27548a59459b61f8b2f93696e9c1c39a07a2cc40a0288766a"
    ),
    "dinov2_registers_large_cls1024": (
        "f9a49daf1c2eacb5b28d6d88a05b6fff07fe9a745cbdaa5f3e65cc9cbd14c446"
    ),
}
EXPECTED_OLD_LABEL_CONSENSUS_IDENTITY = (
    "21c242dc796d5c8baa4568c9f82add0d1b64c984477cf8698efbbca5889e166a"
)
EXPECTED_OLD_LABEL_MANIFEST_IDENTITY = (
    "857c1454f3ec6f0cc24b743932e7da494a6a391d71e607574ca1904856ad818e"
)
EXPECTED_OLD_LABEL_MANIFEST_FILE_SHA256 = (
    "8983a254c7d190587f71f9b339dedd45dcde8d52cccbb601e82b586e23edb748"
)
EXPECTED_OLD_LABEL_CONSENSUS_FILE_SHA256 = (
    "5de85939ed73f1217c8b9225a41899cb5939ba5b27b7c689a212f49f1f4762b5"
)
EXPECTED_OLD_AUDIT_MANIFEST_IDENTITY = (
    "4510c4489ba082a8961793851537484efef299acda9d395e2f3fd36e48ae9335"
)
EXPECTED_OLD_AUDIT_MANIFEST_FILE_SHA256 = (
    "437819f7c8239c16be1243735b5fadc3607949425ca70a874701a484ca4de013"
)
EXPECTED_OLD_AUDIT_METRIC_RESULTS_FILE_SHA256 = (
    "bb0d20e9dace871bbc9b654ea94924d099fa511307fd8b13b7f2d5594ea2a7d1"
)
EXPECTED_OLD_EMBEDDING_COMPLETION_FILE_SHA256 = (
    "e33305de8b340c363101f02382241a35779b967dfef72c76e10b75cd69fae746"
)
EXPECTED_OLD_LABEL_COMPLETION_FILE_SHA256 = (
    "60758a2fc1998eab4720c35696e1f5621dd87f331778e5345e60c8a3c9278c3b"
)
EXPECTED_OLD_AUDIT_COMPLETION_FILE_SHA256 = (
    "7bf46b8a5819fff434d232c18a2b74a4d2735c030637b18f4d36c937ec18da01"
)
EXPECTED_REFERENCE_SUMMARY_IDENTITY = (
    "e0538863a1edc02e6f21d6ad7c6ddd32f8cc57046adcd4b88016d7ae5e1889fd"
)
EXPECTED_REFERENCE_ARRAY_LOGICAL_IDENTITY = (
    "51c3e80e712912a39c65cca977756b4bf125feba83579c4820ba0f0c143e5070"
)

CLASSES = (207, 602, 795)
SEEDS = tuple(range(250, 850))
TRAJECTORY_COUNT = len(CLASSES) * len(SEEDS)
REPRESENTATIONS = {
    "inception_fid_pool2048": 2048,
    "dinov2_registers_large_cls1024": 1024,
}
DISTANCES = (
    "cosine_to_class_clean_centroid",
    "shared_ledoitwolf_mahalanobis",
    "knn5_mean_cosine_to_class_clean",
)
METRICS = tuple(
    f"{representation}__{distance}"
    for representation in REPRESENTATIONS
    for distance in DISTANCES
)
E1_METRIC = "inception_fid_pool2048__cosine_to_class_clean_centroid"
E2_METRIC = "dinov2_registers_large_cls1024__shared_ledoitwolf_mahalanobis"
SECONDARY_IDS = ("E1_inception_blur_centroid", "E2_dino_allbad_mahalanobis")
EVENT_MIN_BLUR = 15
EVENT_MIN_TOTAL_BAD = 30
PERMUTATION_DRAWS = 100_000
PERMUTATION_SEED = 2026082802
PERMUTATION_BATCH = 256
MINIMUM_GROUP_SIZE = 5
SOURCE_BASENAMES = (
    "evaluate_dit_third_pool_endpoint_secondary.py",
    "freeze_dit_third_pool_endpoint_secondary.py",
    "extract_dit_endpoint_embeddings_label_free.py",
    "audit_dit_endpoint_representation_distances.py",
    "evaluate_dit_bad_good_third_pool_confirmation.py",
)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def without_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("identity_sha256", None)
    return result


def require_hex64(value: Any, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"{description} must be a lowercase SHA-256")
    return value


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def require_regular(path: Path, description: str) -> Path:
    path = path.expanduser().absolute()
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{description} must be a regular non-symlink file: {path}")
    return path.resolve()


def require_directory(path: Path, description: str) -> Path:
    path = path.expanduser().absolute()
    if not path.is_dir() or path.is_symlink():
        raise RuntimeError(f"{description} must be a real non-symlink directory: {path}")
    return path.resolve()


def load_module(path: Path, name: str) -> ModuleType:
    path = require_regular(path, f"module source {name}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def exact_tree(
    root: Path, *, expected_files: set[str], expected_directories: set[str]
) -> None:
    files: set[str] = set()
    directories: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise RuntimeError(f"lock contains a symlink: {path}")
        if path.is_file():
            files.add(relative)
        elif path.is_dir():
            directories.add(relative)
        else:
            raise RuntimeError(f"lock contains a special entry: {path}")
    if files != expected_files or directories != expected_directories:
        raise RuntimeError(
            f"lock tree changed: missing_files={sorted(expected_files-files)}, "
            f"extra_files={sorted(files-expected_files)}, "
            f"missing_dirs={sorted(expected_directories-directories)}, "
            f"extra_dirs={sorted(directories-expected_directories)}"
        )


def artifact_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {"manifest.json", "completion.json"}:
            continue
        if path.is_symlink():
            raise RuntimeError(f"artifact is a symlink: {path}")
        records.append(
            {
                "name": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def validate_record_lock(
    root: Path, *, artifact_kind: str, record_name: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = require_directory(root, artifact_kind)
    exact_tree(
        root,
        expected_files={record_name, "manifest.json", "completion.json", "evaluator_source.py"},
        expected_directories=set(),
    )
    record_path = require_regular(root / record_name, record_name)
    manifest_path = require_regular(root / "manifest.json", "manifest")
    completion_path = require_regular(root / "completion.json", "completion")
    record = load_json(record_path)
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    record_identity = require_hex64(record.get("identity_sha256"), "record identity")
    manifest_identity = require_hex64(
        manifest.get("identity_sha256"), "manifest identity"
    )
    if (
        canonical_sha256(without_identity(record)) != record_identity
        or canonical_sha256(without_identity(manifest)) != manifest_identity
        or manifest.get("artifact_kind") != artifact_kind
        or manifest.get("status") != "complete"
        or manifest.get("primary_record_name") != record_name
        or manifest.get("primary_record_identity_sha256") != record_identity
        or manifest.get("files") != artifact_records(root)
        or completion.get("complete") is not True
        or completion.get("artifact_kind") != artifact_kind
        or completion.get("primary_record_name") != record_name
        or completion.get("primary_record_identity_sha256") != record_identity
        or completion.get("primary_record_file_sha256") != sha256_file(record_path)
        or completion.get("manifest_identity_sha256") != manifest_identity
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
    ):
        raise RuntimeError(f"record lock validation failed: {root}")
    return record, manifest


def publish_record_lock(
    output: Path,
    *,
    artifact_kind: str,
    record_name: str,
    record: Mapping[str, Any],
) -> Path:
    output = output.expanduser().absolute()
    if os.path.lexists(output):
        raise RuntimeError(f"refusing to overwrite record lock: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        write_json(staging / record_name, record)
        shutil.copy2(Path(__file__).resolve(), staging / "evaluator_source.py")
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "artifact_kind": artifact_kind,
            "status": "complete",
            "primary_record_name": record_name,
            "primary_record_identity_sha256": record["identity_sha256"],
            "files": artifact_records(staging),
        }
        manifest["identity_sha256"] = canonical_sha256(manifest)
        write_json(staging / "manifest.json", manifest)
        write_json(
            staging / "completion.json",
            {
                "complete": True,
                "artifact_kind": artifact_kind,
                "primary_record_name": record_name,
                "primary_record_identity_sha256": record["identity_sha256"],
                "primary_record_file_sha256": sha256_file(staging / record_name),
                "manifest_identity_sha256": manifest["identity_sha256"],
                "manifest_file_sha256": sha256_file(staging / "manifest.json"),
            },
        )
        validate_record_lock(
            staging, artifact_kind=artifact_kind, record_name=record_name
        )
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def validate_source_lock(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = require_directory(root, "endpoint secondary protocol lock")
    expected_files = {
        "protocol.json",
        "reference_models.npz",
        "reference_summary.json",
        "manifest.json",
        "completion.json",
        *(f"sources/{name}" for name in SOURCE_BASENAMES),
    }
    exact_tree(root, expected_files=expected_files, expected_directories={"sources"})
    protocol_path = require_regular(root / "protocol.json", "endpoint protocol")
    summary_path = require_regular(root / "reference_summary.json", "reference summary")
    reference_path = require_regular(root / "reference_models.npz", "reference arrays")
    manifest_path = require_regular(root / "manifest.json", "endpoint manifest")
    completion_path = require_regular(root / "completion.json", "endpoint completion")
    protocol = load_json(protocol_path)
    summary = load_json(summary_path)
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    protocol_identity = require_hex64(
        protocol.get("identity_sha256"), "endpoint protocol identity"
    )
    summary_identity = require_hex64(
        summary.get("identity_sha256"), "reference summary identity"
    )
    manifest_identity = require_hex64(
        manifest.get("identity_sha256"), "endpoint manifest identity"
    )
    if (
        canonical_sha256(without_identity(protocol)) != protocol_identity
        or canonical_sha256(without_identity(summary)) != summary_identity
        or canonical_sha256(without_identity(manifest)) != manifest_identity
        or summary_identity != EXPECTED_REFERENCE_SUMMARY_IDENTITY
        or protocol.get("status")
        != "FROZEN_BEFORE_ANY_THIRD_POOL_LABEL_IMAGE_SCORE_OR_EMBEDDING_ACCESS"
        or summary.get("status")
        != "COMPLETE_OLD_DISCOVERY_CLEAN_REFERENCE_FROZEN_BEFORE_THIRD_LABEL_ACCESS"
        or manifest.get("status") != "complete"
        or manifest.get("experiment")
        != "dit_third_pool_endpoint_secondary_protocol_lock_v1"
        or manifest.get("protocol_identity_sha256") != protocol_identity
        or manifest.get("reference_summary_identity_sha256") != summary_identity
        or manifest.get("files") != artifact_records(root)
        or completion.get("complete") is not True
        or completion.get("protocol_identity_sha256") != protocol_identity
        or completion.get("protocol_file_sha256") != sha256_file(protocol_path)
        or completion.get("reference_summary_identity_sha256") != summary_identity
        or completion.get("reference_summary_file_sha256") != sha256_file(summary_path)
        or completion.get("reference_models_file_sha256") != sha256_file(reference_path)
        or completion.get("manifest_identity_sha256") != manifest_identity
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
        or completion.get("third_pool_data_opened") is not False
    ):
        raise RuntimeError("endpoint secondary protocol lock failed")

    foundation = protocol.get("foundation_identity_pins", {})
    family = protocol.get("secondary_family", {})
    gate = protocol.get("stage_a_gate_contract", {})
    if (
        foundation.get("sampling_protocol_identity_sha256")
        != EXPECTED_SAMPLING_PROTOCOL_IDENTITY
        or foundation.get("sampling_manifest_identity_sha256")
        != EXPECTED_SAMPLING_MANIFEST_IDENTITY
        or foundation.get("primary_evaluation_contract_identity_sha256")
        != EXPECTED_PRIMARY_CONTRACT_IDENTITY
        or foundation.get("primary_evaluation_source_manifest_identity_sha256")
        != EXPECTED_PRIMARY_SOURCE_MANIFEST_IDENTITY
        or foundation.get("primary_evaluator_source_sha256")
        != EXPECTED_PRIMARY_EVALUATOR_SHA256
        or family.get("family_size") != 2
        or tuple(family.get("ordered_ids", ())) != SECONDARY_IDS
        or family.get("method") != "Holm step-down across exactly E1 and E2"
        or family.get("strict_alpha") != 0.05
        or family.get("candidate_combination_allowed") is not False
        or gate.get("minimum_blur_or_soft_fusion_clear_bad") != EVENT_MIN_BLUR
        or gate.get("minimum_total_clear_bad") != EVENT_MIN_TOTAL_BAD
        or gate.get("both_minima_required") is not True
        or gate.get("authority")
        != "validate_stage_a_receipt from the exact frozen primary evaluator v5"
        or gate.get("failure_action")
        != "fail before endpoint product, consensus rows, sampling pool, or reference access"
    ):
        raise RuntimeError("endpoint secondary foundation/family/gate changed")

    reference_contract = protocol.get("reference", {})
    if reference_contract != {
        "summary_identity_sha256": EXPECTED_REFERENCE_SUMMARY_IDENTITY,
        "array_bundle_logical_identity_sha256": EXPECTED_REFERENCE_ARRAY_LOGICAL_IDENTITY,
        "old_discovery_seeds_inclusive": [50, 129],
        "old_discovery_clean_good_only": True,
        "clean_good_count_by_class": {"207": 71, "602": 70, "795": 75},
        "third_pool_refit_or_adaptation_allowed": False,
        "normalization": "rowwise L2",
        "knn_k": 5,
        "pooled_covariance": "within-class residual Ledoit-Wolf, one per representation",
    }:
        raise RuntimeError("endpoint clean reference contract changed")

    hypotheses = protocol.get("secondary_hypotheses", {})
    if hypotheses != {
        "E1_inception_blur_centroid": {
            "metric": E1_METRIC,
            "direction": "distance_high_is_bad",
            "endpoint": "blur_or_soft_fusion_clear_bad_vs_clean_good",
            "pair_weighted_auc_at_least": 0.75,
            "Holm_adjusted_p_strictly_below": 0.05,
            "role": "secondary_corroboration_only",
        },
        "E2_dino_allbad_mahalanobis": {
            "metric": E2_METRIC,
            "direction": "distance_high_is_bad",
            "endpoint": "all_clear_bad_vs_clean_good",
            "pair_weighted_auc_at_least": 0.70,
            "Holm_adjusted_p_strictly_below": 0.05,
            "role": "secondary_corroboration_only",
        },
    }:
        raise RuntimeError("endpoint secondary hypotheses changed")
    statistics = protocol.get("statistics", {})
    if (
        statistics.get("primary_auc")
        != (
            "sum over classes of distance-high concordant bad-good pairs, ties=0.5, "
            "divided by sum over classes n_positive_class*n_clean_good_class; "
            "zero total pair denominator fails closed"
        )
        or statistics.get("permutation_draws") != PERMUTATION_DRAWS
        or statistics.get("permutation_seed") != PERMUTATION_SEED
        or statistics.get("permutation_unit")
        != "one intact ordered three-class severity/phenotype block per global seed"
        or statistics.get("other_four_metrics_inferential_p_values_allowed") is not False
        or tuple(protocol.get("all_fixed_metrics", ())) != METRICS
    ):
        raise RuntimeError("endpoint secondary statistics changed")

    extraction = protocol.get("label_free_extraction", {})
    if (
        extraction.get("existing_extractor_compatible") is not True
        or extraction.get("inventory_adapter_required") is not False
        or extraction.get("required_seed_axis_half_open") != "250:850"
        or extraction.get("required_classes_ordered") != list(CLASSES)
        or extraction.get("expected_trajectory_count") != TRAJECTORY_COUNT
        or extraction.get("representations") != REPRESENTATIONS
        or extraction.get("labels_reviews_scores_or_distances_read_by_extractor") is not False
        or extraction.get("inception_weights", {}).get("sha256")
        != "6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2"
        or extraction.get("dino_snapshot", {}).get("revision")
        != "e4c89a4e05589de9b3e188688a303d0f3c04d0f3"
    ):
        raise RuntimeError("endpoint label-free extraction contract changed")

    expected_summary_cohort = {
        "old_discovery_seeds_inclusive": [50, 129],
        "ordered_classes": list(CLASSES),
        "old_discovery_trajectory_count": 240,
        "clean_good_count_by_class": {"207": 71, "602": 70, "795": 75},
        "clean_good_total": 216,
    }
    access = summary.get("evidence_access_audit", {})
    lineage = summary.get("input_lineage", {})
    if (
        summary.get("cohort") != expected_summary_cohort
        or summary.get("array_bundle_logical_identity_sha256")
        != EXPECTED_REFERENCE_ARRAY_LOGICAL_IDENTITY
        or access
        != {
            "old_label_free_endpoint_embeddings_opened": True,
            "old_discovery_final_consensus_rows_opened_for_clean_reference": True,
            "old_aggregate_endpoint_audit_opened": True,
            "old_expansion_consensus_rows_opened": False,
            "third_pool_path_opened_statted_or_hashed": False,
            "third_pool_image_trace_embedding_label_review_score_or_screen_opened": False,
            "third_pool_score_label_join_performed": False,
        }
        or lineage.get("old_endpoint_protocol")
        != {
            "identity_sha256": EXPECTED_OLD_ENDPOINT_PROTOCOL_IDENTITY,
            "file_sha256": EXPECTED_OLD_ENDPOINT_PROTOCOL_FILE_SHA256,
        }
        or lineage.get("old_endpoint_embeddings", {}).get("manifest_identity_sha256")
        != EXPECTED_OLD_EMBEDDING_MANIFEST_IDENTITY
        or lineage.get("old_endpoint_embeddings", {}).get("manifest_file_sha256")
        != EXPECTED_OLD_EMBEDDING_MANIFEST_FILE_SHA256
        or lineage.get("old_endpoint_embeddings", {}).get("embeddings_file_sha256")
        != EXPECTED_OLD_EMBEDDINGS_FILE_SHA256
        or lineage.get("old_endpoint_embeddings", {}).get("embedding_array_sha256")
        != EXPECTED_OLD_EMBEDDING_ARRAY_SHA256
        or lineage.get("old_discovery_consensus", {}).get("consensus_identity_sha256")
        != EXPECTED_OLD_LABEL_CONSENSUS_IDENTITY
        or lineage.get("old_discovery_consensus", {}).get("manifest_identity_sha256")
        != EXPECTED_OLD_LABEL_MANIFEST_IDENTITY
        or lineage.get("old_discovery_consensus", {}).get("manifest_file_sha256")
        != EXPECTED_OLD_LABEL_MANIFEST_FILE_SHA256
        or lineage.get("old_discovery_consensus", {}).get("consensus_file_sha256")
        != EXPECTED_OLD_LABEL_CONSENSUS_FILE_SHA256
        or lineage.get("old_discovery_consensus", {}).get("completion_file_sha256")
        != EXPECTED_OLD_LABEL_COMPLETION_FILE_SHA256
        or lineage.get("old_aggregate_endpoint_audit", {}).get("manifest_identity_sha256")
        != EXPECTED_OLD_AUDIT_MANIFEST_IDENTITY
        or lineage.get("old_aggregate_endpoint_audit", {}).get("manifest_file_sha256")
        != EXPECTED_OLD_AUDIT_MANIFEST_FILE_SHA256
        or lineage.get("old_aggregate_endpoint_audit", {}).get("completion_file_sha256")
        != EXPECTED_OLD_AUDIT_COMPLETION_FILE_SHA256
        or lineage.get("old_aggregate_endpoint_audit", {}).get("metric_results_file_sha256")
        != EXPECTED_OLD_AUDIT_METRIC_RESULTS_FILE_SHA256
    ):
        raise RuntimeError("endpoint clean reference lineage/access contract changed")

    source_records = protocol.get("source_snapshots", {})
    if set(source_records) != set(SOURCE_BASENAMES):
        raise RuntimeError("endpoint source snapshot set changed")
    for basename in SOURCE_BASENAMES:
        source = root / "sources" / basename
        if sha256_file(source) != source_records[basename].get("sha256"):
            raise RuntimeError(f"endpoint source snapshot changed: {basename}")
    if source_records["extract_dit_endpoint_embeddings_label_free.py"]["sha256"] != EXPECTED_ENDPOINT_EXTRACTOR_SHA256:
        raise RuntimeError("endpoint extractor source pin changed")
    if source_records["audit_dit_endpoint_representation_distances.py"]["sha256"] != EXPECTED_OLD_DISTANCE_HELPER_SHA256:
        raise RuntimeError("old endpoint-distance helper source pin changed")
    if source_records["evaluate_dit_bad_good_third_pool_confirmation.py"]["sha256"] != EXPECTED_PRIMARY_EVALUATOR_SHA256:
        raise RuntimeError("primary evaluator snapshot changed")
    invoked = Path(__file__).resolve()
    if (
        sha256_file(invoked) != source_records[invoked.name]["sha256"]
        or protocol.get("implementation_source_sha256")
        != source_records["evaluate_dit_third_pool_endpoint_secondary.py"]["sha256"]
    ):
        raise RuntimeError("invoked endpoint evaluator differs from frozen snapshot")

    expected_records = summary.get("array_records")
    if not isinstance(expected_records, dict):
        raise RuntimeError("reference array records missing")
    expected_shapes: dict[str, list[int]] = {}
    clean_counts = {207: 71, 602: 70, 795: 75}
    for representation, dimension in REPRESENTATIONS.items():
        for class_id in CLASSES:
            expected_shapes[_reference_key(representation, "center", class_id)] = [dimension]
            expected_shapes[_reference_key(representation, "unit_center", class_id)] = [dimension]
            expected_shapes[_reference_key(representation, "clean_points", class_id)] = [
                clean_counts[class_id],
                dimension,
            ]
        expected_shapes[_reference_key(representation, "covariance_basis")] = [216, dimension]
        expected_shapes[_reference_key(representation, "covariance_eigenvalues")] = [216]
        expected_shapes[_reference_key(representation, "shrinkage")] = [1]
        expected_shapes[_reference_key(representation, "isotropic")] = [1]
    if set(expected_records) != set(expected_shapes):
        raise RuntimeError("reference array member family changed")
    with np.load(reference_path, allow_pickle=False) as archive:
        if set(archive.files) != set(expected_shapes):
            raise RuntimeError("reference array member set changed")
        for name in archive.files:
            value = archive[name]
            record = expected_records[name]
            if (
                list(value.shape) != expected_shapes[name]
                or record.get("shape") != expected_shapes[name]
                or value.dtype.str != "<f8"
                or record.get("dtype") != "<f8"
                or sha256_array(value) != record.get("raw_sha256")
                or not np.isfinite(value).all()
            ):
                raise RuntimeError(f"reference array changed: {name}")
    if canonical_sha256(expected_records) != EXPECTED_REFERENCE_ARRAY_LOGICAL_IDENTITY:
        raise RuntimeError("reference array logical identity changed")
    if protocol.get("reference", {}).get("summary_identity_sha256") != summary_identity:
        raise RuntimeError("reference summary identity binding changed")
    return protocol, manifest


def primary_helper(source_lock: Path) -> ModuleType:
    # Import the live file only after proving it is byte-identical to the frozen
    # snapshot.  The primary evaluator resolves its foundation locks relative to
    # the repository, so importing the relocated snapshot would change ROOT.
    path = ROOT / "experiments/evaluate_dit_bad_good_third_pool_confirmation.py"
    if sha256_file(path) != EXPECTED_PRIMARY_EVALUATOR_SHA256:
        raise RuntimeError("live primary evaluator differs from the frozen source")
    return load_module(path, "_endpoint_secondary_primary_evaluator")


def endpoint_extractor_helper(source_lock: Path) -> ModuleType:
    validate_source_lock(source_lock)
    path = ROOT / "experiments/extract_dit_endpoint_embeddings_label_free.py"
    if sha256_file(path) != EXPECTED_ENDPOINT_EXTRACTOR_SHA256:
        raise RuntimeError("live endpoint extractor differs from the frozen source")
    return load_module(path, "_endpoint_secondary_embedding_extractor")


def bind_inputs(
    *,
    source_lock: Path,
    primary_input_lock: Path,
    endpoint_product: Path,
    endpoint_manifest_identity: str,
    output: Path,
) -> Path:
    protocol, source_manifest = validate_source_lock(source_lock)
    primary = primary_helper(source_lock)
    primary_binding, primary_binding_manifest = primary.validate_input_binding(
        primary_input_lock
    )
    record: dict[str, Any] = {
        "schema_version": 1,
        "status": "BOUND_PATHS_AND_IDENTITIES_WITHOUT_OPENING_ENDPOINT_OR_THIRD_DATA",
        "endpoint_protocol_identity_sha256": protocol["identity_sha256"],
        "endpoint_source_manifest_identity_sha256": source_manifest["identity_sha256"],
        "primary_input_lock": {
            "path": str(Path(primary_input_lock).expanduser().absolute()),
            "identity_sha256": primary_binding["identity_sha256"],
            "manifest_identity_sha256": primary_binding_manifest["identity_sha256"],
        },
        "endpoint_product": {
            "path": str(Path(endpoint_product).expanduser().absolute()),
            "manifest_identity_sha256": require_hex64(
                endpoint_manifest_identity, "endpoint embedding manifest identity"
            ),
        },
        "access_audit": {
            "primary_input_lock_validated": True,
            "primary_bound_third_paths_opened_or_statted": False,
            "endpoint_product_path_opened_or_statted": False,
            "consensus_rows_opened": False,
            "scientific_override_interface_exists": False,
        },
        "implementation_source_sha256": sha256_file(Path(__file__).resolve()),
    }
    record["identity_sha256"] = canonical_sha256(record)
    return publish_record_lock(
        output,
        artifact_kind="dit_third_pool_endpoint_secondary_input_binding_v1",
        record_name="input_binding.json",
        record=record,
    )


def validate_input_binding(
    root: Path, source_lock: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    record, manifest = validate_input_binding_envelope(root)
    protocol, source_manifest = validate_source_lock(source_lock)
    if (
        record.get("endpoint_protocol_identity_sha256") != protocol["identity_sha256"]
        or record.get("endpoint_source_manifest_identity_sha256")
        != source_manifest["identity_sha256"]
        or record.get("implementation_source_sha256")
        != protocol["implementation_source_sha256"]
    ):
        raise RuntimeError("endpoint input binding differs from frozen phase-0 lock")
    return record, manifest


def validate_input_binding_envelope(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate only the small binding; do not touch the reference/source lock."""

    record, manifest = validate_record_lock(
        root,
        artifact_kind="dit_third_pool_endpoint_secondary_input_binding_v1",
        record_name="input_binding.json",
    )
    if (
        record.get("status")
        != "BOUND_PATHS_AND_IDENTITIES_WITHOUT_OPENING_ENDPOINT_OR_THIRD_DATA"
        or record.get("implementation_source_sha256")
        != sha256_file(Path(__file__).resolve())
        or record.get("access_audit")
        != {
            "primary_input_lock_validated": True,
            "primary_bound_third_paths_opened_or_statted": False,
            "endpoint_product_path_opened_or_statted": False,
            "consensus_rows_opened": False,
            "scientific_override_interface_exists": False,
        }
    ):
        raise RuntimeError("endpoint input-binding contract changed")
    for key in ("primary_input_lock", "endpoint_product"):
        item = record.get(key)
        if not isinstance(item, dict) or not Path(str(item.get("path", ""))).is_absolute():
            raise RuntimeError(f"malformed endpoint input binding: {key}")
        require_hex64(item.get("manifest_identity_sha256"), f"{key} manifest identity")
    require_hex64(
        record["primary_input_lock"].get("identity_sha256"),
        "primary input-binding identity",
    )
    require_hex64(
        record.get("endpoint_protocol_identity_sha256"), "endpoint protocol identity"
    )
    require_hex64(
        record.get("endpoint_source_manifest_identity_sha256"),
        "endpoint source manifest identity",
    )
    return record, manifest


def normalize_rows(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2:
        raise RuntimeError("endpoint representation must be a matrix")
    norms = np.linalg.norm(value, axis=1, keepdims=True)
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        raise RuntimeError("endpoint representation has invalid row norms")
    return value / norms


def _reference_key(representation: str, component: str, class_id: int | None = None) -> str:
    suffix = f"__class{class_id}" if class_id is not None else ""
    return f"{representation}__{component}{suffix}"


def score_reference(
    source_lock: Path,
    arrays: Mapping[str, np.ndarray],
    classes: np.ndarray,
) -> dict[str, np.ndarray]:
    results: dict[str, np.ndarray] = {}
    with np.load(source_lock / "reference_models.npz", allow_pickle=False) as reference:
        for representation, dimension in REPRESENTATIONS.items():
            points = normalize_rows(arrays[representation])
            if points.shape != (TRAJECTORY_COUNT, dimension):
                raise RuntimeError(f"wrong endpoint representation shape: {representation}")
            cosine = np.empty(len(points), dtype=np.float64)
            mahalanobis = np.empty(len(points), dtype=np.float64)
            knn = np.empty(len(points), dtype=np.float64)
            basis = np.asarray(
                reference[_reference_key(representation, "covariance_basis")],
                dtype=np.float64,
            )
            eigenvalues = np.asarray(
                reference[_reference_key(representation, "covariance_eigenvalues")],
                dtype=np.float64,
            )
            shrinkage = float(
                reference[_reference_key(representation, "shrinkage")].reshape(-1)[0]
            )
            isotropic = float(
                reference[_reference_key(representation, "isotropic")].reshape(-1)[0]
            )
            parallel_denominator = isotropic + (1.0 - shrinkage) * eigenvalues
            for class_id in CLASSES:
                mask = classes == class_id
                subset = points[mask]
                center = np.asarray(
                    reference[_reference_key(representation, "center", class_id)],
                    dtype=np.float64,
                )
                unit_center = np.asarray(
                    reference[_reference_key(representation, "unit_center", class_id)],
                    dtype=np.float64,
                )
                clean = np.asarray(
                    reference[_reference_key(representation, "clean_points", class_id)],
                    dtype=np.float64,
                )
                cosine[mask] = 1.0 - subset @ unit_center
                residual = subset - center
                projections = residual @ basis.T
                residual_norm2 = np.sum(residual**2, axis=1)
                parallel_norm2 = np.sum(projections**2, axis=1)
                orthogonal_norm2 = np.maximum(residual_norm2 - parallel_norm2, 0.0)
                distance2 = orthogonal_norm2 / isotropic + np.sum(
                    projections**2 / parallel_denominator[None, :], axis=1
                )
                mahalanobis[mask] = np.sqrt(np.maximum(distance2, 0.0))
                neighbor_distance = np.clip(1.0 - subset @ clean.T, 0.0, 2.0)
                nearest = np.partition(neighbor_distance, 4, axis=1)[:, :5]
                knn[mask] = nearest.mean(axis=1)
            for distance, values in (
                ("cosine_to_class_clean_centroid", cosine),
                ("shared_ledoitwolf_mahalanobis", mahalanobis),
                ("knn5_mean_cosine_to_class_clean", knn),
            ):
                if not np.isfinite(values).all():
                    raise RuntimeError(f"non-finite endpoint distance: {representation}/{distance}")
                results[f"{representation}__{distance}"] = values
    if set(results) != set(METRICS):
        raise RuntimeError("endpoint distance family changed")
    return results


def load_endpoint_product(
    ref: Mapping[str, Any],
    source_lock: Path,
    pool_trace_lineage: Mapping[int, Mapping[str, str]],
    primary_binding: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    root = require_directory(Path(str(ref["path"])), "endpoint embedding product")
    extractor = endpoint_extractor_helper(source_lock)
    receipt = extractor.validate_output(root)
    manifest_path = require_regular(root / "manifest.json", "endpoint manifest")
    manifest = load_json(manifest_path)
    if (
        manifest.get("identity_sha256") != ref.get("manifest_identity_sha256")
        or receipt.get("manifest_identity_sha256") != ref.get("manifest_identity_sha256")
        or manifest.get("analysis_source_sha256") != EXPECTED_ENDPOINT_EXTRACTOR_SHA256
        or receipt.get("sample_count") != TRAJECTORY_COUNT
        or receipt.get("seed_count") != len(SEEDS)
        or receipt.get("representation_count") != len(REPRESENTATIONS)
    ):
        raise RuntimeError("endpoint embedding identity/cohort changed")
    rows: list[dict[str, Any]] = []
    with (root / "sample_index.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "sample_index",
            "global_seed",
            "class_slot",
            "class_id",
            "trace_identity_sha256",
            "endpoint_sha256",
            "endpoint_pixel_sha256",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise RuntimeError("endpoint embedding index columns changed")
        if any(name in reader.fieldnames for name in ("label", "primary_label", "severity")):
            raise RuntimeError("endpoint embedding index contains supervision")
        for raw in reader:
            try:
                sample_index = int(raw["sample_index"])
                seed = int(raw["global_seed"])
                slot = int(raw["class_slot"])
                class_id = int(raw["class_id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError("invalid endpoint embedding identifier") from exc
            if (
                sample_index != (seed - SEEDS[0]) * len(CLASSES) + slot
                or seed not in SEEDS
                or slot not in range(len(CLASSES))
                or class_id != CLASSES[slot]
                or raw["trace_identity_sha256"]
                != pool_trace_lineage[seed]["identity_sha256"]
            ):
                raise RuntimeError("endpoint embedding axis/trace lineage changed")
            require_hex64(raw["endpoint_sha256"], "endpoint PNG hash")
            require_hex64(raw["endpoint_pixel_sha256"], "endpoint pixel hash")
            rows.append(
                {
                    "sample_index": sample_index,
                    "global_seed": seed,
                    "class_slot": slot,
                    "class_id": class_id,
                }
            )
    if len(rows) != TRAJECTORY_COUNT or [row["sample_index"] for row in rows] != list(
        range(TRAJECTORY_COUNT)
    ):
        raise RuntimeError("endpoint embedding index is not exact ordered third pool")
    inventory = load_json(root / "source_inventory.json")
    source_bindings = inventory.get("input_label_free_source_analyses")
    primary_ref = primary_binding["inputs"]["primary_label_free_product"]
    if (
        inventory.get("ordered_seeds") != list(SEEDS)
        or inventory.get("ordered_classes") != list(CLASSES)
        or not isinstance(source_bindings, list)
        or not any(
            isinstance(item, dict)
            and item.get("manifest_identity_sha256")
            == primary_ref["manifest_identity_sha256"]
            and str(Path(str(item.get("path", ""))).absolute())
            == str((Path(primary_ref["path"]).absolute() / "source_inventory.json"))
            for item in source_bindings
        )
    ):
        raise RuntimeError("endpoint embeddings are not bound to primary label-free inventory")
    arrays: dict[str, np.ndarray] = {}
    with np.load(root / "embeddings.npz", allow_pickle=False) as archive:
        if set(archive.files) != set(REPRESENTATIONS):
            raise RuntimeError("endpoint representation set changed")
        for name, dimension in REPRESENTATIONS.items():
            value = np.asarray(archive[name])
            if (
                value.shape != (TRAJECTORY_COUNT, dimension)
                or value.dtype.str != "<f4"
                or not np.isfinite(value).all()
            ):
                raise RuntimeError(f"endpoint representation changed: {name}")
            arrays[name] = value
    return rows, arrays, {
        "manifest_identity_sha256": manifest["identity_sha256"],
        "manifest_file_sha256": sha256_file(manifest_path),
        "embeddings_file_sha256": sha256_file(root / "embeddings.npz"),
        "sample_index_file_sha256": sha256_file(root / "sample_index.csv"),
    }


def auc_bad_high(
    scores: np.ndarray, positive: np.ndarray, good: np.ndarray
) -> tuple[float, int, float]:
    bad_scores = np.asarray(scores[positive], dtype=np.float64)
    good_scores = np.sort(np.asarray(scores[good], dtype=np.float64))
    pairs = len(bad_scores) * len(good_scores)
    if pairs == 0:
        return float("nan"), 0, 0.0
    left = np.searchsorted(good_scores, bad_scores, side="left")
    right = np.searchsorted(good_scores, bad_scores, side="right")
    wins = float(np.sum(left + 0.5 * (right - left)))
    return wins / pairs, pairs, wins


def auc_summary(
    scores: np.ndarray,
    positive: np.ndarray,
    good: np.ndarray,
    classes: np.ndarray,
    *,
    fail_on_zero_pairs: bool = True,
) -> dict[str, Any]:
    numerator = 0.0
    denominator = 0
    per_class: dict[str, Any] = {}
    for class_id in CLASSES:
        mask = classes == class_id
        auc, pairs, wins = auc_bad_high(scores[mask], positive[mask], good[mask])
        positives = int(np.sum(positive[mask]))
        cleans = int(np.sum(good[mask]))
        per_class[str(class_id)] = {
            "positive_count": positives,
            "clean_good_count": cleans,
            "pair_count": pairs,
            "auc": auc if positives >= MINIMUM_GROUP_SIZE and cleans >= MINIMUM_GROUP_SIZE else None,
            "suppressed_below_minimum_group_size": not (
                positives >= MINIMUM_GROUP_SIZE and cleans >= MINIMUM_GROUP_SIZE
            ),
        }
        numerator += wins
        denominator += pairs
    if denominator == 0 and fail_on_zero_pairs:
        raise RuntimeError("endpoint AUC has zero total class-matched pair denominator")
    return {
        "pair_weighted_auc": numerator / denominator if denominator else None,
        "pair_count": denominator,
        "positive_count": int(np.sum(positive)),
        "clean_good_count": int(np.sum(good)),
        "per_class": per_class,
    }


def _tie_groups(sorted_scores: np.ndarray) -> list[tuple[int, int]]:
    boundaries = np.flatnonzero(np.diff(sorted_scores) != 0) + 1
    starts = np.r_[0, boundaries]
    stops = np.r_[boundaries, len(sorted_scores)]
    return [
        (int(start), int(stop))
        for start, stop in zip(starts, stops, strict=True)
        if stop - start > 1
    ]


def block_permutation_pvalues(
    score_by_secondary: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
    *,
    draws: int,
    seed: int,
    batch_size: int = PERMUTATION_BATCH,
) -> dict[str, float]:
    ordered = sorted(rows, key=lambda row: int(row["sample_index"]))
    if [row["sample_index"] for row in ordered] != list(range(TRAJECTORY_COUNT)):
        raise RuntimeError("permutation rows are not in exact sample order")
    good = np.asarray([row["final_severity"] == "clean_good" for row in ordered]).reshape(
        len(SEEDS), len(CLASSES)
    )
    positives = {
        "E1_inception_blur_centroid": np.asarray(
            [
                row["final_severity"] == "clear_bad"
                and row["blur_component_consensus"]
                for row in ordered
            ]
        ).reshape(len(SEEDS), len(CLASSES)),
        "E2_dino_allbad_mahalanobis": np.asarray(
            [row["final_severity"] == "clear_bad" for row in ordered]
        ).reshape(len(SEEDS), len(CLASSES)),
    }
    observed: dict[str, tuple[float, int]] = {}
    score_matrices: dict[str, np.ndarray] = {}
    class_array = np.tile(np.asarray(CLASSES), len(SEEDS))
    for secondary in SECONDARY_IDS:
        score = np.asarray(score_by_secondary[secondary], dtype=np.float64)
        summary = auc_summary(
            score,
            positives[secondary].reshape(-1),
            good.reshape(-1),
            class_array,
        )
        observed[secondary] = (summary["pair_weighted_auc"], summary["pair_count"])
        score_matrices[secondary] = score.reshape(len(SEEDS), len(CLASSES)).T
    exceed = {secondary: 0 for secondary in SECONDARY_IDS}
    rng = np.random.default_rng(seed)
    completed = 0
    while completed < draws:
        size = min(batch_size, draws - completed)
        permutations = np.argsort(
            rng.random((size, len(SEEDS))), axis=1, kind="quicksort"
        )
        permuted_good = good[permutations]
        for secondary in SECONDARY_IDS:
            permuted_positive = positives[secondary][permutations]
            numerator = np.zeros(size, dtype=np.float64)
            scores_by_class = score_matrices[secondary]
            for class_index in range(len(CLASSES)):
                scores = scores_by_class[class_index]
                order = np.argsort(scores, kind="mergesort")
                good_ordered = permuted_good[:, :, class_index][:, order]
                positive_ordered = permuted_positive[:, :, class_index][:, order]
                cumulative_good = np.cumsum(good_ordered, axis=1)
                credit = cumulative_good - good_ordered + 0.5 * good_ordered
                for start, stop in _tie_groups(scores[order]):
                    before = (
                        cumulative_good[:, start - 1]
                        if start
                        else np.zeros(size, dtype=np.float64)
                    )
                    within = np.sum(good_ordered[:, start:stop], axis=1)
                    credit[:, start:stop] = before[:, None] + 0.5 * within[:, None]
                numerator += np.sum(credit * positive_ordered, axis=1)
            observed_auc, denominator = observed[secondary]
            exceed[secondary] += int(
                np.sum(numerator >= observed_auc * denominator - 1e-12)
            )
        completed += size
    return {
        secondary: (exceed[secondary] + 1.0) / (draws + 1.0)
        for secondary in SECONDARY_IDS
    }


def holm_two(pvalues: Mapping[str, float]) -> dict[str, float]:
    if set(pvalues) != set(SECONDARY_IDS):
        raise RuntimeError("Holm family differs from exactly E1/E2")
    order = sorted(SECONDARY_IDS, key=lambda name: (pvalues[name], name))
    first, second = order
    return {
        first: min(1.0, 2.0 * pvalues[first]),
        second: min(1.0, max(2.0 * pvalues[first], pvalues[second])),
    }


def endpoint_masks(rows: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    return {
        "all_clear_bad_vs_clean_good": np.asarray(
            [row["final_severity"] == "clear_bad" for row in rows]
        ),
        "blur_or_soft_fusion_clear_bad_vs_clean_good": np.asarray(
            [
                row["final_severity"] == "clear_bad"
                and row["blur_component_consensus"]
                for row in rows
            ]
        ),
        "mixed_blur_and_structure_clear_bad_vs_clean_good": np.asarray(
            [
                row["final_severity"] == "clear_bad"
                and row["blur_component_consensus"]
                and row["discrete_structure_component_consensus"]
                for row in rows
            ]
        ),
        "structural_non_blur_clear_bad_vs_clean_good": np.asarray(
            [
                row["final_severity"] == "clear_bad"
                and not row["blur_component_consensus"]
                and row["discrete_structure_component_consensus"]
                for row in rows
            ]
        ),
    }


def build_results(
    scores: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
    *,
    draws: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ordered = sorted(rows, key=lambda row: int(row["sample_index"]))
    classes = np.asarray([row["class_id"] for row in ordered], dtype=int)
    good = np.asarray([row["final_severity"] == "clean_good" for row in ordered])
    masks = endpoint_masks(ordered)
    secondary_scores = {
        SECONDARY_IDS[0]: scores[E1_METRIC],
        SECONDARY_IDS[1]: scores[E2_METRIC],
    }
    primary_summaries = {
        SECONDARY_IDS[0]: auc_summary(
            secondary_scores[SECONDARY_IDS[0]],
            masks["blur_or_soft_fusion_clear_bad_vs_clean_good"],
            good,
            classes,
        ),
        SECONDARY_IDS[1]: auc_summary(
            secondary_scores[SECONDARY_IDS[1]],
            masks["all_clear_bad_vs_clean_good"],
            good,
            classes,
        ),
    }
    raw_p = block_permutation_pvalues(
        secondary_scores,
        ordered,
        draws=draws,
        seed=PERMUTATION_SEED,
    )
    holm = holm_two(raw_p)
    gates = {SECONDARY_IDS[0]: 0.75, SECONDARY_IDS[1]: 0.70}
    secondary_results: dict[str, Any] = {}
    for secondary in SECONDARY_IDS:
        passed = bool(
            primary_summaries[secondary]["pair_weighted_auc"] >= gates[secondary]
            and holm[secondary] < 0.05
        )
        secondary_results[secondary] = {
            **primary_summaries[secondary],
            "raw_block_permutation_p_one_sided": raw_p[secondary],
            "Holm_adjusted_p_across_exactly_two": holm[secondary],
            "auc_gate_at_least": gates[secondary],
            "strict_Holm_gate_below": 0.05,
            "all_frozen_gates_pass": passed,
            "can_rescue_primary_candidate": False,
            "can_authorize_intervention": False,
        }
    descriptive: list[dict[str, Any]] = []
    for metric in METRICS:
        for endpoint, positive in masks.items():
            summary = auc_summary(
                scores[metric],
                positive,
                good,
                classes,
                fail_on_zero_pairs=False,
            )
            descriptive.append(
                {
                    "metric": metric,
                    "endpoint": endpoint,
                    "direction": "distance_high_is_bad",
                    "pair_weighted_auc": summary["pair_weighted_auc"],
                    "pair_count": summary["pair_count"],
                    "positive_count": summary["positive_count"],
                    "clean_good_count": summary["clean_good_count"],
                    "per_class": summary["per_class"],
                    "inferential_p_value_computed": metric in {E1_METRIC, E2_METRIC}
                    and (
                        (metric == E1_METRIC and endpoint == "blur_or_soft_fusion_clear_bad_vs_clean_good")
                        or (metric == E2_METRIC and endpoint == "all_clear_bad_vs_clean_good")
                    ),
                    "descriptive_cannot_replace_secondary": True,
                }
            )
    result = {
        "secondary_family": secondary_results,
        "family_size": 2,
        "multiple_testing": "Holm step-down across exactly E1 and E2",
        "candidate_combination_performed": False,
        "primary_B_or_C_conclusion_changed": False,
        "intervention_authorized": False,
    }
    return result, descriptive


def publish_result(
    output: Path,
    *,
    source_lock: Path,
    binding: Mapping[str, Any],
    binding_manifest: Mapping[str, Any],
    stage_a: Mapping[str, Any],
    stage_a_manifest: Mapping[str, Any],
    secondary: Mapping[str, Any],
    descriptive: Sequence[Mapping[str, Any]],
    lineage: Mapping[str, Any],
) -> Path:
    output = output.expanduser().absolute()
    if os.path.lexists(output):
        raise RuntimeError(f"refusing to overwrite endpoint result: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        result: dict[str, Any] = {
            "schema_version": 1,
            "status": "COMPLETE_AGGREGATE_ONLY_ENDPOINT_SECONDARY",
            "endpoint_protocol_identity_sha256": binding[
                "endpoint_protocol_identity_sha256"
            ],
            "endpoint_input_binding_identity_sha256": binding["identity_sha256"],
            "endpoint_input_binding_manifest_identity_sha256": binding_manifest[
                "identity_sha256"
            ],
            "stage_a_receipt_identity_sha256": stage_a["identity_sha256"],
            "stage_a_manifest_identity_sha256": stage_a_manifest["identity_sha256"],
            "stage_a_event_gate": stage_a["event_gate"],
            "statistics": dict(secondary),
            "input_lineage": dict(lineage),
            "output_scope": {
                "aggregate_only": True,
                "sample_rows_emitted": False,
                "sample_distances_or_ranks_emitted": False,
                "images_or_trajectory_paths_emitted": False,
                "permutation_draws_emitted": False,
            },
            "access_audit": {
                "stage_a_validated_before_any_third_endpoint_or_row_access": True,
                "both_event_minima_replayed_as_passed": True,
                "endpoint_embedding_opened_after_gate": True,
                "consensus_rows_opened_after_gate": True,
                "primary_or_visual_BC_score_products_opened": False,
            },
            "implementation_source_sha256": sha256_file(Path(__file__).resolve()),
        }
        result["identity_sha256"] = canonical_sha256(result)
        write_json(staging / "secondary_results.json", result)
        write_json(staging / "descriptive_guardrails.json", {"rows": list(descriptive)})
        shutil.copy2(Path(__file__).resolve(), staging / "evaluator_source.py")
        protocol, source_manifest = validate_source_lock(source_lock)
        write_json(
            staging / "methodology.json",
            {
                "status": "AGGREGATE_ONLY_SECONDARY_CORROBORATION",
                "protocol_identity_sha256": protocol["identity_sha256"],
                "source_manifest_identity_sha256": source_manifest["identity_sha256"],
                "warning": (
                    "Terminal supervised typicality audit only; it cannot define labels, "
                    "rescue B/C, authorize intervention, or act as an online signal."
                ),
            },
        )
        members = []
        for path in sorted(staging.iterdir()):
            members.append(
                {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            )
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "complete",
            "experiment": "dit_third_pool_endpoint_secondary_v1",
            "result_identity_sha256": result["identity_sha256"],
            "files": members,
        }
        manifest["identity_sha256"] = canonical_sha256(manifest)
        write_json(staging / "manifest.json", manifest)
        write_json(
            staging / "completion.json",
            {
                "complete": True,
                "result_identity_sha256": result["identity_sha256"],
                "result_file_sha256": sha256_file(staging / "secondary_results.json"),
                "manifest_identity_sha256": manifest["identity_sha256"],
                "manifest_file_sha256": sha256_file(staging / "manifest.json"),
                "aggregate_only": True,
            },
        )
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def run_evaluation(
    *,
    source_lock: Path,
    input_lock: Path,
    stage_a_receipt: Path,
    output: Path,
    _self_test_draws: int | None = None,
) -> Path:
    # Open only the small endpoint binding and the primary evaluator's own locks
    # until Stage A is proven.  In particular, reference_models.npz is not opened,
    # statted, or hashed on a failed event gate.
    binding, binding_manifest = validate_input_binding_envelope(input_lock)
    primary = primary_helper(source_lock)
    primary_binding_root = Path(binding["primary_input_lock"]["path"])
    primary_binding, primary_binding_manifest = primary.validate_input_binding(
        primary_binding_root
    )
    if (
        primary_binding["identity_sha256"]
        != binding["primary_input_lock"]["identity_sha256"]
        or primary_binding_manifest["identity_sha256"]
        != binding["primary_input_lock"]["manifest_identity_sha256"]
    ):
        raise RuntimeError("primary input binding differs from endpoint binding")

    # Mandatory first access boundary.  This function raises on every failed or
    # forged gate before any path below is read, statted, or hashed.
    stage_a, stage_a_manifest = primary.validate_stage_a_receipt(
        stage_a_receipt, primary_binding, primary_binding_manifest
    )
    if (
        stage_a["event_gate"].get("both_minima_met") is not True
        or stage_a["event_gate"].get("stage_B_authorized") is not True
    ):
        raise RuntimeError("endpoint secondary is not authorized by Stage A")

    protocol, _ = validate_source_lock(source_lock)
    fully_validated_binding, fully_validated_manifest = validate_input_binding(
        input_lock, source_lock
    )
    if (
        fully_validated_binding != binding
        or fully_validated_manifest != binding_manifest
    ):
        raise RuntimeError("endpoint input binding changed across the Stage-A boundary")

    pool_trace_lineage, pool_lineage = primary.validate_sampling_pool(
        primary_binding["inputs"]["sampling_pool"]
    )
    consensus_rows, consensus_lineage = primary.load_full_consensus(
        primary_binding["inputs"]["consensus"],
        {**stage_a["consensus_receipt"], "counts": stage_a["aggregate_counts"]},
    )
    endpoint_rows, embeddings, endpoint_lineage = load_endpoint_product(
        binding["endpoint_product"],
        source_lock,
        pool_trace_lineage,
        primary_binding,
    )
    endpoint_keys = [
        (row["sample_index"], row["global_seed"], row["class_slot"], row["class_id"])
        for row in endpoint_rows
    ]
    consensus_keys = [
        (row["sample_index"], row["global_seed"], row["class_slot"], row["class_id"])
        for row in sorted(consensus_rows, key=lambda item: int(item["sample_index"]))
    ]
    if endpoint_keys != consensus_keys:
        raise RuntimeError("endpoint embedding and consensus axes differ")
    ordered_consensus = sorted(consensus_rows, key=lambda item: int(item["sample_index"]))
    classes = np.asarray([row["class_id"] for row in ordered_consensus], dtype=int)
    scores = score_reference(source_lock, embeddings, classes)
    draws = PERMUTATION_DRAWS if _self_test_draws is None else _self_test_draws
    secondary, descriptive = build_results(scores, ordered_consensus, draws=draws)
    return publish_result(
        output,
        source_lock=source_lock,
        binding=binding,
        binding_manifest=binding_manifest,
        stage_a=stage_a,
        stage_a_manifest=stage_a_manifest,
        secondary=secondary,
        descriptive=descriptive,
        lineage={
            "sampling_pool": pool_lineage,
            "consensus": consensus_lineage,
            "endpoint_embeddings": endpoint_lineage,
            "reference_summary_identity_sha256": protocol["reference"][
                "summary_identity_sha256"
            ],
        },
    )


def require_gate_before_loader(
    gate: Mapping[str, Any], loader: Callable[[], Any]
) -> Any:
    if (
        gate.get("both_minima_met") is not True
        or gate.get("stage_B_authorized") is not True
        or gate.get("observed_blur_or_soft_fusion_clear_bad", -1) < EVENT_MIN_BLUR
        or gate.get("observed_total_clear_bad", -1) < EVENT_MIN_TOTAL_BAD
    ):
        raise RuntimeError("synthetic Stage-A gate blocks endpoint loader")
    return loader()


def synthetic_self_test() -> None:
    touched = False

    def poisoned_loader() -> None:
        nonlocal touched
        touched = True
        raise AssertionError("failed gate touched poisoned endpoint loader")

    failed_gate = {
        "observed_blur_or_soft_fusion_clear_bad": 14,
        "observed_total_clear_bad": 30,
        "both_minima_met": False,
        "stage_B_authorized": False,
    }
    try:
        require_gate_before_loader(failed_gate, poisoned_loader)
    except RuntimeError:
        pass
    else:
        raise AssertionError("failed event gate was accepted")
    assert touched is False

    rng = np.random.default_rng(91)
    rows: list[dict[str, Any]] = []
    scores = {metric: np.empty(TRAJECTORY_COUNT, dtype=np.float64) for metric in METRICS}
    for seed_index, seed in enumerate(SEEDS):
        for slot, class_id in enumerate(CLASSES):
            index = seed_index * len(CLASSES) + slot
            bad = seed_index < 20
            blur = seed_index < 10
            severity = "clear_bad" if bad else ("mild_or_disputed" if seed_index == 20 else "clean_good")
            rows.append(
                {
                    "sample_index": index,
                    "global_seed": seed,
                    "class_slot": slot,
                    "class_id": class_id,
                    "final_severity": severity,
                    "blur_component_consensus": blur,
                    "discrete_structure_component_consensus": bad and not blur,
                }
            )
            for metric in METRICS:
                shift = 2.0 if bad else 0.0
                if metric == E1_METRIC and blur:
                    shift += 2.0
                scores[metric][index] = shift + rng.normal(0.0, 0.2)
    result, descriptive = build_results(scores, rows, draws=99)
    assert set(result["secondary_family"]) == set(SECONDARY_IDS)
    assert result["family_size"] == 2
    assert len(descriptive) == len(METRICS) * 4
    assert all(
        math.isfinite(item["pair_weighted_auc"])
        for item in descriptive
        if item["pair_count"] > 0
    )
    assert holm_two({SECONDARY_IDS[0]: 0.01, SECONDARY_IDS[1]: 0.04}) == {
        SECONDARY_IDS[0]: 0.02,
        SECONDARY_IDS[1]: 0.04,
    }
    print(
        "synthetic endpoint-secondary self-test passed: gate-before-loader, exact "
        "two-test Holm family, class-pair AUC, block permutation, and aggregate guards"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-lock", type=Path, default=DEFAULT_SOURCE_LOCK)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--validate-source-lock", action="store_true")
    mode.add_argument("--bind-inputs", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    parser.add_argument("--primary-input-lock", type=Path)
    parser.add_argument("--endpoint-product", type=Path)
    parser.add_argument("--endpoint-manifest-identity")
    parser.add_argument("--input-lock", type=Path)
    parser.add_argument("--stage-a-receipt", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        synthetic_self_test()
        return 0
    if args.validate_source_lock:
        protocol, manifest = validate_source_lock(args.source_lock)
        print(
            json.dumps(
                {
                    "protocol_identity_sha256": protocol["identity_sha256"],
                    "manifest_identity_sha256": manifest["identity_sha256"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.bind_inputs:
        if (
            args.primary_input_lock is None
            or args.endpoint_product is None
            or args.endpoint_manifest_identity is None
            or args.output is None
        ):
            raise ValueError("--bind-inputs requires primary lock, endpoint path/identity, output")
        path = bind_inputs(
            source_lock=args.source_lock,
            primary_input_lock=args.primary_input_lock,
            endpoint_product=args.endpoint_product,
            endpoint_manifest_identity=args.endpoint_manifest_identity,
            output=args.output,
        )
        print(path)
        return 0
    if args.input_lock is None or args.stage_a_receipt is None or args.output is None:
        raise ValueError("--evaluate requires --input-lock, --stage-a-receipt, and --output")
    path = run_evaluation(
        source_lock=args.source_lock,
        input_lock=args.input_lock,
        stage_a_receipt=args.stage_a_receipt,
        output=args.output,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
