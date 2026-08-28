#!/usr/bin/env python3
"""Freeze every source file and command used after confirmation trace generation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_LOCK = ROOT / "experiments/locks/dit_bad_good_candidate_confirmation_lock_v5"
OUTPUT = ROOT / "experiments/locks/dit_bad_good_confirmation_execution_source_lock_v6"
DATA_ROOT = Path(os.environ.get("EQVAE_DATA_ROOT", "/data/users/zhoushunyu/eqvae"))
TRACE_ROOT = (
    DATA_ROOT / "cross_scale_evidence/dit_bad_good_confirmation_v1_custom_traces_cfg_locked"
)
FEATURE_ROOT = DATA_ROOT / "cross_scale_evidence/bad_good_metric_confirmation_v1"
EXPECTED_PROTOCOL_IDENTITY = "198a82a7c8a0ab79d901c76a5c810f4a40889604a66f18e995d0699f73c12bce"
PRIOR_EXECUTION_LOCKS = {
    "pre_feature_extraction_v3": (
        ROOT / "experiments/locks/dit_bad_good_confirmation_execution_source_lock_v3",
        "299dccb91eb903cd16d892c94a249ccec34508c0b3383a6f3788b42e75f3d5ec",
    ),
    "pre_candidate_scoring_v4": (
        ROOT / "experiments/locks/dit_bad_good_confirmation_execution_source_lock_v4",
        "a2e0f9d9865f9a0eda2e25b7b695f19db7d1632b52b790aba11d824284ee15f9",
    ),
    "post_execution_source_audit_v5": (
        ROOT / "experiments/locks/dit_bad_good_confirmation_execution_source_lock_v5",
        "2f485c184d8f3ce2a3369306b6f5c27b025c984ef90dbc4bcb74cb94ed2a6d73",
    ),
}
EXECUTED_PRODUCT_ROOTS = {
    "primary_label_free_features": FEATURE_ROOT / "custom_label_free_v1",
    "posterior_label_free_features": FEATURE_ROOT / "posterior_label_free_v1",
    "frozen_candidate_scores": FEATURE_ROOT / "frozen_candidate_scores_label_free_v1",
    "label_free_calibration": ROOT / "experiments/locks/dit_bad_good_conformal_calibration_lock_v1",
    "calibrated_evaluation_alerts": FEATURE_ROOT / "calibrated_evaluation_alerts_label_free_v1",
}

SOURCE_FILES = (
    ROOT / "experiments/trace_dit_imagenet256_custom_batch.py",
    ROOT / "experiments/sample_dit_imagenet256_custom.py",
    ROOT / "experiments/reproduce_dit_imagenet256.py",
    ROOT / "experiments/run_dit_bad_good_confirmation_pool.py",
    ROOT / "experiments/analyze_dit_bad_good_custom_traces.py",
    ROOT / "experiments/analyze_dit_posterior_evidence_metrics.py",
    ROOT / "experiments/score_dit_bad_good_frozen_candidates.py",
    ROOT / "experiments/calibrate_dit_bad_good_conformal_thresholds.py",
    ROOT / "experiments/apply_dit_bad_good_conformal_thresholds.py",
    Path(__file__).resolve(),
)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def validate_candidate_lock() -> dict[str, Any]:
    manifest_path = CANDIDATE_LOCK / "manifest.json"
    protocol_path = CANDIDATE_LOCK / "candidate_protocol.json"
    manifest = load_json(manifest_path)
    completion = load_json(CANDIDATE_LOCK / "completion.json")
    protocol = load_json(protocol_path)
    if (
        manifest.get("status") != "complete"
        or completion.get("complete") is not True
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
        or completion.get("protocol_file_sha256") != sha256_file(protocol_path)
        or protocol.get("identity_sha256") != EXPECTED_PROTOCOL_IDENTITY
        or completion.get("protocol_identity_sha256") != EXPECTED_PROTOCOL_IDENTITY
    ):
        raise RuntimeError("candidate v5 lock is invalid")
    return protocol


def validate_prior_execution_locks() -> dict[str, Any]:
    lineage: dict[str, Any] = {}
    source_maps: dict[str, dict[str, str]] = {}
    for role, (root, expected_identity) in PRIOR_EXECUTION_LOCKS.items():
        manifest_path = root / "manifest.json"
        record_path = root / "execution_sources_locked.json"
        manifest = load_json(manifest_path)
        completion = load_json(root / "completion.json")
        record = load_json(record_path)
        record_without_identity = dict(record)
        observed_identity = record_without_identity.pop("identity_sha256", None)
        manifest_without_identity = dict(manifest)
        manifest_identity = manifest_without_identity.pop("identity_sha256", None)
        if (
            manifest.get("status") != "complete"
            or completion.get("complete") is not True
            or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
            or completion.get("manifest_identity_sha256") != manifest_identity
            or manifest_identity != canonical_sha256(manifest_without_identity)
            or completion.get("execution_sources_file_sha256") != sha256_file(record_path)
            or completion.get("execution_sources_identity_sha256") != expected_identity
            or observed_identity != expected_identity
            or observed_identity != canonical_sha256(record_without_identity)
            or record.get("candidate_protocol_identity_sha256") != EXPECTED_PROTOCOL_IDENTITY
        ):
            raise RuntimeError(f"invalid prior execution lock: {role}")
        members = {item["name"]: item for item in manifest.get("files", [])}
        for name, item in members.items():
            path = root / name
            if (
                not path.is_file()
                or path.is_symlink()
                or path.stat().st_size != item.get("bytes")
                or sha256_file(path) != item.get("sha256")
            ):
                raise RuntimeError(f"prior execution member changed: {path}")
        source_map = record.get("source_sha256_by_basename")
        if not isinstance(source_map, dict):
            raise RuntimeError(f"prior execution lock lacks source hashes: {role}")
        source_maps[role] = source_map
        lineage[role] = {
            "path": str(root.resolve()),
            "execution_sources_identity_sha256": observed_identity,
            "execution_sources_file_sha256": sha256_file(record_path),
            "manifest_identity_sha256": manifest_identity,
            "manifest_file_sha256": sha256_file(manifest_path),
            "status": record.get("status"),
        }
    shared_names = set.intersection(*(set(value) for value in source_maps.values()))
    shared_names.discard(Path(__file__).name)
    for name in shared_names:
        if len({source_maps[role][name] for role in source_maps}) != 1:
            raise RuntimeError(f"prior execution locks disagree on frozen source: {name}")
    lineage["cross_lock_source_audit"] = {
        "hash_identical_source_basenames_excluding_locker": sorted(shared_names),
        "locker_excluded_because_each_lock_snapshots_its_own_version": True,
    }
    return lineage


def validate_executed_products(source_hashes: dict[str, str]) -> dict[str, Any]:
    manifests: dict[str, dict[str, Any]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    lineage: dict[str, Any] = {}
    for role, root in EXECUTED_PRODUCT_ROOTS.items():
        if not root.is_dir() or root.is_symlink():
            raise RuntimeError(f"executed product is missing or indirect: {root}")
        manifest_path = root / "manifest.json"
        completion_path = root / "completion.json"
        manifest = load_json(manifest_path)
        completion = load_json(completion_path)
        manifest_without_identity = dict(manifest)
        manifest_identity = manifest_without_identity.pop("identity_sha256", None)
        if (
            manifest.get("status") != "complete"
            or completion.get("complete") is not True
            or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
            or completion.get("manifest_identity_sha256") != manifest_identity
            or manifest_identity != canonical_sha256(manifest_without_identity)
        ):
            raise RuntimeError(f"invalid executed product receipt: {role}")
        members = {item["name"]: item for item in manifest.get("files", [])}
        if len(members) != len(manifest.get("files", [])):
            raise RuntimeError(f"duplicate executed product member: {role}")
        for name, item in members.items():
            path = root / name
            if (
                not path.is_file()
                or path.is_symlink()
                or path.stat().st_size != item.get("bytes")
                or sha256_file(path) != item.get("sha256")
            ):
                raise RuntimeError(f"executed product member changed: {path}")
        if (root / "summary.json").is_file():
            semantic_path = root / "summary.json"
        elif (root / "calibration_locked.json").is_file():
            semantic_path = root / "calibration_locked.json"
        else:
            raise RuntimeError(f"executed product lacks semantic receipt: {role}")
        semantic = load_json(semantic_path)
        manifests[role] = manifest
        metadata[role] = semantic
        lineage[role] = {
            "path": str(root.resolve()),
            "manifest_identity_sha256": manifest_identity,
            "manifest_file_sha256": sha256_file(manifest_path),
            "completion_file_sha256": sha256_file(completion_path),
            "semantic_receipt_name": semantic_path.name,
            "semantic_receipt_sha256": sha256_file(semantic_path),
            "member_count": len(members),
        }

    primary_manifest = manifests["primary_label_free_features"]
    posterior_manifest = manifests["posterior_label_free_features"]
    score_manifest = manifests["frozen_candidate_scores"]
    calibration_manifest = manifests["label_free_calibration"]
    alert_manifest = manifests["calibrated_evaluation_alerts"]
    primary_summary = metadata["primary_label_free_features"]
    posterior_summary = metadata["posterior_label_free_features"]
    score_summary = metadata["frozen_candidate_scores"]
    calibration_record = metadata["label_free_calibration"]
    alert_summary = metadata["calibrated_evaluation_alerts"]
    if (
        primary_summary.get("labels_joined") is not False
        or posterior_summary.get("supervision_audit", {}).get("labels_read_or_emitted")
        is not False
        or score_summary.get("labels_read_or_emitted") is not False
        or calibration_record.get("visual_labels_read_or_emitted") is not False
        or alert_summary.get("labels_read_or_emitted") is not False
    ):
        raise RuntimeError("an executed product read or emitted visual labels")
    if (
        primary_manifest.get("trace_identity_sha256_ordered")
        != posterior_manifest.get("trace_identity_sha256_ordered")
        or len(primary_manifest.get("trace_identity_sha256_ordered", [])) != 100
        or primary_manifest.get("analysis_source_sha256")
        != source_hashes["analyze_dit_bad_good_custom_traces.py"]
        or posterior_manifest.get("analysis_source_sha256")
        != source_hashes["analyze_dit_posterior_evidence_metrics.py"]
        or posterior_manifest.get("imported_validation_helper_sha256")
        != source_hashes["analyze_dit_bad_good_custom_traces.py"]
    ):
        raise RuntimeError("executed feature products have inconsistent trace/source lineage")
    score_identity = score_manifest.get("identity_sha256")
    calibration_identity = calibration_record.get("identity_sha256")
    if (
        score_manifest.get("candidate_protocol_identity_sha256")
        != EXPECTED_PROTOCOL_IDENTITY
        or score_summary.get("candidate_protocol_identity_sha256")
        != EXPECTED_PROTOCOL_IDENTITY
        or score_summary.get("primary_manifest_identity_sha256")
        != primary_manifest.get("identity_sha256")
        or score_summary.get("posterior_manifest_identity_sha256")
        != posterior_manifest.get("identity_sha256")
        or calibration_manifest.get("candidate_protocol_identity_sha256")
        != EXPECTED_PROTOCOL_IDENTITY
        or calibration_record.get("candidate_protocol_identity_sha256")
        != EXPECTED_PROTOCOL_IDENTITY
        or calibration_record.get("candidate_score_manifest_identity_sha256")
        != score_identity
        or alert_manifest.get("candidate_protocol_identity_sha256")
        != EXPECTED_PROTOCOL_IDENTITY
        or alert_summary.get("candidate_protocol_identity_sha256")
        != EXPECTED_PROTOCOL_IDENTITY
        or alert_summary.get("candidate_score_manifest_identity_sha256")
        != score_identity
        or alert_manifest.get("calibration_identity_sha256") != calibration_identity
        or alert_summary.get("calibration_identity_sha256") != calibration_identity
    ):
        raise RuntimeError("executed score/calibration/alert lineage is inconsistent")
    member_hashes = {
        role: {item["name"]: item["sha256"] for item in manifest.get("files", [])}
        for role, manifest in manifests.items()
    }
    expected_copies = {
        ("primary_label_free_features", "analysis_source.py"): source_hashes[
            "analyze_dit_bad_good_custom_traces.py"
        ],
        ("posterior_label_free_features", "analysis_source.py"): source_hashes[
            "analyze_dit_posterior_evidence_metrics.py"
        ],
        ("frozen_candidate_scores", "scorer_source.py"): source_hashes[
            "score_dit_bad_good_frozen_candidates.py"
        ],
        ("label_free_calibration", "calibrator_source.py"): source_hashes[
            "calibrate_dit_bad_good_conformal_thresholds.py"
        ],
        ("label_free_calibration", "scorer_helper_source.py"): source_hashes[
            "score_dit_bad_good_frozen_candidates.py"
        ],
        ("calibrated_evaluation_alerts", "applicator_source.py"): source_hashes[
            "apply_dit_bad_good_conformal_thresholds.py"
        ],
        ("calibrated_evaluation_alerts", "calibrator_helper_source.py"): source_hashes[
            "calibrate_dit_bad_good_conformal_thresholds.py"
        ],
        ("calibrated_evaluation_alerts", "scorer_helper_source.py"): source_hashes[
            "score_dit_bad_good_frozen_candidates.py"
        ],
    }
    for (role, name), expected_hash in expected_copies.items():
        if member_hashes.get(role, {}).get(name) != expected_hash:
            raise RuntimeError(f"executed product source copy differs: {role}/{name}")
    lineage["cross_product_audit"] = {
        "primary_and_posterior_trace_identity_lists_equal": True,
        "trace_run_count": 100,
        "all_candidate_lineage_matches_v5": True,
        "score_to_calibration_to_alert_lineage_matches": True,
        "all_embedded_source_copies_match_execution_sources": True,
        "visual_labels_read_or_emitted": False,
    }
    return lineage


def build_record(protocol: dict[str, Any]) -> dict[str, Any]:
    prior_execution_lineage = validate_prior_execution_locks()
    for path in SOURCE_FILES:
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"execution source is missing or indirect: {path}")
    hashes = {path.name: sha256_file(path) for path in SOURCE_FILES}
    if len(hashes) != len(SOURCE_FILES):
        raise RuntimeError("execution source basenames are not unique")
    executed_product_lineage = validate_executed_products(hashes)
    frozen = protocol["source_products"]
    sampler = protocol["sampler_lineage_contract"]["source_snapshot_sha256"]
    if (
        hashes["trace_dit_imagenet256_custom_batch.py"] != sampler["runner_source.py"]
        or hashes["sample_dit_imagenet256_custom.py"] != sampler["custom_baseline_helper.py"]
        or hashes["reproduce_dit_imagenet256.py"] != sampler["strict_reproduction_helper.py"]
        or hashes["analyze_dit_bad_good_custom_traces.py"]
        != frozen["primary_label_free"]["analysis_source_sha256"]
        or hashes["analyze_dit_posterior_evidence_metrics.py"]
        != frozen["posterior_label_free"]["analysis_source_sha256"]
    ):
        raise RuntimeError("execution source hashes differ from the candidate/sampler lock")

    seeds_csv = ",".join(str(seed) for seed in range(30, 130))
    primary_output = FEATURE_ROOT / "custom_label_free_v1"
    posterior_output = FEATURE_ROOT / "posterior_label_free_v1"
    score_output = FEATURE_ROOT / "frozen_candidate_scores_label_free_v1"
    alert_output = FEATURE_ROOT / "calibrated_evaluation_alerts_label_free_v1"
    python = sys.executable
    commands_by_stage = {
        "primary_feature_extraction": [
            python,
            str(ROOT / "experiments/analyze_dit_bad_good_custom_traces.py"),
            "--trace-root",
            str(TRACE_ROOT),
            "--trace-glob",
            "confirmation_v1_seed*",
            "--expected-classes",
            "207,602,795",
            "--expected-seeds",
            seeds_csv,
            "--output-dir",
            str(primary_output),
        ],
        "posterior_feature_extraction": [
            python,
            str(ROOT / "experiments/analyze_dit_posterior_evidence_metrics.py"),
            "--trace-root",
            str(TRACE_ROOT),
            "--trace-glob",
            "confirmation_v1_seed*",
            "--expected-classes",
            "207,602,795",
            "--expected-seeds",
            seeds_csv,
            "--output-dir",
            str(posterior_output),
        ],
        "frozen_candidate_scoring": [
            python,
            str(ROOT / "experiments/score_dit_bad_good_frozen_candidates.py"),
            "--lock-root",
            str(CANDIDATE_LOCK),
            "--primary-root",
            str(primary_output),
            "--posterior-root",
            str(posterior_output),
            "--output",
            str(score_output),
        ],
        "label_free_calibration": [
            python,
            str(ROOT / "experiments/calibrate_dit_bad_good_conformal_thresholds.py"),
            "--candidate-lock",
            str(CANDIDATE_LOCK),
            "--scores-root",
            str(score_output),
            "--output",
            str(ROOT / "experiments/locks/dit_bad_good_conformal_calibration_lock_v1"),
        ],
        "apply_calibration_to_evaluation": [
            python,
            str(ROOT / "experiments/apply_dit_bad_good_conformal_thresholds.py"),
            "--candidate-lock",
            str(CANDIDATE_LOCK),
            "--calibration-lock",
            str(ROOT / "experiments/locks/dit_bad_good_conformal_calibration_lock_v1"),
            "--scores-root",
            str(score_output),
            "--output",
            str(alert_output),
        ],
    }
    stage_order = (
        "primary_feature_extraction",
        "posterior_feature_extraction",
        "frozen_candidate_scoring",
        "label_free_calibration",
        "apply_calibration_to_evaluation",
    )
    commands_in_order = [
        {"stage": stage, "argv": commands_by_stage[stage]} for stage in stage_order
    ]
    record: dict[str, Any] = {
        "schema_version": 1,
        "status": "AUDITED_AFTER_LABEL_FREE_EXECUTION_BEFORE_ANY_VISUAL_LABEL_JOIN",
        "candidate_protocol_identity_sha256": protocol["identity_sha256"],
        "candidate_manifest_file_sha256": sha256_file(CANDIDATE_LOCK / "manifest.json"),
        "candidate_protocol_file_sha256": sha256_file(
            CANDIDATE_LOCK / "candidate_protocol.json"
        ),
        "source_sha256_by_basename": hashes,
        "prior_execution_lineage": prior_execution_lineage,
        "executed_product_lineage": executed_product_lineage,
        "timeline_interpretation": {
            "candidate_v5": "frozen after observation-only sampling but before feature extraction, endpoint review, or candidate scoring",
            "execution_v3": "frozen before feature extraction; commands were addressed by stage name despite dictionary serialization order",
            "execution_v4": "frozen after label-free feature extraction and before frozen candidate scoring; command dependency order represented as a list",
            "execution_v5": "post-execution source-chain audit refreeze before visual label join",
            "execution_v6": "post-execution product-and-source receipt before visual label join; it does not claim prospective timing for itself",
        },
        "import_contracts": {
            "posterior_imports_primary_analysis_helper_sha256": hashes[
                "analyze_dit_bad_good_custom_traces.py"
            ],
            "calibrator_imports_scorer_sha256": hashes[
                "score_dit_bad_good_frozen_candidates.py"
            ],
            "applicator_imports_calibrator_sha256": hashes[
                "calibrate_dit_bad_good_conformal_thresholds.py"
            ],
            "applicator_imports_scorer_sha256": hashes[
                "score_dit_bad_good_frozen_candidates.py"
            ],
        },
        "trace_root": str(TRACE_ROOT),
        "trace_glob": "confirmation_v1_seed*",
        "commands_in_order": commands_in_order,
        "labels_read_or_emitted_by_any_command": False,
    }
    record["identity_sha256"] = canonical_sha256(record)
    return record


def publish() -> Path:
    protocol = validate_candidate_lock()
    record = build_record(protocol)
    if os.path.lexists(OUTPUT):
        raise RuntimeError(f"refusing to overwrite execution-source lock: {OUTPUT}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{OUTPUT.name}.tmp-", dir=OUTPUT.parent))
    try:
        write_json(staging / "execution_sources_locked.json", record)
        sources_dir = staging / "sources"
        sources_dir.mkdir()
        for source in SOURCE_FILES:
            shutil.copy2(source, sources_dir / source.name)
        files = []
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                files.append(
                    {
                        "name": path.relative_to(staging).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "complete",
            "execution_sources_identity_sha256": record["identity_sha256"],
            "files": files,
        }
        manifest["identity_sha256"] = canonical_sha256(manifest)
        write_json(staging / "manifest.json", manifest)
        write_json(
            staging / "completion.json",
            {
                "complete": True,
                "manifest_file_sha256": sha256_file(staging / "manifest.json"),
                "manifest_identity_sha256": manifest["identity_sha256"],
                "execution_sources_file_sha256": sha256_file(
                    staging / "execution_sources_locked.json"
                ),
                "execution_sources_identity_sha256": record["identity_sha256"],
            },
        )
        os.replace(staging, OUTPUT)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return OUTPUT


def main() -> int:
    output = publish()
    record = load_json(output / "execution_sources_locked.json")
    print(
        json.dumps(
            {
                "output": str(output),
                "status": record["status"],
                "identity_sha256": record["identity_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
