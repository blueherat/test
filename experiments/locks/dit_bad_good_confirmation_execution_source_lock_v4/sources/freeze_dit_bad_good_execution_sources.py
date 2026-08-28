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
OUTPUT = ROOT / "experiments/locks/dit_bad_good_confirmation_execution_source_lock_v4"
DATA_ROOT = Path(os.environ.get("EQVAE_DATA_ROOT", "/data/users/zhoushunyu/eqvae"))
TRACE_ROOT = (
    DATA_ROOT / "cross_scale_evidence/dit_bad_good_confirmation_v1_custom_traces_cfg_locked"
)
FEATURE_ROOT = DATA_ROOT / "cross_scale_evidence/bad_good_metric_confirmation_v1"
EXPECTED_PROTOCOL_IDENTITY = "198a82a7c8a0ab79d901c76a5c810f4a40889604a66f18e995d0699f73c12bce"

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


def build_record(protocol: dict[str, Any]) -> dict[str, Any]:
    for path in SOURCE_FILES:
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"execution source is missing or indirect: {path}")
    hashes = {path.name: sha256_file(path) for path in SOURCE_FILES}
    if len(hashes) != len(SOURCE_FILES):
        raise RuntimeError("execution source basenames are not unique")
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
        "status": "FROZEN_BEFORE_ANY_FRESH_SCORE_EXTRACTION_OR_VISUAL_LABEL_JOIN",
        "candidate_protocol_identity_sha256": protocol["identity_sha256"],
        "candidate_manifest_file_sha256": sha256_file(CANDIDATE_LOCK / "manifest.json"),
        "candidate_protocol_file_sha256": sha256_file(
            CANDIDATE_LOCK / "candidate_protocol.json"
        ),
        "source_sha256_by_basename": hashes,
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
