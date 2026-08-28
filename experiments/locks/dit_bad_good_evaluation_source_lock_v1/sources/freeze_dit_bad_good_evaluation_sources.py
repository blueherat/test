#!/usr/bin/env python3
"""Freeze the final label-lock and prospective evaluator sources before joining them."""

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
DATA_ROOT = Path(os.environ.get("EQVAE_DATA_ROOT", "/data/users/zhoushunyu/eqvae"))
CANDIDATE_LOCK = ROOT / "experiments/locks/dit_bad_good_candidate_confirmation_lock_v5"
CALIBRATION_LOCK = ROOT / "experiments/locks/dit_bad_good_conformal_calibration_lock_v1"
ALERTS_ROOT = (
    DATA_ROOT
    / "cross_scale_evidence/bad_good_metric_confirmation_v1/"
    "calibrated_evaluation_alerts_label_free_v1"
)
FINAL_CONSENSUS_LOCK = (
    ROOT / "experiments/annotations/dit_fresh_eval240_adjudicated_consensus_lock_v2"
)
RESULT_ROOT = (
    DATA_ROOT
    / "cross_scale_evidence/bad_good_metric_confirmation_v1/"
    "prospective_confirmation_result_v1"
)
OUTPUT = ROOT / "experiments/locks/dit_bad_good_evaluation_source_lock_v1"
CANDIDATE_IDENTITY = "198a82a7c8a0ab79d901c76a5c810f4a40889604a66f18e995d0699f73c12bce"
BLIND_PACK_IDENTITY = "59791e2fe6b319bb312060991efed01e6b1e9d5ad608e8a5b38e77c6f4a241ff"
SOURCE_FILES = (
    ROOT / "experiments/lock_dit_fresh_eval240_consensus.py",
    ROOT / "experiments/lock_dit_fresh_eval240_adjudicated_consensus.py",
    ROOT / "experiments/evaluate_dit_bad_good_prospective_confirmation.py",
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
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def validate_hash_lock(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"hash lock/product is missing or indirect: {root}")
    manifest_path = root / "manifest.json"
    manifest = load_json(manifest_path)
    completion = load_json(root / "completion.json")
    manifest_without_identity = dict(manifest)
    identity = manifest_without_identity.pop("identity_sha256", None)
    if (
        manifest.get("status") != "complete"
        or completion.get("complete") is not True
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
        or completion.get("manifest_identity_sha256") != identity
        or identity != canonical_sha256(manifest_without_identity)
    ):
        raise RuntimeError(f"invalid hash lock/product: {root}")
    members = {item["name"]: item for item in manifest.get("files", [])}
    for name, item in members.items():
        path = root / name
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != item.get("bytes")
            or sha256_file(path) != item.get("sha256")
        ):
            raise RuntimeError(f"hash lock/product member changed: {path}")
    return manifest, completion


def build_record() -> dict[str, Any]:
    candidate_manifest, candidate_completion = validate_hash_lock(CANDIDATE_LOCK)
    calibration_manifest, calibration_completion = validate_hash_lock(CALIBRATION_LOCK)
    alert_manifest, alert_completion = validate_hash_lock(ALERTS_ROOT)
    protocol = load_json(CANDIDATE_LOCK / "candidate_protocol.json")
    calibration = load_json(CALIBRATION_LOCK / "calibration_locked.json")
    alert_summary = load_json(ALERTS_ROOT / "summary.json")
    if (
        protocol.get("identity_sha256") != CANDIDATE_IDENTITY
        or candidate_completion.get("protocol_identity_sha256") != CANDIDATE_IDENTITY
        or calibration.get("candidate_protocol_identity_sha256") != CANDIDATE_IDENTITY
        or calibration.get("visual_labels_read_or_emitted") is not False
        or alert_summary.get("candidate_protocol_identity_sha256") != CANDIDATE_IDENTITY
        or alert_summary.get("labels_read_or_emitted") is not False
        or alert_manifest.get("calibration_identity_sha256")
        != calibration.get("identity_sha256")
    ):
        raise RuntimeError("candidate/calibration/alert lineage is invalid or supervised")
    if os.path.lexists(FINAL_CONSENSUS_LOCK):
        raise RuntimeError("final consensus already exists; evaluation source freeze is too late")
    if os.path.lexists(RESULT_ROOT):
        raise RuntimeError("prospective result already exists; source freeze is too late")
    for path in SOURCE_FILES:
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"evaluation source is missing or indirect: {path}")
    source_hashes = {path.name: sha256_file(path) for path in SOURCE_FILES}
    if len(source_hashes) != len(SOURCE_FILES):
        raise RuntimeError("evaluation source basenames are not unique")
    command = [
        sys.executable,
        str(ROOT / "experiments/evaluate_dit_bad_good_prospective_confirmation.py"),
        "--candidate-lock",
        str(CANDIDATE_LOCK),
        "--calibration-lock",
        str(CALIBRATION_LOCK),
        "--alerts-root",
        str(ALERTS_ROOT),
        "--consensus-lock",
        str(FINAL_CONSENSUS_LOCK),
        "--output",
        str(RESULT_ROOT),
    ]
    record: dict[str, Any] = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_FINAL_VISUAL_LABEL_LOCK_OR_ANY_LABEL_SCORE_JOIN",
        "candidate_protocol_identity_sha256": CANDIDATE_IDENTITY,
        "blind_pack_identity_sha256": BLIND_PACK_IDENTITY,
        "source_sha256_by_basename": source_hashes,
        "input_lineage": {
            "candidate_manifest_identity_sha256": candidate_manifest["identity_sha256"],
            "candidate_manifest_file_sha256": sha256_file(CANDIDATE_LOCK / "manifest.json"),
            "calibration_manifest_identity_sha256": calibration_manifest["identity_sha256"],
            "calibration_manifest_file_sha256": sha256_file(
                CALIBRATION_LOCK / "manifest.json"
            ),
            "calibration_identity_sha256": calibration["identity_sha256"],
            "alerts_manifest_identity_sha256": alert_manifest["identity_sha256"],
            "alerts_manifest_file_sha256": sha256_file(ALERTS_ROOT / "manifest.json"),
            "candidate_score_manifest_identity_sha256": alert_summary[
                "candidate_score_manifest_identity_sha256"
            ],
            "calibration_completion_file_sha256": sha256_file(
                CALIBRATION_LOCK / "completion.json"
            ),
            "alerts_completion_file_sha256": sha256_file(ALERTS_ROOT / "completion.json"),
            "candidate_completion_file_sha256": sha256_file(
                CANDIDATE_LOCK / "completion.json"
            ),
            "unused_receipt_fields_are_intentionally_not_score_values": bool(
                calibration_completion and alert_completion
            ),
        },
        "planned_final_consensus_lock": str(FINAL_CONSENSUS_LOCK),
        "planned_result_root": str(RESULT_ROOT),
        "evaluation_command": command,
        "individual_scores_thresholds_alerts_or_labels_read_by_this_freezer": False,
    }
    record["identity_sha256"] = canonical_sha256(record)
    return record


def publish() -> Path:
    record = build_record()
    if os.path.lexists(OUTPUT):
        raise RuntimeError(f"refusing to overwrite evaluation source lock: {OUTPUT}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{OUTPUT.name}.tmp-", dir=OUTPUT.parent))
    try:
        write_json(staging / "evaluation_sources_locked.json", record)
        sources = staging / "sources"
        sources.mkdir()
        for path in SOURCE_FILES:
            shutil.copy2(path, sources / path.name)
        members = [
            {
                "name": path.relative_to(staging).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        ]
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "complete",
            "evaluation_sources_identity_sha256": record["identity_sha256"],
            "files": members,
        }
        manifest["identity_sha256"] = canonical_sha256(manifest)
        write_json(staging / "manifest.json", manifest)
        write_json(
            staging / "completion.json",
            {
                "complete": True,
                "manifest_file_sha256": sha256_file(staging / "manifest.json"),
                "manifest_identity_sha256": manifest["identity_sha256"],
                "evaluation_sources_file_sha256": sha256_file(
                    staging / "evaluation_sources_locked.json"
                ),
                "evaluation_sources_identity_sha256": record["identity_sha256"],
            },
        )
        os.replace(staging, OUTPUT)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return OUTPUT


def main() -> int:
    output = publish()
    record = load_json(output / "evaluation_sources_locked.json")
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
