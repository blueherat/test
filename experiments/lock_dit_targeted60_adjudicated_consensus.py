#!/usr/bin/env python3
"""Apply visual-only conservative adjudication to the locked targeted60 consensus."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "experiments/annotations/dit_targeted60_consensus_lock_v1"
ADJUDICATION = (
    ROOT / "experiments/annotations/dit_targeted60_majority2_adjudication_v1_draft.json"
)
OUTPUT = ROOT / "experiments/annotations/dit_targeted60_adjudicated_consensus_lock_v2"
EXPECTED_COUNTS = {"clear_bad": 2, "clean_good": 49, "mild_or_disputed": 9}


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(raw).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_raw_lock() -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = RAW_ROOT / "manifest.json"
    consensus_path = RAW_ROOT / "consensus_locked.json"
    manifest = load_json(manifest_path)
    completion = load_json(RAW_ROOT / "completion.json")
    consensus = load_json(consensus_path)
    if (
        manifest.get("status") != "LOCKED_BEFORE_ANY_TARGETED60_TRAJECTORY_METRIC_JOIN"
        or completion.get("complete") is not True
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
        or completion.get("consensus_file_sha256") != sha256_file(consensus_path)
        or completion.get("manifest_identity_sha256") != manifest.get("identity_sha256")
        or completion.get("consensus_identity_sha256") != consensus.get("identity_sha256")
        or consensus.get("counts") != EXPECTED_COUNTS
    ):
        raise RuntimeError("raw targeted60 consensus lock is invalid")
    return consensus, manifest


def publish() -> Path:
    raw, raw_manifest = validate_raw_lock()
    adjudication = load_json(ADJUDICATION)
    rows = adjudication.get("rows")
    if (
        adjudication.get("status")
        != "DRAFT_VISUAL_ONLY_ADJUDICATION_BEFORE_ANY_TARGETED60_METRIC_JOIN"
        or adjudication.get("metrics_seen_during_adjudication") is not False
        or adjudication.get("trajectories_seen_during_adjudication") is not False
        or adjudication.get("candidate_signal_values_seen_during_adjudication") is not False
        or not isinstance(rows, list)
        or {row.get("sample_key") for row in rows}
        != {"class0444_seed19", "class0981_seed26"}
        or any(row.get("decision") not in {"retain_clear_bad", "downgrade_to_mild"} for row in rows)
    ):
        raise RuntimeError("visual adjudication draft is invalid")
    raw_bad = {row["sample_key"] for row in raw["rows"] if row["primary_label"] == "clear_bad"}
    if raw_bad != {row["sample_key"] for row in rows}:
        raise RuntimeError("adjudication does not cover exactly the raw majority bad rows")
    decision = {row["sample_key"]: row for row in rows}
    final_rows = []
    for raw_row in raw["rows"]:
        row = dict(raw_row)
        row["raw_primary_label"] = row["primary_label"]
        if row["sample_key"] in decision:
            adjudicated = decision[row["sample_key"]]
            row["adjudication"] = adjudicated
            row["primary_label"] = (
                "clear_bad"
                if adjudicated["decision"] == "retain_clear_bad"
                else "mild_or_disputed"
            )
        else:
            row["adjudication"] = None
        row["binary_primary_included"] = row["primary_label"] in {"clear_bad", "clean_good"}
        final_rows.append(row)
    counts = {label: sum(row["primary_label"] == label for row in final_rows) for label in EXPECTED_COUNTS}
    if counts != EXPECTED_COUNTS:
        raise RuntimeError(f"adjudicated counts changed unexpectedly: {counts}")
    final: dict[str, Any] = {
        "schema_version": 2,
        "status": "LOCKED_ADJUDICATED_BEFORE_ANY_TARGETED60_TRAJECTORY_METRIC_JOIN",
        "raw_consensus_identity_sha256": raw["identity_sha256"],
        "adjudication_source_sha256": sha256_file(ADJUDICATION),
        "counts": counts,
        "rows": final_rows,
    }
    final["identity_sha256"] = canonical_sha256(final)
    if os.path.lexists(OUTPUT):
        raise RuntimeError(f"refusing to overwrite adjudicated lock: {OUTPUT}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{OUTPUT.name}.tmp-", dir=OUTPUT.parent))
    try:
        write_json(staging / "consensus_locked.json", final)
        shutil.copy2(ADJUDICATION, staging / "adjudication_locked.json")
        shutil.copy2(Path(__file__).resolve(), staging / "locker_source.py")
        for name in (
            "consensus_locked.json",
            "completion.json",
            "manifest.json",
            "review_D_locked.json",
            "review_E_locked.json",
            "review_F_locked.json",
            "locker_source.py",
        ):
            source = RAW_ROOT / name
            if source.is_file():
                shutil.copy2(source, staging / f"raw_{name}")
        manifest: dict[str, Any] = {
            "schema_version": 2,
            "status": final["status"],
            "final_consensus_file_sha256": sha256_file(staging / "consensus_locked.json"),
            "final_consensus_identity_sha256": final["identity_sha256"],
            "raw_manifest_identity_sha256": raw_manifest["identity_sha256"],
            "raw_manifest_file_sha256": sha256_file(RAW_ROOT / "manifest.json"),
            "adjudication_file_sha256": sha256_file(staging / "adjudication_locked.json"),
            "locker_source_sha256": sha256_file(staging / "locker_source.py"),
            "counts": counts,
        }
        manifest["identity_sha256"] = canonical_sha256(manifest)
        write_json(staging / "manifest.json", manifest)
        completion = {
            "complete": True,
            "consensus_file_sha256": manifest["final_consensus_file_sha256"],
            "consensus_identity_sha256": final["identity_sha256"],
            "manifest_file_sha256": sha256_file(staging / "manifest.json"),
            "manifest_identity_sha256": manifest["identity_sha256"],
            "locked_row_count": 60,
        }
        completion["payload_sha256"] = canonical_sha256(completion)
        write_json(staging / "completion.json", completion)
        os.replace(staging, OUTPUT)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return OUTPUT


def main() -> int:
    output = publish()
    final = load_json(output / "consensus_locked.json")
    print(json.dumps({"output": str(output), "counts": final["counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
