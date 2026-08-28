#!/usr/bin/env python3
"""Conservatively adjudicate majority clear-bad rows without score access."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

try:
    from .lock_dit_fresh_eval240_consensus import (
        BLIND_PACK_IDENTITY,
        CANDIDATE_PROTOCOL_IDENTITY,
        CLASSES,
        SEEDS,
        canonical_sha256,
        expected_keys,
        load_json,
        sha256_file,
        write_json,
    )
except ImportError:  # pragma: no cover - direct CLI execution.
    from lock_dit_fresh_eval240_consensus import (
        BLIND_PACK_IDENTITY,
        CANDIDATE_PROTOCOL_IDENTITY,
        CLASSES,
        SEEDS,
        canonical_sha256,
        expected_keys,
        load_json,
        sha256_file,
        write_json,
    )


ROOT = Path(__file__).resolve().parents[1]
RAW_LOCK = ROOT / "experiments/annotations/dit_fresh_eval240_consensus_lock_v1"
ADJUDICATION_DRAFT = (
    ROOT / "experiments/annotations/dit_fresh_eval240_adjudication_v1_draft.json"
)
DEFAULT_OUTPUT = (
    ROOT / "experiments/annotations/dit_fresh_eval240_adjudicated_consensus_lock_v2"
)
CONSENSUS_HELPER = ROOT / "experiments/lock_dit_fresh_eval240_consensus.py"
ALLOWED_DECISIONS = {"retain_clear_bad", "downgrade_to_mild"}


def validate_raw_lock(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"raw consensus lock is missing or indirect: {root}")
    manifest_path = root / "manifest.json"
    consensus_path = root / "consensus_locked.json"
    manifest = load_json(manifest_path)
    completion = load_json(root / "completion.json")
    consensus = load_json(consensus_path)
    manifest_without_identity = dict(manifest)
    manifest_identity = manifest_without_identity.pop("identity_sha256", None)
    consensus_without_identity = dict(consensus)
    consensus_identity = consensus_without_identity.pop("identity_sha256", None)
    if (
        manifest.get("status") != "complete"
        or completion.get("complete") is not True
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
        or completion.get("manifest_identity_sha256") != manifest_identity
        or manifest_identity != canonical_sha256(manifest_without_identity)
        or completion.get("consensus_file_sha256") != sha256_file(consensus_path)
        or completion.get("consensus_identity_sha256") != consensus_identity
        or manifest.get("consensus_identity_sha256") != consensus_identity
        or consensus_identity != canonical_sha256(consensus_without_identity)
        or consensus.get("status")
        != "LOCKED_WITHOUT_SCORE_OR_ALERT_ACCESS_BEFORE_ANY_LABEL_SCORE_JOIN"
        or consensus.get("blind_pack_identity_sha256") != BLIND_PACK_IDENTITY
        or consensus.get("candidate_protocol_identity_sha256")
        != CANDIDATE_PROTOCOL_IDENTITY
        or completion.get("locked_row_count") != 240
    ):
        raise RuntimeError("raw consensus lock is invalid")
    members = {item["name"]: item for item in manifest.get("files", [])}
    for name, item in members.items():
        path = root / name
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != item.get("bytes")
            or sha256_file(path) != item.get("sha256")
        ):
            raise RuntimeError(f"raw consensus member changed: {path}")
    rows = consensus.get("rows")
    if not isinstance(rows, list) or len(rows) != 240:
        raise RuntimeError("raw consensus does not contain 240 rows")
    keys = [row.get("sample_key") for row in rows]
    if set(keys) != expected_keys() or len(keys) != len(set(keys)):
        raise RuntimeError("raw consensus rows are not the exact Cartesian product")
    audit = consensus.get("blinding_audit", {})
    if (
        audit.get("reviewer_count") != 3
        or audit.get("endpoint_only_review") is not True
        or audit.get("metric_values_visible_to_reviewers") is not False
        or audit.get("alert_decisions_visible_to_reviewers") is not False
        or audit.get("trajectories_visible_to_reviewers") is not False
        or audit.get("labels_locked_before_score_join") is not True
    ):
        raise RuntimeError("raw consensus lacks the frozen blinding audit")
    return manifest, consensus


def validate_adjudication(path: Path, raw_bad_keys: set[str]) -> dict[str, dict[str, str]]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"adjudication draft is missing or indirect: {path}")
    document = load_json(path)
    if (
        document.get("visual_only_adjudication") is not True
        or document.get("metrics_seen") is not False
        or document.get("candidate_scores_seen") is not False
        or document.get("calibration_thresholds_seen") is not False
        or document.get("alert_decisions_seen") is not False
        or document.get("trajectories_seen") is not False
        or document.get("other_samples_promoted") is not False
        or document.get("blind_pack_identity_sha256") != BLIND_PACK_IDENTITY
        or document.get("adjudication_scope") != "raw_majority_clear_bad_only"
    ):
        raise RuntimeError("adjudication blinding declaration is invalid")
    decisions = document.get("decisions")
    if not isinstance(decisions, dict) or set(decisions) != raw_bad_keys:
        raise RuntimeError("adjudication must cover exactly the raw clear-bad keys")
    for key, row in decisions.items():
        if (
            not isinstance(row, dict)
            or row.get("decision") not in ALLOWED_DECISIONS
            or not isinstance(row.get("reason"), str)
            or not row["reason"].strip()
        ):
            raise RuntimeError(f"invalid adjudication decision: {key}")
    return decisions


def build_final(
    raw: dict[str, Any], decisions: dict[str, dict[str, str]]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for raw_row in raw["rows"]:
        row = dict(raw_row)
        raw_label = row["primary_label"]
        row["raw_primary_label"] = raw_label
        if raw_label == "clear_bad":
            decision = decisions[row["sample_key"]]
            row["adjudication"] = decision
            row["primary_label"] = (
                "clear_bad"
                if decision["decision"] == "retain_clear_bad"
                else "mild_or_disputed"
            )
        else:
            row["adjudication"] = None
        row["binary_primary_included"] = row["primary_label"] in {
            "clear_bad",
            "clean_good",
        }
        rows.append(row)
    counts = {
        label: sum(row["primary_label"] == label for row in rows)
        for label in ("clear_bad", "clean_good", "mild_or_disputed")
    }
    raw_bad_count = sum(row["raw_primary_label"] == "clear_bad" for row in rows)
    retained = sum(row["primary_label"] == "clear_bad" for row in rows)
    final: dict[str, Any] = {
        "schema_version": 1,
        "status": "FINAL_VISUAL_LABELS_LOCKED_BEFORE_ANY_LABEL_SCORE_JOIN",
        "blind_pack_identity_sha256": BLIND_PACK_IDENTITY,
        "candidate_protocol_identity_sha256": CANDIDATE_PROTOCOL_IDENTITY,
        "raw_consensus_identity_sha256": raw["identity_sha256"],
        "blinding_audit": {
            **raw["blinding_audit"],
            "adjudicator_saw_metric_values": False,
            "adjudicator_saw_alert_decisions": False,
            "adjudicator_saw_trajectories": False,
            "adjudication_could_only_retain_or_downgrade_raw_clear_bad": True,
            "labels_locked_before_score_join": True,
        },
        "adjudication_rule": {
            "scope": "raw majority clear-bad rows only",
            "retain": (
                "conspicuous severe blur, fusion/duplication, gross topology, or major "
                "limb/object attachment failure clearly below the same-model class band"
            ),
            "downgrade": (
                "plausibly explained by crop, motion, pose, occlusion, distance, or ordinary "
                "model texture; mild or ambiguous defects are excluded from binary evaluation"
            ),
            "promotion_allowed": False,
        },
        "raw_clear_bad_count": raw_bad_count,
        "retained_clear_bad_count": retained,
        "counts": counts,
        "rows": rows,
    }
    final["identity_sha256"] = canonical_sha256(final)
    return final


def publish(raw_lock: Path, adjudication: Path, output: Path) -> Path:
    raw_manifest, raw = validate_raw_lock(raw_lock)
    raw_bad_keys = {
        row["sample_key"] for row in raw["rows"] if row["primary_label"] == "clear_bad"
    }
    decisions = validate_adjudication(adjudication, raw_bad_keys)
    final = build_final(raw, decisions)
    if os.path.lexists(output):
        raise RuntimeError(f"refusing to overwrite final consensus lock: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        write_json(staging / "consensus_locked.json", final)
        shutil.copy2(adjudication, staging / "adjudication_locked.json")
        shutil.copy2(Path(__file__).resolve(), staging / "adjudicator_locker_source.py")
        shutil.copy2(CONSENSUS_HELPER, staging / "consensus_helper_source.py")
        members = [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(staging.iterdir())
        ]
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "complete",
            "consensus_identity_sha256": final["identity_sha256"],
            "raw_consensus_identity_sha256": raw["identity_sha256"],
            "candidate_protocol_identity_sha256": CANDIDATE_PROTOCOL_IDENTITY,
            "raw_consensus_manifest_identity_sha256": raw_manifest["identity_sha256"],
            "raw_consensus_manifest_file_sha256": sha256_file(raw_lock / "manifest.json"),
            "adjudication_file_sha256": sha256_file(adjudication),
            "counts": final["counts"],
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
                "consensus_file_sha256": sha256_file(staging / "consensus_locked.json"),
                "consensus_identity_sha256": final["identity_sha256"],
                "locked_row_count": 240,
            },
        )
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def self_test() -> None:
    assert len(expected_keys()) == len(CLASSES) * len(SEEDS) == 240
    raw = {
        "identity_sha256": "raw",
        "blinding_audit": {},
        "rows": [
            {
                "sample_key": "a",
                "primary_label": "clear_bad",
                "binary_primary_included": True,
            },
            {
                "sample_key": "b",
                "primary_label": "clean_good",
                "binary_primary_included": True,
            },
        ],
    }
    final = build_final(
        raw, {"a": {"decision": "downgrade_to_mild", "reason": "ambiguous"}}
    )
    assert final["counts"] == {
        "clear_bad": 0,
        "clean_good": 1,
        "mild_or_disputed": 1,
    }
    print("self-test passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-lock", type=Path, default=RAW_LOCK)
    parser.add_argument("--adjudication", type=Path, default=ADJUDICATION_DRAFT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    output = publish(
        args.raw_lock.expanduser().resolve(),
        args.adjudication.expanduser().resolve(),
        args.output.expanduser().absolute(),
    )
    final = load_json(output / "consensus_locked.json")
    print(json.dumps({"output": str(output), "counts": final["counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
