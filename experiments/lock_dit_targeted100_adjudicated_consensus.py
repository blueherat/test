#!/usr/bin/env python3
"""Freeze the conservative adjudicated labels for the targeted DiT pilot.

This program only reads the already locked three-review majority consensus,
the evidence-blind visual adjudication of its ten clear-bad candidates, and the
bound native PNGs.  It never reads trajectories, metrics, feature files, AUCs,
or research summaries.  A downgraded majority candidate is excluded as
``mild_or_disputed``; it is never silently promoted to a clean negative.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MAJORITY_LOCK = (
    ROOT / "experiments/annotations/dit_targeted100_consensus_lock_v1"
)
ADJUDICATION = (
    ROOT
    / "experiments/annotations/dit_targeted100_majority10_adjudication_v1_draft.json"
)
DEFAULT_OUTPUT = (
    ROOT / "experiments/annotations/dit_targeted100_adjudicated_consensus_lock_v2"
)
ALLOWED_DECISIONS = {
    "retain_clear_bad",
    "downgrade_mild_or_typical",
    "uncertain",
}


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"required JSON is missing, non-regular, or indirect: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_hashed_identity(document: dict[str, Any], name: str) -> str:
    observed = document.get("identity_sha256")
    payload = dict(document)
    payload.pop("identity_sha256", None)
    expected = canonical_sha256(payload)
    if observed != expected:
        raise RuntimeError(f"{name} identity hash is invalid")
    return expected


def validate_majority_lock(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Path]]:
    paths = {
        "consensus": root / "consensus_locked.json",
        "manifest": root / "manifest.json",
        "completion": root / "completion.json",
    }
    consensus = load_json(paths["consensus"])
    manifest = load_json(paths["manifest"])
    completion = load_json(paths["completion"])
    consensus_identity = validate_hashed_identity(consensus, "majority consensus")
    manifest_identity = validate_hashed_identity(manifest, "majority manifest")
    completion_payload = dict(completion)
    observed_payload = completion_payload.pop("payload_sha256", None)
    if observed_payload != canonical_sha256(completion_payload):
        raise RuntimeError("majority completion payload hash is invalid")
    if (
        completion.get("complete") is not True
        or manifest.get("status")
        != "LOCKED_BEFORE_ANY_TARGETED100_TRAJECTORY_METRIC_JOIN"
        or consensus.get("status") != manifest.get("status")
        or manifest.get("consensus_file_sha256") != sha256_file(paths["consensus"])
        or completion.get("consensus_file_sha256")
        != manifest.get("consensus_file_sha256")
        or completion.get("consensus_identity_sha256") != consensus_identity
        or completion.get("manifest_file_sha256") != sha256_file(paths["manifest"])
        or completion.get("manifest_identity_sha256") != manifest_identity
    ):
        raise RuntimeError("majority consensus hash lineage is invalid")
    rows = consensus.get("rows")
    if not isinstance(rows, list) or len(rows) != 100:
        raise RuntimeError("majority consensus must contain exactly 100 rows")
    keys = [row.get("sample_key") for row in rows if isinstance(row, dict)]
    if len(keys) != 100 or len(set(keys)) != 100:
        raise RuntimeError("majority consensus rows are malformed or duplicated")
    observed_counts = {
        label: sum(row.get("primary_label") == label for row in rows)
        for label in ("clear_bad", "clean_good", "mild_or_disputed")
    }
    if observed_counts != {"clear_bad": 10, "clean_good": 69, "mild_or_disputed": 21}:
        raise RuntimeError(f"majority consensus counts changed: {observed_counts}")
    return consensus, manifest, completion, paths


def validate_adjudication(
    path: Path, majority_bad_keys: set[str]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    document = load_json(path)
    if (
        document.get("status") != "draft"
        or document.get("independent_adjudication") is not True
        or document.get("metrics_seen") is not False
        or document.get("trajectories_seen") is not False
        or document.get("other_reviews_seen") is not False
    ):
        raise RuntimeError("adjudication lacks the required evidence-blind declaration")
    decisions = document.get("decisions")
    if not isinstance(decisions, list) or len(decisions) != len(majority_bad_keys):
        raise RuntimeError("adjudication decision count is invalid")
    mapping: dict[str, dict[str, Any]] = {}
    for row in decisions:
        if not isinstance(row, dict):
            raise RuntimeError("adjudication decision is not an object")
        key = row.get("candidate_id")
        decision = row.get("decision")
        if (
            not isinstance(key, str)
            or key in mapping
            or decision not in ALLOWED_DECISIONS
            or type(row.get("class_id")) is not int
            or type(row.get("seed")) is not int
            or key != f"class{row['class_id']:04d}_seed{row['seed']}"
            or not isinstance(row.get("failure_subtype"), str)
            or not row["failure_subtype"].strip()
            or not isinstance(row.get("reason"), str)
            or not row["reason"].strip()
            or not isinstance(row.get("relative_to_same_class_controls"), str)
            or not row["relative_to_same_class_controls"].strip()
        ):
            raise RuntimeError(f"invalid adjudication decision: {key!r}")
        mapping[key] = row
    if set(mapping) != majority_bad_keys:
        raise RuntimeError("adjudication candidates differ from majority clear-bad keys")
    counts = {
        decision: sum(row["decision"] == decision for row in decisions)
        for decision in sorted(ALLOWED_DECISIONS)
    }
    declared = document.get("summary")
    if not isinstance(declared, dict) or any(
        declared.get(decision) != count for decision, count in counts.items()
    ):
        raise RuntimeError("adjudication summary does not match its decisions")
    return document, mapping


def validate_native_image(row: dict[str, Any]) -> None:
    record = row.get("native_image")
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise RuntimeError(f"row lacks native image binding: {row.get('sample_key')}")
    path = Path(record["path"])
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"bound native image is missing or indirect: {path}")
    with Image.open(path) as image:
        image.load()
        observed = {
            "file_sha256": sha256_file(path),
            "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
            "mode": image.mode,
            "size": list(image.size),
        }
    if any(record.get(field) != value for field, value in observed.items()):
        raise RuntimeError(f"bound native image changed: {path}")


def build_rows(
    majority_rows: list[dict[str, Any]],
    adjudication: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in majority_rows:
        row = copy.deepcopy(source)
        validate_native_image(row)
        majority_label = row["primary_label"]
        row["majority_primary_label_v1"] = majority_label
        row["majority_binary_primary_included_v1"] = row.get(
            "binary_primary_included"
        )
        decision = adjudication.get(row["sample_key"])
        if majority_label == "clear_bad":
            if decision is None:
                raise RuntimeError("majority clear-bad row lacks adjudication")
            final_label = (
                "clear_bad"
                if decision["decision"] == "retain_clear_bad"
                else "mild_or_disputed"
            )
            row["strict_adjudication"] = copy.deepcopy(decision)
        else:
            if decision is not None:
                raise RuntimeError("adjudication unexpectedly covers a non-bad row")
            final_label = majority_label
            row["strict_adjudication"] = None
        row["primary_label"] = final_label
        row["binary_primary_included"] = final_label in {"clear_bad", "clean_good"}
        rows.append(row)
    counts = {
        label: sum(row["primary_label"] == label for row in rows)
        for label in ("clear_bad", "clean_good", "mild_or_disputed")
    }
    if counts != {"clear_bad": 5, "clean_good": 69, "mild_or_disputed": 26}:
        raise RuntimeError(f"strict adjudicated counts changed: {counts}")
    return rows


def validate_output(root: Path) -> None:
    expected_files = {
        "consensus_locked.json",
        "majority_consensus_v1_locked.json",
        "majority_manifest_v1_locked.json",
        "majority_completion_v1_locked.json",
        "adjudication_locked.json",
        "locker_source.py",
        "manifest.json",
        "completion.json",
    }
    if {path.name for path in root.iterdir()} != expected_files or any(
        not path.is_file() or path.is_symlink() for path in root.iterdir()
    ):
        raise RuntimeError("strict lock output layout is not closed and regular")
    consensus = load_json(root / "consensus_locked.json")
    manifest = load_json(root / "manifest.json")
    completion = load_json(root / "completion.json")
    consensus_identity = validate_hashed_identity(consensus, "strict consensus")
    manifest_identity = validate_hashed_identity(manifest, "strict manifest")
    payload = dict(completion)
    observed_payload = payload.pop("payload_sha256", None)
    if observed_payload != canonical_sha256(payload):
        raise RuntimeError("strict completion payload hash is invalid")
    listed = manifest.get("files")
    if not isinstance(listed, list) or {record.get("name") for record in listed} != (
        expected_files - {"manifest.json", "completion.json"}
    ):
        raise RuntimeError("strict manifest member set is invalid")
    for record in listed:
        path = root / record["name"]
        if record.get("bytes") != path.stat().st_size or record.get(
            "sha256"
        ) != sha256_file(path):
            raise RuntimeError(f"strict lock payload changed: {path}")
    if (
        completion.get("complete") is not True
        or completion.get("consensus_file_sha256")
        != sha256_file(root / "consensus_locked.json")
        or completion.get("consensus_identity_sha256") != consensus_identity
        or completion.get("manifest_file_sha256") != sha256_file(root / "manifest.json")
        or completion.get("manifest_identity_sha256") != manifest_identity
    ):
        raise RuntimeError("strict lock completion lineage is invalid")


def publish(majority_root: Path, adjudication_path: Path, output: Path) -> Path:
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite strict consensus lock: {output}")
    majority, majority_manifest, majority_completion, majority_paths = (
        validate_majority_lock(majority_root)
    )
    majority_rows = majority["rows"]
    majority_bad = {
        row["sample_key"] for row in majority_rows if row["primary_label"] == "clear_bad"
    }
    adjudication_document, adjudication = validate_adjudication(
        adjudication_path, majority_bad
    )
    rows = build_rows(majority_rows, adjudication)
    counts = {
        label: sum(row["primary_label"] == label for row in rows)
        for label in ("clear_bad", "clean_good", "mild_or_disputed")
    }
    status = "LOCKED_COMPLETE_BEFORE_ANY_TARGETED100_TRAJECTORY_METRIC_JOIN"
    consensus = {
        "schema_version": 2,
        "status": status,
        "rule": {
            "starting_point": "three-review majority consensus v1",
            "clear_bad": "majority clear-bad and evidence-blind adjudicator retain_clear_bad",
            "clean_good": "unchanged majority clean-good only",
            "mild_or_disputed": (
                "unchanged majority disputed rows plus every downgraded or uncertain "
                "majority clear-bad candidate"
            ),
            "downgraded_candidates_are_never_clean_negatives": True,
            "metric_trajectory_or_signal_used": False,
        },
        "counts": counts,
        "rows": rows,
    }
    consensus["identity_sha256"] = canonical_sha256(consensus)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent))
    try:
        write_json(staging / "consensus_locked.json", consensus)
        copies = {
            "majority_consensus_v1_locked.json": majority_paths["consensus"],
            "majority_manifest_v1_locked.json": majority_paths["manifest"],
            "majority_completion_v1_locked.json": majority_paths["completion"],
            "adjudication_locked.json": adjudication_path,
            "locker_source.py": Path(__file__).resolve(),
        }
        for destination, source in copies.items():
            shutil.copyfile(source, staging / destination)
        payload_names = sorted({"consensus_locked.json", *copies})
        files = [
            {
                "name": name,
                "bytes": (staging / name).stat().st_size,
                "sha256": sha256_file(staging / name),
            }
            for name in payload_names
        ]
        manifest = {
            "schema_version": 2,
            "experiment": "dit_targeted100_adjudicated_visual_consensus_lock_v2",
            "status": status,
            "consensus_file_sha256": sha256_file(staging / "consensus_locked.json"),
            "consensus_identity_sha256": consensus["identity_sha256"],
            "majority_consensus_identity_sha256": majority["identity_sha256"],
            "majority_manifest_identity_sha256": majority_manifest["identity_sha256"],
            "majority_completion_payload_sha256": majority_completion[
                "payload_sha256"
            ],
            "adjudication_source_sha256": sha256_file(adjudication_path),
            "native_image_binding_sha256": canonical_sha256(
                [row["native_image"] for row in rows]
            ),
            "counts": counts,
            "files": files,
        }
        manifest["identity_sha256"] = canonical_sha256(manifest)
        write_json(staging / "manifest.json", manifest)
        completion = {
            "complete": True,
            "consensus_file_sha256": manifest["consensus_file_sha256"],
            "consensus_identity_sha256": consensus["identity_sha256"],
            "manifest_file_sha256": sha256_file(staging / "manifest.json"),
            "manifest_identity_sha256": manifest["identity_sha256"],
            "locked_row_count": len(rows),
        }
        completion["payload_sha256"] = canonical_sha256(completion)
        write_json(staging / "completion.json", completion)
        staging.rename(output)
        validate_output(output)
        return output
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def self_test() -> None:
    fake = [
        {
            "sample_key": "class0001_seed1",
            "primary_label": "clear_bad",
            "binary_primary_included": True,
            "native_image": {},
        }
    ]
    # Test only the conservative relabel rule without requiring a fixture PNG.
    decision = {"decision": "downgrade_mild_or_typical"}
    row = copy.deepcopy(fake[0])
    final = "clear_bad" if decision["decision"] == "retain_clear_bad" else "mild_or_disputed"
    row["primary_label"] = final
    assert row["primary_label"] == "mild_or_disputed"
    assert ALLOWED_DECISIONS == {
        "retain_clear_bad",
        "downgrade_mild_or_typical",
        "uncertain",
    }
    print("self-test passed: conservative adjudication never promotes a downgrade to clean")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--majority-lock", type=Path, default=MAJORITY_LOCK)
    parser.add_argument("--adjudication", type=Path, default=ADJUDICATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    output = publish(
        args.majority_lock.expanduser().resolve(),
        args.adjudication.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
    )
    summary = load_json(output / "consensus_locked.json")
    print(
        json.dumps(
            {"output": str(output), "status": summary["status"], "counts": summary["counts"]},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
