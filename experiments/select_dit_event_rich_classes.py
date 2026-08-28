#!/usr/bin/env python3
"""Select endpoint-risk classes without reading any trajectory score.

The program has two deliberately separated stages.  ``rank`` consumes only
blind endpoint consensus labels from the broad mini-screen.  ``anchor`` then
checks the selected class sets on a second, independent endpoint-only seed
cohort and emits a prospective trace-sampling plan.  Neither command accepts a
feature table, trajectory archive, embedding, or candidate score as input.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = ROOT / "experiments/locks/dit_event_rich_confirmation_protocol_lock_v3"


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def without_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(value)
    out.pop("identity_sha256", None)
    return out


def write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def load_protocol(lock_root: Path) -> dict[str, Any]:
    lock_root = lock_root.expanduser().absolute()
    if not lock_root.is_dir() or lock_root.is_symlink():
        raise RuntimeError(f"protocol lock must be a real directory: {lock_root}")
    protocol_path = lock_root / "protocol.json"
    manifest_path = lock_root / "manifest.json"
    completion_path = lock_root / "completion.json"
    protocol = load_json(protocol_path)
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    identity = protocol.get("identity_sha256")
    if canonical_sha256(without_identity(protocol)) != identity:
        raise RuntimeError("protocol identity mismatch")
    if (
        canonical_sha256(without_identity(manifest)) != manifest.get("identity_sha256")
        or manifest.get("status") != "complete"
        or manifest.get("protocol_identity_sha256") != identity
        or completion.get("complete") is not True
        or completion.get("protocol_identity_sha256") != identity
        or completion.get("protocol_file_sha256") != sha256_file(protocol_path)
        or completion.get("manifest_identity_sha256") != manifest.get("identity_sha256")
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
    ):
        raise RuntimeError("protocol lock manifest/completion mismatch")
    expected_names = {
        "protocol.json",
        "instructional_anchor_catalog.json",
        "label_audit/AUDIT_REPORT.md",
        "label_audit/audit_summary.json",
        "sources/freeze_dit_event_rich_confirmation_protocol.py",
        "sources/select_dit_event_rich_classes.py",
    }
    catalog = protocol.get("label_system", {}).get("instructional_anchor_catalog")
    if not isinstance(catalog, list) or not catalog:
        raise RuntimeError("protocol instructional-anchor catalog is missing")
    for row in catalog:
        if not isinstance(row, dict) or not isinstance(row.get("frozen_relative_path"), str):
            raise RuntimeError("protocol instructional-anchor catalog is malformed")
        expected_names.add(row["frozen_relative_path"])
    listed = manifest.get("files")
    if not isinstance(listed, list) or not all(isinstance(row, dict) for row in listed):
        raise RuntimeError("protocol lock manifest file list is malformed")
    by_name = {row.get("name"): row for row in listed}
    if set(by_name) != expected_names or len(listed) != len(expected_names):
        raise RuntimeError("protocol lock member set changed")
    for relative, record in by_name.items():
        path = lock_root / str(relative)
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"protocol lock member missing or indirect: {path}")
        if record.get("bytes") != path.stat().st_size or record.get("sha256") != sha256_file(path):
            raise RuntimeError(f"protocol lock member hash/size changed: {path}")
    source_digest = sha256_file(Path(__file__).resolve())
    if source_digest != by_name["sources/select_dit_event_rich_classes.py"].get("sha256"):
        raise RuntimeError("running selector source differs from frozen snapshot")
    if protocol.get("selector_source", {}).get("sha256") != source_digest:
        raise RuntimeError("protocol selector-source binding changed")
    catalog_artifact = load_json(lock_root / "instructional_anchor_catalog.json")
    if (
        canonical_sha256(without_identity(catalog_artifact))
        != catalog_artifact.get("identity_sha256")
        or catalog_artifact.get("status")
        != "FROZEN_VISIBLE_INSTRUCTIONAL_ANCHORS_NOT_QUALIFICATION_GOLD"
        or catalog_artifact.get("anchors") != catalog
        or protocol.get("label_system", {}).get(
            "instructional_anchor_catalog_artifact_identity_sha256"
        )
        != catalog_artifact.get("identity_sha256")
    ):
        raise RuntimeError("instructional-anchor catalog binding changed")
    for row in catalog:
        relative = row["frozen_relative_path"]
        if (
            by_name[relative].get("bytes") != row.get("bytes")
            or by_name[relative].get("sha256") != row.get("sha256")
        ):
            raise RuntimeError(f"instructional-anchor payload binding changed: {relative}")
    lineage = protocol.get("label_system", {}).get("instructional_anchor_lineage", {})
    if (
        lineage.get("label_reliability_audit_summary_sha256")
        != by_name["label_audit/audit_summary.json"].get("sha256")
        or lineage.get("label_reliability_audit_report_sha256")
        != by_name["label_audit/AUDIT_REPORT.md"].get("sha256")
    ):
        raise RuntimeError("label-reliability audit binding changed")
    return protocol


def read_consensus(path: Path) -> tuple[list[dict[str, Any]], str]:
    path = path.expanduser().absolute()
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"consensus must be a regular non-symlink file: {path}")
    payload = path.read_bytes()
    allowed = {"phase", "class_id", "global_seed", "final_severity", "blur_component"}
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("consensus must be UTF-8") from exc
    with io.StringIO(text, newline="") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or ()) != allowed:
            raise RuntimeError(f"consensus columns must be exactly {sorted(allowed)}")
        rows: list[dict[str, Any]] = []
        for row in reader:
            severity = row["final_severity"]
            if severity not in {"clean_good", "mild_or_disputed", "clear_bad"}:
                raise RuntimeError(f"invalid final_severity: {severity}")
            if row["blur_component"] not in {"0", "1"}:
                raise RuntimeError("blur_component must be 0 or 1")
            blur = row["blur_component"] == "1"
            if blur and severity != "clear_bad":
                raise RuntimeError("blur_component=1 is allowed only for retained clear_bad")
            rows.append(
                {
                    "phase": row["phase"],
                    "class_id": int(row["class_id"]),
                    "global_seed": int(row["global_seed"]),
                    "severity": severity,
                    "blur": blur,
                }
            )
    return rows, hashlib.sha256(payload).hexdigest()


def validate_axis(
    rows: Iterable[Mapping[str, Any]], *, phase: str, classes: tuple[int, ...], seeds: tuple[int, ...]
) -> list[dict[str, Any]]:
    materialized = [dict(row) for row in rows]
    expected = {(class_id, seed) for class_id in classes for seed in seeds}
    observed = {(row["class_id"], row["global_seed"]) for row in materialized}
    if len(materialized) != len(expected) or observed != expected:
        raise RuntimeError("consensus row axis is incomplete, duplicated, or unexpected")
    if any(row["phase"] != phase for row in materialized):
        raise RuntimeError(f"every row phase must be {phase}")
    return materialized


def aggregate(rows: Iterable[Mapping[str, Any]], classes: tuple[int, ...]) -> list[dict[str, int]]:
    result: list[dict[str, int]] = []
    for class_id in classes:
        group = [row for row in rows if row["class_id"] == class_id]
        result.append(
            {
                "class_id": class_id,
                "n": len(group),
                "clear_bad": sum(row["severity"] == "clear_bad" for row in group),
                "blur_clear_bad": sum(bool(row["blur"]) for row in group),
                "clean_good": sum(row["severity"] == "clean_good" for row in group),
            }
        )
    return result


def validate_selection(protocol: Mapping[str, Any], selection: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema_version", "status", "protocol_identity_sha256",
        "consensus_file_sha256", "B_selected_classes", "C_selected_classes",
        "union_selected_classes", "aggregate_counts", "score_or_embedding_input_used",
        "identity_sha256",
    }
    if set(selection) != expected_keys:
        raise RuntimeError("selection schema changed")
    if (
        selection.get("schema_version") != 1
        or selection.get("status")
        != "SCREEN_DISCOVERY_CLASSES_SELECTED_BEFORE_ANCHOR_SAMPLING"
        or selection.get("protocol_identity_sha256") != protocol["identity_sha256"]
        or selection.get("score_or_embedding_input_used") is not False
        or canonical_sha256(without_identity(selection)) != selection.get("identity_sha256")
    ):
        raise RuntimeError("selection identity or frozen contract mismatch")
    consensus_hash = selection.get("consensus_file_sha256")
    if not isinstance(consensus_hash, str) or len(consensus_hash) != 64:
        raise RuntimeError("selection discovery-consensus hash is malformed")

    screen = protocol["endpoint_screen"]
    classes = tuple(int(row["class_id"]) for row in screen["class_roster"])
    order = {class_id: index for index, class_id in enumerate(classes)}
    take = int(screen["classes_selected_per_candidate"])
    expected_n = len(tuple(screen["discovery_seeds"]))
    counts = selection.get("aggregate_counts")
    if not isinstance(counts, list) or len(counts) != len(classes):
        raise RuntimeError("selection aggregate count axis changed")
    normalized: list[dict[str, int]] = []
    for expected_class, row in zip(classes, counts, strict=True):
        if not isinstance(row, dict) or set(row) != {
            "class_id", "n", "clear_bad", "blur_clear_bad", "clean_good"
        }:
            raise RuntimeError("selection aggregate count schema changed")
        if any(type(row[key]) is not int for key in row):
            raise RuntimeError("selection aggregate counts must be integers")
        if row["class_id"] != expected_class or row["n"] != expected_n:
            raise RuntimeError("selection aggregate class order or denominator changed")
        if not (
            0 <= row["blur_clear_bad"] <= row["clear_bad"] <= row["n"]
            and 0 <= row["clean_good"] <= row["n"] - row["clear_bad"]
        ):
            raise RuntimeError("selection aggregate counts are impossible")
        normalized.append(dict(row))
    expected_b = [
        row["class_id"] for row in sorted(
            normalized,
            key=lambda row: (-row["blur_clear_bad"], -row["clear_bad"], order[row["class_id"]]),
        )[:take]
    ]
    expected_c = [
        row["class_id"] for row in sorted(
            normalized,
            key=lambda row: (-row["clear_bad"], -row["blur_clear_bad"], order[row["class_id"]]),
        )[:take]
    ]
    expected_union = sorted(set(expected_b + expected_c), key=order.__getitem__)
    if (
        selection.get("B_selected_classes") != expected_b
        or selection.get("C_selected_classes") != expected_c
        or selection.get("union_selected_classes") != expected_union
    ):
        raise RuntimeError("selection class lists do not reproduce the frozen ranking")


def one_sided_wilson_lower(successes: int, total: int, z: float) -> float:
    if total <= 0 or not 0 <= successes <= total:
        raise RuntimeError("invalid binomial count")
    p = successes / total
    denominator = 1.0 + z * z / total
    center = p + z * z / (2.0 * total)
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
    return max(0.0, (center - radius) / denominator)


def rank(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.lock)
    screen = protocol["endpoint_screen"]
    classes = tuple(row["class_id"] for row in screen["class_roster"])
    seeds = tuple(screen["discovery_seeds"])
    raw_rows, consensus_sha256 = read_consensus(args.consensus)
    rows = validate_axis(raw_rows, phase="discovery", classes=classes, seeds=seeds)
    counts = aggregate(rows, classes)
    order = {class_id: index for index, class_id in enumerate(classes)}
    take = int(screen["classes_selected_per_candidate"])
    b_rows = sorted(
        counts,
        key=lambda row: (-row["blur_clear_bad"], -row["clear_bad"], order[row["class_id"]]),
    )[:take]
    c_rows = sorted(
        counts,
        key=lambda row: (-row["clear_bad"], -row["blur_clear_bad"], order[row["class_id"]]),
    )[:take]
    output: dict[str, Any] = {
        "schema_version": 1,
        "status": "SCREEN_DISCOVERY_CLASSES_SELECTED_BEFORE_ANCHOR_SAMPLING",
        "protocol_identity_sha256": protocol["identity_sha256"],
        "consensus_file_sha256": consensus_sha256,
        "B_selected_classes": [row["class_id"] for row in b_rows],
        "C_selected_classes": [row["class_id"] for row in c_rows],
        "union_selected_classes": sorted(
            {row["class_id"] for row in b_rows + c_rows}, key=order.__getitem__
        ),
        "aggregate_counts": counts,
        "score_or_embedding_input_used": False,
    }
    output["identity_sha256"] = canonical_sha256(output)
    validate_selection(protocol, output)
    write_exclusive(args.output, output)


def anchor(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.lock)
    selection = load_json(args.selection)
    validate_selection(protocol, selection)

    screen = protocol["endpoint_screen"]
    b_classes = tuple(selection["B_selected_classes"])
    c_classes = tuple(selection["C_selected_classes"])
    union = tuple(selection["union_selected_classes"])
    raw_rows, consensus_sha256 = read_consensus(args.consensus)
    rows = validate_axis(
        raw_rows,
        phase="anchor",
        classes=union,
        seeds=tuple(screen["anchor_seeds"]),
    )
    counts = aggregate(rows, union)
    by_class = {row["class_id"]: row for row in counts}
    z = float(protocol["forecast_rule"]["one_sided_wilson_z"])
    per_class_n = int(protocol["confirmation"]["samples_per_selected_class"])

    def summarize(candidate: str, selected: tuple[int, ...]) -> dict[str, Any]:
        event_key = "blur_clear_bad" if candidate == "B_blur_mean" else "clear_bad"
        total = sum(by_class[class_id]["n"] for class_id in selected)
        events = sum(by_class[class_id][event_key] for class_id in selected)
        clean = sum(by_class[class_id]["clean_good"] for class_id in selected)
        bearing = sum(by_class[class_id][event_key] > 0 for class_id in selected)
        lower = one_sided_wilson_lower(events, total, z)
        target = float(protocol["forecast_rule"][candidate]["conservative_expected_target"])
        conservative_expected = lower * len(selected) * per_class_n
        rule = protocol["forecast_rule"][candidate]
        go = (
            events >= int(rule["minimum_anchor_events"])
            and bearing >= int(rule["minimum_event_bearing_classes"])
            and clean >= int(rule["minimum_anchor_clean_good"])
            and conservative_expected >= target
        )
        return {
            "candidate": candidate,
            "selected_classes": list(selected),
            "anchor_rows": total,
            "anchor_events": events,
            "anchor_clean_good": clean,
            "event_bearing_classes": bearing,
            "one_sided_80pct_wilson_lower": lower,
            "planned_confirmation_rows": len(selected) * per_class_n if go else 0,
            "conservative_expected_confirmation_events": conservative_expected,
            "go": go,
        }

    b = summarize("B_blur_mean", b_classes)
    c = summarize("C_c3_low_jump", c_classes)
    active = set(b_classes if b["go"] else ()) | set(c_classes if c["go"] else ())
    roster_order = {
        row["class_id"]: index for index, row in enumerate(screen["class_roster"])
    }
    output: dict[str, Any] = {
        "schema_version": 1,
        "status": "PROSPECTIVE_TRACE_PLAN_LOCKED_AFTER_INDEPENDENT_ENDPOINT_ANCHOR",
        "protocol_identity_sha256": protocol["identity_sha256"],
        "selection_identity_sha256": selection["identity_sha256"],
        "anchor_consensus_file_sha256": consensus_sha256,
        "B_decision": b,
        "C_decision": c,
        "active_union_classes": sorted(active, key=roster_order.__getitem__),
        "calibration_seeds": protocol["confirmation"]["calibration_seeds"],
        "confirmation_seeds": protocol["confirmation"]["confirmation_seeds"],
        "total_full_trace_rows": len(active) * per_class_n,
        "candidate_products_must_be_physically_separate": True,
        "score_or_embedding_input_used": False,
    }
    output["identity_sha256"] = canonical_sha256(output)
    write_exclusive(args.output, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    sub = parser.add_subparsers(dest="command", required=True)
    rank_parser = sub.add_parser("rank")
    rank_parser.add_argument("--consensus", type=Path, required=True)
    rank_parser.add_argument("--output", type=Path, required=True)
    rank_parser.set_defaults(func=rank)
    anchor_parser = sub.add_parser("anchor")
    anchor_parser.add_argument("--selection", type=Path, required=True)
    anchor_parser.add_argument("--consensus", type=Path, required=True)
    anchor_parser.add_argument("--output", type=Path, required=True)
    anchor_parser.set_defaults(func=anchor)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    parsed.func(parsed)
