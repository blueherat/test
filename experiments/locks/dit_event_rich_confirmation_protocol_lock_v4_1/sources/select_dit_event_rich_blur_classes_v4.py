#!/usr/bin/env python3
"""Freeze the one blur-enriched class set and its independent anchor decision.

This selector is deliberately narrow.  ``rank`` reads only the already locked,
endpoint-only blind consensus for the 84-class discovery screen and ranks
classes by the number of retained blur/soft-fusion clear-bad events.  ``anchor``
reads only an independent endpoint-only consensus and decides whether the one
six-class set has enough blur events to justify internal B/E traces.

The visual labels are external experimental ascertainment.  They never become
an input to B, E, their calibration thresholds, a per-sample method ranking, or
an intervention decision.  This program has no argument for a trace, score,
embedding, FID, Inception, DINO, CLIP, or learned quality posterior.
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
DEFAULT_LOCK = ROOT / "experiments/locks/dit_event_rich_confirmation_protocol_lock_v4_1"
SELECTION_KIND = "EVENT_RICH_BLUR_SCREEN_SELECTION_LOCK_V1"
ANCHOR_KIND = "EVENT_RICH_BLUR_ANCHOR_PLAN_LOCK_V1"


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def without_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(value)
    output.pop("identity_sha256", None)
    return output


def write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def load_protocol(lock_root: Path) -> dict[str, Any]:
    lock_root = lock_root.expanduser().absolute()
    if not lock_root.is_dir() or lock_root.is_symlink():
        raise RuntimeError(f"v4 protocol lock must be a real directory: {lock_root}")
    protocol_path = lock_root / "protocol.json"
    manifest_path = lock_root / "manifest.json"
    completion_path = lock_root / "completion.json"
    protocol = load_json(protocol_path)
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    identity = protocol.get("identity_sha256")
    if canonical_sha256(without_identity(protocol)) != identity:
        raise RuntimeError("v4 protocol identity mismatch")
    if (
        protocol.get("schema_version") != 4
        or protocol.get("scientific_revision") != "v4.1"
        or protocol.get("status")
        != "SCIENTIFIC_V4_1_CORRECTED_FROZEN_EXECUTION_NOT_READY"
        or canonical_sha256(without_identity(manifest)) != manifest.get("identity_sha256")
        or manifest.get("status") != "complete"
        or manifest.get("protocol_identity_sha256") != identity
        or completion.get("complete") is not True
        or completion.get("protocol_identity_sha256") != identity
        or completion.get("protocol_file_sha256") != sha256_file(protocol_path)
        or completion.get("manifest_identity_sha256") != manifest.get("identity_sha256")
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
    ):
        raise RuntimeError("v4 protocol manifest/completion mismatch")
    listed = manifest.get("files")
    if not isinstance(listed, list) or not all(isinstance(row, dict) for row in listed):
        raise RuntimeError("v4 protocol manifest file list is malformed")
    by_name = {row.get("name"): row for row in listed}
    if len(by_name) != len(listed):
        raise RuntimeError("v4 protocol manifest contains duplicate members")
    expected_names = {
        "pre_sampling_zero_audit.json",
        "protocol.json",
        "sources/freeze_dit_event_rich_confirmation_protocol_v4.py",
        "sources/select_dit_event_rich_blur_classes_v4.py",
        "sources/selftest_dit_event_rich_scientific_v4.py",
        "upstream/event_completion_v3.json",
        "upstream/event_manifest_v3.json",
        "upstream/event_protocol_v3.json",
        "upstream/event_completion_v4_superseded.json",
        "upstream/event_manifest_v4_superseded.json",
        "upstream/event_protocol_v4_superseded.json",
        "upstream/matched_q_power_gate.json",
        "upstream/method_completion.json",
        "upstream/method_manifest.json",
        "upstream/method_protocol.json",
    }
    if set(by_name) != expected_names:
        raise RuntimeError("v4 protocol lock member set changed")
    for relative, record in by_name.items():
        path = lock_root / str(relative)
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"v4 lock member missing or indirect: {path}")
        if record.get("bytes") != path.stat().st_size or record.get("sha256") != sha256_file(path):
            raise RuntimeError(f"v4 lock member hash/size changed: {path}")
    source_digest = sha256_file(Path(__file__).resolve())
    if source_digest != by_name["sources/select_dit_event_rich_blur_classes_v4.py"].get("sha256"):
        raise RuntimeError("running selector differs from the v4 frozen snapshot")
    if protocol.get("selector_contract", {}).get("source_sha256") != source_digest:
        raise RuntimeError("v4 selector source binding changed")
    if protocol.get("execution_readiness", {}).get("ready_for_real_sampling") is not False:
        raise RuntimeError("design-only v4 lock unexpectedly authorizes real sampling")
    return protocol


def read_consensus(path: Path) -> tuple[list[dict[str, Any]], str]:
    path = path.expanduser().absolute()
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"consensus must be a regular non-symlink file: {path}")
    payload = path.read_bytes()
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("consensus must be UTF-8") from exc
    allowed = {"phase", "class_id", "global_seed", "final_severity", "blur_component"}
    with io.StringIO(text, newline="") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or ()) != allowed:
            raise RuntimeError(
                "consensus columns must be exactly phase,class_id,global_seed,"
                "final_severity,blur_component; score/embedding columns are forbidden"
            )
        rows: list[dict[str, Any]] = []
        for raw in reader:
            if raw["final_severity"] not in {
                "clean_good", "mild_or_disputed", "clear_bad"
            }:
                raise RuntimeError("invalid final_severity")
            if raw["blur_component"] not in {"0", "1"}:
                raise RuntimeError("blur_component must be 0 or 1")
            blur = raw["blur_component"] == "1"
            if blur and raw["final_severity"] != "clear_bad":
                raise RuntimeError("blur_component=1 requires final clear_bad")
            try:
                class_id = int(raw["class_id"])
                global_seed = int(raw["global_seed"])
            except ValueError as exc:
                raise RuntimeError("class_id/global_seed must be canonical integers") from exc
            if str(class_id) != raw["class_id"] or str(global_seed) != raw["global_seed"]:
                raise RuntimeError("class_id/global_seed must use canonical decimal spelling")
            rows.append(
                {
                    "phase": raw["phase"],
                    "class_id": class_id,
                    "global_seed": global_seed,
                    "severity": raw["final_severity"],
                    "blur": blur,
                }
            )
    return rows, hashlib.sha256(payload).hexdigest()


def validate_axis(
    rows: Iterable[Mapping[str, Any]],
    *,
    phase: str,
    classes: tuple[int, ...],
    seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    materialized = [dict(row) for row in rows]
    expected_order = [(class_id, seed) for seed in seeds for class_id in classes]
    observed = [(int(row["class_id"]), int(row["global_seed"])) for row in materialized]
    if len(observed) != len(set(observed)):
        raise RuntimeError("consensus axis contains duplicate class/seed pairs")
    if set(observed) != set(expected_order):
        raise RuntimeError("consensus axis is incomplete or contains unexpected class/seed pairs")
    if any(row["phase"] != phase for row in materialized):
        raise RuntimeError(f"every consensus row phase must be {phase}")
    return materialized


def aggregate(rows: Iterable[Mapping[str, Any]], classes: tuple[int, ...]) -> list[dict[str, int]]:
    materialized = list(rows)
    result: list[dict[str, int]] = []
    for class_id in classes:
        group = [row for row in materialized if int(row["class_id"]) == class_id]
        result.append(
            {
                "class_id": class_id,
                "n": len(group),
                "blur_clear_bad": sum(bool(row["blur"]) for row in group),
                "clean_good": sum(row["severity"] == "clean_good" for row in group),
            }
        )
    return result


def validate_selection(protocol: Mapping[str, Any], selection: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema_version",
        "artifact_kind",
        "status",
        "protocol_identity_sha256",
        "discovery_consensus_file_sha256",
        "selected_classes",
        "aggregate_counts",
        "ranking_rule",
        "external_visual_labels_used_only_for_cohort_enrichment",
        "method_score_threshold_intervention_or_external_representation_input_used",
        "identity_sha256",
    }
    if set(selection) != expected_keys:
        raise RuntimeError("v4 selection schema changed")
    if (
        selection.get("schema_version") != 1
        or selection.get("artifact_kind") != SELECTION_KIND
        or selection.get("status") != "BLUR_ENRICHED_CLASSES_SELECTED_BEFORE_ANCHOR"
        or selection.get("protocol_identity_sha256") != protocol.get("identity_sha256")
        or selection.get("external_visual_labels_used_only_for_cohort_enrichment") is not True
        or selection.get(
            "method_score_threshold_intervention_or_external_representation_input_used"
        )
        is not False
        or canonical_sha256(without_identity(selection)) != selection.get("identity_sha256")
    ):
        raise RuntimeError("v4 selection identity or boundary changed")
    digest = selection.get("discovery_consensus_file_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError("discovery consensus hash is malformed")
    screen = protocol["endpoint_screen"]
    classes = tuple(int(row["class_id"]) for row in screen["class_roster"])
    roster_order = {class_id: index for index, class_id in enumerate(classes)}
    expected_n = len(tuple(screen["discovery_seeds"]))
    counts = selection.get("aggregate_counts")
    if not isinstance(counts, list) or len(counts) != len(classes):
        raise RuntimeError("selection aggregate class axis changed")
    normalized: list[dict[str, int]] = []
    for expected_class, row in zip(classes, counts, strict=True):
        if not isinstance(row, dict) or set(row) != {
            "class_id", "n", "blur_clear_bad", "clean_good"
        }:
            raise RuntimeError("selection aggregate schema changed")
        if any(type(row[key]) is not int for key in row):
            raise RuntimeError("selection aggregate values must be integers")
        if row["class_id"] != expected_class or row["n"] != expected_n:
            raise RuntimeError("selection aggregate class order or denominator changed")
        if not (
            0 <= row["blur_clear_bad"] <= row["n"]
            and 0 <= row["clean_good"] <= row["n"] - row["blur_clear_bad"]
        ):
            raise RuntimeError("selection aggregate counts are impossible")
        normalized.append(dict(row))
    take = int(screen["selected_class_count"])
    expected_classes = [
        row["class_id"]
        for row in sorted(
            normalized,
            key=lambda row: (-row["blur_clear_bad"], roster_order[row["class_id"]]),
        )[:take]
    ]
    if selection.get("ranking_rule") != screen["ranking_rule"]:
        raise RuntimeError("selection ranking text differs from protocol")
    if selection.get("selected_classes") != expected_classes:
        raise RuntimeError("selected class list does not reproduce blur-only ranking")


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
    classes = tuple(int(row["class_id"]) for row in screen["class_roster"])
    seeds = tuple(int(seed) for seed in screen["discovery_seeds"])
    raw_rows, consensus_hash = read_consensus(args.consensus)
    rows = validate_axis(raw_rows, phase="discovery", classes=classes, seeds=seeds)
    counts = aggregate(rows, classes)
    order = {class_id: index for index, class_id in enumerate(classes)}
    selected = [
        row["class_id"]
        for row in sorted(
            counts, key=lambda row: (-row["blur_clear_bad"], order[row["class_id"]])
        )[: int(screen["selected_class_count"])]
    ]
    output: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": SELECTION_KIND,
        "status": "BLUR_ENRICHED_CLASSES_SELECTED_BEFORE_ANCHOR",
        "protocol_identity_sha256": protocol["identity_sha256"],
        "discovery_consensus_file_sha256": consensus_hash,
        "selected_classes": selected,
        "aggregate_counts": counts,
        "ranking_rule": screen["ranking_rule"],
        "external_visual_labels_used_only_for_cohort_enrichment": True,
        "method_score_threshold_intervention_or_external_representation_input_used": False,
    }
    output["identity_sha256"] = canonical_sha256(output)
    validate_selection(protocol, output)
    write_exclusive(args.output, output)


def anchor(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.lock)
    selection = load_json(args.selection)
    validate_selection(protocol, selection)
    selected = tuple(int(value) for value in selection["selected_classes"])
    screen = protocol["endpoint_screen"]
    raw_rows, consensus_hash = read_consensus(args.consensus)
    rows = validate_axis(
        raw_rows,
        phase="anchor",
        classes=selected,
        seeds=tuple(int(seed) for seed in screen["anchor_seeds"]),
    )
    counts = aggregate(rows, selected)
    events = sum(row["blur_clear_bad"] for row in counts)
    clean = sum(row["clean_good"] for row in counts)
    bearing = sum(row["blur_clear_bad"] > 0 for row in counts)
    rule = protocol["anchor_go_rule"]
    gates = {
        "blur_events": events >= int(rule["minimum_blur_clear_bad"]),
        "event_bearing_classes": bearing >= int(rule["minimum_event_bearing_classes"]),
        "clean_good": clean >= int(rule["minimum_clean_good"]),
    }
    go = all(gates.values())
    z = float(rule["descriptive_wilson_z"])
    output: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": ANCHOR_KIND,
        "status": "BLUR_ANCHOR_GO_DECISION_LOCKED_BEFORE_INTERNAL_TRACES",
        "protocol_identity_sha256": protocol["identity_sha256"],
        "selection_identity_sha256": selection["identity_sha256"],
        "anchor_consensus_file_sha256": consensus_hash,
        "selected_classes": list(selected),
        "aggregate_counts": counts,
        "decision": {
            "anchor_rows": len(rows),
            "blur_clear_bad": events,
            "event_bearing_classes": bearing,
            "clean_good": clean,
            "gates": gates,
            "go": go,
            "failed_gates": [name for name, passed in gates.items() if not passed],
        },
        "descriptive_only": {
            "one_sided_80pct_wilson_lower": one_sided_wilson_lower(events, len(rows), z),
            "not_a_go_input_or_auc_power_guarantee": True,
        },
        "calibration_seeds": protocol["dynamic_axis"]["calibration_seeds"] if go else [],
        "confirmation_seeds": protocol["dynamic_axis"]["confirmation_seeds"] if go else [],
        "calibration_trace_rows": len(selected)
        * len(protocol["dynamic_axis"]["calibration_seeds"])
        if go
        else 0,
        "confirmation_trace_rows": len(selected)
        * len(protocol["dynamic_axis"]["confirmation_seeds"])
        if go
        else 0,
        "B_and_E_share_exact_selected_class_set": True,
        "external_visual_labels_used_only_for_cohort_enrichment_and_go": True,
        "method_score_threshold_intervention_or_external_representation_input_used": False,
    }
    output["identity_sha256"] = canonical_sha256(output)
    write_exclusive(args.output, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
