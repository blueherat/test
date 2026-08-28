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
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = ROOT / "experiments/locks/dit_event_rich_confirmation_protocol_lock_v1"


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
    protocol = load_json(lock_root / "protocol.json")
    identity = protocol.get("identity_sha256")
    if canonical_sha256(without_identity(protocol)) != identity:
        raise RuntimeError("protocol identity mismatch")
    return protocol


def read_consensus(path: Path) -> list[dict[str, Any]]:
    allowed = {"phase", "class_id", "global_seed", "final_severity", "blur_component"}
    with path.open("r", encoding="utf-8", newline="") as handle:
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
    return rows


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
    rows = validate_axis(
        read_consensus(args.consensus), phase="discovery", classes=classes, seeds=seeds
    )
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
        "consensus_file_sha256": hashlib.sha256(args.consensus.read_bytes()).hexdigest(),
        "B_selected_classes": [row["class_id"] for row in b_rows],
        "C_selected_classes": [row["class_id"] for row in c_rows],
        "union_selected_classes": sorted(
            {row["class_id"] for row in b_rows + c_rows}, key=order.__getitem__
        ),
        "aggregate_counts": counts,
        "score_or_embedding_input_used": False,
    }
    output["identity_sha256"] = canonical_sha256(output)
    write_exclusive(args.output, output)


def anchor(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.lock)
    selection = load_json(args.selection)
    if canonical_sha256(without_identity(selection)) != selection.get("identity_sha256"):
        raise RuntimeError("selection identity mismatch")
    if selection.get("protocol_identity_sha256") != protocol["identity_sha256"]:
        raise RuntimeError("selection belongs to another protocol")

    screen = protocol["endpoint_screen"]
    b_classes = tuple(selection["B_selected_classes"])
    c_classes = tuple(selection["C_selected_classes"])
    union = tuple(selection["union_selected_classes"])
    rows = validate_axis(
        read_consensus(args.consensus),
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
        "anchor_consensus_file_sha256": hashlib.sha256(args.consensus.read_bytes()).hexdigest(),
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
