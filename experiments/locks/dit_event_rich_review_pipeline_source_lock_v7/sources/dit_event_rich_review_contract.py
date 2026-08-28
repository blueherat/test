#!/usr/bin/env python3
"""Shared fail-closed contract for event-rich endpoint-only visual labels.

This module contains no labels and performs no image scoring.  It binds the
review infrastructure to the frozen event-rich protocol v3 and supplies
strict CSV, provenance, hashing, and immutable-artifact helpers.  Production
scripts are expected to be executed from, or byte-identical to, a frozen
review source lock.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
EVENT_PROTOCOL_LOCK = ROOT / "experiments/locks/dit_event_rich_confirmation_protocol_lock_v3"
REVIEW_SOURCE_LOCK = ROOT / "experiments/locks/dit_event_rich_review_pipeline_source_lock_v7"
REVIEW_SOURCE_LOCK_KIND = "EVENT_RICH_REVIEW_PIPELINE_SOURCE_LOCK_V7"
EVENT_PROTOCOL_IDENTITY = "04e933793992e2a7ce62aa4ac66836412f3c4f221cce731f2e072da97e892dd7"

ROLE_SLOTS = (
    "reviewer_1",
    "reviewer_2",
    "reviewer_3",
    "adjudicator_1",
    "adjudicator_2",
)
REVIEWER_SLOTS = ROLE_SLOTS[:3]
ADJUDICATOR_SLOTS = ROLE_SLOTS[3:]
PHASES = ("discovery", "anchor", "confirmation")
PHASE_SEEDS = {
    "discovery": tuple(range(1000, 1012)),
    "anchor": tuple(range(1012, 1036)),
    "confirmation": tuple(range(1200, 1328)),
}

COMPONENTS = (
    "global_blur",
    "local_blur",
    "soft_fusion_or_melting",
    "discrete_duplication_or_extra_part",
    "detachment_or_floating_part",
    "topology_or_attachment_error",
    "limb_or_object_misalignment",
    "texture_break",
    "other",
    "none",
)
BLUR_COMPONENTS = frozenset(COMPONENTS[:3])
STRUCTURE_COMPONENTS = frozenset(COMPONENTS[3:7])

VISIBLE_RATIFICATION_FIELDS = (
    "anchor_id",
    "image_sha256",
    "proposed_severity",
    "proposed_component_group",
    "proposed_reason",
    "decision",
    "correction_severity",
    "correction_component_group",
    "correction_reason",
    "expert_role_token",
    "independence_attestation",
)
HIDDEN_ITEM_FIELDS = (
    "item_id",
    "class_id",
    "class_name",
    "image_path",
    "image_sha256",
    "image_pixel_sha256",
    "width",
    "height",
    "mode",
)
EXPERT_LABEL_FIELDS = (
    "item_id",
    "severity",
    "components",
    "localization_reason",
    "expert_role_token",
    "independence_attestation",
)
QUALIFICATION_RESPONSE_FIELDS = (
    "qualification_id",
    "severity",
    "components",
    "localization_reason",
    "role_slot",
    "role_token",
    "independence_attestation",
)
REVIEW_RESPONSE_FIELDS = (
    "blind_id",
    "severity",
    "components",
    "localization_reason",
    "role_slot",
    "role_token",
    "independence_attestation",
)
ADJUDICATION_RESPONSE_FIELDS = (
    "adjudication_id",
    "decision",
    "components",
    "localization_reason",
    "role_slot",
    "role_token",
    "independence_attestation",
)

FORBIDDEN_NAME_FRAGMENTS = (
    "score",
    "metric",
    "feature",
    "trajectory",
    "trace",
    "embedding",
    "inception",
    "dino",
    "fid",
    "candidate",
    "threshold",
    "alert",
    "rank",
    "stratum",
    "vote_count",
    "reviewer_identity",
    "trigger_source",
    "gold_label",
)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def without_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("identity_sha256", None)
    return result


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


def load_json(path: Path) -> dict[str, Any]:
    path = require_regular(path, "JSON input")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def read_csv_exact(
    path: Path,
    fields: Sequence[str],
    *,
    allow_empty: bool = False,
) -> list[dict[str, str]]:
    path = require_regular(path, "CSV input")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        observed = tuple(reader.fieldnames or ())
        if observed != tuple(fields):
            raise RuntimeError(
                f"CSV schema changed for {path}: expected={list(fields)}, observed={list(observed)}"
            )
        rows = [dict(row) for row in reader]
    if not rows and not allow_empty:
        raise RuntimeError(f"CSV contains no data rows: {path}")
    return rows


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for source in rows:
            row = {field: source.get(field, "") for field in fields}
            if set(source) - set(fields):
                raise RuntimeError(f"attempt to write undeclared CSV fields: {sorted(set(source)-set(fields))}")
            writer.writerow(row)


def validate_unique_axis(
    rows: Sequence[Mapping[str, Any]],
    key_fields: Sequence[str],
    expected_keys: Sequence[tuple[Any, ...]] | None = None,
) -> tuple[tuple[Any, ...], ...]:
    keys = tuple(tuple(row[field] for field in key_fields) for row in rows)
    if len(set(keys)) != len(keys):
        raise RuntimeError(f"duplicate row key for fields {list(key_fields)}")
    if expected_keys is not None and keys != tuple(expected_keys):
        expected = tuple(expected_keys)
        missing = sorted(set(expected) - set(keys), key=str)[:8]
        extra = sorted(set(keys) - set(expected), key=str)[:8]
        raise RuntimeError(
            f"row axis changed for {list(key_fields)}: missing={missing}, extra={extra}, order_equal={keys == expected}"
        )
    return keys


def reject_forbidden_columns(fields: Iterable[str], *, allow: Iterable[str] = ()) -> None:
    allowed = set(allow)
    bad: list[str] = []
    for field in fields:
        normalized = field.strip().lower()
        if field in allowed:
            continue
        if any(fragment in normalized for fragment in FORBIDDEN_NAME_FRAGMENTS):
            bad.append(field)
        if normalized in {"label", "raw_label", "consensus", "bad", "good"}:
            bad.append(field)
    if bad:
        raise RuntimeError(f"forbidden evidence/label columns present: {sorted(set(bad))}")


def parse_int(value: str, name: str, minimum: int, maximum: int) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if str(result) != str(value).strip() or not minimum <= result <= maximum:
        raise RuntimeError(f"{name} must lie in [{minimum},{maximum}] with canonical spelling")
    return result


def parse_components(value: str, severity: int) -> tuple[str, ...]:
    pieces = tuple(part.strip() for part in value.split(";") if part.strip())
    if not pieces or len(set(pieces)) != len(pieces):
        raise RuntimeError("components must be a nonempty, duplicate-free semicolon list")
    if any(piece not in COMPONENTS for piece in pieces):
        raise RuntimeError(f"unknown component in {pieces}")
    if "none" in pieces and len(pieces) != 1:
        raise RuntimeError("component 'none' cannot be combined with another component")
    if severity == 0 and pieces != ("none",):
        raise RuntimeError("severity 0 requires components=none")
    if severity >= 2 and pieces == ("none",):
        raise RuntimeError("severity 2/3 requires a localized non-none component")
    return tuple(sorted(pieces, key=COMPONENTS.index))


def validate_reason(reason: str, severity: int, *, always: bool = False) -> str:
    reason = reason.strip()
    if (always or severity >= 2) and len(reason) < 12:
        raise RuntimeError("a specific localization/reason of at least 12 characters is required")
    if "\n" in reason or "\r" in reason:
        raise RuntimeError("localization/reason must be one line")
    return reason


def validate_attestation(value: str) -> None:
    if value != "I independently reviewed endpoint pixels only under the frozen rubric":
        raise RuntimeError("independence attestation is absent or changed")


def validate_severity_row(row: Mapping[str, str], *, always_reason: bool = False) -> dict[str, Any]:
    severity = parse_int(row["severity"], "severity", 0, 3)
    components = parse_components(row["components"], severity)
    reason = validate_reason(row["localization_reason"], severity, always=always_reason)
    return {"severity": severity, "components": list(components), "localization_reason": reason}


def artifact_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"artifact contains a symlink: {path}")
        if not path.is_file() or path.name in {"manifest.json", "completion.json"}:
            continue
        records.append(
            {
                "name": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def publish_artifact(
    output: Path,
    *,
    identity: Mapping[str, Any],
    builder: Callable[[Path], None],
) -> Path:
    output = output.expanduser().absolute()
    if os.path.lexists(output):
        raise RuntimeError(f"refusing to overwrite immutable output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".tmp-", dir=str(output.parent)))
    try:
        builder(temporary)
        frozen_identity = dict(identity)
        frozen_identity["identity_sha256"] = canonical_sha256(frozen_identity)
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "complete",
            "identity": frozen_identity,
            "identity_sha256": frozen_identity["identity_sha256"],
            "files": artifact_records(temporary),
        }
        manifest["manifest_identity_sha256"] = canonical_sha256(manifest)
        write_json(temporary / "manifest.json", manifest)
        completion = {
            "complete": True,
            "identity_sha256": frozen_identity["identity_sha256"],
            "manifest_identity_sha256": manifest["manifest_identity_sha256"],
            "manifest_file_sha256": sha256_file(temporary / "manifest.json"),
            "file_count": len(manifest["files"]),
        }
        write_json(temporary / "completion.json", completion)
        os.rename(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def validate_artifact(root: Path, *, expected_kind: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    root = require_directory(root, "immutable artifact")
    manifest_path = require_regular(root / "manifest.json", "artifact manifest")
    completion_path = require_regular(root / "completion.json", "artifact completion")
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    identity = manifest.get("identity")
    if not isinstance(identity, dict):
        raise RuntimeError("artifact identity is missing")
    identity_sha = canonical_sha256(without_identity(identity))
    expected_completion = {
        "complete": True,
        "identity_sha256": identity_sha,
        "manifest_identity_sha256": manifest.get("manifest_identity_sha256"),
        "manifest_file_sha256": sha256_file(manifest_path),
        "file_count": len(manifest.get("files", [])),
    }
    manifest_without_id = dict(manifest)
    manifest_without_id.pop("manifest_identity_sha256", None)
    if (
        identity.get("identity_sha256") != identity_sha
        or manifest.get("identity_sha256") != identity_sha
        or canonical_sha256(manifest_without_id) != manifest.get("manifest_identity_sha256")
        or manifest.get("files") != artifact_records(root)
        or completion != expected_completion
        or manifest.get("status") != "complete"
        or (expected_kind is not None and identity.get("artifact_kind") != expected_kind)
    ):
        raise RuntimeError(f"immutable artifact failed validation: {root}")
    return manifest, completion


def validate_event_protocol_lock(root: Path = EVENT_PROTOCOL_LOCK) -> tuple[dict[str, Any], dict[str, Any]]:
    root = require_directory(root, "event protocol v3 lock")
    protocol_path = require_regular(root / "protocol.json", "event protocol")
    manifest_path = require_regular(root / "manifest.json", "event protocol manifest")
    completion_path = require_regular(root / "completion.json", "event protocol completion")
    protocol = load_json(protocol_path)
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    protocol_identity = canonical_sha256(without_identity(protocol))
    if (
        protocol.get("schema_version") != 3
        or protocol.get("status") != "FROZEN_BEFORE_REVIEWER_QUALIFICATION_OR_EVENT_RICH_SCREEN"
        or protocol.get("identity_sha256") != EVENT_PROTOCOL_IDENTITY
        or protocol_identity != EVENT_PROTOCOL_IDENTITY
        or completion.get("complete") is not True
        or completion.get("protocol_identity_sha256") != EVENT_PROTOCOL_IDENTITY
        or completion.get("protocol_file_sha256") != sha256_file(protocol_path)
        or manifest.get("protocol_identity_sha256") != EVENT_PROTOCOL_IDENTITY
    ):
        raise RuntimeError("frozen event protocol v3 identity/contract changed")
    quality = protocol.get("label_system", {}).get("reviewer_qualification", {})
    dual = protocol.get("label_system", {}).get("dual_adjudication_and_miss_audit", {})
    if (
        quality.get("hidden_set_size") != 60
        or quality.get("every_pair_minimum_positive_agreement") != 0.6
        or quality.get("every_pair_minimum_binary_cohen_kappa") != 0.5
        or quality.get("individual_minimum_clear_bad_recall_against_hidden_gold") != 0.8
        or quality.get("individual_minimum_non_clear_bad_specificity_against_hidden_gold") != 0.8
        or dual.get("unanimous_three_reviewer_clear_bad_is_never_downgradable") is not True
        or dual.get("single_adjudicator_can_change_final_severity") is not False
    ):
        raise RuntimeError("event protocol v3 label-quality constants changed")
    return protocol, {
        "path": str(root),
        "protocol_identity_sha256": EVENT_PROTOCOL_IDENTITY,
        "protocol_file_sha256": sha256_file(protocol_path),
        "manifest_file_sha256": sha256_file(manifest_path),
        "completion_file_sha256": sha256_file(completion_path),
    }


def validate_source_lock(
    root: Path = REVIEW_SOURCE_LOCK,
    *,
    invoked_source: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest, _ = validate_artifact(root, expected_kind=REVIEW_SOURCE_LOCK_KIND)
    identity = manifest["identity"]
    if identity.get("event_protocol_identity_sha256") != EVENT_PROTOCOL_IDENTITY:
        raise RuntimeError("review source lock is not bound to event protocol v3")
    contract = load_json(root / "review_contract.json")
    if contract.get("event_protocol_identity_sha256") != EVENT_PROTOCOL_IDENTITY:
        raise RuntimeError("frozen review contract identity changed")
    snapshots = identity.get("source_snapshots")
    if not isinstance(snapshots, dict):
        raise RuntimeError("source lock lacks source snapshots")
    for basename, record in snapshots.items():
        snapshot = require_regular(root / "sources" / basename, f"frozen source {basename}")
        if not isinstance(record, dict) or sha256_file(snapshot) != record.get("sha256"):
            raise RuntimeError(f"frozen review source changed: {basename}")
    if invoked_source is not None:
        invoked_source = require_regular(invoked_source, "invoked review source")
        expected = snapshots.get(invoked_source.name)
        if not isinstance(expected, dict) or sha256_file(invoked_source) != expected.get("sha256"):
            raise RuntimeError(
                "invoked review script is not byte-identical to its frozen source; run the frozen snapshot"
            )
    return manifest, contract


def binary_cohen_kappa(left: Sequence[int], right: Sequence[int]) -> float:
    if len(left) != len(right) or not left:
        raise RuntimeError("kappa inputs must have equal nonzero length")
    if any(value not in (0, 1) for value in (*left, *right)):
        raise RuntimeError("kappa inputs must be binary")
    n = len(left)
    observed = sum(a == b for a, b in zip(left, right)) / n
    p_left = sum(left) / n
    p_right = sum(right) / n
    expected = p_left * p_right + (1 - p_left) * (1 - p_right)
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else float("-inf")
    return (observed - expected) / (1 - expected)


def positive_agreement(left: Sequence[int], right: Sequence[int]) -> float:
    n11 = sum(a == 1 and b == 1 for a, b in zip(left, right))
    n10 = sum(a == 1 and b == 0 for a, b in zip(left, right))
    n01 = sum(a == 0 and b == 1 for a, b in zip(left, right))
    denominator = 2 * n11 + n10 + n01
    return 1.0 if denominator == 0 else 2 * n11 / denominator


def qualification_metrics(gold: Sequence[int], predictions: Mapping[str, Sequence[int]]) -> dict[str, Any]:
    if len(gold) != 60 or sum(gold) != 30:
        raise RuntimeError("qualification gold must contain exactly 30 binary positives and 30 negatives")
    roles = tuple(predictions)
    if roles != ROLE_SLOTS:
        raise RuntimeError(f"qualification panel roles/order changed: {roles}")
    individuals: dict[str, Any] = {}
    for role, values in predictions.items():
        if len(values) != 60:
            raise RuntimeError(f"qualification response count changed for {role}")
        tp = sum(g == 1 and p == 1 for g, p in zip(gold, values))
        fn = sum(g == 1 and p == 0 for g, p in zip(gold, values))
        tn = sum(g == 0 and p == 0 for g, p in zip(gold, values))
        fp = sum(g == 0 and p == 1 for g, p in zip(gold, values))
        recall = tp / (tp + fn)
        specificity = tn / (tn + fp)
        individuals[role] = {
            "tp": tp,
            "fn": fn,
            "tn": tn,
            "fp": fp,
            "clear_bad_recall": recall,
            "non_clear_bad_specificity": specificity,
            "passes": recall >= 0.8 and specificity >= 0.8,
        }
    pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(roles):
        for right in roles[left_index + 1 :]:
            a = list(predictions[left])
            b = list(predictions[right])
            n11 = sum(x == 1 and y == 1 for x, y in zip(a, b))
            n10 = sum(x == 1 and y == 0 for x, y in zip(a, b))
            n01 = sum(x == 0 and y == 1 for x, y in zip(a, b))
            n00 = sum(x == 0 and y == 0 for x, y in zip(a, b))
            agreement = positive_agreement(a, b)
            kappa = binary_cohen_kappa(a, b)
            pairs.append(
                {
                    "left": left,
                    "right": right,
                    "n11": n11,
                    "n10": n10,
                    "n01": n01,
                    "n00": n00,
                    "positive_agreement": agreement,
                    "binary_cohen_kappa": kappa,
                    "passes": agreement >= 0.6 and kappa >= 0.5,
                }
            )
    passed = all(row["passes"] for row in individuals.values()) and all(row["passes"] for row in pairs)
    return {
        "individuals": individuals,
        "pairs": pairs,
        "panel_passed": passed,
        "thresholds": {
            "individual_clear_bad_recall_minimum": 0.8,
            "individual_non_clear_bad_specificity_minimum": 0.8,
            "every_pair_positive_agreement_minimum": 0.6,
            "every_pair_binary_cohen_kappa_minimum": 0.5,
        },
    }


def qualified_role_tokens_valid(value: Any) -> bool:
    """Return whether a serialized PASS token map covers the exact five roles.

    JSON objects are unordered.  The check intentionally compares the key set
    rather than insertion order, while still requiring five distinct nonempty
    role tokens.
    """

    return (
        isinstance(value, dict)
        and set(value) == set(ROLE_SLOTS)
        and all(isinstance(token, str) and bool(token) for token in value.values())
        and len(set(value.values())) == len(ROLE_SLOTS)
    )


def stable_blind_id(domain: str, *parts: Any, length: int = 20) -> str:
    payload = "\0".join([domain, *(str(part) for part in parts)]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]
