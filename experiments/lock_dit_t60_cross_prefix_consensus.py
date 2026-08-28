#!/usr/bin/env python3
"""Seal two independent reviews and one evidence-blind consensus annotation.

``seal-review`` converts one fully completed reviewer template into an
immutable, self-hashed file.  Run it separately for reviewer A and reviewer B
before either reviewer sees the other's labels.  ``lock-consensus`` then
requires both sealed reviews plus a fully completed adjudication template,
checks every disagreement/uncertain call, and atomically writes one closed
consensus-lock bundle.  No sampler shard, mapping, trace, score, or alarm is
opened by this program.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import re
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

try:
    from .build_dit_t60_cross_prefix_blind_pack import (
        ADJUDICATION_DECLARATION,
        ADJUDICATION_TEMPLATE_NAME,
        ANNOTATION_FIELDS,
        EXPERIMENT as BLIND_EXPERIMENT,
        PUBLIC_DIR_NAME,
        PUBLIC_MANIFEST_NAME,
        REVIEW_A_TEMPLATE_NAME,
        REVIEW_B_TEMPLATE_NAME,
        REVIEW_DECLARATION,
        RUBRIC_NAME,
        TOTAL_POOL_BRANCHES,
        _atomic_install_directory_noreplace,
        _canonical_self_hash,
        _read_self_hashed_json,
        _reject_special_entries,
        validate_completed_annotation_row,
        validate_public_pack,
    )
    from .reproduce_dit_imagenet256 import atomic_json_dump, load_json, sha256_file
except ImportError:  # pragma: no cover - direct CLI execution.
    from build_dit_t60_cross_prefix_blind_pack import (
        ADJUDICATION_DECLARATION,
        ADJUDICATION_TEMPLATE_NAME,
        ANNOTATION_FIELDS,
        EXPERIMENT as BLIND_EXPERIMENT,
        PUBLIC_DIR_NAME,
        PUBLIC_MANIFEST_NAME,
        REVIEW_A_TEMPLATE_NAME,
        REVIEW_B_TEMPLATE_NAME,
        REVIEW_DECLARATION,
        RUBRIC_NAME,
        TOTAL_POOL_BRANCHES,
        _atomic_install_directory_noreplace,
        _canonical_self_hash,
        _read_self_hashed_json,
        _reject_special_entries,
        validate_completed_annotation_row,
        validate_public_pack,
    )
    from reproduce_dit_imagenet256 import atomic_json_dump, load_json, sha256_file


EXPERIMENT = "dit_imagenet256_t60_cross_prefix_visual_consensus_lock"
SCHEMA_VERSION = 1
REVIEW_SCHEMA = "dit_t60_cross_prefix_independent_visual_review_v1"
CONSENSUS_SCHEMA = "dit_t60_cross_prefix_visual_consensus_v1"
REVIEW_STATUS = "LOCKED_COMPLETE_INDEPENDENT_REVIEW"
CONSENSUS_STATUS = "LOCKED_COMPLETE_CONSENSUS_BEFORE_EVIDENCE_UNSEAL"
REVIEW_A_COPY_NAME = "review_A_locked.json"
REVIEW_B_COPY_NAME = "review_B_locked.json"
CONSENSUS_NAME = "consensus_locked.json"
MANIFEST_NAME = "manifest.json"
COMPLETION_NAME = "completion.json"
RUNNER = Path(__file__).resolve()

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_VISIBLE_REASON_FORBIDDEN = re.compile(
    r"(?:seed|shard|slot|generation[ _-]?order|evidence|alarm|score|rank|"
    r"likelihood|theta|delta[ _-]?nu|e[ _-]?mix|trace)",
    flags=re.IGNORECASE,
)


def _parse_utc(value: Any, context: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise RuntimeError(f"{context} must be RFC3339 UTC ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise RuntimeError(f"{context} is not a valid timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise RuntimeError(f"{context} must be UTC")
    return parsed


def _identifier(value: Any, context: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise RuntimeError(f"{context} must match {_IDENTIFIER_RE.pattern}")
    return value


def _canonical_plain_directory(path: Path, context: str) -> Path:
    value = path.expanduser().absolute()
    if value.is_symlink() or not value.is_dir():
        raise RuntimeError(f"{context} must be a plain directory")
    resolved = value.resolve(strict=True)
    if resolved != value:
        raise RuntimeError(f"{context} must use one canonical non-symlink path")
    return resolved


def _canonical_plain_file(path: Path, context: str) -> Path:
    value = path.expanduser().absolute()
    if value.is_symlink() or not value.is_file():
        raise RuntimeError(f"{context} must be a plain file")
    resolved = value.resolve(strict=True)
    if resolved != value:
        raise RuntimeError(f"{context} must use one canonical non-symlink path")
    return resolved


def _canonical_new_output(path: Path, context: str) -> Path:
    value = path.expanduser().absolute()
    if os.path.lexists(value):
        raise RuntimeError(f"{context} already exists")
    resolved = value.resolve(strict=False)
    if resolved != value:
        raise RuntimeError(
            f"{context} must use a canonical path with no symlinked parent"
        )
    return resolved


def _sealed_bundle_scope(blind_bundle: Path) -> Path:
    return blind_bundle.parent if blind_bundle.name == PUBLIC_DIR_NAME else blind_bundle


def _paths_overlap(left: Path, right: Path) -> bool:
    return (
        left == right
        or left in right.parents
        or right in left.parents
    )


def _require_output_disjoint(
    output: Path, protected: dict[str, Path], *, context: str
) -> None:
    for label, path in protected.items():
        if _paths_overlap(output, path):
            raise RuntimeError(f"{context} overlaps protected {label}")


def _require_distinct_inputs(paths: tuple[Path, ...]) -> None:
    if len(set(paths)) != len(paths):
        raise RuntimeError("review A, review B, and adjudication inputs must be distinct")


def _atomic_json_file_noreplace(payload: dict[str, Any], target: Path) -> None:
    if os.path.lexists(target):
        raise FileExistsError(errno.EEXIST, "refusing to overwrite locked file", target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{target.name}.staging-", dir=target.parent
    ) as temporary:
        source = Path(temporary) / target.name
        atomic_json_dump(payload, source)
        try:
            os.link(source, target, follow_symlinks=False)
        except FileExistsError:
            raise FileExistsError(
                errno.EEXIST, "refusing to overwrite locked file", target
            ) from None


def _public_context(blind_bundle: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    root = _canonical_plain_directory(blind_bundle, "blind bundle")
    if root.name == PUBLIC_DIR_NAME:
        public = root
    else:
        public = root / PUBLIC_DIR_NAME
    manifest, _ = validate_public_pack(public)
    rubric = _read_self_hashed_json(public / RUBRIC_NAME, "rubric_identity_sha256")
    return public, manifest, rubric


def _validate_review_payload(
    payload: dict[str, Any],
    *,
    role: str,
    public_manifest: dict[str, Any],
    rubric: dict[str, Any],
    allow_missing_hash: bool,
) -> tuple[dict[str, Any], datetime]:
    expected_keys = {
        "schema_version",
        "annotation_schema",
        "role",
        "blind_pack_manifest_identity_sha256",
        "rubric_identity_sha256",
        "status",
        "reviewer",
        "rows",
        "annotation_identity_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise RuntimeError("independent review schema changed")
    if role not in {"reviewer_A", "reviewer_B"} or payload.get("role") != role:
        raise RuntimeError("independent reviewer role changed")
    if (
        payload.get("schema_version") != 1
        or payload.get("annotation_schema") != REVIEW_SCHEMA
        or payload.get("blind_pack_manifest_identity_sha256")
        != public_manifest["identity_sha256"]
        or payload.get("rubric_identity_sha256")
        != rubric["rubric_identity_sha256"]
        or payload.get("status") != REVIEW_STATUS
    ):
        raise RuntimeError("independent review identity/status binding changed")
    reviewer = payload.get("reviewer")
    reviewer_keys = {
        "reviewer_id",
        "started_at_utc",
        "completed_at_utc",
        "evidence_and_lineage_unseen",
        "other_review_unseen_before_completion",
        "declaration",
    }
    if not isinstance(reviewer, dict) or set(reviewer) != reviewer_keys:
        raise RuntimeError("independent reviewer declaration schema changed")
    _identifier(reviewer["reviewer_id"], f"{role}.reviewer_id")
    started = _parse_utc(reviewer["started_at_utc"], f"{role}.started_at_utc")
    completed = _parse_utc(
        reviewer["completed_at_utc"], f"{role}.completed_at_utc"
    )
    if completed < started:
        raise RuntimeError(f"{role} completed before it started")
    if (
        reviewer["evidence_and_lineage_unseen"] is not True
        or reviewer["other_review_unseen_before_completion"] is not True
        or reviewer["declaration"] != REVIEW_DECLARATION
    ):
        raise RuntimeError(f"{role} did not make the exact blind-review declarations")
    rows = payload.get("rows")
    blind_ids = tuple(public_manifest["blind_ids"])
    if not isinstance(rows, list) or len(rows) != TOTAL_POOL_BRANCHES:
        raise RuntimeError("independent review row count changed")
    if tuple(row.get("blind_id") for row in rows if isinstance(row, dict)) != blind_ids:
        raise RuntimeError("independent review rows are not in frozen blind order")
    for index, row in enumerate(rows):
        validate_completed_annotation_row(row, context=f"{role}/row-{index}")
    identity = payload.get("annotation_identity_sha256")
    expected_identity = _canonical_self_hash(payload, "annotation_identity_sha256")
    if identity is None and allow_missing_hash:
        pass
    elif identity != expected_identity:
        raise RuntimeError(f"{role} annotation self-hash changed")
    return payload, completed


def seal_review(
    blind_bundle: Path,
    source: Path,
    target: Path,
    *,
    role: str,
) -> dict[str, Any]:
    blind_bundle = _canonical_plain_directory(blind_bundle, "blind bundle")
    source = _canonical_plain_file(source, "review input")
    target = _canonical_new_output(target, "sealed-review output")
    _require_output_disjoint(
        target,
        {
            "blind bundle": _sealed_bundle_scope(blind_bundle),
            "review input": source,
        },
        context="sealed-review output",
    )
    _, public_manifest, rubric = _public_context(blind_bundle)
    payload = load_json(source)
    _validate_review_payload(
        payload,
        role=role,
        public_manifest=public_manifest,
        rubric=rubric,
        allow_missing_hash=True,
    )
    payload = json.loads(json.dumps(payload, ensure_ascii=False))
    payload["annotation_identity_sha256"] = _canonical_self_hash(
        payload, "annotation_identity_sha256"
    )
    _validate_review_payload(
        payload,
        role=role,
        public_manifest=public_manifest,
        rubric=rubric,
        allow_missing_hash=False,
    )
    _atomic_json_file_noreplace(payload, target)
    if load_json(target) != payload:
        raise RuntimeError("sealed review did not round-trip exactly")
    return payload


def _structured_row(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row[field] for field in ANNOTATION_FIELDS if field != "notes"}


def _row_needs_adjudication(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if _structured_row(left) != _structured_row(right):
        return True
    uncertain_values = {
        "uncertain",
        "uncertain_or_not_scorable",
    }
    return any(value in uncertain_values for value in _structured_row(left).values())


def _validate_adjudication(
    payload: dict[str, Any],
    *,
    review_a: dict[str, Any],
    completed_a: datetime,
    review_b: dict[str, Any],
    completed_b: datetime,
    public_manifest: dict[str, Any],
    rubric: dict[str, Any],
    allow_missing_hash: bool,
) -> tuple[dict[str, Any], datetime]:
    expected_keys = {
        "schema_version",
        "annotation_schema",
        "role",
        "blind_pack_manifest_identity_sha256",
        "rubric_identity_sha256",
        "status",
        "review_A_annotation_identity_sha256",
        "review_B_annotation_identity_sha256",
        "adjudicator",
        "rows",
        "annotation_identity_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise RuntimeError("consensus adjudication schema changed")
    if (
        payload.get("schema_version") != 1
        or payload.get("annotation_schema") != CONSENSUS_SCHEMA
        or payload.get("role") != "consensus_adjudication"
        or payload.get("blind_pack_manifest_identity_sha256")
        != public_manifest["identity_sha256"]
        or payload.get("rubric_identity_sha256")
        != rubric["rubric_identity_sha256"]
        or payload.get("status") != CONSENSUS_STATUS
        or payload.get("review_A_annotation_identity_sha256")
        != review_a["annotation_identity_sha256"]
        or payload.get("review_B_annotation_identity_sha256")
        != review_b["annotation_identity_sha256"]
    ):
        raise RuntimeError("consensus/review/public binding changed")
    adjudicator = payload.get("adjudicator")
    adjudicator_keys = {
        "adjudicator_id",
        "started_at_utc",
        "completed_at_utc",
        "evidence_and_lineage_unseen",
        "inspected_every_disagreement_and_uncertain_call",
        "declaration",
    }
    if not isinstance(adjudicator, dict) or set(adjudicator) != adjudicator_keys:
        raise RuntimeError("adjudicator declaration schema changed")
    adjudicator_id = _identifier(adjudicator["adjudicator_id"], "adjudicator_id")
    reviewer_ids = {
        review_a["reviewer"]["reviewer_id"],
        review_b["reviewer"]["reviewer_id"],
    }
    if len(reviewer_ids) != 2 or adjudicator_id in reviewer_ids:
        raise RuntimeError("reviewers and adjudicator must be three distinct identities")
    started = _parse_utc(adjudicator["started_at_utc"], "adjudicator.started_at_utc")
    completed = _parse_utc(
        adjudicator["completed_at_utc"], "adjudicator.completed_at_utc"
    )
    if started < max(completed_a, completed_b) or completed < started:
        raise RuntimeError("adjudication must start after both independent reviews lock")
    if (
        adjudicator["evidence_and_lineage_unseen"] is not True
        or adjudicator["inspected_every_disagreement_and_uncertain_call"] is not True
        or adjudicator["declaration"] != ADJUDICATION_DECLARATION
    ):
        raise RuntimeError("adjudicator did not make the exact blind declarations")

    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != TOTAL_POOL_BRANCHES:
        raise RuntimeError("consensus row count changed")
    expected_ids = tuple(public_manifest["blind_ids"])
    if tuple(row.get("blind_id") for row in rows if isinstance(row, dict)) != expected_ids:
        raise RuntimeError("consensus rows are not in frozen blind order")
    for index, (row, left, right) in enumerate(
        zip(rows, review_a["rows"], review_b["rows"])
    ):
        if not isinstance(row, dict) or set(row) != {
            "blind_id",
            *ANNOTATION_FIELDS,
            "adjudication_reason",
        }:
            raise RuntimeError(f"consensus row {index} schema changed")
        quality_row = {key: value for key, value in row.items() if key != "adjudication_reason"}
        validate_completed_annotation_row(quality_row, context=f"consensus/row-{index}")
        reason = row["adjudication_reason"]
        if not isinstance(reason, str) or len(reason) > 2_000:
            raise RuntimeError("adjudication reason must be visible-only text <=2000 chars")
        if _VISIBLE_REASON_FORBIDDEN.search(reason):
            raise RuntimeError("adjudication reason mentions forbidden information")
        needs = _row_needs_adjudication(left, right)
        if needs and not reason.strip():
            raise RuntimeError(
                f"{row['blind_id']}: disagreement/uncertain call lacks adjudication reason"
            )
        if not needs and _structured_row(quality_row) != _structured_row(left):
            raise RuntimeError(
                f"{row['blind_id']}: unanimous non-uncertain structured labels changed"
            )
    identity = payload.get("annotation_identity_sha256")
    expected_identity = _canonical_self_hash(payload, "annotation_identity_sha256")
    if identity is None and allow_missing_hash:
        pass
    elif identity != expected_identity:
        raise RuntimeError("consensus annotation self-hash changed")
    return payload, completed


def _copy_bytes_noreplace(source: Path, destination: Path) -> None:
    if os.path.lexists(destination):
        raise RuntimeError("refusing to overwrite locked annotation copy")
    destination.write_bytes(source.read_bytes())


def lock_consensus(
    blind_bundle: Path,
    review_a_path: Path,
    review_b_path: Path,
    adjudication_path: Path,
    outdir: Path,
) -> None:
    blind_bundle = _canonical_plain_directory(blind_bundle, "blind bundle")
    review_a_path = _canonical_plain_file(review_a_path, "sealed review A")
    review_b_path = _canonical_plain_file(review_b_path, "sealed review B")
    adjudication_path = _canonical_plain_file(
        adjudication_path, "consensus adjudication input"
    )
    _require_distinct_inputs((review_a_path, review_b_path, adjudication_path))
    outdir = _canonical_new_output(outdir, "consensus-lock output")
    _require_output_disjoint(
        outdir,
        {
            "blind bundle": _sealed_bundle_scope(blind_bundle),
            "sealed review A": review_a_path,
            "sealed review B": review_b_path,
            "adjudication input": adjudication_path,
        },
        context="consensus-lock output",
    )
    _, public_manifest, rubric = _public_context(blind_bundle)
    review_a = _read_self_hashed_json(review_a_path, "annotation_identity_sha256")
    review_b = _read_self_hashed_json(review_b_path, "annotation_identity_sha256")
    review_a, completed_a = _validate_review_payload(
        review_a,
        role="reviewer_A",
        public_manifest=public_manifest,
        rubric=rubric,
        allow_missing_hash=False,
    )
    review_b, completed_b = _validate_review_payload(
        review_b,
        role="reviewer_B",
        public_manifest=public_manifest,
        rubric=rubric,
        allow_missing_hash=False,
    )
    raw_consensus = load_json(adjudication_path)
    _validate_adjudication(
        raw_consensus,
        review_a=review_a,
        completed_a=completed_a,
        review_b=review_b,
        completed_b=completed_b,
        public_manifest=public_manifest,
        rubric=rubric,
        allow_missing_hash=True,
    )
    consensus = json.loads(json.dumps(raw_consensus, ensure_ascii=False))
    consensus["annotation_identity_sha256"] = _canonical_self_hash(
        consensus, "annotation_identity_sha256"
    )
    _, completed_consensus = _validate_adjudication(
        consensus,
        review_a=review_a,
        completed_a=completed_a,
        review_b=review_b,
        completed_b=completed_b,
        public_manifest=public_manifest,
        rubric=rubric,
        allow_missing_hash=False,
    )
    outdir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{outdir.name}.staging-", dir=outdir.parent
    ) as temporary:
        staging = Path(temporary) / "lock"
        staging.mkdir()
        _copy_bytes_noreplace(review_a_path, staging / REVIEW_A_COPY_NAME)
        _copy_bytes_noreplace(review_b_path, staging / REVIEW_B_COPY_NAME)
        atomic_json_dump(consensus, staging / CONSENSUS_NAME)
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "status": CONSENSUS_STATUS,
            "blind_pack_manifest_identity_sha256": public_manifest[
                "identity_sha256"
            ],
            "rubric_identity_sha256": rubric["rubric_identity_sha256"],
            "review_A": {
                "annotation_identity_sha256": review_a[
                    "annotation_identity_sha256"
                ],
                "file_sha256": sha256_file(staging / REVIEW_A_COPY_NAME),
            },
            "review_B": {
                "annotation_identity_sha256": review_b[
                    "annotation_identity_sha256"
                ],
                "file_sha256": sha256_file(staging / REVIEW_B_COPY_NAME),
            },
            "consensus": {
                "annotation_identity_sha256": consensus[
                    "annotation_identity_sha256"
                ],
                "file_sha256": sha256_file(staging / CONSENSUS_NAME),
                "locked_at_utc": completed_consensus.isoformat().replace(
                    "+00:00", "Z"
                ),
                "row_count": TOTAL_POOL_BRANCHES,
            },
            "locker": {"filename": RUNNER.name, "sha256": sha256_file(RUNNER)},
        }
        manifest["identity_sha256"] = _canonical_self_hash(
            manifest, "identity_sha256"
        )
        atomic_json_dump(manifest, staging / MANIFEST_NAME)
        completion: dict[str, Any] = {
            "complete": True,
            "manifest_identity_sha256": manifest["identity_sha256"],
            "manifest_file_sha256": sha256_file(staging / MANIFEST_NAME),
            "consensus_annotation_identity_sha256": consensus[
                "annotation_identity_sha256"
            ],
            "consensus_file_sha256": sha256_file(staging / CONSENSUS_NAME),
        }
        completion["payload_sha256"] = _canonical_self_hash(
            completion, "payload_sha256"
        )
        atomic_json_dump(completion, staging / COMPLETION_NAME)
        validate_consensus_lock(staging, blind_bundle)
        _atomic_install_directory_noreplace(staging, outdir)
    validate_consensus_lock(outdir, blind_bundle)


def validate_consensus_lock(
    root: Path, blind_bundle: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = _canonical_plain_directory(root, "consensus lock")
    blind_bundle = _canonical_plain_directory(blind_bundle, "blind bundle")
    if _paths_overlap(root, _sealed_bundle_scope(blind_bundle)):
        raise RuntimeError("consensus lock must be outside the sealed blind bundle")
    _reject_special_entries(root)
    _, public_manifest, rubric = _public_context(blind_bundle)
    manifest = _read_self_hashed_json(root / MANIFEST_NAME, "identity_sha256")
    review_a = _read_self_hashed_json(
        root / REVIEW_A_COPY_NAME, "annotation_identity_sha256"
    )
    review_b = _read_self_hashed_json(
        root / REVIEW_B_COPY_NAME, "annotation_identity_sha256"
    )
    consensus = _read_self_hashed_json(
        root / CONSENSUS_NAME, "annotation_identity_sha256"
    )
    review_a, completed_a = _validate_review_payload(
        review_a,
        role="reviewer_A",
        public_manifest=public_manifest,
        rubric=rubric,
        allow_missing_hash=False,
    )
    review_b, completed_b = _validate_review_payload(
        review_b,
        role="reviewer_B",
        public_manifest=public_manifest,
        rubric=rubric,
        allow_missing_hash=False,
    )
    _, completed_consensus = _validate_adjudication(
        consensus,
        review_a=review_a,
        completed_a=completed_a,
        review_b=review_b,
        completed_b=completed_b,
        public_manifest=public_manifest,
        rubric=rubric,
        allow_missing_hash=False,
    )
    expected_manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "status": CONSENSUS_STATUS,
        "blind_pack_manifest_identity_sha256": public_manifest["identity_sha256"],
        "rubric_identity_sha256": rubric["rubric_identity_sha256"],
        "review_A": {
            "annotation_identity_sha256": review_a["annotation_identity_sha256"],
            "file_sha256": sha256_file(root / REVIEW_A_COPY_NAME),
        },
        "review_B": {
            "annotation_identity_sha256": review_b["annotation_identity_sha256"],
            "file_sha256": sha256_file(root / REVIEW_B_COPY_NAME),
        },
        "consensus": {
            "annotation_identity_sha256": consensus["annotation_identity_sha256"],
            "file_sha256": sha256_file(root / CONSENSUS_NAME),
            "locked_at_utc": completed_consensus.isoformat().replace("+00:00", "Z"),
            "row_count": TOTAL_POOL_BRANCHES,
        },
        "locker": {"filename": RUNNER.name, "sha256": sha256_file(RUNNER)},
        "identity_sha256": manifest.get("identity_sha256"),
    }
    if manifest != expected_manifest:
        raise RuntimeError("consensus-lock manifest binding changed")
    completion = _read_self_hashed_json(root / COMPLETION_NAME, "payload_sha256")
    expected_completion = {
        "complete": True,
        "manifest_identity_sha256": manifest["identity_sha256"],
        "manifest_file_sha256": sha256_file(root / MANIFEST_NAME),
        "consensus_annotation_identity_sha256": consensus[
            "annotation_identity_sha256"
        ],
        "consensus_file_sha256": sha256_file(root / CONSENSUS_NAME),
        "payload_sha256": completion.get("payload_sha256"),
    }
    if completion != expected_completion:
        raise RuntimeError("consensus-lock completion binding changed")
    expected_files = {
        (root / REVIEW_A_COPY_NAME).resolve(),
        (root / REVIEW_B_COPY_NAME).resolve(),
        (root / CONSENSUS_NAME).resolve(),
        (root / MANIFEST_NAME).resolve(),
        (root / COMPLETION_NAME).resolve(),
    }
    actual_files = {path.resolve() for path in root.rglob("*") if path.is_file()}
    actual_dirs = {path.resolve() for path in root.rglob("*") if path.is_dir()}
    if actual_files != expected_files or actual_dirs:
        raise RuntimeError("consensus lock is not a closed flat file set")
    return manifest, completion, consensus


def _synthetic_public_context() -> tuple[dict[str, Any], dict[str, Any]]:
    ids = [f"xr1_{index:016x}" for index in range(TOTAL_POOL_BRANCHES)]
    manifest = {"identity_sha256": "a" * 64, "blind_ids": ids}
    rubric = {"rubric_identity_sha256": "b" * 64}
    return manifest, rubric


def _synthetic_row(identifier: str, *, bad: bool = False) -> dict[str, Any]:
    return {
        "blind_id": identifier,
        "primary_overall_structural_quality": (
            "clear_overall_structural_bad"
            if bad
            else "not_clear_overall_structural_bad"
        ),
        "secondary_hind_limb_topology": "not_clear_failure",
        "tail_R_root_attachment": 0,
        "tail_T_taper_and_volume": 0,
        "tail_F_feather_or_hair_flow": 0,
        "tail_D_distal_tip": 0,
        "tail_P_paddle_like": 0,
        "tail_B_short_or_blunt": 0,
        "tail_S_abrupt_filament_transition": 0,
        "tail_identity": "clear",
        "tail_scorable": "yes",
        "tail_confidence": "high",
        "tail_derived_label": "natural",
        "notes": "",
    }


def _synthetic_review(
    role: str,
    reviewer_id: str,
    public: dict[str, Any],
    rubric: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "annotation_schema": REVIEW_SCHEMA,
        "role": role,
        "blind_pack_manifest_identity_sha256": public["identity_sha256"],
        "rubric_identity_sha256": rubric["rubric_identity_sha256"],
        "status": REVIEW_STATUS,
        "reviewer": {
            "reviewer_id": reviewer_id,
            "started_at_utc": "2026-08-27T01:00:00Z",
            "completed_at_utc": "2026-08-27T02:00:00Z",
            "evidence_and_lineage_unseen": True,
            "other_review_unseen_before_completion": True,
            "declaration": REVIEW_DECLARATION,
        },
        "rows": [_synthetic_row(identifier) for identifier in public["blind_ids"]],
        "annotation_identity_sha256": None,
    }
    payload["annotation_identity_sha256"] = _canonical_self_hash(
        payload, "annotation_identity_sha256"
    )
    return payload


def _expect_failure(operation: Callable[[], Any], label: str) -> None:
    try:
        operation()
    except RuntimeError:
        return
    raise AssertionError(f"negative self-test did not fail: {label}")


def run_self_test() -> None:
    public, rubric = _synthetic_public_context()
    review_a = _synthetic_review("reviewer_A", "reviewer-a", public, rubric)
    review_b = _synthetic_review("reviewer_B", "reviewer-b", public, rubric)
    _, completed_a = _validate_review_payload(
        review_a,
        role="reviewer_A",
        public_manifest=public,
        rubric=rubric,
        allow_missing_hash=False,
    )
    _, completed_b = _validate_review_payload(
        review_b,
        role="reviewer_B",
        public_manifest=public,
        rubric=rubric,
        allow_missing_hash=False,
    )
    consensus = {
        "schema_version": 1,
        "annotation_schema": CONSENSUS_SCHEMA,
        "role": "consensus_adjudication",
        "blind_pack_manifest_identity_sha256": public["identity_sha256"],
        "rubric_identity_sha256": rubric["rubric_identity_sha256"],
        "status": CONSENSUS_STATUS,
        "review_A_annotation_identity_sha256": review_a[
            "annotation_identity_sha256"
        ],
        "review_B_annotation_identity_sha256": review_b[
            "annotation_identity_sha256"
        ],
        "adjudicator": {
            "adjudicator_id": "adjudicator",
            "started_at_utc": "2026-08-27T03:00:00Z",
            "completed_at_utc": "2026-08-27T04:00:00Z",
            "evidence_and_lineage_unseen": True,
            "inspected_every_disagreement_and_uncertain_call": True,
            "declaration": ADJUDICATION_DECLARATION,
        },
        "rows": [
            {**_synthetic_row(identifier), "adjudication_reason": ""}
            for identifier in public["blind_ids"]
        ],
        "annotation_identity_sha256": None,
    }
    consensus["annotation_identity_sha256"] = _canonical_self_hash(
        consensus, "annotation_identity_sha256"
    )
    _validate_adjudication(
        consensus,
        review_a=review_a,
        completed_a=completed_a,
        review_b=review_b,
        completed_b=completed_b,
        public_manifest=public,
        rubric=rubric,
        allow_missing_hash=False,
    )
    mutated_b = json.loads(json.dumps(review_b))
    mutated_b["rows"][0]["primary_overall_structural_quality"] = (
        "clear_overall_structural_bad"
    )
    mutated_b["annotation_identity_sha256"] = _canonical_self_hash(
        mutated_b, "annotation_identity_sha256"
    )
    bad_consensus = json.loads(json.dumps(consensus))
    bad_consensus["review_B_annotation_identity_sha256"] = mutated_b[
        "annotation_identity_sha256"
    ]
    bad_consensus["annotation_identity_sha256"] = _canonical_self_hash(
        bad_consensus, "annotation_identity_sha256"
    )
    _expect_failure(
        lambda: _validate_adjudication(
            bad_consensus,
            review_a=review_a,
            completed_a=completed_a,
            review_b=mutated_b,
            completed_b=completed_b,
            public_manifest=public,
            rubric=rubric,
            allow_missing_hash=False,
        ),
        "unresolved disagreement",
    )
    leaked = json.loads(json.dumps(consensus))
    leaked["rows"][0]["adjudication_reason"] = "evidence rank was high"
    leaked["annotation_identity_sha256"] = _canonical_self_hash(
        leaked, "annotation_identity_sha256"
    )
    _expect_failure(
        lambda: _validate_adjudication(
            leaked,
            review_a=review_a,
            completed_a=completed_a,
            review_b=review_b,
            completed_b=completed_b,
            public_manifest=public,
            rubric=rubric,
            allow_missing_hash=False,
        ),
        "adjudication lineage/evidence leakage",
    )
    with tempfile.TemporaryDirectory(prefix="cross-prefix-lock-path-selftest-") as temporary:
        root = Path(temporary)
        bundle = root / "blind_bundle"
        bundle.mkdir()
        review_a_path = root / "review_a.json"
        review_b_path = root / "review_b.json"
        adjudication_path = root / "adjudication.json"
        for path in (review_a_path, review_b_path, adjudication_path):
            path.write_text("{}\n", encoding="utf-8")
        canonical_inputs = tuple(
            _canonical_plain_file(path, "synthetic annotation input")
            for path in (review_a_path, review_b_path, adjudication_path)
        )
        _require_distinct_inputs(canonical_inputs)
        safe_output = _canonical_new_output(
            root / "locks" / "consensus", "synthetic consensus output"
        )
        _require_output_disjoint(
            safe_output,
            {
                "blind bundle": bundle,
                "review A": canonical_inputs[0],
                "review B": canonical_inputs[1],
                "adjudication": canonical_inputs[2],
            },
            context="synthetic consensus output",
        )
        _expect_failure(
            lambda: _require_distinct_inputs(
                (canonical_inputs[0], canonical_inputs[0], canonical_inputs[2])
            ),
            "duplicate review/consensus input",
        )
        nested_output = _canonical_new_output(
            bundle / "poisoned_lock", "synthetic nested output"
        )
        _expect_failure(
            lambda: _require_output_disjoint(
                nested_output,
                {"blind bundle": bundle},
                context="synthetic nested output",
            ),
            "output inside sealed blind bundle",
        )
        _expect_failure(
            lambda: _require_output_disjoint(
                root / "future_parent",
                {"review input": root / "future_parent" / "review.json"},
                context="synthetic covering output",
            ),
            "output covering an input",
        )
        real_parent = root / "real_parent"
        real_parent.mkdir()
        alias_parent = root / "alias_parent"
        alias_parent.symlink_to(real_parent, target_is_directory=True)
        _expect_failure(
            lambda: _canonical_new_output(
                alias_parent / "sealed.json", "symlink-parent output"
            ),
            "output through symlinked parent",
        )
        input_alias = root / "review_alias.json"
        input_alias.symlink_to(review_a_path)
        _expect_failure(
            lambda: _canonical_plain_file(input_alias, "symlinked review input"),
            "symlinked annotation input",
        )
    print(
        "self-test passed: two distinct blind reviews, consensus timing/binding, "
        "disagreement coverage, mutation failure, visible-only adjudication notes, "
        "canonical non-symlink paths, and output/input/bundle disjointness"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    seal = subparsers.add_parser("seal-review")
    seal.add_argument("--blind-bundle", type=Path, required=True)
    seal.add_argument("--role", choices=("reviewer_A", "reviewer_B"), required=True)
    seal.add_argument("--input", type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)
    lock = subparsers.add_parser("lock-consensus")
    lock.add_argument("--blind-bundle", type=Path, required=True)
    lock.add_argument("--review-a", type=Path, required=True)
    lock.add_argument("--review-b", type=Path, required=True)
    lock.add_argument("--adjudication", type=Path, required=True)
    lock.add_argument("--outdir", type=Path, required=True)
    subparsers.add_parser("self-test")
    return parser


def _plain_existing(path: Path, parser: argparse.ArgumentParser, label: str) -> Path:
    try:
        return _canonical_plain_file(path, label)
    except RuntimeError as exc:
        parser.error(str(exc))


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "self-test":
        run_self_test()
        return 0
    try:
        bundle = _canonical_plain_directory(args.blind_bundle, "--blind-bundle")
    except RuntimeError as exc:
        parser.error(str(exc))
    if args.command == "seal-review":
        source = _plain_existing(args.input, parser, "--input")
        try:
            target = _canonical_new_output(args.output, "--output")
            _require_output_disjoint(
                target,
                {
                    "blind bundle": _sealed_bundle_scope(bundle),
                    "review input": source,
                },
                context="--output",
            )
        except RuntimeError as exc:
            parser.error(str(exc))
        seal_review(bundle, source, target, role=args.role)
        print(json.dumps({"status": "sealed", "role": args.role}, sort_keys=True))
        return 0
    review_a = _plain_existing(args.review_a, parser, "--review-a")
    review_b = _plain_existing(args.review_b, parser, "--review-b")
    adjudication = _plain_existing(args.adjudication, parser, "--adjudication")
    try:
        outdir = _canonical_new_output(args.outdir, "--outdir")
        _require_distinct_inputs((review_a, review_b, adjudication))
        _require_output_disjoint(
            outdir,
            {
                "blind bundle": _sealed_bundle_scope(bundle),
                "sealed review A": review_a,
                "sealed review B": review_b,
                "adjudication input": adjudication,
            },
            context="--outdir",
        )
    except RuntimeError as exc:
        parser.error(str(exc))
    lock_consensus(bundle, review_a, review_b, adjudication, outdir)
    print(json.dumps({"status": "consensus_locked", "rows": 64}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
