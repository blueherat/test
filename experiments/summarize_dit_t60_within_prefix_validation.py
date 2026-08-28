#!/usr/bin/env python3
"""One-time blinded readout for the frozen DiT x60 validation pool.

The order of operations in this program is part of the protocol.  It first
loads only the frozen protocol and the consensus annotation, requires a
complete/self-hashed 32-row annotation plus evidence-unseen declarations from
every reviewer and the consensus adjudicator, and writes the exact annotation
bytes into a staged lock.  Only the resulting in-memory gate token can call the
four-shard validator or open ``trace_private.npz``.

The readout is deliberately narrow:

* primary: the preregistered +theta/tile_12/log(5) ever-alarm 2x2 table,
  TPR, FPR, TPR-FPR, two-sided 95% exact binomial intervals, a one-sided
  Fisher test in the preregistered direction, and the total alarm budget;
* secondary: ROC AUC and tie-aware descending-rank summaries for only the
  frozen 34-path +/- mixture terminal and running-maximum scores, against the
  non-uncertain primary endpoint;
* overall structure: the preregistered descriptive alarm cross-tab for clear
  structural bad versus not-clear structural bad, with uncertain excluded;
* tail: separate descriptive distributions for R/T/F/D/P/B/S.  No composite
  tail score and no alarm association are computed.

There is no option for scanning a tile, sign, Delta-nu, K cap, threshold,
endpoint, subset, or individual component.  No image, trace, per-branch score,
or per-branch alarm is copied into the output.  The result is an atomically
installed, no-overwrite, self-hashed closed bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

sys.dont_write_bytecode = True

import numpy as np
from scipy.stats import beta, fisher_exact, rankdata

try:
    from .intervene_dit_imagenet256_suffix import _atomic_install_directory_noreplace
    from .reproduce_dit_imagenet256 import atomic_json_dump, sha256_file, sha256_json
    from .run_dit_t60_within_prefix_validation_pool import (
        ALARM_LOG_E,
        ALPHA_E,
        BRANCHES_PER_SHARD,
        DELTA_NU,
        EXPERIMENT as SHARD_EXPERIMENT,
        PRIMARY_TILE_INDEX,
        PROTOCOL_COPY_NAME,
        SIGNED_COMPONENT_COUNT,
        TOTAL_K_PER_COMPONENT,
        TOTAL_POOL_BRANCHES,
        TOTAL_SHARDS,
        TRACE_NAME,
        _load_protocol,
        blind_id,
        validate_output_bundle as validate_shard_bundle,
    )
except ImportError:  # pragma: no cover - direct CLI execution.
    from intervene_dit_imagenet256_suffix import _atomic_install_directory_noreplace
    from reproduce_dit_imagenet256 import atomic_json_dump, sha256_file, sha256_json
    from run_dit_t60_within_prefix_validation_pool import (
        ALARM_LOG_E,
        ALPHA_E,
        BRANCHES_PER_SHARD,
        DELTA_NU,
        EXPERIMENT as SHARD_EXPERIMENT,
        PRIMARY_TILE_INDEX,
        PROTOCOL_COPY_NAME,
        SIGNED_COMPONENT_COUNT,
        TOTAL_K_PER_COMPONENT,
        TOTAL_POOL_BRANCHES,
        TOTAL_SHARDS,
        TRACE_NAME,
        _load_protocol,
        blind_id,
        validate_output_bundle as validate_shard_bundle,
    )


EXPERIMENT = "dit_imagenet256_t60_within_prefix_validation_closed_summary"
SCHEMA_VERSION = 1
ANNOTATION_SCHEMA = "dit_imagenet256_t60_within_prefix_consensus_annotation_v1"
ANNOTATION_STATUS = "LOCKED_COMPLETE_BEFORE_EVIDENCE_UNSEAL"
ANNOTATION_HASH_KEY = "annotation_identity_sha256"
LOCKED_ANNOTATION_NAME = "annotation_locked.json"
MANIFEST_NAME = "manifest.json"
SUMMARY_NAME = "summary.json"
COMPLETION_NAME = "completion.json"
CONFIDENCE_LEVEL = 0.95

EVIDENCE_UNSEEN_DECLARATION = (
    "I declare that no reviewer or consensus adjudicator saw any path-evidence "
    "value, alarm flag, component score, private trace, shard result metadata, "
    "branch index, or evidence-derived ranking before this complete 32-row "
    "consensus annotation was locked."
)

PRIMARY_LABELS = {"clear_failure", "not_clear_failure", "uncertain"}
OVERALL_LABELS = {
    "clear_structural_bad",
    "not_clear_structural_bad",
    "uncertain",
}
TAIL_IDENTITIES = {"clear", "plausible", "unclear"}
TAIL_CONFIDENCES = {"high", "medium", "low"}
TAIL_DERIVED_LABELS = {"natural", "odd", "malformed", "uncertain"}
TAIL_SCORABLE_VALUES = {"yes", "no"}
TERNARY_TAIL_FIELDS = (
    "tail_R_root_attachment",
    "tail_T_taper_and_volume",
    "tail_F_feather_or_hair_flow",
    "tail_D_distal_tip",
)
BINARY_TAIL_FIELDS = (
    "tail_P_paddle_like",
    "tail_B_short_blunt",
    "tail_S_abrupt_filament_transition",
)
TAIL_REPORT_ORDER = (
    "tail_R_root_attachment",
    "tail_T_taper_and_volume",
    "tail_F_feather_or_hair_flow",
    "tail_D_distal_tip",
    "tail_P_paddle_like",
    "tail_B_short_blunt",
    "tail_S_abrupt_filament_transition",
)

ANNOTATION_TOP_LEVEL_KEYS = {
    "schema_version",
    "annotation_schema",
    "protocol_identity_sha256",
    "status",
    "reviewers",
    "consensus",
    "annotations",
    ANNOTATION_HASH_KEY,
}
REVIEWER_KEYS = {
    "reviewer_id",
    "reviewed_at_utc",
    "evidence_unseen",
    "evidence_unseen_declaration",
}
CONSENSUS_KEYS = {
    "adjudicator_id",
    "locked_at_utc",
    "method",
    "evidence_unseen",
    "evidence_unseen_declaration",
    "row_count",
}
CONSENSUS_METHODS = {
    "unanimous_independent_reviews",
    "independent_reviews_then_adjudication",
    "single_reviewer_consensus",
}
ANNOTATION_ROW_KEYS = {
    "blind_id",
    "notes",
    "overall_structural_secondary",
    "primary_hind_limb_topology",
    "tail_B_short_blunt",
    "tail_D_distal_tip",
    "tail_F_feather_or_hair_flow",
    "tail_P_paddle_like",
    "tail_R_root_attachment",
    "tail_S_abrupt_filament_transition",
    "tail_T_taper_and_volume",
    "tail_confidence",
    "tail_derived_label",
    "tail_identity",
    "tail_scorable",
}
SUMMARY_TOP_LEVEL_KEYS = {
    "schema_version",
    "experiment",
    "scope",
    "annotation_identity_sha256",
    "primary",
    "secondary_overall_structural_descriptive",
    "secondary_frozen_34_path_mixture_only",
    "tail_R_T_F_D_P_B_S_separate",
    "multiplicity_guard",
    "payload_sha256",
}
PRIMARY_OUTPUT_KEYS = {
    "candidate",
    "cross_tab_ever_alarm_by_primary_endpoint",
    "TPR",
    "FPR",
    "TPR_minus_FPR",
    "fisher_exact_one_sided",
    "total_alarms",
    "predeclared_event_count",
}
SECONDARY_OUTPUT_KEYS = {
    "frozen_e_process",
    "terminal_fixed_34_path_mixture_log_e",
    "running_max_fixed_34_path_mixture_log_e",
}
OVERALL_OUTPUT_KEYS = {
    "role",
    "cross_tab_ever_alarm_by_overall_structural_endpoint",
    "alarm_rate_clear_structural_bad",
    "alarm_rate_not_clear_structural_bad",
    "alarm_rate_difference",
    "inferential_test_performed",
}

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_BLIND_ID_RE = re.compile(r"^vp1_[0-9a-f]{12}$")
_EVIDENCE_NOTE_RE = re.compile(
    r"(?:log[ _-]?e|e[ _-]?value|alarm|evidence|score|rank|tile[ _-]?12|"
    r"theta|delta[ _-]?nu|likelihood|martingale|threshold|\bK\s*=)",
    flags=re.IGNORECASE,
)

_GATE_NONCE = object()


@dataclass(frozen=True)
class _AnnotationLockToken:
    """Unforgeable-in-module capability required before any shard access."""

    _nonce: object
    payload: dict[str, Any]
    source_file_sha256: str
    copied_file_sha256: str
    copied_path: Path


@dataclass(frozen=True)
class _BranchReadout:
    blind_id: str
    primary_label: str
    overall_label: str
    primary_alarm: int
    secondary_terminal: float
    secondary_running_max: float


def _canonical_self_hash(payload: dict[str, Any], key: str) -> str:
    stripped = dict(payload)
    stripped.pop(key, None)
    return sha256_json(stripped)


def _validate_frozen_secondary_protocol(protocol: dict[str, Any]) -> None:
    """Bind this readout to the exact preregistered secondary evidence object."""

    expected = {
        "aggregation": "uniform fixed mixture over complete path likelihood ratios",
        "components": (
            "global plus 16 row-major 4x4 latent tiles, each with +theta and -theta"
        ),
        "delta_nu": DELTA_NU,
        "forbidden_aggregation": "posthoc maximum over components",
        "saved_values": [
            "terminal fixed-mixture log-e",
            "running maximum fixed-mixture log-e including the initial value log(1)=0",
        ],
        "total_suffix_K_cap_per_component": TOTAL_K_PER_COMPONENT,
    }
    if protocol.get("frozen_secondary_evidence") != expected:
        raise RuntimeError("protocol frozen secondary evidence definition changed")


def _require_exact_keys(value: Any, expected: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        observed = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise RuntimeError(
            f"{context} must use the exact frozen schema; observed={observed}"
        )
    return value


def _parse_utc_timestamp(value: Any, context: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise RuntimeError(f"{context} must be an RFC3339 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise RuntimeError(f"{context} is not a valid RFC3339 UTC timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise RuntimeError(f"{context} must be UTC")
    return parsed


def _require_identifier(value: Any, context: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise RuntimeError(f"{context} must match {_IDENTIFIER_RE.pattern}")
    return value


def _validate_declaration(value: dict[str, Any], context: str) -> None:
    if value["evidence_unseen"] is not True:
        raise RuntimeError(f"{context} must explicitly declare evidence_unseen=true")
    if value["evidence_unseen_declaration"] != EVIDENCE_UNSEEN_DECLARATION:
        raise RuntimeError(f"{context} evidence-unseen declaration text changed")


def _expected_tail_derived_label(row: dict[str, Any]) -> str:
    if row["tail_scorable"] == "no" or row["tail_identity"] == "unclear":
        return "uncertain"
    dimensions = [int(row[key]) for key in TERNARY_TAIL_FIELDS]
    if any(value == 2 for value in dimensions):
        return "malformed"
    if any(value == 1 for value in dimensions):
        return "odd"
    if row["tail_identity"] == "clear":
        return "natural"
    return "uncertain"  # plausible identity with all-zero defect dimensions.


def _validate_annotation_row(row: Any, *, row_index: int) -> dict[str, Any]:
    value = _require_exact_keys(row, ANNOTATION_ROW_KEYS, f"annotation row {row_index}")
    identifier = value["blind_id"]
    if not isinstance(identifier, str) or _BLIND_ID_RE.fullmatch(identifier) is None:
        raise RuntimeError(f"annotation row {row_index} has an invalid blind_id")
    if value["primary_hind_limb_topology"] not in PRIMARY_LABELS:
        raise RuntimeError(f"{identifier}: invalid primary endpoint label")
    if value["overall_structural_secondary"] not in OVERALL_LABELS:
        raise RuntimeError(f"{identifier}: invalid overall structural label")
    if value["tail_scorable"] not in TAIL_SCORABLE_VALUES:
        raise RuntimeError(f"{identifier}: invalid tail_scorable")
    if value["tail_identity"] not in TAIL_IDENTITIES:
        raise RuntimeError(f"{identifier}: invalid tail_identity")
    if value["tail_confidence"] not in TAIL_CONFIDENCES:
        raise RuntimeError(f"{identifier}: invalid tail_confidence")
    if value["tail_derived_label"] not in TAIL_DERIVED_LABELS:
        raise RuntimeError(f"{identifier}: invalid tail_derived_label")
    notes = value["notes"]
    if not isinstance(notes, str) or len(notes) > 2_000:
        raise RuntimeError(f"{identifier}: notes must be a string of at most 2000 characters")
    if _EVIDENCE_NOTE_RE.search(notes):
        raise RuntimeError(f"{identifier}: notes contain forbidden evidence-related language")

    tail_fields = TERNARY_TAIL_FIELDS + BINARY_TAIL_FIELDS
    if value["tail_scorable"] == "no":
        if any(value[key] is not None for key in tail_fields):
            raise RuntimeError(f"{identifier}: unscorable tail fields must all be null")
    else:
        for key in TERNARY_TAIL_FIELDS:
            if type(value[key]) is not int or value[key] not in (0, 1, 2):
                raise RuntimeError(f"{identifier}: {key} must be integer 0, 1, or 2")
        for key in BINARY_TAIL_FIELDS:
            if type(value[key]) is not int or value[key] not in (0, 1):
                raise RuntimeError(f"{identifier}: {key} must be integer 0 or 1")
    expected_derived = _expected_tail_derived_label(value)
    if value["tail_derived_label"] != expected_derived:
        raise RuntimeError(
            f"{identifier}: tail_derived_label disagrees with the frozen R/T/F/D rule"
        )
    return value


def validate_consensus_annotation(
    payload: Any, *, protocol: dict[str, Any]
) -> dict[str, Any]:
    """Validate the complete annotation without touching a shard or trace."""

    value = _require_exact_keys(payload, ANNOTATION_TOP_LEVEL_KEYS, "annotation")
    if type(value["schema_version"]) is not int or value["schema_version"] != SCHEMA_VERSION:
        raise RuntimeError("annotation schema_version changed")
    if value["annotation_schema"] != ANNOTATION_SCHEMA:
        raise RuntimeError("annotation_schema changed")
    if value["protocol_identity_sha256"] != protocol["protocol_identity_sha256"]:
        raise RuntimeError("annotation is not bound to the frozen protocol")
    if value["status"] != ANNOTATION_STATUS:
        raise RuntimeError("annotation is not explicitly locked before evidence unseal")
    observed_hash = value[ANNOTATION_HASH_KEY]
    if not isinstance(observed_hash, str) or len(observed_hash) != 64:
        raise RuntimeError("annotation self-hash is missing or malformed")
    if observed_hash != _canonical_self_hash(value, ANNOTATION_HASH_KEY):
        raise RuntimeError("annotation canonical self-hash failed")

    reviewers = value["reviewers"]
    if not isinstance(reviewers, list) or not reviewers:
        raise RuntimeError("annotation requires at least one named reviewer")
    reviewer_ids: list[str] = []
    review_times: list[datetime] = []
    for index, reviewer in enumerate(reviewers):
        item = _require_exact_keys(reviewer, REVIEWER_KEYS, f"reviewer {index}")
        reviewer_ids.append(_require_identifier(item["reviewer_id"], f"reviewer {index} id"))
        review_times.append(
            _parse_utc_timestamp(item["reviewed_at_utc"], f"reviewer {index} reviewed_at_utc")
        )
        _validate_declaration(item, f"reviewer {index}")
    if len(set(reviewer_ids)) != len(reviewer_ids):
        raise RuntimeError("reviewer_id values must be unique")

    consensus = _require_exact_keys(value["consensus"], CONSENSUS_KEYS, "consensus")
    _require_identifier(consensus["adjudicator_id"], "consensus adjudicator_id")
    locked_at = _parse_utc_timestamp(consensus["locked_at_utc"], "consensus locked_at_utc")
    if locked_at < max(review_times):
        raise RuntimeError("consensus lock timestamp precedes a reviewer completion timestamp")
    if consensus["method"] not in CONSENSUS_METHODS:
        raise RuntimeError("consensus method is outside the frozen enum")
    if consensus["method"] == "single_reviewer_consensus" and len(reviewers) != 1:
        raise RuntimeError("single_reviewer_consensus requires exactly one reviewer")
    if consensus["method"] != "single_reviewer_consensus" and len(reviewers) < 2:
        raise RuntimeError("independent-review consensus methods require at least two reviewers")
    if type(consensus["row_count"]) is not int or consensus["row_count"] != TOTAL_POOL_BRANCHES:
        raise RuntimeError("consensus row_count must equal 32")
    _validate_declaration(consensus, "consensus adjudicator")

    rows = value["annotations"]
    if not isinstance(rows, list) or len(rows) != TOTAL_POOL_BRANCHES:
        raise RuntimeError("annotation must contain exactly 32 rows")
    validated = [
        _validate_annotation_row(row, row_index=index) for index, row in enumerate(rows)
    ]
    identifiers = [row["blind_id"] for row in validated]
    if len(set(identifiers)) != TOTAL_POOL_BRANCHES:
        raise RuntimeError("annotation blind_id values must be unique")
    expected_ids = {blind_id(index) for index in range(TOTAL_POOL_BRANCHES)}
    if set(identifiers) != expected_ids:
        missing = sorted(expected_ids - set(identifiers))
        unexpected = sorted(set(identifiers) - expected_ids)
        raise RuntimeError(
            f"annotation does not cover the exact frozen 32 blind IDs: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return value


def _atomic_bytes_dump(value: bytes, path: Path) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite staged lock: {path}")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def lock_annotation_before_unseal(
    annotation_path: Path,
    *,
    protocol: dict[str, Any],
    staging: Path,
) -> _AnnotationLockToken:
    """Validate and stage exact bytes; this is the sole gate-token constructor."""

    if annotation_path.is_symlink() or not annotation_path.is_file():
        raise RuntimeError("annotation must be a regular non-symlink file")
    source_bytes = annotation_path.read_bytes()
    try:
        payload = json.loads(source_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("annotation is not valid UTF-8 JSON") from error
    validated = validate_consensus_annotation(payload, protocol=protocol)
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    copied_path = staging / LOCKED_ANNOTATION_NAME
    _atomic_bytes_dump(source_bytes, copied_path)
    copied_bytes = copied_path.read_bytes()
    copied_sha = hashlib.sha256(copied_bytes).hexdigest()
    if copied_bytes != source_bytes or copied_sha != source_sha:
        raise RuntimeError("staged annotation lock is not byte-identical to its source")
    copied_payload = json.loads(copied_bytes.decode("utf-8"))
    if copied_payload != validated:
        raise RuntimeError("staged annotation lock changed after validation")
    validate_consensus_annotation(copied_payload, protocol=protocol)
    return _AnnotationLockToken(
        _nonce=_GATE_NONCE,
        payload=copied_payload,
        source_file_sha256=source_sha,
        copied_file_sha256=copied_sha,
        copied_path=copied_path,
    )


def _require_gate(token: _AnnotationLockToken) -> None:
    if not isinstance(token, _AnnotationLockToken) or token._nonce is not _GATE_NONCE:
        raise RuntimeError(
            "annotation gate is closed: validate and stage the complete blind annotation first"
        )
    if (
        not token.copied_path.is_file()
        or sha256_file(token.copied_path) != token.copied_file_sha256
    ):
        raise RuntimeError("staged annotation lock is missing or changed")
    try:
        locked_payload = json.loads(token.copied_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("staged annotation lock is no longer valid UTF-8 JSON") from error
    if locked_payload != token.payload or locked_payload.get(
        ANNOTATION_HASH_KEY
    ) != _canonical_self_hash(locked_payload, ANNOTATION_HASH_KEY):
        raise RuntimeError("in-memory annotation or staged annotation lock changed")


def _load_frozen_trace_summaries(
    token: _AnnotationLockToken,
    *,
    shard_root: Path,
    results: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Open only the three preregistered readout arrays, after the hard gate."""

    _require_gate(token)
    record = results["private_trace"]
    trace_path = shard_root / TRACE_NAME
    if (
        record.get("relative_path") != TRACE_NAME
        or not trace_path.is_file()
        or trace_path.stat().st_size != record.get("bytes")
        or sha256_file(trace_path) != record.get("sha256")
    ):
        raise RuntimeError("private trace identity changed after shard validation")
    with np.load(trace_path, allow_pickle=False) as archive:
        primary = np.array(archive["primary_ever_alarm"], copy=True)
        terminal = np.array(
            archive["secondary_terminal_path_mixture_log_e"], copy=True
        )
        running = np.array(
            archive["secondary_running_max_path_mixture_log_e"], copy=True
        )
    if primary.shape != (BRANCHES_PER_SHARD,) or primary.dtype != np.uint8:
        raise RuntimeError("primary ever-alarm readout schema changed")
    if not np.isin(primary, (0, 1)).all():
        raise RuntimeError("primary ever-alarm readout is not binary")
    for name, values in (("terminal", terminal), ("running maximum", running)):
        if values.shape != (BRANCHES_PER_SHARD,) or values.dtype != np.float64:
            raise RuntimeError(f"secondary {name} readout schema changed")
        if not np.isfinite(values).all():
            raise RuntimeError(f"secondary {name} readout is non-finite")
    if np.any(running < -1e-15):
        raise RuntimeError("secondary running maximum excluded the initial E=1 value")
    return primary, terminal, running


def validate_and_load_pool(
    token: _AnnotationLockToken, shard_dirs: Sequence[Path]
) -> tuple[list[_BranchReadout], list[dict[str, Any]]]:
    """Validate exactly four shards and join only after annotation lock."""

    _require_gate(token)
    if len(shard_dirs) != TOTAL_SHARDS:
        raise RuntimeError("exactly four shard directories are required")
    annotations = {row["blind_id"]: row for row in token.payload["annotations"]}
    readouts: list[_BranchReadout] = []
    shard_records: list[dict[str, Any]] = []
    seen_shard_indices: set[int] = set()
    seen_blind_ids: set[str] = set()
    reference: dict[str, Any] | None = None
    for supplied in shard_dirs:
        unresolved_root = supplied.expanduser()
        if unresolved_root.is_symlink():
            raise RuntimeError("shard directory must not be a symlink")
        root = unresolved_root.resolve()
        manifest, results = validate_shard_bundle(root)
        if manifest.get("experiment") != SHARD_EXPERIMENT:
            raise RuntimeError("unexpected shard experiment")
        shard_index = int(manifest["pool"]["this_shard_index"])
        if shard_index in seen_shard_indices:
            raise RuntimeError("duplicate validation shard index")
        seen_shard_indices.add(shard_index)
        compatibility = {
            "protocol_identity_sha256": manifest["protocol"]["protocol_identity_sha256"],
            "input_prefix": manifest["input_prefix"],
            "schedule": manifest["schedule"],
            "frozen_primary_candidate": manifest["frozen_primary_candidate"],
            "frozen_secondary": manifest["frozen_secondary"],
            "pool_constants": {
                key: manifest["pool"][key]
                for key in (
                    "prefix_seed",
                    "target_batch_index",
                    "target_class_id",
                    "rollback_internal_timestep",
                    "pool_seed",
                    "total_shards",
                    "branches_per_shard",
                    "total_pool_branches",
                )
            },
            "runner_sha256": manifest["runner"]["sha256"],
        }
        if compatibility["protocol_identity_sha256"] != token.payload[
            "protocol_identity_sha256"
        ]:
            raise RuntimeError("shard protocol differs from the locked annotation")
        if reference is None:
            reference = compatibility
        elif compatibility != reference:
            raise RuntimeError("validation shards are not mutually compatible")

        primary, terminal, running = _load_frozen_trace_summaries(
            token, shard_root=root, results=results
        )
        branch_records = results["branch_records"]
        if len(branch_records) != BRANCHES_PER_SHARD:
            raise RuntimeError("validated shard branch count changed")
        for local_index, record in enumerate(branch_records):
            identifier = record["blind_id"]
            if identifier in seen_blind_ids:
                raise RuntimeError("duplicate blind ID across validation shards")
            seen_blind_ids.add(identifier)
            row = annotations.get(identifier)
            if row is None:
                raise RuntimeError("validated shard blind ID is absent from locked annotation")
            readouts.append(
                _BranchReadout(
                    blind_id=identifier,
                    primary_label=row["primary_hind_limb_topology"],
                    overall_label=row["overall_structural_secondary"],
                    primary_alarm=int(primary[local_index]),
                    secondary_terminal=float(terminal[local_index]),
                    secondary_running_max=float(running[local_index]),
                )
            )
        shard_records.append(
            {
                "shard_index": shard_index,
                "root": str(root),
                "runner_sha256": manifest["runner"]["sha256"],
                "manifest_identity_sha256": manifest["identity_sha256"],
                "manifest_file_sha256": sha256_file(root / MANIFEST_NAME),
                "results_payload_sha256": results["payload_sha256"],
                "results_file_sha256": sha256_file(root / "results.json"),
                "completion_file_sha256": sha256_file(root / "completion.json"),
                "private_trace_sha256": results["private_trace"]["sha256"],
                "protocol_copy_file_sha256": sha256_file(root / PROTOCOL_COPY_NAME),
            }
        )
    if seen_shard_indices != set(range(TOTAL_SHARDS)):
        raise RuntimeError("the four validated shards do not cover shard indices 0..3")
    expected_ids = {blind_id(index) for index in range(TOTAL_POOL_BRANCHES)}
    if seen_blind_ids != expected_ids or len(readouts) != TOTAL_POOL_BRANCHES:
        raise RuntimeError("the four validated shards do not cover the exact 32 blind IDs")
    return readouts, sorted(shard_records, key=lambda item: item["shard_index"])


def _exact_binomial_interval(successes: int, total: int) -> dict[str, float] | None:
    if total == 0:
        return None
    alpha = 1.0 - CONFIDENCE_LEVEL
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2.0, successes, total - successes + 1))
    upper = 1.0 if successes == total else float(
        beta.ppf(1.0 - alpha / 2.0, successes + 1, total - successes)
    )
    return {
        "confidence_level": CONFIDENCE_LEVEL,
        "method": "Clopper-Pearson exact two-sided",
        "lower": lower,
        "upper": upper,
    }


def _rate(successes: int, total: int) -> float | None:
    return None if total == 0 else successes / total


def primary_readout(rows: Sequence[_BranchReadout]) -> dict[str, Any]:
    clear = [row for row in rows if row.primary_label == "clear_failure"]
    not_clear = [row for row in rows if row.primary_label == "not_clear_failure"]
    uncertain = [row for row in rows if row.primary_label == "uncertain"]
    clear_alarm = sum(row.primary_alarm for row in clear)
    not_clear_alarm = sum(row.primary_alarm for row in not_clear)
    clear_no_alarm = len(clear) - clear_alarm
    not_clear_no_alarm = len(not_clear) - not_clear_alarm
    tpr = _rate(clear_alarm, len(clear))
    fpr = _rate(not_clear_alarm, len(not_clear))
    difference = None if tpr is None or fpr is None else tpr - fpr
    if clear and not_clear:
        fisher = fisher_exact(
            [[clear_alarm, clear_no_alarm], [not_clear_alarm, not_clear_no_alarm]],
            alternative="greater",
        )
        odds_ratio = float(fisher.statistic)
        fisher_payload = {
            "alternative": "TPR>FPR",
            "odds_ratio": odds_ratio if math.isfinite(odds_ratio) else None,
            "odds_ratio_is_positive_infinity": math.isinf(odds_ratio) and odds_ratio > 0,
            "p_value": float(fisher.pvalue),
        }
    else:
        fisher_payload = {
            "alternative": "TPR>FPR",
            "odds_ratio": None,
            "odds_ratio_is_positive_infinity": False,
            "p_value": None,
        }
    total_alarms = sum(row.primary_alarm for row in rows)
    return {
        "candidate": {
            "delta_nu": DELTA_NU,
            "tile_index_row_major_4x4": PRIMARY_TILE_INDEX,
            "sign": "+theta",
            "total_suffix_K_cap": TOTAL_K_PER_COMPONENT,
            "alarm_log_e": ALARM_LOG_E,
            "alpha_e": ALPHA_E,
        },
        "cross_tab_ever_alarm_by_primary_endpoint": {
            "rows": ["clear_failure", "not_clear_failure"],
            "columns": ["alarm", "no_alarm"],
            "counts": [
                [clear_alarm, clear_no_alarm],
                [not_clear_alarm, not_clear_no_alarm],
            ],
            "excluded_uncertain_count": len(uncertain),
        },
        "TPR": {
            "numerator": clear_alarm,
            "denominator": len(clear),
            "value": tpr,
            "exact_interval": _exact_binomial_interval(clear_alarm, len(clear)),
        },
        "FPR": {
            "numerator": not_clear_alarm,
            "denominator": len(not_clear),
            "value": fpr,
            "exact_interval": _exact_binomial_interval(not_clear_alarm, len(not_clear)),
        },
        "TPR_minus_FPR": difference,
        "fisher_exact_one_sided": fisher_payload,
        "total_alarms": {
            "count": total_alarms,
            "denominator": len(rows),
            "fraction": total_alarms / len(rows),
            "conditional_Ville_expected_count_upper_bound": ALPHA_E * len(rows),
            "alpha_e_is_not_FPR": True,
        },
        "predeclared_event_count": {
            "clear_failure_count": len(clear),
            "event_limited_inconclusive_if_fewer_than_three": len(clear) < 3,
            "alarmed_clear_failure_count": clear_alarm,
            "at_least_two_clear_failures_crossed_for_rollback_gate": clear_alarm >= 2,
        },
    }


def overall_structural_readout(rows: Sequence[_BranchReadout]) -> dict[str, Any]:
    """Frozen descriptive cross-tab; this secondary endpoint has no test."""

    clear = [row for row in rows if row.overall_label == "clear_structural_bad"]
    not_clear = [
        row for row in rows if row.overall_label == "not_clear_structural_bad"
    ]
    uncertain = [row for row in rows if row.overall_label == "uncertain"]
    clear_alarm = sum(row.primary_alarm for row in clear)
    not_clear_alarm = sum(row.primary_alarm for row in not_clear)
    clear_rate = _rate(clear_alarm, len(clear))
    not_clear_rate = _rate(not_clear_alarm, len(not_clear))
    difference = (
        None
        if clear_rate is None or not_clear_rate is None
        else clear_rate - not_clear_rate
    )
    return {
        "role": (
            "preregistered descriptive secondary endpoint; not the primary "
            "hind-limb endpoint"
        ),
        "cross_tab_ever_alarm_by_overall_structural_endpoint": {
            "rows": ["clear_structural_bad", "not_clear_structural_bad"],
            "columns": ["alarm", "no_alarm"],
            "counts": [
                [clear_alarm, len(clear) - clear_alarm],
                [not_clear_alarm, len(not_clear) - not_clear_alarm],
            ],
            "excluded_uncertain_count": len(uncertain),
        },
        "alarm_rate_clear_structural_bad": {
            "numerator": clear_alarm,
            "denominator": len(clear),
            "value": clear_rate,
        },
        "alarm_rate_not_clear_structural_bad": {
            "numerator": not_clear_alarm,
            "denominator": len(not_clear),
            "value": not_clear_rate,
        },
        "alarm_rate_difference": difference,
        "inferential_test_performed": False,
    }


def _tie_aware_auc_and_ranks(
    rows: Sequence[_BranchReadout], *, attribute: str
) -> dict[str, Any]:
    included = [row for row in rows if row.primary_label != "uncertain"]
    positives = np.asarray(
        [getattr(row, attribute) for row in included if row.primary_label == "clear_failure"],
        dtype=np.float64,
    )
    negatives = np.asarray(
        [
            getattr(row, attribute)
            for row in included
            if row.primary_label == "not_clear_failure"
        ],
        dtype=np.float64,
    )
    scores = np.asarray([getattr(row, attribute) for row in included], dtype=np.float64)
    labels = np.asarray(
        [row.primary_label == "clear_failure" for row in included], dtype=bool
    )
    if positives.size and negatives.size:
        comparisons = positives[:, None] - negatives[None, :]
        wins = int(np.count_nonzero(comparisons > 0.0))
        ties = int(np.count_nonzero(comparisons == 0.0))
        pair_count = int(comparisons.size)
        auc = (wins + 0.5 * ties) / pair_count
        rank_biserial = 2.0 * auc - 1.0
        tie_fraction = ties / pair_count
    else:
        auc = None
        rank_biserial = None
        tie_fraction = None
    descending = rankdata(-scores, method="average") if scores.size else np.asarray([])
    return {
        "endpoint": "primary_hind_limb_topology; uncertain excluded",
        "positive": "clear_failure",
        "negative": "not_clear_failure",
        "n_positive": int(positives.size),
        "n_negative": int(negatives.size),
        "n_uncertain_excluded": len(rows) - len(included),
        "roc_auc_ties_half": auc,
        "rank_biserial_correlation": rank_biserial,
        "positive_negative_pair_tie_fraction": tie_fraction,
        "tie_aware_descending_rank": {
            "method": "average rank; rank 1 is largest frozen mixture score",
            "mean_clear_failure": (
                None if not positives.size else float(np.mean(descending[labels]))
            ),
            "mean_not_clear_failure": (
                None if not negatives.size else float(np.mean(descending[~labels]))
            ),
        },
        "inferential_test_performed": False,
    }


def secondary_readout(rows: Sequence[_BranchReadout]) -> dict[str, Any]:
    return {
        "frozen_e_process": {
            "delta_nu": DELTA_NU,
            "components": SIGNED_COMPONENT_COUNT,
            "definition": "uniform fixed path mixture: global + 16 tiles, each +/-theta",
            "total_suffix_K_cap_per_component": TOTAL_K_PER_COMPONENT,
            "individual_component_scan_performed": False,
        },
        "terminal_fixed_34_path_mixture_log_e": _tie_aware_auc_and_ranks(
            rows, attribute="secondary_terminal"
        ),
        "running_max_fixed_34_path_mixture_log_e": _tie_aware_auc_and_ranks(
            rows, attribute="secondary_running_max"
        ),
    }


def tail_readout(annotation: dict[str, Any]) -> dict[str, Any]:
    rows = annotation["annotations"]
    scorable = [row for row in rows if row["tail_scorable"] == "yes"]
    output: dict[str, Any] = {
        "scope": "descriptive only; no composite and no evidence/alarm association",
        "total_images": len(rows),
        "tail_scorable_yes": len(scorable),
        "tail_scorable_no": len(rows) - len(scorable),
        "dimensions": {},
    }
    meanings = {
        "tail_R_root_attachment": "R: root attachment; 0 natural, 1 mild/uncertain, 2 clear defect",
        "tail_T_taper_and_volume": "T: taper/volume; 0 natural, 1 mild/uncertain, 2 clear defect",
        "tail_F_feather_or_hair_flow": "F: feather/hair flow; 0 natural, 1 mild/uncertain, 2 clear defect",
        "tail_D_distal_tip": "D: distal tip; 0 natural, 1 mild/uncertain, 2 clear defect",
        "tail_P_paddle_like": "P: paddle-like flag; 0 absent, 1 present",
        "tail_B_short_blunt": "B: short/blunt flag; 0 absent, 1 present",
        "tail_S_abrupt_filament_transition": "S: abrupt filament transition flag; 0 absent, 1 present",
    }
    for key in TAIL_REPORT_ORDER:
        levels = (0, 1, 2) if key in TERNARY_TAIL_FIELDS else (0, 1)
        counts = {str(level): sum(row[key] == level for row in scorable) for level in levels}
        if sum(counts.values()) != len(scorable):
            raise AssertionError(f"tail distribution denominator failed: {key}")
        output["dimensions"][key] = {
            "meaning": meanings[key],
            "denominator_tail_scorable_yes": len(scorable),
            "counts": counts,
            "unscorable_null_count": len(rows) - len(scorable),
        }
    return output


def build_summary(
    rows: Sequence[_BranchReadout], annotation: dict[str, Any]
) -> dict[str, Any]:
    if len(rows) != TOTAL_POOL_BRANCHES:
        raise RuntimeError("summary requires all 32 retained suffixes")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "scope": {
            "prospective_within_prefix_conditional_validation": True,
            "general_confirmation": False,
            "shared_saved_x60_prefix": True,
            "all_32_new_suffix_streams_retained": True,
        },
        "annotation_identity_sha256": annotation[ANNOTATION_HASH_KEY],
        "primary": primary_readout(rows),
        "secondary_overall_structural_descriptive": overall_structural_readout(
            rows
        ),
        "secondary_frozen_34_path_mixture_only": secondary_readout(rows),
        "tail_R_T_F_D_P_B_S_separate": tail_readout(annotation),
        "multiplicity_guard": {
            "tile_sign_delta_nu_K_threshold_endpoint_subset_scan_performed": False,
            "individual_component_results_reported": False,
            "per_branch_evidence_or_alarm_reported": False,
        },
    }
    payload["payload_sha256"] = _canonical_self_hash(payload, "payload_sha256")
    return payload


def _build_manifest(
    *,
    protocol_path: Path,
    protocol_file_sha256: str,
    protocol: dict[str, Any],
    annotation_path: Path,
    token: _AnnotationLockToken,
    shard_records: list[dict[str, Any]],
) -> dict[str, Any]:
    runner_path = Path(__file__).resolve()
    pool_runner_path = Path(sys.modules[validate_shard_bundle.__module__].__file__).resolve()
    pool_runner_sha256 = sha256_file(pool_runner_path)
    shard_runner_hashes = {
        record.get("runner_sha256") for record in shard_records
    }
    if shard_runner_hashes != {pool_runner_sha256}:
        raise RuntimeError("validated shard runner source changed during the readout")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "role": "IMMUTABLE_CLOSED_ONE_TIME_BLINDED_READOUT",
        "scope": {
            "within_prefix_only": True,
            "general_confirmation": False,
            "branch_level_evidence_exported": False,
            "raw_trace_or_png_copied": False,
        },
        "annotation_gate": {
            "validated_and_staged_before_any_shard_validator_or_trace_loader": True,
            "source_path": str(annotation_path.expanduser().resolve()),
            "source_file_sha256": token.source_file_sha256,
            "copied_relative_path": LOCKED_ANNOTATION_NAME,
            "copied_file_sha256": token.copied_file_sha256,
            "annotation_identity_sha256": token.payload[ANNOTATION_HASH_KEY],
            "row_count": TOTAL_POOL_BRANCHES,
            "all_reviewers_and_adjudicator_declared_evidence_unseen": True,
        },
        "protocol": {
            "path": str(protocol_path.expanduser().resolve()),
            "file_sha256": protocol_file_sha256,
            "identity_sha256": protocol["protocol_identity_sha256"],
        },
        "frozen_analysis": {
            "primary": "+theta/tile_12/Delta-nu=0.25/K=0.5/log(5) ever alarm",
            "overall_structure": (
                "descriptive alarm cross-tab for clear versus not-clear "
                "structural bad; uncertain excluded; no inferential test"
            ),
            "secondary": "terminal and running maximum of fixed uniform 34-path +/- mixture",
            "tail": "R/T/F/D/P/B/S separate descriptive counts",
            "confidence_level": CONFIDENCE_LEVEL,
            "fisher_alternative": "greater (TPR>FPR)",
            "search_parameters_available": False,
        },
        "shards": shard_records,
        "runner": {"path": str(runner_path), "sha256": sha256_file(runner_path)},
        "validated_shard_runner": {
            "path": str(pool_runner_path),
            "sha256": sha256_file(pool_runner_path),
        },
        "outputs": {
            "files": [
                MANIFEST_NAME,
                SUMMARY_NAME,
                COMPLETION_NAME,
                LOCKED_ANNOTATION_NAME,
            ],
            "atomic_no_replace": True,
            "closed_file_set": True,
        },
    }
    payload["identity_sha256"] = _canonical_self_hash(payload, "identity_sha256")
    return payload


def validate_closed_summary_bundle(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")):
        raise RuntimeError("closed summary bundle must not contain symlinks")
    expected_files = {
        (root / MANIFEST_NAME).resolve(),
        (root / SUMMARY_NAME).resolve(),
        (root / COMPLETION_NAME).resolve(),
        (root / LOCKED_ANNOTATION_NAME).resolve(),
    }
    actual_files = {path.resolve() for path in root.rglob("*") if path.is_file()}
    actual_directories = {path.resolve() for path in root.rglob("*") if path.is_dir()}
    if actual_files != expected_files or actual_directories:
        raise RuntimeError("closed summary bundle file set changed")
    with (root / MANIFEST_NAME).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    with (root / SUMMARY_NAME).open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    with (root / COMPLETION_NAME).open("r", encoding="utf-8") as handle:
        completion = json.load(handle)
    with (root / LOCKED_ANNOTATION_NAME).open("r", encoding="utf-8") as handle:
        annotation = json.load(handle)
    if manifest.get("identity_sha256") != _canonical_self_hash(manifest, "identity_sha256"):
        raise RuntimeError("closed manifest self-hash failed")
    if summary.get("payload_sha256") != _canonical_self_hash(summary, "payload_sha256"):
        raise RuntimeError("closed summary self-hash failed")
    if set(summary) != SUMMARY_TOP_LEVEL_KEYS:
        raise RuntimeError("closed summary top-level schema changed")
    if set(summary.get("primary", {})) != PRIMARY_OUTPUT_KEYS:
        raise RuntimeError("closed primary readout schema changed")
    if set(summary.get("secondary_overall_structural_descriptive", {})) != (
        OVERALL_OUTPUT_KEYS
    ):
        raise RuntimeError("closed overall-structure descriptive readout changed")
    if set(summary.get("secondary_frozen_34_path_mixture_only", {})) != SECONDARY_OUTPUT_KEYS:
        raise RuntimeError("closed secondary readout escaped its two frozen scores")
    if set(
        summary.get("tail_R_T_F_D_P_B_S_separate", {})
        .get("dimensions", {})
        .keys()
    ) != set(TAIL_REPORT_ORDER):
        raise RuntimeError("closed tail readout is not separate R/T/F/D/P/B/S")
    if summary.get("multiplicity_guard") != {
        "tile_sign_delta_nu_K_threshold_endpoint_subset_scan_performed": False,
        "individual_component_results_reported": False,
        "per_branch_evidence_or_alarm_reported": False,
    }:
        raise RuntimeError("closed summary multiplicity guard changed")
    forbidden_summary_keys = {
        "blind_id",
        "branch_global_index",
        "primary_alarm",
        "secondary_terminal",
        "secondary_running_max",
    }

    def _walk(value: Any) -> Iterable[str]:
        if isinstance(value, dict):
            for key, child in value.items():
                yield key
                yield from _walk(child)
        elif isinstance(value, list):
            for child in value:
                yield from _walk(child)

    # This checks the analytical summary, not annotation_locked.json: the latter
    # necessarily retains blind IDs to make the pre-unseal label lock auditable.
    if forbidden_summary_keys.intersection(_walk(summary)):
        raise RuntimeError("closed analytical summary leaks a per-branch join")
    if annotation.get(ANNOTATION_HASH_KEY) != _canonical_self_hash(
        annotation, ANNOTATION_HASH_KEY
    ):
        raise RuntimeError("locked annotation self-hash failed")
    fixed_completion = {
        "complete": True,
        "manifest_identity_sha256": manifest["identity_sha256"],
        "manifest_file_sha256": sha256_file(root / MANIFEST_NAME),
        "summary_payload_sha256": summary["payload_sha256"],
        "summary_file_sha256": sha256_file(root / SUMMARY_NAME),
        "annotation_identity_sha256": annotation[ANNOTATION_HASH_KEY],
        "annotation_file_sha256": sha256_file(root / LOCKED_ANNOTATION_NAME),
        "file_count": 4,
        "raw_trace_or_png_present": False,
    }
    if set(completion) != {*fixed_completion, "payload_sha256"} or any(
        completion.get(key) != value for key, value in fixed_completion.items()
    ):
        raise RuntimeError("closed completion links changed")
    if completion.get("payload_sha256") != _canonical_self_hash(
        completion, "payload_sha256"
    ):
        raise RuntimeError("closed completion self-hash failed")
    if manifest["annotation_gate"]["copied_file_sha256"] != sha256_file(
        root / LOCKED_ANNOTATION_NAME
    ):
        raise RuntimeError("manifest annotation lock file hash changed")
    if summary["annotation_identity_sha256"] != annotation[ANNOTATION_HASH_KEY]:
        raise RuntimeError("summary/annotation identity link changed")
    return manifest, summary


def _paths_overlap(left: Path, right: Path) -> bool:
    left_resolved = left.expanduser().resolve()
    right_resolved = right.expanduser().resolve()
    return (
        left_resolved == right_resolved
        or left_resolved in right_resolved.parents
        or right_resolved in left_resolved.parents
    )


def run_real(args: argparse.Namespace) -> None:
    # Allowed pre-gate inputs: frozen source protocol and consensus annotation only.
    protocol_path = args.protocol.expanduser().resolve()
    protocol_file_sha256 = sha256_file(protocol_path)
    protocol = _load_protocol(protocol_path)
    if sha256_file(protocol_path) != protocol_file_sha256:
        raise RuntimeError("frozen protocol file changed while it was being loaded")
    if protocol.get("protocol_status") != "FROZEN_BEFORE_GPU_EXECUTION":
        raise RuntimeError("readout requires the exact frozen-before-GPU protocol status")
    _validate_frozen_secondary_protocol(protocol)
    # Do not resolve away a possible annotation symlink before the gate checks it.
    annotation_path = args.annotation.expanduser()
    # The first staging area is independent of both output and shard paths.  This
    # prevents an accidentally nested --outdir from mutating a shard before the
    # annotation gate exists.
    with tempfile.TemporaryDirectory(prefix="dit-t60-annotation-gate-") as gate_temporary:
        gate_staging = Path(gate_temporary) / "lock"
        gate_staging.mkdir()
        # Hard procedural boundary.  No shard path is resolved, validated,
        # listed, opened, or modified above this line.
        token = lock_annotation_before_unseal(
            annotation_path, protocol=protocol, staging=gate_staging
        )
        # Even output-path metadata is inspected only after the annotation lock;
        # this preserves the gate under adversarial path aliasing.
        if args.outdir.expanduser().exists():
            raise RuntimeError(f"refusing to overwrite existing output path: {args.outdir}")
        rows, shard_records = validate_and_load_pool(token, args.shard_dirs)
        if sha256_file(protocol_path) != protocol_file_sha256:
            raise RuntimeError("frozen protocol file changed after annotation lock")
        summary = build_summary(rows, token.payload)
        outdir = args.outdir.expanduser().resolve()
        for record in shard_records:
            if _paths_overlap(outdir, Path(record["root"])):
                raise RuntimeError("--outdir must be outside every immutable shard bundle")
        if _paths_overlap(outdir, annotation_path) or _paths_overlap(outdir, protocol_path):
            raise RuntimeError("--outdir must not contain or replace protocol/annotation inputs")
        outdir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{outdir.name}.staging-", dir=outdir.parent
        ) as output_temporary:
            staging = Path(output_temporary) / "bundle"
            staging.mkdir()
            locked_bytes = token.copied_path.read_bytes()
            _atomic_bytes_dump(locked_bytes, staging / LOCKED_ANNOTATION_NAME)
            if sha256_file(staging / LOCKED_ANNOTATION_NAME) != token.copied_file_sha256:
                raise RuntimeError("final annotation lock copy changed")
            manifest = _build_manifest(
                protocol_path=protocol_path,
                protocol_file_sha256=protocol_file_sha256,
                protocol=protocol,
                annotation_path=annotation_path,
                token=token,
                shard_records=shard_records,
            )
            atomic_json_dump(manifest, staging / MANIFEST_NAME)
            atomic_json_dump(summary, staging / SUMMARY_NAME)
            completion: dict[str, Any] = {
                "complete": True,
                "manifest_identity_sha256": manifest["identity_sha256"],
                "manifest_file_sha256": sha256_file(staging / MANIFEST_NAME),
                "summary_payload_sha256": summary["payload_sha256"],
                "summary_file_sha256": sha256_file(staging / SUMMARY_NAME),
                "annotation_identity_sha256": token.payload[ANNOTATION_HASH_KEY],
                "annotation_file_sha256": sha256_file(staging / LOCKED_ANNOTATION_NAME),
                "file_count": 4,
                "raw_trace_or_png_present": False,
            }
            completion["payload_sha256"] = _canonical_self_hash(
                completion, "payload_sha256"
            )
            atomic_json_dump(completion, staging / COMPLETION_NAME)
            validate_closed_summary_bundle(staging)
            _atomic_install_directory_noreplace(staging, outdir)
        validate_closed_summary_bundle(outdir)
    print(
        json.dumps(
            {
                "status": "complete",
                "outdir": str(outdir),
                "annotation_identity_sha256": token.payload[ANNOTATION_HASH_KEY],
                "branches": TOTAL_POOL_BRANCHES,
                "per_branch_evidence_or_alarm_exposed": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def _synthetic_annotation(protocol: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index in range(TOTAL_POOL_BRANCHES):
        scorable = index % 7 != 0
        identity = "clear" if scorable else "unclear"
        ternary = 0 if scorable else None
        binary = 0 if scorable else None
        rows.append(
            {
                "blind_id": blind_id(index),
                "notes": "synthetic visual annotation",
                "overall_structural_secondary": (
                    "clear_structural_bad" if index % 5 == 0 else "not_clear_structural_bad"
                ),
                "primary_hind_limb_topology": (
                    "uncertain"
                    if index in (30, 31)
                    else "clear_failure"
                    if index % 4 == 0
                    else "not_clear_failure"
                ),
                "tail_B_short_blunt": binary,
                "tail_D_distal_tip": ternary,
                "tail_F_feather_or_hair_flow": ternary,
                "tail_P_paddle_like": binary,
                "tail_R_root_attachment": ternary,
                "tail_S_abrupt_filament_transition": binary,
                "tail_T_taper_and_volume": ternary,
                "tail_confidence": "high" if scorable else "low",
                "tail_derived_label": "natural" if scorable else "uncertain",
                "tail_identity": identity,
                "tail_scorable": "yes" if scorable else "no",
            }
        )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "annotation_schema": ANNOTATION_SCHEMA,
        "protocol_identity_sha256": protocol["protocol_identity_sha256"],
        "status": ANNOTATION_STATUS,
        "reviewers": [
            {
                "reviewer_id": "synthetic_reviewer",
                "reviewed_at_utc": "2026-08-27T08:00:00Z",
                "evidence_unseen": True,
                "evidence_unseen_declaration": EVIDENCE_UNSEEN_DECLARATION,
            }
        ],
        "consensus": {
            "adjudicator_id": "synthetic_adjudicator",
            "locked_at_utc": "2026-08-27T08:01:00Z",
            "method": "single_reviewer_consensus",
            "evidence_unseen": True,
            "evidence_unseen_declaration": EVIDENCE_UNSEEN_DECLARATION,
            "row_count": TOTAL_POOL_BRANCHES,
        },
        "annotations": rows,
    }
    payload[ANNOTATION_HASH_KEY] = _canonical_self_hash(payload, ANNOTATION_HASH_KEY)
    return payload


def run_self_test(protocol_path: Path) -> None:
    """CPU-only synthetic test; it never accepts or discovers a shard path."""

    protocol = _load_protocol(protocol_path)
    _validate_frozen_secondary_protocol(protocol)
    if protocol.get("protocol_status") != "FROZEN_BEFORE_GPU_EXECUTION":
        raise AssertionError("self-test protocol is not the frozen-before-GPU version")
    payload = _synthetic_annotation(protocol)
    validate_consensus_annotation(payload, protocol=protocol)
    tampered = json.loads(json.dumps(payload))
    tampered["reviewers"][0]["evidence_unseen"] = False
    tampered[ANNOTATION_HASH_KEY] = _canonical_self_hash(tampered, ANNOTATION_HASH_KEY)
    try:
        validate_consensus_annotation(tampered, protocol=protocol)
    except RuntimeError:
        pass
    else:
        raise AssertionError("evidence-unseen declaration gate accepted a false declaration")
    missing_row = json.loads(json.dumps(payload))
    missing_row["annotations"].pop()
    missing_row["consensus"]["row_count"] = TOTAL_POOL_BRANCHES - 1
    missing_row[ANNOTATION_HASH_KEY] = _canonical_self_hash(
        missing_row, ANNOTATION_HASH_KEY
    )
    try:
        validate_consensus_annotation(missing_row, protocol=protocol)
    except RuntimeError:
        pass
    else:
        raise AssertionError("incomplete consensus annotation was accepted")
    inconsistent_tail = json.loads(json.dumps(payload))
    first_scorable = next(
        row for row in inconsistent_tail["annotations"] if row["tail_scorable"] == "yes"
    )
    first_scorable["tail_T_taper_and_volume"] = 2
    inconsistent_tail[ANNOTATION_HASH_KEY] = _canonical_self_hash(
        inconsistent_tail, ANNOTATION_HASH_KEY
    )
    try:
        validate_consensus_annotation(inconsistent_tail, protocol=protocol)
    except RuntimeError:
        pass
    else:
        raise AssertionError("inconsistent frozen tail-derived label was accepted")

    rng = np.random.default_rng(20260827)
    synthetic_rows: list[_BranchReadout] = []
    by_id = {row["blind_id"]: row for row in payload["annotations"]}
    for index in range(TOTAL_POOL_BRANCHES):
        row = by_id[blind_id(index)]
        label = row["primary_hind_limb_topology"]
        alarm = int(index in (0, 4, 7, 12, 17))
        shift = 0.7 if label == "clear_failure" else 0.0
        terminal = float(rng.normal(shift, 0.4))
        running = float(max(0.0, terminal, rng.normal(shift + 0.1, 0.3)))
        synthetic_rows.append(
            _BranchReadout(
                blind_id=blind_id(index),
                primary_label=label,
                overall_label=row["overall_structural_secondary"],
                primary_alarm=alarm,
                secondary_terminal=terminal,
                secondary_running_max=running,
            )
        )
    summary = build_summary(synthetic_rows, payload)
    primary = summary["primary"]
    if primary["total_alarms"]["count"] != 5:
        raise AssertionError("synthetic primary alarm count failed")
    if primary["fisher_exact_one_sided"]["alternative"] != "TPR>FPR":
        raise AssertionError("synthetic Fisher direction changed")
    overall = summary["secondary_overall_structural_descriptive"]
    if set(overall) != OVERALL_OUTPUT_KEYS:
        raise AssertionError("overall-structure descriptive schema changed")
    if overall["inferential_test_performed"] is not False:
        raise AssertionError("overall-structure endpoint escaped descriptive-only scope")
    if _exact_binomial_interval(0, 10) != {
        "confidence_level": CONFIDENCE_LEVEL,
        "method": "Clopper-Pearson exact two-sided",
        "lower": 0.0,
        "upper": float(beta.ppf(0.975, 1, 10)),
    }:
        raise AssertionError("exact binomial boundary interval changed")
    secondary = summary["secondary_frozen_34_path_mixture_only"]
    if set(secondary) != {
        "frozen_e_process",
        "terminal_fixed_34_path_mixture_log_e",
        "running_max_fixed_34_path_mixture_log_e",
    }:
        raise AssertionError("secondary readout escaped the frozen two-score boundary")
    tail = summary["tail_R_T_F_D_P_B_S_separate"]["dimensions"]
    if tuple(tail) != TAIL_REPORT_ORDER:
        raise AssertionError("tail R/T/F/D/P/B/S ordering changed")

    with tempfile.TemporaryDirectory(prefix="dit-t60-summary-self-test-") as temporary:
        root = Path(temporary)
        annotation_path = root / "synthetic_annotation.json"
        atomic_json_dump(payload, annotation_path)
        staging = root / "staging"
        staging.mkdir()
        token = lock_annotation_before_unseal(
            annotation_path, protocol=protocol, staging=staging
        )
        _require_gate(token)
        forged = _AnnotationLockToken(
            _nonce=object(),
            payload=token.payload,
            source_file_sha256=token.source_file_sha256,
            copied_file_sha256=token.copied_file_sha256,
            copied_path=token.copied_path,
        )
        try:
            _require_gate(forged)
        except RuntimeError:
            pass
        else:
            raise AssertionError("forged annotation gate token was accepted")
        try:
            validate_and_load_pool(forged, ())
        except RuntimeError as error:
            if "annotation gate is closed" not in str(error):
                raise AssertionError("pool loader checked inputs before its gate") from error
        else:
            raise AssertionError("pool loader accepted a forged annotation gate token")
        manifest = _build_manifest(
            protocol_path=protocol_path,
            protocol_file_sha256=sha256_file(protocol_path),
            protocol=protocol,
            annotation_path=annotation_path,
            token=token,
            shard_records=[
                {
                    "synthetic_shard_index": index,
                    "runner_sha256": sha256_file(
                        Path(sys.modules[validate_shard_bundle.__module__].__file__).resolve()
                    ),
                }
                for index in range(TOTAL_SHARDS)
            ],
        )
        atomic_json_dump(manifest, staging / MANIFEST_NAME)
        atomic_json_dump(summary, staging / SUMMARY_NAME)
        completion: dict[str, Any] = {
            "complete": True,
            "manifest_identity_sha256": manifest["identity_sha256"],
            "manifest_file_sha256": sha256_file(staging / MANIFEST_NAME),
            "summary_payload_sha256": summary["payload_sha256"],
            "summary_file_sha256": sha256_file(staging / SUMMARY_NAME),
            "annotation_identity_sha256": token.payload[ANNOTATION_HASH_KEY],
            "annotation_file_sha256": sha256_file(staging / LOCKED_ANNOTATION_NAME),
            "file_count": 4,
            "raw_trace_or_png_present": False,
        }
        completion["payload_sha256"] = _canonical_self_hash(
            completion, "payload_sha256"
        )
        atomic_json_dump(completion, staging / COMPLETION_NAME)
        validate_closed_summary_bundle(staging)
    print(
        "self-test passed: exact 32-row annotation schema/self-hash, reviewer and "
        "adjudicator evidence-unseen gates, staged byte lock, unforgeable shard-access "
        "token, primary 2x2/TPR/FPR/Fisher/exact-CI readout, preregistered overall-"
        "structure descriptive cross-tab, frozen two-score 34-path "
        "secondary AUC/ranks, and separate R/T/F/D/P/B/S summaries; no shard path used"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).resolve().parent
        / "configs/dit_imagenet256_t60_within_prefix_validation_v1.json",
    )
    parser.add_argument("--annotation", type=Path)
    parser.add_argument("--shard-dirs", type=Path, nargs=TOTAL_SHARDS)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run only CPU synthetic checks; no shard/result/trace/image path is accepted or read.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.self_test:
        if args.annotation is not None or args.shard_dirs is not None or args.outdir is not None:
            raise RuntimeError("--self-test refuses annotation, shard, and output arguments")
        run_self_test(args.protocol.expanduser().resolve())
        return
    if args.annotation is None or args.shard_dirs is None or args.outdir is None:
        raise RuntimeError("real readout requires --annotation, exactly four --shard-dirs, and --outdir")
    run_real(args)


if __name__ == "__main__":
    main()
