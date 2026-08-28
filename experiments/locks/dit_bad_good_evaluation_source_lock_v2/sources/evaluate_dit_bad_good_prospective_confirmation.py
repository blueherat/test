#!/usr/bin/env python3
"""Evaluate the frozen DiT bad/good confirmation after blind labels are locked.

This is deliberately a prospective-only evaluator for candidate protocol v5.
Its input order is part of the safety contract: the candidate protocol and the
blind consensus lock are fully validated first.  Only then may the program hash
the calibration-score member or open the fresh score/alert CSV.  Thus the first
score-label join can only occur after a completed, hash-bound label lock exists.

The production statistical constants are not CLI-tunable:

* 100,000 within-class label permutations, PCG64 seed 2026082701;
* 100,000 global-seed cluster bootstrap draws, PCG64 seed 2026082702;
* 100,000 within-(class,label) AUC bootstrap draws, PCG64 seed 2026082703.

Run ``--self-test`` for a synthetic-only smoke test.  The self-test never opens
the prospective score, endpoint, or label products.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import beta


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATE_LOCK = (
    ROOT / "experiments/locks/dit_bad_good_candidate_confirmation_lock_v5"
)
DEFAULT_CALIBRATION_LOCK = (
    ROOT / "experiments/locks/dit_bad_good_conformal_calibration_lock_v1"
)

EXPECTED_CANDIDATE_PROTOCOL_IDENTITY = (
    "198a82a7c8a0ab79d901c76a5c810f4a40889604a66f18e995d0699f73c12bce"
)
EXPECTED_CANDIDATE_MANIFEST_IDENTITY = (
    "773d564419517bc7eb9b9b75a7677aa08d466c4abe0d0caa90a34c2d49809b7b"
)
EXPECTED_CALIBRATION_IDENTITY = (
    "77c09e844ae510342f7370cbaacbbfb9e7de3378b6efaa6ddfada26c7375b03a"
)
EXPECTED_CALIBRATION_MANIFEST_IDENTITY = (
    "37b2bd0f9e5df239a604e4b12caff30a0102dd1315678e26c18f4b01570361c3"
)

EVALUATION_CLASSES = (207, 602, 795)
EVALUATION_SEEDS = tuple(range(50, 130))
CALIBRATION_SEEDS = tuple(range(30, 50))
EXPECTED_EVALUATION_ROWS = 240

PERMUTATION_DRAWS = 100_000
PERMUTATION_SEED = 2026082701
CLUSTER_BOOTSTRAP_DRAWS = 100_000
CLUSTER_BOOTSTRAP_SEED = 2026082702
AUC_BOOTSTRAP_DRAWS = 100_000
AUC_BOOTSTRAP_SEED = 2026082703
CONFIDENCE_LEVEL = 0.95

PRIMARY_SCORE = "S_UNION"
PRIMARY_ALERT_010 = "alert_alpha0p10_conformal"
PRIMARY_ALERT_005 = "alert_alpha0p05_conformal"
LABEL_BAD = "clear_bad"
LABEL_GOOD = "clean_good"
LABEL_EXCLUDED = "mild_or_disputed"
ALLOWED_LABELS = {LABEL_BAD, LABEL_GOOD, LABEL_EXCLUDED}


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256_bytes(raw)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _without_identity(value: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    payload = dict(value)
    identity = payload.pop("identity_sha256", None)
    return payload, identity


def _require_canonical_identity(value: dict[str, Any], description: str) -> str:
    payload, identity = _without_identity(value)
    if not isinstance(identity, str) or identity != canonical_sha256(payload):
        raise RuntimeError(f"{description} canonical identity mismatch")
    return identity


def _safe_member_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise RuntimeError(f"unsafe manifest member name: {relative!r}")
    path = root / relative
    resolved_root = root.resolve()
    resolved = path.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise RuntimeError(f"manifest member escapes its lock: {relative}")
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"manifest member is missing or symlinked: {path}")
    return path


@dataclass(frozen=True)
class LockedDirectory:
    root: Path
    manifest: dict[str, Any]
    completion: dict[str, Any]
    manifest_identity: str
    members: dict[str, dict[str, Any]]


def validate_locked_directory(root: Path, description: str) -> LockedDirectory:
    root = root.expanduser().absolute()
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"{description} must be a real non-symlink directory: {root}")
    manifest_path = root / "manifest.json"
    completion_path = root / "completion.json"
    if (
        not manifest_path.is_file()
        or manifest_path.is_symlink()
        or not completion_path.is_file()
        or completion_path.is_symlink()
    ):
        raise RuntimeError(f"{description} lacks real manifest/completion files")
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    manifest_identity = _require_canonical_identity(
        manifest, f"{description} manifest"
    )
    if completion.get("complete") is not True:
        raise RuntimeError(f"{description} is not marked complete")
    if completion.get("manifest_file_sha256") != sha256_file(manifest_path):
        raise RuntimeError(f"{description} manifest file changed after completion")
    if completion.get("manifest_identity_sha256") != manifest_identity:
        raise RuntimeError(f"{description} completion/manifest identity mismatch")

    raw_members = manifest.get("files")
    if not isinstance(raw_members, list) or not raw_members:
        raise RuntimeError(f"{description} manifest has no locked file inventory")
    members: dict[str, dict[str, Any]] = {}
    for item in raw_members:
        if not isinstance(item, dict):
            raise RuntimeError(f"{description} contains a non-object file record")
        name = item.get("name", item.get("relative_path"))
        if not isinstance(name, str) or name in members:
            raise RuntimeError(f"{description} has duplicate/invalid member names")
        path = _safe_member_path(root, name)
        if (
            path.stat().st_size != item.get("bytes")
            or sha256_file(path) != item.get("sha256")
        ):
            raise RuntimeError(f"{description} member hash/size mismatch: {path}")
        members[name] = item
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"{description} contains a symlink: {path}")
    actual_files = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    expected_files = set(members) | {"manifest.json", "completion.json"}
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        extra = sorted(actual_files - expected_files)
        raise RuntimeError(
            f"{description} file inventory mismatch; missing={missing}, extra={extra}"
        )
    return LockedDirectory(root, manifest, completion, manifest_identity, members)


def _require_member(lock: LockedDirectory, name: str, description: str) -> Path:
    if name not in lock.members:
        raise RuntimeError(f"{description} is not manifest-bound: {name}")
    return lock.root / name


def validate_candidate_lock(root: Path) -> dict[str, Any]:
    lock = validate_locked_directory(root, "candidate v5 lock")
    protocol_path = _require_member(
        lock, "candidate_protocol.json", "candidate protocol"
    )
    protocol = load_json(protocol_path)
    protocol_identity = _require_canonical_identity(protocol, "candidate protocol")
    if (
        lock.manifest_identity != EXPECTED_CANDIDATE_MANIFEST_IDENTITY
        or protocol_identity != EXPECTED_CANDIDATE_PROTOCOL_IDENTITY
        or lock.manifest.get("protocol_identity_sha256") != protocol_identity
        or lock.completion.get("protocol_file_sha256") != sha256_file(protocol_path)
        or lock.completion.get("protocol_identity_sha256") != protocol_identity
    ):
        raise RuntimeError("candidate v5 lock is not the preregistered immutable lock")
    if (
        protocol.get("schema_version") != 5
        or protocol.get("status")
        != "FROZEN_BEFORE_ANY_FRESH_SCORE_EXTRACTION_OR_ENDPOINT_VISUAL_REVIEW"
        or protocol.get("primary_candidate", {}).get("name") != PRIMARY_SCORE
        or protocol.get("primary_candidate", {}).get("formula") != "max(z_A, z_B)"
        or protocol.get("primary_candidate", {}).get("orientation")
        != "higher_is_more_bad_like"
    ):
        raise RuntimeError("candidate protocol v5 primary-score contract changed")
    fresh = protocol.get("fresh_confirmation", {})
    eval_spec = fresh.get("inferential_evaluation", {})
    calibration_spec = fresh.get("label_free_conformal_calibration", {})
    if (
        tuple(fresh.get("classes", ())) != EVALUATION_CLASSES
        or fresh.get("trajectory_count") != 300
        or eval_spec.get("trajectory_count") != EXPECTED_EVALUATION_ROWS
        or eval_spec.get("visual_labels_must_be_locked_before_score_join") is not True
        or tuple(
            range(
                eval_spec.get("seeds", {}).get("start_inclusive", 0),
                eval_spec.get("seeds", {}).get("stop_inclusive", -1) + 1,
            )
        )
        != EVALUATION_SEEDS
        or calibration_spec.get("trajectory_count") != 60
        or calibration_spec.get("visual_labels_used") is not False
        or tuple(
            range(
                calibration_spec.get("seeds", {}).get("start_inclusive", 0),
                calibration_spec.get("seeds", {}).get("stop_inclusive", -1) + 1,
            )
        )
        != CALIBRATION_SEEDS
    ):
        raise RuntimeError("candidate v5 frozen cohort/split contract changed")
    evaluation = protocol.get("evaluation", {})
    randomization = evaluation.get("primary_randomization_test", {})
    intervals = evaluation.get("uncertainty_intervals", {})
    gate = evaluation.get("initial_go_gate", {})
    if (
        randomization.get("draws") != PERMUTATION_DRAWS
        or randomization.get("rng")
        != "numpy.random.Generator(PCG64(seed=2026082701))"
        or randomization.get("sidedness") != "one-sided"
        or gate.get("minimum_clear_bad_events_for_decision") != 15
        or gate.get("S_UNION_class_matched_auc_at_least") != 0.75
        or gate.get("S_UNION_stratified_permutation_one_sided_p_below") != 0.05
        or gate.get("alpha_0p10_TPR_minus_FPR_point_above") != 0.0
        or gate.get("no_class_with_two_or_more_bad_events_has_auc_below") != 0.6
        or "100000" not in intervals.get("TPR_minus_FPR", "")
        or "2026082702" not in intervals.get("TPR_minus_FPR", "")
        or "100000" not in intervals.get("auc", "")
        or "2026082703" not in intervals.get("auc", "")
    ):
        raise RuntimeError("candidate v5 preregistered statistical contract changed")
    return protocol


def validate_calibration_lock(
    root: Path, protocol: dict[str, Any]
) -> tuple[dict[str, Any], LockedDirectory]:
    lock = validate_locked_directory(root, "conformal calibration lock v1")
    record_path = _require_member(
        lock, "calibration_locked.json", "calibration record"
    )
    record = load_json(record_path)
    record_identity = _require_canonical_identity(record, "calibration record")
    if (
        lock.manifest_identity != EXPECTED_CALIBRATION_MANIFEST_IDENTITY
        or record_identity != EXPECTED_CALIBRATION_IDENTITY
        or lock.manifest.get("calibration_identity_sha256") != record_identity
        or lock.completion.get("calibration_file_sha256") != sha256_file(record_path)
        or lock.completion.get("calibration_identity_sha256") != record_identity
        or record.get("candidate_protocol_identity_sha256")
        != protocol["identity_sha256"]
        or lock.manifest.get("candidate_protocol_identity_sha256")
        != protocol["identity_sha256"]
        or record.get("status")
        != "FROZEN_LABEL_FREE_BEFORE_INFERENTIAL_VISUAL_LABEL_JOIN"
        or record.get("visual_labels_read_or_emitted") is not False
        or record.get("score_function_was_selected_on_calibration_data") is not False
        or record.get("calibrated_score") != PRIMARY_SCORE
        or tuple(record.get("calibration_classes", ())) != EVALUATION_CLASSES
        or tuple(record.get("calibration_seeds", ())) != CALIBRATION_SEEDS
        or tuple(record.get("evaluation_seeds_excluded", ())) != EVALUATION_SEEDS
        or record.get("calibration_sample_count") != 60
    ):
        raise RuntimeError("calibration lock is not the preregistered label-free lock")
    thresholds = record.get("thresholds")
    if not isinstance(thresholds, dict) or set(thresholds) != {
        str(value) for value in EVALUATION_CLASSES
    }:
        raise RuntimeError("calibration lock class thresholds are incomplete")
    expected_rules = {
        "alpha_0p10": (19, 2.0 / 21.0),
        "alpha_0p05": (20, 1.0 / 21.0),
    }
    for class_id in EVALUATION_CLASSES:
        class_record = thresholds[str(class_id)]
        for alpha_name, (order, bound) in expected_rules.items():
            item = class_record.get(alpha_name, {})
            threshold = item.get("threshold")
            if (
                item.get("calibration_count") != 20
                or item.get("calibration_order_statistic_1_based") != order
                or item.get("strict_comparison")
                != "evaluation_S_UNION > threshold"
                or not math.isclose(
                    float(item.get("finite_sample_marginal_trigger_probability_upper_bound", math.nan)),
                    bound,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
                or not isinstance(threshold, (int, float))
                or not math.isfinite(float(threshold))
            ):
                raise RuntimeError(
                    f"calibration rule changed for class {class_id} {alpha_name}"
                )
    return record, lock


def _manifest_consensus_hash(manifest: dict[str, Any]) -> Any:
    explicit = manifest.get(
        "consensus_file_sha256", manifest.get("final_consensus_file_sha256")
    )
    if explicit is not None:
        return explicit
    for item in manifest.get("files", []):
        if item.get("name", item.get("relative_path")) == "consensus_locked.json":
            return item.get("sha256")
    return None


def _manifest_consensus_identity(manifest: dict[str, Any]) -> Any:
    return manifest.get(
        "consensus_identity_sha256", manifest.get("final_consensus_identity_sha256")
    )


def _validate_blinding_audit(
    consensus: dict[str, Any], manifest: dict[str, Any]
) -> None:
    audit = consensus.get("blinding_audit", manifest.get("blinding_audit"))
    if not isinstance(audit, dict):
        raise RuntimeError("consensus lock lacks the required explicit blinding_audit")
    expected = {
        "reviewer_count": 3,
        "endpoint_only_review": True,
        "metric_values_visible_to_reviewers": False,
        "alert_decisions_visible_to_reviewers": False,
        "trajectories_visible_to_reviewers": False,
        "labels_locked_before_score_join": True,
    }
    mismatched = {key: (audit.get(key), value) for key, value in expected.items() if audit.get(key) != value}
    if mismatched:
        raise RuntimeError(f"consensus blinding audit failed: {mismatched}")


def _row_seed(row: dict[str, Any]) -> int:
    seed = row.get("global_seed", row.get("seed"))
    if "global_seed" in row and "seed" in row and row["global_seed"] != row["seed"]:
        raise RuntimeError("consensus row seed/global_seed disagree")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise RuntimeError("consensus row seed is not an integer")
    return seed


def _review_scores(row: dict[str, Any]) -> tuple[int, int, int]:
    raw = row.get("review_scores")
    values: Iterable[Any]
    if isinstance(raw, dict):
        if len(raw) != 3:
            raise RuntimeError("each consensus row must contain exactly three reviewers")
        values = [raw[key] for key in sorted(raw)]
    elif isinstance(raw, list) and len(raw) == 3:
        values = raw
    else:
        raise RuntimeError("each consensus row must preserve three review_scores")
    scores: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError("review severity must be numeric")
        integer = int(value)
        if float(value) != integer or integer not in {0, 1, 2, 3}:
            raise RuntimeError("review severity must be one of 0,1,2,3")
        scores.append(integer)
    return tuple(scores)  # type: ignore[return-value]


def _native_image_hash(row: dict[str, Any]) -> str:
    record = row.get("native_image", row.get("endpoint_image"))
    if not isinstance(record, dict):
        raise RuntimeError("consensus row lacks native_image provenance")
    value = record.get("file_sha256", record.get("sha256"))
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value.lower())
    ):
        raise RuntimeError("consensus native_image lacks a valid file SHA-256")
    return value.lower()


def validate_consensus_lock(
    root: Path, protocol: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any], LockedDirectory]:
    """Validate labels completely before any score CSV is opened."""

    lock = validate_locked_directory(root, "fresh blind consensus lock")
    consensus_path = _require_member(lock, "consensus_locked.json", "consensus")
    consensus = load_json(consensus_path)
    consensus_identity = _require_canonical_identity(consensus, "fresh consensus")
    status = str(consensus.get("status", ""))
    status_upper = status.upper()
    if (
        "LOCKED" not in status_upper
        or "BEFORE" not in status_upper
        or not any(token in status_upper for token in ("SCORE_JOIN", "METRIC_JOIN"))
    ):
        raise RuntimeError("consensus status does not prove pre-join locking")
    if (
        lock.completion.get("consensus_file_sha256") != sha256_file(consensus_path)
        or lock.completion.get("consensus_identity_sha256") != consensus_identity
        or _manifest_consensus_hash(lock.manifest) != sha256_file(consensus_path)
        or _manifest_consensus_identity(lock.manifest) != consensus_identity
    ):
        raise RuntimeError("consensus payload is not cross-bound by manifest/completion")
    candidate_identity = consensus.get(
        "candidate_protocol_identity_sha256",
        lock.manifest.get("candidate_protocol_identity_sha256"),
    )
    if candidate_identity != protocol["identity_sha256"]:
        raise RuntimeError("consensus lock is not bound to candidate protocol v5")
    pack_identity = consensus.get(
        "blind_review_pack_identity_sha256",
        consensus.get(
            "blind_pack_identity_sha256",
            lock.manifest.get(
                "blind_review_pack_identity_sha256",
                lock.manifest.get("blind_pack_identity_sha256"),
            ),
        ),
    )
    if (
        not isinstance(pack_identity, str)
        or len(pack_identity) != 64
        or any(char not in "0123456789abcdef" for char in pack_identity.lower())
    ):
        raise RuntimeError("consensus lock lacks a valid blind-review-pack identity")
    _validate_blinding_audit(consensus, lock.manifest)

    rows = consensus.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_EVALUATION_ROWS:
        raise RuntimeError("consensus is not the complete 240-row evaluation cohort")
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for raw in rows:
        if not isinstance(raw, dict):
            raise RuntimeError("consensus contains a non-object row")
        seed = _row_seed(raw)
        class_id = raw.get("class_id")
        if isinstance(class_id, bool) or not isinstance(class_id, int):
            raise RuntimeError("consensus row class_id is not an integer")
        key = (seed, class_id)
        if key in seen:
            raise RuntimeError(f"duplicate consensus key: {key}")
        seen.add(key)
        sample_key = f"class{class_id:04d}_seed{seed:03d}"
        if raw.get("sample_key") != sample_key:
            raise RuntimeError(f"consensus sample_key mismatch: expected {sample_key}")
        label = raw.get("primary_label")
        if label not in ALLOWED_LABELS:
            raise RuntimeError(f"unsupported primary label: {label!r}")
        scores = _review_scores(raw)
        bad_votes = sum(value in {2, 3} for value in scores)
        good_votes = sum(value == 0 for value in scores)
        if label == LABEL_BAD and bad_votes < 2:
            raise RuntimeError(f"clear_bad lacks 2-of-3 high-severity votes: {sample_key}")
        if label == LABEL_GOOD and good_votes < 2:
            raise RuntimeError(f"clean_good lacks 2-of-3 zero-severity votes: {sample_key}")
        included = label in {LABEL_BAD, LABEL_GOOD}
        if (
            "binary_primary_included" in raw
            and bool(raw["binary_primary_included"]) is not included
        ):
            raise RuntimeError(f"binary inclusion flag mismatch: {sample_key}")
        normalized.append(
            {
                "global_seed": seed,
                "class_id": class_id,
                "sample_key": sample_key,
                "primary_label": label,
                "binary_primary_included": included,
                "clear_bad_vote_count": bad_votes,
                "clean_good_vote_count": good_votes,
                "native_image_file_sha256": _native_image_hash(raw),
            }
        )
    expected = {(seed, class_id) for seed in EVALUATION_SEEDS for class_id in EVALUATION_CLASSES}
    if seen != expected:
        missing = sorted(expected - seen)[:10]
        extra = sorted(seen - expected)[:10]
        raise RuntimeError(
            f"consensus is not exact seed x class Cartesian product; missing={missing}, extra={extra}"
        )
    computed_counts = {
        label: sum(row["primary_label"] == label for row in normalized)
        for label in (LABEL_BAD, LABEL_GOOD, LABEL_EXCLUDED)
    }
    if consensus.get("counts") != computed_counts or lock.manifest.get("counts") != computed_counts:
        raise RuntimeError("consensus/manifest label counts do not equal row counts")
    if lock.completion.get("locked_row_count", EXPECTED_EVALUATION_ROWS) != EXPECTED_EVALUATION_ROWS:
        raise RuntimeError("consensus completion row count is not 240")
    frame = pd.DataFrame(normalized).sort_values(
        ["global_seed", "class_id"], kind="mergesort"
    ).reset_index(drop=True)
    return frame, consensus, lock


def _coerce_boolean_series(series: pd.Series, name: str) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.to_numpy(dtype=bool)
    lowered = series.astype(str).str.strip().str.lower()
    if not lowered.isin({"true", "false", "1", "0"}).all():
        raise RuntimeError(f"column is not strictly boolean: {name}")
    return lowered.isin({"true", "1"}).to_numpy(dtype=bool)


def _assert_close(actual: np.ndarray, expected: np.ndarray, description: str) -> None:
    if not np.allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=False):
        difference = float(np.max(np.abs(actual - expected)))
        raise RuntimeError(f"{description} does not replay; max_abs_diff={difference}")


def _expected_cartesian_frame(frame: pd.DataFrame, description: str) -> None:
    if len(frame) != EXPECTED_EVALUATION_ROWS:
        raise RuntimeError(f"{description} row count is not 240")
    if frame[["global_seed", "class_id"]].duplicated().any():
        raise RuntimeError(f"{description} has duplicate seed/class keys")
    observed = {
        (int(row.global_seed), int(row.class_id))
        for row in frame[["global_seed", "class_id"]].itertuples(index=False)
    }
    expected = {(seed, class_id) for seed in EVALUATION_SEEDS for class_id in EVALUATION_CLASSES}
    if observed != expected:
        raise RuntimeError(f"{description} is not the exact frozen Cartesian product")


def _exact_control_columns(protocol: dict[str, Any]) -> list[str]:
    return [
        "control_" + name.replace("__full_maximum", "")
        for name in protocol["negative_controls"]["exact_path_evidence_running_maxima"]
    ]


def validate_alert_product(
    root: Path,
    protocol: dict[str, Any],
    calibration: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], LockedDirectory]:
    lock = validate_locked_directory(root, "fresh label-free alert product")
    summary_path = _require_member(lock, "summary.json", "alert summary")
    csv_path = _require_member(
        lock,
        "evaluation_scores_and_alerts_label_free.csv",
        "fresh label-free evaluation score table",
    )
    summary = load_json(summary_path)
    if (
        summary.get("status") != "COMPLETE_LABEL_FREE_CALIBRATED_EVALUATION_ALERTS"
        or summary.get("candidate_protocol_identity_sha256")
        != protocol["identity_sha256"]
        or summary.get("calibration_identity_sha256") != calibration["identity_sha256"]
        or summary.get("sample_count") != EXPECTED_EVALUATION_ROWS
        or tuple(summary.get("seeds", ())) != EVALUATION_SEEDS
        or summary.get("labels_read_or_emitted") is not False
        or summary.get("thresholds_reestimated") is not False
        or lock.manifest.get("candidate_protocol_identity_sha256")
        != protocol["identity_sha256"]
        or lock.manifest.get("calibration_identity_sha256")
        != calibration["identity_sha256"]
        or lock.completion.get("alerts_file_sha256") != sha256_file(csv_path)
    ):
        raise RuntimeError("alert product is incomplete, supervised, or bound elsewhere")
    source_names = {
        "applicator": "applicator_source.py",
        "imported_calibrator_helper": "calibrator_helper_source.py",
        "imported_scorer_helper": "scorer_helper_source.py",
    }
    expected_source_hashes = {
        key: lock.members.get(name, {}).get("sha256") for key, name in source_names.items()
    }
    if summary.get("implementation_source_sha256") != expected_source_hashes:
        raise RuntimeError("alert implementation hashes are not self-bound")

    # This is intentionally the first fresh score-table read in the program.
    frame = pd.read_csv(csv_path)
    forbidden_tokens = ("label", "review", "consensus", "severity", "adjudic")
    forbidden_columns = [
        name for name in frame.columns if any(token in name.lower() for token in forbidden_tokens)
    ]
    if forbidden_columns:
        raise RuntimeError(f"label-free alert table leaked supervision: {forbidden_columns}")
    controls = _exact_control_columns(protocol)
    required = {
        "sample_index",
        "run_index",
        "global_seed",
        "class_slot",
        "class_id",
        "trace_dir",
        "endpoint_png_path",
        "cohort_role",
        "A_posterior_logstd_concentration_jump",
        "B_withheld_channel_predx0_cusum",
        "old_fixed_predicted_clean_score_control",
        "z_A_low_is_bad",
        "z_B_high_is_bad",
        "S_INTERSECTION",
        "S_UNION",
        PRIMARY_ALERT_010,
        PRIMARY_ALERT_005,
    }
    for control in controls:
        required.update(
            {
                control,
                f"{control}_trigger_alpha0p10",
                f"{control}_trigger_alpha0p05",
            }
        )
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"alert table lacks frozen columns: {missing}")
    _expected_cartesian_frame(frame, "fresh alert table")
    if not frame["cohort_role"].eq("inferential_evaluation").all():
        raise RuntimeError("alert product contains non-evaluation cohort rows")
    expected_slots = {class_id: index for index, class_id in enumerate(EVALUATION_CLASSES)}
    if any(
        int(row.class_slot) != expected_slots[int(row.class_id)]
        for row in frame[["class_slot", "class_id"]].itertuples(index=False)
    ):
        raise RuntimeError("alert class slots differ from the frozen class order")
    numeric_columns = [
        "sample_index",
        "run_index",
        "global_seed",
        "class_slot",
        "class_id",
        "A_posterior_logstd_concentration_jump",
        "B_withheld_channel_predx0_cusum",
        "old_fixed_predicted_clean_score_control",
        "z_A_low_is_bad",
        "z_B_high_is_bad",
        "S_INTERSECTION",
        "S_UNION",
        *controls,
    ]
    numeric = frame[numeric_columns].apply(pd.to_numeric, errors="raise").to_numpy(float)
    if not np.isfinite(numeric).all():
        raise RuntimeError("alert product contains non-finite numeric values")

    references = protocol["normalization"]["class_reference"]
    expected_z_a = np.empty(len(frame), dtype=np.float64)
    expected_z_b = np.empty(len(frame), dtype=np.float64)
    for class_id in EVALUATION_CLASSES:
        mask = frame["class_id"].to_numpy(int) == class_id
        stats = references[str(class_id)]["statistics"]
        a = frame.loc[mask, "A_posterior_logstd_concentration_jump"].to_numpy(float)
        b = frame.loc[mask, "B_withheld_channel_predx0_cusum"].to_numpy(float)
        expected_z_a[mask] = (
            -a - float(stats["A_low_is_bad"]["median"])
        ) / float(stats["A_low_is_bad"]["scale"])
        expected_z_b[mask] = (
            b - float(stats["B_high_is_bad"]["median"])
        ) / float(stats["B_high_is_bad"]["scale"])
    observed_z_a = frame["z_A_low_is_bad"].to_numpy(float)
    observed_z_b = frame["z_B_high_is_bad"].to_numpy(float)
    _assert_close(observed_z_a, expected_z_a, "z_A")
    _assert_close(observed_z_b, expected_z_b, "z_B")
    _assert_close(
        frame["S_UNION"].to_numpy(float),
        np.maximum(observed_z_a, observed_z_b),
        "S_UNION",
    )
    _assert_close(
        frame["S_INTERSECTION"].to_numpy(float),
        np.minimum(observed_z_a, observed_z_b),
        "S_INTERSECTION",
    )

    for class_id in EVALUATION_CLASSES:
        mask = frame["class_id"].to_numpy(int) == class_id
        class_scores = frame.loc[mask, "S_UNION"].to_numpy(float)
        class_thresholds = calibration["thresholds"][str(class_id)]
        expected_010 = class_scores > float(class_thresholds["alpha_0p10"]["threshold"])
        expected_005 = class_scores > float(class_thresholds["alpha_0p05"]["threshold"])
        observed_010 = _coerce_boolean_series(frame.loc[mask, PRIMARY_ALERT_010], PRIMARY_ALERT_010)
        observed_005 = _coerce_boolean_series(frame.loc[mask, PRIMARY_ALERT_005], PRIMARY_ALERT_005)
        if not np.array_equal(observed_010, expected_010) or not np.array_equal(
            observed_005, expected_005
        ):
            raise RuntimeError(f"conformal alerts do not replay for class {class_id}")
    thresholds = protocol["negative_controls"]["exact_e_value_alert_log_thresholds"]
    for control in controls:
        values = frame[control].to_numpy(float)
        for suffix, threshold_key in (
            ("alpha0p10", "alpha_0p10"),
            ("alpha0p05", "alpha_0p05"),
        ):
            name = f"{control}_trigger_{suffix}"
            observed = _coerce_boolean_series(frame[name], name)
            expected = values >= float(thresholds[threshold_key])
            if not np.array_equal(observed, expected):
                raise RuntimeError(f"exact-LR trigger does not replay: {name}")
            frame[name] = observed
    frame[PRIMARY_ALERT_010] = _coerce_boolean_series(frame[PRIMARY_ALERT_010], PRIMARY_ALERT_010)
    frame[PRIMARY_ALERT_005] = _coerce_boolean_series(frame[PRIMARY_ALERT_005], PRIMARY_ALERT_005)
    frame = frame.sort_values(["global_seed", "class_id"], kind="mergesort").reset_index(drop=True)
    return frame, summary, lock


def join_scores_after_label_lock(
    scores: pd.DataFrame, labels: pd.DataFrame
) -> pd.DataFrame:
    _expected_cartesian_frame(scores, "score side of score-label join")
    _expected_cartesian_frame(labels, "label side of score-label join")
    joined = scores.merge(
        labels,
        on=["global_seed", "class_id"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if len(joined) != EXPECTED_EVALUATION_ROWS or not joined["_merge"].eq("both").all():
        raise RuntimeError("score-label join lost or multiplied evaluation rows")
    joined = joined.drop(columns="_merge")
    endpoint_hashes: list[str] = []
    for path_text in joined["endpoint_png_path"]:
        path = Path(str(path_text)).expanduser().absolute()
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"score endpoint is missing or symlinked: {path}")
        endpoint_hashes.append(sha256_file(path))
    joined["score_endpoint_file_sha256"] = endpoint_hashes
    if not joined["score_endpoint_file_sha256"].eq(
        joined["native_image_file_sha256"]
    ).all():
        bad = joined.loc[
            ~joined["score_endpoint_file_sha256"].eq(joined["native_image_file_sha256"]),
            ["sample_key", "score_endpoint_file_sha256", "native_image_file_sha256"],
        ]
        raise RuntimeError(
            "labels and scores refer to different endpoint bytes: "
            + bad.head(5).to_json(orient="records")
        )
    _expected_cartesian_frame(joined, "joined evaluation table")
    return joined.sort_values(["global_seed", "class_id"], kind="mergesort").reset_index(drop=True)


def _midranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("midranks require a finite one-dimensional array")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    return ranks


def binary_auc(scores: np.ndarray, is_bad: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    is_bad = np.asarray(is_bad, dtype=bool)
    if scores.ndim != 1 or is_bad.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("invalid AUC arrays")
    n_bad = int(is_bad.sum())
    n_good = int((~is_bad).sum())
    if n_bad == 0 or n_good == 0:
        raise ValueError("AUC requires both labels")
    ranks = _midranks(scores)
    u = float(ranks[is_bad].sum() - n_bad * (n_bad + 1) / 2.0)
    return u / (n_bad * n_good)


def continuous_auc_summary(
    frame: pd.DataFrame, score_column: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    binary = frame.loc[frame["binary_primary_included"]].copy()
    class_rows: list[dict[str, Any]] = []
    numerator = 0.0
    denominator = 0
    aucs: list[float] = []
    for class_id in EVALUATION_CLASSES:
        group = binary.loc[binary["class_id"] == class_id]
        is_bad = group["primary_label"].eq(LABEL_BAD).to_numpy(bool)
        n_bad = int(is_bad.sum())
        n_good = int((~is_bad).sum())
        auc: float | None = None
        pair_count = n_bad * n_good
        if pair_count:
            auc = binary_auc(group[score_column].to_numpy(float), is_bad)
            numerator += auc * pair_count
            denominator += pair_count
            aucs.append(auc)
        class_rows.append(
            {
                "score": score_column,
                "class_id": class_id,
                "n_clear_bad": n_bad,
                "n_clean_good": n_good,
                "bad_good_pair_count": pair_count,
                "auc_higher_is_bad": auc,
            }
        )
    return (
        {
            "score": score_column,
            "orientation": "higher_is_bad",
            "eligible_class_count": len(aucs),
            "bad_good_pair_count": denominator,
            "class_matched_pair_weighted_auc": (
                numerator / denominator if denominator else None
            ),
            "macro_within_class_auc": float(np.mean(aucs)) if aucs else None,
        },
        class_rows,
    )


def primary_bad_ranks(frame: pd.DataFrame) -> list[dict[str, Any]]:
    binary = frame.loc[frame["binary_primary_included"]].copy()
    rows: list[dict[str, Any]] = []
    for class_id in EVALUATION_CLASSES:
        group = binary.loc[binary["class_id"] == class_id]
        all_scores = group[PRIMARY_SCORE].to_numpy(float)
        good_scores = group.loc[group["primary_label"] == LABEL_GOOD, PRIMARY_SCORE].to_numpy(float)
        for row in group.loc[group["primary_label"] == LABEL_BAD].itertuples(index=False):
            score = float(getattr(row, PRIMARY_SCORE))
            descending_midrank = (
                1.0
                + float(np.sum(all_scores > score))
                + 0.5 * float(np.sum(all_scores == score) - 1)
            )
            clean_win_fraction = (
                float(np.mean((score > good_scores) + 0.5 * (score == good_scores)))
                if len(good_scores)
                else None
            )
            rows.append(
                {
                    "sample_key": row.sample_key,
                    "global_seed": int(row.global_seed),
                    "class_id": class_id,
                    "S_UNION": score,
                    "descending_midrank_within_binary_class_1_is_most_bad_like": descending_midrank,
                    "binary_class_size": int(len(group)),
                    "clean_good_comparison_count": int(len(good_scores)),
                    "clean_pair_win_fraction_ties_half": clean_win_fraction,
                }
            )
    return rows


def stratified_permutation_test(
    frame: pd.DataFrame,
    *,
    draws: int = PERMUTATION_DRAWS,
    seed: int = PERMUTATION_SEED,
    chunk_size: int = 2048,
) -> dict[str, Any]:
    binary = frame.loc[frame["binary_primary_included"]].copy()
    groups: list[dict[str, Any]] = []
    observed_numerator = 0.0
    total_pairs = 0
    for class_id in EVALUATION_CLASSES:
        group = binary.loc[binary["class_id"] == class_id]
        labels = group["primary_label"].eq(LABEL_BAD).to_numpy(bool)
        n_bad = int(labels.sum())
        n_good = int((~labels).sum())
        if n_bad == 0 or n_good == 0:
            continue
        ranks = _midranks(group[PRIMARY_SCORE].to_numpy(float))
        observed_u = float(ranks[labels].sum() - n_bad * (n_bad + 1) / 2.0)
        observed_numerator += observed_u
        total_pairs += n_bad * n_good
        groups.append(
            {
                "class_id": class_id,
                "ranks": ranks,
                "n_bad": n_bad,
                "n_good": n_good,
            }
        )
    if not groups or total_pairs == 0:
        return {
            "available": False,
            "reason": "no class contains both clear_bad and clean_good",
            "draws": 0,
        }
    rng = np.random.Generator(np.random.PCG64(seed))
    null_auc = np.empty(draws, dtype=np.float64)
    offset = 0
    count_ge = 0
    while offset < draws:
        count = min(chunk_size, draws - offset)
        permuted_numerator = np.zeros(count, dtype=np.float64)
        for item in groups:
            ranks = item["ranks"]
            n_bad = item["n_bad"]
            # IID continuous random keys give an exactly uniform random subset
            # (up to zero-probability floating ties), hence a within-class label
            # permutation preserving the observed bad count.
            keys = rng.random((count, len(ranks)))
            selected = np.argpartition(keys, n_bad - 1, axis=1)[:, :n_bad]
            u = ranks[selected].sum(axis=1) - n_bad * (n_bad + 1) / 2.0
            permuted_numerator += u
        values = permuted_numerator / total_pairs
        null_auc[offset : offset + count] = values
        count_ge += int(np.sum(permuted_numerator >= observed_numerator))
        offset += count
    observed_auc = observed_numerator / total_pairs
    return {
        "available": True,
        "method": "within-class label permutation preserving class bad/good counts",
        "statistic": "class-matched pair-weighted AUC of S_UNION",
        "orientation": "higher_is_bad",
        "sidedness": "one-sided_greater_or_equal",
        "draws": draws,
        "rng": f"numpy.random.Generator(PCG64(seed={seed}))",
        "observed_auc": observed_auc,
        "permuted_greater_or_equal_count": count_ge,
        "p_value_add_one": (1.0 + count_ge) / (draws + 1.0),
        "null_auc_mean": float(np.mean(null_auc)),
        "null_auc_standard_deviation": float(np.std(null_auc, ddof=1)),
        "null_auc_draws_float64_sha256": sha256_bytes(
            np.asarray(null_auc, dtype="<f8").tobytes(order="C")
        ),
    }


def clopper_pearson_interval(
    successes: int, trials: int, confidence: float = CONFIDENCE_LEVEL
) -> tuple[float, float]:
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("invalid binomial counts")
    tail = (1.0 - confidence) / 2.0
    lower = 0.0 if successes == 0 else float(beta.ppf(tail, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(1.0 - tail, successes + 1, trials - successes))
    return lower, upper


def operating_point_rows(
    frame: pd.DataFrame, alert_columns: Sequence[tuple[str, str, str]]
) -> list[dict[str, Any]]:
    binary = frame.loc[frame["binary_primary_included"]].copy()
    is_bad = binary["primary_label"].eq(LABEL_BAD).to_numpy(bool)
    rows: list[dict[str, Any]] = []
    for name, family, alpha in alert_columns:
        alert = _coerce_boolean_series(binary[name], name)
        bad_hits = int(np.sum(alert & is_bad))
        bad_total = int(is_bad.sum())
        good_hits = int(np.sum(alert & ~is_bad))
        good_total = int((~is_bad).sum())
        tpr_interval = clopper_pearson_interval(bad_hits, bad_total) if bad_total else None
        fpr_interval = clopper_pearson_interval(good_hits, good_total) if good_total else None
        tpr = bad_hits / bad_total if bad_total else None
        fpr = good_hits / good_total if good_total else None
        rows.append(
            {
                "alert_column": name,
                "family": family,
                "alpha_nominal": alpha,
                "clear_bad_alert_count": bad_hits,
                "clear_bad_count": bad_total,
                "TPR": tpr,
                "TPR_CP95_low": tpr_interval[0] if tpr_interval else None,
                "TPR_CP95_high": tpr_interval[1] if tpr_interval else None,
                "clean_good_alert_count": good_hits,
                "clean_good_count": good_total,
                "FPR": fpr,
                "FPR_CP95_low": fpr_interval[0] if fpr_interval else None,
                "FPR_CP95_high": fpr_interval[1] if fpr_interval else None,
                "TPR_minus_FPR": (tpr - fpr) if tpr is not None and fpr is not None else None,
            }
        )
    return rows


def cluster_bootstrap_tpr_minus_fpr(
    frame: pd.DataFrame,
    alert_columns: Sequence[str],
    *,
    draws: int = CLUSTER_BOOTSTRAP_DRAWS,
    seed: int = CLUSTER_BOOTSTRAP_SEED,
    chunk_size: int = 2048,
) -> dict[str, dict[str, Any]]:
    seed_to_slot = {value: index for index, value in enumerate(EVALUATION_SEEDS)}
    n_seed = len(EVALUATION_SEEDS)
    n_metric = len(alert_columns)
    bad_count = np.zeros(n_seed, dtype=np.int64)
    good_count = np.zeros(n_seed, dtype=np.int64)
    bad_alerts = np.zeros((n_seed, n_metric), dtype=np.int64)
    good_alerts = np.zeros((n_seed, n_metric), dtype=np.int64)
    for row in frame.itertuples(index=False):
        if not bool(row.binary_primary_included):
            continue
        slot = seed_to_slot[int(row.global_seed)]
        is_bad = row.primary_label == LABEL_BAD
        if is_bad:
            bad_count[slot] += 1
        else:
            good_count[slot] += 1
        for metric_index, name in enumerate(alert_columns):
            triggered = bool(getattr(row, name))
            if is_bad:
                bad_alerts[slot, metric_index] += int(triggered)
            else:
                good_alerts[slot, metric_index] += int(triggered)
    rng = np.random.Generator(np.random.PCG64(seed))
    differences = np.full((draws, n_metric), np.nan, dtype=np.float64)
    offset = 0
    while offset < draws:
        count = min(chunk_size, draws - offset)
        sampled = rng.integers(0, n_seed, size=(count, n_seed))
        sampled_bad_count = bad_count[sampled].sum(axis=1)
        sampled_good_count = good_count[sampled].sum(axis=1)
        valid = (sampled_bad_count > 0) & (sampled_good_count > 0)
        for metric_index in range(n_metric):
            bad_hits = bad_alerts[sampled, metric_index].sum(axis=1)
            good_hits = good_alerts[sampled, metric_index].sum(axis=1)
            values = np.full(count, np.nan, dtype=np.float64)
            values[valid] = (
                bad_hits[valid] / sampled_bad_count[valid]
                - good_hits[valid] / sampled_good_count[valid]
            )
            differences[offset : offset + count, metric_index] = values
        offset += count
    results: dict[str, dict[str, Any]] = {}
    for metric_index, name in enumerate(alert_columns):
        valid_values = differences[:, metric_index]
        valid_values = valid_values[np.isfinite(valid_values)]
        if not len(valid_values):
            results[name] = {
                "available": False,
                "draws": draws,
                "valid_draws": 0,
                "invalid_empty_label_draws": draws,
            }
            continue
        low, high = np.quantile(valid_values, [0.025, 0.975], method="linear")
        results[name] = {
            "available": True,
            "method": "percentile global-seed cluster bootstrap retaining all three class rows",
            "draws": draws,
            "valid_draws": int(len(valid_values)),
            "invalid_empty_label_draws": int(draws - len(valid_values)),
            "rng": f"numpy.random.Generator(PCG64(seed={seed}))",
            "TPR_minus_FPR_bootstrap95_low": float(low),
            "TPR_minus_FPR_bootstrap95_high": float(high),
            "valid_draws_float64_sha256": sha256_bytes(
                np.asarray(valid_values, dtype="<f8").tobytes(order="C")
            ),
        }
    return results


def stratified_auc_bootstrap(
    frame: pd.DataFrame,
    *,
    draws: int = AUC_BOOTSTRAP_DRAWS,
    seed: int = AUC_BOOTSTRAP_SEED,
    chunk_size: int = 512,
) -> dict[str, Any]:
    binary = frame.loc[frame["binary_primary_included"]].copy()
    groups: list[dict[str, Any]] = []
    for class_id in EVALUATION_CLASSES:
        group = binary.loc[binary["class_id"] == class_id]
        bad = group.loc[group["primary_label"] == LABEL_BAD, PRIMARY_SCORE].to_numpy(float)
        good = group.loc[group["primary_label"] == LABEL_GOOD, PRIMARY_SCORE].to_numpy(float)
        if len(bad) and len(good):
            groups.append(
                {
                    "class_id": class_id,
                    "bad": bad,
                    "good": good,
                    "weight": len(bad) * len(good),
                }
            )
    if not groups:
        return {
            "available": False,
            "reason": "no class contains both clear_bad and clean_good",
            "draws": 0,
        }
    total_weight = sum(item["weight"] for item in groups)
    pair_weighted = np.empty(draws, dtype=np.float64)
    macro = np.empty(draws, dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(seed))
    offset = 0
    while offset < draws:
        count = min(chunk_size, draws - offset)
        class_aucs = np.empty((count, len(groups)), dtype=np.float64)
        for class_index, item in enumerate(groups):
            bad = item["bad"]
            good = item["good"]
            sampled_bad = bad[rng.integers(0, len(bad), size=(count, len(bad)))]
            sampled_good = good[rng.integers(0, len(good), size=(count, len(good)))]
            wins = sampled_bad[:, :, None] > sampled_good[:, None, :]
            ties = sampled_bad[:, :, None] == sampled_good[:, None, :]
            class_aucs[:, class_index] = np.mean(wins + 0.5 * ties, axis=(1, 2))
        weights = np.asarray([item["weight"] for item in groups], dtype=np.float64)
        pair_weighted[offset : offset + count] = class_aucs @ weights / total_weight
        macro[offset : offset + count] = class_aucs.mean(axis=1)
        offset += count
    pair_low, pair_high = np.quantile(pair_weighted, [0.025, 0.975], method="linear")
    macro_low, macro_high = np.quantile(macro, [0.025, 0.975], method="linear")
    return {
        "available": True,
        "method": "percentile bootstrap resampling independently within each (class,label) stratum",
        "score": PRIMARY_SCORE,
        "draws": draws,
        "rng": f"numpy.random.Generator(PCG64(seed={seed}))",
        "eligible_classes": [item["class_id"] for item in groups],
        "class_matched_pair_weighted_auc_bootstrap95_low": float(pair_low),
        "class_matched_pair_weighted_auc_bootstrap95_high": float(pair_high),
        "macro_within_class_auc_bootstrap95_low": float(macro_low),
        "macro_within_class_auc_bootstrap95_high": float(macro_high),
        "pair_weighted_draws_float64_sha256": sha256_bytes(
            np.asarray(pair_weighted, dtype="<f8").tobytes(order="C")
        ),
        "macro_draws_float64_sha256": sha256_bytes(
            np.asarray(macro, dtype="<f8").tobytes(order="C")
        ),
    }


def _score_specs(protocol: dict[str, Any]) -> list[dict[str, str]]:
    specs = [
        {"score": "S_UNION", "role": "primary"},
        {"score": "z_A_low_is_bad", "role": "single_feature_A_mechanism"},
        {"score": "z_B_high_is_bad", "role": "single_feature_B_mechanism"},
        {"score": "S_INTERSECTION", "role": "retired_descriptive_subtype_control"},
        {
            "score": "old_fixed_predicted_clean_score_control",
            "role": "old_fixed_score_negative_control",
        },
    ]
    specs.extend(
        {"score": name, "role": "exact_path_LR_negative_control"}
        for name in _exact_control_columns(protocol)
    )
    return specs


def _alert_specs(protocol: dict[str, Any]) -> list[tuple[str, str, str]]:
    specs = [
        (PRIMARY_ALERT_010, "S_UNION_split_conformal", "0.10"),
        (PRIMARY_ALERT_005, "S_UNION_split_conformal", "0.05"),
    ]
    for control in _exact_control_columns(protocol):
        specs.extend(
            [
                (f"{control}_trigger_alpha0p10", "exact_path_LR_negative_control", "0.10"),
                (f"{control}_trigger_alpha0p05", "exact_path_LR_negative_control", "0.05"),
            ]
        )
    return specs


def evaluate_statistics(
    joined: pd.DataFrame,
    protocol: dict[str, Any],
    *,
    permutation_draws: int = PERMUTATION_DRAWS,
    cluster_draws: int = CLUSTER_BOOTSTRAP_DRAWS,
    auc_draws: int = AUC_BOOTSTRAP_DRAWS,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    score_summaries: list[dict[str, Any]] = []
    class_metric_rows: list[dict[str, Any]] = []
    for spec in _score_specs(protocol):
        summary, per_class = continuous_auc_summary(joined, spec["score"])
        summary["role"] = spec["role"]
        score_summaries.append(summary)
        for row in per_class:
            row["role"] = spec["role"]
            class_metric_rows.append(row)

    permutation = stratified_permutation_test(
        joined, draws=permutation_draws, seed=PERMUTATION_SEED
    )
    auc_bootstrap = stratified_auc_bootstrap(
        joined, draws=auc_draws, seed=AUC_BOOTSTRAP_SEED
    )
    alert_specs = _alert_specs(protocol)
    operating_rows = operating_point_rows(joined, alert_specs)
    cluster = cluster_bootstrap_tpr_minus_fpr(
        joined,
        [item[0] for item in alert_specs],
        draws=cluster_draws,
        seed=CLUSTER_BOOTSTRAP_SEED,
    )
    for row in operating_rows:
        interval = cluster[row["alert_column"]]
        row.update(
            {
                key: value
                for key, value in interval.items()
                if key.startswith("TPR_minus_FPR_bootstrap95_")
                or key in {"valid_draws", "invalid_empty_label_draws"}
            }
        )

    primary = next(row for row in score_summaries if row["score"] == PRIMARY_SCORE)
    primary_classes = [
        row for row in class_metric_rows if row["score"] == PRIMARY_SCORE
    ]
    primary_operating = next(
        row for row in operating_rows if row["alert_column"] == PRIMARY_ALERT_010
    )
    n_bad = int(joined["primary_label"].eq(LABEL_BAD).sum())
    n_good = int(joined["primary_label"].eq(LABEL_GOOD).sum())
    n_excluded = int(joined["primary_label"].eq(LABEL_EXCLUDED).sum())
    gate_spec = protocol["evaluation"]["initial_go_gate"]
    low_classes = [
        {
            "class_id": int(row["class_id"]),
            "n_clear_bad": int(row["n_clear_bad"]),
            "auc": row["auc_higher_is_bad"],
        }
        for row in primary_classes
        if row["n_clear_bad"] >= 2
        and (
            row["auc_higher_is_bad"] is None
            or row["auc_higher_is_bad"]
            < gate_spec["no_class_with_two_or_more_bad_events_has_auc_below"]
        )
    ]
    criteria = {
        "minimum_clear_bad_events_for_decision": n_bad
        >= gate_spec["minimum_clear_bad_events_for_decision"],
        "S_UNION_class_matched_auc_at_least_0p75": (
            primary["class_matched_pair_weighted_auc"] is not None
            and primary["class_matched_pair_weighted_auc"]
            >= gate_spec["S_UNION_class_matched_auc_at_least"]
        ),
        "S_UNION_permutation_one_sided_p_below_0p05": (
            permutation.get("available") is True
            and permutation["p_value_add_one"]
            < gate_spec["S_UNION_stratified_permutation_one_sided_p_below"]
        ),
        "alpha0p10_TPR_minus_FPR_point_above_zero": (
            primary_operating["TPR_minus_FPR"] is not None
            and primary_operating["TPR_minus_FPR"]
            > gate_spec["alpha_0p10_TPR_minus_FPR_point_above"]
        ),
        "no_class_with_at_least_two_bad_has_auc_below_0p60": not low_classes,
    }
    if n_bad < gate_spec["minimum_clear_bad_events_for_decision"]:
        decision = "PILOT_ONLY_EXPAND_DISJOINT_SEEDS_WITHOUT_CHANGING_FORMULAS"
        gate_evaluated = False
        gate_passed: bool | None = None
    else:
        gate_evaluated = True
        gate_passed = all(criteria.values())
        decision = "INITIAL_GO" if gate_passed else "CONFIRMATION_GATE_FAILED"

    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE_PROSPECTIVE_CONFIRMATION_EVALUATION",
        "orientation_contract": "all continuous AUCs use higher-is-more-bad-like; z_A already reverses raw A",
        "cohort": {
            "classes": list(EVALUATION_CLASSES),
            "seeds": list(EVALUATION_SEEDS),
            "trajectory_count": EXPECTED_EVALUATION_ROWS,
            "clear_bad_count": n_bad,
            "clean_good_count": n_good,
            "mild_or_disputed_excluded_count": n_excluded,
            "binary_metric_row_count": n_bad + n_good,
        },
        "primary_score": primary,
        "continuous_scores": score_summaries,
        "primary_randomization_test": permutation,
        "primary_auc_uncertainty": auc_bootstrap,
        "operating_points": operating_rows,
        "cluster_bootstrap_details": cluster,
        "initial_go_gate": {
            "minimum_bad_gate": gate_spec["minimum_clear_bad_events_for_decision"],
            "evaluated": gate_evaluated,
            "criteria": criteria,
            "classes_failing_minimum_auc_guardrail": low_classes,
            "passed": gate_passed,
            "decision": decision,
            "pilot_rule_if_under_15_bad": protocol["evaluation"][
                "interpretation_if_fewer_than_15_bad_events"
            ],
        },
        "claim_limits": {
            "calibration_alpha_is_not_clean_good_conditional_FPR": True,
            "S_INTERSECTION_is_descriptive_not_a_gate": True,
            "A_B_and_negative_controls_do_not_enter_the_gate": True,
            "time_to_signal_is_exploratory_and_not_evaluated_here": True,
            "cross_class_or_cross_model_generalization_not_established": True,
        },
        "randomness_contract": {
            "permutation": {"draws": permutation_draws, "seed": PERMUTATION_SEED},
            "global_seed_cluster_bootstrap": {
                "draws": cluster_draws,
                "seed": CLUSTER_BOOTSTRAP_SEED,
            },
            "within_class_label_auc_bootstrap": {
                "draws": auc_draws,
                "seed": AUC_BOOTSTRAP_SEED,
            },
        },
    }
    tables = {
        "continuous_score_metrics.csv": pd.DataFrame(score_summaries),
        "per_class_score_metrics.csv": pd.DataFrame(class_metric_rows),
        "primary_bad_ranks.csv": pd.DataFrame(primary_bad_ranks(joined)),
        "operating_point_metrics.csv": pd.DataFrame(operating_rows),
    }
    return result, tables


def publish(
    candidate_lock: Path,
    calibration_lock: Path,
    alerts_root: Path,
    consensus_lock: Path,
    output: Path,
) -> Path:
    # Critical ordering: no fresh score CSV (including the calibration-score
    # member that is hash-checked by validate_calibration_lock) is opened before
    # this consensus validation returns successfully.
    protocol = validate_candidate_lock(candidate_lock)
    labels, consensus, consensus_directory = validate_consensus_lock(
        consensus_lock, protocol
    )
    calibration, calibration_directory = validate_calibration_lock(
        calibration_lock, protocol
    )
    scores, alert_summary, alert_directory = validate_alert_product(
        alerts_root, protocol, calibration
    )
    joined = join_scores_after_label_lock(scores, labels)
    result, tables = evaluate_statistics(joined, protocol)
    result["input_lineage"] = {
        "candidate_protocol_identity_sha256": protocol["identity_sha256"],
        "candidate_manifest_identity_sha256": EXPECTED_CANDIDATE_MANIFEST_IDENTITY,
        "calibration_identity_sha256": calibration["identity_sha256"],
        "calibration_manifest_identity_sha256": calibration_directory.manifest_identity,
        "blind_consensus_identity_sha256": consensus["identity_sha256"],
        "blind_consensus_manifest_identity_sha256": consensus_directory.manifest_identity,
        "blind_review_pack_identity_sha256": consensus.get(
            "blind_review_pack_identity_sha256",
            consensus.get(
                "blind_pack_identity_sha256",
                consensus_directory.manifest.get(
                    "blind_review_pack_identity_sha256",
                    consensus_directory.manifest.get("blind_pack_identity_sha256"),
                ),
            ),
        ),
        "alert_manifest_identity_sha256": alert_directory.manifest_identity,
        "candidate_score_manifest_identity_sha256": alert_summary[
            "candidate_score_manifest_identity_sha256"
        ],
        "validation_order": [
            "candidate_protocol_lock",
            "blind_consensus_lock",
            "label_free_calibration_lock",
            "label_free_evaluation_alert_product",
            "one_to_one_score_label_join",
        ],
        "consensus_lock_validated_before_any_fresh_score_csv_open": True,
        "fresh_score_label_join_validate": "one_to_one_exact_240_row_cartesian",
        "endpoint_file_sha256_matched_for_every_joined_row": True,
    }
    result["identity_sha256"] = canonical_sha256(result)

    output = output.expanduser().absolute()
    if os.path.lexists(output):
        raise RuntimeError(f"refusing to overwrite confirmation evaluation: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        write_json(staging / "confirmation_results.json", result)
        joined.to_csv(staging / "joined_evaluation_rows_locked.csv", index=False)
        for name, frame in tables.items():
            frame.to_csv(staging / name, index=False)
        shutil.copy2(Path(__file__).resolve(), staging / "evaluator_source.py")
        members = [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(staging.iterdir())
        ]
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "complete",
            "candidate_protocol_identity_sha256": protocol["identity_sha256"],
            "calibration_identity_sha256": calibration["identity_sha256"],
            "consensus_identity_sha256": consensus["identity_sha256"],
            "alert_manifest_identity_sha256": alert_directory.manifest_identity,
            "result_identity_sha256": result["identity_sha256"],
            "files": members,
        }
        manifest["identity_sha256"] = canonical_sha256(manifest)
        write_json(staging / "manifest.json", manifest)
        completion = {
            "complete": True,
            "manifest_file_sha256": sha256_file(staging / "manifest.json"),
            "manifest_identity_sha256": manifest["identity_sha256"],
            "result_file_sha256": sha256_file(staging / "confirmation_results.json"),
            "result_identity_sha256": result["identity_sha256"],
            "joined_rows_file_sha256": sha256_file(
                staging / "joined_evaluation_rows_locked.csv"
            ),
            "joined_row_count": EXPECTED_EVALUATION_ROWS,
        }
        write_json(staging / "completion.json", completion)
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def synthetic_self_test() -> dict[str, Any]:
    scores = np.asarray([0.0, 1.0, 1.0, 2.0])
    labels = np.asarray([False, True, False, True])
    # P(1 > 0) + .5 P(1 == 1) + P(2 > 0) + P(2 > 1), divided by 4.
    if binary_auc(scores, labels) != 0.875:
        raise AssertionError("tie-aware AUC self-test failed")
    if _midranks(scores).tolist() != [1.0, 2.5, 2.5, 4.0]:
        raise AssertionError("midrank self-test failed")
    low0, high0 = clopper_pearson_interval(0, 10)
    low10, high10 = clopper_pearson_interval(10, 10)
    if low0 != 0.0 or high10 != 1.0 or not (0.0 < high0 < 1.0) or not (0.0 < low10 < 1.0):
        raise AssertionError("Clopper-Pearson boundary self-test failed")

    # Synthetic 3 x 80 cohort.  Scores and labels are mathematical fixtures,
    # not reads from any prospective product.
    rows: list[dict[str, Any]] = []
    for seed in EVALUATION_SEEDS:
        for class_id in EVALUATION_CLASSES:
            bad = (seed + class_id) % 17 == 0
            excluded = (seed + 2 * class_id) % 29 == 0 and not bad
            label = LABEL_BAD if bad else LABEL_EXCLUDED if excluded else LABEL_GOOD
            score = float((seed % 11) / 5.0 + (1.25 if bad else 0.0))
            row: dict[str, Any] = {
                "global_seed": seed,
                "class_id": class_id,
                "sample_key": f"class{class_id:04d}_seed{seed:03d}",
                "primary_label": label,
                "binary_primary_included": label in {LABEL_BAD, LABEL_GOOD},
                "S_UNION": score,
                "z_A_low_is_bad": score - 0.1,
                "z_B_high_is_bad": score,
                "S_INTERSECTION": score - 0.1,
                "old_fixed_predicted_clean_score_control": (seed % 7) / 7.0,
                PRIMARY_ALERT_010: score > 2.0,
                PRIMARY_ALERT_005: score > 2.5,
            }
            rows.append(row)
    frame = pd.DataFrame(rows)
    _expected_cartesian_frame(frame, "synthetic cohort")
    first = stratified_permutation_test(frame, draws=257, seed=PERMUTATION_SEED, chunk_size=31)
    second = stratified_permutation_test(frame, draws=257, seed=PERMUTATION_SEED, chunk_size=31)
    if first != second or not 0.0 < first["p_value_add_one"] <= 1.0:
        raise AssertionError("permutation determinism self-test failed")
    auc_first = stratified_auc_bootstrap(frame, draws=257, seed=AUC_BOOTSTRAP_SEED, chunk_size=37)
    auc_second = stratified_auc_bootstrap(frame, draws=257, seed=AUC_BOOTSTRAP_SEED, chunk_size=37)
    if auc_first != auc_second:
        raise AssertionError("AUC bootstrap determinism self-test failed")
    cluster_first = cluster_bootstrap_tpr_minus_fpr(
        frame,
        [PRIMARY_ALERT_010, PRIMARY_ALERT_005],
        draws=257,
        seed=CLUSTER_BOOTSTRAP_SEED,
        chunk_size=41,
    )
    cluster_second = cluster_bootstrap_tpr_minus_fpr(
        frame,
        [PRIMARY_ALERT_010, PRIMARY_ALERT_005],
        draws=257,
        seed=CLUSTER_BOOTSTRAP_SEED,
        chunk_size=41,
    )
    if cluster_first != cluster_second:
        raise AssertionError("cluster bootstrap determinism self-test failed")
    synthetic_protocol = {
        "negative_controls": {"exact_path_evidence_running_maxima": []},
        "evaluation": {
            "initial_go_gate": {
                "minimum_clear_bad_events_for_decision": 15,
                "S_UNION_class_matched_auc_at_least": 0.75,
                "S_UNION_stratified_permutation_one_sided_p_below": 0.05,
                "alpha_0p10_TPR_minus_FPR_point_above": 0.0,
                "no_class_with_two_or_more_bad_events_has_auc_below": 0.6,
            },
            "interpretation_if_fewer_than_15_bad_events": (
                "pilot only; expand using disjoint seeds without changing formulas"
            ),
        },
    }
    synthetic_result, synthetic_tables = evaluate_statistics(
        frame,
        synthetic_protocol,
        permutation_draws=257,
        cluster_draws=257,
        auc_draws=257,
    )
    if (
        synthetic_result["cohort"]["clear_bad_count"] != 14
        or synthetic_result["initial_go_gate"]["evaluated"] is not False
        or synthetic_result["initial_go_gate"]["passed"] is not None
        or synthetic_result["initial_go_gate"]["decision"]
        != "PILOT_ONLY_EXPAND_DISJOINT_SEEDS_WITHOUT_CHANGING_FORMULAS"
        or len(synthetic_tables["per_class_score_metrics.csv"]) != 15
    ):
        raise AssertionError("under-15 prospective pilot-gate self-test failed")
    duplicated = pd.concat([frame.iloc[:-1], frame.iloc[[0]]], ignore_index=True)
    try:
        _expected_cartesian_frame(duplicated, "synthetic duplicate-negative-control")
    except RuntimeError:
        duplicate_rejected = True
    else:
        duplicate_rejected = False
    if not duplicate_rejected:
        raise AssertionError("Cartesian duplicate negative-control failed")
    with tempfile.TemporaryDirectory(prefix="dit-confirmation-evaluator-selftest-") as temporary:
        temporary_root = Path(temporary)
        endpoint = temporary_root / "synthetic_endpoint.bin"
        endpoint.write_bytes(b"synthetic endpoint bytes only\n")
        endpoint_hash = sha256_file(endpoint)
        consensus_root = temporary_root / "consensus_lock"
        consensus_root.mkdir()
        consensus_rows: list[dict[str, Any]] = []
        for row in frame.itertuples(index=False):
            if row.primary_label == LABEL_BAD:
                reviews = {"G": 2, "H": 2, "I": 0}
            elif row.primary_label == LABEL_GOOD:
                reviews = {"G": 0, "H": 0, "I": 1}
            else:
                reviews = {"G": 1, "H": 1, "I": 0}
            consensus_rows.append(
                {
                    "global_seed": int(row.global_seed),
                    "seed": int(row.global_seed),
                    "class_id": int(row.class_id),
                    "sample_key": row.sample_key,
                    "primary_label": row.primary_label,
                    "binary_primary_included": bool(row.binary_primary_included),
                    "review_scores": reviews,
                    "native_image": {"file_sha256": endpoint_hash},
                }
            )
        counts = {
            label: sum(row["primary_label"] == label for row in consensus_rows)
            for label in (LABEL_BAD, LABEL_GOOD, LABEL_EXCLUDED)
        }
        consensus_payload: dict[str, Any] = {
            "schema_version": 1,
            "status": "LOCKED_COMPLETE_BEFORE_ANY_FRESH_SCORE_JOIN",
            "candidate_protocol_identity_sha256": EXPECTED_CANDIDATE_PROTOCOL_IDENTITY,
            "blind_pack_identity_sha256": "a" * 64,
            "blinding_audit": {
                "reviewer_count": 3,
                "endpoint_only_review": True,
                "metric_values_visible_to_reviewers": False,
                "alert_decisions_visible_to_reviewers": False,
                "trajectories_visible_to_reviewers": False,
                "labels_locked_before_score_join": True,
            },
            "counts": counts,
            "rows": consensus_rows,
        }
        consensus_payload["identity_sha256"] = canonical_sha256(consensus_payload)
        consensus_path = consensus_root / "consensus_locked.json"
        write_json(consensus_path, consensus_payload)
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "complete",
            "candidate_protocol_identity_sha256": EXPECTED_CANDIDATE_PROTOCOL_IDENTITY,
            "blind_pack_identity_sha256": "a" * 64,
            "consensus_file_sha256": sha256_file(consensus_path),
            "consensus_identity_sha256": consensus_payload["identity_sha256"],
            "counts": counts,
            "files": [
                {
                    "name": "consensus_locked.json",
                    "bytes": consensus_path.stat().st_size,
                    "sha256": sha256_file(consensus_path),
                }
            ],
        }
        manifest["identity_sha256"] = canonical_sha256(manifest)
        manifest_path = consensus_root / "manifest.json"
        write_json(manifest_path, manifest)
        write_json(
            consensus_root / "completion.json",
            {
                "complete": True,
                "manifest_file_sha256": sha256_file(manifest_path),
                "manifest_identity_sha256": manifest["identity_sha256"],
                "consensus_file_sha256": sha256_file(consensus_path),
                "consensus_identity_sha256": consensus_payload["identity_sha256"],
                "locked_row_count": EXPECTED_EVALUATION_ROWS,
            },
        )
        locked_labels, _, _ = validate_consensus_lock(
            consensus_root,
            {"identity_sha256": EXPECTED_CANDIDATE_PROTOCOL_IDENTITY},
        )
        synthetic_scores = frame[["global_seed", "class_id"]].copy()
        synthetic_scores["endpoint_png_path"] = str(endpoint)
        joined_synthetic = join_scores_after_label_lock(synthetic_scores, locked_labels)
        if len(joined_synthetic) != EXPECTED_EVALUATION_ROWS:
            raise AssertionError("synthetic strict score-label join failed")
        mismatched_labels = locked_labels.copy()
        mismatched_labels.loc[0, "native_image_file_sha256"] = "b" * 64
        try:
            join_scores_after_label_lock(synthetic_scores, mismatched_labels)
        except RuntimeError:
            endpoint_mismatch_rejected = True
        else:
            endpoint_mismatch_rejected = False
        if not endpoint_mismatch_rejected:
            raise AssertionError("endpoint hash mismatch negative-control failed")
        consensus_path.write_text("{}\n", encoding="utf-8")
        try:
            validate_consensus_lock(
                consensus_root,
                {"identity_sha256": EXPECTED_CANDIDATE_PROTOCOL_IDENTITY},
            )
        except RuntimeError:
            lock_tamper_rejected = True
        else:
            lock_tamper_rejected = False
        if not lock_tamper_rejected:
            raise AssertionError("consensus lock tamper negative-control failed")
    return {
        "status": "synthetic_self_test_passed",
        "prospective_files_opened": False,
        "tie_aware_auc": 0.875,
        "synthetic_rows": len(frame),
        "permutation_draws": 257,
        "auc_bootstrap_draws": 257,
        "cluster_bootstrap_draws": 257,
        "duplicate_cartesian_rejected": True,
        "under_15_bad_pilot_gate": True,
        "strict_consensus_lock_validated": True,
        "endpoint_hash_mismatch_rejected": True,
        "consensus_lock_tamper_rejected": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-lock", type=Path, default=DEFAULT_CANDIDATE_LOCK)
    parser.add_argument("--calibration-lock", type=Path, default=DEFAULT_CALIBRATION_LOCK)
    parser.add_argument("--alerts-root", type=Path)
    parser.add_argument("--consensus-lock", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run only synthetic in-memory tests; never open prospective inputs",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        print(json.dumps(synthetic_self_test(), indent=2, sort_keys=True))
        return 0
    missing = [
        name
        for name, value in (
            ("--alerts-root", args.alerts_root),
            ("--consensus-lock", args.consensus_lock),
            ("--output", args.output),
        )
        if value is None
    ]
    if missing:
        raise SystemExit("missing required production arguments: " + ", ".join(missing))
    output = publish(
        args.candidate_lock,
        args.calibration_lock,
        args.alerts_root,
        args.consensus_lock,
        args.output,
    )
    print(json.dumps({"output": str(output), "status": "complete"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
