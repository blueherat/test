#!/usr/bin/env python3
"""Aggregate-only confirmation over original seeds50..129 plus expansion130..249.

Security/order contract
-----------------------
1. Validate candidate v5 and the *aggregate-only* original event receipt.
2. Validate the final expansion visual-label lock and count new clear-bad rows.
3. If 8 + new clear-bad < 15, publish only aggregate counts and return before
   opening the original row-label lock, calibration lock, or either score/alert
   product.
4. Only above the event gate, validate original row labels, immutable
   seeds30..49 calibration, and both label-free alert products; join in memory.
5. Publish aggregate statistics only.  Joined rows, per-sample ranks, scores,
   endpoints, hashes, labels, seeds, and alert decisions are never exported.

Candidate-v5 A/B/S_UNION, discovery normalizers, calibration thresholds,
100,000-draw within-class permutation test, RNG seeds, and all five initial-go
criteria remain unchanged.  Expansion is increased sample size, not a second
chance to edit the detector.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.stats import beta

try:
    from .apply_dit_bad_good_expansion_thresholds import validate_calibration_lock
    from .dit_bad_good_expansion_contract import (
        ALL_CONFIRMATION_SEEDS,
        AUC_BOOTSTRAP_DRAWS,
        AUC_BOOTSTRAP_SEED,
        CANDIDATE_LOCK,
        CANDIDATE_PROTOCOL_IDENTITY,
        CLASSES,
        CLUSTER_BOOTSTRAP_DRAWS,
        CLUSTER_BOOTSTRAP_SEED,
        EXPANSION_SEEDS,
        LABEL_BAD,
        LABEL_EXCLUDED,
        LABEL_GOOD,
        LABELS,
        MINIMUM_CLEAR_BAD_EVENTS,
        ORIGINAL_CLEAR_BAD_EVENTS,
        ORIGINAL_EVALUATION_SEEDS,
        PERMUTATION_DRAWS,
        PERMUTATION_SEED,
        PRIMARY_ALERT_005,
        PRIMARY_ALERT_010,
        PRIMARY_SCORE,
        REVIEWERS,
        canonical_sha256,
        load_json,
        require_canonical_identity,
        require_planned_path,
        sample_key,
        sha256_bytes,
        sha256_file,
        validate_candidate_lock,
        validate_expansion_lock,
        validate_pipeline_source_lock,
        write_json,
    )
except ImportError:  # pragma: no cover
    from apply_dit_bad_good_expansion_thresholds import validate_calibration_lock
    from dit_bad_good_expansion_contract import (
        ALL_CONFIRMATION_SEEDS,
        AUC_BOOTSTRAP_DRAWS,
        AUC_BOOTSTRAP_SEED,
        CANDIDATE_LOCK,
        CANDIDATE_PROTOCOL_IDENTITY,
        CLASSES,
        CLUSTER_BOOTSTRAP_DRAWS,
        CLUSTER_BOOTSTRAP_SEED,
        EXPANSION_SEEDS,
        LABEL_BAD,
        LABEL_EXCLUDED,
        LABEL_GOOD,
        LABELS,
        MINIMUM_CLEAR_BAD_EVENTS,
        ORIGINAL_CLEAR_BAD_EVENTS,
        ORIGINAL_EVALUATION_SEEDS,
        PERMUTATION_DRAWS,
        PERMUTATION_SEED,
        PRIMARY_ALERT_005,
        PRIMARY_ALERT_010,
        PRIMARY_SCORE,
        REVIEWERS,
        canonical_sha256,
        load_json,
        require_canonical_identity,
        require_planned_path,
        sample_key,
        sha256_bytes,
        sha256_file,
        validate_candidate_lock,
        validate_expansion_lock,
        validate_pipeline_source_lock,
        write_json,
    )


CONFIDENCE_LEVEL = 0.95
ORIGINAL_FINAL_STATUS = "FINAL_VISUAL_LABELS_LOCKED_BEFORE_ANY_LABEL_SCORE_JOIN"
EXPANSION_FINAL_STATUS = "FINAL_EXPANSION_VISUAL_LABELS_LOCKED_BEFORE_ANY_SCORE_JOIN"
ORIGINAL_ALERT_STATUS = "COMPLETE_LABEL_FREE_CALIBRATED_EVALUATION_ALERTS"
EXPANSION_ALERT_STATUS = "COMPLETE_LABEL_FREE_CALIBRATED_EXPANSION_ALERTS"
FORBIDDEN_ROW_KEYS = {
    "sample_key",
    "seed",
    "global_seed",
    "endpoint_png_path",
    "trace_dir",
    "native_image_file_sha256",
    "score_endpoint_file_sha256",
    "primary_label",
    "review_scores",
}


def _members(root: Path, manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    members = {str(item.get("name")): item for item in manifest.get("files", [])}
    if not members:
        raise RuntimeError(f"manifest has no members: {root}")
    for name, item in members.items():
        path = root / name
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != item.get("bytes")
            or sha256_file(path) != item.get("sha256")
        ):
            raise RuntimeError(f"manifest member changed: {path}")
    return members


def _assert_no_row_payload(value: Any) -> None:
    if isinstance(value, dict):
        leaked = FORBIDDEN_ROW_KEYS.intersection(value)
        if leaked:
            raise RuntimeError(f"aggregate output leaked row keys: {sorted(leaked)}")
        for child in value.values():
            _assert_no_row_payload(child)
    elif isinstance(value, list):
        if len(value) >= 80 and value and isinstance(value[0], dict):
            raise RuntimeError("aggregate output contains a row-like payload")
        for child in value:
            _assert_no_row_payload(child)


def validate_original_event_receipt(
    root: Path, protocol: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read only the already-published aggregate original-pilot receipt."""

    root = root.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"original event receipt must be a real directory: {root}")
    manifest_path = root / "manifest.json"
    result_path = root / "confirmation_results.json"
    manifest = load_json(manifest_path)
    completion = load_json(root / "completion.json")
    result = load_json(result_path)
    manifest_identity = require_canonical_identity(manifest, "original receipt manifest")
    result_identity = require_canonical_identity(result, "original event receipt")
    if (
        manifest.get("status") != "complete"
        or manifest.get("aggregate_only") is not True
        or completion.get("complete") is not True
        or completion.get("aggregate_only") is not True
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
        or completion.get("manifest_identity_sha256") != manifest_identity
        or completion.get("result_file_sha256") != sha256_file(result_path)
        or completion.get("result_identity_sha256") != result_identity
        or manifest.get("result_identity_sha256") != result_identity
        or result.get("status") != "EVENT_COUNT_ONLY_PILOT_EXPANSION_REQUIRED"
    ):
        raise RuntimeError("original aggregate event receipt is invalid")
    _members(root, manifest)
    _assert_no_row_payload(result)
    event_gate = result.get("event_gate", {})
    cohort = result.get("cohort", {})
    audit = result.get("evidence_access_audit", {})
    lineage = result.get("input_lineage", {})
    if (
        event_gate.get("minimum_clear_bad_events_for_decision")
        != MINIMUM_CLEAR_BAD_EVENTS
        or event_gate.get("observed_clear_bad_events") != ORIGINAL_CLEAR_BAD_EVENTS
        or event_gate.get("evaluated") is not False
        or cohort.get("classes") != list(CLASSES)
        or cohort.get("trajectory_count") != 240
        or cohort.get("aggregate_label_counts", {}).get(LABEL_BAD)
        != ORIGINAL_CLEAR_BAD_EVENTS
        or lineage.get("candidate_protocol_identity_sha256")
        != protocol["identity_sha256"]
        or audit.get("calibration_lock_or_members_opened") is not False
        or audit.get("evaluation_alert_lock_or_score_CSV_opened") is not False
        or audit.get("score_label_join_performed") is not False
        or audit.get("aggregate_only") is not True
    ):
        raise RuntimeError("original receipt does not certify the frozen eight-event pilot")
    return result, {
        "manifest_identity_sha256": manifest_identity,
        "result_identity_sha256": result_identity,
        "manifest_file_sha256": sha256_file(manifest_path),
    }


def _raw_label_from_scores(scores: Sequence[int]) -> str:
    if sum(score >= 2 for score in scores) >= 2:
        return LABEL_BAD
    if sum(score == 0 for score in scores) >= 2:
        return LABEL_GOOD
    return LABEL_EXCLUDED


def validate_final_consensus(
    root: Path,
    protocol: dict[str, Any],
    *,
    seeds: tuple[int, ...],
    status: str,
    reviewer_ids: tuple[str, ...],
    description: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = root.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"{description} must be a real directory: {root}")
    manifest_path = root / "manifest.json"
    consensus_path = root / "consensus_locked.json"
    manifest = load_json(manifest_path)
    completion = load_json(root / "completion.json")
    consensus = load_json(consensus_path)
    manifest_identity = require_canonical_identity(manifest, f"{description} manifest")
    consensus_identity = require_canonical_identity(consensus, description)
    expected_count = len(seeds) * len(CLASSES)
    if (
        manifest.get("status") != "complete"
        or completion.get("complete") is not True
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
        or completion.get("manifest_identity_sha256") != manifest_identity
        or completion.get("consensus_file_sha256") != sha256_file(consensus_path)
        or completion.get("consensus_identity_sha256") != consensus_identity
        or completion.get("locked_row_count") != expected_count
        or manifest.get("consensus_identity_sha256") != consensus_identity
        or consensus.get("status") != status
        or consensus.get("candidate_protocol_identity_sha256")
        != protocol["identity_sha256"]
    ):
        raise RuntimeError(f"{description} completion/identity validation failed")
    _members(root, manifest)
    audit = consensus.get("blinding_audit", {})
    if (
        audit.get("reviewer_count") != 3
        or audit.get("endpoint_only_review") is not True
        or audit.get("metric_values_visible_to_reviewers") is not False
        or audit.get("alert_decisions_visible_to_reviewers") is not False
        or audit.get("trajectories_visible_to_reviewers") is not False
        or audit.get("labels_locked_before_score_join") is not True
        or audit.get("adjudication_could_only_retain_or_downgrade_raw_clear_bad")
        is not True
    ):
        raise RuntimeError(f"{description} blinding/only-downgrade audit failed")
    rows = consensus.get("rows")
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise RuntimeError(f"{description} does not contain exact row count")
    expected = {(seed, class_id) for seed in seeds for class_id in CLASSES}
    observed: set[tuple[int, int]] = set()
    reduced: list[dict[str, Any]] = []
    for row in rows:
        seed = row.get("global_seed", row.get("seed"))
        class_id = row.get("class_id")
        if type(seed) is not int or type(class_id) is not int:
            raise RuntimeError(f"{description} has malformed row key")
        key = sample_key(class_id, seed)
        if row.get("sample_key") != key or (seed, class_id) in observed:
            raise RuntimeError(f"{description} sample-key or uniqueness failure")
        observed.add((seed, class_id))
        review_scores = row.get("review_scores")
        if (
            not isinstance(review_scores, dict)
            or set(review_scores) != set(reviewer_ids)
            or any(type(value) is not int or value not in range(4) for value in review_scores.values())
        ):
            raise RuntimeError(f"{description} review-score lineage failed: {key}")
        raw = _raw_label_from_scores(list(review_scores.values()))
        final = row.get("primary_label")
        adjudication = row.get("adjudication")
        if row.get("raw_primary_label") != raw or final not in LABELS:
            raise RuntimeError(f"{description} raw/final label lineage failed: {key}")
        if raw == LABEL_BAD:
            if not isinstance(adjudication, dict) or adjudication.get("decision") not in {
                "retain_clear_bad",
                "downgrade_to_mild",
            }:
                raise RuntimeError(f"{description} invalid bad adjudication: {key}")
            expected_final = (
                LABEL_BAD
                if adjudication["decision"] == "retain_clear_bad"
                else LABEL_EXCLUDED
            )
        else:
            if adjudication is not None:
                raise RuntimeError(f"{description} promoted/adjudicated nonbad row: {key}")
            expected_final = raw
        if final != expected_final or row.get("binary_primary_included") is not (
            final in {LABEL_BAD, LABEL_GOOD}
        ):
            raise RuntimeError(f"{description} final-label transition failed: {key}")
        native = row.get("native_image")
        if (
            not isinstance(native, dict)
            or not isinstance(native.get("file_sha256"), str)
            or len(native["file_sha256"]) != 64
        ):
            raise RuntimeError(f"{description} endpoint binding missing: {key}")
        reduced.append(
            {
                "global_seed": seed,
                "class_id": class_id,
                "primary_label": final,
                "binary_primary_included": final in {LABEL_BAD, LABEL_GOOD},
                "native_image_file_sha256": native["file_sha256"],
            }
        )
    if observed != expected:
        raise RuntimeError(f"{description} is not exact seed x class product")
    frame = pd.DataFrame(reduced).sort_values(
        ["global_seed", "class_id"], kind="mergesort"
    ).reset_index(drop=True)
    counts = {label: int(frame["primary_label"].eq(label).sum()) for label in LABELS}
    if (
        consensus.get("counts") != counts
        or consensus.get("retained_clear_bad_count") != counts[LABEL_BAD]
        or sum(counts.values()) != expected_count
    ):
        raise RuntimeError(f"{description} aggregate counts do not replay")
    return frame, {
        "consensus_identity_sha256": consensus_identity,
        "manifest_identity_sha256": manifest_identity,
        "manifest_file_sha256": sha256_file(manifest_path),
        "blind_pack_identity_sha256": consensus["blind_pack_identity_sha256"],
        "counts": counts,
    }


def _coerce_boolean(series: pd.Series, name: str) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.to_numpy(bool)
    lowered = series.astype(str).str.strip().str.lower()
    if not lowered.isin({"true", "false", "1", "0"}).all():
        raise RuntimeError(f"column is not strictly boolean: {name}")
    return lowered.isin({"true", "1"}).to_numpy(bool)


def validate_alert_product(
    root: Path,
    protocol: dict[str, Any],
    calibration: dict[str, Any],
    *,
    seeds: tuple[int, ...],
    status: str,
    cohort_role: str,
    description: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = root.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"{description} must be a real directory: {root}")
    manifest_path = root / "manifest.json"
    manifest = load_json(manifest_path)
    completion = load_json(root / "completion.json")
    summary = load_json(root / "summary.json")
    manifest_identity = require_canonical_identity(manifest, f"{description} manifest")
    members = _members(root, manifest)
    csv_path = root / "evaluation_scores_and_alerts_label_free.csv"
    if (
        manifest.get("status") != "complete"
        or completion.get("complete") is not True
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
        or completion.get("manifest_identity_sha256") != manifest_identity
        or completion.get("alerts_file_sha256") != members.get(csv_path.name, {}).get(
            "sha256"
        )
        or summary.get("status") != status
        or summary.get("candidate_protocol_identity_sha256")
        != protocol["identity_sha256"]
        or summary.get("calibration_identity_sha256")
        != calibration["identity_sha256"]
        or summary.get("sample_count") != len(seeds) * len(CLASSES)
        or tuple(summary.get("seeds", ())) != seeds
        or summary.get("labels_read_or_emitted") is not False
        or summary.get("thresholds_reestimated") is not False
    ):
        raise RuntimeError(f"{description} integrity/supervision contract failed")
    frame = pd.read_csv(csv_path)
    forbidden = ("label", "review", "consensus", "severity", "adjudic")
    leaked = [name for name in frame if any(token in name.lower() for token in forbidden)]
    if leaked:
        raise RuntimeError(f"{description} leaked supervision columns: {leaked}")
    required = {
        "global_seed",
        "class_id",
        "class_slot",
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
    if not required.issubset(frame.columns):
        raise RuntimeError(f"{description} lacks frozen score/alert columns")
    expected = {(seed, class_id) for seed in seeds for class_id in CLASSES}
    observed = {
        (int(row.global_seed), int(row.class_id))
        for row in frame[["global_seed", "class_id"]].itertuples(index=False)
    }
    if (
        len(frame) != len(expected)
        or observed != expected
        or frame[["global_seed", "class_id"]].duplicated().any()
        or not frame["cohort_role"].eq(cohort_role).all()
    ):
        raise RuntimeError(f"{description} is not exact Cartesian product")
    references = protocol["normalization"]["class_reference"]
    expected_a = np.empty(len(frame), dtype=np.float64)
    expected_b = np.empty(len(frame), dtype=np.float64)
    for class_id in CLASSES:
        mask = frame["class_id"].to_numpy(int) == class_id
        stats = references[str(class_id)]["statistics"]
        raw_a = frame.loc[mask, "A_posterior_logstd_concentration_jump"].to_numpy(float)
        raw_b = frame.loc[mask, "B_withheld_channel_predx0_cusum"].to_numpy(float)
        expected_a[mask] = (
            -raw_a - float(stats["A_low_is_bad"]["median"])
        ) / float(stats["A_low_is_bad"]["scale"])
        expected_b[mask] = (
            raw_b - float(stats["B_high_is_bad"]["median"])
        ) / float(stats["B_high_is_bad"]["scale"])
        thresholds = calibration["thresholds"][str(class_id)]
        scores = frame.loc[mask, "S_UNION"].to_numpy(float)
        if not np.array_equal(
            _coerce_boolean(frame.loc[mask, PRIMARY_ALERT_010], PRIMARY_ALERT_010),
            scores > float(thresholds["alpha_0p10"]["threshold"]),
        ) or not np.array_equal(
            _coerce_boolean(frame.loc[mask, PRIMARY_ALERT_005], PRIMARY_ALERT_005),
            scores > float(thresholds["alpha_0p05"]["threshold"]),
        ):
            raise RuntimeError(f"{description} conformal alerts do not replay")
    observed_a = frame["z_A_low_is_bad"].to_numpy(float)
    observed_b = frame["z_B_high_is_bad"].to_numpy(float)
    for actual, expected_values, name in (
        (observed_a, expected_a, "z_A"),
        (observed_b, expected_b, "z_B"),
        (frame["S_UNION"].to_numpy(float), np.maximum(observed_a, observed_b), "S_UNION"),
        (
            frame["S_INTERSECTION"].to_numpy(float),
            np.minimum(observed_a, observed_b),
            "S_INTERSECTION",
        ),
    ):
        if not np.allclose(actual, expected_values, rtol=1e-12, atol=1e-12):
            raise RuntimeError(f"{description} frozen formula does not replay: {name}")
    numeric = frame.select_dtypes(include=[np.number]).to_numpy(float)
    if not np.isfinite(numeric).all():
        raise RuntimeError(f"{description} contains non-finite numbers")
    frame[PRIMARY_ALERT_010] = _coerce_boolean(frame[PRIMARY_ALERT_010], PRIMARY_ALERT_010)
    frame[PRIMARY_ALERT_005] = _coerce_boolean(frame[PRIMARY_ALERT_005], PRIMARY_ALERT_005)
    return frame.sort_values(["global_seed", "class_id"], kind="mergesort").reset_index(
        drop=True
    ), {
        "manifest_identity_sha256": manifest_identity,
        "manifest_file_sha256": sha256_file(manifest_path),
        "candidate_score_manifest_identity_sha256": summary.get(
            "candidate_score_manifest_identity_sha256"
        ),
    }


def join_in_memory(scores: pd.DataFrame, labels: pd.DataFrame, description: str) -> pd.DataFrame:
    joined = scores.merge(
        labels,
        on=["global_seed", "class_id"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if len(joined) != len(scores) or not joined["_merge"].eq("both").all():
        raise RuntimeError(f"{description} score-label join lost/multiplied rows")
    joined = joined.drop(columns="_merge")
    hashes: list[str] = []
    for raw in joined["endpoint_png_path"]:
        path = Path(str(raw)).expanduser().absolute()
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"{description} score endpoint is missing: {path}")
        hashes.append(sha256_file(path))
    if not np.array_equal(
        np.asarray(hashes, dtype=object),
        joined["native_image_file_sha256"].to_numpy(object),
    ):
        raise RuntimeError(f"{description} label/score endpoint hashes differ")
    return joined


def _midranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("midranks require finite one-dimensional values")
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
    n_bad = int(is_bad.sum())
    n_good = int((~is_bad).sum())
    if scores.shape != is_bad.shape or not n_bad or not n_good:
        raise ValueError("AUC requires aligned scores and both labels")
    ranks = _midranks(scores)
    u = float(ranks[is_bad].sum() - n_bad * (n_bad + 1) / 2.0)
    return u / (n_bad * n_good)


def continuous_summary(
    frame: pd.DataFrame, score: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    binary = frame.loc[frame["binary_primary_included"]].copy()
    numerator = 0.0
    denominator = 0
    aucs: list[float] = []
    class_rows: list[dict[str, Any]] = []
    for class_id in CLASSES:
        group = binary.loc[binary["class_id"].eq(class_id)]
        is_bad = group["primary_label"].eq(LABEL_BAD).to_numpy(bool)
        n_bad = int(is_bad.sum())
        n_good = int((~is_bad).sum())
        pairs = n_bad * n_good
        auc = binary_auc(group[score].to_numpy(float), is_bad) if pairs else None
        if auc is not None:
            numerator += auc * pairs
            denominator += pairs
            aucs.append(auc)
        class_rows.append(
            {
                "score": score,
                "class_id": class_id,
                "n_clear_bad": n_bad,
                "n_clean_good": n_good,
                "bad_good_pair_count": pairs,
                "auc_higher_is_bad": auc,
            }
        )
    return {
        "score": score,
        "orientation": "higher_is_bad",
        "eligible_class_count": len(aucs),
        "bad_good_pair_count": denominator,
        "class_matched_pair_weighted_auc": numerator / denominator if denominator else None,
        "macro_within_class_auc": float(np.mean(aucs)) if aucs else None,
    }, class_rows


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
    for class_id in CLASSES:
        group = binary.loc[binary["class_id"].eq(class_id)]
        labels = group["primary_label"].eq(LABEL_BAD).to_numpy(bool)
        n_bad = int(labels.sum())
        n_good = int((~labels).sum())
        if not n_bad or not n_good:
            continue
        ranks = _midranks(group[PRIMARY_SCORE].to_numpy(float))
        observed_numerator += float(
            ranks[labels].sum() - n_bad * (n_bad + 1) / 2.0
        )
        total_pairs += n_bad * n_good
        groups.append({"class_id": class_id, "ranks": ranks, "n_bad": n_bad})
    if not groups:
        return {"available": False, "draws": 0, "reason": "no eligible class"}
    rng = np.random.Generator(np.random.PCG64(seed))
    null_auc = np.empty(draws, dtype=np.float64)
    count_ge = 0
    offset = 0
    while offset < draws:
        count = min(chunk_size, draws - offset)
        permuted = np.zeros(count, dtype=np.float64)
        for item in groups:
            ranks = item["ranks"]
            n_bad = item["n_bad"]
            keys = rng.random((count, len(ranks)))
            selected = np.argpartition(keys, n_bad - 1, axis=1)[:, :n_bad]
            permuted += ranks[selected].sum(axis=1) - n_bad * (n_bad + 1) / 2.0
        values = permuted / total_pairs
        null_auc[offset : offset + count] = values
        count_ge += int(np.sum(permuted >= observed_numerator))
        offset += count
    return {
        "available": True,
        "method": "within-class label permutation preserving observed bad/good counts",
        "statistic": "class-matched pair-weighted AUC of S_UNION",
        "orientation": "higher_is_bad",
        "sidedness": "one-sided_greater_or_equal",
        "draws": draws,
        "rng": f"numpy.random.Generator(PCG64(seed={seed}))",
        "observed_auc": observed_numerator / total_pairs,
        "permuted_greater_or_equal_count": count_ge,
        "p_value_add_one": (1.0 + count_ge) / (draws + 1.0),
        "null_auc_mean": float(np.mean(null_auc)),
        "null_auc_standard_deviation": float(np.std(null_auc, ddof=1)),
        "null_auc_draws_float64_sha256": sha256_bytes(
            np.asarray(null_auc, dtype="<f8").tobytes(order="C")
        ),
    }


def clopper_pearson(successes: int, trials: int) -> tuple[float, float]:
    if not trials or successes < 0 or successes > trials:
        raise ValueError("invalid binomial counts")
    tail = (1.0 - CONFIDENCE_LEVEL) / 2.0
    low = 0.0 if successes == 0 else float(beta.ppf(tail, successes, trials - successes + 1))
    high = 1.0 if successes == trials else float(beta.ppf(1.0 - tail, successes + 1, trials - successes))
    return low, high


def operating_points(
    frame: pd.DataFrame, alert_specs: Sequence[tuple[str, str, str]]
) -> list[dict[str, Any]]:
    binary = frame.loc[frame["binary_primary_included"]].copy()
    is_bad = binary["primary_label"].eq(LABEL_BAD).to_numpy(bool)
    rows: list[dict[str, Any]] = []
    for column, family, alpha in alert_specs:
        alert = _coerce_boolean(binary[column], column)
        bad_hits = int(np.sum(alert & is_bad))
        bad_total = int(is_bad.sum())
        good_hits = int(np.sum(alert & ~is_bad))
        good_total = int((~is_bad).sum())
        tpr = bad_hits / bad_total if bad_total else None
        fpr = good_hits / good_total if good_total else None
        tpr_ci = clopper_pearson(bad_hits, bad_total) if bad_total else None
        fpr_ci = clopper_pearson(good_hits, good_total) if good_total else None
        rows.append(
            {
                "alert_column": column,
                "family": family,
                "alpha_nominal": alpha,
                "clear_bad_alert_count": bad_hits,
                "clear_bad_count": bad_total,
                "TPR": tpr,
                "TPR_CP95_low": tpr_ci[0] if tpr_ci else None,
                "TPR_CP95_high": tpr_ci[1] if tpr_ci else None,
                "clean_good_alert_count": good_hits,
                "clean_good_count": good_total,
                "FPR": fpr,
                "FPR_CP95_low": fpr_ci[0] if fpr_ci else None,
                "FPR_CP95_high": fpr_ci[1] if fpr_ci else None,
                "TPR_minus_FPR": tpr - fpr if tpr is not None and fpr is not None else None,
            }
        )
    return rows


def cluster_bootstrap_difference(
    frame: pd.DataFrame,
    alert_columns: Sequence[str],
    *,
    draws: int = CLUSTER_BOOTSTRAP_DRAWS,
    seed: int = CLUSTER_BOOTSTRAP_SEED,
    chunk_size: int = 2048,
) -> dict[str, dict[str, Any]]:
    seeds = tuple(sorted(int(value) for value in frame["global_seed"].unique()))
    if seeds != ALL_CONFIRMATION_SEEDS:
        raise RuntimeError("cluster bootstrap does not see exact combined seed set")
    slot = {value: index for index, value in enumerate(seeds)}
    bad_count = np.zeros(len(seeds), dtype=np.int64)
    good_count = np.zeros(len(seeds), dtype=np.int64)
    bad_hits = np.zeros((len(seeds), len(alert_columns)), dtype=np.int64)
    good_hits = np.zeros_like(bad_hits)
    for row in frame.itertuples(index=False):
        if not bool(row.binary_primary_included):
            continue
        index = slot[int(row.global_seed)]
        is_bad = row.primary_label == LABEL_BAD
        if is_bad:
            bad_count[index] += 1
        else:
            good_count[index] += 1
        for metric, column in enumerate(alert_columns):
            triggered = int(bool(getattr(row, column)))
            if is_bad:
                bad_hits[index, metric] += triggered
            else:
                good_hits[index, metric] += triggered
    rng = np.random.Generator(np.random.PCG64(seed))
    differences = np.full((draws, len(alert_columns)), np.nan, dtype=np.float64)
    offset = 0
    while offset < draws:
        count = min(chunk_size, draws - offset)
        sampled = rng.integers(0, len(seeds), size=(count, len(seeds)))
        n_bad = bad_count[sampled].sum(axis=1)
        n_good = good_count[sampled].sum(axis=1)
        valid = (n_bad > 0) & (n_good > 0)
        for metric in range(len(alert_columns)):
            values = np.full(count, np.nan, dtype=np.float64)
            values[valid] = (
                bad_hits[sampled, metric].sum(axis=1)[valid] / n_bad[valid]
                - good_hits[sampled, metric].sum(axis=1)[valid] / n_good[valid]
            )
            differences[offset : offset + count, metric] = values
        offset += count
    result: dict[str, dict[str, Any]] = {}
    for metric, column in enumerate(alert_columns):
        values = differences[:, metric]
        values = values[np.isfinite(values)]
        if not len(values):
            result[column] = {"available": False, "draws": draws, "valid_draws": 0}
            continue
        low, high = np.quantile(values, [0.025, 0.975], method="linear")
        result[column] = {
            "available": True,
            "method": "global-seed/run cluster percentile bootstrap retaining all three class rows",
            "draws": draws,
            "valid_draws": int(len(values)),
            "rng": f"numpy.random.Generator(PCG64(seed={seed}))",
            "TPR_minus_FPR_bootstrap95_low": float(low),
            "TPR_minus_FPR_bootstrap95_high": float(high),
            "valid_draws_float64_sha256": sha256_bytes(
                np.asarray(values, dtype="<f8").tobytes(order="C")
            ),
        }
    return result


def stratified_auc_bootstrap(
    frame: pd.DataFrame,
    *,
    draws: int = AUC_BOOTSTRAP_DRAWS,
    seed: int = AUC_BOOTSTRAP_SEED,
    chunk_size: int = 512,
) -> dict[str, Any]:
    binary = frame.loc[frame["binary_primary_included"]]
    groups: list[dict[str, Any]] = []
    for class_id in CLASSES:
        group = binary.loc[binary["class_id"].eq(class_id)]
        bad = group.loc[group["primary_label"].eq(LABEL_BAD), PRIMARY_SCORE].to_numpy(float)
        good = group.loc[group["primary_label"].eq(LABEL_GOOD), PRIMARY_SCORE].to_numpy(float)
        if len(bad) and len(good):
            groups.append({"class_id": class_id, "bad": bad, "good": good, "weight": len(bad) * len(good)})
    if not groups:
        return {"available": False, "draws": 0, "reason": "no eligible class"}
    total_weight = sum(item["weight"] for item in groups)
    weights = np.asarray([item["weight"] for item in groups], dtype=np.float64)
    pair = np.empty(draws, dtype=np.float64)
    macro = np.empty(draws, dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(seed))
    offset = 0
    while offset < draws:
        count = min(chunk_size, draws - offset)
        class_aucs = np.empty((count, len(groups)), dtype=np.float64)
        for index, item in enumerate(groups):
            bad = item["bad"]
            good = item["good"]
            sampled_bad = bad[rng.integers(0, len(bad), size=(count, len(bad)))]
            sampled_good = good[rng.integers(0, len(good), size=(count, len(good)))]
            wins = sampled_bad[:, :, None] > sampled_good[:, None, :]
            ties = sampled_bad[:, :, None] == sampled_good[:, None, :]
            class_aucs[:, index] = np.mean(wins + 0.5 * ties, axis=(1, 2))
        pair[offset : offset + count] = class_aucs @ weights / total_weight
        macro[offset : offset + count] = class_aucs.mean(axis=1)
        offset += count
    pair_ci = np.quantile(pair, [0.025, 0.975], method="linear")
    macro_ci = np.quantile(macro, [0.025, 0.975], method="linear")
    return {
        "available": True,
        "method": "percentile bootstrap independently within each (class,label) stratum",
        "draws": draws,
        "rng": f"numpy.random.Generator(PCG64(seed={seed}))",
        "eligible_classes": [item["class_id"] for item in groups],
        "class_matched_pair_weighted_auc_bootstrap95_low": float(pair_ci[0]),
        "class_matched_pair_weighted_auc_bootstrap95_high": float(pair_ci[1]),
        "macro_within_class_auc_bootstrap95_low": float(macro_ci[0]),
        "macro_within_class_auc_bootstrap95_high": float(macro_ci[1]),
        "pair_weighted_draws_float64_sha256": sha256_bytes(
            np.asarray(pair, dtype="<f8").tobytes(order="C")
        ),
        "macro_draws_float64_sha256": sha256_bytes(
            np.asarray(macro, dtype="<f8").tobytes(order="C")
        ),
    }


def _score_specs(protocol: dict[str, Any]) -> list[tuple[str, str]]:
    result = [
        ("S_UNION", "primary"),
        ("z_A_low_is_bad", "single_feature_A_mechanism"),
        ("z_B_high_is_bad", "single_feature_B_mechanism"),
        ("S_INTERSECTION", "retired_descriptive_subtype_control"),
        ("old_fixed_predicted_clean_score_control", "old_fixed_score_negative_control"),
    ]
    result.extend(
        (
            "control_" + name.replace("__full_maximum", ""),
            "exact_path_LR_negative_control",
        )
        for name in protocol["negative_controls"]["exact_path_evidence_running_maxima"]
    )
    return result


def _alert_specs(protocol: dict[str, Any]) -> list[tuple[str, str, str]]:
    result = [
        (PRIMARY_ALERT_010, "S_UNION_split_conformal", "0.10"),
        (PRIMARY_ALERT_005, "S_UNION_split_conformal", "0.05"),
    ]
    for name in protocol["negative_controls"]["exact_path_evidence_running_maxima"]:
        control = "control_" + name.replace("__full_maximum", "")
        result.extend(
            [
                (f"{control}_trigger_alpha0p10", "exact_path_LR_negative_control", "0.10"),
                (f"{control}_trigger_alpha0p05", "exact_path_LR_negative_control", "0.05"),
            ]
        )
    return result


def evaluate_statistics(
    frame: pd.DataFrame,
    protocol: dict[str, Any],
    *,
    permutation_draws: int = PERMUTATION_DRAWS,
    cluster_draws: int = CLUSTER_BOOTSTRAP_DRAWS,
    auc_draws: int = AUC_BOOTSTRAP_DRAWS,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    n_bad = int(frame["primary_label"].eq(LABEL_BAD).sum())
    if n_bad < MINIMUM_CLEAR_BAD_EVENTS:
        raise RuntimeError("full statistics forbidden below 15-event gate")
    scores: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    for column, role in _score_specs(protocol):
        if column not in frame:
            raise RuntimeError(f"combined table lacks frozen score: {column}")
        summary, per_class = continuous_summary(frame, column)
        summary["role"] = role
        scores.append(summary)
        for row in per_class:
            row["role"] = role
            class_rows.append(row)
    permutation = stratified_permutation_test(
        frame, draws=permutation_draws, seed=PERMUTATION_SEED
    )
    auc_uncertainty = stratified_auc_bootstrap(
        frame, draws=auc_draws, seed=AUC_BOOTSTRAP_SEED
    )
    alert_specs = _alert_specs(protocol)
    operating = operating_points(frame, alert_specs)
    cluster = cluster_bootstrap_difference(
        frame,
        [row[0] for row in alert_specs],
        draws=cluster_draws,
        seed=CLUSTER_BOOTSTRAP_SEED,
    )
    for row in operating:
        interval = cluster[row["alert_column"]]
        row.update(
            {
                key: value
                for key, value in interval.items()
                if key.startswith("TPR_minus_FPR_bootstrap95_") or key == "valid_draws"
            }
        )
    primary = next(row for row in scores if row["score"] == PRIMARY_SCORE)
    primary_classes = [row for row in class_rows if row["score"] == PRIMARY_SCORE]
    primary_operating = next(
        row for row in operating if row["alert_column"] == PRIMARY_ALERT_010
    )
    gate = protocol["evaluation"]["initial_go_gate"]
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
            < gate["no_class_with_two_or_more_bad_events_has_auc_below"]
        )
    ]
    criteria = {
        "minimum_clear_bad_events_for_decision": n_bad
        >= gate["minimum_clear_bad_events_for_decision"],
        "S_UNION_class_matched_auc_at_least_0p75": (
            primary["class_matched_pair_weighted_auc"] is not None
            and primary["class_matched_pair_weighted_auc"]
            >= gate["S_UNION_class_matched_auc_at_least"]
        ),
        "S_UNION_permutation_one_sided_p_below_0p05": (
            permutation.get("available") is True
            and permutation["p_value_add_one"]
            < gate["S_UNION_stratified_permutation_one_sided_p_below"]
        ),
        "alpha0p10_TPR_minus_FPR_point_above_zero": (
            primary_operating["TPR_minus_FPR"] is not None
            and primary_operating["TPR_minus_FPR"]
            > gate["alpha_0p10_TPR_minus_FPR_point_above"]
        ),
        "no_class_with_at_least_two_bad_has_auc_below_0p60": not low_classes,
    }
    counts = {label: int(frame["primary_label"].eq(label).sum()) for label in LABELS}
    result = {
        "schema_version": 1,
        "status": "COMPLETE_COMBINED_ORIGINAL_PLUS_EXPANSION_CONFIRMATION",
        "cohort": {
            "classes": list(CLASSES),
            "original_trajectory_count": 240,
            "expansion_trajectory_count": 360,
            "combined_trajectory_count": 600,
            "aggregate_label_counts": counts,
            "binary_metric_row_count": counts[LABEL_BAD] + counts[LABEL_GOOD],
        },
        "primary_score": primary,
        "continuous_scores": scores,
        "primary_randomization_test": permutation,
        "primary_auc_uncertainty": auc_uncertainty,
        "operating_points": operating,
        "cluster_bootstrap_details": cluster,
        "initial_go_gate": {
            "criteria": criteria,
            "classes_failing_minimum_auc_guardrail": low_classes,
            "passed": all(criteria.values()),
            "decision": "INITIAL_GO" if all(criteria.values()) else "CONFIRMATION_GATE_FAILED",
            "gates_changed_after_original_candidate_v5": False,
        },
        "frozen_design_audit": {
            "A_B_S_UNION_formula_changed": False,
            "discovery_normalizers_reestimated": False,
            "calibration_thresholds_reestimated": False,
            "permutation_method_changed": False,
            "permutation_draws": permutation_draws,
            "permutation_seed": PERMUTATION_SEED,
            "expansion_decision_used_score_or_alert_performance": False,
        },
        "claim_limits": {
            "calibration_alpha_is_not_clean_good_conditional_FPR": True,
            "mild_or_disputed_rows_excluded": True,
            "cross_class_or_cross_model_generalization_not_established": True,
            "Clopper_Pearson_intervals_are_descriptive_under_row_independence": True,
        },
    }
    tables = {
        "continuous_score_metrics.csv": pd.DataFrame(scores),
        "per_class_score_metrics.csv": pd.DataFrame(class_rows),
        "operating_point_metrics.csv": pd.DataFrame(operating),
    }
    return result, tables


def _publish(
    output: Path,
    result: dict[str, Any],
    tables: dict[str, pd.DataFrame],
    lineage: dict[str, Any],
) -> Path:
    _assert_no_row_payload(result)
    for name, frame in tables.items():
        if FORBIDDEN_ROW_KEYS.intersection(frame.columns):
            raise RuntimeError(f"aggregate table leaked row identifiers: {name}")
        if len(frame) >= 80:
            raise RuntimeError(f"aggregate table has suspicious row cardinality: {name}")
    result = dict(result)
    result["input_lineage"] = lineage
    result["identity_sha256"] = canonical_sha256(result)
    output = output.expanduser().absolute()
    if os.path.lexists(output):
        raise RuntimeError(f"refusing to overwrite combined evaluation: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        write_json(staging / "confirmation_results.json", result)
        for name, frame in tables.items():
            frame.to_csv(staging / name, index=False)
        shutil.copy2(Path(__file__).resolve(), staging / "evaluator_source.py")
        helper = Path(__file__).resolve().with_name("dit_bad_good_expansion_contract.py")
        shutil.copy2(helper, staging / "expansion_contract_source.py")
        members = [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(staging.iterdir())
        ]
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "complete",
            "candidate_protocol_identity_sha256": CANDIDATE_PROTOCOL_IDENTITY,
            "result_identity_sha256": result["identity_sha256"],
            "aggregate_only": True,
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
                "result_file_sha256": sha256_file(staging / "confirmation_results.json"),
                "result_identity_sha256": result["identity_sha256"],
                "aggregate_only": True,
                "published_table_count": len(tables),
            },
        )
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def publish(
    *,
    candidate_lock: Path,
    original_event_receipt: Path,
    expansion_consensus_lock: Path,
    original_consensus_lock: Path,
    calibration_lock: Path,
    original_alerts_root: Path,
    expansion_alerts_root: Path,
    output: Path,
) -> Path:
    protocol = validate_candidate_lock(candidate_lock)
    validate_expansion_lock()
    pipeline = validate_pipeline_source_lock(Path(__file__).name)
    for key, path in (
        ("candidate_v5_lock", candidate_lock),
        ("original_event_receipt", original_event_receipt),
        ("final_consensus_lock", expansion_consensus_lock),
        ("original_final_consensus_lock", original_consensus_lock),
        ("calibration_lock", calibration_lock),
        ("original_label_free_alerts", original_alerts_root),
        ("label_free_calibrated_alerts", expansion_alerts_root),
        ("combined_aggregate_result", output),
    ):
        require_planned_path(pipeline, key, path)
    original_receipt, original_receipt_lineage = validate_original_event_receipt(
        original_event_receipt, protocol
    )
    expansion_labels, expansion_lineage = validate_final_consensus(
        expansion_consensus_lock,
        protocol,
        seeds=EXPANSION_SEEDS,
        status=EXPANSION_FINAL_STATUS,
        reviewer_ids=REVIEWERS,
        description="final expansion consensus",
    )
    expansion_counts = {
        label: int(expansion_labels["primary_label"].eq(label).sum()) for label in LABELS
    }
    total_bad = ORIGINAL_CLEAR_BAD_EVENTS + expansion_counts[LABEL_BAD]
    common_lineage = {
        "candidate_protocol_identity_sha256": protocol["identity_sha256"],
        "original_aggregate_event_receipt": original_receipt_lineage,
        "expansion_consensus": expansion_lineage,
    }
    if total_bad < MINIMUM_CLEAR_BAD_EVENTS:
        # Hard evidence-access boundary: no function below this return receives,
        # stats, resolves, hashes, validates, or opens the remaining four inputs.
        result = {
            "schema_version": 1,
            "status": "COMBINED_EVENT_COUNT_ONLY_FURTHER_EXPANSION_REQUIRED",
            "event_gate": {
                "minimum_clear_bad_events_for_decision": MINIMUM_CLEAR_BAD_EVENTS,
                "original_clear_bad_events": ORIGINAL_CLEAR_BAD_EVENTS,
                "new_expansion_clear_bad_events": expansion_counts[LABEL_BAD],
                "combined_clear_bad_events": total_bad,
                "evaluated": False,
                "decision": "PILOT_ONLY_FURTHER_DISJOINT_EXPANSION_REQUIRED",
            },
            "expansion_aggregate_label_counts": expansion_counts,
            "evidence_access_audit": {
                "original_row_label_lock_opened": False,
                "calibration_lock_or_members_opened": False,
                "original_score_or_alert_product_opened": False,
                "expansion_score_or_alert_product_opened": False,
                "score_label_join_performed": False,
                "aggregate_only": True,
            },
        }
        return _publish(output, result, {}, common_lineage)

    original_labels, original_label_lineage = validate_final_consensus(
        original_consensus_lock,
        protocol,
        seeds=ORIGINAL_EVALUATION_SEEDS,
        status=ORIGINAL_FINAL_STATUS,
        reviewer_ids=("G", "H", "I"),
        description="final original consensus",
    )
    original_bad = int(original_labels["primary_label"].eq(LABEL_BAD).sum())
    if original_bad != ORIGINAL_CLEAR_BAD_EVENTS:
        raise RuntimeError("original row-label lock disagrees with aggregate eight-event receipt")
    calibration, calibration_manifest_identity = validate_calibration_lock(
        calibration_lock, protocol
    )
    original_scores, original_score_lineage = validate_alert_product(
        original_alerts_root,
        protocol,
        calibration,
        seeds=ORIGINAL_EVALUATION_SEEDS,
        status=ORIGINAL_ALERT_STATUS,
        cohort_role="inferential_evaluation",
        description="original label-free alert product",
    )
    expansion_scores, expansion_score_lineage = validate_alert_product(
        expansion_alerts_root,
        protocol,
        calibration,
        seeds=EXPANSION_SEEDS,
        status=EXPANSION_ALERT_STATUS,
        cohort_role="inferential_expansion",
        description="expansion label-free alert product",
    )
    original_joined = join_in_memory(original_scores, original_labels, "original")
    expansion_joined = join_in_memory(expansion_scores, expansion_labels, "expansion")
    combined = pd.concat([original_joined, expansion_joined], ignore_index=True)
    expected = {(seed, class_id) for seed in ALL_CONFIRMATION_SEEDS for class_id in CLASSES}
    observed = {
        (int(row.global_seed), int(row.class_id))
        for row in combined[["global_seed", "class_id"]].itertuples(index=False)
    }
    if len(combined) != 600 or observed != expected:
        raise RuntimeError("combined in-memory join is not exact 3x200 cohort")
    result, tables = evaluate_statistics(combined, protocol)
    lineage = {
        **common_lineage,
        "original_consensus": original_label_lineage,
        "calibration_identity_sha256": calibration["identity_sha256"],
        "calibration_manifest_identity_sha256": calibration_manifest_identity,
        "original_alert_product": original_score_lineage,
        "expansion_alert_product": expansion_score_lineage,
        "validation_order": [
            "candidate_v5",
            "original_aggregate_event_receipt",
            "final_expansion_visual_labels",
            "combined_minimum_15_event_gate",
            "original_row_visual_labels",
            "immutable_calibration",
            "original_label_free_scores_and_alerts",
            "expansion_label_free_scores_and_alerts",
            "two_one_to_one_in_memory_joins_with_endpoint_hash_match",
            "aggregate_statistics_only",
        ],
        "joined_rows_or_individual_ranks_published": False,
    }
    return _publish(output, result, tables, lineage)


def self_test() -> None:
    # Tie-aware AUC.
    assert binary_auc(np.asarray([0.0, 1.0, 1.0, 2.0]), np.asarray([False, True, False, True])) == 0.875
    rows: list[dict[str, Any]] = []
    for seed in ALL_CONFIRMATION_SEEDS:
        for class_id in CLASSES:
            bad = seed < 55 and class_id == 207
            score = 10.0 if bad else float((seed + class_id) % 17) / 17.0
            rows.append(
                {
                    "global_seed": seed,
                    "class_id": class_id,
                    "primary_label": LABEL_BAD if bad else LABEL_GOOD,
                    "binary_primary_included": True,
                    "S_UNION": score,
                    PRIMARY_ALERT_010: bad,
                    PRIMARY_ALERT_005: bad,
                }
            )
    frame = pd.DataFrame(rows)
    permutation = stratified_permutation_test(frame, draws=37, seed=PERMUTATION_SEED)
    assert permutation["draws"] == 37 and permutation["observed_auc"] > 0.9
    cluster = cluster_bootstrap_difference(
        frame, [PRIMARY_ALERT_010], draws=41, seed=CLUSTER_BOOTSTRAP_SEED
    )
    assert cluster[PRIMARY_ALERT_010]["valid_draws"] == 41
    result = {
        "status": "test",
        "counts": {LABEL_BAD: 5, LABEL_GOOD: 595, LABEL_EXCLUDED: 0},
    }
    _assert_no_row_payload(result)
    assert ORIGINAL_CLEAR_BAD_EVENTS + 6 < MINIMUM_CLEAR_BAD_EVENTS
    assert ORIGINAL_CLEAR_BAD_EVENTS + 7 >= MINIMUM_CLEAR_BAD_EVENTS
    print("self-test passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-lock", type=Path, default=CANDIDATE_LOCK)
    parser.add_argument("--original-event-receipt", type=Path)
    parser.add_argument("--expansion-consensus-lock", type=Path)
    parser.add_argument("--original-consensus-lock", type=Path)
    parser.add_argument("--calibration-lock", type=Path)
    parser.add_argument("--original-alerts-root", type=Path)
    parser.add_argument("--expansion-alerts-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    required = (
        "original_event_receipt",
        "expansion_consensus_lock",
        "original_consensus_lock",
        "calibration_lock",
        "original_alerts_root",
        "expansion_alerts_root",
        "output",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error("missing required inputs: " + ", ".join(missing))
    output = publish(
        candidate_lock=args.candidate_lock,
        original_event_receipt=args.original_event_receipt,
        expansion_consensus_lock=args.expansion_consensus_lock,
        original_consensus_lock=args.original_consensus_lock,
        calibration_lock=args.calibration_lock,
        original_alerts_root=args.original_alerts_root,
        expansion_alerts_root=args.expansion_alerts_root,
        output=args.output,
    )
    result = load_json(output / "confirmation_results.json")
    print(json.dumps({"output": str(output), "status": result["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
