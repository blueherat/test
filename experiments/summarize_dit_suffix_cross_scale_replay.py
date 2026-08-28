#!/usr/bin/env python3
"""Summarize completed DiT suffix cross-scale replay bundles offline.

The raw replay bundle is always validated by its own frozen validator before
any array is used.  This script does not run DiT, alter raw traces, select a
threshold, inspect pixels, or consume an image-dependent ROI.

For every branch, Delta-nu, and fixed component (global plus row-major 4x4
latent tiles), it reports cross-scale theta RMS, raw/applied conditional KL,
cap saturation, and the final/running-maximum log e-value for the recorded
+theta high-noise alternative.

Two fixed mixtures are additionally reconstructed from the saved Gaussian LR
decomposition R=<u,z>, K=||u||^2/2:

* a 50/50 sign mixture, with L_plus=R-K and L_minus=-R-K (not -L_plus);
* a uniform change-point mixture whose possible start indices and weights are
  fixed at suffix entry.  A component whose start lies in the future remains
  exactly E=1 until its start time.  The combined scale/tile/sign/start mixture
  therefore remains a fixed prior mixture rather than a posthoc maximum.

An optional v2 visual-label file enables descriptive discovery-only matched
and cohort comparisons.  Labels never affect metric construction, ranking, or
mixture weights.  No significance test, TPR/FPR, threshold, or success claim is
produced; held-out validation is still required.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

try:
    from .intervene_dit_imagenet256_suffix import _atomic_install_directory_noreplace
    from .reproduce_dit_imagenet256 import atomic_json_dump, sha256_file, sha256_json
    from .replay_dit_suffix_cross_scale_diagnostics import (
        COMPONENT_COUNT,
        EXPERIMENT as RAW_EXPERIMENT,
        TRACE_NAME as RAW_TRACE_NAME,
        validate_output_bundle as validate_raw_bundle,
    )
except ImportError:  # pragma: no cover - direct CLI execution.
    from intervene_dit_imagenet256_suffix import _atomic_install_directory_noreplace
    from reproduce_dit_imagenet256 import atomic_json_dump, sha256_file, sha256_json
    from replay_dit_suffix_cross_scale_diagnostics import (
        COMPONENT_COUNT,
        EXPERIMENT as RAW_EXPERIMENT,
        TRACE_NAME as RAW_TRACE_NAME,
        validate_output_bundle as validate_raw_bundle,
    )


EXPERIMENT = "dit_imagenet256_suffix_cross_scale_replay_offline_summary"
SCHEMA_VERSION = 1
COMPONENT_CSV = "branch_component_metrics.csv"
MIXTURE_CSV = "branch_mixture_metrics.csv"
SUMMARY_JSON = "summary.json"
RANK_RELATIVE_TOLERANCE = 1e-12
RANK_ABSOLUTE_TOLERANCE = 1e-15

COMPONENT_METRICS = (
    "theta_rms_mean",
    "theta_rms_max",
    "raw_K_total",
    "applied_K_total",
    "cap_saturation_fraction",
    "plus_final_log_e",
    "plus_running_max_log_e",
    "sign_mixture_final_log_e",
    "sign_mixture_running_max_log_e",
    "change_point_sign_mixture_final_log_e",
    "change_point_sign_mixture_running_max_log_e",
)

MIXTURE_METRICS = (
    "theta_rms_mean",
    "raw_K_mean_per_component",
    "applied_K_mean_per_component",
    "cap_saturation_fraction",
    "plus_mixture_final_log_e",
    "plus_mixture_running_max_log_e",
    "sign_mixture_final_log_e",
    "sign_mixture_running_max_log_e",
    "change_point_sign_mixture_final_log_e",
    "change_point_sign_mixture_running_max_log_e",
)


def _canonical_self_hash(payload: dict[str, Any], key: str) -> str:
    stripped = dict(payload)
    stripped.pop(key, None)
    return sha256_json(stripped)


def _logsumexp(values: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    maximum = np.max(values, axis=axis, keepdims=True)
    result = maximum + np.log(np.sum(np.exp(values - maximum), axis=axis, keepdims=True))
    axes = (axis,) if isinstance(axis, int) else axis
    for item in sorted((value % values.ndim for value in axes), reverse=True):
        result = np.squeeze(result, axis=item)
    return np.asarray(result, dtype=np.float64)


def _logmeanexp(values: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray:
    axes = (axis,) if isinstance(axis, int) else axis
    count = 1
    for item in axes:
        count *= values.shape[item % values.ndim]
    return _logsumexp(values, axis=axis) - math.log(count)


def uniform_change_point_log_mixture(
    increments: np.ndarray, *, start_count: int
) -> np.ndarray:
    """Return a uniform fixed-prior change-point log mixture.

    ``increments`` has arbitrary leading component axes and time last.  Starts
    are the fixed indices 0..start_count-1.  At reported time k, starts j>k
    have not launched and contribute E=1, not zero weight.  Entries after
    ``start_count`` may be deterministic terminal bookkeeping such as DiT t=0.
    """

    values = np.asarray(increments, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] < 1:
        raise ValueError("increments must have a non-empty time axis")
    if not np.isfinite(values).all():
        raise ValueError("increments contain non-finite values")
    if not 1 <= start_count <= values.shape[-1]:
        raise ValueError("start_count must lie within the time axis")
    prefix = np.cumsum(values, axis=-1, dtype=np.float64)
    output = np.empty_like(prefix)
    leading = values.shape[:-1]
    for time_index in range(values.shape[-1]):
        launched = min(time_index + 1, start_count)
        starts = np.arange(launched, dtype=np.int64)
        before = np.zeros(leading + (launched,), dtype=np.float64)
        if launched > 1:
            before[..., 1:] = prefix[..., starts[1:] - 1]
        active = prefix[..., time_index, None] - before
        future_count = start_count - launched
        if future_count:
            terms = np.concatenate(
                [active, np.zeros(leading + (future_count,), dtype=np.float64)], axis=-1
            )
        else:
            terms = active
        output[..., time_index] = _logsumexp(terms, axis=-1) - math.log(start_count)
    return np.ascontiguousarray(output)


def fixed_sign_log_mixture(plus: np.ndarray, minus: np.ndarray) -> np.ndarray:
    if plus.shape != minus.shape:
        raise ValueError("plus/minus paths must have matching shapes")
    return _logmeanexp(np.stack([plus, minus], axis=0), axis=0)


def rank_interval_and_margins(values: Sequence[float], index: int) -> dict[str, Any]:
    """Descending rank interval with fixed near-tie tolerance and strict margins."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("rank values must be a finite non-empty vector")
    if not 0 <= index < len(array):
        raise IndexError("rank index out of range")
    value = float(array[index])
    tied = np.isclose(
        array,
        value,
        rtol=RANK_RELATIVE_TOLERANCE,
        atol=RANK_ABSOLUTE_TOLERANCE,
    )
    strictly_higher = array[~tied & (array > value)]
    strictly_lower = array[~tied & (array < value)]
    rank_min = int(strictly_higher.size + 1)
    rank_max = int(rank_min + tied.sum() - 1)
    return {
        "rank_desc_min": rank_min,
        "rank_desc_max": rank_max,
        "tie_count": int(tied.sum()),
        "distance_to_nearest_higher": (
            None if not strictly_higher.size else float(strictly_higher.min()) - value
        ),
        "margin_above_nearest_lower": (
            None if not strictly_lower.size else value - float(strictly_lower.max())
        ),
    }


def _load_trace(root: Path, record: dict[str, Any]) -> dict[str, np.ndarray]:
    path = root / RAW_TRACE_NAME
    if (
        record.get("relative_path") != RAW_TRACE_NAME
        or not path.is_file()
        or path.stat().st_size != record.get("bytes")
        or sha256_file(path) != record.get("sha256")
    ):
        raise RuntimeError(f"raw replay trace identity failed: {root}")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.ascontiguousarray(archive[key]) for key in archive.files}
    if sorted(arrays) != record.get("keys"):
        raise RuntimeError("raw trace key set changed after bundle validation")
    return arrays


def _bundle_identity(root: Path, manifest: dict[str, Any]) -> tuple[int, str]:
    record = manifest["input_suffix_bundle"]
    suffix_root = Path(record["root"])
    suffix_manifest_path = suffix_root / "manifest.json"
    if (
        not suffix_manifest_path.is_file()
        or sha256_file(suffix_manifest_path) != record["manifest_file_sha256"]
    ):
        raise RuntimeError("raw replay's bound suffix manifest is unavailable or changed")
    with suffix_manifest_path.open("r", encoding="utf-8") as handle:
        suffix_manifest = json.load(handle)
    if suffix_manifest.get("identity_sha256") != _canonical_self_hash(
        suffix_manifest, "identity_sha256"
    ) or suffix_manifest["identity_sha256"] != record["manifest_identity_sha256"]:
        raise RuntimeError("raw replay's bound suffix manifest self-identity failed")
    rollback = int(
        suffix_manifest["frozen_screen_protocol"]["this_invocation_rollback_internal_timestep"]
    )
    return rollback, str(manifest["input_suffix_bundle"]["manifest_identity_sha256"])


def _load_labels(path: Path | None) -> tuple[dict[str, str], dict[str, Any] | None]:
    if path is None:
        return {}, None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if (
        payload.get("schema_version") != 2
        or payload.get("status") != "posthoc_discovery_labels_not_for_confirmatory_claims"
    ):
        raise RuntimeError("optional label file is not the expected discovery-only v2 schema")
    labels: dict[str, str] = {}
    for item in payload.get("branch_reviews", []):
        key = f"t{int(item['internal_timestep'])}_attempt{int(item['attempt']):03d}"
        value = item.get("binary_discovery_label")
        if value is not None:
            if value not in {"good", "bad"}:
                raise RuntimeError(f"invalid binary discovery label: {key}/{value}")
            labels[key] = value
    cohorts = payload.get("discovery_cohorts", {})
    expected_matched = cohorts.get("primary_matched_late_suffix", {})
    expected_broader = cohorts.get("broader_clear_quality", {})
    if expected_matched.get("bad") != ["t60_attempt003"] or expected_matched.get(
        "good"
    ) != ["t60_attempt004"]:
        raise RuntimeError("v2 matched-pair cohort changed")
    if len(expected_broader.get("bad", [])) != 3 or len(expected_broader.get("good", [])) != 10:
        raise RuntimeError("v2 broader 3-bad/10-good cohort changed")
    return labels, {
        "path": str(path),
        "sha256": sha256_file(path),
        "status": payload["status"],
        "cohorts": cohorts,
    }


def _running_max(path: np.ndarray) -> float:
    return float(max(0.0, np.max(path, initial=-np.inf)))


def _aggregate_paths(
    plus_log_e: np.ndarray,
    sign_log_e: np.ndarray,
    cp_sign_log_e: np.ndarray,
) -> dict[str, float]:
    plus_mix = _logmeanexp(plus_log_e, axis=0)
    sign_mix = _logmeanexp(sign_log_e, axis=0)
    cp_mix = _logmeanexp(cp_sign_log_e, axis=0)
    return {
        "plus_mixture_final_log_e": float(plus_mix[-1]),
        "plus_mixture_running_max_log_e": _running_max(plus_mix),
        "sign_mixture_final_log_e": float(sign_mix[-1]),
        "sign_mixture_running_max_log_e": _running_max(sign_mix),
        "change_point_sign_mixture_final_log_e": float(cp_mix[-1]),
        "change_point_sign_mixture_running_max_log_e": _running_max(cp_mix),
    }


def summarize_bundle(
    root: Path, *, labels: dict[str, str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    manifest, results = validate_raw_bundle(root)
    if manifest.get("experiment") != RAW_EXPERIMENT:
        raise RuntimeError("unexpected raw replay experiment")
    arrays = _load_trace(root, results["trace"])
    rollback, suffix_identity = _bundle_identity(root, manifest)

    attempts = arrays["attempt_index"].astype(int)
    scales = arrays["delta_nu"].astype(np.float64)
    internal = arrays["internal_timestep"].astype(int)
    if internal[0] != rollback or internal[-1] != 0 or not np.array_equal(
        internal, np.arange(rollback, -1, -1)
    ):
        raise RuntimeError("raw replay suffix time axis changed")
    start_count = int(np.count_nonzero(internal > 0))
    R = arrays["R_component"].astype(np.float64, copy=False)  # [D,A,S,C]
    K = arrays["K_component"].astype(np.float64, copy=False)
    raw_K = arrays["raw_K_component"].astype(np.float64, copy=False)
    plus_increment = R - K
    minus_increment = -R - K
    if not np.array_equal(plus_increment, arrays["L_component"]):
        raise RuntimeError("raw +theta L is not exactly R-K")
    # Reorder to [D,A,C,S] for path operations.
    plus_path = np.cumsum(np.moveaxis(plus_increment, 2, -1), axis=-1, dtype=np.float64)
    minus_path = np.cumsum(np.moveaxis(minus_increment, 2, -1), axis=-1, dtype=np.float64)
    sign_path = fixed_sign_log_mixture(plus_path, minus_path)
    cp_plus = uniform_change_point_log_mixture(
        np.moveaxis(plus_increment, 2, -1), start_count=start_count
    )
    cp_minus = uniform_change_point_log_mixture(
        np.moveaxis(minus_increment, 2, -1), start_count=start_count
    )
    cp_sign_path = fixed_sign_log_mixture(cp_plus, cp_minus)

    effective = arrays["effective_nonidentity"].astype(bool)  # [D,S]
    cap = arrays["per_step_K_cap"].astype(np.float64)
    component_names = arrays["component_name"].astype(str).tolist()
    if len(component_names) != COMPONENT_COUNT:
        raise RuntimeError("component count changed")
    component_rows: list[dict[str, Any]] = []
    mixture_rows: list[dict[str, Any]] = []
    for scale_index, delta_nu in enumerate(scales.tolist()):
        active = effective[scale_index]
        if not np.any(active):
            raise RuntimeError("scale has no effective stochastic step")
        for branch_index, attempt in enumerate(attempts.tolist()):
            branch_key = f"t{rollback}_attempt{attempt:03d}"
            for component_index, component_name in enumerate(component_names):
                theta_values = (
                    arrays["theta_global_rms"][scale_index, branch_index]
                    if component_index == 0
                    else arrays["theta_tile_rms"][
                        scale_index, branch_index, :, component_index - 1
                    ]
                ).astype(np.float64)
                raw_values = raw_K[scale_index, branch_index, :, component_index]
                applied_values = K[scale_index, branch_index, :, component_index]
                saturated = arrays["component_scale"][
                    scale_index, branch_index, :, component_index
                ] < 1.0
                plus = plus_path[scale_index, branch_index, component_index]
                sign = sign_path[scale_index, branch_index, component_index]
                cp_sign = cp_sign_path[scale_index, branch_index, component_index]
                component_rows.append(
                    {
                        "bundle": root.name,
                        "rollback_internal_timestep": rollback,
                        "branch_key": branch_key,
                        "attempt_index": attempt,
                        "binary_discovery_label": labels.get(branch_key),
                        "delta_nu": delta_nu,
                        "component_index": component_index,
                        "component_name": component_name,
                        "effective_step_count": int(active.sum()),
                        "per_step_K_cap": float(cap[scale_index]),
                        "theta_rms_mean": float(theta_values[active].mean()),
                        "theta_rms_max": float(theta_values[active].max()),
                        "raw_K_total": float(raw_values.sum(dtype=np.float64)),
                        "applied_K_total": float(applied_values.sum(dtype=np.float64)),
                        "cap_saturation_count": int(np.count_nonzero(saturated & active)),
                        "cap_saturation_fraction": float(np.mean(saturated[active])),
                        "plus_final_log_e": float(plus[-1]),
                        "plus_running_max_log_e": _running_max(plus),
                        "minus_final_log_e": float(minus_path[scale_index, branch_index, component_index, -1]),
                        "minus_running_max_log_e": _running_max(
                            minus_path[scale_index, branch_index, component_index]
                        ),
                        "sign_mixture_final_log_e": float(sign[-1]),
                        "sign_mixture_running_max_log_e": _running_max(sign),
                        "change_point_sign_mixture_final_log_e": float(cp_sign[-1]),
                        "change_point_sign_mixture_running_max_log_e": _running_max(cp_sign),
                    }
                )

            scale_plus = plus_path[scale_index, branch_index]
            scale_sign = sign_path[scale_index, branch_index]
            scale_cp = cp_sign_path[scale_index, branch_index]
            scale_component_rows = component_rows[-COMPONENT_COUNT:]
            row = {
                "bundle": root.name,
                "rollback_internal_timestep": rollback,
                "branch_key": branch_key,
                "attempt_index": attempt,
                "binary_discovery_label": labels.get(branch_key),
                "scope": "single_scale",
                "delta_nu": delta_nu,
                "theta_rms_mean": float(
                    statistics.fmean(item["theta_rms_mean"] for item in scale_component_rows)
                ),
                "raw_K_mean_per_component": float(
                    statistics.fmean(item["raw_K_total"] for item in scale_component_rows)
                ),
                "applied_K_mean_per_component": float(
                    statistics.fmean(item["applied_K_total"] for item in scale_component_rows)
                ),
                "cap_saturation_fraction": float(
                    statistics.fmean(item["cap_saturation_fraction"] for item in scale_component_rows)
                ),
            }
            row.update(_aggregate_paths(scale_plus, scale_sign, scale_cp))
            mixture_rows.append(row)

    # All-scales uniform mixture.  Metrics preceding evidence values are plain
    # descriptive averages over the fixed scale/component family.
    for branch_index, attempt in enumerate(attempts.tolist()):
        branch_key = f"t{rollback}_attempt{attempt:03d}"
        rows = [item for item in component_rows if item["branch_key"] == branch_key]
        all_plus = plus_path[:, branch_index].reshape(-1, plus_path.shape[-1])
        all_sign = sign_path[:, branch_index].reshape(-1, sign_path.shape[-1])
        all_cp = cp_sign_path[:, branch_index].reshape(-1, cp_sign_path.shape[-1])
        row = {
            "bundle": root.name,
            "rollback_internal_timestep": rollback,
            "branch_key": branch_key,
            "attempt_index": attempt,
            "binary_discovery_label": labels.get(branch_key),
            "scope": "all_scales",
            "delta_nu": None,
            "theta_rms_mean": float(statistics.fmean(item["theta_rms_mean"] for item in rows)),
            "raw_K_mean_per_component": float(
                statistics.fmean(item["raw_K_total"] for item in rows)
            ),
            "applied_K_mean_per_component": float(
                statistics.fmean(item["applied_K_total"] for item in rows)
            ),
            "cap_saturation_fraction": float(
                statistics.fmean(item["cap_saturation_fraction"] for item in rows)
            ),
        }
        row.update(_aggregate_paths(all_plus, all_sign, all_cp))
        mixture_rows.append(row)

    bundle_record = {
        "root": str(root),
        "manifest_identity_sha256": manifest["identity_sha256"],
        "manifest_file_sha256": sha256_file(root / "manifest.json"),
        "results_file_sha256": sha256_file(root / "results.json"),
        "completion_file_sha256": sha256_file(root / "completion.json"),
        "trace_sha256": results["trace"]["sha256"],
        "input_suffix_manifest_identity_sha256": suffix_identity,
        "rollback_internal_timestep": rollback,
        "fresh_attempt_indices": attempts.tolist(),
        "delta_nu": scales.tolist(),
        "strict_raw_bundle_validation_passed": True,
    }
    return component_rows, mixture_rows, bundle_record


def add_tie_aware_ranks(rows: list[dict[str, Any]], metrics: Sequence[str]) -> None:
    """Add ranks within checkpoint/scope/scale/component, never across lengths."""

    group_fields = (
        ("rollback_internal_timestep", "delta_nu", "component_index")
        if "component_index" in rows[0]
        else ("rollback_internal_timestep", "scope", "delta_nu")
    )
    groups: dict[tuple[Any, ...], list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(tuple(row[field] for field in group_fields), []).append(index)
    for indices in groups.values():
        for metric in metrics:
            values = [float(rows[index][metric]) for index in indices]
            for local_index, row_index in enumerate(indices):
                rank = rank_interval_and_margins(values, local_index)
                for key, value in rank.items():
                    rows[row_index][f"{metric}__{key}"] = value


def _descriptive(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def label_comparisons(
    mixture_rows: list[dict[str, Any]], label_info: dict[str, Any] | None
) -> dict[str, Any] | None:
    if label_info is None:
        return None
    cohorts = label_info["cohorts"]
    matched = cohorts["primary_matched_late_suffix"]
    broader = cohorts["broader_clear_quality"]
    by_key_scope = {
        (row["branch_key"], row["scope"], row["delta_nu"]): row for row in mixture_rows
    }
    scopes = sorted(
        {(row["scope"], row["delta_nu"]) for row in mixture_rows},
        key=lambda item: (item[0], -1.0 if item[1] is None else float(item[1])),
    )
    matched_output = []
    for scope, delta_nu in scopes:
        bad_key, good_key = matched["bad"][0], matched["good"][0]
        bad = by_key_scope.get((bad_key, scope, delta_nu))
        good = by_key_scope.get((good_key, scope, delta_nu))
        if bad is None or good is None:
            continue
        matched_output.append(
            {
                "scope": scope,
                "delta_nu": delta_nu,
                "bad_branch": bad_key,
                "good_branch": good_key,
                "metrics": {
                    metric: {
                        "bad": bad[metric],
                        "good": good[metric],
                        "bad_minus_good": float(bad[metric] - good[metric]),
                        "bad_rank_desc_interval": [
                            bad[f"{metric}__rank_desc_min"],
                            bad[f"{metric}__rank_desc_max"],
                        ],
                        "good_rank_desc_interval": [
                            good[f"{metric}__rank_desc_min"],
                            good[f"{metric}__rank_desc_max"],
                        ],
                        "bad_tie_and_margin_details": {
                            key: bad[f"{metric}__{key}"]
                            for key in (
                                "tie_count",
                                "distance_to_nearest_higher",
                                "margin_above_nearest_lower",
                            )
                        },
                        "good_tie_and_margin_details": {
                            key: good[f"{metric}__{key}"]
                            for key in (
                                "tie_count",
                                "distance_to_nearest_higher",
                                "margin_above_nearest_lower",
                            )
                        },
                    }
                    for metric in MIXTURE_METRICS
                },
            }
        )

    broader_output = []
    for scope, delta_nu in scopes:
        good_rows = [
            by_key_scope[key]
            for branch in broader["good"]
            if (key := (branch, scope, delta_nu)) in by_key_scope
        ]
        bad_rows = [
            by_key_scope[key]
            for branch in broader["bad"]
            if (key := (branch, scope, delta_nu)) in by_key_scope
        ]
        # Only emit the requested broader comparison once all 10/3 members
        # exist.  Partial raw-bundle availability is reported separately.
        if len(good_rows) != 10 or len(bad_rows) != 3:
            continue
        broader_output.append(
            {
                "scope": scope,
                "delta_nu": delta_nu,
                "good_count": 10,
                "bad_count": 3,
                "metrics": {
                    metric: {
                        "good": _descriptive([row[metric] for row in good_rows]),
                        "bad": _descriptive([row[metric] for row in bad_rows]),
                        "bad_mean_minus_good_mean": float(
                            statistics.fmean(row[metric] for row in bad_rows)
                            - statistics.fmean(row[metric] for row in good_rows)
                        ),
                        "bad_median_minus_good_median": float(
                            statistics.median(row[metric] for row in bad_rows)
                            - statistics.median(row[metric] for row in good_rows)
                        ),
                    }
                    for metric in MIXTURE_METRICS
                },
            }
        )
    available = {row["branch_key"] for row in mixture_rows}
    return {
        "status": "POSTHOC_DISCOVERY_ONLY_NOT_CONFIRMATORY",
        "no_threshold_selection": True,
        "no_significance_test": True,
        "no_TPR_FPR_or_generalization_estimate": True,
        "matched_t60_attempt003_bad_vs_attempt004_good": matched_output,
        "broader_3_bad_vs_10_good": broader_output,
        "broader_complete": all(
            branch in available for branch in broader["good"] + broader["bad"]
        ),
        "missing_broader_members": [
            branch for branch in broader["good"] + broader["bad"] if branch not in available
        ],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty CSV")
    fields = list(rows[0])
    if any(set(row) != set(fields) for row in rows):
        raise RuntimeError("CSV rows have inconsistent schemas")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def run_summary(args: argparse.Namespace) -> None:
    labels, label_info = _load_labels(args.labels)
    component_rows: list[dict[str, Any]] = []
    mixture_rows: list[dict[str, Any]] = []
    bundle_records = []
    seen_rollbacks: set[int] = set()
    for root in args.bundle_dir:
        components, mixtures, bundle = summarize_bundle(root, labels=labels)
        rollback = int(bundle["rollback_internal_timestep"])
        if rollback in seen_rollbacks:
            raise RuntimeError(f"duplicate rollback bundle supplied: t={rollback}")
        seen_rollbacks.add(rollback)
        component_rows.extend(components)
        mixture_rows.extend(mixtures)
        bundle_records.append(bundle)
    add_tie_aware_ranks(component_rows, COMPONENT_METRICS)
    add_tie_aware_ranks(mixture_rows, MIXTURE_METRICS)
    comparisons = label_comparisons(mixture_rows, label_info)

    runner = Path(__file__).resolve()
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "role": "OFFLINE_RECOMPUTATION_AND_DISCOVERY_SUMMARY",
        "raw_replay_bundles": bundle_records,
        "labels": label_info,
        "labels_affect_metric_or_mixture_construction": False,
        "label_comparisons_discovery_only": label_info is not None,
        "image_or_label_dependent_ROI_used": False,
        "threshold_selection_performed": False,
        "significance_testing_performed": False,
        "mixtures": {
            "sign_prior": {"plus_theta": 0.5, "minus_theta": 0.5},
            "minus_increment": "L_minus=-R-K (not -L_plus)",
            "change_point_start_prior": "uniform over every stochastic suffix transition, fixed at suffix entry",
            "future_not_started_component_e_value": 1.0,
            "scale_prior": "uniform over predeclared raw replay Delta-nu values",
            "spatial_component_prior": "uniform over fixed global + row-major 4x4 latent components",
        },
        "ranks": {
            "direction": "descending numeric value; not interpreted as quality direction",
            "within": "same rollback, scale/scope, and component only",
            "relative_tolerance": RANK_RELATIVE_TOLERANCE,
            "absolute_tolerance": RANK_ABSOLUTE_TOLERANCE,
            "tie_aware_interval_and_nearest_distinct_margins": True,
        },
        "runner": {"path": str(runner), "sha256": sha256_file(runner)},
        "outdir": str(args.outdir),
        "no_overwrite": True,
    }
    manifest["identity_sha256"] = _canonical_self_hash(manifest, "identity_sha256")
    args.outdir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{args.outdir.name}.staging-", dir=args.outdir.parent
    ) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        atomic_json_dump(manifest, staging / "manifest.json")
        _write_csv(staging / COMPONENT_CSV, component_rows)
        _write_csv(staging / MIXTURE_CSV, mixture_rows)
        summary: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "manifest_identity_sha256": manifest["identity_sha256"],
            "discovery_only": True,
            "component_row_count": len(component_rows),
            "mixture_row_count": len(mixture_rows),
            "rollbacks": sorted(seen_rollbacks),
            "label_comparisons": comparisons,
            "files": [
                _file_record(staging / COMPONENT_CSV, staging),
                _file_record(staging / MIXTURE_CSV, staging),
            ],
        }
        summary["payload_sha256"] = _canonical_self_hash(summary, "payload_sha256")
        atomic_json_dump(summary, staging / SUMMARY_JSON)
        completion: dict[str, Any] = {
            "complete": True,
            "manifest_identity_sha256": manifest["identity_sha256"],
            "manifest_file_sha256": sha256_file(staging / "manifest.json"),
            "summary_payload_sha256": summary["payload_sha256"],
            "summary_file_sha256": sha256_file(staging / SUMMARY_JSON),
            "component_csv_sha256": sha256_file(staging / COMPONENT_CSV),
            "mixture_csv_sha256": sha256_file(staging / MIXTURE_CSV),
        }
        completion["payload_sha256"] = _canonical_self_hash(completion, "payload_sha256")
        atomic_json_dump(completion, staging / "completion.json")
        if args.outdir.exists():
            raise RuntimeError("summary output appeared during staging; refusing overwrite")
        _atomic_install_directory_noreplace(staging, args.outdir)
    print(
        json.dumps(
            {
                "outdir": str(args.outdir),
                "component_rows": len(component_rows),
                "mixture_rows": len(mixture_rows),
                "rollbacks": sorted(seen_rollbacks),
                "broader_comparison_complete": bool(
                    comparisons and comparisons["broader_complete"]
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def run_self_test() -> None:
    # One component with increments [a,b,c].  At k=0 the fixed three-start
    # mixture is (exp(a)+1+1)/3; future starts must remain E=1.
    increments = np.asarray([[0.2, -0.1, 0.3]], dtype=np.float64)
    observed = uniform_change_point_log_mixture(increments, start_count=3)[0]
    expected = np.asarray(
        [
            math.log((math.exp(0.2) + 2.0) / 3.0),
            math.log((math.exp(0.1) + math.exp(-0.1) + 1.0) / 3.0),
            math.log((math.exp(0.4) + math.exp(0.2) + math.exp(0.3)) / 3.0),
        ]
    )
    if not np.allclose(observed, expected, rtol=0.0, atol=2e-16):
        raise AssertionError("fixed-prior change-point mixture is wrong")

    # Terminal bookkeeping after all starts: no new start is introduced.
    terminal = uniform_change_point_log_mixture(
        np.asarray([[0.2, -0.1, 0.0]], dtype=np.float64), start_count=2
    )[0]
    if terminal[-1] != terminal[-2]:
        raise AssertionError("zero terminal increment changed change-point evidence")

    R = np.asarray([0.5, -0.25], dtype=np.float64)
    K = np.asarray([0.125, 0.0625], dtype=np.float64)
    plus_increment = R - K
    minus_increment = -R - K
    if np.array_equal(minus_increment, -plus_increment):
        raise AssertionError("test fixture failed to distinguish -R-K from -L")
    expected_minus = np.asarray([-0.625, 0.1875])
    if not np.array_equal(minus_increment, expected_minus):
        raise AssertionError("negative-theta Gaussian LR must be -R-K")
    plus_path = np.cumsum(plus_increment)
    minus_path = np.cumsum(minus_increment)
    sign = fixed_sign_log_mixture(plus_path, minus_path)
    manual = np.log((np.exp(plus_path) + np.exp(minus_path)) / 2.0)
    if not np.allclose(sign, manual, rtol=0.0, atol=2e-16):
        raise AssertionError("fixed 50/50 sign mixture is wrong")

    ranks = [rank_interval_and_margins([2.0, 2.0 + 1e-16, 1.0], index) for index in range(3)]
    if ranks[0]["rank_desc_min"] != 1 or ranks[0]["rank_desc_max"] != 2:
        raise AssertionError("near-tie rank interval failed")
    if ranks[1]["tie_count"] != 2 or ranks[2]["rank_desc_min"] != 3:
        raise AssertionError("tie-aware rank count/order failed")
    if ranks[0]["margin_above_nearest_lower"] != 1.0:
        raise AssertionError("nearest distinct lower margin failed")

    # Monte Carlo sanity: fixed 50/50 sign and fixed-start mixtures each have
    # mean E approximately one under standard-normal innovations.
    rng = np.random.default_rng(7)
    sign_e = []
    cp_e = []
    u = 0.12
    for _ in range(20_000):
        z = rng.normal(size=3)
        reward = u * z
        cost = np.full(3, 0.5 * u * u)
        plus = reward - cost
        minus = -reward - cost
        plus_log = np.cumsum(plus)
        minus_log = np.cumsum(minus)
        sign_e.append(float(np.exp(fixed_sign_log_mixture(plus_log, minus_log)[-1])))
        cp_plus = uniform_change_point_log_mixture(plus[None], start_count=3)[0, -1]
        cp_minus = uniform_change_point_log_mixture(minus[None], start_count=3)[0, -1]
        cp_e.append(float(np.exp(fixed_sign_log_mixture(np.asarray(cp_plus), np.asarray(cp_minus)))))
    if abs(float(np.mean(sign_e)) - 1.0) > 0.015:
        raise AssertionError("sign-mixture Monte Carlo calibration failed")
    if abs(float(np.mean(cp_e)) - 1.0) > 0.015:
        raise AssertionError("change-point/sign mixture Monte Carlo calibration failed")
    print("self-test passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        action="append",
        default=[],
        help="Completed raw replay bundle; repeat once per rollback bundle.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Optional v2 posthoc discovery label JSON.",
    )
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--self-test", action="store_true")
    return parser


def normalize_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.self_test:
        return
    if not args.bundle_dir:
        parser.error("at least one --bundle-dir is required")
    args.bundle_dir = [path.expanduser().absolute().resolve() for path in args.bundle_dir]
    if len(set(args.bundle_dir)) != len(args.bundle_dir):
        parser.error("duplicate --bundle-dir")
    for path in args.bundle_dir:
        if not path.is_dir():
            parser.error(f"raw replay bundle is not a directory: {path}")
    args.labels = None if args.labels is None else args.labels.expanduser().absolute().resolve()
    if args.labels is not None and not args.labels.is_file():
        parser.error(f"label file does not exist: {args.labels}")
    identity = hashlib.sha256(
        "\0".join(str(path) for path in sorted(args.bundle_dir)).encode("utf-8")
    ).hexdigest()[:8]
    data_root = Path(os.environ.get("EQVAE_DATA_ROOT", "/home/zhoushunyu/data/eqvae"))
    default_out = (
        data_root
        / "cross_scale_evidence/dit_imagenet256_suffix_cross_scale_summary"
        / f"replay_summary_{identity}_{sha256_file(Path(__file__).resolve())[:7]}"
    )
    args.outdir = (
        default_out.expanduser().absolute().resolve()
        if args.outdir is None
        else args.outdir.expanduser().absolute().resolve()
    )
    if os.path.lexists(args.outdir):
        parser.error(f"no-overwrite target already exists: {args.outdir}")
    protected = [*args.bundle_dir, Path(__file__).resolve().parent.parent]
    if args.labels is not None:
        protected.append(args.labels)
    for path in protected:
        if args.outdir == path or args.outdir in path.parents or path in args.outdir.parents:
            parser.error(f"--outdir overlaps protected input/source: {path}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    normalize_args(args, parser)
    if args.self_test:
        run_self_test()
    else:
        run_summary(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
