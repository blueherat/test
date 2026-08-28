#!/usr/bin/env python3
"""Extract frozen, sampler-internal branch-consensus features from DiT suffix traces.

This program intentionally has no argument for reviews, labels, mappings, decoded
images, FID, Inception, DINO, CLIP, embeddings, AUCs, or thresholds.  It reads
only each job's manifest, each branch's mechanical metadata, and two arrays from
trace.npz.  Attempt 0 is retrospectively selected in the existing pilot, so its
rank is emitted as descriptive and explicitly not conformal-calibrated.  The
fresh attempts 1..4 remain conditionally exchangeable with one another.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


sys.dont_write_bytecode = True

EXPERIMENT = "dit_v22_branch_consensus_label_free_v1"
DEFAULT_INPUT_ROOT = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "dit_v22_repairability_pilot_v1_2_outputs"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "dit_v22_branch_consensus_label_free_v1"
)
DEFAULT_CONFIG = (
    Path(__file__).resolve().parent
    / "configs"
    / "dit_v22_branch_consensus_label_free_v1.json"
)

TRACE_KEYS_READ = ("internal_timestep", "target_pred_xstart")
FORBIDDEN_INPUT_TOKENS = (
    "review",
    "consensus",
    "mapping",
    "annotation",
    "label",
    "inception",
    "dino",
    "clip",
    "embedding",
    "fid",
    "auc",
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def require_regular(path: Path, label: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{label} must be a regular non-symlink file: {path}")
    return path


def require_directory(path: Path, label: str) -> Path:
    if not path.is_dir() or path.is_symlink():
        raise RuntimeError(f"{label} must be a real directory: {path}")
    return path


def reject_forbidden_input_path(path: Path) -> None:
    lowered = str(path.resolve()).lower()
    hits = sorted(token for token in FORBIDDEN_INPUT_TOKENS if token in lowered)
    if hits:
        raise RuntimeError(
            f"input root contains forbidden external/supervision token(s) {hits}: {path}"
        )


def validate_config(config: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": 1,
        "method_version": "dit_v22_prefix_conditional_branch_consensus_v1",
        "scientific_role": "posthoc_label_free_feature_discovery_only",
        "primary_horizon": 10,
        "secondary_horizons": [5, 20],
        "all_horizons": [5, 10, 20],
        "pool_scales": [1, 2, 4],
        "expected_jobs": 32,
        "expected_attempts_per_job": 5,
        "fresh_attempt_indices": [1, 2, 3, 4],
        "trace_array": "target_pred_xstart",
        "prefix_index": 0,
    }
    for key, expected_value in expected.items():
        if config.get(key) != expected_value:
            raise RuntimeError(
                f"frozen config field {key!r} changed: "
                f"expected {expected_value!r}, got {config.get(key)!r}"
            )
    epsilon = config.get("epsilon")
    if not isinstance(epsilon, (int, float)) or float(epsilon) != 1e-6:
        raise RuntimeError("frozen epsilon must be exactly 1e-6")
    boundary = config.get("evaluation_contract")
    if not isinstance(boundary, dict) or not all(
        boundary.get(key) is True
        for key in (
            "extractor_must_finish_and_hash_outputs_before_any_label_join",
            "current_selected_pilot_is_posthoc_discovery_only",
            "future_confirmation_requires_unselected_symmetric_current_and_scout_suffixes",
        )
    ):
        raise RuntimeError("evaluation boundary contract changed")


def spatial_center(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError(f"expected [C,H,W], got {array.shape}")
    return array - np.mean(array, axis=(-2, -1), keepdims=True, dtype=np.float64)


def average_pool(value: np.ndarray, scale: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 3 or scale <= 0:
        raise ValueError("average_pool expects [C,H,W] and a positive scale")
    channels, height, width = array.shape
    if height % scale or width % scale:
        raise ValueError(f"shape {array.shape} is not divisible by pooling scale {scale}")
    return np.mean(
        array.reshape(
            channels,
            height // scale,
            scale,
            width // scale,
            scale,
        ),
        axis=(2, 4),
        dtype=np.float64,
    )


def rms(value: np.ndarray) -> float:
    array = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(array * array, dtype=np.float64)))


def prefix_normalizers(
    prefix: np.ndarray, scales: Sequence[int], epsilon: float
) -> tuple[dict[int, float], dict[int, float]]:
    centered = spatial_center(prefix)
    raw = {int(scale): rms(average_pool(centered, int(scale))) for scale in scales}
    effective = {scale: value + epsilon for scale, value in raw.items()}
    if not all(math.isfinite(value) and value > 0.0 for value in effective.values()):
        raise RuntimeError("prefix normalization is non-finite or non-positive")
    return raw, effective


def multiscale_distance(
    left: np.ndarray,
    right: np.ndarray,
    scales: Sequence[int],
    effective_normalizers: Mapping[int, float],
) -> float:
    delta = spatial_center(
        np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    )
    terms = [
        rms(average_pool(delta, int(scale))) / effective_normalizers[int(scale)]
        for scale in scales
    ]
    value = float(np.mean(np.asarray(terms, dtype=np.float64), dtype=np.float64))
    if not math.isfinite(value) or value < 0.0:
        raise RuntimeError("branch distance is non-finite or negative")
    return value


def distance_matrix(
    branches: np.ndarray,
    scales: Sequence[int],
    effective_normalizers: Mapping[int, float],
) -> np.ndarray:
    values = np.asarray(branches)
    if values.ndim != 4:
        raise ValueError(f"branches must be [K,C,H,W], got {values.shape}")
    count = values.shape[0]
    matrix = np.zeros((count, count), dtype=np.float64)
    for left in range(count):
        for right in range(left + 1, count):
            value = multiscale_distance(
                values[left], values[right], scales, effective_normalizers
            )
            matrix[left, right] = value
            matrix[right, left] = value
    if not np.array_equal(matrix, matrix.T) or np.any(np.diag(matrix) != 0.0):
        raise RuntimeError("distance matrix symmetry/diagonal invariant failed")
    return matrix


def mean_nonconformity(matrix: np.ndarray, indices: Sequence[int]) -> dict[int, float]:
    result: dict[int, float] = {}
    for index in indices:
        others = [other for other in indices if other != index]
        if not others:
            raise ValueError("at least two branches are required")
        result[int(index)] = float(
            np.mean(matrix[index, others], dtype=np.float64)
        )
    return result


def conservative_rank_p(nonconformity: Mapping[int, float]) -> dict[int, float]:
    count = len(nonconformity)
    return {
        int(index): float(
            sum(value >= score for value in nonconformity.values()) / count
        )
        for index, score in nonconformity.items()
    }


def stable_medoid(nonconformity: Mapping[int, float]) -> tuple[int, bool]:
    minimum = min(nonconformity.values())
    tied = sorted(
        index
        for index, value in nonconformity.items()
        if math.isclose(value, minimum, rel_tol=0.0, abs_tol=1e-15)
    )
    return int(tied[0]), len(tied) > 1


def feature_record(
    *,
    identity: Mapping[str, Any],
    horizon: int,
    internal_timestep: int,
    branches: np.ndarray,
    scales: Sequence[int],
    epsilon: float,
    fresh_indices: Sequence[int],
    primary_horizon: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_normalizers, effective_normalizers = prefix_normalizers(
        branches[0], scales, epsilon
    )
    matrix = distance_matrix(branches, scales, effective_normalizers)

    fresh_pairs = [
        float(matrix[left, right])
        for position, left in enumerate(fresh_indices)
        for right in fresh_indices[position + 1 :]
    ]
    if len(fresh_pairs) != 6:
        raise RuntimeError("four fresh branches must produce six pairwise distances")
    fresh_dispersion = float(np.median(np.asarray(fresh_pairs, dtype=np.float64)))
    attempt0_to_fresh = float(
        np.mean(matrix[0, list(fresh_indices)], dtype=np.float64)
    )
    attempt0_outlier_ratio = attempt0_to_fresh / (fresh_dispersion + epsilon)

    fresh_nonconformity = mean_nonconformity(matrix, fresh_indices)
    fresh_rank_p = conservative_rank_p(fresh_nonconformity)
    fresh_medoid, fresh_medoid_tie = stable_medoid(fresh_nonconformity)

    all_indices = list(range(branches.shape[0]))
    all_nonconformity = mean_nonconformity(matrix, all_indices)
    all_rank_p = conservative_rank_p(all_nonconformity)
    all_pairs = [
        float(matrix[left, right])
        for left in all_indices
        for right in all_indices[left + 1 :]
    ]

    scalar = {
        **identity,
        "horizon": int(horizon),
        "horizon_internal_timestep": int(internal_timestep),
        "primary_horizon": int(horizon) == int(primary_horizon),
        "branch_count": int(branches.shape[0]),
        "fresh_branch_count": len(fresh_indices),
        "attempt0_exchangeable_with_fresh": False,
        "attempt0_rank_calibrated": False,
        "fresh_only_rank_exchangeability_eligible": True,
        "normalizer_scale1_raw": raw_normalizers[1],
        "normalizer_scale2_raw": raw_normalizers[2],
        "normalizer_scale4_raw": raw_normalizers[4],
        "fresh_dispersion_D": fresh_dispersion,
        "attempt0_to_fresh_mean_A": attempt0_to_fresh,
        "attempt0_outlier_ratio_O": attempt0_outlier_ratio,
        "all5_dispersion_descriptive": float(
            np.median(np.asarray(all_pairs, dtype=np.float64))
        ),
        "attempt0_all5_nonconformity_descriptive": all_nonconformity[0],
        "attempt0_all5_rank_p_descriptive": all_rank_p[0],
        "fresh_medoid_attempt": fresh_medoid,
        "fresh_medoid_tie": fresh_medoid_tie,
        "fresh_nonconformity_min": min(fresh_nonconformity.values()),
        "fresh_nonconformity_mean": float(
            np.mean(list(fresh_nonconformity.values()), dtype=np.float64)
        ),
        "fresh_nonconformity_max": max(fresh_nonconformity.values()),
    }
    for index in fresh_indices:
        scalar[f"fresh_attempt{index}_nonconformity"] = fresh_nonconformity[index]
        scalar[f"fresh_attempt{index}_rank_p"] = fresh_rank_p[index]

    detailed = {
        **identity,
        "horizon": int(horizon),
        "horizon_internal_timestep": int(internal_timestep),
        "distance_matrix_attempt_order_0_to_4": matrix.tolist(),
        "prefix_normalizer_raw_by_scale": {
            str(key): value for key, value in raw_normalizers.items()
        },
        "prefix_normalizer_effective_by_scale": {
            str(key): value for key, value in effective_normalizers.items()
        },
        "fresh_pairwise_distances": fresh_pairs,
        "fresh_dispersion_D": fresh_dispersion,
        "attempt0_to_fresh_mean_A": attempt0_to_fresh,
        "attempt0_outlier_ratio_O": attempt0_outlier_ratio,
        "fresh_nonconformity": {
            str(key): value for key, value in fresh_nonconformity.items()
        },
        "fresh_rank_p": {str(key): value for key, value in fresh_rank_p.items()},
        "fresh_medoid_attempt": fresh_medoid,
        "fresh_medoid_tie": fresh_medoid_tie,
        "all5_nonconformity_descriptive": {
            str(key): value for key, value in all_nonconformity.items()
        },
        "all5_rank_p_descriptive": {
            str(key): value for key, value in all_rank_p.items()
        },
        "calibration_warning": (
            "attempt0 was retrospectively selected using future B/E path information; "
            "attempt0 ranks are descriptive, not conformal p-values"
        ),
    }
    return scalar, detailed


def float_text(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise RuntimeError("non-finite scalar cannot be serialized")
        return format(float(value), ".17g")
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    return value


def rows_to_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        raise RuntimeError("feature table is empty")
    columns = list(rows[0])
    if any(list(row) != columns for row in rows):
        raise RuntimeError("feature rows do not share one frozen column order")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: float_text(value) for key, value in row.items()})
    return buffer.getvalue()


def jsonl_text(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        for row in rows
    )


def branch_arrays_and_inventory(
    job_dir: Path,
    expected_attempts: int,
) -> tuple[list[np.ndarray], np.ndarray, list[dict[str, Any]]]:
    arrays: list[np.ndarray] = []
    reference_timesteps: np.ndarray | None = None
    inventory: list[dict[str, Any]] = []
    expected_names = [f"attempt_{index:03d}" for index in range(expected_attempts)]
    actual_names = sorted(
        path.name
        for path in require_directory(job_dir / "branches", "branches directory").iterdir()
        if path.is_dir() and not path.is_symlink()
    )
    if actual_names != expected_names:
        raise RuntimeError(
            f"{job_dir.name} branch directories changed: {actual_names!r}"
        )

    for attempt_index, name in enumerate(expected_names):
        branch_dir = job_dir / "branches" / name
        branch_path = require_regular(branch_dir / "branch.json", "branch metadata")
        trace_path = require_regular(branch_dir / "trace.npz", "branch trace")
        branch = load_json(branch_path)
        if branch.get("attempt_index") != attempt_index or branch.get("branch") != name:
            raise RuntimeError(f"branch identity mismatch in {branch_path}")
        if attempt_index == 0 and branch.get("role") != "exact_baseline_replay":
            raise RuntimeError("attempt 0 is not the exact replay")
        if attempt_index > 0 and branch.get("role") != "fresh_target_suffix":
            raise RuntimeError(f"attempt {attempt_index} is not a fresh suffix")
        recorded_trace = branch.get("trace_npz")
        if not isinstance(recorded_trace, dict):
            raise RuntimeError(f"missing trace record in {branch_path}")
        trace_sha256 = sha256_file(trace_path)
        if recorded_trace.get("sha256") != trace_sha256:
            raise RuntimeError(f"trace hash mismatch: {trace_path}")

        with np.load(trace_path, allow_pickle=False) as payload:
            for key in TRACE_KEYS_READ:
                if key not in payload.files:
                    raise RuntimeError(f"trace key {key!r} missing from {trace_path}")
            timesteps = np.asarray(payload["internal_timestep"])
            pred = np.asarray(payload["target_pred_xstart"])
        if timesteps.ndim != 1 or not np.issubdtype(timesteps.dtype, np.integer):
            raise RuntimeError(f"invalid internal_timestep schema in {trace_path}")
        if pred.dtype != np.float32 or pred.shape != (len(timesteps), 4, 32, 32):
            raise RuntimeError(f"invalid target_pred_xstart schema in {trace_path}: {pred.shape}")
        if not np.isfinite(pred).all():
            raise RuntimeError(f"non-finite target_pred_xstart in {trace_path}")
        if reference_timesteps is None:
            reference_timesteps = timesteps.copy()
        elif not np.array_equal(reference_timesteps, timesteps):
            raise RuntimeError(f"branch timestep grids differ in {job_dir}")
        arrays.append(np.ascontiguousarray(pred))
        inventory.extend(
            [
                {
                    "path": str(branch_path.relative_to(job_dir.parent)),
                    "bytes": branch_path.stat().st_size,
                    "sha256": sha256_file(branch_path),
                    "opened_as": "mechanical_branch_metadata",
                },
                {
                    "path": str(trace_path.relative_to(job_dir.parent)),
                    "bytes": trace_path.stat().st_size,
                    "sha256": trace_sha256,
                    "opened_as": "npz_internal_arrays_only",
                    "array_keys_read": list(TRACE_KEYS_READ),
                },
            ]
        )
    if reference_timesteps is None:
        raise RuntimeError(f"no branches found in {job_dir}")
    return arrays, reference_timesteps, inventory


def extract_product(
    *,
    input_root: Path,
    output_root: Path,
    config_path: Path,
) -> dict[str, Any]:
    input_root = require_directory(input_root.resolve(), "input root")
    reject_forbidden_input_path(input_root)
    config_path = require_regular(config_path.resolve(), "frozen config")
    config = load_json(config_path)
    validate_config(config)
    if output_root.exists():
        raise RuntimeError(f"output already exists; refusing overwrite: {output_root}")

    job_dirs = sorted(
        path
        for path in input_root.iterdir()
        if path.is_dir() and not path.is_symlink() and (path / "manifest.json").is_file()
    )
    if len(job_dirs) != int(config["expected_jobs"]):
        raise RuntimeError(
            f"expected {config['expected_jobs']} job directories, found {len(job_dirs)}"
        )

    source_path = require_regular(Path(__file__).resolve(), "extractor source")
    source_sha256 = sha256_file(source_path)
    config_sha256 = sha256_file(config_path)
    scales = tuple(int(value) for value in config["pool_scales"])
    horizons = tuple(int(value) for value in config["all_horizons"])
    fresh_indices = tuple(int(value) for value in config["fresh_attempt_indices"])
    epsilon = float(config["epsilon"])

    features: list[dict[str, Any]] = []
    matrices: list[dict[str, Any]] = []
    input_inventory: list[dict[str, Any]] = []
    common_prefix_hashes: list[dict[str, Any]] = []

    for job_dir in job_dirs:
        manifest_path = require_regular(job_dir / "manifest.json", "job manifest")
        manifest = load_json(manifest_path)
        if manifest.get("experiment") != "dit_v22_custom_trace_suffix_repairability":
            raise RuntimeError(f"unexpected experiment in {manifest_path}")
        if manifest.get("quality_scores_or_labels_used_by_runner") is not False:
            raise RuntimeError(f"runner supervision boundary failed in {manifest_path}")
        if manifest.get("FID_Inception_DINO_CLIP_or_embeddings_used") is not False:
            raise RuntimeError(f"runner external-feature boundary failed in {manifest_path}")
        if manifest.get("attempt_ranking_or_selection") is not False:
            raise RuntimeError(f"runner selected an attempt in {manifest_path}")
        if manifest.get("best_of_n") is not False:
            raise RuntimeError(f"runner used best-of-N in {manifest_path}")

        target = manifest.get("target")
        rollback = manifest.get("rollback")
        pilot = manifest.get("pilot_binding")
        if not all(isinstance(value, dict) for value in (target, rollback, pilot)):
            raise RuntimeError(f"missing target/rollback/pilot identity in {manifest_path}")
        identity = {
            "job_id": job_dir.name,
            "pair_index": int(pilot["pair_index"]),
            "legacy_selection_role": str(pilot["selected_role"]),
            "global_seed": int(target["global_seed"]),
            "class_id": int(target["class_id"]),
            "target_slot": int(target["slot"]),
            "rollback_step": int(rollback["sampling_step_index_zero_based"]),
            "rollback_internal_timestep": int(rollback["internal_timestep"]),
            "source_manifest_identity_sha256": str(manifest["identity_sha256"]),
        }

        branch_arrays, timesteps, branch_inventory = branch_arrays_and_inventory(
            job_dir, int(config["expected_attempts_per_job"])
        )
        input_inventory.append(
            {
                "path": str(manifest_path.relative_to(input_root)),
                "bytes": manifest_path.stat().st_size,
                "sha256": sha256_file(manifest_path),
                "opened_as": "job_manifest",
            }
        )
        input_inventory.extend(branch_inventory)
        if int(timesteps[0]) != identity["rollback_internal_timestep"]:
            raise RuntimeError(f"rollback timestep mismatch in {job_dir}")
        if len(timesteps) != int(rollback["suffix_transition_count_including_t0"]):
            raise RuntimeError(f"trace length mismatch in {job_dir}")
        if any(horizon >= len(timesteps) for horizon in horizons):
            raise RuntimeError(f"frozen horizon exceeds trace length in {job_dir}")

        prefix = branch_arrays[0][int(config["prefix_index"])]
        prefix_raw = np.ascontiguousarray(prefix).tobytes(order="C")
        prefix_sha256 = hashlib.sha256(prefix_raw).hexdigest()
        for attempt_index, array in enumerate(branch_arrays[1:], start=1):
            if not np.array_equal(prefix, array[int(config["prefix_index"])]):
                raise RuntimeError(
                    f"branches do not share exact predicted-clean prefix in {job_dir}; "
                    f"attempt={attempt_index}"
                )
        common_prefix_hashes.append(
            {
                **identity,
                "target_pred_xstart_prefix_raw_sha256": prefix_sha256,
                "all_five_exactly_equal": True,
            }
        )

        for horizon in horizons:
            branch_horizon = np.stack(
                [array[horizon] for array in branch_arrays], axis=0
            )
            scalar, detailed = feature_record(
                identity=identity,
                horizon=horizon,
                internal_timestep=int(timesteps[horizon]),
                branches=branch_horizon,
                scales=scales,
                epsilon=epsilon,
                fresh_indices=fresh_indices,
                primary_horizon=int(config["primary_horizon"]),
            )
            features.append(scalar)
            matrices.append(detailed)

    expected_feature_rows = int(config["expected_jobs"]) * len(horizons)
    if len(features) != expected_feature_rows or len(matrices) != expected_feature_rows:
        raise RuntimeError("feature row count changed")
    if len({row["job_id"] for row in features}) != int(config["expected_jobs"]):
        raise RuntimeError("job identity count changed")
    if any(
        value
        for row in features
        for key, value in row.items()
        if isinstance(value, float) and not math.isfinite(value)
    ):
        raise RuntimeError("feature product contains a non-finite value")

    inventory_payload = {
        "schema_version": 1,
        "input_root": str(input_root),
        "opened_file_count": len(input_inventory),
        "opened_files": sorted(input_inventory, key=lambda row: row["path"]),
        "opened_basenames": ["manifest.json", "branch.json", "trace.npz"],
        "trace_array_keys_read": list(TRACE_KEYS_READ),
        "png_files_opened": False,
        "review_label_mapping_or_external_feature_files_opened": False,
    }
    inventory_identity = canonical_sha256(inventory_payload)
    inventory_payload["identity_sha256"] = inventory_identity

    catalog = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "method_version": config["method_version"],
        "formula": {
            "channel_center": "C(X)=X-spatial_mean_per_latent_channel(X)",
            "distance": (
                "d_h(m,n)=mean_{s in {1,2,4}} RMS(A_s C(X_h^m-X_h^n)) / "
                "(RMS(A_s C(R))+1e-6)"
            ),
            "fresh_dispersion_D": "median of six pairwise d values among attempts 1..4",
            "attempt0_to_fresh_A": "mean_{r=1..4} d(0,r)",
            "attempt0_outlier_O": "A/(D+1e-6)",
            "branch_nonconformity": "mean distance to all other branches in the declared set",
            "conservative_rank": "fraction of declared-set nonconformities >= own value",
        },
        "horizons": list(horizons),
        "primary_horizon": config["primary_horizon"],
        "pool_scales": list(scales),
        "epsilon": epsilon,
        "interpretation": {
            "low_D_high_O": "candidate sampling accident: fresh alternatives agree and attempt0 is isolated",
            "high_D": "prefix-conditional futures are broadly unstable; repair by resampling is uncertain",
            "low_D_low_O": "attempt0 follows the common conditional future; shared errors are not detectable",
        },
        "calibration_boundary": {
            "attempt0": "descriptive only because its future participated in retrospective B/E selection",
            "fresh_attempts_1_to_4": "mutually exchangeable rank eligible, minimum attainable p=0.25",
            "bad_image_probability_claim": False,
            "external_quality_semantics_required_later": True,
        },
        "external_metrics_role": "evaluation_only_and_not_opened_by_this_product",
    }

    manifest_payload = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "scientific_role": config["scientific_role"],
        "method_version": config["method_version"],
        "source_sha256": source_sha256,
        "config_sha256": config_sha256,
        "input_inventory_identity_sha256": inventory_identity,
        "counts": {
            "jobs": len(job_dirs),
            "branches": len(job_dirs) * int(config["expected_attempts_per_job"]),
            "feature_rows": len(features),
            "distance_matrix_rows": len(matrices),
            "horizons": len(horizons),
        },
        "roles": {
            "method_inputs": "frozen sampler-internal predicted-clean latent trajectories only",
            "external_judges": "not read; later evaluator only",
            "online_trigger_ready": False,
            "current_product_confirmatory": False,
        },
        "attempt0_exchangeability_warning": (
            "attempt0 future participated in retrospective B/E selection; "
            "do not interpret its rank as conformal or super-uniform"
        ),
        "all_outputs_retained": True,
        "feature_selection_auc_thresholding_or_label_join_performed": False,
        "png_decoding_or_external_model_used": False,
        "common_prefix_exact_equality_verified_for_all_jobs": True,
        "common_prefix_records_sha256": canonical_sha256(common_prefix_hashes),
    }
    manifest_payload["identity_sha256"] = canonical_sha256(manifest_payload)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent)
    )
    try:
        (temporary / "features.csv").write_text(
            rows_to_csv(features), encoding="utf-8"
        )
        (temporary / "features.jsonl").write_text(
            jsonl_text(features), encoding="utf-8"
        )
        (temporary / "distance_matrices.jsonl").write_text(
            jsonl_text(matrices), encoding="utf-8"
        )
        (temporary / "common_prefix_hashes.jsonl").write_text(
            jsonl_text(common_prefix_hashes), encoding="utf-8"
        )
        (temporary / "input_inventory.json").write_text(
            json.dumps(inventory_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (temporary / "catalog.json").write_text(
            json.dumps(catalog, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (temporary / "manifest.json").write_text(
            json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        shutil.copyfile(source_path, temporary / "extractor_source.py")
        shutil.copyfile(config_path, temporary / "frozen_config.json")

        payload_files = sorted(
            path for path in temporary.iterdir() if path.is_file()
        )
        file_records = {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in payload_files
        }
        completion_payload = {
            "schema_version": 1,
            "experiment": EXPERIMENT,
            "manifest_identity_sha256": manifest_payload["identity_sha256"],
            "source_sha256": source_sha256,
            "config_sha256": config_sha256,
            "input_inventory_identity_sha256": inventory_identity,
            "files": file_records,
            "all_required_files_hashed": True,
            "label_free_boundary_passed": True,
        }
        completion_payload["product_identity_sha256"] = canonical_sha256(
            completion_payload
        )
        (temporary / "completion.json").write_text(
            json.dumps(completion_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.rename(output_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return {
        "output": str(output_root),
        "product_identity_sha256": completion_payload["product_identity_sha256"],
        "source_sha256": source_sha256,
        "config_sha256": config_sha256,
        "jobs": len(job_dirs),
        "branches": len(job_dirs) * int(config["expected_attempts_per_job"]),
        "feature_rows": len(features),
        "attempt0_rank_calibrated": False,
        "fresh_only_rank_minimum_p": 0.25,
        "external_judges_opened": False,
    }


def self_test() -> None:
    rng = np.random.default_rng(20260828)
    prefix = rng.normal(size=(4, 32, 32)).astype(np.float32)
    raw, effective = prefix_normalizers(prefix, (1, 2, 4), 1e-6)
    assert set(raw) == {1, 2, 4}
    assert all(value > 0 for value in effective.values())

    left = rng.normal(size=(4, 32, 32)).astype(np.float32)
    constants = np.asarray([1.5, -2.0, 0.25, 3.0], dtype=np.float32)[:, None, None]
    assert math.isclose(
        multiscale_distance(left, left + constants, (1, 2, 4), effective),
        0.0,
        rel_tol=0.0,
        abs_tol=1e-7,
    )
    right = left.copy()
    right[0, :8, :8] += 1.0
    assert multiscale_distance(left, right, (1, 2, 4), effective) > 0.0

    branches = np.stack(
        [left, right, left * 0.8, left * 1.1, left * 1.4], axis=0
    )
    matrix = distance_matrix(branches, (1, 2, 4), effective)
    assert matrix.shape == (5, 5)
    assert np.array_equal(matrix, matrix.T)
    assert np.all(np.diag(matrix) == 0.0)
    nonconformity = mean_nonconformity(matrix, (1, 2, 3, 4))
    rank = conservative_rank_p(nonconformity)
    assert set(rank.values()).issubset({0.25, 0.5, 0.75, 1.0})
    medoid, _ = stable_medoid(nonconformity)
    assert medoid in {1, 2, 3, 4}
    print(
        json.dumps(
            {
                "self_test": "passed",
                "distance_symmetric": True,
                "channel_dc_invariant": True,
                "fresh_rank_grid": [0.25, 0.5, 0.75, 1.0],
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    result = extract_product(
        input_root=args.input_root,
        output_root=args.output.resolve(),
        config_path=args.config,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
