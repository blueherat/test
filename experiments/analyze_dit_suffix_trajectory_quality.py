#!/usr/bin/env python3
"""Offline, post-hoc trajectory diagnostics for the frozen DiT suffix screen.

This program intentionally does *not* train a detector, select an output, or
make a confirmatory quality claim.  It binds the four completed suffix bundles
to their recorded hashes, reconstructs every target transition, and compares a
small set of predeclared trajectory summaries with the v2 visual-review labels.
Those labels were assigned after images were seen, so every label-dependent
result produced here is discovery-only and must be validated on new paths.
For quantities fixed before the current transition innovation, branch summaries
exclude suffix ``step_index=0``: every fresh branch shares that restored entry
state/model call, so an entry maximum can otherwise create a meaningless tie
for rank one.  Innovation-dependent updates and e-processes retain that step.

The scalar e-process diagnostics are operational null checks, not image-quality
e-values.  At stochastic transition ``t`` their unit direction is constructed
only from the already observed change in the predicted clean latent,

    d_t = x0_hat(t) - x0_hat(t + 1),

either over the full latent or restricted to the highest-energy member of a
fixed 4x4 spatial tiling.  Thus ``Z_t = <u_t, epsilon_t>`` is conditionally
N(0, 1) under the implemented Gaussian sampler.  With fixed ``m = 0.5``, each
increment is ``m Z_t - m^2 / 2``.  A second diagnostic is the uniform mixture
over all change points fixed at suffix entry.  The consumed but zero-multiplied
proposal at t=0 is excluded from all e-process updates and branch aggregates.

Outputs are written through a same-filesystem staging directory and an existing
output directory is never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

sys.dont_write_bytecode = True

import numpy as np


SCHEMA_VERSION = 1
EXPECTED_ROLLBACKS = (60, 120, 180, 225)
EXPECTED_ATTEMPTS = tuple(range(5))
TARGET_BATCH_INDEX = 0
LATENT_SHAPE = (4, 32, 32)
FIRST_HALF_BATCH = 8
FULL_BATCH = 16
TILE_GRID = 4
TILE_SIDE = 8
BET_M = 0.5
RECONSTRUCTION_ATOL = 2e-6

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUFFIX_ROOT = Path(
    "/home/zhoushunyu/data/eqvae/cross_scale_evidence/"
    "dit_imagenet256_suffix_repairability"
)
DEFAULT_ANNOTATION = (
    REPO_ROOT
    / "experiments/annotations/dit_imagenet256_seed2_suffix_quality_review_v2.json"
)
DEFAULT_OUTPUT_PARENT = Path(
    "/home/zhoushunyu/data/eqvae/cross_scale_evidence/"
    "dit_imagenet256_suffix_trajectory_quality"
)

TRACE_DTYPES: dict[str, np.dtype[Any]] = {
    "final_first_half": np.dtype(np.float32),
    "fresh_full_proposal": np.dtype(np.float32),
    "target_p_mean": np.dtype(np.float32),
    "target_p_standard_deviation": np.dtype(np.float32),
    "target_pred_xstart": np.dtype(np.float32),
    "target_state_before": np.dtype(np.float32),
    "transition_internal_timestep": np.dtype(np.int16),
}

STEP_FIELDS = (
    "branch_key",
    "rollback_internal_timestep",
    "attempt",
    "role",
    "absolute_quality",
    "binary_discovery_label",
    "prefix_preservation",
    "step_index",
    "internal_timestep",
    "original_timestep",
    "alpha_bar",
    "log_snr",
    "stochastic_effect",
    "used_innovation_sha256",
    "x0_jump_rms",
    "x0_logsnr_velocity_rms",
    "x0_logsnr_velocity_change_rms",
    "x0_logsnr_velocity_cosine_previous",
    "x0_logsnr_reversal",
    "jump_hot_tile_index",
    "jump_tile_max_energy_share",
    "pred_logvar_change_rms",
    "pred_logvar_change_centered_rms",
    "state_drift_rms",
    "state_update_rms",
    "stochastic_update_rms",
    "innovation_energy",
    "projection_full_z",
    "projection_hot_tile_z",
    "log_e_full",
    "log_e_hot_tile",
    "log_e_full_uniform_changepoint",
    "log_e_hot_tile_uniform_changepoint",
)

PRE_INNOVATION_STATE_METRICS = (
    "x0_jump_rms",
    "x0_logsnr_velocity_change_rms",
    "x0_logsnr_reversal",
    "jump_tile_max_energy_share",
    "pred_logvar_change_centered_rms",
    "state_drift_rms",
)

INNOVATION_DEPENDENT_METRICS = (
    "state_update_rms",
)

DISCOVERY_FEATURES = (
    "max_post_divergence_x0_jump_rms",
    "max_post_divergence_x0_logsnr_velocity_change_rms",
    "max_post_divergence_x0_logsnr_reversal",
    "max_post_divergence_jump_tile_max_energy_share",
    "max_post_divergence_pred_logvar_change_centered_rms",
    "max_post_divergence_state_drift_rms",
    "max_state_update_rms",
    "max_abs_innovation_energy_deviation",
    "max_log_e_full_uniform_changepoint",
    "max_log_e_hot_tile_uniform_changepoint",
)

FEATURE_SHORT_NAMES = {
    "max_post_divergence_x0_jump_rms": "x0 jump (post-div.)",
    "max_post_divergence_x0_logsnr_velocity_change_rms": "velocity change (post-div.)",
    "max_post_divergence_x0_logsnr_reversal": "velocity reversal (post-div.)",
    "max_post_divergence_jump_tile_max_energy_share": "jump concentration (post-div.)",
    "max_post_divergence_pred_logvar_change_centered_rms": "log-var change (post-div.)",
    "max_post_divergence_state_drift_rms": "state drift (post-div.)",
    "max_state_update_rms": "state update",
    "max_abs_innovation_energy_deviation": "noise energy |E-1|",
    "max_log_e_full_uniform_changepoint": "full CP log-e",
    "max_log_e_hot_tile_uniform_changepoint": "hot-tile CP log-e",
}


@dataclass(frozen=True)
class Label:
    absolute_quality: str
    binary_discovery_label: str | None
    prefix_preservation: str
    reason: str


@dataclass
class Observer:
    root: Path
    trace_path: Path
    results_path: Path
    results: dict[str, Any]
    arrays: dict[str, np.ndarray]
    row_for_t: dict[int, int]
    validation: dict[str, Any]


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def raw_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def validate_self_hash(payload: dict[str, Any], key: str, path: Path) -> None:
    expected = payload.get(key)
    stripped = dict(payload)
    stripped.pop(key, None)
    if not isinstance(expected, str) or expected != json_sha256(stripped):
        raise RuntimeError(f"invalid {key}: {path}")


def write_json(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def finite_float(value: float | np.floating[Any]) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"non-finite metric: {result}")
    return result


def rms(array: np.ndarray) -> float:
    values = np.asarray(array, dtype=np.float64)
    return finite_float(np.sqrt(np.mean(np.square(values), dtype=np.float64)))


def logsumexp(values: np.ndarray) -> float:
    values64 = np.asarray(values, dtype=np.float64)
    maximum = float(np.max(values64))
    return finite_float(maximum + math.log(float(np.exp(values64 - maximum).sum())))


def fixed_tiles() -> list[tuple[int, int, int, int]]:
    return [
        (row * TILE_SIDE, (row + 1) * TILE_SIDE, column * TILE_SIDE, (column + 1) * TILE_SIDE)
        for row in range(TILE_GRID)
        for column in range(TILE_GRID)
    ]


TILES = fixed_tiles()


def unit_direction(array: np.ndarray) -> tuple[np.ndarray, bool]:
    values = np.asarray(array, dtype=np.float64)
    norm = float(np.linalg.norm(values.ravel()))
    if math.isfinite(norm) and norm > 0.0:
        return values / norm, False
    fallback = np.zeros_like(values, dtype=np.float64)
    fallback.flat[0] = 1.0
    return fallback, True


def hot_tile_direction(jump: np.ndarray) -> tuple[np.ndarray, int, float, bool]:
    jump64 = np.asarray(jump, dtype=np.float64)
    energies = np.asarray(
        [
            np.square(jump64[:, y0:y1, x0:x1]).sum(dtype=np.float64)
            for y0, y1, x0, x1 in TILES
        ],
        dtype=np.float64,
    )
    tile_index = int(np.argmax(energies))
    total = float(energies.sum())
    share = float(energies[tile_index] / total) if total > 0.0 else 1.0 / len(TILES)
    restricted = np.zeros_like(jump64)
    y0, y1, x0, x1 = TILES[tile_index]
    restricted[:, y0:y1, x0:x1] = jump64[:, y0:y1, x0:x1]
    direction, fallback = unit_direction(restricted)
    return direction, tile_index, finite_float(share), fallback


def expected_trace_shapes(rollback: int) -> dict[str, tuple[int, ...]]:
    steps = rollback + 1
    target = (steps,) + LATENT_SHAPE
    return {
        "transition_internal_timestep": (steps,),
        "target_state_before": target,
        "target_pred_xstart": target,
        "target_p_mean": target,
        "target_p_standard_deviation": target,
        "fresh_full_proposal": (steps, FULL_BATCH) + LATENT_SHAPE,
        "final_first_half": (FIRST_HALF_BATCH,) + LATENT_SHAPE,
    }


def load_npz_validated(
    path: Path,
    record: dict[str, Any],
    root: Path,
    expected_dtypes: dict[str, np.dtype[Any]],
    expected_shapes: dict[str, tuple[int, ...]] | None = None,
    required_keys: Iterable[str] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    relative = path.relative_to(root).as_posix()
    if record.get("relative_path") != relative:
        raise RuntimeError(f"trace relative path mismatch: {path}")
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"trace is missing or a symlink: {path}")
    observed_file = {
        "relative_path": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if observed_file["bytes"] != record.get("bytes") or observed_file["sha256"] != record.get("sha256"):
        raise RuntimeError(f"trace file identity mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        keys = list(archive.files)
        wanted = set(keys if required_keys is None else required_keys)
        if not wanted.issubset(keys):
            raise RuntimeError(f"trace missing required arrays: {path}: {sorted(wanted - set(keys))}")
        arrays = {key: np.ascontiguousarray(archive[key]) for key in wanted}
    if required_keys is None and set(arrays) != set(expected_dtypes):
        raise RuntimeError(f"branch trace key set changed: {path}")
    array_validation: dict[str, Any] = {}
    for key in sorted(arrays):
        array = arrays[key]
        expected_record = record.get("arrays", {}).get(key)
        actual_record = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "raw_bytes_sha256": raw_sha256(array),
        }
        if expected_record != actual_record:
            raise RuntimeError(f"trace array identity mismatch: {path}/{key}")
        if key in expected_dtypes and array.dtype != expected_dtypes[key]:
            raise RuntimeError(f"trace array dtype mismatch: {path}/{key}")
        if expected_shapes is not None and key in expected_shapes and array.shape != expected_shapes[key]:
            raise RuntimeError(f"trace array shape mismatch: {path}/{key}")
        if not np.isfinite(array).all():
            raise RuntimeError(f"trace contains non-finite values: {path}/{key}")
        array_validation[key] = actual_record
    return arrays, {**observed_file, "arrays": array_validation}


def resolve_observer_root(manifests: Sequence[dict[str, Any]], override: Path | None) -> Path:
    identities = {
        item.get("frozen_observe_bundle", {}).get("manifest_identity_sha256")
        for item in manifests
    }
    if len(identities) != 1 or None in identities:
        raise RuntimeError("suffix bundles do not share one observer identity")
    if override is not None:
        return override.resolve()
    roots = {
        item.get("frozen_observe_bundle", {}).get("root") for item in manifests
    }
    if len(roots) != 1 or None in roots:
        raise RuntimeError("suffix bundles do not share one observer root")
    candidate = Path(next(iter(roots))).resolve()
    if candidate.is_dir():
        return candidate
    local_candidate = (
        DEFAULT_SUFFIX_ROOT.parent
        / "dit_imagenet256_path_evidence"
        / candidate.name
    )
    if local_candidate.is_dir():
        return local_candidate.resolve()
    raise FileNotFoundError(f"cannot resolve observer bundle: {candidate}")


def load_observer(root: Path, expected_from_suffix: dict[str, Any]) -> Observer:
    results_path = root / "results.json"
    completion_path = root / "completion.json"
    manifest_path = root / "manifest.json"
    for path in (manifest_path, results_path, completion_path):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"observer metadata is missing or a symlink: {path}")
    manifest = load_json(manifest_path)
    results = load_json(results_path)
    completion = load_json(completion_path)
    validate_self_hash(manifest, "identity_sha256", manifest_path)
    validate_self_hash(results, "payload_sha256", results_path)
    if completion.get("complete") is not True:
        raise RuntimeError("observer bundle is not complete")
    expected_files = {
        "manifest_file_sha256": sha256_file(manifest_path),
        "results_file_sha256": sha256_file(results_path),
        "completion_file_sha256": sha256_file(completion_path),
        "manifest_identity_sha256": manifest["identity_sha256"],
        "results_payload_sha256": results["payload_sha256"],
    }
    for key, value in expected_files.items():
        if expected_from_suffix.get(key) != value:
            raise RuntimeError(f"suffix-to-observer provenance mismatch: {key}")
    trace_record = results.get("trace")
    if not isinstance(trace_record, dict):
        raise RuntimeError("observer results has no trace record")
    trace_path = root / str(trace_record.get("relative_path"))
    required = {
        "current_alpha_bar",
        "current_original_timestep",
        "final_latents_first_half",
        "innovation_first_half",
        "internal_timestep",
        "p_mean_first_half",
        "p_standard_deviation",
        "pred_xstart",
        "x_t",
    }
    observer_dtypes = {
        "current_alpha_bar": np.dtype(np.float64),
        "current_original_timestep": np.dtype(np.int16),
        "final_latents_first_half": np.dtype(np.float32),
        "innovation_first_half": np.dtype(np.float32),
        "internal_timestep": np.dtype(np.int16),
        "p_mean_first_half": np.dtype(np.float32),
        "p_standard_deviation": np.dtype(np.float32),
        "pred_xstart": np.dtype(np.float32),
        "x_t": np.dtype(np.float32),
    }
    arrays, trace_validation = load_npz_validated(
        trace_path,
        trace_record,
        root,
        observer_dtypes,
        required_keys=required,
    )
    if trace_validation["sha256"] != expected_from_suffix.get("trace_sha256"):
        raise RuntimeError("suffix manifest observer trace hash mismatch")
    shapes = {
        "current_alpha_bar": (250,),
        "current_original_timestep": (250,),
        "internal_timestep": (250,),
        "innovation_first_half": (250, FIRST_HALF_BATCH) + LATENT_SHAPE,
        "p_mean_first_half": (250, FIRST_HALF_BATCH) + LATENT_SHAPE,
        "p_standard_deviation": (250, FIRST_HALF_BATCH) + LATENT_SHAPE,
        "pred_xstart": (250, FIRST_HALF_BATCH) + LATENT_SHAPE,
        "x_t": (250, FIRST_HALF_BATCH) + LATENT_SHAPE,
        "final_latents_first_half": (FIRST_HALF_BATCH,) + LATENT_SHAPE,
    }
    for key, shape in shapes.items():
        if arrays[key].shape != shape:
            raise RuntimeError(f"observer shape changed: {key}: {arrays[key].shape}")
    expected_axis = np.arange(249, -1, -1, dtype=np.int16)
    if not np.array_equal(arrays["internal_timestep"], expected_axis):
        raise RuntimeError("observer internal timestep axis changed")
    alpha = arrays["current_alpha_bar"]
    if not np.logical_and(alpha > 0.0, alpha < 1.0).all():
        raise RuntimeError("observer alpha_bar is outside (0, 1)")
    row_for_t = {int(t): index for index, t in enumerate(expected_axis)}
    return Observer(
        root=root,
        trace_path=trace_path,
        results_path=results_path,
        results=results,
        arrays=arrays,
        row_for_t=row_for_t,
        validation={
            "root": str(root),
            "manifest": {"path": str(manifest_path), "sha256": expected_files["manifest_file_sha256"]},
            "results": {"path": str(results_path), "sha256": expected_files["results_file_sha256"]},
            "completion": {"path": str(completion_path), "sha256": expected_files["completion_file_sha256"]},
            "trace": trace_validation,
            "loaded_schedule_arrays": [
                "current_alpha_bar",
                "current_original_timestep",
                "internal_timestep",
            ],
            "loaded_transition_arrays": sorted(required - {"current_alpha_bar", "current_original_timestep", "internal_timestep"}),
            "validated": True,
        },
    )


def load_labels(path: Path) -> tuple[dict[str, Label], dict[str, Any], dict[str, Any]]:
    annotation = load_json(path)
    if annotation.get("schema_version") != 2:
        raise RuntimeError("expected v2 suffix quality annotation")
    if annotation.get("status") != "posthoc_discovery_labels_not_for_confirmatory_claims":
        raise RuntimeError("annotation is not explicitly scoped to post-hoc discovery")
    labels: dict[str, Label] = {}
    for review in annotation.get("branch_reviews", []):
        rollback = int(review["internal_timestep"])
        attempt = int(review["attempt"])
        key = f"t{rollback}_attempt{attempt:03d}"
        if key in labels:
            raise RuntimeError(f"duplicate annotation branch: {key}")
        binary = review.get("binary_discovery_label")
        if binary not in (None, "good", "bad"):
            raise RuntimeError(f"invalid binary discovery label: {key}")
        labels[key] = Label(
            absolute_quality=str(review["absolute_quality"]),
            binary_discovery_label=binary,
            prefix_preservation=str(review["prefix_preservation"]),
            reason=str(review["reason"]),
        )
    expected = {
        f"t{rollback}_attempt{attempt:03d}"
        for rollback in EXPECTED_ROLLBACKS
        for attempt in EXPECTED_ATTEMPTS
    }
    if set(labels) != expected:
        raise RuntimeError(f"annotation branch coverage mismatch: {sorted(expected ^ set(labels))}")
    validation = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "schema_version": 2,
        "status": annotation["status"],
        "branch_count": len(labels),
        "labels_are_posthoc_discovery_only": True,
        "confirmatory_or_error_rate_claims_allowed": False,
        "validated": True,
    }
    return labels, annotation, validation


def branch_root_for_t(suffix_root: Path, rollback: int) -> Path:
    matches = sorted(suffix_root.glob(f"*_t{rollback}_n4"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one suffix bundle for t={rollback}, found {len(matches)}")
    return matches[0].resolve()


def manifest_results_completion(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    paths = {name: root / f"{name}.json" for name in ("manifest", "results", "completion")}
    if root.is_symlink() or any(not path.is_file() or path.is_symlink() for path in paths.values()):
        raise RuntimeError(f"invalid suffix bundle metadata: {root}")
    manifest = load_json(paths["manifest"])
    results = load_json(paths["results"])
    completion = load_json(paths["completion"])
    validate_self_hash(manifest, "identity_sha256", paths["manifest"])
    validate_self_hash(results, "payload_sha256", paths["results"])
    expected_completion = {
        "complete": True,
        "manifest_file_sha256": sha256_file(paths["manifest"]),
        "results_file_sha256": sha256_file(paths["results"]),
        "manifest_identity_sha256": manifest["identity_sha256"],
        "results_payload_sha256": results["payload_sha256"],
    }
    for key, value in expected_completion.items():
        if completion.get(key) != value:
            raise RuntimeError(f"suffix completion mismatch: {root}/{key}")
    metadata_validation = {
        "root": str(root),
        "manifest": {"sha256": expected_completion["manifest_file_sha256"]},
        "results": {"sha256": expected_completion["results_file_sha256"]},
        "completion": {"sha256": sha256_file(paths["completion"])},
        "manifest_identity_sha256": manifest["identity_sha256"],
        "results_payload_sha256": results["payload_sha256"],
    }
    return manifest, results, completion, metadata_validation


def log_snr(alpha_bar: float) -> float:
    alpha = float(alpha_bar)
    return finite_float(math.log(alpha) - math.log1p(-alpha))


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).ravel()
    b = np.asarray(right, dtype=np.float64).ravel()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return 1.0
    return finite_float(np.dot(a, b) / denominator)


def used_innovation(
    arrays: dict[str, np.ndarray], observer: Observer, attempt: int, step: int, internal_t: int
) -> np.ndarray:
    if attempt == 0:
        row = observer.row_for_t[internal_t]
        return np.ascontiguousarray(
            observer.arrays["innovation_first_half"][row, TARGET_BATCH_INDEX]
        )
    return np.ascontiguousarray(arrays["fresh_full_proposal"][step, TARGET_BATCH_INDEX])


def transition_record_raw_sha(record: Any, context: str) -> str:
    if not isinstance(record, dict):
        raise RuntimeError(f"missing tensor record: {context}")
    digest = record.get("raw_bytes_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError(f"invalid tensor digest: {context}")
    return digest


def analyze_branch(
    rollback: int,
    attempt: int,
    record: dict[str, Any],
    arrays: dict[str, np.ndarray],
    observer: Observer,
    label: Label,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    key = f"t{rollback}_attempt{attempt:03d}"
    role = str(record["role"])
    transitions = record.get("transitions")
    if not isinstance(transitions, list) or len(transitions) != rollback + 1:
        raise RuntimeError(f"transition record count mismatch: {key}")

    stochastic_horizon = rollback
    component_full = np.zeros(stochastic_horizon, dtype=np.float64)
    component_hot = np.zeros(stochastic_horizon, dtype=np.float64)
    cumulative_full = 0.0
    cumulative_hot = 0.0
    stochastic_index = 0
    rows: list[dict[str, Any]] = []
    maximum_reconstruction_error = 0.0
    exact_reconstruction_steps = 0
    direction_fallbacks = {"full": 0, "hot_tile": 0}
    maximum_unit_norm_error = {"full": 0.0, "hot_tile": 0.0}

    for step, internal_t in enumerate(range(rollback, -1, -1)):
        observer_row = observer.row_for_t[internal_t]
        alpha = float(observer.arrays["current_alpha_bar"][observer_row])
        original_t = int(observer.arrays["current_original_timestep"][observer_row])
        current_log_snr = log_snr(alpha)
        x = arrays["target_state_before"][step]
        predicted = arrays["target_pred_xstart"][step]
        mean = arrays["target_p_mean"][step]
        sigma = arrays["target_p_standard_deviation"][step]
        noise = used_innovation(arrays, observer, attempt, step, internal_t)
        stochastic = internal_t > 0
        after = (
            arrays["target_state_before"][step + 1]
            if step + 1 < len(transitions)
            else arrays["final_first_half"][TARGET_BATCH_INDEX]
        )
        reconstructed = mean + np.float32(1.0 if stochastic else 0.0) * sigma * noise
        difference = np.abs(reconstructed.astype(np.float64) - after.astype(np.float64))
        max_error = float(np.max(difference, initial=0.0))
        maximum_reconstruction_error = max(maximum_reconstruction_error, max_error)
        if np.array_equal(reconstructed, after):
            exact_reconstruction_steps += 1
        if max_error > RECONSTRUCTION_ATOL:
            raise RuntimeError(f"transition reconstruction failed: {key}/t={internal_t}: {max_error}")
        transition = transitions[step]
        expected_transition_identity = {
            "step_index": step,
            "internal_timestep": internal_t,
            "original_timestep": original_t,
            "stochastic_effect": stochastic,
        }
        if any(transition.get(k) != v for k, v in expected_transition_identity.items()):
            raise RuntimeError(f"transition identity mismatch: {key}/t={internal_t}")
        observed_noise_sha = raw_sha256(noise)
        recorded_noise_sha = transition_record_raw_sha(
            transition.get("target_used_innovation"), f"{key}/t={internal_t}/innovation"
        )
        if observed_noise_sha != recorded_noise_sha:
            raise RuntimeError(f"used innovation mismatch: {key}/t={internal_t}")

        previous_t = internal_t + 1
        if step == 0:
            previous = observer.arrays["pred_xstart"][
                observer.row_for_t[previous_t], TARGET_BATCH_INDEX
            ]
            previous_sigma = observer.arrays["p_standard_deviation"][
                observer.row_for_t[previous_t], TARGET_BATCH_INDEX
            ]
        else:
            previous = arrays["target_pred_xstart"][step - 1]
            previous_sigma = arrays["target_p_standard_deviation"][step - 1]
        previous_log_snr = log_snr(
            float(observer.arrays["current_alpha_bar"][observer.row_for_t[previous_t]])
        )
        jump = predicted.astype(np.float64) - previous.astype(np.float64)
        log_snr_step = current_log_snr - previous_log_snr
        if not log_snr_step > 0.0:
            raise RuntimeError(f"non-positive sampling log-SNR step: {key}/t={internal_t}")
        velocity = jump / log_snr_step

        previous_previous_t = internal_t + 2
        if step == 0:
            previous_previous = observer.arrays["pred_xstart"][
                observer.row_for_t[previous_previous_t], TARGET_BATCH_INDEX
            ]
        elif step == 1:
            previous_previous = observer.arrays["pred_xstart"][
                observer.row_for_t[rollback + 1], TARGET_BATCH_INDEX
            ]
        else:
            previous_previous = arrays["target_pred_xstart"][step - 2]
        previous_previous_log_snr = log_snr(
            float(
                observer.arrays["current_alpha_bar"][
                    observer.row_for_t[previous_previous_t]
                ]
            )
        )
        previous_velocity = (
            previous.astype(np.float64) - previous_previous.astype(np.float64)
        ) / (previous_log_snr - previous_previous_log_snr)
        velocity_cosine = max(-1.0, min(1.0, cosine(velocity, previous_velocity)))
        reversal = max(0.0, -velocity_cosine)

        direction_full, full_fallback = unit_direction(jump)
        direction_hot, hot_index, hot_share, hot_fallback = hot_tile_direction(jump)
        direction_fallbacks["full"] += int(full_fallback)
        direction_fallbacks["hot_tile"] += int(hot_fallback)
        maximum_unit_norm_error["full"] = max(
            maximum_unit_norm_error["full"],
            abs(float(np.square(direction_full).sum(dtype=np.float64)) - 1.0),
        )
        maximum_unit_norm_error["hot_tile"] = max(
            maximum_unit_norm_error["hot_tile"],
            abs(float(np.square(direction_hot).sum(dtype=np.float64)) - 1.0),
        )

        log_variance = np.log(np.maximum(np.square(sigma.astype(np.float64)), 1e-300))
        previous_log_variance = np.log(
            np.maximum(np.square(previous_sigma.astype(np.float64)), 1e-300)
        )
        log_variance_change = log_variance - previous_log_variance
        centered_log_variance_change = log_variance_change - float(log_variance_change.mean())
        drift = mean.astype(np.float64) - x.astype(np.float64)
        update = after.astype(np.float64) - x.astype(np.float64)
        innovation_energy = float(np.square(noise.astype(np.float64)).mean())

        if stochastic:
            projection_full = finite_float(
                np.sum(direction_full * noise.astype(np.float64), dtype=np.float64)
            )
            projection_hot = finite_float(
                np.sum(direction_hot * noise.astype(np.float64), dtype=np.float64)
            )
            increment_full = BET_M * projection_full - 0.5 * BET_M * BET_M
            increment_hot = BET_M * projection_hot - 0.5 * BET_M * BET_M
            cumulative_full += increment_full
            cumulative_hot += increment_hot
            component_full[: stochastic_index + 1] += increment_full
            component_hot[: stochastic_index + 1] += increment_hot
            log_cp_full = logsumexp(component_full) - math.log(stochastic_horizon)
            log_cp_hot = logsumexp(component_hot) - math.log(stochastic_horizon)
            stochastic_index += 1
            stochastic_update = sigma.astype(np.float64) * noise.astype(np.float64)
            e_values: dict[str, float | None] = {
                "projection_full_z": projection_full,
                "projection_hot_tile_z": projection_hot,
                "log_e_full": finite_float(cumulative_full),
                "log_e_hot_tile": finite_float(cumulative_hot),
                "log_e_full_uniform_changepoint": finite_float(log_cp_full),
                "log_e_hot_tile_uniform_changepoint": finite_float(log_cp_hot),
                "stochastic_update_rms": rms(stochastic_update),
                "innovation_energy": finite_float(innovation_energy),
            }
        else:
            # The proposal is recorded and hash-validated but has no transition
            # effect.  Blank values prevent it from contaminating diagnostics.
            e_values = {
                "projection_full_z": None,
                "projection_hot_tile_z": None,
                "log_e_full": None,
                "log_e_hot_tile": None,
                "log_e_full_uniform_changepoint": None,
                "log_e_hot_tile_uniform_changepoint": None,
                "stochastic_update_rms": None,
                "innovation_energy": None,
            }

        row: dict[str, Any] = {
            "branch_key": key,
            "rollback_internal_timestep": rollback,
            "attempt": attempt,
            "role": role,
            "absolute_quality": label.absolute_quality,
            "binary_discovery_label": label.binary_discovery_label,
            "prefix_preservation": label.prefix_preservation,
            "step_index": step,
            "internal_timestep": internal_t,
            "original_timestep": original_t,
            "alpha_bar": finite_float(alpha),
            "log_snr": current_log_snr,
            "stochastic_effect": int(stochastic),
            "used_innovation_sha256": observed_noise_sha,
            "x0_jump_rms": rms(jump),
            "x0_logsnr_velocity_rms": rms(velocity),
            "x0_logsnr_velocity_change_rms": rms(velocity - previous_velocity),
            "x0_logsnr_velocity_cosine_previous": velocity_cosine,
            "x0_logsnr_reversal": reversal,
            "jump_hot_tile_index": hot_index,
            "jump_tile_max_energy_share": hot_share,
            "pred_logvar_change_rms": rms(log_variance_change),
            "pred_logvar_change_centered_rms": rms(centered_log_variance_change),
            "state_drift_rms": rms(drift),
            "state_update_rms": rms(update),
            **e_values,
        }
        if set(row) != set(STEP_FIELDS):
            raise AssertionError(f"step metric schema mismatch: {set(row) ^ set(STEP_FIELDS)}")
        rows.append(row)

    if stochastic_index != stochastic_horizon:
        raise AssertionError(f"e-process horizon mismatch: {key}")
    stochastic_rows = [row for row in rows if row["stochastic_effect"] == 1]
    post_divergence_rows = [row for row in stochastic_rows if row["step_index"] > 0]
    if len(post_divergence_rows) != len(stochastic_rows) - 1:
        raise AssertionError(f"post-divergence row accounting mismatch: {key}")
    summary: dict[str, Any] = {
        "branch_key": key,
        "rollback_internal_timestep": rollback,
        "attempt": attempt,
        "role": role,
        "absolute_quality": label.absolute_quality,
        "binary_discovery_label": label.binary_discovery_label,
        "prefix_preservation": label.prefix_preservation,
        "stochastic_step_count": len(stochastic_rows),
        "post_divergence_step_count": len(post_divergence_rows),
    }
    # These quantities are fixed after the current model call but before the
    # current innovation.  At step_index=0 all fresh branches share exactly the
    # same restored state and model prediction, so that row has no branch-level
    # discrimination and is excluded from discovery aggregates.
    for metric in PRE_INNOVATION_STATE_METRICS:
        values = np.asarray(
            [row[metric] for row in post_divergence_rows], dtype=np.float64
        )
        maximum_index = int(np.argmax(values))
        summary[f"mean_post_divergence_{metric}"] = finite_float(values.mean())
        summary[f"max_post_divergence_{metric}"] = finite_float(values[maximum_index])
        summary[f"argmax_t_post_divergence_{metric}"] = int(
            post_divergence_rows[maximum_index]["internal_timestep"]
        )
    # Realized state updates already contain the branch-specific innovation at
    # step_index=0 and therefore retain the full stochastic suffix.
    for metric in INNOVATION_DEPENDENT_METRICS:
        values = np.asarray([row[metric] for row in stochastic_rows], dtype=np.float64)
        maximum_index = int(np.argmax(values))
        summary[f"mean_{metric}"] = finite_float(values.mean())
        summary[f"max_{metric}"] = finite_float(values[maximum_index])
        summary[f"argmax_t_{metric}"] = int(
            stochastic_rows[maximum_index]["internal_timestep"]
        )
    innovation_values = np.asarray(
        [row["innovation_energy"] for row in stochastic_rows], dtype=np.float64
    )
    summary["mean_innovation_energy"] = finite_float(innovation_values.mean())
    deviations = np.abs(innovation_values - 1.0)
    deviation_index = int(np.argmax(deviations))
    summary["max_abs_innovation_energy_deviation"] = finite_float(deviations[deviation_index])
    summary["argmax_t_abs_innovation_energy_deviation"] = int(
        stochastic_rows[deviation_index]["internal_timestep"]
    )
    for metric in (
        "log_e_full",
        "log_e_hot_tile",
        "log_e_full_uniform_changepoint",
        "log_e_hot_tile_uniform_changepoint",
    ):
        values = np.asarray([row[metric] for row in stochastic_rows], dtype=np.float64)
        maximum_index = int(np.argmax(values))
        summary[f"max_{metric}"] = finite_float(values[maximum_index])
        summary[f"argmax_t_{metric}"] = int(stochastic_rows[maximum_index]["internal_timestep"])
        summary[f"final_{metric}"] = finite_float(values[-1])
    validation = {
        "branch_key": key,
        "transition_count": len(rows),
        "stochastic_step_count": stochastic_index,
        "t0_proposal_hash_validated_but_excluded": True,
        "exact_float32_reconstruction_step_count": exact_reconstruction_steps,
        "maximum_reconstruction_absolute_error": finite_float(maximum_reconstruction_error),
        "reconstruction_tolerance": RECONSTRUCTION_ATOL,
        "direction_fallback_counts": direction_fallbacks,
        "maximum_unit_direction_squared_norm_error": {
            key_: finite_float(value) for key_, value in maximum_unit_norm_error.items()
        },
        "all_used_innovation_records_matched": True,
        "validated": True,
    }
    return rows, summary, validation


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            formatted = {
                key: "" if value is None else (format(value, ".17g") if isinstance(value, float) else value)
                for key, value in row.items()
            }
            writer.writerow(formatted)
        handle.flush()
        os.fsync(handle.fileno())


def branch_fields(rows: Sequence[dict[str, Any]]) -> list[str]:
    preferred = [
        "branch_key",
        "rollback_internal_timestep",
        "attempt",
        "role",
        "absolute_quality",
        "binary_discovery_label",
        "prefix_preservation",
        "stochastic_step_count",
    ]
    remaining = sorted(set(rows[0]) - set(preferred))
    fields = preferred + remaining
    if any(set(row) != set(fields) for row in rows):
        raise RuntimeError("branch metric schema is inconsistent")
    return fields


def validate_shared_suffix_entries(
    step_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    validations: list[dict[str, Any]] = []
    for rollback in EXPECTED_ROLLBACKS:
        entry_rows = [
            row
            for row in step_rows
            if row["rollback_internal_timestep"] == rollback
            and row["attempt"] > 0
            and row["step_index"] == 0
        ]
        if len(entry_rows) != 4:
            raise RuntimeError(f"shared-entry fresh-branch count mismatch: t={rollback}")
        metric_records: dict[str, Any] = {}
        for metric in PRE_INNOVATION_STATE_METRICS:
            values = [float(row[metric]) for row in entry_rows]
            if any(value != values[0] for value in values[1:]):
                raise RuntimeError(
                    f"pre-innovation state metric differs at shared entry: t={rollback}/{metric}"
                )
            metric_records[metric] = {
                "exactly_equal_across_four_fresh_branches": True,
                "shared_value": finite_float(values[0]),
            }
        validations.append(
            {
                "rollback_internal_timestep": rollback,
                "fresh_branch_count": len(entry_rows),
                "step_index": 0,
                "metrics": metric_records,
                "validated": True,
            }
        )
    return validations


RANK_TIE_RTOL = 1e-12
RANK_TIE_ATOL = 1e-15


def values_tied(left: float, right: float) -> bool:
    return math.isclose(
        float(left), float(right), rel_tol=RANK_TIE_RTOL, abs_tol=RANK_TIE_ATOL
    )


def rank_diagnostics(
    target: dict[str, Any], peers: Sequence[dict[str, Any]], feature: str
) -> dict[str, Any]:
    target_key = str(target["branch_key"])
    value = float(target[feature])
    if len(peers) < 2 or sum(str(peer["branch_key"]) == target_key for peer in peers) != 1:
        raise RuntimeError(f"invalid ranking cohort: {target_key}/{feature}")
    tie_count = sum(values_tied(float(peer[feature]), value) for peer in peers)
    strictly_greater = sum(
        float(peer[feature]) > value
        and not values_tied(float(peer[feature]), value)
        for peer in peers
    )
    descending_rank = 1 + strictly_greater
    others = [peer for peer in peers if str(peer["branch_key"]) != target_key]
    best_other_value = max(float(peer[feature]) for peer in others)
    best_other_keys = [
        str(peer["branch_key"])
        for peer in others
        if values_tied(float(peer[feature]), best_other_value)
    ]
    absolute_margin = value - best_other_value
    relative_scale = max(abs(value), abs(best_other_value), 1e-12)
    relative_margin = absolute_margin / relative_scale
    unique_largest = descending_rank == 1 and tie_count == 1
    midrank = descending_rank + 0.5 * (tie_count - 1)
    return {
        "descending_competition_rank": descending_rank,
        "cohort_size": len(peers),
        "tie_count_at_target_value": tie_count,
        "tie_tolerance": {"relative": RANK_TIE_RTOL, "absolute": RANK_TIE_ATOL},
        "unique_largest": unique_largest,
        "best_other_value": finite_float(best_other_value),
        "best_other_branch_keys": best_other_keys,
        "absolute_margin_over_best_other": finite_float(absolute_margin),
        "relative_margin_over_best_other": finite_float(relative_margin),
        "tie_adjusted_midrank": finite_float(midrank),
    }


def make_discovery_summary(
    branch_rows: Sequence[dict[str, Any]], annotation: dict[str, Any], annotation_validation: dict[str, Any]
) -> dict[str, Any]:
    by_key = {str(row["branch_key"]): row for row in branch_rows}
    matched_good = by_key["t60_attempt004"]
    matched_bad = by_key["t60_attempt003"]
    matched = {
        feature: {
            "posthoc_good_t60_attempt004": finite_float(matched_good[feature]),
            "posthoc_bad_t60_attempt003": finite_float(matched_bad[feature]),
            "bad_minus_good": finite_float(matched_bad[feature] - matched_good[feature]),
            "bad_within_t60_fresh_rank_diagnostics": rank_diagnostics(
                matched_bad,
                [
                    row
                    for row in branch_rows
                    if row["rollback_internal_timestep"] == 60 and row["attempt"] > 0
                ],
                feature,
            ),
        }
        for feature in DISCOVERY_FEATURES
    }
    broader_good = [row for row in branch_rows if row["binary_discovery_label"] == "good"]
    broader_bad = [row for row in branch_rows if row["binary_discovery_label"] == "bad"]
    broader = {
        feature: {
            "posthoc_good_count": len(broader_good),
            "posthoc_bad_count": len(broader_bad),
            "good_median": finite_float(np.median([row[feature] for row in broader_good])),
            "bad_median": finite_float(np.median([row[feature] for row in broader_bad])),
            "bad_minus_good_median": finite_float(
                np.median([row[feature] for row in broader_bad])
                - np.median([row[feature] for row in broader_good])
            ),
        }
        for feature in DISCOVERY_FEATURES
    }
    bad_ranks: list[dict[str, Any]] = []
    for row in broader_bad:
        rollback = int(row["rollback_internal_timestep"])
        peers = [
            candidate
            for candidate in branch_rows
            if candidate["rollback_internal_timestep"] == rollback
            and candidate["attempt"] > 0
        ]
        bad_ranks.append(
            {
                "branch_key": row["branch_key"],
                "within_checkpoint_fresh_branch_count": len(peers),
                "feature_rank_diagnostics": {
                    feature: rank_diagnostics(row, peers, feature)
                    for feature in DISCOVERY_FEATURES
                },
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "posthoc_metric_discovery_only",
        "annotation": annotation_validation,
        "warnings": [
            "Images were inspected before these quality labels and comparisons were made.",
            "The t=60 matched comparison contains one labeled-good and one labeled-bad endpoint.",
            "The broader pool has only three labeled-bad endpoints and is checkpoint/composition confounded.",
            "At each checkpoint, all fresh branches share the suffix-entry state and model call. A maximum attained on that shared step can create a tied rank of 1 with no discrimination; all pre-innovation state-metric discovery aggregates therefore exclude step_index=0.",
            "A descending competition rank of 1 is not interpreted as a candidate signal unless unique_largest is true; every rank report includes its tie count and margins over the best other branch.",
            "No hypothesis-test, error-rate, detection-power, or generalization quantity is computed.",
            "The predictable e-processes test Gaussian innovation alignment with frozen directions; they are not calibrated image-quality scores.",
            "Any candidate pattern must be frozen and evaluated on newly sampled prefixes and seeds.",
        ],
        "metric_definitions": {
            "x0_jump_rms": "RMS of x0_hat(t) - x0_hat(t+1); discovery aggregation uses step_index > 0.",
            "x0_logsnr_velocity_change_rms": "RMS change in the predicted-x0 finite-difference velocity per unit increase in log SNR; discovery aggregation uses step_index > 0.",
            "x0_logsnr_reversal": "max(0, minus cosine) between consecutive log-SNR velocities; discovery aggregation uses step_index > 0.",
            "jump_tile_max_energy_share": "Largest share of predicted-x0 jump squared energy in one fixed 8x8 tile of a fixed 4x4 tiling; discovery aggregation uses step_index > 0.",
            "pred_logvar_change_centered_rms": "Spatial/channel RMS after centering the stepwise change in log predicted transition variance; discovery aggregation uses step_index > 0.",
            "state_drift_rms": "RMS of p_mean(t) - x_t; discovery aggregation uses step_index > 0.",
            "state_update_rms": "RMS of realized x_(t-1) - x_t.",
            "innovation_energy": "Mean squared exact transition innovation; t=0 excluded.",
            "predictable_full_e_process": "Fixed m=0.5 exponential e-process using the normalized already-observed predicted-x0 jump as direction.",
            "predictable_hot_tile_e_process": "Same bet restricted to the previous jump's deterministic highest-energy fixed tile.",
            "uniform_changepoint_mixture": "Uniform prior fixed at suffix entry over every stochastic start index; future components remain one.",
            "rank_diagnostics": "Descending competition rank with numerical tie count, unique-largest flag, and absolute/relative margin over the best other fresh branch. Relative margin is scaled by max(abs(target), abs(best other), 1e-12).",
        },
        "predeclared_discovery_features": list(DISCOVERY_FEATURES),
        "primary_matched_late_suffix": {
            "annotation_cohort": annotation["discovery_cohorts"]["primary_matched_late_suffix"],
            "feature_comparison": matched,
        },
        "broader_clear_quality_descriptive_only": {
            "annotation_cohort": annotation["discovery_cohorts"]["broader_clear_quality"],
            "feature_medians": broader,
        },
        "bad_branch_within_checkpoint_ranks": bad_ranks,
    }


def make_plots(
    output: Path, step_rows: Sequence[dict[str, Any]], branch_rows: Sequence[dict[str, Any]]
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_paths: list[Path] = []
    matched_keys = ("t60_attempt003", "t60_attempt004")
    colors = {"t60_attempt003": "#c23b22", "t60_attempt004": "#2474b5"}
    labels = {
        "t60_attempt003": "post-hoc bad: attempt 003",
        "t60_attempt004": "post-hoc good: attempt 004",
    }
    panels = (
        ("x0_jump_rms", "predicted-x0 jump RMS"),
        ("x0_logsnr_velocity_change_rms", "logSNR velocity-change RMS"),
        ("x0_logsnr_reversal", "velocity reversal"),
        ("jump_tile_max_energy_share", "max 4x4-tile jump share"),
        ("pred_logvar_change_centered_rms", "centered log-var change RMS"),
        ("log_e_hot_tile_uniform_changepoint", "hot-tile uniform-CP log-e"),
    )
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    for axis, (metric, title) in zip(axes.ravel(), panels):
        for key in matched_keys:
            selected = [
                row
                for row in step_rows
                if row["branch_key"] == key and row["stochastic_effect"] == 1
            ]
            x = [int(row["step_index"]) for row in selected]
            y = [float(row[metric]) for row in selected]
            axis.plot(x, y, color=colors[key], linewidth=1.25, label=labels[key])
        axis.set_title(title)
        axis.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("suffix progress (0 = transition t60; t0 excluded)")
    axes[-1, 1].set_xlabel("suffix progress (0 = transition t60; t0 excluded)")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=2,
        frameon=False,
    )
    fig.suptitle("t60 matched pair — post-hoc discovery view (not validation)", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    matched_path = output / "t60_matched_pair_trajectories.png"
    fig.savefig(matched_path, dpi=180, metadata={"Software": "analyze_dit_suffix_trajectory_quality.py"})
    plt.close(fig)
    plot_paths.append(matched_path)

    bad_rows = [row for row in branch_rows if row["binary_discovery_label"] == "bad"]
    matrix = np.empty((len(bad_rows), len(DISCOVERY_FEATURES)), dtype=np.float64)
    annotations: list[list[str]] = []
    for row_index, row in enumerate(bad_rows):
        rollback = int(row["rollback_internal_timestep"])
        peers = [
            candidate
            for candidate in branch_rows
            if candidate["rollback_internal_timestep"] == rollback
            and candidate["attempt"] > 0
        ]
        annotation_row: list[str] = []
        for column, feature in enumerate(DISCOVERY_FEATURES):
            diagnostics = rank_diagnostics(row, peers, feature)
            rank = int(diagnostics["descending_competition_rank"])
            tie_count = int(diagnostics["tie_count_at_target_value"])
            midrank = float(diagnostics["tie_adjusted_midrank"])
            matrix[row_index, column] = (len(peers) - midrank) / max(
                1, len(peers) - 1
            )
            if diagnostics["unique_largest"]:
                status = "unique"
            elif tie_count > 1:
                status = f"tie×{tie_count}"
            else:
                status = ""
            annotation_row.append(
                f"{rank}/{len(peers)} {status}\n"
                f"Δ={diagnostics['absolute_margin_over_best_other']:+.1e}\n"
                f"rel={diagnostics['relative_margin_over_best_other']:+.1%}"
            )
        annotations.append(annotation_row)
    fig, axis = plt.subplots(figsize=(14, 4.3))
    image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="YlOrRd", aspect="auto")
    axis.set_xticks(range(len(DISCOVERY_FEATURES)))
    axis.set_xticklabels(
        [FEATURE_SHORT_NAMES[feature] for feature in DISCOVERY_FEATURES],
        rotation=35,
        ha="right",
    )
    axis.set_yticks(range(len(bad_rows)))
    axis.set_yticklabels([str(row["branch_key"]) for row in bad_rows])
    for row_index in range(len(bad_rows)):
        for column in range(len(DISCOVERY_FEATURES)):
            axis.text(
                column,
                row_index,
                annotations[row_index][column],
                ha="center",
                va="center",
                fontsize=6.5,
            )
    colorbar = fig.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    colorbar.set_label("tie-adjusted descending midrank score")
    axis.set_title(
        "Post-hoc bad-branch rank within checkpoint — state metrics exclude shared entry"
    )
    fig.tight_layout()
    rank_path = output / "bad_branch_within_checkpoint_rank_heatmap.png"
    fig.savefig(rank_path, dpi=180, metadata={"Software": "analyze_dit_suffix_trajectory_quality.py"})
    plt.close(fig)
    plot_paths.append(rank_path)
    return plot_paths


def self_test() -> dict[str, Any]:
    jump = np.zeros(LATENT_SHAPE, dtype=np.float64)
    jump[:, 8:16, 16:24] = 2.0
    hot, index, share, fallback = hot_tile_direction(jump)
    if index != 6 or share != 1.0 or fallback:
        raise AssertionError("fixed-tile self-test failed")
    if not math.isclose(float(np.square(hot).sum()), 1.0, rel_tol=0.0, abs_tol=1e-14):
        raise AssertionError("unit-direction self-test failed")
    increments = np.asarray([0.2, -0.1, 0.4], dtype=np.float64)
    components = np.zeros(3, dtype=np.float64)
    observed: list[float] = []
    for index_, increment in enumerate(increments):
        components[: index_ + 1] += increment
        observed.append(logsumexp(components) - math.log(3.0))
    explicit = [
        math.log((math.exp(0.2) + 1.0 + 1.0) / 3.0),
        math.log((math.exp(0.1) + math.exp(-0.1) + 1.0) / 3.0),
        math.log((math.exp(0.5) + math.exp(0.3) + math.exp(0.4)) / 3.0),
    ]
    if not np.allclose(observed, explicit, rtol=0.0, atol=1e-14):
        raise AssertionError("uniform change-point mixture self-test failed")
    ranking_peers = [
        {"branch_key": "a", "metric": 2.0},
        {"branch_key": "b", "metric": 2.0},
        {"branch_key": "c", "metric": 1.0},
    ]
    tied = rank_diagnostics(ranking_peers[0], ranking_peers, "metric")
    if (
        tied["descending_competition_rank"] != 1
        or tied["tie_count_at_target_value"] != 2
        or tied["unique_largest"]
        or tied["absolute_margin_over_best_other"] != 0.0
    ):
        raise AssertionError("ranking tie self-test failed")
    unique = rank_diagnostics(ranking_peers[2], ranking_peers, "metric")
    if unique["descending_competition_rank"] != 3 or unique["unique_largest"]:
        raise AssertionError("ranking order self-test failed")
    return {
        "fixed_tile_index_test": "passed",
        "unit_direction_test": "passed",
        "uniform_changepoint_mixture_test": "passed",
        "ranking_tie_and_margin_test": "passed",
        "passed": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suffix-root", type=Path, default=DEFAULT_SUFFIX_ROOT)
    parser.add_argument("--annotation", type=Path, default=DEFAULT_ANNOTATION)
    parser.add_argument(
        "--observer-dir",
        type=Path,
        default=None,
        help="Optional observer bundle override; otherwise use the path frozen in suffix manifests.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="New output directory. Default includes the first eight characters of this script's SHA256.",
    )
    parser.add_argument("--self-test-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tests = self_test()
    if args.self_test_only:
        print(json.dumps(tests, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    script_path = Path(__file__).resolve()
    script_sha = sha256_file(script_path)
    output = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (DEFAULT_OUTPUT_PARENT / f"seed2_quality_v2_{script_sha[:8]}").resolve()
    )
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        labels, annotation, annotation_validation = load_labels(args.annotation.resolve())
        roots = [branch_root_for_t(args.suffix_root.resolve(), rollback) for rollback in EXPECTED_ROLLBACKS]
        metadata = [manifest_results_completion(root) for root in roots]
        manifests = [item[0] for item in metadata]
        frozen_observer_records = [item.get("frozen_observe_bundle", {}) for item in manifests]
        first_observer_record = frozen_observer_records[0]
        if any(record != first_observer_record for record in frozen_observer_records[1:]):
            raise RuntimeError("four suffix manifests do not freeze identical observer provenance")
        observer_root = resolve_observer_root(manifests, args.observer_dir)
        observer = load_observer(observer_root, first_observer_record)

        all_steps: list[dict[str, Any]] = []
        all_branches: list[dict[str, Any]] = []
        bundle_validation: list[dict[str, Any]] = []
        for root, (manifest, results, completion, metadata_validation), rollback in zip(
            roots, metadata, EXPECTED_ROLLBACKS
        ):
            if int(results.get("rollback_internal_timestep", -1)) != rollback:
                raise RuntimeError(f"rollback mismatch: {root}")
            if manifest.get("target", {}).get("batch_index") != TARGET_BATCH_INDEX:
                raise RuntimeError(f"target batch index changed: {root}")
            branches = results.get("branches")
            if not isinstance(branches, list) or len(branches) != len(EXPECTED_ATTEMPTS):
                raise RuntimeError(f"branch count mismatch: {root}")
            one_bundle_validation = {
                **metadata_validation,
                "rollback_internal_timestep": rollback,
                "branch_traces": [],
                "branch_reconstruction": [],
            }
            for attempt, record in zip(EXPECTED_ATTEMPTS, branches):
                if record.get("attempt_index") != attempt or record.get("branch_id") != f"attempt_{attempt:03d}":
                    raise RuntimeError(f"branch order/identity mismatch: {root}/attempt={attempt}")
                trace_record = record.get("trace")
                if not isinstance(trace_record, dict):
                    raise RuntimeError(f"branch trace record missing: {root}/attempt={attempt}")
                trace_path = root / str(trace_record.get("relative_path"))
                arrays, trace_validation = load_npz_validated(
                    trace_path,
                    trace_record,
                    root,
                    TRACE_DTYPES,
                    expected_shapes=expected_trace_shapes(rollback),
                )
                expected_axis = np.arange(rollback, -1, -1, dtype=np.int16)
                if not np.array_equal(arrays["transition_internal_timestep"], expected_axis):
                    raise RuntimeError(f"branch timestep axis mismatch: {trace_path}")
                start_observer_row = observer.row_for_t[rollback]
                if not np.array_equal(
                    arrays["target_state_before"][0],
                    observer.arrays["x_t"][start_observer_row, TARGET_BATCH_INDEX],
                ):
                    raise RuntimeError(f"branch start does not reconstruct observer state: {trace_path}")
                if attempt == 0:
                    for step, internal_t in enumerate(range(rollback, -1, -1)):
                        row = observer.row_for_t[internal_t]
                        replay_arrays = {
                            "target_state_before": observer.arrays["x_t"][row, TARGET_BATCH_INDEX],
                            "target_pred_xstart": observer.arrays["pred_xstart"][row, TARGET_BATCH_INDEX],
                            "target_p_mean": observer.arrays["p_mean_first_half"][row, TARGET_BATCH_INDEX],
                            "target_p_standard_deviation": observer.arrays["p_standard_deviation"][row, TARGET_BATCH_INDEX],
                        }
                        if any(not np.array_equal(arrays[key][step], value) for key, value in replay_arrays.items()):
                            raise RuntimeError(f"attempt 0 differs from observer: t={rollback}/step={step}")
                    if not np.array_equal(
                        arrays["final_first_half"], observer.arrays["final_latents_first_half"]
                    ):
                        raise RuntimeError(f"attempt 0 final latents differ: t={rollback}")
                key = f"t{rollback}_attempt{attempt:03d}"
                steps, summary, reconstruction = analyze_branch(
                    rollback, attempt, record, arrays, observer, labels[key]
                )
                all_steps.extend(steps)
                all_branches.append(summary)
                one_bundle_validation["branch_traces"].append(trace_validation)
                one_bundle_validation["branch_reconstruction"].append(reconstruction)
            one_bundle_validation["validated"] = True
            bundle_validation.append(one_bundle_validation)

        if len(all_steps) != sum((rollback + 1) * 5 for rollback in EXPECTED_ROLLBACKS):
            raise AssertionError("unexpected total step row count")
        if len(all_branches) != 20:
            raise AssertionError("unexpected total branch row count")
        shared_entry_validation = validate_shared_suffix_entries(all_steps)

        write_csv(staging / "step_metrics.csv", all_steps, STEP_FIELDS)
        branch_field_order = branch_fields(all_branches)
        write_csv(staging / "branch_metrics.csv", all_branches, branch_field_order)
        discovery = make_discovery_summary(all_branches, annotation, annotation_validation)
        write_json(staging / "discovery_summary.json", discovery)
        input_validation = {
            "schema_version": SCHEMA_VERSION,
            "validated": True,
            "script": {"path": str(script_path), "sha256": script_sha},
            "analysis_scope": "posthoc_discovery_only",
            "t0_policy": "proposal hash-validated but excluded from e-processes and branch aggregates",
            "shared_suffix_entry_policy": {
                "pre_innovation_state_metrics": list(PRE_INNOVATION_STATE_METRICS),
                "aggregation": "step_index > 0 only",
                "reason": "step_index=0 state and model call are identical across fresh branches at a checkpoint",
                "innovation_dependent_metrics_retaining_step_index_0": list(
                    INNOVATION_DEPENDENT_METRICS
                ),
                "exact_shared_entry_validation": shared_entry_validation,
            },
            "rank_tie_policy": {
                "relative_tolerance": RANK_TIE_RTOL,
                "absolute_tolerance": RANK_TIE_ATOL,
                "rank_one_requires_unique_largest_for_candidate_interpretation": True,
                "margins_are_against_best_other_fresh_branch": True,
            },
            "e_process_contract": {
                "bet_m": BET_M,
                "direction_predictability": "full and hot-tile directions use only x0 predictions available before the current innovation",
                "projection_null": "conditional N(0,1) under the implemented Gaussian transition when the unit direction is nonzero; deterministic fixed-coordinate fallback preserves unit norm",
                "change_point_prior": "fixed uniform over all stochastic suffix transitions at suffix entry",
                "image_quality_interpretation": "none without independent validation",
            },
            "self_tests": tests,
            "annotation": annotation_validation,
            "observer": observer.validation,
            "suffix_bundles": bundle_validation,
            "row_counts": {"step_metrics": len(all_steps), "branch_metrics": len(all_branches)},
        }
        write_json(staging / "input_validation.json", input_validation)
        plot_paths = make_plots(staging, all_steps, all_branches)

        output_files = sorted(path for path in staging.iterdir() if path.is_file())
        manifest_payload = {
            "schema_version": SCHEMA_VERSION,
            "analysis_scope": "posthoc_discovery_only",
            "script_sha256": script_sha,
            "files": {
                path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in output_files
            },
            "plots": [path.name for path in plot_paths],
        }
        write_json(staging / "analysis_manifest.json", manifest_payload)
        for path in staging.iterdir():
            if path.is_file():
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
