#!/usr/bin/env python3
"""Offline 16-fixed-tile mixture evidence for completed ADM64 local traces.

This script is read-only with respect to a completed
``observe_adm64_local_path_evidence.py`` run.  It reconstructs the predictable
whitened cross-scale direction ``w=sigma*theta`` from each saved trace and
defines sixteen alternatives, one for every fixed cell of a 4x4 spatial grid.
The tile locations and their uniform prior are fixed before any path is read.

For every tile j and every one of the 230 effective non-identity transitions,
the conditional-KL allowance is fixed at

    K_allow = K_total / 230.

Unused allowance is discarded rather than carried to another transition.  A
2e-12 relative numerical guard is applied below that declared allowance so
floating-point reductions cannot exceed it; both values are recorded.  A
tile's scale depends only on the current saved history statistic
``sigma*theta``; the innovation is supplied only to a separate LR function
after the shift has been fixed.  Thus each component is an operational
same-covariance Gaussian likelihood-ratio process.

The only calibrated aggregate is the predeclared uniform mixture

    E_mix,k = (1/16) * sum_j E_j,k,

computed with log-sum-exp.  Component posteriors are reported for spatial
localization at the mixture maximum.  ``max_j E_j`` is never used as a
calibrated evidence process, threshold, crossing, or candidate/control score.
The frozen candidate flags remain exploratory single-reviewer labels and are
not formal relative-bad endpoints.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCHEMA_VERSION = 1
EXPERIMENT = "adm64_fixed_4x4_tile_uniform_mixture_evidence_offline"
EXPECTED_INPUT_EXPERIMENT = "adm64_local_cross_scale_path_evidence_observe_only"
EXPECTED_OBSERVER_SHA256 = (
    "5c3a25a680e8319b2ac2d217c1ec35209f3b56557c8b534dbe20ba5bfd64f9ec"
)
FROZEN_PRESELECTION_SPECS = {
    "61693d1f270ca4456e53b41ca7e7ea26856cc5f1cf1de37ab21b32ef1ddd6820": {
        "status": "exploratory_single_reviewer_preselection_not_a_formal_bad_label",
        "seeds": (104, 105, 116),
    },
    "8bd0a57daf054ec610718e5e0f43ae5a38a1e747ec21afa9df1bf9a6045dc13d": {
        "status": "user_identified_exploratory_candidate_before_path_evidence",
        "seeds": (115,),
    },
}
TOTAL_K_BUDGET_CHOICES = (0.5, 1.0)
GRID_SIZE = 4
IMAGE_SIZE = 64
CHANNELS = 3
TILE_SIZE = IMAGE_SIZE // GRID_SIZE
TILE_COUNT = GRID_SIZE * GRID_SIZE
STOCHASTIC_STEPS = 249
EFFECTIVE_STEPS = 230
INTERNAL_TIMESTEPS = np.arange(249, 0, -1, dtype=np.int16)
ALPHAS_FOR_VALID_MIXTURE_CROSSINGS = (0.20, 0.10, 0.05)
NUMERICAL_CAP_RELATIVE_GUARD = 2.0e-12
REQUIRED_TRACE_DTYPES = {
    "theta": np.dtype(np.float64),
    "p_standard_deviation": np.dtype(np.float32),
    "innovation": np.dtype(np.float32),
    "effective_nonidentity": np.dtype(np.uint8),
    "internal_timestep": np.dtype(np.int16),
    "current_original_timestep": np.dtype(np.int16),
    "shifted_original_timestep": np.dtype(np.int16),
}
CANDIDATE_COLOR = "#E17C05"
CONTROL_COLORS = ("#4C78A8", "#707070")


@dataclass(frozen=True)
class CandidateBlock:
    seed: int
    candidate_class_id: int
    candidate_reason: str
    control_class_ids: tuple[int, int]


@dataclass(frozen=True)
class FrozenPreselection:
    path: Path
    sha256: str
    payload: dict[str, Any]
    blocks: tuple[CandidateBlock, ...]

    @property
    def pairs(self) -> tuple[tuple[int, int], ...]:
        pairs = []
        for block in self.blocks:
            classes = (block.candidate_class_id, *block.control_class_ids)
            pairs.extend((class_id, block.seed) for class_id in classes)
        return tuple(sorted(pairs))

    @property
    def classes(self) -> tuple[int, ...]:
        return tuple(sorted({class_id for class_id, _ in self.pairs}))

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(sorted({seed for _, seed in self.pairs}))


@dataclass(frozen=True)
class RunProvenance:
    run_dir: Path
    manifest: dict[str, Any]
    completion: dict[str, Any]
    manifest_sha256: str
    completion_sha256: str
    pairs: tuple[tuple[int, int], ...]
    effective_mask: np.ndarray


@dataclass(frozen=True)
class PredictableTileShift:
    whitened_shift: np.ndarray
    raw_K: np.ndarray
    scale: np.ndarray
    applied_K: np.ndarray
    numerical_rescale_count: int


@dataclass(frozen=True)
class PathEvidence:
    class_id: int
    seed: int
    exploratory_candidate: bool
    candidate_reason: str | None
    innovation_raw_sha256: str
    tile_K: np.ndarray
    tile_R: np.ndarray
    tile_L: np.ndarray
    component_cumulative_log_e: np.ndarray
    log_mixture_e: np.ndarray
    posterior_tile_probability: np.ndarray
    summary: dict[str, Any]


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_payload_sha(payload: dict[str, Any], digest_key: str) -> str:
    copied = dict(payload)
    copied.pop(digest_key, None)
    return sha256_json(copied)


def raw_array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(array).tobytes(order="C")
    ).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"cannot read JSON {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected one JSON object: {path}")
    return payload


def atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_csv_dump(rows: Sequence[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def fixed_tile_bounds(tile_index: int) -> tuple[int, int, int, int]:
    if not 0 <= tile_index < TILE_COUNT:
        raise ValueError(f"tile index is outside 0..{TILE_COUNT - 1}: {tile_index}")
    tile_y, tile_x = divmod(tile_index, GRID_SIZE)
    y0, x0 = tile_y * TILE_SIZE, tile_x * TILE_SIZE
    return y0, x0, y0 + TILE_SIZE, x0 + TILE_SIZE


def fixed_tile_records() -> list[dict[str, Any]]:
    return [
        {
            "tile_index": tile_index,
            "tile_row": tile_index // GRID_SIZE,
            "tile_column": tile_index % GRID_SIZE,
            "bounds_yxyx": list(fixed_tile_bounds(tile_index)),
            "prior_weight": 1.0 / TILE_COUNT,
        }
        for tile_index in range(TILE_COUNT)
    ]


def stable_logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    values64 = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values64).all():
        raise ValueError("logsumexp input must be finite")
    maximum = np.max(values64, axis=axis, keepdims=True)
    result = maximum + np.log(
        np.exp(values64 - maximum).sum(axis=axis, keepdims=True, dtype=np.float64)
    )
    return np.squeeze(result, axis=axis)


def load_frozen_preselection(path: Path) -> FrozenPreselection:
    path = path.resolve()
    digest = sha256_file(path)
    specification = FROZEN_PRESELECTION_SPECS.get(digest)
    if specification is None:
        raise RuntimeError(
            "candidate preselection is not in the frozen SHA-256 allowlist: "
            f"{digest}"
        )
    payload = read_json(path)
    expected = {
        "schema_version": 1,
        "status": specification["status"],
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"frozen preselection identity changed: {mismatches}")
    constraints = payload.get("selection_constraints", {})
    if constraints.get("path_evidence_seen") is not False:
        raise RuntimeError("candidate preselection must precede path-evidence inspection")
    if constraints.get("formal_endpoint") is not False:
        raise RuntimeError("candidate flags must not be presented as formal endpoints")
    if int(constraints.get("reviewers", 0)) != 1:
        raise RuntimeError("frozen micro-audit must retain its single-reviewer status")

    blocks = []
    for raw in payload.get("matched_seed_blocks", []):
        controls = tuple(int(value) for value in raw.get("same_innovation_controls", []))
        if len(controls) != 2 or len(set(controls)) != 2:
            raise RuntimeError("every matched block must contain exactly two distinct controls")
        candidate = int(raw["candidate_class_id"])
        if candidate in controls:
            raise RuntimeError("candidate cannot also be a same-innovation control")
        blocks.append(
            CandidateBlock(
                seed=int(raw["seed"]),
                candidate_class_id=candidate,
                candidate_reason=str(raw["candidate_reason"]),
                control_class_ids=(controls[0], controls[1]),
            )
        )
    expected_seeds = tuple(int(seed) for seed in specification["seeds"])
    observed_seeds = tuple(sorted(block.seed for block in blocks))
    if observed_seeds != expected_seeds or len(set(observed_seeds)) != len(observed_seeds):
        raise RuntimeError(
            "frozen matched micro-audit seed blocks changed: "
            f"{observed_seeds} != {expected_seeds}"
        )
    classes = {
        class_id
        for block in blocks
        for class_id in (block.candidate_class_id, *block.control_class_ids)
    }
    for block in blocks:
        if set((block.candidate_class_id, *block.control_class_ids)) != classes:
            raise RuntimeError("every seed block must contain the same three class IDs")
    return FrozenPreselection(path, digest, payload, tuple(sorted(blocks, key=lambda b: b.seed)))


def _expected_pair_set_sha(pairs: Sequence[tuple[int, int]]) -> str:
    class_major = sorted(pairs, key=lambda pair: (pair[0], pair[1]))
    return sha256_json([[class_id, seed] for class_id, seed in class_major])


def load_run_provenance(
    run_dir: Path,
    preselection: FrozenPreselection,
    total_K_budget: float,
) -> RunProvenance:
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "manifest.json"
    completion_path = run_dir / "completion.json"
    if not manifest_path.is_file() or not completion_path.is_file():
        raise RuntimeError("input local-trace run lacks manifest.json or completion.json")
    manifest = read_json(manifest_path)
    completion = read_json(completion_path)
    identity = manifest.get("identity_sha256")
    if not isinstance(identity, str) or identity != canonical_payload_sha(
        manifest, "identity_sha256"
    ):
        raise RuntimeError("input run manifest self-hash is invalid")

    expected_manifest = {
        "schema_version": 1,
        "experiment": EXPECTED_INPUT_EXPERIMENT,
        "class_ids": list(preselection.classes),
        "seeds": list(preselection.seeds),
        "sample_count": len(preselection.pairs),
        "pair_set_sha256": _expected_pair_set_sha(preselection.pairs),
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected_manifest.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"input run/preselection manifest mismatch: {mismatches}")
    if manifest.get("runner", {}).get("sha256") != EXPECTED_OBSERVER_SHA256:
        raise RuntimeError("input run was not produced by the frozen all-step observer")
    local_Q = manifest.get("local_operational_Q", {})
    if float(local_Q.get("total_K_budget", math.nan)) != total_K_budget:
        raise RuntimeError("input run K budget differs from requested fixed-tile budget")
    if int(local_Q.get("grid_size_per_axis", -1)) != GRID_SIZE:
        raise RuntimeError("input trace grid size is not the frozen 4x4 geometry")
    mapping = local_Q.get("mapping", {})
    current = np.asarray(mapping.get("current_original_timestep"), dtype=np.int64)
    shifted = np.asarray(mapping.get("shifted_original_timestep"), dtype=np.int64)
    if current.shape != (STOCHASTIC_STEPS,) or shifted.shape != (STOCHASTIC_STEPS,):
        raise RuntimeError("input heat mapping does not cover 249 stochastic steps")
    effective_mask = shifted != current
    if int(effective_mask.sum()) != EFFECTIVE_STEPS:
        raise RuntimeError("input heat mapping does not have exactly 230 effective steps")
    if int(mapping.get("effective_nonidentity_step_count", -1)) != EFFECTIVE_STEPS:
        raise RuntimeError("manifest effective-step count disagrees with its mapping")
    if mapping.get("internal_timesteps_reverse_order") != INTERNAL_TIMESTEPS.tolist():
        raise RuntimeError("input stochastic timestep order changed")

    expected_completion = {
        "complete": True,
        "manifest_identity_sha256": identity,
        "pair_set_sha256": manifest["pair_set_sha256"],
        "total_expected": len(preselection.pairs),
        "total_complete": len(preselection.pairs),
        "effective_nonidentity_steps_per_path": EFFECTIVE_STEPS,
        "stochastic_steps_observed_per_path": STOCHASTIC_STEPS,
        "interventions": 0,
    }
    completion_mismatches = {
        key: (completion.get(key), value)
        for key, value in expected_completion.items()
        if completion.get(key) != value
    }
    if completion_mismatches:
        raise RuntimeError(f"input completion record is incompatible: {completion_mismatches}")

    expected_signal_paths = {
        (
            run_dir
            / "signals"
            / f"class_{class_id:04d}"
            / f"{seed:019d}.json"
        ).resolve()
        for class_id, seed in preselection.pairs
    }
    expected_trace_paths = {
        (
            run_dir
            / "traces"
            / f"class_{class_id:04d}"
            / f"{seed:019d}.npz"
        ).resolve()
        for class_id, seed in preselection.pairs
    }
    actual_signals = {
        path.resolve() for path in (run_dir / "signals").glob("class_*/*.json")
    }
    actual_traces = {
        path.resolve() for path in (run_dir / "traces").glob("class_*/*.npz")
    }
    if actual_signals != expected_signal_paths or actual_traces != expected_trace_paths:
        raise RuntimeError("input run signal/trace file set is not exactly the frozen nine pairs")
    return RunProvenance(
        run_dir=run_dir,
        manifest=manifest,
        completion=completion,
        manifest_sha256=sha256_file(manifest_path),
        completion_sha256=sha256_file(completion_path),
        pairs=tuple(sorted(preselection.pairs)),
        effective_mask=np.ascontiguousarray(effective_mask),
    )


def _validate_trace_record(
    provenance: RunProvenance,
    pair: tuple[int, int],
) -> tuple[dict[str, np.ndarray], str]:
    class_id, seed = pair
    signal_path = (
        provenance.run_dir
        / "signals"
        / f"class_{class_id:04d}"
        / f"{seed:019d}.json"
    )
    signal = read_json(signal_path)
    if signal.get("payload_sha256") != canonical_payload_sha(signal, "payload_sha256"):
        raise RuntimeError(f"input signal self-hash is invalid: {pair}")
    expected_signal = {
        "schema_version": 1,
        "experiment": EXPECTED_INPUT_EXPERIMENT,
        "class_id": class_id,
        "seed": seed,
        "manifest_identity_sha256": provenance.manifest["identity_sha256"],
        "runner_sha256": EXPECTED_OBSERVER_SHA256,
        "observer_changed_P": False,
        "operational_LR_only": True,
        "ideal_heat_marginal_ratio_claimed": False,
        "image_quality_claimed": False,
    }
    mismatches = {
        key: (signal.get(key), value)
        for key, value in expected_signal.items()
        if signal.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"input signal identity mismatch for {pair}: {mismatches}")
    trace_record = signal.get("trace", {})
    expected_relative = f"traces/class_{class_id:04d}/{seed:019d}.npz"
    if trace_record.get("relative_path") != expected_relative:
        raise RuntimeError(f"trace path identity changed for {pair}")
    trace_path = provenance.run_dir / expected_relative
    if not trace_path.is_file() or trace_path.stat().st_size != trace_record.get("bytes"):
        raise RuntimeError(f"trace file is missing or has wrong byte count: {pair}")
    if sha256_file(trace_path) != trace_record.get("sha256"):
        raise RuntimeError(f"trace file SHA-256 failed: {pair}")

    manifest_trace = provenance.manifest.get("trace", {})
    expected_keys = sorted(manifest_trace.get("keys", []))
    if trace_record.get("keys") != expected_keys:
        raise RuntimeError(f"signal/manifest trace key set mismatch: {pair}")
    record_arrays = trace_record.get("arrays", {})
    if sorted(record_arrays) != expected_keys:
        raise RuntimeError(f"signal trace array records are incomplete: {pair}")
    try:
        with np.load(trace_path, allow_pickle=False) as archive:
            if sorted(archive.files) != expected_keys:
                raise RuntimeError(f"NPZ key set differs from signed record: {pair}")
            arrays = {
                key: np.ascontiguousarray(archive[key])
                for key in REQUIRED_TRACE_DTYPES
            }
    except Exception as exc:
        raise RuntimeError(f"cannot load required trace arrays: {pair}") from exc

    tensor_shape = (STOCHASTIC_STEPS, CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    expected_shapes = {
        "theta": tensor_shape,
        "p_standard_deviation": tensor_shape,
        "innovation": tensor_shape,
        "effective_nonidentity": (STOCHASTIC_STEPS,),
        "internal_timestep": (STOCHASTIC_STEPS,),
        "current_original_timestep": (STOCHASTIC_STEPS,),
        "shifted_original_timestep": (STOCHASTIC_STEPS,),
    }
    for key, expected_dtype in REQUIRED_TRACE_DTYPES.items():
        value = arrays[key]
        if value.shape != expected_shapes[key] or value.dtype != expected_dtype:
            raise RuntimeError(
                f"required trace schema mismatch for {pair}/{key}: "
                f"{value.shape}/{value.dtype}"
            )
        expected_record = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "raw_bytes_sha256": raw_array_sha256(value),
        }
        if record_arrays.get(key) != expected_record:
            raise RuntimeError(f"required trace raw-array hash failed: {pair}/{key}")
    for key in ("theta", "p_standard_deviation", "innovation"):
        if not np.isfinite(arrays[key]).all():
            raise RuntimeError(f"non-finite required trace array: {pair}/{key}")
    if np.any(arrays["p_standard_deviation"] <= 0):
        raise RuntimeError(f"non-positive P standard deviation: {pair}")
    if not np.array_equal(arrays["internal_timestep"], INTERNAL_TIMESTEPS):
        raise RuntimeError(f"stochastic timestep order mismatch: {pair}")
    current = np.asarray(
        provenance.manifest["local_operational_Q"]["mapping"][
            "current_original_timestep"
        ],
        dtype=np.int16,
    )
    shifted = np.asarray(
        provenance.manifest["local_operational_Q"]["mapping"][
            "shifted_original_timestep"
        ],
        dtype=np.int16,
    )
    if not np.array_equal(arrays["current_original_timestep"], current) or not np.array_equal(
        arrays["shifted_original_timestep"], shifted
    ):
        raise RuntimeError(f"trace heat mapping mismatch: {pair}")
    effective = arrays["effective_nonidentity"].astype(bool)
    if not np.array_equal(effective, provenance.effective_mask):
        raise RuntimeError(f"trace effective mask mismatch: {pair}")
    if np.count_nonzero(arrays["theta"][~effective]):
        raise RuntimeError(f"identity steps must have theta=0: {pair}")
    return arrays, str(record_arrays["innovation"]["raw_bytes_sha256"])


def predictable_fixed_tile_shift(
    theta: np.ndarray,
    p_standard_deviation: np.ndarray,
    effective_mask: np.ndarray,
    tile_index: int,
    K_allow: float,
) -> PredictableTileShift:
    """Construct one fixed-position shift without accepting an innovation."""

    if theta.shape != p_standard_deviation.shape:
        raise ValueError("theta and P standard deviation shapes differ")
    if theta.ndim != 4 or theta.shape[0] != effective_mask.size:
        raise ValueError("expected [steps,channels,height,width] predictable arrays")
    if not math.isfinite(K_allow) or K_allow <= 0:
        raise ValueError("K_allow must be finite and positive")
    y0, x0, y1, x1 = fixed_tile_bounds(tile_index)
    raw = (
        p_standard_deviation[:, :, y0:y1, x0:x1].astype(np.float64)
        * theta[:, :, y0:y1, x0:x1].astype(np.float64)
    )
    raw_K = 0.5 * np.square(raw).reshape(raw.shape[0], -1).sum(
        axis=1, dtype=np.float64
    )
    guarded_cap = K_allow * (1.0 - NUMERICAL_CAP_RELATIVE_GUARD)
    scale = np.ones(raw.shape[0], dtype=np.float64)
    capped = effective_mask & (raw_K > guarded_cap)
    scale[capped] = np.sqrt(guarded_cap / raw_K[capped])
    scale[~effective_mask] = 0.0
    whitened = raw * scale[:, None, None, None]
    applied_K = 0.5 * np.square(whitened).reshape(whitened.shape[0], -1).sum(
        axis=1, dtype=np.float64
    )

    # A deterministic second rescale handles a possible final-ulp overshoot.
    # It is still predictable: no innovation is an argument to this function.
    overshoot = applied_K > K_allow
    numerical_rescale_count = int(overshoot.sum())
    if overshoot.any():
        target = guarded_cap
        correction = np.sqrt(target / applied_K[overshoot])
        scale[overshoot] *= correction
        whitened[overshoot] *= correction[:, None, None, None]
        applied_K[overshoot] = 0.5 * np.square(whitened[overshoot]).reshape(
            numerical_rescale_count, -1
        ).sum(axis=1, dtype=np.float64)
    if np.any(applied_K > K_allow):
        raise AssertionError("fixed tile exceeded its per-step K allowance")
    if np.count_nonzero(whitened[~effective_mask]) or np.count_nonzero(
        applied_K[~effective_mask]
    ):
        raise AssertionError("identity transition must define Q=P")
    return PredictableTileShift(
        whitened_shift=np.ascontiguousarray(whitened),
        raw_K=np.ascontiguousarray(raw_K),
        scale=np.ascontiguousarray(scale),
        applied_K=np.ascontiguousarray(applied_K),
        numerical_rescale_count=numerical_rescale_count,
    )


def fixed_shift_log_lr_from_innovation(
    shift: PredictableTileShift,
    innovation: np.ndarray,
    tile_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the already-constructed predictable shift to the P innovation."""

    y0, x0, y1, x1 = fixed_tile_bounds(tile_index)
    selected_noise = innovation[:, :, y0:y1, x0:x1].astype(np.float64)
    if selected_noise.shape != shift.whitened_shift.shape:
        raise ValueError("innovation tile and predictable shift shapes differ")
    R = (shift.whitened_shift * selected_noise).reshape(
        selected_noise.shape[0], -1
    ).sum(axis=1, dtype=np.float64)
    K = shift.applied_K.copy()
    L = R - K
    if not np.isfinite(L).all():
        raise ValueError("fixed-tile log LR is non-finite")
    return np.ascontiguousarray(K), np.ascontiguousarray(R), np.ascontiguousarray(L)


def compute_fixed_tile_mixture(
    theta: np.ndarray,
    p_standard_deviation: np.ndarray,
    innovation: np.ndarray,
    effective_mask: np.ndarray,
    total_K_budget: float,
    *,
    declared_effective_steps: int = EFFECTIVE_STEPS,
) -> dict[str, Any]:
    if total_K_budget not in TOTAL_K_BUDGET_CHOICES:
        raise ValueError(f"total K budget must be one of {TOTAL_K_BUDGET_CHOICES}")
    if int(effective_mask.sum()) != declared_effective_steps:
        raise ValueError("effective mask count differs from the declared budget denominator")
    K_allow = float(total_K_budget) / declared_effective_steps
    steps = theta.shape[0]
    tile_K = np.zeros((steps, TILE_COUNT), dtype=np.float64)
    tile_R = np.zeros_like(tile_K)
    tile_L = np.zeros_like(tile_K)
    raw_K = np.zeros_like(tile_K)
    scale = np.zeros_like(tile_K)
    numerical_rescale_count = 0
    for tile_index in range(TILE_COUNT):
        # This function cannot inspect innovation by construction.
        shift = predictable_fixed_tile_shift(
            theta,
            p_standard_deviation,
            effective_mask,
            tile_index,
            K_allow,
        )
        K, R, L = fixed_shift_log_lr_from_innovation(
            shift, innovation, tile_index
        )
        tile_K[:, tile_index] = K
        tile_R[:, tile_index] = R
        tile_L[:, tile_index] = L
        raw_K[:, tile_index] = shift.raw_K
        scale[:, tile_index] = shift.scale
        numerical_rescale_count += shift.numerical_rescale_count

    component_cumulative = np.cumsum(tile_L, axis=0, dtype=np.float64)
    log_normalizer = stable_logsumexp(component_cumulative, axis=1)
    log_mixture = log_normalizer - math.log(TILE_COUNT)
    posterior = np.exp(component_cumulative - log_normalizer[:, None])
    if not np.allclose(
        posterior.sum(axis=1, dtype=np.float64), 1.0, rtol=3e-14, atol=3e-14
    ):
        raise AssertionError("tile posterior probabilities do not sum to one")
    per_tile_total_K = tile_K.sum(axis=0, dtype=np.float64)
    if np.any(per_tile_total_K > total_K_budget):
        raise AssertionError("a fixed tile component exceeded total K budget")
    if np.any(tile_K > K_allow):
        raise AssertionError("a fixed tile component exceeded per-step K allowance")
    if np.count_nonzero(tile_K[~effective_mask]) or np.count_nonzero(
        tile_L[~effective_mask]
    ):
        raise AssertionError("identity transitions contributed fixed-tile evidence")
    return {
        "K_allow": K_allow,
        "tile_K": tile_K,
        "tile_R": tile_R,
        "tile_L": tile_L,
        "raw_K": raw_K,
        "scale": scale,
        "component_cumulative_log_e": component_cumulative,
        "log_mixture_e": np.ascontiguousarray(log_mixture),
        "posterior_tile_probability": np.ascontiguousarray(posterior),
        "per_tile_total_K": np.ascontiguousarray(per_tile_total_K),
        "numerical_rescale_count": numerical_rescale_count,
    }


def _candidate_for_pair(
    preselection: FrozenPreselection,
    pair: tuple[int, int],
) -> tuple[bool, str | None]:
    class_id, seed = pair
    block = next(block for block in preselection.blocks if block.seed == seed)
    if class_id == block.candidate_class_id:
        return True, block.candidate_reason
    if class_id not in block.control_class_ids:
        raise RuntimeError(f"pair is neither frozen candidate nor control: {pair}")
    return False, None


def summarize_path(
    pair: tuple[int, int],
    computed: dict[str, Any],
    internal_timesteps: np.ndarray,
    exploratory_candidate: bool,
) -> dict[str, Any]:
    log_mixture = computed["log_mixture_e"]
    posterior = computed["posterior_tile_probability"]
    observed_max_index = int(np.argmax(log_mixture))
    observed_max = float(log_mixture[observed_max_index])
    if observed_max > 0.0:
        max_index: int | None = observed_max_index
        max_log_mixture = observed_max
        max_internal_t: int | None = int(internal_timesteps[max_index])
        posterior_at_max = posterior[max_index]
        top_value = float(posterior_at_max.max())
        top_indices = np.flatnonzero(posterior_at_max == top_value).astype(int).tolist()
    else:
        # E0=1 is part of the anytime-valid running maximum.  Localization is
        # deliberately reported as a 16-way tie rather than choosing a tile.
        max_index = None
        max_log_mixture = 0.0
        max_internal_t = None
        posterior_at_max = np.full(TILE_COUNT, 1.0 / TILE_COUNT, dtype=np.float64)
        top_value = 1.0 / TILE_COUNT
        top_indices = list(range(TILE_COUNT))
    unique_top = len(top_indices) == 1
    top_index = top_indices[0] if unique_top else None
    positive_posterior = posterior_at_max[posterior_at_max > 0]
    entropy = -float(
        (positive_posterior * np.log(positive_posterior)).sum(dtype=np.float64)
    )
    sorted_posterior = np.sort(posterior_at_max)
    posterior_margin = float(sorted_posterior[-1] - sorted_posterior[-2])
    per_tile_total_K = computed["per_tile_total_K"]
    summary: dict[str, Any] = {
        "class_id": pair[0],
        "seed": pair[1],
        "exploratory_candidate": exploratory_candidate,
        "formal_relative_bad": "not_evaluated",
        "total_K_budget_per_component": float(
            computed["K_allow"] * EFFECTIVE_STEPS
        ),
        "fixed_K_allowance_per_effective_step": float(computed["K_allow"]),
        "effective_step_count": EFFECTIVE_STEPS,
        "component_count": TILE_COUNT,
        "mixture_prior": "fixed_uniform_1_over_16",
        "final_log_uniform_mixture_e": float(log_mixture[-1]),
        "max_log_uniform_mixture_e_from_E0": max_log_mixture,
        "mixture_max_reverse_index": max_index,
        "mixture_max_internal_t": max_internal_t,
        "mixture_max_is_E0": max_index is None,
        "top_posterior_tile_unique": unique_top,
        "top_posterior_tile_index": top_index,
        "top_posterior_tile_indices_tied": top_indices,
        "top_posterior_tile_row": None if top_index is None else top_index // GRID_SIZE,
        "top_posterior_tile_column": None if top_index is None else top_index % GRID_SIZE,
        "top_posterior_tile_bounds_yxyx": (
            None if top_index is None else list(fixed_tile_bounds(top_index))
        ),
        "top_posterior_tile_mass": top_value,
        "top_minus_second_posterior_mass": posterior_margin,
        "posterior_entropy_nats_at_mixture_max": entropy,
        "posterior_entropy_fraction_of_uniform": entropy / math.log(TILE_COUNT),
        "min_component_total_applied_K": float(per_tile_total_K.min()),
        "max_component_total_applied_K": float(per_tile_total_K.max()),
        "mean_component_total_applied_K": float(per_tile_total_K.mean()),
        "numerical_rescale_count": int(computed["numerical_rescale_count"]),
        "componentwise_max_used_as_calibrated_evidence": False,
    }
    for alpha in ALPHAS_FOR_VALID_MIXTURE_CROSSINGS:
        label = str(alpha).replace(".", "p")
        summary[f"uniform_mixture_crossed_alpha_{label}"] = bool(
            max_log_mixture >= -math.log(alpha)
        )
    return summary


def analyze_paths(
    provenance: RunProvenance,
    preselection: FrozenPreselection,
    total_K_budget: float,
) -> dict[tuple[int, int], PathEvidence]:
    results = {}
    for pair in provenance.pairs:
        arrays, innovation_hash = _validate_trace_record(provenance, pair)
        computed = compute_fixed_tile_mixture(
            arrays["theta"],
            arrays["p_standard_deviation"],
            arrays["innovation"],
            arrays["effective_nonidentity"].astype(bool),
            total_K_budget,
        )
        is_candidate, reason = _candidate_for_pair(preselection, pair)
        summary = summarize_path(
            pair,
            computed,
            arrays["internal_timestep"],
            is_candidate,
        )
        results[pair] = PathEvidence(
            class_id=pair[0],
            seed=pair[1],
            exploratory_candidate=is_candidate,
            candidate_reason=reason,
            innovation_raw_sha256=innovation_hash,
            tile_K=computed["tile_K"],
            tile_R=computed["tile_R"],
            tile_L=computed["tile_L"],
            component_cumulative_log_e=computed["component_cumulative_log_e"],
            log_mixture_e=computed["log_mixture_e"],
            posterior_tile_probability=computed["posterior_tile_probability"],
            summary=summary,
        )

    for block in preselection.blocks:
        paths = [results[(class_id, block.seed)] for class_id in preselection.classes]
        innovation_hashes = {path.innovation_raw_sha256 for path in paths}
        if len(innovation_hashes) != 1:
            raise RuntimeError(
                f"matched classes do not share an identical innovation trace: seed {block.seed}"
            )
    return results


def _finite_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if np.std(left) == 0 or np.std(right) == 0:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if math.isfinite(value) else None


def matched_diagnostics(
    results: dict[tuple[int, int], PathEvidence],
    preselection: FrozenPreselection,
) -> list[dict[str, Any]]:
    diagnostics = []
    for block in preselection.blocks:
        candidate = results[(block.candidate_class_id, block.seed)]
        controls = [results[(class_id, block.seed)] for class_id in block.control_class_ids]
        candidate_max = float(
            candidate.summary["max_log_uniform_mixture_e_from_E0"]
        )
        control_max = [
            float(control.summary["max_log_uniform_mixture_e_from_E0"])
            for control in controls
        ]
        block_paths = [candidate, *controls]
        correlations = []
        for left_index in range(len(block_paths)):
            for right_index in range(left_index + 1, len(block_paths)):
                correlations.append(
                    _finite_correlation(
                        block_paths[left_index].log_mixture_e,
                        block_paths[right_index].log_mixture_e,
                    )
                )
        finite_correlations = [value for value in correlations if value is not None]
        diagnostics.append(
            {
                "seed": block.seed,
                "candidate_class_id": block.candidate_class_id,
                "candidate_reason": block.candidate_reason,
                "control_class_ids": list(block.control_class_ids),
                "candidate_max_log_uniform_mixture_e": candidate_max,
                "control_max_log_uniform_mixture_e": control_max,
                "candidate_minus_control_mean_max_log_uniform_mixture_e": (
                    candidate_max - float(np.mean(control_max))
                ),
                "candidate_rank_among_three_by_valid_mixture_max": 1
                + sum(value > candidate_max for value in control_max),
                "candidate_strictly_exceeds_both_controls": bool(
                    candidate_max > max(control_max)
                ),
                "candidate_top_posterior_tile_at_mixture_max": {
                    "index": candidate.summary["top_posterior_tile_index"],
                    "bounds_yxyx": candidate.summary[
                        "top_posterior_tile_bounds_yxyx"
                    ],
                    "mass": candidate.summary["top_posterior_tile_mass"],
                    "unique": candidate.summary["top_posterior_tile_unique"],
                },
                "within_seed_uniform_mixture_curve_correlations": correlations,
                "within_seed_min_finite_curve_correlation": (
                    min(finite_correlations) if finite_correlations else None
                ),
                "componentwise_max_used_as_calibrated_evidence": False,
            }
        )
    return diagnostics


def csv_rows(results: dict[tuple[int, int], PathEvidence]) -> list[dict[str, Any]]:
    rows = []
    for pair in sorted(results):
        path = results[pair]
        summary = path.summary
        rows.append(
            {
                "class_id": path.class_id,
                "seed": path.seed,
                "exploratory_candidate": path.exploratory_candidate,
                "formal_relative_bad": "not_evaluated",
                "total_K_budget_per_component": summary[
                    "total_K_budget_per_component"
                ],
                "fixed_K_allowance_per_effective_step": summary[
                    "fixed_K_allowance_per_effective_step"
                ],
                "final_log_uniform_mixture_e": summary[
                    "final_log_uniform_mixture_e"
                ],
                "max_log_uniform_mixture_e_from_E0": summary[
                    "max_log_uniform_mixture_e_from_E0"
                ],
                "mixture_max_reverse_index": summary["mixture_max_reverse_index"],
                "mixture_max_internal_t": summary["mixture_max_internal_t"],
                "mixture_max_is_E0": summary["mixture_max_is_E0"],
                "top_posterior_tile_unique": summary["top_posterior_tile_unique"],
                "top_posterior_tile_index": summary["top_posterior_tile_index"],
                "top_posterior_tile_row": summary["top_posterior_tile_row"],
                "top_posterior_tile_column": summary["top_posterior_tile_column"],
                "top_posterior_tile_bounds_yxyx": json.dumps(
                    summary["top_posterior_tile_bounds_yxyx"],
                    separators=(",", ":"),
                ),
                "top_posterior_tile_mass": summary["top_posterior_tile_mass"],
                "posterior_entropy_fraction_of_uniform": summary[
                    "posterior_entropy_fraction_of_uniform"
                ],
                "min_component_total_applied_K": summary[
                    "min_component_total_applied_K"
                ],
                "max_component_total_applied_K": summary[
                    "max_component_total_applied_K"
                ],
                "uniform_mixture_crossed_alpha_0p2": summary[
                    "uniform_mixture_crossed_alpha_0p2"
                ],
                "uniform_mixture_crossed_alpha_0p1": summary[
                    "uniform_mixture_crossed_alpha_0p1"
                ],
                "uniform_mixture_crossed_alpha_0p05": summary[
                    "uniform_mixture_crossed_alpha_0p05"
                ],
                "componentwise_max_used_as_calibrated_evidence": False,
            }
        )
    return rows


def render_uniform_mixture_curves(
    results: dict[tuple[int, int], PathEvidence],
    preselection: FrozenPreselection,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(
        len(preselection.blocks),
        1,
        figsize=(11, 3.5 * len(preselection.blocks)),
        sharex=True,
    )
    if len(preselection.blocks) == 1:
        axes = [axes]
    x = np.arange(-1, STOCHASTIC_STEPS, dtype=np.int64)
    for axis, block in zip(axes, preselection.blocks):
        ordered_classes = (block.candidate_class_id, *block.control_class_ids)
        control_index = 0
        for class_id in ordered_classes:
            path = results[(class_id, block.seed)]
            curve = np.concatenate(([0.0], path.log_mixture_e))
            if path.exploratory_candidate:
                color, style, width = CANDIDATE_COLOR, "-", 2.5
                label = f"class {class_id} exploratory candidate"
            else:
                color = CONTROL_COLORS[control_index]
                style = ("--", ":")[control_index]
                width = 1.7
                label = f"class {class_id} same-innovation control"
                control_index += 1
            axis.plot(x, curve, color=color, linestyle=style, linewidth=width, label=label)
            max_index = path.summary["mixture_max_reverse_index"]
            marker_x = -1 if max_index is None else int(max_index)
            marker_y = float(path.summary["max_log_uniform_mixture_e_from_E0"])
            axis.scatter([marker_x], [marker_y], color=color, s=24, zorder=4)
        axis.axhline(0.0, color="#303030", linewidth=0.8)
        axis.axhline(
            -math.log(0.20),
            color="#606060",
            linestyle="-.",
            linewidth=0.9,
            label="valid uniform-mixture threshold: alpha=0.20",
        )
        axis.axhline(
            -math.log(0.05),
            color="#303030",
            linestyle=(0, (5, 3)),
            linewidth=0.9,
            label="valid uniform-mixture threshold: alpha=0.05",
        )
        axis.set_title(f"Shared innovation seed {block.seed}", loc="left", fontsize=11)
        axis.set_ylabel("log uniform-mixture E")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        axis.legend(loc="upper left", ncol=2, fontsize=8, frameon=False)
    axes[-1].set_xlabel(
        "reverse transition index (-1 = E0, 0 = noisiest stochastic step, 248 = t=1)"
    )
    figure.suptitle(
        "ADM64 fixed 4x4-tile operational evidence — calibrated uniform mixture",
        fontsize=13,
    )
    figure.text(
        0.5,
        0.006,
        (
            "Only the fixed 1/16 mixture is calibrated. Component posterior tiles are "
            "descriptive; max_j E_j is not a valid unadjusted evidence threshold."
        ),
        ha="center",
        fontsize=9,
        color="#404040",
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.97))
    temporary = output_path.with_name(output_path.name + ".tmp")
    figure.savefig(temporary, format="png", dpi=190, facecolor="white")
    plt.close(figure)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, output_path)


def build_summary(
    provenance: RunProvenance,
    preselection: FrozenPreselection,
    results: dict[tuple[int, int], PathEvidence],
    diagnostics: list[dict[str, Any]],
    total_K_budget: float,
) -> dict[str, Any]:
    path_summaries = [results[pair].summary for pair in sorted(results)]
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "input": {
            "run_dir": str(provenance.run_dir),
            "run_manifest_identity_sha256": provenance.manifest["identity_sha256"],
            "run_manifest_file_sha256": provenance.manifest_sha256,
            "run_completion_file_sha256": provenance.completion_sha256,
            "observer_runner_sha256": EXPECTED_OBSERVER_SHA256,
            "preselection": str(preselection.path),
            "preselection_sha256": preselection.sha256,
        },
        "method": {
            "operational_LR_only": True,
            "ideal_heat_marginal_ratio_claimed": False,
            "fixed_tile_geometry": fixed_tile_records(),
            "tile_position_rule": (
                "all sixteen positions and their row-major identities are fixed before "
                "any path is read; no current or future innovation selects a tile"
            ),
            "component_count": TILE_COUNT,
            "fixed_uniform_prior": [1.0 / TILE_COUNT] * TILE_COUNT,
            "total_K_budget_per_component": total_K_budget,
            "effective_step_count": EFFECTIVE_STEPS,
            "fixed_K_allowance_per_effective_step": total_K_budget / EFFECTIVE_STEPS,
            "numerical_cap_relative_guard": NUMERICAL_CAP_RELATIVE_GUARD,
            "guarded_numerical_cap_per_effective_step": (
                total_K_budget
                / EFFECTIVE_STEPS
                * (1.0 - NUMERICAL_CAP_RELATIVE_GUARD)
            ),
            "unused_budget_policy": "discarded at each step; never carried",
            "calibrated_evidence": "E_mix,k=(1/16)*sum_j E_j,k via stable logsumexp",
            "componentwise_max_policy": (
                "max_j E_j is prohibited as an unadjusted calibrated evidence process, "
                "crossing statistic, or candidate/control score"
            ),
            "posterior_tile_use": (
                "descriptive spatial localization at the valid mixture maximum only"
            ),
        },
        "path_count": len(results),
        "candidate_count": sum(path.exploratory_candidate for path in results.values()),
        "formal_bad_endpoint_used": False,
        "path_summaries": path_summaries,
        "matched_candidate_control_diagnostics": diagnostics,
        "candidate_strictly_exceeds_both_controls_count": sum(
            bool(item["candidate_strictly_exceeds_both_controls"])
            for item in diagnostics
        ),
        "valid_uniform_mixture_crossing_counts": {
            str(alpha): sum(
                bool(
                    path.summary[
                        f"uniform_mixture_crossed_alpha_{str(alpha).replace('.', 'p')}"
                    ]
                )
                for path in results.values()
            )
            for alpha in ALPHAS_FOR_VALID_MIXTURE_CROSSINGS
        },
        "outputs": {
            "path_csv": "fixed_tile_uniform_mixture_path_summary.csv",
            "curve_plot": "fixed_tile_uniform_mixture_curves.png",
        },
        "interpretation_guard": (
            f"This frozen {len(results)}-path single-reviewer discovery diagnostic cannot estimate "
            "TPR, FPR, artifact prevalence, or image-quality improvement. Only the uniform "
            "mixture, not a componentwise maximum, has the stated calibration."
        ),
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    summary["payload_sha256"] = canonical_payload_sha(summary, "payload_sha256")
    return summary


def write_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    results: dict[tuple[int, int], PathEvidence],
    preselection: FrozenPreselection,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "fixed_tile_uniform_mixture_path_summary.csv"
    plot_path = output_dir / "fixed_tile_uniform_mixture_curves.png"
    summary_path = output_dir / "summary.json"
    atomic_csv_dump(rows, csv_path)
    render_uniform_mixture_curves(results, preselection, plot_path)
    atomic_json_dump(summary, summary_path)
    files = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in (summary_path, csv_path, plot_path)
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "summary_payload_sha256": summary["payload_sha256"],
        "runner_sha256": summary["runner"]["sha256"],
        "files": files,
    }
    manifest["identity_sha256"] = canonical_payload_sha(manifest, "identity_sha256")
    atomic_json_dump(manifest, output_dir / "analysis_manifest.json")
    return manifest


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    preselection_path = args.preselection.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir == run_dir or output_dir in run_dir.parents or run_dir in output_dir.parents:
        raise ValueError("analysis output directory must not overlap the read-only input run")
    preselection = load_frozen_preselection(preselection_path)
    provenance = load_run_provenance(
        run_dir, preselection, args.total_K_budget
    )
    results = analyze_paths(provenance, preselection, args.total_K_budget)
    diagnostics = matched_diagnostics(results, preselection)
    rows = csv_rows(results)
    summary = build_summary(
        provenance,
        preselection,
        results,
        diagnostics,
        args.total_K_budget,
    )
    output_manifest = write_outputs(
        output_dir, summary, rows, results, preselection
    )
    result = {
        "output_dir": str(output_dir),
        "analysis_manifest_identity_sha256": output_manifest["identity_sha256"],
        "summary": summary,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def run_self_test() -> None:
    generator = np.random.default_rng(20260826)
    steps, channels, height, width = 6, 2, 64, 64
    effective = np.asarray([False, True, True, False, True, False])
    theta = generator.normal(size=(steps, channels, height, width)).astype(np.float64)
    sigma = np.exp(
        0.1 * generator.normal(size=(steps, channels, height, width))
    ).astype(np.float32)
    innovation = generator.normal(size=(steps, channels, height, width)).astype(np.float32)
    theta[~effective] = 0.0
    result = compute_fixed_tile_mixture(
        theta,
        sigma,
        innovation,
        effective,
        0.5,
        declared_effective_steps=3,
    )
    allowance = 0.5 / 3
    if np.any(result["tile_K"] > allowance):
        raise AssertionError("self-test per-step K cap failed")
    if np.any(result["per_tile_total_K"] > 0.5):
        raise AssertionError("self-test total K budget failed")
    if np.count_nonzero(result["tile_L"][~effective]):
        raise AssertionError("self-test identity Q=P failed")
    direct_log_mixture = np.log(
        np.exp(result["component_cumulative_log_e"]).mean(axis=1)
    )
    if not np.allclose(result["log_mixture_e"], direct_log_mixture, atol=2e-14):
        raise AssertionError("self-test uniform mixture identity failed")
    if not np.allclose(
        result["posterior_tile_probability"].sum(axis=1), 1.0, atol=2e-14
    ):
        raise AssertionError("self-test posterior normalization failed")

    # Compare one same-covariance Gaussian increment to direct log densities.
    tile_index, step = 7, 2
    shift = predictable_fixed_tile_shift(
        theta, sigma, effective, tile_index, allowance
    )
    K, _, L = fixed_shift_log_lr_from_innovation(shift, innovation, tile_index)
    u = shift.whitened_shift[step].reshape(-1)
    eps = innovation[
        step,
        :,
        fixed_tile_bounds(tile_index)[0] : fixed_tile_bounds(tile_index)[2],
        fixed_tile_bounds(tile_index)[1] : fixed_tile_bounds(tile_index)[3],
    ].astype(np.float64).reshape(-1)
    x = eps
    direct = -0.5 * np.square(x - u).sum() + 0.5 * np.square(x).sum()
    if not math.isclose(float(L[step]), float(direct), rel_tol=2e-12, abs_tol=2e-12):
        raise AssertionError("self-test direct Normal log-density ratio failed")
    if not math.isclose(float(K[step]), 0.5 * float(np.dot(u, u)), rel_tol=2e-12):
        raise AssertionError("self-test conditional K identity failed")

    huge = np.asarray([[1000.0] + [900.0] * 15], dtype=np.float64)
    stable = stable_logsumexp(huge, axis=1)[0] - math.log(TILE_COUNT)
    if not math.isfinite(float(stable)) or not math.isclose(
        float(stable), 1000.0 - math.log(TILE_COUNT), rel_tol=1e-14
    ):
        raise AssertionError("self-test stable logsumexp failed")

    with tempfile.TemporaryDirectory(prefix="adm64-fixed-tile-mixture-self-test-") as temporary:
        root = Path(temporary)
        payload = {"finite": True, "componentwise_max_calibrated": False}
        payload["payload_sha256"] = canonical_payload_sha(payload, "payload_sha256")
        path = root / "payload.json"
        atomic_json_dump(payload, path)
        loaded = read_json(path)
        if loaded["payload_sha256"] != canonical_payload_sha(loaded, "payload_sha256"):
            raise AssertionError("self-test output self-hash failed")
    print(
        "self-test passed: 16 fixed pre-innovation tiles, non-carrying K budget, "
        "exact Gaussian LR, stable uniform mixture, posterior normalization, and hashes"
    )


def build_parser() -> argparse.ArgumentParser:
    data_root = Path(os.environ.get("EQVAE_DATA_ROOT", "/home/zhoushunyu/data/eqvae"))
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="completed all-step local trace run; default follows --total-K-budget",
    )
    parser.add_argument(
        "--preselection",
        type=Path,
        default=repo_root / "experiments" / "annotations" / "adm64_relative_bad_preselection_v1.json",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--total-K-budget",
        type=float,
        choices=TOTAL_K_BUDGET_CHOICES,
        default=0.5,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    parser.set_defaults(_data_root=data_root)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.self_test:
        run_self_test()
        return
    suffix = str(args.total_K_budget).replace(".", "p")
    if args.run_dir is None:
        args.run_dir = (
            args._data_root
            / "cross_scale_evidence"
            / "adm64_local_path_evidence"
            / f"matched_seed104_105_116_K{suffix}_grid4"
        )
    if args.output_dir is None:
        args.output_dir = (
            args._data_root
            / "cross_scale_evidence"
            / "adm64_fixed_tile_mixture_analysis"
            / args.run_dir.name
        )
    if args.dry_run:
        frozen_preselection = load_frozen_preselection(args.preselection)
        print(
            json.dumps(
                {
                    "run_dir": str(args.run_dir.resolve()),
                    "preselection": str(frozen_preselection.path),
                    "frozen_preselection_sha256": frozen_preselection.sha256,
                    "frozen_pairs": [list(pair) for pair in frozen_preselection.pairs],
                    "output_dir": str(args.output_dir.resolve()),
                    "total_K_budget_per_component": args.total_K_budget,
                    "fixed_K_allowance_per_effective_step": (
                        args.total_K_budget / EFFECTIVE_STEPS
                    ),
                    "numerical_cap_relative_guard": NUMERICAL_CAP_RELATIVE_GUARD,
                    "guarded_numerical_cap_per_effective_step": (
                        args.total_K_budget
                        / EFFECTIVE_STEPS
                        * (1.0 - NUMERICAL_CAP_RELATIVE_GUARD)
                    ),
                    "fixed_grid": "4x4 positions, 16x16 pixels each",
                    "component_count": TILE_COUNT,
                    "calibrated_aggregate": "fixed uniform 1/16 mixture via logsumexp",
                    "componentwise_max_calibrated": False,
                    "gpu_required": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    run_analysis(args)


if __name__ == "__main__":
    main()
