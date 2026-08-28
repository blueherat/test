#!/usr/bin/env python3
"""Prospective class-207/seed-2 x60 *within-prefix* validation pool.

This runner creates one immutable shard of eight fresh baseline-P DiT suffixes.
Four disjoint shards form the predeclared 32-path pool.  Every branch starts at
the same already-saved class-207/seed-2 target latent ``x_60``.  Consequently,
this is conditional validation on one discovery prefix, not confirmation over
new prefixes, classes, prompts, or full samples.

The first half of every released DiT ``forward_with_cfg`` call contains the
eight independently evolving target branches.  The model reconstructs a
conditional/null 8 -> 16 batch internally, exactly as in the pinned upstream
implementation.  There is no cross-branch selection, rejection, rollback, or
guidance: the saved images are ordinary baseline-P suffix draws.

Before each branch's next innovation is generated, the runner computes the
frozen discovery-selected evidence candidate:

* additive normalized-heat shift Delta-nu = 0.25;
* row-major 4x4 latent ``tile_12``;
* positive ``+theta`` direction;
* total suffix conditional-KL cap K = 0.5 per path component; and
* alarm boundary log(5), equivalently alpha_e = 0.2.

A fixed secondary e-process averages 34 complete path likelihood ratios:
global plus all 16 fixed tiles, each with both +theta and -theta.  It is a
fixed *path mixture* (average after exponentiating cumulative log-LRs), not a
posthoc maximum over components.  Its terminal and running-maximum log-e
values are saved in the private trace.

All evidence is observation-only.  It never changes a transition.  The public
results JSON and stdout deliberately contain no evidence values, alarm flags,
or branch ranking.  Reviewers must label the PNGs and lock/hash annotations
before opening ``trace_private.npz``.  This is procedural blinding, not
cryptographic encryption.

Each branch owns a domain-separated ``torch.Generator`` seed and innovation
stream.  No suffix proposal uses global CPU/CUDA RNG.  Actual innovations,
stream seeds, and per-draw generator-state hashes are retained.  Bundles are
staged, completely reconstructed and validated, self-hashed, and atomically
installed without replacement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import socket
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

sys.dont_write_bytecode = True

import numpy as np
import torch
from PIL import Image

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

try:
    from .adm64_path_evidence import nearest_additive_heat_shift
    from .intervene_dit_imagenet256_suffix import (
        OBSERVER_TRACE_NAME,
        _atomic_install_directory_noreplace,
        _with_upstream_imports,
        load_observed_input,
    )
    from .observe_dit_imagenet256_path_evidence import (
        BATCH_SIZE,
        FULL_BATCH_SIZE,
        fixed_tile_bounds,
        fixed_tile_masks,
        load_schedule,
    )
    from .reproduce_dit_imagenet256 import (
        CFG_SCALE,
        CHECKPOINT_FILENAME,
        CLASS_IDS,
        IMAGE_SIZE,
        LATENT_CHANNELS,
        LATENT_SIZE,
        MODEL_NAME,
        NULL_CLASS_ID,
        NUM_CLASSES,
        NUM_SAMPLING_STEPS,
        VAE_REVISION,
        VAE_SCALING_FACTOR,
        atomic_json_dump,
        checkpoint_dry_probe,
        dependency_identity,
        ensure_single_process,
        inspect_png,
        load_json,
        sha256_file,
        sha256_json,
        validate_checkpoint,
        validate_repository,
        validate_vae_snapshot,
    )
except ImportError:  # pragma: no cover - direct CLI execution.
    from adm64_path_evidence import nearest_additive_heat_shift
    from intervene_dit_imagenet256_suffix import (
        OBSERVER_TRACE_NAME,
        _atomic_install_directory_noreplace,
        _with_upstream_imports,
        load_observed_input,
    )
    from observe_dit_imagenet256_path_evidence import (
        BATCH_SIZE,
        FULL_BATCH_SIZE,
        fixed_tile_bounds,
        fixed_tile_masks,
        load_schedule,
    )
    from reproduce_dit_imagenet256 import (
        CFG_SCALE,
        CHECKPOINT_FILENAME,
        CLASS_IDS,
        IMAGE_SIZE,
        LATENT_CHANNELS,
        LATENT_SIZE,
        MODEL_NAME,
        NULL_CLASS_ID,
        NUM_CLASSES,
        NUM_SAMPLING_STEPS,
        VAE_REVISION,
        VAE_SCALING_FACTOR,
        atomic_json_dump,
        checkpoint_dry_probe,
        dependency_identity,
        ensure_single_process,
        inspect_png,
        load_json,
        sha256_file,
        sha256_json,
        validate_checkpoint,
        validate_repository,
        validate_vae_snapshot,
    )


EXPERIMENT = "dit_imagenet256_t60_within_prefix_validation_pool"
SCHEMA_VERSION = 1
PREFIX_SEED = 2
TARGET_BATCH_INDEX = 0
TARGET_CLASS_ID = 207
ROLLBACK_INTERNAL_TIMESTEP = 60
TOTAL_SHARDS = 4
BRANCHES_PER_SHARD = BATCH_SIZE
TOTAL_POOL_BRANCHES = TOTAL_SHARDS * BRANCHES_PER_SHARD
POOL_SEED = 20_260_827
RNG_NAMESPACE = "eqvae-dit256-t60-within-prefix-validation-pool-v1"
BLIND_NAMESPACE = "eqvae-dit256-t60-blind-id-v1"

DELTA_NU = 0.25
TOTAL_K_PER_COMPONENT = 0.5
ALPHA_E = 0.2
ALARM_LOG_E = math.log(1.0 / ALPHA_E)
GRID_SIZE = 4
LOCAL_COMPONENT_COUNT = GRID_SIZE * GRID_SIZE
BASE_COMPONENT_COUNT = 1 + LOCAL_COMPONENT_COUNT
PRIMARY_TILE_INDEX = 12
PRIMARY_BASE_COMPONENT_INDEX = 1 + PRIMARY_TILE_INDEX
SIGN_VALUES = (1, -1)
SIGNED_COMPONENT_COUNT = len(SIGN_VALUES) * BASE_COMPONENT_COUNT
TRACE_NAME = "trace_private.npz"
PROTOCOL_COPY_NAME = "protocol.json"

RUNNER_DIR = Path(__file__).resolve().parent
DEFAULT_PROTOCOL_PATH = (
    RUNNER_DIR / "configs/dit_imagenet256_t60_within_prefix_validation_v1.json"
)

TRACE_DTYPES: dict[str, np.dtype[Any]] = {
    "branch_global_index": np.dtype(np.int16),
    "branch_stream_seed": np.dtype(np.int64),
    "generator_state_sha256_after": np.dtype("<U64"),
    "generator_state_sha256_before": np.dtype("<U64"),
    "internal_timestep": np.dtype(np.int16),
    "original_timestep": np.dtype(np.int16),
    "full_internal_alpha_bar": np.dtype(np.float64),
    "full_original_timestep_map": np.dtype(np.int64),
    "current_alpha_bar": np.dtype(np.float64),
    "shifted_internal_timestep": np.dtype(np.int16),
    "shifted_original_timestep": np.dtype(np.int16),
    "shifted_alpha_bar": np.dtype(np.float64),
    "rho": np.dtype(np.float64),
    "effective_nonidentity": np.dtype(np.uint8),
    "per_step_K_cap": np.dtype(np.float64),
    "tile_bounds_yxyx": np.dtype(np.int16),
    "base_component_name": np.dtype("<U16"),
    "signed_component_base_index": np.dtype(np.int16),
    "signed_component_sign": np.dtype(np.int8),
    "state_before": np.dtype(np.float32),
    "pred_xstart": np.dtype(np.float32),
    "p_mean": np.dtype(np.float32),
    "p_standard_deviation": np.dtype(np.float32),
    "transition_innovation": np.dtype(np.float32),
    "epsilon_current_reconstructed": np.dtype(np.float32),
    "epsilon_shifted": np.dtype(np.float32),
    "theta": np.dtype(np.float64),
    "secondary_raw_K": np.dtype(np.float64),
    "secondary_component_scale": np.dtype(np.float64),
    "secondary_K": np.dtype(np.float64),
    "secondary_R": np.dtype(np.float64),
    "secondary_L": np.dtype(np.float64),
    "secondary_component_log_e": np.dtype(np.float64),
    "secondary_path_mixture_log_e": np.dtype(np.float64),
    "secondary_terminal_path_mixture_log_e": np.dtype(np.float64),
    "secondary_running_max_path_mixture_log_e": np.dtype(np.float64),
    "primary_raw_K": np.dtype(np.float64),
    "primary_component_scale": np.dtype(np.float64),
    "primary_K": np.dtype(np.float64),
    "primary_R": np.dtype(np.float64),
    "primary_L": np.dtype(np.float64),
    "primary_log_e": np.dtype(np.float64),
    "primary_alarm_after_transition": np.dtype(np.uint8),
    "primary_ever_alarm": np.dtype(np.uint8),
    "primary_first_alarm_step_index": np.dtype(np.int16),
    "primary_terminal_log_e": np.dtype(np.float64),
    "primary_running_max_log_e": np.dtype(np.float64),
    "final_latents": np.dtype(np.float32),
}


def _canonical_self_hash(payload: dict[str, Any], key: str) -> str:
    stripped = dict(payload)
    stripped.pop(key, None)
    return sha256_json(stripped)


def _read_self_hashed_json(path: Path, key: str) -> dict[str, Any]:
    payload = load_json(path)
    observed = payload.get(key)
    if not isinstance(observed, str) or observed != _canonical_self_hash(payload, key):
        raise RuntimeError(f"invalid {key} in {path}")
    return payload


def _array_raw_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _copy_npz_array_preserve_shape(value: np.ndarray) -> np.ndarray:
    """Copy an NPZ member without turning a zero-dimensional scalar into shape (1,)."""

    return np.array(value, copy=True, order="C")


def _generator_state_sha256(generator: torch.Generator) -> str:
    state = generator.get_state().cpu().numpy()
    return hashlib.sha256(np.ascontiguousarray(state).tobytes(order="C")).hexdigest()


def _global_rng_state_sha256(device: torch.device) -> str:
    if device.type == "cuda":
        state = torch.cuda.get_rng_state(device)
    else:
        state = torch.get_rng_state()
    return hashlib.sha256(state.cpu().numpy().tobytes(order="C")).hexdigest()


def _logmeanexp(value: np.ndarray, axis: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    maximum = np.max(array, axis=axis, keepdims=True)
    answer = maximum + np.log(
        np.mean(np.exp(array - maximum), axis=axis, keepdims=True, dtype=np.float64)
    )
    return np.asarray(np.squeeze(answer, axis=axis), dtype=np.float64)


def branch_stream_seed(observer_identity_sha256: str, global_index: int) -> int:
    if len(observer_identity_sha256) != 64:
        raise ValueError("observer identity must be a SHA-256")
    if global_index not in range(TOTAL_POOL_BRANCHES):
        raise ValueError("global branch index is outside the frozen pool")
    payload = (
        f"{RNG_NAMESPACE}\0{observer_identity_sha256}\0{POOL_SEED}\0"
        f"{PREFIX_SEED}\0{TARGET_CLASS_ID}\0{ROLLBACK_INTERNAL_TIMESTEP}\0{global_index}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def blind_id(global_index: int) -> str:
    if global_index not in range(TOTAL_POOL_BRANCHES):
        raise ValueError("global branch index is outside the frozen pool")
    digest = hashlib.sha256(f"{BLIND_NAMESPACE}\0{global_index}".encode("ascii")).hexdigest()
    return f"vp1_{digest[:12]}"


def shard_global_indices(shard_index: int) -> tuple[int, ...]:
    if shard_index not in range(TOTAL_SHARDS):
        raise ValueError("shard index is outside the frozen pool")
    start = shard_index * BRANCHES_PER_SHARD
    return tuple(range(start, start + BRANCHES_PER_SHARD))


def _load_protocol(path: Path) -> dict[str, Any]:
    payload = _read_self_hashed_json(path, "protocol_identity_sha256")
    expected = {
        "schema_version": 1,
        "protocol_name": "dit_imagenet256_t60_within_prefix_validation_v1",
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise RuntimeError("protocol identity/schema mismatch")
    candidate = payload.get("frozen_primary_candidate", {})
    candidate_expected = {
        "delta_nu": DELTA_NU,
        "tile_index_row_major_4x4": PRIMARY_TILE_INDEX,
        "sign": "+theta",
        "total_suffix_K_cap": TOTAL_K_PER_COMPONENT,
        "alpha_e": ALPHA_E,
        "alarm_log_e": ALARM_LOG_E,
    }
    for key, expected_value in candidate_expected.items():
        observed = candidate.get(key)
        if isinstance(expected_value, float):
            if not isinstance(observed, (int, float)) or not math.isclose(
                float(observed), expected_value, rel_tol=0.0, abs_tol=1e-15
            ):
                raise RuntimeError(f"protocol primary candidate changed: {key}")
        elif observed != expected_value:
            raise RuntimeError(f"protocol primary candidate changed: {key}")
    pool = payload.get("pool", {})
    pool_expected = {
        "prefix_seed": PREFIX_SEED,
        "target_batch_index": TARGET_BATCH_INDEX,
        "target_class_id": TARGET_CLASS_ID,
        "rollback_internal_timestep": ROLLBACK_INTERNAL_TIMESTEP,
        "shard_count": TOTAL_SHARDS,
        "branches_per_shard": BRANCHES_PER_SHARD,
        "total_branches": TOTAL_POOL_BRANCHES,
        "pool_seed": POOL_SEED,
    }
    if any(pool.get(key) != value for key, value in pool_expected.items()):
        raise RuntimeError("protocol pool constants changed")
    if payload.get("scope", {}).get("general_confirmation") is not False:
        raise RuntimeError("protocol must explicitly reject a general-confirmation interpretation")
    required_binding_keys = {
        "observer_manifest_identity_sha256",
        "observer_manifest_file_sha256",
        "observer_results_file_sha256",
        "observer_completion_file_sha256",
        "observer_trace_sha256",
        "baseline_manifest_identity_sha256",
        "target_x60_raw_sha256",
        "alpha_bar_raw_sha256",
        "original_timestep_map_raw_sha256",
    }
    binding = payload.get("frozen_input_binding")
    if not isinstance(binding, dict) or set(binding) != required_binding_keys:
        raise RuntimeError("protocol does not bind the exact observer/prefix/schedule inputs")
    if any(not isinstance(binding[key], str) or len(binding[key]) != 64 for key in binding):
        raise RuntimeError("protocol frozen-input bindings must all be SHA-256 strings")
    return payload


def _validate_frozen_input_binding(
    protocol: dict[str, Any],
    observed: Any,
    alpha: np.ndarray,
    timestep_map: np.ndarray,
) -> None:
    """Require every shard to use the exact preregistered x60 and source trace."""

    rows = {
        int(value): index for index, value in enumerate(observed.arrays["internal_timestep"])
    }
    prefix = np.ascontiguousarray(
        observed.arrays["x_t"][rows[ROLLBACK_INTERNAL_TIMESTEP], TARGET_BATCH_INDEX],
        dtype=np.float32,
    )
    actual = {
        "observer_manifest_identity_sha256": observed.identity_sha256,
        "observer_manifest_file_sha256": sha256_file(observed.root / "manifest.json"),
        "observer_results_file_sha256": sha256_file(observed.root / "results.json"),
        "observer_completion_file_sha256": sha256_file(observed.root / "completion.json"),
        "observer_trace_sha256": str(observed.results["trace"]["sha256"]),
        "baseline_manifest_identity_sha256": observed.baseline.identity_sha256,
        "target_x60_raw_sha256": _array_raw_sha256(prefix),
        "alpha_bar_raw_sha256": _array_raw_sha256(
            np.ascontiguousarray(alpha, dtype=np.float64)
        ),
        "original_timestep_map_raw_sha256": _array_raw_sha256(
            np.ascontiguousarray(timestep_map, dtype=np.int64)
        ),
    }
    if protocol["frozen_input_binding"] != actual:
        differences = {
            key: {
                "protocol": protocol["frozen_input_binding"].get(key),
                "actual": value,
            }
            for key, value in actual.items()
            if protocol["frozen_input_binding"].get(key) != value
        }
        raise RuntimeError(f"frozen observer/prefix/schedule binding changed: {differences}")


def _schedule(alpha: np.ndarray) -> dict[str, np.ndarray | float | int]:
    internal = np.arange(ROLLBACK_INTERNAL_TIMESTEP, -1, -1, dtype=np.int64)
    shifted = np.zeros_like(internal)
    stochastic = internal > 0
    mapping = nearest_additive_heat_shift(alpha, internal[stochastic], DELTA_NU)
    shifted[stochastic] = mapping.shifted_timestep
    shifted[~stochastic] = 0
    effective = stochastic & (shifted != internal)
    effective_count = int(effective.sum())
    if effective_count <= 0:
        raise RuntimeError("frozen Delta-nu has no effective stochastic t60 suffix step")
    per_step_cap = TOTAL_K_PER_COMPONENT / effective_count * (1.0 - 2e-12)
    return {
        "internal_timestep": internal,
        "shifted_internal_timestep": shifted,
        "effective_nonidentity": effective.astype(np.uint8),
        "per_step_K_cap": float(per_step_cap),
        "effective_step_count": effective_count,
    }


def reconstruct_current_epsilon(
    state: np.ndarray, pred_xstart: np.ndarray, alpha_bar: float
) -> np.ndarray:
    if state.shape != pred_xstart.shape or state.ndim != 4:
        raise ValueError("state and pred_xstart must match [branch,C,H,W]")
    if not 0.0 < alpha_bar < 1.0:
        raise ValueError("alpha_bar must lie in (0,1)")
    epsilon = (
        state.astype(np.float64, copy=False)
        - math.sqrt(alpha_bar) * pred_xstart.astype(np.float64, copy=False)
    ) / math.sqrt(1.0 - alpha_bar)
    if not np.isfinite(epsilon).all():
        raise ValueError("reconstructed epsilon is non-finite")
    return np.ascontiguousarray(epsilon, dtype=np.float32)


def _base_masks() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bounds = fixed_tile_bounds(grid_size=GRID_SIZE, height=LATENT_SIZE, width=LATENT_SIZE)
    local = fixed_tile_masks(bounds, height=LATENT_SIZE, width=LATENT_SIZE)
    global_mask = np.ones((1, 1, LATENT_SIZE, LATENT_SIZE), dtype=np.float64)
    masks = np.ascontiguousarray(np.concatenate([global_mask, local], axis=0))
    names = np.asarray(
        ["global", *(f"tile_{index:02d}" for index in range(LOCAL_COMPONENT_COUNT))],
        dtype="<U16",
    )
    return masks, bounds, names


def construct_signed_components_before_innovation(
    theta: np.ndarray,
    p_sigma: np.ndarray,
    masks: np.ndarray,
    per_step_K_cap: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build all predictable +/- whitened shifts; accepts no innovation."""

    if theta.shape != p_sigma.shape or theta.ndim != 4:
        raise ValueError("theta and p_sigma must match [branch,C,H,W]")
    if theta.dtype != np.float64 or p_sigma.dtype != np.float32:
        raise TypeError("theta must be float64 and P sigma must be float32")
    if masks.shape != (BASE_COMPONENT_COUNT, 1, LATENT_SIZE, LATENT_SIZE):
        raise ValueError("global+tile mask geometry changed")
    if np.any(p_sigma <= 0.0) or not np.isfinite(theta).all():
        raise ValueError("invalid predictable Q construction input")
    if not math.isfinite(per_step_K_cap) or per_step_K_cap <= 0.0:
        raise ValueError("per-step K cap must be finite and positive")
    unsigned = np.ascontiguousarray(
        p_sigma.astype(np.float64, copy=False)[:, None] * theta[:, None] * masks[None]
    )
    signed = np.ascontiguousarray(
        np.concatenate([float(sign) * unsigned for sign in SIGN_VALUES], axis=1)
    )
    raw_K = 0.5 * np.sum(np.square(signed), axis=(2, 3, 4), dtype=np.float64)
    scale = np.ones_like(raw_K)
    positive = raw_K > 0.0
    scale[positive] = np.minimum(1.0, np.sqrt(per_step_K_cap / raw_K[positive]))
    whitened = np.ascontiguousarray(signed * scale[:, :, None, None, None])
    K = 0.5 * np.sum(np.square(whitened), axis=(2, 3, 4), dtype=np.float64)
    if np.any(K > per_step_K_cap * (1.0 + 2e-12)):
        raise AssertionError("a signed component exceeded its per-step K cap")
    return raw_K, scale, K, whitened


def evaluate_components_after_innovation(
    whitened_shift: np.ndarray, innovation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if whitened_shift.shape != (
        innovation.shape[0],
        SIGNED_COMPONENT_COUNT,
        innovation.shape[1],
        innovation.shape[2],
        innovation.shape[3],
    ):
        raise ValueError("innovation does not match the already-constructed shifts")
    if whitened_shift.dtype != np.float64 or innovation.dtype != np.float32:
        raise TypeError("LR evaluator requires float64 shifts and float32 innovation")
    reward = np.sum(
        whitened_shift * innovation.astype(np.float64, copy=False)[:, None],
        axis=(2, 3, 4),
        dtype=np.float64,
    )
    K = 0.5 * np.sum(np.square(whitened_shift), axis=(2, 3, 4), dtype=np.float64)
    return np.ascontiguousarray(reward), np.ascontiguousarray(reward - K)


def _draw_branch_innovations(
    generators: list[torch.Generator],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Draw only after Q construction, one explicit stream per branch."""

    if len(generators) != BRANCHES_PER_SHARD:
        raise ValueError("one generator per branch is required")
    global_before = _global_rng_state_sha256(device)
    before = np.empty((BRANCHES_PER_SHARD,), dtype="<U64")
    after = np.empty_like(before)
    draws = []
    for index, generator in enumerate(generators):
        before[index] = _generator_state_sha256(generator)
        draws.append(
            torch.randn(
                (LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE),
                dtype=dtype,
                device=device,
                generator=generator,
            )
        )
        after[index] = _generator_state_sha256(generator)
        if before[index] == after[index]:
            raise RuntimeError("branch generator state did not advance")
    if _global_rng_state_sha256(device) != global_before:
        raise RuntimeError("explicit branch generators modified global RNG")
    return torch.stack(draws, dim=0), before, after


def _trace_shapes() -> dict[str, tuple[int, ...]]:
    branches = BRANCHES_PER_SHARD
    steps = ROLLBACK_INTERNAL_TIMESTEP + 1
    state = (branches, steps, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE)
    component = (branches, steps, SIGNED_COMPONENT_COUNT)
    branch_step = (branches, steps)
    return {
        "branch_global_index": (branches,),
        "branch_stream_seed": (branches,),
        "generator_state_sha256_after": branch_step,
        "generator_state_sha256_before": branch_step,
        "internal_timestep": (steps,),
        "original_timestep": (steps,),
        "full_internal_alpha_bar": (NUM_SAMPLING_STEPS,),
        "full_original_timestep_map": (NUM_SAMPLING_STEPS,),
        "current_alpha_bar": (steps,),
        "shifted_internal_timestep": (steps,),
        "shifted_original_timestep": (steps,),
        "shifted_alpha_bar": (steps,),
        "rho": (steps,),
        "effective_nonidentity": (steps,),
        "per_step_K_cap": (),
        "tile_bounds_yxyx": (LOCAL_COMPONENT_COUNT, 4),
        "base_component_name": (BASE_COMPONENT_COUNT,),
        "signed_component_base_index": (SIGNED_COMPONENT_COUNT,),
        "signed_component_sign": (SIGNED_COMPONENT_COUNT,),
        "state_before": state,
        "pred_xstart": state,
        "p_mean": state,
        "p_standard_deviation": state,
        "transition_innovation": state,
        "epsilon_current_reconstructed": state,
        "epsilon_shifted": state,
        "theta": state,
        "secondary_raw_K": component,
        "secondary_component_scale": component,
        "secondary_K": component,
        "secondary_R": component,
        "secondary_L": component,
        "secondary_component_log_e": component,
        "secondary_path_mixture_log_e": branch_step,
        "secondary_terminal_path_mixture_log_e": (branches,),
        "secondary_running_max_path_mixture_log_e": (branches,),
        "primary_raw_K": branch_step,
        "primary_component_scale": branch_step,
        "primary_K": branch_step,
        "primary_R": branch_step,
        "primary_L": branch_step,
        "primary_log_e": branch_step,
        "primary_alarm_after_transition": branch_step,
        "primary_ever_alarm": (branches,),
        "primary_first_alarm_step_index": (branches,),
        "primary_terminal_log_e": (branches,),
        "primary_running_max_log_e": (branches,),
        "final_latents": (branches, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE),
    }


def _atomic_npz_dump(arrays: dict[str, np.ndarray], path: Path) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite trace: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _trace_record(path: Path, arrays: dict[str, np.ndarray], root: Path) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "keys": sorted(arrays),
        # Shapes and dtypes are public schema.  Do not publish unsalted
        # per-array hashes: alarm arrays have very low entropy and their hashes
        # could be brute-forced before blind endpoint labels are locked.  The
        # whole high-entropy NPZ remains content-addressed above, and the
        # validator reconstructs every array from its mathematical inputs.
        "schema": {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            for key, value in sorted(arrays.items())
        },
    }


def _png_record(path: Path, root: Path, size: tuple[int, int]) -> dict[str, Any]:
    record = {"relative_path": path.relative_to(root).as_posix()}
    record.update(inspect_png(path, "RGB", size))
    return record


def _grid_tile_pixels(grid_path: Path, local_index: int) -> np.ndarray:
    if local_index not in range(BRANCHES_PER_SHARD):
        raise ValueError("grid tile index is invalid")
    row, column = divmod(local_index, 4)
    left = 2 + column * (IMAGE_SIZE + 2)
    top = 2 + row * (IMAGE_SIZE + 2)
    with Image.open(grid_path) as image:
        image.load()
        if image.mode != "RGB" or image.size != (1_034, 518):
            raise RuntimeError("unexpected blind-grid geometry")
        return np.ascontiguousarray(
            np.asarray(image.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE)), dtype=np.uint8)
        )


def _paths_overlap(left: Path, right: Path) -> bool:
    left, right = left.resolve(), right.resolve()
    return left == right or left in right.parents or right in left.parents


def _source_dependencies() -> dict[str, dict[str, str]]:
    names = (
        "adm64_path_evidence.py",
        "intervene_dit_imagenet256_suffix.py",
        "observe_dit_imagenet256_path_evidence.py",
        "reproduce_dit_imagenet256.py",
    )
    return {
        name: {"path": str(RUNNER_DIR / name), "sha256": sha256_file(RUNNER_DIR / name)}
        for name in names
    }


def canonical_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--shard-index",
        str(args.shard_index),
        "--device-index",
        str(args.device_index),
        "--protocol",
        str(args.protocol),
        "--observe-dir",
        str(args.observe_dir),
        "--baseline-dir",
        str(args.baseline_dir),
        "--dit-root",
        str(args.dit_root),
        "--checkpoint",
        str(args.checkpoint),
        "--vae-snapshot",
        str(args.vae_snapshot),
        "--outdir",
        str(args.outdir),
    ]
    return command


def build_manifest(
    args: argparse.Namespace,
    *,
    protocol: dict[str, Any],
    observed: Any,
    source: dict[str, Any],
    checkpoint: dict[str, Any],
    vae: dict[str, Any],
    alpha: np.ndarray,
    timestep_map: np.ndarray,
) -> dict[str, Any]:
    indices = shard_global_indices(args.shard_index)
    all_seeds = [branch_stream_seed(observed.identity_sha256, index) for index in range(TOTAL_POOL_BRANCHES)]
    if len(set(all_seeds)) != TOTAL_POOL_BRANCHES:
        raise AssertionError("domain-separated validation stream seeds collided")
    schedule = _schedule(alpha)
    runner = Path(__file__).resolve()
    protocol_bytes_sha = sha256_file(args.protocol)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "role": "PROSPECTIVE_WITHIN_PREFIX_VALIDATION_SHARD",
        "scope": {
            "conditional_on_single_saved_prefix": True,
            "same_prefix_as_discovery": True,
            "new_suffix_innovation_streams": True,
            "general_confirmation": False,
            "cross_prefix_cross_seed_cross_class_claim_eligible": False,
            "reason": (
                "configuration was selected on four discovery suffixes from this exact x60; "
                "only the subsequent innovations are held out"
            ),
        },
        "sampling_distribution": {
            "baseline_P_suffix_unchanged": True,
            "evidence_changes_transition": False,
            "intervention_rejection_rollback_retry_or_guidance": False,
            "automatic_scoring_ranking_or_selection": False,
            "all_generated_branches_retained": True,
        },
        "pool": {
            "prefix_seed": PREFIX_SEED,
            "target_batch_index": TARGET_BATCH_INDEX,
            "target_class_id": TARGET_CLASS_ID,
            "rollback_internal_timestep": ROLLBACK_INTERNAL_TIMESTEP,
            "pool_seed": POOL_SEED,
            "total_shards": TOTAL_SHARDS,
            "branches_per_shard": BRANCHES_PER_SHARD,
            "total_pool_branches": TOTAL_POOL_BRANCHES,
            "this_shard_index": args.shard_index,
            "this_shard_global_branch_indices": list(indices),
            "all_four_shards_required": True,
        },
        "frozen_primary_candidate": {
            "selection_origin": (
                "posthoc t60 discovery comparison; frozen before these 32 innovation streams"
            ),
            "delta_nu": DELTA_NU,
            "tile": f"tile_{PRIMARY_TILE_INDEX:02d}",
            "tile_index_row_major_4x4": PRIMARY_TILE_INDEX,
            "sign": "+theta",
            "total_suffix_K_cap": TOTAL_K_PER_COMPONENT,
            "per_effective_step_K_cap": schedule["per_step_K_cap"],
            "alpha_e": ALPHA_E,
            "alarm_log_e": ALARM_LOG_E,
            "alarm_rule": "running primary log-e reaches or exceeds log(5)",
        },
        "frozen_secondary": {
            "delta_nu": DELTA_NU,
            "base_components": "global plus row-major 4x4 latent tiles 00..15",
            "signs": ["+theta", "-theta"],
            "component_count": SIGNED_COMPONENT_COUNT,
            "total_suffix_K_cap_per_component": TOTAL_K_PER_COMPONENT,
            "aggregation": "uniform fixed mixture of complete path likelihood ratios",
            "saved_summaries": ["terminal_log_e", "running_max_log_e_including_initial_zero"],
            "posthoc_component_max_used": False,
        },
        "predictability": {
            "Q_built_before_innovation_draw": True,
            "Q_inputs": [
                "current x_t",
                "current implemented-P pred_xstart",
                "current implemented-P sigma",
                "frozen shifted DiT epsilon",
            ],
            "innovation_function_not_accepted_by_Q_constructor": True,
            "log_lr_increment": "<u,z>-0.5*||u||^2",
            "same_covariance_operational_Q": True,
            "ideal_heat_marginal_ratio_claimed": False,
        },
        "cfg_contract": {
            "model": MODEL_NAME,
            "cfg_scale": CFG_SCALE,
            "first_half_branch_count": BATCH_SIZE,
            "model_batch_count": FULL_BATCH_SIZE,
            "all_first_half_labels": TARGET_CLASS_ID,
            "all_second_half_labels": NULL_CLASS_ID,
            "upstream_forward_with_cfg": True,
            "first_half_duplicated_inside_upstream_model": True,
            "guided_epsilon_channels": [0, 1, 2],
            "fourth_epsilon_channel_retained": True,
            "incoming_second_half_is_computational_carrier_not_sampled_path": True,
        },
        "rng": {
            "namespace": RNG_NAMESPACE,
            "one_explicit_generator_per_branch": True,
            "global_rng_used_for_suffix_innovations": False,
            "one_draw_per_branch_per_transition_including_t0": True,
            "draw_shape": [LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE],
            "branch_streams": [
                {
                    "global_index": index,
                    "blind_id": blind_id(index),
                    "seed": all_seeds[index],
                }
                for index in indices
            ],
            "conditional_independence_scope": (
                "pseudorandom streams are disjointly seeded and transitions have no cross-example "
                "operation in the pinned DiT; every path still shares the same fixed x60"
            ),
        },
        "blind_review": {
            "evidence_trace": TRACE_NAME,
            "procedurally_sealed_until_annotations_are_locked_and_hashed": True,
            "cryptographically_encrypted": False,
            "public_results_contain_evidence_values_alarm_flags_or_ranks": False,
            "sampling_stdout_contains_evidence_values_alarm_flags_or_ranks": False,
            "image_names_depend_on_evidence": False,
            "branch_output_selection": "none",
        },
        "input_prefix": {
            "observer_root": str(observed.root),
            "observer_manifest_identity_sha256": observed.identity_sha256,
            "observer_manifest_file_sha256": sha256_file(observed.root / "manifest.json"),
            "observer_results_file_sha256": sha256_file(observed.root / "results.json"),
            "observer_completion_file_sha256": sha256_file(observed.root / "completion.json"),
            "observer_trace_relative_path": OBSERVER_TRACE_NAME,
            "observer_trace_sha256": observed.results["trace"]["sha256"],
            "baseline_root": str(observed.baseline.root),
            "baseline_manifest_identity_sha256": observed.baseline.identity_sha256,
            "strictly_validated_before_use": True,
        },
        "schedule": {
            "num_internal_steps": NUM_SAMPLING_STEPS,
            "internal_timestep_axis": list(range(ROLLBACK_INTERNAL_TIMESTEP, -1, -1)),
            "effective_shifted_stochastic_steps": schedule["effective_step_count"],
            "alpha_bar_raw_sha256": _array_raw_sha256(np.ascontiguousarray(alpha, dtype=np.float64)),
            "original_timestep_map_raw_sha256": _array_raw_sha256(
                np.ascontiguousarray(timestep_map, dtype=np.int64)
            ),
        },
        "protocol": {
            "source_path": str(args.protocol),
            "copied_relative_path": PROTOCOL_COPY_NAME,
            "protocol_identity_sha256": protocol["protocol_identity_sha256"],
            "source_file_sha256": protocol_bytes_sha,
        },
        "sources": {
            "dit": source,
            "checkpoint": checkpoint,
            "vae": vae,
            "local_dependencies": _source_dependencies(),
        },
        "runner": {"path": str(runner), "sha256": sha256_file(runner)},
        "dependencies": dependency_identity(),
        "device_index": args.device_index,
        "canonical_command": canonical_command(args),
        "outputs": {
            "private_trace": TRACE_NAME,
            "blind_image_directory": "blind_images",
            "blind_grid": "blind_grid.png",
            "all_outputs_retained": True,
            "atomic_no_replace": True,
            "no_overwrite": True,
        },
    }
    payload["identity_sha256"] = _canonical_self_hash(payload, "identity_sha256")
    return payload


def run_sampling_shard(
    args: argparse.Namespace,
    *,
    observed: Any,
    alpha: np.ndarray,
    timestep_map: np.ndarray,
) -> tuple[dict[str, np.ndarray], torch.Tensor, dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real validation shard")
    if args.device_index >= torch.cuda.device_count():
        raise RuntimeError("--device-index is not visible")
    ensure_single_process()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["DIFFUSERS_OFFLINE"] = "1"
    schedule = _schedule(alpha)
    internal = np.asarray(schedule["internal_timestep"], dtype=np.int64)
    shifted = np.asarray(schedule["shifted_internal_timestep"], dtype=np.int64)
    effective = np.asarray(schedule["effective_nonidentity"], dtype=np.uint8).astype(bool)
    per_step_cap = float(schedule["per_step_K_cap"])
    steps = len(internal)
    masks, tile_bounds, component_names = _base_masks()
    global_indices = shard_global_indices(args.shard_index)
    seeds = [branch_stream_seed(observed.identity_sha256, index) for index in global_indices]

    state_lists: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "state_before",
            "pred_xstart",
            "p_mean",
            "p_standard_deviation",
            "transition_innovation",
            "epsilon_current_reconstructed",
            "epsilon_shifted",
            "theta",
            "secondary_raw_K",
            "secondary_component_scale",
            "secondary_K",
            "secondary_R",
            "secondary_L",
        )
    }
    generator_before: list[np.ndarray] = []
    generator_after: list[np.ndarray] = []
    shifted_forward_calls = 0

    rows = {
        int(value): index for index, value in enumerate(observed.arrays["internal_timestep"])
    }
    prefix = np.ascontiguousarray(
        observed.arrays["x_t"][rows[ROLLBACK_INTERNAL_TIMESTEP], TARGET_BATCH_INDEX],
        dtype=np.float32,
    )

    def _execute() -> tuple[torch.Tensor, np.ndarray, dict[str, Any]]:
        nonlocal shifted_forward_calls
        from diffusion import create_diffusion
        from diffusers.models import AutoencoderKL
        from download import find_model
        from models import DiT_models

        imported = {
            "diffusion": Path(sys.modules["diffusion"].__file__).resolve(),
            "download": Path(sys.modules["download"].__file__).resolve(),
            "models": Path(sys.modules["models"].__file__).resolve(),
        }
        expected_imports = {
            "diffusion": (args.dit_root / "diffusion/__init__.py").resolve(),
            "download": (args.dit_root / "download.py").resolve(),
            "models": (args.dit_root / "models.py").resolve(),
        }
        if imported != expected_imports:
            raise RuntimeError(f"upstream import shadowing: {imported} != {expected_imports}")
        torch.cuda.set_device(args.device_index)
        device = torch.device("cuda", args.device_index)
        model = DiT_models[MODEL_NAME](input_size=LATENT_SIZE, num_classes=NUM_CLASSES).to(device)
        model.load_state_dict(find_model(str(args.checkpoint)))
        model.eval()
        diffusion = create_diffusion(str(NUM_SAMPLING_STEPS))
        if not np.array_equal(np.asarray(diffusion.timestep_map), timestep_map):
            raise RuntimeError("runtime DiT timestep map differs from the validated observer")
        vae = AutoencoderKL.from_pretrained(
            str(args.vae_snapshot), local_files_only=True, use_safetensors=True
        ).to(device)
        vae.eval()

        first = torch.from_numpy(prefix).to(device=device).unsqueeze(0).repeat(BATCH_SIZE, 1, 1, 1)
        if tuple(first.shape) != (BATCH_SIZE, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE):
            raise AssertionError("restored x60 batch shape changed")
        y = torch.cat(
            [
                torch.full((BATCH_SIZE,), TARGET_CLASS_ID, dtype=torch.long, device=device),
                torch.full((BATCH_SIZE,), NULL_CLASS_ID, dtype=torch.long, device=device),
            ],
            dim=0,
        )
        model_kwargs = {"y": y, "cfg_scale": CFG_SCALE}
        generators = [torch.Generator(device=device).manual_seed(seed) for seed in seeds]
        setup_global_rng = _global_rng_state_sha256(device)
        previous_grad = torch.is_grad_enabled()
        torch.set_grad_enabled(False)
        try:
            for step_index, internal_t in enumerate(internal.tolist()):
                full_x = torch.cat([first, first], dim=0)
                t_internal = torch.full(
                    (FULL_BATCH_SIZE,), internal_t, dtype=torch.long, device=device
                )
                rng_before_model = _global_rng_state_sha256(device)
                out = diffusion.p_mean_variance(
                    model.forward_with_cfg,
                    full_x,
                    t_internal,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                )
                if _global_rng_state_sha256(device) != rng_before_model:
                    raise RuntimeError("implemented-P model call consumed global CUDA RNG")
                mean = out["mean"][:BATCH_SIZE].contiguous()
                pred = out["pred_xstart"][:BATCH_SIZE].contiguous()
                sigma = torch.exp(0.5 * out["log_variance"][:BATCH_SIZE]).contiguous()
                expected_state_shape = (
                    BATCH_SIZE,
                    LATENT_CHANNELS,
                    LATENT_SIZE,
                    LATENT_SIZE,
                )
                if any(tuple(value.shape) != expected_state_shape for value in (mean, pred, sigma)):
                    raise RuntimeError("implemented-P transition output shape changed")

                state_np = np.ascontiguousarray(first.cpu().numpy(), dtype=np.float32)
                pred_np = np.ascontiguousarray(pred.cpu().numpy(), dtype=np.float32)
                mean_np = np.ascontiguousarray(mean.cpu().numpy(), dtype=np.float32)
                sigma_np = np.ascontiguousarray(sigma.cpu().numpy(), dtype=np.float32)
                current_eps = reconstruct_current_epsilon(
                    state_np, pred_np, float(alpha[internal_t])
                )
                if effective[step_index]:
                    shifted_t = int(shifted[step_index])
                    rho = math.sqrt(float(alpha[shifted_t]) / float(alpha[internal_t]))
                    shifted_first = first * rho
                    shifted_full = torch.cat([shifted_first, shifted_first], dim=0)
                    t_original = torch.full(
                        (FULL_BATCH_SIZE,),
                        int(timestep_map[shifted_t]),
                        dtype=torch.long,
                        device=device,
                    )
                    rng_before_shifted = _global_rng_state_sha256(device)
                    shifted_output = model.forward_with_cfg(
                        shifted_full, t_original, y=y, cfg_scale=CFG_SCALE
                    )
                    if _global_rng_state_sha256(device) != rng_before_shifted:
                        raise RuntimeError("shifted model call consumed global CUDA RNG")
                    expected_shifted_shape = (
                        FULL_BATCH_SIZE,
                        2 * LATENT_CHANNELS,
                        LATENT_SIZE,
                        LATENT_SIZE,
                    )
                    if tuple(shifted_output.shape) != expected_shifted_shape:
                        raise RuntimeError(
                            f"shifted DiT output shape changed: {tuple(shifted_output.shape)}"
                        )
                    shifted_eps = np.ascontiguousarray(
                        shifted_output[:BATCH_SIZE, :LATENT_CHANNELS].cpu().numpy(),
                        dtype=np.float32,
                    )
                    direction = (
                        -rho
                        * shifted_eps.astype(np.float64, copy=False)
                        / math.sqrt(1.0 - float(alpha[shifted_t]))
                        + current_eps.astype(np.float64, copy=False)
                        / math.sqrt(1.0 - float(alpha[internal_t]))
                    )
                    direction = np.ascontiguousarray(direction, dtype=np.float64)
                    raw_K, component_scale, applied_K, whitened = (
                        construct_signed_components_before_innovation(
                            direction, sigma_np, masks, per_step_cap
                        )
                    )
                    shifted_forward_calls += 1
                else:
                    shifted_eps = current_eps.copy()
                    direction = np.zeros_like(current_eps, dtype=np.float64)
                    raw_K = np.zeros(
                        (BATCH_SIZE, SIGNED_COMPONENT_COUNT), dtype=np.float64
                    )
                    component_scale = np.ones_like(raw_K)
                    applied_K = np.zeros_like(raw_K)
                    whitened = np.zeros(
                        (
                            BATCH_SIZE,
                            SIGNED_COMPONENT_COUNT,
                            LATENT_CHANNELS,
                            LATENT_SIZE,
                            LATENT_SIZE,
                        ),
                        dtype=np.float64,
                    )

                # Strict predictability boundary: innovations do not exist until
                # every branch's operational Q shift has been fully constructed.
                innovation, before_hash, after_hash = _draw_branch_innovations(
                    generators, device=device, dtype=first.dtype
                )
                innovation_np = np.ascontiguousarray(
                    innovation.cpu().numpy(), dtype=np.float32
                )
                reward, increment = evaluate_components_after_innovation(
                    whitened, innovation_np
                )
                nonzero = float(internal_t != 0)
                following = mean + nonzero * sigma * innovation

                state_lists["state_before"].append(state_np)
                state_lists["pred_xstart"].append(pred_np)
                state_lists["p_mean"].append(mean_np)
                state_lists["p_standard_deviation"].append(sigma_np)
                state_lists["transition_innovation"].append(innovation_np)
                state_lists["epsilon_current_reconstructed"].append(current_eps)
                state_lists["epsilon_shifted"].append(shifted_eps)
                state_lists["theta"].append(direction)
                state_lists["secondary_raw_K"].append(raw_K)
                state_lists["secondary_component_scale"].append(component_scale)
                state_lists["secondary_K"].append(applied_K)
                state_lists["secondary_R"].append(reward)
                state_lists["secondary_L"].append(increment)
                generator_before.append(before_hash)
                generator_after.append(after_hash)
                first = following.detach()
                if step_index % 10 == 0 or step_index + 1 == steps:
                    print(
                        f"shard {args.shard_index}: sampled {step_index + 1}/{steps} suffix transitions",
                        flush=True,
                    )
            if _global_rng_state_sha256(device) != setup_global_rng:
                raise RuntimeError("suffix sampling changed global CUDA RNG")
            decoded = vae.decode(first / VAE_SCALING_FACTOR).sample
            if _global_rng_state_sha256(device) != setup_global_rng:
                raise RuntimeError("VAE decode changed global CUDA RNG")
            torch.cuda.synchronize(device)
            execution = {
                "device": str(device),
                "cuda_device_name": torch.cuda.get_device_name(device),
                "cuda_device_capability": list(torch.cuda.get_device_capability(device)),
                "shifted_forward_calls": shifted_forward_calls,
                "implemented_P_forward_calls": steps,
                "branch_count": BATCH_SIZE,
                "transition_count_per_branch_including_t0": steps,
                "explicit_generator_draw_count_per_branch": steps,
                "global_cuda_rng_unchanged_after_setup": True,
                "CFG_8_to_16_shape_preserved": True,
            }
            final_latents = np.ascontiguousarray(first.cpu().numpy(), dtype=np.float32)
            return decoded, final_latents, execution
        finally:
            torch.set_grad_enabled(previous_grad)

    decoded, final_latents, execution = _with_upstream_imports(args.dit_root, _execute)

    stacked: dict[str, np.ndarray] = {}
    for key, values in state_lists.items():
        # Lists are step-major; saved trace is branch-major for blind joins.
        stacked[key] = np.ascontiguousarray(np.stack(values, axis=1), dtype=TRACE_DTYPES[key])
    generator_before_np = np.ascontiguousarray(
        np.stack(generator_before, axis=1), dtype=TRACE_DTYPES["generator_state_sha256_before"]
    )
    generator_after_np = np.ascontiguousarray(
        np.stack(generator_after, axis=1), dtype=TRACE_DTYPES["generator_state_sha256_after"]
    )
    secondary_cumulative = np.cumsum(stacked["secondary_L"], axis=1, dtype=np.float64)
    secondary_mixture = _logmeanexp(secondary_cumulative, axis=2)
    primary_slice = PRIMARY_BASE_COMPONENT_INDEX  # +theta block is first.
    primary_L = stacked["secondary_L"][:, :, primary_slice]
    primary_log_e = np.cumsum(primary_L, axis=1, dtype=np.float64)
    alarm = primary_log_e >= ALARM_LOG_E
    ever = alarm.any(axis=1)
    first_alarm = np.full((BATCH_SIZE,), -1, dtype=np.int16)
    for branch_pos in range(BATCH_SIZE):
        if ever[branch_pos]:
            first_alarm[branch_pos] = int(np.flatnonzero(alarm[branch_pos])[0])
    arrays: dict[str, np.ndarray] = {
        "branch_global_index": np.asarray(global_indices, dtype=np.int16),
        "branch_stream_seed": np.asarray(seeds, dtype=np.int64),
        "generator_state_sha256_before": generator_before_np,
        "generator_state_sha256_after": generator_after_np,
        "internal_timestep": np.ascontiguousarray(internal, dtype=np.int16),
        "original_timestep": np.ascontiguousarray(timestep_map[internal], dtype=np.int16),
        "full_internal_alpha_bar": np.ascontiguousarray(alpha, dtype=np.float64),
        "full_original_timestep_map": np.ascontiguousarray(timestep_map, dtype=np.int64),
        "current_alpha_bar": np.ascontiguousarray(alpha[internal], dtype=np.float64),
        "shifted_internal_timestep": np.ascontiguousarray(shifted, dtype=np.int16),
        "shifted_original_timestep": np.ascontiguousarray(timestep_map[shifted], dtype=np.int16),
        "shifted_alpha_bar": np.ascontiguousarray(alpha[shifted], dtype=np.float64),
        "rho": np.ascontiguousarray(np.sqrt(alpha[shifted] / alpha[internal]), dtype=np.float64),
        "effective_nonidentity": np.ascontiguousarray(effective.astype(np.uint8)),
        "per_step_K_cap": np.asarray(per_step_cap, dtype=np.float64),
        "tile_bounds_yxyx": np.ascontiguousarray(tile_bounds, dtype=np.int16),
        "base_component_name": component_names,
        "signed_component_base_index": np.asarray(
            list(range(BASE_COMPONENT_COUNT)) * len(SIGN_VALUES), dtype=np.int16
        ),
        "signed_component_sign": np.asarray(
            [sign for sign in SIGN_VALUES for _ in range(BASE_COMPONENT_COUNT)], dtype=np.int8
        ),
        **stacked,
        "secondary_component_log_e": np.ascontiguousarray(
            secondary_cumulative, dtype=np.float64
        ),
        "secondary_path_mixture_log_e": np.ascontiguousarray(
            secondary_mixture, dtype=np.float64
        ),
        "secondary_terminal_path_mixture_log_e": np.ascontiguousarray(
            secondary_mixture[:, -1], dtype=np.float64
        ),
        "secondary_running_max_path_mixture_log_e": np.ascontiguousarray(
            np.maximum(0.0, secondary_mixture.max(axis=1)), dtype=np.float64
        ),
        "primary_raw_K": np.ascontiguousarray(
            stacked["secondary_raw_K"][:, :, primary_slice], dtype=np.float64
        ),
        "primary_component_scale": np.ascontiguousarray(
            stacked["secondary_component_scale"][:, :, primary_slice], dtype=np.float64
        ),
        "primary_K": np.ascontiguousarray(
            stacked["secondary_K"][:, :, primary_slice], dtype=np.float64
        ),
        "primary_R": np.ascontiguousarray(
            stacked["secondary_R"][:, :, primary_slice], dtype=np.float64
        ),
        "primary_L": np.ascontiguousarray(primary_L, dtype=np.float64),
        "primary_log_e": np.ascontiguousarray(primary_log_e, dtype=np.float64),
        "primary_alarm_after_transition": np.ascontiguousarray(alarm.astype(np.uint8)),
        "primary_ever_alarm": np.ascontiguousarray(ever.astype(np.uint8)),
        "primary_first_alarm_step_index": first_alarm,
        "primary_terminal_log_e": np.ascontiguousarray(primary_log_e[:, -1], dtype=np.float64),
        "primary_running_max_log_e": np.ascontiguousarray(
            np.maximum(0.0, primary_log_e.max(axis=1)), dtype=np.float64
        ),
        "final_latents": final_latents,
    }
    expected_shapes = _trace_shapes()
    if set(arrays) != set(TRACE_DTYPES) or set(arrays) != set(expected_shapes):
        raise AssertionError("private trace key set changed")
    for key, value in arrays.items():
        if value.shape != expected_shapes[key] or value.dtype != TRACE_DTYPES[key]:
            raise AssertionError(
                f"private trace contract changed: {key}: {value.shape}/{value.dtype}"
            )
        if value.dtype.kind not in "US" and not np.isfinite(value).all():
            raise RuntimeError(f"non-finite private trace value: {key}")
    return arrays, decoded, execution


def _save_blind_outputs(
    staging: Path,
    decoded: torch.Tensor,
    global_indices: tuple[int, ...],
    *,
    save_image: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    directory = staging / "blind_images"
    directory.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    for local_index, global_index in enumerate(global_indices):
        identifier = blind_id(global_index)
        path = directory / f"{identifier}.png"
        save_image(
            decoded[local_index], path, nrow=1, padding=0, normalize=True, value_range=(-1, 1)
        )
        records.append(
            {
                "local_index": local_index,
                "global_index": global_index,
                "blind_id": identifier,
                "stream_seed": None,  # Filled by caller; no evidence statistic is exposed.
                "image": _png_record(path, staging, (IMAGE_SIZE, IMAGE_SIZE)),
            }
        )
    grid_path = staging / "blind_grid.png"
    save_image(decoded, grid_path, nrow=4, normalize=True, value_range=(-1, 1))
    grid_record = _png_record(grid_path, staging, (1_034, 518))
    for local_index, record in enumerate(records):
        tile_sha = _array_raw_sha256(_grid_tile_pixels(grid_path, local_index))
        if tile_sha != record["image"]["pixel_sha256"]:
            raise RuntimeError("blind image differs from its blind-grid tile")
        record["grid_tile_pixel_sha256"] = tile_sha
    return records, grid_record


def _load_trace(path: Path, record: dict[str, Any], root: Path) -> dict[str, np.ndarray]:
    if set(record) != {"relative_path", "bytes", "sha256", "keys", "schema"}:
        raise RuntimeError("private trace public record schema changed or leaks extra metadata")
    if record.get("relative_path") != path.relative_to(root).as_posix():
        raise RuntimeError("private trace relative path changed")
    if (
        not path.is_file()
        or path.stat().st_size != record.get("bytes")
        or sha256_file(path) != record.get("sha256")
    ):
        raise RuntimeError("private trace file identity failed")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {
            key: _copy_npz_array_preserve_shape(archive[key]) for key in archive.files
        }
    if sorted(arrays) != record.get("keys"):
        raise RuntimeError("private trace key set changed")
    if set(record.get("schema", {})) != set(arrays):
        raise RuntimeError("private trace public dtype/shape schema changed")
    shapes = _trace_shapes()
    if set(arrays) != set(TRACE_DTYPES) or set(arrays) != set(shapes):
        raise RuntimeError("private trace schema key set changed")
    for key, value in arrays.items():
        expected_record = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        if (
            value.shape != shapes[key]
            or value.dtype != TRACE_DTYPES[key]
            or record.get("schema", {}).get(key) != expected_record
        ):
            raise RuntimeError(f"private trace schema/identity failed: {key}")
        if value.dtype.kind not in "US" and not np.isfinite(value).all():
            raise RuntimeError(f"private trace contains non-finite values: {key}")
    return arrays


def _validate_trace_math(arrays: dict[str, np.ndarray], manifest: dict[str, Any]) -> None:
    shard = int(manifest["pool"]["this_shard_index"])
    expected_indices = np.asarray(shard_global_indices(shard), dtype=np.int16)
    observer_identity = str(manifest["input_prefix"]["observer_manifest_identity_sha256"])
    expected_seeds = np.asarray(
        [branch_stream_seed(observer_identity, int(index)) for index in expected_indices],
        dtype=np.int64,
    )
    if not np.array_equal(arrays["branch_global_index"], expected_indices):
        raise RuntimeError("trace global branch allocation changed")
    if not np.array_equal(arrays["branch_stream_seed"], expected_seeds):
        raise RuntimeError("trace branch stream seed derivation changed")
    internal = np.arange(ROLLBACK_INTERNAL_TIMESTEP, -1, -1, dtype=np.int16)
    if not np.array_equal(arrays["internal_timestep"], internal):
        raise RuntimeError("trace suffix timestep axis changed")
    full_alpha = arrays["full_internal_alpha_bar"]
    full_timestep_map = arrays["full_original_timestep_map"]
    if (
        np.any(full_alpha <= 0.0)
        or np.any(full_alpha >= 1.0)
        or np.any(np.diff(full_alpha) >= 0.0)
        or full_timestep_map[0] != 0
        or np.any(np.diff(full_timestep_map) <= 0)
        or int(full_timestep_map[-1]) > 999
    ):
        raise RuntimeError("saved full DiT schedule is invalid")
    if _array_raw_sha256(full_alpha) != manifest["schedule"]["alpha_bar_raw_sha256"]:
        raise RuntimeError("saved full alpha schedule differs from the manifest")
    if _array_raw_sha256(full_timestep_map) != manifest["schedule"][
        "original_timestep_map_raw_sha256"
    ]:
        raise RuntimeError("saved full timestep map differs from the manifest")
    if arrays["original_timestep"][-1] != 0 or np.any(
        np.diff(arrays["original_timestep"].astype(np.int64)) >= 0
    ):
        raise RuntimeError("trace original timestep axis is invalid")
    if not np.array_equal(arrays["original_timestep"], full_timestep_map[internal]):
        raise RuntimeError("suffix original timesteps do not match the full schedule")
    alpha_current = arrays["current_alpha_bar"]
    alpha_shifted = arrays["shifted_alpha_bar"]
    if (
        np.any(alpha_current <= 0.0)
        or np.any(alpha_current >= 1.0)
        or np.any(alpha_shifted <= 0.0)
        or np.any(alpha_shifted >= 1.0)
        or np.any(arrays["shifted_internal_timestep"] < internal)
    ):
        raise RuntimeError("saved scale coordinates are invalid")
    if not np.array_equal(alpha_current, full_alpha[internal]):
        raise RuntimeError("current alpha-bar does not match the full schedule")
    if not np.array_equal(alpha_shifted, full_alpha[arrays["shifted_internal_timestep"]]):
        raise RuntimeError("shifted alpha-bar does not match the full schedule")
    if not np.array_equal(
        arrays["shifted_original_timestep"],
        full_timestep_map[arrays["shifted_internal_timestep"]],
    ):
        raise RuntimeError("shifted original timesteps do not match the full schedule")
    remapped = _schedule(full_alpha)
    if not np.array_equal(
        arrays["shifted_internal_timestep"], remapped["shifted_internal_timestep"]
    ):
        raise RuntimeError("nearest additive Delta-nu=0.25 mapping changed")
    if not np.allclose(
        arrays["rho"], np.sqrt(alpha_shifted / alpha_current), rtol=0.0, atol=2e-16
    ):
        raise RuntimeError("saved rho does not reconstruct")
    effective = arrays["effective_nonidentity"].astype(bool)
    expected_effective = (arrays["shifted_internal_timestep"] != internal) & (internal > 0)
    if not np.array_equal(effective, expected_effective):
        raise RuntimeError("effective shifted-step flags changed")
    expected_cap = TOTAL_K_PER_COMPONENT / int(effective.sum()) * (1.0 - 2e-12)
    if arrays["per_step_K_cap"].item() != expected_cap:
        raise RuntimeError("per-step K allocation changed")
    masks, expected_bounds, expected_names = _base_masks()
    if not np.array_equal(arrays["tile_bounds_yxyx"], expected_bounds) or not np.array_equal(
        arrays["base_component_name"], expected_names
    ):
        raise RuntimeError("fixed component geometry changed")
    expected_base = np.asarray(
        list(range(BASE_COMPONENT_COUNT)) * len(SIGN_VALUES), dtype=np.int16
    )
    expected_sign = np.asarray(
        [sign for sign in SIGN_VALUES for _ in range(BASE_COMPONENT_COUNT)], dtype=np.int8
    )
    if not np.array_equal(arrays["signed_component_base_index"], expected_base) or not np.array_equal(
        arrays["signed_component_sign"], expected_sign
    ):
        raise RuntimeError("signed path-component ordering changed")

    initial = arrays["state_before"][:, 0]
    if not np.array_equal(initial, np.broadcast_to(initial[0], initial.shape)):
        raise RuntimeError("validation branches did not share exactly one x60 prefix")
    if _array_raw_sha256(initial[0]) != manifest["input_prefix"].get(
        "target_x60_raw_sha256"
    ):
        raise RuntimeError("saved x60 prefix hash changed")
    if np.any(arrays["p_standard_deviation"] <= 0.0):
        raise RuntimeError("implemented-P sigma must be strictly positive")
    steps = len(internal)
    expected_eps = np.empty_like(arrays["epsilon_current_reconstructed"])
    expected_theta = np.zeros_like(arrays["theta"])
    expected_raw = np.zeros_like(arrays["secondary_raw_K"])
    expected_scale = np.ones_like(arrays["secondary_component_scale"])
    expected_K = np.zeros_like(arrays["secondary_K"])
    expected_R = np.zeros_like(arrays["secondary_R"])
    expected_L = np.zeros_like(arrays["secondary_L"])
    for step_index in range(steps):
        expected_eps[:, step_index] = reconstruct_current_epsilon(
            arrays["state_before"][:, step_index],
            arrays["pred_xstart"][:, step_index],
            float(alpha_current[step_index]),
        )
        if effective[step_index]:
            direction = (
                -float(arrays["rho"][step_index])
                * arrays["epsilon_shifted"][:, step_index].astype(np.float64, copy=False)
                / math.sqrt(1.0 - float(alpha_shifted[step_index]))
                + expected_eps[:, step_index].astype(np.float64, copy=False)
                / math.sqrt(1.0 - float(alpha_current[step_index]))
            )
            expected_theta[:, step_index] = direction
            raw, scale, K, u = construct_signed_components_before_innovation(
                np.ascontiguousarray(direction, dtype=np.float64),
                arrays["p_standard_deviation"][:, step_index],
                masks,
                expected_cap,
            )
            R, L = evaluate_components_after_innovation(
                u, arrays["transition_innovation"][:, step_index]
            )
            expected_raw[:, step_index] = raw
            expected_scale[:, step_index] = scale
            expected_K[:, step_index] = K
            expected_R[:, step_index] = R
            expected_L[:, step_index] = L
        elif not np.array_equal(
            arrays["epsilon_shifted"][:, step_index], expected_eps[:, step_index]
        ):
            raise RuntimeError("inactive shifted epsilon must equal current epsilon")
    if not np.array_equal(arrays["epsilon_current_reconstructed"], expected_eps):
        error = float(
            np.max(np.abs(arrays["epsilon_current_reconstructed"] - expected_eps), initial=0.0)
        )
        if error > 1e-6:
            raise RuntimeError(f"current epsilon reconstruction failed: max_abs={error}")
    theta_error = float(np.max(np.abs(arrays["theta"] - expected_theta), initial=0.0))
    if theta_error > 2e-13:
        raise RuntimeError(f"theta reconstruction failed: max_abs={theta_error}")
    for key, expected in (
        ("secondary_raw_K", expected_raw),
        ("secondary_component_scale", expected_scale),
        ("secondary_K", expected_K),
        ("secondary_R", expected_R),
        ("secondary_L", expected_L),
    ):
        if not np.array_equal(arrays[key], expected):
            error = float(np.max(np.abs(arrays[key] - expected), initial=0.0))
            raise RuntimeError(f"evidence reconstruction failed: {key}/max_abs={error}")
    cumulative = np.cumsum(expected_L, axis=1, dtype=np.float64)
    mixture = _logmeanexp(cumulative, axis=2)
    if not np.array_equal(arrays["secondary_component_log_e"], cumulative):
        raise RuntimeError("secondary path log-e does not reconstruct")
    if not np.array_equal(arrays["secondary_path_mixture_log_e"], mixture):
        raise RuntimeError("secondary fixed path mixture does not reconstruct")
    if not np.array_equal(arrays["secondary_terminal_path_mixture_log_e"], mixture[:, -1]):
        raise RuntimeError("secondary terminal mixture summary changed")
    if not np.array_equal(
        arrays["secondary_running_max_path_mixture_log_e"],
        np.maximum(0.0, mixture.max(axis=1)),
    ):
        raise RuntimeError("secondary running maximum summary changed")
    primary_index = PRIMARY_BASE_COMPONENT_INDEX
    primary_bindings = {
        "primary_raw_K": expected_raw[:, :, primary_index],
        "primary_component_scale": expected_scale[:, :, primary_index],
        "primary_K": expected_K[:, :, primary_index],
        "primary_R": expected_R[:, :, primary_index],
        "primary_L": expected_L[:, :, primary_index],
    }
    for key, expected in primary_bindings.items():
        if not np.array_equal(arrays[key], expected):
            raise RuntimeError(f"primary candidate is not bound to +theta/tile_12: {key}")
    primary_log_e = np.cumsum(primary_bindings["primary_L"], axis=1, dtype=np.float64)
    alarm = primary_log_e >= ALARM_LOG_E
    ever = alarm.any(axis=1)
    first = np.full((BATCH_SIZE,), -1, dtype=np.int16)
    for branch_pos in range(BATCH_SIZE):
        if ever[branch_pos]:
            first[branch_pos] = int(np.flatnonzero(alarm[branch_pos])[0])
    primary_summaries = {
        "primary_log_e": primary_log_e,
        "primary_alarm_after_transition": alarm.astype(np.uint8),
        "primary_ever_alarm": ever.astype(np.uint8),
        "primary_first_alarm_step_index": first,
        "primary_terminal_log_e": primary_log_e[:, -1],
        "primary_running_max_log_e": np.maximum(0.0, primary_log_e.max(axis=1)),
    }
    for key, expected in primary_summaries.items():
        if not np.array_equal(arrays[key], expected):
            raise RuntimeError(f"primary evidence summary changed: {key}")
    if np.any(arrays["secondary_K"].sum(axis=1) > TOTAL_K_PER_COMPONENT * (1 + 2e-12)):
        raise RuntimeError("a path component exceeded total suffix K=0.5")
    if np.any(arrays["primary_K"].sum(axis=1) > TOTAL_K_PER_COMPONENT * (1 + 2e-12)):
        raise RuntimeError("primary path exceeded total suffix K=0.5")

    for step_index, internal_t in enumerate(internal.tolist()):
        expected_next = (
            arrays["p_mean"][:, step_index]
            + np.float32(1.0 if internal_t > 0 else 0.0)
            * arrays["p_standard_deviation"][:, step_index]
            * arrays["transition_innovation"][:, step_index]
        )
        actual_next = (
            arrays["state_before"][:, step_index + 1]
            if step_index + 1 < steps
            else arrays["final_latents"]
        )
        if not np.array_equal(expected_next, actual_next):
            error = float(
                np.max(
                    np.abs(expected_next.astype(np.float64) - actual_next.astype(np.float64)),
                    initial=0.0,
                )
            )
            if error > 2e-6:
                raise RuntimeError(
                    f"baseline-P transition does not reconstruct at t={internal_t}: max_abs={error}"
                )
    for key in ("generator_state_sha256_before", "generator_state_sha256_after"):
        if np.any(np.char.str_len(arrays[key]) != 64):
            raise RuntimeError(f"invalid generator-state hashes: {key}")
    if np.any(
        arrays["generator_state_sha256_before"] == arrays["generator_state_sha256_after"]
    ):
        raise RuntimeError("a branch generator failed to advance")
    if not np.array_equal(
        arrays["generator_state_sha256_after"][:, :-1],
        arrays["generator_state_sha256_before"][:, 1:],
    ):
        raise RuntimeError("a branch-local generator stream is discontinuous")


def validate_output_bundle(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")):
        raise RuntimeError("validation shard bundle must not contain symlinks")
    manifest = _read_self_hashed_json(root / "manifest.json", "identity_sha256")
    fixed_manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "role": "PROSPECTIVE_WITHIN_PREFIX_VALIDATION_SHARD",
    }
    if any(manifest.get(key) != value for key, value in fixed_manifest.items()):
        raise RuntimeError("validation shard manifest identity changed")
    if manifest.get("scope", {}).get("general_confirmation") is not False:
        raise RuntimeError("bundle misstates the within-prefix scope")
    if manifest.get("sampling_distribution", {}).get("baseline_P_suffix_unchanged") is not True:
        raise RuntimeError("bundle does not declare an unchanged baseline-P sampler")
    runner = Path(__file__).resolve()
    if manifest.get("runner", {}).get("sha256") != sha256_file(runner):
        raise RuntimeError("bundle was produced by a different runner source")
    for name, record in manifest.get("sources", {}).get("local_dependencies", {}).items():
        path = Path(record.get("path", ""))
        if path.name != name or not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise RuntimeError(f"local dependency changed: {name}")
    protocol_path = root / PROTOCOL_COPY_NAME
    protocol = _read_self_hashed_json(protocol_path, "protocol_identity_sha256")
    protocol_record = manifest.get("protocol", {})
    if (
        protocol.get("protocol_identity_sha256")
        != protocol_record.get("protocol_identity_sha256")
        or sha256_file(protocol_path) != protocol_record.get("source_file_sha256")
    ):
        raise RuntimeError("copied protocol identity changed")
    _load_protocol(protocol_path)
    binding = protocol["frozen_input_binding"]
    manifest_binding = {
        "observer_manifest_identity_sha256": manifest["input_prefix"][
            "observer_manifest_identity_sha256"
        ],
        "observer_manifest_file_sha256": manifest["input_prefix"][
            "observer_manifest_file_sha256"
        ],
        "observer_results_file_sha256": manifest["input_prefix"][
            "observer_results_file_sha256"
        ],
        "observer_completion_file_sha256": manifest["input_prefix"][
            "observer_completion_file_sha256"
        ],
        "observer_trace_sha256": manifest["input_prefix"]["observer_trace_sha256"],
        "baseline_manifest_identity_sha256": manifest["input_prefix"][
            "baseline_manifest_identity_sha256"
        ],
        "target_x60_raw_sha256": manifest["input_prefix"]["target_x60_raw_sha256"],
        "alpha_bar_raw_sha256": manifest["schedule"]["alpha_bar_raw_sha256"],
        "original_timestep_map_raw_sha256": manifest["schedule"][
            "original_timestep_map_raw_sha256"
        ],
    }
    if manifest_binding != binding:
        raise RuntimeError("bundle manifest differs from the protocol's frozen input binding")
    results = _read_self_hashed_json(root / "results.json", "payload_sha256")
    fixed_results = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "manifest_identity_sha256": manifest["identity_sha256"],
        "within_prefix_validation": True,
        "general_confirmation": False,
        "baseline_P_sampling_unchanged": True,
        "evidence_values_exposed": False,
        "alarm_flags_exposed": False,
        "branch_ranking_or_selection_performed": False,
    }
    if any(results.get(key) != value for key, value in fixed_results.items()):
        raise RuntimeError("validation results scope/identity changed")
    expected_result_keys = {
        *fixed_results,
        "shard_index",
        "branch_records",
        "blind_grid",
        "private_trace",
        "execution",
        "wall_seconds_before_publication",
        "platform",
        "payload_sha256",
    }
    if set(results) != expected_result_keys:
        raise RuntimeError("results JSON schema changed")
    shard = int(manifest["pool"]["this_shard_index"])
    if results.get("shard_index") != shard:
        raise RuntimeError("results shard identity changed")
    records = results.get("branch_records")
    indices = shard_global_indices(shard)
    if not isinstance(records, list) or len(records) != BATCH_SIZE:
        raise RuntimeError("blind branch record count changed")
    expected_files = {
        (root / "manifest.json").resolve(),
        (root / "results.json").resolve(),
        (root / "completion.json").resolve(),
        protocol_path.resolve(),
        (root / TRACE_NAME).resolve(),
        (root / "blind_grid.png").resolve(),
    }
    grid_record = _png_record(root / "blind_grid.png", root, (1_034, 518))
    if results.get("blind_grid") != grid_record:
        raise RuntimeError("blind grid identity changed")
    observer_identity = manifest["input_prefix"]["observer_manifest_identity_sha256"]
    for local_index, global_index in enumerate(indices):
        identifier = blind_id(global_index)
        path = root / "blind_images" / f"{identifier}.png"
        expected_files.add(path.resolve())
        image_record = _png_record(path, root, (IMAGE_SIZE, IMAGE_SIZE))
        expected_record = {
            "local_index": local_index,
            "global_index": global_index,
            "blind_id": identifier,
            "stream_seed": branch_stream_seed(observer_identity, global_index),
            "image": image_record,
            "grid_tile_pixel_sha256": image_record["pixel_sha256"],
        }
        if records[local_index] != expected_record:
            raise RuntimeError(f"blind branch record changed: {identifier}")
        if _array_raw_sha256(_grid_tile_pixels(root / "blind_grid.png", local_index)) != image_record[
            "pixel_sha256"
        ]:
            raise RuntimeError(f"blind image/grid tile mismatch: {identifier}")
    arrays = _load_trace(root / TRACE_NAME, results["private_trace"], root)
    _validate_trace_math(arrays, manifest)
    if not np.array_equal(
        arrays["branch_stream_seed"],
        np.asarray([record["stream_seed"] for record in records], dtype=np.int64),
    ):
        raise RuntimeError("public seed accounting is not bound to private trace")
    actual_files = {path.resolve() for path in root.rglob("*") if path.is_file()}
    if actual_files != expected_files:
        raise RuntimeError("validation shard file set changed")
    expected_directories = {(root / "blind_images").resolve()}
    actual_directories = {path.resolve() for path in root.rglob("*") if path.is_dir()}
    if actual_directories != expected_directories:
        raise RuntimeError("validation shard directory set changed")
    completion = _read_self_hashed_json(root / "completion.json", "payload_sha256")
    fixed_completion = {
        "complete": True,
        "manifest_identity_sha256": manifest["identity_sha256"],
        "manifest_file_sha256": sha256_file(root / "manifest.json"),
        "results_payload_sha256": results["payload_sha256"],
        "results_file_sha256": sha256_file(root / "results.json"),
        "private_trace_sha256": results["private_trace"]["sha256"],
        "shard_index": shard,
        "branch_count": BATCH_SIZE,
        "evidence_values_exposed": False,
    }
    if any(completion.get(key) != value for key, value in fixed_completion.items()):
        raise RuntimeError("completion links/hashes changed")
    return manifest, results


def run_real(
    args: argparse.Namespace,
    *,
    protocol: dict[str, Any],
    observed: Any,
    source: dict[str, Any],
    checkpoint: dict[str, Any],
    vae: dict[str, Any],
    alpha: np.ndarray,
    timestep_map: np.ndarray,
) -> None:
    if protocol.get("protocol_status") != "FROZEN_BEFORE_GPU_EXECUTION":
        raise RuntimeError(
            "real GPU execution is disabled while the protocol remains a draft; "
            "review it, set protocol_status=FROZEN_BEFORE_GPU_EXECUTION, and recompute "
            "protocol_identity_sha256 before launching any shard"
        )
    if args.outdir.exists():
        raise RuntimeError(f"refusing to overwrite existing output path: {args.outdir}")
    manifest = build_manifest(
        args,
        protocol=protocol,
        observed=observed,
        source=source,
        checkpoint=checkpoint,
        vae=vae,
        alpha=alpha,
        timestep_map=timestep_map,
    )
    rows = {
        int(value): index for index, value in enumerate(observed.arrays["internal_timestep"])
    }
    prefix = np.ascontiguousarray(
        observed.arrays["x_t"][rows[ROLLBACK_INTERNAL_TIMESTEP], TARGET_BATCH_INDEX],
        dtype=np.float32,
    )
    manifest["input_prefix"]["target_x60_raw_sha256"] = _array_raw_sha256(prefix)
    manifest["identity_sha256"] = _canonical_self_hash(manifest, "identity_sha256")
    args.outdir.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with tempfile.TemporaryDirectory(
        prefix=f".{args.outdir.name}.staging-", dir=args.outdir.parent
    ) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        atomic_json_dump(manifest, staging / "manifest.json")
        # Preserve the exact reviewed protocol bytes, in addition to its
        # canonical self-hash, so whitespace reserialization cannot break the
        # manifest's source-file identity.
        shutil.copyfile(args.protocol, staging / PROTOCOL_COPY_NAME)
        arrays, decoded, execution = run_sampling_shard(
            args, observed=observed, alpha=alpha, timestep_map=timestep_map
        )

        def _save() -> tuple[list[dict[str, Any]], dict[str, Any]]:
            from torchvision.utils import save_image

            return _save_blind_outputs(
                staging,
                decoded,
                shard_global_indices(args.shard_index),
                save_image=save_image,
            )

        branch_records, grid_record = _save()
        for local_index, record in enumerate(branch_records):
            record["stream_seed"] = int(arrays["branch_stream_seed"][local_index])
        trace_path = staging / TRACE_NAME
        _atomic_npz_dump(arrays, trace_path)
        trace_record = _trace_record(trace_path, arrays, staging)
        results: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "manifest_identity_sha256": manifest["identity_sha256"],
            "within_prefix_validation": True,
            "general_confirmation": False,
            "baseline_P_sampling_unchanged": True,
            "evidence_values_exposed": False,
            "alarm_flags_exposed": False,
            "branch_ranking_or_selection_performed": False,
            "shard_index": args.shard_index,
            "branch_records": branch_records,
            "blind_grid": grid_record,
            "private_trace": trace_record,
            "execution": execution,
            "wall_seconds_before_publication": time.time() - started,
            "platform": {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python": sys.version,
                "dependencies": dependency_identity(),
            },
        }
        results["payload_sha256"] = _canonical_self_hash(results, "payload_sha256")
        atomic_json_dump(results, staging / "results.json")
        completion: dict[str, Any] = {
            "complete": True,
            "manifest_identity_sha256": manifest["identity_sha256"],
            "manifest_file_sha256": sha256_file(staging / "manifest.json"),
            "results_payload_sha256": results["payload_sha256"],
            "results_file_sha256": sha256_file(staging / "results.json"),
            "private_trace_sha256": trace_record["sha256"],
            "shard_index": args.shard_index,
            "branch_count": BATCH_SIZE,
            "evidence_values_exposed": False,
            "finished_unix": time.time(),
            "wall_seconds": time.time() - started,
        }
        completion["payload_sha256"] = _canonical_self_hash(completion, "payload_sha256")
        atomic_json_dump(completion, staging / "completion.json")
        validate_output_bundle(staging)
        _atomic_install_directory_noreplace(staging, args.outdir)
    validate_output_bundle(args.outdir)
    print(
        json.dumps(
            {
                "status": "complete",
                "outdir": str(args.outdir),
                "shard_index": args.shard_index,
                "blind_ids": [blind_id(index) for index in shard_global_indices(args.shard_index)],
                "evidence_values_exposed": False,
                "review_before_opening": TRACE_NAME,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def run_self_test(protocol_path: Path) -> None:
    if torch.cuda.is_initialized():
        raise RuntimeError("self-test must begin before CUDA initialization")
    protocol = _load_protocol(protocol_path)
    if protocol["scope"]["general_confirmation"] is not False:
        raise AssertionError("protocol scope guard failed")
    scalar = np.asarray(0.5, dtype=np.float64)
    if _copy_npz_array_preserve_shape(scalar).shape != ():
        raise AssertionError("NPZ scalar loader changed a zero-dimensional array shape")
    identity = "a" * 64
    seeds = [branch_stream_seed(identity, index) for index in range(TOTAL_POOL_BRANCHES)]
    if len(set(seeds)) != TOTAL_POOL_BRANCHES:
        raise AssertionError("pool seed collision")
    if sorted(index for shard in range(TOTAL_SHARDS) for index in shard_global_indices(shard)) != list(
        range(TOTAL_POOL_BRANCHES)
    ):
        raise AssertionError("shards do not partition the 32-path pool")
    ids = [blind_id(index) for index in range(TOTAL_POOL_BRANCHES)]
    if len(set(ids)) != TOTAL_POOL_BRANCHES:
        raise AssertionError("blind-id collision")
    first_draws = []
    for seed in seeds[:8]:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        first_draws.append(torch.randn((4, 5, 5), generator=generator))
        replay = torch.randn((4, 5, 5), generator=torch.Generator(device="cpu").manual_seed(seed))
        if not torch.equal(first_draws[-1], replay):
            raise AssertionError("branch stream is not reproducible")
    if any(torch.equal(first_draws[0], value) for value in first_draws[1:]):
        raise AssertionError("distinct branch seeds produced identical toy draws")

    rng = np.random.default_rng(23)
    theta = np.ascontiguousarray(rng.normal(size=(8, 4, 32, 32)), dtype=np.float64)
    sigma = np.ascontiguousarray(np.exp(rng.normal(-2.0, 0.1, size=theta.shape)), dtype=np.float32)
    masks, bounds, names = _base_masks()
    if bounds.shape != (16, 4) or names[PRIMARY_BASE_COMPONENT_INDEX] != "tile_12":
        raise AssertionError("primary tile indexing changed")
    raw, scale, K, u = construct_signed_components_before_innovation(theta, sigma, masks, 0.01)
    if raw.shape != (8, 34) or np.any(scale > 1.0) or np.any(K > 0.01 * (1 + 2e-12)):
        raise AssertionError("signed component construction failed")
    if not np.array_equal(u[:, :BASE_COMPONENT_COUNT], -u[:, BASE_COMPONENT_COUNT:]):
        raise AssertionError("+/- component pairing changed")
    z = np.ascontiguousarray(rng.normal(size=theta.shape), dtype=np.float32)
    R, L = evaluate_components_after_innovation(u, z)
    if not np.array_equal(L, R - K):
        raise AssertionError("Gaussian LR decomposition failed")
    cumulative = np.cumsum(np.stack([L, L], axis=1), axis=1, dtype=np.float64)
    mixture = _logmeanexp(cumulative, axis=2)
    manual = np.log(np.mean(np.exp(cumulative), axis=2))
    if not np.allclose(mixture, manual, rtol=0.0, atol=2e-15):
        raise AssertionError("fixed path mixture calculation failed")

    toy_u = np.zeros((1, SIGNED_COMPONENT_COUNT, 1, 1, 1), dtype=np.float64)
    toy_u[0, 0, 0, 0, 0] = 0.15
    toy_u[0, BASE_COMPONENT_COUNT, 0, 0, 0] = -0.15
    values = []
    for _ in range(20_000):
        toy_z = np.ascontiguousarray(rng.normal(size=(1, 1, 1, 1)), dtype=np.float32)
        _, toy_L = evaluate_components_after_innovation(toy_u, toy_z)
        values.append(float(np.exp(_logmeanexp(toy_L, axis=1)[0])))
    if abs(float(np.mean(values)) - 1.0) > 0.01:
        raise AssertionError("fixed signed-mixture e-value calibration failed")
    if torch.cuda.is_initialized():
        raise AssertionError("CPU self-test initialized CUDA")
    print(
        "self-test passed: protocol guards, 4x8 shard partition, 32 unique/reproducible "
        "branch streams, blind IDs, tile_12 binding, +/- fixed path mixture, K cap, "
        "Gaussian LR identity, Monte Carlo calibration, and CPU-only execution"
    )


def dry_run(
    args: argparse.Namespace,
    *,
    protocol: dict[str, Any],
    observed: Any,
    source: dict[str, Any],
    vae: dict[str, Any],
    alpha: np.ndarray,
    timestep_map: np.ndarray,
) -> None:
    schedule = _schedule(alpha)
    rows = {
        int(value): index for index, value in enumerate(observed.arrays["internal_timestep"])
    }
    prefix = np.ascontiguousarray(
        observed.arrays["x_t"][rows[ROLLBACK_INTERNAL_TIMESTEP], TARGET_BATCH_INDEX],
        dtype=np.float32,
    )
    indices = shard_global_indices(args.shard_index)
    all_seeds = [branch_stream_seed(observed.identity_sha256, index) for index in range(TOTAL_POOL_BRANCHES)]
    probe = checkpoint_dry_probe(args.checkpoint)
    payload = {
        "status": "dry-run",
        "experiment": EXPERIMENT,
        "gpu_model_loaded": False,
        "gpu_sampling_started": False,
        "protocol_status": protocol.get("protocol_status"),
        "protocol_identity_sha256": protocol["protocol_identity_sha256"],
        "within_prefix_validation_only": True,
        "general_confirmation": False,
        "prefix": {
            "seed": PREFIX_SEED,
            "class_id": TARGET_CLASS_ID,
            "internal_timestep": ROLLBACK_INTERNAL_TIMESTEP,
            "state_shape": list(prefix.shape),
            "state_raw_sha256": _array_raw_sha256(prefix),
            "observer_identity_sha256": observed.identity_sha256,
        },
        "pool": {
            "shards": TOTAL_SHARDS,
            "branches_per_shard": BATCH_SIZE,
            "total_branches": TOTAL_POOL_BRANCHES,
            "all_32_stream_seeds_unique": len(set(all_seeds)) == TOTAL_POOL_BRANCHES,
            "this_shard_index": args.shard_index,
            "this_shard_global_indices": list(indices),
            "this_shard_blind_ids": [blind_id(index) for index in indices],
            "this_shard_stream_seeds": [all_seeds[index] for index in indices],
        },
        "candidate": {
            "delta_nu": DELTA_NU,
            "primary": "+theta/tile_12",
            "total_K": TOTAL_K_PER_COMPONENT,
            "alarm_log_e": ALARM_LOG_E,
            "secondary": "fixed path mixture: (global+16 tiles) x (+/-theta)",
            "effective_stochastic_steps": schedule["effective_step_count"],
            "per_step_K_cap": schedule["per_step_K_cap"],
        },
        "CFG": {
            "first_half_target_branches": BATCH_SIZE,
            "full_model_batch": FULL_BATCH_SIZE,
            "official_shape_preserved": True,
        },
        "checkpoint_probe": probe,
        "source": source,
        "vae": vae,
        "schedule_hashes": {
            "alpha": _array_raw_sha256(np.ascontiguousarray(alpha, dtype=np.float64)),
            "timestep_map": _array_raw_sha256(np.ascontiguousarray(timestep_map, dtype=np.int64)),
        },
        "static_inputs_ready": bool(
            probe["exists"] and probe["size_matches"] and probe["sha256_pinned"]
        ),
        "evidence_scores_computed_or_exposed": False,
        "outdir": str(args.outdir),
        "canonical_command": canonical_command(args),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    data_root = Path(os.environ.get("EQVAE_DATA_ROOT", "/home/zhoushunyu/data/eqvae"))
    default_dit = data_root / "baselines/DiT"
    default_vae = (
        Path.home()
        / ".cache/huggingface/hub/models--stabilityai--sd-vae-ft-mse/snapshots"
        / VAE_REVISION
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-index", type=int, choices=range(TOTAL_SHARDS), default=0)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--observe-dir", type=Path, default=None)
    parser.add_argument("--baseline-dir", type=Path, default=None)
    parser.add_argument("--dit-root", type=Path, default=default_dit)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--vae-snapshot", type=Path, default=default_vae)
    parser.add_argument("--outdir", type=Path, default=None)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--self-test", action="store_true")
    modes.add_argument("--validate-bundle", type=Path, default=None)
    return parser


def normalize_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    args.protocol = args.protocol.expanduser().absolute().resolve()
    if not args.protocol.is_file():
        parser.error(f"protocol JSON is missing: {args.protocol}")
    if args.self_test:
        return
    if args.validate_bundle is not None:
        args.validate_bundle = args.validate_bundle.expanduser().absolute().resolve()
        if not args.validate_bundle.is_dir():
            parser.error(f"validation bundle is not a directory: {args.validate_bundle}")
        return
    if args.device_index < 0:
        parser.error("--device-index must be nonnegative")
    data_root = Path(os.environ.get("EQVAE_DATA_ROOT", "/home/zhoushunyu/data/eqvae")).expanduser().absolute().resolve()
    args.dit_root = args.dit_root.expanduser().absolute().resolve()
    args.checkpoint = (
        args.dit_root / "pretrained_models" / CHECKPOINT_FILENAME
        if args.checkpoint is None
        else args.checkpoint.expanduser().absolute().resolve()
    )
    args.vae_snapshot = args.vae_snapshot.expanduser().absolute().resolve()
    args.baseline_dir = (
        data_root / "cross_scale_evidence/dit_imagenet256" / "official_demo_seed2"
        if args.baseline_dir is None
        else args.baseline_dir.expanduser().absolute().resolve()
    )
    if args.observe_dir is None:
        observer_sha = sha256_file(
            RUNNER_DIR / "observe_dit_imagenet256_path_evidence.py"
        )[:7]
        args.observe_dir = (
            data_root
            / "cross_scale_evidence/dit_imagenet256_path_evidence"
            / f"official_seed2_dnu1_K0p5_grid4_{observer_sha}"
        )
    else:
        args.observe_dir = args.observe_dir.expanduser().absolute().resolve()
    runner_sha = sha256_file(Path(__file__).resolve())[:7]
    default_out = (
        data_root
        / "cross_scale_evidence/dit_imagenet256_t60_within_prefix_validation"
        / f"seed2_class0207_t60_poolv1_shard{args.shard_index:02d}of04_{runner_sha}"
    )
    requested = (
        default_out.expanduser().absolute()
        if args.outdir is None
        else args.outdir.expanduser().absolute()
    )
    if os.path.lexists(requested):
        parser.error(f"no-overwrite target already exists: {requested}")
    args.outdir = requested.resolve()
    protected = (
        args.protocol,
        args.observe_dir,
        args.baseline_dir,
        args.dit_root,
        args.vae_snapshot,
        RUNNER_DIR.parent,
    )
    for path in protected:
        if _paths_overlap(args.outdir, path):
            parser.error(f"--outdir overlaps protected input/source path: {path}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    normalize_args(args, parser)
    if args.self_test:
        run_self_test(args.protocol)
        return 0
    if args.validate_bundle is not None:
        validate_output_bundle(args.validate_bundle)
        print(json.dumps({"status": "valid", "bundle": str(args.validate_bundle)}, indent=2))
        return 0
    protocol = _load_protocol(args.protocol)
    if not args.dry_run and protocol.get("protocol_status") != "FROZEN_BEFORE_GPU_EXECUTION":
        raise RuntimeError(
            "protocol is still DRAFT_FOR_REVIEW_NO_GPU_EXECUTION; real GPU sampling is "
            "intentionally locked until the reviewed protocol is frozen and re-hashed"
        )
    source = validate_repository(args.dit_root, args.checkpoint)
    vae = validate_vae_snapshot(args.vae_snapshot)
    observed_args = SimpleNamespace(
        observe_dir=args.observe_dir,
        baseline_dir=args.baseline_dir,
        seed=PREFIX_SEED,
        dit_root=args.dit_root,
    )
    checkpoint = None if args.dry_run else validate_checkpoint(args.checkpoint)
    observed = load_observed_input(
        observed_args,
        source=source,
        checkpoint=checkpoint,
        vae=vae,
    )
    alpha, timestep_map = load_schedule(args.dit_root)
    if not np.array_equal(timestep_map, observed.timestep_map):
        raise RuntimeError("validated schedule differs from observer")
    _validate_frozen_input_binding(protocol, observed, alpha, timestep_map)
    if args.dry_run:
        dry_run(
            args,
            protocol=protocol,
            observed=observed,
            source=source,
            vae=vae,
            alpha=alpha,
            timestep_map=timestep_map,
        )
    else:
        if checkpoint is None:
            raise AssertionError("real validation sampling requires a validated checkpoint")
        run_real(
            args,
            protocol=protocol,
            observed=observed,
            source=source,
            checkpoint=checkpoint,
            vae=vae,
            alpha=alpha,
            timestep_map=timestep_map,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
