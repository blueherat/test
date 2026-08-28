#!/usr/bin/env python3
"""Replay cross-scale diagnostics on one completed fresh-branch DiT suffix bundle.

This is a posthoc DISCOVERY diagnostic, not an online detector, intervention,
or paper-level validation.  It never changes the frozen sampler and never
overwrites an existing artifact.  The input suffix bundle is validated through
the original suffix runner before any GPU work starts.

For each saved fresh target branch (attempts 1..4), internal transition ``k``
has saved state ``x_k``, current prediction ``x0_hat_k``, implemented P
standard deviation ``sigma_k``, and the actually used standard-normal
innovation ``z_{k+1}``.  Current epsilon is reconstructed without a model call:

    eps_k = (x_k - sqrt(alpha_bar_k) * x0_hat_k) / sqrt(1-alpha_bar_k).

For each predeclared additive normalized-heat shift Delta-nu, the frozen DiT is
evaluated at the nearest higher-noise internal scale using

    nu=(1-alpha_bar)/alpha_bar,  rho=sqrt(alpha_plus/alpha),
    x_plus=rho*x_k.

The pulled-back cross-scale direction is

    theta = -rho*eps_plus/sqrt(1-alpha_plus)
            +eps_k/sqrt(1-alpha).

Seventeen alternatives are fixed independently of image content: one global
latent mask and a row-major 4x4 partition.  With a deterministic per-step KL
cap, their same-covariance Gaussian mean shifts are

    delta = sigma_k**2 * gamma * mask * theta,
    u = delta/sigma_k = sigma_k * gamma * mask * theta,

and the exact operational log likelihood-ratio increment is

    log(Q/P) = <u,z_{k+1}> - 0.5*||u||**2.

Critically, ``u`` is fully constructed from x_k and saved P quantities before
the saved innovation is read by the LR evaluator.  Replaying after the fact
does not make this an online experiment, but it does preserve predictability
of the audited Q/P construction.  At t=0, and whenever the discrete scale map
is the identity, Q is set equal to P and the LR increment is exactly zero.

DiT predicts learned variance, but shifted learned-variance channels are not
used.  The operational Q deliberately shares the *saved implemented P*
covariance.  CFG is evaluated with the released ``forward_with_cfg`` method:
the requested ImageNet class is paired with the null class, scale 4.0 is used,
and only epsilon channels 0..2 are guided by the upstream implementation.

No image-dependent ROI, tail box, endpoint label, branch ranking, threshold
tuning, or selection is accepted by this script.  Outputs are descriptive
traces for hypothesis discovery and must be followed by a frozen held-out run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import socket
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

sys.dont_write_bytecode = True

import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

try:
    from .adm64_path_evidence import nearest_additive_heat_shift
    from .intervene_dit_imagenet256_suffix import (
        FROZEN_FRESH_ATTEMPT_COUNT,
        OBSERVER_TRACE_NAME,
        _atomic_install_directory_noreplace,
        _load_npz_exact,
        _read_self_hashed_json,
        _with_upstream_imports,
        branch_id,
        load_observed_input,
        validate_bundle as validate_suffix_bundle,
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
        LATENT_CHANNELS,
        LATENT_SIZE,
        MODEL_NAME,
        NULL_CLASS_ID,
        NUM_CLASSES,
        NUM_SAMPLING_STEPS,
        atomic_json_dump,
        checkpoint_dry_probe,
        dependency_identity,
        ensure_single_process,
        sha256_file,
        sha256_json,
        validate_checkpoint,
        validate_repository,
        validate_vae_snapshot,
    )
except ImportError:  # pragma: no cover - direct CLI execution.
    from adm64_path_evidence import nearest_additive_heat_shift
    from intervene_dit_imagenet256_suffix import (
        FROZEN_FRESH_ATTEMPT_COUNT,
        OBSERVER_TRACE_NAME,
        _atomic_install_directory_noreplace,
        _load_npz_exact,
        _read_self_hashed_json,
        _with_upstream_imports,
        branch_id,
        load_observed_input,
        validate_bundle as validate_suffix_bundle,
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
        LATENT_CHANNELS,
        LATENT_SIZE,
        MODEL_NAME,
        NULL_CLASS_ID,
        NUM_CLASSES,
        NUM_SAMPLING_STEPS,
        atomic_json_dump,
        checkpoint_dry_probe,
        dependency_identity,
        ensure_single_process,
        sha256_file,
        sha256_json,
        validate_checkpoint,
        validate_repository,
        validate_vae_snapshot,
    )


EXPERIMENT = "dit_imagenet256_fresh_suffix_cross_scale_replay_discovery"
SCHEMA_VERSION = 2
DEFAULT_DELTA_NU = (0.25, 1.0, 4.0)
DEFAULT_TOTAL_K_PER_COMPONENT = 0.5
GRID_SIZE = 4
LOCAL_COMPONENT_COUNT = GRID_SIZE * GRID_SIZE
COMPONENT_COUNT = 1 + LOCAL_COMPONENT_COUNT
TRACE_NAME = "cross_scale_replay.npz"


def _canonical_self_hash(payload: dict[str, Any], key: str) -> str:
    stripped = dict(payload)
    stripped.pop(key, None)
    return sha256_json(stripped)


def _array_raw_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _logmeanexp(values: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    maximum = np.max(values, axis=axis, keepdims=True)
    result = maximum + np.log(np.mean(np.exp(values - maximum), axis=axis, keepdims=True))
    axes = (axis,) if isinstance(axis, int) else axis
    for item in sorted((value % values.ndim for value in axes), reverse=True):
        result = np.squeeze(result, axis=item)
    return np.asarray(result, dtype=np.float64)


def _component_masks() -> tuple[np.ndarray, np.ndarray]:
    bounds = fixed_tile_bounds(grid_size=GRID_SIZE, height=LATENT_SIZE, width=LATENT_SIZE)
    local = fixed_tile_masks(bounds, height=LATENT_SIZE, width=LATENT_SIZE)
    global_mask = np.ones((1, 1, LATENT_SIZE, LATENT_SIZE), dtype=np.float64)
    return np.ascontiguousarray(np.concatenate([global_mask, local], axis=0)), bounds


def reconstruct_current_epsilon(
    state: np.ndarray, pred_xstart: np.ndarray, alpha_bar: float
) -> np.ndarray:
    """Reconstruct the four epsilon channels used by the implemented P mean."""

    if state.shape != pred_xstart.shape or state.ndim != 4:
        raise ValueError("state and pred_xstart must match [branch,C,H,W]")
    if not 0.0 < alpha_bar < 1.0:
        raise ValueError("alpha_bar must lie strictly between zero and one")
    epsilon = (
        state.astype(np.float64, copy=False)
        - math.sqrt(alpha_bar) * pred_xstart.astype(np.float64, copy=False)
    ) / math.sqrt(1.0 - alpha_bar)
    if not np.isfinite(epsilon).all():
        raise ValueError("reconstructed epsilon is non-finite")
    return np.ascontiguousarray(epsilon, dtype=np.float32)


def epsilon_reconstruction_control(
    observed: Any, alpha: np.ndarray, *, rollback: int, target_index: int
) -> dict[str, float]:
    """Compare float32 x/x0 reconstruction against captured baseline epsilon."""

    rows = {int(value): index for index, value in enumerate(observed.arrays["internal_timestep"])}
    maximum_absolute_error = 0.0
    maximum_rms_error = 0.0
    maximum_relative_rms_error = 0.0
    for internal_t in range(rollback, -1, -1):
        row = rows[internal_t]
        reconstructed = reconstruct_current_epsilon(
            observed.arrays["x_t"][row, target_index][None],
            observed.arrays["pred_xstart"][row, target_index][None],
            float(alpha[internal_t]),
        )[0].astype(np.float64)
        captured = observed.arrays["epsilon_current"][row, target_index].astype(np.float64)
        difference = reconstructed - captured
        absolute = float(np.max(np.abs(difference), initial=0.0))
        rms = float(np.sqrt(np.mean(np.square(difference), dtype=np.float64)))
        reference_rms = float(np.sqrt(np.mean(np.square(captured), dtype=np.float64)))
        relative = rms / max(reference_rms, np.finfo(np.float64).tiny)
        maximum_absolute_error = max(maximum_absolute_error, absolute)
        maximum_rms_error = max(maximum_rms_error, rms)
        maximum_relative_rms_error = max(maximum_relative_rms_error, relative)
    if maximum_absolute_error > 1e-4:
        raise RuntimeError(
            "saved x_t/pred_xstart epsilon reconstruction is unexpectedly inaccurate: "
            f"max_abs={maximum_absolute_error}"
        )
    return {
        "maximum_absolute_error": maximum_absolute_error,
        "maximum_rms_error": maximum_rms_error,
        "maximum_relative_rms_error": maximum_relative_rms_error,
        "fail_closed_maximum_absolute_tolerance": 1e-4,
    }


def construct_components_before_innovation(
    theta: np.ndarray,
    p_sigma: np.ndarray,
    masks: np.ndarray,
    per_step_K_cap: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct predictable whitened shifts; deliberately has no innovation input."""

    if theta.ndim != 4 or theta.shape != p_sigma.shape:
        raise ValueError("theta and p_sigma must match [branch,C,H,W]")
    if theta.dtype != np.float64 or p_sigma.dtype != np.float32:
        raise TypeError("theta must be float64 and saved P sigma must be float32")
    if masks.shape != (COMPONENT_COUNT, 1, theta.shape[2], theta.shape[3]):
        raise ValueError("fixed global+tile mask contract changed")
    if not np.isfinite(theta).all() or not np.isfinite(p_sigma).all():
        raise ValueError("non-finite Q construction input")
    if np.any(p_sigma <= 0.0):
        raise ValueError("stochastic P sigma must be strictly positive")
    if not math.isfinite(per_step_K_cap) or per_step_K_cap <= 0.0:
        raise ValueError("per-step K cap must be finite and positive")

    raw_u = np.ascontiguousarray(
        p_sigma.astype(np.float64, copy=False)[:, None]
        * theta[:, None]
        * masks[None]
    )
    raw_K = 0.5 * np.sum(np.square(raw_u), axis=(2, 3, 4), dtype=np.float64)
    scale = np.ones_like(raw_K)
    positive = raw_K > 0.0
    scale[positive] = np.minimum(1.0, np.sqrt(per_step_K_cap / raw_K[positive]))
    u = np.ascontiguousarray(raw_u * scale[:, :, None, None, None])
    K = 0.5 * np.sum(np.square(u), axis=(2, 3, 4), dtype=np.float64)
    if np.any(K > per_step_K_cap * (1.0 + 2e-12)):
        raise AssertionError("component exceeded deterministic per-step K cap")
    return raw_K, scale, K, u


def evaluate_after_innovation(
    whitened_shift: np.ndarray, innovation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate an already-constructed Q/P shift on the subsequent innovation."""

    if whitened_shift.ndim != 5 or innovation.shape != (
        whitened_shift.shape[0],
        whitened_shift.shape[2],
        whitened_shift.shape[3],
        whitened_shift.shape[4],
    ):
        raise ValueError("innovation shape does not match constructed shifts")
    if whitened_shift.dtype != np.float64 or innovation.dtype != np.float32:
        raise TypeError("LR evaluator requires float64 shifts and float32 innovation")
    R = np.sum(
        whitened_shift * innovation.astype(np.float64, copy=False)[:, None],
        axis=(2, 3, 4),
        dtype=np.float64,
    )
    K = 0.5 * np.sum(np.square(whitened_shift), axis=(2, 3, 4), dtype=np.float64)
    return np.ascontiguousarray(R), np.ascontiguousarray(R - K)


def _schedule_maps(
    alpha: np.ndarray, rollback: int, delta_nu: Sequence[float]
) -> dict[str, np.ndarray]:
    internal = np.arange(rollback, -1, -1, dtype=np.int64)
    stochastic = internal > 0
    shifted = np.empty((len(delta_nu), len(internal)), dtype=np.int64)
    target_nu = np.empty_like(shifted, dtype=np.float64)
    actual_shift = np.empty_like(shifted, dtype=np.float64)
    mapping_error = np.empty_like(shifted, dtype=np.float64)
    for scale_index, shift in enumerate(delta_nu):
        # Map stochastic steps only.  At t=0 the sampler's transition-noise
        # multiplier is zero (even though its learned sigma tensor is positive),
        # so the implemented transition is deterministic and Q=P.
        mapped = nearest_additive_heat_shift(alpha, internal[stochastic], shift)
        shifted[scale_index, stochastic] = mapped.shifted_timestep
        shifted[scale_index, ~stochastic] = 0
        target_nu[scale_index, stochastic] = mapped.target_heat_variance
        target_nu[scale_index, ~stochastic] = (1.0 - alpha[0]) / alpha[0]
        actual_shift[scale_index, stochastic] = mapped.actual_heat_shift
        actual_shift[scale_index, ~stochastic] = 0.0
        mapping_error[scale_index, stochastic] = mapped.absolute_mapping_error
        mapping_error[scale_index, ~stochastic] = 0.0
    return {
        "internal_timestep": internal,
        "shifted_internal_timestep": shifted,
        "target_heat_variance": target_nu,
        "actual_heat_shift": actual_shift,
        "mapping_absolute_error": mapping_error,
    }


def _load_and_validate_inputs(
    args: argparse.Namespace, *, full_checkpoint: bool
) -> tuple[
    Any,
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, np.ndarray]],
    np.ndarray,
    np.ndarray,
    dict[str, Any],
    dict[str, Any] | None,
    dict[str, Any],
]:
    manifest = _read_self_hashed_json(args.suffix_dir / "manifest.json", "identity_sha256")
    seed = int(manifest.get("seed", -1))
    target = int(manifest.get("target", {}).get("batch_index", -1))
    rollback = int(
        manifest.get("frozen_screen_protocol", {}).get(
            "this_invocation_rollback_internal_timestep", -1
        )
    )
    if seed < 0 or target not in range(BATCH_SIZE) or rollback not in range(NUM_SAMPLING_STEPS):
        raise RuntimeError("suffix manifest lacks a valid seed/target/rollback identity")
    if target != args.target_batch_index:
        raise RuntimeError("--target-batch-index disagrees with the suffix manifest")

    source = validate_repository(args.dit_root, args.checkpoint)
    vae = validate_vae_snapshot(args.vae_snapshot)
    checkpoint = validate_checkpoint(args.checkpoint) if full_checkpoint else None
    observed_args = SimpleNamespace(
        observe_dir=args.observe_dir,
        baseline_dir=args.baseline_dir,
        seed=seed,
        dit_root=args.dit_root,
    )
    observed = load_observed_input(
        observed_args, source=source, checkpoint=checkpoint, vae=vae
    )
    results = validate_suffix_bundle(
        args.suffix_dir,
        manifest=manifest,
        observed=observed,
        rollback_internal_timestep=rollback,
        target_batch_index=target,
        require_completion=True,
    )

    branch_records = [item for item in results["branches"] if int(item["attempt_index"]) > 0]
    if [int(item["attempt_index"]) for item in branch_records] != list(
        range(1, FROZEN_FRESH_ATTEMPT_COUNT + 1)
    ):
        raise RuntimeError("fresh branch set/order changed")
    branch_arrays = []
    for record in branch_records:
        path = args.suffix_dir / "branches" / str(record["branch_id"]) / "trace.npz"
        branch_arrays.append(_load_npz_exact(path, record["trace"], args.suffix_dir))

    alpha, timestep_map = load_schedule(args.dit_root)
    if not np.array_equal(timestep_map, observed.timestep_map):
        raise RuntimeError("runtime schedule differs from the observer-bound schedule")
    return (
        observed,
        manifest,
        branch_records,
        branch_arrays,
        alpha,
        timestep_map,
        source,
        checkpoint,
        vae,
    )


def _trace_record(path: Path, arrays: dict[str, np.ndarray], root: Path) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "keys": sorted(arrays),
        "arrays": {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "raw_bytes_sha256": _array_raw_sha256(value),
            }
            for key, value in sorted(arrays.items())
        },
    }


def _atomic_npz_dump(arrays: dict[str, np.ndarray], path: Path) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite trace: {path}")
    if path.parent.exists() and not path.parent.is_dir():
        raise RuntimeError(f"trace parent is not a directory: {path.parent}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def replay(
    args: argparse.Namespace,
    *,
    observed: Any,
    suffix_manifest: dict[str, Any],
    branch_records: list[dict[str, Any]],
    branch_arrays: list[dict[str, np.ndarray]],
    alpha: np.ndarray,
    timestep_map: np.ndarray,
    staging: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for shifted DiT replay")
    ensure_single_process()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["DIFFUSERS_OFFLINE"] = "1"
    rollback = int(
        suffix_manifest["frozen_screen_protocol"]["this_invocation_rollback_internal_timestep"]
    )
    target_index = int(suffix_manifest["target"]["batch_index"])
    schedule = _schedule_maps(alpha, rollback, args.delta_nu)
    internal = schedule["internal_timestep"]
    shifted = schedule["shifted_internal_timestep"]
    attempt_count = len(branch_arrays)
    scale_count = len(args.delta_nu)
    step_count = len(internal)
    masks, tile_bounds = _component_masks()

    state = np.ascontiguousarray(
        np.stack([item["target_state_before"] for item in branch_arrays]), dtype=np.float32
    )
    pred_xstart = np.ascontiguousarray(
        np.stack([item["target_pred_xstart"] for item in branch_arrays]), dtype=np.float32
    )
    p_sigma = np.ascontiguousarray(
        np.stack([item["target_p_standard_deviation"] for item in branch_arrays]), dtype=np.float32
    )
    innovations = np.ascontiguousarray(
        np.stack(
            [
                item["fresh_full_proposal"][:, target_index]
                for item in branch_arrays
            ]
        ),
        dtype=np.float32,
    )
    expected = (attempt_count, step_count, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE)
    for name, array in (
        ("state", state),
        ("pred_xstart", pred_xstart),
        ("p_sigma", p_sigma),
        ("innovations", innovations),
    ):
        if array.shape != expected or not np.isfinite(array).all():
            raise RuntimeError(f"invalid stacked {name} trace: {array.shape}")

    epsilon_current = np.empty_like(state)
    epsilon_shifted = np.empty((scale_count,) + expected, dtype=np.float32)
    theta = np.zeros((scale_count,) + expected, dtype=np.float64)
    raw_K = np.zeros((scale_count, attempt_count, step_count, COMPONENT_COUNT), dtype=np.float64)
    component_scale = np.ones_like(raw_K)
    K = np.zeros_like(raw_K)
    R = np.zeros_like(raw_K)
    L = np.zeros_like(raw_K)
    theta_rms = np.zeros((scale_count, attempt_count, step_count), dtype=np.float64)
    theta_tile_rms = np.zeros(
        (scale_count, attempt_count, step_count, LOCAL_COMPONENT_COUNT), dtype=np.float64
    )
    shifted_forward_calls = 0

    effective_count = np.count_nonzero(
        (shifted != internal[None]) & (internal[None] > 0), axis=1
    )
    if np.any(effective_count <= 0):
        raise RuntimeError("a requested Delta-nu has no effective stochastic suffix step")
    per_step_cap = args.total_K_per_component / effective_count.astype(np.float64)
    per_step_cap *= 1.0 - 2e-12
    reconstruction_control = epsilon_reconstruction_control(
        observed, alpha, rollback=rollback, target_index=target_index
    )

    def _execute() -> None:
        nonlocal shifted_forward_calls
        from download import find_model
        from models import DiT_models

        imported_models = Path(sys.modules["models"].__file__).resolve()
        imported_download = Path(sys.modules["download"].__file__).resolve()
        if imported_models != (args.dit_root / "models.py").resolve() or imported_download != (
            args.dit_root / "download.py"
        ).resolve():
            raise RuntimeError("upstream import shadowing detected")

        device = torch.device("cuda")
        model = DiT_models[MODEL_NAME](input_size=LATENT_SIZE, num_classes=NUM_CLASSES).to(device)
        model.load_state_dict(find_model(str(args.checkpoint)))
        model.eval()
        if attempt_count > BATCH_SIZE:
            raise RuntimeError("fresh branch batch exceeds the released B=8 CFG half-batch")
        # Preserve the released 8 -> 16 CFG tensor-shape contract.  The first
        # four slots are the four fresh branches; deterministic repeated pads
        # fill the unused first-half slots and are discarded after the call.
        # DiT has no cross-example normalization/attention, so pads cannot
        # change another example's mathematical output, while retaining the
        # official full-batch GEMM shape avoids a needless batch-shape change.
        y_cond = torch.full(
            (BATCH_SIZE,), int(CLASS_IDS[target_index]), dtype=torch.long, device=device
        )
        y_null = torch.full((BATCH_SIZE,), NULL_CLASS_ID, dtype=torch.long, device=device)
        y = torch.cat([y_cond, y_null], dim=0)

        prior_grad = torch.is_grad_enabled()
        torch.set_grad_enabled(False)
        try:
            for step_index, internal_t in enumerate(internal.tolist()):
                alpha_current = float(alpha[internal_t])
                current = reconstruct_current_epsilon(
                    state[:, step_index], pred_xstart[:, step_index], alpha_current
                )
                epsilon_current[:, step_index] = current
                x_current = torch.from_numpy(state[:, step_index]).to(device=device)

                for scale_index in range(scale_count):
                    shifted_t = int(shifted[scale_index, step_index])
                    effective = internal_t > 0 and shifted_t != internal_t
                    if not effective:
                        epsilon_shifted[scale_index, :, step_index] = current
                        continue
                    alpha_shifted = float(alpha[shifted_t])
                    rho = math.sqrt(alpha_shifted / alpha_current)
                    shifted_x_fresh = x_current * rho
                    padding = shifted_x_fresh[
                        torch.arange(BATCH_SIZE - attempt_count, device=device) % attempt_count
                    ]
                    shifted_x_half = torch.cat([shifted_x_fresh, padding], dim=0)
                    shifted_x = torch.cat([shifted_x_half, shifted_x_half], dim=0)
                    original_t = int(timestep_map[shifted_t])
                    t_tensor = torch.full(
                        (FULL_BATCH_SIZE,), original_t, dtype=torch.long, device=device
                    )
                    rng_before = torch.cuda.get_rng_state().clone()
                    output = model.forward_with_cfg(
                        shifted_x, t_tensor, y=y, cfg_scale=CFG_SCALE
                    )
                    if not torch.equal(rng_before, torch.cuda.get_rng_state()):
                        raise RuntimeError("shifted DiT forward unexpectedly consumed CUDA RNG")
                    expected_output = (
                        FULL_BATCH_SIZE,
                        2 * LATENT_CHANNELS,
                        LATENT_SIZE,
                        LATENT_SIZE,
                    )
                    if tuple(output.shape) != expected_output:
                        raise RuntimeError(f"shifted DiT output shape changed: {tuple(output.shape)}")
                    shifted_eps = np.ascontiguousarray(
                        output[:attempt_count, :LATENT_CHANNELS].cpu().numpy(), dtype=np.float32
                    )
                    epsilon_shifted[scale_index, :, step_index] = shifted_eps
                    shifted_forward_calls += 1

                    direction = (
                        -rho
                        * shifted_eps.astype(np.float64, copy=False)
                        / math.sqrt(1.0 - alpha_shifted)
                        + current.astype(np.float64, copy=False)
                        / math.sqrt(1.0 - alpha_current)
                    )
                    direction = np.ascontiguousarray(direction, dtype=np.float64)
                    theta[scale_index, :, step_index] = direction
                    theta_rms[scale_index, :, step_index] = np.sqrt(
                        np.mean(np.square(direction), axis=(1, 2, 3), dtype=np.float64)
                    )
                    local_sq = np.square(direction[:, None]) * masks[None, 1:]
                    pixels_per_local_component = (
                        LATENT_CHANNELS * LATENT_SIZE * LATENT_SIZE / LOCAL_COMPONENT_COUNT
                    )
                    theta_tile_rms[scale_index, :, step_index] = np.sqrt(
                        np.sum(local_sq, axis=(2, 3, 4), dtype=np.float64)
                        / pixels_per_local_component
                    )

                    # Predictability boundary: construct_components_before_innovation
                    # has no access to z.  Only after it returns do we retrieve and
                    # pass this transition's saved innovation to the LR evaluator.
                    raw, scale, applied_K, u = construct_components_before_innovation(
                        direction,
                        p_sigma[:, step_index],
                        masks,
                        float(per_step_cap[scale_index]),
                    )
                    innovation_after_Q = innovations[:, step_index]
                    reward, increment = evaluate_after_innovation(u, innovation_after_Q)
                    raw_K[scale_index, :, step_index] = raw
                    component_scale[scale_index, :, step_index] = scale
                    K[scale_index, :, step_index] = applied_K
                    R[scale_index, :, step_index] = reward
                    L[scale_index, :, step_index] = increment
                if step_index % 20 == 0 or step_index + 1 == step_count:
                    print(f"replayed {step_index + 1}/{step_count} suffix states", flush=True)
            torch.cuda.synchronize()
        finally:
            torch.set_grad_enabled(prior_grad)

    _with_upstream_imports(args.dit_root, _execute)

    # Axis order for evidence arrays is [scale,branch,step,component].
    cumulative = np.cumsum(L, axis=2, dtype=np.float64)
    per_scale_mixture_log_e = _logmeanexp(cumulative, axis=3)
    all_component_mixture_log_e = _logmeanexp(cumulative, axis=(0, 3))
    component_names = np.asarray(
        ["global", *(f"tile_{index:02d}" for index in range(LOCAL_COMPONENT_COUNT))],
        dtype="<U16",
    )
    arrays = {
        "attempt_index": np.asarray([item["attempt_index"] for item in branch_records], dtype=np.int16),
        "delta_nu": np.asarray(args.delta_nu, dtype=np.float64),
        "full_internal_alpha_bar": np.ascontiguousarray(alpha, dtype=np.float64),
        "full_original_timestep_map": np.ascontiguousarray(timestep_map, dtype=np.int16),
        "internal_timestep": np.ascontiguousarray(internal, dtype=np.int16),
        "original_timestep": np.ascontiguousarray(timestep_map[internal], dtype=np.int16),
        "current_alpha_bar": np.ascontiguousarray(alpha[internal], dtype=np.float64),
        "current_heat_variance": np.ascontiguousarray((1.0 - alpha[internal]) / alpha[internal], dtype=np.float64),
        "shifted_internal_timestep": np.ascontiguousarray(shifted, dtype=np.int16),
        "shifted_original_timestep": np.ascontiguousarray(timestep_map[shifted], dtype=np.int16),
        "shifted_alpha_bar": np.ascontiguousarray(alpha[shifted], dtype=np.float64),
        "rho": np.ascontiguousarray(np.sqrt(alpha[shifted] / alpha[internal][None]), dtype=np.float64),
        "effective_nonidentity": np.ascontiguousarray(((shifted != internal[None]) & (internal[None] > 0)).astype(np.uint8)),
        "target_heat_variance": np.ascontiguousarray(schedule["target_heat_variance"], dtype=np.float64),
        "actual_heat_shift": np.ascontiguousarray(schedule["actual_heat_shift"], dtype=np.float64),
        "mapping_absolute_error": np.ascontiguousarray(schedule["mapping_absolute_error"], dtype=np.float64),
        "component_name": component_names,
        "tile_bounds_yxyx": np.ascontiguousarray(tile_bounds, dtype=np.int16),
        "per_step_K_cap": np.ascontiguousarray(per_step_cap, dtype=np.float64),
        # Preserve the exact float arrays consumed by the replay.  This lets a
        # completed bundle reconstruct epsilon, theta, KL and reward without
        # trusting its own summary metadata or reopening mutable source files.
        "saved_state_before": state,
        "saved_pred_xstart": pred_xstart,
        "saved_p_standard_deviation": p_sigma,
        "saved_transition_innovation": innovations,
        "epsilon_current_reconstructed": epsilon_current,
        "epsilon_shifted": epsilon_shifted,
        "theta": theta,
        "theta_global_rms": theta_rms,
        "theta_tile_rms": theta_tile_rms,
        "raw_K_component": raw_K,
        "component_scale": component_scale,
        "K_component": K,
        "R_component": R,
        "L_component": L,
        "component_log_e": cumulative,
        "per_scale_mixture_log_e": per_scale_mixture_log_e,
        "all_component_mixture_log_e": all_component_mixture_log_e,
    }
    for key, value in arrays.items():
        if value.dtype.kind not in "US" and not np.isfinite(value).all():
            raise RuntimeError(f"non-finite replay output: {key}")

    trace_path = staging / TRACE_NAME
    _atomic_npz_dump(arrays, trace_path)
    execution = {
        "shifted_forward_calls": shifted_forward_calls,
        "current_forward_calls": 0,
        "fresh_branch_count": attempt_count,
        "suffix_step_count_including_t0": step_count,
        "requested_delta_nu_count": scale_count,
        "naive_unbatched_shifted_forward_equivalents": shifted_forward_calls * attempt_count,
        "batched_over_fresh_branches": True,
        "peak_model_input_batch": FULL_BATCH_SIZE,
        "released_CFG_batch_shape_preserved": True,
        "fresh_branch_padding_slots_discarded": BATCH_SIZE - attempt_count,
        "current_epsilon_reconstruction_baseline_control": reconstruction_control,
    }
    return arrays, execution


def _manifest(
    args: argparse.Namespace,
    *,
    suffix_manifest: dict[str, Any],
    branch_records: list[dict[str, Any]],
    source: dict[str, Any],
    checkpoint: dict[str, Any],
    vae: dict[str, Any],
) -> dict[str, Any]:
    runner = Path(__file__).resolve()
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "role": "POSTHOC_DISCOVERY_ONLY_CROSS_SCALE_REPLAY",
        "online_sampling_method": False,
        "intervention_performed": False,
        "quality_claim_eligible": False,
        "held_out_validation": False,
        "image_or_label_dependent_ROI_used": False,
        "automatic_branch_ranking_or_selection": False,
        "delta_nu": list(args.delta_nu),
        "total_K_per_fixed_component": args.total_K_per_component,
        "component_family": "uniform fixed mixture over each Delta-nu x (global + row-major 4x4 latent tiles)",
        "fresh_attempt_indices": [int(item["attempt_index"]) for item in branch_records],
        "predictability_contract": (
            "Q shift uses only saved x_t/current P prediction/current P sigma and shifted DiT output; "
            "the saved transition innovation is passed only after Q construction"
        ),
        "evidence_indexing": (
            "trace row with internal timestep t stores the increment for transition t->t-1; "
            "component_log_e at that row is post-transition evidence"
        ),
        "operational_Q": {
            "same_covariance_as_saved_implemented_P": True,
            "conditional_on_saved_rollback_prefix": True,
            "suffix_likelihood_ratio_initial_value": 1.0,
            "initial_density_ratio_included": False,
            "reason_no_initial_density_ratio": (
                "P and Q start from the same fixed saved rollback state; this is a conditional suffix ratio"
            ),
            "mean_shift": "delta=sigma_P^2*gamma*mask*theta",
            "whitened_shift": "u=delta/sigma_P=sigma_P*gamma*mask*theta",
            "log_lr_increment": "<u,z>-0.5*||u||^2",
            "learned_variance": (
                "P uses saved sigma from the actual DiT learned-variance transition; shifted "
                "learned-variance channels are ignored because Q changes mean only"
            ),
            "ideal_heat_marginal_ratio_claimed": False,
        },
        "time_coordinates": {
            "internal_sampler_index": "250-step respaced DDPM index",
            "original_model_timestep": "strictly increasing map from internal index to [0,999]",
            "normalized_heat_variance": "nu=(1-alpha_bar)/alpha_bar",
            "scale_map": "nearest available higher-noise internal index to nu+Delta-nu; ties go higher-noise",
            "state_pullback": "x_plus=sqrt(alpha_plus/alpha_current)*x_current",
            "t0": "Q=P because the actual transition has zero noise multiplier; LR=0",
            "identity_mapping": "Q=P and LR=0",
        },
        "cfg": {
            "scale": CFG_SCALE,
            "conditional_class_id": int(CLASS_IDS[int(suffix_manifest["target"]["batch_index"])]),
            "null_class_id": NULL_CLASS_ID,
            "upstream_forward_with_cfg_used": True,
            "released_first_half_to_full_batch_shape": [BATCH_SIZE, FULL_BATCH_SIZE],
            "four_fresh_branches_occupy_first_slots": True,
            "remaining_first_half_slots": "deterministic repeated pads discarded after forward",
            "guided_epsilon_channels": [0, 1, 2],
            "fourth_epsilon_channel_retained": True,
        },
        "input_suffix_bundle": {
            "root": str(args.suffix_dir),
            "manifest_identity_sha256": suffix_manifest["identity_sha256"],
            "manifest_file_sha256": sha256_file(args.suffix_dir / "manifest.json"),
            "results_file_sha256": sha256_file(args.suffix_dir / "results.json"),
            "completion_file_sha256": sha256_file(args.suffix_dir / "completion.json"),
            "validated_by_original_suffix_runner": True,
        },
        "input_observer_bundle": {
            "root": str(args.observe_dir),
            "trace_relative_path": OBSERVER_TRACE_NAME,
            "manifest_identity_sha256": suffix_manifest["frozen_observe_bundle"][
                "manifest_identity_sha256"
            ],
            "trace_sha256": suffix_manifest["frozen_observe_bundle"]["trace_sha256"],
        },
        "sources": {"dit": source, "checkpoint": checkpoint, "vae": vae},
        "runner": {"path": str(runner), "sha256": sha256_file(runner)},
        "dependencies": dependency_identity(),
        "outdir": str(args.outdir),
        "outputs": {
            "trace": TRACE_NAME,
            "atomic_no_replace": True,
            "no_overwrite": True,
            "saved_inputs_support_independent_lr_reconstruction": True,
        },
    }
    payload["identity_sha256"] = _canonical_self_hash(payload, "identity_sha256")
    return payload


def _validate_trace_contract_and_math(
    arrays: dict[str, np.ndarray], manifest: dict[str, Any]
) -> None:
    """Reconstruct the full schema and LR math from arrays retained in the artifact."""

    scale_count = len(manifest.get("delta_nu", []))
    attempts = manifest.get("fresh_attempt_indices")
    if scale_count <= 0 or attempts != list(range(1, FROZEN_FRESH_ATTEMPT_COUNT + 1)):
        raise RuntimeError("manifest scale/fresh-attempt contract changed")
    branch_count = len(attempts)
    internal = arrays.get("internal_timestep")
    if internal is None or internal.ndim != 1 or internal.size < 2:
        raise RuntimeError("trace internal timestep axis is invalid")
    step_count = int(internal.size)

    state_shape = (
        branch_count,
        step_count,
        LATENT_CHANNELS,
        LATENT_SIZE,
        LATENT_SIZE,
    )
    shifted_state_shape = (scale_count,) + state_shape
    component_shape = (scale_count, branch_count, step_count, COMPONENT_COUNT)
    contracts: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {
        "attempt_index": ((branch_count,), np.dtype(np.int16)),
        "delta_nu": ((scale_count,), np.dtype(np.float64)),
        "full_internal_alpha_bar": (
            (NUM_SAMPLING_STEPS,),
            np.dtype(np.float64),
        ),
        "full_original_timestep_map": (
            (NUM_SAMPLING_STEPS,),
            np.dtype(np.int16),
        ),
        "internal_timestep": ((step_count,), np.dtype(np.int16)),
        "original_timestep": ((step_count,), np.dtype(np.int16)),
        "current_alpha_bar": ((step_count,), np.dtype(np.float64)),
        "current_heat_variance": ((step_count,), np.dtype(np.float64)),
        "shifted_internal_timestep": ((scale_count, step_count), np.dtype(np.int16)),
        "shifted_original_timestep": ((scale_count, step_count), np.dtype(np.int16)),
        "shifted_alpha_bar": ((scale_count, step_count), np.dtype(np.float64)),
        "rho": ((scale_count, step_count), np.dtype(np.float64)),
        "effective_nonidentity": ((scale_count, step_count), np.dtype(np.uint8)),
        "target_heat_variance": ((scale_count, step_count), np.dtype(np.float64)),
        "actual_heat_shift": ((scale_count, step_count), np.dtype(np.float64)),
        "mapping_absolute_error": ((scale_count, step_count), np.dtype(np.float64)),
        "component_name": ((COMPONENT_COUNT,), np.dtype("<U16")),
        "tile_bounds_yxyx": ((LOCAL_COMPONENT_COUNT, 4), np.dtype(np.int16)),
        "per_step_K_cap": ((scale_count,), np.dtype(np.float64)),
        "saved_state_before": (state_shape, np.dtype(np.float32)),
        "saved_pred_xstart": (state_shape, np.dtype(np.float32)),
        "saved_p_standard_deviation": (state_shape, np.dtype(np.float32)),
        "saved_transition_innovation": (state_shape, np.dtype(np.float32)),
        "epsilon_current_reconstructed": (state_shape, np.dtype(np.float32)),
        "epsilon_shifted": (shifted_state_shape, np.dtype(np.float32)),
        "theta": (shifted_state_shape, np.dtype(np.float64)),
        "theta_global_rms": (
            (scale_count, branch_count, step_count),
            np.dtype(np.float64),
        ),
        "theta_tile_rms": (
            (scale_count, branch_count, step_count, LOCAL_COMPONENT_COUNT),
            np.dtype(np.float64),
        ),
        "raw_K_component": (component_shape, np.dtype(np.float64)),
        "component_scale": (component_shape, np.dtype(np.float64)),
        "K_component": (component_shape, np.dtype(np.float64)),
        "R_component": (component_shape, np.dtype(np.float64)),
        "L_component": (component_shape, np.dtype(np.float64)),
        "component_log_e": (component_shape, np.dtype(np.float64)),
        "per_scale_mixture_log_e": (
            (scale_count, branch_count, step_count),
            np.dtype(np.float64),
        ),
        "all_component_mixture_log_e": (
            (branch_count, step_count),
            np.dtype(np.float64),
        ),
    }
    if set(arrays) != set(contracts):
        missing = sorted(set(contracts) - set(arrays))
        extra = sorted(set(arrays) - set(contracts))
        raise RuntimeError(f"trace schema key mismatch: missing={missing}, extra={extra}")
    for key, (shape, dtype) in contracts.items():
        if arrays[key].shape != shape or arrays[key].dtype != dtype:
            raise RuntimeError(
                f"trace schema contract failed for {key}: "
                f"got {arrays[key].shape}/{arrays[key].dtype}, expected {shape}/{dtype}"
            )

    expected_attempts = np.asarray(attempts, dtype=np.int16)
    expected_delta = np.asarray(manifest["delta_nu"], dtype=np.float64)
    if not np.array_equal(arrays["attempt_index"], expected_attempts):
        raise RuntimeError("trace fresh-attempt identity changed")
    if not np.array_equal(arrays["delta_nu"], expected_delta):
        raise RuntimeError("trace Delta-nu values disagree with the manifest")
    if not np.array_equal(internal, np.arange(int(internal[0]), -1, -1, dtype=np.int16)):
        raise RuntimeError("trace internal timesteps must be a complete descending suffix ending at zero")
    if int(internal[0]) >= NUM_SAMPLING_STEPS:
        raise RuntimeError("trace rollback timestep is outside the frozen sampler schedule")
    full_alpha = arrays["full_internal_alpha_bar"]
    full_timestep_map = arrays["full_original_timestep_map"]
    if (
        np.any(full_alpha <= 0.0)
        or np.any(full_alpha >= 1.0)
        or np.any(np.diff(full_alpha) >= 0.0)
        or full_timestep_map[0] != 0
        or np.any(np.diff(full_timestep_map.astype(np.int64)) <= 0)
        or int(full_timestep_map[-1]) > 999
    ):
        raise RuntimeError("saved full sampler schedule is invalid")
    original = arrays["original_timestep"]
    if (
        original[-1] != 0
        or np.any(np.diff(original.astype(np.int64)) >= 0)
        or not np.array_equal(original, full_timestep_map[internal])
    ):
        raise RuntimeError("trace original timestep map is not strictly descending to zero")

    alpha_current = arrays["current_alpha_bar"]
    alpha_shifted = arrays["shifted_alpha_bar"]
    shifted_internal = arrays["shifted_internal_timestep"]
    shifted_original = arrays["shifted_original_timestep"]
    if (
        np.any(alpha_current <= 0.0)
        or np.any(alpha_current >= 1.0)
        or np.any(alpha_shifted <= 0.0)
        or np.any(alpha_shifted >= 1.0)
        or np.any(shifted_internal < 0)
        or np.any(shifted_internal >= NUM_SAMPLING_STEPS)
        or np.any(shifted_internal < internal[None])
        or np.any(shifted_original < original[None])
        or np.any(alpha_shifted > alpha_current[None])
    ):
        raise RuntimeError("trace scale coordinates violate the higher-noise mapping contract")
    if not np.array_equal(alpha_current, full_alpha[internal]):
        raise RuntimeError("current alpha-bar does not match the saved full schedule")
    if not np.array_equal(alpha_shifted, full_alpha[shifted_internal]):
        raise RuntimeError("shifted alpha-bar does not match the saved full schedule")
    if not np.array_equal(shifted_original, full_timestep_map[shifted_internal]):
        raise RuntimeError("shifted original timestep does not match the saved full schedule")
    remapped = _schedule_maps(full_alpha, int(internal[0]), manifest["delta_nu"])
    remap_keys = {
        "internal_timestep": "internal_timestep",
        "shifted_internal_timestep": "shifted_internal_timestep",
        "target_heat_variance": "target_heat_variance",
        "actual_heat_shift": "actual_heat_shift",
        "mapping_absolute_error": "mapping_absolute_error",
    }
    for trace_key, remap_key in remap_keys.items():
        expected = remapped[remap_key]
        if not np.array_equal(arrays[trace_key], expected):
            raise RuntimeError(f"nearest additive heat-shift mapping changed: {trace_key}")
    expected_current_nu = (1.0 - alpha_current) / alpha_current
    expected_shifted_nu = (1.0 - alpha_shifted) / alpha_shifted
    stochastic = internal[None] > 0
    expected_target_nu = expected_current_nu[None] + expected_delta[:, None]
    expected_target_nu = np.where(stochastic, expected_target_nu, expected_current_nu[None])
    expected_actual_shift = np.where(
        stochastic, expected_shifted_nu - expected_current_nu[None], 0.0
    )
    expected_mapping_error = np.where(
        stochastic, np.abs(expected_shifted_nu - expected_target_nu), 0.0
    )
    expected_effective = ((shifted_internal != internal[None]) & stochastic).astype(np.uint8)
    coordinate_checks = {
        "current_heat_variance": expected_current_nu,
        "target_heat_variance": expected_target_nu,
        "actual_heat_shift": expected_actual_shift,
        "mapping_absolute_error": expected_mapping_error,
        "rho": np.sqrt(alpha_shifted / alpha_current[None]),
    }
    for key, expected in coordinate_checks.items():
        if not np.allclose(arrays[key], expected, rtol=0.0, atol=2e-16):
            raise RuntimeError(f"trace coordinate reconstruction failed: {key}")
    if not np.array_equal(arrays["effective_nonidentity"], expected_effective):
        raise RuntimeError("trace effective scale map changed")

    masks, expected_bounds = _component_masks()
    expected_names = np.asarray(
        ["global", *(f"tile_{index:02d}" for index in range(LOCAL_COMPONENT_COUNT))],
        dtype="<U16",
    )
    if not np.array_equal(arrays["component_name"], expected_names) or not np.array_equal(
        arrays["tile_bounds_yxyx"], expected_bounds
    ):
        raise RuntimeError("fixed global/tile component contract changed")
    effective_count = expected_effective.sum(axis=1, dtype=np.int64)
    expected_cap = (
        float(manifest["total_K_per_fixed_component"])
        / effective_count.astype(np.float64)
        * (1.0 - 2e-12)
    )
    if not np.array_equal(arrays["per_step_K_cap"], expected_cap):
        raise RuntimeError("per-step deterministic KL allocation changed")

    expected_epsilon = np.empty_like(arrays["epsilon_current_reconstructed"])
    for step_index, alpha_value in enumerate(alpha_current.tolist()):
        expected_epsilon[:, step_index] = reconstruct_current_epsilon(
            arrays["saved_state_before"][:, step_index],
            arrays["saved_pred_xstart"][:, step_index],
            float(alpha_value),
        )
    if not np.array_equal(arrays["epsilon_current_reconstructed"], expected_epsilon):
        raise RuntimeError("current epsilon does not reconstruct from saved state/pred-x0")
    if np.any(arrays["saved_p_standard_deviation"] <= 0.0):
        raise RuntimeError("saved implemented P sigma must remain positive")

    expected_theta = np.zeros_like(arrays["theta"])
    expected_raw_K = np.zeros_like(arrays["raw_K_component"])
    expected_scale = np.ones_like(arrays["component_scale"])
    expected_K = np.zeros_like(arrays["K_component"])
    expected_R = np.zeros_like(arrays["R_component"])
    expected_L = np.zeros_like(arrays["L_component"])
    for scale_index in range(scale_count):
        for step_index, current_t in enumerate(internal.tolist()):
            if not bool(expected_effective[scale_index, step_index]):
                if not np.array_equal(
                    arrays["epsilon_shifted"][scale_index, :, step_index],
                    expected_epsilon[:, step_index],
                ):
                    raise RuntimeError("inactive/t0 shifted epsilon must equal current epsilon")
                continue
            rho = math.sqrt(
                float(alpha_shifted[scale_index, step_index])
                / float(alpha_current[step_index])
            )
            direction = (
                -rho
                * arrays["epsilon_shifted"][scale_index, :, step_index].astype(
                    np.float64, copy=False
                )
                / math.sqrt(1.0 - float(alpha_shifted[scale_index, step_index]))
                + expected_epsilon[:, step_index].astype(np.float64, copy=False)
                / math.sqrt(1.0 - float(alpha_current[step_index]))
            )
            expected_theta[scale_index, :, step_index] = direction
            raw, scale, applied_K, whitened = construct_components_before_innovation(
                np.ascontiguousarray(direction, dtype=np.float64),
                arrays["saved_p_standard_deviation"][:, step_index],
                masks,
                float(expected_cap[scale_index]),
            )
            reward, increment = evaluate_after_innovation(
                whitened, arrays["saved_transition_innovation"][:, step_index]
            )
            expected_raw_K[scale_index, :, step_index] = raw
            expected_scale[scale_index, :, step_index] = scale
            expected_K[scale_index, :, step_index] = applied_K
            expected_R[scale_index, :, step_index] = reward
            expected_L[scale_index, :, step_index] = increment
    theta_error = float(np.max(np.abs(arrays["theta"] - expected_theta), initial=0.0))
    if theta_error > 2e-13:
        raise RuntimeError(f"cross-scale theta reconstruction failed: max_abs={theta_error}")
    exact_reconstructions = {
        "raw_K_component": expected_raw_K,
        "component_scale": expected_scale,
        "K_component": expected_K,
        "R_component": expected_R,
        "L_component": expected_L,
    }
    for key, expected in exact_reconstructions.items():
        if not np.array_equal(arrays[key], expected):
            error = float(np.max(np.abs(arrays[key] - expected), initial=0.0))
            raise RuntimeError(f"core LR reconstruction failed for {key}: max_abs={error}")

    expected_global_rms = np.sqrt(
        np.mean(np.square(expected_theta), axis=(3, 4, 5), dtype=np.float64)
    )
    pixels_per_tile = LATENT_CHANNELS * LATENT_SIZE * LATENT_SIZE / LOCAL_COMPONENT_COUNT
    expected_tile_rms = np.zeros_like(arrays["theta_tile_rms"])
    for scale_index in range(scale_count):
        for step_index in range(step_count):
            local_sq = (
                np.square(expected_theta[scale_index, :, step_index, None])
                * masks[None, 1:]
            )
            expected_tile_rms[scale_index, :, step_index] = np.sqrt(
                np.sum(local_sq, axis=(2, 3, 4), dtype=np.float64) / pixels_per_tile
            )
    if not np.allclose(
        arrays["theta_global_rms"], expected_global_rms, rtol=2e-15, atol=2e-15
    ):
        raise RuntimeError("global theta RMS does not reconstruct")
    if not np.allclose(
        arrays["theta_tile_rms"], expected_tile_rms, rtol=2e-15, atol=2e-15
    ):
        raise RuntimeError("tile theta RMS does not reconstruct")


def validate_output_bundle(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fail-closed validation for a completed replay artifact."""

    if root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")):
        raise RuntimeError("replay bundle must not contain symlinks")
    manifest = _read_self_hashed_json(root / "manifest.json", "identity_sha256")
    fixed_manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "role": "POSTHOC_DISCOVERY_ONLY_CROSS_SCALE_REPLAY",
        "online_sampling_method": False,
        "intervention_performed": False,
        "quality_claim_eligible": False,
        "held_out_validation": False,
        "image_or_label_dependent_ROI_used": False,
        "automatic_branch_ranking_or_selection": False,
    }
    if any(manifest.get(key) != value for key, value in fixed_manifest.items()):
        raise RuntimeError("replay manifest scope/identity changed")
    if manifest.get("runner", {}).get("sha256") != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("replay artifact was produced by a different runner source")
    results = _read_self_hashed_json(root / "results.json", "payload_sha256")
    if (
        results.get("schema_version") != SCHEMA_VERSION
        or results.get("experiment") != EXPERIMENT
        or results.get("manifest_identity_sha256") != manifest["identity_sha256"]
        or results.get("discovery_only") is not True
        or results.get("intervention_performed") is not False
        or results.get("image_quality_labels_consumed") is not False
    ):
        raise RuntimeError("replay results scope/identity changed")
    trace_record = results.get("trace", {})
    trace_path = root / TRACE_NAME
    if trace_record.get("relative_path") != TRACE_NAME:
        raise RuntimeError("replay trace relative path changed")
    if (
        not trace_path.is_file()
        or trace_path.stat().st_size != trace_record.get("bytes")
        or sha256_file(trace_path) != trace_record.get("sha256")
    ):
        raise RuntimeError("replay trace file identity failed")
    with np.load(trace_path, allow_pickle=False) as archive:
        arrays = {key: np.ascontiguousarray(archive[key]) for key in archive.files}
    if sorted(arrays) != trace_record.get("keys"):
        raise RuntimeError("replay trace key set changed")
    for key, value in arrays.items():
        expected = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "raw_bytes_sha256": _array_raw_sha256(value),
        }
        if trace_record.get("arrays", {}).get(key) != expected:
            raise RuntimeError(f"replay trace array identity failed: {key}")
        if value.dtype.kind not in "US" and not np.isfinite(value).all():
            raise RuntimeError(f"replay trace contains non-finite values: {key}")
    _validate_trace_contract_and_math(arrays, manifest)
    if not np.array_equal(
        arrays["L_component"], arrays["R_component"] - arrays["K_component"]
    ):
        raise RuntimeError("saved Gaussian LR decomposition does not reconstruct exactly")
    cumulative = np.cumsum(arrays["L_component"], axis=2, dtype=np.float64)
    if not np.array_equal(cumulative, arrays["component_log_e"]):
        raise RuntimeError("saved cumulative component evidence does not reconstruct")
    if not np.array_equal(
        _logmeanexp(cumulative, axis=3), arrays["per_scale_mixture_log_e"]
    ):
        raise RuntimeError("saved per-scale fixed-component mixture does not reconstruct")
    if not np.array_equal(
        _logmeanexp(cumulative, axis=(0, 3)), arrays["all_component_mixture_log_e"]
    ):
        raise RuntimeError("saved scale/component mixture does not reconstruct")
    effective = arrays["effective_nonidentity"].astype(bool)
    inactive = ~effective[:, None, :, None]
    for key in ("K_component", "R_component", "L_component"):
        if np.any(np.where(inactive, arrays[key], 0.0) != 0.0):
            raise RuntimeError(f"inactive/t0 Q must equal P exactly: {key}")
    inactive_state = ~effective[:, None, :, None, None, None]
    if np.any(np.where(inactive_state, arrays["theta"], 0.0) != 0.0):
        raise RuntimeError("inactive/t0 theta must be exactly zero")
    expected_inactive_shifted = np.broadcast_to(
        arrays["epsilon_current_reconstructed"][None], arrays["epsilon_shifted"].shape
    )
    if np.any(
        np.where(
            inactive_state,
            arrays["epsilon_shifted"] - expected_inactive_shifted,
            np.float32(0.0),
        )
        != 0.0
    ):
        raise RuntimeError("inactive/t0 shifted epsilon must equal current epsilon exactly")
    total_K = arrays["K_component"].sum(axis=2, dtype=np.float64)
    total_budget = float(manifest["total_K_per_fixed_component"])
    if np.any(total_K > total_budget * (1.0 + 2e-12)):
        raise RuntimeError("a replay component exceeded its declared suffix KL budget")
    completion = _read_self_hashed_json(root / "completion.json", "payload_sha256")
    fixed_completion = {
        "complete": True,
        "manifest_identity_sha256": manifest["identity_sha256"],
        "manifest_file_sha256": sha256_file(root / "manifest.json"),
        "results_payload_sha256": results["payload_sha256"],
        "results_file_sha256": sha256_file(root / "results.json"),
        "trace_sha256": trace_record["sha256"],
    }
    if any(completion.get(key) != value for key, value in fixed_completion.items()):
        raise RuntimeError("replay completion links/hashes changed")
    expected_files = {
        (root / "manifest.json").resolve(),
        (root / "results.json").resolve(),
        (root / "completion.json").resolve(),
        trace_path.resolve(),
    }
    actual_files = {path.resolve() for path in root.rglob("*") if path.is_file()}
    if actual_files != expected_files:
        raise RuntimeError("replay bundle file set changed")
    return manifest, results


def run_real(args: argparse.Namespace) -> None:
    (
        observed,
        suffix_manifest,
        records,
        branch_arrays,
        alpha,
        timestep_map,
        source,
        checkpoint,
        vae,
    ) = _load_and_validate_inputs(args, full_checkpoint=True)
    if checkpoint is None:
        raise AssertionError("real replay requires a fully validated checkpoint")
    manifest = _manifest(
        args,
        suffix_manifest=suffix_manifest,
        branch_records=records,
        source=source,
        checkpoint=checkpoint,
        vae=vae,
    )
    args.outdir.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with tempfile.TemporaryDirectory(
        prefix=f".{args.outdir.name}.staging-", dir=args.outdir.parent
    ) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        atomic_json_dump(manifest, staging / "manifest.json")
        arrays, execution = replay(
            args,
            observed=observed,
            suffix_manifest=suffix_manifest,
            branch_records=records,
            branch_arrays=branch_arrays,
            alpha=alpha,
            timestep_map=timestep_map,
            staging=staging,
        )
        trace = _trace_record(staging / TRACE_NAME, arrays, staging)
        results: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "manifest_identity_sha256": manifest["identity_sha256"],
            "discovery_only": True,
            "intervention_performed": False,
            "image_quality_labels_consumed": False,
            "trace": trace,
            "execution": execution,
            "per_branch": [],
            "wall_seconds_before_publication": time.time() - started,
            "platform": {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python": sys.version,
                "cuda_device_name": torch.cuda.get_device_name(torch.cuda.current_device()),
            },
        }
        for branch_pos, record in enumerate(records):
            results["per_branch"].append(
                {
                    "branch_id": record["branch_id"],
                    "attempt_index": int(record["attempt_index"]),
                    "final_all_component_mixture_log_e": float(
                        arrays["all_component_mixture_log_e"][branch_pos, -1]
                    ),
                    "running_max_all_component_mixture_log_e": float(
                        max(0.0, arrays["all_component_mixture_log_e"][branch_pos].max())
                    ),
                    "final_per_scale_mixture_log_e": arrays[
                        "per_scale_mixture_log_e"
                    ][:, branch_pos, -1].tolist(),
                    "maximum_total_K_any_component": float(
                        arrays["K_component"][:, branch_pos].sum(axis=1).max()
                    ),
                }
            )
        results["payload_sha256"] = _canonical_self_hash(results, "payload_sha256")
        atomic_json_dump(results, staging / "results.json")
        completion: dict[str, Any] = {
            "complete": True,
            "manifest_identity_sha256": manifest["identity_sha256"],
            "manifest_file_sha256": sha256_file(staging / "manifest.json"),
            "results_payload_sha256": results["payload_sha256"],
            "results_file_sha256": sha256_file(staging / "results.json"),
            "trace_sha256": trace["sha256"],
            "wall_seconds": time.time() - started,
        }
        completion["payload_sha256"] = _canonical_self_hash(completion, "payload_sha256")
        atomic_json_dump(completion, staging / "completion.json")
        validate_output_bundle(staging)
        if args.outdir.exists():
            raise RuntimeError("output target appeared during replay; refusing overwrite")
        _atomic_install_directory_noreplace(staging, args.outdir)
    validate_output_bundle(args.outdir)
    print(json.dumps(results["per_branch"], ensure_ascii=False, indent=2))


def run_self_test() -> None:
    if torch.cuda.is_initialized():
        raise RuntimeError("self-test must run before CUDA initialization")
    rng = np.random.default_rng(19)
    theta = np.ascontiguousarray(rng.normal(size=(3, 4, 32, 32)), dtype=np.float64)
    sigma = np.ascontiguousarray(np.exp(rng.normal(-2.0, 0.1, size=theta.shape)), dtype=np.float32)
    masks, bounds = _component_masks()
    if bounds.shape != (16, 4) or masks.shape != (17, 1, 32, 32):
        raise AssertionError("fixed component geometry changed")
    raw, scale, K, u = construct_components_before_innovation(theta, sigma, masks, 0.002)
    z = np.ascontiguousarray(rng.normal(size=theta.shape), dtype=np.float32)
    R, L = evaluate_after_innovation(u, z)
    if raw.shape != (3, 17) or np.any(scale > 1.0) or np.any(K > 0.002 * (1 + 2e-12)):
        raise AssertionError("KL-capped component construction failed")
    if not np.allclose(L, R - K, rtol=0.0, atol=0.0):
        raise AssertionError("Gaussian LR decomposition failed")
    state = np.ascontiguousarray(rng.normal(size=(3, 4, 32, 32)), dtype=np.float32)
    eps = np.ascontiguousarray(rng.normal(size=state.shape), dtype=np.float32)
    alpha = 0.63
    x0 = (state.astype(np.float64) - math.sqrt(1.0 - alpha) * eps) / math.sqrt(alpha)
    reconstructed = reconstruct_current_epsilon(state, x0, alpha)
    if not np.allclose(reconstructed, eps, rtol=2e-5, atol=2e-5):
        raise AssertionError("current epsilon reconstruction failed")
    # Monte Carlo calibration for a fixed, predictable component mixture.
    toy_u = np.zeros((1, 2, 1, 1, 2), dtype=np.float64)
    toy_u[0, 0, 0, 0, 0] = 0.15
    toy_u[0, 1, 0, 0, 1] = -0.11
    e_values = []
    for _ in range(20_000):
        toy_z = np.ascontiguousarray(rng.normal(size=(1, 1, 1, 2)), dtype=np.float32)
        _, toy_l = evaluate_after_innovation(toy_u, toy_z)
        e_values.append(float(np.exp(_logmeanexp(toy_l, axis=(0, 1)))))
    if abs(float(np.mean(e_values)) - 1.0) > 0.01:
        raise AssertionError("fixed-mixture e-value calibration failed")
    print("self-test passed")


def build_parser() -> argparse.ArgumentParser:
    data_root = Path(os.environ.get("EQVAE_DATA_ROOT", "/home/zhoushunyu/data/eqvae"))
    default_dit = data_root / "baselines/DiT"
    default_vae = Path.home() / (
        ".cache/huggingface/hub/models--stabilityai--sd-vae-ft-mse/snapshots/"
        "31f26fdeee1355a5c34592e401dd41e45d25a493"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suffix-dir", type=Path, required=False)
    parser.add_argument("--observe-dir", type=Path, default=None)
    parser.add_argument("--baseline-dir", type=Path, default=None)
    parser.add_argument("--dit-root", type=Path, default=default_dit)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--vae-snapshot", type=Path, default=default_vae)
    parser.add_argument("--target-batch-index", type=int, default=0, choices=range(BATCH_SIZE))
    parser.add_argument(
        "--delta-nu",
        type=float,
        nargs="+",
        default=list(DEFAULT_DELTA_NU),
        help="Predeclared positive additive normalized-heat shifts (default: 0.25 1 4).",
    )
    parser.add_argument(
        "--total-K-per-component",
        type=float,
        default=DEFAULT_TOTAL_K_PER_COMPONENT,
        help="Maximum suffix KL for each fixed scale/mask component (default: 0.5).",
    )
    parser.add_argument("--outdir", type=Path, default=None)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--self-test", action="store_true")
    return parser


def normalize_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.self_test:
        return
    if args.suffix_dir is None:
        parser.error("--suffix-dir is required unless --self-test is used")
    args.suffix_dir = args.suffix_dir.expanduser().absolute().resolve()
    if not args.suffix_dir.is_dir():
        parser.error(f"suffix bundle is not a directory: {args.suffix_dir}")
    args.dit_root = args.dit_root.expanduser().absolute().resolve()
    args.checkpoint = (
        args.dit_root / "pretrained_models" / CHECKPOINT_FILENAME
        if args.checkpoint is None
        else args.checkpoint.expanduser().absolute().resolve()
    )
    args.vae_snapshot = args.vae_snapshot.expanduser().absolute().resolve()
    manifest = _read_self_hashed_json(args.suffix_dir / "manifest.json", "identity_sha256")
    args.observe_dir = (
        Path(manifest["frozen_observe_bundle"]["root"]).expanduser().absolute().resolve()
        if args.observe_dir is None
        else args.observe_dir.expanduser().absolute().resolve()
    )
    args.baseline_dir = (
        Path(manifest["frozen_baseline"]["root"]).expanduser().absolute().resolve()
        if args.baseline_dir is None
        else args.baseline_dir.expanduser().absolute().resolve()
    )
    if (
        not args.delta_nu
        or any(not math.isfinite(value) or value <= 0.0 for value in args.delta_nu)
        or len(set(args.delta_nu)) != len(args.delta_nu)
    ):
        parser.error("--delta-nu values must be distinct, finite, and positive")
    args.delta_nu = tuple(sorted(float(value) for value in args.delta_nu))
    if not math.isfinite(args.total_K_per_component) or args.total_K_per_component <= 0.0:
        parser.error("--total-K-per-component must be finite and positive")
    default_out = (
        Path(os.environ.get("EQVAE_DATA_ROOT", "/home/zhoushunyu/data/eqvae"))
        / "cross_scale_evidence/dit_imagenet256_suffix_cross_scale_replay"
        / f"{args.suffix_dir.name}_dnu{'-'.join(format(x, 'g') for x in args.delta_nu)}_{sha256_file(Path(__file__).resolve())[:7]}"
    )
    args.outdir = (
        default_out.expanduser().absolute().resolve()
        if args.outdir is None
        else args.outdir.expanduser().absolute().resolve()
    )
    if os.path.lexists(args.outdir):
        parser.error(f"no-overwrite target already exists: {args.outdir}")
    protected = [
        args.suffix_dir,
        args.observe_dir,
        args.baseline_dir,
        args.dit_root,
        args.vae_snapshot,
        Path(__file__).resolve().parent.parent,
    ]
    for path in protected:
        if args.outdir == path or args.outdir in path.parents or path in args.outdir.parents:
            parser.error(f"--outdir overlaps protected input/source: {path}")


def dry_run(args: argparse.Namespace) -> None:
    (
        observed,
        manifest,
        records,
        branch_arrays,
        alpha,
        _,
        _,
        _,
        _,
    ) = _load_and_validate_inputs(args, full_checkpoint=False)
    rollback = int(manifest["frozen_screen_protocol"]["this_invocation_rollback_internal_timestep"])
    schedule = _schedule_maps(alpha, rollback, args.delta_nu)
    effective = np.count_nonzero(
        (schedule["shifted_internal_timestep"] != schedule["internal_timestep"][None])
        & (schedule["internal_timestep"][None] > 0),
        axis=1,
    )
    probe = checkpoint_dry_probe(args.checkpoint)
    reconstruction_control = epsilon_reconstruction_control(
        observed,
        alpha,
        rollback=rollback,
        target_index=int(manifest["target"]["batch_index"]),
    )
    payload = {
        "status": "dry-run",
        "experiment": EXPERIMENT,
        "gpu_model_loaded": False,
        "suffix_bundle_strictly_validated": True,
        "suffix_manifest_identity_sha256": manifest["identity_sha256"],
        "observer_identity_sha256": observed.identity_sha256,
        "fresh_attempt_indices": [int(item["attempt_index"]) for item in records],
        "trace_shapes": [list(item["target_state_before"].shape) for item in branch_arrays],
        "delta_nu": list(args.delta_nu),
        "effective_stochastic_steps": effective.tolist(),
        "expected_shifted_forward_calls": int(effective.sum()),
        "expected_naive_unbatched_forward_equivalents": int(effective.sum() * len(records)),
        "current_epsilon_reconstruction_baseline_control": reconstruction_control,
        "checkpoint_probe": probe,
        "static_inputs_ready": bool(
            probe["exists"] and probe["size_matches"] and probe["sha256_pinned"]
        ),
        "outdir": str(args.outdir),
        "discovery_only": True,
        "image_or_label_dependent_ROI_used": False,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    normalize_args(args, parser)
    if args.self_test:
        run_self_test()
    elif args.dry_run:
        dry_run(args)
    else:
        run_real(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
