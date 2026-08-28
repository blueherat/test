#!/usr/bin/env python3
"""Observe local cross-scale path evidence on the frozen official DiT demo.

This runner preserves the released DiT-XL/2 ImageNet-256 sampling path and
records, but never changes, every ancestral DDPM transition.  In particular it
keeps the official batch-of-eight -> duplicated batch-of-sixteen CFG contract,
draws a full 2B Gaussian innovation at all 250 internal timesteps (including
the zero-multiplied draw at t=0), and uses ``forward_with_cfg`` exactly as
released.  That method guides only epsilon channels 0..2; channel 3 is still
recorded as part of the four-channel diffusion epsilon.

At stochastic timesteps, the diagnostic evaluates the same normalized latent
state at the current scale and at the nearest higher-noise *internal* scale
whose normalized heat time approximates ``nu + 1``.  The pulled-back
four-channel score difference defines sixteen predeclared 8x8 latent-tile
same-covariance Gaussian alternatives.  A component is one fixed tile for the
whole path; the component weights are uniformly fixed before sampling.  Each
component has at most K_total=0.5 conditional KL over the complete path.

For component j and implemented P transition ``mu + sigma * epsilon``, the
stored operational likelihood-ratio increment is exactly

    L[j] = <u[j], epsilon> - 0.5 ||u[j]||^2,
    u[j] = gamma[j] * mask[j] * sigma * theta.

All tile shifts are constructed before the transition innovation is drawn.
The single run-level anytime alarm is the uniform mixture over all 8 images and
16 fixed-tile path alternatives, compared with ``1 / alpha_total``.  It does
not stop or alter sampling.

"Operational exactness" here means exactness for the two explicitly
implemented, learned, discretized same-covariance Markov transitions.  It is
not a claim that this Q is an ideal heat-flow marginal ratio.  The output is
observe-only, performs no image-quality scoring, and is not an intervention.

The final decoded eight images (and the official grid) must be pixel-identical
to a separately completed frozen ``reproduce_dit_imagenet256.py`` run.  The
baseline, source tree, checkpoint, VAE, every trace array, every PNG, and all
manifest/result/completion links are validated fail-closed.  Output targets
are staged atomically and are never overwritten.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

# Match the module-scope settings in the frozen baseline runner/upstream demo.
sys.dont_write_bytecode = True

import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

try:  # Package and direct CLI imports.
    from .adm64_path_evidence import nearest_additive_heat_shift
    from .reproduce_dit_imagenet256 import (
        CFG_SCALE,
        CHECKPOINT_FILENAME,
        CLASS_IDS,
        COMPLETION_SCHEMA as BASELINE_COMPLETION_SCHEMA,
        DIT_REVISION,
        IMAGE_SIZE,
        LATENT_CHANNELS,
        LATENT_SIZE,
        MANIFEST_SCHEMA as BASELINE_MANIFEST_SCHEMA,
        MODEL_NAME,
        NULL_CLASS_ID,
        NUM_CLASSES,
        NUM_SAMPLING_STEPS,
        VAE_KIND,
        VAE_MODEL_ID,
        VAE_REVISION,
        VAE_SCALING_FACTOR,
        atomic_json_dump,
        checkpoint_dry_probe,
        dependency_identity,
        ensure_single_process,
        expected_output_specs,
        individual_relative_paths,
        inspect_png,
        load_json,
        save_outputs,
        sha256_file,
        sha256_json,
        tensor_sha256,
        cuda_rng_state_sha256,
        validate_checkpoint,
        validate_completed_output,
        validate_repository,
        validate_vae_snapshot,
    )
except ImportError:  # pragma: no cover - direct script execution.
    from adm64_path_evidence import nearest_additive_heat_shift
    from reproduce_dit_imagenet256 import (
        CFG_SCALE,
        CHECKPOINT_FILENAME,
        CLASS_IDS,
        COMPLETION_SCHEMA as BASELINE_COMPLETION_SCHEMA,
        DIT_REVISION,
        IMAGE_SIZE,
        LATENT_CHANNELS,
        LATENT_SIZE,
        MANIFEST_SCHEMA as BASELINE_MANIFEST_SCHEMA,
        MODEL_NAME,
        NULL_CLASS_ID,
        NUM_CLASSES,
        NUM_SAMPLING_STEPS,
        VAE_KIND,
        VAE_MODEL_ID,
        VAE_REVISION,
        VAE_SCALING_FACTOR,
        atomic_json_dump,
        checkpoint_dry_probe,
        dependency_identity,
        ensure_single_process,
        expected_output_specs,
        individual_relative_paths,
        inspect_png,
        load_json,
        save_outputs,
        sha256_file,
        sha256_json,
        tensor_sha256,
        cuda_rng_state_sha256,
        validate_checkpoint,
        validate_completed_output,
        validate_repository,
        validate_vae_snapshot,
    )


EXPERIMENT = "dit_imagenet256_local_cross_scale_path_evidence_observe_only"
SCHEMA_VERSION = 1
ADDITIVE_NORMALIZED_HEAT_SHIFT = 1.0
TOTAL_K_BUDGET = 0.5
GRID_SIZE = 4
TILE_COUNT = GRID_SIZE * GRID_SIZE
DEFAULT_ALPHA_TOTAL = 0.05
BATCH_SIZE = len(CLASS_IDS)
FULL_BATCH_SIZE = 2 * BATCH_SIZE
TRACE_NAME = "traces/full_batch_trace.npz"

TRACE_DTYPES: dict[str, np.dtype[Any]] = {
    "K_component": np.dtype(np.float64),
    "L_component": np.dtype(np.float64),
    "R_component": np.dtype(np.float64),
    "component_log_e": np.dtype(np.float64),
    "component_scale": np.dtype(np.float64),
    "component_weight": np.dtype(np.float64),
    "current_alpha_bar": np.dtype(np.float64),
    "current_original_timestep": np.dtype(np.int16),
    "effective_nonidentity": np.dtype(np.uint8),
    "epsilon_current": np.dtype(np.float32),
    "epsilon_shifted": np.dtype(np.float32),
    "final_latents_first_half": np.dtype(np.float32),
    "final_latents_second_half": np.dtype(np.float32),
    "innovation_first_half": np.dtype(np.float32),
    "internal_timestep": np.dtype(np.int16),
    "p_mean_first_half": np.dtype(np.float32),
    "p_standard_deviation": np.dtype(np.float32),
    "pred_xstart": np.dtype(np.float32),
    "raw_K_component": np.dtype(np.float64),
    "rho": np.dtype(np.float64),
    "run_mixture_log_e": np.dtype(np.float64),
    "sample_mixture_log_e": np.dtype(np.float64),
    "shifted_alpha_bar": np.dtype(np.float64),
    "shifted_internal_timestep": np.dtype(np.int16),
    "shifted_original_timestep": np.dtype(np.int16),
    "theta": np.dtype(np.float64),
    "tile_bounds_yxyx": np.dtype(np.int16),
    "transition_noise_multiplier": np.dtype(np.uint8),
    "x_t": np.dtype(np.float32),
}


@dataclass(frozen=True)
class BaselineRun:
    root: Path
    manifest: dict[str, Any]
    completion: dict[str, Any]
    output_records: tuple[dict[str, Any], ...]

    @property
    def identity_sha256(self) -> str:
        return str(self.manifest["identity_sha256"])


@dataclass(frozen=True)
class EvidenceSpec:
    internal_alpha_bar: np.ndarray
    timestep_map: np.ndarray
    shifted_internal_timestep: np.ndarray
    current_heat_variance: np.ndarray
    target_heat_variance: np.ndarray
    shifted_heat_variance: np.ndarray
    actual_heat_shift: np.ndarray
    absolute_mapping_error: np.ndarray
    effective_nonidentity_steps: int
    fixed_K_allowance: float
    guarded_K_cap: float
    tile_bounds_yxyx: np.ndarray


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _array_raw_sha256(array: np.ndarray) -> str:
    return _sha256_bytes(np.ascontiguousarray(array).tobytes(order="C"))


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


def _tensor_numpy(tensor: torch.Tensor, dtype: np.dtype[Any]) -> np.ndarray:
    return np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=dtype)


def _logmeanexp(values: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    maximum = np.max(values, axis=axis, keepdims=True)
    centered = np.exp(values - maximum)
    count = 1
    axes = (axis,) if isinstance(axis, int) else axis
    normalized_axes = tuple(item % values.ndim for item in axes)
    for item in normalized_axes:
        count *= values.shape[item]
    result = maximum + np.log(
        np.sum(centered, axis=axis, keepdims=True, dtype=np.float64) / count
    )
    for item in sorted(normalized_axes, reverse=True):
        result = np.squeeze(result, axis=item)
    return np.asarray(result, dtype=np.float64)


def fixed_tile_bounds(
    *, grid_size: int = GRID_SIZE, height: int = LATENT_SIZE, width: int = LATENT_SIZE
) -> np.ndarray:
    if grid_size <= 0 or height % grid_size or width % grid_size:
        raise ValueError("latent height/width must be divisible by the fixed grid")
    tile_h, tile_w = height // grid_size, width // grid_size
    bounds = []
    for tile_y in range(grid_size):
        for tile_x in range(grid_size):
            bounds.append(
                (tile_y * tile_h, tile_x * tile_w, (tile_y + 1) * tile_h, (tile_x + 1) * tile_w)
            )
    return np.asarray(bounds, dtype=np.int16)


def fixed_tile_masks(
    bounds: np.ndarray, *, height: int = LATENT_SIZE, width: int = LATENT_SIZE
) -> np.ndarray:
    if bounds.ndim != 2 or bounds.shape[1] != 4:
        raise ValueError("tile bounds must have shape [components,4]")
    masks = np.zeros((len(bounds), 1, height, width), dtype=np.float64)
    for index, (y0, x0, y1, x1) in enumerate(bounds.tolist()):
        if not (0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width):
            raise ValueError("invalid half-open tile bounds")
        masks[index, 0, y0:y1, x0:x1] = 1.0
    if not np.array_equal(masks.sum(axis=0), np.ones((1, height, width))):
        raise ValueError("fixed tiles must partition the latent plane exactly once")
    return masks


def construct_predictable_tile_shifts(
    theta: np.ndarray,
    p_sigma: np.ndarray,
    masks: np.ndarray,
    guarded_K_cap: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct all component shifts without accepting an innovation argument."""

    if theta.shape != p_sigma.shape or theta.ndim != 4:
        raise ValueError("theta and p_sigma must match [B,C,H,W]")
    if theta.dtype != np.float64 or p_sigma.dtype != np.float32:
        raise TypeError("canonical Q construction expects float64 theta and float32 sigma")
    if masks.shape != (TILE_COUNT, 1, theta.shape[2], theta.shape[3]):
        raise ValueError("fixed mask shape changed")
    if not np.isfinite(theta).all() or not np.isfinite(p_sigma).all():
        raise ValueError("Q inputs contain non-finite values")
    if np.any(p_sigma <= 0) or not math.isfinite(guarded_K_cap) or guarded_K_cap <= 0:
        raise ValueError("P sigma and guarded K cap must be strictly positive")
    base = p_sigma.astype(np.float64, copy=False) * theta
    raw_u = np.ascontiguousarray(base[:, None] * masks[None])
    raw_K = 0.5 * np.sum(
        np.square(raw_u), axis=(2, 3, 4), dtype=np.float64
    )
    scale = np.ones_like(raw_K)
    positive = raw_K > 0
    scale[positive] = np.minimum(
        1.0, np.sqrt(float(guarded_K_cap) / raw_K[positive])
    )
    u = np.ascontiguousarray(raw_u * scale[:, :, None, None, None])
    K = 0.5 * np.sum(np.square(u), axis=(2, 3, 4), dtype=np.float64)
    if np.any(K > guarded_K_cap * (1 + 1e-12)):
        raise AssertionError("a fixed-tile component exceeded the declared step cap")
    return raw_K, scale, K, u


def evaluate_tile_log_lr(
    whitened_shift: np.ndarray, innovation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if whitened_shift.ndim != 5 or innovation.shape != (
        whitened_shift.shape[0],
        whitened_shift.shape[2],
        whitened_shift.shape[3],
        whitened_shift.shape[4],
    ):
        raise ValueError("innovation does not match the predictable tile shifts")
    if innovation.dtype != np.float32 or whitened_shift.dtype != np.float64:
        raise TypeError("canonical LR expects float64 shifts and float32 P innovation")
    R = np.sum(
        whitened_shift * innovation.astype(np.float64, copy=False)[:, None],
        axis=(2, 3, 4),
        dtype=np.float64,
    )
    K = 0.5 * np.sum(
        np.square(whitened_shift), axis=(2, 3, 4), dtype=np.float64
    )
    return np.ascontiguousarray(R), np.ascontiguousarray(R - K)


def build_evidence_spec(
    internal_alpha_bar: np.ndarray,
    timestep_map: np.ndarray,
    *,
    total_K_budget: float = TOTAL_K_BUDGET,
) -> EvidenceSpec:
    alpha = np.asarray(internal_alpha_bar, dtype=np.float64)
    mapping = np.asarray(timestep_map, dtype=np.int64)
    if alpha.shape != (NUM_SAMPLING_STEPS,) or mapping.shape != (NUM_SAMPLING_STEPS,):
        raise ValueError("DiT respaced schedule must contain exactly 250 internal steps")
    if np.any(alpha <= 0) or np.any(alpha >= 1) or not np.all(np.diff(alpha) < 0):
        raise ValueError("internal alpha_bar schedule is invalid")
    if mapping[0] != 0 or mapping[-1] != 999 or not np.all(np.diff(mapping) > 0):
        raise ValueError("DiT internal-to-original timestep map is invalid")
    if total_K_budget != TOTAL_K_BUDGET:
        raise ValueError(f"this observer freezes total K to {TOTAL_K_BUDGET}")

    current = np.arange(NUM_SAMPLING_STEPS - 1, 0, -1, dtype=np.int64)
    shifted = nearest_additive_heat_shift(
        alpha, current, ADDITIVE_NORMALIZED_HEAT_SHIFT
    )
    shifted_internal = np.concatenate(
        [shifted.shifted_timestep.astype(np.int64), np.asarray([0], dtype=np.int64)]
    )
    effective = int(np.count_nonzero(shifted.shifted_timestep != current))
    if effective <= 0:
        raise RuntimeError("Delta-nu mapping unexpectedly has no effective stochastic steps")
    allowance = float(total_K_budget) / effective
    guarded = allowance * (1.0 - 2e-12)
    current_nu = np.concatenate(
        [shifted.current_heat_variance, np.asarray([(1 - alpha[0]) / alpha[0]])]
    )
    target_nu = np.concatenate(
        [shifted.target_heat_variance, np.asarray([(1 - alpha[0]) / alpha[0]])]
    )
    shifted_nu = np.concatenate(
        [shifted.shifted_heat_variance, np.asarray([(1 - alpha[0]) / alpha[0]])]
    )
    actual_shift = np.concatenate([shifted.actual_heat_shift, np.asarray([0.0])])
    mapping_error = np.concatenate([shifted.absolute_mapping_error, np.asarray([0.0])])
    return EvidenceSpec(
        internal_alpha_bar=alpha,
        timestep_map=mapping,
        shifted_internal_timestep=shifted_internal,
        current_heat_variance=np.ascontiguousarray(current_nu),
        target_heat_variance=np.ascontiguousarray(target_nu),
        shifted_heat_variance=np.ascontiguousarray(shifted_nu),
        actual_heat_shift=np.ascontiguousarray(actual_shift),
        absolute_mapping_error=np.ascontiguousarray(mapping_error),
        effective_nonidentity_steps=effective,
        fixed_K_allowance=allowance,
        guarded_K_cap=guarded,
        tile_bounds_yxyx=fixed_tile_bounds(),
    )


def _with_upstream_imports(root: Path, callback: Any) -> Any:
    old_cwd = Path.cwd()
    old_sys_path = list(sys.path)
    preexisting = {
        name
        for name in sys.modules
        if name in {"models", "download", "diffusion"} or name.startswith("diffusion.")
    }
    if preexisting:
        raise RuntimeError("ambiguous pre-imported upstream modules: " + repr(sorted(preexisting)))
    try:
        os.chdir(root)
        sys.path.insert(0, str(root))
        return callback()
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_sys_path
        for name in list(sys.modules):
            if name in {"models", "download", "diffusion"} or name.startswith("diffusion."):
                if name not in preexisting:
                    sys.modules.pop(name, None)


def load_schedule(root: Path) -> tuple[np.ndarray, np.ndarray]:
    def _load() -> tuple[np.ndarray, np.ndarray]:
        from diffusion import create_diffusion

        diffusion = create_diffusion(str(NUM_SAMPLING_STEPS))
        if diffusion.num_timesteps != NUM_SAMPLING_STEPS:
            raise RuntimeError("upstream create_diffusion no longer produces 250 steps")
        alpha = np.asarray(diffusion.alphas_cumprod, dtype=np.float64)
        mapping = np.asarray(diffusion.timestep_map, dtype=np.int64)
        return alpha, mapping

    return _with_upstream_imports(root, _load)


def _expected_baseline_protocol(seed: int) -> dict[str, Any]:
    return {
        "upstream_entry": "sample.py",
        "model": MODEL_NAME,
        "image_size": IMAGE_SIZE,
        "latent_shape_before_duplication": [BATCH_SIZE, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE],
        "latent_shape_after_duplication": [FULL_BATCH_SIZE, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE],
        "num_classes": NUM_CLASSES,
        "class_ids": list(CLASS_IDS),
        "null_class_id": NULL_CLASS_ID,
        "num_sampling_steps": NUM_SAMPLING_STEPS,
        "sampler": "ancestral DDPM (upstream p_sample_loop)",
        "clip_denoised": False,
        "cfg_scale": CFG_SCALE,
        "cfg_epsilon_channels": 3,
        "vae": VAE_KIND,
        "vae_scaling_factor": VAE_SCALING_FACTOR,
        "global_torch_seed": seed,
        "one_seed_per_eight_image_run": True,
    }


def validate_baseline_run(
    root: Path,
    *,
    seed: int,
    source: dict[str, Any] | None = None,
    checkpoint: dict[str, Any] | None = None,
    vae: dict[str, Any] | None = None,
) -> BaselineRun:
    if root.name != f"official_demo_seed{seed}":
        raise RuntimeError("baseline directory must be the exact official_demo_seedN run")
    manifest = load_json(root / "manifest.json")
    completion = load_json(root / "completion.json")
    identity = manifest.get("identity")
    if not isinstance(identity, dict):
        raise RuntimeError("baseline manifest lacks an identity object")
    # Reuse the frozen baseline runner's complete output validator.
    validate_completed_output(root, identity)
    if manifest.get("schema") != BASELINE_MANIFEST_SCHEMA or manifest.get("status") != "complete":
        raise RuntimeError("baseline manifest is not complete")
    if completion.get("schema") != BASELINE_COMPLETION_SCHEMA:
        raise RuntimeError("baseline completion schema changed")
    baseline_runner = Path(__file__).with_name("reproduce_dit_imagenet256.py").resolve()
    fixed = {
        "runner": "reproduce_dit_imagenet256",
        "schema": BASELINE_MANIFEST_SCHEMA,
        "baseline_only": True,
        "counterfactual_q": None,
        "path_likelihood_ratio": None,
        "rollback": None,
        "protocol": _expected_baseline_protocol(seed),
    }
    mismatches = {key: (identity.get(key), value) for key, value in fixed.items() if identity.get(key) != value}
    if mismatches:
        raise RuntimeError(f"frozen DiT baseline identity mismatch: {mismatches}")
    runner_record = identity.get("runner_source", {})
    if runner_record.get("sha256") != sha256_file(baseline_runner):
        raise RuntimeError("baseline was not produced by the current frozen reproduction runner")
    if source is not None and identity.get("source") != source:
        raise RuntimeError("baseline DiT source identity differs from the validated source")
    if checkpoint is not None and identity.get("checkpoint") != checkpoint:
        raise RuntimeError("baseline checkpoint identity differs from the validated checkpoint")
    if vae is not None and identity.get("vae_snapshot") != vae:
        raise RuntimeError("baseline VAE identity differs from the validated VAE")
    records = tuple(manifest.get("outputs", ()))
    if len(records) != 1 + BATCH_SIZE:
        raise RuntimeError("baseline does not contain the official grid plus eight images")
    return BaselineRun(root.resolve(), manifest, completion, records)


def _mapping_record(spec: EvidenceSpec) -> dict[str, Any]:
    reverse = np.arange(NUM_SAMPLING_STEPS - 1, -1, -1, dtype=np.int64)
    shifted = spec.shifted_internal_timestep
    return {
        "internal_timesteps_reverse_order": reverse.tolist(),
        "current_original_timestep": spec.timestep_map[reverse].tolist(),
        "shifted_internal_timestep": shifted.tolist(),
        "shifted_original_timestep": spec.timestep_map[shifted].tolist(),
        "current_alpha_bar": spec.internal_alpha_bar[reverse].tolist(),
        "shifted_alpha_bar": spec.internal_alpha_bar[shifted].tolist(),
        "current_heat_variance": spec.current_heat_variance.tolist(),
        "target_heat_variance": spec.target_heat_variance.tolist(),
        "shifted_heat_variance": spec.shifted_heat_variance.tolist(),
        "actual_heat_shift": spec.actual_heat_shift.tolist(),
        "absolute_mapping_error": spec.absolute_mapping_error.tolist(),
        "effective_nonidentity_stochastic_steps": spec.effective_nonidentity_steps,
        "identity_stochastic_steps": NUM_SAMPLING_STEPS - 1 - spec.effective_nonidentity_steps,
        "terminal_t0_forced_identity": True,
    }


def canonical_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--seed",
        str(args.seed),
        "--alpha-total",
        repr(args.alpha_total),
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


def build_manifest(
    args: argparse.Namespace,
    *,
    baseline: BaselineRun,
    source: dict[str, Any],
    checkpoint: dict[str, Any],
    vae: dict[str, Any],
    spec: EvidenceSpec,
) -> dict[str, Any]:
    runner = Path(__file__).resolve()
    primitive = runner.with_name("adm64_path_evidence.py")
    baseline_runner = runner.with_name("reproduce_dit_imagenet256.py")
    weights = [1.0 / TILE_COUNT] * TILE_COUNT
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "role": "STRICT_OBSERVE_ONLY_OPERATIONAL_PATH_EVIDENCE",
        "observe_only": True,
        "interventions": 0,
        "sampling_distribution_P_changed": False,
        "automatic_image_quality_scoring": False,
        "operational_exactness": (
            "exact finite-step Q/P LR for the implemented learned discretized "
            "same-covariance Gaussian transitions"
        ),
        "ideal_marginal_ratio_claimed": False,
        "seed": args.seed,
        "class_ids_in_official_batch_order": list(CLASS_IDS),
        "frozen_baseline": {
            "root": str(baseline.root),
            "identity_sha256": baseline.identity_sha256,
            "manifest_sha256": sha256_file(baseline.root / "manifest.json"),
            "completion_sha256": sha256_file(baseline.root / "completion.json"),
            "outputs_sha256": baseline.manifest["outputs_sha256"],
        },
        "official_P_contract": {
            "model": MODEL_NAME,
            "steps": NUM_SAMPLING_STEPS,
            "internal_timestep_order": "249 down to 0 inclusive",
            "initial_batch_shape": [BATCH_SIZE, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE],
            "sampler_batch_shape": [FULL_BATCH_SIZE, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE],
            "cfg_scale": CFG_SCALE,
            "forward_with_cfg_guided_epsilon_channels": [0, 1, 2],
            "captured_epsilon_channels": [0, 1, 2, 3],
            "forward_with_cfg_channel_3_is_unguided_rest": True,
            "full_2B_noise_draw_each_step": True,
            "terminal_t0_full_2B_noise_draw_consumed_then_zero_multiplied": True,
            "second_half_transition_state_retained_for_rng_and_sampler_fidelity_then_discarded": True,
            "clip_denoised": False,
        },
        "operational_Q": {
            "additive_normalized_heat_shift": ADDITIVE_NORMALIZED_HEAT_SHIFT,
            "shift_mapping": "nearest higher-noise discrete internal timestep; ties prefer higher noise",
            "normalized_state_pullback": "x_shifted = rho*x_current, rho=sqrt(alpha_shifted/alpha_current)",
            "score_difference": "rho*s_shifted(rho*x)-s_current(x), from captured four-channel CFG epsilon",
            "same_covariance": True,
            "mean_shift": "delta_j = sigma^2 * gamma_j * mask_j * theta",
            "component_definition": "one fixed row-major latent tile for the entire path",
            "grid_size_per_axis": GRID_SIZE,
            "component_count": TILE_COUNT,
            "tile_size_latent": [LATENT_SIZE // GRID_SIZE, LATENT_SIZE // GRID_SIZE],
            "tile_bounds_yxyx_half_open": spec.tile_bounds_yxyx.tolist(),
            "component_weights": weights,
            "component_weights_sum": float(sum(weights)),
            "total_K_budget_per_image_component": TOTAL_K_BUDGET,
            "fixed_K_allowance_per_effective_step": spec.fixed_K_allowance,
            "guarded_numerical_cap_per_effective_step": spec.guarded_K_cap,
            "unused_step_allowance_carried_forward": False,
            "tile_selected_after_innovation": False,
            "terminal_t0_Q_equals_P": True,
            "mapping": _mapping_record(spec),
        },
        "anytime_alarm": {
            "alpha_total": args.alpha_total,
            "threshold_E": 1.0 / args.alpha_total,
            "threshold_log_E": math.log(1.0 / args.alpha_total),
            "process": "uniform mixture over 8 image paths x 16 fixed-tile path alternatives",
            "stops_sampling": False,
            "triggers_intervention": False,
        },
        "sources": {
            "dit": source,
            "checkpoint": checkpoint,
            "vae": vae,
            "baseline_runner": {"path": str(baseline_runner), "sha256": sha256_file(baseline_runner)},
            "path_evidence_primitive": {"path": str(primitive), "sha256": sha256_file(primitive)},
        },
        "runner": {"path": str(runner), "sha256": sha256_file(runner)},
        "dependencies": dependency_identity(),
        "canonical_command": canonical_command(args),
        "output_contract": {
            "trace": TRACE_NAME,
            "official_grid": "sample.png",
            "individual_images": list(individual_relative_paths()),
            "final_eight_images_pixel_equal_frozen_baseline_required": True,
            "no_overwrite": True,
        },
    }
    manifest["identity_sha256"] = _canonical_self_hash(manifest, "identity_sha256")
    return manifest


def _atomic_npz_dump(arrays: dict[str, np.ndarray], path: Path) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite trace: {path}")
    path.parent.mkdir(parents=True, exist_ok=False)
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
        "arrays": {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "raw_bytes_sha256": _array_raw_sha256(value),
            }
            for key, value in sorted(arrays.items())
        },
    }


def _collect_png_records(root: Path) -> list[dict[str, Any]]:
    records = []
    for relative, (mode, size) in sorted(expected_output_specs().items()):
        record = {"relative_path": relative}
        record.update(inspect_png(root / relative, mode, size))
        records.append(record)
    return records


def _finalize_trace(
    lists: dict[str, list[np.ndarray | float | int]],
    *,
    component_weight: np.ndarray,
    tile_bounds: np.ndarray,
    final_first: np.ndarray,
    final_second: np.ndarray,
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for key, values in lists.items():
        arrays[key] = np.ascontiguousarray(np.stack(values, axis=0))
    arrays["component_weight"] = np.ascontiguousarray(component_weight)
    arrays["tile_bounds_yxyx"] = np.ascontiguousarray(tile_bounds)
    arrays["final_latents_first_half"] = np.ascontiguousarray(final_first)
    arrays["final_latents_second_half"] = np.ascontiguousarray(final_second)
    arrays = {key: arrays[key] for key in sorted(arrays)}
    if set(arrays) != set(TRACE_DTYPES):
        raise AssertionError(f"trace key set changed: {sorted(set(arrays) ^ set(TRACE_DTYPES))}")
    for key, expected_dtype in TRACE_DTYPES.items():
        if arrays[key].dtype != expected_dtype:
            raise AssertionError(f"trace dtype changed for {key}: {arrays[key].dtype}")
    return arrays


def _new_trace_lists() -> dict[str, list[np.ndarray | float | int]]:
    static_or_final = {"component_weight", "tile_bounds_yxyx", "final_latents_first_half", "final_latents_second_half"}
    return {key: [] for key in TRACE_DTYPES if key not in static_or_final}


def run_observe(
    args: argparse.Namespace,
    *,
    spec: EvidenceSpec,
    baseline: BaselineRun,
    staging: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real DiT-XL/2 observer run")
    ensure_single_process()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["DIFFUSERS_OFFLINE"] = "1"
    prior_grad = torch.is_grad_enabled()

    def _execute() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        from diffusion import create_diffusion
        from diffusers.models import AutoencoderKL
        from download import find_model
        from models import DiT_models
        from torchvision.utils import save_image

        imported = {
            "diffusion": Path(sys.modules["diffusion"].__file__).resolve(),
            "download": Path(sys.modules["download"].__file__).resolve(),
            "models": Path(sys.modules["models"].__file__).resolve(),
        }
        expected = {
            "diffusion": (args.dit_root / "diffusion/__init__.py").resolve(),
            "download": (args.dit_root / "download.py").resolve(),
            "models": (args.dit_root / "models.py").resolve(),
        }
        if imported != expected:
            raise RuntimeError(f"upstream import shadowing detected: {imported} != {expected}")

        # Preserve the official sample.py/reproduction statement order.
        torch.manual_seed(args.seed)
        torch.set_grad_enabled(False)
        device = torch.device("cuda")
        rng_after_manual_seed = cuda_rng_state_sha256()
        model = DiT_models[MODEL_NAME](input_size=LATENT_SIZE, num_classes=NUM_CLASSES).to(device)
        model.load_state_dict(find_model(str(args.checkpoint)))
        model.eval()
        diffusion = create_diffusion(str(NUM_SAMPLING_STEPS))
        vae = AutoencoderKL.from_pretrained(
            str(args.vae_snapshot), local_files_only=True, use_safetensors=True
        ).to(device)
        if not np.array_equal(np.asarray(diffusion.timestep_map), spec.timestep_map) or not np.array_equal(
            np.asarray(diffusion.alphas_cumprod), spec.internal_alpha_bar
        ):
            raise RuntimeError("runtime diffusion schedule differs from the prevalidated manifest schedule")

        z_initial = torch.randn(
            BATCH_SIZE, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE, device=device
        )
        initial_noise_hash = tensor_sha256(z_initial)
        rng_after_initial_noise = cuda_rng_state_sha256()
        y_conditional = torch.tensor(CLASS_IDS, device=device)
        x = torch.cat([z_initial, z_initial], dim=0)
        y_null = torch.tensor([NULL_CLASS_ID] * BATCH_SIZE, device=device)
        y = torch.cat([y_conditional, y_null], dim=0)
        model_kwargs = {"y": y, "cfg_scale": CFG_SCALE}

        masks = fixed_tile_masks(spec.tile_bounds_yxyx)
        component_weight = np.full(TILE_COUNT, 1.0 / TILE_COUNT, dtype=np.float64)
        cumulative = np.zeros((BATCH_SIZE, TILE_COUNT), dtype=np.float64)
        trace_lists = _new_trace_lists()
        transitions: list[dict[str, Any]] = []
        first_alarm_internal_timestep: int | None = None
        threshold_log = math.log(1.0 / args.alpha_total)
        shifted_eval_count = 0

        for step_index, internal_t in enumerate(range(NUM_SAMPLING_STEPS - 1, -1, -1)):
            t = torch.full((FULL_BATCH_SIZE,), internal_t, dtype=torch.long, device=device)
            current_box: list[torch.Tensor] = []
            current_t_box: list[torch.Tensor] = []

            def capturing_forward_with_cfg(
                x_in: torch.Tensor,
                original_t: torch.Tensor,
                y: torch.Tensor,
                cfg_scale: float,
            ) -> torch.Tensor:
                value = model.forward_with_cfg(x_in, original_t, y=y, cfg_scale=cfg_scale)
                current_box.append(value)
                current_t_box.append(original_t.detach().clone())
                return value

            rng_before_models = cuda_rng_state_sha256()
            state_before_hash = tensor_sha256(x)
            out = diffusion.p_mean_variance(
                capturing_forward_with_cfg,
                x,
                t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )
            if len(current_box) != 1 or len(current_t_box) != 1:
                raise AssertionError("p_mean_variance did not make exactly one captured current model call")
            current_original_t = int(spec.timestep_map[internal_t])
            observed_ts = current_t_box[0]
            if observed_ts.shape != (FULL_BATCH_SIZE,) or not torch.all(observed_ts == current_original_t):
                raise AssertionError("SpacedDiffusion supplied the wrong original current timestep")
            current_full_output = current_box[0]
            expected_shape = (FULL_BATCH_SIZE, 2 * LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE)
            if tuple(current_full_output.shape) != expected_shape:
                raise AssertionError(f"captured forward_with_cfg output shape changed: {tuple(current_full_output.shape)}")
            rng_after_current = cuda_rng_state_sha256()
            if rng_after_current != rng_before_models:
                raise RuntimeError("current DiT evaluation unexpectedly consumed CUDA RNG")

            shifted_internal_t = int(spec.shifted_internal_timestep[step_index])
            shifted_original_t = int(spec.timestep_map[shifted_internal_t])
            alpha_current = float(spec.internal_alpha_bar[internal_t])
            alpha_shifted = float(spec.internal_alpha_bar[shifted_internal_t])
            rho = math.sqrt(alpha_shifted / alpha_current)
            effective = internal_t > 0 and shifted_internal_t != internal_t
            epsilon_current_tensor = current_full_output[:BATCH_SIZE, :LATENT_CHANNELS]
            if effective:
                shifted_t = torch.full(
                    (FULL_BATCH_SIZE,), shifted_original_t, dtype=torch.long, device=device
                )
                shifted_full_output = model.forward_with_cfg(
                    x * rho, shifted_t, y=y, cfg_scale=CFG_SCALE
                )
                shifted_eval_count += 1
                if tuple(shifted_full_output.shape) != expected_shape:
                    raise AssertionError("shifted forward_with_cfg output shape changed")
                epsilon_shifted_tensor = shifted_full_output[:BATCH_SIZE, :LATENT_CHANNELS]
            else:
                shifted_full_output = current_full_output
                epsilon_shifted_tensor = epsilon_current_tensor
            rng_after_shifted = cuda_rng_state_sha256()
            if rng_after_shifted != rng_before_models:
                raise RuntimeError("cross-scale DiT observation unexpectedly consumed CUDA RNG")

            x_first = _tensor_numpy(x[:BATCH_SIZE], np.dtype(np.float32))
            pred_xstart = _tensor_numpy(out["pred_xstart"][:BATCH_SIZE], np.dtype(np.float32))
            p_mean = _tensor_numpy(out["mean"][:BATCH_SIZE], np.dtype(np.float32))
            p_sigma = _tensor_numpy(
                torch.exp(0.5 * out["log_variance"][:BATCH_SIZE]), np.dtype(np.float32)
            )
            epsilon_current = _tensor_numpy(epsilon_current_tensor, np.dtype(np.float32))
            epsilon_shifted = _tensor_numpy(epsilon_shifted_tensor, np.dtype(np.float32))
            if effective:
                theta = (
                    -rho
                    * epsilon_shifted.astype(np.float64)
                    / math.sqrt(1.0 - alpha_shifted)
                    + epsilon_current.astype(np.float64) / math.sqrt(1.0 - alpha_current)
                )
                theta = np.ascontiguousarray(theta, dtype=np.float64)
                raw_K, component_scale, K, whitened_shift = construct_predictable_tile_shifts(
                    theta, p_sigma, masks, spec.guarded_K_cap
                )
            else:
                theta = np.zeros_like(epsilon_current, dtype=np.float64)
                raw_K = np.zeros((BATCH_SIZE, TILE_COUNT), dtype=np.float64)
                component_scale = np.ones_like(raw_K)
                K = np.zeros_like(raw_K)
                whitened_shift = np.zeros(
                    (BATCH_SIZE, TILE_COUNT, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE),
                    dtype=np.float64,
                )

            # Critical boundary: all Q components above are fully constructed
            # before this first and only transition-noise draw.
            rng_before_noise = cuda_rng_state_sha256()
            if rng_before_noise != rng_before_models:
                raise RuntimeError("RNG changed before the official transition draw")
            noise = torch.randn_like(x)
            rng_after_noise = cuda_rng_state_sha256()
            nonzero_mask = (t != 0).float().view(-1, 1, 1, 1)
            p_sigma_full = torch.exp(0.5 * out["log_variance"])
            x_next = out["mean"] + nonzero_mask * p_sigma_full * noise
            innovation = _tensor_numpy(noise[:BATCH_SIZE], np.dtype(np.float32))
            if effective:
                R, L = evaluate_tile_log_lr(whitened_shift, innovation)
                if not np.allclose(
                    K,
                    0.5 * np.sum(np.square(whitened_shift), axis=(2, 3, 4), dtype=np.float64),
                    rtol=0.0,
                    atol=0.0,
                ):
                    raise AssertionError("canonical K did not reconstruct")
            else:
                R = np.zeros_like(K)
                L = np.zeros_like(K)
            cumulative = np.ascontiguousarray(cumulative + L)
            sample_mixture = _logmeanexp(cumulative, axis=1)
            run_mixture = float(_logmeanexp(cumulative, axis=(0, 1)))
            if first_alarm_internal_timestep is None and run_mixture >= threshold_log:
                first_alarm_internal_timestep = internal_t

            trace_lists["internal_timestep"].append(np.asarray(internal_t, dtype=np.int16))
            trace_lists["current_original_timestep"].append(np.asarray(current_original_t, dtype=np.int16))
            trace_lists["shifted_internal_timestep"].append(np.asarray(shifted_internal_t, dtype=np.int16))
            trace_lists["shifted_original_timestep"].append(np.asarray(shifted_original_t, dtype=np.int16))
            trace_lists["current_alpha_bar"].append(np.asarray(alpha_current, dtype=np.float64))
            trace_lists["shifted_alpha_bar"].append(np.asarray(alpha_shifted, dtype=np.float64))
            trace_lists["rho"].append(np.asarray(rho, dtype=np.float64))
            trace_lists["effective_nonidentity"].append(np.asarray(int(effective), dtype=np.uint8))
            trace_lists["transition_noise_multiplier"].append(np.asarray(int(internal_t > 0), dtype=np.uint8))
            trace_lists["x_t"].append(x_first)
            trace_lists["pred_xstart"].append(pred_xstart)
            trace_lists["p_mean_first_half"].append(p_mean)
            trace_lists["p_standard_deviation"].append(p_sigma)
            trace_lists["epsilon_current"].append(epsilon_current)
            trace_lists["epsilon_shifted"].append(epsilon_shifted)
            trace_lists["theta"].append(theta)
            trace_lists["innovation_first_half"].append(innovation)
            trace_lists["raw_K_component"].append(raw_K)
            trace_lists["component_scale"].append(component_scale)
            trace_lists["K_component"].append(K)
            trace_lists["R_component"].append(R)
            trace_lists["L_component"].append(L)
            trace_lists["component_log_e"].append(cumulative.copy())
            trace_lists["sample_mixture_log_e"].append(sample_mixture)
            trace_lists["run_mixture_log_e"].append(np.asarray(run_mixture, dtype=np.float64))

            transitions.append(
                {
                    "step_index": step_index,
                    "internal_timestep": internal_t,
                    "current_original_timestep": current_original_t,
                    "shifted_internal_timestep": shifted_internal_t,
                    "shifted_original_timestep": shifted_original_t,
                    "effective_nonidentity": effective,
                    "noise_draw_ordinal": step_index + 1,
                    "noise_shape": list(x.shape),
                    "terminal_noise_draw_zero_multiplied": internal_t == 0,
                    "rng_state_sha256": {
                        "before_models": rng_before_models,
                        "after_current_model": rng_after_current,
                        "after_shifted_model_and_Q_construction": rng_before_noise,
                        "after_full_2B_noise_draw": rng_after_noise,
                    },
                    "full_batch_tensor_sha256": {
                        "state_before": state_before_hash,
                        "captured_current_forward_with_cfg_output": tensor_sha256(current_full_output),
                        "captured_shifted_forward_with_cfg_output": tensor_sha256(shifted_full_output),
                        "pred_xstart": tensor_sha256(out["pred_xstart"]),
                        "p_mean": tensor_sha256(out["mean"]),
                        "p_standard_deviation": tensor_sha256(p_sigma_full),
                        "innovation": tensor_sha256(noise),
                        "state_after": tensor_sha256(x_next),
                    },
                    "first_half_trace_row_raw_sha256": {
                        "x_t": _array_raw_sha256(x_first),
                        "pred_xstart": _array_raw_sha256(pred_xstart),
                        "p_mean_first_half": _array_raw_sha256(p_mean),
                        "p_standard_deviation": _array_raw_sha256(p_sigma),
                        "epsilon_current": _array_raw_sha256(epsilon_current),
                        "epsilon_shifted": _array_raw_sha256(epsilon_shifted),
                        "theta": _array_raw_sha256(theta),
                        "innovation_first_half": _array_raw_sha256(innovation),
                        "raw_K_component": _array_raw_sha256(raw_K),
                        "component_scale": _array_raw_sha256(component_scale),
                        "K_component": _array_raw_sha256(K),
                        "R_component": _array_raw_sha256(R),
                        "L_component": _array_raw_sha256(L),
                        "component_log_e": _array_raw_sha256(cumulative),
                        "sample_mixture_log_e": _array_raw_sha256(sample_mixture),
                        "run_mixture_log_e": _array_raw_sha256(
                            np.asarray(run_mixture, dtype=np.float64)
                        ),
                    },
                    "run_mixture_log_e_after_transition": run_mixture,
                    "run_level_alpha_total_crossed": run_mixture >= threshold_log,
                }
            )
            x = x_next.detach()

        rng_after_diffusion = cuda_rng_state_sha256()
        final_first, final_second = x.chunk(2, dim=0)
        final_first_hash = tensor_sha256(final_first)
        final_second_hash = tensor_sha256(final_second)
        decoded = vae.decode(final_first / VAE_SCALING_FACTOR).sample
        decoded_hash = tensor_sha256(decoded)
        save_outputs(decoded, staging, save_image)
        torch.cuda.synchronize()

        final_first_np = _tensor_numpy(final_first, np.dtype(np.float32))
        final_second_np = _tensor_numpy(final_second, np.dtype(np.float32))
        arrays = _finalize_trace(
            trace_lists,
            component_weight=component_weight,
            tile_bounds=spec.tile_bounds_yxyx,
            final_first=final_first_np,
            final_second=final_second_np,
        )
        execution = {
            "rng_state_sha256": {
                "after_manual_seed": rng_after_manual_seed,
                "after_initial_noise": rng_after_initial_noise,
                "after_250_transition_noise_draws": rng_after_diffusion,
            },
            "tensor_sha256": {
                "initial_noise_b": initial_noise_hash,
                "final_latents_first_half_b": final_first_hash,
                "final_latents_discarded_second_half_b": final_second_hash,
                "decoded_samples_b": decoded_hash,
            },
            "accounting": {
                "current_forward_with_cfg_evaluations": NUM_SAMPLING_STEPS,
                "shifted_forward_with_cfg_evaluations": shifted_eval_count,
                "full_2B_transition_noise_draws": NUM_SAMPLING_STEPS,
                "terminal_t0_full_noise_draws": 1,
                "P_interventions": 0,
            },
            "transitions": transitions,
            "first_run_level_alarm_internal_timestep": first_alarm_internal_timestep,
        }
        baseline_execution = baseline.manifest.get("execution", {})
        if execution["rng_state_sha256"] != baseline_execution.get("rng_state_sha256"):
            raise RuntimeError("observer CUDA RNG trajectory endpoint differs from frozen baseline")
        if execution["tensor_sha256"] != baseline_execution.get("tensor_sha256"):
            raise RuntimeError("observer latent/decoded tensor hashes differ from frozen baseline")
        return arrays, execution

    try:
        return _with_upstream_imports(args.dit_root, _execute)
    finally:
        torch.set_grad_enabled(prior_grad)


def _expected_trace_shapes() -> dict[str, tuple[int, ...]]:
    step = NUM_SAMPLING_STEPS
    state = (step, BATCH_SIZE, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE)
    component = (step, BATCH_SIZE, TILE_COUNT)
    return {
        "K_component": component,
        "L_component": component,
        "R_component": component,
        "component_log_e": component,
        "component_scale": component,
        "component_weight": (TILE_COUNT,),
        "current_alpha_bar": (step,),
        "current_original_timestep": (step,),
        "effective_nonidentity": (step,),
        "epsilon_current": state,
        "epsilon_shifted": state,
        "final_latents_first_half": (BATCH_SIZE, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE),
        "final_latents_second_half": (BATCH_SIZE, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE),
        "innovation_first_half": state,
        "internal_timestep": (step,),
        "p_mean_first_half": state,
        "p_standard_deviation": state,
        "pred_xstart": state,
        "raw_K_component": component,
        "rho": (step,),
        "run_mixture_log_e": (step,),
        "sample_mixture_log_e": (step, BATCH_SIZE),
        "shifted_alpha_bar": (step,),
        "shifted_internal_timestep": (step,),
        "shifted_original_timestep": (step,),
        "theta": state,
        "tile_bounds_yxyx": (TILE_COUNT, 4),
        "transition_noise_multiplier": (step,),
        "x_t": state,
    }


def _load_trace_exact(path: Path, record: dict[str, Any], root: Path) -> dict[str, np.ndarray]:
    if record.get("relative_path") != path.relative_to(root).as_posix():
        raise RuntimeError("trace relative path changed")
    if not path.is_file() or path.stat().st_size != record.get("bytes") or sha256_file(path) != record.get("sha256"):
        raise RuntimeError("trace file identity failed")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.ascontiguousarray(archive[key]) for key in archive.files}
    if set(arrays) != set(TRACE_DTYPES) or sorted(arrays) != record.get("keys"):
        raise RuntimeError("trace key set changed")
    shapes = _expected_trace_shapes()
    for key, value in arrays.items():
        expected = {
            "shape": list(shapes[key]),
            "dtype": str(TRACE_DTYPES[key]),
            "raw_bytes_sha256": _array_raw_sha256(value),
        }
        if value.shape != shapes[key] or value.dtype != TRACE_DTYPES[key] or record.get("arrays", {}).get(key) != expected:
            raise RuntimeError(f"trace array identity failed: {key}")
        if not np.isfinite(value).all():
            raise RuntimeError(f"trace contains non-finite values: {key}")
    return arrays


def _validate_trace_math(arrays: dict[str, np.ndarray], spec: EvidenceSpec, alpha_total: float) -> dict[str, Any]:
    reverse = np.arange(NUM_SAMPLING_STEPS - 1, -1, -1, dtype=np.int16)
    shifted = spec.shifted_internal_timestep.astype(np.int16)
    if not np.array_equal(arrays["internal_timestep"], reverse):
        raise RuntimeError("trace internal timestep order changed")
    if not np.array_equal(arrays["shifted_internal_timestep"], shifted):
        raise RuntimeError("trace shifted timestep mapping changed")
    if not np.array_equal(arrays["current_original_timestep"], spec.timestep_map[reverse].astype(np.int16)):
        raise RuntimeError("trace current original timesteps changed")
    if not np.array_equal(arrays["shifted_original_timestep"], spec.timestep_map[shifted].astype(np.int16)):
        raise RuntimeError("trace shifted original timesteps changed")
    if not np.array_equal(arrays["tile_bounds_yxyx"], spec.tile_bounds_yxyx):
        raise RuntimeError("fixed tile bounds changed")
    if not np.array_equal(arrays["component_weight"], np.full(TILE_COUNT, 1.0 / TILE_COUNT)):
        raise RuntimeError("component mixture weights changed")
    expected_effective = (shifted != reverse).astype(np.uint8)
    expected_effective[-1] = 0
    if not np.array_equal(arrays["effective_nonidentity"], expected_effective):
        raise RuntimeError("effective mapping flags changed")
    expected_multiplier = (reverse > 0).astype(np.uint8)
    if not np.array_equal(arrays["transition_noise_multiplier"], expected_multiplier):
        raise RuntimeError("terminal/stochastic transition flags changed")
    expected_current_alpha = spec.internal_alpha_bar[reverse]
    expected_shifted_alpha = spec.internal_alpha_bar[shifted]
    if not np.array_equal(arrays["current_alpha_bar"], expected_current_alpha) or not np.array_equal(
        arrays["shifted_alpha_bar"], expected_shifted_alpha
    ):
        raise RuntimeError("trace alpha_bar values differ from the frozen schedule mapping")
    expected_rho = np.sqrt(expected_shifted_alpha / expected_current_alpha)
    if not np.allclose(arrays["rho"], expected_rho, rtol=0.0, atol=2e-16):
        raise RuntimeError("trace normalized-state rho values changed")

    maximum_theta_error = 0.0
    maximum_pred_xstart_error = 0.0
    for row in range(NUM_SAMPLING_STEPS):
        if arrays["effective_nonidentity"][row]:
            expected_theta = (
                -float(arrays["rho"][row])
                * arrays["epsilon_shifted"][row].astype(np.float64)
                / math.sqrt(1.0 - float(arrays["shifted_alpha_bar"][row]))
                + arrays["epsilon_current"][row].astype(np.float64)
                / math.sqrt(1.0 - float(arrays["current_alpha_bar"][row]))
            )
        else:
            if not np.array_equal(arrays["epsilon_shifted"][row], arrays["epsilon_current"][row]):
                raise RuntimeError(f"identity Q mapping changed epsilon at trace row {row}")
            expected_theta = np.zeros_like(arrays["theta"][row])
        theta_error = float(
            np.max(np.abs(expected_theta - arrays["theta"][row]), initial=0.0)
        )
        maximum_theta_error = max(maximum_theta_error, theta_error)
        if theta_error > 2e-13:
            raise RuntimeError(f"cross-scale theta does not reconstruct at trace row {row}")

        alpha_row = float(arrays["current_alpha_bar"][row])
        coefficient_x = np.float32(math.sqrt(1.0 / alpha_row))
        coefficient_epsilon = np.float32(math.sqrt(1.0 / alpha_row - 1.0))
        expected_pred = (
            coefficient_x * arrays["x_t"][row]
            - coefficient_epsilon * arrays["epsilon_current"][row]
        )
        pred_error = float(
            np.max(
                np.abs(
                    expected_pred.astype(np.float64)
                    - arrays["pred_xstart"][row].astype(np.float64)
                ),
                initial=0.0,
            )
        )
        maximum_pred_xstart_error = max(maximum_pred_xstart_error, pred_error)
        if pred_error > 2e-4:
            raise RuntimeError(
                f"captured four-channel epsilon does not reconstruct pred_xstart at row {row}: "
                f"max_abs={pred_error}"
            )
    if not np.array_equal(arrays["x_t"][1:], arrays["p_mean_first_half"][:-1] + arrays["p_standard_deviation"][:-1] * arrays["innovation_first_half"][:-1]):
        # GPU elementwise execution may differ by one ulp from NumPy; a strict
        # tolerance audit follows before accepting that representational gap.
        reconstructed = arrays["p_mean_first_half"][:-1] + arrays["p_standard_deviation"][:-1] * arrays["innovation_first_half"][:-1]
        maximum = float(np.max(np.abs(reconstructed.astype(np.float64) - arrays["x_t"][1:].astype(np.float64))))
        if maximum > 2e-6:
            raise RuntimeError(f"first-half P state chain does not reconstruct: max_abs={maximum}")
    if not np.array_equal(arrays["final_latents_first_half"], arrays["p_mean_first_half"][-1]):
        raise RuntimeError("t=0 final first-half state is not exactly the deterministic P mean")

    masks = fixed_tile_masks(spec.tile_bounds_yxyx)
    cumulative = np.zeros((BATCH_SIZE, TILE_COUNT), dtype=np.float64)
    for row in range(NUM_SAMPLING_STEPS):
        if arrays["effective_nonidentity"][row]:
            raw_K, scale, K, u = construct_predictable_tile_shifts(
                arrays["theta"][row], arrays["p_standard_deviation"][row], masks, spec.guarded_K_cap
            )
            R, L = evaluate_tile_log_lr(u, arrays["innovation_first_half"][row])
        else:
            raw_K = np.zeros((BATCH_SIZE, TILE_COUNT), dtype=np.float64)
            scale = np.ones_like(raw_K)
            K = R = L = np.zeros_like(raw_K)
        for key, expected in (
            ("raw_K_component", raw_K),
            ("component_scale", scale),
            ("K_component", K),
            ("R_component", R),
            ("L_component", L),
        ):
            if not np.allclose(arrays[key][row], expected, rtol=2e-13, atol=2e-13):
                raise RuntimeError(f"operational LR does not reconstruct: {key}/row={row}")
        cumulative = cumulative + L
        if not np.allclose(arrays["component_log_e"][row], cumulative, rtol=0.0, atol=2e-13):
            raise RuntimeError(f"component cumulative log evidence changed at row {row}")
        sample_mix = _logmeanexp(cumulative, axis=1)
        run_mix = float(_logmeanexp(cumulative, axis=(0, 1)))
        if not np.allclose(arrays["sample_mixture_log_e"][row], sample_mix, rtol=0.0, atol=2e-13):
            raise RuntimeError(f"sample mixture evidence changed at row {row}")
        if not math.isclose(float(arrays["run_mixture_log_e"][row]), run_mix, rel_tol=0.0, abs_tol=2e-13):
            raise RuntimeError(f"run mixture evidence changed at row {row}")
    total_K = arrays["K_component"].sum(axis=0, dtype=np.float64)
    if np.any(total_K > TOTAL_K_BUDGET):
        raise RuntimeError("a path/tile component exceeded total K=0.5")
    threshold = math.log(1.0 / alpha_total)
    crossings = np.flatnonzero(arrays["run_mixture_log_e"] >= threshold)
    first_alarm = None if crossings.size == 0 else int(arrays["internal_timestep"][crossings[0]])
    return {
        "all_250_first_half_P_transitions_reconstructed": True,
        "terminal_t0_noise_recorded_but_zero_multiplied": True,
        "cross_scale_theta_reconstructed_from_captured_four_channel_epsilon": True,
        "maximum_theta_reconstruction_absolute_error": maximum_theta_error,
        "pred_xstart_reconstructed_from_captured_current_epsilon": True,
        "maximum_pred_xstart_reconstruction_absolute_error": maximum_pred_xstart_error,
        "operational_LR_reconstructed": True,
        "maximum_total_K_over_image_tile_components": float(total_K.max()),
        "first_run_level_alarm_internal_timestep": first_alarm,
        "final_run_mixture_log_e": float(arrays["run_mixture_log_e"][-1]),
        "running_max_run_mixture_log_e": float(max(0.0, arrays["run_mixture_log_e"].max())),
    }


def validate_output_bundle(root: Path, *, baseline: BaselineRun, spec: EvidenceSpec) -> dict[str, Any]:
    manifest = _read_self_hashed_json(root / "manifest.json", "identity_sha256")
    results = _read_self_hashed_json(root / "results.json", "payload_sha256")
    completion = _read_self_hashed_json(root / "completion.json", "payload_sha256")
    if manifest.get("experiment") != EXPERIMENT or manifest.get("observe_only") is not True or manifest.get("sampling_distribution_P_changed") is not False:
        raise RuntimeError("manifest lost the observe-only P-unchanged contract")
    runner = Path(__file__).resolve()
    if manifest.get("runner", {}).get("sha256") != sha256_file(runner):
        raise RuntimeError("output was produced by a different observer source")
    if manifest.get("frozen_baseline", {}).get("identity_sha256") != baseline.identity_sha256:
        raise RuntimeError("output baseline identity changed")
    fixed_results = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "manifest_identity_sha256": manifest["identity_sha256"],
        "observe_only": True,
        "intervention_count": 0,
        "sampling_distribution_P_changed": False,
        "ideal_marginal_ratio_claimed": False,
        "automatic_image_quality_scoring": False,
    }
    mismatches = {key: (results.get(key), value) for key, value in fixed_results.items() if results.get(key) != value}
    if mismatches:
        raise RuntimeError(f"results scope/identity mismatch: {mismatches}")
    trace_path = root / TRACE_NAME
    arrays = _load_trace_exact(trace_path, results.get("trace", {}), root)
    trace_audit = _validate_trace_math(arrays, spec, float(manifest["anytime_alarm"]["alpha_total"]))
    if results.get("trace_math_audit") != trace_audit:
        raise RuntimeError("persisted trace math audit differs from reconstruction")
    transitions = results.get("execution", {}).get("transitions")
    if not isinstance(transitions, list) or len(transitions) != NUM_SAMPLING_STEPS:
        raise RuntimeError("full transition hash trail is incomplete")
    previous_after = None
    for row, record in enumerate(transitions):
        internal_t = NUM_SAMPLING_STEPS - 1 - row
        if record.get("step_index") != row or record.get("internal_timestep") != internal_t or record.get("noise_draw_ordinal") != row + 1:
            raise RuntimeError(f"transition identity failed at row {row}")
        rng = record.get("rng_state_sha256", {})
        if not (rng.get("before_models") == rng.get("after_current_model") == rng.get("after_shifted_model_and_Q_construction")):
            raise RuntimeError(f"model/Q observation consumed RNG at t={internal_t}")
        hashes = record.get("full_batch_tensor_sha256", {})
        if previous_after is not None and hashes.get("state_before") != previous_after:
            raise RuntimeError(f"full 2B state hash chain broke at t={internal_t}")
        previous_after = hashes.get("state_after")
        if not all(isinstance(value, str) and len(value) == 64 for value in hashes.values()) or len(hashes) != 8:
            raise RuntimeError(f"full batch hash provenance is incomplete at t={internal_t}")
        row_hashes = record.get("first_half_trace_row_raw_sha256", {})
        expected_row_keys = {
            "x_t",
            "pred_xstart",
            "p_mean_first_half",
            "p_standard_deviation",
            "epsilon_current",
            "epsilon_shifted",
            "theta",
            "innovation_first_half",
            "raw_K_component",
            "component_scale",
            "K_component",
            "R_component",
            "L_component",
            "component_log_e",
            "sample_mixture_log_e",
            "run_mixture_log_e",
        }
        if set(row_hashes) != expected_row_keys:
            raise RuntimeError(f"first-half trace-row hash set changed at t={internal_t}")
        for key in expected_row_keys:
            if row_hashes[key] != _array_raw_sha256(arrays[key][row]):
                raise RuntimeError(f"trace row is not bound to execution record: {key}/t={internal_t}")
        if not math.isclose(
            float(record.get("run_mixture_log_e_after_transition")),
            float(arrays["run_mixture_log_e"][row]),
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise RuntimeError(f"transition run-mixture evidence changed at t={internal_t}")
        expected_crossed = float(arrays["run_mixture_log_e"][row]) >= math.log(
            1.0 / float(manifest["anytime_alarm"]["alpha_total"])
        )
        if record.get("run_level_alpha_total_crossed") is not expected_crossed:
            raise RuntimeError(f"transition alpha_total alarm flag changed at t={internal_t}")
    if results.get("execution", {}).get(
        "first_run_level_alarm_internal_timestep"
    ) != trace_audit["first_run_level_alarm_internal_timestep"]:
        raise RuntimeError("execution alarm time differs from reconstructed mixture evidence")
    if results.get("execution", {}).get("rng_state_sha256") != baseline.manifest.get("execution", {}).get("rng_state_sha256"):
        raise RuntimeError("completed observer RNG endpoint differs from baseline")
    if results.get("execution", {}).get("tensor_sha256") != baseline.manifest.get("execution", {}).get("tensor_sha256"):
        raise RuntimeError("completed observer tensor hashes differ from baseline")

    png_records = _collect_png_records(root)
    if results.get("outputs") != png_records or results.get("outputs_sha256") != sha256_json(png_records):
        raise RuntimeError("observer PNG records changed")
    baseline_by_path = {record["relative_path"]: record for record in baseline.output_records}
    if any(record["pixel_sha256"] != baseline_by_path[record["relative_path"]]["pixel_sha256"] for record in png_records):
        raise RuntimeError("one or more observer PNGs are not pixel-identical to frozen baseline")
    expected_files = {
        (root / "manifest.json").resolve(),
        (root / "results.json").resolve(),
        (root / "completion.json").resolve(),
        trace_path.resolve(),
        *((root / relative).resolve() for relative in expected_output_specs()),
    }
    actual_files = {path.resolve() for path in root.rglob("*") if path.is_file()}
    if actual_files != expected_files:
        raise RuntimeError(
            f"output file set changed; missing={sorted(expected_files-actual_files)[:2]}, extra={sorted(actual_files-expected_files)[:2]}"
        )
    fixed_completion = {
        "complete": True,
        "manifest_identity_sha256": manifest["identity_sha256"],
        "manifest_file_sha256": sha256_file(root / "manifest.json"),
        "results_payload_sha256": results["payload_sha256"],
        "results_file_sha256": sha256_file(root / "results.json"),
        "trace_sha256": results["trace"]["sha256"],
        "outputs_sha256": results["outputs_sha256"],
    }
    if any(completion.get(key) != value for key, value in fixed_completion.items()):
        raise RuntimeError("completion links/hashes are invalid")
    return results


def run_real(
    args: argparse.Namespace,
    *,
    baseline: BaselineRun,
    source: dict[str, Any],
    checkpoint: dict[str, Any],
    vae: dict[str, Any],
    spec: EvidenceSpec,
) -> None:
    if args.outdir.exists():
        raise RuntimeError(f"refusing to overwrite existing output path: {args.outdir}")
    manifest = build_manifest(
        args, baseline=baseline, source=source, checkpoint=checkpoint, vae=vae, spec=spec
    )
    started = time.time()
    args.outdir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{args.outdir.name}.staging-", dir=args.outdir.parent) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        atomic_json_dump(manifest, staging / "manifest.json")
        arrays, execution = run_observe(args, spec=spec, baseline=baseline, staging=staging)
        trace_path = staging / TRACE_NAME
        _atomic_npz_dump(arrays, trace_path)
        trace_record = _trace_record(trace_path, arrays, staging)
        trace_math = _validate_trace_math(arrays, spec, args.alpha_total)
        outputs = _collect_png_records(staging)
        baseline_by_path = {record["relative_path"]: record for record in baseline.output_records}
        per_output_equality = [
            {
                "relative_path": record["relative_path"],
                "pixel_equal": record["pixel_sha256"] == baseline_by_path[record["relative_path"]]["pixel_sha256"],
                "observer_pixel_sha256": record["pixel_sha256"],
                "baseline_pixel_sha256": baseline_by_path[record["relative_path"]]["pixel_sha256"],
                "png_file_sha256_equal": record["sha256"] == baseline_by_path[record["relative_path"]]["sha256"],
            }
            for record in outputs
        ]
        if not all(item["pixel_equal"] for item in per_output_equality):
            raise RuntimeError("observer endpoint images differ pixelwise from the frozen baseline")

        per_image = []
        total_K = arrays["K_component"].sum(axis=0, dtype=np.float64)
        for index, class_id in enumerate(CLASS_IDS):
            per_image.append(
                {
                    "batch_index": index,
                    "class_id": class_id,
                    "final_component_log_e": arrays["component_log_e"][-1, index].tolist(),
                    "final_sample_mixture_log_e": float(arrays["sample_mixture_log_e"][-1, index]),
                    "running_max_sample_mixture_log_e": float(max(0.0, arrays["sample_mixture_log_e"][:, index].max())),
                    "total_applied_K_by_fixed_tile": total_K[index].tolist(),
                    "maximum_total_applied_K": float(total_K[index].max()),
                }
            )
        results: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "manifest_identity_sha256": manifest["identity_sha256"],
            "observe_only": True,
            "intervention_count": 0,
            "sampling_distribution_P_changed": False,
            "ideal_marginal_ratio_claimed": False,
            "automatic_image_quality_scoring": False,
            "operational_Q_path_mixture": "uniform over 16 fixed latent tiles per image",
            "run_level_path_mixture": "uniform over all 8 images x 16 tile components",
            "alpha_total": args.alpha_total,
            "trace": trace_record,
            "trace_math_audit": trace_math,
            "per_image": per_image,
            "execution": execution,
            "outputs": outputs,
            "outputs_sha256": sha256_json(outputs),
            "frozen_baseline_pixel_equality": {
                "all_eight_individual_images_pixel_equal": all(
                    item["pixel_equal"] for item in per_output_equality if item["relative_path"] != "sample.png"
                ),
                "official_grid_pixel_equal": next(item["pixel_equal"] for item in per_output_equality if item["relative_path"] == "sample.png"),
                "records": per_output_equality,
            },
            "wall_seconds_before_bundle_validation": time.time() - started,
            "platform": {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python": sys.version,
                "dependencies": dependency_identity(),
                "cuda_device_count_visible": torch.cuda.device_count(),
                "cuda_current_device": torch.cuda.current_device(),
                "cuda_device_name": torch.cuda.get_device_name(torch.cuda.current_device()),
                "cuda_device_capability": list(torch.cuda.get_device_capability()),
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
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
            "trace_sha256": trace_record["sha256"],
            "outputs_sha256": results["outputs_sha256"],
            "finished_unix": time.time(),
            "wall_seconds": time.time() - started,
        }
        completion["payload_sha256"] = _canonical_self_hash(completion, "payload_sha256")
        atomic_json_dump(completion, staging / "completion.json")
        validate_output_bundle(staging, baseline=baseline, spec=spec)
        if args.outdir.exists():
            raise RuntimeError("output target appeared during staging; refusing overwrite")
        os.replace(staging, args.outdir)
    final = validate_output_bundle(args.outdir, baseline=baseline, spec=spec)
    print(json.dumps(final["trace_math_audit"], ensure_ascii=False, indent=2, sort_keys=True))


def run_self_test() -> None:
    if torch.cuda.is_initialized():
        raise RuntimeError("self-test must start without CUDA initialization")
    bounds = fixed_tile_bounds(grid_size=2, height=4, width=4)
    masks_small = fixed_tile_masks(bounds, height=4, width=4)
    if bounds.tolist() != [[0, 0, 2, 2], [0, 2, 2, 4], [2, 0, 4, 2], [2, 2, 4, 4]]:
        raise AssertionError("row-major fixed tile bounds failed")
    if not np.array_equal(masks_small.sum(axis=0), np.ones((1, 4, 4))):
        raise AssertionError("fixed tiles do not partition the plane")

    # Exercise the production 16-tile construction on a small batch.
    generator = np.random.default_rng(123)
    theta = np.ascontiguousarray(generator.normal(size=(2, 4, 32, 32)), dtype=np.float64)
    sigma = np.ascontiguousarray(np.full(theta.shape, 0.2), dtype=np.float32)
    masks = fixed_tile_masks(fixed_tile_bounds())
    raw_K, scale, K, u = construct_predictable_tile_shifts(theta, sigma, masks, 0.002)
    if raw_K.shape != (2, 16) or np.any(K > 0.002 * (1 + 1e-12)) or np.any(scale > 1):
        raise AssertionError("fixed-tile KL cap failed")
    noise = np.ascontiguousarray(generator.normal(size=theta.shape), dtype=np.float32)
    R, L = evaluate_tile_log_lr(u, noise)
    if not np.allclose(L, R - K, rtol=0.0, atol=0.0):
        raise AssertionError("exact Gaussian LR decomposition failed")

    # Monte Carlo E calibration for a fixed predictable 4-component toy shift.
    toy_u = np.zeros((1, 4, 1, 2, 2), dtype=np.float64)
    for component in range(4):
        toy_u[0, component, 0, component // 2, component % 2] = 0.12
    log_values = []
    for _ in range(20_000):
        toy_noise = np.ascontiguousarray(generator.normal(size=(1, 1, 2, 2)), dtype=np.float32)
        _, toy_L = evaluate_tile_log_lr(toy_u, toy_noise)
        log_values.append(float(_logmeanexp(toy_L, axis=(0, 1))))
    empirical = float(np.mean(np.exp(log_values), dtype=np.float64))
    if abs(empirical - 1.0) > 0.01:
        raise AssertionError(f"uniform mixture E calibration failed: {empirical}")

    # The terminal draw is consumed in full 2B shape even though it is masked.
    a = torch.Generator(device="cpu").manual_seed(91)
    x_a = torch.randn((2, 3), generator=a)
    for t in (2, 1, 0):
        full_noise = torch.randn((4, 3), generator=a)
        x_a = x_a + float(t != 0) * full_noise[:2]
    state_a = a.get_state()
    b = torch.Generator(device="cpu").manual_seed(91)
    x_b = torch.randn((2, 3), generator=b)
    for t in (2, 1, 0):
        full_noise = torch.randn((4, 3), generator=b)
        x_b = x_b + float(t != 0) * full_noise[:2]
    if not torch.equal(x_a, x_b) or not torch.equal(state_a, b.get_state()):
        raise AssertionError("2B/t0 RNG replay failed")

    payload = {"experiment": EXPERIMENT, "toy": True}
    payload["payload_sha256"] = _canonical_self_hash(payload, "payload_sha256")
    with tempfile.TemporaryDirectory(prefix="dit-path-evidence-self-test-") as temporary:
        path = Path(temporary) / "self.json"
        atomic_json_dump(payload, path)
        if _read_self_hashed_json(path, "payload_sha256") != payload:
            raise AssertionError("self-hashed JSON roundtrip failed")
    if torch.cuda.is_initialized():
        raise AssertionError("CPU self-test initialized CUDA")
    print(
        "self-test passed: fixed row-major tiles, per-component KL cap, exact LR, "
        f"uniform-mixture E calibration ({empirical:.6f}), 2B/t0 RNG, hashes, CPU-only"
    )


def _paths_overlap(left: Path, right: Path) -> bool:
    left, right = left.resolve(), right.resolve()
    return left == right or left in right.parents or right in left.parents


def build_parser() -> argparse.ArgumentParser:
    data_root = Path(os.environ.get("EQVAE_DATA_ROOT", "/home/zhoushunyu/data/eqvae"))
    default_dit = data_root / "baselines/DiT"
    default_vae = (
        Path.home()
        / ".cache/huggingface/hub/models--stabilityai--sd-vae-ft-mse/snapshots"
        / VAE_REVISION
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2, help="Official batch-of-eight global seed.")
    parser.add_argument("--alpha-total", type=float, default=DEFAULT_ALPHA_TOTAL)
    parser.add_argument("--total-k-budget", type=float, choices=(TOTAL_K_BUDGET,), default=TOTAL_K_BUDGET)
    parser.add_argument("--grid-size", type=int, choices=(GRID_SIZE,), default=GRID_SIZE)
    parser.add_argument("--additive-heat-shift", type=float, choices=(ADDITIVE_NORMALIZED_HEAT_SHIFT,), default=ADDITIVE_NORMALIZED_HEAT_SHIFT)
    parser.add_argument("--baseline-dir", type=Path, default=None)
    parser.add_argument("--dit-root", type=Path, default=default_dit)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--vae-snapshot", type=Path, default=default_vae)
    parser.add_argument("--outdir", type=Path, default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    return parser


def normalize_and_validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not 0 <= args.seed < 1 << 63:
        parser.error("--seed must lie in [0,2^63-1]")
    if not math.isfinite(args.alpha_total) or not 0 < args.alpha_total < 1:
        parser.error("--alpha-total must lie strictly between 0 and 1")
    args.dit_root = args.dit_root.expanduser().absolute().resolve()
    args.checkpoint = (
        (args.dit_root / "pretrained_models" / CHECKPOINT_FILENAME)
        if args.checkpoint is None
        else args.checkpoint.expanduser().absolute().resolve()
    )
    args.vae_snapshot = args.vae_snapshot.expanduser().absolute().resolve()
    data_root = Path(os.environ.get("EQVAE_DATA_ROOT", "/home/zhoushunyu/data/eqvae")).expanduser().absolute().resolve()
    if args.baseline_dir is None:
        args.baseline_dir = data_root / "cross_scale_evidence/dit_imagenet256" / f"official_demo_seed{args.seed}"
    else:
        args.baseline_dir = args.baseline_dir.expanduser().absolute().resolve()
    if args.outdir is None:
        args.outdir = data_root / "cross_scale_evidence/dit_imagenet256_path_evidence" / f"official_demo_seed{args.seed}_dnu1_K0p5_grid4"
    else:
        args.outdir = args.outdir.expanduser().absolute().resolve()
    if args.outdir.exists():
        parser.error(f"no-overwrite target already exists: {args.outdir}")
    protected = {
        "frozen baseline": args.baseline_dir,
        "DiT source/checkpoint": args.dit_root,
        "VAE snapshot": args.vae_snapshot,
        "research repository": Path(__file__).resolve().parent.parent,
    }
    overlaps = [label for label, path in protected.items() if _paths_overlap(args.outdir, path)]
    if overlaps:
        parser.error("--outdir overlaps protected input/source path(s): " + ", ".join(overlaps))


def dry_run(
    args: argparse.Namespace,
    *,
    baseline: BaselineRun,
    source: dict[str, Any],
    checkpoint_probe: dict[str, Any],
    vae: dict[str, Any],
    spec: EvidenceSpec,
) -> None:
    blockers = []
    if not checkpoint_probe["exists"]:
        blockers.append("checkpoint file missing")
    elif not checkpoint_probe["size_matches"]:
        blockers.append("checkpoint size mismatch/incomplete download")
    if not checkpoint_probe["sha256_pinned"]:
        blockers.append("checkpoint SHA is not pinned")
    payload = {
        "status": "dry-run",
        "experiment": EXPERIMENT,
        "observe_only": True,
        "gpu_model_loaded": False,
        "seed": args.seed,
        "class_ids": list(CLASS_IDS),
        "baseline": {
            "root": str(baseline.root),
            "identity_sha256": baseline.identity_sha256,
            "strict_completion_validated": True,
        },
        "source": source,
        "checkpoint_probe": checkpoint_probe,
        "vae": vae,
        "operational_Q": {
            "delta_nu": ADDITIVE_NORMALIZED_HEAT_SHIFT,
            "grid": [GRID_SIZE, GRID_SIZE],
            "components": TILE_COUNT,
            "total_K_per_component": TOTAL_K_BUDGET,
            "effective_stochastic_steps": spec.effective_nonidentity_steps,
            "fixed_K_allowance": spec.fixed_K_allowance,
            "alpha_total": args.alpha_total,
        },
        "P_contract": {
            "full_2B_noise_draws": NUM_SAMPLING_STEPS,
            "t0_draw_consumed": True,
            "captured_epsilon_channels": 4,
            "guided_epsilon_channels": 3,
            "final_pixel_equality_required": True,
        },
        "outdir": str(args.outdir),
        "real_run_blockers": blockers,
        "static_inputs_ready": not blockers,
        "cuda_available": torch.cuda.is_available(),
        "canonical_command": canonical_command(args),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        run_self_test()
        return 0
    normalize_and_validate_args(args, parser)
    source = validate_repository(args.dit_root, args.checkpoint)
    vae = validate_vae_snapshot(args.vae_snapshot)
    alpha, timestep_map = load_schedule(args.dit_root)
    spec = build_evidence_spec(alpha, timestep_map, total_K_budget=args.total_k_budget)
    if args.dry_run:
        baseline = validate_baseline_run(args.baseline_dir, seed=args.seed, source=source, vae=vae)
        dry_run(
            args,
            baseline=baseline,
            source=source,
            checkpoint_probe=checkpoint_dry_probe(args.checkpoint),
            vae=vae,
            spec=spec,
        )
        return 0
    checkpoint = validate_checkpoint(args.checkpoint)
    baseline = validate_baseline_run(
        args.baseline_dir,
        seed=args.seed,
        source=source,
        checkpoint=checkpoint,
        vae=vae,
    )
    run_real(args, baseline=baseline, source=source, checkpoint=checkpoint, vae=vae, spec=spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
