#!/usr/bin/env python3
"""Observe a predictable local-Q path likelihood ratio on frozen ADM64 P paths.

This runner is deliberately diagnostic-only.  It executes the frozen official
classifier-guided 250-step stochastic ADM sampler exactly as implemented by
``reproduce_adm64_guided.py`` and observes all 249 stochastic reverse
transitions.  It never rejects, rolls back, resamples, or changes a P state.

At every stochastic transition, a single fixed cross-scale alternative with
additive normalized-heat shift ``Delta nu = 1`` is evaluated.  Before the
transition innovation is drawn, the highest-energy contiguous tile of
``sigma * theta`` is selected by
``adm64_local_path_evidence.predictable_max_energy_tile_shift``.  The
same-covariance alternative is restricted to that tile.  For effective
non-identity heat mappings, every step receives the same predeclared
conditional-KL allowance

    K_allow = K_total / N_effective.

Unused allowance is not carried to later steps, and each path is checked to
satisfy ``sum_k K_k <= K_total``.  Identity mappings have Q=P and zero
increment.  The actual P innovation is drawn only after tile selection and
tilt construction, then the operational finite-step likelihood ratio is

    Delta log E_k = <u_k, epsilon_k> - 0.5 ||u_k||^2,
    u_k = Sigma_k^{-1/2} (mu_Q,k - mu_P,k).

This exactness is only relative to the two implemented, learned and
discretized Markov transitions P and Q.  It is not a claim that the product is
the ideal heat-flow marginal density ratio, and it does not automatically
inherit a Self-Guidance scale interpretation.  No image-quality conclusion is
made by this runner.

For each path, a compressed NPZ stores every stochastic step's pre-transition
``x_t``, ``pred_xstart``, current/shifted epsilon predictions, ``theta``, exact
P standard deviation, actual innovation, spatial ``K_map``, ``R_map``, and
``L_map=R_map-K_map``, plus tile and heat-mapping metadata.
The map sums are the canonical recorded scalar K, R, and log-LR values and are
strictly revalidated on resume.  The decoded endpoint must be pixel-identical
to a separately completed frozen ADM baseline.  Manifests, signals, PNGs, and
NPZ tensors are all fail-closed and self-authenticating.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

# Deterministic cuBLAS must be configured before importing torch/CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo

try:  # Support both CLI and package-style imports.
    from .adm64_local_path_evidence import predictable_max_energy_tile_shift
    from .adm64_path_evidence import (
        normalized_heat_score_pullback_difference,
        same_covariance_log_lr_from_noise,
    )
    from .observe_adm64_cross_scale_evidence import (
        BaselineReference,
        ComponentSpec,
        _canonical_payload_sha,
        _close,
        build_component_specs,
        decoded_pixels,
        load_baseline_reference,
        original_schedule_and_timestep_map,
        pixel_sha256,
    )
    from .reproduce_adm64_guided import (
        CLASSIFIER_CHECKPOINT,
        CLASSIFIER_SCALE,
        DIFFUSION_CHECKPOINT,
        GUIDED_DIFFUSION_REVISION,
        IMAGE_SIZE,
        NUM_SPACED_STEPS,
        OFFICIAL_CLASSIFIER_CONFIG,
        OFFICIAL_MODEL_CONFIG,
        Pair,
        Protocol,
        SeedRandomStreams,
        atomic_json_dump,
        chunks,
        configure_determinism,
        git_revision,
        git_tracked_dirty,
        load_official_models,
        make_guided_functions,
        pair_path as baseline_pair_path,
        parse_int_spec,
        pixels_from_sample,
        protocol_from_args,
        sample_batch_invariant,
        sample_stream_seed,
        sha256_file,
        sha256_json,
        sha256_python_tree,
        validate_checkpoint,
        validate_existing_completion,
    )
except ImportError:  # pragma: no cover - exercised by the CLI entry point.
    from adm64_local_path_evidence import predictable_max_energy_tile_shift
    from adm64_path_evidence import (
        normalized_heat_score_pullback_difference,
        same_covariance_log_lr_from_noise,
    )
    from observe_adm64_cross_scale_evidence import (
        BaselineReference,
        ComponentSpec,
        _canonical_payload_sha,
        _close,
        build_component_specs,
        decoded_pixels,
        load_baseline_reference,
        original_schedule_and_timestep_map,
        pixel_sha256,
    )
    from reproduce_adm64_guided import (
        CLASSIFIER_CHECKPOINT,
        CLASSIFIER_SCALE,
        DIFFUSION_CHECKPOINT,
        GUIDED_DIFFUSION_REVISION,
        IMAGE_SIZE,
        NUM_SPACED_STEPS,
        OFFICIAL_CLASSIFIER_CONFIG,
        OFFICIAL_MODEL_CONFIG,
        Pair,
        Protocol,
        SeedRandomStreams,
        atomic_json_dump,
        chunks,
        configure_determinism,
        git_revision,
        git_tracked_dirty,
        load_official_models,
        make_guided_functions,
        pair_path as baseline_pair_path,
        parse_int_spec,
        pixels_from_sample,
        protocol_from_args,
        sample_batch_invariant,
        sample_stream_seed,
        sha256_file,
        sha256_json,
        sha256_python_tree,
        validate_checkpoint,
        validate_existing_completion,
    )


EXPERIMENT = "adm64_local_cross_scale_path_evidence_observe_only"
SCHEMA_VERSION = 1
ADDITIVE_HEAT_SHIFT = 1.0
TOTAL_K_BUDGET_CHOICES = (0.5, 1.0)
DEFAULT_TOTAL_K_BUDGET = 0.5
DEFAULT_GRID_SIZE = 4
STOCHASTIC_INTERNAL_TIMESTEPS = tuple(range(NUM_SPACED_STEPS - 1, 0, -1))
TRACE_KEYS = (
    "K_map",
    "K_scalar",
    "L_map",
    "L_scalar",
    "R_map",
    "R_scalar",
    "active_K_allowance",
    "cumulative_log_e",
    "current_alpha_bar",
    "current_original_timestep",
    "effective_nonidentity",
    "epsilon_current",
    "epsilon_shifted",
    "innovation",
    "internal_timestep",
    "local_scale",
    "p_standard_deviation",
    "pred_xstart",
    "raw_local_K",
    "rho",
    "selected_energy_fraction",
    "shifted_alpha_bar",
    "shifted_model_evaluated",
    "shifted_original_timestep",
    "theta",
    "tile_bounds_yxyx",
    "tile_index",
    "x_t",
)
TRACE_DTYPES = {
    "K_map": np.dtype(np.float64),
    "K_scalar": np.dtype(np.float64),
    "L_map": np.dtype(np.float64),
    "L_scalar": np.dtype(np.float64),
    "R_map": np.dtype(np.float64),
    "R_scalar": np.dtype(np.float64),
    "active_K_allowance": np.dtype(np.float64),
    "cumulative_log_e": np.dtype(np.float64),
    "current_alpha_bar": np.dtype(np.float64),
    "current_original_timestep": np.dtype(np.int16),
    "effective_nonidentity": np.dtype(np.uint8),
    "epsilon_current": np.dtype(np.float32),
    "epsilon_shifted": np.dtype(np.float32),
    "innovation": np.dtype(np.float32),
    "internal_timestep": np.dtype(np.int16),
    "local_scale": np.dtype(np.float64),
    "p_standard_deviation": np.dtype(np.float32),
    "pred_xstart": np.dtype(np.float32),
    "raw_local_K": np.dtype(np.float64),
    "rho": np.dtype(np.float64),
    "selected_energy_fraction": np.dtype(np.float64),
    "shifted_alpha_bar": np.dtype(np.float64),
    "shifted_model_evaluated": np.dtype(np.uint8),
    "shifted_original_timestep": np.dtype(np.int16),
    "theta": np.dtype(np.float64),
    "tile_bounds_yxyx": np.dtype(np.int16),
    "tile_index": np.dtype(np.int16),
    "x_t": np.dtype(np.float32),
}
if set(TRACE_KEYS) != set(TRACE_DTYPES):  # pragma: no cover - source-edit guard.
    raise AssertionError("every local-Q trace key must have one exact declared dtype")


@dataclass(frozen=True)
class LocalEvidenceSpec:
    component: ComponentSpec
    effective_nonidentity_steps: int
    fixed_K_allowance: float
    guarded_step_cap: float


@dataclass
class ObservedBatch:
    final_states: torch.Tensor
    traces: list[dict[str, np.ndarray]]
    summaries: list[dict[str, Any]]
    accounting: dict[str, int]


def _array_raw_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _tensor_numpy(
    tensor: torch.Tensor, *, dtype: np.dtype[Any] | None = None
) -> np.ndarray:
    array = np.ascontiguousarray(tensor.detach().cpu().numpy())
    if dtype is not None:
        array = np.ascontiguousarray(array, dtype=dtype)
    return array


def _row_sum_float64(array: np.ndarray) -> np.ndarray:
    if array.ndim < 2 or array.dtype != np.float64:
        raise ValueError("row-wise canonical sums require a float64 array")
    return np.ascontiguousarray(
        array.reshape(array.shape[0], -1).sum(axis=1, dtype=np.float64)
    )


def _atomic_npz_dump(arrays: dict[str, np.ndarray], path: Path) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite tensor trace: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _trace_file_record(
    path: Path, arrays: dict[str, np.ndarray], output_dir: Path
) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(output_dir).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
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


def image_path(output_dir: Path, pair: Pair) -> Path:
    return output_dir / "images" / f"class_{pair[0]:04d}" / f"{pair[1]:019d}.png"


def signal_path(output_dir: Path, pair: Pair) -> Path:
    return output_dir / "signals" / f"class_{pair[0]:04d}" / f"{pair[1]:019d}.json"


def trace_path(output_dir: Path, pair: Pair) -> Path:
    return output_dir / "traces" / f"class_{pair[0]:04d}" / f"{pair[1]:019d}.npz"


def _paths_overlap(left: Path, right: Path) -> bool:
    left_resolved, right_resolved = left.resolve(), right.resolve()
    return (
        left_resolved == right_resolved
        or left_resolved in right_resolved.parents
        or right_resolved in left_resolved.parents
    )


def _assert_output_isolated(args: argparse.Namespace) -> None:
    protected = {
        "frozen baseline": args.baseline_dir,
        "guided-diffusion source/weights": args.guided_diffusion_root,
        "diffusion checkpoint": args.model_path,
        "classifier checkpoint": args.classifier_path,
        "research source tree": Path(__file__).resolve().parent.parent,
    }
    overlaps = [
        label
        for label, path in protected.items()
        if _paths_overlap(args.output_dir, path)
    ]
    if overlaps:
        raise ValueError(
            "local-Q output directory overlaps protected input/source path(s): "
            + ", ".join(overlaps)
        )


def build_local_spec(
    original_alpha_bar: np.ndarray,
    timestep_map: np.ndarray,
    total_K_budget: float,
) -> LocalEvidenceSpec:
    if total_K_budget not in TOTAL_K_BUDGET_CHOICES:
        raise ValueError(
            f"total K budget must be one of {TOTAL_K_BUDGET_CHOICES}, found {total_K_budget}"
        )
    components = build_component_specs(
        original_alpha_bar,
        timestep_map,
        (ADDITIVE_HEAT_SHIFT,),
        checkpoints=STOCHASTIC_INTERNAL_TIMESTEPS,
    )
    if len(components) != 1:
        raise AssertionError("single-scale local Q must have exactly one component")
    component = components[0]
    mapping = component.mapping
    effective = int(np.count_nonzero(mapping.shifted_timestep != mapping.current_timestep))
    if effective <= 0 or effective > len(STOCHASTIC_INTERNAL_TIMESTEPS):
        raise RuntimeError(f"invalid effective non-identity step count: {effective}")
    allowance = float(total_K_budget) / effective
    # Reserve only a 2e-12 relative numerical margin.  This prevents reduction
    # roundoff from making a stored finite-precision sum exceed the declared
    # mathematical budget; it is fixed in advance and is never redistributed.
    guarded_cap = allowance * (1.0 - 2e-12)
    return LocalEvidenceSpec(component, effective, allowance, guarded_cap)


def _new_trace_lists() -> dict[str, list[np.ndarray | float | int]]:
    derived = {"K_scalar", "R_scalar", "L_scalar", "cumulative_log_e"}
    return {key: [] for key in TRACE_KEYS if key not in derived}


def _finalize_trace(
    lists: dict[str, list[np.ndarray | float | int]],
    *,
    spec: LocalEvidenceSpec,
    total_K_budget: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    count = len(STOCHASTIC_INTERNAL_TIMESTEPS)
    if any(len(values) != count for values in lists.values()):
        missing = {key: len(values) for key, values in lists.items() if len(values) != count}
        raise AssertionError(f"incomplete stochastic trace: {missing}")

    stack_keys = {
        "x_t",
        "pred_xstart",
        "theta",
        "innovation",
        "p_standard_deviation",
        "epsilon_current",
        "epsilon_shifted",
        "K_map",
        "R_map",
        "L_map",
        "tile_bounds_yxyx",
    }
    arrays: dict[str, np.ndarray] = {}
    for key, values in lists.items():
        if key in stack_keys:
            arrays[key] = np.ascontiguousarray(np.stack(values, axis=0))
        else:
            arrays[key] = np.ascontiguousarray(np.asarray(values))

    for key in ("K_map", "R_map", "L_map", "theta"):
        arrays[key] = np.ascontiguousarray(arrays[key], dtype=np.float64)
    for key in (
        "x_t",
        "pred_xstart",
        "innovation",
        "p_standard_deviation",
        "epsilon_current",
        "epsilon_shifted",
    ):
        arrays[key] = np.ascontiguousarray(arrays[key], dtype=np.float32)
    for key in (
        "active_K_allowance",
        "current_alpha_bar",
        "local_scale",
        "raw_local_K",
        "rho",
        "selected_energy_fraction",
        "shifted_alpha_bar",
    ):
        arrays[key] = np.ascontiguousarray(arrays[key], dtype=np.float64)
    for key in ("current_original_timestep", "internal_timestep", "shifted_original_timestep"):
        arrays[key] = np.ascontiguousarray(arrays[key], dtype=np.int16)
    arrays["tile_bounds_yxyx"] = np.ascontiguousarray(
        arrays["tile_bounds_yxyx"], dtype=np.int16
    )
    arrays["tile_index"] = np.ascontiguousarray(arrays["tile_index"], dtype=np.int16)
    for key in ("effective_nonidentity", "shifted_model_evaluated"):
        arrays[key] = np.ascontiguousarray(arrays[key], dtype=np.uint8)

    # These are canonical scalar back-labels: validation repeats precisely the
    # same row-major float64 reduction on the persisted maps.
    arrays["K_scalar"] = _row_sum_float64(arrays["K_map"])
    arrays["R_scalar"] = _row_sum_float64(arrays["R_map"])
    arrays["L_scalar"] = _row_sum_float64(arrays["L_map"])
    arrays["cumulative_log_e"] = np.ascontiguousarray(
        np.cumsum(arrays["L_scalar"], dtype=np.float64)
    )
    arrays = {key: arrays[key] for key in sorted(arrays)}

    total_K = float(arrays["K_scalar"].sum(dtype=np.float64))
    if total_K > total_K_budget:
        raise AssertionError(f"local-Q total K budget exceeded: {total_K} > {total_K_budget}")
    if np.any(arrays["K_scalar"] > arrays["active_K_allowance"]):
        raise AssertionError("a local-Q step exceeded its fixed K allowance")
    effective_count = int(arrays["effective_nonidentity"].sum(dtype=np.int64))
    if effective_count != spec.effective_nonidentity_steps:
        raise AssertionError("effective non-identity accounting changed")
    final_log_e = float(arrays["cumulative_log_e"][-1])
    running_max = float(max(0.0, float(arrays["cumulative_log_e"].max())))
    summary = {
        "stochastic_step_count": count,
        "effective_nonidentity_step_count": effective_count,
        "shifted_model_evaluation_count": int(
            arrays["shifted_model_evaluated"].sum(dtype=np.int64)
        ),
        "total_K_budget": float(total_K_budget),
        "fixed_K_allowance_per_effective_step": spec.fixed_K_allowance,
        "guarded_numerical_cap_per_effective_step": spec.guarded_step_cap,
        "total_applied_K": total_K,
        "unused_K_budget": float(total_K_budget - total_K),
        "final_cumulative_log_e": final_log_e,
        "running_max_log_e": running_max,
        "intervention_count": 0,
    }
    return arrays, summary


def sample_local_observe_batch(
    diffusion: Any,
    model: torch.nn.Module,
    cond_fn: Callable[..., torch.Tensor],
    pairs: Sequence[Pair],
    *,
    device: torch.device,
    original_alpha_bar: np.ndarray,
    timestep_map: np.ndarray,
    spec: LocalEvidenceSpec,
    total_K_budget: float,
    grid_size: int,
    channels: int = 3,
    image_size: int = IMAGE_SIZE,
    dtype: torch.dtype = torch.float32,
) -> ObservedBatch:
    """Sample P unchanged and record all-step predictable local Q/P evidence."""

    if not pairs:
        raise ValueError("cannot sample an empty pair batch")
    if diffusion.num_timesteps != len(timestep_map):
        raise ValueError("diffusion and supplied timestep map have different lengths")
    if image_size % grid_size:
        raise ValueError("image size must be divisible by grid size")
    if spec.component.additive_heat_shift != ADDITIVE_HEAT_SHIFT:
        raise ValueError("local runner supports only the frozen Delta-nu=1 component")
    expected_stochastic = tuple(range(diffusion.num_timesteps - 1, 0, -1))
    if (
        diffusion.num_timesteps == NUM_SPACED_STEPS
        and expected_stochastic != STOCHASTIC_INTERNAL_TIMESTEPS
    ):
        raise AssertionError("frozen ADM stochastic schedule changed")
    if len(spec.component.mapping.current_timestep) != len(expected_stochastic):
        raise ValueError("local evidence mapping length does not cover every stochastic step")

    streams = SeedRandomStreams(device, pairs)
    states = torch.cat(
        [
            streams.randn(index, (1, channels, image_size, image_size), dtype)
            for index in range(len(pairs))
        ],
        dim=0,
    )
    trace_lists = [_new_trace_lists() for _ in pairs]
    shifted_evaluations = 0
    current_evaluations = 0
    mapping = spec.component.mapping

    for internal_t in range(diffusion.num_timesteps - 1, -1, -1):
        next_states: list[torch.Tensor] = []
        for path_index, (class_id, _) in enumerate(pairs):
            x = states[path_index : path_index + 1]
            t = torch.tensor([internal_t], dtype=torch.long, device=device)
            y = torch.tensor([class_id], dtype=torch.long, device=device)
            model_kwargs = {"y": y}
            raw_box: list[torch.Tensor] = []
            raw_timestep_box: list[int] = []

            def capturing_model_fn(
                x_in: torch.Tensor,
                original_t: torch.Tensor,
                y: torch.Tensor | None = None,
            ) -> torch.Tensor:
                if y is None:
                    raise ValueError("class label y is required")
                raw = model(x_in, original_t, y)
                raw_box.append(raw)
                raw_timestep_box.append(int(original_t.item()))
                return raw

            with torch.no_grad():
                out = diffusion.p_mean_variance(
                    capturing_model_fn,
                    x,
                    t,
                    clip_denoised=True,
                    model_kwargs=model_kwargs,
                )
                current_evaluations += 1
                if len(raw_box) != 1:
                    raise AssertionError("P p_mean_variance did not make exactly one U-Net call")
                expected_original_t = int(timestep_map[internal_t])
                if raw_timestep_box != [expected_original_t]:
                    raise AssertionError(
                        f"wrapped current timestep mismatch: {raw_timestep_box} != {expected_original_t}"
                    )
                raw_current = raw_box[0]
                if raw_current.shape[1] != channels * 2:
                    raise AssertionError("learned-variance U-Net must return exactly 2*C channels")
                epsilon_current = raw_current[:, :channels]
                guided_mean = diffusion.condition_mean(
                    cond_fn,
                    out,
                    x,
                    t,
                    model_kwargs=model_kwargs,
                )

                if internal_t > 0:
                    step_index = diffusion.num_timesteps - 1 - internal_t
                    current_original_t = int(mapping.current_timestep[step_index])
                    shifted_original_t = int(mapping.shifted_timestep[step_index])
                    if current_original_t != expected_original_t:
                        raise AssertionError("precomputed heat mapping is inconsistent")
                    alpha_current = float(original_alpha_bar[current_original_t])
                    alpha_shifted = float(original_alpha_bar[shifted_original_t])
                    rho = math.sqrt(alpha_shifted / alpha_current)
                    effective = shifted_original_t != current_original_t
                    if effective:
                        shifted_input = x * rho
                        shifted_t = torch.tensor(
                            [shifted_original_t], dtype=torch.long, device=device
                        )
                        raw_shifted = model(shifted_input, shifted_t, y)
                        shifted_evaluations += 1
                        if raw_shifted.shape[1] != channels * 2:
                            raise AssertionError("shifted U-Net output must have exactly 2*C channels")
                        epsilon_shifted = raw_shifted[:, :channels]
                    else:
                        epsilon_shifted = epsilon_current

                    theta = normalized_heat_score_pullback_difference(
                        epsilon_current,
                        epsilon_shifted,
                        torch.tensor([alpha_current], dtype=torch.float64, device=device),
                        torch.tensor([alpha_shifted], dtype=torch.float64, device=device),
                    )
                    p_sigma = torch.exp(0.5 * out["log_variance"])

                    # Critical predictability boundary: selection and the full
                    # whitened Q shift are fixed here, before streams.randn.
                    selected = predictable_max_energy_tile_shift(
                        theta,
                        p_sigma,
                        grid_size=grid_size,
                        max_conditional_kl=spec.guarded_step_cap,
                    )
                    whitened_shift = selected.whitened_shift

                    # This is the first access to the current transition's P
                    # innovation.  The observer uses the same path-owned stream
                    # and exact draw count as the frozen baseline.
                    noise = streams.randn(path_index, x.shape, dtype)
                    x_next = guided_mean + p_sigma * noise

                    u64 = whitened_shift.to(torch.float64)
                    noise64 = noise.to(torch.float64)
                    K_map_t = 0.5 * u64.square().sum(dim=1)[0]
                    R_map_t = (u64 * noise64).sum(dim=1)[0]
                    L_map_t = R_map_t - K_map_t
                    K_map = _tensor_numpy(K_map_t, dtype=np.float64)
                    R_map = _tensor_numpy(R_map_t, dtype=np.float64)
                    L_map = _tensor_numpy(L_map_t, dtype=np.float64)
                    K_scalar = float(K_map.reshape(-1).sum(dtype=np.float64))
                    R_scalar = float(R_map.reshape(-1).sum(dtype=np.float64))
                    L_scalar = float(L_map.reshape(-1).sum(dtype=np.float64))
                    exact = same_covariance_log_lr_from_noise(whitened_shift, noise)
                    if not _close(K_scalar, float(exact.conditional_kl.item()), 2e-11):
                        raise AssertionError("K_map does not reconstruct exact operational KL")
                    if not _close(R_scalar, float(exact.innovation_projection.item()), 2e-11):
                        raise AssertionError("R_map does not reconstruct exact innovation projection")
                    if not _close(L_scalar, float(exact.value.item()), 2e-11):
                        raise AssertionError("L_map does not reconstruct exact operational log LR")

                    lists = trace_lists[path_index]
                    lists["internal_timestep"].append(internal_t)
                    lists["current_original_timestep"].append(current_original_t)
                    lists["shifted_original_timestep"].append(shifted_original_t)
                    lists["current_alpha_bar"].append(alpha_current)
                    lists["shifted_alpha_bar"].append(alpha_shifted)
                    lists["rho"].append(rho)
                    lists["effective_nonidentity"].append(int(effective))
                    lists["shifted_model_evaluated"].append(int(effective))
                    lists["active_K_allowance"].append(
                        spec.fixed_K_allowance if effective else 0.0
                    )
                    lists["x_t"].append(_tensor_numpy(x[0], dtype=np.float32))
                    lists["pred_xstart"].append(
                        _tensor_numpy(out["pred_xstart"][0], dtype=np.float32)
                    )
                    lists["theta"].append(_tensor_numpy(theta[0], dtype=np.float64))
                    lists["epsilon_current"].append(
                        _tensor_numpy(epsilon_current[0], dtype=np.float32)
                    )
                    lists["epsilon_shifted"].append(
                        _tensor_numpy(epsilon_shifted[0], dtype=np.float32)
                    )
                    lists["innovation"].append(_tensor_numpy(noise[0], dtype=np.float32))
                    lists["p_standard_deviation"].append(
                        _tensor_numpy(p_sigma[0], dtype=np.float32)
                    )
                    lists["K_map"].append(K_map)
                    lists["R_map"].append(R_map)
                    lists["L_map"].append(L_map)
                    lists["raw_local_K"].append(float(selected.raw_kl.item()))
                    lists["local_scale"].append(float(selected.scale.item()))
                    lists["tile_index"].append(int(selected.tile_index.item()))
                    lists["tile_bounds_yxyx"].append(
                        _tensor_numpy(selected.tile_bounds_yxyx[0], dtype=np.int16)
                    )
                    lists["selected_energy_fraction"].append(
                        float(selected.selected_energy_fraction.item())
                    )
                else:
                    # Deterministic t=0 remains P=Q and is intentionally outside
                    # the 249-step likelihood-ratio trace.
                    x_next = guided_mean
            next_states.append(x_next.detach())
        states = torch.cat(next_states, dim=0)

    expected_draws = diffusion.num_timesteps
    if streams.draw_counts != [expected_draws] * len(pairs):
        raise AssertionError(
            f"observer changed P RNG consumption: {streams.draw_counts} != {expected_draws}"
        )
    traces: list[dict[str, np.ndarray]] = []
    summaries: list[dict[str, Any]] = []
    for lists in trace_lists:
        trace, summary = _finalize_trace(
            lists, spec=spec, total_K_budget=total_K_budget
        )
        traces.append(trace)
        summaries.append(summary)
    return ObservedBatch(
        final_states=states,
        traces=traces,
        summaries=summaries,
        accounting={
            "reverse_steps": int(diffusion.num_timesteps),
            "stochastic_reverse_steps": int(diffusion.num_timesteps - 1),
            "deterministic_final_steps": 1,
            "gaussian_draws_per_path_including_initial": expected_draws,
            "current_unet_evaluations_per_path": current_evaluations // len(pairs),
            "shifted_unet_evaluations_per_path": shifted_evaluations // len(pairs),
            "neural_eval_batch_size": 1,
            "interventions": 0,
        },
    )


def _mapping_manifest_record(spec: LocalEvidenceSpec) -> dict[str, Any]:
    mapping = spec.component.mapping
    return {
        "additive_normalized_heat_shift": ADDITIVE_HEAT_SHIFT,
        "internal_timesteps_reverse_order": list(STOCHASTIC_INTERNAL_TIMESTEPS),
        "current_original_timestep": mapping.current_timestep.tolist(),
        "shifted_original_timestep": mapping.shifted_timestep.tolist(),
        "current_heat_variance": mapping.current_heat_variance.tolist(),
        "target_heat_variance": mapping.target_heat_variance.tolist(),
        "shifted_heat_variance": mapping.shifted_heat_variance.tolist(),
        "actual_heat_shift": mapping.actual_heat_shift.tolist(),
        "absolute_mapping_error": mapping.absolute_mapping_error.tolist(),
        "effective_nonidentity_step_count": spec.effective_nonidentity_steps,
        "identity_step_count": len(STOCHASTIC_INTERNAL_TIMESTEPS)
        - spec.effective_nonidentity_steps,
    }


def build_manifest(
    args: argparse.Namespace,
    protocol: Protocol,
    device: torch.device,
    diffusion_checkpoint: dict[str, Any],
    classifier_checkpoint: dict[str, Any],
    baseline: BaselineReference,
    original_alpha_bar: np.ndarray,
    timestep_map: np.ndarray,
    spec: LocalEvidenceSpec,
) -> dict[str, Any]:
    source_root = args.guided_diffusion_root.resolve()
    revision = git_revision(source_root)
    dirty = git_tracked_dirty(source_root)
    if revision != GUIDED_DIFFUSION_REVISION:
        raise RuntimeError(
            f"guided-diffusion revision mismatch: {revision} != {GUIDED_DIFFUSION_REVISION}"
        )
    if dirty:
        raise RuntimeError("guided-diffusion has tracked source edits")
    if dirty is None:
        raise RuntimeError("cannot verify guided-diffusion tracked-source cleanliness")

    runner = Path(__file__).resolve()
    baseline_runner = runner.with_name("reproduce_adm64_guided.py")
    observe_runner = runner.with_name("observe_adm64_cross_scale_evidence.py")
    evidence_primitives = runner.with_name("adm64_path_evidence.py")
    local_primitives = runner.with_name("adm64_local_path_evidence.py")
    alpha_sha = hashlib.sha256(
        np.ascontiguousarray(original_alpha_bar, dtype=np.float64).tobytes(order="C")
    ).hexdigest()
    mapping_record = _mapping_manifest_record(spec)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "role": "observe_only_no_intervention_no_rejection_no_rollback_no_resampling",
        "protocol": args.protocol,
        "class_ids": list(protocol.class_ids),
        "seeds": list(protocol.seeds),
        "pair_order": "class_major_then_seed_major",
        "pair_set_sha256": sha256_json(
            [[class_id, seed] for class_id, seed in protocol.pairs]
        ),
        "sample_count": len(protocol.pairs),
        "checkpoints": {
            "diffusion": diffusion_checkpoint,
            "classifier": classifier_checkpoint,
        },
        "pure_p_baseline": {
            "root": str(baseline.root),
            "manifest_identity_sha256": baseline.manifest_identity_sha256,
            "runner_sha256": baseline.runner_sha256,
            "pair_set_sha256": baseline.pair_set_sha256,
            "acceptance_rule": "decoded observer endpoint must be pixel-identical to frozen P baseline",
        },
        "official_model_config": OFFICIAL_MODEL_CONFIG,
        "official_classifier_config": OFFICIAL_CLASSIFIER_CONFIG,
        "p_sampler": {
            "name": "OpenAI classifier-guided ancestral DDPM",
            "classifier_scale": CLASSIFIER_SCALE,
            "timestep_respacing": "250",
            "clip_denoised": True,
            "stochastic_reverse_steps": NUM_SPACED_STEPS - 1,
            "deterministic_final_step": "t=0 P mean, no Gaussian and no LR",
            "state_and_innovation_dtype": "torch.float32",
        },
        "local_operational_Q": {
            "initial_distribution": "Q and P share the exact same initial standard Gaussian; E_0=1",
            "mapping": mapping_record,
            "mapping_sha256": sha256_json(mapping_record),
            "grid_size_per_axis": args.grid_size,
            "tile_shape": [IMAGE_SIZE // args.grid_size, IMAGE_SIZE // args.grid_size],
            "tile_rule": (
                "argmax row-major contiguous-tile energy sum((sigma*theta)^2), "
                "selected before the current innovation is drawn"
            ),
            "theta": (
                "rho*(-epsilon_shift/sqrt(1-alpha_bar_shift))"
                "-(-epsilon_current/sqrt(1-alpha_bar_current))"
            ),
            "same_heat_state": "shifted U-Net input rho*x, rho=sqrt(alpha_shift/alpha_current)",
            "q_mean": "mu_Q=mu_P+sigma*(localized whitened shift)",
            "q_covariance": "exactly the implemented P learned diagonal covariance",
            "total_K_budget": args.total_K_budget,
            "fixed_K_allowance_per_effective_step": spec.fixed_K_allowance,
            "guarded_numerical_cap_per_effective_step": spec.guarded_step_cap,
            "budget_policy": (
                "same fixed allowance at each predeclared effective non-identity step; "
                "unused allowance is discarded and never carried forward"
            ),
            "collapse_guard_diagnostics": {
                "pathwise_whitened_squared_shift_bound_D": 2.0 * args.total_K_budget,
                "likelihood_ratio_second_moment_upper_proxy": math.exp(
                    2.0 * args.total_K_budget
                ),
                "importance_ESS_fraction_proxy": math.exp(-2.0 * args.total_K_budget),
            },
            "K_definition": "K_map=0.5*sum_channels(u^2)",
            "R_definition": "R_map=sum_channels(u*actual_P_innovation)",
            "L_definition": "L_map=R_map-K_map",
            "increment": "sum(L_map)=log Q_k/P_k for the implemented same-covariance transitions",
            "exactness_scope": (
                "operational learned/discretized Markov chains only; not asserted equal to the "
                "ideal heat-flow marginal ratio and does not automatically inherit SG semantics"
            ),
            "quality_claim": "none; all crossings or correlations are diagnostics only",
        },
        "rng": {
            "owner": "one torch.Generator per path initialized solely from public seed",
            "seed_function": "sample_stream_seed imported unchanged from frozen baseline runner",
            "model_and_local-Q_calls_consume_no_random_numbers": True,
            "draw_order": "tile/tilt fixed first, then exact baseline P innovation drawn once",
            "expected_draws_per_path_including_initial": NUM_SPACED_STEPS,
        },
        "trace": {
            "format": "compressed NPZ without pickle",
            "stochastic_step_count": len(STOCHASTIC_INTERNAL_TIMESTEPS),
            "keys": list(TRACE_KEYS),
            "exact_dtypes": {
                key: str(dtype) for key, dtype in sorted(TRACE_DTYPES.items())
            },
            "tensor_shapes": {
                "x_t,pred_xstart,epsilon_current,epsilon_shifted,"
                "innovation,p_standard_deviation": [
                    len(STOCHASTIC_INTERNAL_TIMESTEPS), 3, 64, 64
                ],
                "theta": [len(STOCHASTIC_INTERNAL_TIMESTEPS), 3, 64, 64],
                "K_map,R_map,L_map": [len(STOCHASTIC_INTERNAL_TIMESTEPS), 64, 64],
            },
            "scalar_backlabel": (
                "persisted float64 maps are row-major summed on save and resume; sums must "
                "exactly equal persisted K_scalar/R_scalar/L_scalar arrays"
            ),
        },
        "outputs": {
            "images": "images/class_{class_id:04d}/{seed:019d}.png",
            "signals": "signals/class_{class_id:04d}/{seed:019d}.json",
            "traces": "traces/class_{class_id:04d}/{seed:019d}.npz",
            "strict_resume": (
                "manifest and signal self-hashes, source/checkpoint hashes, NPZ file and raw-array "
                "hashes, map/scalar identities, exact expected file set, and baseline pixel identity"
            ),
        },
        "sources": {
            "guided_diffusion_root": str(source_root),
            "guided_diffusion_revision": revision,
            "guided_diffusion_tracked_dirty": dirty,
            "guided_diffusion_python_tree_sha256": sha256_python_tree(
                source_root / "guided_diffusion"
            ),
            "baseline_runner": {"path": str(baseline_runner), "sha256": sha256_file(baseline_runner)},
            "global_observe_helpers": {"path": str(observe_runner), "sha256": sha256_file(observe_runner)},
            "evidence_primitives": {
                "path": str(evidence_primitives),
                "sha256": sha256_file(evidence_primitives),
            },
            "local_Q_primitives": {
                "path": str(local_primitives),
                "sha256": sha256_file(local_primitives),
            },
        },
        "determinism": {
            "torch_deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cudnn_allow_tf32": False,
            "cuda_matmul_allow_tf32": False,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pillow": getattr(Image, "__version__", None),
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device_type": device.type,
            "device_name": torch.cuda.get_device_name(device),
            "device_capability": list(torch.cuda.get_device_capability(device)),
        },
        "runner": {"path": str(runner), "sha256": sha256_file(runner)},
    }
    manifest["identity_sha256"] = sha256_json(manifest)
    return manifest


def create_or_validate_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    path = output_dir / "manifest.json"
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"cannot read existing manifest: {path}") from exc
        if existing.get("identity_sha256") != _canonical_payload_sha(
            existing, "identity_sha256"
        ):
            raise RuntimeError("existing local-Q manifest self-hash is invalid")
        if existing != manifest:
            differing = sorted(
                key for key in set(existing) | set(manifest) if existing.get(key) != manifest.get(key)
            )
            raise RuntimeError(
                f"output directory has incompatible manifest keys: {', '.join(differing)}"
            )
        return
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory without manifest: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(manifest, path)


def _validate_trace_arrays(
    arrays: dict[str, np.ndarray],
    *,
    spec: LocalEvidenceSpec,
    total_K_budget: float,
    grid_size: int,
) -> dict[str, Any]:
    if sorted(arrays) != sorted(TRACE_KEYS):
        raise RuntimeError("local-Q trace key set changed")
    n = len(STOCHASTIC_INTERNAL_TIMESTEPS)
    expected_shapes = {
        "x_t": (n, 3, IMAGE_SIZE, IMAGE_SIZE),
        "pred_xstart": (n, 3, IMAGE_SIZE, IMAGE_SIZE),
        "theta": (n, 3, IMAGE_SIZE, IMAGE_SIZE),
        "epsilon_current": (n, 3, IMAGE_SIZE, IMAGE_SIZE),
        "epsilon_shifted": (n, 3, IMAGE_SIZE, IMAGE_SIZE),
        "innovation": (n, 3, IMAGE_SIZE, IMAGE_SIZE),
        "p_standard_deviation": (n, 3, IMAGE_SIZE, IMAGE_SIZE),
        "K_map": (n, IMAGE_SIZE, IMAGE_SIZE),
        "R_map": (n, IMAGE_SIZE, IMAGE_SIZE),
        "L_map": (n, IMAGE_SIZE, IMAGE_SIZE),
        "tile_bounds_yxyx": (n, 4),
    }
    scalar_keys = set(TRACE_KEYS) - set(expected_shapes)
    expected_shapes.update({key: (n,) for key in scalar_keys})
    wrong_shapes = {
        key: (arrays[key].shape, shape)
        for key, shape in expected_shapes.items()
        if arrays[key].shape != shape
    }
    if wrong_shapes:
        raise RuntimeError(f"local-Q trace shape mismatch: {wrong_shapes}")
    wrong_dtypes = {
        key: (arrays[key].dtype, TRACE_DTYPES[key])
        for key in TRACE_KEYS
        if arrays[key].dtype != TRACE_DTYPES[key]
    }
    if wrong_dtypes:
        raise RuntimeError(f"local-Q trace dtype mismatch: {wrong_dtypes}")
    if any(
        not np.isfinite(value).all()
        for value in arrays.values()
        if np.issubdtype(value.dtype, np.floating)
    ):
        raise RuntimeError("local-Q trace contains non-finite values")

    if not np.array_equal(
        arrays["internal_timestep"], np.asarray(STOCHASTIC_INTERNAL_TIMESTEPS, dtype=np.int16)
    ):
        raise RuntimeError("stochastic internal-timestep trace changed")
    mapping = spec.component.mapping
    if not np.array_equal(
        arrays["current_original_timestep"],
        mapping.current_timestep.astype(np.int16),
    ):
        raise RuntimeError("current original-timestep trace changed")
    if not np.array_equal(
        arrays["shifted_original_timestep"],
        mapping.shifted_timestep.astype(np.int16),
    ):
        raise RuntimeError("shifted original-timestep trace changed")
    expected_current_alpha = 1.0 / (1.0 + mapping.current_heat_variance)
    expected_shifted_alpha = 1.0 / (1.0 + mapping.shifted_heat_variance)
    if not np.allclose(
        arrays["current_alpha_bar"],
        expected_current_alpha,
        rtol=2e-13,
        atol=0.0,
    ):
        raise RuntimeError("current alpha_bar does not reconstruct from frozen heat mapping")
    if not np.allclose(
        arrays["shifted_alpha_bar"],
        expected_shifted_alpha,
        rtol=2e-13,
        atol=0.0,
    ):
        raise RuntimeError("shifted alpha_bar does not reconstruct from frozen heat mapping")
    if np.any(arrays["current_alpha_bar"] <= 0) or np.any(
        arrays["current_alpha_bar"] >= 1
    ):
        raise RuntimeError("persisted current alpha_bar is outside (0,1)")
    if np.any(arrays["shifted_alpha_bar"] <= 0) or np.any(
        arrays["shifted_alpha_bar"] > arrays["current_alpha_bar"]
    ):
        raise RuntimeError("persisted shifted alpha_bar is invalid")
    expected_rho = np.sqrt(expected_shifted_alpha / expected_current_alpha)
    if not np.allclose(arrays["rho"], expected_rho, rtol=2e-13, atol=0.0):
        raise RuntimeError("rho does not reconstruct from frozen current/shifted scales")
    if np.any(arrays["p_standard_deviation"] <= 0):
        raise RuntimeError("persisted P standard deviation must be strictly positive")

    expanded_current_alpha = arrays["current_alpha_bar"][:, None, None, None]
    expanded_shifted_alpha = arrays["shifted_alpha_bar"][:, None, None, None]
    expanded_rho = arrays["rho"][:, None, None, None]
    reconstructed_theta = (
        -expanded_rho
        * arrays["epsilon_shifted"].astype(np.float64)
        / np.sqrt(1.0 - expanded_shifted_alpha)
        + arrays["epsilon_current"].astype(np.float64)
        / np.sqrt(1.0 - expanded_current_alpha)
    )
    if not np.allclose(
        arrays["theta"], reconstructed_theta, rtol=2e-11, atol=2e-12
    ):
        raise RuntimeError("stored epsilon predictions/scales do not reconstruct theta")
    effective = (mapping.shifted_timestep != mapping.current_timestep).astype(np.uint8)
    if not np.array_equal(arrays["effective_nonidentity"], effective):
        raise RuntimeError("effective non-identity mask changed")
    if not np.array_equal(arrays["shifted_model_evaluated"], effective):
        raise RuntimeError("shifted model-evaluation mask changed")
    identity = effective == 0
    if not np.array_equal(
        arrays["epsilon_shifted"][identity], arrays["epsilon_current"][identity]
    ):
        raise RuntimeError("identity heat mapping must reuse the current epsilon prediction")
    expected_allowance = effective.astype(np.float64) * spec.fixed_K_allowance
    if not np.array_equal(arrays["active_K_allowance"], expected_allowance):
        raise RuntimeError("fixed per-step allowance trace changed")

    if not np.array_equal(arrays["L_map"], arrays["R_map"] - arrays["K_map"]):
        raise RuntimeError("persisted L_map is not exactly persisted R_map-K_map")
    for map_key, scalar_key in (
        ("K_map", "K_scalar"),
        ("R_map", "R_scalar"),
        ("L_map", "L_scalar"),
    ):
        reconstructed = _row_sum_float64(arrays[map_key])
        if not np.array_equal(reconstructed, arrays[scalar_key]):
            raise RuntimeError(f"{map_key} does not strictly back-label {scalar_key}")
    cumulative = np.cumsum(arrays["L_scalar"], dtype=np.float64)
    if not np.array_equal(cumulative, arrays["cumulative_log_e"]):
        raise RuntimeError("cumulative log-E trace does not reconstruct from L_scalar")
    if np.any(arrays["K_map"] < 0):
        raise RuntimeError("K_map contains a negative entry")
    if np.any(arrays["K_scalar"] > arrays["active_K_allowance"]):
        raise RuntimeError("a persisted step exceeds its fixed K allowance")
    total_K = float(arrays["K_scalar"].sum(dtype=np.float64))
    if total_K > total_K_budget:
        raise RuntimeError("persisted path exceeds its total K budget")
    scalar_error = np.abs(
        arrays["L_scalar"] - (arrays["R_scalar"] - arrays["K_scalar"])
    )
    scalar_tolerance = 5e-12 * (
        1.0
        + np.abs(arrays["L_scalar"])
        + np.abs(arrays["R_scalar"])
        + arrays["K_scalar"]
    )
    if np.any(scalar_error > scalar_tolerance):
        raise RuntimeError("persisted scalar likelihood decomposition is inconsistent")

    tile_size = IMAGE_SIZE // grid_size
    for index, bounds in enumerate(arrays["tile_bounds_yxyx"].tolist()):
        y0, x0, y1, x1 = (int(value) for value in bounds)
        valid = (
            0 <= y0 < y1 <= IMAGE_SIZE
            and 0 <= x0 < x1 <= IMAGE_SIZE
            and y1 - y0 == tile_size
            and x1 - x0 == tile_size
            and y0 % tile_size == 0
            and x0 % tile_size == 0
        )
        if not valid:
            raise RuntimeError(f"invalid contiguous tile bounds at step {index}: {bounds}")
        outside = arrays["K_map"][index].copy()
        outside[y0:y1, x0:x1] = 0.0
        if np.count_nonzero(outside):
            raise RuntimeError(f"K_map is nonzero outside selected tile at step {index}")
        if int(arrays["tile_index"][index]) != (
            (y0 // tile_size) * grid_size + x0 // tile_size
        ):
            raise RuntimeError("tile index and bounds disagree")
        raw_u = (
            arrays["p_standard_deviation"][index].astype(np.float64)
            * arrays["theta"][index]
        )
        tiled_energy = np.square(raw_u).reshape(
            raw_u.shape[0],
            grid_size,
            tile_size,
            grid_size,
            tile_size,
        ).sum(axis=(0, 2, 4), dtype=np.float64)
        expected_tile_index = int(tiled_energy.reshape(-1).argmax())
        if int(arrays["tile_index"][index]) != expected_tile_index:
            raise RuntimeError("persisted tile is not the predictable row-major energy argmax")
        localized_raw_u = np.zeros_like(raw_u)
        localized_raw_u[:, y0:y1, x0:x1] = raw_u[:, y0:y1, x0:x1]
        raw_K = 0.5 * float(
            localized_raw_u.reshape(-1).dot(localized_raw_u.reshape(-1))
        )
        if not _close(raw_K, float(arrays["raw_local_K"][index]), 2e-11):
            raise RuntimeError("stored sigma/theta/tile do not reconstruct raw local K")
        scale = float(arrays["local_scale"][index])
        if not 0 < scale <= 1.0:
            raise RuntimeError("invalid persisted local tempering scale")
        expected_scale = (
            min(1.0, math.sqrt(spec.guarded_step_cap / raw_K))
            if raw_K > 0
            else 1.0
        )
        if not _close(scale, expected_scale, 2e-11):
            raise RuntimeError("persisted local scale is not the fixed-cap tempering scale")
        reconstructed_u = localized_raw_u * scale
        reconstructed_K_map = 0.5 * np.square(reconstructed_u).sum(
            axis=0, dtype=np.float64
        )
        reconstructed_R_map = (
            reconstructed_u
            * arrays["innovation"][index].astype(np.float64)
        ).sum(axis=0, dtype=np.float64)
        if not np.allclose(
            reconstructed_K_map, arrays["K_map"][index], rtol=5e-13, atol=5e-15
        ):
            raise RuntimeError("stored sigma/theta/tile/scale do not reconstruct K_map")
        if not np.allclose(
            reconstructed_R_map, arrays["R_map"][index], rtol=5e-13, atol=5e-15
        ):
            raise RuntimeError("stored sigma/theta/tile/scale/innovation do not reconstruct R_map")
        total_energy = float(np.square(raw_u).sum(dtype=np.float64))
        selected_energy = 2.0 * raw_K
        expected_fraction = selected_energy / total_energy if total_energy > 0 else 0.0
        if not _close(
            expected_fraction,
            float(arrays["selected_energy_fraction"][index]),
            2e-11,
        ):
            raise RuntimeError("stored tile does not reconstruct selected energy fraction")
    if np.count_nonzero(arrays["theta"][identity]) or np.count_nonzero(arrays["K_map"][identity]):
        raise RuntimeError("identity heat mapping must have theta=0 and Q=P")

    return {
        "stochastic_step_count": n,
        "effective_nonidentity_step_count": int(effective.sum(dtype=np.int64)),
        "shifted_model_evaluation_count": int(
            arrays["shifted_model_evaluated"].sum(dtype=np.int64)
        ),
        "total_K_budget": float(total_K_budget),
        "fixed_K_allowance_per_effective_step": spec.fixed_K_allowance,
        "guarded_numerical_cap_per_effective_step": spec.guarded_step_cap,
        "total_applied_K": total_K,
        "unused_K_budget": float(total_K_budget - total_K),
        "final_cumulative_log_e": float(arrays["cumulative_log_e"][-1]),
        "running_max_log_e": float(max(0.0, float(arrays["cumulative_log_e"].max()))),
        "intervention_count": 0,
    }


def _validate_trace_file(
    path: Path,
    record: dict[str, Any],
    output_dir: Path,
    *,
    spec: LocalEvidenceSpec,
    total_K_budget: float,
    grid_size: int,
) -> dict[str, Any]:
    if record.get("relative_path") != path.relative_to(output_dir).as_posix():
        raise RuntimeError("trace relative path identity changed")
    if not path.is_file() or path.stat().st_size != record.get("bytes"):
        raise RuntimeError("trace file is missing or has wrong byte count")
    if sha256_file(path) != record.get("sha256"):
        raise RuntimeError("trace file SHA-256 is invalid")
    try:
        with np.load(path, allow_pickle=False) as archive:
            if sorted(archive.files) != record.get("keys"):
                raise RuntimeError("trace archive key set changed")
            arrays = {key: np.ascontiguousarray(archive[key]) for key in archive.files}
    except Exception as exc:
        raise RuntimeError(f"cannot validate local-Q trace {path}") from exc
    for key, array_record in record.get("arrays", {}).items():
        value = arrays.get(key)
        expected = None if value is None else {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "raw_bytes_sha256": _array_raw_sha256(value),
        }
        if expected != array_record:
            raise RuntimeError(f"trace array identity failed for {key}")
    if set(record.get("arrays", {})) != set(arrays):
        raise RuntimeError("trace record does not cover every persisted array")
    return _validate_trace_arrays(
        arrays,
        spec=spec,
        total_K_budget=total_K_budget,
        grid_size=grid_size,
    )


def save_pair(
    pixels: np.ndarray,
    trace: dict[str, np.ndarray],
    summary: dict[str, Any],
    pair: Pair,
    output_dir: Path,
    baseline: BaselineReference,
    baseline_pixels: np.ndarray,
    manifest_identity: str,
    runner_sha: str,
) -> None:
    paths = (
        image_path(output_dir, pair),
        signal_path(output_dir, pair),
        trace_path(output_dir, pair),
    )
    if any(path.exists() for path in paths):
        raise RuntimeError(f"refusing to overwrite an existing/partial local-Q pair: {pair}")
    if not np.array_equal(pixels, baseline_pixels):
        raise RuntimeError(f"observe-only endpoint changed frozen P pixels: {pair}")
    _atomic_npz_dump(trace, paths[2])
    trace_record = _trace_file_record(paths[2], trace, output_dir)
    signal: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "class_id": pair[0],
        "seed": pair[1],
        "sample_stream_seed": sample_stream_seed(pair[1]),
        "manifest_identity_sha256": manifest_identity,
        "runner_sha256": runner_sha,
        "baseline_manifest_identity_sha256": baseline.manifest_identity_sha256,
        "pixel_sha256": pixel_sha256(pixels),
        "baseline_pixel_sha256": pixel_sha256(baseline_pixels),
        "observer_changed_P": False,
        "operational_LR_only": True,
        "ideal_heat_marginal_ratio_claimed": False,
        "image_quality_claimed": False,
        "summary": summary,
        "trace": trace_record,
    }
    signal["payload_sha256"] = _canonical_payload_sha(signal, "payload_sha256")
    atomic_json_dump(signal, paths[1])

    metadata = PngInfo()
    for key, value in {
        "experiment": EXPERIMENT,
        "class_id": str(pair[0]),
        "seed": str(pair[1]),
        "sample_stream_seed": str(sample_stream_seed(pair[1])),
        "pixel_sha256": pixel_sha256(pixels),
        "baseline_pixel_sha256": pixel_sha256(baseline_pixels),
        "signal_payload_sha256": signal["payload_sha256"],
        "trace_sha256": trace_record["sha256"],
        "manifest_identity_sha256": manifest_identity,
        "runner_sha256": runner_sha,
    }.items():
        metadata.add_text(key, value)
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    temporary = paths[0].with_name(paths[0].name + ".tmp")
    Image.fromarray(pixels, mode="RGB").save(temporary, format="PNG", pnginfo=metadata)
    os.replace(temporary, paths[0])


def validate_output_set(
    output_dir: Path,
    baseline: BaselineReference,
    pairs: Sequence[Pair],
    manifest_identity: str,
    runner_sha: str,
    *,
    spec: LocalEvidenceSpec,
    total_K_budget: float,
    grid_size: int,
    require_all: bool,
) -> set[Pair]:
    allowed_root_entries = {"manifest.json", "completion.json", "images", "signals", "traces"}
    unexpected_root = sorted(
        path for path in output_dir.iterdir() if path.name not in allowed_root_entries
    )
    if unexpected_root:
        raise RuntimeError(f"output contains unexpected root artifact: {unexpected_root[0]}")
    for name in ("images", "signals", "traces"):
        path = output_dir / name
        if path.exists() and not path.is_dir():
            raise RuntimeError(f"expected output subtree is not a directory: {path}")
    for name in ("manifest.json", "completion.json"):
        path = output_dir / name
        if path.exists() and not path.is_file():
            raise RuntimeError(f"expected output metadata is not a regular file: {path}")
    expected_images = {image_path(output_dir, pair).resolve() for pair in pairs}
    expected_signals = {signal_path(output_dir, pair).resolve() for pair in pairs}
    expected_traces = {trace_path(output_dir, pair).resolve() for pair in pairs}
    actual_images = {
        path.resolve()
        for path in (output_dir / "images").rglob("*")
        if path.is_file()
    } if (output_dir / "images").exists() else set()
    actual_signals = {
        path.resolve()
        for path in (output_dir / "signals").rglob("*")
        if path.is_file()
    } if (output_dir / "signals").exists() else set()
    actual_traces = {
        path.resolve()
        for path in (output_dir / "traces").rglob("*")
        if path.is_file()
    } if (output_dir / "traces").exists() else set()
    unexpected = sorted(
        (actual_images - expected_images)
        | (actual_signals - expected_signals)
        | (actual_traces - expected_traces)
    )
    if unexpected:
        raise RuntimeError(f"output contains unexpected local-Q artifact: {unexpected[0]}")

    complete: set[Pair] = set()
    for pair in pairs:
        png = image_path(output_dir, pair)
        json_file = signal_path(output_dir, pair)
        npz = trace_path(output_dir, pair)
        present = (png.is_file(), json_file.is_file(), npz.is_file())
        if any(present) and not all(present):
            raise RuntimeError(f"strict resume found a partial local-Q output: {pair}/{present}")
        if not any(present):
            if png.exists() or json_file.exists() or npz.exists():
                raise RuntimeError(f"expected local-Q output path is not a regular file: {pair}")
            continue
        try:
            signal = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"cannot read local-Q signal {json_file}") from exc
        if signal.get("payload_sha256") != _canonical_payload_sha(signal, "payload_sha256"):
            raise RuntimeError(f"local-Q signal self-hash is invalid: {pair}")
        expected_signal = {
            "schema_version": SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "class_id": pair[0],
            "seed": pair[1],
            "sample_stream_seed": sample_stream_seed(pair[1]),
            "manifest_identity_sha256": manifest_identity,
            "runner_sha256": runner_sha,
            "baseline_manifest_identity_sha256": baseline.manifest_identity_sha256,
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
            raise RuntimeError(f"local-Q signal identity mismatch for {pair}: {mismatches}")
        reconstructed_summary = _validate_trace_file(
            npz,
            signal.get("trace", {}),
            output_dir,
            spec=spec,
            total_K_budget=total_K_budget,
            grid_size=grid_size,
        )
        if signal.get("summary") != reconstructed_summary:
            raise RuntimeError(f"local-Q summary does not reconstruct from trace: {pair}")

        with Image.open(png) as image:
            metadata = dict(image.info)
            if image.mode != "RGB" or image.size != (IMAGE_SIZE, IMAGE_SIZE):
                raise RuntimeError(f"wrong local-Q PNG mode/size: {png}")
            image.verify()
        pixels = decoded_pixels(png)
        baseline_pixels = decoded_pixels(baseline_pair_path(baseline.root, pair))
        if not np.array_equal(pixels, baseline_pixels):
            raise RuntimeError(f"local-Q observer endpoint differs from frozen P baseline: {pair}")
        decoded_sha = pixel_sha256(pixels)
        baseline_sha = pixel_sha256(baseline_pixels)
        expected_metadata = {
            "experiment": EXPERIMENT,
            "class_id": str(pair[0]),
            "seed": str(pair[1]),
            "sample_stream_seed": str(sample_stream_seed(pair[1])),
            "pixel_sha256": decoded_sha,
            "baseline_pixel_sha256": baseline_sha,
            "signal_payload_sha256": signal["payload_sha256"],
            "trace_sha256": signal["trace"]["sha256"],
            "manifest_identity_sha256": manifest_identity,
            "runner_sha256": runner_sha,
        }
        metadata_mismatches = {
            key: (metadata.get(key), value)
            for key, value in expected_metadata.items()
            if metadata.get(key) != value
        }
        if metadata_mismatches:
            raise RuntimeError(f"local-Q PNG metadata mismatch for {pair}: {metadata_mismatches}")
        if (
            signal.get("pixel_sha256") != decoded_sha
            or signal.get("baseline_pixel_sha256") != baseline_sha
        ):
            raise RuntimeError(f"local-Q signal pixel identity mismatch for {pair}")
        complete.add(pair)
    if require_all and len(complete) != len(pairs):
        raise RuntimeError(f"only {len(complete)}/{len(pairs)} local-Q pairs are complete")
    return complete


def run_observation(args: argparse.Namespace, protocol: Protocol) -> None:
    if args.batch < 1:
        raise ValueError("--batch must be positive")
    if IMAGE_SIZE % args.grid_size:
        raise ValueError("--grid-size must divide the ADM image size 64")
    _assert_output_isolated(args)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the official FP16 ADM64 sampler requires CUDA")

    configure_determinism()
    torch.cuda.set_device(device)
    diffusion_checkpoint = validate_checkpoint(args.model_path, DIFFUSION_CHECKPOINT)
    classifier_checkpoint = validate_checkpoint(args.classifier_path, CLASSIFIER_CHECKPOINT)
    original_alpha_bar, timestep_map = original_schedule_and_timestep_map(
        args.guided_diffusion_root
    )
    spec = build_local_spec(original_alpha_bar, timestep_map, args.total_K_budget)
    baseline = load_baseline_reference(
        args.baseline_dir,
        protocol,
        expected_model_sha256=diffusion_checkpoint["sha256"],
        expected_classifier_sha256=classifier_checkpoint["sha256"],
    )
    manifest = build_manifest(
        args,
        protocol,
        device,
        diffusion_checkpoint,
        classifier_checkpoint,
        baseline,
        original_alpha_bar,
        timestep_map,
        spec,
    )
    create_or_validate_manifest(args.output_dir, manifest)
    identity = manifest["identity_sha256"]
    runner_sha = manifest["runner"]["sha256"]
    complete = validate_output_set(
        args.output_dir,
        baseline,
        protocol.pairs,
        identity,
        runner_sha,
        spec=spec,
        total_K_budget=args.total_K_budget,
        grid_size=args.grid_size,
        require_all=False,
    )
    completion_path = args.output_dir / "completion.json"
    old_completion = validate_existing_completion(
        completion_path,
        manifest_identity_sha256=identity,
        pair_set_sha256=manifest["pair_set_sha256"],
        total_expected=len(protocol.pairs),
    )
    if old_completion is not None:
        if len(complete) != len(protocol.pairs):
            raise RuntimeError("completion exists but strict local-Q artifacts are incomplete")
        print(json.dumps(old_completion, ensure_ascii=False, indent=2))
        return

    pending = [pair for pair in protocol.pairs if pair not in complete]
    start = time.monotonic()
    generated = 0
    accounting: dict[str, int] | None = None
    if pending:
        model, diffusion, classifier = load_official_models(
            args.guided_diffusion_root, args.model_path, args.classifier_path, device
        )
        if list(diffusion.timestep_map) != timestep_map.tolist():
            raise RuntimeError("loaded SpacedDiffusion timestep map differs from manifest map")
        _, cond_fn = make_guided_functions(model, classifier)
        for logical_batch in chunks(pending, args.batch):
            observed = sample_local_observe_batch(
                diffusion,
                model,
                cond_fn,
                logical_batch,
                device=device,
                original_alpha_bar=original_alpha_bar,
                timestep_map=timestep_map,
                spec=spec,
                total_K_budget=args.total_K_budget,
                grid_size=args.grid_size,
            )
            accounting = observed.accounting
            if accounting["stochastic_reverse_steps"] != len(STOCHASTIC_INTERNAL_TIMESTEPS):
                raise AssertionError("local-Q stochastic-step accounting changed")
            if accounting["shifted_unet_evaluations_per_path"] != spec.effective_nonidentity_steps:
                raise AssertionError("local-Q shifted-evaluation accounting changed")
            for index, pair in enumerate(logical_batch):
                pixels = pixels_from_sample(observed.final_states[index])
                baseline_pixels = decoded_pixels(baseline_pair_path(baseline.root, pair))
                save_pair(
                    pixels,
                    observed.traces[index],
                    observed.summaries[index],
                    pair,
                    args.output_dir,
                    baseline,
                    baseline_pixels,
                    identity,
                    runner_sha,
                )
                generated += 1
            print(
                f"observed {generated}/{len(pending)} new all-step local-Q paths "
                f"({len(complete)} already complete, {time.monotonic() - start:.1f}s)",
                flush=True,
            )

    final = validate_output_set(
        args.output_dir,
        baseline,
        protocol.pairs,
        identity,
        runner_sha,
        spec=spec,
        total_K_budget=args.total_K_budget,
        grid_size=args.grid_size,
        require_all=True,
    )
    completion = {
        "complete": True,
        "manifest_identity_sha256": identity,
        "pair_set_sha256": manifest["pair_set_sha256"],
        "generated_this_run": generated,
        "already_complete": len(complete),
        "total_expected": len(protocol.pairs),
        "total_complete": len(final),
        "logical_batch_requested": args.batch,
        "neural_eval_batch_size": 1,
        "stochastic_steps_observed_per_path": len(STOCHASTIC_INTERNAL_TIMESTEPS),
        "effective_nonidentity_steps_per_path": spec.effective_nonidentity_steps,
        "interventions": 0,
        "sampling_record": accounting,
        "wall_seconds": time.monotonic() - start,
        "finished_at_unix": time.time(),
    }
    atomic_json_dump(completion, completion_path)
    print(json.dumps(completion, ensure_ascii=False, indent=2))


class _ToyDiffusion:
    num_timesteps = 5
    timestep_map = [0, 2, 4, 6, 8]

    def p_mean_variance(
        self,
        model_fn: Callable[..., torch.Tensor],
        x: torch.Tensor,
        t: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        original = torch.tensor(self.timestep_map, dtype=t.dtype, device=t.device)[t]
        raw = model_fn(x, original, **kwargs["model_kwargs"])
        channels = x.shape[1]
        epsilon, variance_logits = raw[:, :channels], raw[:, channels:]
        variance = torch.exp(-3.0 + 0.02 * variance_logits)
        return {
            "mean": 0.91 * x + 0.015 * epsilon,
            "variance": variance,
            "log_variance": variance.log(),
            "pred_xstart": epsilon,
        }

    def condition_mean(
        self,
        cond_fn: Callable[..., torch.Tensor],
        out: dict[str, torch.Tensor],
        x: torch.Tensor,
        t: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        original = torch.tensor(self.timestep_map, dtype=t.dtype, device=t.device)[t]
        return out["mean"] + out["variance"] * cond_fn(
            x, original, **kwargs["model_kwargs"]
        )


class _ToyModel(torch.nn.Module):
    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor:
        if y is None or x.shape[0] != 1:
            raise AssertionError("toy model requires singleton class conditioning")
        epsilon = 0.07 * x + t.view(-1, 1, 1, 1).to(x.dtype) * 0.002
        epsilon = epsilon + y.view(-1, 1, 1, 1).to(x.dtype) * 0.0001
        variance_logits = torch.tanh(0.03 * x)
        return torch.cat([epsilon, variance_logits], dim=1)


def run_self_test() -> None:
    # Temporarily substitute the five-step toy stochastic schedule; production
    # uses the frozen 249-entry module constant.
    global STOCHASTIC_INTERNAL_TIMESTEPS
    production_steps = STOCHASTIC_INTERNAL_TIMESTEPS
    STOCHASTIC_INTERNAL_TIMESTEPS = (4, 3, 2, 1)
    try:
        device = torch.device("cpu")
        diffusion = _ToyDiffusion()
        alpha_bar = np.asarray(
            [0.99, 0.96, 0.90, 0.80, 0.68, 0.54, 0.39, 0.24, 0.10],
            dtype=np.float64,
        )
        timestep_map = np.asarray(diffusion.timestep_map, dtype=np.int64)
        spec = build_local_spec(alpha_bar, timestep_map, 0.5)
        pairs: tuple[Pair, ...] = ((3, 7), (9, 7), (3, 8))

        def toy_cond(
            x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None
        ) -> torch.Tensor:
            if y is None:
                raise AssertionError("toy classifier requires y")
            return torch.full_like(x, 0.003) + y.view(-1, 1, 1, 1) * 0.00001

        def grouped(size: int) -> tuple[dict[Pair, torch.Tensor], dict[Pair, dict[str, np.ndarray]]]:
            samples: dict[Pair, torch.Tensor] = {}
            traces: dict[Pair, dict[str, np.ndarray]] = {}
            for logical_batch in chunks(pairs, size):
                observed = sample_local_observe_batch(
                    diffusion,
                    _ToyModel(),
                    toy_cond,
                    logical_batch,
                    device=device,
                    original_alpha_bar=alpha_bar,
                    timestep_map=timestep_map,
                    spec=spec,
                    total_K_budget=0.5,
                    grid_size=2,
                    channels=1,
                    image_size=4,
                )
                if observed.accounting["gaussian_draws_per_path_including_initial"] != 5:
                    raise AssertionError("toy local-Q observer changed RNG draw count")
                for index, pair in enumerate(logical_batch):
                    samples[pair] = observed.final_states[index].clone()
                    traces[pair] = observed.traces[index]
            return samples, traces

        singleton_samples, singleton_traces = grouped(1)
        for size in (2, 3):
            samples, traces = grouped(size)
            for pair in pairs:
                if not torch.equal(samples[pair], singleton_samples[pair]):
                    raise AssertionError("toy P endpoint changed with logical grouping")
                for key in TRACE_KEYS:
                    if not np.array_equal(traces[pair][key], singleton_traces[pair][key]):
                        raise AssertionError(f"toy local-Q trace changed with grouping: {key}")

        baseline, _ = sample_batch_invariant(
            diffusion,
            _ToyModel(),
            toy_cond,
            pairs,
            device=device,
            channels=1,
            image_size=4,
        )
        for index, pair in enumerate(pairs):
            if not torch.equal(baseline[index], singleton_samples[pair]):
                raise AssertionError("instrumented toy P path differs from uninstrumented P")

        # Adapt shape-only production validator checks to this tiny trace here;
        # exact map/scalar, budget, locality, and ordering invariants are tested
        # directly without pretending the toy tensors are 3x64x64.
        for trace in singleton_traces.values():
            for key, expected_dtype in TRACE_DTYPES.items():
                if trace[key].dtype != expected_dtype:
                    raise AssertionError(f"toy exact dtype schema failed: {key}")
            for map_key, scalar_key in (("K_map", "K_scalar"), ("R_map", "R_scalar"), ("L_map", "L_scalar")):
                if not np.array_equal(_row_sum_float64(trace[map_key]), trace[scalar_key]):
                    raise AssertionError("toy map/scalar back-label failed")
            if not np.array_equal(trace["L_map"], trace["R_map"] - trace["K_map"]):
                raise AssertionError("toy L-map decomposition failed")
            if float(trace["K_scalar"].sum(dtype=np.float64)) > 0.5:
                raise AssertionError("toy total K budget failed")
            if np.any(trace["K_scalar"] > trace["active_K_allowance"]):
                raise AssertionError("toy fixed step allowance failed")
            if not np.array_equal(
                np.cumsum(trace["L_scalar"], dtype=np.float64),
                trace["cumulative_log_e"],
            ):
                raise AssertionError("toy cumulative log E failed")
            current_alpha = trace["current_alpha_bar"][:, None, None, None]
            shifted_alpha = trace["shifted_alpha_bar"][:, None, None, None]
            rho = trace["rho"][:, None, None, None]
            reconstructed_theta = (
                -rho
                * trace["epsilon_shifted"].astype(np.float64)
                / np.sqrt(1.0 - shifted_alpha)
                + trace["epsilon_current"].astype(np.float64)
                / np.sqrt(1.0 - current_alpha)
            )
            if not np.array_equal(reconstructed_theta, trace["theta"]):
                raise AssertionError("toy epsilon/scale theta provenance failed")
            if np.any(trace["p_standard_deviation"] <= 0):
                raise AssertionError("toy positive P-sigma schema failed")

        import tempfile

        with tempfile.TemporaryDirectory(prefix="adm64-local-observe-self-test-") as temporary:
            root = Path(temporary)
            path = root / "trace.npz"
            trace = singleton_traces[pairs[0]]
            _atomic_npz_dump(trace, path)
            record = _trace_file_record(path, trace, root)
            if sha256_file(path) != record["sha256"] or record["keys"] != sorted(TRACE_KEYS):
                raise AssertionError("toy fail-closed NPZ record failed")
            payload = {"trace_sha256": record["sha256"], "finite": True}
            payload["payload_sha256"] = _canonical_payload_sha(payload, "payload_sha256")
            if payload["payload_sha256"] != _canonical_payload_sha(payload, "payload_sha256"):
                raise AssertionError("toy signal self-hash failed")

        print(
            "self-test passed: pre-innovation local tile, all stochastic steps, "
            "fixed non-carrying total-K budget, exact map/scalar LR back-labels, "
            "bitwise unchanged toy P/RNG, grouping invariance, and NPZ hashes"
        )
    finally:
        STOCHASTIC_INTERNAL_TIMESTEPS = production_steps


def build_parser() -> argparse.ArgumentParser:
    data_root = Path(os.environ.get("EQVAE_DATA_ROOT", "/home/zhoushunyu/data/eqvae"))
    guided_root = data_root / "baselines" / "guided-diffusion"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("smoke", "custom"), default="smoke")
    parser.add_argument("--classes", type=parse_int_spec, default=None)
    parser.add_argument("--seeds", type=parse_int_spec, default=None)
    parser.add_argument(
        "--total-K-budget",
        type=float,
        choices=TOTAL_K_BUDGET_CHOICES,
        default=DEFAULT_TOTAL_K_BUDGET,
        help="predeclared pathwise conditional-KL budget; only locked 0.5/1.0 screens",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        choices=(1, 2, 4, 8, 16),
        default=DEFAULT_GRID_SIZE,
        help="number of fixed contiguous tiles per spatial axis",
    )
    parser.add_argument("--guided-diffusion-root", type=Path, default=guided_root)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=guided_root / "checkpoints" / DIFFUSION_CHECKPOINT.filename,
    )
    parser.add_argument(
        "--classifier-path",
        type=Path,
        default=guided_root / "checkpoints" / CLASSIFIER_CHECKPOINT.filename,
    )
    parser.add_argument("--baseline-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="logical scheduling batch; neural calls stay singleton and traces are large",
    )
    parser.add_argument("--device", default="cuda:0")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.self_test:
        run_self_test()
        return
    protocol = protocol_from_args(args)
    data_root = Path(os.environ.get("EQVAE_DATA_ROOT", "/home/zhoushunyu/data/eqvae"))
    if args.baseline_dir is None:
        args.baseline_dir = data_root / "cross_scale_evidence" / "adm64_guided" / args.protocol
    if args.output_dir is None:
        suffix = str(args.total_K_budget).replace(".", "p")
        args.output_dir = (
            data_root
            / "cross_scale_evidence"
            / "adm64_local_path_evidence"
            / f"{args.protocol}_K{suffix}_grid{args.grid_size}"
        )
    args.guided_diffusion_root = args.guided_diffusion_root.resolve()
    args.model_path = args.model_path.resolve()
    args.classifier_path = args.classifier_path.resolve()
    args.baseline_dir = args.baseline_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.batch < 1:
        raise ValueError("--batch must be positive")
    _assert_output_isolated(args)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "protocol": args.protocol,
                    "class_ids": list(protocol.class_ids),
                    "seeds": list(protocol.seeds),
                    "sample_count": len(protocol.pairs),
                    "baseline_dir": str(args.baseline_dir),
                    "output_dir": str(args.output_dir),
                    "additive_normalized_heat_shift": ADDITIVE_HEAT_SHIFT,
                    "total_K_budget": args.total_K_budget,
                    "supported_total_K_budgets": list(TOTAL_K_BUDGET_CHOICES),
                    "grid_size_per_axis": args.grid_size,
                    "stochastic_steps_observed": len(STOCHASTIC_INTERNAL_TIMESTEPS),
                    "budget_policy": "fixed allowance over effective non-identity steps; no carry",
                    "observe_only": True,
                    "interventions": 0,
                    "operational_LR_only": True,
                    "ideal_heat_marginal_ratio_claimed": False,
                    "image_quality_claimed": False,
                    "note": (
                        "effective mapping count and exact allowance are fail-closed "
                        "in the real manifest"
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    run_observation(args, protocol)


if __name__ == "__main__":
    main()
