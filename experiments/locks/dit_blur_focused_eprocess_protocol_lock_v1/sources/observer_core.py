#!/usr/bin/env python3
"""Label-free core for a blur-gated cross-scale DiT e-process.

This module deliberately has a narrow input boundary.  It consumes only nine
preterminal DiT observations (sampling steps 69..149), frozen label-free blur
calibration thresholds, cross-scale score-gap directions constructed before
the corresponding transition innovation, the implemented P standard
deviation, and that innovation.  It never reads an endpoint image, a visual
label, a reviewer decision, an external representation, or a quality score.

There are three separate outputs:

* ``B_persistence`` is the already fixed mean decoded-pred-xstart blur statistic.
  It is predictable at step 149 but is not an e-process.
* ``E_blur_gated`` is an exact operational P/Q* path likelihood-ratio mixture.
  Before each innovation, the current B state may turn its mean shift on, and
  the two weakest-edge active tiles select a local subspace.  A pathwise total
  KL budget explicitly shrinks Q*.  This changes the semantic alternative; it
  is not the full high-noise marginal likelihood ratio.
* ``E_no_state_gate`` is a frozen exact ablation using the same local masks,
  directions, shifts, and KL budget while omitting only the B state gate.  It
  is not a replacement candidate.

The crucial implementation boundary is expressed in two calls:
``construct_predictable_shift`` has no innovation argument, while
``gaussian_log_lr`` observes the innovation only after the shift is complete.
The offline calculation is mathematically equivalent to this online ordering
because every input to the first call is measurable before the saved P draw.
Production lineage must nevertheless attest that ``theta`` was constructed
from the current state/model outputs alone.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from scipy import ndimage


EXPERIMENT = "dit_blur_focused_cross_scale_eprocess_label_free"
SCHEMA_VERSION = 1
CHECKPOINTS = tuple(range(69, 150, 10))
INTERNAL_TIMESTEPS = tuple(249 - value for value in CHECKPOINTS)
HEAT_SHIFTS = (1.0, 4.0)
SHIFTED_INTERNAL_TIMESTEPS = (
    (180, 170, 160, 150, 141, 131, 122, 114, 105),
    (180, 171, 161, 152, 143, 135, 128, 122, 116),
)
EFFECTIVE_NONIDENTITY = tuple(
    tuple(int(shifted != current) for shifted, current in zip(row, INTERNAL_TIMESTEPS))
    for row in SHIFTED_INTERNAL_TIMESTEPS
)
MIXTURE_WEIGHTS = (0.5, 0.5)
TOTAL_K_PER_SCALE = 2.0
ALPHA_E = 0.10
MATCHED_Q_POWER_DRAWS = 400_000
MATCHED_Q_POWER_SEED = 2026082808
MATCHED_Q_ANYTIME_POWER_MINIMUM = 0.30
REAL_LABEL_FREE_GATE_MINIMUM_SAMPLES = 60
REAL_LABEL_FREE_GATE_OPEN_FRACTION_MINIMUM = 0.50
REAL_LABEL_FREE_K_UTILIZATION_FRACTION_MINIMUM = 0.50
REAL_LABEL_FREE_K_UTILIZATION_THRESHOLD = 1.5
GRID_SIZE = 4
ACTIVE_TILE_COUNT = 8
FOCUS_TILE_COUNT = 2
EPS = 1e-12

INPUT_ARRAY_NAMES = (
    "decoded_pred_xstart_rgb",
    "theta",
    "p_standard_deviation",
    "transition_innovation",
    "sampling_step",
    "shifted_internal_timestep",
    "heat_shift",
    "effective_nonidentity",
    "blur_gate_threshold",
    "blur_score_threshold",
    "class_id",
)

FORBIDDEN_INPUT_KEY_TOKENS = (
    "label",
    "review",
    "endpoint",
    "human",
    "adjudicat",
    "inception",
    "dino",
    "clip_embedding",
    "fid",
    "auc",
)


@dataclass(frozen=True)
class BlurTiles:
    severity: np.ndarray
    q_value: np.ndarray
    variance: np.ndarray
    active_index: np.ndarray
    focus_index: np.ndarray
    latent_mask: np.ndarray


@dataclass(frozen=True)
class EProcessTracks:
    raw_K: np.ndarray
    applied_K: np.ndarray
    component_increment: np.ndarray
    component_log_e: np.ndarray
    mixture_log_e: np.ndarray
    running_max_log_e: np.ndarray
    alarm: np.ndarray


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_json_dump(value: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _array_record(value: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(value)
    return {
        "shape": list(contiguous.shape),
        "dtype": contiguous.dtype.str,
        "raw_sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
    }


def _tile_slices(height: int, width: int) -> list[tuple[slice, slice]]:
    if height < GRID_SIZE or width < GRID_SIZE or height % GRID_SIZE or width % GRID_SIZE:
        raise ValueError("image/latent dimensions must be positive multiples of the 4x4 grid")
    tile_h, tile_w = height // GRID_SIZE, width // GRID_SIZE
    return [
        (
            slice(row * tile_h, (row + 1) * tile_h),
            slice(col * tile_w, (col + 1) * tile_w),
        )
        for row in range(GRID_SIZE)
        for col in range(GRID_SIZE)
    ]


def _stable_lowest(values: np.ndarray, candidates: np.ndarray, count: int) -> np.ndarray:
    """Sort by value and then row-major tile index, independent of prior order."""

    if candidates.ndim != 1 or values.ndim != 1:
        raise ValueError("stable tile selection expects one-dimensional inputs")
    order = np.lexsort((candidates.astype(np.int64), values[candidates]))
    return candidates[order[:count]]


def compute_blur_tiles(images: np.ndarray, *, latent_size: int = 32) -> BlurTiles:
    """Compute the frozen B statistic and its predictable weakest-edge tiles.

    ``images`` is ``[N,T,3,H,W]`` clipped preterminal RGB.  The returned latent
    mask is ``[N,T,1,latent_size,latent_size]`` and selects exactly the two
    lowest-q tiles among the eight highest-variance image tiles.
    """

    value = np.asarray(images)
    if value.ndim != 5 or value.shape[1] != len(CHECKPOINTS) or value.shape[2] != 3:
        raise ValueError(
            f"decoded_pred_xstart_rgb must be [N,{len(CHECKPOINTS)},3,H,W], got {value.shape}"
        )
    if not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
        raise ValueError("decoded preterminal RGB must be finite floating point")
    if np.any(value < 0.0) or np.any(value > 1.0):
        raise ValueError("decoded preterminal RGB must already be clipped to [0,1]")

    batch, time_count, _, height, width = value.shape
    if batch < 1:
        raise ValueError("decoded preterminal RGB batch must be nonempty")
    image_tiles = _tile_slices(height, width)
    latent_tiles = _tile_slices(latent_size, latent_size)
    flat = value.astype(np.float64, copy=False).reshape(-1, 3, height, width)
    gray = 0.2989 * flat[:, 0] + 0.5870 * flat[:, 1] + 0.1140 * flat[:, 2]

    q_value = np.empty((len(flat), GRID_SIZE * GRID_SIZE), dtype=np.float64)
    variance = np.empty_like(q_value)
    active_index = np.empty((len(flat), ACTIVE_TILE_COUNT), dtype=np.int16)
    focus_index = np.empty((len(flat), FOCUS_TILE_COUNT), dtype=np.int16)
    severity = np.empty(len(flat), dtype=np.float64)

    for row, source in enumerate(gray):
        smooth = ndimage.gaussian_filter(source, sigma=0.7, mode="reflect")
        gx = ndimage.sobel(smooth, axis=1, mode="reflect") / 8.0
        gy = ndimage.sobel(smooth, axis=0, mode="reflect") / 8.0
        laplacian = ndimage.laplace(smooth, mode="reflect")
        gradient_energy = gx * gx + gy * gy
        variance[row] = np.asarray(
            [float(np.var(source[ys, xs], dtype=np.float64)) for ys, xs in image_tiles],
            dtype=np.float64,
        )
        q_value[row] = np.asarray(
            [
                float(np.mean(laplacian[ys, xs] ** 2, dtype=np.float64))
                / (float(np.mean(gradient_energy[ys, xs], dtype=np.float64)) + EPS)
                for ys, xs in image_tiles
            ],
            dtype=np.float64,
        )
        active = np.argsort(-variance[row], kind="stable")[:ACTIVE_TILE_COUNT]
        focus = _stable_lowest(q_value[row], active, FOCUS_TILE_COUNT)
        active_index[row] = active.astype(np.int16)
        focus_index[row] = focus.astype(np.int16)
        severity[row] = -math.log(
            float(np.percentile(q_value[row, active], 25, method="linear")) + EPS
        )

    mask = np.zeros((len(flat), 1, latent_size, latent_size), dtype=np.float64)
    for row, indices in enumerate(focus_index):
        for tile_index in indices.tolist():
            ys, xs = latent_tiles[int(tile_index)]
            mask[row, 0, ys, xs] = 1.0
    expected_pixels = FOCUS_TILE_COUNT * (latent_size // GRID_SIZE) ** 2
    if not np.all(np.sum(mask, axis=(1, 2, 3)) == expected_pixels):
        raise AssertionError("focus masks do not contain exactly two fixed latent tiles")

    def shaped(array: np.ndarray, *tail: int) -> np.ndarray:
        return np.ascontiguousarray(array.reshape(batch, time_count, *tail))

    return BlurTiles(
        severity=shaped(severity),
        q_value=shaped(q_value, GRID_SIZE * GRID_SIZE),
        variance=shaped(variance, GRID_SIZE * GRID_SIZE),
        active_index=shaped(active_index, ACTIVE_TILE_COUNT),
        focus_index=shaped(focus_index, FOCUS_TILE_COUNT),
        latent_mask=shaped(mask, 1, latent_size, latent_size),
    )


def construct_predictable_shift(
    theta: np.ndarray,
    p_standard_deviation: np.ndarray,
    local_mask: np.ndarray,
    state_gate: np.ndarray,
    *,
    K_allowance: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct whitened Q* mean shifts without observing an innovation.

    For P ``x_next=mu+sigma*z``, the returned ``u`` defines
    Q* ``N(mu+sigma*u, diag(sigma^2))``.  ``u`` starts from
    ``mask*sigma*theta`` and is explicitly scaled so that
    ``0.5*||u||^2 <= K_allowance``.  Merely normalizing the same full shift to a
    one-dimensional direction would not reduce its LR or cure collapse; the
    cap here changes the operational alternative whenever it binds.
    """

    theta_value = np.asarray(theta)
    sigma = np.asarray(p_standard_deviation)
    mask = np.asarray(local_mask)
    gate = np.asarray(state_gate)
    if theta_value.ndim != 4 or theta_value.shape != sigma.shape:
        raise ValueError("theta and P sigma must have the same [N,C,H,W] shape")
    if mask.shape != (theta_value.shape[0], 1, theta_value.shape[2], theta_value.shape[3]):
        raise ValueError("local mask must be [N,1,H,W]")
    if gate.shape != (theta_value.shape[0],):
        raise ValueError("state gate must be [N]")
    if theta_value.dtype != np.dtype(np.float64) or sigma.dtype != np.dtype(np.float32):
        raise TypeError("canonical theta must be float64 and P sigma must be float32")
    if mask.dtype != np.dtype(np.float64) or gate.dtype != np.dtype(np.bool_):
        raise TypeError("canonical local mask must be float64 and state gate bool")
    if not all(np.isfinite(item).all() for item in (theta_value, sigma, mask)):
        raise ValueError("predictable shift inputs must be finite")
    if np.any(sigma <= 0.0) or np.any((mask != 0.0) & (mask != 1.0)):
        raise ValueError("P sigma must be positive and local mask must be binary")
    allowance = np.asarray(K_allowance, dtype=np.float64)
    if allowance.ndim == 0:
        allowance = np.full(theta_value.shape[0], float(allowance), dtype=np.float64)
    if allowance.shape != (theta_value.shape[0],):
        raise ValueError("K_allowance must be scalar or finite [N]")
    if not np.isfinite(allowance).all() or np.any(allowance < 0.0):
        raise ValueError("K_allowance must be finite and nonnegative")

    raw = (
        sigma.astype(np.float64, copy=False)
        * theta_value
        * mask
        * gate[:, None, None, None]
    )
    raw_K = 0.5 * np.sum(raw * raw, axis=(1, 2, 3), dtype=np.float64)
    scale = np.ones_like(raw_K)
    positive = raw_K > 0.0
    scale[positive] = np.minimum(
        1.0, np.sqrt(allowance[positive] / raw_K[positive])
    )
    whitened_shift = raw * scale[:, None, None, None]
    applied_K = 0.5 * np.sum(
        whitened_shift * whitened_shift, axis=(1, 2, 3), dtype=np.float64
    )
    if np.any(applied_K > allowance * (1.0 + 1e-12)):
        raise AssertionError("predictable Q* shift exceeded its predeclared KL allowance")
    return (
        np.ascontiguousarray(whitened_shift, dtype=np.float64),
        np.ascontiguousarray(raw_K, dtype=np.float64),
        np.ascontiguousarray(applied_K, dtype=np.float64),
    )


def gaussian_log_lr(
    whitened_shift: np.ndarray, transition_innovation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Observe the exact same-covariance Gaussian log-LR after Q* is fixed."""

    shift = np.asarray(whitened_shift)
    innovation = np.asarray(transition_innovation)
    if shift.ndim != 4 or shift.shape != innovation.shape:
        raise ValueError("whitened shift and innovation must match [N,C,H,W]")
    if shift.dtype != np.dtype(np.float64) or innovation.dtype != np.dtype(np.float32):
        raise TypeError("canonical shift must be float64 and innovation float32")
    if not np.isfinite(shift).all() or not np.isfinite(innovation).all():
        raise ValueError("LR inputs must be finite")
    linear = np.sum(
        shift * innovation.astype(np.float64, copy=False),
        axis=(1, 2, 3),
        dtype=np.float64,
    )
    K = 0.5 * np.sum(shift * shift, axis=(1, 2, 3), dtype=np.float64)
    return np.ascontiguousarray(linear), np.ascontiguousarray(linear - K)


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(
        maximum
        + np.log(np.sum(np.exp(values - maximum), axis=axis, keepdims=True)),
        axis=axis,
    )


def compute_eprocess_tracks(
    *,
    theta: np.ndarray,
    p_standard_deviation: np.ndarray,
    transition_innovation: np.ndarray,
    local_mask: np.ndarray,
    blur_severity: np.ndarray,
    blur_gate_threshold: np.ndarray,
    effective_nonidentity: np.ndarray,
    use_state_gate: bool,
) -> EProcessTracks:
    """Compute the fixed two-scale mixture for one frozen gate choice."""

    theta_value = np.asarray(theta)
    sigma = np.asarray(p_standard_deviation)
    innovation = np.asarray(transition_innovation)
    mask = np.asarray(local_mask)
    blur = np.asarray(blur_severity)
    gate_threshold = np.asarray(blur_gate_threshold)
    effective = np.asarray(effective_nonidentity)
    if theta_value.ndim != 6:
        raise ValueError("theta must be [N,scale,time,C,H,W]")
    batch, scale_count, time_count, channels, height, width = theta_value.shape
    if (scale_count, time_count) != (len(HEAT_SHIFTS), len(CHECKPOINTS)):
        raise ValueError("theta scale/time axes differ from the frozen protocol")
    expected_state = (batch, time_count, channels, height, width)
    if sigma.shape != expected_state or innovation.shape != expected_state:
        raise ValueError("P sigma and innovation must be [N,time,C,H,W]")
    if mask.shape != (batch, time_count, 1, height, width):
        raise ValueError("local mask must be [N,time,1,H,W]")
    if blur.shape != (batch, time_count) or gate_threshold.shape != blur.shape:
        raise ValueError("blur severity and gate threshold must be [N,time]")
    if effective.shape != (scale_count, time_count) or effective.dtype != np.uint8:
        raise ValueError("effective_nonidentity must be uint8 [scale,time]")
    if not np.array_equal(effective, np.asarray(EFFECTIVE_NONIDENTITY, dtype=np.uint8)):
        raise ValueError("effective_nonidentity differs from the frozen DiT-250 mapping")
    if theta_value.dtype != np.float64 or sigma.dtype != np.float32 or innovation.dtype != np.float32:
        raise TypeError("theta/sigma/innovation dtypes must be float64/float32/float32")
    if mask.dtype != np.float64 or blur.dtype != np.float64 or gate_threshold.dtype != np.float64:
        raise TypeError("mask, blur severity, and blur thresholds must be float64")
    if not all(np.isfinite(item).all() for item in (theta_value, sigma, innovation, mask, blur, gate_threshold)):
        raise ValueError("e-process arrays must be finite")
    if np.any(theta_value[:, effective == 0] != 0.0):
        raise ValueError("theta must be exactly zero at identity scale/checkpoint pairs")

    raw_K = np.zeros((batch, scale_count, time_count), dtype=np.float64)
    applied_K = np.zeros_like(raw_K)
    increment = np.zeros_like(raw_K)
    component_log_e = np.zeros_like(raw_K)
    cumulative = np.zeros((batch, scale_count), dtype=np.float64)
    mixture_log_e = np.zeros((batch, time_count), dtype=np.float64)
    log_weight = np.log(np.asarray(MIXTURE_WEIGHTS, dtype=np.float64))[None, :]
    state_gate = blur > gate_threshold if use_state_gate else np.ones_like(blur, dtype=np.bool_)

    # A sparse B gate would make an even per-checkpoint allowance use only a
    # small fraction of K_total and would fail the matched-Q* power audit.  The
    # remaining budget is therefore available at every later eligible gate.
    # It is computed before the innovation and can only decrease, so the path
    # total remains bounded.  A sufficiently strong first gated direction may
    # spend the whole budget; a weak direction leaves a predictable residual.
    remaining_K = np.full((batch, scale_count), TOTAL_K_PER_SCALE, dtype=np.float64)
    for time_index in range(time_count):
        for scale_index in range(scale_count):
            if effective[scale_index, time_index] == 0:
                continue
            shift, raw, applied = construct_predictable_shift(
                theta_value[:, scale_index, time_index],
                sigma[:, time_index],
                mask[:, time_index],
                state_gate[:, time_index],
                K_allowance=remaining_K[:, scale_index],
            )
            _, observed_increment = gaussian_log_lr(shift, innovation[:, time_index])
            raw_K[:, scale_index, time_index] = raw
            applied_K[:, scale_index, time_index] = applied
            remaining_K[:, scale_index] = np.maximum(
                0.0, remaining_K[:, scale_index] - applied
            )
            increment[:, scale_index, time_index] = observed_increment
            cumulative[:, scale_index] += observed_increment
        component_log_e[:, :, time_index] = cumulative
        mixture_log_e[:, time_index] = _logsumexp(cumulative + log_weight, axis=1)

    total_K = np.sum(applied_K, axis=2)
    if np.any(total_K > TOTAL_K_PER_SCALE * (1.0 + 1e-12)):
        raise AssertionError("a scale component exceeded the frozen total path KL budget")
    running_max = np.maximum.accumulate(
        np.concatenate([np.zeros((batch, 1)), mixture_log_e], axis=1), axis=1
    )[:, 1:]
    alarm = np.any(running_max >= math.log(1.0 / ALPHA_E), axis=1)
    return EProcessTracks(
        raw_K=np.ascontiguousarray(raw_K),
        applied_K=np.ascontiguousarray(applied_K),
        component_increment=np.ascontiguousarray(increment),
        component_log_e=np.ascontiguousarray(component_log_e),
        mixture_log_e=np.ascontiguousarray(mixture_log_e),
        running_max_log_e=np.ascontiguousarray(running_max),
        alarm=np.ascontiguousarray(alarm),
    )


def blur_summary_tracks(severity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return frozen B persistence and a diagnostic-only formation slope."""

    value = np.asarray(severity)
    if value.ndim != 2 or value.shape[1] != len(CHECKPOINTS):
        raise ValueError("blur severity must be [N,9]")
    if value.dtype != np.float64 or not np.isfinite(value).all():
        raise ValueError("blur severity must be finite float64")
    coordinate = np.arange(value.shape[1], dtype=np.float64)
    centered = coordinate - np.mean(coordinate)
    slope = np.sum(value * centered[None, :], axis=1) / np.sum(centered * centered)
    return np.mean(value, axis=1), slope


def matched_q_power_reference(
    *,
    total_K: float = TOTAL_K_PER_SCALE,
    alpha_e: float = ALPHA_E,
    draws: int = MATCHED_Q_POWER_DRAWS,
    seed: int = MATCHED_Q_POWER_SEED,
) -> dict[str, Any]:
    """Pure-Gaussian necessary power audit for the fixed path mixture.

    Each scale receives its complete K budget at its first eligible gate.  The
    two component directions are represented by orthogonal standard-normal
    coordinates.  Under matched Q_d, coordinate d has exactly the declared
    mean shift and the other coordinate remains P.  This is an oracle
    full-budget reference, not a claim about how often a real B gate opens or
    whether raw cross-scale energy saturates the cap.  If even this reference
    is weak, the real candidate must stop before sampling.
    """

    if not math.isfinite(total_K) or total_K <= 0.0:
        raise ValueError("total_K must be finite and positive")
    if not math.isfinite(alpha_e) or not 0.0 < alpha_e < 1.0:
        raise ValueError("alpha_e must lie in (0,1)")
    if type(draws) is not int or draws < 100_000:
        raise ValueError("matched-Q power audit requires at least 100,000 draws")
    effective = np.asarray(EFFECTIVE_NONIDENTITY, dtype=bool)
    shift = np.zeros((len(HEAT_SHIFTS), len(CHECKPOINTS)), dtype=np.float64)
    for scale_index in range(len(HEAT_SHIFTS)):
        first = int(np.flatnonzero(effective[scale_index])[0])
        shift[scale_index, first] = math.sqrt(2.0 * total_K)
    threshold = math.log(1.0 / alpha_e)
    log_weights = np.log(np.asarray(MIXTURE_WEIGHTS, dtype=np.float64))[None, :, None]
    generator = np.random.default_rng(seed)
    component_results: list[dict[str, float | int]] = []
    for truth in range(len(HEAT_SHIFTS)):
        noise = generator.normal(size=(draws, len(HEAT_SHIFTS), len(CHECKPOINTS)))
        noise[:, truth, :] += shift[truth]
        increment = noise * shift[None, :, :] - 0.5 * shift[None, :, :] ** 2
        component_log = np.cumsum(increment, axis=2)
        mixture_log = _logsumexp(component_log + log_weights, axis=1)
        component_results.append(
            {
                "matched_scale_index": truth,
                "matched_heat_shift": float(HEAT_SHIFTS[truth]),
                "anytime_power": float(np.mean(np.max(mixture_log, axis=1) >= threshold)),
                "terminal_power": float(np.mean(mixture_log[:, -1] >= threshold)),
            }
        )
    mixture_weight = float(min(MIXTURE_WEIGHTS))
    other_fixed_e = 1.0
    required_component_e = (
        1.0 / alpha_e - (1.0 - mixture_weight) * other_fixed_e
    ) / mixture_weight
    z_value = (math.log(required_component_e) - total_K) / math.sqrt(2.0 * total_K)
    analytic_terminal_reference = 0.5 * math.erfc(z_value / math.sqrt(2.0))
    minimum_anytime = min(float(row["anytime_power"]) for row in component_results)
    return {
        "model": "two orthogonal Gaussian component directions; matched Q_d; full K at first eligible gate",
        "draws": draws,
        "seed": seed,
        "total_K_per_component": total_K,
        "alpha_e": alpha_e,
        "component_results": component_results,
        "minimum_anytime_power": minimum_anytime,
        "analytic_terminal_reference_other_component_fixed_at_one": analytic_terminal_reference,
        "minimum_required_anytime_power": MATCHED_Q_ANYTIME_POWER_MINIMUM,
        "passes": minimum_anytime >= MATCHED_Q_ANYTIME_POWER_MINIMUM,
        "scope": "necessary oracle full-budget power gate only; real label-free gate-open and K-utilization gates remain required",
    }


def label_free_operational_power_gate(
    *,
    blur_severity: np.ndarray,
    blur_gate_threshold: np.ndarray,
    applied_K: np.ndarray,
    effective_nonidentity: np.ndarray,
) -> dict[str, Any]:
    """Evaluate the frozen real-data power gate without labels or endpoints."""

    blur = np.asarray(blur_severity)
    threshold = np.asarray(blur_gate_threshold)
    K = np.asarray(applied_K)
    effective = np.asarray(effective_nonidentity)
    if blur.ndim != 2 or blur.shape != threshold.shape:
        raise ValueError("blur severity and threshold must match [N,time]")
    if K.shape != (blur.shape[0], len(HEAT_SHIFTS), blur.shape[1]):
        raise ValueError("applied_K must be [N,scale,time]")
    if effective.shape != (len(HEAT_SHIFTS), blur.shape[1]):
        raise ValueError("effective_nonidentity must be [scale,time]")
    if not all(np.isfinite(value).all() for value in (blur, threshold, K)):
        raise ValueError("label-free operational gate arrays must be finite")
    if np.any(K < 0.0):
        raise ValueError("applied K cannot be negative")
    state_gate = blur > threshold
    scale_rows: list[dict[str, Any]] = []
    for scale_index, heat_shift in enumerate(HEAT_SHIFTS):
        eligible = effective[scale_index].astype(bool)
        open_path = np.any(state_gate[:, eligible], axis=1)
        open_count = int(np.sum(open_path))
        total_K = np.sum(K[:, scale_index], axis=1)
        utilized = open_path & (total_K >= REAL_LABEL_FREE_K_UTILIZATION_THRESHOLD)
        open_fraction = float(np.mean(open_path)) if len(open_path) else 0.0
        utilization_fraction = (
            float(np.sum(utilized) / open_count) if open_count > 0 else 0.0
        )
        scale_rows.append(
            {
                "scale_index": scale_index,
                "heat_shift": float(heat_shift),
                "gate_open_path_count": open_count,
                "gate_open_path_fraction": open_fraction,
                "K_at_least_1p5_among_gate_open_count": int(np.sum(utilized)),
                "K_at_least_1p5_among_gate_open_fraction": utilization_fraction,
                "passes_fraction_thresholds": (
                    open_fraction >= REAL_LABEL_FREE_GATE_OPEN_FRACTION_MINIMUM
                    and utilization_fraction
                    >= REAL_LABEL_FREE_K_UTILIZATION_FRACTION_MINIMUM
                ),
            }
        )
    enough_samples = blur.shape[0] >= REAL_LABEL_FREE_GATE_MINIMUM_SAMPLES
    passes = enough_samples and all(row["passes_fraction_thresholds"] for row in scale_rows)
    return {
        "status": "PASS" if passes else (
            "INSUFFICIENT_SAMPLE_COUNT" if not enough_samples else "FAIL_STOP_BEFORE_LABELS"
        ),
        "sample_count": int(blur.shape[0]),
        "minimum_sample_count": REAL_LABEL_FREE_GATE_MINIMUM_SAMPLES,
        "gate_open_fraction_minimum": REAL_LABEL_FREE_GATE_OPEN_FRACTION_MINIMUM,
        "K_utilization_threshold": REAL_LABEL_FREE_K_UTILIZATION_THRESHOLD,
        "K_utilization_fraction_minimum": (
            REAL_LABEL_FREE_K_UTILIZATION_FRACTION_MINIMUM
        ),
        "scale_results": scale_rows,
        "passes": passes,
        "labels_endpoint_images_external_representations_used": False,
        "failure_action": "STOP before label join; do not tune K, alpha, B gate, masks, shifts, or weights",
    }


def _validate_input_keys(keys: Iterable[str]) -> None:
    names = tuple(keys)
    if set(names) != set(INPUT_ARRAY_NAMES):
        raise RuntimeError(
            f"observer input keys differ from the exact label-free schema: {sorted(names)}"
        )
    for name in names:
        lowered = name.lower()
        if any(token in lowered for token in FORBIDDEN_INPUT_KEY_TOKENS):
            raise RuntimeError(f"forbidden supervision/external-representation input key: {name}")


def validate_observer_input(arrays: Mapping[str, np.ndarray]) -> None:
    _validate_input_keys(arrays.keys())
    images = arrays["decoded_pred_xstart_rgb"]
    theta = arrays["theta"]
    sigma = arrays["p_standard_deviation"]
    innovation = arrays["transition_innovation"]
    if images.dtype != np.float32 or images.ndim != 5:
        raise RuntimeError("decoded_pred_xstart_rgb must be float32 [N,9,3,H,W]")
    batch = images.shape[0]
    if theta.dtype != np.float64 or theta.shape != (
        batch,
        len(HEAT_SHIFTS),
        len(CHECKPOINTS),
        4,
        32,
        32,
    ):
        raise RuntimeError("theta has the wrong frozen shape or dtype")
    expected_state = (batch, len(CHECKPOINTS), 4, 32, 32)
    if sigma.dtype != np.float32 or sigma.shape != expected_state:
        raise RuntimeError("p_standard_deviation has the wrong frozen shape or dtype")
    if innovation.dtype != np.float32 or innovation.shape != expected_state:
        raise RuntimeError("transition_innovation has the wrong frozen shape or dtype")
    if arrays["sampling_step"].dtype != np.int16 or not np.array_equal(
        arrays["sampling_step"], np.asarray(CHECKPOINTS, dtype=np.int16)
    ):
        raise RuntimeError("sampling_step differs from 69..149 by ten")
    if arrays["shifted_internal_timestep"].dtype != np.int16 or not np.array_equal(
        arrays["shifted_internal_timestep"],
        np.asarray(SHIFTED_INTERNAL_TIMESTEPS, dtype=np.int16),
    ):
        raise RuntimeError("shifted timestep mapping differs from the frozen schedule")
    if arrays["heat_shift"].dtype != np.float64 or not np.array_equal(
        arrays["heat_shift"], np.asarray(HEAT_SHIFTS, dtype=np.float64)
    ):
        raise RuntimeError("heat_shift differs from the frozen two-scale mixture")
    if arrays["effective_nonidentity"].dtype != np.uint8 or not np.array_equal(
        arrays["effective_nonidentity"], np.asarray(EFFECTIVE_NONIDENTITY, dtype=np.uint8)
    ):
        raise RuntimeError("effective_nonidentity differs from the frozen schedule")
    if arrays["blur_gate_threshold"].dtype != np.float64 or arrays[
        "blur_gate_threshold"
    ].shape != (batch, len(CHECKPOINTS)):
        raise RuntimeError("blur_gate_threshold must be float64 [N,9]")
    if arrays["blur_score_threshold"].dtype != np.float64 or arrays[
        "blur_score_threshold"
    ].shape != (batch,):
        raise RuntimeError("blur_score_threshold must be float64 [N]")
    if arrays["class_id"].shape != (batch,) or not np.issubdtype(
        arrays["class_id"].dtype, np.integer
    ):
        raise RuntimeError("class_id must be an integer [N] array")
    floating = (
        images,
        theta,
        sigma,
        innovation,
        arrays["blur_gate_threshold"],
        arrays["blur_score_threshold"],
    )
    if not all(np.isfinite(item).all() for item in floating):
        raise RuntimeError("observer input contains non-finite values")
    if np.any(images < 0.0) or np.any(images > 1.0) or np.any(sigma <= 0.0):
        raise RuntimeError("RGB must be in [0,1] and P sigma must be positive")
    effective = arrays["effective_nonidentity"].astype(bool)
    if np.any(theta[:, ~effective] != 0.0):
        raise RuntimeError("identity cross-scale rows must have exactly zero theta")


def extract_label_free(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    validate_observer_input(arrays)
    blur = compute_blur_tiles(arrays["decoded_pred_xstart_rgb"])
    B_persistence, formation_slope = blur_summary_tracks(blur.severity)
    gated = compute_eprocess_tracks(
        theta=arrays["theta"],
        p_standard_deviation=arrays["p_standard_deviation"],
        transition_innovation=arrays["transition_innovation"],
        local_mask=blur.latent_mask,
        blur_severity=blur.severity,
        blur_gate_threshold=arrays["blur_gate_threshold"],
        effective_nonidentity=arrays["effective_nonidentity"],
        use_state_gate=True,
    )
    ungated = compute_eprocess_tracks(
        theta=arrays["theta"],
        p_standard_deviation=arrays["p_standard_deviation"],
        transition_innovation=arrays["transition_innovation"],
        local_mask=blur.latent_mask,
        blur_severity=blur.severity,
        blur_gate_threshold=arrays["blur_gate_threshold"],
        effective_nonidentity=arrays["effective_nonidentity"],
        use_state_gate=False,
    )
    B_alarm = B_persistence > arrays["blur_score_threshold"]
    return {
        "blur": blur,
        "B_persistence": np.ascontiguousarray(B_persistence),
        "B_formation_slope_diagnostic": np.ascontiguousarray(formation_slope),
        "B_alarm": np.ascontiguousarray(B_alarm),
        "gated": gated,
        "ungated": ungated,
    }


def _protocol_snapshot() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "status": "LABEL_FREE_OBSERVATION_ONLY_NOT_VALIDATED_FOR_INTERVENTION",
        "checkpoints": list(CHECKPOINTS),
        "latest_internal_timestep": INTERNAL_TIMESTEPS[-1],
        "heat_shifts": list(HEAT_SHIFTS),
        "shifted_internal_timesteps": [list(row) for row in SHIFTED_INTERNAL_TIMESTEPS],
        "mixture_weights": list(MIXTURE_WEIGHTS),
        "total_K_per_scale_component": TOTAL_K_PER_SCALE,
        "K_spending": (
            "at each eligible gate, cap by the complete remaining per-scale path budget; "
            "unused residual carries forward predictably and total K can only decrease"
        ),
        "alpha_e": ALPHA_E,
        "matched_Q_power_gate": {
            "draws": MATCHED_Q_POWER_DRAWS,
            "seed": MATCHED_Q_POWER_SEED,
            "minimum_anytime_power": MATCHED_Q_ANYTIME_POWER_MINIMUM,
            "reference": (
                "two orthogonal scale directions, data from each matched Q_d in turn, "
                "full K spent at first eligible gate"
            ),
            "necessary_not_sufficient": True,
        },
        "real_label_free_operational_power_gate": {
            "minimum_samples": REAL_LABEL_FREE_GATE_MINIMUM_SAMPLES,
            "per_scale_gate_open_path_fraction_minimum": (
                REAL_LABEL_FREE_GATE_OPEN_FRACTION_MINIMUM
            ),
            "per_scale_K_threshold": REAL_LABEL_FREE_K_UTILIZATION_THRESHOLD,
            "fraction_of_gate_open_paths_reaching_K_threshold_minimum": (
                REAL_LABEL_FREE_K_UTILIZATION_FRACTION_MINIMUM
            ),
            "failure_action": "STOP before labels; no tuning on this pool",
        },
        "localization": {
            "grid": "fixed 4x4 in decoded draft and matching normalized latent coordinates",
            "active_tiles": ACTIVE_TILE_COUNT,
            "focus_tiles": FOCUS_TILE_COUNT,
            "focus_rule": "lowest q, then row-major index, among highest-variance active tiles",
        },
        "candidate_roles": {
            "B_persistence": "predictable heuristic anchor; not an e-process",
            "E_blur_gated": "primary exact operational P/Q* e-process candidate",
            "E_no_state_gate": "exact mechanism ablation only; cannot replace the primary",
            "B_formation_slope": "diagnostic-only predictable statistic",
        },
        "external_representation_policy": (
            "FID, Inception, DINO, CLIP, endpoint images, visual labels, reviewer decisions, "
            "and quality posteriors are forbidden inputs; they may only be joined later by a "
            "separately locked evaluator after label/event gates"
        ),
        "exactness_scope": (
            "same-covariance Gaussian operational Q* with predictable gate/mask/direction; "
            "localization and KL scaling change the alternative, so this is not the ideal "
            "cross-scale marginal density ratio"
        ),
    }


def publish(input_path: Path, output_dir: Path) -> Path:
    if output_dir.exists() or output_dir.is_symlink():
        raise RuntimeError(f"refusing pre-existing output path: {output_dir}")
    if not input_path.is_file() or input_path.is_symlink():
        raise RuntimeError("input must be one regular, non-symlink NPZ")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    try:
        with np.load(input_path, allow_pickle=False) as archive:
            _validate_input_keys(archive.files)
            arrays = {name: np.ascontiguousarray(archive[name]) for name in INPUT_ARRAY_NAMES}
        result = extract_label_free(arrays)
        blur: BlurTiles = result["blur"]
        gated: EProcessTracks = result["gated"]
        ungated: EProcessTracks = result["ungated"]

        tracks = {
            "sampling_step": np.asarray(CHECKPOINTS, dtype=np.int16),
            "heat_shift": np.asarray(HEAT_SHIFTS, dtype=np.float64),
            "blur_severity": blur.severity,
            "blur_q_by_tile": blur.q_value,
            "blur_variance_by_tile": blur.variance,
            "active_tile_index": blur.active_index,
            "focus_tile_index": blur.focus_index,
            "gated_raw_K": gated.raw_K,
            "gated_applied_K": gated.applied_K,
            "gated_component_increment": gated.component_increment,
            "gated_component_log_e": gated.component_log_e,
            "gated_mixture_log_e": gated.mixture_log_e,
            "ungated_raw_K": ungated.raw_K,
            "ungated_applied_K": ungated.applied_K,
            "ungated_component_increment": ungated.component_increment,
            "ungated_component_log_e": ungated.component_log_e,
            "ungated_mixture_log_e": ungated.mixture_log_e,
        }
        tracks_path = staging / "tracks.npz"
        with tracks_path.open("wb") as handle:
            np.savez(handle, **tracks)

        feature_rows: list[dict[str, Any]] = []
        for row in range(len(result["B_persistence"])):
            feature_rows.append(
                {
                    "sample_index": row,
                    "class_id": int(arrays["class_id"][row]),
                    "B_persistence": float(result["B_persistence"][row]),
                    "B_formation_slope_diagnostic": float(
                        result["B_formation_slope_diagnostic"][row]
                    ),
                    "B_alarm": int(result["B_alarm"][row]),
                    "E_blur_gated_running_max_log": float(
                        np.max(gated.running_max_log_e[row], initial=0.0)
                    ),
                    "E_blur_gated_alarm": int(gated.alarm[row]),
                    "E_no_state_gate_running_max_log_ablation": float(
                        np.max(ungated.running_max_log_e[row], initial=0.0)
                    ),
                    "E_no_state_gate_alarm_ablation": int(ungated.alarm[row]),
                    "blur_gate_open_count": int(
                        np.sum(blur.severity[row] > arrays["blur_gate_threshold"][row])
                    ),
                    "gated_total_K_max_scale": float(np.max(np.sum(gated.applied_K[row], axis=1))),
                    "ungated_total_K_max_scale": float(
                        np.max(np.sum(ungated.applied_K[row], axis=1))
                    ),
                }
            )
        csv_path = staging / "sample_features.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(feature_rows[0]))
            writer.writeheader()
            writer.writerows(feature_rows)

        catalog = {
            "features": [
                {
                    "name": "B_persistence",
                    "role": "co-primary internal heuristic anchor",
                    "direction": "bad_high",
                    "availability": "predictable_at_sampling_step_149",
                    "exact_eprocess": False,
                },
                {
                    "name": "E_blur_gated_running_max_log",
                    "role": "co-primary internal exact operational e-process score",
                    "direction": "bad_high",
                    "availability": "online_causal_through_sampling_step_149",
                    "exact_eprocess": (
                        "the underlying E_k is exact; running-max log is its reporting score, "
                        "not itself a martingale"
                    ),
                },
                {
                    "name": "E_no_state_gate_running_max_log_ablation",
                    "role": "fixed exact ablation; no candidate substitution",
                    "direction": "bad_high",
                    "exact_eprocess": True,
                },
                {
                    "name": "B_formation_slope_diagnostic",
                    "role": "diagnostic only; no candidate substitution",
                    "direction": "larger means less sharpening/more risk",
                    "exact_eprocess": False,
                },
            ],
            "labels_read_or_emitted": False,
            "endpoint_images_read": False,
            "external_representations_read": False,
        }
        _atomic_json_dump(_protocol_snapshot(), staging / "protocol_snapshot.json")
        _atomic_json_dump(catalog, staging / "feature_catalog.json")
        operational_gate = label_free_operational_power_gate(
            blur_severity=blur.severity,
            blur_gate_threshold=arrays["blur_gate_threshold"],
            applied_K=gated.applied_K,
            effective_nonidentity=arrays["effective_nonidentity"],
        )
        _atomic_json_dump(
            operational_gate, staging / "label_free_operational_power_gate.json"
        )

        outputs = []
        for path in sorted(staging.iterdir(), key=lambda item: item.name):
            if path.name == "manifest.json":
                continue
            outputs.append(
                {
                    "relative_path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "status": "complete",
            "input": {
                "path": str(input_path.resolve()),
                "bytes": input_path.stat().st_size,
                "sha256": _sha256_file(input_path),
                "arrays": {name: _array_record(arrays[name]) for name in INPUT_ARRAY_NAMES},
            },
            "implementation": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256_file(Path(__file__).resolve()),
            },
            "sample_count": len(feature_rows),
            "supervision": "none",
            "label_free_operational_power_gate_status": operational_gate["status"],
            "label_free_operational_power_gate_passed": operational_gate["passes"],
            "label_join_authorized_by_this_product": False,
            "outputs": outputs,
            "outputs_sha256": _sha256_json(outputs),
        }
        manifest["identity_sha256"] = _sha256_json(manifest)
        _atomic_json_dump(manifest, staging / "manifest.json")
        os.replace(staging, output_dir)
        return output_dir
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def self_test() -> None:
    rng = np.random.default_rng(2026082807)
    height = width = 64
    checker = (np.indices((height, width)).sum(axis=0) % 2).astype(np.float32)
    sharp = np.stack([checker, checker, checker], axis=0)
    blurred_gray = ndimage.gaussian_filter(checker, sigma=3.0, mode="reflect")
    blurred = np.stack([blurred_gray, blurred_gray, blurred_gray], axis=0).astype(np.float32)
    image_rows = []
    for image in (sharp, blurred):
        image_rows.append(np.stack([image] * len(CHECKPOINTS), axis=0))
    images = np.stack(image_rows, axis=0).astype(np.float32)
    blur = compute_blur_tiles(images)
    if not np.all(blur.severity[1] > blur.severity[0]):
        raise AssertionError("synthetic blurred drafts are not ordered above sharp drafts")
    if blur.focus_index.shape != (2, len(CHECKPOINTS), FOCUS_TILE_COUNT):
        raise AssertionError("focus tile shape changed")

    # The Gaussian increment must equal a direct same-covariance density ratio.
    shift = np.ascontiguousarray(np.asarray([[[[0.2, -0.1, 0.15]]]], dtype=np.float64))
    noise = np.ascontiguousarray(np.asarray([[[[0.4, -0.7, 0.2]]]], dtype=np.float32))
    _, observed = gaussian_log_lr(shift, noise)
    x = noise.astype(np.float64)
    direct = -0.5 * np.sum((x - shift) ** 2, axis=(1, 2, 3)) + 0.5 * np.sum(
        x * x, axis=(1, 2, 3)
    )
    if not np.allclose(observed, direct, rtol=0.0, atol=1e-15):
        raise AssertionError("Gaussian LR differs from direct Normal density ratio")

    # A rank-one rewrite that preserves ||u|| is exactly the same LR and does
    # not solve collapse.  Only the explicit K cap below changes Q*.
    raw_vector = rng.normal(size=(1, 4, 8, 8)).astype(np.float64)
    raw_norm = float(np.linalg.norm(raw_vector))
    rank_one_reconstruction = (raw_vector / raw_norm) * raw_norm
    if not np.array_equal(rank_one_reconstruction, raw_vector):
        if not np.allclose(rank_one_reconstruction, raw_vector, rtol=0.0, atol=1e-15):
            raise AssertionError("rank-one full-direction LR identity failed")
    theta = np.ascontiguousarray(raw_vector, dtype=np.float64)
    sigma = np.ascontiguousarray(np.full(theta.shape, 0.5), dtype=np.float32)
    mask = np.ascontiguousarray(np.ones((1, 1, 8, 8)), dtype=np.float64)
    gate = np.asarray([True], dtype=np.bool_)
    capped, raw_K, applied_K = construct_predictable_shift(
        theta, sigma, mask, gate, K_allowance=0.01
    )
    if not raw_K[0] > applied_K[0] or applied_K[0] > 0.01 * (1.0 + 1e-12):
        raise AssertionError("explicit Q* KL shrinkage test failed")
    if not np.linalg.norm(capped) < np.linalg.norm(sigma.astype(np.float64) * theta):
        raise AssertionError("the binding cap did not actually change the operational Q")

    # Monte Carlo calibration of one exact Gaussian e-value.
    monte_shift = np.asarray([0.2, -0.1, 0.15], dtype=np.float64)
    monte_noise = rng.normal(size=(250_000, 3))
    log_lr = monte_noise @ monte_shift - 0.5 * float(monte_shift @ monte_shift)
    empirical = float(np.mean(np.exp(log_lr)))
    if not abs(empirical - 1.0) < 0.006:
        raise AssertionError(f"Monte Carlo e-value calibration failed: {empirical}")

    # Selecting the largest component after seeing z is not a mixture e-value.
    two_noise = rng.normal(size=(250_000, 2))
    component_e = np.exp(0.35 * two_noise - 0.5 * 0.35**2)
    invalid_posthoc = float(np.mean(np.max(component_e, axis=1)))
    valid_mixture = float(np.mean(np.mean(component_e, axis=1)))
    if not invalid_posthoc > 1.1 or not abs(valid_mixture - 1.0) < 0.006:
        raise AssertionError("post-innovation maximum versus fixed mixture witness failed")

    batch = 2
    synthetic_theta = rng.normal(
        size=(batch, len(HEAT_SHIFTS), len(CHECKPOINTS), 4, 32, 32)
    ).astype(np.float64)
    effective = np.asarray(EFFECTIVE_NONIDENTITY, dtype=np.uint8)
    synthetic_theta[:, effective == 0] = 0.0
    synthetic_sigma = np.full((batch, len(CHECKPOINTS), 4, 32, 32), 0.2, dtype=np.float32)
    synthetic_noise = rng.normal(size=synthetic_sigma.shape).astype(np.float32)
    thresholds = np.repeat(np.mean(blur.severity, axis=0, keepdims=True), batch, axis=0)
    tracks = compute_eprocess_tracks(
        theta=np.ascontiguousarray(synthetic_theta),
        p_standard_deviation=synthetic_sigma,
        transition_innovation=synthetic_noise,
        local_mask=blur.latent_mask,
        blur_severity=blur.severity,
        blur_gate_threshold=np.ascontiguousarray(thresholds, dtype=np.float64),
        effective_nonidentity=effective,
        use_state_gate=True,
    )
    if np.any(np.sum(tracks.applied_K, axis=2) > TOTAL_K_PER_SCALE * (1 + 1e-12)):
        raise AssertionError("full-track K_total bound failed")

    # K=0.5 was initially attractive for second moments but is analytically
    # too weak after the 50/50 mixture dilution.  Reject it before any real
    # trace, then require the frozen K=2 design to clear the matched-Q gate.
    weak_power = matched_q_power_reference(total_K=0.5, draws=120_000, seed=91)
    if weak_power["minimum_anytime_power"] >= 0.03:
        raise AssertionError("the deliberately weak K=0.5 witness was unexpectedly powerful")
    power = matched_q_power_reference()
    if power["passes"] is not True:
        raise AssertionError(f"matched-Q power gate failed: {power}")
    operational_blur = np.ones(
        (REAL_LABEL_FREE_GATE_MINIMUM_SAMPLES, len(CHECKPOINTS)), dtype=np.float64
    )
    operational_threshold = np.zeros_like(operational_blur)
    operational_K = np.zeros(
        (
            REAL_LABEL_FREE_GATE_MINIMUM_SAMPLES,
            len(HEAT_SHIFTS),
            len(CHECKPOINTS),
        ),
        dtype=np.float64,
    )
    for scale_index, row in enumerate(np.asarray(EFFECTIVE_NONIDENTITY, dtype=bool)):
        operational_K[:, scale_index, int(np.flatnonzero(row)[0])] = TOTAL_K_PER_SCALE
    operational = label_free_operational_power_gate(
        blur_severity=operational_blur,
        blur_gate_threshold=operational_threshold,
        applied_K=operational_K,
        effective_nonidentity=np.asarray(EFFECTIVE_NONIDENTITY, dtype=np.uint8),
    )
    if operational["passes"] is not True:
        raise AssertionError(f"label-free operational power gate fixture failed: {operational}")

    poison_keys = list(INPUT_ARRAY_NAMES[:-1]) + ["visual_label"]
    try:
        _validate_input_keys(poison_keys)
    except RuntimeError:
        pass
    else:
        raise AssertionError("label-like poison input was not rejected")
    print(
        "self-test passed: B ordering/focus masks, direct Gaussian LR, explicit Q* "
        f"shrinkage, E calibration ({empirical:.6f}), invalid post-hoc max witness "
        f"({invalid_posthoc:.6f}), fixed mixture ({valid_mixture:.6f}), total-K bound, "
        f"matched-Q anytime power ({power['minimum_anytime_power']:.6f}; weak K=0.5 "
        f"witness {weak_power['minimum_anytime_power']:.6f}), label-free operational "
        "power gate, and supervision poison rejection"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.self_test:
        if args.input is not None or args.output_dir is not None:
            parser.error("--self-test cannot be combined with real inputs")
        self_test()
        return 0
    if args.input is None or args.output_dir is None:
        parser.error("real extraction requires both --input and --output-dir")
    result = publish(args.input.expanduser().resolve(), args.output_dir.expanduser().absolute())
    print(f"published label-free blur-focused e-process features: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
