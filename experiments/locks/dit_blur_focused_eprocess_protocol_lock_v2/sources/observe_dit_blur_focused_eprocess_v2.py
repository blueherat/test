#!/usr/bin/env python3
"""Corrected label-free blur-latched cross-scale DiT e-process (method v2).

Version 2 jointly corrects the start, information schedule, and operational
direction magnitude.  For each
scale, the first B threshold crossing that still has at least three effective
checkpoints (including the current one) starts the alternative.  At that
predictable first hit, ``h`` is frozen to the remaining effective-checkpoint
count and every checkpoint from then on receives the fixed allowance
``kappa=K_total/h``.  Each current nonzero predictable direction is normalized
to spend exactly that amount; if a later direction is numerically unavailable,
the last valid unit direction is reused.  Consequently, one early gate cannot exhaust the path
budget, while a rare B crossing initiates a genuinely multi-step alternative.

The B statistic, predictable B gate, localization mask, score-gap direction,
same-covariance Gaussian likelihood ratio, and fixed two-scale mixture are
unchanged from immutable method v1.  External labels, endpoint images, FID,
Inception, DINO, CLIP, and learned quality scores remain forbidden inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

try:
    from . import observe_dit_blur_focused_eprocess as v1
except ImportError:  # pragma: no cover - direct CLI execution
    import observe_dit_blur_focused_eprocess as v1


EXPERIMENT = "dit_blur_focused_cross_scale_eprocess_label_free_v2"
SCHEMA_VERSION = 2

# Frozen and unchanged scientific ingredients inherited from immutable v1.
CHECKPOINTS = v1.CHECKPOINTS
INTERNAL_TIMESTEPS = v1.INTERNAL_TIMESTEPS
HEAT_SHIFTS = v1.HEAT_SHIFTS
SHIFTED_INTERNAL_TIMESTEPS = v1.SHIFTED_INTERNAL_TIMESTEPS
EFFECTIVE_NONIDENTITY = v1.EFFECTIVE_NONIDENTITY
MIXTURE_WEIGHTS = v1.MIXTURE_WEIGHTS
TOTAL_K_PER_SCALE = v1.TOTAL_K_PER_SCALE
ALPHA_E = v1.ALPHA_E
MATCHED_Q_POWER_DRAWS = v1.MATCHED_Q_POWER_DRAWS
MATCHED_Q_POWER_SEED = v1.MATCHED_Q_POWER_SEED
MATCHED_Q_ANYTIME_POWER_MINIMUM = v1.MATCHED_Q_ANYTIME_POWER_MINIMUM
GRID_SIZE = v1.GRID_SIZE
ACTIVE_TILE_COUNT = v1.ACTIVE_TILE_COUNT
FOCUS_TILE_COUNT = v1.FOCUS_TILE_COUNT
EPS = v1.EPS
RAW_DIRECTION_NORM_FLOOR = 1e-8
INPUT_ARRAY_NAMES = v1.INPUT_ARRAY_NAMES
FORBIDDEN_INPUT_KEY_TOKENS = v1.FORBIDDEN_INPUT_KEY_TOKENS
V1_OBSERVER_SOURCE_SHA256 = "edd492d4e9e32031d5aa27c7e932ed9d6ed398d46c43fc7e79cd692bdeecb199"

EFFECTIVE_STEP_COUNT_PER_SCALE = tuple(
    int(sum(row)) for row in EFFECTIVE_NONIDENTITY
)
FIXED_K_PER_EFFECTIVE_CHECKPOINT = tuple(
    TOTAL_K_PER_SCALE / count for count in EFFECTIVE_STEP_COUNT_PER_SCALE
)

# The real-data gate is deliberately a mechanics audit, not a proxy quality
# target.  It asks only whether Q* actually operates over multiple times.
PATH_MECHANICS_MINIMUM_SAMPLES = 768
PATH_MECHANICS_MINIMUM_STARTED_PATHS_PER_SCALE = 12
PATH_MECHANICS_MINIMUM_STARTED_CLASSES_PER_SCALE = 3
PATH_MECHANICS_COMPLETE_COVERAGE_FRACTION_MINIMUM = 1.0
PATH_MECHANICS_MAX_REUSED_DIRECTION_FRACTION = 0.01
MINIMUM_REMAINING_EFFECTIVE_AT_START = 3

BlurTiles = v1.BlurTiles
compute_blur_tiles = v1.compute_blur_tiles
gaussian_log_lr = v1.gaussian_log_lr
blur_summary_tracks = v1.blur_summary_tracks
_sha256_file = v1._sha256_file
_sha256_json = v1._sha256_json
_atomic_json_dump = v1._atomic_json_dump
_array_record = v1._array_record
_logsumexp = v1._logsumexp

if _sha256_file(Path(v1.__file__).resolve()) != V1_OBSERVER_SOURCE_SHA256:
    raise RuntimeError("method v2 refuses an unpinned v1 observer dependency")


@dataclass(frozen=True)
class EProcessTracks:
    raw_K: np.ndarray
    applied_K: np.ndarray
    component_increment: np.ndarray
    component_log_e: np.ndarray
    mixture_log_e: np.ndarray
    running_max_log_e: np.ndarray
    alarm: np.ndarray
    start_time_index: np.ndarray
    start_remaining_effective_count: np.ndarray
    frozen_K_per_step_after_start: np.ndarray
    direction_reused: np.ndarray


def construct_predictable_shift(
    theta: np.ndarray,
    p_standard_deviation: np.ndarray,
    local_mask: np.ndarray,
    state_gate: np.ndarray,
    *,
    K_allowance: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct the fixed-information directional operational alternative.

    Let ``w=mask*sigma*theta``.  When the predictable gate is open and
    ``||w||`` clears the frozen numerical floor, v2 sets
    ``u=sqrt(2*kappa)*w/||w||``.  Thus the Gaussian alternative spends exactly
    ``kappa`` nats of one-step KL regardless of the raw direction magnitude.
    This is an explicit new directional Q*, not a claim that normalizing the
    old full shift preserves its likelihood ratio.  This standalone constructor
    returns ``u=0`` below the floor; the latched path routine instead reuses its
    already-valid previous unit direction after start.
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
    if theta_value.dtype != np.float64 or sigma.dtype != np.float32:
        raise TypeError("canonical theta must be float64 and P sigma float32")
    if mask.dtype != np.float64 or gate.dtype != np.bool_:
        raise TypeError("canonical mask must be float64 and gate bool")
    if not all(np.isfinite(item).all() for item in (theta_value, sigma, mask)):
        raise ValueError("direction inputs must be finite")
    if np.any(sigma <= 0.0) or np.any((mask != 0.0) & (mask != 1.0)):
        raise ValueError("P sigma must be positive and mask binary")
    allowance = np.asarray(K_allowance, dtype=np.float64)
    if allowance.ndim == 0:
        allowance = np.full(theta_value.shape[0], float(allowance), dtype=np.float64)
    if allowance.shape != (theta_value.shape[0],):
        raise ValueError("K_allowance must be scalar or [N]")
    if not np.isfinite(allowance).all() or np.any(allowance < 0.0):
        raise ValueError("K_allowance must be finite and nonnegative")

    raw = sigma.astype(np.float64, copy=False) * theta_value * mask
    norm_squared = np.sum(raw * raw, axis=(1, 2, 3), dtype=np.float64)
    raw_K = 0.5 * norm_squared
    norm = np.sqrt(norm_squared)
    direction_valid = gate & (norm >= RAW_DIRECTION_NORM_FLOOR) & (allowance > 0.0)
    scale = np.zeros_like(norm)
    scale[direction_valid] = (
        np.sqrt(2.0 * allowance[direction_valid]) / norm[direction_valid]
    )
    whitened_shift = raw * scale[:, None, None, None]
    applied_K = 0.5 * np.sum(
        whitened_shift * whitened_shift, axis=(1, 2, 3), dtype=np.float64
    )
    if not np.allclose(
        applied_K[direction_valid], allowance[direction_valid], rtol=2e-12, atol=1e-14
    ):
        raise AssertionError("fixed-information shift did not spend its declared KL")
    if np.any(applied_K[~direction_valid] != 0.0):
        raise AssertionError("invalid or closed directions spent KL")
    return (
        np.ascontiguousarray(whitened_shift, dtype=np.float64),
        np.ascontiguousarray(raw_K, dtype=np.float64),
        np.ascontiguousarray(applied_K, dtype=np.float64),
        np.ascontiguousarray(direction_valid, dtype=np.bool_),
    )


def _validate_track_inputs(
    *,
    theta: np.ndarray,
    p_standard_deviation: np.ndarray,
    transition_innovation: np.ndarray,
    local_mask: np.ndarray,
    blur_severity: np.ndarray,
    blur_gate_threshold: np.ndarray,
    effective_nonidentity: np.ndarray,
) -> tuple[np.ndarray, ...]:
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
        raise ValueError("theta scale/time axes differ from method v2")
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
        raise ValueError("effective_nonidentity differs from frozen DiT-250 mapping")
    if (
        theta_value.dtype != np.float64
        or sigma.dtype != np.float32
        or innovation.dtype != np.float32
    ):
        raise TypeError("theta/sigma/innovation dtypes must be float64/float32/float32")
    if (
        mask.dtype != np.float64
        or blur.dtype != np.float64
        or gate_threshold.dtype != np.float64
    ):
        raise TypeError("mask, blur severity, and blur thresholds must be float64")
    if not all(
        np.isfinite(item).all()
        for item in (theta_value, sigma, innovation, mask, blur, gate_threshold)
    ):
        raise ValueError("e-process arrays must be finite")
    if np.any(theta_value[:, effective == 0] != 0.0):
        raise ValueError("theta must be exactly zero at identity pairs")
    return theta_value, sigma, innovation, mask, blur, gate_threshold, effective


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
    one_shot_full_budget: bool = False,
) -> EProcessTracks:
    """Compute method-v2 tracks with a predictable start and no carry.

    ``use_state_gate=True`` gives the primary latched alternative.  The first
    qualifying B crossing freezes its start and per-step allowance; after that
    the alternative remains active even if B falls below threshold.  With
    ``one_shot_full_budget=True`` uses the complete K=2 budget at that
    predictable first hit and makes all later increments identity.  This is a
    distinct exact one-shot alternative, not a stopped version of the primary
    (the primary spends only 2/h at its first step).  With
    ``use_state_gate=False`` the no-gate ablation starts at the first effective
    checkpoint and distributes K over all effective checkpoints.
    """

    if one_shot_full_budget and not use_state_gate:
        raise ValueError("one_shot_full_budget requires the predictable B start rule")

    (
        theta_value,
        sigma,
        innovation,
        mask,
        blur,
        gate_threshold,
        effective,
    ) = _validate_track_inputs(
        theta=theta,
        p_standard_deviation=p_standard_deviation,
        transition_innovation=transition_innovation,
        local_mask=local_mask,
        blur_severity=blur_severity,
        blur_gate_threshold=blur_gate_threshold,
        effective_nonidentity=effective_nonidentity,
    )
    batch, scale_count, time_count = theta_value.shape[:3]
    raw_K = np.zeros((batch, scale_count, time_count), dtype=np.float64)
    applied_K = np.zeros_like(raw_K)
    increment = np.zeros_like(raw_K)
    component_log_e = np.zeros_like(raw_K)
    cumulative = np.zeros((batch, scale_count), dtype=np.float64)
    mixture_log_e = np.zeros((batch, time_count), dtype=np.float64)
    log_weight = np.log(np.asarray(MIXTURE_WEIGHTS, dtype=np.float64))[None, :]
    crossing = blur > gate_threshold
    started = np.zeros((batch, scale_count), dtype=np.bool_)
    start_time_index = np.full((batch, scale_count), -1, dtype=np.int16)
    start_remaining = np.zeros((batch, scale_count), dtype=np.int16)
    frozen_allowance = np.zeros((batch, scale_count), dtype=np.float64)
    last_unit_direction = np.zeros(
        (batch, scale_count, theta_value.shape[3], theta_value.shape[4], theta_value.shape[5]),
        dtype=np.float64,
    )
    direction_reused = np.zeros((batch, scale_count, time_count), dtype=np.bool_)
    remaining_by_scale_time = np.flip(
        np.cumsum(np.flip(effective, axis=1), axis=1), axis=1
    ).astype(np.int16)

    # Crucial v2 correction: after the predictable start, allowance is frozen
    # per step.  It never depends on realized prior KL, and no checkpoint's
    # kappa can be transferred to another.  The one-shot ablation uses K_total once to
    # compare a separate equal-total-budget single-step alternative.
    for time_index in range(time_count):
        for scale_index in range(scale_count):
            if effective[scale_index, time_index] == 0:
                continue
            raw_direction = (
                sigma[:, time_index].astype(np.float64, copy=False)
                * theta_value[:, scale_index, time_index]
                * mask[:, time_index]
            )
            direction_available = (
                np.sqrt(
                    np.sum(
                        raw_direction * raw_direction,
                        axis=(1, 2, 3),
                        dtype=np.float64,
                    )
                )
                >= RAW_DIRECTION_NORM_FLOOR
            )
            if use_state_gate:
                qualifying = (
                    ~started[:, scale_index]
                    & crossing[:, time_index]
                    & direction_available
                    & (
                        remaining_by_scale_time[scale_index, time_index]
                        >= MINIMUM_REMAINING_EFFECTIVE_AT_START
                    )
                )
            else:
                qualifying = (
                    ~started[:, scale_index]
                    & direction_available
                    & (
                        remaining_by_scale_time[scale_index, time_index]
                        >= MINIMUM_REMAINING_EFFECTIVE_AT_START
                    )
                )
            if np.any(qualifying):
                started[qualifying, scale_index] = True
                start_time_index[qualifying, scale_index] = time_index
                h = int(remaining_by_scale_time[scale_index, time_index])
                start_remaining[qualifying, scale_index] = h
                frozen_allowance[qualifying, scale_index] = (
                    TOTAL_K_PER_SCALE if one_shot_full_budget else TOTAL_K_PER_SCALE / h
                )
            # Refresh the unit direction whenever the current predictable raw
            # direction is numerically available.  If it is unavailable after
            # start, reuse the most recent valid unit direction.  The start
            # itself requires validity, so every started path always has one.
            refresh = started[:, scale_index] & direction_available
            if np.any(refresh):
                last_unit_direction[refresh, scale_index] = (
                    raw_direction[refresh]
                    / np.sqrt(
                        np.sum(
                            raw_direction[refresh] * raw_direction[refresh],
                            axis=(1, 2, 3),
                            dtype=np.float64,
                        )
                    )[:, None, None, None]
                )
            if one_shot_full_budget:
                active = qualifying
            else:
                active = started[:, scale_index]
            direction_reused[:, scale_index, time_index] = (
                active & ~direction_available
            )
            shift = (
                last_unit_direction[:, scale_index]
                * np.sqrt(2.0 * frozen_allowance[:, scale_index])[:, None, None, None]
                * active[:, None, None, None]
            )
            raw = 0.5 * np.sum(
                raw_direction * raw_direction, axis=(1, 2, 3), dtype=np.float64
            )
            applied = 0.5 * np.sum(shift * shift, axis=(1, 2, 3), dtype=np.float64)
            if not np.allclose(
                applied[active],
                frozen_allowance[active, scale_index],
                rtol=2e-12,
                atol=1e-14,
            ):
                raise AssertionError("latched fixed-information step did not spend kappa")
            _, observed_increment = gaussian_log_lr(shift, innovation[:, time_index])
            raw_K[:, scale_index, time_index] = raw
            applied_K[:, scale_index, time_index] = applied
            increment[:, scale_index, time_index] = observed_increment
            cumulative[:, scale_index] += observed_increment
        component_log_e[:, :, time_index] = cumulative
        mixture_log_e[:, time_index] = _logsumexp(cumulative + log_weight, axis=1)

    for scale_index in range(scale_count):
        eligible = effective[scale_index].astype(bool)
        allowed = frozen_allowance[:, scale_index, None]
        if np.any(applied_K[:, scale_index, eligible] > allowed * (1.0 + 1e-12)):
            raise AssertionError("a checkpoint exceeded its fixed v2 KL allowance")
        if np.any(applied_K[:, scale_index, ~eligible] != 0.0):
            raise AssertionError("an identity checkpoint spent KL")
    total_K = np.sum(applied_K, axis=2)
    if np.any(total_K > TOTAL_K_PER_SCALE * (1.0 + 1e-12)):
        raise AssertionError("a scale exceeded the frozen total path KL upper bound")
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
        start_time_index=np.ascontiguousarray(start_time_index),
        start_remaining_effective_count=np.ascontiguousarray(start_remaining),
        frozen_K_per_step_after_start=np.ascontiguousarray(frozen_allowance),
        direction_reused=np.ascontiguousarray(direction_reused),
    )


def matched_q_power_reference(
    *,
    total_K: float = TOTAL_K_PER_SCALE,
    alpha_e: float = ALPHA_E,
    draws: int = MATCHED_Q_POWER_DRAWS,
    seed: int = MATCHED_Q_POWER_SEED,
) -> dict[str, Any]:
    """Conditional matched-Q power check for every allowed start depth.

    Conditional on the predictable start history, the valid start direction
    and last-valid-direction fallback make all ``h`` shifts have fixed norm.
    Terminal log E is therefore exactly ``N(K,2K)`` under its matched
    directional Q*.  We enumerate every allowed h and estimate the
    sufficient event ``E_component>=20``.  That event alone forces the 50/50
    mixture above 10, regardless of the other component or its dependence.
    """

    if not math.isfinite(total_K) or total_K <= 0.0:
        raise ValueError("total_K must be finite and positive")
    if not math.isfinite(alpha_e) or not 0.0 < alpha_e < 1.0:
        raise ValueError("alpha_e must lie in (0,1)")
    if type(draws) is not int or draws < 100_000:
        raise ValueError("matched-Q power audit requires at least 100,000 draws")
    effective = np.asarray(EFFECTIVE_NONIDENTITY, dtype=bool)
    mixture_weight = float(min(MIXTURE_WEIGHTS))
    required_component_e = 1.0 / (alpha_e * mixture_weight)
    sufficient_log_threshold = math.log(required_component_e)
    generator = np.random.default_rng(seed)
    component_results: list[dict[str, float | int]] = []
    for scale_index, heat_shift in enumerate(HEAT_SHIFTS):
        maximum_h = int(np.sum(effective[scale_index]))
        for h in range(MINIMUM_REMAINING_EFFECTIVE_AT_START, maximum_h + 1):
            kappa = total_K / h
            shift = math.sqrt(2.0 * kappa)
            noise = generator.normal(size=(draws, h)) + shift
            increment = noise * shift - kappa
            component_log = np.cumsum(increment, axis=1)
            component_results.append(
                {
                    "matched_scale_index": scale_index,
                    "matched_heat_shift": float(heat_shift),
                    "start_remaining_effective_count_h": h,
                    "fixed_K_per_step_kappa": float(kappa),
                    "anytime_sufficient_event_power": float(
                        np.mean(np.max(component_log, axis=1) >= sufficient_log_threshold)
                    ),
                    "terminal_sufficient_event_power": float(
                        np.mean(component_log[:, -1] >= sufficient_log_threshold)
                    ),
                }
            )
    z_value = (math.log(required_component_e) - total_K) / math.sqrt(2.0 * total_K)
    analytic_terminal_lower_bound = 0.5 * math.erfc(z_value / math.sqrt(2.0))
    minimum_anytime = min(
        float(row["anytime_sufficient_event_power"]) for row in component_results
    )
    return {
        "model": (
            "conditional directional matched Q* after a qualifying start; enumerate every "
            "allowed h and split K exactly as kappa=K/h with no carry"
        ),
        "draws": draws,
        "seed": seed,
        "total_K_per_component": total_K,
        "maximum_effective_checkpoint_counts": list(EFFECTIVE_STEP_COUNT_PER_SCALE),
        "allowed_start_remaining_counts_by_scale": [
            list(range(MINIMUM_REMAINING_EFFECTIVE_AT_START, count + 1))
            for count in EFFECTIVE_STEP_COUNT_PER_SCALE
        ],
        "unused_allowance_carried_forward": False,
        "alpha_e": alpha_e,
        "component_results": component_results,
        "minimum_anytime_power": minimum_anytime,
        "dependence_robust_conditional_terminal_power_lower_bound": (
            analytic_terminal_lower_bound
        ),
        "sufficient_matched_component_terminal_e_threshold": required_component_e,
        "lower_bound_reason": (
            "if E_matched>=1/(alpha*pi)=20 then the positive fixed mixture crosses 10, "
            "without assumptions on the other component or component dependence"
        ),
        "minimum_required_anytime_power": MATCHED_Q_ANYTIME_POWER_MINIMUM,
        "passes": (
            analytic_terminal_lower_bound >= MATCHED_Q_ANYTIME_POWER_MINIMUM
            and minimum_anytime >= MATCHED_Q_ANYTIME_POWER_MINIMUM
        ),
        "scope": (
            "conditional only on a real qualifying predictable start history; the start "
            "direction clears the floor by definition and later unavailable directions "
            "reuse the last valid unit direction, so no future-validity conditioning occurs"
        ),
    }


def label_free_path_mechanics_audit(
    *,
    applied_K: np.ndarray,
    start_time_index: np.ndarray,
    start_remaining_effective_count: np.ndarray,
    direction_reused: np.ndarray,
    class_id: np.ndarray,
    effective_nonidentity: np.ndarray,
) -> dict[str, Any]:
    """Audit non-degenerate multi-step operation without any quality outcome.

    The pass condition is structural: every started path must spend its frozen
    kappa at every one of the h remaining effective checkpoints.  Total K,
    max-share, and participation-ratio effective-step count are reported to
    verify the resulting identities K=2, max-share=1/h, and N_eff=h; they are
    not tuned quality targets.  Unstarted paths remain legitimate P paths but
    provide no matched-Q power guarantee.
    """

    K = np.asarray(applied_K)
    start = np.asarray(start_time_index)
    start_remaining = np.asarray(start_remaining_effective_count)
    reused = np.asarray(direction_reused)
    classes = np.asarray(class_id)
    effective = np.asarray(effective_nonidentity)
    if K.ndim != 3 or K.shape[1:] != (len(HEAT_SHIFTS), len(CHECKPOINTS)):
        raise ValueError("applied_K must be [N,scale,time]")
    if K.dtype != np.float64 or not np.isfinite(K).all() or np.any(K < 0.0):
        raise ValueError("applied_K must be finite nonnegative float64")
    if effective.dtype != np.uint8 or not np.array_equal(
        effective, np.asarray(EFFECTIVE_NONIDENTITY, dtype=np.uint8)
    ):
        raise ValueError("effective_nonidentity differs from method v2")
    if (
        start.dtype != np.int16
        or start_remaining.dtype != np.int16
        or start.shape != K.shape[:2]
        or start_remaining.shape != K.shape[:2]
    ):
        raise ValueError("start arrays must be int16 [N,scale]")
    if classes.shape != (K.shape[0],) or not np.issubdtype(classes.dtype, np.integer):
        raise ValueError("class_id must be integer [N]")
    if reused.dtype != np.bool_ or reused.shape != K.shape:
        raise ValueError("direction_reused must be bool [N,scale,time]")
    if np.any(reused & (K == 0.0)):
        raise ValueError("a reused-direction marker has zero applied KL")
    if np.any(K[:, effective == 0] != 0.0):
        raise ValueError("identity checkpoints may not spend positive KL")

    scale_rows: list[dict[str, Any]] = []
    for scale_index, heat_shift in enumerate(HEAT_SHIFTS):
        eligible = effective[scale_index].astype(bool)
        values = K[:, scale_index, eligible]
        positive_count = np.sum(values > 0.0, axis=1)
        total = np.sum(values, axis=1)
        started = start[:, scale_index] >= 0
        if np.any(started & (start_remaining[:, scale_index] < MINIMUM_REMAINING_EFFECTIVE_AT_START)):
            raise ValueError("a primary path started with fewer than three effective checkpoints")
        if np.any((~started) & (start_remaining[:, scale_index] != 0)):
            raise ValueError("an unstarted path has nonzero start metadata")
        positive_path = total > 0.0
        started_count = int(np.sum(started))
        started_class_count = int(np.unique(classes[started]).size)
        complete = np.zeros(K.shape[0], dtype=np.bool_)
        eligible_indices = np.flatnonzero(eligible)
        for row_index in np.flatnonzero(started):
            start_index = int(start[row_index, scale_index])
            if start_index not in eligible_indices:
                raise ValueError("start_time_index is not an effective checkpoint")
            remaining_indices = eligible_indices[eligible_indices >= start_index]
            h = int(start_remaining[row_index, scale_index])
            if len(remaining_indices) != h:
                raise ValueError("start h differs from the actual remaining effective count")
            if np.any(K[row_index, scale_index, :start_index] != 0.0):
                raise ValueError("a path spent KL before its predictable start")
            kappa = TOTAL_K_PER_SCALE / h
            complete[row_index] = np.allclose(
                K[row_index, scale_index, remaining_indices],
                kappa,
                rtol=2e-12,
                atol=1e-14,
            )
        maximum_share = np.divide(
            np.max(values, axis=1),
            total,
            out=np.zeros_like(total),
            where=positive_path,
        )
        squared_total = np.sum(values * values, axis=1)
        effective_step_count = np.divide(
            total * total,
            squared_total,
            out=np.zeros_like(total),
            where=squared_total > 0.0,
        )
        complete_fraction = float(np.mean(complete[started])) if started_count else 0.0
        reused_steps = int(np.sum(reused[:, scale_index, eligible]))
        started_steps = int(np.sum(positive_count[started]))
        reused_fraction = reused_steps / started_steps if started_steps else 0.0
        reused_path_count = int(np.sum(np.any(reused[:, scale_index], axis=1)))
        maximum_consecutive_reused = 0
        for row_index in np.flatnonzero(started):
            run = 0
            for flag in reused[row_index, scale_index].tolist():
                run = run + 1 if flag else 0
                maximum_consecutive_reused = max(maximum_consecutive_reused, run)
        histogram = {
            str(count): int(np.sum(positive_count == count))
            for count in range(int(np.sum(eligible)) + 1)
        }
        if started_count:
            share_quantiles: dict[str, float | None] = {
                "q25": float(np.quantile(maximum_share[started], 0.25)),
                "median": float(np.quantile(maximum_share[started], 0.50)),
                "q75": float(np.quantile(maximum_share[started], 0.75)),
            }
            effective_quantiles: dict[str, float | None] = {
                "q25": float(np.quantile(effective_step_count[started], 0.25)),
                "median": float(np.quantile(effective_step_count[started], 0.50)),
                "q75": float(np.quantile(effective_step_count[started], 0.75)),
            }
        else:
            share_quantiles = {"q25": None, "median": None, "q75": None}
            effective_quantiles = {"q25": None, "median": None, "q75": None}
        passes = (
            started_count >= PATH_MECHANICS_MINIMUM_STARTED_PATHS_PER_SCALE
            and started_class_count >= PATH_MECHANICS_MINIMUM_STARTED_CLASSES_PER_SCALE
            and complete_fraction
            >= PATH_MECHANICS_COMPLETE_COVERAGE_FRACTION_MINIMUM
            and reused_fraction <= PATH_MECHANICS_MAX_REUSED_DIRECTION_FRACTION
        )
        scale_rows.append(
            {
                "scale_index": scale_index,
                "heat_shift": float(heat_shift),
                "effective_checkpoint_count": int(np.sum(eligible)),
                "earliest_start_fixed_K_allowance": float(
                    FIXED_K_PER_EFFECTIVE_CHECKPOINT[scale_index]
                ),
                "qualifying_started_path_count": started_count,
                "qualifying_started_path_fraction": float(np.mean(started)) if len(started) else 0.0,
                "qualifying_started_class_count": started_class_count,
                "positive_K_path_count": int(np.sum(positive_path)),
                "positive_K_step_count_histogram_all_paths": histogram,
                "reused_direction_step_count": reused_steps,
                "reused_direction_path_count": reused_path_count,
                "maximum_consecutive_reused_direction_steps": maximum_consecutive_reused,
                "reused_direction_fraction_among_started_steps": (
                    float(reused_fraction)
                ),
                "fraction_started_paths_with_positive_K_step_count_equal_to_start_h": (
                    float(
                        np.mean(
                            positive_count[started]
                            == start_remaining[started, scale_index]
                        )
                    )
                    if started_count
                    else 0.0
                ),
                "fraction_started_paths_with_exact_complete_fixed_information_coverage": (
                    complete_fraction
                ),
                "KL_max_share_quantiles_among_started_paths": share_quantiles,
                "KL_effective_step_count_quantiles_among_started_paths": effective_quantiles,
                "total_K_quantiles_mechanics_only": (
                    {
                        "q25": float(np.quantile(total[started], 0.25)),
                        "median": float(np.quantile(total[started], 0.50)),
                        "q75": float(np.quantile(total[started], 0.75)),
                    }
                    if started_count
                    else {"q25": None, "median": None, "q75": None}
                ),
                "passes_non_degenerate_multi_step_conditions": passes,
            }
        )
    enough_samples = K.shape[0] >= PATH_MECHANICS_MINIMUM_SAMPLES
    passes = enough_samples and all(
        row["passes_non_degenerate_multi_step_conditions"] for row in scale_rows
    )
    return {
        "status": "PASS" if passes else (
            "INSUFFICIENT_SAMPLE_COUNT" if not enough_samples else "FAIL_STOP_BEFORE_LABELS"
        ),
        "sample_count": int(K.shape[0]),
        "minimum_sample_count": PATH_MECHANICS_MINIMUM_SAMPLES,
        "minimum_qualifying_started_paths_per_scale": (
            PATH_MECHANICS_MINIMUM_STARTED_PATHS_PER_SCALE
        ),
        "minimum_qualifying_started_classes_per_scale": (
            PATH_MECHANICS_MINIMUM_STARTED_CLASSES_PER_SCALE
        ),
        "minimum_fraction_started_paths_with_exact_complete_fixed_information_coverage": (
            PATH_MECHANICS_COMPLETE_COVERAGE_FRACTION_MINIMUM
        ),
        "maximum_reused_direction_fraction_among_started_steps": (
            PATH_MECHANICS_MAX_REUSED_DIRECTION_FRACTION
        ),
        "scale_results": scale_rows,
        "passes": passes,
        "quality_or_power_interpretation": False,
        "labels_endpoint_images_external_representations_used": False,
        "failure_action": (
            "STOP E before label join; do not tune K, alpha, B gate, masks, shifts, "
            "weights, or mechanics thresholds on this pool"
        ),
    }


def gate_only_start_schedule_score(tracks: EProcessTracks) -> np.ndarray:
    """Return the frozen innovation-free start-schedule diagnostic G_start.

    Earlier starts have larger remaining fractions.  This score contains the
    B/direction-availability start decision but none of the subsequently
    observed Gaussian innovations, so it exposes whether E merely inherits a
    useful start schedule rather than gaining from innovation alignment.
    """

    start = np.asarray(tracks.start_time_index)
    remaining = np.asarray(tracks.start_remaining_effective_count)
    if (
        start.ndim != 2
        or start.shape[1] != len(HEAT_SHIFTS)
        or remaining.shape != start.shape
        or start.dtype != np.int16
        or remaining.dtype != np.int16
    ):
        raise ValueError("EProcessTracks start metadata is malformed")
    active = start >= 0
    normalized = (
        remaining.astype(np.float64)
        / np.asarray(EFFECTIVE_STEP_COUNT_PER_SCALE, dtype=np.float64)[None, :]
    ) * active
    score = normalized @ np.asarray(MIXTURE_WEIGHTS, dtype=np.float64)
    return np.ascontiguousarray(score, dtype=np.float64)


def _validate_input_keys(keys: Iterable[str]) -> None:
    v1._validate_input_keys(keys)


def validate_observer_input(arrays: Mapping[str, np.ndarray]) -> None:
    # Input tensors and the high-noise construction are unchanged from v1.
    v1.validate_observer_input(arrays)


def extract_label_free(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    validate_observer_input(arrays)
    blur = compute_blur_tiles(arrays["decoded_pred_xstart_rgb"])
    B_persistence, formation_slope = blur_summary_tracks(blur.severity)
    common = dict(
        theta=arrays["theta"],
        p_standard_deviation=arrays["p_standard_deviation"],
        transition_innovation=arrays["transition_innovation"],
        local_mask=blur.latent_mask,
        blur_severity=blur.severity,
        blur_gate_threshold=arrays["blur_gate_threshold"],
        effective_nonidentity=arrays["effective_nonidentity"],
    )
    gated = compute_eprocess_tracks(**common, use_state_gate=True)
    ungated = compute_eprocess_tracks(**common, use_state_gate=False)
    one_shot = compute_eprocess_tracks(
        **common, use_state_gate=True, one_shot_full_budget=True
    )
    G_start = gate_only_start_schedule_score(gated)
    return {
        "blur": blur,
        "B_persistence": np.ascontiguousarray(B_persistence),
        "B_formation_slope_diagnostic": np.ascontiguousarray(formation_slope),
        "B_alarm": np.ascontiguousarray(B_persistence > arrays["blur_score_threshold"]),
        "gated": gated,
        "ungated": ungated,
        "one_shot": one_shot,
        "G_start_schedule_diagnostic": G_start,
    }


def _protocol_snapshot() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "status": "LABEL_FREE_OBSERVATION_ONLY_NOT_VALIDATED_FOR_INTERVENTION",
        "checkpoints": list(CHECKPOINTS),
        "heat_shifts": list(HEAT_SHIFTS),
        "shifted_internal_timesteps": [list(row) for row in SHIFTED_INTERNAL_TIMESTEPS],
        "mixture_weights": list(MIXTURE_WEIGHTS),
        "total_K_per_scale_component_upper_bound": TOTAL_K_PER_SCALE,
        "effective_checkpoint_counts": list(EFFECTIVE_STEP_COUNT_PER_SCALE),
        "earliest_start_fixed_K_per_effective_checkpoint": list(
            FIXED_K_PER_EFFECTIVE_CHECKPOINT
        ),
        "K_spending": (
            "first qualifying B crossing with h>=3 remaining effective checkpoints "
            "freezes kappa=K_total/h; the alternative latches on, every remaining "
            "effective checkpoint gets exactly kappa, and kappa is never reallocated "
            "across checkpoints"
        ),
        "directional_Q_star": (
            "start only when ||mask*sigma*theta||>=1e-8; thereafter normalize each "
            "available predictable direction to ||u||=sqrt(2*kappa), reusing the last "
            "valid unit direction when the current norm is below floor; this is a new "
            "fixed-information operational Q*, not the ideal score-gap magnitude"
        ),
        "alpha_e": ALPHA_E,
        "real_label_free_gate": (
            "on all 768 confirmation traces before labels: qualifying-start coverage, "
            "positive-K step count, KL max-share, and KL effective-step count only"
        ),
        "threshold_lineage_requirement": (
            "B thresholds must be frozen from disjoint calibration seeds before any of the "
            "768 confirmation paths; replay on threshold-fitting paths is in-sample "
            "diagnostic only and cannot support e-process exactness"
        ),
        "candidate_roles": {
            "B_persistence": "predictable heuristic anchor; not an e-process",
            "E_blur_gated": "primary exact operational P/Q* e-process candidate",
            "E_no_state_gate": "exact no-B-start mechanism ablation only",
            "E_first_hit_full_budget": "distinct exact one-shot ablation only",
            "G_start_schedule": "fixed innovation-free start/h schedule diagnostic",
            "B_formation_slope": "diagnostic-only predictable statistic",
        },
        "external_representation_policy": (
            "endpoint images, labels, reviews, FID, Inception, DINO, CLIP, quality "
            "posteriors, and representation distances are forbidden method inputs"
        ),
        "exactness_scope": (
            "operational same-covariance Q*/P likelihood ratio with predictable gate, mask, "
            "direction, last-valid fallback, and fixed per-checkpoint information; "
            "not an ideal heat marginal ratio"
        ),
    }


def publish(input_path: Path, output_dir: Path) -> Path:
    if output_dir.exists() or output_dir.is_symlink():
        raise RuntimeError(f"refusing pre-existing output path: {output_dir}")
    if not input_path.is_file() or input_path.is_symlink():
        raise RuntimeError("input must be one regular, non-symlink NPZ")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        with np.load(input_path, allow_pickle=False) as archive:
            _validate_input_keys(archive.files)
            arrays = {name: np.ascontiguousarray(archive[name]) for name in INPUT_ARRAY_NAMES}
        result = extract_label_free(arrays)
        blur: BlurTiles = result["blur"]
        gated: EProcessTracks = result["gated"]
        ungated: EProcessTracks = result["ungated"]
        one_shot: EProcessTracks = result["one_shot"]
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
            "gated_start_time_index": gated.start_time_index,
            "gated_start_remaining_effective_count": gated.start_remaining_effective_count,
            "gated_frozen_K_per_step_after_start": gated.frozen_K_per_step_after_start,
            "gated_direction_reused": gated.direction_reused,
            "ungated_raw_K": ungated.raw_K,
            "ungated_applied_K": ungated.applied_K,
            "ungated_component_increment": ungated.component_increment,
            "ungated_component_log_e": ungated.component_log_e,
            "ungated_mixture_log_e": ungated.mixture_log_e,
            "one_shot_raw_K": one_shot.raw_K,
            "one_shot_applied_K": one_shot.applied_K,
            "one_shot_component_increment": one_shot.component_increment,
            "one_shot_component_log_e": one_shot.component_log_e,
            "one_shot_mixture_log_e": one_shot.mixture_log_e,
        }
        tracks_path = staging / "tracks.npz"
        with tracks_path.open("wb") as handle:
            np.savez(handle, **tracks)

        feature_rows: list[dict[str, Any]] = []
        for row in range(len(result["B_persistence"])):
            K = gated.applied_K[row]
            total = np.sum(K, axis=1)
            maximum = np.max(K, axis=1)
            shares = np.divide(maximum, total, out=np.zeros_like(total), where=total > 0.0)
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
                    "E_first_hit_full_budget_running_max_log_ablation": float(
                        np.max(one_shot.running_max_log_e[row], initial=0.0)
                    ),
                    "E_first_hit_full_budget_alarm_ablation": int(one_shot.alarm[row]),
                    "G_start_schedule_diagnostic": float(
                        result["G_start_schedule_diagnostic"][row]
                    ),
                    "blur_gate_open_count": int(
                        np.sum(blur.severity[row] > arrays["blur_gate_threshold"][row])
                    ),
                    "gated_positive_K_steps_scale_0": int(np.sum(K[0] > 0.0)),
                    "gated_positive_K_steps_scale_1": int(np.sum(K[1] > 0.0)),
                    "gated_KL_max_share_scale_0": float(shares[0]),
                    "gated_KL_max_share_scale_1": float(shares[1]),
                    "gated_reused_direction_steps_scale_0": int(
                        np.sum(gated.direction_reused[row, 0])
                    ),
                    "gated_reused_direction_steps_scale_1": int(
                        np.sum(gated.direction_reused[row, 1])
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
                {"name": "B_persistence", "role": "co-primary heuristic", "direction": "bad_high"},
                {
                    "name": "E_blur_gated_running_max_log",
                    "role": "co-primary exact operational e-process report",
                    "direction": "bad_high",
                },
                {
                    "name": "E_no_state_gate_running_max_log_ablation",
                    "role": "fixed exact ablation only",
                    "direction": "bad_high",
                },
                {
                    "name": "E_first_hit_full_budget_running_max_log_ablation",
                    "role": "distinct exact predictable one-shot full-budget ablation only",
                    "direction": "bad_high",
                },
                {
                    "name": "G_start_schedule_diagnostic",
                    "role": "innovation-free start/h schedule diagnostic only",
                    "direction": "bad_high",
                },
            ],
            "labels_read_or_emitted": False,
            "endpoint_images_read": False,
            "external_representations_read": False,
        }
        _atomic_json_dump(_protocol_snapshot(), staging / "protocol_snapshot.json")
        _atomic_json_dump(catalog, staging / "feature_catalog.json")
        mechanics = label_free_path_mechanics_audit(
            applied_K=gated.applied_K,
            start_time_index=gated.start_time_index,
            start_remaining_effective_count=gated.start_remaining_effective_count,
            direction_reused=gated.direction_reused,
            class_id=arrays["class_id"],
            effective_nonidentity=arrays["effective_nonidentity"],
        )
        _atomic_json_dump(mechanics, staging / "label_free_path_mechanics_audit.json")
        outputs = [
            {"relative_path": path.name, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}
            for path in sorted(staging.iterdir(), key=lambda item: item.name)
            if path.name != "manifest.json"
        ]
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
                "immutable_v1_dependency_sha256": _sha256_file(Path(v1.__file__).resolve()),
            },
            "sample_count": len(feature_rows),
            "supervision": "none",
            "label_free_path_mechanics_status": mechanics["status"],
            "label_free_path_mechanics_passed": mechanics["passes"],
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


def adaptive_predictable_null_reference(
    *, draws: int = 250_000, seed: int = 2026082821
) -> dict[str, Any]:
    """Monte Carlo witness for adaptive predictable directions under P.

    Directions rotate as a deterministic function of past shared innovations,
    so the two components are adaptive and dependent.  They never use the
    current innovation.  This checks terminal mean-one calibration of each
    component/fixed mixture and a finite-sample Ville witness.
    """

    if type(draws) is not int or draws < 100_000:
        raise ValueError("adaptive null reference needs at least 100,000 draws")
    rng = np.random.default_rng(seed)
    time_count = len(CHECKPOINTS)
    shared_z = rng.normal(size=(draws, time_count, 2))
    component_log = np.zeros((draws, len(HEAT_SHIFTS), time_count), dtype=np.float64)
    cumulative_log = np.zeros((draws, len(HEAT_SHIFTS)), dtype=np.float64)
    past = np.zeros((draws, 2), dtype=np.float64)
    effective = np.asarray(EFFECTIVE_NONIDENTITY, dtype=bool)
    for time_index in range(time_count):
        for scale_index in range(len(HEAT_SHIFTS)):
            if not effective[scale_index, time_index]:
                continue
            kappa = FIXED_K_PER_EFFECTIVE_CHECKPOINT[scale_index]
            phi = 0.41 * past[:, 0] - 0.23 * past[:, 1] + 0.7 * scale_index
            norm = math.sqrt(2.0 * kappa)
            u0 = norm * np.cos(phi)
            u1 = norm * np.sin(phi)
            increment = (
                u0 * shared_z[:, time_index, 0]
                + u1 * shared_z[:, time_index, 1]
                - kappa
            )
            cumulative_log[:, scale_index] += increment
        component_log[:, :, time_index] = cumulative_log
        past += shared_z[:, time_index]
    component_terminal_mean = np.mean(np.exp(component_log[:, :, -1]), axis=0)
    log_weight = np.log(np.asarray(MIXTURE_WEIGHTS, dtype=np.float64))[None, :, None]
    mixture_log = _logsumexp(component_log + log_weight, axis=1)
    mixture_terminal_mean = float(np.mean(np.exp(mixture_log[:, -1])))
    anytime_trigger_fraction = float(
        np.mean(np.max(mixture_log, axis=1) >= math.log(1.0 / ALPHA_E))
    )
    passes = (
        np.all(np.abs(component_terminal_mean - 1.0) < 0.04)
        and abs(mixture_terminal_mean - 1.0) < 0.03
        and anytime_trigger_fraction <= ALPHA_E + 0.005
    )
    return {
        "model": "two dependent components with directions adapted only to past shared innovations",
        "draws": draws,
        "seed": seed,
        "component_terminal_e_means": [float(value) for value in component_terminal_mean],
        "fixed_mixture_terminal_e_mean": mixture_terminal_mean,
        "anytime_threshold": 1.0 / ALPHA_E,
        "anytime_trigger_fraction_under_P": anytime_trigger_fraction,
        "ville_upper_bound": ALPHA_E,
        "finite_monte_carlo_tolerance": 0.005,
        "passes": bool(passes),
    }


def self_test() -> None:
    # Retain every v1 primitive/external-boundary check.
    v1.self_test()
    rng = np.random.default_rng(2026082817)
    batch = PATH_MECHANICS_MINIMUM_SAMPLES
    theta = rng.normal(
        size=(batch, len(HEAT_SHIFTS), len(CHECKPOINTS), 4, 8, 8)
    ).astype(np.float64)
    effective = np.asarray(EFFECTIVE_NONIDENTITY, dtype=np.uint8)
    theta[:, effective == 0] = 0.0
    sigma = np.full((batch, len(CHECKPOINTS), 4, 8, 8), 0.5, dtype=np.float32)
    innovation = rng.normal(size=sigma.shape).astype(np.float32)
    mask = np.ones((batch, len(CHECKPOINTS), 1, 8, 8), dtype=np.float64)
    blur = np.ones((batch, len(CHECKPOINTS)), dtype=np.float64)
    threshold = np.zeros_like(blur)
    tracks = compute_eprocess_tracks(
        theta=np.ascontiguousarray(theta),
        p_standard_deviation=sigma,
        transition_innovation=innovation,
        local_mask=mask,
        blur_severity=blur,
        blur_gate_threshold=threshold,
        effective_nonidentity=effective,
        use_state_gate=True,
    )
    poison_tracks = compute_eprocess_tracks(
        theta=np.ascontiguousarray(theta),
        p_standard_deviation=sigma,
        transition_innovation=rng.normal(size=sigma.shape).astype(np.float32),
        local_mask=mask,
        blur_severity=blur,
        blur_gate_threshold=threshold,
        effective_nonidentity=effective,
        use_state_gate=True,
    )
    for name in (
        "raw_K",
        "applied_K",
        "start_time_index",
        "start_remaining_effective_count",
        "frozen_K_per_step_after_start",
        "direction_reused",
    ):
        if not np.array_equal(getattr(tracks, name), getattr(poison_tracks, name)):
            raise AssertionError(f"innovation poison changed predictable track {name}")
    if np.array_equal(tracks.component_increment, poison_tracks.component_increment):
        raise AssertionError("innovation poison did not change observed LR increments")
    if not np.array_equal(
        gate_only_start_schedule_score(tracks),
        gate_only_start_schedule_score(poison_tracks),
    ):
        raise AssertionError("innovation poison changed G_start schedule diagnostic")
    for scale_index, allowance in enumerate(FIXED_K_PER_EFFECTIVE_CHECKPOINT):
        eligible = effective[scale_index].astype(bool)
        if np.any(tracks.applied_K[:, scale_index, eligible] > allowance * (1 + 1e-12)):
            raise AssertionError("fixed per-checkpoint allowance failed")

    # The latest permitted start has h=3 and must spend kappa=2/3 at exactly
    # those three checkpoints, not the earliest-start 0.4/0.25 values.
    late_blur = np.full((1, len(CHECKPOINTS)), -1.0, dtype=np.float64)
    late_blur[0, 6:] = 1.0
    late_theta = np.ones(
        (1, len(HEAT_SHIFTS), len(CHECKPOINTS), 4, 8, 8), dtype=np.float64
    )
    late_theta[:, effective == 0] = 0.0
    late = compute_eprocess_tracks(
        theta=late_theta,
        p_standard_deviation=np.ones((1, len(CHECKPOINTS), 4, 8, 8), dtype=np.float32),
        transition_innovation=np.zeros((1, len(CHECKPOINTS), 4, 8, 8), dtype=np.float32),
        local_mask=np.ones((1, len(CHECKPOINTS), 1, 8, 8), dtype=np.float64),
        blur_severity=late_blur,
        blur_gate_threshold=np.zeros_like(late_blur),
        effective_nonidentity=effective,
        use_state_gate=True,
    )
    if not np.array_equal(late.start_remaining_effective_count, np.asarray([[3, 3]], dtype=np.int16)):
        raise AssertionError("late h=3 start was not frozen")
    if not np.allclose(late.frozen_K_per_step_after_start, 2.0 / 3.0):
        raise AssertionError("late h=3 kappa differs from 2/3")
    if not np.all(np.sum(late.applied_K > 0.0, axis=2) == 3):
        raise AssertionError("late-start path did not cover exactly three steps")

    # Direct no-carry witness: after a valid start, an unavailable middle
    # direction reuses the last unit direction at the same kappa; it cannot
    # enlarge a later step.
    synthetic = np.zeros((1, len(HEAT_SHIFTS), len(CHECKPOINTS), 4, 8, 8), dtype=np.float64)
    first_three = np.flatnonzero(effective[0])[:3]
    synthetic[0, 0, first_three[0], 0, 0, 0] = 0.2
    synthetic[0, 0, first_three[2]] = 100.0
    no_carry = compute_eprocess_tracks(
        theta=synthetic,
        p_standard_deviation=np.ones((1, len(CHECKPOINTS), 4, 8, 8), dtype=np.float32),
        transition_innovation=np.zeros((1, len(CHECKPOINTS), 4, 8, 8), dtype=np.float32),
        local_mask=np.ones((1, len(CHECKPOINTS), 1, 8, 8), dtype=np.float64),
        blur_severity=np.ones((1, len(CHECKPOINTS)), dtype=np.float64),
        blur_gate_threshold=np.zeros((1, len(CHECKPOINTS)), dtype=np.float64),
        effective_nonidentity=effective,
        use_state_gate=True,
    )
    if not np.isclose(
        no_carry.applied_K[0, 0, first_three[0]],
        FIXED_K_PER_EFFECTIVE_CHECKPOINT[0],
        rtol=0.0,
        atol=1e-12,
    ):
        raise AssertionError("valid start did not spend fixed information")
    if not np.isclose(
        no_carry.applied_K[0, 0, first_three[1]],
        FIXED_K_PER_EFFECTIVE_CHECKPOINT[0],
        rtol=0.0,
        atol=1e-12,
    ) or not no_carry.direction_reused[0, 0, first_three[1]]:
        raise AssertionError("unavailable direction did not reuse last unit at fixed kappa")
    if not np.isclose(
        no_carry.applied_K[0, 0, first_three[2]],
        FIXED_K_PER_EFFECTIVE_CHECKPOINT[0],
        rtol=0.0,
        atol=1e-12,
    ):
        raise AssertionError("middle fallback changed the later fixed kappa")

    fixture_K = np.zeros((batch, len(HEAT_SHIFTS), len(CHECKPOINTS)), dtype=np.float64)
    for scale_index, row in enumerate(effective.astype(bool)):
        indices = np.flatnonzero(row)
        fixture_K[:, scale_index, indices] = FIXED_K_PER_EFFECTIVE_CHECKPOINT[scale_index]
    fixture_start = np.stack(
        [
            np.full(batch, int(np.flatnonzero(effective[index])[0]), dtype=np.int16)
            for index in range(len(HEAT_SHIFTS))
        ],
        axis=1,
    )
    fixture_remaining = np.repeat(
        np.asarray(EFFECTIVE_STEP_COUNT_PER_SCALE, dtype=np.int16)[None, :],
        batch,
        axis=0,
    )
    fixture_classes = np.arange(batch, dtype=np.int16) % 3
    mechanics = label_free_path_mechanics_audit(
        applied_K=fixture_K,
        start_time_index=fixture_start,
        start_remaining_effective_count=fixture_remaining,
        direction_reused=np.zeros_like(fixture_K, dtype=np.bool_),
        class_id=fixture_classes,
        effective_nonidentity=effective,
    )
    if mechanics["passes"] is not True:
        raise AssertionError(f"multi-step mechanics fixture failed: {mechanics}")
    concentrated = np.zeros_like(fixture_K)
    for scale_index, row in enumerate(effective.astype(bool)):
        concentrated[:, scale_index, int(np.flatnonzero(row)[0])] = 0.1
    failed = label_free_path_mechanics_audit(
        applied_K=concentrated,
        start_time_index=fixture_start,
        start_remaining_effective_count=fixture_remaining,
        direction_reused=np.zeros_like(concentrated, dtype=np.bool_),
        class_id=fixture_classes,
        effective_nonidentity=effective,
    )
    if failed["passes"] is not False:
        raise AssertionError("single-step mechanics fixture was not rejected")
    stale_reuse = np.zeros_like(fixture_K, dtype=np.bool_)
    for scale_index, row in enumerate(effective.astype(bool)):
        stale_reuse[:, scale_index, int(np.flatnonzero(row)[1])] = True
    stale = label_free_path_mechanics_audit(
        applied_K=fixture_K,
        start_time_index=fixture_start,
        start_remaining_effective_count=fixture_remaining,
        direction_reused=stale_reuse,
        class_id=fixture_classes,
        effective_nonidentity=effective,
    )
    if stale["passes"] is not False:
        raise AssertionError("stale-direction fraction fixture was not rejected")
    one_shot = compute_eprocess_tracks(
        theta=np.ascontiguousarray(theta),
        p_standard_deviation=sigma,
        transition_innovation=innovation,
        local_mask=mask,
        blur_severity=blur,
        blur_gate_threshold=threshold,
        effective_nonidentity=effective,
        use_state_gate=True,
        one_shot_full_budget=True,
    )
    if np.any(np.sum(one_shot.applied_K > 0.0, axis=2) != 1) or not np.allclose(
        np.sum(one_shot.applied_K, axis=2), TOTAL_K_PER_SCALE
    ):
        raise AssertionError("one-shot full-budget ablation is not exactly one-step K=2")
    power = matched_q_power_reference()
    if power["passes"] is not True:
        raise AssertionError(f"distributed matched-Q gate failed: {power}")
    adaptive_null = adaptive_predictable_null_reference()
    if adaptive_null["passes"] is not True:
        raise AssertionError(f"adaptive predictable null witness failed: {adaptive_null}")
    print(
        "v2 self-test passed: immutable v1 primitives, predictable latch/late h=3, "
        "innovation-poison invariance, no carry, one-shot ablation, complete-path "
        "mechanics pass/single-step and stale-direction rejection, and all-h conditional "
        f"matched-Q anytime power {power['minimum_anytime_power']:.6f}; adaptive null "
        f"mixture mean {adaptive_null['fixed_mixture_terminal_e_mean']:.6f}, Ville "
        f"trigger {adaptive_null['anytime_trigger_fraction_under_P']:.6f}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.self_test:
        if args.input is not None or args.output_dir is not None:
            raise RuntimeError("--self-test cannot be combined with real inputs")
        self_test()
        return 0
    if args.input is None or args.output_dir is None:
        raise RuntimeError("real extraction requires --input and --output-dir")
    result = publish(args.input.expanduser().resolve(), args.output_dir.expanduser().absolute())
    print(f"published method-v2 label-free e-process features: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
