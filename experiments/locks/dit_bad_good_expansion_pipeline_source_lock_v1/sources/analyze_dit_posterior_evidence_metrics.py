#!/usr/bin/env python3
"""Extract label-free posterior and weak-process evidence from custom DiT traces.

This is a supplementary, observation-only analyzer for the completed outputs of
``trace_dit_imagenet256_custom_batch.py``.  It deliberately does not read or
join labels, compute AUCs, select thresholds, fit combinations, or authorize an
intervention.  The existing trace validator is imported and reused so every
input archive, source snapshot, manifest, completion receipt, transition
replay, CFG identity, and decoded endpoint is checked before extraction.

For sampling row ``k``, let ``x`` be ``state_before``, ``mu`` the saved guided
posterior mean, ``sigma`` the saved conditional learned-range standard
deviation, and ``z`` the saved standard-normal transition innovation.  Rows
``k=0..248`` are the stochastic transitions ``t=249..1``.  The recorded random
draw at ``k=249, t=0`` is masked by the sampler and is excluded from every
realized-transition and likelihood-ratio quantity.

Two predeclared weak alternatives are evaluated against the actual guided P:

* conditional CFG=1 Q, using the raw conditional epsilon branch mean; and
* unconditional Q, using the raw unconditional epsilon branch mean.

Both Q processes change the posterior mean only and deliberately retain P's
actual saved diagonal covariance.  With ``delta = mu_Q - mu_P`` and
``w = delta / sigma``, the exact one-step same-covariance log likelihood ratio
under the realized P draw is ``w dot z - 0.5 * ||w||^2``.  Sixteen additional
fixed alternatives restrict the same predictable mean shift to one 8x8 cell
of a predeclared 4x4 grid.  Their cumulative exact likelihood ratios are mixed
with a uniform 1/16 prior using stable log-mean-exp; the componentwise maximum
is never treated as evidence.

Output directories are immutable: any existing path is refused.  A staged
publication is atomically renamed only after all payloads, formulas, source
bindings, array inventories, and hashes have been written.  The final output
contains no labels or supervised analysis products.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Set both controls before importing the existing analyzer.  In particular,
# validation must not create __pycache__ beside the frozen helper source.
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_dit_bad_good_custom_traces import (  # noqa: E402
    CHANNELS,
    EPS,
    LATENT_SIZE,
    STEPS,
    TrackSpec,
    TraceRecord,
    _add_track,
    _combine_trace_tracks,
    _cosine,
    _parse_csv_ints,
    _require_regular,
    _rms,
    _safe_relative,
    _tile_concentration,
    atomic_json_dump,
    canonical_sha256,
    discover_trace_dirs,
    load_json,
    load_validated_trace,
    reduce_tracks,
    sha256_array,
    sha256_file,
)


ANALYSIS_SCHEMA_VERSION = 1
EXPERIMENT = "dit_targeted_posterior_evidence_label_free"
GRID_SIZE = 4
TILE_COUNT = GRID_SIZE * GRID_SIZE
TILE_SIZE = LATENT_SIZE // GRID_SIZE
STOCHASTIC_STEPS = STEPS - 1
LOGSTD_RECONSTRUCTION_TOLERANCE = 1e-5
MEAN_RECONSTRUCTION_TOLERANCE = 2e-6
TRANSITION_RECONSTRUCTION_TOLERANCE = 2e-6
DENOISING_IDENTITY_TOLERANCE = 2e-6
GROUPS: dict[str, slice | tuple[int, ...]] = {
    "guided3": slice(0, 3),
    "channel4": (3,),
    "all4": slice(0, 4),
}
Q_NAMES = ("weak_conditional_cfg1", "weak_unconditional")
IDENTIFIER_COLUMNS = (
    "sample_index",
    "run_index",
    "global_seed",
    "class_slot",
    "class_id",
    "trace_dir",
    "endpoint_png_path",
)


PROTOCOL: dict[str, Any] = {
    "schema_version": ANALYSIS_SCHEMA_VERSION,
    "experiment": EXPERIMENT,
    "status": "LABEL_FREE_SUPPLEMENTARY_OBSERVATION_ONLY",
    "input_validation": (
        "imports load_validated_trace from analyze_dit_bad_good_custom_traces.py; "
        "validates trace manifests, completion receipts, bound source snapshots, all "
        "archive member hashes/shapes/dtypes/finiteness, exact float32 transition replay, "
        "three-channel CFG reconstruction, and decoded endpoint PNG equality"
    ),
    "sampling_axis": {
        "sampling_step": "k=0..249",
        "internal_timestep": "t=249-k",
        "predictable": "available after the current model call and before its draw",
        "post_transition": "available only after the current stochastic transition",
        "stochastic_transition_rows": "k=0..248, internal t=249..1",
        "masked_t0": (
            "the saved k=249, internal-t=0 random draw is multiplied by zero upstream "
            "and is excluded from realized updates, sigma-standardized operational "
            "metrics, D, LR increments, cumulative log-E, and tile-mixture evidence"
        ),
    },
    "channel_groups": {
        "guided3": "epsilon/latent channels 0,1,2; the released DiT applies CFG here",
        "channel4": "epsilon/latent channel 3; actual P retains the conditional branch",
        "all4": "all four latent channels",
    },
    "reverse_mean_drift": {
        "raw": "d_k = mu_P,k - x_k",
        "relative_rms": "RMS_g(d_k) / (RMS_g(x_k) + 1e-12)",
        "relative_field": (
            "d_k / (RMS_g(x_k)+1e-12); its tile concentration is exactly invariant "
            "to this positive per-row scalar normalization"
        ),
        "sigma_standardized": "d_k / sigma_P,k on stochastic rows only",
        "tile_concentration": (
            "max of sixteen equal-area tile mean-squared energies divided by their sum"
        ),
        "realized_update": "u_k = x_{k+1} - x_k = d_k + sigma_P,k * z_k",
        "drift_noise_alignment": "cosine(d_k, sigma_P,k*z_k)",
        "stochastic_to_drift_ratio": (
            "RMS_g(sigma_P,k*z_k) / (RMS_g(d_k)+1e-12)"
        ),
    },
    "guided_fields": {
        "epsilon": (
            "eps_P[0:3]=eps_uncond+cfg_scale*(eps_cond-eps_uncond); "
            "eps_P[3]=eps_cond[3]"
        ),
        "score": "s_k = -eps_P,k / sqrt(1-alpha_bar_k)",
        "denoising_displacement": (
            "r_k = (1-alpha_bar_k)*s_k = -sqrt(1-alpha_bar_k)*eps_P,k "
            "= sqrt(alpha_bar_k)*pred_x0_P,k - x_k up to recorded float32 rounding"
        ),
        "reductions": "group RMS and fixed 4x4 spatial tile concentration",
    },
    "learned_range_transform": {
        "schedule_derivation": (
            "alpha_t=alpha_bar_t/alpha_bar_{t-1}, beta_t=1-alpha_t, "
            "alpha_bar_{-1}=1"
        ),
        "posterior_variance": (
            "beta_t*(1-alpha_bar_{t-1})/(1-alpha_bar_t)"
        ),
        "posterior_log_variance_clipped": (
            "log([posterior_variance_1, posterior_variance_1, "
            "posterior_variance_2, ...]) in forward t order"
        ),
        "raw_to_logstd": (
            "frac=(raw+1)/2 without clipping; logstd=0.5*(frac*log(beta_t) "
            "+ (1-frac)*posterior_log_variance_clipped_t), replayed in upstream "
            "float32 operation order"
        ),
        "actual_covariance": (
            "P uses the conditional learned-range head because forward_with_cfg leaves "
            "the model's variance channels in the conditional first-B half"
        ),
        "verification": (
            "transformed conditional logstd must reconstruct log(saved p_standard_deviation) "
            "with maximum absolute error <=1e-5, including the recorded t=0 head"
        ),
    },
    "posterior_mean": {
        "coef1": (
            "beta_t*sqrt(alpha_bar_{t-1})/(1-alpha_bar_t)"
        ),
        "coef2": (
            "(1-alpha_bar_{t-1})*sqrt(alpha_t)/(1-alpha_bar_t)"
        ),
        "branch_pred_x0": (
            "(x_t-sqrt(1-alpha_bar_t)*eps_branch)/sqrt(alpha_bar_t)"
        ),
        "branch_mean": "mu_branch=coef1_t*pred_x0_branch+coef2_t*x_t",
        "verification": (
            "branch x0 and mean use upstream float32 operation order; the mean derived "
            "from actual guided epsilon must reconstruct saved p_mean with maximum "
            "absolute error <=2e-6, and conditional-Q channel 3 must be bitwise unchanged"
        ),
    },
    "weak_processes": {
        "P": "actual saved three-channel-CFG guided mean and saved diagonal covariance",
        "weak_conditional_cfg1_Q": (
            "raw conditional epsilon branch posterior mean (CFG=1), with P covariance"
        ),
        "weak_unconditional_Q": (
            "raw unconditional epsilon branch posterior mean, with P covariance"
        ),
        "fixed_covariance_policy": (
            "both Q alternatives change the mean only and retain P covariance; raw Q "
            "variance heads are transformed and measured but never substituted into Q"
        ),
        "delta": "delta_k=mu_Q,k-mu_P,k",
        "whitened_shift": "w_k=delta_k/sigma_P,k",
        "predictable_D": "D_k=0.5*sum_{c,h,w}(w_k^2)",
        "post_transition_increment": "inc_k=sum(w_k*z_k)-D_k = log(q_k/p_k)",
        "cumulative_log_e": "log E_k=sum_{s<=k} inc_s",
        "running_max": "max(0, log E_0, ..., log E_k), including initial E=1",
    },
    "fixed_tile_mixture": {
        "grid": "fixed row-major 4x4 grid of sixteen 8x8 latent tiles",
        "component": (
            "component j uses the same Q-minus-P mean shift only inside fixed tile j, "
            "equals P outside it, and retains P covariance everywhere"
        ),
        "component_increment": (
            "inc_{k,j}=sum_{c,h,w in tile j}(w*z)-0.5*sum_{c,h,w in tile j}(w^2)"
        ),
        "mixture_log_e": (
            "log E_mix,k=logmeanexp_j(sum_{s<=k} inc_{s,j}) with a fixed uniform 1/16 prior"
        ),
        "purpose": "mitigate full-dimensional likelihood-ratio collapse",
        "componentwise_max_policy": "never emitted or used as calibrated evidence",
    },
    "analysis_policy": {
        "labels": "none read, joined, inferred, or emitted",
        "auc": "not computed",
        "thresholds": "not selected or evaluated",
        "trained_combinations": "none",
        "intervention": "none",
    },
}


@dataclass(frozen=True)
class ReverseSchedule:
    """Schedule arrays in saved reverse-row order k=0..249."""

    alpha_bar: np.ndarray
    alpha_bar_previous: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    posterior_variance: np.ndarray
    posterior_log_variance_clipped: np.ndarray
    posterior_mean_coef1: np.ndarray
    posterior_mean_coef2: np.ndarray


@dataclass(frozen=True)
class Extraction:
    record: TraceRecord
    tracks: dict[str, np.ndarray]
    specs: dict[str, TrackSpec]
    auxiliary: dict[str, np.ndarray]
    diagnostics: dict[str, float]
    alpha_bar: np.ndarray


def _array_record(array: np.ndarray) -> dict[str, Any]:
    value = np.ascontiguousarray(array)
    return {
        "shape": list(value.shape),
        "dtype": value.dtype.str,
        "raw_sha256": sha256_array(value),
    }


def derive_reverse_schedule(alpha_bar: np.ndarray) -> ReverseSchedule:
    """Reconstruct exact diffusion coefficients using only saved alpha_bar."""

    abar = np.asarray(alpha_bar, dtype=np.float64)
    if (
        abar.shape != (STEPS,)
        or not np.isfinite(abar).all()
        or np.any(abar <= 0.0)
        or np.any(abar > 1.0)
        or np.any(np.diff(abar) <= 0.0)
    ):
        raise RuntimeError("alpha_bar must be finite and strictly increase in reverse order")

    # For saved row k (internal t=249-k), the forward predecessor alpha_bar_{t-1}
    # is the next saved row; t=0 uses the conventional alpha_bar_{-1}=1.
    abar_previous = np.concatenate((abar[1:], np.ones(1, dtype=np.float64)))
    alpha = abar / abar_previous
    beta = 1.0 - alpha
    if np.any(alpha <= 0.0) or np.any(alpha >= 1.0) or np.any(beta <= 0.0):
        raise RuntimeError("alpha_bar-derived one-step alpha/beta is invalid")

    posterior_variance = beta * (1.0 - abar_previous) / (1.0 - abar)
    if (
        not np.isfinite(posterior_variance).all()
        or np.any(posterior_variance[:-1] <= 0.0)
        or posterior_variance[-1] != 0.0
    ):
        raise RuntimeError("alpha_bar-derived posterior variance is invalid")
    posterior_variance_clipped = posterior_variance.copy()
    posterior_variance_clipped[-1] = posterior_variance[-2]
    posterior_log_variance_clipped = np.log(posterior_variance_clipped)

    denominator = 1.0 - abar
    coef1 = beta * np.sqrt(abar_previous) / denominator
    coef2 = (1.0 - abar_previous) * np.sqrt(alpha) / denominator
    arrays = (
        abar_previous,
        alpha,
        beta,
        posterior_variance,
        posterior_log_variance_clipped,
        coef1,
        coef2,
    )
    if not all(np.isfinite(value).all() for value in arrays):
        raise RuntimeError("derived reverse schedule contains non-finite values")
    if not np.allclose((coef1[-1], coef2[-1]), (1.0, 0.0), rtol=0.0, atol=2e-12):
        raise RuntimeError("derived t=0 posterior mean coefficients are not (1,0)")
    return ReverseSchedule(
        alpha_bar=np.ascontiguousarray(abar),
        alpha_bar_previous=np.ascontiguousarray(abar_previous),
        alpha=np.ascontiguousarray(alpha),
        beta=np.ascontiguousarray(beta),
        posterior_variance=np.ascontiguousarray(posterior_variance),
        posterior_log_variance_clipped=np.ascontiguousarray(
            posterior_log_variance_clipped
        ),
        posterior_mean_coef1=np.ascontiguousarray(coef1),
        posterior_mean_coef2=np.ascontiguousarray(coef2),
    )


def learned_range_logstd(raw: np.ndarray, schedule: ReverseSchedule) -> np.ndarray:
    """Apply upstream float32 LEARNED_RANGE; raw values are intentionally unclipped."""

    value = np.asarray(raw)
    if value.ndim != 5 or value.shape[1:] != (
        STEPS,
        CHANNELS,
        LATENT_SIZE,
        LATENT_SIZE,
    ):
        raise ValueError(f"unexpected learned-range head shape: {value.shape}")
    # _extract_into_tensor casts schedule arrays to float32 before the upstream
    # elementwise expression.  Preserve that operation order rather than doing
    # a mathematically equivalent float64 interpolation.
    frac = (value.astype(np.float32, copy=False) + np.float32(1.0)) / np.float32(2.0)
    maximum_log_variance = np.log(schedule.beta).astype(np.float32)[
        None, :, None, None, None
    ]
    minimum_log_variance = schedule.posterior_log_variance_clipped.astype(np.float32)[
        None, :, None, None, None
    ]
    result = np.float32(0.5) * (
        frac * maximum_log_variance
        + (np.float32(1.0) - frac) * minimum_log_variance
    )
    if not np.isfinite(result).all():
        raise RuntimeError("learned-range transformation produced non-finite logstd")
    return np.ascontiguousarray(result)


def guided_epsilon(
    conditional: np.ndarray, unconditional: np.ndarray, cfg_scale: float
) -> np.ndarray:
    """Reproduce released float32 three-channel CFG semantics."""

    if conditional.shape != unconditional.shape or conditional.dtype != np.float32:
        raise ValueError("raw epsilon branches must be shape-matched float32 arrays")
    result = conditional.copy()
    scale = np.float32(cfg_scale)
    result[:, :, :3] = unconditional[:, :, :3] + scale * (
        conditional[:, :, :3] - unconditional[:, :, :3]
    )
    return np.ascontiguousarray(result)


def branch_pred_x0(
    state: np.ndarray, epsilon: np.ndarray, schedule: ReverseSchedule
) -> np.ndarray:
    # Upstream _extract_into_tensor converts these schedule arrays to float32.
    sqrt_recip = np.sqrt(1.0 / schedule.alpha_bar).astype(np.float32)[
        None, :, None, None, None
    ]
    sqrt_recipm1 = np.sqrt(1.0 / schedule.alpha_bar - 1.0).astype(np.float32)[
        None, :, None, None, None
    ]
    return np.ascontiguousarray(
        sqrt_recip * state.astype(np.float32, copy=False)
        - sqrt_recipm1 * epsilon.astype(np.float32, copy=False)
    )


def branch_posterior_mean(
    state: np.ndarray, epsilon: np.ndarray, schedule: ReverseSchedule
) -> np.ndarray:
    pred = branch_pred_x0(state, epsilon, schedule)
    coef1 = schedule.posterior_mean_coef1.astype(np.float32)[
        None, :, None, None, None
    ]
    coef2 = schedule.posterior_mean_coef2.astype(np.float32)[
        None, :, None, None, None
    ]
    result = coef1 * pred + coef2 * state.astype(np.float32, copy=False)
    del pred
    return np.ascontiguousarray(result)


def _stable_logmeanexp(values: np.ndarray, axis: int) -> np.ndarray:
    value = np.asarray(values, dtype=np.float64)
    maximum = np.max(value, axis=axis, keepdims=True)
    result = maximum + np.log(np.mean(np.exp(value - maximum), axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis)


def _running_max_from_initial_zero(values: np.ndarray) -> np.ndarray:
    value = np.asarray(values, dtype=np.float64)
    initial = np.zeros((value.shape[0], 1), dtype=np.float64)
    return np.maximum.accumulate(np.concatenate((initial, value), axis=1), axis=1)[:, 1:]


def _group_mean(values: np.ndarray, channels: slice | Sequence[int]) -> np.ndarray:
    return np.mean(values[:, :, channels], axis=(2, 3, 4), dtype=np.float64)


def _add_field_rms_and_concentration(
    tracks: dict[str, np.ndarray],
    specs: dict[str, TrackSpec],
    field: np.ndarray,
    *,
    prefix: str,
    family: str,
    availability: str,
    observation_offset_steps: int,
    field_formula: str,
    uses_realized_innovation: bool = False,
    deployment_note: str = "none",
) -> None:
    for group, channels in GROUPS.items():
        _add_track(
            tracks,
            specs,
            _rms(field, channels),
            name=f"{prefix}_rms_{group}",
            family=family,
            availability=availability,
            observation_offset_steps=observation_offset_steps,
            formula=f"RMS over {group} of {field_formula}",
            uses_realized_innovation=uses_realized_innovation,
            deployment_note=deployment_note,
        )
        _add_track(
            tracks,
            specs,
            _tile_concentration(field, channels, grid=GRID_SIZE),
            name=f"{prefix}_tile4x4_concentration_{group}",
            family=family,
            availability=availability,
            observation_offset_steps=observation_offset_steps,
            formula=(
                f"max/sum of sixteen equal-area tile mean-squared energies of "
                f"{field_formula}, {group}"
            ),
            uses_realized_innovation=uses_realized_innovation,
            deployment_note=deployment_note,
        )


def exact_same_covariance_evidence(
    delta: np.ndarray,
    sigma: np.ndarray,
    innovation: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute full and fixed-tile exact q/p likelihood-ratio processes."""

    expected = (delta.shape[0], STOCHASTIC_STEPS, CHANNELS, LATENT_SIZE, LATENT_SIZE)
    if delta.shape != (delta.shape[0], STEPS, CHANNELS, LATENT_SIZE, LATENT_SIZE):
        raise ValueError(f"unexpected posterior mean shift shape: {delta.shape}")
    if sigma.shape != delta.shape or innovation.shape != delta.shape:
        raise ValueError("delta, sigma, and innovation shapes differ")
    sigma_effective = sigma[:, :-1].astype(np.float64)
    if np.any(sigma_effective <= 0.0):
        raise ValueError("P standard deviation must be positive")
    whitened = delta[:, :-1].astype(np.float64, copy=False) / sigma_effective
    z = innovation[:, :-1]
    if whitened.shape != expected:
        raise AssertionError("masked-t0 exclusion produced an unexpected shape")

    full_D = 0.5 * np.einsum(
        "btchw,btchw->bt", whitened, whitened, dtype=np.float64, optimize=True
    )
    full_linear = np.einsum(
        "btchw,btchw->bt", whitened, z, dtype=np.float64, optimize=True
    )
    full_increment = full_linear - full_D
    full_cumulative = np.cumsum(full_increment, axis=1, dtype=np.float64)
    full_running_max = _running_max_from_initial_zero(full_cumulative)

    batch = delta.shape[0]
    tile_D = np.empty((batch, STOCHASTIC_STEPS, TILE_COUNT), dtype=np.float64)
    tile_increment = np.empty_like(tile_D)
    tile_index = 0
    for row in range(GRID_SIZE):
        for column in range(GRID_SIZE):
            region = (
                slice(None),
                slice(None),
                slice(None),
                slice(row * TILE_SIZE, (row + 1) * TILE_SIZE),
                slice(column * TILE_SIZE, (column + 1) * TILE_SIZE),
            )
            w_tile = whitened[region]
            z_tile = z[region]
            component_D = 0.5 * np.einsum(
                "btchw,btchw->bt", w_tile, w_tile, dtype=np.float64, optimize=True
            )
            component_linear = np.einsum(
                "btchw,btchw->bt", w_tile, z_tile, dtype=np.float64, optimize=True
            )
            tile_D[:, :, tile_index] = component_D
            tile_increment[:, :, tile_index] = component_linear - component_D
            tile_index += 1
    tile_cumulative = np.cumsum(tile_increment, axis=1, dtype=np.float64)
    mixture_log_e = _stable_logmeanexp(tile_cumulative, axis=2)
    mixture_running_max = _running_max_from_initial_zero(mixture_log_e)

    # Equal-sized tiles partition all coordinates.  This independently checks
    # both the full LR implementation and the component formulas.
    if not np.allclose(tile_D.sum(axis=2), full_D, rtol=3e-13, atol=3e-11):
        raise RuntimeError("fixed-tile D components do not reconstruct full D")
    if not np.allclose(
        tile_increment.sum(axis=2), full_increment, rtol=3e-13, atol=3e-11
    ):
        raise RuntimeError("fixed-tile LR increments do not reconstruct full LR")
    results = {
        "D": full_D,
        "increment": full_increment,
        "cumulative_log_e": full_cumulative,
        "running_max_log_e": full_running_max,
        "tile_component_D": tile_D,
        "tile_component_increment": tile_increment,
        "tile_component_cumulative_log_e": tile_cumulative,
        "tile_D_mean": np.mean(tile_D, axis=2),
        "tile_D_maximum": np.max(tile_D, axis=2),
        "tile_mixture_log_e": mixture_log_e,
        "tile_mixture_running_max_log_e": mixture_running_max,
    }
    if not all(np.isfinite(value).all() for value in results.values()):
        raise RuntimeError("same-covariance evidence contains non-finite values")
    return {name: np.ascontiguousarray(value) for name, value in results.items()}


def _add_q_shift_tracks(
    tracks: dict[str, np.ndarray],
    specs: dict[str, TrackSpec],
    delta: np.ndarray,
    sigma: np.ndarray,
    q_name: str,
) -> None:
    _add_field_rms_and_concentration(
        tracks,
        specs,
        delta,
        prefix=f"{q_name}_posterior_mean_shift",
        family="branch_posterior_mean_shift",
        availability="predictable",
        observation_offset_steps=0,
        field_formula=f"mu_{q_name}-saved actual guided mu_P",
        deployment_note="t=0 mean shift is descriptive only; no covariance or LR is applied",
    )
    standardized = delta[:, :-1] / sigma[:, :-1].astype(np.float64)
    _add_field_rms_and_concentration(
        tracks,
        specs,
        standardized,
        prefix=f"{q_name}_posterior_mean_shift_sigma_standardized",
        family="branch_posterior_mean_shift",
        availability="predictable",
        observation_offset_steps=0,
        field_formula=f"(mu_{q_name}-mu_P)/sigma_P on stochastic rows",
        deployment_note="masked t=0 draw/covariance excluded",
    )
    del standardized


def _add_evidence_tracks(
    tracks: dict[str, np.ndarray],
    specs: dict[str, TrackSpec],
    auxiliary: dict[str, np.ndarray],
    q_name: str,
    evidence: Mapping[str, np.ndarray],
) -> None:
    predictable = {
        "full_D_nats": (
            evidence["D"],
            "0.5*sum_all4(((mu_Q-mu_P)/sigma_P)^2)",
        ),
        "tile4x4_component_D_mean_nats": (
            evidence["tile_D_mean"],
            "mean over sixteen fixed-tile component D values",
        ),
        "tile4x4_component_D_maximum_nats": (
            evidence["tile_D_maximum"],
            "maximum diagnostic over sixteen fixed-tile component D values; not evidence",
        ),
    }
    for suffix, (values, formula) in predictable.items():
        _add_track(
            tracks,
            specs,
            values,
            name=f"{q_name}_{suffix}",
            family="same_covariance_weak_process_evidence",
            availability="predictable",
            observation_offset_steps=0,
            formula=formula,
            deployment_note="stochastic k=0..248 only; masked t=0 excluded",
        )
    post_transition = {
        "full_log_lr_increment": (
            evidence["increment"],
            "sum_all4((delta/sigma)*z)-0.5*sum_all4((delta/sigma)^2)",
        ),
        "full_cumulative_log_e": (
            evidence["cumulative_log_e"],
            "cumulative sum of exact full-dimensional one-step log(q/p)",
        ),
        "full_running_max_log_e_from_E0": (
            evidence["running_max_log_e"],
            "running maximum of full cumulative log-E including initial log(E)=0",
        ),
        "tile4x4_uniform_mixture_log_e": (
            evidence["tile_mixture_log_e"],
            "logmeanexp of sixteen fixed-tile component cumulative exact log LRs",
        ),
        "tile4x4_uniform_mixture_running_max_log_e_from_E0": (
            evidence["tile_mixture_running_max_log_e"],
            "running maximum of uniform fixed-tile mixture log-E including initial log(E)=0",
        ),
    }
    for suffix, (values, formula) in post_transition.items():
        _add_track(
            tracks,
            specs,
            values,
            name=f"{q_name}_{suffix}",
            family="same_covariance_weak_process_evidence",
            availability="online_causal",
            observation_offset_steps=1,
            formula=formula,
            uses_realized_innovation=True,
            deployment_note="known after transition; masked t=0 draw excluded",
        )
    for suffix in (
        "tile_component_D",
        "tile_component_increment",
        "tile_component_cumulative_log_e",
    ):
        name = f"{q_name}_tile4x4_{suffix}"
        if name in auxiliary:
            raise RuntimeError(f"duplicate auxiliary array: {name}")
        auxiliary[name] = np.ascontiguousarray(evidence[suffix])


def extract_trace(record: TraceRecord, arrays: Mapping[str, np.ndarray]) -> Extraction:
    tracks: dict[str, np.ndarray] = {}
    specs: dict[str, TrackSpec] = {}
    auxiliary: dict[str, np.ndarray] = {}
    schedule = derive_reverse_schedule(arrays["alpha_bar"])
    state = arrays["state_before"]
    p_mean = arrays["p_mean"]
    sigma = arrays["p_standard_deviation"]
    innovation = arrays["transition_innovation"]
    conditional_epsilon = arrays["conditional_epsilon_raw"]
    unconditional_epsilon = arrays["unconditional_epsilon_raw"]

    drift = p_mean.astype(np.float64) - state.astype(np.float64)
    _add_field_rms_and_concentration(
        tracks,
        specs,
        drift,
        prefix="reverse_mean_drift_raw",
        family="reverse_mean_drift",
        availability="predictable",
        observation_offset_steps=0,
        field_formula="saved p_mean - state_before",
        deployment_note="t=0 drift is the deterministic endpoint update",
    )
    for group, channels in GROUPS.items():
        raw_rms = _rms(drift, channels)
        state_rms = _rms(state, channels)
        relative_rms = raw_rms / (state_rms + EPS)
        _add_track(
            tracks,
            specs,
            relative_rms,
            name=f"reverse_mean_drift_relative_rms_{group}",
            family="reverse_mean_drift",
            availability="predictable",
            observation_offset_steps=0,
            formula=(
                f"RMS_{group}(p_mean-state_before)/(RMS_{group}(state_before)+1e-12)"
            ),
            deployment_note="t=0 drift is the deterministic endpoint update",
        )
        _add_track(
            tracks,
            specs,
            _tile_concentration(drift, channels, grid=GRID_SIZE),
            name=f"reverse_mean_drift_relative_tile4x4_concentration_{group}",
            family="reverse_mean_drift",
            availability="predictable",
            observation_offset_steps=0,
            formula=(
                f"tile concentration of drift/(RMS_{group}(state_before)+1e-12); "
                "equal to raw drift concentration because normalization is one positive scalar"
            ),
            deployment_note="reported explicitly to complete the raw/relative audit",
        )
    standardized_drift = drift[:, :-1] / sigma[:, :-1].astype(np.float64)
    _add_field_rms_and_concentration(
        tracks,
        specs,
        standardized_drift,
        prefix="reverse_mean_drift_sigma_standardized",
        family="reverse_mean_drift",
        availability="predictable",
        observation_offset_steps=0,
        field_formula="(saved p_mean-state_before)/saved P sigma",
        deployment_note="stochastic k=0..248 only; masked t=0 covariance excluded",
    )
    del standardized_drift

    stochastic_displacement = sigma[:, :-1].astype(np.float64) * innovation[
        :, :-1
    ].astype(np.float64)
    realized_update = state[:, 1:].astype(np.float64) - state[:, :-1].astype(np.float64)
    update_replay_error = float(
        np.max(np.abs(realized_update - (drift[:, :-1] + stochastic_displacement)))
    )
    if update_replay_error > TRANSITION_RECONSTRUCTION_TOLERANCE:
        raise RuntimeError(
            f"realized update != drift + sigma*z: max_abs={update_replay_error}"
        )
    _add_field_rms_and_concentration(
        tracks,
        specs,
        realized_update,
        prefix="realized_update",
        family="realized_transition_geometry",
        availability="online_causal",
        observation_offset_steps=1,
        field_formula="state_before[k+1]-state_before[k]",
        uses_realized_innovation=True,
        deployment_note="stochastic k=0..248 only; masked t=0 excluded",
    )
    for group, channels in GROUPS.items():
        stochastic_rms = _rms(stochastic_displacement, channels)
        drift_rms = _rms(drift[:, :-1], channels)
        _add_track(
            tracks,
            specs,
            stochastic_rms,
            name=f"stochastic_displacement_rms_{group}",
            family="realized_transition_geometry",
            availability="online_causal",
            observation_offset_steps=1,
            formula=f"RMS over {group} of saved sigma*z",
            uses_realized_innovation=True,
            deployment_note="stochastic k=0..248 only; masked t=0 excluded",
        )
        _add_track(
            tracks,
            specs,
            _cosine(drift[:, :-1], stochastic_displacement, channels),
            name=f"drift_noise_cosine_{group}",
            family="realized_transition_geometry",
            availability="online_causal",
            observation_offset_steps=1,
            formula=f"cosine(saved p_mean-state_before, saved sigma*z), {group}",
            uses_realized_innovation=True,
            deployment_note="stochastic k=0..248 only; masked t=0 excluded",
        )
        _add_track(
            tracks,
            specs,
            stochastic_rms / (drift_rms + EPS),
            name=f"stochastic_to_drift_rms_ratio_{group}",
            family="realized_transition_geometry",
            availability="online_causal",
            observation_offset_steps=1,
            formula=f"RMS_{group}(sigma*z)/(RMS_{group}(p_mean-state)+1e-12)",
            uses_realized_innovation=True,
            deployment_note="stochastic k=0..248 only; masked t=0 excluded",
        )
    del realized_update, stochastic_displacement

    actual_guided_epsilon = guided_epsilon(
        conditional_epsilon, unconditional_epsilon, record.cfg_scale
    )
    _add_field_rms_and_concentration(
        tracks,
        specs,
        actual_guided_epsilon,
        prefix="guided_epsilon",
        family="guided_epsilon_score_displacement",
        availability="predictable",
        observation_offset_steps=0,
        field_formula="actual released three-channel-CFG epsilon_P",
    )
    one_minus = (1.0 - schedule.alpha_bar)[None, :, None, None, None]
    guided_score = -actual_guided_epsilon.astype(np.float64) / np.sqrt(one_minus)
    _add_field_rms_and_concentration(
        tracks,
        specs,
        guided_score,
        prefix="guided_score",
        family="guided_epsilon_score_displacement",
        availability="predictable",
        observation_offset_steps=0,
        field_formula="-epsilon_P/sqrt(1-alpha_bar)",
    )
    del guided_score
    denoising_displacement = -np.sqrt(one_minus) * actual_guided_epsilon.astype(
        np.float64
    )
    _add_field_rms_and_concentration(
        tracks,
        specs,
        denoising_displacement,
        prefix="guided_denoising_displacement",
        family="guided_epsilon_score_displacement",
        availability="predictable",
        observation_offset_steps=0,
        field_formula="-sqrt(1-alpha_bar)*epsilon_P = (1-alpha_bar)*score_P",
    )
    denoising_identity_error = float(
        np.max(
            np.abs(
                denoising_displacement
                - (
                    np.sqrt(schedule.alpha_bar)[None, :, None, None, None]
                    * arrays["pred_xstart"].astype(np.float64)
                    - state.astype(np.float64)
                )
            )
        )
    )
    if denoising_identity_error > DENOISING_IDENTITY_TOLERANCE:
        raise RuntimeError(
            "guided denoising displacement identity failed: "
            f"max_abs={denoising_identity_error}"
        )
    del denoising_displacement

    conditional_logstd = learned_range_logstd(
        arrays["conditional_variance_values_raw"], schedule
    )
    unconditional_logstd = learned_range_logstd(
        arrays["unconditional_variance_values_raw"], schedule
    )
    saved_logstd = np.log(sigma.astype(np.float64))
    logstd_reconstruction_error = float(
        np.max(np.abs(conditional_logstd - saved_logstd))
    )
    if logstd_reconstruction_error > LOGSTD_RECONSTRUCTION_TOLERANCE:
        raise RuntimeError(
            "conditional learned-range logstd does not reconstruct saved pstd: "
            f"max_abs={logstd_reconstruction_error}"
        )
    transformed_gap = conditional_logstd - unconditional_logstd
    for group, channels in GROUPS.items():
        for branch, value in (
            ("conditional", conditional_logstd),
            ("unconditional", unconditional_logstd),
        ):
            _add_track(
                tracks,
                specs,
                _group_mean(value, channels),
                name=f"learned_range_{branch}_logstd_mean_{group}",
                family="transformed_learned_range_logstd",
                availability="predictable",
                observation_offset_steps=0,
                formula=(
                    f"mean over {group} of exact unclipped learned-range {branch} logstd"
                ),
                deployment_note="includes the recorded but operationally masked t=0 head",
            )
        _add_track(
            tracks,
            specs,
            _group_mean(transformed_gap, channels),
            name=f"learned_range_cond_minus_uncond_logstd_signed_mean_{group}",
            family="transformed_learned_range_logstd",
            availability="predictable",
            observation_offset_steps=0,
            formula=f"mean over {group} of transformed conditional-unconditional logstd",
            deployment_note="Q evidence retains conditional P covariance; gap is descriptive",
        )
        _add_track(
            tracks,
            specs,
            _rms(transformed_gap, channels),
            name=f"learned_range_cond_minus_uncond_logstd_gap_rms_{group}",
            family="transformed_learned_range_logstd",
            availability="predictable",
            observation_offset_steps=0,
            formula=f"RMS over {group} of transformed conditional-unconditional logstd",
            deployment_note="Q evidence retains conditional P covariance; gap is descriptive",
        )
        _add_track(
            tracks,
            specs,
            _tile_concentration(transformed_gap, channels, grid=GRID_SIZE),
            name=(
                f"learned_range_cond_minus_uncond_logstd_gap_"
                f"tile4x4_concentration_{group}"
            ),
            family="transformed_learned_range_logstd",
            availability="predictable",
            observation_offset_steps=0,
            formula=(
                f"fixed 4x4 tile concentration of transformed conditional-unconditional "
                f"logstd gap, {group}"
            ),
            deployment_note="Q evidence retains conditional P covariance; gap is descriptive",
        )
    del saved_logstd, transformed_gap, unconditional_logstd, conditional_logstd

    guided_mean = branch_posterior_mean(state, actual_guided_epsilon, schedule)
    guided_mean_reconstruction_error = float(
        np.max(np.abs(guided_mean - p_mean.astype(np.float64)))
    )
    guided_mean_bitwise_equal = np.array_equal(guided_mean, p_mean)
    if guided_mean_reconstruction_error > MEAN_RECONSTRUCTION_TOLERANCE:
        raise RuntimeError(
            "alpha_bar-derived guided posterior mean does not reconstruct saved p_mean: "
            f"max_abs={guided_mean_reconstruction_error}"
        )
    del guided_mean

    for q_name, branch_epsilon in (
        ("weak_conditional_cfg1", conditional_epsilon),
        ("weak_unconditional", unconditional_epsilon),
    ):
        q_mean = branch_posterior_mean(state, branch_epsilon, schedule)
        if q_name == "weak_conditional_cfg1" and not np.array_equal(
            q_mean[:, :, 3], p_mean[:, :, 3]
        ):
            error = float(
                np.max(
                    np.abs(
                        q_mean[:, :, 3].astype(np.float64)
                        - p_mean[:, :, 3].astype(np.float64)
                    )
                )
            )
            raise RuntimeError(
                "conditional-Q channel 3 must equal actual P mean bitwise: "
                f"max_abs={error}"
            )
        delta = q_mean - p_mean.astype(np.float64)
        del q_mean
        _add_q_shift_tracks(tracks, specs, delta, sigma, q_name)
        evidence = exact_same_covariance_evidence(delta, sigma, innovation)
        _add_evidence_tracks(tracks, specs, auxiliary, q_name, evidence)
        del evidence, delta

    diagnostics = {
        "conditional_logstd_reconstruction_max_abs_error": logstd_reconstruction_error,
        "guided_posterior_mean_reconstruction_max_abs_error": (
            guided_mean_reconstruction_error
        ),
        "transition_update_reconstruction_max_abs_error": update_replay_error,
        "guided_denoising_displacement_identity_max_abs_error": (
            denoising_identity_error
        ),
        "guided_posterior_mean_bitwise_equal": float(
            guided_mean_bitwise_equal
        ),
        "conditional_q_channel4_mean_bitwise_equal": 1.0,
        "conditional_variance_raw_minimum": float(
            np.min(arrays["conditional_variance_values_raw"])
        ),
        "conditional_variance_raw_maximum": float(
            np.max(arrays["conditional_variance_values_raw"])
        ),
        "unconditional_variance_raw_minimum": float(
            np.min(arrays["unconditional_variance_values_raw"])
        ),
        "unconditional_variance_raw_maximum": float(
            np.max(arrays["unconditional_variance_values_raw"])
        ),
    }
    if not all(math.isfinite(value) for value in diagnostics.values()):
        raise RuntimeError("trace diagnostics contain non-finite values")
    return Extraction(
        record=record,
        tracks=tracks,
        specs=specs,
        auxiliary=auxiliary,
        diagnostics=diagnostics,
        alpha_bar=schedule.alpha_bar,
    )


def _combine_auxiliary(
    extractions: Sequence[Extraction],
) -> dict[str, np.ndarray]:
    if not extractions:
        raise RuntimeError("no extractions to combine")
    names = set(extractions[0].auxiliary)
    shapes = {
        name: extractions[0].auxiliary[name].shape[1:] for name in sorted(names)
    }
    for extraction in extractions:
        if set(extraction.auxiliary) != names:
            raise RuntimeError("auxiliary time-series schemas differ across traces")
        for name in names:
            value = extraction.auxiliary[name]
            if value.shape[1:] != shapes[name] or not np.isfinite(value).all():
                raise RuntimeError(f"invalid auxiliary time series: {name}")
    return {
        name: np.ascontiguousarray(
            np.concatenate(
                [extraction.auxiliary[name] for extraction in extractions], axis=0
            )
        )
        for name in sorted(names)
    }


def _feature_frame_and_catalog(
    extractions: Sequence[Extraction],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], dict[str, TrackSpec]]:
    per_trace = [(item.tracks, item.specs) for item in extractions]
    tracks, specs = _combine_trace_tracks(per_trace)
    scalar_frame, catalog_map = reduce_tracks(tracks, specs)
    sample_rows: list[dict[str, Any]] = []
    for run_index, extraction in enumerate(extractions):
        record = extraction.record
        for class_slot, class_id in enumerate(record.classes):
            sample_rows.append(
                {
                    "sample_index": len(sample_rows),
                    "run_index": run_index,
                    "global_seed": record.global_seed,
                    "class_slot": class_slot,
                    "class_id": class_id,
                    "trace_dir": str(record.root),
                    "endpoint_png_path": str(
                        record.root / f"images/{class_slot:02d}_class{class_id:04d}.png"
                    ),
                }
            )
    identity = pd.DataFrame(sample_rows, columns=IDENTIFIER_COLUMNS)
    if len(identity) != len(scalar_frame):
        raise RuntimeError("sample identity and scalar feature row counts differ")
    frame = pd.concat(
        [identity.reset_index(drop=True), scalar_frame.reset_index(drop=True)], axis=1
    )
    catalog = pd.DataFrame(
        [catalog_map[name] for name in sorted(catalog_map)],
    )
    catalog.insert(0, "feature_index", np.arange(len(catalog), dtype=np.int32))
    feature_names = catalog["feature"].astype(str).tolist()
    if len(feature_names) != len(set(feature_names)):
        raise RuntimeError("feature catalog contains duplicate names")
    if tuple(frame.columns[: len(IDENTIFIER_COLUMNS)]) != IDENTIFIER_COLUMNS:
        raise RuntimeError("sample identifier column order changed")
    if set(frame.columns[len(IDENTIFIER_COLUMNS) :]) != set(feature_names):
        raise RuntimeError("sample feature columns and catalog differ")
    numeric = frame[feature_names].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise RuntimeError("sample feature table contains non-finite values")
    return frame, catalog, tracks, specs


def _time_series_payload(
    frame: pd.DataFrame,
    tracks: Mapping[str, np.ndarray],
    auxiliary: Mapping[str, np.ndarray],
    schedule: ReverseSchedule,
) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {
        "sample_index": frame["sample_index"].to_numpy(np.int32),
        "global_seed": frame["global_seed"].to_numpy(np.int64),
        "class_slot": frame["class_slot"].to_numpy(np.int16),
        "class_id": frame["class_id"].to_numpy(np.int16),
        "sampling_step_250": np.arange(STEPS, dtype=np.int16),
        "internal_timestep_250": np.arange(STEPS - 1, -1, -1, dtype=np.int16),
        "stochastic_sampling_step_249": np.arange(STOCHASTIC_STEPS, dtype=np.int16),
        "stochastic_internal_timestep_249": np.arange(
            STEPS - 1, 0, -1, dtype=np.int16
        ),
        "alpha_bar_250": schedule.alpha_bar,
        "alpha_bar_previous_250": schedule.alpha_bar_previous,
        "alpha_250": schedule.alpha,
        "beta_250": schedule.beta,
        "posterior_variance_250": schedule.posterior_variance,
        "posterior_log_variance_clipped_250": (
            schedule.posterior_log_variance_clipped
        ),
        "posterior_mean_coef1_250": schedule.posterior_mean_coef1,
        "posterior_mean_coef2_250": schedule.posterior_mean_coef2,
    }
    for source in (tracks, auxiliary):
        for name, value in source.items():
            if name in payload:
                raise RuntimeError(f"duplicate time-series array name: {name}")
            payload[name] = np.ascontiguousarray(value)
    if not all(np.isfinite(value).all() for value in payload.values()):
        raise RuntimeError("time-series payload contains non-finite values")
    return payload


def publish(args: argparse.Namespace) -> Path:
    trace_dirs = discover_trace_dirs(args)
    extractions: list[Extraction] = []
    for index, path in enumerate(trace_dirs, start=1):
        print(f"validating/extracting trace {index}/{len(trace_dirs)}: {path}", flush=True)
        record, arrays = load_validated_trace(path)
        extraction = extract_trace(record, arrays)
        extractions.append(extraction)
        del arrays

    seeds = [item.record.global_seed for item in extractions]
    if len(seeds) != len(set(seeds)):
        raise RuntimeError(f"duplicate seeds across trace inputs: {seeds}")
    extractions.sort(key=lambda item: item.record.global_seed)
    records = [item.record for item in extractions]
    observed_seeds = tuple(record.global_seed for record in records)
    if args.expected_seeds is not None and observed_seeds != tuple(
        sorted(args.expected_seeds)
    ):
        raise RuntimeError(
            f"observed seeds differ from --expected-seeds: {observed_seeds}"
        )
    class_orders = {record.classes for record in records}
    fingerprints = {record.scientific_fingerprint_sha256 for record in records}
    if len(class_orders) != 1 or len(fingerprints) != 1:
        raise RuntimeError("trace class order or scientific sampler fingerprint differs")
    classes = records[0].classes
    if args.expected_classes is not None and classes != args.expected_classes:
        raise RuntimeError(
            f"ordered classes differ: observed={classes}, expected={args.expected_classes}"
        )
    reference_alpha = extractions[0].alpha_bar
    if any(not np.array_equal(item.alpha_bar, reference_alpha) for item in extractions):
        raise RuntimeError("alpha_bar schedule differs across trace runs")

    frame, catalog, tracks, specs = _feature_frame_and_catalog(extractions)
    auxiliary = _combine_auxiliary(extractions)
    schedule = derive_reverse_schedule(reference_alpha)
    time_series = _time_series_payload(frame, tracks, auxiliary, schedule)

    output = args.output_dir.expanduser().absolute()
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite existing output path: {output}")
    output = output.resolve()
    input_roots = [record.root for record in records]
    if any(
        root == output or root in output.parents or output in root.parents
        for root in input_roots
    ):
        raise RuntimeError("analysis output must not overlap a trace input")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent))
    try:
        analysis_source = Path(__file__).resolve()
        helper_source = ROOT / "experiments/analyze_dit_bad_good_custom_traces.py"
        shutil.copyfile(analysis_source, staging / "analysis_source.py")
        atomic_json_dump(PROTOCOL, staging / "protocol_snapshot.json")
        frame.to_csv(staging / "sample_features.csv", index=False, float_format="%.17g")
        catalog.to_csv(staging / "feature_catalog.csv", index=False)
        np.savez_compressed(staging / "time_series.npz", **time_series)

        family_counts = catalog["family"].value_counts().sort_index().to_dict()
        track_family_counts: dict[str, int] = {}
        for spec in specs.values():
            track_family_counts[spec.family] = track_family_counts.get(spec.family, 0) + 1
        diagnostics = {
            str(item.record.global_seed): item.diagnostics for item in extractions
        }
        time_series_inventory = {
            name: _array_record(value) for name, value in sorted(time_series.items())
        }
        source_inventory = {
            "analysis_source": {
                "path": str(analysis_source),
                "sha256": sha256_file(analysis_source),
            },
            "imported_validation_helper": {
                "path": str(helper_source),
                "sha256": sha256_file(helper_source),
                "imported_contract": "load_validated_trace plus shared reductions/hash helpers",
                "python_bytecode_disabled": True,
            },
            "protocol_canonical_sha256": canonical_sha256(PROTOCOL),
            "ordered_classes": list(classes),
            "ordered_seeds": list(observed_seeds),
            "scientific_fingerprint_sha256": records[0].scientific_fingerprint_sha256,
            "trace_runs": [
                {
                    **asdict(record),
                    "root": str(record.root),
                    "classes": list(record.classes),
                    "posterior_evidence_diagnostics": extraction.diagnostics,
                }
                for record, extraction in zip(records, extractions, strict=True)
            ],
            "time_series_arrays": time_series_inventory,
        }
        atomic_json_dump(source_inventory, staging / "source_inventory.json")

        summary = {
            "schema_version": ANALYSIS_SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "status": "COMPLETE_LABEL_FREE_SUPPLEMENTARY_ANALYSIS",
            "sample_count": len(frame),
            "run_count": len(records),
            "ordered_classes": list(classes),
            "ordered_seeds": list(observed_seeds),
            "track_count": len(tracks),
            "auxiliary_tensor_count": len(auxiliary),
            "time_series_array_count": len(time_series),
            "scalar_feature_count": len(catalog),
            "track_family_counts": {
                name: int(count) for name, count in sorted(track_family_counts.items())
            },
            "scalar_feature_family_counts": {
                str(name): int(count) for name, count in family_counts.items()
            },
            "timing_audit": {
                "predictable_D_rows": STOCHASTIC_STEPS,
                "post_transition_increment_and_log_e_rows": STOCHASTIC_STEPS,
                "masked_t0_random_draw_excluded": True,
                "raw_and_relative_drift_rows": STEPS,
                "sigma_standardized_operational_drift_rows": STOCHASTIC_STEPS,
            },
            "reconstruction_audit": {
                "conditional_logstd_max_abs_tolerance": (
                    LOGSTD_RECONSTRUCTION_TOLERANCE
                ),
                "guided_posterior_mean_max_abs_tolerance": (
                    MEAN_RECONSTRUCTION_TOLERANCE
                ),
                "realized_update_max_abs_tolerance": (
                    TRANSITION_RECONSTRUCTION_TOLERANCE
                ),
                "denoising_identity_max_abs_tolerance": (
                    DENOISING_IDENTITY_TOLERANCE
                ),
                "maximum_conditional_logstd_error": max(
                    value[
                        "conditional_logstd_reconstruction_max_abs_error"
                    ]
                    for value in diagnostics.values()
                ),
                "maximum_guided_posterior_mean_error": max(
                    value[
                        "guided_posterior_mean_reconstruction_max_abs_error"
                    ]
                    for value in diagnostics.values()
                ),
                "maximum_realized_update_replay_error": max(
                    value[
                        "transition_update_reconstruction_max_abs_error"
                    ]
                    for value in diagnostics.values()
                ),
                "maximum_guided_denoising_displacement_identity_error": max(
                    value[
                        "guided_denoising_displacement_identity_max_abs_error"
                    ]
                    for value in diagnostics.values()
                ),
                "all_guided_posterior_means_bitwise_equal": all(
                    value["guided_posterior_mean_bitwise_equal"] == 1.0
                    for value in diagnostics.values()
                ),
                "all_conditional_q_channel4_means_bitwise_equal": all(
                    value["conditional_q_channel4_mean_bitwise_equal"] == 1.0
                    for value in diagnostics.values()
                ),
            },
            "finite_value_audit": {
                "sample_features_all_finite": True,
                "time_series_all_finite": True,
            },
            "weak_Q_covariance_policy": (
                "conditional CFG=1 Q and unconditional Q change mean only and retain "
                "the actual guided P conditional learned-range covariance"
            ),
            "fixed_tile_mixture_policy": (
                "uniform logmeanexp over sixteen fixed 8x8 tile-restricted exact LR "
                "components; componentwise maximum is diagnostic only and is not emitted "
                "as calibrated evidence"
            ),
            "supervision_audit": {
                "labels_read_or_emitted": False,
                "auc_computed": False,
                "threshold_selected": False,
                "trained_combination_fit": False,
            },
            "formula_location": "protocol_snapshot.json and feature_catalog.csv",
        }
        atomic_json_dump(summary, staging / "summary.json")

        payload_files = []
        for path in sorted(staging.iterdir()):
            if path.name in {"manifest.json", "completion.json"}:
                continue
            if not path.is_file() or path.is_symlink():
                raise RuntimeError(f"unexpected staging entry: {path}")
            payload_files.append(
                {
                    "name": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        manifest = {
            "schema_version": ANALYSIS_SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "status": "complete",
            "analysis_source_sha256": sha256_file(staging / "analysis_source.py"),
            "imported_validation_helper_sha256": sha256_file(helper_source),
            "protocol_snapshot_sha256": sha256_file(
                staging / "protocol_snapshot.json"
            ),
            "source_inventory_sha256": sha256_file(staging / "source_inventory.json"),
            "trace_identity_sha256_ordered": [
                record.identity_sha256 for record in records
            ],
            "files": payload_files,
        }
        manifest["identity_sha256"] = canonical_sha256(manifest)
        atomic_json_dump(manifest, staging / "manifest.json")
        completion = {
            "complete": True,
            "manifest_identity_sha256": manifest["identity_sha256"],
            "manifest_file_sha256": sha256_file(staging / "manifest.json"),
            "summary_file_sha256": sha256_file(staging / "summary.json"),
        }
        completion["payload_sha256"] = canonical_sha256(completion)
        atomic_json_dump(completion, staging / "completion.json")
        staging.rename(output)
        validate_analysis_output(output)
        return output
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def validate_analysis_output(output: Path) -> None:
    output = output.expanduser().absolute().resolve()
    if not output.is_dir() or output.is_symlink():
        raise RuntimeError(f"invalid analysis output directory: {output}")
    manifest_path = output / "manifest.json"
    completion_path = output / "completion.json"
    _require_regular(manifest_path, "analysis manifest")
    _require_regular(completion_path, "analysis completion")
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    identity_payload = dict(manifest)
    observed_identity = identity_payload.pop("identity_sha256", None)
    if observed_identity != canonical_sha256(identity_payload):
        raise RuntimeError("analysis manifest identity hash is invalid")
    if manifest.get("experiment") != EXPERIMENT or manifest.get("status") != "complete":
        raise RuntimeError("analysis manifest experiment/status is invalid")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError("analysis manifest file inventory is malformed")
    expected_names: set[str] = set()
    for record in files:
        if not isinstance(record, dict) or not isinstance(record.get("name"), str):
            raise RuntimeError("analysis manifest contains a malformed file record")
        name = str(record["name"])
        if name in expected_names:
            raise RuntimeError(f"duplicate analysis payload record: {name}")
        expected_names.add(name)
        path = _safe_relative(output, name)
        _require_regular(path, "analysis payload")
        if (
            record.get("bytes") != path.stat().st_size
            or record.get("sha256") != sha256_file(path)
        ):
            raise RuntimeError(f"analysis payload size/hash changed: {path}")
    observed_files: set[str] = set()
    for path in output.iterdir():
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"analysis output contains non-regular entry: {path}")
        observed_files.add(path.name)
    if observed_files != expected_names | {"manifest.json", "completion.json"}:
        raise RuntimeError("analysis output member set differs from the manifest")

    completion_payload = dict(completion)
    completion_hash = completion_payload.pop("payload_sha256", None)
    if completion_hash != canonical_sha256(completion_payload):
        raise RuntimeError("analysis completion payload hash is invalid")
    if (
        completion.get("complete") is not True
        or completion.get("manifest_identity_sha256") != observed_identity
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
        or completion.get("summary_file_sha256") != sha256_file(output / "summary.json")
    ):
        raise RuntimeError("analysis completion receipt is invalid")

    protocol = load_json(output / "protocol_snapshot.json")
    inventory = load_json(output / "source_inventory.json")
    summary = load_json(output / "summary.json")
    if canonical_sha256(protocol) != inventory.get("protocol_canonical_sha256"):
        raise RuntimeError("protocol canonical hash differs from source inventory")
    if sha256_file(output / "analysis_source.py") != manifest.get(
        "analysis_source_sha256"
    ):
        raise RuntimeError("analysis source snapshot binding is invalid")
    if inventory.get("imported_validation_helper", {}).get(
        "sha256"
    ) != manifest.get("imported_validation_helper_sha256"):
        raise RuntimeError("imported validation helper binding is invalid")

    frame = pd.read_csv(output / "sample_features.csv")
    catalog = pd.read_csv(output / "feature_catalog.csv")
    if tuple(frame.columns[: len(IDENTIFIER_COLUMNS)]) != IDENTIFIER_COLUMNS:
        raise RuntimeError("sample feature identifiers changed")
    if any("label" in column.lower() for column in frame.columns):
        raise RuntimeError("label-like column found in label-free sample features")
    features = catalog["feature"].astype(str).tolist()
    if len(features) != len(set(features)) or set(features) != set(
        frame.columns[len(IDENTIFIER_COLUMNS) :]
    ):
        raise RuntimeError("feature catalog and sample feature columns differ")
    if not np.isfinite(frame[features].to_numpy(dtype=np.float64)).all():
        raise RuntimeError("published sample features contain non-finite values")
    if (
        len(frame) != summary.get("sample_count")
        or len(catalog) != summary.get("scalar_feature_count")
    ):
        raise RuntimeError("published table counts differ from summary")
    required_catalog_columns = {
        "feature",
        "family",
        "track",
        "track_formula",
        "feature_formula",
        "availability",
        "observation_timing",
        "uses_realized_innovation",
    }
    if not required_catalog_columns.issubset(catalog.columns):
        raise RuntimeError("feature catalog lacks required audit columns")
    if catalog[list(required_catalog_columns)].isna().any().any():
        raise RuntimeError("feature catalog contains missing audit metadata")

    array_inventory = inventory.get("time_series_arrays")
    if not isinstance(array_inventory, dict):
        raise RuntimeError("time-series array inventory is missing")
    with np.load(output / "time_series.npz", allow_pickle=False) as archive:
        if set(archive.files) != set(array_inventory):
            raise RuntimeError("time-series archive member set differs from inventory")
        for name in archive.files:
            value = np.ascontiguousarray(archive[name])
            if not np.isfinite(value).all() or _array_record(value) != array_inventory[name]:
                raise RuntimeError(f"time-series array audit failed: {name}")
        if not np.array_equal(
            archive["sample_index"], frame["sample_index"].to_numpy(np.int32)
        ):
            raise RuntimeError("time-series/sample-table sample_index differs")
        for name in (
            "weak_conditional_cfg1_full_D_nats",
            "weak_unconditional_full_D_nats",
        ):
            if archive[name].shape != (len(frame), STOCHASTIC_STEPS):
                raise RuntimeError(f"predictable D shape is invalid: {name}")
        for q_name in Q_NAMES:
            for suffix in (
                "tile_component_D",
                "tile_component_increment",
                "tile_component_cumulative_log_e",
            ):
                name = f"{q_name}_tile4x4_{suffix}"
                if archive[name].shape != (
                    len(frame),
                    STOCHASTIC_STEPS,
                    TILE_COUNT,
                ):
                    raise RuntimeError(f"fixed-tile component shape is invalid: {name}")
    if summary.get("supervision_audit") != {
        "labels_read_or_emitted": False,
        "auc_computed": False,
        "threshold_selected": False,
        "trained_combination_fit": False,
    }:
        raise RuntimeError("label-free/supervision policy audit changed")


def self_test() -> None:
    rng = np.random.default_rng(917)
    forward_beta = np.linspace(1e-4, 0.02, STEPS, dtype=np.float64)
    forward_alpha_bar = np.cumprod(1.0 - forward_beta)
    reverse_alpha_bar = forward_alpha_bar[::-1].copy()
    schedule = derive_reverse_schedule(reverse_alpha_bar)
    assert np.allclose(schedule.beta[::-1], forward_beta, rtol=2e-13, atol=2e-15)

    # Upstream posterior definitions, independently constructed in forward order.
    forward_previous = np.concatenate((np.ones(1), forward_alpha_bar[:-1]))
    forward_posterior_variance = (
        forward_beta
        * (1.0 - forward_previous)
        / (1.0 - forward_alpha_bar)
    )
    forward_clipped = np.log(
        np.concatenate(
            (forward_posterior_variance[1:2], forward_posterior_variance[1:])
        )
    )
    assert np.allclose(
        schedule.posterior_log_variance_clipped[::-1],
        forward_clipped,
        rtol=2e-13,
        atol=2e-13,
    )

    raw = rng.normal(size=(2, STEPS, CHANNELS, 1, 1)).astype(np.float32)
    # Expand only after creating a compact deterministic fixture.
    raw = np.broadcast_to(raw, (2, STEPS, CHANNELS, LATENT_SIZE, LATENT_SIZE)).copy()
    logstd = learned_range_logstd(raw, schedule)
    raw_minus = np.full_like(raw, -1.0)
    raw_plus = np.full_like(raw, 1.0)
    assert np.allclose(
        learned_range_logstd(raw_minus, schedule)[:, :, 0, 0, 0],
        0.5 * schedule.posterior_log_variance_clipped[None, :],
    )
    assert np.allclose(
        learned_range_logstd(raw_plus, schedule)[:, :, 0, 0, 0],
        0.5 * np.log(schedule.beta)[None, :],
    )
    assert np.isfinite(logstd).all()
    del raw, raw_minus, raw_plus, logstd

    small_shape = (2, STEPS, CHANNELS, 2, 2)
    state = rng.normal(size=small_shape).astype(np.float32)
    epsilon = rng.normal(size=small_shape).astype(np.float32)
    pred = branch_pred_x0(state, epsilon, schedule)
    mean = branch_posterior_mean(state, epsilon, schedule)
    c1 = schedule.posterior_mean_coef1.astype(np.float32)[
        None, :, None, None, None
    ]
    c2 = schedule.posterior_mean_coef2.astype(np.float32)[
        None, :, None, None, None
    ]
    assert np.array_equal(mean, c1 * pred + c2 * state)
    assert np.array_equal(mean[:, -1], pred[:, -1])

    # Exact q/p sign and the fixed spatial partition on a real latent shape.
    batch = 2
    delta = rng.normal(scale=0.01, size=(batch, STEPS, CHANNELS, LATENT_SIZE, LATENT_SIZE))
    sigma = np.exp(
        rng.normal(scale=0.1, size=delta.shape)
    ).astype(np.float32)
    innovation = rng.normal(size=delta.shape).astype(np.float32)
    evidence = exact_same_covariance_evidence(delta, sigma, innovation)
    w0 = delta[0, 0] / sigma[0, 0]
    z0 = innovation[0, 0].astype(np.float64)
    observed = float(evidence["increment"][0, 0])
    direct = float(np.sum(w0 * z0) - 0.5 * np.sum(w0 * w0))
    # Independent diagonal Gaussian density difference log q(x)-log p(x).
    direct_density = float(
        -0.5 * np.sum((z0 - w0) ** 2) + 0.5 * np.sum(z0**2)
    )
    assert math.isclose(observed, direct, rel_tol=2e-14, abs_tol=2e-14)
    assert math.isclose(observed, direct_density, rel_tol=2e-13, abs_tol=2e-13)
    assert np.allclose(
        evidence["tile_component_increment"].sum(axis=2),
        evidence["increment"],
        rtol=3e-13,
        atol=3e-11,
    )
    assert evidence["increment"].shape[1] == STOCHASTIC_STEPS
    assert evidence["tile_component_increment"].shape[2] == TILE_COUNT
    assert np.all(evidence["running_max_log_e"] >= 0.0)
    assert np.all(evidence["tile_mixture_running_max_log_e"] >= 0.0)

    # The masked t=0 draw must be observationally irrelevant.
    changed = innovation.copy()
    changed[:, -1] = np.float32(1e30)
    changed_evidence = exact_same_covariance_evidence(delta, sigma, changed)
    for key in evidence:
        assert np.array_equal(evidence[key], changed_evidence[key])
    print(
        "self-test passed: alpha_bar schedule reconstruction, exact unclipped "
        "learned-range logstd endpoints, posterior mean coefficients, exact Gaussian "
        "q/p LR sign, fixed 4x4 mixture partition, running maxima from E0, and masked-t0 exclusion"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-dir",
        type=Path,
        action="append",
        help="completed custom trace directory; repeat for multiple runs",
    )
    parser.add_argument(
        "--trace-root",
        type=Path,
        action="append",
        help="parent whose immediate children matching --trace-glob are traces",
    )
    parser.add_argument("--trace-glob", default="targeted_scan_v1_seed*")
    parser.add_argument("--expected-classes", type=_parse_csv_ints)
    parser.add_argument("--expected-seeds", type=_parse_csv_ints)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required")
    if not args.trace_dir and not args.trace_root:
        parser.error("at least one --trace-dir or --trace-root is required")
    output = publish(args)
    summary = load_json(output / "summary.json")
    print(
        json.dumps(
            {
                "output": str(output),
                "status": summary["status"],
                "samples": summary["sample_count"],
                "tracks": summary["track_count"],
                "scalar_features": summary["scalar_feature_count"],
                "time_series_arrays": summary["time_series_array_count"],
                "labels": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
