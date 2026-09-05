#!/usr/bin/env python3
"""Executable identities behind the counterfactual-residual view of PFR.

This module is deliberately model-free.  It records two facts that distinguish
Projected Future Reference (PFR) from a numerical-integration correction:

1. a finite time difference is a coboundary whose accumulated action is a
   boundary contrast; and
2. local velocity MSE and terminal distribution error need not have the same
   ordering, even for a field with the exact PFR algebra.

The counterexample uses one-dimensional translated Gaussians and the deployed
two-stage guidance schedule, look-ahead clamp, and late strong-only field, so
every number in the emitted summary has a closed form.  It is a logical
counterexample, not a claim that the toy strong and weak fields are
Bayes-optimal predictors.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def discrete_coboundary(values: np.ndarray, lag: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Return both sides of the exact discrete telescoping identity.

    For a sequence ``w_0,...,w_N`` and positive integer ``lag``,

        sum_{k=0}^{N-lag} (w_{k+lag} - w_k)
        = sum_{k=N-lag+1}^{N} w_k - sum_{k=0}^{lag-1} w_k.

    The result can be vector-valued; summation is always over axis zero.
    """

    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 1:
        raise ValueError("values must have a sequence axis")
    if not 1 <= lag < len(array):
        raise ValueError("lag must be in [1, len(values) - 1]")
    bulk = np.sum(array[lag:] - array[:-lag], axis=0)
    boundary = np.sum(array[-lag:], axis=0) - np.sum(array[:lag], axis=0)
    return bulk, boundary


def terminal_mean_witness(
    reference: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, float]:
    """Exact decomposition of a terminal feature-mean improvement.

    With ``e = mu_ref - mu_base`` and ``d = mu_candidate - mu_base``,

        ||e||^2 - ||e-d||^2 = 2 <e,d> - ||d||^2.

    The right side separates useful alignment from intervention energy.  This
    is an endpoint statement and does not inspect local velocity accuracy.
    """

    arrays = [
        np.asarray(value, dtype=np.float64)
        for value in (reference, baseline, candidate)
    ]
    if any(value.ndim != 2 or len(value) < 1 for value in arrays):
        raise ValueError("reference, baseline, and candidate must have shape [N,D]")
    if len({value.shape[1] for value in arrays}) != 1:
        raise ValueError("feature dimensions must agree")
    reference_mean, baseline_mean, candidate_mean = (
        value.mean(axis=0) for value in arrays
    )
    residual = reference_mean - baseline_mean
    shift = candidate_mean - baseline_mean
    residual_norm = float(np.linalg.norm(residual))
    shift_norm = float(np.linalg.norm(shift))
    alignment = float(residual @ shift)
    intervention_energy = shift_norm * shift_norm
    witness = 2.0 * alignment - intervention_energy
    direct = float(
        np.square(reference_mean - baseline_mean).sum()
        - np.square(reference_mean - candidate_mean).sum()
    )
    if not np.isclose(witness, direct, rtol=1e-11, atol=1e-11):
        raise RuntimeError("terminal mean witness identity failed")
    cosine = alignment / max(residual_norm * shift_norm, 1e-30)
    return {
        "baseline_mean_error": residual_norm * residual_norm,
        "candidate_mean_error": float(
            np.square(reference_mean - candidate_mean).sum()
        ),
        "residual_shift_inner_product": alignment,
        "shift_energy": intervention_energy,
        "residual_shift_cosine": cosine,
        "mean_error_improvement": witness,
        "benefit_margin_ratio": 2.0 * alignment / max(intervention_energy, 1e-30),
    }


def legacy_batch_seed_overlap(
    first_seed: int,
    second_seed: int,
    *,
    num_samples: int,
    batch_size: int,
) -> dict[str, Any]:
    """Audit overlap from the historical ``seed + batch_index`` RNG scheme."""

    if num_samples <= 0 or batch_size <= 0:
        raise ValueError("num_samples and batch_size must be positive")

    def seed_to_batch_count(run_seed: int) -> dict[int, int]:
        result: dict[int, int] = {}
        cursor = 0
        batch_index = 0
        while cursor < num_samples:
            count = min(batch_size, num_samples - cursor)
            result[run_seed + batch_index] = count
            cursor += count
            batch_index += 1
        return result

    first = seed_to_batch_count(first_seed)
    second = seed_to_batch_count(second_seed)
    shared = sorted(set(first) & set(second))
    overlapping_samples = sum(min(first[key], second[key]) for key in shared)
    return {
        "rng_scheme": "manual_seed(run_seed + batch_index)",
        "first_run_seed": int(first_seed),
        "second_run_seed": int(second_seed),
        "num_samples_per_run": int(num_samples),
        "batch_size": int(batch_size),
        "batch_count_per_run": len(first),
        "shared_batch_rng_seed_count": len(shared),
        "overlapping_sample_count": int(overlapping_samples),
        "overlapping_sample_fraction": overlapping_samples / num_samples,
        "batch_rng_seed_disjoint": not shared,
    }


def _canonical_toy_response_moments(
    *,
    horizon: float,
    schedule_break: float,
    intervention_end: float,
    beta_first: float,
    beta_second: float,
) -> tuple[float, float]:
    """Integrate ``beta(t) Delta psi(t)`` and its square in closed form.

    Here ``h(t)=min(horizon, intervention_end-t)`` on the active window and
    ``psi(t)=(t-schedule_break)(t-intervention_end)``.  The beta schedule
    changes once at ``schedule_break``.  The restrictions below keep the
    boundary taper wholly inside the second schedule segment.
    """

    if not 0.0 < schedule_break < intervention_end:
        raise ValueError("schedule_break must lie inside the intervention window")
    if not 0.0 < horizon <= intervention_end - schedule_break:
        raise ValueError(
            "horizon must be positive and no longer than the second segment"
        )
    if beta_first <= 0.0 or beta_second <= 0.0:
        raise ValueError("beta values must be positive")
    basis = np.polynomial.Polynomial(
        [
            schedule_break * intervention_end,
            -(schedule_break + intervention_end),
            1.0,
        ]
    )
    shifted_basis = basis(np.polynomial.Polynomial([horizon, 1.0]))
    constant_delta = shifted_basis - basis
    # In the final horizon-wide layer every query is clamped to tau, where
    # psi(tau)=0.
    boundary_delta = -basis

    def integrate(poly: np.polynomial.Polynomial, left: float, right: float) -> float:
        antiderivative = poly.integ()
        return float(antiderivative(right) - antiderivative(left))

    boundary_start = intervention_end - horizon
    first_moment = (
        beta_first * integrate(constant_delta, 0.0, schedule_break)
        + beta_second
        * integrate(constant_delta, schedule_break, boundary_start)
        + beta_second
        * integrate(boundary_delta, boundary_start, intervention_end)
    )
    second_moment = (
        beta_first**2
        * integrate(constant_delta * constant_delta, 0.0, schedule_break)
        + beta_second**2
        * integrate(
            constant_delta * constant_delta,
            schedule_break,
            boundary_start,
        )
        + beta_second**2
        * integrate(
            boundary_delta * boundary_delta,
            boundary_start,
            intervention_end,
        )
    )
    return first_moment, second_moment


def pfr_counterexample_fields(
    time: np.ndarray,
    *,
    horizon: float = 1.0 / 32.0,
    gamma_first: float = 0.6,
    gamma_second: float = 0.7,
    schedule_break: float = 0.25,
    intervention_end: float = 0.5,
) -> dict[str, np.ndarray]:
    """Evaluate a schedule- and clamp-matched PFR counterexample.

    The target field is zero and ordinary guidance is identically one.  The
    weak field is quadratic.  Its scale is chosen so the clamped PFR residual
    on ``[0, intervention_end)`` integrates to minus one; after the
    intervention the strong-only field is again one.  Thus both flows start
    from N(0,1), while ordinary guidance ends at N(1,1) and PFR ends at
    N(0,1).

    State independence makes time-only and projected queries coincide.  The
    construction matches the deployed temporal schedule, boundary clamp, and
    late strong-only switch; it is not a claim about learned SiT fields.
    """

    time = np.asarray(time, dtype=np.float64)
    if time.ndim != 1:
        raise ValueError("time must be one-dimensional")
    if np.any((time < 0.0) | (time > 1.0)):
        raise ValueError("time points must lie in [0, 1]")
    if gamma_first < 0.0 or gamma_second < 0.0:
        raise ValueError("gamma values must be non-negative")
    beta_first = 1.0 + gamma_first
    beta_second = 1.0 + gamma_second
    response_integral, _ = _canonical_toy_response_moments(
        horizon=horizon,
        schedule_break=schedule_break,
        intervention_end=intervention_end,
        beta_first=beta_first,
        beta_second=beta_second,
    )
    if np.isclose(response_integral, 0.0):
        raise ValueError("toy response integral is zero")
    weak_scale = 1.0 / response_integral
    active = time < intervention_end
    gamma = np.where(
        time < schedule_break,
        gamma_first,
        np.where(active, gamma_second, 0.0),
    )
    beta = 1.0 + gamma
    query_horizon = np.where(
        active,
        np.minimum(horizon, np.maximum(intervention_end - time, 0.0)),
        0.0,
    )

    def basis(value: np.ndarray) -> np.ndarray:
        return (value - schedule_break) * (value - intervention_end)

    # The offset makes both W and the derived S continuous at the schedule
    # break and the late strong-only switch.  It cancels from finite
    # differences.
    weak = 1.0 + weak_scale * basis(time)
    weak_query = 1.0 + weak_scale * basis(time + query_horizon)
    guided = np.ones_like(time)
    strong = np.where(
        active,
        (guided + (beta - 1.0) * weak) / beta,
        guided,
    )
    guided_from_heads = np.where(active, weak + beta * (strong - weak), strong)
    pfr = np.where(active, weak + beta * (strong - weak_query), strong)
    expected_pfr = np.where(
        active,
        guided - beta * (weak_query - weak),
        1.0,
    )
    if not np.allclose(guided_from_heads, guided, rtol=1e-11, atol=1e-11):
        raise RuntimeError("analytic ordinary-guidance construction identity failed")
    if not np.allclose(pfr, expected_pfr, rtol=1e-11, atol=1e-11):
        raise RuntimeError("analytic PFR construction identity failed")
    return {
        "target": np.zeros_like(time),
        "beta": beta,
        "query_horizon": query_horizon,
        "weak": weak,
        "strong": strong,
        "guided": guided,
        "weak_query": weak_query,
        "pfr": pfr,
    }


def analytic_counterexample_summary(
    *,
    horizon: float = 1.0 / 32.0,
    gamma_first: float = 0.6,
    gamma_second: float = 0.7,
    schedule_break: float = 0.25,
    intervention_end: float = 0.5,
) -> dict[str, Any]:
    """Return the exact risks and endpoint distances of the Gaussian toy."""

    beta_first = 1.0 + gamma_first
    beta_second = 1.0 + gamma_second
    response_integral, response_squared_integral = _canonical_toy_response_moments(
        horizon=horizon,
        schedule_break=schedule_break,
        intervention_end=intervention_end,
        beta_first=beta_first,
        beta_second=beta_second,
    )
    if np.isclose(response_integral, 0.0):
        raise ValueError("toy response integral is zero")
    weak_scale = 1.0 / response_integral
    pfr_local_mse = response_squared_integral / response_integral**2 - 1.0

    return {
        "format": "eqvae_pfr_counterfactual_residual_theory_v2",
        "parameters": {
            "horizon": float(horizon),
            "gamma_first": float(gamma_first),
            "gamma_second": float(gamma_second),
            "schedule_break": float(schedule_break),
            "intervention_end": float(intervention_end),
            "horizon_rule": "min(H, intervention_end-t)",
            "weak_field": "1+k*(t-schedule_break)*(t-intervention_end)",
            "weak_scale_k": weak_scale,
            "unscaled_response_integral": response_integral,
            "unscaled_squared_response_integral": response_squared_integral,
            "initial_distribution": "N(0,1)",
            "target_distribution": "N(0,1)",
        },
        "ordinary_guidance": {
            "field": "1",
            "integrated_local_velocity_mse": 1.0,
            "terminal_distribution": "N(1,1)",
            "terminal_squared_w2_and_gaussian_fid": 1.0,
        },
        "pfr": {
            "field": (
                "1-beta(t)*k*(psi(t+delta(t))-psi(t)) for "
                "t<intervention_end; 1 otherwise"
            ),
            "formula": (
                "G(t)-beta(t)*(W(t+min(H,intervention_end-t))-W(t)) for "
                "t<intervention_end; "
                "strong(t)=1 otherwise"
            ),
            "integrated_local_velocity_mse": pfr_local_mse,
            "terminal_distribution": "N(0,1)",
            "terminal_squared_w2_and_gaussian_fid": 0.0,
        },
        "exact_conclusion": (
            "Schedule- and clamp-matched PFR strictly worsens time-integrated local "
            "velocity MSE while "
            "eliminating terminal distribution error. This disproves any general monotone "
            "implication from integrated local MSE to terminal risk, and this instance "
            "contains no numerical quadrature error."
        ),
        "scope": (
            "Logical, state-independent specialization of the deployed temporal schedule, "
            "look-ahead clamp, and late strong-only switch; it does not assert that the "
            "constructed strong and weak fields are Bayes predictors."
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main(args: argparse.Namespace) -> None:
    output = args.output_dir.expanduser().resolve()
    # Midpoints avoid assigning either schedule discontinuity to a trapezoid
    # endpoint and integrate the piecewise-polynomial witness accurately.
    time = (np.arange(args.grid_size, dtype=np.float64) + 0.5) / args.grid_size
    fields = pfr_counterexample_fields(time, horizon=args.horizon)
    rows = [
        {"time": float(time[index]), **{key: float(value[index]) for key, value in fields.items()}}
        for index in range(len(time))
    ]
    summary = analytic_counterexample_summary(horizon=args.horizon)
    summary["historical_sampling_seed_audit"] = {
        "fid5k_run_seed_0_vs_1": legacy_batch_seed_overlap(
            0, 1, num_samples=5000, batch_size=8
        ),
        "fid1k_run_seed_0_vs_1": legacy_batch_seed_overlap(
            0, 1, num_samples=1000, batch_size=8
        ),
        "query_control_seed_0_vs_1000003": legacy_batch_seed_overlap(
            0, 1_000_003, num_samples=1000, batch_size=8
        ),
    }
    # Numerical checks are secondary to the closed-form values, but make the
    # emitted artifact independently auditable.
    summary["dense_grid_check"] = {
        "grid_size": args.grid_size,
        "quadrature": "uniform midpoint",
        "ordinary_endpoint_mean": float(np.mean(fields["guided"])),
        "pfr_endpoint_mean": float(np.mean(fields["pfr"])),
        "ordinary_local_velocity_mse": float(np.mean(np.square(fields["guided"]))),
        "pfr_local_velocity_mse": float(np.mean(np.square(fields["pfr"]))),
    }
    _write_csv(output / "analytic_counterexample_fields.csv", rows)
    _atomic_json(output / "analytic_counterexample_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=float, default=1.0 / 32.0)
    parser.add_argument("--grid-size", type=int, default=4096)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
