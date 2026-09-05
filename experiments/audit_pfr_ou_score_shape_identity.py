#!/usr/bin/env python3
"""Verify the finite-scale OU relative-score shape identity.

For the OU semigroup with generator ``L = Delta - y dot grad``, let
``r_s = grad log(p_s / phi)`` be the score relative to the standard Gaussian.
The score Fokker--Planck equation implies

    (partial_s + 1) r_s = L r_s + 2 J(r_s) r_s =: C[r_s].

Consequently, for ``delta = s_current - s_future > 0``,

    r_current(y) - exp(-delta) r_future(y)
      = integral_0^delta exp(-(delta-u)) C[r_{future+u}](y) du.

The audit evaluates both sides independently for solvable one-dimensional
Gaussian mixtures.  The integral uses Gauss--Legendre quadrature and the score
derivatives use closed-form Gaussian-mixture density derivatives.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.audit_pfr_ou_conditional_score_identity import (
    GaussianMixture1D,
    bridge_signal,
    default_mixtures,
    relative_score,
    sample_noised_mixture,
)


def relative_score_with_derivatives(
    value: np.ndarray,
    mixture: GaussianMixture1D,
    signal: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the 1D relative score and its first two space derivatives."""

    mixture.validate()
    if not 0.0 < signal < 1.0:
        raise ValueError("signal must lie in (0, 1)")
    values = np.asarray(value, dtype=np.float64)
    means = signal * mixture.means
    variances = signal * signal * mixture.variances + (1.0 - signal * signal)
    expanded = values[..., None]
    component_scores = -(expanded - means) / variances
    logits = (
        np.log(mixture.weights)
        - 0.5 * np.log(2.0 * np.pi * variances)
        - 0.5 * np.square(expanded - means) / variances
    )
    logits -= logits.max(axis=-1, keepdims=True)
    posterior = np.exp(logits)
    posterior /= posterior.sum(axis=-1, keepdims=True)

    density_d1 = np.sum(posterior * component_scores, axis=-1)
    density_d2_ratio = np.sum(
        posterior * (np.square(component_scores) - 1.0 / variances), axis=-1
    )
    density_d3_ratio = np.sum(
        posterior
        * (np.power(component_scores, 3) - 3.0 * component_scores / variances),
        axis=-1,
    )
    density_score_d1 = density_d2_ratio - np.square(density_d1)
    density_score_d2 = (
        density_d3_ratio
        - 3.0 * density_d1 * density_d2_ratio
        + 2.0 * np.power(density_d1, 3)
    )
    return (
        density_d1 + values,
        density_score_d1 + 1.0,
        density_score_d2,
    )


def relative_score_shape_operator(
    value: np.ndarray,
    mixture: GaussianMixture1D,
    signal: float,
) -> np.ndarray:
    """Evaluate ``L r + 2 J(r) r`` for a one-dimensional relative score."""

    score, score_d1, score_d2 = relative_score_with_derivatives(
        value, mixture, signal
    )
    values = np.asarray(value, dtype=np.float64)
    return score_d2 - values * score_d1 + 2.0 * score * score_d1


def integrated_shape_operator(
    value: np.ndarray,
    mixture: GaussianMixture1D,
    *,
    current_signal: float,
    future_signal: float,
    quadrature_order: int,
) -> np.ndarray:
    """Integrate the exact score-shape operator between two OU scales."""

    if not 0.0 < current_signal < future_signal < 1.0:
        raise ValueError("signals must satisfy 0 < current < future < 1")
    if quadrature_order < 8:
        raise ValueError("quadrature_order must be at least eight")
    current_semigroup_time = -math.log(current_signal)
    future_semigroup_time = -math.log(future_signal)
    delta = current_semigroup_time - future_semigroup_time
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    offsets = 0.5 * delta * (nodes + 1.0)
    result = np.zeros_like(np.asarray(value, dtype=np.float64))
    for offset, weight in zip(offsets, weights, strict=True):
        semigroup_time = future_semigroup_time + float(offset)
        operator = relative_score_shape_operator(
            value,
            mixture,
            math.exp(-semigroup_time),
        )
        result += float(weight) * math.exp(-(delta - float(offset))) * operator
    return 0.5 * delta * result


def evaluate_case(
    mixture: GaussianMixture1D,
    *,
    time: float,
    future_time: float,
    samples: int,
    quadrature_order: int,
    rng: np.random.Generator,
) -> dict[str, float | int | str]:
    """Compare the finite difference with the integrated score-FPE operator."""

    current_signal = bridge_signal(time)
    future_signal = bridge_signal(future_time)
    channel_signal = current_signal / future_signal
    values = sample_noised_mixture(
        mixture,
        current_signal,
        samples=samples,
        rng=rng,
    )
    finite_defect = relative_score(values, mixture, current_signal) - (
        channel_signal * relative_score(values, mixture, future_signal)
    )
    integrated = integrated_shape_operator(
        values,
        mixture,
        current_signal=current_signal,
        future_signal=future_signal,
        quadrature_order=quadrature_order,
    )
    residual = finite_defect - integrated

    def rms(value: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(value))))

    denominator = np.linalg.norm(finite_defect) * np.linalg.norm(integrated)
    cosine = (
        float(np.dot(finite_defect, integrated) / denominator)
        if denominator > 1e-24
        else float("nan")
    )
    return {
        "case": mixture.name,
        "time": time,
        "future_time": future_time,
        "samples": samples,
        "quadrature_order": quadrature_order,
        "ou_delta": math.log(future_signal / current_signal),
        "finite_defect_rms": rms(finite_defect),
        "integrated_shape_operator_rms": rms(integrated),
        "identity_residual_rms": rms(residual),
        "identity_residual_max_abs": float(np.max(np.abs(residual))),
        "finite_integrated_cosine": cosine,
    }


def parse_times(value: str) -> tuple[float, ...]:
    try:
        times = tuple(float(item.strip()) for item in value.split(",") if item)
    except ValueError as error:
        raise argparse.ArgumentTypeError("times must be comma-separated floats") from error
    if not times or tuple(sorted(set(times))) != times:
        raise argparse.ArgumentTypeError("times must be unique and increasing")
    return times


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--times",
        type=parse_times,
        default=parse_times("0.02,0.05,0.1,0.2,0.4"),
    )
    parser.add_argument("--horizon", type=float, default=1.0 / 32.0)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--quadrature-order", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260904)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    if args.horizon <= 0.0 or args.samples <= 0:
        raise ValueError("horizon and samples must be positive")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, float | int | str]] = []
    for mixture in default_mixtures():
        for time in args.times:
            future_time = time + args.horizon
            if not 0.0 < time < future_time < 1.0:
                raise ValueError("time and time + horizon must lie in (0, 1)")
            row = evaluate_case(
                mixture,
                time=time,
                future_time=future_time,
                samples=args.samples,
                quadrature_order=args.quadrature_order,
                rng=rng,
            )
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    csv_path = output_dir / "score_shape_identity.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "format": "eqvae_pfr_ou_score_shape_identity_v1",
        "scope": (
            "Analytic one-dimensional Gaussian-mixture audit of the exact "
            "finite-scale score-FPE identity."
        ),
        "times": list(args.times),
        "horizon": args.horizon,
        "samples": args.samples,
        "quadrature_order": args.quadrature_order,
        "seed": args.seed,
        "rows": rows,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
