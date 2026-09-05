#!/usr/bin/env python3
"""Validate the exact OU score identity and the degree-1 null family.

The audit uses one-dimensional Gaussian mixtures, whose noised relative scores
and cleaner-state posterior are both available analytically. Gauss--Hermite
quadrature evaluates the only nonlinear conditional expectation. This keeps
the check independent of a neural network and of any image-quality metric.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class GaussianMixture1D:
    name: str
    weights: np.ndarray
    means: np.ndarray
    variances: np.ndarray

    def validate(self) -> None:
        if not (
            self.weights.ndim
            == self.means.ndim
            == self.variances.ndim
            == 1
        ):
            raise ValueError("mixture parameters must be one-dimensional")
        if not (
            len(self.weights) == len(self.means) == len(self.variances)
        ):
            raise ValueError("mixture parameters must have equal lengths")
        if np.any(self.weights <= 0.0) or np.any(self.variances <= 0.0):
            raise ValueError("weights and variances must be positive")
        if not np.isclose(self.weights.sum(), 1.0):
            raise ValueError("weights must sum to one")


def bridge_signal(time: float) -> float:
    if not 0.0 < time < 1.0:
        raise ValueError("time must lie in (0, 1)")
    scale = math.sqrt(time * time + (1.0 - time) ** 2)
    return time / scale


def noised_component_parameters(
    mixture: GaussianMixture1D,
    signal: float,
) -> tuple[np.ndarray, np.ndarray]:
    means = signal * mixture.means
    variances = signal * signal * mixture.variances + (1.0 - signal * signal)
    return means, variances


def component_posterior(
    value: np.ndarray,
    mixture: GaussianMixture1D,
    signal: float,
) -> np.ndarray:
    means, variances = noised_component_parameters(mixture, signal)
    expanded = np.asarray(value, dtype=np.float64)[..., None]
    logits = (
        np.log(mixture.weights)
        - 0.5 * np.log(2.0 * np.pi * variances)
        - 0.5 * (expanded - means) ** 2 / variances
    )
    logits -= logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=-1, keepdims=True)


def relative_score(
    value: np.ndarray,
    mixture: GaussianMixture1D,
    signal: float,
) -> np.ndarray:
    """Return score(mu_signal) minus the standard-Gaussian score."""

    means, variances = noised_component_parameters(mixture, signal)
    expanded = np.asarray(value, dtype=np.float64)[..., None]
    posterior = component_posterior(expanded[..., 0], mixture, signal)
    component_scores = -(expanded - means) / variances
    return np.sum(posterior * component_scores, axis=-1) + expanded[..., 0]


def conditional_future_parameters(
    current_value: np.ndarray,
    mixture: GaussianMixture1D,
    current_signal: float,
    future_signal: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return p(component, Y_future | Y_current) parameters."""

    if not 0.0 < current_signal < future_signal < 1.0:
        raise ValueError("signals must satisfy 0 < current < future < 1")
    channel_signal = current_signal / future_signal
    future_means, future_variances = noised_component_parameters(
        mixture, future_signal
    )
    current_means, current_variances = noised_component_parameters(
        mixture, current_signal
    )
    expanded = np.asarray(current_value, dtype=np.float64)[..., None]
    gain = channel_signal * future_variances / current_variances
    conditional_means = future_means + gain * (
        expanded - current_means
    )
    conditional_variances = future_variances - (
        channel_signal * channel_signal
        * np.square(future_variances)
        / current_variances
    )
    posterior = component_posterior(
        expanded[..., 0], mixture, current_signal
    )
    return posterior, conditional_means, conditional_variances


def expected_future_relative_score(
    current_value: np.ndarray,
    mixture: GaussianMixture1D,
    current_signal: float,
    future_signal: float,
    *,
    quadrature_order: int,
) -> np.ndarray:
    """Evaluate E[r_future(Y_future) | Y_current] by quadrature."""

    if quadrature_order < 8:
        raise ValueError("quadrature_order must be at least eight")
    posterior, means, variances = conditional_future_parameters(
        current_value,
        mixture,
        current_signal,
        future_signal,
    )
    nodes, weights = np.polynomial.hermite.hermgauss(quadrature_order)
    queries = means[..., None] + np.sqrt(2.0 * variances[..., None]) * nodes
    scores = relative_score(queries, mixture, future_signal)
    component_expectations = np.sum(scores * weights, axis=-1) / math.sqrt(
        math.pi
    )
    return np.sum(posterior * component_expectations, axis=-1)


def sample_noised_mixture(
    mixture: GaussianMixture1D,
    signal: float,
    *,
    samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if samples <= 0:
        raise ValueError("samples must be positive")
    means, variances = noised_component_parameters(mixture, signal)
    components = rng.choice(len(mixture.weights), size=samples, p=mixture.weights)
    return rng.normal(means[components], np.sqrt(variances[components]))


def default_mixtures() -> tuple[GaussianMixture1D, ...]:
    return (
        GaussianMixture1D(
            "location",
            np.array([1.0]),
            np.array([1.5]),
            np.array([1.0]),
        ),
        GaussianMixture1D(
            "covariance",
            np.array([1.0]),
            np.array([0.0]),
            np.array([2.5]),
        ),
        GaussianMixture1D(
            "symmetric_bimodal",
            np.array([0.5, 0.5]),
            np.array([-2.0, 2.0]),
            np.array([0.25, 0.25]),
        ),
        GaussianMixture1D(
            "skewed_mixture",
            np.array([0.35, 0.65]),
            np.array([-1.5, 0.75]),
            np.array([0.3, 1.2]),
        ),
    )


def evaluate_case(
    mixture: GaussianMixture1D,
    *,
    time: float,
    future_time: float,
    samples: int,
    quadrature_order: int,
    rng: np.random.Generator,
) -> dict[str, float | int | str]:
    mixture.validate()
    current_signal = bridge_signal(time)
    future_signal = bridge_signal(future_time)
    channel_signal = current_signal / future_signal
    current_value = sample_noised_mixture(
        mixture,
        current_signal,
        samples=samples,
        rng=rng,
    )
    current_score = relative_score(current_value, mixture, current_signal)
    future_score_fixed = relative_score(
        current_value, mixture, future_signal
    )
    expected_future_score = expected_future_relative_score(
        current_value,
        mixture,
        current_signal,
        future_signal,
        quadrature_order=quadrature_order,
    )
    identity_residual = current_score - channel_signal * expected_future_score
    degree1_defect = current_score - channel_signal * future_score_fixed
    degree2_defect = current_score - channel_signal**2 * future_score_fixed

    def rms(value: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(value))))

    return {
        "case": mixture.name,
        "time": time,
        "future_time": future_time,
        "samples": samples,
        "quadrature_order": quadrature_order,
        "current_signal": current_signal,
        "future_signal": future_signal,
        "channel_signal": channel_signal,
        "current_score_rms": rms(current_score),
        "conditional_identity_residual_rms": rms(identity_residual),
        "conditional_identity_residual_max_abs": float(
            np.max(np.abs(identity_residual))
        ),
        "degree1_fixed_coordinate_defect_rms": rms(degree1_defect),
        "degree2_fixed_coordinate_defect_rms": rms(degree2_defect),
        "normalized_information_innovation_rms": rms(
            degree1_defect / current_signal
        ),
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
    parser.add_argument("--times", type=parse_times, default=parse_times("0.02,0.05,0.1,0.2,0.4"))
    parser.add_argument("--horizon", type=float, default=1.0 / 32.0)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--quadrature-order", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260904)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    if args.horizon <= 0.0:
        raise ValueError("horizon must be positive")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, float | int | str]] = []
    for mixture in default_mixtures():
        for time in args.times:
            future_time = time + args.horizon
            if future_time >= 1.0:
                raise ValueError("time + horizon must be below one")
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

    csv_path = output_dir / "conditional_score_identity.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "format": "eqvae_pfr_ou_conditional_score_identity_v1",
        "scope": (
            "Analytic one-dimensional Gaussian-mixture audit; no neural model "
            "and no image-quality metric."
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
