#!/usr/bin/env python3
"""Exact toy audit for EMA moments in a moving feature coordinate system.

The underlying real/fake Gaussian pair is fixed.  Only an equivalent
orthogonal feature parameterization rotates over training time.  A correctly
frame-consistent distance is therefore constant.  Directly averaging moments
from different feature frames need not preserve that invariance.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--betas", type=float, nargs="+", default=(0.9, 0.99, 0.999))
    parser.add_argument(
        "--degrees-per-step",
        type=float,
        nargs="+",
        default=(0.0, 0.1, 0.5, 1.0, 5.0, 15.0, 45.0, 90.0),
    )
    parser.add_argument("--anisotropy", type=float, default=0.8)
    parser.add_argument("--mean-offset", type=float, default=1.0)
    parser.add_argument("--whiten-eps", type=float, default=1e-3)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def rotation(angle: float) -> np.ndarray:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float64)


def real_whitened_fd(
    real_mean: np.ndarray,
    real_covariance: np.ndarray,
    fake_mean: np.ndarray,
    fake_covariance: np.ndarray,
    *,
    epsilon: float,
) -> tuple[float, float, float]:
    """AdvFD's epsilon-regularized real-whitened Gaussian objective."""

    dimension = real_mean.size
    identity = np.eye(dimension, dtype=np.float64)
    real_regularized = 0.5 * (real_covariance + real_covariance.T) + epsilon * identity
    real_eigenvalues, real_eigenvectors = np.linalg.eigh(real_regularized)
    inverse_roots = np.maximum(real_eigenvalues, epsilon) ** -0.5
    mean_white = ((fake_mean - real_mean) @ real_eigenvectors) * inverse_roots

    fake_regularized = 0.5 * (fake_covariance + fake_covariance.T) + epsilon * identity
    fake_in_real_basis = real_eigenvectors.T @ fake_regularized @ real_eigenvectors
    fake_white = (
        fake_in_real_basis
        * inverse_roots[:, None]
        * inverse_roots[None, :]
    )
    fake_white = 0.5 * (fake_white + fake_white.T)
    generalized_eigenvalues = np.maximum(np.linalg.eigvalsh(fake_white), 0.0)

    mean_term = float(np.square(mean_white).sum())
    covariance_term = float(np.square(np.sqrt(generalized_eigenvalues) - 1.0).sum())
    return mean_term + covariance_term, mean_term, covariance_term


def transform_moments(
    mean: np.ndarray,
    covariance: np.ndarray,
    transform: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return transform @ mean, transform @ covariance @ transform.T


def update_moments(
    old_mean: np.ndarray,
    old_covariance: np.ndarray,
    current_mean: np.ndarray,
    current_covariance: np.ndarray,
    *,
    beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """EMA uncentered moments, matching AdvFD's FeatureStatsEMA."""

    old_second = old_covariance + np.outer(old_mean, old_mean)
    current_second = current_covariance + np.outer(current_mean, current_mean)
    mean = beta * old_mean + (1.0 - beta) * current_mean
    second = beta * old_second + (1.0 - beta) * current_second
    covariance = second - np.outer(mean, mean)
    covariance = 0.5 * (covariance + covariance.T)
    return mean, covariance


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "first": float(array[0]),
        "last": float(array[-1]),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
        "relative_range": float((array.max() - array.min()) / max(abs(array[0]), 1e-30)),
    }


def run_condition(
    *,
    beta: float,
    degrees_per_step: float,
    steps: int,
    anisotropy: float,
    mean_offset: float,
    epsilon: float,
) -> tuple[dict, list[dict]]:
    if not 0.0 <= beta < 1.0:
        raise ValueError("beta must be in [0, 1)")
    if not 0.0 <= anisotropy < 1.0:
        raise ValueError("anisotropy must be in [0, 1)")
    if steps <= 0:
        raise ValueError("steps must be positive")

    real_mean_base = np.zeros(2, dtype=np.float64)
    real_covariance_base = np.eye(2, dtype=np.float64)
    fake_mean_base = np.asarray((mean_offset, 0.0), dtype=np.float64)
    fake_covariance_base = np.diag((1.0 + anisotropy, 1.0 - anisotropy))

    angle_step = math.radians(degrees_per_step)
    initial_rotation = rotation(0.0)
    real_mean, real_covariance = transform_moments(
        real_mean_base, real_covariance_base, initial_rotation
    )
    fake_mean, fake_covariance = transform_moments(
        fake_mean_base, fake_covariance_base, initial_rotation
    )
    naive_real_mean = real_mean.copy()
    naive_real_covariance = real_covariance.copy()
    naive_fake_mean = fake_mean.copy()
    naive_fake_covariance = fake_covariance.copy()
    transported_real_mean = real_mean.copy()
    transported_real_covariance = real_covariance.copy()
    transported_fake_mean = fake_mean.copy()
    transported_fake_covariance = fake_covariance.copy()

    rows: list[dict] = []
    for step in range(steps + 1):
        angle = step * angle_step
        current_rotation = rotation(angle)
        real_mean, real_covariance = transform_moments(
            real_mean_base, real_covariance_base, current_rotation
        )
        fake_mean, fake_covariance = transform_moments(
            fake_mean_base, fake_covariance_base, current_rotation
        )

        if step > 0:
            naive_real_mean, naive_real_covariance = update_moments(
                naive_real_mean,
                naive_real_covariance,
                real_mean,
                real_covariance,
                beta=beta,
            )
            naive_fake_mean, naive_fake_covariance = update_moments(
                naive_fake_mean,
                naive_fake_covariance,
                fake_mean,
                fake_covariance,
                beta=beta,
            )

            frame_transport = rotation(angle_step)
            transported_real_mean, transported_real_covariance = transform_moments(
                transported_real_mean,
                transported_real_covariance,
                frame_transport,
            )
            transported_fake_mean, transported_fake_covariance = transform_moments(
                transported_fake_mean,
                transported_fake_covariance,
                frame_transport,
            )
            transported_real_mean, transported_real_covariance = update_moments(
                transported_real_mean,
                transported_real_covariance,
                real_mean,
                real_covariance,
                beta=beta,
            )
            transported_fake_mean, transported_fake_covariance = update_moments(
                transported_fake_mean,
                transported_fake_covariance,
                fake_mean,
                fake_covariance,
                beta=beta,
            )

        current_fd, current_mean_fd, current_covariance_fd = real_whitened_fd(
            real_mean,
            real_covariance,
            fake_mean,
            fake_covariance,
            epsilon=epsilon,
        )
        naive_fd, naive_mean_fd, naive_covariance_fd = real_whitened_fd(
            naive_real_mean,
            naive_real_covariance,
            naive_fake_mean,
            naive_fake_covariance,
            epsilon=epsilon,
        )
        transported_fd, transported_mean_fd, transported_covariance_fd = real_whitened_fd(
            transported_real_mean,
            transported_real_covariance,
            transported_fake_mean,
            transported_fake_covariance,
            epsilon=epsilon,
        )
        rows.append(
            {
                "step": step,
                "angle_degrees": math.degrees(angle),
                "current_fd": current_fd,
                "current_mean_term": current_mean_fd,
                "current_covariance_term": current_covariance_fd,
                "naive_ema_fd": naive_fd,
                "naive_ema_mean_term": naive_mean_fd,
                "naive_ema_covariance_term": naive_covariance_fd,
                "transported_ema_fd": transported_fd,
                "transported_ema_mean_term": transported_mean_fd,
                "transported_ema_covariance_term": transported_covariance_fd,
                "naive_to_current_ratio": naive_fd / current_fd,
                "transported_to_current_ratio": transported_fd / current_fd,
            }
        )

    current_values = [row["current_fd"] for row in rows]
    naive_values = [row["naive_ema_fd"] for row in rows]
    transported_values = [row["transported_ema_fd"] for row in rows]
    summary = {
        "beta": beta,
        "degrees_per_step": degrees_per_step,
        "steps": steps,
        "current_fd": summarize(current_values),
        "naive_ema_fd": summarize(naive_values),
        "transported_ema_fd": summarize(transported_values),
        "naive_final_to_current": float(naive_values[-1] / current_values[-1]),
        "transported_final_to_current": float(
            transported_values[-1] / current_values[-1]
        ),
        "naive_max_relative_error": float(
            np.max(
                np.abs(np.asarray(naive_values) - np.asarray(current_values))
                / np.maximum(np.abs(current_values), 1e-30)
            )
        ),
        "transported_max_relative_error": float(
            np.max(
                np.abs(np.asarray(transported_values) - np.asarray(current_values))
                / np.maximum(np.abs(current_values), 1e-30)
            )
        ),
    }
    return summary, rows


def main() -> None:
    args = parse_args()
    conditions = []
    csv_rows = []
    for beta in args.betas:
        for degrees_per_step in args.degrees_per_step:
            summary, rows = run_condition(
                beta=beta,
                degrees_per_step=degrees_per_step,
                steps=args.steps,
                anisotropy=args.anisotropy,
                mean_offset=args.mean_offset,
                epsilon=args.whiten_eps,
            )
            conditions.append(summary)
            for row in rows:
                csv_rows.append(
                    {
                        "beta": beta,
                        "degrees_per_step": degrees_per_step,
                        **row,
                    }
                )

    payload = {
        "protocol": "advfd_rotating_gaussian_frame_audit_v1",
        "premise": (
            "The underlying Gaussian pair is stationary and every feature-map "
            "change is a shared orthogonal reparameterization. Current-frame "
            "AdvFD must therefore remain invariant."
        ),
        "steps": args.steps,
        "betas": args.betas,
        "degrees_per_step": args.degrees_per_step,
        "anisotropy": args.anisotropy,
        "mean_offset": args.mean_offset,
        "whiten_epsilon": args.whiten_eps,
        "conditions": conditions,
        "interpretation_boundary": (
            "The toy proves non-invariance of direct cross-frame moment EMA "
            "under equivalent feature rotations. It does not by itself prove "
            "that the same effect harms generator quality in AdvFD training."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fieldnames = list(csv_rows[0])
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
