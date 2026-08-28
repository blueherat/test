#!/usr/bin/env python3
"""Pure numerical helpers for Projected Tweedie-cone Violation (PTCV).

The ideal raw conditional posterior-mean denoiser has a symmetric positive
semidefinite input Jacobian.  This module builds a fixed low/mid-frequency
orthonormal latent basis, projects directional derivatives into that basis,
and measures the exact Frobenius distance to the symmetric PSD cone.
"""

from __future__ import annotations

import argparse
import math
from typing import Any, Iterable, Sequence

import numpy as np


DEFAULT_FREQUENCIES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 0),
    (1, 1),
    (2, 2),
)
LATENT_CHANNELS = 4
LATENT_SIZE = 32
SMALL_RELATIVE_RADIUS = 2.0**-9
LARGE_RELATIVE_RADIUS = 2.0**-8
EPSILON = 1e-30


def dct_vector(size: int, frequency: int) -> np.ndarray:
    """Return one orthonormal DCT-II basis vector in float64."""

    if size <= 0 or not 0 <= frequency < size:
        raise ValueError("invalid DCT size or frequency")
    positions = np.arange(size, dtype=np.float64)
    scale = math.sqrt(1.0 / size) if frequency == 0 else math.sqrt(2.0 / size)
    values = scale * np.cos(math.pi * (2.0 * positions + 1.0) * frequency / (2.0 * size))
    if not math.isclose(float(np.dot(values, values)), 1.0, rel_tol=1e-13, abs_tol=1e-13):
        raise RuntimeError("DCT vector normalization failed")
    return values


def build_channel_dct_basis(
    *,
    channels: int = LATENT_CHANNELS,
    size: int = LATENT_SIZE,
    frequencies: Sequence[tuple[int, int]] = DEFAULT_FREQUENCIES,
) -> tuple[np.ndarray, list[dict[str, int]]]:
    """Build channel-local 2-D DCT directions with shape ``[r,C,H,W]``."""

    if channels <= 0 or not frequencies:
        raise ValueError("basis needs channels and frequencies")
    if len(set(frequencies)) != len(frequencies):
        raise ValueError("DCT frequency list contains duplicates")
    vectors: list[np.ndarray] = []
    metadata: list[dict[str, int]] = []
    for vertical, horizontal in frequencies:
        row = dct_vector(size, vertical)
        column = dct_vector(size, horizontal)
        spatial = np.outer(row, column)
        for channel in range(channels):
            vector = np.zeros((channels, size, size), dtype=np.float64)
            vector[channel] = spatial
            vectors.append(vector)
            metadata.append(
                {
                    "basis_index": len(vectors) - 1,
                    "latent_channel": channel,
                    "vertical_frequency": vertical,
                    "horizontal_frequency": horizontal,
                }
            )
    basis = np.ascontiguousarray(np.stack(vectors, axis=0), dtype=np.float64)
    flat = basis.reshape(len(basis), -1)
    gram = flat @ flat.T
    error = float(np.max(np.abs(gram - np.eye(len(basis), dtype=np.float64))))
    if error > 2e-13:
        raise RuntimeError(f"projected basis is not orthonormal; max_abs={error}")
    return basis, metadata


def build_hadamard_dct_basis(
    *,
    size: int = LATENT_SIZE,
    frequencies: Sequence[tuple[int, int]] = DEFAULT_FREQUENCIES,
) -> tuple[np.ndarray, list[dict[str, int]]]:
    """Build a fixed 4-channel Hadamard x 2-D DCT orthonormal basis."""

    hadamard = 0.5 * np.asarray(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0, 1.0],
        ],
        dtype=np.float64,
    )
    if np.max(np.abs(hadamard @ hadamard.T - np.eye(4))) > 1e-15:
        raise RuntimeError("Hadamard channel basis is not orthonormal")
    if not frequencies or len(set(frequencies)) != len(frequencies):
        raise ValueError("DCT frequency list must be non-empty and unique")
    vectors: list[np.ndarray] = []
    metadata: list[dict[str, int]] = []
    for vertical, horizontal in frequencies:
        spatial = np.outer(
            dct_vector(size, vertical), dct_vector(size, horizontal)
        )
        for channel_mode, channel_vector in enumerate(hadamard):
            vectors.append(channel_vector[:, None, None] * spatial[None, :, :])
            metadata.append(
                {
                    "basis_index": len(vectors) - 1,
                    "hadamard_channel_mode": channel_mode,
                    "vertical_frequency": vertical,
                    "horizontal_frequency": horizontal,
                }
            )
    basis = np.ascontiguousarray(np.stack(vectors, axis=0), dtype=np.float64)
    flat = basis.reshape(len(basis), -1)
    error = float(np.max(np.abs(flat @ flat.T - np.eye(len(basis)))))
    if error > 2e-13:
        raise RuntimeError(f"Hadamard-DCT basis is not orthonormal; max_abs={error}")
    return basis, metadata


def projected_matrix(basis: np.ndarray, directional_derivatives: np.ndarray) -> np.ndarray:
    """Return ``B[a,b] = <q_a, J q_b>`` from directional derivatives."""

    directions = np.asarray(basis, dtype=np.float64)
    derivatives = np.asarray(directional_derivatives, dtype=np.float64)
    if directions.ndim != 4 or derivatives.shape != directions.shape:
        raise ValueError("basis and directional derivatives must share [r,C,H,W]")
    if not np.isfinite(directions).all() or not np.isfinite(derivatives).all():
        raise ValueError("basis and directional derivatives must be finite")
    flat_basis = directions.reshape(len(directions), -1)
    flat_derivatives = derivatives.reshape(len(derivatives), -1)
    return np.ascontiguousarray(flat_basis @ flat_derivatives.T, dtype=np.float64)


def richardson_matrix(small_radius: np.ndarray, large_radius: np.ndarray) -> np.ndarray:
    """Cancel the O(h^2) error of centered differences at h and 2h."""

    small = np.asarray(small_radius, dtype=np.float64)
    large = np.asarray(large_radius, dtype=np.float64)
    if small.ndim != 2 or small.shape[0] != small.shape[1] or large.shape != small.shape:
        raise ValueError("Richardson inputs must be square matrices with one shape")
    if not np.isfinite(small).all() or not np.isfinite(large).all():
        raise ValueError("Richardson inputs must be finite")
    return np.ascontiguousarray((4.0 * small - large) / 3.0, dtype=np.float64)


def cone_metrics(matrix: np.ndarray) -> dict[str, Any]:
    """Measure the normalized squared distance to the symmetric PSD cone."""

    value = np.asarray(matrix, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != value.shape[1] or not np.isfinite(value).all():
        raise ValueError("cone input must be one finite square matrix")
    symmetric = 0.5 * (value + value.T)
    skew = 0.5 * (value - value.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    negative = np.minimum(eigenvalues, 0.0)
    skew_energy = float(np.einsum("ij,ij->", skew, skew, dtype=np.float64))
    negative_energy = float(np.dot(negative, negative))
    total_energy = float(np.einsum("ij,ij->", value, value, dtype=np.float64))
    distance_squared = skew_energy + negative_energy
    normalized = distance_squared / max(total_energy, EPSILON)
    if distance_squared < -1e-12 or not -1e-12 <= normalized <= 1.0 + 1e-10:
        raise RuntimeError("PSD-cone distance violated its algebraic bounds")
    positive = np.maximum(eigenvalues, 0.0)
    projected = (eigenvectors * positive) @ eigenvectors.T
    direct_distance = float(np.sum((value - projected) ** 2, dtype=np.float64))
    if not math.isclose(
        direct_distance, distance_squared, rel_tol=2e-10, abs_tol=2e-12
    ):
        raise RuntimeError("PSD-cone projection identity failed")
    return {
        "dimension": int(value.shape[0]),
        "cone_distance_squared": distance_squared,
        "matrix_energy": total_energy,
        "normalized_cone_violation": float(max(0.0, min(1.0, normalized))),
        "skew_energy": skew_energy,
        "negative_eigen_energy": negative_energy,
        "skew_fraction": float(skew_energy / max(total_energy, EPSILON)),
        "negative_eigen_fraction": float(negative_energy / max(total_energy, EPSILON)),
        "minimum_symmetric_eigenvalue": float(eigenvalues[0]),
        "maximum_symmetric_eigenvalue": float(eigenvalues[-1]),
        "negative_eigenvalue_count": int(np.count_nonzero(eigenvalues < 0.0)),
        "symmetric_eigenvalues": [float(item) for item in eigenvalues],
    }


def finite_difference_stability(
    small: np.ndarray, large: np.ndarray, extrapolated: np.ndarray
) -> dict[str, float]:
    """Return label-free numerical stability controls for the two radii."""

    first = np.asarray(small, dtype=np.float64)
    second = np.asarray(large, dtype=np.float64)
    final = np.asarray(extrapolated, dtype=np.float64)
    if first.shape != second.shape or first.shape != final.shape:
        raise ValueError("stability matrices must have one shape")
    difference = float(np.linalg.norm(first - second, ord="fro"))
    final_norm = float(np.linalg.norm(final, ord="fro"))
    mean_input_norm = 0.5 * (
        float(np.linalg.norm(first, ord="fro"))
        + float(np.linalg.norm(second, ord="fro"))
    )
    return {
        "small_large_frobenius_difference": difference,
        "difference_over_richardson_norm": difference / max(final_norm, EPSILON),
        "difference_over_mean_input_norm": difference / max(mean_input_norm, EPSILON),
    }


def self_test() -> None:
    basis, metadata = build_hadamard_dct_basis()
    assert basis.shape == (16, 4, 32, 32) and len(metadata) == 16
    flat = basis.reshape(len(basis), -1)
    assert np.max(np.abs(flat @ flat.T - np.eye(len(basis)))) < 2e-13

    diagonal = np.diag([2.0, 1.0, 0.0])
    valid = cone_metrics(diagonal)
    assert valid["normalized_cone_violation"] < 1e-15

    indefinite = np.diag([2.0, -1.0])
    broken = cone_metrics(indefinite)
    assert math.isclose(broken["normalized_cone_violation"], 0.2)
    assert broken["negative_eigenvalue_count"] == 1

    rotation = np.asarray([[0.0, 1.0], [-1.0, 0.0]])
    curl = cone_metrics(rotation)
    assert math.isclose(curl["normalized_cone_violation"], 1.0)

    random = np.random.default_rng(20260828).normal(size=(5, 5))
    psd = random @ random.T
    assert cone_metrics(psd)["normalized_cone_violation"] < 1e-24

    # For m(x)=c*x, the projected Jacobian is cI and must be admissible.
    derivatives = 0.7 * basis
    projected = projected_matrix(basis, derivatives)
    assert np.max(np.abs(projected - 0.7 * np.eye(len(basis)))) < 2e-13
    assert cone_metrics(projected)["normalized_cone_violation"] < 1e-24

    base = np.asarray([[1.0, 0.2], [0.2, 0.5]])
    error = np.asarray([[0.1, -0.3], [0.2, 0.4]])
    h = 0.01
    at_h = base + h * h * error
    at_2h = base + 4.0 * h * h * error
    assert np.max(np.abs(richardson_matrix(at_h, at_2h) - base)) < 1e-15
    print("self-test passed")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    arguments = parse_args()
    if not arguments.self_test:
        raise SystemExit("this helper only exposes --self-test")
    self_test()
