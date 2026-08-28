#!/usr/bin/env python3
"""Pure numerical helpers for Finite Posterior Cyclic Violation (FPCV).

For an exact Gaussian denoising posterior, the raw class-conditional posterior
mean is the gradient of a convex potential (up to a positive scalar and an
affine term).  Its graph is therefore cyclically monotone.  Given finitely
many input/output pairs, the observed pairing must maximize total inner
product over all output permutations.  FPCV measures the gain available to a
maximum-weight reassignment.

All projection, centering, affinity construction, assignment, and scoring in
this module use float64.  Network evaluation belongs in the separate runner.
"""

from __future__ import annotations

import argparse
import itertools
import math
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment


CROSS_POLYTOPE_DIMENSION = 16
CROSS_POLYTOPE_POINT_COUNT = 2 * CROSS_POLYTOPE_DIMENSION + 1
SMALL_RELATIVE_RADIUS = 1.0 / 64.0
LARGE_RELATIVE_RADIUS = 1.0 / 32.0
EPSILON = 1e-30


def cross_polytope_coordinates(
    dimension: int = CROSS_POLYTOPE_DIMENSION,
    *,
    radius: float = 1.0,
) -> np.ndarray:
    """Return ``{0,+r*e_1,-r*e_1,...,+r*e_d,-r*e_d}`` in fixed order."""

    if dimension <= 0 or not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("cross-polytope dimension and radius must be positive")
    points = np.zeros((2 * dimension + 1, dimension), dtype=np.float64)
    for axis in range(dimension):
        points[1 + 2 * axis, axis] = radius
        points[2 + 2 * axis, axis] = -radius
    return np.ascontiguousarray(points, dtype=np.float64)


def project_outputs(basis: np.ndarray, outputs: np.ndarray) -> np.ndarray:
    """Project ``[n,C,H,W]`` outputs onto an orthonormal ``[r,C,H,W]`` basis."""

    directions = np.asarray(basis, dtype=np.float64)
    values = np.asarray(outputs, dtype=np.float64)
    if directions.ndim != 4 or values.ndim != 4:
        raise ValueError("basis and outputs must have [r,C,H,W] and [n,C,H,W]")
    if values.shape[1:] != directions.shape[1:]:
        raise ValueError("basis and output latent shapes differ")
    if not np.isfinite(directions).all() or not np.isfinite(values).all():
        raise ValueError("basis and outputs must be finite")
    projected = values.reshape(len(values), -1) @ directions.reshape(len(directions), -1).T
    return np.ascontiguousarray(projected, dtype=np.float64)


def brute_force_max_assignment(affinity: np.ndarray) -> tuple[np.ndarray, float]:
    """Exact factorial-time reference solver for small self-test matrices."""

    matrix = np.asarray(affinity, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or matrix.shape[0] == 0
        or matrix.shape[0] > 9
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("brute-force affinity must be finite square with n<=9")
    size = matrix.shape[0]
    rows = np.arange(size)
    best_permutation: tuple[int, ...] | None = None
    best_value = -math.inf
    for permutation in itertools.permutations(range(size)):
        value = float(np.sum(matrix[rows, permutation], dtype=np.float64))
        if value > best_value:
            best_value = value
            best_permutation = permutation
    if best_permutation is None:  # pragma: no cover - size zero is rejected below.
        raise RuntimeError("brute-force assignment produced no result")
    return np.asarray(best_permutation, dtype=np.int64), best_value


def hungarian_max_assignment(affinity: np.ndarray) -> tuple[np.ndarray, float]:
    """Maximize a float64 square affinity matrix with SciPy's Hungarian solver."""

    matrix = np.ascontiguousarray(affinity, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or matrix.shape[0] == 0
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("Hungarian affinity must be one non-empty finite square matrix")
    row_indices, column_indices = linear_sum_assignment(matrix, maximize=True)
    expected = np.arange(matrix.shape[0], dtype=np.int64)
    if not np.array_equal(row_indices, expected):
        raise RuntimeError("Hungarian solver did not return every row in order")
    permutation = np.asarray(column_indices, dtype=np.int64)
    if not np.array_equal(np.sort(permutation), expected):
        raise RuntimeError("Hungarian solver did not return a permutation")
    value = float(np.sum(matrix[expected, permutation], dtype=np.float64))
    return permutation, value


def finite_cyclic_metrics(
    input_points: np.ndarray,
    output_points: np.ndarray,
    *,
    epsilon: float = EPSILON,
) -> dict[str, Any]:
    """Compute finite cyclic-monotonicity violation and its bounded ratio.

    For centered row matrices ``HY`` and ``HZ``, let ``A_id`` be the affinity
    of the observed identity matching and ``A_star`` the maximum affinity over
    all permutations.  The raw violation is ``V=A_star-A_id`` and

    ``D = V / (2*||HY||_F*||HZ||_F + epsilon)``.

    The denominator follows from ``||(P-I)HZ||_F <= 2||HZ||_F`` and therefore
    makes ``D`` lie in ``[0,1]`` up to floating-point tolerance.
    """

    inputs = np.ascontiguousarray(input_points, dtype=np.float64)
    outputs = np.ascontiguousarray(output_points, dtype=np.float64)
    if (
        inputs.ndim != 2
        or outputs.shape != inputs.shape
        or len(inputs) == 0
        or not np.isfinite(inputs).all()
        or not np.isfinite(outputs).all()
        or not math.isfinite(epsilon)
        or epsilon <= 0.0
    ):
        raise ValueError("finite cyclic inputs must be same-shape finite [n,r] arrays")
    centered_inputs = inputs - np.mean(inputs, axis=0, keepdims=True, dtype=np.float64)
    centered_outputs = outputs - np.mean(outputs, axis=0, keepdims=True, dtype=np.float64)
    affinity = np.ascontiguousarray(centered_inputs @ centered_outputs.T, dtype=np.float64)
    identity_affinity = float(np.trace(affinity, dtype=np.float64))
    permutation, optimal_affinity = hungarian_max_assignment(affinity)
    raw_gain = optimal_affinity - identity_affinity
    scale = max(1.0, abs(optimal_affinity), abs(identity_affinity))
    tolerance = 2e-12 * scale
    if raw_gain < -tolerance:
        raise RuntimeError("identity matching exceeded the computed Hungarian optimum")
    violation = float(max(0.0, raw_gain))
    input_norm = float(np.linalg.norm(centered_inputs, ord="fro"))
    output_norm = float(np.linalg.norm(centered_outputs, ord="fro"))
    denominator = 2.0 * input_norm * output_norm + epsilon
    normalized = violation / denominator
    if normalized < -1e-14 or normalized > 1.0 + 2e-12:
        raise RuntimeError("normalized finite cyclic violation left its [0,1] bound")
    return {
        "point_count": int(len(inputs)),
        "projected_dimension": int(inputs.shape[1]),
        "identity_affinity": identity_affinity,
        "optimal_affinity": optimal_affinity,
        "cyclic_violation": violation,
        "centered_input_frobenius_norm": input_norm,
        "centered_output_frobenius_norm": output_norm,
        "normalization_denominator": denominator,
        "normalized_cyclic_violation": float(max(0.0, min(1.0, normalized))),
        "optimal_permutation": [int(value) for value in permutation],
        "identity_is_optimal_within_tolerance": bool(violation <= tolerance),
        "numerical_tolerance": tolerance,
    }


def score_projected_cross_polytope(
    projected_outputs: np.ndarray,
    *,
    absolute_radius: float,
) -> dict[str, Any]:
    """Score projected outputs queried on the fixed 16-D cross-polytope."""

    outputs = np.asarray(projected_outputs, dtype=np.float64)
    expected = (CROSS_POLYTOPE_POINT_COUNT, CROSS_POLYTOPE_DIMENSION)
    if outputs.shape != expected:
        raise ValueError(f"projected cross-polytope outputs must have shape {expected}")
    inputs = cross_polytope_coordinates(radius=absolute_radius)
    return finite_cyclic_metrics(inputs, outputs)


def _assert_hungarian_matches_bruteforce() -> None:
    generator = np.random.default_rng(20260828)
    for size in range(1, 8):
        for _ in range(4):
            affinity = generator.normal(size=(size, size)).astype(np.float64)
            permutation, value = hungarian_max_assignment(affinity)
            brute_permutation, brute_value = brute_force_max_assignment(affinity)
            del permutation, brute_permutation
            assert math.isclose(value, brute_value, rel_tol=2e-14, abs_tol=2e-14)


def self_test() -> None:
    _assert_hungarian_matches_bruteforce()

    coordinates = cross_polytope_coordinates(3, radius=0.7)
    assert coordinates.shape == (7, 3)
    assert np.count_nonzero(coordinates[0]) == 0

    # A symmetric PSD affine map is a convex gradient, hence zero violation.
    factor = np.asarray([[1.0, -0.2, 0.3], [0.4, 0.8, -0.1]], dtype=np.float64)
    positive_semidefinite = factor.T @ factor
    translation = np.asarray([2.0, -4.0, 0.5], dtype=np.float64)
    convex_outputs = coordinates @ positive_semidefinite.T + translation
    convex = finite_cyclic_metrics(coordinates, convex_outputs)
    assert convex["cyclic_violation"] < 2e-14
    assert convex["normalized_cyclic_violation"] < 2e-14

    # Rotation and a negative slope are not gradients of convex functions.
    two_dimensional = cross_polytope_coordinates(2)
    rotation = np.asarray([[0.0, -1.0], [1.0, 0.0]], dtype=np.float64)
    rotated = finite_cyclic_metrics(two_dimensional, two_dimensional @ rotation.T)
    negative = finite_cyclic_metrics(two_dimensional, -two_dimensional)
    assert rotated["normalized_cyclic_violation"] > 0.1
    assert negative["normalized_cyclic_violation"] > 0.1

    # Centering removes arbitrary input/output translations.  A positive
    # output rescaling multiplies V and its denominator by the same amount.
    generator = np.random.default_rng(37)
    inputs = generator.normal(size=(7, 4))
    outputs = generator.normal(size=(7, 4))
    reference = finite_cyclic_metrics(inputs, outputs)
    shifted = finite_cyclic_metrics(
        inputs + np.asarray([8.0, -3.0, 0.2, 4.0]),
        outputs + np.asarray([-1.0, 6.0, 2.0, 9.0]),
    )
    scaled = finite_cyclic_metrics(inputs, 7.25 * outputs)
    assert math.isclose(
        reference["normalized_cyclic_violation"],
        shifted["normalized_cyclic_violation"],
        rel_tol=2e-14,
        abs_tol=2e-14,
    )
    assert math.isclose(
        reference["normalized_cyclic_violation"],
        scaled["normalized_cyclic_violation"],
        rel_tol=2e-14,
        abs_tol=2e-14,
    )

    # Random cases exercise the analytic [0,1] bound.
    for point_count in (2, 5, 11, 33):
        for dimension in (1, 3, 16):
            inputs = generator.normal(size=(point_count, dimension))
            outputs = generator.normal(size=(point_count, dimension))
            score = finite_cyclic_metrics(inputs, outputs)["normalized_cyclic_violation"]
            assert 0.0 <= score <= 1.0

    # Projection itself must occur in float64 even for float32 network output.
    basis = np.zeros((2, 2, 1, 1), dtype=np.float64)
    basis[0, 0, 0, 0] = 1.0
    basis[1, 1, 0, 0] = 1.0
    raw = np.asarray([[[[1.25]], [[-0.5]]]], dtype=np.float32)
    projected = project_outputs(basis, raw)
    assert projected.dtype == np.float64
    assert np.array_equal(projected, np.asarray([[1.25, -0.5]], dtype=np.float64))
    print("self-test passed")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    arguments = parse_args()
    if not arguments.self_test:
        raise SystemExit("nothing to do; pass --self-test")
    self_test()
