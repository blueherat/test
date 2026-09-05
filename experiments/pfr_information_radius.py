"""Information-distance calibration for cross-time linear-bridge queries.

Write every linear interpolation in the common data-progress coordinate

    Z_u = u X + (1 - u) E,  E ~ N(0, I),  u: 0 -> 1.

For fixed clean endpoint ``X=x``, the bridge channel at progress ``u`` is
Gaussian.  This module measures a cross-time query by the conditional KL
between two such channels instead of by raw clock distance.  The quantity is
exact for the conditional channels and is an upper bound on the corresponding
marginal KL after mixing over ``X``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ConditionalBridgeKL:
    """Expected conditional KL split into variance and endpoint-mean terms."""

    total: float
    variance: float
    endpoint_mean: float
    per_dimension: float


def expected_conditional_bridge_kl(
    *,
    progress: float,
    query_progress: float,
    dimension: int,
    clean_squared_norm_mean: float,
) -> ConditionalBridgeKL:
    """Return E_X KL(q_u(.|X) || q_v(.|X)) for a linear Gaussian bridge.

    Here ``q_u(.|x) = N(u x, (1-u)^2 I)``.  The expectation over ``X`` only
    requires ``E ||X||^2``.  The direction matters: this function measures a
    state drawn at ``progress`` but presented to a predictor at
    ``query_progress``.
    """

    u = float(progress)
    v = float(query_progress)
    d = int(dimension)
    clean_norm = float(clean_squared_norm_mean)
    if d <= 0:
        raise ValueError("dimension must be positive")
    if not 0.0 <= u < 1.0 or not 0.0 <= v < 1.0:
        raise ValueError("progress values must lie in [0, 1)")
    if clean_norm < 0.0 or not math.isfinite(clean_norm):
        raise ValueError("clean_squared_norm_mean must be finite and non-negative")

    sigma_u = 1.0 - u
    sigma_v = 1.0 - v
    variance_ratio = (sigma_u / sigma_v) ** 2
    variance = 0.5 * d * (
        variance_ratio - 1.0 - math.log(variance_ratio)
    )
    endpoint_mean = 0.5 * (v - u) ** 2 * clean_norm / sigma_v**2
    total = variance + endpoint_mean
    # Roundoff can only produce a tiny negative value at the identity.
    if total < 0.0 and total > -1e-12:
        total = 0.0
    return ConditionalBridgeKL(
        total=total,
        variance=variance,
        endpoint_mean=endpoint_mean,
        per_dimension=total / d,
    )


def conditional_progress_fisher_information(
    *,
    progress: float,
    dimension: int,
    clean_squared_norm_mean: float,
) -> float:
    """Fisher information of ``q_u(.|X)`` with respect to progress ``u``.

    Averaging the Gaussian-channel Fisher information over clean endpoints
    gives ``(2 D + E||X||^2) / (1-u)^2``.  Therefore a fixed small information
    radius requires a raw horizon proportional to ``D^{-1/2}`` when the clean
    per-coordinate energy is held fixed.
    """

    u = float(progress)
    d = int(dimension)
    clean_norm = float(clean_squared_norm_mean)
    if d <= 0:
        raise ValueError("dimension must be positive")
    if not 0.0 <= u < 1.0:
        raise ValueError("progress must lie in [0, 1)")
    if clean_norm < 0.0 or not math.isfinite(clean_norm):
        raise ValueError("clean_squared_norm_mean must be finite and non-negative")
    return (2.0 * d + clean_norm) / (1.0 - u) ** 2


def solve_query_progress_for_kl(
    *,
    progress: float,
    target_kl: float,
    dimension: int,
    clean_squared_norm_mean: float,
    maximum_progress: float = 1.0 - 1e-8,
    iterations: int = 80,
) -> float:
    """Find the unique dataward query progress with a requested conditional KL."""

    u = float(progress)
    target = float(target_kl)
    upper = float(maximum_progress)
    if target < 0.0 or not math.isfinite(target):
        raise ValueError("target_kl must be finite and non-negative")
    if not u <= upper < 1.0:
        raise ValueError("maximum_progress must lie in [progress, 1)")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if target == 0.0:
        return u

    maximum = expected_conditional_bridge_kl(
        progress=u,
        query_progress=upper,
        dimension=dimension,
        clean_squared_norm_mean=clean_squared_norm_mean,
    ).total
    if target > maximum:
        raise ValueError("target KL exceeds the permitted progress interval")

    low = u
    high = upper
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        value = expected_conditional_bridge_kl(
            progress=u,
            query_progress=middle,
            dimension=dimension,
            clean_squared_norm_mean=clean_squared_norm_mean,
        ).total
        if value < target:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)
