from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from run_semigroup_consistent_guidance_toy import (  # noqa: E402
    GaussianMixture1D,
    conditional_jensen_terms,
    normalize_density,
    soft_bellman_score_tables,
)


def test_conditional_jensen_gradient_is_missing_score() -> None:
    grid = np.linspace(-16.0, 16.0, 4096, endpoint=False)
    weak = GaussianMixture1D((0.45, 0.55), (-2.0, 2.0), (1.0, 0.9))
    strong = GaussianMixture1D((0.30, 0.70), (-1.8, 2.2), (0.6, 0.7))
    weak_clean = normalize_density(weak.density(grid), grid)
    strong_clean = normalize_density(strong.density(grid), grid)
    audit = conditional_jensen_terms(
        grid=grid,
        weak_clean=weak_clean,
        strong_clean=strong_clean,
        beta=2.0,
        heat_variance=1.0,
    )
    mask = audit["target_noisy"] > audit["target_noisy"].max() * 1e-7
    np.testing.assert_allclose(
        audit["correction"][mask],
        audit["delta_gradient"][mask],
            atol=2e-3,
            rtol=2e-3,
    )


def test_power_tilt_and_noise_do_not_commute() -> None:
    grid = np.linspace(-16.0, 16.0, 4096, endpoint=False)
    weak = GaussianMixture1D((0.5, 0.5), (-2.0, 2.0), (1.0, 1.0))
    strong = GaussianMixture1D((0.25, 0.75), (-2.0, 2.0), (0.6, 0.6))
    audit = conditional_jensen_terms(
        grid=grid,
        weak_clean=normalize_density(weak.density(grid), grid),
        strong_clean=normalize_density(strong.density(grid), grid),
        beta=2.0,
        heat_variance=1.0,
    )
    rms = np.sqrt(
        np.trapezoid(
            audit["target_noisy"] * np.square(audit["correction"]), grid
        )
    )
    assert rms > 1e-2


def test_soft_bellman_recovers_exact_semigroup_score_from_scores_only() -> None:
    grid = np.linspace(-12.0, 12.0, 2048, endpoint=False)
    components = {
        "means": (-2.5, 0.0, 2.5),
        "stds": (0.8, 1.0, 0.8),
    }
    weak = GaussianMixture1D((0.35, 0.40, 0.25), **components)
    strong = GaussianMixture1D((0.20, 0.60, 0.20), **components)
    weak_clean = normalize_density(weak.density(grid), grid)
    strong_clean = normalize_density(strong.density(grid), grid)
    beta = 2.0
    taus = np.linspace(0.0, 1.0, 401)

    recovered = soft_bellman_score_tables(
        grid=grid,
        weak=weak,
        strong=strong,
        beta=beta,
        forward_taus=taus,
        max_substep=0.005,
    )[-1]
    exact = conditional_jensen_terms(
        grid=grid,
        weak_clean=weak_clean,
        strong_clean=strong_clean,
        beta=beta,
        heat_variance=1.0,
    )
    weight = exact["target_noisy"]
    error_rms = np.sqrt(
        np.trapezoid(
            weight * np.square(recovered - exact["target_score"]), grid
        )
    )
    exact_rms = np.sqrt(
        np.trapezoid(weight * np.square(exact["target_score"]), grid)
    )
    assert error_rms / exact_rms < 2e-3
