from __future__ import annotations

import numpy as np

from experiments.analyze_rae_spc_path_exposure import (
    detail_coefficients,
    exposure_summary,
    shifted_logit_normal,
)


def test_detail_coefficients_preserve_data_endpoint() -> None:
    state, velocity = detail_coefficients(np.array([0.0, 1.0]))
    np.testing.assert_allclose(state, [1.0, 0.2])
    np.testing.assert_allclose(velocity, [2.6, 0.2])


def test_shifted_time_moves_distribution_toward_noise() -> None:
    base = shifted_logit_normal(10000, seed=7, shift=1.0)
    shifted = shifted_logit_normal(10000, seed=7, shift=48**0.5)
    assert np.median(base) < np.median(shifted)


def test_exposure_summary_reports_both_sides() -> None:
    time = np.array([0.0, 0.5, 1.0])
    state, velocity = detail_coefficients(time)
    summary = exposure_summary(time, state, velocity)
    assert 0 < summary["fraction_velocity_weaker_than_static"] < 1
    assert 0 < summary["fraction_velocity_stronger_than_static"] < 1
