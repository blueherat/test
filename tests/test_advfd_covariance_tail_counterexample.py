from __future__ import annotations

import pytest
import torch

from experiments.frechet_residual_score_toy import weighted_inner
from experiments.run_advfd_covariance_tail_counterexample import (
    advfd_values,
    analytic_feature_moments,
    exponential_advfd_field,
    gaussian_pair,
    standardized_exponential,
)


def test_standardized_exponential_has_expected_source_moments() -> None:
    concentration = 1.1
    shift = 0.75
    _, source = gaussian_pair(shift, device=torch.device("cpu"))
    states, weights = source.quadrature(order=40)
    normalized = weights / weights.sum()
    feature, _ = standardized_exponential(
        states, concentration, shift=shift, calibration="real"
    )
    observed_mean = (normalized * feature).sum()
    observed_variance = (
        normalized * (feature - observed_mean).square()
    ).sum()
    _, _, expected_mean, expected_std = analytic_feature_moments(
        concentration, shift
    )
    assert float(observed_mean) == pytest.approx(expected_mean, rel=1e-9)
    assert float(observed_variance.sqrt()) == pytest.approx(
        expected_std, rel=1e-9
    )


@pytest.mark.parametrize("calibration", ["real", "pooled"])
def test_analytic_feature_moments_match_quadrature(calibration: str) -> None:
    concentration = 0.8
    shift = 0.75
    target, source = gaussian_pair(shift, device=torch.device("cpu"))
    expected = analytic_feature_moments(
        concentration, shift, calibration=calibration
    )
    observed = []
    for distribution in (target, source):
        states, weights = distribution.quadrature(order=40)
        normalized = weights / weights.sum()
        feature, _ = standardized_exponential(
            states,
            concentration,
            shift=shift,
            calibration=calibration,
        )
        mean = (normalized * feature).sum()
        variance = (normalized * (feature - mean).square()).sum()
        observed.extend((float(mean), float(variance.sqrt())))
    assert observed == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_pooled_calibration_bounds_value_but_not_field_concentration() -> None:
    shift = 0.75
    target, source = gaussian_pair(shift, device=torch.device("cpu"))
    states, weights = source.quadrature(order=64)
    normalized = weights / weights.sum()
    score = torch.full_like(states, -shift)

    real_value = advfd_values(3.0, shift, calibration="real")["advfd_full"]
    pooled_value = advfd_values(3.0, shift, calibration="pooled")[
        "advfd_full"
    ]
    assert real_value > 50.0
    assert pooled_value < 2.1

    pooled_field = exponential_advfd_field(
        3.0, shift, "full", calibration="pooled"
    )(states, False)
    cosine = weighted_inner(pooled_field, score, normalized) / (
        weighted_inner(pooled_field, pooled_field, normalized).sqrt()
        * weighted_inner(score, score, normalized).sqrt()
    )
    assert float(cosine) < 1e-6


def test_full_field_is_sum_of_mean_and_covariance_fields() -> None:
    shift = 0.75
    _, source = gaussian_pair(shift, device=torch.device("cpu"))
    states, _ = source.quadrature(order=16)
    mean = exponential_advfd_field(1.2, shift, "mean")(states, False)
    covariance = exponential_advfd_field(1.2, shift, "covariance")(
        states, False
    )
    full = exponential_advfd_field(1.2, shift, "full")(states, False)
    torch.testing.assert_close(full, mean + covariance)


def test_covariance_objective_grows_while_score_alignment_drops() -> None:
    shift = 0.75
    target, source = gaussian_pair(shift, device=torch.device("cpu"))
    states, weights = source.quadrature(order=48)
    normalized = weights / weights.sum()
    score = torch.full_like(states, -shift)

    def cosine(concentration: float) -> float:
        velocity = exponential_advfd_field(concentration, shift, "full")(
            states, False
        )
        return float(
            weighted_inner(velocity, score, normalized)
            / (
                weighted_inner(velocity, velocity, normalized).sqrt()
                * weighted_inner(score, score, normalized).sqrt()
            )
        )

    low = advfd_values(0.5, shift)["advfd_covariance"]
    high = advfd_values(2.5, shift)["advfd_covariance"]
    assert high > 100 * low
    assert cosine(2.5) < cosine(0.5)
    assert target.dimension == 1
