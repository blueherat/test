from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.audit_pfr_ou_conditional_score_identity import (
    bridge_signal,
    default_mixtures,
    relative_score,
)
from experiments.audit_pfr_ou_score_shape_identity import (
    integrated_shape_operator,
    relative_score_shape_operator,
    relative_score_with_derivatives,
)


def test_closed_form_relative_score_derivatives_match_location_family() -> None:
    mixture = default_mixtures()[0]
    signal = 0.4
    values = np.linspace(-3.0, 3.0, 17)

    score, score_d1, score_d2 = relative_score_with_derivatives(
        values, mixture, signal
    )

    np.testing.assert_allclose(score, relative_score(values, mixture, signal))
    np.testing.assert_allclose(score, signal * mixture.means[0], atol=1e-13)
    np.testing.assert_allclose(score_d1, 0.0, atol=1e-13)
    np.testing.assert_allclose(score_d2, 0.0, atol=1e-13)
    np.testing.assert_allclose(
        relative_score_shape_operator(values, mixture, signal),
        0.0,
        atol=1e-13,
    )


@pytest.mark.parametrize("mixture_index", [1, 2, 3])
def test_integrated_shape_operator_equals_finite_degree1_defect(
    mixture_index: int,
) -> None:
    mixture = default_mixtures()[mixture_index]
    current_signal = bridge_signal(0.1)
    future_signal = bridge_signal(0.1 + 1.0 / 32.0)
    values = np.linspace(-4.0, 4.0, 129)
    expected = relative_score(values, mixture, current_signal) - (
        current_signal
        / future_signal
        * relative_score(values, mixture, future_signal)
    )

    actual = integrated_shape_operator(
        values,
        mixture,
        current_signal=current_signal,
        future_signal=future_signal,
        quadrature_order=64,
    )

    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-11)


def test_integrated_shape_operator_rejects_invalid_scales() -> None:
    mixture = default_mixtures()[1]
    with pytest.raises(ValueError, match="signals"):
        integrated_shape_operator(
            np.array([0.0]),
            mixture,
            current_signal=0.5,
            future_signal=0.4,
            quadrature_order=32,
        )
