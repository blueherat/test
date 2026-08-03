from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.analyze_raev2_ig_window_response import (
    endpoint_response_metrics,
    predicted_injected_energy,
    signed_pair_metrics,
)


def test_predicted_energy_uses_gamma_and_window() -> None:
    curve = pd.DataFrame(
        {
            "time": [0.9, 0.5, 0.1],
            "raw_head_gap_rms": [2.0, 3.0, 4.0],
            "h_over_t_safe": [0.1, 0.2, 0.5],
        }
    )
    steps, energy = predicted_injected_energy(
        curve, interval=(0.4, 1.0), gamma=0.5
    )
    assert steps == 2
    assert np.isclose(energy, 0.25 * (0.2**2 + 0.6**2))


def test_endpoint_response_is_zero_for_exact_replay() -> None:
    baseline = np.ones((4, 2, 2), dtype=np.float32)
    result = endpoint_response_metrics(baseline.copy(), baseline)
    assert result["endpoint_delta_rms_mean"] == 0.0


def test_signed_pair_separates_linear_and_even_response() -> None:
    baseline = np.zeros((2, 3), dtype=np.float32)
    positive = np.full_like(baseline, 0.3)
    negative = np.full_like(baseline, -0.1)
    result = signed_pair_metrics(positive, negative, baseline, gamma_abs=0.2)
    assert np.isclose(result["odd_endpoint_rms_mean"], 0.2)
    assert np.isclose(result["even_endpoint_rms_mean"], 0.1)
    assert np.isclose(result["even_over_odd_mean"], 0.5)
    assert np.isclose(result["central_response_per_gamma"], 1.0)
