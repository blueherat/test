import numpy as np
import pandas as pd
import pytest

from experiments.rae_path_schedule_screen import (
    Schedule,
    delay_retention,
    endpoint_observation_factor,
    floor_power_coefficient,
    floor_rational_coefficient,
    random_control_table,
    screen_schedules,
)


def test_floor_power_factor_matches_closed_form_and_stays_above_floor():
    time = np.linspace(0.0, 1.0, 101)
    schedule = Schedule("floor_power", 0.1, 2.0)
    coefficient, derivative = floor_power_coefficient(time, floor=0.1, power=2.0)
    factor = endpoint_observation_factor(time, schedule)
    expected = 0.1 + 0.9 * (1.0 + 2.0 * time) * (1.0 - time) ** 2
    np.testing.assert_allclose(factor, coefficient - time * (1.0 - time) * derivative)
    np.testing.assert_allclose(factor, expected)
    assert factor.min() == pytest.approx(0.1)


def test_floor_rational_derivative_and_factor_are_positive():
    time = np.linspace(0.0, 1.0, 1001)
    schedule = Schedule("floor_rational", 0.05, 2.0)
    coefficient, derivative = floor_rational_coefficient(time, floor=0.05, alpha=2.0)
    numerical = np.gradient(coefficient, time)
    np.testing.assert_allclose(derivative[1:-1], numerical[1:-1], rtol=2e-5, atol=2e-5)
    assert endpoint_observation_factor(time, schedule).min() == pytest.approx(0.05)


def test_delay_retention_has_expected_power_closed_form():
    assert delay_retention(Schedule("floor_power", 0.0, 2.0)) == pytest.approx(1.0)
    expected = (1.0 - 0.1) * (2.0 / 3.0) / (2.0 / 3.0)
    assert delay_retention(Schedule("floor_power", 0.1, 2.0)) == pytest.approx(
        expected, rel=1e-6
    )


def test_screen_recovers_lower_risk_for_positive_floor():
    time = np.array([0.95, 0.8, 0.5, 0.2])
    current = Schedule("floor_power", 0.0, 2.0)
    current_factor = endpoint_observation_factor(time, current)
    raw = np.full_like(time, 0.1)
    rows = []
    for sample in range(2):
        for index, value in enumerate(time):
            rows.append(
                {
                    "sample_index": sample,
                    "step_index": index,
                    "time": value,
                    "semantic_relative_error": 0.2,
                    "basis_relative_error": raw[index] / current_factor[index],
                    "basis_factor_abs": current_factor[index],
                }
            )
    summary, details = screen_schedules(
        pd.DataFrame(rows),
        [Schedule("floor_power", 0.3, 2.0)],
        semantic_weight=2.0,
        basis_weight=1.0,
    )
    assert len(details) == 8
    assert summary.iloc[0].path_excess_risk_ratio < 0.70
    assert summary.iloc[0].worst_step_excess_risk_ratio < 0.70
    assert bool(summary.iloc[0].passes_gate)


def test_random_controls_make_rank_energy_tradeoff_explicit():
    table = random_control_table(
        channels=768, guided_rank=16, guided_explained_fraction=0.1374007372
    ).set_index("control")
    assert table.loc["old_scaled_rank_matched", "latent_scale"] == pytest.approx(
        2.5681190366
    )
    assert not bool(table.loc["old_scaled_rank_matched", "clean_path_geometry"])
    assert table.loc["same_rank_unscaled", "rank"] == 16
    assert table.loc["energy_rank_unscaled", "rank"] == 106
    assert bool(table.loc["energy_rank_unscaled", "energy_matched"])
