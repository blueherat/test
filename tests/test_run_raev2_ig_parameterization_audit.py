from __future__ import annotations

import pytest

from experiments.run_raev2_ig_parameterization_audit import parameterization_scales


def test_parameterization_scales_clean_gap_by_inverse_time_and_step() -> None:
    values = parameterization_scales(
        2.0,
        ig_scale=1.5,
        time=0.2,
        step_size=0.04,
        t_eps=0.05,
        active=True,
    )
    assert values["guided_clean_gap_rms"] == pytest.approx(1.0)
    assert values["velocity_gap_rms"] == pytest.approx(5.0)
    assert values["euler_impulse_rms"] == pytest.approx(0.2)


def test_parameterization_uses_t_floor() -> None:
    values = parameterization_scales(
        1.0,
        ig_scale=2.0,
        time=0.01,
        step_size=0.01,
        t_eps=0.05,
        active=True,
    )
    assert values["velocity_gap_rms"] == pytest.approx(20.0)


def test_parameterization_zeroes_actual_guidance_outside_interval() -> None:
    values = parameterization_scales(
        3.0,
        ig_scale=1.78,
        time=0.07,
        step_size=0.07,
        t_eps=0.05,
        active=False,
    )
    assert values == {
        "guided_clean_gap_rms": 0.0,
        "velocity_gap_rms": 0.0,
        "euler_impulse_rms": 0.0,
    }
