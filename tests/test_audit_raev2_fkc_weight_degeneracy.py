from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.audit_raev2_fkc_weight_degeneracy import (
    clean_gap_to_heat_score_gap_squared_norm,
    effective_sample_size,
    finite_audit_grid,
    fkc_running_potential,
    grouped_weight_statistics,
    heat_variance,
)


def test_clean_gap_maps_to_full_dimensional_heat_score_norm() -> None:
    gap = torch.tensor([[[[1.0, 2.0]]], [[[3.0, 4.0]]]])
    times = torch.tensor([0.5, 0.8])
    actual = clean_gap_to_heat_score_gap_squared_norm(gap, times)
    expected = torch.tensor(
        [
            1.0**2 + 2.0**2,
            (3.0**2 + 4.0**2) * ((1.0 - 0.8) / 0.8) ** 4,
        ]
    )
    torch.testing.assert_close(actual, expected)


def test_running_potential_uses_sum_not_dimension_mean() -> None:
    norm_squared = torch.tensor([2.0, 8.0])
    beta = 1.5
    actual = fkc_running_potential(norm_squared, beta=beta)
    torch.testing.assert_close(actual, 0.5 * beta * (beta - 1.0) * norm_squared)


def test_ess_is_stable_and_detects_particle_collapse() -> None:
    equal = torch.zeros(2, 4)
    torch.testing.assert_close(
        effective_sample_size(equal), torch.full((2,), 4.0, dtype=torch.float64)
    )

    collapsed = torch.tensor([[0.0, -1000.0, -1000.0, -1000.0]])
    assert effective_sample_size(collapsed).item() == pytest.approx(1.0)
    statistics = grouped_weight_statistics(collapsed)
    assert statistics["ess_fraction_mean"] == pytest.approx(0.25)
    assert statistics["max_weight_mean"] == pytest.approx(1.0)


def test_audit_grid_preserves_native_points_and_ends_at_switch() -> None:
    native = torch.tensor([1.0, 0.9, 0.7, 0.4, 0.0])
    actual = finite_audit_grid(native, switch_time=0.5)
    torch.testing.assert_close(actual, torch.tensor([1.0, 0.9, 0.7, 0.5]))


def test_heat_variance_matches_rae_coordinate_and_rejects_endpoints() -> None:
    assert heat_variance(0.5) == pytest.approx(1.0)
    assert heat_variance(0.8) == pytest.approx(16.0)
    with pytest.raises(ValueError):
        heat_variance(1.0)
    with pytest.raises(ValueError):
        heat_variance(0.0)


def test_invalid_inputs_fail_closed() -> None:
    with pytest.raises(ValueError):
        fkc_running_potential(torch.ones(2), beta=0.9)
    with pytest.raises(ValueError):
        clean_gap_to_heat_score_gap_squared_norm(
            torch.ones(2, 3), torch.tensor([0.5])
        )
    with pytest.raises(ValueError):
        effective_sample_size(torch.empty(2, 0))
    with pytest.raises(ValueError):
        grouped_weight_statistics(torch.ones(3))
    assert math.isfinite(heat_variance(0.999))
