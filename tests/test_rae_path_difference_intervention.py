from __future__ import annotations

import pandas as pd
import pytest
import torch

from experiments.rae_cycle_direction_intervention import sample_rms
from experiments.rae_path_difference_intervention import (
    PATH_PAIRS,
    component_energy_fraction,
    feature_progress,
    fit_global_direction,
    matched_path_directions,
    path_difference_gate,
    projected_frechet_distance,
    random_unit_directions,
    rms_preserving_lerp,
    spherical_interpolate,
    spatial_components,
    standardized_sliced_wasserstein,
)


def test_matched_path_controls_have_equal_sample_rms() -> None:
    delta = torch.randn(8, 5, 4, 4)
    global_direction = fit_global_direction(torch.randn(12, 5, 4, 4))
    directions = matched_path_directions(delta, global_direction, seed=7)
    expected = sample_rms(delta)
    for value in directions.values():
        torch.testing.assert_close(sample_rms(value), expected)
    torch.testing.assert_close(directions["opposite"], -delta)


def test_spatial_components_reconstruct_and_are_orthogonal() -> None:
    delta = torch.randn(7, 6, 5, 5)
    components = spatial_components(delta)
    torch.testing.assert_close(
        components["token_mean"] + components["spatial_residual"], delta
    )
    assert float(components["spatial_residual"].mean(dim=(-2, -1)).abs().max()) < 1e-6
    fractions = sum(component_energy_fraction(value, delta) for value in components.values())
    torch.testing.assert_close(fractions, torch.ones_like(fractions), atol=1e-5, rtol=1e-5)


def test_feature_progress_has_expected_endpoints() -> None:
    good = torch.randn(9, 11)
    bad = torch.randn(9, 11)
    torch.testing.assert_close(feature_progress(good, good, bad), torch.zeros(9))
    torch.testing.assert_close(feature_progress(bad, good, bad), torch.ones(9))
    midpoint = 0.75 * good + 0.25 * bad
    torch.testing.assert_close(feature_progress(midpoint, good, bad), torch.full((9,), 0.25))


@pytest.mark.parametrize("interpolate", [rms_preserving_lerp, spherical_interpolate])
def test_geometry_controls_preserve_interpolated_sample_rms(interpolate) -> None:
    start = torch.randn(7, 6, 4, 4)
    end = torch.randn(7, 6, 4, 4) * 1.7
    alpha = 0.35
    result = interpolate(start, end, alpha)
    expected = torch.lerp(sample_rms(start), sample_rms(end), alpha)
    torch.testing.assert_close(sample_rms(result), expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(interpolate(start, end, 0.0), start)
    torch.testing.assert_close(interpolate(start, end, 1.0), end)


def test_distribution_metrics_are_zero_for_identical_inputs() -> None:
    values = torch.randn(32, 20)
    directions = random_unit_directions(20, 16, seed=3)
    projection = random_unit_directions(20, 8, seed=5)
    assert standardized_sliced_wasserstein(values, values.clone(), directions) == pytest.approx(0.0)
    assert projected_frechet_distance(values, values.clone(), projection) == pytest.approx(
        0.0, abs=1e-8
    )


def test_gate_accepts_consistent_quality_direction() -> None:
    distribution_rows = []
    sample_rows = []
    source_score = {"static": 1.0, "random": 2.0, "annealed": 3.0, "reverse": 4.0}
    for good, bad in PATH_PAIRS:
        pair = f"{good}_to_{bad}"
        start, stop = source_score[good], source_score[bad]
        for alpha, suffix in ((0.0, "0"), (0.25, "25"), (0.5, "50"), (0.75, "75"), (1.0, "100")):
            value = start + alpha * (stop - start)
            distribution_rows.append(
                {"pair": pair, "condition": f"own_a{suffix}", "alpha": alpha, "projected_frechet": value, "swd": value}
            )
        for condition, value in (
            ("bad_shuffled", stop - 0.05 * (stop - start)),
            ("bad_random", stop),
        ):
            distribution_rows.append(
                {"pair": pair, "condition": condition, "alpha": 0.25, "projected_frechet": value, "swd": value}
            )
        for condition, progress in (("own_a25", 0.25), ("good_shuffled", 0.02), ("good_random", 0.0)):
            for index in range(8):
                sample_rows.append({"pair": pair, "condition": condition, "sample_index": index, "feature_progress": progress})
    result = path_difference_gate(pd.DataFrame(distribution_rows), pd.DataFrame(sample_rows))
    assert result["pass"] is True
    assert result["counts"] == {"dose_wins": 4, "correction_wins": 4, "specificity_wins": 4}
