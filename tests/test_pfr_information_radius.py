import math
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.pfr_information_radius import (
    conditional_progress_fisher_information,
    expected_conditional_bridge_kl,
    solve_query_progress_for_kl,
)


def test_bridge_kl_is_zero_at_identical_progress() -> None:
    result = expected_conditional_bridge_kl(
        progress=0.25,
        query_progress=0.25,
        dimension=4096,
        clean_squared_norm_mean=3000.0,
    )
    assert result.total == pytest.approx(0.0, abs=1e-12)
    assert result.variance == pytest.approx(0.0, abs=1e-12)
    assert result.endpoint_mean == pytest.approx(0.0, abs=1e-12)


def test_bridge_kl_matches_scalar_gaussian_formula() -> None:
    result = expected_conditional_bridge_kl(
        progress=0.2,
        query_progress=0.3,
        dimension=1,
        clean_squared_norm_mean=4.0,
    )
    variance_ratio = 0.8**2 / 0.7**2
    expected = 0.5 * (
        variance_ratio - 1.0 - math.log(variance_ratio) + 0.1**2 * 4.0 / 0.7**2
    )
    assert result.total == pytest.approx(expected)
    assert result.total == pytest.approx(result.variance + result.endpoint_mean)


def test_fisher_information_is_local_kl_curvature() -> None:
    kwargs = {
        "progress": 0.35,
        "dimension": 128,
        "clean_squared_norm_mean": 96.0,
    }
    fisher = conditional_progress_fisher_information(**kwargs)
    for horizon in (1e-3, 5e-4, 2.5e-4):
        result = expected_conditional_bridge_kl(
            **kwargs,
            query_progress=kwargs["progress"] + horizon,
        )
        assert 2.0 * result.total / horizon**2 == pytest.approx(
            fisher, rel=5e-3
        )


def test_equal_local_information_radius_scales_as_inverse_sqrt_dimension() -> None:
    progress = 0.2
    per_coordinate_clean_energy = 0.75
    d_small = 4096
    d_large = 262144
    h_small = 1e-4
    h_large = h_small * math.sqrt(d_small / d_large)
    fisher_small = conditional_progress_fisher_information(
        progress=progress,
        dimension=d_small,
        clean_squared_norm_mean=d_small * per_coordinate_clean_energy,
    )
    fisher_large = conditional_progress_fisher_information(
        progress=progress,
        dimension=d_large,
        clean_squared_norm_mean=d_large * per_coordinate_clean_energy,
    )
    assert 0.5 * fisher_small * h_small**2 == pytest.approx(
        0.5 * fisher_large * h_large**2
    )


def test_query_progress_solver_inverts_exact_kl() -> None:
    source = 0.1
    target = 0.137
    kwargs = {
        "dimension": 4096,
        "clean_squared_norm_mean": 2600.0,
    }
    target_kl = expected_conditional_bridge_kl(
        progress=source,
        query_progress=target,
        **kwargs,
    ).total
    solved = solve_query_progress_for_kl(
        progress=source,
        target_kl=target_kl,
        **kwargs,
    )
    assert solved == pytest.approx(target, abs=1e-10)


def test_query_progress_solver_rejects_unreachable_target() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        solve_query_progress_for_kl(
            progress=0.1,
            target_kl=1e12,
            dimension=4,
            clean_squared_norm_mean=4.0,
            maximum_progress=0.2,
        )
