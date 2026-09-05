from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.pfr_eulerian_decomposition import (
    EulerianDecompositionField,
    finite_eulerian_components,
)


def test_finite_eulerian_decomposition_is_exact() -> None:
    generator = torch.Generator().manual_seed(7)
    weak_now = torch.randn(3, 4, 2, 2, generator=generator)
    weak_time = torch.randn(3, 4, 2, 2, generator=generator)
    weak_material = torch.randn(3, 4, 2, 2, generator=generator)

    parts = finite_eulerian_components(weak_now, weak_time, weak_material)

    torch.testing.assert_close(parts.eulerian, parts.material + parts.frame)


def test_linear_field_secants_have_expected_small_h_form() -> None:
    state = torch.tensor([[1.0, -2.0]])
    velocity = torch.tensor([[0.5, 3.0]])
    matrix = torch.tensor([[2.0, -1.0], [0.25, 0.5]])
    time_slope = torch.tensor([[4.0, -3.0]])
    time = 0.2
    horizon = 0.03125

    def weak(value: torch.Tensor, at: float) -> torch.Tensor:
        return value @ matrix.T + at * time_slope

    weak_now = weak(state, time)
    weak_time = weak(state, time + horizon)
    weak_material = weak(state + horizon * velocity, time + horizon)
    parts = finite_eulerian_components(weak_now, weak_time, weak_material)

    torch.testing.assert_close(parts.eulerian, horizon * time_slope)
    torch.testing.assert_close(
        parts.material,
        horizon * (time_slope + velocity @ matrix.T),
    )
    torch.testing.assert_close(parts.frame, -horizon * (velocity @ matrix.T))


def test_scaled_eulerian_field_validates_query_parameters() -> None:
    labels = torch.zeros(1, dtype=torch.long)
    with pytest.raises(ValueError, match="anchor_horizon"):
        EulerianDecompositionField(
            object(), labels, "time_only", anchor_horizon=0.0
        )
    with pytest.raises(ValueError, match="revision_scale"):
        EulerianDecompositionField(
            object(), labels, "time_only", revision_scale=float("nan")
        )
