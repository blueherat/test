from __future__ import annotations

import sys
from pathlib import Path

import torch

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from semigroup_consistent_guidance import (  # noqa: E402
    local_jensen_velocity_coefficient,
    local_jensen_velocity_correction,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.run_imagenet100_sit_path_extrapolated_ig import (  # noqa: E402
    Condition,
    condition_from_payload,
)


def test_local_jensen_coefficient_vanishes_at_required_boundaries() -> None:
    times = torch.tensor([0.0, 0.25, 1.0])
    coefficient = local_jensen_velocity_coefficient(1.7, times)
    assert coefficient[0].item() == 0.0
    assert coefficient[2].item() == 0.0
    assert coefficient[1].item() > 0.0


def test_local_jensen_correction_is_the_gap_energy_vjp() -> None:
    state = torch.tensor(
        [[0.2, -0.5], [1.0, 0.3]], dtype=torch.float64, requires_grad=True
    )
    matrix = torch.tensor([[1.2, -0.4], [0.7, 0.9]], dtype=torch.float64)
    offset = torch.tensor([0.1, -0.2], dtype=torch.float64)
    gap = state @ matrix.T + offset
    time_value = torch.tensor(0.3, dtype=torch.float64)
    beta = 1.6

    correction = local_jensen_velocity_correction(
        gap,
        state=state,
        time_value=time_value,
        beta=beta,
    )
    expected = (
        beta
        * (beta - 1.0)
        * time_value
        * (1.0 - time_value)
        * (gap @ matrix)
    )
    torch.testing.assert_close(correction, expected)


def test_semigroup_condition_roundtrips_canonically() -> None:
    condition = Condition("semigroup_local_jensen")
    assert condition.name == "ig_depth4_semigroup_local_jensen"
    assert condition_from_payload(condition.payload()) == condition
    assert condition.payload()["foresight_material_derivative"] == {
        "endpoint_target": "p_strong^beta * p_weak^(1-beta)",
        "missing_potential": "log E_q[r^beta|z_t] - beta*log E_q[r|z_t]",
        "approximation": "first local heat-time term",
        "extra_tunable_coefficients": 0,
    }
