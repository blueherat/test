from __future__ import annotations

import torch

from experiments.analyze_rae_spc_subspace_signal import (
    projected_variance_per_active_dimension,
    snr_rows,
)


def test_projected_variance_uses_only_active_dimensions() -> None:
    residual = torch.zeros(2, 4, 2, 2)
    residual[:, :2] = 3.0
    basis = torch.eye(4)[:, :2]
    assert torch.isclose(
        projected_variance_per_active_dimension(residual, basis),
        torch.tensor(9.0, dtype=torch.float64),
    )


def test_spc_only_reduces_guided_state_snr() -> None:
    table = snr_rows(
        {
            "guided_variance_per_active_dim": 4.0,
            "control_variance_per_active_dim": 1.0,
            "complement_variance_per_active_dim": 2.0,
        },
        times=(0.5,),
        floor=0.2,
        power=2.0,
    ).set_index("subspace")
    assert table.loc["guided", "static_state_snr"] == 4.0
    assert table.loc["guided", "spc_state_snr"] < 4.0
    assert table.loc["control", "spc_over_static_snr"] == 1.0
