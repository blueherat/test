from __future__ import annotations

import torch

from experiments.run_rae_spc_directional_sensitivity import (
    match_per_sample_norm,
    orthogonal_control_basis,
)


def test_control_basis_is_orthogonal_to_guided_basis() -> None:
    guided, _ = torch.linalg.qr(torch.randn(12, 3, dtype=torch.float64))
    control = orthogonal_control_basis(guided.float(), seed=7)
    assert torch.allclose(
        guided.float().T @ control, torch.zeros(3, 3), atol=1e-6
    )
    assert torch.allclose(control.T @ control, torch.eye(3), atol=1e-6)


def test_match_per_sample_norm() -> None:
    control = torch.randn(4, 3, 2, 2)
    reference = torch.randn_like(control)
    matched = match_per_sample_norm(control, reference)
    assert torch.allclose(
        torch.linalg.vector_norm(matched.flatten(1), dim=1),
        torch.linalg.vector_norm(reference.flatten(1), dim=1),
        atol=1e-5,
    )
