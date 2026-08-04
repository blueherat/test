from __future__ import annotations

import torch

from experiments.run_raev2_ig_curl_audit import (
    antisymmetric_bilinear,
    normalize_rms,
)


def test_normalize_rms_sets_unit_rms() -> None:
    value = normalize_rms(torch.tensor([[1.0, 2.0, 3.0]]))
    assert torch.allclose(value.square().mean().sqrt(), torch.tensor(1.0))


def test_symmetric_linear_map_has_zero_antisymmetric_form() -> None:
    matrix = torch.tensor([[2.0, 1.0], [1.0, 3.0]])
    u = torch.tensor([[1.0, -2.0]])
    v = torch.tensor([[3.0, 0.5]])
    j_u = u @ matrix.T
    j_v = v @ matrix.T
    anti, _, _ = antisymmetric_bilinear(u, v, j_u, j_v)
    assert torch.allclose(anti, torch.zeros_like(anti), atol=1e-6)


def test_skew_linear_map_has_nonzero_antisymmetric_form() -> None:
    matrix = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    u = torch.tensor([[1.0, 0.0]])
    v = torch.tensor([[0.0, 1.0]])
    j_u = u @ matrix.T
    j_v = v @ matrix.T
    anti, _, _ = antisymmetric_bilinear(u, v, j_u, j_v)
    assert anti.abs().item() > 0.5
