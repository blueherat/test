from __future__ import annotations

import torch

from experiments.rae_layerwise_path import plan_layerwise_path
from experiments.run_rae_spc_cross_path_study import (
    component_losses_per_sample,
    component_shift_per_sample,
    evaluation_path_kwargs,
)


def test_semantic_target_is_unchanged_when_only_detail_path_changes() -> None:
    clean = torch.randn(2, 4, 3, 3)
    noise = torch.randn_like(clean)
    basis = torch.eye(4)[:, :2]
    time = torch.tensor([0.85, 0.3])
    paths = evaluation_path_kwargs(
        {
            "path_power": 2.0,
            "path_family": "power",
            "path_floor": 0.2,
            "path_alpha": 1.0,
            "detail_scale": 1.0,
        }
    )
    static = plan_layerwise_path(clean, noise, time, basis, **paths["static"])
    spc = plan_layerwise_path(clean, noise, time, basis, **paths["spc"])
    semantic_shift, basis_shift = component_shift_per_sample(
        static.target, spc.target, basis
    )
    assert torch.allclose(semantic_shift, torch.zeros_like(semantic_shift), atol=1e-12)
    assert torch.all(basis_shift > 0)


def test_component_losses_sum_to_total_mse() -> None:
    prediction = torch.randn(3, 4, 2, 2)
    target = torch.randn_like(prediction)
    basis = torch.eye(4)[:, :2]
    semantic, detail = component_losses_per_sample(prediction, target, basis)
    total = (prediction - target).square().flatten(1).mean(1)
    assert torch.allclose(semantic + detail, total, atol=1e-6)
