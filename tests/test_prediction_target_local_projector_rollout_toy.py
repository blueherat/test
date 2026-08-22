from __future__ import annotations

import torch

from experiments.run_prediction_target_extrapolation_toy_v4 import CurvedEmbedding
from experiments.run_prediction_target_local_projector_rollout_toy import (
    tangent_selected_clean,
)


def test_tangent_selector_uses_v_tangent_and_x_normal() -> None:
    device = torch.device("cpu")
    embedding = CurvedEmbedding(
        11,
        curvature=0.5,
        frequency_scale=4.0,
        seed=7,
        device=device,
        scale_mode="unit_rms",
    )
    intrinsic = torch.tensor([[0.2, -0.3], [-0.4, 0.1]])
    geometry = embedding.embed(intrinsic)
    generator = torch.Generator().manual_seed(8)
    x_clean = torch.randn(2, 11, generator=generator)
    v_clean = torch.randn(2, 11, generator=generator)

    selected = tangent_selected_clean(
        x_clean=x_clean,
        v_clean=v_clean,
        geometry_clean=geometry,
        embedding=embedding,
    )
    basis = embedding.tangent_basis(intrinsic)
    selected_tangent, selected_normal = embedding.split_with_tangent_basis(
        selected, basis
    )
    v_tangent, _ = embedding.split_with_tangent_basis(v_clean, basis)
    _, x_normal = embedding.split_with_tangent_basis(x_clean, basis)

    torch.testing.assert_close(selected_tangent, v_tangent, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(selected_normal, x_normal, atol=1e-5, rtol=1e-5)
