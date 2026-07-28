from __future__ import annotations

import torch

from experiments.rae_dual_stream import (
    SemanticConditionedDetailDDT,
    fuse_semantic_coefficients,
    split_semantic_coefficients,
)
from experiments.rae_layerwise_path import random_detail_basis


def test_dual_stream_split_and_fuse_is_exact() -> None:
    generator = torch.Generator().manual_seed(5)
    final = torch.randn((2, 12, 4, 4), generator=generator)
    basis = random_detail_basis(12, 3, seed=7)
    semantic, coefficients = split_semantic_coefficients(final, basis)
    fused = fuse_semantic_coefficients(semantic, coefficients, basis)
    torch.testing.assert_close(fused, final, atol=2e-6, rtol=0)
    torch.testing.assert_close(
        coefficients.mean(dim=(-2, -1)), torch.zeros((2, 3)), atol=2e-6, rtol=0
    )


def test_conditioned_detail_model_has_expected_shape() -> None:
    model = SemanticConditionedDetailDDT(
        detail_channels=4, semantic_channels=8, input_size=4, num_classes=10
    )
    detail = torch.randn((2, 4, 4, 4))
    semantic = torch.randn((2, 8, 4, 4))
    output = model(detail, torch.tensor([0.2, 0.8]), torch.tensor([1, 2]), semantic)
    assert output.shape == detail.shape
