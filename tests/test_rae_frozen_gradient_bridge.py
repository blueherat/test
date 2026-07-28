import pytest
import torch

from experiments.rae_frozen_gradient_bridge import (
    band_gradient_tables,
    band_losses,
)
from experiments.rae_spectral_gradient_audit import (
    dct2_basis,
    radial_band_masks,
    random_orthogonal_basis,
)


def test_band_losses_partition_orthogonal_mse_for_both_bases():
    error = torch.randn((3, 2, 4, 4), generator=torch.Generator().manual_seed(4))
    masks = radial_band_masks(4, 4)
    for basis in (dct2_basis(4).float(), random_orthogonal_basis(4, seed=5).float()):
        losses = band_losses(error, basis, masks)
        assert torch.allclose(sum(losses), error.square().mean(), atol=1e-6)


def test_band_gradient_tables_distinguish_aligned_and_decoupled_groups():
    weights = torch.tensor([0.2, 1.2, 1.2])
    aligned = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    decoupled = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    aligned_aggregate, _, aligned_cosine = band_gradient_tables(aligned, weights)
    decoupled_aggregate, _, decoupled_cosine = band_gradient_tables(decoupled, weights)

    assert aligned_aggregate["coarse_descent_ratio"] == pytest.approx(2.6 / 3.0)
    assert decoupled_aggregate["coarse_descent_ratio"] == pytest.approx(0.2)
    assert aligned_cosine[0, 1] == pytest.approx(1.0)
    assert decoupled_cosine[0, 1] == pytest.approx(0.0)
