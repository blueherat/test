import torch

from experiments.rae_spectral_direction_loss import radial_band_index
from experiments.small_image_basis_transport import (
    OrthogonalDirectionLoss,
    dct_pixel_basis,
)
from experiments.small_image_group_splice import blend_velocity


def test_group0_and_nonzero_splices_recompose_full_difference():
    size = 4
    groups = radial_band_index(size, 4).flatten()
    moments = torch.tensor([2.0, 0.8, 0.3, 0.1])[groups]
    analyzer = OrthogonalDirectionLoss(
        dct_pixel_basis(size),
        moments,
        groups,
        gamma=0.5,
    )
    generator = torch.Generator().manual_seed(0)
    baseline = torch.randn((5, 1, size, size), generator=generator)
    weighted = torch.randn((5, 1, size, size), generator=generator)
    group0 = blend_velocity(baseline, weighted, analyzer, "group0")
    nonzero = blend_velocity(baseline, weighted, analyzer, "nonzero")

    assert torch.allclose(
        group0 + nonzero - baseline,
        weighted,
        atol=3e-6,
        rtol=3e-6,
    )
    assert torch.equal(blend_velocity(baseline, weighted, analyzer, "baseline"), baseline)
    assert torch.equal(blend_velocity(baseline, weighted, analyzer, "all"), weighted)
