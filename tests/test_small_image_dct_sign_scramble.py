import torch

from experiments.small_image_basis_transport import dct_pixel_basis
from experiments.rae_spectral_direction_loss import radial_band_index
from experiments.small_image_dct_sign_scramble import (
    dct_sign_scramble,
    grouped_coefficient_power,
)


def test_sign_scramble_preserves_every_dct_component_power_and_dc():
    images = torch.randn(
        (11, 1, 8, 8), generator=torch.Generator().manual_seed(3)
    )
    basis = dct_pixel_basis(8)
    scrambled, coefficients, scrambled_coefficients = dct_sign_scramble(
        images, basis, seed=7
    )

    assert torch.allclose(
        coefficients.square(),
        scrambled_coefficients.square(),
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(coefficients[:, 0], scrambled_coefficients[:, 0])
    groups = radial_band_index(8, 4).flatten()
    assert torch.allclose(
        grouped_coefficient_power(coefficients, groups),
        grouped_coefficient_power(scrambled_coefficients, groups),
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        images.mean(dim=(1, 2, 3)),
        scrambled.mean(dim=(1, 2, 3)),
        atol=1e-6,
        rtol=1e-6,
    )
    assert not torch.allclose(images, scrambled)
