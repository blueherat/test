import torch

from experiments.rae_spectral_direction_loss import DCTDirectionLoss
from experiments.small_image_basis_transport import (
    OrthogonalDirectionLoss,
    build_direction_analyzer,
    dct_pixel_basis,
)


def test_explicit_dct_basis_matches_matrix_dct_loss():
    size = 4
    moments = torch.tensor([0.2, 0.7, 1.4, 2.0])
    reference = DCTDirectionLoss(
        size,
        moments.tolist(),
        gamma=0.5,
        damping=1e-4,
        min_weight=0.2,
        max_weight=2.0,
    )
    groups = reference.band_index.flatten()
    explicit = OrthogonalDirectionLoss(
        dct_pixel_basis(size),
        moments[groups],
        groups,
        gamma=0.5,
        damping=1e-4,
        min_weight=0.2,
        max_weight=2.0,
    )
    prediction = torch.randn((7, 1, size, size), generator=torch.Generator().manual_seed(0))
    target = torch.randn((7, 1, size, size), generator=torch.Generator().manual_seed(1))
    time = torch.linspace(0.1, 0.9, len(prediction))

    expected_loss, _ = reference(prediction, target, time)
    actual_loss, _ = explicit(prediction, target, time)

    assert torch.allclose(
        explicit.transform(prediction),
        reference.transform(prediction).flatten(1),
        atol=2e-6,
        rtol=2e-6,
    )
    assert torch.allclose(actual_loss, expected_loss, atol=2e-6, rtol=2e-6)
    assert torch.allclose(
        explicit.band_mse(prediction - target),
        reference.band_mse(prediction - target),
        atol=2e-6,
        rtol=2e-6,
    )


def test_all_bases_have_identical_time_dependent_weight_spectrum():
    generator = torch.Generator().manual_seed(4)
    train = torch.randn((64, 1, 28, 28), generator=generator)
    analyzers = {
        name: build_direction_analyzer(
            train,
            name,
            band_count=8,
            gamma=0.5,
            seed=3,
        )[0]
        for name in ("dct", "pca", "random")
    }
    times = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
    spectra = {
        name: torch.sort(analyzer.weights(times), dim=1).values
        for name, analyzer in analyzers.items()
    }

    assert torch.allclose(spectra["dct"], spectra["pca"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(spectra["dct"], spectra["random"], atol=1e-6, rtol=1e-6)
    for analyzer in analyzers.values():
        assert torch.allclose(analyzer.weights(times).mean(dim=1), torch.ones(len(times)))
