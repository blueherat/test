import torch

from experiments.rae_spectral_gradient_audit import (
    dct2_basis,
    head_gradient_sketch,
    inverse_spatial_transform,
    project_spatial_bands,
    radial_band_masks,
    sample_shifted_logit_normal,
    spatial_transform,
)


def test_dct_basis_is_orthonormal_and_round_trips():
    basis = dct2_basis(4)
    assert torch.allclose(basis @ basis.T, torch.eye(16, dtype=basis.dtype), atol=1e-12)
    x = torch.randn(3, 5, 4, 4, dtype=torch.float64)
    reconstructed = inverse_spatial_transform(spatial_transform(x, basis), basis, 4)
    assert torch.allclose(reconstructed, x, atol=1e-11, rtol=1e-11)


def test_radial_bands_form_an_exact_orthogonal_partition():
    basis = dct2_basis(4).float()
    masks = radial_band_masks(4, 4)
    assert torch.equal(masks.sum(dim=0), torch.ones(16, dtype=torch.long))
    x = torch.randn(2, 3, 4, 4)
    projected = project_spatial_bands(x, basis, masks)
    assert torch.allclose(projected.sum(dim=1), x, atol=2e-6, rtol=2e-6)
    energy = projected.square().sum(dim=(1, 2, 3, 4))
    assert torch.allclose(energy, x.square().sum(dim=(1, 2, 3)), atol=2e-5, rtol=2e-5)


def test_head_gradient_sketch_matches_explicit_linear_gradient_projection():
    torch.manual_seed(0)
    batch, tokens, channels, hidden_dim, rank = 2, 4, 5, 7, 3
    error_tokens = torch.randn(batch, tokens, channels, dtype=torch.float64)
    error = error_tokens.reshape(batch, 2, 2, channels).permute(0, 3, 1, 2)
    hidden = torch.randn(batch, tokens, hidden_dim, dtype=torch.float64)
    output_projection = torch.randn(channels, rank, dtype=torch.float64)
    hidden_projection = torch.randn(hidden_dim, rank, dtype=torch.float64)
    sketch = head_gradient_sketch(error, hidden, output_projection, hidden_projection)
    explicit = []
    for sample in range(batch):
        gradient = 2.0 * error_tokens[sample].T @ hidden[sample] / (tokens * channels)
        explicit.append(output_projection.T @ gradient @ hidden_projection)
    assert torch.allclose(sketch, torch.stack(explicit), atol=1e-12, rtol=1e-12)


def test_shifted_logit_normal_matches_official_monotone_shift():
    generator = torch.Generator().manual_seed(0)
    shifted = sample_shifted_logit_normal(20_000, 7.0, generator=generator)
    assert bool(torch.all((shifted > 0.0) & (shifted < 1.0)))
    assert float(shifted.mean()) > 0.75

