from __future__ import annotations

import torch

from experiments.prediction_target_spectral_projector import (
    SpectralProjector,
    SpectralTarget,
    estimate_pca_projector,
    estimate_soft_spectral_projector,
    projector_alignment,
)


def orthonormal_basis(D: int, rank: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(D, rank, generator=generator, dtype=torch.float64)
    basis, _ = torch.linalg.qr(raw, mode="reduced")
    return basis.float()


def test_soft_spectral_target_recovers_exact_velocity() -> None:
    D = 19
    basis = orthonormal_basis(D, 5, 3)
    projector = SpectralProjector(
        basis=basis,
        weights=torch.tensor([1.0, 0.85, 0.5, 0.2, 0.05]),
        source="test",
    )
    target = SpectralTarget(
        "soft", projector=projector, tangent_k=0.5, normal_k=0.9
    )
    generator = torch.Generator().manual_seed(5)
    clean = torch.randn(23, D, generator=generator)
    epsilon = torch.randn(23, D, generator=generator)
    time = torch.linspace(0.03, 0.97, len(clean))
    state = (1.0 - time[:, None]) * clean + time[:, None] * epsilon
    native = target.target(clean, epsilon)
    recovered = target.velocity(native, state, time, conversion_clip=1e-7)
    torch.testing.assert_close(recovered, epsilon - clean, atol=3e-5, rtol=3e-5)


def test_hard_pca_estimator_recovers_unknown_linear_subspace() -> None:
    D = 31
    rank = 3
    true_basis = orthonormal_basis(D, rank, 7)
    generator = torch.Generator().manual_seed(11)
    coordinates = torch.randn(512, rank, generator=generator)
    samples = coordinates @ true_basis.T
    estimated, _eigenvalues = estimate_pca_projector(
        samples, rank=rank, source="train_split"
    )
    diagnostics = projector_alignment(estimated, true_basis)
    assert diagnostics["principal_cosine_min"] > 0.9999
    assert diagnostics["projector_frobenius_error"] < 1e-4


def test_soft_spectrum_suppresses_zero_variance_directions() -> None:
    D = 29
    rank = 2
    true_basis = orthonormal_basis(D, rank, 13)
    generator = torch.Generator().manual_seed(17)
    coordinates = torch.randn(1024, rank, generator=generator)
    samples = coordinates @ true_basis.T
    estimated, _eigenvalues, _tau = estimate_soft_spectral_projector(
        samples,
        tau_ratio=1e-3,
        max_rank=16,
        min_weight=1e-3,
        source="soft_train_split",
    )
    diagnostics = projector_alignment(estimated, true_basis)
    assert estimated.rank == rank
    assert diagnostics["principal_cosine_min"] > 0.9999
    assert diagnostics["projector_frobenius_error"] < 0.01


def test_projector_estimation_is_detached_from_sample_bank() -> None:
    samples = torch.randn(64, 8, requires_grad=True)
    estimated, _eigenvalues = estimate_pca_projector(
        samples, rank=2, source="detached"
    )
    assert not estimated.basis.requires_grad
    assert not estimated.weights.requires_grad
