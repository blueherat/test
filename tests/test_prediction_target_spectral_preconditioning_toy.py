from __future__ import annotations

import torch

from experiments.run_prediction_target_spectral_preconditioning_toy import (
    ConditionSpec,
    SpectralStats,
    centered_constant_target,
    constant_k_values,
    linear_path_moments,
    lmmse_model_input,
    lmmse_residual_target,
    subspace_velocity_target,
    velocity_from_centered_constant_output,
    velocity_from_lmmse_output,
    velocity_from_subspace_output,
)


def make_stats(D: int = 7) -> SpectralStats:
    generator = torch.Generator().manual_seed(17)
    q, _ = torch.linalg.qr(torch.randn(D, D, generator=generator))
    eigenvalues = torch.tensor([3.0, 0.7] + [0.0] * (D - 2))
    return SpectralStats(
        mean=torch.linspace(-0.2, 0.3, D),
        basis=q,
        eigenvalues=eigenvalues,
        active=eigenvalues > 0,
        threshold=1e-6,
        samples=1000,
    )


def test_centered_constant_operator_recovers_exact_velocity() -> None:
    stats = make_stats()
    generator = torch.Generator().manual_seed(23)
    clean = torch.randn(31, stats.D, generator=generator) + stats.mean
    epsilon = torch.randn(clean.shape, generator=generator)
    time = torch.linspace(0.04, 0.96, len(clean))
    state = (1.0 - time[:, None]) * clean + time[:, None] * epsilon
    spec = ConditionSpec("test", "kdiff_spectrum")
    k_values = constant_k_values(spec, stats)
    target = centered_constant_target(clean, epsilon, stats, k_values)
    recovered = velocity_from_centered_constant_output(
        target, state, time, stats, k_values, clip=1e-7
    )
    torch.testing.assert_close(recovered, epsilon - clean, atol=2e-5, rtol=2e-5)


def test_lmmse_normalized_residual_recovers_exact_velocity() -> None:
    stats = make_stats()
    generator = torch.Generator().manual_seed(29)
    modes = torch.randn(37, stats.D, generator=generator)
    modes[:, 2:] = 0.0
    clean = modes @ stats.basis.T + stats.mean
    epsilon = torch.randn(clean.shape, generator=generator)
    time = torch.linspace(0.04, 0.96, len(clean))
    state = (1.0 - time[:, None]) * clean + time[:, None] * epsilon
    target = lmmse_residual_target(
        clean, epsilon, state, time, stats, clip=1e-7
    )
    recovered, effective, _ = velocity_from_lmmse_output(
        target, state, time, stats, clip=1e-7
    )
    torch.testing.assert_close(recovered, epsilon - clean, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(effective, target, atol=2e-5, rtol=2e-5)


def test_lmmse_residual_has_unit_unconditional_variance_for_gaussian_modes() -> None:
    stats = make_stats()
    generator = torch.Generator().manual_seed(31)
    count = 200_000
    clean_modes = torch.randn(count, stats.D, generator=generator)
    clean_modes *= stats.eigenvalues.sqrt()
    clean = clean_modes @ stats.basis.T + stats.mean
    epsilon = torch.randn(clean.shape, generator=generator)
    time = torch.full((count,), 0.37)
    state = (1.0 - time[:, None]) * clean + time[:, None] * epsilon
    target = lmmse_residual_target(
        clean, epsilon, state, time, stats, clip=1e-7
    )
    target_modes = target @ stats.basis
    variance = target_modes.var(dim=0)
    torch.testing.assert_close(variance[:2], torch.ones(2), atol=1.5e-2, rtol=1.5e-2)
    torch.testing.assert_close(variance[2:], torch.zeros(stats.D - 2), atol=1e-7, rtol=0)


def test_kdiff_and_lmmse_formulas_match_at_half_time() -> None:
    stats = make_stats()
    kdiff = constant_k_values(ConditionSpec("test", "kdiff_spectrum"), stats)
    time = torch.tensor([0.5])
    state_variance, coefficient, scale = linear_path_moments(time, stats, 1e-7)
    expected = 1.0 / (1.0 + stats.eigenvalues)
    torch.testing.assert_close(kdiff, expected)
    assert torch.all(state_variance > 0)
    assert torch.all(torch.isfinite(coefficient))
    assert torch.all(scale[0, ~stats.active] == 0)


def test_subspace_velocity_recovers_exact_linear_manifold_velocity() -> None:
    stats = make_stats()
    generator = torch.Generator().manual_seed(37)
    clean_modes = torch.randn(41, stats.D, generator=generator)
    clean_modes[:, 2:] = 0.0
    clean = clean_modes @ stats.basis.T + stats.mean
    epsilon = torch.randn(clean.shape, generator=generator)
    time = torch.linspace(0.04, 0.96, len(clean))
    state = (1.0 - time[:, None]) * clean + time[:, None] * epsilon
    target = subspace_velocity_target(clean, epsilon, stats)
    recovered, effective = velocity_from_subspace_output(
        target, state, time, stats, clip=1e-7
    )
    torch.testing.assert_close(recovered, epsilon - clean, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(effective, target, atol=2e-5, rtol=2e-5)


def test_lmmse_input_modes_transform_only_requested_subspaces() -> None:
    stats = make_stats()
    generator = torch.Generator().manual_seed(41)
    state = torch.randn(13, stats.D, generator=generator)
    time = torch.linspace(0.1, 0.9, len(state))
    centered = state - (1.0 - time[:, None]) * stats.mean[None]
    raw_modes = centered @ stats.basis
    variance, _, _ = linear_path_moments(time, stats, clip=1e-7)
    white_modes = raw_modes / variance.sqrt()

    active_white = lmmse_model_input(
        state, time, stats, input_mode="active_whitened", clip=1e-7
    ) @ stats.basis
    projected_raw = lmmse_model_input(
        state, time, stats, input_mode="projected_raw", clip=1e-7
    ) @ stats.basis
    projected_white = lmmse_model_input(
        state, time, stats, input_mode="projected_whitened", clip=1e-7
    ) @ stats.basis

    torch.testing.assert_close(
        active_white[:, stats.active],
        white_modes[:, stats.active],
        atol=2e-5,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        active_white[:, ~stats.active],
        raw_modes[:, ~stats.active],
        atol=2e-5,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        projected_raw[:, stats.active],
        raw_modes[:, stats.active],
        atol=2e-5,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        projected_white[:, stats.active],
        white_modes[:, stats.active],
        atol=2e-5,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        projected_raw[:, ~stats.active],
        torch.zeros_like(projected_raw[:, ~stats.active]),
        atol=2e-5,
        rtol=0,
    )
    torch.testing.assert_close(
        projected_white[:, ~stats.active],
        torch.zeros_like(projected_white[:, ~stats.active]),
        atol=2e-5,
        rtol=0,
    )


def test_sit_forward_affine_residual_satisfies_normal_equations() -> None:
    """Independently check the noise-to-data convention used by SiT."""
    generator = torch.Generator().manual_seed(53)
    count, dimension = 200_000, 4
    basis, _ = torch.linalg.qr(
        torch.randn(dimension, dimension, generator=generator, dtype=torch.float64)
    )
    eigenvalues = torch.tensor([3.0, 0.7, 0.2, 0.05], dtype=torch.float64)
    covariance = basis @ torch.diag(eigenvalues) @ basis.T
    mean = torch.linspace(-0.3, 0.2, dimension, dtype=torch.float64)

    clean_modes = torch.randn(
        count, dimension, generator=generator, dtype=torch.float64
    ) * eigenvalues.sqrt()
    clean = clean_modes @ basis.T + mean
    noise = torch.randn(clean.shape, generator=generator, dtype=torch.float64)
    time = 0.37
    one_minus_time = 1.0 - time
    state = one_minus_time * noise + time * clean
    velocity = clean - noise

    state_covariance = (
        one_minus_time**2 * torch.eye(dimension, dtype=torch.float64)
        + time**2 * covariance
    )
    velocity_state_covariance = (
        time * covariance
        - one_minus_time * torch.eye(dimension, dtype=torch.float64)
    )
    affine_matrix = velocity_state_covariance @ torch.linalg.inv(state_covariance)
    centered_state = state - time * mean
    affine_velocity = mean + centered_state @ affine_matrix.T
    residual = velocity - affine_velocity

    residual_mean = residual.mean(dim=0)
    residual_state_cross = residual.T @ centered_state / count
    residual_covariance = residual.T @ residual / count
    expected_residual_covariance = covariance @ torch.linalg.inv(state_covariance)

    torch.testing.assert_close(
        residual_mean, torch.zeros_like(residual_mean), atol=1.5e-2, rtol=0
    )
    torch.testing.assert_close(
        residual_state_cross,
        torch.zeros_like(residual_state_cross),
        atol=1.5e-2,
        rtol=0,
    )
    torch.testing.assert_close(
        residual_covariance,
        expected_residual_covariance,
        atol=2.5e-2,
        rtol=2.5e-2,
    )
