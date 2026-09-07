"""Compare the structured implementation with small, dense Gaussian models."""
import torch

from experiments.raev2_directional_variance import colored_noise as old_colored_noise
from experiments.raev2_spatial_guidance_covariance import (
    PLAN, colored_noise, reflect, standardized_residual, token_basis,
)


def fixture():
    rng = torch.Generator().manual_seed(1337)
    full = torch.randn(1, 3, 2, 2, generator=rng)
    base = torch.randn(1, 3, 2, 2, generator=rng)
    u, w, active = token_basis(full, base)
    dense = torch.zeros(12, 4, dtype=torch.float64)
    for j in range(4):
        dense[j::4, j] = u.flatten(2)[0, :, j]
    a = torch.tensor([[1.8, .3, -.2], [.3, .9, .15], [-.2, .15, 1.3]], dtype=torch.float64)
    values, vectors = torch.linalg.eigh(a)
    root = (vectors * values.sqrt()) @ vectors.T
    return full, base, dense, w[0], a, root


def test_fixed_bank_sizes_are_numeric():
    assert PLAN['train'] == 64000 and PLAN['validation'] == 8000


def test_local_columns_and_global_direction():
    full, base, u, w, _, _ = fixture()
    torch.testing.assert_close(u.T @ u, torch.eye(4, dtype=torch.float64), atol=1e-13, rtol=1e-13)
    delta = (full - base).double().flatten()
    torch.testing.assert_close(u @ w, delta / delta.norm(), atol=1e-13, rtol=1e-13)


def test_noise_matches_dense_covariance_and_preserves_global_eigenvalue():
    full, base, u, w, k, root = fixture()
    kappa = 3.5
    ug = u @ w
    h = reflect(torch.eye(4, dtype=torch.float64), w.expand(4, -1)).T
    v = u @ h[:, 1:]
    torch.testing.assert_close(v.T @ ug, torch.zeros(3, dtype=torch.float64), atol=1e-13, rtol=0)
    target = torch.eye(12, dtype=torch.float64) + (kappa - 1) * torch.outer(ug, ug) + v @ (k - torch.eye(3)) @ v.T
    torch.testing.assert_close(target @ ug, kappa * ug, atol=1e-12, rtol=1e-12)
    basis_noise = torch.eye(12).reshape(12, 3, 2, 2)
    result, _ = colored_noise(basis_noise, full.expand(12, -1, -1, -1), base.expand(12, -1, -1, -1), root, kappa)
    factor = result.double().flatten(1).T
    torch.testing.assert_close(factor @ factor.T, target, atol=5e-7, rtol=5e-7)


def test_unit_matrix_and_zero_token_recover_old_method_exactly():
    full, base, _, _, _, _ = fixture()
    noise = torch.randn(full.shape, generator=torch.Generator().manual_seed(123))
    expected, _ = old_colored_noise(noise, full, base, 17.2)
    actual, _ = colored_noise(noise, full, base, torch.eye(3), 17.2)
    assert torch.equal(actual, expected)
    full[:, :, 0, 0] = base[:, :, 0, 0]
    actual, active = colored_noise(noise, full, base, 3 * torch.eye(3), 17.2)
    expected, _ = old_colored_noise(noise, full, base, 17.2)
    assert not active.any() and torch.equal(actual, expected)


def test_likelihood_difference_equals_dense_gaussian_fixed_mean():
    full, base, u, w, k, _ = fixture()
    error = torch.randn(full.shape, generator=torch.Generator().manual_seed(711)).double()
    kappa, mse = 4.3, 1.7
    r, active, _, _ = standardized_residual(error, full, base, torch.tensor([mse], dtype=torch.float64))
    ug = u @ w
    h = reflect(torch.eye(4, dtype=torch.float64), w.expand(4, -1)).T
    v = u @ h[:, 1:]
    s0 = torch.eye(12, dtype=torch.float64) + (kappa ** .5 - 1) * torch.outer(ug, ug)
    c0 = mse * s0 @ s0
    ck = c0 + mse * v @ (k - torch.eye(3)) @ v.T
    e = error.flatten()
    dense = .5 * (torch.linalg.slogdet(ck)[1] - torch.linalg.slogdet(c0)[1]
                    + e @ torch.linalg.solve(ck, e) - e @ torch.linalg.solve(c0, e))
    projected = .5 * (torch.linalg.slogdet(k)[1] + r[0] @ torch.linalg.solve(k, r[0]) - r.square().sum())
    assert active.all()
    torch.testing.assert_close(projected, dense, atol=1e-12, rtol=1e-12)
