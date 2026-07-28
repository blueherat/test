from __future__ import annotations

import pytest
import torch

from experiments.rae_layerwise_path import (
    MiddleFinalCovariance,
    fit_detail_subspace,
    fit_fractional_predictability_subspace,
    plan_layerwise_path,
    random_detail_basis,
    split_semantic_detail,
    subspace_regression_metrics,
)


def test_split_is_exact_and_detail_has_zero_token_mean() -> None:
    generator = torch.Generator().manual_seed(3)
    latent = torch.randn((4, 12, 5, 5), generator=generator)
    basis = random_detail_basis(12, 5, seed=7)
    semantic, detail = split_semantic_detail(latent, basis)
    torch.testing.assert_close(semantic + detail, latent, atol=2e-6, rtol=0)
    torch.testing.assert_close(
        detail.mean(dim=(-2, -1)),
        torch.zeros((4, 12)),
        atol=2e-6,
        rtol=0,
    )


def test_scaled_random_control_preserves_the_same_clean_endpoint() -> None:
    generator = torch.Generator().manual_seed(19)
    clean = torch.randn((3, 12, 4, 4), generator=generator)
    basis = random_detail_basis(12, 3, seed=23)
    semantic, detail = split_semantic_detail(clean, basis, detail_scale=2.5)
    torch.testing.assert_close(semantic + detail, clean, atol=2e-6, rtol=0)
    torch.testing.assert_close(
        detail.mean(dim=(-2, -1)),
        torch.zeros((3, 12)),
        atol=2e-6,
        rtol=0,
    )


def test_middle_guided_fit_recovers_predictable_final_subspace() -> None:
    generator = torch.Generator().manual_seed(11)
    channels = 10
    true_basis = random_detail_basis(channels, 3, seed=13, dtype=torch.float64)
    middle = torch.randn((32, channels, 4, 4), generator=generator, dtype=torch.float64)
    rows = middle.permute(0, 2, 3, 1).reshape(-1, channels)
    predictable = (rows @ true_basis) @ true_basis.transpose(0, 1)
    final_rows = predictable + 0.02 * torch.randn(
        predictable.shape, generator=generator, dtype=torch.float64
    )
    final = final_rows.reshape(32, 4, 4, channels).permute(0, 3, 1, 2)
    moments = MiddleFinalCovariance.zeros(channels, device="cpu")
    moments.update(middle, final)
    fitted = fit_detail_subspace(moments, rank=3, ridge=1e-5)
    overlap = torch.linalg.svdvals(fitted.basis.double().T @ true_basis).mean()
    assert float(overlap) > 0.995


def test_fractional_predictability_separates_r2_from_absolute_variance() -> None:
    # Direction 0 has much more predictable energy in absolute units, while
    # direction 1 has a much larger predictable fraction (R^2).
    moments = MiddleFinalCovariance(
        middle_gram=torch.eye(2, dtype=torch.float64),
        middle_final=torch.diag(torch.tensor([5.0, 0.95], dtype=torch.float64)),
        final_gram=torch.diag(torch.tensor([100.0, 1.0], dtype=torch.float64)),
        token_count=1,
    )
    absolute = fit_detail_subspace(moments, rank=1, ridge=0.0)
    fractional = fit_fractional_predictability_subspace(
        moments,
        rank=1,
        ridge=0.0,
        final_ridge=0.0,
    )

    assert float(absolute.basis[0].abs()) > 0.999
    assert float(fractional.basis[1].abs()) > 0.999

    absolute_metrics = subspace_regression_metrics(
        moments, moments, absolute.basis, ridge=0.0
    )
    fractional_metrics = subspace_regression_metrics(
        moments, moments, fractional.basis, ridge=0.0
    )
    assert absolute_metrics["r2"] == pytest.approx(0.25)
    assert fractional_metrics["r2"] == pytest.approx(0.9025)


@pytest.mark.parametrize("mode", ["static", "annealed", "reverse"])
def test_path_target_matches_central_finite_difference(mode: str) -> None:
    generator = torch.Generator().manual_seed(17)
    clean = torch.randn((3, 8, 3, 3), generator=generator, dtype=torch.float64)
    noise = torch.randn((3, 8, 3, 3), generator=generator, dtype=torch.float64)
    basis = random_detail_basis(8, 3, seed=19, dtype=torch.float64)
    time = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64)
    epsilon = 1e-5
    plan = plan_layerwise_path(clean, noise, time, basis, mode=mode, power=2.0)
    plus = plan_layerwise_path(clean, noise, time + epsilon, basis, mode=mode, power=2.0)
    minus = plan_layerwise_path(clean, noise, time - epsilon, basis, mode=mode, power=2.0)
    finite_difference = (plus.state - minus.state) / (2.0 * epsilon)
    torch.testing.assert_close(plan.target, finite_difference, atol=2e-8, rtol=2e-8)


@pytest.mark.parametrize("mode", ["static", "annealed", "reverse"])
def test_all_paths_share_clean_and_noise_endpoints(mode: str) -> None:
    generator = torch.Generator().manual_seed(23)
    clean = torch.randn((2, 6, 4, 4), generator=generator)
    noise = torch.randn((2, 6, 4, 4), generator=generator)
    basis = random_detail_basis(6, 2, seed=29)
    at_data = plan_layerwise_path(
        clean, noise, torch.zeros(2), basis, mode=mode, power=2.0
    )
    at_noise = plan_layerwise_path(
        clean, noise, torch.ones(2), basis, mode=mode, power=2.0
    )
    torch.testing.assert_close(at_data.state, clean, atol=2e-6, rtol=0)
    torch.testing.assert_close(at_noise.state, noise, atol=2e-6, rtol=0)


@pytest.mark.parametrize(
    ("family", "power", "floor", "alpha"),
    [
        ("power", 1.0, 0.05, 1.0),
        ("power", 2.0, 0.2, 1.0),
        ("rational", 2.0, 0.15, 0.5),
    ],
)
def test_floored_path_target_matches_finite_difference(
    family: str, power: float, floor: float, alpha: float
) -> None:
    generator = torch.Generator().manual_seed(101)
    clean = torch.randn((3, 8, 3, 3), generator=generator, dtype=torch.float64)
    noise = torch.randn((3, 8, 3, 3), generator=generator, dtype=torch.float64)
    basis = random_detail_basis(8, 3, seed=103, dtype=torch.float64)
    time = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64)
    epsilon = 1e-5
    kwargs = {
        "mode": "annealed",
        "power": power,
        "family": family,
        "floor": floor,
        "alpha": alpha,
    }
    plan = plan_layerwise_path(clean, noise, time, basis, **kwargs)
    plus = plan_layerwise_path(clean, noise, time + epsilon, basis, **kwargs)
    minus = plan_layerwise_path(clean, noise, time - epsilon, basis, **kwargs)
    finite_difference = (plus.state - minus.state) / (2.0 * epsilon)
    torch.testing.assert_close(plan.target, finite_difference, atol=2e-8, rtol=2e-8)
