from __future__ import annotations

import torch

from experiments.rae_spectral_direction_loss import DCTDirectionLoss, radial_band_index


MOMENTS = [8.0280, 0.6731, 0.4064, 0.3170, 0.2710, 0.2494, 0.2304, 0.2087]


def test_radial_bands_are_equal_cardinality():
    index = radial_band_index(16, 8)
    counts = torch.bincount(index.flatten(), minlength=8)
    assert counts.tolist() == [32] * 8
    assert sorted(index.unique().tolist()) == list(range(8))


def test_gamma_zero_is_official_mse_under_parseval():
    generator = torch.Generator().manual_seed(7)
    prediction = torch.randn((3, 5, 16, 16), generator=generator)
    target = torch.randn((3, 5, 16, 16), generator=generator)
    time = torch.tensor([0.55, 0.8, 0.95])
    loss_module = DCTDirectionLoss(16, MOMENTS, gamma=0.0)

    loss, details = loss_module(prediction, target, time)
    official = (prediction - target).square().flatten(1).mean(dim=1)

    assert torch.allclose(loss_module.weights(time), torch.ones((3, 8)), atol=2e-6, rtol=0)
    assert torch.allclose(details["raw_mse"], official, atol=0, rtol=0)
    assert torch.allclose(loss, official, atol=2e-6, rtol=2e-6)


def test_partial_weights_are_bounded_and_coefficient_mean_one():
    loss_module = DCTDirectionLoss(
        16,
        MOMENTS,
        gamma=0.5,
        damping=1e-4,
        min_weight=0.2,
        max_weight=2.0,
    )
    time = torch.linspace(0.01, 0.99, 101)
    weights = loss_module.weights(time)
    coefficient_mean = (weights * loss_module.band_counts[None]).sum(1) / loss_module.band_counts.sum()

    assert torch.all(weights >= 0.2 - 1e-6)
    assert torch.all(weights <= 2.0 + 1e-6)
    assert torch.allclose(coefficient_mean, torch.ones_like(coefficient_mean), atol=2e-6, rtol=0)


def test_partial_weighting_reallocates_budget_from_low_to_high_frequency():
    loss_module = DCTDirectionLoss(16, MOMENTS, gamma=0.5)
    weights = loss_module.weights(torch.tensor([0.8]))[0]
    assert weights[0] < 1.0
    assert weights[-1] > 1.0
    assert torch.all(weights[1:] >= weights[:-1])


def test_forward_preserves_gradient_flow_only_through_prediction():
    prediction = torch.randn((2, 3, 16, 16), requires_grad=True)
    target = torch.randn_like(prediction)
    time = torch.tensor([0.7, 0.9])
    loss_module = DCTDirectionLoss(16, MOMENTS, gamma=0.5)
    loss, _ = loss_module(prediction, target, time)
    loss.mean().backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
