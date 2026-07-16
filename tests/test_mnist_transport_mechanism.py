from __future__ import annotations

import torch

from experiments.mnist_transport_mechanism import (
    band_cross_mean,
    blend_velocity_bands,
    calibrate_band_energy,
    hutchinson_divergence,
    hybrid_rollout_state,
    reference_band_drift,
    reference_band_energy,
)
from experiments.rae_spectral_direction_loss import DCTDirectionLoss


def test_reference_band_drift_matches_finite_difference():
    clean_energy = torch.tensor([0.2, 1.0, 3.0])
    time = 0.43
    epsilon = 1e-4
    finite_difference = (
        reference_band_energy(clean_energy, time + epsilon)
        - reference_band_energy(clean_energy, time - epsilon)
    ) / (2.0 * epsilon)
    torch.testing.assert_close(
        finite_difference,
        reference_band_drift(clean_energy, time),
        atol=8e-4,
        rtol=8e-4,
    )


def test_band_cross_mean_preserves_global_inner_product():
    analyzer = DCTDirectionLoss(4, [1.0, 1.0, 1.0, 1.0], gamma=0.0)
    first = torch.randn(3, 2, 4, 4, generator=torch.Generator().manual_seed(3))
    second = torch.randn(3, 2, 4, 4, generator=torch.Generator().manual_seed(5))
    cross = band_cross_mean(first, second, analyzer)
    weighted = (cross * analyzer.band_counts[None]).sum(dim=1) / analyzer.band_counts.sum()
    expected = (first * second).flatten(1).mean(dim=1)
    torch.testing.assert_close(weighted, expected, atol=2e-6, rtol=2e-6)


def test_band_energy_calibration_hits_each_target():
    analyzer = DCTDirectionLoss(8, torch.ones(8), gamma=0.0)
    state = torch.randn(32, 3, 8, 8, generator=torch.Generator().manual_seed(7))
    target = torch.linspace(0.2, 2.0, 8)
    calibrated = calibrate_band_energy(state, target, analyzer)
    actual = analyzer.band_mse(calibrated).mean(dim=0)
    torch.testing.assert_close(actual, target, atol=3e-6, rtol=3e-6)


class LinearVelocity(torch.nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = float(scale)

    def forward(self, state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        del time
        return self.scale * state


def test_hutchinson_divergence_is_exact_for_scalar_linear_field():
    state = torch.randn(5, 1, 4, 4)
    estimate = hutchinson_divergence(
        LinearVelocity(0.37), state, 0.5, probes=1, seed=11
    )
    torch.testing.assert_close(estimate, torch.full((5,), 0.37), atol=1e-7, rtol=0)


def test_hybrid_rollout_uses_selected_vector_field():
    baseline = LinearVelocity(0.0)
    weighted = LinearVelocity(1.0)
    initial = torch.ones(2, 1, 1, 1)
    times = torch.tensor([1.0, 0.5, 0.0])
    final = hybrid_rollout_state(
        baseline,
        weighted,
        initial,
        times,
        batch_size=2,
        use_weighted=lambda time: time < 0.75,
    )
    torch.testing.assert_close(final, torch.full_like(final, 0.5))


def test_velocity_band_blending_uses_exact_selected_coefficients():
    analyzer = DCTDirectionLoss(8, torch.ones(4), gamma=0.0)
    baseline = torch.randn(2, 1, 8, 8, generator=torch.Generator().manual_seed(17))
    weighted = torch.randn(2, 1, 8, 8, generator=torch.Generator().manual_seed(19))
    blended = blend_velocity_bands(baseline, weighted, (0, 2), analyzer)
    blended_coefficients = analyzer.transform(blended)
    baseline_coefficients = analyzer.transform(baseline)
    weighted_coefficients = analyzer.transform(weighted)
    for band in range(4):
        mask = analyzer.band_index == band
        expected = weighted_coefficients[:, :, mask] if band in (0, 2) else baseline_coefficients[:, :, mask]
        torch.testing.assert_close(
            blended_coefficients[:, :, mask], expected, atol=2e-6, rtol=2e-6
        )
