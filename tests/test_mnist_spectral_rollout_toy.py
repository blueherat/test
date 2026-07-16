from __future__ import annotations

import torch

from experiments.mnist_spectral_rollout_toy import (
    MNISTToyConfig,
    TinyVelocityUNet,
    descending_time_grid,
    estimate_band_second_moments,
    euler_sample,
    frechet_distance,
    shifted_uniform,
    sliced_wasserstein,
    train_paired_velocity_fields,
)
from experiments.rae_spectral_direction_loss import DCTDirectionLoss


def test_descending_time_grid_has_exact_endpoints_and_is_monotone():
    times = descending_time_grid(10, shift=3.0)
    assert times.shape == (11,)
    assert float(times[0]) == 1.0
    assert float(times[-1]) == 0.0
    assert torch.all(times[:-1] > times[1:])


def test_shifted_uniform_stays_inside_unit_interval():
    generator = torch.Generator().manual_seed(3)
    time = shifted_uniform(1000, 4.0, device=torch.device("cpu"), generator=generator)
    assert torch.all((time >= 0.0) & (time <= 1.0))
    assert float(time.mean()) > 0.5


def test_tiny_velocity_unet_preserves_image_shape():
    model = TinyVelocityUNet(width=8, depth=1)
    value = torch.randn(3, 1, 28, 28)
    prediction = model(value, torch.tensor([0.1, 0.5, 0.9]))
    assert prediction.shape == value.shape
    assert torch.isfinite(prediction).all()


def test_band_moment_estimator_is_positive_and_has_expected_shape():
    clean = torch.randn(17, 1, 28, 28)
    moments = estimate_band_second_moments(clean, band_count=7, batch_size=5)
    assert moments.shape == (7,)
    assert torch.all(moments > 0.0)


def test_gamma_zero_paired_training_is_numerically_identical():
    clean = torch.randn(8, 1, 28, 28, generator=torch.Generator().manual_seed(5))
    moments = estimate_band_second_moments(clean, band_count=4)
    analyzer = DCTDirectionLoss(28, moments.tolist(), gamma=0.0)
    config = MNISTToyConfig(
        train_size=8,
        test_size=8,
        sample_count=8,
        batch_size=4,
        steps=1,
        width=8,
        depth=1,
        gamma=0.0,
        device="cpu",
        save=False,
    )
    torch.manual_seed(7)
    models, _ = train_paired_velocity_fields(clean, config, analyzer)
    for baseline, weighted in zip(
        models["baseline"].state_dict().values(), models["weighted"].state_dict().values()
    ):
        torch.testing.assert_close(baseline, weighted, atol=2e-6, rtol=2e-6)


class ConstantVelocity(torch.nn.Module):
    def forward(self, value: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        del time
        return torch.ones_like(value)


def test_euler_sample_uses_descending_increment():
    initial = torch.zeros(2, 1, 2, 2)
    times = torch.tensor([1.0, 0.6, 0.0])
    sample = euler_sample(ConstantVelocity(), initial, times, batch_size=1)
    torch.testing.assert_close(sample, torch.full_like(sample, -1.0))


def test_sliced_wasserstein_detects_a_shift():
    reference = torch.randn(64, 5, generator=torch.Generator().manual_seed(11))
    directions = torch.eye(5)
    assert sliced_wasserstein(reference, reference.clone(), directions) == 0.0
    assert sliced_wasserstein(reference, reference + 1.0, directions) > 0.9


def test_frechet_distance_is_stable_for_rank_deficient_features():
    reference = torch.randn(16, 32, generator=torch.Generator().manual_seed(13))
    identity = frechet_distance(reference, reference.clone())
    shifted = frechet_distance(reference, reference + 0.5)
    assert identity < 1e-8
    assert shifted > 7.9
