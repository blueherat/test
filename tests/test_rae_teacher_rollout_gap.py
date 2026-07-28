import math

import pytest
import torch
from omegaconf import OmegaConf

from experiments.rae_spectral_direction_loss import DCTDirectionLoss
from experiments.rae_teacher_rollout_gap import (
    _infer_decoder_input_channels,
    band_prediction_calibration,
    clean_from_velocity,
    euler_rollout,
    fixed_gaussian_matrix,
    latent_summary,
    official_time_grid,
    select_time_indices,
    sliced_wasserstein,
)


class ConstantVelocity(torch.nn.Module):
    def __init__(self, velocity: float):
        super().__init__()
        self.velocity = float(velocity)

    def forward(self, state, time, y):
        del time, y
        return torch.full_like(state, self.velocity)


def test_official_time_grid_matches_shifted_sampler_contract():
    times = official_time_grid()
    assert times.shape == (50,)
    assert torch.all(times[:-1] > times[1:])
    assert math.isclose(float(times[0]), 1.0, abs_tol=1e-7)
    assert math.isclose(float(times[-1]), 0.006886, rel_tol=2e-3)
    selected = select_time_indices(times, (0.95, 0.55, 0.0))
    assert selected == sorted(set(selected))
    assert selected[-1] == 49


def test_clean_estimate_is_exact_on_linear_flow_path():
    generator = torch.Generator().manual_seed(7)
    clean = torch.randn((3, 4, 2, 2), generator=generator)
    noise = torch.randn((3, 4, 2, 2), generator=generator)
    time = torch.tensor((0.2, 0.5, 0.9))
    expanded = time.reshape(-1, 1, 1, 1)
    state = (1.0 - expanded) * clean + expanded * noise
    velocity = noise - clean
    estimate = clean_from_velocity(state, velocity, time)
    torch.testing.assert_close(estimate, clean)


def test_euler_rollout_uses_descending_time_increment():
    model = ConstantVelocity(2.0)
    initial = torch.zeros((2, 1, 1, 1))
    labels = torch.zeros(2, dtype=torch.long)
    times = torch.tensor((1.0, 0.6, 0.1))
    states = euler_rollout(model, initial, labels, times)
    torch.testing.assert_close(states[-1], torch.full_like(initial, -1.8))


def test_sliced_wasserstein_detects_shift_and_is_zero_for_identity():
    generator = torch.Generator().manual_seed(11)
    reference = torch.randn((64, 6), generator=generator)
    directions = fixed_gaussian_matrix(6, 24, 13)
    identity = sliced_wasserstein(reference, reference.clone(), directions)
    shifted = sliced_wasserstein(reference, reference + 0.5, directions)
    assert float(identity) == 0.0
    assert float(shifted) > 0.05


def test_latent_summary_combines_channel_and_frequency_statistics():
    analyzer = DCTDirectionLoss(4, [1.0, 0.8, 0.6, 0.4], gamma=0.0)
    latent = torch.randn((5, 8, 4, 4), generator=torch.Generator().manual_seed(17))
    projection = fixed_gaussian_matrix(8, 3, 19)
    summary = latent_summary(latent, analyzer, projection)
    assert summary.shape == (5, 7)
    assert torch.isfinite(summary).all()


def test_band_calibration_is_exact_for_perfect_prediction():
    analyzer = DCTDirectionLoss(4, [1.0, 0.8, 0.6, 0.4], gamma=0.0)
    target = torch.randn((3, 5, 4, 4), generator=torch.Generator().manual_seed(23))
    metrics = band_prediction_calibration(target.clone(), target, analyzer)
    torch.testing.assert_close(
        metrics["prediction_energy_log_ratio_to_target"],
        torch.zeros((3, 4)),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        metrics["prediction_target_cosine"], torch.ones((3, 4)), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        metrics["prediction_target_slope"], torch.ones((3, 4)), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        metrics["velocity_error_mse"], torch.zeros((3, 4)), atol=1e-7, rtol=0
    )


def test_decoder_channels_follow_latent_statistics_when_not_configured() -> None:
    stage_1 = OmegaConf.create({"params": {"encoder_params": {}}})
    stats = {"mean": torch.zeros(1024, 16, 16)}
    assert _infer_decoder_input_channels(stage_1, stats) == 1024


def test_explicit_decoder_channels_take_precedence() -> None:
    stage_1 = OmegaConf.create({"params": {"encoder_params": {"hidden_size": 768}}})
    stats = {"mean": torch.zeros(1024, 16, 16)}
    assert _infer_decoder_input_channels(stage_1, stats) == 768


def test_decoder_channels_require_a_reliable_source() -> None:
    stage_1 = OmegaConf.create({"params": {"encoder_params": {}}})
    with pytest.raises(ValueError, match="cannot infer decoder input channels"):
        _infer_decoder_input_channels(stage_1, {})
