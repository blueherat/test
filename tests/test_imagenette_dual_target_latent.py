from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.imagenette_dual_target_latent import (
    DualTargetLatentUNet,
    clean_from_noise,
    extrapolated_clean,
    prediction_losses,
    velocity_from_clean,
    velocity_from_noise,
)


def test_prediction_conversions_are_exact_for_oracles() -> None:
    generator = torch.Generator().manual_seed(7)
    clean = torch.randn((3, 4, 8, 8), generator=generator)
    noise = torch.randn((3, 4, 8, 8), generator=generator)
    time_value = torch.tensor([0.1, 0.5, 0.9])
    state = (1.0 - time_value[:, None, None, None]) * clean + time_value[
        :, None, None, None
    ] * noise
    target_velocity = noise - clean

    assert torch.allclose(clean_from_noise(state, time_value, noise), clean, atol=1e-5)
    assert torch.allclose(
        velocity_from_clean(state, time_value, clean), target_velocity, atol=1e-5
    )
    assert torch.allclose(
        velocity_from_noise(state, time_value, noise), target_velocity, atol=1e-5
    )


def test_extrapolation_endpoints_match_clean_heads() -> None:
    generator = torch.Generator().manual_seed(11)
    clean_prediction = torch.randn((2, 4, 8, 8), generator=generator)
    noise_prediction = torch.randn((2, 4, 8, 8), generator=generator)
    state = torch.randn((2, 4, 8, 8), generator=generator)
    time_value = torch.tensor([0.25, 0.75])
    converted_noise = clean_from_noise(state, time_value, noise_prediction)

    gamma_zero = extrapolated_clean(
        state, time_value, clean_prediction, noise_prediction, gamma=0.0
    )
    gamma_minus_one = extrapolated_clean(
        state, time_value, clean_prediction, noise_prediction, gamma=-1.0
    )
    assert torch.equal(gamma_zero, clean_prediction)
    assert torch.allclose(gamma_minus_one, converted_noise, atol=1e-6)


def test_oracle_heads_have_zero_common_velocity_loss() -> None:
    generator = torch.Generator().manual_seed(17)
    clean = torch.randn((4, 4, 8, 8), generator=generator)
    noise = torch.randn((4, 4, 8, 8), generator=generator)
    time_value = torch.linspace(0.1, 0.9, len(clean))
    state = (1.0 - time_value[:, None, None, None]) * clean + time_value[
        :, None, None, None
    ] * noise
    losses = prediction_losses(
        state=state,
        clean=clean,
        noise=noise,
        time_value=time_value,
        clean_prediction=clean,
        noise_prediction=noise,
        loss_space="v",
    )
    assert float(losses["loss"]) < 1e-10
    assert float(losses["x_v"]) < 1e-10
    assert float(losses["eps_v"]) < 1e-10


def test_shared_model_returns_two_equal_shape_heads() -> None:
    model = DualTargetLatentUNet(base_channels=32)
    value = torch.randn((2, 4, 8, 8))
    time_value = torch.tensor([0.2, 0.8])
    labels = torch.tensor([1, 7])
    clean, noise = model(value, time_value, labels)
    assert clean.shape == value.shape
    assert noise.shape == value.shape
    assert clean.data_ptr() != noise.data_ptr()
