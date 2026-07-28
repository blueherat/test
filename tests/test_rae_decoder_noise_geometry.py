from __future__ import annotations

import torch

from experiments.rae_decoder_noise_geometry import (
    hidden_deviation_profile,
    matched_noise_geometry,
    relative_cycle_error,
    sample_rms,
)


def test_matched_noise_geometry_controls_both_coordinate_norms() -> None:
    generator = torch.Generator().manual_seed(7)
    noise = torch.randn(4, 3, 5, 5, generator=generator)
    std = torch.linspace(0.2, 2.0, 75).reshape(3, 5, 5)
    severity = torch.tensor([0.1, 0.3, 0.5, 0.8])
    result = matched_noise_geometry(noise, std, severity)

    torch.testing.assert_close(sample_rms(result.raw_sphere_raw), severity)
    torch.testing.assert_close(
        sample_rms(result.stage2_sphere_raw_matched_raw), severity
    )
    torch.testing.assert_close(
        sample_rms(result.stage2_sphere_normalized_matched),
        sample_rms(result.raw_sphere_normalized),
    )


def test_coordinate_geometries_coincide_for_constant_variance() -> None:
    noise = torch.randn(2, 4, 3, 3, generator=torch.Generator().manual_seed(11))
    std = torch.full((4, 3, 3), 1.7)
    severity = torch.tensor([0.2, 0.6])
    result = matched_noise_geometry(noise, std, severity)

    torch.testing.assert_close(
        result.raw_sphere_normalized,
        result.stage2_sphere_raw_matched_normalized,
    )
    torch.testing.assert_close(
        result.raw_sphere_normalized,
        result.stage2_sphere_normalized_matched,
    )


def test_hidden_deviation_and_cycle_metrics() -> None:
    reference = (torch.zeros(2, 3, 4), torch.zeros(2, 3, 4))
    candidate = (torch.ones(2, 3, 4), torch.full((2, 3, 4), 2.0))
    deviation, gain = hidden_deviation_profile(candidate, reference)
    torch.testing.assert_close(deviation, torch.tensor([[1.0, 2.0], [1.0, 2.0]]))
    torch.testing.assert_close(gain, torch.full((2, 1), 2.0))

    latent = torch.ones(2, 3, 2, 2)
    cycle = latent + 0.25
    torch.testing.assert_close(relative_cycle_error(cycle, latent), torch.full((2,), 0.25))
