import torch

from experiments.rae_frequency_time_switch_probe import (
    FREQUENCY_SCHEDULES,
    blend_velocity_bands,
    frequency_switched_endpoint,
    schedule_spec,
)
from experiments.rae_spectral_direction_loss import DCTDirectionLoss


class ConstantVelocity(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = float(value)

    def forward(self, state, time, y):
        del time, y
        return torch.full_like(state, self.value)


def test_frequency_schedule_windows_and_bands():
    assert schedule_spec("baseline", 0.9, 8) == (False, ())
    assert schedule_spec("partial_high_band0", 0.9, 8) == (True, (0,))
    assert schedule_spec("partial_high_band7", 0.9, 8) == (True, (7,))
    assert schedule_spec("partial_high_band0", 0.8, 8) == (False, (0,))
    assert schedule_spec("partial_mid_nonzero", 0.5, 4) == (True, (1, 2, 3))
    assert len(FREQUENCY_SCHEDULES) == 7


def test_velocity_blending_selects_exact_dct_bands():
    analyzer = DCTDirectionLoss(4, torch.ones(4), gamma=0.0)
    baseline = torch.randn(2, 3, 4, 4, generator=torch.Generator().manual_seed(3))
    partial = torch.randn(2, 3, 4, 4, generator=torch.Generator().manual_seed(5))
    blended = blend_velocity_bands(baseline, partial, (0, 2), analyzer)
    blended_coefficients = analyzer.transform(blended)
    baseline_coefficients = analyzer.transform(baseline)
    partial_coefficients = analyzer.transform(partial)
    for band in range(4):
        mask = analyzer.band_index == band
        expected = partial_coefficients[:, :, mask] if band in (0, 2) else baseline_coefficients[:, :, mask]
        torch.testing.assert_close(
            blended_coefficients[:, :, mask], expected, atol=2e-6, rtol=2e-6
        )


def test_frequency_switched_endpoint_uses_time_window():
    analyzer = DCTDirectionLoss(1, [1.0], gamma=0.0)
    baseline = ConstantVelocity(1.0)
    partial = ConstantVelocity(3.0)
    noise = torch.zeros(2, 1, 1, 1)
    labels = torch.zeros(2, dtype=torch.long)
    times = torch.tensor([1.0, 0.8, 0.2, 0.0])
    endpoint = frequency_switched_endpoint(
        baseline,
        partial,
        noise,
        labels,
        times,
        analyzer,
        "partial_high_band0",
    )
    torch.testing.assert_close(endpoint, torch.full_like(noise, -1.4))
