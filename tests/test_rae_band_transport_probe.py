import torch

from experiments.rae_band_transport_probe import band_cross_mean, selected_rollout_states
from experiments.rae_spectral_direction_loss import DCTDirectionLoss


class ConstantVelocity(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = float(value)

    def forward(self, state, time, y):
        del time, y
        return torch.full_like(state, self.value)


def test_band_cross_mean_matches_spatial_inner_product():
    analyzer = DCTDirectionLoss(4, torch.ones(4), gamma=0.0)
    first = torch.randn(3, 2, 4, 4, generator=torch.Generator().manual_seed(7))
    second = torch.randn(3, 2, 4, 4, generator=torch.Generator().manual_seed(11))
    cross = band_cross_mean(first, second, analyzer)
    weighted = (cross * analyzer.band_counts[None]).sum(dim=1) / analyzer.band_counts.sum()
    torch.testing.assert_close(
        weighted, (first * second).flatten(1).mean(dim=1), atol=2e-6, rtol=2e-6
    )


def test_selected_rollout_states_returns_preupdate_states():
    model = ConstantVelocity(2.0)
    noise = torch.zeros(2, 1, 1, 1)
    labels = torch.zeros(2, dtype=torch.long)
    times = torch.tensor([1.0, 0.6, 0.1])
    states = selected_rollout_states(model, noise, labels, times, {0, 1, 2})
    torch.testing.assert_close(states[0], torch.zeros_like(noise))
    torch.testing.assert_close(states[1], torch.full_like(noise, -0.8))
    torch.testing.assert_close(states[2], torch.full_like(noise, -1.8))
