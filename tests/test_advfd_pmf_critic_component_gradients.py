import pytest
import torch

from experiments.advfd_cleanroom.audit_pmf_critic_component_gradients import (
    gradient_trial,
)


class TinyCritic(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(4, 3, bias=False)

    def forward(self, images: torch.Tensor):
        features = self.projection(images.flatten(1))
        return features, None


def test_component_gradients_add_to_full_gradient() -> None:
    torch.manual_seed(41)
    critic = TinyCritic().double()
    real_images = torch.randn(8, 1, 2, 2, dtype=torch.float64)
    fake_images = torch.randn(8, 1, 2, 2, dtype=torch.float64)
    anchor_mean = torch.randn(3, dtype=torch.float64)
    anchor_matrix = torch.randn(3, 3, dtype=torch.float64)
    anchor_covariance = anchor_matrix @ anchor_matrix.mT + 0.5 * torch.eye(3)

    result = gradient_trial(
        critic,
        real_images,
        fake_images,
        (anchor_mean, anchor_covariance),
        (anchor_mean + 0.2, anchor_covariance * 1.1),
        ema_beta=0.9,
        epsilon=1e-3,
    )

    assert result["raw_components"]["full"] == pytest.approx(
        result["raw_components"]["mean"]
        + result["raw_components"]["covariance"],
        abs=1e-7,
    )
    assert result["full_gradient_additivity_relative_error"] < 1e-6
    for value in result["gradient_norms"].values():
        assert value > 0.0
