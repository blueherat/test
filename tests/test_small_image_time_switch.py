from pathlib import Path

import torch
from torch import nn

from experiments.small_image_time_switch import (
    TimeSwitchConfig,
    schedule_variant,
    time_switch_sample,
)


class ConstantField(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = float(value)

    def forward(self, state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return torch.full_like(state, self.value)


def test_schedule_boundaries_partition_the_time_grid():
    config = TimeSwitchConfig(study_dir=Path("."), ode_steps=10)
    assert schedule_variant("high", 0.7000000, config) == "weighted"
    assert schedule_variant("middle", 0.7000000, config) == "baseline"
    assert schedule_variant("middle", 0.3000000, config) == "weighted"
    assert schedule_variant("low", 0.3000000, config) == "baseline"
    assert schedule_variant("low", 0.299, config) == "weighted"


def test_window_switches_add_up_to_the_full_weighted_field():
    config = TimeSwitchConfig(
        study_dir=Path("."),
        devices=("cpu",),
        sample_count=2,
        batch_size=2,
        ode_steps=10,
    )
    models = {"baseline": ConstantField(0.0), "weighted": ConstantField(1.0)}
    initial = torch.zeros((2, 1, 1, 1))
    endpoint = {
        name: time_switch_sample(models, initial, name, config)
        for name in ("baseline", "weighted", "high", "middle", "low", "high_middle")
    }

    assert torch.allclose(endpoint["baseline"], torch.zeros_like(initial))
    assert torch.allclose(endpoint["weighted"], -torch.ones_like(initial))
    assert torch.allclose(
        endpoint["high"] + endpoint["middle"] + endpoint["low"],
        endpoint["weighted"],
        atol=1e-6,
    )
    assert torch.allclose(
        endpoint["high_middle"] + endpoint["low"],
        endpoint["weighted"],
        atol=1e-6,
    )
