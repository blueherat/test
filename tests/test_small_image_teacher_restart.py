from pathlib import Path

import torch
from torch import nn

from experiments.small_image_teacher_restart import (
    TeacherRestartConfig,
    _field_variant,
    integrate_interval,
)


class ConstantField(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = float(value)

    def forward(self, state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return torch.full_like(state, self.value)


def test_middle_restart_uses_only_the_registered_window():
    config = TeacherRestartConfig(
        study_dir=Path("."),
        devices=("cpu",),
        sample_count=2,
        batch_size=2,
        ode_steps_per_unit=10,
        restart_time=0.7,
        middle_threshold=0.3,
    )
    assert _field_variant("middle", 0.7, config) == "baseline"
    assert _field_variant("middle", 0.6, config) == "weighted"
    assert _field_variant("middle", 0.3, config) == "weighted"
    assert _field_variant("middle", 0.2, config) == "baseline"

    models = {"baseline": ConstantField(0.0), "weighted": ConstantField(1.0)}
    initial = torch.zeros((2, 1, 1, 1))
    baseline = integrate_interval(
        models, initial, 0.7, 0.0, schedule="baseline", config=config
    )
    middle = integrate_interval(
        models, initial, 0.7, 0.0, schedule="middle", config=config
    )

    assert torch.allclose(baseline, torch.zeros_like(initial))
    assert torch.allclose(middle, torch.full_like(initial, -0.4), atol=1e-6)
