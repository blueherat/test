from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from experiments.run_prediction_target_extrapolation_toy_v3 import (
    sample_condition,
    sample_training_times,
)


class RejectZeroTimeModel(nn.Module):
    def forward(self, state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        if bool((time <= 0.0).any()):
            raise AssertionError("model evaluated at t=0")
        return torch.zeros_like(state)


def test_jit_logit_normal_is_expressed_in_noise_time() -> None:
    args = SimpleNamespace(
        time_sampler="logit_normal",
        time_logit_mean=0.8,
        time_logit_std=0.8,
        t_min=1e-4,
        t_max=1.0 - 1e-4,
    )
    generator = torch.Generator().manual_seed(17)
    time = sample_training_times(
        20000,
        args=args,
        device=torch.device("cpu"),
        generator=generator,
    )
    assert 0.66 < float(time.mean()) < 0.70


def test_continuous_toy_heun_uses_euler_at_data_endpoint() -> None:
    args = SimpleNamespace(
        sample_t_max=1.0,
        sample_t_min=0.0,
        sample_steps=4,
        conversion_clip=0.05,
        sampler="heun_state",
    )
    model = RejectZeroTimeModel()
    samples = sample_condition(
        models={"x": model, "v": model, "eps": model},
        D=2,
        condition="x",
        gamma=0.0,
        n=8,
        args=args,
        device=torch.device("cpu"),
        seed=23,
    )
    assert samples.shape == (8, 2)
    assert bool(torch.isfinite(samples).all())
