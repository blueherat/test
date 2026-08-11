from __future__ import annotations

import torch
import torch.nn as nn

from experiments.evaluate_prediction_target_autoguidance import (
    condition_clean,
    sample_condition,
)
from experiments.run_prediction_target_bayes_oracle_v5 import (
    TangentGaussianMixture,
)


class ConstantModel(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = float(value)

    def forward(self, state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        del time
        return torch.full_like(state, self.value)


class ConstantTwoHeadModel(nn.Module):
    def __init__(self, intermediate: float, final: float) -> None:
        super().__init__()
        self.intermediate = float(intermediate)
        self.final = float(final)

    def forward(
        self, state: torch.Tensor, time: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del time
        return (
            torch.full_like(state, self.intermediate),
            torch.full_like(state, self.final),
        )


class RejectZeroTimeModel(nn.Module):
    def forward(self, state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        if bool((time <= 0.0).any()):
            raise AssertionError("model evaluated at the singular endpoint")
        return torch.zeros_like(state)


def test_same_target_autoguidance_uses_weak_plus_weighted_gap() -> None:
    state = torch.zeros(4, 2)
    time = torch.full((4,), 0.5)
    output = condition_clean(
        condition="ag_early",
        weight=2.0,
        state=state,
        time=time,
        mixture=None,  # type: ignore[arg-type]
        strong_x=ConstantModel(2.0),
        strong_v=ConstantModel(99.0),
        strong_eps=ConstantModel(99.0),
        weak_early_x=ConstantModel(1.0),
        weak_small_x=ConstantModel(-1.0),
        internal_model=None,
        internal_v_model=None,
        conversion_clip=0.02,
    )
    assert torch.equal(output, torch.full_like(state, 3.0))


def test_autoguidance_weight_one_is_exactly_the_strong_model() -> None:
    state = torch.zeros(4, 2)
    time = torch.full((4,), 0.5)
    output = condition_clean(
        condition="ag_small",
        weight=1.0,
        state=state,
        time=time,
        mixture=None,  # type: ignore[arg-type]
        strong_x=ConstantModel(2.0),
        strong_v=ConstantModel(99.0),
        strong_eps=ConstantModel(99.0),
        weak_early_x=ConstantModel(1.0),
        weak_small_x=ConstantModel(-1.0),
        internal_model=None,
        internal_v_model=None,
        conversion_clip=0.02,
    )
    assert torch.equal(output, torch.full_like(state, 2.0))


def test_reverse_prediction_target_guidance_extrapolates_from_x_to_v() -> None:
    state = torch.zeros(4, 2)
    time = torch.full((4,), 0.5)
    output = condition_clean(
        condition="ptg_reverse",
        weight=2.0,
        state=state,
        time=time,
        mixture=None,  # type: ignore[arg-type]
        strong_x=ConstantModel(2.0),
        # Raw v=2 maps to clean x_t-t*v=-1 for x_t=0 and t=0.5.
        strong_v=ConstantModel(2.0),
        strong_eps=ConstantModel(99.0),
        weak_early_x=ConstantModel(99.0),
        weak_small_x=ConstantModel(99.0),
        internal_model=None,
        internal_v_model=None,
        conversion_clip=0.02,
    )
    # weak clean=2, strong clean=-1, so 2+2*(-1-2)=-4.
    assert torch.equal(output, torch.full_like(state, -4.0))


def test_guidance_window_returns_strong_prediction_outside_interval() -> None:
    state = torch.zeros(2, 2)
    time = torch.tensor([0.2, 0.5])
    output = condition_clean(
        condition="ag_early",
        weight=2.0,
        state=state,
        time=time,
        mixture=None,  # type: ignore[arg-type]
        strong_x=ConstantModel(2.0),
        strong_v=ConstantModel(99.0),
        strong_eps=ConstantModel(99.0),
        weak_early_x=ConstantModel(1.0),
        weak_small_x=ConstantModel(-1.0),
        internal_model=None,
        internal_v_model=None,
        conversion_clip=0.02,
        guidance_t_min=0.3,
        guidance_t_max=1.0,
    )
    assert torch.equal(output[0], torch.full_like(output[0], 2.0))
    assert torch.equal(output[1], torch.full_like(output[1], 3.0))


def test_internal_v_head_is_converted_to_clean_before_extrapolation() -> None:
    state = torch.full((4, 2), 4.0)
    time = torch.full((4,), 0.5)
    output = condition_clean(
        condition="ig_v",
        weight=2.0,
        state=state,
        time=time,
        mixture=None,  # type: ignore[arg-type]
        strong_x=ConstantModel(99.0),
        strong_v=ConstantModel(99.0),
        strong_eps=ConstantModel(99.0),
        weak_early_x=ConstantModel(99.0),
        weak_small_x=ConstantModel(99.0),
        internal_model=None,
        internal_v_model=ConstantTwoHeadModel(2.0, 5.0),  # type: ignore[arg-type]
        conversion_clip=0.02,
    )
    # v=2 maps to clean x=4-0.5*2=3; 3+2*(5-3)=7.
    assert torch.equal(output, torch.full_like(state, 7.0))


def test_noise_target_is_converted_to_clean_before_extrapolation() -> None:
    state = torch.full((4, 2), 4.0)
    time = torch.full((4,), 0.5)
    output = condition_clean(
        condition="ptg_eps",
        weight=2.0,
        state=state,
        time=time,
        mixture=None,  # type: ignore[arg-type]
        strong_x=ConstantModel(5.0),
        strong_v=ConstantModel(99.0),
        strong_eps=ConstantModel(2.0),
        weak_early_x=ConstantModel(99.0),
        weak_small_x=ConstantModel(99.0),
        internal_model=None,
        internal_v_model=None,
        conversion_clip=0.02,
    )
    # eps=2 maps to clean x=(4-0.5*2)/(1-0.5)=6; 6+2*(5-6)=4.
    assert torch.equal(output, torch.full_like(state, 4.0))


def test_heun_sampler_uses_final_euler_without_evaluating_t_zero() -> None:
    device = torch.device("cpu")
    mixture = TangentGaussianMixture(
        D=4,
        components=3,
        curvature=0.2,
        frequency_scale=3.0,
        center_rms=0.6,
        sigma_tangent=0.2,
        sigma_normal=0.03,
        seed=11,
        device=device,
    )
    model = RejectZeroTimeModel()
    samples = sample_condition(
        condition="x",
        weight=1.0,
        sample_count=8,
        batch_size=4,
        steps=4,
        t_max=1.0,
        t_min=0.0,
        seed=31,
        mixture=mixture,
        strong_x=model,
        strong_v=model,
        strong_eps=model,
        weak_early_x=model,
        weak_small_x=model,
        internal_model=None,
        internal_v_model=None,
        conversion_clip=0.05,
        sampler="heun_state",
        initial_state="gaussian_approx",
    )
    assert samples.shape == (8, 4)
    assert torch.isfinite(torch.from_numpy(samples)).all()
