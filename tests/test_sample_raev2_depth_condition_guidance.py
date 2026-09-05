from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.sample_raev2_depth_condition_guidance import (  # noqa: E402
    SamplingCondition,
    evaluate_four_corners,
    parse_condition,
)


class FakeDualDepthModel(nn.Module):
    def forward(self, state, times, *, context, attn_mask=None):
        del times, attn_mask
        condition = context.to(state.dtype).view(-1, 1)
        full = state + condition
        base = 2.0 * state - condition
        return full, base


def test_evaluate_four_corners_uses_conditional_then_null_order() -> None:
    state = torch.tensor([[1.0], [3.0]])
    times = torch.tensor([0.8, 0.6])
    labels = torch.tensor([2, 4])
    full_c, base_c, full_u, base_u = evaluate_four_corners(
        FakeDualDepthModel(), state, times, labels, null_label=10
    )
    torch.testing.assert_close(full_c, torch.tensor([[3.0], [7.0]]))
    torch.testing.assert_close(base_c, torch.tensor([[0.0], [2.0]]))
    torch.testing.assert_close(full_u, torch.tensor([[11.0], [13.0]]))
    torch.testing.assert_close(base_u, torch.tensor([[-8.0], [-4.0]]))


def test_high_noise_interval_is_open_at_half() -> None:
    condition = SamplingCondition("test", "conditional_depth")
    assert condition.active_at(1.0)
    assert condition.active_at(0.50001)
    assert not condition.active_at(0.5)
    assert not condition.active_at(0.1)


def test_parser_and_validation_fail_closed() -> None:
    condition = parse_condition("m,marginal_depth,1.78,0.5,1.0")
    assert condition.mode == "marginal_depth"
    with pytest.raises(argparse.ArgumentTypeError):
        parse_condition("bad,unknown,1.78,0.5,1.0")


def test_condition_validation_accepts_orthogonal_controls() -> None:
    for mode in (
        "conditional_marginal_orthogonal_positive",
        "conditional_marginal_orthogonal_negative",
        "conditional_marginal_orthogonal_donor",
    ):
        SamplingCondition("safe_name", mode).validate()


def test_condition_validation_accepts_characteristic_control() -> None:
    condition = parse_condition(
        "characteristic,characteristic_one_step,1.78,0.5,1.0"
    )
    assert condition.mode == "characteristic_one_step"
    assert condition.active_at(0.75)
