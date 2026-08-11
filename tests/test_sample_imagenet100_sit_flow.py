from __future__ import annotations

import sys
from pathlib import Path

import torch


EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from sample_imagenet100_sit_flow import (  # noqa: E402
    official_cfg_velocity,
    parse_float_list,
    parse_int_list,
)


class FakeSiT(torch.nn.Module):
    def forward_with_cfg(self, x, t, y, cfg_scale):
        half = len(x) // 2
        cond = torch.zeros_like(x[:half])
        uncond = torch.zeros_like(x[:half])
        cond[:, :3] = 5.0
        cond[:, 3:] = 7.0
        uncond[:, :3] = 1.0
        uncond[:, 3:] = 2.0
        guided = uncond[:, :3] + cfg_scale * (cond[:, :3] - uncond[:, :3])
        first = torch.cat([guided, cond[:, 3:]], dim=1)
        second = torch.cat([guided, uncond[:, 3:]], dim=1)
        return torch.cat([first, second], dim=0)


def test_official_cfg_velocity_guides_only_first_three_channels() -> None:
    labels = torch.tensor([2, 3])
    velocity, counter = official_cfg_velocity(
        FakeSiT(), labels, 1.5, autocast_dtype=None
    )
    state = torch.randn(2, 4, 2, 2)
    prediction = velocity(torch.tensor(0.4), state)
    assert torch.allclose(prediction[:, :3], torch.full_like(prediction[:, :3], 7.0))
    assert torch.allclose(prediction[:, 3:], torch.full_like(prediction[:, 3:], 7.0))
    assert counter == {"nfe": 1}


def test_cli_list_parsers() -> None:
    assert parse_float_list("1,1.5,4") == [1.0, 1.5, 4.0]
    assert parse_int_list("0,6,99") == [0, 6, 99]
