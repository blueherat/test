from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.audit_raev2_depth_condition_geometry import component_geometry


def test_component_geometry_recovers_expected_vectors() -> None:
    base_u = torch.tensor([[1.0, 2.0]])
    depth = torch.tensor([[3.0, 0.0]])
    condition = torch.tensor([[0.0, 5.0]])
    interaction = torch.tensor([[0.0, 4.0]])
    metrics = component_geometry(
        full_conditional=base_u + depth + condition + interaction,
        base_conditional=base_u + condition,
        full_unconditional=base_u + depth,
        base_unconditional=base_u,
    )
    assert metrics["cos_depth_interaction"].item() == pytest.approx(0.0)
    assert metrics["conditional_depth_rms"].item() == pytest.approx(
        torch.tensor([[3.0, 4.0]]).square().mean().sqrt().item()
    )
    assert metrics["equal_action_multiplier"].item() == pytest.approx(1.25)
    assert metrics["depth_ascent_under_conditional_depth"].item() == pytest.approx(9.0)
    assert metrics["interaction_ascent_under_conditional_depth"].item() == pytest.approx(16.0)
    assert 0.0 <= metrics["consensus_weight"].item() <= 1.0
    assert metrics["depth_ascent_under_consensus"].item() >= metrics[
        "consensus_rms"
    ].square().item() * depth.numel()
    assert metrics["conditional_depth_ascent_under_consensus"].item() >= metrics[
        "consensus_rms"
    ].square().item() * depth.numel()
    assert metrics["cos_full_cfg_base_cfg"].shape == torch.Size([1])


def test_component_geometry_handles_zero_interaction() -> None:
    zero = torch.zeros(2, 3)
    one = torch.ones(2, 3)
    metrics = component_geometry(
        full_conditional=one,
        base_conditional=zero,
        full_unconditional=one,
        base_unconditional=zero,
    )
    assert torch.equal(metrics["equal_action_multiplier"], torch.zeros(2))
    assert torch.equal(metrics["cos_depth_interaction"], torch.zeros(2))
