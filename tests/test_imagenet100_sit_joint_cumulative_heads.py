from __future__ import annotations

import torch
import torch.nn as nn

from experiments.imagenet100_sit_joint_cumulative_heads import (
    CumulativeReadoutStack,
    select_joint_cumulative_field,
    sequence_losses,
)


class ConstantHead(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.tensor(float(value)))

    def forward(self, hidden: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        del conditioning
        return self.value.expand(len(hidden), 4, 1)


class TinySource(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.x_embedder = nn.Module()
        self.x_embedder.patch_size = (1, 1)


def make_stack() -> CumulativeReadoutStack:
    return CumulativeReadoutStack(
        nn.ModuleDict({"d1": ConstantHead(1.0), "d2": ConstantHead(2.0), "d3": ConstantHead(-0.5)}),
        depths=(1, 2, 3),
        source=TinySource(),
        latent_channels=1,
    )


def test_outputs_are_exact_cumulative_sums() -> None:
    stack = make_stack()
    features = tuple(torch.zeros(2, 4, 3) for _ in range(3))
    conditioning = torch.zeros(2, 3)

    outputs, innovations = stack(features, conditioning)

    torch.testing.assert_close(innovations[0], torch.ones_like(innovations[0]))
    torch.testing.assert_close(outputs[0], torch.ones_like(outputs[0]))
    torch.testing.assert_close(outputs[1], torch.full_like(outputs[1], 3.0))
    torch.testing.assert_close(outputs[2], torch.full_like(outputs[2], 2.5))


def test_all_heads_receive_joint_gradient_from_later_losses() -> None:
    stack = make_stack()
    features = tuple(torch.zeros(2, 4, 3) for _ in range(3))
    conditioning = torch.zeros(2, 3)
    outputs, _ = stack(features, conditioning)
    target = torch.zeros_like(outputs[0])

    losses = sequence_losses(
        outputs, target, monotonic_weight=0.2, contraction_ratio=0.9
    )
    losses["optimized"].backward()

    for head in stack.heads.values():
        assert head.value.grad is not None
        assert torch.isfinite(head.value.grad)


def test_monotonic_penalty_detects_worsening_stage() -> None:
    target = torch.zeros(2, 1, 2, 2)
    improving = (
        torch.full_like(target, 2.0),
        torch.full_like(target, 1.0),
        torch.full_like(target, 0.25),
    )
    worsening = (
        torch.full_like(target, 1.0),
        torch.full_like(target, 2.0),
        torch.full_like(target, 0.5),
    )

    good = sequence_losses(
        improving, target, monotonic_weight=1.0, contraction_ratio=1.0
    )
    bad = sequence_losses(
        worsening, target, monotonic_weight=1.0, contraction_ratio=1.0
    )

    assert float(good["monotonic"]) == 0.0
    assert float(good["strict_monotonic_fraction"]) == 1.0
    assert float(bad["monotonic"]) > 0.0
    assert float(bad["strict_monotonic_fraction"]) == 0.0


def test_joint_field_selection_uses_exact_last_increment() -> None:
    strong = torch.full((2, 1, 2, 2), 9.0)
    outputs = (
        torch.full_like(strong, 1.0),
        torch.full_like(strong, 3.0),
        torch.full_like(strong, 2.5),
    )
    innovations = (
        torch.full_like(strong, 1.0),
        torch.full_like(strong, 2.0),
        torch.full_like(strong, -0.5),
    )

    torch.testing.assert_close(
        select_joint_cumulative_field(
            strong, outputs, innovations, mode="strong", gamma=0.0
        ),
        strong,
    )
    torch.testing.assert_close(
        select_joint_cumulative_field(
            strong, outputs, innovations, mode="final", gamma=0.0
        ),
        outputs[-1],
    )
    torch.testing.assert_close(
        select_joint_cumulative_field(
            strong,
            outputs,
            innovations,
            mode="stage",
            gamma=0.0,
            stage_index=1,
        ),
        outputs[1],
    )
    extrapolated = select_joint_cumulative_field(
        strong, outputs, innovations, mode="last_extrapolation", gamma=3.0
    )
    torch.testing.assert_close(extrapolated, outputs[-1] + 3.0 * innovations[-1])
    torch.testing.assert_close(innovations[-1], outputs[-1] - outputs[-2])
