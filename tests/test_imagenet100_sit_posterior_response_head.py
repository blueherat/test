from __future__ import annotations

import torch

from experiments.imagenet100_sit_posterior_response_head import (
    diagonal_response_action,
    finite_difference_clean_response_action,
    rademacher_probe_like,
)


class LinearClean(torch.nn.Module):
    def __init__(self, matrix: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("matrix", matrix)
        self.calls = 0

    def forward(
        self,
        state: torch.Tensor,
        time_value: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        del time_value, labels
        self.calls += 1
        return state @ self.matrix.T


def test_batched_clean_response_recovers_linear_action() -> None:
    matrix = torch.tensor(
        [[1.0, 0.2, 0.0], [0.2, 0.7, -0.1], [0.0, -0.1, 0.4]]
    )
    model = LinearClean(matrix)
    state = torch.randn(5, 3)
    direction = torch.randn_like(state)
    alpha = torch.linspace(0.2, 0.8, len(state))
    actual = finite_difference_clean_response_action(
        model,
        state=state,
        time_value=torch.linspace(0.1, 0.9, len(state)),
        labels=torch.arange(len(state)),
        direction=direction,
        alpha=alpha,
        relative_step=0.01,
    )
    expected = alpha[:, None] * (direction @ matrix.T)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    assert model.calls == 1


def test_diagonal_action_and_probe_contract() -> None:
    direction = torch.randn(7, 4, 3, 3)
    gain = torch.full_like(direction, 0.25)
    torch.testing.assert_close(
        diagonal_response_action(gain, direction),
        0.25 * direction,
    )
    probe = rademacher_probe_like(direction)
    assert torch.equal(probe.square(), torch.ones_like(probe))
