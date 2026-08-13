from __future__ import annotations

import torch

from experiments.sample_imagenet100_sit_frozen_guidance_fid import frozen_derivative


def test_frozen_derivative_uses_current_anchor_and_baseline_gap() -> None:
    anchor_baseline = torch.tensor([[2.0, 3.0]])
    anchor_frozen = torch.tensor([[5.0, 7.0]])
    other_baseline = torch.tensor([[1.0, -1.0]])
    actual = frozen_derivative(
        anchor_baseline, anchor_frozen, other_baseline, gamma=0.5
    )
    expected = anchor_frozen + 0.5 * (anchor_baseline - other_baseline)
    torch.testing.assert_close(actual, expected)


def test_zero_gamma_is_unguided_current_state_anchor() -> None:
    anchor_baseline = torch.randn(2, 3)
    anchor_frozen = torch.randn(2, 3)
    other_baseline = torch.randn(2, 3)
    torch.testing.assert_close(
        frozen_derivative(anchor_baseline, anchor_frozen, other_baseline, gamma=0.0),
        anchor_frozen,
    )
