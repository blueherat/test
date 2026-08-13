from __future__ import annotations

import torch

from experiments.analyze_imagenet100_sit_future_training_direction import (
    future_alignment_metrics,
)


def test_future_alignment_distinguishes_maturity_and_unrelated_directions() -> None:
    anchor = torch.tensor([[[[1.0, 0.0, 0.0]]]], dtype=torch.float64)
    future = torch.tensor([[[[1.0, 1.0, 0.0]]]], dtype=torch.float64)
    x_other = torch.tensor([[[[1.0, -2.0, 0.0]]]], dtype=torch.float64)
    v_other = torch.tensor([[[[1.0, 0.0, -3.0]]]], dtype=torch.float64)

    metrics = future_alignment_metrics(anchor, future, x_other, v_other)

    torch.testing.assert_close(
        metrics["x400_full_cosine"],
        torch.ones(1, dtype=torch.float64),
    )
    torch.testing.assert_close(
        metrics["x400_full_projection_coefficient"],
        torch.full((1,), 2.0, dtype=torch.float64),
    )
    torch.testing.assert_close(
        metrics["v270_full_cosine"],
        torch.zeros(1, dtype=torch.float64),
    )
    torch.testing.assert_close(
        metrics["x400_orthogonal_cosine"],
        torch.ones(1, dtype=torch.float64),
    )
    torch.testing.assert_close(
        metrics["v270_orthogonal_cosine"],
        torch.zeros(1, dtype=torch.float64),
    )
