from __future__ import annotations

import torch

from experiments.imagenet100_sit_error_triangulated_guidance import (
    TARGETS,
    fuse_predictions,
    guided_field,
    regularize_private_variances,
    three_cornered_hat,
    time_bin_index,
)


def test_three_cornered_hat_recovers_uncorrelated_scalar_variances() -> None:
    expected = torch.tensor([[2.0], [3.0], [5.0]])
    recovered = three_cornered_hat(
        {
            "velocity_clean": expected[0] + expected[1],
            "velocity_epsilon": expected[0] + expected[2],
            "clean_epsilon": expected[1] + expected[2],
        }
    )
    torch.testing.assert_close(recovered, expected)


def test_regularized_inverse_variance_weights_favor_reliable_head() -> None:
    raw = torch.tensor([[1.0], [4.0], [9.0]])
    regularized, weights = regularize_private_variances(
        raw,
        shrinkage=0.0,
        ridge_fraction=0.0,
    )
    torch.testing.assert_close(regularized, raw)
    assert weights[0] > weights[1] > weights[2]
    torch.testing.assert_close(weights.sum(dim=-2), torch.ones(1))


def test_negative_variance_is_preserved_then_stabilized() -> None:
    raw = three_cornered_hat(
        {
            "velocity_clean": torch.tensor([1.0]),
            "velocity_epsilon": torch.tensor([1.0]),
            "clean_epsilon": torch.tensor([3.0]),
        }
    )
    assert raw[0, 0] < 0
    regularized, weights = regularize_private_variances(
        raw,
        shrinkage=0.05,
        ridge_fraction=0.01,
    )
    assert torch.all(regularized > 0)
    torch.testing.assert_close(weights.sum(dim=-2), torch.ones(1))


def test_fuse_predictions_supports_global_and_channel_weights() -> None:
    predictions = {
        target: torch.full((2, 4, 3, 3), float(index + 1))
        for index, target in enumerate(TARGETS)
    }
    global_fused = fuse_predictions(predictions, torch.tensor([0.5, 0.25, 0.25]))
    torch.testing.assert_close(global_fused, torch.full_like(global_fused, 1.75))

    channel_weights = torch.zeros(3, 4)
    channel_weights[torch.tensor([0, 1, 2, 0]), torch.arange(4)] = 1.0
    channel_fused = fuse_predictions(predictions, channel_weights)
    expected = torch.tensor([1.0, 2.0, 3.0, 1.0]).reshape(1, 4, 1, 1)
    torch.testing.assert_close(channel_fused, expected.expand_as(channel_fused))


def test_guided_field_private_residual_uses_head_minus_common() -> None:
    full = torch.full((1, 1, 1, 1), 10.0)
    weak = {
        "velocity": torch.full_like(full, 1.0),
        "clean": torch.full_like(full, 2.0),
        "epsilon": torch.full_like(full, 3.0),
    }
    weights = torch.tensor([0.5, 0.25, 0.25])
    common = 1.75
    etg = guided_field(full, weak, mode="etg", gamma=0.5, weights=weights)
    private = guided_field(
        full,
        weak,
        mode="private",
        gamma=0.5,
        weights=weights,
        private_target="epsilon",
    )
    torch.testing.assert_close(etg, torch.full_like(full, 10 + 0.5 * (10 - common)))
    torch.testing.assert_close(private, torch.full_like(full, 10 + 0.5 * (3 - common)))


def test_time_bin_index_handles_endpoints() -> None:
    edges = [0.0, 0.2, 0.7, 1.0]
    assert time_bin_index(0.0, edges) == 0
    assert time_bin_index(0.2, edges) == 1
    assert time_bin_index(0.999, edges) == 2
    assert time_bin_index(1.0, edges) == 2
