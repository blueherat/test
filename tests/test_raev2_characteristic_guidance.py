from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.raev2_characteristic_guidance import (  # noqa: E402
    characteristic_clean_from_shifted_predictions,
    evaluate_first_characteristic_clean,
    first_characteristic_queries,
)


class AffineDualHead(nn.Module):
    def forward(self, state, time, *, context, attn_mask=None):
        del time, attn_mask
        label = context.to(state.dtype).view(-1, *([1] * (state.ndim - 1)))
        return 2.0 * state + label, -state + 3.0 * label


def test_first_iterate_matches_noise_prediction_fixed_point_map() -> None:
    state = torch.tensor([[2.0, -1.0], [4.0, 3.0]])
    time = torch.tensor([0.8, 0.25])
    full = torch.tensor([[1.0, 5.0], [-2.0, 7.0]])
    base = torch.tensor([[3.0, 2.0], [6.0, -1.0]])
    beta = 1.78
    queries = first_characteristic_queries(
        state,
        time,
        full,
        base,
        guidance_scale=beta,
    )

    signal = (1.0 - time).unsqueeze(1)
    noise = time.unsqueeze(1)
    epsilon_full = (state - signal * full) / noise
    epsilon_base = (state - signal * base) / noise
    expected_displacement = noise * (epsilon_base - epsilon_full)
    torch.testing.assert_close(queries.displacement, expected_displacement)
    torch.testing.assert_close(
        queries.full_query,
        state + (beta - 1.0) * expected_displacement,
    )
    torch.testing.assert_close(
        queries.base_query,
        state + beta * expected_displacement,
    )


def test_noise_endpoint_and_equal_heads_reduce_to_ordinary_guidance() -> None:
    state = torch.randn(3, 4)
    full = torch.randn(3, 4)
    beta = 1.78
    endpoint = first_characteristic_queries(
        state,
        torch.ones(3),
        full,
        torch.randn(3, 4),
        guidance_scale=beta,
    )
    torch.testing.assert_close(endpoint.displacement, torch.zeros_like(state))
    torch.testing.assert_close(endpoint.full_query, state)
    torch.testing.assert_close(endpoint.base_query, state)

    equal = first_characteristic_queries(
        state,
        torch.full((3,), 0.7),
        full,
        full,
        guidance_scale=beta,
    )
    torch.testing.assert_close(equal.displacement, torch.zeros_like(state))
    actual = characteristic_clean_from_shifted_predictions(
        full,
        full,
        guidance_scale=beta,
    )
    torch.testing.assert_close(actual, full)


def test_shifted_queries_are_evaluated_in_one_correctly_ordered_batch() -> None:
    state = torch.tensor([[1.0], [2.0]])
    time = torch.tensor([0.8, 0.6])
    labels = torch.tensor([3, 5])
    full = torch.tensor([[4.0], [7.0]])
    base = torch.tensor([[2.0], [-1.0]])
    beta = 1.5
    clean, queries = evaluate_first_characteristic_clean(
        AffineDualHead(),
        state,
        time,
        labels,
        full,
        base,
        guidance_scale=beta,
    )
    shifted_full = 2.0 * queries.full_query + labels[:, None]
    shifted_base = -queries.base_query + 3.0 * labels[:, None]
    expected = beta * shifted_full - (beta - 1.0) * shifted_base
    torch.testing.assert_close(clean, expected)


def test_invalid_scale_and_shape_fail_closed() -> None:
    state = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="guidance_scale"):
        first_characteristic_queries(
            state,
            torch.full((2,), 0.5),
            state,
            state,
            guidance_scale=0.9,
        )
    with pytest.raises(ValueError, match="identical shapes"):
        characteristic_clean_from_shifted_predictions(
            torch.zeros(2, 3),
            torch.zeros(2, 4),
            guidance_scale=1.5,
        )
