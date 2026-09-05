from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.audit_raev2_ig_within_time_structure import (  # noqa: E402
    as_tokens,
    pearson_per_sample,
    spatial_cv,
    token_cosine,
    token_rms,
)


def test_token_helpers_preserve_spatial_layout() -> None:
    value = torch.tensor(
        [[[[3.0, 0.0]], [[4.0, 5.0]]]],
    )
    tokens = as_tokens(value)
    assert tokens.shape == (1, 2, 2)
    torch.testing.assert_close(tokens[0, 0], torch.tensor([3.0, 4.0]))
    torch.testing.assert_close(tokens[0, 1], torch.tensor([0.0, 5.0]))
    torch.testing.assert_close(
        token_rms(value), torch.tensor([[math_sqrt_12p5(), math_sqrt_12p5()]])
    )


def math_sqrt_12p5() -> float:
    return 12.5**0.5


def test_token_cosine_is_computed_across_channels() -> None:
    left = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    right = torch.tensor([[[[2.0, 1.0]], [[0.0, 0.0]]]])
    cosine = token_cosine(left, right)
    torch.testing.assert_close(cosine, torch.tensor([[1.0, 0.0]]))


def test_spatial_cv_and_pearson_are_per_sample() -> None:
    values = torch.tensor([[1.0, 1.0, 1.0], [1.0, 2.0, 3.0]])
    assert float(spatial_cv(values)[0]) == 0.0
    correlation = pearson_per_sample(values, values.flip(1))
    assert float(correlation[0]) == 0.0
    torch.testing.assert_close(correlation[1], torch.tensor(-1.0))
