from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.pfr_retiming_controls import (
    rms_match_per_sample,
    select_retiming_revision,
    split_weak_retiming_against_strong,
)


def _sample_rms(value: torch.Tensor) -> torch.Tensor:
    return value.float().flatten(1).square().mean(1).sqrt()


def test_weak_common_and_unique_exactly_recompose_revision() -> None:
    generator = torch.Generator().manual_seed(71)
    weak = torch.randn(4, 3, 2, 2, generator=generator)
    strong = torch.randn(4, 3, 2, 2, generator=generator)

    split = split_weak_retiming_against_strong(weak, strong)

    torch.testing.assert_close(split.weak_common + split.weak_unique, weak)
    dot = (
        split.weak_unique.float().flatten(1)
        * strong.float().flatten(1)
    ).sum(1)
    torch.testing.assert_close(dot, torch.zeros_like(dot), atol=2e-6, rtol=0.0)


def test_rms_matched_strong_has_weak_revision_scale() -> None:
    value = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    reference = torch.tensor([[2.0, -1.0], [0.5, -0.25]])

    matched = rms_match_per_sample(value, reference)

    torch.testing.assert_close(_sample_rms(matched), _sample_rms(reference))


def test_zero_strong_revision_stays_finite_and_zero() -> None:
    value = torch.zeros(2, 4)
    reference = torch.ones(2, 4)

    matched = rms_match_per_sample(value, reference)

    assert torch.isfinite(matched).all()
    torch.testing.assert_close(matched, value)


def test_all_retiming_controls_select_the_intended_exact_component() -> None:
    weak = torch.tensor([[2.0, 1.0]])
    strong = torch.tensor([[1.0, 0.0]])
    split = split_weak_retiming_against_strong(weak, strong)

    torch.testing.assert_close(
        select_retiming_revision(split, "weak_time_pair"), weak
    )
    torch.testing.assert_close(
        select_retiming_revision(split, "strong_time"), strong
    )
    torch.testing.assert_close(
        select_retiming_revision(split, "weak_common_strong"),
        torch.tensor([[2.0, 0.0]]),
    )
    torch.testing.assert_close(
        select_retiming_revision(split, "weak_unique_strong"),
        torch.tensor([[0.0, 1.0]]),
    )
    with pytest.raises(ValueError, match="unknown retiming control"):
        select_retiming_revision(split, "missing")
