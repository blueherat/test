from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.pfr_ou_semigroup_controls import (
    OUSpectralControlField,
    energy_adaptive_ou_revision,
    project_onto_sample_span,
    select_ou_spectral_revision,
    split_raw_revision_against_ou_degree1,
    strong_anchored_angular_guidance,
)


def test_ou_common_and_unique_exactly_recompose_raw_revision() -> None:
    generator = torch.Generator().manual_seed(23)
    raw = torch.randn(4, 3, 2, 2, generator=generator)
    spectral = torch.randn(4, 3, 2, 2, generator=generator)

    split = split_raw_revision_against_ou_degree1(raw, spectral)

    torch.testing.assert_close(split.common + split.unique, raw)
    dot = (
        split.unique.float().flatten(1)
        * spectral.float().flatten(1)
    ).sum(1)
    torch.testing.assert_close(dot, torch.zeros_like(dot), atol=3e-6, rtol=0.0)


def test_strong_anchored_angular_guidance_preserves_extrapolation_norm() -> None:
    generator = torch.Generator().manual_seed(29)
    strong = torch.randn(4, 3, 2, 2, generator=generator)
    weak = torch.randn(4, 3, 2, 2, generator=generator)
    revision = torch.randn(4, 3, 2, 2, generator=generator)
    gamma = 0.7

    actual = strong_anchored_angular_guidance(
        strong, weak, revision, gamma=gamma
    )
    expected_norm = gamma * (strong - weak).float().flatten(1).norm(dim=1)
    actual_norm = (actual - strong).float().flatten(1).norm(dim=1)
    torch.testing.assert_close(actual_norm, expected_norm, rtol=1e-5, atol=1e-5)


def test_strong_anchored_angular_guidance_discards_parallel_revision() -> None:
    generator = torch.Generator().manual_seed(31)
    strong = torch.randn(4, 3, 2, 2, generator=generator)
    weak = torch.randn(4, 3, 2, 2, generator=generator)
    gap = strong - weak
    gamma = 0.6

    actual = strong_anchored_angular_guidance(
        strong, weak, 3.0 * gap, gamma=gamma
    )
    expected = strong + gamma * gap
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_ou_revision_selection_and_first_segment_splice() -> None:
    raw = torch.tensor([[2.0, 1.0]])
    spectral = torch.tensor([[1.0, 0.0]])
    split = split_raw_revision_against_ou_degree1(raw, spectral)

    torch.testing.assert_close(
        select_ou_spectral_revision(split, "ou_d1_common", time_value=0.4),
        torch.tensor([[2.0, 0.0]]),
    )
    torch.testing.assert_close(
        select_ou_spectral_revision(split, "ou_d1_unique", time_value=0.1),
        torch.tensor([[0.0, 1.0]]),
    )
    torch.testing.assert_close(
        select_ou_spectral_revision(
            split, "ou_d1_common_first", time_value=0.3
        ),
        raw,
    )
    torch.testing.assert_close(
        select_ou_spectral_revision(
            split, "ou_d1_unique_first", time_value=0.1
        ),
        torch.tensor([[0.0, 1.0]]),
    )
    torch.testing.assert_close(
        select_ou_spectral_revision(
            split,
            "ou_d1_common_norm_raw_direction_first",
            time_value=0.1,
        ),
        torch.tensor([[1.7888544, 0.8944272]]),
    )
    torch.testing.assert_close(
        select_ou_spectral_revision(
            split,
            "ou_d1_common_direction_raw_norm_first",
            time_value=0.1,
        ),
        torch.tensor([[2.236068, 0.0]]),
    )
    for condition in (
        "ou_d1_common_norm_raw_direction_first",
        "ou_d1_common_direction_raw_norm_first",
    ):
        torch.testing.assert_close(
            select_ou_spectral_revision(split, condition, time_value=0.3),
            raw,
        )
    torch.testing.assert_close(
        select_ou_spectral_revision(
            split, "ou_d2_common_first", time_value=0.1
        ),
        torch.tensor([[2.0, 0.0]]),
    )
    torch.testing.assert_close(
        select_ou_spectral_revision(
            split, "ou_d2_unique_first", time_value=0.3
        ),
        raw,
    )
    with pytest.raises(ValueError, match="unknown OU spectral control"):
        select_ou_spectral_revision(split, "missing", time_value=0.1)


def test_energy_adaptive_revision_has_exact_limiting_cases() -> None:
    partially_aligned = split_raw_revision_against_ou_degree1(
        torch.tensor([[2.0, 1.0]]), torch.tensor([[1.0, 0.0]])
    )
    torch.testing.assert_close(
        energy_adaptive_ou_revision(partially_aligned),
        torch.tensor([[2.0, 0.2]]),
    )

    aligned = split_raw_revision_against_ou_degree1(
        torch.tensor([[2.0, 0.0]]), torch.tensor([[1.0, 0.0]])
    )
    orthogonal = split_raw_revision_against_ou_degree1(
        torch.tensor([[0.0, 3.0]]), torch.tensor([[1.0, 0.0]])
    )
    torch.testing.assert_close(energy_adaptive_ou_revision(aligned), aligned.raw)
    torch.testing.assert_close(
        energy_adaptive_ou_revision(orthogonal), orthogonal.raw
    )


def test_project_onto_sample_span_recovers_each_sample_plane() -> None:
    value = torch.tensor([[2.0, -3.0, 7.0], [4.0, 5.0, -6.0]])
    first = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    second = torch.tensor([[1.0, 1.0, 0.0], [2.0, -1.0, 0.0]])

    projected = project_onto_sample_span(value, (first, second))

    torch.testing.assert_close(
        projected,
        torch.tensor([[2.0, -3.0, 0.0], [4.0, 5.0, 0.0]]),
        atol=2e-5,
        rtol=0.0,
    )


def test_project_onto_sample_span_handles_collinear_references() -> None:
    value = torch.tensor([[2.0, -3.0, 7.0]])
    first = torch.tensor([[1.0, 2.0, 0.0]])
    second = 3.0 * first

    projected = project_onto_sample_span(value, (first, second))

    torch.testing.assert_close(
        projected,
        torch.tensor([[-0.8, -1.6, 0.0]]),
        atol=2e-5,
        rtol=0.0,
    )


class _CountingRuntime:
    def __init__(self) -> None:
        self.pair_calls = 0
        self.weak_calls = 0

    def evaluate_pair(
        self, time: torch.Tensor, state: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del labels
        self.pair_calls += 1
        return state + 2.0 * time, state - time

    def evaluate_weak(
        self, time: torch.Tensor, state: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        del labels
        self.weak_calls += 1
        return state - time


@pytest.mark.parametrize(
    "condition",
    [
        "ou_d1_strong_common_first",
        "ou_d1_strong_unique_first",
        "ou_d1_strong_common_norm_raw_direction_first",
        "ou_d1_strong_common_direction_raw_norm_first",
        "ou_d1_strong_anchored_common_direction_raw_norm_first",
        "ou_d1_strong_anchored_angular_first",
        "ou_d2_strong_common_first",
        "ou_d2_strong_common_direction_raw_norm_first",
    ],
)
def test_strong_certificate_uses_one_full_ou_query_only_in_first_segment(
    condition: str,
) -> None:
    runtime = _CountingRuntime()
    field = OUSpectralControlField(runtime, torch.tensor([0]), condition)
    state = torch.tensor([[1.0, 2.0]])

    early = field(torch.tensor(0.1), state)

    assert torch.isfinite(early).all()
    assert runtime.pair_calls == 2
    assert runtime.weak_calls == 1
    assert field.query_nfe == 1
    assert field.full_query_nfe == 1

    later = field(torch.tensor(0.3), state)

    assert torch.isfinite(later).all()
    assert runtime.pair_calls == 3
    assert runtime.weak_calls == 2
    assert field.query_nfe == 2
    assert field.full_query_nfe == 1


def test_strong_certificate_common_and_unique_recompose_raw_revision() -> None:
    state = torch.tensor([[1.0, 2.0]])
    labels = torch.tensor([0])
    time = torch.tensor(0.1)
    common = OUSpectralControlField(
        _CountingRuntime(), labels, "ou_d1_strong_common_first"
    )(time, state)
    unique = OUSpectralControlField(
        _CountingRuntime(), labels, "ou_d1_strong_unique_first"
    )(time, state)

    runtime = _CountingRuntime()
    strong, weak = runtime.evaluate_pair(time, state, labels)
    gamma = 0.6
    guided = strong + gamma * (strong - weak)
    future_weak = runtime.evaluate_weak(time + 1.0 / 32.0, state, labels)
    raw_revision = weak - future_weak

    torch.testing.assert_close(
        (common - guided) + (unique - guided),
        (1.0 + gamma) * raw_revision,
        atol=2e-5,
        rtol=0.0,
    )


def test_strong_certificate_direction_and_norm_controls_exchange_only_one_factor(
) -> None:
    state = torch.tensor([[1.0, 2.0]])
    labels = torch.tensor([0])
    time = torch.tensor(0.1)
    runtime = _CountingRuntime()
    strong, weak = runtime.evaluate_pair(time, state, labels)
    guided = strong + 0.6 * (strong - weak)

    common_norm = OUSpectralControlField(
        _CountingRuntime(),
        labels,
        "ou_d1_strong_common_norm_raw_direction_first",
    )(time, state)
    common = OUSpectralControlField(
        _CountingRuntime(), labels, "ou_d1_strong_common_first"
    )(time, state)
    common_direction = OUSpectralControlField(
        _CountingRuntime(),
        labels,
        "ou_d1_strong_common_direction_raw_norm_first",
    )(time, state)
    norm_revision = (common_norm - guided) / 1.6
    common_revision = (common - guided) / 1.6
    direction_revision = (common_direction - guided) / 1.6

    future_weak = runtime.evaluate_weak(time + 1.0 / 32.0, state, labels)
    raw_revision = weak - future_weak
    raw_direction_cosine = torch.nn.functional.cosine_similarity(
        norm_revision, raw_revision
    )
    torch.testing.assert_close(raw_direction_cosine, torch.ones_like(raw_direction_cosine))
    common_direction_cosine = torch.nn.functional.cosine_similarity(
        direction_revision, common_revision
    )
    torch.testing.assert_close(
        common_direction_cosine, torch.ones_like(common_direction_cosine)
    )
    torch.testing.assert_close(
        norm_revision.square().mean(1),
        common_revision.square().mean(1),
        atol=2e-5,
        rtol=0.0,
    )
    torch.testing.assert_close(
        direction_revision.square().mean(1),
        raw_revision.square().mean(1),
        atol=2e-5,
        rtol=0.0,
    )


def test_strong_anchored_ou_revision_scales_with_extrapolation_only() -> None:
    state = torch.tensor([[1.0, 2.0]])
    labels = torch.tensor([0])
    time = torch.tensor(0.1)
    runtime = _CountingRuntime()
    strong, weak = runtime.evaluate_pair(time, state, labels)
    gamma = 0.6
    guided = strong + gamma * (strong - weak)

    ordinary = OUSpectralControlField(
        _CountingRuntime(),
        labels,
        "ou_d1_strong_common_direction_raw_norm_first",
    )(time, state)
    anchored = OUSpectralControlField(
        _CountingRuntime(),
        labels,
        "ou_d1_strong_anchored_common_direction_raw_norm_first",
    )(time, state)

    torch.testing.assert_close(
        anchored - guided,
        gamma / (1.0 + gamma) * (ordinary - guided),
        atol=2e-5,
        rtol=0.0,
    )
