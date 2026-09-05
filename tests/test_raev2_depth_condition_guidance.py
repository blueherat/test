from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.raev2_depth_condition_guidance import (
    guided_clean_prediction,
    guidance_direction,
    matched_donor_orthogonal,
    minimum_norm_convex_consensus,
    mobius_components,
    orthogonal_residual,
    reconstruct_corner,
)


def _corners() -> tuple[torch.Tensor, ...]:
    base_u = torch.tensor([1.0, -2.0])
    depth = torch.tensor([3.0, 5.0])
    condition = torch.tensor([-7.0, 11.0])
    interaction = torch.tensor([13.0, -17.0])
    full_u = base_u + depth
    base_c = base_u + condition
    full_c = base_u + depth + condition + interaction
    return full_c, base_c, full_u, base_u


def test_mobius_reconstructs_all_four_corners() -> None:
    full_c, base_c, full_u, base_u = _corners()
    components = mobius_components(
        full_conditional=full_c,
        base_conditional=base_c,
        full_unconditional=full_u,
        base_unconditional=base_u,
    )
    torch.testing.assert_close(
        reconstruct_corner(components, depth_coordinate=0.0, condition_coordinate=0.0),
        base_u,
    )
    torch.testing.assert_close(
        reconstruct_corner(components, depth_coordinate=1.0, condition_coordinate=0.0),
        full_u,
    )
    torch.testing.assert_close(
        reconstruct_corner(components, depth_coordinate=0.0, condition_coordinate=1.0),
        base_c,
    )
    torch.testing.assert_close(
        reconstruct_corner(components, depth_coordinate=1.0, condition_coordinate=1.0),
        full_c,
    )


def test_conditional_depth_is_marginal_plus_interaction() -> None:
    full_c, base_c, full_u, base_u = _corners()
    kwargs = dict(
        full_conditional=full_c,
        base_conditional=base_c,
        full_unconditional=full_u,
        base_unconditional=base_u,
    )
    conditional = guidance_direction(**kwargs, mode="conditional_depth")
    marginal = guidance_direction(**kwargs, mode="marginal_depth")
    interaction = guidance_direction(**kwargs, mode="interaction")
    torch.testing.assert_close(conditional, marginal + interaction)


def test_ordinary_ig_scale_convention_is_preserved() -> None:
    full_c, base_c, full_u, base_u = _corners()
    beta = 1.78
    actual = guided_clean_prediction(
        full_conditional=full_c,
        base_conditional=base_c,
        full_unconditional=full_u,
        base_unconditional=base_u,
        guidance_scale=beta,
        mode="conditional_depth",
    )
    expected = base_c + beta * (full_c - base_c)
    torch.testing.assert_close(actual, expected)


def test_marginal_mode_keeps_conditional_strong_anchor() -> None:
    full_c, base_c, full_u, base_u = _corners()
    beta = 1.5
    actual = guided_clean_prediction(
        full_conditional=full_c,
        base_conditional=base_c,
        full_unconditional=full_u,
        base_unconditional=base_u,
        guidance_scale=beta,
        mode="marginal_depth",
    )
    torch.testing.assert_close(actual, full_c + 0.5 * (full_u - base_u))


def test_midpoint_is_depth_plus_half_interaction() -> None:
    full_c, base_c, full_u, base_u = _corners()
    kwargs = dict(
        full_conditional=full_c,
        base_conditional=base_c,
        full_unconditional=full_u,
        base_unconditional=base_u,
    )
    midpoint = guidance_direction(
        **kwargs,
        mode="conditional_marginal_midpoint",
    )
    expected = (full_u - base_u) + 0.5 * (
        full_c - full_u - base_c + base_u
    )
    torch.testing.assert_close(midpoint, expected)


def test_minimum_norm_consensus_has_common_ascent() -> None:
    first = torch.tensor([[1.0, 0.0], [2.0, 0.0], [1.0, 0.0]])
    second = torch.tensor([[0.0, 1.0], [1.0, 0.0], [-1.0, 0.0]])
    consensus, weight = minimum_norm_convex_consensus(first, second)
    torch.testing.assert_close(
        consensus,
        torch.tensor([[0.5, 0.5], [1.0, 0.0], [0.0, 0.0]]),
    )
    torch.testing.assert_close(weight, torch.tensor([0.5, 1.0, 0.5]))
    consensus_norm_sq = consensus.square().sum(dim=1)
    assert torch.all((first * consensus).sum(dim=1) >= consensus_norm_sq)
    assert torch.all((second * consensus).sum(dim=1) >= consensus_norm_sq)


def test_orthogonal_residual_preserves_main_effective_scale() -> None:
    generator = torch.Generator().manual_seed(11)
    conditional = torch.randn(2, 3, 2, 2, generator=generator)
    marginal = torch.randn(2, 3, 2, 2, generator=generator)
    residual = orthogonal_residual(conditional, marginal)
    positive = conditional + residual
    negative = conditional - residual
    conditional_flat = conditional.flatten(1)
    marginal_flat = marginal.flatten(1)
    residual_flat = residual.flatten(1)

    torch.testing.assert_close(
        (conditional_flat * residual_flat).sum(dim=1),
        torch.zeros(2),
        atol=1e-6,
        rtol=0.0,
    )
    baseline_conditional = conditional_flat.square().sum(dim=1)
    torch.testing.assert_close(
        (conditional_flat * positive.flatten(1)).sum(dim=1),
        baseline_conditional,
        atol=1e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        (conditional_flat * negative.flatten(1)).sum(dim=1),
        baseline_conditional,
        atol=1e-6,
        rtol=0.0,
    )
    baseline_marginal = (marginal_flat * conditional_flat).sum(dim=1)
    residual_energy = residual_flat.square().sum(dim=1)
    torch.testing.assert_close(
        (marginal_flat * positive.flatten(1)).sum(dim=1),
        baseline_marginal + residual_energy,
        atol=1e-5,
        rtol=0.0,
    )
    torch.testing.assert_close(
        (marginal_flat * negative.flatten(1)).sum(dim=1),
        baseline_marginal - residual_energy,
        atol=1e-5,
        rtol=0.0,
    )


def test_donor_control_is_orthogonal_and_norm_matched() -> None:
    generator = torch.Generator().manual_seed(17)
    reference = torch.randn(4, 3, 2, 2, generator=generator)
    candidate = torch.randn(4, 3, 2, 2, generator=generator)
    residual = orthogonal_residual(reference, candidate)
    donor = matched_donor_orthogonal(reference, residual)

    torch.testing.assert_close(
        (reference.flatten(1) * donor.flatten(1)).sum(dim=1),
        torch.zeros(4),
        atol=1e-5,
        rtol=0.0,
    )
    torch.testing.assert_close(
        donor.flatten(1).norm(dim=1),
        residual.flatten(1).norm(dim=1),
        atol=1e-5,
        rtol=0.0,
    )


def test_orthogonal_guidance_modes_match_explicit_construction() -> None:
    full_c, base_c, full_u, base_u = _corners()
    kwargs = dict(
        full_conditional=full_c[None],
        base_conditional=base_c[None],
        full_unconditional=full_u[None],
        base_unconditional=base_u[None],
    )
    conditional = kwargs["full_conditional"] - kwargs["base_conditional"]
    marginal = kwargs["full_unconditional"] - kwargs["base_unconditional"]
    residual = orthogonal_residual(conditional, marginal)

    positive = guidance_direction(
        **kwargs,
        mode="conditional_marginal_orthogonal_positive",
    )
    negative = guidance_direction(
        **kwargs,
        mode="conditional_marginal_orthogonal_negative",
    )
    torch.testing.assert_close(positive, conditional + residual)
    torch.testing.assert_close(negative, conditional - residual)


def test_shape_mismatch_and_unknown_mode_fail_closed() -> None:
    full_c, base_c, full_u, base_u = _corners()
    with pytest.raises(ValueError, match="identical shapes"):
        mobius_components(
            full_conditional=full_c,
            base_conditional=base_c[:1],
            full_unconditional=full_u,
            base_unconditional=base_u,
        )
    with pytest.raises(ValueError, match="unknown guidance mode"):
        guidance_direction(
            full_conditional=full_c,
            base_conditional=base_c,
            full_unconditional=full_u,
            base_unconditional=base_u,
            mode="invented",
        )
