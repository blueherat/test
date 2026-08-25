from __future__ import annotations

import torch

from experiments.frechet_residual_score_toy import build_toy_regimes
from experiments.advfd_feature_pullback import (
    build_feature_force_context,
    feature_force_and_jacobian,
    learned_pullback_field,
)
from experiments.run_frechet_residual_score_toy import CriticConfig, FeatureCritic
from experiments.run_frechet_residual_score_toy import advfd_functional_field


def test_pseudoinverse_minimizes_local_feature_tracking_residual() -> None:
    torch.manual_seed(7)
    target, source = build_toy_regimes()["shape_only"]
    critic = FeatureCritic(
        CriticConfig(hidden_dim=16, depth=2, feature_dim=4)
    ).to(dtype=torch.float64)
    critic.eval().requires_grad_(False)
    context = build_feature_force_context(
        critic, target, source, order=8, whitening_epsilon=1e-3
    )
    states, _ = source.quadrature(order=6)
    force, jacobian = feature_force_and_jacobian(critic, context, states)
    transpose = learned_pullback_field(
        critic, context, mode="transpose"
    )(states, False)
    pseudoinverse = learned_pullback_field(
        critic, context, mode="pseudoinverse", relative_damping=0.0
    )(states, False)
    transpose_residual = torch.einsum("nfi,ni->nf", jacobian, transpose) - force
    pseudoinverse_residual = (
        torch.einsum("nfi,ni->nf", jacobian, pseudoinverse) - force
    )
    assert pseudoinverse_residual.square().sum() <= transpose_residual.square().sum()
    assert torch.isfinite(pseudoinverse).all()


def test_transpose_pullback_matches_existing_advfd_field() -> None:
    torch.manual_seed(11)
    target, source = build_toy_regimes()["shape_only"]
    critic = FeatureCritic(
        CriticConfig(hidden_dim=16, depth=2, feature_dim=4)
    ).to(dtype=torch.float64)
    critic.eval().requires_grad_(False)
    context = build_feature_force_context(
        critic, target, source, order=10, whitening_epsilon=1e-3
    )
    new_field = learned_pullback_field(
        critic, context, mode="transpose"
    )
    existing_field, distance = advfd_functional_field(
        critic,
        target,
        source,
        order=10,
        whitening_epsilon=1e-3,
        objective_mode="official_regularized",
    )
    states, _ = source.quadrature(order=8)
    torch.testing.assert_close(
        new_field(states, False),
        existing_field(states, False),
        atol=1e-9,
        rtol=1e-9,
    )
    assert abs(context["distance"] - distance) < 1e-10
