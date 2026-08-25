from __future__ import annotations

import torch

from experiments.advfd_feature_pullback import (
    build_feature_force_context,
    feature_potential,
    learned_pullback_field,
)
from experiments.run_advfd_smoothed_retraining_transport import (
    build_rotated_ring_pair,
)
from experiments.run_frechet_residual_score_toy import CriticConfig, FeatureCritic
from experiments.run_frechet_residual_score_toy import (
    advfd_from_raw_moments,
)
from experiments.frechet_residual_score_toy import weighted_moments


def test_feature_potential_gradient_matches_transpose_field() -> None:
    torch.manual_seed(7)
    target, source = build_rotated_ring_pair(rotation=0.2, device="cpu")
    config = CriticConfig(feature_dim=3, objective_mode="official_regularized")
    critic = FeatureCritic(config).to(dtype=torch.float64).eval()
    for parameter in critic.parameters():
        parameter.requires_grad_(False)
    context = build_feature_force_context(
        critic,
        target,
        source,
        order=8,
        whitening_epsilon=1e-3,
    )
    states = source.sample(11, seed=13).requires_grad_(True)
    potential = feature_potential(critic, context, states)
    expected = -torch.autograd.grad(potential.sum(), states)[0]
    actual = learned_pullback_field(
        critic, context, mode="transpose"
    )(states.detach(), False)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


def test_mean_and_covariance_contexts_zero_unused_gradient() -> None:
    torch.manual_seed(11)
    target, source = build_rotated_ring_pair(rotation=0.2, device="cpu")
    config = CriticConfig(feature_dim=3)
    critic = FeatureCritic(config).to(dtype=torch.float64).eval()
    for parameter in critic.parameters():
        parameter.requires_grad_(False)
    mean_context = build_feature_force_context(
        critic,
        target,
        source,
        order=8,
        whitening_epsilon=1e-3,
        objective_mode="official_mean_only",
    )
    covariance_context = build_feature_force_context(
        critic,
        target,
        source,
        order=8,
        whitening_epsilon=1e-3,
        objective_mode="official_covariance_only",
    )
    torch.testing.assert_close(
        mean_context["second_gradient"],
        torch.zeros_like(mean_context["second_gradient"]),
    )
    torch.testing.assert_close(
        covariance_context["mean_gradient"],
        torch.zeros_like(covariance_context["mean_gradient"]),
    )


def test_pooled_context_matches_pooled_forward_objective() -> None:
    torch.manual_seed(17)
    target, source = build_rotated_ring_pair(rotation=0.2, device="cpu")
    config = CriticConfig(feature_dim=3)
    critic = FeatureCritic(config).to(dtype=torch.float64).eval()
    for parameter in critic.parameters():
        parameter.requires_grad_(False)
    target_points, target_weights = target.quadrature(8)
    source_points, source_weights = source.quadrature(8)
    with torch.no_grad():
        target_raw = weighted_moments(critic(target_points), target_weights)
        source_raw = weighted_moments(critic(source_points), source_weights)
        expected, _ = advfd_from_raw_moments(
            target_raw,
            source_raw,
            whitening_epsilon=1e-3,
            objective_mode="pooled_full",
        )
    context = build_feature_force_context(
        critic,
        target,
        source,
        order=8,
        whitening_epsilon=1e-3,
        objective_mode="pooled_full",
    )
    torch.testing.assert_close(
        expected.new_tensor(context["distance"]),
        expected,
        atol=1e-10,
        rtol=1e-10,
    )
