from __future__ import annotations

import pytest
import torch

from experiments.frechet_residual_score_toy import (
    build_toy_regimes,
    pearson_divergence,
    pearson_field,
)
from experiments.run_advfd_fisher_gap_audit import (
    fisher_value,
    reference_diagnostics,
    train_scalar_critic,
)
from experiments.run_frechet_residual_score_toy import CriticConfig, FeatureCritic


def _flatten_gradients(gradients) -> torch.Tensor:
    return torch.cat([gradient.flatten() for gradient in gradients])


def test_stopgrad_and_rayleigh_have_same_value_but_different_gradient() -> None:
    target, source = build_toy_regimes()["shape_only"]
    torch.manual_seed(11)
    critic = FeatureCritic(
        CriticConfig(hidden_dim=12, depth=2, feature_dim=1)
    ).to(dtype=torch.float64)
    stopped = fisher_value(
        critic,
        target,
        source,
        order=8,
        epsilon=1e-3,
        stopgrad_real=True,
    )
    stopped_gradient = _flatten_gradients(
        torch.autograd.grad(stopped, tuple(critic.parameters()))
    )
    rayleigh = fisher_value(
        critic,
        target,
        source,
        order=8,
        epsilon=1e-3,
        stopgrad_real=False,
    )
    rayleigh_gradient = _flatten_gradients(
        torch.autograd.grad(rayleigh, tuple(critic.parameters()))
    )
    assert float(stopped.detach()) == pytest.approx(
        float(rayleigh.detach()), abs=1e-12
    )
    assert not torch.allclose(stopped_gradient, rayleigh_gradient)


def test_pearson_field_is_its_own_reference() -> None:
    target, source = build_toy_regimes()["shape_only"]
    diagnostics = reference_diagnostics(
        target,
        source,
        pearson_field(target, source),
        order=12,
    )
    assert diagnostics["pearson_cosine"] == pytest.approx(1.0, abs=1e-10)
    assert diagnostics["pearson_derivative"] < 0


def test_scalar_critic_smoke_runs_all_training_modes() -> None:
    target, source = build_toy_regimes()["shape_only"]
    exact = pearson_divergence(target, source, quadrature_order=10)
    for mode in ("official_stopgrad", "fisher_rayleigh", "supervised_ratio"):
        critic, curve = train_scalar_critic(
            target,
            source,
            mode=mode,
            seed=5,
            steps=2,
            learning_rate=5e-4,
            epsilon=1e-3,
            quadrature_order=8,
            device=torch.device("cpu"),
        )
        assert len(curve) == 3
        assert curve[-1]["fisher_value"] >= 0
        assert exact > 0
        assert critic(torch.zeros(2, 2, dtype=torch.float64)).shape == (2, 1)
