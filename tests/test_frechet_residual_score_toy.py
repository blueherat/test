from __future__ import annotations

import pytest
import torch

from experiments.run_frechet_residual_score_toy import (
    CriticConfig,
    FeatureCritic,
    advfd_from_raw_moments,
    advfd_functional_field,
)
from experiments.frechet_residual_score_toy import (
    build_toy_regimes,
    density_ratio,
    field_diagnostics,
    finite_pushforward_kl,
    finite_pushforward_pearson,
    frechet_value,
    pearson_correction,
    pearson_divergence,
    pearson_field,
    pooled_fisher_correction,
    pooled_fisher_divergence,
    project_onto_fixed_moment_tangent,
    score_correction,
    score_field,
    solve_symmetric_lyapunov,
    static_field,
    tangent_field_from_projection,
    weighted_inner,
    weighted_moments,
)


def test_shape_regime_matches_mean_and_covariance_exactly() -> None:
    target, source = build_toy_regimes()["shape_only"]
    torch.testing.assert_close(target.moments().mean, source.moments().mean)
    torch.testing.assert_close(target.moments().covariance, source.moments().covariance)
    assert frechet_value(target, source) < 1e-12


def test_isotropic_convolution_preserves_means_and_adds_covariance() -> None:
    target, _ = build_toy_regimes()["shape_only"]
    smoothed = target.convolve_isotropic(0.4)
    torch.testing.assert_close(smoothed.moments().mean, target.moments().mean)
    torch.testing.assert_close(
        smoothed.moments().covariance,
        target.moments().covariance + 0.4**2 * torch.eye(2, dtype=torch.float64),
    )


def test_symmetric_lyapunov_solver_satisfies_equation() -> None:
    covariance = torch.tensor([[1.4, 0.2], [0.2, 0.7]], dtype=torch.float64)
    right_hand_side = torch.tensor([[0.8, -0.3], [-0.3, 1.1]], dtype=torch.float64)
    solution = solve_symmetric_lyapunov(covariance, right_hand_side)
    torch.testing.assert_close(
        covariance @ solution + solution @ covariance,
        right_hand_side,
        atol=1e-11,
        rtol=1e-11,
    )
    torch.testing.assert_close(solution, solution.mT)


def test_moment_tangent_projection_is_orthogonal_and_preserves_moments() -> None:
    target, source = build_toy_regimes()["combined"]
    states, weights = source.quadrature(order=18)
    correction = score_correction(target, source, states)
    projection = project_onto_fixed_moment_tangent(states, correction, weights)
    assert projection.mean_derivative.norm() < 1e-10
    assert projection.covariance_derivative.norm() < 1e-10
    assert projection.orthogonality_error < 1e-10
    assert abs(float(weighted_inner(projection.tangent, projection.normal, weights))) < 1e-10


def test_gaussian_score_has_no_shape_residual() -> None:
    target, source = build_toy_regimes()["gaussian_only"]
    states, weights = source.quadrature(order=14)
    correction = score_correction(target, source, states)
    projection = project_onto_fixed_moment_tangent(states, correction, weights)
    assert projection.tangent.square().mean().sqrt() < 1e-10


def test_shape_residual_strictly_descends_kl_without_moment_drift() -> None:
    target, source = build_toy_regimes()["shape_only"]
    states, weights = source.quadrature(order=20)
    correction = score_correction(target, source, states)
    projection = project_onto_fixed_moment_tangent(states, correction, weights)
    field = tangent_field_from_projection(target, source, projection)
    diagnostics = field_diagnostics(target, source, field, quadrature_order=20)
    assert diagnostics["reverse_kl_derivative"] < 0
    assert diagnostics["mean_derivative_norm"] < 1e-9
    assert diagnostics["covariance_derivative_norm"] < 1e-9


def test_full_score_and_shape_score_reduce_kl_for_small_finite_step() -> None:
    for regime in ("shape_only", "combined"):
        target, source = build_toy_regimes()[regime]
        states, weights = source.quadrature(order=18)
        correction = score_correction(target, source, states)
        projection = project_onto_fixed_moment_tangent(states, correction, weights)
        fields = (
            score_field(target, source),
            tangent_field_from_projection(target, source, projection),
        )
        for field in fields:
            result = finite_pushforward_kl(
                target, source, field, step_size=0.001, quadrature_order=18
            )
            assert result["positive_jacobian_fraction"] == 1.0
            assert result["kl_change"] < 0


def test_pearson_correction_equals_ratio_weighted_score_correction() -> None:
    target, source = build_toy_regimes()["shape_only"]
    states, _ = source.quadrature(order=10)
    expected = density_ratio(target, source, states)[:, None] * score_correction(
        target, source, states
    )
    torch.testing.assert_close(
        pearson_correction(target, source, states), expected
    )


def test_pearson_field_reduces_pearson_divergence_after_smoothing() -> None:
    target, source = build_toy_regimes()["shape_only"]
    target = target.convolve_isotropic(0.4)
    source = source.convolve_isotropic(0.4)
    before = pearson_divergence(target, source, quadrature_order=18)
    result = finite_pushforward_pearson(
        target,
        source,
        pearson_field(target, source),
        step_size=0.001,
        quadrature_order=18,
    )
    assert result["positive_jacobian_fraction"] == 1.0
    assert result["pearson_before"] == pytest.approx(before)
    assert result["pearson_change"] < 0


def test_pooled_fisher_discrepancy_is_bounded_and_zero_at_equality() -> None:
    target, source = build_toy_regimes()["gaussian_only"]
    assert pooled_fisher_divergence(
        target, target, quadrature_order=20
    ) == pytest.approx(0.0, abs=1e-12)
    value = pooled_fisher_divergence(target, source, quadrature_order=24)
    assert 0.0 < value < 4.0


def test_pooled_fisher_field_has_exact_density_ratio_weight() -> None:
    target, source = build_toy_regimes()["gaussian_only"]
    states, _ = source.quadrature(order=8)
    target_log_probability, _ = target.log_prob_and_score(states)
    source_log_probability, _ = source.log_prob_and_score(states)
    source_probability = torch.sigmoid(
        source_log_probability - target_log_probability
    )
    expected = (
        16.0 * source_probability * (1.0 - source_probability).square()
    )[:, None] * score_correction(target, source, states)
    observed = pooled_fisher_correction(target, source, states)
    torch.testing.assert_close(observed, expected)


def test_static_fd_field_is_zero_for_shape_only_mismatch() -> None:
    target, source = build_toy_regimes()["shape_only"]
    states, _ = source.quadrature(order=12)
    velocity = static_field(target, source)(states, False)
    assert velocity.abs().max() < 1e-10


def test_advfd_functional_field_matches_direct_particle_autograd() -> None:
    target, source = build_toy_regimes()["shape_only"]
    torch.manual_seed(123)
    critic = FeatureCritic(
        CriticConfig(hidden_dim=16, depth=2, feature_dim=4)
    ).to(dtype=torch.float64)
    critic.eval().requires_grad_(False)
    order = 8
    target_points, target_weights = target.quadrature(order)
    source_points, source_weights = source.quadrature(order)

    for objective_mode in (
        "paper_affine",
        "official_regularized",
        "official_mean_only",
        "official_covariance_only",
    ):
        variable_points = source_points.clone().requires_grad_(True)
        target_features = critic(target_points).detach()
        source_features = critic(variable_points)
        target_moments = weighted_moments(target_features, target_weights)
        source_moments = weighted_moments(source_features, source_weights)
        distance, _ = advfd_from_raw_moments(
            target_moments,
            source_moments,
            whitening_epsilon=1e-3,
            objective_mode=objective_mode,
        )
        direct_gradient = torch.autograd.grad(distance, variable_points)[0]
        normalized_weights = source_weights / source_weights.sum()
        direct_field = -direct_gradient / normalized_weights[:, None]

        functional_field, _ = advfd_functional_field(
            critic,
            target,
            source,
            order=order,
            whitening_epsilon=1e-3,
            objective_mode=objective_mode,
        )
        formula_field = functional_field(source_points, False)
        torch.testing.assert_close(
            formula_field,
            direct_field,
            atol=2e-8,
            rtol=2e-8,
        )


def test_official_mean_and_covariance_fields_sum_to_full_field() -> None:
    target, source = build_toy_regimes()["shape_only"]
    torch.manual_seed(321)
    critic = FeatureCritic(
        CriticConfig(hidden_dim=16, depth=2, feature_dim=4)
    ).to(dtype=torch.float64)
    critic.eval().requires_grad_(False)
    fields = {}
    values = {}
    for mode in (
        "official_regularized",
        "official_mean_only",
        "official_covariance_only",
    ):
        fields[mode], values[mode] = advfd_functional_field(
            critic,
            target,
            source,
            order=8,
            whitening_epsilon=1e-3,
            objective_mode=mode,
        )
    states, _ = source.quadrature(order=8)
    full = fields["official_regularized"](states, False)
    decomposed = fields["official_mean_only"](
        states, False
    ) + fields["official_covariance_only"](states, False)
    torch.testing.assert_close(full, decomposed, atol=2e-8, rtol=2e-8)
    assert values["official_regularized"] == pytest.approx(
        values["official_mean_only"] + values["official_covariance_only"],
        abs=1e-10,
    )
