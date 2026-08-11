from __future__ import annotations

import torch

from experiments.run_dual_target_closed_loop_toy import (
    SpiralGaussianMixture,
    analytic_scalar_gate,
    available_cross_gate_conditions,
    endpoint_velocities,
    gate_value,
    scaled_gate_residual,
    velocity_gate_residual,
)


def test_cross_gate_controls_cover_own_branches_and_common_d0_heads() -> None:
    conditions = available_cross_gate_conditions(
        ["D0_xeps", "D1_scaled", "D2_velocity", "D4_safe"]
    )

    assert conditions == [
        "D1_x_own",
        "D1_eps_own",
        "D1_gate_on_D0",
        "D2_x_own",
        "D2_eps_own",
        "D2_gate_on_D0",
        "D4_x_own",
        "D4_eps_own",
        "D4_gate_on_D0",
    ]


def test_perfect_endpoint_predictions_recover_sit_velocity() -> None:
    generator = torch.Generator().manual_seed(3)
    clean = torch.randn(7, 5, generator=generator)
    epsilon = torch.randn(7, 5, generator=generator)
    time_value = torch.linspace(0.1, 0.9, len(clean))
    state = (1.0 - time_value[:, None]) * epsilon + time_value[:, None] * clean

    velocity_x, velocity_epsilon = endpoint_velocities(
        state=state,
        time_value=time_value,
        clean_prediction=clean,
        epsilon_prediction=epsilon,
        denominator_floor=1e-3,
    )

    target = clean - epsilon
    assert torch.allclose(velocity_x, target, atol=1e-6)
    assert torch.allclose(velocity_epsilon, target, atol=1e-6)


def test_scaled_residual_is_time_weighted_velocity_residual() -> None:
    generator = torch.Generator().manual_seed(5)
    clean = torch.randn(11, 4, generator=generator)
    epsilon = torch.randn(11, 4, generator=generator)
    clean_prediction = torch.randn(11, 4, generator=generator)
    epsilon_prediction = torch.randn(11, 4, generator=generator)
    time_value = torch.linspace(0.05, 0.95, len(clean))
    gate = torch.rand(11, 1, generator=generator)

    scaled = scaled_gate_residual(
        gate=gate,
        clean_prediction=clean_prediction,
        epsilon_prediction=epsilon_prediction,
        clean_target=clean,
        epsilon_target=epsilon,
        time_value=time_value,
    )
    velocity = velocity_gate_residual(
        gate=gate,
        clean_prediction=clean_prediction,
        epsilon_prediction=epsilon_prediction,
        clean_target=clean,
        epsilon_target=epsilon,
        time_value=time_value,
        denominator_floor=1e-4,
    )

    weight = (time_value * (1.0 - time_value))[:, None]
    assert torch.allclose(scaled, weight * velocity, atol=1e-6)


def test_asymptotic_safe_zero_correction_equals_one_minus_t() -> None:
    time_value = torch.tensor([0.001, 0.1, 0.5, 0.9, 0.999])
    gate = gate_value(
        torch.zeros(len(time_value), 1),
        time_value,
        asymptotic_safe=True,
        denominator_floor=1e-3,
    )

    assert torch.allclose(gate[:, 0], 1.0 - time_value, atol=2e-6)
    assert torch.allclose(gate[:, 0] / (1.0 - time_value), torch.ones(5), atol=2e-5)
    assert torch.allclose((1.0 - gate[:, 0]) / time_value, torch.ones(5), atol=2e-5)


def test_one_minus_t_gate_cancels_endpoint_denominators() -> None:
    generator = torch.Generator().manual_seed(6)
    state = torch.randn(9, 7, generator=generator)
    clean_prediction = torch.randn(9, 7, generator=generator)
    epsilon_prediction = torch.randn(9, 7, generator=generator)
    time_value = torch.linspace(0.01, 0.99, len(state))
    velocity_x, velocity_epsilon = endpoint_velocities(
        state=state,
        time_value=time_value,
        clean_prediction=clean_prediction,
        epsilon_prediction=epsilon_prediction,
        denominator_floor=1e-4,
    )
    gate = 1.0 - time_value[:, None]
    mixed = gate * velocity_x + (1.0 - gate) * velocity_epsilon

    assert torch.allclose(mixed, clean_prediction - epsilon_prediction, atol=2e-5)


def test_clipped_oracle_gate_never_worse_than_either_branch() -> None:
    generator = torch.Generator().manual_seed(7)
    velocity_x = torch.randn(29, 13, generator=generator)
    velocity_epsilon = torch.randn(29, 13, generator=generator)
    target = torch.randn(29, 13, generator=generator)
    gate = analytic_scalar_gate(
        velocity_x, velocity_epsilon, target, clip=True
    )
    mixed = gate * velocity_x + (1.0 - gate) * velocity_epsilon

    mixed_error = (mixed - target).square().sum(dim=1)
    x_error = (velocity_x - target).square().sum(dim=1)
    epsilon_error = (velocity_epsilon - target).square().sum(dim=1)
    assert torch.all(mixed_error <= x_error + 1e-5)
    assert torch.all(mixed_error <= epsilon_error + 1e-5)
    assert torch.all((gate >= 0.0) & (gate <= 1.0))


def test_bayes_velocity_matches_population_field_at_noise_endpoint() -> None:
    device = torch.device("cpu")
    distribution = SpiralGaussianMixture(
        8,
        components=12,
        component_std=0.04,
        seed=11,
        device=device,
    )
    generator = torch.Generator().manual_seed(13)
    state = torch.randn(17, 8, generator=generator)
    time_value = torch.zeros(len(state))

    clean_mean = distribution.embed(distribution.centers.mean(dim=0, keepdim=True))
    expected = clean_mean - state
    actual = distribution.bayes_velocity(
        state, time_value, denominator_floor=1e-3
    )

    assert torch.allclose(actual, expected, atol=1e-5)


def test_bayes_clean_converges_to_on_distribution_state_near_data_endpoint() -> None:
    distribution = SpiralGaussianMixture(
        16,
        components=16,
        component_std=0.05,
        seed=17,
        device=torch.device("cpu"),
    )
    generator = torch.Generator().manual_seed(19)
    clean, _, _ = distribution.sample(64, generator=generator)
    time_value = torch.full((len(clean),), 1.0)

    posterior_clean = distribution.bayes_clean(clean, time_value)

    assert torch.allclose(posterior_clean, clean, atol=2e-5)
