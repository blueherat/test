from __future__ import annotations

import math

import pytest
import torch

from experiments.internal_guidance_path_extrapolation import (
    PathEndpointPair,
    calibration_split_foresight_velocity,
    decompose_endpoint_posterior_change,
    decompose_material_change,
    decompose_future_weak_drift,
    decompose_euler_foresight_roundtrip,
    extrapolate_path_endpoints,
    finite_lie_bracket_change,
    forecast_weak_reference,
    foresight_weak_guidance,
    match_sample_rms,
    mix_characteristic_velocity,
    mix_material_curvature,
    project_per_sample,
    project_to_forward_ray,
    relax_future_weak_reference,
    richardson_forward_change,
    sample_rms,
    split_internal_guidance,
)


def test_internal_guidance_split_is_exact() -> None:
    strong = torch.randn(2, 4, 3, 3, dtype=torch.float64)
    weak = torch.randn_like(strong)
    gamma = 0.7
    weak_base, calibration = split_internal_guidance(
        strong,
        weak,
        gamma=gamma,
    )
    torch.testing.assert_close(
        weak_base + calibration,
        strong + gamma * (strong - weak),
        rtol=0,
        atol=1e-12,
    )


def test_calibration_split_foresight_has_theory_fixed_strength() -> None:
    strong = torch.randn(2, 4, 3, 3, dtype=torch.float64)
    weak_now = torch.randn_like(strong)
    weak_query = torch.randn_like(strong)
    gamma = 0.6
    observed = calibration_split_foresight_velocity(
        strong,
        weak_now,
        weak_query,
        gamma=gamma,
    )
    guided = strong + gamma * (strong - weak_now)
    expected = guided + (1.0 + gamma) * (weak_now - weak_query)
    torch.testing.assert_close(observed, expected, rtol=0, atol=1e-12)


def test_forward_ray_projection_is_the_constrained_minimizer() -> None:
    value = torch.tensor([[[-2.0, 1.0]], [[3.0, 4.0]]])
    direction = torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]])
    projection = project_to_forward_ray(value, direction)
    torch.testing.assert_close(projection.coefficient, torch.tensor([0.0, 3.0]))
    torch.testing.assert_close(
        projection.parallel,
        torch.tensor([[[0.0, 0.0]], [[3.0, 0.0]]]),
    )
    torch.testing.assert_close(value, projection.parallel + projection.orthogonal)


def test_endpoint_posterior_change_exactly_recovers_velocity_change() -> None:
    generator = torch.Generator().manual_seed(9102)
    state_now = torch.randn(3, 4, 2, 2, generator=generator, dtype=torch.float64)
    state_query = torch.randn_like(state_now)
    velocity_now = torch.randn_like(state_now)
    velocity_query = torch.randn_like(state_now)
    parts = decompose_endpoint_posterior_change(
        state_now,
        0.17,
        velocity_now,
        state_query,
        0.23,
        velocity_query,
    )
    torch.testing.assert_close(
        parts.clean + parts.negative_noise,
        parts.velocity,
        rtol=0,
        atol=1e-12,
    )
    torch.testing.assert_close(
        parts.velocity,
        velocity_now - velocity_query,
        rtol=0,
        atol=0,
    )


def test_endpoint_posterior_contrast_cancels_common_coordinate_shift() -> None:
    velocity_now = torch.tensor([[1.0, -2.0]], dtype=torch.float64)
    velocity_query = torch.tensor([[0.25, 0.5]], dtype=torch.float64)
    state_now = torch.tensor([[3.0, 4.0]], dtype=torch.float64)
    shifts = (torch.tensor([[0.0, 0.0]]), torch.tensor([[100.0, -75.0]]))
    contrasts = []
    for shift in shifts:
        parts = decompose_endpoint_posterior_change(
            state_now,
            0.2,
            velocity_now,
            state_now + shift,
            0.3,
            velocity_query,
        )
        contrasts.append(parts.clean + parts.negative_noise)
    torch.testing.assert_close(contrasts[0], contrasts[1], rtol=0, atol=1e-12)


def test_endpoint_posterior_contrast_has_unique_coordinate_invariant_weight() -> None:
    state_now = torch.tensor([[3.0, 4.0]], dtype=torch.float64)
    velocity_now = torch.tensor([[1.0, -2.0]], dtype=torch.float64)
    velocity_query = torch.tensor([[0.25, 0.5]], dtype=torch.float64)
    shift = torch.tensor([[100.0, -75.0]], dtype=torch.float64)

    def contrast(query_shift: torch.Tensor, weight: float) -> torch.Tensor:
        parts = decompose_endpoint_posterior_change(
            state_now,
            0.2,
            velocity_now,
            state_now + query_shift,
            0.3,
            velocity_query,
        )
        return parts.clean + weight * parts.negative_noise

    for weight in (0.5, 1.0, 1.5):
        observed_change = contrast(shift, weight) - contrast(
            torch.zeros_like(shift), weight
        )
        torch.testing.assert_close(
            observed_change,
            (weight - 1.0) * shift,
            rtol=0,
            atol=1e-12,
        )


def test_characteristic_velocity_has_exact_weak_and_guided_anchors() -> None:
    weak = torch.randn(2, 4, 3, 3)
    guided = torch.randn_like(weak)
    assert mix_characteristic_velocity(weak, guided, rho=0.0) is weak
    assert mix_characteristic_velocity(weak, guided, rho=1.0) is guided
    torch.testing.assert_close(
        mix_characteristic_velocity(weak, guided, rho=0.25),
        0.75 * weak + 0.25 * guided,
    )


def test_characteristic_velocity_contains_strong_ig_field() -> None:
    weak = torch.randn(2, 4, 3, 3, dtype=torch.float64)
    strong = torch.randn_like(weak)
    gamma = 0.7
    guided = strong + gamma * (strong - weak)
    recovered = mix_characteristic_velocity(
        weak,
        guided,
        rho=1.0 / (1.0 + gamma),
    )
    torch.testing.assert_close(recovered, strong, rtol=0, atol=1e-12)


def test_weak_reference_forecast_has_exact_anchors_and_secant_extrapolation() -> None:
    weak_now = torch.randn(2, 4, 3, 3)
    weak_query = torch.randn_like(weak_now)
    assert forecast_weak_reference(weak_now, weak_query, factor=0.0) is weak_now
    assert forecast_weak_reference(weak_now, weak_query, factor=1.0) is weak_query
    torch.testing.assert_close(
        forecast_weak_reference(weak_now, weak_query, factor=2.0),
        2.0 * weak_query - weak_now,
    )
    with pytest.raises(ValueError, match="finite and non-negative"):
        forecast_weak_reference(weak_now, weak_query, factor=-1.0)


def test_material_change_has_exact_time_state_split() -> None:
    weak_now = torch.randn(3, 4, 2, 2, dtype=torch.float64)
    weak_future_same_state = torch.randn_like(weak_now)
    weak_future_along_path = torch.randn_like(weak_now)
    parts = decompose_material_change(
        weak_now,
        weak_future_same_state,
        weak_future_along_path,
    )
    torch.testing.assert_close(
        parts.temporal + parts.advective,
        parts.combined,
        rtol=0,
        atol=1e-12,
    )
    torch.testing.assert_close(
        parts.combined,
        weak_now - weak_future_along_path,
        rtol=0,
        atol=0,
    )


def test_finite_lie_bracket_vanishes_for_identical_field_queries() -> None:
    now = torch.randn(2, 4, 3, 3)
    future = torch.randn_like(now)
    change = finite_lie_bracket_change(now, future, now, future)
    assert torch.equal(change, torch.zeros_like(change))


def test_finite_lie_bracket_matches_noncommuting_linear_fields() -> None:
    # G(z)=A z, W(z)=B z. The forward difference returns
    # h*(A B-B A)z = -h*[G,W] under [G,W]=B A-A B.
    state = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
    guided_matrix = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.float64)
    weak_matrix = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    horizon = 0.1
    guided = guided_matrix @ state
    weak = weak_matrix @ state
    weak_along_guided = weak_matrix @ (state + horizon * guided)
    guided_along_weak = guided_matrix @ (state + horizon * weak)
    actual = finite_lie_bracket_change(
        weak,
        weak_along_guided,
        guided,
        guided_along_weak,
    )
    expected = horizon * (
        guided_matrix @ weak_matrix - weak_matrix @ guided_matrix
    ) @ state
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)


def test_future_weak_drift_has_exact_gap_and_strong_split() -> None:
    generator = torch.Generator().manual_seed(17)
    fields = [
        torch.randn(3, 4, 2, 2, generator=generator, dtype=torch.float64)
        for _ in range(4)
    ]
    decomposition = decompose_future_weak_drift(*fields)

    torch.testing.assert_close(
        decomposition.weak_drift_correction,
        decomposition.gap_change + decomposition.strong_curvature_correction,
        rtol=0,
        atol=1e-12,
    )
    torch.testing.assert_close(
        decomposition.gap_now,
        fields[0] - fields[1],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        decomposition.gap_future,
        fields[2] - fields[3],
        rtol=0,
        atol=0,
    )


def test_foresight_weak_guidance_has_exact_ig_anchors() -> None:
    strong = torch.randn(2, 4, 3, 3)
    weak = torch.randn_like(strong)
    weak_future = torch.randn_like(strong)
    gamma = 0.7

    ordinary = foresight_weak_guidance(
        strong, weak, weak_future, gamma=gamma, alpha=0.0
    )
    future = foresight_weak_guidance(
        strong, weak, weak_future, gamma=gamma, alpha=1.0
    )
    torch.testing.assert_close(ordinary, strong + gamma * (strong - weak))
    torch.testing.assert_close(future, strong + gamma * (strong - weak_future))

    alpha = 0.35
    mixed = foresight_weak_guidance(
        strong, weak, weak_future, gamma=gamma, alpha=alpha
    )
    reference = strong + gamma * (
        strong - ((1.0 - alpha) * weak + alpha * weak_future)
    )
    torch.testing.assert_close(mixed, reference)


def test_relaxed_future_reference_matches_picard_velocity_update() -> None:
    generator = torch.Generator().manual_seed(31)
    strong = torch.randn(3, 4, 2, 2, generator=generator, dtype=torch.float64)
    weak = torch.randn_like(strong)
    weak_future = torch.randn_like(strong)
    gamma = 0.7
    eta = 0.6

    velocity = strong + gamma * (strong - weak)
    candidate = strong + gamma * (strong - weak_future)
    rho = eta / gamma
    expected = (1.0 - rho) * velocity + rho * candidate
    updated_reference = relax_future_weak_reference(
        weak,
        weak_future,
        gamma=gamma,
        eta=eta,
    )
    actual = strong + gamma * (strong - updated_reference)

    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
    torch.testing.assert_close(
        velocity + eta * (weak - weak_future),
        expected,
        rtol=0,
        atol=1e-12,
    )


def test_richardson_forward_change_cancels_quadratic_term() -> None:
    horizon = 0.2
    constant = torch.tensor([1.25], dtype=torch.float64)
    linear = torch.tensor([-0.7], dtype=torch.float64)
    quadratic = torch.tensor([2.3], dtype=torch.float64)

    def field(offset: float) -> torch.Tensor:
        return constant + linear * offset + quadratic * offset**2

    actual = richardson_forward_change(
        field(0.0), field(0.5 * horizon), field(horizon)
    )
    torch.testing.assert_close(actual, -horizon * linear, rtol=0, atol=1e-14)


def test_material_curvature_mix_has_exact_anchors() -> None:
    first = torch.tensor([[1.0, 2.0]])
    finite = torch.tensor([[3.0, 6.0]])
    assert mix_material_curvature(
        first, finite, curvature_weight=0.0
    ) is first
    assert mix_material_curvature(
        first, finite, curvature_weight=1.0
    ) is finite
    mixed = mix_material_curvature(first, finite, curvature_weight=0.25)
    assert torch.equal(mixed, torch.tensor([[1.5, 3.0]]))


def test_euler_foresight_decomposition_is_exact() -> None:
    state = torch.randn(3, 4, 2, 2)
    guided = torch.randn_like(state)
    weak_now = torch.randn_like(state)
    weak_future = torch.randn_like(state)
    horizon = 0.125
    future = state + horizon * guided
    roundtrip = future - horizon * weak_future

    parts = decompose_euler_foresight_roundtrip(
        guided,
        weak_now,
        weak_future,
        horizon=horizon,
    )

    torch.testing.assert_close(parts.roundtrip_displacement, roundtrip - state)
    torch.testing.assert_close(
        parts.local_displacement + parts.future_displacement,
        parts.roundtrip_displacement,
    )


def test_path_endpoint_controls_are_bitwise_exact() -> None:
    strong = torch.randn(3, 4, 2, 2)
    guided = torch.randn_like(strong)
    pair = PathEndpointPair(strong=strong, guided=guided)

    assert extrapolate_path_endpoints(pair, rho=0.0) is strong
    assert extrapolate_path_endpoints(pair, rho=1.0) is guided
    torch.testing.assert_close(
        extrapolate_path_endpoints(pair, rho=2.0),
        2.0 * guided - strong,
        rtol=1e-6,
        atol=1e-6,
    )


def test_path_endpoint_validation_rejects_invalid_inputs() -> None:
    strong = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="identical shapes"):
        extrapolate_path_endpoints(
            PathEndpointPair(strong=strong, guided=torch.zeros(2, 4)), rho=1.0
        )
    for rho in (-0.1, math.inf, math.nan):
        with pytest.raises(ValueError, match="finite and non-negative"):
            extrapolate_path_endpoints(
                PathEndpointPair(strong=strong, guided=torch.ones_like(strong)),
                rho=rho,
            )


def test_path_extrapolation_is_not_local_scale_for_nonlinear_flow_map() -> None:
    """Linear state dynamics already separate path and field extrapolation.

    For dz/dt=a*z and dz/dt=b*z, endpoint extrapolation is an affine secant of
    exp(a*h) and exp(b*h).  Scaling the instantaneous field instead exponentiates
    the affine coefficient, so the two agree at rho=0/1 but not in general.
    """

    state = torch.tensor([[1.0]])
    a, b, horizon, rho = -0.4, 0.7, 0.8, 1.6
    strong = state * math.exp(a * horizon)
    guided = state * math.exp(b * horizon)
    path = extrapolate_path_endpoints(
        PathEndpointPair(strong=strong, guided=guided), rho=rho
    )
    local_scaled = state * math.exp((a + rho * (b - a)) * horizon)

    assert not torch.allclose(path, local_scaled, rtol=1e-4, atol=1e-4)
    assert torch.equal(
        extrapolate_path_endpoints(
            PathEndpointPair(strong=strong, guided=guided), rho=1.0
        ),
        guided,
    )


def test_sample_rms() -> None:
    value = torch.tensor([[[3.0, 4.0]], [[0.0, 2.0]]])
    expected = torch.tensor([math.sqrt(12.5), math.sqrt(2.0)])
    torch.testing.assert_close(sample_rms(value), expected)


def test_projection_is_per_sample_and_reconstructs() -> None:
    reference = torch.tensor(
        [[[[1.0, 0.0]]], [[[0.0, 2.0]]], [[[0.0, 0.0]]]]
    )
    value = torch.tensor(
        [[[[3.0, 4.0]]], [[[5.0, 6.0]]], [[[7.0, 8.0]]]]
    )
    projection = project_per_sample(value, reference)
    torch.testing.assert_close(
        projection.coefficient, torch.tensor([3.0, 3.0, 0.0])
    )
    torch.testing.assert_close(projection.parallel + projection.orthogonal, value)
    dot = (
        projection.orthogonal.flatten(1) * reference.flatten(1)
    ).sum(dim=1)
    torch.testing.assert_close(dot, torch.zeros_like(dot))
    assert torch.equal(projection.orthogonal[-1], value[-1])


def test_match_sample_rms_preserves_direction_and_matches_reference() -> None:
    value = torch.tensor([[[[3.0, 4.0]]], [[[0.0, 0.0]]]])
    reference = torch.tensor([[[[2.0, 2.0]]], [[[5.0, 5.0]]]])
    matched = match_sample_rms(value, reference)
    torch.testing.assert_close(sample_rms(matched), torch.tensor([2.0, 0.0]))
    torch.testing.assert_close(
        matched[0, 0, 0, 0] / matched[0, 0, 0, 1], torch.tensor(0.75)
    )
