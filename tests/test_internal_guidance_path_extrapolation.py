from __future__ import annotations

import math

import pytest
import torch

from experiments.internal_guidance_path_extrapolation import (
    PathEndpointPair,
    affine_counterfactual_ratio_velocity,
    align_linear_path_state_to_endpoint_coordinate,
    calibration_split_foresight_velocity,
    counterfactual_telescoping_velocity,
    decompose_cross_time_velocity_change,
    decompose_endpoint_posterior_change,
    decompose_material_change,
    decompose_future_weak_drift,
    decompose_euler_foresight_roundtrip,
    extrapolate_path_endpoints,
    endpoint_score_to_linear_velocity,
    factorized_scale_space_guidance_velocity,
    finite_lie_bracket_change,
    forecast_weak_reference,
    foresight_weak_guidance,
    match_sample_rms,
    marginal_score_to_linear_velocity,
    mix_characteristic_velocity,
    mix_material_curvature,
    project_per_sample,
    project_to_forward_ray,
    relax_future_weak_reference,
    richardson_forward_change,
    sample_rms,
    split_internal_guidance,
    telescoping_scale_space_guidance_velocity,
    transported_internal_gap_velocity,
    linear_velocity_to_endpoint_score,
    linear_velocity_to_marginal_score,
)


def test_counterfactual_telescoping_single_transition_is_pfr() -> None:
    generator = torch.Generator().manual_seed(1901)
    weak = torch.randn(2, 4, 2, 2, generator=generator, dtype=torch.float64)
    strong = torch.randn_like(weak)
    query = torch.randn_like(weak)
    observed = counterfactual_telescoping_velocity(
        (weak, strong), (query,), gamma=0.7
    )
    expected = calibration_split_foresight_velocity(
        strong, weak, query, gamma=0.7
    )
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)


def test_counterfactual_telescoping_collapsed_query_is_ordinary_ig() -> None:
    generator = torch.Generator().manual_seed(1902)
    shallow = torch.randn(2, 4, 2, 2, generator=generator, dtype=torch.float64)
    middle = torch.randn_like(shallow)
    strong = torch.randn_like(shallow)
    gamma = 0.6
    observed = counterfactual_telescoping_velocity(
        (shallow, middle, strong),
        (shallow, middle),
        gamma=gamma,
    )
    expected = strong + gamma * (strong - shallow)
    torch.testing.assert_close(observed, expected, rtol=0, atol=2e-15)


def test_counterfactual_telescoping_has_affine_score_closure() -> None:
    generator = torch.Generator().manual_seed(1903)
    state = torch.randn(2, 4, 2, 2, generator=generator, dtype=torch.float64)
    shallow = torch.randn_like(state)
    middle = torch.randn_like(state)
    strong = torch.randn_like(state)
    shallow_query = torch.randn_like(state)
    middle_query = torch.randn_like(state)
    gamma = 0.7
    beta = 1.0 + gamma
    time = 0.35
    observed = counterfactual_telescoping_velocity(
        (shallow, middle, strong),
        (shallow_query, middle_query),
        gamma=gamma,
    )
    observed_score = linear_velocity_to_marginal_score(observed, state, time)
    score = lambda value: linear_velocity_to_marginal_score(value, state, time)
    expected_score = score(shallow) + beta * (
        score(middle) - score(shallow_query) + score(strong) - score(middle_query)
    )
    torch.testing.assert_close(observed_score, expected_score, rtol=0, atol=5e-14)


def test_affine_counterfactual_single_reference_is_historical_pfr() -> None:
    generator = torch.Generator().manual_seed(1801)
    strong = torch.randn(3, 4, 2, 2, generator=generator, dtype=torch.float64)
    weak = torch.randn_like(strong)
    reference = torch.randn_like(strong)
    observed = affine_counterfactual_ratio_velocity(
        strong, weak, (reference,), (1.0,), gamma=0.7
    )
    expected = calibration_split_foresight_velocity(
        strong, weak, reference, gamma=0.7
    )
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)


def test_affine_counterfactual_geomean_has_exact_score_closure() -> None:
    generator = torch.Generator().manual_seed(1802)
    state = torch.randn(3, 4, 2, 2, generator=generator, dtype=torch.float64)
    strong = torch.randn_like(state)
    weak = torch.randn_like(state)
    first = torch.randn_like(state)
    second = torch.randn_like(state)
    gamma = 0.6
    time = 0.3
    observed = affine_counterfactual_ratio_velocity(
        strong, weak, (first, second), (0.5, 0.5), gamma=gamma
    )
    observed_score = linear_velocity_to_marginal_score(observed, state, time)
    beta = 1.0 + gamma
    expected_score = (
        linear_velocity_to_marginal_score(weak, state, time)
        + beta
        * (
            linear_velocity_to_marginal_score(strong, state, time)
            - 0.5 * linear_velocity_to_marginal_score(first, state, time)
            - 0.5 * linear_velocity_to_marginal_score(second, state, time)
        )
    )
    torch.testing.assert_close(observed_score, expected_score, rtol=0, atol=3e-14)


def test_affine_counterfactual_rejects_non_barycentric_references() -> None:
    value = torch.randn(2, 3)
    with pytest.raises(ValueError, match="sum to one"):
        affine_counterfactual_ratio_velocity(
            value, value, (value, value), (0.6, 0.6), gamma=0.7
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


def test_endpoint_score_velocity_conversion_is_exact() -> None:
    generator = torch.Generator().manual_seed(1701)
    state = torch.randn(3, 4, 2, 2, generator=generator, dtype=torch.float64)
    velocity = torch.randn_like(state)
    time = torch.tensor([0.2, 0.4, 0.7], dtype=torch.float64)
    score = linear_velocity_to_endpoint_score(velocity, state, time)
    recovered = endpoint_score_to_linear_velocity(score, state, time)
    torch.testing.assert_close(recovered, velocity, rtol=0, atol=2e-14)


def test_endpoint_coordinate_alignment_holds_state_over_time_fixed() -> None:
    state = torch.tensor(
        [[[2.0, -1.0]], [[0.5, 3.0]]], dtype=torch.float64
    )
    time = torch.tensor([0.4, 0.5], dtype=torch.float64)
    reference_time = torch.tensor([0.2, 0.25], dtype=torch.float64)
    aligned = align_linear_path_state_to_endpoint_coordinate(
        state, time, reference_time
    )
    current_y = state / time.reshape(2, 1, 1)
    reference_y = aligned / reference_time.reshape(2, 1, 1)
    torch.testing.assert_close(reference_y, current_y, rtol=0, atol=1e-12)


def test_telescoping_score_guidance_recovers_internal_guidance_at_same_time() -> None:
    generator = torch.Generator().manual_seed(1702)
    state = torch.randn(2, 4, 3, 3, generator=generator, dtype=torch.float64)
    strong = torch.randn_like(state)
    weak = torch.randn_like(state)
    time = torch.tensor([0.3, 0.6], dtype=torch.float64)
    gamma = 0.7
    observed = telescoping_scale_space_guidance_velocity(
        strong,
        weak,
        state,
        state,
        time,
        time,
        gamma=gamma,
    )
    expected = strong + gamma * (strong - weak)
    torch.testing.assert_close(observed, expected, rtol=0, atol=2e-14)


def test_telescoping_score_guidance_has_exact_zero_gamma_anchor() -> None:
    state = torch.randn(2, 4, 3, 3)
    strong = torch.randn_like(state)
    weak = torch.randn_like(state)
    observed = telescoping_scale_space_guidance_velocity(
        strong,
        weak,
        state,
        state * 0.5,
        0.4,
        0.2,
        gamma=0.0,
    )
    assert observed is strong


def test_endpoint_score_conversion_rejects_singular_times() -> None:
    state = torch.randn(2, 3)
    with pytest.raises(ValueError, match="time in \\(0, 1\\)"):
        linear_velocity_to_endpoint_score(state, state, 0.0)
    with pytest.raises(ValueError, match="time in \\(0, 1\\)"):
        endpoint_score_to_linear_velocity(state, state, 1.0)


def test_marginal_score_velocity_conversion_is_exact() -> None:
    generator = torch.Generator().manual_seed(1703)
    state = torch.randn(3, 4, 2, 2, generator=generator, dtype=torch.float64)
    velocity = torch.randn_like(state)
    time = torch.tensor([0.1, 0.4, 0.8], dtype=torch.float64)
    score = linear_velocity_to_marginal_score(velocity, state, time)
    recovered = marginal_score_to_linear_velocity(score, state, time)
    torch.testing.assert_close(recovered, velocity, rtol=0, atol=3e-14)


def test_factorized_scale_space_zero_temporal_weight_is_ordinary_ig() -> None:
    generator = torch.Generator().manual_seed(1704)
    state = torch.randn(2, 4, 2, 2, generator=generator, dtype=torch.float64)
    strong = torch.randn_like(state)
    weak = torch.randn_like(state)
    reference = torch.randn_like(state)
    observed = factorized_scale_space_guidance_velocity(
        strong,
        weak,
        weak,
        reference,
        state,
        state,
        0.3,
        0.2,
        gamma=0.7,
        temporal_weight=0.0,
    )
    expected = strong + 0.7 * (strong - weak)
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)


def test_factorized_weak_ratio_telescopes_at_unit_weight() -> None:
    generator = torch.Generator().manual_seed(1705)
    state = torch.randn(2, 4, 2, 2, generator=generator, dtype=torch.float64)
    strong = torch.randn_like(state)
    weak = torch.randn_like(state)
    weak_reference = torch.randn_like(state)
    gamma = 0.6
    time = 0.4
    reference_time = 0.2
    observed = factorized_scale_space_guidance_velocity(
        strong,
        weak,
        weak,
        weak_reference,
        state,
        state,
        time,
        reference_time,
        gamma=gamma,
        temporal_weight=1.0,
    )
    strong_score = linear_velocity_to_marginal_score(strong, state, time)
    reference_score = linear_velocity_to_marginal_score(
        weak_reference, state, reference_time
    )
    expected_score = strong_score + gamma * (strong_score - reference_score)
    expected = marginal_score_to_linear_velocity(expected_score, state, time)
    torch.testing.assert_close(observed, expected, rtol=0, atol=3e-14)


def test_factorized_scale_space_accepts_gaussian_prior_reference_at_time_zero() -> None:
    state = torch.randn(2, 4, 2, 2, dtype=torch.float64)
    velocity = torch.randn_like(state)
    prior_score = linear_velocity_to_marginal_score(
        velocity, state, 0.0
    )
    torch.testing.assert_close(prior_score, -state, rtol=0, atol=0)


def test_cross_time_velocity_decomposition_is_exact() -> None:
    generator = torch.Generator().manual_seed(1706)
    state = torch.randn(3, 4, 2, 2, generator=generator, dtype=torch.float64)
    velocity_now = torch.randn_like(state)
    velocity_reference = torch.randn_like(state)
    parts = decompose_cross_time_velocity_change(
        velocity_now,
        velocity_reference,
        state,
        torch.tensor([0.1, 0.2, 0.4], dtype=torch.float64),
        torch.tensor([0.15, 0.25, 0.45], dtype=torch.float64),
    )
    torch.testing.assert_close(
        parts.total,
        velocity_now - velocity_reference,
        rtol=0,
        atol=5e-14,
    )


def test_parameterization_transport_holds_marginal_score_fixed() -> None:
    state = torch.randn(2, 4, 2, 2, dtype=torch.float64)
    velocity_now = torch.randn_like(state)
    velocity_reference = torch.randn_like(state)
    time_now = 0.2
    time_reference = 0.3
    parts = decompose_cross_time_velocity_change(
        velocity_now,
        velocity_reference,
        state,
        time_now,
        time_reference,
    )
    transported = velocity_now - parts.parameterization_transport
    score_now = linear_velocity_to_marginal_score(
        velocity_now, state, time_now
    )
    score_transported = linear_velocity_to_marginal_score(
        transported, state, time_reference
    )
    torch.testing.assert_close(score_transported, score_now, rtol=0, atol=2e-14)


def test_recomposed_cross_time_change_matches_time_only_reference_revision() -> None:
    generator = torch.Generator().manual_seed(2109)
    state = torch.randn(3, 4, 2, 2, generator=generator, dtype=torch.float64)
    strong = torch.randn_like(state)
    weak_now = torch.randn_like(state)
    weak_reference = torch.randn_like(state)
    gamma = 0.7
    parts = decompose_cross_time_velocity_change(
        weak_now,
        weak_reference,
        state,
        0.2,
        0.23125,
    )
    recomposed = (
        strong
        + gamma * (strong - weak_now)
        + (1.0 + gamma) * parts.total
    )
    expected = calibration_split_foresight_velocity(
        strong,
        weak_now,
        weak_reference,
        gamma=gamma,
    )
    torch.testing.assert_close(recomposed, expected, rtol=0, atol=2e-14)


def test_transported_gap_is_aligned_query_refinement() -> None:
    generator = torch.Generator().manual_seed(1407)
    strong_now = torch.randn(2, 4, 3, 3, generator=generator, dtype=torch.float64)
    weak_now = torch.randn_like(strong_now)
    strong_query = torch.randn_like(strong_now)
    weak_query = torch.randn_like(strong_now)
    gamma = 0.7
    observed = transported_internal_gap_velocity(
        strong_now,
        weak_now,
        strong_query,
        weak_query,
        gamma=gamma,
    )
    expected = weak_now + (1.0 + gamma) * (strong_query - weak_query)
    torch.testing.assert_close(observed, expected, rtol=0, atol=1e-12)


def test_transported_gap_recovers_ordinary_ig_at_identical_query() -> None:
    strong = torch.randn(2, 4, 3, 3, dtype=torch.float64)
    weak = torch.randn_like(strong)
    gamma = 0.6
    observed = transported_internal_gap_velocity(
        strong,
        weak,
        strong,
        weak,
        gamma=gamma,
    )
    expected = strong + gamma * (strong - weak)
    torch.testing.assert_close(observed, expected, rtol=0, atol=1e-12)


def test_transported_gap_has_diagonal_consistency() -> None:
    field_now = torch.randn(2, 4, 3, 3, dtype=torch.float64)
    field_query = torch.randn_like(field_now)
    observed = transported_internal_gap_velocity(
        field_now,
        field_now,
        field_query,
        field_query,
        gamma=0.7,
    )
    torch.testing.assert_close(observed, field_now, rtol=0, atol=0)


def test_transported_gap_extra_correction_is_common_mode_invariant() -> None:
    generator = torch.Generator().manual_seed(1408)
    strong_now = torch.randn(2, 4, 3, 3, generator=generator, dtype=torch.float64)
    weak_now = torch.randn_like(strong_now)
    strong_query = torch.randn_like(strong_now)
    weak_query = torch.randn_like(strong_now)
    common_now = torch.randn_like(strong_now)
    common_query = torch.randn_like(strong_now)
    gamma = 0.6

    ordinary = strong_now + gamma * (strong_now - weak_now)
    transported = transported_internal_gap_velocity(
        strong_now,
        weak_now,
        strong_query,
        weak_query,
        gamma=gamma,
    )
    shifted_ordinary = (strong_now + common_now) + gamma * (
        strong_now + common_now - weak_now - common_now
    )
    shifted_transported = transported_internal_gap_velocity(
        strong_now + common_now,
        weak_now + common_now,
        strong_query + common_query,
        weak_query + common_query,
        gamma=gamma,
    )
    torch.testing.assert_close(
        transported - ordinary,
        shifted_transported - shifted_ordinary,
        rtol=0,
        atol=1e-12,
    )


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
