from __future__ import annotations

import pytest
import torch

from experiments.imagenet100_sit_static_pair import (
    DUAL_OUTPUT_PROTOCOL,
    LEGACY_PROTOCOL,
    SINGLE_TARGET_PROTOCOL,
    common_unique_guided_velocity,
    common_unique_orthogonal_directions,
    controlled_pair_velocity,
    decompose_relative_to_anchor,
    output_to_field_velocity,
    post_floor_window,
    project_onto_direction,
    resolve_field_semantics,
    static_pair_velocity,
    with_inference_denominator_floor,
    x_floor_coefficient,
)
from experiments.sample_imagenet100_sit_static_pair_fid import (
    conditional_common_unique_velocity,
    conditional_static_pair_velocity,
    validate_pair_compatibility,
)
from experiments.run_imagenet100_sit_static_pair_fid5k import format_scale


def test_static_pair_has_exact_boundary_paths() -> None:
    anchor = torch.randn(3, 4, 2, 2)
    other = torch.randn_like(anchor)

    assert torch.equal(static_pair_velocity(anchor, other, scale=0.0), anchor)
    assert torch.equal(static_pair_velocity(anchor, other, scale=1.0), other)


def test_static_pair_interpolates_and_extrapolates_in_one_orientation() -> None:
    anchor = torch.zeros(1, 1, 1, 1)
    other = torch.full_like(anchor, 2.0)

    assert torch.equal(
        static_pair_velocity(anchor, other, scale=0.25),
        torch.full_like(anchor, 0.5),
    )
    assert torch.equal(
        static_pair_velocity(anchor, other, scale=-0.5),
        torch.full_like(anchor, -1.0),
    )
    assert torch.equal(
        static_pair_velocity(anchor, other, scale=1.5),
        torch.full_like(anchor, 3.0),
    )


def test_jit_x_floor_coefficient_matches_analytic_endpoint_attenuation() -> None:
    times = torch.tensor([0.0, 0.9, 0.95, 0.975, 1.0])
    actual = x_floor_coefficient(times, denominator_floor=0.05)
    torch.testing.assert_close(
        actual,
        torch.tensor([1.0, 1.0, 1.0, 0.5, 0.0]),
        rtol=1e-5,
        atol=1e-6,
    )


def test_floor_and_residual_controls_exactly_decompose_full_pair() -> None:
    anchor = torch.randn(3, 4, 2, 2)
    other = torch.randn_like(anchor)
    times = torch.tensor([0.3, 0.96, 0.99])
    kwargs = {
        "time_value": times,
        "scale": -0.75,
        "other_prediction_target": "x",
        "other_denominator_floor": 0.05,
        "window_transition_width": 0.01,
    }
    full = controlled_pair_velocity(anchor, other, mode="full_pair", **kwargs)
    floor = controlled_pair_velocity(anchor, None, mode="floor_only", **kwargs)
    residual = controlled_pair_velocity(anchor, other, mode="floor_residual", **kwargs)

    torch.testing.assert_close(
        full - anchor,
        (floor - anchor) + (residual - anchor),
        rtol=2e-6,
        atol=2e-6,
    )


def test_floor_only_negative_one_is_late_velocity_boost() -> None:
    anchor = torch.ones(1, 1, 1, 1)
    actual = controlled_pair_velocity(
        anchor,
        None,
        time_value=torch.tensor([0.975]),
        scale=-1.0,
        mode="floor_only",
        other_prediction_target="x",
        other_denominator_floor=0.05,
        window_transition_width=0.01,
    )
    torch.testing.assert_close(actual, torch.full_like(anchor, 1.5))


def test_pre_and_post_floor_windows_partition_full_perturbation() -> None:
    times = torch.tensor([0.8, 0.945, 0.95, 0.955, 0.99])
    post = post_floor_window(
        times,
        denominator_floor=0.05,
        transition_width=0.01,
    )
    torch.testing.assert_close(post + (1.0 - post), torch.ones_like(post))

    anchor = torch.randn(5, 2)
    other = torch.randn_like(anchor)
    kwargs = {
        "time_value": times,
        "scale": -1.0,
        "other_prediction_target": "x",
        "other_denominator_floor": 0.05,
        "window_transition_width": 0.01,
    }
    full = controlled_pair_velocity(anchor, other, mode="full_pair", **kwargs)
    pre = controlled_pair_velocity(anchor, other, mode="pre_floor_pair", **kwargs)
    post_field = controlled_pair_velocity(
        anchor, other, mode="post_floor_pair", **kwargs
    )
    torch.testing.assert_close(
        full - anchor,
        (pre - anchor) + (post_field - anchor),
        rtol=2e-6,
        atol=2e-6,
    )


def test_parallel_and_orthogonal_controls_decompose_the_full_pair() -> None:
    anchor = torch.randn(4, 3, 2, 2)
    other = torch.randn_like(anchor)
    kwargs = {
        "time_value": torch.full((4,), 0.5),
        "scale": -0.7,
        "other_prediction_target": "x",
        "other_denominator_floor": 0.05,
        "window_transition_width": 0.01,
    }
    full = controlled_pair_velocity(anchor, other, mode="full_pair", **kwargs)
    parallel = controlled_pair_velocity(
        anchor, other, mode="parallel_pair", **kwargs
    )
    orthogonal = controlled_pair_velocity(
        anchor, other, mode="orthogonal_pair", **kwargs
    )

    torch.testing.assert_close(
        full - anchor,
        (parallel - anchor) + (orthogonal - anchor),
        rtol=2e-6,
        atol=2e-6,
    )
    parallel_delta = parallel - anchor
    orthogonal_delta = orthogonal - anchor
    dot = (parallel_delta * orthogonal_delta).flatten(1).sum(1)
    torch.testing.assert_close(dot, torch.zeros_like(dot), atol=2e-5, rtol=0)


def test_relative_decomposition_reconstructs_direction_per_sample() -> None:
    anchor = torch.randn(5, 4, 3, 2, dtype=torch.float64)
    direction = torch.randn_like(anchor)

    parallel, orthogonal = decompose_relative_to_anchor(anchor, direction)

    torch.testing.assert_close(parallel + orthogonal, direction)
    dot = (anchor * orthogonal).flatten(1).sum(1)
    torch.testing.assert_close(dot, torch.zeros_like(dot), atol=1e-12, rtol=0)


def test_projection_onto_direction_is_paired_per_sample() -> None:
    value = torch.tensor([[[[2.0, 3.0]]], [[[5.0, 7.0]]]])
    direction = torch.tensor([[[[1.0, 0.0]]], [[[0.0, 2.0]]]])

    projected = project_onto_direction(value, direction)

    torch.testing.assert_close(
        projected,
        torch.tensor([[[[2.0, 0.0]]], [[[0.0, 7.0]]]]),
    )


def test_reciprocal_common_unique_components_reconstruct_both_directions() -> None:
    anchor = torch.randn(5, 4, 3, 2, dtype=torch.float64)
    x_velocity = torch.randn_like(anchor)
    v_velocity = torch.randn_like(anchor)

    parts = common_unique_orthogonal_directions(anchor, x_velocity, v_velocity)

    torch.testing.assert_close(
        parts["x_common_on_v"] + parts["x_unique_to_v"],
        parts["x_orthogonal"],
    )
    torch.testing.assert_close(
        parts["v_common_on_x"] + parts["v_unique_to_x"],
        parts["v_orthogonal"],
    )
    x_unique_dot_v = (
        parts["x_unique_to_v"] * parts["v_orthogonal"]
    ).flatten(1).sum(1)
    v_unique_dot_x = (
        parts["v_unique_to_x"] * parts["x_orthogonal"]
    ).flatten(1).sum(1)
    torch.testing.assert_close(
        x_unique_dot_v,
        torch.zeros_like(x_unique_dot_v),
        atol=1e-12,
        rtol=0,
    )
    torch.testing.assert_close(
        v_unique_dot_x,
        torch.zeros_like(v_unique_dot_x),
        atol=1e-12,
        rtol=0,
    )


def test_common_unique_guidance_uses_positive_empirical_orientation() -> None:
    anchor = torch.tensor([[[[1.0, 0.0, 0.0]]]])
    x_velocity = torch.tensor([[[[1.0, -2.0, -3.0]]]])
    v_velocity = torch.tensor([[[[1.0, -4.0, 0.0]]]])

    actual = common_unique_guided_velocity(
        anchor,
        x_velocity,
        v_velocity,
        scale=1.0,
        component="x_common_on_v",
    )

    torch.testing.assert_close(actual, torch.tensor([[[[1.0, 2.0, 0.0]]]]))


def test_single_output_semantics_use_checkpoint_prediction_target() -> None:
    legacy = resolve_field_semantics(
        protocol=LEGACY_PROTOCOL,
        config={},
        requested_path="auto",
    )
    jit_x = resolve_field_semantics(
        protocol=SINGLE_TARGET_PROTOCOL,
        config={"prediction_target": "x", "denominator_floor": 0.05},
        requested_path="auto",
    )

    assert legacy.prediction_target == "velocity"
    assert legacy.denominator_floor == 1e-3
    assert jit_x.prediction_target == "x"
    assert jit_x.denominator_floor == 0.05


def test_inference_floor_override_is_explicit_and_x_only() -> None:
    x_semantics = resolve_field_semantics(
        protocol=SINGLE_TARGET_PROTOCOL,
        config={"prediction_target": "x", "denominator_floor": 0.05},
        requested_path="auto",
    )
    overridden = with_inference_denominator_floor(x_semantics, 1e-3)

    assert x_semantics.denominator_floor == 0.05
    assert overridden.denominator_floor == 1e-3
    assert with_inference_denominator_floor(x_semantics, None) is x_semantics

    velocity_semantics = resolve_field_semantics(
        protocol=SINGLE_TARGET_PROTOCOL,
        config={"prediction_target": "velocity", "denominator_floor": 1e-3},
        requested_path="auto",
    )
    with pytest.raises(ValueError, match="requires an x field"):
        with_inference_denominator_floor(velocity_semantics, 1e-3)
    with pytest.raises(ValueError, match="must be in"):
        with_inference_denominator_floor(x_semantics, 0.0)


def test_dual_output_semantics_require_an_explicit_path() -> None:
    config = {"denominator_floor": 1e-3, "gate_activation": "sigmoid"}
    semantics = resolve_field_semantics(
        protocol=DUAL_OUTPUT_PROTOCOL,
        config=config,
        requested_path="epsilon",
    )
    assert semantics.field_path == "epsilon"
    assert semantics.gate_activation == "sigmoid"

    with pytest.raises(ValueError, match="require x, epsilon, or dynamic"):
        resolve_field_semantics(
            protocol=DUAL_OUTPUT_PROTOCOL,
            config=config,
            requested_path="auto",
        )


def test_output_conversion_matches_exact_linear_flow_for_single_targets() -> None:
    clean = torch.randn(2, 4, 3, 3)
    noise = torch.randn_like(clean)
    times = torch.tensor([0.2, 0.7])
    time_image = times[:, None, None, None]
    state = (1.0 - time_image) * noise + time_image * clean
    target = clean - noise

    for prediction_target, output in (
        ("velocity", target),
        ("x", clean),
        ("epsilon", noise),
    ):
        semantics = resolve_field_semantics(
            protocol=SINGLE_TARGET_PROTOCOL,
            config={
                "prediction_target": prediction_target,
                "denominator_floor": 1e-3,
            },
            requested_path="auto",
        )
        actual = output_to_field_velocity(
            output,
            state=state,
            time_value=times,
            semantics=semantics,
        )
        torch.testing.assert_close(actual, target, rtol=2e-6, atol=2e-6)


class ConstantField(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value
        self.calls = 0

    def forward(self, state, times, labels):
        del times, labels
        self.calls += 1
        return torch.full_like(state, self.value)


class IdentityCleanField(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, state, times, labels):
        del labels
        self.calls += 1
        scale = times.reshape((len(times),) + (1,) * (state.ndim - 1))
        return state / scale


def test_conditional_pair_short_circuits_exact_endpoints() -> None:
    semantics = resolve_field_semantics(
        protocol=SINGLE_TARGET_PROTOCOL,
        config={"prediction_target": "velocity", "denominator_floor": 1e-3},
        requested_path="auto",
    )
    labels = torch.tensor([1, 2])
    state = torch.zeros(2, 4, 3, 3)

    anchor = ConstantField(2.0)
    other = ConstantField(5.0)
    at_anchor, anchor_counter = conditional_static_pair_velocity(
        anchor,
        other,
        labels,
        anchor_semantics=semantics,
        other_semantics=semantics,
        scale=0.0,
        autocast_dtype=None,
    )
    assert torch.equal(at_anchor(torch.tensor(0.5), state), torch.full_like(state, 2.0))
    assert anchor.calls == 1
    assert other.calls == 0
    assert anchor_counter == {"nfe": 1, "anchor_forwards": 1, "other_forwards": 0}

    anchor = ConstantField(2.0)
    other = ConstantField(5.0)
    at_other, other_counter = conditional_static_pair_velocity(
        anchor,
        other,
        labels,
        anchor_semantics=semantics,
        other_semantics=semantics,
        scale=1.0,
        autocast_dtype=None,
    )
    assert torch.equal(at_other(torch.tensor(0.5), state), torch.full_like(state, 5.0))
    assert anchor.calls == 0
    assert other.calls == 1
    assert other_counter == {"nfe": 1, "anchor_forwards": 0, "other_forwards": 1}


def test_floor_only_control_does_not_evaluate_x_model() -> None:
    velocity_semantics = resolve_field_semantics(
        protocol=SINGLE_TARGET_PROTOCOL,
        config={"prediction_target": "velocity", "denominator_floor": 1e-3},
        requested_path="auto",
    )
    x_semantics = resolve_field_semantics(
        protocol=SINGLE_TARGET_PROTOCOL,
        config={"prediction_target": "x", "denominator_floor": 0.05},
        requested_path="auto",
    )
    anchor = ConstantField(2.0)
    other = ConstantField(5.0)
    velocity, counter = conditional_static_pair_velocity(
        anchor,
        None,
        torch.tensor([1, 2]),
        anchor_semantics=velocity_semantics,
        other_semantics=x_semantics,
        scale=-1.0,
        control_mode="floor_only",
        autocast_dtype=None,
    )
    state = torch.zeros(2, 4, 3, 3)
    actual = velocity(torch.tensor(0.975), state)

    torch.testing.assert_close(actual, torch.full_like(state, 3.0))
    assert anchor.calls == 1
    assert other.calls == 0
    assert counter == {"nfe": 1, "anchor_forwards": 1, "other_forwards": 0}


def test_posterior_response_control_recovers_anchor_for_identity_response() -> None:
    velocity_semantics = resolve_field_semantics(
        protocol=SINGLE_TARGET_PROTOCOL,
        config={"prediction_target": "velocity", "denominator_floor": 1e-3},
        requested_path="auto",
    )
    x_semantics = resolve_field_semantics(
        protocol=SINGLE_TARGET_PROTOCOL,
        config={"prediction_target": "x", "denominator_floor": 0.05},
        requested_path="auto",
    )
    anchor = ConstantField(2.0)
    clean = IdentityCleanField()
    velocity, counter = conditional_static_pair_velocity(
        anchor,
        clean,
        torch.tensor([1, 2]),
        anchor_semantics=velocity_semantics,
        other_semantics=x_semantics,
        scale=1.0,
        control_mode="posterior_response",
        posterior_response_relative_step=0.01,
        autocast_dtype=None,
    )
    state = torch.randn(2, 4, 3, 3)
    actual = velocity(torch.tensor(0.5), state)

    torch.testing.assert_close(actual, torch.full_like(state, 2.0), atol=2e-5, rtol=2e-5)
    assert anchor.calls == 1
    assert clean.calls == 3
    assert counter == {"nfe": 1, "anchor_forwards": 1, "other_forwards": 3}


def test_conditional_common_unique_velocity_evaluates_three_paired_fields() -> None:
    semantics = resolve_field_semantics(
        protocol=SINGLE_TARGET_PROTOCOL,
        config={"prediction_target": "velocity", "denominator_floor": 1e-3},
        requested_path="auto",
    )
    anchor = ConstantField(0.0)
    x_model = ConstantField(-2.0)
    v_model = ConstantField(-4.0)
    velocity, counter = conditional_common_unique_velocity(
        anchor,
        x_model,
        v_model,
        torch.tensor([1, 2]),
        anchor_semantics=semantics,
        x_semantics=semantics,
        v_semantics=semantics,
        scale=1.0,
        component="x_common_on_v",
        autocast_dtype=None,
    )

    state = torch.zeros(2, 4, 3, 3)
    actual = velocity(torch.tensor(0.5), state)

    torch.testing.assert_close(actual, torch.full_like(state, 2.0))
    assert counter == {
        "nfe": 1,
        "anchor_forwards": 1,
        "other_forwards": 1,
        "reference_forwards": 1,
    }


def test_pair_compatibility_allows_only_training_world_size_to_differ() -> None:
    common = {
        "step": 400_000,
        "data_manifest_sha256": "data",
        "official_sit": {"models_sha256": "source"},
    }
    anchor = {
        **common,
        "config": {"global_batch_size": 256, "seed": 0, "world_size": 4},
    }
    other = {
        **common,
        "config": {"global_batch_size": 256, "seed": 0, "world_size": 2},
    }
    metadata = {"checkpoint_step": 400_000, "model_name": "SiT-S/2"}
    validate_pair_compatibility(anchor, other, metadata, metadata)

    incompatible = {**other, "config": {**other["config"], "seed": 1}}
    with pytest.raises(ValueError, match="seed"):
        validate_pair_compatibility(anchor, incompatible, metadata, metadata)


def test_pair_compatibility_can_explicitly_allow_checkpoint_step_mismatch() -> None:
    common_config = {"global_batch_size": 256, "seed": 0, "world_size": 4}
    anchor = {
        "data_manifest_sha256": "data",
        "official_sit": {"models_sha256": "source"},
        "config": common_config,
    }
    other = {**anchor, "config": {**common_config, "world_size": 2}}
    anchor_metadata = {"checkpoint_step": 400_000, "model_name": "SiT-S/2"}
    other_metadata = {"checkpoint_step": 300_000, "model_name": "SiT-S/2"}

    with pytest.raises(ValueError, match="checkpoint_step"):
        validate_pair_compatibility(
            anchor,
            other,
            anchor_metadata,
            other_metadata,
        )
    validate_pair_compatibility(
        anchor,
        other,
        anchor_metadata,
        other_metadata,
        allow_step_mismatch=True,
    )


def test_scale_names_are_stable_for_interpolation_and_extrapolation() -> None:
    assert format_scale(-0.5) == "m0p5"
    assert format_scale(0.0) == "0"
    assert format_scale(1.25) == "1p25"
