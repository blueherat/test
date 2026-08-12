from __future__ import annotations

import pytest
import torch

from experiments.imagenet100_sit_static_pair import (
    DUAL_OUTPUT_PROTOCOL,
    LEGACY_PROTOCOL,
    SINGLE_TARGET_PROTOCOL,
    output_to_field_velocity,
    resolve_field_semantics,
    static_pair_velocity,
)
from experiments.sample_imagenet100_sit_static_pair_fid import (
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


def test_scale_names_are_stable_for_interpolation_and_extrapolation() -> None:
    assert format_scale(-0.5) == "m0p5"
    assert format_scale(0.0) == "0"
    assert format_scale(1.25) == "1p25"
