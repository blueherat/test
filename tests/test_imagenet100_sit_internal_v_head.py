from __future__ import annotations

import copy

import torch

from experiments.imagenet100_sit_internal_v_head import (
    create_internal_velocity_head,
    extract_internal_features,
    freeze_source_model,
    full_and_internal_velocity,
    full_velocity_from_features,
    internal_velocity_from_features,
    select_internal_guidance_field,
)
from experiments.train_imagenet100_sit_flow import (
    DEFAULT_OFFICIAL_SIT_REPO,
    LATENT_SHAPE,
    NUM_CLASSES,
    load_official_sit_module,
)


def make_sit(sit_module):
    return sit_module.SiT_models["SiT-S/2"](
        input_size=LATENT_SHAPE[-1],
        num_classes=NUM_CLASSES,
        class_dropout_prob=0.1,
    )


def test_official_internal_head_shape_zero_init_and_rng() -> None:
    sit_module, _ = load_official_sit_module(DEFAULT_OFFICIAL_SIT_REPO)
    model = make_sit(sit_module)
    state_before = torch.get_rng_state().clone()

    head = create_internal_velocity_head(
        sit_module,
        model,
        latent_channels=LATENT_SHAPE[0],
    )

    assert torch.equal(torch.get_rng_state(), state_before)
    assert sum(parameter.numel() for parameter in head.parameters()) == 301_840
    assert all(torch.count_nonzero(parameter) == 0 for parameter in head.parameters())


def test_prefix_plus_suffix_exactly_reproduces_source_forward() -> None:
    sit_module, _ = load_official_sit_module(DEFAULT_OFFICIAL_SIT_REPO)
    torch.manual_seed(41)
    model = make_sit(sit_module).eval()
    state = torch.randn(2, *LATENT_SHAPE)
    time_value = torch.tensor([0.2, 0.8])
    labels = torch.tensor([3, 7])

    with torch.inference_mode():
        expected = model(state, time_value, labels)
        features, conditioning = extract_internal_features(
            model,
            state,
            time_value,
            labels,
            internal_depth=8,
        )
        actual = full_velocity_from_features(
            model,
            features,
            conditioning,
            internal_depth=8,
            latent_channels=LATENT_SHAPE[0],
        )

    assert torch.equal(actual, expected)


def test_only_internal_head_receives_gradients() -> None:
    sit_module, _ = load_official_sit_module(DEFAULT_OFFICIAL_SIT_REPO)
    model = freeze_source_model(make_sit(sit_module))
    head = create_internal_velocity_head(
        sit_module,
        model,
        latent_channels=LATENT_SHAPE[0],
    )
    state = torch.randn(2, *LATENT_SHAPE)
    time_value = torch.tensor([0.2, 0.8])
    labels = torch.tensor([3, 7])

    with torch.no_grad():
        features, conditioning = extract_internal_features(
            model,
            state,
            time_value,
            labels,
            internal_depth=8,
        )
    internal = internal_velocity_from_features(
        model,
        head,
        features,
        conditioning,
        latent_channels=LATENT_SHAPE[0],
    )
    internal.square().mean().backward()

    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(parameter.grad is not None for parameter in head.parameters())


def test_single_backbone_pass_returns_exact_full_and_internal() -> None:
    sit_module, _ = load_official_sit_module(DEFAULT_OFFICIAL_SIT_REPO)
    torch.manual_seed(53)
    model = make_sit(sit_module).eval()
    head = create_internal_velocity_head(
        sit_module,
        model,
        latent_channels=LATENT_SHAPE[0],
    ).eval()
    with torch.no_grad():
        head.linear.weight.normal_()
    state = torch.randn(2, *LATENT_SHAPE)
    time_value = torch.tensor([0.3, 0.7])
    labels = torch.tensor([5, 9])

    with torch.inference_mode():
        expected_full = model(state, time_value, labels)
        full, internal = full_and_internal_velocity(
            model,
            head,
            state,
            time_value,
            labels,
            internal_depth=8,
            latent_channels=LATENT_SHAPE[0],
        )

    assert torch.equal(full, expected_full)
    assert internal.shape == expected_full.shape


def test_internal_guidance_field_endpoints_and_extrapolation() -> None:
    full = torch.randn(2, 4, 3, 3)
    internal = torch.randn_like(full)

    assert torch.equal(
        select_internal_guidance_field(full, internal, mode="full"),
        full.float(),
    )
    assert torch.equal(
        select_internal_guidance_field(full, internal, mode="internal"),
        internal.float(),
    )
    assert torch.equal(
        select_internal_guidance_field(
            full,
            internal,
            mode="extrapolation",
            gamma=0.0,
        ),
        full.float(),
    )
    torch.testing.assert_close(
        select_internal_guidance_field(
            full,
            internal,
            mode="extrapolation",
            gamma=0.4,
        ),
        full.float() + 0.4 * (full.float() - internal.float()),
    )


def test_internal_depth_changes_only_auxiliary_readout_location() -> None:
    sit_module, _ = load_official_sit_module(DEFAULT_OFFICIAL_SIT_REPO)
    torch.manual_seed(67)
    model = make_sit(sit_module).eval()
    head = create_internal_velocity_head(
        sit_module,
        model,
        latent_channels=LATENT_SHAPE[0],
    ).eval()
    state = torch.randn(2, *LATENT_SHAPE)
    time_value = torch.tensor([0.25, 0.75])
    labels = torch.tensor([2, 4])

    with torch.inference_mode():
        expected = model(state, time_value, labels)
        full_depth_4, _ = full_and_internal_velocity(
            model,
            copy.deepcopy(head),
            state,
            time_value,
            labels,
            internal_depth=4,
            latent_channels=LATENT_SHAPE[0],
        )
        full_depth_8, _ = full_and_internal_velocity(
            model,
            copy.deepcopy(head),
            state,
            time_value,
            labels,
            internal_depth=8,
            latent_channels=LATENT_SHAPE[0],
        )

    assert torch.equal(full_depth_4, expected)
    assert torch.equal(full_depth_8, expected)
