from __future__ import annotations

import copy

import torch

from experiments.imagenet100_sit_vx_dual_head import (
    VelocityCleanProjection,
    clean_prediction_to_velocity,
    freeze_except_clean_head,
    retrofit_velocity_clean_heads,
    select_velocity_clean_field,
    split_velocity_clean_output,
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


def test_projection_packs_heads_per_patch_pixel() -> None:
    projection = VelocityCleanProjection(
        in_features=1,
        latent_channels=2,
        patch_size=2,
        bias=True,
    )
    with torch.no_grad():
        projection.velocity_head.weight.zero_()
        projection.clean_head.weight.zero_()
        projection.velocity_head.bias.copy_(torch.arange(8))
        projection.clean_head.bias.copy_(100 + torch.arange(8))

    packed = projection(torch.zeros(1, 1, 1)).reshape(1, 1, 2, 2, 4)

    assert torch.equal(packed[..., :2], torch.arange(8).reshape(1, 1, 2, 2, 2))
    assert torch.equal(
        packed[..., 2:],
        (100 + torch.arange(8)).reshape(1, 1, 2, 2, 2),
    )


def test_clean_conversion_recovers_linear_flow_away_from_floor() -> None:
    clean = torch.randn(3, 4, 3, 3)
    noise = torch.randn_like(clean)
    time_value = torch.tensor([0.1, 0.5, 0.9])
    time_image = time_value[:, None, None, None]
    state = (1.0 - time_image) * noise + time_image * clean

    recovered = clean_prediction_to_velocity(
        clean,
        state=state,
        time_value=time_value,
        denominator_floor=0.05,
    )

    assert torch.allclose(recovered, clean - noise, atol=2e-6)


def test_velocity_clean_field_selection_and_extrapolation() -> None:
    state = torch.randn(2, 4, 3, 3)
    velocity = torch.randn_like(state)
    clean = torch.randn_like(state)
    time_value = torch.tensor([0.2, 0.7])
    clean_velocity = clean_prediction_to_velocity(
        clean,
        state=state,
        time_value=time_value,
    )

    selected_velocity = select_velocity_clean_field(
        velocity,
        clean,
        state=state,
        time_value=time_value,
        mode="velocity",
        gamma=99.0,
    )
    selected_clean = select_velocity_clean_field(
        velocity,
        clean,
        state=state,
        time_value=time_value,
        mode="clean",
    )
    extrapolated_zero = select_velocity_clean_field(
        velocity,
        clean,
        state=state,
        time_value=time_value,
        mode="extrapolation",
        gamma=0.0,
    )
    extrapolated = select_velocity_clean_field(
        velocity,
        clean,
        state=state,
        time_value=time_value,
        mode="extrapolation",
        gamma=1.5,
    )

    assert torch.equal(selected_velocity, velocity.float())
    assert torch.equal(extrapolated_zero, velocity.float())
    assert torch.equal(selected_clean, clean_velocity)
    assert torch.allclose(
        extrapolated,
        velocity.float() + 1.5 * (velocity.float() - clean_velocity),
    )


def test_retrofit_preserves_rng_parameters_and_existing_v_output() -> None:
    sit_module, _ = load_official_sit_module(DEFAULT_OFFICIAL_SIT_REPO)
    torch.manual_seed(23)
    original = make_sit(sit_module)
    with torch.no_grad():
        original.final_layer.linear.weight.normal_()
        original.final_layer.linear.bias.normal_()
    dual = copy.deepcopy(original)
    state_before = torch.get_rng_state().clone()
    original_parameter_count = sum(parameter.numel() for parameter in original.parameters())

    retrofit_velocity_clean_heads(dual, latent_channels=LATENT_SHAPE[0])

    assert torch.equal(torch.get_rng_state(), state_before)
    assert sum(parameter.numel() for parameter in dual.parameters()) == original_parameter_count
    assert dual.out_channels == 2 * LATENT_SHAPE[0]
    assert dual.learn_sigma is False
    assert isinstance(dual.final_layer.linear, VelocityCleanProjection)

    original.eval()
    dual.eval()
    state = torch.randn(2, *LATENT_SHAPE)
    time_value = torch.tensor([0.2, 0.8])
    labels = torch.tensor([1, 2])
    with torch.inference_mode():
        original_velocity = original(state, time_value, labels)
        dual_output = dual(state, time_value, labels)
    dual_velocity, _ = split_velocity_clean_output(
        dual_output,
        latent_channels=LATENT_SHAPE[0],
    )

    assert dual_output.shape == (2, 8, *LATENT_SHAPE[1:])
    assert torch.equal(dual_velocity, original_velocity)


def test_shared_initialization_matches_single_head_seed() -> None:
    sit_module, _ = load_official_sit_module(DEFAULT_OFFICIAL_SIT_REPO)
    torch.manual_seed(31)
    baseline = make_sit(sit_module)
    torch.manual_seed(31)
    dual = make_sit(sit_module)
    retrofit_velocity_clean_heads(dual, latent_channels=LATENT_SHAPE[0])

    dual_state = dual.state_dict()
    for key, value in baseline.state_dict().items():
        if key.startswith("final_layer.linear."):
            continue
        assert torch.equal(value, dual_state[key]), key


def test_frozen_probe_only_updates_clean_output_layer() -> None:
    sit_module, _ = load_official_sit_module(DEFAULT_OFFICIAL_SIT_REPO)
    model = make_sit(sit_module)
    retrofit_velocity_clean_heads(model, latent_channels=LATENT_SHAPE[0])
    freeze_except_clean_head(model)

    output = model(
        torch.randn(2, *LATENT_SHAPE),
        torch.tensor([0.2, 0.8]),
        torch.tensor([1, 2]),
    )
    _, clean = split_velocity_clean_output(
        output,
        latent_channels=LATENT_SHAPE[0],
    )
    clean.square().mean().backward()

    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    gradients = [name for name, parameter in model.named_parameters() if parameter.grad is not None]
    assert trainable == [
        "final_layer.linear.clean_head.weight",
        "final_layer.linear.clean_head.bias",
    ]
    assert gradients == trainable
    assert model.training is False
