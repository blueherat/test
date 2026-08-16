from __future__ import annotations

import pytest
import torch

from experiments.imagenet100_sit_hidden_state_extrapolation import (
    frozen_hidden_state_field,
    internal_and_final_hidden_states,
    select_hidden_state_field,
    velocity_from_hidden_state,
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


def test_final_hidden_readout_exactly_reproduces_source_forward() -> None:
    sit_module, _ = load_official_sit_module(DEFAULT_OFFICIAL_SIT_REPO)
    torch.manual_seed(101)
    model = make_sit(sit_module).eval()
    state = torch.randn(2, *LATENT_SHAPE)
    times = torch.tensor([0.25, 0.75])
    labels = torch.tensor([1, 9])

    with torch.inference_mode():
        expected = model(state, times, labels)
        internal, final, conditioning = internal_and_final_hidden_states(
            model,
            state,
            times,
            labels,
            internal_depth=8,
        )
        actual = velocity_from_hidden_state(
            model,
            final,
            conditioning,
            latent_channels=LATENT_SHAPE[0],
        )

    assert internal.shape == final.shape
    assert torch.equal(actual, expected)


def test_zero_gamma_is_bitwise_identical_to_source_forward() -> None:
    sit_module, _ = load_official_sit_module(DEFAULT_OFFICIAL_SIT_REPO)
    torch.manual_seed(103)
    model = make_sit(sit_module).eval()
    state = torch.randn(2, *LATENT_SHAPE)
    times = torch.tensor([0.3, 0.7])
    labels = torch.tensor([2, 8])

    with torch.inference_mode():
        expected = model(state, times, labels).float()
        for extrapolation_space in ("hidden", "output"):
            actual = frozen_hidden_state_field(
                model,
                state,
                times,
                labels,
                internal_depth=8,
                latent_channels=LATENT_SHAPE[0],
                mode="extrapolation",
                gamma=0.0,
                extrapolation_space=extrapolation_space,
            )
            assert torch.equal(actual, expected)


def test_interpolation_endpoints_are_exact() -> None:
    sit_module, _ = load_official_sit_module(DEFAULT_OFFICIAL_SIT_REPO)
    torch.manual_seed(105)
    model = make_sit(sit_module).eval()
    with torch.no_grad():
        model.final_layer.linear.weight.normal_(std=0.1)
    state = torch.randn(2, *LATENT_SHAPE)
    times = torch.tensor([0.3, 0.7])
    labels = torch.tensor([2, 8])

    with torch.inference_mode():
        internal, final, conditioning = internal_and_final_hidden_states(
            model,
            state,
            times,
            labels,
            internal_depth=8,
        )
        expected_final = velocity_from_hidden_state(
            model,
            final,
            conditioning,
            latent_channels=LATENT_SHAPE[0],
        ).float()
        expected_internal = velocity_from_hidden_state(
            model,
            internal,
            conditioning,
            latent_channels=LATENT_SHAPE[0],
        ).float()
        alpha_zero = select_hidden_state_field(
            model,
            internal,
            final,
            conditioning,
            latent_channels=LATENT_SHAPE[0],
            mode="interpolation",
            alpha=0.0,
        )
        alpha_one = select_hidden_state_field(
            model,
            internal,
            final,
            conditioning,
            latent_channels=LATENT_SHAPE[0],
            mode="interpolation",
            alpha=1.0,
        )

    assert torch.equal(alpha_zero, expected_final)
    assert torch.equal(alpha_one, expected_internal)


def test_hidden_and_output_extrapolation_are_distinct_after_adaln() -> None:
    sit_module, _ = load_official_sit_module(DEFAULT_OFFICIAL_SIT_REPO)
    torch.manual_seed(107)
    model = make_sit(sit_module).eval()
    with torch.no_grad():
        model.final_layer.adaLN_modulation[-1].weight.normal_(std=0.1)
        model.final_layer.adaLN_modulation[-1].bias.normal_(std=0.1)
        model.final_layer.linear.weight.normal_(std=0.1)
    token_count = model.x_embedder.num_patches
    hidden_size = model.pos_embed.shape[-1]
    internal = torch.randn(2, token_count, hidden_size)
    final = internal + 0.25 * torch.randn_like(internal)
    conditioning = torch.randn(2, hidden_size)

    with torch.inference_mode():
        hidden = select_hidden_state_field(
            model,
            internal,
            final,
            conditioning,
            latent_channels=LATENT_SHAPE[0],
            mode="extrapolation",
            gamma=0.4,
            extrapolation_space="hidden",
        )
        output = select_hidden_state_field(
            model,
            internal,
            final,
            conditioning,
            latent_channels=LATENT_SHAPE[0],
            mode="extrapolation",
            gamma=0.4,
            extrapolation_space="output",
        )

    assert not torch.allclose(hidden, output)


def test_invalid_hidden_state_protocol_is_rejected() -> None:
    sit_module, _ = load_official_sit_module(DEFAULT_OFFICIAL_SIT_REPO)
    model = make_sit(sit_module).eval()
    hidden = torch.randn(1, 256, model.pos_embed.shape[-1])
    conditioning = torch.randn(1, model.pos_embed.shape[-1])

    with pytest.raises(ValueError, match="gamma is only meaningful"):
        select_hidden_state_field(
            model,
            hidden,
            hidden,
            conditioning,
            latent_channels=LATENT_SHAPE[0],
            mode="final",
            gamma=0.1,
        )
    with pytest.raises(ValueError, match="unsupported extrapolation space"):
        select_hidden_state_field(
            model,
            hidden,
            hidden,
            conditioning,
            latent_channels=LATENT_SHAPE[0],
            mode="extrapolation",
            gamma=0.1,
            extrapolation_space="weights",
        )
