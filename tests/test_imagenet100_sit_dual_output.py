from __future__ import annotations

import torch

from experiments.imagenet100_sit_dual_output import (
    dual_output_flow_losses,
    dual_output_velocities,
    retrofit_dual_output_head,
)
from experiments.sample_imagenet100_sit_dual_fid import conditional_dual_velocity
from experiments.train_imagenet100_sit_flow import (
    DEFAULT_OFFICIAL_SIT_REPO,
    LATENT_SHAPE,
    NUM_CLASSES,
    load_official_sit_module,
)


def pack_output(
    epsilon: torch.Tensor,
    clean: torch.Tensor,
    gate_logits: torch.Tensor,
) -> torch.Tensor:
    return torch.cat((epsilon, clean, gate_logits), dim=1)


def test_exact_dual_predictions_recover_linear_flow_velocity() -> None:
    generator = torch.Generator().manual_seed(7)
    clean = torch.randn(3, 4, 5, 5, generator=generator)
    epsilon = torch.randn(3, 4, 5, 5, generator=generator)
    time_value = torch.tensor([0.2, 0.5, 0.8])
    time_image = time_value[:, None, None, None]
    state = (1.0 - time_image) * epsilon + time_image * clean
    output = pack_output(epsilon, clean, torch.zeros(3, 1, 5, 5))

    losses = dual_output_flow_losses(
        output,
        clean_target=clean,
        epsilon_target=epsilon,
        time_value=time_value,
    )
    velocities = dual_output_velocities(
        output,
        state=state,
        time_value=time_value,
    )
    target = clean - epsilon

    assert losses["total"].item() == 0.0
    assert torch.allclose(velocities["x"], target, atol=1e-6)
    assert torch.allclose(velocities["epsilon"], target, atol=1e-6)
    assert torch.allclose(velocities["dynamic"], target, atol=1e-6)


def test_dynamic_velocity_is_finite_and_exact_at_flow_endpoints() -> None:
    clean = torch.randn(2, 4, 3, 3)
    epsilon = torch.randn_like(clean)
    time_value = torch.tensor([0.0, 1.0])
    state = torch.stack((epsilon[0], clean[1]))
    output = pack_output(epsilon, clean, torch.zeros(2, 1, 3, 3))

    velocity = dual_output_velocities(
        output,
        state=state,
        time_value=time_value,
    )["dynamic"]

    assert torch.isfinite(velocity).all()
    assert torch.allclose(velocity, clean - epsilon, atol=1e-6)


def test_gate_loss_stop_gradient_only_updates_gate_logits() -> None:
    output = torch.randn(2, 9, 3, 3, requires_grad=True)
    clean = torch.randn(2, 4, 3, 3)
    epsilon = torch.randn(2, 4, 3, 3)
    losses = dual_output_flow_losses(
        output,
        clean_target=clean,
        epsilon_target=epsilon,
        time_value=torch.tensor([0.3, 0.7]),
    )

    losses["gate"].backward()

    assert output.grad is not None
    assert torch.count_nonzero(output.grad[:, :8]).item() == 0
    assert torch.count_nonzero(output.grad[:, 8:]).item() > 0


def test_gate_prefers_the_exact_branch() -> None:
    clean = torch.ones(1, 4, 2, 2)
    epsilon = torch.zeros_like(clean)
    wrong_epsilon = torch.ones_like(epsilon)
    time_value = torch.tensor([0.5])
    gate_near_x = torch.full((1, 1, 2, 2), 10.0)
    gate_near_epsilon = torch.full((1, 1, 2, 2), -10.0)

    x_gate_loss = dual_output_flow_losses(
        pack_output(wrong_epsilon, clean, gate_near_x),
        clean_target=clean,
        epsilon_target=epsilon,
        time_value=time_value,
    )["gate"]
    epsilon_gate_loss = dual_output_flow_losses(
        pack_output(wrong_epsilon, clean, gate_near_epsilon),
        clean_target=clean,
        epsilon_target=epsilon,
        time_value=time_value,
    )["gate"]

    assert x_gate_loss < epsilon_gate_loss


def test_retrofit_preserves_rng_and_emits_two_c_plus_one_channels() -> None:
    sit_module, _ = load_official_sit_module(
        DEFAULT_OFFICIAL_SIT_REPO
    )
    torch.manual_seed(11)
    model = sit_module.SiT_models["SiT-S/2"](
        input_size=LATENT_SHAPE[-1],
        num_classes=NUM_CLASSES,
        class_dropout_prob=0.1,
    )
    state_before = torch.get_rng_state().clone()
    retrofit_dual_output_head(model, latent_channels=LATENT_SHAPE[0])

    assert torch.equal(torch.get_rng_state(), state_before)
    assert model.out_channels == 9
    assert model.learn_sigma is False
    output = model(
        torch.randn(2, *LATENT_SHAPE),
        torch.tensor([0.2, 0.8]),
        torch.tensor([1, 2]),
    )
    assert output.shape == (2, 9, *LATENT_SHAPE[1:])


class ExactEndpointModel(torch.nn.Module):
    def forward(self, state, time_value, labels):
        del labels
        time_image = time_value[:, None, None, None]
        epsilon = torch.zeros_like(state)
        clean = torch.ones_like(state)
        expected_state = time_image.expand_as(state)
        assert torch.allclose(state, expected_state)
        gate_logits = torch.zeros(
            len(state), 1, state.shape[2], state.shape[3], device=state.device
        )
        return torch.cat((epsilon, clean, gate_logits), dim=1)


def test_fid_sampling_modes_recover_the_same_exact_velocity() -> None:
    state = torch.full((2, 4, 3, 3), 0.4)
    labels = torch.tensor([1, 2])
    for mode in ("x", "epsilon", "dynamic"):
        velocity, counter = conditional_dual_velocity(
            ExactEndpointModel(),
            labels,
            mode=mode,
            gate_activation="sigmoid",
            denominator_floor=1e-3,
            autocast_dtype=None,
        )
        result = velocity(torch.tensor(0.4), state)
        assert torch.allclose(result, torch.ones_like(result))
        assert counter == {"nfe": 1}
