from __future__ import annotations

import torch
import torch.nn.functional as F

from experiments.imagenet100_sit_prediction_targets import (
    native_prediction_target,
    prediction_losses,
    prediction_to_velocity,
)


def make_path() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(19)
    data = torch.randn(5, 4, 3, 3, generator=generator)
    noise = torch.randn(5, 4, 3, 3, generator=generator)
    time_value = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
    time_image = time_value[:, None, None, None]
    state = (1.0 - time_image) * noise + time_image * data
    return data, noise, time_value, state


def test_native_targets_match_linear_path_definition() -> None:
    data, noise, _, _ = make_path()
    assert torch.equal(
        native_prediction_target(
            data=data, noise=noise, prediction_target="velocity"
        ),
        data - noise,
    )
    assert torch.equal(
        native_prediction_target(data=data, noise=noise, prediction_target="x"),
        data,
    )
    assert torch.equal(
        native_prediction_target(
            data=data, noise=noise, prediction_target="epsilon"
        ),
        noise,
    )


def test_exact_native_predictions_recover_identical_velocity() -> None:
    data, noise, time_value, state = make_path()
    expected = data - noise
    predictions = {
        "velocity": expected,
        "x": data,
        "epsilon": noise,
    }
    for target, prediction in predictions.items():
        actual = prediction_to_velocity(
            prediction,
            state=state,
            time_value=time_value,
            prediction_target=target,
            denominator_floor=1e-3,
        )
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)


def test_common_velocity_loss_is_equal_for_same_implied_field() -> None:
    data, noise, time_value, state = make_path()
    generator = torch.Generator().manual_seed(23)
    implied_velocity = torch.randn(state.shape, generator=generator)
    time_image = time_value[:, None, None, None]
    predictions = {
        "velocity": implied_velocity,
        "x": state + (1.0 - time_image) * implied_velocity,
        "epsilon": state - time_image * implied_velocity,
    }
    losses = {}
    for target, prediction in predictions.items():
        losses[target] = prediction_losses(
            prediction,
            state=state,
            data=data,
            noise=noise,
            time_value=time_value,
            prediction_target=target,
            loss_space="velocity",
            denominator_floor=1e-3,
        )["optimized"]
    torch.testing.assert_close(losses["x"], losses["velocity"], rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(
        losses["epsilon"], losses["velocity"], rtol=2e-6, atol=2e-6
    )


def test_velocity_fast_path_is_the_original_direct_mse() -> None:
    data, noise, time_value, state = make_path()
    prediction = torch.randn_like(state)
    actual = prediction_losses(
        prediction,
        state=state,
        data=data,
        noise=noise,
        time_value=time_value,
        prediction_target="velocity",
        loss_space="velocity",
        denominator_floor=1e-3,
    )["optimized"]
    expected = F.mse_loss(prediction.float(), (data - noise).float())
    assert torch.equal(actual, expected)


def test_native_loss_is_direct_mse_for_each_prediction_target() -> None:
    data, noise, time_value, state = make_path()
    prediction = torch.randn_like(state)
    targets = {
        "velocity": data - noise,
        "x": data,
        "epsilon": noise,
    }
    for target, expected_target in targets.items():
        actual = prediction_losses(
            prediction,
            state=state,
            data=data,
            noise=noise,
            time_value=time_value,
            prediction_target=target,
            loss_space="native",
            denominator_floor=1e-3,
        )["optimized"]
        expected = F.mse_loss(prediction.float(), expected_target.float())
        assert torch.equal(actual, expected)


def test_common_velocity_loss_has_the_same_gradient_for_same_implied_field() -> None:
    data, noise, time_value, state = make_path()
    generator = torch.Generator().manual_seed(29)
    implied_velocity = torch.randn(state.shape, generator=generator)
    gradients = {}
    for target in ("velocity", "x", "epsilon"):
        variable = implied_velocity.detach().clone().requires_grad_(True)
        time_image = time_value[:, None, None, None]
        if target == "velocity":
            prediction = variable
        elif target == "x":
            prediction = state + (1.0 - time_image) * variable
        else:
            prediction = state - time_image * variable
        loss = prediction_losses(
            prediction,
            state=state,
            data=data,
            noise=noise,
            time_value=time_value,
            prediction_target=target,
            loss_space="velocity",
            denominator_floor=1e-3,
        )["optimized"]
        gradients[target] = torch.autograd.grad(loss, variable)[0]
    torch.testing.assert_close(gradients["x"], gradients["velocity"], rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(
        gradients["epsilon"], gradients["velocity"], rtol=2e-6, atol=2e-6
    )


def test_endpoint_floor_keeps_x_and_epsilon_fields_finite() -> None:
    data = torch.randn(2, 4, 3, 3)
    noise = torch.randn_like(data)
    state = torch.stack((noise[0], data[1]))
    times = torch.tensor([0.0, 1.0])
    for target in ("x", "epsilon"):
        result = prediction_to_velocity(
            torch.zeros_like(state),
            state=state,
            time_value=times,
            prediction_target=target,
            denominator_floor=1e-3,
        )
        assert torch.isfinite(result).all()
