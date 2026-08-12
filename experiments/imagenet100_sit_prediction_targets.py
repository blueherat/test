"""Prediction-target conversions for the ImageNet-100 linear SiT path."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F


PredictionTarget = Literal["velocity", "x", "epsilon"]
LossSpace = Literal["velocity", "native"]
PREDICTION_TARGETS = ("velocity", "x", "epsilon")
LOSS_SPACES = ("velocity", "native")


def _time_image(time_value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if time_value.shape != (reference.shape[0],):
        raise ValueError("time_value must have shape [B]")
    return time_value.reshape(-1, *([1] * (reference.ndim - 1)))


def native_prediction_target(
    *,
    data: torch.Tensor,
    noise: torch.Tensor,
    prediction_target: PredictionTarget,
) -> torch.Tensor:
    if data.shape != noise.shape:
        raise ValueError("data and noise must have identical shapes")
    if prediction_target == "velocity":
        return data - noise
    if prediction_target == "x":
        return data
    if prediction_target == "epsilon":
        return noise
    raise ValueError(f"unsupported prediction target: {prediction_target}")


def prediction_to_velocity(
    prediction: torch.Tensor,
    *,
    state: torch.Tensor,
    time_value: torch.Tensor,
    prediction_target: PredictionTarget,
    denominator_floor: float,
) -> torch.Tensor:
    """Convert a native model output to the linear-path velocity field."""

    if prediction.shape != state.shape:
        raise ValueError("prediction and state must have identical shapes")
    if denominator_floor <= 0 or denominator_floor >= 0.5:
        raise ValueError("denominator_floor must be in (0, 0.5)")
    if prediction_target == "velocity":
        return prediction.float()

    time_image = _time_image(time_value, state).to(dtype=state.dtype)
    if prediction_target == "x":
        return (prediction.float() - state.float()) / (
            1.0 - time_image
        ).clamp_min(denominator_floor)
    if prediction_target == "epsilon":
        return (state.float() - prediction.float()) / time_image.clamp_min(
            denominator_floor
        )
    raise ValueError(f"unsupported prediction target: {prediction_target}")


def prediction_losses(
    prediction: torch.Tensor,
    *,
    state: torch.Tensor,
    data: torch.Tensor,
    noise: torch.Tensor,
    time_value: torch.Tensor,
    prediction_target: PredictionTarget,
    loss_space: LossSpace,
    denominator_floor: float,
) -> dict[str, torch.Tensor]:
    """Return the optimized loss plus native- and velocity-space diagnostics."""

    if not (prediction.shape == state.shape == data.shape == noise.shape):
        raise ValueError("prediction, state, data, and noise shapes must match")
    native_target = native_prediction_target(
        data=data,
        noise=noise,
        prediction_target=prediction_target,
    )
    native_loss = F.mse_loss(
        prediction.float(), native_target.float(), reduction="mean"
    )

    # Preserve the original SiT velocity objective as a literal direct MSE.
    if prediction_target == "velocity":
        velocity_prediction = prediction.float()
        velocity_target = data.float() - noise.float()
        velocity_loss = F.mse_loss(
            velocity_prediction, velocity_target, reduction="mean"
        )
    else:
        velocity_prediction = prediction_to_velocity(
            prediction,
            state=state,
            time_value=time_value,
            prediction_target=prediction_target,
            denominator_floor=denominator_floor,
        )
        # JiT converts both the prediction and its exact native target with
        # the same clamped denominator. This keeps an exact x/epsilon
        # predictor at zero loss even inside the endpoint clamp region.
        velocity_target = prediction_to_velocity(
            native_target,
            state=state,
            time_value=time_value,
            prediction_target=prediction_target,
            denominator_floor=denominator_floor,
        )
        velocity_loss = F.mse_loss(
            velocity_prediction, velocity_target, reduction="mean"
        )

    if loss_space == "velocity":
        optimized = velocity_loss
    elif loss_space == "native":
        optimized = native_loss
    else:
        raise ValueError(f"unsupported loss space: {loss_space}")
    return {
        "optimized": optimized,
        "native": native_loss,
        "velocity": velocity_loss,
    }
