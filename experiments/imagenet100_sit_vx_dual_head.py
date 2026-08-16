"""Shared-backbone velocity/clean dual heads for ImageNet-100 SiT.

The official SiT final layer produces one flattened patch projection.  This
module replaces only that projection with two explicit linear layers while
retaining the official final AdaLN and the complete transformer backbone.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VelocityCleanProjection(nn.Module):
    """Two output linears with official SiT patch/channel packing."""

    def __init__(
        self,
        *,
        in_features: int,
        latent_channels: int,
        patch_size: int,
        bias: bool,
    ) -> None:
        super().__init__()
        if min(in_features, latent_channels, patch_size) < 1:
            raise ValueError("projection dimensions must be positive")
        self.latent_channels = int(latent_channels)
        self.patch_size = int(patch_size)
        output_features = self.patch_size**2 * self.latent_channels
        self.velocity_head = nn.Linear(in_features, output_features, bias=bias)
        self.clean_head = nn.Linear(in_features, output_features, bias=bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        velocity = self._unflatten_patch(self.velocity_head(hidden))
        clean = self._unflatten_patch(self.clean_head(hidden))
        # SiT unpatchify expects [patch_y, patch_x, channel], so the two heads
        # must be joined on the channel axis rather than concatenated flat.
        return torch.cat((velocity, clean), dim=-1).flatten(-3)

    def _unflatten_patch(self, value: torch.Tensor) -> torch.Tensor:
        return value.reshape(
            *value.shape[:-1],
            self.patch_size,
            self.patch_size,
            self.latent_channels,
        )


def retrofit_velocity_clean_heads(
    model: nn.Module,
    *,
    latent_channels: int,
) -> nn.Module:
    """Replace only SiT's final projection with explicit v/x output heads.

    Official SiT-S/2 defaults to ``learn_sigma=True`` and therefore already
    allocates ``2C`` output channels even though the flow trainer discards the
    second half.  We repurpose those parameters for the clean head, keeping
    parameter count and all shared initialization exactly matched.
    """

    if latent_channels < 1:
        raise ValueError("latent_channels must be positive")
    linear = model.final_layer.linear
    if not isinstance(linear, nn.Linear):
        raise TypeError("expected the untouched official SiT final linear")
    if linear.weight.device.type != "cpu":
        raise ValueError("retrofit the dual heads before moving SiT to CUDA")
    patch_size = int(model.x_embedder.patch_size[0])
    expected_features = patch_size**2 * 2 * latent_channels
    if linear.out_features != expected_features:
        raise ValueError(
            "official final projection must expose 2C channels: "
            f"expected {expected_features}, found {linear.out_features}"
        )

    # nn.Linear initialization advances RNG even though the source weights are
    # copied immediately. Preserve the training data/noise stream exactly.
    with torch.random.fork_rng(devices=[]):
        replacement = VelocityCleanProjection(
            in_features=linear.in_features,
            latent_channels=latent_channels,
            patch_size=patch_size,
            bias=linear.bias is not None,
        )

    # The original flattened order is [patch_y, patch_x, 2C]. Split the two
    # channel groups per patch position, not at the midpoint of the flat axis.
    source_weight = linear.weight.detach().reshape(
        patch_size,
        patch_size,
        2 * latent_channels,
        linear.in_features,
    )
    with torch.no_grad():
        replacement.velocity_head.weight.copy_(
            source_weight[:, :, :latent_channels].reshape_as(
                replacement.velocity_head.weight
            )
        )
        replacement.clean_head.weight.copy_(
            source_weight[:, :, latent_channels:].reshape_as(
                replacement.clean_head.weight
            )
        )
        if linear.bias is not None:
            source_bias = linear.bias.detach().reshape(
                patch_size,
                patch_size,
                2 * latent_channels,
            )
            replacement.velocity_head.bias.copy_(
                source_bias[:, :, :latent_channels].reshape_as(
                    replacement.velocity_head.bias
                )
            )
            replacement.clean_head.bias.copy_(
                source_bias[:, :, latent_channels:].reshape_as(
                    replacement.clean_head.bias
                )
            )

    model.final_layer.linear = replacement
    model.out_channels = 2 * latent_channels
    model.learn_sigma = False
    return model


def split_velocity_clean_output(
    output: torch.Tensor,
    *,
    latent_channels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    expected_channels = 2 * latent_channels
    if output.ndim != 4 or output.shape[1] != expected_channels:
        raise ValueError(
            f"expected dual output [B,{expected_channels},H,W], "
            f"found {tuple(output.shape)}"
        )
    return torch.split(output, latent_channels, dim=1)


def freeze_except_clean_head(model: nn.Module) -> nn.Module:
    """Freeze the loaded velocity model and expose only the clean linear."""

    projection = model.final_layer.linear
    if not isinstance(projection, VelocityCleanProjection):
        raise TypeError("model must be retrofitted before freezing")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in projection.clean_head.parameters():
        parameter.requires_grad_(True)
    # The frozen representation should match sampling-time features. In
    # particular, class-label dropout must remain disabled during the probe.
    model.eval()
    return model


def clean_prediction_to_velocity(
    clean_prediction: torch.Tensor,
    *,
    state: torch.Tensor,
    time_value: torch.Tensor,
    denominator_floor: float = 0.05,
) -> torch.Tensor:
    """Convert the clean head to a stable linear-flow velocity diagnostic."""

    if clean_prediction.shape != state.shape:
        raise ValueError("clean prediction and state must have identical shapes")
    if time_value.shape != (len(state),):
        raise ValueError("time_value must have shape [B]")
    if not 0 < denominator_floor < 0.5:
        raise ValueError("denominator_floor must be in (0, 0.5)")
    time_image = time_value.reshape(-1, *([1] * (state.ndim - 1)))
    denominator = (1.0 - time_image).clamp_min(float(denominator_floor))
    return (clean_prediction.float() - state.float()) / denominator


def select_velocity_clean_field(
    velocity_prediction: torch.Tensor,
    clean_prediction: torch.Tensor,
    *,
    state: torch.Tensor,
    time_value: torch.Tensor,
    mode: str,
    gamma: float = 0.0,
    denominator_floor: float = 0.05,
) -> torch.Tensor:
    """Select one head or extrapolate from the clean-derived field.

    ``gamma`` follows AutoGuidance notation: positive values move beyond the
    frozen velocity head, away from the weaker clean-derived field.
    """

    if velocity_prediction.shape != state.shape:
        raise ValueError("velocity prediction and state must have identical shapes")
    if mode == "velocity":
        return velocity_prediction.float()
    clean_velocity = clean_prediction_to_velocity(
        clean_prediction,
        state=state,
        time_value=time_value,
        denominator_floor=denominator_floor,
    )
    if mode == "clean":
        return clean_velocity
    if mode != "extrapolation":
        raise ValueError(f"unsupported velocity/clean field mode: {mode!r}")
    if not torch.isfinite(torch.tensor(float(gamma))):
        raise ValueError("gamma must be finite")
    if float(gamma) == 0.0:
        return velocity_prediction.float()
    velocity = velocity_prediction.float()
    return velocity + float(gamma) * (velocity - clean_velocity)
