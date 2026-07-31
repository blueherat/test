"""Contrast-preserving common residual adapter for RAEv2."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F


COMMON_ADAPTER_FORMAT = "raev2_contrast_preserving_common_adapter_v1"


def _channel_rms_normalize(
    value: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    scale = value.float().square().mean(dim=1, keepdim=True).add(float(eps)).rsqrt()
    return value * scale.to(dtype=value.dtype)


def _time_features(
    time: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    if time.ndim != 1:
        raise ValueError("time must contain one scalar per sample")
    time = time.to(dtype=dtype)
    return torch.stack(
        (
            time,
            time.square(),
            torch.sin(math.pi * time),
            torch.cos(math.pi * time),
        ),
        dim=1,
    )


def internal_guidance_prediction(
    full: torch.Tensor,
    base: torch.Tensor,
    time: torch.Tensor,
    *,
    scale: float,
    interval: tuple[float, float],
) -> torch.Tensor:
    """Reproduce RAEv2's official internal-guidance head combination."""

    if full.shape != base.shape:
        raise ValueError("full and base predictions must have identical shapes")
    if time.shape != (full.shape[0],):
        raise ValueError("time must have shape [B]")
    lower, upper = (float(value) for value in interval)
    if not lower < upper:
        raise ValueError("internal-guidance interval must satisfy lower < upper")
    active = ((time >= lower) & (time <= upper)).view(
        -1,
        *([1] * (full.ndim - 1)),
    )
    return torch.where(
        active,
        base + float(scale) * (full - base),
        full,
    )


def forward_with_internalguidance_common_adapter(
    source_model: nn.Module,
    adapter: "CommonResidualAdapter",
    noisy_latent: torch.Tensor,
    time: torch.Tensor,
    *,
    ig_scale: float,
    ig_interval: tuple[float, float] = (0.0, 1.0),
    **condition_kwargs: torch.Tensor | None,
) -> torch.Tensor:
    """Apply the adapter after reproducing the source model's IG arithmetic.

    Computing ``full + a`` and ``base + a`` separately in BF16 can perturb
    ``full - base`` through rounding.  This path first computes the frozen
    source's guided output exactly as the official sampler does, then adds the
    one common correction.  A zero adapter is therefore value-identical to the
    official source sampler.
    """

    half_batch = noisy_latent.shape[0] // 2
    if half_batch <= 0 or noisy_latent.shape[0] != 2 * half_batch:
        raise ValueError("internal-guidance sampling expects an even batch")
    half = noisy_latent[:half_batch]
    half_time = time[:half_batch]
    half_kwargs = {
        key: value[:half_batch] if value is not None else None
        for key, value in condition_kwargs.items()
    }
    with torch.no_grad():
        source_output = source_model(half, half_time, **half_kwargs)
    if not (isinstance(source_output, tuple) and len(source_output) == 2):
        raise ValueError("source model must return (full, base)")
    full, base = source_output
    correction = adapter(half, half_time, full, base)

    channels = int(getattr(source_model, "in_channels"))
    source_full = full[:, :channels]
    source_base = base[:, :channels]
    source_guided = internal_guidance_prediction(
        source_full,
        source_base,
        half_time,
        scale=float(ig_scale),
        interval=ig_interval,
    )
    adapted = source_guided.float() + correction[:, :channels].float()
    return torch.cat((adapted, adapted), dim=0)


class CommonResidualAdapter(nn.Module):
    """Predict one correction that is added identically to full and base.

    The adapter observes the noisy state and the frozen model's two predictions.
    Its final projection is zero initialized, so wrapping a source checkpoint is
    initially an exact identity operation.
    """

    def __init__(
        self,
        channels: int,
        *,
        hidden_channels: int = 64,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")

        self.channels = int(channels)
        self.hidden_channels = int(hidden_channels)
        self.eps = float(eps)
        groups = math.gcd(self.hidden_channels, 8)

        self.input_projection = nn.Conv2d(
            3 * self.channels,
            self.hidden_channels,
            kernel_size=1,
        )
        self.input_norm = nn.GroupNorm(groups, self.hidden_channels)
        self.time_projection = nn.Sequential(
            nn.Linear(4, self.hidden_channels),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels),
        )
        self.spatial_mixing = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            kernel_size=3,
            padding=1,
            groups=self.hidden_channels,
        )
        self.output_projection = nn.Conv2d(
            self.hidden_channels,
            self.channels,
            kernel_size=1,
        )
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def config_dict(self) -> dict[str, int | float]:
        return {
            "channels": self.channels,
            "hidden_channels": self.hidden_channels,
            "eps": self.eps,
        }

    def forward(
        self,
        noisy_latent: torch.Tensor,
        time: torch.Tensor,
        full: torch.Tensor,
        base: torch.Tensor,
    ) -> torch.Tensor:
        expected = (noisy_latent.shape[0], self.channels, *noisy_latent.shape[2:])
        for name, value in (
            ("noisy_latent", noisy_latent),
            ("full", full),
            ("base", base),
        ):
            if value.ndim != 4:
                raise ValueError(f"{name} must have shape [B, C, H, W]")
            if value.shape != expected:
                raise ValueError(
                    f"{name} shape {tuple(value.shape)} does not match {expected}"
                )
        if time.shape != (noisy_latent.shape[0],):
            raise ValueError("time must have shape [B]")

        inputs = torch.cat(
            tuple(
                _channel_rms_normalize(value, eps=self.eps)
                for value in (noisy_latent, full, base)
            ),
            dim=1,
        )
        hidden = self.input_projection(inputs)
        hidden = self.input_norm(hidden)
        time_bias = self.time_projection(
            _time_features(time, dtype=hidden.dtype)
        ).unsqueeze(-1).unsqueeze(-1)
        hidden = F.silu(hidden + time_bias)
        hidden = hidden + self.spatial_mixing(hidden)
        return self.output_projection(F.silu(hidden))


class ContrastPreservingCommonAdapterModel(nn.Module):
    """Freeze a dual-head model and add the same learned residual to both heads."""

    def __init__(
        self,
        source_model: nn.Module,
        adapter: CommonResidualAdapter,
    ) -> None:
        super().__init__()
        source_channels = int(getattr(source_model, "in_channels"))
        if source_channels != adapter.channels:
            raise ValueError(
                "source model and adapter channel counts differ: "
                f"{source_channels} != {adapter.channels}"
            )
        self.source_model = source_model
        self.adapter = adapter
        self.source_model.requires_grad_(False)
        self.source_model.eval()

    @property
    def in_channels(self) -> int:
        return self.adapter.channels

    def train(self, mode: bool = True):
        super().train(mode)
        self.source_model.eval()
        self.adapter.train(mode)
        return self

    def trainable_state_dict(self) -> Mapping[str, torch.Tensor]:
        return self.adapter.state_dict()

    def forward(
        self,
        noisy_latent: torch.Tensor,
        time: torch.Tensor,
        return_intermediate: bool = False,
        **condition_kwargs: torch.Tensor | None,
    ):
        with torch.no_grad():
            source_output = self.source_model(
                noisy_latent,
                time,
                return_intermediate=return_intermediate,
                **condition_kwargs,
            )

        intermediate = None
        if return_intermediate:
            if not (
                isinstance(source_output, tuple)
                and len(source_output) == 2
                and isinstance(source_output[0], tuple)
            ):
                raise ValueError(
                    "source model must return ((full, base), intermediate)"
                )
            source_output, intermediate = source_output
        if not (isinstance(source_output, tuple) and len(source_output) == 2):
            raise ValueError("source model must return (full, base)")
        full, base = source_output
        correction = self.adapter(noisy_latent, time, full, base)
        correction = correction.float()
        corrected = (
            full.float() + correction,
            base.float() + correction,
        )
        if return_intermediate:
            return corrected, intermediate
        return corrected


def load_common_adapter_checkpoint(
    adapter: CommonResidualAdapter,
    checkpoint: Mapping[str, object],
    *,
    state_key: str = "adapter",
) -> None:
    if checkpoint.get("format") != COMMON_ADAPTER_FORMAT:
        raise ValueError("unsupported common-adapter checkpoint format")
    stored_config = checkpoint.get("adapter_config")
    if stored_config != adapter.config_dict():
        raise ValueError(
            "adapter config mismatch: "
            f"checkpoint={stored_config}, requested={adapter.config_dict()}"
        )
    state = checkpoint.get(state_key)
    if not isinstance(state, Mapping):
        raise KeyError(f"checkpoint has no adapter state {state_key!r}")
    adapter.load_state_dict(state, strict=True)
