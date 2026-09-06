"""Small full-channel vector field for a teacher-paired finite-step bridge.

The field predicts a = E[(W-Y)/beta | U_tau,t,s,tau,c]. Its only state
input is the CURRENT U_tau. It has no frozen Y/Z/backbone feature input.
Production: C=1024, 16x16, two full-width residual blocks, 3,652,224 params.
Widths and initialization are fixed numerical choices, not theorem constants.
All precision/autocast policy belongs to the caller. Native arithmetic helpers
require FP32 states and retain the original Euler order and t-floor.
"""
from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor, nn

TIME_FLOOR = 0.05


def _state(value: Tensor, name: str, *, fp32: bool = False) -> None:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating tensor")
    if value.ndim != 4 or any(size == 0 for size in value.shape):
        raise ValueError(f"{name} must be nonempty [B,C,H,W]")
    if fp32 and value.dtype != torch.float32:
        raise ValueError(f"{name} must be FP32 for native arithmetic")


def _matching(reference: Tensor, value: Tensor, name: str) -> None:
    _state(value, name, fp32=True)
    if value.shape != reference.shape or value.device != reference.device:
        raise ValueError(f"{name} must match the state shape and device")


def _time(value: Tensor | float, like: Tensor, name: str) -> Tensor:
    if isinstance(value, Tensor) and value.device != like.device:
        raise ValueError(f"{name} must share the state's device")
    result = torch.as_tensor(value, dtype=like.dtype, device=like.device)
    if result.ndim == 0:
        result = result.expand(len(like))
    if result.shape != (len(like),):
        raise ValueError(f"{name} must be scalar or have shape [B]")
    return result


def _pair(t: Tensor | float, s: Tensor | float, like: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    t, s = _time(t, like, "t"), _time(s, like, "s")
    valid = torch.isfinite(t) & torch.isfinite(s) & (s >= 0) & (s < t) & (t <= 1)
    if not bool(valid.all()):
        raise ValueError("times must satisfy finite 0 <= s < t <= 1")
    beta = (t - s) / t.clamp_min(TIME_FLOOR)
    return t, s, beta


def _tau(value: Tensor | float, like: Tensor) -> Tensor:
    tau = _time(value, like, "tau")
    if not bool((torch.isfinite(tau) & (tau >= 0) & (tau <= 1)).all()):
        raise ValueError("tau must be finite and in [0,1]")
    return tau


def _broadcast(value: Tensor) -> Tensor:
    return value[:, None, None, None]


class _FullChannelBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pointwise = nn.Conv2d(channels, channels, 1)
        self.activation = nn.SiLU(inplace=False)

    def forward(self, hidden: Tensor) -> Tensor:
        return hidden + self.pointwise(self.activation(self.depthwise(hidden)))


class PairedBridgeField(nn.Module):
    """Return normalized bridge velocity, with a zero-initialized full readout.

    cond = MLP(concat(class64, sincos32(1000*t), sincos32(1000*s),
                      sincos32(1000*tau), sincos32(1000*beta)))
    h = U + cond + position
    h = h + pointwise_C_to_C(SiLU(depthwise3x3(h)))  [twice]
    a = readout_C_to_C(h)

    beta=(t-s)/max(t,.05) is computed internally. Each sinusoidal embedding
    concatenates cos then sin, frequencies exp(-log(10000)*arange(16)/16).
    Class/linear/convolution layers use PyTorch defaults; position and readout
    weight AND bias start at zero. No normalization, clipping, or randomness
    occurs in forward. The two spatial blocks have a 5x5 receptive field.
    Small C/spatial sizes/classes exist only for CPU unit tests.
    """
    class_features = 64
    time_features = 32
    condition_hidden = 128

    def __init__(
        self,
        *,
        latent_channels: int = 1024,
        spatial_size: tuple[int, int] = (16, 16),
        num_classes: int = 1000,
    ) -> None:
        super().__init__()
        for name, value in (("latent_channels", latent_channels), ("num_classes", num_classes)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (not isinstance(spatial_size, tuple) or len(spatial_size) != 2
                or any(type(value) is not int or value <= 0 for value in spatial_size)):
            raise ValueError("spatial_size must be a pair of positive integers")
        self.latent_channels = latent_channels
        self.spatial_size = spatial_size
        self.num_classes = num_classes
        self.class_embedding = nn.Embedding(num_classes, self.class_features)
        self.condition_mlp = nn.Sequential(
            nn.Linear(self.class_features + 4 * self.time_features, self.condition_hidden),
            nn.SiLU(inplace=False),
            nn.Linear(self.condition_hidden, latent_channels),
        )
        self.position = nn.Parameter(torch.zeros(1, latent_channels, *spatial_size))
        self.blocks = nn.ModuleList([_FullChannelBlock(latent_channels) for _ in range(2)])
        self.readout = nn.Conv2d(latent_channels, latent_channels, 1)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)
        self.register_buffer("time_frequencies", torch.exp(
            -math.log(10000.0) * torch.arange(self.time_features // 2) / (self.time_features // 2),
        ))

    def _embed_time(self, value: Tensor) -> Tensor:
        angles = (1000.0 * value[:, None]) * self.time_frequencies[None, :]
        return torch.cat((angles.cos(), angles.sin()), dim=-1)

    def forward(
        self,
        u: Tensor,
        t: Tensor | float,
        s: Tensor | float,
        tau: Tensor | float,
        labels: Tensor,
    ) -> Tensor:
        _state(u, "u")
        if tuple(u.shape[1:]) != (self.latent_channels, *self.spatial_size):
            raise ValueError("u must have the configured channel and spatial shape")
        if u.dtype != self.readout.weight.dtype or u.device != self.readout.weight.device:
            raise ValueError("u must match the field parameter dtype and device")
        if not isinstance(labels, Tensor) or labels.dtype != torch.long:
            raise TypeError("labels must be a torch.long tensor")
        if labels.shape != (len(u),) or labels.device != u.device:
            raise ValueError("labels must have shape [B] and share u's device")
        if not bool(((labels >= 0) & (labels < self.num_classes)).all()):
            raise ValueError("labels outside the configured class range")
        t, s, beta = _pair(t, s, u)
        tau = _tau(tau, u)
        condition = self.condition_mlp(torch.cat((
            self.class_embedding(labels),
            self._embed_time(t),
            self._embed_time(s),
            self._embed_time(tau),
            self._embed_time(beta),
        ), dim=-1))
        hidden = u + condition[:, :, None, None] + self.position
        for block in self.blocks:
            hidden = block(hidden)
        return self.readout(hidden)


def native_euler(z: Tensor, g: Tensor, t: Tensor | float, s: Tensor | float) -> Tensor:
    """FP32 z - (t-s)*((z-g)/max(t,.05)); do not factor into beta*(z-g)."""
    _state(z, "z", fp32=True)
    _matching(z, g, "g")
    t, s, _ = _pair(t, s, z)
    return z - _broadcast(t - s) * ((z - g) / _broadcast(t.clamp_min(TIME_FLOOR)))


def construct_bridge(
    clean: Tensor,
    noise: Tensor,
    g: Tensor,
    t: Tensor | float,
    s: Tensor | float,
    tau: Tensor | float,
) -> dict[str, Tensor]:
    """Construct one fixed teacher coupling using the same clean/noise pair.

    g must be the native clean prediction evaluated at the returned z. The
    caller owns that evaluation and freezing the teacher. R is computed as
    W-Y, never beta*(clean-g), because the native denominator has a floor.
    Shapes: beta [B]; all other returned tensors [B,C,H,W]. Target is R/beta.
    No targets are fitted here and no decoder or model calls occur.
    """
    _state(clean, "clean", fp32=True)
    _matching(clean, noise, "noise")
    _matching(clean, g, "g")
    t, s, beta = _pair(t, s, clean)
    tau = _tau(tau, clean)
    z = (1 - _broadcast(t)) * clean + _broadcast(t) * noise
    y = native_euler(z, g, t, s)
    w = (1 - _broadcast(s)) * clean + _broadcast(s) * noise
    residual = w - y
    u = (1 - _broadcast(tau)) * y + _broadcast(tau) * w
    return {
        "z": z, "Y": y, "W": w, "R": residual, "U": u,
        "beta": beta, "target": residual / _broadcast(beta),
    }


def bridge_midpoint(
    field: Callable,
    y: Tensor,
    t: Tensor | float,
    s: Tensor | float,
    labels: Tensor,
    *,
    tau_locked_zero: bool = False,
) -> Tensor:
    """One explicit midpoint step across auxiliary tau in [0,1].

    k1=beta*a(y,0), mid=y+.5*k1, k2=beta*a(mid,.5), result=y+k2.
    The matched mean-only control queries tau=0 at BOTH evolving states.
    It is not a single frozen conditional-mean projection. Both modes make
    exactly two field forwards; beta is applied once per velocity evaluation.
    """
    _state(y, "y", fp32=True)
    if type(tau_locked_zero) is not bool:
        raise TypeError("tau_locked_zero must be bool")
    t, s, beta = _pair(t, s, y)
    tau0 = torch.zeros_like(t)
    first = field(y, t, s, tau0, labels)
    _matching(y, first, "field output")
    k1 = _broadcast(beta) * first
    mid = y + 0.5 * k1
    second_tau = tau0 if tau_locked_zero else torch.full_like(t, 0.5)
    second = field(mid, t, s, second_tau, labels)
    _matching(y, second, "field output")
    k2 = _broadcast(beta) * second
    return y + k2
