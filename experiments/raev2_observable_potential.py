"""Scalar-potential approximation to RAEv2's observable clean error.

The production architecture is fixed at latent [B,1024,16,16], hidden width
128, 1000 classes, and a 32-dimensional sinusoidal time embedding. Its widths
are numerical approximation choices, not consequences of a transport theorem.
The correction is c = grad_z Phi; the sampler adds c to its supplied official
clean prediction, equivalently c/t to an unclamped dataward velocity. This
module adds no guidance gain, time window, sampler, or backbone features.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class ScalarGuidancePotential(nn.Module):
    """A batch-separable scalar potential with a zero-initialized readout.

Exact layer order:
  h = SiLU(input_1x1(z)) + class(y) + time_mlp(sincos32(t)) + position
  h = SiLU(conv3x3(h)); h = SiLU(conv3x3(h))
  Phi = readout_1x1(h).sum(channel, height, width)

Time is the ORIGINAL RAEv2 noise time t in [0,1], with no rescaling.
Frequencies are exp(-log(10000)*arange(16)/16), concatenating cos then sin.
The time MLP is 32 -> hidden -> hidden, with one non-inplace SiLU.
Position and the bias-free readout weight start at zero; all other layers use
their PyTorch defaults. There is no normalization or stochastic layer.

Nondefault dimensions are available for small CPU numerical tests; they do not
specify alternative production candidates. Default trainable parameter count:
608000. Inputs and module parameters must share device and floating dtype.
Precision/autocast is controlled by the caller.
"""

    time_features = 32

    def __init__(
        self,
        *,
        latent_channels: int = 1024,
        spatial_size: tuple[int, int] = (16, 16),
        hidden_channels: int = 128,
        num_classes: int = 1000,
    ) -> None:
        super().__init__()
        for name, value in (("latent_channels", latent_channels),
                            ("hidden_channels", hidden_channels),
                            ("num_classes", num_classes)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (not isinstance(spatial_size, tuple) or len(spatial_size) != 2
                or any(type(value) is not int or value <= 0 for value in spatial_size)):
            raise ValueError("spatial_size must be a pair of positive integers")
        self.latent_channels = latent_channels
        self.spatial_size = spatial_size
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes

        self.input_conv = nn.Conv2d(latent_channels, hidden_channels, 1)
        self.class_embedding = nn.Embedding(num_classes, hidden_channels)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_features, hidden_channels),
            nn.SiLU(inplace=False),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.position = nn.Parameter(torch.zeros(1, hidden_channels, *spatial_size))
        self.hidden_convs = nn.ModuleList([
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
        ])
        self.activation = nn.SiLU(inplace=False)
        self.readout = nn.Conv2d(hidden_channels, 1, 1, bias=False)
        nn.init.zeros_(self.readout.weight)
        frequencies = torch.exp(
            -math.log(10000.0) * torch.arange(self.time_features // 2)
            / (self.time_features // 2)
        )
        self.register_buffer("time_frequencies", frequencies)

    def _inputs(self, z: Tensor, t: Tensor | float, y: Tensor) -> tuple[Tensor, Tensor]:
        if not isinstance(z, Tensor) or not z.is_floating_point():
            raise TypeError("z must be a floating-point tensor")
        if (z.ndim != 4 or z.shape[0] == 0
                or tuple(z.shape[1:]) != (self.latent_channels, *self.spatial_size)):
            raise ValueError("z must be nonempty [B,C,H,W] with the configured latent shape")
        if z.device != self.input_conv.weight.device or z.dtype != self.input_conv.weight.dtype:
            raise ValueError("z must match the potential parameter device and dtype")
        if not isinstance(y, Tensor) or y.dtype != torch.long:
            raise TypeError("y must be a torch.long class tensor")
        if y.shape != (len(z),) or y.device != z.device:
            raise ValueError("y must have shape [B] and share z's device")
        if bool(((y < 0) | (y >= self.num_classes)).any()):
            raise ValueError("class labels are outside the configured range")
        time = torch.as_tensor(t, dtype=z.dtype, device=z.device)
        if time.ndim == 0:
            time = time.expand(len(z))
        if time.shape != (len(z),):
            raise ValueError("t must be scalar or have shape [B]")
        if not bool(torch.isfinite(time).all()) or bool(((time < 0) | (time > 1)).any()):
            raise ValueError("original noise time t must be finite and in [0,1]")
        return time, y

    def forward(self, z: Tensor, t: Tensor | float, y: Tensor) -> Tensor:
        """Return Phi of shape [B], retaining all ordinary autograd dependencies."""
        time, labels = self._inputs(z, t, y)
        angles = time[:, None] * self.time_frequencies[None, :]
        time_embedding = torch.cat((angles.cos(), angles.sin()), dim=-1)
        condition = self.class_embedding(labels) + self.time_mlp(time_embedding)
        hidden = self.activation(self.input_conv(z))
        hidden = hidden + condition[:, :, None, None] + self.position
        for layer in self.hidden_convs:
            hidden = self.activation(layer(hidden))
        return self.readout(hidden).flatten(1).sum(dim=1)

    def clean_correction(
        self, z: Tensor, t: Tensor | float, y: Tensor, create_graph: bool = False,
    ) -> Tensor:
        """Return c=grad_z Phi, shape equal to z, without accumulating .grad.

        If z already requires gradients, its existing graph is preserved.
        Otherwise a detached local leaf is made; the caller's z is not mutated.
        create_graph=True retains the graph through the input derivative, so
        training the loss can differentiate with respect to potential parameters
        (and an existing upstream z graph). There is no zero-readout shortcut.

        create_graph=False returns a detached correction for inference. Internal
        enable_grad supports an outer torch.no_grad block. torch.inference_mode
        and inference tensors are deliberately unsupported: enable_grad alone
        cannot restore their autograd semantics. Use no_grad for the backbone
        and ordinary tensors when calling this method.
        """
        if torch.is_inference_mode_enabled() or (isinstance(z, Tensor) and torch.is_inference(z)):
            raise RuntimeError("clean_correction supports no_grad, not inference_mode or inference tensors")
        if not isinstance(z, Tensor) or not z.is_floating_point():
            raise TypeError("z must be a floating-point tensor")
        with torch.enable_grad():
            local_z = z if z.requires_grad else z.detach().requires_grad_(True)
            scalar = self(local_z, t, y)
            correction, = torch.autograd.grad(
                scalar.sum(), local_z, create_graph=create_graph,
            )
        return correction if create_graph else correction.detach()


def observable_error_loss(
    clean_correction: Tensor,
    clean_target: Tensor,
    official_clean: Tensor,
    *,
    sample_weights: Tensor | None = None,
) -> Tensor:
    """Mean clean-potential residual objective, with optional importance weights.

    Each image contributes mean_dimensions(.5*c**2 - (X-G_official)*c).
    This is averaged across images, optionally after multiplying each image by
    its sample weight. Weights are NOT normalized by their sum: the runner owns
    time sampling and importance factors. In particular this helper introduces
    no hidden t**-2 weighting, time schedule, or guidance coefficient.

    X, G_official and weights are detached targets. Gradients propagate only
    through c, which must be computed with create_graph=True during training.
    Half/bfloat inputs accumulate in float32; any float64 input uses float64.
    """
    values = (clean_correction, clean_target, official_clean)
    if any(not isinstance(value, Tensor) or not value.is_floating_point() for value in values):
        raise TypeError("correction, clean target and official prediction must be floating tensors")
    if clean_correction.ndim < 2 or any(size == 0 for size in clean_correction.shape):
        raise ValueError("correction must have nonempty [B,...] dimensions")
    if any(value.shape != clean_correction.shape or value.device != clean_correction.device
           for value in values[1:]):
        raise ValueError("correction, clean target and official prediction must share shape and device")
    dtype = torch.float64 if any(value.dtype == torch.float64 for value in values) else torch.float32
    correction = clean_correction.to(dtype=dtype)
    residual = clean_target.detach().to(dtype=dtype) - official_clean.detach().to(dtype=dtype)
    per_image = (.5 * correction.square() - residual * correction).flatten(1).mean(dim=1)
    if sample_weights is not None:
        if not isinstance(sample_weights, Tensor) or not sample_weights.is_floating_point():
            raise TypeError("sample_weights must be a floating-point tensor")
        if sample_weights.shape != (len(correction),) or sample_weights.device != correction.device:
            raise ValueError("sample_weights must have shape [B] and share correction's device")
        if (not bool(torch.isfinite(sample_weights).all())
                or bool((sample_weights < 0).any())):
            raise ValueError("sample_weights must be finite and nonnegative")
        per_image = per_image * sample_weights.detach().to(dtype=dtype)
    return per_image.mean()
