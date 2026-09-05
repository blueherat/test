"""OU-coordinate value functions for semigroup-consistent RAEv2 guidance.

RAEv2 follows ``z_t=(1-t)x+t*eps``.  Dividing by the total standard
deviation maps this path to the variance-preserving OU semigroup

``y_s = alpha_s*x + sqrt(1-alpha_s**2)*eps``.

The module keeps the coordinate algebra and the dimension-normalized HJB
backup independent from any RAE model-loading code.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _batch_scale(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if value.ndim != 1 or value.shape[0] != target.shape[0]:
        raise ValueError("coefficient must have one value per sample")
    while value.ndim < target.ndim:
        value = value.unsqueeze(-1)
    return value


def rae_ou_coefficients(
    noise_time: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total scale, signal, noise, and OU semigroup time."""

    if noise_time.ndim != 1 or torch.any((noise_time < 0.0) | (noise_time >= 1.0)):
        raise ValueError("noise time must have shape [batch] and lie in [0,1)")
    data = 1.0 - noise_time
    noise = noise_time
    total = torch.sqrt(data.square() + noise.square())
    signal = data / total
    noise_coefficient = noise / total
    semigroup_time = -torch.log(signal)
    return total, signal, noise_coefficient, semigroup_time


def noise_time_from_ou_time(semigroup_time: torch.Tensor) -> torch.Tensor:
    """Invert ``s=-log((1-t)/sqrt((1-t)^2+t^2))``."""

    if torch.any(semigroup_time < 0.0):
        raise ValueError("OU semigroup time must be non-negative")
    signal = torch.exp(-semigroup_time)
    odds = torch.sqrt(torch.expm1(2.0 * semigroup_time).clamp_min(0.0))
    noise_time = odds / (1.0 + odds)
    # The expression through ``odds`` is stable near the data endpoint while
    # ``signal`` documents and checks the intended OU coordinate.
    if not torch.allclose(
        signal.square() + (noise_time / (1.0 - noise_time)).square()
        * signal.square(),
        torch.ones_like(signal),
        atol=2e-5,
        rtol=2e-5,
    ):
        raise FloatingPointError("OU inverse lost its normalization identity")
    return noise_time


def state_to_ou(state: torch.Tensor, noise_time: torch.Tensor) -> torch.Tensor:
    total, _, _, _ = rae_ou_coefficients(noise_time)
    return state / _batch_scale(total, state)


def ou_to_state(ou_state: torch.Tensor, noise_time: torch.Tensor) -> torch.Tensor:
    total, _, _, _ = rae_ou_coefficients(noise_time)
    return ou_state * _batch_scale(total, ou_state)


def clean_prediction_to_ou_score(
    clean: torch.Tensor,
    *,
    ou_state: torch.Tensor,
    noise_time: torch.Tensor,
) -> torch.Tensor:
    """Map a clean conditional mean to the corresponding OU score."""

    if clean.shape != ou_state.shape:
        raise ValueError("clean prediction and OU state must have identical shapes")
    _, signal, noise, _ = rae_ou_coefficients(noise_time)
    if torch.any(noise <= 0.0):
        raise ValueError("OU score is undefined at the data endpoint")
    return (
        _batch_scale(signal, clean) * clean - ou_state
    ) / _batch_scale(noise.square(), clean)


def clean_gap_to_ou_score_gap(
    clean_gap: torch.Tensor,
    *,
    noise_time: torch.Tensor,
) -> torch.Tensor:
    """Map ``x_full-x_base`` to ``score_full-score_base``."""

    _, signal, noise, _ = rae_ou_coefficients(noise_time)
    if torch.any(noise <= 0.0):
        raise ValueError("OU score gap is undefined at the data endpoint")
    return _batch_scale(signal / noise.square(), clean_gap) * clean_gap


def ou_potential_gradient_to_clean_correction(
    potential_gradient: torch.Tensor,
    *,
    noise_time: torch.Tensor,
) -> torch.Tensor:
    """Convert ``grad_y delta`` into the corresponding clean prediction shift."""

    _, signal, noise, _ = rae_ou_coefficients(noise_time)
    if torch.any(signal <= 0.0):
        raise ValueError("clean correction is undefined at the pure-noise endpoint")
    return _batch_scale(noise.square() / signal, potential_gradient) * potential_gradient


def semigroup_value_guided_clean(
    full_clean: torch.Tensor,
    base_clean: torch.Tensor,
    normalized_value_gradient: torch.Tensor,
    *,
    noise_time: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the uniquely scaled HJB correction in native clean space.

    The learned value is ``phi=delta/D``.  Consequently the score correction
    is exactly ``grad(delta)=D*grad(phi)``; there is no inference gain.
    """

    if not (
        full_clean.shape
        == base_clean.shape
        == normalized_value_gradient.shape
    ):
        raise ValueError("clean predictions and value gradient must match")
    if beta < 1.0 or not math.isfinite(beta):
        raise ValueError("beta must be finite and at least one")
    dimension = full_clean[0].numel()
    correction = ou_potential_gradient_to_clean_correction(
        float(dimension) * normalized_value_gradient,
        noise_time=noise_time,
    )
    ordinary = base_clean + beta * (full_clean - base_clean)
    return ordinary + correction, correction


def ou_relative_retention(
    noise_time: torch.Tensor,
    *,
    switch_time: float,
) -> torch.Tensor:
    """Return ``exp(-(s-s_switch))`` on the noise side of the switch."""

    if not 0.0 < switch_time < 1.0:
        raise ValueError("switch time must lie in (0,1)")
    _, signal, _, _ = rae_ou_coefficients(noise_time)
    switch = torch.tensor(
        [switch_time], device=noise_time.device, dtype=noise_time.dtype
    )
    _, switch_signal, _, _ = rae_ou_coefficients(switch)
    retention = signal / switch_signal[0]
    if torch.any((retention < 0.0) | (retention > 1.0 + 1e-5)):
        raise ValueError("noise time lies on the data side of the switch")
    return retention.clamp(0.0, 1.0)


def normalized_hjb_running_cost(
    score_gap: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    """Return ``c/D`` while preserving the full HJB after rescaling value."""

    if score_gap.ndim < 2:
        raise ValueError("score gap must include batch and feature dimensions")
    if beta < 1.0 or not math.isfinite(beta):
        raise ValueError("beta must be finite and at least one")
    return beta * (beta - 1.0) * score_gap.float().flatten(1).square().mean(1)


def normalized_hjb_target(
    particle_values: torch.Tensor,
    value_gradient: torch.Tensor,
    *,
    running_cost_per_dimension: torch.Tensor,
    semigroup_step: torch.Tensor,
    ambient_dimension: int,
) -> torch.Tensor:
    """First-order HJB target for ``phi=delta/D``.

    ``particle_values`` estimates the diffusion and drift action on the value.
    The explicit quadratic term retains the large-deviation correction that a
    finite-particle log-mean-exp would miss after its weights collapse.
    """

    if particle_values.ndim != 2:
        raise ValueError("particle values must have shape [particles, batch]")
    if value_gradient.shape[0] != particle_values.shape[1]:
        raise ValueError("value gradient batch does not match particle values")
    expected = particle_values.shape[1:]
    if (
        running_cost_per_dimension.shape != expected
        or semigroup_step.shape != expected
    ):
        raise ValueError("running cost and semigroup step must have shape [batch]")
    if ambient_dimension <= 0 or torch.any(semigroup_step <= 0.0):
        raise ValueError("dimension and semigroup steps must be positive")
    gradient_energy = value_gradient.float().flatten(1).square().sum(1)
    return particle_values.float().mean(0) + semigroup_step * (
        float(ambient_dimension) * gradient_energy + running_cost_per_dimension
    )


def sinusoidal_embedding(values: torch.Tensor, dimension: int) -> torch.Tensor:
    if values.ndim != 1 or dimension < 4 or dimension % 2:
        raise ValueError("sinusoidal embedding requires even dimension >= 4")
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(
            dimension // 2, device=values.device, dtype=values.dtype
        )
        / (dimension // 2 - 1)
    )
    phases = values[:, None] * frequencies[None]
    return torch.cat((torch.cos(phases), torch.sin(phases)), dim=1)


class FiLMResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        groups = min(32, width)
        while width % groups:
            groups -= 1
        self.norm1 = nn.GroupNorm(groups, width, affine=False)
        self.norm2 = nn.GroupNorm(groups, width, affine=False)
        self.conv1 = nn.Conv2d(width, width, 3, padding=1)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)
        self.modulation = nn.Linear(width, 2 * width)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, features: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        shift, scale = self.modulation(conditioning).chunk(2, dim=1)
        hidden = self.norm1(features)
        hidden = hidden * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        hidden = self.conv1(torch.nn.functional.silu(hidden))
        hidden = self.conv2(torch.nn.functional.silu(self.norm2(hidden)))
        return features + hidden


class RAEv2NormalizedOUValue(nn.Module):
    """Small amortized scalar free-energy head on raw normalized RAE latents."""

    def __init__(
        self,
        latent_channels: int,
        num_classes: int,
        *,
        width: int = 64,
        depth: int = 3,
        switch_time: float = 0.5,
    ) -> None:
        super().__init__()
        if latent_channels <= 0 or num_classes <= 0 or width < 8 or depth <= 0:
            raise ValueError("invalid value-network dimensions")
        if not 0.0 < switch_time < 1.0:
            raise ValueError("switch time must lie in (0,1)")
        self.latent_channels = int(latent_channels)
        self.num_classes = int(num_classes)
        self.width = int(width)
        self.depth = int(depth)
        self.switch_time = float(switch_time)
        self.stem = nn.Conv2d(latent_channels, width, 1)
        self.class_embedding = nn.Embedding(num_classes, width)
        self.time_mlp = nn.Sequential(
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.blocks = nn.ModuleList(FiLMResidualBlock(width) for _ in range(depth))
        groups = min(32, width)
        while width % groups:
            groups -= 1
        self.output_norm = nn.GroupNorm(groups, width, affine=False)
        self.spatial_output = nn.Conv2d(width, 1, 1)
        self.baseline_output = nn.Sequential(
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )
        nn.init.zeros_(self.spatial_output.weight)
        nn.init.zeros_(self.spatial_output.bias)
        nn.init.zeros_(self.baseline_output[-1].weight)
        nn.init.zeros_(self.baseline_output[-1].bias)

    def forward(
        self,
        ou_state: torch.Tensor,
        noise_time: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if ou_state.ndim != 4 or ou_state.shape[1] != self.latent_channels:
            raise ValueError("OU state must be BCHW with the configured channels")
        if noise_time.shape != (len(ou_state),) or labels.shape != (len(ou_state),):
            raise ValueError("time and labels must have shape [batch]")
        if torch.any((labels < 0) | (labels >= self.num_classes)):
            raise ValueError("class label is out of range")
        _, _, _, semigroup_time = rae_ou_coefficients(noise_time)
        time_embedding = sinusoidal_embedding(semigroup_time, self.width)
        conditioning = self.class_embedding(labels) + self.time_mlp(time_embedding)
        features = self.stem(ou_state)
        for block in self.blocks:
            features = block(features, conditioning)
        spatial = self.spatial_output(
            torch.nn.functional.silu(self.output_norm(features))
        ).mean(dim=(1, 2, 3))
        baseline = self.baseline_output(conditioning).squeeze(1)
        retention = ou_relative_retention(
            noise_time, switch_time=self.switch_time
        )
        return (
            (1.0 - retention) * baseline
            + retention * (1.0 - retention) * spatial
        )
