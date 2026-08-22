"""Learnable operator-valued x/velocity prediction parameterization.

For the SiT path ``z_t = (1-t) * noise + t * data``, define a native target

    y = P v + (I - P) x,

where ``v = data - noise`` and ``0 <= P <= I``.  In an eigenbasis of P, a
network output y is converted back to velocity exactly as

    v = (y - (1-p) z_t) / (1-t+t*p).

The selector below is full-rank in latent space.  It uses an orthonormal 2-D
DCT basis and a learned orthonormal channel basis, with smooth time-dependent
eigenvalues.  It therefore permits dense spatial/channel mixing while keeping
the inverse analytic and numerically auditable.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def orthonormal_dct_matrix(side: int, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if side < 1:
        raise ValueError("side must be positive")
    positions = torch.arange(side, dtype=dtype) + 0.5
    frequencies = torch.arange(side, dtype=dtype)[:, None]
    matrix = torch.cos(math.pi * frequencies * positions[None, :] / side)
    matrix[0] *= math.sqrt(1.0 / side)
    if side > 1:
        matrix[1:] *= math.sqrt(2.0 / side)
    return matrix


def bernstein_basis(time_value: torch.Tensor, terms: int) -> torch.Tensor:
    """Return nonnegative degree-(terms-1) Bernstein weights summing to one."""
    if time_value.ndim != 1:
        raise ValueError("time_value must have shape [B]")
    if terms < 1:
        raise ValueError("terms must be positive")
    degree = terms - 1
    values = []
    for index in range(terms):
        coefficient = math.comb(degree, index)
        values.append(
            coefficient
            * time_value.pow(index)
            * (1.0 - time_value).pow(degree - index)
        )
    return torch.stack(values, dim=1)


class SpectralTargetSelector(nn.Module):
    """A symmetric full-rank soft selector P with an analytic inverse."""

    def __init__(
        self,
        *,
        channels: int,
        side: int,
        time_terms: int = 8,
        initial_x_fraction: float = 1e-3,
        maximum_x_fraction: float = 0.999,
    ) -> None:
        super().__init__()
        if channels < 1 or side < 1 or time_terms < 1:
            raise ValueError("channels, side, and time_terms must be positive")
        if not 0.0 <= initial_x_fraction < maximum_x_fraction < 1.0:
            raise ValueError(
                "require 0 <= initial_x_fraction < maximum_x_fraction < 1"
            )
        self.channels = int(channels)
        self.side = int(side)
        self.time_terms = int(time_terms)
        self.maximum_x_fraction = float(maximum_x_fraction)
        initial_raw = math.atanh(
            float(initial_x_fraction) / float(maximum_x_fraction)
        )
        self.gate_raw = nn.Parameter(
            torch.full(
                (self.time_terms, self.channels, self.side, self.side),
                initial_raw,
            )
        )
        self.channel_raw = nn.Parameter(torch.eye(self.channels))
        self.register_buffer(
            "dct",
            orthonormal_dct_matrix(self.side),
            persistent=True,
        )

    def channel_basis(self) -> torch.Tensor:
        basis, _upper = torch.linalg.qr(self.channel_raw.float())
        return basis

    def to_spectral(self, value: torch.Tensor) -> torch.Tensor:
        self._validate_value(value)
        dct = self.dct.float()
        spatial = torch.einsum("kh,bchw,lw->bckl", dct, value.float(), dct)
        basis = self.channel_basis()
        return torch.einsum("dc,bdhw->bchw", basis, spatial)

    def from_spectral(self, value: torch.Tensor) -> torch.Tensor:
        self._validate_value(value)
        basis = self.channel_basis()
        spatial = torch.einsum("dc,bchw->bdhw", basis, value.float())
        dct = self.dct.float()
        return torch.einsum("kh,bckl,lw->bchw", dct, spatial, dct)

    def eigenvalues(self, time_value: torch.Tensor) -> torch.Tensor:
        weights = bernstein_basis(time_value.float(), self.time_terms)
        raw = torch.einsum("bk,kchw->bchw", weights, self.gate_raw.float())
        x_fraction = self.maximum_x_fraction * torch.tanh(raw.clamp_min(0.0))
        return 1.0 - x_fraction

    def apply(
        self,
        value: torch.Tensor,
        *,
        eigenvalues: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_pair(value, eigenvalues)
        return self.from_spectral(self.to_spectral(value) * eigenvalues.float())

    def native_target(
        self,
        *,
        data: torch.Tensor,
        velocity: torch.Tensor,
        eigenvalues: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_pair(data, eigenvalues)
        self._validate_value(velocity)
        data_spectral = self.to_spectral(data)
        velocity_spectral = self.to_spectral(velocity)
        target_spectral = (
            eigenvalues.float() * velocity_spectral
            + (1.0 - eigenvalues.float()) * data_spectral
        )
        return self.from_spectral(target_spectral)

    def output_to_velocity(
        self,
        output: torch.Tensor,
        *,
        state: torch.Tensor,
        time_value: torch.Tensor,
        eigenvalues: torch.Tensor,
        denominator_floor: float,
    ) -> torch.Tensor:
        self._validate_pair(output, eigenvalues)
        self._validate_value(state)
        if time_value.shape != (len(output),):
            raise ValueError("time_value must have shape [B]")
        if not 0.0 < denominator_floor < 1.0:
            raise ValueError("denominator_floor must lie in (0,1)")
        output_spectral = self.to_spectral(output)
        state_spectral = self.to_spectral(state)
        time = time_value.float().reshape(-1, 1, 1, 1)
        eigenvalues = eigenvalues.float()
        numerator = output_spectral - (1.0 - eigenvalues) * state_spectral
        denominator = (1.0 - time + time * eigenvalues).clamp_min(
            float(denominator_floor)
        )
        return self.from_spectral(numerator / denominator)

    @torch.no_grad()
    def project_parameters_(self) -> None:
        """Projected optimization keeps every selector eigenvalue in [0,1]."""
        self.gate_raw.clamp_(min=0.0, max=8.0)

    def _validate_value(self, value: torch.Tensor) -> None:
        expected = (self.channels, self.side, self.side)
        if value.ndim != 4 or tuple(value.shape[1:]) != expected:
            raise ValueError(f"expected value [B,{expected}], found {tuple(value.shape)}")

    def _validate_pair(
        self,
        value: torch.Tensor,
        eigenvalues: torch.Tensor,
    ) -> None:
        self._validate_value(value)
        if eigenvalues.shape != value.shape:
            raise ValueError("eigenvalues and value must have identical shapes")
