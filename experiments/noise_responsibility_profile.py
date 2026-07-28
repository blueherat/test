"""Paired metrics for noise-resolved latent responsibility diagnostics.

The module is model-agnostic. Callers must evaluate all condition branches on
the same noisy state, timestep, prediction target, and non-latent conditions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
import torch


BRANCHES = ("real", "null", "shuffle")


@dataclass(frozen=True)
class ResponsibilityBatch:
    timestep: torch.Tensor
    target: torch.Tensor
    predictions: Mapping[str, torch.Tensor]
    sample_index: torch.Tensor | None = None

    def validate(self) -> None:
        if self.target.ndim < 2:
            raise ValueError("target must have a batch dimension and feature dimensions")
        batch_size = self.target.shape[0]
        if self.timestep.ndim != 1 or self.timestep.shape[0] != batch_size:
            raise ValueError("timestep must be a [B] tensor")
        missing = set(BRANCHES) - set(self.predictions)
        if missing:
            raise ValueError(f"missing prediction branches: {sorted(missing)}")
        for name in BRANCHES:
            if self.predictions[name].shape != self.target.shape:
                raise ValueError(
                    f"{name} prediction shape {tuple(self.predictions[name].shape)} "
                    f"does not match target {tuple(self.target.shape)}"
                )
        if self.sample_index is not None and (
            self.sample_index.ndim != 1 or self.sample_index.shape[0] != batch_size
        ):
            raise ValueError("sample_index must be a [B] tensor")


def derangement(size: int, *, seed: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Return a deterministic permutation with no fixed points."""

    size = int(size)
    if size < 2:
        raise ValueError("a shuffled-condition batch needs at least two samples")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    base = torch.arange(size)
    for _ in range(1024):
        permutation = torch.randperm(size, generator=generator)
        if bool(torch.all(permutation != base)):
            return permutation.to(device=device)
    # Deterministic fallback cannot contain a fixed point for size >= 2.
    return torch.roll(base, shifts=1).to(device=device)


def per_sample_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    return (prediction.float() - target.float()).flatten(1).square().mean(dim=1)


def _as_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy()


def responsibility_rows(batch: ResponsibilityBatch) -> pd.DataFrame:
    """Return one paired row per sample without aggregating away sign reversals."""

    batch.validate()
    losses = {
        name: per_sample_mse(batch.predictions[name], batch.target)
        for name in BRANCHES
    }
    null_scale = losses["null"].clamp_min(torch.finfo(torch.float32).eps)
    shuffle_scale = losses["shuffle"].clamp_min(torch.finfo(torch.float32).eps)
    sample_index = (
        torch.arange(batch.target.shape[0], device=batch.target.device)
        if batch.sample_index is None
        else batch.sample_index
    )
    return pd.DataFrame(
        {
            "sample_index": _as_numpy(sample_index).astype(np.int64),
            "timestep": _as_numpy(batch.timestep),
            "loss_real": _as_numpy(losses["real"]),
            "loss_null": _as_numpy(losses["null"]),
            "loss_shuffle": _as_numpy(losses["shuffle"]),
            "delta_null": _as_numpy(losses["null"] - losses["real"]),
            "delta_shuffle": _as_numpy(losses["shuffle"] - losses["real"]),
            "gain_null": _as_numpy((losses["null"] - losses["real"]) / null_scale),
            "gain_shuffle": _as_numpy(
                (losses["shuffle"] - losses["real"]) / shuffle_scale
            ),
        }
    )


def aggregate_profile(
    rows: pd.DataFrame,
    *,
    timestep_decimals: int = 6,
) -> pd.DataFrame:
    """Aggregate paired rows at each supported timestep."""

    required = {
        "timestep",
        "loss_real",
        "loss_null",
        "loss_shuffle",
        "delta_null",
        "delta_shuffle",
        "gain_null",
        "gain_shuffle",
    }
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    table = rows.copy()
    table["timestep"] = table["timestep"].round(int(timestep_decimals))
    metrics = sorted(required - {"timestep"})
    grouped = table.groupby("timestep", sort=True, dropna=False)
    records: list[dict[str, float | int]] = []
    for timestep, frame in grouped:
        record: dict[str, float | int] = {
            "timestep": float(timestep),
            "count": int(len(frame)),
        }
        for metric in metrics:
            values = frame[metric].to_numpy(dtype=np.float64)
            record[f"{metric}_mean"] = float(np.mean(values))
            record[f"{metric}_median"] = float(np.median(values))
            record[f"{metric}_positive_rate"] = float(np.mean(values > 0.0))
        records.append(record)
    return pd.DataFrame.from_records(records)


def identity_control_error(first: torch.Tensor, second: torch.Tensor) -> dict[str, float]:
    """Numerical guardrail for repeated forwards with identical inputs."""

    if first.shape != second.shape:
        raise ValueError("identity-control tensors must have identical shapes")
    difference = (first.float() - second.float()).flatten(1)
    reference = first.float().flatten(1)
    rms = torch.sqrt(difference.square().mean(dim=1))
    reference_rms = torch.sqrt(reference.square().mean(dim=1)).clamp_min(
        torch.finfo(torch.float32).eps
    )
    return {
        "absolute_rms_max": float(rms.max()),
        "relative_rms_max": float((rms / reference_rms).max()),
    }


def radial_frequency_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    boundaries: tuple[float, ...] = (0.25, 0.5),
) -> torch.Tensor:
    """Per-sample Fourier MSE in normalized radial-frequency bands.

    Returns `[B, len(boundaries) + 1]`. The value is normalized by the number
    of selected Fourier coefficients, so bands are comparable in scale.
    """

    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("frequency MSE expects equal BCHW prediction and target")
    if tuple(sorted(boundaries)) != tuple(boundaries):
        raise ValueError("boundaries must be sorted")
    if any(value <= 0.0 or value >= 1.0 for value in boundaries):
        raise ValueError("boundaries must lie strictly between zero and one")
    error = prediction.float() - target.float()
    spectrum = torch.fft.fft2(error, norm="ortho").abs().square()
    height, width = error.shape[-2:]
    fy = torch.fft.fftfreq(height, device=error.device)
    fx = torch.fft.fftfreq(width, device=error.device)
    radius = torch.sqrt(fy[:, None].square() + fx[None, :].square())
    radius = radius / radius.max().clamp_min(torch.finfo(torch.float32).eps)
    edges = (0.0, *boundaries, 1.0 + 1e-6)
    values = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (radius >= lower) & (radius < upper)
        values.append(spectrum[..., mask].mean(dim=(1, 2)))
    return torch.stack(values, dim=1)
