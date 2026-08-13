"""Time-sampling distributions for the ImageNet-100 SiT target study."""

from __future__ import annotations

import math

import torch


TIME_SAMPLERS = ("uniform", "logit_normal")


def validate_time_sampling(
    time_sampler: str,
    logit_mean: float,
    logit_std: float,
) -> None:
    if time_sampler not in TIME_SAMPLERS:
        raise ValueError(f"unsupported time sampler: {time_sampler!r}")
    if not math.isfinite(logit_mean):
        raise ValueError("logit-normal mean must be finite")
    if not math.isfinite(logit_std) or logit_std <= 0:
        raise ValueError("logit-normal standard deviation must be finite and positive")


def sample_time_values(
    count: int,
    *,
    device: torch.device | str,
    time_sampler: str,
    logit_mean: float = -0.8,
    logit_std: float = 0.8,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample path coordinates, matching JiT's logit-normal formula exactly."""

    if count < 1:
        raise ValueError("time sample count must be positive")
    validate_time_sampling(time_sampler, logit_mean, logit_std)
    if time_sampler == "uniform":
        return torch.rand(count, device=device, generator=generator)
    logits = torch.randn(count, device=device, generator=generator)
    return torch.sigmoid(logits.mul(float(logit_std)).add(float(logit_mean)))


def time_distribution_metadata(
    time_sampler: str,
    logit_mean: float,
    logit_std: float,
) -> dict[str, float | str]:
    validate_time_sampling(time_sampler, logit_mean, logit_std)
    if time_sampler == "uniform":
        return {"name": "uniform", "interval": "[0,1)"}
    return {
        "name": "logit_normal",
        "logit_mean": float(logit_mean),
        "logit_std": float(logit_std),
        "formula": "t=sigmoid(N(logit_mean,logit_std^2))",
    }
