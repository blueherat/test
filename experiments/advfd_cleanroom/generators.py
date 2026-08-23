"""Generator adapters used by the paper-only AdvFD reproduction."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def import_pmf(repo: Path) -> Any:
    repo = Path(repo).expanduser().resolve()
    if not (repo / "pmf.py").is_file():
        raise FileNotFoundError(f"pMF repository not found: {repo}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return importlib.import_module("pmf")


def checkpoint_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    """Extract a model state dictionary without silently choosing EMA weights."""

    if isinstance(payload, dict) and payload and all(
        isinstance(key, str) and torch.is_tensor(value)
        for key, value in payload.items()
    ):
        return payload
    if isinstance(payload, dict):
        candidates = [
            key
            for key in ("model", "state_dict", "online_model")
            if isinstance(payload.get(key), dict)
        ]
        if len(candidates) == 1:
            return payload[candidates[0]]
        if candidates:
            raise ValueError(
                "Checkpoint contains multiple online-looking state dictionaries: "
                f"{candidates}"
            )
        if any(key in payload for key in ("ema", "model_ema", "ema_model")):
            raise ValueError(
                "Checkpoint exposes only/ambiguously EMA weights; select the online "
                "state explicitly before loading"
            )
    raise TypeError("Could not identify a plain online model state dictionary")


def load_pmf_b16(
    *, repo: Path, checkpoint: Path, device: torch.device
) -> nn.Module:
    pmf = import_pmf(repo)
    model = pmf.pixelMeanFlow("pmfDiT_B_16", img_size=256)
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    state = checkpoint_state_dict(payload)
    incompatible = model.load_state_dict(state, strict=False)
    allowed_missing = {"net.rope_freqs"}
    disallowed_missing = set(incompatible.missing_keys) - allowed_missing
    if disallowed_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "pMF checkpoint is not architecture-exact: "
            f"missing={sorted(disallowed_missing)[:20]}, "
            f"unexpected={incompatible.unexpected_keys[:20]}"
        )
    return model.to(device)


def pmf_one_step(
    model: nn.Module,
    noise: torch.Tensor,
    labels: torch.Tensor,
    *,
    cfg_omega: float = 8.5,
    interval_min: float = 0.1,
    interval_max: float = 0.7,
) -> torch.Tensor:
    """Differentiable one-step pMF-B generation using the paper configuration."""

    batch = noise.shape[0]
    dtype = noise.dtype
    device = noise.device
    t = torch.ones(batch, dtype=dtype, device=device)
    h = torch.ones(batch, dtype=dtype, device=device)
    omega = torch.full((batch,), cfg_omega, dtype=dtype, device=device)
    t_min = torch.full((batch,), interval_min, dtype=dtype, device=device)
    t_max = torch.full((batch,), interval_max, dtype=dtype, device=device)
    predicted_average_velocity = model.u_fn(
        noise, t, h, omega, t_min, t_max, labels
    )[0]
    return noise - predicted_average_velocity


def seeded_pmf_noise(
    model: nn.Module,
    *,
    batch_size: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(
        batch_size,
        model.img_channels,
        model.img_size,
        model.img_size,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    return noise * float(model.noise_scale)
