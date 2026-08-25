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


def pmf_state_dict_for_advfd(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert an upstream pMF state dict to the public AdvFD model layout.

    The two repositories implement the same pMF network but expose wrapper
    parameters under different names and store type tokens with different
    singleton dimensions. RoPE frequencies are deterministic buffers in the
    upstream model and are recomputed by AdvFD.
    """

    converted: dict[str, torch.Tensor] = {}
    for source_key, source_value in state_dict.items():
        if "rope_freqs" in source_key:
            continue
        target_key = source_key.replace("._flax_linear.", ".linear.")
        target_key = target_key.replace("._flax_embedding.", ".embedding.")
        target_value = source_value
        if (
            target_key.endswith("_tokens")
            and target_value.ndim == 3
            and target_value.shape[0] == 1
        ):
            target_value = target_value.squeeze(0)
        if target_key in converted:
            raise ValueError(f"duplicate converted pMF key: {target_key}")
        converted[target_key] = target_value
    return converted


def pmf_state_dict_from_advfd(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert the public AdvFD pMF layout back to the upstream pMF layout."""

    converted: dict[str, torch.Tensor] = {}
    for source_key, source_value in state_dict.items():
        target_key = source_key
        if ".linear." in target_key:
            prefix, suffix = target_key.rsplit(".linear.", maxsplit=1)
            target_key = f"{prefix}._flax_linear.{suffix}"
        if ".embedding." in target_key:
            prefix, suffix = target_key.rsplit(".embedding.", maxsplit=1)
            target_key = f"{prefix}._flax_embedding.{suffix}"
        target_value = source_value
        if target_key.endswith("_tokens") and target_value.ndim == 2:
            target_value = target_value.unsqueeze(0)
        if target_key in converted:
            raise ValueError(f"duplicate converted pMF key: {target_key}")
        converted[target_key] = target_value
    return converted


def load_pmf_b16(
    *, repo: Path, checkpoint: Path, device: torch.device
) -> nn.Module:
    pmf = import_pmf(repo)
    model = pmf.pixelMeanFlow("pmfDiT_B_16", img_size=256)
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    state = checkpoint_state_dict(payload)
    if any(".linear." in key or ".embedding." in key for key in state):
        state = pmf_state_dict_from_advfd(state)
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
