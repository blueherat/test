from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]


def _load_adapter_class():
    from experiments.latent_equiv_adapter import InvertibleLatentAdapter

    return InvertibleLatentAdapter


def _load_adapter_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    channels: Optional[int] = None,
    hidden_channels: Optional[int] = None,
    blocks: Optional[int] = None,
) -> nn.Module:
    path = Path(checkpoint_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu")
    cfg = dict(checkpoint.get("config", {}))
    state_dict = checkpoint.get("state_dict", checkpoint)
    inferred = _infer_adapter_arch(state_dict)
    channels = int(channels if channels is not None else checkpoint.get("channels", cfg.get("channels", inferred[0])))
    hidden_channels = int(
        hidden_channels
        if hidden_channels is not None
        else checkpoint.get("hidden_channels", cfg.get("hidden_channels", inferred[1]))
    )
    blocks = int(blocks if blocks is not None else checkpoint.get("blocks", cfg.get("blocks", inferred[2])))
    adapter = _load_adapter_class()(channels=channels, hidden_channels=hidden_channels, blocks=blocks)
    adapter.load_state_dict(state_dict, strict=True)
    return adapter


def _infer_adapter_arch(state_dict: dict[str, torch.Tensor]) -> tuple[int, int, int]:
    first_weight = state_dict.get("blocks.0.net.net.0.weight")
    if first_weight is None:
        return 768, 128, 4
    hidden_channels = int(first_weight.shape[0])
    channels = int(first_weight.shape[1]) * 2
    block_ids = []
    for key in state_dict:
        parts = key.split(".")
        if len(parts) > 1 and parts[0] == "blocks" and parts[1].isdigit():
            block_ids.append(int(parts[1]))
    blocks = max(block_ids) + 1 if block_ids else 4
    return channels, hidden_channels, blocks


def _resolve_base_rae_params(params: dict, rae_repo_path: str | Path) -> dict:
    resolved = dict(params)
    repo_path = Path(rae_repo_path).expanduser()
    if not repo_path.is_absolute():
        repo_path = ROOT / repo_path
    for name in ("decoder_config_path", "pretrained_decoder_path", "normalization_stat_path"):
        value = resolved.get(name)
        if value is None:
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            resolved[name] = str(repo_path / path)
    return resolved


class DecoderAdaptedRAE(nn.Module):
    """RAE with separate encoder-side and decoder-side latent adapters.

    The public latent is y = A(z).  Decoding uses G^{-1}(y) before the frozen
    RAE decoder.  G is initialized from A, but can be fine-tuned independently
    as a decoder-side inverse adapter.
    """

    def __init__(
        self,
        adapter_checkpoint_path: str,
        decoder_adapter_checkpoint_path: Optional[str] = None,
        base_rae_params: Optional[dict] = None,
        rae_repo_path: str = "external/RAE",
        channels: Optional[int] = None,
        hidden_channels: Optional[int] = None,
        blocks: Optional[int] = None,
        freeze_encoder_adapter: bool = True,
        freeze_decoder_adapter: bool = True,
    ):
        super().__init__()
        from stage1 import RAE

        base_rae_params = {} if base_rae_params is None else dict(base_rae_params)
        base_rae_params = _resolve_base_rae_params(base_rae_params, rae_repo_path)
        self.base_rae = RAE(**base_rae_params)
        self.encoder_adapter = _load_adapter_from_checkpoint(
            adapter_checkpoint_path,
            channels=channels,
            hidden_channels=hidden_channels,
            blocks=blocks,
        )
        decoder_path = decoder_adapter_checkpoint_path or adapter_checkpoint_path
        self.decoder_adapter = _load_adapter_from_checkpoint(
            decoder_path,
            channels=channels,
            hidden_channels=hidden_channels,
            blocks=blocks,
        )

        if freeze_encoder_adapter:
            self.encoder_adapter.requires_grad_(False)
        if freeze_decoder_adapter:
            self.decoder_adapter.requires_grad_(False)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.base_rae.encode(x)
        return self.encoder_adapter(z)

    def decode(self, y: torch.Tensor) -> torch.Tensor:
        z = self.decoder_adapter.inverse(y)
        return self.base_rae.decode(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.encode(x)
        return self.decode(y)
