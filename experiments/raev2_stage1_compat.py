"""Compatibility shim for RAEv2 decoder configs on newer Transformers."""

from __future__ import annotations

import json
from math import sqrt
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig


PLACEHOLDER = "SHOULD BE RELOADED"


def resolve_decoder_config(
    config_path: str | Path,
    *,
    hidden_size: int,
    patch_size: int,
    num_patches: int,
) -> Any | None:
    """Resolve RAEv2's runtime patch-size placeholder before validation."""

    config_file = Path(config_path) / "config.json"
    if not config_file.is_file():
        return None
    config_dict = json.loads(config_file.read_text(encoding="utf-8"))
    if config_dict.get("patch_size") != PLACEHOLDER:
        return None
    model_type = config_dict.pop("model_type")
    config_dict["hidden_size"] = int(hidden_size)
    config_dict["patch_size"] = int(patch_size)
    config_dict["image_size"] = int(patch_size * sqrt(num_patches))
    return AutoConfig.for_model(model_type, **config_dict)


def install_raev2_decoder_config_compat() -> None:
    """Patch the vendored loader without modifying the ignored external repo."""

    import stage1.rae as rae_module

    if getattr(rae_module, "_eqvae_decoder_config_compat", False):
        return
    original_loader = rae_module._load_decoder

    def compatible_loader(
        config_path,
        hidden_size,
        patch_size,
        num_patches,
        pretrained_path=None,
    ):
        config = resolve_decoder_config(
            config_path,
            hidden_size=hidden_size,
            patch_size=patch_size,
            num_patches=num_patches,
        )
        if config is None:
            return original_loader(
                config_path,
                hidden_size,
                patch_size,
                num_patches,
                pretrained_path,
            )
        decoder = rae_module.GeneralDecoder(config, num_patches=num_patches)
        if pretrained_path is not None:
            print(f"Loading pretrained decoder from {pretrained_path}")
            state_dict = torch.load(
                pretrained_path,
                map_location="cpu",
                weights_only=False,
            )
            keys = decoder.load_state_dict(state_dict, strict=False)
            if keys.missing_keys:
                print(f"Missing keys: {keys.missing_keys}")
        return decoder

    rae_module._load_decoder = compatible_loader
    rae_module._eqvae_decoder_config_compat = True
