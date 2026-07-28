"""Convert the official diffusers DINOv2-L RAE into this repo's native layout."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import (
    AutoImageProcessor,
    Dinov2WithRegistersConfig,
    Dinov2WithRegistersModel,
    ViTMAEConfig,
)


ROOT = Path(__file__).resolve().parents[1]
RAE_ROOT = ROOT / "external" / "RAE"
RAE_SRC = RAE_ROOT / "src"
for path in (ROOT, RAE_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from stage1.decoders import GeneralDecoder  # noqa: E402


DEFAULT_SOURCE = (
    Path.home()
    / "data/eqvae/models/RAE-dinov2-wReg-large-ViTXL-n08/"
    "diffusion_pytorch_model.safetensors"
)
DEFAULT_OUTPUT = Path.home() / "data/eqvae/models/RAE/dinov2/wReg_large"
DEFAULT_ENCODER_SOURCE = "facebook/dinov2-with-registers-large"
OFFICIAL_MODEL_ID = "nyu-visionx/RAE-dinov2-wReg-large-ViTXL-n08"
OFFICIAL_REVISION = "de59cdf32cc515014081afcfc3ec24d82605b7fb"


def convert_diffusers_decoder_key(key: str) -> str | None:
    """Map one diffusers RAEDecoder parameter key to RAE-main naming."""

    if not key.startswith("decoder."):
        return None
    converted = key.removeprefix("decoder.")
    replacements = (
        (".attention.to_q.", ".attention.attention.query."),
        (".attention.to_k.", ".attention.attention.key."),
        (".attention.to_v.", ".attention.attention.value."),
        (".attention.to_out.0.", ".attention.output.dense."),
    )
    for source, target in replacements:
        converted = converted.replace(source, target)
    return converted


def _atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _load_selected(
    source: Path,
    prefix: str,
    *,
    key_converter,
) -> dict[str, torch.Tensor]:
    result = {}
    with safe_open(source, framework="pt", device="cpu") as handle:
        for source_key in handle.keys():
            if not source_key.startswith(prefix):
                continue
            target_key = key_converter(source_key)
            if target_key is None:
                continue
            if target_key in result:
                raise KeyError(f"duplicate converted key {target_key!r}")
            result[target_key] = handle.get_tensor(source_key).contiguous()
    return result


def _assert_exact_state(
    expected: dict[str, torch.Tensor],
    actual: dict[str, torch.Tensor],
    *,
    component: str,
) -> None:
    missing = sorted(expected.keys() - actual.keys())
    unexpected = sorted(actual.keys() - expected.keys())
    shape_mismatch = sorted(
        key
        for key in expected.keys() & actual.keys()
        if tuple(expected[key].shape) != tuple(actual[key].shape)
    )
    if missing or unexpected or shape_mismatch:
        raise RuntimeError(
            f"{component} conversion mismatch: missing={missing[:8]}, "
            f"unexpected={unexpected[:8]}, shape_mismatch={shape_mismatch[:8]}"
        )


def convert_decoder(source: Path, output: Path) -> dict[str, object]:
    config_payload = json.loads(
        (RAE_ROOT / "configs/decoder/ViTXL/config.json").read_text(encoding="utf-8")
    )
    config_payload["patch_size"] = 16
    decoder_config = ViTMAEConfig.from_dict(config_payload)
    decoder_config.hidden_size = 1024
    decoder_config.patch_size = 16
    decoder_config.image_size = 256
    decoder = GeneralDecoder(decoder_config, num_patches=256)
    expected = decoder.state_dict()

    converted = _load_selected(
        source,
        "decoder.",
        key_converter=convert_diffusers_decoder_key,
    )
    # Diffusers regenerates this fixed sinusoidal buffer instead of serializing it.
    converted["decoder_pos_embed"] = expected["decoder_pos_embed"].clone()
    _assert_exact_state(expected, converted, component="decoder")
    decoder.load_state_dict(converted, strict=True)
    _atomic_torch_save(converted, output)
    return {
        "path": str(output),
        "tensor_count": len(converted),
        "parameter_count": int(
            sum(value.numel() for key, value in converted.items() if key != "decoder_pos_embed")
        ),
        "input_channels": 1024,
        "decoder_hidden_size": int(decoder_config.decoder_hidden_size),
        "decoder_depth": int(decoder_config.decoder_num_hidden_layers),
    }


def convert_encoder(
    source: Path,
    output: Path,
    *,
    encoder_source: str,
) -> dict[str, object]:
    config = Dinov2WithRegistersConfig.from_pretrained(encoder_source)
    if (
        int(config.hidden_size) != 1024
        or int(config.num_hidden_layers) != 24
        or int(config.patch_size) != 14
    ):
        raise ValueError(
            "expected DINOv2-L/14 with registers: "
            f"hidden={config.hidden_size}, depth={config.num_hidden_layers}, "
            f"patch={config.patch_size}"
        )
    model = Dinov2WithRegistersModel(config)
    model.layernorm.elementwise_affine = False
    model.layernorm.weight = None
    model.layernorm.bias = None
    expected = model.state_dict()
    converted = _load_selected(
        source,
        "encoder.",
        key_converter=lambda key: key.removeprefix("encoder."),
    )
    _assert_exact_state(expected, converted, component="encoder")
    model.load_state_dict(converted, strict=True)

    output.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(output)
    processor = AutoImageProcessor.from_pretrained(encoder_source)
    processor.save_pretrained(output)
    temporary = output / "model.safetensors.tmp"
    save_file(converted, temporary)
    os.replace(temporary, output / "model.safetensors")
    return {
        "path": str(output),
        "tensor_count": len(converted),
        "parameter_count": int(sum(value.numel() for value in converted.values())),
        "hidden_size": int(config.hidden_size),
        "depth": int(config.num_hidden_layers),
        "patch_size": int(config.patch_size),
    }


def convert_stats(source: Path, output: Path) -> dict[str, object]:
    with safe_open(source, framework="pt", device="cpu") as handle:
        mean = handle.get_tensor("_latents_mean").contiguous()
        std = handle.get_tensor("_latents_std").contiguous()
    if tuple(mean.shape) != (1024, 16, 16) or mean.shape != std.shape:
        raise ValueError(f"unexpected Large latent statistics: {mean.shape}, {std.shape}")
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError("Large latent statistics contain non-finite values")
    if not torch.all(std > 0):
        raise ValueError("Large latent standard deviations must be positive")
    _atomic_torch_save({"mean": mean, "var": std.square()}, output)
    return {
        "path": str(output),
        "shape": list(mean.shape),
        "mean_min": float(mean.min()),
        "mean_max": float(mean.max()),
        "std_min": float(std.min()),
        "std_max": float(std.max()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--encoder-source", default=DEFAULT_ENCODER_SOURCE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    decoder_path = output / "decoder/ViTXL_n08/model.pt"
    encoder_path = output / "encoder"
    stats_path = output / "stats/imagenet1k/stat.pt"
    payload = {
        "source_model_id": OFFICIAL_MODEL_ID,
        "source_revision": OFFICIAL_REVISION,
        "source_path": str(source),
        "source_bytes": source.stat().st_size,
        "decoder": convert_decoder(source, decoder_path),
        "encoder": convert_encoder(
            source,
            encoder_path,
            encoder_source=args.encoder_source,
        ),
        "stats": convert_stats(source, stats_path),
    }
    manifest_path = output / "conversion_manifest.json"
    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(manifest_path)


if __name__ == "__main__":
    main()
