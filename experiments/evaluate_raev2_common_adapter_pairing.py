"""Evaluate common adapters on fixed, unseen image/noise pairs.

Training-time LPL values cannot be compared across updates because every update
uses different images, times, and noise.  This probe holds those inputs fixed
and changes only the common-adapter checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
RAEV2_SRC = ROOT / "external" / "RAEv2" / "src"
for path in (RAEV2_SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.rae_lpl_detach_audit import (  # noqa: E402
    decoder_feature_objective_per_sample,
    tensor_rms,
)
from experiments.rae_strict_lpl import (  # noqa: E402
    decoder_feature_pyramid,
    decoder_hidden_indices,
)
from experiments.raev2_common_adapter import (  # noqa: E402
    COMMON_ADAPTER_FORMAT,
    CommonResidualAdapter,
    internal_guidance_prediction,
    load_common_adapter_checkpoint,
)
from experiments.raev2_stage1_compat import (  # noqa: E402
    install_raev2_decoder_config_compat,
)
from experiments.raev2_training_core import (  # noqa: E402
    DeterministicImageNetParquet,
    file_sha256,
    official_flow_loss_map,
    validate_full_stage2_checkpoint,
)


LPL_LAYER_FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("adapter must be NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("adapter name cannot be empty")
    return name, Path(raw_path).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-state-key", choices=("model", "ema"), default="model")
    parser.add_argument(
        "--adapter",
        action="append",
        type=parse_named_path,
        required=True,
    )
    parser.add_argument(
        "--adapter-state-key",
        choices=("adapter", "adapter_ema"),
        default="adapter",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument(
        "--noise-ratio",
        action="append",
        type=float,
        dest="noise_ratios",
    )
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument(
        "--dino-ckpt-dir",
        type=Path,
        default=Path("/home/zhoushunyu/data/eqvae/models/RAEv2/encoders/dinov3"),
    )
    parser.add_argument("--dino-repo-dir", type=Path)
    return parser.parse_args()


def autocast_context(precision: str):
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def load_config(path: Path) -> Any:
    from configs.stage2 import Stage2Config
    from stage2.utils import validate_stage2_config

    config = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(Stage2Config), OmegaConf.load(path))
    )
    config.post_process()
    validate_stage2_config(config)
    if config.transport.prediction != "x":
        raise ValueError("fixed pairing probe requires RAEv2 x-prediction")
    config.prepare_model_params()
    return config


def load_adapters(
    named_paths: list[tuple[str, Path]],
    *,
    source_sha256: str,
    source_state_key: str,
    state_key: str,
    device: torch.device,
) -> list[tuple[str, CommonResidualAdapter, dict[str, object]]]:
    names = [name for name, _ in named_paths]
    if len(names) != len(set(names)):
        raise ValueError("adapter names must be unique")

    loaded = []
    for name, raw_path in named_paths:
        path = raw_path.resolve()
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("format") != COMMON_ADAPTER_FORMAT:
            raise ValueError(f"{path} is not a common-adapter checkpoint")
        metadata = checkpoint.get("common_adapter")
        if not isinstance(metadata, dict):
            raise ValueError(f"{path} has no common_adapter metadata")
        if metadata.get("source_sha256") != source_sha256:
            raise ValueError(f"{path} was trained from a different source checkpoint")
        if metadata.get("source_state_key") != source_state_key:
            raise ValueError(f"{path} source state differs from the requested source")

        adapter_config = checkpoint.get("adapter_config")
        if not isinstance(adapter_config, dict):
            raise ValueError(f"{path} has no adapter_config")
        adapter = CommonResidualAdapter(
            int(adapter_config["channels"]),
            hidden_channels=int(adapter_config["hidden_channels"]),
            eps=float(adapter_config["eps"]),
        ).to(device)
        load_common_adapter_checkpoint(adapter, checkpoint, state_key=state_key)
        adapter.eval()
        adapter.requires_grad_(False)
        loaded.append(
            (
                name,
                adapter,
                {
                    "checkpoint_path": str(path),
                    "checkpoint_sha256": file_sha256(path),
                    "branch_update": int(metadata["branch_update"]),
                    "training_objective": str(metadata["objective"]),
                    "lpl_variant": metadata.get("lpl_variant"),
                },
            )
        )
        del checkpoint
    return loaded


def main() -> None:
    from stage2.transport import create_transport
    from utils.model_utils import instantiate_from_config

    install_raev2_decoder_config_compat()
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    noise_ratios = tuple(args.noise_ratios or (0.5, 1.0, 3.0))
    if any(ratio <= 0 for ratio in noise_ratios):
        raise ValueError("noise ratios must be positive")

    os.environ["DINOV3_CKPT_DIR"] = str(args.dino_ckpt_dir.resolve())
    if args.dino_repo_dir is not None:
        os.environ["DINOV3_REPO_DIR"] = str(args.dino_repo_dir.resolve())

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    config = load_config(args.config)
    guidance_scale = (
        float(args.guidance_scale)
        if args.guidance_scale is not None
        else float(config.guidance.ig.scale)
    )
    guidance_interval = (
        float(config.guidance.ig.t_min),
        float(config.guidance.ig.t_max),
    )

    source_path = args.source_checkpoint.resolve()
    source_sha256 = file_sha256(source_path)
    source_checkpoint = torch.load(
        source_path,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    validate_full_stage2_checkpoint(source_checkpoint)
    source_model = instantiate_from_config(config.stage_2).to(device).eval()
    source_model.load_state_dict(
        source_checkpoint[args.source_state_key],
        strict=True,
    )
    source_model.requires_grad_(False)
    source_metadata = {
        "source_checkpoint": str(source_path),
        "source_sha256": source_sha256,
        "source_state_key": args.source_state_key,
        "source_step": int(source_checkpoint["step"]),
        "source_epoch": int(source_checkpoint["epoch"]),
    }
    del source_checkpoint

    adapters = load_adapters(
        args.adapter,
        source_sha256=source_sha256,
        source_state_key=args.source_state_key,
        state_key=args.adapter_state_key,
        device=device,
    )
    if any(adapter.channels != int(source_model.in_channels) for _, adapter, _ in adapters):
        raise ValueError("an adapter channel count differs from the source model")

    dataset = DeterministicImageNetParquet(
        args.data_path,
        split="validation",
        image_size=int(config.training.image_size),
        augmentation_seed=int(args.seed),
        horizontal_flip=False,
        index_map_path=None,
    )
    if args.samples > len(dataset):
        raise ValueError("--samples exceeds the validation split size")

    rae = instantiate_from_config(config.stage_1).to(device).eval()
    rae.requires_grad_(False)
    layer_indices = decoder_hidden_indices(
        len(rae.decoder.decoder_layers),
        fractions=LPL_LAYER_FRACTIONS,
    )
    layer_weights = (1.0,) * len(layer_indices)
    latent_size = tuple(config.misc.latent_size)
    time_dist_shift = math.sqrt(
        (config.misc.time_dist_shift_dim or math.prod(latent_size))
        / config.misc.time_dist_shift_base
    )
    transport = create_transport(
        config=config.transport,
        time_dist_shift=time_dist_shift,
    )

    rows = []
    for sample_index in range(args.samples):
        image, label, data_index = dataset[sample_index]
        image = image.unsqueeze(0).to(device)
        context = torch.tensor([label], device=device, dtype=torch.long)
        with torch.inference_mode():
            clean = rae.encode(image).float()
            with autocast_context(args.precision):
                target_features = tuple(
                    feature.float()
                    for feature in decoder_feature_pyramid(
                        rae,
                        clean,
                        layer_indices=layer_indices,
                    )
                )

        for ratio_index, ratio in enumerate(noise_ratios):
            time_value = float(ratio / (1.0 + ratio))
            time = torch.full((1,), time_value, device=device)
            generator = torch.Generator(device="cpu").manual_seed(
                int(args.seed) + 10_000 * sample_index + ratio_index
            )
            noise = torch.randn(
                clean.shape,
                generator=generator,
                dtype=torch.float32,
            ).to(device)
            scale = time.reshape(1, 1, 1, 1)
            noisy = (1.0 - scale) * clean + scale * noise
            target_velocity = (noisy - clean) / scale

            with torch.inference_mode(), autocast_context(args.precision):
                source_full, source_base = source_model(
                    noisy,
                    time,
                    context=context,
                    attn_mask=None,
                )
            source_guided = internal_guidance_prediction(
                source_full,
                source_base,
                time,
                scale=guidance_scale,
                interval=guidance_interval,
            ).float()

            for name, adapter, metadata in adapters:
                with torch.inference_mode(), autocast_context(args.precision):
                    correction = adapter(
                        noisy,
                        time,
                        source_full,
                        source_base,
                    ).float()
                    corrected_full = source_full.float() + correction
                    corrected_base = source_base.float() + correction
                    corrected_guided = source_guided + correction
                    predicted_features = tuple(
                        feature.float()
                        for feature in decoder_feature_pyramid(
                            rae,
                            corrected_guided,
                            layer_indices=layer_indices,
                        )
                    )
                lpl_values, _ = decoder_feature_objective_per_sample(
                    "raw",
                    target_features,
                    predicted_features,
                    layer_weights=layer_weights,
                )
                flow_map, _ = official_flow_loss_map(
                    transport,
                    (corrected_full, corrected_base),
                    target_velocity=target_velocity,
                    noisy_latent=noisy,
                    time=time,
                    base_model_coeff=float(config.internal_guidance.base_model_coeff),
                )
                rows.append(
                    {
                        "branch": name,
                        **metadata,
                        "sample_index": sample_index,
                        "data_index": int(data_index),
                        "label": int(label),
                        "noise_to_signal_ratio": float(ratio),
                        "time": time_value,
                        "flow_loss": float(flow_map.mean()),
                        "raw_guided_lpl": float(lpl_values.mean()),
                        "guided_relative_error_rms": float(
                            tensor_rms(corrected_guided - clean)
                            / tensor_rms(clean).clamp_min(1e-30)
                        ),
                        "correction_rms": float(tensor_rms(correction)),
                        "correction_over_source_guided": float(
                            tensor_rms(correction)
                            / tensor_rms(source_guided).clamp_min(1e-30)
                        ),
                    }
                )

    raw = pd.DataFrame(rows)
    group_keys = [
        "branch",
        "checkpoint_path",
        "checkpoint_sha256",
        "branch_update",
        "training_objective",
        "lpl_variant",
        "noise_to_signal_ratio",
        "time",
    ]
    summary = (
        raw.groupby(group_keys, as_index=False, dropna=False)
        .mean(numeric_only=True)
        .sort_values(["branch_update", "branch", "noise_to_signal_ratio"])
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output_dir / "fixed_pairing_raw.csv", index=False)
    summary.to_csv(output_dir / "fixed_pairing_summary.csv", index=False)
    manifest = {
        "format_version": 1,
        "scope": "fixed_unseen_validation_pairing",
        **source_metadata,
        "config": str(args.config.resolve()),
        "data_path": str(args.data_path.resolve()),
        "split": "validation",
        "samples": int(args.samples),
        "noise_ratios": list(noise_ratios),
        "seed": int(args.seed),
        "precision": args.precision,
        "guidance_scale": guidance_scale,
        "guidance_interval": list(guidance_interval),
        "adapter_state_key": args.adapter_state_key,
        "validation_used_for_training": False,
        "interpretation_limit": (
            "This measures fixed one-step clean-latent and decoder-feature "
            "objectives; it does not replace full ODE sampling metrics."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
