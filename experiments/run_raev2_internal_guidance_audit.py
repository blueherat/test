"""Audit whether RAEv2 internal guidance amplifies LPL checkpoint drift.

The probe keeps data, noise, time, labels, and decoder fixed.  It compares the
model's full and base predictions with the prediction used by sampling:

    guided = base + ig_scale * (full - base).

It is diagnostic only and never updates a checkpoint.
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

import matplotlib.pyplot as plt
import pandas as pd
import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
RAEV2_SRC = ROOT / "external" / "RAEv2" / "src"
for path in (RAEV2_SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.rae_lpl_detach_audit import (  # noqa: E402
    lpl_loss_variants_per_sample,
    tensor_rms,
)
from experiments.rae_strict_lpl import (  # noqa: E402
    decoder_feature_pyramid,
    decoder_hidden_indices,
)
from experiments.raev2_training_core import (  # noqa: E402
    DeterministicImageNetParquet,
    split_internal_guidance_output,
    validate_full_stage2_checkpoint,
)
from experiments.raev2_stage1_compat import (  # noqa: E402
    install_raev2_decoder_config_compat,
)


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must be NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("checkpoint name cannot be empty")
    return name, Path(raw_path).expanduser()


def internal_guidance_prediction(
    full: torch.Tensor,
    base: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    if full.shape != base.shape:
        raise ValueError("full and base predictions must have identical shapes")
    return base + float(scale) * (full - base)


def relative_rms(error: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return tensor_rms(error) / tensor_rms(reference).clamp_min(1e-30)


def cosine_per_sample(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_flat = left.flatten(1)
    right_flat = right.flatten(1)
    return torch.nn.functional.cosine_similarity(left_flat, right_flat, dim=1)


def load_config(path: Path) -> Any:
    from configs.stage2 import Stage2Config
    from stage2.utils import validate_stage2_config

    config = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(Stage2Config), OmegaConf.load(path))
    )
    config.post_process()
    validate_stage2_config(config)
    if config.transport.prediction != "x":
        raise ValueError("this audit requires RAEv2 clean-latent prediction")
    return config


def autocast_context(precision: str):
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--index-map", type=Path)
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=parse_named_path,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--noise-ratio",
        action="append",
        type=float,
        dest="noise_ratios",
    )
    parser.add_argument("--ig-scale", type=float, default=1.78)
    parser.add_argument("--state-key", choices=("model", "ema"), default="model")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--dino-ckpt-dir",
        type=Path,
        default=Path("/home/zhoushunyu/data/eqvae/models/RAEv2/encoders/dinov3"),
    )
    parser.add_argument("--dino-repo-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    from stage2.transport import create_transport
    from utils.model_utils import instantiate_from_config

    install_raev2_decoder_config_compat()
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if args.split != "train" and args.index_map is not None:
        raise ValueError("--index-map is only valid for the train split")
    if args.ig_scale < 0:
        raise ValueError("--ig-scale must be non-negative")
    noise_ratios = tuple(args.noise_ratios or (0.5, 1.0, 3.0))
    if any(ratio <= 0 for ratio in noise_ratios):
        raise ValueError("noise ratios must be positive")
    names = [name for name, _ in args.checkpoint]
    if len(names) != len(set(names)):
        raise ValueError("checkpoint names must be unique")

    os.environ["DINOV3_CKPT_DIR"] = str(args.dino_ckpt_dir.resolve())
    if args.dino_repo_dir is not None:
        os.environ["DINOV3_REPO_DIR"] = str(args.dino_repo_dir.resolve())

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    config = load_config(args.config)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = DeterministicImageNetParquet(
        args.data_path,
        split=args.split,
        image_size=int(config.training.image_size),
        augmentation_seed=42,
        horizontal_flip=False,
        index_map_path=args.index_map,
    )
    if args.start_index + args.samples > len(dataset):
        raise ValueError(
            "requested sample range exceeds the dataset: "
            f"start={args.start_index}, samples={args.samples}, size={len(dataset)}"
        )
    rae = instantiate_from_config(config.stage_1).to(device).eval()
    rae.requires_grad_(False)
    cached = []
    with torch.inference_mode():
        for sample_offset in range(args.samples):
            sample_index = args.start_index + sample_offset
            image, label, data_index = dataset[sample_index]
            clean = rae.encode(image.unsqueeze(0).to(device)).float()
            cached.append(
                {
                    "sample_offset": sample_offset,
                    "sample_index": sample_index,
                    "data_index": data_index,
                    "label": label,
                    "clean": clean.cpu(),
                }
            )
    del rae.encoder
    torch.cuda.empty_cache()

    layer_indices = decoder_hidden_indices(len(rae.decoder.decoder_layers))
    for item in cached:
        with torch.inference_mode(), autocast_context(args.precision):
            features = decoder_feature_pyramid(
                rae,
                item["clean"].to(device),
                layer_indices=layer_indices,
            )
        item["target_features"] = tuple(feature.float().cpu() for feature in features)

    config.prepare_model_params()
    model = instantiate_from_config(config.stage_2).to(device).eval()
    model.requires_grad_(False)
    latent_size = tuple(config.misc.latent_size)
    time_dist_shift = math.sqrt(
        (config.misc.time_dist_shift_dim or math.prod(latent_size))
        / config.misc.time_dist_shift_base
    )
    transport = create_transport(
        config=config.transport,
        time_dist_shift=time_dist_shift,
    )

    fixed_noises = {}
    for item in cached:
        for ratio_index, ratio in enumerate(noise_ratios):
            generator = torch.Generator(device="cpu").manual_seed(
                int(args.seed) + 10_000 * item["sample_offset"] + ratio_index
            )
            fixed_noises[(item["sample_index"], ratio)] = torch.randn(
                item["clean"].shape,
                generator=generator,
                dtype=torch.float32,
            )

    rows = []
    relation_rows = []
    checkpoint_manifest = {}
    for checkpoint_name, checkpoint_path in args.checkpoint:
        checkpoint_path = checkpoint_path.resolve()
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
        validate_full_stage2_checkpoint(checkpoint)
        model.load_state_dict(checkpoint[args.state_key], strict=True)
        checkpoint_manifest[checkpoint_name] = {
            "path": str(checkpoint_path),
            "step": int(checkpoint["step"]),
            "branch_update": int(
                checkpoint.get("raev2_lpl", {}).get("branch_update", 0)
            ),
        }
        del checkpoint
        torch.cuda.empty_cache()

        for item in cached:
            clean = item["clean"].to(device)
            label = torch.tensor([item["label"]], device=device, dtype=torch.long)
            target_features = tuple(
                feature.to(device) for feature in item["target_features"]
            )
            for ratio in noise_ratios:
                time_value = float(ratio / (1.0 + ratio))
                time = torch.full((1,), time_value, device=device)
                noise = fixed_noises[(item["sample_index"], ratio)].to(device)
                time_map = time.reshape(1, 1, 1, 1)
                noisy = (1.0 - time_map) * clean + time_map * noise
                target_velocity = (noisy - clean) / time_map

                with torch.inference_mode(), autocast_context(args.precision):
                    output = model(
                        noisy,
                        time,
                        context=label,
                        attn_mask=None,
                    )
                full, base = split_internal_guidance_output(output)
                if base is None:
                    raise RuntimeError("the configured RAEv2 model has no base output")
                predictions = {
                    "full": full.float(),
                    "base": base.float(),
                    "guided": internal_guidance_prediction(
                        full.float(), base.float(), args.ig_scale
                    ),
                }
                full_error = predictions["full"] - clean
                base_error = predictions["base"] - clean
                gap = predictions["full"] - predictions["base"]
                guided_delta = predictions["guided"] - predictions["full"]
                relation_rows.append(
                    {
                        "checkpoint": checkpoint_name,
                        "sample_index": item["sample_index"],
                        "data_index": item["data_index"],
                        "label": item["label"],
                        "noise_to_signal_ratio": ratio,
                        "time": time_value,
                        "ig_scale": args.ig_scale,
                        "full_base_gap_rms": float(tensor_rms(gap).cpu()),
                        "full_base_gap_over_full_error": float(
                            (
                                tensor_rms(gap)
                                / tensor_rms(full_error).clamp_min(1e-30)
                            ).cpu()
                        ),
                        "full_base_error_cosine": float(
                            cosine_per_sample(full_error, base_error).mean().cpu()
                        ),
                        "guidance_delta_rms": float(
                            tensor_rms(guided_delta).cpu()
                        ),
                    }
                )

                for prediction_name, prediction in predictions.items():
                    with torch.inference_mode(), autocast_context(args.precision):
                        predicted_features = tuple(
                            feature.float()
                            for feature in decoder_feature_pyramid(
                                rae,
                                prediction,
                                layer_indices=layer_indices,
                            )
                        )
                    feature_losses, feature_details = lpl_loss_variants_per_sample(
                        target_features,
                        predicted_features,
                    )
                    flow_loss = transport.compute_loss(
                        prediction,
                        target_velocity,
                        noisy,
                        time,
                    ).mean(dim=(1, 2, 3))
                    error = prediction - clean
                    rows.append(
                        {
                            "checkpoint": checkpoint_name,
                            "prediction": prediction_name,
                            "sample_index": item["sample_index"],
                            "data_index": item["data_index"],
                            "label": item["label"],
                            "noise_to_signal_ratio": ratio,
                            "time": time_value,
                            "ig_scale": args.ig_scale,
                            "flow_loss": float(flow_loss.float().mean().cpu()),
                            "latent_error_rms": float(tensor_rms(error).cpu()),
                            "relative_latent_error": float(
                                relative_rms(error, clean).cpu()
                            ),
                            "raw_feature_loss": float(
                                feature_losses["raw"].float().mean().cpu()
                            ),
                            "strict_feature_loss": float(
                                feature_losses["prediction_full"].float().mean().cpu()
                            ),
                            "target_normalized_feature_loss": float(
                                feature_losses["target_normalized"].float().mean().cpu()
                            ),
                            "symmetric_feature_loss": float(
                                feature_losses["symmetric"].float().mean().cpu()
                            ),
                            "prediction_over_target_variance": float(
                                feature_details[
                                    "prediction_over_target_variance_layers"
                                ]
                                .float()
                                .mean()
                                .cpu()
                            ),
                            "centered_feature_cosine": float(
                                feature_details["centered_cosine_layers"]
                                .float()
                                .mean()
                                .cpu()
                            ),
                            "normalized_feature_mean_error": float(
                                feature_details["normalized_mean_error_layers"]
                                .float()
                                .mean()
                                .cpu()
                            ),
                        }
                    )

    raw = pd.DataFrame(rows)
    relations = pd.DataFrame(relation_rows)
    summary = (
        raw.groupby(
            ["checkpoint", "prediction", "noise_to_signal_ratio"],
            as_index=False,
        )
        .agg(
            samples=("sample_index", "size"),
            flow_loss=("flow_loss", "mean"),
            latent_error_rms=("latent_error_rms", "mean"),
            relative_latent_error=("relative_latent_error", "mean"),
            raw_feature_loss=("raw_feature_loss", "mean"),
            strict_feature_loss=("strict_feature_loss", "mean"),
            target_normalized_feature_loss=(
                "target_normalized_feature_loss",
                "mean",
            ),
            symmetric_feature_loss=("symmetric_feature_loss", "mean"),
            prediction_over_target_variance=(
                "prediction_over_target_variance",
                "mean",
            ),
            centered_feature_cosine=("centered_feature_cosine", "mean"),
            normalized_feature_mean_error=(
                "normalized_feature_mean_error",
                "mean",
            ),
        )
        .sort_values(["noise_to_signal_ratio", "checkpoint", "prediction"])
    )
    relation_summary = (
        relations.groupby(
            ["checkpoint", "noise_to_signal_ratio"],
            as_index=False,
        )
        .agg(
            samples=("sample_index", "size"),
            full_base_gap_rms=("full_base_gap_rms", "mean"),
            full_base_gap_over_full_error=(
                "full_base_gap_over_full_error",
                "mean",
            ),
            full_base_error_cosine=("full_base_error_cosine", "mean"),
            guidance_delta_rms=("guidance_delta_rms", "mean"),
        )
        .sort_values(["noise_to_signal_ratio", "checkpoint"])
    )
    raw.to_csv(output_dir / "internal_guidance_raw.csv", index=False)
    relations.to_csv(output_dir / "internal_guidance_relations_raw.csv", index=False)
    summary.to_csv(output_dir / "internal_guidance_summary.csv", index=False)
    relation_summary.to_csv(
        output_dir / "internal_guidance_relations_summary.csv",
        index=False,
    )

    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for (checkpoint, prediction), frame in summary.groupby(
        ["checkpoint", "prediction"]
    ):
        label = f"{checkpoint}/{prediction}"
        axes[0].plot(
            frame["noise_to_signal_ratio"],
            frame["flow_loss"],
            marker="o",
            label=label,
        )
        axes[1].plot(
            frame["noise_to_signal_ratio"],
            frame["strict_feature_loss"],
            marker="o",
            label=label,
        )
    axes[0].set_title("On-data-path endpoint Flow loss")
    axes[1].set_title("Strict decoder feature loss")
    for axis in axes:
        axis.set_xlabel("noise / signal ratio")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("loss")
    axes[1].set_ylabel("loss")
    axes[1].legend(fontsize=8, ncol=2)
    figure.savefig(output_dir / "internal_guidance_audit.png", dpi=180)
    plt.close(figure)

    manifest = {
        "format_version": 1,
        "scope": "fixed_input_internal_guidance_audit",
        "config": str(args.config.resolve()),
        "checkpoints": checkpoint_manifest,
        "state_key": args.state_key,
        "split": args.split,
        "start_index": args.start_index,
        "samples": args.samples,
        "noise_ratios": list(noise_ratios),
        "ig_scale": args.ig_scale,
        "seed": args.seed,
        "layer_indices": list(layer_indices),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(relation_summary.to_string(index=False))


if __name__ == "__main__":
    main()
