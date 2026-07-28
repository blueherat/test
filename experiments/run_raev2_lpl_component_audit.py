"""Audit numerator and prediction-variance gradients in RAEv2 LPL.

This is a mechanism probe, not another training branch. It holds images,
latents, labels, noise, time, decoder, and checkpoint state fixed while
decomposing the strict LPL output-latent gradient into

    g_full = g_error + g_variance.

The output-latent audit is deliberately run before any parameter-level or
variance-only training study.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
RAEV2_SRC = ROOT / "external" / "RAEv2" / "src"
for path in (RAEV2_SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.rae_lpl_detach_audit import (  # noqa: E402
    cosine_per_sample,
    gradient_decomposition_metrics,
    lpl_loss_variants_per_sample,
    tensor_rms,
)
from experiments.rae_strict_lpl import (  # noqa: E402
    decoder_feature_pyramid,
    decoder_hidden_indices,
)
from experiments.raev2_training_core import (  # noqa: E402
    DeterministicImageNetParquet,
    predicted_clean_latent,
    split_internal_guidance_output,
    validate_full_stage2_checkpoint,
)
def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must be NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("checkpoint name cannot be empty")
    return name, Path(raw_path).expanduser()


def load_config(path: Path) -> Any:
    from configs.stage2 import Stage2Config
    from stage2.utils import validate_stage2_config

    config = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(Stage2Config), OmegaConf.load(path))
    )
    config.post_process()
    validate_stage2_config(config)
    if config.transport.prediction != "x":
        raise ValueError("this audit currently requires clean-latent prediction")
    return config


def autocast_context(precision: str):
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def dot_per_sample(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape != right.shape:
        raise ValueError("dot-product inputs must have identical shapes")
    return (left.flatten(1) * right.flatten(1)).sum(dim=1)


def normalized_descent_effect(
    metric_gradient: torch.Tensor,
    step_gradient: torch.Tensor,
) -> torch.Tensor:
    """First-order metric change under a unit-RMS descent step."""

    dimensions = step_gradient[0].numel()
    normalized = step_gradient / tensor_rms(step_gradient).clamp_min(1e-30).view(
        -1, *([1] * (step_gradient.ndim - 1))
    )
    return -dot_per_sample(metric_gradient, normalized) / math.sqrt(dimensions)


def component_metrics(
    *,
    flow_gradient: torch.Tensor,
    raw_gradient: torch.Tensor,
    error_gradient: torch.Tensor,
    variance_gradient: torch.Tensor,
    full_gradient: torch.Tensor,
    log_variance_gradient: torch.Tensor,
) -> Mapping[str, torch.Tensor]:
    """Return interpretable magnitudes, alignments, and directional effects."""

    decomposition = gradient_decomposition_metrics(
        raw_gradient,
        error_gradient,
        full_gradient,
        log_variance_gradient,
    )
    consistency = tensor_rms(
        full_gradient - error_gradient - variance_gradient
    ) / tensor_rms(full_gradient).clamp_min(1e-30)
    return {
        **decomposition,
        "flow_gradient_rms": tensor_rms(flow_gradient),
        "variance_only_gradient_rms": tensor_rms(variance_gradient),
        "variance_over_error_gradient_rms": tensor_rms(variance_gradient)
        / tensor_rms(error_gradient).clamp_min(1e-30),
        "gradient_split_relative_residual": consistency,
        "flow_error_gradient_cosine": cosine_per_sample(
            flow_gradient, error_gradient
        ),
        "flow_variance_gradient_cosine": cosine_per_sample(
            flow_gradient, variance_gradient
        ),
        "flow_full_gradient_cosine": cosine_per_sample(
            flow_gradient, full_gradient
        ),
        "error_descent_raw_change": normalized_descent_effect(
            raw_gradient, error_gradient
        ),
        "variance_descent_raw_change": normalized_descent_effect(
            raw_gradient, variance_gradient
        ),
        "full_descent_raw_change": normalized_descent_effect(
            raw_gradient, full_gradient
        ),
        "error_descent_log_variance_change": normalized_descent_effect(
            log_variance_gradient, error_gradient
        ),
        "variance_descent_log_variance_change": normalized_descent_effect(
            log_variance_gradient, variance_gradient
        ),
        "full_descent_log_variance_change": normalized_descent_effect(
            log_variance_gradient, full_gradient
        ),
    }


def gradient(
    loss: torch.Tensor,
    prediction: torch.Tensor,
    *,
    retain_graph: bool,
) -> torch.Tensor:
    return torch.autograd.grad(
        loss.sum(),
        prediction,
        retain_graph=retain_graph,
        create_graph=False,
    )[0]


def checkpoint_metadata(checkpoint: dict, path: Path) -> dict[str, object]:
    metadata = checkpoint.get("raev2_lpl", {})
    return {
        "checkpoint_path": str(path.resolve()),
        "checkpoint_step": int(checkpoint.get("step", 0)),
        "branch_update": int(metadata.get("branch_update", 0)),
        "training_objective": str(metadata.get("objective", "official")),
    }


def scalar(value: torch.Tensor) -> float:
    return float(value.detach().float().mean().cpu())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--index-map", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=parse_named_path,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument(
        "--noise-ratio",
        action="append",
        type=float,
        dest="noise_ratios",
    )
    parser.add_argument("--state-key", choices=("model", "ema"), default="model")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--lpl-noise-threshold", type=float, default=3.0)
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

    args = parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    noise_ratios = tuple(args.noise_ratios or (0.5, 1.0, 3.0, 5.0))
    if any(ratio <= 0 for ratio in noise_ratios):
        raise ValueError("noise ratios must be positive")
    if args.lpl_noise_threshold <= 0:
        raise ValueError("--lpl-noise-threshold must be positive")
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

    dataset = DeterministicImageNetParquet(
        args.data_path,
        split="train",
        image_size=int(config.training.image_size),
        augmentation_seed=42,
        horizontal_flip=False,
        index_map_path=args.index_map,
    )
    rae = instantiate_from_config(config.stage_1).to(device).eval()
    rae.requires_grad_(False)
    cached = []
    with torch.inference_mode():
        for sample_index in range(args.samples):
            image, label, data_index = dataset[sample_index]
            clean = rae.encode(image.unsqueeze(0).to(device)).float()
            cached.append(
                {
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
            target_features = decoder_feature_pyramid(
                rae,
                item["clean"].to(device),
                layer_indices=layer_indices,
            )
        item["target_features"] = tuple(
            feature.float().cpu() for feature in target_features
        )

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
                int(args.seed) + 10_000 * item["sample_index"] + ratio_index
            )
            fixed_noises[(item["sample_index"], ratio)] = torch.randn(
                item["clean"].shape,
                generator=generator,
                dtype=torch.float32,
            )

    rows = []
    layer_rows = []
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
        metadata = checkpoint_metadata(checkpoint, checkpoint_path)
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
                scale = time.reshape(1, 1, 1, 1)
                noisy = (1.0 - scale) * clean + scale * noise
                target_velocity = (noisy - clean) / scale
                with torch.inference_mode(), autocast_context(args.precision):
                    output = model(
                        noisy,
                        time,
                        context=label,
                        attn_mask=None,
                    )
                    primary, _ = split_internal_guidance_output(output)
                    clean_prediction = predicted_clean_latent(
                        primary,
                        prediction=config.transport.prediction,
                        noisy_latent=noisy,
                        time=time,
                    )

                prediction = (
                    clean_prediction.float().detach().clone().requires_grad_(True)
                )
                with autocast_context(args.precision):
                    predicted_features = tuple(
                        feature.float()
                        for feature in decoder_feature_pyramid(
                            rae,
                            prediction,
                            layer_indices=layer_indices,
                        )
                    )
                losses, details = lpl_loss_variants_per_sample(
                    target_features,
                    predicted_features,
                )
                flow_loss = transport.compute_loss(
                    prediction,
                    target_velocity,
                    noisy,
                    time,
                ).mean(dim=(1, 2, 3))

                flow_gradient = gradient(flow_loss, prediction, retain_graph=True)
                raw_gradient = gradient(
                    losses["raw"], prediction, retain_graph=True
                )
                error_gradient = gradient(
                    losses["prediction_detach"],
                    prediction,
                    retain_graph=True,
                )
                variance_gradient = gradient(
                    losses["variance_only"],
                    prediction,
                    retain_graph=True,
                )
                full_gradient = gradient(
                    losses["prediction_full"],
                    prediction,
                    retain_graph=True,
                )
                log_variance_gradient = gradient(
                    details["mean_log_prediction_variance"],
                    prediction,
                    retain_graph=True,
                )
                target_normalized_gradient = gradient(
                    losses["target_normalized"],
                    prediction,
                    retain_graph=True,
                )
                symmetric_gradient = gradient(
                    losses["symmetric"],
                    prediction,
                    retain_graph=False,
                )
                metrics = component_metrics(
                    flow_gradient=flow_gradient,
                    raw_gradient=raw_gradient,
                    error_gradient=error_gradient,
                    variance_gradient=variance_gradient,
                    full_gradient=full_gradient,
                    log_variance_gradient=log_variance_gradient,
                )
                row = {
                    "checkpoint": checkpoint_name,
                    "state_key": args.state_key,
                    **metadata,
                    "sample_index": item["sample_index"],
                    "data_index": item["data_index"],
                    "label": item["label"],
                    "noise_to_signal_ratio": ratio,
                    "time": time_value,
                    "lpl_gate_active": ratio <= args.lpl_noise_threshold,
                    "flow_loss": scalar(flow_loss),
                    **{
                        f"{name}_loss": scalar(loss)
                        for name, loss in losses.items()
                    },
                    "target_normalized_gradient_rms": scalar(
                        tensor_rms(target_normalized_gradient)
                    ),
                    "symmetric_gradient_rms": scalar(
                        tensor_rms(symmetric_gradient)
                    ),
                    **{name: scalar(value) for name, value in metrics.items()},
                }
                rows.append(row)

                for layer_position, layer_index in enumerate(layer_indices):
                    layer_rows.append(
                        {
                            "checkpoint": checkpoint_name,
                            "state_key": args.state_key,
                            **metadata,
                            "sample_index": item["sample_index"],
                            "noise_to_signal_ratio": ratio,
                            "decoder_layer_index": layer_index,
                            **{
                                key.removesuffix("_layers"): scalar(
                                    value[:, layer_position]
                                )
                                for key, value in details.items()
                                if key.endswith("_layers")
                            },
                        }
                    )
                del prediction, predicted_features

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_frame = pd.DataFrame(rows)
    layer_frame = pd.DataFrame(layer_rows)
    group_keys = [
        "checkpoint",
        "state_key",
        "checkpoint_step",
        "branch_update",
        "training_objective",
        "noise_to_signal_ratio",
        "time",
        "lpl_gate_active",
    ]
    summary = (
        raw_frame.groupby(group_keys, as_index=False)
        .mean(numeric_only=True)
        .sort_values(["checkpoint", "noise_to_signal_ratio"])
    )
    raw_frame.to_csv(output_dir / "component_audit_raw.csv", index=False)
    layer_frame.to_csv(output_dir / "component_audit_layers.csv", index=False)
    summary.to_csv(output_dir / "component_audit_summary.csv", index=False)
    manifest = {
        "format_version": 1,
        "scope": "output_latent_gradient_phase0",
        "interpretation_limit": (
            "This measures gradients with respect to the predicted clean latent, "
            "not full Stage-2 parameter gradients."
        ),
        "config": str(args.config.resolve()),
        "checkpoints": {
            name: str(path.resolve()) for name, path in args.checkpoint
        },
        "state_key": args.state_key,
        "samples": args.samples,
        "noise_ratios": noise_ratios,
        "seed": args.seed,
        "lpl_noise_threshold": args.lpl_noise_threshold,
        "layer_indices": layer_indices,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = (
        ("variance_over_error_gradient_rms", "||g_V|| / ||g_E||"),
        ("flow_variance_gradient_cosine", "cos(g_flow, g_V)"),
        (
            "variance_descent_log_variance_change",
            "Change in log variance under -g_V",
        ),
    )
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for axis, (column, title) in zip(axes, panels, strict=True):
        for checkpoint_name, subset in summary.groupby("checkpoint"):
            axis.plot(
                subset["noise_to_signal_ratio"],
                subset[column],
                marker="o",
                linewidth=2,
                label=checkpoint_name,
            )
        axis.axhline(0.0, color="#111827", linewidth=1, alpha=0.5)
        axis.axvline(
            args.lpl_noise_threshold,
            color="#6b7280",
            linestyle="--",
            linewidth=1,
        )
        axis.set_xscale("log")
        axis.set_xlabel("Noise / signal coefficient ratio")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=len(labels))
    figure.suptitle("RAEv2 LPL numerator/denominator gradient audit")
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    figure.savefig(
        output_dir / "component_audit.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
