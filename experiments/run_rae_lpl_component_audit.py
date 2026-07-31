"""Audit strict-LPL output gradients for the original deterministic RAE.

The output schema intentionally matches ``run_raev2_lpl_component_audit.py``.
Both systems are evaluated at fixed noise ratios and all gradients are taken
with respect to the predicted clean latent, so velocity and x-prediction use
the same coordinate system.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
RAE_SRC = ROOT / "external" / "RAE" / "src"
for path in (RAE_SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.rae_latent_cache import CachedRAELatentDataset  # noqa: E402
from experiments.rae_lpl_detach_audit import (  # noqa: E402
    lpl_loss_variants_per_sample,
    tensor_rms,
)
from experiments.rae_strict_lpl import (  # noqa: E402
    decoder_feature_pyramid,
    decoder_hidden_indices,
    flow_clean_estimate,
)
from experiments.lpl_component_metrics import (  # noqa: E402
    component_metrics,
    gradient,
    scalar,
)
from experiments.train_rae_strict_lpl import (  # noqa: E402
    model_state_from_checkpoint,
    resolve_rae_paths,
)


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must be NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("checkpoint name cannot be empty")
    return name, Path(raw_path).expanduser()


def autocast_context(precision: str):
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def checkpoint_state(
    checkpoint: Any,
    *,
    state_key: str,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    if isinstance(checkpoint, dict) and state_key in checkpoint:
        candidate = checkpoint[state_key]
        if isinstance(candidate, dict) and candidate:
            return candidate, {
                "checkpoint_step": int(checkpoint.get("step", 0)),
                "branch_update": int(
                    checkpoint.get("step", 0) - checkpoint.get("branch_start_step", 0)
                ),
                "training_objective": str(checkpoint.get("objective", "continued")),
            }
    return model_state_from_checkpoint(checkpoint), {
        "checkpoint_step": 0,
        "branch_update": 0,
        "training_objective": "official",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--latent-cache", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=parse_named_path,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument(
        "--noise-ratio",
        action="append",
        type=float,
        dest="noise_ratios",
    )
    parser.add_argument("--state-key", choices=("model", "ema"), default="model")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--lpl-noise-threshold", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    from utils.model_utils import instantiate_from_config

    args = parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    noise_ratios = tuple(args.noise_ratios or (1 / 3, 2 / 3, 11 / 9, 7 / 3))
    if any(ratio <= 0 for ratio in noise_ratios):
        raise ValueError("noise ratios must be positive")
    names = [name for name, _ in args.checkpoint]
    if len(names) != len(set(names)):
        raise ValueError("checkpoint names must be unique")

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    config = OmegaConf.load(args.config)
    resolve_rae_paths(config)
    if str(config.transport.params.prediction) != "velocity":
        raise ValueError("original RAE audit requires velocity prediction")

    cache = CachedRAELatentDataset(args.latent_cache)
    calibration_count = int(cache.manifest.get("calibration_count", 0))
    heldout = CachedRAELatentDataset(
        args.latent_cache,
        start=calibration_count,
        stop=calibration_count + int(args.samples),
    )
    cached = []
    for sample_index in range(len(heldout)):
        clean, label = heldout[sample_index]
        cached.append(
            {
                "sample_index": sample_index,
                "data_index": calibration_count + sample_index,
                "label": int(label),
                "clean": clean.unsqueeze(0),
            }
        )

    rae = instantiate_from_config(config.stage_1).to(device).eval()
    rae.requires_grad_(False)
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

    model = instantiate_from_config(config.stage_2).to(device).eval()
    model.requires_grad_(False)
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

    rows: list[dict[str, object]] = []
    layer_rows: list[dict[str, object]] = []
    for checkpoint_name, checkpoint_path in args.checkpoint:
        checkpoint_path = checkpoint_path.resolve()
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
        state, metadata = checkpoint_state(checkpoint, state_key=args.state_key)
        model.load_state_dict(state, strict=True)
        del checkpoint, state
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
                with torch.inference_mode(), autocast_context(args.precision):
                    velocity = model(noisy, time, y=label)
                    clean_prediction = flow_clean_estimate(noisy, velocity, time)

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
                flow_loss = (
                    (prediction - clean).square()
                    / scale.square().clamp_min(1e-30)
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
                rows.append(
                    {
                        "system": "rae",
                        "prediction_target": "single",
                        "checkpoint": checkpoint_name,
                        "state_key": args.state_key,
                        "checkpoint_path": str(checkpoint_path),
                        **metadata,
                        "sample_index": item["sample_index"],
                        "data_index": item["data_index"],
                        "label": item["label"],
                        "noise_to_signal_ratio": ratio,
                        "time": time_value,
                        "lpl_gate_active": ratio <= args.lpl_noise_threshold,
                        "flow_loss": scalar(flow_loss),
                        "latent_relative_error_rms": scalar(
                            tensor_rms(prediction - clean)
                            / tensor_rms(clean).clamp_min(1e-30)
                        ),
                        "prediction_over_target_variance": scalar(
                            details["prediction_over_target_variance_layers"]
                            .clamp_min(1e-30)
                            .log()
                            .mean(dim=1)
                            .exp()
                        ),
                        **{
                            f"{name}_loss": scalar(loss)
                            for name, loss in losses.items()
                        },
                        **{name: scalar(value) for name, value in metrics.items()},
                    }
                )
                for layer_position, layer_index in enumerate(layer_indices):
                    layer_rows.append(
                        {
                            "system": "rae",
                            "prediction_target": "single",
                            "checkpoint": checkpoint_name,
                            "state_key": args.state_key,
                            **metadata,
                            "sample_index": item["sample_index"],
                            "noise_to_signal_ratio": ratio,
                            "time": time_value,
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
        "system",
        "prediction_target",
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
        .sort_values(["checkpoint", "time"])
    )
    raw_frame.to_csv(output_dir / "component_audit_raw.csv", index=False)
    layer_frame.to_csv(output_dir / "component_audit_layers.csv", index=False)
    summary.to_csv(output_dir / "component_audit_summary.csv", index=False)
    manifest = {
        "format_version": 1,
        "scope": "original_rae_output_latent_gradient",
        "config": str(args.config.resolve()),
        "latent_cache": str(args.latent_cache.resolve()),
        "checkpoints": {
            name: str(path.resolve()) for name, path in args.checkpoint
        },
        "state_key": args.state_key,
        "samples": args.samples,
        "noise_ratios": noise_ratios,
        "times": [ratio / (1.0 + ratio) for ratio in noise_ratios],
        "seed": args.seed,
        "lpl_noise_threshold": args.lpl_noise_threshold,
        "layer_indices": layer_indices,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
