"""Compare RAEv2 endpoint prediction on data-path and self-induced states.

LPL supervises a clean-latent endpoint estimate made from a state on the
real latent/noise interpolation path. During ODE sampling, later endpoint
queries receive states produced by earlier model outputs. This audit starts
from the same known interpolation state and varies the number of recursive
endpoint queries while retaining the true clean latent as an evaluation
target.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
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


ROLLOUT_MODES = ("full", "base", "guided")


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must be NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("checkpoint name cannot be empty")
    return name, Path(raw_path).expanduser()


def parse_positive_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("rollout step counts must be unique")
    return values


def shifted_time_grid(
    start_time: float,
    *,
    num_steps: int,
    shift: float,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the official shifted Euler grid restricted to ``start_time -> 0``."""

    if not 0.0 < float(start_time) <= 1.0:
        raise ValueError("start_time must be in (0, 1]")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if shift <= 0:
        raise ValueError("shift must be positive")
    start_raw = float(start_time) / (
        float(shift) - (float(shift) - 1.0) * float(start_time)
    )
    raw = torch.linspace(
        start_raw,
        0.0,
        num_steps + 1,
        device=device,
        dtype=dtype,
    )
    return float(shift) * raw / (1.0 + (float(shift) - 1.0) * raw)


def select_endpoint_prediction(
    full: torch.Tensor,
    base: torch.Tensor,
    *,
    mode_ids: torch.Tensor,
    time: torch.Tensor,
    ig_scale: float,
    ig_interval: tuple[float, float],
) -> torch.Tensor:
    """Select full/base/guided endpoint predictions for a mode-stacked batch."""

    if full.shape != base.shape:
        raise ValueError("full and base predictions must have identical shapes")
    if mode_ids.ndim != 1 or mode_ids.shape[0] != full.shape[0]:
        raise ValueError("mode_ids must contain one entry per prediction")
    if time.ndim != 1 or time.shape[0] != full.shape[0]:
        raise ValueError("time must contain one entry per prediction")
    if ig_scale < 0:
        raise ValueError("ig_scale must be non-negative")
    t_min, t_max = (float(value) for value in ig_interval)
    if not t_min < t_max:
        raise ValueError("ig_interval must satisfy min < max")

    active = ((time >= t_min) & (time <= t_max)).reshape(
        (time.shape[0],) + (1,) * (full.ndim - 1)
    )
    guided = torch.where(active, base + float(ig_scale) * (full - base), full)
    mode_shape = (mode_ids.shape[0],) + (1,) * (full.ndim - 1)
    full_mask = mode_ids.eq(0).reshape(mode_shape)
    base_mask = mode_ids.eq(1).reshape(mode_shape)
    return torch.where(full_mask, full, torch.where(base_mask, base, guided))


def endpoint_rollout(
    *,
    model: torch.nn.Module,
    transport: Any,
    initial_state: torch.Tensor,
    clean: torch.Tensor,
    noise: torch.Tensor,
    labels: torch.Tensor,
    start_time: float,
    num_steps: int,
    time_shift: float,
    modes: Sequence[str],
    ig_scale: float,
    ig_interval: tuple[float, float],
    precision: str,
) -> tuple[torch.Tensor, list[dict[str, float | int | str]]]:
    """Recursively query endpoint predictions and integrate to time zero."""

    modes = tuple(modes)
    if any(mode not in ROLLOUT_MODES for mode in modes):
        raise ValueError(f"unsupported rollout modes: {modes}")
    if len(set(modes)) != len(modes):
        raise ValueError("rollout modes must be unique")
    batch_size = initial_state.shape[0]
    mode_ids = torch.tensor(
        [ROLLOUT_MODES.index(mode) for mode in modes],
        device=initial_state.device,
        dtype=torch.long,
    ).repeat_interleave(batch_size)
    state = initial_state.repeat(len(modes), 1, 1, 1)
    target = clean.repeat(len(modes), 1, 1, 1)
    source_noise = noise.repeat(len(modes), 1, 1, 1)
    context = labels.repeat(len(modes))
    grid = shifted_time_grid(
        start_time,
        num_steps=num_steps,
        shift=time_shift,
        device=initial_state.device,
        dtype=torch.float32,
    )

    rows: list[dict[str, float | int | str]] = []
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if precision == "bf16"
        else nullcontext()
    )
    for step in range(num_steps):
        time_value = grid[step]
        next_time = grid[step + 1]
        time = torch.full(
            (state.shape[0],),
            float(time_value),
            device=state.device,
            dtype=torch.float32,
        )
        exact_state = (
            (1.0 - time_value) * target + time_value * source_noise
        )
        with torch.inference_mode(), autocast:
            output = model(state, time, context=context, attn_mask=None)
        full, base = split_internal_guidance_output(output)
        if base is None:
            raise RuntimeError("endpoint rollout requires a dual-output IG model")
        endpoint = select_endpoint_prediction(
            full.float(),
            base.float(),
            mode_ids=mode_ids,
            time=time,
            ig_scale=ig_scale,
            ig_interval=ig_interval,
        )
        drift = transport.convert_model_pred(endpoint, state, time)
        state_error = tensor_rms(state.float() - exact_state.float())
        endpoint_error = tensor_rms(endpoint - target.float())
        full_base_gap = tensor_rms(full.float() - base.float())
        for mode_index, mode in enumerate(modes):
            start = mode_index * batch_size
            stop = start + batch_size
            rows.append(
                {
                    "step": step,
                    "num_steps": num_steps,
                    "mode": mode,
                    "time": float(time_value),
                    "next_time": float(next_time),
                    "state_path_error_rms": float(
                        state_error[start:stop].mean().cpu()
                    ),
                    "endpoint_latent_error_rms": float(
                        endpoint_error[start:stop].mean().cpu()
                    ),
                    "full_base_gap_rms": float(
                        full_base_gap[start:stop].mean().cpu()
                    ),
                }
            )
        state = state - (time_value - next_time) * drift
    return state.float(), rows


def load_config(path: Path) -> Any:
    from configs.stage2 import Stage2Config
    from stage2.utils import validate_stage2_config

    config = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(Stage2Config), OmegaConf.load(path))
    )
    config.post_process()
    validate_stage2_config(config)
    if config.transport.prediction != "x":
        raise ValueError("this audit requires clean-latent prediction")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--index-map", type=Path)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=parse_named_path,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rollout-steps", type=parse_positive_ints, default=(1, 2, 4, 8, 16))
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
    if args.samples <= 0 or args.start_index < 0:
        raise ValueError("samples must be positive and start-index non-negative")
    if args.split != "train" and args.index_map is not None:
        raise ValueError("--index-map is only valid for the train split")
    noise_ratios = tuple(args.noise_ratios or (0.5, 1.0, 3.0))
    if any(ratio <= 0 for ratio in noise_ratios):
        raise ValueError("noise ratios must be positive")
    checkpoint_names = [name for name, _ in args.checkpoint]
    if len(checkpoint_names) != len(set(checkpoint_names)):
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
        split=args.split,
        image_size=int(config.training.image_size),
        augmentation_seed=42,
        horizontal_flip=False,
        index_map_path=args.index_map,
    )
    if args.start_index + args.samples > len(dataset):
        raise ValueError("requested sample range exceeds the dataset")

    rae = instantiate_from_config(config.stage_1).to(device).eval()
    rae.requires_grad_(False)
    images = []
    labels = []
    data_indices = []
    for offset in range(args.samples):
        image, label, data_index = dataset[args.start_index + offset]
        images.append(image)
        labels.append(label)
        data_indices.append(data_index)
    image_batch = torch.stack(images).to(device)
    label_batch = torch.tensor(labels, device=device, dtype=torch.long)
    with torch.inference_mode():
        clean = rae.encode(image_batch).float()
    del rae.encoder
    torch.cuda.empty_cache()

    layer_indices = decoder_hidden_indices(len(rae.decoder.decoder_layers))
    with torch.inference_mode(), (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.precision == "bf16"
        else nullcontext()
    ):
        target_features = tuple(
            feature.float()
            for feature in decoder_feature_pyramid(
                rae,
                clean,
                layer_indices=layer_indices,
            )
        )

    fixed_noise = {}
    for ratio_index, ratio in enumerate(noise_ratios):
        generator = torch.Generator(device="cpu").manual_seed(
            int(args.seed) + ratio_index
        )
        fixed_noise[ratio] = torch.randn(
            clean.shape,
            generator=generator,
            dtype=torch.float32,
        ).to(device)

    config.prepare_model_params()
    model = instantiate_from_config(config.stage_2).to(device).eval()
    model.requires_grad_(False)
    latent_size = tuple(config.misc.latent_size)
    time_shift = math.sqrt(
        (config.misc.time_dist_shift_dim or math.prod(latent_size))
        / config.misc.time_dist_shift_base
    )
    transport = create_transport(config=config.transport, time_dist_shift=time_shift)
    modes = ROLLOUT_MODES
    final_rows = []
    trajectory_rows = []
    checkpoint_manifest = {}
    for checkpoint_name, checkpoint_path in args.checkpoint:
        checkpoint_path = checkpoint_path.resolve()
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
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

        for ratio in noise_ratios:
            start_time = float(ratio / (1.0 + ratio))
            noise = fixed_noise[ratio]
            initial = (1.0 - start_time) * clean + start_time * noise
            for num_steps in args.rollout_steps:
                final, local_trajectory = endpoint_rollout(
                    model=model,
                    transport=transport,
                    initial_state=initial,
                    clean=clean,
                    noise=noise,
                    labels=label_batch,
                    start_time=start_time,
                    num_steps=num_steps,
                    time_shift=time_shift,
                    modes=modes,
                    ig_scale=float(args.ig_scale),
                    ig_interval=(
                        float(config.guidance.ig.t_min),
                        float(config.guidance.ig.t_max),
                    ),
                    precision=args.precision,
                )
                for row in local_trajectory:
                    row.update(
                        {
                            "checkpoint": checkpoint_name,
                            "noise_to_signal_ratio": ratio,
                        }
                    )
                    trajectory_rows.append(row)

                with torch.inference_mode(), (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if args.precision == "bf16"
                    else nullcontext()
                ):
                    final_features = tuple(
                        feature.float()
                        for feature in decoder_feature_pyramid(
                            rae,
                            final,
                            layer_indices=layer_indices,
                        )
                    )
                repeated_targets = tuple(
                    feature.repeat(len(modes), 1, 1, 1)
                    for feature in target_features
                )
                losses, feature_details = lpl_loss_variants_per_sample(
                    repeated_targets,
                    final_features,
                )
                latent_errors = tensor_rms(
                    final - clean.repeat(len(modes), 1, 1, 1)
                )
                for mode_index, mode in enumerate(modes):
                    start = mode_index * args.samples
                    stop = start + args.samples
                    final_rows.append(
                        {
                            "checkpoint": checkpoint_name,
                            "mode": mode,
                            "noise_to_signal_ratio": ratio,
                            "start_time": start_time,
                            "num_steps": num_steps,
                            "samples": args.samples,
                            "final_latent_error_rms": float(
                                latent_errors[start:stop].mean().cpu()
                            ),
                            "final_raw_feature_loss": float(
                                losses["raw"][start:stop].mean().cpu()
                            ),
                            "final_strict_feature_loss": float(
                                losses["prediction_full"][start:stop].mean().cpu()
                            ),
                            "final_target_normalized_feature_loss": float(
                                losses["target_normalized"][start:stop].mean().cpu()
                            ),
                            "final_symmetric_feature_loss": float(
                                losses["symmetric"][start:stop].mean().cpu()
                            ),
                            "final_prediction_over_target_variance": float(
                                feature_details[
                                    "prediction_over_target_variance_layers"
                                ][start:stop]
                                .float()
                                .mean()
                                .cpu()
                            ),
                            "final_centered_feature_cosine": float(
                                feature_details["centered_cosine_layers"][start:stop]
                                .float()
                                .mean()
                                .cpu()
                            ),
                            "final_normalized_feature_mean_error": float(
                                feature_details[
                                    "normalized_mean_error_layers"
                                ][start:stop]
                                .float()
                                .mean()
                                .cpu()
                            ),
                        }
                    )
                del final, final_features

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    final_frame = pd.DataFrame(final_rows)
    trajectory_frame = pd.DataFrame(trajectory_rows)
    final_frame.to_csv(output_dir / "endpoint_rollout_final.csv", index=False)
    trajectory_frame.to_csv(
        output_dir / "endpoint_rollout_trajectory.csv",
        index=False,
    )

    figure, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)
    metric_specs = (
        ("final_latent_error_rms", "Final latent RMS"),
        ("final_raw_feature_loss", "Final raw feature loss"),
        ("final_strict_feature_loss", "Final strict feature loss"),
    )
    for axis, (metric, title) in zip(axes, metric_specs):
        for (checkpoint, mode, ratio), frame in final_frame.groupby(
            ["checkpoint", "mode", "noise_to_signal_ratio"]
        ):
            axis.plot(
                frame["num_steps"],
                frame[metric],
                marker="o",
                label=f"{checkpoint}/{mode}/r={ratio:g}",
            )
        axis.set_xscale("log", base=2)
        axis.set_xticks(list(args.rollout_steps))
        axis.set_xticklabels([str(value) for value in args.rollout_steps])
        axis.set_xlabel("recursive endpoint queries")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[-1].legend(fontsize=7, ncol=2, bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.savefig(output_dir / "endpoint_rollout_curves.png", dpi=180)
    plt.close(figure)

    manifest = {
        "format_version": 1,
        "scope": "raev2_endpoint_rollout_audit",
        "config": str(args.config.resolve()),
        "checkpoints": checkpoint_manifest,
        "state_key": args.state_key,
        "split": args.split,
        "start_index": args.start_index,
        "samples": args.samples,
        "data_indices": data_indices,
        "noise_ratios": list(noise_ratios),
        "rollout_steps": list(args.rollout_steps),
        "ig_scale": float(args.ig_scale),
        "ig_interval": [
            float(config.guidance.ig.t_min),
            float(config.guidance.ig.t_max),
        ],
        "time_shift": time_shift,
        "seed": args.seed,
        "layer_indices": list(layer_indices),
        "terminology": {
            "one_query": "endpoint estimate from an on-data-path state",
            "multiple_queries": "recursive endpoint estimates on self-induced states",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(final_frame.to_string(index=False))


if __name__ == "__main__":
    main()
