"""Test common adapters after recursive endpoint queries on self-induced states."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
RAEV2_SRC = ROOT / "external" / "RAEv2" / "src"
for path in (RAEV2_SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.evaluate_raev2_common_adapter_pairing import (  # noqa: E402
    autocast_context,
    load_adapters,
    load_config,
    parse_named_path,
)
from experiments.rae_lpl_detach_audit import (  # noqa: E402
    decoder_feature_objective_per_sample,
    tensor_rms,
)
from experiments.rae_strict_lpl import (  # noqa: E402
    decoder_feature_pyramid,
    decoder_hidden_indices,
)
from experiments.raev2_common_adapter import (  # noqa: E402
    internal_guidance_prediction,
)
from experiments.raev2_stage1_compat import (  # noqa: E402
    install_raev2_decoder_config_compat,
)
from experiments.raev2_training_core import (  # noqa: E402
    DeterministicImageNetParquet,
    file_sha256,
    validate_full_stage2_checkpoint,
)
from experiments.run_raev2_endpoint_rollout_audit import (  # noqa: E402
    shifted_time_grid,
)


LPL_LAYER_FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)


def parse_positive_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("rollout step counts must be unique")
    return values


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
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--rollout-steps",
        type=parse_positive_ints,
        default=(1, 4, 16),
    )
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


def main() -> None:
    from stage2.transport import create_transport
    from utils.model_utils import instantiate_from_config

    install_raev2_decoder_config_compat()
    args = parse_args()
    if args.samples <= 0 or args.batch_size <= 0:
        raise ValueError("--samples and --batch-size must be positive")
    noise_ratios = tuple(args.noise_ratios or (1.0, 3.0))
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
    source_step = int(source_checkpoint["step"])
    source_epoch = int(source_checkpoint["epoch"])
    del source_checkpoint

    adapters = load_adapters(
        args.adapter,
        source_sha256=source_sha256,
        source_state_key=args.source_state_key,
        state_key=args.adapter_state_key,
        device=device,
    )
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
    cached_clean = []
    cached_labels = []
    data_indices = []

    rae = instantiate_from_config(config.stage_1).to(device).eval()
    rae.requires_grad_(False)
    for start in range(0, args.samples, args.batch_size):
        stop = min(start + args.batch_size, args.samples)
        images = []
        labels = []
        for sample_index in range(start, stop):
            image, label, data_index = dataset[sample_index]
            images.append(image)
            labels.append(label)
            data_indices.append(int(data_index))
        with torch.inference_mode():
            clean_batch = rae.encode(torch.stack(images).to(device)).float()
        cached_clean.append(clean_batch.cpu())
        cached_labels.extend(labels)
    clean_cache = torch.cat(cached_clean)
    label_cache = torch.tensor(cached_labels, dtype=torch.long)
    del cached_clean
    del rae.encoder
    torch.cuda.empty_cache()
    layer_indices = decoder_hidden_indices(
        len(rae.decoder.decoder_layers),
        fractions=LPL_LAYER_FRACTIONS,
    )
    layer_weights = (1.0,) * len(layer_indices)
    latent_size = tuple(config.misc.latent_size)
    time_shift = math.sqrt(
        (config.misc.time_dist_shift_dim or math.prod(latent_size))
        / config.misc.time_dist_shift_base
    )
    transport = create_transport(config=config.transport, time_dist_shift=time_shift)
    rows = []
    for batch_start in range(0, args.samples, args.batch_size):
        batch_stop = min(batch_start + args.batch_size, args.samples)
        clean = clean_cache[batch_start:batch_stop].to(device)
        labels = label_cache[batch_start:batch_stop].to(device)
        batch_size = batch_stop - batch_start
        with torch.inference_mode(), autocast_context(args.precision):
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
            values = []
            for sample_index in range(batch_start, batch_stop):
                generator = torch.Generator(device="cpu").manual_seed(
                    int(args.seed) + 10_000 * sample_index + ratio_index
                )
                values.append(
                    torch.randn(
                        clean.shape[1:],
                        generator=generator,
                        dtype=torch.float32,
                    )
                )
            fixed_noise[ratio] = torch.stack(values).to(device)

        for name, adapter, metadata in adapters:
            for ratio in noise_ratios:
                start_time = float(ratio / (1.0 + ratio))
                noise = fixed_noise[ratio]
                initial = (1.0 - start_time) * clean + start_time * noise
                for num_steps in args.rollout_steps:
                    state = initial.clone()
                    grid = shifted_time_grid(
                        start_time,
                        num_steps=num_steps,
                        shift=time_shift,
                        device=device,
                        dtype=torch.float32,
                    )
                    state_path_errors = []
                    endpoint_errors = []
                    correction_ratios = []
                    for step_index in range(num_steps):
                        time_value = grid[step_index]
                        next_time = grid[step_index + 1]
                        time = torch.full(
                            (batch_size,),
                            float(time_value),
                            device=device,
                            dtype=torch.float32,
                        )
                        exact_state = (
                            (1.0 - time_value) * clean + time_value * noise
                        )
                        with torch.inference_mode(), autocast_context(args.precision):
                            source_full, source_base = source_model(
                                state,
                                time,
                                context=labels,
                                attn_mask=None,
                            )
                            correction = adapter(
                                state,
                                time,
                                source_full,
                                source_base,
                            ).float()
                        source_guided = internal_guidance_prediction(
                            source_full,
                            source_base,
                            time,
                            scale=guidance_scale,
                            interval=guidance_interval,
                        ).float()
                        endpoint = source_guided + correction
                        drift = transport.convert_model_pred(endpoint, state, time)
                        state_path_errors.append(tensor_rms(state - exact_state))
                        endpoint_errors.append(tensor_rms(endpoint - clean))
                        correction_ratios.append(
                            tensor_rms(correction)
                            / tensor_rms(source_guided).clamp_min(1e-30)
                        )
                        state = state - (time_value - next_time) * drift

                    with torch.inference_mode(), autocast_context(args.precision):
                        final_features = tuple(
                            feature.float()
                            for feature in decoder_feature_pyramid(
                                rae,
                                state.float(),
                                layer_indices=layer_indices,
                            )
                        )
                    final_lpl, _ = decoder_feature_objective_per_sample(
                        "raw",
                        target_features,
                        final_features,
                        layer_weights=layer_weights,
                    )
                    final_latent_error = tensor_rms(state.float() - clean)
                    state_path = torch.stack(state_path_errors)
                    endpoint_path = torch.stack(endpoint_errors)
                    correction_path = torch.stack(correction_ratios)
                    for local_index in range(batch_size):
                        sample_index = batch_start + local_index
                        rows.append(
                            {
                                "branch": name,
                                **metadata,
                                "sample_index": sample_index,
                                "data_index": data_indices[sample_index],
                                "label": int(labels[local_index]),
                                "noise_to_signal_ratio": float(ratio),
                                "start_time": start_time,
                                "num_steps": int(num_steps),
                                "final_latent_error_rms": float(
                                    final_latent_error[local_index]
                                ),
                                "final_raw_guided_lpl": float(
                                    final_lpl[local_index]
                                ),
                                "mean_state_path_error_rms": float(
                                    state_path[:, local_index].mean()
                                ),
                                "max_state_path_error_rms": float(
                                    state_path[:, local_index].max()
                                ),
                                "mean_endpoint_error_rms": float(
                                    endpoint_path[:, local_index].mean()
                                ),
                                "last_endpoint_error_rms": float(
                                    endpoint_path[-1, local_index]
                                ),
                                "mean_correction_over_source_guided": float(
                                    correction_path[:, local_index].mean()
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
        "start_time",
        "num_steps",
    ]
    summary = (
        raw.groupby(group_keys, as_index=False, dropna=False)
        .mean(numeric_only=True)
        .sort_values(["branch_update", "branch", "noise_to_signal_ratio", "num_steps"])
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output_dir / "rollout_raw.csv", index=False)
    summary.to_csv(output_dir / "rollout_summary.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for (branch, ratio), frame in summary.groupby(
        ["branch", "noise_to_signal_ratio"]
    ):
        label = f"{branch}, ratio={ratio:g}"
        axes[0].plot(
            frame["num_steps"],
            frame["final_raw_guided_lpl"],
            marker="o",
            label=label,
        )
        axes[1].plot(
            frame["num_steps"],
            frame["final_latent_error_rms"],
            marker="o",
            label=label,
        )
    for axis, title in zip(
        axes,
        ("Final decoder-feature loss", "Final latent RMS error"),
    ):
        axis.set_xscale("log", base=2)
        axis.set_xticks(list(args.rollout_steps))
        axis.set_xticklabels([str(value) for value in args.rollout_steps])
        axis.set_xlabel("recursive endpoint queries")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=8, ncol=2, bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.savefig(output_dir / "rollout_curves.png", dpi=180)
    plt.close(figure)

    manifest = {
        "format_version": 1,
        "scope": "common_adapter_recursive_endpoint_audit",
        "source_checkpoint": str(source_path),
        "source_sha256": source_sha256,
        "source_state_key": args.source_state_key,
        "source_step": source_step,
        "source_epoch": source_epoch,
        "adapter_state_key": args.adapter_state_key,
        "config": str(args.config.resolve()),
        "data_path": str(args.data_path.resolve()),
        "split": "validation",
        "samples": int(args.samples),
        "batch_size": int(args.batch_size),
        "data_indices": data_indices,
        "noise_ratios": list(noise_ratios),
        "rollout_steps": list(args.rollout_steps),
        "time_shift": time_shift,
        "guidance_scale": guidance_scale,
        "guidance_interval": list(guidance_interval),
        "seed": int(args.seed),
        "precision": args.precision,
        "validation_used_for_training": False,
        "interpretation_limit": (
            "The rollout begins from a known data/noise interpolation state so "
            "the clean target remains available. It is a controlled recursive "
            "state-shift audit, not unconditional FID."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
