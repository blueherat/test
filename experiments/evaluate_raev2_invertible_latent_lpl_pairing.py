"""Evaluate invertible RAEv2 latent adapters on fixed validation pairs.

The source Stage-2 model, RAE encoder/decoder, validation images, labels,
Gaussian noise, and time values are held fixed.  Only the invertible adapter
checkpoint changes.  This makes checkpoint and Flow-vs-LPL comparisons paired.
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
import torch.distributed as dist
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
RAEV2_SRC = ROOT / "external" / "RAEv2" / "src"
for path in (RAEV2_SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from configs.stage2 import Stage2Config  # noqa: E402
from experiments.latent_equiv_adapter import InvertibleLatentAdapter  # noqa: E402
from experiments.rae_lpl_detach_audit import (  # noqa: E402
    decoder_feature_objective_per_sample,
)
from experiments.rae_strict_lpl import (  # noqa: E402
    decoder_feature_pyramid,
    decoder_hidden_indices,
)
from experiments.raev2_common_adapter import (  # noqa: E402
    internal_guidance_prediction,
)
from experiments.raev2_invertible_latent_lpl import (  # noqa: E402
    INVERTIBLE_LATENT_LPL_FORMAT,
    adapter_config,
    cycle_metrics,
    make_reparameterized_path,
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
from experiments.train_raev2_strict_lpl import (  # noqa: E402
    set_seed,
    setup_distributed,
)
from stage2.transport import create_transport  # noqa: E402
from stage2.utils import validate_stage2_config  # noqa: E402
from utils.model_utils import instantiate_from_config  # noqa: E402


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
    parser.add_argument("--adapter", action="append", type=parse_named_path, default=[])
    parser.add_argument(
        "--adapter-state-key",
        choices=("adapter", "adapter_ema"),
        default="adapter",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--noise-ratio", action="append", type=float, dest="noise_ratios")
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument(
        "--prediction-target",
        choices=("full", "guided"),
        default="full",
    )
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--hidden-channels", type=int, default=32)
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


def load_config(path: Path) -> Stage2Config:
    config = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(Stage2Config), OmegaConf.load(path))
    )
    config.post_process()
    validate_stage2_config(config)
    if config.transport.prediction != "x":
        raise ValueError("paired invertible-adapter evaluation requires x-prediction")
    config.prepare_model_params()
    return config


def load_adapters(
    named_paths: list[tuple[str, Path]],
    *,
    channels: int,
    blocks: int,
    hidden_channels: int,
    source_sha256: str,
    source_state_key: str,
    state_key: str,
    device: torch.device,
) -> list[tuple[str, InvertibleLatentAdapter, dict[str, Any]]]:
    names = ["identity", *(name for name, _ in named_paths)]
    if len(names) != len(set(names)):
        raise ValueError("adapter names must be unique and cannot be 'identity'")

    identity = InvertibleLatentAdapter(
        channels=channels,
        hidden_channels=hidden_channels,
        blocks=blocks,
    ).to(device).eval().requires_grad_(False)
    loaded: list[tuple[str, InvertibleLatentAdapter, dict[str, Any]]] = [
        (
            "identity",
            identity,
            {
                "checkpoint_path": None,
                "checkpoint_sha256": None,
                "branch_update": 0,
                "training_objective": "identity",
            },
        )
    ]
    expected_config = adapter_config(identity)
    for name, raw_path in named_paths:
        path = raw_path.resolve()
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("format") != INVERTIBLE_LATENT_LPL_FORMAT:
            raise ValueError(f"{path} is not an invertible latent LPL checkpoint")
        if checkpoint.get("adapter_config") != expected_config:
            raise ValueError(f"{path} has a different adapter architecture")
        metadata = checkpoint.get("invertible_latent_lpl", {})
        if metadata.get("source_sha256") != source_sha256:
            raise ValueError(f"{path} was trained from a different source checkpoint")
        if metadata.get("source_state_key") != source_state_key:
            raise ValueError(f"{path} was trained from a different source state")
        adapter = InvertibleLatentAdapter(**expected_config).to(device)
        adapter.load_state_dict(checkpoint[state_key], strict=True)
        adapter.eval().requires_grad_(False)
        loaded.append(
            (
                name,
                adapter,
                {
                    "checkpoint_path": str(path),
                    "checkpoint_sha256": file_sha256(path),
                    "branch_update": int(metadata["branch_update"]),
                    "training_objective": str(metadata["objective"]),
                },
            )
        )
        del checkpoint
    return loaded


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "branch",
        "branch_update",
        "training_objective",
        "checkpoint_path",
        "checkpoint_sha256",
        "noise_to_signal_ratio",
        "time",
    ]
    means = raw.groupby(keys, as_index=False, dropna=False).mean(numeric_only=True)
    counts = raw.groupby(keys, as_index=False, dropna=False).size()
    stds = (
        raw.groupby(keys, as_index=False, dropna=False)[["flow_loss", "lpl_loss"]]
        .std()
        .rename(columns={"flow_loss": "flow_loss_std", "lpl_loss": "lpl_loss_std"})
    )
    summary = means.merge(counts, on=keys).merge(stds, on=keys)
    summary["flow_loss_sem"] = summary["flow_loss_std"] / summary["size"].pow(0.5)
    summary["lpl_loss_sem"] = summary["lpl_loss_std"] / summary["size"].pow(0.5)
    identity_rows = raw.loc[
        raw["branch"] == "identity",
        ["sample_index", "noise_to_signal_ratio", "flow_loss", "lpl_loss"],
    ].rename(
        columns={
            "flow_loss": "identity_flow_loss",
            "lpl_loss": "identity_lpl_loss",
        }
    )
    paired = raw.merge(
        identity_rows,
        on=["sample_index", "noise_to_signal_ratio"],
        how="left",
        validate="many_to_one",
    )
    paired["flow_delta_vs_identity"] = (
        paired["flow_loss"] - paired["identity_flow_loss"]
    )
    paired["lpl_delta_vs_identity"] = (
        paired["lpl_loss"] - paired["identity_lpl_loss"]
    )
    paired_stats = (
        paired.groupby(keys, as_index=False, dropna=False)
        .agg(
            identity_flow_loss=("identity_flow_loss", "mean"),
            identity_lpl_loss=("identity_lpl_loss", "mean"),
            flow_delta_vs_identity=("flow_delta_vs_identity", "mean"),
            lpl_delta_vs_identity=("lpl_delta_vs_identity", "mean"),
            flow_delta_std=("flow_delta_vs_identity", "std"),
            lpl_delta_std=("lpl_delta_vs_identity", "std"),
        )
    )
    summary = summary.merge(paired_stats, on=keys, validate="one_to_one")
    summary["flow_delta_sem"] = summary["flow_delta_std"] / summary["size"].pow(0.5)
    summary["lpl_delta_sem"] = summary["lpl_delta_std"] / summary["size"].pow(0.5)
    summary["lpl_delta_z"] = summary["lpl_delta_vs_identity"] / summary[
        "lpl_delta_sem"
    ].replace(0.0, float("nan"))
    summary["lpl_relative_change_vs_identity"] = (
        summary["lpl_delta_vs_identity"] / summary["identity_lpl_loss"]
    )
    return summary.sort_values(
        ["noise_to_signal_ratio", "training_objective", "branch_update", "branch"]
    )


def plot_summary(summary: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    for ratio, group in summary.groupby("noise_to_signal_ratio"):
        for objective, style in (("flow", "--"), ("lpl", "-")):
            selected = group[group["training_objective"] == objective]
            if selected.empty:
                continue
            axes[0].plot(
                selected["branch_update"],
                selected["lpl_relative_change_vs_identity"] * 100.0,
                marker="o",
                linestyle=style,
                label=f"{objective}, noise/signal={ratio:g}",
            )
            axes[1].plot(
                selected["branch_update"],
                selected["flow_delta_vs_identity"],
                marker="o",
                linestyle=style,
                label=f"{objective}, noise/signal={ratio:g}",
            )
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set(xlabel="adapter update", ylabel="LPL change vs identity (%)")
    axes[0].set_title("Decoder-feature objective")
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set(xlabel="adapter update", ylabel="Flow loss change vs identity")
    axes[1].set_title("Frozen Stage-2 objective")
    axes[1].legend(loc="best", fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.samples <= 0 or args.batch_size <= 0:
        raise ValueError("--samples and --batch-size must be positive")
    noise_ratios = tuple(args.noise_ratios or (0.5, 1.0, 3.0))
    if any(ratio <= 0 for ratio in noise_ratios):
        raise ValueError("noise ratios must be positive")

    install_raev2_decoder_config_compat()
    os.environ["DINO_CKPT_DIR"] = str(args.dino_ckpt_dir.resolve())
    if args.dino_repo_dir is not None:
        os.environ["DINOV3_REPO_DIR"] = str(args.dino_repo_dir.resolve())
    rank, world_size, device = setup_distributed()
    set_seed(int(args.seed) * world_size + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    config = load_config(args.config)
    source_path = args.source_checkpoint.resolve()
    source_sha256 = file_sha256(source_path) if rank == 0 else ""
    hashes = [source_sha256]
    dist.broadcast_object_list(hashes, src=0)
    source_sha256 = hashes[0]
    checkpoint = torch.load(
        source_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    validate_full_stage2_checkpoint(checkpoint)
    source_model = instantiate_from_config(config.stage_2).to(device).eval()
    source_model.load_state_dict(checkpoint[args.source_state_key], strict=True)
    source_model.requires_grad_(False)
    source_metadata = {
        "source_checkpoint": str(source_path),
        "source_sha256": source_sha256,
        "source_state_key": args.source_state_key,
        "source_step": int(checkpoint["step"]),
        "source_epoch": int(checkpoint["epoch"]),
    }
    del checkpoint

    adapters = load_adapters(
        args.adapter,
        channels=int(source_model.in_channels),
        blocks=int(args.blocks),
        hidden_channels=int(args.hidden_channels),
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
    rae = instantiate_from_config(config.stage_1).to(device).eval()
    rae.requires_grad_(False)
    layer_indices = decoder_hidden_indices(
        len(rae.decoder.decoder_layers),
        fractions=LPL_LAYER_FRACTIONS,
    )
    layer_weights = (1.0,) * len(layer_indices)
    time_dist_shift = math.sqrt(
        (config.misc.time_dist_shift_dim or math.prod(tuple(config.misc.latent_size)))
        / config.misc.time_dist_shift_base
    )
    transport = create_transport(config=config.transport, time_dist_shift=time_dist_shift)
    guidance_interval = (
        float(config.guidance.ig.t_min),
        float(config.guidance.ig.t_max),
    )

    rank_indices = list(range(rank, int(args.samples), world_size))
    rows: list[dict[str, Any]] = []
    for start in range(0, len(rank_indices), int(args.batch_size)):
        batch_indices = rank_indices[start : start + int(args.batch_size)]
        samples = [dataset[index] for index in batch_indices]
        images = torch.stack([sample[0] for sample in samples]).to(device)
        labels = torch.tensor(
            [int(sample[1]) for sample in samples], device=device, dtype=torch.long
        )
        data_indices = [int(sample[2]) for sample in samples]
        with torch.inference_mode():
            clean = rae.encode(images).float()
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
            time = torch.full((len(batch_indices),), time_value, device=device)
            noise_parts = []
            for sample_index in batch_indices:
                generator = torch.Generator(device="cpu").manual_seed(
                    int(args.seed) + 10_000 * sample_index + ratio_index
                )
                noise_parts.append(
                    torch.randn(clean[0].shape, generator=generator, dtype=torch.float32)
                )
            noise = torch.stack(noise_parts).to(device)

            for name, adapter, metadata in adapters:
                with torch.inference_mode(), autocast_context(args.precision):
                    path = make_reparameterized_path(
                        adapter,
                        clean,
                        noise,
                        time,
                        t_eps=float(config.transport.t_eps),
                    )
                    source_full, source_base = source_model(
                        path.noisy_transformed,
                        time,
                        context=labels,
                        attn_mask=None,
                    )
                    flow_map, _ = official_flow_loss_map(
                        transport,
                        (source_full, source_base),
                        target_velocity=path.target_velocity,
                        noisy_latent=path.noisy_transformed,
                        time=time,
                        base_model_coeff=float(config.internal_guidance.base_model_coeff),
                    )
                    if args.prediction_target == "guided":
                        prediction = internal_guidance_prediction(
                            source_full,
                            source_base,
                            time,
                            scale=float(config.guidance.ig.scale),
                            interval=guidance_interval,
                        )
                    else:
                        prediction = source_full
                    predicted_original = adapter.inverse(prediction)
                    predicted_features = tuple(
                        feature.float()
                        for feature in decoder_feature_pyramid(
                            rae,
                            predicted_original,
                            layer_indices=layer_indices,
                        )
                    )
                    lpl_values, _ = decoder_feature_objective_per_sample(
                        "prediction_full",
                        target_features,
                        predicted_features,
                        layer_weights=layer_weights,
                    )
                    cycle = cycle_metrics(adapter, clean)
                flow_values = flow_map.flatten(1).mean(dim=1)
                for local_index, sample_index in enumerate(batch_indices):
                    rows.append(
                        {
                            "branch": name,
                            **metadata,
                            "sample_index": int(sample_index),
                            "data_index": data_indices[local_index],
                            "label": int(labels[local_index]),
                            "noise_to_signal_ratio": float(ratio),
                            "time": time_value,
                            "flow_loss": float(flow_values[local_index]),
                            "lpl_loss": float(lpl_values[local_index]),
                            "cycle_max_abs": float(cycle["cycle_max_abs"]),
                            "cycle_relative_mse": float(cycle["cycle_relative_mse"]),
                            "forward_relative_mse": float(cycle["forward_relative_mse"]),
                        }
                    )

    gathered = [None] * world_size if rank == 0 else None
    dist.gather_object(rows, gathered, dst=0)
    if rank == 0:
        all_rows = [row for rank_rows in gathered for row in rank_rows]
        raw = pd.DataFrame(all_rows)
        summary = summarize(raw)
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        raw.to_csv(output_dir / "fixed_pairing_raw.csv", index=False)
        summary.to_csv(output_dir / "fixed_pairing_summary.csv", index=False)
        plot_summary(summary, output_dir / "fixed_pairing_curves.png")
        manifest = {
            "format_version": 1,
            "scope": "fixed_unseen_validation_pairing_invertible_latent_adapter",
            **source_metadata,
            "config": str(args.config.resolve()),
            "data_path": str(args.data_path.resolve()),
            "split": "validation",
            "samples": int(args.samples),
            "noise_ratios": list(noise_ratios),
            "seed": int(args.seed),
            "precision": args.precision,
            "prediction_target": args.prediction_target,
            "adapter_state_key": args.adapter_state_key,
            "world_size": world_size,
            "validation_used_for_training": False,
            "interpretation_limit": (
                "This is a fixed one-step decoder-feature and Flow diagnostic; "
                "it does not replace endpoint sampling and FID."
            ),
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(summary.to_string(index=False))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
