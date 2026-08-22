"""Paired held-out audit for native and diagonal-moment SiT checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

try:
    from experiments.imagenet100_sit_moment_residual import (
        diagonal_lmmse_terms,
        diagonal_stats_from_payload,
        moment_residual_to_velocity,
    )
    from experiments.train_imagenet100_sit_flow import (
        DEFAULT_CACHE_DIR,
        DEFAULT_OFFICIAL_SIT_REPO,
        LATENT_SHAPE,
        NUM_CLASSES,
        NpyMomentsDataset,
        SD_VAE_SCALING_FACTOR,
        atomic_json_dump,
        autocast_context,
        linear_flow_state_target,
        load_official_sit_module,
        sample_sdvae_posterior,
    )
except ModuleNotFoundError:
    from imagenet100_sit_moment_residual import (
        diagonal_lmmse_terms,
        diagonal_stats_from_payload,
        moment_residual_to_velocity,
    )
    from train_imagenet100_sit_flow import (
        DEFAULT_CACHE_DIR,
        DEFAULT_OFFICIAL_SIT_REPO,
        LATENT_SHAPE,
        NUM_CLASSES,
        NpyMomentsDataset,
        SD_VAE_SCALING_FACTOR,
        atomic_json_dump,
        autocast_context,
        linear_flow_state_target,
        load_official_sit_module,
        sample_sdvae_posterior,
    )


TIME_EDGES = np.asarray([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0], dtype=np.float64)


def load_model(checkpoint: dict, weights: str, sit_module, device: torch.device):
    config = checkpoint["config"]
    model = sit_module.SiT_models[config["model_name"]](
        input_size=LATENT_SHAPE[-1],
        num_classes=NUM_CLASSES,
        class_dropout_prob=float(config["cfg_dropout"]),
    )
    model.load_state_dict(checkpoint[weights], strict=True)
    return model.to(device).eval().requires_grad_(False)


def paired_summary(native: np.ndarray, residual: np.ndarray) -> dict[str, float]:
    difference = residual - native
    count = len(difference)
    standard_error = float(difference.std(ddof=1) / np.sqrt(count)) if count > 1 else 0.0
    return {
        "count": count,
        "native_mean": float(native.mean()),
        "residual_mean": float(residual.mean()),
        "residual_minus_native": float(difference.mean()),
        "relative_percent": float(100.0 * difference.mean() / native.mean()),
        "paired_standard_error": standard_error,
        "paired_ci95_low": float(difference.mean() - 1.96 * standard_error),
        "paired_ci95_high": float(difference.mean() + 1.96 * standard_error),
        "residual_win_fraction": float(np.mean(difference < 0)),
    }


@torch.inference_mode()
def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    native_checkpoint = torch.load(
        args.native_checkpoint, map_location="cpu", weights_only=False
    )
    residual_checkpoint = torch.load(
        args.residual_checkpoint, map_location="cpu", weights_only=False
    )
    if native_checkpoint["step"] != residual_checkpoint["step"]:
        raise ValueError("paired checkpoints must have the same training step")
    if native_checkpoint["official_sit"] != residual_checkpoint["official_sit"]:
        raise ValueError("paired checkpoints use different SiT sources")
    if (
        native_checkpoint["data_manifest_sha256"]
        != residual_checkpoint["data_manifest_sha256"]
    ):
        raise ValueError("paired checkpoints use different data caches")
    if native_checkpoint["config"].get("velocity_decomposition", "native") != "native":
        raise ValueError("first checkpoint is not the native velocity baseline")
    if (
        residual_checkpoint["config"].get("velocity_decomposition")
        != "diagonal_lmmse"
    ):
        raise ValueError("second checkpoint is not diagonal_lmmse")

    sit_module, source_metadata = load_official_sit_module(
        args.official_sit_repo.expanduser().resolve(),
        verify_source=args.verify_sit_source,
    )
    if source_metadata != native_checkpoint["official_sit"]:
        raise ValueError("local SiT source does not match checkpoints")
    native_model = load_model(native_checkpoint, args.weights, sit_module, device)
    residual_model = load_model(residual_checkpoint, args.weights, sit_module, device)
    stats = diagonal_stats_from_payload(residual_checkpoint["moment_stats"]).to(device)
    variance_floor = float(
        residual_checkpoint["config"].get("moment_variance_floor", 1e-6)
    )

    dataset = NpyMomentsDataset(args.cache_dir.expanduser().resolve(), "validation")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    native_losses: list[np.ndarray] = []
    residual_losses: list[np.ndarray] = []
    analytic_losses: list[np.ndarray] = []
    times: list[np.ndarray] = []
    cursor = 0
    while cursor < min(args.samples, len(dataset)):
        stop = min(cursor + args.batch_size, args.samples, len(dataset))
        records = [dataset[index] for index in range(cursor, stop)]
        moments = torch.stack([record[0] for record in records]).to(device)
        labels = torch.tensor([record[1] for record in records], device=device)
        posterior_noise = torch.randn(
            (len(records), *LATENT_SHAPE), generator=generator, device=device
        )
        data = sample_sdvae_posterior(moments, posterior_noise)
        source_noise = torch.randn(data.shape, generator=generator, device=device)
        time_value = torch.rand(len(data), generator=generator, device=device)
        state, velocity_target = linear_flow_state_target(
            data, source_noise, time_value
        )
        with autocast_context(args.precision):
            native_velocity = native_model(state, time_value, labels).float()
            residual_output = residual_model(state, time_value, labels)
        residual_velocity = moment_residual_to_velocity(
            residual_output,
            state=state,
            time_value=time_value,
            stats=stats,
            variance_floor=variance_floor,
        )
        analytic_velocity, _ = diagonal_lmmse_terms(
            state,
            time_value,
            stats,
            variance_floor=variance_floor,
        )
        native_losses.append(
            (native_velocity - velocity_target).square().flatten(1).mean(1).cpu().numpy()
        )
        residual_losses.append(
            (residual_velocity - velocity_target)
            .square()
            .flatten(1)
            .mean(1)
            .cpu()
            .numpy()
        )
        analytic_losses.append(
            (analytic_velocity - velocity_target)
            .square()
            .flatten(1)
            .mean(1)
            .cpu()
            .numpy()
        )
        times.append(time_value.cpu().numpy())
        cursor = stop

    native_array = np.concatenate(native_losses).astype(np.float64)
    residual_array = np.concatenate(residual_losses).astype(np.float64)
    analytic_array = np.concatenate(analytic_losses).astype(np.float64)
    time_array = np.concatenate(times).astype(np.float64)
    by_time = []
    for lower, upper in zip(TIME_EDGES[:-1], TIME_EDGES[1:]):
        mask = (time_array >= lower) & (
            time_array <= upper if upper == 1.0 else time_array < upper
        )
        if not mask.any():
            continue
        by_time.append(
            {
                "time_low": float(lower),
                "time_high": float(upper),
                **paired_summary(native_array[mask], residual_array[mask]),
                "analytic_mean": float(analytic_array[mask].mean()),
            }
        )
    result = {
        "format": "eqvae_imagenet100_sit_moment_residual_pair_audit_v1",
        "checkpoint_step": int(native_checkpoint["step"]),
        "weights": args.weights,
        "seed": args.seed,
        "precision": args.precision,
        "cache_split": "validation",
        "posterior_sampling": "mean+std*N(0,I)",
        "time_sampling": "uniform",
        "overall": {
            **paired_summary(native_array, residual_array),
            "analytic_mean": float(analytic_array.mean()),
        },
        "by_time": by_time,
    }
    atomic_json_dump(result, args.output.expanduser().resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-checkpoint", type=Path, required=True)
    parser.add_argument("--residual-checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--official-sit-repo", type=Path, default=DEFAULT_OFFICIAL_SIT_REPO)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", choices=("model", "ema"), default="model")
    parser.add_argument("--samples", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=700_000)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--verify-sit-source", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
