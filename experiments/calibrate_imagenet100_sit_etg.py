#!/usr/bin/env python3
"""Calibrate and audit Error-Triangulated Guidance on frozen v800 heads.

The deployable weights are estimated without clean targets on unguided v800
rollouts.  A separate teacher-bridge audit uses cached validation latents only
to test whether those weights actually reduce weak-field error; teacher targets
never enter the deployed ETG weights.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchdiffeq import odeint

try:
    from experiments.imagenet100_sit_error_triangulated_guidance import (
        PAIR_NAMES,
        TARGETS,
        full_and_internal_predictions,
        fuse_predictions,
        load_etg_model,
        pairwise_squared_differences,
        predictions_to_velocity,
        regularize_private_variances,
        three_cornered_hat,
    )
    from experiments.train_imagenet100_sit_flow import (
        DEFAULT_CACHE_DIR,
        DEFAULT_OFFICIAL_SIT_REPO,
        LATENT_SHAPE,
        NUM_CLASSES,
        NpyMomentsDataset,
        atomic_json_dump,
        linear_flow_state_target,
        load_official_sit_module,
        sample_sdvae_posterior,
    )
except ModuleNotFoundError:
    from imagenet100_sit_error_triangulated_guidance import (
        PAIR_NAMES,
        TARGETS,
        full_and_internal_predictions,
        fuse_predictions,
        load_etg_model,
        pairwise_squared_differences,
        predictions_to_velocity,
        regularize_private_variances,
        three_cornered_hat,
    )
    from train_imagenet100_sit_flow import (
        DEFAULT_CACHE_DIR,
        DEFAULT_OFFICIAL_SIT_REPO,
        LATENT_SHAPE,
        NUM_CLASSES,
        NpyMomentsDataset,
        atomic_json_dump,
        linear_flow_state_target,
        load_official_sit_module,
        sample_sdvae_posterior,
    )


DEFAULT_HEADS = {
    "velocity": Path(
        "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/"
        "sit-s-2_v800-ema_frozen-internal-v-depth8_seed0/checkpoints/step_00050000.pt"
    ),
    "clean": Path(
        "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/"
        "sit-s-2_v800-ema_frozen-internal-x-depth8_seed0/checkpoints/step_00050000.pt"
    ),
    "epsilon": Path(
        "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/"
        "sit-s-2_v800-ema_frozen-internal-eps-depth8_seed0/checkpoints/step_00050000.pt"
    ),
}
DEFAULT_OUTPUT = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/"
    "error_triangulated_guidance_v800_depth8_v1/calibration.json"
)
DEFAULT_TIME_EDGES = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)


def parse_float_list(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if len(values) < 2:
        raise argparse.ArgumentTypeError("expected at least two time-bin edges")
    if values[0] != 0.0 or values[-1] != 1.0:
        raise argparse.ArgumentTypeError("time-bin edges must start at 0 and end at 1")
    if any(right <= left for left, right in zip(values[:-1], values[1:], strict=True)):
        raise argparse.ArgumentTypeError("time-bin edges must be strictly increasing")
    return values


def parse_int_list(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer seed")
    return values


def time_centers(edges: tuple[float, ...]) -> torch.Tensor:
    return torch.tensor(
        [(left + right) * 0.5 for left, right in zip(edges[:-1], edges[1:], strict=True)],
        dtype=torch.float32,
    )


def empty_pair_accumulator(time_count: int) -> dict[str, torch.Tensor]:
    return {
        name: torch.zeros(time_count, LATENT_SHAPE[0], dtype=torch.float64)
        for name in PAIR_NAMES
    }


def add_pairwise_sums(
    destination: dict[str, torch.Tensor],
    source: Mapping[str, torch.Tensor],
    time_index: int,
) -> None:
    for name in PAIR_NAMES:
        destination[name][time_index] += source[name].double().sum(dim=(0, 2, 3)).cpu()


def strong_velocity(model, labels: torch.Tensor):
    def velocity(time_value: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        times = time_value.expand(len(state))
        output = model(state, times, labels)
        return output[:, : LATENT_SHAPE[0]].float()

    return velocity


@torch.inference_mode()
def collect_rollout_pairwise(
    *,
    model,
    heads,
    internal_depth: int,
    denominator_floor: float,
    device: torch.device,
    centers: torch.Tensor,
    sample_count: int,
    batch_size: int,
    seed: int,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    pair_sums = empty_pair_accumulator(len(centers))
    element_counts = torch.zeros(len(centers), dtype=torch.float64)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    points = torch.cat((torch.zeros(1), centers)).to(device)
    produced = 0
    while produced < sample_count:
        current = min(batch_size, sample_count - produced)
        noise = torch.randn(current, *LATENT_SHAPE, generator=generator, device=device)
        labels = torch.randint(
            0,
            NUM_CLASSES,
            (current,),
            generator=generator,
            device=device,
        )
        trajectory = odeint(
            strong_velocity(model, labels),
            noise.float(),
            points,
            method="dopri5",
            atol=float(atol),
            rtol=float(rtol),
        )[1:]
        for time_index, (time_value, state) in enumerate(zip(centers, trajectory, strict=True)):
            times = time_value.to(device).expand(current)
            _, native = full_and_internal_predictions(
                model,
                heads,
                state,
                times,
                labels,
                internal_depth=internal_depth,
            )
            velocities = predictions_to_velocity(
                native,
                state=state,
                time_value=times,
                denominator_floor=denominator_floor,
            )
            add_pairwise_sums(
                pair_sums,
                pairwise_squared_differences(velocities),
                time_index,
            )
            element_counts[time_index] += current * LATENT_SHAPE[1] * LATENT_SHAPE[2]
        produced += current
        print(
            json.dumps(
                {
                    "event": "rollout_calibration_progress",
                    "seed": seed,
                    "samples": produced,
                    "total": sample_count,
                }
            ),
            flush=True,
        )
    pairwise = {name: values / element_counts[:, None] for name, values in pair_sums.items()}
    return {
        "seed": int(seed),
        "sample_count": int(sample_count),
        "pairwise_channel": pairwise,
    }


def pooled_pairwise(seed_results: list[dict[str, object]]) -> dict[str, torch.Tensor]:
    total = sum(int(row["sample_count"]) for row in seed_results)
    return {
        name: sum(
            row["pairwise_channel"][name] * int(row["sample_count"])
            for row in seed_results
        )
        / total
        for name in PAIR_NAMES
    }


def scalarize_pairwise(
    pairwise_channel: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {name: value.mean(dim=-1, keepdim=True) for name, value in pairwise_channel.items()}


def estimate_payload(
    pairwise: Mapping[str, torch.Tensor],
    *,
    shrinkage: float,
    ridge_fraction: float,
) -> dict[str, torch.Tensor | float]:
    raw = three_cornered_hat(pairwise)
    regularized, weights = regularize_private_variances(
        raw,
        shrinkage=shrinkage,
        ridge_fraction=ridge_fraction,
    )
    negative = (-raw.clamp_max(0.0)).sum()
    absolute = raw.abs().sum().clamp_min(torch.finfo(raw.dtype).eps)
    clipped = raw.clamp_min(0.0)
    return {
        "raw_private_variance": raw,
        "regularized_private_variance": regularized,
        "weights": weights,
        "negative_entry_fraction": float((raw < 0).double().mean().item()),
        "negative_mass_fraction": float((negative / absolute).item()),
        "clipping_relative_l2": float(
            ((clipped - raw).square().sum().sqrt() / raw.square().sum().sqrt().clamp_min(1e-30)).item()
        ),
        "mean_max_weight": float(weights.max(dim=-2).values.mean().item()),
        "near_single_head_fraction": float((weights.max(dim=-2).values > 0.9).double().mean().item()),
    }


def error_cosines(errors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for left, right, name in (
        ("velocity", "clean", "velocity_clean"),
        ("velocity", "epsilon", "velocity_epsilon"),
        ("clean", "epsilon", "clean_epsilon"),
    ):
        a = errors[left].float().flatten(1)
        b = errors[right].float().flatten(1)
        result[name] = (
            (a * b).sum(dim=1)
            / (a.square().sum(dim=1).sqrt() * b.square().sum(dim=1).sqrt()).clamp_min(1e-12)
        )
    return result


@torch.inference_mode()
def teacher_audit(
    *,
    model,
    heads,
    internal_depth: int,
    denominator_floor: float,
    device: torch.device,
    centers: torch.Tensor,
    cache_dir: Path,
    sample_count: int,
    batch_size: int,
    seed: int,
    scalar_weights: torch.Tensor,
    channel_weights: torch.Tensor,
) -> dict[str, object]:
    dataset = NpyMomentsDataset(cache_dir, "validation")
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[
        :sample_count
    ].tolist()
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    condition_names = (*TARGETS, "equal_mean", "etg_scalar", "etg_channel")
    mse_sums = {
        name: torch.zeros(len(centers), dtype=torch.float64) for name in condition_names
    }
    cosine_sums = {
        name: torch.zeros(len(centers), dtype=torch.float64) for name in PAIR_NAMES
    }
    counts = torch.zeros(len(centers), dtype=torch.float64)
    generator = torch.Generator(device=device).manual_seed(seed + 1_000_003)
    for batch_index, (moments, labels) in enumerate(loader):
        moments = moments.to(device, dtype=torch.float32, non_blocking=True)
        labels = labels.to(device, dtype=torch.long, non_blocking=True)
        posterior_noise = torch.randn(
            (len(moments), *LATENT_SHAPE), generator=generator, device=device
        )
        clean = sample_sdvae_posterior(moments, posterior_noise)
        noise = torch.randn(clean.shape, generator=generator, device=device)
        for time_index, time_value in enumerate(centers):
            times = time_value.to(device).expand(len(clean))
            state, target = linear_flow_state_target(clean, noise, times)
            _, native = full_and_internal_predictions(
                model,
                heads,
                state,
                times,
                labels,
                internal_depth=internal_depth,
            )
            velocities = predictions_to_velocity(
                native,
                state=state,
                time_value=times,
                denominator_floor=denominator_floor,
            )
            predictions = {
                **velocities,
                "equal_mean": fuse_predictions(
                    velocities,
                    torch.full((3,), 1.0 / 3.0, device=device),
                ),
                "etg_scalar": fuse_predictions(
                    velocities,
                    scalar_weights[time_index, :, 0].to(device),
                ),
                "etg_channel": fuse_predictions(
                    velocities,
                    channel_weights[time_index].to(device),
                ),
            }
            for name in condition_names:
                mse_sums[name][time_index] += (
                    (predictions[name].float() - target.float()).square().mean(dim=(1, 2, 3)).sum().cpu()
                )
            errors = {name: velocities[name] - target for name in TARGETS}
            for name, values in error_cosines(errors).items():
                cosine_sums[name][time_index] += values.double().sum().cpu()
            counts[time_index] += len(clean)
        print(
            json.dumps(
                {
                    "event": "teacher_audit_progress",
                    "batches": batch_index + 1,
                    "samples": min((batch_index + 1) * batch_size, sample_count),
                    "total": sample_count,
                }
            ),
            flush=True,
        )
    mse = {name: values / counts for name, values in mse_sums.items()}
    cosines = {name: values / counts for name, values in cosine_sums.items()}
    return {
        "sample_count": int(sample_count),
        "seed": int(seed),
        "mse_by_time": mse,
        "mean_mse": {name: float(values.mean().item()) for name, values in mse.items()},
        "error_cosine_by_time": cosines,
        "mean_error_cosine": {
            name: float(values.mean().item()) for name, values in cosines.items()
        },
    }


def to_jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


@torch.inference_mode()
def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if args.allow_tf32 else "highest")

    sit_module, source_metadata = load_official_sit_module(
        args.official_sit_repo.expanduser().resolve(),
        verify_source=args.verify_sit_source,
    )
    checkpoint_paths = {
        "velocity": args.velocity_checkpoint,
        "clean": args.clean_checkpoint,
        "epsilon": args.epsilon_checkpoint,
    }
    model, heads, model_metadata = load_etg_model(
        checkpoint_paths=checkpoint_paths,
        head_weights=args.head_weights,
        sit_module=sit_module,
        source_metadata=source_metadata,
        device=device,
    )
    edges = tuple(args.time_edges)
    centers = time_centers(edges)
    internal_depth = int(model_metadata["source"]["internal_depth"])
    denominator_floor = float(model_metadata["source"]["denominator_floor"])

    seed_results = [
        collect_rollout_pairwise(
            model=model,
            heads=heads,
            internal_depth=internal_depth,
            denominator_floor=denominator_floor,
            device=device,
            centers=centers,
            sample_count=args.calibration_samples_per_seed,
            batch_size=args.batch_size,
            seed=seed,
            atol=args.atol,
            rtol=args.rtol,
        )
        for seed in args.calibration_seeds
    ]
    pairwise_channel = pooled_pairwise(seed_results)
    pairwise_scalar = scalarize_pairwise(pairwise_channel)
    channel_estimate = estimate_payload(
        pairwise_channel,
        shrinkage=args.shrinkage,
        ridge_fraction=args.ridge_fraction,
    )
    scalar_estimate = estimate_payload(
        pairwise_scalar,
        shrinkage=args.shrinkage,
        ridge_fraction=args.ridge_fraction,
    )

    seed_weights = []
    for row in seed_results:
        estimate = estimate_payload(
            scalarize_pairwise(row["pairwise_channel"]),
            shrinkage=args.shrinkage,
            ridge_fraction=args.ridge_fraction,
        )
        seed_weights.append(estimate["weights"])
    weight_stack = torch.stack(seed_weights)
    stability = {
        "mean_absolute_deviation_from_seed_mean": float(
            (weight_stack - weight_stack.mean(dim=0, keepdim=True)).abs().mean().item()
        ),
        "max_absolute_seed_difference": float(
            (weight_stack.max(dim=0).values - weight_stack.min(dim=0).values).max().item()
        ),
        "scalar_weights_by_seed": weight_stack,
    }

    teacher = teacher_audit(
        model=model,
        heads=heads,
        internal_depth=internal_depth,
        denominator_floor=denominator_floor,
        device=device,
        centers=centers,
        cache_dir=args.cache_dir.expanduser().resolve(),
        sample_count=args.teacher_samples,
        batch_size=args.batch_size,
        seed=args.teacher_seed,
        scalar_weights=scalar_estimate["weights"],
        channel_weights=channel_estimate["weights"],
    )

    payload = {
        "format": "eqvae_imagenet100_sit_etg_calibration_v1",
        "scope": "v800 depth-8 three-head ETG; deployable weights use baseline rollout only",
        "model": model_metadata,
        "official_sit": source_metadata,
        "config": {
            "time_bin_edges": list(edges),
            "time_bin_centers": centers.tolist(),
            "calibration_seeds": list(args.calibration_seeds),
            "calibration_samples_per_seed": int(args.calibration_samples_per_seed),
            "teacher_samples": int(args.teacher_samples),
            "teacher_seed": int(args.teacher_seed),
            "batch_size": int(args.batch_size),
            "shrinkage": float(args.shrinkage),
            "ridge_fraction": float(args.ridge_fraction),
            "sampler": {"method": "dopri5", "atol": args.atol, "rtol": args.rtol},
            "endpoint_floor": denominator_floor,
        },
        "rollout_calibration": {
            "pairwise_scalar": pairwise_scalar,
            "pairwise_channel": pairwise_channel,
            "scalar": scalar_estimate,
            "channel": channel_estimate,
            "seed_stability": stability,
        },
        "teacher_audit": teacher,
        "method_boundary": {
            "teacher_targets_used_for_weights": False,
            "teacher_audit_is_diagnostic_only": True,
            "clean_and_epsilon_are_not_exactly_bayes_equivalent_inside_the_0.05_endpoint_floors": True,
        },
    }
    output = args.output.expanduser().resolve()
    atomic_json_dump(to_jsonable(payload), output)
    print(json.dumps({"event": "complete", "output": str(output)}, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--velocity-checkpoint", type=Path, default=DEFAULT_HEADS["velocity"])
    parser.add_argument("--clean-checkpoint", type=Path, default=DEFAULT_HEADS["clean"])
    parser.add_argument("--epsilon-checkpoint", type=Path, default=DEFAULT_HEADS["epsilon"])
    parser.add_argument("--head-weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--official-sit-repo", type=Path, default=DEFAULT_OFFICIAL_SIT_REPO)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--time-edges", type=parse_float_list, default=DEFAULT_TIME_EDGES)
    parser.add_argument("--calibration-seeds", type=parse_int_list, default=(20260829, 20260830))
    parser.add_argument("--calibration-samples-per-seed", type=int, default=256)
    parser.add_argument("--teacher-samples", type=int, default=512)
    parser.add_argument("--teacher-seed", type=int, default=20260831)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--shrinkage", type=float, default=0.05)
    parser.add_argument("--ridge-fraction", type=float, default=0.01)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verify-sit-source", action=argparse.BooleanOptionalAction, default=True)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
