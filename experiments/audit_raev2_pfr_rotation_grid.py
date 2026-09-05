#!/usr/bin/env python3
"""Measure PFR direction rotation over a grid of information horizons."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.raev2_pfr_retiming import (  # noqa: E402
    bridge_latentized_counterfactual_state,
    clean_to_velocity,
    dataward_future_time,
    evaluate_base_head_only,
)
from experiments.sample_raev2_pfr_retiming import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    load_config,
    shifted_time_grid,
)
from experiments.raev2_stage1_compat import (  # noqa: E402
    install_raev2_decoder_config_compat,
)
from experiments.raev2_training_core import file_sha256  # noqa: E402
from utils.model_utils import instantiate_from_config  # noqa: E402


REQUESTED_TIMES = (0.95, 0.8, 0.6, 0.4, 0.2, 0.14)


def parse_float_list(text: str) -> tuple[float, ...]:
    values = tuple(float(part) for part in text.split(",") if part)
    if not values or any(value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError("values must be positive comma-separated floats")
    return values


def flatten(value: torch.Tensor) -> torch.Tensor:
    return value.float().flatten(1)


def cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = flatten(left)
    right = flatten(right)
    return (left * right).sum(dim=1) / (
        left.norm(dim=1) * right.norm(dim=1)
    ).clamp_min(1e-20)


def rms(value: torch.Tensor) -> torch.Tensor:
    return flatten(value).square().mean(dim=1).sqrt()


def direction_metrics(
    depth_gap: torch.Tensor,
    revision: torch.Tensor,
) -> dict[str, torch.Tensor]:
    depth = flatten(depth_gap)
    change = flatten(revision)
    depth_energy = depth.square().sum(dim=1, keepdim=True).clamp_min(1e-20)
    coefficient = (change * depth).sum(dim=1, keepdim=True) / depth_energy
    orthogonal = change - coefficient * depth
    additive = depth + change
    orthogonal_additive = depth + orthogonal
    additive_cosine = cosine(additive, depth)
    orthogonal_cosine = cosine(orthogonal_additive, depth)
    return {
        "revision_to_depth_rms": rms(revision) / rms(depth_gap).clamp_min(1e-20),
        "revision_depth_cosine": cosine(revision, depth_gap),
        "revision_parallel_coefficient": coefficient[:, 0],
        "revision_orthogonal_energy_fraction": orthogonal.square().sum(dim=1)
        / change.square().sum(dim=1).clamp_min(1e-20),
        "additive_norm_ratio": additive.norm(dim=1)
        / depth.norm(dim=1).clamp_min(1e-20),
        "additive_rotation_degrees": torch.rad2deg(
            torch.acos(additive_cosine.clamp(-1.0, 1.0))
        ),
        "orthogonal_rotation_degrees": torch.rad2deg(
            torch.acos(orthogonal_cosine.clamp(-1.0, 1.0))
        ),
    }


def nearest_indices(grid: torch.Tensor) -> dict[int, float]:
    selected: dict[int, float] = {}
    for requested in REQUESTED_TIMES:
        index = int(torch.argmin((grid[:-1] - requested).abs()).item())
        selected[index] = requested
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--guidance-scale", type=float, default=1.78)
    parser.add_argument(
        "--audit-mode",
        choices=("retiming", "bridge_counterfactual"),
        default="retiming",
    )
    parser.add_argument(
        "--horizons",
        type=parse_float_list,
        default=parse_float_list("0.004,0.008,0.016,0.032,0.064,0.128"),
    )
    parser.add_argument(
        "--horizon-coordinate",
        choices=("raw_time", "log_odds"),
        default="log_odds",
    )
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    args = parser.parse_args()
    if args.samples <= 0 or args.batch_size <= 0:
        raise ValueError("samples and batch size must be positive")

    install_raev2_decoder_config_compat()
    config = load_config(args.config.expanduser().resolve())
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = instantiate_from_config(config.stage_2).to(device).eval().requires_grad_(False)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    model.load_state_dict(checkpoint["ema"], strict=True)
    checkpoint_step = int(checkpoint.get("step", 0))
    del checkpoint

    shift = math.sqrt(
        (config.misc.time_dist_shift_dim or math.prod(config.misc.latent_size))
        / config.misc.time_dist_shift_base
    )
    grid = shifted_time_grid(config.sampler.num_steps, shift, device)
    selected = nearest_indices(grid)
    audit_horizons = (
        args.horizons if args.audit_mode == "retiming" else (0.0,)
    )
    collected: dict[tuple[int, float], dict[str, list[float]]] = {
        (index, horizon): {} for index in selected for horizon in audit_horizons
    }
    generator = torch.Generator(device=device).manual_seed(args.seed)
    t_floor = float(config.transport.t_eps)
    ig_min = float(config.guidance.ig.t_min)
    ig_max = float(config.guidance.ig.t_max)
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.precision == "bf16"
        else nullcontext()
    )

    with torch.inference_mode(), autocast:
        for start in range(0, args.samples, args.batch_size):
            batch = min(args.batch_size, args.samples - start)
            state = torch.randn(
                batch,
                *config.misc.latent_size,
                generator=generator,
                device=device,
            )
            labels = torch.arange(start, start + batch, device=device) % 1000
            kwargs = {"context": labels, "attn_mask": None}
            for index in range(len(grid) - 1):
                current = float(grid[index].item())
                following = float(grid[index + 1].item())
                times = torch.full((batch,), current, device=device)
                strong_clean, weak_clean = model(state, times, **kwargs)
                strong = clean_to_velocity(
                    strong_clean, state, times, denominator_floor=t_floor
                )
                weak = clean_to_velocity(
                    weak_clean, state, times, denominator_floor=t_floor
                )
                if index in selected:
                    if args.audit_mode == "bridge_counterfactual":
                        counterfactual_state = bridge_latentized_counterfactual_state(
                            state,
                            strong_clean,
                            weak_clean,
                            times,
                            guidance_scale=args.guidance_scale,
                        )
                        counterfactual_weak = evaluate_base_head_only(
                            model, counterfactual_state, times, **kwargs
                        )
                        comparisons = ((0.0, counterfactual_weak),)
                        depth_gap = strong_clean - weak_clean
                    else:
                        comparisons = []
                        depth_gap = strong - weak
                        for horizon in audit_horizons:
                            future = dataward_future_time(
                                current,
                                horizon,
                                coordinate=args.horizon_coordinate,
                                minimum_time=ig_min,
                            )
                            future_times = torch.full_like(times, future)
                            future_clean = evaluate_base_head_only(
                                model, state, future_times, **kwargs
                            )
                            future_weak = clean_to_velocity(
                                future_clean,
                                state,
                                future_times,
                                denominator_floor=t_floor,
                            )
                            comparisons.append((horizon, future_weak))
                    for horizon, reference in comparisons:
                        reference_origin = (
                            weak_clean
                            if args.audit_mode == "bridge_counterfactual"
                            else weak
                        )
                        metrics = direction_metrics(
                            depth_gap, reference_origin - reference
                        )
                        bucket = collected[(index, horizon)]
                        for name, values in metrics.items():
                            bucket.setdefault(name, []).extend(values.cpu().tolist())

                drift = (
                    weak + args.guidance_scale * (strong - weak)
                    if ig_min <= current <= ig_max
                    else strong
                )
                state = state - (current - following) * drift

    rows: list[dict[str, float]] = []
    for (index, horizon), bucket in collected.items():
        current = float(grid[index].item())
        future = (
            current
            if args.audit_mode == "bridge_counterfactual"
            else dataward_future_time(
                current,
                horizon,
                coordinate=args.horizon_coordinate,
                minimum_time=ig_min,
            )
        )
        rows.append(
            {
                "requested_time": selected[index],
                "actual_time": current,
                "horizon": horizon,
                "future_time": future,
                **{
                    name: float(sum(values) / len(values))
                    for name, values in bucket.items()
                },
            }
        )
    rows.sort(key=lambda row: (row["horizon"], -row["actual_time"]))
    with (output_dir / "rotation_grid.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "protocol": "raev2_pfr_rotation_grid_v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "samples": args.samples,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "precision": args.precision,
        "guidance_scale": args.guidance_scale,
        "audit_mode": args.audit_mode,
        "horizon_coordinate": args.horizon_coordinate,
        "horizons": list(args.horizons),
        "rows": rows,
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
