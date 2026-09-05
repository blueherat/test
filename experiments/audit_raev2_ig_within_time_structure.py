#!/usr/bin/env python3
"""Audit sample- and token-level structure of the RAEv2 internal gap.

The audit follows one ordinary internal-guidance rollout and records whether
``full - base`` contains meaningful within-time variation.  It is diagnostic
only: no statistic computed here is fed back into sampling.
"""

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

from experiments.raev2_pfr_retiming import clean_to_velocity  # noqa: E402
from experiments.raev2_stage1_compat import (  # noqa: E402
    install_raev2_decoder_config_compat,
)
from experiments.raev2_training_core import file_sha256  # noqa: E402
from experiments.sample_raev2_pfr_retiming import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    load_config,
    shifted_time_grid,
)
from utils.model_utils import instantiate_from_config  # noqa: E402


REQUESTED_TIMES = (0.95, 0.8, 0.65, 0.5, 0.4, 0.3, 0.2, 0.14)


def as_tokens(value: torch.Tensor) -> torch.Tensor:
    """Return ``[batch, spatial_tokens, channels]`` for BCHW tensors."""

    if value.ndim != 4:
        raise ValueError(f"expected BCHW tensor, got shape {tuple(value.shape)}")
    return value.float().permute(0, 2, 3, 1).flatten(1, 2)


def token_rms(value: torch.Tensor) -> torch.Tensor:
    return as_tokens(value).square().mean(dim=-1).sqrt()


def token_cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_tokens = as_tokens(left)
    right_tokens = as_tokens(right)
    numerator = (left_tokens * right_tokens).sum(dim=-1)
    denominator = left_tokens.norm(dim=-1) * right_tokens.norm(dim=-1)
    return numerator / denominator.clamp_min(1e-12)


def sample_rms(value: torch.Tensor) -> torch.Tensor:
    return value.float().flatten(1).square().mean(dim=1).sqrt()


def sample_cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_flat = left.float().flatten(1)
    right_flat = right.float().flatten(1)
    numerator = (left_flat * right_flat).sum(dim=1)
    denominator = left_flat.norm(dim=1) * right_flat.norm(dim=1)
    return numerator / denominator.clamp_min(1e-12)


def nearest_indices(grid: torch.Tensor) -> dict[int, float]:
    selected: dict[int, float] = {}
    for requested in REQUESTED_TIMES:
        index = int(torch.argmin((grid[:-1] - requested).abs()).item())
        selected[index] = requested
    return selected


def _quantile(values: torch.Tensor, probability: float) -> float:
    return float(torch.quantile(values.double(), probability).item())


def summarize_values(prefix: str, values: torch.Tensor) -> dict[str, float]:
    flat = values.detach().double().flatten().cpu()
    return {
        f"{prefix}_mean": float(flat.mean().item()),
        f"{prefix}_std": float(flat.std(unbiased=False).item()),
        f"{prefix}_q10": _quantile(flat, 0.10),
        f"{prefix}_median": _quantile(flat, 0.50),
        f"{prefix}_q90": _quantile(flat, 0.90),
    }


def spatial_cv(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 2:
        raise ValueError("spatial_cv expects [batch, tokens]")
    return values.std(dim=1, unbiased=False) / values.mean(dim=1).clamp_min(1e-12)


def pearson_per_sample(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("pearson_per_sample expects matching [batch, tokens] tensors")
    left_centered = left - left.mean(dim=1, keepdim=True)
    right_centered = right - right.mean(dim=1, keepdim=True)
    numerator = (left_centered * right_centered).sum(dim=1)
    denominator = left_centered.norm(dim=1) * right_centered.norm(dim=1)
    return numerator / denominator.clamp_min(1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--guidance-scale", type=float, default=2.3)
    parser.add_argument("--guidance-min-time", type=float, default=0.3)
    parser.add_argument("--guidance-max-time", type=float, default=1.0)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    args = parser.parse_args()
    if args.samples <= 0 or args.batch_size <= 0:
        raise ValueError("samples and batch size must be positive")
    if not 0.0 <= args.guidance_min_time <= args.guidance_max_time <= 1.0:
        raise ValueError("guidance window must satisfy 0 <= min <= max <= 1")
    if args.guidance_scale < 0.0 or not math.isfinite(args.guidance_scale):
        raise ValueError("guidance scale must be finite and non-negative")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    install_raev2_decoder_config_compat()
    config = load_config(args.config.expanduser().resolve())
    checkpoint_path = args.checkpoint.expanduser().resolve()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    allow_tf32 = args.precision != "fp32"
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32

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
    buckets: dict[int, dict[str, list[torch.Tensor]]] = {
        index: {} for index in selected
    }
    generator = torch.Generator(device=device).manual_seed(args.seed)
    t_floor = float(config.transport.t_eps)
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
            previous_gap: torch.Tensor | None = None
            previous_selected_gap: torch.Tensor | None = None
            for index in range(len(grid) - 1):
                current = float(grid[index].item())
                following = float(grid[index + 1].item())
                times = torch.full((batch,), current, device=device)
                full_clean, base_clean = model(state, times, **kwargs)
                full_velocity = clean_to_velocity(
                    full_clean, state, times, denominator_floor=t_floor
                )
                base_velocity = clean_to_velocity(
                    base_clean, state, times, denominator_floor=t_floor
                )
                gap_clean = full_clean - base_clean

                if index in selected:
                    gap_token_rms = token_rms(gap_clean)
                    move_token_rms = token_rms(full_clean - state)
                    relative_token_gap = gap_token_rms / move_token_rms.clamp_min(1e-12)
                    gap_move_cosine = token_cosine(gap_clean, full_clean - state)
                    local_persistence = (
                        torch.ones_like(gap_token_rms)
                        if previous_gap is None
                        else token_cosine(gap_clean, previous_gap)
                    )
                    interval_persistence = (
                        torch.ones_like(gap_token_rms)
                        if previous_selected_gap is None
                        else token_cosine(gap_clean, previous_selected_gap)
                    )
                    metrics: dict[str, torch.Tensor] = {
                        "sample_gap_rms": sample_rms(gap_clean),
                        "sample_gap_to_move_ratio": sample_rms(gap_clean)
                        / sample_rms(full_clean - state).clamp_min(1e-12),
                        "sample_gap_move_cosine": sample_cosine(
                            gap_clean, full_clean - state
                        ),
                        "token_gap_rms": gap_token_rms,
                        "token_gap_to_move_ratio": relative_token_gap,
                        "token_gap_move_cosine": gap_move_cosine,
                        "token_local_persistence": local_persistence,
                        "token_interval_persistence": interval_persistence,
                        "sample_token_gap_cv": spatial_cv(gap_token_rms),
                        "sample_token_relative_gap_cv": spatial_cv(relative_token_gap),
                        "sample_persistence_gap_correlation": pearson_per_sample(
                            local_persistence, gap_token_rms
                        ),
                        "sample_persistence_relative_gap_correlation": pearson_per_sample(
                            local_persistence, relative_token_gap
                        ),
                    }
                    bucket = buckets[index]
                    for name, value in metrics.items():
                        bucket.setdefault(name, []).append(value.detach().cpu())
                    previous_selected_gap = gap_clean.detach().clone()

                scale = (
                    args.guidance_scale
                    if args.guidance_min_time <= current <= args.guidance_max_time
                    else 1.0
                )
                drift = base_velocity + scale * (full_velocity - base_velocity)
                state = state - (current - following) * drift
                previous_gap = gap_clean.detach().clone()

    rows: list[dict[str, float]] = []
    for index, requested in selected.items():
        row: dict[str, float] = {
            "requested_time": requested,
            "actual_time": float(grid[index].item()),
        }
        for name, chunks in buckets[index].items():
            values = torch.cat(chunks)
            row.update(summarize_values(name, values))
            if name.startswith("token_"):
                row[f"{name}_negative_fraction"] = float((values < 0).float().mean())
        rows.append(row)
    rows.sort(key=lambda row: row["actual_time"], reverse=True)

    csv_path = output_dir / "within_time_structure.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "protocol": "raev2_ig_within_time_structure_v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "state_key": "ema",
        "samples": args.samples,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "precision": args.precision,
        "guidance_scale": args.guidance_scale,
        "guidance_window": [args.guidance_min_time, args.guidance_max_time],
        "sampler_steps": int(config.sampler.num_steps),
        "rows": rows,
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
