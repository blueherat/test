#!/usr/bin/env python3
"""Audit unnormalized Feynman--Kac weight spread on RAEv2 IG states.

This is a feasibility audit, not an FKC sampler.  It evaluates the exact
full-dimensional running potential implied by the frozen full/depth4 clean
predictions along the existing ordinary-IG probability-flow trajectory.  The
pure-noise heat-time tail is excluded because ``t=1`` maps to infinite heat
variance; all reported cumulative weights therefore begin at the first finite
point of the established shifted Euler grid.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
RAEV2_SRC = ROOT / "external" / "RAEv2" / "src"
for path in (RAEV2_SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

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


PROTOCOL = "raev2_fkc_weight_degeneracy_audit_v1"
DEFAULT_PARTICLE_COUNTS = (4, 8, 16)


def clean_gap_to_heat_score_gap_squared_norm(
    clean_gap: torch.Tensor,
    noise_time: torch.Tensor,
) -> torch.Tensor:
    """Return ``||s_full-s_base||^2`` in RAE heat coordinates.

    RAE uses ``z_t=(1-t)x+t*eps``.  With ``y=z_t/(1-t)`` and
    ``tau=(t/(1-t))^2``, Tweedie's identity gives

    ``s_full-s_base = ((1-t)/t)^2 * (x_full-x_base)``.
    """

    if clean_gap.ndim < 2 or noise_time.shape != (clean_gap.shape[0],):
        raise ValueError("noise_time must have one value per clean-gap sample")
    if torch.any((noise_time <= 0.0) | (noise_time > 1.0)):
        raise ValueError("noise time must lie in (0, 1]")
    factor_squared = ((1.0 - noise_time) / noise_time).pow(4)
    return clean_gap.float().flatten(1).square().sum(dim=1) * factor_squared


def fkc_running_potential(
    score_gap_squared_norm: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    """Exact geometric-mixture reaction potential, without clipping."""

    if beta < 1.0 or not math.isfinite(beta):
        raise ValueError("beta must be finite and at least one")
    if torch.any(score_gap_squared_norm < 0.0):
        raise ValueError("squared score-gap norms must be non-negative")
    return 0.5 * beta * (beta - 1.0) * score_gap_squared_norm


def effective_sample_size(log_weights: torch.Tensor) -> torch.Tensor:
    """Return numerically stable ESS along the final tensor dimension."""

    if log_weights.ndim < 1 or log_weights.shape[-1] < 1:
        raise ValueError("log_weights must have a non-empty particle dimension")
    weights = torch.softmax(log_weights.double(), dim=-1)
    return weights.square().sum(dim=-1).reciprocal()


def grouped_weight_statistics(
    log_weights: torch.Tensor,
) -> dict[str, float]:
    """Summarize per-group importance-weight concentration."""

    if log_weights.ndim != 2:
        raise ValueError("grouped log weights must have shape [groups, particles]")
    weights = torch.softmax(log_weights.double(), dim=1)
    ess = effective_sample_size(log_weights)
    particles = log_weights.shape[1]
    entropy_ess = torch.exp(
        -(weights * weights.clamp_min(torch.finfo(weights.dtype).tiny).log()).sum(1)
    )
    ranges = log_weights.double().amax(1) - log_weights.double().amin(1)
    return {
        "ess_mean": float(ess.mean().item()),
        "ess_min": float(ess.min().item()),
        "ess_fraction_mean": float((ess / particles).mean().item()),
        "ess_fraction_min": float((ess / particles).min().item()),
        "entropy_ess_fraction_mean": float(
            (entropy_ess / particles).mean().item()
        ),
        "max_weight_mean": float(weights.amax(1).mean().item()),
        "max_weight_max": float(weights.amax(1).max().item()),
        "log_weight_range_mean": float(ranges.mean().item()),
        "log_weight_range_max": float(ranges.max().item()),
    }


def heat_variance(noise_time: float) -> float:
    if not 0.0 < noise_time < 1.0:
        raise ValueError("finite heat variance requires noise time in (0,1)")
    return (noise_time / (1.0 - noise_time)) ** 2


def finite_audit_grid(
    native_grid: torch.Tensor,
    *,
    switch_time: float,
) -> torch.Tensor:
    """Keep the native noise-to-data grid and end exactly at the switch."""

    if native_grid.ndim != 1 or len(native_grid) < 2:
        raise ValueError("native grid must be one-dimensional")
    if not 0.0 < switch_time < 1.0:
        raise ValueError("switch time must lie in (0,1)")
    values = [float(native_grid[0].item())]
    values.extend(
        float(value.item()) for value in native_grid[1:] if float(value) > switch_time
    )
    values.append(float(switch_time))
    if any(left <= right for left, right in zip(values, values[1:])):
        raise ValueError("audit grid must decrease strictly")
    return torch.tensor(values, device=native_grid.device, dtype=native_grid.dtype)


def _quantile(values: torch.Tensor, probability: float) -> float:
    return float(torch.quantile(values.detach().double().cpu(), probability).item())


def _potential_statistics(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().double().cpu()
    return {
        "potential_mean": float(values.mean().item()),
        "potential_std": float(values.std(unbiased=False).item()),
        "potential_q10": _quantile(values, 0.10),
        "potential_median": _quantile(values, 0.50),
        "potential_q90": _quantile(values, 0.90),
        "potential_min": float(values.min().item()),
        "potential_max": float(values.max().item()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--particles", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--beta", type=float, default=1.78)
    parser.add_argument("--switch-time", type=float, default=0.5)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--state-key", choices=("ema", "model"), default="ema")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.groups <= 0 or args.particles <= 0 or args.batch_size <= 0:
        raise ValueError("groups, particles, and batch size must be positive")
    particle_counts = tuple(
        count for count in DEFAULT_PARTICLE_COUNTS if count <= args.particles
    )
    if args.particles not in particle_counts:
        particle_counts = (*particle_counts, args.particles)
    if not particle_counts:
        raise ValueError("at least one particle count is required")
    if args.beta <= 1.0 or not math.isfinite(args.beta):
        raise ValueError("this audit requires finite beta > 1")
    if not 0.0 < args.switch_time < 1.0:
        raise ValueError("switch time must lie in (0,1)")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.config.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("RAEv2 config or checkpoint is missing")

    install_raev2_decoder_config_compat()
    config = load_config(config_path)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    allow_tf32 = args.precision != "fp32"
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.cuda.reset_peak_memory_stats(device)

    model = instantiate_from_config(config.stage_2).to(device).eval().requires_grad_(False)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    model.load_state_dict(checkpoint[args.state_key], strict=True)
    checkpoint_step = int(checkpoint.get("step", 0))
    del checkpoint

    shift = math.sqrt(
        (config.misc.time_dist_shift_dim or math.prod(config.misc.latent_size))
        / config.misc.time_dist_shift_base
    )
    native_grid = shifted_time_grid(config.sampler.num_steps, shift, device)
    grid = finite_audit_grid(native_grid, switch_time=args.switch_time)
    total_samples = args.groups * args.particles
    class_ids = torch.linspace(
        0,
        int(config.misc.num_classes) - 1,
        args.groups,
        device=device,
    ).round().long()
    labels = class_ids.repeat_interleave(args.particles)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    state = torch.randn(
        total_samples,
        *config.misc.latent_size,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    cumulative = torch.zeros(
        args.groups, args.particles, device=device, dtype=torch.float64
    )
    previous_potential: torch.Tensor | None = None
    previous_tau: float | None = None
    rows: list[dict[str, float | int]] = []
    t_floor = float(config.transport.t_eps)
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.precision == "bf16"
        else nullcontext()
    )
    started = time.perf_counter()

    with torch.inference_mode(), autocast:
        for index, current_tensor in enumerate(grid):
            current = float(current_tensor.item())
            following = (
                float(grid[index + 1].item()) if index + 1 < len(grid) else None
            )
            potentials: list[torch.Tensor] = []
            next_states: list[torch.Tensor] = []
            for start in range(0, total_samples, args.batch_size):
                stop = min(start + args.batch_size, total_samples)
                state_chunk = state[start:stop]
                label_chunk = labels[start:stop]
                times = torch.full(
                    (stop - start,), current, device=device, dtype=torch.float32
                )
                full_clean, base_clean = model(
                    state_chunk,
                    times,
                    context=label_chunk,
                    attn_mask=None,
                )
                if current < 1.0:
                    score_norm2 = clean_gap_to_heat_score_gap_squared_norm(
                        full_clean - base_clean,
                        times,
                    )
                    potentials.append(
                        fkc_running_potential(score_norm2, beta=args.beta)
                    )
                if following is not None:
                    guided_clean = base_clean + args.beta * (full_clean - base_clean)
                    drift = clean_to_velocity(
                        guided_clean,
                        state_chunk,
                        times,
                        denominator_floor=t_floor,
                    )
                    next_states.append(
                        state_chunk - (current - following) * drift.float()
                    )

            if current < 1.0:
                potential = torch.cat(potentials).reshape(
                    args.groups, args.particles
                ).double()
                tau = heat_variance(current)
                heat_step = 0.0
                increment = torch.zeros_like(potential)
                if previous_potential is not None and previous_tau is not None:
                    heat_step = previous_tau - tau
                    if heat_step <= 0.0:
                        raise RuntimeError("heat variance must decrease during sampling")
                    increment = 0.5 * (previous_potential + potential) * heat_step
                    cumulative.add_(increment)

                row: dict[str, float | int] = {
                    "grid_index": index,
                    "noise_time": current,
                    "heat_variance": tau,
                    "heat_step_from_previous": heat_step,
                    "cumulative_log_weight_mean": float(cumulative.mean().item()),
                    "cumulative_log_weight_std": float(
                        cumulative.std(unbiased=False).item()
                    ),
                    "increment_mean": float(increment.mean().item()),
                    "increment_std": float(increment.std(unbiased=False).item()),
                }
                row.update(_potential_statistics(potential))
                for count in particle_counts:
                    statistics = grouped_weight_statistics(cumulative[:, :count])
                    row.update(
                        {f"k{count}_{name}": value for name, value in statistics.items()}
                    )
                rows.append(row)
                previous_potential = potential
                previous_tau = tau

            if next_states:
                state = torch.cat(next_states)

    elapsed = time.perf_counter() - started
    csv_path = output_dir / "by_time.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    requested_times = (0.95, 0.9, 0.8, 0.7, 0.6, 0.55, args.switch_time)
    snapshots = []
    for requested in requested_times:
        selected = min(rows, key=lambda row: abs(float(row["noise_time"]) - requested))
        snapshots.append({"requested_time": requested, **selected})
    summary = {
        "protocol": PROTOCOL,
        "scope": (
            "Unclipped full-dimensional FKC potential evaluated on the ordinary "
            "depth4-IG probability-flow trajectory; not an SMC sampler."
        ),
        "excluded_tail": "t=1 to first finite shifted-grid point",
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "state_key": args.state_key,
        "beta": args.beta,
        "switch_time": args.switch_time,
        "groups": args.groups,
        "particles_per_group": args.particles,
        "particle_counts": list(particle_counts),
        "class_ids": class_ids.cpu().tolist(),
        "seed": args.seed,
        "precision": args.precision,
        "batch_size": args.batch_size,
        "sampler_steps": int(config.sampler.num_steps),
        "finite_grid_points": len(rows),
        "elapsed_seconds": elapsed,
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "snapshots": snapshots,
        "final": rows[-1],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
