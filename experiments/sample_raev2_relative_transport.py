#!/usr/bin/env python3
"""Paired RAEv2 screen of one finite relative flow-map iteration.

Let ``R`` be the full-head Euler prefix and ``G`` the piecewise-IG Euler
prefix on the guidance-active interval.  The candidate applies
``T = G o R^{-1}`` once more to ``G(z)``.  The run also emits the controls
needed to distinguish this finite map operation from ordinary scale doubling,
switch-space linear extrapolation, and numerical inverse artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import save_image


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "external/RAEv2/src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.raev2_pfr_retiming import clean_to_velocity  # noqa: E402
from experiments.raev2_relative_transport import (  # noqa: E402
    first_index_at_or_below,
    invert_euler_map_fixed_point,
    sample_rms,
)
from experiments.raev2_stage1_compat import (  # noqa: E402
    install_raev2_decoder_config_compat,
)
from experiments.raev2_training_core import file_sha256  # noqa: E402
from experiments.sample_raev2_pfr_retiming import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    generator_sha256,
    load_config,
    shifted_time_grid,
    tensor_sha256,
)
from utils.model_utils import instantiate_from_config  # noqa: E402


PROTOCOL = "raev2_relative_transport_iteration_v1"
BRANCHES = (
    "full",
    "piecewise_ig",
    "local_double",
    "switch_linear",
    "relative_iterate",
    "cycle_null",
    "cycle_guided",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--guidance-scale", type=float, default=1.78)
    parser.add_argument("--switch-time", type=float, default=0.5)
    parser.add_argument("--inverse-tolerance", type=float, default=3e-5)
    parser.add_argument("--inverse-maximum-iterations", type=int, default=24)
    parser.add_argument("--suffix-branch-chunk", type=int, default=2)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    if min(args.sample_count, args.batch_size, args.inverse_maximum_iterations) <= 0:
        parser.error("sample count, batch size, and inverse iterations must be positive")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        parser.error("invalid shard index or shard count")
    if args.num_shards > math.ceil(args.sample_count / args.batch_size):
        parser.error("every shard must own at least one complete RNG batch")
    if not 1 <= args.suffix_branch_chunk <= len(BRANCHES):
        parser.error("suffix branch chunk must be between one and branch count")
    if not math.isfinite(args.guidance_scale) or args.guidance_scale <= 1.0:
        parser.error("guidance scale must be finite and greater than one")
    if not 0.0 < args.switch_time < 1.0:
        parser.error("switch time must lie strictly inside (0, 1)")
    return args


def sample_batches(
    sample_count: int, batch_size: int, shard_index: int, num_shards: int
):
    """Yield every RNG batch and whether this worker owns its model work."""

    for batch_index, start in enumerate(range(0, sample_count, batch_size)):
        stop = min(start + batch_size, sample_count)
        yield start, stop, batch_index % num_shards == shard_index


def _repeat_labels(labels: torch.Tensor, count: int) -> torch.Tensor:
    return torch.cat([labels] * count, dim=0)


def _repeat_scales(
    scales: list[float], batch_size: int, reference: torch.Tensor
) -> torch.Tensor:
    values = torch.tensor(scales, device=reference.device, dtype=torch.float32)
    values = values.repeat_interleave(batch_size)
    return values.view(len(values), *([1] * (reference.ndim - 1)))


def _model_velocities(
    model: torch.nn.Module,
    state: torch.Tensor,
    times: torch.Tensor,
    labels: torch.Tensor,
    *,
    denominator_floor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    full_clean, base_clean = model(
        state, times, context=labels, attn_mask=None
    )
    full_velocity = clean_to_velocity(
        full_clean, state, times, denominator_floor=denominator_floor
    ).float()
    base_velocity = clean_to_velocity(
        base_clean, state, times, denominator_floor=denominator_floor
    ).float()
    return full_velocity, base_velocity


def _integrate_group(
    *,
    model: torch.nn.Module,
    states: list[torch.Tensor],
    labels: torch.Tensor,
    scales: list[float],
    time_grid: list[float],
    denominator_floor: float,
    call_counter: dict[str, int],
    phase: str,
) -> list[torch.Tensor]:
    if len(states) != len(scales) or not states:
        raise ValueError("states and scales must have the same nonzero length")
    batch_size = len(states[0])
    if any(state.shape != states[0].shape for state in states):
        raise ValueError("all grouped states must have identical shapes")
    state = torch.cat(states, dim=0)
    contexts = _repeat_labels(labels, len(states))
    scale_tensor = _repeat_scales(scales, batch_size, state)
    exact_full = scale_tensor == 1.0
    for current, following in zip(time_grid[:-1], time_grid[1:]):
        times = torch.full(
            (len(state),), current, device=state.device, dtype=torch.float32
        )
        full_velocity, base_velocity = _model_velocities(
            model,
            state,
            times,
            contexts,
            denominator_floor=denominator_floor,
        )
        guided = base_velocity + scale_tensor * (full_velocity - base_velocity)
        velocity = torch.where(exact_full, full_velocity, guided)
        state = state + (following - current) * velocity
        call_counter[phase] = call_counter.get(phase, 0) + 1
        call_counter[f"{phase}_sample_evaluations"] = (
            call_counter.get(f"{phase}_sample_evaluations", 0) + len(state)
        )
    return list(state.split(batch_size, dim=0))


def _summarize_tensor(value: torch.Tensor) -> dict[str, float]:
    flat = value.float().flatten(1)
    return {
        "mean": float(flat.mean().item()),
        "rms": float(flat.square().mean().sqrt().item()),
        "sample_rms_mean": float(sample_rms(value).mean().item()),
        "sample_rms_max": float(sample_rms(value).max().item()),
    }


def main() -> None:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)
    install_raev2_decoder_config_compat()
    os.environ.setdefault(
        "DINOV3_CKPT_DIR",
        "/home/zhoushunyu/data/eqvae/models/RAEv2/encoders/dinov3",
    )
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    allow_tf32 = args.precision == "bf16"
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.cuda.reset_peak_memory_stats(device)

    config = load_config(args.config.expanduser().resolve())
    decoder = instantiate_from_config(config.stage_1).to(device).eval().requires_grad_(False)
    del decoder.encoder
    torch.cuda.empty_cache()
    model = instantiate_from_config(config.stage_2).to(device).eval().requires_grad_(False)
    payload = torch.load(
        args.checkpoint.expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    model.load_state_dict(payload["ema"], strict=True)
    checkpoint_step = int(payload.get("step", 0))
    del payload

    shift = math.sqrt(
        (config.misc.time_dist_shift_dim or math.prod(config.misc.latent_size))
        / config.misc.time_dist_shift_base
    )
    grid = shifted_time_grid(config.sampler.num_steps, shift, device)
    grid_cpu = [float(value) for value in grid.cpu().tolist()]
    switch_index = first_index_at_or_below(grid_cpu, args.switch_time)
    prefix_grid = grid_cpu[: switch_index + 1]
    suffix_grid = grid_cpu[switch_index:]
    effective_switch = prefix_grid[-1]
    doubled_scale = 1.0 + 2.0 * (args.guidance_scale - 1.0)
    request = {
        "protocol": PROTOCOL,
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "checkpoint_step": checkpoint_step,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "config_sha256": file_sha256(args.config),
        "state_key": "ema",
        "time_shift": shift,
        "time_grid": grid_cpu,
        "switch_index": switch_index,
        "effective_switch_time": effective_switch,
        "prefix_steps": len(prefix_grid) - 1,
        "suffix_steps": len(suffix_grid) - 1,
        "local_double_scale": doubled_scale,
        "branches": BRANCHES,
        "sharding_rule": (
            "whole original batch_index modulo num_shards; every worker "
            "generates and hashes every original RNG batch"
        ),
        "scientific_free_parameters": 0,
        "source_sha256": {
            Path(__file__).name: file_sha256(Path(__file__)),
            "raev2_relative_transport.py": file_sha256(
                ROOT / "experiments/raev2_relative_transport.py"
            ),
        },
    }
    (output / "request.json").write_text(
        json.dumps(request, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    generator = torch.Generator(device=device).manual_seed(args.seed)
    initial_rng = generator_sha256(generator)
    noise_hash = hashlib.sha256()
    label_hash = hashlib.sha256()
    images: dict[str, list[np.ndarray]] = {name: [] for name in BRANCHES}
    sample_ids: list[np.ndarray] = []
    telemetry_rows: list[dict[str, float | int | bool]] = []
    inverse_aggregates: dict[int, dict[str, float | int]] = {}
    call_counter: dict[str, int] = {}
    started = time.perf_counter()
    denominator_floor = float(config.transport.t_eps)
    cast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if args.precision == "bf16"
        else nullcontext()
    )

    batches = list(
        sample_batches(
            args.sample_count, args.batch_size, args.shard_index, args.num_shards
        )
    )
    local_total = sum(stop - start for start, stop, assigned in batches if assigned)
    local_completed = 0
    with torch.inference_mode(), cast:
        for start, stop, assigned in batches:
            batch_size = stop - start
            noise = torch.randn(
                batch_size,
                *config.misc.latent_size,
                device=device,
                generator=generator,
                dtype=torch.float32,
            )
            labels = torch.arange(start, stop, device=device) % int(config.misc.num_classes)
            noise_hash.update(noise.cpu().contiguous().numpy().tobytes())
            label_hash.update(labels.cpu().contiguous().numpy().tobytes())
            if not assigned:
                continue
            sample_ids.append(np.arange(start, stop, dtype=np.int64))

            reference_switch, guided_switch, doubled_switch = _integrate_group(
                model=model,
                states=[noise, noise, noise],
                labels=labels,
                scales=[1.0, args.guidance_scale, doubled_scale],
                time_grid=prefix_grid,
                denominator_floor=denominator_floor,
                call_counter=call_counter,
                phase="prefix",
            )

            inverse_labels = _repeat_labels(labels, 2)

            def reference_velocity(value: torch.Tensor, current: float) -> torch.Tensor:
                times = torch.full(
                    (len(value),), current, device=value.device, dtype=torch.float32
                )
                full_velocity, _ = _model_velocities(
                    model,
                    value,
                    times,
                    inverse_labels,
                    denominator_floor=denominator_floor,
                )
                call_counter["inverse"] = call_counter.get("inverse", 0) + 1
                call_counter["inverse_sample_evaluations"] = (
                    call_counter.get("inverse_sample_evaluations", 0) + len(value)
                )
                return full_velocity

            inverse = invert_euler_map_fixed_point(
                torch.cat([reference_switch, guided_switch], dim=0),
                prefix_grid,
                reference_velocity,
                tolerance=args.inverse_tolerance,
                maximum_iterations=args.inverse_maximum_iterations,
            )
            if not inverse.converged:
                failed = [step.index for step in inverse.steps if not step.converged]
                raise RuntimeError(
                    "reference-map inverse did not converge at prefix steps "
                    f"{failed[:8]} (maximum residual={inverse.maximum_relative_residual:.3e})"
                )
            cycle_noise, relative_noise = inverse.state.split(batch_size, dim=0)
            for step in inverse.steps:
                aggregate = inverse_aggregates.setdefault(
                    step.index,
                    {
                        "current_time": step.current_time,
                        "following_time": step.following_time,
                        "iterations_sum": 0,
                        "iterations_max": 0,
                        "residual_max": 0.0,
                        "batches": 0,
                    },
                )
                aggregate["iterations_sum"] += step.iterations
                aggregate["iterations_max"] = max(
                    int(aggregate["iterations_max"]), step.iterations
                )
                aggregate["residual_max"] = max(
                    float(aggregate["residual_max"]),
                    step.relative_fixed_point_residual,
                )
                aggregate["batches"] += 1

            null_cycle_switch, guided_cycle_switch, relative_switch = _integrate_group(
                model=model,
                states=[cycle_noise, relative_noise, relative_noise],
                labels=labels,
                scales=[1.0, 1.0, args.guidance_scale],
                time_grid=prefix_grid,
                denominator_floor=denominator_floor,
                call_counter=call_counter,
                phase="reapply",
            )
            switch_linear = 2.0 * guided_switch - reference_switch
            switch_states = {
                "full": reference_switch,
                "piecewise_ig": guided_switch,
                "local_double": doubled_switch,
                "switch_linear": switch_linear,
                "relative_iterate": relative_switch,
                "cycle_null": null_cycle_switch,
                "cycle_guided": guided_cycle_switch,
            }

            endpoint_states: dict[str, torch.Tensor] = {}
            for offset in range(0, len(BRANCHES), args.suffix_branch_chunk):
                names = BRANCHES[offset : offset + args.suffix_branch_chunk]
                results = _integrate_group(
                    model=model,
                    states=[switch_states[name] for name in names],
                    labels=labels,
                    scales=[1.0] * len(names),
                    time_grid=suffix_grid,
                    denominator_floor=denominator_floor,
                    call_counter=call_counter,
                    phase="suffix",
                )
                endpoint_states.update(zip(names, results))

            signal = guided_switch - reference_switch
            second_increment = relative_switch - guided_switch
            telemetry = {
                "batch_start": start,
                "batch_size": batch_size,
                "inverse_maximum_iterations": inverse.maximum_iterations,
                "inverse_maximum_relative_residual": inverse.maximum_relative_residual,
                "signal_rms": float(sample_rms(signal).mean().item()),
                "second_increment_rms": float(sample_rms(second_increment).mean().item()),
                "inverse_noise_shift_rms": float(sample_rms(relative_noise - noise).mean().item()),
                "inverse_noise_shift_to_noise": float(
                    (sample_rms(relative_noise - noise) / sample_rms(noise).clamp_min(1e-8)).mean().item()
                ),
                "null_noise_cycle_rms": float(sample_rms(cycle_noise - noise).mean().item()),
                "null_switch_cycle_rms": float(
                    sample_rms(null_cycle_switch - reference_switch).mean().item()
                ),
                "guided_switch_cycle_rms": float(
                    sample_rms(guided_cycle_switch - guided_switch).mean().item()
                ),
                "guided_cycle_to_signal": float(
                    (
                        sample_rms(guided_cycle_switch - guided_switch)
                        / sample_rms(signal).clamp_min(1e-8)
                    ).mean().item()
                ),
                "guided_cycle_to_second_increment": float(
                    (
                        sample_rms(guided_cycle_switch - guided_switch)
                        / sample_rms(second_increment).clamp_min(1e-8)
                    ).mean().item()
                ),
            }
            telemetry_rows.append(telemetry)

            for name in BRANCHES:
                decoded = decoder.decode(endpoint_states[name]).clamp(0, 1)
                branch_dir = output / name
                branch_dir.mkdir(exist_ok=True)
                if start == 0:
                    save_image(decoded.float().cpu(), branch_dir / "preview.png", nrow=4)
                images[name].append(
                    decoded.mul(255)
                    .permute(0, 2, 3, 1)
                    .to(device="cpu", dtype=torch.uint8)
                    .numpy()
                )

            local_completed += batch_size
            elapsed = time.perf_counter() - started
            progress = {
                "completed": local_completed,
                "total": local_total,
                "global_sample_count": args.sample_count,
                "global_rng_samples_consumed": stop,
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "elapsed_seconds": elapsed,
                "estimated_remaining_seconds": (
                    elapsed * (local_total - local_completed) / local_completed
                ),
            }
            print(json.dumps(progress), flush=True)
            (output / "progress.json").write_text(
                json.dumps(progress, indent=2) + "\n", encoding="utf-8"
            )

    branch_summaries = {}
    for name in BRANCHES:
        archive = output / name / "samples.npz"
        np.savez(archive, np.concatenate(images[name], axis=0))
        branch_summaries[name] = {
            "archive": str(archive),
            "archive_sha256": file_sha256(archive),
            "samples": local_total,
        }
        (output / name / "summary.json").write_text(
            json.dumps(branch_summaries[name], indent=2) + "\n", encoding="utf-8"
        )
    sample_ids_path = output / "sample_ids.npy"
    np.save(sample_ids_path, np.concatenate(sample_ids))

    with (output / "batch_telemetry.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(telemetry_rows[0]))
        writer.writeheader()
        writer.writerows(telemetry_rows)
    inverse_rows = []
    for index in sorted(inverse_aggregates):
        row = inverse_aggregates[index]
        inverse_rows.append(
            {
                "index": index,
                "current_time": row["current_time"],
                "following_time": row["following_time"],
                "iterations_mean": row["iterations_sum"] / row["batches"],
                "iterations_max": row["iterations_max"],
                "relative_residual_max": row["residual_max"],
            }
        )
    with (output / "inverse_steps.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(inverse_rows[0]))
        writer.writeheader()
        writer.writerows(inverse_rows)

    relative_noise_stats = _summarize_tensor(relative_noise)
    summary = {
        "protocol": PROTOCOL,
        "samples": local_total,
        "global_sample_count": args.sample_count,
        "local_sample_count": local_total,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "sample_ids_sha256": file_sha256(sample_ids_path),
        "seed": args.seed,
        "branches": branch_summaries,
        "initial_generator_sha256": initial_rng,
        "final_generator_sha256": generator_sha256(generator),
        "noise_sha256": noise_hash.hexdigest(),
        "labels_sha256": label_hash.hexdigest(),
        "effective_switch_time": effective_switch,
        "call_counter": call_counter,
        "telemetry_mean": {
            key: float(np.mean([float(row[key]) for row in telemetry_rows]))
            for key in telemetry_rows[0]
            if key not in {"batch_start", "batch_size"}
        },
        "last_batch_relative_noise": relative_noise_stats,
        "elapsed_seconds": time.perf_counter() - started,
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "complete": True,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
