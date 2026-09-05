#!/usr/bin/env python3
"""Audit whether the full RAEv2 denoiser can geometrically precondition IG.

For the internal clean-space gap ``d = full - base``, the exact MMSE denoiser
identity implies that ``J_full d`` is proportional to posterior-covariance
action on ``d``.  This script estimates that action with centered finite
differences along an ordinary-IG rollout.  It is diagnostic only and never
feeds the estimate back into sampling.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.raev2_pfr_retiming import clean_to_velocity  # noqa: E402
from experiments.raev2_stage1_compat import (  # noqa: E402
    install_raev2_decoder_config_compat,
)
from experiments.raev2_training_core import (  # noqa: E402
    file_sha256,
    split_internal_guidance_output,
)
from experiments.sample_raev2_pfr_retiming import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    load_config,
    shifted_time_grid,
)
from utils.model_utils import instantiate_from_config  # noqa: E402


REQUESTED_TIMES = (0.95, 0.8, 0.65, 0.5, 0.4, 0.3, 0.2, 0.14)


def sample_mean_product(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape != right.shape or left.ndim < 2:
        raise ValueError("expected matching batched tensors")
    return (left.float() * right.float()).flatten(1).mean(dim=1)


def sample_rms(value: torch.Tensor) -> torch.Tensor:
    if value.ndim < 2:
        raise ValueError("expected a batched tensor")
    return value.float().flatten(1).square().mean(dim=1).sqrt()


def unit_sample_rms(value: torch.Tensor, tiny: float = 1e-12) -> torch.Tensor:
    scale = sample_rms(value).clamp_min(tiny)
    return value.float() / scale.reshape(len(value), *([1] * (value.ndim - 1)))


def centered_directional_derivative(
    plus: torch.Tensor,
    minus: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    if plus.shape != minus.shape:
        raise ValueError("finite-difference outputs must have matching shapes")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    return (plus.float() - minus.float()) / (2.0 * float(epsilon))


def covariance_action_metrics(
    gap: torch.Tensor,
    unit_gap_action: torch.Tensor,
    tiny: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """Summarize ``J_full (gap / rms(gap))`` sample by sample.

    If the action has positive Rayleigh quotient, multiplying it by the gap
    RMS and dividing by ``parallel_coefficient`` gives a direction whose
    projection onto ``gap`` exactly matches ordinary IG.  The reported matched
    ratios quantify the additional orthogonal energy of that fair comparison.
    """

    if gap.shape != unit_gap_action.shape:
        raise ValueError("gap and action must have matching shapes")
    direction = unit_sample_rms(gap, tiny=tiny)
    action_rms = sample_rms(unit_gap_action)
    parallel = sample_mean_product(unit_gap_action, direction)
    cosine = parallel / action_rms.clamp_min(tiny)
    parallel_view = parallel.reshape(
        len(parallel), *([1] * (unit_gap_action.ndim - 1))
    )
    orthogonal_rms = sample_rms(unit_gap_action - parallel_view * direction)
    positive = parallel > tiny
    safe_parallel = parallel.abs().clamp_min(tiny)
    return {
        "gap_rms": sample_rms(gap),
        "action_rms": action_rms,
        "rayleigh": parallel,
        "action_gap_cosine": cosine,
        "orthogonal_rms": orthogonal_rms,
        "positive_rayleigh": positive.float(),
        "matched_total_rms_over_gap": action_rms / safe_parallel,
        "matched_orthogonal_rms_over_gap": orthogonal_rms / safe_parallel,
    }


def nearest_indices(grid: torch.Tensor) -> dict[int, float]:
    selected: dict[int, float] = {}
    for requested in REQUESTED_TIMES:
        index = int(torch.argmin((grid[:-1] - requested).abs()).item())
        selected[index] = requested
    return selected


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _parse_epsilons(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated floats") from error
    if not result or any(not math.isfinite(item) or item <= 0.0 for item in result):
        raise argparse.ArgumentTypeError("finite positive epsilons are required")
    return result


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--guidance-scale", type=float, default=1.78)
    parser.add_argument("--guidance-min-time", type=float, default=0.5)
    parser.add_argument("--guidance-max-time", type=float, default=1.0)
    parser.add_argument("--epsilons", type=_parse_epsilons, default=(1e-2, 3e-3))
    args = parser.parse_args()

    if args.samples <= 0 or args.batch_size <= 0:
        raise ValueError("samples and batch size must be positive")
    if not 0.0 <= args.guidance_min_time <= args.guidance_max_time <= 1.0:
        raise ValueError("guidance window must satisfy 0 <= min <= max <= 1")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    install_raev2_decoder_config_compat()
    config_path = args.config.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    config = load_config(config_path)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

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
    generator = torch.Generator(device=device).manual_seed(args.seed)
    t_floor = float(config.transport.t_eps)
    raw_rows: list[dict[str, object]] = []
    first_noise_sha256 = None

    with torch.inference_mode():
        for start in range(0, args.samples, args.batch_size):
            batch = min(args.batch_size, args.samples - start)
            state = torch.randn(
                batch,
                *config.misc.latent_size,
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
            if first_noise_sha256 is None:
                first_noise_sha256 = tensor_sha256(state)
            labels = torch.arange(start, start + batch, device=device) % 1000
            model_kwargs = {"context": labels, "attn_mask": None}

            for index in range(len(grid) - 1):
                current = float(grid[index].item())
                following = float(grid[index + 1].item())
                times = torch.full((batch,), current, device=device)
                full_clean, base_clean = split_internal_guidance_output(
                    model(state, times, **model_kwargs)
                )
                if base_clean is None:
                    raise RuntimeError("checkpoint does not expose an internal base head")
                gap = full_clean.float() - base_clean.float()

                if index in selected:
                    direction = unit_sample_rms(gap)
                    for epsilon in args.epsilons:
                        perturbed_state = torch.cat(
                            (state + epsilon * direction, state - epsilon * direction),
                            dim=0,
                        )
                        perturbed_times = times.repeat(2)
                        perturbed_kwargs = {
                            "context": labels.repeat(2),
                            "attn_mask": None,
                        }
                        perturbed_full, _ = split_internal_guidance_output(
                            model(perturbed_state, perturbed_times, **perturbed_kwargs)
                        )
                        action = centered_directional_derivative(
                            perturbed_full[:batch],
                            perturbed_full[batch:],
                            epsilon,
                        )
                        metrics = covariance_action_metrics(gap, action)
                        for local_index in range(batch):
                            row: dict[str, object] = {
                                "sample_id": start + local_index,
                                "requested_time": selected[index],
                                "actual_time": current,
                                "epsilon": epsilon,
                            }
                            for name, values in metrics.items():
                                row[name] = float(values[local_index].item())
                            raw_rows.append(row)

                full_velocity = clean_to_velocity(
                    full_clean, state, times, denominator_floor=t_floor
                )
                base_velocity = clean_to_velocity(
                    base_clean, state, times, denominator_floor=t_floor
                )
                scale = (
                    args.guidance_scale
                    if args.guidance_min_time <= current <= args.guidance_max_time
                    else 1.0
                )
                drift = base_velocity + scale * (full_velocity - base_velocity)
                state = state - (current - following) * drift

    if not raw_rows:
        raise RuntimeError("no requested diagnostic time was reached")
    _write_csv(output_dir / "posterior_geometry_raw.csv", raw_rows)

    aggregate_rows: list[dict[str, object]] = []
    numeric_fields = tuple(
        key
        for key in raw_rows[0]
        if key not in {"sample_id", "requested_time", "actual_time", "epsilon"}
    )
    groups: dict[tuple[float, float, float], list[dict[str, object]]] = {}
    for row in raw_rows:
        key = (
            float(row["requested_time"]),
            float(row["actual_time"]),
            float(row["epsilon"]),
        )
        groups.setdefault(key, []).append(row)
    for (requested, actual, epsilon), rows in groups.items():
        aggregate: dict[str, object] = {
            "requested_time": requested,
            "actual_time": actual,
            "epsilon": epsilon,
            "samples": len(rows),
        }
        for field in numeric_fields:
            values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
            aggregate[f"{field}_mean"] = float(values.mean())
            aggregate[f"{field}_std"] = float(values.std())
            aggregate[f"{field}_min"] = float(values.min())
        aggregate_rows.append(aggregate)
    aggregate_rows.sort(
        key=lambda row: (-float(row["actual_time"]), -float(row["epsilon"]))
    )
    _write_csv(output_dir / "posterior_geometry_summary.csv", aggregate_rows)

    summary = {
        "protocol": "raev2_posterior_geometry_audit_v1",
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "state_key": "ema",
        "samples": args.samples,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "epsilons": list(args.epsilons),
        "guidance_scale": args.guidance_scale,
        "guidance_window": [args.guidance_min_time, args.guidance_max_time],
        "first_noise_sha256": first_noise_sha256,
        "sampler_steps": int(config.sampler.num_steps),
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "rows": aggregate_rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
