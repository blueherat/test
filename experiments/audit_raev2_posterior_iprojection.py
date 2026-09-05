#!/usr/bin/env python3
"""Audit finite and tangent posterior I-projection candidates for RAEv2 IG.

This is a diagnostic, not a sampler.  Along an ordinary-IG rollout it checks
whether the full-head Jacobian is sufficiently self-adjoint to be interpreted
as a posterior covariance, and whether a finite input-space exponential tilt
can attain ordinary IG's clean-space progress without a large query shift.
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

from experiments.audit_raev2_posterior_geometry import (  # noqa: E402
    centered_directional_derivative,
    sample_mean_product,
    sample_rms,
    unit_sample_rms,
)
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


DEFAULT_TIMES = (0.95, 0.8, 0.65, 0.5, 0.4, 0.3, 0.2, 0.14)
DEFAULT_QUERY_RMS_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)


def _parse_positive_csv(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated floats") from error
    if not result or any(not math.isfinite(item) or item <= 0.0 for item in result):
        raise argparse.ArgumentTypeError("finite positive values are required")
    return result


def _parse_time_csv(value: str) -> tuple[float, ...]:
    result = _parse_positive_csv(value)
    if any(item >= 1.0 for item in result):
        raise argparse.ArgumentTypeError("diagnostic times must lie strictly inside (0, 1)")
    return result


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def cosine(left: torch.Tensor, right: torch.Tensor, tiny: float = 1e-12) -> torch.Tensor:
    numerator = sample_mean_product(left, right)
    denominator = sample_rms(left) * sample_rms(right)
    return numerator / denominator.clamp_min(tiny)


def jacobian_pair_metrics(
    direction: torch.Tensor,
    jvp: torch.Tensor,
    vjp: torch.Tensor,
    tiny: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """Compare forward, adjoint, and self-adjoint Jacobian actions."""

    if direction.shape != jvp.shape or direction.shape != vjp.shape:
        raise ValueError("direction and Jacobian actions must have matching shapes")
    symmetric = 0.5 * (jvp.float() + vjp.float())
    antisymmetric = 0.5 * (jvp.float() - vjp.float())
    jvp_rms = sample_rms(jvp)
    vjp_rms = sample_rms(vjp)
    symmetric_rms = sample_rms(symmetric)
    return {
        "jvp_direction_cosine": cosine(jvp, direction, tiny=tiny),
        "vjp_direction_cosine": cosine(vjp, direction, tiny=tiny),
        "symmetric_direction_cosine": cosine(symmetric, direction, tiny=tiny),
        "jvp_vjp_cosine": cosine(jvp, vjp, tiny=tiny),
        "jvp_rayleigh": sample_mean_product(jvp, direction),
        "vjp_rayleigh": sample_mean_product(vjp, direction),
        "symmetric_rayleigh": sample_mean_product(symmetric, direction),
        "jvp_rms": jvp_rms,
        "vjp_rms": vjp_rms,
        "symmetric_rms": symmetric_rms,
        "antisymmetric_rms": sample_rms(antisymmetric),
        "antisymmetric_over_symmetric": sample_rms(antisymmetric)
        / symmetric_rms.clamp_min(tiny),
        "jvp_vjp_relative_difference": sample_rms(jvp.float() - vjp.float())
        / (0.5 * (jvp_rms + vjp_rms)).clamp_min(tiny),
    }


def first_crossing_linear(
    coordinates: torch.Tensor,
    values: torch.Tensor,
    targets: torch.Tensor,
    tiny: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Interpolate the first nonnegative target crossing in batched curves.

    ``coordinates`` is one dimensional and includes zero. ``values`` has shape
    ``[batch, len(coordinates)]`` and is expected to start at zero.  Curves are
    not assumed monotone: the first adjacent interval that crosses the target
    is used, while ``monotone_to_crossing`` records whether all preceding
    finite differences were nonnegative.
    """

    if coordinates.ndim != 1 or values.ndim != 2:
        raise ValueError("expected coordinates [grid] and values [batch, grid]")
    if values.shape[1] != len(coordinates) or targets.shape != (len(values),):
        raise ValueError("incompatible curve shapes")
    if len(coordinates) < 2 or not torch.all(coordinates[1:] > coordinates[:-1]):
        raise ValueError("coordinates must be strictly increasing")

    batch = len(values)
    roots = torch.full_like(targets, float("nan"))
    found = torch.zeros(batch, dtype=torch.bool, device=values.device)
    monotone = torch.ones(batch, dtype=torch.bool, device=values.device)
    for index in range(1, len(coordinates)):
        delta = values[:, index] - values[:, index - 1]
        monotone = torch.where(~found, monotone & (delta >= -tiny), monotone)
        newly_found = (~found) & (values[:, index] >= targets)
        denominator = delta.clamp_min(tiny)
        fraction = ((targets - values[:, index - 1]) / denominator).clamp(0.0, 1.0)
        candidate = coordinates[index - 1] + fraction * (
            coordinates[index] - coordinates[index - 1]
        )
        roots = torch.where(newly_found, candidate, roots)
        found = found | newly_found
    return roots, found, monotone


def finite_candidate_metrics(
    gap: torch.Tensor,
    candidate_shift: torch.Tensor,
    target_progress: torch.Tensor,
    ordinary_scale: float,
    tiny: float = 1e-12,
) -> dict[str, torch.Tensor]:
    if not math.isfinite(ordinary_scale) or ordinary_scale < 0.0:
        raise ValueError("ordinary scale must be finite and nonnegative")
    progress = sample_mean_product(gap, candidate_shift)
    gap_rms = sample_rms(gap)
    shift_rms = sample_rms(candidate_shift)
    parallel = progress / gap_rms.square().clamp_min(tiny)
    parallel_view = parallel.reshape(
        len(parallel), *([1] * (candidate_shift.ndim - 1))
    )
    orthogonal = candidate_shift.float() - parallel_view * gap.float()
    return {
        "finite_progress_ratio": progress / target_progress.clamp_min(tiny),
        "finite_shift_over_ordinary": shift_rms
        / (ordinary_scale * gap_rms).clamp_min(tiny),
        "finite_gap_cosine": cosine(candidate_shift, gap, tiny=tiny),
        "finite_parallel_scale": parallel,
        "finite_orthogonal_over_gap": sample_rms(orthogonal)
        / gap_rms.clamp_min(tiny),
    }


def _nearest_indices(grid: torch.Tensor, requested_times: tuple[float, ...]) -> dict[int, float]:
    selected: dict[int, float] = {}
    for requested in requested_times:
        index = int(torch.argmin((grid[:-1] - requested).abs()).item())
        selected[index] = requested
    return selected


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _full_prediction(
    model: torch.nn.Module,
    state: torch.Tensor,
    times: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    full, _ = split_internal_guidance_output(
        model(state, times, context=labels, attn_mask=None)
    )
    return full.float()


def _vjp_full_prediction(
    model: torch.nn.Module,
    state: torch.Tensor,
    times: torch.Tensor,
    labels: torch.Tensor,
    direction: torch.Tensor,
) -> torch.Tensor:
    differentiable_state = state.detach().requires_grad_(True)
    with torch.enable_grad():
        full = _full_prediction(model, differentiable_state, times, labels)
        objective = (full * direction.detach()).flatten(1).sum(dim=1).sum()
        (action,) = torch.autograd.grad(objective, differentiable_state)
    return action.detach().float()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--guidance-scale", type=float, default=1.78)
    parser.add_argument("--guidance-min-time", type=float, default=0.5)
    parser.add_argument("--guidance-max-time", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.003)
    parser.add_argument("--times", type=_parse_time_csv, default=DEFAULT_TIMES)
    parser.add_argument(
        "--query-rms-grid",
        type=_parse_positive_csv,
        default=DEFAULT_QUERY_RMS_GRID,
    )
    args = parser.parse_args()

    if args.samples <= 0 or args.batch_size <= 0:
        raise ValueError("samples and batch size must be positive")
    if not math.isfinite(args.guidance_scale) or args.guidance_scale < 1.0:
        raise ValueError("guidance scale must be finite and at least one")
    if not math.isfinite(args.epsilon) or args.epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    if not 0.0 <= args.guidance_min_time <= args.guidance_max_time <= 1.0:
        raise ValueError("guidance window must satisfy 0 <= min <= max <= 1")
    if any(
        right <= left
        for left, right in zip(args.query_rms_grid, args.query_rms_grid[1:])
    ):
        raise ValueError("query RMS grid must be strictly increasing")

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
    selected = _nearest_indices(grid, args.times)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    t_floor = float(config.transport.t_eps)
    gamma = args.guidance_scale - 1.0
    query_coordinates = torch.tensor(
        (0.0, *args.query_rms_grid), device=device, dtype=torch.float32
    )
    raw_rows: list[dict[str, object]] = []
    first_noise_sha256 = None

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

        for index in range(len(grid) - 1):
            current = float(grid[index].item())
            following = float(grid[index + 1].item())
            times = torch.full((batch,), current, device=device)
            with torch.no_grad():
                full_clean, base_clean = split_internal_guidance_output(
                    model(state, times, context=labels, attn_mask=None)
                )
            if base_clean is None:
                raise RuntimeError("checkpoint does not expose an internal base head")
            full_clean = full_clean.float()
            base_clean = base_clean.float()
            gap = full_clean - base_clean

            if index in selected:
                direction = unit_sample_rms(gap)
                with torch.no_grad():
                    plus_minus_state = torch.cat(
                        (
                            state + args.epsilon * direction,
                            state - args.epsilon * direction,
                        ),
                        dim=0,
                    )
                    plus_minus_full = _full_prediction(
                        model,
                        plus_minus_state,
                        times.repeat(2),
                        labels.repeat(2),
                    )
                    jvp = centered_directional_derivative(
                        plus_minus_full[:batch],
                        plus_minus_full[batch:],
                        args.epsilon,
                    )
                vjp = _vjp_full_prediction(
                    model, state, times, labels, direction
                )
                pair_metrics = jacobian_pair_metrics(direction, jvp, vjp)

                nonzero_coordinates = query_coordinates[1:]
                query_states = torch.cat(
                    [
                        state + coordinate * direction
                        for coordinate in nonzero_coordinates
                    ],
                    dim=0,
                )
                with torch.no_grad():
                    query_full = _full_prediction(
                        model,
                        query_states,
                        times.repeat(len(nonzero_coordinates)),
                        labels.repeat(len(nonzero_coordinates)),
                    )
                query_full = query_full.reshape(
                    len(nonzero_coordinates), batch, *full_clean.shape[1:]
                )
                progress = torch.stack(
                    [
                        sample_mean_product(gap, candidate - full_clean)
                        for candidate in query_full
                    ],
                    dim=1,
                )
                progress = torch.cat(
                    (torch.zeros(batch, 1, device=device), progress), dim=1
                )
                target_progress = gamma * sample_rms(gap).square()
                roots, roots_found, monotone = first_crossing_linear(
                    query_coordinates, progress, target_progress
                )
                safe_roots = torch.nan_to_num(roots, nan=0.0)
                root_state = state + safe_roots.reshape(
                    batch, *([1] * (state.ndim - 1))
                ) * direction
                with torch.no_grad():
                    root_full = _full_prediction(model, root_state, times, labels)
                finite_metrics = finite_candidate_metrics(
                    gap,
                    root_full - full_clean,
                    target_progress,
                    ordinary_scale=gamma,
                )

                for local_index in range(batch):
                    row: dict[str, object] = {
                        "sample_id": start + local_index,
                        "requested_time": selected[index],
                        "actual_time": current,
                        "epsilon": args.epsilon,
                        "gap_rms": float(sample_rms(gap)[local_index].item()),
                        "target_progress": float(target_progress[local_index].item()),
                        "finite_root_found": float(roots_found[local_index].item()),
                        "finite_monotone_to_crossing": float(
                            monotone[local_index].item()
                        ),
                        "finite_query_rms": float(roots[local_index].item()),
                        "max_progress_ratio": float(
                            (
                                progress[local_index, -1]
                                / target_progress[local_index].clamp_min(1e-12)
                            ).item()
                        ),
                    }
                    for name, values in pair_metrics.items():
                        row[name] = float(values[local_index].item())
                    for name, values in finite_metrics.items():
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
            state = (state - (current - following) * drift).detach()

    if not raw_rows:
        raise RuntimeError("no requested diagnostic time was reached")
    _write_csv(output_dir / "posterior_iprojection_raw.csv", raw_rows)

    aggregate_rows: list[dict[str, object]] = []
    numeric_fields = tuple(
        key
        for key in raw_rows[0]
        if key not in {"sample_id", "requested_time", "actual_time", "epsilon"}
    )
    groups: dict[tuple[float, float], list[dict[str, object]]] = {}
    for row in raw_rows:
        key = (float(row["requested_time"]), float(row["actual_time"]))
        groups.setdefault(key, []).append(row)
    for (requested, actual), rows in groups.items():
        aggregate: dict[str, object] = {
            "requested_time": requested,
            "actual_time": actual,
            "samples": len(rows),
        }
        for field in numeric_fields:
            values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
            finite = values[np.isfinite(values)]
            aggregate[f"{field}_mean"] = (
                float(finite.mean()) if len(finite) else float("nan")
            )
            aggregate[f"{field}_std"] = (
                float(finite.std()) if len(finite) else float("nan")
            )
            aggregate[f"{field}_min"] = (
                float(finite.min()) if len(finite) else float("nan")
            )
        aggregate_rows.append(aggregate)
    aggregate_rows.sort(key=lambda row: -float(row["actual_time"]))
    _write_csv(output_dir / "posterior_iprojection_summary.csv", aggregate_rows)

    summary = {
        "protocol": "raev2_posterior_iprojection_audit_v1",
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "state_key": "ema",
        "samples": args.samples,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "guidance_scale": args.guidance_scale,
        "guidance_window": [args.guidance_min_time, args.guidance_max_time],
        "epsilon": args.epsilon,
        "requested_times": list(args.times),
        "query_rms_grid": list(args.query_rms_grid),
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
