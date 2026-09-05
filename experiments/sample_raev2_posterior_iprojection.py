#!/usr/bin/env python3
"""Paired RAEv2 sampling for posterior I-projected internal guidance.

Ordinary IG moves the full clean prediction by ``gamma * (full - base)``.
The I-projected condition instead queries the full denoiser along that gap and
chooses the first query whose output has the same projection onto the gap.
The reflected control keeps that projection and flips only the extra
orthogonal component.  All root-search settings are numerical tolerances, not
generation-quality parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torchvision.utils import save_image


ROOT = Path(__file__).resolve().parents[1]
RAEV2_SRC = ROOT / "external" / "RAEv2" / "src"
for path in (RAEV2_SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.raev2_pfr_retiming import clean_to_velocity  # noqa: E402
from experiments.raev2_posterior_iprojection import (  # noqa: E402
    first_crossing_bracket,
    reflect_same_progress_shift,
    regula_falsi_coordinate,
    same_progress_shift,
    sample_mean_product,
    sample_rms,
    unit_sample_rms,
    update_crossing_bracket,
)
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
    generator_sha256,
    load_config,
    shifted_time_grid,
    tensor_sha256,
)
from utils.model_utils import instantiate_from_config  # noqa: E402


PROTOCOL = "raev2_posterior_iprojection_sampling_v1"
METHODS = {"ordinary", "tangent", "iprojection", "reflected"}
DEFAULT_QUERY_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)


@dataclass(frozen=True)
class SamplingCondition:
    name: str
    method: str

    def validate(self) -> None:
        if not self.name or any(char in self.name for char in "/=,:"):
            raise ValueError("condition name must be a safe path component")
        if self.method not in METHODS:
            raise ValueError(f"unknown method: {self.method}")


DEFAULT_CONDITIONS = (
    SamplingCondition("ordinary_window_s1p78", "ordinary"),
    SamplingCondition("posterior_tangent_s1p78", "tangent"),
    SamplingCondition("posterior_iprojection_s1p78", "iprojection"),
    SamplingCondition("posterior_iprojection_reflected_s1p78", "reflected"),
)


def parse_condition(value: str) -> SamplingCondition:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("condition must be NAME,METHOD")
    condition = SamplingCondition(parts[0], parts[1])
    try:
        condition.validate()
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return condition


def parse_positive_csv(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated floats") from error
    if not result or any(not math.isfinite(item) or item <= 0.0 for item in result):
        raise argparse.ArgumentTypeError("query coordinates must be finite and positive")
    if any(right <= left for left, right in zip(result, result[1:])):
        raise argparse.ArgumentTypeError("query coordinates must be strictly increasing")
    return result


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


def _sample_view(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return value.reshape(len(value), *([1] * (reference.ndim - 1)))


def tangent_guided_clean(
    *,
    model: torch.nn.Module,
    state: torch.Tensor,
    times: torch.Tensor,
    labels: torch.Tensor,
    full_clean: torch.Tensor,
    gap: torch.Tensor,
    gamma: float,
    epsilon: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], int]:
    direction = unit_sample_rms(gap)
    query_state = torch.cat(
        (state + epsilon * direction, state - epsilon * direction), dim=0
    )
    query_full = _full_prediction(
        model, query_state, times.repeat(2), labels.repeat(2)
    )
    action = (query_full[: len(state)] - query_full[len(state) :]) / (2.0 * epsilon)
    shift, valid = same_progress_shift(action, gap, gamma)
    ordinary_rms = float(gamma) * sample_rms(gap)
    return (
        full_clean + shift,
        {
            "root_found": valid.float(),
            "monotone": torch.ones_like(valid, dtype=torch.float32),
            "query_rms": torch.full_like(ordinary_rms, epsilon),
            "progress_ratio": sample_mean_product(gap, shift)
            / (float(gamma) * sample_mean_product(gap, gap)).clamp_min(1e-12),
            "shift_over_ordinary": sample_rms(shift)
            / ordinary_rms.clamp_min(1e-12),
        },
        2,
    )


def iprojected_guided_clean(
    *,
    model: torch.nn.Module,
    state: torch.Tensor,
    times: torch.Tensor,
    labels: torch.Tensor,
    full_clean: torch.Tensor,
    gap: torch.Tensor,
    gamma: float,
    query_grid: tuple[float, ...],
    refinements: int,
    reflect: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], int]:
    batch = len(state)
    direction = unit_sample_rms(gap)
    coordinates = torch.tensor(
        (0.0, *query_grid), device=state.device, dtype=torch.float32
    )
    nonzero = coordinates[1:]
    query_states = torch.cat(
        [state + coordinate * direction for coordinate in nonzero], dim=0
    )
    query_full = _full_prediction(
        model,
        query_states,
        times.repeat(len(nonzero)),
        labels.repeat(len(nonzero)),
    ).reshape(len(nonzero), batch, *full_clean.shape[1:])
    values = torch.stack(
        [sample_mean_product(gap, candidate - full_clean) for candidate in query_full],
        dim=1,
    )
    values = torch.cat((torch.zeros(batch, 1, device=state.device), values), dim=1)
    target = float(gamma) * sample_mean_product(gap, gap)
    bracket = first_crossing_bracket(coordinates, values, target)

    candidate_clean = full_clean + float(gamma) * gap
    candidate_coordinate = torch.zeros_like(target)
    candidate_progress = target.clone()
    full_equivalents = len(nonzero)
    for _ in range(refinements):
        coordinate = regula_falsi_coordinate(bracket, target)
        coordinate = torch.where(bracket.found, coordinate, torch.zeros_like(coordinate))
        queried = _full_prediction(
            model,
            state + _sample_view(coordinate, state) * direction,
            times,
            labels,
        )
        progress = sample_mean_product(gap, queried - full_clean)
        found_view = _sample_view(bracket.found, queried)
        candidate_clean = torch.where(found_view, queried, candidate_clean)
        candidate_coordinate = torch.where(
            bracket.found, coordinate, candidate_coordinate
        )
        candidate_progress = torch.where(bracket.found, progress, candidate_progress)
        bracket = update_crossing_bracket(bracket, coordinate, progress, target)
        full_equivalents += 1

    shift = candidate_clean - full_clean
    if reflect:
        shift = reflect_same_progress_shift(shift, gap)
        candidate_clean = full_clean + shift
    ordinary_rms = float(gamma) * sample_rms(gap)
    diagnostics = {
        "root_found": bracket.found.float(),
        "monotone": bracket.monotone_to_crossing.float(),
        "query_rms": candidate_coordinate,
        "progress_ratio": sample_mean_product(gap, shift)
        / target.clamp_min(1e-12),
        "shift_over_ordinary": sample_rms(shift)
        / ordinary_rms.clamp_min(1e-12),
        "root_residual_ratio": (candidate_progress - target).abs()
        / target.clamp_min(1e-12),
    }
    return candidate_clean, diagnostics, full_equivalents


def _accumulate(
    totals: dict[str, float], diagnostics: dict[str, torch.Tensor]
) -> None:
    count = len(next(iter(diagnostics.values())))
    totals["observations"] = totals.get("observations", 0.0) + count
    for key, value in diagnostics.items():
        finite = torch.nan_to_num(value.float(), nan=0.0, posinf=0.0, neginf=0.0)
        totals[f"{key}_sum"] = totals.get(f"{key}_sum", 0.0) + float(
            finite.sum().item()
        )
        totals[f"{key}_max"] = max(
            totals.get(f"{key}_max", float("-inf")), float(finite.max().item())
        )


def sample_condition(
    *,
    model: torch.nn.Module,
    decoder: torch.nn.Module,
    condition: SamplingCondition,
    config: object,
    time_grid: torch.Tensor,
    global_ids: np.ndarray,
    per_rank_batch: int,
    sampling_seed: int,
    precision: str,
    output_dir: Path,
    rank: int,
    world_size: int,
    guidance_scale: float,
    guidance_min_time: float,
    guidance_max_time: float,
    finite_difference_rms: float,
    query_grid: tuple[float, ...],
    root_refinements: int,
) -> dict[str, object]:
    condition.validate()
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device=time_grid.device)
    generator.manual_seed(int(sampling_seed) * world_size + rank)
    initial_rng = generator_sha256(generator)
    images_local: list[np.ndarray] = []
    preview = None
    first_noise_sha256 = None
    first_label_sha256 = None
    full_forward_equivalents = 0
    numerical_boundary_fallbacks = 0
    diagnostic_totals: dict[str, float] = {}
    started = time.perf_counter()
    gamma = guidance_scale - 1.0
    t_floor = float(config.transport.t_eps)
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if precision == "bf16"
        else nullcontext()
    )

    with torch.inference_mode(), autocast:
        for start in range(0, len(global_ids), per_rank_batch):
            ids = global_ids[start : start + per_rank_batch]
            batch = len(ids)
            state = torch.randn(
                batch,
                *config.misc.latent_size,
                generator=generator,
                device=time_grid.device,
                dtype=torch.float32,
            )
            labels = torch.from_numpy(ids % 1000).to(
                device=time_grid.device, dtype=torch.long
            )
            if first_noise_sha256 is None:
                first_noise_sha256 = tensor_sha256(state)
                first_label_sha256 = tensor_sha256(labels)

            for index in range(len(time_grid) - 1):
                current = float(time_grid[index].item())
                following = float(time_grid[index + 1].item())
                step = current - following
                times = torch.full(
                    (batch,), current, device=state.device, dtype=torch.float32
                )
                full_clean, base_clean = split_internal_guidance_output(
                    model(state, times, context=labels, attn_mask=None)
                )
                full_forward_equivalents += 1
                if base_clean is None:
                    raise RuntimeError("checkpoint does not expose an internal base head")
                full_clean = full_clean.float()
                base_clean = base_clean.float()
                active = guidance_min_time <= current <= guidance_max_time

                if not active:
                    guided_clean = full_clean
                else:
                    gap = full_clean - base_clean
                    if condition.method == "ordinary" or current >= 1.0 - 1e-7:
                        guided_clean = full_clean + gamma * gap
                        if condition.method != "ordinary":
                            numerical_boundary_fallbacks += batch
                    elif condition.method == "tangent":
                        guided_clean, diagnostics, equivalents = tangent_guided_clean(
                            model=model,
                            state=state,
                            times=times,
                            labels=labels,
                            full_clean=full_clean,
                            gap=gap,
                            gamma=gamma,
                            epsilon=finite_difference_rms,
                        )
                        full_forward_equivalents += equivalents
                        _accumulate(diagnostic_totals, diagnostics)
                    else:
                        guided_clean, diagnostics, equivalents = iprojected_guided_clean(
                            model=model,
                            state=state,
                            times=times,
                            labels=labels,
                            full_clean=full_clean,
                            gap=gap,
                            gamma=gamma,
                            query_grid=query_grid,
                            refinements=root_refinements,
                            reflect=condition.method == "reflected",
                        )
                        full_forward_equivalents += equivalents
                        _accumulate(diagnostic_totals, diagnostics)

                drift = clean_to_velocity(
                    guided_clean, state, times, denominator_floor=t_floor
                )
                state = state - step * drift

            decoded = decoder.decode(state).clamp(0, 1)
            if preview is None:
                preview = decoded[: min(16, len(decoded))].float().cpu()
            images_local.append(
                decoded.mul(255)
                .permute(0, 2, 3, 1)
                .to(device="cpu", dtype=torch.uint8)
                .numpy()
            )

    images = np.concatenate(images_local, axis=0)
    np.save(output_dir / f"images-rank{rank:02d}.npy", images)
    np.save(output_dir / f"ids-rank{rank:02d}.npy", global_ids)
    if preview is not None:
        save_image(preview, output_dir / f"preview-rank{rank:02d}.png", nrow=4)
    observations = diagnostic_totals.get("observations", 0.0)
    diagnostic_means = {
        key.removesuffix("_sum") + "_mean": value / observations
        for key, value in diagnostic_totals.items()
        if key.endswith("_sum") and observations > 0
    }
    audit = {
        "protocol": PROTOCOL,
        "condition": asdict(condition),
        "rank": rank,
        "world_size": world_size,
        "sampling_seed": sampling_seed,
        "sample_count": int(len(global_ids)),
        "per_rank_batch": per_rank_batch,
        "precision": precision,
        "initial_generator_sha256": initial_rng,
        "final_generator_sha256": generator_sha256(generator),
        "first_noise_sha256": first_noise_sha256,
        "first_label_sha256": first_label_sha256,
        "full_forward_equivalents_per_batch": full_forward_equivalents,
        "boundary_fallback_samples": numerical_boundary_fallbacks,
        "diagnostic_observations": int(observations),
        **diagnostic_means,
        **{key: value for key, value in diagnostic_totals.items() if key.endswith("_max")},
        "elapsed_seconds": time.perf_counter() - started,
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    (output_dir / f"sampling_audit_rank{rank}.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=1000)
    parser.add_argument("--per-rank-batch", type=int, default=2)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--sampling-seed", type=int, default=20260903)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--state-key", choices=("ema", "model"), default="ema")
    parser.add_argument("--guidance-scale", type=float, default=1.78)
    parser.add_argument("--guidance-min-time", type=float, default=0.5)
    parser.add_argument("--guidance-max-time", type=float, default=1.0)
    parser.add_argument("--finite-difference-rms", type=float, default=0.003)
    parser.add_argument(
        "--query-grid", type=parse_positive_csv, default=DEFAULT_QUERY_GRID
    )
    parser.add_argument("--root-refinements", type=int, default=3)
    parser.add_argument("--condition", action="append", type=parse_condition)
    parser.add_argument(
        "--dino-ckpt-dir",
        type=Path,
        default=Path("/home/zhoushunyu/data/eqvae/models/RAEv2/encoders/dinov3"),
    )
    parser.add_argument("--dino-repo-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    install_raev2_decoder_config_compat()
    args = parse_args()
    conditions = tuple(args.condition or DEFAULT_CONDITIONS)
    for condition in conditions:
        condition.validate()
    if len({condition.name for condition in conditions}) != len(conditions):
        raise ValueError("condition names must be unique")
    if args.sample_count <= 0 or args.per_rank_batch <= 0:
        raise ValueError("sample count and per-rank batch must be positive")
    if args.num_steps is not None and args.num_steps <= 0:
        raise ValueError("number of steps must be positive")
    if not 1.0 <= args.guidance_scale or not math.isfinite(args.guidance_scale):
        raise ValueError("guidance scale must be finite and at least one")
    if not 0.0 <= args.guidance_min_time <= args.guidance_max_time <= 1.0:
        raise ValueError("invalid guidance window")
    if args.finite_difference_rms <= 0.0 or args.root_refinements <= 0:
        raise ValueError("numerical finite difference and refinements must be positive")

    args.config = args.config.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.results_dir = args.results_dir.expanduser().resolve()
    if not args.config.is_file() or not args.checkpoint.is_file():
        raise FileNotFoundError("RAEv2 config or checkpoint is missing")
    os.environ["DINOV3_CKPT_DIR"] = str(args.dino_ckpt_dir.expanduser().resolve())
    if args.dino_repo_dir is not None:
        os.environ["DINOV3_REPO_DIR"] = str(args.dino_repo_dir.expanduser().resolve())

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    allow_tf32 = args.precision != "fp32"
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32

    config = load_config(args.config)
    decoder = instantiate_from_config(config.stage_1).to(device).eval()
    decoder.requires_grad_(False)
    del decoder.encoder
    torch.cuda.empty_cache()
    model = instantiate_from_config(config.stage_2).to(device).eval().requires_grad_(False)
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    model.load_state_dict(checkpoint[args.state_key], strict=True)
    checkpoint_step = int(checkpoint.get("step", 0))
    del checkpoint

    shift = math.sqrt(
        (config.misc.time_dist_shift_dim or math.prod(config.misc.latent_size))
        / config.misc.time_dist_shift_base
    )
    num_steps = int(config.sampler.num_steps if args.num_steps is None else args.num_steps)
    time_grid = shifted_time_grid(num_steps, shift, device)
    global_ids = np.arange(rank, args.sample_count, world_size, dtype=np.int64)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        request = {
            "protocol": PROTOCOL,
            "config": str(args.config),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "checkpoint_step": checkpoint_step,
            "state_key": args.state_key,
            "conditions": [asdict(condition) for condition in conditions],
            "sample_count": args.sample_count,
            "per_rank_batch": args.per_rank_batch,
            "sampling_seed": args.sampling_seed,
            "precision": args.precision,
            "world_size": world_size,
            "sampler_steps": num_steps,
            "time_shift": shift,
            "guidance_scale": args.guidance_scale,
            "guidance_window": [args.guidance_min_time, args.guidance_max_time],
            "finite_difference_rms": args.finite_difference_rms,
            "query_grid": list(args.query_grid),
            "root_refinements": args.root_refinements,
            "transport_prediction": str(config.transport.prediction),
            "transport_t_eps": float(config.transport.t_eps),
        }
        (args.results_dir / "request.json").write_text(
            json.dumps(request, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    dist.barrier()

    for condition in conditions:
        output = args.results_dir / condition.name
        sample_condition(
            model=model,
            decoder=decoder,
            condition=condition,
            config=config,
            time_grid=time_grid,
            global_ids=global_ids,
            per_rank_batch=args.per_rank_batch,
            sampling_seed=args.sampling_seed,
            precision=args.precision,
            output_dir=output,
            rank=rank,
            world_size=world_size,
            guidance_scale=args.guidance_scale,
            guidance_min_time=args.guidance_min_time,
            guidance_max_time=args.guidance_max_time,
            finite_difference_rms=args.finite_difference_rms,
            query_grid=args.query_grid,
            root_refinements=args.root_refinements,
        )
        dist.barrier()
        if rank == 0:
            ids_parts = []
            image_parts = []
            audits = []
            for shard in range(world_size):
                ids_parts.append(np.load(output / f"ids-rank{shard:02d}.npy"))
                image_parts.append(np.load(output / f"images-rank{shard:02d}.npy"))
                audits.append(
                    json.loads(
                        (output / f"sampling_audit_rank{shard}.json").read_text(
                            encoding="utf-8"
                        )
                    )
                )
            ids = np.concatenate(ids_parts)
            images = np.concatenate(image_parts)
            order = np.argsort(ids)
            ids = ids[order]
            images = images[order]
            if not np.array_equal(ids, np.arange(args.sample_count)):
                raise RuntimeError("distributed sample IDs are incomplete or duplicated")
            archive = output / "samples.npz"
            np.savez(archive, images)
            summary = {
                "protocol": PROTOCOL,
                "condition": asdict(condition),
                "samples": int(len(images)),
                "archive": str(archive),
                "archive_sha256": file_sha256(archive),
                "checkpoint_step": checkpoint_step,
                "state_key": args.state_key,
                "max_rank_elapsed_seconds": max(
                    float(audit["elapsed_seconds"]) for audit in audits
                ),
                "max_memory_allocated_bytes": max(
                    int(audit["max_memory_allocated_bytes"]) for audit in audits
                ),
                "root_found_mean": float(
                    np.mean([audit.get("root_found_mean", 1.0) for audit in audits])
                ),
                "root_residual_ratio_mean": float(
                    np.mean(
                        [audit.get("root_residual_ratio_mean", 0.0) for audit in audits]
                    )
                ),
            }
            (output / "sampling_summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            for shard in range(world_size):
                (output / f"ids-rank{shard:02d}.npy").unlink()
                (output / f"images-rank{shard:02d}.npy").unlink()
            print(json.dumps(summary, ensure_ascii=False), flush=True)
        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
