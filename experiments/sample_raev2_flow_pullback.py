#!/usr/bin/env python3
"""Paired early-time RAEv2 guidance with a frozen future gap or its pullback.

The forecast follows the full head, independently of guidance. Both candidate
directions match the current depth gap's per-sample Euclidean norm. Sampling
uses the radius screen's noise call shapes, model layout, and FP32 clean mix.
Forecast and input-VJP costs are recorded separately, without equal-compute
claims. The outer context is no_grad; only a pullback locally enables a graph.
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
from collections import Counter
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torchvision.utils import save_image


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "external/RAEv2/src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.raev2_flow_pullback import (  # noqa: E402
    flow_pullback_direction,
    full_euler_flow,
    normalize_like,
)
from experiments.raev2_pfr_retiming import clean_to_velocity  # noqa: E402
from experiments.raev2_stage1_compat import install_raev2_decoder_config_compat  # noqa: E402
from experiments.raev2_training_core import file_sha256  # noqa: E402
from experiments.sample_raev2_pfr_retiming import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    load_config,
    shifted_time_grid,
)
from utils.model_utils import instantiate_from_config  # noqa: E402


PROTOCOL = "raev2_flow_pullback_sampling_v1"


def sample_batches(sample_count: int, batch_size: int, shard_index: int = 0, num_shards: int = 1):
    """Yield every original RNG batch and whether this worker owns its model work.

    Callers must generate and hash noise before testing ``assigned``. Sharding
    changes only which complete batches are evaluated, never RNG call shapes.
    """

    if min(sample_count, batch_size, num_shards) <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError("invalid sample, batch or shard specification")
    if num_shards > math.ceil(sample_count / batch_size):
        raise ValueError("every shard must own at least one complete RNG batch")
    for batch_index, start in enumerate(range(0, sample_count, batch_size)):
        yield start, min(start + batch_size, sample_count), batch_index % num_shards == shard_index


class CountedModel(nn.Module):
    """Count actual model-forward entries, including checkpoint recomputation.

    Checkpoint recomputation can stop before the complete forward returns.
    Counts are therefore invocations, not claims of full-forward FLOPs.
    input_vjp_calls counts autograd VJP requests, separately from forwards.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.phase = "solver"
        self.forward_invocations: Counter[str] = Counter()
        self.forward_sample_invocations: Counter[str] = Counter()
        self.input_vjp_calls = 0
        self.input_vjp_samples = 0
        self.pullback_primal_forward_invocations = 0
        self.pullback_checkpoint_recompute_invocations = 0

    def forward(self, state: Tensor, times: Tensor, **kwargs):
        self.forward_invocations[self.phase] += 1
        self.forward_sample_invocations[self.phase] += len(state)
        return self.model(state, times, **kwargs)

    @contextmanager
    def counting_phase(self, phase: str):
        previous = self.phase
        self.phase = phase
        try:
            yield
        finally:
            self.phase = previous

    def summary(self) -> dict:
        return {
            "forward_invocations_by_phase": dict(self.forward_invocations),
            "forward_sample_invocations_by_phase": dict(self.forward_sample_invocations),
            "total_model_forward_invocations": sum(self.forward_invocations.values()),
            "total_model_forward_sample_invocations": sum(self.forward_sample_invocations.values()),
            "input_vjp_calls": self.input_vjp_calls,
            "input_vjp_samples": self.input_vjp_samples,
            "pullback_primal_forward_invocations": self.pullback_primal_forward_invocations,
            "pullback_checkpoint_recompute_forward_invocations": self.pullback_checkpoint_recompute_invocations,
            "cost_unit_note": "Actual forward entries and input-VJP requests; checkpoint recomputation may stop early. These are not equal FLOP units.",
        }


def _norm(value: Tensor) -> Tensor:
    return value.float().flatten(1).norm(dim=1)


def _cosine(left: Tensor, right: Tensor) -> Tensor:
    numerator = (left.float().flatten(1) * right.float().flatten(1)).sum(1)
    return (numerator / (_norm(left) * _norm(right)).clamp_min(1e-12)).clamp(-1, 1)


def select_direction(
    model: CountedModel,
    state: Tensor,
    labels: Tensor,
    current_gap: Tensor,
    *,
    mode: str,
    start_time: float,
    horizon: float,
    substeps: int,
    denominator_floor: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Select a direction at one already-authorized intervention step.

    The raw branch performs no VJP. With frozen model parameters and the
    outer no_grad state, full_euler_flow builds no graph in that branch even
    though its implementation locally enables gradients.
    """

    if mode == "ordinary":
        return current_gap, {}
    future = max(0.0, start_time - horizon)
    if mode == "raw_future":
        with model.counting_phase("raw_forecast"):
            endpoint = full_euler_flow(
                model, state.detach(), labels, start_time, future, substeps,
                denominator_floor=denominator_floor, checkpoint_forward=False,
            )
        with torch.no_grad(), model.counting_phase("raw_future_query"):
            future_times = torch.full(
                (len(state),), future, device=state.device, dtype=torch.float32
            )
            future_full, future_base = model(
                endpoint.detach(), future_times, context=labels, attn_mask=None
            )
            future_gap = (future_full.float() - future_base.float()).detach()
            direction = normalize_like(future_gap, current_gap)
        telemetry = {
            "future_gap_rms": _norm(future_gap) / math.sqrt(current_gap[0].numel()),
            "future_current_cosine": _cosine(future_gap, current_gap),
            "future_fallback": _norm(future_gap) <= 1e-12,
        }
        return direction.detach(), telemetry
    if mode != "pullback":
        raise ValueError(f"unknown flow-guidance mode: {mode}")
    before = model.forward_invocations["pullback"]
    with model.counting_phase("pullback"):
        result = flow_pullback_direction(
            model, state, labels, current_gap, start_time, future, substeps,
            denominator_floor=denominator_floor, checkpoint_forward=True,
        )
    model.input_vjp_calls += 1
    model.input_vjp_samples += len(state)
    primal_calls = (substeps if future < start_time else 0) + 1
    observed_calls = model.forward_invocations["pullback"] - before
    if observed_calls < primal_calls:
        raise RuntimeError("observed pullback forward count is below its primal count")
    model.pullback_primal_forward_invocations += primal_calls
    model.pullback_checkpoint_recompute_invocations += observed_calls - primal_calls
    telemetry = {
        key: value.detach()
        for key, value in result.telemetry.items()
        if isinstance(value, Tensor)
    }
    return result.direction.float(), telemetry


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("ordinary", "raw_future", "pullback"), required=True)
    parser.add_argument("--sample-count", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--seed", type=int, default=202609051)
    parser.add_argument("--guidance-scale", type=float, default=1.78)
    parser.add_argument("--guidance-min-time", type=float, default=0.1)
    parser.add_argument("--guidance-max-time", type=float, default=1.0)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--calibration-steps", type=int, default=20)
    parser.add_argument("--horizon", type=float, default=1 / 32)
    parser.add_argument("--substeps", "--forecast-substeps", type=int, default=4)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    if min(args.sample_count, args.batch_size, args.num_steps, args.substeps) <= 0:
        parser.error("sample count, batch size, steps and substeps must be positive")
    if not 0 <= args.calibration_steps <= args.num_steps:
        parser.error("calibration steps must be between zero and num-steps")
    if not 0 <= args.guidance_min_time <= args.guidance_max_time <= 1:
        parser.error("invalid guidance interval")
    if not math.isfinite(args.guidance_scale) or args.guidance_scale < 1:
        parser.error("guidance scale must be finite and >= 1")
    if not math.isfinite(args.horizon) or not 0 < args.horizon <= 1:
        parser.error("raw-time forecast horizon must be finite and in (0, 1]")
    try:
        list(sample_batches(args.sample_count, args.batch_size, args.shard_index, args.num_shards))
    except ValueError as error:
        parser.error(str(error))
    args.config = args.config.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    return args


def main():
    args = parse_args()
    out = args.output_dir.expanduser().resolve()
    if (out / "request.json").exists():
        raise FileExistsError(f"refusing to overwrite an existing run: {out}")
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("DINOV3_CKPT_DIR", "/home/zhoushunyu/data/eqvae/models/RAEv2/encoders/dinov3")
    install_raev2_decoder_config_compat()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    tf32 = args.precision == "bf16"
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    config = load_config(args.config)
    decoder = instantiate_from_config(config.stage_1).to(device).eval().requires_grad_(False)
    del decoder.encoder
    torch.cuda.empty_cache()
    base_model = instantiate_from_config(config.stage_2).to(device).eval().requires_grad_(False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    base_model.load_state_dict(checkpoint["ema"], strict=True)
    checkpoint_step = int(checkpoint.get("step", 0))
    del checkpoint
    model = CountedModel(base_model).eval()
    shift = math.sqrt((config.misc.time_dist_shift_dim or math.prod(config.misc.latent_size)) / config.misc.time_dist_shift_base)
    grid = shifted_time_grid(args.num_steps, shift, device)
    grid_cpu = grid.cpu().tolist()
    denominator_floor = float(config.transport.t_eps)
    batches = list(sample_batches(args.sample_count, args.batch_size, args.shard_index, args.num_shards))
    local_sample_count = sum(stop - start for start, stop, assigned in batches if assigned)
    decoder_artifacts = {}
    for name in ("pretrained_decoder_path", "normalization_stat_path"):
        raw_path = config.stage_1.params.get(name)
        if raw_path:
            artifact = Path(raw_path).expanduser().resolve()
            decoder_artifacts[name] = {"path": str(artifact), "sha256": file_sha256(artifact)}
    request = {
        "protocol": PROTOCOL,
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "global_sample_count": args.sample_count, "local_sample_count": local_sample_count,
        "sharding_rule": "whole original batch_index modulo num_shards; all workers generate/hash every original RNG batch",
        "checkpoint_step": checkpoint_step, "checkpoint_sha256": file_sha256(args.checkpoint),
        "decoder_artifacts": decoder_artifacts,
        "config_sha256": file_sha256(args.config), "state_key": "ema", "world_size": 1,
        "guidance_arithmetic": "fp32_clean_full_plus_beta_minus_one_gap",
        "forward_layout": "single_conditional_batch", "outer_grad_context": "no_grad",
        "forecast_head": "full", "forecast_grid": "uniform_raw_time",
        "forecast_future_time": "max(0, current_noise_time - horizon)",
        "future_covector": "stopgrad(full_future.float() - base_future.float())",
        "direction_norm": "per_sample_match_current_full_minus_base_gap",
        "calibration_rule": "solver_index < calibration_steps and guidance interval active and beta != 1",
        "tf32": tf32, "time_grid": grid_cpu, "torch_version": torch.__version__,
        "transport_t_eps": denominator_floor,
        "source_sha256": {path.name: file_sha256(path) for path in (
            Path(__file__), ROOT / "experiments/raev2_flow_pullback.py",
            ROOT / "experiments/raev2_pfr_retiming.py",
            ROOT / "experiments/sample_raev2_pfr_retiming.py",
        )},
    }
    (out / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    rng = torch.Generator(device=device).manual_seed(args.seed)
    noise_hash, label_hash = hashlib.sha256(), hashlib.sha256()
    initial_rng_hash = hashlib.sha256(rng.get_state().cpu().numpy().tobytes()).hexdigest()
    images = []
    local_ids = []
    completed_local = 0
    completed_local_batches = 0
    aggregate = [dict() for _ in range(args.num_steps)]
    aggregate_counts = [dict() for _ in range(args.num_steps)]
    correction_steps = []
    for index, current in enumerate(grid_cpu[:-1]):
        correction_steps.append(
            args.mode != "ordinary" and index < args.calibration_steps
            and args.guidance_min_time <= current <= args.guidance_max_time
            and args.guidance_scale != 1.0
        )
    started = time.perf_counter()
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if args.precision == "bf16" else nullcontext()
    with torch.no_grad(), autocast:
        for start, stop, assigned in batches:
            state = torch.randn(stop - start, *config.misc.latent_size, device=device, generator=rng, dtype=torch.float32)
            labels = torch.arange(start, stop, device=device) % int(config.misc.num_classes)
            noise_hash.update(state.cpu().contiguous().numpy().tobytes())
            label_hash.update(labels.cpu().numpy().tobytes())
            if not assigned:
                continue
            local_ids.append(np.arange(start, stop, dtype=np.int64))
            for index, (current, following) in enumerate(zip(grid_cpu[:-1], grid_cpu[1:])):
                times = torch.full((len(state),), current, device=device, dtype=torch.float32)
                full, base = model(state, times, context=labels, attn_mask=None)
                full = full.float()
                gap = full - base.float()
                beta = args.guidance_scale if args.guidance_min_time <= current <= args.guidance_max_time else 1.0
                direction, telemetry = gap, {}
                if correction_steps[index]:
                    direction, telemetry = select_direction(
                        model, state, labels, gap, mode=args.mode,
                        start_time=current, horizon=args.horizon, substeps=args.substeps,
                        denominator_floor=denominator_floor,
                    )
                    if not bool(torch.isfinite(direction).all()):
                        raise FloatingPointError(f"nonfinite {args.mode} direction at batch {start}, step {index}")
                clean = full if beta == 1.0 else full + (beta - 1.0) * direction
                gap_norm = _norm(gap)
                direction_norm = _norm(direction)
                dimension_root = math.sqrt(gap[0].numel())
                telemetry.update({
                    "current_gap_rms": gap_norm / dimension_root,
                    "selected_direction_rms": direction_norm / dimension_root,
                    "selected_current_cosine": _cosine(direction, gap),
                    "selected_current_norm_ratio": torch.where(
                        gap_norm > 0, direction_norm / gap_norm.clamp_min(1e-12), torch.ones_like(gap_norm)
                    ),
                    "zero_current_gap": gap_norm <= 1e-12,
                })
                for key, value in telemetry.items():
                    values = value.detach().double()
                    aggregate[index][key] = aggregate[index].get(key, 0.0) + values.sum()
                    aggregate_counts[index][key] = aggregate_counts[index].get(key, 0) + values.numel()
                drift = clean_to_velocity(clean, state, times, denominator_floor=denominator_floor)
                state = state - (current - following) * drift
            if not bool(torch.isfinite(state).all()):
                raise FloatingPointError(f"nonfinite endpoint in batch {start}")
            decoded = decoder.decode(state)
            if not bool(torch.isfinite(decoded).all()):
                raise FloatingPointError(f"nonfinite decoder output in batch {start}")
            decoded = decoded.clamp(0, 1)
            if completed_local == 0:
                save_image(decoded.float().cpu(), out / "preview.png", nrow=4)
            images.append(decoded.mul(255).permute(0, 2, 3, 1).to(device="cpu", dtype=torch.uint8).numpy())
            completed_local += stop - start
            completed_local_batches += 1
            if completed_local_batches == 1 or completed_local_batches % 8 == 0 or completed_local == local_sample_count:
                elapsed = time.perf_counter() - started
                progress = {"completed": completed_local, "total": local_sample_count,
                            "global_sample_count": args.sample_count, "shard_index": args.shard_index,
                            "num_shards": args.num_shards, "global_rng_samples_consumed": stop,
                            "elapsed_seconds": elapsed,
                            "estimated_remaining_seconds": elapsed * (local_sample_count - completed_local) / completed_local,
                            "model_cost": model.summary()}
                print(json.dumps(progress), flush=True)
                (out / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
    archive = out / "samples.npz"
    np.savez(archive, np.concatenate(images))
    sample_ids_path = out / "sample_ids.npy"
    np.save(sample_ids_path, np.concatenate(local_ids))
    rows = []
    for index, stats in enumerate(aggregate):
        row = {"index": index, "noise_time": grid_cpu[index],
               "correction_active": correction_steps[index],
               "future_time": max(0.0, grid_cpu[index] - args.horizon) if correction_steps[index] else ""}
        row.update({key: float(value.item()) / aggregate_counts[index][key] for key, value in stats.items()})
        rows.append(row)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with (out / "geometry.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "protocol": PROTOCOL, "mode": args.mode, "samples": completed_local,
        "global_sample_count": args.sample_count, "local_sample_count": completed_local,
        "shard_index": args.shard_index, "num_shards": args.num_shards,
        "seed": args.seed, "batch_size": args.batch_size, "world_size": 1,
        "noise_sha256": noise_hash.hexdigest(), "labels_sha256": label_hash.hexdigest(),
        "initial_generator_sha256": initial_rng_hash,
        "final_generator_sha256": hashlib.sha256(rng.get_state().cpu().numpy().tobytes()).hexdigest(),
        "archive_sha256": file_sha256(archive), "elapsed_seconds": time.perf_counter() - started,
        "sample_ids_sha256": file_sha256(sample_ids_path),
        "correction_steps_per_batch": sum(correction_steps), "model_cost": model.summary(),
        "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(), "complete": True,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
