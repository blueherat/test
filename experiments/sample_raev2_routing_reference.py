#!/usr/bin/env python3
"""Paired direction-only RAEv2 guidance from same-time attention negatives.

Identity is a PAG control; uniform and preserve_self edit the selected final
DDT decoder attention. Their full-minus-negative direction is matched to the
native full-minus-base gap's per-sample norm, retaining the same IG strength.
Whole-batch sharding preserves every original noise draw and full-bank hash.
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
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torchvision.utils import save_image


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "external/RAEv2/src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.raev2_flow_pullback import normalize_like  # noqa: E402
from experiments.raev2_pfr_retiming import clean_to_velocity  # noqa: E402
from experiments.raev2_routing_reference import evaluate_routing_reference  # noqa: E402
from experiments.raev2_stage1_compat import install_raev2_decoder_config_compat  # noqa: E402
from experiments.raev2_training_core import file_sha256  # noqa: E402
from experiments.sample_raev2_flow_pullback import sample_batches  # noqa: E402
from experiments.sample_raev2_pfr_retiming import (  # noqa: E402
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, load_config, shifted_time_grid,
)
from utils.model_utils import instantiate_from_config  # noqa: E402


PROTOCOL = "raev2_routing_reference_sampling_v1"
MODES = ("ordinary", "identity", "preserve_self", "uniform")
ROUTING_METRICS = ("routing_information", "self_mass", "negative_self_mass")


def _norm(value: Tensor) -> Tensor:
    return value.float().flatten(1).norm(dim=1)


def _cosine(left: Tensor, right: Tensor) -> Tensor:
    numerator = (left.float().flatten(1) * right.float().flatten(1)).sum(1)
    return (numerator / (_norm(left) * _norm(right)).clamp_min(1e-12)).clamp(-1, 1)


@torch.no_grad()
def guided_routing_prediction(
    model,
    state: Tensor,
    times: Tensor,
    labels: Tensor,
    *,
    mode: str,
    guidance_scale: float,
    decoder_block: int = 1,
) -> tuple[Tensor, dict[str, Tensor], dict[str, int]]:
    """Return FP32 clean guidance, per-sample telemetry and execution counts.

    Counts identify architectural units in the actual evaluation path, not
    equivalent whole-model FLOPs. The negative branch recomputes all decoder
    blocks, of which exactly one uses the modified attention operation.
    Native-to-identity routing KL can be infinite and is deliberately neither
    checked for finiteness nor aggregated; routing_information remains finite.
    """

    if mode not in MODES:
        raise ValueError(f"unknown routing mode: {mode}")
    if not math.isfinite(guidance_scale) or guidance_scale < 1:
        raise ValueError("guidance_scale must be finite and >= 1")
    candidate = mode != "ordinary" and guidance_scale != 1.0
    routing = {}
    if candidate:
        reference = evaluate_routing_reference(
            model, state, times, labels, mode=mode, decoder_block=decoder_block
        )
        full, base = reference.full.float(), reference.base.float()
        raw = full - reference.negative.float()
        # Reduce head/token rows within each sample before across-sample means.
        for key in ROUTING_METRICS:
            value = reference.telemetry[key].float()
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError(f"nonfinite routing diagnostic: {key}")
            routing[key] = value.flatten(1).mean(1)
    else:
        full, base = model(state, times, context=labels, attn_mask=None)
        full, base = full.float(), base.float()
        raw = full - base
    gap = full - base
    direction = normalize_like(raw, gap) if candidate else gap
    if candidate and not bool(torch.isfinite(direction).all() & torch.isfinite(raw).all()):
        raise FloatingPointError("nonfinite routing-reference direction")
    clean = full if guidance_scale == 1.0 else full + (guidance_scale - 1.0) * direction
    gap_norm, raw_norm, direction_norm = _norm(gap), _norm(raw), _norm(direction)
    dimension_root = math.sqrt(gap[0].numel())
    telemetry = {
        "raw_gap_rms": raw_norm / dimension_root,
        "current_gap_rms": gap_norm / dimension_root,
        "selected_direction_rms": direction_norm / dimension_root,
        "direction_current_cosine": _cosine(direction, gap),
        "raw_current_cosine": _cosine(raw, gap),
        "direction_current_norm_ratio": torch.where(
            gap_norm > 0, direction_norm / gap_norm.clamp_min(1e-12), torch.ones_like(gap_norm)
        ),
        "zero_current_gap": gap_norm <= 1e-12,
        "raw_reference_fallback": raw_norm <= 1e-12,
        **routing,
    }
    costs = {
        "solver_evaluations": 1,
        "native_model_forward_calls": int(not candidate),
        "routing_reference_calls": int(candidate),
        "encoder_prefix_calls": 1,
        "prefix_calls_shared_with_negative": int(candidate),
        "encoder_block_calls": int(model.num_enc_blocks),
        "normal_decoder_head_block_calls": int(model.num_dec_blocks),
        "negative_decoder_head_block_calls": int(model.num_dec_blocks) if candidate else 0,
        "modified_attention_calls": int(candidate),
        "normal_full_final_head_calls": 1,
        "normal_base_final_head_calls": 1,
        "negative_final_head_calls": int(candidate),
        "input_vjp_calls": 0,
    }
    return clean, {key: value.detach() for key, value in telemetry.items()}, costs


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--decoder-block", type=int, default=1)
    parser.add_argument("--sample-count", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--seed", type=int, default=202609051)
    parser.add_argument("--guidance-scale", type=float, default=1.78)
    parser.add_argument("--guidance-min-time", type=float, default=0.1)
    parser.add_argument("--guidance-max-time", type=float, default=1.0)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    if min(args.sample_count, args.batch_size, args.num_steps) <= 0:
        parser.error("sample count, batch size and steps must be positive")
    if args.decoder_block < 0:
        parser.error("decoder block must be nonnegative")
    if not 0 <= args.guidance_min_time <= args.guidance_max_time <= 1:
        parser.error("invalid guidance interval")
    if not math.isfinite(args.guidance_scale) or args.guidance_scale < 1:
        parser.error("guidance scale must be finite and >= 1")
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
    model = instantiate_from_config(config.stage_2).to(device).eval().requires_grad_(False)
    if args.decoder_block >= model.num_dec_blocks:
        raise ValueError("decoder-block exceeds the model's DDT decoder depth")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    model.load_state_dict(checkpoint["ema"], strict=True)
    checkpoint_step = int(checkpoint.get("step", 0))
    del checkpoint
    shift = math.sqrt((config.misc.time_dist_shift_dim or math.prod(config.misc.latent_size)) / config.misc.time_dist_shift_base)
    grid_cpu = shifted_time_grid(args.num_steps, shift, device).cpu().tolist()
    denominator_floor = float(config.transport.t_eps)
    batches = list(sample_batches(args.sample_count, args.batch_size, args.shard_index, args.num_shards))
    local_count = sum(stop - start for start, stop, assigned in batches if assigned)
    decoder_artifacts = {}
    for key in ("pretrained_decoder_path", "normalization_stat_path"):
        raw_path = config.stage_1.params.get(key)
        if raw_path:
            artifact = Path(raw_path).expanduser().resolve()
            decoder_artifacts[key] = {"path": str(artifact), "sha256": file_sha256(artifact)}
    request = {
        "protocol": PROTOCOL,
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "global_sample_count": args.sample_count, "local_sample_count": local_count,
        "checkpoint_step": checkpoint_step, "checkpoint_sha256": file_sha256(args.checkpoint),
        "decoder_artifacts": decoder_artifacts, "config_sha256": file_sha256(args.config),
        "state_key": "ema", "world_size": 1,
        "sharding_rule": "whole original batch_index modulo num_shards; all workers generate/hash every original RNG batch",
        "guidance_arithmetic": "fp32_clean_full_plus_beta_minus_one_gap",
        "forward_layout": "single_conditional_batch_native_full_base_shared_prefix_with_negative",
        "direction_norm": "per_sample_match_current_full_minus_base_gap",
        "negative_reference": "selected DDT decoder attention edit at identical state/time/label",
        "aggregated_routing_metrics": list(ROUTING_METRICS),
        "routing_kl_note": "native-to-identity KL may be infinite and is never aggregated",
        "outer_grad_context": "no_grad", "tf32": tf32, "time_grid": grid_cpu,
        "torch_version": torch.__version__, "transport_t_eps": denominator_floor,
        "encoder_blocks": int(model.num_enc_blocks), "decoder_blocks": int(model.num_dec_blocks),
        "source_sha256": {path.name: file_sha256(path) for path in (
            Path(__file__), ROOT / "experiments/raev2_routing_reference.py",
            ROOT / "experiments/raev2_flow_pullback.py",
            ROOT / "experiments/sample_raev2_flow_pullback.py",
            ROOT / "experiments/raev2_pfr_retiming.py",
            ROOT / "experiments/sample_raev2_pfr_retiming.py",
        )},
    }
    (out / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    rng = torch.Generator(device=device).manual_seed(args.seed)
    noise_hash, label_hash = hashlib.sha256(), hashlib.sha256()
    initial_rng_hash = hashlib.sha256(rng.get_state().cpu().numpy().tobytes()).hexdigest()
    images, local_ids = [], []
    aggregate = [dict() for _ in range(args.num_steps)]
    aggregate_counts = [dict() for _ in range(args.num_steps)]
    calls, sample_calls = Counter(), Counter()
    completed, completed_batches = 0, 0

    def cost_summary():
        return {"architectural_calls": dict(calls), "architectural_sample_calls": dict(sample_calls),
                "cost_unit_note": "Counts of executed encoder prefixes, decoder blocks and final heads; edited decoder blocks retain MLP/modulation work. Not whole-forward equivalents."}

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
                beta = args.guidance_scale if args.guidance_min_time <= current <= args.guidance_max_time else 1.0
                clean, telemetry, costs = guided_routing_prediction(
                    model, state, times, labels, mode=args.mode,
                    guidance_scale=beta, decoder_block=args.decoder_block,
                )
                calls.update(costs)
                sample_calls.update({key: value * len(state) for key, value in costs.items()})
                for key, value in telemetry.items():
                    values = value.detach().double()
                    aggregate[index][key] = aggregate[index].get(key, 0.0) + values.sum()
                    aggregate_counts[index][key] = aggregate_counts[index].get(key, 0) + values.numel()
                state = state - (current - following) * clean_to_velocity(
                    clean, state, times, denominator_floor=denominator_floor
                )
            if not bool(torch.isfinite(state).all()):
                raise FloatingPointError(f"nonfinite endpoint in batch {start}")
            decoded = decoder.decode(state)
            if not bool(torch.isfinite(decoded).all()):
                raise FloatingPointError(f"nonfinite decoder output in batch {start}")
            decoded = decoded.clamp(0, 1)
            if completed == 0:
                save_image(decoded.float().cpu(), out / "preview.png", nrow=4)
            images.append(decoded.mul(255).permute(0, 2, 3, 1).to(device="cpu", dtype=torch.uint8).numpy())
            completed += stop - start
            completed_batches += 1
            if completed_batches == 1 or completed_batches % 8 == 0 or completed == local_count:
                elapsed = time.perf_counter() - started
                progress = {"completed": completed, "total": local_count,
                            "global_sample_count": args.sample_count, "shard_index": args.shard_index,
                            "num_shards": args.num_shards, "global_rng_samples_consumed": stop,
                            "elapsed_seconds": elapsed,
                            "estimated_remaining_seconds": elapsed * (local_count - completed) / completed,
                            "model_cost": cost_summary()}
                print(json.dumps(progress), flush=True)
                (out / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
    archive_path, ids_path = out / "samples.npz", out / "sample_ids.npy"
    np.savez(archive_path, np.concatenate(images))
    np.save(ids_path, np.concatenate(local_ids))
    rows = []
    for index, stats in enumerate(aggregate):
        current = grid_cpu[index]
        row = {"index": index, "noise_time": current,
               "correction_active": args.mode != "ordinary" and args.guidance_scale != 1.0 and args.guidance_min_time <= current <= args.guidance_max_time}
        row.update({key: float(value.item()) / aggregate_counts[index][key] for key, value in stats.items()})
        rows.append(row)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with (out / "geometry.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "protocol": PROTOCOL, "mode": args.mode, "samples": completed,
        "global_sample_count": args.sample_count, "local_sample_count": completed,
        "shard_index": args.shard_index, "num_shards": args.num_shards,
        "seed": args.seed, "batch_size": args.batch_size, "world_size": 1,
        "noise_sha256": noise_hash.hexdigest(), "labels_sha256": label_hash.hexdigest(),
        "initial_generator_sha256": initial_rng_hash,
        "final_generator_sha256": hashlib.sha256(rng.get_state().cpu().numpy().tobytes()).hexdigest(),
        "archive_sha256": file_sha256(archive_path), "sample_ids_sha256": file_sha256(ids_path),
        "elapsed_seconds": time.perf_counter() - started, "model_cost": cost_summary(),
        "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(), "complete": True,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
