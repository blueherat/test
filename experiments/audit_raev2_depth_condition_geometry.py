#!/usr/bin/env python3
"""Measure depth-by-condition component geometry on RAEv2 trajectories."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
RAEV2_SRC = ROOT / "external" / "RAEv2" / "src"
for path in (RAEV2_SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.raev2_depth_condition_guidance import (  # noqa: E402
    minimum_norm_convex_consensus,
    mobius_components,
)
from experiments.raev2_pfr_retiming import clean_to_velocity  # noqa: E402
from experiments.sample_raev2_depth_condition_guidance import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    evaluate_four_corners,
)
from experiments.sample_raev2_pfr_retiming import (  # noqa: E402
    load_config,
    shifted_time_grid,
)
from utils.model_utils import instantiate_from_config  # noqa: E402


def _rms(vector: torch.Tensor) -> torch.Tensor:
    return vector.square().mean(dim=1).sqrt()


def _cosine(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    numerator = (first * second).sum(dim=1)
    denominator = first.square().sum(dim=1).sqrt() * second.square().sum(dim=1).sqrt()
    return torch.where(denominator > 0, numerator / denominator, torch.zeros_like(numerator))


def component_geometry(
    *,
    full_conditional: torch.Tensor,
    base_conditional: torch.Tensor,
    full_unconditional: torch.Tensor,
    base_unconditional: torch.Tensor,
) -> dict[str, torch.Tensor]:
    components = mobius_components(
        full_conditional=full_conditional,
        base_conditional=base_conditional,
        full_unconditional=full_unconditional,
        base_unconditional=base_unconditional,
    )
    depth = components.depth_main.flatten(1).float()
    interaction = components.interaction.flatten(1).float()
    conditional_depth = (full_conditional - base_conditional).flatten(1).float()
    full_cfg = (full_conditional - full_unconditional).flatten(1).float()
    base_cfg = (base_conditional - base_unconditional).flatten(1).float()
    consensus, consensus_weight = minimum_norm_convex_consensus(
        conditional_depth,
        depth,
    )
    midpoint = 0.5 * (conditional_depth + depth)
    depth_rms = _rms(depth)
    interaction_rms = _rms(interaction)
    conditional_depth_rms = _rms(conditional_depth)
    consensus_rms = _rms(consensus)
    return {
        "depth_rms": depth_rms,
        "interaction_rms": interaction_rms,
        "conditional_depth_rms": conditional_depth_rms,
        "full_cfg_rms": _rms(full_cfg),
        "consensus_rms": consensus_rms,
        "midpoint_rms": _rms(midpoint),
        "consensus_weight": consensus_weight,
        "consensus_to_conditional_depth_rms": torch.where(
            conditional_depth_rms > 0,
            consensus_rms / conditional_depth_rms,
            torch.zeros_like(consensus_rms),
        ),
        "interaction_to_conditional_depth_rms": torch.where(
            conditional_depth_rms > 0,
            interaction_rms / conditional_depth_rms,
            torch.zeros_like(interaction_rms),
        ),
        "equal_action_multiplier": torch.where(
            interaction_rms > 0,
            conditional_depth_rms / interaction_rms,
            torch.zeros_like(interaction_rms),
        ),
        "cos_depth_interaction": _cosine(depth, interaction),
        "cos_interaction_conditional_depth": _cosine(
            interaction, conditional_depth
        ),
        "cos_interaction_full_cfg": _cosine(interaction, full_cfg),
        "cos_interaction_base_cfg": _cosine(interaction, base_cfg),
        "cos_full_cfg_base_cfg": _cosine(full_cfg, base_cfg),
        "cos_depth_full_cfg": _cosine(depth, full_cfg),
        "depth_ascent_under_conditional_depth": (
            depth * conditional_depth
        ).sum(dim=1),
        "interaction_ascent_under_conditional_depth": (
            interaction * conditional_depth
        ).sum(dim=1),
        "depth_ascent_under_consensus": (depth * consensus).sum(dim=1),
        "conditional_depth_ascent_under_consensus": (
            conditional_depth * consensus
        ).sum(dim=1),
        "full_cfg_ascent_under_interaction": (full_cfg * interaction).sum(dim=1),
        "base_cfg_ascent_under_interaction": (base_cfg * interaction).sum(dim=1),
        "full_cfg_ascent_under_conditional_depth": (
            full_cfg * conditional_depth
        ).sum(dim=1),
        "base_cfg_ascent_under_conditional_depth": (
            base_cfg * conditional_depth
        ).sum(dim=1),
    }


def summarize_step(records: dict[str, list[np.ndarray]], step: int, time: float) -> dict:
    row: dict[str, float | int] = {"step": step, "noise_time": time}
    for name, parts in records.items():
        values = np.concatenate(parts).astype(np.float64, copy=False)
        row[f"{name}_mean"] = float(values.mean())
        row[f"{name}_median"] = float(np.median(values))
        row[f"{name}_q05"] = float(np.quantile(values, 0.05))
        row[f"{name}_q95"] = float(np.quantile(values, 0.95))
        if name.endswith("ascent_under_conditional_depth"):
            row[f"{name}_positive_fraction"] = float(np.mean(values > 0))
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sampling-seed", type=int, default=20260903)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--state-key", choices=("ema", "model"), default="ema")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dino-ckpt-dir",
        type=Path,
        default=Path("/home/zhoushunyu/data/eqvae/models/RAEv2/encoders/dinov3"),
    )
    parser.add_argument("--dino-repo-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_count <= 0 or args.batch_size <= 0:
        raise ValueError("sample count and batch size must be positive")
    if args.num_steps is not None and args.num_steps <= 0:
        raise ValueError("number of steps must be positive")
    os.environ["DINOV3_CKPT_DIR"] = str(args.dino_ckpt_dir.expanduser().resolve())
    if args.dino_repo_dir is not None:
        os.environ["DINOV3_REPO_DIR"] = str(args.dino_repo_dir.expanduser().resolve())

    device = torch.device(args.device)
    config = load_config(args.config.expanduser().resolve())
    model = instantiate_from_config(config.stage_2).to(device).eval().requires_grad_(False)
    checkpoint = torch.load(
        args.checkpoint.expanduser().resolve(),
        map_location="cpu",
        mmap=True,
        weights_only=False,
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
    generator = torch.Generator(device=device).manual_seed(args.sampling_seed)
    rows_by_step: list[dict[str, list[np.ndarray]]] = [dict() for _ in range(num_steps)]
    null_label = int(config.misc.num_classes)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16")

    with torch.inference_mode(), autocast:
        for start in range(0, args.sample_count, args.batch_size):
            size = min(args.batch_size, args.sample_count - start)
            state = torch.randn(
                size,
                *config.misc.latent_size,
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
            labels = torch.arange(start, start + size, device=device) % null_label
            for step in range(num_steps):
                current = float(time_grid[step])
                following = float(time_grid[step + 1])
                times = torch.full((size,), current, device=device)
                corners = evaluate_four_corners(
                    model, state, times, labels, null_label=null_label
                )
                metrics = component_geometry(
                    full_conditional=corners[0],
                    base_conditional=corners[1],
                    full_unconditional=corners[2],
                    base_unconditional=corners[3],
                )
                for name, values in metrics.items():
                    rows_by_step[step].setdefault(name, []).append(
                        values.float().cpu().numpy()
                    )
                drift = clean_to_velocity(
                    corners[0], state, times, denominator_floor=float(config.transport.t_eps)
                )
                state = state - (current - following) * drift

    rows = [
        summarize_step(records, step, float(time_grid[step]))
        for step, records in enumerate(rows_by_step)
    ]
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "geometry_by_time.csv", index=False)
    summary = {
        "protocol": "raev2_depth_condition_geometry_v1",
        "sampling_seed": args.sampling_seed,
        "sample_count": args.sample_count,
        "batch_size": args.batch_size,
        "checkpoint_step": checkpoint_step,
        "state_key": args.state_key,
        "num_steps": num_steps,
        "time_shift": shift,
        "trajectory": "unguided full-conditional Euler rollout",
        "geometry_space": "clean prediction; cosine and same-time relative norms equal velocity-space values",
        "high_noise_summary": {
            column: float(frame.loc[frame.noise_time > 0.5, column].mean())
            for column in frame.columns
            if column.endswith(("_mean", "_positive_fraction"))
        },
    }
    (output_dir / "geometry_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
