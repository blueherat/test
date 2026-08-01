"""Measure how RAEv2 x-prediction parameterization scales the IG head gap.

The official sampler and guidance function are left unchanged.  A forward hook
on the frozen stage-2 model observes its full/base clean predictions at every
Euler step and reports the raw gap, the induced velocity gap, and the actual
one-step Euler impulse h * gap / t.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import math
import os
import sys
from functools import partial
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist


ROOT = Path(__file__).resolve().parents[1]
RAEV2_SRC = ROOT / "external" / "RAEv2" / "src"
for path in (RAEV2_SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.raev2_stage1_compat import (  # noqa: E402
    install_raev2_decoder_config_compat,
)
from experiments.raev2_training_core import validate_full_stage2_checkpoint  # noqa: E402
from experiments.run_raev2_distribution_auc import (  # noqa: E402
    autocast_context,
    build_requested_labels,
    load_config,
    shifted_solver_grid,
)
from stage2.transport import create_sampler, create_transport  # noqa: E402
from utils.guidance_utils import forward_with_internalguidance  # noqa: E402
from utils.model_utils import instantiate_from_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/configs/raev2_strict_lpl_dinov3l_k7.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/home/zhoushunyu/data/eqvae/models/RAEv2/stage2/imagenet/"
            "dinov3l-k7/checkpoint.pt"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--per-rank-batch", type=int, default=4)
    parser.add_argument("--log-every-batches", type=int, default=25)
    parser.add_argument("--ig-scale", type=float)
    parser.add_argument("--state-key", choices=("ema", "model"), default="ema")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument(
        "--dino-ckpt-dir",
        type=Path,
        default=Path("/home/zhoushunyu/data/eqvae/models/RAEv2/encoders/dinov3"),
    )
    parser.add_argument("--dino-repo-dir", type=Path)
    return parser.parse_args()


def parameterization_scales(
    raw_head_gap_rms: float,
    *,
    ig_scale: float,
    time: float,
    step_size: float,
    t_eps: float,
    active: bool,
) -> dict[str, float]:
    multiplier = abs(float(ig_scale) - 1.0) if active else 0.0
    clean_gap = multiplier * float(raw_head_gap_rms)
    velocity_gap = clean_gap / max(float(time), float(t_eps))
    return {
        "guided_clean_gap_rms": clean_gap,
        "velocity_gap_rms": velocity_gap,
        "euler_impulse_rms": float(step_size) * velocity_gap,
    }


class DualHeadGapHook:
    def __init__(self, grid: torch.Tensor, in_channels: int) -> None:
        self.grid = grid.detach().cpu().double()
        self.in_channels = int(in_channels)
        self.gap_sumsq = torch.zeros(len(grid) - 1, dtype=torch.float64)
        self.full_sumsq = torch.zeros(len(grid) - 1, dtype=torch.float64)
        self.count = torch.zeros(len(grid) - 1, dtype=torch.float64)
        self.call_index = 0

    def begin_batch(self) -> None:
        if self.call_index not in (0, len(self.grid) - 1):
            raise RuntimeError("previous sampler batch did not complete all Euler steps")
        self.call_index = 0

    def __call__(self, _module: torch.nn.Module, args: tuple[Any, ...], output: Any) -> None:
        index = self.call_index
        if index >= len(self.grid) - 1:
            raise RuntimeError("model hook observed too many calls")
        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError("internal-guidance model did not return full/base heads")
        if len(args) < 2:
            raise RuntimeError("model hook cannot observe solver time")
        observed_time = float(args[1][0].item())
        expected_time = float(self.grid[index].item())
        if abs(observed_time - expected_time) > 1e-6:
            raise RuntimeError(
                f"hook time mismatch at step {index}: {observed_time} != {expected_time}"
            )
        full = output[0][:, : self.in_channels].detach().float()
        base = output[1][:, : self.in_channels].detach().float()
        gap = full - base
        self.gap_sumsq[index] += gap.square().sum().cpu().double()
        self.full_sumsq[index] += full.square().sum().cpu().double()
        self.count[index] += float(gap.numel())
        self.call_index += 1

    def validate_batch(self) -> None:
        if self.call_index != len(self.grid) - 1:
            raise RuntimeError(
                f"model hook observed {self.call_index} calls, expected {len(self.grid) - 1}"
            )

    def reduce(self, device: torch.device) -> None:
        for value in (self.gap_sumsq, self.full_sumsq, self.count):
            reduced = value.to(device=device)
            dist.all_reduce(reduced)
            value.copy_(reduced.cpu())


def _plot(frame: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    for branch, marker in (("full_path", "o"), ("ig_path", "s")):
        subset = frame[frame["branch"] == branch]
        axes[0].plot(subset["time"], subset["raw_head_gap_rms"], marker=marker,
                     markevery=8, label=branch)
        axes[1].plot(subset["time"], subset["velocity_gap_rms"], marker=marker,
                     markevery=8, label=branch)
        axes[2].plot(subset["time"], subset["euler_impulse_rms"], marker=marker,
                     markevery=8, label=branch)
    titles = ("Raw full/base clean gap", "Induced velocity gap", "Per-step Euler impulse")
    ylabels = ("RMS(full - base)", "RMS(Delta v)", "RMS(h Delta v)")
    for axis, title, ylabel in zip(axes, titles, ylabels):
        axis.set_title(title)
        axis.set_xlabel("Solver time t (sampling: 1 to 0)")
        axis.set_ylabel(ylabel)
        axis.invert_xaxis()
        axis.grid(True, alpha=0.22)
        axis.legend(frameon=False)
        axis.axvline(0.1, color="#333333", linestyle="--", linewidth=1)
    fig.suptitle("RAEv2 Internal Guidance: Raw Gap vs 1/t Parameterization")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    install_raev2_decoder_config_compat()
    args = parse_args()
    if args.samples <= 0 or args.per_rank_batch <= 0:
        raise ValueError("sample and batch counts must be positive")
    os.environ["DINOV3_CKPT_DIR"] = str(args.dino_ckpt_dir.expanduser().resolve())
    if args.dino_repo_dir is not None:
        os.environ["DINOV3_REPO_DIR"] = str(args.dino_repo_dir.expanduser().resolve())

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True

    output_dir = args.output_dir.expanduser().resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    config = load_config(args.config.expanduser().resolve())
    latent_size = tuple(int(value) for value in config.misc.latent_size)
    shift = math.sqrt(
        (config.misc.time_dist_shift_dim or math.prod(latent_size))
        / config.misc.time_dist_shift_base
    )
    grid = shifted_solver_grid(int(config.sampler.num_steps), shift)
    model = instantiate_from_config(config.stage_2).to(device).eval()
    model.requires_grad_(False)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    validate_full_stage2_checkpoint(checkpoint)
    model.load_state_dict(checkpoint[args.state_key], strict=True)
    checkpoint_step = int(checkpoint.get("step", 0))
    checkpoint_epoch = int(checkpoint.get("epoch", 0))
    del checkpoint

    transport = create_transport(config=config.transport, time_dist_shift=shift)
    sampler = create_sampler(transport, guidance_config=config.guidance)
    sample_fn = sampler.sample_ode(**dataclasses.asdict(config.sampler))
    model_fn = partial(forward_with_internalguidance, model)
    official_scale = (
        float(args.ig_scale) if args.ig_scale is not None else float(config.guidance.ig.scale)
    )
    interval = (float(config.guidance.ig.t_min), float(config.guidance.ig.t_max))
    labels = build_requested_labels(args.samples, int(config.misc.num_classes))
    local_ids = np.arange(rank, args.samples, world_size, dtype=np.int64)
    local_labels = labels[local_ids]
    generator = torch.Generator(device="cpu").manual_seed(
        int(args.seed) + 1_000_003 * rank
    )
    local_noise = torch.randn(
        (local_ids.size, *latent_size), generator=generator, dtype=torch.float32
    )

    branch_hooks: dict[str, DualHeadGapHook] = {}
    total_batches = math.ceil(local_ids.size / args.per_rank_batch)
    for branch, branch_scale in (("full_path", 1.0), ("ig_path", official_scale)):
        hook = DualHeadGapHook(grid, int(model.in_channels))
        handle = model.register_forward_hook(hook)
        with torch.inference_mode():
            for batch_index, start in enumerate(
                range(0, local_ids.size, args.per_rank_batch)
            ):
                end = min(start + args.per_rank_batch, local_ids.size)
                noise = local_noise[start:end].to(device=device)
                batch_labels = torch.from_numpy(local_labels[start:end]).to(device=device)
                null = torch.full(
                    (noise.shape[0],),
                    int(config.misc.num_classes),
                    device=device,
                    dtype=torch.long,
                )
                hook.begin_batch()
                with autocast_context(args.precision):
                    sample_fn(
                        torch.cat((noise, noise), dim=0),
                        model_fn,
                        context=torch.cat((batch_labels.long(), null), dim=0),
                        attn_mask=None,
                        ig_scale=float(branch_scale),
                        ig_interval=interval,
                    )
                hook.validate_batch()
                if rank == 0 and (
                    (batch_index + 1) % args.log_every_batches == 0
                    or batch_index + 1 == total_batches
                ):
                    print(
                        f"[{branch}] batches {batch_index + 1}/{total_batches}",
                        flush=True,
                    )
        handle.remove()
        hook.reduce(device)
        branch_hooks[branch] = hook
        dist.barrier()

    if rank == 0:
        rows = []
        t_eps = float(config.transport.t_eps)
        for branch, hook in branch_hooks.items():
            for index in range(len(grid) - 1):
                count = float(hook.count[index].item())
                raw = math.sqrt(float(hook.gap_sumsq[index].item()) / count)
                full_rms = math.sqrt(float(hook.full_sumsq[index].item()) / count)
                time = float(grid[index].item())
                next_time = float(grid[index + 1].item())
                step_size = time - next_time
                active = interval[0] <= time <= interval[1]
                scales = parameterization_scales(
                    raw,
                    ig_scale=official_scale,
                    time=time,
                    step_size=step_size,
                    t_eps=t_eps,
                    active=active,
                )
                rows.append(
                    {
                        "branch": branch,
                        "solver_index": index,
                        "time": time,
                        "next_time": next_time,
                        "step_size": step_size,
                        "inverse_t_safe": 1.0 / max(time, t_eps),
                        "h_over_t_safe": step_size / max(time, t_eps),
                        "ig_active": active,
                        "raw_head_gap_rms": raw,
                        "raw_gap_over_full_rms": raw / max(full_rms, 1e-30),
                        **scales,
                    }
                )
        frame = pd.DataFrame(rows)
        frame.to_csv(output_dir / "ig_parameterization_curve.csv", index=False)
        _plot(frame, output_dir / "ig_parameterization_curve.png")
        manifest = {
            "protocol": "raev2_ig_parameterization_audit_v1",
            "inference_only": True,
            "official_sampler_unchanged": True,
            "measurement": "top-level model forward hook",
            "checkpoint": str(checkpoint_path),
            "checkpoint_step": checkpoint_step,
            "checkpoint_epoch": checkpoint_epoch,
            "state_key": args.state_key,
            "samples": args.samples,
            "seed": args.seed,
            "world_size": world_size,
            "ig_scale": official_scale,
            "ig_interval": interval,
            "prediction": config.transport.prediction,
            "t_eps": t_eps,
            "time_dist_shift": shift,
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        selected = frame[
            frame["solver_index"].isin((0, 67, 84, 92, 97, 98, 99))
        ]
        print(selected.to_string(index=False))

    del model, sampler, transport
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
