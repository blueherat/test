"""Paired SDE interval ablation for the public SiT-XL/2+IG checkpoint.

This follows the official 250-step Euler-Maruyama equations while sharing the
initial noise, class labels, and every Brownian increment across conditions.
The public checkpoint is SiT-XL/2 with an eighth-layer auxiliary head, whereas
the paper's interval table used SiT-B/2 with a fourth-layer head.  Results from
this script are therefore a trend replication, not a numeric table replication.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.raev2_training_core import file_sha256  # noqa: E402
from experiments.run_internal_guidance_sit_audit import load_model  # noqa: E402
from experiments.run_raev2_ig_impulse_response import (  # noqa: E402
    _atomic_json,
    _load_condition,
    _open_memmap,
    build_validation_labels,
    deterministic_noise,
)
from experiments.run_sit_ig_endpoint_dynamics import sample_rms  # noqa: E402


PROTOCOL = "sit_ig_interval_ablation_v1"


@dataclass(frozen=True)
class IntervalCondition:
    name: str
    scale: float
    low: float
    high: float
    source: str

    def coefficient(self, time: float) -> float:
        if self.scale <= 1.0 or not self.low <= float(time) <= self.high:
            return 0.0
        return self.scale - 1.0


def paper_and_missing_conditions() -> tuple[IntervalCondition, ...]:
    return (
        IntervalCondition("no_ig", 1.0, 0.0, 1.0, "paper anchor"),
        IntervalCondition("scale1p9_all", 1.9, 0.0, 1.0, "paper anchor"),
        IntervalCondition("scale2p3_all", 2.3, 0.0, 1.0, "paper anchor"),
        IntervalCondition("scale2p3_t0p3_1p0", 2.3, 0.3, 1.0, "paper anchor"),
        IntervalCondition("scale2p3_t0p3_0p7", 2.3, 0.3, 0.7, "paper anchor"),
        IntervalCondition("scale2p3_t0p0_0p7", 2.3, 0.0, 0.7, "paper anchor"),
        IntervalCondition("scale2p3_t0p0_0p3", 2.3, 0.0, 0.3, "missing interval"),
        IntervalCondition("scale2p3_t0p7_1p0", 2.3, 0.7, 1.0, "missing interval"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=Path,
        default=ROOT / "research_repos/internal_guidance_study/Internal-Guidance/SiT",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/home/zhoushunyu/data/eqvae/models/Internal-Guidance/official/SiT/"
            "SiT-XL-IG-ImageNet256-800EP.pt"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="SiT-XL/2")
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--state-key", choices=("ema", "model"), default="ema")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--num-steps", type=int, default=250)
    parser.add_argument("--per-rank-batch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument(
        "--label-mode",
        choices=("sequential", "random_without_replacement"),
        default="sequential",
    )
    parser.add_argument("--label-seed", type=int)
    parser.add_argument("--brownian-seed-offset", type=int, default=900_000_007)
    parser.add_argument("--log-every-samples", type=int, default=20)
    return parser.parse_args()


def sde_time_grid(num_steps: int) -> torch.Tensor:
    if num_steps < 2:
        raise ValueError("SDE sampling requires at least two steps")
    return torch.cat(
        (
            torch.linspace(1.0, 0.04, num_steps, dtype=torch.float64),
            torch.zeros(1, dtype=torch.float64),
        )
    )


def linear_path_sde_drift(
    state: torch.Tensor,
    velocity: torch.Tensor,
    time: float,
) -> torch.Tensor:
    if not 0.0 < float(time) <= 1.0:
        raise ValueError("the official linear-path SDE evaluates only at t in (0, 1]")
    reverse_alpha_ratio = -(1.0 - float(time))
    variance = float(time)
    score = (reverse_alpha_ratio * velocity - state) / variance
    diffusion = 2.0 * float(time)
    return velocity - 0.5 * diffusion * score


def brownian_noise(
    sample_ids: np.ndarray,
    shape: tuple[int, ...],
    *,
    seed: int,
    step: int,
) -> torch.Tensor:
    return deterministic_noise(
        sample_ids,
        shape,
        seed=int(seed) + 10_007 * (int(step) + 1),
    )


def simulate_conditions(
    *,
    model: torch.nn.Module,
    noise: torch.Tensor,
    labels: torch.Tensor,
    sample_ids: np.ndarray,
    grid: torch.Tensor,
    conditions: tuple[IntervalCondition, ...],
    brownian_seed_value: int,
) -> torch.Tensor:
    condition_count, batch_size = len(conditions), len(noise)
    state = noise.unsqueeze(0).expand(condition_count, *noise.shape).reshape(
        condition_count * batch_size, *noise.shape[1:]
    ).double().contiguous()
    contexts = labels.unsqueeze(0).expand(condition_count, batch_size).reshape(-1)
    with torch.inference_mode():
        for step in range(len(grid) - 1):
            time = float(grid[step])
            dt = float(grid[step + 1] - grid[step])
            times = torch.full(
                (len(state),), time, device=state.device, dtype=torch.float32
            )
            output = model(state.float(), times, contexts)
            full = output[0].double().reshape(
                condition_count, batch_size, *state.shape[1:]
            )
            base = output[1].double().reshape_as(full)
            gap = full - base
            coefficients = torch.tensor(
                [condition.coefficient(time) for condition in conditions],
                device=state.device,
                dtype=torch.float64,
            ).reshape((condition_count, 1) + (1,) * (full.ndim - 2))
            velocity = full + coefficients * gap
            current = state.reshape_as(full)
            drift = linear_path_sde_drift(current, velocity, time)
            next_state = current + dt * drift
            if step < len(grid) - 2:
                epsilon = brownian_noise(
                    sample_ids,
                    tuple(noise.shape[1:]),
                    seed=brownian_seed_value,
                    step=step,
                ).to(device=state.device, dtype=torch.float64)
                diffusion = 2.0 * time
                stochastic = np.sqrt(diffusion * abs(dt)) * epsilon
                next_state = next_state + stochastic.unsqueeze(0)
            state = next_state.reshape(
                condition_count * batch_size, *state.shape[1:]
            )
    return state.reshape(condition_count, batch_size, *state.shape[1:]).float()


def analyze_endpoints(
    output_dir: Path,
    *,
    conditions: tuple[IntervalCondition, ...],
    samples: int,
    world_size: int,
) -> None:
    baseline = _load_condition(
        output_dir, condition_index=0, samples=samples, world_size=world_size
    ).astype(np.float64)
    baseline_rms = sample_rms(baseline)
    rows = []
    for index, condition in enumerate(conditions):
        endpoint = _load_condition(
            output_dir,
            condition_index=index,
            samples=samples,
            world_size=world_size,
        ).astype(np.float64)
        delta = sample_rms(endpoint - baseline)
        rows.append(
            {
                **asdict(condition),
                "samples": samples,
                "endpoint_rms_mean": float(sample_rms(endpoint).mean()),
                "delta_vs_no_ig_rms_mean": float(delta.mean()),
                "relative_delta_vs_no_ig_mean": float(
                    np.mean(delta / np.maximum(baseline_rms, 1e-30))
                ),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "endpoint_summary.csv", index=False)
    print(frame.to_string(index=False), flush=True)


def main() -> None:
    args = parse_args()
    counts = (args.samples, args.num_steps, args.per_rank_batch, args.log_every_samples)
    if any(value <= 0 for value in counts):
        raise ValueError("counts must be positive")
    conditions = paper_and_missing_conditions()
    if args.per_rank_batch * len(conditions) > 16:
        raise ValueError("effective model batch must not exceed 16")
    dist.init_process_group("nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output_dir = args.output_dir.expanduser().resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint_hash = file_sha256(checkpoint_path) if rank == 0 else ""
    objects = [checkpoint_hash]
    dist.broadcast_object_list(objects, src=0)
    label_seed = int(args.seed if args.label_seed is None else args.label_seed)
    brownian_seed_value = int(args.seed + args.brownian_seed_offset)
    grid = sde_time_grid(args.num_steps)
    manifest = {
        "protocol": PROTOCOL,
        "status": "running",
        "training": False,
        "paper_scope": "trend replication: public XL/2 checkpoint, not Table 3 B/2",
        "repo": str(args.repo.expanduser().resolve()),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": objects[0],
        "model_name": args.model_name,
        "encoder_depth": args.encoder_depth,
        "state_key": args.state_key,
        "samples": args.samples,
        "seed": args.seed,
        "brownian_seed": brownian_seed_value,
        "world_size": world_size,
        "precision": "fp32 model, fp64 state",
        "tf32": False,
        "latent_size": [4, 32, 32],
        "num_steps": args.num_steps,
        "solver_grid": grid.tolist(),
        "sampler": "official Euler-Maruyama equations, paired deterministic RNG",
        "cfg_scale": 1.0,
        "conditions": [asdict(condition) for condition in conditions],
        "label_mode": args.label_mode,
        "label_seed": label_seed,
        "same_initial_noise_labels_and_brownian_path": True,
    }
    manifest_path = output_dir / "manifest.json"
    if rank == 0:
        if manifest_path.is_file():
            current = json.loads(manifest_path.read_text(encoding="utf-8"))
            keys = (
                "protocol",
                "checkpoint_sha256",
                "model_name",
                "encoder_depth",
                "state_key",
                "samples",
                "seed",
                "brownian_seed",
                "world_size",
                "latent_size",
                "num_steps",
                "solver_grid",
                "conditions",
                "label_mode",
                "label_seed",
            )
            changed = [key for key in keys if current.get(key) != manifest.get(key)]
            if changed:
                raise RuntimeError(f"cannot resume changed interval protocol: {changed}")
        else:
            _atomic_json(manifest_path, manifest)
            labels = build_validation_labels(
                args.samples,
                1000,
                mode=args.label_mode,
                seed=label_seed,
            )
            np.savez_compressed(
                output_dir / "sample_protocol.npz",
                sample_ids=np.arange(args.samples, dtype=np.int64),
                labels=labels,
            )
    dist.barrier()
    labels = np.load(output_dir / "sample_protocol.npz")["labels"].astype(np.int64)
    local_ids = np.arange(rank, args.samples, world_size, dtype=np.int64)
    model, metadata = load_model(
        repo=args.repo,
        checkpoint_path=checkpoint_path,
        model_name=args.model_name,
        encoder_depth=args.encoder_depth,
        state_key=args.state_key,
        device=device,
    )
    endpoints = _open_memmap(
        output_dir / f"endpoints_rank{rank:02d}.npy",
        shape=(len(conditions), len(local_ids), 4, 32, 32),
        dtype=np.float32,
    )
    progress_path = output_dir / f"progress_rank{rank:02d}.npy"
    progress_existed = progress_path.is_file()
    progress = _open_memmap(
        progress_path, shape=(len(local_ids),), dtype=np.bool_
    )
    if not progress_existed:
        progress.fill(False)
        progress.flush()
    for start in range(0, len(local_ids), args.per_rank_batch):
        stop = min(start + args.per_rank_batch, len(local_ids))
        if bool(np.asarray(progress[start:stop]).all()):
            continue
        if bool(np.asarray(progress[start:stop]).any()):
            raise RuntimeError("partially complete batch cannot be resumed safely")
        ids = local_ids[start:stop]
        noise = deterministic_noise(ids, (4, 32, 32), seed=args.seed).to(device)
        batch_labels = torch.from_numpy(labels[ids]).to(
            device=device, dtype=torch.long
        )
        endpoint = simulate_conditions(
            model=model,
            noise=noise,
            labels=batch_labels,
            sample_ids=ids,
            grid=grid,
            conditions=conditions,
            brownian_seed_value=brownian_seed_value,
        )
        endpoints[:, start:stop] = endpoint.cpu().numpy()
        if not (output_dir / f"grouping_check_rank{rank:02d}.json").is_file():
            reference = simulate_conditions(
                model=model,
                noise=noise,
                labels=batch_labels,
                sample_ids=ids,
                grid=grid,
                conditions=(conditions[0],),
                brownian_seed_value=brownian_seed_value,
            )[0]
            delta = endpoint[0].double() - reference.double()
            check = {
                "rank": rank,
                "samples_checked": len(ids),
                "rms": float(delta.square().mean().sqrt().cpu()),
                "maximum_absolute": float(delta.abs().max().cpu()),
            }
            _atomic_json(output_dir / f"grouping_check_rank{rank:02d}.json", check)
            if check["rms"] > 1e-6 or check["maximum_absolute"] > 1e-4:
                raise RuntimeError(f"grouped baseline mismatch: {check}")
        endpoints.flush()
        progress[start:stop] = True
        progress.flush()
        if rank == 0 and (
            stop % args.log_every_samples == 0 or stop == len(local_ids)
        ):
            print(f"[rank 0] local samples {stop}/{len(local_ids)}", flush=True)
    _atomic_json(
        output_dir / f"complete_rank{rank:02d}.json",
        {"rank": rank, "local_rows": len(local_ids), "complete": True},
    )
    dist.barrier()
    if rank == 0:
        analyze_endpoints(
            output_dir,
            conditions=conditions,
            samples=args.samples,
            world_size=world_size,
        )
        final = json.loads(manifest_path.read_text(encoding="utf-8"))
        final["status"] = "complete"
        final["model_metadata"] = metadata
        _atomic_json(manifest_path, final)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
