"""Compare SiT internal-guidance pulses with matched random directions.

At each selected Euler step, the official ``full - base`` velocity gap is
compared against deterministic random directions that are orthogonal to the
gap and normalized to the same per-sample RMS.  All branches receive equal
one-step state-injection energy and then follow the frozen full-head flow.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.raev2_training_core import file_sha256
from experiments.run_internal_guidance_sit_audit import load_model
from experiments.run_raev2_ig_impulse_response import (
    _atomic_json,
    _load_condition,
    _load_small_shards,
    _open_memmap,
    bootstrap_mean_interval,
    build_validation_labels,
    deterministic_noise,
)
from experiments.run_sit_ig_endpoint_dynamics import (
    official_baseline,
    parse_int_list,
    sample_rms,
)

PROTOCOL = "sit_ig_direction_specificity_v1"


@dataclass(frozen=True)
class DirectionCondition:
    name: str
    family: str
    step: int | None
    gamma: float
    sign: int
    probe_index: int | None
    pair_name: str | None

    def active(self, step: int) -> bool:
        return self.step is not None and int(step) == self.step


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
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--pulse-steps", type=parse_int_list, default=(5, 15, 25, 35, 45, 49))
    parser.add_argument("--gamma", type=float, default=0.01)
    parser.add_argument("--probe-count", type=int, default=4)
    parser.add_argument("--per-rank-batch", type=int, default=2)
    parser.add_argument("--condition-group-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--label-seed", type=int)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--log-every-samples", type=int, default=4)
    return parser.parse_args()


def build_conditions(
    pulse_steps: tuple[int, ...],
    *,
    gamma: float,
    probe_count: int,
    num_steps: int,
) -> tuple[DirectionCondition, ...]:
    if gamma <= 0 or not np.isfinite(gamma) or probe_count <= 0:
        raise ValueError("positive gamma and probe count are required")
    if len(set(pulse_steps)) != len(pulse_steps) or any(
        step < 0 or step >= num_steps for step in pulse_steps
    ):
        raise ValueError("pulse steps must be unique and valid")
    result = [DirectionCondition("baseline", "baseline", None, 0.0, 0, None, None)]
    for step in pulse_steps:
        pair = f"step_{step:03d}_ig"
        result.extend(
            (
                DirectionCondition(pair + "_pos", "ig", step, gamma, 1, None, pair),
                DirectionCondition(pair + "_neg", "ig", step, gamma, -1, None, pair),
            )
        )
        for probe in range(probe_count):
            pair = f"step_{step:03d}_random_{probe:02d}"
            result.extend(
                (
                    DirectionCondition(pair + "_pos", "random", step, gamma, 1, probe, pair),
                    DirectionCondition(pair + "_neg", "random", step, gamma, -1, probe, pair),
                )
            )
    return tuple(result)


def deterministic_probe(
    sample_ids: np.ndarray,
    shape: tuple[int, ...],
    *,
    seed: int,
    step: int,
    probe_index: int,
) -> torch.Tensor:
    rows = []
    for sample_id in sample_ids.tolist():
        generator = torch.Generator(device="cpu").manual_seed(
            int(seed)
            + 1_000_003 * int(sample_id)
            + 10_007 * int(step)
            + 1_009 * int(probe_index)
        )
        rows.append(torch.randn(shape, generator=generator, dtype=torch.float32))
    return torch.stack(rows)


def matched_orthogonal_direction(random: torch.Tensor, gap: torch.Tensor) -> torch.Tensor:
    if random.shape != gap.shape:
        raise ValueError("random and gap tensors must align")
    dims = tuple(range(1, gap.ndim))
    projection = (random * gap).mean(dim=dims) / gap.square().mean(dim=dims).clamp_min(1e-30)
    shape = (len(gap),) + (1,) * (gap.ndim - 1)
    orthogonal = random - projection.reshape(shape) * gap
    gap_rms = gap.square().mean(dim=dims).sqrt()
    orthogonal_rms = orthogonal.square().mean(dim=dims).sqrt().clamp_min(1e-30)
    return orthogonal * (gap_rms / orthogonal_rms).reshape(shape)


def direction_diagnostics(direction: torch.Tensor, gap: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, gap.ndim))
    dot = (direction * gap).mean(dim=dims)
    direction_rms = direction.square().mean(dim=dims).sqrt()
    gap_rms = gap.square().mean(dim=dims).sqrt()
    cosine = dot / (direction_rms * gap_rms).clamp_min(1e-30)
    rms_ratio = direction_rms / gap_rms.clamp_min(1e-30)
    return torch.stack((cosine, rms_ratio), dim=-1)


def simulate_group(
    *,
    model: torch.nn.Module,
    noise: torch.Tensor,
    labels: torch.Tensor,
    sample_ids: np.ndarray,
    grid: torch.Tensor,
    conditions: tuple[DirectionCondition, ...],
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    condition_count, batch_size = len(conditions), len(noise)
    state = noise.unsqueeze(0).expand(condition_count, *noise.shape).reshape(
        condition_count * batch_size, *noise.shape[1:]
    ).double().contiguous()
    contexts = labels.unsqueeze(0).expand(condition_count, batch_size).reshape(-1).contiguous()
    unit_norm = torch.zeros(condition_count, batch_size, device=state.device, dtype=torch.float64)
    diagnostics = torch.full(
        (condition_count, batch_size, 2),
        torch.nan,
        device=state.device,
        dtype=torch.float64,
    )
    with torch.inference_mode():
        for step in range(len(grid) - 1):
            time, next_time = float(grid[step]), float(grid[step + 1])
            dt = next_time - time
            times = torch.full((len(state),), time, device=state.device, dtype=torch.float32)
            output = model(state.float(), times, contexts)
            full = output[0].double().reshape(condition_count, batch_size, *state.shape[1:])
            base = output[1].double().reshape_as(full)
            gap = full - base
            perturbation = torch.zeros_like(full)
            for index, condition in enumerate(conditions):
                if not condition.active(step):
                    continue
                if condition.family == "ig":
                    direction = gap[index]
                elif condition.family == "random":
                    raw = deterministic_probe(
                        sample_ids,
                        tuple(noise.shape[1:]),
                        seed=seed,
                        step=step,
                        probe_index=int(condition.probe_index),
                    ).to(device=state.device, dtype=torch.float64)
                    direction = matched_orthogonal_direction(raw, gap[index])
                else:
                    raise RuntimeError(f"unsupported active family: {condition.family}")
                perturbation[index] = float(condition.sign) * float(condition.gamma) * direction
                unit_norm[index] = abs(dt) * direction.flatten(1).square().mean(1).sqrt()
                diagnostics[index] = direction_diagnostics(direction, gap[index])
            velocity = full + perturbation
            state = (state.reshape_as(full) + dt * velocity).reshape(
                condition_count * batch_size, *state.shape[1:]
            )
    endpoint = state.reshape(condition_count, batch_size, *state.shape[1:]).float()
    return endpoint, unit_norm, diagnostics


def analyze_results(
    *,
    output_dir: Path,
    conditions: tuple[DirectionCondition, ...],
    grid: torch.Tensor,
    samples: int,
    world_size: int,
    repeats: int,
    seed: int,
) -> None:
    baseline = _load_condition(
        output_dir, condition_index=0, samples=samples, world_size=world_size
    ).astype(np.float64)
    unit_norms = _load_small_shards(
        output_dir,
        filename="unit_injected_norm_rank{rank:02d}.npy",
        samples=samples,
        world_size=world_size,
    )[:, :, 0]
    diagnostics = _load_small_shards(
        output_dir,
        filename="direction_diagnostics_rank{rank:02d}.npy",
        samples=samples,
        world_size=world_size,
    )
    by_pair: dict[str, tuple[DirectionCondition, np.ndarray]] = {}
    rows = []
    pair_names = sorted({item.pair_name for item in conditions if item.pair_name})
    for pair_index, pair in enumerate(pair_names):
        positive_index = next(
            index
            for index, item in enumerate(conditions)
            if item.pair_name == pair and item.sign > 0
        )
        negative_index = next(
            index
            for index, item in enumerate(conditions)
            if item.pair_name == pair and item.sign < 0
        )
        item = conditions[positive_index]
        positive = _load_condition(
            output_dir,
            condition_index=positive_index,
            samples=samples,
            world_size=world_size,
        ).astype(np.float64)
        negative = _load_condition(
            output_dir,
            condition_index=negative_index,
            samples=samples,
            world_size=world_size,
        ).astype(np.float64)
        derivative = 0.5 * (positive - negative) / item.gamma
        response = sample_rms(derivative)
        unit_norm = 0.5 * (unit_norms[:, positive_index] + unit_norms[:, negative_index])
        gain = response / np.maximum(unit_norm, 1e-30)
        by_pair[pair] = (item, gain)
        low, high = bootstrap_mean_interval(gain, repeats=repeats, seed=seed + 1009 * pair_index)
        rows.append(
            {
                "pair_name": pair,
                "family": item.family,
                "step": item.step,
                "time": float(grid[item.step]),
                "probe_index": item.probe_index,
                "samples": samples,
                "gain_mean": float(gain.mean()),
                "gain_ci_low": low,
                "gain_ci_high": high,
                "gain_median": float(np.median(gain)),
                "gain_q95": float(np.quantile(gain, 0.95)),
            }
        )
    gain_frame = pd.DataFrame(rows).sort_values(["step", "family", "probe_index"])
    gain_frame.to_csv(output_dir / "direction_gain.csv", index=False)

    specificity_rows = []
    active_steps = sorted(
        {int(item.step) for item in conditions if item.step is not None}
    )
    for step_index, step in enumerate(active_steps):
        ig = by_pair[f"step_{step:03d}_ig"][1]
        random = np.stack(
            [
                gain
                for item, gain in by_pair.values()
                if item.step == step and item.family == "random"
            ],
            axis=1,
        )
        random_mean = random.mean(axis=1)
        ratio = ig / np.maximum(random_mean, 1e-30)
        low, high = bootstrap_mean_interval(
            ratio, repeats=repeats, seed=seed + 5003 * step_index
        )
        specificity_rows.append(
            {
                "step": step,
                "time": float(grid[step]),
                "samples": samples,
                "probe_count": random.shape[1],
                "ig_gain_mean": float(ig.mean()),
                "random_gain_mean": float(random_mean.mean()),
                "ig_over_random_mean": float(ratio.mean()),
                "ig_over_random_ci_low": low,
                "ig_over_random_ci_high": high,
                "ig_over_random_median": float(np.median(ratio)),
                "fraction_ig_above_random_mean": float(np.mean(ig > random_mean)),
            }
        )
    specificity = pd.DataFrame(specificity_rows)
    specificity.to_csv(output_dir / "direction_specificity.csv", index=False)
    diagnostic_rows = []
    for index, item in enumerate(conditions):
        if item.family not in {"ig", "random"}:
            continue
        values = diagnostics[:, index]
        diagnostic_rows.append(
            {
                "condition": item.name,
                "family": item.family,
                "step": item.step,
                "probe_index": item.probe_index,
                "absolute_cosine_max": float(np.nanmax(np.abs(values[:, 0]))),
                "rms_ratio_min": float(np.nanmin(values[:, 1])),
                "rms_ratio_max": float(np.nanmax(values[:, 1])),
            }
        )
    pd.DataFrame(diagnostic_rows).to_csv(
        output_dir / "direction_construction_audit.csv", index=False
    )
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(specificity.time, specificity.ig_gain_mean, "o-", label="IG gap")
    axis.plot(
        specificity.time,
        specificity.random_gain_mean,
        "s-",
        label="matched random orthogonal",
    )
    axis.invert_xaxis()
    axis.set(
        title="SiT endpoint gain by intervention direction",
        xlabel="solver time t",
        ylabel="endpoint derivative / injected norm",
    )
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "direction_specificity.png", dpi=180)
    plt.close(figure)
    print(specificity.to_string(index=False), flush=True)


def main() -> None:
    args = parse_args()
    counts = (
        args.samples,
        args.num_steps,
        args.probe_count,
        args.per_rank_batch,
        args.condition_group_size,
        args.bootstrap_repeats,
        args.log_every_samples,
    )
    if any(value <= 0 for value in counts):
        raise ValueError("counts must be positive")
    if args.per_rank_batch * args.condition_group_size > 16:
        raise ValueError("effective model batch must not exceed 16")
    conditions = build_conditions(
        tuple(args.pulse_steps),
        gamma=args.gamma,
        probe_count=args.probe_count,
        num_steps=args.num_steps,
    )
    grid = torch.linspace(1.0, 0.0, args.num_steps + 1, dtype=torch.float64)
    dist.init_process_group("nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output_dir = args.output_dir.expanduser().resolve()
    if rank == 0: output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint_hash = file_sha256(checkpoint_path) if rank == 0 else ""
    objects = [checkpoint_hash]
    dist.broadcast_object_list(objects, src=0)
    label_seed = int(args.seed if args.label_seed is None else args.label_seed)
    manifest = {
        "protocol": PROTOCOL,
        "status": "running",
        "training": False,
        "repo": str(args.repo.expanduser().resolve()),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": objects[0],
        "model_name": args.model_name,
        "encoder_depth": args.encoder_depth,
        "state_key": args.state_key,
        "samples": args.samples,
        "seed": args.seed,
        "world_size": world_size,
        "precision": "fp32",
        "tf32": False,
        "latent_size": [4, 32, 32],
        "num_steps": args.num_steps,
        "solver_grid": grid.tolist(),
        "gamma": args.gamma,
        "probe_count": args.probe_count,
        "conditions": [asdict(item) for item in conditions],
        "label_mode": "random_without_replacement",
        "label_seed": label_seed,
        "same_noise_and_labels_across_conditions": True,
        "random_directions_orthogonal_to_gap": True,
        "random_directions_rms_matched_to_gap": True,
        "cfg_scale": 1.0,
        "sampler": "official deterministic Euler ODE",
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
                "world_size",
                "latent_size",
                "num_steps",
                "solver_grid",
                "gamma",
                "probe_count",
                "conditions",
                "label_seed",
            )
            changed = [
                key for key in keys if current.get(key) != manifest.get(key)
            ]
            if changed:
                raise RuntimeError(
                    f"cannot resume changed specificity protocol: {changed}"
                )
        else:
            _atomic_json(manifest_path, manifest)
            labels = build_validation_labels(
                args.samples,
                1000,
                mode="random_without_replacement",
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
    unit_norm = _open_memmap(
        output_dir / f"unit_injected_norm_rank{rank:02d}.npy",
        shape=(len(conditions), len(local_ids), 1),
        dtype=np.float64,
    )
    direction_audit = _open_memmap(
        output_dir / f"direction_diagnostics_rank{rank:02d}.npy",
        shape=(len(conditions), len(local_ids), 2),
        dtype=np.float64,
    )
    progress_path = output_dir / f"progress_rank{rank:02d}.npy"
    existed = progress_path.is_file()
    progress = _open_memmap(
        progress_path, shape=(len(local_ids),), dtype=np.bool_
    )
    if not existed:
        progress.fill(False)
        progress.flush()
    for start in range(0, len(local_ids), args.per_rank_batch):
        stop = min(start + args.per_rank_batch, len(local_ids))
        if bool(np.asarray(progress[start:stop]).all()): continue
        if bool(np.asarray(progress[start:stop]).any()):
            raise RuntimeError("partially complete batch cannot be resumed safely")
        ids = local_ids[start:stop]
        noise = deterministic_noise(ids, (4, 32, 32), seed=args.seed).to(device)
        batch_labels = torch.from_numpy(labels[ids]).to(
            device=device, dtype=torch.long
        )
        ranges = [(0, 1)] + [
            (begin, min(begin + args.condition_group_size, len(conditions)))
            for begin in range(1, len(conditions), args.condition_group_size)
        ]
        explicit_baseline: torch.Tensor | None = None
        for begin, end in ranges:
            endpoint, norms, diagnostics = simulate_group(
                model=model,
                noise=noise,
                labels=batch_labels,
                sample_ids=ids,
                grid=grid,
                conditions=conditions[begin:end],
                seed=args.seed + 97,
            )
            endpoints[begin:end, start:stop] = endpoint.cpu().numpy()
            unit_norm[begin:end, start:stop, 0] = norms.cpu().numpy()
            direction_audit[begin:end, start:stop] = diagnostics.cpu().numpy()
            if begin == 0:
                explicit_baseline = endpoint[0]
        check_path = output_dir / f"official_baseline_check_rank{rank:02d}.json"
        if not check_path.is_file():
            official = official_baseline(model, noise, batch_labels, args.num_steps)
            delta = explicit_baseline.double() - official.double()
            check = {
                "rank": rank,
                "rms": float(delta.square().mean().sqrt().cpu()),
                "maximum_absolute": float(delta.abs().max().cpu()),
            }
            _atomic_json(check_path, check)
            if check["rms"] > 1e-7 or check["maximum_absolute"] > 1e-5:
                raise RuntimeError(f"explicit sampler mismatch: {check}")
        endpoints.flush()
        unit_norm.flush()
        direction_audit.flush()
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
        analyze_results(
            output_dir=output_dir,
            conditions=conditions,
            grid=grid,
            samples=args.samples,
            world_size=world_size,
            repeats=args.bootstrap_repeats,
            seed=args.seed + 17,
        )
        final = json.loads(manifest_path.read_text(encoding="utf-8"))
        final["status"] = "complete"
        final["model_metadata"] = metadata
        _atomic_json(manifest_path, final)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
