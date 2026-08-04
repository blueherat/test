"""Frozen endpoint-dynamics diagnostics for the official SiT-XL/2+IG model.

The experiment uses the official deterministic Euler ODE and applies small
signed interventions around the full-head baseline.  It supports single-step
pulses, equal-step windows, and four-corner interactions between two windows.
Unlike RAEv2 clean-prediction sampling, SiT predicts velocity directly, so a
one-step intervention enters as ``dt * gamma * (full - base)`` without ``1/t``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
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
    _load_small_shards,
    _open_memmap,
    bootstrap_mean_interval,
    build_validation_labels,
    deterministic_noise,
    equal_step_ranges,
)


PROTOCOL = "sit_ig_endpoint_dynamics_v1"
STAT_FIELDS = ("unit_injected_energy", "actual_injected_energy", "unit_coherent_rms")


@dataclass(frozen=True)
class Schedule:
    name: str
    family: str
    gamma: float
    segments: tuple[tuple[int, int, int], ...]
    pair_name: str | None = None
    pulse_step: int | None = None
    window_index: int | None = None
    left_window: int | None = None
    right_window: int | None = None

    def coefficient(self, step: int) -> float:
        return float(self.gamma) * sum(
            sign for start, end, sign in self.segments if start <= int(step) < end
        )

    def active(self, step: int) -> bool:
        return any(start <= int(step) < end for start, end, _ in self.segments)


def parse_int_list(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return result


def parse_float_list(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated floats") from error
    if not result or any(not math.isfinite(item) or item <= 0 for item in result):
        raise argparse.ArgumentTypeError("finite positive values are required")
    return result


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
    parser.add_argument("--per-rank-batch", type=int, default=1)
    parser.add_argument("--condition-group-size", type=int, default=8)
    parser.add_argument("--pulse-steps", type=parse_int_list, default=(5, 15, 25, 35, 45, 49))
    parser.add_argument("--pulse-gammas", type=parse_float_list, default=(0.01, 0.05))
    parser.add_argument("--window-count", type=int, default=5)
    parser.add_argument("--window-gamma", type=float, default=0.01)
    parser.add_argument("--interaction-windows", type=parse_int_list, default=(0, 2, 4))
    parser.add_argument("--skip-pulses", action="store_true")
    parser.add_argument("--skip-windows", action="store_true")
    parser.add_argument("--skip-interactions", action="store_true")
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument(
        "--label-mode",
        choices=("sequential", "random_without_replacement"),
        default="random_without_replacement",
    )
    parser.add_argument("--label-seed", type=int)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--log-every-samples", type=int, default=4)
    return parser.parse_args()


def build_schedules(
    *,
    num_steps: int,
    pulse_steps: Iterable[int],
    pulse_gammas: Iterable[float],
    windows: tuple[tuple[int, int], ...],
    window_gamma: float,
    interaction_windows: Iterable[int],
    include_pulses: bool,
    include_windows: bool,
    include_interactions: bool,
) -> tuple[Schedule, ...]:
    result = [Schedule("baseline", "baseline", 0.0, ())]
    pulses = tuple(int(step) for step in pulse_steps)
    gammas = tuple(float(value) for value in pulse_gammas)
    if any(step < 0 or step >= num_steps for step in pulses):
        raise ValueError("pulse step is out of range")
    if any(value <= 0 or not math.isfinite(value) for value in (*gammas, window_gamma)):
        raise ValueError("gammas must be finite and positive")
    if include_pulses:
        for step in pulses:
            for gamma in gammas:
                pair = f"pulse_{step:03d}_g{gamma:.6f}".replace(".", "p")
                result.extend(
                    (
                        Schedule(pair + "_pos", "pulse", gamma, ((step, step + 1, 1),), pair, pulse_step=step),
                        Schedule(pair + "_neg", "pulse", gamma, ((step, step + 1, -1),), pair, pulse_step=step),
                    )
                )
    if include_windows or include_interactions:
        for index, (start, end) in enumerate(windows):
            pair = f"window_{index:02d}_{start:03d}_{end:03d}"
            result.extend(
                (
                    Schedule(pair + "_pos", "window", window_gamma, ((start, end, 1),), pair, window_index=index),
                    Schedule(pair + "_neg", "window", window_gamma, ((start, end, -1),), pair, window_index=index),
                )
            )
    selected = tuple(int(index) for index in interaction_windows)
    if include_interactions:
        if len(selected) < 2 or len(set(selected)) != len(selected):
            raise ValueError("interaction windows must contain at least two unique indices")
        if any(index < 0 or index >= len(windows) for index in selected):
            raise ValueError("interaction window is out of range")
        for position, left in enumerate(selected):
            for right in selected[position + 1 :]:
                left_range, right_range = windows[left], windows[right]
                for left_sign, right_sign, suffix in ((1, 1, "pp"), (1, -1, "pm"), (-1, 1, "mp"), (-1, -1, "mm")):
                    result.append(
                        Schedule(
                            f"interaction_w{left}_w{right}_{suffix}",
                            "interaction",
                            window_gamma,
                            ((left_range[0], left_range[1], left_sign), (right_range[0], right_range[1], right_sign)),
                            left_window=left,
                            right_window=right,
                        )
                    )
    if not include_windows:
        result = [item for item in result if item.family != "window" or item.window_index in selected]
    names = [item.name for item in result]
    if len(set(names)) != len(names) or len(result) == 1:
        raise ValueError("schedules must be unique and contain an intervention")
    return tuple(result)


def simulate_group(
    *,
    model: torch.nn.Module,
    noise: torch.Tensor,
    labels: torch.Tensor,
    grid: torch.Tensor,
    schedules: tuple[Schedule, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    condition_count = len(schedules)
    batch_size = len(noise)
    state = noise.unsqueeze(0).expand(condition_count, *noise.shape).reshape(
        condition_count * batch_size, *noise.shape[1:]
    ).double().contiguous()
    contexts = labels.unsqueeze(0).expand(condition_count, batch_size).reshape(-1).contiguous()
    unit_energy = torch.zeros(condition_count, batch_size, device=state.device, dtype=torch.float64)
    coherent = torch.zeros_like(state).reshape(condition_count, batch_size, *state.shape[1:])
    with torch.inference_mode():
        for step in range(len(grid) - 1):
            time = float(grid[step])
            dt = float(grid[step + 1] - grid[step])
            times = torch.full((len(state),), time, device=state.device, dtype=torch.float32)
            output = model(state.float(), times, contexts)
            full, base = output[0].double(), output[1].double()
            full = full.reshape(condition_count, batch_size, *state.shape[1:])
            base = base.reshape_as(full)
            gap = full - base
            coefficients = torch.tensor(
                [item.coefficient(step) for item in schedules],
                device=state.device,
                dtype=torch.float64,
            ).reshape((condition_count, 1) + (1,) * (full.ndim - 2))
            velocity = full + coefficients * gap
            state = (state.reshape_as(full) + dt * velocity).reshape(
                condition_count * batch_size, *state.shape[1:]
            )
            active = torch.tensor([item.active(step) for item in schedules], device=state.device, dtype=torch.bool)
            unit_impulse = dt * gap
            unit_energy += unit_impulse.flatten(2).square().mean(2) * active[:, None]
            coherent += unit_impulse * active.reshape((condition_count, 1) + (1,) * (unit_impulse.ndim - 2))
    gamma = torch.tensor([abs(item.gamma) for item in schedules], device=state.device, dtype=torch.float64)[:, None]
    coherent_rms = coherent.flatten(2).square().mean(2).sqrt()
    stats = torch.stack((unit_energy, gamma.square() * unit_energy, coherent_rms), dim=2)
    return state.reshape(condition_count, batch_size, *state.shape[1:]).float(), stats


def official_baseline(
    model: torch.nn.Module,
    noise: torch.Tensor,
    labels: torch.Tensor,
    num_steps: int,
) -> torch.Tensor:
    from samplers import euler_sampler

    with torch.inference_mode():
        return euler_sampler(
            model=model,
            latents=noise,
            y=labels,
            num_steps=num_steps,
            heun=False,
            cfg_scale=1.0,
        ).float()


def sample_rms(value: np.ndarray) -> np.ndarray:
    flat = np.asarray(value, dtype=np.float64).reshape(len(value), -1)
    return np.sqrt(np.mean(np.square(flat), axis=1))


def sample_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64).reshape(len(left), -1)
    right = np.asarray(right, dtype=np.float64).reshape(len(right), -1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return np.sum(left * right, axis=1) / np.maximum(denominator, 1e-30)


def interaction_metrics(
    baseline: np.ndarray,
    a_positive: np.ndarray,
    a_negative: np.ndarray,
    b_positive: np.ndarray,
    b_negative: np.ndarray,
    pp: np.ndarray,
    pm: np.ndarray,
    mp: np.ndarray,
    mm: np.ndarray,
    *,
    gamma: float,
) -> dict[str, np.ndarray]:
    values = [
        np.asarray(value, dtype=np.float64)
        for value in (
            baseline,
            a_positive,
            a_negative,
            b_positive,
            b_negative,
            pp,
            pm,
            mp,
            mm,
        )
    ]
    baseline, a_positive, a_negative, b_positive, b_negative, pp, pm, mp, mm = values
    d_a = (a_positive - a_negative) / (2 * gamma)
    d_b = (b_positive - b_negative) / (2 * gamma)
    d_joint = (pp - mm) / (2 * gamma)
    d_sum = d_a + d_b
    scale = 0.5 * (sample_rms(d_joint) + sample_rms(d_sum))
    mixed_effect = 0.25 * (pp - pm - mp + mm)
    joint_odd = 0.5 * (pp - mm)
    return {
        "derivative_relative_error": sample_rms(d_joint - d_sum) / np.maximum(scale, 1e-30),
        "derivative_cosine": sample_cosine(d_joint, d_sum),
        "mixed_over_joint": sample_rms(mixed_effect) / np.maximum(sample_rms(joint_odd), 1e-30),
        "positive_additivity_error": sample_rms(
            pp - a_positive - b_positive + baseline
        )
        / np.maximum(sample_rms(pp - baseline), 1e-30),
        "negative_additivity_error": sample_rms(
            mm - a_negative - b_negative + baseline
        )
        / np.maximum(sample_rms(mm - baseline), 1e-30),
    }


def summarize(values: np.ndarray, *, repeats: int, seed: int) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    low, high = bootstrap_mean_interval(values, repeats=repeats, seed=seed)
    median = float(np.median(values))
    q95 = float(np.quantile(values, 0.95))
    return {
        "mean": float(values.mean()),
        "ci_low": low,
        "ci_high": high,
        "median": median,
        "q95": q95,
        "maximum": float(values.max()),
        "q95_over_median": q95 / max(median, 1e-30),
    }


def analyze_results(
    *,
    output_dir: Path,
    schedules: tuple[Schedule, ...],
    grid: torch.Tensor,
    windows: tuple[tuple[int, int], ...],
    samples: int,
    world_size: int,
    repeats: int,
    seed: int,
) -> None:
    baseline = _load_condition(output_dir, condition_index=0, samples=samples, world_size=world_size)
    stats = _load_small_shards(
        output_dir,
        filename="injection_stats_rank{rank:02d}.npy",
        samples=samples,
        world_size=world_size,
    )
    by_name = {item.name: index for index, item in enumerate(schedules)}
    signed_rows: list[dict[str, Any]] = []
    derivatives: dict[tuple[int, float], np.ndarray] = {}
    for pair_index, pair in enumerate(sorted({item.pair_name for item in schedules if item.pair_name})):
        positive_index = next(
            index
            for index, item in enumerate(schedules)
            if item.pair_name == pair and item.segments[0][2] > 0
        )
        negative_index = next(
            index
            for index, item in enumerate(schedules)
            if item.pair_name == pair and item.segments[0][2] < 0
        )
        item = schedules[positive_index]
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
        even = 0.5 * (positive + negative) - baseline.astype(np.float64)
        response = sample_rms(derivative)
        even_rms = sample_rms(even)
        odd_rms = item.gamma * response
        unit_norm = np.sqrt(np.maximum(0.5 * (stats[:, positive_index, 0] + stats[:, negative_index, 0]), 0.0))
        response_summary = summarize(
            response,
            repeats=repeats,
            seed=seed + 1009 * pair_index,
        )
        row = {
            "pair_name": pair,
            "family": item.family,
            "gamma": item.gamma,
            "pulse_step": item.pulse_step,
            "window_index": item.window_index,
            "start_time": float(grid[item.pulse_step])
            if item.pulse_step is not None
            else float(grid[windows[item.window_index][0]]),
            "active_steps": 1
            if item.pulse_step is not None
            else windows[item.window_index][1] - windows[item.window_index][0],
            **{f"response_{key}": value for key, value in response_summary.items()},
            "even_over_odd_mean": float(
                np.mean(even_rms / np.maximum(odd_rms, 1e-30))
            ),
            "propagation_gain_mean": float(
                np.mean(response / np.maximum(unit_norm, 1e-30))
            ),
        }
        signed_rows.append(row)
        if item.family == "pulse":
            derivatives[(int(item.pulse_step), float(item.gamma))] = derivative
    signed = pd.DataFrame(signed_rows).sort_values(["family", "pulse_step", "window_index", "gamma"])
    signed.to_csv(output_dir / "signed_response.csv", index=False)

    linearity_rows = []
    pulse_gammas = sorted({gamma for _, gamma in derivatives})
    for step_index, step in enumerate(sorted({step for step, _ in derivatives})):
        if len(pulse_gammas) != 2:
            break
        small, large = (derivatives[(step, gamma)] for gamma in pulse_gammas)
        small_rms, large_rms = sample_rms(small), sample_rms(large)
        relative = sample_rms(small - large) / np.maximum(0.5 * (small_rms + large_rms), 1e-30)
        cosine = sample_cosine(small, large)
        relative_summary = summarize(
            relative,
            repeats=repeats,
            seed=seed + 2003 * step_index,
        )
        linearity_rows.append(
            {
                "step": step,
                "time": float(grid[step]),
                "small_gamma": pulse_gammas[0],
                "large_gamma": pulse_gammas[1],
                **{
                    f"relative_error_{key}": value
                    for key, value in relative_summary.items()
                },
                "derivative_cosine_mean": float(cosine.mean()),
                "amplitude_ratio_mean": float(
                    np.mean(small_rms / np.maximum(large_rms, 1e-30))
                ),
            }
        )
    linearity = pd.DataFrame(linearity_rows)
    linearity.to_csv(output_dir / "cross_gamma_linearity.csv", index=False)

    interaction_rows = []
    selected = sorted(
        {item.left_window for item in schedules if item.left_window is not None}
        | {item.right_window for item in schedules if item.right_window is not None}
    )
    for pair_index, left in enumerate(selected):
        for right in selected[pair_index + 1 :]:
            def condition(name: str) -> np.ndarray:
                return _load_condition(
                    output_dir,
                    condition_index=by_name[name],
                    samples=samples,
                    world_size=world_size,
                )

            left_pair = next(
                item.pair_name
                for item in schedules
                if item.family == "window" and item.window_index == left
            )
            right_pair = next(
                item.pair_name
                for item in schedules
                if item.family == "window" and item.window_index == right
            )
            interaction_gamma = next(
                item.gamma for item in schedules if item.family == "interaction"
            )
            values = interaction_metrics(
                baseline,
                condition(left_pair + "_pos"),
                condition(left_pair + "_neg"),
                condition(right_pair + "_pos"),
                condition(right_pair + "_neg"),
                condition(f"interaction_w{left}_w{right}_pp"),
                condition(f"interaction_w{left}_w{right}_pm"),
                condition(f"interaction_w{left}_w{right}_mp"),
                condition(f"interaction_w{left}_w{right}_mm"),
                gamma=interaction_gamma,
            )
            row: dict[str, Any] = {
                "left_window": left,
                "right_window": right,
                "left_steps": f"{windows[left][0]}:{windows[left][1]}",
                "right_steps": f"{windows[right][0]}:{windows[right][1]}",
            }
            for metric_index, (name, per_sample) in enumerate(values.items()):
                metric_summary = summarize(
                    per_sample,
                    repeats=repeats,
                    seed=seed + 3001 * pair_index + metric_index,
                )
                row.update(
                    {
                        f"{name}_{key}": value
                        for key, value in metric_summary.items()
                    }
                )
            interaction_rows.append(row)
    interaction = pd.DataFrame(interaction_rows)
    interaction.to_csv(output_dir / "window_interaction.csv", index=False)

    figure, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    for gamma, part in signed[signed.family.eq("pulse")].groupby("gamma"):
        axes[0].plot(part.start_time, part.propagation_gain_mean, "o-", label=f"gamma={gamma:g}")
        axes[1].plot(part.start_time, part.response_q95_over_median, "o-", label=f"gamma={gamma:g}")
    if not linearity.empty:
        axes[2].plot(linearity.time, linearity.relative_error_mean, "o-", label="relative derivative error")
        axes[2].plot(linearity.time, 1 - linearity.derivative_cosine_mean, "s-", label="1 - cosine")
    titles = (
        "Single-step propagation gain",
        "Endpoint response tail",
        "Cross-gamma nonlinearity",
    )
    ylabels = (
        "endpoint derivative / injected norm",
        "q95 / median",
        "relative value",
    )
    for axis, title, ylabel in zip(axes, titles, ylabels):
        axis.invert_xaxis()
        axis.set(title=title, xlabel="solver time t", ylabel=ylabel)
        axis.grid(alpha=0.2)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "sit_ig_endpoint_dynamics.png", dpi=180)
    plt.close(figure)
    print(signed.to_string(index=False), flush=True)
    if not interaction.empty:
        print(interaction.to_string(index=False), flush=True)


def main() -> None:
    args = parse_args()
    counts = (
        args.samples,
        args.num_steps,
        args.per_rank_batch,
        args.condition_group_size,
        args.window_count,
        args.bootstrap_repeats,
        args.log_every_samples,
    )
    if any(value <= 0 for value in counts):
        raise ValueError("counts must be positive")
    if args.per_rank_batch * args.condition_group_size > 16:
        raise ValueError("per-rank batch times condition-group-size must not exceed 16")
    dist.init_process_group("nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    grid = torch.linspace(1.0, 0.0, args.num_steps + 1, dtype=torch.float64)
    windows = equal_step_ranges(args.num_steps, args.window_count)
    schedules = build_schedules(
        num_steps=args.num_steps,
        pulse_steps=args.pulse_steps,
        pulse_gammas=args.pulse_gammas,
        windows=windows,
        window_gamma=args.window_gamma,
        interaction_windows=args.interaction_windows,
        include_pulses=not args.skip_pulses,
        include_windows=not args.skip_windows,
        include_interactions=not args.skip_interactions,
    )
    output_dir = args.output_dir.expanduser().resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
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
        "windows": windows,
        "schedules": [asdict(item) for item in schedules],
        "label_mode": args.label_mode,
        "label_seed": label_seed,
        "same_noise_and_labels_across_conditions": True,
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
                "windows",
                "schedules",
                "label_mode",
                "label_seed",
            )
            changed = [key for key in keys if current.get(key) != manifest.get(key)]
            if changed:
                raise RuntimeError(f"cannot resume changed SiT protocol: {changed}")
        else:
            _atomic_json(manifest_path, manifest)
            labels = build_validation_labels(args.samples, 1000, mode=args.label_mode, seed=label_seed)
            np.savez_compressed(
                output_dir / "sample_protocol.npz",
                sample_ids=np.arange(args.samples, dtype=np.int64),
                labels=labels,
            )
    dist.barrier()
    labels = np.load(output_dir / "sample_protocol.npz")["labels"].astype(np.int64)
    local_ids = np.arange(rank, args.samples, world_size, dtype=np.int64)
    model, model_metadata = load_model(
        repo=args.repo,
        checkpoint_path=checkpoint_path,
        model_name=args.model_name,
        encoder_depth=args.encoder_depth,
        state_key=args.state_key,
        device=device,
    )
    endpoints = _open_memmap(
        output_dir / f"endpoints_rank{rank:02d}.npy",
        shape=(len(schedules), len(local_ids), 4, 32, 32),
        dtype=np.float32,
    )
    injection = _open_memmap(
        output_dir / f"injection_stats_rank{rank:02d}.npy",
        shape=(len(schedules), len(local_ids), len(STAT_FIELDS)),
        dtype=np.float64,
    )
    progress_path = output_dir / f"progress_rank{rank:02d}.npy"
    progress_existed = progress_path.is_file()
    progress = _open_memmap(
        progress_path, shape=(len(local_ids),), dtype=np.bool_
    )
    if not progress_existed:
        progress.fill(False); progress.flush()
    for start in range(0, len(local_ids), args.per_rank_batch):
        stop = min(start + args.per_rank_batch, len(local_ids))
        if bool(np.asarray(progress[start:stop]).all()): continue
        if bool(np.asarray(progress[start:stop]).any()):
            raise RuntimeError("partially complete batch cannot be resumed safely")
        ids = local_ids[start:stop]
        noise = deterministic_noise(ids, (4, 32, 32), seed=args.seed).to(device)
        batch_labels = torch.from_numpy(labels[ids]).to(device=device, dtype=torch.long)
        ranges = [(0, 1)] + [
            (begin, min(begin + args.condition_group_size, len(schedules)))
            for begin in range(1, len(schedules), args.condition_group_size)
        ]
        explicit_baseline: torch.Tensor | None = None
        for begin, end in ranges:
            endpoint, stats = simulate_group(
                model=model,
                noise=noise,
                labels=batch_labels,
                grid=grid,
                schedules=schedules[begin:end],
            )
            endpoints[begin:end, start:stop] = endpoint.cpu().numpy()
            injection[begin:end, start:stop] = stats.cpu().numpy()
            if begin == 0:
                explicit_baseline = endpoint[0]
        check_path = output_dir / f"official_baseline_check_rank{rank:02d}.json"
        if not check_path.is_file():
            official = official_baseline(model, noise, batch_labels, args.num_steps)
            delta = explicit_baseline.double() - official.double()
            check = {
                "rank": rank,
                "samples_checked": len(noise),
                "rms": float(delta.square().mean().sqrt().cpu()),
                "maximum_absolute": float(delta.abs().max().cpu()),
            }
            _atomic_json(check_path, check)
            if check["rms"] > 1e-7 or check["maximum_absolute"] > 1e-5:
                raise RuntimeError(f"explicit sampler mismatch: {check}")
        endpoints.flush()
        injection.flush()
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
            schedules=schedules,
            grid=grid,
            windows=windows,
            samples=args.samples,
            world_size=world_size,
            repeats=args.bootstrap_repeats,
            seed=args.seed + 17,
        )
        final = json.loads(manifest_path.read_text(encoding="utf-8"))
        final["status"] = "complete"
        final["model_metadata"] = model_metadata
        final["stat_fields"] = list(STAT_FIELDS)
        _atomic_json(manifest_path, final)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
