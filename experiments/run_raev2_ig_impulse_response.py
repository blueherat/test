"""Measure step-local and equal-step RAEv2 internal-guidance responses.

This is a frozen-model system-identification experiment.  Every intervention
uses the official shifted Euler grid and the exact clean-prediction rule

    guided = full + gamma * (full - base).

Unlike broad intervals in physical ``t``, equal-step windows remove the large
67/18/8/4/2 active-step imbalance induced by the shifted solver grid.  The
runner also records injection energy on the same samples and trajectories used
for endpoint evaluation.  It never trains or mutates a checkpoint.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, Iterable

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
from experiments.raev2_training_core import (  # noqa: E402
    file_sha256,
    split_internal_guidance_output,
    validate_full_stage2_checkpoint,
)
from experiments.run_raev2_distribution_auc import (  # noqa: E402
    build_requested_labels,
    load_config,
)
from stage2.transport import create_sampler, create_transport  # noqa: E402
from utils.guidance_utils import forward_with_internalguidance  # noqa: E402
from utils.model_utils import instantiate_from_config  # noqa: E402


PROTOCOL = "raev2_ig_impulse_response_v1"
STAT_FIELDS = (
    "unit_injected_energy",
    "actual_injected_energy",
    "unit_coherent_rms",
    "actual_coherent_rms",
)
STEP_FIELDS = ("raw_gap_rms", "unit_impulse_rms")


@dataclass(frozen=True)
class Intervention:
    name: str
    family: str
    start_step: int
    end_step: int
    gamma: float
    pair_name: str | None = None

    def active(self, step: int) -> bool:
        return self.start_step <= int(step) < self.end_step


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
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--per-rank-batch", type=int, default=1)
    parser.add_argument("--condition-group-size", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument(
        "--pulse-gammas",
        type=parse_float_list,
        help="Optional positive gamma sweep for every selected single-step pulse.",
    )
    parser.add_argument(
        "--pulse-steps",
        type=parse_int_list,
        default=(10, 30, 50, 70, 90, 98),
    )
    parser.add_argument("--window-count", type=int, default=5)
    parser.add_argument("--skip-pulses", action="store_true")
    parser.add_argument("--skip-windows", action="store_true")
    parser.add_argument(
        "--window-gammas",
        type=parse_float_list,
        help="Optional positive gamma per equal-step window, for held-out equal-energy runs.",
    )
    parser.add_argument("--state-key", choices=("ema", "model"), default="ema")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument(
        "--label-mode",
        choices=("sequential", "random_without_replacement"),
        default="sequential",
    )
    parser.add_argument(
        "--label-seed",
        type=int,
        help="Label sampling seed; defaults to --seed.",
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--log-every-samples", type=int, default=2)
    parser.add_argument("--skip-official-baseline-check", action="store_true")
    parser.add_argument(
        "--dino-ckpt-dir",
        type=Path,
        default=Path("/home/zhoushunyu/data/eqvae/models/RAEv2/encoders/dinov3"),
    )
    parser.add_argument(
        "--dino-repo-dir",
        type=Path,
        default=Path("/home/zhoushunyu/data/eqvae/models/RAEv2/dinov3_repo"),
    )
    return parser.parse_args()


def equal_step_ranges(num_steps: int, window_count: int) -> tuple[tuple[int, int], ...]:
    if num_steps <= 0 or window_count <= 0 or window_count > num_steps:
        raise ValueError("window count must lie in [1, num_steps]")
    chunks = np.array_split(np.arange(num_steps, dtype=np.int64), window_count)
    ranges = tuple((int(chunk[0]), int(chunk[-1]) + 1) for chunk in chunks)
    if ranges[0][0] != 0 or ranges[-1][1] != num_steps:
        raise AssertionError("equal-step windows do not cover the solver")
    return ranges


def official_shifted_solver_grid(num_steps: int, shift: float) -> torch.Tensor:
    """Reproduce ``stage2.transport.sampler.Sampler.sample_ode`` exactly."""
    if num_steps <= 0 or shift <= 0:
        raise ValueError("num_steps and shift must be positive")
    grid = torch.linspace(1.0, 0.0, num_steps + 1)
    return shift * grid / (1.0 + (shift - 1.0) * grid)


def build_interventions(
    *,
    num_steps: int,
    pulse_steps: Iterable[int],
    window_count: int,
    gamma: float,
    pulse_gammas: Iterable[float] | None = None,
    window_gammas: Iterable[float] | None = None,
) -> tuple[Intervention, ...]:
    gamma = float(gamma)
    if not math.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be finite and positive")
    pulses = tuple(int(step) for step in pulse_steps)
    if len(set(pulses)) != len(pulses):
        raise ValueError("pulse steps must be unique")
    if any(step < 0 or step >= num_steps for step in pulses):
        raise ValueError("pulse steps must be valid Euler-step indices")
    result = [Intervention("baseline", "baseline", 0, 0, 0.0, None)]
    resolved_pulse_gammas = (
        tuple(float(value) for value in pulse_gammas)
        if pulse_gammas is not None
        else (gamma,)
    )
    if any(not math.isfinite(value) or value <= 0 for value in resolved_pulse_gammas):
        raise ValueError("pulse gammas must be finite and positive")
    for step in pulses:
        for pulse_gamma in resolved_pulse_gammas:
            pair = f"pulse_{step:03d}"
            if len(resolved_pulse_gammas) > 1:
                pair += f"_g{pulse_gamma:.6f}".replace(".", "p")
            result.extend(
                (
                    Intervention(
                        f"{pair}_pos", "pulse", step, step + 1, pulse_gamma, pair
                    ),
                    Intervention(
                        f"{pair}_neg", "pulse", step, step + 1, -pulse_gamma, pair
                    ),
                )
            )
    ranges = equal_step_ranges(num_steps, window_count)
    resolved_window_gammas = (
        tuple(float(value) for value in window_gammas)
        if window_gammas is not None
        else (gamma,) * len(ranges)
    )
    if len(resolved_window_gammas) != len(ranges):
        raise ValueError("window_gammas must contain one value per equal-step window")
    if any(not math.isfinite(value) or value <= 0 for value in resolved_window_gammas):
        raise ValueError("window gammas must be finite and positive")
    for index, ((start, end), window_gamma) in enumerate(
        zip(ranges, resolved_window_gammas)
    ):
        pair = f"window_{index:02d}_{start:03d}_{end:03d}"
        result.extend(
            (
                Intervention(pair + "_pos", "window", start, end, window_gamma, pair),
                Intervention(pair + "_neg", "window", start, end, -window_gamma, pair),
            )
        )
    names = [item.name for item in result]
    if len(set(names)) != len(names):
        raise AssertionError("intervention names are not unique")
    return tuple(result)


def autocast_context(precision: str):
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "fp32":
        return nullcontext()
    raise ValueError(f"unsupported precision: {precision}")


def guided_clean_prediction(
    full: torch.Tensor,
    base: torch.Tensor,
    coefficients: torch.Tensor,
) -> torch.Tensor:
    if full.shape != base.shape or coefficients.ndim != 1 or len(coefficients) != len(full):
        raise ValueError("full/base/coefficient shapes do not align")
    shape = (len(coefficients),) + (1,) * (full.ndim - 1)
    return full + coefficients.reshape(shape).to(full) * (full - base)


def euler_x_prediction_step(
    state: torch.Tensor,
    clean_prediction: torch.Tensor,
    *,
    time: float,
    step_size: float,
    t_eps: float,
) -> torch.Tensor:
    denominator = max(float(time), float(t_eps))
    return state - float(step_size) * (state - clean_prediction) / denominator


def bootstrap_mean_interval(
    values: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size < 2 or repeats <= 0:
        raise ValueError("bootstrap requires at least two values and positive repeats")
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(int(repeats), dtype=np.float64)
    for start in range(0, repeats, 256):
        count = min(256, repeats - start)
        indices = rng.integers(0, values.size, size=(count, values.size))
        estimates[start : start + count] = values[indices].mean(axis=1)
    low, high = np.quantile(estimates, (0.025, 0.975))
    return float(low), float(high)


def deterministic_noise(
    sample_ids: np.ndarray,
    latent_size: tuple[int, ...],
    *,
    seed: int,
) -> torch.Tensor:
    rows = []
    for sample_id in sample_ids.tolist():
        generator = torch.Generator(device="cpu").manual_seed(
            int(seed) + 1_000_003 * int(sample_id)
        )
        rows.append(torch.randn(latent_size, generator=generator, dtype=torch.float32))
    return torch.stack(rows)


def build_validation_labels(
    sample_count: int,
    num_classes: int,
    *,
    mode: str,
    seed: int,
) -> np.ndarray:
    if mode == "sequential":
        return build_requested_labels(sample_count, num_classes)
    if mode == "random_without_replacement":
        if sample_count > num_classes:
            raise ValueError("without-replacement labels require samples <= num_classes")
        return np.random.default_rng(int(seed)).permutation(num_classes)[:sample_count].astype(
            np.int64
        )
    raise ValueError(f"unsupported label mode: {mode}")


def simulate_group(
    *,
    model: torch.nn.Module,
    noise: torch.Tensor,
    labels: torch.Tensor,
    grid: torch.Tensor,
    interventions: tuple[Intervention, ...],
    t_eps: float,
    precision: str,
    record_baseline_steps: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    condition_count = len(interventions)
    batch_size = len(noise)
    state = (
        noise.unsqueeze(0)
        .expand(condition_count, *noise.shape)
        .reshape(condition_count * batch_size, *noise.shape[1:])
        .contiguous()
    )
    contexts = (
        labels.unsqueeze(0)
        .expand(condition_count, batch_size)
        .reshape(condition_count * batch_size)
        .contiguous()
    )
    unit_energy = torch.zeros(condition_count, batch_size, device=state.device, dtype=torch.float64)
    coherent = torch.zeros_like(state, dtype=torch.float32).reshape(
        condition_count, batch_size, *state.shape[1:]
    )
    baseline_steps = (
        torch.empty(batch_size, len(grid) - 1, len(STEP_FIELDS), dtype=torch.float64)
        if record_baseline_steps
        else None
    )
    baseline_index = next(
        (index for index, item in enumerate(interventions) if item.family == "baseline"),
        None,
    )
    with torch.inference_mode():
        for step in range(len(grid) - 1):
            time = float(grid[step])
            next_time = float(grid[step + 1])
            step_size = time - next_time
            time_batch = torch.full((len(state),), time, device=state.device)
            with autocast_context(precision):
                output = model(state, time_batch, context=contexts, attn_mask=None)
            full, base = split_internal_guidance_output(output)
            if base is None:
                raise RuntimeError("configured checkpoint does not expose an IG base head")
            full = full.float().reshape(condition_count, batch_size, *state.shape[1:])
            base = base.float().reshape_as(full)
            gap = full - base
            active = torch.tensor(
                [item.active(step) for item in interventions],
                device=state.device,
                dtype=torch.bool,
            )
            gamma = torch.tensor(
                [item.gamma if item.active(step) else 0.0 for item in interventions],
                device=state.device,
                dtype=torch.float32,
            )
            guided = guided_clean_prediction(
                full.reshape_as(state),
                base.reshape_as(state),
                gamma.repeat_interleave(batch_size),
            ).reshape_as(full)
            state = euler_x_prediction_step(
                state.reshape_as(full),
                guided,
                time=time,
                step_size=step_size,
                t_eps=t_eps,
            ).reshape(condition_count * batch_size, *state.shape[1:])

            unit_impulse = (step_size / max(time, t_eps)) * gap
            flat = unit_impulse.flatten(2)
            step_energy = flat.square().mean(dim=2).double()
            unit_energy += step_energy * active[:, None]
            coherent += unit_impulse * active.reshape(
                (condition_count, 1) + (1,) * (unit_impulse.ndim - 2)
            )
            if baseline_steps is not None and baseline_index is not None:
                baseline_gap = gap[baseline_index].flatten(1)
                baseline_impulse = unit_impulse[baseline_index].flatten(1)
                baseline_steps[:, step, 0] = baseline_gap.square().mean(1).sqrt().cpu().double()
                baseline_steps[:, step, 1] = (
                    baseline_impulse.square().mean(1).sqrt().cpu().double()
                )

    gamma_abs = torch.tensor(
        [abs(item.gamma) for item in interventions], dtype=torch.float64, device=state.device
    )[:, None]
    coherent_rms = coherent.flatten(2).square().mean(2).sqrt().double()
    stats = torch.stack(
        (
            unit_energy,
            gamma_abs.square() * unit_energy,
            coherent_rms,
            gamma_abs * coherent_rms,
        ),
        dim=2,
    )
    endpoint = state.reshape(condition_count, batch_size, *state.shape[1:]).float()
    return endpoint, stats, baseline_steps


def official_baseline_endpoint(
    *,
    model: torch.nn.Module,
    noise: torch.Tensor,
    labels: torch.Tensor,
    config: Any,
    shift: float,
    precision: str,
) -> torch.Tensor:
    transport = create_transport(config=config.transport, time_dist_shift=shift)
    sampler = create_sampler(transport, guidance_config=config.guidance)
    sample_fn = sampler.sample_ode(**dataclasses.asdict(config.sampler))
    model_fn = partial(forward_with_internalguidance, model)
    doubled_noise = torch.cat((noise, noise), dim=0)
    null_labels = torch.full_like(labels, int(config.misc.num_classes))
    context = torch.cat((labels, null_labels), dim=0)
    interval = (float(config.guidance.ig.t_min), float(config.guidance.ig.t_max))
    with torch.inference_mode(), autocast_context(precision):
        endpoint = sample_fn(
            doubled_noise,
            model_fn,
            context=context,
            attn_mask=None,
            ig_scale=1.0,
            ig_interval=interval,
        )[-1]
    return endpoint.chunk(2, dim=0)[0].float()


def _open_memmap(path: Path, *, shape: tuple[int, ...], dtype: Any) -> np.memmap:
    if path.is_file():
        result = np.lib.format.open_memmap(path, mode="r+")
        if result.shape != shape or result.dtype != np.dtype(dtype):
            raise RuntimeError(f"cannot resume incompatible memmap: {path}")
        return result
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _load_condition(
    output_dir: Path,
    *,
    condition_index: int,
    samples: int,
    world_size: int,
) -> np.ndarray:
    result: np.ndarray | None = None
    for rank in range(world_size):
        ids = np.arange(rank, samples, world_size, dtype=np.int64)
        shard = np.load(output_dir / f"endpoints_rank{rank:02d}.npy", mmap_mode="r")
        local = np.asarray(shard[condition_index], dtype=np.float32)
        if len(local) != len(ids):
            raise RuntimeError("endpoint shard length mismatch")
        if result is None:
            result = np.empty((samples, *local.shape[1:]), dtype=np.float32)
        result[ids] = local
    if result is None:
        raise RuntimeError("no endpoint shards found")
    return result


def _load_small_shards(
    output_dir: Path,
    *,
    filename: str,
    samples: int,
    world_size: int,
) -> np.ndarray:
    result: np.ndarray | None = None
    for rank in range(world_size):
        ids = np.arange(rank, samples, world_size, dtype=np.int64)
        local = np.load(output_dir / filename.format(rank=rank), mmap_mode="r")
        if len(local.shape) < 2 or local.shape[1] != len(ids):
            raise RuntimeError(f"small shard shape mismatch: {filename}, rank {rank}")
        moved = np.moveaxis(np.asarray(local), 1, 0)
        if result is None:
            result = np.empty((samples, *moved.shape[1:]), dtype=moved.dtype)
        result[ids] = moved
    if result is None:
        raise RuntimeError("no shards found")
    return result


def analyze_results(
    *,
    output_dir: Path,
    interventions: tuple[Intervention, ...],
    grid: torch.Tensor,
    samples: int,
    world_size: int,
    repeats: int,
    seed: int,
) -> None:
    baseline = _load_condition(
        output_dir, condition_index=0, samples=samples, world_size=world_size
    )
    stats_by_sample = _load_small_shards(
        output_dir,
        filename="injection_stats_rank{rank:02d}.npy",
        samples=samples,
        world_size=world_size,
    )
    # _load_small_shards returns [sample, condition, field].
    step_by_sample = _load_small_shards(
        output_dir,
        filename="baseline_steps_rank{rank:02d}.npy",
        samples=samples,
        world_size=world_size,
    )
    if step_by_sample.shape[1] != 1:
        raise RuntimeError("baseline step shards must contain one baseline condition")
    step_by_sample = step_by_sample[:, 0]
    condition_rows: list[dict[str, Any]] = []
    for index, item in enumerate(interventions):
        candidate = baseline if index == 0 else _load_condition(
            output_dir, condition_index=index, samples=samples, world_size=world_size
        )
        delta = (candidate.astype(np.float64) - baseline.astype(np.float64)).reshape(samples, -1)
        response = np.sqrt(np.mean(np.square(delta), axis=1))
        low, high = bootstrap_mean_interval(
            response, repeats=repeats, seed=seed + 101 * index
        )
        row = {
            **asdict(item),
            "active_steps": int(item.end_step - item.start_step),
            "start_time": float(grid[item.start_step]) if item.family != "baseline" else np.nan,
            "end_time": float(grid[item.end_step]) if item.family != "baseline" else np.nan,
            "endpoint_delta_rms_mean": float(response.mean()),
            "endpoint_delta_rms_ci_low": low,
            "endpoint_delta_rms_ci_high": high,
        }
        for field_index, field in enumerate(STAT_FIELDS):
            row[field + "_mean"] = float(stats_by_sample[:, index, field_index].mean())
        condition_rows.append(row)
    condition_frame = pd.DataFrame(condition_rows)

    pair_rows: list[dict[str, Any]] = []
    pairs = sorted({item.pair_name for item in interventions if item.pair_name is not None})
    for pair_index, pair_name in enumerate(pairs):
        positive_index = next(
            i for i, item in enumerate(interventions)
            if item.pair_name == pair_name and item.gamma > 0
        )
        negative_index = next(
            i for i, item in enumerate(interventions)
            if item.pair_name == pair_name and item.gamma < 0
        )
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
        odd = 0.5 * (positive - negative)
        even = 0.5 * (positive + negative) - baseline.astype(np.float64)
        odd_rms = np.sqrt(np.mean(np.square(odd).reshape(samples, -1), axis=1))
        even_rms = np.sqrt(np.mean(np.square(even).reshape(samples, -1), axis=1))
        positive_item = interventions[positive_index]
        gamma_abs = abs(float(positive_item.gamma))
        response = odd_rms / gamma_abs
        low, high = bootstrap_mean_interval(
            response, repeats=repeats, seed=seed + 1009 * pair_index
        )
        unit_energy = 0.5 * (
            stats_by_sample[:, positive_index, 0] + stats_by_sample[:, negative_index, 0]
        )
        direct_norm = np.sqrt(np.maximum(unit_energy, 0.0))
        pair_rows.append(
            {
                "pair_name": pair_name,
                "family": positive_item.family,
                "start_step": positive_item.start_step,
                "end_step": positive_item.end_step,
                "active_steps": positive_item.end_step - positive_item.start_step,
                "start_time": float(grid[positive_item.start_step]),
                "end_time": float(grid[positive_item.end_step]),
                "gamma_abs": gamma_abs,
                "central_response_per_gamma_mean": float(response.mean()),
                "central_response_per_gamma_ci_low": low,
                "central_response_per_gamma_ci_high": high,
                "central_response_per_gamma_median": float(np.median(response)),
                "central_response_per_gamma_q90": float(np.quantile(response, 0.90)),
                "central_response_per_gamma_q95": float(np.quantile(response, 0.95)),
                "central_response_per_gamma_max": float(response.max()),
                "central_response_tail_over_median": float(
                    np.quantile(response, 0.95) / max(np.median(response), 1e-30)
                ),
                "odd_endpoint_rms_mean": float(odd_rms.mean()),
                "even_endpoint_rms_mean": float(even_rms.mean()),
                "even_over_odd": float(even_rms.mean() / max(odd_rms.mean(), 1e-30)),
                "unit_injected_norm_mean": float(direct_norm.mean()),
                "response_per_unit_injected_norm": float(
                    response.mean() / max(direct_norm.mean(), 1e-30)
                ),
                "positive_actual_energy_mean": float(
                    stats_by_sample[:, positive_index, 1].mean()
                ),
                "negative_actual_energy_mean": float(
                    stats_by_sample[:, negative_index, 1].mean()
                ),
                "positive_unit_coherent_rms_mean": float(
                    stats_by_sample[:, positive_index, 2].mean()
                ),
                "negative_unit_coherent_rms_mean": float(
                    stats_by_sample[:, negative_index, 2].mean()
                ),
            }
        )
    pair_frame = pd.DataFrame(pair_rows).sort_values(["family", "start_step"])

    step_rows = []
    for step in range(len(grid) - 1):
        h = float(grid[step] - grid[step + 1])
        step_rows.append(
            {
                "step": step,
                "time": float(grid[step]),
                "next_time": float(grid[step + 1]),
                "step_size": h,
                "h_over_t": h / max(float(grid[step]), 1e-30),
                "raw_gap_rms_mean": float(step_by_sample[:, step, 0].mean()),
                "unit_impulse_rms_mean": float(step_by_sample[:, step, 1].mean()),
            }
        )
    step_frame = pd.DataFrame(step_rows)
    condition_frame.to_csv(output_dir / "condition_response.csv", index=False)
    pair_frame.to_csv(output_dir / "signed_response.csv", index=False)
    step_frame.to_csv(output_dir / "baseline_step_response.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    pulse = pair_frame[pair_frame["family"].eq("pulse")].sort_values("start_time")
    axes[0].errorbar(
        pulse["start_time"],
        pulse["central_response_per_gamma_mean"],
        yerr=np.vstack(
            (
                pulse["central_response_per_gamma_mean"] - pulse["central_response_per_gamma_ci_low"],
                pulse["central_response_per_gamma_ci_high"] - pulse["central_response_per_gamma_mean"],
            )
        ),
        fmt="o-",
        capsize=4,
    )
    axes[0].invert_xaxis()
    axes[0].set(title="Single-step terminal response", xlabel="solver time t", ylabel="odd endpoint RMS / gamma")
    window = pair_frame[pair_frame["family"].eq("window")].sort_values("start_step")
    positions = np.arange(len(window))
    axes[1].bar(positions, window["central_response_per_gamma_mean"], color="#2563EB")
    axes[1].set_xticks(positions, [f"{a}:{b}" for a, b in zip(window.start_step, window.end_step)])
    axes[1].set(title="Equal-step windows", xlabel="Euler-step range", ylabel="odd endpoint RMS / gamma")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "impulse_response.png", dpi=180)
    plt.close(figure)
    print(pair_frame.to_string(index=False), flush=True)


def main() -> None:
    install_raev2_decoder_config_compat()
    args = parse_args()
    positive = (
        args.samples,
        args.per_rank_batch,
        args.condition_group_size,
        args.window_count,
        args.bootstrap_repeats,
        args.log_every_samples,
    )
    if any(int(value) <= 0 for value in positive):
        raise ValueError("sample, batch, group, window, bootstrap, and logging values must be positive")
    os.environ["DINOV3_CKPT_DIR"] = str(args.dino_ckpt_dir.expanduser().resolve())
    os.environ["DINOV3_REPO_DIR"] = str(args.dino_repo_dir.expanduser().resolve())

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.backends.cuda.matmul.allow_tf32 = args.precision != "fp32"
    torch.backends.cudnn.allow_tf32 = args.precision != "fp32"
    if args.samples < world_size:
        raise ValueError("samples must be at least world size")

    output_dir = args.output_dir.expanduser().resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    config = load_config(args.config.expanduser().resolve())
    config.prepare_model_params()
    latent_size = tuple(int(value) for value in config.misc.latent_size)
    shift = math.sqrt(
        (config.misc.time_dist_shift_dim or math.prod(latent_size))
        / config.misc.time_dist_shift_base
    )
    grid = official_shifted_solver_grid(int(config.sampler.num_steps), shift)
    interventions = build_interventions(
        num_steps=len(grid) - 1,
        pulse_steps=() if args.skip_pulses else args.pulse_steps,
        window_count=args.window_count,
        gamma=args.gamma,
        pulse_gammas=args.pulse_gammas,
        window_gammas=args.window_gammas,
    )
    if args.skip_windows:
        interventions = tuple(
            item for item in interventions if item.family in ("baseline", "pulse")
        )
    if len(interventions) == 1:
        raise ValueError("at least one pulse or window intervention is required")
    if args.per_rank_batch * args.condition_group_size > 4:
        raise ValueError("per-rank batch times condition-group-size must not exceed 4")

    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint_hash = file_sha256(checkpoint_path) if rank == 0 else ""
    objects = [checkpoint_hash]
    dist.broadcast_object_list(objects, src=0)
    checkpoint_hash = objects[0]
    manifest_path = output_dir / "manifest.json"
    expected_manifest = {
        "protocol": PROTOCOL,
        "status": "running",
        "training": False,
        "config": str(args.config.expanduser().resolve()),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "state_key": args.state_key,
        "samples": int(args.samples),
        "seed": int(args.seed),
        "label_mode": args.label_mode,
        "label_seed": int(args.seed if args.label_seed is None else args.label_seed),
        "world_size": int(world_size),
        "precision": args.precision,
        "tf32": args.precision != "fp32",
        "latent_size": list(latent_size),
        "solver_grid": grid.double().tolist(),
        "solver_grid_dtype": str(grid.dtype),
        "interventions": [asdict(item) for item in interventions],
        "same_noise_and_labels_across_conditions": True,
        "stat_fields": list(STAT_FIELDS),
        "step_fields": list(STEP_FIELDS),
    }
    if rank == 0:
        if manifest_path.is_file():
            current = json.loads(manifest_path.read_text(encoding="utf-8"))
            # Runs started before label sampling became explicit used sequential
            # labels with the experiment seed. Preserve their resumability.
            current.setdefault("label_mode", "sequential")
            current.setdefault("label_seed", int(current["seed"]))
            keys = (
                "protocol", "checkpoint_sha256", "state_key", "samples", "seed",
                "label_mode", "label_seed", "world_size", "precision", "latent_size",
                "solver_grid", "interventions",
            )
            changed = [key for key in keys if current.get(key) != expected_manifest.get(key)]
            if changed:
                raise RuntimeError(f"cannot resume changed impulse protocol: {changed}")
        else:
            _atomic_json(manifest_path, expected_manifest)
            labels = build_validation_labels(
                args.samples,
                int(config.misc.num_classes),
                mode=args.label_mode,
                seed=int(args.seed if args.label_seed is None else args.label_seed),
            )
            np.savez_compressed(
                output_dir / "sample_protocol.npz",
                sample_ids=np.arange(args.samples, dtype=np.int64),
                labels=labels,
            )
    dist.barrier()
    labels = np.load(output_dir / "sample_protocol.npz")["labels"].astype(np.int64)
    local_ids = np.arange(rank, args.samples, world_size, dtype=np.int64)

    model = instantiate_from_config(config.stage_2).to(device).eval().requires_grad_(False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    validate_full_stage2_checkpoint(checkpoint)
    model.load_state_dict(checkpoint[args.state_key], strict=True)
    del checkpoint

    endpoints = _open_memmap(
        output_dir / f"endpoints_rank{rank:02d}.npy",
        shape=(len(interventions), len(local_ids), *latent_size),
        dtype=np.float32,
    )
    injection_stats = _open_memmap(
        output_dir / f"injection_stats_rank{rank:02d}.npy",
        shape=(len(interventions), len(local_ids), len(STAT_FIELDS)),
        dtype=np.float64,
    )
    baseline_steps = _open_memmap(
        output_dir / f"baseline_steps_rank{rank:02d}.npy",
        shape=(1, len(local_ids), len(grid) - 1, len(STEP_FIELDS)),
        dtype=np.float64,
    )
    progress_path = output_dir / f"progress_rank{rank:02d}.npy"
    progress_existed = progress_path.is_file()
    progress = _open_memmap(progress_path, shape=(len(local_ids),), dtype=np.bool_)
    if not progress_existed:
        progress.fill(False)
        progress.flush()
    total = len(local_ids)
    for start in range(0, total, args.per_rank_batch):
        stop = min(start + args.per_rank_batch, total)
        if bool(np.asarray(progress[start:stop]).all()):
            continue
        if bool(np.asarray(progress[start:stop]).any()):
            raise RuntimeError("partially complete batch cannot be resumed safely")
        ids = local_ids[start:stop]
        noise = deterministic_noise(ids, latent_size, seed=args.seed).to(device)
        batch_labels = torch.from_numpy(labels[ids]).to(device=device, dtype=torch.long)
        group_ranges = [(0, 1)] + [
            (group_start, min(group_start + args.condition_group_size, len(interventions)))
            for group_start in range(1, len(interventions), args.condition_group_size)
        ]
        explicit_baseline: torch.Tensor | None = None
        for group_start, group_stop in group_ranges:
            group = interventions[group_start:group_stop]
            endpoint, stats, step_stats = simulate_group(
                model=model,
                noise=noise,
                labels=batch_labels,
                grid=grid,
                interventions=group,
                t_eps=float(config.transport.t_eps),
                precision=args.precision,
                record_baseline_steps=any(item.family == "baseline" for item in group),
            )
            endpoints[group_start:group_stop, start:stop] = endpoint.cpu().numpy()
            injection_stats[group_start:group_stop, start:stop] = stats.cpu().numpy()
            if group_start == 0:
                explicit_baseline = endpoint[0]
            if step_stats is not None:
                baseline_steps[0, start:stop] = step_stats.numpy()
        verification_path = output_dir / f"official_baseline_check_rank{rank:02d}.json"
        if not args.skip_official_baseline_check and not verification_path.is_file():
            if explicit_baseline is None:
                raise AssertionError("explicit baseline was not evaluated")
            official = official_baseline_endpoint(
                model=model,
                noise=noise,
                labels=batch_labels,
                config=config,
                shift=shift,
                precision=args.precision,
            )
            delta = explicit_baseline.double() - official.double()
            check = {
                "rank": rank,
                "samples_checked": int(len(noise)),
                "rms": float(delta.square().mean().sqrt().cpu()),
                "maximum_absolute": float(delta.abs().max().cpu()),
                "explicit_rms": float(explicit_baseline.double().square().mean().sqrt().cpu()),
            }
            _atomic_json(verification_path, check)
            if check["rms"] > 1e-5 or check["maximum_absolute"] > 1e-3:
                raise RuntimeError(f"explicit sampler does not match official baseline: {check}")
        endpoints.flush()
        injection_stats.flush()
        baseline_steps.flush()
        progress[start:stop] = True
        progress.flush()
        if rank == 0 and (stop % args.log_every_samples == 0 or stop == total):
            print(f"[rank 0] local samples {stop}/{total}", flush=True)
    _atomic_json(
        output_dir / f"complete_rank{rank:02d}.json",
        {"rank": rank, "local_rows": total, "complete": True},
    )
    dist.barrier()
    if rank == 0:
        analyze_results(
            output_dir=output_dir,
            interventions=interventions,
            grid=grid,
            samples=args.samples,
            world_size=world_size,
            repeats=args.bootstrap_repeats,
            seed=args.seed + 17,
        )
        final_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        final_manifest["status"] = "complete"
        _atomic_json(manifest_path, final_manifest)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
