#!/usr/bin/env python3
"""Eight-way RAEv2 IG stage ablation built on the proven impulse-response runner.

This file intentionally does *not* implement another sampler.  It imports the
working rollout core from ``experiments.run_raev2_ig_impulse_response`` and only
adds non-contiguous early/middle/late schedules plus endpoint attribution.

The official IG-active Euler steps are divided into three equal-count phases.
All 2^3 combinations are evaluated from identical noise and labels:

    no_ig, early, middle, late,
    early_middle, early_late, middle_late, full_ig.

Outputs include endpoint recovery toward full IG, exact vector Shapley values,
pair/triple interactions, injection statistics, pairwise endpoint distances,
and an optional same-noise decoded preview.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import shutil
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
RAEV2_SRC = ROOT / "external" / "RAEv2" / "src"
for _path in (RAEV2_SRC, ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from experiments.raev2_stage1_compat import (  # noqa: E402
    install_raev2_decoder_config_compat,
)
from experiments.raev2_training_core import (  # noqa: E402
    file_sha256,
    validate_full_stage2_checkpoint,
)
from experiments.run_raev2_distribution_auc import load_config  # noqa: E402
from experiments.run_raev2_ig_impulse_response import (  # noqa: E402
    STAT_FIELDS,
    _atomic_json,
    autocast_context,
    build_validation_labels,
    deterministic_noise,
    official_baseline_endpoint,
    official_shifted_solver_grid,
    simulate_group,
)
from utils.model_utils import instantiate_from_config  # noqa: E402

PROTOCOL = "raev2_ig_stage_ablation_reuse_v1"
SCRIPT_VERSION = "reuse_v1"
PHASE_NAMES = ("early", "middle", "late")
SCHEDULE_ORDER = (
    "no_ig",
    "early",
    "middle",
    "late",
    "early_middle",
    "early_late",
    "middle_late",
    "full_ig",
)
EPS = 1e-30


@dataclass(frozen=True)
class StageIntervention:
    """Interface-compatible schedule consumed by impulse_response.simulate_group."""

    name: str
    family: str
    segments: tuple[tuple[int, int], ...]
    gamma: float
    phase_mask: int
    pair_name: str | None = None

    def active(self, step: int) -> bool:
        value = int(step)
        return any(start <= value < end for start, end in self.segments)


@dataclass(frozen=True)
class PhaseDefinition:
    name: str
    index: int
    start_step: int
    end_step: int
    start_time: float
    end_time: float

    @property
    def active_steps(self) -> int:
        return self.end_step - self.start_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/configs/raev2_strict_lpl_dinov3l_k7.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/data/users/zhoushunyu/eqvae/models/RAEv2/stage2/imagenet/"
            "dinov3l-k7/checkpoint.pt"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--per-rank-batch", type=int, default=1)
    parser.add_argument(
        "--condition-group-size",
        type=int,
        default=2,
        help="Number of stage schedules evaluated together by the proven runner.",
    )
    parser.add_argument("--ig-scale", type=float)
    parser.add_argument("--state-key", choices=("ema", "model"), default="ema")
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16"),
        default="fp32",
        help="fp32 is recommended for the first mechanism run and baseline check.",
    )
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument(
        "--label-mode",
        choices=("sequential", "random_without_replacement"),
        default="sequential",
    )
    parser.add_argument("--label-seed", type=int)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--log-every-batches", type=int, default=1)
    parser.add_argument(
        "--official-baseline-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the exact check already used by run_raev2_ig_impulse_response.py.",
    )
    parser.add_argument(
        "--save-preview",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the named output directory before starting. Otherwise it must be new.",
    )
    parser.add_argument(
        "--dino-ckpt-dir",
        type=Path,
        default=Path("/data/users/zhoushunyu/eqvae/models/RAEv2/encoders/dinov3"),
    )
    parser.add_argument(
        "--dino-repo-dir",
        type=Path,
        default=Path("/data/users/zhoushunyu/eqvae/models/RAEv2/dinov3_repo"),
    )
    return parser.parse_args()


def prepare_fresh_output(path: Path, *, overwrite: bool) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists():
        if overwrite:
            shutil.rmtree(resolved)
        elif any(resolved.iterdir()):
            raise FileExistsError(
                f"Refusing to reuse non-empty output directory: {resolved}. "
                "Use a new versioned directory name or pass --overwrite."
            )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def build_phase_definitions(
    grid: torch.Tensor | np.ndarray,
    interval: tuple[float, float],
) -> tuple[PhaseDefinition, ...]:
    values = np.asarray(grid, dtype=np.float64)
    active = np.asarray(
        [
            step
            for step in range(len(values) - 1)
            if float(interval[0]) <= float(values[step]) <= float(interval[1])
        ],
        dtype=np.int64,
    )
    if len(active) < 3:
        raise ValueError("The official IG interval contains fewer than three steps.")
    expected = np.arange(int(active[0]), int(active[-1]) + 1, dtype=np.int64)
    if not np.array_equal(active, expected):
        raise ValueError("The official IG-active solver steps are not contiguous.")
    chunks = tuple(np.asarray(chunk, dtype=np.int64) for chunk in np.array_split(active, 3))
    result: list[PhaseDefinition] = []
    for index, chunk in enumerate(chunks):
        start = int(chunk[0])
        end = int(chunk[-1]) + 1
        result.append(
            PhaseDefinition(
                name=PHASE_NAMES[index],
                index=index,
                start_step=start,
                end_step=end,
                start_time=float(values[start]),
                end_time=float(values[end]),
            )
        )
    covered = np.concatenate(
        [np.arange(item.start_step, item.end_step, dtype=np.int64) for item in result]
    )
    if not np.array_equal(covered, active):
        raise AssertionError("Phase definitions do not exactly cover the official interval.")
    return tuple(result)


def subset_name(phases: Iterable[int]) -> str:
    subset = frozenset(int(value) for value in phases)
    if not subset:
        return "no_ig"
    if len(subset) == 3:
        return "full_ig"
    return "_".join(PHASE_NAMES[index] for index in sorted(subset))


def build_stage_interventions(
    phases: tuple[PhaseDefinition, ...],
    *,
    gamma: float,
) -> tuple[StageIntervention, ...]:
    by_name: dict[str, StageIntervention] = {}
    for size in range(4):
        for subset_tuple in itertools.combinations(range(3), size):
            subset = frozenset(subset_tuple)
            name = subset_name(subset)
            segments = tuple(
                (phases[index].start_step, phases[index].end_step)
                for index in sorted(subset)
            )
            mask = sum(1 << index for index in subset)
            by_name[name] = StageIntervention(
                name=name,
                family="baseline" if not subset else "stage_schedule",
                segments=segments,
                gamma=0.0 if not subset else float(gamma),
                phase_mask=mask,
            )
    result = tuple(by_name[name] for name in SCHEDULE_ORDER)
    if len(result) != 8 or len({item.phase_mask for item in result}) != 8:
        raise AssertionError("The stage schedule must contain all 2^3 subsets.")
    return result


def np_rms(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    return np.sqrt(np.mean(np.square(array.reshape(array.shape[0], -1)), axis=1))


def np_cosine(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    left = np.asarray(first, dtype=np.float64).reshape(first.shape[0], -1)
    right = np.asarray(second, dtype=np.float64).reshape(second.shape[0], -1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return np.sum(left * right, axis=1) / np.maximum(denominator, EPS)


def exact_shapley_vectors(
    endpoints_by_subset: dict[frozenset[int], np.ndarray],
) -> dict[int, np.ndarray]:
    players = frozenset(range(3))
    shape = next(iter(endpoints_by_subset.values())).shape
    result = {player: np.zeros(shape, dtype=np.float32) for player in players}
    for player in players:
        others = sorted(players - {player})
        for size in range(len(others) + 1):
            weight = (
                math.factorial(size)
                * math.factorial(3 - size - 1)
                / math.factorial(3)
            )
            for subset_tuple in itertools.combinations(others, size):
                subset = frozenset(subset_tuple)
                result[player] += float(weight) * (
                    endpoints_by_subset[subset | {player}]
                    - endpoints_by_subset[subset]
                ).astype(np.float32, copy=False)
    return result


def summarize(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p90": float(np.quantile(array, 0.90)),
    }


def analyze_endpoints(
    *,
    endpoints: np.ndarray,
    interventions: tuple[StageIntervention, ...],
    injection_stats: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    index_by_name = {item.name: index for index, item in enumerate(interventions)}
    baseline = endpoints[index_by_name["no_ig"]].astype(np.float32, copy=False)
    full = endpoints[index_by_name["full_ig"]].astype(np.float32, copy=False)
    full_effect = full - baseline
    full_flat = full_effect.reshape(len(full_effect), -1).astype(np.float64)
    full_energy = np.sum(full_flat * full_flat, axis=1)
    full_rms = np_rms(full_effect)

    schedule_rows: list[dict[str, Any]] = []
    for index, item in enumerate(interventions):
        candidate = endpoints[index].astype(np.float32, copy=False)
        effect = candidate - baseline
        effect_flat = effect.reshape(len(effect), -1).astype(np.float64)
        projection = np.sum(effect_flat * full_flat, axis=1) / np.maximum(
            full_energy, EPS
        )
        projection_view = projection.reshape(
            (len(projection),) + (1,) * (effect.ndim - 1)
        )
        off_axis = effect - projection_view.astype(np.float32) * full_effect
        row: dict[str, Any] = {
            "schedule": item.name,
            "phase_mask": item.phase_mask,
            "active_phases": "+".join(
                PHASE_NAMES[index]
                for index in range(3)
                if item.phase_mask & (1 << index)
            )
            or "none",
            "endpoint_delta_from_no_ig_rms_mean": float(np_rms(effect).mean()),
            "endpoint_distance_to_full_ig_rms_mean": float(
                np_rms(candidate - full).mean()
            ),
            "full_ig_effect_projection_mean": float(projection.mean()),
            "full_ig_effect_projection_median": float(np.median(projection)),
            "effect_cosine_to_full_ig_mean": (
                float(np_cosine(effect, full_effect).mean())
                if item.name != "no_ig"
                else float("nan")
            ),
            "off_axis_residual_over_full_effect_mean": float(
                (np_rms(off_axis) / np.maximum(full_rms, EPS)).mean()
            ),
        }
        for field_index, field in enumerate(STAT_FIELDS):
            row[field + "_mean"] = float(
                np.asarray(injection_stats[index, :, field_index], dtype=np.float64).mean()
            )
        schedule_rows.append(row)
    schedule_frame = pd.DataFrame(schedule_rows)

    subset_endpoint = {
        frozenset(index for index in range(3) if item.phase_mask & (1 << index)):
        endpoints[position].astype(np.float32, copy=False)
        for position, item in enumerate(interventions)
    }
    interaction_rows: list[dict[str, Any]] = []
    for first, second in itertools.combinations(range(3), 2):
        pair = frozenset((first, second))
        interaction = (
            subset_endpoint[pair]
            - subset_endpoint[frozenset((first,))]
            - subset_endpoint[frozenset((second,))]
            + subset_endpoint[frozenset()]
        )
        interaction_rows.append(
            {
                "interaction": f"{PHASE_NAMES[first]}:{PHASE_NAMES[second]}",
                "order": 2,
                "rms_mean": float(np_rms(interaction).mean()),
                "rms_over_full_effect_mean": float(
                    (np_rms(interaction) / np.maximum(full_rms, EPS)).mean()
                ),
                "cosine_to_full_effect_mean": float(
                    np_cosine(interaction, full_effect).mean()
                ),
            }
        )
    triple = (
        subset_endpoint[frozenset((0, 1, 2))]
        - subset_endpoint[frozenset((0, 1))]
        - subset_endpoint[frozenset((0, 2))]
        - subset_endpoint[frozenset((1, 2))]
        + subset_endpoint[frozenset((0,))]
        + subset_endpoint[frozenset((1,))]
        + subset_endpoint[frozenset((2,))]
        - subset_endpoint[frozenset()]
    )
    interaction_rows.append(
        {
            "interaction": "early:middle:late",
            "order": 3,
            "rms_mean": float(np_rms(triple).mean()),
            "rms_over_full_effect_mean": float(
                (np_rms(triple) / np.maximum(full_rms, EPS)).mean()
            ),
            "cosine_to_full_effect_mean": float(
                np_cosine(triple, full_effect).mean()
            ),
        }
    )
    interaction_frame = pd.DataFrame(interaction_rows)

    shapley = exact_shapley_vectors(subset_endpoint)
    shapley_rows: list[dict[str, Any]] = []
    for phase in range(3):
        vector = shapley[phase]
        vector_flat = vector.reshape(len(vector), -1).astype(np.float64)
        projection = np.sum(vector_flat * full_flat, axis=1) / np.maximum(
            full_energy, EPS
        )
        shapley_rows.append(
            {
                "phase": PHASE_NAMES[phase],
                "shapley_vector_rms_mean": float(np_rms(vector).mean()),
                "shapley_projection_fraction_mean": float(projection.mean()),
                "shapley_projection_fraction_median": float(np.median(projection)),
                "shapley_cosine_to_full_effect_mean": float(
                    np_cosine(vector, full_effect).mean()
                ),
            }
        )
    shapley_frame = pd.DataFrame(shapley_rows)

    pairwise_rows: list[dict[str, Any]] = []
    for left_index, left in enumerate(interventions):
        for right_index, right in enumerate(interventions):
            pairwise_rows.append(
                {
                    "left": left.name,
                    "right": right.name,
                    "endpoint_rms": float(
                        np_rms(endpoints[left_index] - endpoints[right_index]).mean()
                    ),
                }
            )
    pairwise_frame = pd.DataFrame(pairwise_rows)

    reconstruction = shapley[0] + shapley[1] + shapley[2]
    summary = {
        "full_ig_effect_rms": summarize(full_rms),
        "shapley_reconstruction_rms": summarize(
            np_rms(reconstruction - full_effect)
        ),
        "stage_projection_ranking": shapley_frame.sort_values(
            "shapley_projection_fraction_mean", ascending=False
        )[["phase", "shapley_projection_fraction_mean"]].to_dict("records"),
        "largest_interaction": interaction_frame.loc[
            interaction_frame["rms_mean"].idxmax(), "interaction"
        ],
    }
    return schedule_frame, interaction_frame, shapley_frame, pairwise_frame, summary


def merge_shards(
    output_dir: Path,
    *,
    samples: int,
    world_size: int,
    condition_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    endpoints: np.ndarray | None = None
    stats = np.empty((condition_count, samples, len(STAT_FIELDS)), dtype=np.float64)
    labels = np.empty(samples, dtype=np.int64)
    for rank in range(world_size):
        path = output_dir / f"stage_reuse_v1_rank{rank:02d}.npz"
        with np.load(path) as payload:
            ids = payload["ids"].astype(np.int64, copy=False)
            local_endpoints = payload["endpoints"].astype(np.float32, copy=False)
            if endpoints is None:
                endpoints = np.empty(
                    (condition_count, samples, *local_endpoints.shape[2:]),
                    dtype=np.float32,
                )
            endpoints[:, ids] = local_endpoints
            stats[:, ids] = payload["injection_stats"]
            labels[ids] = payload["labels"]
    if endpoints is None:
        raise RuntimeError("No stage-reuse rank shards were found.")
    return endpoints, labels, stats


def save_plots(frame: pd.DataFrame, output_dir: Path) -> None:
    x = np.arange(len(frame))
    figure, axis = plt.subplots(figsize=(11, 5.2))
    axis.bar(x, frame["full_ig_effect_projection_mean"])
    axis.set_xticks(x, frame["schedule"], rotation=28, ha="right")
    axis.axhline(0.0, linewidth=1)
    axis.axhline(1.0, linewidth=1, linestyle="--")
    axis.set_ylabel("Projection onto full-IG endpoint displacement")
    axis.set_title("RAEv2 IG stage-effect recovery")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "stage_effect_recovery_reuse_v1.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(11, 5.2))
    axis.bar(x, frame["endpoint_distance_to_full_ig_rms_mean"])
    axis.set_xticks(x, frame["schedule"], rotation=28, ha="right")
    axis.set_ylabel("RMS distance to full-IG endpoint")
    axis.set_title("Endpoint distance to full IG")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "distance_to_full_ig_reuse_v1.png", dpi=180)
    plt.close(figure)


def decode_preview(
    *,
    config: Any,
    endpoints: np.ndarray,
    interventions: tuple[StageIntervention, ...],
    device: torch.device,
    precision: str,
    output_dir: Path,
) -> None:
    rae = instantiate_from_config(config.stage_1).to(device).eval().requires_grad_(False)
    if hasattr(rae, "encoder"):
        del rae.encoder
    images: list[tuple[str, torch.Tensor]] = []
    with torch.inference_mode():
        for index, intervention in enumerate(interventions):
            latent = torch.from_numpy(endpoints[index, :1]).to(device=device)
            with autocast_context(precision):
                image = rae.decode(latent).float().clamp(0, 1)[0]
            images.append((intervention.name, image.cpu()))
    del rae
    torch.cuda.empty_cache()
    figure, axes = plt.subplots(2, 4, figsize=(14, 7.2))
    for axis, (name, image) in zip(axes.flat, images):
        axis.imshow(image.permute(1, 2, 0).numpy())
        axis.set_title(name)
        axis.axis("off")
    figure.suptitle("Same noise/class: IG stage ablation reuse_v1")
    figure.tight_layout()
    figure.savefig(output_dir / "stage_ablation_preview_reuse_v1.png", dpi=180)
    plt.close(figure)


def main() -> None:
    install_raev2_decoder_config_compat()
    args = parse_args()
    positive = (
        args.samples,
        args.per_rank_batch,
        args.condition_group_size,
        args.bootstrap_repeats,
        args.log_every_batches,
    )
    if any(int(value) <= 0 for value in positive):
        raise ValueError("Samples, batches, groups, bootstrap and logging must be positive.")
    if args.per_rank_batch * args.condition_group_size > 4:
        raise ValueError(
            "per-rank-batch * condition-group-size must not exceed 4, matching "
            "the proven impulse-response protocol."
        )

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
        raise ValueError("samples must be at least the distributed world size")

    if rank == 0:
        output_dir = prepare_fresh_output(args.output_dir, overwrite=args.overwrite)
        output_string = str(output_dir)
    else:
        output_string = ""
    objects = [output_string]
    dist.broadcast_object_list(objects, src=0)
    output_dir = Path(objects[0])
    dist.barrier()

    config = load_config(args.config.expanduser().resolve())
    config.prepare_model_params()
    latent_size = tuple(int(value) for value in config.misc.latent_size)
    shift = math.sqrt(
        (config.misc.time_dist_shift_dim or math.prod(latent_size))
        / config.misc.time_dist_shift_base
    )
    grid = official_shifted_solver_grid(int(config.sampler.num_steps), shift)
    interval = (
        float(config.guidance.ig.t_min),
        float(config.guidance.ig.t_max),
    )
    ig_scale = (
        float(args.ig_scale)
        if args.ig_scale is not None
        else float(config.guidance.ig.scale)
    )
    gamma = ig_scale - 1.0
    phases = build_phase_definitions(grid, interval)
    interventions = build_stage_interventions(phases, gamma=gamma)

    label_seed = int(args.seed if args.label_seed is None else args.label_seed)
    labels = build_validation_labels(
        args.samples,
        int(config.misc.num_classes),
        mode=args.label_mode,
        seed=label_seed,
    )
    local_ids = np.arange(rank, args.samples, world_size, dtype=np.int64)
    local_labels = labels[local_ids]

    checkpoint_path = args.checkpoint.expanduser().resolve()
    model = instantiate_from_config(config.stage_2).to(device).eval().requires_grad_(False)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    validate_full_stage2_checkpoint(checkpoint)
    checkpoint_step = int(checkpoint.get("step", 0))
    checkpoint_epoch = int(checkpoint.get("epoch", 0))
    model.load_state_dict(checkpoint[args.state_key], strict=True)
    del checkpoint

    local_endpoints = np.empty(
        (len(interventions), len(local_ids), *latent_size), dtype=np.float32
    )
    local_stats = np.empty(
        (len(interventions), len(local_ids), len(STAT_FIELDS)), dtype=np.float64
    )
    baseline_check: dict[str, Any] | None = None
    total_batches = math.ceil(len(local_ids) / args.per_rank_batch)
    for batch_index, start in enumerate(range(0, len(local_ids), args.per_rank_batch)):
        stop = min(start + args.per_rank_batch, len(local_ids))
        ids = local_ids[start:stop]
        noise = deterministic_noise(ids, latent_size, seed=args.seed).to(device)
        batch_labels = torch.from_numpy(labels[ids]).to(device=device, dtype=torch.long)

        ranges = [(0, 1)] + [
            (group_start, min(group_start + args.condition_group_size, len(interventions)))
            for group_start in range(1, len(interventions), args.condition_group_size)
        ]
        explicit_baseline: torch.Tensor | None = None
        for group_start, group_stop in ranges:
            group = interventions[group_start:group_stop]
            endpoint, stats, _ = simulate_group(
                model=model,
                noise=noise,
                labels=batch_labels,
                grid=grid,
                interventions=group,
                t_eps=float(config.transport.t_eps),
                precision=args.precision,
                record_baseline_steps=False,
            )
            local_endpoints[group_start:group_stop, start:stop] = endpoint.cpu().numpy()
            local_stats[group_start:group_stop, start:stop] = stats.cpu().numpy()
            if group_start == 0:
                explicit_baseline = endpoint[0]

        if args.official_baseline_check and baseline_check is None:
            if explicit_baseline is None:
                raise AssertionError("The no_ig baseline was not evaluated.")
            official = official_baseline_endpoint(
                model=model,
                noise=noise,
                labels=batch_labels,
                config=config,
                shift=shift,
                precision=args.precision,
            )
            delta = explicit_baseline.double() - official.double()
            baseline_check = {
                "rank": rank,
                "samples_checked": int(len(noise)),
                "rms": float(delta.square().mean().sqrt().cpu()),
                "maximum_absolute": float(delta.abs().max().cpu()),
                "explicit_rms": float(
                    explicit_baseline.double().square().mean().sqrt().cpu()
                ),
            }
            _atomic_json(
                output_dir / f"official_baseline_check_reuse_v1_rank{rank:02d}.json",
                baseline_check,
            )
            # Exact thresholds copied from the proven impulse-response runner.
            if baseline_check["rms"] > 1e-5 or baseline_check["maximum_absolute"] > 1e-3:
                raise RuntimeError(
                    "The imported proven rollout core failed its own official baseline "
                    f"check: {baseline_check}"
                )

        if rank == 0 and (
            (batch_index + 1) % args.log_every_batches == 0
            or batch_index + 1 == total_batches
        ):
            print(
                f"[stage reuse_v1] batches {batch_index + 1}/{total_batches}; "
                f"local samples {stop}/{len(local_ids)}",
                flush=True,
            )

    np.savez_compressed(
        output_dir / f"stage_reuse_v1_rank{rank:02d}.npz",
        ids=local_ids,
        labels=local_labels,
        endpoints=local_endpoints,
        injection_stats=local_stats,
        schedule_names=np.asarray([item.name for item in interventions]),
    )
    _atomic_json(
        output_dir / f"complete_reuse_v1_rank{rank:02d}.json",
        {"rank": rank, "local_rows": int(len(local_ids)), "complete": True},
    )
    dist.barrier()

    if rank == 0:
        endpoints, merged_labels, injection_stats = merge_shards(
            output_dir,
            samples=args.samples,
            world_size=world_size,
            condition_count=len(interventions),
        )
        schedule_frame, interaction_frame, shapley_frame, pairwise_frame, summary = (
            analyze_endpoints(
                endpoints=endpoints,
                interventions=interventions,
                injection_stats=injection_stats,
            )
        )
        schedule_frame.to_csv(
            output_dir / "stage_ablation_summary_reuse_v1.csv", index=False
        )
        interaction_frame.to_csv(
            output_dir / "stage_interactions_reuse_v1.csv", index=False
        )
        shapley_frame.to_csv(
            output_dir / "stage_shapley_reuse_v1.csv", index=False
        )
        pairwise_frame.to_csv(
            output_dir / "endpoint_pairwise_rms_reuse_v1.csv", index=False
        )
        np.savez_compressed(
            output_dir / "stage_endpoints_merged_reuse_v1.npz",
            endpoints=endpoints,
            labels=merged_labels,
            injection_stats=injection_stats,
            schedule_names=np.asarray([item.name for item in interventions]),
        )
        manifest = {
            "protocol": PROTOCOL,
            "script_version": SCRIPT_VERSION,
            "script_name": Path(__file__).name,
            "rollout_core": "experiments.run_raev2_ig_impulse_response.simulate_group",
            "training": False,
            "config": str(args.config.expanduser().resolve()),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "checkpoint_step": checkpoint_step,
            "checkpoint_epoch": checkpoint_epoch,
            "state_key": args.state_key,
            "precision": args.precision,
            "samples": args.samples,
            "world_size": world_size,
            "seed": args.seed,
            "label_mode": args.label_mode,
            "label_seed": label_seed,
            "ig_scale": ig_scale,
            "ig_gamma": gamma,
            "official_ig_interval": interval,
            "phases": [asdict(item) | {"active_steps": item.active_steps} for item in phases],
            "interventions": [asdict(item) for item in interventions],
            "same_noise_and_labels_across_conditions": True,
            "stat_fields": list(STAT_FIELDS),
            "official_baseline_check": args.official_baseline_check,
            "summary": summary,
        }
        _atomic_json(output_dir / "manifest_reuse_v1.json", manifest)
        _atomic_json(output_dir / "summary_reuse_v1.json", summary)
        save_plots(schedule_frame, output_dir)
        if args.save_preview:
            del model
            torch.cuda.empty_cache()
            decode_preview(
                config=config,
                endpoints=endpoints,
                interventions=interventions,
                device=device,
                precision=args.precision,
                output_dir=output_dir,
            )
        print("\nStage-ablation reuse_v1 summary")
        print(schedule_frame.to_string(index=False))
        print("\nExact stage Shapley decomposition")
        print(shapley_frame.to_string(index=False))
        print("\nInteractions")
        print(interaction_frame.to_string(index=False))
        print(f"\nOutputs: {output_dir}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()