#!/usr/bin/env python3
"""Fast screening experiment for RAEv2 input-frequency-axis extrapolation.

This program tests a different idea from frequency-selective decomposition of
``Full - Base``.  Here the same Full model is evaluated on two inputs that lie
on a controlled clean-signal frequency-degradation axis:

    y0 = Full(x_t, t)
    x_t^tau = x_t + (1-t) [H_tau(y0) - y0]
    y_tau = Full(x_t^tau, t)
    q_tau = y0 - y_tau
    y_ext = y0 + alpha q_tau

``H_tau`` is an isotropic Gaussian low-pass operator on each 16x16 latent
channel.  The counterfactual construction keeps the noise inferred from y0
exactly unchanged:

    eps_hat(x_t, y0) == eps_hat(x_t^tau, H_tau(y0)).

The script is deliberately a cheap screening experiment, not a final metric
suite.  It has three goals:

1. Teacher-forced direction audit
   - Does q_tau point toward the true clean latent?
   - Is the frequency axis locally smooth across blur scales?
   - Does the inference-time self-constructed axis agree with an oracle axis
     built from the true clean latent?
   - Is q_tau redundant with or complementary to the RAEv2 Full-Base gap?

2. Tiny same-noise rollout
   - Compare no IG, official scalar IG, and a few positive/negative frequency
     extrapolation coefficients on only a handful of samples.
   - Produce previews, endpoint latent statistics, and paired endpoint distances.
   - No FID/KID is computed because a tiny sample set cannot support them.

3. Explicit screening report
   - Separate mathematical identities, direct measurements, and heuristic
     go/no-go indicators.

The script imports the exact RAEv2 Euler arithmetic and scale-1 semantics from
``run_raev2_spectral_ig_mechanism_suite_v3.py`` so the no-IG branch remains
bitwise consistent with the released sampler.

Create a new versioned file for future revisions; do not overwrite this v1.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import math
import os
import shutil
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from PIL import Image


PROTOCOL = "raev2_frequency_axis_extrapolation_screen_v1"
SCRIPT_VERSION = "v1"
EPS = 1e-12

ROOT = Path(__file__).resolve().parents[1]
RAEV2_SRC = ROOT / "external" / "RAEv2" / "src"

DEFAULT_CONFIG = ROOT / "experiments/configs/raev2_strict_lpl_dinov3l_k7.yaml"
DEFAULT_CHECKPOINT = Path(
    "/data/users/zhoushunyu/eqvae/models/RAEv2/stage2/imagenet/"
    "dinov3l-k7/checkpoint.pt"
)
DEFAULT_PACKED_DATA = Path("/data/shared/imagenet-1k/random_access_v1")
DEFAULT_PARQUET_DATA = Path("/data/shared/imagenet-1k")
DEFAULT_DINO_CKPT = Path("/data/users/zhoushunyu/eqvae/models/RAEv2/encoders/dinov3")
DEFAULT_DINO_REPO = Path("/data/users/zhoushunyu/eqvae/models/RAEv2/dinov3_repo")


# ---------------------------------------------------------------------------
# Pure helpers: locally unit-testable without the RAEv2 repository.
# ---------------------------------------------------------------------------


def parse_csv_floats(value: str, *, allow_empty: bool = False) -> tuple[float, ...]:
    try:
        result = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated finite floats") from error
    if not result and not allow_empty:
        raise argparse.ArgumentTypeError("at least one float is required")
    if any(not math.isfinite(item) for item in result):
        raise argparse.ArgumentTypeError("all values must be finite")
    return result


def parse_csv_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return result


def parse_stages(value: str) -> tuple[str, ...]:
    allowed = {"teacher", "sample", "decode", "report"}
    result = tuple(item.strip() for item in str(value).split(",") if item.strip())
    unknown = sorted(set(result) - allowed)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown stages: {unknown}")
    if not result:
        raise argparse.ArgumentTypeError("at least one stage is required")
    return result


def alpha_tag(value: float) -> str:
    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def gaussian_frequency_mask(
    height: int,
    width: int,
    sigma_pixels: float,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Fourier multiplier of a periodic Gaussian blur with spatial std sigma.

    ``torch.fft.fftfreq`` returns cycles per pixel.  A spatial Gaussian with
    standard deviation sigma has Fourier multiplier

        exp(-2*pi^2*sigma^2*(fx^2+fy^2)).

    The semigroup/extrapolation coordinate is therefore tau=sigma^2.
    """

    sigma = float(sigma_pixels)
    if sigma < 0 or not math.isfinite(sigma):
        raise ValueError("sigma_pixels must be a finite non-negative number")
    fy = torch.fft.fftfreq(int(height), d=1.0, device=device, dtype=dtype)
    fx = torch.fft.fftfreq(int(width), d=1.0, device=device, dtype=dtype)
    radius_sq = fy[:, None].square() + fx[None, :].square()
    return torch.exp(-2.0 * math.pi**2 * sigma**2 * radius_sq)


def gaussian_blur_latent(value: torch.Tensor, sigma_pixels: float) -> torch.Tensor:
    """Apply channelwise periodic Gaussian blur to a BCHW latent tensor."""

    if value.ndim != 4:
        raise ValueError("value must be a BCHW tensor")
    if float(sigma_pixels) == 0.0:
        return value.float()
    transform = torch.fft.fft2(value.float(), dim=(-2, -1), norm="ortho")
    mask = gaussian_frequency_mask(
        value.shape[-2],
        value.shape[-1],
        sigma_pixels,
        device=value.device,
        dtype=torch.float32,
    )
    result = torch.fft.ifft2(
        transform * mask[None, None], dim=(-2, -1), norm="ortho"
    ).real
    return result


def inferred_noise(
    state: torch.Tensor,
    clean_prediction: torch.Tensor,
    time: float,
    *,
    t_eps: float = 1e-6,
) -> torch.Tensor:
    t = max(float(time), float(t_eps))
    return (state.float() - (1.0 - float(time)) * clean_prediction.float()) / t


def clean_frequency_counterfactual(
    state: torch.Tensor,
    clean_prediction: torch.Tensor,
    time: float,
    sigma_pixels: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Blur only the inferred clean component while preserving inferred noise.

    Returns ``(counterfactual_state, blurred_clean_prediction)``.
    """

    blurred = gaussian_blur_latent(clean_prediction, sigma_pixels)
    counterfactual = state.float() + (1.0 - float(time)) * (
        blurred - clean_prediction.float()
    )
    return counterfactual, blurred


def rowwise_inner(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left.double().flatten(1) * right.double().flatten(1)).sum(dim=1)


def rowwise_rms(value: torch.Tensor) -> torch.Tensor:
    return value.double().flatten(1).square().mean(dim=1).sqrt()


def rowwise_cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    numerator = rowwise_inner(left, right)
    denominator = (
        left.double().flatten(1).norm(dim=1)
        * right.double().flatten(1).norm(dim=1)
    ).clamp_min(EPS)
    return numerator / denominator


def optimal_alpha(target_residual: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Per-sample MSE-optimal alpha for y0 + alpha*direction toward target."""

    return rowwise_inner(target_residual, direction) / rowwise_inner(
        direction, direction
    ).clamp_min(EPS)


def relative_mse_change(
    base_prediction: torch.Tensor,
    target: torch.Tensor,
    direction: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    base = (base_prediction.double() - target.double()).flatten(1).square().mean(dim=1)
    candidate = (
        base_prediction.double() + float(alpha) * direction.double() - target.double()
    ).flatten(1).square().mean(dim=1)
    return (candidate - base) / base.clamp_min(EPS)


def counterfactual_noise_error(
    state: torch.Tensor,
    clean: torch.Tensor,
    counterfactual: torch.Tensor,
    blurred_clean: torch.Tensor,
    time: float,
) -> torch.Tensor:
    original_noise = inferred_noise(state, clean, time)
    counterfactual_noise = inferred_noise(counterfactual, blurred_clean, time)
    return rowwise_rms(original_noise - counterfactual_noise)


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Repository imports and distributed setup.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device


def initialize_repo_imports() -> None:
    for path in (RAEV2_SRC, ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def repo_symbols() -> dict[str, Any]:
    initialize_repo_imports()
    from experiments.raev2_stage1_compat import install_raev2_decoder_config_compat
    from experiments.raev2_training_core import (
        DeterministicImageNetPacked,
        split_internal_guidance_output,
        validate_full_stage2_checkpoint,
    )
    from experiments.run_raev2_distribution_auc import (
        build_requested_labels,
        load_config,
        select_matching_imagenet_rows,
    )
    from experiments.run_raev2_ig_impulse_response import (
        autocast_context,
        deterministic_noise,
        official_shifted_solver_grid,
    )
    from experiments.run_raev2_spectral_ig_mechanism_suite_v3 import (
        official_euler_x_prediction_step,
        official_prediction_components,
        official_sampler_model_kwargs,
    )
    from utils.model_utils import instantiate_from_config

    return locals()


def init_distributed() -> DistributedContext:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    try:
        dist.init_process_group("nccl", device_id=device)
    except TypeError:
        dist.init_process_group("nccl")
    return DistributedContext(
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        local_rank=local_rank,
        device=device,
    )


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def gather_object(value: Any, ctx: DistributedContext) -> list[Any]:
    gathered: list[Any] = [None for _ in range(ctx.world_size)]
    dist.all_gather_object(gathered, value)
    return gathered


def prepare_output(path: Path, *, resume: bool, overwrite: bool) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() and overwrite:
        shutil.rmtree(resolved)
    if resolved.exists() and any(resolved.iterdir()) and not resume:
        raise FileExistsError(
            f"Refusing non-empty output root {resolved}; use a new versioned path, "
            "--resume, or --overwrite."
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def load_stage2(
    args: argparse.Namespace,
    config: Any,
    ctx: DistributedContext,
    symbols: dict[str, Any],
) -> torch.nn.Module:
    model = symbols["instantiate_from_config"](config.stage_2).to(ctx.device).eval()
    model.requires_grad_(False)
    checkpoint = torch.load(
        args.checkpoint.expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    symbols["validate_full_stage2_checkpoint"](checkpoint)
    model.load_state_dict(checkpoint[args.state_key], strict=True)
    del checkpoint
    return model


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Fast RAEv2 input-frequency-axis extrapolation screen.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--stages",
        type=parse_stages,
        default=parse_stages("teacher,sample,decode,report"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    model = parser.add_argument_group("model and data")
    model.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    model.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    model.add_argument("--state-key", choices=("ema", "model"), default="ema")
    model.add_argument("--packed-data-path", type=Path, default=DEFAULT_PACKED_DATA)
    model.add_argument("--parquet-data-path", type=Path, default=DEFAULT_PARQUET_DATA)
    model.add_argument("--dino-ckpt-dir", type=Path, default=DEFAULT_DINO_CKPT)
    model.add_argument("--dino-repo-dir", type=Path, default=DEFAULT_DINO_REPO)
    model.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    model.add_argument("--per-rank-batch", type=int, default=1)
    model.add_argument("--seed", type=int, default=20260807)

    teacher = parser.add_argument_group("teacher-forced screen")
    teacher.add_argument("--teacher-samples", type=int, default=32)
    teacher.add_argument(
        "--teacher-steps",
        type=parse_csv_ints,
        default=parse_csv_ints("30,50,70,85,95,98"),
        help="Official solver input indices, 0-based.",
    )
    teacher.add_argument(
        "--blur-sigmas",
        type=lambda value: parse_csv_floats(value),
        default=parse_csv_floats("0.35,0.70,1.00"),
        help="Spatial Gaussian std on the 16x16 latent patch grid.",
    )
    teacher.add_argument(
        "--alpha-sweep",
        type=lambda value: parse_csv_floats(value),
        default=parse_csv_floats("-0.5,0.25,0.5,1.0"),
    )
    teacher.add_argument("--teacher-noise-seed", type=int, default=20264807)

    sample = parser.add_argument_group("tiny recursive rollout")
    sample.add_argument("--sample-count", type=int, default=8)
    sample.add_argument(
        "--sample-alphas",
        type=lambda value: parse_csv_floats(value),
        default=parse_csv_floats("-0.5,0.5,1.0"),
    )
    sample.add_argument("--sample-blur-sigma", type=float, default=0.70)
    sample.add_argument("--freq-t-min", type=float, default=0.10)
    sample.add_argument("--freq-t-max", type=float, default=0.95)
    sample.add_argument("--sample-noise-seed", type=int, default=20265807)
    sample.add_argument("--preview-count", type=int, default=8)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Shared model evaluation helpers.
# ---------------------------------------------------------------------------


def model_components(
    *,
    model: torch.nn.Module,
    state: torch.Tensor,
    labels: torch.Tensor,
    time: float,
    config: Any,
    precision: str,
    symbols: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    time_batch = torch.full(
        (len(state),), float(time), device=state.device, dtype=torch.float32
    )
    with symbols["autocast_context"](precision):
        output = model(
            state,
            time_batch,
            context=labels,
            attn_mask=None,
            **symbols["official_sampler_model_kwargs"](
                config, len(state), state.device
            ),
        )
    full, base = symbols["split_internal_guidance_output"](output)
    if base is None:
        raise RuntimeError("checkpoint has no RAEv2 internal-guidance base head")
    baseline, gap = symbols["official_prediction_components"](
        full,
        base,
        time=float(time),
        interval=(
            float(config.guidance.ig.t_min),
            float(config.guidance.ig.t_max),
        ),
    )
    return baseline.float(), gap.float(), base.float()


def frequency_direction(
    *,
    model: torch.nn.Module,
    state: torch.Tensor,
    labels: torch.Tensor,
    time: float,
    baseline: torch.Tensor,
    sigma: float,
    config: Any,
    precision: str,
    symbols: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    counterfactual, blurred_clean = clean_frequency_counterfactual(
        state, baseline, time, sigma
    )
    degraded_prediction, degraded_gap, _ = model_components(
        model=model,
        state=counterfactual,
        labels=labels,
        time=time,
        config=config,
        precision=precision,
        symbols=symbols,
    )
    direction = baseline - degraded_prediction
    return direction, counterfactual, blurred_clean, degraded_gap


# ---------------------------------------------------------------------------
# Teacher-forced screen.
# ---------------------------------------------------------------------------


def teacher_rows_for_direction(
    *,
    sample_ids: np.ndarray,
    step: int,
    time: float,
    sigma: float,
    kind: str,
    z: torch.Tensor,
    state: torch.Tensor,
    baseline: torch.Tensor,
    gap: torch.Tensor,
    direction: torch.Tensor,
    counterfactual: torch.Tensor,
    blurred_clean: torch.Tensor,
    counterfactual_source_clean: torch.Tensor,
    oracle_direction: torch.Tensor | None,
    alpha_sweep: Sequence[float],
) -> list[dict[str, Any]]:
    residual = z.float() - baseline.float()
    alpha_star = optimal_alpha(residual, direction)
    cosine_target = rowwise_cosine(residual, direction)
    cosine_gap = rowwise_cosine(gap, direction)
    rows: list[dict[str, Any]] = []
    noise_error = counterfactual_noise_error(
        state, counterfactual_source_clean, counterfactual, blurred_clean, time
    )
    oracle_cosine = (
        rowwise_cosine(direction, oracle_direction)
        if oracle_direction is not None
        else torch.full_like(alpha_star, float("nan"))
    )
    base_mse = (
        baseline.double() - z.double()
    ).flatten(1).square().mean(dim=1)
    for local_index, sample_id in enumerate(sample_ids.tolist()):
        row: dict[str, Any] = {
            "sample_id": int(sample_id),
            "step": int(step),
            "time": float(time),
            "sigma": float(sigma),
            "tau_sigma_squared": float(sigma) ** 2,
            "construction": str(kind),
            "alpha_star": float(alpha_star[local_index].cpu()),
            "target_direction_cosine": float(cosine_target[local_index].cpu()),
            "model_gap_cosine": float(cosine_gap[local_index].cpu()),
            "pred_oracle_direction_cosine": float(oracle_cosine[local_index].cpu()),
            "direction_rms": float(rowwise_rms(direction)[local_index].cpu()),
            "target_residual_rms": float(rowwise_rms(residual)[local_index].cpu()),
            "state_perturbation_relative_rms": float(
                rowwise_rms(counterfactual - state)[local_index]
                / rowwise_rms(state)[local_index].clamp_min(EPS)
            ),
            "noise_preservation_rms_error": float(noise_error[local_index].cpu()),
            "base_mse": float(base_mse[local_index].cpu()),
        }
        for alpha in alpha_sweep:
            row[f"relative_mse_change_alpha_{alpha_tag(alpha)}"] = float(
                relative_mse_change(baseline, z, direction, alpha)[
                    local_index
                ].cpu()
            )
        rows.append(row)
    return rows


def run_teacher_screen(
    *,
    args: argparse.Namespace,
    config: Any,
    ctx: DistributedContext,
    model: torch.nn.Module,
    grid: torch.Tensor,
    output_root: Path,
    symbols: dict[str, Any],
) -> None:
    stage_dir = output_root / "01_teacher_screen"
    stage_dir.mkdir(parents=True, exist_ok=True)
    shard_path = stage_dir / f"teacher_rank{ctx.rank:02d}.csv"
    if shard_path.is_file() and args.resume:
        barrier()
        if ctx.rank == 0:
            merge_teacher_results(args, ctx, output_root)
        barrier()
        return

    labels_all = symbols["build_requested_labels"](
        args.teacher_samples, int(config.misc.num_classes)
    )
    if ctx.rank == 0:
        selected_rows = symbols["select_matching_imagenet_rows"](
            args.parquet_data_path.expanduser().resolve(),
            labels_all,
            args.seed + 31,
        )
    else:
        selected_rows = np.empty(args.teacher_samples, dtype=np.int64)
    row_tensor = torch.from_numpy(selected_rows).to(ctx.device)
    dist.broadcast(row_tensor, src=0)
    selected_rows = row_tensor.cpu().numpy().astype(np.int64)

    local_ids = np.arange(ctx.rank, args.teacher_samples, ctx.world_size, dtype=np.int64)
    local_rows = selected_rows[local_ids]
    local_labels_np = labels_all[local_ids]
    latent_size = tuple(int(value) for value in config.misc.latent_size)
    local_noise = symbols["deterministic_noise"](
        local_ids, latent_size, seed=args.teacher_noise_seed
    )

    dataset = symbols["DeterministicImageNetPacked"](
        args.packed_data_path.expanduser().resolve(),
        split="train",
        image_size=int(config.training.image_size),
        horizontal_flip=False,
    )
    rae = symbols["instantiate_from_config"](config.stage_1).to(ctx.device).eval()
    rae.requires_grad_(False)
    if hasattr(rae, "decoder"):
        del rae.decoder

    selected_steps = tuple(sorted(set(int(item) for item in args.teacher_steps)))
    if any(step < 0 or step >= len(grid) - 1 for step in selected_steps):
        raise ValueError("teacher steps must lie inside the official solver inputs")
    sigmas = tuple(sorted(set(float(item) for item in args.blur_sigmas)))
    if any(sigma <= 0 for sigma in sigmas):
        raise ValueError("teacher blur sigmas must be positive")

    all_rows: list[dict[str, Any]] = []
    slope_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for start in range(0, len(local_ids), args.per_rank_batch):
            end = min(start + args.per_rank_batch, len(local_ids))
            ids = local_ids[start:end]
            images = []
            for source_row, expected_label in zip(
                local_rows[start:end].tolist(), local_labels_np[start:end].tolist()
            ):
                image, actual_label, _ = dataset[int(source_row)]
                if int(actual_label) != int(expected_label):
                    raise RuntimeError(
                        f"ImageNet label mismatch: row {source_row} has {actual_label}, "
                        f"expected {expected_label}"
                    )
                images.append(image)
            image_batch = torch.stack(images).to(ctx.device)
            with symbols["autocast_context"](args.precision):
                z = rae.encode(image_batch).float()
            noise = local_noise[start:end].to(ctx.device)
            labels = torch.from_numpy(local_labels_np[start:end]).to(
                ctx.device, torch.long
            )

            for step in selected_steps:
                time = float(grid[step])
                state = (1.0 - time) * z + time * noise
                baseline, gap, _ = model_components(
                    model=model,
                    state=state,
                    labels=labels,
                    time=time,
                    config=config,
                    precision=args.precision,
                    symbols=symbols,
                )
                pred_directions: list[torch.Tensor] = []
                oracle_directions: list[torch.Tensor] = []
                pred_payloads: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
                oracle_payloads: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

                for sigma in sigmas:
                    pred_direction, pred_state, pred_blurred, _ = frequency_direction(
                        model=model,
                        state=state,
                        labels=labels,
                        time=time,
                        baseline=baseline,
                        sigma=sigma,
                        config=config,
                        precision=args.precision,
                        symbols=symbols,
                    )
                    oracle_blurred = gaussian_blur_latent(z, sigma)
                    oracle_state = state + (1.0 - time) * (oracle_blurred - z)
                    oracle_prediction, _, _ = model_components(
                        model=model,
                        state=oracle_state,
                        labels=labels,
                        time=time,
                        config=config,
                        precision=args.precision,
                        symbols=symbols,
                    )
                    oracle_direction = baseline - oracle_prediction

                    pred_directions.append(pred_direction)
                    oracle_directions.append(oracle_direction)
                    pred_payloads.append((pred_state, pred_blurred, oracle_direction))
                    oracle_payloads.append((oracle_state, oracle_blurred, pred_direction))

                for sigma_index, sigma in enumerate(sigmas):
                    pred_state, pred_blurred, oracle_direction = pred_payloads[sigma_index]
                    all_rows.extend(
                        teacher_rows_for_direction(
                            sample_ids=ids,
                            step=step,
                            time=time,
                            sigma=sigma,
                            kind="predicted_clean_axis",
                            z=z,
                            state=state,
                            baseline=baseline,
                            gap=gap,
                            direction=pred_directions[sigma_index],
                            counterfactual=pred_state,
                            blurred_clean=pred_blurred,
                            counterfactual_source_clean=baseline,
                            oracle_direction=oracle_direction,
                            alpha_sweep=args.alpha_sweep,
                        )
                    )
                    oracle_state, oracle_blurred, pred_direction = oracle_payloads[sigma_index]
                    all_rows.extend(
                        teacher_rows_for_direction(
                            sample_ids=ids,
                            step=step,
                            time=time,
                            sigma=sigma,
                            kind="oracle_clean_axis",
                            z=z,
                            state=state,
                            baseline=baseline,
                            gap=gap,
                            direction=oracle_directions[sigma_index],
                            counterfactual=oracle_state,
                            blurred_clean=oracle_blurred,
                            counterfactual_source_clean=z,
                            oracle_direction=pred_direction,
                            alpha_sweep=args.alpha_sweep,
                        )
                    )

                for kind, directions in (
                    ("predicted_clean_axis", pred_directions),
                    ("oracle_clean_axis", oracle_directions),
                ):
                    for sigma_index in range(len(sigmas) - 1):
                        left_sigma = sigmas[sigma_index]
                        right_sigma = sigmas[sigma_index + 1]
                        # Gaussian blur is smooth in tau=sigma^2, not sigma.
                        left_slope = directions[sigma_index] / max(left_sigma**2, EPS)
                        right_slope = directions[sigma_index + 1] / max(
                            right_sigma**2, EPS
                        )
                        cosines = rowwise_cosine(left_slope, right_slope)
                        norm_ratio = rowwise_rms(right_slope) / rowwise_rms(
                            left_slope
                        ).clamp_min(EPS)
                        for local_index, sample_id in enumerate(ids.tolist()):
                            slope_rows.append(
                                {
                                    "sample_id": int(sample_id),
                                    "step": int(step),
                                    "time": float(time),
                                    "construction": kind,
                                    "left_sigma": left_sigma,
                                    "right_sigma": right_sigma,
                                    "slope_direction_cosine": float(
                                        cosines[local_index].cpu()
                                    ),
                                    "slope_rms_ratio": float(
                                        norm_ratio[local_index].cpu()
                                    ),
                                }
                            )
            if ctx.rank == 0:
                print(f"[teacher] local {end}/{len(local_ids)}", flush=True)

    pd.DataFrame(all_rows).to_csv(shard_path, index=False)
    pd.DataFrame(slope_rows).to_csv(
        stage_dir / f"axis_smoothness_rank{ctx.rank:02d}.csv", index=False
    )
    del rae, dataset
    gc.collect()
    torch.cuda.empty_cache()
    barrier()
    if ctx.rank == 0:
        merge_teacher_results(args, ctx, output_root)
    barrier()


def merge_teacher_results(
    args: argparse.Namespace,
    ctx: DistributedContext,
    output_root: Path,
) -> None:
    directory = output_root / "01_teacher_screen"
    frame = pd.concat(
        [
            pd.read_csv(directory / f"teacher_rank{rank:02d}.csv")
            for rank in range(ctx.world_size)
        ],
        ignore_index=True,
    )
    smooth = pd.concat(
        [
            pd.read_csv(directory / f"axis_smoothness_rank{rank:02d}.csv")
            for rank in range(ctx.world_size)
        ],
        ignore_index=True,
    )
    frame.to_csv(directory / "teacher_per_sample.csv", index=False)
    smooth.to_csv(directory / "axis_smoothness_per_sample.csv", index=False)

    aggregation: dict[str, tuple[str, str]] = {
        "samples": ("sample_id", "nunique"),
        "alpha_star_global_mean": ("alpha_star", "mean"),
        "alpha_star_median": ("alpha_star", "median"),
        "positive_alpha_star_fraction": (
            "alpha_star",
            lambda series: float((series > 0).mean()),
        ),
        "target_direction_cosine_mean": ("target_direction_cosine", "mean"),
        "target_direction_cosine_median": (
            "target_direction_cosine",
            "median",
        ),
        "model_gap_cosine_mean": ("model_gap_cosine", "mean"),
        "pred_oracle_direction_cosine_mean": (
            "pred_oracle_direction_cosine",
            "mean",
        ),
        "direction_rms_mean": ("direction_rms", "mean"),
        "state_perturbation_relative_rms_mean": (
            "state_perturbation_relative_rms",
            "mean",
        ),
        "noise_preservation_rms_error_max": (
            "noise_preservation_rms_error",
            "max",
        ),
    }
    for alpha in args.alpha_sweep:
        column = f"relative_mse_change_alpha_{alpha_tag(alpha)}"
        aggregation[f"{column}_mean"] = (column, "mean")
        aggregation[f"{column}_improved_fraction"] = (
            column,
            lambda series: float((series < 0).mean()),
        )
    summary = (
        frame.groupby(["construction", "step", "time", "sigma"], as_index=False)
        .agg(**aggregation)
        .sort_values(["construction", "step", "sigma"])
    )
    summary.to_csv(directory / "teacher_summary.csv", index=False)
    smooth_summary = (
        smooth.groupby(
            [
                "construction",
                "step",
                "time",
                "left_sigma",
                "right_sigma",
            ],
            as_index=False,
        )
        .agg(
            slope_direction_cosine_mean=("slope_direction_cosine", "mean"),
            slope_direction_cosine_median=("slope_direction_cosine", "median"),
            slope_rms_ratio_mean=("slope_rms_ratio", "mean"),
        )
        .sort_values(["construction", "step", "left_sigma"])
    )
    smooth_summary.to_csv(directory / "axis_smoothness_summary.csv", index=False)


# ---------------------------------------------------------------------------
# Tiny same-noise recursive rollout and decoding.
# ---------------------------------------------------------------------------


def build_conditions(args: argparse.Namespace) -> tuple[dict[str, Any], ...]:
    conditions: list[dict[str, Any]] = [
        {"name": "no_ig", "kind": "no_ig"},
        {"name": "scalar_ig", "kind": "scalar_ig"},
    ]
    for alpha in args.sample_alphas:
        conditions.append(
            {
                "name": f"freq_alpha_{alpha_tag(alpha)}",
                "kind": "frequency",
                "alpha": float(alpha),
            }
        )
    return tuple(conditions)


def endpoint_path(output_root: Path, condition: str, rank: int) -> Path:
    return output_root / "02_tiny_rollout" / condition / f"endpoint_rank{rank:02d}.npy"


def ids_path(output_root: Path, condition: str, rank: int) -> Path:
    return output_root / "02_tiny_rollout" / condition / f"ids_rank{rank:02d}.npy"


def images_path(output_root: Path, condition: str, rank: int) -> Path:
    return output_root / "02_tiny_rollout" / condition / f"images_rank{rank:02d}.npy"


def run_tiny_rollout(
    *,
    args: argparse.Namespace,
    config: Any,
    ctx: DistributedContext,
    model: torch.nn.Module,
    grid: torch.Tensor,
    output_root: Path,
    symbols: dict[str, Any],
) -> None:
    conditions = build_conditions(args)
    local_ids = np.arange(ctx.rank, args.sample_count, ctx.world_size, dtype=np.int64)
    labels_all = symbols["build_requested_labels"](
        args.sample_count, int(config.misc.num_classes)
    )
    latent_size = tuple(int(value) for value in config.misc.latent_size)
    t_steps = grid.to(ctx.device, torch.float32)
    official_scale = float(config.guidance.ig.scale)
    official_interval = (
        float(config.guidance.ig.t_min),
        float(config.guidance.ig.t_max),
    )
    diagnostics: list[dict[str, Any]] = []

    for condition in conditions:
        path = endpoint_path(output_root, condition["name"], ctx.rank)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and ids_path(
            output_root, condition["name"], ctx.rank
        ).is_file() and args.resume:
            barrier()
            continue
        endpoint_map = np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=np.float16,
            shape=(len(local_ids), *latent_size),
        )
        for start in range(0, len(local_ids), args.per_rank_batch):
            end = min(start + args.per_rank_batch, len(local_ids))
            ids = local_ids[start:end]
            labels = torch.from_numpy(labels_all[ids]).to(ctx.device, torch.long)
            state = symbols["deterministic_noise"](
                ids, latent_size, seed=args.sample_noise_seed
            ).to(ctx.device)
            q_rms_sum = 0.0
            q_gap_cosine_sum = 0.0
            q_count = 0
            with torch.inference_mode():
                for step in range(len(grid) - 1):
                    time = float(t_steps[step].item())
                    baseline, gap, base = model_components(
                        model=model,
                        state=state,
                        labels=labels,
                        time=time,
                        config=config,
                        precision=args.precision,
                        symbols=symbols,
                    )
                    if condition["kind"] == "no_ig":
                        guided = baseline
                    elif condition["kind"] == "scalar_ig" and (
                        official_interval[0] <= time <= official_interval[1]
                    ):
                        guided = base + official_scale * gap
                    elif condition["kind"] == "frequency" and (
                        float(args.freq_t_min) <= time <= float(args.freq_t_max)
                    ):
                        direction, _, _, _ = frequency_direction(
                            model=model,
                            state=state,
                            labels=labels,
                            time=time,
                            baseline=baseline,
                            sigma=float(args.sample_blur_sigma),
                            config=config,
                            precision=args.precision,
                            symbols=symbols,
                        )
                        guided = baseline + float(condition["alpha"]) * direction
                        q_rms_sum += float(rowwise_rms(direction).mean().cpu())
                        q_gap_cosine_sum += float(
                            rowwise_cosine(direction, gap).mean().cpu()
                        )
                        q_count += 1
                    else:
                        guided = baseline
                    state = symbols["official_euler_x_prediction_step"](
                        state,
                        guided,
                        t_steps=t_steps,
                        step=step,
                        t_eps=float(config.transport.t_eps),
                    )
            endpoint_map[start:end] = state.cpu().numpy().astype(np.float16)
            diagnostics.append(
                {
                    "rank": ctx.rank,
                    "condition": condition["name"],
                    "batch_start": int(start),
                    "batch_end": int(end),
                    "active_frequency_steps": int(q_count),
                    "mean_frequency_direction_rms": q_rms_sum / max(q_count, 1),
                    "mean_frequency_model_gap_cosine": q_gap_cosine_sum
                    / max(q_count, 1),
                }
            )
            if ctx.rank == 0:
                print(
                    f"[sample:{condition['name']}] local {end}/{len(local_ids)}",
                    flush=True,
                )
        endpoint_map.flush()
        np.save(
            ids_path(output_root, condition["name"], ctx.rank),
            local_ids,
            allow_pickle=False,
        )
        barrier()

    pd.DataFrame(diagnostics).to_csv(
        output_root / "02_tiny_rollout" / f"diagnostics_rank{ctx.rank:02d}.csv",
        index=False,
    )
    barrier()
    if ctx.rank == 0:
        diagnostics_all = pd.concat(
            [
                pd.read_csv(
                    output_root
                    / "02_tiny_rollout"
                    / f"diagnostics_rank{rank:02d}.csv"
                )
                for rank in range(ctx.world_size)
            ],
            ignore_index=True,
        )
        diagnostics_all.to_csv(
            output_root / "02_tiny_rollout" / "rollout_diagnostics.csv",
            index=False,
        )
    barrier()


def save_preview(images: np.ndarray, path: Path, count: int) -> None:
    count = min(int(count), len(images))
    if count <= 0:
        return
    columns = min(8, count)
    rows = int(math.ceil(count / columns))
    h, w = images.shape[1:3]
    canvas = np.zeros((rows * h, columns * w, 3), dtype=np.uint8)
    for index, image in enumerate(images[:count]):
        row, column = divmod(index, columns)
        canvas[row * h : (row + 1) * h, column * w : (column + 1) * w] = image
    Image.fromarray(canvas, mode="RGB").save(path)


def save_comparison_preview(
    condition_images: dict[str, np.ndarray],
    path: Path,
    count: int,
) -> None:
    if not condition_images:
        return
    names = list(condition_images)
    count = min(int(count), *(len(condition_images[name]) for name in names))
    if count <= 0:
        return
    sample = condition_images[names[0]]
    h, w = sample.shape[1:3]
    label_height = 28
    canvas = np.zeros(
        (len(names) * (h + label_height), count * w, 3), dtype=np.uint8
    )
    from PIL import ImageDraw

    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    for row, name in enumerate(names):
        y0 = row * (h + label_height)
        draw.text((4, y0 + 5), name, fill=(255, 255, 255))
        data = condition_images[name]
        for column in range(count):
            tile = Image.fromarray(data[column])
            image.paste(tile, (column * w, y0 + label_height))
    image.save(path)


def decode_tiny_rollout(
    *,
    args: argparse.Namespace,
    config: Any,
    ctx: DistributedContext,
    output_root: Path,
    symbols: dict[str, Any],
) -> None:
    conditions = build_conditions(args)
    rae = symbols["instantiate_from_config"](config.stage_1).to(ctx.device).eval()
    rae.requires_grad_(False)
    if hasattr(rae, "encoder"):
        del rae.encoder
    image_size = int(config.training.image_size)

    for condition in conditions:
        endpoint = endpoint_path(output_root, condition["name"], ctx.rank)
        ids_file = ids_path(output_root, condition["name"], ctx.rank)
        image_file = images_path(output_root, condition["name"], ctx.rank)
        endpoints = np.load(endpoint, mmap_mode="r", allow_pickle=False)
        ids = np.load(ids_file, allow_pickle=False)
        images = np.lib.format.open_memmap(
            image_file,
            mode="w+",
            dtype=np.uint8,
            shape=(len(endpoints), image_size, image_size, 3),
        )
        with torch.inference_mode():
            for start in range(0, len(endpoints), args.per_rank_batch):
                end = min(start + args.per_rank_batch, len(endpoints))
                latent = torch.from_numpy(
                    np.asarray(endpoints[start:end], dtype=np.float32)
                ).to(ctx.device)
                with symbols["autocast_context"](args.precision):
                    decoded = rae.decode(latent).float().clamp(0, 1)
                images[start:end] = (
                    decoded.mul(255)
                    .permute(0, 2, 3, 1)
                    .to("cpu", torch.uint8)
                    .numpy()
                )
        images.flush()
        barrier()
        if ctx.rank == 0:
            merged = np.empty(
                (args.sample_count, image_size, image_size, 3), dtype=np.uint8
            )
            seen = np.zeros(args.sample_count, dtype=bool)
            for rank in range(ctx.world_size):
                rank_ids = np.load(
                    ids_path(output_root, condition["name"], rank),
                    allow_pickle=False,
                )
                rank_images = np.load(
                    images_path(output_root, condition["name"], rank),
                    mmap_mode="r",
                )
                merged[rank_ids] = rank_images
                seen[rank_ids] = True
            if not seen.all():
                raise RuntimeError("incomplete distributed sample IDs")
            merged_path = (
                output_root
                / "02_tiny_rollout"
                / condition["name"]
                / "samples.npz"
            )
            np.savez_compressed(merged_path, samples=merged)
            save_preview(
                merged,
                merged_path.with_name("preview.png"),
                args.preview_count,
            )
        barrier()
        # Endpoints are intentionally retained until report generation because
        # the report computes paired latent distances from no-IG.  They are tiny
        # in this screening experiment and can be removed manually afterwards.
        barrier()

    if ctx.rank == 0:
        condition_images: dict[str, np.ndarray] = {}
        for condition in conditions:
            payload = np.load(
                output_root
                / "02_tiny_rollout"
                / condition["name"]
                / "samples.npz",
                allow_pickle=False,
            )
            condition_images[condition["name"]] = payload["samples"]
        save_comparison_preview(
            condition_images,
            output_root / "02_tiny_rollout" / "same_noise_comparison.png",
            args.preview_count,
        )
    barrier()
    del rae
    gc.collect()
    torch.cuda.empty_cache()


def endpoint_pairwise_summary(
    args: argparse.Namespace,
    ctx: DistributedContext,
    output_root: Path,
) -> pd.DataFrame:
    conditions = build_conditions(args)
    rows = []
    for condition in conditions:
        local_values = []
        local_ids = []
        for rank in range(ctx.world_size):
            endpoint = endpoint_path(output_root, condition["name"], rank)
            ids_file = ids_path(output_root, condition["name"], rank)
            if not endpoint.is_file() or not ids_file.is_file():
                continue
            local_values.append(np.load(endpoint, mmap_mode="r", allow_pickle=False))
            local_ids.append(np.load(ids_file, allow_pickle=False))
        if not local_values:
            continue
        merged = np.empty((args.sample_count, *local_values[0].shape[1:]), dtype=np.float32)
        for values, ids in zip(local_values, local_ids):
            merged[ids] = np.asarray(values, dtype=np.float32)
        rows.append({"condition": condition["name"], "endpoint": merged})
    if not rows:
        return pd.DataFrame()
    baseline = next(item["endpoint"] for item in rows if item["condition"] == "no_ig")
    summary = []
    for item in rows:
        delta = item["endpoint"].astype(np.float64) - baseline.astype(np.float64)
        per_sample = np.sqrt(np.mean(delta**2, axis=(1, 2, 3)))
        summary.append(
            {
                "condition": item["condition"],
                "endpoint_rms_from_no_ig_mean": float(per_sample.mean()),
                "endpoint_rms_from_no_ig_std": float(per_sample.std(ddof=1))
                if len(per_sample) > 1
                else 0.0,
                "endpoint_rms": float(
                    np.sqrt(np.mean(item["endpoint"].astype(np.float64) ** 2))
                ),
            }
        )
    return pd.DataFrame(summary)


# ---------------------------------------------------------------------------
# Report.
# ---------------------------------------------------------------------------


def build_report(args: argparse.Namespace, ctx: DistributedContext, output_root: Path) -> None:
    if ctx.rank != 0:
        return
    report: dict[str, Any] = {
        "protocol": PROTOCOL,
        "script_version": SCRIPT_VERSION,
        "definition": {
            "counterfactual_state": "x_tau = x + (1-t)*(H_tau(y0)-y0)",
            "frequency_direction": "q_tau = y0 - F(x_tau,t)",
            "extrapolated_prediction": "y_ext = y0 + alpha*q_tau",
            "axis_coordinate": "tau = sigma_pixels^2",
        },
        "configuration": {
            "teacher_samples": args.teacher_samples,
            "teacher_steps": list(args.teacher_steps),
            "blur_sigmas": list(args.blur_sigmas),
            "alpha_sweep": list(args.alpha_sweep),
            "sample_count": args.sample_count,
            "sample_alphas": list(args.sample_alphas),
            "sample_blur_sigma": args.sample_blur_sigma,
            "frequency_interval": [args.freq_t_min, args.freq_t_max],
            "checkpoint": str(args.checkpoint.expanduser().resolve()),
            "checkpoint_sha256": file_sha256(args.checkpoint.expanduser().resolve()),
        },
    }

    teacher_summary_path = output_root / "01_teacher_screen" / "teacher_summary.csv"
    smooth_path = output_root / "01_teacher_screen" / "axis_smoothness_summary.csv"
    if teacher_summary_path.is_file():
        teacher = pd.read_csv(teacher_summary_path)
        predicted = teacher[teacher.construction == "predicted_clean_axis"]
        report["teacher_screen"] = {
            "mean_alpha_star": float(predicted.alpha_star_global_mean.mean()),
            "positive_alpha_star_fraction": float(
                predicted.positive_alpha_star_fraction.mean()
            ),
            "mean_target_direction_cosine": float(
                predicted.target_direction_cosine_mean.mean()
            ),
            "mean_model_gap_cosine": float(predicted.model_gap_cosine_mean.mean()),
            "mean_pred_oracle_direction_cosine": float(
                predicted.pred_oracle_direction_cosine_mean.mean()
            ),
            "maximum_noise_preservation_rms_error": float(
                predicted.noise_preservation_rms_error_max.max()
            ),
        }
        for alpha in args.alpha_sweep:
            column = f"relative_mse_change_alpha_{alpha_tag(alpha)}_mean"
            improved = f"relative_mse_change_alpha_{alpha_tag(alpha)}_improved_fraction"
            report["teacher_screen"][f"alpha_{alpha}_mean_relative_mse_change"] = float(
                predicted[column].mean()
            )
            report["teacher_screen"][f"alpha_{alpha}_improved_fraction"] = float(
                predicted[improved].mean()
            )
    if smooth_path.is_file():
        smooth = pd.read_csv(smooth_path)
        predicted = smooth[smooth.construction == "predicted_clean_axis"]
        report["axis_smoothness"] = {
            "mean_slope_direction_cosine": float(
                predicted.slope_direction_cosine_mean.mean()
            ),
            "mean_slope_rms_ratio": float(predicted.slope_rms_ratio_mean.mean()),
        }

    endpoint_summary = endpoint_pairwise_summary(args, ctx, output_root)
    if not endpoint_summary.empty:
        endpoint_summary.to_csv(
            output_root / "02_tiny_rollout" / "endpoint_pairwise_summary.csv",
            index=False,
        )
        report["tiny_rollout"] = endpoint_summary.to_dict(orient="records")

    # These thresholds are explicitly heuristic screening rules, not claims.
    teacher_screen = report.get("teacher_screen", {})
    axis = report.get("axis_smoothness", {})
    report["heuristic_screening"] = {
        "axis_is_locally_coherent": bool(
            axis.get("mean_slope_direction_cosine", -1.0) >= 0.80
        ),
        "self_constructed_axis_matches_oracle": bool(
            teacher_screen.get("mean_pred_oracle_direction_cosine", -1.0) >= 0.70
        ),
        "positive_extrapolation_often_points_to_target": bool(
            teacher_screen.get("positive_alpha_star_fraction", 0.0) >= 0.60
        ),
        "frequency_direction_is_not_nearly_identical_to_model_gap": bool(
            abs(teacher_screen.get("mean_model_gap_cosine", 1.0)) <= 0.90
        ),
        "note": "Thresholds are engineering screening heuristics, not statistical tests.",
    }
    json_dump(output_root / "final_report.json", report)

    lines = [
        "RAEv2 Input-Frequency-Axis Extrapolation Screen v1",
        "===================================================",
        "",
        "This is a fast screening experiment. It does not compute FID/KID and",
        "does not establish generation-quality improvement.",
        "",
        "Core definition:",
        "  x_tau = x + (1-t) * (H_tau(y0) - y0)",
        "  q_tau = y0 - Full(x_tau, t)",
        "  y_ext = y0 + alpha * q_tau",
        "",
    ]
    if teacher_screen:
        lines.extend(
            [
                "Teacher-forced aggregate:",
                f"  mean alpha*: {teacher_screen.get('mean_alpha_star', float('nan')):.6f}",
                "  positive alpha* fraction: "
                f"{teacher_screen.get('positive_alpha_star_fraction', float('nan')):.4f}",
                "  target-direction cosine: "
                f"{teacher_screen.get('mean_target_direction_cosine', float('nan')):.4f}",
                "  predicted/oracle direction cosine: "
                f"{teacher_screen.get('mean_pred_oracle_direction_cosine', float('nan')):.4f}",
                "  frequency/model-gap cosine: "
                f"{teacher_screen.get('mean_model_gap_cosine', float('nan')):.4f}",
                "",
            ]
        )
    if axis:
        lines.extend(
            [
                "Axis smoothness:",
                "  mean cosine between adjacent finite-difference slopes: "
                f"{axis.get('mean_slope_direction_cosine', float('nan')):.4f}",
                "",
            ]
        )
    lines.extend(
        [
            "Heuristic screening flags:",
            *[
                f"  {key}: {value}"
                for key, value in report["heuristic_screening"].items()
                if key != "note"
            ],
            "",
            "Interpretation must use teacher_summary.csv, axis_smoothness_summary.csv,",
            "same_noise_comparison.png, and endpoint_pairwise_summary.csv together.",
        ]
    )
    (output_root / "final_report.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    if args.teacher_samples <= 0 or args.sample_count <= 0:
        raise ValueError("sample counts must be positive")
    if args.per_rank_batch <= 0:
        raise ValueError("per-rank batch must be positive")
    if not 0.0 <= args.freq_t_min <= args.freq_t_max <= 1.0:
        raise ValueError("frequency interval must lie in [0,1]")
    if args.sample_blur_sigma <= 0:
        raise ValueError("sample blur sigma must be positive")

    symbols = repo_symbols()
    symbols["install_raev2_decoder_config_compat"]()
    os.environ["DINOV3_CKPT_DIR"] = str(args.dino_ckpt_dir.expanduser().resolve())
    os.environ["DINOV3_REPO_DIR"] = str(args.dino_repo_dir.expanduser().resolve())

    ctx = init_distributed()
    if args.teacher_samples < ctx.world_size or args.sample_count < ctx.world_size:
        raise ValueError("teacher/sample counts must be at least world_size")
    output_root = prepare_output(
        args.output_root, resume=args.resume, overwrite=args.overwrite
    )
    if ctx.rank == 0:
        json_dump(
            output_root / "manifest.json",
            {
                "protocol": PROTOCOL,
                "script_version": SCRIPT_VERSION,
                "stages": list(args.stages),
                "args": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
            },
        )
    barrier()

    config = symbols["load_config"](args.config.expanduser().resolve())
    latent_size = tuple(int(value) for value in config.misc.latent_size)
    shift = math.sqrt(
        (config.misc.time_dist_shift_dim or math.prod(latent_size))
        / config.misc.time_dist_shift_base
    )
    grid = symbols["official_shifted_solver_grid"](
        int(config.sampler.num_steps), shift
    ).to(torch.float32)
    model = load_stage2(args, config, ctx, symbols)

    if "teacher" in args.stages:
        run_teacher_screen(
            args=args,
            config=config,
            ctx=ctx,
            model=model,
            grid=grid,
            output_root=output_root,
            symbols=symbols,
        )
    if "sample" in args.stages:
        run_tiny_rollout(
            args=args,
            config=config,
            ctx=ctx,
            model=model,
            grid=grid,
            output_root=output_root,
            symbols=symbols,
        )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    barrier()

    if "decode" in args.stages:
        decode_tiny_rollout(
            args=args,
            config=config,
            ctx=ctx,
            output_root=output_root,
            symbols=symbols,
        )
    if "report" in args.stages:
        barrier()
        build_report(args, ctx, output_root)
        barrier()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()