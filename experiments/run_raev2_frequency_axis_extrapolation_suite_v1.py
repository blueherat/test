#!/usr/bin/env python3
"""Comprehensive RAEv2 input-frequency-axis extrapolation suite (v1).

This suite evaluates frequency-axis extrapolation as a CLOSED-LOOP sampling
intervention rather than judging it primarily by teacher-forced clean-prediction
MSE.  It operationalizes the frequency axis

    y0        = Full_scale1(x_t, t)
    x_tau     = x_t + (1-t) * (H_tau(y0) - y0)
    y_tau     = Full_scale1(x_tau, t)
    g_freq    = (y0 - y_tau) / tau
    g_model   = Full - Base
    g_mix     = (g_model(x_t) - g_model(x_tau)) / tau

and updates the clean prediction with

    y_guided = model_guided + eta * g_freq
                           + [gamma * eta * g_mix, optional]

where ``eta`` is the actual extrapolation distance along negative blur time and
``tau = sigma**2``.  This avoids comparing incomparable fixed ``alpha`` values
across blur scales.

The experiment is intentionally complete and stageable:

1. reference
   Encode real ImageNet samples, construct teacher-forced states at selected
   solver inputs, and save compact linear projections of the state, scale-1
   prediction, and Full-Base gap.

2. rollout
   Run every protocol condition from identical deterministic noises and labels.
   Save the same compact features at every audit checkpoint, plus axis and
   spectral diagnostics.

3. metrics
   Compare rollout and teacher-forced distributions using PCA-space Fréchet,
   sliced Wasserstein, RBF MMD, energy distance, and cross-validated linear
   classifier two-sample tests.  Report the full path and normalized path AUC.

4. pulse
   From no-IG and official scalar-IG baselines, apply paired +/- frequency-axis
   interventions in early, middle, and late windows.  Recursive and replayed
   directions separate state-dependent feedback from propagation by the Full
   field.  Distribution derivatives are evaluated at the window end and final
   endpoint.

5. sample/decode/evaluate
   Generate same-noise ImageNet samples for all protocol conditions, decode with
   the official RAE decoder, and compute repository-standard IS/FID/KID.  Image
   metrics never enter condition fitting because this suite contains no fitted
   controller.

The exact scale-1 semantics and Euler arithmetic are imported from
``run_raev2_spectral_ig_mechanism_suite_v3.py``, whose explicit baseline was
verified bitwise against the released sampler.  Do not overwrite this file in a
future revision; create ``..._v2.py`` instead.
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
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from PIL import Image


PROTOCOL = "raev2_frequency_axis_extrapolation_suite_v1"
SCRIPT_VERSION = "v1"
EPS = 1e-12

ROOT = Path(__file__).resolve().parents[1]
RAEV2_SRC = ROOT / "external" / "RAEv2" / "src"
DEFAULT_CONFIG = ROOT / "experiments/configs/raev2_strict_lpl_dinov3l_k7.yaml"
DEFAULT_PROTOCOL_JSON = (
    ROOT / "experiments/configs/frequency_axis_extrapolation_suite_v1_full.json"
)
DEFAULT_CHECKPOINT = Path(
    "/data/users/zhoushunyu/eqvae/models/RAEv2/stage2/imagenet/"
    "dinov3l-k7/checkpoint.pt"
)
DEFAULT_PACKED_DATA = Path("/data/shared/imagenet-1k/random_access_v1")
DEFAULT_PARQUET_DATA = Path("/data/shared/imagenet-1k")
DEFAULT_REFERENCE = Path("/data/shared/adm_refs/VIRTUAL_imagenet256_labeled.npz")
DEFAULT_DINO_CKPT = Path("/data/users/zhoushunyu/eqvae/models/RAEv2/encoders/dinov3")
DEFAULT_DINO_REPO = Path("/data/users/zhoushunyu/eqvae/models/RAEv2/dinov3_repo")


# ---------------------------------------------------------------------------
# Pure helpers and protocol types.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device


@dataclasses.dataclass(frozen=True)
class AxisConfig:
    construction: str
    fixed_sigma: float
    target_relative_state_rms: float
    sigma_min: float
    sigma_max: float
    bisection_steps: int
    tau_floor: float


@dataclasses.dataclass(frozen=True)
class ProjectionConfig:
    channel_dim: int
    spatial_dim: int
    pca_dim: int
    seed: int


@dataclasses.dataclass(frozen=True)
class ConditionSpec:
    name: str
    model_gamma: float
    frequency_eta: float
    frequency_window: str
    frequency_t_min: float
    frequency_t_max: float
    interaction: bool
    audit: bool
    sample: bool


@dataclasses.dataclass(frozen=True)
class ProtocolConfig:
    raw: Mapping[str, Any]
    axis: AxisConfig
    projection: ProjectionConfig
    checkpoint_steps: tuple[int, ...]
    band_edges: tuple[float, ...]
    band_names: tuple[str, ...]
    windows: Mapping[str, tuple[float, float]]
    conditions: tuple[ConditionSpec, ...]
    metric_config: Mapping[str, Any]
    pulse_config: Mapping[str, Any]


@dataclasses.dataclass(frozen=True)
class ProjectionMatrices:
    channel: torch.Tensor  # [C, Pc]
    spatial: torch.Tensor  # [H*W, Ps]

    @property
    def feature_dim(self) -> int:
        return int(self.channel.shape[1] * self.spatial.shape[1])


@dataclasses.dataclass(frozen=True)
class FrequencyAxisResult:
    derivative: torch.Tensor
    raw_difference: torch.Tensor
    interaction_derivative: torch.Tensor
    sigma: torch.Tensor
    tau: torch.Tensor
    achieved_relative_state_rms: torch.Tensor
    degraded_prediction: torch.Tensor
    degraded_gap: torch.Tensor


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parse_stages(value: str) -> tuple[str, ...]:
    allowed = {
        "reference",
        "rollout",
        "metrics",
        "pulse",
        "sample",
        "decode",
        "evaluate",
        "report",
    }
    result = tuple(item.strip() for item in str(value).split(",") if item.strip())
    unknown = sorted(set(result) - allowed)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown stages: {unknown}")
    if not result:
        raise argparse.ArgumentTypeError("at least one stage is required")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Comprehensive RAEv2 frequency-axis extrapolation suite.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol-json", type=Path, default=DEFAULT_PROTOCOL_JSON)
    parser.add_argument(
        "--stages",
        type=parse_stages,
        default=parse_stages(
            "reference,rollout,metrics,pulse,sample,decode,evaluate,report"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    model = parser.add_argument_group("model and data")
    model.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    model.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    model.add_argument("--state-key", choices=("ema", "model"), default="ema")
    model.add_argument("--packed-data-path", type=Path, default=DEFAULT_PACKED_DATA)
    model.add_argument("--parquet-data-path", type=Path, default=DEFAULT_PARQUET_DATA)
    model.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    model.add_argument("--dino-ckpt-dir", type=Path, default=DEFAULT_DINO_CKPT)
    model.add_argument("--dino-repo-dir", type=Path, default=DEFAULT_DINO_REPO)
    model.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    model.add_argument("--per-rank-batch", type=int, default=1)

    counts = parser.add_argument_group("sample counts")
    counts.add_argument("--reference-samples", type=int, default=1024)
    counts.add_argument("--rollout-samples", type=int, default=1024)
    counts.add_argument("--pulse-samples", type=int, default=128)
    counts.add_argument("--sample-count", type=int, default=5000)
    counts.add_argument("--metric-batch-size", type=int, default=64)
    counts.add_argument("--preview-count", type=int, default=32)

    seeds = parser.add_argument_group("independent deterministic seeds")
    seeds.add_argument("--data-seed", type=int, default=20267807)
    seeds.add_argument("--reference-noise-seed", type=int, default=20268807)
    seeds.add_argument("--rollout-noise-seed", type=int, default=20269807)
    seeds.add_argument("--pulse-noise-seed", type=int, default=20270807)
    seeds.add_argument("--sample-noise-seed", type=int, default=20271807)
    seeds.add_argument("--metric-seed", type=int, default=20272807)

    storage = parser.add_argument_group("storage")
    storage.add_argument("--keep-endpoints", action="store_true")
    storage.add_argument(
        "--sample-condition-names",
        type=str,
        default="",
        help="Optional comma-separated subset overriding protocol sample=true flags.",
    )
    return parser.parse_args()


def _require_keys(mapping: Mapping[str, Any], keys: Iterable[str], where: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(f"missing protocol keys at {where}: {missing}")


def load_protocol(path: Path) -> ProtocolConfig:
    resolved = path.expanduser().resolve()
    raw = json.loads(resolved.read_text(encoding="utf-8"))
    _require_keys(
        raw,
        (
            "axis",
            "projection",
            "distribution_metrics",
            "audit_checkpoint_steps",
            "frequency_bands",
            "windows",
            "conditions",
            "pulse",
        ),
        "root",
    )
    axis_raw = raw["axis"]
    _require_keys(
        axis_raw,
        (
            "construction",
            "fixed_sigma",
            "target_relative_state_rms",
            "sigma_min",
            "sigma_max",
            "bisection_steps",
            "tau_floor",
        ),
        "axis",
    )
    axis = AxisConfig(
        construction=str(axis_raw["construction"]),
        fixed_sigma=float(axis_raw["fixed_sigma"]),
        target_relative_state_rms=float(axis_raw["target_relative_state_rms"]),
        sigma_min=float(axis_raw["sigma_min"]),
        sigma_max=float(axis_raw["sigma_max"]),
        bisection_steps=int(axis_raw["bisection_steps"]),
        tau_floor=float(axis_raw["tau_floor"]),
    )
    if axis.construction not in ("fixed_sigma", "adaptive_state_rms"):
        raise ValueError("axis.construction must be fixed_sigma or adaptive_state_rms")
    if not 0.0 < axis.sigma_min <= axis.sigma_max:
        raise ValueError("axis sigma bounds are invalid")
    if axis.fixed_sigma <= 0.0:
        raise ValueError("axis.fixed_sigma must be positive")
    if axis.target_relative_state_rms <= 0.0:
        raise ValueError("axis target perturbation must be positive")
    if axis.tau_floor <= 0.0:
        raise ValueError("axis.tau_floor must be positive")

    projection_raw = raw["projection"]
    projection = ProjectionConfig(
        channel_dim=int(projection_raw["channel_dim"]),
        spatial_dim=int(projection_raw["spatial_dim"]),
        pca_dim=int(projection_raw["pca_dim"]),
        seed=int(projection_raw["seed"]),
    )
    checkpoint_steps = tuple(
        sorted(set(int(item) for item in raw["audit_checkpoint_steps"]))
    )
    bands = raw["frequency_bands"]
    band_edges = tuple(float(item) for item in bands["edges"])
    band_names = tuple(str(item) for item in bands["names"])
    if len(band_edges) != len(band_names) + 1:
        raise ValueError("frequency band edges/names mismatch")
    if any(right <= left for left, right in zip(band_edges, band_edges[1:])):
        raise ValueError("frequency band edges must be strictly increasing")

    windows: dict[str, tuple[float, float]] = {}
    for name, values in raw["windows"].items():
        if len(values) != 2:
            raise ValueError(f"window {name} must contain [t_min,t_max]")
        t_min, t_max = float(values[0]), float(values[1])
        if not 0.0 <= t_min <= t_max <= 1.0:
            raise ValueError(f"invalid time window {name}: {values}")
        windows[str(name)] = (t_min, t_max)

    conditions: list[ConditionSpec] = []
    names: set[str] = set()
    for item in raw["conditions"]:
        name = str(item["name"])
        if name in names:
            raise ValueError(f"duplicate condition name: {name}")
        names.add(name)
        window_name = str(item.get("frequency_window", "all"))
        if window_name not in windows:
            raise ValueError(f"condition {name} references unknown window {window_name}")
        t_min, t_max = windows[window_name]
        conditions.append(
            ConditionSpec(
                name=name,
                model_gamma=float(item.get("model_gamma", 0.0)),
                frequency_eta=float(item.get("frequency_eta", 0.0)),
                frequency_window=window_name,
                frequency_t_min=t_min,
                frequency_t_max=t_max,
                interaction=bool(item.get("interaction", False)),
                audit=bool(item.get("audit", True)),
                sample=bool(item.get("sample", False)),
            )
        )
    if "no_ig" not in names:
        raise ValueError("protocol must define no_ig")
    return ProtocolConfig(
        raw=raw,
        axis=axis,
        projection=projection,
        checkpoint_steps=checkpoint_steps,
        band_edges=band_edges,
        band_names=band_names,
        windows=windows,
        conditions=tuple(conditions),
        metric_config=dict(raw["distribution_metrics"]),
        pulse_config=dict(raw["pulse"]),
    )


def prepare_output(path: Path, *, resume: bool, overwrite: bool) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() and overwrite:
        shutil.rmtree(resolved)
    if resolved.exists() and any(resolved.iterdir()) and not resume:
        raise FileExistsError(
            f"Refusing non-empty output root {resolved}; use a new path, "
            "--resume, or --overwrite."
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


# ---------------------------------------------------------------------------
# Repository imports and distributed helpers.
# ---------------------------------------------------------------------------


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
        official_baseline_endpoint,
        official_shifted_solver_grid,
    )
    from experiments.run_raev2_spectral_ig_mechanism_suite_v3 import (
        BandDefinition,
        official_euler_x_prediction_step,
        official_prediction_components,
        official_sampler_model_kwargs,
        rfft_band_masks,
        tensor_band_power,
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


def load_stage2(
    args: argparse.Namespace,
    config: Any,
    ctx: DistributedContext,
    symbols: Mapping[str, Any],
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


def model_components(
    *,
    model: torch.nn.Module,
    state: torch.Tensor,
    labels: torch.Tensor,
    time: float,
    config: Any,
    precision: str,
    symbols: Mapping[str, Any],
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
        raise RuntimeError("checkpoint has no internal-guidance base head")
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


def validate_no_ig_sampler(
    *,
    args: argparse.Namespace,
    config: Any,
    ctx: DistributedContext,
    model: torch.nn.Module,
    grid: torch.Tensor,
    symbols: Mapping[str, Any],
    output_root: Path,
) -> None:
    """Require exact agreement with the released scale-1 sampler."""

    path = output_root / "00_official_baseline_check.json"
    if path.is_file() and args.resume:
        barrier()
        return
    latent_size = tuple(int(value) for value in config.misc.latent_size)
    ids = np.asarray([ctx.rank], dtype=np.int64)
    noise = symbols["deterministic_noise"](
        ids, latent_size, seed=args.rollout_noise_seed
    ).to(ctx.device)
    labels = torch.tensor(
        [ctx.rank % int(config.misc.num_classes)],
        device=ctx.device,
        dtype=torch.long,
    )
    state = noise.clone()
    t_steps = grid.to(ctx.device, torch.float32)
    with torch.inference_mode():
        for step in range(len(grid) - 1):
            time = float(t_steps[step].item())
            baseline, _gap, _base = model_components(
                model=model,
                state=state,
                labels=labels,
                time=time,
                config=config,
                precision=args.precision,
                symbols=symbols,
            )
            state = symbols["official_euler_x_prediction_step"](
                state,
                baseline,
                t_steps=t_steps,
                step=step,
                t_eps=float(config.transport.t_eps),
            )
        shift = math.sqrt(
            (config.misc.time_dist_shift_dim or math.prod(latent_size))
            / config.misc.time_dist_shift_base
        )
        official = symbols["official_baseline_endpoint"](
            model=model,
            noise=noise,
            labels=labels,
            config=config,
            shift=shift,
            precision=args.precision,
        )
    delta = state.double() - official.double()
    official_rms = float(rowwise_rms(official).mean().cpu())
    row = {
        "rank": ctx.rank,
        "rms": float(rowwise_rms(delta).mean().cpu()),
        "relative_rms": float(
            rowwise_rms(delta).mean().cpu() / max(official_rms, 1e-30)
        ),
        "max_abs": float(delta.abs().max().cpu()),
        "official_state_rms": official_rms,
    }
    gathered = gather_object(row, ctx)
    if ctx.rank == 0:
        result = {
            "worst_rms": max(item["rms"] for item in gathered),
            "worst_relative_rms": max(
                item["relative_rms"] for item in gathered
            ),
            "worst_max_abs": max(item["max_abs"] for item in gathered),
            "per_rank": gathered,
        }
        json_dump(path, result)
        if result["worst_rms"] != 0.0 or result["worst_max_abs"] != 0.0:
            raise RuntimeError(
                "explicit scale-1 rollout is not bitwise identical to the "
                f"released sampler: {result}"
            )
    barrier()


# ---------------------------------------------------------------------------
# Frequency-axis construction.
# ---------------------------------------------------------------------------


def rowwise_rms(value: torch.Tensor) -> torch.Tensor:
    return value.double().flatten(1).square().mean(dim=1).sqrt()


def rowwise_cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_flat = left.double().flatten(1)
    right_flat = right.double().flatten(1)
    numerator = (left_flat * right_flat).sum(dim=1)
    denominator = (
        left_flat.norm(dim=1) * right_flat.norm(dim=1)
    ).clamp_min(EPS)
    return numerator / denominator


def gaussian_blur_latent_batch(
    value: torch.Tensor, sigma_pixels: torch.Tensor | float
) -> torch.Tensor:
    if value.ndim != 4:
        raise ValueError("value must be BCHW")
    if isinstance(sigma_pixels, torch.Tensor):
        sigma = sigma_pixels.to(value.device, torch.float32).reshape(-1)
    else:
        sigma = torch.full(
            (len(value),), float(sigma_pixels), device=value.device, dtype=torch.float32
        )
    if len(sigma) != len(value):
        raise ValueError("one sigma per batch element is required")
    fy = torch.fft.fftfreq(value.shape[-2], d=1.0, device=value.device)
    fx = torch.fft.fftfreq(value.shape[-1], d=1.0, device=value.device)
    radius_sq = fy[:, None].square() + fx[None, :].square()
    masks = torch.exp(
        -2.0
        * math.pi**2
        * sigma[:, None, None].square()
        * radius_sq[None]
    )
    transform = torch.fft.fft2(value.float(), dim=(-2, -1), norm="ortho")
    return torch.fft.ifft2(
        transform * masks[:, None], dim=(-2, -1), norm="ortho"
    ).real


def counterfactual_relative_rms(
    state: torch.Tensor,
    clean_prediction: torch.Tensor,
    time: float,
    sigma: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    blurred = gaussian_blur_latent_batch(clean_prediction, sigma)
    counterfactual = state.float() + (1.0 - float(time)) * (
        blurred - clean_prediction.float()
    )
    relative = rowwise_rms(counterfactual - state) / rowwise_rms(state).clamp_min(EPS)
    return relative, counterfactual, blurred


def resolve_axis_sigma(
    *,
    state: torch.Tensor,
    clean_prediction: torch.Tensor,
    time: float,
    axis: AxisConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = len(state)
    if axis.construction == "fixed_sigma":
        sigma = torch.full(
            (batch,), axis.fixed_sigma, device=state.device, dtype=torch.float32
        )
        relative, counterfactual, blurred = counterfactual_relative_rms(
            state, clean_prediction, time, sigma
        )
        return sigma, relative, counterfactual, blurred

    low = torch.full(
        (batch,), axis.sigma_min, device=state.device, dtype=torch.float32
    )
    high = torch.full(
        (batch,), axis.sigma_max, device=state.device, dtype=torch.float32
    )
    target = torch.full_like(low, axis.target_relative_state_rms)
    relative_low, _, _ = counterfactual_relative_rms(
        state, clean_prediction, time, low
    )
    relative_high, _, _ = counterfactual_relative_rms(
        state, clean_prediction, time, high
    )
    below_range = relative_high < target
    above_range = relative_low > target
    for _ in range(axis.bisection_steps):
        middle = 0.5 * (low + high)
        relative_middle, _, _ = counterfactual_relative_rms(
            state, clean_prediction, time, middle
        )
        move_low = relative_middle < target
        low = torch.where(move_low, middle, low)
        high = torch.where(move_low, high, middle)
    sigma = 0.5 * (low + high)
    sigma = torch.where(below_range, torch.full_like(sigma, axis.sigma_max), sigma)
    sigma = torch.where(above_range, torch.full_like(sigma, axis.sigma_min), sigma)
    relative, counterfactual, blurred = counterfactual_relative_rms(
        state, clean_prediction, time, sigma
    )
    return sigma, relative, counterfactual, blurred


def frequency_axis_result(
    *,
    model: torch.nn.Module,
    state: torch.Tensor,
    labels: torch.Tensor,
    time: float,
    baseline: torch.Tensor,
    gap: torch.Tensor,
    axis: AxisConfig,
    config: Any,
    precision: str,
    symbols: Mapping[str, Any],
) -> FrequencyAxisResult:
    sigma, achieved, counterfactual, _blurred = resolve_axis_sigma(
        state=state,
        clean_prediction=baseline,
        time=time,
        axis=axis,
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
    tau = sigma.square().clamp_min(axis.tau_floor)
    shape = (len(state),) + (1,) * (state.ndim - 1)
    raw = baseline.float() - degraded_prediction.float()
    derivative = raw / tau.view(shape)
    interaction = (gap.float() - degraded_gap.float()) / tau.view(shape)
    return FrequencyAxisResult(
        derivative=derivative,
        raw_difference=raw,
        interaction_derivative=interaction,
        sigma=sigma,
        tau=tau,
        achieved_relative_state_rms=achieved,
        degraded_prediction=degraded_prediction,
        degraded_gap=degraded_gap,
    )


def time_in_window(time: float, t_min: float, t_max: float) -> bool:
    return float(t_min) <= float(time) <= float(t_max)


def guided_prediction(
    *,
    condition: ConditionSpec,
    model: torch.nn.Module,
    state: torch.Tensor,
    labels: torch.Tensor,
    time: float,
    baseline: torch.Tensor,
    gap: torch.Tensor,
    base: torch.Tensor,
    axis: AxisConfig,
    config: Any,
    precision: str,
    symbols: Mapping[str, Any],
) -> tuple[torch.Tensor, FrequencyAxisResult | None]:
    model_active = time_in_window(
        time, float(config.guidance.ig.t_min), float(config.guidance.ig.t_max)
    )
    if model_active and condition.model_gamma != 0.0:
        model_guided = base.float() + (1.0 + condition.model_gamma) * gap.float()
    else:
        model_guided = baseline.float()
    frequency_active = (
        condition.frequency_eta != 0.0
        and time_in_window(
            time, condition.frequency_t_min, condition.frequency_t_max
        )
    )
    if not frequency_active:
        return model_guided, None
    axis_result = frequency_axis_result(
        model=model,
        state=state,
        labels=labels,
        time=time,
        baseline=baseline,
        gap=gap,
        axis=axis,
        config=config,
        precision=precision,
        symbols=symbols,
    )
    guided = model_guided + condition.frequency_eta * axis_result.derivative
    if condition.interaction and model_active and condition.model_gamma != 0.0:
        guided = guided + (
            condition.model_gamma
            * condition.frequency_eta
            * axis_result.interaction_derivative
        )
    return guided, axis_result


# ---------------------------------------------------------------------------
# Compact feature representation and spectral statistics.
# ---------------------------------------------------------------------------


def _orthonormal_matrix(rows: int, columns: int, seed: int) -> torch.Tensor:
    if columns <= 0 or columns > rows:
        raise ValueError(f"invalid orthonormal matrix shape {(rows, columns)}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    matrix = torch.randn(rows, columns, generator=generator, dtype=torch.float64)
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    return q.float()


def build_projection_matrices(
    latent_size: Sequence[int], projection: ProjectionConfig, device: torch.device
) -> ProjectionMatrices:
    channels, height, width = (int(item) for item in latent_size)
    channel = _orthonormal_matrix(
        channels, projection.channel_dim, projection.seed
    ).to(device)
    spatial = _orthonormal_matrix(
        height * width, projection.spatial_dim, projection.seed + 1
    ).to(device)
    return ProjectionMatrices(channel=channel, spatial=spatial)


def project_latent(value: torch.Tensor, matrices: ProjectionMatrices) -> torch.Tensor:
    flat = value.float().flatten(2)
    result = torch.einsum(
        "bcs,cp,sq->bpq", flat, matrices.channel, matrices.spatial
    )
    return result.flatten(1)


def build_band_objects(
    protocol: ProtocolConfig, symbols: Mapping[str, Any]
) -> tuple[Any, ...]:
    return tuple(
        symbols["BandDefinition"](
            protocol.band_names[index],
            protocol.band_edges[index],
            protocol.band_edges[index + 1],
        )
        for index in range(len(protocol.band_names))
    )


def stats_names(protocol: ProtocolConfig) -> tuple[str, ...]:
    names = [
        "state_rms",
        "prediction_rms",
        "gap_rms",
        "state_mean",
        "prediction_mean",
        "gap_mean",
        "axis_sigma",
        "axis_tau",
        "axis_achieved_relative_state_rms",
        "axis_derivative_rms",
        "axis_raw_difference_rms",
        "axis_gap_cosine",
        "axis_interaction_rms",
    ]
    for prefix in ("state", "prediction", "gap"):
        names.extend(f"{prefix}_power_{name}" for name in protocol.band_names)
    return tuple(names)


def extract_features_and_stats(
    *,
    state: torch.Tensor,
    prediction: torch.Tensor,
    gap: torch.Tensor,
    matrices: ProjectionMatrices,
    masks: torch.Tensor,
    multiplicity: torch.Tensor,
    protocol: ProtocolConfig,
    axis_result: FrequencyAxisResult | None,
    symbols: Mapping[str, Any],
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    features = {
        "state": project_latent(state, matrices),
        "prediction": project_latent(prediction, matrices),
        "gap": project_latent(gap, matrices),
    }
    columns: list[torch.Tensor] = [
        rowwise_rms(state).float(),
        rowwise_rms(prediction).float(),
        rowwise_rms(gap).float(),
        state.float().flatten(1).mean(dim=1),
        prediction.float().flatten(1).mean(dim=1),
        gap.float().flatten(1).mean(dim=1),
    ]
    if axis_result is None:
        zeros = torch.zeros(len(state), device=state.device, dtype=torch.float32)
        columns.extend([zeros] * 7)
    else:
        columns.extend(
            [
                axis_result.sigma.float(),
                axis_result.tau.float(),
                axis_result.achieved_relative_state_rms.float(),
                rowwise_rms(axis_result.derivative).float(),
                rowwise_rms(axis_result.raw_difference).float(),
                rowwise_cosine(axis_result.derivative, gap).float(),
                rowwise_rms(axis_result.interaction_derivative).float(),
            ]
        )
    for value in (state, prediction, gap):
        power = symbols["tensor_band_power"](
            value, masks, multiplicity
        ).float()
        columns.extend(power[:, index] for index in range(power.shape[1]))
    stats = torch.stack(columns, dim=1)
    expected = len(stats_names(protocol))
    if stats.shape[1] != expected:
        raise RuntimeError(f"stats width {stats.shape[1]} != {expected}")
    return features, stats


# ---------------------------------------------------------------------------
# Reference teacher-forced features.
# ---------------------------------------------------------------------------


def clean_cache_path(output_root: Path, rank: int) -> Path:
    return output_root / "01_reference" / f"clean_rank{rank:02d}.npz"


def reference_feature_path(output_root: Path, rank: int) -> Path:
    return output_root / "01_reference" / f"features_rank{rank:02d}.npz"


def select_and_encode_clean_latents(
    *,
    args: argparse.Namespace,
    config: Any,
    ctx: DistributedContext,
    output_root: Path,
    symbols: Mapping[str, Any],
) -> Path:
    path = clean_cache_path(output_root, ctx.rank)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and args.resume:
        return path
    labels_all = symbols["build_requested_labels"](
        args.reference_samples, int(config.misc.num_classes)
    )
    if ctx.rank == 0:
        selected_rows = symbols["select_matching_imagenet_rows"](
            args.parquet_data_path.expanduser().resolve(),
            labels_all,
            args.data_seed,
        )
    else:
        selected_rows = np.empty(args.reference_samples, dtype=np.int64)
    row_tensor = torch.from_numpy(selected_rows).to(ctx.device)
    dist.broadcast(row_tensor, src=0)
    selected_rows = row_tensor.cpu().numpy().astype(np.int64)
    local_ids = np.arange(
        ctx.rank, args.reference_samples, ctx.world_size, dtype=np.int64
    )
    local_rows = selected_rows[local_ids]
    local_labels = labels_all[local_ids]
    latent_size = tuple(int(value) for value in config.misc.latent_size)
    latents = np.empty((len(local_ids), *latent_size), dtype=np.float16)
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
    with torch.inference_mode():
        for start in range(0, len(local_ids), args.per_rank_batch):
            end = min(start + args.per_rank_batch, len(local_ids))
            images = []
            for row, expected in zip(local_rows[start:end], local_labels[start:end]):
                image, actual, _ = dataset[int(row)]
                if int(actual) != int(expected):
                    raise RuntimeError(
                        f"ImageNet label mismatch at row {row}: {actual} != {expected}"
                    )
                images.append(image)
            image_batch = torch.stack(images).to(ctx.device)
            with symbols["autocast_context"](args.precision):
                encoded = rae.encode(image_batch).float()
            latents[start:end] = encoded.cpu().numpy().astype(np.float16)
            if ctx.rank == 0:
                print(f"[reference:encode] local {end}/{len(local_ids)}", flush=True)
    np.savez_compressed(
        path,
        ids=local_ids,
        rows=local_rows,
        labels=local_labels,
        latents=latents,
    )
    del rae, dataset
    gc.collect()
    torch.cuda.empty_cache()
    barrier()
    return path


def run_reference_features(
    *,
    args: argparse.Namespace,
    protocol: ProtocolConfig,
    config: Any,
    ctx: DistributedContext,
    model: torch.nn.Module,
    grid: torch.Tensor,
    output_root: Path,
    matrices: ProjectionMatrices,
    masks: torch.Tensor,
    multiplicity: torch.Tensor,
    symbols: Mapping[str, Any],
) -> None:
    cache = select_and_encode_clean_latents(
        args=args,
        config=config,
        ctx=ctx,
        output_root=output_root,
        symbols=symbols,
    )
    path = reference_feature_path(output_root, ctx.rank)
    if path.is_file() and args.resume:
        barrier()
        return
    payload = np.load(cache, allow_pickle=False)
    ids = payload["ids"].astype(np.int64)
    labels_np = payload["labels"].astype(np.int64)
    latents = payload["latents"]
    latent_size = tuple(int(value) for value in config.misc.latent_size)
    noises = symbols["deterministic_noise"](
        ids, latent_size, seed=args.reference_noise_seed
    )
    step_count = len(protocol.checkpoint_steps)
    feature_dim = matrices.feature_dim
    stat_count = len(stats_names(protocol))
    state_features = np.empty((len(ids), step_count, feature_dim), dtype=np.float16)
    prediction_features = np.empty_like(state_features)
    gap_features = np.empty_like(state_features)
    stats = np.empty((len(ids), step_count, stat_count), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(ids), args.per_rank_batch):
            end = min(start + args.per_rank_batch, len(ids))
            z = torch.from_numpy(
                np.asarray(latents[start:end], dtype=np.float32)
            ).to(ctx.device)
            noise = noises[start:end].to(ctx.device)
            labels = torch.from_numpy(labels_np[start:end]).to(ctx.device, torch.long)
            for position, step in enumerate(protocol.checkpoint_steps):
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
                features, row_stats = extract_features_and_stats(
                    state=state,
                    prediction=baseline,
                    gap=gap,
                    matrices=matrices,
                    masks=masks,
                    multiplicity=multiplicity,
                    protocol=protocol,
                    axis_result=None,
                    symbols=symbols,
                )
                state_features[start:end, position] = (
                    features["state"].cpu().numpy().astype(np.float16)
                )
                prediction_features[start:end, position] = (
                    features["prediction"].cpu().numpy().astype(np.float16)
                )
                gap_features[start:end, position] = (
                    features["gap"].cpu().numpy().astype(np.float16)
                )
                stats[start:end, position] = row_stats.cpu().numpy()
            if ctx.rank == 0:
                print(f"[reference:features] local {end}/{len(ids)}", flush=True)
    np.savez_compressed(
        path,
        ids=ids,
        labels=labels_np,
        steps=np.asarray(protocol.checkpoint_steps, dtype=np.int64),
        state_features=state_features,
        prediction_features=prediction_features,
        gap_features=gap_features,
        stats=stats,
        stat_names=np.asarray(stats_names(protocol)),
    )
    barrier()


# ---------------------------------------------------------------------------
# Rollout feature collection.
# ---------------------------------------------------------------------------


def rollout_feature_path(output_root: Path, condition: str, rank: int) -> Path:
    return output_root / "02_rollout" / condition / f"features_rank{rank:02d}.npz"


def run_condition_rollout_features(
    *,
    args: argparse.Namespace,
    protocol: ProtocolConfig,
    condition: ConditionSpec,
    config: Any,
    ctx: DistributedContext,
    model: torch.nn.Module,
    grid: torch.Tensor,
    output_root: Path,
    matrices: ProjectionMatrices,
    masks: torch.Tensor,
    multiplicity: torch.Tensor,
    symbols: Mapping[str, Any],
) -> None:
    path = rollout_feature_path(output_root, condition.name, ctx.rank)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and args.resume:
        barrier()
        return
    local_ids = np.arange(
        ctx.rank, args.rollout_samples, ctx.world_size, dtype=np.int64
    )
    labels_all = symbols["build_requested_labels"](
        args.rollout_samples, int(config.misc.num_classes)
    )
    labels_np = labels_all[local_ids]
    latent_size = tuple(int(value) for value in config.misc.latent_size)
    step_count = len(protocol.checkpoint_steps)
    feature_dim = matrices.feature_dim
    stat_count = len(stats_names(protocol))
    state_features = np.empty((len(local_ids), step_count, feature_dim), dtype=np.float16)
    prediction_features = np.empty_like(state_features)
    gap_features = np.empty_like(state_features)
    stats = np.empty((len(local_ids), step_count, stat_count), dtype=np.float32)
    checkpoint_positions = {
        int(step): position for position, step in enumerate(protocol.checkpoint_steps)
    }
    t_steps = grid.to(ctx.device, torch.float32)
    for start in range(0, len(local_ids), args.per_rank_batch):
        end = min(start + args.per_rank_batch, len(local_ids))
        ids = local_ids[start:end]
        labels = torch.from_numpy(labels_np[start:end]).to(ctx.device, torch.long)
        state = symbols["deterministic_noise"](
            ids, latent_size, seed=args.rollout_noise_seed
        ).to(ctx.device)
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
                guided, axis_result = guided_prediction(
                    condition=condition,
                    model=model,
                    state=state,
                    labels=labels,
                    time=time,
                    baseline=baseline,
                    gap=gap,
                    base=base,
                    axis=protocol.axis,
                    config=config,
                    precision=args.precision,
                    symbols=symbols,
                )
                if step in checkpoint_positions:
                    position = checkpoint_positions[step]
                    features, row_stats = extract_features_and_stats(
                        state=state,
                        prediction=baseline,
                        gap=gap,
                        matrices=matrices,
                        masks=masks,
                        multiplicity=multiplicity,
                        protocol=protocol,
                        axis_result=axis_result,
                        symbols=symbols,
                    )
                    state_features[start:end, position] = (
                        features["state"].cpu().numpy().astype(np.float16)
                    )
                    prediction_features[start:end, position] = (
                        features["prediction"].cpu().numpy().astype(np.float16)
                    )
                    gap_features[start:end, position] = (
                        features["gap"].cpu().numpy().astype(np.float16)
                    )
                    stats[start:end, position] = row_stats.cpu().numpy()
                state = symbols["official_euler_x_prediction_step"](
                    state,
                    guided,
                    t_steps=t_steps,
                    step=step,
                    t_eps=float(config.transport.t_eps),
                )
        if ctx.rank == 0:
            print(
                f"[rollout:{condition.name}] local {end}/{len(local_ids)}",
                flush=True,
            )
    np.savez_compressed(
        path,
        ids=local_ids,
        labels=labels_np,
        steps=np.asarray(protocol.checkpoint_steps, dtype=np.int64),
        state_features=state_features,
        prediction_features=prediction_features,
        gap_features=gap_features,
        stats=stats,
        stat_names=np.asarray(stats_names(protocol)),
    )
    barrier()


def run_all_rollout_features(**kwargs: Any) -> None:
    protocol: ProtocolConfig = kwargs["protocol"]
    for condition in protocol.conditions:
        if condition.audit:
            run_condition_rollout_features(condition=condition, **kwargs)


# ---------------------------------------------------------------------------
# Feature merging and distribution metrics.
# ---------------------------------------------------------------------------


def merge_rank_archives(paths: Sequence[Path]) -> dict[str, np.ndarray]:
    payloads = [np.load(path, allow_pickle=False) for path in paths]
    ids = np.concatenate([item["ids"].astype(np.int64) for item in payloads])
    order = np.argsort(ids)
    result: dict[str, np.ndarray] = {"ids": ids[order]}
    for key in payloads[0].files:
        if key == "ids":
            continue
        if key in ("steps", "stat_names"):
            result[key] = payloads[0][key]
        else:
            result[key] = np.concatenate([item[key] for item in payloads], axis=0)[
                order
            ]
    return result


def merge_reference_archive(output_root: Path, world_size: int) -> dict[str, np.ndarray]:
    return merge_rank_archives(
        [reference_feature_path(output_root, rank) for rank in range(world_size)]
    )


def merge_rollout_archive(
    output_root: Path, condition: str, world_size: int
) -> dict[str, np.ndarray]:
    return merge_rank_archives(
        [
            rollout_feature_path(output_root, condition, rank)
            for rank in range(world_size)
        ]
    )


def stable_seed(*items: Any) -> int:
    digest = hashlib.sha256("|".join(str(item) for item in items).encode()).digest()
    return int.from_bytes(digest[:4], "little")




def match_reference_indices_by_labels(
    reference_labels: np.ndarray, candidate_labels: np.ndarray
) -> np.ndarray:
    """Return deterministic reference indices matching candidate label order.

    This prevents differences in class marginals from contaminating distribution
    distances when reference and rollout/pulse sample counts differ.  Reference
    samples are reused cyclically only when a requested class has fewer cached
    examples than the candidate set.
    """

    reference_labels = np.asarray(reference_labels, dtype=np.int64)
    candidate_labels = np.asarray(candidate_labels, dtype=np.int64)
    pools: dict[int, np.ndarray] = {
        int(label): np.flatnonzero(reference_labels == int(label))
        for label in np.unique(candidate_labels)
    }
    missing = [label for label, indices in pools.items() if len(indices) == 0]
    if missing:
        raise RuntimeError(f"reference cache lacks labels: {missing[:10]}")
    counters = {label: 0 for label in pools}
    selected = np.empty(len(candidate_labels), dtype=np.int64)
    for position, label_value in enumerate(candidate_labels.tolist()):
        label = int(label_value)
        indices = pools[label]
        selected[position] = indices[counters[label] % len(indices)]
        counters[label] += 1
    return selected

def fit_global_pca_transforms(
    *,
    reference: Mapping[str, np.ndarray],
    protocol: ProtocolConfig,
    output_root: Path,
) -> dict[str, dict[str, np.ndarray]]:
    from sklearn.decomposition import PCA

    directory = output_root / "03_distribution" / "transforms"
    directory.mkdir(parents=True, exist_ok=True)
    transforms: dict[str, dict[str, np.ndarray]] = {}
    for field in protocol.metric_config.get(
        "compute_fields", ["state", "prediction", "gap"]
    ):
        key = f"{field}_features"
        values = reference[key].astype(np.float32).reshape(-1, reference[key].shape[-1])
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale[scale < 1e-6] = 1.0
        standardized = (values - mean) / scale
        components = min(
            protocol.projection.pca_dim,
            standardized.shape[0] - 1,
            standardized.shape[1],
        )
        pca = PCA(
            n_components=components,
            svd_solver="randomized",
            random_state=protocol.projection.seed + stable_seed(field),
        )
        pca.fit(standardized)
        transform = {
            "mean": mean.astype(np.float32),
            "scale": scale.astype(np.float32),
            "components": pca.components_.astype(np.float32),
            "explained_variance": pca.explained_variance_.astype(np.float32),
            "explained_variance_ratio": pca.explained_variance_ratio_.astype(
                np.float32
            ),
        }
        transforms[field] = transform
        np.savez_compressed(directory / f"{field}_pca.npz", **transform)
    return transforms


def load_pca_transforms(
    protocol: ProtocolConfig, output_root: Path
) -> dict[str, dict[str, np.ndarray]]:
    result = {}
    for field in protocol.metric_config.get(
        "compute_fields", ["state", "prediction", "gap"]
    ):
        payload = np.load(
            output_root / "03_distribution" / "transforms" / f"{field}_pca.npz",
            allow_pickle=False,
        )
        result[field] = {key: payload[key] for key in payload.files}
    return result


def transform_features(
    values: np.ndarray, transform: Mapping[str, np.ndarray]
) -> np.ndarray:
    standardized = (
        values.astype(np.float32) - transform["mean"]
    ) / transform["scale"]
    return standardized @ transform["components"].T


def covariance_matrix(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=0, keepdims=True)
    return centered.T @ centered / max(len(values) - 1, 1)


def gaussian_frechet(left: np.ndarray, right: np.ndarray) -> float:
    mean_left = left.mean(axis=0)
    mean_right = right.mean(axis=0)
    covariance_left = covariance_matrix(left)
    covariance_right = covariance_matrix(right)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_left)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    square_left = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    middle = square_left @ covariance_right @ square_left
    middle_eigenvalues = np.linalg.eigvalsh(0.5 * (middle + middle.T))
    trace_sqrt = np.sqrt(np.clip(middle_eigenvalues, 0.0, None)).sum()
    mean_term = float(np.square(mean_left - mean_right).sum())
    covariance_term = float(
        np.trace(covariance_left)
        + np.trace(covariance_right)
        - 2.0 * trace_sqrt
    )
    return max(mean_term + covariance_term, 0.0)


def sliced_wasserstein(
    left: np.ndarray, right: np.ndarray, directions: int, seed: int
) -> float:
    count = min(len(left), len(right))
    rng = np.random.default_rng(seed)
    left_indices = rng.choice(len(left), count, replace=False)
    right_indices = rng.choice(len(right), count, replace=False)
    dimension = left.shape[1]
    projections = rng.normal(size=(dimension, directions)).astype(np.float32)
    projections /= np.linalg.norm(projections, axis=0, keepdims=True) + EPS
    left_projected = np.sort(left[left_indices] @ projections, axis=0)
    right_projected = np.sort(right[right_indices] @ projections, axis=0)
    return float(np.sqrt(np.square(left_projected - right_projected).mean()))


def pairwise_squared(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.square(left).sum(axis=1, keepdims=True)
    right_norm = np.square(right).sum(axis=1, keepdims=True).T
    return np.maximum(left_norm + right_norm - 2.0 * left @ right.T, 0.0)


def subsample_pair(
    left: np.ndarray, right: np.ndarray, maximum: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if len(left) > maximum:
        left = left[rng.choice(len(left), maximum, replace=False)]
    if len(right) > maximum:
        right = right[rng.choice(len(right), maximum, replace=False)]
    return left, right


def rbf_mmd(
    left: np.ndarray, right: np.ndarray, maximum: int, seed: int
) -> tuple[float, float]:
    left, right = subsample_pair(left, right, maximum, seed)
    combined = np.concatenate([left, right], axis=0)
    distances = pairwise_squared(combined, combined)
    upper = distances[np.triu_indices(len(combined), k=1)]
    positive = upper[upper > 0]
    bandwidth_sq = float(np.median(positive)) if len(positive) else 1.0
    bandwidth_sq = max(bandwidth_sq, 1e-8)
    kxx = np.exp(-pairwise_squared(left, left) / (2.0 * bandwidth_sq))
    kyy = np.exp(-pairwise_squared(right, right) / (2.0 * bandwidth_sq))
    kxy = np.exp(-pairwise_squared(left, right) / (2.0 * bandwidth_sq))
    value = float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean())
    return max(value, 0.0), math.sqrt(bandwidth_sq)


def energy_distance(
    left: np.ndarray, right: np.ndarray, maximum: int, seed: int
) -> float:
    left, right = subsample_pair(left, right, maximum, seed)
    cross = np.sqrt(pairwise_squared(left, right) + 1e-12).mean()
    within_left = np.sqrt(pairwise_squared(left, left) + 1e-12).mean()
    within_right = np.sqrt(pairwise_squared(right, right) + 1e-12).mean()
    return max(float(2.0 * cross - within_left - within_right), 0.0)


def c2st_auc(
    left: np.ndarray,
    right: np.ndarray,
    folds: int,
    maximum_per_class: int,
    seed: int,
) -> tuple[float, float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    left, right = subsample_pair(left, right, maximum_per_class, seed)
    values = np.concatenate([left, right], axis=0)
    labels = np.concatenate(
        [np.zeros(len(left), dtype=np.int64), np.ones(len(right), dtype=np.int64)]
    )
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    aucs = []
    for train, test in splitter.split(values, labels):
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                solver="liblinear",
                max_iter=1000,
                random_state=seed,
            ),
        )
        classifier.fit(values[train], labels[train])
        probability = classifier.predict_proba(values[test])[:, 1]
        aucs.append(float(roc_auc_score(labels[test], probability)))
    auc = float(np.mean(aucs))
    return auc, abs(auc - 0.5) * 2.0


def distribution_metric_row(
    *,
    reference: np.ndarray,
    candidate: np.ndarray,
    metric_config: Mapping[str, Any],
    seed: int,
) -> dict[str, float]:
    mean_difference = reference.mean(axis=0) - candidate.mean(axis=0)
    covariance_reference = covariance_matrix(reference)
    covariance_candidate = covariance_matrix(candidate)
    covariance_scale = np.linalg.norm(covariance_reference, ord="fro") + EPS
    mmd, bandwidth = rbf_mmd(
        reference,
        candidate,
        int(metric_config.get("pairwise_max_samples", 512)),
        seed + 1,
    )
    c2st, separation = c2st_auc(
        reference,
        candidate,
        int(metric_config.get("c2st_folds", 5)),
        int(metric_config.get("c2st_max_samples_per_class", 768)),
        seed + 2,
    )
    return {
        "mean_l2": float(np.linalg.norm(mean_difference) / math.sqrt(reference.shape[1])),
        "covariance_relative_frobenius": float(
            np.linalg.norm(covariance_reference - covariance_candidate, ord="fro")
            / covariance_scale
        ),
        "pca_frechet": gaussian_frechet(reference, candidate),
        "sliced_wasserstein": sliced_wasserstein(
            reference,
            candidate,
            int(metric_config.get("swd_directions", 256)),
            seed + 3,
        ),
        "rbf_mmd": mmd,
        "rbf_bandwidth": bandwidth,
        "energy_distance": energy_distance(
            reference,
            candidate,
            int(metric_config.get("pairwise_max_samples", 512)),
            seed + 4,
        ),
        "c2st_auc": c2st,
        "c2st_separation": separation,
    }


def normalized_path_auc(times: np.ndarray, values: np.ndarray) -> float:
    order = np.argsort(times)
    x = times[order]
    y = values[order]
    width = float(x[-1] - x[0]) if len(x) > 1 else 1.0
    if width <= 0:
        return float(y.mean())
    area = np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return float(area / width)


def analyze_distribution_paths(
    *,
    args: argparse.Namespace,
    protocol: ProtocolConfig,
    ctx: DistributedContext,
    grid: torch.Tensor,
    output_root: Path,
) -> None:
    if ctx.rank != 0:
        barrier()
        return
    directory = output_root / "03_distribution"
    directory.mkdir(parents=True, exist_ok=True)
    reference = merge_reference_archive(output_root, ctx.world_size)
    transforms = fit_global_pca_transforms(
        reference=reference, protocol=protocol, output_root=output_root
    )
    metric_rows: list[dict[str, Any]] = []
    stat_rows: list[dict[str, Any]] = []
    stat_name_list = [str(item) for item in reference["stat_names"].tolist()]
    steps = reference["steps"].astype(np.int64)
    for condition in protocol.conditions:
        if not condition.audit:
            continue
        candidate = merge_rollout_archive(output_root, condition.name, ctx.world_size)
        if not np.array_equal(candidate["steps"], steps):
            raise RuntimeError(f"checkpoint mismatch for {condition.name}")
        matched_reference_indices = match_reference_indices_by_labels(
            reference["labels"], candidate["labels"]
        )
        for field in protocol.metric_config.get(
            "compute_fields", ["state", "prediction", "gap"]
        ):
            transform = transforms[field]
            for position, step in enumerate(steps):
                reference_values = transform_features(
                    reference[f"{field}_features"][matched_reference_indices, position],
                    transform,
                )
                candidate_values = transform_features(
                    candidate[f"{field}_features"][:, position], transform
                )
                seed = args.metric_seed + stable_seed(condition.name, field, int(step))
                row = distribution_metric_row(
                    reference=reference_values,
                    candidate=candidate_values,
                    metric_config=protocol.metric_config,
                    seed=seed,
                )
                metric_rows.append(
                    {
                        "condition": condition.name,
                        "field": field,
                        "step": int(step),
                        "time": float(grid[int(step)]),
                        **row,
                    }
                )
        reference_stats = reference["stats"][matched_reference_indices].astype(
            np.float64
        )
        candidate_stats = candidate["stats"].astype(np.float64)
        for position, step in enumerate(steps):
            for column, name in enumerate(stat_name_list):
                ref_values = reference_stats[:, position, column]
                cand_values = candidate_stats[:, position, column]
                stat_rows.append(
                    {
                        "condition": condition.name,
                        "step": int(step),
                        "time": float(grid[int(step)]),
                        "stat": name,
                        "reference_mean": float(ref_values.mean()),
                        "candidate_mean": float(cand_values.mean()),
                        "mean_difference": float(cand_values.mean() - ref_values.mean()),
                        "reference_std": float(ref_values.std(ddof=1)),
                        "candidate_std": float(cand_values.std(ddof=1)),
                    }
                )
    metrics = pd.DataFrame(metric_rows)
    stats = pd.DataFrame(stat_rows)
    metrics.to_csv(directory / "distribution_metrics_by_time.csv", index=False)
    stats.to_csv(directory / "diagnostic_stats_by_time.csv", index=False)
    distance_columns = [
        "mean_l2",
        "covariance_relative_frobenius",
        "pca_frechet",
        "sliced_wasserstein",
        "rbf_mmd",
        "energy_distance",
        "c2st_separation",
    ]
    auc_rows = []
    for (condition, field), subset in metrics.groupby(["condition", "field"]):
        row: dict[str, Any] = {"condition": condition, "field": field}
        for column in distance_columns:
            row[f"auc_{column}"] = normalized_path_auc(
                subset["time"].to_numpy(float), subset[column].to_numpy(float)
            )
        auc_rows.append(row)
    auc = pd.DataFrame(auc_rows)
    auc.to_csv(directory / "distribution_path_auc.csv", index=False)
    json_dump(
        directory / "distribution_summary.json",
        {
            "rows": metric_rows,
            "path_auc": auc_rows,
            "reference_samples": int(len(reference["ids"])),
            "feature_dim": int(reference["state_features"].shape[-1]),
            "pca_dims": {
                field: int(transform["components"].shape[0])
                for field, transform in transforms.items()
            },
        },
    )
    barrier()


# ---------------------------------------------------------------------------
# Paired window pulse experiment.
# ---------------------------------------------------------------------------


def condition_by_name(protocol: ProtocolConfig, name: str) -> ConditionSpec:
    for condition in protocol.conditions:
        if condition.name == name:
            return condition
    raise KeyError(name)


def steps_for_time_window(
    grid: torch.Tensor, window: tuple[float, float]
) -> tuple[int, int]:
    active = [
        step
        for step in range(len(grid) - 1)
        if window[0] <= float(grid[step]) <= window[1]
    ]
    if not active:
        raise ValueError(f"window {window} contains no solver steps")
    return min(active), max(active) + 1


def pulse_file(
    output_root: Path,
    baseline: str,
    window: str,
    mode: str,
    eta: float,
    rank: int,
) -> Path:
    tag = str(eta).replace("-", "m").replace(".", "p")
    return (
        output_root
        / "04_pulse"
        / baseline
        / window
        / mode
        / f"eta_{tag}_rank{rank:02d}.npz"
    )


def run_baseline_trace_for_pulse(
    *,
    base_condition: ConditionSpec,
    model: torch.nn.Module,
    noise: torch.Tensor,
    labels: torch.Tensor,
    grid: torch.Tensor,
    window_steps: tuple[int, int],
    protocol: ProtocolConfig,
    config: Any,
    precision: str,
    matrices: ProjectionMatrices,
    symbols: Mapping[str, Any],
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    state = noise.float()
    replay_directions: list[torch.Tensor] = []
    window_end_state: torch.Tensor | None = None
    t_steps = grid.to(state.device, torch.float32)
    with torch.inference_mode():
        for step in range(len(grid) - 1):
            time = float(t_steps[step].item())
            baseline, gap, base = model_components(
                model=model,
                state=state,
                labels=labels,
                time=time,
                config=config,
                precision=precision,
                symbols=symbols,
            )
            if window_steps[0] <= step < window_steps[1]:
                axis_result = frequency_axis_result(
                    model=model,
                    state=state,
                    labels=labels,
                    time=time,
                    baseline=baseline,
                    gap=gap,
                    axis=protocol.axis,
                    config=config,
                    precision=precision,
                    symbols=symbols,
                )
                replay_directions.append(axis_result.derivative.detach().cpu())
            guided, _ = guided_prediction(
                condition=base_condition,
                model=model,
                state=state,
                labels=labels,
                time=time,
                baseline=baseline,
                gap=gap,
                base=base,
                axis=protocol.axis,
                config=config,
                precision=precision,
                symbols=symbols,
            )
            state = symbols["official_euler_x_prediction_step"](
                state,
                guided,
                t_steps=t_steps,
                step=step,
                t_eps=float(config.transport.t_eps),
            )
            if step + 1 == window_steps[1]:
                window_end_state = state.detach().clone()
    if window_end_state is None:
        raise RuntimeError("pulse window end was not reached")
    return (
        replay_directions,
        window_end_state,
        state.detach().clone(),
        project_latent(window_end_state, matrices).cpu(),
        project_latent(state, matrices).cpu(),
    )


def run_pulse_branch(
    *,
    base_condition: ConditionSpec,
    sign_eta: float,
    mode: str,
    replay_directions: Sequence[torch.Tensor],
    model: torch.nn.Module,
    noise: torch.Tensor,
    labels: torch.Tensor,
    grid: torch.Tensor,
    window_steps: tuple[int, int],
    protocol: ProtocolConfig,
    config: Any,
    precision: str,
    matrices: ProjectionMatrices,
    symbols: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    state = noise.float()
    t_steps = grid.to(state.device, torch.float32)
    replay_index = 0
    window_end_state: torch.Tensor | None = None
    with torch.inference_mode():
        for step in range(len(grid) - 1):
            time = float(t_steps[step].item())
            baseline, gap, base = model_components(
                model=model,
                state=state,
                labels=labels,
                time=time,
                config=config,
                precision=precision,
                symbols=symbols,
            )
            base_guided, _ = guided_prediction(
                condition=base_condition,
                model=model,
                state=state,
                labels=labels,
                time=time,
                baseline=baseline,
                gap=gap,
                base=base,
                axis=protocol.axis,
                config=config,
                precision=precision,
                symbols=symbols,
            )
            if window_steps[0] <= step < window_steps[1]:
                if mode == "recursive":
                    axis_result = frequency_axis_result(
                        model=model,
                        state=state,
                        labels=labels,
                        time=time,
                        baseline=baseline,
                        gap=gap,
                        axis=protocol.axis,
                        config=config,
                        precision=precision,
                        symbols=symbols,
                    )
                    direction = axis_result.derivative
                elif mode == "replay":
                    direction = replay_directions[replay_index].to(
                        state.device, torch.float32
                    )
                else:
                    raise ValueError(mode)
                base_guided = base_guided + float(sign_eta) * direction
                replay_index += 1
            state = symbols["official_euler_x_prediction_step"](
                state,
                base_guided,
                t_steps=t_steps,
                step=step,
                t_eps=float(config.transport.t_eps),
            )
            if step + 1 == window_steps[1]:
                window_end_state = state.detach().clone()
    if window_end_state is None:
        raise RuntimeError("pulse window end was not reached")
    return (
        project_latent(window_end_state, matrices).cpu(),
        project_latent(state, matrices).cpu(),
    )


def run_pulse_suite(
    *,
    args: argparse.Namespace,
    protocol: ProtocolConfig,
    config: Any,
    ctx: DistributedContext,
    model: torch.nn.Module,
    grid: torch.Tensor,
    output_root: Path,
    matrices: ProjectionMatrices,
    symbols: Mapping[str, Any],
) -> None:
    pulse = protocol.pulse_config
    baselines = [str(item) for item in pulse.get("baselines", ["no_ig"])]
    windows = [str(item) for item in pulse.get("windows", ["early", "middle", "late"])]
    modes = [str(item) for item in pulse.get("modes", ["recursive", "replay"])]
    etas = [float(item) for item in pulse.get("etas", [0.02])]
    local_ids = np.arange(
        ctx.rank, args.pulse_samples, ctx.world_size, dtype=np.int64
    )
    labels_all = symbols["build_requested_labels"](
        args.pulse_samples, int(config.misc.num_classes)
    )
    labels_np = labels_all[local_ids]
    latent_size = tuple(int(value) for value in config.misc.latent_size)
    for baseline_name in baselines:
        base_condition = condition_by_name(protocol, baseline_name)
        for window_name in windows:
            window_steps = steps_for_time_window(grid, protocol.windows[window_name])
            for mode in modes:
                for eta in etas:
                    path = pulse_file(
                        output_root, baseline_name, window_name, mode, eta, ctx.rank
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if path.is_file() and args.resume:
                        barrier()
                        continue
                    zero_window = []
                    zero_final = []
                    plus_window = []
                    plus_final = []
                    minus_window = []
                    minus_final = []
                    for start in range(0, len(local_ids), args.per_rank_batch):
                        end = min(start + args.per_rank_batch, len(local_ids))
                        ids = local_ids[start:end]
                        noise = symbols["deterministic_noise"](
                            ids, latent_size, seed=args.pulse_noise_seed
                        ).to(ctx.device)
                        labels = torch.from_numpy(labels_np[start:end]).to(
                            ctx.device, torch.long
                        )
                        (
                            replay_directions,
                            _zero_window_state,
                            _zero_final_state,
                            zero_window_feature,
                            zero_final_feature,
                        ) = run_baseline_trace_for_pulse(
                            base_condition=base_condition,
                            model=model,
                            noise=noise,
                            labels=labels,
                            grid=grid,
                            window_steps=window_steps,
                            protocol=protocol,
                            config=config,
                            precision=args.precision,
                            matrices=matrices,
                            symbols=symbols,
                        )
                        plus_window_feature, plus_final_feature = run_pulse_branch(
                            base_condition=base_condition,
                            sign_eta=eta,
                            mode=mode,
                            replay_directions=replay_directions,
                            model=model,
                            noise=noise,
                            labels=labels,
                            grid=grid,
                            window_steps=window_steps,
                            protocol=protocol,
                            config=config,
                            precision=args.precision,
                            matrices=matrices,
                            symbols=symbols,
                        )
                        minus_window_feature, minus_final_feature = run_pulse_branch(
                            base_condition=base_condition,
                            sign_eta=-eta,
                            mode=mode,
                            replay_directions=replay_directions,
                            model=model,
                            noise=noise,
                            labels=labels,
                            grid=grid,
                            window_steps=window_steps,
                            protocol=protocol,
                            config=config,
                            precision=args.precision,
                            matrices=matrices,
                            symbols=symbols,
                        )
                        zero_window.append(zero_window_feature.numpy())
                        zero_final.append(zero_final_feature.numpy())
                        plus_window.append(plus_window_feature.numpy())
                        plus_final.append(plus_final_feature.numpy())
                        minus_window.append(minus_window_feature.numpy())
                        minus_final.append(minus_final_feature.numpy())
                        if ctx.rank == 0:
                            print(
                                f"[pulse:{baseline_name}:{window_name}:{mode}:eta={eta}] "
                                f"local {end}/{len(local_ids)}",
                                flush=True,
                            )
                    np.savez_compressed(
                        path,
                        ids=local_ids,
                        labels=labels_np,
                        baseline=np.asarray(baseline_name),
                        window=np.asarray(window_name),
                        mode=np.asarray(mode),
                        eta=np.asarray(eta, dtype=np.float32),
                        window_end_step=np.asarray(window_steps[1], dtype=np.int64),
                        zero_window=np.concatenate(zero_window).astype(np.float16),
                        zero_final=np.concatenate(zero_final).astype(np.float16),
                        plus_window=np.concatenate(plus_window).astype(np.float16),
                        plus_final=np.concatenate(plus_final).astype(np.float16),
                        minus_window=np.concatenate(minus_window).astype(np.float16),
                        minus_final=np.concatenate(minus_final).astype(np.float16),
                    )
                    barrier()
    if ctx.rank == 0:
        analyze_pulse_distributions(
            args=args,
            protocol=protocol,
            ctx=ctx,
            grid=grid,
            output_root=output_root,
        )
    barrier()


def load_clean_latents_all(output_root: Path, world_size: int) -> dict[str, np.ndarray]:
    payloads = [
        np.load(clean_cache_path(output_root, rank), allow_pickle=False)
        for rank in range(world_size)
    ]
    ids = np.concatenate([item["ids"] for item in payloads]).astype(np.int64)
    order = np.argsort(ids)
    return {
        "ids": ids[order],
        "labels": np.concatenate([item["labels"] for item in payloads])[order],
        "latents": np.concatenate([item["latents"] for item in payloads], axis=0)[
            order
        ],
    }


def pulse_reference_features(
    *,
    args: argparse.Namespace,
    protocol: ProtocolConfig,
    grid: torch.Tensor,
    output_root: Path,
    world_size: int,
    step: int,
    sample_count: int,
) -> np.ndarray:
    clean = load_clean_latents_all(output_root, world_size)
    initialize_repo_imports()
    from experiments.run_raev2_distribution_auc import build_requested_labels
    from experiments.run_raev2_ig_impulse_response import deterministic_noise

    desired_labels = build_requested_labels(sample_count, 1000)
    selected = match_reference_indices_by_labels(clean["labels"], desired_labels)
    z = torch.from_numpy(
        np.asarray(clean["latents"][selected], dtype=np.float32)
    )
    if step >= len(grid) - 1:
        state = z
    else:
        latent_size = tuple(z.shape[1:])
        ids = np.arange(sample_count, dtype=np.int64)
        noise = deterministic_noise(
            ids, latent_size, seed=args.reference_noise_seed
        ).float()
        time = float(grid[step])
        state = (1.0 - time) * z + time * noise
    matrices = build_projection_matrices(
        state.shape[1:], protocol.projection, torch.device("cpu")
    )
    return project_latent(state, matrices).numpy()


def merge_pulse_branch(
    paths: Sequence[Path], key: str
) -> tuple[np.ndarray, np.ndarray, int]:
    payloads = [np.load(path, allow_pickle=False) for path in paths]
    ids = np.concatenate([item["ids"] for item in payloads]).astype(np.int64)
    values = np.concatenate([item[key] for item in payloads], axis=0)
    order = np.argsort(ids)
    step = int(payloads[0]["window_end_step"])
    return ids[order], values[order].astype(np.float32), step


def analyze_pulse_distributions(
    *,
    args: argparse.Namespace,
    protocol: ProtocolConfig,
    ctx: DistributedContext,
    grid: torch.Tensor,
    output_root: Path,
) -> None:
    transforms = load_pca_transforms(protocol, output_root)
    state_transform = transforms["state"]
    pulse = protocol.pulse_config
    rows: list[dict[str, Any]] = []
    derivatives: list[dict[str, Any]] = []
    for baseline in pulse.get("baselines", ["no_ig"]):
        for window in pulse.get("windows", ["early", "middle", "late"]):
            for mode in pulse.get("modes", ["recursive", "replay"]):
                for eta_value in pulse.get("etas", [0.02]):
                    eta = float(eta_value)
                    paths = [
                        pulse_file(
                            output_root,
                            str(baseline),
                            str(window),
                            str(mode),
                            eta,
                            rank,
                        )
                        for rank in range(ctx.world_size)
                    ]
                    branch_metrics: dict[tuple[str, str], dict[str, float]] = {}
                    for measurement, suffix in (
                        ("window", "window"),
                        ("final", "final"),
                    ):
                        ids, zero, window_end_step = merge_pulse_branch(
                            paths, f"zero_{suffix}"
                        )
                        _, plus, _ = merge_pulse_branch(paths, f"plus_{suffix}")
                        _, minus, _ = merge_pulse_branch(paths, f"minus_{suffix}")
                        reference_step = (
                            window_end_step if measurement == "window" else len(grid) - 1
                        )
                        reference_raw = pulse_reference_features(
                            args=args,
                            protocol=protocol,
                            grid=grid,
                            output_root=output_root,
                            world_size=ctx.world_size,
                            step=reference_step,
                            sample_count=len(ids),
                        )
                        reference = transform_features(reference_raw, state_transform)
                        transformed = {
                            "minus": transform_features(minus, state_transform),
                            "zero": transform_features(zero, state_transform),
                            "plus": transform_features(plus, state_transform),
                        }
                        for branch, values in transformed.items():
                            seed = args.metric_seed + stable_seed(
                                baseline, window, mode, eta, measurement, branch
                            )
                            metrics = distribution_metric_row(
                                reference=reference,
                                candidate=values,
                                metric_config=protocol.metric_config,
                                seed=seed,
                            )
                            branch_metrics[(measurement, branch)] = metrics
                            rows.append(
                                {
                                    "baseline": baseline,
                                    "window": window,
                                    "mode": mode,
                                    "eta": eta,
                                    "measurement": measurement,
                                    "branch": branch,
                                    **metrics,
                                }
                            )
                        columns = [
                            "mean_l2",
                            "covariance_relative_frobenius",
                            "pca_frechet",
                            "sliced_wasserstein",
                            "rbf_mmd",
                            "energy_distance",
                            "c2st_separation",
                        ]
                        derivative_row: dict[str, Any] = {
                            "baseline": baseline,
                            "window": window,
                            "mode": mode,
                            "eta": eta,
                            "measurement": measurement,
                        }
                        for column in columns:
                            minus_value = branch_metrics[(measurement, "minus")][column]
                            plus_value = branch_metrics[(measurement, "plus")][column]
                            zero_value = branch_metrics[(measurement, "zero")][column]
                            derivative_row[f"central_derivative_{column}"] = (
                                plus_value - minus_value
                            ) / (2.0 * eta)
                            derivative_row[f"plus_delta_{column}"] = plus_value - zero_value
                            derivative_row[f"minus_delta_{column}"] = minus_value - zero_value
                        derivatives.append(derivative_row)
    directory = output_root / "04_pulse"
    pd.DataFrame(rows).to_csv(
        directory / "pulse_distribution_metrics.csv", index=False
    )
    pd.DataFrame(derivatives).to_csv(
        directory / "pulse_distribution_derivatives.csv", index=False
    )
    json_dump(
        directory / "pulse_summary.json",
        {"metrics": rows, "derivatives": derivatives},
    )


# ---------------------------------------------------------------------------
# Large generation, decoding, and image metrics.
# ---------------------------------------------------------------------------


def sample_conditions(
    args: argparse.Namespace, protocol: ProtocolConfig
) -> tuple[ConditionSpec, ...]:
    override = [
        item.strip()
        for item in args.sample_condition_names.split(",")
        if item.strip()
    ]
    if override:
        return tuple(condition_by_name(protocol, name) for name in override)
    return tuple(condition for condition in protocol.conditions if condition.sample)


def endpoint_path(output_root: Path, condition: str, rank: int) -> Path:
    return output_root / "05_samples" / condition / f"endpoint_rank{rank:02d}.npy"


def ids_path(output_root: Path, condition: str, rank: int) -> Path:
    return output_root / "05_samples" / condition / f"ids_rank{rank:02d}.npy"


def images_path(output_root: Path, condition: str, rank: int) -> Path:
    return output_root / "05_samples" / condition / f"images_rank{rank:02d}.npy"


def run_generation(
    *,
    args: argparse.Namespace,
    protocol: ProtocolConfig,
    config: Any,
    ctx: DistributedContext,
    model: torch.nn.Module,
    grid: torch.Tensor,
    output_root: Path,
    symbols: Mapping[str, Any],
) -> None:
    conditions = sample_conditions(args, protocol)
    local_ids = np.arange(ctx.rank, args.sample_count, ctx.world_size, dtype=np.int64)
    labels_all = symbols["build_requested_labels"](
        args.sample_count, int(config.misc.num_classes)
    )
    latent_size = tuple(int(value) for value in config.misc.latent_size)
    t_steps = grid.to(ctx.device, torch.float32)
    for condition in conditions:
        path = endpoint_path(output_root, condition.name, ctx.rank)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and ids_path(
            output_root, condition.name, ctx.rank
        ).is_file() and args.resume:
            barrier()
            continue
        endpoint_map = np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=np.float16,
            shape=(len(local_ids), *latent_size),
        )
        diagnostics = {
            "condition": condition.name,
            "axis_steps": 0,
            "sigma_sum": 0.0,
            "achieved_sum": 0.0,
            "derivative_rms_sum": 0.0,
            "gap_cosine_sum": 0.0,
        }
        for start in range(0, len(local_ids), args.per_rank_batch):
            end = min(start + args.per_rank_batch, len(local_ids))
            ids = local_ids[start:end]
            labels = torch.from_numpy(labels_all[ids]).to(ctx.device, torch.long)
            state = symbols["deterministic_noise"](
                ids, latent_size, seed=args.sample_noise_seed
            ).to(ctx.device)
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
                    guided, axis_result = guided_prediction(
                        condition=condition,
                        model=model,
                        state=state,
                        labels=labels,
                        time=time,
                        baseline=baseline,
                        gap=gap,
                        base=base,
                        axis=protocol.axis,
                        config=config,
                        precision=args.precision,
                        symbols=symbols,
                    )
                    if axis_result is not None:
                        diagnostics["axis_steps"] += len(state)
                        diagnostics["sigma_sum"] += float(axis_result.sigma.sum().cpu())
                        diagnostics["achieved_sum"] += float(
                            axis_result.achieved_relative_state_rms.sum().cpu()
                        )
                        diagnostics["derivative_rms_sum"] += float(
                            rowwise_rms(axis_result.derivative).sum().cpu()
                        )
                        diagnostics["gap_cosine_sum"] += float(
                            rowwise_cosine(axis_result.derivative, gap).sum().cpu()
                        )
                    state = symbols["official_euler_x_prediction_step"](
                        state,
                        guided,
                        t_steps=t_steps,
                        step=step,
                        t_eps=float(config.transport.t_eps),
                    )
            endpoint_map[start:end] = state.cpu().numpy().astype(np.float16)
            if ctx.rank == 0:
                print(
                    f"[sample:{condition.name}] local {end}/{len(local_ids)}",
                    flush=True,
                )
        endpoint_map.flush()
        np.save(ids_path(output_root, condition.name, ctx.rank), local_ids)
        gathered = gather_object(diagnostics, ctx)
        if ctx.rank == 0:
            total_steps = sum(int(item["axis_steps"]) for item in gathered)
            summary = {
                "condition": condition.name,
                "axis_steps": total_steps,
                "mean_sigma": sum(item["sigma_sum"] for item in gathered)
                / max(total_steps, 1),
                "mean_achieved_relative_state_rms": sum(
                    item["achieved_sum"] for item in gathered
                )
                / max(total_steps, 1),
                "mean_axis_derivative_rms": sum(
                    item["derivative_rms_sum"] for item in gathered
                )
                / max(total_steps, 1),
                "mean_axis_gap_cosine": sum(
                    item["gap_cosine_sum"] for item in gathered
                )
                / max(total_steps, 1),
            }
            json_dump(path.parent / "generation_diagnostics.json", summary)
        barrier()


def save_preview(images: np.ndarray, path: Path, count: int) -> None:
    count = min(int(count), len(images))
    if count <= 0:
        return
    columns = min(8, count)
    rows = int(math.ceil(count / columns))
    height, width = images.shape[1:3]
    canvas = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for index, image in enumerate(images[:count]):
        row, column = divmod(index, columns)
        canvas[
            row * height : (row + 1) * height,
            column * width : (column + 1) * width,
        ] = image
    Image.fromarray(canvas, mode="RGB").save(path)


def decode_samples(
    *,
    args: argparse.Namespace,
    protocol: ProtocolConfig,
    config: Any,
    ctx: DistributedContext,
    output_root: Path,
    symbols: Mapping[str, Any],
) -> None:
    conditions = sample_conditions(args, protocol)
    rae = symbols["instantiate_from_config"](config.stage_1).to(ctx.device).eval()
    rae.requires_grad_(False)
    if hasattr(rae, "encoder"):
        del rae.encoder
    image_size = int(config.training.image_size)
    for condition in conditions:
        endpoint = endpoint_path(output_root, condition.name, ctx.rank)
        id_file = ids_path(output_root, condition.name, ctx.rank)
        image_file = images_path(output_root, condition.name, ctx.rank)
        merged_path = output_root / "05_samples" / condition.name / "samples.npz"
        if image_file.is_file() and merged_path.is_file() and args.resume:
            barrier()
            continue
        endpoints = np.load(endpoint, mmap_mode="r", allow_pickle=False)
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
                rank_ids = np.load(ids_path(output_root, condition.name, rank))
                rank_images = np.load(
                    images_path(output_root, condition.name, rank), mmap_mode="r"
                )
                if seen[rank_ids].any():
                    raise RuntimeError("duplicate distributed sample IDs")
                merged[rank_ids] = rank_images
                seen[rank_ids] = True
            if not seen.all():
                raise RuntimeError("incomplete distributed samples")
            np.savez(merged_path, samples=merged)
            save_preview(merged, merged_path.with_name("preview.png"), args.preview_count)
        barrier()
        if not args.keep_endpoints:
            endpoint.unlink(missing_ok=True)
            id_file.unlink(missing_ok=True)
        barrier()
    del rae
    gc.collect()
    torch.cuda.empty_cache()


def evaluate_samples(
    *,
    args: argparse.Namespace,
    protocol: ProtocolConfig,
    output_root: Path,
) -> None:
    initialize_repo_imports()
    from experiments.evaluate_raev2_samples import (
        NumpyRGBDataset,
        torch_fidelity_metrics,
    )

    reference = NumpyRGBDataset(args.reference.expanduser().resolve())
    rows = []
    for condition in sample_conditions(args, protocol):
        sample_path = output_root / "05_samples" / condition.name / "samples.npz"
        samples = NumpyRGBDataset(sample_path)
        metrics = torch_fidelity_metrics(
            samples,
            reference,
            batch_size=int(args.metric_batch_size),
            cache_name="raev2_imagenet256_virtual_reference",
            rng_seed=int(args.metric_seed),
        )
        row = {
            "condition": condition.name,
            "samples": len(samples),
            "sample_path": str(sample_path),
            "sample_sha256": file_sha256(sample_path),
            **metrics,
        }
        rows.append(row)
        json_dump(output_root / "06_metrics_partial.json", rows)
    pd.DataFrame(rows).to_csv(output_root / "06_metrics.csv", index=False)
    json_dump(output_root / "06_metrics.json", rows)


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------


def build_report(
    *, args: argparse.Namespace, protocol: ProtocolConfig, output_root: Path
) -> None:
    report: dict[str, Any] = {
        "protocol": PROTOCOL,
        "script_version": SCRIPT_VERSION,
        "protocol_config": protocol.raw,
    }
    auc_path = output_root / "03_distribution" / "distribution_path_auc.csv"
    if auc_path.is_file():
        auc = pd.read_csv(auc_path)
        report["distribution_path_auc"] = auc.to_dict(orient="records")
        state = auc[auc["field"] == "state"].copy()
        if not state.empty:
            ranking_columns = [
                "auc_pca_frechet",
                "auc_sliced_wasserstein",
                "auc_c2st_separation",
            ]
            rankings = {}
            for column in ranking_columns:
                rankings[column] = (
                    state.sort_values(column)[["condition", column]]
                    .to_dict(orient="records")
                )
            report["state_path_rankings"] = rankings
    pulse_path = output_root / "04_pulse" / "pulse_distribution_derivatives.csv"
    if pulse_path.is_file():
        pulse = pd.read_csv(pulse_path)
        report["pulse_derivatives"] = pulse.to_dict(orient="records")
    metric_path = output_root / "06_metrics.csv"
    if metric_path.is_file():
        metrics = pd.read_csv(metric_path)
        report["image_metrics"] = metrics.to_dict(orient="records")
    json_dump(output_root / "final_report.json", report)

    lines = [
        "RAEv2 Frequency-Axis Extrapolation Suite v1",
        "============================================",
        "",
        "Primary decision criteria:",
        "  1. intermediate rollout-vs-teacher distribution path metrics;",
        "  2. paired +/- pulse distribution derivatives;",
        "  3. independent same-noise IS/FID/KID.",
        "",
        "Teacher-forced clean-prediction MSE is deliberately not used as a",
        "go/no-go criterion because official IG can worsen paired MSE while",
        "improving the closed-loop generative distribution.",
        "",
    ]
    if "state_path_rankings" in report:
        lines.append("Best state-distribution path AUC conditions:")
        for column, rows in report["state_path_rankings"].items():
            lines.append(f"  {column}:")
            for row in rows[:5]:
                lines.append(f"    {row['condition']}: {row[column]:.8g}")
        lines.append("")
    if "image_metrics" in report:
        lines.append("Independent image metrics:")
        metrics = pd.DataFrame(report["image_metrics"])
        columns = [
            column
            for column in (
                "condition",
                "samples",
                "inception_score_mean",
                "frechet_inception_distance",
                "kernel_inception_distance_mean",
            )
            if column in metrics.columns
        ]
        lines.append(metrics[columns].to_string(index=False))
        lines.append("")
    lines.extend(
        [
            "See:",
            "  03_distribution/distribution_metrics_by_time.csv",
            "  03_distribution/distribution_path_auc.csv",
            "  04_pulse/pulse_distribution_derivatives.csv",
            "  06_metrics.csv",
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
    for name in (
        "reference_samples",
        "rollout_samples",
        "pulse_samples",
        "sample_count",
        "per_rank_batch",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    protocol = load_protocol(args.protocol_json)
    symbols = repo_symbols()
    symbols["install_raev2_decoder_config_compat"]()
    os.environ["DINOV3_CKPT_DIR"] = str(args.dino_ckpt_dir.expanduser().resolve())
    os.environ["DINOV3_REPO_DIR"] = str(args.dino_repo_dir.expanduser().resolve())
    ctx = init_distributed()
    for count_name in (
        "reference_samples",
        "rollout_samples",
        "pulse_samples",
        "sample_count",
    ):
        if int(getattr(args, count_name)) < ctx.world_size:
            raise ValueError(f"{count_name} must be at least world_size")
    output_root = prepare_output(
        args.output_root, resume=args.resume, overwrite=args.overwrite
    )
    if ctx.rank == 0:
        json_dump(
            output_root / "manifest.json",
            {
                "protocol": PROTOCOL,
                "script_version": SCRIPT_VERSION,
                "protocol_json": str(args.protocol_json.expanduser().resolve()),
                "protocol_sha256": file_sha256(args.protocol_json.expanduser().resolve()),
                "checkpoint": str(args.checkpoint.expanduser().resolve()),
                "checkpoint_sha256": file_sha256(args.checkpoint.expanduser().resolve()),
                "args": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "protocol_config": protocol.raw,
            },
        )
    barrier()

    config = symbols["load_config"](args.config.expanduser().resolve())
    latent_size = tuple(int(value) for value in config.misc.latent_size)
    if any(step < 0 or step >= int(config.sampler.num_steps) for step in protocol.checkpoint_steps):
        raise ValueError("audit checkpoint step is outside the official solver")
    shift = math.sqrt(
        (config.misc.time_dist_shift_dim or math.prod(latent_size))
        / config.misc.time_dist_shift_base
    )
    grid = symbols["official_shifted_solver_grid"](
        int(config.sampler.num_steps), shift
    ).to(torch.float32)
    model = load_stage2(args, config, ctx, symbols)
    validate_no_ig_sampler(
        args=args,
        config=config,
        ctx=ctx,
        model=model,
        grid=grid,
        symbols=symbols,
        output_root=output_root,
    )
    matrices = build_projection_matrices(latent_size, protocol.projection, ctx.device)
    bands = build_band_objects(protocol, symbols)
    masks, multiplicity = symbols["rfft_band_masks"](
        latent_size[-2], latent_size[-1], bands, device=ctx.device
    )

    if "reference" in args.stages:
        run_reference_features(
            args=args,
            protocol=protocol,
            config=config,
            ctx=ctx,
            model=model,
            grid=grid,
            output_root=output_root,
            matrices=matrices,
            masks=masks,
            multiplicity=multiplicity,
            symbols=symbols,
        )
    if "rollout" in args.stages:
        run_all_rollout_features(
            args=args,
            protocol=protocol,
            config=config,
            ctx=ctx,
            model=model,
            grid=grid,
            output_root=output_root,
            matrices=matrices,
            masks=masks,
            multiplicity=multiplicity,
            symbols=symbols,
        )
    if "metrics" in args.stages:
        barrier()
        analyze_distribution_paths(
            args=args,
            protocol=protocol,
            ctx=ctx,
            grid=grid,
            output_root=output_root,
        )
        barrier()
    if "pulse" in args.stages:
        run_pulse_suite(
            args=args,
            protocol=protocol,
            config=config,
            ctx=ctx,
            model=model,
            grid=grid,
            output_root=output_root,
            matrices=matrices,
            symbols=symbols,
        )
    if "sample" in args.stages:
        run_generation(
            args=args,
            protocol=protocol,
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
        decode_samples(
            args=args,
            protocol=protocol,
            config=config,
            ctx=ctx,
            output_root=output_root,
            symbols=symbols,
        )
    barrier()
    if "evaluate" in args.stages and ctx.rank == 0:
        evaluate_samples(args=args, protocol=protocol, output_root=output_root)
    barrier()
    if "report" in args.stages and ctx.rank == 0:
        build_report(args=args, protocol=protocol, output_root=output_root)
    barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()