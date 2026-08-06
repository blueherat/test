#!/usr/bin/env python3
"""RAEv2 frequency-selective internal-guidance mechanism suite (v3).

This program is deliberately mechanism-first.  It does NOT optimize FID.
Instead it tests and operationalizes the hypothesis

    full - base is a frequency-structured estimator of rollout exposure bias.

The experiment has five connected stages:

1. Teacher-forced spectral prior
   Encode real ImageNet images into clean RAE latents z, construct the official
   forward interpolation x_t=(1-t)z+t*eps at every solver input, and measure:

       * clean / full / base / gap band power;
       * full-gap cross power;
       * the bandwise clean-prediction-optimal extrapolation coefficient;
       * full/base clean-latent MSE by time and frequency.

2. Closed-loop rollout audit and controller identification
   Run scale=1 and scalar-IG trajectories from the same deterministic noises.
   At every step measure the rollout full prediction and gap in identical
   frequency bands.  Fit a small time-window x frequency-band gain table using
   only the spectral transport mismatch

       log Power(F + gamma_b P_b(F-B)) - log Power_TF(F).

   The fit uses baseline-rollout scalar statistics (A,C,Q):

       Power(F + gamma D) = A + 2 gamma C + gamma^2 Q.

   Optional policy iterations recollect A,C,Q under the newly fitted recursive
   controller and refit.  No image metric enters this identification objective.

3. Causal cross-frequency pulse response
   For each time window and input frequency band, apply paired +/- narrow-band
   IG interventions.  Measure endpoint response energy in every output band.
   Both recursive gap feedback and replayed baseline-gap modes are supported,
   separating nonlinear propagation by the full field from state dependence of
   the IG actuator.

4. Same-noise generation
   Generate scale=1, official scalar IG, learned spectral IG, static spectral
   control, and optional single-band ablations with a proven explicit RAEv2
   Euler core.  The no-IG endpoint is checked against the official sampler.

5. Independent image evaluation
   Decode 5k endpoints and reuse the repository-standard torch-fidelity
   evaluator for IS/FID/KID.  These metrics validate the mechanism-derived
   controller; they are never used to fit it.

The script is versioned and protocol-locked.  Do not overwrite it in future
revisions; create ``..._v4.py`` instead.
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
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RAEV2_SRC = ROOT / "external" / "RAEv2" / "src"
for _path in (RAEV2_SRC, ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from experiments.raev2_stage1_compat import install_raev2_decoder_config_compat  # noqa: E402
from experiments.raev2_training_core import (  # noqa: E402
    DeterministicImageNetPacked,
    file_sha256,
    split_internal_guidance_output,
    validate_full_stage2_checkpoint,
)
from experiments.run_raev2_distribution_auc import (  # noqa: E402
    build_requested_labels,
    load_config,
    select_matching_imagenet_rows,
)
from experiments.run_raev2_ig_impulse_response import (  # noqa: E402
    _atomic_json,
    autocast_context,
    deterministic_noise,
    official_baseline_endpoint,
    official_shifted_solver_grid,
)
from utils.model_utils import instantiate_from_config  # noqa: E402


PROTOCOL = "raev2_spectral_ig_mechanism_suite_v3"
SCRIPT_VERSION = "v3"
EPS = 1e-12

DEFAULT_CONFIG = ROOT / "experiments/configs/raev2_strict_lpl_dinov3l_k7.yaml"
DEFAULT_CHECKPOINT = Path(
    "/data/users/zhoushunyu/eqvae/models/RAEv2/stage2/imagenet/"
    "dinov3l-k7/checkpoint.pt"
)
DEFAULT_PACKED_DATA = Path("/data/shared/imagenet-1k/random_access_v1")
DEFAULT_PARQUET_DATA = Path("/data/shared/imagenet-1k")
DEFAULT_REFERENCE = Path("/data/shared/adm_refs/VIRTUAL_imagenet256_labeled.npz")
DEFAULT_DINO_CKPT = Path("/data/users/zhoushunyu/eqvae/models/RAEv2/encoders/dinov3")
DEFAULT_DINO_REPO = Path("/data/users/zhoushunyu/eqvae/models/RAEv2/dinov3_repo")


@dataclasses.dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device


@dataclasses.dataclass(frozen=True)
class BandDefinition:
    name: str
    low: float
    high: float


@dataclasses.dataclass(frozen=True)
class Controller:
    gains: np.ndarray  # [window, band], gamma = scale - 1
    step_to_window: np.ndarray  # [num_steps], -1 outside active IG interval
    bands: tuple[BandDefinition, ...]
    source: str

    def gain_for_step(self, step: int) -> np.ndarray:
        window = int(self.step_to_window[int(step)])
        if window < 0:
            return np.zeros(len(self.bands), dtype=np.float32)
        return np.asarray(self.gains[window], dtype=np.float32)


@dataclasses.dataclass(frozen=True)
class SampleCondition:
    name: str
    kind: str
    description: str


# ---------------------------------------------------------------------------
# CLI and protocol helpers
# ---------------------------------------------------------------------------


def parse_csv_floats(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated floats") from error
    if len(result) < 2 or any(not math.isfinite(item) for item in result):
        raise argparse.ArgumentTypeError("at least two finite band edges are required")
    if any(right <= left for left, right in zip(result, result[1:])):
        raise argparse.ArgumentTypeError("band edges must be strictly increasing")
    return result


def parse_csv_strings(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("at least one value is required")
    return result


def parse_stages(value: str) -> tuple[str, ...]:
    allowed = {
        "encode",
        "calibrate",
        "fit",
        "pulse",
        "sample",
        "decode",
        "evaluate",
        "report",
    }
    values = parse_csv_strings(value)
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown stages: {unknown}")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Mechanism-first frequency-selective RAEv2 internal guidance suite.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--stages",
        type=parse_stages,
        default=parse_stages("encode,calibrate,fit,pulse,sample,decode,evaluate,report"),
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
    model.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    model.add_argument("--per-rank-batch", type=int, default=1)
    model.add_argument("--seed", type=int, default=20260806)
    model.add_argument(
        "--calibration-noise-seed", type=int, default=20260906,
        help="Noise seed for teacher-forced spectral calibration."
    )
    model.add_argument(
        "--policy-noise-seed", type=int, default=20261806,
        help="Noise seed for controller identification rollouts."
    )
    model.add_argument(
        "--pulse-noise-seed", type=int, default=20262806,
        help="Independent noise seed for causal pulse responses."
    )
    model.add_argument(
        "--sample-noise-seed", type=int, default=20263806,
        help="Independent held-out noise seed for final image evaluation."
    )

    spectral = parser.add_argument_group("spectral representation")
    spectral.add_argument(
        "--band-edges",
        type=parse_csv_floats,
        default=parse_csv_floats("0,2.5,5.5,100"),
        help="Radial FFT edges in cycles over the 16x16 latent grid.",
    )
    spectral.add_argument(
        "--band-names",
        type=parse_csv_strings,
        default=parse_csv_strings("low,mid,high"),
    )
    spectral.add_argument("--time-windows", type=int, default=8)
    spectral.add_argument(
        "--calibration-step-stride",
        type=int,
        default=1,
        help="1 measures all 100 solver inputs; larger values are cheaper smoke tests.",
    )

    calibration = parser.add_argument_group("teacher-forced and rollout calibration")
    calibration.add_argument("--calibration-samples", type=int, default=1024)
    calibration.add_argument("--policy-samples", type=int, default=512)
    calibration.add_argument("--policy-iterations", type=int, default=2)
    calibration.add_argument("--fit-steps", type=int, default=2500)
    calibration.add_argument("--fit-lr", type=float, default=0.03)
    calibration.add_argument("--fit-smoothness", type=float, default=0.03)
    calibration.add_argument("--fit-shrinkage", type=float, default=0.01)
    calibration.add_argument("--fit-initial-gamma", type=float, default=0.78)
    calibration.add_argument("--fit-min-gamma", type=float, default=-0.5)
    calibration.add_argument("--fit-max-gamma", type=float, default=1.5)
    calibration.add_argument(
        "--controller-path",
        type=Path,
        help="Optional fitted controller JSON for a separate held-out sampling run.",
    )

    pulse = parser.add_argument_group("causal pulse audit")
    pulse.add_argument("--pulse-samples", type=int, default=64)
    pulse.add_argument("--pulse-gamma", type=float, default=0.10)
    pulse.add_argument(
        "--pulse-modes",
        type=parse_csv_strings,
        default=parse_csv_strings("recursive,replay"),
    )
    pulse.add_argument(
        "--pulse-window-count",
        type=int,
        default=3,
        help="Equal active-step windows used for cross-frequency causal responses.",
    )

    evaluation = parser.add_argument_group("generation and independent metrics")
    evaluation.add_argument("--sample-count", type=int, default=5000)
    evaluation.add_argument(
        "--conditions",
        type=parse_csv_strings,
        default=parse_csv_strings(
            "no_ig,scalar_ig,spectral_learned,spectral_static,low_only,mid_only,high_only"
        ),
    )
    evaluation.add_argument("--condition-group-size", type=int, default=1)
    evaluation.add_argument("--metric-batch-size", type=int, default=64)
    evaluation.add_argument("--metric-seed", type=int, default=20260806)
    evaluation.add_argument("--preview-count", type=int, default=32)
    evaluation.add_argument("--keep-endpoints", action="store_true")
    return parser.parse_args()


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def prepare_output(path: Path, *, resume: bool, overwrite: bool) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() and overwrite:
        shutil.rmtree(resolved)
    if resolved.exists() and any(resolved.iterdir()) and not resume:
        raise FileExistsError(
            f"Refusing non-empty output root: {resolved}. Use a new versioned name, "
            "--resume, or --overwrite."
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


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


def broadcast_object(value: Any, ctx: DistributedContext) -> Any:
    objects = [value if ctx.rank == 0 else None]
    dist.broadcast_object_list(objects, src=0)
    return objects[0]


def gather_object(value: Any, ctx: DistributedContext) -> list[Any]:
    gathered: list[Any] = [None for _ in range(ctx.world_size)]
    dist.all_gather_object(gathered, value)
    return gathered


# ---------------------------------------------------------------------------
# Frequency definitions and exact RFFT bookkeeping
# ---------------------------------------------------------------------------


def build_bands(edges: Sequence[float], names: Sequence[str]) -> tuple[BandDefinition, ...]:
    if len(edges) != len(names) + 1:
        raise ValueError("band edge count must equal band name count plus one")
    result = tuple(
        BandDefinition(str(name), float(edges[index]), float(edges[index + 1]))
        for index, name in enumerate(names)
    )
    if result[0].low > 0:
        raise ValueError("the first band must include the DC frequency")
    return result


def rfft_band_masks(
    height: int,
    width: int,
    bands: Sequence[BandDefinition],
    *,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return disjoint radial masks and Parseval multiplicity weights.

    ``torch.fft.rfft2`` stores only nonnegative horizontal frequencies.  The
    multiplicity is one at kx=0 (and Nyquist for even width), two elsewhere.
    With ``norm='ortho'``, weighted spectral energy equals spatial energy.
    """

    ky = torch.fft.fftfreq(int(height), d=1.0, device=device) * float(height)
    kx = torch.fft.rfftfreq(int(width), d=1.0, device=device) * float(width)
    radius = torch.sqrt(ky[:, None].square() + kx[None, :].square())
    masks = []
    for index, band in enumerate(bands):
        if index == len(bands) - 1:
            mask = (radius >= band.low) & (radius <= band.high)
        else:
            mask = (radius >= band.low) & (radius < band.high)
        masks.append(mask)
    stacked = torch.stack(masks, dim=0)
    coverage = stacked.sum(dim=0)
    if not torch.all(coverage == 1):
        missing = int((coverage == 0).sum().item())
        overlap = int((coverage > 1).sum().item())
        raise ValueError(f"frequency bands do not partition RFFT grid: missing={missing}, overlap={overlap}")
    multiplicity = torch.full((height, width // 2 + 1), 2.0, device=device)
    multiplicity[:, 0] = 1.0
    if width % 2 == 0:
        multiplicity[:, -1] = 1.0
    return stacked, multiplicity


def fft_band_quadratics(
    full: torch.Tensor,
    gap: torch.Tensor,
    masks: torch.Tensor,
    multiplicity: torch.Tensor,
    target: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Per-sample, per-band A/C/Q statistics for F + gamma(F-B)."""

    if full.shape != gap.shape or full.ndim != 4:
        raise ValueError("full and gap must be aligned BCHW tensors")
    transform_full = torch.fft.rfft2(full.float(), dim=(-2, -1), norm="ortho")
    transform_gap = torch.fft.rfft2(gap.float(), dim=(-2, -1), norm="ortho")
    weight = masks.to(full.device, torch.float32) * multiplicity.to(full.device, torch.float32)
    weight = weight[None, :, None, :, :]
    f = transform_full[:, None]
    d = transform_gap[:, None]
    denominator = float(full.shape[1] * full.shape[2] * full.shape[3])
    A = (weight * f.abs().square()).sum(dim=(2, 3, 4)) / denominator
    C = (weight * (f.conj() * d).real).sum(dim=(2, 3, 4)) / denominator
    Q = (weight * d.abs().square()).sum(dim=(2, 3, 4)) / denominator
    result = {"A": A, "C": C, "Q": Q}
    if target is not None:
        if target.shape != full.shape:
            raise ValueError("target and full shapes differ")
        transform_target = torch.fft.rfft2(target.float(), dim=(-2, -1), norm="ortho")
        z = transform_target[:, None]
        T = (weight * z.abs().square()).sum(dim=(2, 3, 4)) / denominator
        E = (weight * ((z - f).conj() * d).real).sum(dim=(2, 3, 4)) / denominator
        mse_full = (weight * (f - z).abs().square()).sum(dim=(2, 3, 4)) / denominator
        mse_base = (weight * (f - d - z).abs().square()).sum(dim=(2, 3, 4)) / denominator
        result.update({"T": T, "E": E, "mse_full": mse_full, "mse_base": mse_base})
    return result


def tensor_band_power(
    value: torch.Tensor,
    masks: torch.Tensor,
    multiplicity: torch.Tensor,
) -> torch.Tensor:
    zeros = torch.zeros_like(value)
    return fft_band_quadratics(value, zeros, masks, multiplicity)["A"]


def official_prediction_components(
    full: torch.Tensor,
    base: torch.Tensor,
    *,
    time: float,
    interval: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the numerically exact official scale-1 prediction and Full-Base gap.

    RAEv2's official internal-guidance wrapper evaluates

        base + scale * (full - base)

    inside the configured IG interval.  At ``scale == 1`` this is algebraically
    equal to ``full`` but not bitwise identical in floating point.  The tiny
    difference can be amplified by a sensitive 100-step closed-loop trajectory.
    Outside the IG interval the wrapper returns ``full`` directly.
    """

    if full.shape != base.shape:
        raise ValueError("full/base tensors must align")
    full_float = full.float()
    base_float = base.float()
    gap = full_float - base_float
    active = float(interval[0]) <= float(time) <= float(interval[1])
    baseline = base_float + gap if active else full_float
    return baseline, gap


def official_sampler_model_kwargs(
    config: Any,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Reproduce the auxiliary tensors inserted by ``Sampler.sample_ode``.

    The released ImageNet checkpoint has ``use_cfg_conds=False``, so these
    tensors do not change its output.  Passing them nevertheless keeps this
    mechanism runner faithful to the official sampler and makes the audit valid
    for configurations whose model consumes CFG-condition tokens.
    """

    result: dict[str, torch.Tensor] = {}
    for name, value in (
        ("omega", config.guidance.cfg.scale),
        ("t_start", config.guidance.cfg.t_min),
        ("t_end", config.guidance.cfg.t_max),
    ):
        if value is not None:
            result[name] = torch.full(
                (int(batch_size),),
                float(value),
                device=device,
                dtype=torch.float32,
            )
    return result


def official_euler_x_prediction_step(
    state: torch.Tensor,
    clean_prediction: torch.Tensor,
    *,
    t_steps: torch.Tensor,
    step: int,
    t_eps: float,
) -> torch.Tensor:
    """Match the released RAEv2 sampler's x-prediction Euler arithmetic.

    This deliberately preserves the official operation order

        drift = (state - prediction) / t_safe
        state = state - h * drift

    with ``h`` and ``t_safe`` as device float32 tensors.  The older shared
    helper evaluated ``h * (state - prediction) / t`` using Python scalars.
    Those expressions are algebraically identical but not bitwise identical;
    rare sensitive 100-step trajectories can amplify the rounding difference.
    """

    if state.shape != clean_prediction.shape:
        raise ValueError("state and clean prediction must align")
    if not 0 <= int(step) < len(t_steps) - 1:
        raise IndexError(step)
    if t_steps.device != state.device:
        raise ValueError("t_steps and state must be on the same device")
    h = t_steps[int(step)] - t_steps[int(step) + 1]
    t_batch = torch.full(
        (len(state),),
        t_steps[int(step)].item(),
        device=state.device,
        dtype=torch.float32,
    )
    t_safe = t_batch.view(
        (len(state),) + (1,) * (state.ndim - 1)
    ).clamp_min(float(t_eps))
    drift = (state - clean_prediction) / t_safe
    return state - h * drift


def apply_frequency_gains(
    full: torch.Tensor,
    gap: torch.Tensor,
    gains: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    """Return full + inverse_RFFT(gain(omega) * RFFT(gap))."""

    if full.shape != gap.shape or full.ndim != 4:
        raise ValueError("full/gap must be aligned BCHW")
    if gains.ndim == 1:
        gains = gains[None].expand(len(full), -1)
    if gains.shape != (len(full), len(masks)):
        raise ValueError(
            f"gain shape {tuple(gains.shape)} != {(len(full), len(masks))}"
        )
    frequency_gain = torch.einsum(
        "bk,khw->bhw", gains.to(full.device, torch.float32), masks.to(full.device, torch.float32)
    )
    transform = torch.fft.rfft2(gap.float(), dim=(-2, -1), norm="ortho")
    correction = torch.fft.irfft2(
        transform * frequency_gain[:, None],
        s=full.shape[-2:],
        dim=(-2, -1),
        norm="ortho",
    )
    return full.float() + correction


def predicted_band_power(A: torch.Tensor, C: torch.Tensor, Q: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    return (A + 2.0 * gamma * C + gamma.square() * Q).clamp_min(EPS)


# ---------------------------------------------------------------------------
# Time windows and controller representation
# ---------------------------------------------------------------------------


def build_step_to_window(
    grid: torch.Tensor,
    interval: tuple[float, float],
    window_count: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    active = np.asarray(
        [
            step
            for step in range(len(grid) - 1)
            if interval[0] <= float(grid[step]) <= interval[1]
        ],
        dtype=np.int64,
    )
    if len(active) < window_count or window_count <= 0:
        raise ValueError("invalid time-window count for active IG solver steps")
    chunks = [np.asarray(chunk, dtype=np.int64) for chunk in np.array_split(active, window_count)]
    mapping = np.full(len(grid) - 1, -1, dtype=np.int64)
    rows = []
    for index, chunk in enumerate(chunks):
        mapping[chunk] = index
        rows.append(
            {
                "window": index,
                "start_step": int(chunk[0]),
                "end_step_exclusive": int(chunk[-1]) + 1,
                "active_steps": int(len(chunk)),
                "start_time": float(grid[int(chunk[0])]),
                "end_time": float(grid[int(chunk[-1]) + 1]),
            }
        )
    return mapping, rows


def controller_to_json(controller: Controller, windows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "script_version": SCRIPT_VERSION,
        "source": controller.source,
        "bands": [dataclasses.asdict(item) for item in controller.bands],
        "windows": list(windows),
        "gains": np.asarray(controller.gains, dtype=np.float64).tolist(),
        "actual_scales": (1.0 + np.asarray(controller.gains, dtype=np.float64)).tolist(),
        "step_to_window": np.asarray(controller.step_to_window, dtype=np.int64).tolist(),
    }


def controller_from_json(path: Path) -> Controller:
    payload = json.loads(path.read_text(encoding="utf-8"))
    bands = tuple(BandDefinition(**item) for item in payload["bands"])
    return Controller(
        gains=np.asarray(payload["gains"], dtype=np.float32),
        step_to_window=np.asarray(payload["step_to_window"], dtype=np.int64),
        bands=bands,
        source=str(payload.get("source", path)),
    )


# ---------------------------------------------------------------------------
# Model/data loading and deterministic latent caches
# ---------------------------------------------------------------------------


def load_stage2(args: argparse.Namespace, config: Any, ctx: DistributedContext) -> torch.nn.Module:
    model = instantiate_from_config(config.stage_2).to(ctx.device).eval().requires_grad_(False)
    checkpoint = torch.load(
        args.checkpoint.expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    validate_full_stage2_checkpoint(checkpoint)
    model.load_state_dict(checkpoint[args.state_key], strict=True)
    del checkpoint
    return model


def encode_clean_latents(
    args: argparse.Namespace,
    config: Any,
    ctx: DistributedContext,
    output_root: Path,
    sample_count: int,
) -> Path:
    stage_dir = output_root / "01_clean_latents"
    stage_dir.mkdir(parents=True, exist_ok=True)
    path = stage_dir / f"clean_rank{ctx.rank:02d}.npz"
    if path.is_file() and args.resume:
        return path

    labels = build_requested_labels(sample_count, int(config.misc.num_classes))
    if ctx.rank == 0:
        rows = select_matching_imagenet_rows(
            args.parquet_data_path.expanduser().resolve(), labels, args.seed + 31
        )
    else:
        rows = np.empty(sample_count, dtype=np.int64)
    row_tensor = torch.from_numpy(rows).to(ctx.device)
    dist.broadcast(row_tensor, src=0)
    rows = row_tensor.cpu().numpy().astype(np.int64)
    local_ids = np.arange(ctx.rank, sample_count, ctx.world_size, dtype=np.int64)
    local_rows = rows[local_ids]
    local_labels = labels[local_ids]

    dataset = DeterministicImageNetPacked(
        args.packed_data_path,
        split="train",
        image_size=int(config.training.image_size),
        horizontal_flip=False,
    )
    rae = instantiate_from_config(config.stage_1).to(ctx.device).eval().requires_grad_(False)
    if hasattr(rae, "decoder"):
        del rae.decoder
    latent_size = tuple(int(value) for value in config.misc.latent_size)
    latents = np.empty((len(local_ids), *latent_size), dtype=np.float16)
    with torch.inference_mode():
        for start in range(0, len(local_ids), args.per_rank_batch):
            end = min(start + args.per_rank_batch, len(local_ids))
            images = []
            for row, expected in zip(local_rows[start:end], local_labels[start:end]):
                image, actual, _ = dataset[int(row)]
                if int(actual) != int(expected):
                    raise RuntimeError(f"ImageNet label mismatch at row {row}: {actual} != {expected}")
                images.append(image)
            batch = torch.stack(images).to(ctx.device)
            with autocast_context(args.precision):
                latent = rae.encode(batch).float()
            latents[start:end] = latent.cpu().numpy().astype(np.float16)
            if ctx.rank == 0:
                print(f"[encode] local {end}/{len(local_ids)}", flush=True)
    np.savez_compressed(path, ids=local_ids, rows=local_rows, labels=local_labels, latents=latents)
    del rae, dataset
    gc.collect()
    torch.cuda.empty_cache()
    barrier()
    return path


def load_local_clean_cache(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload = np.load(path, allow_pickle=False)
    return (
        payload["ids"].astype(np.int64),
        payload["labels"].astype(np.int64),
        payload["latents"].astype(np.float16),
    )


# ---------------------------------------------------------------------------
# Teacher-forced spectral calibration
# ---------------------------------------------------------------------------


def calibration_steps(num_steps: int, stride: int) -> np.ndarray:
    if stride <= 0:
        raise ValueError("calibration step stride must be positive")
    result = list(range(0, num_steps, stride))
    if num_steps - 1 not in result:
        result.append(num_steps - 1)
    return np.asarray(sorted(set(result)), dtype=np.int64)


def run_teacher_forced_calibration(
    args: argparse.Namespace,
    config: Any,
    ctx: DistributedContext,
    output_root: Path,
    clean_cache: Path,
    model: torch.nn.Module,
    grid: torch.Tensor,
    masks: torch.Tensor,
    multiplicity: torch.Tensor,
) -> Path:
    stage_dir = output_root / "02_teacher_forced"
    stage_dir.mkdir(parents=True, exist_ok=True)
    shard = stage_dir / f"teacher_forced_rank{ctx.rank:02d}.npz"
    if shard.is_file() and args.resume:
        return shard
    ids, labels, clean_np = load_local_clean_cache(clean_cache)
    steps = calibration_steps(len(grid) - 1, args.calibration_step_stride)
    fields = ("A", "C", "Q", "T", "E", "mse_full", "mse_base", "state_power")
    values = np.empty((len(ids), len(steps), len(masks), len(fields)), dtype=np.float32)
    latent_size = tuple(int(value) for value in config.misc.latent_size)
    for start in range(0, len(ids), args.per_rank_batch):
        end = min(start + args.per_rank_batch, len(ids))
        clean = torch.from_numpy(clean_np[start:end].astype(np.float32)).to(ctx.device)
        noise = deterministic_noise(ids[start:end], latent_size, seed=args.calibration_noise_seed).to(ctx.device)
        batch_labels = torch.from_numpy(labels[start:end]).to(ctx.device, torch.long)
        with torch.inference_mode():
            for position, step in enumerate(steps.tolist()):
                time = float(grid[step])
                state = (1.0 - time) * clean + time * noise
                time_batch = torch.full((len(state),), time, device=ctx.device)
                with autocast_context(args.precision):
                    output = model(
                        state,
                        time_batch,
                        context=batch_labels,
                        attn_mask=None,
                        **official_sampler_model_kwargs(config, len(state), state.device),
                    )
                full, base = split_internal_guidance_output(output)
                if base is None:
                    raise RuntimeError("checkpoint has no internal-guidance base head")
                baseline, gap = official_prediction_components(
                    full,
                    base,
                    time=time,
                    interval=(
                        float(config.guidance.ig.t_min),
                        float(config.guidance.ig.t_max),
                    ),
                )
                stats = fft_band_quadratics(
                    baseline, gap, masks, multiplicity, target=clean
                )
                stats["state_power"] = tensor_band_power(state, masks, multiplicity)
                for field_index, field in enumerate(fields):
                    values[start:end, position, :, field_index] = stats[field].cpu().numpy()
        if ctx.rank == 0:
            print(f"[teacher-forced] local {end}/{len(ids)}", flush=True)
    np.savez_compressed(shard, ids=ids, labels=labels, steps=steps, fields=np.asarray(fields), values=values)
    barrier()
    if ctx.rank == 0:
        analyze_teacher_forced(stage_dir, ctx.world_size, grid)
    barrier()
    return shard


def merge_feature_shards(directory: Path, prefix: str, world_size: int) -> dict[str, np.ndarray]:
    payloads = [np.load(directory / f"{prefix}_rank{rank:02d}.npz", allow_pickle=False) for rank in range(world_size)]
    common = {key: payloads[0][key] for key in payloads[0].files if key not in ("ids", "labels", "values", "endpoints")}
    ids = np.concatenate([payload["ids"] for payload in payloads]).astype(np.int64)
    labels = np.concatenate([payload["labels"] for payload in payloads]).astype(np.int64)
    values = np.concatenate([payload["values"] for payload in payloads], axis=0)
    order = np.argsort(ids)
    result = {**common, "ids": ids[order], "labels": labels[order], "values": values[order]}
    return result


def analyze_teacher_forced(directory: Path, world_size: int, grid: torch.Tensor) -> None:
    merged = merge_feature_shards(directory, "teacher_forced", world_size)
    fields = [str(item) for item in merged["fields"].tolist()]
    index = {name: fields.index(name) for name in fields}
    values = merged["values"].astype(np.float64)
    steps = merged["steps"].astype(np.int64)
    rows = []
    for position, step in enumerate(steps):
        for band in range(values.shape[2]):
            A = values[:, position, band, index["A"]]
            C = values[:, position, band, index["C"]]
            Q = values[:, position, band, index["Q"]]
            E = values[:, position, band, index["E"]]
            gamma_tf_global = float(E.sum() / max(Q.sum(), EPS))
            gamma_tf_sample = E / np.maximum(Q, EPS)
            rows.append(
                {
                    "step": int(step),
                    "time": float(grid[int(step)]),
                    "band": band,
                    "teacher_full_power": float(A.mean()),
                    "teacher_clean_power": float(values[:, position, band, index["T"]].mean()),
                    "teacher_state_power": float(values[:, position, band, index["state_power"]].mean()),
                    "teacher_gap_cross": float(C.mean()),
                    "teacher_gap_power": float(Q.mean()),
                    "gamma_tf_global": gamma_tf_global,
                    "gamma_tf_sample_median": float(np.median(gamma_tf_sample)),
                    "gamma_tf_sample_q10": float(np.quantile(gamma_tf_sample, 0.10)),
                    "gamma_tf_sample_q90": float(np.quantile(gamma_tf_sample, 0.90)),
                    "mse_full": float(values[:, position, band, index["mse_full"]].mean()),
                    "mse_base": float(values[:, position, band, index["mse_base"]].mean()),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(directory / "teacher_forced_by_time_band.csv", index=False)
    np.savez_compressed(directory / "teacher_forced_merged.npz", **merged)
    plot_heatmap(frame, "gamma_tf_global", directory / "teacher_forced_gamma_heatmap.png", "Teacher-forced MSE-optimal gamma")
    plot_heatmap(frame, "teacher_full_power", directory / "teacher_forced_power_heatmap.png", "Teacher-forced Full predicted-clean power", log=True)


# ---------------------------------------------------------------------------
# Closed-loop feature collection and controller fit
# ---------------------------------------------------------------------------


def step_target_power(
    teacher_directory: Path,
    num_steps: int,
) -> np.ndarray:
    frame = pd.read_csv(teacher_directory / "teacher_forced_by_time_band.csv")
    band_count = int(frame["band"].max()) + 1
    result = np.empty((num_steps, band_count), dtype=np.float64)
    measured_steps = np.sort(frame["step"].unique())
    for band in range(band_count):
        subset = frame[frame.band == band].sort_values("step")
        result[:, band] = np.interp(
            np.arange(num_steps), subset["step"], subset["teacher_full_power"]
        )
    return result


def step_target_state_power(
    teacher_directory: Path,
    num_steps: int,
) -> np.ndarray:
    frame = pd.read_csv(teacher_directory / "teacher_forced_by_time_band.csv")
    band_count = int(frame["band"].max()) + 1
    result = np.empty((num_steps, band_count), dtype=np.float64)
    for band in range(band_count):
        subset = frame[frame.band == band].sort_values("step")
        result[:, band] = np.interp(
            np.arange(num_steps), subset["step"], subset["teacher_state_power"]
        )
    return result


def run_explicit_rollout_features(
    *,
    args: argparse.Namespace,
    config: Any,
    ctx: DistributedContext,
    model: torch.nn.Module,
    grid: torch.Tensor,
    masks: torch.Tensor,
    multiplicity: torch.Tensor,
    sample_count: int,
    noise_seed: int,
    controller: Controller | None,
    scalar_gamma: float | None,
    output_path: Path,
    branch_name: str,
) -> Path:
    if controller is not None and scalar_gamma is not None:
        raise ValueError("controller and scalar gamma are mutually exclusive")
    local_ids = np.arange(ctx.rank, sample_count, ctx.world_size, dtype=np.int64)
    labels = build_requested_labels(sample_count, int(config.misc.num_classes))[local_ids]
    latent_size = tuple(int(value) for value in config.misc.latent_size)
    field_names = ("A", "C", "Q", "state_power")
    values = np.empty((len(local_ids), len(grid) - 1, len(masks), len(field_names)), dtype=np.float32)
    endpoints = np.empty((len(local_ids), *latent_size), dtype=np.float16)
    t_steps = grid.to(device=ctx.device, dtype=torch.float32)
    for start in range(0, len(local_ids), args.per_rank_batch):
        end = min(start + args.per_rank_batch, len(local_ids))
        ids = local_ids[start:end]
        state = deterministic_noise(ids, latent_size, seed=int(noise_seed)).to(ctx.device)
        batch_labels = torch.from_numpy(labels[start:end]).to(ctx.device, torch.long)
        with torch.inference_mode():
            for step in range(len(grid) - 1):
                time = float(t_steps[step].item())
                time_batch = torch.full((len(state),), time, device=ctx.device)
                with autocast_context(args.precision):
                    output = model(
                        state,
                        time_batch,
                        context=batch_labels,
                        attn_mask=None,
                        **official_sampler_model_kwargs(config, len(state), state.device),
                    )
                full, base = split_internal_guidance_output(output)
                if base is None:
                    raise RuntimeError("checkpoint has no internal-guidance base head")
                baseline, gap = official_prediction_components(
                    full,
                    base,
                    time=time,
                    interval=(
                        float(config.guidance.ig.t_min),
                        float(config.guidance.ig.t_max),
                    ),
                )
                stats = fft_band_quadratics(baseline, gap, masks, multiplicity)
                values[start:end, step, :, 0] = stats["A"].cpu().numpy()
                values[start:end, step, :, 1] = stats["C"].cpu().numpy()
                values[start:end, step, :, 2] = stats["Q"].cpu().numpy()
                values[start:end, step, :, 3] = tensor_band_power(state, masks, multiplicity).cpu().numpy()
                if controller is not None:
                    gain = torch.from_numpy(controller.gain_for_step(step)).to(ctx.device)
                    guided = apply_frequency_gains(baseline, gap, gain, masks)
                elif scalar_gamma is not None and (
                    float(config.guidance.ig.t_min) <= time <= float(config.guidance.ig.t_max)
                ):
                    # Match the official wrapper's arithmetic exactly:
                    # base + scale * (full - base), where scale = 1 + gamma.
                    guided = base.float() + (1.0 + float(scalar_gamma)) * gap
                else:
                    guided = baseline
                state = official_euler_x_prediction_step(
                    state,
                    guided,
                    t_steps=t_steps,
                    step=step,
                    t_eps=float(config.transport.t_eps),
                )
        endpoints[start:end] = state.cpu().numpy().astype(np.float16)
        if ctx.rank == 0:
            print(f"[{branch_name}] local {end}/{len(local_ids)}", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        ids=local_ids,
        labels=labels,
        fields=np.asarray(field_names),
        values=values,
        endpoints=endpoints,
    )
    return output_path


def validate_explicit_baseline(
    *,
    args: argparse.Namespace,
    config: Any,
    ctx: DistributedContext,
    model: torch.nn.Module,
    grid: torch.Tensor,
) -> dict[str, float]:
    ids = np.asarray([ctx.rank], dtype=np.int64)
    latent_size = tuple(int(value) for value in config.misc.latent_size)
    noise = deterministic_noise(ids, latent_size, seed=args.policy_noise_seed).to(ctx.device)
    label_value = int(ctx.rank % int(config.misc.num_classes))
    labels = torch.tensor([label_value], device=ctx.device, dtype=torch.long)
    state = noise.clone()
    t_steps = grid.to(device=ctx.device, dtype=torch.float32)
    with torch.inference_mode():
        for step in range(len(grid) - 1):
            time = float(t_steps[step].item())
            time_batch = torch.full((1,), time, device=ctx.device)
            with autocast_context(args.precision):
                output = model(
                    state,
                    time_batch,
                    context=labels,
                    attn_mask=None,
                    **official_sampler_model_kwargs(config, len(state), state.device),
                )
            full, base = split_internal_guidance_output(output)
            if base is None:
                raise RuntimeError("checkpoint has no internal-guidance base head")
            baseline, _gap = official_prediction_components(
                full,
                base,
                time=time,
                interval=(
                    float(config.guidance.ig.t_min),
                    float(config.guidance.ig.t_max),
                ),
            )
            state = official_euler_x_prediction_step(
                state,
                baseline,
                t_steps=t_steps,
                step=step,
                t_eps=float(config.transport.t_eps),
            )
        official = official_baseline_endpoint(
            model=model,
            noise=noise,
            labels=labels,
            config=config,
            shift=math.sqrt(
                (config.misc.time_dist_shift_dim or math.prod(latent_size))
                / config.misc.time_dist_shift_base
            ),
            precision=args.precision,
        )
    delta = state.double() - official.double()
    state_rms = float(official.double().square().mean().sqrt().cpu())
    delta_rms = float(delta.square().mean().sqrt().cpu())
    local = {
        "rank": ctx.rank,
        "rms": delta_rms,
        "relative_rms": delta_rms / max(state_rms, 1e-30),
        "max_abs": float(delta.abs().max().cpu()),
        "official_state_rms": state_rms,
    }
    all_rows = gather_object(local, ctx)
    if ctx.rank == 0:
        worst_rms = max(row["rms"] for row in all_rows)
        worst_abs = max(row["max_abs"] for row in all_rows)
        result = {"worst_rms": worst_rms, "worst_max_abs": worst_abs, "per_rank": all_rows}
        worst_relative = max(row["relative_rms"] for row in all_rows)
        result["worst_relative_rms"] = worst_relative
        if worst_relative > 5e-6 or worst_abs > 2e-4:
            raise RuntimeError(
                "official-arithmetic no-IG sampler does not match the released "
                f"sampler: {result}"
            )
    else:
        result = {}
    return broadcast_object(result, ctx)


def fit_frequency_controller(
    *,
    feature_values: np.ndarray,
    target_power: np.ndarray,
    step_to_window: np.ndarray,
    bands: tuple[BandDefinition, ...],
    args: argparse.Namespace,
    source: str,
) -> tuple[Controller, pd.DataFrame]:
    """Fit a time-band table from A/C/Q rollout statistics only."""

    A = torch.from_numpy(feature_values[..., 0].astype(np.float32))
    C = torch.from_numpy(feature_values[..., 1].astype(np.float32))
    Q = torch.from_numpy(feature_values[..., 2].astype(np.float32))
    target = torch.from_numpy(target_power.astype(np.float32))
    active_steps = np.flatnonzero(step_to_window >= 0)
    A = A[:, active_steps]
    C = C[:, active_steps]
    Q = Q[:, active_steps]
    target = target[active_steps]
    windows = torch.from_numpy(step_to_window[active_steps]).long()
    window_count = int(step_to_window.max()) + 1
    raw = torch.nn.Parameter(
        torch.full((window_count, len(bands)), float(args.fit_initial_gamma), dtype=torch.float32)
    )
    optimizer = torch.optim.Adam([raw], lr=float(args.fit_lr))
    history = []
    for iteration in range(int(args.fit_steps)):
        optimizer.zero_grad(set_to_none=True)
        gamma = raw.clamp(float(args.fit_min_gamma), float(args.fit_max_gamma))
        step_gamma = gamma[windows][None]
        predicted = predicted_band_power(A, C, Q, step_gamma)
        # The teacher-forced prior is a population statistic, so fit the
        # population mean power at each step/band rather than forcing every
        # individual sample to equal the population mean.
        predicted_mean = predicted.mean(dim=0)
        data_loss = (
            torch.log(predicted_mean) - torch.log(target.clamp_min(EPS))
        ).square().mean()
        smooth = (
            (gamma[1:] - gamma[:-1]).square().mean()
            if len(gamma) > 1
            else torch.zeros((), dtype=gamma.dtype)
        )
        shrink = (gamma - float(args.fit_initial_gamma)).square().mean()
        loss = data_loss + float(args.fit_smoothness) * smooth + float(args.fit_shrinkage) * shrink
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            raw.clamp_(float(args.fit_min_gamma), float(args.fit_max_gamma))
        if iteration % 25 == 0 or iteration + 1 == int(args.fit_steps):
            history.append(
                {
                    "iteration": iteration,
                    "loss": float(loss.detach()),
                    "spectral_data_loss": float(data_loss.detach()),
                    "time_smoothness": float(smooth.detach()),
                    "shrinkage": float(shrink.detach()),
                }
            )
    gains = raw.detach().clamp(float(args.fit_min_gamma), float(args.fit_max_gamma)).numpy()
    controller = Controller(
        gains=gains,
        step_to_window=step_to_window.copy(),
        bands=bands,
        source=source,
    )
    return controller, pd.DataFrame(history)


def analyze_rollout_features(
    *,
    directory: Path,
    prefix: str,
    world_size: int,
    target_power: np.ndarray,
    target_state_power: np.ndarray,
    grid: torch.Tensor,
) -> dict[str, np.ndarray]:
    merged = merge_feature_shards(directory, prefix, world_size)
    values = merged["values"].astype(np.float64)
    fields = [str(item) for item in merged["fields"].tolist()]
    index = {name: fields.index(name) for name in fields}
    A = values[..., index["A"]]
    C = values[..., index["C"]]
    Q = values[..., index["Q"]]
    state_power = values[..., index["state_power"]]
    mismatch = np.log(np.maximum(A.mean(axis=0), EPS)) - np.log(np.maximum(target_power, EPS))
    state_mismatch = (
        np.log(np.maximum(state_power.mean(axis=0), EPS))
        - np.log(np.maximum(target_state_power, EPS))
    )
    q_effect = 2.0 * C.mean(axis=0)
    alignment = -mismatch * q_effect
    rows = []
    for step in range(A.shape[1]):
        for band in range(A.shape[2]):
            rows.append(
                {
                    "step": step,
                    "time": float(grid[step]),
                    "band": band,
                    "rollout_full_power": float(A[:, step, band].mean()),
                    "target_teacher_power": float(target_power[step, band]),
                    "log_power_mismatch": float(mismatch[step, band]),
                    "rollout_state_power": float(state_power[:, step, band].mean()),
                    "target_teacher_state_power": float(target_state_power[step, band]),
                    "log_state_power_mismatch": float(state_mismatch[step, band]),
                    "gap_power_derivative": float(q_effect[step, band]),
                    "spectral_defect_alignment": float(alignment[step, band]),
                    "gap_power": float(Q[:, step, band].mean()),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(directory / f"{prefix}_spectral_audit.csv", index=False)
    plot_heatmap(frame, "log_power_mismatch", directory / f"{prefix}_mismatch_heatmap.png", f"{prefix}: rollout vs teacher log-power")
    plot_heatmap(frame, "spectral_defect_alignment", directory / f"{prefix}_alignment_heatmap.png", f"{prefix}: signed spectral defect alignment")
    plot_heatmap(frame, "log_state_power_mismatch", directory / f"{prefix}_state_mismatch_heatmap.png", f"{prefix}: state-distribution spectral mismatch")
    np.savez_compressed(directory / f"{prefix}_merged.npz", **merged)
    return merged


# ---------------------------------------------------------------------------
# Cross-frequency causal pulse response
# ---------------------------------------------------------------------------


def equal_active_windows(grid: torch.Tensor, interval: tuple[float, float], count: int) -> list[tuple[int, int]]:
    active = np.asarray(
        [step for step in range(len(grid) - 1) if interval[0] <= float(grid[step]) <= interval[1]],
        dtype=np.int64,
    )
    chunks = [np.asarray(chunk, dtype=np.int64) for chunk in np.array_split(active, count)]
    return [(int(chunk[0]), int(chunk[-1]) + 1) for chunk in chunks]


def baseline_rollout_with_gaps(
    *,
    model: torch.nn.Module,
    noise: torch.Tensor,
    labels: torch.Tensor,
    grid: torch.Tensor,
    config: Any,
    precision: str,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    state = noise.float()
    gaps = []
    t_steps = grid.to(device=state.device, dtype=torch.float32)
    with torch.inference_mode():
        for step in range(len(grid) - 1):
            time = float(t_steps[step].item())
            time_batch = torch.full((len(state),), time, device=state.device)
            with autocast_context(precision):
                output = model(
                    state,
                    time_batch,
                    context=labels,
                    attn_mask=None,
                    **official_sampler_model_kwargs(config, len(state), state.device),
                )
            full, base = split_internal_guidance_output(output)
            if base is None:
                raise RuntimeError("checkpoint has no IG base head")
            baseline, gap = official_prediction_components(
                full,
                base,
                time=time,
                interval=(
                    float(config.guidance.ig.t_min),
                    float(config.guidance.ig.t_max),
                ),
            )
            gaps.append(gap.detach())
            state = official_euler_x_prediction_step(
                state,
                baseline,
                t_steps=t_steps,
                step=step,
                t_eps=float(config.transport.t_eps),
            )
    return state, tuple(gaps)


def pulse_pair_rollout(
    *,
    model: torch.nn.Module,
    noise: torch.Tensor,
    labels: torch.Tensor,
    grid: torch.Tensor,
    config: Any,
    masks: torch.Tensor,
    baseline_gaps: tuple[torch.Tensor, ...],
    window: tuple[int, int],
    input_band: int,
    gamma: float,
    mode: str,
    precision: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mode not in ("recursive", "replay"):
        raise ValueError("pulse mode must be recursive or replay")
    state = noise.unsqueeze(0).expand(2, *noise.shape).reshape(2 * len(noise), *noise.shape[1:]).contiguous()
    contexts = labels.unsqueeze(0).expand(2, len(labels)).reshape(-1).contiguous()
    signs = torch.tensor((gamma, -gamma), device=noise.device, dtype=torch.float32)
    gain_vectors = torch.zeros((2, len(masks)), device=noise.device, dtype=torch.float32)
    gain_vectors[:, input_band] = signs
    t_steps = grid.to(device=noise.device, dtype=torch.float32)
    with torch.inference_mode():
        for step in range(len(grid) - 1):
            time = float(t_steps[step].item())
            time_batch = torch.full((len(state),), time, device=state.device)
            with autocast_context(precision):
                output = model(
                    state,
                    time_batch,
                    context=contexts,
                    attn_mask=None,
                    **official_sampler_model_kwargs(config, len(state), state.device),
                )
            full, base = split_internal_guidance_output(output)
            if base is None:
                raise RuntimeError("checkpoint has no IG base head")
            full = full.float().reshape(2, len(noise), *state.shape[1:])
            base = base.float().reshape_as(full)
            baseline, current_gap = official_prediction_components(
                full,
                base,
                time=time,
                interval=(
                    float(config.guidance.ig.t_min),
                    float(config.guidance.ig.t_max),
                ),
            )
            if mode == "replay":
                gap = baseline_gaps[step].unsqueeze(0).expand_as(current_gap)
            else:
                gap = current_gap
            if window[0] <= step < window[1]:
                guided = apply_frequency_gains(
                    baseline.reshape_as(state), gap.reshape_as(state), gain_vectors.repeat_interleave(len(noise), dim=0), masks
                ).reshape_as(full)
            else:
                guided = baseline
            state = official_euler_x_prediction_step(
                state.reshape_as(full),
                guided,
                t_steps=t_steps,
                step=step,
                t_eps=float(config.transport.t_eps),
            ).reshape(2 * len(noise), *state.shape[1:])
    endpoints = state.reshape(2, len(noise), *state.shape[1:])
    return endpoints[0], endpoints[1]


def run_pulse_audit(
    *,
    args: argparse.Namespace,
    config: Any,
    ctx: DistributedContext,
    output_root: Path,
    model: torch.nn.Module,
    grid: torch.Tensor,
    masks: torch.Tensor,
    multiplicity: torch.Tensor,
    bands: tuple[BandDefinition, ...],
) -> None:
    stage_dir = output_root / "04_pulse_response"
    stage_dir.mkdir(parents=True, exist_ok=True)
    shard = stage_dir / f"pulse_rank{ctx.rank:02d}.npz"
    if shard.is_file() and args.resume:
        barrier()
        if ctx.rank == 0:
            analyze_pulse_audit(stage_dir, ctx.world_size, bands)
        barrier()
        return
    local_ids = np.arange(ctx.rank, args.pulse_samples, ctx.world_size, dtype=np.int64)
    labels_np = build_requested_labels(args.pulse_samples, int(config.misc.num_classes))[local_ids]
    latent_size = tuple(int(value) for value in config.misc.latent_size)
    windows = equal_active_windows(
        grid,
        (float(config.guidance.ig.t_min), float(config.guidance.ig.t_max)),
        int(args.pulse_window_count),
    )
    modes = tuple(args.pulse_modes)
    records = []
    response_values = []
    for start in range(0, len(local_ids), args.per_rank_batch):
        end = min(start + args.per_rank_batch, len(local_ids))
        ids = local_ids[start:end]
        noise = deterministic_noise(ids, latent_size, seed=args.pulse_noise_seed).to(ctx.device)
        labels = torch.from_numpy(labels_np[start:end]).to(ctx.device, torch.long)
        _baseline, baseline_gaps = baseline_rollout_with_gaps(
            model=model, noise=noise, labels=labels, grid=grid, config=config, precision=args.precision
        )
        batch_rows = []
        for window_index, window in enumerate(windows):
            for input_band in range(len(bands)):
                for mode in modes:
                    positive, negative = pulse_pair_rollout(
                        model=model,
                        noise=noise,
                        labels=labels,
                        grid=grid,
                        config=config,
                        masks=masks,
                        baseline_gaps=baseline_gaps,
                        window=window,
                        input_band=input_band,
                        gamma=float(args.pulse_gamma),
                        mode=mode,
                        precision=args.precision,
                    )
                    response = (positive - negative) / (2.0 * float(args.pulse_gamma))
                    band_rms = tensor_band_power(response, masks, multiplicity).sqrt().cpu().numpy()
                    total_rms = response.flatten(1).square().mean(1).sqrt().cpu().numpy()
                    for local_position in range(len(ids)):
                        batch_rows.append(
                            (
                                int(ids[local_position]),
                                window_index,
                                input_band,
                                modes.index(mode),
                                float(total_rms[local_position]),
                                *band_rms[local_position].tolist(),
                            )
                        )
        response_values.extend(batch_rows)
        if ctx.rank == 0:
            print(f"[pulse] local {end}/{len(local_ids)}", flush=True)
    columns = ("id", "window", "input_band", "mode", "total_rms", *[f"out_{b.name}" for b in bands])
    array = np.asarray(response_values, dtype=np.float64)
    np.savez_compressed(shard, columns=np.asarray(columns), values=array, modes=np.asarray(modes), windows=np.asarray(windows))
    barrier()
    if ctx.rank == 0:
        analyze_pulse_audit(stage_dir, ctx.world_size, bands)
    barrier()


def analyze_pulse_audit(directory: Path, world_size: int, bands: Sequence[BandDefinition]) -> None:
    payloads = [np.load(directory / f"pulse_rank{rank:02d}.npz", allow_pickle=False) for rank in range(world_size)]
    columns = [str(item) for item in payloads[0]["columns"].tolist()]
    modes = [str(item) for item in payloads[0]["modes"].tolist()]
    windows = payloads[0]["windows"]
    values = np.concatenate([payload["values"] for payload in payloads], axis=0)
    frame = pd.DataFrame(values, columns=columns)
    frame["id"] = frame["id"].astype(int)
    frame["window"] = frame["window"].astype(int)
    frame["input_band"] = frame["input_band"].astype(int)
    frame["mode"] = frame["mode"].astype(int).map(dict(enumerate(modes)))
    frame.to_csv(directory / "pulse_response_by_sample.csv", index=False)
    rows = []
    for (window, input_band, mode), subset in frame.groupby(["window", "input_band", "mode"]):
        row = {
            "window": int(window),
            "start_step": int(windows[int(window), 0]),
            "end_step_exclusive": int(windows[int(window), 1]),
            "input_band": bands[int(input_band)].name,
            "mode": mode,
            "total_rms_mean": float(subset["total_rms"].mean()),
        }
        for band in bands:
            row[f"output_{band.name}_rms_mean"] = float(subset[f"out_{band.name}"].mean())
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(directory / "pulse_response_matrix.csv", index=False)
    for mode in modes:
        for window in sorted(summary.window.unique()):
            subset = summary[(summary.mode == mode) & (summary.window == window)]
            matrix = subset[[f"output_{band.name}_rms_mean" for band in bands]].to_numpy()
            figure, axis = plt.subplots(figsize=(6.5, 5.4))
            image = axis.imshow(matrix, aspect="auto")
            axis.set_xticks(range(len(bands)), [band.name for band in bands])
            axis.set_yticks(range(len(bands)), [band.name for band in bands])
            axis.set_xlabel("endpoint output band")
            axis.set_ylabel("intervened input band")
            axis.set_title(f"Cross-frequency response: {mode}, window {window}")
            figure.colorbar(image, ax=axis)
            figure.tight_layout()
            figure.savefig(directory / f"pulse_matrix_{mode}_window{window}.png", dpi=180)
            plt.close(figure)


# ---------------------------------------------------------------------------
# Generation, decoding and repository-standard metrics
# ---------------------------------------------------------------------------


def build_conditions(names: Sequence[str], bands: Sequence[BandDefinition]) -> tuple[SampleCondition, ...]:
    allowed = {"no_ig", "scalar_ig", "spectral_learned", "spectral_static"}
    allowed.update(f"{band.name}_only" for band in bands)
    unknown = sorted(set(names) - allowed)
    if unknown:
        raise ValueError(f"unsupported sample conditions: {unknown}")
    return tuple(SampleCondition(name, name, name.replace("_", " ")) for name in names)


def condition_gains(
    condition: SampleCondition,
    step: int,
    controller: Controller,
    official_gamma: float,
) -> np.ndarray:
    band_count = len(controller.bands)
    active = controller.step_to_window[step] >= 0
    if condition.kind == "no_ig" or not active:
        return np.zeros(band_count, dtype=np.float32)
    if condition.kind == "scalar_ig":
        return np.full(band_count, official_gamma, dtype=np.float32)
    if condition.kind == "spectral_learned":
        return controller.gain_for_step(step)
    if condition.kind == "spectral_static":
        return np.asarray(controller.gains.mean(axis=0), dtype=np.float32)
    if condition.kind.endswith("_only"):
        target = condition.kind[: -len("_only")]
        result = np.zeros(band_count, dtype=np.float32)
        index = [band.name for band in controller.bands].index(target)
        result[index] = controller.gain_for_step(step)[index]
        return result
    raise ValueError(condition.kind)


def endpoint_shard_path(output_root: Path, condition: str, rank: int) -> Path:
    return output_root / "05_samples" / condition / f"endpoint_rank{rank:02d}.npy"


def image_shard_path(output_root: Path, condition: str, rank: int) -> Path:
    return output_root / "05_samples" / condition / f"images_rank{rank:02d}.npy"


def run_generation(
    *,
    args: argparse.Namespace,
    config: Any,
    ctx: DistributedContext,
    output_root: Path,
    model: torch.nn.Module,
    grid: torch.Tensor,
    masks: torch.Tensor,
    multiplicity: torch.Tensor,
    controller: Controller,
    conditions: Sequence[SampleCondition],
) -> None:
    latent_size = tuple(int(value) for value in config.misc.latent_size)
    local_ids = np.arange(ctx.rank, args.sample_count, ctx.world_size, dtype=np.int64)
    labels_all = build_requested_labels(args.sample_count, int(config.misc.num_classes))
    official_gamma = float(config.guidance.ig.scale) - 1.0
    t_steps = grid.to(device=ctx.device, dtype=torch.float32)
    for condition in conditions:
        path = endpoint_shard_path(output_root, condition.name, ctx.rank)
        path.parent.mkdir(parents=True, exist_ok=True)
        ids_path = path.with_name(f"ids_rank{ctx.rank:02d}.npy")
        if path.is_file() and ids_path.is_file() and args.resume:
            barrier()
            continue
        endpoint_map = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float16, shape=(len(local_ids), *latent_size)
        )
        endpoint_power_sum = torch.zeros(len(masks), device=ctx.device, dtype=torch.float64)
        endpoint_power_sq_sum = torch.zeros_like(endpoint_power_sum)
        endpoint_count = torch.zeros((), device=ctx.device, dtype=torch.float64)
        for start in range(0, len(local_ids), args.per_rank_batch):
            end = min(start + args.per_rank_batch, len(local_ids))
            ids = local_ids[start:end]
            labels = torch.from_numpy(labels_all[ids]).to(ctx.device, torch.long)
            state = deterministic_noise(ids, latent_size, seed=args.sample_noise_seed).to(ctx.device)
            with torch.inference_mode():
                for step in range(len(grid) - 1):
                    time = float(t_steps[step].item())
                    time_batch = torch.full((len(state),), time, device=ctx.device)
                    with autocast_context(args.precision):
                        output = model(
                            state,
                            time_batch,
                            context=labels,
                            attn_mask=None,
                            **official_sampler_model_kwargs(
                                config, len(state), state.device
                            ),
                        )
                    full, base = split_internal_guidance_output(output)
                    if base is None:
                        raise RuntimeError("checkpoint has no IG base head")
                    baseline, gap = official_prediction_components(
                        full,
                        base,
                        time=time,
                        interval=(
                            float(config.guidance.ig.t_min),
                            float(config.guidance.ig.t_max),
                        ),
                    )
                    if condition.kind == "scalar_ig" and controller.step_to_window[step] >= 0:
                        # Exact official arithmetic: base + scale * (full - base).
                        guided = base.float() + (1.0 + official_gamma) * gap
                    elif condition.kind == "no_ig":
                        guided = baseline
                    else:
                        gain = torch.from_numpy(
                            condition_gains(condition, step, controller, official_gamma)
                        ).to(ctx.device)
                        guided = apply_frequency_gains(baseline, gap, gain, masks)
                    state = official_euler_x_prediction_step(
                        state,
                        guided,
                        t_steps=t_steps,
                        step=step,
                        t_eps=float(config.transport.t_eps),
                    )
            endpoint_power = tensor_band_power(state, masks, multiplicity).double()
            endpoint_power_sum.add_(endpoint_power.sum(dim=0))
            endpoint_power_sq_sum.add_(endpoint_power.square().sum(dim=0))
            endpoint_count.add_(float(len(state)))
            endpoint_map[start:end] = state.cpu().numpy().astype(np.float16)
            if ctx.rank == 0:
                print(f"[sample:{condition.name}] local {end}/{len(local_ids)}", flush=True)
        endpoint_map.flush()
        np.save(ids_path, local_ids, allow_pickle=False)
        dist.all_reduce(endpoint_power_sum)
        dist.all_reduce(endpoint_power_sq_sum)
        dist.all_reduce(endpoint_count)
        if ctx.rank == 0:
            mean = endpoint_power_sum / endpoint_count.clamp_min(1.0)
            variance = (
                (endpoint_power_sq_sum - endpoint_power_sum.square() / endpoint_count.clamp_min(1.0))
                / (endpoint_count - 1.0).clamp_min(1.0)
            ).clamp_min(0.0)
            json_dump(
                path.parent / "endpoint_spectral_stats.json",
                {
                    "condition": condition.name,
                    "samples": int(endpoint_count.item()),
                    "bands": [band.name for band in controller.bands],
                    "mean_power": mean.cpu().tolist(),
                    "std_power": variance.sqrt().cpu().tolist(),
                },
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


def decode_conditions(
    *,
    args: argparse.Namespace,
    config: Any,
    ctx: DistributedContext,
    output_root: Path,
    conditions: Sequence[SampleCondition],
) -> None:
    rae = instantiate_from_config(config.stage_1).to(ctx.device).eval().requires_grad_(False)
    if hasattr(rae, "encoder"):
        del rae.encoder
    image_size = int(config.training.image_size)
    for condition in conditions:
        endpoint_path = endpoint_shard_path(output_root, condition.name, ctx.rank)
        ids_path = endpoint_path.with_name(f"ids_rank{ctx.rank:02d}.npy")
        image_path = image_shard_path(output_root, condition.name, ctx.rank)
        merged_path = output_root / "05_samples" / condition.name / "samples.npz"
        if image_path.is_file() and merged_path.is_file() and args.resume:
            barrier()
            continue
        endpoints = np.load(endpoint_path, mmap_mode="r", allow_pickle=False)
        ids = np.load(ids_path, allow_pickle=False)
        images = np.lib.format.open_memmap(
            image_path,
            mode="w+",
            dtype=np.uint8,
            shape=(len(endpoints), image_size, image_size, 3),
        )
        with torch.inference_mode():
            for start in range(0, len(endpoints), args.per_rank_batch):
                end = min(start + args.per_rank_batch, len(endpoints))
                latent = torch.from_numpy(np.asarray(endpoints[start:end], dtype=np.float32)).to(ctx.device)
                with autocast_context(args.precision):
                    decoded = rae.decode(latent).float().clamp(0, 1)
                images[start:end] = (
                    decoded.mul(255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
                )
        images.flush()
        barrier()
        if ctx.rank == 0:
            merged = np.empty((args.sample_count, image_size, image_size, 3), dtype=np.uint8)
            seen = np.zeros(args.sample_count, dtype=bool)
            for rank in range(ctx.world_size):
                rank_ids = np.load(
                    endpoint_shard_path(output_root, condition.name, rank).with_name(f"ids_rank{rank:02d}.npy"),
                    allow_pickle=False,
                )
                rank_images = np.load(image_shard_path(output_root, condition.name, rank), mmap_mode="r")
                if seen[rank_ids].any():
                    raise RuntimeError("duplicate distributed sample IDs")
                merged[rank_ids] = rank_images
                seen[rank_ids] = True
            if not seen.all():
                raise RuntimeError("incomplete distributed image IDs")
            np.savez(merged_path, merged)
            save_preview(
                merged,
                output_root / "05_samples" / condition.name / "preview.png",
                args.preview_count,
            )
        # Rank 0 must finish reading every rank's IDs before any rank deletes
        # endpoint shards.
        barrier()
        if not args.keep_endpoints:
            endpoint_path.unlink(missing_ok=True)
            ids_path.unlink(missing_ok=True)
        barrier()
    del rae
    gc.collect()
    torch.cuda.empty_cache()


def evaluate_conditions(
    *,
    args: argparse.Namespace,
    output_root: Path,
    conditions: Sequence[SampleCondition],
) -> pd.DataFrame:
    from experiments.evaluate_raev2_samples import NumpyRGBDataset, torch_fidelity_metrics

    reference = NumpyRGBDataset(args.reference.expanduser().resolve())
    rows = []
    for condition in conditions:
        sample_path = output_root / "05_samples" / condition.name / "samples.npz"
        samples = NumpyRGBDataset(sample_path)
        metrics = torch_fidelity_metrics(
            samples,
            reference,
            batch_size=int(args.metric_batch_size),
            cache_name="raev2_imagenet256_virtual_reference",
            rng_seed=int(args.metric_seed),
        )
        rows.append(
            {
                "condition": condition.name,
                "samples": len(samples),
                "sample_path": str(sample_path),
                "sample_sha256": file_sha256(sample_path),
                **metrics,
            }
        )
        json_dump(output_root / "06_metrics_partial.json", rows)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "06_metrics.csv", index=False)
    json_dump(output_root / "06_metrics.json", rows)
    return frame


# ---------------------------------------------------------------------------
# Reporting and plots
# ---------------------------------------------------------------------------


def plot_heatmap(
    frame: pd.DataFrame,
    value: str,
    output: Path,
    title: str,
    *,
    log: bool = False,
) -> None:
    pivot = frame.pivot(index="band", columns="step", values=value).sort_index()
    matrix = pivot.to_numpy(dtype=float)
    if log:
        matrix = np.log10(np.maximum(matrix, EPS))
    figure, axis = plt.subplots(figsize=(13, 4.8))
    image = axis.imshow(matrix, aspect="auto", origin="lower")
    axis.set_xlabel("solver step (sampling: 1 -> 0)")
    axis.set_ylabel("frequency band index")
    axis.set_title(title)
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_controller_plot(controller: Controller, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.4))
    for band_index, band in enumerate(controller.bands):
        axis.plot(np.arange(len(controller.gains)), controller.gains[:, band_index], marker="o", label=band.name)
    axis.axhline(0.78, linestyle="--", linewidth=1, label="official scalar gamma=0.78")
    axis.axhline(0.0, linewidth=1)
    axis.set_xlabel("equal-active-step time window")
    axis.set_ylabel("fitted gamma (scale - 1)")
    axis.set_title("Mechanism-derived frequency-selective IG controller")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def build_report(output_root: Path) -> None:
    report = {
        "protocol": PROTOCOL,
        "controller": json.loads((output_root / "03_fit" / "controller_final.json").read_text(encoding="utf-8")),
    }
    teacher_path = output_root / "02_teacher_forced" / "teacher_forced_by_time_band.csv"
    if teacher_path.is_file():
        teacher = pd.read_csv(teacher_path)
        report["teacher_forced"] = {
            "mean_gamma_tf_by_band": teacher.groupby("band")["gamma_tf_global"].mean().to_dict(),
            "full_mse_better_fraction": float((teacher.mse_full < teacher.mse_base).mean()),
        }
    audit_summaries = {}
    for name in ("policy_iter00", "scalar_ig", "spectral_final"):
        path = output_root / "03_fit" / f"{name}_spectral_audit.csv"
        if path.is_file():
            audit = pd.read_csv(path)
            audit_summaries[name] = {
                "mean_abs_log_power_mismatch": float(audit.log_power_mismatch.abs().mean()),
                "positive_alignment_fraction": float((audit.spectral_defect_alignment > 0).mean()),
                "mean_abs_state_log_power_mismatch": float(audit.log_state_power_mismatch.abs().mean()),
                "by_band_mean_abs_mismatch": audit.assign(
                    abs_mismatch=audit.log_power_mismatch.abs()
                ).groupby("band")["abs_mismatch"].mean().to_dict(),
            }
    if audit_summaries:
        report["rollout_spectral_audits"] = audit_summaries

    endpoint_stats = {}
    sample_root = output_root / "05_samples"
    if sample_root.is_dir():
        for path in sample_root.glob("*/endpoint_spectral_stats.json"):
            endpoint_stats[path.parent.name] = json.loads(path.read_text(encoding="utf-8"))
    if endpoint_stats:
        report["endpoint_spectral_stats"] = endpoint_stats

    metrics_path = output_root / "06_metrics.csv"
    if metrics_path.is_file():
        metrics = pd.read_csv(metrics_path)
        report["metrics"] = metrics.to_dict("records")
    pulse_path = output_root / "04_pulse_response" / "pulse_response_matrix.csv"
    if pulse_path.is_file():
        pulse = pd.read_csv(pulse_path)
        report["pulse_rows"] = pulse.to_dict("records")
    json_dump(output_root / "final_report.json", report)
    lines = [
        "RAEv2 Spectral IG Mechanism Suite v3",
        "====================================",
        "",
        "The controller was fitted only to teacher-forced versus rollout spectral mismatch.",
        "Image metrics are independent validation and did not enter the fit.",
        "",
        "Controller gains (gamma = scale - 1):",
        np.array2string(np.asarray(report["controller"]["gains"]), precision=4),
    ]
    if "metrics" in report:
        lines.extend(["", "Independent image metrics:", pd.DataFrame(report["metrics"]).to_string(index=False)])
    (output_root / "final_report.txt").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def main() -> None:
    install_raev2_decoder_config_compat()
    args = parse_args()
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    positive = (
        args.per_rank_batch,
        args.calibration_samples,
        args.policy_samples,
        args.policy_iterations,
        args.fit_steps,
        args.time_windows,
        args.pulse_samples,
        args.pulse_window_count,
        args.sample_count,
        args.metric_batch_size,
    )
    if any(int(value) <= 0 for value in positive):
        raise ValueError("all sample, batch, iteration and window counts must be positive")
    if not set(args.pulse_modes).issubset({"recursive", "replay"}):
        raise ValueError("pulse modes must be recursive and/or replay")

    os.environ["DINOV3_CKPT_DIR"] = str(args.dino_ckpt_dir.expanduser().resolve())
    os.environ["DINOV3_REPO_DIR"] = str(args.dino_repo_dir.expanduser().resolve())
    ctx = init_distributed()
    for name, count in (
        ("calibration_samples", args.calibration_samples),
        ("policy_samples", args.policy_samples),
        ("pulse_samples", args.pulse_samples),
        ("sample_count", args.sample_count),
    ):
        if int(count) < ctx.world_size:
            raise ValueError(f"{name} must be at least world_size={ctx.world_size}")
    torch.backends.cuda.matmul.allow_tf32 = args.precision != "fp32"
    torch.backends.cudnn.allow_tf32 = args.precision != "fp32"

    if ctx.rank == 0:
        output_root = prepare_output(args.output_root, resume=args.resume, overwrite=args.overwrite)
        output_string = str(output_root)
    else:
        output_string = None
    output_root = Path(broadcast_object(output_string, ctx))
    config = load_config(args.config.expanduser().resolve())
    latent_size = tuple(int(value) for value in config.misc.latent_size)
    shift = math.sqrt(
        (config.misc.time_dist_shift_dim or math.prod(latent_size))
        / config.misc.time_dist_shift_base
    )
    grid = official_shifted_solver_grid(int(config.sampler.num_steps), shift)
    bands = build_bands(args.band_edges, args.band_names)
    masks, multiplicity = rfft_band_masks(latent_size[-2], latent_size[-1], bands, device=ctx.device)
    interval = (float(config.guidance.ig.t_min), float(config.guidance.ig.t_max))
    step_to_window, window_rows = build_step_to_window(grid, interval, args.time_windows)

    if ctx.rank == 0:
        manifest = {
            "protocol": PROTOCOL,
            "script_version": SCRIPT_VERSION,
            "stages": list(args.stages),
            "checkpoint": str(args.checkpoint.expanduser().resolve()),
            "checkpoint_sha256": file_sha256(args.checkpoint.expanduser().resolve()),
            "state_key": args.state_key,
            "precision": args.precision,
            "seed": args.seed,
            "calibration_noise_seed": args.calibration_noise_seed,
            "policy_noise_seed": args.policy_noise_seed,
            "pulse_noise_seed": args.pulse_noise_seed,
            "sample_noise_seed": args.sample_noise_seed,
            "latent_size": list(latent_size),
            "solver_grid": grid.tolist(),
            "ig_interval": interval,
            "bands": [dataclasses.asdict(item) for item in bands],
            "controller_windows": window_rows,
            "calibration_samples": args.calibration_samples,
            "policy_samples": args.policy_samples,
            "policy_iterations": args.policy_iterations,
            "external_controller_path": (
                str(args.controller_path.expanduser().resolve())
                if args.controller_path is not None
                else None
            ),
            "pulse_samples": args.pulse_samples,
            "sample_count": args.sample_count,
            "important_note": "FID/KID are validation only; controller fit uses spectral exposure bias.",
        }
        json_dump(output_root / "manifest.json", manifest)
    barrier()

    clean_cache = output_root / "01_clean_latents" / f"clean_rank{ctx.rank:02d}.npz"
    if "encode" in args.stages:
        clean_cache = encode_clean_latents(
            args, config, ctx, output_root, int(args.calibration_samples)
        )

    needs_stage2 = any(stage in args.stages for stage in ("calibrate", "fit", "pulse", "sample"))
    model = load_stage2(args, config, ctx) if needs_stage2 else None

    if "calibrate" in args.stages:
        if not clean_cache.is_file():
            raise FileNotFoundError("clean latent cache missing; run the encode stage")
        assert model is not None
        run_teacher_forced_calibration(
            args, config, ctx, output_root, clean_cache, model, grid, masks, multiplicity
        )

    controller_path = output_root / "03_fit" / "controller_final.json"
    controller: Controller | None = None
    if "fit" in args.stages:
        assert model is not None
        teacher_dir = output_root / "02_teacher_forced"
        if not (teacher_dir / "teacher_forced_by_time_band.csv").is_file():
            raise FileNotFoundError("teacher-forced calibration missing")
        target_power = step_target_power(teacher_dir, len(grid) - 1)
        target_state_power = step_target_state_power(teacher_dir, len(grid) - 1)
        validation = validate_explicit_baseline(
            args=args, config=config, ctx=ctx, model=model, grid=grid
        )
        if ctx.rank == 0:
            json_dump(output_root / "03_fit" / "official_baseline_check.json", validation)

        current_controller: Controller | None = None
        for iteration in range(int(args.policy_iterations)):
            fit_dir = output_root / "03_fit"
            fit_dir.mkdir(parents=True, exist_ok=True)
            prefix = f"policy_iter{iteration:02d}"
            shard = fit_dir / f"{prefix}_rank{ctx.rank:02d}.npz"
            run_explicit_rollout_features(
                args=args,
                config=config,
                ctx=ctx,
                model=model,
                grid=grid,
                masks=masks,
                multiplicity=multiplicity,
                sample_count=int(args.policy_samples),
                noise_seed=int(args.policy_noise_seed),
                controller=current_controller,
                scalar_gamma=None,
                output_path=shard,
                branch_name=prefix,
            )
            barrier()
            if ctx.rank == 0:
                merged = analyze_rollout_features(
                    directory=fit_dir,
                    prefix=prefix,
                    world_size=ctx.world_size,
                    target_power=target_power,
                    target_state_power=target_state_power,
                    grid=grid,
                )
                fields = [str(item) for item in merged["fields"].tolist()]
                field_index = {name: fields.index(name) for name in fields}
                compact = merged["values"][..., [field_index["A"], field_index["C"], field_index["Q"]]]
                fitted, history = fit_frequency_controller(
                    feature_values=compact,
                    target_power=target_power,
                    step_to_window=step_to_window,
                    bands=bands,
                    args=args,
                    source=f"policy iteration {iteration} on {prefix}",
                )
                iteration_path = fit_dir / f"controller_iter{iteration:02d}.json"
                json_dump(iteration_path, controller_to_json(fitted, window_rows))
                history.to_csv(fit_dir / f"fit_history_iter{iteration:02d}.csv", index=False)
                payload = controller_to_json(fitted, window_rows)
            else:
                payload = None
            payload = broadcast_object(payload, ctx)
            current_controller = Controller(
                gains=np.asarray(payload["gains"], dtype=np.float32),
                step_to_window=np.asarray(payload["step_to_window"], dtype=np.int64),
                bands=tuple(BandDefinition(**item) for item in payload["bands"]),
                source=str(payload["source"]),
            )
        controller = current_controller
        assert controller is not None
        if ctx.rank == 0:
            controller_path.parent.mkdir(parents=True, exist_ok=True)
            json_dump(controller_path, controller_to_json(controller, window_rows))
            save_controller_plot(controller, controller_path.with_suffix(".png"))
        barrier()

        # Scalar-IG audit on the same policy sample count, for direct comparison.
        scalar_dir = output_root / "03_fit"
        scalar_shard = scalar_dir / f"scalar_ig_rank{ctx.rank:02d}.npz"
        run_explicit_rollout_features(
            args=args,
            config=config,
            ctx=ctx,
            model=model,
            grid=grid,
            masks=masks,
            multiplicity=multiplicity,
            sample_count=int(args.policy_samples),
            noise_seed=int(args.policy_noise_seed) + 97,
            controller=None,
            scalar_gamma=float(config.guidance.ig.scale) - 1.0,
            output_path=scalar_shard,
            branch_name="scalar_ig",
        )
        barrier()
        if ctx.rank == 0:
            analyze_rollout_features(
                directory=scalar_dir,
                prefix="scalar_ig",
                world_size=ctx.world_size,
                target_power=target_power,
                target_state_power=target_state_power,
                grid=grid,
            )
        barrier()

        # Audit the final fitted controller itself.  The last policy-iteration
        # feature collection was generated by the previous controller, so this
        # extra held-out rollout is required for an honest post-fit diagnosis.
        final_shard = scalar_dir / f"spectral_final_rank{ctx.rank:02d}.npz"
        run_explicit_rollout_features(
            args=args,
            config=config,
            ctx=ctx,
            model=model,
            grid=grid,
            masks=masks,
            multiplicity=multiplicity,
            sample_count=int(args.policy_samples),
            noise_seed=int(args.policy_noise_seed) + 97,
            controller=controller,
            scalar_gamma=None,
            output_path=final_shard,
            branch_name="spectral_final",
        )
        barrier()
        if ctx.rank == 0:
            analyze_rollout_features(
                directory=scalar_dir,
                prefix="spectral_final",
                world_size=ctx.world_size,
                target_power=target_power,
                target_state_power=target_state_power,
                grid=grid,
            )
        barrier()

    resolved_controller_path = (
        args.controller_path.expanduser().resolve()
        if args.controller_path is not None
        else controller_path
    )
    if controller is None and resolved_controller_path.is_file():
        controller = controller_from_json(resolved_controller_path)
        # Copy an externally fitted controller into this versioned output root so
        # the sampling protocol remains self-contained and final_report can be
        # reproduced without the source run.
        if ctx.rank == 0 and resolved_controller_path != controller_path:
            controller_path.parent.mkdir(parents=True, exist_ok=True)
            source_payload = json.loads(resolved_controller_path.read_text(encoding="utf-8"))
            json_dump(controller_path, source_payload)
            save_controller_plot(controller, controller_path.with_suffix(".png"))
        barrier()

    if "pulse" in args.stages:
        assert model is not None
        run_pulse_audit(
            args=args,
            config=config,
            ctx=ctx,
            output_root=output_root,
            model=model,
            grid=grid,
            masks=masks,
            multiplicity=multiplicity,
            bands=bands,
        )

    conditions = build_conditions(args.conditions, bands)
    if "sample" in args.stages:
        if controller is None:
            raise FileNotFoundError("fitted controller missing")
        assert model is not None
        run_generation(
            args=args,
            config=config,
            ctx=ctx,
            output_root=output_root,
            model=model,
            grid=grid,
            masks=masks,
            multiplicity=multiplicity,
            controller=controller,
            conditions=conditions,
        )

    if model is not None:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        barrier()

    if "decode" in args.stages:
        decode_conditions(
            args=args,
            config=config,
            ctx=ctx,
            output_root=output_root,
            conditions=conditions,
        )

    if "evaluate" in args.stages and ctx.rank == 0:
        evaluate_conditions(args=args, output_root=output_root, conditions=conditions)
    barrier()

    if "report" in args.stages and ctx.rank == 0:
        build_report(output_root)
    barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()