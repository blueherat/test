"""Causal decoder-feature variance intervention for RAEv2 ImageNet sampling.

This script keeps the Stage-2 endpoint latent fixed and changes only one frozen
Stage-1 decoder hidden state.  It supports:

* exact same-noise scale=1 and target-IG endpoint caching;
* clean / scale=1 / target-IG decoder spatial-variance calibration;
* mean-preserving layer-global or per-channel variance intervention;
* direct fixed-alpha sweeps and log-space calibration toward IG or clean variance;
* reverse and over-correction controls on the target-IG endpoint;
* distributed decoding, sample archives, previews, clipping audits, and
  repository-standard torch-fidelity metrics.

The decoder intervention is applied after a complete ViT-MAE decoder block.
Only patch tokens are changed; the CLS token is left untouched:

    h' = mean_space(h) + alpha * (h - mean_space(h)).

Run from the repository root with torchrun.  See --help and the example at the
bottom of this file's companion response.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import os
import sys
from contextlib import AbstractContextManager, nullcontext
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RAEV2_SRC = ROOT / "external" / "RAEv2" / "src"
for _path in (RAEV2_SRC, ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from configs.stage2 import Stage2Config  # noqa: E402
from experiments.rae_strict_lpl import decoder_feature_pyramid  # noqa: E402
from experiments.raev2_stage1_compat import (  # noqa: E402
    install_raev2_decoder_config_compat,
)
from experiments.raev2_training_core import (  # noqa: E402
    DeterministicImageNetParquet,
    file_sha256,
    tensor_fingerprint,
)
from stage2.transport import create_sampler, create_transport  # noqa: E402
from stage2.utils import validate_stage2_config  # noqa: E402
from utils.guidance_utils import forward_with_internalguidance  # noqa: E402
from utils.model_utils import instantiate_from_config  # noqa: E402


PROTOCOL = "raev2_decoder_spatial_variance_intervention_v1"
DEFAULT_CONFIG = ROOT / "experiments/configs/raev2_strict_lpl_dinov3l_k7.yaml"
DEFAULT_REFERENCE = Path("/data/shared/adm_refs/VIRTUAL_imagenet256_labeled.npz")


def _first_existing(*paths: str) -> Path:
    candidates = [Path(value).expanduser() for value in paths]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


DEFAULT_CHECKPOINT = _first_existing(
    "/data/users/zhoushunyu/eqvae/models/RAEv2/stage2/imagenet/dinov3l-k7/checkpoint.pt",
    "/home/zhoushunyu/data/eqvae/models/RAEv2/stage2/imagenet/dinov3l-k7/checkpoint.pt",
)
DEFAULT_DINO_CKPT = _first_existing(
    "/data/users/zhoushunyu/eqvae/models/RAEv2/encoders/dinov3",
    "/home/zhoushunyu/data/eqvae/models/RAEv2/encoders/dinov3",
)
DEFAULT_DINO_REPO = _first_existing(
    "/data/users/zhoushunyu/eqvae/models/RAEv2/dinov3_repo",
    "/home/zhoushunyu/data/eqvae/models/RAEv2/dinov3_repo",
)


@dataclasses.dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    initialized: bool


@dataclasses.dataclass(frozen=True)
class CachePaths:
    ids: Path
    scale1: Path
    target_ig: Path
    audit: Path


@dataclasses.dataclass(frozen=True)
class VarianceStats:
    per_channel: np.ndarray
    sample_count: int

    @property
    def global_variance(self) -> float:
        return float(np.asarray(self.per_channel, dtype=np.float64).mean())

    def summary(self) -> dict[str, Any]:
        values = np.asarray(self.per_channel, dtype=np.float64)
        return {
            "sample_count": int(self.sample_count),
            "channels": int(values.size),
            "global_variance": float(values.mean()),
            "channel_variance_min": float(values.min()),
            "channel_variance_median": float(np.median(values)),
            "channel_variance_max": float(values.max()),
        }


@dataclasses.dataclass(frozen=True)
class InterventionSetting:
    name: str
    source: str
    target: str
    gamma: float | None
    alpha: np.ndarray
    description: str

    def summary(self, source_variance: np.ndarray) -> dict[str, Any]:
        alpha = np.asarray(self.alpha, dtype=np.float64)
        source = np.asarray(source_variance, dtype=np.float64)
        if alpha.size == 1:
            predicted = source * float(alpha.item()) ** 2
        else:
            if alpha.shape != source.shape:
                raise ValueError("alpha and source variance channel shapes differ")
            predicted = source * alpha.square()
        return {
            "name": self.name,
            "source": self.source,
            "target": self.target,
            "gamma": None if self.gamma is None else float(self.gamma),
            "description": self.description,
            "alpha_count": int(alpha.size),
            "alpha_min": float(alpha.min()),
            "alpha_mean": float(alpha.mean()),
            "alpha_median": float(np.median(alpha)),
            "alpha_max": float(alpha.max()),
            "predicted_global_variance": float(predicted.mean()),
        }


def decoder_patch_tokens(hidden: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Return spatial patch tokens and the number of untouched prefix tokens."""

    if hidden.ndim != 3:
        raise ValueError(f"expected BNC decoder tokens, got {tuple(hidden.shape)}")
    token_count = int(hidden.shape[1])
    side_without_prefix = math.isqrt(max(token_count - 1, 0))
    side_without_cls = side_without_prefix * side_without_prefix == token_count - 1
    side_with_no_prefix = math.isqrt(token_count)
    no_prefix = side_with_no_prefix * side_with_no_prefix == token_count
    if side_without_cls:
        return hidden[:, 1:, :], 1
    if no_prefix:
        return hidden, 0
    raise ValueError(
        "cannot identify decoder patch-token layout: "
        f"token_count={token_count} is neither square nor one-plus-square"
    )


class SpatialVarianceAccumulator:
    """Accumulate mean per-sample/per-channel variance over spatial positions."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.sum_per_channel: torch.Tensor | None = None
        self.sample_count = torch.zeros((), device=device, dtype=torch.float64)

    def update_bchw(self, feature: torch.Tensor) -> None:
        if feature.ndim != 4:
            raise ValueError(f"expected BCHW feature, got {tuple(feature.shape)}")
        values = feature.detach().to(dtype=torch.float32)
        centered = values - values.mean(dim=(-2, -1), keepdim=True)
        per_sample_channel = centered.square().mean(dim=(-2, -1)).to(torch.float64)
        self._add(per_sample_channel)

    def update_tokens(self, hidden: torch.Tensor) -> None:
        patch, _prefix_tokens = decoder_patch_tokens(hidden)
        patch = patch.detach().to(dtype=torch.float32)
        centered = patch - patch.mean(dim=1, keepdim=True)
        per_sample_channel = centered.square().mean(dim=1).to(torch.float64)
        self._add(per_sample_channel)

    def _add(self, per_sample_channel: torch.Tensor) -> None:
        if per_sample_channel.ndim != 2:
            raise ValueError("per-sample channel variance must be BC")
        channel_sum = per_sample_channel.sum(dim=0)
        if self.sum_per_channel is None:
            self.sum_per_channel = torch.zeros_like(channel_sum, dtype=torch.float64)
        if self.sum_per_channel.shape != channel_sum.shape:
            raise ValueError("decoder hidden channel count changed between batches")
        self.sum_per_channel.add_(channel_sum)
        self.sample_count.add_(float(per_sample_channel.shape[0]))

    def distributed_finalize(self, ctx: DistributedContext) -> VarianceStats:
        if self.sum_per_channel is None or float(self.sample_count.item()) <= 0:
            raise RuntimeError("no decoder features were accumulated")
        channel_sum = self.sum_per_channel.clone()
        count = self.sample_count.clone()
        if ctx.initialized:
            dist.all_reduce(channel_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
        per_channel = (channel_sum / count.clamp_min(1.0)).cpu().numpy()
        return VarianceStats(per_channel=per_channel, sample_count=int(count.item()))


class InterventionAudit:
    def __init__(self, device: torch.device) -> None:
        self.pre = SpatialVarianceAccumulator(device)
        self.post = SpatialVarianceAccumulator(device)
        self.max_abs_mean_shift = torch.zeros((), device=device, dtype=torch.float64)
        self.calls = torch.zeros((), device=device, dtype=torch.float64)

    def update(self, before: torch.Tensor, after: torch.Tensor) -> None:
        self.pre.update_tokens(before)
        self.post.update_tokens(after)
        before_patch, before_prefix = decoder_patch_tokens(before)
        after_patch, after_prefix = decoder_patch_tokens(after)
        if before_prefix != after_prefix:
            raise RuntimeError("decoder token prefix layout changed during intervention")
        before_mean = before_patch.float().mean(dim=1)
        after_mean = after_patch.float().mean(dim=1)
        shift = (after_mean - before_mean).abs().max().to(torch.float64)
        self.max_abs_mean_shift.copy_(torch.maximum(self.max_abs_mean_shift, shift))
        self.calls.add_(1.0)

    def distributed_summary(self, ctx: DistributedContext) -> dict[str, Any]:
        pre = self.pre.distributed_finalize(ctx)
        post = self.post.distributed_finalize(ctx)
        maximum = self.max_abs_mean_shift.clone()
        calls = self.calls.clone()
        if ctx.initialized:
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
            dist.all_reduce(calls, op=dist.ReduceOp.SUM)
        return {
            "hook_calls_total": int(calls.item()),
            "max_abs_spatial_mean_shift": float(maximum.item()),
            "pre": pre.summary(),
            "post": post.summary(),
            "observed_global_variance_ratio": float(
                post.global_variance / max(pre.global_variance, 1e-30)
            ),
        }


class DecoderVarianceHook(AbstractContextManager["DecoderVarianceHook"]):
    """Mean-preserving variance scaling after a 1-based decoder block index."""

    def __init__(
        self,
        decoder: torch.nn.Module,
        *,
        layer_index: int,
        alpha: np.ndarray,
        device: torch.device,
    ) -> None:
        depth = len(decoder.decoder_layers)
        if not 1 <= int(layer_index) <= depth:
            raise ValueError(f"layer_index must be in [1, {depth}], got {layer_index}")
        values = np.asarray(alpha, dtype=np.float64).reshape(-1)
        if values.size < 1 or not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("all intervention alpha values must be finite and positive")
        self.module = decoder.decoder_layers[int(layer_index) - 1]
        self.alpha_cpu = values
        self.device = device
        self.audit = InterventionAudit(device)
        self.handle: Any | None = None

    @staticmethod
    def _unpack(output: Any) -> tuple[torch.Tensor, Any]:
        if torch.is_tensor(output):
            return output, None
        if isinstance(output, (tuple, list)):
            items = list(output)
            tensor_indices = [
                index for index, item in enumerate(items) if torch.is_tensor(item)
            ]
            if not tensor_indices:
                raise TypeError("decoder block tuple/list output contains no Tensor")
            index = tensor_indices[0]
            return items[index], (type(output), items, index)
        raise TypeError("decoder block output must be a Tensor, tuple, or list")

    @staticmethod
    def _repack(hidden: torch.Tensor, remainder: Any) -> Any:
        if remainder is None:
            return hidden
        container, items, index = remainder
        items[index] = hidden
        return tuple(items) if container is tuple else items

    def _hook(self, _module: torch.nn.Module, _inputs: Any, output: Any) -> Any:
        hidden, remainder = self._unpack(output)
        patch, prefix_tokens = decoder_patch_tokens(hidden)
        channel_count = patch.shape[-1]
        if self.alpha_cpu.size not in (1, channel_count):
            raise ValueError(
                f"alpha has {self.alpha_cpu.size} values but hidden state has "
                f"{channel_count} channels"
            )
        alpha = torch.as_tensor(
            self.alpha_cpu,
            device=patch.device,
            dtype=torch.float32,
        ).reshape(1, 1, -1)
        patch32 = patch.float()
        spatial_mean = patch32.mean(dim=1, keepdim=True)
        changed32 = spatial_mean + alpha * (patch32 - spatial_mean)

        # Casting back to bf16 can introduce a tiny mean error.  One correction
        # before the final cast keeps the intended invariant as tight as the
        # decoder activation dtype permits.
        changed = changed32.to(dtype=patch.dtype)
        correction = spatial_mean - changed.float().mean(dim=1, keepdim=True)
        changed = (changed.float() + correction).to(dtype=patch.dtype)
        updated = (
            torch.cat([hidden[:, :prefix_tokens, :], changed], dim=1)
            if prefix_tokens
            else changed
        )
        self.audit.update(hidden, updated)
        return self._repack(updated, remainder)

    def __enter__(self) -> "DecoderVarianceHook":
        if self.handle is not None:
            raise RuntimeError("decoder intervention hook is already registered")
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--state-key", choices=("ema", "model"), default="ema")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Reusable endpoint cache. Defaults to OUTPUT_DIR/endpoint_cache.",
    )
    parser.add_argument("--sample-count", type=int, default=5000)
    parser.add_argument("--per-rank-batch", type=int, default=4)
    parser.add_argument("--sampling-seed", type=int, default=20260805)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--target-ig-scale", type=float, default=1.78)
    parser.add_argument("--cache-dtype", choices=("float32", "float16"), default="float16")
    parser.add_argument(
        "--overwrite-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--verify-cache-sha",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    layer = parser.add_argument_group("decoder intervention")
    layer.add_argument("--layer-fraction", type=float, default=0.6)
    layer.add_argument(
        "--layer-index",
        type=int,
        help="1-based decoder block index; overrides --layer-fraction.",
    )
    layer.add_argument(
        "--calibration-mode",
        choices=("global", "channel"),
        default="global",
    )
    layer.add_argument(
        "--gamma",
        dest="gammas",
        type=float,
        action="append",
        help="Log-space fraction toward target variance. Repeatable.",
    )
    layer.add_argument(
        "--fixed-alpha",
        dest="fixed_alphas",
        type=float,
        action="append",
        help=(
            "Direct mean-preserving spatial-standard-deviation multiplier on "
            "the scale=1 endpoint. Repeatable; independent of calibration targets."
        ),
    )
    layer.add_argument("--alpha-min", type=float, default=0.0)
    layer.add_argument("--alpha-max", type=float, default=0.0)
    layer.add_argument(
        "--include-controls",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include IG->scale1 reverse and IG further-amplification controls.",
    )

    clean = parser.add_argument_group("clean variance calibration")
    clean.add_argument(
        "--data-path",
        type=Path,
        help="ImageNet parquet root. Defaults to config.dataset.data_dir.",
    )
    clean.add_argument("--clean-split", choices=("train", "validation"), default="train")
    clean.add_argument("--clean-count", type=int, default=2048)
    clean.add_argument("--clean-start-index", type=int, default=0)
    clean.add_argument("--clean-seed", type=int, default=20260805)
    clean.add_argument("--clean-index-map", type=Path)

    metrics = parser.add_argument_group("output and metrics")
    metrics.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    metrics.add_argument("--metric-batch-size", type=int, default=64)
    metrics.add_argument("--metric-seed", type=int, default=0)
    metrics.add_argument("--preview-count", type=int, default=64)
    metrics.add_argument(
        "--skip-metrics",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    metrics.add_argument(
        "--keep-shards",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    environment = parser.add_argument_group("RAEv2 environment")
    environment.add_argument("--dino-ckpt-dir", type=Path, default=DEFAULT_DINO_CKPT)
    environment.add_argument("--dino-repo-dir", type=Path, default=DEFAULT_DINO_REPO)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.sample_count <= 0 or args.sample_count % 1000:
        raise ValueError("--sample-count must be a positive multiple of 1000")
    if args.per_rank_batch <= 0:
        raise ValueError("--per-rank-batch must be positive")
    if args.target_ig_scale <= 0:
        raise ValueError("--target-ig-scale must be positive")
    if args.clean_count <= 0 or args.clean_start_index < 0:
        raise ValueError("clean count must be positive and clean start index non-negative")
    if args.clean_split != "train" and args.clean_index_map is not None:
        raise ValueError("--clean-index-map is only supported with --clean-split train")
    if args.layer_index is None and not 0.0 < args.layer_fraction <= 1.0:
        raise ValueError("--layer-fraction must be in (0, 1]")
    if args.alpha_min < 0 or args.alpha_max < 0:
        raise ValueError("alpha clamps must be non-negative; zero disables that bound")
    if args.alpha_min and args.alpha_max and args.alpha_min > args.alpha_max:
        raise ValueError("--alpha-min cannot exceed --alpha-max")
    if args.metric_batch_size <= 0:
        raise ValueError("metric batch size must be positive")
    if args.preview_count < 0:
        raise ValueError("preview count must be non-negative")
    gammas = args.gammas or [-0.5, 0.5, 1.0, 1.5]
    if not all(math.isfinite(value) for value in gammas):
        raise ValueError("all gamma values must be finite")
    args.gammas = tuple(dict.fromkeys(float(value) for value in gammas))
    fixed_alphas = tuple(dict.fromkeys(float(value) for value in (args.fixed_alphas or ())))
    if any((not math.isfinite(value)) or value <= 0 for value in fixed_alphas):
        raise ValueError("all fixed alpha values must be finite and positive")
    args.fixed_alphas = fixed_alphas


def init_distributed() -> DistributedContext:
    initialized = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if initialized:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return DistributedContext(rank, world_size, local_rank, device, initialized)


def barrier(ctx: DistributedContext) -> None:
    if ctx.initialized:
        dist.barrier()


def destroy_distributed(ctx: DistributedContext) -> None:
    if ctx.initialized:
        dist.destroy_process_group()


def broadcast_object(value: Any, ctx: DistributedContext, *, source: int = 0) -> Any:
    if not ctx.initialized:
        return value
    payload = [value if ctx.rank == source else None]
    dist.broadcast_object_list(payload, src=source)
    return payload[0]


def autocast_context(precision: str):
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def load_config(path: Path) -> Stage2Config:
    config = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(Stage2Config), OmegaConf.load(path))
    )
    config.post_process()
    validate_stage2_config(config)
    if config.transport.prediction != "x":
        raise ValueError("this experiment requires RAEv2 clean-latent x-prediction")
    if config.guidance.ig is None:
        raise ValueError("the Stage-2 config has no internal-guidance configuration")
    if abs(float(config.guidance.cfg.scale) - 1.0) > 1e-12:
        raise ValueError(
            "this causal experiment assumes CFG scale=1; use the current RAEv2 ImageNet config"
        )
    config.prepare_model_params()
    return config


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def generator_fingerprint(generator: torch.Generator) -> str:
    return hashlib.sha256(generator.get_state().cpu().numpy().tobytes()).hexdigest()


def resolve_layer_index(
    decoder: torch.nn.Module,
    *,
    layer_index: int | None,
    layer_fraction: float,
) -> tuple[int, int]:
    depth = len(decoder.decoder_layers)
    if layer_index is None:
        selected = min(max(int(round(float(layer_fraction) * depth)), 1), depth)
    else:
        selected = int(layer_index)
    if not 1 <= selected <= depth:
        raise ValueError(f"decoder layer index must be in [1, {depth}], got {selected}")
    return selected, depth


def cache_paths(cache_dir: Path, rank: int) -> CachePaths:
    return CachePaths(
        ids=cache_dir / f"ids-rank{rank:02d}.npy",
        scale1=cache_dir / f"scale1-latents-rank{rank:02d}.npy",
        target_ig=cache_dir / f"target-ig-latents-rank{rank:02d}.npy",
        audit=cache_dir / f"cache-audit-rank{rank:02d}.json",
    )


def expected_cache_manifest(
    args: argparse.Namespace,
    config: Stage2Config,
    ctx: DistributedContext,
    checkpoint_sha256: str,
    checkpoint_step: int,
    global_ids: np.ndarray,
) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "rank": ctx.rank,
        "world_size": ctx.world_size,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": int(checkpoint_step),
        "state_key": args.state_key,
        "sample_count": int(args.sample_count),
        "local_sample_count": int(len(global_ids)),
        "first_global_id": int(global_ids[0]) if len(global_ids) else None,
        "last_global_id": int(global_ids[-1]) if len(global_ids) else None,
        "sampling_seed": int(args.sampling_seed),
        "precision": args.precision,
        "cache_dtype": args.cache_dtype,
        "target_ig_scale": float(args.target_ig_scale),
        "ig_t_min": float(config.guidance.ig.t_min),
        "ig_t_max": float(config.guidance.ig.t_max),
        "sampler_steps": int(config.sampler.num_steps),
        "latent_size": list(config.misc.latent_size),
        "num_classes": int(config.misc.num_classes),
    }


def _manifest_matches(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    ignored = {
        "initial_generator_sha256",
        "final_generator_sha256",
        "first_noise_sha256",
        "first_label_sha256",
        "scale1_cache_sha256",
        "target_ig_cache_sha256",
    }
    return all(actual.get(key) == value for key, value in expected.items() if key not in ignored)


def cache_is_valid(
    paths: CachePaths,
    expected: Mapping[str, Any],
    *,
    latent_size: Sequence[int],
    verify_sha: bool,
) -> bool:
    required = (paths.ids, paths.scale1, paths.target_ig, paths.audit)
    if not all(path.is_file() for path in required):
        return False
    try:
        audit = json.loads(paths.audit.read_text(encoding="utf-8"))
        if not _manifest_matches(audit, expected):
            return False
        ids = np.load(paths.ids, allow_pickle=False)
        expected_shape = (int(expected["local_sample_count"]), *map(int, latent_size))
        scale1 = np.load(paths.scale1, mmap_mode="r", allow_pickle=False)
        target = np.load(paths.target_ig, mmap_mode="r", allow_pickle=False)
        if ids.shape != (expected_shape[0],):
            return False
        if tuple(scale1.shape) != expected_shape or tuple(target.shape) != expected_shape:
            return False
        if str(scale1.dtype) != str(expected["cache_dtype"]):
            return False
        if str(target.dtype) != str(expected["cache_dtype"]):
            return False
        if verify_sha:
            if audit.get("scale1_cache_sha256") != file_sha256(paths.scale1):
                return False
            if audit.get("target_ig_cache_sha256") != file_sha256(paths.target_ig):
                return False
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    return True


def build_pure_ig_sampler(
    model: torch.nn.Module,
    config: Stage2Config,
    transport: Any,
    scale: float,
) -> tuple[Any, Any, dict[str, Any]]:
    guidance = copy.deepcopy(config.guidance)
    guidance.ig.scale = float(scale)
    sampler = create_sampler(transport, guidance_config=guidance)
    sample_fn = sampler.sample_ode(**dataclasses.asdict(config.sampler))
    model_fn = partial(forward_with_internalguidance, model)
    kwargs = {
        "ig_scale": float(scale),
        "ig_interval": (
            float(guidance.ig.t_min),
            float(guidance.ig.t_max),
        ),
    }
    return sample_fn, model_fn, kwargs


def sample_endpoint(
    *,
    noise: torch.Tensor,
    labels: torch.Tensor,
    sample_fn: Any,
    model_fn: Any,
    model_kwargs_base: Mapping[str, Any],
    num_classes: int,
    precision: str,
) -> torch.Tensor:
    batch_size = noise.shape[0]
    doubled_noise = torch.cat([noise, noise], dim=0)
    null = torch.full(
        (batch_size,),
        int(num_classes),
        device=noise.device,
        dtype=torch.long,
    )
    context = torch.cat([labels, null], dim=0)
    model_kwargs = dict(model_kwargs_base)
    model_kwargs.update(context=context, attn_mask=None)
    with torch.inference_mode(), autocast_context(precision):
        endpoint = sample_fn(doubled_noise, model_fn, **model_kwargs)[-1]
        if endpoint.shape[0] == 2 * batch_size:
            endpoint = endpoint.chunk(2, dim=0)[0]
        elif endpoint.shape[0] != batch_size:
            raise RuntimeError(
                f"unexpected sampler endpoint batch {endpoint.shape[0]} "
                f"for input batch {batch_size}"
            )
    return endpoint.float()


def generate_endpoint_cache(
    *,
    args: argparse.Namespace,
    config: Stage2Config,
    ctx: DistributedContext,
    model: torch.nn.Module,
    checkpoint_sha256: str,
    checkpoint_step: int,
    cache_dir: Path,
    force: bool = False,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = cache_paths(cache_dir, ctx.rank)
    global_ids = np.arange(ctx.rank, args.sample_count, ctx.world_size, dtype=np.int64)
    expected = expected_cache_manifest(
        args,
        config,
        ctx,
        checkpoint_sha256,
        checkpoint_step,
        global_ids,
    )
    if not force and cache_is_valid(
        paths,
        expected,
        latent_size=config.misc.latent_size,
        verify_sha=args.verify_cache_sha,
    ):
        if ctx.rank == 0:
            print(f"Reusing endpoint cache under {cache_dir}", flush=True)
        return
    if not (args.overwrite_cache or force) and any(
        path.exists() for path in (paths.ids, paths.scale1, paths.target_ig, paths.audit)
    ):
        raise RuntimeError(
            f"rank {ctx.rank} cache exists but is incomplete or mismatched; "
            "pass --overwrite-cache after checking the cache directory"
        )

    for path in (paths.ids, paths.scale1, paths.target_ig, paths.audit):
        if path.exists():
            path.unlink()
    np.save(paths.ids, global_ids, allow_pickle=False)
    dtype = np.float32 if args.cache_dtype == "float32" else np.float16
    shape = (len(global_ids), *map(int, config.misc.latent_size))
    scale1_map = np.lib.format.open_memmap(paths.scale1, mode="w+", dtype=dtype, shape=shape)
    target_map = np.lib.format.open_memmap(paths.target_ig, mode="w+", dtype=dtype, shape=shape)

    latent_size = tuple(config.misc.latent_size)
    shift = math.sqrt(
        (config.misc.time_dist_shift_dim or math.prod(latent_size))
        / config.misc.time_dist_shift_base
    )
    transport = create_transport(config=config.transport, time_dist_shift=shift)
    runtime_scale1 = build_pure_ig_sampler(model, config, transport, 1.0)
    runtime_target = build_pure_ig_sampler(model, config, transport, args.target_ig_scale)

    generator = torch.Generator(device=ctx.device)
    generator.manual_seed(int(args.sampling_seed) * ctx.world_size + ctx.rank)
    initial_generator_sha256 = generator_fingerprint(generator)
    first_noise_sha256: str | None = None
    first_label_sha256: str | None = None

    for start in range(0, len(global_ids), args.per_rank_batch):
        ids = global_ids[start : start + args.per_rank_batch]
        batch_size = len(ids)
        noise = torch.randn(
            batch_size,
            *latent_size,
            generator=generator,
            device=ctx.device,
            dtype=torch.float32,
        )
        labels = torch.from_numpy(ids % int(config.misc.num_classes)).to(
            device=ctx.device,
            dtype=torch.long,
        )
        if first_noise_sha256 is None:
            first_noise_sha256 = tensor_fingerprint(noise)
            first_label_sha256 = tensor_fingerprint(labels)

        scale1 = sample_endpoint(
            noise=noise.clone(),
            labels=labels,
            sample_fn=runtime_scale1[0],
            model_fn=runtime_scale1[1],
            model_kwargs_base=runtime_scale1[2],
            num_classes=int(config.misc.num_classes),
            precision=args.precision,
        )
        target = sample_endpoint(
            noise=noise.clone(),
            labels=labels,
            sample_fn=runtime_target[0],
            model_fn=runtime_target[1],
            model_kwargs_base=runtime_target[2],
            num_classes=int(config.misc.num_classes),
            precision=args.precision,
        )
        scale1_map[start : start + batch_size] = scale1.cpu().numpy().astype(dtype, copy=False)
        target_map[start : start + batch_size] = target.cpu().numpy().astype(dtype, copy=False)
        if ctx.rank == 0 and (start == 0 or (start // args.per_rank_batch) % 50 == 0):
            print(
                f"Caching endpoints: local {start + batch_size}/{len(global_ids)}",
                flush=True,
            )

    scale1_map.flush()
    target_map.flush()
    del scale1_map, target_map
    audit = {
        **expected,
        "initial_generator_sha256": initial_generator_sha256,
        "final_generator_sha256": generator_fingerprint(generator),
        "first_noise_sha256": first_noise_sha256,
        "first_label_sha256": first_label_sha256,
        "scale1_cache_sha256": file_sha256(paths.scale1),
        "target_ig_cache_sha256": file_sha256(paths.target_ig),
    }
    json_dump(paths.audit, audit)
    barrier(ctx)


def _batch_tensor(array: np.ndarray, start: int, end: int, device: torch.device) -> torch.Tensor:
    # Copy out of a read-only memmap so torch never sees non-writable NumPy storage.
    return torch.from_numpy(np.array(array[start:end], copy=True)).to(
        device=device,
        dtype=torch.float32,
        non_blocking=False,
    )


def measure_cached_variance(
    *,
    rae: torch.nn.Module,
    cache_path: Path,
    layer_index: int,
    batch_size: int,
    precision: str,
    ctx: DistributedContext,
) -> VarianceStats:
    array = np.load(cache_path, mmap_mode="r", allow_pickle=False)
    accumulator = SpatialVarianceAccumulator(ctx.device)
    with torch.inference_mode():
        for start in range(0, len(array), batch_size):
            latent = _batch_tensor(array, start, min(start + batch_size, len(array)), ctx.device)
            with autocast_context(precision):
                feature = decoder_feature_pyramid(
                    rae,
                    latent,
                    layer_indices=(int(layer_index),),
                )[0]
            accumulator.update_bchw(feature)
    return accumulator.distributed_finalize(ctx)


def measure_clean_variance(
    *,
    args: argparse.Namespace,
    config: Stage2Config,
    rae: torch.nn.Module,
    layer_index: int,
    ctx: DistributedContext,
) -> VarianceStats:
    data_path = (
        args.data_path.expanduser().resolve()
        if args.data_path is not None
        else Path(config.dataset.data_dir).expanduser().resolve()
    )
    dataset = DeterministicImageNetParquet(
        data_path,
        split=args.clean_split,
        image_size=int(config.training.image_size),
        augmentation_seed=0,
        horizontal_flip=False,
        index_map_path=args.clean_index_map,
    )
    available = len(dataset) - int(args.clean_start_index)
    count = min(int(args.clean_count), available)
    if count < ctx.world_size:
        raise ValueError("clean calibration count must be at least the distributed world size")
    # Use a fixed uniform subset instead of the first parquet rows, which may
    # retain class or shard ordering and bias the variance target.
    rng = np.random.default_rng(int(args.clean_seed))
    selected = rng.choice(available, size=count, replace=False).astype(np.int64)
    selected += int(args.clean_start_index)
    indices = selected[ctx.rank :: ctx.world_size]
    accumulator = SpatialVarianceAccumulator(ctx.device)
    with torch.inference_mode():
        for start in range(0, len(indices), args.per_rank_batch):
            batch_indices = indices[start : start + args.per_rank_batch]
            images = torch.stack([dataset[int(index)][0] for index in batch_indices]).to(ctx.device)
            with autocast_context(args.precision):
                latent = rae.encode(images).float()
                feature = decoder_feature_pyramid(
                    rae,
                    latent,
                    layer_indices=(int(layer_index),),
                )[0]
            accumulator.update_bchw(feature)
    return accumulator.distributed_finalize(ctx)


def _slug_float(value: float) -> str:
    text = f"{float(value):.6g}"
    return text.replace("-", "m").replace("+", "").replace(".", "p")


def _clamp_alpha(alpha: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    values = np.asarray(alpha, dtype=np.float64)
    lower = float(args.alpha_min) if args.alpha_min > 0 else -np.inf
    upper = float(args.alpha_max) if args.alpha_max > 0 else np.inf
    return np.clip(values, lower, upper)


def make_alpha(
    source: VarianceStats,
    target: VarianceStats,
    *,
    gamma: float,
    mode: str,
    args: argparse.Namespace,
    eps: float = 1e-30,
) -> np.ndarray:
    if source.per_channel.shape != target.per_channel.shape:
        raise ValueError("source and target decoder channel counts differ")
    if mode == "global":
        log_ratio = math.log(max(target.global_variance, eps)) - math.log(
            max(source.global_variance, eps)
        )
        alpha = np.asarray([math.exp(0.5 * float(gamma) * log_ratio)], dtype=np.float64)
    elif mode == "channel":
        source_values = np.maximum(source.per_channel.astype(np.float64), eps)
        target_values = np.maximum(target.per_channel.astype(np.float64), eps)
        alpha = np.exp(0.5 * float(gamma) * (np.log(target_values) - np.log(source_values)))
    else:
        raise ValueError(f"unknown calibration mode: {mode}")
    return _clamp_alpha(alpha, args)


def build_settings(
    *,
    args: argparse.Namespace,
    scale1: VarianceStats,
    target_ig: VarianceStats,
    clean: VarianceStats,
) -> list[InterventionSetting]:
    settings = [
        InterventionSetting(
            name="s1_identity",
            source="scale1",
            target="identity",
            gamma=0.0,
            alpha=np.asarray([1.0], dtype=np.float64),
            description="Scale=1 endpoint, ordinary frozen decoder.",
        ),
        InterventionSetting(
            name=f"ig_target_s{_slug_float(args.target_ig_scale)}",
            source="target_ig",
            target="identity",
            gamma=0.0,
            alpha=np.asarray([1.0], dtype=np.float64),
            description="True target internal-guidance endpoint, ordinary decoder.",
        ),
    ]
    for alpha in args.fixed_alphas:
        if abs(float(alpha) - 1.0) < 1e-15:
            continue
        settings.append(
            InterventionSetting(
                name=f"s1_fixed_a{_slug_float(alpha)}",
                source="scale1",
                target="fixed_alpha",
                gamma=None,
                alpha=np.asarray([float(alpha)], dtype=np.float64),
                description=(
                    "Scale=1 endpoint with direct mean-preserving decoder spatial "
                    f"standard-deviation multiplier alpha={alpha:g}."
                ),
            )
        )
    targets = (("ig", target_ig), ("clean", clean))
    for target_name, target_stats in targets:
        for gamma in args.gammas:
            if abs(float(gamma)) < 1e-15:
                continue
            settings.append(
                InterventionSetting(
                    name=f"s1_to_{target_name}_g{_slug_float(gamma)}_{args.calibration_mode}",
                    source="scale1",
                    target=target_name,
                    gamma=float(gamma),
                    alpha=make_alpha(
                        scale1,
                        target_stats,
                        gamma=float(gamma),
                        mode=args.calibration_mode,
                        args=args,
                    ),
                    description=(
                        f"Scale=1 endpoint; decoder variance moved gamma={gamma:g} "
                        f"in log space toward {target_name}."
                    ),
                )
            )
    if args.include_controls:
        settings.extend(
            [
                InterventionSetting(
                    name=f"ig_to_s1_g1_{args.calibration_mode}",
                    source="target_ig",
                    target="scale1",
                    gamma=1.0,
                    alpha=make_alpha(
                        target_ig,
                        scale1,
                        gamma=1.0,
                        mode=args.calibration_mode,
                        args=args,
                    ),
                    description=(
                        "True IG endpoint with decoder variance attenuated "
                        "back to scale=1."
                    ),
                ),
                InterventionSetting(
                    name=f"ig_further_by_s1_to_ig_g1_{args.calibration_mode}",
                    source="target_ig",
                    target="further_ig_direction",
                    gamma=1.0,
                    alpha=make_alpha(
                        scale1,
                        target_ig,
                        gamma=1.0,
                        mode=args.calibration_mode,
                        args=args,
                    ),
                    description=(
                        "True IG endpoint with the same multiplicative shift "
                        "applied once more."
                    ),
                ),
            ]
        )
    names = [setting.name for setting in settings]
    if len(names) != len(set(names)):
        raise RuntimeError("intervention setting names are not unique")
    return settings


def source_stats_for(
    source: str,
    *,
    scale1: VarianceStats,
    target_ig: VarianceStats,
) -> VarianceStats:
    if source == "scale1":
        return scale1
    if source == "target_ig":
        return target_ig
    raise ValueError(f"unknown latent source {source!r}")


def setting_paths(output_dir: Path, name: str, rank: int) -> dict[str, Path]:
    directory = output_dir / "settings" / name
    return {
        "directory": directory,
        "images": directory / f"images-rank{rank:02d}.npy",
        "ids": directory / f"ids-rank{rank:02d}.npy",
        "audit": directory / f"intervention-audit-rank{rank:02d}.json",
        "archive": directory / "samples.npz",
        "summary": directory / "sampling_summary.json",
        "preview": directory / "preview.png",
    }


def save_preview(images: np.ndarray, path: Path, *, count: int) -> None:
    count = min(int(count), len(images))
    if count <= 0:
        return
    selected = np.asarray(images[:count], dtype=np.uint8)
    columns = min(8, count)
    rows = int(math.ceil(count / columns))
    height, width = selected.shape[1:3]
    canvas = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for index, image in enumerate(selected):
        row, column = divmod(index, columns)
        canvas[row * height : (row + 1) * height, column * width : (column + 1) * width] = image
    Image.fromarray(canvas, mode="RGB").save(path)


def decode_setting(
    *,
    args: argparse.Namespace,
    config: Stage2Config,
    ctx: DistributedContext,
    rae: torch.nn.Module,
    cache_dir: Path,
    output_dir: Path,
    layer_index: int,
    setting: InterventionSetting,
    source_variance: VarianceStats,
) -> Path:
    paths = setting_paths(output_dir, setting.name, ctx.rank)
    paths["directory"].mkdir(parents=True, exist_ok=True)
    caches = cache_paths(cache_dir, ctx.rank)
    source_path = caches.scale1 if setting.source == "scale1" else caches.target_ig
    ids = np.load(caches.ids, allow_pickle=False)
    latent = np.load(source_path, mmap_mode="r", allow_pickle=False)
    if len(ids) != len(latent):
        raise RuntimeError("endpoint cache IDs and latent count differ")
    np.save(paths["ids"], ids, allow_pickle=False)

    image_size = int(config.training.image_size)
    image_map = np.lib.format.open_memmap(
        paths["images"],
        mode="w+",
        dtype=np.uint8,
        shape=(len(ids), image_size, image_size, 3),
    )
    use_hook = not (setting.alpha.size == 1 and abs(float(setting.alpha.item()) - 1.0) < 1e-15)
    hook_context: AbstractContextManager[Any]
    hook: DecoderVarianceHook | None
    if use_hook:
        hook = DecoderVarianceHook(
            rae.decoder,
            layer_index=layer_index,
            alpha=setting.alpha,
            device=ctx.device,
        )
        hook_context = hook
    else:
        hook = None
        hook_context = nullcontext()

    raw_min = torch.full((), float("inf"), device=ctx.device, dtype=torch.float64)
    raw_max = torch.full((), float("-inf"), device=ctx.device, dtype=torch.float64)
    clipped_low = torch.zeros((), device=ctx.device, dtype=torch.float64)
    clipped_high = torch.zeros((), device=ctx.device, dtype=torch.float64)
    pixel_count = torch.zeros((), device=ctx.device, dtype=torch.float64)
    with hook_context, torch.inference_mode():
        for start in range(0, len(latent), args.per_rank_batch):
            end = min(start + args.per_rank_batch, len(latent))
            batch = _batch_tensor(latent, start, end, ctx.device)
            with autocast_context(args.precision):
                decoded_raw = rae.decode(batch).float()
            if not torch.isfinite(decoded_raw).all():
                raise FloatingPointError(f"decoder produced non-finite pixels for {setting.name}")
            raw_min.copy_(torch.minimum(raw_min, decoded_raw.min().double()))
            raw_max.copy_(torch.maximum(raw_max, decoded_raw.max().double()))
            clipped_low.add_((decoded_raw < 0).sum().double())
            clipped_high.add_((decoded_raw > 1).sum().double())
            pixel_count.add_(float(decoded_raw.numel()))
            decoded = decoded_raw.clamp(0, 1)
            uint8 = (
                decoded.mul(255)
                .permute(0, 2, 3, 1)
                .to(device="cpu", dtype=torch.uint8)
                .numpy()
            )
            image_map[start:end] = uint8
    image_map.flush()
    del image_map
    if ctx.initialized:
        dist.all_reduce(raw_min, op=dist.ReduceOp.MIN)
        dist.all_reduce(raw_max, op=dist.ReduceOp.MAX)
        dist.all_reduce(clipped_low, op=dist.ReduceOp.SUM)
        dist.all_reduce(clipped_high, op=dist.ReduceOp.SUM)
        dist.all_reduce(pixel_count, op=dist.ReduceOp.SUM)
    pixel_audit = {
        "raw_min": float(raw_min.item()),
        "raw_max": float(raw_max.item()),
        "clipped_low_fraction": float((clipped_low / pixel_count.clamp_min(1.0)).item()),
        "clipped_high_fraction": float((clipped_high / pixel_count.clamp_min(1.0)).item()),
    }

    if hook is None:
        audit = {
            "identity": True,
            "hook_calls_total": 0,
            "max_abs_spatial_mean_shift": 0.0,
            "pre": source_variance.summary(),
            "post": source_variance.summary(),
            "observed_global_variance_ratio": 1.0,
        }
    else:
        audit = {"identity": False, **hook.audit.distributed_summary(ctx)}
    json_dump(
        paths["audit"],
        {
            "protocol": PROTOCOL,
            "rank": ctx.rank,
            "world_size": ctx.world_size,
            "layer_index": int(layer_index),
            "setting": setting.summary(source_variance.per_channel),
            "audit": audit,
            "pixel_audit": pixel_audit,
        },
    )
    barrier(ctx)

    if ctx.rank == 0:
        total = int(args.sample_count)
        merged_path = paths["directory"] / "samples-merged.npy"
        merged = np.lib.format.open_memmap(
            merged_path,
            mode="w+",
            dtype=np.uint8,
            shape=(total, image_size, image_size, 3),
        )
        seen = np.zeros(total, dtype=np.bool_)
        for rank in range(ctx.world_size):
            rank_paths = setting_paths(output_dir, setting.name, rank)
            rank_ids = np.load(rank_paths["ids"], allow_pickle=False)
            rank_images = np.load(rank_paths["images"], mmap_mode="r", allow_pickle=False)
            if np.any(rank_ids < 0) or np.any(rank_ids >= total):
                raise RuntimeError("distributed sample IDs fall outside the expected range")
            if seen[rank_ids].any():
                raise RuntimeError("distributed sample IDs contain duplicates")
            merged[rank_ids] = rank_images
            seen[rank_ids] = True
        if not seen.all():
            raise RuntimeError("distributed sample IDs are incomplete")
        merged.flush()
        save_preview(merged, paths["preview"], count=args.preview_count)
        np.savez(paths["archive"], merged)
        del merged
        merged_path.unlink()
        summary = {
            "protocol": PROTOCOL,
            "setting": setting.summary(source_variance.per_channel),
            "samples": total,
            "shape": [total, image_size, image_size, 3],
            "archive": str(paths["archive"]),
            "archive_sha256": file_sha256(paths["archive"]),
            "preview": str(paths["preview"]) if paths["preview"].is_file() else None,
        }
        json_dump(paths["summary"], summary)
        if not args.keep_shards:
            for rank in range(ctx.world_size):
                rank_paths = setting_paths(output_dir, setting.name, rank)
                rank_paths["images"].unlink(missing_ok=True)
                rank_paths["ids"].unlink(missing_ok=True)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    barrier(ctx)
    return paths["archive"]


def evaluate_settings(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    settings: Sequence[InterventionSetting],
    stats: Mapping[str, VarianceStats],
) -> pd.DataFrame:
    # Reuse the repository's established evaluator verbatim so this experiment
    # has the same Inception preprocessing, reference cache and metric options
    # as prior RAEv2 scale sweeps.
    from experiments.evaluate_raev2_samples import (
        NumpyRGBDataset,
        torch_fidelity_metrics,
    )

    reference_path = args.reference.expanduser().resolve()
    reference = NumpyRGBDataset(reference_path)
    reference_sha256 = file_sha256(reference_path)
    rows: list[dict[str, Any]] = []
    for setting in settings:
        archive = setting_paths(output_dir, setting.name, 0)["archive"]
        samples = NumpyRGBDataset(archive)
        metrics = torch_fidelity_metrics(
            samples,
            reference,
            batch_size=int(args.metric_batch_size),
            cache_name="raev2_imagenet256_virtual_reference",
            rng_seed=int(args.metric_seed),
        )
        source_stats = stats[setting.source]
        row = {
            "setting": setting.name,
            "source": setting.source,
            "target": setting.target,
            "gamma": setting.gamma,
            "sample_path": str(archive),
            "sample_sha256": file_sha256(archive),
            "sample_count": len(samples),
            "reference_path": str(reference_path),
            "reference_sha256": reference_sha256,
            **setting.summary(source_stats.per_channel),
            **metrics,
        }
        rows.append(row)
        json_dump(output_dir / "metrics_partial.json", rows)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "metrics.csv", index=False)
    json_dump(output_dir / "metrics.json", rows)
    return frame

def checkpoint_identity(
    path: Path,
    ctx: DistributedContext,
) -> tuple[str, int, dict[str, Any] | None]:
    checkpoint_sha256: str | None = None
    if ctx.rank == 0:
        checkpoint_sha256 = file_sha256(path)
    checkpoint_sha256 = broadcast_object(checkpoint_sha256, ctx)
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    checkpoint_step = int(checkpoint.get("step", 0))
    return str(checkpoint_sha256), checkpoint_step, checkpoint


def main() -> None:
    install_raev2_decoder_config_compat()
    args = parse_args()
    validate_args(args)
    args.config = args.config.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.cache_dir = (
        args.cache_dir.expanduser().resolve()
        if args.cache_dir is not None
        else args.output_dir / "endpoint_cache"
    )
    args.dino_ckpt_dir = args.dino_ckpt_dir.expanduser().resolve()
    args.dino_repo_dir = args.dino_repo_dir.expanduser().resolve()
    os.environ["DINOV3_CKPT_DIR"] = str(args.dino_ckpt_dir)
    os.environ["DINOV3_REPO_DIR"] = str(args.dino_repo_dir)

    ctx = init_distributed()
    try:
        if args.sample_count < ctx.world_size or args.clean_count < ctx.world_size:
            raise ValueError("sample and clean counts must each be at least world size")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        config = load_config(args.config)

        # Load Stage 1 first.  Clean calibration needs the encoder, but endpoint
        # sampling does not; deleting it before Stage 2 keeps peak memory low.
        rae = instantiate_from_config(config.stage_1).to(ctx.device).eval()
        rae.requires_grad_(False)
        layer_index, decoder_depth = resolve_layer_index(
            rae.decoder,
            layer_index=args.layer_index,
            layer_fraction=args.layer_fraction,
        )
        clean_stats = measure_clean_variance(
            args=args,
            config=config,
            rae=rae,
            layer_index=layer_index,
            ctx=ctx,
        )
        del rae.encoder
        torch.cuda.empty_cache()

        checkpoint_sha256, checkpoint_step, checkpoint = checkpoint_identity(
            args.checkpoint,
            ctx,
        )
        if args.state_key not in checkpoint:
            raise KeyError(f"checkpoint has no {args.state_key!r} state")
        global_ids = np.arange(ctx.rank, args.sample_count, ctx.world_size, dtype=np.int64)
        expected = expected_cache_manifest(
            args,
            config,
            ctx,
            checkpoint_sha256,
            checkpoint_step,
            global_ids,
        )
        paths = cache_paths(args.cache_dir, ctx.rank)
        valid_cache = cache_is_valid(
            paths,
            expected,
            latent_size=config.misc.latent_size,
            verify_sha=args.verify_cache_sha,
        )
        validity = torch.tensor(1 if valid_cache else 0, device=ctx.device, dtype=torch.int32)
        cache_exists = torch.tensor(
            1
            if any(
                path.exists()
                for path in (paths.ids, paths.scale1, paths.target_ig, paths.audit)
            )
            else 0,
            device=ctx.device,
            dtype=torch.int32,
        )
        if ctx.initialized:
            dist.all_reduce(validity, op=dist.ReduceOp.MIN)
            dist.all_reduce(cache_exists, op=dist.ReduceOp.MAX)
        all_valid = bool(validity.item())
        any_cache_exists = bool(cache_exists.item())
        if not all_valid:
            if any_cache_exists and not args.overwrite_cache:
                raise RuntimeError(
                    "endpoint cache is incomplete or protocol-mismatched on at least one rank; "
                    "inspect it, then rerun with --overwrite-cache to regenerate all ranks"
                )
            model = instantiate_from_config(config.stage_2).to(ctx.device).eval()
            model.requires_grad_(False)
            model.load_state_dict(checkpoint[args.state_key], strict=True)
            generate_endpoint_cache(
                args=args,
                config=config,
                ctx=ctx,
                model=model,
                checkpoint_sha256=checkpoint_sha256,
                checkpoint_step=checkpoint_step,
                cache_dir=args.cache_dir,
                force=True,
            )
            del model
            torch.cuda.empty_cache()
        elif ctx.rank == 0:
            print(f"Validated reusable endpoint cache: {args.cache_dir}", flush=True)
        del checkpoint
        torch.cuda.empty_cache()
        barrier(ctx)

        scale1_stats = measure_cached_variance(
            rae=rae,
            cache_path=paths.scale1,
            layer_index=layer_index,
            batch_size=args.per_rank_batch,
            precision=args.precision,
            ctx=ctx,
        )
        target_ig_stats = measure_cached_variance(
            rae=rae,
            cache_path=paths.target_ig,
            layer_index=layer_index,
            batch_size=args.per_rank_batch,
            precision=args.precision,
            ctx=ctx,
        )
        stats = {
            "scale1": scale1_stats,
            "target_ig": target_ig_stats,
            "clean": clean_stats,
        }
        settings = build_settings(
            args=args,
            scale1=scale1_stats,
            target_ig=target_ig_stats,
            clean=clean_stats,
        )

        if ctx.rank == 0:
            np.savez(
                args.output_dir / "calibration_variances.npz",
                scale1=scale1_stats.per_channel,
                target_ig=target_ig_stats.per_channel,
                clean=clean_stats.per_channel,
            )
            calibration = {
                "protocol": PROTOCOL,
                "config": str(args.config),
                "checkpoint": str(args.checkpoint),
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_step": checkpoint_step,
                "state_key": args.state_key,
                "decoder_depth": decoder_depth,
                "layer_index": layer_index,
                "layer_fraction_realized": layer_index / decoder_depth,
                "calibration_mode": args.calibration_mode,
                "target_ig_scale": args.target_ig_scale,
                "statistics": {name: value.summary() for name, value in stats.items()},
                "settings": [
                    setting.summary(
                        source_stats_for(
                            setting.source,
                            scale1=scale1_stats,
                            target_ig=target_ig_stats,
                        ).per_channel
                    )
                    for setting in settings
                ],
            }
            json_dump(args.output_dir / "calibration.json", calibration)
            print(json.dumps(calibration["statistics"], ensure_ascii=False), flush=True)
        barrier(ctx)

        for setting in settings:
            source_stats = source_stats_for(
                setting.source,
                scale1=scale1_stats,
                target_ig=target_ig_stats,
            )
            decode_setting(
                args=args,
                config=config,
                ctx=ctx,
                rae=rae,
                cache_dir=args.cache_dir,
                output_dir=args.output_dir,
                layer_index=layer_index,
                setting=setting,
                source_variance=source_stats,
            )

        del rae
        torch.cuda.empty_cache()
        barrier(ctx)
        if ctx.rank == 0 and not args.skip_metrics:
            frame = evaluate_settings(
                args=args,
                output_dir=args.output_dir,
                settings=settings,
                stats=stats,
            )
            print(frame.to_string(index=False), flush=True)
        barrier(ctx)
    finally:
        destroy_distributed(ctx)


if __name__ == "__main__":
    main()