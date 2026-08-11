"""High-throughput ImageNet-100 SiT training on cached SD-VAE moments.

The model is loaded from the official SiT repository at a pinned source hash.
The flow path and target exactly match the official linear velocity setup:

    x_t = (1 - t) * noise + t * data
    velocity = data - noise

Only the systems layer is changed: cached posterior moments, BF16 autocast,
``torch.compile``, fused AdamW, static-graph DDP, low-sync logging, and exact
per-rank RNG checkpointing.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

try:
    from experiments.imagenet100_sit_prediction_targets import (
        LOSS_SPACES,
        PREDICTION_TARGETS,
        native_prediction_target,
        prediction_losses,
        prediction_to_velocity,
    )
except ModuleNotFoundError:
    from imagenet100_sit_prediction_targets import (
        LOSS_SPACES,
        PREDICTION_TARGETS,
        native_prediction_target,
        prediction_losses,
        prediction_to_velocity,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/imagenet100_cmc_sdvae"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/sit_s2_seed0"
)
DEFAULT_OFFICIAL_SIT_REPO = Path("/home/zhoushunyu/data/research_repos/SiT")
OFFICIAL_SIT_COMMIT = "cbde832a40b153ccc79603412409da9c9b0c568c"
OFFICIAL_MODELS_SHA256 = (
    "677cfa2cf5a7db122abd014d6d92d7ac5f39745f571773a6adcc26c7d2f33d89"
)
SD_VAE_SCALING_FACTOR = 0.18215
LATENT_SHAPE = (4, 32, 32)
MOMENT_SHAPE = (8, 32, 32)
NUM_CLASSES = 100
LEGACY_PROTOCOL = "imagenet100_sit_linear_flow_v1"
TARGET_PROTOCOL = "imagenet100_sit_single_target_linear_flow_v2"


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_dump(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def git_value(repo: Path, *arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *arguments],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def load_official_sit_module(repo: Path, verify_source: bool = True):
    models_path = repo / "models.py"
    if not models_path.is_file():
        raise FileNotFoundError(f"missing official SiT model source: {models_path}")
    source_hash = sha256_file(models_path)
    commit = git_value(repo, "rev-parse", "HEAD")
    if verify_source and source_hash != OFFICIAL_MODELS_SHA256:
        raise RuntimeError(
            "official SiT models.py does not match the audited source: "
            f"expected {OFFICIAL_MODELS_SHA256}, found {source_hash}. "
            "Use --no-verify-sit-source only for an intentional model change."
        )
    spec = importlib.util.spec_from_file_location("eqvae_official_sit_models", models_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {models_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, {"models_sha256": source_hash, "git_commit": commit}


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialize_distributed(device_argument: str) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id=device)
        return DistributedContext(dist.get_rank(), local_rank, world_size, device)
    device = torch.device(device_argument)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SiT training and benchmarking require CUDA")
    torch.cuda.set_device(device)
    return DistributedContext(0, int(device.index or 0), 1, device)


def cleanup_distributed(context: DistributedContext) -> None:
    if context.world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


def barrier(context: DistributedContext) -> None:
    if context.world_size > 1:
        dist.barrier()


def configure_runtime(seed: int, rank: int, allow_tf32: bool) -> None:
    rank_seed = int(seed) + int(rank)
    random.seed(rank_seed)
    np.random.seed(rank_seed % (2**32))
    torch.manual_seed(rank_seed)
    torch.cuda.manual_seed(rank_seed)
    torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")


class NpyMomentsDataset(Dataset):
    """Worker-safe read-only view over contiguous posterior moments."""

    def __init__(self, cache_dir: Path, split: str):
        self.moments_path = cache_dir / f"{split}_moments.npy"
        self.labels_path = cache_dir / f"{split}_labels.npy"
        if not self.moments_path.is_file() or not self.labels_path.is_file():
            raise FileNotFoundError(
                f"missing {split} cache under {cache_dir}; run "
                "experiments/prepare_imagenet100_sdvae_cache.py first"
            )
        moments = np.load(self.moments_path, mmap_mode="r", allow_pickle=False)
        labels = np.load(self.labels_path, mmap_mode="r", allow_pickle=False)
        if moments.dtype != np.float32 or tuple(moments.shape[1:]) != MOMENT_SHAPE:
            raise ValueError(
                f"unexpected {split} moments shape/dtype: {moments.shape}/{moments.dtype}"
            )
        if labels.ndim != 1 or len(labels) != len(moments):
            raise ValueError(f"invalid {split} label array")
        if labels.min(initial=0) < 0 or labels.max(initial=0) >= NUM_CLASSES:
            raise ValueError(f"out-of-range labels in {self.labels_path}")
        self.length = len(moments)
        self._moments: np.ndarray | None = None
        self._labels: np.ndarray | None = None

    def __len__(self) -> int:
        return self.length

    def _open(self) -> None:
        if self._moments is None:
            # Copy-on-write mmap is writable from NumPy's perspective, avoiding a
            # torch.from_numpy warning; worker writes are never performed.
            self._moments = np.load(
                self.moments_path, mmap_mode="c", allow_pickle=False
            )
            self._labels = np.load(self.labels_path, mmap_mode="r", allow_pickle=False)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        self._open()
        assert self._moments is not None and self._labels is not None
        return torch.from_numpy(self._moments[index]), int(self._labels[index])

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_moments"] = None
        state["_labels"] = None
        return state


def sample_sdvae_posterior(
    moments: torch.Tensor,
    posterior_noise: torch.Tensor,
    scaling_factor: float = SD_VAE_SCALING_FACTOR,
) -> torch.Tensor:
    if moments.ndim != 4 or tuple(moments.shape[1:]) != MOMENT_SHAPE:
        raise ValueError(f"expected moments [B,8,32,32], found {tuple(moments.shape)}")
    mean, std = moments.chunk(2, dim=1)
    if posterior_noise.shape != mean.shape:
        raise ValueError("posterior noise does not match the cached posterior mean")
    return (mean + std * posterior_noise) * float(scaling_factor)


def linear_flow_state_target(
    data: torch.Tensor,
    noise: torch.Tensor,
    time_value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Official SiT linear interpolant, oriented from noise (t=0) to data (t=1)."""
    if data.shape != noise.shape or time_value.shape != (data.shape[0],):
        raise ValueError("incompatible data, noise, or time shapes")
    time_image = time_value.reshape(-1, *([1] * (data.ndim - 1)))
    state = (1.0 - time_image) * noise + time_image * data
    target = data - noise
    return state, target


class ModelEMA:
    def __init__(self, model: nn.Module):
        self.module = copy.deepcopy(model).eval()
        self.names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        source = dict(model.named_parameters())
        destination = dict(self.module.named_parameters())
        self.source_parameters = [source[name] for name in self.names]
        self.ema_parameters = [destination[name] for name in self.names]
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, decay: float) -> None:
        torch._foreach_mul_(self.ema_parameters, float(decay))
        torch._foreach_add_(
            self.ema_parameters,
            self.source_parameters,
            alpha=1.0 - float(decay),
        )

    def load_state_dict(self, state_dict: dict) -> None:
        self.module.load_state_dict(state_dict)

    def state_dict(self) -> dict:
        return self.module.state_dict()


def autocast_context(precision: str):
    if precision == "fp32":
        return nullcontext()
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    raise ValueError(f"unsupported precision: {precision}")


def create_loader(
    *,
    cache_dir: Path,
    split: str,
    local_batch_size: int,
    context: DistributedContext,
    seed: int,
    shuffle: bool,
    num_workers: int,
    prefetch_factor: int,
    drop_last: bool,
) -> tuple[DataLoader, DistributedSampler]:
    dataset = NpyMomentsDataset(cache_dir, split)
    sampler = DistributedSampler(
        dataset,
        num_replicas=context.world_size,
        rank=context.rank,
        shuffle=shuffle,
        seed=int(seed),
        drop_last=drop_last,
    )
    kwargs = {
        "dataset": dataset,
        "batch_size": int(local_batch_size),
        "sampler": sampler,
        "shuffle": False,
        "num_workers": int(num_workers),
        "pin_memory": True,
        "drop_last": drop_last,
        "persistent_workers": int(num_workers) > 0,
        "generator": torch.Generator().manual_seed(int(seed) + 91_337 + context.rank),
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(**kwargs), sampler


def infinite_train_batches(
    loader: DataLoader,
    sampler: DistributedSampler,
    start_step: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    batches_per_epoch = len(loader)
    if batches_per_epoch < 1:
        raise ValueError("training loader has no complete batch")
    epoch = int(start_step) // batches_per_epoch
    skip_batches = int(start_step) % batches_per_epoch
    while True:
        sampler.set_epoch(epoch)
        for batch_index, batch in enumerate(loader):
            if batch_index < skip_batches:
                continue
            yield batch
        epoch += 1
        skip_batches = 0


def capture_rng_state(device: torch.device) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device),
    }


def restore_rng_state(state: dict, device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # Loading the full checkpoint with map_location=cuda also moves these ByteTensors.
    # Both RNG APIs require their serialized state tensor on CPU.
    torch.set_rng_state(state["torch_cpu"].cpu())
    torch.cuda.set_rng_state(state["torch_cuda"].cpu(), device)


def gather_rng_states(context: DistributedContext) -> list[dict]:
    local_state = capture_rng_state(context.device)
    if context.world_size == 1:
        return [local_state]
    gathered: list[dict | None] = [None] * context.world_size
    dist.all_gather_object(gathered, local_state)
    assert all(state is not None for state in gathered)
    return [state for state in gathered if state is not None]


@dataclass(frozen=True)
class TrainConfig:
    cache_dir: str
    output_dir: str
    official_sit_repo: str
    model_name: str
    prediction_target: str
    loss_space: str
    denominator_floor: float
    global_batch_size: int
    max_steps: int
    learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    cfg_dropout: float
    ema_decay: float
    precision: str
    compile: bool
    compile_mode: str
    allow_tf32: bool
    num_workers: int
    prefetch_factor: int
    log_every: int
    validation_every: int
    validation_batches: int
    save_every: int
    seed: int


def latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = sorted((output_dir / "checkpoints").glob("step_*.pt"))
    return checkpoints[-1] if checkpoints else None


def resolve_resume(value: str, output_dir: Path) -> Path | None:
    if value.lower() in {"none", "false", "no"}:
        return None
    if value.lower() == "auto":
        return latest_checkpoint(output_dir)
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {path}")
    return path


def validate_resume(stored: dict, current: TrainConfig, world_size: int) -> None:
    immutable = (
        "cache_dir",
        "official_sit_repo",
        "model_name",
        "prediction_target",
        "loss_space",
        "denominator_floor",
        "global_batch_size",
        "learning_rate",
        "weight_decay",
        "beta1",
        "beta2",
        "cfg_dropout",
        "ema_decay",
        "precision",
        "compile",
        "compile_mode",
        "allow_tf32",
        "seed",
    )
    current_values = asdict(current)
    legacy_defaults = {
        "prediction_target": "velocity",
        "loss_space": "velocity",
        "denominator_floor": 1e-3,
    }
    mismatches = [
        f"{key}: checkpoint={stored.get(key, legacy_defaults.get(key))!r}, "
        f"current={current_values[key]!r}"
        for key in immutable
        if stored.get(key, legacy_defaults.get(key)) != current_values[key]
    ]
    stored_world_size = int(stored.get("world_size", world_size))
    if stored_world_size != world_size:
        mismatches.append(
            f"world_size: checkpoint={stored_world_size}, current={world_size}"
        )
    if mismatches:
        raise ValueError("incompatible resume configuration:\n  " + "\n  ".join(mismatches))


def reduce_sum(value: torch.Tensor, context: DistributedContext) -> torch.Tensor:
    if context.world_size > 1:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def reduce_max(value: torch.Tensor, context: DistributedContext) -> torch.Tensor:
    if context.world_size > 1:
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return value


@torch.inference_mode()
def validation_loss(
    *,
    model: nn.Module,
    loader: DataLoader,
    context: DistributedContext,
    precision: str,
    batches: int,
    seed: int,
    prediction_target: str,
    loss_space: str,
    denominator_floor: float,
) -> dict[str, float]:
    generator = torch.Generator(device=context.device).manual_seed(int(seed))
    total = torch.zeros(4, device=context.device, dtype=torch.float64)
    model.eval()
    for batch_index, (moments, labels) in enumerate(loader):
        if batch_index >= batches:
            break
        moments = moments.to(context.device, dtype=torch.float32, non_blocking=True)
        labels = labels.to(context.device, dtype=torch.long, non_blocking=True)
        posterior_noise = torch.randn(
            (len(moments), *LATENT_SHAPE),
            generator=generator,
            device=context.device,
        )
        data = sample_sdvae_posterior(moments, posterior_noise)
        source_noise = torch.randn(
            data.shape, generator=generator, device=context.device
        )
        time_value = torch.rand(
            (len(data),), generator=generator, device=context.device
        )
        state, _ = linear_flow_state_target(data, source_noise, time_value)
        with autocast_context(precision):
            prediction = model(state, time_value, labels)
        native_target = native_prediction_target(
            data=data,
            noise=source_noise,
            prediction_target=prediction_target,
        )
        velocity_prediction = prediction_to_velocity(
            prediction,
            state=state,
            time_value=time_value,
            prediction_target=prediction_target,
            denominator_floor=denominator_floor,
        )
        native_losses = (
            (prediction.float() - native_target.float()).square().flatten(1).mean(1)
        )
        velocity_losses = (
            (velocity_prediction - (data.float() - source_noise.float()))
            .square()
            .flatten(1)
            .mean(1)
        )
        optimized_losses = (
            velocity_losses if loss_space == "velocity" else native_losses
        )
        total[0] += optimized_losses.double().sum()
        total[1] += native_losses.double().sum()
        total[2] += velocity_losses.double().sum()
        total[3] += len(optimized_losses)
    reduce_sum(total, context)
    if total[3].item() == 0:
        raise RuntimeError("validation loader produced no samples")
    return {
        "optimized": float((total[0] / total[3]).item()),
        "native": float((total[1] / total[3]).item()),
        "velocity": float((total[2] / total[3]).item()),
    }


def protocol_for_config(config: TrainConfig) -> str:
    if config.prediction_target == "velocity" and config.loss_space == "velocity":
        return LEGACY_PROTOCOL
    return TARGET_PROTOCOL


def build_run_metadata(
    *,
    config: TrainConfig,
    context: DistributedContext,
    source_metadata: dict,
    model: nn.Module,
    cache_manifest: dict,
) -> dict:
    import timm

    return {
        "protocol": protocol_for_config(config),
        "config": asdict(config),
        "world_size": context.world_size,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "official_sit": source_metadata,
        "objective": {
            "path": "x_t=(1-t)*noise+t*data",
            "prediction_target": config.prediction_target,
            "loss_space": config.loss_space,
            "denominator_floor": config.denominator_floor,
            "velocity_target": "data-noise",
            "time_distribution": "Uniform[0,1)",
        },
        "latent": {
            "posterior": "mean+std*N(0,I)",
            "scaling_factor": SD_VAE_SCALING_FACTOR,
            "shape": list(LATENT_SHAPE),
        },
        "data_manifest": cache_manifest,
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "timm": timm.__version__,
            "gpus": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
            "repo_git_commit": git_value(REPO_ROOT, "rev-parse", "HEAD"),
        },
    }


def train(args: argparse.Namespace) -> None:
    context = initialize_distributed(args.device)
    try:
        if args.global_batch_size % context.world_size:
            raise ValueError("--global-batch-size must be divisible by world size")
        local_batch_size = args.global_batch_size // context.world_size
        config = TrainConfig(
            cache_dir=str(args.cache_dir.expanduser().resolve()),
            output_dir=str(args.output_dir.expanduser().resolve()),
            official_sit_repo=str(args.official_sit_repo.expanduser().resolve()),
            model_name=args.model,
            prediction_target=args.prediction_target,
            loss_space=args.loss_space,
            denominator_floor=float(args.denominator_floor),
            global_batch_size=int(args.global_batch_size),
            max_steps=int(args.max_steps),
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            beta1=float(args.beta1),
            beta2=float(args.beta2),
            cfg_dropout=float(args.cfg_dropout),
            ema_decay=float(args.ema_decay),
            precision=args.precision,
            compile=bool(args.compile),
            compile_mode=args.compile_mode,
            allow_tf32=bool(args.allow_tf32),
            num_workers=int(args.num_workers),
            prefetch_factor=int(args.prefetch_factor),
            log_every=int(args.log_every),
            validation_every=int(args.validation_every),
            validation_batches=int(args.validation_batches),
            save_every=int(args.save_every),
            seed=int(args.seed),
        )
        if min(
            config.global_batch_size,
            config.max_steps,
            config.log_every,
            config.save_every,
        ) < 1:
            raise ValueError("batch size, max steps, log interval and save interval must be positive")
        if not 0 <= config.cfg_dropout < 1:
            raise ValueError("--cfg-dropout must be in [0,1)")
        if config.denominator_floor <= 0 or config.denominator_floor >= 0.5:
            raise ValueError("--denominator-floor must be in (0,0.5)")
        configure_runtime(config.seed, context.rank, config.allow_tf32)

        cache_dir = Path(config.cache_dir)
        output_dir = Path(config.output_dir)
        cache_manifest_path = cache_dir / "manifest.json"
        cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
        cache_manifest_sha256 = sha256_file(cache_manifest_path)
        if cache_manifest.get("format") != "eqvae_imagenet100_cmc_sdvae_moments_v1":
            raise ValueError(f"unsupported data manifest: {cache_manifest_path}")

        sit_module, source_metadata = load_official_sit_module(
            Path(config.official_sit_repo), verify_source=args.verify_sit_source
        )
        if config.model_name not in sit_module.SiT_models:
            raise ValueError(f"unknown official SiT model: {config.model_name}")
        raw_model = sit_module.SiT_models[config.model_name](
            input_size=LATENT_SHAPE[-1],
            num_classes=NUM_CLASSES,
            class_dropout_prob=config.cfg_dropout,
        ).to(context.device)
        ema = ModelEMA(raw_model)
        optimizer = torch.optim.AdamW(
            raw_model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
            fused=True,
        )

        train_loader, train_sampler = create_loader(
            cache_dir=cache_dir,
            split="train",
            local_batch_size=local_batch_size,
            context=context,
            seed=config.seed,
            shuffle=True,
            num_workers=config.num_workers,
            prefetch_factor=config.prefetch_factor,
            drop_last=True,
        )
        validation_loader, validation_sampler = create_loader(
            cache_dir=cache_dir,
            split="validation",
            local_batch_size=local_batch_size,
            context=context,
            seed=config.seed + 1,
            shuffle=False,
            num_workers=max(0, min(2, config.num_workers)),
            prefetch_factor=config.prefetch_factor,
            drop_last=False,
        )
        validation_sampler.set_epoch(0)

        resume_path = resolve_resume(args.resume, output_dir)
        start_step = 0
        restored_rng: dict | None = None
        if resume_path is not None:
            checkpoint = torch.load(
                resume_path, map_location=context.device, weights_only=False
            )
            validate_resume(checkpoint["config"], config, context.world_size)
            if checkpoint.get("data_manifest_sha256") != cache_manifest_sha256:
                raise ValueError("checkpoint data manifest does not match the current cache")
            if checkpoint.get("official_sit") != source_metadata:
                raise ValueError("checkpoint official SiT source does not match this run")
            raw_model.load_state_dict(checkpoint["model"])
            ema.load_state_dict(checkpoint["ema"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_step = int(checkpoint["step"])
            rng_states = checkpoint.get("rng_states")
            if not isinstance(rng_states, list) or len(rng_states) != context.world_size:
                raise ValueError("checkpoint is missing per-rank RNG states")
            restored_rng = rng_states[context.rank]

        train_model: nn.Module = raw_model
        if config.compile:
            train_model = torch.compile(
                raw_model,
                mode=config.compile_mode,
                fullgraph=True,
                dynamic=False,
            )
        if context.world_size > 1:
            train_model = DDP(
                train_model,
                device_ids=[context.local_rank],
                output_device=context.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
                static_graph=True,
            )

        # DDP broadcasts rank-0 model parameters during construction.  Mirror
        # the official decay=0 EMA initialization so every rank validates the
        # same EMA, even though model initialization used rank-specific RNG.
        if resume_path is None:
            ema.load_state_dict(raw_model.state_dict())

        if restored_rng is not None:
            restore_rng_state(restored_rng, context.device)
        raw_model.train()
        ema.module.eval()
        batches = infinite_train_batches(train_loader, train_sampler, start_step)

        if context.is_main:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            metadata = build_run_metadata(
                config=config,
                context=context,
                source_metadata=source_metadata,
                model=raw_model,
                cache_manifest=cache_manifest,
            )
            atomic_json_dump(metadata, output_dir / "run_config.json")
            print(
                json.dumps(
                    {
                        "event": "start",
                        "step": start_step,
                        "model": config.model_name,
                        "prediction_target": config.prediction_target,
                        "loss_space": config.loss_space,
                        "parameters": metadata["parameter_count"],
                        "world_size": context.world_size,
                        "global_batch": config.global_batch_size,
                        "local_batch": local_batch_size,
                        "precision": config.precision,
                        "compile": config.compile,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        barrier(context)

        metrics_path = output_dir / "train_metrics.jsonl"
        running_losses = torch.zeros(3, device=context.device, dtype=torch.float64)
        running_steps = 0
        interval_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)

        for step in range(start_step + 1, config.max_steps + 1):
            moments, labels = next(batches)
            moments = moments.to(context.device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(context.device, dtype=torch.long, non_blocking=True)
            posterior_noise = torch.randn(
                (len(moments), *LATENT_SHAPE), device=context.device
            )
            data = sample_sdvae_posterior(moments, posterior_noise)
            source_noise = torch.randn_like(data)
            time_value = torch.rand((len(data),), device=context.device)
            state, _ = linear_flow_state_target(data, source_noise, time_value)

            with autocast_context(config.precision):
                prediction = train_model(state, time_value, labels)
            losses = prediction_losses(
                prediction,
                state=state,
                data=data,
                noise=source_noise,
                time_value=time_value,
                prediction_target=config.prediction_target,
                loss_space=config.loss_space,
                denominator_floor=config.denominator_floor,
            )
            loss = losses["optimized"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at step {step}")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            ema.update(config.ema_decay)
            running_losses += torch.stack(
                (
                    losses["optimized"].detach(),
                    losses["native"].detach(),
                    losses["velocity"].detach(),
                )
            ).double()
            running_steps += 1

            if step % config.log_every == 0 or step == config.max_steps:
                torch.cuda.synchronize(context.device)
                elapsed = time.perf_counter() - interval_started
                values = torch.tensor(
                    [*running_losses.tolist(), running_steps],
                    device=context.device,
                    dtype=torch.float64,
                )
                reduce_sum(values, context)
                elapsed_tensor = torch.tensor(elapsed, device=context.device)
                reduce_max(elapsed_tensor, context)
                memory_tensor = torch.tensor(
                    torch.cuda.max_memory_allocated(context.device) / 2**30,
                    device=context.device,
                )
                reduce_max(memory_tensor, context)
                row = {
                    "step": step,
                    "train_loss": float(values[0].item() / values[3].item()),
                    "train_native_loss": float(values[1].item() / values[3].item()),
                    "train_velocity_loss": float(values[2].item() / values[3].item()),
                    "steps_per_second": float(running_steps / elapsed_tensor.item()),
                    "images_per_second": float(
                        running_steps * config.global_batch_size / elapsed_tensor.item()
                    ),
                    "max_allocated_gb": float(memory_tensor.item()),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
                if context.is_main:
                    with metrics_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
                    print(json.dumps(row, sort_keys=True), flush=True)
                running_losses.zero_()
                running_steps = 0
                interval_started = time.perf_counter()
                torch.cuda.reset_peak_memory_stats(context.device)

            should_validate = config.validation_every > 0 and (
                step % config.validation_every == 0 or step == config.max_steps
            )
            if should_validate:
                pause_started = time.perf_counter()
                raw_value = validation_loss(
                    model=raw_model,
                    loader=validation_loader,
                    context=context,
                    precision=config.precision,
                    batches=config.validation_batches,
                    seed=config.seed + 700_000,
                    prediction_target=config.prediction_target,
                    loss_space=config.loss_space,
                    denominator_floor=config.denominator_floor,
                )
                ema_value = validation_loss(
                    model=ema.module,
                    loader=validation_loader,
                    context=context,
                    precision=config.precision,
                    batches=config.validation_batches,
                    seed=config.seed + 700_000,
                    prediction_target=config.prediction_target,
                    loss_space=config.loss_space,
                    denominator_floor=config.denominator_floor,
                )
                raw_model.train()
                if context.is_main:
                    row = {
                        "step": step,
                        "raw_validation_loss": raw_value["optimized"],
                        "ema_validation_loss": ema_value["optimized"],
                        "raw_validation_native_loss": raw_value["native"],
                        "ema_validation_native_loss": ema_value["native"],
                        "raw_validation_velocity_loss": raw_value["velocity"],
                        "ema_validation_velocity_loss": ema_value["velocity"],
                    }
                    with metrics_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
                    print(json.dumps(row, sort_keys=True), flush=True)
                interval_started += time.perf_counter() - pause_started

            should_save = step % config.save_every == 0 or step == config.max_steps
            if should_save:
                pause_started = time.perf_counter()
                rng_states = gather_rng_states(context)
                if context.is_main:
                    checkpoint_path = output_dir / "checkpoints" / f"step_{step:08d}.pt"
                    atomic_torch_save(
                        {
                            "protocol": protocol_for_config(config),
                            "step": step,
                            "model": raw_model.state_dict(),
                            "ema": ema.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "rng_states": rng_states,
                            "config": {**asdict(config), "world_size": context.world_size},
                            "official_sit": source_metadata,
                            "data_manifest_sha256": cache_manifest_sha256,
                        },
                        checkpoint_path,
                    )
                    print(json.dumps({"event": "checkpoint", "path": str(checkpoint_path)}), flush=True)
                barrier(context)
                interval_started += time.perf_counter() - pause_started
    finally:
        cleanup_distributed(context)


def benchmark(args: argparse.Namespace) -> None:
    context = initialize_distributed(args.device)
    try:
        configure_runtime(args.seed, context.rank, args.allow_tf32)
        sit_module, source_metadata = load_official_sit_module(
            args.official_sit_repo.expanduser().resolve(),
            verify_source=args.verify_sit_source,
        )
        raw_model = sit_module.SiT_models[args.model](
            input_size=LATENT_SHAPE[-1],
            num_classes=NUM_CLASSES,
            class_dropout_prob=args.cfg_dropout,
        ).to(context.device)
        ema = ModelEMA(raw_model)
        optimizer = torch.optim.AdamW(
            raw_model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.0,
            fused=True,
        )
        model: nn.Module = raw_model
        if args.compile:
            model = torch.compile(
                raw_model,
                mode=args.compile_mode,
                fullgraph=True,
                dynamic=False,
            )
        if context.world_size > 1:
            model = DDP(
                model,
                device_ids=[context.local_rank],
                output_device=context.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
                static_graph=True,
            )
        model.train()
        batch_size = int(args.local_batch_size)
        moments = torch.randn((batch_size, *MOMENT_SHAPE), device=context.device)
        moments[:, 4:].abs_().mul_(0.001)
        labels = torch.randint(NUM_CLASSES, (batch_size,), device=context.device)
        total_steps = args.benchmark_warmup + args.benchmark_steps
        started = None
        torch.cuda.reset_peak_memory_stats(context.device)
        for index in range(total_steps):
            data = sample_sdvae_posterior(
                moments, torch.randn((batch_size, *LATENT_SHAPE), device=context.device)
            )
            source_noise = torch.randn_like(data)
            time_value = torch.rand((batch_size,), device=context.device)
            state, _ = linear_flow_state_target(data, source_noise, time_value)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(args.precision):
                prediction = model(state, time_value, labels)
            losses = prediction_losses(
                prediction,
                state=state,
                data=data,
                noise=source_noise,
                time_value=time_value,
                prediction_target=args.prediction_target,
                loss_space=args.loss_space,
                denominator_floor=args.denominator_floor,
            )
            losses["optimized"].backward()
            optimizer.step()
            ema.update(args.ema_decay)
            if index + 1 == args.benchmark_warmup:
                barrier(context)
                torch.cuda.synchronize(context.device)
                started = time.perf_counter()
        barrier(context)
        torch.cuda.synchronize(context.device)
        elapsed = time.perf_counter() - float(started)
        elapsed_tensor = torch.tensor(elapsed, device=context.device)
        reduce_max(elapsed_tensor, context)
        memory = torch.tensor(
            torch.cuda.max_memory_allocated(context.device) / 2**30,
            device=context.device,
        )
        reduce_max(memory, context)
        if context.is_main:
            print(
                json.dumps(
                    {
                        "model": args.model,
                        "prediction_target": args.prediction_target,
                        "loss_space": args.loss_space,
                        "official_sit": source_metadata,
                        "world_size": context.world_size,
                        "local_batch_size": batch_size,
                        "global_batch_size": batch_size * context.world_size,
                        "precision": args.precision,
                        "compile": args.compile,
                        "compile_mode": args.compile_mode,
                        "steps_per_second": args.benchmark_steps / elapsed_tensor.item(),
                        "images_per_second": (
                            args.benchmark_steps
                            * batch_size
                            * context.world_size
                            / elapsed_tensor.item()
                        ),
                        "max_allocated_gb": memory.item(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        cleanup_distributed(context)


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--official-sit-repo", type=Path, default=DEFAULT_OFFICIAL_SIT_REPO)
    parser.add_argument("--model", default="SiT-S/2")
    parser.add_argument(
        "--prediction-target", choices=PREDICTION_TARGETS, default="velocity"
    )
    parser.add_argument("--loss-space", choices=LOSS_SPACES, default="velocity")
    parser.add_argument("--denominator-floor", type=float, default=1e-3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--cfg-dropout", type=float, default=0.1)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument(
        "--allow-tf32", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--verify-sit-source", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train on cached ImageNet-100 moments.")
    add_shared_arguments(train_parser)
    train_parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    train_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    train_parser.add_argument("--global-batch-size", type=int, default=256)
    train_parser.add_argument("--max-steps", type=int, default=100_000)
    train_parser.add_argument("--weight-decay", type=float, default=0.0)
    train_parser.add_argument("--beta1", type=float, default=0.9)
    train_parser.add_argument("--beta2", type=float, default=0.999)
    train_parser.add_argument("--num-workers", type=int, default=4)
    train_parser.add_argument("--prefetch-factor", type=int, default=4)
    train_parser.add_argument("--log-every", type=int, default=50)
    train_parser.add_argument("--validation-every", type=int, default=5_000)
    train_parser.add_argument("--validation-batches", type=int, default=8)
    train_parser.add_argument("--save-every", type=int, default=10_000)
    train_parser.add_argument(
        "--resume",
        default="auto",
        help="auto, none, or an explicit checkpoint path.",
    )

    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Benchmark the exact optimizer/EMA training step on synthetic latents."
    )
    add_shared_arguments(benchmark_parser)
    benchmark_parser.add_argument("--local-batch-size", type=int, default=64)
    benchmark_parser.add_argument("--benchmark-warmup", type=int, default=10)
    benchmark_parser.add_argument("--benchmark-steps", type=int, default=30)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "benchmark":
        benchmark(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
