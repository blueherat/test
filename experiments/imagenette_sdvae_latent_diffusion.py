"""Train a conventional class-conditional latent diffusion on Imagenette.

This is a compact Stable-Diffusion-style baseline rather than a mechanism toy:

* frozen ``stabilityai/sd-vae-ft-mse`` tokenizer;
* online 128x128 image augmentation and posterior sampling;
* 4x16x16 latent diffusion with an 83M-parameter U-Net;
* Stable Diffusion v1 scaled-linear noise schedule and epsilon prediction;
* classifier-free class conditioning, EMA, AdamW, warmup, and DDIM sampling.

The training entry point supports both one GPU and ``torchrun`` DDP. All model,
VAE, and optimizer tensors are float32; TF32 is an explicit recorded option.
Artifacts are stored outside the repository by default.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image


DEFAULT_DATA_ROOT = Path("/data/shared/imagenette2-320")
DEFAULT_OUTPUT_ROOT = Path.home() / "data/eqvae/imagenette_sdvae_ldm128"
DEFAULT_VAE = "stabilityai/sd-vae-ft-mse"
IMAGE_SIZE = 128
LATENT_CHANNELS = 4
LATENT_SIZE = 16
NUM_CLASSES = 10
NULL_CLASS = NUM_CLASSES
SD_BETA_START = 0.00085
SD_BETA_END = 0.012
SD_BETA_SCHEDULE = "scaled_linear"
SD_TIMESTEPS = 1000
FORMAL_CHANNELS = (128, 256, 384, 512)


def atomic_torch_save(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_json_dump(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def configure_runtime(seed: int, allow_tf32: bool) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")


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
        rank = dist.get_rank()
    else:
        rank = 0
        local_rank = 0
        device = torch.device(device_argument)
    if device.type != "cuda":
        raise RuntimeError("this training baseline requires CUDA")
    if world_size == 1:
        torch.cuda.set_device(device)
    return DistributedContext(rank, local_rank, world_size, device)


def cleanup_distributed(context: DistributedContext) -> None:
    if context.world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


def barrier(context: DistributedContext) -> None:
    if context.world_size > 1:
        dist.barrier()


def all_reduce_sum(value: torch.Tensor, context: DistributedContext) -> torch.Tensor:
    if context.world_size > 1:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def ddp_uses_static_graph(gradient_accumulation: int) -> bool:
    """Avoid a PyTorch reducer assertion for compiled DDP ``no_sync`` passes."""
    return int(gradient_accumulation) == 1


def latent_size_for_image_size(image_size: int) -> int:
    image_size = int(image_size)
    if image_size < 32 or image_size % 8 != 0:
        raise ValueError("image size must be at least 32 and divisible by the SD-VAE factor 8")
    return image_size // 8


def imagenette_transforms(
    image_size: int = IMAGE_SIZE,
) -> tuple[transforms.Compose, transforms.Compose]:
    image_size = int(image_size)
    latent_size_for_image_size(image_size)
    normalize = transforms.Lambda(lambda image: image.mul(2.0).sub(1.0))
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.70, 1.0),
                ratio=(0.85, 1.15),
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    validation_transform = transforms.Compose(
        [
            transforms.Resize(round(image_size * 1.125), antialias=True),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, validation_transform


def load_datasets(
    data_root: str | Path, image_size: int = IMAGE_SIZE
) -> tuple[ImageFolder, ImageFolder]:
    root = Path(data_root).expanduser()
    train_transform, validation_transform = imagenette_transforms(image_size)
    train = ImageFolder(str(root / "train"), transform=train_transform)
    validation = ImageFolder(str(root / "val"), transform=validation_transform)
    if train.class_to_idx != validation.class_to_idx:
        raise RuntimeError("train and validation class mappings differ")
    if len(train.classes) != NUM_CLASSES:
        raise RuntimeError(f"expected {NUM_CLASSES} Imagenette classes, got {len(train.classes)}")
    return train, validation


def fixed_validation_subset(dataset: ImageFolder, count: int, seed: int) -> Subset:
    count = min(int(count), len(dataset))
    generator = torch.Generator().manual_seed(int(seed))
    indices = torch.randperm(len(dataset), generator=generator)[:count].tolist()
    return Subset(dataset, indices)


def stable_diffusion_noise_scheduler():
    from diffusers import DDPMScheduler

    return DDPMScheduler(
        num_train_timesteps=SD_TIMESTEPS,
        beta_start=SD_BETA_START,
        beta_end=SD_BETA_END,
        beta_schedule=SD_BETA_SCHEDULE,
        prediction_type="epsilon",
        variance_type="fixed_small",
        clip_sample=False,
    )


def block_types(channels: Sequence[int]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if len(channels) < 3:
        raise ValueError("the U-Net needs at least three resolution levels")
    down = ["DownBlock2D"] * len(channels)
    down[-2] = "AttnDownBlock2D"
    up = ["UpBlock2D"] * len(channels)
    up[1] = "AttnUpBlock2D"
    return tuple(down), tuple(up)


def build_unet(
    channels: Sequence[int] = FORMAL_CHANNELS,
    sample_size: int = LATENT_SIZE,
):
    from diffusers import UNet2DModel

    channels = tuple(int(channel) for channel in channels)
    down, up = block_types(channels)
    return UNet2DModel(
        sample_size=int(sample_size),
        in_channels=LATENT_CHANNELS,
        out_channels=LATENT_CHANNELS,
        layers_per_block=2,
        block_out_channels=channels,
        down_block_types=down,
        up_block_types=up,
        num_class_embeds=NUM_CLASSES + 1,
        attention_head_dim=8,
        norm_num_groups=32,
        dropout=0.0,
        add_attention=True,
    )


def apply_condition_dropout(
    labels: torch.Tensor, probability: float, generator: torch.Generator | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    if probability <= 0.0:
        mask = torch.zeros_like(labels, dtype=torch.bool)
        return labels, mask
    mask = torch.rand(labels.shape, device=labels.device, generator=generator) < float(probability)
    return labels.masked_fill(mask, NULL_CLASS), mask


class EMA:
    def __init__(self, module: nn.Module):
        self.module = copy.deepcopy(module).eval()
        self.parameters = list(self.module.parameters())
        self.buffers = list(self.module.buffers())
        for parameter in self.parameters:
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, source: nn.Module, decay: float, step: int) -> None:
        effective_decay = min(float(decay), (1.0 + float(step)) / (10.0 + float(step)))
        source_parameters = [parameter.detach() for parameter in source.parameters()]
        torch._foreach_lerp_(self.parameters, source_parameters, 1.0 - effective_decay)
        for target, source_buffer in zip(self.buffers, source.buffers()):
            target.copy_(source_buffer)


@dataclass(frozen=True)
class TrainConfig:
    data_root: str
    output_dir: str
    vae_repo: str
    seed: int
    image_size: int
    channels: tuple[int, ...]
    batch_size: int
    gradient_accumulation: int
    steps: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    class_dropout: float
    ema_decay: float
    gradient_clip: float
    num_workers: int
    validation_count: int
    validation_every: int
    log_every: int
    save_every: int
    extra_save_steps: tuple[int, ...]
    compile: bool
    compile_mode: str
    allow_tf32: bool
    world_size: int

    @property
    def latent_size(self) -> int:
        return latent_size_for_image_size(self.image_size)


def parse_int_tuple(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("integer list cannot be empty")
    return values


def latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = sorted(output_dir.glob("checkpoint_step*.pt"))
    return checkpoints[-1] if checkpoints else None


def resume_batch_offset(
    completed_steps: int, gradient_accumulation: int, loader_length: int
) -> int:
    if completed_steps < 0 or gradient_accumulation < 1 or loader_length < 1:
        raise ValueError("invalid resume offset arguments")
    completed_microbatches = int(completed_steps) * int(gradient_accumulation)
    return completed_microbatches % int(loader_length)


def validate_resume_config(stored: dict, current: TrainConfig) -> None:
    current_dict = asdict(current)
    stored = dict(stored)
    # Checkpoints produced before the resolution sweep were fixed at 128 px.
    stored.setdefault("image_size", IMAGE_SIZE)
    allowed_to_change = {"steps", "compile", "compile_mode"}
    mismatches = {
        key: (stored.get(key), value)
        for key, value in current_dict.items()
        if key not in allowed_to_change and stored.get(key) != value
    }
    if mismatches:
        raise ValueError(f"resume configuration mismatch: {mismatches}")


def make_loaders(
    config: TrainConfig,
    context: DistributedContext,
) -> tuple[DataLoader, DistributedSampler, DataLoader, DistributedSampler, dict]:
    train_dataset, validation_dataset = load_datasets(config.data_root, config.image_size)
    validation_subset = fixed_validation_subset(
        validation_dataset, config.validation_count, config.seed + 30_001
    )
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=context.world_size,
        rank=context.rank,
        shuffle=True,
        seed=config.seed + 31_001,
        drop_last=True,
    )
    validation_sampler = DistributedSampler(
        validation_subset,
        num_replicas=context.world_size,
        rank=context.rank,
        shuffle=False,
        drop_last=False,
    )
    common = {
        "num_workers": config.num_workers,
        "pin_memory": True,
        "persistent_workers": config.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        drop_last=True,
        **common,
    )
    validation_loader = DataLoader(
        validation_subset,
        batch_size=config.batch_size,
        sampler=validation_sampler,
        drop_last=False,
        **common,
    )
    metadata = {
        "train_count": len(train_dataset),
        "validation_total_count": len(validation_dataset),
        "validation_subset_count": len(validation_subset),
        "class_to_idx": train_dataset.class_to_idx,
    }
    return train_loader, train_sampler, validation_loader, validation_sampler, metadata


@torch.no_grad()
def encode_images(vae: nn.Module, images: torch.Tensor, scaling_factor: float) -> torch.Tensor:
    posterior = vae.encode(images).latent_dist
    return posterior.sample().mul(scaling_factor)


@torch.no_grad()
def encode_validation_images(
    vae: nn.Module, images: torch.Tensor, scaling_factor: float
) -> torch.Tensor:
    posterior = vae.encode(images).latent_dist
    return posterior.mode().mul(scaling_factor)


def unwrap_prediction(output) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    return output.sample


@torch.no_grad()
def validation_loss(
    *,
    model: nn.Module,
    vae: nn.Module,
    loader: DataLoader,
    noise_scheduler,
    scaling_factor: float,
    context: DistributedContext,
    seed: int,
) -> float:
    model.eval()
    generator = torch.Generator(device=context.device).manual_seed(
        int(seed) + context.rank * 100_003
    )
    squared_error = torch.zeros((), device=context.device, dtype=torch.float64)
    element_count = torch.zeros((), device=context.device, dtype=torch.float64)
    for images, labels in loader:
        images = images.to(
            device=context.device,
            dtype=torch.float32,
            non_blocking=True,
            memory_format=torch.channels_last,
        )
        labels = labels.to(device=context.device, dtype=torch.long, non_blocking=True)
        latents = encode_validation_images(vae, images, scaling_factor).contiguous(
            memory_format=torch.channels_last
        )
        noise = torch.randn(
            latents.shape,
            device=context.device,
            dtype=torch.float32,
            generator=generator,
        ).contiguous(memory_format=torch.channels_last)
        timesteps = torch.randint(
            0,
            SD_TIMESTEPS,
            (len(latents),),
            device=context.device,
            generator=generator,
        ).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        prediction = unwrap_prediction(
            model(noisy_latents, timesteps, class_labels=labels, return_dict=False)
        )
        squared_error += (prediction.double() - noise.double()).square().sum()
        element_count += noise.numel()
    all_reduce_sum(squared_error, context)
    all_reduce_sum(element_count, context)
    model.train()
    return float((squared_error / element_count.clamp_min(1.0)).item())


def checkpoint_payload(
    *,
    step: int,
    epoch: int,
    model: nn.Module,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: TrainConfig,
) -> dict:
    return {
        "protocol": "imagenette_sdvae_ldm128_v1",
        "step": int(step),
        "epoch": int(epoch),
        "model": model.state_dict(),
        "ema": ema.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "config": asdict(config),
        "noise_scheduler": dict(stable_diffusion_noise_scheduler().config),
    }


def build_train_config(args: argparse.Namespace, context: DistributedContext) -> TrainConfig:
    return TrainConfig(
        data_root=str(Path(args.data_root).expanduser()),
        output_dir=str(Path(args.output_dir).expanduser()),
        vae_repo=str(args.vae_repo),
        seed=int(args.seed),
        image_size=int(args.image_size),
        channels=parse_int_tuple(args.channels),
        batch_size=int(args.batch_size),
        gradient_accumulation=int(args.gradient_accumulation),
        steps=int(args.steps),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        warmup_steps=int(args.warmup_steps),
        class_dropout=float(args.class_dropout),
        ema_decay=float(args.ema_decay),
        gradient_clip=float(args.gradient_clip),
        num_workers=int(args.num_workers),
        validation_count=int(args.validation_count),
        validation_every=int(args.validation_every),
        log_every=int(args.log_every),
        save_every=int(args.save_every),
        extra_save_steps=tuple(
            sorted(value for value in parse_int_tuple(args.extra_save_steps) if value > 0)
        ),
        compile=bool(args.compile),
        compile_mode=str(args.compile_mode),
        allow_tf32=bool(args.allow_tf32),
        world_size=context.world_size,
    )


def train(args: argparse.Namespace) -> None:
    context = initialize_distributed(args.device)
    try:
        configure_runtime(args.seed + context.rank, args.allow_tf32)
        config = build_train_config(args, context)
        if min(config.batch_size, config.gradient_accumulation, config.steps) < 1:
            raise ValueError("batch size, accumulation, and steps must be positive")
        if not 0.0 <= config.class_dropout < 1.0:
            raise ValueError("class dropout must lie in [0, 1)")

        output_dir = Path(config.output_dir)
        if context.is_main:
            output_dir.mkdir(parents=True, exist_ok=True)
        barrier(context)

        train_loader, train_sampler, validation_loader, validation_sampler, data_metadata = (
            make_loaders(config, context)
        )
        if len(train_loader) == 0:
            raise RuntimeError("training loader is empty; reduce the per-GPU batch size")

        from diffusers.models import AutoencoderKL

        vae = AutoencoderKL.from_pretrained(config.vae_repo, local_files_only=True)
        vae = vae.to(
            device=context.device, dtype=torch.float32, memory_format=torch.channels_last
        ).eval()
        vae.requires_grad_(False)
        scaling_factor = float(getattr(vae.config, "scaling_factor", 0.18215))

        torch.manual_seed(config.seed + 101)
        raw_model = build_unet(config.channels, config.latent_size).to(
            device=context.device, dtype=torch.float32, memory_format=torch.channels_last
        )
        ema = EMA(raw_model) if context.is_main else None
        optimizer = torch.optim.AdamW(
            raw_model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=config.weight_decay,
            fused=True,
        )

        def lr_lambda(step_index: int) -> float:
            if config.warmup_steps <= 0:
                return 1.0
            return min(1.0, float(step_index + 1) / float(config.warmup_steps))

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        start_step = 0
        start_epoch = 0
        checkpoint = latest_checkpoint(output_dir) if args.resume else None
        if checkpoint is not None:
            restored = torch.load(checkpoint, map_location=context.device, weights_only=False)
            validate_resume_config(restored["config"], config)
            raw_model.load_state_dict(restored["model"])
            if context.is_main:
                assert ema is not None
                ema.module.load_state_dict(restored["ema"])
            optimizer.load_state_dict(restored["optimizer"])
            lr_scheduler.load_state_dict(restored["lr_scheduler"])
            start_step = int(restored["step"])
            completed_microbatches = start_step * config.gradient_accumulation
            start_epoch = completed_microbatches // len(train_loader)
            if context.is_main:
                print(f"[resume] {checkpoint} at step {start_step}", flush=True)
            del restored

        forward_model: nn.Module = raw_model
        if config.compile:
            if context.is_main:
                print(f"[compile] mode={config.compile_mode}", flush=True)
            forward_model = torch.compile(
                raw_model, mode=config.compile_mode, dynamic=False, fullgraph=False
            )
        if context.world_size > 1:
            forward_model = DistributedDataParallel(
                forward_model,
                device_ids=[context.local_rank],
                output_device=context.local_rank,
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
                static_graph=ddp_uses_static_graph(config.gradient_accumulation),
            )

        # Model initialization must be identical across ranks, but stochastic
        # training draws should not be. Re-seed only after DDP has synchronized
        # the shared initial state.
        torch.manual_seed(config.seed + 10_001 + context.rank)
        torch.cuda.manual_seed(config.seed + 10_001 + context.rank)

        noise_scheduler = stable_diffusion_noise_scheduler()
        if context.is_main:
            metadata = {
                "protocol": "imagenette_sdvae_ldm128_v1",
                "config": asdict(config),
                "parameter_count": sum(parameter.numel() for parameter in raw_model.parameters()),
                "vae_parameter_count": sum(parameter.numel() for parameter in vae.parameters()),
                "vae_scaling_factor": scaling_factor,
                "image_size": config.image_size,
                "latent_shape": [LATENT_CHANNELS, config.latent_size, config.latent_size],
                "dtype": "float32",
                "data": data_metadata,
                "noise_scheduler": dict(noise_scheduler.config),
                "global_batch_size": (
                    config.batch_size * context.world_size * config.gradient_accumulation
                ),
                "git_commit": os.popen("git rev-parse HEAD 2>/dev/null").read().strip(),
            }
            atomic_json_dump(metadata, output_dir / "run_config.json")

        step = start_step
        epoch = start_epoch
        first_epoch_batch_offset = resume_batch_offset(
            start_step, config.gradient_accumulation, len(train_loader)
        )
        if context.is_main and start_step:
            print(
                f"[resume] skip {first_epoch_batch_offset} consumed batches "
                f"in epoch {start_epoch}",
                flush=True,
            )
        running_loss = 0.0
        running_microbatches = 0
        running_images = 0
        micro_step = start_step * config.gradient_accumulation
        interval_started = time.perf_counter()
        log_path = output_dir / "train_metrics.jsonl"
        raw_model.train()
        optimizer.zero_grad(set_to_none=True)
        while step < config.steps:
            train_sampler.set_epoch(epoch)
            for batch_index, (images, labels) in enumerate(train_loader):
                if epoch == start_epoch and batch_index < first_epoch_batch_offset:
                    continue
                images = images.to(
                    device=context.device,
                    dtype=torch.float32,
                    non_blocking=True,
                    memory_format=torch.channels_last,
                )
                labels = labels.to(device=context.device, dtype=torch.long, non_blocking=True)
                with torch.no_grad():
                    latents = encode_images(vae, images, scaling_factor).contiguous(
                        memory_format=torch.channels_last
                    )
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, SD_TIMESTEPS, (len(latents),), device=context.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                dropped_labels, _ = apply_condition_dropout(labels, config.class_dropout)
                micro_step += 1
                is_update = micro_step % config.gradient_accumulation == 0
                synchronization = (
                    forward_model.no_sync()
                    if isinstance(forward_model, DistributedDataParallel) and not is_update
                    else nullcontext()
                )
                with synchronization:
                    prediction = unwrap_prediction(
                        forward_model(
                            noisy_latents,
                            timesteps,
                            class_labels=dropped_labels,
                            return_dict=False,
                        )
                    )
                    loss = F.mse_loss(prediction.float(), noise.float())
                    (loss / config.gradient_accumulation).backward()
                running_loss += float(loss.detach())
                running_microbatches += 1
                running_images += len(images) * context.world_size

                if not is_update:
                    continue

                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    raw_model.parameters(), config.gradient_clip
                )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if context.is_main:
                    assert ema is not None
                    ema.update(raw_model, config.ema_decay, step)

                if step % config.log_every == 0 or step == config.steps:
                    torch.cuda.synchronize(context.device)
                    elapsed = max(time.perf_counter() - interval_started, 1e-6)
                    loss_tensor = torch.tensor(
                        [running_loss, float(running_microbatches)],
                        device=context.device,
                        dtype=torch.float64,
                    )
                    all_reduce_sum(loss_tensor, context)
                    if context.is_main:
                        row = {
                            "step": step,
                            "epoch": epoch,
                            "train_loss": float(loss_tensor[0] / loss_tensor[1].clamp_min(1.0)),
                            "learning_rate": float(optimizer.param_groups[0]["lr"]),
                            "gradient_norm": float(gradient_norm),
                            "global_images_per_second": running_images / elapsed,
                            "gpu_memory_allocated_gb": (
                                torch.cuda.max_memory_allocated(context.device) / 2**30
                            ),
                            "gpu_memory_reserved_gb": (
                                torch.cuda.max_memory_reserved(context.device) / 2**30
                            ),
                        }
                        with log_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(row, sort_keys=True) + "\n")
                        print(json.dumps(row, sort_keys=True), flush=True)
                    running_loss = 0.0
                    running_microbatches = 0
                    running_images = 0
                    interval_started = time.perf_counter()
                    torch.cuda.reset_peak_memory_stats(context.device)

                if step % config.validation_every == 0 or step == config.steps:
                    maintenance_started = time.perf_counter()
                    validation_sampler.set_epoch(0)
                    value = validation_loss(
                        model=forward_model,
                        vae=vae,
                        loader=validation_loader,
                        noise_scheduler=noise_scheduler,
                        scaling_factor=scaling_factor,
                        context=context,
                        seed=config.seed + 40_001,
                    )
                    if context.is_main:
                        row = {"step": step, "epoch": epoch, "validation_loss": value}
                        with (output_dir / "validation_metrics.jsonl").open(
                            "a", encoding="utf-8"
                        ) as handle:
                            handle.write(json.dumps(row, sort_keys=True) + "\n")
                        print(json.dumps(row, sort_keys=True), flush=True)
                    interval_started += time.perf_counter() - maintenance_started

                should_save = (
                    step % config.save_every == 0
                    or step in config.extra_save_steps
                    or step == config.steps
                )
                if should_save:
                    maintenance_started = time.perf_counter()
                    barrier(context)
                    if context.is_main:
                        assert ema is not None
                        payload = checkpoint_payload(
                            step=step,
                            epoch=epoch,
                            model=raw_model,
                            ema=ema,
                            optimizer=optimizer,
                            lr_scheduler=lr_scheduler,
                            config=config,
                        )
                        checkpoint_path = output_dir / f"checkpoint_step{step:07d}.pt"
                        atomic_torch_save(payload, checkpoint_path)
                        print(f"[checkpoint] {checkpoint_path}", flush=True)
                    barrier(context)
                    interval_started += time.perf_counter() - maintenance_started

                if step >= config.steps:
                    break
            epoch += 1
    finally:
        cleanup_distributed(context)


@torch.inference_mode()
def sample_batch(
    *,
    model: nn.Module,
    scheduler,
    labels: torch.Tensor,
    latent_size: int,
    guidance_scale: float,
    generator: torch.Generator,
) -> torch.Tensor:
    device = labels.device
    latents = torch.randn(
        (len(labels), LATENT_CHANNELS, int(latent_size), int(latent_size)),
        device=device,
        dtype=torch.float32,
        generator=generator,
    ).contiguous(memory_format=torch.channels_last)
    latents = latents * scheduler.init_noise_sigma
    for timestep in scheduler.timesteps:
        timestep_batch = timestep.expand(len(labels))
        if float(guidance_scale) == 1.0:
            prediction = unwrap_prediction(
                model(latents, timestep_batch, class_labels=labels, return_dict=False)
            )
        else:
            model_input = torch.cat((latents, latents), dim=0)
            timestep_input = timestep.expand(2 * len(labels))
            class_input = torch.cat((torch.full_like(labels, NULL_CLASS), labels), dim=0)
            unconditional, conditional = unwrap_prediction(
                model(
                    model_input,
                    timestep_input,
                    class_labels=class_input,
                    return_dict=False,
                )
            ).chunk(2)
            prediction = unconditional + float(guidance_scale) * (
                conditional - unconditional
            )
        latents = scheduler.step(prediction, timestep, latents, eta=0.0).prev_sample
    return latents


def sample_shard_bounds(total: int, rank: int, world_size: int) -> tuple[int, int]:
    if total < 0 or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("invalid distributed sample shard arguments")
    start = int(total) * int(rank) // int(world_size)
    end = int(total) * (int(rank) + 1) // int(world_size)
    return start, end


@torch.inference_mode()
def sample(args: argparse.Namespace) -> None:
    context = initialize_distributed(args.device)
    try:
        configure_runtime(args.seed + context.rank, args.allow_tf32)
        device = context.device
        checkpoint_path = Path(args.checkpoint).expanduser()
        # Sampling does not need AdamW or scheduler tensors on GPU. Loading the
        # complete training checkpoint on CPU avoids about a gigabyte of
        # needless per-rank device allocation.
        restored = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = restored["config"]
        image_size = int(config.get("image_size", IMAGE_SIZE))
        latent_size = latent_size_for_image_size(image_size)
        model = build_unet(tuple(config["channels"]), latent_size).to(
            device=device, dtype=torch.float32, memory_format=torch.channels_last
        )
        model.load_state_dict(restored["ema"] if args.use_ema else restored["model"])
        del restored
        model.eval()
        if args.compile:
            model = torch.compile(model, mode=args.compile_mode, dynamic=False, fullgraph=False)

        from diffusers import DDIMScheduler

        scheduler = DDIMScheduler(
            num_train_timesteps=SD_TIMESTEPS,
            beta_start=SD_BETA_START,
            beta_end=SD_BETA_END,
            beta_schedule=SD_BETA_SCHEDULE,
            prediction_type="epsilon",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
        scheduler.set_timesteps(args.sample_steps, device=device)

        output_dir = Path(args.output_dir).expanduser()
        if context.is_main:
            output_dir.mkdir(parents=True, exist_ok=True)
        barrier(context)
        shard_start, shard_end = sample_shard_bounds(
            int(args.sample_count), context.rank, context.world_size
        )
        shard_count = shard_end - shard_start
        generator = torch.Generator(device=device).manual_seed(
            int(args.seed) + context.rank * 1_000_003
        )
        latent_chunks = []
        labels_all = []
        remaining = shard_count
        generated = 0
        while remaining > 0:
            actual_count = min(remaining, int(args.batch_size))
            # Keep the compiled graph shape fixed for the final partial batch.
            draw_count = int(args.batch_size)
            labels = (
                torch.arange(
                    shard_start + generated,
                    shard_start + generated + draw_count,
                    device=device,
                )
                % NUM_CLASSES
            ).long()
            latents = sample_batch(
                model=model,
                scheduler=scheduler,
                labels=labels,
                latent_size=latent_size,
                guidance_scale=args.guidance_scale,
                generator=generator,
            )[:actual_count]
            labels = labels[:actual_count]
            latent_chunks.append(latents.cpu())
            labels_all.append(labels.cpu())
            remaining -= actual_count
            generated += actual_count
            print(
                f"[sample rank={context.rank}] {generated}/{shard_count}", flush=True
            )
        if shard_count:
            latents_tensor = torch.cat(latent_chunks)
            labels_tensor = torch.cat(labels_all)
        else:
            latents_tensor = torch.empty(
                (0, LATENT_CHANNELS, latent_size, latent_size), dtype=torch.float32
            )
            labels_tensor = torch.empty((0,), dtype=torch.long)

        # The compiled CFG graph and SD-VAE decoder have high but disjoint peak
        # memory requirements. Decode only after all tiny latents are on CPU and
        # the denoiser has been released; this is mathematically identical to
        # per-batch decoding and prevents their activation peaks from stacking.
        del model, latent_chunks
        gc.collect()
        torch.cuda.empty_cache()
        from diffusers.models import AutoencoderKL

        vae = AutoencoderKL.from_pretrained(config["vae_repo"], local_files_only=True)
        vae = vae.to(device=device, dtype=torch.float32, memory_format=torch.channels_last).eval()
        vae.requires_grad_(False)
        scaling_factor = float(getattr(vae.config, "scaling_factor", 0.18215))
        images = []
        for decode_start in range(0, shard_count, int(args.decode_batch_size)):
            decode_end = min(decode_start + int(args.decode_batch_size), shard_count)
            latent_batch = latents_tensor[decode_start:decode_end].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
                memory_format=torch.channels_last,
            )
            decoded = vae.decode(latent_batch / scaling_factor).sample.clamp(-1.0, 1.0)
            images.append(decoded.cpu())
            print(
                f"[decode rank={context.rank}] {decode_end}/{shard_count}", flush=True
            )
        if shard_count:
            images_tensor = torch.cat(images)
        else:
            images_tensor = torch.empty((0, 3, image_size, image_size), dtype=torch.float32)
        del vae, latents_tensor, images
        torch.cuda.empty_cache()

        shard_path = (
            output_dir / f"samples_rank{context.rank:03d}.pt"
            if context.world_size > 1
            else output_dir / "samples.pt"
        )
        atomic_torch_save(
            {"images": images_tensor, "labels": labels_tensor}, shard_path
        )
        if args.save_individual:
            image_dir = output_dir / "images"
            image_dir.mkdir(parents=True, exist_ok=True)
            for local_index, image in enumerate(images_tensor):
                save_image(
                    image.add(1.0).div(2.0),
                    image_dir / f"{shard_start + local_index:06d}.png",
                )
        del labels_all, images_tensor, labels_tensor
        barrier(context)

        if context.is_main:
            if context.world_size > 1:
                shard_payloads = [
                    torch.load(
                        output_dir / f"samples_rank{rank:03d}.pt",
                        map_location="cpu",
                        weights_only=False,
                    )
                    for rank in range(context.world_size)
                ]
                images_tensor = torch.cat([payload["images"] for payload in shard_payloads])
                labels_tensor = torch.cat([payload["labels"] for payload in shard_payloads])
                if len(images_tensor) != int(args.sample_count):
                    raise RuntimeError("distributed sample shards do not match requested count")
                atomic_torch_save(
                    {"images": images_tensor, "labels": labels_tensor},
                    output_dir / "samples.pt",
                )
                for rank in range(context.world_size):
                    (output_dir / f"samples_rank{rank:03d}.pt").unlink()
            else:
                payload = torch.load(
                    output_dir / "samples.pt", map_location="cpu", weights_only=False
                )
                images_tensor = payload["images"]
                labels_tensor = payload["labels"]
            grid_count = min(args.grid_count, len(images_tensor))
            if grid_count:
                nrow = max(1, int(math.sqrt(grid_count)))
                grid = make_grid(
                    images_tensor[:grid_count].add(1.0).div(2.0), nrow=nrow
                )
                save_image(grid, output_dir / "grid.png")
            atomic_json_dump(
                {
                    "protocol": "imagenette_sdvae_ldm128_sampling_v2",
                    "checkpoint": str(checkpoint_path),
                    "use_ema": bool(args.use_ema),
                    "sample_count": int(args.sample_count),
                    "sample_steps": int(args.sample_steps),
                    "guidance_scale": float(args.guidance_scale),
                    "seed": int(args.seed),
                    "world_size": context.world_size,
                    "rank_seed_stride": 1_000_003,
                    "balanced_class_labels": True,
                    "image_size": image_size,
                    "latent_shape": [LATENT_CHANNELS, latent_size, latent_size],
                },
                output_dir / "sampling_config.json",
            )
        barrier(context)
    finally:
        cleanup_distributed(context)


def benchmark(args: argparse.Namespace) -> None:
    context = initialize_distributed(args.device)
    try:
        configure_runtime(args.seed, args.allow_tf32)
        device = context.device
        image_size = int(args.image_size)
        latent_size = latent_size_for_image_size(image_size)
        channels = parse_int_tuple(args.channels)
        batch_size = int(args.batch_size)
        gradient_accumulation = int(args.gradient_accumulation)
        if gradient_accumulation < 1:
            raise ValueError("gradient accumulation must be positive")
        from diffusers.models import AutoencoderKL

        vae = AutoencoderKL.from_pretrained(args.vae_repo, local_files_only=True)
        vae = vae.to(
            device=device, dtype=torch.float32, memory_format=torch.channels_last
        ).eval()
        vae.requires_grad_(False)
        scaling_factor = float(getattr(vae.config, "scaling_factor", 0.18215))

        torch.manual_seed(args.seed + 101)
        raw_model = build_unet(channels, latent_size).to(
            device=device, dtype=torch.float32, memory_format=torch.channels_last
        )
        ema = EMA(raw_model) if context.is_main and args.include_ema else None
        optimizer = torch.optim.AdamW(raw_model.parameters(), lr=1e-4, fused=True)
        forward_model: nn.Module = raw_model
        if args.compile:
            forward_model = torch.compile(
                raw_model, mode=args.compile_mode, dynamic=False, fullgraph=False
            )
        if context.world_size > 1:
            forward_model = DistributedDataParallel(
                forward_model,
                device_ids=[context.local_rank],
                output_device=context.local_rank,
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
                static_graph=ddp_uses_static_graph(gradient_accumulation),
            )

        generator = torch.Generator(device=device).manual_seed(
            args.seed + 51_001 + context.rank * 100_003
        )
        images = torch.randn(
            (batch_size, 3, image_size, image_size),
            device=device,
            dtype=torch.float32,
            generator=generator,
        ).clamp(-1.0, 1.0).contiguous(memory_format=torch.channels_last)
        labels = torch.randint(
            0, NUM_CLASSES, (batch_size,), device=device, generator=generator
        )
        scheduler = stable_diffusion_noise_scheduler()
        warmup = max(2, int(args.warmup))
        measured = max(3, int(args.measured_steps))
        torch.cuda.reset_peak_memory_stats(device)
        started = None
        for index in range(warmup + measured):
            optimizer.zero_grad(set_to_none=True)
            for micro_index in range(gradient_accumulation):
                with torch.no_grad():
                    latents = encode_images(vae, images, scaling_factor).contiguous(
                        memory_format=torch.channels_last
                    )
                noise = torch.randn(latents.shape, device=device, generator=generator)
                timesteps = torch.randint(
                    0, SD_TIMESTEPS, (batch_size,), device=device, generator=generator
                ).long()
                noisy = scheduler.add_noise(latents, noise, timesteps)
                is_update = micro_index + 1 == gradient_accumulation
                synchronization = (
                    forward_model.no_sync()
                    if isinstance(forward_model, DistributedDataParallel) and not is_update
                    else nullcontext()
                )
                with synchronization:
                    prediction = unwrap_prediction(
                        forward_model(
                            noisy, timesteps, class_labels=labels, return_dict=False
                        )
                    )
                    loss = F.mse_loss(prediction, noise)
                    (loss / gradient_accumulation).backward()
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
            optimizer.step()
            if ema is not None:
                ema.update(raw_model, 0.9999, index + 1)
            if index + 1 == warmup:
                torch.cuda.synchronize(device)
                barrier(context)
                started = time.perf_counter()
        torch.cuda.synchronize(device)
        elapsed = torch.tensor(
            time.perf_counter() - float(started), device=device, dtype=torch.float64
        )
        if context.world_size > 1:
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        memory = torch.tensor(
            [
                torch.cuda.max_memory_allocated(device) / 2**30,
                torch.cuda.max_memory_reserved(device) / 2**30,
            ],
            device=device,
            dtype=torch.float64,
        )
        if context.world_size > 1:
            dist.all_reduce(memory, op=dist.ReduceOp.MAX)
        if context.is_main:
            elapsed_value = float(elapsed.item())
            result = {
                "per_gpu_batch_size": batch_size,
                "gradient_accumulation": gradient_accumulation,
                "global_batch_size": (
                    batch_size * context.world_size * gradient_accumulation
                ),
                "world_size": context.world_size,
                "image_size": image_size,
                "latent_shape": [LATENT_CHANNELS, latent_size, latent_size],
                "channels": list(channels),
                "parameter_count": sum(
                    parameter.numel() for parameter in raw_model.parameters()
                ),
                "steps_per_second": measured / elapsed_value,
                "global_images_per_second": (
                    measured
                    * batch_size
                    * context.world_size
                    * gradient_accumulation
                    / elapsed_value
                ),
                "max_allocated_gb": float(memory[0].item()),
                "max_reserved_gb": float(memory[1].item()),
                "online_vae": True,
                "rank0_ema": bool(args.include_ema),
                "compile": bool(args.compile),
                "compile_mode": args.compile_mode if args.compile else None,
                "dtype": "float32",
                "tf32": bool(args.allow_tf32),
            }
            print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    finally:
        cleanup_distributed(context)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    train_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT / "formal_v1"))
    train_parser.add_argument("--vae-repo", default=DEFAULT_VAE)
    train_parser.add_argument("--device", default="cuda:0")
    train_parser.add_argument("--seed", type=int, default=20260809)
    train_parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    train_parser.add_argument("--channels", default=",".join(map(str, FORMAL_CHANNELS)))
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--gradient-accumulation", type=int, default=1)
    train_parser.add_argument("--steps", type=int, default=50_000)
    train_parser.add_argument("--learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_parser.add_argument("--warmup-steps", type=int, default=1_000)
    train_parser.add_argument("--class-dropout", type=float, default=0.10)
    train_parser.add_argument("--ema-decay", type=float, default=0.9999)
    train_parser.add_argument("--gradient-clip", type=float, default=1.0)
    train_parser.add_argument("--num-workers", type=int, default=4)
    train_parser.add_argument("--validation-count", type=int, default=2_048)
    train_parser.add_argument("--validation-every", type=int, default=1_000)
    train_parser.add_argument("--log-every", type=int, default=100)
    train_parser.add_argument("--save-every", type=int, default=5_000)
    train_parser.add_argument("--extra-save-steps", default="1000,2000")
    train_parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    train_parser.add_argument("--compile-mode", default="default")
    train_parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    train_parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    train_parser.set_defaults(function=train)

    sample_parser = subparsers.add_parser("sample")
    sample_parser.add_argument("--checkpoint", required=True)
    sample_parser.add_argument("--output-dir", required=True)
    sample_parser.add_argument("--device", default="cuda:0")
    sample_parser.add_argument("--seed", type=int, default=0)
    sample_parser.add_argument("--sample-count", type=int, default=1_000)
    sample_parser.add_argument("--batch-size", type=int, default=32)
    sample_parser.add_argument("--decode-batch-size", type=int, default=16)
    sample_parser.add_argument("--sample-steps", type=int, default=50)
    sample_parser.add_argument("--guidance-scale", type=float, default=2.0)
    sample_parser.add_argument("--grid-count", type=int, default=100)
    sample_parser.add_argument("--save-individual", action="store_true")
    sample_parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    sample_parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    sample_parser.add_argument("--compile-mode", default="default")
    sample_parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    sample_parser.set_defaults(function=sample)

    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("--vae-repo", default=DEFAULT_VAE)
    benchmark_parser.add_argument("--device", default="cuda:0")
    benchmark_parser.add_argument("--seed", type=int, default=0)
    benchmark_parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    benchmark_parser.add_argument("--channels", default=",".join(map(str, FORMAL_CHANNELS)))
    benchmark_parser.add_argument("--batch-size", type=int, default=16)
    benchmark_parser.add_argument("--gradient-accumulation", type=int, default=1)
    benchmark_parser.add_argument("--warmup", type=int, default=3)
    benchmark_parser.add_argument("--measured-steps", type=int, default=5)
    benchmark_parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    benchmark_parser.add_argument("--compile-mode", default="default")
    benchmark_parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    benchmark_parser.add_argument(
        "--include-ema", action=argparse.BooleanOptionalAction, default=True
    )
    benchmark_parser.set_defaults(function=benchmark)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
