"""Controlled dual-prediction-target experiment on frozen Imagenette latents.

The experiment deliberately separates three components:

1. A frozen SD-VAE maps deterministic Imagenette-64 crops to 4x8x8 latents.
2. One shared U-Net trunk predicts both the clean latent and the source noise.
3. Sampling can use either head or extrapolate away from the noise head in
   clean-latent coordinates.

All trainable tensors remain float32. TF32 is an explicit speed option and is
recorded in the run metadata. Artifacts live outside the repository by default.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image


DEFAULT_DATA_ROOT = Path("/data/shared/imagenette2-320")
DEFAULT_CACHE_ROOT = Path.home() / "data/eqvae/imagenette_sdvae_latents_64"
DEFAULT_OUTPUT_ROOT = Path.home() / "data/eqvae/imagenette_dual_target_latent"
DEFAULT_VAE = "stabilityai/sd-vae-ft-mse"
IMAGE_SIZE = 64
LATENT_CHANNELS = 4
LATENT_SIZE = 8
NUM_CLASSES = 10


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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")


def cuda_device_index(device: torch.device) -> int:
    if device.type != "cuda":
        raise ValueError(f"expected a CUDA device, received {device}")
    return torch.cuda.current_device() if device.index is None else int(device.index)


def image_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(72, antialias=True),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Lambda(lambda image: image.mul(2.0).sub(1.0)),
        ]
    )


class IndexedImageFolder(ImageFolder):
    def __getitem__(self, index: int):
        image, label = super().__getitem__(index)
        return image, int(label), int(index)


@torch.inference_mode()
def cache_split(
    *,
    data_root: Path,
    split: str,
    cache_root: Path,
    vae_repo: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    overwrite: bool,
) -> dict:
    output_path = cache_root / f"{split}.pt"
    if output_path.exists() and not overwrite:
        payload = torch.load(output_path, map_location="cpu", weights_only=False)
        return {
            "split": split,
            "count": int(len(payload["latents"])),
            "shape": list(payload["latents"].shape[1:]),
            "reused": True,
        }

    from diffusers.models import AutoencoderKL

    dataset = IndexedImageFolder(str(data_root / split), transform=image_transform())
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=True,
        persistent_workers=int(num_workers) > 0,
    )
    vae = AutoencoderKL.from_pretrained(vae_repo, local_files_only=True)
    vae = vae.to(device=device, dtype=torch.float32).eval()
    scaling_factor = float(getattr(vae.config, "scaling_factor", 0.18215))

    latent_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    index_parts: list[torch.Tensor] = []
    started = time.perf_counter()
    for batch_index, (images, labels, indices) in enumerate(loader, start=1):
        images = images.to(device=device, dtype=torch.float32, non_blocking=True)
        posterior = vae.encode(images).latent_dist
        latents = posterior.mode().mul(scaling_factor)
        latent_parts.append(latents.cpu())
        label_parts.append(labels.to(torch.int64).cpu())
        index_parts.append(indices.to(torch.int64).cpu())
        if batch_index % 20 == 0 or batch_index == len(loader):
            elapsed = max(time.perf_counter() - started, 1e-6)
            seen = min(batch_index * int(batch_size), len(dataset))
            print(
                f"[cache:{split}] {seen}/{len(dataset)} "
                f"({seen / elapsed:.1f} images/s)",
                flush=True,
            )

    latents = torch.cat(latent_parts).float().contiguous()
    labels = torch.cat(label_parts).long().contiguous()
    indices = torch.cat(index_parts).long().contiguous()
    order = torch.argsort(indices)
    latents, labels, indices = latents[order], labels[order], indices[order]
    expected = torch.arange(len(dataset), dtype=torch.long)
    if not torch.equal(indices, expected):
        raise RuntimeError(f"{split} cache indices are incomplete or duplicated")
    if tuple(latents.shape[1:]) != (LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE):
        raise RuntimeError(f"unexpected latent shape: {tuple(latents.shape)}")

    atomic_torch_save(
        {
            "latents": latents,
            "labels": labels,
            "indices": indices,
            "class_to_idx": dataset.class_to_idx,
            "vae_repo": vae_repo,
            "vae_scaling_factor": scaling_factor,
            "preprocess": "Resize(72)-CenterCrop(64)-ToTensor-[-1,1]",
        },
        output_path,
    )
    del vae
    torch.cuda.empty_cache()
    return {
        "split": split,
        "count": len(dataset),
        "shape": list(latents.shape[1:]),
        "reused": False,
    }


def prepare_cache(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    cache_root = Path(args.cache_root).expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for split in ("train", "val"):
        summaries.append(
            cache_split(
                data_root=Path(args.data_root).expanduser(),
                split=split,
                cache_root=cache_root,
                vae_repo=args.vae_repo,
                device=device,
                batch_size=args.cache_batch_size,
                num_workers=args.num_workers,
                overwrite=args.overwrite_cache,
            )
        )

    train = torch.load(cache_root / "train.pt", map_location="cpu", weights_only=False)
    train_latents = train["latents"].float()
    channel_mean = train_latents.mean(dim=(0, 2, 3), keepdim=True)
    channel_std = train_latents.std(dim=(0, 2, 3), keepdim=True, unbiased=False).clamp_min(1e-6)
    stats = {
        "channel_mean": channel_mean,
        "channel_std": channel_std,
        "train_count": int(len(train_latents)),
        "vae_repo": args.vae_repo,
    }
    atomic_torch_save(stats, cache_root / "stats.pt")
    atomic_json_dump(
        {
            "protocol": "imagenette_sdvae_latent_cache_v1",
            "image_size": IMAGE_SIZE,
            "latent_shape": [LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE],
            "summaries": summaries,
            "channel_mean": channel_mean.flatten().tolist(),
            "channel_std": channel_std.flatten().tolist(),
            "dtype": "float32",
        },
        cache_root / "metadata.json",
    )
    print(json.dumps({"cache_root": str(cache_root), "summaries": summaries}, indent=2))


def group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def sinusoidal_embedding(time_value: torch.Tensor, dimension: int) -> torch.Tensor:
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=time_value.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    phase = time_value.float()[:, None] * frequencies[None] * 1_000.0
    embedding = torch.cat((phase.sin(), phase.cos()), dim=1)
    if embedding.shape[1] < dimension:
        embedding = F.pad(embedding, (0, dimension - embedding.shape[1]))
    return embedding


class ConditionedResBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, embedding_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(group_count(input_channels), input_channels)
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.modulation = nn.Linear(embedding_dim, 2 * output_channels)
        self.norm2 = nn.GroupNorm(group_count(output_channels), output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.skip = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv2d(input_channels, output_channels, 1)
        )

    def forward(self, value: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(value)))
        scale, shift = self.modulation(F.silu(embedding)).chunk(2, dim=1)
        hidden = self.norm2(hidden)
        hidden = hidden * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        hidden = self.conv2(F.silu(hidden))
        return self.skip(value) + hidden


class SpatialAttention(nn.Module):
    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        if channels % heads != 0:
            raise ValueError("attention channels must be divisible by heads")
        self.channels = int(channels)
        self.heads = int(heads)
        self.norm = nn.GroupNorm(group_count(channels), channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.output = nn.Conv2d(channels, channels, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = value.shape
        hidden = self.qkv(self.norm(value))
        query, key, val = hidden.chunk(3, dim=1)
        head_dim = channels // self.heads

        def reshape(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(batch, self.heads, head_dim, height * width).transpose(-1, -2)

        attended = F.scaled_dot_product_attention(reshape(query), reshape(key), reshape(val))
        attended = attended.transpose(-1, -2).reshape(batch, channels, height, width)
        return value + self.output(attended)


class DualTargetLatentUNet(nn.Module):
    """One shared latent U-Net with independent clean and noise output heads."""

    def __init__(self, base_channels: int = 96, num_classes: int = NUM_CLASSES):
        super().__init__()
        base = int(base_channels)
        embedding_dim = 4 * base
        c0, c1, c2 = base, 2 * base, 3 * base
        self.embedding_dim = embedding_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.class_embedding = nn.Embedding(int(num_classes), embedding_dim)
        self.input = nn.Conv2d(LATENT_CHANNELS, c0, 3, padding=1)

        self.down0 = nn.ModuleList(
            [ConditionedResBlock(c0, c0, embedding_dim) for _ in range(2)]
        )
        self.downsample0 = nn.Conv2d(c0, c1, 3, stride=2, padding=1)
        self.down1 = nn.ModuleList(
            [ConditionedResBlock(c1, c1, embedding_dim) for _ in range(2)]
        )
        self.attention1 = SpatialAttention(c1, heads=4)
        self.downsample1 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.middle0 = ConditionedResBlock(c2, c2, embedding_dim)
        self.middle_attention = SpatialAttention(c2, heads=4)
        self.middle1 = ConditionedResBlock(c2, c2, embedding_dim)

        self.upsample1 = nn.Conv2d(c2, c1, 3, padding=1)
        self.up1 = nn.ModuleList(
            [
                ConditionedResBlock(2 * c1, c1, embedding_dim),
                ConditionedResBlock(c1, c1, embedding_dim),
            ]
        )
        self.up_attention1 = SpatialAttention(c1, heads=4)
        self.upsample0 = nn.Conv2d(c1, c0, 3, padding=1)
        self.up0 = nn.ModuleList(
            [
                ConditionedResBlock(2 * c0, c0, embedding_dim),
                ConditionedResBlock(c0, c0, embedding_dim),
            ]
        )
        self.output_norm = nn.GroupNorm(group_count(c0), c0)
        self.clean_head = nn.Conv2d(c0, LATENT_CHANNELS, 3, padding=1)
        self.noise_head = nn.Conv2d(c0, LATENT_CHANNELS, 3, padding=1)
        nn.init.zeros_(self.clean_head.weight)
        nn.init.zeros_(self.clean_head.bias)
        nn.init.zeros_(self.noise_head.weight)
        nn.init.zeros_(self.noise_head.bias)

    @staticmethod
    def run_blocks(
        blocks: Iterable[nn.Module], value: torch.Tensor, embedding: torch.Tensor
    ) -> torch.Tensor:
        for block in blocks:
            value = block(value, embedding)
        return value

    def forward(
        self, value: torch.Tensor, time_value: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        time_embedding = self.time_mlp(sinusoidal_embedding(time_value, self.embedding_dim))
        embedding = time_embedding + self.class_embedding(labels)

        skip0 = self.run_blocks(self.down0, self.input(value), embedding)
        skip1 = self.run_blocks(self.down1, self.downsample0(skip0), embedding)
        skip1 = self.attention1(skip1)
        hidden = self.downsample1(skip1)
        hidden = self.middle0(hidden, embedding)
        hidden = self.middle_attention(hidden)
        hidden = self.middle1(hidden, embedding)

        hidden = F.interpolate(hidden, size=skip1.shape[-2:], mode="nearest")
        hidden = self.upsample1(hidden)
        hidden = self.run_blocks(self.up1, torch.cat((hidden, skip1), dim=1), embedding)
        hidden = self.up_attention1(hidden)
        hidden = F.interpolate(hidden, size=skip0.shape[-2:], mode="nearest")
        hidden = self.upsample0(hidden)
        hidden = self.run_blocks(self.up0, torch.cat((hidden, skip0), dim=1), embedding)
        hidden = F.silu(self.output_norm(hidden))
        return self.clean_head(hidden), self.noise_head(hidden)


def clean_from_noise(
    state: torch.Tensor, time_value: torch.Tensor, noise_prediction: torch.Tensor
) -> torch.Tensor:
    time_image = time_value[:, None, None, None]
    return (state - time_image * noise_prediction) / (1.0 - time_image)


def velocity_from_clean(
    state: torch.Tensor, time_value: torch.Tensor, clean_prediction: torch.Tensor
) -> torch.Tensor:
    time_image = time_value[:, None, None, None]
    return (state - clean_prediction) / time_image


def velocity_from_noise(
    state: torch.Tensor, time_value: torch.Tensor, noise_prediction: torch.Tensor
) -> torch.Tensor:
    time_image = time_value[:, None, None, None]
    return (noise_prediction - state) / (1.0 - time_image)


def extrapolated_clean(
    state: torch.Tensor,
    time_value: torch.Tensor,
    clean_prediction: torch.Tensor,
    noise_prediction: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    noise_clean = clean_from_noise(state, time_value, noise_prediction)
    return clean_prediction + float(gamma) * (clean_prediction - noise_clean)


def prediction_losses(
    *,
    state: torch.Tensor,
    clean: torch.Tensor,
    noise: torch.Tensor,
    time_value: torch.Tensor,
    clean_prediction: torch.Tensor,
    noise_prediction: torch.Tensor,
    loss_space: str,
) -> dict[str, torch.Tensor]:
    target_velocity = noise - clean
    clean_velocity = velocity_from_clean(state, time_value, clean_prediction)
    noise_velocity = velocity_from_noise(state, time_value, noise_prediction)
    clean_direct = F.mse_loss(clean_prediction, clean)
    noise_direct = F.mse_loss(noise_prediction, noise)
    clean_velocity_loss = F.mse_loss(clean_velocity, target_velocity)
    noise_velocity_loss = F.mse_loss(noise_velocity, target_velocity)
    if loss_space == "v":
        total = 0.5 * (clean_velocity_loss + noise_velocity_loss)
    elif loss_space == "direct":
        total = 0.5 * (clean_direct + noise_direct)
    else:
        raise ValueError(f"unsupported loss space: {loss_space}")
    converted_noise_clean = clean_from_noise(state, time_value, noise_prediction)
    disagreement = F.mse_loss(clean_prediction, converted_noise_clean)
    return {
        "loss": total,
        "x_direct": clean_direct,
        "eps_direct": noise_direct,
        "x_v": clean_velocity_loss,
        "eps_v": noise_velocity_loss,
        "clean_disagreement": disagreement,
    }


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
    cache_root: str
    output_dir: str
    seed: int
    base_channels: int
    batch_size: int
    steps: int
    learning_rate: float
    weight_decay: float
    t_min: float
    t_max: float
    loss_space: str
    ema_decay: float
    gradient_clip: float
    log_every: int
    save_every: int
    extra_save_steps: tuple[int, ...]
    compile: bool
    compile_mode: str
    allow_tf32: bool
    device: str


def load_normalized_cache(cache_root: Path, split: str) -> tuple[torch.Tensor, torch.Tensor, dict]:
    payload = torch.load(cache_root / f"{split}.pt", map_location="cpu", weights_only=False)
    stats = torch.load(cache_root / "stats.pt", map_location="cpu", weights_only=False)
    mean = stats["channel_mean"].float()
    std = stats["channel_std"].float()
    latents = payload["latents"].float().sub(mean).div(std).contiguous()
    return latents, payload["labels"].long().contiguous(), stats


def latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = sorted(output_dir.glob("checkpoint_step*.pt"))
    return checkpoints[-1] if checkpoints else None


def validate_resume_config(stored: dict, current: TrainConfig) -> None:
    current_dict = asdict(current)
    allowed_to_change = {"steps", "device", "compile", "compile_mode"}
    mismatches = {
        key: (stored.get(key), current_value)
        for key, current_value in current_dict.items()
        if key not in allowed_to_change and stored.get(key) != current_value
    }
    if mismatches:
        raise ValueError(f"resume configuration mismatch: {mismatches}")


def model_parameter_count(base_channels: int) -> int:
    return sum(parameter.numel() for parameter in DualTargetLatentUNet(base_channels).parameters())


@torch.inference_mode()
def evaluate_fixed_batch(
    model: nn.Module,
    clean: torch.Tensor,
    labels: torch.Tensor,
    noise: torch.Tensor,
    time_value: torch.Tensor,
    loss_space: str,
) -> dict[str, float]:
    state = (1.0 - time_value[:, None, None, None]) * clean + time_value[
        :, None, None, None
    ] * noise
    clean_prediction, noise_prediction = model(state, time_value, labels)
    metrics = prediction_losses(
        state=state,
        clean=clean,
        noise=noise,
        time_value=time_value,
        clean_prediction=clean_prediction,
        noise_prediction=noise_prediction,
        loss_space=loss_space,
    )
    return {key: float(value) for key, value in metrics.items()}


def checkpoint_payload(
    *,
    step: int,
    model: nn.Module,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    config: TrainConfig,
) -> dict:
    return {
        "protocol": "imagenette_dual_target_latent_v1",
        "step": int(step),
        "model": model.state_dict(),
        "ema": ema.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "generator_state": generator.get_state().cpu(),
        "config": asdict(config),
    }


def train(args: argparse.Namespace) -> None:
    config = TrainConfig(
        cache_root=str(Path(args.cache_root).expanduser()),
        output_dir=str(Path(args.output_dir).expanduser()),
        seed=int(args.seed),
        base_channels=int(args.base_channels),
        batch_size=int(args.batch_size),
        steps=int(args.steps),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        t_min=float(args.t_min),
        t_max=float(args.t_max),
        loss_space=str(args.loss_space),
        ema_decay=float(args.ema_decay),
        gradient_clip=float(args.gradient_clip),
        log_every=int(args.log_every),
        save_every=int(args.save_every),
        extra_save_steps=tuple(
            sorted(
                {
                    int(value)
                    for value in args.extra_save_steps.split(",")
                    if value.strip() and int(value) > 0
                }
            )
        ),
        compile=bool(args.compile),
        compile_mode=str(args.compile_mode),
        allow_tf32=bool(args.allow_tf32),
        device=str(args.device),
    )
    if not 0.0 < config.t_min < config.t_max < 1.0:
        raise ValueError("expected 0 < t_min < t_max < 1")
    if min(config.batch_size, config.steps, config.log_every, config.save_every) < 1:
        raise ValueError("batch size, steps, log interval and save interval must be positive")
    configure_runtime(config.seed, config.allow_tf32)
    device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_latents, train_labels, stats = load_normalized_cache(Path(config.cache_root), "train")
    val_latents, val_labels, _ = load_normalized_cache(Path(config.cache_root), "val")
    train_latents = train_latents.to(device=device, memory_format=torch.channels_last)
    train_labels = train_labels.to(device=device)
    validation_count = min(2_048, len(val_latents))
    validation_index_generator = torch.Generator().manual_seed(config.seed + 19_991)
    validation_indices = torch.randperm(
        len(val_latents), generator=validation_index_generator
    )[:validation_count]
    validation_clean = val_latents[validation_indices].to(
        device=device, memory_format=torch.channels_last
    )
    validation_labels = val_labels[validation_indices].to(device=device)

    generator = torch.Generator(device=device).manual_seed(config.seed + 10_001)
    validation_generator = torch.Generator(device=device).manual_seed(config.seed + 20_001)
    validation_noise = torch.randn(
        validation_clean.shape,
        generator=validation_generator,
        device=device,
        dtype=torch.float32,
    ).contiguous(memory_format=torch.channels_last)
    validation_time = torch.rand(
        (validation_count,), generator=validation_generator, device=device
    ).mul(config.t_max - config.t_min).add(config.t_min)

    torch.manual_seed(config.seed + 101)
    raw_model = DualTargetLatentUNet(config.base_channels).to(
        device=device, dtype=torch.float32, memory_format=torch.channels_last
    )
    ema = EMA(raw_model)
    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        fused=True,
    )
    start_step = 0
    checkpoint = latest_checkpoint(output_dir) if args.resume else None
    if checkpoint is not None:
        restored = torch.load(checkpoint, map_location=device, weights_only=False)
        validate_resume_config(restored["config"], config)
        raw_model.load_state_dict(restored["model"])
        ema.module.load_state_dict(restored["ema"])
        optimizer.load_state_dict(restored["optimizer"])
        generator.set_state(restored["generator_state"].to(device))
        start_step = int(restored["step"])
        print(f"[resume] {checkpoint} at step {start_step}", flush=True)

    train_model: nn.Module = raw_model
    if config.compile:
        print(f"[compile] mode={config.compile_mode}", flush=True)
        train_model = torch.compile(
            raw_model, mode=config.compile_mode, fullgraph=True, dynamic=False
        )

    metadata = {
        "protocol": "imagenette_dual_target_latent_v1",
        "config": asdict(config),
        "parameter_count": sum(parameter.numel() for parameter in raw_model.parameters()),
        "train_count": int(len(train_latents)),
        "val_count": int(len(val_latents)),
        "validation_subset_count": int(validation_count),
        "validation_subset_class_counts": torch.bincount(
            validation_labels.cpu(), minlength=NUM_CLASSES
        ).tolist(),
        "latent_shape": list(train_latents.shape[1:]),
        "dtype": "float32",
        "tf32": config.allow_tf32,
        "vae_repo": stats["vae_repo"],
        "git_commit": os.popen("git rev-parse HEAD 2>/dev/null").read().strip(),
    }
    atomic_json_dump(metadata, output_dir / "run_config.json")
    log_path = output_dir / "train_metrics.jsonl"

    running = {key: 0.0 for key in ("loss", "x_direct", "eps_direct", "x_v", "eps_v", "clean_disagreement")}
    running_count = 0
    interval_started = time.perf_counter()
    raw_model.train()
    for step in range(start_step + 1, config.steps + 1):
        indices = torch.randint(
            len(train_latents),
            (config.batch_size,),
            generator=generator,
            device=device,
        )
        clean = train_latents[indices].contiguous(memory_format=torch.channels_last)
        labels = train_labels[indices]
        noise = torch.randn(
            clean.shape, generator=generator, device=device, dtype=torch.float32
        ).contiguous(memory_format=torch.channels_last)
        time_value = torch.rand(
            (config.batch_size,), generator=generator, device=device
        ).mul(config.t_max - config.t_min).add(config.t_min)
        time_image = time_value[:, None, None, None]
        state = ((1.0 - time_image) * clean + time_image * noise).contiguous(
            memory_format=torch.channels_last
        )

        clean_prediction, noise_prediction = train_model(state, time_value, labels)
        losses = prediction_losses(
            state=state,
            clean=clean,
            noise=noise,
            time_value=time_value,
            clean_prediction=clean_prediction,
            noise_prediction=noise_prediction,
            loss_space=config.loss_space,
        )
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            raw_model.parameters(), config.gradient_clip
        )
        optimizer.step()
        ema.update(raw_model, config.ema_decay, step)

        for key in running:
            running[key] += float(losses[key].detach())
        running_count += 1

        should_log = step % config.log_every == 0 or step == config.steps
        if should_log:
            torch.cuda.synchronize(device)
            elapsed = max(time.perf_counter() - interval_started, 1e-6)
            validation = evaluate_fixed_batch(
                ema.module,
                validation_clean,
                validation_labels,
                validation_noise,
                validation_time,
                config.loss_space,
            )
            row = {
                "step": step,
                "train": {key: value / running_count for key, value in running.items()},
                "validation": validation,
                "gradient_norm": float(gradient_norm),
                "steps_per_second": running_count / elapsed,
                "images_per_second": config.batch_size * running_count / elapsed,
                "gpu_memory_allocated_gb": torch.cuda.max_memory_allocated(
                    cuda_device_index(device)
                )
                / 2**30,
                "gpu_memory_reserved_gb": torch.cuda.max_memory_reserved(
                    cuda_device_index(device)
                )
                / 2**30,
                "x_head_better_v_loss": bool(validation["x_v"] < validation["eps_v"]),
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps(row, sort_keys=True), flush=True)
            running = {key: 0.0 for key in running}
            running_count = 0
            interval_started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats(cuda_device_index(device))

        if (
            step % config.save_every == 0
            or step in config.extra_save_steps
            or step == config.steps
        ):
            payload = checkpoint_payload(
                step=step,
                model=raw_model,
                ema=ema,
                optimizer=optimizer,
                generator=generator,
                config=config,
            )
            path = output_dir / f"checkpoint_step{step:07d}.pt"
            atomic_torch_save(payload, path)
            atomic_torch_save(payload, output_dir / "latest.pt")
            print(f"[checkpoint] {path}", flush=True)


def parse_gamma_list(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("gamma list cannot be empty")
    return values


@torch.inference_mode()
def rollout(
    model: nn.Module,
    initial_noise: torch.Tensor,
    labels: torch.Tensor,
    *,
    steps: int,
    t_min: float,
    t_max: float,
    mode: str,
    gamma: float = 0.0,
) -> torch.Tensor:
    state = initial_noise.clone(memory_format=torch.preserve_format)
    grid = torch.linspace(t_max, t_min, int(steps) + 1, device=state.device)
    for index in range(int(steps)):
        current = grid[index].expand(len(state))
        next_time = grid[index + 1]
        clean_prediction, noise_prediction = model(state, current, labels)
        if mode == "x":
            velocity = velocity_from_clean(state, current, clean_prediction)
        elif mode == "eps":
            velocity = velocity_from_noise(state, current, noise_prediction)
        elif mode == "extrapolate":
            clean = extrapolated_clean(
                state,
                current,
                clean_prediction,
                noise_prediction,
                gamma,
            )
            velocity = velocity_from_clean(state, current, clean)
        else:
            raise ValueError(f"unsupported rollout mode: {mode}")
        state = state + (next_time - grid[index]) * velocity
    final_time = grid[-1].expand(len(state))
    clean_prediction, noise_prediction = model(state, final_time, labels)
    if mode == "eps":
        return clean_from_noise(state, final_time, noise_prediction)
    if mode == "extrapolate":
        return extrapolated_clean(
            state, final_time, clean_prediction, noise_prediction, gamma
        )
    return clean_prediction


def sliced_wasserstein(
    reference: torch.Tensor,
    sample: torch.Tensor,
    *,
    projections: int,
    seed: int,
) -> float:
    count = min(len(reference), len(sample))
    generator = torch.Generator(device=reference.device).manual_seed(int(seed))
    reference_indices = torch.randperm(len(reference), generator=generator, device=reference.device)[:count]
    sample_indices = torch.randperm(len(sample), generator=generator, device=sample.device)[:count]
    reference_flat = reference[reference_indices].flatten(1)
    sample_flat = sample[sample_indices].flatten(1)
    directions = torch.randn(
        (reference_flat.shape[1], int(projections)),
        generator=generator,
        device=reference.device,
    )
    directions = F.normalize(directions, dim=0)
    reference_projected = torch.sort(reference_flat @ directions, dim=0).values
    sample_projected = torch.sort(sample_flat @ directions, dim=0).values
    return float((reference_projected - sample_projected).square().mean().sqrt())


@torch.inference_mode()
def decode_latents(
    normalized_latents: torch.Tensor,
    *,
    cache_root: Path,
    vae_repo: str,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    from diffusers.models import AutoencoderKL

    stats = torch.load(cache_root / "stats.pt", map_location="cpu", weights_only=False)
    mean = stats["channel_mean"].to(device=device)
    std = stats["channel_std"].to(device=device)
    vae = AutoencoderKL.from_pretrained(vae_repo, local_files_only=True)
    vae = vae.to(device=device, dtype=torch.float32).eval()
    scaling_factor = float(getattr(vae.config, "scaling_factor", 0.18215))
    images = []
    for part in normalized_latents.split(int(batch_size)):
        raw = part * std + mean
        decoded = vae.decode(raw / scaling_factor).sample.clamp(-1.0, 1.0)
        images.append(decoded.cpu())
    return torch.cat(images)


def sample(args: argparse.Namespace) -> None:
    configure_runtime(args.seed, args.allow_tf32)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    checkpoint_path = Path(args.checkpoint).expanduser()
    restored = torch.load(checkpoint_path, map_location=device, weights_only=False)
    stored_config = restored["config"]
    base_channels = int(stored_config["base_channels"])
    model = DualTargetLatentUNet(base_channels).to(
        device=device, dtype=torch.float32, memory_format=torch.channels_last
    )
    model.load_state_dict(restored["ema"] if args.use_ema else restored["model"])
    model.eval()
    if args.compile:
        model = torch.compile(model, mode=args.compile_mode, fullgraph=True, dynamic=False)

    cache_root = Path(args.cache_root).expanduser()
    validation, _, _ = load_normalized_cache(cache_root, "val")
    validation = validation.to(device=device, memory_format=torch.channels_last)
    generator = torch.Generator(device=device).manual_seed(args.seed + 40_001)
    initial_noise = torch.randn(
        (args.sample_count, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE),
        generator=generator,
        device=device,
    ).contiguous(memory_format=torch.channels_last)
    labels = (torch.arange(args.sample_count, device=device) % NUM_CLASSES).long()
    permutations = torch.randperm(args.sample_count, generator=generator, device=device)
    labels = labels[permutations]

    conditions: list[tuple[str, str, float]] = [("x", "x", 0.0), ("eps", "eps", -1.0)]
    for gamma in parse_gamma_list(args.gammas):
        conditions.append((f"gamma_{gamma:+.3f}", "extrapolate", gamma))
    output_dir = Path(args.sample_output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for name, mode, gamma in conditions:
        generated_parts = []
        for indices in torch.arange(args.sample_count, device=device).split(args.sample_batch_size):
            generated_parts.append(
                rollout(
                    model,
                    initial_noise[indices],
                    labels[indices],
                    steps=args.sample_steps,
                    t_min=float(stored_config["t_min"]),
                    t_max=float(stored_config["t_max"]),
                    mode=mode,
                    gamma=gamma,
                )
            )
        generated = torch.cat(generated_parts)
        swd = sliced_wasserstein(
            validation,
            generated,
            projections=args.swd_projections,
            seed=args.seed + 50_001,
        )
        decoded = decode_latents(
            generated[: args.grid_count],
            cache_root=cache_root,
            vae_repo=args.vae_repo,
            device=device,
            batch_size=args.decode_batch_size,
        )
        grid = make_grid(decoded.add(1.0).div(2.0), nrow=int(math.sqrt(args.grid_count)))
        save_image(grid, output_dir / f"{name}.png")
        atomic_torch_save(
            {"latents": generated.cpu(), "labels": labels.cpu()},
            output_dir / f"{name}_latents.pt",
        )
        row = {"condition": name, "mode": mode, "gamma": gamma, "latent_swd": swd}
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    with (output_dir / "sample_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_json_dump(
        {
            "protocol": "imagenette_dual_target_latent_sampling_v1",
            "checkpoint": str(checkpoint_path),
            "use_ema": bool(args.use_ema),
            "sample_count": int(args.sample_count),
            "sample_steps": int(args.sample_steps),
            "rows": rows,
        },
        output_dir / "sampling_summary.json",
    )


def benchmark(args: argparse.Namespace) -> None:
    configure_runtime(args.seed, args.allow_tf32)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    candidates = [int(value) for value in args.benchmark_batches.split(",") if value.strip()]
    rows = []
    for batch_size in candidates:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(cuda_device_index(device))
        try:
            torch.manual_seed(args.seed + 101)
            raw_model = DualTargetLatentUNet(args.base_channels).to(
                device=device, dtype=torch.float32, memory_format=torch.channels_last
            )
            model: nn.Module = raw_model
            if args.compile:
                model = torch.compile(
                    raw_model, mode=args.compile_mode, fullgraph=True, dynamic=False
                )
            optimizer = torch.optim.AdamW(raw_model.parameters(), lr=2e-4, fused=True)
            generator = torch.Generator(device=device).manual_seed(args.seed + batch_size)
            clean = torch.randn(
                (batch_size, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE),
                generator=generator,
                device=device,
            ).contiguous(memory_format=torch.channels_last)
            labels = torch.randint(
                NUM_CLASSES, (batch_size,), generator=generator, device=device
            )
            warmup = max(3, int(args.benchmark_warmup))
            measured = max(5, int(args.benchmark_steps))
            started = None
            for step in range(warmup + measured):
                noise = torch.randn(
                    clean.shape, generator=generator, device=device
                ).contiguous(memory_format=torch.channels_last)
                time_value = torch.rand(
                    (batch_size,), generator=generator, device=device
                ).mul(0.90).add(0.05)
                state = (1.0 - time_value[:, None, None, None]) * clean + time_value[
                    :, None, None, None
                ] * noise
                clean_prediction, noise_prediction = model(state, time_value, labels)
                losses = prediction_losses(
                    state=state,
                    clean=clean,
                    noise=noise,
                    time_value=time_value,
                    clean_prediction=clean_prediction,
                    noise_prediction=noise_prediction,
                    loss_space=args.loss_space,
                )
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                optimizer.step()
                if step + 1 == warmup:
                    torch.cuda.synchronize(device)
                    started = time.perf_counter()
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - float(started)
            row = {
                "batch_size": batch_size,
                "steps_per_second": measured / elapsed,
                "images_per_second": measured * batch_size / elapsed,
                "allocated_gb": torch.cuda.max_memory_allocated(cuda_device_index(device))
                / 2**30,
                "reserved_gb": torch.cuda.max_memory_reserved(cuda_device_index(device))
                / 2**30,
                "status": "ok",
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            del model, raw_model, optimizer, clean, labels
        except torch.cuda.OutOfMemoryError:
            row = {"batch_size": batch_size, "status": "oom"}
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
        finally:
            torch.cuda.empty_cache()
    print(json.dumps({"parameter_count": model_parameter_count(args.base_channels), "rows": rows}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    cache = subparsers.add_parser("cache")
    cache.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    cache.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    cache.add_argument("--vae-repo", default=DEFAULT_VAE)
    cache.add_argument("--device", default="cuda:0")
    cache.add_argument("--cache-batch-size", type=int, default=128)
    cache.add_argument("--num-workers", type=int, default=8)
    cache.add_argument("--overwrite-cache", action="store_true")
    cache.set_defaults(function=prepare_cache)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    train_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT / "seed0"))
    train_parser.add_argument("--device", default="cuda:0")
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--base-channels", type=int, default=96)
    train_parser.add_argument("--batch-size", type=int, default=1024)
    train_parser.add_argument("--steps", type=int, default=30_000)
    train_parser.add_argument("--learning-rate", type=float, default=2e-4)
    train_parser.add_argument("--weight-decay", type=float, default=0.0)
    train_parser.add_argument("--t-min", type=float, default=0.05)
    train_parser.add_argument("--t-max", type=float, default=0.95)
    train_parser.add_argument("--loss-space", choices=("v", "direct"), default="v")
    train_parser.add_argument("--ema-decay", type=float, default=0.9999)
    train_parser.add_argument("--gradient-clip", type=float, default=10.0)
    train_parser.add_argument("--log-every", type=int, default=100)
    train_parser.add_argument("--save-every", type=int, default=2_000)
    train_parser.add_argument("--extra-save-steps", default="200,500,1000")
    train_parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    train_parser.add_argument("--compile-mode", default="max-autotune")
    train_parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    train_parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    train_parser.set_defaults(function=train)

    sample_parser = subparsers.add_parser("sample")
    sample_parser.add_argument("--checkpoint", required=True)
    sample_parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    sample_parser.add_argument("--sample-output", required=True)
    sample_parser.add_argument("--device", default="cuda:0")
    sample_parser.add_argument("--vae-repo", default=DEFAULT_VAE)
    sample_parser.add_argument("--seed", type=int, default=0)
    sample_parser.add_argument("--sample-count", type=int, default=1_024)
    sample_parser.add_argument("--sample-batch-size", type=int, default=256)
    sample_parser.add_argument("--sample-steps", type=int, default=100)
    sample_parser.add_argument("--gammas", default="-0.5,-0.25,-0.1,0.05,0.1,0.2,0.4,0.8")
    sample_parser.add_argument("--grid-count", type=int, default=100)
    sample_parser.add_argument("--decode-batch-size", type=int, default=128)
    sample_parser.add_argument("--swd-projections", type=int, default=256)
    sample_parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    sample_parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    sample_parser.add_argument("--compile-mode", default="max-autotune")
    sample_parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    sample_parser.set_defaults(function=sample)

    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("--device", default="cuda:0")
    benchmark_parser.add_argument("--seed", type=int, default=0)
    benchmark_parser.add_argument("--base-channels", type=int, default=96)
    benchmark_parser.add_argument("--benchmark-batches", default="256,512,1024,1536,2048")
    benchmark_parser.add_argument("--benchmark-warmup", type=int, default=5)
    benchmark_parser.add_argument("--benchmark-steps", type=int, default=10)
    benchmark_parser.add_argument("--loss-space", choices=("v", "direct"), default="v")
    benchmark_parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    benchmark_parser.add_argument("--compile-mode", default="max-autotune")
    benchmark_parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    benchmark_parser.set_defaults(function=benchmark)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
