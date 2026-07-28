"""Controlled Imagenette-64 study of naturally learned latent responsibility.

The latent is available at every flow timestep. A ten-percent zero-condition
dropout makes the null branch part of the training distribution; no temporal
condition gate is implemented in this experiment.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.utils import make_grid, save_image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mnist_spectral_rollout_toy import (  # noqa: E402
    _group_count,
    configure_fp32,
    descending_time_grid,
    frechet_distance,
    sinusoidal_time_embedding,
)
from experiments.noise_responsibility_profile import (  # noqa: E402
    derangement,
    identity_control_error,
    radial_frequency_mse,
)


DEFAULT_DATA_ROOT = Path("/data/shared/imagenette2-320")
DEFAULT_OUTPUT_ROOT = Path.home() / "data/eqvae/imagenette_noise_responsibility"
CAPACITIES = (16, 64, 256)
FREQUENCY_BANDS = ("low", "mid", "high")
IMAGENETTE_SYNSET_TO_IMAGENET_INDEX = {
    "n01440764": 0,
    "n02102040": 217,
    "n02979186": 482,
    "n03000684": 491,
    "n03028079": 497,
    "n03394916": 566,
    "n03417042": 569,
    "n03425413": 571,
    "n03445777": 574,
    "n03888257": 701,
}


@dataclass(frozen=True)
class ImagenetteResponsibilityConfig:
    data_root: Path = DEFAULT_DATA_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    image_size: int = 64
    latent_dim: int = 64
    encoder_width: int = 32
    model_width: int = 48
    batch_size: int = 64
    steps: int = 20_000
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    condition_dropout: float = 0.10
    ema_decay: float = 0.999
    gradient_clip: float = 1.0
    num_workers: int = 4
    log_every: int = 500
    eval_count: int = 1_024
    quality_count: int = 1_024
    eval_batch_size: int = 64
    ode_steps: int = 50
    eval_times: tuple[float, ...] = (
        0.05,
        0.15,
        0.25,
        0.35,
        0.45,
        0.55,
        0.65,
        0.75,
        0.85,
        0.95,
        0.99,
    )
    frequency_boundaries: tuple[float, ...] = (0.25, 0.50)
    seed: int = 0
    device: str = "cuda:0"
    overwrite: bool = False
    save: bool = True

    def validate(self) -> None:
        if int(self.image_size) != 64:
            raise ValueError("the preregistered experiment fixes image_size=64")
        if int(self.latent_dim) not in CAPACITIES:
            raise ValueError(f"latent_dim must be one of {CAPACITIES}")
        if not 0.0 < float(self.condition_dropout) < 1.0:
            raise ValueError("condition_dropout must lie in (0, 1)")
        if int(self.steps) < 1 or int(self.batch_size) < 2:
            raise ValueError("steps must be positive and batch_size must be at least two")
        if len(self.frequency_boundaries) != 2:
            raise ValueError("the preregistered experiment uses exactly three frequency bands")


def _to_minus_one_one(image: torch.Tensor) -> torch.Tensor:
    return image.mul(2.0).sub(1.0)


def imagenette_transforms(image_size: int = 64) -> tuple[transforms.Compose, transforms.Compose]:
    train = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                int(image_size), scale=(0.75, 1.0), ratio=(0.9, 1.1), antialias=True
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(_to_minus_one_one),
        ]
    )
    evaluation = transforms.Compose(
        [
            transforms.Resize(int(round(image_size * 1.125)), antialias=True),
            transforms.CenterCrop(int(image_size)),
            transforms.ToTensor(),
            transforms.Lambda(_to_minus_one_one),
        ]
    )
    return train, evaluation


def load_imagenette_datasets(
    data_root: str | Path,
    image_size: int = 64,
) -> tuple[ImageFolder, ImageFolder]:
    root = Path(data_root)
    train_transform, evaluation_transform = imagenette_transforms(image_size)
    train_root = root / "train"
    val_root = root / "val"
    if not train_root.is_dir() or not val_root.is_dir():
        raise FileNotFoundError(f"expected Imagenette train/val under {root}")
    train = ImageFolder(str(train_root), transform=train_transform)
    val = ImageFolder(str(val_root), transform=evaluation_transform)
    if train.class_to_idx != val.class_to_idx:
        raise RuntimeError("Imagenette train and val class mappings differ")
    expected = set(IMAGENETTE_SYNSET_TO_IMAGENET_INDEX)
    if set(train.class_to_idx) != expected:
        raise RuntimeError("Imagenette synsets do not match the preregistered ten classes")
    return train, val


def make_train_loader(
    dataset: ImageFolder,
    config: ImagenetteResponsibilityConfig,
) -> DataLoader:
    generator = torch.Generator().manual_seed(int(config.seed) + 1001)
    return DataLoader(
        dataset,
        batch_size=int(config.batch_size),
        shuffle=True,
        drop_last=True,
        num_workers=int(config.num_workers),
        pin_memory=True,
        persistent_workers=int(config.num_workers) > 0,
        generator=generator,
    )


def fixed_eval_subset(dataset: ImageFolder, count: int, seed: int = 2027) -> Subset:
    count = min(int(count), len(dataset))
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:count]
    return Subset(dataset, indices.tolist())


class EncoderBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, stride=2, padding=1),
            nn.GroupNorm(_group_count(output_channels), output_channels),
            nn.SiLU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.GroupNorm(_group_count(output_channels), output_channels),
            nn.SiLU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(value)


class CompactImagenetteEncoder(nn.Module):
    """Small from-scratch image encoder with a fixed-dimensional vector code."""

    def __init__(self, latent_dim: int, width: int = 32):
        super().__init__()
        width = int(width)
        self.latent_dim = int(latent_dim)
        self.features = nn.Sequential(
            EncoderBlock(3, width),
            EncoderBlock(width, 2 * width),
            EncoderBlock(2 * width, 4 * width),
            EncoderBlock(4 * width, 4 * width),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.project = nn.Linear(4 * width * 4 * 4, self.latent_dim)
        self.normalize = nn.LayerNorm(self.latent_dim, elementwise_affine=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        hidden = self.pool(self.features(image)).flatten(1)
        return self.normalize(self.project(hidden))


class ConditionedResBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, embedding_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(input_channels), input_channels)
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.embedding = nn.Linear(embedding_dim, 2 * output_channels)
        self.norm2 = nn.GroupNorm(_group_count(output_channels), output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.skip = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv2d(input_channels, output_channels, 1)
        )

    def forward(self, value: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(value)))
        scale, shift = self.embedding(F.silu(embedding)).chunk(2, dim=1)
        hidden = self.norm2(hidden)
        hidden = hidden * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        hidden = self.conv2(F.silu(hidden))
        return self.skip(value) + hidden


class ImagenetteConditionalUNet(nn.Module):
    """64x64 conditional velocity U-Net with no temporal condition gate."""

    def __init__(self, latent_dim: int, width: int = 48):
        super().__init__()
        width = int(width)
        embedding_dim = 4 * width
        self.embedding_dim = embedding_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        c0, c1, c2, c3 = width, 2 * width, 3 * width, 4 * width
        self.input = nn.Conv2d(3, c0, 3, padding=1)
        self.down0 = nn.ModuleList(
            [ConditionedResBlock(c0, c0, embedding_dim), ConditionedResBlock(c0, c0, embedding_dim)]
        )
        self.downsample0 = nn.Conv2d(c0, c1, 3, stride=2, padding=1)
        self.down1 = nn.ModuleList(
            [ConditionedResBlock(c1, c1, embedding_dim), ConditionedResBlock(c1, c1, embedding_dim)]
        )
        self.downsample1 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.down2 = nn.ModuleList(
            [ConditionedResBlock(c2, c2, embedding_dim), ConditionedResBlock(c2, c2, embedding_dim)]
        )
        self.downsample2 = nn.Conv2d(c2, c3, 3, stride=2, padding=1)
        self.middle = nn.ModuleList(
            [ConditionedResBlock(c3, c3, embedding_dim), ConditionedResBlock(c3, c3, embedding_dim)]
        )

        self.upsample2 = nn.Conv2d(c3, c2, 3, padding=1)
        self.up2 = nn.ModuleList(
            [ConditionedResBlock(2 * c2, c2, embedding_dim), ConditionedResBlock(c2, c2, embedding_dim)]
        )
        self.upsample1 = nn.Conv2d(c2, c1, 3, padding=1)
        self.up1 = nn.ModuleList(
            [ConditionedResBlock(2 * c1, c1, embedding_dim), ConditionedResBlock(c1, c1, embedding_dim)]
        )
        self.upsample0 = nn.Conv2d(c1, c0, 3, padding=1)
        self.up0 = nn.ModuleList(
            [ConditionedResBlock(2 * c0, c0, embedding_dim), ConditionedResBlock(c0, c0, embedding_dim)]
        )
        self.output_norm = nn.GroupNorm(_group_count(c0), c0)
        self.output = nn.Conv2d(c0, 3, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        # Construct the only latent-dimension-dependent decoder module last so
        # every shared trunk tensor receives identical initialization across
        # bottleneck capacities for a fixed seed.
        self.condition_mlp = nn.Sequential(
            nn.Linear(int(latent_dim), embedding_dim, bias=False),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim, bias=False),
        )

    def condition_embedding(self, condition: torch.Tensor) -> torch.Tensor:
        raw = self.condition_mlp(condition)
        # A zero latent remains exactly zero. Non-null conditions all have fixed RMS.
        centered = raw - raw.mean(dim=1, keepdim=True)
        inverse_rms = torch.rsqrt(centered.square().mean(dim=1, keepdim=True) + 1e-8)
        return centered * inverse_rms

    def combined_embedding(
        self,
        time: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        time_embedding = self.time_mlp(sinusoidal_time_embedding(time, self.embedding_dim))
        return time_embedding + self.condition_embedding(condition)

    @staticmethod
    def _run(blocks: Iterable[nn.Module], value: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        for block in blocks:
            value = block(value, embedding)
        return value

    def forward(
        self,
        value: torch.Tensor,
        time: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        embedding = self.combined_embedding(time, condition)
        skip0 = self._run(self.down0, self.input(value), embedding)
        skip1 = self._run(self.down1, self.downsample0(skip0), embedding)
        skip2 = self._run(self.down2, self.downsample1(skip1), embedding)
        hidden = self._run(self.middle, self.downsample2(skip2), embedding)

        hidden = F.interpolate(hidden, size=skip2.shape[-2:], mode="nearest")
        hidden = self.upsample2(hidden)
        hidden = self._run(self.up2, torch.cat([hidden, skip2], dim=1), embedding)
        hidden = F.interpolate(hidden, size=skip1.shape[-2:], mode="nearest")
        hidden = self.upsample1(hidden)
        hidden = self._run(self.up1, torch.cat([hidden, skip1], dim=1), embedding)
        hidden = F.interpolate(hidden, size=skip0.shape[-2:], mode="nearest")
        hidden = self.upsample0(hidden)
        hidden = self._run(self.up0, torch.cat([hidden, skip0], dim=1), embedding)
        return self.output(F.silu(self.output_norm(hidden)))


class EMA:
    def __init__(self, module: nn.Module):
        self.module = copy.deepcopy(module).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, source: nn.Module, decay: float, step: int) -> None:
        effective_decay = min(float(decay), (1.0 + float(step)) / (10.0 + float(step)))
        source_parameters = dict(source.named_parameters())
        for name, target in self.module.named_parameters():
            target.lerp_(source_parameters[name].detach(), 1.0 - effective_decay)
        source_buffers = dict(source.named_buffers())
        for name, target in self.module.named_buffers():
            target.copy_(source_buffers[name])


def state_dict_sha256(
    module: nn.Module,
    *,
    exclude_prefixes: tuple[str, ...] = (),
) -> str:
    hasher = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        if any(name.startswith(prefix) for prefix in exclude_prefixes):
            continue
        hasher.update(name.encode("utf-8"))
        hasher.update(str(tuple(value.shape)).encode("ascii"))
        hasher.update(value.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


def build_fresh_models(
    config: ImagenetteResponsibilityConfig,
    device: torch.device,
) -> tuple[CompactImagenetteEncoder, ImagenetteConditionalUNet, dict[str, str]]:
    torch.manual_seed(int(config.seed) + 1101)
    encoder = CompactImagenetteEncoder(config.latent_dim, config.encoder_width)
    torch.manual_seed(int(config.seed) + 1103)
    model = ImagenetteConditionalUNet(config.latent_dim, config.model_width)
    hashes = {
        "encoder_shared_initial_sha256": state_dict_sha256(
            encoder, exclude_prefixes=("project.",)
        ),
        "decoder_shared_initial_sha256": state_dict_sha256(
            model, exclude_prefixes=("condition_mlp.",)
        ),
    }
    return encoder.to(device), model.to(device), hashes


def apply_condition_dropout(
    latent: torch.Tensor,
    probability: float,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = torch.rand((len(latent),), device=latent.device, generator=generator) < float(probability)
    dropped = latent.masked_fill(mask[:, None], 0.0)
    return dropped, mask


def _next_batch(iterator, loader: DataLoader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


@torch.no_grad()
def fixed_validation_loss(
    encoder: nn.Module,
    model: nn.Module,
    clean: torch.Tensor,
    noise: torch.Tensor,
    time: torch.Tensor,
    batch_size: int,
) -> float:
    expanded = time[:, None, None, None]
    state = (1.0 - expanded) * clean + expanded * noise
    target = noise - clean
    total = 0.0
    for indices in torch.arange(len(clean), device=clean.device).split(int(batch_size)):
        latent = encoder(clean[indices])
        prediction = model(state[indices], time[indices], latent)
        total += float(F.mse_loss(prediction, target[indices], reduction="sum"))
    return total / target.numel()


def train_model(
    train_dataset: ImageFolder,
    validation_clean: torch.Tensor,
    config: ImagenetteResponsibilityConfig,
) -> tuple[EMA, EMA, pd.DataFrame, dict[str, float | int | str]]:
    device = validation_clean.device
    configure_fp32(config.seed)
    encoder, model, initialization_hashes = build_fresh_models(config, device)
    encoder_ema = EMA(encoder)
    model_ema = EMA(model)
    parameters = list(encoder.parameters()) + list(model.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    loader = make_train_loader(train_dataset, config)
    iterator = iter(loader)
    noise_generator = torch.Generator(device=device).manual_seed(int(config.seed) + 1201)
    time_generator = torch.Generator(device=device).manual_seed(int(config.seed) + 1203)
    dropout_generator = torch.Generator(device=device).manual_seed(int(config.seed) + 1207)

    validation_generator = torch.Generator(device=device).manual_seed(91001)
    validation_noise = torch.randn(
        validation_clean.shape, device=device, generator=validation_generator
    )
    validation_time = torch.rand(
        (len(validation_clean),), device=device, generator=validation_generator
    )
    rows: list[dict[str, float | int]] = []
    dropped_total = 0
    example_total = 0
    stream_hasher = hashlib.sha256()
    encoder.train()
    model.train()
    for step in range(1, int(config.steps) + 1):
        (clean, _labels), iterator = _next_batch(iterator, loader)
        clean = clean.to(device, non_blocking=True)
        noise = torch.randn(clean.shape, device=device, generator=noise_generator)
        time = torch.rand((len(clean),), device=device, generator=time_generator)
        expanded = time[:, None, None, None]
        state = (1.0 - expanded) * clean + expanded * noise
        target = noise - clean
        latent = encoder(clean)
        condition, dropped = apply_condition_dropout(
            latent, config.condition_dropout, generator=dropout_generator
        )
        prediction = model(state, time, condition)
        loss = F.mse_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, float(config.gradient_clip))
        optimizer.step()
        encoder_ema.update(encoder, config.ema_decay, step)
        model_ema.update(model, config.ema_decay, step)
        dropped_total += int(dropped.sum())
        example_total += len(dropped)
        if step <= 32:
            stream_hasher.update(clean.detach().float().cpu().numpy().tobytes())

        if step == 1 or step % int(config.log_every) == 0 or step == int(config.steps):
            encoder_ema.module.eval()
            model_ema.module.eval()
            validation_loss = fixed_validation_loss(
                encoder_ema.module,
                model_ema.module,
                validation_clean,
                validation_noise,
                validation_time,
                config.eval_batch_size,
            )
            rows.append(
                {
                    "step": step,
                    "training_velocity_mse": float(loss.detach()),
                    "validation_velocity_mse": validation_loss,
                    "gradient_norm": float(gradient_norm),
                    "dropout_rate_so_far": dropped_total / example_total,
                }
            )
            print(
                json.dumps(
                    {
                        "step": step,
                        "latent_dim": int(config.latent_dim),
                        "seed": int(config.seed),
                        "train_mse": float(loss.detach()),
                        "val_mse": validation_loss,
                        "dropout_rate": dropped_total / example_total,
                    }
                ),
                flush=True,
            )
            encoder.train()
            model.train()

    encoder_ema.module.eval()
    model_ema.module.eval()
    metadata: dict[str, float | int | str] = {
        "dropout_count": dropped_total,
        "training_example_count": example_total,
        "observed_dropout_rate": dropped_total / example_total,
        "stream_hash_first_32_batches": stream_hasher.hexdigest(),
        "encoder_parameters": sum(parameter.numel() for parameter in encoder.parameters()),
        "decoder_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "total_parameters": sum(parameter.numel() for parameter in parameters),
        **initialization_hashes,
    }
    return encoder_ema, model_ema, pd.DataFrame(rows), metadata


@torch.no_grad()
def load_eval_tensors(
    dataset: ImageFolder,
    count: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    subset = fixed_eval_subset(dataset, count)
    loader = DataLoader(subset, batch_size=int(batch_size), shuffle=False, num_workers=0)
    images: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for batch_images, batch_labels in loader:
        images.append(batch_images)
        labels.append(batch_labels)
    subset_indices = torch.tensor(subset.indices, dtype=torch.long)
    return (
        torch.cat(images).to(device),
        torch.cat(labels).long().to(device),
        subset_indices.to(device),
    )


def within_class_derangement(labels: torch.Tensor, seed: int) -> torch.Tensor:
    permutation = torch.empty_like(labels)
    for class_index in labels.unique(sorted=True).tolist():
        positions = torch.where(labels == int(class_index))[0]
        local = derangement(len(positions), seed=int(seed) + 31 * int(class_index), device=labels.device)
        permutation[positions] = positions[local]
    if bool(torch.any(permutation == torch.arange(len(labels), device=labels.device))):
        raise RuntimeError("within-class shuffle contains a fixed point")
    if not torch.equal(labels[permutation], labels):
        raise RuntimeError("within-class shuffle changed a label")
    return permutation


def _paired_rows(
    target: torch.Tensor,
    predictions: dict[str, torch.Tensor],
) -> dict[str, np.ndarray]:
    losses = {
        name: (prediction.float() - target.float()).flatten(1).square().mean(dim=1)
        for name, prediction in predictions.items()
    }
    result: dict[str, np.ndarray] = {}
    for name, value in losses.items():
        result[f"loss_{name}"] = value.cpu().numpy()
    real = losses["real"]
    for name in ("null", "shuffle", "within_class"):
        delta = losses[name] - real
        result[f"delta_{name}"] = delta.cpu().numpy()
        result[f"gain_{name}"] = (
            delta / losses[name].clamp_min(torch.finfo(torch.float32).eps)
        ).cpu().numpy()
    return result


def _frequency_rows(
    target: torch.Tensor,
    predictions: dict[str, torch.Tensor],
    boundaries: tuple[float, ...],
) -> dict[str, torch.Tensor]:
    losses = {
        name: radial_frequency_mse(prediction, target, boundaries=boundaries)
        for name, prediction in predictions.items()
    }
    result: dict[str, torch.Tensor] = {}
    for name, value in losses.items():
        result[f"loss_{name}"] = value
    real = losses["real"]
    for name in ("null", "shuffle", "within_class"):
        result[f"delta_{name}"] = losses[name] - real
    return result


def aggregate_paired_rows(
    rows: pd.DataFrame,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    metric_columns = [
        column
        for column in rows.columns
        if column.startswith("loss_") or column.startswith("delta_") or column.startswith("gain_")
    ]
    records: list[dict[str, float | int | str]] = []
    for keys, frame in rows.groupby(list(group_columns), sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record: dict[str, float | int | str] = dict(zip(group_columns, keys))
        record["count"] = len(frame)
        for metric in metric_columns:
            values = frame[metric].to_numpy(dtype=np.float64)
            mean = float(values.mean())
            sem = float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
            record[f"{metric}_mean"] = mean
            record[f"{metric}_median"] = float(np.median(values))
            record[f"{metric}_sem"] = sem
            record[f"{metric}_ci95_low"] = mean - 1.96 * sem
            record[f"{metric}_ci95_high"] = mean + 1.96 * sem
            if metric.startswith("delta_"):
                record[f"{metric}_positive_rate"] = float(np.mean(values > 0.0))
        records.append(record)
    return pd.DataFrame.from_records(records)


@torch.no_grad()
def evaluate_responsibility(
    encoder: nn.Module,
    model: ImagenetteConditionalUNet,
    clean: torch.Tensor,
    labels: torch.Tensor,
    sample_indices: torch.Tensor,
    config: ImagenetteResponsibilityConfig,
) -> dict[str, pd.DataFrame | dict[str, float]]:
    device = clean.device
    latents = torch.cat(
        [encoder(batch) for batch in clean.split(int(config.eval_batch_size))]
    )
    random_permutation = derangement(len(clean), seed=22001, device=device)
    class_permutation = within_class_derangement(labels, seed=22003)
    generator = torch.Generator(device=device).manual_seed(22007)
    noise = torch.randn(clean.shape, device=device, generator=generator)
    paired_parts: list[pd.DataFrame] = []
    frequency_parts: list[pd.DataFrame] = []
    identity_rows: list[dict[str, float]] = []

    for time_value in config.eval_times:
        for batch_index, local_indices in enumerate(
            torch.arange(len(clean), device=device).split(int(config.eval_batch_size))
        ):
            batch_clean = clean[local_indices]
            batch_noise = noise[local_indices]
            time = torch.full((len(local_indices),), float(time_value), device=device)
            expanded = time[:, None, None, None]
            state = (1.0 - expanded) * batch_clean + expanded * batch_noise
            target = batch_noise - batch_clean
            conditions = {
                "real": latents[local_indices],
                "null": torch.zeros_like(latents[local_indices]),
                "shuffle": latents[random_permutation[local_indices]],
                "within_class": latents[class_permutation[local_indices]],
            }
            predictions = {
                name: model(state, time, condition)
                for name, condition in conditions.items()
            }
            if batch_index == 0:
                repeated = model(state, time, conditions["real"])
                control = identity_control_error(predictions["real"], repeated)
                identity_rows.append({"time": float(time_value), **control})

            values = _paired_rows(target, predictions)
            frame = pd.DataFrame(values)
            frame.insert(0, "time", float(time_value))
            frame.insert(0, "label", labels[local_indices].cpu().numpy())
            frame.insert(0, "shuffle_label", labels[random_permutation[local_indices]].cpu().numpy())
            frame.insert(0, "shuffle_index", sample_indices[random_permutation[local_indices]].cpu().numpy())
            frame.insert(0, "sample_index", sample_indices[local_indices].cpu().numpy())
            paired_parts.append(frame)

            frequency = _frequency_rows(target, predictions, config.frequency_boundaries)
            for band_index, band_name in enumerate(FREQUENCY_BANDS):
                band_frame = pd.DataFrame(
                    {
                        name: value[:, band_index].cpu().numpy()
                        for name, value in frequency.items()
                    }
                )
                band_frame.insert(0, "band", band_name)
                band_frame.insert(0, "time", float(time_value))
                band_frame.insert(0, "label", labels[local_indices].cpu().numpy())
                band_frame.insert(0, "sample_index", sample_indices[local_indices].cpu().numpy())
                frequency_parts.append(band_frame)

    paired = pd.concat(paired_parts, ignore_index=True)
    frequency_paired = pd.concat(frequency_parts, ignore_index=True)
    profile = aggregate_paired_rows(paired, ["time"])
    frequency_profile = aggregate_paired_rows(frequency_paired, ["time", "band"])

    projected = torch.cat(
        [model.condition_embedding(batch) for batch in latents.split(int(config.eval_batch_size))]
    )
    if not torch.isfinite(latents).all() or not torch.isfinite(projected).all():
        raise FloatingPointError("non-finite latent or condition embedding during evaluation")
    centered = latents - latents.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(len(latents) - 1, 1)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = torch.linalg.eigvalsh(covariance.double().cpu()).clamp_min(0.0)
    effective_rank = float(eigenvalues.sum().square() / eigenvalues.square().sum().clamp_min(1e-12))
    diagnostics = {
        "latent_l2_mean": float(latents.norm(dim=1).mean()),
        "latent_rms_mean": float(latents.square().mean(dim=1).sqrt().mean()),
        "condition_embedding_l2_mean": float(projected.norm(dim=1).mean()),
        "condition_embedding_rms_mean": float(projected.square().mean(dim=1).sqrt().mean()),
        "latent_effective_rank": effective_rank,
        "random_shuffle_fixed_points": int((random_permutation == torch.arange(len(clean), device=device)).sum()),
        "within_class_fixed_points": int((class_permutation == torch.arange(len(clean), device=device)).sum()),
        "within_class_label_mismatches": int((labels[class_permutation] != labels).sum()),
    }
    return {
        "paired": paired,
        "profile": profile,
        "frequency_paired": frequency_paired,
        "frequency_profile": frequency_profile,
        "identity_controls": pd.DataFrame(identity_rows),
        "diagnostics": diagnostics,
    }


def noise_region(time: float) -> str:
    if float(time) < 1.0 / 3.0:
        return "low_noise"
    if float(time) < 2.0 / 3.0:
        return "mid_noise"
    return "high_noise"


def curve_summary(profile: pd.DataFrame, frequency_profile: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float | str]] = []
    for source, table, bands in (
        ("total", profile.assign(band="all"), ("all",)),
        ("frequency", frequency_profile, FREQUENCY_BANDS),
    ):
        for band in bands:
            frame = table[table.band == band].copy().sort_values("time")
            frame["region"] = frame.time.map(noise_region)
            for metric in ("delta_shuffle_mean", "delta_null_mean", "delta_within_class_mean"):
                positive = frame[metric].clip(lower=0.0)
                region_values = {
                    region: float(positive[frame.region == region].mean())
                    for region in ("low_noise", "mid_noise", "high_noise")
                }
                denominator = sum(region_values.values())
                record: dict[str, float | str] = {
                    "source": source,
                    "band": band,
                    "metric": metric.removesuffix("_mean"),
                    **{f"{region}_positive_mean": value for region, value in region_values.items()},
                    "positive_region_sum": denominator,
                }
                for region, value in region_values.items():
                    record[f"{region}_fraction"] = value / denominator if denominator > 0.0 else float("nan")
                records.append(record)
    return pd.DataFrame.from_records(records)


class ResNet18Evaluator(nn.Module):
    def __init__(self):
        super().__init__()
        network = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(network.children())[:-1])
        self.classifier = network.fc
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        images = images.add(1.0).mul(0.5).clamp(0.0, 1.0)
        images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False, antialias=True)
        mean = images.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
        std = images.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
        features = self.features((images - mean) / std).flatten(1)
        return features, self.classifier(features)


@torch.no_grad()
def conditional_euler_sample(
    model: nn.Module,
    initial: torch.Tensor,
    condition: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    state = initial
    grid = descending_time_grid(int(steps), device=state.device)
    for current, following in zip(grid[:-1], grid[1:]):
        time = torch.full((len(state),), float(current), device=state.device)
        state = state + (following - current) * model(state, time, condition)
    return state


@torch.no_grad()
def evaluate_conditional_quality(
    encoder: nn.Module,
    model: nn.Module,
    clean: torch.Tensor,
    labels: torch.Tensor,
    class_to_idx: dict[str, int],
    config: ImagenetteResponsibilityConfig,
) -> tuple[dict[str, float], torch.Tensor]:
    device = clean.device
    evaluator = ResNet18Evaluator().to(device).eval()
    imagenet_indices = torch.empty(len(class_to_idx), dtype=torch.long, device=device)
    for synset, local_index in class_to_idx.items():
        imagenet_indices[local_index] = IMAGENETTE_SYNSET_TO_IMAGENET_INDEX[synset]
    generator = torch.Generator(device=device).manual_seed(33001)
    generated_parts: list[torch.Tensor] = []
    reference_features: list[torch.Tensor] = []
    generated_features: list[torch.Tensor] = []
    correct = 0
    squared_error_sum = 0.0
    value_count = 0
    for batch_clean, batch_labels in zip(
        clean.split(int(config.eval_batch_size)),
        labels.split(int(config.eval_batch_size)),
    ):
        latent = encoder(batch_clean)
        initial = torch.randn(batch_clean.shape, device=device, generator=generator)
        generated = conditional_euler_sample(model, initial, latent, config.ode_steps)
        generated_clamped = generated.clamp(-1.0, 1.0)
        real_feature, _ = evaluator(batch_clean)
        fake_feature, fake_logits = evaluator(generated_clamped)
        local_prediction = fake_logits[:, imagenet_indices].argmax(dim=1)
        correct += int((local_prediction == batch_labels).sum())
        squared_error_sum += float(F.mse_loss(generated_clamped, batch_clean, reduction="sum"))
        value_count += batch_clean.numel()
        generated_parts.append(generated_clamped.cpu())
        reference_features.append(real_feature.cpu())
        generated_features.append(fake_feature.cpu())
    generated_cpu = torch.cat(generated_parts)
    real_features = torch.cat(reference_features)
    fake_features = torch.cat(generated_features)
    metrics = {
        "conditional_feature_fid": frechet_distance(real_features, fake_features),
        "conditional_source_pixel_mse": squared_error_sum / value_count,
        "conditional_source_class_match": correct / len(clean),
        "quality_count": len(clean),
    }
    return metrics, generated_cpu


def _json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_ready) + "\n")


def run(config: ImagenetteResponsibilityConfig) -> Path | None:
    config.validate()
    configure_fp32(config.seed)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    train_dataset, val_dataset = load_imagenette_datasets(config.data_root, config.image_size)
    eval_total = max(int(config.eval_count), int(config.quality_count), 256)
    eval_clean, eval_labels, eval_indices = load_eval_tensors(
        val_dataset, eval_total, config.eval_batch_size, device
    )
    validation_clean = eval_clean[: min(256, len(eval_clean))]

    result_dir = config.output_root / f"d{config.latent_dim}_seed{config.seed}"
    if config.save:
        if result_dir.exists() and config.overwrite:
            shutil.rmtree(result_dir)
        if result_dir.exists():
            raise FileExistsError(f"result directory already exists: {result_dir}")
        result_dir.mkdir(parents=True)
        _write_json(result_dir / "config.json", asdict(config))

    encoder_ema, model_ema, history, training_metadata = train_model(
        train_dataset, validation_clean, config
    )
    responsibility = evaluate_responsibility(
        encoder_ema.module,
        model_ema.module,
        eval_clean[: int(config.eval_count)],
        eval_labels[: int(config.eval_count)],
        eval_indices[: int(config.eval_count)],
        config,
    )
    curves = curve_summary(
        responsibility["profile"], responsibility["frequency_profile"]
    )
    quality_metrics, generated = evaluate_conditional_quality(
        encoder_ema.module,
        model_ema.module,
        eval_clean[: int(config.quality_count)],
        eval_labels[: int(config.quality_count)],
        val_dataset.class_to_idx,
        config,
    )
    diagnostics = dict(responsibility["diagnostics"])
    identity = responsibility["identity_controls"]
    summary = {
        "latent_dim": int(config.latent_dim),
        "seed": int(config.seed),
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "final_validation_velocity_mse": float(history.iloc[-1].validation_velocity_mse),
        "identity_absolute_rms_max": float(identity.absolute_rms_max.max()),
        "identity_relative_rms_max": float(identity.relative_rms_max.max()),
        **training_metadata,
        **diagnostics,
        **quality_metrics,
    }

    if config.save:
        history.to_csv(result_dir / "history.csv", index=False)
        responsibility["paired"].to_csv(result_dir / "responsibility_paired.csv", index=False)
        responsibility["profile"].to_csv(result_dir / "responsibility_profile.csv", index=False)
        responsibility["frequency_paired"].to_csv(result_dir / "frequency_paired.csv", index=False)
        responsibility["frequency_profile"].to_csv(result_dir / "frequency_profile.csv", index=False)
        responsibility["identity_controls"].to_csv(result_dir / "identity_controls.csv", index=False)
        curves.to_csv(result_dir / "curve_summary.csv", index=False)
        _write_json(result_dir / "summary.json", summary)
        torch.save(
            {
                "config": asdict(config),
                "encoder_ema": encoder_ema.module.state_dict(),
                "model_ema": model_ema.module.state_dict(),
                "class_to_idx": val_dataset.class_to_idx,
                "summary": summary,
            },
            result_dir / "state.pt",
        )
        count = min(32, len(generated))
        comparison = torch.cat([eval_clean[:count].cpu(), generated[:count]], dim=0)
        grid = make_grid(comparison.add(1.0).mul(0.5), nrow=8)
        save_image(grid, result_dir / "conditional_samples.png")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return result_dir
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--latent-dim", type=int, choices=CAPACITIES, default=64)
    parser.add_argument("--encoder-width", type=int, default=32)
    parser.add_argument("--model-width", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--condition-dropout", type=float, default=0.10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--eval-count", type=int, default=1_024)
    parser.add_argument("--quality-count", type=int, default=1_024)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--ode-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> Path | None:
    args = build_parser().parse_args(argv)
    config = ImagenetteResponsibilityConfig(
        data_root=args.data_root,
        output_root=args.output_root,
        latent_dim=args.latent_dim,
        encoder_width=args.encoder_width,
        model_width=args.model_width,
        batch_size=args.batch_size,
        steps=args.steps,
        learning_rate=args.learning_rate,
        condition_dropout=args.condition_dropout,
        num_workers=args.num_workers,
        log_every=args.log_every,
        eval_count=args.eval_count,
        quality_count=args.quality_count,
        eval_batch_size=args.eval_batch_size,
        ode_steps=args.ode_steps,
        seed=args.seed,
        device=args.device,
        overwrite=args.overwrite,
        save=not args.no_save,
    )
    return run(config)


if __name__ == "__main__":
    main()
