"""Close the Imagenette-64 two-stage generator with equal-budget latent priors.

This experiment freezes the fifteen encoders and pixel decoders from
``imagenette_noise_responsibility.py``. Latents of every nominal capacity are
isometrically embedded into one 256-dimensional interface, while both the data
and Gaussian source remain in the corresponding active subspace.
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
from pathlib import Path, PosixPath
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.imagenette_noise_responsibility import (  # noqa: E402
    CAPACITIES,
    DEFAULT_DATA_ROOT,
    IMAGENETTE_SYNSET_TO_IMAGENET_INDEX,
    CompactImagenetteEncoder,
    EMA,
    ImagenetteConditionalUNet,
    ResNet18Evaluator,
    conditional_euler_sample,
    fixed_eval_subset,
    imagenette_transforms,
    state_dict_sha256,
)
from experiments.mnist_spectral_rollout_toy import (  # noqa: E402
    configure_fp32,
    descending_time_grid,
    frechet_distance,
    sinusoidal_time_embedding,
)


DEFAULT_CHECKPOINT_ROOT = (
    Path.home() / "data/eqvae/imagenette_noise_responsibility_formal"
)
DEFAULT_OUTPUT_ROOT = Path.home() / "data/eqvae/imagenette_latent_prior_tradeoff"
INTERFACE_DIM = 256
FORMAL_SEEDS = (0, 1, 2, 3, 4)
ROLLOUT_MODES = ("oracle", "empirical", "prior", "gaussian")


@dataclass(frozen=True)
class LatentPriorTradeoffConfig:
    data_root: Path = DEFAULT_DATA_ROOT
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    latent_dim: int = 64
    frozen_seed: int = 0
    prior_replicate: int = 0
    interface_dim: int = INTERFACE_DIM
    image_size: int = 64
    encoder_width: int = 32
    decoder_width: int = 48
    prior_width: int = 512
    prior_depth: int = 6
    prior_steps: int = 20_000
    prior_batch_size: int = 512
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    ema_decay: float = 0.999
    gradient_clip: float = 1.0
    log_every: int = 500
    num_workers: int = 4
    encode_batch_size: int = 128
    eval_batch_size: int = 64
    quality_count: int = 3_925
    prior_ode_steps: int = 100
    pixel_ode_steps: int = 50
    sliced_directions: int = 256
    basis_seed: int = 61_423
    device: str = "cuda:0"
    overwrite: bool = False
    resume: bool = False
    save: bool = True

    @property
    def prior_seed(self) -> int:
        return 71_000 + int(self.frozen_seed) + 1_000 * int(self.prior_replicate)

    @property
    def frozen_run(self) -> Path:
        return self.checkpoint_root.expanduser() / (
            f"d{int(self.latent_dim)}_seed{int(self.frozen_seed)}"
        )

    @property
    def result_dir(self) -> Path:
        return self.output_root.expanduser() / (
            f"d{int(self.latent_dim)}_seed{int(self.frozen_seed)}_p{int(self.prior_replicate)}"
        )

    def validate(self) -> None:
        if int(self.latent_dim) not in CAPACITIES:
            raise ValueError(f"latent_dim must be one of {CAPACITIES}")
        if int(self.interface_dim) != INTERFACE_DIM:
            raise ValueError(f"the preregistered interface dimension is {INTERFACE_DIM}")
        if int(self.image_size) != 64:
            raise ValueError("the preregistered image size is 64")
        if int(self.prior_steps) < 1 or int(self.prior_batch_size) < 2:
            raise ValueError("prior_steps and prior_batch_size must be positive")
        if int(self.prior_ode_steps) < 1 or int(self.pixel_ode_steps) < 1:
            raise ValueError("ODE step counts must be positive")
        if int(self.quality_count) < 2:
            raise ValueError("quality_count must be at least two")


def _json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_ready) + "\n"
    )


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    hasher = hashlib.sha256()
    hasher.update(str(tuple(array.shape)).encode("ascii"))
    hasher.update(array.tobytes())
    return hasher.hexdigest()


def path_list_sha256(paths: Iterable[str]) -> str:
    hasher = hashlib.sha256()
    for path in paths:
        hasher.update(str(path).encode("utf-8"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def fixed_orthogonal_basis(size: int = INTERFACE_DIM, seed: int = 61_423) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    matrix = torch.randn((int(size), int(size)), generator=generator, dtype=torch.float64)
    basis, triangular = torch.linalg.qr(matrix)
    sign = torch.sign(torch.diagonal(triangular))
    sign[sign == 0] = 1
    return (basis * sign[None, :]).float().contiguous()


class OrthogonalLatentInterface:
    """An isometric d-dimensional coordinate system inside a 256D interface."""

    def __init__(self, latent_dim: int, basis: torch.Tensor):
        if basis.ndim != 2 or basis.shape[0] != basis.shape[1]:
            raise ValueError("basis must be square")
        if not 0 < int(latent_dim) <= basis.shape[1]:
            raise ValueError("latent_dim is incompatible with basis")
        self.latent_dim = int(latent_dim)
        self.interface_dim = int(basis.shape[0])
        self.basis = basis[:, : self.latent_dim].contiguous()

    def to(self, device: torch.device | str) -> "OrthogonalLatentInterface":
        converted = object.__new__(OrthogonalLatentInterface)
        converted.latent_dim = self.latent_dim
        converted.interface_dim = self.interface_dim
        converted.basis = self.basis.to(device)
        return converted

    def embed(self, coordinates: torch.Tensor) -> torch.Tensor:
        if coordinates.shape[-1] != self.latent_dim:
            raise ValueError("coordinate dimension does not match interface")
        return coordinates @ self.basis.T

    def recover(self, embedded: torch.Tensor) -> torch.Tensor:
        if embedded.shape[-1] != self.interface_dim:
            raise ValueError("embedded dimension does not match interface")
        coordinates = embedded @ self.basis
        # One residual-refinement step removes the O(1e-6) FP32 round-trip
        # error of a dense orthogonal basis without changing the represented
        # subspace or introducing a capacity-dependent operation.
        residual = embedded - coordinates @ self.basis.T
        return coordinates + residual @ self.basis

    def project(self, embedded: torch.Tensor) -> torch.Tensor:
        return self.embed(self.recover(embedded))


class ResidualPriorBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(int(width))
        self.network = nn.Sequential(
            nn.Linear(int(width), 2 * int(width)),
            nn.SiLU(),
            nn.Linear(2 * int(width), int(width)),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.network(self.norm(value))


class UnifiedLatentVelocityMLP(nn.Module):
    """A capacity-independent 256D rectified-flow velocity model."""

    def __init__(self, interface_dim: int = 256, width: int = 512, depth: int = 6):
        super().__init__()
        self.time_dim = int(width)
        self.input = nn.Linear(int(interface_dim), int(width))
        self.time = nn.Sequential(
            nn.Linear(int(width), int(width)),
            nn.SiLU(),
            nn.Linear(int(width), int(width)),
        )
        self.blocks = nn.ModuleList(
            [ResidualPriorBlock(int(width)) for _ in range(int(depth))]
        )
        self.norm = nn.LayerNorm(int(width))
        self.output = nn.Linear(int(width), int(interface_dim))
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, value: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        hidden = self.input(value) + self.time(
            sinusoidal_time_embedding(time, self.time_dim)
        )
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(self.norm(hidden))


def build_prior(config: LatentPriorTradeoffConfig, device: torch.device) -> UnifiedLatentVelocityMLP:
    torch.manual_seed(int(config.prior_seed) + 101)
    if device.type == "cuda":
        torch.cuda.manual_seed(int(config.prior_seed) + 101)
    return UnifiedLatentVelocityMLP(
        config.interface_dim, config.prior_width, config.prior_depth
    ).to(device)


def deterministic_datasets(
    data_root: str | Path,
    image_size: int,
) -> tuple[ImageFolder, ImageFolder]:
    _, evaluation_transform = imagenette_transforms(image_size)
    root = Path(data_root)
    train = ImageFolder(str(root / "train"), transform=evaluation_transform)
    val = ImageFolder(str(root / "val"), transform=evaluation_transform)
    if train.class_to_idx != val.class_to_idx:
        raise RuntimeError("Imagenette train and val class mappings differ")
    if set(train.class_to_idx) != set(IMAGENETTE_SYNSET_TO_IMAGENET_INDEX):
        raise RuntimeError("unexpected Imagenette class mapping")
    train_paths = {str(Path(path).resolve()) for path, _ in train.samples}
    val_paths = {str(Path(path).resolve()) for path, _ in val.samples}
    overlap = train_paths.intersection(val_paths)
    if overlap:
        raise RuntimeError(f"train/val leakage detected for {len(overlap)} files")
    return train, val


def load_frozen_models(
    config: LatentPriorTradeoffConfig,
    device: torch.device,
) -> tuple[CompactImagenetteEncoder, ImagenetteConditionalUNet, dict]:
    state_path = config.frozen_run / "state.pt"
    if not state_path.is_file():
        raise FileNotFoundError(f"missing frozen checkpoint: {state_path}")
    # The locally produced checkpoint stores Path values inside its config.
    # Allow only that otherwise-safe type while retaining weights_only loading.
    with torch.serialization.safe_globals([PosixPath]):
        state = torch.load(state_path, map_location="cpu", weights_only=True)
    saved = state["config"]
    if int(saved["latent_dim"]) != int(config.latent_dim):
        raise RuntimeError("checkpoint latent_dim mismatch")
    if int(saved["seed"]) != int(config.frozen_seed):
        raise RuntimeError("checkpoint frozen_seed mismatch")
    encoder = CompactImagenetteEncoder(config.latent_dim, config.encoder_width)
    decoder = ImagenetteConditionalUNet(config.latent_dim, config.decoder_width)
    encoder.load_state_dict(state["encoder_ema"])
    decoder.load_state_dict(state["model_ema"])
    encoder.to(device).eval()
    decoder.to(device).eval()
    for module in (encoder, decoder):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    metadata = {
        "frozen_encoder_sha256": state_dict_sha256(encoder),
        "frozen_decoder_sha256": state_dict_sha256(decoder),
        "source_summary": state["summary"],
        "class_to_idx": state["class_to_idx"],
    }
    return encoder, decoder, metadata


@torch.no_grad()
def encode_dataset(
    encoder: nn.Module,
    dataset: ImageFolder,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=device.type == "cuda",
        persistent_workers=int(num_workers) > 0,
    )
    latents: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for images, batch_labels in loader:
        latents.append(encoder(images.to(device, non_blocking=True)).cpu())
        labels.append(batch_labels.cpu())
    paths = [str(Path(path).resolve()) for path, _ in dataset.samples]
    latent = torch.cat(latents)
    label = torch.cat(labels)
    if len(latent) != len(paths) or not torch.isfinite(latent).all():
        raise RuntimeError("invalid deterministic latent cache")
    return latent, label, paths


def cache_frozen_latents(
    encoder: nn.Module,
    train: ImageFolder,
    val: ImageFolder,
    config: LatentPriorTradeoffConfig,
    device: torch.device,
    output: Path | None,
) -> dict:
    cache_path = None if output is None else output / "latent_cache.pt"
    if cache_path is not None and cache_path.is_file():
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        expected = {
            "latent_dim": int(config.latent_dim),
            "frozen_seed": int(config.frozen_seed),
            "train_count": len(train),
            "val_count": len(val),
        }
        if any(int(cache[key]) != value for key, value in expected.items()):
            raise RuntimeError("latent cache metadata mismatch")
        return cache
    train_latent, train_labels, train_paths = encode_dataset(
        encoder,
        train,
        batch_size=config.encode_batch_size,
        num_workers=config.num_workers,
        device=device,
    )
    val_latent, val_labels, val_paths = encode_dataset(
        encoder,
        val,
        batch_size=config.encode_batch_size,
        num_workers=config.num_workers,
        device=device,
    )
    if set(train_paths).intersection(val_paths):
        raise RuntimeError("latent cache contains train/val path overlap")
    cache = {
        "latent_dim": int(config.latent_dim),
        "frozen_seed": int(config.frozen_seed),
        "train_count": len(train_latent),
        "val_count": len(val_latent),
        "train_latent": train_latent,
        "train_labels": train_labels,
        "val_latent": val_latent,
        "val_labels": val_labels,
        "train_path_sha256": path_list_sha256(train_paths),
        "val_path_sha256": path_list_sha256(val_paths),
        "train_latent_sha256": tensor_sha256(train_latent),
        "val_latent_sha256": tensor_sha256(val_latent),
    }
    if cache_path is not None:
        torch.save(cache, cache_path)
    return cache


@torch.no_grad()
def fixed_prior_validation_loss(
    model: nn.Module,
    val_latent: torch.Tensor,
    interface: OrthogonalLatentInterface,
    *,
    batch_size: int,
    seed: int,
) -> float:
    device = next(model.parameters()).device
    count = min(1_024, len(val_latent))
    data = val_latent[:count].to(device)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    base_noise = torch.randn((count, interface.interface_dim), device=device, generator=generator)
    time = torch.rand((count,), device=device, generator=generator)
    total = 0.0
    values = 0
    for indices in torch.arange(count, device=device).split(int(batch_size)):
        target_coordinates = data[indices]
        noise_coordinates = base_noise[indices, : interface.latent_dim]
        target = noise_coordinates - target_coordinates
        state = interface.embed(
            (1.0 - time[indices, None]) * target_coordinates
            + time[indices, None] * noise_coordinates
        )
        prediction = interface.recover(model(state, time[indices]))
        total += float(F.mse_loss(prediction, target, reduction="sum"))
        values += target.numel()
    return total / values


def train_prior(
    train_latent: torch.Tensor,
    val_latent: torch.Tensor,
    interface: OrthogonalLatentInterface,
    config: LatentPriorTradeoffConfig,
) -> tuple[EMA, pd.DataFrame, dict]:
    device = torch.device(config.device)
    model = build_prior(config, device)
    initial_hash = state_dict_sha256(model)
    ema = EMA(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    data = train_latent.to(device)
    index_generator = torch.Generator(device=device).manual_seed(config.prior_seed + 211)
    noise_generator = torch.Generator(device=device).manual_seed(config.prior_seed + 223)
    time_generator = torch.Generator(device=device).manual_seed(config.prior_seed + 227)
    stream_hashers = {
        "indices": hashlib.sha256(),
        "base_noise": hashlib.sha256(),
        "time": hashlib.sha256(),
    }
    rows: list[dict[str, float | int]] = []
    model.train()
    parameters = list(model.parameters())
    for step in range(1, int(config.prior_steps) + 1):
        indices = torch.randint(
            len(data),
            (int(config.prior_batch_size),),
            device=device,
            generator=index_generator,
        )
        target_coordinates = data[indices]
        base_noise = torch.randn(
            (len(indices), int(config.interface_dim)),
            device=device,
            generator=noise_generator,
        )
        noise_coordinates = base_noise[:, : int(config.latent_dim)]
        time = torch.rand((len(indices),), device=device, generator=time_generator)
        mixed_coordinates = (
            (1.0 - time[:, None]) * target_coordinates
            + time[:, None] * noise_coordinates
        )
        state = interface.embed(mixed_coordinates)
        target_velocity = noise_coordinates - target_coordinates
        predicted_velocity = interface.recover(model(state, time))
        loss = F.mse_loss(predicted_velocity, target_velocity)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters, float(config.gradient_clip)
        )
        optimizer.step()
        ema.update(model, config.ema_decay, step)
        if step <= 32:
            stream_hashers["indices"].update(
                indices.detach().cpu().contiguous().numpy().tobytes()
            )
            stream_hashers["base_noise"].update(
                base_noise.detach().cpu().contiguous().numpy().tobytes()
            )
            stream_hashers["time"].update(
                time.detach().cpu().contiguous().numpy().tobytes()
            )
        if step == 1 or step % int(config.log_every) == 0 or step == int(config.prior_steps):
            ema.module.eval()
            validation_loss = fixed_prior_validation_loss(
                ema.module,
                val_latent,
                interface,
                batch_size=config.eval_batch_size,
                seed=config.prior_seed + 311,
            )
            record = {
                "step": step,
                "train_flow_mse": float(loss.detach()),
                "heldout_flow_mse": validation_loss,
                "gradient_norm": float(gradient_norm),
            }
            if not all(math.isfinite(float(value)) for value in record.values()):
                raise FloatingPointError("non-finite prior training metric")
            rows.append(record)
            print(
                json.dumps(
                    {
                        "latent_dim": int(config.latent_dim),
                        "frozen_seed": int(config.frozen_seed),
                        **record,
                    }
                ),
                flush=True,
            )
            ema.module.eval()
            model.train()
    ema.module.eval()
    metadata = {
        "prior_initial_sha256": initial_hash,
        "prior_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "stream_indices_first_32_sha256": stream_hashers["indices"].hexdigest(),
        "stream_base_noise_first_32_sha256": stream_hashers["base_noise"].hexdigest(),
        "stream_time_first_32_sha256": stream_hashers["time"].hexdigest(),
    }
    return ema, pd.DataFrame(rows), metadata


@torch.no_grad()
def sample_prior_coordinates(
    model: nn.Module,
    interface: OrthogonalLatentInterface,
    count: int,
    steps: int,
    *,
    seed: int,
    batch_size: int,
) -> torch.Tensor:
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(int(seed))
    results: list[torch.Tensor] = []
    remaining = int(count)
    grid = descending_time_grid(int(steps), device=device)
    while remaining > 0:
        current_count = min(int(batch_size), remaining)
        base_noise = torch.randn(
            (current_count, interface.interface_dim),
            device=device,
            generator=generator,
        )
        state = interface.embed(base_noise[:, : interface.latent_dim])
        for current, following in zip(grid[:-1], grid[1:]):
            time = torch.full((current_count,), float(current), device=device)
            velocity = interface.project(model(state, time))
            state = state + (following - current) * velocity
        results.append(interface.recover(state).cpu())
        remaining -= current_count
    result = torch.cat(results)
    if not torch.isfinite(result).all():
        raise FloatingPointError("non-finite sampled latent")
    return result


def covariance_statistics(real: torch.Tensor, generated: torch.Tensor) -> dict[str, float]:
    real64 = real.double()
    generated64 = generated.double()
    real_centered = real64 - real64.mean(dim=0, keepdim=True)
    generated_centered = generated64 - generated64.mean(dim=0, keepdim=True)
    real_covariance = real_centered.T @ real_centered / max(len(real64) - 1, 1)
    generated_covariance = (
        generated_centered.T @ generated_centered / max(len(generated64) - 1, 1)
    )
    real_eigenvalues = torch.linalg.eigvalsh(real_covariance).clamp_min(0.0)
    generated_eigenvalues = torch.linalg.eigvalsh(generated_covariance).clamp_min(0.0)

    def effective_rank(values: torch.Tensor) -> float:
        return float(values.sum().square() / values.square().sum().clamp_min(1e-18))

    relative_covariance_error = float(
        torch.linalg.vector_norm(generated_covariance - real_covariance)
        / torch.linalg.vector_norm(real_covariance).clamp_min(1e-18)
    )
    eigenvalue_overlap = float(
        torch.minimum(real_eigenvalues, generated_eigenvalues).sum()
        / torch.maximum(real_eigenvalues, generated_eigenvalues).sum().clamp_min(1e-18)
    )
    return {
        "latent_mean_relative_error": float(
            torch.linalg.vector_norm(generated64.mean(dim=0) - real64.mean(dim=0))
            / real64.square().mean().sqrt().clamp_min(1e-18)
        ),
        "latent_covariance_relative_error": relative_covariance_error,
        "latent_covariance_eigenvalue_overlap": eigenvalue_overlap,
        "real_latent_effective_rank": effective_rank(real_eigenvalues),
        "generated_latent_effective_rank": effective_rank(generated_eigenvalues),
    }


def sliced_wasserstein_distance(
    real: torch.Tensor,
    generated: torch.Tensor,
    *,
    directions: int,
    seed: int,
) -> float:
    if len(real) != len(generated):
        count = min(len(real), len(generated))
        real = real[:count]
        generated = generated[:count]
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    projection = torch.randn(
        (real.shape[1], int(directions)), generator=generator, dtype=torch.float64
    )
    projection = F.normalize(projection, dim=0)
    real_projected = torch.sort(real.double() @ projection, dim=0).values
    generated_projected = torch.sort(generated.double() @ projection, dim=0).values
    return float((real_projected - generated_projected).square().mean().sqrt())


def latent_distribution_metrics(
    real: torch.Tensor,
    generated: torch.Tensor,
    config: LatentPriorTradeoffConfig,
) -> dict[str, float]:
    metrics = covariance_statistics(real, generated)
    metrics["latent_sliced_wasserstein"] = sliced_wasserstein_distance(
        real,
        generated,
        directions=config.sliced_directions,
        seed=config.prior_seed + 701,
    )
    return metrics


def restricted_class_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_to_idx: dict[str, int],
) -> dict[str, float | list[float]]:
    imagenet_indices = torch.empty(len(class_to_idx), dtype=torch.long)
    for synset, local_index in class_to_idx.items():
        imagenet_indices[local_index] = IMAGENETTE_SYNSET_TO_IMAGENET_INDEX[synset]
    local_logits = logits[:, imagenet_indices]
    probabilities = local_logits.softmax(dim=1).mean(dim=0)
    predictions = local_logits.argmax(dim=1)
    histogram = torch.bincount(predictions, minlength=len(class_to_idx)).float()
    histogram = histogram / histogram.sum()
    target = torch.bincount(labels, minlength=len(class_to_idx)).float()
    target = target / target.sum()
    entropy = -(histogram * histogram.clamp_min(1e-12).log()).sum()
    return {
        "predicted_class_entropy": float(entropy),
        "predicted_effective_classes": float(entropy.exp()),
        "predicted_min_class_fraction": float(histogram.min()),
        "predicted_class_tv": float(0.5 * (histogram - target).abs().sum()),
        "mean_restricted_probability_entropy": float(
            -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
        ),
        "predicted_class_histogram": histogram.tolist(),
    }


@torch.no_grad()
def compute_real_features(
    dataset: ImageFolder,
    evaluator: nn.Module,
    *,
    count: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=device.type == "cuda",
        persistent_workers=int(num_workers) > 0,
    )
    images: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    features: list[torch.Tensor] = []
    seen = 0
    for batch_images, batch_labels in loader:
        if seen >= int(count):
            break
        take = min(len(batch_images), int(count) - seen)
        batch_images = batch_images[:take]
        batch_labels = batch_labels[:take]
        feature, _ = evaluator(batch_images.to(device, non_blocking=True))
        images.append(batch_images.cpu())
        labels.append(batch_labels.cpu())
        features.append(feature.cpu())
        seen += take
    return torch.cat(images), torch.cat(labels), torch.cat(features)


@torch.no_grad()
def evaluate_rollout(
    mode: str,
    decoder: nn.Module,
    conditions: torch.Tensor,
    real_images: torch.Tensor,
    real_labels: torch.Tensor,
    real_features: torch.Tensor,
    evaluator: nn.Module,
    class_to_idx: dict[str, int],
    config: LatentPriorTradeoffConfig,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    if mode not in ROLLOUT_MODES:
        raise ValueError(f"unknown rollout mode: {mode}")
    if len(conditions) != len(real_images):
        raise ValueError("condition and evaluation sample counts differ")
    device = next(decoder.parameters()).device
    # Match the source experiment exactly and share pixel noise across every
    # capacity, frozen seed, and rollout mode.
    generator = torch.Generator(device=device).manual_seed(33_001)
    feature_parts: list[torch.Tensor] = []
    logit_parts: list[torch.Tensor] = []
    preview: list[torch.Tensor] = []
    pixel_squared_error = 0.0
    pixel_values = 0
    for start in range(0, len(conditions), int(config.eval_batch_size)):
        end = min(start + int(config.eval_batch_size), len(conditions))
        condition = conditions[start:end].to(device)
        initial = torch.randn(
            (end - start, 3, config.image_size, config.image_size),
            device=device,
            generator=generator,
        )
        generated = conditional_euler_sample(
            decoder, initial, condition, config.pixel_ode_steps
        ).clamp(-1.0, 1.0)
        feature, logits = evaluator(generated)
        feature_parts.append(feature.cpu())
        logit_parts.append(logits.cpu())
        if len(preview) < 1:
            preview.append(generated[: min(32, len(generated))].cpu())
        if mode == "oracle":
            target = real_images[start:end].to(device)
            pixel_squared_error += float(F.mse_loss(generated, target, reduction="sum"))
            pixel_values += target.numel()
    generated_features = torch.cat(feature_parts)
    logits = torch.cat(logit_parts)
    metrics: dict = {
        "mode": mode,
        "feature_fid": frechet_distance(real_features, generated_features),
        **restricted_class_metrics(logits, real_labels, class_to_idx),
    }
    if len(real_features) >= 1_024:
        metrics["feature_fid_first_1024"] = frechet_distance(
            real_features[:1_024], generated_features[:1_024]
        )
    if mode == "oracle":
        metrics["source_pixel_mse"] = pixel_squared_error / pixel_values
        imagenet_indices = torch.empty(len(class_to_idx), dtype=torch.long)
        for synset, local_index in class_to_idx.items():
            imagenet_indices[local_index] = IMAGENETTE_SYNSET_TO_IMAGENET_INDEX[synset]
        metrics["source_class_match"] = float(
            (logits[:, imagenet_indices].argmax(dim=1) == real_labels).float().mean()
        )
    return metrics, generated_features, torch.cat(preview)


def _prepare_output(config: LatentPriorTradeoffConfig) -> Path | None:
    if not config.save:
        return None
    output = config.result_dir
    if output.exists() and config.overwrite:
        shutil.rmtree(output)
    if output.exists() and not config.resume:
        raise FileExistsError(f"result directory already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    values = asdict(config)
    if config_path.is_file():
        existing = json.loads(config_path.read_text())
        normalized = json.loads(json.dumps(values, default=_json_ready))
        ignored = {"overwrite", "resume", "device"}
        if any(existing.get(key) != value for key, value in normalized.items() if key not in ignored):
            raise RuntimeError("resume config does not match existing run")
    else:
        _write_json(config_path, values)
    return output


def run(config: LatentPriorTradeoffConfig) -> Path | None:
    config.validate()
    configure_fp32(config.prior_seed)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output = _prepare_output(config)
    if output is not None and (output / "summary.json").is_file() and config.resume:
        print(f"complete run already exists: {output}", flush=True)
        return output

    train_dataset, val_dataset = deterministic_datasets(config.data_root, config.image_size)
    encoder, decoder, frozen_metadata = load_frozen_models(config, device)
    frozen_hashes_before = {
        "encoder": state_dict_sha256(encoder),
        "decoder": state_dict_sha256(decoder),
    }
    cache = cache_frozen_latents(
        encoder, train_dataset, val_dataset, config, device, output
    )
    basis = fixed_orthogonal_basis(config.interface_dim, config.basis_seed)
    interface = OrthogonalLatentInterface(config.latent_dim, basis).to(device)
    roundtrip = interface.recover(
        interface.embed(cache["val_latent"][:64].to(device))
    ).cpu()
    roundtrip_error = float((roundtrip - cache["val_latent"][:64]).abs().max())
    if roundtrip_error > 1e-6:
        raise RuntimeError(f"orthogonal interface roundtrip failed: {roundtrip_error}")

    prior_state_path = None if output is None else output / "prior_state.pt"
    history_path = None if output is None else output / "history.csv"
    if prior_state_path is not None and prior_state_path.is_file() and config.resume:
        saved_prior = torch.load(prior_state_path, map_location="cpu", weights_only=True)
        prior = build_prior(config, device)
        prior.load_state_dict(saved_prior["prior_ema"])
        prior.eval()
        for parameter in prior.parameters():
            parameter.requires_grad_(False)
        prior_ema = EMA(prior)
        prior_ema.module.load_state_dict(prior.state_dict())
        history = pd.read_csv(history_path)
        training_metadata = saved_prior["training_metadata"]
    else:
        prior_ema, history, training_metadata = train_prior(
            cache["train_latent"], cache["val_latent"], interface, config
        )
        if prior_state_path is not None:
            history.to_csv(history_path, index=False)
            torch.save(
                {
                    "prior_ema": prior_ema.module.state_dict(),
                    "training_metadata": training_metadata,
                },
                prior_state_path,
            )

    prior = prior_ema.module.to(device).eval()
    for parameter in prior.parameters():
        parameter.requires_grad_(False)
    count = min(int(config.quality_count), len(val_dataset))
    evaluator = ResNet18Evaluator().to(device).eval()
    eval_subset = fixed_eval_subset(val_dataset, count, seed=2_027)
    eval_indices = torch.as_tensor(eval_subset.indices, dtype=torch.long)
    real_images, real_labels, real_features = compute_real_features(
        eval_subset,
        evaluator,
        count=count,
        batch_size=config.eval_batch_size,
        num_workers=config.num_workers,
        device=device,
    )
    val_conditions = cache["val_latent"][eval_indices]
    empirical_generator = torch.Generator(device="cpu").manual_seed(config.prior_seed + 1_101)
    empirical_indices = torch.randint(
        len(cache["train_latent"]), (count,), generator=empirical_generator
    )
    empirical_conditions = cache["train_latent"][empirical_indices]
    prior_conditions = sample_prior_coordinates(
        prior,
        interface,
        count,
        config.prior_ode_steps,
        seed=config.prior_seed + 1_201,
        batch_size=config.prior_batch_size,
    )
    gaussian_generator = torch.Generator(device="cpu").manual_seed(config.prior_seed + 1_201)
    gaussian_base = torch.randn((count, config.interface_dim), generator=gaussian_generator)
    gaussian_conditions = gaussian_base[:, : config.latent_dim]
    conditions = {
        "oracle": val_conditions,
        "empirical": empirical_conditions,
        "prior": prior_conditions,
        "gaussian": gaussian_conditions,
    }
    rollout_rows: list[dict] = []
    previews: list[torch.Tensor] = []
    for mode in ROLLOUT_MODES:
        metric_path = None if output is None else output / f"rollout_{mode}.json"
        feature_path = None if output is None else output / f"features_{mode}.pt"
        preview_path = None if output is None else output / f"samples_{mode}.png"
        if (
            config.resume
            and metric_path is not None
            and metric_path.is_file()
            and feature_path is not None
            and feature_path.is_file()
        ):
            metrics = json.loads(metric_path.read_text())
        else:
            metrics, generated_features, preview = evaluate_rollout(
                mode,
                decoder,
                conditions[mode],
                real_images,
                real_labels,
                real_features,
                evaluator,
                frozen_metadata["class_to_idx"],
                config,
            )
            if metric_path is not None:
                _write_json(metric_path, metrics)
                torch.save(
                    {
                        "real_features": real_features,
                        "generated_features": generated_features,
                    },
                    feature_path,
                )
                grid = make_grid(preview.add(1.0).mul(0.5), nrow=8)
                save_image(grid, preview_path)
            previews.append(preview)
        rollout_rows.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False), flush=True)

    rollout = pd.DataFrame(rollout_rows).set_index("mode")
    latent_metrics = latent_distribution_metrics(
        val_conditions, prior_conditions, config
    )
    frozen_hashes_after = {
        "encoder": state_dict_sha256(encoder),
        "decoder": state_dict_sha256(decoder),
    }
    if frozen_hashes_after != frozen_hashes_before:
        raise RuntimeError("frozen encoder or decoder changed during prior experiment")
    summary = {
        "latent_dim": int(config.latent_dim),
        "frozen_seed": int(config.frozen_seed),
        "prior_replicate": int(config.prior_replicate),
        "prior_seed": int(config.prior_seed),
        "train_count": int(cache["train_count"]),
        "val_count": int(cache["val_count"]),
        "quality_count": count,
        "oracle_feature_fid": float(rollout.loc["oracle", "feature_fid"]),
        "empirical_feature_fid": float(rollout.loc["empirical", "feature_fid"]),
        "end_to_end_feature_fid": float(rollout.loc["prior", "feature_fid"]),
        "gaussian_feature_fid": float(rollout.loc["gaussian", "feature_fid"]),
        "total_prior_gap": float(
            rollout.loc["prior", "feature_fid"] - rollout.loc["oracle", "feature_fid"]
        ),
        "modeling_gap": float(
            rollout.loc["prior", "feature_fid"] - rollout.loc["empirical", "feature_fid"]
        ),
        "oracle_source_pixel_mse": float(rollout.loc["oracle", "source_pixel_mse"]),
        "oracle_source_class_match": float(rollout.loc["oracle", "source_class_match"]),
        "final_train_flow_mse": float(history.iloc[-1].train_flow_mse),
        "heldout_prior_flow_mse": float(history.iloc[-1].heldout_flow_mse),
        "orthogonal_roundtrip_max_abs": roundtrip_error,
        "frozen_hashes_unchanged": frozen_hashes_before == frozen_hashes_after,
        "train_path_sha256": cache["train_path_sha256"],
        "val_path_sha256": cache["val_path_sha256"],
        "train_latent_sha256": cache["train_latent_sha256"],
        "val_latent_sha256": cache["val_latent_sha256"],
        "frozen_encoder_sha256": frozen_metadata["frozen_encoder_sha256"],
        "frozen_decoder_sha256": frozen_metadata["frozen_decoder_sha256"],
        "source_conditional_feature_fid_1024": float(
            frozen_metadata["source_summary"]["conditional_feature_fid"]
        ),
        "source_final_validation_velocity_mse": float(
            frozen_metadata["source_summary"]["final_validation_velocity_mse"]
        ),
        **training_metadata,
        **latent_metrics,
    }
    if count >= 1_024:
        recomputed_1024 = float(rollout.loc["oracle", "feature_fid_first_1024"])
        summary["oracle_feature_fid_1024_recomputed"] = recomputed_1024
        summary["source_oracle_fid_1024_abs_diff"] = abs(
            recomputed_1024 - summary["source_conditional_feature_fid_1024"]
        )
    if not all(
        math.isfinite(float(value))
        for value in summary.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ):
        raise FloatingPointError("non-finite final summary metric")
    if output is not None:
        rollout.reset_index().to_csv(output / "rollout_metrics.csv", index=False)
        _write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--latent-dim", type=int, choices=CAPACITIES, default=64)
    parser.add_argument("--frozen-seed", type=int, default=0)
    parser.add_argument("--prior-replicate", type=int, default=0)
    parser.add_argument("--prior-width", type=int, default=512)
    parser.add_argument("--prior-depth", type=int, default=6)
    parser.add_argument("--prior-steps", type=int, default=20_000)
    parser.add_argument("--prior-batch-size", type=int, default=512)
    parser.add_argument("--quality-count", type=int, default=3_925)
    parser.add_argument("--prior-ode-steps", type=int, default=100)
    parser.add_argument("--pixel-ode-steps", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> Path | None:
    args = build_parser().parse_args(argv)
    return run(
        LatentPriorTradeoffConfig(
            data_root=args.data_root,
            checkpoint_root=args.checkpoint_root,
            output_root=args.output_root,
            latent_dim=args.latent_dim,
            frozen_seed=args.frozen_seed,
            prior_replicate=args.prior_replicate,
            prior_width=args.prior_width,
            prior_depth=args.prior_depth,
            prior_steps=args.prior_steps,
            prior_batch_size=args.prior_batch_size,
            quality_count=args.quality_count,
            prior_ode_steps=args.prior_ode_steps,
            pixel_ode_steps=args.pixel_ode_steps,
            num_workers=args.num_workers,
            device=args.device,
            overwrite=args.overwrite,
            resume=args.resume,
            save=not args.no_save,
        )
    )


if __name__ == "__main__":
    main()
