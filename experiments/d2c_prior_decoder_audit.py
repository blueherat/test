"""Frozen external audit of the official D2C FFHQ-256 generator.

The audit never trains a model.  It compares empirical encoder latents,
official latent-diffusion samples, and a Gaussian with the empirical sample
covariance.  Decoder output randomness is reset to the same stream for every
latent source and repeated across seeds.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import pil_to_tensor
from torchvision.utils import make_grid, save_image

from experiments.diffae_prior_decoder_audit import (
    build_fid_feature_extractor,
    configure_fp32,
    inception_features,
    kernel_inception_distance,
    mean_cosine_distance,
    score_decoded_feature_sets,
    standardized_sliced_wasserstein,
)


DEFAULT_D2C_REPO = Path.home() / "data/external/d2c"
DEFAULT_CHECKPOINT = (
    Path.home() / "data/checkpoints/d2c/checkpoints/ffhq_256/model.ckpt"
)
DEFAULT_PARQUET_ROOT = Path.home() / "data/datasets/ffhq256-hf/data"
DEFAULT_RESULT_ROOT = Path.home() / "data/results/d2c_gate_b"


@dataclass(frozen=True)
class D2CAuditConfig:
    d2c_repo: Path = DEFAULT_D2C_REPO
    checkpoint: Path = DEFAULT_CHECKPOINT
    parquet_root: Path = DEFAULT_PARQUET_ROOT
    result_root: Path = DEFAULT_RESULT_ROOT
    device: str = "cuda:2"
    count: int = 2_000
    fit_count: int = 2_000
    batch_size: int = 4
    latent_batch_size: int = 32
    prior_skip: int = 100
    data_start: int = 0
    latent_seed: int = 20_260_722
    decoder_seed: int = 30_260_722
    metric_seed: int = 40_260_722

    def validate(
        self,
        *,
        require_checkpoint: bool = True,
        require_data: bool = False,
    ) -> None:
        if not self.d2c_repo.is_dir():
            raise FileNotFoundError(f"missing D2C repository: {self.d2c_repo}")
        if require_checkpoint and not self.checkpoint.is_file():
            raise FileNotFoundError(f"missing D2C checkpoint: {self.checkpoint}")
        if require_data and not list(self.parquet_root.glob("*.parquet")):
            raise FileNotFoundError(f"no FFHQ parquet shards in {self.parquet_root}")
        if self.count < 2 or self.fit_count < 2:
            raise ValueError("count and fit_count must both be >= 2")
        if self.batch_size < 1 or self.latent_batch_size < 1:
            raise ValueError("batch sizes must be positive")
        if self.prior_skip < 1 or self.prior_skip > 1_000:
            raise ValueError("prior_skip must be in [1, 1000]")
        if self.data_start < 0:
            raise ValueError("data_start must be non-negative")


@dataclass
class FrozenD2C:
    model: Any
    config: Any
    metadata: dict[str, Any]


class ImageBytesDataset(Dataset):
    def __init__(self, records: Sequence[bytes], resolution: int = 256):
        self.records = list(records)
        self.resolution = int(resolution)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(io.BytesIO(self.records[index])) as image:
            image = image.convert("RGB")
            if image.size != (self.resolution, self.resolution):
                image = image.resize(
                    (self.resolution, self.resolution), Image.Resampling.BILINEAR
                )
            return pil_to_tensor(image).float().div_(255)


def _dict_to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _dict_to_namespace(item) for key, item in value.items()})
    return value


def _sha256(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _select_cuda_device(device: str) -> torch.device:
    selected = torch.device(device)
    if selected.type != "cuda":
        raise ValueError("the official D2C implementation requires a CUDA device")
    torch.cuda.set_device(selected)
    return selected


def _load_official_config(repo: Path) -> Any:
    path = repo / "configs/ffhq_256.yml"
    raw = yaml.safe_load(path.read_text())
    return _dict_to_namespace(raw)


def _validate_and_load_component(
    module: torch.nn.Module,
    state: dict[str, torch.Tensor],
    *,
    name: str,
) -> None:
    expected = module.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    mismatched = sorted(
        key
        for key in set(expected) & set(state)
        if tuple(expected[key].shape) != tuple(state[key].shape)
    )
    if missing or unexpected or mismatched:
        raise RuntimeError(
            f"incompatible D2C {name} checkpoint: missing={missing[:5]}, "
            f"unexpected={unexpected[:5]}, shape_mismatch={mismatched[:5]}"
        )
    module.load_state_dict(state, strict=True)


def load_frozen_d2c(config: D2CAuditConfig) -> FrozenD2C:
    config.validate(require_checkpoint=True)
    device = _select_cuda_device(config.device)
    configure_fp32(config.latent_seed)
    if str(config.d2c_repo) not in sys.path:
        sys.path.insert(0, str(config.d2c_repo))
    from d2c import D2C  # type: ignore

    official_config = _load_official_config(config.d2c_repo)
    model = D2C(SimpleNamespace(), official_config)
    checkpoint = torch.load(config.checkpoint, map_location="cpu", weights_only=False)
    required = {"autoencoder", "diffusion", "latent_mod"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise RuntimeError(f"D2C checkpoint must contain {sorted(required)}")

    autoencoder_state = {
        key: value
        for key, value in checkpoint["autoencoder"].items()
        if not key.startswith("encoder_q.fc") and not key.startswith("encoder_k.fc")
    }
    _validate_and_load_component(
        model.autoencoder, autoencoder_state, name="autoencoder"
    )
    _validate_and_load_component(model.diffusion, checkpoint["diffusion"], name="diffusion")
    _validate_and_load_component(
        model.latent_mod, checkpoint["latent_mod"], name="latent_mod"
    )
    model.eval()
    modules = (model.autoencoder, model.diffusion, model.latent_mod)
    for module in modules:
        module.requires_grad_(False)
    if any(parameter.requires_grad for module in modules for parameter in module.parameters()):
        raise RuntimeError("D2C audit requires all model parameters to be frozen")
    if tuple(model.latent_size) != (8, 32, 32):
        raise RuntimeError(f"unexpected D2C latent shape: {model.latent_size}")

    counts = {
        name: sum(parameter.numel() for parameter in module.parameters())
        for name, module in zip(("autoencoder", "diffusion", "latent_mod"), modules)
    }
    metadata = {
        "checkpoint_sha256": _sha256(config.checkpoint),
        "checkpoint_size_bytes": config.checkpoint.stat().st_size,
        "latent_shape": list(model.latent_size),
        "component_parameter_counts": counts,
        "total_parameter_count": sum(counts.values()),
        "device": str(device),
        "torch_version": str(torch.__version__),
        "fp32": True,
        "tf32": False,
        "all_parameters_frozen": True,
    }
    return FrozenD2C(model=model, config=official_config, metadata=metadata)


def read_parquet_image_bytes(root: str | Path, start: int, count: int) -> list[bytes]:
    import pyarrow.parquet as pq

    if start < 0 or count < 1:
        raise ValueError("start must be non-negative and count must be positive")
    paths = sorted(Path(root).glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquet shards found in {root}")
    stop = start + count
    cursor = 0
    records: list[bytes] = []
    for path in paths:
        parquet = pq.ParquetFile(path)
        rows = parquet.metadata.num_rows
        shard_stop = cursor + rows
        if shard_stop <= start:
            cursor = shard_stop
            continue
        if cursor >= stop:
            break
        table = parquet.read(columns=["image"])
        local_start = max(start - cursor, 0)
        local_stop = min(stop - cursor, rows)
        column = table.column("image")
        for index in range(local_start, local_stop):
            item = column[index].as_py()
            value = item.get("bytes") if isinstance(item, dict) else item
            if not isinstance(value, (bytes, bytearray)):
                raise TypeError(f"unsupported image value in {path}: {type(value)}")
            records.append(bytes(value))
        cursor = shard_stop
    if len(records) != count:
        raise IndexError(f"requested {count} images from {start}, found {len(records)}")
    return records


def split_fit_evaluation_records(
    records: Sequence[bytes], fit_count: int, evaluation_count: int
) -> tuple[list[bytes], list[bytes]]:
    required = int(fit_count) + int(evaluation_count)
    if len(records) < required:
        raise ValueError(f"need {required} records, received {len(records)}")
    fit = list(records[:fit_count])
    evaluation = list(records[fit_count:required])
    return fit, evaluation


def _image_loader(records: Sequence[bytes], batch_size: int) -> DataLoader:
    return DataLoader(
        ImageBytesDataset(records),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )


@torch.inference_mode()
def encode_records(
    frozen: FrozenD2C,
    records: Sequence[bytes],
    *,
    batch_size: int,
) -> torch.Tensor:
    latents = []
    for images in _image_loader(records, batch_size):
        latent = frozen.model.image_to_latent(images)
        latents.append(latent.detach().float().cpu())
    result = torch.cat(latents)
    if result.shape != (len(records), 8, 32, 32):
        raise RuntimeError(f"unexpected empirical latent shape: {tuple(result.shape)}")
    if not torch.isfinite(result).all():
        raise FloatingPointError("non-finite empirical D2C latent")
    return result


@torch.inference_mode()
def sample_prior_latents(
    frozen: FrozenD2C,
    count: int,
    *,
    skip: int,
    seed: int,
    batch_size: int,
) -> torch.Tensor:
    configure_fp32(seed)
    latents = []
    for begin in range(0, count, batch_size):
        size = min(batch_size, count - begin)
        latent = frozen.model.sample_latent(size, skip=skip)
        latents.append(latent.detach().float().cpu())
    result = torch.cat(latents)
    if result.shape != (count, 8, 32, 32) or not torch.isfinite(result).all():
        raise RuntimeError("invalid learned-prior D2C latent sample")
    return result


def sample_empirical_covariance_gaussian(
    fit_latents: torch.Tensor,
    count: int,
    *,
    seed: int,
    device: str | torch.device = "cpu",
    batch_size: int = 128,
) -> torch.Tensor:
    """Sample N(mean(X), cov(X)) without materializing a d-by-d covariance."""

    if fit_latents.ndim < 2 or len(fit_latents) < 2 or count < 1:
        raise ValueError("Gaussian fitting requires at least two latent samples")
    shape = fit_latents.shape[1:]
    flat = fit_latents.float().reshape(len(fit_latents), -1)
    mean = flat.mean(dim=0, keepdim=True)
    centered = flat - mean
    selected = torch.device(device)
    centered = centered.to(selected)
    mean = mean.to(selected)
    generator = torch.Generator(device=selected).manual_seed(int(seed))
    scale = math.sqrt(max(len(centered) - 1, 1))
    chunks = []
    for begin in range(0, count, batch_size):
        size = min(batch_size, count - begin)
        coefficients = torch.randn(
            size, len(centered), generator=generator, device=selected
        )
        sample = coefficients @ centered / scale + mean
        chunks.append(sample.cpu())
    result = torch.cat(chunks).reshape(count, *shape)
    if not torch.isfinite(result).all():
        raise FloatingPointError("non-finite covariance-matched Gaussian latent")
    return result


def _paired_metric_subset(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    max_samples: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    count = min(len(reference), len(candidate), int(max_samples))
    if count < 2:
        raise ValueError("distribution metrics require at least two samples")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    left_indices = torch.randperm(len(reference), generator=generator)[:count]
    if len(reference) == len(candidate):
        right_indices = left_indices
    else:
        right_indices = torch.randperm(len(candidate), generator=generator)[:count]
    return reference[left_indices], candidate[right_indices]


def covariance_relative_error_from_gram(
    reference: torch.Tensor, candidate: torch.Tensor
) -> float:
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("Gram covariance metric expects equal-sized matrices")
    left = reference.float() - reference.float().mean(dim=0, keepdim=True)
    right = candidate.float() - candidate.float().mean(dim=0, keepdim=True)
    scale = float(max(len(left) - 1, 1))
    left_gram = left @ left.T / scale
    right_gram = right @ right.T / scale
    cross_gram = left @ right.T / scale
    left_sq = left_gram.square().sum().double()
    right_sq = right_gram.square().sum().double()
    cross_sq = cross_gram.square().sum().double()
    numerator_sq = (left_sq + right_sq - 2 * cross_sq).clamp_min(0)
    return float(numerator_sq.sqrt() / left_sq.sqrt().clamp_min(1e-30))


def effective_rank_from_gram(values: torch.Tensor) -> float:
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("effective rank expects a sample matrix")
    centered = values.float() - values.float().mean(dim=0, keepdim=True)
    gram = (centered @ centered.T / max(len(centered) - 1, 1)).double()
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0)
    probabilities = eigenvalues / eigenvalues.sum().clamp_min(1e-30)
    entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum()
    return float(entropy.exp())


def _project_for_c2st(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    dimensions: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = reference.mean(dim=0, keepdim=True)
    std = reference.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    left = (reference - mean) / std
    right = (candidate - mean) / std
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    projection = torch.randn(reference.shape[1], dimensions, generator=generator)
    projection /= projection.norm(dim=0, keepdim=True).clamp_min(1e-12)
    return left @ projection, right @ projection


def _c2st(reference: torch.Tensor, candidate: torch.Tensor, *, seed: int) -> dict[str, float]:
    values = torch.cat((reference, candidate), dim=0).numpy()
    labels = np.concatenate((np.zeros(len(reference)), np.ones(len(candidate))))
    train_x, test_x, train_y, test_y = train_test_split(
        values, labels, test_size=0.35, random_state=int(seed), stratify=labels
    )
    scaler = StandardScaler().fit(train_x)
    classifier = LogisticRegression(
        max_iter=2_000, solver="lbfgs", random_state=int(seed)
    ).fit(scaler.transform(train_x), train_y)
    probability = classifier.predict_proba(scaler.transform(test_x))[:, 1]
    prediction = probability >= 0.5
    return {
        "projected_c2st_accuracy": float(accuracy_score(test_y, prediction)),
        "projected_c2st_auc": float(roc_auc_score(test_y, probability)),
    }


def high_dimensional_latent_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    seed: int,
    max_samples: int = 1_024,
    projection_dimensions: int = 256,
) -> dict[str, float]:
    left, right = _paired_metric_subset(
        reference.reshape(len(reference), -1),
        candidate.reshape(len(candidate), -1),
        max_samples=max_samples,
        seed=seed,
    )
    mean_scale = left.double().std(dim=0, unbiased=False).norm().clamp_min(1e-30)
    projected_left, projected_right = _project_for_c2st(
        left, right, dimensions=projection_dimensions, seed=seed + 1
    )
    metrics = {
        "metric_sample_count": len(left),
        "projection_dimensions": projection_dimensions,
        "standardized_swd": standardized_sliced_wasserstein(
            left, right, directions=projection_dimensions, seed=seed + 2
        ),
        "mean_shift_relative": float(
            (left.double().mean(dim=0) - right.double().mean(dim=0)).norm()
            / mean_scale
        ),
        "covariance_relative_error_gram": covariance_relative_error_from_gram(
            left, right
        ),
        "reference_effective_rank_gram": effective_rank_from_gram(left),
        "candidate_effective_rank_gram": effective_rank_from_gram(right),
        **_c2st(projected_left, projected_right, seed=seed + 3),
    }
    if not all(math.isfinite(float(value)) for value in metrics.values()):
        raise FloatingPointError("non-finite D2C latent metric")
    return metrics


def _cache_stem(config: D2CAuditConfig) -> str:
    return f"n{config.count}_fit{config.fit_count}_start{config.data_start}"


def run_cache_empirical(config: D2CAuditConfig) -> Path:
    config.validate(require_checkpoint=True, require_data=True)
    total = config.fit_count + config.count
    records = read_parquet_image_bytes(config.parquet_root, config.data_start, total)
    fit_records, evaluation_records = split_fit_evaluation_records(
        records, config.fit_count, config.count
    )
    frozen = load_frozen_d2c(config)
    fit = encode_records(frozen, fit_records, batch_size=config.batch_size)
    evaluation = encode_records(frozen, evaluation_records, batch_size=config.batch_size)
    config.result_root.mkdir(parents=True, exist_ok=True)
    output = config.result_root / f"empirical_{_cache_stem(config)}.pt"
    torch.save(
        {
            "fit": fit,
            "evaluation": evaluation,
            "fit_indices": torch.arange(config.data_start, config.data_start + config.fit_count),
            "evaluation_indices": torch.arange(
                config.data_start + config.fit_count,
                config.data_start + config.fit_count + config.count,
            ),
            "checkpoint_metadata": frozen.metadata,
        },
        output,
    )
    print(output, flush=True)
    return output


def run_latent_audit(config: D2CAuditConfig) -> Path:
    empirical_path = config.result_root / f"empirical_{_cache_stem(config)}.pt"
    if not empirical_path.is_file():
        raise FileNotFoundError(f"missing {empirical_path}; run cache-empirical first")
    state = torch.load(empirical_path, map_location="cpu", weights_only=True)
    frozen = load_frozen_d2c(config)
    prior = sample_prior_latents(
        frozen,
        config.count,
        skip=config.prior_skip,
        seed=config.latent_seed,
        batch_size=config.latent_batch_size,
    )
    gaussian = sample_empirical_covariance_gaussian(
        state["fit"],
        config.count,
        seed=config.latent_seed + 1,
        device=config.device,
        batch_size=config.latent_batch_size,
    )
    triplet = {
        "empirical": state["evaluation"].float(),
        "prior": prior,
        "gaussian": gaussian,
    }
    # Pair the empirical subset, random projections, and classifier split
    # across candidates so differences are not projection Monte Carlo noise.
    metrics = {
        name: high_dimensional_latent_metrics(
            triplet["empirical"], value, seed=config.metric_seed
        )
        for name, value in triplet.items()
        if name != "empirical"
    }
    empirical_split_control = high_dimensional_latent_metrics(
        triplet["empirical"], state["fit"].float(), seed=config.metric_seed
    )
    payload = {
        "protocol": "frozen_official_d2c_ffhq256",
        "count": config.count,
        "fit_count": config.fit_count,
        "fit_and_evaluation_disjoint": True,
        "prior_skip": config.prior_skip,
        "checkpoint": str(config.checkpoint),
        "checkpoint_metadata": frozen.metadata,
        "metrics": metrics,
        "empirical_split_control": empirical_split_control,
        "warning": (
            "Latent covariance/effective-rank metrics use a fixed 1024-sample "
            "subset and Gram identities; C2ST uses a fixed 256d random projection."
        ),
    }
    config.result_root.mkdir(parents=True, exist_ok=True)
    output = config.result_root / f"latent_audit_{_cache_stem(config)}.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    torch.save(
        {
            **triplet,
            "fit_indices": state["fit_indices"],
            "evaluation_indices": state["evaluation_indices"],
        },
        config.result_root / f"latent_triplet_{_cache_stem(config)}.pt",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    return output


@torch.inference_mode()
def _extract_real_features(
    records: Sequence[bytes],
    feature_model: torch.nn.Module,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    features = []
    for images in _image_loader(records, max(batch_size, 16)):
        features.append(inception_features(feature_model, images.to(device)).cpu())
    return torch.cat(features)


@torch.inference_mode()
def _decode_latent_features(
    frozen: FrozenD2C,
    latents: torch.Tensor,
    feature_model: torch.nn.Module,
    *,
    seed: int,
    batch_size: int,
    preview_count: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    # D2C samples from its discretized mixture-logistic output. Resetting both
    # CPU and CUDA RNGs before each condition gives the three conditions the
    # same Gumbel/logistic random-number stream batch by batch.
    configure_fp32(seed)
    features = []
    previews = []
    for begin in range(0, len(latents), batch_size):
        latent = latents[begin : begin + batch_size].to(config_device(frozen))
        images = frozen.model.latent_to_image(latent).float().clamp(0, 1)
        features.append(inception_features(feature_model, images).cpu())
        if sum(len(chunk) for chunk in previews) < preview_count:
            previews.append(images[:preview_count].cpu())
    return torch.cat(features), torch.cat(previews)[:preview_count]


def config_device(frozen: FrozenD2C) -> torch.device:
    return next(frozen.model.autoencoder.parameters()).device


def cache_real_features(config: D2CAuditConfig) -> Path:
    config.validate(require_checkpoint=False, require_data=True)
    device = _select_cuda_device(config.device)
    configure_fp32(config.metric_seed)
    records = read_parquet_image_bytes(
        config.parquet_root, config.data_start + config.fit_count, config.count
    )
    feature_model = build_fid_feature_extractor(device)
    features = _extract_real_features(
        records, feature_model, device=device, batch_size=config.batch_size
    )
    indices = torch.arange(
        config.data_start + config.fit_count,
        config.data_start + config.fit_count + config.count,
    )
    config.result_root.mkdir(parents=True, exist_ok=True)
    output = config.result_root / f"real_inception_{_cache_stem(config)}.pt"
    torch.save({"features": features, "indices": indices}, output)
    print(output, flush=True)
    return output


def run_smoke(config: D2CAuditConfig) -> Path:
    triplet_path = config.result_root / f"latent_triplet_{_cache_stem(config)}.pt"
    if not triplet_path.is_file():
        raise FileNotFoundError(f"missing {triplet_path}; run latent first")
    state = torch.load(triplet_path, map_location="cpu", weights_only=True)
    frozen = load_frozen_d2c(config)
    rows = []
    for name in ("empirical", "prior", "gaussian"):
        configure_fp32(config.decoder_seed)
        images = frozen.model.latent_to_image(
            state[name][:4].to(config_device(frozen))
        ).float().clamp(0, 1)
        rows.append(images.cpu())
    output = config.result_root / f"smoke_{_cache_stem(config)}.png"
    save_image(make_grid(torch.cat(rows), nrow=4, padding=2), output)
    print(output, flush=True)
    return output


def run_feature_audit(config: D2CAuditConfig) -> Path:
    triplet_path = config.result_root / f"latent_triplet_{_cache_stem(config)}.pt"
    real_path = config.result_root / f"real_inception_{_cache_stem(config)}.pt"
    if not triplet_path.is_file() or not real_path.is_file():
        raise FileNotFoundError("run latent and real-features before features")
    triplet_state = torch.load(triplet_path, map_location="cpu", weights_only=True)
    real_state = torch.load(real_path, map_location="cpu", weights_only=True)
    triplet = {
        name: triplet_state[name].float()
        for name in ("empirical", "prior", "gaussian")
    }
    frozen = load_frozen_d2c(config)
    feature_model = build_fid_feature_extractor(config_device(frozen))
    decoded_features = {}
    preview_rows = []
    for name, latent in triplet.items():
        features, preview = _decode_latent_features(
            frozen,
            latent,
            feature_model,
            seed=config.decoder_seed,
            batch_size=config.batch_size,
        )
        decoded_features[name] = features
        preview_rows.append(preview)
    comparisons, gaps = score_decoded_feature_sets(
        real_state["features"], decoded_features, seed=config.metric_seed
    )
    payload = {
        "protocol": "frozen_official_d2c_ffhq256_shared_output_noise",
        "count": config.count,
        "fit_count": config.fit_count,
        "prior_skip": config.prior_skip,
        "decoder_seed": config.decoder_seed,
        "checkpoint_metadata": frozen.metadata,
        "comparisons": comparisons,
        "gaps": gaps,
        "paired_metric_randomness": True,
        "warning": (
            "Diagnostic KID/SWD/C2ST at 2k samples; this is not an official "
            "50k FID reproduction. D2C output sampling is stochastic but uses "
            "the same random-number stream for all latent sources."
        ),
    }
    stem = (
        f"feature_seed{config.decoder_seed}_{_cache_stem(config)}_skip{config.prior_skip}"
    )
    output = config.result_root / f"{stem}.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    torch.save(
        {
            "real": real_state["features"],
            **decoded_features,
            "real_indices": real_state["indices"],
        },
        config.result_root / f"{stem}.pt",
    )
    save_image(
        make_grid(torch.cat(preview_rows), nrow=len(preview_rows[0]), padding=2),
        config.result_root / f"{stem}.png",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    return output


def summarize_feature_audits(config: D2CAuditConfig) -> Path:
    pattern = f"feature_seed*_{_cache_stem(config)}_skip{config.prior_skip}.json"
    paths = sorted(config.result_root.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no feature audits match {pattern}")
    rows = [json.loads(path.read_text()) for path in paths]
    gap_summary = {}
    for key in sorted(rows[0]["gaps"]):
        values = torch.tensor([row["gaps"][key] for row in rows], dtype=torch.float64)
        gap_summary[key] = {
            "mean": float(values.mean()),
            "std_across_decoder_seeds": float(values.std(unbiased=len(values) > 1)),
            "positive_seed_count": int((values > 0).sum()),
            "seed_count": len(values),
        }

    states = []
    for row in rows:
        stem = (
            f"feature_seed{row['decoder_seed']}_{_cache_stem(config)}_skip{config.prior_skip}.pt"
        )
        states.append(torch.load(config.result_root / stem, map_location="cpu", weights_only=True))
    noise_control = {}
    for condition in ("empirical", "prior", "gaussian"):
        cosine_values, kid_values, swd_values = [], [], []
        for pair_index, (left_index, right_index) in enumerate(
            combinations(range(len(states)), 2)
        ):
            left = states[left_index][condition]
            right = states[right_index][condition]
            cosine_values.append(mean_cosine_distance(left, right))
            kid, _ = kernel_inception_distance(
                left, right, seed=config.metric_seed + 2_001 + pair_index
            )
            kid_values.append(kid)
            swd_values.append(
                standardized_sliced_wasserstein(
                    left,
                    right,
                    directions=128,
                    seed=config.metric_seed + 3_001 + pair_index,
                )
            )
        noise_control[condition] = {
            "paired_feature_cosine_distance_mean": float(np.mean(cosine_values)),
            "distribution_kid_mean": float(np.mean(kid_values)),
            "distribution_swd_mean": float(np.mean(swd_values)),
        }

    source_control = {}
    for name, candidate in (
        ("empirical_vs_prior", "prior"),
        ("empirical_vs_gaussian", "gaussian"),
    ):
        values = [
            mean_cosine_distance(state["empirical"], state[candidate])
            for state in states
        ]
        source_control[name] = {
            "paired_feature_cosine_distance_mean": float(np.mean(values)),
            "paired_feature_cosine_distance_std": float(np.std(values)),
            "cosine_distance_over_empirical_noise": float(
                np.mean(values)
                / max(
                    noise_control["empirical"][
                        "paired_feature_cosine_distance_mean"
                    ],
                    1e-12,
                )
            ),
        }

    source_distribution = {}
    for name in ("empirical_vs_prior", "empirical_vs_gaussian"):
        kid_values = [row["comparisons"][name]["kid_mean"] for row in rows]
        swd_values = [row["comparisons"][name]["standardized_swd"] for row in rows]
        source_distribution[name] = {
            "kid_mean": float(np.mean(kid_values)),
            "kid_std_across_decoder_seeds": float(np.std(kid_values, ddof=1)),
            "standardized_swd_mean": float(np.mean(swd_values)),
            "standardized_swd_std_across_decoder_seeds": float(
                np.std(swd_values, ddof=1)
            ),
            "swd_over_empirical_cross_seed_noise": float(
                np.mean(swd_values)
                / max(noise_control["empirical"]["distribution_swd_mean"], 1e-12)
            ),
        }
    payload = {
        "count": config.count,
        "fit_count": config.fit_count,
        "prior_skip": config.prior_skip,
        "decoder_seeds": [row["decoder_seed"] for row in rows],
        "gap_summary": gap_summary,
        "decoder_noise_control": noise_control,
        "latent_source_control": source_control,
        "latent_source_distribution_control": source_distribution,
        "interpretation_rule": (
            "A positive prior gap means official D2C prior latents decode farther "
            "from real FFHQ than empirical encoder latents."
        ),
    }
    output = config.result_root / f"feature_summary_{_cache_stem(config)}.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    return output


def run_inspect(config: D2CAuditConfig) -> None:
    frozen = load_frozen_d2c(config)
    print(json.dumps(frozen.metadata, indent=2, ensure_ascii=False), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "inspect",
            "cache-empirical",
            "latent",
            "smoke",
            "real-features",
            "features",
            "summary",
        ),
    )
    parser.add_argument("--d2c-repo", type=Path, default=DEFAULT_D2C_REPO)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--parquet-root", type=Path, default=DEFAULT_PARQUET_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--count", type=int, default=2_000)
    parser.add_argument("--fit-count", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--latent-batch-size", type=int, default=32)
    parser.add_argument("--prior-skip", type=int, default=100)
    parser.add_argument("--data-start", type=int, default=0)
    parser.add_argument("--latent-seed", type=int, default=20_260_722)
    parser.add_argument("--decoder-seed", type=int, default=30_260_722)
    parser.add_argument("--metric-seed", type=int, default=40_260_722)
    return parser


def main(argv: Sequence[str] | None = None) -> Path | None:
    args = build_parser().parse_args(argv)
    config = D2CAuditConfig(
        d2c_repo=args.d2c_repo.expanduser(),
        checkpoint=args.checkpoint.expanduser(),
        parquet_root=args.parquet_root.expanduser(),
        result_root=args.result_root.expanduser(),
        device=args.device,
        count=args.count,
        fit_count=args.fit_count,
        batch_size=args.batch_size,
        latent_batch_size=args.latent_batch_size,
        prior_skip=args.prior_skip,
        data_start=args.data_start,
        latent_seed=args.latent_seed,
        decoder_seed=args.decoder_seed,
        metric_seed=args.metric_seed,
    )
    actions = {
        "inspect": run_inspect,
        "cache-empirical": run_cache_empirical,
        "latent": run_latent_audit,
        "smoke": run_smoke,
        "real-features": cache_real_features,
        "features": run_feature_audit,
        "summary": summarize_feature_audits,
    }
    return actions[args.mode](config)


if __name__ == "__main__":
    main()
