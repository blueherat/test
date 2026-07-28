"""Frozen external audit of the official DiffAE FFHQ-128 generator.

This module never trains a model.  It loads only the EMA branch from the
official DiffAE checkpoint and compares empirical, learned-prior, and matched
Gaussian semantic latents under shared pixel-decoder noise.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import types
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import pil_to_tensor
from torchvision.utils import make_grid, save_image


DEFAULT_DIFFAE_REPO = Path.home() / "data/external/diffae"
DEFAULT_CHECKPOINT = (
    Path.home() / "data/checkpoints/diffae/ffhq128_autoenc_latent/last_bak.ckpt"
)
DEFAULT_LATENT_STATS = (
    Path.home() / "data/checkpoints/diffae/ffhq128_autoenc_130M/latent.pkl"
)
DEFAULT_RESULT_ROOT = Path.home() / "data/results/diffae_gate_a"
DEFAULT_FFHQ_ROOT = Path.home() / "data/datasets/ffhq128/thumbnails128x128"


@dataclass(frozen=True)
class DiffAEAuditConfig:
    diffae_repo: Path = DEFAULT_DIFFAE_REPO
    checkpoint: Path = DEFAULT_CHECKPOINT
    latent_stats: Path = DEFAULT_LATENT_STATS
    result_root: Path = DEFAULT_RESULT_ROOT
    ffhq_root: Path = DEFAULT_FFHQ_ROOT
    device: str = "cuda:0"
    count: int = 2_000
    batch_size: int = 8
    pixel_steps: int = 10
    latent_steps: int = 10
    latent_seed: int = 20_260_722
    decoder_seed: int = 30_260_722
    empirical_seed: int = 40_260_722

    def validate(self, *, require_checkpoint: bool = True) -> None:
        if not self.diffae_repo.is_dir():
            raise FileNotFoundError(f"missing DiffAE repository: {self.diffae_repo}")
        if require_checkpoint and not self.checkpoint.is_file():
            raise FileNotFoundError(f"missing DiffAE checkpoint: {self.checkpoint}")
        if not self.latent_stats.is_file():
            raise FileNotFoundError(f"missing DiffAE latent statistics: {self.latent_stats}")
        if self.count < 2 or self.batch_size < 1:
            raise ValueError("count must be >= 2 and batch_size must be positive")
        if self.pixel_steps < 1 or self.latent_steps < 1:
            raise ValueError("sampling step counts must be positive")


@dataclass
class FrozenDiffAE:
    model: torch.nn.Module
    conf: Any
    conds: torch.Tensor
    conds_mean: torch.Tensor
    conds_std: torch.Tensor
    checkpoint_metadata: dict[str, Any]


class FlatImageDataset(Dataset):
    def __init__(self, root: str | Path, indices: torch.Tensor):
        root = Path(root)
        paths = sorted(root.glob("*.png"))
        if not paths:
            raise FileNotFoundError(f"no PNG images found in {root}")
        if int(indices.max()) >= len(paths):
            raise IndexError("FFHQ evaluation index is out of range")
        self.paths = [paths[int(index)] for index in indices]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
            return pil_to_tensor(image).float().div_(255)


def configure_fp32(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def load_latent_statistics(path: str | Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    required = {"conds", "conds_mean", "conds_std"}
    if not isinstance(state, dict) or not required.issubset(state):
        raise RuntimeError(f"latent statistics must contain {sorted(required)}")
    conds = state["conds"].detach().float().cpu().contiguous()
    mean = state["conds_mean"].detach().float().cpu().contiguous()
    std = state["conds_std"].detach().float().cpu().contiguous()
    if conds.ndim != 2 or mean.shape != conds.shape[1:] or std.shape != mean.shape:
        raise RuntimeError("incompatible DiffAE latent-statistics shapes")
    if not torch.isfinite(conds).all() or not torch.isfinite(mean).all():
        raise FloatingPointError("non-finite official DiffAE latent statistics")
    if not torch.isfinite(std).all() or bool((std <= 0).any()):
        raise FloatingPointError("DiffAE latent standard deviations must be finite and positive")
    recomputed_mean = conds.mean(dim=0)
    recomputed_std = conds.std(dim=0, unbiased=False)
    torch.testing.assert_close(mean, recomputed_mean, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(std, recomputed_std, atol=2e-6, rtol=2e-5)
    return conds, mean, std


def extract_complete_ema_state(
    checkpoint_state: dict[str, Any], expected_state: dict[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    raw_state = checkpoint_state.get("state_dict", checkpoint_state)
    if not isinstance(raw_state, dict):
        raise RuntimeError("DiffAE checkpoint does not contain a state_dict")
    prefix = "ema_model."
    ema_state = {
        key[len(prefix) :]: value
        for key, value in raw_state.items()
        if key.startswith(prefix)
    }
    expected_keys = set(expected_state)
    actual_keys = set(ema_state)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    required_prefixes = ("encoder.", "latent_net.", "input_blocks.", "output_blocks.")
    absent_components = [
        name for name in required_prefixes if not any(key.startswith(name) for key in actual_keys)
    ]
    if missing or unexpected or absent_components:
        raise RuntimeError(
            "incomplete DiffAE EMA checkpoint: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}, "
            f"absent_components={absent_components}"
        )
    metadata = {
        "global_step": int(checkpoint_state.get("global_step", -1)),
        "ema_tensor_count": len(ema_state),
        "ema_parameter_count": int(sum(value.numel() for value in ema_state.values())),
        "required_components": [name.rstrip(".") for name in required_prefixes],
    }
    return ema_state, metadata


def _import_diffae_template(repo: Path):
    repo = repo.resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    # DiffAE imports this private NumPy 1.x module only for ``flip``.  Provide
    # the removed module name without changing the external official clone.
    legacy_numpy_module = "numpy.lib.function_base"
    if legacy_numpy_module not in sys.modules:
        compatibility = types.ModuleType(legacy_numpy_module)
        compatibility.flip = np.flip
        sys.modules[legacy_numpy_module] = compatibility
    from templates_latent import ffhq128_autoenc_latent  # type: ignore

    return ffhq128_autoenc_latent


def load_frozen_diffae(config: DiffAEAuditConfig) -> FrozenDiffAE:
    config.validate(require_checkpoint=True)
    configure_fp32(config.latent_seed)
    make_conf = _import_diffae_template(config.diffae_repo)
    conf = make_conf()
    conf.fp16 = False
    conf.pretrain = None
    conf.latent_infer_path = None
    model = conf.make_model_conf().make_model()
    checkpoint_state = torch.load(config.checkpoint, map_location="cpu", weights_only=False)
    ema_state, metadata = extract_complete_ema_state(checkpoint_state, model.state_dict())
    model.load_state_dict(ema_state, strict=True)
    model.eval().requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("DiffAE audit requires every model parameter to be frozen")
    model.to(torch.device(config.device), dtype=torch.float32)
    conds, mean, std = load_latent_statistics(config.latent_stats)
    return FrozenDiffAE(model, conf, conds, mean, std, metadata)


def select_empirical_latents(
    conds: torch.Tensor, count: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if count > len(conds):
        raise ValueError("requested more empirical latents than available")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    indices = torch.randperm(len(conds), generator=generator)[:count]
    return conds[indices].clone(), indices


def fit_full_covariance_gaussian(
    conds: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    eigenvalue_floor: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    normalized = ((conds - mean) / std).double()
    normalized_mean = normalized.mean(dim=0)
    centered = normalized - normalized_mean
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(float(eigenvalue_floor))
    factor = eigenvectors * eigenvalues.sqrt().unsqueeze(0)
    return normalized_mean.float(), factor.float()


def sample_matched_gaussian(
    count: int,
    normalized_mean: torch.Tensor,
    factor: torch.Tensor,
    data_mean: torch.Tensor,
    data_std: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn((count, factor.shape[0]), generator=generator)
    normalized = noise @ factor.T + normalized_mean
    return normalized * data_std + data_mean


def _device_generator(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(int(seed))


@torch.inference_mode()
def sample_prior_latents(
    frozen: FrozenDiffAE,
    count: int,
    *,
    steps: int,
    seed: int,
    batch_size: int,
) -> torch.Tensor:
    device = next(frozen.model.parameters()).device
    sampler = frozen.conf._make_latent_diffusion_conf(int(steps)).make_sampler()
    generator = _device_generator(device, seed)
    output = []
    for start in range(0, count, batch_size):
        size = min(batch_size, count - start)
        noise = torch.randn(
            (size, frozen.conf.style_ch), device=device, generator=generator
        )
        normalized = sampler.sample(
            model=frozen.model.latent_net,
            noise=noise,
            clip_denoised=frozen.conf.latent_clip_sample,
        )
        latent = normalized * frozen.conds_std.to(device) + frozen.conds_mean.to(device)
        output.append(latent.float().cpu())
    return torch.cat(output, dim=0)


@torch.inference_mode()
def decode_latents(
    frozen: FrozenDiffAE,
    latents: torch.Tensor,
    *,
    steps: int,
    seed: int,
    batch_size: int,
) -> torch.Tensor:
    device = next(frozen.model.parameters()).device
    sampler = frozen.conf._make_diffusion_conf(int(steps)).make_sampler()
    generator = _device_generator(device, seed)
    output = []
    for latent in latents.split(batch_size):
        latent = latent.to(device=device, dtype=torch.float32)
        pixel_noise = torch.randn(
            (len(latent), 3, frozen.conf.img_size, frozen.conf.img_size),
            device=device,
            generator=generator,
        )
        image = sampler.sample(model=frozen.model, noise=pixel_noise, cond=latent)
        output.append(((image.float().clamp(-1, 1) + 1) / 2).cpu())
    return torch.cat(output, dim=0)


def build_fid_feature_extractor(device: torch.device) -> torch.nn.Module:
    from pytorch_fid.inception import InceptionV3

    block = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block], resize_input=True, normalize_input=True)
    model.to(device).eval().requires_grad_(False)
    return model


@torch.inference_mode()
def inception_features(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    feature = model(images)[0]
    if feature.shape[-2:] != (1, 1):
        feature = F.adaptive_avg_pool2d(feature, output_size=(1, 1))
    return feature.flatten(1).float()


@torch.inference_mode()
def extract_real_features(
    root: str | Path,
    *,
    count: int,
    seed: int,
    batch_size: int,
    device: torch.device,
    feature_model: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    paths = sorted(Path(root).glob("*.png"))
    if count > len(paths):
        raise ValueError("requested more FFHQ images than available")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    indices = torch.randperm(len(paths), generator=generator)[:count]
    dataset = FlatImageDataset(root, indices)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    features = []
    for images in loader:
        features.append(inception_features(feature_model, images.to(device)).cpu())
    return torch.cat(features), indices


@torch.inference_mode()
def decode_latent_features(
    frozen: FrozenDiffAE,
    latents: torch.Tensor,
    feature_model: torch.nn.Module,
    *,
    steps: int,
    seed: int,
    batch_size: int,
    preview_count: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = next(frozen.model.parameters()).device
    sampler = frozen.conf._make_diffusion_conf(int(steps)).make_sampler()
    generator = _device_generator(device, seed)
    features = []
    previews = []
    for latent in latents.split(batch_size):
        latent = latent.to(device=device, dtype=torch.float32)
        pixel_noise = torch.randn(
            (len(latent), 3, frozen.conf.img_size, frozen.conf.img_size),
            device=device,
            generator=generator,
        )
        image = sampler.sample(model=frozen.model, noise=pixel_noise, cond=latent)
        image = (image.float().clamp(-1, 1) + 1) / 2
        features.append(inception_features(feature_model, image).cpu())
        if sum(len(value) for value in previews) < preview_count:
            previews.append(image[:preview_count].cpu())
    preview = torch.cat(previews)[:preview_count]
    return torch.cat(features), preview


def standardized_sliced_wasserstein(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    directions: int = 256,
    seed: int = 0,
) -> float:
    if reference.ndim != 2 or candidate.ndim != 2 or reference.shape[1] != candidate.shape[1]:
        raise ValueError("SWD expects two matrices with the same feature dimension")
    mean = reference.mean(dim=0, keepdim=True)
    std = reference.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    left = (reference - mean) / std
    right = (candidate - mean) / std
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    projection = torch.randn(
        (reference.shape[1], int(directions)), generator=generator
    )
    projection = projection / projection.norm(dim=0, keepdim=True).clamp_min(1e-12)
    left = (left @ projection).sort(dim=0).values
    right = (right @ projection).sort(dim=0).values
    count = min(len(left), len(right))
    return float((left[:count] - right[:count]).abs().mean())


def effective_rank(values: torch.Tensor) -> float:
    centered = values.double() - values.double().mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    probabilities = eigenvalues / eigenvalues.sum().clamp_min(1e-30)
    entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum()
    return float(entropy.exp())


def c2st_metrics(
    reference: torch.Tensor, candidate: torch.Tensor, *, seed: int = 0
) -> dict[str, float]:
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
        "c2st_accuracy": float(accuracy_score(test_y, prediction)),
        "c2st_auc": float(roc_auc_score(test_y, probability)),
    }


def covariance_relative_error(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    left = torch.cov(reference.double().T)
    right = torch.cov(candidate.double().T)
    return float(torch.linalg.norm(left - right) / torch.linalg.norm(left).clamp_min(1e-30))


def latent_distribution_metrics(
    reference: torch.Tensor, candidate: torch.Tensor, *, seed: int
) -> dict[str, float]:
    mean_scale = reference.double().std(dim=0, unbiased=False).norm().clamp_min(1e-30)
    metrics = {
        "standardized_swd": standardized_sliced_wasserstein(
            reference, candidate, seed=seed
        ),
        "mean_shift_relative": float(
            (reference.double().mean(dim=0) - candidate.double().mean(dim=0)).norm()
            / mean_scale
        ),
        "covariance_relative_error": covariance_relative_error(reference, candidate),
        "reference_effective_rank": effective_rank(reference),
        "candidate_effective_rank": effective_rank(candidate),
    }
    metrics.update(c2st_metrics(reference, candidate, seed=seed))
    if not all(math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("non-finite latent-distribution metric")
    return metrics


def polynomial_mmd_unbiased(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("unbiased polynomial MMD expects equal-sized feature matrices")
    count, dimension = left.shape
    if count < 2:
        raise ValueError("polynomial MMD requires at least two samples")
    left = left.double()
    right = right.double()
    scale = float(dimension)
    left_kernel = (left @ left.T / scale + 1.0).pow(3)
    right_kernel = (right @ right.T / scale + 1.0).pow(3)
    cross_kernel = (left @ right.T / scale + 1.0).pow(3)
    within_left = (left_kernel.sum() - left_kernel.diagonal().sum()) / (count * (count - 1))
    within_right = (right_kernel.sum() - right_kernel.diagonal().sum()) / (count * (count - 1))
    return float(within_left + within_right - 2 * cross_kernel.mean())


def kernel_inception_distance(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    seed: int,
    subsets: int = 20,
    subset_size: int = 256,
) -> tuple[float, float]:
    subset_size = min(int(subset_size), len(reference), len(candidate))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    estimates = []
    for _ in range(int(subsets)):
        left_index = torch.randperm(len(reference), generator=generator)[:subset_size]
        right_index = torch.randperm(len(candidate), generator=generator)[:subset_size]
        estimates.append(
            polynomial_mmd_unbiased(reference[left_index], candidate[right_index])
        )
    values = torch.tensor(estimates, dtype=torch.float64)
    return float(values.mean()), float(values.std(unbiased=True))


def feature_distribution_metrics(
    reference: torch.Tensor, candidate: torch.Tensor, *, seed: int
) -> dict[str, float]:
    kid_mean, kid_std = kernel_inception_distance(
        reference, candidate, seed=seed
    )
    metrics = {
        "kid_mean": kid_mean,
        "kid_subset_std": kid_std,
        "standardized_swd": standardized_sliced_wasserstein(
            reference, candidate, directions=128, seed=seed + 1
        ),
        **c2st_metrics(reference, candidate, seed=seed + 2),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("non-finite decoded-feature metric")
    return metrics


def score_decoded_feature_sets(
    real_features: torch.Tensor,
    decoded_features: dict[str, torch.Tensor],
    *,
    seed: int,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    # Reuse the same real subset, KID subsets, SWD projections, and C2ST split
    # across conditions so differences are paired Monte Carlo estimates.
    comparisons = {
        f"real_vs_{name}": feature_distribution_metrics(
            real_features, decoded_features[name], seed=seed
        )
        for name in ("empirical", "prior", "gaussian")
    }
    comparisons["empirical_vs_prior"] = feature_distribution_metrics(
        decoded_features["empirical"], decoded_features["prior"], seed=seed + 101
    )
    comparisons["empirical_vs_gaussian"] = feature_distribution_metrics(
        decoded_features["empirical"], decoded_features["gaussian"], seed=seed + 103
    )
    empirical_metrics = comparisons["real_vs_empirical"]
    prior_metrics = comparisons["real_vs_prior"]
    gaussian_metrics = comparisons["real_vs_gaussian"]
    gap_metrics = ("kid_mean", "standardized_swd", "c2st_accuracy", "c2st_auc")
    gaps = {
        f"prior_gap_{key}": float(prior_metrics[key] - empirical_metrics[key])
        for key in gap_metrics
    }
    gaps.update(
        {
            f"gaussian_gap_{key}": float(gaussian_metrics[key] - empirical_metrics[key])
            for key in gap_metrics
        }
    )
    return comparisons, gaps


def mean_cosine_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("cosine distance expects paired equal-sized feature matrices")
    return float((1.0 - F.cosine_similarity(left, right, dim=1)).mean())


def prepare_latent_triplet(
    frozen: FrozenDiffAE, config: DiffAEAuditConfig
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    empirical, indices = select_empirical_latents(
        frozen.conds, config.count, config.empirical_seed
    )
    prior = sample_prior_latents(
        frozen,
        config.count,
        steps=config.latent_steps,
        seed=config.latent_seed,
        batch_size=config.batch_size,
    )
    normalized_mean, factor = fit_full_covariance_gaussian(
        frozen.conds, frozen.conds_mean, frozen.conds_std
    )
    gaussian = sample_matched_gaussian(
        config.count,
        normalized_mean,
        factor,
        frozen.conds_mean,
        frozen.conds_std,
        seed=config.latent_seed + 1,
    )
    metadata = {
        "empirical_indices": indices,
        "gaussian_normalized_mean": normalized_mean,
        "gaussian_factor": factor,
    }
    return {"empirical": empirical, "prior": prior, "gaussian": gaussian}, metadata


def run_latent_audit(config: DiffAEAuditConfig) -> Path:
    frozen = load_frozen_diffae(config)
    triplet, metadata = prepare_latent_triplet(frozen, config)
    result_dir = config.result_root.expanduser()
    result_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        name: latent_distribution_metrics(
            triplet["empirical"], values, seed=config.latent_seed + offset
        )
        for offset, (name, values) in enumerate(triplet.items())
        if name != "empirical"
    }
    payload = {
        "protocol": "frozen_official_diffae_ffhq128",
        "count": config.count,
        "latent_steps": config.latent_steps,
        "checkpoint": str(config.checkpoint),
        "latent_stats": str(config.latent_stats),
        "checkpoint_metadata": frozen.checkpoint_metadata,
        "metrics": metrics,
    }
    output = result_dir / f"latent_audit_n{config.count}.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    torch.save(
        {
            **triplet,
            "empirical_indices": metadata["empirical_indices"],
            "checkpoint_metadata": frozen.checkpoint_metadata,
        },
        result_dir / f"latent_triplet_n{config.count}.pt",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    return output


def run_smoke(config: DiffAEAuditConfig) -> Path:
    frozen = load_frozen_diffae(config)
    triplet, _ = prepare_latent_triplet(frozen, config)
    images = {
        name: decode_latents(
            frozen,
            latent,
            steps=config.pixel_steps,
            seed=config.decoder_seed,
            batch_size=config.batch_size,
        )
        for name, latent in triplet.items()
    }
    # Rows share exactly the same pixel-noise stream; only z_sem changes.
    ordered = torch.cat([images[name] for name in ("empirical", "prior", "gaussian")])
    grid = make_grid(ordered, nrow=config.count, padding=2)
    config.result_root.mkdir(parents=True, exist_ok=True)
    output = config.result_root / (
        f"smoke_n{config.count}_pixel{config.pixel_steps}_latent{config.latent_steps}.png"
    )
    save_image(grid, output)
    print(output, flush=True)
    return output


def run_feature_audit(config: DiffAEAuditConfig) -> Path:
    config.validate(require_checkpoint=True)
    if not config.ffhq_root.is_dir():
        raise FileNotFoundError(f"missing FFHQ image root: {config.ffhq_root}")
    result_dir = config.result_root.expanduser()
    triplet_path = result_dir / f"latent_triplet_n{config.count}.pt"
    if not triplet_path.is_file():
        raise FileNotFoundError(
            f"missing {triplet_path}; run the latent mode once before feature mode"
        )
    latent_state = torch.load(triplet_path, map_location="cpu", weights_only=True)
    triplet = {name: latent_state[name].float() for name in ("empirical", "prior", "gaussian")}
    frozen = load_frozen_diffae(config)
    device = next(frozen.model.parameters()).device
    feature_model = build_fid_feature_extractor(device)
    real_path = result_dir / f"real_inception_n{config.count}.pt"
    if real_path.is_file():
        real_state = torch.load(real_path, map_location="cpu", weights_only=True)
        real_features = real_state["features"]
        real_indices = real_state["indices"]
    else:
        real_features, real_indices = extract_real_features(
            config.ffhq_root,
            count=config.count,
            seed=config.empirical_seed + 101,
            batch_size=max(config.batch_size, 16),
            device=device,
            feature_model=feature_model,
        )
        torch.save({"features": real_features, "indices": real_indices}, real_path)
    decoded_features = {}
    preview_rows = []
    for name, latent in triplet.items():
        features, preview = decode_latent_features(
            frozen,
            latent,
            feature_model,
            steps=config.pixel_steps,
            seed=config.decoder_seed,
            batch_size=config.batch_size,
        )
        decoded_features[name] = features
        preview_rows.append(preview)
    comparisons, gaps = score_decoded_feature_sets(
        real_features,
        decoded_features,
        seed=config.decoder_seed + 701,
    )
    payload = {
        "protocol": "frozen_official_diffae_ffhq128_fixed_pixel_noise",
        "count": config.count,
        "pixel_steps": config.pixel_steps,
        "latent_steps": config.latent_steps,
        "decoder_seed": config.decoder_seed,
        "checkpoint_metadata": frozen.checkpoint_metadata,
        "comparisons": comparisons,
        "gaps": gaps,
        "warning": "2k metrics are diagnostic; KID/SWD/C2ST are not an official FID reproduction",
    }
    stem = f"feature_seed{config.decoder_seed}_n{config.count}_p{config.pixel_steps}"
    output = result_dir / f"{stem}.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    torch.save(
        {"real": real_features, **decoded_features, "real_indices": real_indices},
        result_dir / f"{stem}.pt",
    )
    save_image(
        make_grid(torch.cat(preview_rows), nrow=len(preview_rows[0]), padding=2),
        result_dir / f"{stem}.png",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    return output


def rescore_feature_audit(config: DiffAEAuditConfig) -> Path:
    stem = f"feature_seed{config.decoder_seed}_n{config.count}_p{config.pixel_steps}"
    state_path = config.result_root / f"{stem}.pt"
    output = config.result_root / f"{stem}.json"
    if not state_path.is_file() or not output.is_file():
        raise FileNotFoundError(f"missing saved feature audit for {stem}")
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    decoded = {name: state[name] for name in ("empirical", "prior", "gaussian")}
    comparisons, gaps = score_decoded_feature_sets(
        state["real"], decoded, seed=config.decoder_seed + 701
    )
    payload = json.loads(output.read_text())
    payload["comparisons"] = comparisons
    payload["gaps"] = gaps
    payload["paired_metric_randomness"] = True
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    return output


def cache_real_features(config: DiffAEAuditConfig) -> Path:
    if not config.ffhq_root.is_dir():
        raise FileNotFoundError(f"missing FFHQ image root: {config.ffhq_root}")
    configure_fp32(config.empirical_seed)
    device = torch.device(config.device)
    feature_model = build_fid_feature_extractor(device)
    features, indices = extract_real_features(
        config.ffhq_root,
        count=config.count,
        seed=config.empirical_seed + 101,
        batch_size=max(config.batch_size, 16),
        device=device,
        feature_model=feature_model,
    )
    config.result_root.mkdir(parents=True, exist_ok=True)
    output = config.result_root / f"real_inception_n{config.count}.pt"
    torch.save({"features": features, "indices": indices}, output)
    print(output, flush=True)
    return output


def summarize_feature_audits(config: DiffAEAuditConfig) -> Path:
    pattern = f"feature_seed*_n{config.count}_p{config.pixel_steps}.json"
    paths = sorted(config.result_root.expanduser().glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no feature audits match {pattern}")
    rows = [json.loads(path.read_text()) for path in paths]
    gap_keys = sorted(rows[0]["gaps"])
    summary = {}
    for key in gap_keys:
        values = torch.tensor([row["gaps"][key] for row in rows], dtype=torch.float64)
        summary[key] = {
            "mean": float(values.mean()),
            "std_across_decoder_seeds": float(values.std(unbiased=len(values) > 1)),
            "positive_seed_count": int((values > 0).sum()),
            "seed_count": len(values),
        }
    states = []
    for row in rows:
        stem = f"feature_seed{row['decoder_seed']}_n{config.count}_p{config.pixel_steps}.pt"
        states.append(
            torch.load(config.result_root / stem, map_location="cpu", weights_only=True)
        )
    cross_seed = {"empirical": [], "prior": [], "gaussian": []}
    cross_seed_kid = {"empirical": [], "prior": [], "gaussian": []}
    cross_seed_swd = {"empirical": [], "prior": [], "gaussian": []}
    for pair_index, (left_index, right_index) in enumerate(combinations(range(len(states)), 2)):
        for condition in cross_seed:
            left = states[left_index][condition]
            right = states[right_index][condition]
            cross_seed[condition].append(mean_cosine_distance(left, right))
            kid, _ = kernel_inception_distance(
                left, right, seed=config.decoder_seed + 2_001 + pair_index
            )
            cross_seed_kid[condition].append(kid)
            cross_seed_swd[condition].append(
                standardized_sliced_wasserstein(
                    left,
                    right,
                    directions=128,
                    seed=config.decoder_seed + 3_001 + pair_index,
                )
            )
    source_cosine = {
        "empirical_vs_prior": [
            mean_cosine_distance(state["empirical"], state["prior"])
            for state in states
        ],
        "empirical_vs_gaussian": [
            mean_cosine_distance(state["empirical"], state["gaussian"])
            for state in states
        ],
    }
    noise_control = {
        condition: {
            "paired_feature_cosine_distance_mean": float(np.mean(cross_seed[condition])),
            "paired_feature_cosine_distance_std": float(np.std(cross_seed[condition])),
            "distribution_kid_mean": float(np.mean(cross_seed_kid[condition])),
            "distribution_swd_mean": float(np.mean(cross_seed_swd[condition])),
        }
        for condition in cross_seed
    }
    source_control = {
        name: {
            "paired_feature_cosine_distance_mean": float(np.mean(values)),
            "paired_feature_cosine_distance_std": float(np.std(values)),
        }
        for name, values in source_cosine.items()
    }
    empirical_noise = noise_control["empirical"]["paired_feature_cosine_distance_mean"]
    source_control["empirical_vs_prior"]["cosine_distance_over_empirical_noise"] = float(
        source_control["empirical_vs_prior"]["paired_feature_cosine_distance_mean"]
        / max(empirical_noise, 1e-12)
    )
    source_control["empirical_vs_gaussian"]["cosine_distance_over_empirical_noise"] = float(
        source_control["empirical_vs_gaussian"]["paired_feature_cosine_distance_mean"]
        / max(empirical_noise, 1e-12)
    )
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
        "pixel_steps": config.pixel_steps,
        "decoder_seeds": [row["decoder_seed"] for row in rows],
        "gap_summary": summary,
        "decoder_noise_control": noise_control,
        "latent_source_control": source_control,
        "latent_source_distribution_control": source_distribution,
        "interpretation_rule": (
            "A positive prior gap means learned-prior latents decode farther from real FFHQ "
            "than empirical latents under the same pixel-noise seed."
        ),
    }
    output = config.result_root / f"feature_summary_n{config.count}_p{config.pixel_steps}.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "inspect",
            "latent",
            "smoke",
            "real-features",
            "features",
            "rescore",
            "summarize",
        ),
    )
    parser.add_argument("--diffae-repo", type=Path, default=DEFAULT_DIFFAE_REPO)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--latent-stats", type=Path, default=DEFAULT_LATENT_STATS)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--ffhq-root", type=Path, default=DEFAULT_FFHQ_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--count", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--pixel-steps", type=int, default=10)
    parser.add_argument("--latent-steps", type=int, default=10)
    parser.add_argument("--decoder-seed", type=int, default=30_260_722)
    return parser


def main(argv: Sequence[str] | None = None) -> Path | None:
    args = build_parser().parse_args(argv)
    config = DiffAEAuditConfig(
        diffae_repo=args.diffae_repo,
        checkpoint=args.checkpoint,
        latent_stats=args.latent_stats,
        result_root=args.result_root,
        ffhq_root=args.ffhq_root,
        device=args.device,
        count=args.count,
        batch_size=args.batch_size,
        pixel_steps=args.pixel_steps,
        latent_steps=args.latent_steps,
        decoder_seed=args.decoder_seed,
    )
    if args.mode == "inspect":
        config.validate(require_checkpoint=False)
        conds, mean, std = load_latent_statistics(config.latent_stats)
        print(
            json.dumps(
                {
                    "shape": list(conds.shape),
                    "mean_rms": float(mean.square().mean().sqrt()),
                    "std_mean": float(std.mean()),
                    "finite": True,
                },
                indent=2,
            )
        )
        return None
    if args.mode == "latent":
        return run_latent_audit(config)
    if args.mode == "features":
        return run_feature_audit(config)
    if args.mode == "real-features":
        return cache_real_features(config)
    if args.mode == "rescore":
        return rescore_feature_audit(config)
    if args.mode == "summarize":
        return summarize_feature_audits(config)
    return run_smoke(config)


if __name__ == "__main__":
    main()
