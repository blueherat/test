"""No-training spectral and gradient audit for the official RAE stage-2 model.

The audit deliberately separates three questions:

1. How anisotropic is the linear-FM residual across latent frequency bands?
2. Does inverse-scale weighting flatten *parameter* gradient noise at the
   official DiT output head, after separating time and direction effects?
3. Does the frozen RAE decoder care about the bands that the loss downweights?

All model paths match the official RAE ImageNet-256 setup: normalized DINOv2
latents, a shifted logit-normal time distribution, the linear interpolation
``x_t = (1 - t) z + t eps``, and velocity target ``eps - z``.
"""

from __future__ import annotations

import gc
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from baselines.dinov2_token_diagnostics import load_named_dataset
from baselines.visual_adapters import load_rae_adapter
from experiments.fm_weighting_gate import GateTreatment


DEFAULT_CACHE = Path.home() / "data/eqvae/cache/gauge_large/rae_dinov2_imagenet_2048_512.pt"
DEFAULT_STAGE2_CONFIG = Path("external/RAE/configs/stage2/training/ImageNet256/DiTDH-S_DINOv2-B.yaml")
DEFAULT_STAGE2_CHECKPOINT = (
    Path.home()
    / "data/eqvae/models/RAE/DiTs/Dinov2/wReg_base/ImageNet256/DiTDH-S_ep14/stage2_model.pt"
)


@dataclass(frozen=True)
class RAEAuditConfig:
    cache_path: str = str(DEFAULT_CACHE)
    dataset_path: str = "/data/shared/imagenet-1k"
    rae_repo_path: str = "external/RAE"
    stage2_config_path: str = str(DEFAULT_STAGE2_CONFIG)
    stage2_checkpoint_path: str = str(DEFAULT_STAGE2_CHECKPOINT)
    device: str = "cuda:3"
    seed: int = 0
    train_count: int = 256
    validation_count: int = 48
    pca_token_count: int = 4096
    spatial_band_count: int = 8
    channel_band_count: int = 4
    time_bin_count: int = 7
    model_batch_size: int = 4
    gradient_microbatches: int = 48
    gradient_sketch_rank: int = 16
    decoder_sample_count: int = 4
    decoder_epsilon: float = 0.02
    damping: float = 1e-4
    time_shift: float = math.sqrt(196608.0 / 4096.0)


@dataclass
class RAEAuditResult:
    residual_table: pd.DataFrame
    basis_control_table: pd.DataFrame
    gradient_summary: pd.DataFrame
    gradient_band_table: pd.DataFrame
    cross_band_correlation: pd.DataFrame
    decoder_sensitivity: pd.DataFrame
    metadata: Dict[str, object]


def configure_fp32(seed: int = 0) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def dct_matrix(size: int, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Return the orthonormal DCT-II matrix."""

    n = torch.arange(int(size), dtype=dtype)[None, :]
    k = torch.arange(int(size), dtype=dtype)[:, None]
    matrix = torch.cos(math.pi * (n + 0.5) * k / int(size))
    matrix[0] *= math.sqrt(1.0 / int(size))
    if size > 1:
        matrix[1:] *= math.sqrt(2.0 / int(size))
    return matrix


def dct2_basis(size: int, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Return row-wise orthonormal 2-D DCT basis vectors for flattened grids."""

    one_dimensional = dct_matrix(size, dtype=dtype)
    return torch.kron(one_dimensional, one_dimensional)


def random_orthogonal_basis(size: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    matrix = torch.randn((size * size, size * size), generator=generator, dtype=torch.float64)
    basis, _ = torch.linalg.qr(matrix)
    return basis.T.contiguous()


def radial_band_masks(size: int, band_count: int) -> torch.Tensor:
    """Partition DCT coefficients into equal-cardinality radial bands."""

    if band_count < 1 or band_count > size * size:
        raise ValueError("band_count must lie in [1, size**2]")
    coordinates = torch.cartesian_prod(torch.arange(size), torch.arange(size))
    radius = coordinates[:, 0].float().square() + coordinates[:, 1].float().square()
    order = torch.argsort(radius, stable=True)
    masks = torch.zeros((band_count, size * size), dtype=torch.bool)
    for band, indices in enumerate(torch.tensor_split(order, band_count)):
        masks[band, indices] = True
    return masks


def contiguous_channel_masks(channel_count: int, band_count: int) -> torch.Tensor:
    if band_count < 1 or band_count > channel_count:
        raise ValueError("band_count must lie in [1, channel_count]")
    masks = torch.zeros((band_count, channel_count), dtype=torch.bool)
    for band, indices in enumerate(torch.tensor_split(torch.arange(channel_count), band_count)):
        masks[band, indices] = True
    return masks


def spatial_transform(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Transform ``[B,C,H,W]`` to coefficients ``[B,C,H*W]``."""

    if x.ndim != 4 or x.shape[-1] != x.shape[-2]:
        raise ValueError("x must have shape [B,C,H,W] with a square grid")
    flattened = x.flatten(2)
    return torch.matmul(flattened, basis.to(device=x.device, dtype=x.dtype).T)


def inverse_spatial_transform(coefficients: torch.Tensor, basis: torch.Tensor, size: int) -> torch.Tensor:
    flattened = torch.matmul(coefficients, basis.to(coefficients).contiguous())
    return flattened.reshape(coefficients.shape[0], coefficients.shape[1], size, size)


def sample_shifted_logit_normal(
    count: int,
    shift: float,
    *,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    base = torch.sigmoid(torch.randn(int(count), device=device, generator=generator))
    return shift * base / (1.0 + (shift - 1.0) * base)


def shifted_time_quantiles(count: int, shift: float) -> torch.Tensor:
    probability = torch.linspace(0.05, 0.95, int(count), dtype=torch.float64)
    normal = torch.distributions.Normal(torch.tensor(0.0), torch.tensor(1.0))
    base = torch.sigmoid(normal.icdf(probability))
    return (shift * base / (1.0 + (shift - 1.0) * base)).float()


def linear_skip_coefficient(second_moment: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return (t - (1.0 - t) * second_moment) / (
        (1.0 - t).square() * second_moment + t.square()
    ).clamp_min(1e-12)


def linear_residual_variance(second_moment: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return second_moment / (
        (1.0 - t).square() * second_moment + t.square()
    ).clamp_min(1e-12)


def project_spatial_bands(
    x: torch.Tensor,
    basis: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    """Return orthogonal spatial projections with shape ``[B,K,C,H,W]``."""

    size = x.shape[-1]
    coefficients = spatial_transform(x, basis)
    projected = coefficients[:, None] * masks.to(x.device, x.dtype)[None, :, None, :]
    flat = torch.matmul(projected, basis.to(x.device, x.dtype))
    return flat.reshape(x.shape[0], masks.shape[0], x.shape[1], size, size)


def head_gradient_sketch(
    error: torch.Tensor,
    hidden: torch.Tensor,
    output_projection: torch.Tensor,
    hidden_projection: torch.Tensor,
) -> torch.Tensor:
    """Sketch each sample's final-linear gradient without backpropagating the DiT.

    ``error`` is the output-space loss residual ``[B,C,H,W]`` and ``hidden`` is
    the captured input to the final linear layer ``[B,H*W,D]``.  The returned
    tensor is ``A^T grad(W) B`` for each sample, including the derivative of a
    per-sample mean-squared loss.
    """

    token_error = error.permute(0, 2, 3, 1).flatten(1, 2)
    if token_error.shape[:2] != hidden.shape[:2]:
        raise ValueError("error token count and hidden token count do not match")
    projected_error = token_error @ output_projection.to(error)
    projected_hidden = hidden @ hidden_projection.to(hidden)
    scale = 2.0 / float(token_error.shape[1] * token_error.shape[2])
    return scale * torch.einsum("btr,bts->brs", projected_error, projected_hidden)


def load_cached_latents(config: RAEAuditConfig) -> Dict[str, object]:
    payload = torch.load(Path(config.cache_path).expanduser(), map_location="cpu", weights_only=False)
    required = {"train_latents", "val_latents", "latent_scale", "val_indices"}
    missing = required.difference(payload)
    if missing:
        raise KeyError(f"latent cache is missing keys: {sorted(missing)}")
    scale = float(payload["latent_scale"])
    train = payload["train_latents"][: config.train_count].float() * scale
    validation = payload["val_latents"][: config.validation_count].float() * scale
    if len(train) < config.train_count or len(validation) < config.validation_count:
        raise ValueError("requested more cached latents than are available")
    return {
        "train": train.contiguous(),
        "validation": validation.contiguous(),
        "validation_indices": list(payload["val_indices"][: config.validation_count]),
        "latent_scale": scale,
        "cache_metadata": payload.get("metadata", {}),
    }


def load_validation_labels(dataset_path: str, indices: Sequence[int]) -> torch.Tensor:
    dataset = load_named_dataset(
        "imagenet_parquet",
        root=dataset_path,
        split="validation",
        dataset_path=dataset_path,
    )
    return torch.tensor([int(dataset[int(index)][1]) for index in indices], dtype=torch.long)


def fit_channel_pca(
    train_latents: torch.Tensor,
    token_count: int,
    *,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit a full channel PCA from a bounded random sample of latent tokens."""

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    total_tokens = train_latents.shape[0] * train_latents.shape[-2] * train_latents.shape[-1]
    count = min(int(token_count), int(total_tokens))
    indices = torch.randperm(total_tokens, generator=generator)[:count]
    tokens = train_latents.permute(0, 2, 3, 1).reshape(-1, train_latents.shape[1])[indices]
    tokens = tokens.to(device)
    mean = tokens.mean(dim=0)
    centered = tokens - mean
    covariance = centered.T @ centered / max(1, count - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    return eigenvectors[:, order].cpu(), eigenvalues[order].cpu()


def _instantiate_stage2(config: RAEAuditConfig, device: torch.device) -> torch.nn.Module:
    repo = Path(config.rae_repo_path).expanduser().resolve()
    source = str(repo / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    from omegaconf import OmegaConf
    from utils.model_utils import instantiate_from_config

    stage2_config = OmegaConf.load(str(Path(config.stage2_config_path).expanduser()))
    model = instantiate_from_config(stage2_config.stage_2).to(device=device, dtype=torch.float32)
    state = torch.load(
        Path(config.stage2_checkpoint_path).expanduser(),
        map_location="cpu",
        weights_only=False,
    )
    if isinstance(state, Mapping) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False).eval()
    return model


def _band_second_moments(
    latents: torch.Tensor,
    spatial_basis: torch.Tensor,
    spatial_masks: torch.Tensor,
    channel_basis: torch.Tensor,
    channel_masks: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int = 16,
) -> torch.Tensor:
    sums = torch.zeros(
        (spatial_masks.shape[0], channel_masks.shape[0]), device=device, dtype=torch.float64
    )
    counts = torch.zeros_like(sums)
    spatial_masks_device = spatial_masks.to(device)
    channel_masks_device = channel_masks.to(device)
    channel_basis_device = channel_basis.to(device=device, dtype=torch.float32)
    for start in range(0, len(latents), batch_size):
        z = latents[start : start + batch_size].to(device)
        coefficients = spatial_transform(z, spatial_basis)
        coefficients = torch.einsum("bcs,ck->bks", coefficients, channel_basis_device)
        squared = coefficients.double().square()
        for spatial_band, spatial_mask in enumerate(spatial_masks_device):
            for channel_band, channel_mask in enumerate(channel_masks_device):
                selected = squared[:, channel_mask][:, :, spatial_mask]
                sums[spatial_band, channel_band] += selected.sum()
                counts[spatial_band, channel_band] += selected.numel()
    return (sums / counts.clamp_min(1.0)).float().cpu()


def _spatial_second_moments(
    latents: torch.Tensor,
    basis: torch.Tensor,
    masks: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int = 16,
) -> torch.Tensor:
    sums = torch.zeros(masks.shape[0], dtype=torch.float64, device=device)
    counts = torch.zeros_like(sums)
    masks_device = masks.to(device)
    for start in range(0, len(latents), batch_size):
        coefficients = spatial_transform(latents[start : start + batch_size].to(device), basis)
        squared = coefficients.double().square()
        for band, mask in enumerate(masks_device):
            selected = squared[:, :, mask]
            sums[band] += selected.sum()
            counts[band] += selected.numel()
    return (sums / counts.clamp_min(1.0)).float().cpu()


def basis_control_statistics(
    train_latents: torch.Tensor,
    times: torch.Tensor,
    dct_basis: torch.Tensor,
    random_basis: torch.Tensor,
    masks: torch.Tensor,
    *,
    device: torch.device,
) -> pd.DataFrame:
    rows = []
    for basis_name, basis in (("DCT", dct_basis), ("random_orthogonal", random_basis)):
        second_moment = _spatial_second_moments(train_latents, basis, masks, device=device)
        for time_index, time_value in enumerate(times):
            residual = linear_residual_variance(second_moment, time_value)
            for band in range(len(second_moment)):
                rows.append(
                    {
                        "basis": basis_name,
                        "time_index": time_index,
                        "t": float(time_value),
                        "spatial_band": band,
                        "second_moment": float(second_moment[band]),
                        "linear_residual_variance": float(residual[band]),
                    }
                )
    return pd.DataFrame(rows)


@torch.no_grad()
def residual_predictability_table(
    model: torch.nn.Module,
    validation_latents: torch.Tensor,
    labels: torch.Tensor,
    times: torch.Tensor,
    second_moment: torch.Tensor,
    spatial_basis: torch.Tensor,
    spatial_masks: torch.Tensor,
    channel_basis: torch.Tensor,
    channel_masks: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    generator = torch.Generator(device=device).manual_seed(70_001 + int(seed))
    channel_basis_device = channel_basis.to(device=device, dtype=torch.float32)
    spatial_masks_device = spatial_masks.to(device)
    channel_masks_device = channel_masks.to(device)
    for time_index, time_value_cpu in enumerate(times):
        time_value = time_value_cpu.to(device)
        sums_linear = torch.zeros_like(second_moment, device=device, dtype=torch.float64)
        sums_teacher = torch.zeros_like(sums_linear)
        counts = torch.zeros_like(sums_linear)
        skip = linear_skip_coefficient(second_moment.to(device), time_value)
        for start in range(0, len(validation_latents), batch_size):
            z = validation_latents[start : start + batch_size].to(device)
            y = labels[start : start + batch_size].to(device)
            noise = torch.randn(z.shape, device=device, generator=generator)
            t = torch.full((len(z),), float(time_value), device=device)
            x_t = (1.0 - time_value) * z + time_value * noise
            target = noise - z
            prediction = model(x_t, t, y)
            target_coefficients = spatial_transform(target, spatial_basis)
            input_coefficients = spatial_transform(x_t, spatial_basis)
            error_coefficients = spatial_transform(prediction - target, spatial_basis)
            target_coefficients = torch.einsum(
                "bcs,ck->bks", target_coefficients, channel_basis_device
            )
            input_coefficients = torch.einsum(
                "bcs,ck->bks", input_coefficients, channel_basis_device
            )
            error_coefficients = torch.einsum(
                "bcs,ck->bks", error_coefficients, channel_basis_device
            )
            for spatial_band, spatial_mask in enumerate(spatial_masks_device):
                for channel_band, channel_mask in enumerate(channel_masks_device):
                    linear_error = (
                        target_coefficients[:, channel_mask][:, :, spatial_mask]
                        - skip[spatial_band, channel_band]
                        * input_coefficients[:, channel_mask][:, :, spatial_mask]
                    )
                    teacher_error = error_coefficients[:, channel_mask][:, :, spatial_mask]
                    sums_linear[spatial_band, channel_band] += linear_error.double().square().sum()
                    sums_teacher[spatial_band, channel_band] += teacher_error.double().square().sum()
                    counts[spatial_band, channel_band] += linear_error.numel()
        linear_mse = (sums_linear / counts).cpu()
        teacher_mse = (sums_teacher / counts).cpu()
        for spatial_band in range(spatial_masks.shape[0]):
            for channel_band in range(channel_masks.shape[0]):
                residual_value = float(linear_mse[spatial_band, channel_band])
                teacher_value = float(teacher_mse[spatial_band, channel_band])
                rho = 1.0 - teacher_value / max(residual_value, 1e-12)
                rows.append(
                    {
                        "time_index": time_index,
                        "t": float(time_value),
                        "spatial_band": spatial_band,
                        "channel_pca_band": channel_band,
                        "second_moment": float(second_moment[spatial_band, channel_band]),
                        "linear_residual_mse": residual_value,
                        "teacher_mse": teacher_value,
                        "rho_lower_raw": rho,
                        "rho_lower_clipped": max(0.0, rho),
                    }
                )
    return pd.DataFrame(rows)


def _spatial_residual_scale(second_moment: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Return R_b(t) with shape ``[B,K]``."""

    moment = second_moment.to(t.device, t.dtype)[None]
    time = t[:, None]
    return linear_residual_variance(moment, time)


def _weight_table(
    residual_scale: torch.Tensor,
    treatment: GateTreatment,
    global_normalizer: float,
    damping: float,
) -> torch.Tensor:
    if treatment.mode == "baseline" or treatment.gamma == 0.0:
        return torch.ones_like(residual_scale)
    raw = (residual_scale + damping).pow(-float(treatment.gamma))
    temporal = raw.mean(dim=1, keepdim=True)
    if treatment.mode == "direction":
        return raw / temporal.clamp_min(1e-12)
    if treatment.mode == "time":
        return (temporal / global_normalizer).expand_as(raw)
    if treatment.mode == "full":
        return raw / global_normalizer
    raise ValueError(f"unknown treatment mode: {treatment.mode}")


def _actual_time_normalizer(
    second_moment: torch.Tensor,
    treatment: GateTreatment,
    config: RAEAuditConfig,
    device: torch.device,
) -> float:
    if treatment.gamma == 0.0:
        return 1.0
    generator = torch.Generator(device=device).manual_seed(90_001 + config.seed)
    times = sample_shifted_logit_normal(
        65_536, config.time_shift, device=device, generator=generator
    )
    residual = _spatial_residual_scale(second_moment, times)
    return float((residual + config.damping).pow(-float(treatment.gamma)).mean())


def _gradient_statistics(vectors: torch.Tensor) -> Dict[str, float]:
    vectors = vectors.double()
    mean = vectors.mean(dim=0)
    centered = vectors - mean
    variance = float(centered.square().sum(dim=1).sum() / max(1, len(vectors) - 1))
    mean_norm_squared = float(mean.square().sum())
    return {
        "mean_norm_squared": mean_norm_squared,
        "variance_trace": variance,
        "gsnr": mean_norm_squared / max(variance, 1e-30),
        "gradient_norm_mean": float(vectors.norm(dim=1).mean()),
    }


@torch.no_grad()
def gradient_noise_audit(
    model: torch.nn.Module,
    validation_latents: torch.Tensor,
    labels: torch.Tensor,
    spatial_basis: torch.Tensor,
    spatial_masks: torch.Tensor,
    spatial_second_moment: torch.Tensor,
    treatments: Sequence[GateTreatment],
    config: RAEAuditConfig,
    *,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Measure projected final-head gradients from actual official-model errors."""

    rank = int(config.gradient_sketch_rank)
    generator = torch.Generator(device=device).manual_seed(100_003 + config.seed)
    output_projection = torch.randn((768, rank), device=device, generator=generator) / math.sqrt(rank)
    hidden_projection = torch.randn((2048, rank), device=device, generator=generator) / math.sqrt(rank)
    normalizers = {
        treatment.name: _actual_time_normalizer(
            spatial_second_moment, treatment, config, device
        )
        for treatment in treatments
    }
    hidden_holder: list[torch.Tensor] = []

    def capture_hidden(_module, args):
        hidden_holder.append(args[0].detach())

    handle = model.final_layer.linear.register_forward_pre_hook(capture_hidden)
    sample_vectors: Dict[str, list[torch.Tensor]] = {treatment.name: [] for treatment in treatments}
    band_vectors: Dict[str, list[torch.Tensor]] = {treatment.name: [] for treatment in treatments}
    residual_rows = []
    try:
        for microbatch in range(int(config.gradient_microbatches)):
            batch_size = int(config.model_batch_size)
            start = (microbatch * batch_size) % len(validation_latents)
            indices = torch.arange(start, start + batch_size) % len(validation_latents)
            z = validation_latents[indices].to(device)
            y = labels[indices].to(device)
            noise = torch.randn(z.shape, device=device, generator=generator)
            t = sample_shifted_logit_normal(
                batch_size, config.time_shift, device=device, generator=generator
            )
            x_t = (1.0 - t[:, None, None, None]) * z + t[:, None, None, None] * noise
            target = noise - z
            hidden_holder.clear()
            prediction = model(x_t, t, y)
            if len(hidden_holder) != 1:
                raise RuntimeError("expected exactly one final-linear hook call")
            hidden = hidden_holder[0]
            error_bands = project_spatial_bands(
                prediction - target, spatial_basis, spatial_masks
            )
            sketches = []
            for band in range(spatial_masks.shape[0]):
                sketches.append(
                    head_gradient_sketch(
                        error_bands[:, band], hidden, output_projection, hidden_projection
                    )
                )
            sketches_tensor = torch.stack(sketches, dim=1)
            residual_scale = _spatial_residual_scale(spatial_second_moment, t)
            residual_rows.append(residual_scale.cpu())
            for treatment in treatments:
                weight = _weight_table(
                    residual_scale,
                    treatment,
                    normalizers[treatment.name],
                    config.damping,
                )
                weighted_bands = sketches_tensor * weight[:, :, None, None]
                total = weighted_bands.sum(dim=1)
                sample_vectors[treatment.name].append(total.flatten(1).cpu())
                band_vectors[treatment.name].append(weighted_bands.flatten(2).cpu())
    finally:
        handle.remove()

    all_residual = torch.cat(residual_rows, dim=0)
    summary_rows = []
    band_rows = []
    vector_tables: Dict[str, torch.Tensor] = {}
    band_tables: Dict[str, torch.Tensor] = {}
    for treatment in treatments:
        vectors = torch.cat(sample_vectors[treatment.name], dim=0)
        bands = torch.cat(band_vectors[treatment.name], dim=0)
        vector_tables[treatment.name] = vectors
        band_tables[treatment.name] = bands
        for effective_batch in (4, 16, 64):
            group_count = len(vectors) // effective_batch
            if group_count < 2:
                continue
            grouped = vectors[: group_count * effective_batch].reshape(
                group_count, effective_batch, -1
            ).mean(dim=1)
            stats = _gradient_statistics(grouped)
            summary_rows.append(
                {
                    "treatment": treatment.name,
                    "mode": treatment.mode,
                    "gamma": treatment.gamma,
                    "effective_batch": effective_batch,
                    "estimate": "empirical",
                    **stats,
                }
            )
        micro_stats = _gradient_statistics(vectors)
        summary_rows.append(
            {
                "treatment": treatment.name,
                "mode": treatment.mode,
                "gamma": treatment.gamma,
                "effective_batch": 1024,
                "estimate": "iid_projection",
                "mean_norm_squared": micro_stats["mean_norm_squared"],
                "variance_trace": micro_stats["variance_trace"] / 1024.0,
                "gsnr": micro_stats["gsnr"] * 1024.0,
                "gradient_norm_mean": float("nan"),
            }
        )
        for band in range(bands.shape[1]):
            stats = _gradient_statistics(bands[:, band])
            band_rows.append(
                {
                    "treatment": treatment.name,
                    "mode": treatment.mode,
                    "gamma": treatment.gamma,
                    "spatial_band": band,
                    "mean_linear_residual_variance": float(all_residual[:, band].mean()),
                    **stats,
                }
            )

    correlation_rows = []
    selected = [name for name in ("baseline:gamma=0", "direction:gamma=0.5") if name in band_tables]
    for treatment_name in selected:
        bands = band_tables[treatment_name].double()
        centered = bands - bands.mean(dim=0, keepdim=True)
        flat = centered.permute(1, 0, 2).reshape(centered.shape[1], -1)
        correlation = flat @ flat.T
        denominator = torch.sqrt(torch.diag(correlation).clamp_min(1e-30))
        correlation = correlation / (denominator[:, None] * denominator[None, :])
        for first in range(correlation.shape[0]):
            for second in range(correlation.shape[1]):
                correlation_rows.append(
                    {
                        "treatment": treatment_name,
                        "band_i": first,
                        "band_j": second,
                        "correlation": float(correlation[first, second]),
                    }
                )
    return pd.DataFrame(summary_rows), pd.DataFrame(band_rows), pd.DataFrame(correlation_rows)


@torch.no_grad()
def decoder_band_sensitivity(
    validation_latents: torch.Tensor,
    spatial_basis: torch.Tensor,
    spatial_masks: torch.Tensor,
    config: RAEAuditConfig,
    *,
    device: torch.device,
) -> pd.DataFrame:
    adapter = load_rae_adapter(
        "rae_dinov2",
        repo_path=config.rae_repo_path,
        device=device,
        dtype=torch.float32,
    )
    generator = torch.Generator(device=device).manual_seed(120_011 + config.seed)
    z = validation_latents[: config.decoder_sample_count].to(device)
    rows = []
    for band, mask in enumerate(spatial_masks):
        coefficients = torch.randn(
            (len(z), z.shape[1], z.shape[-1] * z.shape[-1]),
            device=device,
            generator=generator,
        )
        coefficients *= mask.to(device=device, dtype=coefficients.dtype)[None, None]
        perturbation = inverse_spatial_transform(coefficients, spatial_basis, z.shape[-1])
        perturbation /= perturbation.square().mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1e-8)
        epsilon = float(config.decoder_epsilon)
        decoded_plus = adapter.decode(z + epsilon * perturbation)
        decoded_minus = adapter.decode(z - epsilon * perturbation)
        sensitivity = (decoded_plus - decoded_minus).square().mean(dim=(1, 2, 3)) / (4.0 * epsilon**2)
        rows.append(
            {
                "spatial_band": band,
                "pixel_sensitivity_mean": float(sensitivity.mean()),
                "pixel_sensitivity_std": float(sensitivity.std(unbiased=False)),
                "sample_count": len(z),
                "epsilon": epsilon,
            }
        )
    del adapter
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def decoder_sensitivity_sweep(
    validation_latents: torch.Tensor,
    spatial_basis: torch.Tensor,
    spatial_masks: torch.Tensor,
    config: RAEAuditConfig,
    epsilons: Iterable[float] = (0.01, 0.02, 0.04),
    *,
    device: torch.device,
) -> pd.DataFrame:
    """Repeat the symmetric finite difference at matched perturbation scales."""

    frames = []
    for epsilon in epsilons:
        frames.append(
            decoder_band_sensitivity(
                validation_latents,
                spatial_basis,
                spatial_masks,
                replace(config, decoder_epsilon=float(epsilon)),
                device=device,
            )
        )
    return pd.concat(frames, ignore_index=True)


def run_rae_spectral_gradient_audit(
    config: RAEAuditConfig = RAEAuditConfig(),
    *,
    treatments: Sequence[GateTreatment] | None = None,
    include_decoder: bool = True,
    verbose: bool = True,
) -> RAEAuditResult:
    """Run the complete audit and return in-memory tables without repo outputs."""

    configure_fp32(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    payload = load_cached_latents(config)
    train_latents = payload["train"]
    validation_latents = payload["validation"]
    labels = load_validation_labels(config.dataset_path, payload["validation_indices"])
    size = train_latents.shape[-1]
    spatial_basis = dct2_basis(size).float()
    random_basis = random_orthogonal_basis(size, seed=config.seed + 31).float()
    spatial_masks = radial_band_masks(size, config.spatial_band_count)
    channel_masks = contiguous_channel_masks(train_latents.shape[1], config.channel_band_count)
    times = shifted_time_quantiles(config.time_bin_count, config.time_shift)
    if verbose:
        print("1/5 fitting channel PCA and band moments")
    channel_basis, channel_eigenvalues = fit_channel_pca(
        train_latents,
        config.pca_token_count,
        seed=config.seed,
        device=device,
    )
    second_moment = _band_second_moments(
        train_latents,
        spatial_basis,
        spatial_masks,
        channel_basis,
        channel_masks,
        device=device,
    )
    spatial_second_moment = _spatial_second_moments(
        train_latents, spatial_basis, spatial_masks, device=device
    )
    if verbose:
        print("2/5 comparing DCT with a random orthogonal basis")
    basis_control = basis_control_statistics(
        train_latents,
        times,
        spatial_basis,
        random_basis,
        spatial_masks,
        device=device,
    )
    if verbose:
        print("3/5 loading official DiTDH-S checkpoint and measuring held-out errors")
    model = _instantiate_stage2(config, device)
    residual_table = residual_predictability_table(
        model,
        validation_latents,
        labels,
        times,
        second_moment,
        spatial_basis,
        spatial_masks,
        channel_basis,
        channel_masks,
        device=device,
        batch_size=config.model_batch_size,
        seed=config.seed,
    )
    if treatments is None:
        treatments = (
            GateTreatment("baseline", 0.0),
            GateTreatment("time", 0.5),
            GateTreatment("direction", 0.5),
            GateTreatment("full", 0.5),
            GateTreatment("direction", 1.0),
            GateTreatment("full", 1.0),
        )
    if verbose:
        print("4/5 measuring actual final-head gradient noise")
    gradient_summary, gradient_band_table, cross_band = gradient_noise_audit(
        model,
        validation_latents,
        labels,
        spatial_basis,
        spatial_masks,
        spatial_second_moment,
        treatments,
        config,
        device=device,
    )
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if include_decoder:
        if verbose:
            print("5/5 measuring frozen decoder sensitivity")
        decoder_sensitivity = decoder_band_sensitivity(
            validation_latents,
            spatial_basis,
            spatial_masks,
            config,
            device=device,
        )
    else:
        decoder_sensitivity = pd.DataFrame()
    metadata = {
        "device": str(device),
        "dtype": "float32",
        "train_count": len(train_latents),
        "validation_count": len(validation_latents),
        "latent_scale_recovered": payload["latent_scale"],
        "time_shift": config.time_shift,
        "time_quantiles": [float(value) for value in times],
        "channel_pca_explained_top_quarter": float(
            channel_eigenvalues[: len(channel_eigenvalues) // 4].sum()
            / channel_eigenvalues.sum().clamp_min(1e-12)
        ),
        "gradient_sketch_rank": config.gradient_sketch_rank,
        "gradient_samples": config.gradient_microbatches * config.model_batch_size,
        "checkpoint": str(Path(config.stage2_checkpoint_path).expanduser()),
        "cache_metadata": payload["cache_metadata"],
    }
    return RAEAuditResult(
        residual_table=residual_table,
        basis_control_table=basis_control,
        gradient_summary=gradient_summary,
        gradient_band_table=gradient_band_table,
        cross_band_correlation=cross_band,
        decoder_sensitivity=decoder_sensitivity,
        metadata=metadata,
    )


__all__ = [
    "RAEAuditConfig",
    "RAEAuditResult",
    "basis_control_statistics",
    "contiguous_channel_masks",
    "dct2_basis",
    "dct_matrix",
    "decoder_band_sensitivity",
    "decoder_sensitivity_sweep",
    "gradient_noise_audit",
    "head_gradient_sketch",
    "inverse_spatial_transform",
    "linear_residual_variance",
    "linear_skip_coefficient",
    "project_spatial_bands",
    "radial_band_masks",
    "random_orthogonal_basis",
    "run_rae_spectral_gradient_audit",
    "sample_shifted_logit_normal",
    "shifted_time_quantiles",
    "spatial_transform",
]
