"""Locate the gap between teacher-forced RAE gains and ODE generation.

This diagnostic deliberately uses two different comparison rules:

* On the linear flow-matching interpolation, ``z0_hat = z_t - t * v_hat``
  is paired with the known clean latent and can be evaluated sample by sample.
* On a self-generated ODE trajectory there is no valid pairing with an
  arbitrary validation image.  Rollout states are therefore compared with the
  corresponding validation interpolation *distribution*, not by paired MSE.

The script is a no-training audit.  It loads one branch's EMA stage-2 model,
the frozen official RAE decoder, and cached ImageNet validation latents.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
RAE_ROOT = ROOT / "external" / "RAE"
RAE_SRC = RAE_ROOT / "src"
for import_path in (ROOT, RAE_SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from experiments.rae_spectral_direction_loss import DCTDirectionLoss  # noqa: E402
from experiments.rae_spectral_gradient_audit import (  # noqa: E402
    RAEAuditConfig,
    load_cached_latents,
    load_validation_labels,
)


DEFAULT_RESULTS = Path.home() / "data/eqvae/experiments/rae_spectral_tiny"
DEFAULT_EVALUATION_SEED = 104729
DEFAULT_TARGET_TIMES = (0.95, 0.85, 0.70, 0.55, 0.30, 0.10, 0.0)


@dataclass(frozen=True)
class GapStudyConfig:
    branch: Path
    device: str = "cuda:0"
    count: int = 64
    batch_size: int = 4
    perceptual_count: int = 12
    perceptual_batch_size: int = 2
    evaluation_seed: int = DEFAULT_EVALUATION_SEED
    num_steps: int = 50
    target_times: tuple[float, ...] = DEFAULT_TARGET_TIMES
    summary_projection_dim: int = 32
    swd_directions: int = 64


@dataclass
class DecoderFeatures:
    image: torch.Tensor
    hidden: tuple[torch.Tensor, ...]
    inception: torch.Tensor


class FrozenRAEDecoder(torch.nn.Module):
    """The exact official decoder path without loading the unused encoder."""

    def __init__(
        self,
        decoder: torch.nn.Module,
        *,
        encoder_mean: Sequence[float],
        encoder_std: Sequence[float],
        latent_mean: torch.Tensor | None,
        latent_var: torch.Tensor | None,
        eps: float,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.reshape_to_2d = True
        self.do_normalization = latent_var is not None
        self.eps = float(eps)
        self.register_buffer(
            "encoder_mean", torch.tensor(encoder_mean, dtype=torch.float32).reshape(1, 3, 1, 1)
        )
        self.register_buffer(
            "encoder_std", torch.tensor(encoder_std, dtype=torch.float32).reshape(1, 3, 1, 1)
        )
        if latent_mean is None:
            self.latent_mean = None
        else:
            self.register_buffer("latent_mean", latent_mean.float())
        if latent_var is None:
            self.latent_var = None
        else:
            self.register_buffer("latent_var", latent_var.float())

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        z = latent
        if self.do_normalization:
            latent_mean = self.latent_mean.to(z) if self.latent_mean is not None else 0
            latent_var = self.latent_var.to(z) if self.latent_var is not None else 1
            z = z * torch.sqrt(latent_var + self.eps) + latent_mean
        if self.reshape_to_2d:
            batch, channels, height, width = z.shape
            z = z.reshape(batch, channels, height * width).transpose(1, 2)
        output = self.decoder(z, drop_cls_token=False)
        image = self.decoder.unpatchify(output.logits)
        return image * self.encoder_std.to(image) + self.encoder_mean.to(image)


def configure_fp32(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.set_float32_matmul_precision("highest")


def official_time_grid(
    num_steps: int = 50,
    *,
    time_shift: float = math.sqrt(196608.0 / 4096.0),
    t0: float = 0.0,
    t1: float = 0.999,
) -> torch.Tensor:
    """Reproduce the RAE sampler's shifted descending ODE time grid."""

    if int(num_steps) < 2:
        raise ValueError("num_steps must be at least two")
    if not 0.0 <= float(t0) < float(t1) <= 1.0:
        raise ValueError("expected 0 <= t0 < t1 <= 1")
    raw = 1.0 - torch.linspace(float(t0), float(t1), int(num_steps), dtype=torch.float64)
    shifted = float(time_shift) * raw / (1.0 + (float(time_shift) - 1.0) * raw)
    return shifted.float()


def select_time_indices(times: torch.Tensor, targets: Sequence[float]) -> list[int]:
    selected: list[int] = []
    for target in targets:
        index = int(torch.argmin((times - float(target)).abs()).item())
        if index not in selected:
            selected.append(index)
    return sorted(selected)


def clean_from_velocity(state: torch.Tensor, velocity: torch.Tensor, time: torch.Tensor | float) -> torch.Tensor:
    """One-step clean estimate for the linear FM convention used by RAE."""

    if isinstance(time, torch.Tensor):
        expanded = time.to(state).reshape(-1, *([1] * (state.ndim - 1)))
    else:
        expanded = state.new_tensor(float(time))
    return state - expanded * velocity


@torch.no_grad()
def euler_rollout(
    model: torch.nn.Module,
    initial: torch.Tensor,
    labels: torch.Tensor,
    times: torch.Tensor,
) -> list[torch.Tensor]:
    """Explicit Euler rollout matching torchdiffeq's fixed-step Euler rule."""

    state = initial
    states = [state]
    for current, following in zip(times[:-1], times[1:]):
        batch_time = torch.full(
            (len(state),), float(current), device=state.device, dtype=state.dtype
        )
        velocity = model(state, batch_time, y=labels)
        state = state + (following.to(state) - current.to(state)) * velocity
        states.append(state)
    return states


def fixed_gaussian_matrix(rows: int, columns: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    matrix = torch.randn((int(rows), int(columns)), generator=generator, dtype=torch.float32)
    return matrix / matrix.square().sum(dim=0, keepdim=True).sqrt().clamp_min(1e-12)


def latent_band_energy(latent: torch.Tensor, analyzer: DCTDirectionLoss) -> torch.Tensor:
    return analyzer.band_mse(latent)


def latent_summary(
    latent: torch.Tensor,
    analyzer: DCTDirectionLoss,
    channel_projection: torch.Tensor,
) -> torch.Tensor:
    """Compact per-sample summary used only for distribution comparisons."""

    channel_mean = latent.mean(dim=(-2, -1))
    projected_mean = channel_mean @ channel_projection.to(channel_mean)
    log_band_energy = latent_band_energy(latent, analyzer).clamp_min(1e-12).log()
    return torch.cat([projected_mean, log_band_energy], dim=1)


def band_prediction_calibration(
    prediction: torch.Tensor,
    target: torch.Tensor,
    analyzer: DCTDirectionLoss,
) -> dict[str, torch.Tensor]:
    """Per-sample calibration of prediction against target in each DCT band."""

    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    predicted = analyzer.transform(prediction).flatten(2)
    expected = analyzer.transform(target).flatten(2)
    band_index = analyzer.band_index.flatten().to(prediction.device)
    rows: dict[str, list[torch.Tensor]] = {
        "prediction_energy_log_ratio_to_target": [],
        "prediction_target_cosine": [],
        "prediction_target_slope": [],
        "velocity_error_mse": [],
    }
    for band in range(analyzer.band_count):
        mask = band_index == band
        predicted_band = predicted[:, :, mask].flatten(1)
        expected_band = expected[:, :, mask].flatten(1)
        predicted_energy = predicted_band.square().mean(dim=1)
        expected_energy = expected_band.square().mean(dim=1)
        cross = (predicted_band * expected_band).mean(dim=1)
        rows["prediction_energy_log_ratio_to_target"].append(
            (predicted_energy / expected_energy.clamp_min(1e-12)).clamp_min(1e-12).log()
        )
        rows["prediction_target_cosine"].append(
            cross
            / (predicted_energy * expected_energy).clamp_min(1e-24).sqrt()
        )
        rows["prediction_target_slope"].append(cross / expected_energy.clamp_min(1e-12))
        rows["velocity_error_mse"].append(
            (predicted_band - expected_band).square().mean(dim=1)
        )
    return {name: torch.stack(values, dim=1) for name, values in rows.items()}


def sliced_wasserstein(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    directions: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Standardized equal-sample sliced Wasserstein distance."""

    if reference.ndim != 2 or candidate.shape != reference.shape:
        raise ValueError("reference and candidate must have the same [N,D] shape")
    if directions.ndim != 2 or directions.shape[0] != reference.shape[1]:
        raise ValueError("directions must have shape [D,K]")
    mean = reference.mean(dim=0, keepdim=True)
    scale = reference.std(dim=0, unbiased=False, keepdim=True).clamp_min(float(epsilon))
    ref_projection = ((reference - mean) / scale) @ directions.to(reference)
    candidate_projection = ((candidate - mean) / scale) @ directions.to(candidate)
    ref_projection = ref_projection.sort(dim=0).values
    candidate_projection = candidate_projection.sort(dim=0).values
    return (ref_projection - candidate_projection).abs().mean()


def _absolute_local_path(value: str, base: Path) -> str:
    path = Path(str(value)).expanduser()
    if path.is_absolute() or str(value).startswith(("facebook/", "google/", "openai/")):
        return str(value)
    return str((base / path).resolve())


def _infer_decoder_input_channels(stage_1: OmegaConf, stats: Mapping[str, torch.Tensor]) -> int:
    configured = OmegaConf.select(stage_1, "params.encoder_params.hidden_size")
    if configured is not None:
        return int(configured)
    latent_mean = stats.get("mean")
    if latent_mean is None or latent_mean.ndim < 1:
        raise ValueError(
            "cannot infer decoder input channels: configure "
            "stage_1.params.encoder_params.hidden_size or provide latent mean statistics"
        )
    return int(latent_mean.shape[0])


def load_frozen_decoder(stage_1: OmegaConf) -> FrozenRAEDecoder:
    from stage1.decoders import GeneralDecoder
    from transformers import ViTMAEConfig

    params = stage_1.params
    decoder_path = Path(_absolute_local_path(params.decoder_config_path, RAE_ROOT))
    weights_path = Path(_absolute_local_path(params.pretrained_decoder_path, RAE_ROOT))
    stats_path = Path(_absolute_local_path(params.normalization_stat_path, RAE_ROOT))
    patch_size = int(params.get("decoder_patch_size", 16))
    stats = torch.load(stats_path, map_location="cpu", weights_only=True)
    latent_dim = _infer_decoder_input_channels(stage_1, stats)
    num_patches = 16 * 16

    decoder_payload = json.loads((decoder_path / "config.json").read_text(encoding="utf-8"))
    decoder_payload["patch_size"] = patch_size
    decoder_config = ViTMAEConfig.from_dict(decoder_payload)
    decoder_config.hidden_size = latent_dim
    decoder_config.patch_size = patch_size
    decoder_config.image_size = int(patch_size * math.sqrt(num_patches))
    decoder = GeneralDecoder(decoder_config, num_patches=num_patches)
    decoder_state = torch.load(weights_path, map_location="cpu", weights_only=True)
    decoder.load_state_dict(decoder_state, strict=True)

    return FrozenRAEDecoder(
        decoder,
        encoder_mean=(0.485, 0.456, 0.406),
        encoder_std=(0.229, 0.224, 0.225),
        latent_mean=stats.get("mean"),
        latent_var=stats.get("var"),
        eps=float(params.get("eps", 1e-5)),
    )


def load_models(branch: Path, device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module, OmegaConf]:
    from utils.model_utils import instantiate_from_config

    config = OmegaConf.load(branch / "config.yaml")
    stage_1 = OmegaConf.create(OmegaConf.to_container(config.stage_1, resolve=True))
    rae = load_frozen_decoder(stage_1)
    rae = rae.to(device=device, dtype=torch.float32).requires_grad_(False).eval()

    stage_2 = OmegaConf.create(OmegaConf.to_container(config.stage_2, resolve=True))
    model = instantiate_from_config(stage_2).to(device=device, dtype=torch.float32)
    ema_path = branch / "generation" / "ema_step-0010000.pt"
    if not ema_path.exists():
        raise FileNotFoundError(f"materialized EMA checkpoint is missing: {ema_path}")
    state = torch.load(ema_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False).eval()
    return model, rae, config


def load_inception(device: torch.device) -> torch.nn.Module:
    from torchvision.models import Inception_V3_Weights, inception_v3

    model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    return model.to(device=device, dtype=torch.float32).requires_grad_(False).eval()


def inception_features(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    resized = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
    mean = resized.new_tensor((0.485, 0.456, 0.406)).reshape(1, 3, 1, 1)
    std = resized.new_tensor((0.229, 0.224, 0.225)).reshape(1, 3, 1, 1)
    return model((resized - mean) / std)


@torch.no_grad()
def decode_features(
    rae: torch.nn.Module,
    inception: torch.nn.Module,
    latent: torch.Tensor,
) -> DecoderFeatures:
    z = latent
    if rae.do_normalization:
        latent_mean = rae.latent_mean.to(z.device) if rae.latent_mean is not None else 0
        latent_var = rae.latent_var.to(z.device) if rae.latent_var is not None else 1
        z = z * torch.sqrt(latent_var + rae.eps) + latent_mean
    if rae.reshape_to_2d:
        batch, channels, height, width = z.shape
        z = z.reshape(batch, channels, height * width).transpose(1, 2)
    output = rae.decoder(z, drop_cls_token=False, output_hidden_states=True)
    image = rae.decoder.unpatchify(output.logits)
    image = (image * rae.encoder_std.to(image.device) + rae.encoder_mean.to(image.device)).clamp(0, 1)
    hidden_indices = (0, len(output.hidden_states) // 2, len(output.hidden_states) - 1)
    hidden = tuple(output.hidden_states[int(index)][:, 1:].float() for index in hidden_indices)
    return DecoderFeatures(image=image, hidden=hidden, inception=inception_features(inception, image))


@torch.no_grad()
def decode_features_batched(
    rae: torch.nn.Module,
    inception: torch.nn.Module,
    latent: torch.Tensor,
    batch_size: int,
) -> DecoderFeatures:
    if int(batch_size) < 1:
        raise ValueError("decoder batch_size must be positive")
    chunks = [
        decode_features(rae, inception, latent[start : start + int(batch_size)])
        for start in range(0, len(latent), int(batch_size))
    ]
    return DecoderFeatures(
        image=torch.cat([chunk.image for chunk in chunks]),
        hidden=tuple(
            torch.cat([chunk.hidden[index] for chunk in chunks])
            for index in range(len(chunks[0].hidden))
        ),
        inception=torch.cat([chunk.inception for chunk in chunks]),
    )


def compare_decoder_features(
    candidate: DecoderFeatures,
    reference: DecoderFeatures,
) -> dict[str, torch.Tensor]:
    metrics: dict[str, torch.Tensor] = {
        "pixel_mse": (candidate.image - reference.image).square().flatten(1).mean(dim=1),
        "inception_cosine_distance": 1.0
        - F.cosine_similarity(candidate.inception, reference.inception, dim=1),
    }
    for name, candidate_hidden, reference_hidden in zip(
        ("input", "middle", "final"), candidate.hidden, reference.hidden
    ):
        metrics[f"decoder_hidden_{name}_cosine_distance"] = 1.0 - F.cosine_similarity(
            candidate_hidden, reference_hidden, dim=-1
        ).mean(dim=1)
    return metrics


def _append_values(store: dict[tuple[int, str], list[torch.Tensor]], index: int, values: Mapping[str, torch.Tensor]) -> None:
    for metric, value in values.items():
        store.setdefault((int(index), str(metric)), []).append(value.detach().float().cpu())


def _append_tensor(store: dict[int, list[torch.Tensor]], index: int, value: torch.Tensor) -> None:
    store.setdefault(int(index), []).append(value.detach().float().cpu())


def _distribution_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    directions: torch.Tensor,
) -> tuple[float, float]:
    swd = float(sliced_wasserstein(reference, candidate, directions))
    mean_gap = float((candidate.mean(0) - reference.mean(0)).abs().mean())
    return swd, mean_gap


def _branch_metadata(branch: Path) -> dict[str, object]:
    return json.loads((branch / "manifest.json").read_text(encoding="utf-8"))


@torch.no_grad()
def run_gap_study(
    cfg: GapStudyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if not torch.cuda.is_available() and str(cfg.device).startswith("cuda"):
        raise RuntimeError("CUDA was requested but is unavailable")
    if cfg.count < 2:
        raise ValueError("count must be at least two for distribution diagnostics")
    configure_fp32(cfg.evaluation_seed)
    device = torch.device(cfg.device)
    branch = cfg.branch.expanduser().resolve()
    manifest = _branch_metadata(branch)
    model, rae, model_config = load_models(branch, device)
    inception = load_inception(device) if cfg.perceptual_count > 0 else None

    audit_config = RAEAuditConfig(train_count=1, validation_count=int(cfg.count))
    payload = load_cached_latents(audit_config)
    clean_all = payload["validation"][: cfg.count].float()
    labels_all = load_validation_labels(
        audit_config.dataset_path, payload["validation_indices"][: cfg.count]
    )

    spectral = model_config.training.spectral_direction_loss
    analyzer = DCTDirectionLoss(
        spatial_size=int(spectral.spatial_size),
        second_moments=list(spectral.second_moments),
        gamma=0.0,
        damping=float(spectral.damping),
        min_weight=float(spectral.min_weight),
        max_weight=float(spectral.max_weight),
    ).to(device)
    time_shift = math.sqrt(
        float(model_config.misc.time_dist_shift_dim) / float(model_config.misc.time_dist_shift_base)
    )
    times = official_time_grid(cfg.num_steps, time_shift=time_shift).to(device)
    selected_indices = select_time_indices(times.cpu(), cfg.target_times)

    noise_generator = torch.Generator(device="cpu").manual_seed(int(cfg.evaluation_seed))
    noise_all = torch.randn(clean_all.shape, generator=noise_generator, dtype=torch.float32)
    channel_projection = fixed_gaussian_matrix(
        clean_all.shape[1], cfg.summary_projection_dim, cfg.evaluation_seed + 11
    ).to(device)
    summary_dimension = int(cfg.summary_projection_dim + analyzer.band_count)
    swd_directions = fixed_gaussian_matrix(
        summary_dimension, cfg.swd_directions, cfg.evaluation_seed + 23
    )

    teacher_values: dict[tuple[int, str], list[torch.Tensor]] = {}
    teacher_band_values: dict[tuple[int, str], list[torch.Tensor]] = {}
    rollout_pair_values: dict[tuple[int, str], list[torch.Tensor]] = {}
    rollout_perceptual_values: dict[tuple[int, str], list[torch.Tensor]] = {}
    step_values: dict[tuple[int, str], list[torch.Tensor]] = {}
    state_summaries: dict[int, list[torch.Tensor]] = {}
    teacher_state_summaries: dict[int, list[torch.Tensor]] = {}
    clean_summaries: dict[int, list[torch.Tensor]] = {}
    teacher_clean_summaries: dict[int, list[torch.Tensor]] = {}
    clean_reference_summaries: list[torch.Tensor] = []
    state_bands: dict[int, list[torch.Tensor]] = {}
    teacher_state_bands: dict[int, list[torch.Tensor]] = {}
    clean_bands: dict[int, list[torch.Tensor]] = {}
    teacher_clean_bands: dict[int, list[torch.Tensor]] = {}
    reference_bands: list[torch.Tensor] = []
    rollout_velocity_bands: dict[int, list[torch.Tensor]] = {}
    teacher_velocity_bands: dict[int, list[torch.Tensor]] = {}
    secant_gain_bands: dict[int, list[torch.Tensor]] = {}

    processed = 0
    for start in range(0, cfg.count, cfg.batch_size):
        end = min(start + cfg.batch_size, cfg.count)
        clean = clean_all[start:end].to(device, non_blocking=True)
        noise = noise_all[start:end].to(device, non_blocking=True)
        labels = labels_all[start:end].to(device, non_blocking=True)
        batch_count = len(clean)
        clean_reference_summaries.append(latent_summary(clean, analyzer, channel_projection).cpu())
        reference_bands.append(latent_band_energy(clean, analyzer).cpu())

        perceptual_local = max(0, min(end, cfg.perceptual_count) - start)
        if perceptual_local > 0 and inception is not None:
            reference_features = decode_features_batched(
                rae,
                inception,
                clean[:perceptual_local],
                cfg.perceptual_batch_size,
            )
        else:
            reference_features = None

        teacher_states: dict[int, torch.Tensor] = {}
        teacher_velocities: dict[int, torch.Tensor] = {}
        teacher_clean: dict[int, torch.Tensor] = {}
        for time_index in selected_indices:
            scalar_time = times[time_index]
            expanded_time = scalar_time.reshape(1, 1, 1, 1)
            state = (1.0 - expanded_time) * clean + expanded_time * noise
            batch_time = torch.full(
                (batch_count,), float(scalar_time), device=device, dtype=clean.dtype
            )
            velocity = model(state, batch_time, y=labels)
            estimate = clean_from_velocity(state, velocity, batch_time)
            teacher_states[time_index] = state
            teacher_velocities[time_index] = velocity
            teacher_clean[time_index] = estimate

            error = estimate - clean
            teacher_metrics = {
                "latent_mse": error.square().flatten(1).mean(dim=1),
            }
            calibration = band_prediction_calibration(
                velocity, noise - clean, analyzer
            )
            for metric, values in calibration.items():
                teacher_band_values.setdefault((time_index, metric), []).append(
                    values.detach().float().cpu()
                )
            band_error = analyzer.band_mse(error)
            for band in range(analyzer.band_count):
                teacher_metrics[f"clean_band_mse_{band}"] = band_error[:, band]
            if reference_features is not None:
                estimate_features = decode_features_batched(
                    rae,
                    inception,
                    estimate[:perceptual_local],
                    cfg.perceptual_batch_size,
                )
                teacher_metrics.update(compare_decoder_features(estimate_features, reference_features))
            _append_values(teacher_values, time_index, teacher_metrics)

            _append_tensor(teacher_state_summaries, time_index, latent_summary(state, analyzer, channel_projection))
            _append_tensor(teacher_clean_summaries, time_index, latent_summary(estimate, analyzer, channel_projection))
            _append_tensor(teacher_state_bands, time_index, latent_band_energy(state, analyzer))
            _append_tensor(teacher_clean_bands, time_index, latent_band_energy(estimate, analyzer))
            _append_tensor(teacher_velocity_bands, time_index, latent_band_energy(velocity, analyzer))

        state = noise
        previous_state: torch.Tensor | None = None
        previous_velocity: torch.Tensor | None = None
        previous_clean_estimate: torch.Tensor | None = None
        previous_time: torch.Tensor | None = None
        rollout_clean_for_perceptual: dict[int, torch.Tensor] = {}
        for time_index, scalar_time in enumerate(times):
            batch_time = torch.full(
                (batch_count,), float(scalar_time), device=device, dtype=clean.dtype
            )
            velocity = model(state, batch_time, y=labels)
            estimate = clean_from_velocity(state, velocity, batch_time)
            if previous_state is not None and previous_velocity is not None and previous_time is not None:
                state_scale = state.square().flatten(1).mean(dim=1).sqrt().clamp_min(1e-12)
                velocity_scale = previous_velocity.square().flatten(1).mean(dim=1).sqrt().clamp_min(1e-12)
                clean_scale = estimate.square().flatten(1).mean(dim=1).sqrt().clamp_min(1e-12)
                step_size = (previous_time - scalar_time).abs().expand(batch_count)
                step_metrics = {
                    "euler_update_relative_rms": (
                        (state - previous_state).square().flatten(1).mean(dim=1).sqrt()
                        / state_scale
                    ),
                    "velocity_relative_change": (
                        (velocity - previous_velocity).square().flatten(1).mean(dim=1).sqrt()
                        / velocity_scale
                    ),
                    "velocity_relative_change_per_time": (
                        (velocity - previous_velocity).square().flatten(1).mean(dim=1).sqrt()
                        / velocity_scale
                        / step_size.clamp_min(1e-12)
                    ),
                    "clean_estimate_relative_change": (
                        (estimate - previous_clean_estimate).square().flatten(1).mean(dim=1).sqrt()
                        / clean_scale
                    ),
                    "step_size": step_size,
                }
                _append_values(step_values, time_index, step_metrics)
            if time_index in selected_indices:
                _append_tensor(state_summaries, time_index, latent_summary(state, analyzer, channel_projection))
                _append_tensor(clean_summaries, time_index, latent_summary(estimate, analyzer, channel_projection))
                _append_tensor(state_bands, time_index, latent_band_energy(state, analyzer))
                _append_tensor(clean_bands, time_index, latent_band_energy(estimate, analyzer))
                _append_tensor(rollout_velocity_bands, time_index, latent_band_energy(velocity, analyzer))
                state_difference = state - teacher_states[time_index]
                velocity_difference = velocity - teacher_velocities[time_index]
                state_rms = state_difference.square().flatten(1).mean(dim=1).sqrt()
                velocity_rms = velocity_difference.square().flatten(1).mean(dim=1).sqrt()
                _append_values(
                    rollout_pair_values,
                    time_index,
                    {
                        "coupled_state_rms": state_rms,
                        "coupled_velocity_rms": velocity_rms,
                        "velocity_state_secant_gain": velocity_rms / state_rms.clamp_min(1e-12),
                    },
                )
                state_difference_band = latent_band_energy(state_difference, analyzer)
                velocity_difference_band = latent_band_energy(velocity_difference, analyzer)
                _append_tensor(
                    secant_gain_bands,
                    time_index,
                    (velocity_difference_band / state_difference_band.clamp_min(1e-12)).sqrt(),
                )
                if perceptual_local > 0:
                    rollout_clean_for_perceptual[time_index] = estimate[:perceptual_local]
            previous_state = state
            previous_velocity = velocity
            previous_clean_estimate = estimate
            previous_time = scalar_time
            if time_index + 1 < len(times):
                state = state + (times[time_index + 1] - scalar_time) * velocity

        if perceptual_local > 0 and inception is not None:
            endpoint_features = decode_features_batched(
                rae,
                inception,
                state[:perceptual_local],
                cfg.perceptual_batch_size,
            )
            for time_index, estimate in rollout_clean_for_perceptual.items():
                estimate_features = decode_features_batched(
                    rae,
                    inception,
                    estimate,
                    cfg.perceptual_batch_size,
                )
                _append_values(
                    rollout_perceptual_values,
                    time_index,
                    {
                        f"predicted_clean_to_endpoint_{key}": value
                        for key, value in compare_decoder_features(
                            estimate_features, endpoint_features
                        ).items()
                    },
                )

        processed += batch_count
        print(f"{branch.name}: {processed}/{cfg.count}", flush=True)

    identity = {
        "branch": branch.name,
        "seed": int(manifest["global_seed"]),
        "treatment": "baseline" if float(manifest["gamma"]) == 0.0 else "partial",
        "gamma": float(manifest["gamma"]),
    }
    teacher_rows: list[dict[str, object]] = []
    for (time_index, metric), chunks in sorted(teacher_values.items()):
        values = torch.cat(chunks)
        teacher_rows.append(
            {
                **identity,
                "time_index": int(time_index),
                "time": float(times[time_index]),
                "metric": metric,
                "value": float(values.mean()),
                "sample_count": int(len(values)),
            }
        )

    teacher_band_rows: list[dict[str, object]] = []
    for (time_index, metric), chunks in sorted(teacher_band_values.items()):
        values = torch.cat(chunks)
        for band in range(values.shape[1]):
            teacher_band_rows.append(
                {
                    **identity,
                    "time_index": int(time_index),
                    "time": float(times[time_index]),
                    "metric": metric,
                    "band": int(band),
                    "value": float(values[:, band].mean()),
                    "sample_count": int(len(values)),
                }
            )

    reference_summary = torch.cat(clean_reference_summaries)
    reference_band = torch.cat(reference_bands).mean(dim=0)
    rollout_rows: list[dict[str, object]] = []
    band_rows: list[dict[str, object]] = []
    for time_index in selected_indices:
        state_summary = torch.cat(state_summaries[time_index])
        teacher_state_summary = torch.cat(teacher_state_summaries[time_index])
        clean_summary = torch.cat(clean_summaries[time_index])
        teacher_estimate_summary = torch.cat(teacher_clean_summaries[time_index])
        state_swd, state_mean_gap = _distribution_metrics(
            teacher_state_summary, state_summary, swd_directions
        )
        clean_swd, clean_mean_gap = _distribution_metrics(
            reference_summary, clean_summary, swd_directions
        )
        teacher_clean_swd, teacher_clean_mean_gap = _distribution_metrics(
            reference_summary, teacher_estimate_summary, swd_directions
        )

        state_energy = torch.cat(state_bands[time_index]).mean(dim=0)
        teacher_state_energy = torch.cat(teacher_state_bands[time_index]).mean(dim=0)
        clean_energy = torch.cat(clean_bands[time_index]).mean(dim=0)
        teacher_clean_energy = torch.cat(teacher_clean_bands[time_index]).mean(dim=0)
        rollout_velocity_energy = torch.cat(rollout_velocity_bands[time_index]).mean(dim=0)
        teacher_velocity_energy = torch.cat(teacher_velocity_bands[time_index]).mean(dim=0)
        log_ratios = {
            "state_energy_log_ratio": (state_energy / teacher_state_energy.clamp_min(1e-12)).log(),
            "clean_energy_log_ratio": (clean_energy / reference_band.clamp_min(1e-12)).log(),
            "teacher_clean_energy_log_ratio": (
                teacher_clean_energy / reference_band.clamp_min(1e-12)
            ).log(),
            "velocity_energy_log_ratio": (
                rollout_velocity_energy / teacher_velocity_energy.clamp_min(1e-12)
            ).log(),
        }
        summary_values = {
            "state_summary_swd": state_swd,
            "state_summary_mean_gap": state_mean_gap,
            "rollout_clean_summary_swd": clean_swd,
            "rollout_clean_summary_mean_gap": clean_mean_gap,
            "teacher_clean_summary_swd": teacher_clean_swd,
            "teacher_clean_summary_mean_gap": teacher_clean_mean_gap,
            **{
                f"{name}_absolute_mean": float(value.abs().mean())
                for name, value in log_ratios.items()
            },
        }
        for metric, value in summary_values.items():
            rollout_rows.append(
                {
                    **identity,
                    "time_index": int(time_index),
                    "time": float(times[time_index]),
                    "metric": metric,
                    "value": float(value),
                    "sample_count": int(cfg.count),
                }
            )
        for (stored_index, metric), chunks in rollout_pair_values.items():
            if stored_index != time_index:
                continue
            values = torch.cat(chunks)
            rollout_rows.append(
                {
                    **identity,
                    "time_index": int(time_index),
                    "time": float(times[time_index]),
                    "metric": metric,
                    "value": float(values.mean()),
                    "sample_count": int(len(values)),
                }
            )
        for (stored_index, metric), chunks in rollout_perceptual_values.items():
            if stored_index != time_index:
                continue
            values = torch.cat(chunks)
            rollout_rows.append(
                {
                    **identity,
                    "time_index": int(time_index),
                    "time": float(times[time_index]),
                    "metric": metric,
                    "value": float(values.mean()),
                    "sample_count": int(len(values)),
                }
            )
        for metric, values in log_ratios.items():
            for band, value in enumerate(values):
                band_rows.append(
                    {
                        **identity,
                        "time_index": int(time_index),
                        "time": float(times[time_index]),
                        "metric": metric,
                        "band": int(band),
                        "value": float(value),
                        "sample_count": int(cfg.count),
                    }
                )
        secant_values = torch.cat(secant_gain_bands[time_index]).mean(dim=0)
        for band, value in enumerate(secant_values):
            band_rows.append(
                {
                    **identity,
                    "time_index": int(time_index),
                    "time": float(times[time_index]),
                    "metric": "velocity_state_secant_gain",
                    "band": int(band),
                    "value": float(value),
                    "sample_count": int(cfg.count),
                }
            )

    step_rows: list[dict[str, object]] = []
    for (time_index, metric), chunks in sorted(step_values.items()):
        values = torch.cat(chunks)
        step_rows.append(
            {
                **identity,
                "time_index": int(time_index),
                "time": float(times[time_index]),
                "metric": metric,
                "value": float(values.mean()),
                "sample_count": int(len(values)),
            }
        )

    metadata: dict[str, object] = {
        **identity,
        "branch_path": str(branch),
        "checkpoint": str(branch / "generation" / "ema_step-0010000.pt"),
        "latent_cache": str(audit_config.cache_path),
        "dataset": "ImageNet-1K cached validation latents",
        "validation_indices": [int(index) for index in payload["validation_indices"][: cfg.count]],
        "count": int(cfg.count),
        "perceptual_count": int(min(cfg.perceptual_count, cfg.count)),
        "evaluation_seed": int(cfg.evaluation_seed),
        "precision": "fp32",
        "tf32": False,
        "sampler": "official shifted 50-step explicit Euler",
        "time_shift": float(time_shift),
        "times": [float(value) for value in times.cpu()],
        "selected_time_indices": selected_indices,
        "selected_times": [float(times[index]) for index in selected_indices],
        "teacher_pairing": "paired z0_hat=z_t-t*v_hat on the known linear interpolation",
        "rollout_comparison": "distributional only; rollout samples are not paired with validation z0",
        "coupled_secant_caveat": "same-noise validation coupling is used only as a vector-field sensitivity probe, not as sample correctness",
        "inception_proxy": "torchvision Inception-v3 pre-logit cosine distance; not the FID network",
        "endpoint_consistency_caveat": "z_t-t*v_hat is a local linear-path destination estimate, not guaranteed to equal the curved ODE endpoint",
    }
    return (
        pd.DataFrame(teacher_rows),
        pd.DataFrame(teacher_band_rows),
        pd.DataFrame(rollout_rows),
        pd.DataFrame(band_rows),
        pd.DataFrame(step_rows),
        metadata,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--perceptual-count", type=int, default=12)
    parser.add_argument("--perceptual-batch-size", type=int, default=2)
    parser.add_argument("--evaluation-seed", type=int, default=DEFAULT_EVALUATION_SEED)
    parser.add_argument("--num-steps", type=int, default=50)
    args = parser.parse_args()

    cfg = GapStudyConfig(
        branch=args.branch,
        device=args.device,
        count=args.count,
        batch_size=args.batch_size,
        perceptual_count=args.perceptual_count,
        perceptual_batch_size=args.perceptual_batch_size,
        evaluation_seed=args.evaluation_seed,
        num_steps=args.num_steps,
    )
    teacher, teacher_bands, rollout, bands, steps, metadata = run_gap_study(cfg)
    output = args.branch.expanduser().resolve() / "gap_study"
    output.mkdir(parents=True, exist_ok=True)
    teacher.to_csv(output / "teacher_metrics.csv", index=False)
    teacher_bands.to_csv(output / "teacher_bands.csv", index=False)
    rollout.to_csv(output / "rollout_metrics.csv", index=False)
    bands.to_csv(output / "rollout_bands.csv", index=False)
    steps.to_csv(output / "step_metrics.csv", index=False)
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(teacher.to_string(index=False), flush=True)
    print(rollout.to_string(index=False), flush=True)
    print(output, flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "GapStudyConfig",
    "band_prediction_calibration",
    "clean_from_velocity",
    "euler_rollout",
    "fixed_gaussian_matrix",
    "latent_summary",
    "official_time_grid",
    "run_gap_study",
    "select_time_indices",
    "sliced_wasserstein",
]
