"""Core metrics for the RAE decoder noise-coordinate geometry audit."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def sample_rms(value: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    """Return one root-mean-square value per sample."""

    if value.ndim < 2:
        raise ValueError("value must include batch and feature dimensions")
    result = value.float().square().flatten(1).mean(dim=1).sqrt()
    return result.clamp_min(float(eps)) if eps > 0 else result


def _expand_scale(scale: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return scale.to(value).reshape(-1, *([1] * (value.ndim - 1)))


@dataclass(frozen=True)
class MatchedPerturbations:
    """Perturbations expressed in both normalized and raw RAE coordinates."""

    raw_sphere_normalized: torch.Tensor
    stage2_sphere_raw_matched_normalized: torch.Tensor
    stage2_sphere_normalized_matched: torch.Tensor
    raw_sphere_raw: torch.Tensor
    stage2_sphere_raw_matched_raw: torch.Tensor
    stage2_sphere_normalized_matched_raw: torch.Tensor


def matched_noise_geometry(
    noise: torch.Tensor,
    latent_std: torch.Tensor,
    severity: torch.Tensor,
    eps: float = 1e-12,
) -> MatchedPerturbations:
    """Construct raw- and Stage-2-spherical perturbations with paired controls.

    ``raw_sphere`` is isotropic in the raw encoder coordinates used to train the
    decoder. ``stage2_sphere`` is isotropic in the normalized coordinates used
    by flow matching. The first Stage-2 control matches raw-space RMS, while the
    second matches normalized-space RMS to the raw-sphere perturbation.
    """

    if noise.ndim != 4:
        raise ValueError(f"noise must have shape [B,C,H,W], got {tuple(noise.shape)}")
    std = latent_std.to(noise).clamp_min(float(eps))
    if std.ndim == 3:
        std = std.unsqueeze(0)
    if std.shape not in {(1, *noise.shape[1:]), noise.shape}:
        raise ValueError("latent_std must broadcast over noise")
    if severity.ndim != 1 or len(severity) != len(noise):
        raise ValueError("severity must have shape [B]")

    target_raw_rms = _expand_scale(severity.float(), noise)
    raw_direction = noise.float() / _expand_scale(sample_rms(noise, eps), noise)
    raw_delta = target_raw_rms * raw_direction

    stage2_raw_direction = std * noise.float()
    stage2_raw_matched = target_raw_rms * stage2_raw_direction / _expand_scale(
        sample_rms(stage2_raw_direction, eps), noise
    )

    raw_delta_normalized = raw_delta / std
    stage2_raw_matched_normalized = stage2_raw_matched / std

    target_normalized_rms = _expand_scale(sample_rms(raw_delta_normalized), noise)
    stage2_normalized_direction = noise.float() / _expand_scale(
        sample_rms(noise, eps), noise
    )
    stage2_normalized_matched_normalized = (
        target_normalized_rms * stage2_normalized_direction
    )
    stage2_normalized_matched_raw = stage2_normalized_matched_normalized * std

    return MatchedPerturbations(
        raw_sphere_normalized=raw_delta_normalized,
        stage2_sphere_raw_matched_normalized=stage2_raw_matched_normalized,
        stage2_sphere_normalized_matched=stage2_normalized_matched_normalized,
        raw_sphere_raw=raw_delta,
        stage2_sphere_raw_matched_raw=stage2_raw_matched,
        stage2_sphere_normalized_matched_raw=stage2_normalized_matched_raw,
    )


def hidden_deviation_profile(
    candidate: tuple[torch.Tensor, ...],
    reference: tuple[torch.Tensor, ...],
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-layer hidden deviation RMS and consecutive amplification."""

    if not candidate or len(candidate) != len(reference):
        raise ValueError("candidate and reference hidden states must align")
    deviations = []
    for candidate_layer, reference_layer in zip(candidate, reference):
        if candidate_layer.shape != reference_layer.shape:
            raise ValueError("hidden-state shapes must match")
        deviations.append(sample_rms(candidate_layer - reference_layer))
    profile = torch.stack(deviations, dim=1)
    gain = profile[:, 1:] / profile[:, :-1].clamp_min(float(eps))
    return profile, gain


def relative_cycle_error(cycle: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
    if cycle.shape != latent.shape:
        raise ValueError("cycle and latent must have equal shape")
    return sample_rms(cycle - latent) / sample_rms(latent, 1e-12)


__all__ = [
    "MatchedPerturbations",
    "hidden_deviation_profile",
    "matched_noise_geometry",
    "relative_cycle_error",
    "sample_rms",
]
