"""Core operators for the ImageNet-100 multiscale-guidance study.

The Fourier projectors operate independently on every latent channel and form
an exact orthogonal partition of the complete 2-D FFT grid.  Frequencies are
therefore latent spatial frequencies, not decoded-image texture frequencies.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch


BAND_NAMES = ("low", "mid", "high")
TIME_NAMES = ("early", "mid", "late")
DEFAULT_DEPTHS = (4, 6, 8, 10, 12)
DEFAULT_LOW_CUTOFF = 0.125
DEFAULT_HIGH_CUTOFF = 0.25
DEFAULT_TIME_BOUNDARIES = (1.0 / 3.0, 2.0 / 3.0)


def observation_time_grid(
    time_min: float,
    time_max: float,
    time_points: int,
    *,
    anchors: Sequence[float] = (),
) -> tuple[float, ...]:
    """Build a strictly increasing grid in the sampler's float32 precision."""

    if not 0.0 < time_min < time_max <= 1.0:
        raise ValueError("time bounds must satisfy 0 < min < max <= 1")
    if time_points < 2:
        raise ValueError("at least two observation times are required")
    if any(not 0.0 < float(value) <= 1.0 for value in anchors):
        raise ValueError("anchor times must lie in (0,1]")
    dense = torch.linspace(
        float(time_min),
        float(time_max),
        int(time_points),
        dtype=torch.float64,
    )
    candidates = torch.cat(
        [dense, torch.as_tensor(tuple(anchors), dtype=torch.float64)]
    ).to(torch.float32)
    unique = torch.unique(candidates, sorted=True)
    if not torch.all(unique[1:] > unique[:-1]):
        raise AssertionError("observation grid is not strictly increasing")
    return tuple(float(value) for value in unique.tolist())


def _validate_field(field: torch.Tensor) -> None:
    if field.ndim != 4:
        raise ValueError("latent field must have shape [B,C,H,W]")
    if not field.is_floating_point():
        raise TypeError("latent field must use a floating dtype")


def radial_rfft_frequencies(
    height: int,
    width: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return radial frequencies for ``rfft2`` in cycles per latent pixel."""

    if min(height, width) < 2:
        raise ValueError("spatial dimensions must be at least two")
    fy = torch.fft.fftfreq(height, device=device, dtype=dtype)
    fx = torch.fft.rfftfreq(width, device=device, dtype=dtype)
    return torch.sqrt(fy[:, None].square() + fx[None, :].square())


def frequency_band_masks(
    height: int,
    width: int,
    *,
    device: torch.device | str,
    low_cutoff: float = DEFAULT_LOW_CUTOFF,
    high_cutoff: float = DEFAULT_HIGH_CUTOFF,
) -> dict[str, torch.Tensor]:
    """Build a disjoint and exhaustive latent-frequency partition.

    The high band intentionally includes FFT corners above 0.5 cycles/pixel.
    Omitting those corners would make ``P_L + P_M + P_H`` fail to reconstruct
    the input field.
    """

    if not 0.0 < low_cutoff < high_cutoff < 0.5:
        raise ValueError("frequency cutoffs must satisfy 0 < low < high < 0.5")
    radius = radial_rfft_frequencies(height, width, device=device)
    return {
        "low": radius < float(low_cutoff),
        "mid": (radius >= float(low_cutoff)) & (radius < float(high_cutoff)),
        "high": radius >= float(high_cutoff),
    }


def split_frequency_bands(
    field: torch.Tensor,
    *,
    low_cutoff: float = DEFAULT_LOW_CUTOFF,
    high_cutoff: float = DEFAULT_HIGH_CUTOFF,
) -> dict[str, torch.Tensor]:
    """Project a real latent field onto three exact Fourier subspaces."""

    _validate_field(field)
    height, width = field.shape[-2:]
    masks = frequency_band_masks(
        height,
        width,
        device=field.device,
        low_cutoff=low_cutoff,
        high_cutoff=high_cutoff,
    )
    spectrum = torch.fft.rfft2(field.float(), norm="ortho")
    return {
        name: torch.fft.irfft2(
            spectrum * mask[None, None],
            s=(height, width),
            norm="ortho",
        ).to(field.dtype)
        for name, mask in masks.items()
    }


def project_frequency_band(
    field: torch.Tensor,
    band: str,
    *,
    low_cutoff: float = DEFAULT_LOW_CUTOFF,
    high_cutoff: float = DEFAULT_HIGH_CUTOFF,
) -> torch.Tensor:
    if band not in BAND_NAMES:
        raise ValueError(f"unsupported frequency band: {band!r}")
    return split_frequency_bands(
        field,
        low_cutoff=low_cutoff,
        high_cutoff=high_cutoff,
    )[band]


def per_sample_mean_square(field: torch.Tensor) -> torch.Tensor:
    _validate_field(field)
    return field.float().square().mean(dim=(1, 2, 3))


def per_sample_rms(field: torch.Tensor) -> torch.Tensor:
    return per_sample_mean_square(field).sqrt()


def frequency_statistics(
    field: torch.Tensor,
    *,
    low_cutoff: float = DEFAULT_LOW_CUTOFF,
    high_cutoff: float = DEFAULT_HIGH_CUTOFF,
) -> dict[str, torch.Tensor]:
    """Return per-sample band fractions, centroid, DC fraction, and RMS."""

    _validate_field(field)
    height, width = field.shape[-2:]
    spectrum = torch.fft.rfft2(field.float(), norm="ortho")
    power = spectrum.abs().square()

    # Interior rFFT columns represent conjugate pairs and need weight two for
    # Parseval-consistent energy accounting. DC and Nyquist columns weight one.
    weights = torch.full(
        (spectrum.shape[-1],),
        2.0,
        device=field.device,
        dtype=power.dtype,
    )
    weights[0] = 1.0
    if width % 2 == 0:
        weights[-1] = 1.0
    weighted_power = power * weights[None, None, None, :]
    total = weighted_power.sum(dim=(1, 2, 3)).clamp_min(
        torch.finfo(weighted_power.dtype).tiny
    )
    masks = frequency_band_masks(
        height,
        width,
        device=field.device,
        low_cutoff=low_cutoff,
        high_cutoff=high_cutoff,
    )
    radius = radial_rfft_frequencies(
        height,
        width,
        device=field.device,
        dtype=power.dtype,
    )
    result: dict[str, torch.Tensor] = {
        "rms": per_sample_rms(field),
        "centroid": (
            weighted_power * radius[None, None]
        ).sum(dim=(1, 2, 3))
        / total,
        "dc_fraction": weighted_power[:, :, 0, 0].sum(dim=1) / total,
    }
    for name, mask in masks.items():
        result[f"{name}_fraction"] = (
            weighted_power * mask[None, None]
        ).sum(dim=(1, 2, 3)) / total
    return result


def _smoothstep(value: torch.Tensor) -> torch.Tensor:
    unit = value.clamp(0.0, 1.0)
    return unit.square() * (3.0 - 2.0 * unit)


def time_partition_weights(
    time_value: torch.Tensor,
    *,
    boundaries: tuple[float, float] = DEFAULT_TIME_BOUNDARIES,
    transition_width: float = 0.04,
) -> dict[str, torch.Tensor]:
    """Return a smooth early/mid/late partition of unity."""

    if time_value.ndim > 1:
        raise ValueError("time_value must be scalar or one-dimensional")
    first, second = (float(value) for value in boundaries)
    if not 0.0 < first < second < 1.0:
        raise ValueError("time boundaries must lie in increasing order in (0,1)")
    if transition_width <= 0.0 or transition_width >= second - first:
        raise ValueError("invalid transition width")
    half = 0.5 * float(transition_width)
    leave_early = _smoothstep((time_value.float() - (first - half)) / transition_width)
    enter_late = _smoothstep((time_value.float() - (second - half)) / transition_width)
    early = 1.0 - leave_early
    late = enter_late
    mid = (leave_early - enter_late).clamp_min(0.0)
    return {"early": early, "mid": mid, "late": late}


def broadcast_sample_weight(weight: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
    if weight.ndim == 0:
        return weight
    if weight.shape != (len(field),):
        raise ValueError("sample weight must be scalar or have shape [B]")
    return weight.reshape(-1, 1, 1, 1)


def band_time_component(
    field: torch.Tensor,
    time_value: torch.Tensor,
    *,
    band: str,
    interval: str,
    scale: float = 1.0,
    transition_width: float = 0.04,
) -> torch.Tensor:
    if interval not in TIME_NAMES:
        raise ValueError(f"unsupported time interval: {interval!r}")
    projected = project_frequency_band(field, band)
    weight = time_partition_weights(
        time_value,
        transition_width=transition_width,
    )[interval]
    return float(scale) * projected * broadcast_sample_weight(weight, projected)


def ordered_band_component(
    field: torch.Tensor,
    time_value: torch.Tensor,
    *,
    order: str,
    cell_scales: Mapping[str, float] | None = None,
    transition_width: float = 0.04,
) -> torch.Tensor:
    """Build coarse-to-fine or reversed fine-to-coarse guidance."""

    schedules = {
        "coarse_to_fine": {"early": "low", "mid": "mid", "late": "high"},
        "fine_to_coarse": {"early": "high", "mid": "mid", "late": "low"},
    }
    if order not in schedules:
        raise ValueError(f"unsupported ordering: {order!r}")
    result = torch.zeros_like(field)
    for interval, band in schedules[order].items():
        key = f"{interval}_{band}"
        scale = 1.0 if cell_scales is None else float(cell_scales[key])
        result = result + band_time_component(
            field,
            time_value,
            band=band,
            interval=interval,
            scale=scale,
            transition_width=transition_width,
        )
    return result


def rms_match_direction(
    direction: torch.Tensor,
    reference: torch.Tensor,
    *,
    max_scale: float = 8.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match per-sample RMS while exposing the applied, capped scale."""

    _validate_field(direction)
    _validate_field(reference)
    if direction.shape != reference.shape:
        raise ValueError("direction and reference must have identical shapes")
    if max_scale <= 0.0:
        raise ValueError("max_scale must be positive")
    tiny = torch.finfo(torch.float32).tiny
    scale = per_sample_rms(reference) / per_sample_rms(direction).clamp_min(tiny)
    scale = scale.clamp(max=float(max_scale))
    return direction * scale[:, None, None, None], scale


def interpolate_time_table(
    time_value: torch.Tensor,
    table_times: Sequence[float],
    table_values: Sequence[float],
) -> torch.Tensor:
    """Linearly interpolate a scalar calibration table on-device."""

    if len(table_times) != len(table_values) or len(table_times) < 2:
        raise ValueError("calibration tables need equal lengths of at least two")
    if any(right <= left for left, right in zip(table_times, table_times[1:])):
        raise ValueError("table times must be strictly increasing")
    times = torch.as_tensor(table_times, device=time_value.device, dtype=torch.float32)
    values = torch.as_tensor(table_values, device=time_value.device, dtype=torch.float32)
    query = time_value.float().clamp(float(times[0]), float(times[-1]))
    upper = torch.searchsorted(times, query, right=True).clamp(1, len(times) - 1)
    lower = upper - 1
    left_t = times[lower]
    right_t = times[upper]
    fraction = (query - left_t) / (right_t - left_t)
    return values[lower] + fraction * (values[upper] - values[lower])


def select_per_sample(
    candidates: Mapping[int, torch.Tensor],
    selected_depth: torch.Tensor,
) -> torch.Tensor:
    """Select one candidate latent field for each sample."""

    if selected_depth.shape != (len(next(iter(candidates.values()))),):
        raise ValueError("selected_depth must have shape [B]")
    depths = sorted(candidates)
    stacked = torch.stack([candidates[depth] for depth in depths], dim=1)
    lookup = torch.as_tensor(depths, device=selected_depth.device)
    matches = selected_depth[:, None] == lookup[None]
    if not matches.any(dim=1).all():
        raise ValueError("selected depth is not available")
    indices = matches.float().argmax(dim=1)
    gather_shape = (len(stacked), 1, *stacked.shape[2:])
    gather_index = indices.reshape(-1, 1, 1, 1, 1).expand(gather_shape)
    return stacked.gather(1, gather_index).squeeze(1)


def weak_head_difference_field(
    strong: torch.Tensor,
    positive_weak: torch.Tensor,
    negative_weak: torch.Tensor,
    *,
    gamma: float,
) -> torch.Tensor:
    """Extrapolate the strong field along an ordered weak-head difference."""

    if not (strong.shape == positive_weak.shape == negative_weak.shape):
        raise ValueError("strong and weak-head fields must have identical shapes")
    if gamma == 0.0:
        return strong
    return strong + float(gamma) * (positive_weak - negative_weak)


def decompose_weak_head_difference(
    strong: torch.Tensor,
    positive_weak: torch.Tensor,
    negative_weak: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a weak-head difference relative to the full-to-weak gap.

    The ordered weak-head difference is ``positive_weak - negative_weak`` and
    the reference is ``strong - negative_weak``.  Projection is performed with
    one scalar per sample over all latent coordinates.
    """

    if not (strong.shape == positive_weak.shape == negative_weak.shape):
        raise ValueError("strong and weak-head fields must have identical shapes")
    if strong.ndim < 2:
        raise ValueError("fields must include batch and feature dimensions")
    difference = positive_weak - negative_weak
    reference = strong - negative_weak
    dims = tuple(range(1, strong.ndim))
    reference_energy = reference.double().square().sum(dim=dims)
    numerator = (difference.double() * reference.double()).sum(dim=dims)
    tiny = torch.finfo(torch.float64).tiny
    coefficient = torch.where(
        reference_energy > tiny,
        numerator / reference_energy.clamp_min(tiny),
        torch.zeros_like(numerator),
    )
    shape = (len(strong),) + (1,) * (strong.ndim - 1)
    parallel = coefficient.to(reference.dtype).reshape(shape) * reference
    orthogonal = difference - parallel
    return difference, parallel, orthogonal, coefficient


def route_depth_by_target_band(
    gaps: Mapping[int, torch.Tensor],
    time_value: torch.Tensor,
    *,
    reverse: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose the depth whose gap is most concentrated in the target scale.

    The target scale is fixed before looking at results: low early, mid in the
    middle, and high late.  ``reverse=True`` is the pre-registered anti-router.
    """

    weights = time_partition_weights(time_value)
    target_scores: list[torch.Tensor] = []
    depths = sorted(gaps)
    for depth in depths:
        stats = frequency_statistics(gaps[depth])
        score = sum(
            weights[interval] * stats[f"{band}_fraction"]
            for interval, band in (
                ("early", "low"),
                ("mid", "mid"),
                ("late", "high"),
            )
        )
        target_scores.append(score)
    score_matrix = torch.stack(target_scores, dim=1)
    indices = score_matrix.argmin(dim=1) if reverse else score_matrix.argmax(dim=1)
    depth_values = torch.as_tensor(depths, device=indices.device)
    selected_depth = depth_values[indices]
    return select_per_sample(gaps, selected_depth), selected_depth


def schedule_depth(
    time_value: torch.Tensor,
    *,
    order: str,
    depths: Sequence[int] = (4, 8, 10),
) -> torch.Tensor:
    # gamma_schedule_sweep_v4_generalized_schedule_depth
    selected = tuple(int(depth) for depth in depths)
    if len(selected) < 2 or any(depth < 1 for depth in selected):
        raise ValueError("at least two positive schedule depths are required")
    if len(set(selected)) != len(selected):
        raise ValueError("schedule depths must be unique")
    if order == "coarse_to_fine":
        pass
    elif order == "fine_to_coarse":
        selected = tuple(reversed(selected))
    else:
        raise ValueError(f"unsupported depth order: {order!r}")

    # Preserve the original implementation exactly for the historical 3-stage
    # schedule, including boundary/tie semantics.
    if len(selected) == 3:
        weights = time_partition_weights(time_value)
        matrix = torch.stack([weights[name] for name in TIME_NAMES], dim=1)
        indices = matrix.argmax(dim=1)
    else:
        # General N-stage schedule: equal time partitions. At an exact boundary
        # enter the next stage, matching the effective float32 behavior of the
        # historical 3-stage implementation.
        boundaries = torch.arange(1, len(selected), device=time_value.device, dtype=torch.float32)
        boundaries = boundaries / float(len(selected))
        indices = torch.bucketize(time_value.float(), boundaries, right=True)
    values = torch.as_tensor(selected, device=time_value.device)
    return values[indices]


def finite_number(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result
