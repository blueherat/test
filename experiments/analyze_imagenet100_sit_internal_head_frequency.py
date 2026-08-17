#!/usr/bin/env python3
"""Compare image frequencies from a frozen SiT full and intermediate v head.

The terminal analysis reuses paired full/internal rollout images.  The optional
state probe follows the full v800 trajectory, converts both velocity outputs at
the same state to clean predictions, and decodes those predictions with the
same SD VAE:

    x0_hat = z_t + (1 - t) * v_hat

This separation prevents rollout drift from being mistaken for a per-head
frequency bias.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    from experiments.imagenet100_sit_internal_v_head import (
        full_and_internal_velocity,
    )
    from experiments.sample_imagenet100_sit_fid import decode_latents_in_chunks
    from experiments.sample_imagenet100_sit_frozen_internal_v_head_fid import (
        load_frozen_internal_model,
    )
    from experiments.train_imagenet100_sit_flow import (
        DEFAULT_OFFICIAL_SIT_REPO,
        LATENT_SHAPE,
        NUM_CLASSES,
        SD_VAE_SCALING_FACTOR,
        load_official_sit_module,
    )
except ModuleNotFoundError:
    from imagenet100_sit_internal_v_head import full_and_internal_velocity
    from sample_imagenet100_sit_fid import decode_latents_in_chunks
    from sample_imagenet100_sit_frozen_internal_v_head_fid import (
        load_frozen_internal_model,
    )
    from train_imagenet100_sit_flow import (
        DEFAULT_OFFICIAL_SIT_REPO,
        LATENT_SHAPE,
        NUM_CLASSES,
        SD_VAE_SCALING_FACTOR,
        load_official_sit_module,
    )


DEFAULT_CHECKPOINT = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/"
    "sit-s-2_v800-ema_frozen-internal-v-depth8_seed0/"
    "checkpoints/step_00050000.pt"
)
DEFAULT_SAMPLE_ROOT = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/"
    "fid1k_v800_frozen_internal_v_depth8_step50000_ema"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/"
    "frequency_v800_frozen_internal_v_depth8_step50000_v1"
)
METRIC_NAMES = (
    "low_fraction",
    "mid_fraction",
    "high_fraction",
    "spectral_centroid_cpp",
    "high_to_low_ratio",
    "rms_contrast",
    "gradient_rms",
    "laplacian_rms",
)


@dataclass
class SpectrumResult:
    frequencies: np.ndarray
    mean_psd: np.ndarray
    mean_energy_fraction: np.ndarray
    per_image: dict[str, np.ndarray]


def _luminance(images: np.ndarray) -> np.ndarray:
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("images must have shape [N,H,W,3]")
    values = images.astype(np.float32, copy=False)
    if images.dtype == np.uint8:
        values = values / 127.5 - 1.0
    return (
        0.2126 * values[..., 0]
        + 0.7152 * values[..., 1]
        + 0.0722 * values[..., 2]
    )


def _frequency_geometry(
    height: int,
    width: int,
    radial_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    if min(height, width, radial_bins) < 2:
        raise ValueError("image dimensions and radial bins must be at least two")
    fy = np.fft.fftshift(np.fft.fftfreq(height))
    fx = np.fft.fftshift(np.fft.fftfreq(width))
    radius = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    edges = np.linspace(0.0, 0.5, radial_bins + 1, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    valid = radius <= 0.5
    masks = [
        valid & (radius >= edges[index]) & (radius < edges[index + 1])
        for index in range(radial_bins)
    ]
    masks[-1] |= valid & np.isclose(radius, 0.5)
    return radius, valid, centers, masks


def analyze_image_frequencies(
    images: np.ndarray,
    *,
    radial_bins: int = 64,
    chunk_size: int = 16,
) -> SpectrumResult:
    """Measure DC-removed, Hann-windowed luminance power spectra."""

    luminance = _luminance(images)
    sample_count, height, width = luminance.shape
    if sample_count < 1 or chunk_size < 1:
        raise ValueError("sample count and chunk size must be positive")
    radius, valid, centers, radial_masks = _frequency_geometry(
        height,
        width,
        radial_bins,
    )
    low_mask = valid & (radius < 0.125)
    mid_mask = valid & (radius >= 0.125) & (radius < 0.25)
    high_mask = valid & (radius >= 0.25)
    window = np.outer(np.hanning(height), np.hanning(width)).astype(np.float32)
    window /= np.sqrt(np.mean(window**2))

    psd_sum = np.zeros(radial_bins, dtype=np.float64)
    energy_fraction_sum = np.zeros(radial_bins, dtype=np.float64)
    metrics = {name: [] for name in METRIC_NAMES}
    radial_counts = np.asarray([mask.sum() for mask in radial_masks], dtype=np.int64)
    if (radial_counts == 0).any():
        raise ValueError("radial binning produced an empty bin")

    for start in range(0, sample_count, chunk_size):
        values = luminance[start : start + chunk_size]
        centered = values - values.mean(axis=(1, 2), keepdims=True)
        windowed = centered * window[None]
        windowed -= windowed.mean(axis=(1, 2), keepdims=True)
        spectrum = np.fft.fftshift(
            np.fft.fft2(windowed, axes=(1, 2), norm="ortho"),
            axes=(1, 2),
        )
        power = np.square(np.abs(spectrum), dtype=np.float64)
        total = power[:, valid].sum(axis=1)
        total = np.maximum(total, np.finfo(np.float64).tiny)

        radial_energy = np.stack(
            [power[:, mask].sum(axis=1) for mask in radial_masks],
            axis=1,
        )
        psd_sum += radial_energy.sum(axis=0) / radial_counts
        energy_fraction_sum += (radial_energy / total[:, None]).sum(axis=0)

        low = power[:, low_mask].sum(axis=1) / total
        mid = power[:, mid_mask].sum(axis=1) / total
        high = power[:, high_mask].sum(axis=1) / total
        centroid = (power[:, valid] * radius[valid][None]).sum(axis=1) / total
        dx = np.diff(values, axis=2)
        dy = np.diff(values, axis=1)
        gradient_rms = np.sqrt(
            0.5
            * (
                np.mean(dx**2, axis=(1, 2))
                + np.mean(dy**2, axis=(1, 2))
            )
        )
        laplacian = (
            -4.0 * values[:, 1:-1, 1:-1]
            + values[:, :-2, 1:-1]
            + values[:, 2:, 1:-1]
            + values[:, 1:-1, :-2]
            + values[:, 1:-1, 2:]
        )
        metrics["low_fraction"].extend(low.tolist())
        metrics["mid_fraction"].extend(mid.tolist())
        metrics["high_fraction"].extend(high.tolist())
        metrics["spectral_centroid_cpp"].extend(centroid.tolist())
        metrics["high_to_low_ratio"].extend(
            (high / np.maximum(low, np.finfo(np.float64).tiny)).tolist()
        )
        metrics["rms_contrast"].extend(
            np.sqrt(np.mean(centered**2, axis=(1, 2))).tolist()
        )
        metrics["gradient_rms"].extend(gradient_rms.tolist())
        metrics["laplacian_rms"].extend(
            np.sqrt(np.mean(laplacian**2, axis=(1, 2))).tolist()
        )

    return SpectrumResult(
        frequencies=centers,
        mean_psd=psd_sum / sample_count,
        mean_energy_fraction=energy_fraction_sum / sample_count,
        per_image={
            name: np.asarray(values, dtype=np.float64)
            for name, values in metrics.items()
        },
    )


def bootstrap_mean_interval(
    values: np.ndarray,
    *,
    reps: int,
    seed: int,
) -> tuple[float, float]:
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("bootstrap input must contain at least two values")
    rng = np.random.default_rng(seed)
    means = np.empty(reps, dtype=np.float64)
    for start in range(0, reps, 256):
        count = min(256, reps - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means[start : start + count] = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def summarize_pair(
    strong: SpectrumResult,
    weak: SpectrumResult,
    *,
    reps: int,
    seed: int,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for metric_index, name in enumerate(METRIC_NAMES):
        strong_values = strong.per_image[name]
        weak_values = weak.per_image[name]
        if strong_values.shape != weak_values.shape:
            raise ValueError(f"unpaired metric arrays for {name}")
        delta = weak_values - strong_values
        low, high = bootstrap_mean_interval(
            delta,
            reps=reps,
            seed=seed + metric_index,
        )
        strong_mean = float(strong_values.mean())
        weak_mean = float(weak_values.mean())
        rows.append(
            {
                "metric": name,
                "strong_mean": strong_mean,
                "weak_mean": weak_mean,
                "weak_minus_strong": float(delta.mean()),
                "weak_over_strong": weak_mean / strong_mean
                if strong_mean != 0
                else math.nan,
                "paired_ci95_low": low,
                "paired_ci95_high": high,
            }
        )
    return rows


def summarize_single(result: SpectrumResult) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for name in METRIC_NAMES:
        values = result.per_image[name]
        rows.append(
            {
                "metric": name,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "median": float(np.median(values)),
                "q05": float(np.quantile(values, 0.05)),
                "q95": float(np.quantile(values, 0.95)),
            }
        )
    return rows


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_paired_terminal_images(
    sample_root: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    manifests = {
        name: _load_json(sample_root / name / "sampling_manifest.json")
        for name in ("full", "internal")
    }
    full_manifest = manifests["full"]
    weak_manifest = manifests["internal"]
    audit_keys = (
        "requested_samples",
        "global_seed",
        "rank_seeds",
        "rank_noise_sha256",
        "rank_label_sha256",
        "label_histogram",
        "sampler",
        "official_sit",
    )
    mismatches = [
        key
        for key in audit_keys
        if full_manifest.get(key) != weak_manifest.get(key)
    ]
    if mismatches:
        raise ValueError(f"terminal samples are not paired: {mismatches}")
    for key in (
        "head_checkpoint_sha256",
        "head_weights",
        "source_checkpoint_sha256",
        "source_step",
        "internal_depth",
    ):
        if full_manifest["model"].get(key) != weak_manifest["model"].get(key):
            raise ValueError(f"terminal model metadata differs for {key}")

    labels = {
        name: np.load(sample_root / name / "sample_labels_unguided_n1000.npy")
        for name in ("full", "internal")
    }
    if not np.array_equal(labels["full"], labels["internal"]):
        raise ValueError("terminal class labels differ")

    images: dict[str, np.ndarray] = {}
    for name in ("full", "internal"):
        path = sample_root / name / "samples_unguided_n1000.npz"
        with np.load(path) as payload:
            if len(payload.files) != 1:
                raise ValueError(f"unexpected sample arrays in {path}")
            images[name] = payload[payload.files[0]]
    if images["full"].shape != images["internal"].shape:
        raise ValueError("terminal image shapes differ")
    return images["full"], images["internal"], {
        "sample_count": int(len(images["full"])),
        "global_seed": int(full_manifest["global_seed"]),
        "rank_noise_sha256": full_manifest["rank_noise_sha256"],
        "rank_label_sha256": full_manifest["rank_label_sha256"],
        "model": full_manifest["model"],
    }


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_spectrum_csv(
    path: Path,
    strong: SpectrumResult,
    weak: SpectrumResult,
    difference: SpectrumResult,
) -> None:
    if not np.array_equal(strong.frequencies, weak.frequencies) or not np.array_equal(
        strong.frequencies,
        difference.frequencies,
    ):
        raise ValueError("frequency grids differ")
    rows = []
    tiny = np.finfo(np.float64).tiny
    for index, frequency in enumerate(strong.frequencies):
        rows.append(
            {
                "frequency_cycles_per_pixel": float(frequency),
                "strong_mean_psd": float(strong.mean_psd[index]),
                "weak_mean_psd": float(weak.mean_psd[index]),
                "difference_mean_psd": float(difference.mean_psd[index]),
                "weak_over_strong_db": float(
                    10.0
                    * np.log10(
                        max(weak.mean_psd[index], tiny)
                        / max(strong.mean_psd[index], tiny)
                    )
                ),
                "strong_energy_fraction": float(
                    strong.mean_energy_fraction[index]
                ),
                "weak_energy_fraction": float(weak.mean_energy_fraction[index]),
                "difference_energy_fraction": float(
                    difference.mean_energy_fraction[index]
                ),
            }
        )
    write_csv(path, rows)


def plot_terminal_spectrum(
    path: Path,
    strong: SpectrumResult,
    weak: SpectrumResult,
    difference: SpectrumResult,
    *,
    sample_count: int,
) -> None:
    colors = {"strong": "#2563eb", "weak": "#d97706", "difference": "#52525b"}
    tiny = np.finfo(np.float64).tiny
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.5), constrained_layout=True)
    for result, label, color, style in (
        (strong, "strong v800", colors["strong"], "-"),
        (weak, "weak depth-8 v", colors["weak"], "--"),
        (difference, "paired difference", colors["difference"], ":"),
    ):
        axes[0].plot(
            result.frequencies,
            10.0 * np.log10(np.maximum(result.mean_psd, tiny)),
            label=label,
            color=color,
            linestyle=style,
            linewidth=2.0,
        )
        axes[1].plot(
            result.frequencies,
            result.mean_energy_fraction,
            label=label,
            color=color,
            linestyle=style,
            linewidth=2.0,
        )
    ratio_db = 10.0 * np.log10(
        np.maximum(weak.mean_psd, tiny) / np.maximum(strong.mean_psd, tiny)
    )
    axes[2].plot(
        strong.frequencies,
        ratio_db,
        color=colors["weak"],
        linewidth=2.0,
    )
    axes[2].axhline(0.0, color="#18181b", linewidth=1.0)
    axes[2].fill_between(
        strong.frequencies,
        0.0,
        ratio_db,
        where=ratio_db >= 0,
        color=colors["weak"],
        alpha=0.16,
    )
    axes[2].fill_between(
        strong.frequencies,
        0.0,
        ratio_db,
        where=ratio_db < 0,
        color=colors["strong"],
        alpha=0.12,
    )
    axes[0].set_title("Mean radial power spectrum")
    axes[0].set_ylabel("Power (dB, DC removed)")
    axes[1].set_title("Mean energy fraction per radial bin")
    axes[1].set_ylabel("Fraction")
    axes[2].set_title("Weak / strong spectral ratio")
    axes[2].set_ylabel("Power ratio (dB)")
    for axis in axes:
        axis.set_xlabel("Spatial frequency (cycles / pixel)")
        axis.grid(alpha=0.22, linewidth=0.7)
        axis.set_xlim(0.0, 0.5)
    axes[0].legend(frameon=False)
    figure.suptitle(
        f"Independent rollout frequency comparison (paired n={sample_count})",
        fontsize=13,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_paired_difference_preview(
    path: Path,
    strong_images: np.ndarray,
    weak_images: np.ndarray,
    *,
    count: int = 6,
) -> None:
    count = min(count, len(strong_images), len(weak_images))
    strong = strong_images[:count].astype(np.float32) / 127.5 - 1.0
    weak = weak_images[:count].astype(np.float32) / 127.5 - 1.0
    difference = strong - weak
    luminance_difference = _luminance(difference)
    absolute_difference = np.mean(np.abs(difference), axis=-1)
    signed_limit = max(
        1e-6,
        float(np.quantile(np.abs(luminance_difference), 0.99)),
    )
    absolute_limit = max(1e-6, float(np.quantile(absolute_difference, 0.99)))
    figure, axes = plt.subplots(
        4,
        count,
        figsize=(2.35 * count, 8.9),
        squeeze=False,
        constrained_layout=True,
    )
    for column in range(count):
        axes[0, column].imshow((strong[column] + 1.0) / 2.0)
        axes[1, column].imshow((weak[column] + 1.0) / 2.0)
        axes[2, column].imshow(
            luminance_difference[column],
            cmap="RdBu_r",
            vmin=-signed_limit,
            vmax=signed_limit,
        )
        axes[3, column].imshow(
            absolute_difference[column],
            cmap="magma",
            vmin=0.0,
            vmax=absolute_limit,
        )
        for row in range(4):
            axes[row, column].axis("off")
    for axis, label in zip(
        axes[:, 0],
        ("strong", "weak", "strong - weak", "|strong - weak|"),
        strict=True,
    ):
        axis.set_ylabel(label, rotation=90)
    figure.suptitle(
        "Paired independent rollouts and their image-space difference",
        fontsize=13,
    )
    figure.savefig(path, dpi=160)
    plt.close(figure)


def terminal_analysis(args: argparse.Namespace) -> dict[str, object]:
    full_images, weak_images, audit = load_paired_terminal_images(args.sample_root)
    strong = analyze_image_frequencies(
        full_images,
        radial_bins=args.radial_bins,
        chunk_size=args.frequency_chunk_size,
    )
    weak = analyze_image_frequencies(
        weak_images,
        radial_bins=args.radial_bins,
        chunk_size=args.frequency_chunk_size,
    )
    difference = analyze_image_frequencies(
        full_images.astype(np.float32) / 127.5
        - weak_images.astype(np.float32) / 127.5,
        radial_bins=args.radial_bins,
        chunk_size=args.frequency_chunk_size,
    )
    rows = summarize_pair(
        strong,
        weak,
        reps=args.bootstrap_reps,
        seed=args.bootstrap_seed,
    )
    write_csv(args.output_dir / "terminal_metrics.csv", rows)
    difference_rows = summarize_single(difference)
    write_csv(
        args.output_dir / "terminal_difference_metrics.csv",
        difference_rows,
    )
    write_spectrum_csv(
        args.output_dir / "terminal_radial_spectrum.csv",
        strong,
        weak,
        difference,
    )
    plot_terminal_spectrum(
        args.output_dir / "terminal_frequency_comparison.png",
        strong,
        weak,
        difference,
        sample_count=int(audit["sample_count"]),
    )
    plot_paired_difference_preview(
        args.output_dir / "terminal_paired_difference_preview.png",
        full_images,
        weak_images,
    )
    return {
        "audit": audit,
        "metrics": rows,
        "difference_metrics": difference_rows,
        "files": {
            "metrics": str(args.output_dir / "terminal_metrics.csv"),
            "difference_metrics": str(
                args.output_dir / "terminal_difference_metrics.csv"
            ),
            "spectrum": str(args.output_dir / "terminal_radial_spectrum.csv"),
            "plot": str(args.output_dir / "terminal_frequency_comparison.png"),
            "difference_preview": str(
                args.output_dir / "terminal_paired_difference_preview.png"
            ),
        },
    }


def _decode_clean_predictions(
    vae: torch.nn.Module,
    latents: torch.Tensor,
    *,
    chunk_size: int,
) -> np.ndarray:
    decoded = decode_latents_in_chunks(
        vae,
        latents,
        scaling_factor=SD_VAE_SCALING_FACTOR,
        chunk_size=chunk_size,
    )
    return (
        decoded.float()
        .clamp(-1.0, 1.0)
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )


def _plot_state_probe(
    output_dir: Path,
    records: list[dict[str, object]],
    previews: list[tuple[float, np.ndarray, np.ndarray]],
) -> None:
    times = np.asarray([float(record["time"]) for record in records])
    metrics = ("high_fraction", "spectral_centroid_cpp", "gradient_rms")
    labels = (
        "High-frequency energy fraction",
        "Spectral centroid (cycles / pixel)",
        "Gradient RMS",
    )
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.3), constrained_layout=True)
    for axis, metric, label in zip(axes, metrics, labels, strict=True):
        axis.plot(
            times,
            [record[f"strong_{metric}"] for record in records],
            color="#2563eb",
            marker="o",
            label="strong v800",
        )
        axis.plot(
            times,
            [record[f"weak_{metric}"] for record in records],
            color="#d97706",
            marker="s",
            linestyle="--",
            label="weak depth-8 v",
        )
        axis.set_xlabel("Baseline trajectory time t")
        axis.set_ylabel(label)
        axis.grid(alpha=0.22, linewidth=0.7)
    axes[0].legend(frameon=False)
    figure.suptitle("Same-state decoded clean-prediction frequencies", fontsize=13)
    figure.savefig(output_dir / "same_state_frequency_metrics.png", dpi=180)
    plt.close(figure)

    ratio_rows = []
    frequencies = None
    for record in records:
        current = np.asarray(record["weak_over_strong_db"], dtype=np.float64)
        ratio_rows.append(current)
        frequencies = np.asarray(record["frequencies"], dtype=np.float64)
    assert frequencies is not None
    ratio = np.stack(ratio_rows)
    limit = max(1.0, float(np.nanpercentile(np.abs(ratio), 98)))
    figure, axis = plt.subplots(figsize=(9.2, 5.0), constrained_layout=True)
    image = axis.imshow(
        ratio,
        aspect="auto",
        origin="lower",
        extent=[frequencies[0], frequencies[-1], -0.5, len(times) - 0.5],
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
    )
    axis.set_title("Weak / strong power ratio at the same baseline state")
    axis.set_xlabel("Spatial frequency (cycles / pixel)")
    axis.set_ylabel("Baseline trajectory time t")
    axis.set_yticks(np.arange(len(times)), [f"{value:g}" for value in times])
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Power ratio (dB)")
    figure.savefig(output_dir / "same_state_spectral_ratio_heatmap.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.3), constrained_layout=True)
    for name, color, marker in (
        ("low_fraction", "#2563eb", "o"),
        ("mid_fraction", "#d97706", "s"),
        ("high_fraction", "#52525b", "^"),
    ):
        axes[0].plot(
            times,
            [record[f"difference_{name}"] for record in records],
            label=name.replace("_fraction", ""),
            color=color,
            marker=marker,
        )
    axes[0].set_ylabel("Difference energy fraction")
    axes[0].legend(frameon=False)
    axes[1].plot(
        times,
        [record["difference_spectral_centroid_cpp"] for record in records],
        color="#d97706",
        marker="s",
    )
    axes[1].set_ylabel("Difference spectral centroid (cycles / pixel)")
    axes[2].plot(
        times,
        [record["difference_rms_contrast"] for record in records],
        color="#2563eb",
        marker="o",
        label="RMS contrast",
    )
    axes[2].plot(
        times,
        [record["difference_gradient_rms"] for record in records],
        color="#d97706",
        marker="s",
        label="gradient RMS",
    )
    axes[2].set_ylabel("Difference magnitude")
    axes[2].legend(frameon=False)
    for axis in axes:
        axis.set_xlabel("Baseline trajectory time t")
        axis.grid(alpha=0.22, linewidth=0.7)
    figure.suptitle("Frequency composition of strong - weak", fontsize=13)
    figure.savefig(output_dir / "same_state_difference_metrics.png", dpi=180)
    plt.close(figure)

    difference_energy = np.stack(
        [
            np.asarray(record["difference_energy_fraction"], dtype=np.float64)
            for record in records
        ]
    )
    difference_db = 10.0 * np.log10(
        np.maximum(difference_energy, np.finfo(np.float64).tiny)
    )
    figure, axis = plt.subplots(figsize=(9.2, 5.0), constrained_layout=True)
    image = axis.imshow(
        difference_db,
        aspect="auto",
        origin="lower",
        extent=[frequencies[0], frequencies[-1], -0.5, len(times) - 0.5],
        cmap="magma",
        interpolation="nearest",
    )
    axis.set_title("Radial energy distribution of strong - weak")
    axis.set_xlabel("Spatial frequency (cycles / pixel)")
    axis.set_ylabel("Baseline trajectory time t")
    axis.set_yticks(np.arange(len(times)), [f"{value:g}" for value in times])
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Energy fraction per radial bin (dB)")
    figure.savefig(output_dir / "same_state_difference_spectrum_heatmap.png", dpi=180)
    plt.close(figure)

    columns = len(previews)
    preview_differences = [strong - weak for _, strong, weak in previews]
    luminance_differences = [_luminance(value[None])[0] for value in preview_differences]
    absolute_differences = [
        np.mean(np.abs(value), axis=-1) for value in preview_differences
    ]
    signed_limit = max(
        1e-6,
        float(np.quantile(np.abs(np.stack(luminance_differences)), 0.99)),
    )
    absolute_limit = max(
        1e-6,
        float(np.quantile(np.stack(absolute_differences), 0.99)),
    )
    figure, axes = plt.subplots(
        4,
        columns,
        figsize=(2.35 * columns, 9.0),
        squeeze=False,
        constrained_layout=True,
    )
    for column, (time_value, strong_image, weak_image) in enumerate(previews):
        axes[0, column].imshow((strong_image + 1.0) / 2.0)
        axes[1, column].imshow((weak_image + 1.0) / 2.0)
        axes[2, column].imshow(
            luminance_differences[column],
            cmap="RdBu_r",
            vmin=-signed_limit,
            vmax=signed_limit,
        )
        axes[3, column].imshow(
            absolute_differences[column],
            cmap="magma",
            vmin=0.0,
            vmax=absolute_limit,
        )
        axes[0, column].set_title(f"t={time_value:g}")
        for row in range(4):
            axes[row, column].axis("off")
    for axis, label in zip(
        axes[:, 0],
        ("strong", "weak", "strong - weak", "|strong - weak|"),
        strict=True,
    ):
        axis.set_ylabel(label, rotation=90)
    figure.suptitle(
        "Decoded clean predictions and differences at identical states",
        fontsize=13,
    )
    figure.savefig(output_dir / "same_state_clean_prediction_preview.png", dpi=160)
    plt.close(figure)


@torch.inference_mode()
def same_state_probe(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("the same-state probe requires CUDA")
    from diffusers.models import AutoencoderKL
    from torchdiffeq import odeint

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
    sit_module, source_metadata = load_official_sit_module(
        args.official_sit_repo.expanduser().resolve(),
        verify_source=args.verify_sit_source,
    )
    model, head, metadata = load_frozen_internal_model(
        head_checkpoint_path=args.checkpoint.expanduser().resolve(),
        head_weights=args.head_weights,
        sit_module=sit_module,
        source_metadata=source_metadata,
        device=device,
    )
    if metadata["prediction_target"] != "velocity":
        raise ValueError("this probe requires a velocity-trained internal head")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        local_files_only=True,
    )
    vae.to(device).eval().requires_grad_(False)

    generator = torch.Generator(device=device).manual_seed(args.probe_seed)
    noise = torch.randn(
        args.probe_samples,
        *LATENT_SHAPE,
        generator=generator,
        device=device,
    )
    labels = torch.randint(
        0,
        NUM_CLASSES,
        (args.probe_samples,),
        generator=generator,
        device=device,
    )
    times = sorted(set(float(value) for value in args.probe_times))
    if not times or times[0] <= 0.0 or times[-1] >= 1.0:
        raise ValueError("probe times must lie strictly inside (0, 1)")
    integration_times = torch.tensor([0.0, *times], device=device, dtype=torch.float32)
    nfe = 0

    def strong_velocity(time_value: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        nonlocal nfe
        nfe += 1
        expanded = time_value.expand(len(state))
        full, _ = full_and_internal_velocity(
            model,
            head,
            state,
            expanded,
            labels,
            internal_depth=int(metadata["internal_depth"]),
            latent_channels=LATENT_SHAPE[0],
        )
        return full.float()

    trajectory = odeint(
        strong_velocity,
        noise.float(),
        integration_times,
        method="dopri5",
        atol=args.atol,
        rtol=args.rtol,
    )[1:]
    records: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    difference_metric_rows: list[dict[str, object]] = []
    spectrum_rows: list[dict[str, object]] = []
    previews: list[tuple[float, np.ndarray, np.ndarray]] = []
    tiny = np.finfo(np.float64).tiny

    for time_index, (time_value, state) in enumerate(zip(times, trajectory, strict=True)):
        expanded = torch.full(
            (len(state),),
            time_value,
            device=device,
            dtype=torch.float32,
        )
        full_velocity, weak_velocity = full_and_internal_velocity(
            model,
            head,
            state,
            expanded,
            labels,
            internal_depth=int(metadata["internal_depth"]),
            latent_channels=LATENT_SHAPE[0],
        )
        strong_clean = state.float() + (1.0 - time_value) * full_velocity.float()
        weak_clean = state.float() + (1.0 - time_value) * weak_velocity.float()
        strong_images = _decode_clean_predictions(
            vae,
            strong_clean,
            chunk_size=args.vae_decode_batch_size,
        )
        weak_images = _decode_clean_predictions(
            vae,
            weak_clean,
            chunk_size=args.vae_decode_batch_size,
        )
        strong = analyze_image_frequencies(
            strong_images,
            radial_bins=args.radial_bins,
            chunk_size=args.frequency_chunk_size,
        )
        weak = analyze_image_frequencies(
            weak_images,
            radial_bins=args.radial_bins,
            chunk_size=args.frequency_chunk_size,
        )
        difference = analyze_image_frequencies(
            strong_images - weak_images,
            radial_bins=args.radial_bins,
            chunk_size=args.frequency_chunk_size,
        )
        pair_rows = summarize_pair(
            strong,
            weak,
            reps=args.bootstrap_reps,
            seed=args.bootstrap_seed + 100 * time_index,
        )
        for row in pair_rows:
            metric_rows.append({"time": time_value, **row})
        for row in summarize_single(difference):
            difference_metric_rows.append({"time": time_value, **row})
        ratio_db = 10.0 * np.log10(
            np.maximum(weak.mean_psd, tiny) / np.maximum(strong.mean_psd, tiny)
        )
        for frequency_index, frequency in enumerate(strong.frequencies):
            spectrum_rows.append(
                {
                    "time": time_value,
                    "frequency_cycles_per_pixel": float(frequency),
                    "strong_mean_psd": float(strong.mean_psd[frequency_index]),
                    "weak_mean_psd": float(weak.mean_psd[frequency_index]),
                    "difference_mean_psd": float(
                        difference.mean_psd[frequency_index]
                    ),
                    "difference_energy_fraction": float(
                        difference.mean_energy_fraction[frequency_index]
                    ),
                    "weak_over_strong_db": float(ratio_db[frequency_index]),
                }
            )
        record: dict[str, object] = {
            "time": time_value,
            "frequencies": strong.frequencies.tolist(),
            "weak_over_strong_db": ratio_db.tolist(),
            "difference_energy_fraction": difference.mean_energy_fraction.tolist(),
        }
        for name in METRIC_NAMES:
            record[f"strong_{name}"] = float(strong.per_image[name].mean())
            record[f"weak_{name}"] = float(weak.per_image[name].mean())
            record[f"difference_{name}"] = float(
                difference.per_image[name].mean()
            )
        records.append(record)
        previews.append((time_value, strong_images[0], weak_images[0]))

    write_csv(args.output_dir / "same_state_metrics.csv", metric_rows)
    write_csv(
        args.output_dir / "same_state_difference_metrics.csv",
        difference_metric_rows,
    )
    write_csv(args.output_dir / "same_state_radial_spectrum.csv", spectrum_rows)
    _plot_state_probe(args.output_dir, records, previews)
    return {
        "probe_samples": int(args.probe_samples),
        "probe_seed": int(args.probe_seed),
        "times": times,
        "baseline_nfe": int(nfe),
        "model": metadata,
        "metrics": metric_rows,
        "difference_metrics": difference_metric_rows,
        "files": {
            "metrics": str(args.output_dir / "same_state_metrics.csv"),
            "difference_metrics": str(
                args.output_dir / "same_state_difference_metrics.csv"
            ),
            "spectrum": str(args.output_dir / "same_state_radial_spectrum.csv"),
            "metric_plot": str(args.output_dir / "same_state_frequency_metrics.png"),
            "ratio_heatmap": str(
                args.output_dir / "same_state_spectral_ratio_heatmap.png"
            ),
            "difference_metric_plot": str(
                args.output_dir / "same_state_difference_metrics.png"
            ),
            "difference_spectrum_heatmap": str(
                args.output_dir / "same_state_difference_spectrum_heatmap.png"
            ),
            "preview": str(
                args.output_dir / "same_state_clean_prediction_preview.png"
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sample-root", type=Path, default=DEFAULT_SAMPLE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--official-sit-repo", type=Path, default=DEFAULT_OFFICIAL_SIT_REPO)
    parser.add_argument("--head-weights", choices=("raw", "ema"), default="ema")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--terminal-only", action="store_true")
    scope.add_argument("--state-probe-only", action="store_true")
    parser.add_argument("--radial-bins", type=int, default=64)
    parser.add_argument("--frequency-chunk-size", type=int, default=16)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260817)
    parser.add_argument("--probe-samples", type=int, default=32)
    parser.add_argument("--probe-seed", type=int, default=20260817)
    parser.add_argument(
        "--probe-times",
        type=float,
        nargs="+",
        default=(0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95),
    )
    parser.add_argument("--vae-decode-batch-size", type=int, default=2)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--verify-sit-source",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.sample_root = args.sample_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    existing = _load_json(summary_path) if summary_path.is_file() else {}
    terminal = (
        existing.get("terminal")
        if args.state_probe_only
        else terminal_analysis(args)
    )
    state_probe = None if args.terminal_only else same_state_probe(args)
    payload = {
        "format": "eqvae_imagenet100_sit_internal_head_frequency_v1",
        "scope": (
            "paired terminal rollouts and same-state decoded clean predictions "
            "for frozen v800 full/depth-8 velocity heads"
        ),
        "frequency_protocol": {
            "channel": "BT.709 luminance",
            "dc": "per-image spatial mean removed",
            "window": "separable Hann normalized to unit RMS",
            "radial_frequency_range_cycles_per_pixel": [0.0, 0.5],
            "bands_cycles_per_pixel": {
                "low": [0.0, 0.125],
                "mid": [0.125, 0.25],
                "high": [0.25, 0.5],
            },
            "paired_bootstrap_reps": int(args.bootstrap_reps),
        },
        "terminal": terminal,
        "same_state_probe": state_probe,
    }
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
