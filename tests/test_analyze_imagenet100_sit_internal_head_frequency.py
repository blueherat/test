from __future__ import annotations

import numpy as np

from experiments.analyze_imagenet100_sit_internal_head_frequency import (
    analyze_image_frequencies,
    bootstrap_mean_interval,
    summarize_pair,
    summarize_single,
)


def sinusoid_images(frequency: int, *, count: int = 8, size: int = 64) -> np.ndarray:
    coordinate = np.arange(size, dtype=np.float32)
    wave = np.sin(2.0 * np.pi * frequency * coordinate / size)
    image = np.broadcast_to(wave[None, :], (size, size))
    rgb = np.repeat(image[..., None], 3, axis=-1)
    return np.repeat(rgb[None], count, axis=0)


def test_frequency_metrics_order_low_and_high_sinusoids() -> None:
    low = analyze_image_frequencies(sinusoid_images(2), radial_bins=16, chunk_size=4)
    high = analyze_image_frequencies(sinusoid_images(24), radial_bins=16, chunk_size=4)

    assert low.per_image["low_fraction"].mean() > 0.95
    assert high.per_image["high_fraction"].mean() > 0.90
    assert (
        high.per_image["spectral_centroid_cpp"].mean()
        > low.per_image["spectral_centroid_cpp"].mean()
    )
    assert high.per_image["gradient_rms"].mean() > low.per_image["gradient_rms"].mean()


def test_pair_summary_uses_weak_minus_strong_sign() -> None:
    strong = analyze_image_frequencies(sinusoid_images(2), radial_bins=16)
    weak = analyze_image_frequencies(sinusoid_images(24), radial_bins=16)
    rows = summarize_pair(strong, weak, reps=100, seed=3)
    by_name = {row["metric"]: row for row in rows}

    assert by_name["high_fraction"]["weak_minus_strong"] > 0
    assert by_name["low_fraction"]["weak_minus_strong"] < 0


def test_bootstrap_constant_delta_is_exact() -> None:
    low, high = bootstrap_mean_interval(np.full(12, 0.25), reps=100, seed=7)
    assert np.isclose(low, 0.25)
    assert np.isclose(high, 0.25)


def test_single_summary_reports_all_frequency_metrics() -> None:
    result = analyze_image_frequencies(sinusoid_images(4), radial_bins=16)
    rows = summarize_single(result)

    assert {row["metric"] for row in rows} == set(result.per_image)
    assert all(np.isfinite(row["mean"]) for row in rows)
