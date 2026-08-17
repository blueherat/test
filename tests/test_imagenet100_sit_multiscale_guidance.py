from __future__ import annotations

import torch

from experiments.analyze_imagenet100_sit_multiscale_guidance import fit_band_delays
from experiments.run_imagenet100_sit_multiscale_guidance_study import (
    build_full_conditions,
    parse_gpu_list,
)
from experiments.imagenet100_sit_multiscale_guidance import (
    BAND_NAMES,
    TIME_NAMES,
    band_time_component,
    frequency_statistics,
    interpolate_time_table,
    observation_time_grid,
    ordered_band_component,
    route_depth_by_target_band,
    schedule_depth,
    select_per_sample,
    split_frequency_bands,
    time_partition_weights,
)


def _sinusoid(frequency_index: int, *, size: int = 32) -> torch.Tensor:
    coordinate = torch.arange(size, dtype=torch.float32)
    wave = torch.cos(2.0 * torch.pi * frequency_index * coordinate / size)
    return wave[None, None, None, :].expand(1, 1, size, size).clone()


def _mixture(low: float, mid: float, high: float) -> torch.Tensor:
    components = (_sinusoid(2), _sinusoid(6), _sinusoid(12))
    weights = (low, mid, high)
    return torch.cat(
        [component * float(weight) ** 0.5 for component, weight in zip(components, weights)],
        dim=1,
    )


def test_frequency_partition_reconstructs_complete_field() -> None:
    generator = torch.Generator().manual_seed(17)
    field = torch.randn(3, 4, 32, 32, generator=generator)
    bands = split_frequency_bands(field)
    reconstructed = sum(bands.values())
    torch.testing.assert_close(reconstructed, field, atol=2e-6, rtol=2e-6)

    statistics = frequency_statistics(field)
    total_fraction = sum(statistics[f"{band}_fraction"] for band in BAND_NAMES)
    torch.testing.assert_close(total_fraction, torch.ones_like(total_fraction))


def test_single_frequency_components_land_in_declared_bands() -> None:
    for expected, index in (("low", 2), ("mid", 6), ("high", 12)):
        statistics = frequency_statistics(_sinusoid(index))
        assert float(statistics[f"{expected}_fraction"].item()) > 0.999999


def test_time_partition_and_nine_cells_are_conservative() -> None:
    field = torch.randn(4, 3, 32, 32, generator=torch.Generator().manual_seed(23))
    times = torch.tensor([0.05, 1.0 / 3.0, 2.0 / 3.0, 0.95])
    weights = time_partition_weights(times)
    total_weight = sum(weights.values())
    torch.testing.assert_close(total_weight, torch.ones_like(times))
    assert all(bool((weight >= 0.0).all()) for weight in weights.values())

    reconstructed = torch.zeros_like(field)
    for band in BAND_NAMES:
        for interval in TIME_NAMES:
            reconstructed += band_time_component(
                field,
                times,
                band=band,
                interval=interval,
            )
    torch.testing.assert_close(reconstructed, field, atol=3e-6, rtol=3e-6)


def test_ordered_bands_use_opposite_early_late_scales() -> None:
    field = _mixture(1.0, 1.0, 1.0).expand(2, -1, -1, -1).clone()
    times = torch.tensor([0.05, 0.95])
    coarse = ordered_band_component(field, times, order="coarse_to_fine")
    reverse = ordered_band_component(field, times, order="fine_to_coarse")

    coarse_stats = frequency_statistics(coarse)
    reverse_stats = frequency_statistics(reverse)
    assert float(coarse_stats["low_fraction"][0]) > 0.999
    assert float(coarse_stats["high_fraction"][1]) > 0.999
    assert float(reverse_stats["high_fraction"][0]) > 0.999
    assert float(reverse_stats["low_fraction"][1]) > 0.999


def test_per_sample_selection_and_depth_schedules() -> None:
    candidates = {
        4: torch.full((3, 1, 2, 2), 4.0),
        8: torch.full((3, 1, 2, 2), 8.0),
        10: torch.full((3, 1, 2, 2), 10.0),
    }
    selected = select_per_sample(candidates, torch.tensor([10, 4, 8]))
    torch.testing.assert_close(selected[:, 0, 0, 0], torch.tensor([10.0, 4.0, 8.0]))

    times = torch.tensor([0.05, 0.5, 0.95])
    forward = schedule_depth(times, order="coarse_to_fine")
    reverse = schedule_depth(times, order="fine_to_coarse")
    assert forward.tolist() == [4, 8, 10]
    assert reverse.tolist() == [10, 8, 4]


def test_spectral_router_and_anti_router_are_predeclared_opposites() -> None:
    gaps = {
        4: _mixture(0.90, 0.08, 0.02).expand(3, -1, -1, -1).clone(),
        8: _mixture(0.10, 0.80, 0.10).expand(3, -1, -1, -1).clone(),
        10: _mixture(0.02, 0.08, 0.90).expand(3, -1, -1, -1).clone(),
    }
    times = torch.tensor([0.05, 0.5, 0.95])
    routed, selected = route_depth_by_target_band(gaps, times)
    anti, anti_selected = route_depth_by_target_band(gaps, times, reverse=True)
    assert selected.tolist() == [4, 8, 10]
    assert anti_selected.tolist() == [10, 4, 4]
    assert not torch.equal(routed, anti)


def test_time_table_interpolation_is_linear_and_clamped() -> None:
    query = torch.tensor([-1.0, 0.25, 0.75, 2.0])
    values = interpolate_time_table(query, [0.0, 0.5, 1.0], [0.0, 1.0, 0.0])
    torch.testing.assert_close(values, torch.tensor([0.0, 0.5, 0.5, 0.0]))


def test_formal_observation_grid_is_strict_after_float32_quantization() -> None:
    times = observation_time_grid(
        0.02,
        0.98,
        49,
        anchors=(0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95),
    )
    tensor = torch.tensor((0.0, *times), dtype=torch.float32)
    assert torch.all(tensor[1:] > tensor[:-1])
    for anchor in (0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95):
        assert float(torch.tensor(anchor, dtype=torch.float32)) in times


def test_delay_fit_reports_physical_time_instead_of_reusing_sampler_steps() -> None:
    times = [0.1 * index for index in range(10)]
    strong = [_mixture(1.0, 1.0, 1.0) * (index + 1) for index in range(10)]
    weak = [strong[max(0, index - 2)].clone() for index in range(10)]
    lag_time, atlas_steps, rows = fit_band_delays(
        times=times,
        strong_clean=strong,
        weak_clean=weak,
        max_lag_steps=2,
    )
    assert atlas_steps == {"low": 2, "mid": 2, "high": 2}
    for band in BAND_NAMES:
        assert abs(lag_time[band] - 0.2) < 1e-6
    assert all("realized_lag_time_mean" in row for row in rows)


def test_full_protocol_can_disable_fid5k_confirmations() -> None:
    conditions = build_full_conditions(
        screen_samples=1_000,
        confirm_samples=0,
        euler_steps=100,
    )
    assert len(conditions) == 99
    assert {condition.payload["evaluation_group"] for condition in conditions} == {
        "screen"
    }
    assert {condition.num_samples for condition in conditions} == {1_000}


def test_evaluation_gpu_list_is_explicit_and_unique() -> None:
    assert parse_gpu_list("2,3") == (2, 3)
