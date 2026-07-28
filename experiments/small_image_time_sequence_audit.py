"""Audit the exact flow-matching time streams used by the small-image toy.

The training stream uses one generator for minibatch indices, Gaussian bridge
noise, and time draws, in that order.  This module replays those calls without
training a model so aggregate coverage can be separated from ordering effects.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mnist_spectral_rollout_toy import shifted_uniform  # noqa: E402
from experiments.small_image_basis_transport import (  # noqa: E402
    build_direction_analyzer,
    load_small_image_tensors,
)


DEFAULT_OUTPUT_ROOT = Path.home() / "data/eqvae/experiments/small_image_time_sequence_audit"


@dataclass(frozen=True)
class TimeSequenceAuditConfig:
    data_root: Path = Path("/data/shared/mnist")
    output_root: Path = DEFAULT_OUTPUT_ROOT
    data_seed: int = 4
    stream_seeds: tuple[int, ...] = (3, 4)
    device: str = "cuda:0"
    train_size: int = 8_192
    batch_size: int = 128
    steps: int = 1_000
    spatial_shape: tuple[int, int, int] = (1, 28, 28)
    time_shift: float = 1.0
    gamma: float = 0.5
    band_count: int = 8
    histogram_bins: int = 20
    training_windows: int = 10
    save: bool = True


@torch.no_grad()
def replay_time_draws(
    *,
    seed: int,
    train_size: int,
    batch_size: int,
    steps: int,
    spatial_shape: Sequence[int],
    time_shift: float,
    device: torch.device | str,
) -> torch.Tensor:
    """Replay exact time draws after advancing index and noise draws."""

    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(int(seed) + 101)
    rows = []
    sample_shape = (int(batch_size), *(int(value) for value in spatial_shape))
    for _ in range(int(steps)):
        torch.randint(
            int(train_size),
            (int(batch_size),),
            device=device,
            generator=generator,
        )
        torch.randn(sample_shape, device=device, generator=generator)
        rows.append(
            shifted_uniform(
                int(batch_size),
                float(time_shift),
                device=device,
                generator=generator,
            )
        )
    return torch.stack(rows).cpu()


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    levels = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95])
    result = torch.quantile(values.float(), levels)
    return {
        "q05": float(result[0]),
        "q25": float(result[1]),
        "q50": float(result[2]),
        "q75": float(result[3]),
        "q95": float(result[4]),
    }


def summarize_time_draws(
    streams: Mapping[int, torch.Tensor],
    *,
    histogram_bins: int,
    training_windows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return global/window summaries, histograms, and per-step statistics."""

    summary_rows: list[dict[str, float | int | str]] = []
    histogram_rows: list[dict[str, float | int]] = []
    step_rows: list[dict[str, float | int]] = []
    for seed, times in sorted(streams.items()):
        if times.ndim != 2:
            raise ValueError("each time stream must have shape [steps, batch]")
        flat = times.flatten().float()
        summary_rows.append(
            {
                "seed": int(seed),
                "window": "all",
                "start_step": 1,
                "end_step": int(times.shape[0]),
                "count": int(flat.numel()),
                "mean": float(flat.mean()),
                "std": float(flat.std(unbiased=False)),
                "low_rate_t_lt_0.1": float((flat < 0.1).float().mean()),
                "middle_rate_0.4_to_0.6": float(
                    ((flat >= 0.4) & (flat < 0.6)).float().mean()
                ),
                "high_rate_t_ge_0.9": float((flat >= 0.9).float().mean()),
                **_quantiles(flat),
            }
        )
        for window_index, window in enumerate(
            torch.tensor_split(times, int(training_windows), dim=0)
        ):
            window_flat = window.flatten().float()
            start = sum(
                part.shape[0]
                for part in torch.tensor_split(times, int(training_windows), dim=0)[
                    :window_index
                ]
            )
            summary_rows.append(
                {
                    "seed": int(seed),
                    "window": f"window_{window_index + 1:02d}",
                    "start_step": int(start + 1),
                    "end_step": int(start + window.shape[0]),
                    "count": int(window_flat.numel()),
                    "mean": float(window_flat.mean()),
                    "std": float(window_flat.std(unbiased=False)),
                    "low_rate_t_lt_0.1": float((window_flat < 0.1).float().mean()),
                    "middle_rate_0.4_to_0.6": float(
                        ((window_flat >= 0.4) & (window_flat < 0.6)).float().mean()
                    ),
                    "high_rate_t_ge_0.9": float((window_flat >= 0.9).float().mean()),
                    **_quantiles(window_flat),
                }
            )
        counts = torch.histc(flat, bins=int(histogram_bins), min=0.0, max=1.0)
        frequencies = counts / counts.sum()
        for index, (count, frequency) in enumerate(zip(counts, frequencies)):
            histogram_rows.append(
                {
                    "seed": int(seed),
                    "bin": int(index),
                    "left": index / int(histogram_bins),
                    "right": (index + 1) / int(histogram_bins),
                    "count": int(count),
                    "frequency": float(frequency),
                }
            )
        for step, row in enumerate(times, start=1):
            step_rows.append(
                {
                    "seed": int(seed),
                    "step": int(step),
                    "mean": float(row.mean()),
                    "std": float(row.std(unbiased=False)),
                    "low_rate_t_lt_0.1": float((row < 0.1).float().mean()),
                    "high_rate_t_ge_0.9": float((row >= 0.9).float().mean()),
                }
            )
    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(histogram_rows),
        pd.DataFrame(step_rows),
    )


def compare_time_streams(
    streams: Mapping[int, torch.Tensor], histogram: pd.DataFrame
) -> pd.DataFrame:
    """Compare empirical distributions and order-sensitive cumulative exposure."""

    rows = []
    seeds = sorted(streams)
    for first_index, first_seed in enumerate(seeds):
        for second_seed in seeds[first_index + 1 :]:
            first = streams[first_seed].float()
            second = streams[second_seed].float()
            if first.shape != second.shape:
                raise ValueError("time streams must have identical shapes")
            first_sorted = first.flatten().sort().values
            second_sorted = second.flatten().sort().values
            merged = torch.cat([first_sorted, second_sorted]).sort().values
            first_cdf = torch.searchsorted(first_sorted, merged, right=True).float()
            second_cdf = torch.searchsorted(second_sorted, merged, right=True).float()
            first_cdf /= first_sorted.numel()
            second_cdf /= second_sorted.numel()
            first_hist = histogram[histogram["seed"].eq(first_seed)][
                "frequency"
            ].to_numpy()
            second_hist = histogram[histogram["seed"].eq(second_seed)][
                "frequency"
            ].to_numpy()
            first_step = first.mean(dim=1)
            second_step = second.mean(dim=1)
            centered_difference = first_step - second_step
            cumulative = centered_difference.cumsum(dim=0)
            rows.append(
                {
                    "seed_a": int(first_seed),
                    "seed_b": int(second_seed),
                    "wasserstein_1": float((first_sorted - second_sorted).abs().mean()),
                    "ks_distance": float((first_cdf - second_cdf).abs().max()),
                    "histogram_tv": float(0.5 * abs(first_hist - second_hist).sum()),
                    "max_histogram_bin_difference": float(
                        abs(first_hist - second_hist).max()
                    ),
                    "mean_abs_step_mean_difference": float(
                        centered_difference.abs().mean()
                    ),
                    "max_abs_cumulative_mean_difference": float(
                        cumulative.abs().max()
                    ),
                    "final_cumulative_mean_difference": float(cumulative[-1]),
                    "step_mean_correlation": float(
                        torch.corrcoef(torch.stack([first_step, second_step]))[0, 1]
                    ),
                }
            )
    return pd.DataFrame(rows)


@torch.no_grad()
def summarize_band_weight_exposure(
    streams: Mapping[int, torch.Tensor],
    analyzer: torch.nn.Module,
    *,
    training_windows: int,
    device: torch.device | str,
) -> pd.DataFrame:
    """Summarize the time-dependent spectral weights actually presented."""

    device = torch.device(device)
    rows: list[dict[str, float | int | str]] = []
    for seed, times in sorted(streams.items()):
        weights = analyzer.weights(times.flatten().to(device))
        band_count = int(analyzer.band_count)
        if weights.shape[1] != band_count:
            if not hasattr(analyzer, "group_index"):
                raise ValueError("component weights require analyzer.group_index")
            group_index = analyzer.group_index.to(device=device)
            if group_index.numel() != weights.shape[1]:
                raise ValueError("group_index does not match component weights")
            band_sums = torch.zeros(
                (weights.shape[0], band_count),
                device=device,
                dtype=weights.dtype,
            )
            band_sums.scatter_add_(
                1, group_index[None].expand(weights.shape[0], -1), weights
            )
            counts = torch.bincount(group_index, minlength=band_count).to(weights.dtype)
            weights = band_sums / counts.clamp_min(1.0)[None]
        weights = weights.cpu().reshape(*times.shape, band_count)
        windows = [("all", 0, weights.shape[0], weights)]
        offset = 0
        for index, window in enumerate(
            torch.tensor_split(weights, int(training_windows), dim=0)
        ):
            windows.append(
                (f"window_{index + 1:02d}", offset, offset + window.shape[0], window)
            )
            offset += window.shape[0]
        for label, start, end, window in windows:
            flattened = window.flatten(0, 1)
            for band in range(flattened.shape[1]):
                rows.append(
                    {
                        "seed": int(seed),
                        "window": label,
                        "start_step": int(start + 1),
                        "end_step": int(end),
                        "band": int(band),
                        "mean_weight": float(flattened[:, band].mean()),
                        "std_weight": float(flattened[:, band].std(unbiased=False)),
                    }
                )
    return pd.DataFrame(rows)


def run_audit(
    config: TimeSequenceAuditConfig = TimeSequenceAuditConfig(),
) -> tuple[dict[str, pd.DataFrame], Path | None]:
    device = torch.device(
        config.device
        if not config.device.startswith("cuda") or torch.cuda.is_available()
        else "cpu"
    )
    loaded = load_small_image_tensors(
        "mnist",
        config.data_root,
        config.train_size,
        1,
        config.data_seed,
        download=True,
    )
    analyzer, _ = build_direction_analyzer(
        loaded["train"].to(device),
        "dct",
        band_count=config.band_count,
        gamma=config.gamma,
        seed=config.data_seed,
    )
    analyzer = analyzer.to(device)
    streams = {
        int(seed): replay_time_draws(
            seed=int(seed),
            train_size=config.train_size,
            batch_size=config.batch_size,
            steps=config.steps,
            spatial_shape=config.spatial_shape,
            time_shift=config.time_shift,
            device=device,
        )
        for seed in config.stream_seeds
    }
    summary, histogram, per_step = summarize_time_draws(
        streams,
        histogram_bins=config.histogram_bins,
        training_windows=config.training_windows,
    )
    comparisons = compare_time_streams(streams, histogram)
    weight_exposure = summarize_band_weight_exposure(
        streams,
        analyzer,
        training_windows=config.training_windows,
        device=device,
    )
    tables = {
        "time_summary": summary,
        "time_histogram": histogram,
        "time_comparisons": comparisons,
        "per_step_summary": per_step,
        "band_weight_exposure": weight_exposure,
    }
    result_dir = None
    if config.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = config.output_root.expanduser() / f"audit_{timestamp}"
        result_dir.mkdir(parents=True, exist_ok=False)
        serialized = asdict(config)
        serialized["data_root"] = str(config.data_root)
        serialized["output_root"] = str(config.output_root.expanduser())
        (result_dir / "config.json").write_text(
            json.dumps(serialized, indent=2), encoding="utf-8"
        )
        for name, table in tables.items():
            table.to_csv(result_dir / f"{name}.csv", index=False)
    return tables, result_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    config = TimeSequenceAuditConfig(device=args.device, save=not args.no_save)
    if args.quick:
        config = TimeSequenceAuditConfig(
            device=args.device,
            train_size=128,
            batch_size=8,
            steps=10,
            spatial_shape=(1, 28, 28),
            histogram_bins=5,
            training_windows=2,
            save=not args.no_save,
        )
    tables, result_dir = run_audit(config)
    print(f"result_dir={result_dir}")
    print(tables["time_summary"].query("window == 'all'").to_string(index=False))
    print(tables["time_comparisons"].to_string(index=False))


if __name__ == "__main__":
    main()
