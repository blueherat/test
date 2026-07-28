"""Quantify how SPC reallocates detail exposure under the RAE time distribution."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


DEFAULT_OUTPUT = (
    Path.home()
    / "data/eqvae/experiments/rae_spc_multiseed_v1/evaluation/path_exposure"
)


def shifted_logit_normal(
    count: int, *, seed: int, shift: float, mu: float = 0.0, sigma: float = 1.0
) -> np.ndarray:
    generator = np.random.default_rng(seed)
    base = 1.0 / (1.0 + np.exp(-(mu + sigma * generator.standard_normal(count))))
    return shift * base / (1.0 + (shift - 1.0) * base)


def detail_coefficients(
    time: np.ndarray, *, floor: float = 0.2, power: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    remaining = np.maximum(1.0 - np.asarray(time), 0.0)
    state = floor + (1.0 - floor) * remaining**power
    velocity = floor + (1.0 - floor) * (1.0 + power) * remaining**power
    return state, velocity


def exposure_summary(time: np.ndarray, state: np.ndarray, velocity: np.ndarray) -> dict[str, object]:
    threshold = 1.0 - math.sqrt(1.0 / 3.0)
    return {
        "sample_count": int(len(time)),
        "time_quantiles": {
            str(q): float(np.quantile(time, q))
            for q in (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
        },
        "detail_velocity_equals_static_threshold_t": threshold,
        "fraction_velocity_weaker_than_static": float(np.mean(velocity < 1.0)),
        "fraction_velocity_stronger_than_static": float(np.mean(velocity > 1.0)),
        "mean_detail_state_exposure_ratio": float(np.mean(state)),
        "median_detail_state_exposure_ratio": float(np.median(state)),
        "mean_detail_velocity_magnitude_ratio": float(np.mean(velocity)),
        "median_detail_velocity_magnitude_ratio": float(np.median(velocity)),
        "mean_squared_detail_velocity_ratio": float(np.mean(velocity**2)),
    }


def plot_exposure(
    time: np.ndarray,
    summary: dict[str, object],
    output: Path,
    *,
    floor: float,
    power: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.linspace(0.0, 1.0, 501)
    state, velocity = detail_coefficients(grid, floor=floor, power=power)
    threshold = float(summary["detail_velocity_equals_static_threshold_t"])
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    axes[0].hist(time, bins=80, density=True, color="#4C78A8", alpha=0.9)
    axes[0].axvline(threshold, color="#E45756", linestyle="--", linewidth=2)
    axes[0].set_title("RAE shifted training-time distribution")
    axes[0].set_xlabel("t (0=data, 1=noise)")
    axes[0].set_ylabel("Density")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].plot(grid, state, label="clean state coefficient", linewidth=2.5, color="#59A14F")
    axes[1].plot(grid, velocity, label="|clean velocity coefficient|", linewidth=2.5, color="#E45756")
    axes[1].axhline(1.0, label="static", color="#333333", linestyle="--", linewidth=1.5)
    axes[1].axvline(threshold, color="#999999", linestyle=":", linewidth=1.5)
    axes[1].set_title("SPC rank-16 detail exposure")
    axes[1].set_xlabel("t (0=data, 1=noise)")
    axes[1].set_ylabel("Coefficient relative to static")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-count", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=20_260_719)
    parser.add_argument("--shift", type=float, default=math.sqrt(48.0))
    parser.add_argument("--floor", type=float, default=0.2)
    parser.add_argument("--power", type=float, default=2.0)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    time = shifted_logit_normal(
        args.sample_count, seed=args.seed, shift=args.shift
    )
    state, velocity = detail_coefficients(
        time, floor=args.floor, power=args.power
    )
    summary = exposure_summary(time, state, velocity)
    summary.update(
        {
            "seed": int(args.seed),
            "time_shift": float(args.shift),
            "floor": float(args.floor),
            "power": float(args.power),
        }
    )
    (output / "path_exposure.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    plot_exposure(
        time,
        summary,
        output / "path_exposure.png",
        floor=args.floor,
        power=args.power,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
