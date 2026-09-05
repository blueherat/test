#!/usr/bin/env python3
"""Audit whether RAEv2's base/full heads resemble nested L2 projections.

The spectral mechanism cache stores enough quadratic statistics to test the
normal equations without another model forward pass.  For target ``Y``, base
prediction ``B``, full prediction ``F``, and innovation ``D = F - B``, exact
nested orthogonal projections would satisfy

    E <B, D> = E <Y - F, D> = E <Y - F, F> = 0.

This script reconstructs those terms per time and frequency band.  It also
reports, but deliberately does not endorse, the scale that would match the
target's raw second moment.  That scale includes irreducible posterior
variance and therefore is not a valid guidance prescription by itself.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


REQUIRED_FIELDS = (
    "A",
    "C",
    "Q",
    "T",
    "E",
    "mse_full",
    "mse_base",
)


def safe_correlation(cross: float, left_power: float, right_power: float) -> float:
    denominator = math.sqrt(max(0.0, left_power) * max(0.0, right_power))
    return float(cross / denominator) if denominator > 0.0 else float("nan")


def positive_second_moment_root(
    base_power: float,
    base_gap_cross: float,
    gap_power: float,
    target_power: float,
) -> float:
    """Solve E||B + sD||^2 = E||Y||^2 for the largest real root."""

    if gap_power <= 0.0:
        return float("nan")
    discriminant = base_gap_cross**2 + gap_power * (target_power - base_power)
    if discriminant < 0.0:
        return float("nan")
    return float((-base_gap_cross + math.sqrt(discriminant)) / gap_power)


def derive_row(
    *,
    step: int,
    time: float,
    band: str,
    means: dict[str, float],
) -> dict[str, object]:
    full_power = means["A"]
    full_gap_cross = means["C"]
    gap_power = means["Q"]
    target_power = means["T"]
    residual_gap_cross = means["E"]
    full_mse = means["mse_full"]
    base_mse = means["mse_base"]

    base_power = full_power - 2.0 * full_gap_cross + gap_power
    base_gap_cross = full_gap_cross - gap_power
    residual_full_cross = 0.5 * (target_power - full_power - full_mse)
    mse_optimal_scale = (
        1.0 + residual_gap_cross / gap_power
        if gap_power > 0.0
        else float("nan")
    )
    moment_scale = positive_second_moment_root(
        base_power,
        base_gap_cross,
        gap_power,
        target_power,
    )

    return {
        "step": int(step),
        "time": float(time),
        "band": band,
        "target_power": target_power,
        "base_power": base_power,
        "full_power": full_power,
        "gap_power": gap_power,
        "full_mse": full_mse,
        "base_mse": base_mse,
        "base_gap_cross": base_gap_cross,
        "residual_gap_cross": residual_gap_cross,
        "residual_full_cross": residual_full_cross,
        "base_gap_correlation": safe_correlation(
            base_gap_cross, base_power, gap_power
        ),
        "residual_gap_correlation": safe_correlation(
            residual_gap_cross, full_mse, gap_power
        ),
        "residual_full_correlation": safe_correlation(
            residual_full_cross, full_mse, full_power
        ),
        "mse_optimal_scale_from_base": mse_optimal_scale,
        "target_second_moment_scale_from_base": moment_scale,
        "target_to_full_power_ratio": (
            target_power / full_power if full_power > 0.0 else float("nan")
        ),
        "gap_to_full_power_ratio": (
            gap_power / full_power if full_power > 0.0 else float("nan")
        ),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite_values(rows: list[dict[str, object]], key: str) -> np.ndarray:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return values[np.isfinite(values)]


def summarize(rows: list[dict[str, object]], active_min_time: float) -> dict[str, object]:
    global_rows = [row for row in rows if row["band"] == "all"]
    active_rows = [row for row in global_rows if float(row["time"]) >= active_min_time]

    def range_for(key: str, selected: list[dict[str, object]]) -> dict[str, float]:
        values = finite_values(selected, key)
        return {
            "min": float(values.min()),
            "mean": float(values.mean()),
            "max": float(values.max()),
        }

    return {
        "hypothesis": (
            "base/full approximately form nested L2 projections; this audit "
            "tests normal-equation geometry only and does not imply that "
            "extrapolation improves generation"
        ),
        "active_min_time": float(active_min_time),
        "times": len(global_rows),
        "active_times": len(active_rows),
        "active_correlation": {
            key: range_for(key, active_rows)
            for key in (
                "base_gap_correlation",
                "residual_gap_correlation",
                "residual_full_correlation",
            )
        },
        "active_scale_ranges": {
            key: range_for(key, active_rows)
            for key in (
                "mse_optimal_scale_from_base",
                "target_second_moment_scale_from_base",
            )
        },
        "interpretation_boundary": [
            "Near-zero cross terms support projection-like geometry, not literal nested sigma-fields.",
            "The MSE-optimal scale should remain near one for an accurate posterior mean.",
            "The target-second-moment scale folds irreducible posterior variance into the estimate and must not be used as a sampler scale without an approximation-tail model.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--active-min-time", type=float, default=0.5)
    args = parser.parse_args()

    source = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = np.load(source, allow_pickle=False)
    fields = [str(value) for value in payload["fields"].tolist()]
    missing = [field for field in REQUIRED_FIELDS if field not in fields]
    if missing:
        raise ValueError(f"missing fields: {missing}")
    values = np.asarray(payload["values"], dtype=np.float64)
    steps = np.asarray(payload["steps"], dtype=np.int64)
    if values.ndim != 4 or values.shape[1] != len(steps):
        raise ValueError("expected values [samples, times, bands, fields]")
    field_index = {name: fields.index(name) for name in REQUIRED_FIELDS}

    # Reconstruct the shifted RAEv2 grid directly from the stored step ids.
    shift = math.sqrt(262144.0 / 4096.0)
    raw_grid = np.linspace(1.0, 0.0, 101, dtype=np.float64)
    shifted_grid = shift * raw_grid / (1.0 + (shift - 1.0) * raw_grid)

    rows: list[dict[str, object]] = []
    for time_position, step in enumerate(steps.tolist()):
        if step < 0 or step >= len(shifted_grid):
            raise ValueError(f"step {step} is outside the expected 100-step grid")
        time = float(shifted_grid[step])
        for band_index in range(values.shape[2]):
            means = {
                name: float(values[:, time_position, band_index, index].mean())
                for name, index in field_index.items()
            }
            rows.append(
                derive_row(
                    step=step,
                    time=time,
                    band=str(band_index),
                    means=means,
                )
            )
        means = {
            name: float(values[:, time_position, :, index].sum(axis=1).mean())
            for name, index in field_index.items()
        }
        rows.append(derive_row(step=step, time=time, band="all", means=means))

    write_csv(output_dir / "projection_hierarchy_by_time_band.csv", rows)
    summary = summarize(rows, args.active_min_time)
    summary.update(
        {
            "protocol": "raev2_projection_hierarchy_audit_v1",
            "source": str(source),
            "samples": int(values.shape[0]),
            "bands": int(values.shape[2]),
        }
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
