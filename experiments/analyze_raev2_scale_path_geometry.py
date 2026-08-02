"""Measure curvature of the RAEv2 guidance-scale endpoint path.

The analysis is post-hoc and inference-free: it reads completed scale-response
artifacts and compares each actual endpoint at scale ``s`` with the straight
chord between the endpoints at two fixed scales.  Running the same comparison
in latent, round-trip, pixel, and feature spaces separates curvature already
present in sampling from curvature introduced by the observation map.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROTOCOL = "raev2_scale_path_geometry_v1"
DEFAULT_OUTPUT = (
    Path.home() / "data/eqvae/experiments/raev2_scale_path_geometry/n5000x2_v1"
)


def parse_named_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("run must be NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("run name cannot be empty")
    return name, Path(raw_path).expanduser().resolve()


def scale_key(scale: float) -> str:
    if not math.isfinite(float(scale)):
        raise ValueError("scale must be finite")
    return f"scale_s{float(scale):.6f}".replace(".", "p")


def low_frequency_features(images: np.ndarray, grid_size: int = 16) -> np.ndarray:
    """Return block means and standard deviations from NHWC images."""

    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("images must be an NHWC RGB array")
    height, width = int(images.shape[1]), int(images.shape[2])
    if grid_size <= 0 or height % grid_size or width % grid_size:
        raise ValueError("grid_size must divide both image dimensions")
    values = images.astype(np.float32, copy=False)
    block_h, block_w = height // grid_size, width // grid_size
    blocks = values.reshape(
        values.shape[0], grid_size, block_h, grid_size, block_w, 3
    )
    means = blocks.mean(axis=(2, 4), dtype=np.float32)
    second = np.square(blocks, dtype=np.float32).mean(axis=(2, 4), dtype=np.float32)
    deviations = np.sqrt(np.maximum(second - np.square(means), 0.0))
    return np.concatenate(
        [means.reshape(len(values), -1), deviations.reshape(len(values), -1)],
        axis=1,
    )


def chord_metrics(
    control: np.ndarray,
    anchor: np.ndarray,
    actual: np.ndarray,
    interpolation: float,
) -> dict[str, np.ndarray | float]:
    """Compare an actual endpoint with the corresponding straight chord point."""

    if control.shape != anchor.shape or control.shape != actual.shape:
        raise ValueError("control, anchor, and actual arrays must align")
    if control.ndim < 2:
        raise ValueError("arrays must contain a batch dimension")
    left = actual.astype(np.float32, copy=False) - control.astype(np.float32, copy=False)
    direction = anchor.astype(np.float32, copy=False) - control.astype(
        np.float32, copy=False
    )
    chord = float(interpolation) * direction
    residual = left - chord
    left = left.reshape(len(left), -1)
    direction = direction.reshape(len(direction), -1)
    chord = chord.reshape(len(chord), -1)
    residual = residual.reshape(len(residual), -1)

    left_sq = np.einsum("ij,ij->i", left, left, dtype=np.float64)
    direction_sq = np.einsum("ij,ij->i", direction, direction, dtype=np.float64)
    chord_sq = np.einsum("ij,ij->i", chord, chord, dtype=np.float64)
    residual_sq = np.einsum("ij,ij->i", residual, residual, dtype=np.float64)
    cross = np.einsum("ij,ij->i", left, chord, dtype=np.float64)
    eps = 1e-30
    valid = (left_sq > eps) & (chord_sq > eps)
    cosine = np.full(len(left), np.nan, dtype=np.float64)
    cosine[valid] = cross[valid] / np.sqrt(left_sq[valid] * chord_sq[valid])
    relative_actual = np.sqrt(residual_sq / np.maximum(left_sq, eps))
    relative_chord = np.sqrt(residual_sq / np.maximum(chord_sq, eps))
    radial_gain = np.sqrt(left_sq / np.maximum(chord_sq, eps))
    return {
        "sample_count": int(len(left)),
        "actual_sq_sum": float(left_sq.sum()),
        "chord_sq_sum": float(chord_sq.sum()),
        "residual_sq_sum": float(residual_sq.sum()),
        "cross_sum": float(cross.sum()),
        "relative_to_actual": relative_actual,
        "relative_to_chord": relative_chord,
        "radial_gain": radial_gain,
        "cosine": cosine,
        "anchor_direction_sq_sum": float(direction_sq.sum()),
    }


def _summary(values: Iterable[float], prefix: str) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_p10": float("nan"),
            f"{prefix}_p90": float("nan"),
        }
    return {
        f"{prefix}_mean": float(array.mean()),
        f"{prefix}_median": float(np.median(array)),
        f"{prefix}_p10": float(np.quantile(array, 0.1)),
        f"{prefix}_p90": float(np.quantile(array, 0.9)),
    }


def _rank_path(root: Path, directory: str, condition: str, rank: int) -> Path:
    return root / directory / f"{condition}_rank{rank:02d}.npy"


def _space_spec(space: str) -> tuple[str, bool]:
    if space == "latent":
        return "latents", False
    if space == "roundtrip":
        return "roundtrip", False
    if space == "decoded_pixel":
        return "decoded", False
    if space == "decoded_lowfreq":
        return "decoded", True
    if space == "decoded_inception":
        return "inception", False
    raise ValueError(f"unknown space: {space}")


def _run_space_scale(
    root: Path,
    *,
    space: str,
    scale: float,
    control_scale: float,
    anchor_scale: float,
    world_size: int,
    chunk_size: int,
    low_frequency_grid: int,
) -> dict[str, float | int | str]:
    directory, make_low_frequency = _space_spec(space)
    control_key = scale_key(control_scale)
    anchor_key = scale_key(anchor_scale)
    actual_key = scale_key(scale)
    interpolation = (float(scale) - float(control_scale)) / (
        float(anchor_scale) - float(control_scale)
    )
    chunks: list[dict[str, np.ndarray | float]] = []
    for rank in range(world_size):
        control = np.load(
            _rank_path(root, directory, control_key, rank),
            mmap_mode="r",
            allow_pickle=False,
        )
        anchor = np.load(
            _rank_path(root, directory, anchor_key, rank),
            mmap_mode="r",
            allow_pickle=False,
        )
        actual = np.load(
            _rank_path(root, directory, actual_key, rank),
            mmap_mode="r",
            allow_pickle=False,
        )
        if control.shape != anchor.shape or control.shape != actual.shape:
            raise RuntimeError(f"rank {rank} array mismatch in {space}")
        for start in range(0, len(control), chunk_size):
            stop = min(start + chunk_size, len(control))
            values = [control[start:stop], anchor[start:stop], actual[start:stop]]
            if make_low_frequency:
                values = [
                    low_frequency_features(value, low_frequency_grid)
                    for value in values
                ]
            chunks.append(chord_metrics(*values, interpolation))

    sample_count = int(sum(int(chunk["sample_count"]) for chunk in chunks))
    actual_sq = float(sum(float(chunk["actual_sq_sum"]) for chunk in chunks))
    chord_sq = float(sum(float(chunk["chord_sq_sum"]) for chunk in chunks))
    residual_sq = float(sum(float(chunk["residual_sq_sum"]) for chunk in chunks))
    cross = float(sum(float(chunk["cross_sum"]) for chunk in chunks))

    def joined(key: str) -> np.ndarray:
        return np.concatenate([np.asarray(chunk[key]) for chunk in chunks])

    aggregate_cosine = cross / max(math.sqrt(actual_sq * chord_sq), 1e-30)
    return {
        "space": space,
        "scale": float(scale),
        "control_scale": float(control_scale),
        "anchor_scale": float(anchor_scale),
        "interpolation": float(interpolation),
        "sample_count": sample_count,
        "aggregate_chord_residual_over_actual": math.sqrt(
            residual_sq / max(actual_sq, 1e-30)
        ),
        "aggregate_chord_residual_over_chord": math.sqrt(
            residual_sq / max(chord_sq, 1e-30)
        ),
        "aggregate_radial_gain": math.sqrt(actual_sq / max(chord_sq, 1e-30)),
        "aggregate_chord_cosine": aggregate_cosine,
        **_summary(joined("relative_to_actual"), "sample_residual_over_actual"),
        **_summary(joined("relative_to_chord"), "sample_residual_over_chord"),
        **_summary(joined("radial_gain"), "sample_radial_gain"),
        **_summary(joined("cosine"), "sample_chord_cosine"),
    }


def analyze_run(
    name: str,
    root: Path,
    *,
    control_scale: float,
    anchor_scale: float,
    chunk_size: int,
    low_frequency_grid: int,
) -> pd.DataFrame:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError(f"run is not complete: {root}")
    scales = [float(value) for value in manifest["scales"]]
    if control_scale not in scales or anchor_scale not in scales:
        raise ValueError("control and anchor scales must exist in every run")
    selected = [value for value in scales if value >= control_scale and value != control_scale]
    spaces = (
        "latent",
        "roundtrip",
        "decoded_pixel",
        "decoded_lowfreq",
        "decoded_inception",
    )
    rows = []
    for space in spaces:
        for scale in selected:
            row = _run_space_scale(
                root,
                space=space,
                scale=scale,
                control_scale=control_scale,
                anchor_scale=anchor_scale,
                world_size=int(manifest["world_size"]),
                chunk_size=chunk_size,
                low_frequency_grid=low_frequency_grid,
            )
            rows.append(
                {
                    "run": name,
                    "seed": int(manifest["seed"]),
                    "artifact": str(root),
                    **row,
                }
            )
    return pd.DataFrame(rows)


def plot_results(frame: pd.DataFrame, path: Path) -> None:
    labels = {
        "latent": "raw latent",
        "roundtrip": "roundtrip latent",
        "decoded_pixel": "clamped pixels",
        "decoded_lowfreq": "low-frequency pixels",
        "decoded_inception": "Inception features",
    }
    colors = {
        "latent": "#4c78a8",
        "roundtrip": "#e45756",
        "decoded_pixel": "#72b7b2",
        "decoded_lowfreq": "#f2cf5b",
        "decoded_inception": "#54a24b",
    }
    figure, axes = plt.subplots(1, 3, figsize=(21, 6.5), constrained_layout=True)
    metrics = (
        ("aggregate_chord_residual_over_actual", "Chord residual / actual displacement"),
        ("aggregate_chord_cosine", "Actual vs chord cosine"),
        ("aggregate_radial_gain", "Actual displacement / chord displacement"),
    )
    for space in labels:
        values = frame[frame.space.eq(space)]
        for axis, (metric, ylabel) in zip(axes, metrics):
            grouped = values.groupby("scale")[metric]
            mean = grouped.mean()
            minimum = grouped.min()
            maximum = grouped.max()
            axis.plot(
                mean.index,
                mean.values,
                marker="o",
                linewidth=2,
                color=colors[space],
                label=labels[space],
            )
            axis.fill_between(
                mean.index,
                minimum.values,
                maximum.values,
                color=colors[space],
                alpha=0.12,
            )
            axis.set_xlabel("Internal-guidance scale")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.2)
    axes[0].axhline(0.0, color="black", linewidth=1, linestyle="--")
    axes[1].axhline(1.0, color="black", linewidth=1, linestyle="--")
    axes[2].axhline(1.0, color="black", linewidth=1, linestyle="--")
    axes[0].legend(frameon=False, fontsize=9)
    figure.suptitle(
        "RAEv2 guidance-scale path geometry (band: two-seed range)", fontsize=16
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=parse_named_run, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--control-scale", type=float, default=1.0)
    parser.add_argument("--anchor-scale", type=float, default=1.78)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--low-frequency-grid", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.anchor_scale == args.control_scale:
        raise ValueError("anchor scale must differ from control scale")
    if args.chunk_size <= 0:
        raise ValueError("chunk size must be positive")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    frames = [
        analyze_run(
            name,
            root,
            control_scale=float(args.control_scale),
            anchor_scale=float(args.anchor_scale),
            chunk_size=int(args.chunk_size),
            low_frequency_grid=int(args.low_frequency_grid),
        )
        for name, root in args.run
    ]
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(output / "path_geometry.csv", index=False)
    plot_results(frame, output / "path_geometry.png")
    anchor = frame[np.isclose(frame.scale, float(args.anchor_scale))]
    tolerance = 1e-5
    summary = {
        "protocol": PROTOCOL,
        "runs": {
            name: str(root) for name, root in args.run
        },
        "control_scale": float(args.control_scale),
        "anchor_scale": float(args.anchor_scale),
        "spaces": sorted(frame.space.unique().tolist()),
        "anchor_identity_check": bool(
            (anchor.aggregate_chord_residual_over_actual <= tolerance).all()
            and (np.abs(anchor.aggregate_chord_cosine - 1.0) <= tolerance).all()
        ),
        "interpretation_guardrail": (
            "Curvature in raw latent implicates recursive scale-dependent sampling; "
            "additional curvature after a map implicates that observation map. This "
            "post-hoc comparison does not identify a local full-base direction."
        ),
    }
    atomic_json(output / "summary.json", summary)
    print(frame.to_string(index=False))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
