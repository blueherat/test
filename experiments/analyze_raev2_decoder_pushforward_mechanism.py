"""Mechanism checks for the RAEv2 guidance scale ranking reversal.

This analysis only reads completed scale-response artifacts. It tests whether
same-noise guidance increments are contracted by E(D(.)), whether decoded
feature shifts oppose the autoencoder reconstruction bias, and whether the
decoded ranking survives in simple low-frequency pixel features.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from experiments.run_raev2_decoded_distribution_audit import (
    feature_probe_scores,
    fit_feature_probe,
)
from experiments.run_raev2_scale_response_study import (
    load_ordered_rank_arrays,
    local_ids_for_rank,
    scale_key,
)


PROTOCOL = "raev2_decoder_pushforward_mechanism_v1"


def parse_named_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("run must be NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("run name cannot be empty")
    return name, Path(raw_path).expanduser().resolve()


def _safe_cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0:
        return float("nan")
    return float(np.dot(left, right) / denominator)


def feature_bias_metrics(
    source: np.ndarray,
    reconstruction: np.ndarray,
    candidate: np.ndarray,
    *,
    control: np.ndarray | None = None,
    variance_ridge_ratio: float = 1e-4,
) -> dict[str, float]:
    """Measure whether an IG feature shift opposes decoded distribution errors."""

    if source.shape != reconstruction.shape or source.shape != candidate.shape:
        raise ValueError("source, reconstruction, and candidate must align")
    if control is None:
        control = reconstruction
    if control.shape != source.shape:
        raise ValueError("control must align with the other feature arrays")
    if source.ndim != 2 or source.shape[0] < 2:
        raise ValueError("feature arrays must be matrices with at least two rows")
    source64 = source.astype(np.float64, copy=False)
    reconstruction64 = reconstruction.astype(np.float64, copy=False)
    control64 = control.astype(np.float64, copy=False)
    candidate64 = candidate.astype(np.float64, copy=False)
    source_mean = source64.mean(axis=0)
    reconstruction_bias = reconstruction64.mean(axis=0) - source_mean
    control_error = control64.mean(axis=0) - source_mean
    control_from_reconstruction = (
        control64.mean(axis=0) - reconstruction64.mean(axis=0)
    )
    ig_shift = candidate64.mean(axis=0) - control64.mean(axis=0)
    source_variance = source64.var(axis=0, ddof=1)
    positive = source_variance[source_variance > 0]
    base_scale = float(np.median(positive)) if positive.size else 1.0
    ridge = max(float(variance_ridge_ratio) * base_scale, 1e-12)
    scale = np.sqrt(source_variance + ridge)
    reconstruction_bias_white = reconstruction_bias / scale
    control_error_white = control_error / scale
    control_from_reconstruction_white = control_from_reconstruction / scale
    ig_shift_white = ig_shift / scale

    def summarize(
        reconstruction_error: np.ndarray,
        baseline_error: np.ndarray,
        baseline_from_reconstruction: np.ndarray,
        shift: np.ndarray,
        prefix: str,
    ) -> dict[str, float]:
        reconstruction_sq = float(np.dot(reconstruction_error, reconstruction_error))
        baseline_sq = float(np.dot(baseline_error, baseline_error))
        shift_sq = float(np.dot(shift, shift))
        cross_reconstruction = float(np.dot(reconstruction_error, shift))
        cross_baseline = float(np.dot(baseline_error, shift))
        total_sq = float(np.dot(baseline_error + shift, baseline_error + shift))
        return {
            f"{prefix}_reconstruction_bias_norm": math.sqrt(
                max(reconstruction_sq, 0.0)
            ),
            f"{prefix}_full_error_norm": math.sqrt(max(baseline_sq, 0.0)),
            f"{prefix}_full_from_reconstruction_norm": float(
                np.linalg.norm(baseline_from_reconstruction)
            ),
            f"{prefix}_ig_shift_norm": math.sqrt(max(shift_sq, 0.0)),
            f"{prefix}_ig_vs_reconstruction_bias_cosine": _safe_cosine(
                reconstruction_error, shift
            ),
            f"{prefix}_ig_vs_full_error_cosine": _safe_cosine(
                baseline_error, shift
            ),
            f"{prefix}_ig_reconstruction_cross_term": 2.0 * cross_reconstruction,
            f"{prefix}_ig_full_cross_term": 2.0 * cross_baseline,
            f"{prefix}_ig_mean_error_ratio": total_sq / max(baseline_sq, 1e-30),
            f"{prefix}_ig_mean_error_improvement": 1.0
            - total_sq / max(baseline_sq, 1e-30),
            f"{prefix}_optimal_ig_multiplier": -cross_baseline
            / max(shift_sq, 1e-30),
        }

    return {
        "sample_count": int(source.shape[0]),
        "variance_ridge": ridge,
        **summarize(
            reconstruction_bias,
            control_error,
            control_from_reconstruction,
            ig_shift,
            "raw",
        ),
        **summarize(
            reconstruction_bias_white,
            control_error_white,
            control_from_reconstruction_white,
            ig_shift_white,
            "diag_white",
        ),
    }


def low_frequency_features(images: np.ndarray, grid_size: int) -> np.ndarray:
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
    standard_deviations = np.sqrt(np.maximum(second - np.square(means), 0.0))
    return np.concatenate(
        [means.reshape(values.shape[0], -1), standard_deviations.reshape(values.shape[0], -1)],
        axis=1,
    )


def _summarize_array(values: Iterable[float], prefix: str) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_p10": float("nan"),
            f"{prefix}_p90": float("nan"),
        }
    return {
        f"{prefix}_mean": float(finite.mean()),
        f"{prefix}_median": float(np.median(finite)),
        f"{prefix}_p10": float(np.quantile(finite, 0.1)),
        f"{prefix}_p90": float(np.quantile(finite, 0.9)),
    }


def paired_increment_metrics(
    raw_control: np.ndarray,
    raw_candidate: np.ndarray,
    roundtrip_control: np.ndarray,
    roundtrip_candidate: np.ndarray,
    *,
    probe_weight: np.ndarray | None = None,
) -> dict[str, float]:
    """Summarize one aligned chunk of same-noise guidance increments."""

    expected = raw_control.shape
    if any(
        value.shape != expected
        for value in (raw_candidate, roundtrip_control, roundtrip_candidate)
    ):
        raise ValueError("all paired arrays must have identical shapes")
    raw = raw_candidate.astype(np.float32) - raw_control.astype(np.float32)
    roundtrip = (
        roundtrip_candidate.astype(np.float32)
        - roundtrip_control.astype(np.float32)
    )
    raw_flat = raw.reshape(raw.shape[0], -1)
    roundtrip_flat = roundtrip.reshape(roundtrip.shape[0], -1)
    raw_sq = np.einsum("ij,ij->i", raw_flat, raw_flat, dtype=np.float64)
    roundtrip_sq = np.einsum(
        "ij,ij->i", roundtrip_flat, roundtrip_flat, dtype=np.float64
    )
    cross = np.einsum("ij,ij->i", raw_flat, roundtrip_flat, dtype=np.float64)
    denominator = np.sqrt(raw_sq * roundtrip_sq)
    cosine = np.full(raw.shape[0], np.nan, dtype=np.float64)
    valid = denominator > 0
    cosine[valid] = cross[valid] / denominator[valid]
    ratio = np.full(raw.shape[0], np.nan, dtype=np.float64)
    valid_raw = raw_sq > 0
    ratio[valid_raw] = np.sqrt(roundtrip_sq[valid_raw] / raw_sq[valid_raw])
    result = {
        "sample_count": int(raw.shape[0]),
        "raw_squared_norm_sum": float(raw_sq.sum()),
        "roundtrip_squared_norm_sum": float(roundtrip_sq.sum()),
        **_summarize_array(np.sqrt(raw_sq / raw_flat.shape[1]), "raw_increment_rms"),
        **_summarize_array(
            np.sqrt(roundtrip_sq / roundtrip_flat.shape[1]),
            "roundtrip_increment_rms",
        ),
        **_summarize_array(ratio, "roundtrip_over_raw_norm"),
        **_summarize_array(cosine, "raw_roundtrip_cosine"),
    }
    if probe_weight is not None:
        weight = probe_weight.reshape(-1).astype(np.float32, copy=False)
        if weight.shape[0] != raw_flat.shape[1]:
            raise ValueError("probe weight does not match flattened latent size")
        raw_projection = raw_flat @ weight
        roundtrip_projection = roundtrip_flat @ weight
        result.update(
            {
                "raw_probe_delta_sum": float(raw_projection.sum(dtype=np.float64)),
                "roundtrip_probe_delta_sum": float(
                    roundtrip_projection.sum(dtype=np.float64)
                ),
                "probe_sign_agreement_count": int(
                    np.sum(np.sign(raw_projection) == np.sign(roundtrip_projection))
                ),
            }
        )
    return result


def _merge_paired_chunks(chunks: list[dict[str, float]]) -> dict[str, float]:
    if not chunks:
        raise ValueError("at least one paired chunk is required")
    total = int(sum(int(chunk["sample_count"]) for chunk in chunks))
    raw_sq = float(sum(chunk["raw_squared_norm_sum"] for chunk in chunks))
    roundtrip_sq = float(
        sum(chunk["roundtrip_squared_norm_sum"] for chunk in chunks)
    )
    result = {
        "sample_count": total,
        "aggregate_roundtrip_over_raw_norm": math.sqrt(
            roundtrip_sq / max(raw_sq, 1e-30)
        ),
    }
    weighted_keys = (
        "raw_increment_rms_mean",
        "roundtrip_increment_rms_mean",
        "roundtrip_over_raw_norm_mean",
        "raw_roundtrip_cosine_mean",
    )
    for key in weighted_keys:
        result[key] = float(
            sum(float(chunk[key]) * int(chunk["sample_count"]) for chunk in chunks)
            / total
        )
    if "raw_probe_delta_sum" in chunks[0]:
        raw_projection = float(sum(chunk["raw_probe_delta_sum"] for chunk in chunks))
        roundtrip_projection = float(
            sum(chunk["roundtrip_probe_delta_sum"] for chunk in chunks)
        )
        result.update(
            {
                "raw_probe_delta_mean": raw_projection / total,
                "roundtrip_on_raw_probe_delta_mean": roundtrip_projection / total,
                "raw_probe_survival_ratio": (
                    roundtrip_projection / raw_projection
                    if abs(raw_projection) > 1e-30
                    else float("nan")
                ),
                "probe_sign_agreement_fraction": float(
                    sum(chunk["probe_sign_agreement_count"] for chunk in chunks)
                    / total
                ),
            }
        )
    return result


def _stream_diagonal_probe(
    root: Path,
    condition: str,
    *,
    samples: int,
    world_size: int,
    train_mask: np.ndarray,
    ridge_ratio: float,
    chunk_size: int,
) -> tuple[np.ndarray, float]:
    sums = None
    square_sums = None
    count = 0
    for rank in range(world_size):
        ids = local_ids_for_rank(samples, rank, world_size)
        local_train = train_mask[ids]
        reference = np.load(
            root / "latents" / f"real_rank{rank:02d}.npy", mmap_mode="r"
        )
        candidate = np.load(
            root / "latents" / f"{condition}_rank{rank:02d}.npy", mmap_mode="r"
        )
        for start in range(0, len(ids), chunk_size):
            stop = min(start + chunk_size, len(ids))
            selected = local_train[start:stop]
            if not selected.any():
                continue
            ref = reference[start:stop][selected].astype(np.float32).reshape(
                int(selected.sum()), -1
            )
            cand = candidate[start:stop][selected].astype(np.float32).reshape(
                int(selected.sum()), -1
            )
            if sums is None:
                sums = np.zeros((2, ref.shape[1]), dtype=np.float64)
                square_sums = np.zeros((2, ref.shape[1]), dtype=np.float64)
            sums[0] += ref.sum(axis=0, dtype=np.float64)
            sums[1] += cand.sum(axis=0, dtype=np.float64)
            square_sums[0] += np.square(ref, dtype=np.float32).sum(
                axis=0, dtype=np.float64
            )
            square_sums[1] += np.square(cand, dtype=np.float32).sum(
                axis=0, dtype=np.float64
            )
            count += int(selected.sum())
    if sums is None or square_sums is None or count < 2:
        raise ValueError("insufficient training samples for latent probe")
    means = sums / count
    variances = np.maximum((square_sums - count * np.square(means)) / (count - 1), 0.0)
    pooled = 0.5 * (variances[0] + variances[1])
    positive = pooled[pooled > 0]
    base_scale = float(np.median(positive)) if positive.size else 1.0
    ridge = max(float(ridge_ratio) * base_scale, 1e-12)
    weight = (means[1] - means[0]) / (pooled + ridge)
    norm = float(np.linalg.norm(weight))
    if norm > 0:
        weight /= norm
    return weight.astype(np.float32), ridge


def paired_pushforward_rows(
    name: str,
    root: Path,
    scales: tuple[float, ...],
    *,
    chunk_size: int,
    ridge_ratio: float,
) -> list[dict[str, object]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    samples = int(manifest["samples"])
    world_size = int(manifest["world_size"])
    with np.load(root / "sample_protocol.npz") as protocol:
        heldout_mask = np.asarray(protocol["test_mask"], dtype=bool)
    train_mask = ~heldout_mask
    rows: list[dict[str, object]] = []
    control = scale_key(1.0)
    for scale in scales:
        if math.isclose(scale, 1.0):
            continue
        print(f"[{name}] paired pushforward scale={scale:g}", flush=True)
        condition = scale_key(scale)
        weight, ridge = _stream_diagonal_probe(
            root,
            condition,
            samples=samples,
            world_size=world_size,
            train_mask=train_mask,
            ridge_ratio=ridge_ratio,
            chunk_size=chunk_size,
        )
        for split, split_mask in (("train", train_mask), ("heldout", heldout_mask)):
            chunks = []
            for rank in range(world_size):
                ids = local_ids_for_rank(samples, rank, world_size)
                local_mask = split_mask[ids]
                raw_control = np.load(
                    root / "latents" / f"{control}_rank{rank:02d}.npy",
                    mmap_mode="r",
                )
                raw_candidate = np.load(
                    root / "latents" / f"{condition}_rank{rank:02d}.npy",
                    mmap_mode="r",
                )
                roundtrip_control = np.load(
                    root / "roundtrip" / f"{control}_rank{rank:02d}.npy",
                    mmap_mode="r",
                )
                roundtrip_candidate = np.load(
                    root / "roundtrip" / f"{condition}_rank{rank:02d}.npy",
                    mmap_mode="r",
                )
                for start in range(0, len(ids), chunk_size):
                    stop = min(start + chunk_size, len(ids))
                    selected = local_mask[start:stop]
                    if not selected.any():
                        continue
                    chunks.append(
                        paired_increment_metrics(
                            raw_control[start:stop][selected],
                            raw_candidate[start:stop][selected],
                            roundtrip_control[start:stop][selected],
                            roundtrip_candidate[start:stop][selected],
                            probe_weight=weight,
                        )
                    )
            rows.append(
                {
                    "run": name,
                    "seed": int(manifest["seed"]),
                    "scale": scale,
                    "split": split,
                    "probe_ridge": ridge,
                    **_merge_paired_chunks(chunks),
                }
            )
    return rows


def bias_cancellation_rows(
    name: str, root: Path, scales: tuple[float, ...]
) -> list[dict[str, object]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    samples = int(manifest["samples"])
    world_size = int(manifest["world_size"])
    with np.load(root / "sample_protocol.npz") as protocol:
        heldout_mask = np.asarray(protocol["test_mask"], dtype=bool)
    source = load_ordered_rank_arrays(
        root / "inception", "source", samples=samples, world_size=world_size
    ).astype(np.float32)
    reconstruction = load_ordered_rank_arrays(
        root / "inception", "real", samples=samples, world_size=world_size
    ).astype(np.float32)
    control = load_ordered_rank_arrays(
        root / "inception", scale_key(1.0), samples=samples, world_size=world_size
    ).astype(np.float32)
    rows = []
    for scale in scales:
        print(f"[{name}] feature-bias scale={scale:g}", flush=True)
        candidate = load_ordered_rank_arrays(
            root / "inception", scale_key(scale), samples=samples, world_size=world_size
        ).astype(np.float32)
        for split, mask in (
            ("train", ~heldout_mask),
            ("heldout", heldout_mask),
            ("all", np.ones(samples, dtype=bool)),
        ):
            rows.append(
                {
                    "run": name,
                    "seed": int(manifest["seed"]),
                    "scale": scale,
                    "split": split,
                    **feature_bias_metrics(
                        source[mask],
                        reconstruction[mask],
                        candidate[mask],
                        control=control[mask],
                    ),
                }
            )
    return rows


def _load_low_frequency_condition(
    root: Path,
    condition: str,
    *,
    samples: int,
    world_size: int,
    grid_size: int,
    chunk_size: int,
) -> np.ndarray:
    parts = []
    ids = []
    for rank in range(world_size):
        rank_ids = local_ids_for_rank(samples, rank, world_size)
        images = np.load(
            root / "decoded" / f"{condition}_rank{rank:02d}.npy", mmap_mode="r"
        )
        features = []
        for start in range(0, len(rank_ids), chunk_size):
            features.append(
                low_frequency_features(images[start : start + chunk_size], grid_size)
            )
        parts.append(np.concatenate(features, axis=0))
        ids.append(rank_ids)
    joined_ids = np.concatenate(ids)
    order = np.argsort(joined_ids)
    return np.concatenate(parts, axis=0)[order]


def low_frequency_probe_rows(
    name: str,
    root: Path,
    scales: tuple[float, ...],
    *,
    grid_size: int,
    chunk_size: int,
    ridge_ratio: float,
) -> list[dict[str, object]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    samples = int(manifest["samples"])
    world_size = int(manifest["world_size"])
    with np.load(root / "sample_protocol.npz") as protocol:
        heldout_mask = np.asarray(protocol["test_mask"], dtype=bool)
    train_mask = ~heldout_mask
    reconstruction = _load_low_frequency_condition(
        root,
        "real",
        samples=samples,
        world_size=world_size,
        grid_size=grid_size,
        chunk_size=chunk_size,
    )
    rows = []
    for scale in scales:
        print(f"[{name}] low-frequency probe scale={scale:g}", flush=True)
        candidate = _load_low_frequency_condition(
            root,
            scale_key(scale),
            samples=samples,
            world_size=world_size,
            grid_size=grid_size,
            chunk_size=chunk_size,
        )
        weight, intercept, ridge = fit_feature_probe(
            reconstruction, candidate, train_mask, ridge_ratio
        )
        reference_scores = feature_probe_scores(
            reconstruction[heldout_mask], weight, intercept
        )
        candidate_scores = feature_probe_scores(candidate[heldout_mask], weight, intercept)
        labels = np.concatenate(
            [
                np.zeros(reference_scores.size, dtype=np.int8),
                np.ones(candidate_scores.size, dtype=np.int8),
            ]
        )
        auc = float(
            roc_auc_score(labels, np.concatenate([reference_scores, candidate_scores]))
        )
        rows.append(
            {
                "run": name,
                "seed": int(manifest["seed"]),
                "scale": scale,
                "grid_size": grid_size,
                "feature_dim": int(reconstruction.shape[1]),
                "heldout_samples": int(heldout_mask.sum()),
                "auc": auc,
                "auc_separability": 0.5 + abs(auc - 0.5),
                "ridge": ridge,
            }
        )
    return rows


def plot_results(
    paired: pd.DataFrame,
    bias: pd.DataFrame,
    low_frequency: pd.DataFrame,
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))
    heldout_paired = paired[paired["split"] == "heldout"]
    heldout_bias = bias[bias["split"] == "heldout"]
    for name, frame in heldout_paired.groupby("run", sort=False):
        frame = frame.sort_values("scale")
        axes[0].plot(
            frame["scale"],
            frame["aggregate_roundtrip_over_raw_norm"],
            "o-",
            label=name,
        )
    axes[0].axhline(1.0, color="#333333", linestyle="--", linewidth=1)
    axes[0].set_title("Paired IG increment after E(D(.))")
    axes[0].set_ylabel("roundtrip / raw increment norm")

    for name, frame in heldout_bias.groupby("run", sort=False):
        frame = frame.sort_values("scale")
        line = axes[1].plot(
            frame["scale"],
            frame["diag_white_ig_vs_reconstruction_bias_cosine"],
            "o-",
            label=f"{name}: reconstruction bias",
        )[0]
        axes[1].plot(
            frame["scale"],
            frame["diag_white_ig_vs_full_error_cosine"],
            "s--",
            color=line.get_color(),
            label=f"{name}: full decoded error",
        )
    axes[1].axhline(0.0, color="#333333", linestyle="--", linewidth=1)
    axes[1].set_title("IG shift alignment in decoded features")
    axes[1].set_ylabel("diagonally whitened cosine")

    for name, frame in low_frequency.groupby("run", sort=False):
        frame = frame.sort_values("scale")
        axes[2].plot(
            frame["scale"], frame["auc_separability"], "o-", label=name
        )
    axes[2].axhline(0.5, color="#333333", linestyle="--", linewidth=1)
    axes[2].set_title("Low-frequency decoded-image C2ST")
    axes[2].set_ylabel("held-out separability (0.5 is best)")

    for axis in axes:
        axis.axvline(1.0, color="#222222", linestyle=":", alpha=0.6)
        axis.axvline(1.78, color="#7c4aa5", linestyle=":", alpha=0.7)
        axis.set_xlabel("guidance scale s")
        axis.grid(True, alpha=0.2)
        axis.legend(frameon=False)
    fig.suptitle("RAEv2 Decoder Pushforward Mechanism Checks")
    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=parse_named_run, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--pixel-grid", type=int, default=16)
    parser.add_argument("--ridge-ratio", type=float, default=1e-4)
    args = parser.parse_args()
    if args.chunk_size <= 0 or args.pixel_grid <= 0 or args.ridge_ratio <= 0:
        raise ValueError("chunk size, pixel grid, and ridge ratio must be positive")
    if len({name for name, _ in args.run}) != len(args.run):
        raise ValueError("run names must be unique")

    paired_rows = []
    bias_rows = []
    low_frequency_rows = []
    run_metadata = {}
    expected_scales = None
    for name, root in args.run:
        print(f"[{name}] validating {root}", flush=True)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise ValueError(f"run {name!r} is not complete")
        scales = tuple(float(value) for value in manifest["scales"])
        if expected_scales is None:
            expected_scales = scales
        elif scales != expected_scales:
            raise ValueError("runs must use identical scale grids")
        run_metadata[name] = {
            "path": str(root),
            "seed": int(manifest["seed"]),
            "samples": int(manifest["samples"]),
        }
        paired_rows.extend(
            paired_pushforward_rows(
                name,
                root,
                scales,
                chunk_size=args.chunk_size,
                ridge_ratio=args.ridge_ratio,
            )
        )
        bias_rows.extend(bias_cancellation_rows(name, root, scales))
        low_frequency_rows.extend(
            low_frequency_probe_rows(
                name,
                root,
                scales,
                grid_size=args.pixel_grid,
                chunk_size=args.chunk_size,
                ridge_ratio=args.ridge_ratio,
            )
        )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paired = pd.DataFrame(paired_rows)
    bias = pd.DataFrame(bias_rows)
    low_frequency = pd.DataFrame(low_frequency_rows)
    paired.to_csv(output_dir / "paired_pushforward_metrics.csv", index=False)
    bias.to_csv(output_dir / "feature_bias_cancellation.csv", index=False)
    low_frequency.to_csv(output_dir / "low_frequency_probe.csv", index=False)
    plot_results(paired, bias, low_frequency, output_dir / "mechanism_curves.png")

    heldout_paired = paired[paired["split"] == "heldout"]
    heldout_bias = bias[bias["split"] == "heldout"]
    selected_scale = 1.78
    selected_paired = heldout_paired[np.isclose(heldout_paired["scale"], selected_scale)]
    selected_bias = heldout_bias[np.isclose(heldout_bias["scale"], selected_scale)]
    summary = {
        "protocol": PROTOCOL,
        "runs": run_metadata,
        "scale_grid": list(expected_scales or ()),
        "selected_scale": selected_scale,
        "selected_scale_all_seeds_contract": bool(
            (selected_paired["aggregate_roundtrip_over_raw_norm"] < 1.0).all()
        ),
        "selected_scale_all_seeds_bias_opposing": bool(
            (
                selected_bias["diag_white_ig_vs_reconstruction_bias_cosine"]
                < 0.0
            ).all()
        ),
        "selected_scale_all_seeds_full_error_opposing": bool(
            (selected_bias["diag_white_ig_vs_full_error_cosine"] < 0.0).all()
        ),
        "low_frequency_best_scale_by_run": {
            name: float(frame.loc[frame["auc_separability"].idxmin(), "scale"])
            for name, frame in low_frequency.groupby("run", sort=False)
        },
        "interpretation_guardrail": (
            "Contraction and negative feature-bias alignment support, but do not by "
            "themselves prove, a decoder-mediated ranking reversal. A fixed-latent "
            "decoder intervention remains the causal test."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
