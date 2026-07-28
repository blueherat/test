"""Aggregate preregistered Imagenette responsibility runs and evaluate gates."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.imagenette_noise_responsibility import (  # noqa: E402
    CAPACITIES,
    FREQUENCY_BANDS,
    noise_region,
)


DEFAULT_ROOT = Path.home() / "data/eqvae/imagenette_noise_responsibility_formal"


def discover_complete_runs(root: Path) -> list[Path]:
    runs = sorted(path.parent for path in root.glob("d*_seed*/summary.json"))
    required = {
        "summary.json",
        "config.json",
        "history.csv",
        "responsibility_paired.csv",
        "responsibility_profile.csv",
        "frequency_profile.csv",
        "curve_summary.csv",
        "identity_controls.csv",
        "state.pt",
    }
    return [path for path in runs if required <= {item.name for item in path.iterdir()}]


def load_tables(root: Path) -> dict[str, pd.DataFrame]:
    summary_rows: list[dict] = []
    paired_parts: list[pd.DataFrame] = []
    profile_parts: list[pd.DataFrame] = []
    frequency_parts: list[pd.DataFrame] = []
    curve_parts: list[pd.DataFrame] = []
    for run in discover_complete_runs(root):
        summary = json.loads((run / "summary.json").read_text())
        latent_dim = int(summary["latent_dim"])
        seed = int(summary["seed"])
        summary_rows.append({"run": str(run), **summary})
        for filename, destination in (
            ("responsibility_paired.csv", paired_parts),
            ("responsibility_profile.csv", profile_parts),
            ("frequency_profile.csv", frequency_parts),
            ("curve_summary.csv", curve_parts),
        ):
            frame = pd.read_csv(run / filename)
            frame.insert(0, "seed", seed)
            frame.insert(0, "latent_dim", latent_dim)
            destination.append(frame)
    if not summary_rows:
        raise FileNotFoundError(f"no complete runs under {root}")
    return {
        "summary": pd.DataFrame(summary_rows).sort_values(["seed", "latent_dim"]),
        "paired": pd.concat(paired_parts, ignore_index=True),
        "profile": pd.concat(profile_parts, ignore_index=True),
        "frequency_profile": pd.concat(frequency_parts, ignore_index=True),
        "curves": pd.concat(curve_parts, ignore_index=True),
    }


def paired_region_statistics(paired: pd.DataFrame) -> pd.DataFrame:
    table = paired.copy()
    table["region"] = table.time.map(noise_region)
    metrics = ("delta_shuffle", "delta_null", "delta_within_class")
    per_sample = (
        table.groupby(["latent_dim", "seed", "region", "sample_index"], as_index=False)[list(metrics)]
        .mean()
    )
    records: list[dict[str, float | int | str]] = []
    for keys, frame in per_sample.groupby(["latent_dim", "seed", "region"], sort=True):
        latent_dim, seed, region = keys
        record: dict[str, float | int | str] = {
            "latent_dim": int(latent_dim),
            "seed": int(seed),
            "region": str(region),
            "count": len(frame),
        }
        for metric in metrics:
            values = frame[metric].to_numpy(dtype=np.float64)
            mean = float(values.mean())
            sem = float(values.std(ddof=1) / math.sqrt(len(values)))
            record[f"{metric}_mean"] = mean
            record[f"{metric}_median"] = float(np.median(values))
            record[f"{metric}_ci95_low"] = mean - 1.96 * sem
            record[f"{metric}_ci95_high"] = mean + 1.96 * sem
            record[f"{metric}_positive_rate"] = float(np.mean(values > 0.0))
        records.append(record)
    return pd.DataFrame.from_records(records)


def total_shuffle_curve_features(curves: pd.DataFrame) -> pd.DataFrame:
    total = curves[
        (curves.source == "total")
        & (curves.band == "all")
        & (curves.metric == "delta_shuffle")
    ][
        [
            "latent_dim",
            "seed",
            "low_noise_fraction",
            "mid_noise_fraction",
            "high_noise_fraction",
        ]
    ].copy()
    frequency = curves[
        (curves.source == "frequency") & (curves.metric == "delta_shuffle")
    ][["latent_dim", "seed", "band", "positive_region_sum"]].copy()
    frequency = frequency.pivot(index=["latent_dim", "seed"], columns="band", values="positive_region_sum")
    denominator = frequency.sum(axis=1).replace(0.0, np.nan)
    frequency["high_frequency_fraction"] = frequency["high"] / denominator
    return total.merge(
        frequency[["high_frequency_fraction"]].reset_index(),
        on=["latent_dim", "seed"],
        validate="one_to_one",
    )


def capacity_permutation_pvalue(features: pd.DataFrame) -> tuple[float, float]:
    complete = features.pivot(index="seed", columns="latent_dim", values="high_noise_fraction")
    complete = complete.dropna(subset=list(CAPACITIES))
    if len(complete) < 2:
        return float("nan"), float("nan")
    x = np.log2(np.asarray(CAPACITIES, dtype=np.float64))

    def slope(values: np.ndarray) -> float:
        centered_x = x - x.mean()
        return float(np.mean((values - values.mean(axis=1, keepdims=True)) * centered_x[None, :]))

    observed_values = complete.loc[:, list(CAPACITIES)].to_numpy()
    observed = slope(observed_values)
    permutations = list(itertools.permutations(range(len(CAPACITIES))))
    null_values = []
    for choices in itertools.product(permutations, repeat=len(complete)):
        permuted = np.stack([observed_values[row, list(order)] for row, order in enumerate(choices)])
        null_values.append(slope(permuted))
    # The preregistered direction is decreasing high-noise fraction with capacity.
    pvalue = (1.0 + sum(value <= observed for value in null_values)) / (1.0 + len(null_values))
    return observed, float(pvalue)


def _ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    train = (train_x - mean) / std
    test = (test_x - mean) / std
    train = np.concatenate([np.ones((len(train), 1)), train], axis=1)
    test = np.concatenate([np.ones((len(test), 1)), test], axis=1)
    penalty = np.eye(train.shape[1]) * float(alpha)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(train.T @ train + penalty, train.T @ train_y)
    return test @ coefficients


def leave_one_seed_out_quality(
    summary: pd.DataFrame,
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    table = summary.merge(features, on=["latent_dim", "seed"], validate="one_to_one")
    feature_sets = {
        "responsibility_curve": [
            "mid_noise_fraction",
            "high_noise_fraction",
            "high_frequency_fraction",
        ],
        "validation_velocity_mse": ["final_validation_velocity_mse"],
        "source_pixel_mse": ["conditional_source_pixel_mse"],
    }
    rows: list[dict[str, float | int | str]] = []
    seeds = sorted(table.seed.unique())
    for held_out_seed in seeds:
        train = table[table.seed != held_out_seed]
        test = table[table.seed == held_out_seed]
        if len(train) < 2 or test.empty:
            continue
        for name, columns in feature_sets.items():
            prediction = _ridge_predict(
                train[columns].to_numpy(dtype=np.float64),
                train.conditional_feature_fid.to_numpy(dtype=np.float64),
                test[columns].to_numpy(dtype=np.float64),
            )
            for (_, item), predicted in zip(test.iterrows(), prediction):
                rows.append(
                    {
                        "held_out_seed": int(held_out_seed),
                        "latent_dim": int(item.latent_dim),
                        "predictor": name,
                        "observed_fid": float(item.conditional_feature_fid),
                        "predicted_fid": float(predicted),
                        "squared_error": float((predicted - item.conditional_feature_fid) ** 2),
                    }
                )
    predictions = pd.DataFrame.from_records(rows)
    if predictions.empty:
        return predictions, {name: float("nan") for name in feature_sets}
    rmse = {
        name: float(np.sqrt(frame.squared_error.mean()))
        for name, frame in predictions.groupby("predictor")
    }
    return predictions, rmse


def evaluate_gates(tables: dict[str, pd.DataFrame]) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    summary = tables["summary"]
    region = paired_region_statistics(tables["paired"])
    features = total_shuffle_curve_features(tables["curves"])
    expected = {(capacity, seed) for capacity in CAPACITIES for seed in sorted(summary.seed.unique())}
    present = set(zip(summary.latent_dim.astype(int), summary.seed.astype(int)))
    complete_grid = expected <= present and len(summary.seed.unique()) >= 3

    p1_by_seed: dict[int, bool] = {}
    for seed, seed_frame in region[region.region.isin(["mid_noise", "high_noise"])].groupby("seed"):
        pass_count = 0
        for latent_dim, capacity_frame in seed_frame.groupby("latent_dim"):
            if (
                (capacity_frame.delta_shuffle_ci95_low > 0.0).all()
                and (capacity_frame.delta_shuffle_positive_rate > 0.6).all()
            ):
                pass_count += 1
        p1_by_seed[int(seed)] = pass_count >= 2
    p1 = complete_grid and all(p1_by_seed.values()) and len(p1_by_seed) >= 3

    pivot = features.pivot(index="seed", columns="latent_dim", values="high_noise_fraction")
    capacity_differences = (
        pivot.get(16, pd.Series(dtype=float)) - pivot.get(256, pd.Series(dtype=float))
    ).dropna()
    p2_threshold = len(capacity_differences) >= 3 and bool((capacity_differences >= 0.10).all())
    slope, permutation_p = capacity_permutation_pvalue(features)
    p2 = complete_grid and (p2_threshold or (np.isfinite(permutation_p) and permutation_p < 0.05))

    within_256 = region[region.latent_dim == 256]
    p3_by_seed = {
        int(seed): int((frame.delta_within_class_mean > 0.0).sum()) >= 2
        for seed, frame in within_256.groupby("seed")
    }
    p3 = complete_grid and len(p3_by_seed) >= 3 and all(p3_by_seed.values())

    embedding_spread = (
        summary.groupby("latent_dim").condition_embedding_rms_mean.mean().max()
        / summary.groupby("latent_dim").condition_embedding_rms_mean.mean().min()
        - 1.0
    )
    p4 = complete_grid and float(embedding_spread) < 0.02 and p2

    predictions, rmse = leave_one_seed_out_quality(summary, features)
    responsibility_rmse = rmse.get("responsibility_curve", float("nan"))
    p5 = (
        complete_grid
        and np.isfinite(responsibility_rmse)
        and responsibility_rmse < rmse.get("validation_velocity_mse", float("inf"))
        and responsibility_rmse < rmse.get("source_pixel_mse", float("inf"))
    )

    identity_ok = bool((summary.identity_absolute_rms_max <= 1e-7).all())
    dropout_ok = bool(summary.observed_dropout_rate.between(0.08, 0.12).all())
    finite_ok = bool(
        np.isfinite(
            summary[
                [
                    "final_validation_velocity_mse",
                    "conditional_feature_fid",
                    "conditional_source_pixel_mse",
                    "condition_embedding_rms_mean",
                ]
            ].to_numpy(dtype=np.float64)
        ).all()
    )
    stream_hash_ok = all(
        len(frame.stream_hash_first_32_batches.unique()) == 1
        for _, frame in summary.groupby("seed")
    )
    decoder_initialization_ok = all(
        len(frame.decoder_shared_initial_sha256.unique()) == 1
        for _, frame in summary.groupby("seed")
    )
    encoder_initialization_ok = all(
        len(frame.encoder_shared_initial_sha256.unique()) == 1
        for _, frame in summary.groupby("seed")
    )
    implementation_audit = (
        identity_ok
        and dropout_ok
        and finite_ok
        and stream_hash_ok
        and decoder_initialization_ok
        and encoder_initialization_ok
    )
    gates = {
        "complete_grid": bool(complete_grid),
        "implementation_audit": bool(implementation_audit),
        "identity_ok": identity_ok,
        "dropout_ok": dropout_ok,
        "finite_ok": finite_ok,
        "stream_hash_ok": stream_hash_ok,
        "decoder_initialization_ok": decoder_initialization_ok,
        "encoder_initialization_ok": encoder_initialization_ok,
        "p1_sample_responsibility": bool(p1),
        "p1_by_seed": p1_by_seed,
        "p2_capacity_curve_shape": bool(p2),
        "p2_threshold": bool(p2_threshold),
        "capacity_high_noise_fraction_d16_minus_d256": {
            str(index): float(value) for index, value in capacity_differences.items()
        },
        "capacity_slope": float(slope),
        "capacity_permutation_p_one_sided": float(permutation_p),
        "p3_within_class_sample_information": bool(p3),
        "p3_by_seed": p3_by_seed,
        "p4_scale_control": bool(p4),
        "condition_embedding_relative_spread": float(embedding_spread),
        "p5_quality_prediction": bool(p5),
        "quality_loso_rmse": rmse,
        "all_prior_gates_pass": bool(implementation_audit and p1 and p2 and p3 and p4 and p5),
    }
    return gates, region, predictions


def plot_profiles(profile: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    colors = {16: "#2274A5", 64: "#E07A1F", 256: "#3A9D5D"}
    for latent_dim in CAPACITIES:
        frame = profile[profile.latent_dim == latent_dim]
        grouped = frame.groupby("time").delta_shuffle_mean
        mean = grouped.mean().sort_index()
        std = grouped.std().reindex(mean.index).fillna(0.0)
        axes[0].plot(mean.index, mean.values, marker="o", color=colors[latent_dim], label=f"{latent_dim}d")
        axes[0].fill_between(mean.index, mean - std, mean + std, color=colors[latent_dim], alpha=0.18)
        normalized_parts = []
        for _, seed_frame in frame.groupby("seed"):
            seed_frame = seed_frame.sort_values("time")
            values = seed_frame.delta_shuffle_mean.clip(lower=0.0)
            denominator = float(values.sum())
            normalized = values / denominator if denominator > 0 else values * np.nan
            normalized_parts.append(pd.Series(normalized.to_numpy(), index=seed_frame.time.to_numpy()))
        normalized_table = pd.concat(normalized_parts, axis=1)
        axes[1].plot(
            normalized_table.index,
            normalized_table.mean(axis=1),
            marker="o",
            color=colors[latent_dim],
            label=f"{latent_dim}d",
        )
        axes[1].fill_between(
            normalized_table.index,
            normalized_table.mean(axis=1) - normalized_table.std(axis=1).fillna(0.0),
            normalized_table.mean(axis=1) + normalized_table.std(axis=1).fillna(0.0),
            color=colors[latent_dim],
            alpha=0.18,
        )
    for axis, title, ylabel in (
        (axes[0], "Paired sample responsibility", "Delta shuffle velocity MSE"),
        (axes[1], "Shape normalized within each run", "Positive responsibility fraction"),
    ):
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.axvline(1 / 3, color="gray", linestyle="--", linewidth=0.8)
        axis.axvline(2 / 3, color="gray", linestyle="--", linewidth=0.8)
        axis.set_title(title)
        axis.set_xlabel("t (1 = high noise)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
        axis.legend()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_frequency_profiles(frequency: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.2), sharex=True, constrained_layout=True)
    colors = {16: "#2274A5", 64: "#E07A1F", 256: "#3A9D5D"}
    for axis, band in zip(axes, FREQUENCY_BANDS):
        table = frequency[frequency.band == band]
        for latent_dim in CAPACITIES:
            frame = table[table.latent_dim == latent_dim]
            grouped = frame.groupby("time").delta_shuffle_mean
            mean = grouped.mean().sort_index()
            std = grouped.std().reindex(mean.index).fillna(0.0)
            axis.plot(mean.index, mean.values, marker="o", color=colors[latent_dim], label=f"{latent_dim}d")
            axis.fill_between(mean.index, mean - std, mean + std, color=colors[latent_dim], alpha=0.18)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.axvline(1 / 3, color="gray", linestyle="--", linewidth=0.8)
        axis.axvline(2 / 3, color="gray", linestyle="--", linewidth=0.8)
        axis.set_title(f"{band} radial frequency")
        axis.set_xlabel("t (1 = high noise)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Delta shuffle band MSE")
    axes[-1].legend()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def run(root: Path, output: Path | None = None) -> dict:
    tables = load_tables(root)
    output = root / "comparison" if output is None else output
    output.mkdir(parents=True, exist_ok=True)
    gates, region, predictions = evaluate_gates(tables)
    tables["summary"].to_csv(output / "run_summaries.csv", index=False)
    tables["profile"].to_csv(output / "responsibility_profiles.csv", index=False)
    tables["frequency_profile"].to_csv(output / "frequency_profiles.csv", index=False)
    tables["curves"].to_csv(output / "curve_features.csv", index=False)
    region.to_csv(output / "region_statistics.csv", index=False)
    predictions.to_csv(output / "quality_loso_predictions.csv", index=False)
    (output / "gates.json").write_text(
        json.dumps(gates, indent=2, ensure_ascii=False, allow_nan=True) + "\n"
    )
    plot_profiles(tables["profile"], output / "responsibility_profiles.png")
    plot_frequency_profiles(tables["frequency_profile"], output / "frequency_profiles.png")
    print(json.dumps(gates, indent=2, ensure_ascii=False, allow_nan=True))
    return gates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    return run(args.root, args.output)


if __name__ == "__main__":
    main()
