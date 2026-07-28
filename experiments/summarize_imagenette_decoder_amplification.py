"""Summarize the preregistered Imagenette decoder-amplification audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CAPACITIES = (16, 64, 256)
SEEDS = (0, 1, 2, 3, 4)
DEFAULT_ROOT = Path.home() / "data/eqvae/imagenette_latent_prior_tradeoff"


def load_audits(root: Path) -> pd.DataFrame:
    rows = []
    for capacity in CAPACITIES:
        for seed in SEEDS:
            path = root / f"d{capacity}_seed{seed}_p0/decoder_amplification_audit.json"
            if path.is_file():
                row = json.loads(path.read_text())
                row["run"] = str(path.parent)
                rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["frozen_seed", "latent_dim"]).reset_index(drop=True)


def single_predictor_loso(
    table: pd.DataFrame,
    predictor: str,
    target: str = "modeling_gap",
) -> tuple[float, pd.DataFrame]:
    records = []
    for heldout_seed in sorted(table.frozen_seed.unique()):
        train = table[table.frozen_seed != heldout_seed]
        test = table[table.frozen_seed == heldout_seed]
        x_train = train[predictor].to_numpy(dtype=np.float64)
        y_train = train[target].to_numpy(dtype=np.float64)
        mean = float(x_train.mean())
        scale = float(x_train.std())
        scale = scale if scale > 1e-12 else 1.0
        design = np.column_stack([np.ones(len(train)), (x_train - mean) / scale])
        coefficient = np.linalg.solve(
            design.T @ design + np.diag([0.0, 1e-6]), design.T @ y_train
        )
        prediction = coefficient[0] + coefficient[1] * (
            test[predictor].to_numpy(dtype=np.float64) - mean
        ) / scale
        for row, predicted in zip(test.itertuples(), prediction):
            records.append(
                {
                    "heldout_seed": int(heldout_seed),
                    "latent_dim": int(row.latent_dim),
                    "predictor": predictor,
                    "observed": float(getattr(row, target)),
                    "predicted": float(predicted),
                    "squared_error": float((predicted - getattr(row, target)) ** 2),
                }
            )
    frame = pd.DataFrame(records)
    return float(np.sqrt(frame.squared_error.mean())), frame


def prediction_table(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictors = (
        "latent_dim",
        "condition_prior_latent_sliced_wasserstein",
        "condition_prior_matched_angle_mean",
        "decoder_weighted_mismatch",
    )
    rows = []
    predictions = []
    for predictor in predictors:
        rmse, frame = single_predictor_loso(table, predictor)
        rows.append({"predictor": predictor, "loso_rmse": rmse})
        predictions.append(frame)
    return (
        pd.DataFrame(rows).sort_values("loso_rmse").reset_index(drop=True),
        pd.concat(predictions, ignore_index=True),
    )


def evaluate_gates(table: pd.DataFrame, prediction: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    expected = {(capacity, seed) for capacity in CAPACITIES for seed in SEEDS}
    present = set(zip(table.latent_dim.astype(int), table.frozen_seed.astype(int)))
    if present != expected:
        return {"complete_grid": False, "present": sorted(present)}, pd.DataFrame()

    feature = table.pivot(
        index="frozen_seed",
        columns="latent_dim",
        values="prior_direction_feature_rms_mean",
    )
    pixel = table.pivot(
        index="frozen_seed",
        columns="latent_dim",
        values="prior_direction_pixel_rms_mean",
    )
    paired = pd.DataFrame(index=SEEDS)
    paired.index.name = "frozen_seed"
    paired["feature_256_over_16"] = feature[256] / feature[16]
    paired["feature_256_gt_16"] = feature[256] > feature[16]
    paired["pixel_256_over_16"] = pixel[256] / pixel[16]
    paired["pixel_256_gt_16"] = pixel[256] > pixel[16]
    alignment_256 = table[table.latent_dim == 256].set_index("frozen_seed")[
        "feature_rms_alignment_ratio"
    ]
    manifold_256 = table[table.latent_dim == 256].set_index("frozen_seed")[
        "feature_rms_manifold_ratio"
    ]
    paired["feature_alignment_ratio_256"] = alignment_256
    paired["feature_alignment_256_gt_1p1"] = alignment_256 > 1.10
    paired["feature_manifold_ratio_256"] = manifold_256
    paired["feature_manifold_256_gt_1p1"] = manifold_256 > 1.10

    velocity_same_direction = False
    velocity_details = {}
    for time in ("0p9", "0p5", "0p1"):
        metric = f"prior_direction_velocity_rms_t{time}_mean"
        pivot = table.pivot(index="frozen_seed", columns="latent_dim", values=metric)
        count = int((pivot[256] > pivot[16]).sum())
        ratio = float(pivot[256].mean() / pivot[16].mean())
        velocity_details[time] = {
            "seed_count_256_gt_16": count,
            "mean_256_over_16": ratio,
        }
        velocity_same_direction |= count >= 4 and ratio > 1.0

    sphere_columns = [
        column
        for column in table.columns
        if column.endswith("condition_abs_mean_max")
        or column.endswith("condition_rms_max_error")
        or column.endswith("fixed_angle_max_error")
    ]
    max_abs_mean = float(
        table[[column for column in sphere_columns if "abs_mean" in column]]
        .to_numpy(dtype=np.float64)
        .max()
    )
    max_rms_error = float(
        table[[column for column in sphere_columns if "rms_max_error" in column]]
        .to_numpy(dtype=np.float64)
        .max()
    )
    max_angle_error = float(
        table[[column for column in sphere_columns if "fixed_angle" in column]]
        .to_numpy(dtype=np.float64)
        .max()
    )
    implementation = bool(
        table.frozen_decoder_matches_formal.all()
        and np.isfinite(table.select_dtypes(include=[np.number]).to_numpy()).all()
        and max_abs_mean <= 2e-5
        and max_rms_error <= 2e-5
        and max_angle_error <= 2e-5
        and table["count"].eq(256).all()
        and np.allclose(table["fixed_angle"], 0.15)
    )
    gate1 = bool(
        paired.feature_256_gt_16.sum() >= 4
        and feature[256].mean() / feature[16].mean() >= 1.20
    )
    gate2 = bool(
        paired.feature_alignment_256_gt_1p1.sum() >= 4
        and paired.feature_manifold_256_gt_1p1.sum() >= 4
    )
    gate3_pixel = bool(
        paired.pixel_256_gt_16.sum() >= 4
        and pixel[256].mean() / pixel[16].mean() > 1.0
    )
    gate3 = bool(gate3_pixel or velocity_same_direction)
    rmse = prediction.set_index("predictor").loso_rmse
    nominal_rmse = float(rmse["latent_dim"])
    decoder_rmse = float(rmse["decoder_weighted_mismatch"])
    gate4 = bool(decoder_rmse <= 0.90 * nominal_rmse)
    mechanism = bool(implementation and gate1 and gate2 and gate3 and gate4)
    gates = {
        "complete_grid": True,
        "implementation_audit": implementation,
        "frozen_decoder_matches_all": bool(table.frozen_decoder_matches_formal.all()),
        "sphere_abs_mean_max": max_abs_mean,
        "sphere_rms_max_error": max_rms_error,
        "fixed_angle_max_error": max_angle_error,
        "gate1_capacity_response": gate1,
        "gate1_seed_count_256_gt_16": int(paired.feature_256_gt_16.sum()),
        "gate1_feature_mean_256_over_16": float(feature[256].mean() / feature[16].mean()),
        "gate2_prior_direction_alignment": gate2,
        "gate2_seed_count_alignment_ratio_256_gt_1p1": int(
            paired.feature_alignment_256_gt_1p1.sum()
        ),
        "gate2_alignment_ratio_256_mean": float(alignment_256.mean()),
        "gate2_seed_count_manifold_ratio_256_gt_1p1": int(
            paired.feature_manifold_256_gt_1p1.sum()
        ),
        "gate2_manifold_ratio_256_mean": float(manifold_256.mean()),
        "gate3_secondary_response": gate3,
        "gate3_pixel_response": gate3_pixel,
        "gate3_velocity_response": velocity_same_direction,
        "velocity_capacity_details": velocity_details,
        "gate4_decoder_weighted_prediction": gate4,
        "decoder_weighted_loso_rmse": decoder_rmse,
        "nominal_dimension_loso_rmse": nominal_rmse,
        "decoder_weighted_relative_rmse": decoder_rmse / nominal_rmse,
        "decoder_amplified_mismatch_supported": mechanism,
    }
    return gates, paired.reset_index()


def _capacity_stats(table: pd.DataFrame, metric: str) -> tuple[np.ndarray, np.ndarray]:
    grouped = table.groupby("latent_dim")[metric]
    mean = grouped.mean().reindex(CAPACITIES).to_numpy(dtype=np.float64)
    sem = grouped.sem().reindex(CAPACITIES).to_numpy(dtype=np.float64)
    return mean, sem


def plot_summary(table: pd.DataFrame, prediction: pd.DataFrame, output: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(19, 10.5), constrained_layout=True)
    x = np.arange(len(CAPACITIES))

    for metric, label, color in (
        ("condition_prior_matched_angle_mean", "prior", "#c44e52"),
        ("condition_empirical_matched_angle_mean", "empirical control", "#4c72b0"),
    ):
        mean, sem = _capacity_stats(table, metric)
        axes[0, 0].errorbar(x, mean, yerr=sem, marker="o", capsize=4, label=label, color=color)
    axes[0, 0].set_title("Matched condition angle")
    axes[0, 0].set_ylabel("radians")
    axes[0, 0].legend()

    for metric, label, color in (
        ("prior_direction_feature_rms_mean", "prior direction", "#c44e52"),
        ("empirical_direction_feature_rms_mean", "empirical direction", "#4c72b0"),
        ("random_direction_feature_rms_mean", "random direction", "#55a868"),
    ):
        mean, sem = _capacity_stats(table, metric)
        axes[0, 1].errorbar(x, mean, yerr=sem, marker="o", capsize=4, label=label, color=color)
    axes[0, 1].set_title("Equal-angle decoded feature response")
    axes[0, 1].set_ylabel("paired ResNet18 feature RMS")
    axes[0, 1].legend()

    for metric, label, color in (
        ("feature_rms_alignment_ratio", "prior / random", "#8172b3"),
        ("feature_rms_manifold_ratio", "prior / empirical", "#dd8452"),
    ):
        mean, sem = _capacity_stats(table, metric)
        axes[0, 2].errorbar(x, mean, yerr=sem, marker="o", capsize=4, label=label, color=color)
    axes[0, 2].axhline(1.0, color="black", linewidth=1, linestyle="--")
    axes[0, 2].axhline(1.1, color="gray", linewidth=1, linestyle=":")
    axes[0, 2].set_title("Mismatch-direction alignment")
    axes[0, 2].set_ylabel("response ratio")
    axes[0, 2].legend()

    colors = {16: "#4c72b0", 64: "#dd8452", 256: "#c44e52"}
    times = np.asarray([0.9, 0.5, 0.1])
    for capacity in CAPACITIES:
        frame = table[table.latent_dim == capacity]
        mean = np.asarray(
            [frame[f"prior_direction_velocity_rms_t{str(time).replace('.', 'p')}_mean"].mean() for time in times]
        )
        sem = np.asarray(
            [frame[f"prior_direction_velocity_rms_t{str(time).replace('.', 'p')}_mean"].sem() for time in times]
        )
        axes[1, 0].errorbar(times, mean, yerr=sem, marker="o", capsize=4, label=f"{capacity}d", color=colors[capacity])
    axes[1, 0].invert_xaxis()
    axes[1, 0].set_title("Velocity response along generation")
    axes[1, 0].set_xlabel("flow time (noise to image)")
    axes[1, 0].set_ylabel("velocity RMS shift")
    axes[1, 0].legend()

    for capacity in CAPACITIES:
        frame = table[table.latent_dim == capacity]
        axes[1, 1].scatter(
            frame.decoder_weighted_mismatch,
            frame.modeling_gap,
            s=60,
            color=colors[capacity],
            label=f"{capacity}d",
        )
    axes[1, 1].set_title("Decoder-weighted mismatch vs decoded gap")
    axes[1, 1].set_xlabel("decoder-weighted mismatch")
    axes[1, 1].set_ylabel("modeling gap (FID)")
    axes[1, 1].legend()

    ordered = prediction.sort_values("loso_rmse")
    labels = [
        {
            "latent_dim": "nominal dim",
            "condition_prior_latent_sliced_wasserstein": "condition SWD",
            "condition_prior_matched_angle_mean": "matched angle",
            "decoder_weighted_mismatch": "decoder weighted",
        }[value]
        for value in ordered.predictor
    ]
    axes[1, 2].barh(labels, ordered.loso_rmse, color="#4c72b0")
    axes[1, 2].invert_yaxis()
    axes[1, 2].set_title("Leave-one-seed-out prediction")
    axes[1, 2].set_xlabel("RMSE for modeling gap (lower is better)")

    for axis in axes.flat:
        if axis is not axes[1, 0]:
            axis.set_xticks(x, [str(value) for value in CAPACITIES]) if axis in axes[0, :] else None
    fig.suptitle("Imagenette-64 frozen decoder amplification diagnostic", fontsize=18)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def summarize(root: Path) -> dict:
    output = root / "comparison_p0"
    output.mkdir(exist_ok=True)
    table = load_audits(root)
    if table.empty:
        raise FileNotFoundError("no decoder amplification audit files found")
    prediction, prediction_rows = prediction_table(table)
    gates, paired = evaluate_gates(table, prediction)
    table.to_csv(output / "decoder_amplification_runs.csv", index=False)
    prediction.to_csv(output / "decoder_amplification_loso.csv", index=False)
    prediction_rows.to_csv(
        output / "decoder_amplification_loso_predictions.csv", index=False
    )
    paired.to_csv(output / "decoder_amplification_paired.csv", index=False)
    (output / "decoder_amplification_gates.json").write_text(
        json.dumps(gates, indent=2, ensure_ascii=False) + "\n"
    )
    if gates.get("complete_grid"):
        plot_summary(
            table, prediction, output / "decoder_amplification_summary.png"
        )
    print(json.dumps(gates, indent=2, ensure_ascii=False))
    return gates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    return summarize(args.root)


if __name__ == "__main__":
    main()
