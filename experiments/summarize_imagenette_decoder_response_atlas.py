"""Summarize held-out prediction and preregistered gates for response atlases."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


CAPACITIES = (16, 64, 256)
SEEDS = (0, 1, 2, 3, 4)
LAYER_ORDER = (
    "condition",
    "down0",
    "down1",
    "down2",
    "middle",
    "up2",
    "up1",
    "up0",
    "velocity",
)
PRIMARY_LAYERS = LAYER_ORDER[1:]
DEFAULT_ROOT = Path.home() / "data/eqvae/imagenette_latent_prior_tradeoff"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def load_atlases(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_rows = []
    response_rows = []
    paired_rows = []
    for latent_dim in CAPACITIES:
        for seed in SEEDS:
            path = root / f"d{latent_dim}_seed{seed}_p0/decoder_response_atlas.json"
            if not path.is_file():
                continue
            payload = json.loads(path.read_text())
            common = {
                "latent_dim": int(payload["latent_dim"]),
                "frozen_seed": int(payload["frozen_seed"]),
                "modeling_gap": float(payload["modeling_gap"]),
                "count": int(payload["count"]),
                "paired_count": int(payload["paired_count"]),
                "pixel_steps": int(payload["pixel_steps"]),
                "projection_dim": int(payload["projection_dim"]),
                "projection_seed": int(payload["projection_seed"]),
                "frozen_decoder_matches_formal": bool(
                    payload["frozen_decoder_matches_formal"]
                ),
                "run": str(path.parent),
            }
            run_rows.append(common)
            response_rows.extend({**common, **row} for row in payload["response_rows"])
            paired_rows.extend({**common, **row} for row in payload["paired_rows"])
    return pd.DataFrame(run_rows), pd.DataFrame(response_rows), pd.DataFrame(paired_rows)


def aggregate_times(response: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        column
        for column in response.columns
        if column
        not in {
            "latent_dim",
            "frozen_seed",
            "modeling_gap",
            "count",
            "paired_count",
            "pixel_steps",
            "projection_dim",
            "projection_seed",
            "frozen_decoder_matches_formal",
            "run",
            "representation",
            "time",
            "layer",
            "feature_dim",
        }
        and pd.api.types.is_numeric_dtype(response[column])
    ]
    group_columns = [
        "latent_dim",
        "frozen_seed",
        "modeling_gap",
        "representation",
        "layer",
    ]
    return (
        response.groupby(group_columns, as_index=False)[metric_columns]
        .mean()
        .sort_values(["representation", "layer", "frozen_seed", "latent_dim"])
        .reset_index(drop=True)
    )


def _ridge_prediction(
    train: pd.DataFrame,
    test: pd.DataFrame,
    predictor: str,
    target: str,
) -> np.ndarray:
    x_train = train[predictor].to_numpy(dtype=np.float64)
    y_train = train[target].to_numpy(dtype=np.float64)
    mean = float(x_train.mean())
    scale = float(x_train.std())
    scale = scale if scale > 1e-12 else 1.0
    design = np.column_stack([np.ones(len(train)), (x_train - mean) / scale])
    coefficient = np.linalg.solve(
        design.T @ design + np.diag([0.0, 1e-6]), design.T @ y_train
    )
    return coefficient[0] + coefficient[1] * (
        test[predictor].to_numpy(dtype=np.float64) - mean
    ) / scale


def heldout_predictions(
    table: pd.DataFrame,
    *,
    predictor: str,
    group: str,
    target: str = "modeling_gap",
) -> tuple[pd.DataFrame, float, float]:
    rows = []
    if table[group].nunique() < 2:
        return pd.DataFrame(), float("nan"), float("nan")
    for heldout in sorted(table[group].unique()):
        train = table[table[group] != heldout]
        test = table[table[group] == heldout]
        if train.empty or test.empty:
            continue
        prediction = _ridge_prediction(train, test, predictor, target)
        for item, predicted in zip(test.itertuples(), prediction):
            rows.append(
                {
                    "heldout_group": str(heldout),
                    "latent_dim": int(item.latent_dim),
                    "frozen_seed": int(item.frozen_seed),
                    "observed": float(getattr(item, target)),
                    "predicted": float(predicted),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, float("nan"), float("nan")
    correlation = spearmanr(frame.observed, frame.predicted).statistic
    correlation = 0.0 if not math.isfinite(float(correlation)) else float(correlation)
    rmse = float(np.sqrt(np.mean((frame.observed - frame.predicted) ** 2)))
    return frame, correlation, rmse


def prediction_summary(aggregated: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_names = (
        "normalized_frechet",
        "covariance_relative_error",
        "normalized_swd",
        "linear_c2st_auc",
    )
    summaries = []
    predictions = []
    for (representation, layer), table in aggregated.groupby(
        ["representation", "layer"], sort=False
    ):
        for metric in metric_names:
            for protocol, group in (
                ("leave_seed_out", "frozen_seed"),
                ("leave_dimension_out", "latent_dim"),
            ):
                frame, correlation, rmse = heldout_predictions(
                    table, predictor=metric, group=group
                )
                summaries.append(
                    {
                        "representation": representation,
                        "layer": layer,
                        "metric": metric,
                        "protocol": protocol,
                        "spearman": correlation,
                        "rmse": rmse,
                    }
                )
                if not frame.empty:
                    frame.insert(0, "protocol", protocol)
                    frame.insert(0, "metric", metric)
                    frame.insert(0, "layer", layer)
                    frame.insert(0, "representation", representation)
                    predictions.append(frame)
    heldout = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    return pd.DataFrame(summaries), heldout


def nominal_capacity_baseline(run_table: pd.DataFrame) -> list[dict[str, float | str]]:
    rows = []
    for protocol, group in (
        ("leave_seed_out", "frozen_seed"),
        ("leave_dimension_out", "latent_dim"),
    ):
        _prediction, correlation, rmse = heldout_predictions(
            run_table, predictor="latent_dim", group=group
        )
        rows.append(
            {
                "protocol": protocol,
                "predictor": "latent_dim",
                "spearman": correlation,
                "rmse": rmse,
            }
        )
    return rows


def evaluate_gates(
    runs: pd.DataFrame,
    response: pd.DataFrame,
    paired: pd.DataFrame,
    aggregated: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    expected_count: int = 256,
) -> tuple[dict, pd.DataFrame]:
    expected = {(latent_dim, seed) for latent_dim in CAPACITIES for seed in SEEDS}
    present = set(zip(runs.latent_dim.astype(int), runs.frozen_seed.astype(int)))
    complete = present == expected
    finite = bool(
        not response.empty
        and np.isfinite(response.select_dtypes(include=[np.number]).to_numpy()).all()
        and np.isfinite(paired.select_dtypes(include=[np.number]).to_numpy()).all()
    )
    consistent_settings = bool(
        complete
        and runs["count"].nunique() == 1
        and int(runs["count"].iloc[0]) == int(expected_count)
        and runs["paired_count"].nunique() == 1
        and int(runs["paired_count"].iloc[0]) == 128
        and runs["pixel_steps"].eq(50).all()
        and runs["projection_dim"].eq(128).all()
        and runs["projection_seed"].eq(48_271).all()
    )
    implementation = bool(
        complete
        and finite
        and consistent_settings
        and runs.frozen_decoder_matches_formal.all()
        and len(response) == 15 * 2 * 3 * 9
        and len(paired) == 15 * 3
    )

    primary = predictions[
        predictions.representation.eq("condition")
        & predictions.metric.eq("normalized_frechet")
        & predictions.layer.isin(PRIMARY_LAYERS)
    ].pivot(index="layer", columns="protocol", values="spearman")
    primary = primary.reindex(PRIMARY_LAYERS)
    primary["both_pass"] = (
        primary["leave_seed_out"].ge(0.60)
        & primary["leave_dimension_out"].ge(0.60)
    )
    adjacent_pairs = []
    for left, right in zip(PRIMARY_LAYERS[:-1], PRIMARY_LAYERS[1:]):
        if bool(primary.loc[left, "both_pass"] and primary.loc[right, "both_pass"]):
            adjacent_pairs.append([left, right])

    selected_layers = sorted({layer for pair in adjacent_pairs for layer in pair})
    primary_aggregated = aggregated[
        aggregated.representation.eq("condition")
        & aggregated.layer.isin(selected_layers)
    ]
    floor_ratio = (
        float(primary_aggregated.frechet_over_real_floor.mean())
        if not primary_aggregated.empty
        else 0.0
    )
    floor_c2st_deviation = float(
        (response.real_real_linear_c2st_auc - 0.5).abs().mean()
    )
    floor_gate = bool(floor_ratio >= 1.5 and floor_c2st_deviation <= 0.10)

    shuffled_by_time = (
        paired.assign(worse=paired.shuffled_over_matched.gt(1.0))
        .groupby("time")
        .agg(
            mean_ratio=("shuffled_over_matched", "mean"),
            worse_count=("worse", "sum"),
        )
        .reset_index()
    )
    required_shuffle = shuffled_by_time[shuffled_by_time.time.isin((0.9, 0.5))]
    shuffled_gate = bool(
        len(shuffled_by_time) == 3
        and len(required_shuffle) == 2
        and required_shuffle.mean_ratio.ge(1.05).all()
        and required_shuffle.worse_count.ge(12).all()
    )
    gate = bool(implementation and adjacent_pairs and floor_gate and shuffled_gate)
    result = {
        "complete_grid": complete,
        "present": sorted([list(item) for item in present]),
        "implementation_audit": implementation,
        "finite_metrics": finite,
        "consistent_settings": consistent_settings,
        "frozen_decoder_matches_all": bool(
            not runs.empty and runs.frozen_decoder_matches_formal.all()
        ),
        "primary_adjacent_pairs": adjacent_pairs,
        "primary_prediction_gate": bool(adjacent_pairs),
        "selected_layer_mean_frechet_over_real_floor": floor_ratio,
        "real_real_c2st_mean_abs_deviation": floor_c2st_deviation,
        "floor_gate": floor_gate,
        "shuffled_gate": shuffled_gate,
        "shuffled_details": shuffled_by_time.to_dict(orient="records"),
        "decoder_response_target_supported": gate,
    }
    return result, primary.reset_index()


def plot_summary(
    aggregated: pd.DataFrame,
    prediction: pd.DataFrame,
    paired: pd.DataFrame,
    output: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    primary = aggregated[
        aggregated.representation.eq("condition")
    ].copy()
    for latent_dim, color in zip(CAPACITIES, ("#4c72b0", "#dd8452", "#c44e52")):
        values = (
            primary[primary.latent_dim.eq(latent_dim)]
            .groupby("layer").normalized_frechet.mean()
            .reindex(LAYER_ORDER)
        )
        axes[0, 0].plot(LAYER_ORDER, values, marker="o", label=f"{latent_dim}d", color=color)
    axes[0, 0].set_title("Conditional response discrepancy")
    axes[0, 0].set_ylabel("normalized Frechet")
    axes[0, 0].tick_params(axis="x", rotation=35)
    axes[0, 0].legend()

    selected = prediction[
        prediction.representation.eq("condition")
        & prediction.metric.eq("normalized_frechet")
        & prediction.layer.isin(PRIMARY_LAYERS)
    ]
    for protocol, marker in (("leave_seed_out", "o"), ("leave_dimension_out", "s")):
        values = selected[selected.protocol.eq(protocol)].set_index("layer").spearman
        axes[0, 1].plot(
            PRIMARY_LAYERS,
            values.reindex(PRIMARY_LAYERS),
            marker=marker,
            label=protocol,
        )
    axes[0, 1].axhline(0.6, color="black", linestyle="--", linewidth=1)
    axes[0, 1].set_ylim(-1.05, 1.05)
    axes[0, 1].set_title("Held-out prediction")
    axes[0, 1].set_ylabel("Spearman")
    axes[0, 1].tick_params(axis="x", rotation=35)
    axes[0, 1].legend()

    floor = primary.groupby("layer").frechet_over_real_floor.mean().reindex(LAYER_ORDER)
    axes[1, 0].bar(LAYER_ORDER, floor, color="#55a868")
    axes[1, 0].axhline(1.5, color="black", linestyle="--", linewidth=1)
    axes[1, 0].set_title("Empirical-prior / real-real floor")
    axes[1, 0].set_ylabel("ratio")
    axes[1, 0].tick_params(axis="x", rotation=35)

    paired_summary = paired.groupby("time").shuffled_over_matched.agg(["mean", "sem"])
    axes[1, 1].errorbar(
        paired_summary.index,
        paired_summary["mean"],
        yerr=paired_summary["sem"],
        marker="o",
        capsize=4,
        color="#8172b3",
    )
    axes[1, 1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1, 1].axhline(1.05, color="gray", linestyle=":", linewidth=1)
    axes[1, 1].set_title("Paired shuffled-latent control")
    axes[1, 1].set_xlabel("decoder time")
    axes[1, 1].set_ylabel("shuffled / matched velocity MSE")
    fig.suptitle("Imagenette-64 frozen decoder response atlas", fontsize=16)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def summarize(
    root: Path,
    *,
    expected_count: int = 256,
    output_name: str = "decoder_response_atlas_summary",
) -> dict:
    output = root / str(output_name)
    output.mkdir(parents=True, exist_ok=True)
    runs, response, paired = load_atlases(root)
    if runs.empty:
        raise FileNotFoundError(f"no decoder response atlases under {root}")
    aggregated = aggregate_times(response)
    prediction, heldout = prediction_summary(aggregated)
    gates, primary = evaluate_gates(
        runs,
        response,
        paired,
        aggregated,
        prediction,
        expected_count=int(expected_count),
    )
    baseline = nominal_capacity_baseline(runs)
    runs.to_csv(output / "runs.csv", index=False)
    response.to_csv(output / "response_rows.csv", index=False)
    paired.to_csv(output / "paired_rows.csv", index=False)
    aggregated.to_csv(output / "time_aggregated.csv", index=False)
    prediction.to_csv(output / "prediction_summary.csv", index=False)
    heldout.to_csv(output / "heldout_predictions.csv", index=False)
    primary.to_csv(output / "primary_layers.csv", index=False)
    _write_json(output / "gates.json", {**gates, "nominal_capacity_baseline": baseline})
    plot_summary(aggregated, prediction, paired, output / "response_atlas.png")
    print(json.dumps(gates, indent=2), flush=True)
    return gates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--expected-count", type=int, default=256)
    parser.add_argument("--output-name", default="decoder_response_atlas_summary")
    args = parser.parse_args()
    summarize(
        args.root,
        expected_count=args.expected_count,
        output_name=args.output_name,
    )


if __name__ == "__main__":
    main()
