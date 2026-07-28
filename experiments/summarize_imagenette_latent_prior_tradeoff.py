"""Summarize preregistered Imagenette latent-prior trade-off runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CAPACITIES = (16, 64, 256)
SEEDS = (0, 1, 2, 3, 4)
DEFAULT_ROOT = Path.home() / "data/eqvae/imagenette_latent_prior_tradeoff"


def load_complete_runs(root: Path, prior_replicate: int = 0) -> pd.DataFrame:
    rows = []
    for capacity in CAPACITIES:
        for seed in SEEDS:
            run = root / f"d{capacity}_seed{seed}_p{int(prior_replicate)}"
            summary_path = run / "summary.json"
            if not summary_path.is_file():
                continue
            row = json.loads(summary_path.read_text())
            row["run"] = str(run)
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["frozen_seed", "latent_dim"]).reset_index(drop=True)


def load_auxiliary_runs(
    root: Path,
    filename: str,
    prior_replicate: int = 0,
) -> pd.DataFrame:
    rows = []
    for capacity in CAPACITIES:
        for seed in SEEDS:
            path = root / f"d{capacity}_seed{seed}_p{int(prior_replicate)}" / filename
            if path.is_file():
                rows.append(json.loads(path.read_text()))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["frozen_seed", "latent_dim"]).reset_index(drop=True)


def nfe_audit_summary(audit: pd.DataFrame) -> dict:
    expected = {(capacity, seed) for capacity in CAPACITIES for seed in SEEDS}
    present = set(zip(audit.latent_dim.astype(int), audit.frozen_seed.astype(int)))
    if present != expected:
        return {"complete": False, "present": sorted(present)}
    pivot = audit.pivot(index="frozen_seed", columns="latent_dim", values="audit_nfe_fid")
    regenerated_columns = [
        column for column in audit.columns if column.startswith("regenerated100_")
    ]
    independent = audit[
        (audit.latent_dim == 256)
        & (audit.frozen_seed == 2)
        & audit.get("independent_nfe100_fid_abs_diff", pd.Series(np.nan, index=audit.index)).notna()
    ]
    return {
        "complete": True,
        "formal_fid_from_saved_features_max_abs_diff": float(
            audit.formal_saved_feature_fid_abs_diff.max()
        ),
        "regenerated100_metric_max_abs_diff": float(
            audit[regenerated_columns].to_numpy(dtype=np.float64).max()
        ),
        "nfe200_mean_fid_change_by_capacity": {
            str(int(capacity)): float(frame.audit_nfe_minus_formal_fid.mean())
            for capacity, frame in audit.groupby("latent_dim")
        },
        "nfe200_order_16_better_64_better_256_seed_count": int(
            ((pivot[16] < pivot[64]) & (pivot[64] < pivot[256])).sum()
        ),
        "independent_d256_seed2_nfe100_exact": bool(
            len(independent) == 1
            and float(independent.iloc[0].independent_nfe100_fid_abs_diff) == 0.0
        ),
        "frozen_decoder_matches_all": bool(audit.frozen_decoder_matches_formal.all()),
    }


def _single_predictor_loso(
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
        x_test = (test[predictor].to_numpy(dtype=np.float64) - mean) / scale
        prediction = coefficient[0] + coefficient[1] * x_test
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
    predictions = pd.DataFrame(records)
    return float(np.sqrt(predictions.squared_error.mean())), predictions


def prediction_table(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictors = (
        "latent_dim",
        "source_final_validation_velocity_mse",
        "real_latent_effective_rank",
        "heldout_prior_flow_mse",
    )
    rows = []
    predictions = []
    for predictor in predictors:
        rmse, frame = _single_predictor_loso(table, predictor)
        rows.append({"predictor": predictor, "loso_rmse": rmse})
        predictions.append(frame)
    return pd.DataFrame(rows).sort_values("loso_rmse"), pd.concat(predictions, ignore_index=True)


def evaluate_gates(table: pd.DataFrame, prediction: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    expected = {(capacity, seed) for capacity in CAPACITIES for seed in SEEDS}
    present = set(zip(table.latent_dim.astype(int), table.frozen_seed.astype(int)))
    complete = present == expected
    if not complete:
        return {"complete_grid": False, "present": sorted(present)}, pd.DataFrame()
    pivots = {
        metric: table.pivot(index="frozen_seed", columns="latent_dim", values=metric)
        for metric in (
            "oracle_feature_fid",
            "end_to_end_feature_fid",
            "modeling_gap",
            "total_prior_gap",
            "gaussian_feature_fid",
        )
    }
    paired = pd.DataFrame(index=SEEDS)
    paired.index.name = "frozen_seed"
    paired["oracle_improvement_16_to_256"] = (
        pivots["oracle_feature_fid"][16] - pivots["oracle_feature_fid"][256]
    )
    paired["modeling_gap_increase_16_to_256"] = (
        pivots["modeling_gap"][256] - pivots["modeling_gap"][16]
    )
    paired["total_gap_increase_16_to_256"] = (
        pivots["total_prior_gap"][256] - pivots["total_prior_gap"][16]
    )
    paired["end_to_end_16"] = pivots["end_to_end_feature_fid"][16]
    paired["end_to_end_64"] = pivots["end_to_end_feature_fid"][64]
    paired["end_to_end_256"] = pivots["end_to_end_feature_fid"][256]
    paired["middle_is_best"] = (
        (paired.end_to_end_64 < paired.end_to_end_16)
        & (paired.end_to_end_64 < paired.end_to_end_256)
    )
    paired["monotonic_larger_is_better"] = (
        (paired.end_to_end_256 <= paired.end_to_end_64)
        & (paired.end_to_end_64 <= paired.end_to_end_16)
    )
    decoder_benefit = bool(
        (paired.oracle_improvement_16_to_256 > 0).sum() >= 4
        and paired.oracle_improvement_16_to_256.mean() >= 2.0
    )
    prior_difficulty = bool(
        (paired.modeling_gap_increase_16_to_256 > 0).sum() >= 4
        and paired.modeling_gap_increase_16_to_256.mean() >= 2.0
        and (paired.total_gap_increase_16_to_256 > 0).sum() >= 4
    )
    capacity_means = table.groupby("latent_dim").end_to_end_feature_fid.mean()
    middle_margin = float(min(capacity_means[16], capacity_means[256]) - capacity_means[64])
    middle_optimum = bool(paired.middle_is_best.sum() >= 4 and middle_margin >= 1.0)
    prior_beats_gaussian_by_capacity = {}
    trained_prior_valid = True
    for capacity in CAPACITIES:
        frame = table[table.latent_dim == capacity]
        improvement = frame.gaussian_feature_fid - frame.end_to_end_feature_fid
        passed = bool((improvement > 0).sum() == 5 and improvement.mean() >= 5.0)
        prior_beats_gaussian_by_capacity[str(capacity)] = {
            "all_five": bool((improvement > 0).sum() == 5),
            "mean_fid_improvement": float(improvement.mean()),
            "pass": passed,
        }
        trained_prior_valid &= passed
    finite = bool(
        np.isfinite(
            table.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
        ).all()
    )
    frozen = bool(table.frozen_hashes_unchanged.all())
    roundtrip = bool(table.orthogonal_roundtrip_max_abs.max() <= 1e-6)
    source_oracle_reproduced = bool(
        "source_oracle_fid_1024_abs_diff" in table
        and table.source_oracle_fid_1024_abs_diff.max() <= 1e-4
    )
    data_boundary_shared = bool(
        table.train_path_sha256.nunique() == 1
        and table.val_path_sha256.nunique() == 1
        and (table.train_path_sha256.iloc[0] != table.val_path_sha256.iloc[0])
    )
    init_shared = all(
        frame.prior_initial_sha256.nunique() == 1
        and frame.prior_parameters.nunique() == 1
        and frame.stream_indices_first_32_sha256.nunique() == 1
        and frame.stream_base_noise_first_32_sha256.nunique() == 1
        and frame.stream_time_first_32_sha256.nunique() == 1
        for _, frame in table.groupby("frozen_seed")
    )
    rmse = prediction.set_index("predictor").loso_rmse
    mechanism_rmse = float(
        min(rmse["real_latent_effective_rank"], rmse["heldout_prior_flow_mse"])
    )
    mechanism_prediction = bool(
        mechanism_rmse <= 0.95 * float(rmse["latent_dim"])
        and mechanism_rmse
        <= 0.95 * float(rmse["source_final_validation_velocity_mse"])
    )
    implementation_audit = bool(
        finite
        and frozen
        and roundtrip
        and source_oracle_reproduced
        and data_boundary_shared
        and init_shared
    )
    positive = bool(
        implementation_audit
        and decoder_benefit
        and prior_difficulty
        and middle_optimum
        and trained_prior_valid
        and mechanism_prediction
    )
    monotonic_count = int(paired.monotonic_larger_is_better.sum())
    opposite_candidate = bool(
        implementation_audit
        and decoder_benefit
        and monotonic_count >= 4
        and paired.modeling_gap_increase_16_to_256.mean() < 2.0
    )
    gates = {
        "complete_grid": True,
        "implementation_audit": implementation_audit,
        "finite_metrics": finite,
        "frozen_hashes_unchanged": frozen,
        "orthogonal_roundtrip": roundtrip,
        "source_oracle_1024_reproduced": source_oracle_reproduced,
        "data_boundary_shared_and_disjoint": data_boundary_shared,
        "shared_prior_initialization_and_streams": bool(init_shared),
        "decoder_benefit": decoder_benefit,
        "decoder_benefit_seed_count": int(
            (paired.oracle_improvement_16_to_256 > 0).sum()
        ),
        "decoder_benefit_mean_fid": float(
            paired.oracle_improvement_16_to_256.mean()
        ),
        "prior_difficulty": prior_difficulty,
        "prior_difficulty_seed_count": int(
            (paired.modeling_gap_increase_16_to_256 > 0).sum()
        ),
        "prior_difficulty_mean_modeling_gap_increase": float(
            paired.modeling_gap_increase_16_to_256.mean()
        ),
        "middle_optimum": middle_optimum,
        "middle_optimum_seed_count": int(paired.middle_is_best.sum()),
        "middle_optimum_mean_fid_margin": middle_margin,
        "trained_prior_valid": bool(trained_prior_valid),
        "prior_beats_gaussian_by_capacity": prior_beats_gaussian_by_capacity,
        "mechanism_prediction": mechanism_prediction,
        "mechanism_best_loso_rmse": mechanism_rmse,
        "nominal_dim_loso_rmse": float(rmse["latent_dim"]),
        "decoder_val_loss_loso_rmse": float(
            rmse["source_final_validation_velocity_mse"]
        ),
        "positive_tradeoff_confirmed": positive,
        "opposite_candidate_requires_independent_audit": opposite_candidate,
        "monotonic_larger_is_better_seed_count": monotonic_count,
    }
    return gates, paired.reset_index()


def plot_summary(table: pd.DataFrame, output: Path) -> None:
    colors = {16: "#0072B2", 64: "#E69F00", 256: "#009E73"}
    figure, axes = plt.subplots(2, 3, figsize=(19, 10), constrained_layout=True)
    capacities = np.asarray(CAPACITIES)
    for metric, label in (
        ("oracle_feature_fid", "Oracle"),
        ("empirical_feature_fid", "Empirical"),
        ("end_to_end_feature_fid", "Prior end-to-end"),
        ("gaussian_feature_fid", "Gaussian control"),
    ):
        grouped = table.groupby("latent_dim")[metric]
        axes[0, 0].errorbar(
            capacities,
            grouped.mean().reindex(CAPACITIES),
            yerr=grouped.std().reindex(CAPACITIES),
            marker="o",
            linewidth=2,
            capsize=4,
            label=label,
        )
    axes[0, 0].set_title("Two-stage image quality")
    axes[0, 0].set_ylabel("Feature FID (lower is better)")
    axes[0, 0].set_xscale("log", base=2)
    axes[0, 0].set_xticks(CAPACITIES, [str(value) for value in CAPACITIES])
    axes[0, 0].legend(frameon=False)

    for metric, label in (("total_prior_gap", "Total gap"), ("modeling_gap", "Modeling gap")):
        grouped = table.groupby("latent_dim")[metric]
        axes[0, 1].errorbar(
            capacities,
            grouped.mean().reindex(CAPACITIES),
            yerr=grouped.std().reindex(CAPACITIES),
            marker="o",
            linewidth=2,
            capsize=4,
            label=label,
        )
    axes[0, 1].axhline(0, color="black", linewidth=1)
    axes[0, 1].set_title("Cost of replacing real latent samples")
    axes[0, 1].set_ylabel("FID increment")
    axes[0, 1].set_xscale("log", base=2)
    axes[0, 1].set_xticks(CAPACITIES, [str(value) for value in CAPACITIES])
    axes[0, 1].legend(frameon=False)

    for seed, frame in table.groupby("frozen_seed"):
        axes[0, 2].plot(
            frame.latent_dim,
            frame.end_to_end_feature_fid,
            marker="o",
            alpha=0.8,
            label=f"seed {seed}",
        )
    axes[0, 2].set_title("End-to-end ordering by seed")
    axes[0, 2].set_ylabel("Feature FID")
    axes[0, 2].set_xscale("log", base=2)
    axes[0, 2].set_xticks(CAPACITIES, [str(value) for value in CAPACITIES])
    axes[0, 2].legend(frameon=False, ncol=2)

    for capacity in CAPACITIES:
        frame = table[table.latent_dim == capacity]
        axes[1, 0].scatter(
            frame.real_latent_effective_rank,
            frame.modeling_gap,
            s=55,
            color=colors[capacity],
            label=f"{capacity}d",
        )
    axes[1, 0].set_title("Effective rank vs prior modeling gap")
    axes[1, 0].set_xlabel("Real latent effective rank")
    axes[1, 0].set_ylabel("Modeling gap")
    axes[1, 0].legend(frameon=False)

    for capacity in CAPACITIES:
        frame = table[table.latent_dim == capacity]
        axes[1, 1].scatter(
            frame.heldout_prior_flow_mse,
            frame.modeling_gap,
            s=55,
            color=colors[capacity],
            label=f"{capacity}d",
        )
    axes[1, 1].set_title("Held-out prior loss vs modeling gap")
    axes[1, 1].set_xlabel("Held-out flow MSE")
    axes[1, 1].set_ylabel("Modeling gap")

    grouped = table.groupby("latent_dim")
    width = 0.32
    positions = np.arange(len(CAPACITIES))
    axes[1, 2].bar(
        positions - width / 2,
        grouped.latent_covariance_relative_error.mean().reindex(CAPACITIES),
        width,
        label="Covariance error",
    )
    axes[1, 2].bar(
        positions + width / 2,
        grouped.latent_sliced_wasserstein.mean().reindex(CAPACITIES),
        width,
        label="Sliced Wasserstein",
    )
    axes[1, 2].set_xticks(positions, [f"{value}d" for value in CAPACITIES])
    axes[1, 2].set_title("Latent distribution fit")
    axes[1, 2].legend(frameon=False)
    for axis in axes.flat:
        axis.grid(alpha=0.22)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def summarize(root: Path, output: Path | None = None, prior_replicate: int = 0) -> dict:
    table = load_complete_runs(root, prior_replicate)
    if table.empty:
        raise FileNotFoundError(f"no complete runs under {root}")
    prediction, prediction_rows = prediction_table(table)
    gates, paired = evaluate_gates(table, prediction)
    destination = output or root / f"comparison_p{int(prior_replicate)}"
    destination.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination / "run_summaries.csv", index=False)
    prediction.to_csv(destination / "modeling_gap_loso.csv", index=False)
    prediction_rows.to_csv(destination / "modeling_gap_loso_predictions.csv", index=False)
    paired.to_csv(destination / "paired_capacity_effects.csv", index=False)
    (destination / "gates.json").write_text(
        json.dumps(gates, indent=2, ensure_ascii=False) + "\n"
    )
    if gates.get("complete_grid"):
        plot_summary(table, destination / "tradeoff_summary.png")
    audit = load_auxiliary_runs(root, "nfe200_audit.json", prior_replicate)
    if not audit.empty:
        audit.to_csv(destination / "nfe200_audit_summaries.csv", index=False)
        audit_gates = nfe_audit_summary(audit)
        (destination / "nfe200_audit.json").write_text(
            json.dumps(audit_gates, indent=2, ensure_ascii=False) + "\n"
        )
    semantic = load_auxiliary_runs(root, "semantic_gap_audit.json", prior_replicate)
    if not semantic.empty:
        semantic.to_csv(destination / "semantic_gap_audit_summaries.csv", index=False)
    print(json.dumps(gates, indent=2, ensure_ascii=False))
    if not paired.empty:
        print("\n" + paired.to_string(index=False))
    print("\n" + prediction.to_string(index=False))
    return gates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prior-replicate", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    return summarize(args.root, args.output, args.prior_replicate)


if __name__ == "__main__":
    main()
