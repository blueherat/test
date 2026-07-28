"""Evaluate the preregistered SPC mechanism predictions across paired seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_RESULTS = Path.home() / "data/eqvae/experiments/rae_spc_multiseed_v1"


def _validate_gradient_table(table: pd.DataFrame) -> None:
    required = {
        "training_seed",
        "condition",
        "checkpoint_step",
        "time",
        "parameter_group",
        "semantic_loss",
        "basis_loss",
        "basis_over_semantic_norm",
        "semantic_basis_cosine",
        "semantic_descent_ratio",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"gradient table lacks columns: {sorted(missing)}")


def build_loss_pairs(
    table: pd.DataFrame,
    *,
    switch_step: int = 2000,
    endpoint: int = 5000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-time and mean-over-time paired loss differences.

    The loss scalars are repeated for each parameter group, so one row per
    seed/condition/step/time is retained before pairing.
    """

    _validate_gradient_table(table)
    selected = table[
        table["checkpoint_step"].isin((switch_step, endpoint))
    ].copy()
    keys = ["training_seed", "condition", "checkpoint_step", "time"]
    loss_columns = ["semantic_loss", "basis_loss"]
    consistency = selected.groupby(keys, dropna=False)[loss_columns].nunique()
    if (consistency > 1).any().any():
        raise ValueError("loss values disagree across parameter groups")
    losses = selected.drop_duplicates(keys)[keys + loss_columns]
    if losses.duplicated(keys).any():
        raise ValueError("duplicate loss probe rows")

    wide = losses.pivot(
        index=["training_seed", "checkpoint_step", "time"],
        columns="condition",
        values=loss_columns,
    )
    for condition in ("static", "spc"):
        if condition not in wide.columns.get_level_values(1):
            raise ValueError(f"missing condition: {condition}")
    rows: list[dict[str, float | int]] = []
    for (seed, step, time), row in wide.iterrows():
        result: dict[str, float | int] = {
            "seed": int(seed),
            "checkpoint_step": int(step),
            "time": float(time),
        }
        for loss in loss_columns:
            static = float(row[(loss, "static")])
            spc = float(row[(loss, "spc")])
            result[f"static_{loss}"] = static
            result[f"spc_{loss}"] = spc
            result[f"delta_{loss}"] = spc - static
            result[f"relative_delta_{loss}"] = (spc - static) / static
        rows.append(result)
    per_time = pd.DataFrame(rows).sort_values(
        ["seed", "checkpoint_step", "time"]
    ).reset_index(drop=True)

    numeric = [
        column
        for column in per_time.columns
        if column not in {"seed", "checkpoint_step", "time"}
    ]
    mean_time = (
        per_time.groupby(["seed", "checkpoint_step"], as_index=False)[numeric]
        .mean()
        .sort_values(["seed", "checkpoint_step"])
        .reset_index(drop=True)
    )
    return per_time, mean_time


def build_seed_mechanism_table(
    mean_time: pd.DataFrame,
    *,
    switch_step: int = 2000,
    endpoint: int = 5000,
) -> pd.DataFrame:
    rows = []
    for seed, seed_table in mean_time.groupby("seed"):
        indexed = seed_table.set_index("checkpoint_step")
        if switch_step not in indexed.index or endpoint not in indexed.index:
            raise ValueError(f"seed {seed} lacks step {switch_step} or {endpoint}")
        early = indexed.loc[switch_step]
        late = indexed.loc[endpoint]
        early_basis_gap = float(early["relative_delta_basis_loss"])
        late_basis_gap = float(late["relative_delta_basis_loss"])
        if early_basis_gap > 0:
            basis_gap_shrinkage = 1.0 - max(late_basis_gap, 0.0) / early_basis_gap
        else:
            basis_gap_shrinkage = np.nan
        rows.append(
            {
                "seed": int(seed),
                "step2000_delta_semantic_loss": float(
                    early["delta_semantic_loss"]
                ),
                "step2000_relative_delta_semantic_loss": float(
                    early["relative_delta_semantic_loss"]
                ),
                "step2000_delta_basis_loss": float(early["delta_basis_loss"]),
                "step2000_relative_delta_basis_loss": early_basis_gap,
                "step5000_delta_semantic_loss": float(late["delta_semantic_loss"]),
                "step5000_relative_delta_semantic_loss": float(
                    late["relative_delta_semantic_loss"]
                ),
                "step5000_delta_basis_loss": float(late["delta_basis_loss"]),
                "step5000_relative_delta_basis_loss": late_basis_gap,
                "basis_gap_shrinkage": basis_gap_shrinkage,
                "p1_capacity_reallocation": bool(
                    early["delta_basis_loss"] > 0
                    and early["delta_semantic_loss"] < 0
                ),
                "p2_basis_catchup_30pct": bool(
                    np.isfinite(basis_gap_shrinkage)
                    and basis_gap_shrinkage >= 0.30
                ),
                "semantic_advantage_reversed_at_5k": bool(
                    early["delta_semantic_loss"] < 0
                    and late["delta_semantic_loss"] > 0
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)


def build_gradient_relation_table(
    table: pd.DataFrame, *, endpoint: int = 5000, time: float = 0.1
) -> pd.DataFrame:
    _validate_gradient_table(table)
    selected = table[
        (table["checkpoint_step"] == endpoint)
        & np.isclose(table["time"].astype(float), time)
        & (table["parameter_group"] == "last_block")
    ].copy()
    columns = [
        "training_seed",
        "condition",
        "semantic_descent_ratio",
        "basis_over_semantic_norm",
        "semantic_basis_cosine",
    ]
    if selected.duplicated(["training_seed", "condition"]).any():
        raise ValueError("duplicate endpoint gradient relation rows")
    return selected[columns].rename(columns={"training_seed": "seed"}).sort_values(
        ["seed", "condition"]
    ).reset_index(drop=True)


def attach_generation_metrics(
    seed_table: pd.DataFrame, metrics: pd.DataFrame | None
) -> pd.DataFrame:
    if metrics is None:
        return seed_table
    required = {"seed", "condition", "frechet_inception_distance"}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"generation table lacks columns: {sorted(missing)}")
    wide = metrics.pivot(
        index="seed", columns="condition", values="frechet_inception_distance"
    )
    generation = pd.DataFrame(
        {
            "seed": wide.index.astype(int),
            "delta_fid": wide["spc"].to_numpy() - wide["static"].to_numpy(),
        }
    )
    return seed_table.merge(generation, on="seed", how="left", validate="one_to_one")


def _finite_correlation(x: pd.Series, y: pd.Series, method: str) -> float | None:
    valid = np.isfinite(x.to_numpy(float)) & np.isfinite(y.to_numpy(float))
    if valid.sum() < 3:
        return None
    return float(x[valid].corr(y[valid], method=method))


def summarize_mechanism(
    seed_table: pd.DataFrame, gradient_relations: pd.DataFrame
) -> dict[str, object]:
    p1_count = int(seed_table["p1_capacity_reallocation"].sum())
    catchup_count = int(seed_table["p2_basis_catchup_30pct"].sum())
    reversal_count = int(seed_table["semantic_advantage_reversed_at_5k"].sum())
    spc_relations = gradient_relations[
        gradient_relations["condition"] == "spc"
    ]
    summary: dict[str, object] = {
        "seed_count": int(len(seed_table)),
        "loss_aggregation": "unweighted mean over preregistered t={0.3,0.1}",
        "p1_capacity_reallocation_count": p1_count,
        "p1_pass": p1_count >= 4,
        "p2_basis_catchup_30pct_count": catchup_count,
        "p2_semantic_advantage_reversal_count": reversal_count,
        "p2_pass": catchup_count >= 4 and reversal_count <= 2,
        "p3_spc_semantic_descent_ratio_ge_0p9_count": int(
            (spc_relations["semantic_descent_ratio"] >= 0.9).sum()
        ),
        "p3_spc_negative_cosine_count": int(
            (spc_relations["semantic_basis_cosine"] < 0).sum()
        ),
        "p3_spc_basis_pressure_gt_one_count": int(
            (spc_relations["basis_over_semantic_norm"] > 1).sum()
        ),
    }
    summary["mechanism_gate_pass"] = bool(summary["p1_pass"] and summary["p2_pass"])
    if "delta_fid" in seed_table:
        # With five points these are descriptive checks, not significance tests.
        summary["p4_fid_vs_early_semantic_delta_pearson"] = _finite_correlation(
            seed_table["delta_fid"],
            seed_table["step2000_delta_semantic_loss"],
            "pearson",
        )
        summary["p4_fid_vs_early_semantic_delta_spearman"] = _finite_correlation(
            seed_table["delta_fid"],
            seed_table["step2000_delta_semantic_loss"],
            "spearman",
        )
        summary["p4_fid_vs_late_basis_gap_pearson"] = _finite_correlation(
            seed_table["delta_fid"],
            seed_table["step5000_relative_delta_basis_loss"],
            "pearson",
        )
        summary["p4_fid_vs_late_basis_gap_spearman"] = _finite_correlation(
            seed_table["delta_fid"],
            seed_table["step5000_relative_delta_basis_loss"],
            "spearman",
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--gradient-dir", type=Path)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--switch-step", type=int, default=2000)
    parser.add_argument("--endpoint", type=int, default=5000)
    args = parser.parse_args()

    root = args.results.expanduser().resolve()
    gradient_dir = (
        args.gradient_dir.expanduser().resolve()
        if args.gradient_dir
        else root / "gradient_probe"
    )
    output = root / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    gradient = pd.read_csv(gradient_dir / "aggregate_metrics.csv")
    per_time, mean_time = build_loss_pairs(
        gradient, switch_step=args.switch_step, endpoint=args.endpoint
    )
    seed_table = build_seed_mechanism_table(
        mean_time, switch_step=args.switch_step, endpoint=args.endpoint
    )
    relations = build_gradient_relation_table(gradient, endpoint=args.endpoint)

    metrics_path = args.metrics
    if metrics_path is None:
        candidate = output / "spc_metrics_model_n5000_50steps.csv"
        metrics_path = candidate if candidate.exists() else None
    metrics = pd.read_csv(metrics_path) if metrics_path else None
    seed_table = attach_generation_metrics(seed_table, metrics)
    summary = summarize_mechanism(seed_table, relations)

    per_time.to_csv(output / "spc_mechanism_loss_pairs_by_time.csv", index=False)
    mean_time.to_csv(output / "spc_mechanism_loss_pairs_mean_time.csv", index=False)
    seed_table.to_csv(output / "spc_mechanism_by_seed.csv", index=False)
    relations.to_csv(output / "spc_mechanism_gradient_relations.csv", index=False)
    (output / "spc_mechanism_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(seed_table.to_string(index=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
