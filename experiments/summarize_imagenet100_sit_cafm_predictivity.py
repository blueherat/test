#!/usr/bin/env python3
"""Join frozen CAFM A/B predictions with pre-existing SiT quality results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


REPO = Path(__file__).resolve().parents[1]
DATA = Path("/home/zhoushunyu/data/eqvae/imagenet_sit_flow")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def aggregate_predictions(audit_root: Path) -> tuple[list[dict], list[dict]]:
    seed_files = sorted(audit_root.glob("seed*/direction_scores.csv"))
    if len(seed_files) < 2:
        raise RuntimeError(f"expected at least two critic audits under {audit_root}")
    raw = []
    for path in seed_files:
        seed = path.parent.name
        for row in read_csv(path):
            row = dict(row)
            row["critic_seed"] = seed
            raw.append(row)

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in raw:
        grouped[(row["direction"], row["time_bin"])].append(row)
    aggregate = []
    numeric_fields = (
        "A_mean",
        "A_se",
        "B_mean",
        "B_se",
        "AB_mean",
        "B2_mean",
        "gamma_hat_mean",
        "gamma_hat_sample_ls",
        "predicted_reduction_gamma1_mean_residual",
        "predicted_reduction_gamma1_sample_ls",
        "B_positive_fraction",
        "direction_rms",
        "euclidean_B_mean",
        "euclidean_cosine_mean",
    )
    for (direction, time_bin), rows in sorted(grouped.items()):
        result = {
            "direction": direction,
            "time_bin": time_bin,
            "critic_seeds": len(rows),
            "samples_per_seed": int(rows[0]["samples"]),
        }
        for field in numeric_fields:
            values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
            finite = values[np.isfinite(values)]
            result[field] = float(finite.mean()) if len(finite) else math.nan
            result[field + "_between_seed_sd"] = (
                float(finite.std(ddof=1)) if len(finite) > 1 else 0.0
            )
        a = np.asarray([float(row["A_mean"]) for row in rows])
        b = np.asarray([float(row["B_mean"]) for row in rows])
        gamma = np.asarray([float(row["gamma_hat_mean"]) for row in rows])
        result["gamma_hat_multicritic"] = float(np.dot(a, b) / max(np.dot(b, b), 1e-20))
        result["A_positive_seed_fraction"] = float(np.mean(a > 0.0))
        result["B_positive_seed_fraction"] = float(np.mean(b > 0.0))
        finite_gamma = gamma[np.isfinite(gamma)]
        result["gamma_positive_seed_fraction"] = (
            float(np.mean(finite_gamma > 0.0)) if len(finite_gamma) else math.nan
        )
        aggregate.append(result)
    return raw, aggregate


def summarize_critic_health(audit_root: Path) -> dict:
    manifests = sorted(audit_root.glob("seed*/manifest.json"))
    if len(manifests) < 2:
        raise RuntimeError(f"expected at least two audit manifests under {audit_root}")
    rows = []
    for path in manifests:
        payload = json.loads(path.read_text(encoding="utf-8"))
        validation = payload.get("critic_validation")
        if not validation:
            raise RuntimeError(f"critic validation missing from {path}")
        rows.append(
            {
                "seed": path.parent.name,
                "step": int(payload["critic_training_step"]),
                "loss": float(validation["loss"]),
                "margin": float(validation["margin"]),
                "real_sign_accuracy": float(validation["real_sign_accuracy"]),
                "fake_sign_accuracy": float(validation["fake_sign_accuracy"]),
            }
        )
    loss = np.asarray([row["loss"] for row in rows])
    margin = np.asarray([row["margin"] for row in rows])
    return {
        "per_seed": rows,
        "loss_mean": float(loss.mean()),
        "loss_between_seed_sd": float(loss.std(ddof=1)),
        "loss_below_zero_critic_baseline_fraction": float(np.mean(loss < 2.0)),
        "margin_mean": float(margin.mean()),
        "margin_between_seed_sd": float(margin.std(ddof=1)),
        "positive_margin_seed_fraction": float(np.mean(margin > 0.0)),
        "all_fixed_step_1000": all(row["step"] == 1000 for row in rows),
    }


def historical_targets() -> list[dict]:
    rows: list[dict] = []
    long_summary_path = (
        DATA / "checkpoint_reference_long_study_v1/summary/final_summary.json"
    )
    long_summary = json.loads(long_summary_path.read_text(encoding="utf-8"))
    baseline = float(long_summary["shapley"]["baseline_fid"])
    for name, result in long_summary["best_static_by_reference"].items():
        rows.append(
            {
                "direction": name,
                "family": "same_target_checkpoint",
                "quality_protocol": "paired_fid1k_best_static_gamma",
                "historical_gamma": float(result["gamma"]),
                "historical_fid": float(result["fid"]),
                "historical_baseline_fid": baseline,
                "historical_fid_gain": baseline - float(result["fid"]),
                "checkpoint_step": int(name[1:]) * 1000,
                "source": str(long_summary_path),
            }
        )

    compact_path = (
        REPO
        / "docs/data/imagenet100_sit_800k_compact_replication/compact_replication_rows.csv"
    )
    compact = read_csv(compact_path)
    for direction, condition in (("x800", "x_closed"), ("v500", "vweak_closed")):
        matched = [row for row in compact if row["condition"] == condition]
        baselines = [row for row in compact if row["condition"] == "baseline"]
        rows.append(
            {
                "direction": direction,
                "family": "independent_model_gamma1_confirm",
                "quality_protocol": "paired_fid5k_two_sampling_seeds_gamma1",
                "historical_gamma": 1.0,
                "historical_fid": float(np.mean([float(row["fid"]) for row in matched])),
                "historical_baseline_fid": float(
                    np.mean([float(row["fid"]) for row in baselines])
                ),
                "historical_fid_gain": float(
                    np.mean([float(row["fid_gain_vs_paired_baseline"]) for row in matched])
                ),
                "checkpoint_step": 800_000 if direction == "x800" else 500_000,
                "source": str(compact_path),
            }
        )

    internal_path = REPO / "internal_head_gamma_schedule_sweep_v4/summary/best_by_experiment.csv"
    for row in read_csv(internal_path):
        if row["kind"] != "static" or not row["experiment"].startswith("depth"):
            continue
        depth = int(row["experiment"].removeprefix("depth").removesuffix("_v"))
        rows.append(
            {
                "direction": f"internal_depth{depth}",
                "family": "internal_head",
                "quality_protocol": "paired_fid1k_best_static_gamma",
                "historical_gamma": float(row["best_gamma"]),
                "historical_fid": float(row["best_fid"]),
                "historical_baseline_fid": float(row["best_fid"])
                + float(row["fid_improvement_vs_baseline"]),
                "historical_fid_gain": float(row["fid_improvement_vs_baseline"]),
                "checkpoint_step": 50_000,
                "source": str(internal_path),
            }
        )

    temporal_path = REPO / "external_v180_temporal_utility_fid1k_v1/summary/summary.json"
    temporal = json.loads(temporal_path.read_text(encoding="utf-8"))
    for direction, key in (
        ("v180_high_noise_only", "best_high_noise"),
        ("v180_low_noise_only", "best_low_noise"),
    ):
        result = temporal[key]
        rows.append(
            {
                "direction": direction,
                "family": "time_window",
                "quality_protocol": "paired_fid1k_best_window_gamma",
                "historical_gamma": float(result["gamma"]),
                "historical_fid": float(result["fid"]),
                "historical_baseline_fid": float(temporal["baseline"]["fid"]),
                "historical_fid_gain": float(temporal["baseline"]["fid"])
                - float(result["fid"]),
                "checkpoint_step": 180_000,
                "source": str(temporal_path),
            }
        )
    return rows


def safe_spearman(x: list[float], y: list[float]) -> dict[str, float]:
    result = spearmanr(x, y)
    return {"rho": float(result.statistic), "pvalue": float(result.pvalue)}


def evaluate_predictivity(joined: list[dict]) -> dict:
    report: dict[str, object] = {}
    for family in ("same_target_checkpoint", "internal_head"):
        rows = [row for row in joined if row["family"] == family]
        target = [float(row["historical_fid_gain"]) for row in rows]
        correlations = {}
        predictors = (
            "cafm_predicted_reduction_at_historical_gamma",
            "B_mean",
            "predicted_reduction_gamma1_mean_residual",
            "direction_rms",
            "euclidean_B_mean",
            "euclidean_cosine_mean",
            "checkpoint_age",
        )
        for predictor in predictors:
            values = [float(row[predictor]) for row in rows]
            if len(set(values)) < 2:
                continue
            correlations[predictor] = safe_spearman(values, target)
        gamma_true = [float(row["historical_gamma"]) for row in rows]
        gamma_pred = [float(row["gamma_hat_multicritic"]) for row in rows]
        finite = [
            (truth, prediction)
            for truth, prediction in zip(gamma_true, gamma_pred)
            if math.isfinite(prediction)
        ]
        report[family] = {
            "count": len(rows),
            "correlations_with_fid_gain": correlations,
            "gamma_hat_spearman": safe_spearman(
                [pair[1] for pair in finite], [pair[0] for pair in finite]
            )
            if len(finite) > 2
            else None,
            "gamma_hat_median_absolute_error": float(
                np.median([abs(truth - prediction) for truth, prediction in finite])
            )
            if finite
            else math.nan,
        }

    windows = {row["direction"]: row for row in joined if row["family"] == "time_window"}
    high = windows["v180_high_noise_only"]
    low = windows["v180_low_noise_only"]
    report["time_window"] = {
        "historical_high_gain_exceeds_low": float(high["historical_fid_gain"])
        > float(low["historical_fid_gain"]),
        "cafm_B_high_exceeds_low": float(high["B_mean"]) > float(low["B_mean"]),
        "cafm_predicted_reduction_high_exceeds_low": float(
            high["cafm_predicted_reduction_at_historical_gamma"]
        )
        > float(low["cafm_predicted_reduction_at_historical_gamma"]),
        "historical_gain_difference": float(high["historical_fid_gain"])
        - float(low["historical_fid_gain"]),
        "B_difference": float(high["B_mean"]) - float(low["B_mean"]),
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-root",
        type=Path,
        default=DATA / "cafm_tangent_predictivity_v1/audits",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA / "cafm_tangent_predictivity_v1/summary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.audit_root = args.audit_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw, aggregate = aggregate_predictions(args.audit_root)
    write_csv(args.output_dir / "all_critic_scores.csv", raw)
    write_csv(args.output_dir / "aggregate_scores.csv", aggregate)

    overall = {row["direction"]: row for row in aggregate if row["time_bin"] == "overall"}
    historical = historical_targets()
    joined = []
    for target in historical:
        prediction = overall[target["direction"]]
        row = {**target, **prediction}
        gamma = float(target["historical_gamma"])
        a = float(row["A_mean"])
        b = float(row["B_mean"])
        row["cafm_predicted_reduction_at_historical_gamma"] = (
            a * a - (a - gamma * b) ** 2
        )
        row["checkpoint_age"] = 800_000 - int(row["checkpoint_step"])
        joined.append(row)
    write_csv(args.output_dir / "predictions_with_historical_quality.csv", joined)
    evaluation = evaluate_predictivity(joined)
    critic_health = summarize_critic_health(args.audit_root)
    payload = {
        "format": "eqvae_cafm_tangent_predictivity_summary_v1",
        "critic_seed_count": len({row["critic_seed"] for row in raw}),
        "generator_updated": False,
        "new_fid_computed": False,
        "evaluation": evaluation,
        "critic_health": critic_health,
        "gate": {
            "requirements": [
                "fixed-step critics show reproducible held-out tangent separation rather than collapse to loss 2",
                "CAFM score predicts same-target FID ranking at least as well as norm/cosine/age",
                "held-out internal-head ordering is nontrivial and correctly ranked",
                "high-noise v180 utility is ranked above low-noise utility",
                "predicted gamma is not catastrophically inconsistent with historical optima",
            ],
            "decision": "requires_result_review",
        },
    }
    atomic_json(payload, args.output_dir / "summary.json")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
