"""Aggregate replicated RAEv2 predicted-clean 2x2 audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONDITIONS = ("full_on_full", "ig_on_full", "full_on_ig", "ig_on_ig")
COLORS = {
    "full_on_full": "#3366aa",
    "ig_on_full": "#ee7733",
    "full_on_ig": "#228833",
    "ig_on_ig": "#aa3377",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_runs(run_dirs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    summaries = []
    effects = []
    manifests = []
    for run_dir in run_dirs:
        path = run_dir.expanduser().resolve()
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("protocol") != "raev2_predicted_clean_2x2_v1":
            raise ValueError(f"unexpected protocol in {path}")
        seed = int(manifest["seed"])
        summary = pd.read_csv(path / "predicted_clean_summary.csv")
        effect = pd.read_csv(path / "predicted_clean_effects.csv")
        summary.insert(0, "seed", seed)
        effect.insert(0, "seed", seed)
        summaries.append(summary)
        effects.append(effect)
        manifests.append(manifest)
    if len({manifest["seed"] for manifest in manifests}) != len(manifests):
        raise ValueError("replicate seeds must be unique")
    invariant_keys = (
        "protocol",
        "checkpoint",
        "state_key",
        "samples",
        "requested_times",
        "matched_times",
        "ig_scale",
        "inception_feature",
    )
    first = manifests[0]
    for manifest in manifests[1:]:
        for key in invariant_keys:
            if manifest.get(key) != first.get(key):
                raise ValueError(f"replicate manifests differ for {key}")
    return pd.concat(summaries, ignore_index=True), pd.concat(effects, ignore_index=True), manifests


def aggregate_runs(
    summaries: pd.DataFrame, effects: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_columns = ("auc", "auc_separability", "fid_real", "fid_reconstruction")
    summary_agg = summaries.groupby(
        ["requested_time", "actual_time", "condition", "head", "state_branch", "on_policy"],
        as_index=False,
    )[list(metric_columns)].agg(["mean", "std", "min", "max"])
    summary_agg.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else str(column)
        for column in summary_agg.columns
    ]

    effect_metrics = (
        "auc_delta",
        "auc_separability_delta",
        "fid_real_delta",
        "fid_reconstruction_delta",
    )
    effect_agg = effects.groupby(
        ["requested_time", "actual_time", "effect", "positive_condition", "negative_condition"],
        as_index=False,
    )[list(effect_metrics)].agg(["mean", "std", "min", "max"])
    effect_agg.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else str(column)
        for column in effect_agg.columns
    ]
    for metric in ("auc_separability_delta", "fid_real_delta", "fid_reconstruction_delta"):
        counts = effects.groupby(["requested_time", "effect"])[metric].apply(
            lambda values: int((values < 0).sum())
        )
        effect_agg[f"{metric}_negative_seed_count"] = [
            counts.loc[(time, effect)]
            for time, effect in zip(effect_agg["requested_time"], effect_agg["effect"])
        ]
    return summary_agg, effect_agg


def load_diagnostics(
    run_dirs: list[Path], manifests: list[dict[str, Any]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    decode_rows = []
    latent_rows = []
    for run_dir, manifest in zip(run_dirs, manifests):
        root = run_dir.expanduser().resolve()
        world_size = int(manifest["world_size"])
        rank_payloads = [
            json.loads(
                (root / f"predicted_clean_diagnostics_rank{rank:02d}.json").read_text(
                    encoding="utf-8"
                )
            )
            for rank in range(world_size)
        ]
        rank_counts = []
        for rank in range(world_size):
            with np.load(root / f"predicted_clean_features_rank{rank:02d}.npz") as shard:
                rank_counts.append(int(shard["ids"].size))
        for branch in ("full", "ig"):
            decode_by_key: dict[tuple[float, str], list[dict[str, float]]] = {}
            latent_by_time: dict[float, list[dict[str, float]]] = {}
            for payload in rank_payloads:
                for row in payload[branch]["decode_rows"]:
                    decode_by_key.setdefault(
                        (float(row["requested_time"]), str(row["condition"])), []
                    ).append(row)
                for row in payload[branch]["latent_rows"]:
                    latent_by_time.setdefault(float(row["requested_time"]), []).append(row)
            total = sum(rank_counts)
            for (requested_time, condition), values in decode_by_key.items():
                decode_rows.append(
                    {
                        "seed": int(manifest["seed"]),
                        "requested_time": requested_time,
                        "condition": condition,
                        "raw_min": min(row["raw_min"] for row in values),
                        "raw_max": max(row["raw_max"] for row in values),
                        "clipped_low_fraction": sum(
                            row["clipped_low_fraction"] * count
                            for row, count in zip(values, rank_counts)
                        ) / total,
                        "clipped_high_fraction": sum(
                            row["clipped_high_fraction"] * count
                            for row, count in zip(values, rank_counts)
                        ) / total,
                    }
                )
            for requested_time, values in latent_by_time.items():
                row = {
                    "seed": int(manifest["seed"]),
                    "requested_time": requested_time,
                    "state_branch": branch,
                }
                for metric in (
                    "full_rms",
                    "base_rms",
                    "head_gap_rms",
                    "guided_minus_full_rms",
                ):
                    row[metric] = float(
                        np.sqrt(
                            sum(value[metric] ** 2 * count for value, count in zip(values, rank_counts))
                            / total
                        )
                    )
                latent_rows.append(row)
    decode = pd.DataFrame(decode_rows)
    decode["clipped_total_fraction"] = (
        decode["clipped_low_fraction"] + decode["clipped_high_fraction"]
    )
    return decode, pd.DataFrame(latent_rows)


def _plot(summary: pd.DataFrame, effects: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    metrics = (
        ("auc", "AUC vs D(E(x))"),
        ("fid_real", "FID to ImageNet"),
        ("fid_reconstruction", "FID to D(E(x))"),
    )
    for condition in CONDITIONS:
        frame = summary[summary["condition"] == condition].sort_values("actual_time")
        for axis, (metric, label) in zip(axes.flat[:3], metrics):
            axis.plot(
                frame["actual_time"],
                frame[f"{metric}_mean"],
                "o-",
                color=COLORS[condition],
                label=condition,
            )
            axis.fill_between(
                frame["actual_time"],
                frame[f"{metric}_min"],
                frame[f"{metric}_max"],
                color=COLORS[condition],
                alpha=0.12,
            )
            axis.set_ylabel(label)
    axes[0, 0].axhline(0.5, color="#333333", linestyle="--", linewidth=1)
    total = effects[effects["effect"] == "on_policy_total"].sort_values("actual_time")
    axes[1, 1].plot(
        total["actual_time"], total["fid_real_delta_mean"], "o-", label="FID real"
    )
    axes[1, 1].plot(
        total["actual_time"],
        total["fid_reconstruction_delta_mean"],
        "s-",
        label="FID reconstruction",
    )
    axes[1, 1].fill_between(
        total["actual_time"],
        total["fid_real_delta_min"],
        total["fid_real_delta_max"],
        alpha=0.12,
    )
    axes[1, 1].axhline(0.0, color="#333333", linestyle="--", linewidth=1)
    axes[1, 1].set_ylabel("On-policy IG - Full (lower is better)")
    for axis in axes.flat:
        axis.set_xlabel("Solver time t (sampling: 1 to 0)")
        axis.invert_xaxis()
        axis.grid(True, alpha=0.22)
        axis.legend(frameon=False)
    fig.suptitle("RAEv2 Predicted-Clean 2x2 Across Seeds")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries, effects, manifests = load_runs(args.run_dir)
    summary_agg, effect_agg = aggregate_runs(summaries, effects)
    decode_diagnostics, latent_diagnostics = load_diagnostics(args.run_dir, manifests)
    summaries.to_csv(output_dir / "all_seed_predicted_clean_summary.csv", index=False)
    effects.to_csv(output_dir / "all_seed_predicted_clean_effects.csv", index=False)
    summary_agg.to_csv(output_dir / "cross_seed_predicted_clean_summary.csv", index=False)
    effect_agg.to_csv(output_dir / "cross_seed_predicted_clean_effects.csv", index=False)
    decode_diagnostics.to_csv(
        output_dir / "per_seed_predicted_clean_decode_diagnostics.csv", index=False
    )
    latent_diagnostics.to_csv(
        output_dir / "per_seed_predicted_clean_latent_head_diagnostics.csv", index=False
    )
    _plot(summary_agg, effect_agg, output_dir / "cross_seed_predicted_clean_curves.png")
    manifest = {
        "protocol": "raev2_predicted_clean_2x2_cross_seed_v1",
        "source_runs": [str(path.expanduser().resolve()) for path in args.run_dir],
        "seeds": [int(item["seed"]) for item in manifests],
        "replicates": len(manifests),
        "effect_sign": "negative means the positive condition is closer/better",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(effect_agg.to_string(index=False))


if __name__ == "__main__":
    main()
