#!/usr/bin/env python3
"""Aggregate multi-seed prediction-target extrapolation toy results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SETTING_KEYS = ["seed", "D", "curvature", "hidden", "loss_space"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def discover_settings(root: Path) -> list[Path]:
    return sorted(
        path.parent
        for path in root.rglob("setting_summary.json")
        if (path.parent / "teacher_metrics.csv").is_file()
        and (path.parent / "generation_metrics.csv").is_file()
    )


def load_results(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries, teacher_frames, generation_frames = [], [], []
    for setting in discover_settings(root):
        summary = json.loads((setting / "setting_summary.json").read_text(encoding="utf-8"))
        summary["setting_path"] = str(setting)
        summaries.append(summary)
        teacher = pd.read_csv(setting / "teacher_metrics.csv")
        generation = pd.read_csv(setting / "generation_metrics.csv")
        for key in SETTING_KEYS:
            if key not in teacher.columns:
                teacher[key] = summary[key]
            if key not in generation.columns:
                generation[key] = summary[key]
        teacher["setting_path"] = str(setting)
        generation["setting_path"] = str(setting)
        teacher_frames.append(teacher)
        generation_frames.append(generation)
    if not summaries:
        raise RuntimeError(f"no complete settings found below {root}")
    return (
        pd.DataFrame(summaries),
        pd.concat(teacher_frames, ignore_index=True),
        pd.concat(generation_frames, ignore_index=True),
    )


def build_contrasts(generation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, frame in generation.groupby(SETTING_KEYS, dropna=False):
        x_rows = frame[frame["condition"] == "x"]
        if len(x_rows) != 1:
            raise RuntimeError(f"expected one x baseline for setting {keys}")
        x_row = x_rows.iloc[0]
        for _, row in frame.iterrows():
            kind = str(row["kind"])
            strength = float(row["strength"])
            if kind not in {"xv", "xeps"}:
                continue
            if strength > 0:
                operation = "extrapolation"
            elif strength < 0:
                operation = "interpolation"
            else:
                operation = "baseline"
            payload = dict(zip(SETTING_KEYS, keys))
            payload.update(
                {
                    "condition": row["condition"],
                    "kind": kind,
                    "strength": strength,
                    "operation": operation,
                    "swd_2d": float(row["swd_2d"]),
                    "x_swd_2d": float(x_row["swd_2d"]),
                    "delta_swd_vs_x": float(row["swd_2d"] - x_row["swd_2d"]),
                    "relative_swd_vs_x": float(
                        row["swd_2d"] / max(float(x_row["swd_2d"]), 1e-12) - 1.0
                    ),
                    "swd_delta_ci_low": float(row["swd_delta_vs_x_ci_low"]),
                    "swd_delta_ci_high": float(row["swd_delta_vs_x_ci_high"]),
                    "mmd_2d": float(row["mmd_2d"]),
                    "x_mmd_2d": float(x_row["mmd_2d"]),
                    "delta_mmd_vs_x": float(row["mmd_2d"] - x_row["mmd_2d"]),
                    "manifold_consistency_rms": float(row["manifold_consistency_rms"]),
                    "x_manifold_consistency_rms": float(
                        x_row["manifold_consistency_rms"]
                    ),
                    "delta_manifold_rms_vs_x": float(
                        row["manifold_consistency_rms"]
                        - x_row["manifold_consistency_rms"]
                    ),
                }
            )
            rows.append(payload)
    return pd.DataFrame(rows)


def build_baseline_regimes(generation: pd.DataFrame) -> pd.DataFrame:
    """Summarize whether a setting is meaningful for x-away-from-v extrapolation.

    This table intentionally reports continuous quantities instead of declaring
    an arbitrary sweet spot. Positive x-v extrapolation is only interpretable
    after x is better than v while retaining a non-zero endpoint defect.
    """
    rows = []
    for keys, frame in generation.groupby(SETTING_KEYS, dropna=False):
        baselines = frame.set_index("condition")
        if "x" not in baselines.index or "v" not in baselines.index:
            raise RuntimeError(f"x/v baselines missing for setting {keys}")
        x_row = baselines.loc["x"]
        v_row = baselines.loc["v"]
        x_swd = float(x_row["swd_2d"])
        v_swd = float(v_row["swd_2d"])
        x_mmd = float(x_row["mmd_2d"])
        v_mmd = float(v_row["mmd_2d"])
        x_manifold = float(x_row["manifold_consistency_rms"])
        v_manifold = float(v_row["manifold_consistency_rms"])
        row = dict(zip(SETTING_KEYS, keys))
        row.update(
            {
                "x_swd": x_swd,
                "v_swd": v_swd,
                "v_over_x_swd": v_swd / max(x_swd, 1e-12),
                "x_better_swd": bool(x_swd < v_swd),
                "x_mmd": x_mmd,
                "v_mmd": v_mmd,
                "v_minus_x_mmd": v_mmd - x_mmd,
                "x_better_mmd": bool(x_mmd < v_mmd),
                "x_manifold_rms": x_manifold,
                "v_manifold_rms": v_manifold,
                "v_over_x_manifold_rms": v_manifold / max(x_manifold, 1e-12),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(SETTING_KEYS).reset_index(drop=True)


def aggregate_baseline_regimes(regimes: pd.DataFrame) -> pd.DataFrame:
    group_keys = ["D", "curvature", "hidden", "loss_space"]
    rows = []
    for keys, frame in regimes.groupby(group_keys, dropna=False):
        row = dict(zip(group_keys, keys))
        row.update(
            {
                "seeds": len(frame),
                "mean_x_swd": float(frame["x_swd"].mean()),
                "mean_v_swd": float(frame["v_swd"].mean()),
                "mean_v_over_x_swd": float(frame["v_over_x_swd"].mean()),
                "x_better_swd_seed_fraction": float(frame["x_better_swd"].mean()),
                "mean_v_minus_x_mmd": float(frame["v_minus_x_mmd"].mean()),
                "x_better_mmd_seed_fraction": float(frame["x_better_mmd"].mean()),
                "mean_x_manifold_rms": float(frame["x_manifold_rms"].mean()),
                "mean_v_manifold_rms": float(frame["v_manifold_rms"].mean()),
                "mean_v_over_x_manifold_rms": float(
                    frame["v_over_x_manifold_rms"].mean()
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_keys).reset_index(drop=True)


def aggregate_contrasts(contrasts: pd.DataFrame) -> pd.DataFrame:
    group_keys = ["D", "curvature", "hidden", "loss_space", "condition", "kind", "strength", "operation"]
    rows = []
    for keys, frame in contrasts.groupby(group_keys, dropna=False):
        delta = frame["delta_swd_vs_x"].to_numpy(float)
        delta_mmd = frame["delta_mmd_vs_x"].to_numpy(float)
        manifold = frame["delta_manifold_rms_vs_x"].to_numpy(float)
        manifold_guardrail = (
            frame["manifold_consistency_rms"].to_numpy(float)
            <= 1.10 * frame["x_manifold_consistency_rms"].to_numpy(float)
        )
        swd_bootstrap_improved = frame["swd_delta_ci_high"].to_numpy(float) < 0
        joint_point_pass = (delta < 0) & (delta_mmd < 0) & manifold_guardrail
        joint_strict_pass = swd_bootstrap_improved & (delta_mmd < 0) & manifold_guardrail
        row = dict(zip(group_keys, keys))
        row.update(
            {
                "seeds": len(frame),
                "mean_delta_swd_vs_x": float(delta.mean()),
                "std_delta_swd_vs_x": float(delta.std(ddof=1)) if len(delta) > 1 else 0.0,
                "swd_improved_seed_fraction": float((delta < 0).mean()),
                "swd_bootstrap_improved_seed_fraction": float(
                    swd_bootstrap_improved.mean()
                ),
                "swd_bootstrap_worsened_seed_fraction": float(
                    (frame["swd_delta_ci_low"].to_numpy(float) > 0).mean()
                ),
                "worst_delta_swd_vs_x": float(delta.max()),
                "mean_relative_swd_vs_x": float(frame["relative_swd_vs_x"].mean()),
                "mean_delta_mmd_vs_x": float(delta_mmd.mean()),
                "mmd_improved_seed_fraction": float((delta_mmd < 0).mean()),
                "mean_delta_manifold_rms_vs_x": float(manifold.mean()),
                "manifold_not_worse_seed_fraction": float((manifold <= 0).mean()),
                "manifold_within_10pct_seed_fraction": float(manifold_guardrail.mean()),
                "joint_point_pass_seed_fraction": float(joint_point_pass.mean()),
                "joint_distribution_pass_seed_fraction": float(joint_strict_pass.mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_keys).reset_index(drop=True)


def plot_tradeoff(contrasts: pd.DataFrame, path: Path) -> None:
    kinds = [kind for kind in ("xv", "xeps") if kind in set(contrasts["kind"])]
    fig, axes = plt.subplots(1, len(kinds), figsize=(7.0 * len(kinds), 5.5), squeeze=False)
    for ax, kind in zip(axes.flat, kinds):
        frame = contrasts[contrasts["kind"] == kind]
        for seed, seed_frame in frame.groupby("seed"):
            seed_frame = seed_frame.sort_values("strength")
            ax.plot(
                seed_frame["delta_swd_vs_x"],
                seed_frame["delta_manifold_rms_vs_x"],
                marker="o",
                alpha=0.7,
                label=f"seed {int(seed)}",
            )
            for _, row in seed_frame.iterrows():
                ax.annotate(f"{row['strength']:+.2g}", (row["delta_swd_vs_x"], row["delta_manifold_rms_vs_x"]), fontsize=8)
        ax.axvline(0.0, color="black", linewidth=1)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_title(f"{kind}: prediction mixture")
        ax.set_xlabel("SWD change vs x (lower is better)")
        ax.set_ylabel("off-manifold RMS change vs x (lower is better)")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(
    summary: pd.DataFrame,
    baseline_aggregate: pd.DataFrame,
    aggregate: pd.DataFrame,
    path: Path,
) -> None:
    primary = aggregate[np.isclose(aggregate["strength"].abs(), 0.03)].copy()
    lines = [
        "Prediction-target extrapolation toy v4: multi-seed summary",
        "============================================================",
        "",
        f"Complete settings: {len(summary)}",
        f"Seeds: {sorted(int(value) for value in summary['seed'].unique())}",
        "",
        "Sign convention:",
        "  gamma > 0: extrapolate from x away from v/epsilon",
        "  gamma < 0: interpolate from x toward v/epsilon",
        "  delta SWD < 0: better intrinsic distribution match than x baseline",
        "  delta MMD < 0: a second distribution metric also improves",
        "  delta manifold RMS > 0: farther from the known clean manifold",
        "  strict joint pass: SWD bootstrap CI < 0, MMD improves, and manifold RMS is within 10% of x",
        "",
        "Baseline eligibility for positive x-away-from-v extrapolation:",
        "  First require x to beat v, while x still has a measurable endpoint defect.",
        "  A huge v/x manifold ratio warns that the gap is dominated by weak-model garbage.",
        baseline_aggregate.to_string(index=False),
        "",
        "Primary |gamma|=0.03 rows:",
        primary.to_string(index=False) if len(primary) else "  unavailable",
        "",
        "This screen is evidence for estimator mixing only. It does not show that",
        "different prediction targets have different population optima; under exact",
        "MSE Bayes prediction their clean estimates coincide algebraically.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.input_root.expanduser().resolve()
    output = (args.output_dir or root / "aggregate").expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary, teacher, generation = load_results(root)
    baseline_regimes = build_baseline_regimes(generation)
    baseline_aggregate = aggregate_baseline_regimes(baseline_regimes)
    contrasts = build_contrasts(generation)
    aggregate = aggregate_contrasts(contrasts)
    summary.to_csv(output / "settings.csv", index=False)
    teacher.to_csv(output / "teacher_metrics.csv", index=False)
    generation.to_csv(output / "generation_metrics.csv", index=False)
    baseline_regimes.to_csv(output / "baseline_regimes.csv", index=False)
    baseline_aggregate.to_csv(output / "aggregate_baseline_regimes.csv", index=False)
    contrasts.to_csv(output / "matched_randomness_contrasts.csv", index=False)
    aggregate.to_csv(output / "aggregate_contrasts.csv", index=False)
    plot_tradeoff(contrasts, output / "swd_manifold_tradeoff.png")
    write_report(summary, baseline_aggregate, aggregate, output / "final_report.txt")
    print(f"Wrote aggregate report to {output}")


if __name__ == "__main__":
    main()
