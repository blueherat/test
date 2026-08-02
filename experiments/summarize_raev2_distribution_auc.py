"""Combine independent RAEv2 distribution-AUC runs without pooling probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def parse_named_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("run must be NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("run name cannot be empty")
    return name, Path(raw_path).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=parse_named_run, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cluster-bootstrap-repeats", type=int, default=2000)
    return parser.parse_args()


def paired_auc(negative: np.ndarray, positive: np.ndarray) -> float:
    labels = np.concatenate(
        (np.zeros(negative.size, dtype=np.int8), np.ones(positive.size, dtype=np.int8))
    )
    return float(roc_auc_score(labels, np.concatenate((negative, positive))))


def class_cluster_bootstrap_delta(
    class_labels: np.ndarray,
    p_full: np.ndarray,
    q_full: np.ndarray,
    p_ig: np.ndarray,
    q_ig: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> tuple[float, float]:
    """Bootstrap paired AUC deltas using ImageNet class as the sampling unit."""

    arrays = (class_labels, p_full, q_full, p_ig, q_ig)
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("cluster bootstrap inputs must be one-dimensional")
    if len({array.size for array in arrays}) != 1:
        raise ValueError("cluster bootstrap inputs must have equal lengths")
    if repeats <= 0:
        raise ValueError("cluster bootstrap repeats must be positive")
    unique_classes = np.unique(class_labels)
    if unique_classes.size < 2:
        raise ValueError("cluster bootstrap requires at least two held-out classes")
    indices_by_class = [np.flatnonzero(class_labels == label) for label in unique_classes]
    rng = np.random.default_rng(seed)
    values = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        selected = rng.integers(0, unique_classes.size, size=unique_classes.size)
        indices = np.concatenate([indices_by_class[index] for index in selected])
        values[repeat] = paired_auc(p_ig[indices], q_ig[indices]) - paired_auc(
            p_full[indices], q_full[indices]
        )
    return tuple(np.quantile(values, (0.025, 0.975)).tolist())


def _cluster_intervals(
    root: Path, frame: pd.DataFrame, *, repeats: int, seed: int
) -> pd.DataFrame:
    protocol_path = root / "sample_protocol.npz"
    scores_path = root / "heldout_probe_scores.npz"
    if not protocol_path.is_file() or not scores_path.is_file():
        raise FileNotFoundError("cluster bootstrap requires saved sample protocol and scores")
    with np.load(protocol_path) as protocol, np.load(scores_path) as scores:
        labels = protocol["labels"]
        rows = []
        for time_index, row in enumerate(frame.itertuples(index=False)):
            suffix = str(float(row.requested_time)).replace(".", "p")
            ids = scores[f"ids_t{suffix}"].astype(np.int64, copy=False)
            heldout_labels = labels[ids]
            low, high = class_cluster_bootstrap_delta(
                heldout_labels,
                scores[f"p_full_t{suffix}"],
                scores[f"q_full_t{suffix}"],
                scores[f"p_ig_t{suffix}"],
                scores[f"q_ig_t{suffix}"],
                repeats=repeats,
                seed=seed + 1009 * time_index,
            )
            rows.append(
                {
                    "requested_time": float(row.requested_time),
                    "delta_ci_low_cluster": low,
                    "delta_ci_high_cluster": high,
                    "heldout_classes": int(np.unique(heldout_labels).size),
                }
            )
    return pd.DataFrame(rows)


def load_runs(
    named_runs: list[tuple[str, Path]], *, cluster_bootstrap_repeats: int = 2000
) -> pd.DataFrame:
    frames = []
    expected_times = None
    for name, root in named_runs:
        root = root.resolve()
        frame = pd.read_csv(root / "auc_delta_ig_minus_full.csv")
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if int(manifest["heldout_pairs"]) < 100:
            raise ValueError(f"run {name!r} has fewer than 100 held-out pairs")
        times = tuple(frame["requested_time"].tolist())
        if expected_times is None:
            expected_times = times
        elif times != expected_times:
            raise ValueError("runs use different requested times")
        cluster = _cluster_intervals(
            root,
            frame,
            repeats=cluster_bootstrap_repeats,
            seed=int(manifest["seed"]) + 400_009,
        )
        frame = frame.merge(cluster, on="requested_time", validate="one_to_one")
        frame = frame.rename(
            columns={
                "delta_ci_low": "delta_ci_low_image_bootstrap",
                "delta_ci_high": "delta_ci_high_image_bootstrap",
                "delta_ci_low_cluster": "delta_ci_low",
                "delta_ci_high_cluster": "delta_ci_high",
            }
        )
        frame.insert(0, "run", name)
        frame["seed"] = int(manifest["seed"])
        frame["heldout_pairs"] = int(manifest["heldout_pairs"])
        frames.append(frame)
    if len({name for name, _ in named_runs}) != len(named_runs):
        raise ValueError("run names must be unique")
    return pd.concat(frames, ignore_index=True)


def summarize(per_run: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (requested, actual), frame in per_run.groupby(
        ["requested_time", "actual_time"], sort=True
    ):
        rows.append(
            {
                "requested_time": requested,
                "actual_time": actual,
                "runs": len(frame),
                "auc_full_mean": frame["auc_full"].mean(),
                "auc_ig_mean": frame["auc_ig"].mean(),
                "auc_delta_mean": frame["auc_delta_ig_minus_full"].mean(),
                "auc_delta_seed_std": frame["auc_delta_ig_minus_full"].std(ddof=1),
                "seeds_significantly_closer": int((frame["delta_ci_high"] < 0).sum()),
                "seeds_significantly_farther": int((frame["delta_ci_low"] > 0).sum()),
                "all_seeds_closer": bool((frame["delta_ci_high"] < 0).all()),
                "all_seeds_farther": bool((frame["delta_ci_low"] > 0).all()),
                "trajectory_relative_rms_mean": frame[
                    "q_ig_vs_full_relative_rms"
                ].mean(),
            }
        )
    return pd.DataFrame(rows)


def conclusion(summary: pd.DataFrame) -> str:
    non_null = summary[summary["requested_time"] < 1.0]
    robust_closer = non_null[non_null["all_seeds_closer"]]
    robust_farther = non_null[non_null["all_seeds_farther"]]
    if not robust_closer.empty and not robust_farther.empty:
        return "stable phase-dependent sign reversal across all seeds"
    if not robust_farther.empty:
        return (
            "no global distribution correction; IG is robustly farther from p_t "
            "at one or more sampled times"
        )
    if not robust_closer.empty:
        return "IG is robustly closer to p_t at one or more sampled times"
    return "no AUC effect is stable across all seeds"


def plot(per_run: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    fig, axis = plt.subplots(figsize=(10.5, 6.5))
    colors = ("#3569a8", "#c94f3d", "#4b8f5a", "#8a5aa8")
    for index, (name, frame) in enumerate(per_run.groupby("run", sort=False)):
        frame = frame.sort_values("actual_time", ascending=False)
        axis.plot(
            frame["actual_time"],
            frame["auc_delta_ig_minus_full"],
            marker="o",
            linewidth=1.7,
            alpha=0.75,
            color=colors[index % len(colors)],
            label=name,
        )
    mean_frame = summary.sort_values("actual_time", ascending=False)
    axis.plot(
        mean_frame["actual_time"],
        mean_frame["auc_delta_mean"],
        color="#111111",
        marker="s",
        linewidth=3.0,
        label="seed mean",
    )
    axis.axhline(0.0, color="#333333", linestyle="--", linewidth=1.3)
    times = per_run["actual_time"].to_numpy(dtype=float)
    span = max(float(times.max() - times.min()), 0.1)
    axis.set_xlim(float(times.max() + 0.03 * span), float(times.min() - 0.03 * span))
    axis.set_xlabel("Actual shifted solver time t")
    axis.set_ylabel("AUC(IG) - AUC(full); positive means IG is farther")
    axis.set_title(
        "RAEv2 Internal Guidance: Distribution-AUC Difference Across Seeds\n"
        "Sampling direction: t=1 to t=0"
    )
    axis.grid(True, alpha=0.22)
    axis.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    per_run = load_runs(
        args.run, cluster_bootstrap_repeats=args.cluster_bootstrap_repeats
    )
    summary = summarize(per_run)
    decision = conclusion(summary)
    per_run.to_csv(output_dir / "per_seed_auc_delta.csv", index=False)
    summary.to_csv(output_dir / "cross_seed_summary.csv", index=False)
    plot(per_run, summary, output_dir / "cross_seed_auc_delta.png")
    report = {
        "protocol": "raev2_distribution_auc_cross_seed_v2",
        "runs": {name: str(path.resolve()) for name, path in args.run},
        "bootstrap_unit": "held-out ImageNet class",
        "cluster_bootstrap_repeats": args.cluster_bootstrap_repeats,
        "conclusion": decision,
        "robustly_closer_times": summary.loc[
            summary["all_seeds_closer"], "actual_time"
        ].tolist(),
        "robustly_farther_times": summary.loc[
            summary["all_seeds_farther"], "actual_time"
        ].tolist(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
