"""Summarize independent RAEv2 decoder-side AUC/FID audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def time_suffix(value: float) -> str:
    return f"t{float(value):.6f}".replace(".", "p")


def parse_named_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("run must be NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("run name cannot be empty")
    return name, Path(raw_path).expanduser()


def load_runs(named_runs: list[tuple[str, Path]]) -> pd.DataFrame:
    frames = []
    expected_times = None
    for name, root in named_runs:
        root = root.resolve()
        frame = pd.read_csv(root / "decoded_distribution_summary.csv")
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if int(manifest["heldout_pairs"]) < 100:
            raise ValueError(f"run {name!r} has fewer than 100 held-out pairs")
        times = tuple(frame["requested_time"].tolist())
        if expected_times is None:
            expected_times = times
        elif times != expected_times:
            raise ValueError("runs use different requested times")
        frame.insert(0, "run", name)
        frame["seed"] = int(manifest["seed"])
        frames.append(frame)
    if len({name for name, _ in named_runs}) != len(named_runs):
        raise ValueError("run names must be unique")
    return pd.concat(frames, ignore_index=True)


def summarize(per_run: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "auc_delta_ig_minus_full",
        "fid_real_delta_ig_minus_full",
        "fid_p_delta_ig_minus_full",
    )
    rows = []
    for (requested, actual), frame in per_run.groupby(
        ["requested_time", "actual_time"], sort=True
    ):
        row = {
            "requested_time": requested,
            "actual_time": actual,
            "runs": len(frame),
            "seeds_auc_significantly_closer": int(
                (frame["auc_delta_ci_high"] < 0).sum()
            ),
            "all_seeds_auc_closer": bool((frame["auc_delta_ci_high"] < 0).all()),
            "all_seeds_real_fid_better": bool(
                (frame["fid_real_delta_ig_minus_full"] < 0).all()
            ),
            "all_seeds_p_fid_closer": bool(
                (frame["fid_p_delta_ig_minus_full"] < 0).all()
            ),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = frame[metric].mean()
            row[f"{metric}_seed_std"] = frame[metric].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def load_clipping(named_runs: list[tuple[str, Path]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for name, root in named_runs:
        root = root.resolve()
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        world_size = int(manifest["world_size"])
        diagnostics = [
            json.loads(
                (root / f"decode_diagnostics_rank{rank:02d}.json").read_text(
                    encoding="utf-8"
                )
            )
            for rank in range(world_size)
        ]
        rank_counts = []
        for rank in range(world_size):
            with np.load(root / f"decoded_features_rank{rank:02d}.npz") as shard:
                rank_counts.append(int(shard["ids"].size))
        for requested_time in manifest["requested_times"]:
            suffix = time_suffix(float(requested_time))
            for condition in ("p", "full", "ig"):
                key = f"{condition}_{suffix}"
                values = [item[key] for item in diagnostics]
                total = sum(rank_counts)
                rows.append(
                    {
                        "run": name,
                        "seed": int(manifest["seed"]),
                        "requested_time": float(requested_time),
                        "condition": condition,
                        "raw_min": min(item["raw_min"] for item in values),
                        "raw_max": max(item["raw_max"] for item in values),
                        "clipped_low_fraction": sum(
                            item["clipped_low_fraction"] * count
                            for item, count in zip(values, rank_counts)
                        ) / total,
                        "clipped_high_fraction": sum(
                            item["clipped_high_fraction"] * count
                            for item, count in zip(values, rank_counts)
                        ) / total,
                    }
                )
    per_run = pd.DataFrame(rows)
    per_run["clipped_total_fraction"] = (
        per_run["clipped_low_fraction"] + per_run["clipped_high_fraction"]
    )
    deltas = []
    for (run, seed, requested_time), frame in per_run.groupby(
        ["run", "seed", "requested_time"]
    ):
        indexed = frame.set_index("condition")
        deltas.append(
            {
                "run": run,
                "seed": seed,
                "requested_time": requested_time,
                "clipped_low_delta_ig_minus_full": (
                    indexed.loc["ig", "clipped_low_fraction"]
                    - indexed.loc["full", "clipped_low_fraction"]
                ),
                "clipped_high_delta_ig_minus_full": (
                    indexed.loc["ig", "clipped_high_fraction"]
                    - indexed.loc["full", "clipped_high_fraction"]
                ),
                "clipped_total_delta_ig_minus_full": (
                    indexed.loc["ig", "clipped_total_fraction"]
                    - indexed.loc["full", "clipped_total_fraction"]
                ),
            }
        )
    return per_run, pd.DataFrame(deltas)


def plot(per_run: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    specs = (
        ("auc_delta_ig_minus_full", "Decoded AUC delta"),
        ("fid_real_delta_ig_minus_full", "FID-to-real delta"),
        ("fid_p_delta_ig_minus_full", "FID-to-D(p_t) delta"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    colors = ("#3569a8", "#c94f3d", "#4b8f5a", "#8a5aa8")
    for axis, (metric, title) in zip(axes, specs):
        for index, (name, frame) in enumerate(per_run.groupby("run", sort=False)):
            frame = frame.sort_values("actual_time", ascending=False)
            axis.plot(
                frame["actual_time"], frame[metric], "o-",
                color=colors[index % len(colors)], alpha=0.75, label=name,
            )
        mean = summary.sort_values("actual_time", ascending=False)
        axis.plot(
            mean["actual_time"], mean[f"{metric}_mean"], "s-",
            color="#111111", linewidth=2.8, label="seed mean",
        )
        axis.axhline(0.0, color="#333333", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("Solver time t (sampling: 1 to 0)")
        axis.invert_xaxis()
        axis.grid(True, alpha=0.22)
        axis.legend(frameon=False)
    fig.suptitle("RAEv2 Internal Guidance After Decoder: Cross-seed Deltas\n"
                 "Negative means IG is closer/better")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=parse_named_run, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    per_run = load_runs(args.run)
    summary = summarize(per_run)
    clipping, clipping_deltas = load_clipping(args.run)
    per_run.to_csv(output_dir / "per_seed_decoded_metrics.csv", index=False)
    summary.to_csv(output_dir / "cross_seed_decoded_summary.csv", index=False)
    clipping.to_csv(output_dir / "per_seed_decode_clipping.csv", index=False)
    clipping_deltas.to_csv(output_dir / "per_seed_decode_clipping_deltas.csv", index=False)
    plot(per_run, summary, output_dir / "cross_seed_decoder_reversal.png")
    non_null = summary[summary["requested_time"] < 1.0]
    report = {
        "protocol": "raev2_decoded_distribution_cross_seed_v1",
        "runs": {name: str(path.resolve()) for name, path in args.run},
        "all_non_null_real_fid_better": bool(
            non_null["all_seeds_real_fid_better"].all()
        ),
        "all_non_null_p_fid_closer": bool(non_null["all_seeds_p_fid_closer"].all()),
        "auc_reversal_times": non_null.loc[
            non_null["all_seeds_auc_closer"], "actual_time"
        ].tolist(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
