"""Summarize the equal-budget all-time versus high-noise latent priors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mnist_spectral_rollout_toy import load_mnist_tensors


def load_prior_runs(prior_root: Path) -> pd.DataFrame:
    rows = []
    for run in sorted(prior_root.iterdir()):
        if not run.is_dir() or not (run / "summary.json").is_file():
            continue
        summary = json.loads((run / "summary.json").read_text())
        config = json.loads((run / "config.json").read_text())
        summary["run"] = str(run)
        summary["stage2_run"] = config["stage2_run"]
        rows.append(summary)
    table = pd.DataFrame(rows)
    if set(table.get("mode", [])) != {"all_time", "high_noise"}:
        raise ValueError("prior root must contain all_time and high_noise runs")
    return table.sort_values(["seed", "mode"]).reset_index(drop=True)


def _stage2_metric(stage2_run: Path, branch: str, metric: str) -> float:
    table = pd.read_csv(stage2_run / "rollout.csv")
    return float(table.loc[table.branch == branch, metric].iloc[0])


def gate_table(priors: pd.DataFrame) -> pd.DataFrame:
    records = []
    for seed, frame in priors.groupby("seed", sort=True):
        models = frame.set_index("mode")
        all_time = models.loc["all_time"]
        high = models.loc["high_noise"]
        stage2_root = Path(high.stage2_run).parent
        none_runs = sorted(stage2_root.glob(f"none_seed{int(seed)}_*"))
        if len(none_runs) != 1:
            raise ValueError(f"expected one none stage2 run for seed {seed}")
        none_fid = _stage2_metric(none_runs[0], "real", "feature_fid")
        all_oracle_fid = _stage2_metric(Path(all_time.stage2_run), "real", "feature_fid")
        high_oracle_fid = _stage2_metric(Path(high.stage2_run), "real", "feature_fid")
        high_fid_ratio = float(high.image_feature_fid / all_time.image_feature_fid)
        high_swd_ratio = float(high.latent_swd / all_time.latent_swd)
        entropy_pass = bool(
            min(
                all_time.latent_class_entropy,
                high.latent_class_entropy,
                all_time.image_class_entropy,
                high.image_class_entropy,
            )
            >= np.log(8.0)
        )
        two_stage_pass = bool(
            high.image_feature_fid <= 0.8 * none_fid
            and all_time.image_feature_fid <= 0.8 * none_fid
        )
        quality_guardrail = high_fid_ratio <= 1.10
        method_value = high_fid_ratio <= 0.95 or high_swd_ratio <= 0.90
        records.append(
            {
                "seed": int(seed),
                "none_image_feature_fid": none_fid,
                "all_oracle_image_fid": all_oracle_fid,
                "high_oracle_image_fid": high_oracle_fid,
                "high_over_all_oracle_fid": high_oracle_fid / all_oracle_fid,
                "all_image_feature_fid": float(all_time.image_feature_fid),
                "high_image_feature_fid": float(high.image_feature_fid),
                "all_prior_fid_increment": float(all_time.image_feature_fid - all_oracle_fid),
                "high_prior_fid_increment": float(high.image_feature_fid - high_oracle_fid),
                "high_over_all_image_fid": high_fid_ratio,
                "all_latent_swd": float(all_time.latent_swd),
                "high_latent_swd": float(high.latent_swd),
                "high_over_all_latent_swd": high_swd_ratio,
                "entropy_pass": entropy_pass,
                "two_stage_pass": two_stage_pass,
                "quality_guardrail": bool(quality_guardrail),
                "method_value": bool(method_value),
                "end_to_end_gate_pass": bool(
                    entropy_pass and two_stage_pass and quality_guardrail and method_value
                ),
            }
        )
    return pd.DataFrame(records)


def _sample_pixels(row: pd.Series) -> torch.Tensor:
    state = torch.load(Path(row.run) / "state.pt", map_location="cpu", weights_only=True)
    stage2 = Path(row.stage2_run)
    stage2_config = json.loads((stage2 / "config.json").read_text())
    data = load_mnist_tensors(
        stage2_config["data_root"],
        stage2_config["train_size"],
        stage2_config["test_size"],
        stage2_config["seed"],
    )
    mean, std = data["normalization"]["mean"], data["normalization"]["std"]
    return (state["samples"][:36, 0] * std + mean).clamp(0.0, 1.0)


def plot_summary(priors: pd.DataFrame, gates: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    width = 0.35
    positions = np.arange(len(gates))
    axes[0, 0].bar(positions - width, gates.none_image_feature_fid, width, label="none")
    axes[0, 0].bar(positions, gates.all_image_feature_fid, width, label="all_time")
    axes[0, 0].bar(positions + width, gates.high_image_feature_fid, width, label="high_noise")
    axes[0, 0].set_xticks(positions, [f"seed {seed}" for seed in gates.seed])
    axes[0, 0].set_ylabel("Feature FID")
    axes[0, 0].set_title("End-to-end image distribution")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].grid(axis="y", alpha=0.25)

    axes[0, 1].bar(positions - width / 2, gates.all_latent_swd, width, label="all_time")
    axes[0, 1].bar(positions + width / 2, gates.high_latent_swd, width, label="high_noise")
    axes[0, 1].set_xticks(positions, [f"seed {seed}" for seed in gates.seed])
    axes[0, 1].set_ylabel("Latent SWD")
    axes[0, 1].set_title("Prior fit")
    axes[0, 1].legend(frameon=False)
    axes[0, 1].grid(axis="y", alpha=0.25)

    axes[0, 2].bar(positions - width / 2, gates.high_over_all_image_fid, width, label="image FID")
    axes[0, 2].bar(positions + width / 2, gates.high_over_all_latent_swd, width, label="latent SWD")
    axes[0, 2].axhline(1.0, color="black", linewidth=1)
    axes[0, 2].set_xticks(positions, [f"seed {seed}" for seed in gates.seed])
    axes[0, 2].set_ylabel("high_noise / all_time")
    axes[0, 2].set_title("Explicit gate value")
    axes[0, 2].legend(frameon=False)
    axes[0, 2].grid(axis="y", alpha=0.25)

    axes[0, 3].axis("off")
    sample_column = 0
    for seed in sorted(priors.seed.unique()):
        rows = priors[priors.seed == seed].set_index("mode")
        all_images = _sample_pixels(rows.loc["all_time"])
        high_images = _sample_pixels(rows.loc["high_noise"])
        for name, images in (("all_time", all_images), ("high_noise", high_images)):
            axis = axes[1, sample_column]
            grid = images.reshape(6, 6, 28, 28).permute(0, 2, 1, 3).reshape(168, 168)
            axis.imshow(grid, cmap="gray", vmin=0.0, vmax=1.0)
            axis.set_title(f"{name}, seed {seed}")
            axis.axis("off")
            sample_column += 1
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    priors = load_prior_runs(args.prior_root)
    gates = gate_table(priors)
    args.output.mkdir(parents=True, exist_ok=True)
    priors.to_csv(args.output / "prior_summary.csv", index=False)
    gates.to_csv(args.output / "prior_gates.csv", index=False)
    plot_summary(priors, gates, args.output / "prior_summary.png")
    print(gates.to_string(index=False))
    print(f"\nEnter end-to-end training: {bool(gates.end_to_end_gate_pass.all())}")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
