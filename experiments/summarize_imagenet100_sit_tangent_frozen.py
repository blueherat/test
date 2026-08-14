#!/usr/bin/env python3
"""Combine the paired 800K tangent-versus-frozen mechanism screen."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.finite_guidance_dynamics import (
    linearity_metrics,
    sample_cosine,
)


FAMILIES = ("x800", "v500")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_results(root: Path, num_samples: int, seed: int) -> tuple[list[dict], dict]:
    rows: list[dict[str, object]] = []
    summaries: dict[str, dict] = {}
    for family in FAMILIES:
        run_dir = root / "tangent_frozen" / family / f"n{num_samples}_seed{seed}"
        metrics_path = run_dir / "metrics.csv"
        summary_path = run_dir / "summary.json"
        if not metrics_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(f"incomplete tangent-frozen run: {run_dir}")
        for row in _read_csv(metrics_path):
            rows.append({"family": family, **row})
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        # Raw shard locations are machine-specific and remain in each run's
        # local summary. The combined artifact is intended to be portable.
        summary.pop("trajectory_shards", None)
        summaries[family] = summary
    return rows, summaries


def _gamma_index(values: torch.Tensor, target: float) -> int:
    distance = (values.float() - float(target)).abs()
    index = int(distance.argmin())
    if float(distance[index]) > 1e-6:
        raise ValueError(f"gamma {target} is missing from {values.tolist()}")
    return index


def _load_sample_metrics(run_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    shard_paths = sorted((run_dir / "trajectory_shards").glob("batch_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"no trajectory shards found in {run_dir}")
    shards = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in shard_paths
    ]
    gammas = shards[0]["gammas"]
    gamma_one = _gamma_index(gammas, 1.0)
    baseline = torch.cat([shard["baseline"] for shard in shards])
    tangent = torch.cat([shard["tangent"] for shard in shards])
    output: dict[str, dict[str, np.ndarray]] = {}
    endpoints: dict[str, torch.Tensor] = {}
    for response in ("frozen", "closed"):
        endpoint = torch.cat(
            [shard[response][gamma_one] for shard in shards],
        )
        endpoints[response] = endpoint
        output[response] = {
            name: values.detach().double().numpy()
            for name, values in linearity_metrics(
                baseline,
                endpoint,
                tangent,
                gamma=1.0,
            ).items()
        }
    output["paired_endpoint"] = {
        "frozen_closed_cosine": sample_cosine(
            endpoints["frozen"] - baseline,
            endpoints["closed"] - baseline,
        )
        .double()
        .numpy(),
    }
    return output


def _bootstrap_mean(values: np.ndarray, rng: np.random.Generator, reps: int) -> dict:
    if values.ndim != 1:
        raise ValueError("bootstrap input must be one-dimensional")
    indices = rng.integers(0, len(values), size=(reps, len(values)))
    bootstrapped = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": [
            float(np.quantile(bootstrapped, 0.025)),
            float(np.quantile(bootstrapped, 0.975)),
        ],
        "probability_gt_zero": float((bootstrapped > 0.0).mean()),
    }


def paired_gamma_one_analysis(
    root: Path,
    num_samples: int,
    seed: int,
    *,
    bootstrap_reps: int,
) -> dict:
    metrics = {
        family: _load_sample_metrics(
            root / "tangent_frozen" / family / f"n{num_samples}_seed{seed}"
        )
        for family in FAMILIES
    }
    rng = np.random.default_rng(seed + 17)
    comparisons = {
        "v500_minus_x800_frozen_cosine": (
            metrics["v500"]["frozen"]["cosine"]
            - metrics["x800"]["frozen"]["cosine"]
        ),
        "x800_minus_v500_frozen_relative_residual": (
            metrics["x800"]["frozen"]["relative_residual"]
            - metrics["v500"]["frozen"]["relative_residual"]
        ),
    }
    for family in FAMILIES:
        comparisons[f"{family}_frozen_minus_closed_cosine"] = (
            metrics[family]["frozen"]["cosine"]
            - metrics[family]["closed"]["cosine"]
        )
        comparisons[f"{family}_closed_minus_frozen_relative_residual"] = (
            metrics[family]["closed"]["relative_residual"]
            - metrics[family]["frozen"]["relative_residual"]
        )
    return {
        "bootstrap_reps": bootstrap_reps,
        "comparisons": {
            name: _bootstrap_mean(values, rng, bootstrap_reps)
            for name, values in comparisons.items()
        },
        "frozen_closed_endpoint_cosine": {
            family: _bootstrap_mean(
                metrics[family]["paired_endpoint"]["frozen_closed_cosine"],
                rng,
                bootstrap_reps,
            )
            for family in FAMILIES
        },
    }


def plot_results(rows: list[dict[str, object]], path: Path) -> None:
    positive = [row for row in rows if float(row["gamma"]) > 0]
    styles = {"frozen": "-", "closed": "--"}
    colors = {"x800": "#0072B2", "v500": "#D55E00"}
    figure, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    metrics = (
        ("cosine_mean", "Endpoint response cosine", (0.0, 1.02)),
        ("relative_residual_mean", "Relative nonlinear residual", None),
        ("magnitude_ratio_mean", "Actual / tangent RMS", None),
    )
    for axis, (metric, title, limits) in zip(axes, metrics, strict=True):
        for family in FAMILIES:
            for response in ("frozen", "closed"):
                selected = sorted(
                    (
                        row
                        for row in positive
                        if row["family"] == family and row["response"] == response
                    ),
                    key=lambda row: float(row["gamma"]),
                )
                axis.plot(
                    [float(row["gamma"]) for row in selected],
                    [float(row[metric]) for row in selected],
                    marker="o",
                    linestyle=styles[response],
                    color=colors[family],
                    label=f"{family} {response}",
                )
        axis.set_xscale("log")
        axis.set_xlabel("guidance gamma")
        axis.set_title(title)
        axis.grid(alpha=0.25)
        if limits is not None:
            axis.set_ylim(*limits)
    axes[0].legend(frameon=False, fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main(args: argparse.Namespace) -> None:
    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve() if args.output else root / "summary"
    rows, summaries = load_results(root, args.num_samples, args.seed)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, output / "tangent_frozen_metrics.csv")
    plot_results(rows, output / "tangent_frozen_gamma_sweep.png")
    payload = {
        "format": "eqvae_sit800_tangent_frozen_summary_v1",
        "num_samples": args.num_samples,
        "seed": args.seed,
        "families": summaries,
        "paired_gamma_one": paired_gamma_one_analysis(
            root,
            args.num_samples,
            args.seed,
            bootstrap_reps=args.bootstrap_reps,
        ),
    }
    (output / "tangent_frozen_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/tangent_frozen_800k_v1"
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--bootstrap-reps", type=int, default=20000)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
