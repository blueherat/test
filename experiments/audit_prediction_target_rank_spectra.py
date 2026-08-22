#!/usr/bin/env python3
"""Audit the affine output-rank burden of x/v/epsilon population targets.

The v4 oracle predicts the conditional clean mean for

    z_t = (1 - t) x + t epsilon.

From that single prediction we construct the population-optimal x, velocity,
and epsilon outputs.  For each target we then compute the best rank-r affine
approximation error from the centered output covariance.  No model is trained
by this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.run_prediction_target_extrapolation_toy_v4 import (
    CurvedEmbedding,
    DenoiseMLP,
    parse_float_list,
    parse_int_list,
    sample_spiral_2d,
    stable_seed,
    tag_float,
)


def save_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot save an empty table")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def covariance_spectrum(values: torch.Tensor) -> torch.Tensor:
    """Return descending eigenvalues of the best affine output subspace."""
    centered = values.double() - values.double().mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    return torch.linalg.eigvalsh(covariance).flip(0).clamp_min(0)


def spectrum_summary(eigenvalues: torch.Tensor, rank: int) -> dict[str, float]:
    total = eigenvalues.sum().clamp_min(torch.finfo(eigenvalues.dtype).tiny)
    squared = eigenvalues.square().sum().clamp_min(torch.finfo(eigenvalues.dtype).tiny)
    rank = min(max(int(rank), 0), len(eigenvalues))
    tail = eigenvalues[rank:].sum()
    cumulative = eigenvalues.cumsum(0) / total

    def coverage_rank(level: float) -> int:
        return int(torch.searchsorted(cumulative, torch.tensor(level, device=cumulative.device)).item() + 1)

    return {
        "total_variance": float(total),
        "variance_per_dimension": float(total / len(eigenvalues)),
        "effective_rank": float(total.square() / squared),
        "rank_95": coverage_rank(0.95),
        "rank_99": coverage_rank(0.99),
        "tail_variance": float(tail),
        "tail_mse_per_dimension": float(tail / len(eigenvalues)),
        "tail_fraction": float(tail / total),
    }


def native_to_velocity_gain(target: str, time: float) -> float:
    if target == "x":
        return 1.0 / time
    if target == "v":
        return 1.0
    if target == "eps":
        return 1.0 / (1.0 - time)
    raise ValueError(target)


@torch.inference_mode()
def audit_setting(
    *,
    root: Path,
    manifest_args: dict,
    seed: int,
    curvature: float,
    times: list[float],
    ranks: list[int],
    samples: int,
    batch_size: int,
    device: torch.device,
) -> list[dict[str, float | int | str]]:
    D = int(manifest_args["dims"][0])
    embedding = CurvedEmbedding(
        D,
        curvature=curvature,
        frequency_scale=float(manifest_args["frequency_scale"]),
        seed=stable_seed(seed, D, int(curvature * 10000), 41),
        device=device,
        scale_mode=str(manifest_args["scale_mode"]),
    )
    oracle = DenoiseMLP(
        D,
        hidden=int(manifest_args["oracle_hidden_dim"]),
        depth=int(manifest_args["oracle_depth"]),
        time_dim=int(manifest_args["time_dim"]),
    ).to(device)
    checkpoint = (
        root
        / f"seed{seed}"
        / f"D{D}"
        / f"curv{tag_float(curvature)}"
        / f"scale_{manifest_args['scale_mode']}"
        / "loss_v"
        / "oracle_x.pt"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    oracle.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    oracle.eval()

    rows: list[dict[str, float | int | str]] = []
    for time_index, time in enumerate(times):
        outputs = {target: [] for target in ("x", "v", "eps")}
        generator = torch.Generator(device=device.type)
        generator.manual_seed(stable_seed(seed, D, int(curvature * 10000), time_index, 501))
        paired_clean_mse = 0.0
        for start in range(0, samples, batch_size):
            n = min(batch_size, samples - start)
            u = sample_spiral_2d(
                n,
                device=device,
                jitter=float(manifest_args["data_jitter"]),
                generator=generator,
            )
            clean = embedding.embed(u)
            eps = torch.randn(clean.shape, device=device, generator=generator)
            t = torch.full((n,), float(time), device=device)
            state = (1.0 - t[:, None]) * clean + t[:, None] * eps
            clean_mean = oracle(state, t)
            velocity_mean = (state - clean_mean) / t[:, None]
            eps_mean = state + (1.0 - t[:, None]) * velocity_mean
            outputs["x"].append(clean_mean.cpu())
            outputs["v"].append(velocity_mean.cpu())
            outputs["eps"].append(eps_mean.cpu())
            paired_clean_mse += float((clean_mean - clean).square().sum().cpu())

        paired_clean_mse /= samples * D
        for target, parts in outputs.items():
            spectrum = covariance_spectrum(torch.cat(parts, dim=0))
            for rank in ranks:
                summary = spectrum_summary(spectrum, rank)
                gain = native_to_velocity_gain(target, time)
                rows.append(
                    {
                        "seed": seed,
                        "D": D,
                        "curvature": curvature,
                        "time": time,
                        "target": target,
                        "rank": rank,
                        "paired_clean_mse": paired_clean_mse,
                        "native_to_velocity_gain": gain,
                        "native_to_velocity_squared_gain": gain**2,
                        **summary,
                        "velocity_tail_mse_per_dimension": (
                            summary["tail_mse_per_dimension"] * gain**2
                        ),
                    }
                )
    return rows


def plot_results(rows: list[dict[str, float | int | str]], output: Path) -> None:
    targets = ("x", "v", "eps")
    colors = {"x": "#1f77b4", "v": "#2ca02c", "eps": "#d62728"}
    seeds = sorted({int(row["seed"]) for row in rows})
    curvatures = sorted({float(row["curvature"]) for row in rows})
    ranks = sorted({int(row["rank"]) for row in rows})
    times = sorted({float(row["time"]) for row in rows})

    fig, axes = plt.subplots(len(curvatures), 2, figsize=(12, 4 * len(curvatures)), squeeze=False)
    for row_index, curvature in enumerate(curvatures):
        for target in targets:
            values = []
            for time in times:
                subset = [
                    float(row["effective_rank"])
                    for row in rows
                    if float(row["curvature"]) == curvature
                    and str(row["target"]) == target
                    and float(row["time"]) == time
                    and int(row["rank"]) == ranks[0]
                ]
                values.append(float(np.mean(subset)))
            axes[row_index, 0].plot(times, values, marker="o", label=target, color=colors[target])

            tail_values = []
            selected_rank = min(ranks, key=lambda value: abs(value - 64))
            for time in times:
                subset = [
                    float(row["tail_fraction"])
                    for row in rows
                    if float(row["curvature"]) == curvature
                    and str(row["target"]) == target
                    and float(row["time"]) == time
                    and int(row["rank"]) == selected_rank
                ]
                tail_values.append(float(np.mean(subset)))
            axes[row_index, 1].plot(times, tail_values, marker="o", label=target, color=colors[target])

        axes[row_index, 0].set_title(f"curvature={curvature}: effective rank ({len(seeds)} seeds)")
        axes[row_index, 0].set_xlabel("noise time t")
        axes[row_index, 0].set_ylabel("participation-ratio rank")
        axes[row_index, 1].set_title(f"curvature={curvature}: tail fraction after rank {selected_rank}")
        axes[row_index, 1].set_xlabel("noise time t")
        axes[row_index, 1].set_ylabel("unexplained variance fraction")
        axes[row_index, 1].set_ylim(bottom=0)
        for axis in axes[row_index]:
            axis.grid(alpha=0.25)
            axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--v4-root",
        type=Path,
        default=Path("/home/zhoushunyu/data/eqvae/experiments/prediction_target_toy_v4_main"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=parse_int_list, default=parse_int_list("20260807,20260808,20260809"))
    parser.add_argument("--curvatures", type=parse_float_list, default=parse_float_list("0,0.5,1"))
    parser.add_argument("--times", type=parse_float_list, default=parse_float_list("0.1,0.3,0.5,0.7,0.9"))
    parser.add_argument("--ranks", type=parse_int_list, default=parse_int_list("2,4,8,16,32,64,128,256"))
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.v4_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest_args = manifest["args"]
    device = torch.device(args.device)

    rows: list[dict[str, float | int | str]] = []
    missing: list[str] = []
    for seed in args.seeds:
        for curvature in args.curvatures:
            print(f"[audit] seed={seed} curvature={curvature}", flush=True)
            try:
                rows.extend(
                    audit_setting(
                        root=root,
                        manifest_args=manifest_args,
                        seed=seed,
                        curvature=curvature,
                        times=args.times,
                        ranks=args.ranks,
                        samples=args.samples,
                        batch_size=args.batch_size,
                        device=device,
                    )
                )
            except FileNotFoundError as error:
                missing.append(str(error))
                print(f"[audit] missing checkpoint; skipping: {error}", flush=True)

    if not rows:
        raise RuntimeError("no available oracle checkpoints matched the request")

    save_csv(output / "rank_spectrum_audit.csv", rows)
    plot_results(rows, output / "rank_spectrum_audit.png")
    run_manifest = {
        "definition": "best affine rank-r approximation of oracle conditional target means",
        "path": "z_t=(1-t)x+t epsilon",
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "missing_checkpoints": missing,
    }
    (output / "manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    print(f"[done] wrote {len(rows)} rows to {output}", flush=True)


if __name__ == "__main__":
    main()
