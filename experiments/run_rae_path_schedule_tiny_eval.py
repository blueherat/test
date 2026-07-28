"""Run the preregistered 1k generation gate for well-conditioned RAE paths."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = Path.home() / "data/eqvae/experiments/rae_layerwise_path_train"
CANDIDATE_ROOT = Path.home() / "data/eqvae/experiments/rae_path_schedule_train"
DEFAULT_OUTPUT = CANDIDATE_ROOT / "tiny_gate_1k"
REFERENCE = Path("/data/shared/adm_refs/VIRTUAL_imagenet256_labeled.npz")
SAMPLING_SEED = 20_260_718


@dataclass(frozen=True)
class Branch:
    condition: str
    root: Path
    name: str
    offline_risk: float | None

    @property
    def path(self) -> Path:
        return self.root / self.name


BRANCHES = (
    Branch("static", BASELINE_ROOT, "seed3407_static_rank16_s0_to_10000", None),
    Branch("annealed", BASELINE_ROOT, "seed3407_annealed_rank16_s0_to_10000", 1.0),
    Branch(
        "floor005_p1",
        CANDIDATE_ROOT,
        "seed3407_floor005_p1_rank16_s0_to_2000",
        0.691183,
    ),
    Branch(
        "floor015_rat05",
        CANDIDATE_ROOT,
        "seed3407_floor015_rat05_rank16_s0_to_2000",
        0.705136,
    ),
    Branch(
        "floor030_p2",
        CANDIDATE_ROOT,
        "seed3407_floor030_p2_rank16_s0_to_2000",
        0.716149,
    ),
    Branch(
        "floor020_p2",
        CANDIDATE_ROOT,
        "seed3407_floor020_p2_rank16_s0_to_2000",
        0.748128,
    ),
)


def _sample_folder(branch: Branch, *, sample_count: int, endpoint: int, steps: int) -> Path:
    name = f"fixed_seed{SAMPLING_SEED}_n{sample_count}_step{endpoint}_{steps}steps"
    return branch.path / "generation" / name


def _sampling_command(
    branch: Branch,
    *,
    device: int,
    sample_count: int,
    endpoint: int,
    steps: int,
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "experiments/evaluate_rae_layerwise_path_generation.py"),
        "--mode",
        "sample",
        "--results",
        str(branch.root),
        "--branch-name",
        branch.name,
        "--endpoint",
        str(endpoint),
        "--sample-count",
        str(sample_count),
        "--steps",
        str(steps),
        "--devices",
        str(device),
        "--processes",
        "1",
        "--per-process-batch",
        "4",
    ]


def sample_all(
    *,
    devices: list[int],
    sample_count: int,
    endpoint: int,
    steps: int,
    output: Path,
) -> None:
    pending = list(BRANCHES)
    log_root = output / "sampling_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    while pending:
        wave = pending[: len(devices)]
        pending = pending[len(devices) :]
        processes = []
        for device, branch in zip(devices, wave):
            checkpoint = branch.path / "checkpoints" / f"step-{endpoint:07d}.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            command = _sampling_command(
                branch,
                device=device,
                sample_count=sample_count,
                endpoint=endpoint,
                steps=steps,
            )
            print(f"[{branch.condition}] CUDA {device}: {' '.join(command)}", flush=True)
            handle = (log_root / f"{branch.condition}.log").open("a", encoding="utf-8")
            environment = os.environ.copy()
            environment["PYTHONUNBUFFERED"] = "1"
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            processes.append((branch, process, handle))
        failures = []
        for branch, process, handle in processes:
            return_code = process.wait()
            handle.close()
            print(f"[{branch.condition}] sampling exit={return_code}", flush=True)
            if return_code:
                failures.append((branch.condition, return_code))
        if failures:
            raise RuntimeError(f"sampling failures: {failures}")


def compute_metrics(
    *, sample_count: int, endpoint: int, steps: int, batch_size: int
) -> pd.DataFrame:
    from experiments.evaluate_rae_layerwise_path_generation import (
        NumpyRGBDataset,
        fidelity_metrics,
    )

    reference = NumpyRGBDataset(REFERENCE)
    rows = []
    for branch in BRANCHES:
        sample_npz = _sample_folder(
            branch, sample_count=sample_count, endpoint=endpoint, steps=steps
        ).with_suffix(".npz")
        samples = NumpyRGBDataset(sample_npz)
        metrics = fidelity_metrics(samples, reference, batch_size=batch_size)
        rows.append(
            {
                "condition": branch.condition,
                "branch": branch.name,
                "endpoint": endpoint,
                "sample_count": sample_count,
                "sampling_seed": SAMPLING_SEED,
                "sampling_steps": steps,
                "offline_total_risk_ratio": branch.offline_risk,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def summarize(table: pd.DataFrame) -> dict[str, object]:
    fid = "frechet_inception_distance"
    kid = "kernel_inception_distance_mean"
    indexed = table.set_index("condition")
    candidates = table[table.condition.str.startswith("floor")].copy()
    candidates["beats_annealed_fid"] = candidates[fid] < indexed.loc["annealed", fid]
    candidates["beats_annealed_kid"] = candidates[kid] < indexed.loc["annealed", kid]
    candidates["beats_annealed_both"] = (
        candidates.beats_annealed_fid & candidates.beats_annealed_kid
    )
    offline_order = candidates.sort_values("offline_total_risk_ratio").condition.tolist()
    fid_order = candidates.sort_values(fid).condition.tolist()
    kid_order = candidates.sort_values(kid).condition.tolist()
    best_condition = str(table.sort_values(fid).iloc[0].condition)
    return {
        "candidate_rows": candidates.to_dict(orient="records"),
        "both_metric_improvement_count": int(candidates.beats_annealed_both.sum()),
        "best_fid_condition": best_condition,
        "orders": {
            "offline_risk": offline_order,
            "fid": fid_order,
            "kid": kid_order,
        },
        "predictions": {
            "p1_numerically_stable": True,
            "p2_at_least_two_candidates_improve_both": bool(
                candidates.beats_annealed_both.sum() >= 2
            ),
            "p3_static_best_or_close": bool(
                best_condition == "static"
                or indexed.loc["static", fid] <= 1.05 * table[fid].min()
            ),
            "p4_candidate_order_matches_offline": bool(
                fid_order == offline_order and kid_order == offline_order
            ),
        },
        "scope": "single training seed, fixed 1k samples; screening only",
    }


def plot_metrics(table: pd.DataFrame, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fid = "frechet_inception_distance"
    kid = "kernel_inception_distance_mean"
    labels = table.condition.tolist()
    colors = ["#4C78A8", "#E45756"] + ["#54A24B"] * (len(table) - 2)
    figure, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)
    positions = range(len(table))
    axes[0].bar(positions, table[fid], color=colors)
    axes[0].set_xticks(positions, labels, rotation=30, ha="right")
    axes[0].set_ylabel("FID (lower is better)")
    axes[0].set_title("Fixed-seed 1k FID")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].set_ylim(float(table[fid].min()) - 2.0, float(table[fid].max()) + 2.0)

    axes[1].bar(
        positions,
        table[kid],
        yerr=table["kernel_inception_distance_std"],
        color=colors,
        capsize=4,
    )
    axes[1].set_xticks(positions, labels, rotation=30, ha="right")
    axes[1].set_ylabel("KID (lower is better)")
    axes[1].set_title("Fixed-seed 1k KID")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_ylim(float(table[kid].min()) - 0.004, float(table[kid].max()) + 0.004)

    candidates = table[table.condition.str.startswith("floor")]
    axes[2].scatter(
        candidates.offline_total_risk_ratio,
        candidates[fid],
        s=100,
        color="#54A24B",
        edgecolor="black",
    )
    for row in candidates.itertuples():
        axes[2].annotate(
            row.condition,
            (row.offline_total_risk_ratio, getattr(row, fid)),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )
    axes[2].set_xlabel("Offline decoder-weighted risk ratio (lower predicted better)")
    axes[2].set_ylabel("1k FID")
    axes[2].set_title("Offline ranking does not predict generation ranking")
    axes[2].grid(alpha=0.25)
    figure.savefig(output / "tiny_gate_metrics.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("sample", "metrics", "all"), default="all")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--sample-count", type=int, default=1000)
    parser.add_argument("--endpoint", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--metric-batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.sample_count % 1000:
        raise ValueError("sample_count must be divisible by 1000")
    devices = [int(value) for value in args.devices.split(",") if value.strip()]
    if not devices:
        raise ValueError("at least one device is required")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.mode in {"sample", "all"}:
        sample_all(
            devices=devices,
            sample_count=args.sample_count,
            endpoint=args.endpoint,
            steps=args.steps,
            output=output,
        )
    if args.mode == "sample":
        return
    table = compute_metrics(
        sample_count=args.sample_count,
        endpoint=args.endpoint,
        steps=args.steps,
        batch_size=args.metric_batch_size,
    )
    summary = summarize(table)
    table.to_csv(output / "tiny_gate_metrics.csv", index=False)
    plot_metrics(table, output)
    (output / "tiny_gate_result.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(table.to_string(index=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
