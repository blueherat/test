"""Pool SiT internal-guidance direction controls across independent seeds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_raev2_ig_impulse_response import (
    _load_condition,
    _load_small_shards,
    bootstrap_mean_interval,
)
from experiments.run_sit_ig_direction_specificity import DirectionCondition
from experiments.run_sit_ig_endpoint_dynamics import sample_rms


PROTOCOL = "sit_ig_direction_specificity_summary_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260816)
    return parser.parse_args()


def load_manifest(run_dir: Path) -> dict[str, Any]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError(f"run is incomplete: {run_dir}")
    if manifest.get("protocol") != "sit_ig_direction_specificity_v1":
        raise RuntimeError(f"unexpected protocol: {run_dir}")
    return manifest


def validate_compatible_manifests(manifests: list[dict[str, Any]]) -> None:
    if len(manifests) < 2:
        raise ValueError("at least two independent runs are required")
    keys = (
        "checkpoint_sha256",
        "model_name",
        "encoder_depth",
        "state_key",
        "latent_size",
        "num_steps",
        "solver_grid",
        "gamma",
        "probe_count",
        "conditions",
        "cfg_scale",
        "sampler",
    )
    reference = manifests[0]
    for index, manifest in enumerate(manifests[1:], start=1):
        changed = [key for key in keys if manifest.get(key) != reference.get(key)]
        if changed:
            raise RuntimeError(f"run {index} is incompatible: {changed}")
    seeds = [int(manifest["seed"]) for manifest in manifests]
    if len(set(seeds)) != len(seeds):
        raise RuntimeError("runs must use distinct experiment seeds")


def summarize(values: np.ndarray, *, repeats: int, seed: int) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    low, high = bootstrap_mean_interval(values, repeats=repeats, seed=seed)
    return {
        "mean": float(values.mean()),
        "ci_low": low,
        "ci_high": high,
        "median": float(np.median(values)),
        "q95": float(np.quantile(values, 0.95)),
    }


def build_summary_manifest(
    manifests: list[dict[str, Any]],
    run_dirs: list[Path],
    *,
    bootstrap_repeats: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "status": "complete",
        "run_dirs": [str(path) for path in run_dirs],
        "run_seeds": [int(manifest["seed"]) for manifest in manifests],
        "total_samples": int(sum(manifest["samples"] for manifest in manifests)),
        "bootstrap_repeats": bootstrap_repeats,
        "seed": seed,
    }


def load_pair_gains(
    run_dir: Path,
    manifest: dict[str, Any],
    conditions: tuple[DirectionCondition, ...],
) -> dict[str, np.ndarray]:
    samples, world_size = int(manifest["samples"]), int(manifest["world_size"])
    unit_norms = _load_small_shards(
        run_dir,
        filename="unit_injected_norm_rank{rank:02d}.npy",
        samples=samples,
        world_size=world_size,
    )[:, :, 0]
    result: dict[str, np.ndarray] = {}
    for pair in sorted({item.pair_name for item in conditions if item.pair_name}):
        positive_index = next(
            index
            for index, item in enumerate(conditions)
            if item.pair_name == pair and item.sign > 0
        )
        negative_index = next(
            index
            for index, item in enumerate(conditions)
            if item.pair_name == pair and item.sign < 0
        )
        gamma = conditions[positive_index].gamma
        positive = _load_condition(
            run_dir,
            condition_index=positive_index,
            samples=samples,
            world_size=world_size,
        ).astype(np.float64)
        negative = _load_condition(
            run_dir,
            condition_index=negative_index,
            samples=samples,
            world_size=world_size,
        ).astype(np.float64)
        derivative = 0.5 * (positive - negative) / gamma
        unit_norm = 0.5 * (
            unit_norms[:, positive_index] + unit_norms[:, negative_index]
        )
        result[str(pair)] = sample_rms(derivative) / np.maximum(unit_norm, 1e-30)
    return result


def main() -> None:
    args = parse_args()
    if args.bootstrap_repeats <= 0:
        raise ValueError("bootstrap repeats must be positive")
    run_dirs = [path.expanduser().resolve() for path in args.run_dirs]
    manifests = [load_manifest(path) for path in run_dirs]
    validate_compatible_manifests(manifests)
    conditions = tuple(
        DirectionCondition(**row) for row in manifests[0]["conditions"]
    )
    grid = np.asarray(manifests[0]["solver_grid"], dtype=np.float64)
    per_run = [
        load_pair_gains(run_dir, manifest, conditions)
        for run_dir, manifest in zip(run_dirs, manifests)
    ]

    gain_rows = []
    pair_names = sorted(per_run[0])
    for pair_index, pair in enumerate(pair_names):
        item = next(condition for condition in conditions if condition.pair_name == pair)
        values = np.concatenate([result[pair] for result in per_run])
        gain_rows.append(
            {
                "pair_name": pair,
                "family": item.family,
                "step": item.step,
                "time": float(grid[int(item.step)]),
                "probe_index": item.probe_index,
                "runs": len(run_dirs),
                "samples": len(values),
                **summarize(
                    values,
                    repeats=args.bootstrap_repeats,
                    seed=args.seed + 1009 * pair_index,
                ),
            }
        )
    gain_frame = pd.DataFrame(gain_rows).sort_values(
        ["step", "family", "probe_index"]
    )

    specificity_rows = []
    active_steps = sorted(
        {int(item.step) for item in conditions if item.step is not None}
    )
    for step_index, step in enumerate(active_steps):
        ratios = []
        ig_values = []
        random_values = []
        for result in per_run:
            ig = result[f"step_{step:03d}_ig"]
            random = np.stack(
                [
                    result[pair]
                    for pair in pair_names
                    if pair.startswith(f"step_{step:03d}_random_")
                ],
                axis=1,
            ).mean(axis=1)
            ratios.append(ig / np.maximum(random, 1e-30))
            ig_values.append(ig)
            random_values.append(random)
        ratio = np.concatenate(ratios)
        ig = np.concatenate(ig_values)
        random = np.concatenate(random_values)
        specificity_rows.append(
            {
                "step": step,
                "time": float(grid[step]),
                "runs": len(run_dirs),
                "samples": len(ratio),
                "probe_count": int(manifests[0]["probe_count"]),
                "ig_gain_mean": float(ig.mean()),
                "random_gain_mean": float(random.mean()),
                **{
                    f"ig_over_random_{key}": value
                    for key, value in summarize(
                        ratio,
                        repeats=args.bootstrap_repeats,
                        seed=args.seed + 5003 * step_index,
                    ).items()
                },
                "fraction_ig_above_random_mean": float(np.mean(ig > random)),
            }
        )
    specificity = pd.DataFrame(specificity_rows)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gain_frame.to_csv(output_dir / "direction_gain_pooled.csv", index=False)
    specificity.to_csv(output_dir / "direction_specificity_pooled.csv", index=False)
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(specificity.time, specificity.ig_gain_mean, "o-", label="IG gap")
    axis.plot(
        specificity.time,
        specificity.random_gain_mean,
        "s-",
        label="matched random orthogonal",
    )
    axis.invert_xaxis()
    axis.set(
        title="Pooled SiT endpoint gain by intervention direction",
        xlabel="solver time t",
        ylabel="endpoint derivative / injected norm",
    )
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "direction_specificity_pooled.png", dpi=180)
    plt.close(figure)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            build_summary_manifest(
                manifests,
                run_dirs,
                bootstrap_repeats=args.bootstrap_repeats,
                seed=args.seed,
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(specificity.to_string(index=False))


if __name__ == "__main__":
    main()
