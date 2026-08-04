"""Pool completed SiT-IG endpoint-dynamics runs across independent seeds."""

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

from experiments.run_raev2_ig_impulse_response import _load_condition, _load_small_shards
from experiments.run_sit_ig_endpoint_dynamics import (
    Schedule,
    interaction_metrics,
    sample_cosine,
    sample_rms,
    summarize,
)


PROTOCOL = "sit_ig_endpoint_validation_summary_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260814)
    return parser.parse_args()


def load_manifest(run_dir: Path) -> dict[str, Any]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError(f"run is incomplete: {run_dir}")
    if manifest.get("protocol") != "sit_ig_endpoint_dynamics_v1":
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
        "windows",
        "schedules",
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


def pooled_row(
    metrics: dict[str, np.ndarray],
    *,
    repeats: int,
    seed: int,
) -> dict[str, float]:
    row: dict[str, float] = {}
    for index, (name, values) in enumerate(metrics.items()):
        row.update(
            {
                f"{name}_{key}": value
                for key, value in summarize(
                    np.asarray(values), repeats=repeats, seed=seed + 1009 * index
                ).items()
            }
        )
    return row


def main() -> None:
    args = parse_args()
    if args.bootstrap_repeats <= 0:
        raise ValueError("bootstrap repeats must be positive")
    run_dirs = [path.expanduser().resolve() for path in args.run_dirs]
    manifests = [load_manifest(path) for path in run_dirs]
    validate_compatible_manifests(manifests)
    schedules = tuple(Schedule(**row) for row in manifests[0]["schedules"])
    windows = tuple(tuple(int(value) for value in row) for row in manifests[0]["windows"])
    grid = np.asarray(manifests[0]["solver_grid"], dtype=np.float64)
    by_name = {item.name: index for index, item in enumerate(schedules)}

    signed_values: dict[str, dict[str, list[np.ndarray]]] = {}
    derivatives: dict[tuple[int, float], list[np.ndarray]] = {}
    interaction_values: dict[tuple[int, int], dict[str, list[np.ndarray]]] = {}
    total_samples = 0
    for run_dir, manifest in zip(run_dirs, manifests):
        samples, world_size = int(manifest["samples"]), int(manifest["world_size"])
        total_samples += samples
        baseline = _load_condition(
            run_dir, condition_index=0, samples=samples, world_size=world_size
        ).astype(np.float64)
        stats = _load_small_shards(
            run_dir,
            filename="injection_stats_rank{rank:02d}.npy",
            samples=samples,
            world_size=world_size,
        )
        pair_names = sorted({item.pair_name for item in schedules if item.pair_name})
        for pair in pair_names:
            positive_index = next(
                index
                for index, item in enumerate(schedules)
                if item.pair_name == pair and item.segments[0][2] > 0
            )
            negative_index = next(
                index
                for index, item in enumerate(schedules)
                if item.pair_name == pair and item.segments[0][2] < 0
            )
            item = schedules[positive_index]
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
            derivative = 0.5 * (positive - negative) / item.gamma
            response = sample_rms(derivative)
            even = 0.5 * (positive + negative) - baseline
            even_over_odd = sample_rms(even) / np.maximum(item.gamma * response, 1e-30)
            unit_norm = np.sqrt(
                np.maximum(
                    0.5 * (stats[:, positive_index, 0] + stats[:, negative_index, 0]),
                    0.0,
                )
            )
            target = signed_values.setdefault(
                pair, {"response": [], "even_over_odd": [], "propagation_gain": []}
            )
            target["response"].append(response)
            target["even_over_odd"].append(even_over_odd)
            target["propagation_gain"].append(response / np.maximum(unit_norm, 1e-30))
            if item.family == "pulse":
                derivatives.setdefault((int(item.pulse_step), float(item.gamma)), []).append(
                    derivative
                )

        selected = sorted(
            {item.left_window for item in schedules if item.left_window is not None}
            | {item.right_window for item in schedules if item.right_window is not None}
        )
        for left_position, left in enumerate(selected):
            for right in selected[left_position + 1 :]:
                def condition(name: str) -> np.ndarray:
                    return _load_condition(
                        run_dir,
                        condition_index=by_name[name],
                        samples=samples,
                        world_size=world_size,
                    )

                left_pair = next(
                    item.pair_name
                    for item in schedules
                    if item.family == "window" and item.window_index == left
                )
                right_pair = next(
                    item.pair_name
                    for item in schedules
                    if item.family == "window" and item.window_index == right
                )
                values = interaction_metrics(
                    baseline,
                    condition(left_pair + "_pos"),
                    condition(left_pair + "_neg"),
                    condition(right_pair + "_pos"),
                    condition(right_pair + "_neg"),
                    condition(f"interaction_w{left}_w{right}_pp"),
                    condition(f"interaction_w{left}_w{right}_pm"),
                    condition(f"interaction_w{left}_w{right}_mp"),
                    condition(f"interaction_w{left}_w{right}_mm"),
                    gamma=next(item.gamma for item in schedules if item.family == "interaction"),
                )
                target = interaction_values.setdefault(
                    (left, right), {name: [] for name in values}
                )
                for name, per_sample in values.items():
                    target[name].append(per_sample)

    signed_rows: list[dict[str, Any]] = []
    for pair_index, (pair, values) in enumerate(sorted(signed_values.items())):
        item = next(schedule for schedule in schedules if schedule.pair_name == pair)
        merged = {name: np.concatenate(parts) for name, parts in values.items()}
        signed_rows.append(
            {
                "pair_name": pair,
                "family": item.family,
                "gamma": item.gamma,
                "pulse_step": item.pulse_step,
                "window_index": item.window_index,
                "time": float(grid[item.pulse_step])
                if item.pulse_step is not None
                else float(grid[windows[item.window_index][0]]),
                "runs": len(run_dirs),
                "samples": len(merged["response"]),
                **pooled_row(
                    merged,
                    repeats=args.bootstrap_repeats,
                    seed=args.seed + 10_007 * pair_index,
                ),
            }
        )
    signed_frame = pd.DataFrame(signed_rows).sort_values(
        ["family", "pulse_step", "window_index", "gamma"]
    )

    gamma_values = sorted({gamma for _, gamma in derivatives})
    linearity_rows = []
    if len(gamma_values) == 2:
        for step_index, step in enumerate(sorted({step for step, _ in derivatives})):
            small = np.concatenate(derivatives[(step, gamma_values[0])])
            large = np.concatenate(derivatives[(step, gamma_values[1])])
            small_rms, large_rms = sample_rms(small), sample_rms(large)
            metrics = {
                "relative_error": sample_rms(small - large)
                / np.maximum(0.5 * (small_rms + large_rms), 1e-30),
                "one_minus_cosine": 1.0 - sample_cosine(small, large),
                "amplitude_ratio": small_rms / np.maximum(large_rms, 1e-30),
            }
            linearity_rows.append(
                {
                    "step": step,
                    "time": float(grid[step]),
                    "runs": len(run_dirs),
                    "samples": len(small),
                    **pooled_row(
                        metrics,
                        repeats=args.bootstrap_repeats,
                        seed=args.seed + 20_011 * step_index,
                    ),
                }
            )
    linearity_frame = pd.DataFrame(linearity_rows)

    interaction_rows = []
    for pair_index, ((left, right), values) in enumerate(sorted(interaction_values.items())):
        merged = {name: np.concatenate(parts) for name, parts in values.items()}
        interaction_rows.append(
            {
                "left_window": left,
                "right_window": right,
                "left_steps": f"{windows[left][0]}:{windows[left][1]}",
                "right_steps": f"{windows[right][0]}:{windows[right][1]}",
                "runs": len(run_dirs),
                "samples": len(next(iter(merged.values()))),
                **pooled_row(
                    merged,
                    repeats=args.bootstrap_repeats,
                    seed=args.seed + 30_013 * pair_index,
                ),
            }
        )
    interaction_frame = pd.DataFrame(interaction_rows)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    signed_frame.to_csv(output_dir / "signed_response_pooled.csv", index=False)
    linearity_frame.to_csv(output_dir / "cross_gamma_linearity_pooled.csv", index=False)
    interaction_frame.to_csv(output_dir / "window_interaction_pooled.csv", index=False)
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    for gamma, part in signed_frame[signed_frame.family.eq("pulse")].groupby("gamma"):
        axes[0].plot(part.time, part.propagation_gain_mean, "o-", label=f"gamma={gamma:g}")
        axes[1].plot(part.time, part.response_q95_over_median, "o-", label=f"gamma={gamma:g}")
    if not linearity_frame.empty:
        axes[2].plot(
            linearity_frame.time,
            linearity_frame.relative_error_mean,
            "o-",
            label="relative derivative error",
        )
        axes[2].plot(linearity_frame.time, linearity_frame.one_minus_cosine_mean, "s-", label="1 - cosine")
    for axis, title in zip(axes, ("Pooled propagation gain", "Pooled response tail", "Pooled local nonlinearity")):
        axis.invert_xaxis(); axis.set(title=title, xlabel="solver time t"); axis.grid(alpha=0.2); axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "sit_ig_endpoint_validation_pooled.png", dpi=180)
    plt.close(figure)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "run_dirs": [str(path) for path in run_dirs],
                "run_seeds": [int(manifest["seed"]) for manifest in manifests],
                "total_samples": total_samples,
                "bootstrap_repeats": args.bootstrap_repeats,
                "seed": args.seed,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(signed_frame.to_string(index=False))
    print(linearity_frame.to_string(index=False))
    print(interaction_frame.to_string(index=False))


if __name__ == "__main__":
    main()
