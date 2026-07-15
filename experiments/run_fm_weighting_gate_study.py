"""Run the normalized FM weighting causal gate and save reusable tables."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.fm_weighting_gate import (
    GateTreatment,
    paired_treatment_effects,
    run_gate_grid_parallel,
    treatment_weight_diagnostics,
)
from experiments.nonlinear_fm_whitening_toy import (
    MixtureFMConfig,
    NeuralTrainConfig,
    distribution_metrics,
    reverse_ode_samples,
    sample_latent_reference,
)


DEFAULT_OUTPUT = Path.home() / "data/eqvae/experiments/fm_weighting_gate"


def decoder_permutation_control(
    history: pd.DataFrame,
    problem: MixtureFMConfig,
    *,
    permutation_count: int = 256,
    seed: int = 0,
) -> pd.DataFrame:
    final = history.sort_values("step").groupby(
        ["architecture", "batch_size", "seed", "treatment"], as_index=False
    ).tail(1)
    direction_columns = sorted(
        (column for column in final.columns if column.startswith("direction_mse_")),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    variance = np.asarray(problem.variance, dtype=np.float64)
    time = np.linspace(problem.t_min, problem.t_max, 4097)[:, None]
    residual = variance[None] / ((1.0 - time) ** 2 * variance[None] + time**2)
    mean_residual = residual.mean(axis=0)
    decoder_gain = np.asarray(problem.decoder_gain, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    permutations = [np.arange(len(decoder_gain))]
    permutations.extend(rng.permutation(len(decoder_gain)) for _ in range(int(permutation_count)))
    rows = []
    for (architecture, batch_size, run_seed), group in final.groupby(
        ["architecture", "batch_size", "seed"]
    ):
        errors = {
            row.treatment: row[direction_columns].to_numpy(dtype=np.float64)
            for _, row in group.iterrows()
        }
        baseline = errors.get("baseline:gamma=0")
        if baseline is None:
            continue
        for permutation_index, permutation in enumerate(permutations):
            gain = decoder_gain[permutation]
            baseline_metric = float(np.sum(baseline * gain**2) / np.sum(gain**2))
            alignment = float(np.corrcoef(np.log(mean_residual), np.log(gain**2))[0, 1])
            for treatment, error in errors.items():
                if treatment == "baseline:gamma=0":
                    continue
                metric = float(np.sum(error * gain**2) / np.sum(gain**2))
                rows.append(
                    {
                        "architecture": architecture,
                        "batch_size": int(batch_size),
                        "seed": int(run_seed),
                        "treatment": treatment,
                        "permutation": permutation_index,
                        "is_original_alignment": permutation_index == 0,
                        "residual_decoder_alignment": alignment,
                        "baseline_decoder_mse": baseline_metric,
                        "treatment_decoder_mse": metric,
                        "relative_gain": (baseline_metric - metric) / baseline_metric,
                    }
                )
    return pd.DataFrame(rows)


def run_study(
    output_dir: Path,
    devices: tuple[str, ...],
    seeds: tuple[int, ...],
    *,
    quick: bool = False,
) -> None:
    problem = MixtureFMConfig()
    treatments = (
        GateTreatment("baseline", 0.0),
        GateTreatment("time", 0.5),
        GateTreatment("direction", 0.5),
        GateTreatment("full", 0.5),
        GateTreatment("direction", 1.0),
        GateTreatment("full", 1.0),
    )
    scale = 0.25 if quick else 1.0
    setups = (
        (
            "mlp_b64",
            NeuralTrainConfig(
                architecture="mlp",
                batch_size=64,
                steps=max(80, int(800 * scale)),
                learning_rate=2e-3,
                hidden_size=96,
                depth=3,
                eval_every=max(20, int(40 * scale)),
                eval_count=4096 if not quick else 1024,
            ),
        ),
        (
            "mini_dit_b64",
            NeuralTrainConfig(
                architecture="mini_dit",
                batch_size=64,
                steps=max(80, int(500 * scale)),
                learning_rate=2e-3,
                hidden_size=64,
                depth=2,
                num_heads=4,
                eval_every=max(20, int(25 * scale)),
                eval_count=4096 if not quick else 1024,
            ),
        ),
        (
            "mini_dit_b256",
            NeuralTrainConfig(
                architecture="mini_dit",
                batch_size=256,
                steps=max(80, int(500 * scale)),
                learning_rate=2e-3,
                hidden_size=64,
                depth=2,
                num_heads=4,
                eval_every=max(20, int(25 * scale)),
                eval_count=4096 if not quick else 1024,
            ),
        ),
    )
    runs, history, summary = run_gate_grid_parallel(
        problem,
        setups,
        treatments,
        seeds,
        devices,
        verbose=True,
    )
    paired = paired_treatment_effects(summary)
    weight_diagnostics = treatment_weight_diagnostics(problem, treatments)
    permutations = decoder_permutation_control(history, problem, permutation_count=256, seed=0)

    generation_rows = []
    sample_count = 1024 if quick else 4096
    ode_steps = 40 if quick else 80
    for seed in seeds:
        reference = sample_latent_reference(problem, sample_count, seed=800_000 + seed)
        for treatment in treatments:
            key = ("mini_dit_b64", treatment.mode, float(treatment.gamma), int(seed))
            run = runs[key]
            generated = reverse_ode_samples(
                problem,
                model=run.model,
                sample_count=sample_count,
                ode_steps=ode_steps,
                seed=800_000 + seed,
                device=devices[seed % len(devices)],
            )
            generation_rows.append(
                {
                    "seed": seed,
                    "treatment": treatment.name,
                    "mode": treatment.mode,
                    "gamma": treatment.gamma,
                    **distribution_metrics(generated, reference),
                }
            )
    generation = pd.DataFrame(generation_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(output_dir / "history.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    paired.to_csv(output_dir / "paired_effects.csv", index=False)
    weight_diagnostics.to_csv(output_dir / "weight_diagnostics.csv", index=False)
    permutations.to_csv(output_dir / "decoder_permutations.csv", index=False)
    generation.to_csv(output_dir / "generation.csv", index=False)
    metadata = {
        "problem": asdict(problem),
        "treatments": [asdict(treatment) for treatment in treatments],
        "seeds": list(seeds),
        "devices": list(devices),
        "quick": quick,
        "sample_count": sample_count,
        "ode_steps": ode_steps,
        "setup_count": len(setups),
        "run_count": len(summary),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(f"saved study tables to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--devices", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.devices:
        devices = tuple(args.devices)
    else:
        devices = tuple(f"cuda:{index}" for index in range(torch.cuda.device_count())) or ("cpu",)
    run_study(args.output_dir, devices, tuple(args.seeds), quick=args.quick)


if __name__ == "__main__":
    main()
