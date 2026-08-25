#!/usr/bin/env python3
"""Separate AdvFD representation discovery from generator transport.

The critic is trained with one AdvFD component, while its frozen representation
is evaluated with the mean, covariance, and full Frechet generator fields.  This
tests whether the covariance objective is useful for discovering a feature
space even when its own sample-space pullback is a poor transport direction.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.advfd_feature_pullback import (
    build_feature_force_context,
    learned_pullback_field,
)
from experiments.frechet_residual_score_toy import field_diagnostics
from experiments.run_advfd_inner_trajectory_audit import TRAJECTORY_MODES
from experiments.run_advfd_monoflow_audit import build_regime, monoflow_diagnostics
from experiments.run_advfd_quotient_gradient_audit import critic_feature_statistics
from experiments.run_frechet_residual_score_toy import (
    CriticConfig,
    FeatureCritic,
    parse_ints,
    parse_strings,
    population_advfd,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluation_objectives(training_objective: str) -> dict[str, str]:
    if training_objective.startswith("pooled_"):
        return {
            "mean": "pooled_mean_only",
            "covariance": "pooled_covariance_only",
            "full": "pooled_full",
        }
    return {
        "mean": "official_mean_only",
        "covariance": "official_covariance_only",
        "full": "official_regularized",
    }


def evaluate_components(
    critic,
    target,
    source,
    *,
    training_mode: str,
    step: int,
    quadrature_order: int,
    sample_count: int,
    sample_seed: int,
    whitening_epsilon: float,
) -> list[dict[str, float | int | str]]:
    rows = []
    training_objective = TRAJECTORY_MODES[training_mode].objective_mode
    feature_stats = critic_feature_statistics(
        critic, target, source, order=quadrature_order
    )
    for generator_component, objective_mode in evaluation_objectives(
        training_objective
    ).items():
        context = build_feature_force_context(
            critic,
            target,
            source,
            order=quadrature_order,
            whitening_epsilon=whitening_epsilon,
            objective_mode=objective_mode,
        )
        field = learned_pullback_field(critic, context, mode="transpose")
        diagnostics = field_diagnostics(
            target, source, field, quadrature_order=quadrature_order
        )
        diagnostics.update(
            monoflow_diagnostics(
                critic,
                context,
                target,
                source,
                sample_count=sample_count,
                sample_seed=sample_seed,
            )
        )
        velocity_rms = diagnostics["velocity_rms"]
        rows.append(
            {
                "training_mode": training_mode,
                "training_objective": training_objective,
                "generator_component": generator_component,
                "generator_objective": objective_mode,
                "step": step,
                "generator_component_value": context["distance"],
                "kl_descent_per_velocity_rms": (
                    -diagnostics["reverse_kl_derivative"]
                    / max(velocity_rms, 1e-30)
                ),
                **diagnostics,
                **feature_stats,
            }
        )
    return rows


def train_and_audit(
    target,
    source,
    *,
    training_mode: str,
    seed: int,
    checkpoints: tuple[int, ...],
    quadrature_order: int,
    sample_count: int,
    whitening_epsilon: float,
    device: torch.device,
) -> list[dict[str, float | int | str]]:
    mode = TRAJECTORY_MODES[training_mode]
    config = CriticConfig(
        steps=max(checkpoints),
        whitening_epsilon=whitening_epsilon,
        quadrature_order=min(quadrature_order, 16),
        objective_mode=mode.objective_mode,
        detach_real=mode.detach_real,
        detach_calibration=mode.detach_calibration,
    )
    seed_everything(seed)
    critic = FeatureCritic(config).to(device=device, dtype=target.means.dtype)
    optimizer = torch.optim.AdamW(
        critic.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    checkpoint_set = set(checkpoints)
    rows = []
    for step in range(max(checkpoints) + 1):
        if step in checkpoint_set:
            with torch.no_grad():
                training_value, _ = population_advfd(
                    critic,
                    target,
                    source,
                    order=config.quadrature_order,
                    whitening_epsilon=config.whitening_epsilon,
                    detach_real=config.detach_real,
                    objective_mode=config.objective_mode,
                    detach_calibration=config.detach_calibration,
                )
            evaluated = evaluate_components(
                critic,
                target,
                source,
                training_mode=training_mode,
                step=step,
                quadrature_order=quadrature_order,
                sample_count=sample_count,
                sample_seed=seed + 100_003 * step,
                whitening_epsilon=whitening_epsilon,
            )
            for row in evaluated:
                rows.append({"training_value": float(training_value), **row})
        if step == max(checkpoints):
            break
        optimizer.zero_grad(set_to_none=True)
        objective, _ = population_advfd(
            critic,
            target,
            source,
            order=config.quadrature_order,
            whitening_epsilon=config.whitening_epsilon,
            detach_real=config.detach_real,
            objective_mode=config.objective_mode,
            detach_calibration=config.detach_calibration,
        )
        (-objective).backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), config.gradient_clip)
        optimizer.step()
    return rows


def run(
    output_root: Path,
    *,
    regimes: tuple[str, ...],
    noise_sigma: float,
    training_modes: tuple[str, ...],
    seeds: tuple[int, ...],
    checkpoints: tuple[int, ...],
    quadrature_order: int,
    sample_count: int,
    whitening_epsilon: float,
    device: torch.device,
) -> None:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    if not checkpoints or checkpoints[0] != 0:
        raise ValueError("checkpoints must begin at zero")
    unknown = set(training_modes) - set(TRAJECTORY_MODES)
    if unknown:
        raise ValueError(f"unknown training modes: {sorted(unknown)}")
    output_root.mkdir(parents=True)

    rows = []
    for regime_name in regimes:
        target_clean, source_clean = build_regime(regime_name, device=device)
        target = target_clean.convolve_isotropic(noise_sigma)
        source = source_clean.convolve_isotropic(noise_sigma)
        for training_mode in training_modes:
            for seed in seeds:
                print(
                    f"regime={regime_name} mode={training_mode} seed={seed}",
                    flush=True,
                )
                audited = train_and_audit(
                    target,
                    source,
                    training_mode=training_mode,
                    seed=seed,
                    checkpoints=checkpoints,
                    quadrature_order=quadrature_order,
                    sample_count=sample_count,
                    whitening_epsilon=whitening_epsilon,
                    device=device,
                )
                for row in audited:
                    rows.append(
                        {
                            "regime": regime_name,
                            "noise_sigma": noise_sigma,
                            "seed": seed,
                            **row,
                        }
                    )

    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "cross_component_trajectory.csv", index=False)
    aggregate = frame.groupby(
        ["regime", "training_mode", "generator_component", "step"],
        as_index=False,
    ).agg(
        training_value=("training_value", "mean"),
        component_value=("generator_component_value", "mean"),
        correctability=("kl_descent_per_velocity_rms", "mean"),
        score_cosine=("score_cosine", "mean"),
        velocity_rms=("velocity_rms", "mean"),
        velocity_effective_fraction=("velocity_effective_fraction", "mean"),
    )
    aggregate.to_csv(output_root / "aggregate.csv", index=False)

    groups = list(aggregate.groupby(["regime", "training_mode"]))
    figure, axes = plt.subplots(
        len(groups), 1, figsize=(8.5, max(4.0, 3.5 * len(groups))), squeeze=False
    )
    for axis, ((regime_name, training_mode), group) in zip(axes[:, 0], groups):
        for component, component_frame in group.groupby("generator_component"):
            axis.plot(
                component_frame["step"],
                component_frame["correctability"],
                marker="o",
                label=component,
            )
        axis.set_title(f"{regime_name}: critic={training_mode}")
        axis.set_xlabel("critic steps")
        axis.set_ylabel("KL descent / field RMS")
        axis.grid(alpha=0.2)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_root / "cross_component_trajectory.png", dpi=180)
    plt.close(figure)

    final = aggregate[aggregate["step"] == max(checkpoints)]
    pivot = final.pivot_table(
        index=["regime", "training_mode"],
        columns="generator_component",
        values="correctability",
    ).reset_index()
    pivot.to_csv(output_root / "final_correctability.csv", index=False)
    summary = {
        "protocol": "advfd_cross_component_trajectory_v1",
        "regimes": list(regimes),
        "noise_sigma": noise_sigma,
        "training_modes": list(training_modes),
        "seeds": list(seeds),
        "checkpoints": list(checkpoints),
        "all_values_finite": bool(
            frame[["training_value", "kl_descent_per_velocity_rms"]]
            .map(lambda value: bool(torch.isfinite(torch.tensor(value))))
            .all()
            .all()
        ),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--regimes", type=parse_strings, default=("rotated_ring", "shape_only")
    )
    parser.add_argument("--noise-sigma", type=float, default=0.4)
    parser.add_argument(
        "--training-modes",
        type=parse_strings,
        default=(
            "real_quotient",
            "real_covariance_quotient",
            "pooled_quotient",
            "pooled_covariance_quotient",
        ),
    )
    parser.add_argument(
        "--seeds", type=parse_ints, default=(20260824, 20260825, 20260826)
    )
    parser.add_argument(
        "--checkpoints",
        type=parse_ints,
        default=(0, 25, 50, 100, 150, 200, 300, 500, 750, 1000),
    )
    parser.add_argument("--quadrature-order", type=int, default=16)
    parser.add_argument("--sample-count", type=int, default=4096)
    parser.add_argument("--whitening-epsilon", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.output_root,
        regimes=args.regimes,
        noise_sigma=args.noise_sigma,
        training_modes=args.training_modes,
        seeds=args.seeds,
        checkpoints=args.checkpoints,
        quadrature_order=args.quadrature_order,
        sample_count=args.sample_count,
        whitening_epsilon=args.whitening_epsilon,
        device=torch.device(args.device),
    )
