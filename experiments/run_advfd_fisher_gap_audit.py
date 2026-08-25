#!/usr/bin/env python3
"""Audit where AdvFD's mean witness departs from its Fisher/Pearson limit."""

from __future__ import annotations

import argparse
import json
import math
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

from experiments.frechet_residual_score_toy import (
    build_toy_regimes,
    density_ratio,
    field_diagnostics,
    finite_pushforward_kl,
    finite_pushforward_pearson,
    pearson_correction,
    pearson_divergence,
    pearson_field,
    score_field,
    weighted_inner,
)
from experiments.run_advfd_smoothed_retraining_transport import (
    build_rotated_ring_pair,
)
from experiments.run_frechet_residual_score_toy import (
    CriticConfig,
    FeatureCritic,
    advfd_functional_field,
)


TRAINING_MODES = ("official_stopgrad", "fisher_rayleigh", "supervised_ratio")


def parse_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in parse_strings(value))


def parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in parse_strings(value))


def build_regime(name: str, *, device: torch.device):
    if name == "shape_only":
        return build_toy_regimes(dtype=torch.float64, device=device)[name]
    if name == "rotated_ring":
        return build_rotated_ring_pair(rotation=0.22, device=device)
    raise ValueError(f"unknown regime: {name}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def scalar_moments(values: torch.Tensor, weights: torch.Tensor):
    normalized = weights / weights.sum()
    mean = (normalized * values).sum()
    variance = (normalized * (values - mean).square()).sum()
    return mean, variance


def fisher_value(
    critic: FeatureCritic,
    target,
    source,
    *,
    order: int,
    epsilon: float,
    stopgrad_real: bool,
) -> torch.Tensor:
    target_points, target_weights = target.quadrature(order)
    source_points, source_weights = source.quadrature(order)
    target_values = critic(target_points).squeeze(1)
    source_values = critic(source_points).squeeze(1)
    target_mean, target_variance = scalar_moments(target_values, target_weights)
    source_mean, _ = scalar_moments(source_values, source_weights)
    if stopgrad_real:
        target_mean = target_mean.detach()
        target_variance = target_variance.detach()
    return (source_mean - target_mean).square() / (target_variance + epsilon)


def supervised_ratio_loss(
    critic: FeatureCritic,
    target,
    source,
    *,
    order: int,
    exact_pearson: float,
) -> torch.Tensor:
    normalizer = math.sqrt(max(exact_pearson, 1e-16))
    losses = []
    for distribution in (target, source):
        states, weights = distribution.quadrature(order)
        desired = (density_ratio(target, source, states) - 1.0) / normalizer
        predicted = critic(states).squeeze(1)
        normalized = weights / weights.sum()
        losses.append((normalized * (predicted - desired).square()).sum())
    return 0.5 * (losses[0] + losses[1])


def train_scalar_critic(
    target,
    source,
    *,
    mode: str,
    seed: int,
    steps: int,
    learning_rate: float,
    epsilon: float,
    quadrature_order: int,
    device: torch.device,
):
    if mode not in TRAINING_MODES:
        raise ValueError(f"unknown training mode: {mode}")
    seed_everything(seed)
    config = CriticConfig(
        hidden_dim=64,
        depth=3,
        feature_dim=1,
        steps=steps,
        learning_rate=learning_rate,
        whitening_epsilon=epsilon,
        quadrature_order=quadrature_order,
        objective_mode="official_mean_only",
    )
    critic = FeatureCritic(config).to(device=device, dtype=torch.float64)
    optimizer = torch.optim.AdamW(
        critic.parameters(), lr=learning_rate, weight_decay=config.weight_decay
    )
    exact_pearson = pearson_divergence(
        target, source, quadrature_order=max(quadrature_order, 20)
    )
    rows = []
    evaluate_every = max(steps // 20, 1)
    for step in range(steps + 1):
        if step % evaluate_every == 0 or step == steps:
            with torch.no_grad():
                value = fisher_value(
                    critic,
                    target,
                    source,
                    order=quadrature_order,
                    epsilon=epsilon,
                    stopgrad_real=False,
                )
                supervised = supervised_ratio_loss(
                    critic,
                    target,
                    source,
                    order=quadrature_order,
                    exact_pearson=exact_pearson,
                )
                target_points, target_weights = target.quadrature(quadrature_order)
                target_values = critic(target_points).squeeze(1)
                target_mean, target_variance = scalar_moments(
                    target_values, target_weights
                )
            rows.append(
                {
                    "step": step,
                    "fisher_value": float(value),
                    "objective_fraction": float(value) / max(exact_pearson, 1e-16),
                    "ratio_supervision_mse": float(supervised),
                    "target_feature_mean": float(target_mean),
                    "target_feature_std": float(target_variance.sqrt()),
                }
            )
        if step == steps:
            break
        optimizer.zero_grad(set_to_none=True)
        if mode == "supervised_ratio":
            loss = supervised_ratio_loss(
                critic,
                target,
                source,
                order=quadrature_order,
                exact_pearson=exact_pearson,
            )
        else:
            objective = fisher_value(
                critic,
                target,
                source,
                order=quadrature_order,
                epsilon=epsilon,
                stopgrad_real=mode == "official_stopgrad",
            )
            loss = -objective
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), config.gradient_clip)
        optimizer.step()
    return critic.eval().requires_grad_(False), rows


def reference_diagnostics(target, source, field, *, order: int):
    states, weights = source.quadrature(order)
    normalized = weights / weights.sum()
    velocity = field(states, False)
    pearson = pearson_correction(target, source, states)
    velocity_norm = weighted_inner(velocity, velocity, normalized).sqrt()
    pearson_norm = weighted_inner(pearson, pearson, normalized).sqrt()
    alignment = weighted_inner(velocity, pearson, normalized)
    return {
        "pearson_field_rms": float(pearson_norm / math.sqrt(source.dimension)),
        "pearson_cosine": float(
            alignment
            / (velocity_norm * pearson_norm).clamp_min(
                torch.finfo(states.dtype).eps
            )
        ),
        "pearson_derivative": float(-2.0 * alignment),
    }


def witness_diagnostics(
    critic: FeatureCritic,
    target,
    source,
    *,
    order: int,
    epsilon: float,
    exact_pearson: float,
):
    states, weights = source.quadrature(order)
    normalized = weights / weights.sum()
    target_points, target_weights = target.quadrature(order)
    with torch.no_grad():
        target_values = critic(target_points).squeeze(1)
        source_values = critic(states).squeeze(1)
        target_mean, target_variance = scalar_moments(
            target_values, target_weights
        )
        calibrated = (source_values - target_mean) / (
            target_variance + epsilon
        ).sqrt()
        source_mean = (normalized * calibrated).sum()
        orientation = torch.where(
            source_mean >= 0,
            torch.ones_like(source_mean),
            -torch.ones_like(source_mean),
        )
        oriented = calibrated * orientation
        optimal = (density_ratio(target, source, states) - 1.0) / math.sqrt(
            max(exact_pearson, 1e-16)
        )
        numerator = (normalized * oriented * optimal).sum()
        denominator = (
            (normalized * oriented.square()).sum().sqrt()
            * (normalized * optimal.square()).sum().sqrt()
        )
    return {
        "witness_cosine_q": float(
            numerator / denominator.clamp_min(torch.finfo(states.dtype).eps)
        ),
        "calibrated_source_mean": float(source_mean),
    }


def evaluate_field(
    target,
    source,
    field,
    *,
    displacements: tuple[float, ...],
    order: int,
):
    diagnostics = field_diagnostics(
        target, source, field, quadrature_order=order
    )
    diagnostics.update(reference_diagnostics(target, source, field, order=order))
    rows = []
    velocity_rms = diagnostics["velocity_rms"]
    for displacement in displacements:
        if velocity_rms <= 1e-14:
            step_size = float("nan")
            finite_kl = {
                "kl_before": float("nan"),
                "kl_after": float("nan"),
                "kl_change": float("nan"),
            }
            finite_pearson = {
                "pearson_before": float("nan"),
                "pearson_after": float("nan"),
                "pearson_change": float("nan"),
            }
        else:
            step_size = displacement / velocity_rms
            finite_kl = finite_pushforward_kl(
                target,
                source,
                field,
                step_size=step_size,
                quadrature_order=order,
            )
            finite_pearson = finite_pushforward_pearson(
                target,
                source,
                field,
                step_size=step_size,
                quadrature_order=order,
            )
        rows.append(
            {
                "target_displacement_rms": displacement,
                "step_size": step_size,
                **diagnostics,
                **{
                    key: value
                    for key, value in finite_kl.items()
                    if key not in {
                        "positive_jacobian_fraction",
                        "minimum_jacobian_determinant",
                    }
                },
                **finite_pearson,
            }
        )
    return rows


def run(
    output_root: Path,
    *,
    regimes: tuple[str, ...],
    noise_sigmas: tuple[float, ...],
    seeds: tuple[int, ...],
    training_modes: tuple[str, ...],
    steps: int,
    learning_rate: float,
    epsilon: float,
    train_order: int,
    eval_order: int,
    displacements: tuple[float, ...],
    device: torch.device,
) -> None:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    output_root.mkdir(parents=True)
    rows = []
    curves = []
    for regime in regimes:
        target_clean, source_clean = build_regime(regime, device=device)
        for sigma in noise_sigmas:
            target = target_clean.convolve_isotropic(sigma)
            source = source_clean.convolve_isotropic(sigma)
            exact_pearson = pearson_divergence(
                target, source, quadrature_order=eval_order
            )
            for analytic_name, field in (
                ("analytic_pearson", pearson_field(target, source)),
                ("analytic_reverse_kl", score_field(target, source)),
            ):
                for result in evaluate_field(
                    target,
                    source,
                    field,
                    displacements=displacements,
                    order=eval_order,
                ):
                    rows.append(
                        {
                            "regime": regime,
                            "noise_sigma": sigma,
                            "seed": -1,
                            "training_mode": analytic_name,
                            "exact_pearson": exact_pearson,
                            "learned_fisher_value": float("nan"),
                            "objective_fraction": float("nan"),
                            "witness_cosine_q": float("nan"),
                            "calibrated_source_mean": float("nan"),
                            **result,
                        }
                    )
            for seed in seeds:
                for mode in training_modes:
                    print(
                        f"regime={regime} sigma={sigma:g} seed={seed} "
                        f"mode={mode}",
                        flush=True,
                    )
                    critic, curve = train_scalar_critic(
                        target,
                        source,
                        mode=mode,
                        seed=seed,
                        steps=steps,
                        learning_rate=learning_rate,
                        epsilon=epsilon,
                        quadrature_order=train_order,
                        device=device,
                    )
                    for item in curve:
                        curves.append(
                            {
                                "regime": regime,
                                "noise_sigma": sigma,
                                "seed": seed,
                                "training_mode": mode,
                                **item,
                            }
                        )
                    field, learned_value = advfd_functional_field(
                        critic,
                        target,
                        source,
                        order=eval_order,
                        whitening_epsilon=epsilon,
                        objective_mode="official_mean_only",
                    )
                    witness = witness_diagnostics(
                        critic,
                        target,
                        source,
                        order=eval_order,
                        epsilon=epsilon,
                        exact_pearson=exact_pearson,
                    )
                    for result in evaluate_field(
                        target,
                        source,
                        field,
                        displacements=displacements,
                        order=eval_order,
                    ):
                        rows.append(
                            {
                                "regime": regime,
                                "noise_sigma": sigma,
                                "seed": seed,
                                "training_mode": mode,
                                "exact_pearson": exact_pearson,
                                "learned_fisher_value": learned_value,
                                "objective_fraction": learned_value
                                / max(exact_pearson, 1e-16),
                                **witness,
                                **result,
                            }
                        )
    frame = pd.DataFrame(rows)
    curve_frame = pd.DataFrame(curves)
    frame.to_csv(output_root / "fisher_gap_audit.csv", index=False)
    curve_frame.to_csv(output_root / "critic_curves.csv", index=False)
    selected = frame[frame["target_displacement_rms"] == min(displacements)]
    learned = selected[selected["seed"] >= 0]
    aggregate = (
        learned.groupby(["regime", "noise_sigma", "training_mode"])
        .agg(
            objective_fraction=("objective_fraction", "mean"),
            witness_cosine_q=("witness_cosine_q", "mean"),
            pearson_cosine=("pearson_cosine", "mean"),
            score_cosine=("score_cosine", "mean"),
            velocity_rms=("velocity_rms", "mean"),
            pearson_change=("pearson_change", "mean"),
            kl_change=("kl_change", "mean"),
        )
        .reset_index()
    )
    aggregate.to_csv(output_root / "aggregate.csv", index=False)
    summary = {
        "protocol": "advfd_fisher_gap_audit_v1",
        "regimes": list(regimes),
        "noise_sigmas": list(noise_sigmas),
        "seeds": list(seeds),
        "training_modes": list(training_modes),
        "steps": steps,
        "learning_rate": learning_rate,
        "whitening_epsilon": epsilon,
        "train_quadrature_order": train_order,
        "eval_quadrature_order": eval_order,
        "target_displacement_rms": list(displacements),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    labels = (
        aggregate["regime"]
        + "\nnoise="
        + aggregate["noise_sigma"].map(lambda value: f"{value:g}")
        + "\n"
        + aggregate["training_mode"]
    )
    for axis, column, title in (
        (axes[0], "objective_fraction", "Learned / exact Pearson"),
        (axes[1], "pearson_cosine", "Field cosine to Pearson descent"),
        (axes[2], "witness_cosine_q", "Witness cosine to density ratio"),
    ):
        axis.bar(range(len(aggregate)), aggregate[column])
        axis.set_title(title)
        axis.set_xticks(range(len(aggregate)), labels, rotation=90)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_root / "fisher_gap_audit.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--regimes", type=parse_strings, default=("shape_only", "rotated_ring"))
    parser.add_argument("--noise-sigmas", type=parse_floats, default=(0.0, 0.4))
    parser.add_argument("--seeds", type=parse_ints, default=(8401, 8402, 8403))
    parser.add_argument("--training-modes", type=parse_strings, default=TRAINING_MODES)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--whitening-epsilon", type=float, default=1e-3)
    parser.add_argument("--train-order", type=int, default=16)
    parser.add_argument("--eval-order", type=int, default=20)
    parser.add_argument("--displacements", type=parse_floats, default=(1e-4, 1e-3, 1e-2))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(
        args.output_root,
        regimes=args.regimes,
        noise_sigmas=args.noise_sigmas,
        seeds=args.seeds,
        training_modes=args.training_modes,
        steps=args.steps,
        learning_rate=args.learning_rate,
        epsilon=args.whitening_epsilon,
        train_order=args.train_order,
        eval_order=args.eval_order,
        displacements=args.displacements,
        device=torch.device(args.device),
    )


if __name__ == "__main__":
    main()
