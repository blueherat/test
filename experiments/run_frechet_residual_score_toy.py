#!/usr/bin/env python3
"""Compare AdvFD and Fréchet-complementary score fields on smooth toys."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn

from experiments.advfd_cleanroom.core import (
    AffineCalibration,
    calibrate_moments,
    frechet_from_moments,
    fit_calibration_from_moments,
    moments_from_mean_and_second,
)
from experiments.frechet_residual_score_toy import (
    GaussianMixture,
    build_toy_regimes,
    field_diagnostics,
    finite_pushforward_kl,
    frechet_value,
    project_onto_fixed_moment_tangent,
    score_correction,
    score_field,
    static_field,
    sum_fields,
    tangent_field_from_projection,
    weighted_moments,
)


@dataclass(frozen=True)
class CriticConfig:
    hidden_dim: int = 64
    depth: int = 3
    feature_dim: int = 8
    steps: int = 1000
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    whitening_epsilon: float = 1e-3
    quadrature_order: int = 14
    objective_mode: Literal[
        "paper_affine",
        "official_regularized",
        "official_mean_only",
        "official_covariance_only",
        "pooled_full",
        "pooled_mean_only",
        "pooled_covariance_only",
    ] = "paper_affine"
    detach_real: bool = True
    detach_calibration: bool = True


class FeatureCritic(nn.Module):
    def __init__(self, config: CriticConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(2, config.hidden_dim), nn.SiLU()]
        for _ in range(config.depth - 1):
            layers.extend((nn.Linear(config.hidden_dim, config.hidden_dim), nn.SiLU()))
        layers.append(nn.Linear(config.hidden_dim, config.feature_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def calibrated_weighted_moments(
    features: torch.Tensor,
    weights: torch.Tensor,
    calibration: AffineCalibration,
):
    return weighted_moments(calibration.apply(features), weights)


def advfd_from_raw_moments(
    target_raw,
    source_raw,
    *,
    whitening_epsilon: float,
    objective_mode: str,
    detach_calibration: bool = True,
) -> tuple[torch.Tensor, AffineCalibration]:
    """Evaluate either the paper or official finite-epsilon AdvFD objective.

    The paper applies one common affine map to both feature distributions.  The
    official code instead treats the whitened real Gaussian as ``N(0, I)`` and
    adds the same ``epsilon I`` loading to the fake covariance before whitening.
    These coincide only as epsilon tends to zero.
    """

    pooled_mode = objective_mode.startswith("pooled_")
    calibration = fit_calibration_from_moments(
        target_raw,
        source_raw,
        mode="pooled" if pooled_mode else "real",
        epsilon=whitening_epsilon,
        detach_statistics=detach_calibration,
    )
    if objective_mode == "paper_affine":
        calibrated_target = moments_from_mean_and_second(
            (target_raw.mean - calibration.center) @ calibration.transform,
            calibration.transform.mT
            @ (
                target_raw.second
                - torch.outer(target_raw.mean, calibration.center)
                - torch.outer(calibration.center, target_raw.mean)
                + torch.outer(calibration.center, calibration.center)
            )
            @ calibration.transform,
        )
        calibrated_source = moments_from_mean_and_second(
            (source_raw.mean - calibration.center) @ calibration.transform,
            calibration.transform.mT
            @ (
                source_raw.second
                - torch.outer(source_raw.mean, calibration.center)
                - torch.outer(calibration.center, source_raw.mean)
                + torch.outer(calibration.center, calibration.center)
            )
            @ calibration.transform,
        )
        distance = frechet_from_moments(
            calibrated_target, calibrated_source
        ).total
    elif objective_mode in {
        "official_regularized",
        "official_mean_only",
        "official_covariance_only",
    }:
        dimension = source_raw.mean.numel()
        identity = torch.eye(
            dimension,
            dtype=source_raw.covariance.dtype,
            device=source_raw.covariance.device,
        )
        source_mean = (
            source_raw.mean - calibration.center
        ) @ calibration.transform
        source_covariance = (
            calibration.transform.mT
            @ (source_raw.covariance + whitening_epsilon * identity)
            @ calibration.transform
        )
        source_regularized = moments_from_mean_and_second(
            source_mean,
            source_covariance + torch.outer(source_mean, source_mean),
        )
        target_identity = moments_from_mean_and_second(
            torch.zeros_like(source_mean), identity
        )
        components = frechet_from_moments(
            target_identity, source_regularized
        )
        if objective_mode == "official_regularized":
            distance = components.total
        elif objective_mode == "official_mean_only":
            distance = components.mean
        else:
            distance = components.covariance
    elif pooled_mode:
        calibrated_target = calibrate_moments(target_raw, calibration)
        calibrated_source = calibrate_moments(source_raw, calibration)
        components = frechet_from_moments(
            calibrated_target, calibrated_source
        )
        if objective_mode == "pooled_full":
            distance = components.total
        elif objective_mode == "pooled_mean_only":
            distance = components.mean
        else:
            distance = components.covariance
    else:
        raise ValueError(f"unknown AdvFD objective mode: {objective_mode!r}")
    return distance, calibration


def population_advfd(
    critic: nn.Module,
    target: GaussianMixture,
    source: GaussianMixture,
    *,
    order: int,
    whitening_epsilon: float,
    detach_real: bool,
    objective_mode: str,
    detach_calibration: bool = True,
) -> tuple[torch.Tensor, AffineCalibration]:
    target_points, target_weights = target.quadrature(order)
    source_points, source_weights = source.quadrature(order)
    target_features = critic(target_points)
    if detach_real:
        target_features = target_features.detach()
    source_features = critic(source_points)
    target_moments = weighted_moments(target_features, target_weights)
    source_moments = weighted_moments(source_features, source_weights)
    return advfd_from_raw_moments(
        target_moments,
        source_moments,
        whitening_epsilon=whitening_epsilon,
        objective_mode=objective_mode,
        detach_calibration=detach_calibration,
    )


def train_advfd_critic(
    target: GaussianMixture,
    source: GaussianMixture,
    *,
    config: CriticConfig,
    seed: int,
    device: torch.device,
) -> tuple[FeatureCritic, list[dict[str, float | int]]]:
    seed_everything(seed)
    critic = FeatureCritic(config).to(device=device, dtype=target.means.dtype)
    optimizer = torch.optim.AdamW(
        critic.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    rows: list[dict[str, float | int]] = []
    eval_every = max(config.steps // 10, 1)
    for step in range(config.steps + 1):
        if step % eval_every == 0 or step == config.steps:
            with torch.no_grad():
                distance, _ = population_advfd(
                    critic,
                    target,
                    source,
                    order=config.quadrature_order,
                    whitening_epsilon=config.whitening_epsilon,
                    detach_real=config.detach_real,
                    detach_calibration=config.detach_calibration,
                    objective_mode=config.objective_mode,
                )
            row: dict[str, float | int] = {
                "step": step,
                "advfd": float(distance),
            }
            field, _ = advfd_functional_field(
                critic,
                target,
                source,
                order=config.quadrature_order,
                whitening_epsilon=config.whitening_epsilon,
                objective_mode=config.objective_mode,
            )
            row.update(
                field_diagnostics(
                    target,
                    source,
                    field,
                    quadrature_order=config.quadrature_order,
                )
            )
            rows.append(row)
        if step == config.steps:
            break
        optimizer.zero_grad(set_to_none=True)
        distance, _ = population_advfd(
            critic,
            target,
            source,
            order=config.quadrature_order,
            whitening_epsilon=config.whitening_epsilon,
            detach_real=config.detach_real,
            detach_calibration=config.detach_calibration,
            objective_mode=config.objective_mode,
        )
        (-distance).backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), config.gradient_clip)
        optimizer.step()
    return critic.eval().requires_grad_(False), rows


def advfd_functional_field(
    critic: nn.Module,
    target: GaussianMixture,
    source: GaussianMixture,
    *,
    order: int,
    whitening_epsilon: float,
    objective_mode: str,
):
    """Build the population input field induced by descending learned FD."""

    target_points, target_weights = target.quadrature(order)
    source_points, source_weights = source.quadrature(order)
    with torch.no_grad():
        target_features = critic(target_points)
        source_features = critic(source_points)
        target_raw = weighted_moments(target_features, target_weights)
        source_raw = weighted_moments(source_features, source_weights)
        pooled_mode = objective_mode.startswith("pooled_")
        calibration = fit_calibration_from_moments(
            target_raw,
            source_raw,
            mode="pooled" if pooled_mode else "real",
            epsilon=whitening_epsilon,
            detach_statistics=True,
        )
        source_moments = calibrated_weighted_moments(
            source_features, source_weights, calibration
        )

    mean = source_moments.mean.detach().requires_grad_(True)
    second = source_moments.second.detach().requires_grad_(True)
    variable_source = moments_from_mean_and_second(mean, second)
    if objective_mode == "paper_affine" or pooled_mode:
        target_moments = calibrated_weighted_moments(
            target_features, target_weights, calibration
        )
        components = frechet_from_moments(
            target_moments.detached(), variable_source
        )
        if objective_mode in {"paper_affine", "pooled_full"}:
            distance = components.total
        elif objective_mode == "pooled_mean_only":
            distance = components.mean
        else:
            distance = components.covariance
    elif objective_mode in {
        "official_regularized",
        "official_mean_only",
        "official_covariance_only",
    }:
        dimension = mean.numel()
        identity = torch.eye(
            dimension, dtype=mean.dtype, device=mean.device
        )
        regularizer = (
            whitening_epsilon
            * calibration.transform.mT
            @ calibration.transform
        )
        regularized_source = moments_from_mean_and_second(
            mean,
            variable_source.covariance
            + regularizer
            + torch.outer(mean, mean),
        )
        target_identity = moments_from_mean_and_second(
            torch.zeros_like(mean), identity
        )
        components = frechet_from_moments(
            target_identity, regularized_source
        )
        if objective_mode == "official_regularized":
            distance = components.total
        elif objective_mode == "official_mean_only":
            distance = components.mean
        else:
            distance = components.covariance
    else:
        raise ValueError(f"unknown AdvFD objective mode: {objective_mode!r}")
    mean_gradient, second_gradient = torch.autograd.grad(
        distance, (mean, second), allow_unused=True
    )
    if mean_gradient is None:
        mean_gradient = torch.zeros_like(mean)
    if second_gradient is None:
        second_gradient = torch.zeros_like(second)
    mean_gradient = mean_gradient.detach()
    second_gradient = second_gradient.detach()
    calibration = AffineCalibration(
        center=calibration.center.detach(), transform=calibration.transform.detach()
    )

    def field(states: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
        states_for_grad = states
        if not states_for_grad.requires_grad:
            states_for_grad = states.detach().requires_grad_(True)
        features = calibration.apply(critic(states_for_grad))
        linear = (features * mean_gradient).sum(dim=1)
        quadratic = torch.einsum(
            "ni,ij,nj->n", features, second_gradient, features
        )
        potential = linear + quadratic
        input_gradient = torch.autograd.grad(
            potential.sum(),
            states_for_grad,
            create_graph=create_graph,
            retain_graph=create_graph,
        )[0]
        if states_for_grad is not states and not create_graph:
            input_gradient = input_gradient.detach()
        return -input_gradient

    return field, float(distance.detach())


def analyze_population_fields(
    target: GaussianMixture,
    source: GaussianMixture,
    *,
    step_sizes: tuple[float, ...],
    quadrature_order: int,
) -> tuple[list[dict[str, Any]], Any]:
    states, weights = source.quadrature(quadrature_order)
    correction = score_correction(target, source, states)
    projection = project_onto_fixed_moment_tangent(states, correction, weights)
    full = score_field(target, source)
    tangent = tangent_field_from_projection(target, source, projection)
    static = static_field(target, source)
    combined = sum_fields(static, tangent)
    fields = {
        "static_fd": static,
        "full_score": full,
        "shape_residual": tangent,
        "static_plus_shape": combined,
    }
    rows = []
    for name, field in fields.items():
        diagnostics = field_diagnostics(
            target, source, field, quadrature_order=quadrature_order
        )
        for step_size in step_sizes:
            finite = finite_pushforward_kl(
                target,
                source,
                field,
                step_size=step_size,
                quadrature_order=quadrature_order,
            )
            rows.append(
                {
                    "method": name,
                    "step_size": step_size,
                    **diagnostics,
                    **finite,
                }
            )
    return rows, projection


def run(
    *,
    output_root: Path,
    seeds: tuple[int, ...],
    config: CriticConfig,
    step_sizes: tuple[float, ...],
    selected_regimes: tuple[str, ...],
    device: torch.device,
) -> None:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    output_root.mkdir(parents=True)
    all_regimes = build_toy_regimes(dtype=torch.float64, device=device)
    unknown = set(selected_regimes) - set(all_regimes)
    if unknown:
        raise ValueError(f"unknown regimes: {sorted(unknown)}")
    regimes = {name: all_regimes[name] for name in selected_regimes}
    population_rows: list[dict[str, Any]] = []
    critic_rows: list[dict[str, Any]] = []
    critic_curve_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    for regime, (target, source) in regimes.items():
        print(f"population regime={regime}", flush=True)
        rows, projection = analyze_population_fields(
            target,
            source,
            step_sizes=step_sizes,
            quadrature_order=20,
        )
        for row in rows:
            population_rows.append(
                {
                    "regime": regime,
                    "frechet_before": frechet_value(target, source),
                    **row,
                }
            )
        projection_rows.append(
            {
                "regime": regime,
                "tangent_rms": float(projection.tangent.square().mean().sqrt()),
                "normal_rms": float(projection.normal.square().mean().sqrt()),
                "mean_derivative_norm": float(projection.mean_derivative.norm()),
                "covariance_derivative_norm": float(
                    projection.covariance_derivative.norm()
                ),
                "orthogonality_error": projection.orthogonality_error,
            }
        )
        for seed in seeds:
            print(f"critic regime={regime} seed={seed}", flush=True)
            critic, curve = train_advfd_critic(
                target,
                source,
                config=config,
                seed=seed,
                device=device,
            )
            for row in curve:
                critic_curve_rows.append({"regime": regime, "seed": seed, **row})
            advfd_field, learned_distance = advfd_functional_field(
                critic,
                target,
                source,
                order=20,
                whitening_epsilon=config.whitening_epsilon,
                objective_mode=config.objective_mode,
            )
            diagnostics = field_diagnostics(
                target, source, advfd_field, quadrature_order=20
            )
            for step_size in step_sizes:
                finite = finite_pushforward_kl(
                    target,
                    source,
                    advfd_field,
                    step_size=step_size,
                    quadrature_order=20,
                )
                critic_rows.append(
                    {
                        "regime": regime,
                        "seed": seed,
                        "method": "learned_advfd",
                        "step_size": step_size,
                        "learned_advfd": learned_distance,
                        **diagnostics,
                        **finite,
                    }
                )
    population = pd.DataFrame(population_rows)
    critics = pd.DataFrame(critic_rows)
    curves = pd.DataFrame(critic_curve_rows)
    projections = pd.DataFrame(projection_rows)
    population.to_csv(output_root / "population_fields.csv", index=False)
    critics.to_csv(output_root / "learned_advfd_fields.csv", index=False)
    curves.to_csv(output_root / "critic_training_curves.csv", index=False)
    projections.to_csv(output_root / "projection_checks.csv", index=False)
    summary = {
        "protocol": "frechet_complementary_score_population_toy_v1",
        "critic": asdict(config),
        "seeds": list(seeds),
        "step_sizes": list(step_sizes),
        "device": str(device),
        "regimes": list(selected_regimes),
        "all_projection_checks_pass": bool(
            (projections["mean_derivative_norm"] < 1e-8).all()
            and (projections["covariance_derivative_norm"] < 1e-8).all()
            and (projections["orthogonality_error"] < 1e-8).all()
        ),
        "shape_only_moments_match": bool(
            population.loc[
                population["regime"] == "shape_only", "frechet_before"
            ].max()
            < 1e-10
        ),
        "score_methods_all_nonascending_first_order": bool(
            (
                population.loc[
                    population["method"].isin(
                        ["full_score", "shape_residual", "static_plus_shape"]
                    ),
                    "reverse_kl_derivative",
                ]
                <= 1e-10
            ).all()
        ),
        "advfd_first_order_failure_count": int(
            (critics["reverse_kl_derivative"] >= 0).sum()
        ),
        "advfd_finite_step_failure_count": int(
            (critics["kl_change"] >= 0).sum()
        ),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    plot_results(population, critics, output_root / "field_comparison.png")
    print(json.dumps(summary, indent=2), flush=True)


def plot_results(
    population: pd.DataFrame, critics: pd.DataFrame, output: Path
) -> None:
    step = float(population["step_size"].min())
    selected = population[population["step_size"] == step]
    critic_selected = critics[critics["step_size"] == step]
    regimes = ["gaussian_only", "shape_only", "combined"]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for index, regime in enumerate(regimes):
        frame = selected[selected["regime"] == regime]
        labels = list(frame["method"])
        values = list(frame["reverse_kl_derivative"])
        critic_values = critic_selected.loc[
            critic_selected["regime"] == regime, "reverse_kl_derivative"
        ]
        labels.append("learned_advfd")
        values.append(float(critic_values.mean()))
        axes[index].bar(range(len(values)), values, color="#3f6f8f")
        if len(critic_values) > 1:
            axes[index].errorbar(
                len(values) - 1,
                float(critic_values.mean()),
                yerr=float(critic_values.std()),
                color="black",
                capsize=3,
            )
        axes[index].axhline(0.0, color="black", linewidth=0.8)
        axes[index].set_xticks(range(len(values)), labels, rotation=35, ha="right")
        axes[index].set_title(regime)
        axes[index].set_ylabel("d KL(q||p) / d tau")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(",") if item)


def parse_strings(value: str) -> tuple[str, ...]:
    return tuple(item for item in value.split(",") if item)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=parse_ints, default=(8101, 8102, 8103))
    parser.add_argument("--critic-steps", type=int, default=1000)
    parser.add_argument(
        "--objective-mode",
        choices=(
            "paper_affine",
            "official_regularized",
            "official_mean_only",
            "official_covariance_only",
        ),
        default="paper_affine",
    )
    parser.add_argument(
        "--regimes",
        type=parse_strings,
        default=("gaussian_only", "shape_only", "combined"),
    )
    parser.add_argument(
        "--step-sizes", type=parse_floats, default=(0.0001, 0.0005, 0.001)
    )
    parser.add_argument("--device", default="cuda:2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CriticConfig(
        steps=args.critic_steps,
        objective_mode=args.objective_mode,
    )
    run(
        output_root=args.output_root,
        seeds=args.seeds,
        config=config,
        step_sizes=args.step_sizes,
        selected_regimes=args.regimes,
        device=torch.device(args.device),
    )


if __name__ == "__main__":
    main()
