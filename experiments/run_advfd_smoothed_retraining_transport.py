#!/usr/bin/env python3
"""Test whether AdvFD exposure produces useful retrained transport.

The clean generator is a location-parameterized Gaussian mixture.  At every
outer round, the AdvFD critic sees equally noised real and fake mixtures.  Its
input field is averaged within each fake component and backpropagated to the
clean component mean.  Retraining the critic after every update makes this a
closed discrepancy-exposure/correction loop rather than a one-step cosine
audit.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
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
    GaussianMixture,
    field_diagnostics,
    frechet_value,
    score_field,
)
from experiments.run_frechet_residual_score_toy import (
    CriticConfig,
    advfd_functional_field,
    parse_floats,
    parse_ints,
    parse_strings,
    train_advfd_critic,
)
from experiments.advfd_feature_pullback import (
    build_feature_force_context,
    learned_pullback_field,
)


def build_rotated_ring_pair(
    *,
    rotation: float,
    radius: float = 2.0,
    component_sigma: float = 0.18,
    components: int = 8,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> tuple[GaussianMixture, GaussianMixture]:
    """Return moment-matched mixtures separated only by angular structure."""

    if components < 3:
        raise ValueError("components must be at least three")
    if component_sigma <= 0:
        raise ValueError("component_sigma must be positive")
    device = torch.device(device)
    angles = torch.arange(components, dtype=dtype, device=device)
    angles = angles * (2.0 * math.pi / components)
    target_means = radius * torch.stack((angles.cos(), angles.sin()), dim=1)
    source_angles = angles + float(rotation)
    source_means = radius * torch.stack(
        (source_angles.cos(), source_angles.sin()), dim=1
    )
    weights = torch.full(
        (components,), 1.0 / components, dtype=dtype, device=device
    )
    covariance = component_sigma**2 * torch.eye(
        2, dtype=dtype, device=device
    )
    target = GaussianMixture(weights, target_means, covariance)
    source = GaussianMixture(weights.clone(), source_means, covariance.clone())
    return target, source


def reverse_kl(
    target: GaussianMixture,
    source: GaussianMixture,
    *,
    quadrature_order: int,
) -> float:
    states, weights = source.quadrature(quadrature_order)
    source_log_prob, _ = source.log_prob_and_score(states)
    target_log_prob, _ = target.log_prob_and_score(states)
    normalized = weights / weights.sum()
    return float((normalized * (source_log_prob - target_log_prob)).sum())


def component_average_field(
    source: GaussianMixture,
    field,
    *,
    quadrature_order: int,
) -> torch.Tensor:
    """Average a sample-space field under every source component."""

    states, mixture_weights = source.quadrature(quadrature_order)
    points_per_component = len(states) // source.components
    values = field(states, False).reshape(
        source.components, points_per_component, source.dimension
    )
    conditional_weights = mixture_weights.reshape(
        source.components, points_per_component
    ) / source.weights[:, None]
    conditional_weights = conditional_weights / conditional_weights.sum(
        dim=1, keepdim=True
    )
    return (conditional_weights[..., None] * values).sum(dim=1)


def weighted_direction_rms(
    direction: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    return (weights[:, None] * direction.square()).sum().sqrt()


def normalized_mean_update(
    source: GaussianMixture,
    direction: torch.Tensor,
    *,
    displacement_rms: float,
) -> tuple[GaussianMixture, float]:
    if direction.shape != source.means.shape:
        raise ValueError("direction must match source means")
    rms = weighted_direction_rms(direction, source.weights)
    if float(rms) <= 1e-14:
        return source, 0.0
    scale = float(displacement_rms) / float(rms)
    updated = GaussianMixture(
        weights=source.weights.clone(),
        means=source.means + scale * direction,
        component_covariance=source.component_covariance.clone(),
    )
    return updated, scale


def cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    denominator = first.norm() * second.norm()
    if float(denominator) <= 1e-14:
        return float("nan")
    return float((first * second).sum() / denominator)


def nearest_target_mean_rms(
    target: GaussianMixture, source: GaussianMixture
) -> float:
    distances = torch.cdist(source.means, target.means)
    return float(distances.min(dim=1).values.square().mean().sqrt())


def distribution_metrics(
    target_clean: GaussianMixture,
    source_clean: GaussianMixture,
    *,
    noise_sigma: float,
    quadrature_order: int,
) -> dict[str, float]:
    target_noised = target_clean.convolve_isotropic(noise_sigma)
    source_noised = source_clean.convolve_isotropic(noise_sigma)
    return {
        "clean_reverse_kl": reverse_kl(
            target_clean, source_clean, quadrature_order=quadrature_order
        ),
        "noised_reverse_kl": reverse_kl(
            target_noised, source_noised, quadrature_order=quadrature_order
        ),
        "clean_frechet": frechet_value(target_clean, source_clean),
        "nearest_target_mean_rms": nearest_target_mean_rms(
            target_clean, source_clean
        ),
    }


def run_method(
    *,
    method: str,
    target_clean: GaussianMixture,
    source_initial: GaussianMixture,
    noise_sigma: float,
    seed: int,
    rounds: int,
    displacement_rms: float,
    critic_config: CriticConfig,
    quadrature_order: int,
    device: torch.device,
) -> list[dict[str, float | int | str]]:
    valid_methods = {
        "advfd",
        "advfd_transpose",
        "advfd_pseudoinverse_d0",
        "advfd_pseudoinverse_d0.1",
        "score",
    }
    if method not in valid_methods:
        raise ValueError(f"unknown method: {method}")
    source_clean = source_initial
    rows: list[dict[str, float | int | str]] = []
    for round_index in range(rounds + 1):
        target_noised = target_clean.convolve_isotropic(noise_sigma)
        source_noised = source_clean.convolve_isotropic(noise_sigma)
        oracle_field = score_field(target_noised, source_noised)
        oracle_direction = component_average_field(
            source_noised,
            oracle_field,
            quadrature_order=quadrature_order,
        )
        learned_advfd = float("nan")
        critic_advfd_initial = float("nan")
        critic_advfd_gain = float("nan")
        field_score_cosine = 1.0
        if method == "score":
            update_field = oracle_field
        else:
            critic, critic_curve = train_advfd_critic(
                target_noised,
                source_noised,
                config=critic_config,
                seed=seed + 100_003 * round_index,
                device=device,
            )
            critic_advfd_initial = float(critic_curve[0]["advfd"])
            if method in {"advfd", "advfd_transpose"}:
                update_field, learned_advfd = advfd_functional_field(
                    critic,
                    target_noised,
                    source_noised,
                    order=quadrature_order,
                    whitening_epsilon=critic_config.whitening_epsilon,
                    objective_mode=critic_config.objective_mode,
                )
            else:
                context = build_feature_force_context(
                    critic,
                    target_noised,
                    source_noised,
                    order=quadrature_order,
                    whitening_epsilon=critic_config.whitening_epsilon,
                )
                relative_damping = (
                    0.0 if method.endswith("d0") else 0.1
                )
                update_field = learned_pullback_field(
                    critic,
                    context,
                    mode="pseudoinverse",
                    relative_damping=relative_damping,
                )
                learned_advfd = context["distance"]
            critic_advfd_gain = learned_advfd - critic_advfd_initial
            diagnostics = field_diagnostics(
                target_noised,
                source_noised,
                update_field,
                quadrature_order=quadrature_order,
            )
            field_score_cosine = diagnostics["score_cosine"]
        direction = component_average_field(
            source_noised,
            update_field,
            quadrature_order=quadrature_order,
        )
        metrics = distribution_metrics(
            target_clean,
            source_clean,
            noise_sigma=noise_sigma,
            quadrature_order=quadrature_order,
        )
        rows.append(
            {
                "method": method,
                "noise_sigma": noise_sigma,
                "seed": seed,
                "round": round_index,
                "cumulative_displacement_budget": round_index
                * displacement_rms,
                "learned_advfd": learned_advfd,
                "critic_advfd_initial": critic_advfd_initial,
                "critic_advfd_gain": critic_advfd_gain,
                "field_score_cosine": field_score_cosine,
                "component_direction_score_cosine": cosine(
                    direction, oracle_direction
                ),
                "direction_rms": float(
                    weighted_direction_rms(direction, source_clean.weights)
                ),
                "oracle_direction_rms": float(
                    weighted_direction_rms(
                        oracle_direction, source_clean.weights
                    )
                ),
                **metrics,
            }
        )
        if round_index == rounds:
            break
        source_clean, _ = normalized_mean_update(
            source_clean,
            direction,
            displacement_rms=displacement_rms,
        )
    return rows


def plot_results(frame: pd.DataFrame, output: Path) -> None:
    aggregate = frame.groupby(
        ["method", "noise_sigma", "round"], as_index=False
    ).agg(
        clean_kl=("clean_reverse_kl", "mean"),
        noised_kl=("noised_reverse_kl", "mean"),
        score_cosine=("component_direction_score_cosine", "mean"),
        nearest_rms=("nearest_target_mean_rms", "mean"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))
    for (method, sigma), group in aggregate.groupby(["method", "noise_sigma"]):
        label = f"{method}, sigma={sigma:g}"
        style = "--" if method == "score" else "-"
        axes[0, 0].plot(group["round"], group["clean_kl"], style, label=label)
        axes[0, 1].plot(group["round"], group["noised_kl"], style, label=label)
        axes[1, 0].plot(
            group["round"], group["score_cosine"], style, label=label
        )
        axes[1, 1].plot(group["round"], group["nearest_rms"], style, label=label)
    axes[0, 0].set_ylabel("clean KL(q || p)")
    axes[0, 1].set_ylabel("smoothed KL(q_sigma || p_sigma)")
    axes[1, 0].set_ylabel("component update cosine with score")
    axes[1, 1].set_ylabel("nearest target-mean RMS")
    for axis in axes.flat:
        axis.set_xlabel("outer correction round")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def run(
    output_root: Path,
    *,
    noise_sigmas: tuple[float, ...],
    methods: tuple[str, ...],
    seeds: tuple[int, ...],
    rounds: int,
    displacement_rms: float,
    rotation: float,
    critic_steps: int,
    quadrature_order: int,
    device: torch.device,
) -> None:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    output_root.mkdir(parents=True)
    target, source = build_rotated_ring_pair(
        rotation=rotation, device=device
    )
    config = CriticConfig(
        steps=critic_steps,
        objective_mode="official_regularized",
        detach_real=True,
        quadrature_order=min(quadrature_order, 16),
    )
    rows: list[dict[str, float | int | str]] = []
    for sigma in noise_sigmas:
        for seed in seeds:
            for method in methods:
                print(
                    f"method={method} sigma={sigma:g} seed={seed}",
                    flush=True,
                )
                rows.extend(
                    run_method(
                        method=method,
                        target_clean=target,
                        source_initial=source,
                        noise_sigma=sigma,
                        seed=seed,
                        rounds=rounds,
                        displacement_rms=displacement_rms,
                        critic_config=config,
                        quadrature_order=quadrature_order,
                        device=device,
                    )
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "transport_curves.csv", index=False)
    final = frame[frame["round"] == rounds]
    initial = frame[frame["round"] == 0][
        ["method", "noise_sigma", "seed", "clean_reverse_kl", "noised_reverse_kl"]
    ].rename(
        columns={
            "clean_reverse_kl": "clean_reverse_kl_initial",
            "noised_reverse_kl": "noised_reverse_kl_initial",
        }
    )
    final = final.merge(initial, on=["method", "noise_sigma", "seed"])
    final["clean_kl_change"] = (
        final["clean_reverse_kl"] - final["clean_reverse_kl_initial"]
    )
    final["noised_kl_change"] = (
        final["noised_reverse_kl"] - final["noised_reverse_kl_initial"]
    )
    final.to_csv(output_root / "final_summary.csv", index=False)
    summary = {
        "protocol": "advfd_smoothed_retrained_transport_v1",
        "noise_sigmas": list(noise_sigmas),
        "methods": list(methods),
        "seeds": list(seeds),
        "rounds": rounds,
        "displacement_rms_per_round": displacement_rms,
        "rotation": rotation,
        "quadrature_order": quadrature_order,
        "critic": asdict(config),
        "initial_clean_frechet": frechet_value(target, source),
        "all_score_runs_lower_smoothed_kl": bool(
            (final.loc[final["method"] == "score", "noised_kl_change"] < 0).all()
        ),
        "advfd_runs_lower_clean_kl": int(
            (
                final.loc[
                    final["method"].str.startswith("advfd"),
                    "clean_kl_change",
                ]
                < 0
            ).sum()
        ),
        "advfd_run_count": int(
            final["method"].str.startswith("advfd").sum()
        ),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    plot_results(frame, output_root / "transport_curves.png")
    print(json.dumps(summary, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--noise-sigmas", type=parse_floats, default=(0.0, 0.2, 0.4, 0.7)
    )
    parser.add_argument(
        "--methods",
        type=parse_strings,
        default=(
            "advfd_transpose",
            "advfd_pseudoinverse_d0",
            "advfd_pseudoinverse_d0.1",
            "score",
        ),
    )
    parser.add_argument("--seeds", type=parse_ints, default=(8301, 8302, 8303))
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--displacement-rms", type=float, default=0.02)
    parser.add_argument("--rotation", type=float, default=0.22)
    parser.add_argument("--critic-steps", type=int, default=1000)
    parser.add_argument("--quadrature-order", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.output_root,
        noise_sigmas=args.noise_sigmas,
        methods=args.methods,
        seeds=args.seeds,
        rounds=args.rounds,
        displacement_rms=args.displacement_rms,
        rotation=args.rotation,
        critic_steps=args.critic_steps,
        quadrature_order=args.quadrature_order,
        device=torch.device(args.device),
    )
