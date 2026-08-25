#!/usr/bin/env python3
"""Show that equal AdvFD witnesses can induce opposite correction fields.

AdvFD depends on feature values at the real and generated samples.  Generator
updates additionally depend on the input Jacobian of that feature map.  This
experiment constructs feature maps with identical values, and hence identical
AdvFD objectives, but helpful, vanishing, or harmful pullback directions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn

from experiments.frechet_residual_score_toy import (
    GaussianMixture,
    score_correction,
    weighted_inner,
    weighted_moments,
)
from experiments.run_frechet_residual_score_toy import advfd_from_raw_moments


class AmbiguousWitness(nn.Module):
    """A feature family agreeing at x in {-2, -1, 1, 2}.

    The added polynomial vanishes at all four support points, while its
    derivative at both generated points +/-2 equals 24.  Consequently
    ``coefficient=0`` preserves the helpful base Jacobian,
    ``coefficient=-1/24`` cancels it, and more negative values reverse it.
    """

    def __init__(self, coefficient: float) -> None:
        super().__init__()
        self.coefficient = float(coefficient)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        coordinate = states[:, 0]
        null_value = coordinate * (coordinate.square() - 1.0) * (
            coordinate.square() - 4.0
        )
        return (coordinate + self.coefficient * null_value)[:, None]


def advfd_particle_field(
    target_states: torch.Tensor,
    target_weights: torch.Tensor,
    source_states: torch.Tensor,
    source_weights: torch.Tensor,
    witness: nn.Module,
    *,
    objective_mode: str,
    whitening_epsilon: float,
) -> tuple[float, torch.Tensor]:
    """Return AdvFD and its negative functional particle gradient."""

    variable_source = source_states.detach().clone().requires_grad_(True)
    target_features = witness(target_states).detach()
    source_features = witness(variable_source)
    target_moments = weighted_moments(target_features, target_weights)
    source_moments = weighted_moments(source_features, source_weights)
    distance, _ = advfd_from_raw_moments(
        target_moments,
        source_moments,
        whitening_epsilon=whitening_epsilon,
        objective_mode=objective_mode,
    )
    particle_gradient = torch.autograd.grad(distance, variable_source)[0]
    normalized_weights = source_weights / source_weights.sum()
    field = -particle_gradient / normalized_weights[:, None]
    return float(distance.detach()), field.detach()


def empirical_rows(
    coefficients: tuple[float, ...],
    *,
    objective_modes: tuple[str, ...],
    whitening_epsilon: float,
    step_size: float,
    device: torch.device,
) -> list[dict[str, Any]]:
    target = torch.tensor([[-1.0], [1.0]], dtype=torch.float64, device=device)
    source = torch.tensor([[-2.0], [2.0]], dtype=torch.float64, device=device)
    weights = torch.full((2,), 0.5, dtype=torch.float64, device=device)
    before_w2 = float((source - target).square().mean())
    rows: list[dict[str, Any]] = []
    for objective_mode in objective_modes:
        for coefficient in coefficients:
            witness = AmbiguousWitness(coefficient).to(device)
            distance, field = advfd_particle_field(
                target,
                weights,
                source,
                weights,
                witness,
                objective_mode=objective_mode,
                whitening_epsilon=whitening_epsilon,
            )
            after = source + step_size * field
            after_w2 = float((after - target).square().mean())
            rows.append(
                {
                    "setting": "empirical_exact",
                    "objective_mode": objective_mode,
                    "coefficient": coefficient,
                    "advfd": distance,
                    "field_at_minus_two": float(field[0, 0]),
                    "field_at_plus_two": float(field[1, 0]),
                    "paired_w2_before": before_w2,
                    "paired_w2_after": after_w2,
                    "paired_w2_change": after_w2 - before_w2,
                }
            )
    return rows


def smooth_rows(
    coefficients: tuple[float, ...],
    standard_deviations: tuple[float, ...],
    *,
    objective_modes: tuple[str, ...],
    whitening_epsilon: float,
    quadrature_order: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    weights = torch.full((2,), 0.5, dtype=torch.float64, device=device)
    target_means = torch.tensor([[-1.0], [1.0]], dtype=torch.float64, device=device)
    source_means = torch.tensor([[-2.0], [2.0]], dtype=torch.float64, device=device)
    rows: list[dict[str, Any]] = []
    for standard_deviation in standard_deviations:
        covariance = torch.tensor(
            [[standard_deviation**2]], dtype=torch.float64, device=device
        )
        target = GaussianMixture(weights, target_means, covariance)
        source = GaussianMixture(weights, source_means, covariance)
        target_states, target_weights = target.quadrature(quadrature_order)
        source_states, source_weights = source.quadrature(quadrature_order)
        true_correction = score_correction(target, source, source_states)
        score_descent = -weighted_inner(
            true_correction, true_correction, source_weights
        )
        for objective_mode in objective_modes:
            for coefficient in coefficients:
                witness = AmbiguousWitness(coefficient).to(device)
                distance, field = advfd_particle_field(
                    target_states,
                    target_weights,
                    source_states,
                    source_weights,
                    witness,
                    objective_mode=objective_mode,
                    whitening_epsilon=whitening_epsilon,
                )
                alignment = weighted_inner(
                    true_correction, field, source_weights
                )
                correction_norm = weighted_inner(
                    true_correction, true_correction, source_weights
                ).sqrt()
                field_norm = weighted_inner(
                    field, field, source_weights
                ).sqrt()
                rows.append(
                    {
                        "setting": "smooth_gaussian_mixture",
                        "objective_mode": objective_mode,
                        "standard_deviation": standard_deviation,
                        "coefficient": coefficient,
                        "advfd": distance,
                        "field_rms": float(field_norm),
                        "score_cosine": float(
                            alignment
                            / (correction_norm * field_norm).clamp_min(1e-15)
                        ),
                        "reverse_kl_derivative": float(-alignment),
                        "score_reverse_kl_derivative": float(score_descent),
                    }
                )
    return rows


def plot_smooth(rows: pd.DataFrame, output: Path) -> None:
    official = rows[rows["objective_mode"] == "official_regularized"]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for coefficient, frame in official.groupby("coefficient"):
        ordered = frame.sort_values("standard_deviation")
        label = f"c={coefficient:.5g}"
        axes[0].plot(
            ordered["standard_deviation"], ordered["advfd"], marker="o", label=label
        )
        axes[1].plot(
            ordered["standard_deviation"],
            ordered["reverse_kl_derivative"],
            marker="o",
            label=label,
        )
    axes[0].set_title("AdvFD witness value")
    axes[0].set_xlabel("component standard deviation")
    axes[0].set_ylabel("official regularized AdvFD")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_title("Induced correction quality")
    axes[1].set_xlabel("component standard deviation")
    axes[1].set_ylabel("d KL(q||p) / d tau")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def run(output_root: Path, device: torch.device) -> None:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    output_root.mkdir(parents=True)
    coefficients = (0.0, -1.0 / 24.0, -0.1)
    objective_modes = ("paper_affine", "official_regularized")
    empirical = pd.DataFrame(
        empirical_rows(
            coefficients,
            objective_modes=objective_modes,
            whitening_epsilon=1e-3,
            step_size=1e-3,
            device=device,
        )
    )
    smooth = pd.DataFrame(
        smooth_rows(
            coefficients,
            (0.2, 0.1, 0.05, 0.02),
            objective_modes=objective_modes,
            whitening_epsilon=1e-3,
            quadrature_order=80,
            device=device,
        )
    )
    empirical.to_csv(output_root / "empirical_exact.csv", index=False)
    smooth.to_csv(output_root / "smooth_limit.csv", index=False)
    exact_spread = empirical.groupby("objective_mode")["advfd"].agg(
        lambda values: float(values.max() - values.min())
    )
    narrow = smooth[smooth["standard_deviation"] == 0.02]
    summary = {
        "protocol": "advfd_witness_jacobian_counterexample_v1",
        "empirical_advfd_identical": bool((exact_spread < 1e-12).all()),
        "empirical_helpful_zero_harmful": bool(
            empirical.groupby("coefficient")["paired_w2_change"].mean().loc[0.0]
            < 0
            and abs(
                empirical.groupby("coefficient")["paired_w2_change"]
                .mean()
                .loc[-1.0 / 24.0]
            )
            < 1e-10
            and empirical.groupby("coefficient")["paired_w2_change"].mean().loc[-0.1]
            > 0
        ),
        "smooth_bad_witness_increases_kl": bool(
            (
                narrow.loc[
                    narrow["coefficient"] == -0.1, "reverse_kl_derivative"
                ]
                > 0
            ).all()
        ),
        "smooth_score_strictly_decreases_kl": bool(
            (narrow["score_reverse_kl_derivative"] < 0).all()
        ),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    plot_smooth(smooth, output_root / "smooth_counterexample.png")
    print(json.dumps(summary, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.output_root, torch.device(args.device))
