#!/usr/bin/env python3
"""Test AdvFD feature-transport pullbacks with learned two-dimensional critics."""

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

from experiments.advfd_feature_pullback import (
    build_feature_force_context,
    feature_force_and_jacobian,
    learned_pullback_field,
)
from experiments.frechet_residual_score_toy import (
    field_diagnostics,
    score_field,
    weighted_inner,
    weighted_moments,
)
from experiments.run_advfd_objective_decomposition import build_regime
from experiments.run_frechet_residual_score_toy import (
    CriticConfig,
    FeatureCritic,
    parse_floats,
    parse_ints,
    parse_strings,
    train_advfd_critic,
)


def pullback_geometry(
    critic,
    context,
    source,
    field,
    *,
    order: int,
) -> dict[str, float]:
    states, weights = source.quadrature(order)
    normalized = weights / weights.sum()
    feature_force, jacobians = feature_force_and_jacobian(critic, context, states)
    input_velocity = field(states, False)
    induced = torch.einsum("nfi,ni->nf", jacobians, input_velocity)
    residual = induced - feature_force
    force_norm = weighted_inner(feature_force, feature_force, normalized).sqrt()
    residual_norm = weighted_inner(residual, residual, normalized).sqrt()
    induced_norm = weighted_inner(induced, induced, normalized).sqrt()
    alignment = weighted_inner(induced, feature_force, normalized)
    gram = torch.einsum("nfi,nfj->nij", jacobians, jacobians)
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(1e-30)
    condition = eigenvalues[:, -1] / eigenvalues[:, 0]
    return {
        "feature_force_rms": float(force_norm / math.sqrt(feature_force.shape[1])),
        "induced_feature_rms": float(
            induced_norm / math.sqrt(feature_force.shape[1])
        ),
        "feature_tracking_relative_error": float(
            residual_norm / force_norm.clamp_min(torch.finfo(states.dtype).eps)
        ),
        "feature_tracking_cosine": float(
            alignment
            / (force_norm * induced_norm).clamp_min(torch.finfo(states.dtype).eps)
        ),
        "jacobian_gram_condition_median": float(condition.median()),
        "jacobian_gram_condition_q95": float(torch.quantile(condition, 0.95)),
    }


def finite_pushforward_kl_numerical(
    target,
    source,
    field,
    *,
    step_size: float,
    quadrature_order: int,
    difference_epsilon: float = 1e-5,
) -> dict[str, float]:
    states, weights = source.quadrature(quadrature_order)
    velocity = field(states, False)
    columns = []
    for coordinate in range(source.dimension):
        offset = torch.zeros_like(states)
        offset[:, coordinate] = difference_epsilon
        columns.append(
            (field(states + offset, False) - field(states - offset, False))
            / (2.0 * difference_epsilon)
        )
    jacobian = torch.stack(columns, dim=2)
    identity = torch.eye(source.dimension, dtype=states.dtype, device=states.device)
    determinant = torch.linalg.det(identity[None] + step_size * jacobian)
    normalized = weights / weights.sum()
    source_log_probability, _ = source.log_prob_and_score(states)
    target_log_before, _ = target.log_prob_and_score(states)
    before = (normalized * (source_log_probability - target_log_before)).sum()
    valid = determinant > 0
    if not bool(valid.all()):
        return {
            "kl_before": float(before),
            "kl_after": float("nan"),
            "kl_change": float("nan"),
            "positive_jacobian_fraction": float(valid.double().mean()),
            "minimum_jacobian_determinant": float(determinant.min()),
        }
    transported = states + step_size * velocity
    target_log_after, _ = target.log_prob_and_score(transported)
    after = (
        normalized
        * (source_log_probability - determinant.log() - target_log_after)
    ).sum()
    return {
        "kl_before": float(before),
        "kl_after": float(after),
        "kl_change": float(after - before),
        "positive_jacobian_fraction": 1.0,
        "minimum_jacobian_determinant": float(determinant.min()),
    }


def run(
    output_root: Path,
    *,
    regimes: tuple[str, ...],
    noise_sigmas: tuple[float, ...],
    seeds: tuple[int, ...],
    critic_steps: int,
    relative_dampings: tuple[float, ...],
    displacement_rms: tuple[float, ...],
    quadrature_order: int,
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
            for seed in seeds:
                config = CriticConfig(
                    steps=critic_steps,
                    objective_mode="official_regularized",
                    quadrature_order=min(quadrature_order, 16),
                )
                print(
                    f"regime={regime} sigma={sigma:g} seed={seed} train",
                    flush=True,
                )
                critic, curve = train_advfd_critic(
                    target,
                    source,
                    config=config,
                    seed=seed,
                    device=device,
                )
                for curve_row in curve:
                    curves.append(
                        {
                            "regime": regime,
                            "noise_sigma": sigma,
                            "seed": seed,
                            **curve_row,
                        }
                    )
                context = build_feature_force_context(
                    critic,
                    target,
                    source,
                    order=quadrature_order,
                    whitening_epsilon=config.whitening_epsilon,
                )
                fields = {
                    "transpose": learned_pullback_field(
                        critic, context, mode="transpose"
                    ),
                    **{
                        f"pseudoinverse_d{damping:g}": learned_pullback_field(
                            critic,
                            context,
                            mode="pseudoinverse",
                            relative_damping=damping,
                        )
                        for damping in relative_dampings
                    },
                    "score": score_field(target, source),
                }
                for name, field in fields.items():
                    diagnostics = field_diagnostics(
                        target, source, field, quadrature_order=quadrature_order
                    )
                    geometry = (
                        pullback_geometry(
                            critic,
                            context,
                            source,
                            field,
                            order=quadrature_order,
                        )
                        if name != "score"
                        else {
                            "feature_force_rms": float("nan"),
                            "induced_feature_rms": float("nan"),
                            "feature_tracking_relative_error": float("nan"),
                            "feature_tracking_cosine": float("nan"),
                            "jacobian_gram_condition_median": float("nan"),
                            "jacobian_gram_condition_q95": float("nan"),
                        }
                    )
                    for displacement in displacement_rms:
                        step_size = displacement / diagnostics["velocity_rms"]
                        finite = finite_pushforward_kl_numerical(
                            target,
                            source,
                            field,
                            step_size=step_size,
                            quadrature_order=quadrature_order,
                        )
                        rows.append(
                            {
                                "regime": regime,
                                "noise_sigma": sigma,
                                "seed": seed,
                                "pullback": name,
                                "target_displacement_rms": displacement,
                                "step_size": step_size,
                                "advfd_value": context["distance"],
                                **diagnostics,
                                **geometry,
                                **finite,
                            }
                        )
    frame = pd.DataFrame(rows)
    curve_frame = pd.DataFrame(curves)
    frame.to_csv(output_root / "learned_pullback_audit.csv", index=False)
    curve_frame.to_csv(output_root / "critic_curves.csv", index=False)
    summary = {
        "protocol": "advfd_learned_pullback_audit_v1",
        "regimes": list(regimes),
        "noise_sigmas": list(noise_sigmas),
        "seeds": list(seeds),
        "critic_steps": critic_steps,
        "relative_dampings": list(relative_dampings),
        "displacement_rms": list(displacement_rms),
        "quadrature_order": quadrature_order,
        "critic": asdict(
            CriticConfig(
                steps=critic_steps,
                objective_mode="official_regularized",
                quadrature_order=min(quadrature_order, 16),
            )
        ),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    plot(frame, output_root / "learned_pullback_audit.png")


def plot(frame: pd.DataFrame, output: Path) -> None:
    selected = frame[
        frame["target_displacement_rms"]
        == frame["target_displacement_rms"].min()
    ]
    aggregate = selected.groupby(
        ["regime", "noise_sigma", "pullback"], as_index=False
    ).agg(
        score_cosine=("score_cosine", "mean"),
        kl_change=("kl_change", "mean"),
        tracking_error=("feature_tracking_relative_error", "mean"),
    )
    conditions = list(
        aggregate[["regime", "noise_sigma"]].drop_duplicates().itertuples(index=False)
    )
    figure, axes = plt.subplots(len(conditions), 3, figsize=(15, 4 * len(conditions)))
    if len(conditions) == 1:
        axes = axes[None, :]
    for row, condition in enumerate(conditions):
        regime, sigma = condition
        data = aggregate[
            (aggregate["regime"] == regime)
            & (aggregate["noise_sigma"] == sigma)
        ]
        labels = data["pullback"]
        axes[row, 0].bar(labels, data["score_cosine"])
        axes[row, 1].bar(labels, data["kl_change"])
        axes[row, 2].bar(labels, data["tracking_error"])
        axes[row, 0].set_ylabel(f"{regime}, sigma={sigma:g}")
        axes[row, 0].set_title("Score cosine")
        axes[row, 1].set_title("Finite KL change")
        axes[row, 2].set_title("Feature tracking error")
        for axis in axes[row]:
            axis.tick_params(axis="x", rotation=25)
            axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--regimes", type=parse_strings, default=("shape_only", "rotated_ring")
    )
    parser.add_argument("--noise-sigmas", type=parse_floats, default=(0.0, 0.4))
    parser.add_argument("--seeds", type=parse_ints, default=(20260824, 20260825, 20260826))
    parser.add_argument("--critic-steps", type=int, default=5000)
    parser.add_argument(
        "--relative-dampings", type=parse_floats, default=(0.0, 0.001, 0.01, 0.1)
    )
    parser.add_argument(
        "--displacement-rms", type=parse_floats, default=(1e-4, 1e-3)
    )
    parser.add_argument("--quadrature-order", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(
        args.output_root,
        regimes=args.regimes,
        noise_sigmas=args.noise_sigmas,
        seeds=args.seeds,
        critic_steps=args.critic_steps,
        relative_dampings=args.relative_dampings,
        displacement_rms=args.displacement_rms,
        quadrature_order=args.quadrature_order,
        device=torch.device(args.device),
    )


if __name__ == "__main__":
    main()
