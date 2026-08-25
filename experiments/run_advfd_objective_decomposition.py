#!/usr/bin/env python3
"""Decompose AdvFD critic training and generator fields into mean/covariance parts."""

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
    build_toy_regimes,
    field_diagnostics,
    finite_pushforward_kl,
    weighted_inner,
)
from experiments.run_advfd_smoothed_retraining_transport import (
    build_rotated_ring_pair,
)
from experiments.run_frechet_residual_score_toy import (
    CriticConfig,
    advfd_functional_field,
    parse_floats,
    parse_ints,
    parse_strings,
    train_advfd_critic,
)


OBJECTIVE_MODES = (
    "official_regularized",
    "official_mean_only",
    "official_covariance_only",
)
SHORT_NAMES = {
    "official_regularized": "full",
    "official_mean_only": "mean",
    "official_covariance_only": "covariance",
}


def build_regime(name: str, *, device: torch.device):
    if name == "shape_only":
        return build_toy_regimes(dtype=torch.float64, device=device)[name]
    if name == "rotated_ring":
        return build_rotated_ring_pair(rotation=0.22, device=device)
    raise ValueError(f"unknown regime: {name}")


def field_geometry(fields: dict[str, object], source, *, order: int) -> dict[str, float]:
    states, weights = source.quadrature(order)
    normalized = weights / weights.sum()
    values = {name: field(states, False) for name, field in fields.items()}

    def norm(name: str) -> torch.Tensor:
        return weighted_inner(values[name], values[name], normalized).sqrt()

    def cosine(first: str, second: str) -> float:
        denominator = norm(first) * norm(second)
        if float(denominator) <= 1e-14:
            return float("nan")
        return float(
            weighted_inner(values[first], values[second], normalized)
            / denominator
        )

    return {
        "full_field_rms": float(norm("full") / math.sqrt(source.dimension)),
        "mean_field_rms": float(norm("mean") / math.sqrt(source.dimension)),
        "covariance_field_rms": float(
            norm("covariance") / math.sqrt(source.dimension)
        ),
        "mean_covariance_cosine": cosine("mean", "covariance"),
        "full_mean_cosine": cosine("full", "mean"),
        "full_covariance_cosine": cosine("full", "covariance"),
    }


def run(
    output_root: Path,
    *,
    regimes: tuple[str, ...],
    noise_sigmas: tuple[float, ...],
    seeds: tuple[int, ...],
    critic_steps: int,
    displacement_rms: tuple[float, ...],
    quadrature_order: int,
    device: torch.device,
) -> None:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    output_root.mkdir(parents=True)
    rows: list[dict[str, float | int | str]] = []
    curves: list[dict[str, float | int | str]] = []
    for regime in regimes:
        target_clean, source_clean = build_regime(regime, device=device)
        for sigma in noise_sigmas:
            target = target_clean.convolve_isotropic(sigma)
            source = source_clean.convolve_isotropic(sigma)
            for seed in seeds:
                for train_mode in OBJECTIVE_MODES:
                    print(
                        f"regime={regime} sigma={sigma:g} seed={seed} "
                        f"train={SHORT_NAMES[train_mode]}",
                        flush=True,
                    )
                    config = CriticConfig(
                        steps=critic_steps,
                        objective_mode=train_mode,
                        detach_real=True,
                        quadrature_order=min(quadrature_order, 16),
                    )
                    critic, critic_curve = train_advfd_critic(
                        target,
                        source,
                        config=config,
                        seed=seed,
                        device=device,
                    )
                    for curve_row in critic_curve:
                        curves.append(
                            {
                                "regime": regime,
                                "noise_sigma": sigma,
                                "seed": seed,
                                "trained_objective": SHORT_NAMES[train_mode],
                                **curve_row,
                            }
                        )
                    fields = {}
                    values = {}
                    for field_mode in OBJECTIVE_MODES:
                        short = SHORT_NAMES[field_mode]
                        fields[short], values[short] = advfd_functional_field(
                            critic,
                            target,
                            source,
                            order=quadrature_order,
                            whitening_epsilon=config.whitening_epsilon,
                            objective_mode=field_mode,
                        )
                    geometry = field_geometry(
                        fields, source, order=quadrature_order
                    )
                    for field_component, field in fields.items():
                        diagnostics = field_diagnostics(
                            target,
                            source,
                            field,
                            quadrature_order=quadrature_order,
                        )
                        velocity_rms = diagnostics["velocity_rms"]
                        for target_rms in displacement_rms:
                            if velocity_rms <= 1e-14:
                                finite = {
                                    "kl_before": float("nan"),
                                    "kl_after": float("nan"),
                                    "kl_change": float("nan"),
                                    "positive_jacobian_fraction": float("nan"),
                                    "minimum_jacobian_determinant": float("nan"),
                                }
                                step_size = float("nan")
                            else:
                                step_size = target_rms / velocity_rms
                                finite = finite_pushforward_kl(
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
                                    "trained_objective": SHORT_NAMES[train_mode],
                                    "field_component": field_component,
                                    "target_displacement_rms": target_rms,
                                    "step_size": step_size,
                                    "trained_objective_initial": float(
                                        critic_curve[0]["advfd"]
                                    ),
                                    "trained_objective_final": float(
                                        critic_curve[-1]["advfd"]
                                    ),
                                    "same_critic_full_value": values["full"],
                                    "same_critic_mean_value": values["mean"],
                                    "same_critic_covariance_value": values[
                                        "covariance"
                                    ],
                                    **geometry,
                                    **diagnostics,
                                    **finite,
                                }
                            )
    frame = pd.DataFrame(rows)
    curve_frame = pd.DataFrame(curves)
    frame.to_csv(output_root / "objective_decomposition.csv", index=False)
    curve_frame.to_csv(output_root / "critic_curves.csv", index=False)
    selected = frame[frame["target_displacement_rms"] == min(displacement_rms)]
    summary = {
        "protocol": "advfd_mean_covariance_objective_decomposition_v1",
        "regimes": list(regimes),
        "noise_sigmas": list(noise_sigmas),
        "seeds": list(seeds),
        "critic_steps": critic_steps,
        "displacement_rms": list(displacement_rms),
        "quadrature_order": quadrature_order,
        "critic_template": asdict(
            CriticConfig(
                steps=critic_steps,
                objective_mode="official_regularized",
                quadrature_order=min(quadrature_order, 16),
            )
        ),
        "full_value_decomposition_max_error": float(
            (
                selected["same_critic_full_value"]
                - selected["same_critic_mean_value"]
                - selected["same_critic_covariance_value"]
            )
            .abs()
            .max()
        ),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    plot(selected, output_root / "objective_decomposition.png")
    print(json.dumps(summary, indent=2), flush=True)


def plot(frame: pd.DataFrame, output: Path) -> None:
    aggregate = frame.groupby(
        ["regime", "noise_sigma", "trained_objective", "field_component"],
        as_index=False,
    ).agg(
        score_cosine=("score_cosine", "mean"),
        kl_change=("kl_change", "mean"),
    )
    regimes = list(aggregate["regime"].unique())
    figure, axes = plt.subplots(
        len(regimes), 2, figsize=(12.0, 4.2 * len(regimes)), squeeze=False
    )
    for row_index, regime in enumerate(regimes):
        selected = aggregate[aggregate["regime"] == regime].copy()
        labels = (
            selected["trained_objective"]
            + "->"
            + selected["field_component"]
            + ", s="
            + selected["noise_sigma"].map(lambda value: f"{value:g}")
        )
        axes[row_index, 0].bar(range(len(selected)), selected["score_cosine"])
        axes[row_index, 1].bar(range(len(selected)), selected["kl_change"])
        for axis in axes[row_index]:
            axis.set_xticks(range(len(selected)), labels, rotation=80, ha="right")
            axis.axhline(0.0, color="black", linewidth=0.8)
            axis.grid(axis="y", alpha=0.2)
        axes[row_index, 0].set_ylabel(f"{regime}: cosine with score")
        axes[row_index, 1].set_ylabel(f"{regime}: finite KL change")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--regimes", type=parse_strings, default=("shape_only", "rotated_ring")
    )
    parser.add_argument("--noise-sigmas", type=parse_floats, default=(0.0, 0.4))
    parser.add_argument("--seeds", type=parse_ints, default=(8401, 8402, 8403))
    parser.add_argument("--critic-steps", type=int, default=2000)
    parser.add_argument(
        "--displacement-rms", type=parse_floats, default=(1e-4, 1e-3, 1e-2)
    )
    parser.add_argument("--quadrature-order", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.output_root,
        regimes=args.regimes,
        noise_sigmas=args.noise_sigmas,
        seeds=args.seeds,
        critic_steps=args.critic_steps,
        displacement_rms=args.displacement_rms,
        quadrature_order=args.quadrature_order,
        device=torch.device(args.device),
    )
