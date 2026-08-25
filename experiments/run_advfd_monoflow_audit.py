#!/usr/bin/env python3
"""Test whether learned AdvFD forces satisfy density-ratio flow conditions.

For a fixed critic, descending feature Frechet distance produces a conservative
input-space field.  A much stronger condition is needed for that field to be a
valid MonoFlow correction: its potential must increase with log(q / p), or
equivalently its velocity must be a positive multiple of score_p - score_q.
This audit separates the official mean and covariance branches and tests that
condition on smooth distributions where the exact density ratio is available.
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

from experiments.advfd_feature_pullback import (
    build_feature_force_context,
    feature_potential,
    learned_pullback_field,
)
from experiments.frechet_residual_score_toy import (
    build_toy_regimes,
    field_diagnostics,
    finite_pushforward_kl,
    score_correction,
    weighted_inner,
)
from experiments.run_advfd_smoothed_retraining_transport import (
    build_rotated_ring_pair,
)
from experiments.run_frechet_residual_score_toy import (
    CriticConfig,
    parse_floats,
    parse_ints,
    parse_strings,
    train_advfd_critic,
)


OBJECTIVE_MODES = {
    "full": "official_regularized",
    "mean": "official_mean_only",
    "covariance": "official_covariance_only",
    "pooled_full": "pooled_full",
    "pooled_mean": "pooled_mean_only",
    "pooled_covariance": "pooled_covariance_only",
}


def build_regime(name: str, *, device: torch.device):
    if name == "rotated_ring":
        return build_rotated_ring_pair(rotation=0.22, device=device)
    if name == "shape_only":
        return build_toy_regimes(dtype=torch.float64, device=device)[name]
    raise ValueError(f"unknown regime: {name}")


def rankdata(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values)
    ranks = torch.empty_like(values)
    ranks[order] = torch.arange(
        len(values), dtype=values.dtype, device=values.device
    )
    return ranks


def correlation(first: torch.Tensor, second: torch.Tensor) -> float:
    first = first - first.mean()
    second = second - second.mean()
    denominator = first.square().mean().sqrt() * second.square().mean().sqrt()
    if float(denominator.detach()) <= 1e-15:
        return float("nan")
    return float(((first * second).mean() / denominator).detach())


def effective_fraction(values: torch.Tensor) -> float:
    values = values.abs()
    denominator = values.square().mean()
    if float(denominator.detach()) <= 1e-30:
        return 0.0
    return float((values.mean().square() / denominator).detach())


def quantile(values: torch.Tensor, probability: float) -> float:
    return float(torch.quantile(values.detach(), probability))


def monoflow_diagnostics(
    critic,
    context,
    target,
    source,
    *,
    sample_count: int,
    sample_seed: int,
) -> dict[str, float]:
    states = source.sample(sample_count, seed=sample_seed)
    target_log_prob, target_score = target.log_prob_and_score(states)
    source_log_prob, source_score = source.log_prob_and_score(states)
    log_ratio = source_log_prob - target_log_prob
    correction = target_score - source_score
    field = learned_pullback_field(critic, context, mode="transpose")
    velocity = field(states, False)
    with torch.no_grad():
        potential = feature_potential(critic, context, states)

    work = (velocity * correction).sum(dim=1)
    correction_square = correction.square().sum(dim=1)
    multiplier = work / correction_square.clamp_min(1e-20)
    projected = multiplier[:, None] * correction
    residual = velocity - projected
    total_energy = velocity.square().sum(dim=1).mean()
    residual_fraction = residual.square().sum(dim=1).mean() / total_energy.clamp_min(
        1e-20
    )
    potential_rank = rankdata(potential)
    ratio_rank = rankdata(log_ratio)
    positive_work = work > 0
    return {
        "potential_log_ratio_pearson": correlation(potential, log_ratio),
        "potential_log_ratio_spearman": correlation(potential_rank, ratio_rank),
        "positive_score_work_fraction": float(
            positive_work.double().mean().detach()
        ),
        "mean_score_work": float(work.mean().detach()),
        "mono_residual_energy_fraction": float(residual_fraction.detach()),
        "multiplier_median": quantile(multiplier, 0.5),
        "multiplier_q05": quantile(multiplier, 0.05),
        "multiplier_q95": quantile(multiplier, 0.95),
        "velocity_effective_fraction": effective_fraction(velocity.norm(dim=1)),
        "score_work_effective_fraction": effective_fraction(work),
    }


def run(
    output_root: Path,
    *,
    regimes: tuple[str, ...],
    noise_sigmas: tuple[float, ...],
    branches: tuple[str, ...],
    seeds: tuple[int, ...],
    critic_steps: int,
    quadrature_order: int,
    sample_count: int,
    displacement_rms: float,
    device: torch.device,
) -> None:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    output_root.mkdir(parents=True)
    rows: list[dict[str, float | int | str]] = []
    curves: list[dict[str, float | int | str]] = []
    for regime in regimes:
        target_clean, source_clean = build_regime(regime, device=device)
        for noise_sigma in noise_sigmas:
            target = target_clean.convolve_isotropic(noise_sigma)
            source = source_clean.convolve_isotropic(noise_sigma)
            for branch in branches:
                if branch not in OBJECTIVE_MODES:
                    raise ValueError(f"unknown branch: {branch}")
                objective_mode = OBJECTIVE_MODES[branch]
                for seed in seeds:
                    print(
                        f"regime={regime} sigma={noise_sigma:g} "
                        f"branch={branch} seed={seed}",
                        flush=True,
                    )
                    config = CriticConfig(
                        steps=critic_steps,
                        objective_mode=objective_mode,
                        detach_real=True,
                        quadrature_order=min(quadrature_order, 16),
                    )
                    critic, curve = train_advfd_critic(
                        target,
                        source,
                        config=config,
                        seed=seed,
                        device=device,
                    )
                    for point in curve:
                        curves.append(
                            {
                                "regime": regime,
                                "noise_sigma": noise_sigma,
                                "branch": branch,
                                "seed": seed,
                                **point,
                            }
                        )
                    context = build_feature_force_context(
                        critic,
                        target,
                        source,
                        order=quadrature_order,
                        whitening_epsilon=config.whitening_epsilon,
                        objective_mode=objective_mode,
                    )
                    field = learned_pullback_field(
                        critic, context, mode="transpose"
                    )
                    diagnostics = field_diagnostics(
                        target,
                        source,
                        field,
                        quadrature_order=quadrature_order,
                    )
                    diagnostics.update(
                        monoflow_diagnostics(
                            critic,
                            context,
                            target,
                            source,
                            sample_count=sample_count,
                            sample_seed=seed + 71_003,
                        )
                    )
                    velocity_rms = diagnostics["velocity_rms"]
                    if velocity_rms > 1e-14:
                        finite = finite_pushforward_kl(
                            target,
                            source,
                            field,
                            step_size=displacement_rms / velocity_rms,
                            quadrature_order=quadrature_order,
                        )
                    else:
                        finite = {
                            "kl_before": float("nan"),
                            "kl_after": float("nan"),
                            "kl_change": float("nan"),
                            "positive_jacobian_fraction": float("nan"),
                            "minimum_jacobian_determinant": float("nan"),
                        }
                    rows.append(
                        {
                            "regime": regime,
                            "noise_sigma": noise_sigma,
                            "branch": branch,
                            "objective_mode": objective_mode,
                            "seed": seed,
                            "critic_advfd_initial": curve[0]["advfd"],
                            "critic_advfd_final": curve[-1]["advfd"],
                            "critic_advfd_gain": curve[-1]["advfd"]
                            - curve[0]["advfd"],
                            "matched_displacement_rms": displacement_rms,
                            **diagnostics,
                            **finite,
                        }
                    )

    frame = pd.DataFrame(rows)
    curve_frame = pd.DataFrame(curves)
    frame.to_csv(output_root / "monoflow_audit.csv", index=False)
    curve_frame.to_csv(output_root / "critic_curves.csv", index=False)
    aggregate = frame.groupby(
        ["regime", "noise_sigma", "branch"], as_index=False
    ).agg(
        score_cosine_mean=("score_cosine", "mean"),
        score_cosine_std=("score_cosine", "std"),
        positive_work_mean=("positive_score_work_fraction", "mean"),
        potential_spearman_mean=("potential_log_ratio_spearman", "mean"),
        mono_residual_mean=("mono_residual_energy_fraction", "mean"),
        kl_change_mean=("kl_change", "mean"),
        velocity_effective_fraction_mean=("velocity_effective_fraction", "mean"),
    )
    aggregate.to_csv(output_root / "aggregate.csv", index=False)

    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.5))
    metrics = (
        ("score_cosine_mean", "global cosine with score correction"),
        ("positive_work_mean", "q-mass with positive score work"),
        ("potential_spearman_mean", "potential vs log(q/p) Spearman"),
        ("kl_change_mean", "matched-step KL change"),
    )
    labels = [
        f"{row.regime}\ns={row.noise_sigma:g}\n{row.branch}"
        for row in aggregate.itertuples()
    ]
    for axis, (metric, title) in zip(axes.flat, metrics):
        axis.bar(range(len(aggregate)), aggregate[metric])
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(title)
        axis.set_xticks(range(len(aggregate)), labels, rotation=45, ha="right")
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_root / "monoflow_audit.png", dpi=180)
    plt.close(figure)

    full = frame[frame["branch"].isin(("full", "pooled_full"))]
    covariance = frame[
        frame["branch"].isin(("covariance", "pooled_covariance"))
    ]
    mean = frame[frame["branch"].isin(("mean", "pooled_mean"))]
    summary = {
        "protocol": "advfd_monoflow_audit_v1",
        "regimes": list(regimes),
        "noise_sigmas": list(noise_sigmas),
        "branches": list(branches),
        "seeds": list(seeds),
        "critic": asdict(
            CriticConfig(
                steps=critic_steps,
                quadrature_order=min(quadrature_order, 16),
            )
        ),
        "full_all_positive_score_work": bool(
            (full["positive_score_work_fraction"] == 1.0).all()
        ),
        "covariance_all_positive_score_work": bool(
            (covariance["positive_score_work_fraction"] == 1.0).all()
        ),
        "mean_all_positive_score_work": bool(
            (mean["positive_score_work_fraction"] == 1.0).all()
        ),
        "full_all_lower_finite_kl": bool((full["kl_change"] < 0).all()),
        "covariance_all_lower_finite_kl": bool(
            (covariance["kl_change"] < 0).all()
        ),
        "mean_all_lower_finite_kl": bool((mean["kl_change"] < 0).all()),
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
    parser.add_argument("--noise-sigmas", type=parse_floats, default=(0.0, 0.4))
    parser.add_argument(
        "--branches", type=parse_strings, default=("full", "mean", "covariance")
    )
    parser.add_argument(
        "--seeds", type=parse_ints, default=(20260824, 20260825, 20260826)
    )
    parser.add_argument("--critic-steps", type=int, default=1000)
    parser.add_argument("--quadrature-order", type=int, default=16)
    parser.add_argument("--sample-count", type=int, default=4096)
    parser.add_argument("--displacement-rms", type=float, default=0.01)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.output_root,
        regimes=args.regimes,
        noise_sigmas=args.noise_sigmas,
        branches=args.branches,
        seeds=args.seeds,
        critic_steps=args.critic_steps,
        quadrature_order=args.quadrature_order,
        sample_count=args.sample_count,
        displacement_rms=args.displacement_rms,
        device=torch.device(args.device),
    )
