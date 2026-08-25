#!/usr/bin/env python3
"""Audit whether AdvFD critic updates respect their affine quotient objective.

The real-whitened AdvFD value is invariant to a common translation of critic
features.  The official stop-gradient update nevertheless differentiates only
through generated features, so it can have a nonzero gradient in this pure
gauge direction.  This script first measures that mismatch directly and then
compares learned critics trained with the stop-gradient surrogate against the
gradient of the recalibrated objective itself.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
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
from experiments.frechet_residual_score_toy import (
    field_diagnostics,
    finite_pushforward_kl,
    weighted_moments,
)
from experiments.run_advfd_monoflow_audit import (
    build_regime,
    monoflow_diagnostics,
)
from experiments.run_frechet_residual_score_toy import (
    CriticConfig,
    FeatureCritic,
    advfd_from_raw_moments,
    parse_floats,
    parse_ints,
    parse_strings,
    train_advfd_critic,
)


@dataclass(frozen=True)
class UpdateMode:
    objective_mode: str
    detach_real: bool
    detach_calibration: bool


UPDATE_MODES = {
    "real_stopgrad": UpdateMode("official_regularized", True, True),
    "real_quotient": UpdateMode("official_regularized", False, False),
    "pooled_stopgrad": UpdateMode("pooled_full", True, True),
    "pooled_quotient": UpdateMode("pooled_full", False, False),
}


def _objective_from_affine_features(
    target_features: torch.Tensor,
    target_weights: torch.Tensor,
    source_features: torch.Tensor,
    source_weights: torch.Tensor,
    matrix: torch.Tensor,
    bias: torch.Tensor,
    *,
    mode: UpdateMode,
    whitening_epsilon: float,
) -> torch.Tensor:
    transformed_target = target_features @ matrix + bias
    transformed_source = source_features @ matrix + bias
    if mode.detach_real:
        transformed_target = transformed_target.detach()
    target_moments = weighted_moments(transformed_target, target_weights)
    source_moments = weighted_moments(transformed_source, source_weights)
    value, _ = advfd_from_raw_moments(
        target_moments,
        source_moments,
        whitening_epsilon=whitening_epsilon,
        objective_mode=mode.objective_mode,
        detach_calibration=mode.detach_calibration,
    )
    return value


def affine_gauge_diagnostic(
    target,
    source,
    *,
    feature_dim: int,
    order: int,
    whitening_epsilon: float,
    seed: int,
    device: torch.device,
) -> list[dict[str, float | int | str]]:
    """Measure gradients and finite changes along a common feature translation."""

    generator = torch.Generator(device=device).manual_seed(seed)
    critic_config = CriticConfig(feature_dim=feature_dim)
    critic = FeatureCritic(critic_config).to(device=device, dtype=target.means.dtype)
    with torch.no_grad():
        for parameter in critic.parameters():
            parameter.normal_(generator=generator, std=0.08)
        target_points, target_weights = target.quadrature(order)
        source_points, source_weights = source.quadrature(order)
        target_features = critic(target_points)
        # Keep the gauge test well-conditioned even when a symmetric toy pair
        # happens to have nearly identical random-feature means.  This fixed
        # base-space discrepancy is transformed together with both populations
        # by the affine post-map below.
        feature_offset = torch.linspace(
            0.2,
            0.8,
            feature_dim,
            dtype=target.means.dtype,
            device=device,
        )
        source_features = critic(source_points) + feature_offset

    matrix = torch.eye(
        feature_dim, dtype=target.means.dtype, device=device, requires_grad=True
    )
    bias = torch.zeros(
        feature_dim, dtype=target.means.dtype, device=device, requires_grad=True
    )
    direction = torch.randn(
        feature_dim,
        dtype=target.means.dtype,
        device=device,
        generator=generator,
    )
    direction = direction / direction.norm()
    finite_shift = 3.0 * direction
    rows: list[dict[str, float | int | str]] = []
    for name, mode in UPDATE_MODES.items():
        value = _objective_from_affine_features(
            target_features,
            target_weights,
            source_features,
            source_weights,
            matrix,
            bias,
            mode=mode,
            whitening_epsilon=whitening_epsilon,
        )
        matrix_gradient, bias_gradient = torch.autograd.grad(
            value, (matrix, bias), allow_unused=True
        )
        if matrix_gradient is None:
            matrix_gradient = torch.zeros_like(matrix)
        if bias_gradient is None:
            bias_gradient = torch.zeros_like(bias)
        with torch.no_grad():
            shifted = _objective_from_affine_features(
                target_features,
                target_weights,
                source_features,
                source_weights,
                matrix,
                bias + finite_shift,
                mode=mode,
                whitening_epsilon=whitening_epsilon,
            )
        rows.append(
            {
                "mode": name,
                "seed": seed,
                "objective": float(value.detach()),
                "finite_translation_change": float(shifted - value.detach()),
                "translation_gradient_norm": float(bias_gradient.norm()),
                "translation_directional_derivative": float(
                    bias_gradient @ direction
                ),
                "linear_gradient_norm": float(matrix_gradient.norm()),
            }
        )
    return rows


def critic_feature_statistics(
    critic, target, source, *, order: int
) -> dict[str, float]:
    with torch.no_grad():
        target_points, target_weights = target.quadrature(order)
        source_points, source_weights = source.quadrature(order)
        target_features = critic(target_points)
        source_features = critic(source_points)
        pooled_features = torch.cat((target_features, source_features), dim=0)
        pooled_weights = torch.cat(
            (0.5 * target_weights, 0.5 * source_weights), dim=0
        )
        pooled = weighted_moments(pooled_features, pooled_weights)
        covariance = pooled.covariance
        trace = torch.trace(covariance)
        effective_rank = trace.square() / covariance.square().sum().clamp_min(1e-30)
        parameter_square = sum(
            parameter.detach().square().sum() for parameter in critic.parameters()
        )
        output_bias = list(critic.modules())[-1].bias
        return {
            "target_feature_rms": float(target_features.square().mean().sqrt()),
            "source_feature_rms": float(source_features.square().mean().sqrt()),
            "pooled_feature_mean_norm": float(pooled.mean.norm()),
            "pooled_feature_effective_rank": float(effective_rank),
            "critic_parameter_norm": float(parameter_square.sqrt()),
            "output_bias_norm": float(output_bias.detach().norm()),
        }


def run(
    output_root: Path,
    *,
    regimes: tuple[str, ...],
    noise_sigmas: tuple[float, ...],
    modes: tuple[str, ...],
    seeds: tuple[int, ...],
    critic_steps: int,
    quadrature_order: int,
    sample_count: int,
    displacement_rms: float,
    whitening_epsilon: float,
    device: torch.device,
) -> None:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    unknown = set(modes) - set(UPDATE_MODES)
    if unknown:
        raise ValueError(f"unknown modes: {sorted(unknown)}")
    output_root.mkdir(parents=True)

    gauge_rows: list[dict[str, float | int | str]] = []
    learned_rows: list[dict[str, float | int | str]] = []
    curve_rows: list[dict[str, float | int | str]] = []
    for regime_name in regimes:
        target_clean, source_clean = build_regime(regime_name, device=device)
        for noise_sigma in noise_sigmas:
            target = target_clean.convolve_isotropic(noise_sigma)
            source = source_clean.convolve_isotropic(noise_sigma)
            for seed in seeds:
                for row in affine_gauge_diagnostic(
                    target,
                    source,
                    feature_dim=8,
                    order=quadrature_order,
                    whitening_epsilon=whitening_epsilon,
                    seed=seed,
                    device=device,
                ):
                    gauge_rows.append(
                        {
                            "regime": regime_name,
                            "noise_sigma": noise_sigma,
                            **row,
                        }
                    )
            for mode_name in modes:
                mode = UPDATE_MODES[mode_name]
                for seed in seeds:
                    print(
                        f"regime={regime_name} sigma={noise_sigma:g} "
                        f"mode={mode_name} seed={seed}",
                        flush=True,
                    )
                    config = CriticConfig(
                        steps=critic_steps,
                        whitening_epsilon=whitening_epsilon,
                        quadrature_order=min(quadrature_order, 16),
                        objective_mode=mode.objective_mode,
                        detach_real=mode.detach_real,
                        detach_calibration=mode.detach_calibration,
                    )
                    critic, curve = train_advfd_critic(
                        target,
                        source,
                        config=config,
                        seed=seed,
                        device=device,
                    )
                    for point in curve:
                        curve_rows.append(
                            {
                                "regime": regime_name,
                                "noise_sigma": noise_sigma,
                                "mode": mode_name,
                                "seed": seed,
                                **point,
                            }
                        )
                    context = build_feature_force_context(
                        critic,
                        target,
                        source,
                        order=quadrature_order,
                        whitening_epsilon=whitening_epsilon,
                        objective_mode=mode.objective_mode,
                    )
                    field = learned_pullback_field(critic, context, mode="transpose")
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
                            sample_seed=seed + 91_003,
                        )
                    )
                    diagnostics.update(
                        critic_feature_statistics(
                            critic, target, source, order=quadrature_order
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
                    learned_rows.append(
                        {
                            "regime": regime_name,
                            "noise_sigma": noise_sigma,
                            "mode": mode_name,
                            "seed": seed,
                            "critic_advfd_initial": curve[0]["advfd"],
                            "critic_advfd_final": curve[-1]["advfd"],
                            "critic_advfd_gain": curve[-1]["advfd"]
                            - curve[0]["advfd"],
                            **diagnostics,
                            **finite,
                        }
                    )

    gauges = pd.DataFrame(gauge_rows)
    learned = pd.DataFrame(learned_rows)
    curves = pd.DataFrame(curve_rows)
    gauges.to_csv(output_root / "affine_gauge_gradients.csv", index=False)
    learned.to_csv(output_root / "learned_critic_audit.csv", index=False)
    curves.to_csv(output_root / "critic_curves.csv", index=False)

    aggregate = learned.groupby(
        ["regime", "noise_sigma", "mode"], as_index=False
    ).agg(
        score_cosine=("score_cosine", "mean"),
        positive_work=("positive_score_work_fraction", "mean"),
        potential_spearman=("potential_log_ratio_spearman", "mean"),
        kl_change=("kl_change", "mean"),
        feature_rms=("source_feature_rms", "mean"),
        feature_effective_rank=("pooled_feature_effective_rank", "mean"),
        output_bias_norm=("output_bias_norm", "mean"),
    )
    aggregate.to_csv(output_root / "aggregate.csv", index=False)

    gauge_aggregate = gauges.groupby("mode", as_index=False).agg(
        translation_gradient_norm=("translation_gradient_norm", "mean"),
        finite_translation_change=("finite_translation_change", "mean"),
        linear_gradient_norm=("linear_gradient_norm", "mean"),
    )
    gauge_aggregate.to_csv(output_root / "gauge_aggregate.csv", index=False)

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    labels = [
        f"{row.regime}\ns={row.noise_sigma:g}\n{row.mode}"
        for row in aggregate.itertuples()
    ]
    for axis, metric, title in (
        (axes[0], "score_cosine", "cosine with score correction"),
        (axes[1], "kl_change", "matched-step KL change"),
        (axes[2], "feature_effective_rank", "critic feature effective rank"),
    ):
        axis.bar(range(len(aggregate)), aggregate[metric])
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(title)
        axis.set_xticks(range(len(aggregate)), labels, rotation=50, ha="right")
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_root / "quotient_gradient_audit.png", dpi=180)
    plt.close(figure)

    stopgrad_gauge = gauge_aggregate[
        gauge_aggregate["mode"].str.endswith("stopgrad")
    ]["translation_gradient_norm"]
    quotient_gauge = gauge_aggregate[
        gauge_aggregate["mode"].str.endswith("quotient")
    ]["translation_gradient_norm"]
    summary = {
        "protocol": "advfd_quotient_gradient_audit_v1",
        "regimes": list(regimes),
        "noise_sigmas": list(noise_sigmas),
        "modes": list(modes),
        "seeds": list(seeds),
        "critic_steps": critic_steps,
        "whitening_epsilon": whitening_epsilon,
        "stopgrad_translation_gradient_mean": float(stopgrad_gauge.mean()),
        "quotient_translation_gradient_mean": float(quotient_gauge.mean()),
        "maximum_finite_translation_change": float(
            gauges["finite_translation_change"].abs().max()
        ),
        "all_finite": bool(
            all(
                math.isfinite(float(value))
                for value in learned.select_dtypes(include="number").to_numpy().flat
            )
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
    parser.add_argument("--noise-sigmas", type=parse_floats, default=(0.0, 0.4))
    parser.add_argument(
        "--modes", type=parse_strings, default=tuple(UPDATE_MODES)
    )
    parser.add_argument(
        "--seeds", type=parse_ints, default=(20260824, 20260825, 20260826)
    )
    parser.add_argument("--critic-steps", type=int, default=1000)
    parser.add_argument("--quadrature-order", type=int, default=16)
    parser.add_argument("--sample-count", type=int, default=4096)
    parser.add_argument("--displacement-rms", type=float, default=0.01)
    parser.add_argument("--whitening-epsilon", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.output_root,
        regimes=args.regimes,
        noise_sigmas=args.noise_sigmas,
        modes=args.modes,
        seeds=args.seeds,
        critic_steps=args.critic_steps,
        quadrature_order=args.quadrature_order,
        sample_count=args.sample_count,
        displacement_rms=args.displacement_rms,
        whitening_epsilon=args.whitening_epsilon,
        device=torch.device(args.device),
    )
