#!/usr/bin/env python3
"""Test decoupled AdvFD discrepancy discovery and generator transport.

The full AdvFD critic can use both feature mean and covariance to discover a
representation, while the generator may follow either the original full
Frechet field or only the mean field in that frozen representation.  The
source is a Gaussian-mixture generator whose component means are updated, so
the clean and smoothed reverse KL remain available after every outer round.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
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
from experiments.frechet_residual_score_toy import field_diagnostics, score_field
from experiments.run_advfd_monoflow_audit import build_regime
from experiments.run_advfd_smoothed_retraining_transport import (
    component_average_field,
    cosine,
    distribution_metrics,
    normalized_mean_update,
    weighted_direction_rms,
)
from experiments.run_frechet_residual_score_toy import (
    CriticConfig,
    parse_ints,
    parse_strings,
    train_advfd_critic,
)


@dataclass(frozen=True)
class Protocol:
    critic_objective: str | None
    generator_objective: str | None
    detach_real: bool = False
    detach_calibration: bool = False


PROTOCOLS = {
    "score": Protocol(None, None),
    "full_stopgrad_full": Protocol(
        "official_regularized", "official_regularized", True, True
    ),
    "full_quotient_full": Protocol(
        "official_regularized", "official_regularized"
    ),
    "full_quotient_mean": Protocol(
        "official_regularized", "official_mean_only"
    ),
    "mean_quotient_mean": Protocol("official_mean_only", "official_mean_only"),
    "pooled_full_quotient_full": Protocol("pooled_full", "pooled_full"),
    "pooled_full_quotient_mean": Protocol("pooled_full", "pooled_mean_only"),
    "pooled_mean_quotient_mean": Protocol(
        "pooled_mean_only", "pooled_mean_only"
    ),
}


def run_protocol(
    *,
    protocol_name: str,
    target_clean,
    source_initial,
    noise_sigma: float,
    seed: int,
    rounds: int,
    displacement_rms: float,
    critic_steps: int,
    quadrature_order: int,
    whitening_epsilon: float,
    device: torch.device,
) -> list[dict[str, float | int | str]]:
    protocol = PROTOCOLS[protocol_name]
    source_clean = source_initial
    rows = []
    for round_index in range(rounds + 1):
        target_noised = target_clean.convolve_isotropic(noise_sigma)
        source_noised = source_clean.convolve_isotropic(noise_sigma)
        oracle_field = score_field(target_noised, source_noised)
        oracle_direction = component_average_field(
            source_noised, oracle_field, quadrature_order=quadrature_order
        )
        critic_value = float("nan")
        if protocol_name == "score":
            update_field = oracle_field
            field_score_cosine = 1.0
        else:
            config = CriticConfig(
                steps=critic_steps,
                whitening_epsilon=whitening_epsilon,
                quadrature_order=min(quadrature_order, 16),
                objective_mode=protocol.critic_objective,
                detach_real=protocol.detach_real,
                detach_calibration=protocol.detach_calibration,
            )
            critic, critic_curve = train_advfd_critic(
                target_noised,
                source_noised,
                config=config,
                seed=seed + 100_003 * round_index,
                device=device,
            )
            critic_value = float(critic_curve[-1]["advfd"])
            context = build_feature_force_context(
                critic,
                target_noised,
                source_noised,
                order=quadrature_order,
                whitening_epsilon=whitening_epsilon,
                objective_mode=protocol.generator_objective,
            )
            update_field = learned_pullback_field(
                critic, context, mode="transpose"
            )
            field_score_cosine = field_diagnostics(
                target_noised,
                source_noised,
                update_field,
                quadrature_order=quadrature_order,
            )["score_cosine"]
        direction = component_average_field(
            source_noised, update_field, quadrature_order=quadrature_order
        )
        metrics = distribution_metrics(
            target_clean,
            source_clean,
            noise_sigma=noise_sigma,
            quadrature_order=quadrature_order,
        )
        rows.append(
            {
                "protocol": protocol_name,
                "seed": seed,
                "round": round_index,
                "critic_steps": critic_steps,
                "critic_value": critic_value,
                "field_score_cosine": field_score_cosine,
                "component_direction_score_cosine": cosine(
                    direction, oracle_direction
                ),
                "direction_rms": float(
                    weighted_direction_rms(direction, source_clean.weights)
                ),
                **metrics,
            }
        )
        if round_index == rounds:
            break
        source_clean, _ = normalized_mean_update(
            source_clean, direction, displacement_rms=displacement_rms
        )
    return rows


def run(
    output_root: Path,
    *,
    regimes: tuple[str, ...],
    noise_sigma: float,
    protocols: tuple[str, ...],
    seeds: tuple[int, ...],
    rounds: int,
    displacement_rms: float,
    critic_steps: int,
    quadrature_order: int,
    whitening_epsilon: float,
    device: torch.device,
) -> None:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    unknown = set(protocols) - set(PROTOCOLS)
    if unknown:
        raise ValueError(f"unknown protocols: {sorted(unknown)}")
    output_root.mkdir(parents=True)
    rows = []
    for regime_name in regimes:
        target, source = build_regime(regime_name, device=device)
        for protocol_name in protocols:
            for seed in seeds:
                print(
                    f"regime={regime_name} protocol={protocol_name} seed={seed}",
                    flush=True,
                )
                audited = run_protocol(
                    protocol_name=protocol_name,
                    target_clean=target,
                    source_initial=source,
                    noise_sigma=noise_sigma,
                    seed=seed,
                    rounds=rounds,
                    displacement_rms=displacement_rms,
                    critic_steps=critic_steps,
                    quadrature_order=quadrature_order,
                    whitening_epsilon=whitening_epsilon,
                    device=device,
                )
                for row in audited:
                    rows.append(
                        {"regime": regime_name, "noise_sigma": noise_sigma, **row}
                    )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "transport_curves.csv", index=False)

    initial = frame[frame["round"] == 0][
        ["regime", "protocol", "seed", "clean_reverse_kl", "noised_reverse_kl"]
    ].rename(
        columns={
            "clean_reverse_kl": "initial_clean_reverse_kl",
            "noised_reverse_kl": "initial_noised_reverse_kl",
        }
    )
    final = frame[frame["round"] == rounds].merge(
        initial, on=["regime", "protocol", "seed"]
    )
    final["clean_kl_change"] = (
        final["clean_reverse_kl"] - final["initial_clean_reverse_kl"]
    )
    final["noised_kl_change"] = (
        final["noised_reverse_kl"] - final["initial_noised_reverse_kl"]
    )
    final.to_csv(output_root / "final_summary.csv", index=False)
    aggregate = final.groupby(["regime", "protocol"], as_index=False).agg(
        clean_kl_change=("clean_kl_change", "mean"),
        noised_kl_change=("noised_kl_change", "mean"),
        final_clean_kl=("clean_reverse_kl", "mean"),
        final_noised_kl=("noised_reverse_kl", "mean"),
        final_nearest_rms=("nearest_target_mean_rms", "mean"),
    )
    aggregate.to_csv(output_root / "aggregate.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    curves = frame.groupby(["regime", "protocol", "round"], as_index=False).agg(
        clean_kl=("clean_reverse_kl", "mean"),
        noised_kl=("noised_reverse_kl", "mean"),
    )
    for (regime_name, protocol_name), group in curves.groupby(
        ["regime", "protocol"]
    ):
        label = f"{regime_name}: {protocol_name}"
        axes[0].plot(group["round"], group["clean_kl"], label=label)
        axes[1].plot(group["round"], group["noised_kl"], label=label)
    axes[0].set_ylabel("clean KL(q || p)")
    axes[1].set_ylabel("smoothed KL(q_sigma || p_sigma)")
    for axis in axes:
        axis.set_xlabel("outer correction round")
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output_root / "transport_curves.png", dpi=180)
    plt.close(figure)

    summary = {
        "protocol": "advfd_decoupled_transport_v1",
        "regimes": list(regimes),
        "noise_sigma": noise_sigma,
        "protocols": list(protocols),
        "seeds": list(seeds),
        "rounds": rounds,
        "displacement_rms": displacement_rms,
        "critic_steps": critic_steps,
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
        "--protocols",
        type=parse_strings,
        default=(
            "score",
            "full_stopgrad_full",
            "full_quotient_full",
            "full_quotient_mean",
            "mean_quotient_mean",
        ),
    )
    parser.add_argument(
        "--seeds", type=parse_ints, default=(20260824, 20260825, 20260826)
    )
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--displacement-rms", type=float, default=0.02)
    parser.add_argument("--critic-steps", type=int, default=300)
    parser.add_argument("--quadrature-order", type=int, default=16)
    parser.add_argument("--whitening-epsilon", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.output_root,
        regimes=args.regimes,
        noise_sigma=args.noise_sigma,
        protocols=args.protocols,
        seeds=args.seeds,
        rounds=args.rounds,
        displacement_rms=args.displacement_rms,
        critic_steps=args.critic_steps,
        quadrature_order=args.quadrature_order,
        whitening_epsilon=args.whitening_epsilon,
        device=torch.device(args.device),
    )
