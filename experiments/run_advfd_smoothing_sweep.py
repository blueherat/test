#!/usr/bin/env python3
"""Test whether Gaussian support smoothing repairs AdvFD correction geometry."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from experiments.frechet_residual_score_toy import (
    build_toy_regimes,
    field_diagnostics,
    finite_pushforward_kl,
    score_field,
)
from experiments.run_frechet_residual_score_toy import (
    CriticConfig,
    advfd_functional_field,
    parse_floats,
    parse_ints,
    train_advfd_critic,
)


def run(
    output_root: Path,
    *,
    noise_sigmas: tuple[float, ...],
    seeds: tuple[int, ...],
    critic_steps: int,
    device: torch.device,
) -> None:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    output_root.mkdir(parents=True)
    target_base, source_base = build_toy_regimes(
        dtype=torch.float64, device=device
    )["shape_only"]
    config = CriticConfig(
        steps=critic_steps,
        objective_mode="official_regularized",
        detach_real=True,
    )
    final_rows = []
    curve_rows = []
    for noise_sigma in noise_sigmas:
        target = target_base.convolve_isotropic(noise_sigma)
        source = source_base.convolve_isotropic(noise_sigma)
        oracle = field_diagnostics(
            target,
            source,
            score_field(target, source),
            quadrature_order=20,
        )
        for seed in seeds:
            print(f"sigma={noise_sigma:g} seed={seed}", flush=True)
            critic, curve = train_advfd_critic(
                target,
                source,
                config=config,
                seed=seed,
                device=device,
            )
            for row in curve:
                curve_rows.append(
                    {"noise_sigma": noise_sigma, "seed": seed, **row}
                )
            field, learned_distance = advfd_functional_field(
                critic,
                target,
                source,
                order=20,
                whitening_epsilon=config.whitening_epsilon,
                objective_mode=config.objective_mode,
            )
            diagnostics = field_diagnostics(
                target, source, field, quadrature_order=20
            )
            velocity_rms = diagnostics["velocity_rms"]
            for displacement_rms in (1e-5, 5e-5, 1e-4):
                step_size = displacement_rms / max(velocity_rms, 1e-12)
                finite = finite_pushforward_kl(
                    target,
                    source,
                    field,
                    step_size=step_size,
                    quadrature_order=20,
                )
                final_rows.append(
                    {
                        "noise_sigma": noise_sigma,
                        "seed": seed,
                        "learned_advfd": learned_distance,
                        "target_displacement_rms": displacement_rms,
                        "step_size": step_size,
                        "oracle_score_rms": oracle["score_rms"],
                        "oracle_reverse_kl_derivative": oracle[
                            "reverse_kl_derivative"
                        ],
                        **diagnostics,
                        **finite,
                    }
                )
    finals = pd.DataFrame(final_rows)
    curves = pd.DataFrame(curve_rows)
    finals.to_csv(output_root / "smoothing_final_fields.csv", index=False)
    curves.to_csv(output_root / "smoothing_training_curves.csv", index=False)
    selected = finals[finals["target_displacement_rms"] == 1e-5]
    summary = {
        "protocol": "advfd_gaussian_smoothing_sweep_v1",
        "critic": asdict(config),
        "noise_sigmas": list(noise_sigmas),
        "seeds": list(seeds),
        "all_score_cosines_below_0p1": bool(
            (selected["score_cosine"].abs() < 0.1).all()
        ),
        "all_oracle_score_derivatives_negative": bool(
            (selected["oracle_reverse_kl_derivative"] < 0).all()
        ),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    plot(selected, output_root / "smoothing_sweep.png")
    print(json.dumps(summary, indent=2), flush=True)


def plot(frame: pd.DataFrame, output: Path) -> None:
    aggregate = frame.groupby("noise_sigma", as_index=False).agg(
        score_cosine_mean=("score_cosine", "mean"),
        score_cosine_std=("score_cosine", "std"),
        advfd_mean=("learned_advfd", "mean"),
        kl_change_mean=("kl_change", "mean"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    axes[0].errorbar(
        aggregate["noise_sigma"],
        aggregate["score_cosine_mean"],
        yerr=aggregate["score_cosine_std"].fillna(0.0),
        marker="o",
        capsize=3,
    )
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("cos(AdvFD field, score correction)")
    axes[1].plot(
        aggregate["noise_sigma"], aggregate["advfd_mean"], marker="o"
    )
    axes[1].set_ylabel("learned AdvFD")
    axes[2].plot(
        aggregate["noise_sigma"], aggregate["kl_change_mean"], marker="o"
    )
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_ylabel("finite KL change")
    for axis in axes:
        axis.set_xlabel("extra Gaussian noise sigma")
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--noise-sigmas", type=parse_floats, default=(0.0, 0.2, 0.4, 0.7, 1.0)
    )
    parser.add_argument("--seeds", type=parse_ints, default=(8101, 8102, 8103))
    parser.add_argument("--critic-steps", type=int, default=2000)
    parser.add_argument("--device", default="cuda:2")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.output_root,
        noise_sigmas=args.noise_sigmas,
        seeds=args.seeds,
        critic_steps=args.critic_steps,
        device=torch.device(args.device),
    )
