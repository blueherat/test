#!/usr/bin/env python3
"""Run the AdvFD witness-gradient and noised-score counterexample audit."""

from __future__ import annotations

import argparse
import json
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

from experiments.advfd_score_counterexample import (
    build_advfd_witness,
    frechet_distance_1d,
    moment_matched_disjoint_pair,
    noised_reverse_kl_and_score_metrics,
    official_loaded_real_whitened_fd_1d,
    paper_regularized_real_whitened_fd_1d,
    shared_support_pearson_control,
    witness_generator_gradient,
    witness_statistics,
)


def parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(",") if item.strip())


def tensor_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


def run_amplitude_scan(
    amplitudes: tuple[float, ...], epsilons: tuple[float, ...]
) -> pd.DataFrame:
    rows = []
    for amplitude in amplitudes:
        coefficients = build_advfd_witness(amplitude)
        stats = witness_statistics(coefficients)
        for epsilon in epsilons:
            paper_fd = paper_regularized_real_whitened_fd_1d(
                stats["real_mean"],
                stats["real_variance"],
                stats["fake_mean"],
                stats["fake_variance"],
                epsilon=epsilon,
            )
            official_fd, gradient = witness_generator_gradient(
                coefficients, epsilon=epsilon
            )
            _, normalized_gradient = witness_generator_gradient(
                coefficients,
                epsilon=epsilon,
                normalization_epsilon=0.01,
            )
            rows.append(
                {
                    "amplitude": amplitude,
                    "epsilon": epsilon,
                    "paper_common_transform_fd": tensor_float(paper_fd),
                    "official_loaded_fd": tensor_float(official_fd),
                    "generator_gradient_l2": tensor_float(gradient.norm()),
                    "normalized_generator_gradient_l2": tensor_float(
                        normalized_gradient.norm()
                    ),
                    "max_fake_feature_derivative_abs": tensor_float(
                        stats["fake_derivatives"].abs().max()
                    ),
                }
            )
    return pd.DataFrame(rows)


def run_jet_scan(amplitude: float, epsilon: float) -> pd.DataFrame:
    patterns = {
        "flat": (0.0, 0.0),
        "plus": (1.0, 1.0),
        "minus": (-1.0, -1.0),
        "split": (1.0, -1.0),
        "split_reverse": (-1.0, 1.0),
        "plus_x10": (10.0, 10.0),
    }
    rows = []
    for name, slopes in patterns.items():
        coefficients = build_advfd_witness(
            amplitude, fake_derivatives=slopes
        )
        stats = witness_statistics(coefficients)
        distance, gradient = witness_generator_gradient(
            coefficients, epsilon=epsilon
        )
        _, normalized_gradient = witness_generator_gradient(
            coefficients,
            epsilon=epsilon,
            normalization_epsilon=0.01,
        )
        rows.append(
            {
                "pattern": name,
                "requested_slope_0": slopes[0],
                "requested_slope_1": slopes[1],
                "real_feature_mean": tensor_float(stats["real_mean"]),
                "real_feature_variance": tensor_float(stats["real_variance"]),
                "fake_feature_mean": tensor_float(stats["fake_mean"]),
                "fake_feature_variance": tensor_float(stats["fake_variance"]),
                "official_loaded_fd": tensor_float(distance),
                "gradient_atom_0": tensor_float(gradient[0]),
                "gradient_atom_1": tensor_float(gradient[1]),
                "gradient_l2": tensor_float(gradient.norm()),
                "normalized_gradient_atom_0": tensor_float(
                    normalized_gradient[0]
                ),
                "normalized_gradient_atom_1": tensor_float(
                    normalized_gradient[1]
                ),
            }
        )
    return pd.DataFrame(rows)


def run_score_scan(
    sigmas: tuple[float, ...], grid_points: int, step_factor: float
) -> pd.DataFrame:
    real, fake = moment_matched_disjoint_pair()
    rows = []
    for sigma in sigmas:
        metrics = noised_reverse_kl_and_score_metrics(
            real,
            fake,
            sigma=sigma,
            grid_points=grid_points,
            step_factor=step_factor,
        )
        rows.append(
            {
                "sigma": sigma,
                "reverse_kl": tensor_float(metrics["reverse_kl"]),
                "fisher_divergence": tensor_float(metrics["fisher_divergence"]),
                "continuity_kl_derivative": tensor_float(
                    metrics["continuity_kl_derivative"]
                ),
                "component_direction_0": tensor_float(
                    metrics["component_directions"][0]
                ),
                "component_direction_1": tensor_float(
                    metrics["component_directions"][1]
                ),
                "parameterized_kl_derivative": tensor_float(
                    metrics["parameterized_kl_derivative"]
                ),
                "step_size": tensor_float(metrics["step_size"]),
                "updated_atom_0": tensor_float(metrics["updated_fake_atoms"][0]),
                "updated_atom_1": tensor_float(metrics["updated_fake_atoms"][1]),
                "updated_reverse_kl": tensor_float(
                    metrics["updated_reverse_kl"]
                ),
                "reverse_kl_change": tensor_float(metrics["reverse_kl_change"]),
            }
        )
    return pd.DataFrame(rows)


def make_figure(
    amplitude_frame: pd.DataFrame,
    jet_frame: pd.DataFrame,
    score_frame: pd.DataFrame,
    output: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))

    for epsilon, group in amplitude_frame.groupby("epsilon"):
        axes[0].plot(
            group["amplitude"],
            group["official_loaded_fd"],
            marker="o",
            label=f"epsilon={epsilon:g}",
        )
    axes[0].set_yscale("symlog", linthresh=1.0)
    axes[0].set_xlabel("off-support feature amplitude M")
    axes[0].set_ylabel("real-whitened FD")
    axes[0].set_title("Discrepancy grows; flat gradient stays zero")
    axes[0].legend()

    x = range(len(jet_frame))
    axes[1].bar(
        [index - 0.18 for index in x],
        jet_frame["gradient_atom_0"],
        width=0.36,
        label="fake atom 0",
    )
    axes[1].bar(
        [index + 0.18 for index in x],
        jet_frame["gradient_atom_1"],
        width=0.36,
        label="fake atom 1",
    )
    axes[1].set_xticks(list(x), jet_frame["pattern"], rotation=35, ha="right")
    axes[1].set_ylabel("generator gradient")
    axes[1].set_title("Same FD, arbitrarily different input gradients")
    axes[1].legend()

    relative_reduction = (
        -100.0 * score_frame["reverse_kl_change"] / score_frame["reverse_kl"]
    )
    axes[2].bar(score_frame["sigma"], relative_reduction, width=0.065)
    axes[2].set_xlabel("Gaussian noise sigma")
    axes[2].set_ylabel("one-step reverse-KL reduction (%)")
    axes[2].set_title("Every finite noised-score step lowers KL")

    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--amplitudes", default="0,1,2,4,8,16,32")
    parser.add_argument("--epsilons", default="0.001,0.1")
    parser.add_argument("--jet-amplitude", type=float, default=4.0)
    parser.add_argument("--jet-epsilon", type=float, default=0.001)
    parser.add_argument("--sigmas", default="0.1,0.2,0.4,0.8")
    parser.add_argument("--grid-points", type=int, default=40001)
    parser.add_argument("--score-step-factor", type=float, default=0.02)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    real, fake = moment_matched_disjoint_pair()
    identity_fd = frechet_distance_1d(
        real.mean, real.variance, fake.mean, fake.variance
    )
    pearson = shared_support_pearson_control()

    amplitude_frame = run_amplitude_scan(
        parse_floats(args.amplitudes), parse_floats(args.epsilons)
    )
    jet_frame = run_jet_scan(args.jet_amplitude, args.jet_epsilon)
    score_frame = run_score_scan(
        parse_floats(args.sigmas), args.grid_points, args.score_step_factor
    )
    amplitude_frame.to_csv(args.output_dir / "amplitude_scan.csv", index=False)
    jet_frame.to_csv(args.output_dir / "witness_jet_scan.csv", index=False)
    score_frame.to_csv(args.output_dir / "noised_score_scan.csv", index=False)
    make_figure(
        amplitude_frame,
        jet_frame,
        score_frame,
        args.output_dir / "counterexample_summary.png",
    )

    flat = jet_frame.loc[jet_frame["pattern"] == "flat"].iloc[0]
    summary = {
        "distributions": {
            "real_atoms": real.atoms.tolist(),
            "real_weights": real.weights.tolist(),
            "fake_atoms": fake.atoms.tolist(),
            "fake_weights": fake.weights.tolist(),
            "real_mean": tensor_float(real.mean),
            "fake_mean": tensor_float(fake.mean),
            "real_variance": tensor_float(real.variance),
            "fake_variance": tensor_float(fake.variance),
            "identity_fd": tensor_float(identity_fd),
        },
        "shared_support_control": {
            key: tensor_float(value) for key, value in pearson.items()
        },
        "flat_witness": {
            "amplitude": args.jet_amplitude,
            "official_loaded_fd": float(flat["official_loaded_fd"]),
            "generator_gradient_l2": float(flat["gradient_l2"]),
        },
        "checks": {
            "all_score_steps_lower_kl": bool(
                (score_frame["reverse_kl_change"] < 0).all()
            ),
            "all_continuity_derivatives_negative": bool(
                (score_frame["continuity_kl_derivative"] < 0).all()
            ),
            "jet_scan_fd_span": float(
                jet_frame["official_loaded_fd"].max()
                - jet_frame["official_loaded_fd"].min()
            ),
        },
        "config": {
            "amplitudes": list(parse_floats(args.amplitudes)),
            "epsilons": list(parse_floats(args.epsilons)),
            "jet_amplitude": args.jet_amplitude,
            "jet_epsilon": args.jet_epsilon,
            "sigmas": list(parse_floats(args.sigmas)),
            "grid_points": args.grid_points,
            "score_step_factor": args.score_step_factor,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
