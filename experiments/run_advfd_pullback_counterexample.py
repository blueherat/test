#!/usr/bin/env python3
"""Audit transpose versus pseudoinverse pullback for an AdvFD witness.

For p=N(0,1), q=N(a,1), and a real-whitened exponential feature, the full
one-dimensional Gaussian Fréchet correction in feature space is valid.  A
standard generator gradient pulls it back with J_h^T and creates a tail field.
The minimum-norm pullback J_h^dagger instead recovers a spatially constant
field, exactly aligned with the reverse-KL score correction.
"""

from __future__ import annotations

import argparse
import json
import math
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

from experiments.frechet_residual_score_toy import (
    field_diagnostics,
    finite_pushforward_kl,
    score_field,
)
from experiments.run_advfd_covariance_tail_counterexample import (
    analytic_feature_moments,
    field_concentration,
    gaussian_pair,
    parse_floats,
    standardized_exponential,
)


def exponential_feature_force(
    states: torch.Tensor,
    concentration: float,
    shift: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return full Fréchet feature velocity and the scalar feature Jacobian."""

    target_mean, target_std, source_mean, source_std = analytic_feature_moments(
        concentration, shift, calibration="real"
    )
    feature, derivative = standardized_exponential(
        states,
        concentration,
        shift=shift,
        calibration="real",
    )
    mean_force = -2.0 * (source_mean - target_mean)
    covariance_force = (
        -2.0
        * (1.0 - target_std / source_std)
        * (feature - source_mean)
    )
    return mean_force + covariance_force, derivative


def exponential_pullback_field(
    concentration: float,
    shift: float,
    mode: str,
):
    if mode not in {"transpose", "pseudoinverse"}:
        raise ValueError(f"unknown pullback mode: {mode}")

    def field(states: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
        del create_graph
        feature_force, derivative = exponential_feature_force(
            states, concentration, shift
        )
        if mode == "transpose":
            velocity = derivative * feature_force
        else:
            velocity = feature_force / derivative
        return velocity[:, None]

    return field


def feature_tracking_diagnostics(
    source,
    *,
    concentration: float,
    shift: float,
    mode: str,
    order: int,
) -> dict[str, float]:
    states, weights = source.quadrature(order)
    normalized = weights / weights.sum()
    feature_force, derivative = exponential_feature_force(
        states, concentration, shift
    )
    input_velocity = exponential_pullback_field(
        concentration, shift, mode
    )(states, False)[:, 0]
    induced = derivative * input_velocity
    residual = induced - feature_force
    force_rms = (normalized * feature_force.square()).sum().sqrt()
    residual_rms = (normalized * residual.square()).sum().sqrt()
    return {
        "feature_force_rms": float(force_rms),
        "induced_feature_velocity_rms": float(
            (normalized * induced.square()).sum().sqrt()
        ),
        "feature_tracking_relative_error": float(
            residual_rms / force_rms.clamp_min(torch.finfo(states.dtype).eps)
        ),
    }


def analytic_pseudoinverse_velocity(concentration: float, shift: float) -> float:
    return -2.0 * (1.0 - math.exp(-concentration * shift)) / concentration


def run(
    output_root: Path,
    *,
    shift: float,
    concentrations: tuple[float, ...],
    displacement_rms: tuple[float, ...],
    quadrature_order: int,
    device: torch.device,
) -> None:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    output_root.mkdir(parents=True)
    target, source = gaussian_pair(shift, device=device)
    rows = []
    for concentration in concentrations:
        fields = {
            "transpose": exponential_pullback_field(
                concentration, shift, "transpose"
            ),
            "pseudoinverse": exponential_pullback_field(
                concentration, shift, "pseudoinverse"
            ),
            "score": score_field(target, source),
        }
        for mode, field in fields.items():
            diagnostics = field_diagnostics(
                target, source, field, quadrature_order=quadrature_order
            )
            concentration_metrics = field_concentration(
                source, field, order=quadrature_order
            )
            tracking = (
                feature_tracking_diagnostics(
                    source,
                    concentration=concentration,
                    shift=shift,
                    mode=mode,
                    order=quadrature_order,
                )
                if mode != "score"
                else {
                    "feature_force_rms": float("nan"),
                    "induced_feature_velocity_rms": float("nan"),
                    "feature_tracking_relative_error": float("nan"),
                }
            )
            for displacement in displacement_rms:
                step_size = displacement / diagnostics["velocity_rms"]
                finite = finite_pushforward_kl(
                    target,
                    source,
                    field,
                    step_size=step_size,
                    quadrature_order=quadrature_order,
                )
                rows.append(
                    {
                        "shift": shift,
                        "concentration": concentration,
                        "pullback": mode,
                        "target_displacement_rms": displacement,
                        "step_size": step_size,
                        "analytic_pseudoinverse_velocity": (
                            analytic_pseudoinverse_velocity(concentration, shift)
                        ),
                        **diagnostics,
                        **concentration_metrics,
                        **tracking,
                        **finite,
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "pullback_counterexample.csv", index=False)
    summary = {
        "protocol": "advfd_pullback_counterexample_v1",
        "target": "N(0,1)",
        "source": f"N({shift:g},1)",
        "feature": "real-whitened exp(c*x)",
        "concentrations": list(concentrations),
        "target_displacement_rms": list(displacement_rms),
        "quadrature_order": quadrature_order,
        "pseudoinverse_identity": "u=-2(1-exp(-c*a))/c",
        "information_used": "critic feature, critic Jacobian, and p/q feature moments only",
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    plot(frame, output_root / "pullback_counterexample.png")


def plot(frame: pd.DataFrame, output: Path) -> None:
    selected = frame[
        frame["target_displacement_rms"]
        == frame["target_displacement_rms"].min()
    ]
    figure, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    for mode, condition in selected.groupby("pullback"):
        axes[0].plot(
            condition["concentration"],
            condition["score_cosine"],
            marker="o",
            label=mode,
        )
        axes[1].plot(
            condition["concentration"],
            condition["field_effective_mass"],
            marker="o",
            label=mode,
        )
        axes[2].plot(
            condition["concentration"],
            condition["kl_change"],
            marker="o",
            label=mode,
        )
    tracking = selected[selected["pullback"] != "score"]
    for mode, condition in tracking.groupby("pullback"):
        axes[3].plot(
            condition["concentration"],
            condition["feature_tracking_relative_error"],
            marker="o",
            label=mode,
        )
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_title("Cosine to reverse-KL score")
    axes[1].set_yscale("log")
    axes[1].set_title("Effective q-mass")
    axes[2].set_title("KL change at matched RMS")
    axes[3].set_yscale("log")
    axes[3].set_title("Feature-transport error")
    for axis in axes:
        axis.set_xlabel("critic concentration c")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shift", type=float, default=0.75)
    parser.add_argument(
        "--concentrations",
        type=parse_floats,
        default=(0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0),
    )
    parser.add_argument(
        "--displacement-rms", type=parse_floats, default=(1e-4, 1e-3, 1e-2)
    )
    parser.add_argument("--quadrature-order", type=int, default=48)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run(
        args.output_root,
        shift=args.shift,
        concentrations=args.concentrations,
        displacement_rms=args.displacement_rms,
        quadrature_order=args.quadrature_order,
        device=torch.device(args.device),
    )


if __name__ == "__main__":
    main()
