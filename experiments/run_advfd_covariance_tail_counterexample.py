#!/usr/bin/env python3
"""An analytic tail counterexample for covariance AdvFD calibration.

For p=N(0,1), q=N(a,1), and h_c(x)=exp(c x), real whitening gives an
exact fake feature standard deviation exp(c a).  The covariance Frechet term
therefore diverges as c grows, while its normalized input field concentrates
in progressively rarer q tails and becomes inefficient for reverse-KL descent.

The pooled control replaces real-only whitening with equally weighted p/q
moments.  Calibration statistics are treated as detached constants when the
sample field is differentiated, matching the generator-side use of a fitted
critic rather than differentiating through population calibration.
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
    GaussianMixture,
    field_diagnostics,
    finite_pushforward_kl,
    score_field,
    weighted_inner,
)


def parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(",") if item.strip())


def gaussian_pair(shift: float, *, device: torch.device):
    dtype = torch.float64
    weights = torch.ones(1, dtype=dtype, device=device)
    covariance = torch.ones(1, 1, dtype=dtype, device=device)
    target = GaussianMixture(
        weights,
        torch.zeros(1, 1, dtype=dtype, device=device),
        covariance,
    )
    source = GaussianMixture(
        weights,
        torch.full((1, 1), shift, dtype=dtype, device=device),
        covariance,
    )
    return target, source


def raw_feature_moments(
    concentration: float, shift: float
) -> tuple[float, float, float, float]:
    c = concentration
    target_mean = math.exp(0.5 * c * c)
    target_variance = math.exp(2.0 * c * c) - math.exp(c * c)
    source_mean = math.exp(c * shift) * target_mean
    source_variance = math.exp(2.0 * c * shift) * target_variance
    return target_mean, target_variance, source_mean, source_variance


def calibration_parameters(
    concentration: float, shift: float, calibration: str
) -> tuple[float, float]:
    target_mean, target_variance, source_mean, source_variance = (
        raw_feature_moments(concentration, shift)
    )
    if calibration == "real":
        return target_mean, math.sqrt(target_variance)
    if calibration == "pooled":
        center = 0.5 * (target_mean + source_mean)
        variance = (
            0.5 * (target_variance + source_variance)
            + 0.25 * (target_mean - source_mean) ** 2
        )
        return center, math.sqrt(variance)
    raise ValueError(f"unknown calibration: {calibration}")


def standardized_exponential(
    states: torch.Tensor,
    concentration: float,
    *,
    shift: float = 0.0,
    calibration: str = "real",
) -> tuple[torch.Tensor, torch.Tensor]:
    c = torch.as_tensor(concentration, dtype=states.dtype, device=states.device)
    center_value, scale_value = calibration_parameters(
        concentration, shift, calibration
    )
    center = torch.as_tensor(center_value, dtype=states.dtype, device=states.device)
    scale = torch.as_tensor(scale_value, dtype=states.dtype, device=states.device)
    raw = torch.exp(c * states[:, 0])
    feature = (raw - center) / scale
    derivative = c * raw / scale
    return feature, derivative


def analytic_feature_moments(
    concentration: float,
    shift: float,
    *,
    calibration: str = "real",
) -> tuple[float, float, float, float]:
    target_mean, target_variance, source_mean, source_variance = (
        raw_feature_moments(concentration, shift)
    )
    center, scale = calibration_parameters(concentration, shift, calibration)
    return (
        (target_mean - center) / scale,
        math.sqrt(target_variance) / scale,
        (source_mean - center) / scale,
        math.sqrt(source_variance) / scale,
    )


def advfd_values(
    concentration: float, shift: float, *, calibration: str = "real"
) -> dict[str, float]:
    target_mean, target_std, source_mean, source_std = analytic_feature_moments(
        concentration, shift, calibration=calibration
    )
    mean = (source_mean - target_mean) ** 2
    covariance = (source_std - target_std) ** 2
    return {
        "advfd_mean": mean,
        "advfd_covariance": covariance,
        "advfd_full": mean + covariance,
        "source_feature_mean": source_mean,
        "source_feature_std": source_std,
    }


def exponential_advfd_field(
    concentration: float,
    shift: float,
    component: str,
    *,
    calibration: str = "real",
):
    if component not in {"full", "mean", "covariance"}:
        raise ValueError(f"unknown field component: {component}")
    target_mean, target_std, source_mean, source_std = analytic_feature_moments(
        concentration, shift, calibration=calibration
    )

    def field(states: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
        del create_graph
        feature, derivative = standardized_exponential(
            states,
            concentration,
            shift=shift,
            calibration=calibration,
        )
        mean_gradient = -2.0 * (source_mean - target_mean) * derivative
        covariance_gradient = (
            -2.0
            * (1.0 - target_std / source_std)
            * (feature - source_mean)
            * derivative
        )
        if component == "mean":
            velocity = mean_gradient
        elif component == "covariance":
            velocity = covariance_gradient
        else:
            velocity = mean_gradient + covariance_gradient
        return velocity[:, None]

    return field


def field_concentration(source, field, *, order: int) -> dict[str, float]:
    states, weights = source.quadrature(order)
    normalized = weights / weights.sum()
    velocity = field(states, False)[:, 0]
    second = (normalized * velocity.square()).sum()
    fourth = (normalized * velocity.pow(4)).sum()
    effective_mass = second.square() / fourth.clamp_min(
        torch.finfo(states.dtype).eps
    )
    mean = (normalized * velocity).sum()
    return {
        "field_effective_mass": float(effective_mass),
        "field_mean": float(mean),
        "field_rms_1d": float(second.sqrt()),
        "field_max_abs_on_quadrature": float(velocity.abs().max()),
    }


def run(
    output_root: Path,
    *,
    shift: float,
    concentrations: tuple[float, ...],
    calibrations: tuple[str, ...],
    displacements: tuple[float, ...],
    quadrature_order: int,
    device: torch.device,
) -> None:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    output_root.mkdir(parents=True)
    target, source = gaussian_pair(shift, device=device)
    rows = []
    for concentration in concentrations:
        for calibration in calibrations:
            values = advfd_values(
                concentration, shift, calibration=calibration
            )
            fields = {
                "mean": exponential_advfd_field(
                    concentration, shift, "mean", calibration=calibration
                ),
                "covariance": exponential_advfd_field(
                    concentration, shift, "covariance", calibration=calibration
                ),
                "full": exponential_advfd_field(
                    concentration, shift, "full", calibration=calibration
                ),
                "score": score_field(target, source),
            }
            for name, field in fields.items():
                diagnostics = field_diagnostics(
                    target, source, field, quadrature_order=quadrature_order
                )
                concentration_metrics = field_concentration(
                    source, field, order=quadrature_order
                )
                for displacement in displacements:
                    velocity_rms = diagnostics["velocity_rms"]
                    step_size = displacement / velocity_rms
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
                            "calibration": calibration,
                            "field_component": name,
                            "target_displacement_rms": displacement,
                            "step_size": step_size,
                            **values,
                            **diagnostics,
                            **concentration_metrics,
                            **finite,
                        }
                    )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "covariance_tail_counterexample.csv", index=False)
    selected = frame[frame["target_displacement_rms"] == min(displacements)]
    summary = {
        "protocol": "advfd_covariance_tail_counterexample_v2",
        "target": "N(0,1)",
        "source": f"N({shift:g},1)",
        "critic_family": "detached calibrated exp(c*x)",
        "concentrations": list(concentrations),
        "calibrations": list(calibrations),
        "target_displacement_rms": list(displacements),
        "quadrature_order": quadrature_order,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    figure, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    full = selected[selected["field_component"] == "full"]
    score = selected[
        (selected["field_component"] == "score")
        & (selected["calibration"] == calibrations[0])
    ]
    for calibration in calibrations:
        condition = full[full["calibration"] == calibration]
        axes[0].plot(
            condition["concentration"],
            condition["advfd_full"],
            label=f"{calibration} full",
        )
    axes[0].set_yscale("log")
    axes[0].set_title("Whitened AdvFD value")
    axes[0].legend()
    for calibration in calibrations:
        condition = full[full["calibration"] == calibration]
        axes[1].plot(
            condition["concentration"],
            condition["score_cosine"],
            marker="o",
            label=calibration,
        )
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Full-field cosine to KL score")
    for calibration in calibrations:
        condition = full[full["calibration"] == calibration]
        axes[2].plot(
            condition["concentration"],
            condition["field_effective_mass"],
            marker="o",
            label=calibration,
        )
    axes[2].set_yscale("log")
    axes[2].set_title("Effective q-mass carrying field")
    for calibration in calibrations:
        condition = full[full["calibration"] == calibration]
        axes[3].plot(
            condition["concentration"],
            condition["kl_change"],
            marker="o",
            label=calibration,
        )
    axes[3].plot(
        score["concentration"], score["kl_change"], marker="o", label="score"
    )
    axes[3].axhline(0.0, color="black", linewidth=1)
    axes[3].set_title(f"KL change at RMS={min(displacements):g}")
    axes[3].legend()
    axes[1].legend()
    axes[2].legend()
    for axis in axes:
        axis.set_xlabel("critic concentration c")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_root / "covariance_tail_counterexample.png", dpi=180)
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
        "--calibrations",
        type=lambda value: tuple(item for item in value.split(",") if item),
        default=("real", "pooled"),
    )
    parser.add_argument(
        "--displacements", type=parse_floats, default=(1e-4, 1e-3, 1e-2)
    )
    parser.add_argument("--quadrature-order", type=int, default=48)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run(
        args.output_root,
        shift=args.shift,
        concentrations=args.concentrations,
        calibrations=args.calibrations,
        displacements=args.displacements,
        quadrature_order=args.quadrature_order,
        device=torch.device(args.device),
    )


if __name__ == "__main__":
    main()
