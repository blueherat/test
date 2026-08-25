#!/usr/bin/env python3
"""Compare the transport implied by AdvFD-related discrepancy choices.

The target is p=N(0,1) and the generator is q=N(a,1).  This family has an
exact reverse-KL score correction for every separation a, while real-only
Fisher whitening induces Pearson chi-square weighting and pooled Fisher
whitening induces twice triangular discrimination.  The sweep separates
three failure modes: tail explosion, overlap saturation, and direct score
transport.
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
    pearson_field,
    pooled_fisher_divergence,
    pooled_fisher_field,
    score_field,
)


def parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(",") if item.strip())


def gaussian_shift_pair(shift: float, *, device: torch.device):
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


def field_concentration(source, field, *, order: int) -> dict[str, float]:
    states, weights = source.quadrature(order)
    normalized = weights / weights.sum()
    velocity = field(states, False)[:, 0]
    second = (normalized * velocity.square()).sum()
    fourth = (normalized * velocity.pow(4)).sum()
    effective_mass = second.square() / fourth.clamp_min(
        torch.finfo(states.dtype).eps
    )
    source_mean = source.moments().mean[None, :]
    typical_velocity = field(source_mean, False)[0, 0]
    return {
        "field_effective_mass": float(effective_mass),
        "field_mean": float((normalized * velocity).sum()),
        "field_at_source_mean": float(typical_velocity),
        "field_max_abs_on_quadrature": float(velocity.abs().max()),
    }


def run(
    output_root: Path,
    *,
    shifts: tuple[float, ...],
    fixed_step_size: float,
    matched_displacement_rms: float,
    quadrature_order: int,
    device: torch.device,
) -> None:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    output_root.mkdir(parents=True)
    rows = []
    for shift in shifts:
        target, source = gaussian_shift_pair(shift, device=device)
        divergences = {
            "reverse_kl": 0.5 * shift * shift,
            "pearson": math.expm1(shift * shift),
            "pooled_fisher": pooled_fisher_divergence(
                target, source, quadrature_order=quadrature_order
            ),
        }
        fields = {
            "score_reverse_kl": score_field(target, source),
            "real_fisher_pearson": pearson_field(target, source),
            "pooled_fisher_triangular": pooled_fisher_field(target, source),
        }
        for name, field in fields.items():
            diagnostics = field_diagnostics(
                target, source, field, quadrature_order=quadrature_order
            )
            concentration = field_concentration(
                source, field, order=quadrature_order
            )
            for step_mode, step_size in (
                ("fixed_step", fixed_step_size),
                (
                    "matched_rms",
                    matched_displacement_rms / diagnostics["velocity_rms"],
                ),
            ):
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
                        "field": name,
                        "step_mode": step_mode,
                        "step_size": step_size,
                        "actual_displacement_rms": (
                            step_size * diagnostics["velocity_rms"]
                        ),
                        **divergences,
                        **diagnostics,
                        **concentration,
                        **finite,
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "divergence_transport_audit.csv", index=False)
    summary = {
        "protocol": "advfd_divergence_transport_audit_v1",
        "target": "N(0,1)",
        "source_family": "N(a,1)",
        "shifts": list(shifts),
        "fixed_step_size": fixed_step_size,
        "matched_displacement_rms": matched_displacement_rms,
        "quadrature_order": quadrature_order,
        "field_definitions": {
            "score_reverse_kl": "score_p-score_q",
            "real_fisher_pearson": "(q/p)(score_p-score_q), up to scale",
            "pooled_fisher_triangular": (
                "16(q/p)/(1+q/p)^3(score_p-score_q)"
            ),
        },
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    plot(frame, output_root / "divergence_transport_audit.png")


def plot(frame: pd.DataFrame, output: Path) -> None:
    selected = frame[frame["step_mode"] == "fixed_step"]
    figure, axes = plt.subplots(1, 5, figsize=(22, 4.3))
    divergence_frame = selected.drop_duplicates("shift")
    for name in ("reverse_kl", "pearson", "pooled_fisher"):
        axes[0].plot(
            divergence_frame["shift"],
            divergence_frame[name],
            marker="o",
            label=name,
        )
    axes[0].set_yscale("log")
    axes[0].set_title("Population discrepancy")
    for field, condition in selected.groupby("field"):
        axes[1].plot(
            condition["shift"], condition["velocity_rms"], marker="o", label=field
        )
        axes[2].plot(
            condition["shift"], condition["score_cosine"], marker="o", label=field
        )
        axes[3].plot(
            condition["shift"],
            condition["field_effective_mass"],
            marker="o",
            label=field,
        )
        axes[4].plot(
            condition["shift"],
            condition["actual_displacement_rms"],
            marker="o",
            label=field,
        )
    axes[1].set_yscale("log")
    axes[1].set_title("Raw correction RMS")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_title("Cosine to reverse-KL score")
    axes[3].set_yscale("log")
    axes[3].set_title("Effective q-mass")
    axes[4].set_yscale("log")
    axes[4].set_title("Move under one fixed step")
    for axis in axes:
        axis.set_xlabel("distribution separation a")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[4].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--shifts",
        type=parse_floats,
        default=(0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0),
    )
    parser.add_argument("--fixed-step-size", type=float, default=1e-4)
    parser.add_argument("--matched-displacement-rms", type=float, default=1e-4)
    parser.add_argument("--quadrature-order", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run(
        args.output_root,
        shifts=args.shifts,
        fixed_step_size=args.fixed_step_size,
        matched_displacement_rms=args.matched_displacement_rms,
        quadrature_order=args.quadrature_order,
        device=torch.device(args.device),
    )


if __name__ == "__main__":
    main()
