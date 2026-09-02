#!/usr/bin/env python3
"""Audit anchored implicit AutoGuidance on analytic one-dimensional densities.

This script does not claim that a one-shot density calibration is a generative
sampler.  It isolates the mathematical subproblem proposed for anchored AG:

    y = x + eta * grad[log p_strong(y) - log p_weak(y)].

One Picard iteration is explicit AutoGuidance.  Further iterations approach
the implicit/proximal solution when the map is contractive.  Compatible blur,
incompatible sharp weak models, and multimodal densities are reported
separately so a numerical success cannot hide the assumptions it relies on.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp
from scipy.stats import wasserstein_distance


@dataclass(frozen=True)
class GaussianMixture1D:
    weights: tuple[float, ...]
    means: tuple[float, ...]
    stds: tuple[float, ...]

    def __post_init__(self) -> None:
        count = len(self.weights)
        if count == 0 or len(self.means) != count or len(self.stds) != count:
            raise ValueError("mixture parameters must have the same nonzero length")
        if any(weight <= 0 for weight in self.weights):
            raise ValueError("mixture weights must be positive")
        if any(std <= 0 for std in self.stds):
            raise ValueError("mixture standard deviations must be positive")

    @property
    def normalized_weights(self) -> np.ndarray:
        weights = np.asarray(self.weights, dtype=np.float64)
        return weights / weights.sum()

    def sample(self, rng: np.random.Generator, count: int) -> np.ndarray:
        components = rng.choice(
            len(self.weights), size=count, p=self.normalized_weights
        )
        means = np.asarray(self.means, dtype=np.float64)[components]
        stds = np.asarray(self.stds, dtype=np.float64)[components]
        return means + stds * rng.standard_normal(count)

    def component_logpdf(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)[..., None]
        means = np.asarray(self.means, dtype=np.float64)
        stds = np.asarray(self.stds, dtype=np.float64)
        return (
            np.log(self.normalized_weights)
            - np.log(stds)
            - 0.5 * np.log(2.0 * np.pi)
            - 0.5 * ((values - means) / stds) ** 2
        )

    def logpdf(self, values: np.ndarray) -> np.ndarray:
        return logsumexp(self.component_logpdf(values), axis=-1)

    def score(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        component_logpdf = self.component_logpdf(values)
        responsibilities = np.exp(
            component_logpdf - logsumexp(component_logpdf, axis=-1, keepdims=True)
        )
        means = np.asarray(self.means, dtype=np.float64)
        variances = np.square(np.asarray(self.stds, dtype=np.float64))
        component_scores = (means - values[..., None]) / variances
        return np.sum(responsibilities * component_scores, axis=-1)


@dataclass(frozen=True)
class DensityCase:
    name: str
    data: GaussianMixture1D
    strong: GaussianMixture1D
    weak: GaussianMixture1D
    interpretation: str


def score_ratio(
    values: np.ndarray, *, strong: GaussianMixture1D, weak: GaussianMixture1D
) -> np.ndarray:
    return strong.score(values) - weak.score(values)


def picard_calibration(
    anchors: np.ndarray,
    *,
    strong: GaussianMixture1D,
    weak: GaussianMixture1D,
    eta: float,
    iterations: int,
) -> tuple[np.ndarray, list[float]]:
    current = anchors.copy()
    move_ratios: list[float] = []
    previous_move = None
    for _ in range(iterations):
        updated = anchors + eta * score_ratio(current, strong=strong, weak=weak)
        move = updated - current
        if previous_move is not None:
            denominator = np.sqrt(np.mean(previous_move**2))
            move_ratios.append(
                float(np.sqrt(np.mean(move**2)) / max(denominator, 1e-15))
            )
        current = updated
        previous_move = move
    return current, move_ratios


def mmd_rbf(left: np.ndarray, right: np.ndarray, *, bandwidth: float) -> float:
    # A deterministic linear-time estimate is sufficient for this mechanism audit.
    count = min(len(left), len(right))
    count -= count % 2
    left = left[:count].reshape(-1, 2)
    right = right[:count].reshape(-1, 2)

    def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.exp(-0.5 * np.square(a - b) / (bandwidth**2))

    return float(
        np.mean(kernel(left[:, 0], left[:, 1]))
        + np.mean(kernel(right[:, 0], right[:, 1]))
        - np.mean(kernel(left[:, 0], right[:, 1]))
        - np.mean(kernel(left[:, 1], right[:, 0]))
    )


def build_cases() -> tuple[DensityCase, ...]:
    weights = (0.25, 0.5, 0.25)
    means = (-2.5, 0.0, 2.5)
    return (
        DensityCase(
            name="compatible_blur",
            data=GaussianMixture1D(weights, means, (0.18, 0.18, 0.18)),
            strong=GaussianMixture1D(weights, means, (0.32, 0.32, 0.32)),
            weak=GaussianMixture1D(weights, means, (0.60, 0.60, 0.60)),
            interpretation="weak is a more blurred version of the same modes",
        ),
        DensityCase(
            name="incompatible_sharp_weak",
            data=GaussianMixture1D(weights, means, (0.18, 0.18, 0.18)),
            strong=GaussianMixture1D(weights, means, (0.32, 0.32, 0.32)),
            weak=GaussianMixture1D(weights, means, (0.16, 0.16, 0.16)),
            interpretation="weak is sharper, so strong-minus-weak points outward",
        ),
        DensityCase(
            name="mode_weight_mismatch",
            data=GaussianMixture1D((0.25, 0.5, 0.25), means, (0.22, 0.22, 0.22)),
            strong=GaussianMixture1D((0.20, 0.60, 0.20), means, (0.34, 0.34, 0.34)),
            weak=GaussianMixture1D((0.08, 0.84, 0.08), means, (0.62, 0.62, 0.62)),
            interpretation="blur and mode-weight errors coexist; ratio is non-concave",
        ),
    )


def main(args: argparse.Namespace) -> None:
    if args.samples <= 0 or args.iterations <= 0:
        raise ValueError("samples and iterations must be positive")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []

    for case in build_cases():
        anchors = case.strong.sample(rng, args.samples)
        reference = case.data.sample(rng, args.samples)
        bandwidth = max(float(np.std(reference)), 1e-3) * 0.2
        for eta in args.etas:
            for iterations in range(0, args.iterations + 1):
                if iterations == 0:
                    calibrated = anchors
                    move_ratios: list[float] = []
                else:
                    calibrated, move_ratios = picard_calibration(
                        anchors,
                        strong=case.strong,
                        weak=case.weak,
                        eta=eta,
                        iterations=iterations,
                    )
                finite = bool(np.isfinite(calibrated).all())
                rows.append(
                    {
                        "case": case.name,
                        "interpretation": case.interpretation,
                        "eta": eta,
                        "iterations": iterations,
                        "finite": finite,
                        "wasserstein_to_data": wasserstein_distance(
                            calibrated, reference
                        )
                        if finite
                        else float("inf"),
                        "data_nll": float(-np.mean(case.data.logpdf(calibrated)))
                        if finite
                        else float("inf"),
                        "linear_mmd_rbf": mmd_rbf(
                            calibrated, reference, bandwidth=bandwidth
                        )
                        if finite
                        else float("inf"),
                        "calibrated_std": float(np.std(calibrated))
                        if finite
                        else float("inf"),
                        "last_move_ratio": move_ratios[-1]
                        if move_ratios
                        else None,
                    }
                )

    csv_path = output_dir / "anchored_ag_density_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, len(build_cases()), figsize=(17, 4.8), sharey=False)
    for axis, case in zip(axes, build_cases(), strict=True):
        case_rows = [row for row in rows if row["case"] == case.name]
        for iterations in range(0, args.iterations + 1):
            selected = [row for row in case_rows if row["iterations"] == iterations]
            axis.plot(
                [float(row["eta"]) for row in selected],
                [float(row["wasserstein_to_data"]) for row in selected],
                marker="o",
                label=f"K={iterations}",
            )
        axis.set_title(case.name.replace("_", " "))
        axis.set_xlabel("eta")
        axis.set_ylabel("Wasserstein to data")
        axis.grid(alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    figure_path = output_dir / "anchored_ag_density_audit.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    payload = {
        "format": "eqvae_anchored_ag_density_audit_v1",
        "scope": "analytic mechanism audit, not a generative benchmark",
        "seed": args.seed,
        "samples": args.samples,
        "etas": args.etas,
        "max_iterations": args.iterations,
        "csv": str(csv_path),
        "figure": str(figure_path),
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/home/zhoushunyu/data/eqvae/experiments/anchored_ag_density_audit_v1"
        ),
    )
    parser.add_argument("--samples", type=int, default=50_000)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument(
        "--etas",
        type=lambda value: [float(item) for item in value.split(",")],
        default=[0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4],
    )
    parser.add_argument("--seed", type=int, default=20260830)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
