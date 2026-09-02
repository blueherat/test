"""Analytic audit of semigroup-consistent power guidance.

The experiment separates two operations that standard score extrapolation
silently conflates:

1. power-tilt the clean endpoint density and then apply the heat semigroup;
2. apply the heat semigroup to two endpoint densities and power-mix the
   resulting noisy marginals.

These operations do not commute.  In one dimension every density and score
can be evaluated on a fine grid, so the missing conditional-Jensen score can
be checked without training a model or tuning a guidance schedule.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class GaussianMixture1D:
    weights: tuple[float, ...]
    means: tuple[float, ...]
    stds: tuple[float, ...]

    def __post_init__(self) -> None:
        if not (len(self.weights) == len(self.means) == len(self.stds)):
            raise ValueError("weights, means, and stds must have equal length")
        if any(weight <= 0.0 for weight in self.weights):
            raise ValueError("mixture weights must be positive")
        if any(std <= 0.0 for std in self.stds):
            raise ValueError("mixture standard deviations must be positive")

    def density(self, values: Array, *, heat_variance: float = 0.0) -> Array:
        variance = np.square(np.asarray(self.stds, dtype=np.float64)) + float(
            heat_variance
        )
        means = np.asarray(self.means, dtype=np.float64)
        weights = np.asarray(self.weights, dtype=np.float64)
        weights = weights / weights.sum()
        centered = values[:, None] - means[None, :]
        components = np.exp(-0.5 * np.square(centered) / variance[None, :])
        components /= np.sqrt(2.0 * math.pi * variance[None, :])
        return components @ weights

    def score(self, values: Array, *, heat_variance: float) -> Array:
        variance = np.square(np.asarray(self.stds, dtype=np.float64)) + float(
            heat_variance
        )
        means = np.asarray(self.means, dtype=np.float64)
        weights = np.asarray(self.weights, dtype=np.float64)
        weights = weights / weights.sum()
        centered = values[:, None] - means[None, :]
        components = np.exp(-0.5 * np.square(centered) / variance[None, :])
        components *= weights[None, :] / np.sqrt(
            2.0 * math.pi * variance[None, :]
        )
        density = components.sum(axis=1)
        derivative = (-centered / variance[None, :] * components).sum(axis=1)
        return derivative / np.maximum(density, 1e-300)


def normalize_density(density: Array, grid: Array) -> Array:
    density = np.maximum(np.asarray(density, dtype=np.float64), 0.0)
    normalizer = np.trapezoid(density, grid)
    if not math.isfinite(normalizer) or normalizer <= 0.0:
        raise ValueError("density has an invalid normalizer")
    return density / normalizer


def heat_apply_periodic(density: Array, grid: Array, variance: float) -> Array:
    """Apply an unnormalized heat kernel using a wide periodic box.

    The box is chosen so boundary mass is negligible.  This spectral form is
    both faster and more accurate than repeatedly truncating a Gaussian
    convolution kernel.
    """

    spacing = float(grid[1] - grid[0])
    frequencies = 2.0 * math.pi * np.fft.rfftfreq(grid.size, d=spacing)
    spectrum = np.fft.rfft(density)
    return np.fft.irfft(
        spectrum * np.exp(-0.5 * float(variance) * np.square(frequencies)),
        n=grid.size,
    )


def heat_evolve_periodic(density: Array, grid: Array, variance: float) -> Array:
    return normalize_density(
        np.maximum(heat_apply_periodic(density, grid, variance), 0.0), grid
    )


def score_from_density(density: Array, grid: Array) -> Array:
    spacing = float(grid[1] - grid[0])
    frequencies = 2.0 * math.pi * np.fft.rfftfreq(grid.size, d=spacing)
    derivative = np.fft.irfft(
        1j * frequencies * np.fft.rfft(density), n=grid.size
    )
    floor = max(float(density.max()) * 1e-13, 1e-300)
    return derivative / np.maximum(density, floor)


def conditional_jensen_terms(
    *,
    grid: Array,
    weak_clean: Array,
    strong_clean: Array,
    beta: float,
    heat_variance: float,
) -> dict[str, Array]:
    """Return the exact conditional moments and their score correction."""

    weak_noisy = heat_evolve_periodic(weak_clean, grid, heat_variance)
    strong_noisy = heat_evolve_periodic(strong_clean, grid, heat_variance)
    ratio_clean = strong_clean / np.maximum(weak_clean, 1e-300)
    first_numerator = heat_apply_periodic(
        weak_clean * ratio_clean, grid, heat_variance
    )
    beta_raw = weak_clean * np.power(ratio_clean, beta)
    beta_numerator = heat_apply_periodic(beta_raw, grid, heat_variance)

    h1 = np.maximum(first_numerator, 0.0) / np.maximum(weak_noisy, 1e-300)
    hbeta = np.maximum(beta_numerator, 0.0) / np.maximum(weak_noisy, 1e-300)
    target_noisy = normalize_density(np.maximum(beta_numerator, 0.0), grid)

    weak_score = score_from_density(weak_noisy, grid)
    strong_score = score_from_density(strong_noisy, grid)
    target_score = score_from_density(target_noisy, grid)
    static_score = beta * strong_score + (1.0 - beta) * weak_score
    correction = target_score - static_score

    delta = np.log(np.maximum(hbeta, 1e-300)) - beta * np.log(
        np.maximum(h1, 1e-300)
    )
    # Differentiating delta directly avoids exponentiating a function whose
    # harmless additive normalization constant can be very large in the tails.
    delta_gradient = np.gradient(delta, grid, edge_order=2)

    return {
        "weak_noisy": weak_noisy,
        "strong_noisy": strong_noisy,
        "target_noisy": target_noisy,
        "h1": h1,
        "hbeta": hbeta,
        "delta": delta,
        "weak_score": weak_score,
        "strong_score": strong_score,
        "target_score": target_score,
        "static_score": static_score,
        "correction": correction,
        "delta_gradient": delta_gradient,
        "first_numerator": first_numerator,
        "beta_numerator": beta_numerator,
    }


def quantiles_from_density(density: Array, grid: Array, count: int) -> Array:
    spacing = float(grid[1] - grid[0])
    cdf = np.cumsum(density) * spacing
    cdf = np.clip(cdf / cdf[-1], 0.0, 1.0)
    probabilities = (np.arange(count, dtype=np.float64) + 0.5) / count
    return np.interp(probabilities, cdf, grid)


def sample_probability_flow(
    *,
    initial: Array,
    taus: Array,
    grid: Array,
    score_tables: Array,
) -> Array:
    """Integrate the heat probability-flow ODE backward with Heun steps."""

    state = initial.copy()
    for index in range(taus.size - 1):
        dt = float(taus[index + 1] - taus[index])
        score_now = np.interp(state, grid, score_tables[index])
        velocity_now = -0.5 * score_now
        proposal = state + dt * velocity_now
        score_next = np.interp(proposal, grid, score_tables[index + 1])
        velocity_next = -0.5 * score_next
        state += 0.5 * dt * (velocity_now + velocity_next)
    return state


def sample_metrics(samples: Array, target_density: Array, grid: Array) -> dict[str, float]:
    target = quantiles_from_density(target_density, grid, samples.size)
    ordered = np.sort(samples)
    difference = ordered - target
    return {
        "w1": float(np.mean(np.abs(difference))),
        "w2": float(np.sqrt(np.mean(np.square(difference)))),
        "mean_error": float(abs(samples.mean() - target.mean())),
        "std_error": float(abs(samples.std() - target.std())),
    }


def build_score_tables(
    *,
    grid: Array,
    weak: GaussianMixture1D,
    strong: GaussianMixture1D,
    target_clean: Array,
    beta: float,
    taus: Array,
) -> tuple[Array, Array, Array]:
    exact_rows = []
    static_rows = []
    local_rows = []
    for tau in taus:
        target_density = heat_evolve_periodic(target_clean, grid, float(tau))
        exact_score = score_from_density(target_density, grid)
        weak_score = weak.score(grid, heat_variance=float(tau))
        strong_score = strong.score(grid, heat_variance=float(tau))
        static_score = beta * strong_score + (1.0 - beta) * weak_score
        relative_score = strong_score - weak_score
        risk = 0.5 * beta * (beta - 1.0) * float(tau) * np.square(
            relative_score
        )
        risk_gradient = np.gradient(risk, grid, edge_order=2)
        exact_rows.append(exact_score)
        static_rows.append(static_score)
        local_rows.append(static_score + risk_gradient)
    return np.stack(exact_rows), np.stack(static_rows), np.stack(local_rows)


def soft_bellman_score_tables(
    *,
    grid: Array,
    weak: GaussianMixture1D,
    strong: GaussianMixture1D,
    beta: float,
    forward_taus: Array,
    max_substep: float = 0.01,
) -> Array:
    """Recover the semigroup correction using only weak/strong score fields.

    For ``k = exp(delta)``, the conditional-Jensen value obeys

        d_tau k = (0.5 Laplacian + s_static * grad) k + c * k,
        c = 0.5 * beta * (beta - 1) * |s_strong - s_weak|^2,
        k(0) = 1.

    A Lie--Trotter Bellman step applies Gaussian expectation, characteristic
    transport, and the running cost. Multiplying ``k`` by a time-dependent
    scalar does not change ``grad log k``, so each step is renormalized for
    numerical stability. No endpoint density ratio is used here.
    """

    taus = np.asarray(forward_taus, dtype=np.float64)
    if taus.ndim != 1 or taus.size < 2:
        raise ValueError("forward_taus must be a one-dimensional time grid")
    if taus[0] != 0.0 or np.any(np.diff(taus) <= 0.0):
        raise ValueError("forward_taus must increase strictly from zero")
    if beta <= 1.0 or max_substep <= 0.0:
        raise ValueError("beta must exceed one and max_substep must be positive")

    value = np.ones_like(grid, dtype=np.float64)
    score_rows = []

    def corrected_score(tau: float) -> Array:
        weak_score = weak.score(grid, heat_variance=tau)
        strong_score = strong.score(grid, heat_variance=tau)
        static_score = beta * strong_score + (1.0 - beta) * weak_score
        value_score = np.gradient(
            np.log(np.maximum(value, 1e-300)), grid, edge_order=2
        )
        return static_score + value_score

    score_rows.append(corrected_score(0.0))
    current_tau = 0.0
    for target_tau in taus[1:]:
        interval = float(target_tau - current_tau)
        substeps = max(1, int(math.ceil(interval / max_substep)))
        dt = interval / substeps
        for _ in range(substeps):
            weak_score = weak.score(grid, heat_variance=current_tau)
            strong_score = strong.score(grid, heat_variance=current_tau)
            gap = strong_score - weak_score
            static_score = weak_score + beta * gap
            running_cost = 0.5 * beta * (beta - 1.0) * np.square(gap)

            diffused = heat_apply_periodic(value, grid, dt)
            query = np.clip(
                grid + dt * static_score,
                float(grid[0]),
                float(grid[-1]),
            )
            transported = np.interp(query, grid, diffused)
            value = np.exp(np.minimum(dt * running_cost, 50.0)) * np.maximum(
                transported, 1e-300
            )
            log_normalizer = float(np.median(np.log(value)))
            value *= math.exp(-log_normalizer)
            if not np.all(np.isfinite(value)):
                raise FloatingPointError("soft Bellman value became non-finite")
            current_tau += dt
        current_tau = float(target_tau)
        score_rows.append(corrected_score(current_tau))
    return np.stack(score_rows)


def plot_results(
    *,
    output: Path,
    grid: Array,
    weak_clean: Array,
    strong_clean: Array,
    target_clean: Array,
    audit: dict[str, Array],
    samples: dict[str, Array],
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    support = audit["target_noisy"] > audit["target_noisy"].max() * 1e-7
    visible = (grid >= -7.0) & (grid <= 7.0)
    score_mask = support & visible
    axes[0, 0].plot(grid, weak_clean, label="weak endpoint", linewidth=1.8)
    axes[0, 0].plot(grid, strong_clean, label="strong endpoint", linewidth=1.8)
    axes[0, 0].plot(grid, target_clean, label="power-tilted target", linewidth=2.2)
    axes[0, 0].set_title("Endpoint densities")
    axes[0, 0].legend()

    axes[0, 1].plot(
        grid[score_mask], audit["static_score"][score_mask], label="static power mix"
    )
    axes[0, 1].plot(
        grid[score_mask], audit["target_score"][score_mask], label="valid semigroup score"
    )
    axes[0, 1].plot(
        grid[score_mask], audit["correction"][score_mask], label="missing Jensen correction"
    )
    axes[0, 1].set_title("Scores at an intermediate noise level")
    axes[0, 1].legend()

    axes[1, 0].plot(grid[score_mask], audit["delta"][score_mask], color="tab:purple")
    axes[1, 0].set_title("Conditional Jensen potential")

    bins = np.linspace(-7.0, 7.0, 180)
    axes[1, 1].plot(grid, target_clean, label="target", color="black", linewidth=2.2)
    for name, values in samples.items():
        axes[1, 1].hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.5,
            label=name,
        )
    axes[1, 1].set_title("Probability-flow endpoints")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.set_xlim(-7.0, 7.0)
        axis.grid(alpha=0.2)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/tmp/semigroup_consistent_guidance_toy"),
    )
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--grid-size", type=int, default=8192)
    parser.add_argument("--time-steps", type=int, default=1000)
    parser.add_argument("--sample-count", type=int, default=20000)
    parser.add_argument("--max-variance", type=float, default=36.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.beta <= 1.0:
        raise ValueError("beta must exceed one for extrapolative guidance")
    args.output_root.mkdir(parents=True, exist_ok=True)

    grid = np.linspace(-24.0, 24.0, args.grid_size, endpoint=False)
    weak = GaussianMixture1D(
        weights=(0.30, 0.44, 0.26),
        means=(-3.2, -0.15, 3.0),
        stds=(0.95, 1.15, 0.90),
    )
    # Shared components make p/q bounded. This removes irrelevant numerical
    # tail explosions while retaining a nontrivial endpoint evidence tilt.
    strong = GaussianMixture1D(
        weights=(0.20, 0.60, 0.20),
        means=weak.means,
        stds=weak.stds,
    )
    weak_clean = normalize_density(weak.density(grid), grid)
    strong_clean = normalize_density(strong.density(grid), grid)
    target_clean = normalize_density(
        np.power(strong_clean, args.beta)
        * np.power(np.maximum(weak_clean, 1e-300), 1.0 - args.beta),
        grid,
    )

    audit_variances = (0.0, 0.25, 1.0, 4.0, args.max_variance)
    audits = {
        variance: conditional_jensen_terms(
            grid=grid,
            weak_clean=weak_clean,
            strong_clean=strong_clean,
            beta=args.beta,
            heat_variance=variance,
        )
        for variance in audit_variances
    }

    # Quadratic spacing resolves the clean endpoint while retaining a broad
    # high-noise start.  Reverse the array for noise-to-data integration.
    forward_taus = np.square(
        np.linspace(0.0, math.sqrt(args.max_variance), args.time_steps + 1)
    )
    taus = forward_taus[::-1].copy()
    exact_scores, static_scores, local_scores = build_score_tables(
        grid=grid,
        weak=weak,
        strong=strong,
        target_clean=target_clean,
        beta=args.beta,
        taus=taus,
    )
    bellman_scores = soft_bellman_score_tables(
        grid=grid,
        weak=weak,
        strong=strong,
        beta=args.beta,
        forward_taus=forward_taus,
    )[::-1].copy()
    target_start = heat_evolve_periodic(target_clean, grid, args.max_variance)
    initial = quantiles_from_density(target_start, grid, args.sample_count)
    samples = {
        "static": sample_probability_flow(
            initial=initial, taus=taus, grid=grid, score_tables=static_scores
        ),
        "local-risk": sample_probability_flow(
            initial=initial, taus=taus, grid=grid, score_tables=local_scores
        ),
        "soft-bellman": sample_probability_flow(
            initial=initial, taus=taus, grid=grid, score_tables=bellman_scores
        ),
        "exact-semigroup": sample_probability_flow(
            initial=initial, taus=taus, grid=grid, score_tables=exact_scores
        ),
    }
    rows = []
    for name, values in samples.items():
        row = {"method": name, **sample_metrics(values, target_clean, grid)}
        rows.append(row)

    identity_rows = []
    for variance, audit in audits.items():
        density_weight = audit["target_noisy"]
        mask = density_weight > density_weight.max() * 1e-8
        relative_score = audit["strong_score"] - audit["weak_score"]
        local_potential = (
            0.5
            * args.beta
            * (args.beta - 1.0)
            * float(variance)
            * np.square(relative_score)
        )
        local_correction = np.gradient(local_potential, grid, edge_order=2)
        identity_error = np.max(
            np.abs(audit["correction"][mask] - audit["delta_gradient"][mask])
        )
        correction_rms = math.sqrt(
            float(
                np.trapezoid(
                    density_weight * np.square(audit["correction"]), grid
                )
            )
        )
        local_rms = math.sqrt(
            float(
                np.trapezoid(
                    density_weight * np.square(local_correction), grid
                )
            )
        )
        cross = float(
            np.trapezoid(
                density_weight * audit["correction"] * local_correction, grid
            )
        )
        denominator = max(correction_rms * local_rms, 1e-300)
        identity_rows.append(
            {
                "heat_variance": variance,
                "max_score_identity_error": float(identity_error),
                "correction_rms_under_target": correction_rms,
                "local_correction_rms_under_target": local_rms,
                "local_exact_weighted_cosine": cross / denominator,
                "delta_spatial_range_on_support": float(
                    audit["delta"][mask].max() - audit["delta"][mask].min()
                ),
            }
        )

    with (args.output_root / "endpoint_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_root / "score_identity.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(identity_rows[0]))
        writer.writeheader()
        writer.writerows(identity_rows)
    np.savez_compressed(
        args.output_root / "paired_endpoints.npz",
        initial=initial,
        **samples,
    )
    plot_results(
        output=args.output_root / "semigroup_guidance_toy.png",
        grid=grid,
        weak_clean=weak_clean,
        strong_clean=strong_clean,
        target_clean=target_clean,
        audit=audits[1.0],
        samples=samples,
    )
    summary = {
        "beta": args.beta,
        "grid_size": args.grid_size,
        "time_steps": args.time_steps,
        "sample_count": args.sample_count,
        "max_variance": args.max_variance,
        "endpoint_metrics": rows,
        "score_identity": identity_rows,
        "interpretation": {
            "static": "heat first, then power-mix noisy marginals",
            "exact_semigroup": "power-tilt endpoint first, then heat-noise",
            "local_risk": "lowest-order squared-relative-score correction",
            "soft_bellman": "full value recovered recursively from score gaps only",
        },
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
