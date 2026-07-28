"""Aggregate the RAE path crossover evidence into one decision and figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_ROOT = Path.home() / "data/eqvae/experiments"
CROSSOVER = DATA_ROOT / "rae_path_crossover_train_v2/crossover_evaluation"
GRADIENT = DATA_ROOT / "rae_path_gradient_interference/crossover_v2_n128_seed20260725"
CLOSURE = DATA_ROOT / "rae_path_schedule_closure"
EMA_CLOSURE = CLOSURE / "crossover_v2_step5000_n64_seed20260726"
ONLINE_CLOSURE = CLOSURE / "crossover_v2_online_step5000_n64_seed20260726"
CONDITIONS = (
    "floor_to_floor",
    "floor_to_static",
    "static_to_static",
    "static_to_floor",
)
COLORS = {
    "floor_to_floor": "#E45756",
    "floor_to_static": "#72B7B2",
    "static_to_static": "#4C78A8",
    "static_to_floor": "#F2CF5B",
}


def paired_median_interval(
    left: np.ndarray,
    right: np.ndarray,
    *,
    seed: int = 20_260_727,
    draws: int = 20_000,
) -> dict[str, float]:
    difference = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    if difference.ndim != 1 or len(difference) < 2:
        raise ValueError("paired bootstrap requires one-dimensional paired samples")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(difference), size=(draws, len(difference)))
    bootstrap = np.median(difference[indices], axis=1)
    low, high = np.quantile(bootstrap, (0.025, 0.975))
    return {
        "median_difference": float(np.median(difference)),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "left_lower_fraction": float(np.mean(difference < 0.0)),
        "sample_count": int(len(difference)),
    }


def generation_gate(table: pd.DataFrame) -> dict[str, object]:
    indexed = table.set_index("condition")
    metrics = ("frechet_inception_distance", "kernel_inception_distance_mean")
    checks = {
        "floor_to_static_beats_floor_to_floor": all(
            indexed.loc["floor_to_static", metric]
            < indexed.loc["floor_to_floor", metric]
            for metric in metrics
        ),
        "floor_to_static_beats_static_to_static": all(
            indexed.loc["floor_to_static", metric]
            < indexed.loc["static_to_static", metric]
            for metric in metrics
        ),
        "static_to_floor_worse_than_static_to_static": all(
            indexed.loc["static_to_floor", metric]
            > indexed.loc["static_to_static", metric]
            for metric in metrics
        ),
    }
    relative = {}
    for comparison, numerator, denominator in (
        ("floor_to_static_vs_static_to_static", "floor_to_static", "static_to_static"),
        ("floor_to_static_vs_floor_to_floor", "floor_to_static", "floor_to_floor"),
        ("static_to_floor_vs_static_to_static", "static_to_floor", "static_to_static"),
    ):
        relative[comparison] = {
            metric: float(indexed.loc[numerator, metric] / indexed.loc[denominator, metric] - 1.0)
            for metric in metrics
        }
    return {
        "pass": bool(all(checks.values())),
        "checks": {key: bool(value) for key, value in checks.items()},
        "relative_changes": relative,
    }


def _closure_intervals(table: pd.DataFrame) -> dict[str, object]:
    output = {}
    for switched, control in (
        ("floor_to_static", "floor_to_floor"),
        ("static_to_floor", "static_to_static"),
    ):
        comparison = f"{switched}_minus_{control}"
        output[comparison] = {}
        for metric in ("cycle_relative_rms", "local_decoder_sensitivity"):
            left = table[table.source == switched][metric].to_numpy()
            right = table[table.source == control][metric].to_numpy()
            output[comparison][metric] = paired_median_interval(left, right)
    return output


def analyze() -> dict[str, object]:
    replay = json.loads((CROSSOVER / "replay_verification.json").read_text())
    gradient_decision = json.loads((GRADIENT / "decision.json").read_text())
    gradient = pd.read_csv(GRADIENT / "aggregate_metrics.csv")
    generation_ema_1k = pd.read_csv(CROSSOVER / "generation_metrics_ema_n1000.csv")
    generation_online_1k = pd.read_csv(CROSSOVER / "generation_metrics_model_n1000.csv")
    generation_online_5k = pd.read_csv(CROSSOVER / "generation_metrics_model_n5000.csv")
    closure_ema = pd.read_csv(EMA_CLOSURE / "closure_metrics.csv")
    closure_online = pd.read_csv(ONLINE_CLOSURE / "closure_metrics.csv")
    return {
        "integrity": replay,
        "gradient": {
            "decision": gradient_decision,
            "t0p1_last_block": gradient[
                (gradient.time == 0.1) & (gradient.parameter_group == "last_block")
            ][
                [
                    "condition",
                    "semantic_descent_ratio",
                    "semantic_basis_cosine",
                    "basis_over_semantic_norm",
                ]
            ].to_dict(orient="records"),
        },
        "generation": {
            "online_5k_gate": generation_gate(generation_online_5k),
            "ema_1k": generation_ema_1k.to_dict(orient="records"),
            "online_1k": generation_online_1k.to_dict(orient="records"),
            "online_5k": generation_online_5k.to_dict(orient="records"),
        },
        "closure": {
            "ema_intervals": _closure_intervals(closure_ema),
            "online_intervals": _closure_intervals(closure_online),
        },
        "interpretation": {
            "late_path_controls_local_gradient_geometry": bool(
                gradient_decision["pass_late_path_gradient_prediction"]
            ),
            "online_curriculum_gate_pass": bool(
                generation_gate(generation_online_5k)["pass"]
            ),
            "short_run_ema_is_a_material_confounder": True,
            "scope": "single training seed; 5k online generation screen, not final gFID",
        },
    }


def plot_summary(result: dict[str, object], output: Path) -> None:
    gradient = pd.DataFrame(result["gradient"]["t0p1_last_block"])
    generation = pd.DataFrame(result["generation"]["online_5k"])
    closure = pd.read_csv(ONLINE_CLOSURE / "closure_summary.csv").set_index("source")
    ema = pd.DataFrame(result["generation"]["ema_1k"]).set_index("condition")
    online = pd.DataFrame(result["generation"]["online_1k"]).set_index("condition")

    figure, axes = plt.subplots(2, 2, figsize=(17, 11), constrained_layout=True)
    positions = np.arange(len(CONDITIONS))
    colors = [COLORS[condition] for condition in CONDITIONS]

    values = gradient.set_index("condition").loc[list(CONDITIONS)]
    axes[0, 0].bar(positions, values.semantic_descent_ratio, color=colors)
    axes[0, 0].axhline(1.0, color="#666666", linestyle="--")
    axes[0, 0].set_title("Online gradient: t=0.1 last block")
    axes[0, 0].set_ylabel("Semantic descent ratio")

    values = generation.set_index("condition").loc[list(CONDITIONS)]
    axes[0, 1].bar(positions, values.frechet_inception_distance, color=colors)
    axes[0, 1].set_title("Online model: fixed-seed n=5000")
    axes[0, 1].set_ylabel("FID (lower is better)")

    width = 0.36
    axes[1, 0].bar(
        positions - width / 2,
        ema.loc[list(CONDITIONS)].frechet_inception_distance,
        width,
        label="EMA",
        color="#9D9DA1",
    )
    axes[1, 0].bar(
        positions + width / 2,
        online.loc[list(CONDITIONS)].frechet_inception_distance,
        width,
        label="online",
        color=colors,
    )
    axes[1, 0].set_title("EMA lag at n=1000")
    axes[1, 0].set_ylabel("FID (lower is better)")
    axes[1, 0].legend(frameon=False)

    closure_values = closure.loc[list(CONDITIONS)]
    cycle = closure_values.cycle_relative_rms_median
    sensitivity = closure_values.local_decoder_sensitivity_median
    axes[1, 1].bar(positions - width / 2, cycle / cycle.loc["static_to_static"], width, label="cycle")
    axes[1, 1].bar(
        positions + width / 2,
        sensitivity / sensitivity.loc["static_to_static"],
        width,
        label="sensitivity",
    )
    axes[1, 1].axhline(1.0, color="#666666", linestyle="--")
    axes[1, 1].set_title("Online decoder closure (static->static = 1)")
    axes[1, 1].set_ylabel("Relative risk")
    axes[1, 1].legend(frameon=False)

    for axis in axes.flat:
        axis.set_xticks(positions, CONDITIONS, rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(output / "crossover_evidence_summary.png", dpi=180)
    plt.close(figure)


def main() -> None:
    output = CROSSOVER
    result = analyze()
    (output / "crossover_analysis.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    plot_summary(result, output)
    print(json.dumps(result["interpretation"], indent=2))
    print(json.dumps(result["generation"]["online_5k_gate"], indent=2))


if __name__ == "__main__":
    main()
