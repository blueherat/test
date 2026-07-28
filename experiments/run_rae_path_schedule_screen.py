"""Screen well-conditioned RAE path schedules using existing oracle results."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from experiments.rae_path_schedule_screen import (
    Schedule,
    endpoint_observation_factor,
    random_control_table,
    schedule_coefficient,
    screen_schedules,
)


DEFAULT_INPUT = (
    Path.home()
    / "data/eqvae/experiments/rae_path_conditioning/oracle_seed20260718_start128_n32"
)
DEFAULT_OUTPUT = Path.home() / "data/eqvae/experiments/rae_path_schedule_screen"


@dataclass(frozen=True)
class ScreenConfig:
    input_root: Path = DEFAULT_INPUT
    output_root: Path = DEFAULT_OUTPUT
    run_name: str = "offline_oracle_n32_v1"
    bootstrap_iterations: int = 500
    bootstrap_seed: int = 20_260_722


def _sample_rms(value: torch.Tensor) -> np.ndarray:
    return value.float().square().mean(dim=1).sqrt().cpu().numpy()


def estimate_decoder_weights(
    input_root: Path, metrics: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Estimate local component sensitivities from component-oracle features."""

    payload = None
    for path in sorted(input_root.glob("features_rank*.pt")):
        candidate = torch.load(path, map_location="cpu", weights_only=True)
        if candidate["path"] == "annealed":
            payload = candidate
            break
    if payload is None:
        raise FileNotFoundError("annealed feature payload was not found")
    endpoint = payload["features"]["step49_corrected"].float()
    rows: list[dict[str, float | int]] = []
    annealed = metrics[metrics.path == "annealed"]
    for step in (32, 40, 44):
        selected = annealed[annealed.step_index == step].sort_values("sample_index")
        if len(selected) != len(endpoint):
            raise ValueError(f"step {step} metrics and feature counts differ")
        semantic_feature_error = _sample_rms(
            payload["features"][f"step{step:02d}_basis_oracle"] - endpoint
        )
        basis_feature_error = _sample_rms(
            payload["features"][f"step{step:02d}_semantic_oracle"] - endpoint
        )
        semantic_error = selected.semantic_relative_error.to_numpy(dtype=np.float64)
        basis_error = selected.basis_relative_error.to_numpy(dtype=np.float64)
        for offset, sample_index in enumerate(selected.sample_index):
            rows.append(
                {
                    "sample_index": int(sample_index),
                    "step_index": step,
                    "semantic_feature_error": float(semantic_feature_error[offset]),
                    "basis_feature_error": float(basis_feature_error[offset]),
                    "semantic_relative_error": float(semantic_error[offset]),
                    "basis_relative_error": float(basis_error[offset]),
                    "semantic_sensitivity_squared": float(
                        (semantic_feature_error[offset] / max(semantic_error[offset], 1e-12))
                        ** 2
                    ),
                    "basis_sensitivity_squared": float(
                        (basis_feature_error[offset] / max(basis_error[offset], 1e-12))
                        ** 2
                    ),
                }
            )
    table = pd.DataFrame(rows)
    weights = {
        "semantic_weight": float(table.semantic_sensitivity_squared.median()),
        "basis_weight": float(table.basis_sensitivity_squared.median()),
        "semantic_to_basis_weight_ratio": float(
            table.semantic_sensitivity_squared.median()
            / table.basis_sensitivity_squared.median()
        ),
    }
    return table, weights


def _candidate_grid() -> list[Schedule]:
    schedules = []
    for floor in (0.02, 0.05, 0.10, 0.15, 0.20, 0.30):
        for power in (1.0, 2.0, 3.0, 4.0):
            schedules.append(Schedule("floor_power", floor, power))
        for alpha in (0.5, 1.0, 2.0, 4.0):
            schedules.append(Schedule("floor_rational", floor, alpha))
    return schedules


def bootstrap_schedule_stability(
    oracle_metrics: pd.DataFrame,
    sensitivity: pd.DataFrame,
    schedules: list[Schedule],
    *,
    semantic_weight: float,
    basis_weight: float,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    """Bootstrap paired samples while re-estimating step-local decoder weights."""

    if iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    sample_ids = np.array(
        sorted(set(oracle_metrics.sample_index).intersection(sensitivity.sample_index))
    )
    if len(sample_ids) < 2:
        raise ValueError("at least two paired samples are required for bootstrap")
    rng = np.random.default_rng(int(seed))
    rows: list[pd.DataFrame] = []
    for iteration in range(int(iterations)):
        drawn = rng.choice(sample_ids, size=len(sample_ids), replace=True)
        metric_parts = []
        sensitivity_parts = []
        for bootstrap_index, sample_index in enumerate(drawn):
            metric_part = oracle_metrics[
                oracle_metrics.sample_index == sample_index
            ].copy()
            sensitivity_part = sensitivity[
                sensitivity.sample_index == sample_index
            ].copy()
            metric_part["sample_index"] = bootstrap_index
            sensitivity_part["sample_index"] = bootstrap_index
            metric_parts.append(metric_part)
            sensitivity_parts.append(sensitivity_part)
        sampled_metrics = pd.concat(metric_parts, ignore_index=True)
        sampled_sensitivity = pd.concat(sensitivity_parts, ignore_index=True)
        step_weights = (
            sampled_sensitivity.groupby("step_index", as_index=False)
            .agg(
                semantic_decoder_weight=(
                    "semantic_sensitivity_squared",
                    "median",
                ),
                basis_decoder_weight=("basis_sensitivity_squared", "median"),
            )
        )
        sampled_metrics = sampled_metrics.merge(
            step_weights, on="step_index", validate="many_to_one"
        )
        summary, _ = screen_schedules(
            sampled_metrics,
            schedules,
            semantic_weight=semantic_weight,
            basis_weight=basis_weight,
        )
        rows.append(
            summary[
                [
                    "schedule",
                    "passes_gate",
                    "total_risk_ratio",
                    "path_excess_risk_ratio",
                    "worst_step_excess_risk_ratio",
                ]
            ].assign(iteration=iteration)
        )
    samples = pd.concat(rows, ignore_index=True)
    return (
        samples.groupby("schedule", as_index=False)
        .agg(
            bootstrap_iterations=("iteration", "nunique"),
            pass_rate=("passes_gate", "mean"),
            total_risk_ratio_median=("total_risk_ratio", "median"),
            total_risk_ratio_q05=("total_risk_ratio", lambda x: x.quantile(0.05)),
            total_risk_ratio_q95=("total_risk_ratio", lambda x: x.quantile(0.95)),
            excess_risk_ratio_median=("path_excess_risk_ratio", "median"),
            excess_risk_ratio_q05=(
                "path_excess_risk_ratio",
                lambda x: x.quantile(0.05),
            ),
            excess_risk_ratio_q95=(
                "path_excess_risk_ratio",
                lambda x: x.quantile(0.95),
            ),
            worst_step_ratio_q95=(
                "worst_step_excess_risk_ratio",
                lambda x: x.quantile(0.95),
            ),
        )
        .sort_values(["pass_rate", "excess_risk_ratio_median"], ascending=[False, True])
        .reset_index(drop=True)
    )


def _plot(summary: pd.DataFrame, schedules: list[Schedule], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=True)
    markers = {"floor_power": "o", "floor_rational": "s"}
    for family, rows in summary.groupby("family"):
        axes[0].scatter(
            rows.delay_retention,
            rows.path_excess_risk_ratio,
            c=rows.floor,
            cmap="viridis",
            marker=markers[family],
            s=90,
            edgecolor="black",
            linewidth=0.5,
            label=family,
        )
    passed = summary[summary.passes_gate]
    axes[0].scatter(
        passed.delay_retention,
        passed.path_excess_risk_ratio,
        facecolors="none",
        edgecolors="red",
        linewidth=2.0,
        s=180,
        label="passes gate",
    )
    axes[0].axvline(0.70, color="black", linestyle="--", linewidth=1)
    axes[0].axhline(0.70, color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Detail delay retention vs original")
    axes[0].set_ylabel("Path-induced decoder risk / original")
    axes[0].set_title("Offline schedule Pareto screen")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    time = np.linspace(0.0, 1.0, 501)
    current = Schedule("floor_power", 0.0, 2.0)
    axes[1].plot(
        time,
        endpoint_observation_factor(time, current),
        color="black",
        linewidth=3,
        label="current: floor=0, p=2",
    )
    selected_names = set(passed.head(5).schedule)
    by_name = {schedule.name: schedule for schedule in schedules}
    for name in selected_names:
        schedule = by_name[name]
        axes[1].plot(
            time,
            endpoint_observation_factor(time, schedule),
            linewidth=2,
            label=name,
        )
    axes[1].axhline(0.05, color="red", linestyle="--", linewidth=1)
    axes[1].set_xlabel("t (1 is high noise)")
    axes[1].set_ylabel("Endpoint observation factor k(t)")
    axes[1].set_title("Conditioning of best passing schedules")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    figure.savefig(output / "schedule_pareto.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 7), constrained_layout=True)
    current_c, _ = schedule_coefficient(time, current)
    axis.plot(time, current_c, color="black", linewidth=3, label="current")
    for name in selected_names:
        schedule = by_name[name]
        coefficient, _ = schedule_coefficient(time, schedule)
        axis.plot(time, coefficient, linewidth=2, label=name)
    axis.set_xlabel("t (1 is high noise)")
    axis.set_ylabel("Detail coefficient c(t)")
    axis.set_title("Delay retained by best conditioned schedules")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=9)
    figure.savefig(output / "schedule_coefficients.png", dpi=180)
    plt.close(figure)


def run(config: ScreenConfig) -> Path:
    input_root = config.input_root.expanduser().resolve()
    output = config.output_root.expanduser().resolve() / config.run_name
    output.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(input_root / "conditioning_metrics.csv")
    annealed = metrics[metrics.path == "annealed"].copy()
    sensitivity, weights = estimate_decoder_weights(input_root, metrics)
    step_weights = (
        sensitivity.groupby("step_index", as_index=False)
        .agg(
            semantic_decoder_weight=("semantic_sensitivity_squared", "median"),
            basis_decoder_weight=("basis_sensitivity_squared", "median"),
        )
    )
    oracle_annealed = annealed[annealed.step_index.isin(step_weights.step_index)].merge(
        step_weights, on="step_index", validate="many_to_one"
    )
    schedules = _candidate_grid()
    summary, details = screen_schedules(
        oracle_annealed,
        schedules,
        semantic_weight=weights["semantic_weight"],
        basis_weight=weights["basis_weight"],
    )
    all_step_summary, all_step_details = screen_schedules(
        annealed,
        schedules,
        semantic_weight=weights["semantic_weight"],
        basis_weight=weights["basis_weight"],
    )
    bootstrap_shortlist_names = set(
        summary[
            (summary.min_observation_factor >= 0.05 - 1e-12)
            & (summary.delay_retention >= 0.70)
            & (summary.worst_step_excess_risk_ratio <= 0.90)
        ].schedule
    )
    bootstrap_schedules = [
        schedule for schedule in schedules if schedule.name in bootstrap_shortlist_names
    ]
    bootstrap = bootstrap_schedule_stability(
        annealed[annealed.step_index.isin(step_weights.step_index)],
        sensitivity,
        bootstrap_schedules,
        semantic_weight=weights["semantic_weight"],
        basis_weight=weights["basis_weight"],
        iterations=config.bootstrap_iterations,
        seed=config.bootstrap_seed,
    )

    result_payload = json.loads((input_root / "result.json").read_text(encoding="utf-8"))
    random_run = next(row for row in result_payload["runs"] if row["path"] == "random")
    annealed_manifest = json.loads(
        (
            Path(result_payload["config"]["branch_root"])
            / "seed3407_annealed_rank16_s0_to_10000/manifest.json"
        ).read_text(encoding="utf-8")
    )
    subspaces = torch.load(
        Path(annealed_manifest["subspace_path"]), map_location="cpu", weights_only=False
    )
    entry = subspaces["subspaces"].get(16, subspaces["subspaces"].get("16"))
    explained = float(entry["explained_final_fraction"])
    controls = random_control_table(
        channels=int(entry["basis"].shape[0]),
        guided_rank=int(entry["basis"].shape[1]),
        guided_explained_fraction=explained,
    )
    old_scale = float(random_run["detail_scale"])
    if not np.isclose(controls.iloc[0].latent_scale, old_scale, rtol=1e-7, atol=1e-9):
        raise RuntimeError("reconstructed random-control scale does not match manifest")

    sensitivity.to_csv(output / "decoder_sensitivity_samples.csv", index=False)
    step_weights.to_csv(output / "decoder_sensitivity_by_step.csv", index=False)
    summary.to_csv(output / "schedule_summary_oracle_steps.csv", index=False)
    details.to_csv(output / "schedule_sample_details_oracle_steps.csv", index=False)
    all_step_summary.to_csv(output / "schedule_summary_all_steps_extrapolated.csv", index=False)
    all_step_details.to_csv(output / "schedule_sample_details_all_steps_extrapolated.csv", index=False)
    bootstrap.to_csv(output / "schedule_bootstrap_stability.csv", index=False)
    controls.to_csv(output / "random_control_redesign.csv", index=False)
    _plot(summary, schedules, output)
    passing = summary[summary.passes_gate]
    result = {
        "config": {
            **asdict(config),
            "input_root": str(config.input_root),
            "output_root": str(config.output_root),
        },
        "decoder_weights": weights,
        "candidate_count": len(summary),
        "passing_count": len(passing),
        "passing_shape_count": int(passing[["family", "shape"]].drop_duplicates().shape[0]),
        "bootstrap_shortlist_count": len(bootstrap_schedules),
        "bootstrap_stable_count": int((bootstrap.pass_rate >= 0.90).sum()),
        "predictions": {
            "p1_floor_005_is_well_conditioned": bool(
                (summary[summary.floor >= 0.05].min_observation_factor >= 0.05 - 1e-12).all()
            ),
            "p2_multiple_shapes_pass": bool(
                passing[["family", "shape"]].drop_duplicates().shape[0] >= 2
            ),
            "p3_total_improvement_smaller_than_excess": bool(
                (
                    (1.0 - passing.total_risk_ratio)
                    < (1.0 - passing.path_excess_risk_ratio)
                ).all()
                if len(passing)
                else False
            ),
            "p4_candidate_exists_in_predicted_region": bool(
                len(
                    passing[
                        (passing.family == "floor_power")
                        & passing.floor.between(0.10, 0.20)
                        & passing["shape"].between(2.0, 3.0)
                    ]
                )
                > 0
            ),
            "p5_random_controls_split": bool(
                int(controls.loc[controls.control == "energy_rank_unscaled", "rank"].iloc[0])
                != 16
                and not bool(
                    controls.loc[
                        controls.control == "old_scaled_rank_matched", "clean_path_geometry"
                    ].iloc[0]
                )
            ),
        },
        "scope": (
            "single-seed n32 offline counterfactual; primary risk uses component-oracle "
            "steps 32/40/44 with step-local sensitivity; fixed raw model error assumption"
        ),
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("\nTop schedules:\n", summary.head(16).to_string(index=False))
    print("\nBootstrap stability:\n", bootstrap.to_string(index=False))
    print("\nRandom controls:\n", controls.to_string(index=False))
    return output


def parse_args() -> ScreenConfig:
    defaults = ScreenConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=defaults.input_root)
    parser.add_argument("--output-root", type=Path, default=defaults.output_root)
    parser.add_argument("--run-name", default=defaults.run_name)
    parser.add_argument(
        "--bootstrap-iterations", type=int, default=defaults.bootstrap_iterations
    )
    parser.add_argument("--bootstrap-seed", type=int, default=defaults.bootstrap_seed)
    args = parser.parse_args()
    return ScreenConfig(
        args.input_root,
        args.output_root,
        args.run_name,
        args.bootstrap_iterations,
        args.bootstrap_seed,
    )


if __name__ == "__main__":
    run(parse_args())
