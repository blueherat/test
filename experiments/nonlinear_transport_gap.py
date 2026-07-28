"""Exact-oracle study of teacher-risk versus self-induced transport.

The underlying Gaussian-mixture flow has an analytic conditional velocity.
This lets the audit evaluate learned fields on teacher states, model rollout
states, and continuous interpolations between them without using either model
as a surrogate ground truth.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.nonlinear_fm_whitening_toy import (
    MixtureFMConfig,
    NeuralRun,
    NeuralTrainConfig,
    distribution_metrics,
    problem_tensors,
    resolve_device,
    run_training_grid_parallel,
    sample_latent_reference,
    velocity_field,
)


@dataclass(frozen=True)
class TransportGapConfig:
    output_root: Path = Path.home() / "data/eqvae/experiments/nonlinear_transport_gap"
    architectures: tuple[str, ...] = ("mlp", "mini_dit")
    hidden_sizes: tuple[int, ...] = (24, 96)
    gammas: tuple[float, ...] = (0.0, 0.5)
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    devices: tuple[str, ...] = ("cuda:0", "cuda:1", "cuda:2", "cuda:3")
    batch_size: int = 128
    steps: int = 800
    learning_rate: float = 2e-3
    depth: int = 2
    num_heads: int = 4
    eval_every: int = 40
    eval_count: int = 4096
    sample_count: int = 4096
    ode_steps: int = 80
    solver_check_steps: tuple[int, ...] = (80, 160)
    target_times: tuple[float, ...] = (0.9, 0.7, 0.5, 0.3, 0.1)
    alphas: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    save: bool = True


@dataclass
class TransportGapResult:
    training_summary: pd.DataFrame
    training_history: pd.DataFrame
    endpoint_metrics: pd.DataFrame
    state_metrics: pd.DataFrame
    direction_metrics: pd.DataFrame
    pair_summary: pd.DataFrame
    solver_audit: pd.DataFrame
    result_dir: Path | None


def _validate_config(config: TransportGapConfig) -> None:
    if set(config.gammas) != {0.0, 0.5}:
        raise ValueError("the preregistered first run requires gammas 0 and 0.5")
    if any(size < 1 for size in config.hidden_sizes):
        raise ValueError("hidden sizes must be positive")
    if any(not 0.0 <= value <= 1.0 for value in config.target_times):
        raise ValueError("target times must lie in [0, 1]")
    if any(not 0.0 <= value <= 1.0 for value in config.alphas):
        raise ValueError("alphas must lie in [0, 1]")
    if config.sample_count < 2 or config.eval_count < 2:
        raise ValueError("sample and evaluation counts must be at least two")
    if config.ode_steps < 2:
        raise ValueError("ode_steps must be at least two")


def _generator(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(int(seed))


@torch.no_grad()
def rollout_snapshots(
    problem: MixtureFMConfig,
    initial: torch.Tensor,
    target_times: Sequence[float],
    *,
    model: torch.nn.Module | None,
    ode_steps: int,
    oracle: bool = False,
) -> dict[float, torch.Tensor]:
    """Integrate with Heun and return states on exact grid-aligned times."""

    if int(ode_steps) < 2:
        raise ValueError("ode_steps must be at least two")
    device = initial.device
    requested = tuple(float(value) for value in target_times)
    grid = torch.linspace(1.0, 0.0, int(ode_steps) + 1, device=device)
    indices = {
        value: int(torch.argmin(torch.abs(grid - value)).item()) for value in requested
    }
    for value, index in indices.items():
        if not math.isclose(float(grid[index]), value, abs_tol=1e-6):
            raise ValueError(f"target time {value} is not aligned to the ODE grid")

    state = initial
    snapshots: dict[float, torch.Tensor] = {}
    if 1.0 in indices:
        snapshots[1.0] = state.detach().cpu()
    for index in range(int(ode_steps)):
        current = grid[index]
        following = grid[index + 1]
        current_batch = torch.full((len(state), 1), current, device=device)
        first = velocity_field(model, state, current_batch, problem, oracle=oracle)
        proposal = state + (following - current) * first
        following_batch = torch.full((len(state), 1), following, device=device)
        second = velocity_field(model, proposal, following_batch, problem, oracle=oracle)
        state = state + 0.5 * (following - current) * (first + second)
        completed_index = index + 1
        for value, requested_index in indices.items():
            if completed_index == requested_index:
                snapshots[value] = state.detach().cpu()
    missing = set(requested) - set(snapshots)
    if missing:
        raise RuntimeError(f"missing rollout snapshots: {sorted(missing)}")
    return snapshots


def _covariance_drift(state: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
    centered_state = state - state.mean(dim=0, keepdim=True)
    centered_velocity = velocity - velocity.mean(dim=0, keepdim=True)
    cross = centered_state.T @ centered_velocity / len(state)
    return cross + cross.T


def _field_metrics(
    state: torch.Tensor,
    prediction: torch.Tensor,
    oracle: torch.Tensor,
) -> tuple[dict[str, float], torch.Tensor]:
    error = prediction - oracle
    coordinate_drift = 2.0 * torch.mean(state * prediction, dim=0)
    oracle_coordinate_drift = 2.0 * torch.mean(state * oracle, dim=0)
    covariance_drift = _covariance_drift(state, prediction)
    oracle_covariance_drift = _covariance_drift(state, oracle)
    covariance_scale = torch.linalg.matrix_norm(oracle_covariance_drift).clamp_min(1e-8)
    metrics = {
        "field_mse": float(error.square().mean().item()),
        "mean_velocity_rmse": float(
            torch.mean((prediction.mean(dim=0) - oracle.mean(dim=0)).square()).sqrt().item()
        ),
        "coordinate_drift_rmse": float(
            torch.mean((coordinate_drift - oracle_coordinate_drift).square()).sqrt().item()
        ),
        "covariance_drift_rel_fro": float(
            (
                torch.linalg.matrix_norm(covariance_drift - oracle_covariance_drift)
                / covariance_scale
            ).item()
        ),
    }
    return metrics, error.square().mean(dim=0)


@torch.no_grad()
def shared_state_audit(
    problem: MixtureFMConfig,
    baseline_model: torch.nn.Module,
    weighted_model: torch.nn.Module,
    teacher_clean: torch.Tensor,
    initial_noise: torch.Tensor,
    baseline_states: Mapping[float, torch.Tensor],
    weighted_states: Mapping[float, torch.Tensor],
    target_times: Sequence[float],
    alphas: Sequence[float],
    *,
    device: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    actual_device = resolve_device(device)
    baseline_model = baseline_model.to(actual_device).eval()
    weighted_model = weighted_model.to(actual_device).eval()
    clean = teacher_clean.to(actual_device)
    noise = initial_noise.to(actual_device)
    state_rows: list[dict[str, float | str]] = []
    direction_rows: list[dict[str, float | int | str]] = []

    for time in target_times:
        time = float(time)
        time_batch = torch.full((len(clean), 1), time, device=actual_device)
        teacher_state = (1.0 - time) * clean + time * noise
        endpoints = {
            "teacher_to_baseline": baseline_states[time].to(actual_device),
            "teacher_to_weighted": weighted_states[time].to(actual_device),
        }
        for path, endpoint in endpoints.items():
            for alpha in alphas:
                alpha = float(alpha)
                state = (1.0 - alpha) * teacher_state + alpha * endpoint
                oracle_velocity = velocity_field(None, state, time_batch, problem, oracle=True)
                predictions = {
                    "baseline": velocity_field(baseline_model, state, time_batch, problem),
                    "weighted": velocity_field(weighted_model, state, time_batch, problem),
                }
                for variant, prediction in predictions.items():
                    metrics, direction_mse = _field_metrics(state, prediction, oracle_velocity)
                    state_rows.append(
                        {
                            "path": path,
                            "time": time,
                            "alpha": alpha,
                            "variant": variant,
                            **metrics,
                        }
                    )
                    for direction, value in enumerate(direction_mse.tolist()):
                        direction_rows.append(
                            {
                                "path": path,
                                "time": time,
                                "alpha": alpha,
                                "variant": variant,
                                "direction": int(direction),
                                "field_mse": float(value),
                            }
                        )

    baseline_model.cpu()
    weighted_model.cpu()
    if actual_device.type == "cuda":
        torch.cuda.empty_cache()
    return pd.DataFrame(state_rows), pd.DataFrame(direction_rows)


def _sliced_wasserstein(
    generated: torch.Tensor,
    reference: torch.Tensor,
    *,
    seed: int,
    directions: int = 128,
) -> float:
    if generated.shape != reference.shape:
        raise ValueError("generated and reference must have equal shape")
    generator = torch.Generator().manual_seed(int(seed))
    vectors = torch.randn(
        (generated.shape[1], int(directions)), generator=generator, dtype=torch.float64
    )
    vectors = vectors / torch.linalg.vector_norm(vectors, dim=0, keepdim=True).clamp_min(1e-12)
    left = torch.sort(generated.double() @ vectors, dim=0).values
    right = torch.sort(reference.double() @ vectors, dim=0).values
    return float(torch.mean(torch.abs(left - right)).item())


def _endpoint_rows(
    generated: Mapping[str, torch.Tensor],
    reference: torch.Tensor,
    *,
    seed: int,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for variant, samples in generated.items():
        metrics = distribution_metrics(samples, reference)
        metrics["sliced_w1"] = _sliced_wasserstein(samples, reference, seed=seed + 401)
        for metric, value in metrics.items():
            rows.append({"variant": variant, "metric": metric, "value": float(value)})
    return rows


def _paired_ratio(
    frame: pd.DataFrame,
    metric: str,
    *,
    filters: Mapping[str, object] | None = None,
) -> float:
    selected = frame[frame["metric"].eq(metric)]
    for column, value in (filters or {}).items():
        selected = selected[selected[column].eq(value)]
    values = selected.groupby("variant")["value"].mean()
    return float(values["weighted"] / max(values["baseline"], 1e-12))


def summarize_pair(
    state_metrics: pd.DataFrame,
    direction_metrics: pd.DataFrame,
    endpoint_metrics: pd.DataFrame,
) -> dict[str, float | int | bool]:
    teacher = state_metrics[
        state_metrics["path"].eq("teacher_to_weighted") & state_metrics["alpha"].eq(0.0)
    ]
    teacher_values = teacher.groupby("variant")["field_mse"].mean()
    teacher_ratio = float(teacher_values["weighted"] / teacher_values["baseline"])

    teacher_direction = direction_metrics[
        direction_metrics["path"].eq("teacher_to_weighted")
        & direction_metrics["alpha"].eq(0.0)
    ].groupby(["variant", "direction"])["field_mse"].mean().unstack("variant")
    improved_directions = int(
        (teacher_direction["weighted"] < teacher_direction["baseline"]).sum()
    )

    paired = state_metrics.pivot_table(
        index=["path", "time", "alpha"],
        columns="variant",
        values=["field_mse", "coordinate_drift_rmse", "covariance_drift_rel_fro"],
    )
    paired.columns = [f"{metric}_{variant}" for metric, variant in paired.columns]
    paired = paired.reset_index()
    for metric in ("field_mse", "coordinate_drift_rmse", "covariance_drift_rel_fro"):
        paired[f"{metric}_ratio"] = paired[f"{metric}_weighted"] / paired[
            f"{metric}_baseline"
        ].clip(lower=1e-12)

    middle = paired[
        paired["path"].eq("teacher_to_weighted")
        & paired["time"].between(0.3, 0.7)
    ]
    slopes = []
    for _, group in middle.groupby("time"):
        slopes.append(
            float(np.polyfit(group["alpha"], np.log(group["field_mse_ratio"]), 1)[0])
        )
    alpha_zero = middle[middle["alpha"].eq(0.0)]["field_mse_ratio"].mean()
    alpha_one = middle[middle["alpha"].eq(1.0)]["field_mse_ratio"].mean()
    high_teacher = paired[
        paired["path"].eq("teacher_to_weighted")
        & paired["alpha"].eq(0.0)
        & paired["time"].ge(0.7)
    ]

    endpoint_w1_ratio = _paired_ratio(endpoint_metrics, "mean_coordinate_w1")
    endpoint_covariance_ratio = _paired_ratio(endpoint_metrics, "covariance_rel_fro")
    endpoint_sliced_ratio = _paired_ratio(endpoint_metrics, "sliced_w1")
    return {
        "teacher_field_mse_ratio": teacher_ratio,
        "teacher_improved_direction_count": improved_directions,
        "middle_offpath_field_ratio": float(alpha_one),
        "middle_teacher_field_ratio": float(alpha_zero),
        "middle_offpath_log_ratio_slope": float(np.mean(slopes)),
        "high_teacher_coordinate_drift_ratio": float(
            high_teacher["coordinate_drift_rmse_ratio"].mean()
        ),
        "high_teacher_covariance_drift_ratio": float(
            high_teacher["covariance_drift_rel_fro_ratio"].mean()
        ),
        "endpoint_coordinate_w1_ratio": endpoint_w1_ratio,
        "endpoint_covariance_ratio": endpoint_covariance_ratio,
        "endpoint_sliced_w1_ratio": endpoint_sliced_ratio,
        "partial_teacher_endpoint_reversal": bool(
            improved_directions > 0 and endpoint_w1_ratio > 1.0
        ),
        "strong_middle_crossover": bool(alpha_zero < 1.0 and alpha_one > 1.0),
    }


def _solver_audit(
    problem: MixtureFMConfig,
    config: TransportGapConfig,
    *,
    seed: int,
    device: str,
) -> pd.DataFrame:
    actual_device = resolve_device(device)
    generator = _generator(actual_device, 600_001 + int(seed))
    initial = torch.randn(
        (config.sample_count, problem.dimension), generator=generator, device=actual_device
    )
    endpoints: dict[int, torch.Tensor] = {}
    rows: list[dict[str, float | int | str]] = []
    reference = sample_latent_reference(problem, config.sample_count, seed + 501)
    for steps in config.solver_check_steps:
        endpoint = rollout_snapshots(
            problem,
            initial,
            (0.0,),
            model=None,
            ode_steps=int(steps),
            oracle=True,
        )[0.0]
        endpoints[int(steps)] = endpoint
        metrics = distribution_metrics(endpoint, reference)
        metrics["sliced_w1"] = _sliced_wasserstein(endpoint, reference, seed=seed + 503)
        for metric, value in metrics.items():
            rows.append(
                {
                    "seed": int(seed),
                    "ode_steps": int(steps),
                    "metric": metric,
                    "value": float(value),
                }
            )
    first, last = int(config.solver_check_steps[0]), int(config.solver_check_steps[-1])
    rows.append(
        {
            "seed": int(seed),
            "ode_steps": last,
            "metric": f"same_noise_rmse_vs_{first}",
            "value": float((endpoints[last] - endpoints[first]).square().mean().sqrt().item()),
        }
    )
    return pd.DataFrame(rows)


def _serialize_config(config: TransportGapConfig) -> dict[str, object]:
    values = asdict(config)
    values["output_root"] = str(config.output_root.expanduser())
    return values


def _save_models(
    path: Path,
    runs: Mapping[tuple, NeuralRun],
) -> None:
    payload = {}
    for key, run in runs.items():
        payload[str(key)] = {
            "problem": asdict(run.problem),
            "config": asdict(run.config),
            "state_dict": run.model.state_dict(),
        }
    torch.save(payload, path)


def run_transport_gap_study(
    config: TransportGapConfig = TransportGapConfig(),
    problem: MixtureFMConfig = MixtureFMConfig(),
) -> TransportGapResult:
    _validate_config(config)
    all_runs: dict[tuple, NeuralRun] = {}
    histories = []
    summaries = []
    for hidden_size in config.hidden_sizes:
        base_config = NeuralTrainConfig(
            batch_size=config.batch_size,
            steps=config.steps,
            learning_rate=config.learning_rate,
            hidden_size=int(hidden_size),
            depth=config.depth,
            num_heads=config.num_heads,
            eval_every=config.eval_every,
            eval_count=config.eval_count,
            device=config.devices[0] if config.devices else "cpu",
        )
        runs, history, summary = run_training_grid_parallel(
            problem,
            base_config,
            architectures=config.architectures,
            gammas=config.gammas,
            batch_sizes=(config.batch_size,),
            seeds=config.seeds,
            devices=config.devices,
            verbose=True,
        )
        for key, run in runs.items():
            all_runs[(int(hidden_size),) + key] = run
        history.insert(0, "hidden_size", int(hidden_size))
        summary.insert(0, "hidden_size", int(hidden_size))
        histories.append(history)
        summaries.append(summary)

    training_history = pd.concat(histories, ignore_index=True)
    training_summary = pd.concat(summaries, ignore_index=True)
    endpoint_frames = []
    state_frames = []
    direction_frames = []
    pair_rows = []
    solver_frames = []

    for seed_index, seed in enumerate(config.seeds):
        device = config.devices[seed_index % len(config.devices)] if config.devices else "cpu"
        solver_frames.append(_solver_audit(problem, config, seed=int(seed), device=device))

    for pair_index, (hidden_size, architecture, seed) in enumerate(
        (hidden_size, architecture, seed)
        for hidden_size in config.hidden_sizes
        for architecture in config.architectures
        for seed in config.seeds
    ):
        device = config.devices[pair_index % len(config.devices)] if config.devices else "cpu"
        actual_device = resolve_device(device)
        baseline = all_runs[
            (int(hidden_size), str(architecture), config.batch_size, 0.0, int(seed))
        ].model
        weighted = all_runs[
            (int(hidden_size), str(architecture), config.batch_size, 0.5, int(seed))
        ].model
        generator = _generator(actual_device, 1_000_003 + int(seed))
        initial = torch.randn(
            (config.sample_count, problem.dimension),
            generator=generator,
            device=actual_device,
        )
        teacher_clean = sample_latent_reference(problem, config.sample_count, int(seed) + 701)
        endpoint_reference = sample_latent_reference(
            problem, config.sample_count, int(seed) + 703
        )
        requested_times = tuple(config.target_times) + (0.0,)
        baseline_states = rollout_snapshots(
            problem,
            initial,
            requested_times,
            model=baseline.to(actual_device).eval(),
            ode_steps=config.ode_steps,
        )
        baseline.cpu()
        weighted_states = rollout_snapshots(
            problem,
            initial,
            requested_times,
            model=weighted.to(actual_device).eval(),
            ode_steps=config.ode_steps,
        )
        weighted.cpu()
        if actual_device.type == "cuda":
            torch.cuda.empty_cache()

        endpoint = pd.DataFrame(
            _endpoint_rows(
                {"baseline": baseline_states[0.0], "weighted": weighted_states[0.0]},
                endpoint_reference,
                seed=int(seed),
            )
        )
        endpoint.insert(0, "seed", int(seed))
        endpoint.insert(0, "architecture", str(architecture))
        endpoint.insert(0, "hidden_size", int(hidden_size))
        endpoint_frames.append(endpoint)

        state, direction = shared_state_audit(
            problem,
            baseline,
            weighted,
            teacher_clean,
            initial.cpu(),
            baseline_states,
            weighted_states,
            config.target_times,
            config.alphas,
            device=device,
        )
        for frame in (state, direction):
            frame.insert(0, "seed", int(seed))
            frame.insert(0, "architecture", str(architecture))
            frame.insert(0, "hidden_size", int(hidden_size))
        state_frames.append(state)
        direction_frames.append(direction)
        pair_rows.append(
            {
                "hidden_size": int(hidden_size),
                "architecture": str(architecture),
                "seed": int(seed),
                **summarize_pair(state, direction, endpoint),
            }
        )
        print(
            f"audit {architecture} h={hidden_size} seed={seed}: "
            f"teacher={pair_rows[-1]['teacher_field_mse_ratio']:.3f}, "
            f"offpath={pair_rows[-1]['middle_offpath_field_ratio']:.3f}, "
            f"endpoint_w1={pair_rows[-1]['endpoint_coordinate_w1_ratio']:.3f}"
        )

    endpoint_metrics = pd.concat(endpoint_frames, ignore_index=True)
    state_metrics = pd.concat(state_frames, ignore_index=True)
    direction_metrics = pd.concat(direction_frames, ignore_index=True)
    pair_summary = pd.DataFrame(pair_rows)
    solver_audit = pd.concat(solver_frames, ignore_index=True)

    result_dir = None
    if config.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = config.output_root.expanduser() / f"preregistered_v1_{timestamp}"
        result_dir.mkdir(parents=True, exist_ok=False)
        (result_dir / "config.json").write_text(
            json.dumps(_serialize_config(config), indent=2), encoding="utf-8"
        )
        (result_dir / "problem.json").write_text(
            json.dumps(asdict(problem), indent=2), encoding="utf-8"
        )
        training_summary.to_csv(result_dir / "training_summary.csv", index=False)
        training_history.to_csv(result_dir / "training_history.csv", index=False)
        endpoint_metrics.to_csv(result_dir / "endpoint_metrics.csv", index=False)
        state_metrics.to_csv(result_dir / "state_metrics.csv", index=False)
        direction_metrics.to_csv(result_dir / "direction_metrics.csv", index=False)
        pair_summary.to_csv(result_dir / "pair_summary.csv", index=False)
        solver_audit.to_csv(result_dir / "solver_audit.csv", index=False)
        _save_models(result_dir / "models.pt", all_runs)

    return TransportGapResult(
        training_summary=training_summary,
        training_history=training_history,
        endpoint_metrics=endpoint_metrics,
        state_metrics=state_metrics,
        direction_metrics=direction_metrics,
        pair_summary=pair_summary,
        solver_audit=solver_audit,
        result_dir=result_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3")
    parser.add_argument("--output-root", type=Path, default=TransportGapConfig.output_root)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    devices = tuple(value.strip() for value in args.devices.split(",") if value.strip())
    config = replace(
        TransportGapConfig(),
        devices=devices or ("cpu",),
        output_root=args.output_root,
        save=not args.no_save,
    )
    if args.quick:
        config = replace(
            config,
            architectures=("mlp",),
            hidden_sizes=(24,),
            seeds=(0,),
            devices=(devices[0],) if devices else ("cpu",),
            steps=4,
            eval_every=2,
            eval_count=32,
            sample_count=32,
            ode_steps=20,
            solver_check_steps=(20, 40),
            target_times=(0.9, 0.7, 0.5, 0.3, 0.1),
        )
    result = run_transport_gap_study(config)
    print(f"result_dir={result.result_dir}")
    print(result.pair_summary.to_string(index=False))


if __name__ == "__main__":
    main()
