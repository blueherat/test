"""Shared-state transport audit for the matched-basis image study.

For each trained field pair, this audit measures diagonal transport moments
and one-step marginal movement on teacher, baseline-rollout, and weighted-
rollout states.  The held-out next-time interpolation marginal is the target;
neither learned field is treated as an oracle.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mnist_spectral_rollout_toy import (  # noqa: E402
    TinyVelocityUNet,
    descending_time_grid,
    sliced_wasserstein,
)
from experiments.small_image_basis_transport import (  # noqa: E402
    BasisStudyConfig,
    OrthogonalDirectionLoss,
    load_small_image_tensors,
)


@dataclass(frozen=True)
class BasisMechanismConfig:
    study_dir: Path
    devices: tuple[str, ...] = ("cuda:0", "cuda:1", "cuda:2", "cuda:3")
    audit_count: int = 512
    batch_size: int = 128
    ode_steps: int = 50
    target_times: tuple[float, ...] = (0.9, 0.7, 0.5, 0.3, 0.1)
    one_step: float = 0.02
    projection_count: int = 64
    save: bool = True


def reference_component_energy(clean_energy: torch.Tensor, time: float) -> torch.Tensor:
    return (1.0 - float(time)) ** 2 * clean_energy + float(time) ** 2


def reference_component_drift(clean_energy: torch.Tensor, time: float) -> torch.Tensor:
    return -2.0 * (1.0 - float(time)) * clean_energy + 2.0 * float(time)


@torch.no_grad()
def component_second_moment(
    images: torch.Tensor,
    analyzer: OrthogonalDirectionLoss,
    batch_size: int,
) -> torch.Tensor:
    total = torch.zeros(analyzer.dimension, device=images.device, dtype=torch.float64)
    seen = 0
    for batch in images.split(int(batch_size)):
        total += analyzer.transform(batch).double().square().sum(dim=0)
        seen += len(batch)
    return (total / max(seen, 1)).float()


@torch.no_grad()
def _predict(
    model: torch.nn.Module,
    state: torch.Tensor,
    time: float,
    batch_size: int,
) -> torch.Tensor:
    outputs = []
    for batch in state.split(int(batch_size)):
        times = torch.full((len(batch),), float(time), device=batch.device)
        outputs.append(model(batch, times))
    return torch.cat(outputs)


@torch.no_grad()
def rollout_snapshots(
    model: torch.nn.Module,
    initial: torch.Tensor,
    target_times: Sequence[float],
    *,
    ode_steps: int,
    batch_size: int,
) -> dict[float, torch.Tensor]:
    times = descending_time_grid(int(ode_steps), device=initial.device)
    requested = {float(value): int(torch.argmin((times - float(value)).abs())) for value in target_times}
    for value, index in requested.items():
        if not np.isclose(float(times[index]), value, atol=1e-6):
            raise ValueError(f"target time {value} is not aligned with the ODE grid")
    state = initial
    snapshots: dict[float, torch.Tensor] = {}
    for index, (current, following) in enumerate(zip(times[:-1], times[1:])):
        state = state + (following - current) * _predict(
            model, state, float(current), int(batch_size)
        )
        completed = index + 1
        for value, target_index in requested.items():
            if completed == target_index:
                snapshots[value] = state.detach().clone()
    if set(snapshots) != set(requested):
        raise RuntimeError("failed to collect every requested rollout state")
    return snapshots


def _load_study_config(study_dir: Path) -> BasisStudyConfig:
    values = json.loads((study_dir / "config.json").read_text(encoding="utf-8"))
    values["data_root"] = Path(values["data_root"])
    values["output_root"] = Path(values["output_root"])
    for name in ("bases", "seeds", "devices", "eval_times"):
        values[name] = tuple(values[name])
    return BasisStudyConfig(**values)


def _load_run(
    run_dir: Path,
    study_config: BasisStudyConfig,
    device: torch.device,
) -> tuple[dict[str, TinyVelocityUNet], OrthogonalDirectionLoss, dict[str, object]]:
    state = torch.load(run_dir / "state.pt", map_location="cpu", weights_only=True)
    analyzer_state = state["analyzer"]
    analyzer = OrthogonalDirectionLoss(
        analyzer_state["basis"],
        analyzer_state["component_moments"],
        analyzer_state["group_index"],
        gamma=study_config.gamma,
        damping=1e-4,
        min_weight=0.2,
        max_weight=2.0,
    ).to(device)
    models = {}
    for variant, model_state in state["models"].items():
        model = TinyVelocityUNet(study_config.width, study_config.depth).to(device)
        model.load_state_dict(model_state)
        model.eval()
        models[variant] = model
    return models, analyzer, state


def _random_directions(
    dimension: int,
    count: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(int(seed))
    directions = torch.randn((int(dimension), int(count)), generator=generator, device=device)
    return directions / torch.linalg.vector_norm(directions, dim=0, keepdim=True).clamp_min(1e-12)


def _normalized_rmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    scale = expected.square().mean().sqrt().clamp_min(1e-8)
    return float(((actual - expected).square().mean().sqrt() / scale).item())


@torch.no_grad()
def audit_run(
    run_dir: Path,
    study_config: BasisStudyConfig,
    config: BasisMechanismConfig,
    *,
    device_name: str,
) -> pd.DataFrame:
    device = torch.device(device_name if torch.cuda.is_available() or "cuda" not in device_name else "cpu")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    seed = int(summary["seed"])
    basis = str(summary["basis"])
    models, analyzer, _ = _load_run(run_dir, study_config, device)
    loaded = load_small_image_tensors(
        study_config.dataset,
        study_config.data_root,
        study_config.train_size,
        study_config.test_size,
        seed,
        download=False,
    )
    train = loaded["train"].to(device)
    test = loaded["test"][: config.audit_count].to(device)
    clean_energy = component_second_moment(train, analyzer, config.batch_size)
    del train

    rollout_generator = torch.Generator(device=device).manual_seed(seed + 307)
    initial = torch.randn(test.shape, generator=rollout_generator, device=device)
    teacher_generator = torch.Generator(device=device).manual_seed(seed + 911)
    teacher_noise = torch.randn(test.shape, generator=teacher_generator, device=device)
    trajectories = {
        variant: rollout_snapshots(
            model,
            initial,
            config.target_times,
            ode_steps=config.ode_steps,
            batch_size=config.batch_size,
        )
        for variant, model in models.items()
    }
    directions = _random_directions(
        test[0].numel(), config.projection_count, seed + 919, device
    )
    rows: list[dict[str, float | int | str]] = []

    for time in config.target_times:
        time = float(time)
        following = time - float(config.one_step)
        if following < 0.0:
            raise ValueError("one-step target time must be non-negative")
        teacher_state = (1.0 - time) * test + time * teacher_noise
        target_state = (1.0 - following) * test + following * teacher_noise
        target_energy = reference_component_energy(clean_energy, time)
        following_energy = reference_component_energy(clean_energy, following)
        target_drift = reference_component_drift(clean_energy, time)
        microscopic_target = teacher_noise - test
        contexts = {
            "teacher": teacher_state,
            "baseline_rollout": trajectories["baseline"][time],
            "weighted_rollout": trajectories["weighted"][time],
        }
        for context, state in contexts.items():
            state_coefficients = analyzer.transform(state)
            state_energy = state_coefficients.square().mean(dim=0)
            current_swd = sliced_wasserstein(
                teacher_state.flatten(1), state.flatten(1), directions
            )
            for variant, model in models.items():
                prediction = _predict(model, state, time, config.batch_size)
                prediction_coefficients = analyzer.transform(prediction)
                predicted_drift = 2.0 * torch.mean(
                    state_coefficients * prediction_coefficients, dim=0
                )
                candidate = state - float(config.one_step) * prediction
                candidate_energy = analyzer.transform(candidate).square().mean(dim=0)
                row: dict[str, float | int | str] = {
                    "basis": basis,
                    "seed": seed,
                    "time": time,
                    "context": context,
                    "variant": variant,
                    "state_energy_nrmse": _normalized_rmse(state_energy, target_energy),
                    "component_drift_nrmse": _normalized_rmse(predicted_drift, target_drift),
                    "current_pixel_swd": float(current_swd),
                    "one_step_pixel_swd": sliced_wasserstein(
                        target_state.flatten(1), candidate.flatten(1), directions
                    ),
                    "one_step_energy_nrmse": _normalized_rmse(
                        candidate_energy, following_energy
                    ),
                }
                if context == "teacher":
                    row["field_mse"] = float((prediction - microscopic_target).square().mean())
                else:
                    row["field_mse"] = float("nan")
                rows.append(row)

    for model in models.values():
        model.cpu()
    analyzer.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def summarize_mechanism(
    state_metrics: pd.DataFrame,
    endpoint_summary: pd.DataFrame,
) -> pd.DataFrame:
    paired = state_metrics.pivot_table(
        index=["basis", "seed", "time", "context"],
        columns="variant",
        values=["field_mse", "component_drift_nrmse", "one_step_pixel_swd", "one_step_energy_nrmse"],
    )
    paired.columns = [f"{metric}_{variant}" for metric, variant in paired.columns]
    paired = paired.reset_index()
    for metric in (
        "field_mse",
        "component_drift_nrmse",
        "one_step_pixel_swd",
        "one_step_energy_nrmse",
    ):
        paired[f"{metric}_ratio"] = paired[f"{metric}_weighted"] / paired[
            f"{metric}_baseline"
        ].clip(lower=1e-12)

    rows = []
    for (basis, seed), group in paired.groupby(["basis", "seed"]):
        teacher = group[group["context"].eq("teacher")]
        high_teacher = teacher[teacher["time"].ge(0.7)]
        middle_teacher = teacher[teacher["time"].between(0.3, 0.7)]
        middle_baseline_rollout = group[
            group["context"].eq("baseline_rollout")
            & group["time"].between(0.3, 0.7)
        ]
        middle_weighted_rollout = group[
            group["context"].eq("weighted_rollout")
            & group["time"].between(0.3, 0.7)
        ]
        endpoint = endpoint_summary[
            endpoint_summary["basis"].eq(basis) & endpoint_summary["seed"].eq(seed)
        ].iloc[0]
        rows.append(
            {
                "basis": basis,
                "seed": int(seed),
                "teacher_field_ratio_all": float(teacher["field_mse_ratio"].mean()),
                "high_teacher_drift_ratio": float(
                    high_teacher["component_drift_nrmse_ratio"].mean()
                ),
                "middle_teacher_one_step_ratio": float(
                    middle_teacher["one_step_pixel_swd_ratio"].mean()
                ),
                "middle_baseline_rollout_one_step_ratio": float(
                    middle_baseline_rollout["one_step_pixel_swd_ratio"].mean()
                ),
                "middle_weighted_rollout_one_step_ratio": float(
                    middle_weighted_rollout["one_step_pixel_swd_ratio"].mean()
                ),
                "middle_weighted_rollout_drift_ratio": float(
                    middle_weighted_rollout["component_drift_nrmse_ratio"].mean()
                ),
                "endpoint_feature_fid_ratio": float(endpoint["rollout_feature_fid_ratio"]),
                "endpoint_latent_swd_ratio": float(endpoint["rollout_latent_swd_ratio"]),
            }
        )
    return pd.DataFrame(rows)


def _audit_group(
    tasks: Sequence[tuple[Path, BasisStudyConfig, BasisMechanismConfig, str]],
) -> list[pd.DataFrame]:
    frames = []
    for run_dir, study_config, config, device in tasks:
        frame = audit_run(run_dir, study_config, config, device_name=device)
        frames.append(frame)
        print(f"audited {run_dir.name}")
    return frames


def run_mechanism_study(config: BasisMechanismConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    study_dir = config.study_dir.expanduser().resolve()
    study_config = _load_study_config(study_dir)
    endpoint_summary = pd.read_csv(study_dir / "study_summary.csv")
    run_dirs = [Path(value) for value in endpoint_summary["run_dir"]]
    devices = config.devices or ("cpu",)
    tasks = [
        (run_dir, study_config, config, devices[index % len(devices)])
        for index, run_dir in enumerate(run_dirs)
    ]
    grouped = [[] for _ in devices]
    for index, task in enumerate(tasks):
        grouped[index % len(devices)].append(task)
    frames: list[pd.DataFrame] = []
    if len(devices) == 1:
        frames = _audit_group(grouped[0])
    else:
        context = get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(devices), mp_context=context) as executor:
            futures = [executor.submit(_audit_group, group) for group in grouped if group]
            for future in as_completed(futures):
                frames.extend(future.result())
    state_metrics = pd.concat(frames, ignore_index=True)
    summary = summarize_mechanism(state_metrics, endpoint_summary)
    if config.save:
        state_metrics.to_csv(study_dir / "mechanism_state_metrics.csv", index=False)
        summary.to_csv(study_dir / "mechanism_summary.csv", index=False)
    return state_metrics, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3")
    parser.add_argument("--audit-count", type=int, default=512)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    devices = tuple(value.strip() for value in args.devices.split(",") if value.strip())
    config = BasisMechanismConfig(
        study_dir=args.study_dir,
        devices=devices or ("cpu",),
        audit_count=args.audit_count,
        save=not args.no_save,
    )
    _, summary = run_mechanism_study(config)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
