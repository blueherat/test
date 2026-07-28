"""Dense-time teacher-field and transport-moment audit."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.small_image_basis_mechanism import (  # noqa: E402
    _load_run,
    _load_study_config,
    _normalized_rmse,
    _predict,
    component_second_moment,
    reference_component_drift,
)
from experiments.small_image_basis_transport import load_small_image_tensors  # noqa: E402


@dataclass(frozen=True)
class DenseTeacherConfig:
    study_dir: Path
    devices: tuple[str, ...] = ("cuda:0", "cuda:1", "cuda:2", "cuda:3")
    audit_count: int = 1024
    batch_size: int = 128
    time_steps: int = 50
    save: bool = True


@torch.no_grad()
def audit_run(
    run_dir: Path,
    config: DenseTeacherConfig,
    *,
    device_name: str,
) -> pd.DataFrame:
    study_config = _load_study_config(config.study_dir)
    device = torch.device(
        device_name if torch.cuda.is_available() or "cuda" not in device_name else "cpu"
    )
    summary = pd.read_json(run_dir / "summary.json", typ="series")
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
    generator = torch.Generator(device=device).manual_seed(seed + 211)
    noise = torch.randn(test.shape, generator=generator, device=device)
    target = noise - test
    times = torch.arange(1, int(config.time_steps), device=device).float() / int(
        config.time_steps
    )
    rows: list[dict[str, float | int | str]] = []
    for time_tensor in times:
        time = float(time_tensor)
        state = (1.0 - time) * test + time * noise
        state_coefficients = analyzer.transform(state)
        reference_drift = reference_component_drift(clean_energy, time)
        for variant, model in models.items():
            prediction = _predict(model, state, time, config.batch_size)
            prediction_coefficients = analyzer.transform(prediction)
            predicted_drift = 2.0 * torch.mean(
                state_coefficients * prediction_coefficients, dim=0
            )
            rows.append(
                {
                    "basis": basis,
                    "seed": seed,
                    "time": time,
                    "variant": variant,
                    "field_mse": float((prediction - target).square().mean()),
                    "component_drift_nrmse": _normalized_rmse(
                        predicted_drift, reference_drift
                    ),
                }
            )
    for model in models.values():
        model.cpu()
    analyzer.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def summarize_dense(
    metrics: pd.DataFrame,
    restart_paired: pd.DataFrame,
) -> pd.DataFrame:
    paired = metrics.pivot_table(
        index=["basis", "seed", "time"],
        columns="variant",
        values=["field_mse", "component_drift_nrmse"],
    )
    paired.columns = [f"{metric}_{variant}" for metric, variant in paired.columns]
    paired = paired.reset_index()
    paired["field_mse_ratio"] = paired["field_mse_weighted"] / paired[
        "field_mse_baseline"
    ].clip(lower=1e-12)
    paired["component_drift_ratio"] = paired[
        "component_drift_nrmse_weighted"
    ] / paired["component_drift_nrmse_baseline"].clip(lower=1e-12)
    rows = []
    for (basis, seed), group in paired.groupby(["basis", "seed"]):
        windows = {
            "low": group[group["time"].lt(0.3)],
            "middle": group[group["time"].ge(0.3) & group["time"].lt(0.7)],
            "high": group[group["time"].ge(0.7)],
            "all": group,
        }
        row: dict[str, float | int | str] = {"basis": basis, "seed": int(seed)}
        for name, window in windows.items():
            row[f"{name}_field_mse_ratio_of_means"] = float(
                window["field_mse_weighted"].mean()
                / max(window["field_mse_baseline"].mean(), 1e-12)
            )
            row[f"{name}_mean_component_drift_ratio"] = float(
                window["component_drift_ratio"].mean()
            )
            row[f"{name}_field_improved_fraction"] = float(
                window["field_mse_ratio"].lt(1.0).mean()
            )
        restart = restart_paired[
            restart_paired["basis"].eq(basis)
            & restart_paired["seed"].eq(seed)
            & restart_paired["start_context"].eq("teacher_restart")
            & restart_paired["schedule"].eq("middle")
            & restart_paired["metric"].eq("feature_fid")
        ]
        row["teacher_restart_middle_fid_ratio"] = float(restart.iloc[0]["ratio"])
        rows.append(row)
    return pd.DataFrame(rows)


def _audit_group(
    tasks: Sequence[tuple[Path, DenseTeacherConfig, str]],
) -> list[pd.DataFrame]:
    frames = []
    for run_dir, config, device in tasks:
        frames.append(audit_run(run_dir, config, device_name=device))
        print(f"audited dense teacher {run_dir.name}")
    return frames


def run_dense_teacher_study(
    config: DenseTeacherConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    endpoint = pd.read_csv(config.study_dir / "study_summary.csv")
    restart = pd.read_csv(config.study_dir / "teacher_restart_paired.csv")
    devices = config.devices or ("cpu",)
    tasks = [
        (Path(row.run_dir), config, devices[index % len(devices)])
        for index, row in enumerate(endpoint.itertuples())
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
    metrics = pd.concat(frames, ignore_index=True)
    summary = summarize_dense(metrics, restart)
    if config.save:
        metrics.to_csv(config.study_dir / "dense_teacher_metrics.csv", index=False)
        summary.to_csv(config.study_dir / "dense_teacher_summary.csv", index=False)
    return metrics, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3")
    parser.add_argument("--audit-count", type=int, default=1024)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    devices = tuple(value.strip() for value in args.devices.split(",") if value.strip())
    config = DenseTeacherConfig(
        study_dir=args.study_dir.expanduser().resolve(),
        devices=devices or ("cpu",),
        audit_count=args.audit_count,
        save=not args.no_save,
    )
    _, summary = run_dense_teacher_study(config)
    columns = [
        "basis",
        "seed",
        "middle_field_mse_ratio_of_means",
        "middle_mean_component_drift_ratio",
        "middle_field_improved_fraction",
        "teacher_restart_middle_fid_ratio",
    ]
    print(summary[columns].to_string(index=False))


if __name__ == "__main__":
    main()
