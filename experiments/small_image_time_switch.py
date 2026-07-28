"""Causal time-window field switches for the small-image basis study."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mnist_spectral_rollout_toy import (  # noqa: E402
    descending_time_grid,
    frechet_distance,
    sliced_wasserstein,
    train_feature_classifier,
)
from experiments.small_image_basis_mechanism import (  # noqa: E402
    _load_run,
    _load_study_config,
)
from experiments.small_image_basis_transport import load_small_image_tensors  # noqa: E402


@dataclass(frozen=True)
class TimeSwitchConfig:
    study_dir: Path
    devices: tuple[str, ...] = ("cuda:0", "cuda:1", "cuda:2", "cuda:3")
    sample_count: int = 1024
    batch_size: int = 128
    ode_steps: int = 50
    high_threshold: float = 0.7
    middle_threshold: float = 0.3
    save: bool = True


def schedule_variant(schedule: str, time: float, config: TimeSwitchConfig) -> str:
    name = schedule.strip().lower()
    high = float(time) >= config.high_threshold - 1e-6
    middle = (
        float(time) >= config.middle_threshold - 1e-6
        and float(time) < config.high_threshold - 1e-6
    )
    low = float(time) < config.middle_threshold - 1e-6
    if name == "baseline":
        return "baseline"
    if name == "weighted":
        return "weighted"
    if name == "high":
        return "weighted" if high else "baseline"
    if name == "middle":
        return "weighted" if middle else "baseline"
    if name == "low":
        return "weighted" if low else "baseline"
    if name == "high_middle":
        return "weighted" if not low else "baseline"
    raise ValueError(f"unknown time-switch schedule: {schedule}")


@torch.no_grad()
def time_switch_sample(
    models: Mapping[str, torch.nn.Module],
    initial: torch.Tensor,
    schedule: str,
    config: TimeSwitchConfig,
) -> torch.Tensor:
    times = descending_time_grid(config.ode_steps, device=initial.device)
    outputs = []
    for batch in initial.split(config.batch_size):
        state = batch
        for current, following in zip(times[:-1], times[1:]):
            variant = schedule_variant(schedule, float(current), config)
            time = torch.full((len(state),), float(current), device=state.device)
            state = state + (following - current) * models[variant](state, time)
        outputs.append(state)
    return torch.cat(outputs)


def _random_directions(
    dimension: int,
    count: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(int(seed))
    value = torch.randn((int(dimension), int(count)), generator=generator, device=device)
    return value / torch.linalg.vector_norm(value, dim=0, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def _sample_metrics(
    reference: torch.Tensor,
    generated: torch.Tensor,
    classifier: torch.nn.Module,
    normalization: Mapping[str, float],
    *,
    seed: int,
) -> dict[str, float]:
    mean = float(normalization["mean"])
    std = float(normalization["std"])
    reference_pixels = (reference * std + mean).clamp(0.0, 1.0)
    generated_pixels = (generated * std + mean).clamp(0.0, 1.0)
    reference_decoded = (reference_pixels - mean) / std
    generated_decoded = (generated_pixels - mean) / std
    _, reference_features = classifier(reference_decoded, return_features=True)
    _, generated_features = classifier(generated_decoded, return_features=True)
    return {
        "latent_swd": sliced_wasserstein(
            reference.flatten(1),
            generated.flatten(1),
            _random_directions(28 * 28, 64, seed + 401, reference.device),
        ),
        "decoded_pixel_swd": sliced_wasserstein(
            reference_pixels.flatten(1),
            generated_pixels.flatten(1),
            _random_directions(28 * 28, 64, seed + 403, reference.device),
        ),
        "feature_swd": sliced_wasserstein(
            reference_features,
            generated_features,
            _random_directions(64, 64, seed + 409, reference.device),
        ),
        "feature_fid": frechet_distance(reference_features, generated_features),
    }


def audit_seed(
    seed: int,
    run_dirs: Mapping[str, Path],
    config: TimeSwitchConfig,
    *,
    device_name: str,
) -> pd.DataFrame:
    study_config = _load_study_config(config.study_dir)
    device = torch.device(
        device_name if torch.cuda.is_available() or "cuda" not in device_name else "cpu"
    )
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    loaded = load_small_image_tensors(
        study_config.dataset,
        study_config.data_root,
        study_config.train_size,
        study_config.test_size,
        int(seed),
        download=False,
    )
    train = loaded["train"].to(device)
    test = loaded["test"][: config.sample_count].to(device)
    classifier, accuracy = train_feature_classifier(
        train,
        loaded["train_labels"].to(device),
        test,
        loaded["test_labels"][: config.sample_count].to(device),
        epochs=study_config.classifier_epochs,
        batch_size=study_config.classifier_batch_size,
        seed=int(seed),
    )
    del train
    generator = torch.Generator(device=device).manual_seed(int(seed) + 307)
    initial = torch.randn(test.shape, generator=generator, device=device)
    schedules = ("baseline", "weighted", "high", "middle", "low", "high_middle")
    rows: list[dict[str, float | int | str]] = []
    for basis, run_dir in run_dirs.items():
        expected_accuracy = float(
            json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))[
                "classifier_accuracy"
            ]
        )
        if abs(float(accuracy) - expected_accuracy) > 0.01:
            raise RuntimeError(
                f"classifier reproduction failed for seed {seed}: {accuracy} != {expected_accuracy}"
            )
        models, analyzer, _ = _load_run(run_dir, study_config, device)
        analyzer.cpu()
        for schedule in schedules:
            generated = time_switch_sample(models, initial, schedule, config)
            metrics = _sample_metrics(
                test,
                generated,
                classifier,
                loaded["normalization"],
                seed=int(seed),
            )
            for metric, value in metrics.items():
                rows.append(
                    {
                        "basis": basis,
                        "seed": int(seed),
                        "schedule": schedule,
                        "metric": metric,
                        "value": float(value),
                        "audit_classifier_accuracy": float(accuracy),
                        "original_classifier_accuracy": expected_accuracy,
                    }
                )
        for model in models.values():
            model.cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def summarize_switch(metrics: pd.DataFrame) -> pd.DataFrame:
    baseline = metrics[metrics["schedule"].eq("baseline")][
        ["basis", "seed", "metric", "value"]
    ].rename(columns={"value": "baseline_value"})
    paired = metrics.merge(baseline, on=["basis", "seed", "metric"], validate="many_to_one")
    paired["ratio"] = paired["value"] / paired["baseline_value"].clip(lower=1e-12)
    return paired


def _seed_group(
    tasks: Sequence[tuple[int, dict[str, Path], TimeSwitchConfig, str]],
) -> list[pd.DataFrame]:
    frames = []
    for seed, run_dirs, config, device in tasks:
        frames.append(audit_seed(seed, run_dirs, config, device_name=device))
        print(f"audited switch seed={seed}")
    return frames


def run_time_switch_study(config: TimeSwitchConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    study_dir = config.study_dir.expanduser().resolve()
    endpoint = pd.read_csv(study_dir / "study_summary.csv")
    tasks = []
    devices = config.devices or ("cpu",)
    for index, (seed, group) in enumerate(endpoint.groupby("seed")):
        run_dirs = {row.basis: Path(row.run_dir) for row in group.itertuples()}
        tasks.append((int(seed), run_dirs, config, devices[index % len(devices)]))
    grouped = [[] for _ in devices]
    for index, task in enumerate(tasks):
        grouped[index % len(devices)].append(task)
    frames: list[pd.DataFrame] = []
    if len(devices) == 1:
        frames = _seed_group(grouped[0])
    else:
        context = get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(devices), mp_context=context) as executor:
            futures = [executor.submit(_seed_group, group) for group in grouped if group]
            for future in as_completed(futures):
                frames.extend(future.result())
    metrics = pd.concat(frames, ignore_index=True)
    paired = summarize_switch(metrics)
    if config.save:
        metrics.to_csv(study_dir / "time_switch_metrics.csv", index=False)
        paired.to_csv(study_dir / "time_switch_paired.csv", index=False)
    return metrics, paired


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3")
    parser.add_argument("--sample-count", type=int, default=1024)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    devices = tuple(value.strip() for value in args.devices.split(",") if value.strip())
    config = TimeSwitchConfig(
        study_dir=args.study_dir,
        devices=devices or ("cpu",),
        sample_count=args.sample_count,
        save=not args.no_save,
    )
    _, paired = run_time_switch_study(config)
    print(
        paired.groupby(["basis", "schedule", "metric"])["ratio"]
        .agg(["mean", "std", "min", "max"])
        .round(4)
        .to_string()
    )


if __name__ == "__main__":
    main()
