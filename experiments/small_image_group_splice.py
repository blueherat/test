"""Causal high-energy-subspace field splices on FashionMNIST."""

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

from experiments.mnist_spectral_rollout_toy import train_feature_classifier  # noqa: E402
from experiments.small_image_basis_mechanism import _load_run, _load_study_config  # noqa: E402
from experiments.small_image_basis_transport import (  # noqa: E402
    OrthogonalDirectionLoss,
    load_small_image_tensors,
)
from experiments.small_image_time_switch import _sample_metrics  # noqa: E402


@dataclass(frozen=True)
class GroupSpliceConfig:
    study_dir: Path
    devices: tuple[str, ...] = ("cuda:0", "cuda:1", "cuda:2", "cuda:3")
    sample_count: int = 1024
    batch_size: int = 128
    ode_steps_per_unit: int = 50
    high_threshold: float = 0.7
    middle_threshold: float = 0.3
    save: bool = True


def blend_velocity(
    baseline: torch.Tensor,
    weighted: torch.Tensor,
    analyzer: OrthogonalDirectionLoss,
    intervention: str,
) -> torch.Tensor:
    name = intervention.strip().lower()
    if name == "baseline":
        return baseline
    if name == "all":
        return weighted
    difference = analyzer.transform(weighted - baseline)
    groups = analyzer.group_index.to(difference.device)
    if name == "group0":
        mask = groups.eq(0)
    elif name == "nonzero":
        mask = groups.ne(0)
    else:
        raise ValueError(f"unknown group intervention: {intervention}")
    selected = difference * mask.to(difference.dtype)[None]
    image_difference = selected @ analyzer.basis.to(
        device=selected.device, dtype=selected.dtype
    ).T
    return baseline + image_difference.reshape_as(baseline)


def _active_window(context: str, time: float, config: GroupSpliceConfig) -> bool:
    if context == "high_rollout":
        return float(time) >= config.high_threshold - 1e-6
    if context == "middle_teacher":
        return (
            float(time) >= config.middle_threshold - 1e-6
            and float(time) < config.high_threshold - 1e-6
        )
    raise ValueError(f"unknown splice context: {context}")


@torch.no_grad()
def integrate_splice(
    models: Mapping[str, torch.nn.Module],
    analyzer: OrthogonalDirectionLoss,
    initial: torch.Tensor,
    start: float,
    *,
    context: str,
    intervention: str,
    config: GroupSpliceConfig,
) -> torch.Tensor:
    interval_count = int(round(float(start) * config.ode_steps_per_unit))
    times = torch.linspace(float(start), 0.0, interval_count + 1, device=initial.device)
    outputs = []
    for batch in initial.split(config.batch_size):
        state = batch
        for current, following in zip(times[:-1], times[1:]):
            time = torch.full((len(state),), float(current), device=state.device)
            baseline = models["baseline"](state, time)
            if _active_window(context, float(current), config) and intervention != "baseline":
                weighted = models["weighted"](state, time)
                velocity = blend_velocity(
                    baseline, weighted, analyzer, intervention
                )
            else:
                velocity = baseline
            state = state + (following - current) * velocity
        outputs.append(state)
    return torch.cat(outputs)


def audit_seed(
    seed: int,
    run_dirs: Mapping[str, Path],
    config: GroupSpliceConfig,
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
    rollout_generator = torch.Generator(device=device).manual_seed(int(seed) + 307)
    rollout_noise = torch.randn(test.shape, generator=rollout_generator, device=device)
    teacher_generator = torch.Generator(device=device).manual_seed(int(seed) + 1201)
    teacher_noise = torch.randn(test.shape, generator=teacher_generator, device=device)
    starts = {
        "high_rollout": (rollout_noise, 1.0),
        "middle_teacher": (
            (1.0 - config.high_threshold) * test
            + config.high_threshold * teacher_noise,
            config.high_threshold,
        ),
    }
    rows: list[dict[str, float | int | str]] = []

    for basis, run_dir in run_dirs.items():
        expected_accuracy = float(
            json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))[
                "classifier_accuracy"
            ]
        )
        if abs(float(accuracy) - expected_accuracy) > 0.01:
            raise RuntimeError("audit classifier accuracy differs by more than one point")
        models, analyzer, _ = _load_run(run_dir, study_config, device)
        for context, (initial, start_time) in starts.items():
            for intervention in ("baseline", "all", "group0", "nonzero"):
                generated = integrate_splice(
                    models,
                    analyzer,
                    initial,
                    start_time,
                    context=context,
                    intervention=intervention,
                    config=config,
                )
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
                            "context": context,
                            "intervention": intervention,
                            "metric": metric,
                            "value": float(value),
                            "audit_classifier_accuracy": float(accuracy),
                        }
                    )
        for model in models.values():
            model.cpu()
        analyzer.cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def summarize_splice(metrics: pd.DataFrame) -> pd.DataFrame:
    baseline = metrics[metrics["intervention"].eq("baseline")][
        ["basis", "seed", "context", "metric", "value"]
    ].rename(columns={"value": "baseline_value"})
    paired = metrics.merge(
        baseline,
        on=["basis", "seed", "context", "metric"],
        validate="many_to_one",
    )
    paired["ratio"] = paired["value"] / paired["baseline_value"].clip(lower=1e-12)
    full = paired[paired["intervention"].eq("all")][
        ["basis", "seed", "context", "metric", "value", "baseline_value"]
    ].rename(columns={"value": "full_value"})
    paired = paired.merge(
        full,
        on=["basis", "seed", "context", "metric", "baseline_value"],
        validate="many_to_one",
    )
    denominator = paired["full_value"] - paired["baseline_value"]
    paired["fraction_of_full_delta"] = (
        (paired["value"] - paired["baseline_value"])
        / denominator.where(denominator.abs() > 1e-12)
    )
    return paired


def _seed_group(
    tasks: Sequence[tuple[int, dict[str, Path], GroupSpliceConfig, str]],
) -> list[pd.DataFrame]:
    frames = []
    for seed, run_dirs, config, device in tasks:
        frames.append(audit_seed(seed, run_dirs, config, device_name=device))
        print(f"audited group splice seed={seed}")
    return frames


def run_group_splice_study(
    config: GroupSpliceConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    endpoint = pd.read_csv(config.study_dir / "study_summary.csv")
    devices = config.devices or ("cpu",)
    tasks = []
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
    paired = summarize_splice(metrics)
    if config.save:
        metrics.to_csv(config.study_dir / "group_splice_metrics.csv", index=False)
        paired.to_csv(config.study_dir / "group_splice_paired.csv", index=False)
    return metrics, paired


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3")
    parser.add_argument("--sample-count", type=int, default=1024)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    devices = tuple(value.strip() for value in args.devices.split(",") if value.strip())
    config = GroupSpliceConfig(
        study_dir=args.study_dir.expanduser().resolve(),
        devices=devices or ("cpu",),
        sample_count=args.sample_count,
        save=not args.no_save,
    )
    _, paired = run_group_splice_study(config)
    selected = paired[paired["metric"].eq("feature_fid")]
    print(
        selected.groupby(["basis", "context", "intervention"])[
            ["ratio", "fraction_of_full_delta"]
        ]
        .agg(["mean", "std", "min", "max"])
        .round(4)
        .to_string()
    )


if __name__ == "__main__":
    main()
