"""Test whether time-sample order, not its marginal distribution, flips quality.

All conditions keep data, initialization, minibatch draws, Gaussian bridge
noise, optimizer, and evaluation fixed.  They differ only in the time schedule:
the original seed-3/seed-4 streams, a step permutation of the seed-3 stream
that preserves every sampled time exactly, and per-batch stratified sampling.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mnist_spectral_rollout_toy import (  # noqa: E402
    MNISTToyConfig,
    TinyVelocityUNet,
    configure_fp32,
    evaluate_rollouts,
    shifted_uniform,
    train_feature_classifier,
)
from experiments.small_image_basis_transport import (  # noqa: E402
    build_direction_analyzer,
    load_small_image_tensors,
)
from experiments.small_image_seed_factorial import (  # noqa: E402
    _model_hash,
    _paired_metric_rows,
    _toy_config,
)
from experiments.small_image_time_sequence_audit import replay_time_draws  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path.home() / "data/eqvae/experiments/small_image_time_order_study"
SCHEDULES = ("iid_seed3", "iid_seed4", "step_permuted_seed3", "stratified_seed3")


def _is_step_permutation_name(name: str) -> bool:
    if not name.startswith("step_permuted_seed") or "_perm" not in name:
        return False
    source, permutation = name.removeprefix("step_permuted_seed").split(
        "_perm", maxsplit=1
    )
    return (
        bool(source)
        and source.lstrip("-").isdigit()
        and bool(permutation)
        and permutation.lstrip("-").isdigit()
    )


@dataclass(frozen=True)
class TimeOrderStudyConfig:
    dataset: str = "mnist"
    data_root: Path = Path("/data/shared/mnist")
    output_root: Path = DEFAULT_OUTPUT_ROOT
    data_seed: int = 4
    init_seed: int = 4
    batch_noise_seed: int = 4
    permutation_seed: int = 90_003
    stratified_seed: int = 3
    schedules: tuple[str, ...] = SCHEDULES
    rollout_seeds: tuple[int, ...] = (1701, 1702, 1703)
    devices: tuple[str, ...] = ("cuda:0", "cuda:1", "cuda:2", "cuda:3")
    train_size: int = 8_192
    test_size: int = 2_048
    sample_count: int = 2_048
    batch_size: int = 128
    steps: int = 1_000
    learning_rate: float = 2e-4
    width: int = 24
    depth: int = 2
    gamma: float = 0.5
    band_count: int = 8
    time_shift: float = 1.0
    ode_steps: int = 50
    classifier_epochs: int = 3
    classifier_batch_size: int = 256
    save: bool = True


def _shift_raw_time(raw: torch.Tensor, shift: float) -> torch.Tensor:
    return float(shift) * raw / (1.0 + (float(shift) - 1.0) * raw)


@torch.no_grad()
def build_time_schedule(
    name: str,
    config: TimeOrderStudyConfig,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    """Build a [steps, batch] schedule on the requested training device."""

    device = torch.device(device)
    replay_args = {
        "train_size": config.train_size,
        "batch_size": config.batch_size,
        "steps": config.steps,
        "spatial_shape": (1, 28, 28),
        "time_shift": config.time_shift,
        "device": device,
    }
    if name.startswith("iid_seed"):
        suffix = name.removeprefix("iid_seed")
        if suffix and suffix.lstrip("-").isdigit():
            return replay_time_draws(seed=int(suffix), **replay_args).to(device)
    if name == "step_permuted_seed3":
        source = replay_time_draws(seed=3, **replay_args)
        permutation = torch.randperm(
            config.steps,
            generator=torch.Generator().manual_seed(config.permutation_seed),
        )
        return source[permutation].to(device)
    if name.startswith("step_permuted_seed") and "_perm" in name:
        source_text, permutation_text = name.removeprefix(
            "step_permuted_seed"
        ).split("_perm", maxsplit=1)
        if not source_text.lstrip("-").isdigit() or not permutation_text.lstrip(
            "-"
        ).isdigit():
            raise ValueError(f"invalid step-permuted schedule: {name}")
        source_seed = int(source_text)
        permutation_seed = int(permutation_text)
        source = replay_time_draws(seed=source_seed, **replay_args)
        permutation = torch.randperm(
            config.steps,
            generator=torch.Generator().manual_seed(90_000 + permutation_seed),
        )
        return source[permutation].to(device)
    if name.startswith("stratified_seed"):
        suffix = name.removeprefix("stratified_seed")
        if not suffix or not suffix.lstrip("-").isdigit():
            raise ValueError(f"invalid stratified schedule: {name}")
        schedule_seed = int(suffix)
        generator = torch.Generator(device=device).manual_seed(
            schedule_seed + 70_001
        )
        base = torch.arange(config.batch_size, device=device, dtype=torch.float32)
        rows = []
        for _ in range(config.steps):
            raw = (base + torch.rand(
                config.batch_size, device=device, generator=generator
            )) / config.batch_size
            permutation = torch.randperm(
                config.batch_size, device=device, generator=generator
            )
            rows.append(_shift_raw_time(raw[permutation], config.time_shift))
        return torch.stack(rows)
    raise ValueError(f"unknown time schedule: {name}")


def _tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()


def train_paired_with_time_schedule(
    clean: torch.Tensor,
    config: MNISTToyConfig,
    analyzer: torch.nn.Module,
    time_schedule: torch.Tensor,
    *,
    init_seed: int,
    batch_noise_seed: int,
) -> dict[str, TinyVelocityUNet]:
    """Train paired models while replacing only the selected time draw."""

    if time_schedule.shape != (int(config.steps), int(config.batch_size)):
        raise ValueError("time_schedule must have shape [steps, batch_size]")
    device = clean.device
    time_schedule = time_schedule.to(device=device, dtype=clean.dtype)
    torch.manual_seed(int(init_seed))
    if device.type == "cuda":
        torch.cuda.manual_seed(int(init_seed))
    baseline = TinyVelocityUNet(config.width, config.depth).to(device)
    weighted = copy.deepcopy(baseline)
    models = {"baseline": baseline, "weighted": weighted}
    optimizers = {
        name: torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=1e-4
        )
        for name, model in models.items()
    }
    generator = torch.Generator(device=device).manual_seed(int(batch_noise_seed) + 101)
    for step in range(int(config.steps)):
        indices = torch.randint(
            len(clean),
            (int(config.batch_size),),
            device=device,
            generator=generator,
        )
        noise = torch.randn(
            (int(config.batch_size), *clean.shape[1:]),
            device=device,
            generator=generator,
        )
        # Preserve the source stream exactly even when its time draw is replaced.
        shifted_uniform(
            config.batch_size,
            config.time_shift,
            device=device,
            generator=generator,
        )
        data = clean[indices]
        time = time_schedule[step]
        expanded = time[:, None, None, None]
        state = (1.0 - expanded) * data + expanded * noise
        target = noise - data
        for name, model in models.items():
            prediction = model(state, time)
            loss = (
                F.mse_loss(prediction, target)
                if name == "baseline"
                else analyzer(prediction, target, time)[0].mean()
            )
            optimizers[name].zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizers[name].step()
    return models


def summarize_conditions(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["schedule", "metric"], as_index=False)
        .agg(
            rollout_seeds=("rollout_seed", "nunique"),
            baseline_mean=("baseline", "mean"),
            weighted_mean=("weighted", "mean"),
            ratio_mean=("ratio", "mean"),
            ratio_std=("ratio", "std"),
            ratio_min=("ratio", "min"),
            ratio_max=("ratio", "max"),
            harm_rate=("ratio", lambda values: float((values > 1.0).mean())),
        )
        .sort_values(["metric", "schedule"])
        .reset_index(drop=True)
    )


def _run_condition(
    config: TimeOrderStudyConfig,
    schedule_name: str,
    device_name: str,
) -> dict[str, object]:
    configure_fp32(config.init_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device(
        device_name
        if not device_name.startswith("cuda") or torch.cuda.is_available()
        else "cpu"
    )
    loaded = load_small_image_tensors(
        config.dataset,
        config.data_root,
        config.train_size,
        config.test_size,
        config.data_seed,
        download=False,
    )
    train = loaded["train"].to(device)
    test = loaded["test"].to(device)
    analyzer, _ = build_direction_analyzer(
        train,
        "dct",
        band_count=config.band_count,
        gamma=config.gamma,
        seed=config.data_seed,
    )
    analyzer = analyzer.to(device)
    toy_config = _toy_config(config, seed=config.batch_noise_seed, device=str(device))
    time_schedule = build_time_schedule(schedule_name, config, device=device)
    models = train_paired_with_time_schedule(
        train,
        toy_config,
        analyzer,
        time_schedule,
        init_seed=config.init_seed,
        batch_noise_seed=config.batch_noise_seed,
    )
    for model in models.values():
        model.eval()
    classifier, classifier_accuracy = train_feature_classifier(
        train,
        loaded["train_labels"].to(device),
        test,
        loaded["test_labels"].to(device),
        epochs=config.classifier_epochs,
        batch_size=config.classifier_batch_size,
        seed=config.data_seed,
    )
    rows = []
    for rollout_seed in config.rollout_seeds:
        evaluation_config = replace(toy_config, seed=int(rollout_seed))
        rollout, _, _ = evaluate_rollouts(
            models,
            test,
            classifier,
            analyzer,
            evaluation_config,
            loaded["normalization"],
        )
        for row in _paired_metric_rows(
            rollout,
            data_seed=config.data_seed,
            init_seed=config.init_seed,
            stream_seed=config.batch_noise_seed,
            rollout_seed=rollout_seed,
        ):
            row["schedule"] = schedule_name
            rows.append(row)
    hashes = {name: _model_hash(model) for name, model in models.items()}
    schedule_audit = {
        "hash": _tensor_hash(time_schedule),
        "sorted_multiset_hash": _tensor_hash(
            time_schedule.flatten().sort().values
        ),
        "mean": float(time_schedule.mean()),
        "std": float(time_schedule.std(unbiased=False)),
        "min": float(time_schedule.min()),
        "max": float(time_schedule.max()),
    }
    del train, test, classifier, analyzer, models, time_schedule
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "schedule": schedule_name,
        "classifier_accuracy": float(classifier_accuracy),
        "hashes": hashes,
        "schedule_audit": schedule_audit,
        "metric_rows": rows,
    }


def _run_group(
    tasks: Sequence[tuple[TimeOrderStudyConfig, str, str]],
) -> list[dict[str, object]]:
    results = []
    for config, schedule, device in tasks:
        result = _run_condition(config, schedule, device)
        ratios = [
            float(row["ratio"])
            for row in result["metric_rows"]
            if row["metric"] == "feature_fid"
        ]
        print(f"done schedule={schedule}: fid_ratio={sum(ratios) / len(ratios):.3f}")
        results.append(result)
    return results


def run_study(
    config: TimeOrderStudyConfig = TimeOrderStudyConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, Path | None]:
    unknown = sorted(
        name
        for name in set(config.schedules)
        if name not in SCHEDULES
        and not (
            name.startswith("iid_seed")
            and name.removeprefix("iid_seed").lstrip("-").isdigit()
        )
        and not (
            name.startswith("stratified_seed")
            and name.removeprefix("stratified_seed").lstrip("-").isdigit()
        )
        and not _is_step_permutation_name(name)
    )
    if unknown:
        raise ValueError(f"unknown schedules: {unknown}")
    load_small_image_tensors(
        config.dataset, config.data_root, 1, 1, 0, download=True
    )
    devices = config.devices or ("cpu",)
    grouped: list[list[tuple[TimeOrderStudyConfig, str, str]]] = [
        [] for _ in devices
    ]
    for index, schedule in enumerate(config.schedules):
        device = devices[index % len(devices)]
        grouped[index % len(devices)].append((config, schedule, device))
    results = []
    if len(devices) == 1:
        results = _run_group(grouped[0])
    else:
        context = get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=len(devices), mp_context=context
        ) as executor:
            futures = [
                executor.submit(_run_group, group) for group in grouped if group
            ]
            for future in as_completed(futures):
                results.extend(future.result())
    metrics = pd.DataFrame(
        row for result in results for row in result["metric_rows"]
    )
    conditions = summarize_conditions(metrics)
    result_dir = None
    if config.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = config.output_root.expanduser() / f"study_{timestamp}"
        result_dir.mkdir(parents=True, exist_ok=False)
        serialized = asdict(config)
        serialized["data_root"] = str(config.data_root)
        serialized["output_root"] = str(config.output_root.expanduser())
        (result_dir / "config.json").write_text(
            json.dumps(serialized, indent=2), encoding="utf-8"
        )
        metrics.to_csv(result_dir / "rollout_metrics.csv", index=False)
        conditions.to_csv(result_dir / "condition_summary.csv", index=False)
        integrity = [
            {
                "schedule": result["schedule"],
                "classifier_accuracy": result["classifier_accuracy"],
                "hashes": result["hashes"],
                "schedule_audit": result["schedule_audit"],
            }
            for result in results
        ]
        (result_dir / "integrity.json").write_text(
            json.dumps(integrity, indent=2), encoding="utf-8"
        )
    return metrics, conditions, result_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    devices = tuple(value.strip() for value in args.devices.split(",") if value.strip())
    config = TimeOrderStudyConfig(devices=devices or ("cpu",), save=not args.no_save)
    if args.quick:
        config = replace(
            config,
            devices=(devices[0],) if devices else ("cpu",),
            train_size=128,
            test_size=64,
            sample_count=64,
            batch_size=16,
            steps=4,
            width=8,
            depth=1,
            ode_steps=5,
            classifier_epochs=1,
            classifier_batch_size=32,
            rollout_seeds=(1701,),
        )
    _, conditions, result_dir = run_study(config)
    print(f"result_dir={result_dir}")
    print(conditions[conditions["metric"].eq("feature_fid")].to_string(index=False))


if __name__ == "__main__":
    main()
