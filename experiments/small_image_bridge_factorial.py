"""Split stochastic-interpolant randomness into Gaussian noise and time draws."""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mnist_spectral_rollout_toy import (  # noqa: E402
    configure_fp32,
    evaluate_rollouts,
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
from experiments.small_image_stream_factorial import (  # noqa: E402
    train_paired_mixed_streams,
)


DEFAULT_OUTPUT_ROOT = Path.home() / "data/eqvae/experiments/small_image_bridge_factorial"


@dataclass(frozen=True)
class BridgeFactorialConfig:
    data_root: Path = Path("/data/shared/mnist")
    output_root: Path = DEFAULT_OUTPUT_ROOT
    data_seed: int = 4
    init_seed: int = 4
    batch_seed: int = 4
    levels: tuple[int, int] = (3, 4)
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
    ode_steps: int = 50
    classifier_epochs: int = 3
    classifier_batch_size: int = 256
    save: bool = True


def summarize_bridge_conditions(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["noise_seed", "time_seed", "metric"], as_index=False)
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
        .sort_values(["metric", "noise_seed", "time_seed"])
        .reset_index(drop=True)
    )


def bridge_factorial_effects(
    conditions: pd.DataFrame,
    *,
    metric: str = "feature_fid",
) -> pd.DataFrame:
    selected = conditions[conditions["metric"].eq(metric)].copy()
    if len(selected) != 4:
        raise ValueError("bridge factorial requires all four 2x2 conditions")
    encodings = {}
    for factor in ("noise_seed", "time_seed"):
        levels = sorted(selected[factor].unique().tolist())
        if len(levels) != 2:
            raise ValueError(f"{factor} must have two levels")
        encodings[factor] = selected[factor].map(
            {levels[0]: -1.0, levels[1]: 1.0}
        )
    response = selected["ratio_mean"].clip(lower=1e-12).map(math.log)
    rows = []
    for name, factors in (
        ("noise", ("noise_seed",)),
        ("time", ("time_seed",)),
        ("noise:time", ("noise_seed", "time_seed")),
    ):
        contrast = pd.Series(1.0, index=selected.index)
        for factor in factors:
            contrast *= encodings[factor]
        log_effect = 2.0 * float((response * contrast).mean())
        rows.append(
            {
                "term": name,
                "log_ratio_effect": log_effect,
                "multiplicative_effect": math.exp(log_effect),
                "absolute_log_effect": abs(log_effect),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "absolute_log_effect", ascending=False
    ).reset_index(drop=True)


def _run_condition(
    config: BridgeFactorialConfig,
    noise_seed: int,
    time_seed: int,
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
        "mnist",
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
    toy_config = _toy_config(config, seed=time_seed, device=str(device))
    models = train_paired_mixed_streams(
        train,
        toy_config,
        analyzer,
        init_seed=config.init_seed,
        batch_seed=config.batch_seed,
        noise_seed=noise_seed,
        time_seed=time_seed,
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
            stream_seed=time_seed,
            rollout_seed=rollout_seed,
        ):
            row["noise_seed"] = int(noise_seed)
            row["time_seed"] = int(time_seed)
            rows.append(row)
    hashes = {name: _model_hash(model) for name, model in models.items()}
    del train, test, classifier, analyzer, models
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "noise_seed": int(noise_seed),
        "time_seed": int(time_seed),
        "classifier_accuracy": float(classifier_accuracy),
        "hashes": hashes,
        "metric_rows": rows,
    }


def _run_group(
    tasks: Sequence[tuple[BridgeFactorialConfig, int, int, str]],
) -> list[dict[str, object]]:
    results = []
    for config, noise_seed, time_seed, device in tasks:
        result = _run_condition(config, noise_seed, time_seed, device)
        fid_rows = [
            row for row in result["metric_rows"] if row["metric"] == "feature_fid"
        ]
        ratio = sum(float(row["ratio"]) for row in fid_rows) / len(fid_rows)
        print(
            f"done noise={noise_seed} time={time_seed}: fid_ratio={ratio:.3f}"
        )
        results.append(result)
    return results


def run_study(
    config: BridgeFactorialConfig = BridgeFactorialConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path | None]:
    if len(set(config.levels)) != 2:
        raise ValueError("levels must contain exactly two distinct seeds")
    load_small_image_tensors(
        "mnist", config.data_root, 1, 1, 0, download=True
    )
    combinations = [
        (noise_seed, time_seed)
        for noise_seed in config.levels
        for time_seed in config.levels
    ]
    devices = config.devices or ("cpu",)
    grouped: list[list[tuple[BridgeFactorialConfig, int, int, str]]] = [
        [] for _ in devices
    ]
    for index, (noise_seed, time_seed) in enumerate(combinations):
        device = devices[index % len(devices)]
        grouped[index % len(devices)].append(
            (config, noise_seed, time_seed, device)
        )
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
    conditions = summarize_bridge_conditions(metrics)
    effects = bridge_factorial_effects(conditions)

    result_dir = None
    if config.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = config.output_root.expanduser() / f"factorial_{timestamp}"
        result_dir.mkdir(parents=True, exist_ok=False)
        serialized = asdict(config)
        serialized["data_root"] = str(config.data_root)
        serialized["output_root"] = str(config.output_root.expanduser())
        (result_dir / "config.json").write_text(
            json.dumps(serialized, indent=2), encoding="utf-8"
        )
        metrics.to_csv(result_dir / "rollout_metrics.csv", index=False)
        conditions.to_csv(result_dir / "condition_summary.csv", index=False)
        effects.to_csv(result_dir / "factorial_effects.csv", index=False)
        integrity = [
            {
                "noise_seed": result["noise_seed"],
                "time_seed": result["time_seed"],
                "classifier_accuracy": result["classifier_accuracy"],
                "hashes": result["hashes"],
            }
            for result in results
        ]
        (result_dir / "integrity.json").write_text(
            json.dumps(integrity, indent=2), encoding="utf-8"
        )
    return metrics, conditions, effects, result_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    devices = tuple(value.strip() for value in args.devices.split(",") if value.strip())
    config = BridgeFactorialConfig(
        devices=devices or ("cpu",), save=not args.no_save
    )
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
    _, conditions, effects, result_dir = run_study(config)
    print(f"result_dir={result_dir}")
    print(
        conditions[conditions["metric"].eq("feature_fid")].to_string(index=False)
    )
    print(effects.to_string(index=False))


if __name__ == "__main__":
    main()
