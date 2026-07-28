"""Split the flow-training stream into minibatch and bridge randomness.

For each source seed, the code advances one generator through exactly the same
index, Gaussian-noise, and time draws as the original training loop.  Selecting
all draws from one source therefore reproduces that original stream exactly;
crossed conditions exchange only minibatch indices versus bridge noise/time.
"""

from __future__ import annotations

import argparse
import copy
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


DEFAULT_OUTPUT_ROOT = Path.home() / "data/eqvae/experiments/small_image_stream_factorial"


@dataclass(frozen=True)
class StreamFactorialConfig:
    data_root: Path = Path("/data/shared/mnist")
    output_root: Path = DEFAULT_OUTPUT_ROOT
    data_seed: int = 4
    init_seed: int = 4
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


def train_paired_mixed_streams(
    clean: torch.Tensor,
    config: MNISTToyConfig,
    analyzer: torch.nn.Module,
    *,
    init_seed: int,
    batch_seed: int,
    bridge_seed: int | None = None,
    noise_seed: int | None = None,
    time_seed: int | None = None,
) -> dict[str, TinyVelocityUNet]:
    """Train using crossed index and bridge draws while preserving source streams."""

    device = clean.device
    if bridge_seed is not None:
        if noise_seed is not None or time_seed is not None:
            raise ValueError("bridge_seed cannot be combined with noise_seed/time_seed")
        noise_seed = int(bridge_seed)
        time_seed = int(bridge_seed)
    if noise_seed is None or time_seed is None:
        raise ValueError("provide bridge_seed or both noise_seed and time_seed")
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
    source_seeds = sorted({int(batch_seed), int(noise_seed), int(time_seed)})
    generators = {
        seed: torch.Generator(device=device).manual_seed(seed + 101)
        for seed in source_seeds
    }
    for _ in range(int(config.steps)):
        draws = {}
        for seed, generator in generators.items():
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
            time = shifted_uniform(
                config.batch_size,
                config.time_shift,
                device=device,
                generator=generator,
            )
            draws[seed] = (indices, noise, time)
        indices = draws[int(batch_seed)][0]
        noise = draws[int(noise_seed)][1]
        time = draws[int(time_seed)][2]
        data = clean[indices]
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


def summarize_stream_conditions(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["batch_seed", "bridge_seed", "metric"], as_index=False)
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
        .sort_values(["metric", "batch_seed", "bridge_seed"])
        .reset_index(drop=True)
    )


def stream_factorial_effects(
    conditions: pd.DataFrame,
    *,
    metric: str = "feature_fid",
) -> pd.DataFrame:
    selected = conditions[conditions["metric"].eq(metric)].copy()
    if len(selected) != 4:
        raise ValueError("stream factorial requires all four 2x2 conditions")
    encodings = {}
    for factor in ("batch_seed", "bridge_seed"):
        levels = sorted(selected[factor].unique().tolist())
        if len(levels) != 2:
            raise ValueError(f"{factor} must have two levels")
        encodings[factor] = selected[factor].map(
            {levels[0]: -1.0, levels[1]: 1.0}
        )
    response = selected["ratio_mean"].clip(lower=1e-12).map(math.log)
    rows = []
    for name, factors in (
        ("batch", ("batch_seed",)),
        ("bridge", ("bridge_seed",)),
        ("batch:bridge", ("batch_seed", "bridge_seed")),
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
    config: StreamFactorialConfig,
    batch_seed: int,
    bridge_seed: int,
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
    toy_config = _toy_config(config, seed=bridge_seed, device=str(device))
    models = train_paired_mixed_streams(
        train,
        toy_config,
        analyzer,
        init_seed=config.init_seed,
        batch_seed=batch_seed,
        bridge_seed=bridge_seed,
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
            stream_seed=bridge_seed,
            rollout_seed=rollout_seed,
        ):
            row["batch_seed"] = int(batch_seed)
            row["bridge_seed"] = int(bridge_seed)
            rows.append(row)
    hashes = {name: _model_hash(model) for name, model in models.items()}
    del train, test, classifier, analyzer, models
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "batch_seed": int(batch_seed),
        "bridge_seed": int(bridge_seed),
        "classifier_accuracy": float(classifier_accuracy),
        "hashes": hashes,
        "metric_rows": rows,
    }


def _run_group(
    tasks: Sequence[tuple[StreamFactorialConfig, int, int, str]],
) -> list[dict[str, object]]:
    results = []
    for config, batch_seed, bridge_seed, device in tasks:
        result = _run_condition(config, batch_seed, bridge_seed, device)
        fid_rows = [
            row for row in result["metric_rows"] if row["metric"] == "feature_fid"
        ]
        ratio = sum(float(row["ratio"]) for row in fid_rows) / len(fid_rows)
        print(
            f"done batch={batch_seed} bridge={bridge_seed}: fid_ratio={ratio:.3f}"
        )
        results.append(result)
    return results


def run_study(
    config: StreamFactorialConfig = StreamFactorialConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path | None]:
    if len(set(config.levels)) != 2:
        raise ValueError("levels must contain exactly two distinct seeds")
    load_small_image_tensors(
        "mnist", config.data_root, 1, 1, 0, download=True
    )
    combinations = [
        (batch_seed, bridge_seed)
        for batch_seed in config.levels
        for bridge_seed in config.levels
    ]
    devices = config.devices or ("cpu",)
    grouped: list[list[tuple[StreamFactorialConfig, int, int, str]]] = [
        [] for _ in devices
    ]
    for index, (batch_seed, bridge_seed) in enumerate(combinations):
        device = devices[index % len(devices)]
        grouped[index % len(devices)].append(
            (config, batch_seed, bridge_seed, device)
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
    conditions = summarize_stream_conditions(metrics)
    effects = stream_factorial_effects(conditions)

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
                "batch_seed": result["batch_seed"],
                "bridge_seed": result["bridge_seed"],
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
    config = StreamFactorialConfig(
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
