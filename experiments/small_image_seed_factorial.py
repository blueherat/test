"""Factor training-seed reversals into data, initialization, and SGD stream.

The matched DCT weighting study used one seed for the data subset, model
initialization, minibatch/noise stream, and evaluation.  This 2x2x2 study
separates the first three sources around the observed MNIST seed-3/seed-4
reversal while resampling each frozen endpoint with independent rollout seeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mnist_spectral_rollout_toy import (  # noqa: E402
    MNISTToyConfig,
    configure_fp32,
    evaluate_rollouts,
    train_feature_classifier,
    train_paired_velocity_fields,
)
from experiments.small_image_basis_transport import (  # noqa: E402
    build_direction_analyzer,
    load_small_image_tensors,
)


DEFAULT_OUTPUT_ROOT = Path.home() / "data/eqvae/experiments/small_image_seed_factorial"


@dataclass(frozen=True)
class SeedFactorialConfig:
    data_root: Path = Path("/data/shared/mnist")
    output_root: Path = DEFAULT_OUTPUT_ROOT
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


def _model_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _toy_config(
    config: SeedFactorialConfig,
    *,
    seed: int,
    device: str,
) -> MNISTToyConfig:
    return MNISTToyConfig(
        data_root=config.data_root,
        output_root=config.output_root,
        train_size=config.train_size,
        test_size=config.test_size,
        sample_count=config.sample_count,
        batch_size=config.batch_size,
        steps=config.steps,
        learning_rate=config.learning_rate,
        width=config.width,
        depth=config.depth,
        gamma=config.gamma,
        band_count=config.band_count,
        ode_steps=config.ode_steps,
        classifier_epochs=config.classifier_epochs,
        classifier_batch_size=config.classifier_batch_size,
        seed=int(seed),
        device=str(device),
        save=False,
    )


def _paired_metric_rows(
    rollout: pd.DataFrame,
    *,
    data_seed: int,
    init_seed: int,
    stream_seed: int,
    rollout_seed: int,
) -> list[dict[str, float | int | str]]:
    values = rollout.set_index("variant")
    rows = []
    for metric in ("feature_fid", "feature_swd", "latent_swd"):
        baseline = float(values.loc["baseline", metric])
        weighted = float(values.loc["weighted", metric])
        rows.append(
            {
                "data_seed": int(data_seed),
                "init_seed": int(init_seed),
                "stream_seed": int(stream_seed),
                "rollout_seed": int(rollout_seed),
                "metric": metric,
                "baseline": baseline,
                "weighted": weighted,
                "ratio": weighted / max(baseline, 1e-12),
                "delta": weighted - baseline,
            }
        )
    return rows


def summarize_conditions(metrics: pd.DataFrame) -> pd.DataFrame:
    required = {
        "data_seed",
        "init_seed",
        "stream_seed",
        "rollout_seed",
        "metric",
        "baseline",
        "weighted",
        "ratio",
    }
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"metrics is missing columns: {sorted(missing)}")
    return (
        metrics.groupby(
            ["data_seed", "init_seed", "stream_seed", "metric"],
            as_index=False,
        )
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
        .sort_values(["metric", "data_seed", "init_seed", "stream_seed"])
        .reset_index(drop=True)
    )


def factorial_effects(
    conditions: pd.DataFrame,
    *,
    metric: str = "feature_fid",
) -> pd.DataFrame:
    """Return saturated 2-level effects on log(weighted/baseline)."""

    selected = conditions[conditions["metric"].eq(metric)].copy()
    if len(selected) != 8:
        raise ValueError("factorial effects require all eight 2x2x2 conditions")
    factors = ("data_seed", "init_seed", "stream_seed")
    encoded: dict[str, pd.Series] = {}
    for factor in factors:
        levels = sorted(selected[factor].unique().tolist())
        if len(levels) != 2:
            raise ValueError(f"{factor} must have exactly two levels")
        encoded[factor] = selected[factor].map({levels[0]: -1.0, levels[1]: 1.0})
    response = selected["ratio_mean"].clip(lower=1e-12).map(math.log)
    rows = []
    terms = (
        ("data", ("data_seed",)),
        ("init", ("init_seed",)),
        ("stream", ("stream_seed",)),
        ("data:init", ("data_seed", "init_seed")),
        ("data:stream", ("data_seed", "stream_seed")),
        ("init:stream", ("init_seed", "stream_seed")),
        ("data:init:stream", factors),
    )
    for name, term_factors in terms:
        contrast = pd.Series(1.0, index=selected.index)
        for factor in term_factors:
            contrast = contrast * encoded[factor]
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
    config: SeedFactorialConfig,
    data_seed: int,
    init_seed: int,
    stream_seed: int,
    device_name: str,
) -> dict[str, object]:
    configure_fp32(init_seed)
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
        data_seed,
        download=False,
    )
    train = loaded["train"].to(device)
    test = loaded["test"].to(device)
    analyzer, _ = build_direction_analyzer(
        train,
        "dct",
        band_count=config.band_count,
        gamma=config.gamma,
        seed=data_seed,
    )
    analyzer = analyzer.to(device)
    train_config = _toy_config(config, seed=stream_seed, device=str(device))
    models, _ = train_paired_velocity_fields(
        train,
        train_config,
        analyzer,
        init_seed=init_seed,
        stream_seed=stream_seed,
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
        seed=data_seed,
    )
    metric_rows = []
    for rollout_seed in config.rollout_seeds:
        evaluation_config = replace(train_config, seed=int(rollout_seed))
        rollout, _, _ = evaluate_rollouts(
            models,
            test,
            classifier,
            analyzer,
            evaluation_config,
            loaded["normalization"],
        )
        metric_rows.extend(
            _paired_metric_rows(
                rollout,
                data_seed=data_seed,
                init_seed=init_seed,
                stream_seed=stream_seed,
                rollout_seed=rollout_seed,
            )
        )
    hashes = {name: _model_hash(model) for name, model in models.items()}
    del train, test, classifier, analyzer, models
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "data_seed": int(data_seed),
        "init_seed": int(init_seed),
        "stream_seed": int(stream_seed),
        "classifier_accuracy": float(classifier_accuracy),
        "hashes": hashes,
        "metric_rows": metric_rows,
    }


def _run_group(
    tasks: Sequence[tuple[SeedFactorialConfig, int, int, int, str]],
) -> list[dict[str, object]]:
    results = []
    for config, data_seed, init_seed, stream_seed, device in tasks:
        result = _run_condition(
            config, data_seed, init_seed, stream_seed, device
        )
        fid_rows = [
            row for row in result["metric_rows"] if row["metric"] == "feature_fid"
        ]
        ratio = sum(float(row["ratio"]) for row in fid_rows) / len(fid_rows)
        print(
            f"done data={data_seed} init={init_seed} stream={stream_seed}: "
            f"fid_ratio={ratio:.3f}"
        )
        results.append(result)
    return results


def run_study(
    config: SeedFactorialConfig = SeedFactorialConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path | None]:
    if len(set(config.levels)) != 2:
        raise ValueError("levels must contain exactly two distinct seeds")
    # Download before workers start to avoid concurrent archive writes.
    load_small_image_tensors(
        "mnist", config.data_root, 1, 1, 0, download=True
    )
    combinations = [
        (data_seed, init_seed, stream_seed)
        for data_seed in config.levels
        for init_seed in config.levels
        for stream_seed in config.levels
    ]
    devices = config.devices or ("cpu",)
    tasks = [
        (*combination, devices[index % len(devices)])
        for index, combination in enumerate(combinations)
    ]
    grouped: list[list[tuple[SeedFactorialConfig, int, int, int, str]]] = [
        [] for _ in devices
    ]
    for index, (data_seed, init_seed, stream_seed, device) in enumerate(tasks):
        grouped[index % len(devices)].append(
            (config, data_seed, init_seed, stream_seed, device)
        )

    results: list[dict[str, object]] = []
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
    effects = factorial_effects(conditions)

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
                "data_seed": result["data_seed"],
                "init_seed": result["init_seed"],
                "stream_seed": result["stream_seed"],
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
    config = SeedFactorialConfig(
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
