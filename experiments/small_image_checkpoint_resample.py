"""Re-evaluate saved small-image fields across independent rollout seeds.

This audit separates training-seed effects from finite-sample endpoint noise. It
never retrains the velocity fields; only the small evaluation classifier is
reconstructed from the original deterministic data split.
"""

from __future__ import annotations

import argparse
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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mnist_spectral_rollout_toy import (  # noqa: E402
    configure_fp32,
    evaluate_rollouts,
    train_feature_classifier,
)
from experiments.small_image_basis_mechanism import (  # noqa: E402
    _load_run,
    _load_study_config,
)
from experiments.small_image_basis_transport import (  # noqa: E402
    _toy_config,
    load_small_image_tensors,
)


@dataclass(frozen=True)
class CheckpointResampleConfig:
    study_dir: Path
    output_root: Path = (
        Path.home() / "data/eqvae/experiments/small_image_checkpoint_resample"
    )
    bases: tuple[str, ...] = ("dct", "pca", "random")
    training_seeds: tuple[int, ...] = (3, 4)
    evaluation_seeds: tuple[int, ...] = (1701, 1702, 1703, 1704, 1705)
    devices: tuple[str, ...] = ("cuda:0", "cuda:1", "cuda:2", "cuda:3")
    test_size: int = 4096
    sample_count: int = 4096
    batch_size: int = 256
    save: bool = True


def paired_endpoint_rows(
    rollout: pd.DataFrame,
    *,
    dataset: str,
    basis: str,
    training_seed: int,
    evaluation_seed: int,
) -> list[dict[str, float | int | str]]:
    required = {"variant", "feature_fid", "feature_swd", "latent_swd"}
    missing = required - set(rollout.columns)
    if missing:
        raise ValueError(f"rollout is missing columns: {sorted(missing)}")
    indexed = rollout.set_index("variant")
    if set(indexed.index) != {"baseline", "weighted"}:
        raise ValueError("rollout must contain exactly baseline and weighted variants")
    rows = []
    for metric in ("feature_fid", "feature_swd", "latent_swd"):
        baseline = float(indexed.loc["baseline", metric])
        weighted = float(indexed.loc["weighted", metric])
        rows.append(
            {
                "dataset": str(dataset),
                "basis": str(basis),
                "training_seed": int(training_seed),
                "evaluation_seed": int(evaluation_seed),
                "metric": metric,
                "baseline": baseline,
                "weighted": weighted,
                "delta": weighted - baseline,
                "ratio": weighted / max(baseline, 1e-12),
            }
        )
    return rows


def summarize_resamples(metrics: pd.DataFrame) -> pd.DataFrame:
    required = {
        "dataset",
        "basis",
        "training_seed",
        "evaluation_seed",
        "metric",
        "baseline",
        "weighted",
        "delta",
        "ratio",
    }
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"metrics is missing columns: {sorted(missing)}")
    return (
        metrics.groupby(
            ["dataset", "basis", "training_seed", "metric"], as_index=False
        )
        .agg(
            evaluation_seeds=("evaluation_seed", "nunique"),
            baseline_mean=("baseline", "mean"),
            baseline_std=("baseline", "std"),
            weighted_mean=("weighted", "mean"),
            weighted_std=("weighted", "std"),
            delta_mean=("delta", "mean"),
            delta_std=("delta", "std"),
            ratio_mean=("ratio", "mean"),
            ratio_std=("ratio", "std"),
            ratio_min=("ratio", "min"),
            ratio_max=("ratio", "max"),
            harm_rate=("ratio", lambda values: float((values > 1.0).mean())),
            practical_harm_rate=(
                "ratio",
                lambda values: float((values > 1.02).mean()),
            ),
        )
        .sort_values(["dataset", "training_seed", "basis", "metric"])
        .reset_index(drop=True)
    )


def _resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def _run_training_seed(
    config: CheckpointResampleConfig,
    training_seed: int,
    device_name: str,
) -> pd.DataFrame:
    study_dir = config.study_dir.expanduser().resolve()
    study_config = _load_study_config(study_dir)
    configure_fp32(int(training_seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = _resolve_device(device_name)
    loaded = load_small_image_tensors(
        study_config.dataset,
        study_config.data_root,
        study_config.train_size,
        config.test_size,
        int(training_seed),
        download=False,
    )
    train = loaded["train"].to(device)
    test = loaded["test"].to(device)
    classifier, classifier_accuracy = train_feature_classifier(
        train,
        loaded["train_labels"].to(device),
        test,
        loaded["test_labels"].to(device),
        epochs=study_config.classifier_epochs,
        batch_size=study_config.classifier_batch_size,
        seed=int(training_seed),
    )
    del train

    rows: list[dict[str, float | int | str]] = []
    for basis in config.bases:
        run_dir = study_dir / f"{basis}_seed{training_seed}"
        models, analyzer, state = _load_run(run_dir, study_config, device)
        normalization = state["normalization"]
        for evaluation_seed in config.evaluation_seeds:
            toy_config = replace(
                _toy_config(
                    study_config,
                    seed=int(evaluation_seed),
                    device=str(device),
                    output_root=config.output_root,
                ),
                sample_count=min(int(config.sample_count), len(test)),
                batch_size=int(config.batch_size),
            )
            rollout, _, _ = evaluate_rollouts(
                models,
                test,
                classifier,
                analyzer,
                toy_config,
                normalization,
            )
            paired = paired_endpoint_rows(
                rollout,
                dataset=study_config.dataset,
                basis=basis,
                training_seed=int(training_seed),
                evaluation_seed=int(evaluation_seed),
            )
            for row in paired:
                row["classifier_accuracy"] = float(classifier_accuracy)
                row["sample_count"] = int(toy_config.sample_count)
            rows.extend(paired)
        for model in models.values():
            model.cpu()
        analyzer.cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    classifier.cpu()
    return pd.DataFrame(rows)


def run_checkpoint_resample(
    config: CheckpointResampleConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, Path | None]:
    devices = config.devices or ("cpu",)
    tasks = [
        (config, int(seed), devices[index % len(devices)])
        for index, seed in enumerate(config.training_seeds)
    ]
    if len(tasks) == 1:
        frames = [_run_training_seed(*tasks[0])]
    else:
        context = get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(len(tasks), len(devices)), mp_context=context
        ) as executor:
            futures = [executor.submit(_run_training_seed, *task) for task in tasks]
            frames = []
            for future in as_completed(futures):
                frames.append(future.result())
    metrics = pd.concat(frames, ignore_index=True).sort_values(
        ["training_seed", "basis", "evaluation_seed", "metric"]
    )
    expected = (
        len(config.training_seeds)
        * len(config.bases)
        * len(config.evaluation_seeds)
        * 3
    )
    if len(metrics) != expected:
        raise RuntimeError(f"expected {expected} metric rows, received {len(metrics)}")
    summary = summarize_resamples(metrics)
    result_dir = None
    if config.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = config.output_root.expanduser() / f"resample_{timestamp}"
        result_dir.mkdir(parents=True, exist_ok=False)
        serialized = asdict(config)
        for key in ("study_dir", "output_root"):
            serialized[key] = str(serialized[key])
        (result_dir / "config.json").write_text(
            json.dumps(serialized, indent=2), encoding="utf-8"
        )
        metrics.to_csv(result_dir / "metrics.csv", index=False)
        summary.to_csv(result_dir / "summary.csv", index=False)
    return metrics, summary, result_dir


def _integers(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())


def _strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--bases", default="dct,pca,random")
    parser.add_argument("--training-seeds", default="3,4")
    parser.add_argument("--evaluation-seeds", default="1701,1702,1703,1704,1705")
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3")
    parser.add_argument("--test-size", type=int, default=4096)
    parser.add_argument("--sample-count", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    config = CheckpointResampleConfig(
        study_dir=args.study_dir,
        output_root=args.output_root or CheckpointResampleConfig.output_root,
        bases=_strings(args.bases),
        training_seeds=_integers(args.training_seeds),
        evaluation_seeds=_integers(args.evaluation_seeds),
        devices=_strings(args.devices) or ("cpu",),
        test_size=args.test_size,
        sample_count=args.sample_count,
        batch_size=args.batch_size,
        save=not args.no_save,
    )
    _, summary, result_dir = run_checkpoint_resample(config)
    fid = summary[summary["metric"].eq("feature_fid")]
    print(fid.round(4).to_string(index=False))
    print(f"result_dir={result_dir}")


if __name__ == "__main__":
    main()
