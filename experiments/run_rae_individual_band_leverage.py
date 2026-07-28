"""Run preregistered high-window individual-band RAE endpoint splices."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.rae_frequency_time_switch_probe import run_probe  # noqa: E402


@dataclass(frozen=True)
class IndividualBandLeverageConfig:
    experiment_root: Path = Path.home() / "data/eqvae/experiments/rae_spectral_tiny"
    output_root: Path = Path.home() / "data/eqvae/experiments/rae_individual_band_leverage"
    seeds: tuple[int, ...] = (3407, 4211, 5821)
    devices: tuple[str, ...] = ("cuda:0", "cuda:1", "cuda:2")
    count: int = 32
    batch_size: int = 4
    evaluation_seed: int = 161803
    band_count: int = 8
    save: bool = True


def individual_schedules(band_count: int) -> tuple[str, ...]:
    if int(band_count) < 1:
        raise ValueError("band_count must be positive")
    return (
        "baseline",
        "partial_high_all",
        *(f"partial_high_band{band}" for band in range(int(band_count))),
    )


def _run_seed(config: IndividualBandLeverageConfig, seed: int, device: str):
    root = config.experiment_root.expanduser().resolve()
    baseline = root / f"seed{seed}_baseline_from_s5000"
    partial = root / f"seed{seed}_partial_from_s5000"
    metrics, bands, metadata = run_probe(
        baseline,
        partial,
        device=device,
        count=config.count,
        batch_size=config.batch_size,
        evaluation_seed=config.evaluation_seed,
        schedules=individual_schedules(config.band_count),
    )
    print(f"audited individual RAE bands seed={seed}", flush=True)
    return metrics, bands, metadata


def run_study(config: IndividualBandLeverageConfig):
    devices = config.devices or ("cpu",)
    tasks = [
        (config, int(seed), devices[index % len(devices)])
        for index, seed in enumerate(config.seeds)
    ]
    if len(tasks) == 1:
        results = [_run_seed(*tasks[0])]
    else:
        context = get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(len(devices), len(tasks)), mp_context=context
        ) as executor:
            futures = [executor.submit(_run_seed, *task) for task in tasks]
            results = [future.result() for future in as_completed(futures)]
    metrics = pd.concat([result[0] for result in results], ignore_index=True)
    bands = pd.concat([result[1] for result in results], ignore_index=True)
    metrics = metrics.sort_values(["seed", "schedule", "metric"]).reset_index(drop=True)
    bands = bands.sort_values(["seed", "schedule", "band"]).reset_index(drop=True)
    baseline = metrics[metrics["schedule"].eq("baseline")][
        ["seed", "metric", "value"]
    ].rename(columns={"value": "baseline_value"})
    paired = metrics.merge(baseline, on=["seed", "metric"], validate="many_to_one")
    paired["delta"] = paired["value"] - paired["baseline_value"]
    paired["ratio"] = paired["value"] / paired["baseline_value"].clip(lower=1e-12)
    result_dir = None
    if config.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = config.output_root.expanduser() / f"preregistered_{timestamp}"
        result_dir.mkdir(parents=True, exist_ok=False)
        serialized = asdict(config)
        serialized["experiment_root"] = str(config.experiment_root.expanduser())
        serialized["output_root"] = str(config.output_root.expanduser())
        metrics.to_csv(result_dir / "metrics.csv", index=False)
        paired.to_csv(result_dir / "paired_metrics.csv", index=False)
        bands.to_csv(result_dir / "bands.csv", index=False)
        (result_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "config": serialized,
                    "seeds": [result[2] for result in results],
                    "scope": "frozen paired EMA high-window individual-band field splices",
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    return metrics, paired, bands, result_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--seeds", default="3407,4211,5821")
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2")
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--evaluation-seed", type=int, default=161803)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    devices = tuple(value.strip() for value in args.devices.split(",") if value.strip())
    config = IndividualBandLeverageConfig(
        experiment_root=args.experiment_root or IndividualBandLeverageConfig.experiment_root,
        output_root=args.output_root or IndividualBandLeverageConfig.output_root,
        seeds=seeds,
        devices=devices or ("cpu",),
        count=args.count,
        batch_size=args.batch_size,
        evaluation_seed=args.evaluation_seed,
        save=not args.no_save,
    )
    _, paired, _, result_dir = run_study(config)
    metric = paired[
        paired["metric"].eq("summary_swd_to_validation")
        & paired["schedule"].str.startswith("partial_high_")
    ]
    print(f"result_dir={result_dir}")
    print(
        metric.groupby("schedule")[["delta", "ratio"]]
        .agg(["mean", "std", "min", "max"])
        .sort_values(("delta", "mean"), ascending=False)
        .round(5)
        .to_string()
    )


if __name__ == "__main__":
    main()
