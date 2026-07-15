from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

from experiments.architecture_gauge import (
    BLUE,
    GOLD,
    INK,
    LIGHT_GREY,
    OLIVE,
    ORANGE,
    PINK,
    CodecDataConfig,
    GaugeSpec,
    ProbeConfig,
    ProbeTrainingConfig,
    configure_reproducibility,
    prepare_codec_data,
    train_probe,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = Path.home() / "data/eqvae/cache/gauge_large/rae_dinov2_imagenet_2048_512.pt"
DEFAULT_RESULT_DIR = Path.home() / "data/eqvae/artifacts/gauge_large_validation"
T_CRITICAL_95 = {1: math.nan, 2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}
PALETTE = (BLUE, ORANGE, GOLD, OLIVE, PINK, "#0891B2", "#6B7280")
GAUGE_STYLES = {
    "identity": {"color": INK, "marker": "x", "linestyle": "--"},
    "allpass_plus": {"color": "#EA580C", "marker": "o", "linestyle": "-"},
    "allpass_minus": {"color": "#FB923C", "marker": "s", "linestyle": "--"},
    "allpass_r3": {"color": "#9A3412", "marker": "^", "linestyle": ":"},
    "channel_plus": {"color": "#2563EB", "marker": "D", "linestyle": "-"},
    "channel_minus": {"color": "#60A5FA", "marker": "v", "linestyle": "--"},
    "haar_2x2": {"color": "#6B7280", "marker": "P", "linestyle": "-"},
}


@dataclass(frozen=True)
class LargeValidationConfig:
    train_counts: tuple[int, ...] = (1024, 2048)
    val_count: int = 512
    steps: int = 2000
    eval_steps: tuple[int, ...] = (0, 100, 500, 1000, 2000)
    batch_size: int = 32
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    hidden: int = 16
    time_bins: int = 8
    learning_rate: float = 1e-3
    dataset_path: str = "/data/shared/imagenet-1k"
    model_key: str = "rae_dinov2"
    data_seed: int = 0


def large_gauges() -> List[GaugeSpec]:
    return [
        GaugeSpec("identity"),
        GaugeSpec("allpass_plus", kind="fourier_allpass", strength=0.25, radius=1),
        GaugeSpec("allpass_minus", kind="fourier_allpass", strength=-0.25, radius=1),
        GaugeSpec("allpass_r3", kind="fourier_allpass", strength=0.65, radius=3),
        GaugeSpec("channel_plus", kind="channel_givens", strength=0.5, seed=0),
        GaugeSpec("channel_minus", kind="channel_givens", strength=-0.5, seed=0),
        GaugeSpec("haar_2x2", kind="block_haar"),
    ]


def large_probes(hidden: int = 16) -> List[ProbeConfig]:
    return [
        ProbeConfig("local_rf5", kind="local", hidden=hidden, depth=1),
        ProbeConfig("local_rf9", kind="local", hidden=hidden, depth=2),
        ProbeConfig("local_rf17", kind="local", hidden=hidden, depth=4),
        ProbeConfig("global_attn", kind="global", hidden=hidden, depth=1, heads=4),
    ]


def training_config(config: LargeValidationConfig) -> ProbeTrainingConfig:
    eval_steps = tuple(sorted(set(config.eval_steps) | {0, config.steps}))
    return ProbeTrainingConfig(
        steps=config.steps,
        eval_steps=eval_steps,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        eval_full_dataset=True,
        time_bins=config.time_bins,
        seed=config.data_seed,
    )


def prepare_latent_cache(
    cache_path: str | Path = DEFAULT_CACHE,
    *,
    config: LargeValidationConfig = LargeValidationConfig(),
    device: str = "cuda:0",
    force: bool = False,
) -> Path:
    cache_path = Path(cache_path).expanduser()
    if cache_path.exists() and not force:
        print(f"cache exists: {cache_path}")
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    max_train = max(config.train_counts)
    bundle = prepare_codec_data(
        CodecDataConfig(
            dataset_name="imagenet_parquet",
            dataset_path=config.dataset_path,
            train_split="train",
            val_split="validation",
            train_count=max_train,
            val_count=config.val_count,
            image_size=256,
            model_key=config.model_key,
            rae_repo_path=str(ROOT / "external/RAE"),
            device=device,
            encode_batch_size=32,
            seed=config.data_seed,
        )
    )
    scale = float(bundle.train_latents.square().mean().sqrt().clamp_min(1e-8))
    payload = {
        "train_latents": (bundle.train_latents / scale).contiguous(),
        "val_latents": (bundle.val_latents / scale).contiguous(),
        "latent_scale": scale,
        "train_indices": bundle.train_indices,
        "val_indices": bundle.val_indices,
        "metadata": {
            **asdict(config),
            "max_train_count": max_train,
            "normalization": "single RMS scalar fitted on max train split",
            "train_source_split": "train",
            "val_source_split": "validation",
            "latent_dtype": "float32",
        },
    }
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, cache_path)
    print(
        f"saved cache: {cache_path} train={tuple(payload['train_latents'].shape)} "
        f"val={tuple(payload['val_latents'].shape)} scale={scale:.6f}"
    )
    return cache_path


def load_latent_cache(cache_path: str | Path = DEFAULT_CACHE) -> Dict[str, object]:
    cache_path = Path(cache_path).expanduser()
    if not cache_path.exists():
        raise FileNotFoundError(f"latent cache missing: {cache_path}")
    return torch.load(cache_path, map_location="cpu", mmap=True, weights_only=False)


def build_tasks(config: LargeValidationConfig) -> List[Dict[str, object]]:
    tasks = []
    for train_count in config.train_counts:
        for probe in large_probes(config.hidden):
            for gauge in large_gauges():
                for seed in config.seeds:
                    tasks.append(
                        {
                            "train_count": int(train_count),
                            "probe": asdict(probe),
                            "gauge": asdict(gauge),
                            "seed": int(seed),
                        }
                    )
    return tasks


def task_name(task: Dict[str, object]) -> str:
    probe = str(task["probe"]["name"])
    gauge = str(task["gauge"]["name"])
    return f"train{task['train_count']}__{probe}__{gauge}__seed{task['seed']}.pt"


def _augment_rows(
    rows: Iterable[Dict[str, object]],
    task: Dict[str, object],
    config: LargeValidationConfig,
) -> List[Dict[str, object]]:
    common = {
        "train_count": int(task["train_count"]),
        "val_count": config.val_count,
        "training_steps": config.steps,
        "full_validation": True,
    }
    return [{**common, **row} for row in rows]


def _worker(
    worker_index: int,
    gpu: int,
    tasks: Sequence[Dict[str, object]],
    cache_path: str,
    result_dir: str,
    config_dict: Dict[str, object],
    force: bool,
) -> None:
    torch.set_num_threads(1)
    config = LargeValidationConfig(**config_dict)
    configure_reproducibility(config.data_seed)
    cache = load_latent_cache(cache_path)
    train_latents = cache["train_latents"]
    val_latents = cache["val_latents"]
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    training = training_config(config)
    task_dir = Path(result_dir) / "tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    for index, task in enumerate(tasks, start=1):
        destination = task_dir / task_name(task)
        if destination.exists() and not force:
            print(f"[gpu{gpu}] skip {index}/{len(tasks)} {destination.name}", flush=True)
            continue
        probe = ProbeConfig(**task["probe"])
        gauge = GaugeSpec(**task["gauge"])
        train_count = int(task["train_count"])
        started = time.perf_counter()
        run, history, time_rows = train_probe(
            train_latents[:train_count],
            val_latents,
            gauge,
            probe,
            training,
            device=device,
            run_seed=int(task["seed"]),
        )
        payload = {
            "task": task,
            "config": asdict(config),
            "history": _augment_rows(history, task, config),
            "time_rows": _augment_rows(time_rows, task, config),
            "run": run,
            "worker": worker_index,
            "gpu": gpu,
            "wall_seconds": time.perf_counter() - started,
        }
        temporary = destination.with_suffix(".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, destination)
        print(
            f"[gpu{gpu}] done {index}/{len(tasks)} {destination.name} "
            f"loss={run.final_relative_mse:.6f} wall={payload['wall_seconds']:.1f}s",
            flush=True,
        )


def run_large_validation(
    cache_path: str | Path = DEFAULT_CACHE,
    result_dir: str | Path = DEFAULT_RESULT_DIR,
    *,
    config: LargeValidationConfig = LargeValidationConfig(),
    gpus: Sequence[int] = (0, 1, 2, 3),
    force: bool = False,
) -> None:
    result_dir = Path(result_dir).expanduser()
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n",
        encoding="utf-8",
    )
    tasks = build_tasks(config)
    existing = {path.name for path in (result_dir / "tasks").glob("*.pt")}
    pending = [task for task in tasks if force or task_name(task) not in existing]
    print(f"tasks total={len(tasks)} existing={len(tasks) - len(pending)} pending={len(pending)}")
    if not pending:
        return
    rng = np.random.default_rng(0)
    order = rng.permutation(len(pending))
    shuffled = [pending[int(index)] for index in order]
    partitions = [shuffled[index :: len(gpus)] for index in range(len(gpus))]
    context = mp.get_context("spawn")
    processes = []
    for worker_index, (gpu, worker_tasks) in enumerate(zip(gpus, partitions)):
        process = context.Process(
            target=_worker,
            args=(
                worker_index,
                int(gpu),
                worker_tasks,
                str(Path(cache_path).expanduser()),
                str(result_dir),
                asdict(config),
                force,
            ),
        )
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
    failures = [process.pid for process in processes if process.exitcode != 0]
    if failures:
        raise RuntimeError(f"large validation workers failed: {failures}")


def load_task_results(result_dir: str | Path = DEFAULT_RESULT_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_dir = Path(result_dir).expanduser()
    history_rows: List[Dict[str, object]] = []
    time_rows: List[Dict[str, object]] = []
    for path in sorted((result_dir / "tasks").glob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        history_rows.extend(payload["history"])
        time_rows.extend(payload["time_rows"])
    if not history_rows:
        raise FileNotFoundError(f"no task results under {result_dir / 'tasks'}")
    return pd.DataFrame(history_rows), pd.DataFrame(time_rows)


def add_large_identity_ratios(history: pd.DataFrame) -> pd.DataFrame:
    keys = ["train_count", "probe", "seed", "step"]
    identity = history[history["gauge"] == "identity"][keys + ["relative_mse"]].rename(
        columns={"relative_mse": "identity_relative_mse"}
    )
    merged = history.merge(identity, on=keys, how="left", validate="many_to_one")
    merged["loss_ratio_to_identity"] = merged["relative_mse"] / merged["identity_relative_mse"]
    return merged


def _confidence_summary(frame: pd.DataFrame, group_keys: Sequence[str], value: str) -> pd.DataFrame:
    summary = frame.groupby(list(group_keys), as_index=False)[value].agg(["mean", "std", "count"]).reset_index()
    summary["sem"] = summary["std"] / np.sqrt(summary["count"])
    critical = summary["count"].map(lambda count: T_CRITICAL_95.get(int(count), 1.96))
    summary["ci95"] = critical * summary["sem"]
    summary["ci95_low"] = summary["mean"] - summary["ci95"]
    summary["ci95_high"] = summary["mean"] + summary["ci95"]
    return summary


def _add_holm_significance(final: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    keys = ["train_count", "probe", "probe_kind", "receptive_field", "gauge"]
    rows = []
    for group_values, group in final.groupby(keys):
        values = group["loss_ratio_to_identity"].to_numpy(dtype=float)
        if len(values) < 2 or np.allclose(values, 1.0):
            p_value = math.nan
        else:
            p_value = float(stats.ttest_1samp(values, popmean=1.0).pvalue)
        rows.append({**dict(zip(keys, group_values)), "p_value": p_value})
    merged = summary.merge(pd.DataFrame(rows), on=keys, how="left", validate="one_to_one")
    merged["p_holm"] = math.nan
    for train_count, group in merged[
        (merged["gauge"] != "identity") & merged["p_value"].notna()
    ].groupby("train_count"):
        ordered = group.sort_values("p_value")
        multiplier = np.arange(len(ordered), 0, -1, dtype=float)
        adjusted = np.maximum.accumulate(ordered["p_value"].to_numpy() * multiplier)
        merged.loc[ordered.index, "p_holm"] = np.minimum(adjusted, 1.0)
    merged["holm_significant"] = merged["p_holm"] < 0.05
    return merged


def _holm_adjust(p_values: pd.Series) -> pd.Series:
    """Return Holm-adjusted p-values while preserving the original index."""
    valid = p_values.dropna().sort_values()
    adjusted = pd.Series(math.nan, index=p_values.index, dtype=float)
    if valid.empty:
        return adjusted
    multiplier = np.arange(len(valid), 0, -1, dtype=float)
    values = np.maximum.accumulate(valid.to_numpy(dtype=float) * multiplier)
    adjusted.loc[valid.index] = np.minimum(values, 1.0)
    return adjusted


def _add_time_holm_significance(
    time_ratios: pd.DataFrame,
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Add paired seed tests with curve-wise and full-screen multiplicity control."""
    keys = ["train_count", "probe", "gauge", "time_bin", "t_center", "logsnr"]
    rows = []
    for group_values, group in time_ratios.groupby(keys):
        values = group["loss_ratio_to_identity"].to_numpy(dtype=float)
        if len(values) < 2 or np.allclose(values, 1.0):
            p_value = math.nan
        else:
            p_value = float(stats.ttest_1samp(values, popmean=1.0).pvalue)
        rows.append({**dict(zip(keys, group_values)), "p_value": p_value})
    merged = summary.merge(pd.DataFrame(rows), on=keys, how="left", validate="one_to_one")
    merged["p_holm_within_curve"] = math.nan
    merged["p_holm_screen"] = math.nan
    candidates = merged[(merged["gauge"] != "identity") & merged["p_value"].notna()]
    for _, group in candidates.groupby(["train_count", "probe", "gauge"]):
        merged.loc[group.index, "p_holm_within_curve"] = _holm_adjust(group["p_value"])
    for _, group in candidates.groupby("train_count"):
        merged.loc[group.index, "p_holm_screen"] = _holm_adjust(group["p_value"])
    merged["holm_screen_significant"] = merged["p_holm_screen"] < 0.05
    merged["holm_screen_better"] = merged["holm_screen_significant"] & (merged["mean"] < 1.0)
    merged["holm_screen_worse"] = merged["holm_screen_significant"] & (merged["mean"] > 1.0)
    return merged


def aggregate_results(result_dir: str | Path = DEFAULT_RESULT_DIR) -> Dict[str, pd.DataFrame]:
    result_dir = Path(result_dir).expanduser()
    result_dir.mkdir(parents=True, exist_ok=True)
    history, time_rows = load_task_results(result_dir)
    ratios = add_large_identity_ratios(history)
    final_step = int(history["step"].max())
    final = ratios[ratios["step"] == final_step].copy()
    final["beats_identity"] = final["loss_ratio_to_identity"] < 1.0
    final_summary = _confidence_summary(
        final,
        ["train_count", "probe", "probe_kind", "receptive_field", "gauge"],
        "loss_ratio_to_identity",
    )
    final_summary = _add_holm_significance(final, final_summary)
    wins = (
        final.groupby(["train_count", "probe", "gauge"], as_index=False)["beats_identity"]
        .sum()
        .rename(columns={"beats_identity": "seed_wins"})
    )
    final_summary = final_summary.merge(wins, on=["train_count", "probe", "gauge"], how="left")
    learning_summary = _confidence_summary(
        ratios,
        ["train_count", "probe", "gauge", "step"],
        "loss_ratio_to_identity",
    )

    time_keys = ["train_count", "probe", "seed", "time_bin"]
    identity_time = time_rows[time_rows["gauge"] == "identity"][time_keys + ["relative_mse"]].rename(
        columns={"relative_mse": "identity_relative_mse"}
    )
    time_ratios = time_rows.merge(identity_time, on=time_keys, how="left", validate="many_to_one")
    time_ratios["loss_ratio_to_identity"] = (
        time_ratios["relative_mse"] / time_ratios["identity_relative_mse"]
    )
    time_summary = _confidence_summary(
        time_ratios,
        ["train_count", "probe", "gauge", "time_bin", "t_center", "logsnr"],
        "loss_ratio_to_identity",
    )
    time_summary = _add_time_holm_significance(time_ratios, time_summary)

    final_wide = final.pivot_table(
        index=["train_count", "probe", "seed"],
        columns="gauge",
        values="relative_mse",
        aggfunc="first",
    ).reset_index()
    for prefix, plus, minus, delta in (
        ("allpass", "allpass_plus", "allpass_minus", 0.25),
        ("channel", "channel_plus", "channel_minus", 0.5),
    ):
        final_wide[f"{prefix}_gradient"] = (final_wide[plus] - final_wide[minus]) / (2 * delta)
        final_wide[f"{prefix}_curvature"] = (
            final_wide[plus] + final_wide[minus] - 2 * final_wide["identity"]
        ) / (delta * delta)
    direction_rows = []
    for metric in ("allpass_gradient", "allpass_curvature", "channel_gradient", "channel_curvature"):
        current = _confidence_summary(final_wide, ["train_count", "probe"], metric)
        current.insert(2, "metric", metric)
        direction_rows.append(current)
    directional_summary = pd.concat(direction_rows, ignore_index=True)

    local_names = ["local_rf5", "local_rf9", "local_rf17"]
    locality = final[
        final["probe"].isin([*local_names, "global_attn"])
        & final["gauge"].str.startswith("allpass")
    ].pivot_table(
        index=["train_count", "gauge", "seed"],
        columns="probe",
        values="loss_ratio_to_identity",
        aggfunc="first",
    ).reset_index()
    locality_rows = []
    for probe in local_names:
        current = locality[["train_count", "gauge", "seed", probe, "global_attn"]].copy()
        current["probe"] = probe
        current["receptive_field"] = {"local_rf5": 5, "local_rf9": 9, "local_rf17": 17}[probe]
        current["locality_penalty"] = current[probe] - current["global_attn"]
        locality_rows.append(
            current[
                ["train_count", "gauge", "seed", "probe", "receptive_field", "locality_penalty"]
            ]
        )
    locality_penalties = pd.concat(locality_rows, ignore_index=True)
    locality_summary = _confidence_summary(
        locality_penalties,
        ["train_count", "gauge", "probe", "receptive_field"],
        "locality_penalty",
    )

    tables = {
        "history": history,
        "ratios": ratios,
        "final": final,
        "final_summary": final_summary,
        "learning_summary": learning_summary,
        "time_ratios": time_ratios,
        "time_summary": time_summary,
        "directional": final_wide,
        "directional_summary": directional_summary,
        "locality_penalties": locality_penalties,
        "locality_summary": locality_summary,
    }
    table_dir = result_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(table_dir / f"{name}.csv", index=False)
    return tables


def task_completion_table(
    result_dir: str | Path = DEFAULT_RESULT_DIR,
    *,
    config: LargeValidationConfig = LargeValidationConfig(),
) -> pd.DataFrame:
    task_dir = Path(result_dir).expanduser() / "tasks"
    rows = []
    for task in build_tasks(config):
        path = task_dir / task_name(task)
        rows.append(
            {
                "train_count": task["train_count"],
                "probe": task["probe"]["name"],
                "gauge": task["gauge"]["name"],
                "seed": task["seed"],
                "complete": path.exists(),
                "path": str(path),
            }
        )
    return pd.DataFrame(rows)


def mechanism_evidence_tables(tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    final_summary = tables["final_summary"]
    h2 = final_summary[final_summary["gauge"] != "identity"].copy()
    h2["relative_improvement"] = 1.0 - h2["mean"]
    h2["ci_excludes_identity"] = h2["ci95_high"] < 1.0
    h2["h2_statistically_supported"] = (
        h2["ci_excludes_identity"] & h2["holm_significant"]
    )
    h2 = h2.sort_values(["train_count", "mean"])

    learning = tables["learning_summary"].copy()
    positive_steps = sorted(step for step in learning["step"].unique() if step > 0)
    early_step = positive_steps[0]
    final_step = max(positive_steps)
    early = learning[learning["step"] == early_step][
        ["train_count", "probe", "gauge", "mean"]
    ].rename(columns={"mean": "early_ratio"})
    late = learning[learning["step"] == final_step][
        ["train_count", "probe", "gauge", "mean", "ci95_low", "ci95_high"]
    ].rename(columns={"mean": "final_ratio"})
    finite_horizon = early.merge(late, on=["train_count", "probe", "gauge"], validate="one_to_one")
    finite_horizon["early_step"] = early_step
    finite_horizon["final_step"] = final_step
    finite_horizon["distance_shrink"] = (
        (finite_horizon["early_ratio"] - 1.0).abs()
        - (finite_horizon["final_ratio"] - 1.0).abs()
    )

    time_summary = tables["time_summary"]
    time_conflict_rows = []
    for keys, group in time_summary[time_summary["gauge"] != "identity"].groupby(
        ["train_count", "probe", "gauge"]
    ):
        train_count, probe, gauge = keys
        time_conflict_rows.append(
            {
                "train_count": train_count,
                "probe": probe,
                "gauge": gauge,
                "min_bin_ratio": group["mean"].min(),
                "max_bin_ratio": group["mean"].max(),
                "has_supported_better_bin": bool(group["holm_screen_better"].any()),
                "has_supported_worse_bin": bool(group["holm_screen_worse"].any()),
            }
        )
    time_conflicts = pd.DataFrame(time_conflict_rows)
    time_conflicts["supported_sign_conflict"] = (
        time_conflicts["has_supported_better_bin"]
        & time_conflicts["has_supported_worse_bin"]
    )
    return {
        "h2": h2,
        "locality": tables["locality_summary"],
        "finite_horizon": finite_horizon,
        "time_conflicts": time_conflicts,
        "directional": tables["directional_summary"],
    }


def plot_final_seed_summary(final_summary: pd.DataFrame) -> plt.Figure:
    train_counts = sorted(final_summary["train_count"].unique())
    preferred_probes = [probe.name for probe in large_probes()]
    preferred_gauges = [gauge.name for gauge in large_gauges()]
    probes = [name for name in preferred_probes if name in set(final_summary["probe"])]
    gauges = [name for name in preferred_gauges if name in set(final_summary["gauge"])]
    fig, axes = plt.subplots(
        len(train_counts),
        len(probes),
        figsize=(5.1 * len(probes), 4.2 * len(train_counts)),
        squeeze=False,
        sharey=True,
    )
    x = np.arange(len(gauges))
    for row, train_count in enumerate(train_counts):
        for column, probe in enumerate(probes):
            axis = axes[row, column]
            rows = (
                final_summary[
                    (final_summary["train_count"] == train_count)
                    & (final_summary["probe"] == probe)
                ]
                .set_index("gauge")
                .reindex(gauges)
            )
            for index, gauge in enumerate(gauges):
                style = GAUGE_STYLES[gauge]
                axis.errorbar(
                    x[index],
                    100.0 * (rows.iloc[index]["mean"] - 1.0),
                    yerr=100.0 * rows.iloc[index]["ci95"],
                    marker=style["marker"],
                    linestyle="none",
                    color=style["color"],
                    capsize=3,
                    markersize=6,
                )
            axis.axhline(0.0, color=INK, linestyle="--", linewidth=1)
            axis.set_xticks(x)
            axis.set_xticklabels(gauges, rotation=32, ha="right")
            axis.set_title(f"train={train_count} | {probe}")
            axis.grid(axis="y", color=LIGHT_GREY, alpha=0.7)
    axes[0, 0].set_ylabel("relative MSE change vs identity, % (95% CI)")
    if len(train_counts) > 1:
        axes[1, 0].set_ylabel("relative MSE change vs identity, % (95% CI)")
    seed_count = int(final_summary["count"].max())
    fig.suptitle(
        f"Gauge comparison\nPaired ImageNet validation, n={seed_count} seeds",
        color=INK,
    )
    fig.tight_layout()
    return fig


def plot_learning_summary(
    learning_summary: pd.DataFrame,
    *,
    probe: str = "local_rf5",
    gauges: Sequence[str] = (
        "identity",
        "allpass_plus",
        "allpass_minus",
        "allpass_r3",
        "channel_plus",
        "channel_minus",
        "haar_2x2",
    ),
) -> plt.Figure:
    train_counts = sorted(learning_summary["train_count"].unique())
    fig, axes = plt.subplots(1, len(train_counts), figsize=(7 * len(train_counts), 4.8), squeeze=False, sharey=True)
    for column, train_count in enumerate(train_counts):
        axis = axes[0, column]
        for gauge in gauges:
            rows = learning_summary[
                (learning_summary["train_count"] == train_count)
                & (learning_summary["probe"] == probe)
                & (learning_summary["gauge"] == gauge)
            ].sort_values("step")
            if rows.empty:
                continue
            style = GAUGE_STYLES[gauge]
            mean_delta = 100.0 * (rows["mean"] - 1.0)
            low_delta = 100.0 * (rows["ci95_low"] - 1.0)
            high_delta = 100.0 * (rows["ci95_high"] - 1.0)
            axis.plot(
                rows["step"],
                mean_delta,
                marker=style["marker"],
                linestyle=style["linestyle"],
                color=style["color"],
                label=gauge,
            )
            axis.fill_between(
                rows["step"],
                low_delta,
                high_delta,
                color=style["color"],
                alpha=0.12,
            )
        axis.axhline(0.0, color=INK, linestyle="--", linewidth=1)
        axis.set_title(f"train={train_count} | {probe}")
        axis.set_xlabel("training step")
        axis.grid(axis="y", color=LIGHT_GREY, alpha=0.7)
    axes[0, 0].set_ylabel("relative MSE change vs identity, % (95% CI)")
    axes[0, -1].legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    seed_count = int(learning_summary["count"].max())
    fig.suptitle(
        f"Finite-horizon coordinate sensitivity\nPaired ImageNet validation, n={seed_count} seeds",
        color=INK,
    )
    fig.tight_layout()
    return fig


def plot_locality_summary(locality_summary: pd.DataFrame) -> plt.Figure:
    train_counts = sorted(locality_summary["train_count"].unique())
    gauges = [name for name in ("allpass_plus", "allpass_minus", "allpass_r3") if name in set(locality_summary["gauge"])]
    fig, axes = plt.subplots(1, len(train_counts), figsize=(7 * len(train_counts), 4.8), squeeze=False, sharey=True)
    for column, train_count in enumerate(train_counts):
        axis = axes[0, column]
        for gauge in gauges:
            rows = locality_summary[
                (locality_summary["train_count"] == train_count)
                & (locality_summary["gauge"] == gauge)
            ].sort_values("receptive_field")
            style = GAUGE_STYLES[gauge]
            axis.errorbar(
                rows["receptive_field"],
                100.0 * rows["mean"],
                yerr=100.0 * rows["ci95"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2,
                capsize=3,
                color=style["color"],
                label=gauge,
            )
        axis.axhline(0.0, color=INK, linestyle="--", linewidth=1)
        axis.set_xticks([5, 9, 17])
        axis.set_xlabel("local probe receptive field")
        axis.set_title(f"train={train_count}")
        axis.grid(axis="y", color=LIGHT_GREY, alpha=0.7)
    axes[0, 0].set_ylabel("local minus global ratio, percentage points (95% CI)")
    axes[0, -1].legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    seed_count = int(locality_summary["count"].max())
    fig.suptitle(
        f"All-pass locality penalty\nPaired ImageNet validation, n={seed_count} seeds",
        color=INK,
    )
    fig.tight_layout()
    return fig


def plot_time_summary(
    time_summary: pd.DataFrame,
    *,
    probe: str = "global_attn",
    gauges: Sequence[str] = ("allpass_r3", "haar_2x2"),
) -> plt.Figure:
    train_counts = sorted(time_summary["train_count"].unique())
    fig, axes = plt.subplots(
        len(gauges),
        len(train_counts),
        figsize=(7 * len(train_counts), 4.0 * len(gauges)),
        squeeze=False,
        sharey="row",
    )
    for row, gauge in enumerate(gauges):
        for column, train_count in enumerate(train_counts):
            axis = axes[row, column]
            rows = time_summary[
                (time_summary["train_count"] == train_count)
                & (time_summary["probe"] == probe)
                & (time_summary["gauge"] == gauge)
            ].sort_values("t_center")
            if rows.empty:
                continue
            style = GAUGE_STYLES[gauge]
            axis.errorbar(
                rows["t_center"],
                100.0 * (rows["mean"] - 1.0),
                yerr=100.0 * rows["ci95"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2,
                capsize=3,
                color=style["color"],
                label=gauge,
            )
            axis.axhline(0.0, color=INK, linestyle="--", linewidth=1)
            axis.set_xlabel("diffusion time t (0=data, 1=noise)")
            axis.set_title(f"train={train_count} | {probe} | {gauge}")
            axis.grid(axis="y", color=LIGHT_GREY, alpha=0.7)
        axes[row, 0].set_ylabel("relative MSE change vs identity, % (95% CI)")
    seed_count = int(time_summary["count"].max())
    fig.suptitle(
        f"Time-dependent coordinate sensitivity\nPaired ImageNet validation, n={seed_count} seeds",
        color=INK,
    )
    fig.tight_layout()
    return fig


def save_summary_figures(tables: Dict[str, pd.DataFrame], result_dir: str | Path) -> None:
    figure_dir = Path(result_dir).expanduser() / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure = plot_final_seed_summary(tables["final_summary"])
    figure.savefig(figure_dir / "final_seed_summary.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    for probe in ("local_rf5", "local_rf9", "local_rf17", "global_attn"):
        figure = plot_learning_summary(tables["learning_summary"], probe=probe)
        figure.savefig(figure_dir / f"learning_{probe}.png", dpi=180, bbox_inches="tight")
        plt.close(figure)
    figure = plot_locality_summary(tables["locality_summary"])
    figure.savefig(figure_dir / "locality_penalty.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    figure = plot_time_summary(tables["time_summary"])
    figure.savefig(figure_dir / "time_global_attn.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_config(args: argparse.Namespace) -> LargeValidationConfig:
    eval_steps = tuple(sorted(set(args.eval_steps + [0, args.steps])))
    return LargeValidationConfig(
        train_counts=tuple(args.train_counts),
        val_count=args.val_count,
        steps=args.steps,
        eval_steps=eval_steps,
        batch_size=args.batch_size,
        seeds=tuple(args.seeds),
        hidden=args.hidden,
        time_bins=args.time_bins,
        dataset_path=args.dataset_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Four-GPU orthogonal gauge validation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(current: argparse.ArgumentParser) -> None:
        current.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
        current.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
        current.add_argument("--train-counts", type=int, nargs="+", default=[1024, 2048])
        current.add_argument("--val-count", type=int, default=512)
        current.add_argument("--steps", type=int, default=2000)
        current.add_argument("--eval-steps", type=int, nargs="+", default=[100, 500, 1000])
        current.add_argument("--batch-size", type=int, default=32)
        current.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
        current.add_argument("--hidden", type=int, default=16)
        current.add_argument("--time-bins", type=int, default=8)
        current.add_argument("--dataset-path", default="/data/shared/imagenet-1k")

    prepare_parser = subparsers.add_parser("prepare")
    add_common(prepare_parser)
    prepare_parser.add_argument("--device", default="cuda:0")
    prepare_parser.add_argument("--force", action="store_true")

    run_parser = subparsers.add_parser("run")
    add_common(run_parser)
    run_parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    run_parser.add_argument("--force", action="store_true")

    aggregate_parser = subparsers.add_parser("aggregate")
    add_common(aggregate_parser)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = parse_config(args)
    if args.command == "prepare":
        prepare_latent_cache(args.cache, config=config, device=args.device, force=args.force)
    elif args.command == "run":
        run_large_validation(
            args.cache,
            args.result_dir,
            config=config,
            gpus=args.gpus,
            force=args.force,
        )
    elif args.command == "aggregate":
        tables = aggregate_results(args.result_dir)
        save_summary_figures(tables, args.result_dir)
        print(f"saved tables and figures under {args.result_dir}")


if __name__ == "__main__":
    main()
