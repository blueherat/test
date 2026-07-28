"""No-training generated-latent closure study for the floor path candidate."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RAE_SRC = ROOT / "external/RAE/src"
for import_path in (ROOT, RAE_SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from experiments.rae_teacher_rollout_gap import configure_fp32, official_time_grid  # noqa: E402
from experiments.run_rae_decoder_risk_phase0 import (  # noqa: E402
    DEFAULT_AUDIT_CACHE,
    _closure_rows,
    _load_cache_tensor,
    _load_full_rae,
    _sample_endpoints,
)
from utils.model_utils import instantiate_from_config  # noqa: E402


BASELINE_ROOT = Path.home() / "data/eqvae/experiments/rae_layerwise_path_train"
CANDIDATE_ROOT = Path.home() / "data/eqvae/experiments/rae_path_schedule_train"
DEFAULT_OUTPUT = Path.home() / "data/eqvae/experiments/rae_path_schedule_closure"
CROSSOVER_ROOT = Path.home() / "data/eqvae/experiments/rae_path_crossover_train_v2"
CONDITIONS = (
    ("static", BASELINE_ROOT / "seed3407_static_rank16_s0_to_10000"),
    ("annealed", BASELINE_ROOT / "seed3407_annealed_rank16_s0_to_10000"),
    ("floor020_p2", CANDIDATE_ROOT / "seed3407_floor020_p2_rank16_s0_to_2000"),
)
CROSSOVER_CONDITIONS = tuple(
    (
        condition,
        CROSSOVER_ROOT / f"seed3407_{condition}_rank16_s2000_to_5000",
    )
    for condition in (
        "floor_to_floor",
        "floor_to_static",
        "static_to_static",
        "static_to_floor",
    )
)
FID_5K = {
    "static": 229.67234904828905,
    "annealed": 267.85094544081,
    "floor020_p2": 267.02965137245855,
}


def evaluate_closure_prediction(summary: pd.DataFrame) -> dict[str, object]:
    metrics = ("cycle_relative_rms_median", "local_decoder_sensitivity_median")

    def value(source: str, metric: str) -> float:
        row = summary[summary.source == source]
        if len(row) != 1:
            raise ValueError(f"expected one summary row for {source}")
        return float(row.iloc[0][metric])

    p1 = all(
        value("static", metric)
        < min(value("annealed", metric), value("floor020_p2", metric))
        for metric in metrics
    )
    ratios = {
        metric: value("floor020_p2", metric) / value("annealed", metric)
        for metric in metrics
    }
    p2 = all(0.95 <= ratio <= 1.05 for ratio in ratios.values())
    positions = {
        metric: (
            (value("floor020_p2", metric) - value("static", metric))
            / (value("annealed", metric) - value("static", metric))
        )
        for metric in metrics
    }
    p3 = all(0.75 <= position <= 1.25 for position in positions.values())
    p4 = all(
        value(source, metric) > value("clean_test", metric)
        for source in ("static", "annealed", "floor020_p2")
        for metric in metrics
    )
    predictions = {
        "p1_static_lowest": bool(p1),
        "p2_floor_matches_annealed": bool(p2),
        "p3_floor_relative_position": bool(p3),
        "p4_generated_worse_than_clean": bool(p4),
    }
    return {
        "pass": bool(all(predictions.values())),
        "predictions": predictions,
        "details": {
            "floor_over_annealed": ratios,
            "floor_position_static_to_annealed": positions,
            "fid_5k": FID_5K,
            "fid_floor_position": (
                (FID_5K["floor020_p2"] - FID_5K["static"])
                / (FID_5K["annealed"] - FID_5K["static"])
            ),
        },
    }


def evaluate_crossover_closure(summary: pd.DataFrame) -> dict[str, object]:
    metrics = ("cycle_relative_rms_median", "local_decoder_sensitivity_median")

    def value(source: str, metric: str) -> float:
        row = summary[summary.source == source]
        if len(row) != 1:
            raise ValueError(f"expected one summary row for {source}")
        return float(row.iloc[0][metric])

    effects = {}
    movements = {}
    for metric in metrics:
        ff = value("floor_to_floor", metric)
        fs = value("floor_to_static", metric)
        ss = value("static_to_static", metric)
        sf = value("static_to_floor", metric)
        effects[metric] = {
            "late_floor_with_early_floor": ff - fs,
            "late_floor_with_early_static": sf - ss,
            "difference_in_differences": (sf - ss) - (ff - fs),
        }
        movements[metric] = {
            "floor_to_static_closer_to_late_control": abs(fs - ss) < abs(fs - ff),
            "static_to_floor_closer_to_late_control": abs(sf - ff) < abs(sf - ss),
        }
    directions = {
        "floor_to_static_improves_both": all(
            value("floor_to_static", metric) < value("floor_to_floor", metric)
            for metric in metrics
        ),
        "static_to_floor_worsens_both": all(
            value("static_to_floor", metric) > value("static_to_static", metric)
            for metric in metrics
        ),
    }
    return {
        "directions": {key: bool(value) for key, value in directions.items()},
        "late_path_effects": effects,
        "movement_toward_late_control": movements,
    }


def plot_closure_summary(summary: pd.DataFrame, output: Path) -> None:
    colors = {
        "clean_test": "#9D9DA1",
        "static": "#4C78A8",
        "annealed": "#E45756",
        "floor020_p2": "#54A24B",
    }
    sources = ("clean_test", "static", "annealed", "floor020_p2")
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for axis, metric, title in (
        (axes[0], "cycle_relative_rms_median", "Cycle residual"),
        (axes[1], "local_decoder_sensitivity_median", "Decoder local sensitivity"),
    ):
        values = [float(summary[summary.source == source].iloc[0][metric]) for source in sources]
        axis.bar(sources, values, color=[colors[source] for source in sources])
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
    generated = ("static", "annealed", "floor020_p2")
    axes[2].bar(generated, [FID_5K[source] for source in generated], color=[colors[s] for s in generated])
    axes[2].set_title("5k FID screen")
    axes[2].tick_params(axis="x", rotation=20)
    axes[2].grid(axis="y", alpha=0.25)
    figure.savefig(output / "closure_vs_generation.png", dpi=180)
    plt.close(figure)


def plot_crossover_closure(summary: pd.DataFrame, output: Path) -> None:
    sources = (
        "clean_test",
        "floor_to_floor",
        "floor_to_static",
        "static_to_static",
        "static_to_floor",
    )
    colors = ["#9D9DA1", "#E45756", "#72B7B2", "#4C78A8", "#F2CF5B"]
    figure, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    for axis, metric, title in (
        (axes[0], "cycle_relative_rms_median", "Cycle residual"),
        (axes[1], "local_decoder_sensitivity_median", "Decoder local sensitivity"),
    ):
        values = [
            float(summary[summary.source == source].iloc[0][metric])
            for source in sources
        ]
        axis.bar(sources, values, color=colors)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(output / "crossover_closure.png", dpi=180)
    plt.close(figure)


def _load_stage2_model(
    branch: Path, step: int, device: torch.device, weight_source: str = "ema"
) -> tuple[torch.nn.Module, OmegaConf]:
    if weight_source not in {"ema", "model"}:
        raise ValueError(f"unknown weight source: {weight_source}")
    config = OmegaConf.load(branch / "config.yaml")
    model = instantiate_from_config(config.stage_2).to(device=device, dtype=torch.float32)
    materialized = branch / f"generation/{weight_source}_step-{int(step):07d}.pt"
    if materialized.exists():
        state = torch.load(materialized, map_location="cpu", weights_only=True)
    else:
        checkpoint = torch.load(
            branch / f"checkpoints/step-{int(step):07d}.pt",
            map_location="cpu",
            weights_only=False,
        )
        state = checkpoint[weight_source]
    model.load_state_dict(state, strict=True)
    return model.requires_grad_(False).eval(), config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-name", default="step5000_n64_seed20260726")
    parser.add_argument("--step", type=int, default=5000)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--sampling-batch-size", type=int, default=8)
    parser.add_argument("--closure-batch-size", type=int, default=4)
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--reference-count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20_260_726)
    parser.add_argument("--perturb-fraction", type=float, default=1e-3)
    parser.add_argument("--audit-cache", type=Path, default=DEFAULT_AUDIT_CACHE)
    parser.add_argument("--study", choices=("original", "crossover"), default="original")
    parser.add_argument("--weight-source", choices=("ema", "model"), default="ema")
    args = parser.parse_args()
    conditions = CONDITIONS if args.study == "original" else CROSSOVER_CONDITIONS
    if "RANK" not in os.environ or not torch.cuda.is_available():
        raise RuntimeError(f"launch with torchrun on exactly {len(conditions)} GPUs")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    if dist.get_world_size() != len(conditions):
        raise ValueError(f"expected exactly {len(conditions)} processes")
    configure_fp32(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    condition, branch = conditions[rank]
    output = args.output.expanduser().resolve() / args.run_name
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    model, config = _load_stage2_model(
        branch, args.step, device, weight_source=args.weight_source
    )
    shift = math.sqrt(
        float(config.misc.time_dist_shift_dim) / float(config.misc.time_dist_shift_base)
    )
    times = official_time_grid(args.sampling_steps, time_shift=shift).to(device)
    endpoints, labels = _sample_endpoints(
        model,
        count=args.count,
        batch_size=args.sampling_batch_size,
        times=times,
        seed=args.seed,
        device=device,
    )
    torch.save(
        {
            "latents": endpoints,
            "labels": labels,
            "condition": condition,
            "step": int(args.step),
            "sampling_steps": int(args.sampling_steps),
            "seed": int(args.seed),
        },
        output / f"generated_latents_rank{rank:02d}.pt",
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    rae = _load_full_rae(config, device)
    clean_reference = _load_cache_tensor(
        args.audit_cache.expanduser(), "calibration", args.reference_count
    )
    rows = _closure_rows(
        rae,
        endpoints,
        clean_reference,
        source=condition,
        batch_size=args.closure_batch_size,
        perturb_fraction=args.perturb_fraction,
        seed=args.seed,
        device=device,
    )
    if rank == 0:
        clean_query = _load_cache_tensor(args.audit_cache.expanduser(), "test", args.count)
        rows.extend(
            _closure_rows(
                rae,
                clean_query,
                clean_reference,
                source="clean_test",
                batch_size=args.closure_batch_size,
                perturb_fraction=args.perturb_fraction,
                seed=args.seed,
                device=device,
            )
        )
    pd.DataFrame(rows).to_csv(output / f"closure_rank{rank:02d}.csv", index=False)
    print(f"rank{rank} {condition} closure complete", flush=True)
    dist.barrier()
    if rank == 0:
        table = pd.concat(
            [
                pd.read_csv(output / f"closure_rank{index:02d}.csv")
                for index in range(len(conditions))
            ],
            ignore_index=True,
        )
        table.to_csv(output / "closure_metrics.csv", index=False)
        metric_columns = [
            "cycle_relative_rms",
            "projected_nearest_clean_distance",
            "local_decoder_sensitivity",
            "decoded_pixel_clipping_fraction",
        ] + [column for column in table if column.startswith("decoder_hidden_rms_z_layer")]
        summary = (
            table.groupby("source")[metric_columns]
            .agg(["median", "mean", "std"])
            .reset_index()
        )
        summary.columns = [
            column if isinstance(column, str) else "_".join(v for v in column if v)
            for column in summary.columns
        ]
        summary.to_csv(output / "closure_summary.csv", index=False)
        decision = (
            evaluate_closure_prediction(summary)
            if args.study == "original"
            else evaluate_crossover_closure(summary)
        )
        (output / "decision.json").write_text(
            json.dumps(decision, indent=2), encoding="utf-8"
        )
        if args.study == "original":
            plot_closure_summary(summary, output)
        else:
            plot_crossover_closure(summary, output)
        print(summary.to_string(index=False), flush=True)
        print(json.dumps(decision, indent=2), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
