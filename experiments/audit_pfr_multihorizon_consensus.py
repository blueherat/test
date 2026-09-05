#!/usr/bin/env python3
"""Test whether weak-head agreement across horizons predicts strong consensus."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.internal_guidance_path_extrapolation import (  # noqa: E402
    project_per_sample,
)
from experiments.run_imagenet100_sit_internal_early_two_segment_gamma_sweep import (  # noqa: E402
    atomic_json,
    detect_adm_python,
    detect_data,
    detect_repo,
)
from experiments.run_imagenet100_sit_path_evidence_pfr_bridge import (  # noqa: E402
    HORIZON,
    INTERVENTION_TIME,
    load_runtime,
)
from experiments.run_imagenet100_sit_pfr_query_controls import (  # noqa: E402
    QueryControlledField,
    integrate_times,
    summarize,
)


def parse_times(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("times must be comma-separated floats") from error
    if not result or tuple(sorted(set(result))) != result:
        raise argparse.ArgumentTypeError("times must be unique and increasing")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument(
        "--times",
        type=parse_times,
        default=parse_times("0.05,0.1,0.2,0.3,0.4,0.4375"),
    )
    parser.add_argument("--horizon", type=float, default=HORIZON)
    parser.add_argument("--long-multiplier", type=float, default=2.0)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--cuda-allocator-limit-gib", type=float, default=6.0)
    return parser.parse_args()


def sample_rms(value: torch.Tensor) -> torch.Tensor:
    return value.float().flatten(1).square().mean(1).sqrt()


def sample_cosine(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first_flat = first.float().flatten(1)
    second_flat = second.float().flatten(1)
    denominator = first_flat.norm(dim=1) * second_flat.norm(dim=1)
    return (first_flat * second_flat).sum(1) / denominator.clamp_min(1e-30)


def append(values: dict[str, list[torch.Tensor]], name: str, value: torch.Tensor) -> None:
    values.setdefault(name, []).append(value.detach().float().cpu())


def summarize_metrics(values: dict[str, list[torch.Tensor]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, chunks in values.items():
        summary = summarize(torch.cat(chunks).tolist())
        if summary is None:
            raise RuntimeError(f"empty metric: {name}")
        for statistic, value in summary.items():
            result[f"{name}_{statistic}"] = float(value)
    return result


@torch.inference_mode()
def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.samples <= 0 or args.batch_size <= 0:
        raise ValueError("samples and batch-size must be positive")
    if args.horizon <= 0.0 or args.long_multiplier <= 1.0:
        raise ValueError("horizon must be positive and long-multiplier greater than one")
    long_horizon = args.horizon * args.long_multiplier
    if any(t <= 0.0 or t + long_horizon > INTERVENTION_TIME for t in args.times):
        raise ValueError("all times must leave room for the complete long horizon")

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    runtime, allocator = load_runtime(
        repo=detect_repo(),
        data=detect_data(),
        adm_python=detect_adm_python(),
        device=device,
        allocator_limit_gib=args.cuda_allocator_limit_gib,
    )

    generator = torch.Generator(device=device).manual_seed(args.seed)
    noise = torch.randn(
        args.samples,
        *runtime.modules["LATENT_SHAPE"],
        generator=generator,
        device=device,
    )
    labels = torch.randint(
        0,
        runtime.modules["NUM_CLASSES"],
        (args.samples,),
        generator=generator,
        device=device,
    )
    ordinary = QueryControlledField(
        runtime, labels, "ordinary_ig", record_diagnostics=False
    )
    states = integrate_times(
        ordinary,
        noise.float(),
        args.times,
        atol=args.atol,
        rtol=args.rtol,
    )

    rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for time_value, all_state in zip(args.times, states, strict=True):
        metrics: dict[str, list[torch.Tensor]] = {}
        for start in range(0, args.samples, args.batch_size):
            stop = min(start + args.batch_size, args.samples)
            state = all_state[start:stop]
            batch_labels = labels[start:stop]
            time = torch.full((len(state),), time_value, device=device)
            short_time = torch.full(
                (len(state),), time_value + args.horizon, device=device
            )
            long_time = torch.full(
                (len(state),), time_value + long_horizon, device=device
            )
            strong, weak = runtime.evaluate_pair(time, state, batch_labels)
            strong_short, weak_short = runtime.evaluate_pair(
                short_time, state, batch_labels
            )
            weak_long = runtime.evaluate_weak(long_time, state, batch_labels)

            weak_revision = weak - weak_short
            strong_revision = strong - strong_short
            long_revision = (weak - weak_long) / args.long_multiplier
            true_projection = project_per_sample(weak_revision, strong_revision)
            horizon_projection = project_per_sample(weak_revision, long_revision)

            batch_metrics = {
                "weak_strong_cosine": sample_cosine(
                    weak_revision, strong_revision
                ),
                "weak_long_cosine": sample_cosine(weak_revision, long_revision),
                "strong_long_cosine": sample_cosine(
                    strong_revision, long_revision
                ),
                "true_common_rms": sample_rms(true_projection.parallel),
                "horizon_common_rms": sample_rms(horizon_projection.parallel),
                "horizon_common_true_common_cosine": sample_cosine(
                    horizon_projection.parallel, true_projection.parallel
                ),
                "horizon_common_strong_cosine": sample_cosine(
                    horizon_projection.parallel, strong_revision
                ),
                "true_common_energy_fraction": (
                    sample_rms(true_projection.parallel).square()
                    / sample_rms(weak_revision).square().clamp_min(1e-30)
                ),
                "horizon_common_energy_fraction": (
                    sample_rms(horizon_projection.parallel).square()
                    / sample_rms(weak_revision).square().clamp_min(1e-30)
                ),
                "horizon_unique_true_common_cosine": sample_cosine(
                    horizon_projection.orthogonal, true_projection.parallel
                ),
                "horizon_common_true_difference_rms": sample_rms(
                    horizon_projection.parallel - true_projection.parallel
                ),
            }
            for name, value in batch_metrics.items():
                append(metrics, name, value)
            cpu_metrics = {
                name: value.detach().float().cpu()
                for name, value in batch_metrics.items()
            }
            for offset in range(len(state)):
                sample_rows.append(
                    {
                        "time": time_value,
                        "sample": start + offset,
                        **{
                            name: float(value[offset])
                            for name, value in cpu_metrics.items()
                        },
                    }
                )

        row: dict[str, Any] = {
            "time": time_value,
            "short_time": time_value + args.horizon,
            "long_time": time_value + long_horizon,
            "samples": args.samples,
        }
        row.update(summarize_metrics(metrics))
        rows.append(row)
        print(
            json.dumps(
                {
                    "event": "time_complete",
                    "time": time_value,
                    "weak_strong_cosine": row["weak_strong_cosine_mean"],
                    "weak_long_cosine": row["weak_long_cosine_mean"],
                    "proxy_true_cosine": row[
                        "horizon_common_true_common_cosine_mean"
                    ],
                }
            ),
            flush=True,
        )

    for path, data_rows in (
        (output / "multihorizon_summary.csv", rows),
        (output / "per_sample_multihorizon.csv", sample_rows),
    ):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data_rows[0]))
            writer.writeheader()
            writer.writerows(data_rows)

    atomic_json(
        output / "manifest.json",
        {
            "format": "eqvae_pfr_multihorizon_consensus_audit_v1",
            "question": (
                "Can cheap weak-head agreement across horizons predict the "
                "weak/strong shared retiming component?"
            ),
            "protocol": {
                "samples": args.samples,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "times": list(args.times),
                "short_horizon": args.horizon,
                "long_horizon": long_horizon,
                "trajectory": "ordinary depth4 internal guidance",
                "query": "same latent at two later affine-flow times",
                "weights": "ema",
            },
            "allocator": allocator,
            "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "summary": str(output / "multihorizon_summary.csv"),
            "per_sample": str(output / "per_sample_multihorizon.csv"),
        },
    )
    print(json.dumps({"event": "complete", "output": str(output)}), flush=True)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
