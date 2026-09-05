#!/usr/bin/env python3
"""Test whether OU degree-1 defects collapse to one leading spectral mode."""

from __future__ import annotations

import argparse
import csv
import json
import math
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
from experiments.pfr_ou_semigroup_spectrum import (  # noqa: E402
    ou_degree1_retiming_velocity_defect,
    transport_state_at_fixed_ou_coordinate,
)
from experiments.run_imagenet100_sit_internal_early_two_segment_gamma_sweep import (  # noqa: E402
    atomic_json,
    detect_adm_python,
    detect_data,
    detect_repo,
)
from experiments.run_imagenet100_sit_path_evidence_pfr_bridge import (  # noqa: E402
    HORIZON,
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
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument(
        "--times",
        type=parse_times,
        default=parse_times("0.02,0.05,0.1,0.2,0.3,0.4"),
    )
    parser.add_argument("--horizon", type=float, default=HORIZON)
    parser.add_argument("--long-multiplier", type=float, default=2.0)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--cuda-allocator-limit-gib", type=float, default=6.0)
    return parser.parse_args()


def ou_degree_response_ratio(
    time: float,
    short_time: float,
    long_time: float,
    degree: float,
) -> float:
    """Return the long/short D1 amplitude ratio for one Hermite degree."""

    if not 1.0 < degree or not 0.0 < time < short_time < long_time < 1.0:
        raise ValueError("require degree > 1 and 0 < time < short < long < 1")
    signal = lambda value: value / math.sqrt(value * value + (1.0 - value) ** 2)
    current = signal(time)
    short = signal(short_time)
    long = signal(long_time)
    exponent = degree - 1.0
    short_response = short**exponent - current**exponent
    long_response = long**exponent - current**exponent
    return long_response / short_response


def infer_effective_degree(
    ratio: float,
    time: float,
    short_time: float,
    long_time: float,
    *,
    minimum: float = 1.001,
    maximum: float = 12.0,
    grid_size: int = 20000,
) -> float:
    """Invert the single-mode horizon ratio on a dense deterministic grid."""

    if not math.isfinite(ratio) or ratio <= 0.0 or grid_size < 2:
        return math.nan
    degrees = torch.linspace(minimum, maximum, grid_size, dtype=torch.float64)
    signal = lambda value: value / math.sqrt(value * value + (1.0 - value) ** 2)
    current = signal(time)
    short = signal(short_time)
    long = signal(long_time)
    exponents = degrees - 1.0
    current_power = torch.exp(exponents * math.log(current))
    short_response = torch.exp(exponents * math.log(short)) - current_power
    long_response = torch.exp(exponents * math.log(long)) - current_power
    predicted = long_response / short_response
    index = int(torch.argmin((predicted.log() - math.log(ratio)).abs()))
    return float(degrees[index])


def sample_rms(value: torch.Tensor) -> torch.Tensor:
    return value.float().flatten(1).square().mean(1).sqrt()


def sample_cosine(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first_flat = first.float().flatten(1)
    second_flat = second.float().flatten(1)
    return (first_flat * second_flat).sum(1) / (
        first_flat.norm(dim=1) * second_flat.norm(dim=1)
    ).clamp_min(1e-30)


def summarize_tensor(value: torch.Tensor) -> dict[str, float]:
    result = summarize(value.detach().float().cpu().tolist())
    if result is None:
        raise RuntimeError("cannot summarize an empty tensor")
    return {key: float(item) for key, item in result.items()}


@torch.inference_mode()
def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.samples <= 0 or args.batch_size <= 0:
        raise ValueError("samples and batch-size must be positive")
    if args.horizon <= 0.0 or args.long_multiplier <= 1.0:
        raise ValueError("horizon must be positive and long-multiplier greater than one")
    long_horizon = args.horizon * args.long_multiplier
    if any(time <= 0.0 or time + long_horizon >= 1.0 for time in args.times):
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
    per_sample_rows: list[dict[str, Any]] = []
    for time_value, all_state in zip(args.times, states, strict=True):
        short_value = time_value + args.horizon
        long_value = time_value + long_horizon
        metrics: dict[str, list[torch.Tensor]] = {}
        for start in range(0, args.samples, args.batch_size):
            stop = min(start + args.batch_size, args.samples)
            state = all_state[start:stop]
            batch_labels = labels[start:stop]
            time = torch.full((len(state),), time_value, device=device)
            short_time = torch.full((len(state),), short_value, device=device)
            long_time = torch.full((len(state),), long_value, device=device)

            strong, weak = runtime.evaluate_pair(time, state, batch_labels)
            short_state = transport_state_at_fixed_ou_coordinate(
                state, time, short_time
            )
            long_state = transport_state_at_fixed_ou_coordinate(
                state, time, long_time
            )
            strong_short, weak_short = runtime.evaluate_pair(
                short_time, short_state, batch_labels
            )
            strong_long, weak_long = runtime.evaluate_pair(
                long_time, long_state, batch_labels
            )
            weak_short_raw = runtime.evaluate_weak(
                short_time, state, batch_labels
            )
            raw_revision = weak - weak_short_raw

            fields = {
                "weak": (
                    ou_degree1_retiming_velocity_defect(
                        weak, weak_short, state, time, short_time
                    ),
                    ou_degree1_retiming_velocity_defect(
                        weak, weak_long, state, time, long_time
                    ),
                ),
                "strong": (
                    ou_degree1_retiming_velocity_defect(
                        strong, strong_short, state, time, short_time
                    ),
                    ou_degree1_retiming_velocity_defect(
                        strong, strong_long, state, time, long_time
                    ),
                ),
            }
            batch_metrics: dict[str, torch.Tensor] = {}
            degree2_ratio = ou_degree_response_ratio(
                time_value, short_value, long_value, 2.0
            )
            for name, (short_defect, long_defect) in fields.items():
                projection = project_per_sample(long_defect, short_defect)
                raw_projection = project_per_sample(raw_revision, short_defect)
                short_rms = sample_rms(short_defect)
                long_rms = sample_rms(long_defect)
                raw_rms = sample_rms(raw_revision)
                raw_common_rms = sample_rms(raw_projection.parallel)
                batch_metrics.update(
                    {
                        f"{name}_short_long_cosine": sample_cosine(
                            short_defect, long_defect
                        ),
                        f"{name}_long_short_rms_ratio": (
                            long_rms / short_rms.clamp_min(1e-30)
                        ),
                        f"{name}_long_unique_energy_fraction": (
                            sample_rms(projection.orthogonal).square()
                            / long_rms.square().clamp_min(1e-30)
                        ),
                        f"{name}_degree2_relative_residual": (
                            sample_rms(long_defect - degree2_ratio * short_defect)
                            / long_rms.clamp_min(1e-30)
                        ),
                        f"{name}_raw_short_cosine": sample_cosine(
                            raw_revision, short_defect
                        ),
                        f"{name}_raw_explained_energy_fraction": (
                            raw_common_rms.square()
                            / raw_rms.square().clamp_min(1e-30)
                        ),
                        f"{name}_raw_common_to_raw_rms_ratio": (
                            raw_common_rms / raw_rms.clamp_min(1e-30)
                        ),
                        f"{name}_short_to_raw_rms_ratio": (
                            short_rms / raw_rms.clamp_min(1e-30)
                        ),
                    }
                )
            for name, value in batch_metrics.items():
                metrics.setdefault(name, []).append(value.detach().float().cpu())
            cpu_metrics = {
                name: value.detach().float().cpu()
                for name, value in batch_metrics.items()
            }
            for offset in range(len(state)):
                per_sample_rows.append(
                    {
                        "time": time_value,
                        "short_time": short_value,
                        "long_time": long_value,
                        "sample": start + offset,
                        **{
                            name: float(value[offset])
                            for name, value in cpu_metrics.items()
                        },
                    }
                )

        row: dict[str, Any] = {
            "time": time_value,
            "short_time": short_value,
            "long_time": long_value,
            "samples": args.samples,
            "degree2_predicted_ratio": ou_degree_response_ratio(
                time_value, short_value, long_value, 2.0
            ),
            "degree3_predicted_ratio": ou_degree_response_ratio(
                time_value, short_value, long_value, 3.0
            ),
        }
        for name, chunks in metrics.items():
            summary = summarize_tensor(torch.cat(chunks))
            for statistic, value in summary.items():
                row[f"{name}_{statistic}"] = value
        for name in ("weak", "strong"):
            row[f"{name}_effective_degree_from_mean_rms_ratio"] = (
                infer_effective_degree(
                    row[f"{name}_long_short_rms_ratio_mean"],
                    time_value,
                    short_value,
                    long_value,
                )
            )
        rows.append(row)
        print(
            json.dumps(
                {
                    "event": "time_complete",
                    "time": time_value,
                    "weak_cosine": row["weak_short_long_cosine_mean"],
                    "weak_unique_energy": row[
                        "weak_long_unique_energy_fraction_mean"
                    ],
                    "weak_effective_degree": row[
                        "weak_effective_degree_from_mean_rms_ratio"
                    ],
                    "weak_raw_cosine": row["weak_raw_short_cosine_mean"],
                    "strong_raw_cosine": row["strong_raw_short_cosine_mean"],
                    "weak_projection_rms_ratio": row[
                        "weak_raw_common_to_raw_rms_ratio_mean"
                    ],
                    "strong_projection_rms_ratio": row[
                        "strong_raw_common_to_raw_rms_ratio_mean"
                    ],
                }
            ),
            flush=True,
        )

    for path, data_rows in (
        (output / "multiscale_rank_summary.csv", rows),
        (output / "per_sample_multiscale_rank.csv", per_sample_rows),
    ):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data_rows[0]))
            writer.writeheader()
            writer.writerows(data_rows)
    atomic_json(
        output / "manifest.json",
        {
            "format": "eqvae_pfr_ou_multiscale_rank_audit_v1",
            "scope": (
                "geometry audit of the leading-mode hypothesis; effective degree "
                "is descriptive when neural fields mix Hermite modes; raw-revision "
                "projection metrics distinguish certificate direction from the "
                "incidental norm contraction of orthogonal projection"
            ),
            "protocol": {
                "samples": args.samples,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "times": list(args.times),
                "short_horizon": args.horizon,
                "long_horizon": long_horizon,
                "trajectory": "ordinary depth4 internal guidance",
                "query": "same normalized OU coordinate",
                "weights": "ema",
            },
            "allocator": allocator,
            "max_memory_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device)
            ),
            "max_memory_reserved_bytes": int(
                torch.cuda.max_memory_reserved(device)
            ),
            "summary": str(output / "multiscale_rank_summary.csv"),
            "per_sample": str(output / "per_sample_multiscale_rank.csv"),
        },
    )
    print(json.dumps({"event": "complete", "output": str(output)}), flush=True)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
