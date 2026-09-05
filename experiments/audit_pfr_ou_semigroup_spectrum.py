#!/usr/bin/env python3
"""Test whether useful PFR revisions resemble an OU spectral innovation."""

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

from experiments.pfr_ou_semigroup_spectrum import (  # noqa: E402
    linear_velocity_to_ou_relative_score,
    ou_mode_retiming_defect,
    ou_relative_score_delta_to_linear_velocity_delta,
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
    if any(not 0.0 < item < INTERVENTION_TIME for item in result):
        raise argparse.ArgumentTypeError("times must lie in (0, 0.5)")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument(
        "--times",
        type=parse_times,
        default=parse_times("0.05,0.1,0.2,0.3,0.4,0.46875"),
    )
    parser.add_argument("--horizon", type=float, default=HORIZON)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--cuda-allocator-limit-gib", type=float, default=6.0)
    return parser.parse_args()


def sample_rms(value: torch.Tensor) -> torch.Tensor:
    return value.float().flatten(1).square().mean(1).sqrt()


def sample_cosine(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first_flat = first.float().flatten(1)
    second_flat = second.float().flatten(1)
    return (first_flat * second_flat).sum(1) / (
        first_flat.norm(dim=1) * second_flat.norm(dim=1)
    ).clamp_min(1e-30)


def summarize_tensor(value: torch.Tensor) -> dict[str, float]:
    summary = summarize(value.detach().float().cpu().tolist())
    if summary is None:
        raise RuntimeError("cannot summarize an empty tensor")
    return {name: float(item) for name, item in summary.items()}


@torch.inference_mode()
def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.samples <= 0 or args.batch_size <= 0:
        raise ValueError("samples and batch-size must be positive")
    if not 0.0 < args.horizon < INTERVENTION_TIME:
        raise ValueError("horizon must lie in (0, 0.5)")

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
        runtime,
        labels,
        "ordinary_ig",
        record_diagnostics=False,
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
        horizon = min(float(args.horizon), INTERVENTION_TIME - time_value)
        future_value = time_value + horizon
        metrics: dict[str, list[torch.Tensor]] = {}
        for start in range(0, args.samples, args.batch_size):
            stop = min(start + args.batch_size, args.samples)
            state = all_state[start:stop]
            batch_labels = labels[start:stop]
            time = torch.full((len(state),), time_value, device=device)
            future_time = torch.full((len(state),), future_value, device=device)

            strong, weak = runtime.evaluate_pair(time, state, batch_labels)
            strong_same, weak_same = runtime.evaluate_pair(
                future_time, state, batch_labels
            )
            ou_future_state = transport_state_at_fixed_ou_coordinate(
                state, time, future_time
            )
            strong_ou, weak_ou = runtime.evaluate_pair(
                future_time, ou_future_state, batch_labels
            )

            pfr_revision = weak - weak_same
            weak_relative = linear_velocity_to_ou_relative_score(
                weak, state, time
            )
            strong_relative = linear_velocity_to_ou_relative_score(
                strong, state, time
            )
            weak_future_relative = linear_velocity_to_ou_relative_score(
                weak_ou, ou_future_state, future_time
            )
            strong_future_relative = linear_velocity_to_ou_relative_score(
                strong_ou, ou_future_state, future_time
            )

            candidate_metrics: dict[str, torch.Tensor] = {
                "pfr_revision_rms": sample_rms(pfr_revision),
                "ou_state_shift_rms": sample_rms(ou_future_state - state),
                "current_weak_relative_score_rms": sample_rms(weak_relative),
            }
            for degree in (1.0, 2.0):
                suffix = f"degree{int(degree)}"
                weak_defect = ou_mode_retiming_defect(
                    weak_relative,
                    weak_future_relative,
                    time,
                    future_time,
                    degree=degree,
                )
                strong_defect = ou_mode_retiming_defect(
                    strong_relative,
                    strong_future_relative,
                    time,
                    future_time,
                    degree=degree,
                )
                weak_velocity_defect = (
                    ou_relative_score_delta_to_linear_velocity_delta(
                        weak_defect, state, time
                    )
                )
                strong_velocity_defect = (
                    ou_relative_score_delta_to_linear_velocity_delta(
                        strong_defect, state, time
                    )
                )
                candidate_metrics.update(
                    {
                        f"{suffix}_weak_defect_rms": sample_rms(
                            weak_velocity_defect
                        ),
                        f"{suffix}_strong_defect_rms": sample_rms(
                            strong_velocity_defect
                        ),
                        f"{suffix}_pfr_cosine": sample_cosine(
                            weak_velocity_defect, pfr_revision
                        ),
                        f"{suffix}_weak_strong_cosine": sample_cosine(
                            weak_velocity_defect, strong_velocity_defect
                        ),
                        f"{suffix}_relative_to_pfr_rms": (
                            sample_rms(weak_velocity_defect)
                            / sample_rms(pfr_revision).clamp_min(1e-30)
                        ),
                    }
                )

            for name, value in candidate_metrics.items():
                metrics.setdefault(name, []).append(value.detach().float().cpu())
            cpu_metrics = {
                name: value.detach().float().cpu()
                for name, value in candidate_metrics.items()
            }
            for offset in range(len(state)):
                per_sample_rows.append(
                    {
                        "time": time_value,
                        "future_time": future_value,
                        "sample": start + offset,
                        **{
                            name: float(value[offset])
                            for name, value in cpu_metrics.items()
                        },
                    }
                )

        row: dict[str, Any] = {
            "time": time_value,
            "future_time": future_value,
            "horizon": horizon,
            "samples": args.samples,
        }
        for name, chunks in metrics.items():
            summary = summarize_tensor(torch.cat(chunks))
            for statistic, value in summary.items():
                row[f"{name}_{statistic}"] = value
        rows.append(row)
        print(
            json.dumps(
                {
                    "event": "time_complete",
                    "time": time_value,
                    "degree1_pfr_cosine": row["degree1_pfr_cosine_mean"],
                    "degree1_weak_strong_cosine": row[
                        "degree1_weak_strong_cosine_mean"
                    ],
                    "degree2_pfr_cosine": row["degree2_pfr_cosine_mean"],
                }
            ),
            flush=True,
        )

    for path, data_rows in (
        (output / "ou_spectrum_summary.csv", rows),
        (output / "per_sample_ou_spectrum.csv", per_sample_rows),
    ):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data_rows[0]))
            writer.writeheader()
            writer.writerows(data_rows)

    manifest = {
        "format": "eqvae_pfr_ou_semigroup_spectrum_audit_v1",
        "scope": "geometry screen only; no claim that finite neural scores are exact OU modes",
        "protocol": {
            "samples": args.samples,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "times": list(args.times),
            "horizon": args.horizon,
            "trajectory": "ordinary depth4 internal guidance",
            "future_query": "same normalized OU coordinate",
            "strong": str(runtime.paths["strong"]),
            "weak": str(runtime.paths["depth4"]),
            "weights": "ema",
        },
        "candidate": {
            "predeclared_degree": 1,
            "degree2_role": "descriptive centered/covariance control",
            "continue_gate": "stable positive degree1 cosine with useful raw PFR revision",
        },
        "allocator": allocator,
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "summary": str(output / "ou_spectrum_summary.csv"),
        "per_sample": str(output / "per_sample_ou_spectrum.csv"),
    }
    atomic_json(output / "manifest.json", manifest)
    print(json.dumps({"event": "complete", "output": str(output)}), flush=True)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
