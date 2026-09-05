#!/usr/bin/env python3
"""Audit the exponential-retiming interpretation on the deployed SiT fields."""

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
from experiments.pfr_exponential_retiming import (  # noqa: E402
    exponential_retiming_defect,
    linear_velocity_to_score,
    reinterpret_future_velocity_score,
    retime_future_score,
    retiming_weight,
    split_exponential_retiming_defect,
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
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260903)
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
    numerator = (first_flat * second_flat).sum(1)
    denominator = first_flat.norm(dim=1) * second_flat.norm(dim=1)
    return numerator / denominator.clamp_min(1e-30)


def relative_rms(error: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return sample_rms(error) / sample_rms(reference).clamp_min(1e-30)


def append_metric(
    values: dict[str, list[torch.Tensor]], name: str, value: torch.Tensor
) -> None:
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
    if not 0.0 < args.horizon < INTERVENTION_TIME:
        raise ValueError("horizon must lie in (0, 0.5)")

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    repo = detect_repo()
    data = detect_data()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    runtime, allocator = load_runtime(
        repo=repo,
        data=data,
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
        future_time_value = time_value + horizon
        metrics: dict[str, list[torch.Tensor]] = {}
        for start in range(0, args.samples, args.batch_size):
            stop = min(start + args.batch_size, args.samples)
            state = all_state[start:stop]
            batch_labels = labels[start:stop]
            time = torch.full((len(state),), time_value, device=device)
            future_time = torch.full(
                (len(state),), future_time_value, device=device
            )
            strong, weak = runtime.evaluate_pair(time, state, batch_labels)
            strong_future, weak_future = runtime.evaluate_pair(
                future_time, state, batch_labels
            )

            strong_score = linear_velocity_to_score(strong, state, time)
            weak_score = linear_velocity_to_score(weak, state, time)
            future_score = linear_velocity_to_score(
                weak_future, state, future_time
            )
            strong_future_score = linear_velocity_to_score(
                strong_future, state, future_time
            )
            retimed = retime_future_score(
                future_score, state, time, future_time
            )
            direct_retimed = reinterpret_future_velocity_score(
                weak_future, state, time
            )
            defect = exponential_retiming_defect(
                weak_score, future_score, state, time, future_time
            )
            strong_defect = exponential_retiming_defect(
                strong_score,
                strong_future_score,
                state,
                time,
                future_time,
            )
            score_evolution, gaussian_retiming = (
                split_exponential_retiming_defect(
                    weak_score, future_score, state, time, future_time
                )
            )
            depth_gap = strong_score - weak_score
            defect_projection = project_per_sample(defect, depth_gap)
            raw_revision = weak - weak_future
            score_revision = (1.0 - time.reshape(-1, 1, 1, 1)) / time.reshape(
                -1, 1, 1, 1
            ) * defect
            component_energy = (
                sample_rms(score_evolution).square()
                + sample_rms(gaussian_retiming).square()
            )
            cancellation_ratio = sample_rms(defect).square() / component_energy.clamp_min(
                1e-30
            )
            weight = retiming_weight(time, future_time, state).flatten(1)[:, 0]

            batch_metrics = {
                "retiming_weight": weight,
                "retiming_identity_relative_error": relative_rms(
                    direct_retimed - retimed, direct_retimed
                ),
                "velocity_score_revision_relative_error": relative_rms(
                    raw_revision - score_revision, raw_revision
                ),
                "depth_gap_rms": sample_rms(depth_gap),
                "geodesic_defect_rms": sample_rms(defect),
                "strong_geodesic_defect_rms": sample_rms(strong_defect),
                "weak_strong_defect_cosine": sample_cosine(
                    defect, strong_defect
                ),
                "weak_strong_defect_difference_rms": sample_rms(
                    defect - strong_defect
                ),
                "defect_depth_cosine": sample_cosine(defect, depth_gap),
                "strong_defect_depth_cosine": sample_cosine(
                    strong_defect, depth_gap
                ),
                "defect_parallel_rms": sample_rms(defect_projection.parallel),
                "defect_orthogonal_rms": sample_rms(defect_projection.orthogonal),
                "defect_orthogonal_energy_fraction": (
                    sample_rms(defect_projection.orthogonal).square()
                    / sample_rms(defect).square().clamp_min(1e-30)
                ),
                "score_evolution_rms": sample_rms(score_evolution),
                "gaussian_retiming_rms": sample_rms(gaussian_retiming),
                "component_cosine": sample_cosine(
                    score_evolution, gaussian_retiming
                ),
                "component_cancellation_ratio": cancellation_ratio,
                "raw_velocity_revision_rms": sample_rms(raw_revision),
            }
            for name, value in batch_metrics.items():
                append_metric(metrics, name, value)

            cpu_metrics = {
                name: value.detach().float().cpu()
                for name, value in batch_metrics.items()
            }
            for offset in range(len(state)):
                per_sample_rows.append(
                    {
                        "time": time_value,
                        "future_time": future_time_value,
                        "sample": start + offset,
                        **{
                            name: float(value[offset])
                            for name, value in cpu_metrics.items()
                        },
                    }
                )

        row: dict[str, Any] = {
            "time": time_value,
            "future_time": future_time_value,
            "horizon": horizon,
            "samples": args.samples,
        }
        row.update(summarize_metrics(metrics))
        rows.append(row)
        print(
            json.dumps(
                {
                    "event": "time_complete",
                    "time": time_value,
                    "weight": row["retiming_weight_mean"],
                    "defect_gap_cosine": row["defect_depth_cosine_mean"],
                    "orthogonal_fraction": row[
                        "defect_orthogonal_energy_fraction_mean"
                    ],
                    "component_cosine": row["component_cosine_mean"],
                }
            ),
            flush=True,
        )

    for path, data_rows in (
        (output / "retiming_summary.csv", rows),
        (output / "per_sample_retiming.csv", per_sample_rows),
    ):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data_rows[0]))
            writer.writeheader()
            writer.writerows(data_rows)

    manifest = {
        "format": "eqvae_pfr_exponential_retiming_audit_v1",
        "theorem": {
            "weight": "a=odds(t)/odds(tau)",
            "retimed_score": "R(s_tau)=a*s_tau+(1-a)*(-z)",
            "density_if_conservative": "qbar proportional to q_tau^a*phi^(1-a)",
            "pfr_time_score": "s_Wt+beta*(s_St-R(s_Wtau))",
            "defect": "delta=s_Wt-R(s_Wtau)",
        },
        "protocol": {
            "samples": args.samples,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "times": list(args.times),
            "horizon": args.horizon,
            "trajectory": "ordinary depth4 internal guidance",
            "query": "same latent at later affine-flow time",
            "strong": str(runtime.paths["strong"]),
            "weak": str(runtime.paths["depth4"]),
            "weights": "ema",
        },
        "allocator": allocator,
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "summary": str(output / "retiming_summary.csv"),
        "per_sample": str(output / "per_sample_retiming.csv"),
    }
    atomic_json(output / "manifest.json", manifest)
    print(json.dumps({"event": "complete", "output": str(output)}), flush=True)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
