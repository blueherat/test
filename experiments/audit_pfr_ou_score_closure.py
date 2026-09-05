#!/usr/bin/env python3
"""Audit whether PFR follows an OU conditional-score closure defect.

For normalized OU coordinates and ``a=alpha_t/alpha_tau``, every valid
relative score obeys

    r_t(y) = a E[r_tau(Y_tau) | Y_t=y].

The exact conditional expectation is unavailable at inference. This audit
uses the current score's exact Tweedie posterior mean as a deterministic
Gaussian closure and asks whether its residual aligns with the useful PFR
time revision. Geometry is screened before any rollout or FID experiment.
"""

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
    ou_future_posterior_mean_state,
    ou_relative_score_consistency_velocity_defect,
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
    parser.add_argument("--device", default="cuda:3")
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


def tensor_summary(value: torch.Tensor) -> dict[str, float]:
    result = summarize(value.detach().float().cpu().tolist())
    if result is None:
        raise RuntimeError("cannot summarize an empty tensor")
    return {name: float(item) for name, item in result.items()}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.samples <= 0 or args.batch_size <= 0:
        raise ValueError("samples and batch-size must be positive")
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
    for scalar_time, all_state in zip(args.times, states, strict=True):
        horizon = min(args.horizon, INTERVENTION_TIME - scalar_time)
        scalar_future = scalar_time + horizon
        collected: dict[str, list[torch.Tensor]] = {}
        for start in range(0, args.samples, args.batch_size):
            stop = min(start + args.batch_size, args.samples)
            state = all_state[start:stop]
            batch_labels = labels[start:stop]
            time = torch.full((len(state),), scalar_time, device=device)
            future_time = torch.full((len(state),), scalar_future, device=device)

            strong, weak = runtime.evaluate_pair(time, state, batch_labels)
            weak_future_raw = runtime.evaluate_weak(
                future_time, state, batch_labels
            )
            pfr_revision = weak - weak_future_raw

            weak_posterior_state = ou_future_posterior_mean_state(
                weak, state, time, future_time
            )
            strong_posterior_state = ou_future_posterior_mean_state(
                strong, state, time, future_time
            )
            weak_future_self = runtime.evaluate_weak(
                future_time, weak_posterior_state, batch_labels
            )
            weak_future_cross = runtime.evaluate_weak(
                future_time, strong_posterior_state, batch_labels
            )
            strong_future_self, _ = runtime.evaluate_pair(
                future_time, strong_posterior_state, batch_labels
            )

            weak_closure = ou_relative_score_consistency_velocity_defect(
                weak,
                weak_future_self,
                state,
                weak_posterior_state,
                time,
                future_time,
            )
            cross_closure = ou_relative_score_consistency_velocity_defect(
                weak,
                weak_future_cross,
                state,
                strong_posterior_state,
                time,
                future_time,
            )
            strong_closure = ou_relative_score_consistency_velocity_defect(
                strong,
                strong_future_self,
                state,
                strong_posterior_state,
                time,
                future_time,
            )

            pfr_rms = sample_rms(pfr_revision).clamp_min(1e-30)
            metrics = {
                "pfr_revision_rms": pfr_rms,
                "weak_posterior_shift_rms": sample_rms(
                    weak_posterior_state - state
                ),
                "strong_posterior_shift_rms": sample_rms(
                    strong_posterior_state - state
                ),
                "weak_closure_rms_ratio": sample_rms(weak_closure) / pfr_rms,
                "cross_closure_rms_ratio": sample_rms(cross_closure) / pfr_rms,
                "strong_closure_rms_ratio": sample_rms(strong_closure) / pfr_rms,
                "weak_closure_pfr_cosine": sample_cosine(
                    weak_closure, pfr_revision
                ),
                "cross_closure_pfr_cosine": sample_cosine(
                    cross_closure, pfr_revision
                ),
                "strong_closure_pfr_cosine": sample_cosine(
                    strong_closure, pfr_revision
                ),
                "weak_strong_closure_cosine": sample_cosine(
                    weak_closure, strong_closure
                ),
                "weak_cross_closure_cosine": sample_cosine(
                    weak_closure, cross_closure
                ),
            }
            for name, value in metrics.items():
                collected.setdefault(name, []).append(value.detach().cpu())
            cpu_metrics = {
                name: value.detach().float().cpu() for name, value in metrics.items()
            }
            for offset in range(len(state)):
                per_sample_rows.append(
                    {
                        "time": scalar_time,
                        "future_time": scalar_future,
                        "sample": start + offset,
                        **{
                            name: float(value[offset])
                            for name, value in cpu_metrics.items()
                        },
                    }
                )

        row: dict[str, Any] = {
            "time": scalar_time,
            "future_time": scalar_future,
            "samples": args.samples,
        }
        for name, chunks in collected.items():
            for statistic, value in tensor_summary(torch.cat(chunks)).items():
                row[f"{name}_{statistic}"] = value
        rows.append(row)
        print(
            json.dumps(
                {
                    "time": scalar_time,
                    "weak_cosine": row["weak_closure_pfr_cosine_mean"],
                    "cross_cosine": row["cross_closure_pfr_cosine_mean"],
                    "strong_cosine": row["strong_closure_pfr_cosine_mean"],
                }
            ),
            flush=True,
        )

    write_csv(output / "ou_score_closure_summary.csv", rows)
    write_csv(output / "per_sample_ou_score_closure.csv", per_sample_rows)
    atomic_json(
        output / "manifest.json",
        {
            "format": "eqvae_pfr_ou_score_closure_audit_v1",
            "question": (
                "Does a deterministic posterior-mean approximation to the exact "
                "OU conditional-score identity align with the useful PFR revision?"
            ),
            "samples": args.samples,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "times": list(args.times),
            "horizon": args.horizon,
            "theory_boundary": (
                "The conditional-score identity and Tweedie posterior mean are exact; "
                "replacing a posterior expectation by one score evaluation at its "
                "mean is exact only for affine scores and is otherwise a closure."
            ),
            "rows": rows,
            **allocator,
            "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
