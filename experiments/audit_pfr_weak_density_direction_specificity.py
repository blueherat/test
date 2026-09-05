#!/usr/bin/env python3
"""Audit whether PFR's weak-density ascent is direction-specific.

The canonical PFR spatial displacement is parallel to the ordinary guided
field.  Orthogonal and donor controls establish sample alignment, but do not
by themselves distinguish a special counterfactual from generic motion toward
the data endpoint.  This audit compares the PFR displacement with several
per-sample norm-matched alternatives and integrates the weak head's implied
score along each finite segment.

For the linear interpolant ``z_t=(1-t)e+t*x`` and a velocity field ``w``, the
corresponding implied score is

    s_w(z,t) = (t*w(z,t)-z)/(1-t).

Finite learned fields need not be conservative, so the reported quantity is a
straight-line score work, not an asserted global log-density difference.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Callable

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.pfr_query_controls import (  # noqa: E402
    matched_donor_shift,
    matched_orthogonal_scramble,
    rms_match_per_sample,
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
    gamma_at,
    load_runtime,
)
from experiments.run_imagenet100_sit_pfr_query_controls import (  # noqa: E402
    QueryControlledField,
    integrate_times,
)
from experiments.information_purification_ig import (  # noqa: E402
    projected_information_query,
)


DEFAULT_TIMES = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.46875)
DIRECTION_NAMES = (
    "pfr_projected",
    "anti_pfr",
    "orthogonal_pfr",
    "donor_pfr",
    "current_strong",
    "current_weak",
    "current_gap",
    "future_strong",
    "future_weak",
    "future_guided",
    "weak_score_ascent",
    "radial_inward",
)


def parse_times(value: str) -> tuple[float, ...]:
    try:
        times = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("times must be comma-separated floats") from error
    if not times or tuple(sorted(set(times))) != times:
        raise argparse.ArgumentTypeError("times must be unique and increasing")
    if any(not 0.0 < item < INTERVENTION_TIME for item in times):
        raise argparse.ArgumentTypeError("times must lie in (0, 0.5)")
    return times


def sample_dot(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return (first.float().flatten(1) * second.float().flatten(1)).sum(dim=1)


def sample_norm(value: torch.Tensor) -> torch.Tensor:
    return value.float().flatten(1).square().sum(dim=1).sqrt()


def sample_cosine(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return sample_dot(first, second) / (
        sample_norm(first) * sample_norm(second)
    ).clamp_min(1e-30)


def implied_score(
    state: torch.Tensor,
    time_value: torch.Tensor,
    velocity: torch.Tensor,
) -> torch.Tensor:
    """Convert a linear-flow velocity estimate to its implied score."""

    return (time_value * velocity - state) / (1.0 - time_value).clamp_min(1e-6)


def gauss_legendre_rule_5(
    *, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Five-point Gauss-Legendre nodes and weights mapped to [0, 1]."""

    nodes = torch.tensor(
        (
            0.046910077030668,
            0.230765344947158,
            0.5,
            0.769234655052842,
            0.953089922969332,
        ),
        device=device,
        dtype=dtype,
    )
    weights = torch.tensor(
        (
            0.118463442528095,
            0.239314335249683,
            0.284444444444444,
            0.239314335249683,
            0.118463442528095,
        ),
        device=device,
        dtype=dtype,
    )
    return nodes, weights


def finite_score_work(
    state: torch.Tensor,
    displacement: torch.Tensor,
    time_value: torch.Tensor,
    evaluate_velocity: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Integrate implied-score work along a straight finite displacement.

    Returns five-point Gauss-Legendre work, endpoint-trapezoid work, and the
    score directional derivative at the segment start.  Values are normalized
    by latent dimensionality, not by displacement magnitude.
    """

    nodes, weights = gauss_legendre_rule_5(device=state.device, dtype=state.dtype)
    feature_count = float(state[0].numel())
    work = torch.zeros(len(state), device=state.device, dtype=torch.float32)
    endpoint_scores: list[torch.Tensor] = []
    for index, (node, weight) in enumerate(zip(nodes, weights, strict=True)):
        query_state = state + node * displacement
        velocity = evaluate_velocity(time_value, query_state)
        score = implied_score(query_state, time_value, velocity)
        work = work + weight.float() * sample_dot(score, displacement)
        if index in (0, len(nodes) - 1):
            endpoint_scores.append(score)

    velocity_start = evaluate_velocity(time_value, state)
    velocity_end = evaluate_velocity(time_value, state + displacement)
    score_start = implied_score(state, time_value, velocity_start)
    score_end = implied_score(state + displacement, time_value, velocity_end)
    trapezoid = 0.5 * sample_dot(score_start + score_end, displacement)
    directional = sample_dot(score_start, displacement)
    return work / feature_count, trapezoid / feature_count, directional / feature_count


def make_directions(
    *,
    state: torch.Tensor,
    pfr_shift: torch.Tensor,
    strong_now: torch.Tensor,
    weak_now: torch.Tensor,
    strong_future: torch.Tensor,
    weak_future: torch.Tensor,
    gamma: float,
    weak_score_future: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build per-sample norm-matched alternatives to the PFR displacement."""

    guided_future = strong_future + gamma * (strong_future - weak_future)
    candidates = {
        "pfr_projected": pfr_shift,
        "anti_pfr": -pfr_shift,
        "orthogonal_pfr": matched_orthogonal_scramble(pfr_shift),
        "donor_pfr": matched_donor_shift(pfr_shift),
        "current_strong": strong_now,
        "current_weak": weak_now,
        "current_gap": strong_now - weak_now,
        "future_strong": strong_future,
        "future_weak": weak_future,
        "future_guided": guided_future,
        "weak_score_ascent": weak_score_future,
        "radial_inward": -state,
    }
    return {
        name: value
        if name in {"pfr_projected", "anti_pfr", "orthogonal_pfr", "donor_pfr"}
        else rms_match_per_sample(value, pfr_shift)
        for name, value in candidates.items()
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[float, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((float(row["time"]), str(row["direction"])), []).append(row)
    summaries: list[dict[str, Any]] = []
    for (time_value, direction), group in sorted(groups.items()):
        active = [row for row in group if bool(row["pfr_active"])]
        result: dict[str, Any] = {
            "time": time_value,
            "direction": direction,
            "sample_count": len(group),
            "active_count": len(active),
            "active_fraction": len(active) / len(group),
        }
        for prefix, selected in (("all", group), ("active", active)):
            if not selected:
                continue
            for key in (
                "work_per_dim",
                "trapezoid_work_per_dim",
                "start_directional_work_per_dim",
                "work_per_shift_energy",
                "cosine_to_pfr",
            ):
                values = [float(row[key]) for row in selected]
                result[f"{prefix}_{key}_mean"] = statistics.mean(values)
                result[f"{prefix}_{key}_median"] = statistics.median(values)
            result[f"{prefix}_positive_fraction"] = statistics.mean(
                float(row["work_per_dim"]) > 0.0 for row in selected
            )
        summaries.append(result)
    return summaries


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
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
        args.num_samples,
        *runtime.modules["LATENT_SHAPE"],
        generator=generator,
        device=device,
    )
    labels = torch.randint(
        0,
        runtime.modules["NUM_CLASSES"],
        (args.num_samples,),
        generator=generator,
        device=device,
    )
    baseline_field = QueryControlledField(
        runtime, labels, "ordinary_ig", record_diagnostics=False
    )
    states = integrate_times(
        baseline_field,
        noise.float(),
        args.times,
        atol=args.atol,
        rtol=args.rtol,
    )

    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for time_scalar, state in zip(args.times, states, strict=True):
            time_now = torch.tensor(time_scalar, device=device, dtype=state.dtype)
            strong_now, weak_now = runtime.evaluate_pair(time_now, state, labels)
            gamma = gamma_at(time_scalar)
            guided_now = strong_now + gamma * (strong_now - weak_now)
            query = projected_information_query(
                state,
                time_now,
                strong_now=strong_now,
                weak_now=weak_now,
                guided_now=guided_now,
                gamma=gamma,
                horizon=HORIZON,
                intervention_time=INTERVENTION_TIME,
            )
            query_time = query.time.to(dtype=state.dtype)
            pfr_shift = query.state - state
            strong_future, weak_future = runtime.evaluate_pair(query_time, state, labels)
            weak_score_future = implied_score(state, query_time, weak_future)
            directions = make_directions(
                state=state,
                pfr_shift=pfr_shift,
                strong_now=strong_now,
                weak_now=weak_now,
                strong_future=strong_future,
                weak_future=weak_future,
                gamma=gamma,
                weak_score_future=weak_score_future,
            )
            pfr_norm = sample_norm(pfr_shift)
            active = pfr_norm > 1e-12
            for name, displacement in directions.items():
                work, trapezoid, directional = finite_score_work(
                    state,
                    displacement,
                    query_time,
                    lambda t, z: runtime.evaluate_weak(t, z, labels),
                )
                displacement_energy_per_dim = sample_dot(
                    displacement, displacement
                ) / float(state[0].numel())
                normalized = work / displacement_energy_per_dim.clamp_min(1e-30)
                cosine = sample_cosine(displacement, pfr_shift)
                for sample_index in range(len(state)):
                    rows.append(
                        {
                            "time": time_scalar,
                            "query_time": float(query_time.item()),
                            "sample": sample_index,
                            "direction": name,
                            "pfr_active": int(active[sample_index]),
                            "shift_rms": float(
                                displacement_energy_per_dim[sample_index].sqrt()
                            ),
                            "cosine_to_pfr": float(cosine[sample_index]),
                            "work_per_dim": float(work[sample_index]),
                            "trapezoid_work_per_dim": float(trapezoid[sample_index]),
                            "start_directional_work_per_dim": float(
                                directional[sample_index]
                            ),
                            "work_per_shift_energy": float(normalized[sample_index]),
                        }
                    )
            print(
                json.dumps(
                    {
                        "event": "time_complete",
                        "time": time_scalar,
                        "active_fraction": float(active.float().mean()),
                    }
                ),
                flush=True,
            )

    summary_rows = summarize_rows(rows)
    write_csv(output / "per_sample_score_work.csv", rows)
    write_csv(output / "direction_score_work_summary.csv", summary_rows)
    summary = {
        "format": "eqvae_pfr_weak_density_direction_specificity_v1",
        "interpretation": (
            "Straight-line work under the weak velocity's implied score; "
            "not a guaranteed global log-density difference."
        ),
        "protocol": {
            "strong": str(runtime.paths["strong"]),
            "weak": str(runtime.paths["depth4"]),
            "weights": "ema",
            "trajectory": "ordinary depth-4 IG",
            "num_samples": args.num_samples,
            "seed": args.seed,
            "times": list(args.times),
            "directions": list(DIRECTION_NAMES),
            "quadrature": "five-point Gauss-Legendre on each straight segment",
            "anchor_horizon": HORIZON,
            "intervention_time": INTERVENTION_TIME,
        },
        "allocator": allocator,
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "rows": str(output / "per_sample_score_work.csv"),
        "summary_rows": str(output / "direction_score_work_summary.csv"),
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps({"event": "complete", "output": str(output)}), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument(
        "--times",
        type=parse_times,
        default=DEFAULT_TIMES,
    )
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--cuda-allocator-limit-gib", type=float, default=6.0)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
