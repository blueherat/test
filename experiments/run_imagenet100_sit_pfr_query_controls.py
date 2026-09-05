#!/usr/bin/env python3
"""Query-specificity audit for Projected Future Reference Internal Guidance.

All sampling conditions use the same strong model, weak internal head, IG
schedule, query horizon, noise, labels, ODE solver, and residualization rule

    V = G - beta * (W(q) - W(p)),  beta = 1 + gamma.

Only the counterfactual query ``q`` changes.  Norm-matched anti, orthogonal,
and donor displacements distinguish a sample-aligned finite response from a
generic future-time, perturbation-energy, or numerical-lookahead effect.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.batch_seed_schema import (  # noqa: E402
    BATCH_SEED_SCHEMAS,
    DEFAULT_BATCH_SEED_SCHEMA,
    batch_rng_manifest,
    batch_seed,
    manifest_uses_batch_rng,
)
from experiments.information_purification_ig import (  # noqa: E402
    lambda_residualized_guidance,
)
from experiments.path_evidence_pfr_bridge import (  # noqa: E402
    project_to_forward_ray,
    sample_cosine,
    sample_rms,
)
from experiments.pfr_query_controls import (  # noqa: E402
    QUERY_KINDS,
    controlled_information_query,
    response_odd_even,
    split_spatial_response,
)
from experiments.pfr_information_clock import (  # noqa: E402
    INFORMATION_CLOCKS,
    matched_information_horizon,
)
from experiments.run_imagenet100_sit_internal_early_two_segment_gamma_sweep import (  # noqa: E402
    atomic_json,
    detect_adm_python,
    detect_data,
    detect_repo,
    parse_gpus,
    read_json,
    runtime_paths,
)
from experiments.run_imagenet100_sit_path_evidence_pfr_bridge import (  # noqa: E402
    EXPECTED_LABEL,
    EXPECTED_NOISE,
    HORIZON,
    INTERVENTION_TIME,
    Runtime,
    gamma_at,
    load_runtime,
)


RESPONSE_CONDITIONS = (
    "projected_temporal_parallel",
    "projected_temporal_orthogonal",
)
ALL_CONDITIONS = ("ordinary_ig", *QUERY_KINDS, *RESPONSE_CONDITIONS)
HISTORICAL_ORDINARY_FID1K = 64.85216325274007
HISTORICAL_PROJECTED_FID1K = 61.924444086644996


def parse_conditions(value: str) -> tuple[str, ...]:
    conditions = tuple(item.strip() for item in value.split(",") if item.strip())
    if (
        not conditions
        or len(set(conditions)) != len(conditions)
        or not set(conditions) <= set(ALL_CONDITIONS)
    ):
        raise argparse.ArgumentTypeError(
            "conditions must be unique values from " + ",".join(ALL_CONDITIONS)
        )
    return conditions


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


def summarize(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "mean": statistics.mean(ordered),
        "median": statistics.median(ordered),
        "min": ordered[0],
        "max": ordered[-1],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def condition_formula(condition: str) -> str:
    if condition == "ordinary_ig":
        return "G=S+gamma*(S-W)"
    if condition == "projected_temporal_parallel":
        return "G-beta*(T+Proj_T(I)); T=W(z,t+h)-W(z,t); I=W(z+d,t+h)-W(z,t+h)"
    if condition == "projected_temporal_orthogonal":
        return "G-beta*(T+I-Proj_T(I)); T=W(z,t+h)-W(z,t); I=W(z+d,t+h)-W(z,t+h)"
    return f"G-beta*(W(q_{condition})-W(p))"


class QueryControlledField:
    def __init__(
        self,
        runtime: Runtime,
        labels: torch.Tensor,
        condition: str,
        *,
        record_diagnostics: bool = True,
        query_clock: str = "raw_t",
        clock_anchor_time: float = 0.25,
    ) -> None:
        self.runtime = runtime
        self.labels = labels
        self.condition = condition
        self.record_diagnostics = bool(record_diagnostics)
        self.query_clock = query_clock
        self.clock_anchor_time = float(clock_anchor_time)
        self.nfe = 0
        self.query_nfe = 0
        self.diagnostics: dict[str, list[float]] = {}

    def _record(self, key: str, values: torch.Tensor) -> None:
        if self.record_diagnostics:
            self.diagnostics.setdefault(key, []).extend(
                values.detach().float().cpu().flatten().tolist()
            )

    def __call__(self, time_value: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        self.nfe += 1
        with torch.inference_mode():
            strong, weak = self.runtime.evaluate_pair(time_value, state, self.labels)
            gamma = gamma_at(float(time_value.detach().float().item()))
            gap = strong - weak
            guided = strong + gamma * gap
            if gamma == 0.0 or self.condition == "ordinary_ig":
                return guided
            horizon = matched_information_horizon(
                float(time_value.detach().float().item()),
                clock=self.query_clock,
                anchor_time=self.clock_anchor_time,
                anchor_horizon=HORIZON,
                intervention_time=INTERVENTION_TIME,
            )
            if horizon <= 0.0:
                return guided
            query_kind = (
                "projected"
                if self.condition in RESPONSE_CONDITIONS
                else self.condition
            )
            query = controlled_information_query(
                state,
                time_value,
                strong_now=strong,
                weak_now=weak,
                guided_now=guided,
                gamma=gamma,
                horizon=horizon,
                intervention_time=INTERVENTION_TIME,
                kind=query_kind,
            )
            weak_query = self.runtime.evaluate_weak(query.time, query.state, self.labels)
            self.query_nfe += 1
            if self.condition in RESPONSE_CONDITIONS:
                time_query = controlled_information_query(
                    state,
                    time_value,
                    strong_now=strong,
                    weak_now=weak,
                    guided_now=guided,
                    gamma=gamma,
                    horizon=horizon,
                    intervention_time=INTERVENTION_TIME,
                    kind="time_only",
                )
                weak_time = self.runtime.evaluate_weak(
                    time_query.time, time_query.state, self.labels
                )
                self.query_nfe += 1
                split = split_spatial_response(weak, weak_time, weak_query)
                spatial = (
                    split.spatial_parallel
                    if self.condition == "projected_temporal_parallel"
                    else split.spatial_orthogonal
                )
                weak_revision = split.temporal + spatial
                self._record("spatial_temporal_coefficient", split.coefficient)
                self._record("spatial_parallel_rms", sample_rms(split.spatial_parallel))
                self._record("spatial_orthogonal_rms", sample_rms(split.spatial_orthogonal))
            else:
                weak_revision = weak_query - weak
            beta = 1.0 + gamma
            projected_shift = query.projected.state - state
            self._record("query_shift_rms", sample_rms(query.spatial_shift))
            self._record(
                "query_shift_projected_cosine",
                sample_cosine(query.spatial_shift, projected_shift),
            )
            self._record("weak_revision_rms", sample_rms(weak_revision))
            self._record(
                "weak_revision_gap_cosine", sample_cosine(weak_revision, gap)
            )
            self._record(
                "pfr_correction_rms", sample_rms(-beta * weak_revision)
            )
            return lambda_residualized_guidance(
                guided,
                weak_revision,
                beta=beta,
                residualization=1.0,
            )


def integrate_times(
    field: Any,
    state: torch.Tensor,
    times: tuple[float, ...],
    *,
    atol: float,
    rtol: float,
) -> torch.Tensor:
    from torchdiffeq import odeint

    grid = torch.tensor((0.0, *times), device=state.device, dtype=torch.float32)
    return odeint(field, state, grid, method="dopri5", atol=atol, rtol=rtol)[1:]


def _sample_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return sample_rms(numerator) / sample_rms(denominator).clamp_min(1e-30)


def _sample_dot(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return (first.float().flatten(1) * second.float().flatten(1)).sum(dim=1)


def _rows_for_local_query(
    *,
    runtime: Runtime,
    state: torch.Tensor,
    labels: torch.Tensor,
    time_scalar: float,
    query_clock: str = "raw_t",
    clock_anchor_time: float = 0.25,
) -> list[dict[str, Any]]:
    time_value = torch.tensor(time_scalar, device=state.device)
    with torch.inference_mode():
        strong, weak = runtime.evaluate_pair(time_value, state, labels)
        gamma = gamma_at(time_scalar)
        beta = 1.0 + gamma
        gap = strong - weak
        guided = strong + gamma * gap
        horizon = matched_information_horizon(
            time_scalar,
            clock=query_clock,
            anchor_time=clock_anchor_time,
            anchor_horizon=HORIZON,
            intervention_time=INTERVENTION_TIME,
        )
        controls = {
            kind: controlled_information_query(
                state,
                time_value,
                strong_now=strong,
                weak_now=weak,
                guided_now=guided,
                gamma=gamma,
                horizon=horizon,
                intervention_time=INTERVENTION_TIME,
                kind=kind,
            )
            for kind in QUERY_KINDS
        }
        weak_queries = {
            kind: runtime.evaluate_weak(query.time, query.state, labels)
            for kind, query in controls.items()
        }

        projected = controls["projected"]
        projected_shift = projected.spatial_shift
        projected_revision = weak_queries["projected"] - weak
        projected_correction = -beta * projected_revision

        center = weak_queries["time_only"]
        positive = weak_queries["projected"]
        negative = weak_queries["anti_projected"]
        odd, even = response_odd_even(center, positive, negative)
        temporal = center - weak
        positive_spatial = positive - center
        response_split = split_spatial_response(weak, center, positive)
        spatial_secant_work = -_sample_dot(positive_spatial, projected_shift)
        projected_shift_energy = _sample_dot(projected_shift, projected_shift)
        spatial_secant_curvature = spatial_secant_work / projected_shift_energy.clamp_min(
            1e-30
        )
        spatial_correction_shift_cosine = sample_cosine(
            -positive_spatial, projected_shift
        )
        temporal_correction_guided_cosine = sample_cosine(-temporal, guided)
        spatial_correction_guided_cosine = sample_cosine(
            -positive_spatial, guided
        )
        query_time = projected.time.to(dtype=state.dtype)
        score_denominator = (1.0 - query_time).clamp_min(1e-6)
        weak_score_center = (query_time * center - state) / score_denominator
        weak_score_positive = (
            query_time * positive - (state + projected_shift)
        ) / score_denominator
        weak_score_negative = (
            query_time * negative - (state - projected_shift)
        ) / score_denominator
        feature_count = float(state[0].numel())
        weak_log_density_shift_positive = 0.5 * _sample_dot(
            weak_score_center + weak_score_positive, projected_shift
        ) / feature_count
        weak_log_density_shift_negative = 0.5 * _sample_dot(
            weak_score_center + weak_score_negative, -projected_shift
        ) / feature_count

        # Test the explicit one-cycle inverse-response prediction.  C0 is the
        # ordinary IG correction relative to W; C1 is PFR's pre-emphasized
        # correction.  A useful inverse response should reduce the one-cycle
        # reconstruction residual, without claiming it optimizes FID.
        c0 = beta * gap
        e0 = projected_revision
        c1 = c0 - beta * e0
        g1 = weak + c1
        projection1 = project_to_forward_ray(c1, g1)
        step = projected.horizon
        q1_state = state + step * projection1.parallel
        weak_q1 = runtime.evaluate_weak(projected.time, q1_state, labels)
        e1 = weak_q1 - weak
        naive_cycle_error = beta * e0
        compensated_cycle_error = c1 + beta * e1 - c0

    rows: list[dict[str, Any]] = []
    for kind in QUERY_KINDS:
        query = controls[kind]
        revision = weak_queries[kind] - weak
        correction = -beta * revision
        if kind == "state_only":
            weak_at_query_time = weak
        else:
            weak_at_query_time = center
        controlled_denominator = (1.0 - query.time).clamp_min(1e-6)
        controlled_score_start = (
            query.time * weak_at_query_time - state
        ) / controlled_denominator
        controlled_score_end = (
            query.time * weak_queries[kind] - query.state
        ) / controlled_denominator
        controlled_density_shift = 0.5 * _sample_dot(
            controlled_score_start + controlled_score_end,
            query.spatial_shift,
        ) / feature_count
        persistent_gap = gap - revision
        revision_energy = _sample_dot(revision, revision)
        gap_energy = _sample_dot(gap, gap)
        revision_gap_dot = _sample_dot(revision, gap)
        projection_identity_ratio = revision_gap_dot / revision_energy.clamp_min(1e-30)
        gap_distance_change_fraction = (
            _sample_dot(persistent_gap, persistent_gap) - gap_energy
        ) / gap_energy.clamp_min(1e-30)
        for index in range(len(state)):
            rows.append(
                {
                    "sample": index,
                    "time": time_scalar,
                    "query": kind,
                    "gamma": gamma,
                    "horizon": query.horizon,
                    "query_shift_rms": float(sample_rms(query.spatial_shift)[index]),
                    "query_shift_projected_cosine": float(
                        sample_cosine(query.spatial_shift, projected_shift)[index]
                    ),
                    "weak_revision_rms": float(sample_rms(revision)[index]),
                    "gap_rms": float(sample_rms(gap)[index]),
                    "persistent_gap_rms": float(sample_rms(persistent_gap)[index]),
                    "weak_revision_gap_cosine": float(
                        sample_cosine(revision, gap)[index]
                    ),
                    "persistent_revision_cosine": float(
                        sample_cosine(persistent_gap, revision)[index]
                    ),
                    "projection_identity_ratio": float(
                        projection_identity_ratio[index]
                    ),
                    "gap_distance_change_fraction": float(
                        gap_distance_change_fraction[index]
                    ),
                    "weak_revision_projected_cosine": float(
                        sample_cosine(revision, projected_revision)[index]
                    ),
                    "correction_projected_cosine": float(
                        sample_cosine(correction, projected_correction)[index]
                    ),
                    "temporal_response_rms": float(sample_rms(temporal)[index]),
                    "positive_spatial_response_rms": float(
                        sample_rms(positive_spatial)[index]
                    ),
                    "spatial_odd_rms": float(sample_rms(odd)[index]),
                    "spatial_even_rms": float(sample_rms(even)[index]),
                    "odd_fraction_of_positive_spatial": float(
                        _sample_ratio(odd, positive_spatial)[index]
                    ),
                    "even_fraction_of_positive_spatial": float(
                        _sample_ratio(even, positive_spatial)[index]
                    ),
                    "temporal_spatial_cosine": float(
                        sample_cosine(temporal, positive_spatial)[index]
                    ),
                    "spatial_secant_work": float(spatial_secant_work[index]),
                    "spatial_secant_curvature": float(
                        spatial_secant_curvature[index]
                    ),
                    "spatial_secant_positive": float(
                        spatial_secant_work[index] > 0.0
                    ),
                    "spatial_correction_shift_cosine": float(
                        spatial_correction_shift_cosine[index]
                    ),
                    "temporal_correction_guided_cosine": float(
                        temporal_correction_guided_cosine[index]
                    ),
                    "spatial_correction_guided_cosine": float(
                        spatial_correction_guided_cosine[index]
                    ),
                    "weak_log_density_shift_positive_per_dim": float(
                        weak_log_density_shift_positive[index]
                    ),
                    "weak_log_density_shift_negative_per_dim": float(
                        weak_log_density_shift_negative[index]
                    ),
                    "weak_density_increases_positive": float(
                        weak_log_density_shift_positive[index] > 0.0
                    ),
                    "weak_density_increases_negative": float(
                        weak_log_density_shift_negative[index] > 0.0
                    ),
                    "controlled_weak_log_density_shift_per_dim": float(
                        controlled_density_shift[index]
                    ),
                    "controlled_weak_density_increases": float(
                        controlled_density_shift[index] > 0.0
                    ),
                    "spatial_temporal_coefficient": float(
                        response_split.coefficient[index]
                    ),
                    "spatial_parallel_rms": float(
                        sample_rms(response_split.spatial_parallel)[index]
                    ),
                    "spatial_orthogonal_rms": float(
                        sample_rms(response_split.spatial_orthogonal)[index]
                    ),
                    "spatial_parallel_energy_fraction": float(
                        sample_rms(response_split.spatial_parallel)[index].square()
                        / sample_rms(positive_spatial)[index].square().clamp_min(1e-30)
                    ),
                    "naive_cycle_error_rms": float(
                        sample_rms(naive_cycle_error)[index]
                    ),
                    "compensated_cycle_error_rms": float(
                        sample_rms(compensated_cycle_error)[index]
                    ),
                    "cycle_error_ratio": float(
                        _sample_ratio(compensated_cycle_error, naive_cycle_error)[index]
                    ),
                }
            )
    return rows


def grouped_means(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["query"]), float(row["time"])), []).append(row)
    output: list[dict[str, Any]] = []
    excluded = {"query", "time", "sample"}
    for (query, time_scalar), group in sorted(groups.items(), key=lambda item: (item[0][1], item[0][0])):
        result: dict[str, Any] = {"query": query, "time": time_scalar}
        for key in group[0]:
            if key in excluded:
                continue
            result[f"{key}_mean"] = statistics.mean(float(row[key]) for row in group)
            result[f"{key}_median"] = statistics.median(float(row[key]) for row in group)
        output.append(result)
    return output


def geometry(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    repo = detect_repo()
    data = detect_data()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
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
    field = QueryControlledField(
        runtime,
        labels,
        "ordinary_ig",
        record_diagnostics=False,
        query_clock=args.query_clock,
        clock_anchor_time=args.clock_anchor_time,
    )
    states = integrate_times(
        field, noise.float(), args.times, atol=args.atol, rtol=args.rtol
    )
    rows: list[dict[str, Any]] = []
    for time_scalar, state in zip(args.times, states, strict=True):
        rows.extend(
            _rows_for_local_query(
                runtime=runtime,
                state=state,
                labels=labels,
                time_scalar=time_scalar,
                query_clock=args.query_clock,
                clock_anchor_time=args.clock_anchor_time,
            )
        )
        print(json.dumps({"event": "time_complete", "time": time_scalar}), flush=True)
    summary_rows = grouped_means(rows)
    write_csv(output / "per_sample_query_geometry.csv", rows)
    write_csv(output / "query_geometry_summary.csv", summary_rows)
    summary = {
        "format": "eqvae_pfr_query_controls_geometry_v1",
        "protocol": {
            "strong": str(runtime.paths["strong"]),
            "weak": str(runtime.paths["depth4"]),
            "weights": "ema",
            "num_samples": args.num_samples,
            "seed": args.seed,
            "times": list(args.times),
            "anchor_horizon": HORIZON,
            "query_clock": args.query_clock,
            "clock_anchor_time": args.clock_anchor_time,
            "intervention_time": INTERVENTION_TIME,
            "conditions": list(QUERY_KINDS),
        },
        "cycle_test": (
            "C0=beta*(S-W); E(C)=W(q(C))-W; "
            "compare beta*E(C0) with C1+beta*E(C1)-C0, "
            "C1=C0-beta*E(C0)"
        ),
        "allocator": allocator,
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "rows": str(output / "per_sample_query_geometry.csv"),
        "summary_rows": str(output / "query_geometry_summary.csv"),
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps({"event": "geometry_complete", "output": str(output)}), flush=True)


def result_reusable(path: Path, condition: str, args: argparse.Namespace) -> bool:
    if not path.is_file():
        return False
    try:
        result = read_json(path)
        manifest = result["sampling_manifest"]
        metrics = result["metrics"]
        query = manifest.get("query", {})
        return (
            result["condition"] == condition
            and int(manifest["sampling"]["num_samples"]) == args.num_samples
            and int(manifest["sampling"]["batch_size"]) == args.batch_size
            and int(manifest["sampling"]["seed"]) == args.seed
            and manifest_uses_batch_rng(
                manifest,
                args.seed,
                schema=args.batch_seed_schema,
            )
            and query.get("clock", "raw_t") == args.query_clock
            and math.isclose(
                float(query.get("clock_anchor_time", 0.25)),
                float(args.clock_anchor_time),
            )
            and all(
                math.isfinite(float(metrics[key]))
                for key in ("fid", "sfid", "inception_score")
            )
        )
    except Exception:
        return False


def fid_worker(args: argparse.Namespace) -> None:
    import numpy as np
    from diffusers.models import AutoencoderKL
    from torchvision.utils import save_image

    repo = args.repo.expanduser().resolve()
    data = args.data.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "condition_result.json"
    if result_reusable(result_path, args.condition, args):
        print(json.dumps({"event": "reuse", "condition": args.condition}), flush=True)
        return
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    runtime, allocator = load_runtime(
        repo=repo,
        data=data,
        adm_python=args.adm_python,
        device=device,
        allocator_limit_gib=args.cuda_allocator_limit_gib,
    )
    vae = (
        AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse", local_files_only=True
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    images = np.empty((args.num_samples, 256, 256, 3), dtype=np.uint8)
    labels_array = np.empty(args.num_samples, dtype=np.int16)
    noise_hash = hashlib.sha256()
    label_hash = hashlib.sha256()
    diagnostics: dict[str, list[float]] = {}
    total_nfe = 0
    total_query_nfe = 0
    cursor = 0
    preview = None
    with torch.inference_mode():
        while cursor < args.num_samples:
            current_batch = min(args.batch_size, args.num_samples - cursor)
            batch_index = cursor // args.batch_size
            generator = torch.Generator(device=device).manual_seed(
                batch_seed(
                    args.seed,
                    batch_index,
                    schema=args.batch_seed_schema,
                )
            )
            noise = torch.randn(
                current_batch,
                *runtime.modules["LATENT_SHAPE"],
                generator=generator,
                device=device,
            )
            labels = torch.randint(
                0,
                runtime.modules["NUM_CLASSES"],
                (current_batch,),
                generator=generator,
                device=device,
            )
            field = QueryControlledField(
                runtime,
                labels,
                args.condition,
                query_clock=args.query_clock,
                clock_anchor_time=args.clock_anchor_time,
            )
            from torchdiffeq import odeint

            endpoint = odeint(
                field,
                noise.float(),
                torch.tensor([0.0, 1.0], device=device),
                method="dopri5",
                atol=args.atol,
                rtol=args.rtol,
            )[-1]
            if not torch.isfinite(endpoint).all():
                raise FloatingPointError(args.condition)
            decoded = runtime.modules["decode_latents_in_chunks"](
                vae,
                endpoint,
                scaling_factor=runtime.modules["SD_VAE_SCALING_FACTOR"],
                chunk_size=args.vae_decode_batch_size,
            )
            stop = cursor + current_batch
            images[cursor:stop] = runtime.modules["official_pixel_quantization"](decoded)
            labels_array[cursor:stop] = labels.cpu().numpy().astype(np.int16, copy=False)
            noise_hash.update(noise.cpu().contiguous().numpy().tobytes())
            label_hash.update(labels.cpu().contiguous().numpy().tobytes())
            if preview is None:
                preview = decoded[: min(16, len(decoded))].cpu()
            total_nfe += field.nfe
            total_query_nfe += field.query_nfe
            for key, values in field.diagnostics.items():
                diagnostics.setdefault(key, []).extend(values)
            cursor = stop
            if cursor == current_batch or cursor == args.num_samples or cursor % 256 == 0:
                print(
                    json.dumps(
                        {
                            "condition": args.condition,
                            "generated": cursor,
                            "total": args.num_samples,
                            "last_batch_nfe": field.nfe,
                        }
                    ),
                    flush=True,
                )

    sample_path = output / f"samples_n{args.num_samples}.npz"
    label_path = output / f"labels_n{args.num_samples}.npy"
    np.savez(sample_path, arr_0=images)
    np.save(label_path, labels_array, allow_pickle=False)
    if preview is None:
        raise RuntimeError("sampling produced no preview")
    save_image(preview, output / "preview.png", nrow=4, normalize=True, value_range=(-1, 1))
    sampling_manifest = {
        "format": "eqvae_pfr_query_control_samples_v1",
        "condition": args.condition,
        "formula": condition_formula(args.condition),
        "sampling": {
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "integrator": "dopri5",
            "atol": args.atol,
            "rtol": args.rtol,
        },
        "batch_rng": batch_rng_manifest(
            args.seed,
            schema=args.batch_seed_schema,
        ),
        "query": {
            "kind": args.condition,
            "anchor_horizon": HORIZON,
            "intervention_time": INTERVENTION_TIME,
            "clock": args.query_clock,
            "clock_anchor_time": args.clock_anchor_time,
        },
        "strong": runtime.strong_metadata,
        "weak_checkpoint": str(runtime.paths["depth4"]),
        "noise_sha256": noise_hash.hexdigest(),
        "label_sha256": label_hash.hexdigest(),
        "total_nfe": total_nfe,
        "total_query_nfe": total_query_nfe,
        "diagnostics": {key: summarize(values) for key, values in diagnostics.items()},
        "samples": str(sample_path),
        "labels": str(label_path),
        **allocator,
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    runtime.modules["atomic_json_dump"](
        sampling_manifest, output / "sampling_manifest.json"
    )
    del vae, runtime
    gc.collect()
    torch.cuda.empty_cache()
    metric_path = output / "adm_metrics.json"
    environment = os.environ.copy()
    environment.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    subprocess.run(
        [
            str(args.adm_python),
            str(repo / "experiments/compute_adm_fid.py"),
            "--reference",
            str(data / "adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz"),
            "--samples",
            str(sample_path),
            "--batch-size",
            str(args.fid_batch_size),
            "--gpu-memory-fraction",
            str(args.fid_gpu_memory_fraction),
            "--output",
            str(metric_path),
        ],
        cwd=repo,
        env=environment,
        check=True,
    )
    metrics = read_json(metric_path)
    result = {
        "format": "eqvae_pfr_query_control_result_v1",
        "condition": args.condition,
        "sampling_manifest": sampling_manifest,
        "metrics": metrics,
        "sample_retained": bool(args.keep_samples),
    }
    atomic_json(result_path, result)
    if not args.keep_samples:
        sample_path.unlink(missing_ok=True)
    print(
        json.dumps(
            {"event": "complete", "condition": args.condition, "fid": metrics["fid"]}
        ),
        flush=True,
    )


def run_one_condition(
    condition: str,
    gpu: int,
    root: Path,
    repo: Path,
    data: Path,
    adm_python: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    output = root / condition
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "condition_result.json"
    if result_reusable(result_path, condition, args):
        result = read_json(result_path)
        print(f"[reuse] {condition}: FID={float(result['metrics']['fid']):.4f}", flush=True)
        return result
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--repo",
        str(repo),
        "--data",
        str(data),
        "--adm-python",
        str(adm_python),
        "--output-dir",
        str(output),
        "--condition",
        condition,
        "--num-samples",
        str(args.num_samples),
        "--batch-size",
        str(args.batch_size),
        "--vae-decode-batch-size",
        str(args.vae_decode_batch_size),
        "--seed",
        str(args.seed),
        "--batch-seed-schema",
        args.batch_seed_schema,
        "--atol",
        str(args.atol),
        "--rtol",
        str(args.rtol),
        "--cuda-allocator-limit-gib",
        str(args.cuda_allocator_limit_gib),
        "--query-clock",
        args.query_clock,
        "--clock-anchor-time",
        str(args.clock_anchor_time),
        "--fid-batch-size",
        str(args.fid_batch_size),
        "--fid-gpu-memory-fraction",
        str(args.fid_gpu_memory_fraction),
    ]
    if args.keep_samples:
        command.append("--keep-samples")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    log_path = output / "run.log"
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            command,
            cwd=repo,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if process.returncode:
        tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-100:])
        raise RuntimeError(f"{condition} failed on GPU {gpu}\n{tail}")
    result = read_json(result_path)
    print(
        f"[GPU {gpu}] {condition}: FID={float(result['metrics']['fid']):.4f}",
        flush=True,
    )
    return result


def fid(args: argparse.Namespace) -> None:
    repo = detect_repo()
    data = detect_data()
    adm_python = detect_adm_python()
    runtime_paths(repo, data, adm_python)
    root = args.output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    lanes: list[list[str]] = [[] for _ in args.gpus]
    for index, condition in enumerate(args.conditions):
        lanes[index % len(args.gpus)].append(condition)

    def lane(gpu: int, conditions: list[str]) -> list[dict[str, Any]]:
        return [
            run_one_condition(condition, gpu, root, repo, data, adm_python, args)
            for condition in conditions
        ]

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [
            pool.submit(lane, gpu, conditions)
            for gpu, conditions in zip(args.gpus, lanes, strict=True)
            if conditions
        ]
        for future in as_completed(futures):
            results.extend(future.result())

    rows: list[dict[str, Any]] = []
    for result in results:
        manifest = result["sampling_manifest"]
        metrics = result["metrics"]
        diagnostics = manifest.get("diagnostics", {})
        rows.append(
            {
                "condition": result["condition"],
                "fid": float(metrics["fid"]),
                "sfid": float(metrics["sfid"]),
                "inception_score": float(metrics["inception_score"]),
                "total_nfe": int(manifest["total_nfe"]),
                "total_query_nfe": int(manifest["total_query_nfe"]),
                "query_shift_rms_mean": (
                    None
                    if not diagnostics.get("query_shift_rms")
                    else diagnostics["query_shift_rms"]["mean"]
                ),
                "weak_revision_rms_mean": (
                    None
                    if not diagnostics.get("weak_revision_rms")
                    else diagnostics["weak_revision_rms"]["mean"]
                ),
                "noise_sha256": manifest["noise_sha256"],
                "label_sha256": manifest["label_sha256"],
            }
        )
    order = {condition: index for index, condition in enumerate(ALL_CONDITIONS)}
    rows.sort(key=lambda row: order[str(row["condition"])])
    if len({row["noise_sha256"] for row in rows}) != 1:
        raise RuntimeError("conditions did not use paired noise")
    if len({row["label_sha256"] for row in rows}) != 1:
        raise RuntimeError("conditions did not use paired labels")
    if (
        args.query_clock == "raw_t"
        and args.num_samples == 1000
        and args.batch_size == 8
        and args.seed == 0
    ):
        if rows[0]["noise_sha256"] != EXPECTED_NOISE:
            raise RuntimeError("noise differs from the historical FID-1K bank")
        if rows[0]["label_sha256"] != EXPECTED_LABEL:
            raise RuntimeError("labels differ from the historical FID-1K bank")
        anchors = {str(row["condition"]): float(row["fid"]) for row in rows}
        if "ordinary_ig" in anchors and abs(
            anchors["ordinary_ig"] - HISTORICAL_ORDINARY_FID1K
        ) > 0.15:
            raise RuntimeError("ordinary IG anchor did not reproduce")
        if "projected" in anchors and abs(
            anchors["projected"] - HISTORICAL_PROJECTED_FID1K
        ) > 0.15:
            raise RuntimeError("projected PFR anchor did not reproduce")
    write_csv(root / "summary.csv", rows)
    summary = {
        "format": "eqvae_pfr_query_controls_fid_v1",
        "protocol": {
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "batch_rng": batch_rng_manifest(
                args.seed,
                schema=args.batch_seed_schema,
            ),
            "conditions": list(args.conditions),
            "anchor_horizon": HORIZON,
            "query_clock": args.query_clock,
            "clock_anchor_time": args.clock_anchor_time,
            "intervention_time": INTERVENTION_TIME,
        },
        "pairing": {
            "verified": True,
            "noise_sha256": rows[0]["noise_sha256"],
            "label_sha256": rows[0]["label_sha256"],
        },
        "best": min(rows, key=lambda row: float(row["fid"])),
        "rows": str(root / "summary.csv"),
    }
    atomic_json(root / "summary.json", summary)
    print(json.dumps(summary["best"], indent=2), flush=True)


def add_sampling_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--vae-decode-batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--batch-seed-schema",
        choices=BATCH_SEED_SCHEMAS,
        default=DEFAULT_BATCH_SEED_SCHEMA,
        help=(
            "batch RNG derivation; use legacy_additive_v1 only to reproduce "
            "historical overlapping seed banks"
        ),
    )
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--cuda-allocator-limit-gib", type=float, default=6.0)
    parser.add_argument("--fid-batch-size", type=int, default=16)
    parser.add_argument("--fid-gpu-memory-fraction", type=float, default=0.25)
    parser.add_argument("--keep-samples", action="store_true")
    add_clock_args(parser)


def add_clock_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--query-clock", choices=INFORMATION_CLOCKS, default="raw_t"
    )
    parser.add_argument("--clock-anchor-time", type=float, default=0.25)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    geometry_parser = subparsers.add_parser("geometry")
    geometry_parser.add_argument("--output-dir", type=Path, required=True)
    geometry_parser.add_argument("--device", default="cuda:0")
    geometry_parser.add_argument("--num-samples", type=int, default=32)
    geometry_parser.add_argument("--seed", type=int, default=20260903)
    geometry_parser.add_argument(
        "--times",
        type=parse_times,
        default=(0.05, 0.15, 0.24, 0.3, 0.4, 0.46875),
    )
    geometry_parser.add_argument("--atol", type=float, default=1e-6)
    geometry_parser.add_argument("--rtol", type=float, default=1e-3)
    geometry_parser.add_argument("--cuda-allocator-limit-gib", type=float, default=6.0)
    add_clock_args(geometry_parser)

    fid_parser = subparsers.add_parser("fid")
    fid_parser.add_argument("--output-root", type=Path, required=True)
    fid_parser.add_argument("--gpus", type=parse_gpus, default=(0, 1, 2, 3))
    fid_parser.add_argument(
        "--conditions", type=parse_conditions, default=ALL_CONDITIONS
    )
    add_sampling_args(fid_parser)

    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--repo", type=Path, required=True)
    worker_parser.add_argument("--data", type=Path, required=True)
    worker_parser.add_argument("--adm-python", type=Path, required=True)
    worker_parser.add_argument("--output-dir", type=Path, required=True)
    worker_parser.add_argument("--condition", choices=ALL_CONDITIONS, required=True)
    add_sampling_args(worker_parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("sample count must be positive")
    if hasattr(args, "batch_size") and args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if hasattr(args, "batch_size"):
        batch_seed(
            args.seed,
            (args.num_samples - 1) // args.batch_size,
            schema=args.batch_seed_schema,
        )
    if args.command == "geometry":
        geometry(args)
    elif args.command == "fid":
        fid(args)
    else:
        fid_worker(args)


if __name__ == "__main__":
    main()
