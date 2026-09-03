#!/usr/bin/env python3
"""Audit the information-purification interpretation of PFR Internal Guidance.

The experiment has two deliberately separate parts:

* ``geometry`` evaluates the complete 2x2 table ``(S_p, W_p, S_q, W_q)``
  along paired ordinary-IG and PFR trajectories; and
* ``fid`` scales only the weak counterfactual revision in

      G_lambda = G_IG - beta * lambda * (W_q - W_p).

Thus lambda zero is exactly ordinary IG and lambda one is exactly the deployed
projected-future-reference controller.  No model is trained in this audit.
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

from experiments.information_purification_ig import (  # noqa: E402
    FourCornerRevision,
    four_corner_revision,
    lambda_residualized_guidance,
    projected_information_query,
)
from experiments.path_evidence_pfr_bridge import (  # noqa: E402
    project_per_sample,
    sample_cosine,
    sample_rms,
)
from experiments.run_imagenet100_sit_internal_early_two_segment_gamma_sweep import (  # noqa: E402
    atomic_json,
    detect_adm_python,
    detect_data,
    detect_repo,
    load_repo_modules,
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


HISTORICAL_ORDINARY_FID1K = 64.85216325274007
HISTORICAL_PFR_FID1K = 61.924444086644996


def parse_floats(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated floats") from error
    if not values or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("values must be non-empty and unique")
    if any(not math.isfinite(item) for item in values):
        raise argparse.ArgumentTypeError("values must be finite")
    return values


def parse_times(value: str) -> tuple[float, ...]:
    values = parse_floats(value)
    if tuple(sorted(values)) != values:
        raise argparse.ArgumentTypeError("times must be increasing")
    if any(not 0.0 < item < INTERVENTION_TIME for item in values):
        raise argparse.ArgumentTypeError("times must lie in (0, 0.5)")
    return values


def lambda_tag(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


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


def sample_dot(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape != right.shape:
        raise ValueError("dot-product inputs must have identical shapes")
    return (left.float().flatten(1) * right.float().flatten(1)).sum(dim=1)


def symmetric_relative_change(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    denominator = 0.5 * (sample_rms(left) + sample_rms(right))
    return sample_rms(left - right) / denominator.clamp_min(1e-12)


class ResidualizedField:
    def __init__(
        self,
        runtime: Runtime,
        labels: torch.Tensor,
        residualization: float,
        *,
        record_diagnostics: bool = False,
    ) -> None:
        self.runtime = runtime
        self.labels = labels
        self.residualization = float(residualization)
        self.record_diagnostics = bool(record_diagnostics)
        self.nfe = 0
        self.query_nfe = 0
        self.diagnostics: dict[str, list[float]] = {}

    def _record(self, key: str, values: torch.Tensor) -> None:
        if not self.record_diagnostics:
            return
        self.diagnostics.setdefault(key, []).extend(
            values.detach().float().cpu().flatten().tolist()
        )

    def __call__(self, time_value: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        self.nfe += 1
        with torch.inference_mode():
            strong, weak = self.runtime.evaluate_pair(time_value, state, self.labels)
            scalar_time = float(time_value.detach().float().item())
            gamma = gamma_at(scalar_time)
            if gamma == 0.0:
                return strong
            gap = strong - weak
            guided = strong + gamma * gap
            if self.residualization == 0.0:
                return guided
            query = projected_information_query(
                state,
                time_value,
                strong_now=strong,
                weak_now=weak,
                guided_now=guided,
                gamma=gamma,
                horizon=HORIZON,
                intervention_time=INTERVENTION_TIME,
            )
            weak_query = self.runtime.evaluate_weak(
                query.time, query.state, self.labels
            )
            self.query_nfe += 1
            weak_revision = weak_query - weak
            beta = 1.0 + gamma
            projection = project_per_sample(weak_revision, gap)
            revision_rms = sample_rms(weak_revision)
            self._record("query_alpha", query.projection.coefficient)
            self._record("weak_revision_rms", revision_rms)
            self._record("gap_now_rms", sample_rms(gap))
            self._record("weak_revision_gap_cosine", sample_cosine(weak_revision, gap))
            self._record(
                "weak_revision_gap_parallel_energy_fraction",
                sample_rms(projection.parallel).square()
                / revision_rms.square().clamp_min(1e-30),
            )
            return lambda_residualized_guidance(
                guided,
                weak_revision,
                beta=beta,
                residualization=self.residualization,
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
    return odeint(
        field,
        state,
        grid,
        method="dopri5",
        atol=atol,
        rtol=rtol,
    )[1:]


def evaluate_four_corners(
    runtime: Runtime,
    state: torch.Tensor,
    labels: torch.Tensor,
    time_scalar: float,
    *,
    query_kind: str,
) -> tuple[FourCornerRevision, torch.Tensor, float]:
    time_value = torch.tensor(time_scalar, device=state.device)
    with torch.inference_mode():
        strong_now, weak_now = runtime.evaluate_pair(time_value, state, labels)
        gamma = gamma_at(time_scalar)
        guided = strong_now + gamma * (strong_now - weak_now)
        projected = projected_information_query(
            state,
            time_value,
            strong_now=strong_now,
            weak_now=weak_now,
            guided_now=guided,
            gamma=gamma,
            horizon=HORIZON,
            intervention_time=INTERVENTION_TIME,
        )
        if query_kind == "projected":
            query_state = projected.state
        elif query_kind == "time_only":
            query_state = state
        else:
            raise ValueError(query_kind)
        strong_query, weak_query = runtime.evaluate_pair(
            projected.time, query_state, labels
        )
    return (
        four_corner_revision(
            strong_now, weak_now, strong_query, weak_query
        ),
        projected.projection.coefficient,
        projected.horizon,
    )


def local_geometry_rows(
    *,
    trajectory: str,
    query_kind: str,
    time_scalar: float,
    parts: FourCornerRevision,
    query_alpha: torch.Tensor,
    horizon: float,
) -> list[dict[str, Any]]:
    weak_on_gap = project_per_sample(parts.weak_revision, parts.gap_now)
    strong_on_gap = project_per_sample(parts.strong_revision, parts.gap_now)
    weak_rms = sample_rms(parts.weak_revision)
    strong_rms = sample_rms(parts.strong_revision)
    gap_rms = sample_rms(parts.gap_now)
    cross_rms = sample_rms(parts.cross_corner_gap)
    interaction_rms = sample_rms(parts.interaction_revision)
    dot_gap_weak = sample_dot(parts.gap_now, parts.weak_revision)
    weak_energy = sample_dot(parts.weak_revision, parts.weak_revision)
    identity_error = sample_rms(
        parts.gap_query - (parts.gap_now - parts.interaction_revision)
    )
    rows: list[dict[str, Any]] = []
    for index in range(len(parts.gap_now)):
        rows.append(
            {
                "trajectory": trajectory,
                "query": query_kind,
                "sample": index,
                "time": time_scalar,
                "horizon": horizon,
                "query_alpha": float(query_alpha[index]),
                "gap_now_rms": float(gap_rms[index]),
                "gap_query_rms": float(sample_rms(parts.gap_query)[index]),
                "cross_corner_gap_rms": float(cross_rms[index]),
                "cross_to_gap_rms_ratio": float(
                    cross_rms[index] / gap_rms[index].clamp_min(1e-12)
                ),
                "weak_revision_rms": float(weak_rms[index]),
                "strong_revision_rms": float(strong_rms[index]),
                "strong_to_weak_revision_rms_ratio": float(
                    strong_rms[index] / weak_rms[index].clamp_min(1e-12)
                ),
                "interaction_revision_rms": float(interaction_rms[index]),
                "weak_revision_gap_cosine": float(
                    sample_cosine(parts.weak_revision, parts.gap_now)[index]
                ),
                "strong_revision_gap_cosine": float(
                    sample_cosine(parts.strong_revision, parts.gap_now)[index]
                ),
                "weak_strong_revision_cosine": float(
                    sample_cosine(parts.weak_revision, parts.strong_revision)[index]
                ),
                "gap_now_query_cosine": float(
                    sample_cosine(parts.gap_now, parts.gap_query)[index]
                ),
                "cross_gap_now_cosine": float(
                    sample_cosine(parts.cross_corner_gap, parts.gap_now)[index]
                ),
                "weak_revision_gap_parallel_energy_fraction": float(
                    sample_rms(weak_on_gap.parallel)[index].square()
                    / weak_rms[index].square().clamp_min(1e-30)
                ),
                "strong_revision_gap_parallel_energy_fraction": float(
                    sample_rms(strong_on_gap.parallel)[index].square()
                    / strong_rms[index].square().clamp_min(1e-30)
                ),
                "least_squares_lambda_D_on_Iw": float(
                    dot_gap_weak[index] / weak_energy[index].clamp_min(1e-30)
                ),
                "four_corner_identity_error_rms": float(identity_error[index]),
            }
        )
    return rows


def grouped_means(
    rows: list[dict[str, Any]], keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    output: list[dict[str, Any]] = []
    excluded = {*keys, "sample"}
    for group_key, group_rows in sorted(groups.items()):
        result = dict(zip(keys, group_key, strict=True))
        for column in group_rows[0]:
            if column in excluded:
                continue
            values = [row[column] for row in group_rows]
            if all(isinstance(value, (int, float)) for value in values):
                result[f"{column}_mean"] = statistics.mean(float(value) for value in values)
                result[f"{column}_median"] = statistics.median(float(value) for value in values)
        output.append(result)
    return output


def geometry(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    repo = detect_repo()
    data = detect_data()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
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

    trajectory_states: dict[str, torch.Tensor] = {}
    for name, residualization in (("ordinary", 0.0), ("pfr", 1.0)):
        field = ResidualizedField(runtime, labels, residualization)
        trajectory_states[name] = integrate_times(
            field,
            noise.float(),
            args.times,
            atol=args.atol,
            rtol=args.rtol,
        )
        print(
            json.dumps(
                {
                    "event": "trajectory_complete",
                    "trajectory": name,
                    "nfe": field.nfe,
                    "query_nfe": field.query_nfe,
                }
            ),
            flush=True,
        )

    rows: list[dict[str, Any]] = []
    tensors: dict[tuple[str, str, float], FourCornerRevision] = {}
    for trajectory, states in trajectory_states.items():
        for time_scalar, state in zip(args.times, states, strict=True):
            for query_kind in ("time_only", "projected"):
                parts, query_alpha, horizon = evaluate_four_corners(
                    runtime,
                    state,
                    labels,
                    time_scalar,
                    query_kind=query_kind,
                )
                tensors[(trajectory, query_kind, time_scalar)] = parts
                rows.extend(
                    local_geometry_rows(
                        trajectory=trajectory,
                        query_kind=query_kind,
                        time_scalar=time_scalar,
                        parts=parts,
                        query_alpha=query_alpha,
                        horizon=horizon,
                    )
                )

    stability_rows: list[dict[str, Any]] = []
    for time_scalar in args.times:
        state_ordinary = trajectory_states["ordinary"][args.times.index(time_scalar)]
        state_pfr = trajectory_states["pfr"][args.times.index(time_scalar)]
        for query_kind in ("time_only", "projected"):
            ordinary = tensors[("ordinary", query_kind, time_scalar)]
            pfr = tensors[("pfr", query_kind, time_scalar)]
            fields = {
                "gap_now": (ordinary.gap_now, pfr.gap_now),
                "cross_corner_gap": (
                    ordinary.cross_corner_gap,
                    pfr.cross_corner_gap,
                ),
                "weak_revision": (
                    ordinary.weak_revision,
                    pfr.weak_revision,
                ),
            }
            for index in range(args.num_samples):
                row: dict[str, Any] = {
                    "query": query_kind,
                    "sample": index,
                    "time": time_scalar,
                    "state_shift_rms": float(
                        sample_rms(state_pfr - state_ordinary)[index]
                    ),
                }
                for name, (ordinary_value, pfr_value) in fields.items():
                    row[f"{name}_trajectory_cosine"] = float(
                        sample_cosine(ordinary_value, pfr_value)[index]
                    )
                    row[f"{name}_symmetric_relative_change"] = float(
                        symmetric_relative_change(ordinary_value, pfr_value)[index]
                    )
                stability_rows.append(row)

    write_csv(output / "four_corner_geometry.csv", rows)
    write_csv(output / "four_corner_geometry_summary.csv", grouped_means(
        rows, ("trajectory", "query", "time")
    ))
    write_csv(output / "trajectory_stability.csv", stability_rows)
    write_csv(output / "trajectory_stability_summary.csv", grouped_means(
        stability_rows, ("query", "time")
    ))
    summary = {
        "format": "eqvae_information_purification_geometry_v1",
        "protocol": {
            "strong": str(runtime.paths["strong"]),
            "weak": str(runtime.paths["depth4"]),
            "weights": "ema",
            "times": list(args.times),
            "horizon": HORIZON,
            "intervention_time": INTERVENTION_TIME,
            "num_samples": args.num_samples,
            "seed": args.seed,
            "queries": ["time_only", "projected"],
            "trajectories": ["ordinary", "pfr"],
        },
        "overall": {
            key: summarize([float(row[key]) for row in rows])
            for key in rows[0]
            if key not in {"trajectory", "query", "sample", "time"}
        },
        "stability": {
            key: summarize([float(row[key]) for row in stability_rows])
            for key in stability_rows[0]
            if key not in {"query", "sample", "time"}
        },
        "allocator": allocator,
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps({"event": "geometry_complete", "output": str(output)}), flush=True)


def result_reusable(path: Path, residualization: float, args: argparse.Namespace) -> bool:
    if not path.is_file():
        return False
    try:
        result = read_json(path)
        manifest = result["sampling_manifest"]
        metrics = result["metrics"]
        return (
            float(result["residualization"]) == float(residualization)
            and int(manifest["sampling"]["num_samples"]) == args.num_samples
            and int(manifest["sampling"]["batch_size"]) == args.batch_size
            and int(manifest["sampling"]["seed"]) == args.seed
            and all(math.isfinite(float(metrics[key])) for key in ("fid", "sfid", "inception_score"))
        )
    except Exception:
        return False


def fid_worker(args: argparse.Namespace) -> None:
    import numpy as np
    from diffusers.models import AutoencoderKL
    from torchvision.utils import save_image

    repo = Path(args.repo).resolve()
    data = Path(args.data).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "condition_result.json"
    if result_reusable(result_path, args.residualization, args):
        print(json.dumps({"event": "reuse", "lambda": args.residualization}), flush=True)
        return
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    runtime, allocator = load_runtime(
        repo=repo,
        data=data,
        adm_python=Path(args.adm_python),
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
            generator = torch.Generator(device=device).manual_seed(args.seed + batch_index)
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
            field = ResidualizedField(
                runtime,
                labels,
                args.residualization,
                record_diagnostics=True,
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
                raise FloatingPointError(f"lambda={args.residualization}")
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
                            "lambda": args.residualization,
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
    assert preview is not None
    save_image(preview, output / "preview.png", nrow=4, normalize=True, value_range=(-1, 1))
    sampling_manifest = {
        "format": "eqvae_information_purification_samples_v1",
        "formula": "G_lambda=G_IG-beta*lambda*(W_q-W_p)",
        "residualization": args.residualization,
        "sampling": {
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "integrator": "dopri5",
            "atol": args.atol,
            "rtol": args.rtol,
        },
        "query": {
            "kind": "projected_minimum_spatial_intervention",
            "horizon": HORIZON,
            "intervention_time": INTERVENTION_TIME,
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
        "format": "eqvae_information_purification_result_v1",
        "residualization": args.residualization,
        "sampling_manifest": sampling_manifest,
        "metrics": metrics,
        "sample_retained": bool(args.keep_samples),
    }
    atomic_json(result_path, result)
    if not args.keep_samples:
        sample_path.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "event": "complete",
                "lambda": args.residualization,
                "fid": metrics["fid"],
            }
        ),
        flush=True,
    )


def run_one_lambda(
    residualization: float,
    gpu: int,
    root: Path,
    repo: Path,
    data: Path,
    adm_python: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    output = root / f"lambda_{lambda_tag(residualization)}"
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "condition_result.json"
    if result_reusable(result_path, residualization, args):
        result = read_json(result_path)
        print(f"[reuse] lambda={residualization:g}", flush=True)
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
        "--residualization",
        str(residualization),
        "--num-samples",
        str(args.num_samples),
        "--batch-size",
        str(args.batch_size),
        "--vae-decode-batch-size",
        str(args.vae_decode_batch_size),
        "--seed",
        str(args.seed),
        "--atol",
        str(args.atol),
        "--rtol",
        str(args.rtol),
        "--cuda-allocator-limit-gib",
        str(args.cuda_allocator_limit_gib),
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
        raise RuntimeError(f"lambda={residualization:g} failed on GPU {gpu}\n{tail}")
    result = read_json(result_path)
    print(
        f"[GPU {gpu}] lambda={residualization:g}: "
        f"FID={float(result['metrics']['fid']):.4f}",
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
    lanes: list[list[float]] = [[] for _ in args.gpus]
    for index, residualization in enumerate(args.residualizations):
        lanes[index % len(args.gpus)].append(residualization)

    def lane(gpu: int, values: list[float]) -> list[dict[str, Any]]:
        return [
            run_one_lambda(value, gpu, root, repo, data, adm_python, args)
            for value in values
        ]

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [
            pool.submit(lane, gpu, values)
            for gpu, values in zip(args.gpus, lanes, strict=True)
            if values
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
                "lambda": float(result["residualization"]),
                "fid": float(metrics["fid"]),
                "sfid": float(metrics["sfid"]),
                "inception_score": float(metrics["inception_score"]),
                "total_nfe": int(manifest["total_nfe"]),
                "total_query_nfe": int(manifest["total_query_nfe"]),
                "weak_revision_gap_cosine_mean": (
                    None
                    if not diagnostics.get("weak_revision_gap_cosine")
                    else diagnostics["weak_revision_gap_cosine"]["mean"]
                ),
                "weak_revision_gap_parallel_energy_fraction_mean": (
                    None
                    if not diagnostics.get("weak_revision_gap_parallel_energy_fraction")
                    else diagnostics["weak_revision_gap_parallel_energy_fraction"]["mean"]
                ),
                "noise_sha256": manifest["noise_sha256"],
                "label_sha256": manifest["label_sha256"],
            }
        )
    rows.sort(key=lambda row: float(row["lambda"]))
    if len({row["noise_sha256"] for row in rows}) != 1:
        raise RuntimeError("lambda conditions did not use paired noise")
    if len({row["label_sha256"] for row in rows}) != 1:
        raise RuntimeError("lambda conditions did not use paired labels")
    if args.num_samples == 1000 and args.batch_size == 8 and args.seed == 0:
        if rows[0]["noise_sha256"] != EXPECTED_NOISE:
            raise RuntimeError("noise differs from the historical FID-1K bank")
        if rows[0]["label_sha256"] != EXPECTED_LABEL:
            raise RuntimeError("labels differ from the historical FID-1K bank")
        anchors = {float(row["lambda"]): float(row["fid"]) for row in rows}
        if 0.0 in anchors and abs(anchors[0.0] - HISTORICAL_ORDINARY_FID1K) > 0.15:
            raise RuntimeError("ordinary IG anchor did not reproduce")
        if 1.0 in anchors and abs(anchors[1.0] - HISTORICAL_PFR_FID1K) > 0.15:
            raise RuntimeError("PFR anchor did not reproduce")
    write_csv(root / "summary.csv", rows)
    summary = {
        "format": "eqvae_information_purification_fid_sweep_v1",
        "formula": "G_lambda=G_IG-beta*lambda*(W_q-W_p)",
        "protocol": {
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "residualizations": list(args.residualizations),
            "horizon": HORIZON,
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
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--cuda-allocator-limit-gib", type=float, default=6.0)
    parser.add_argument("--fid-batch-size", type=int, default=16)
    parser.add_argument("--fid-gpu-memory-fraction", type=float, default=0.25)
    parser.add_argument("--keep-samples", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    geometry_parser = subparsers.add_parser("geometry")
    geometry_parser.add_argument("--output-dir", type=Path, required=True)
    geometry_parser.add_argument("--device", default="cuda:0")
    geometry_parser.add_argument("--num-samples", type=int, default=16)
    geometry_parser.add_argument("--seed", type=int, default=20260903)
    geometry_parser.add_argument(
        "--times", type=parse_times, default=(0.05, 0.15, 0.24, 0.3, 0.4, 0.46875)
    )
    geometry_parser.add_argument("--atol", type=float, default=1e-6)
    geometry_parser.add_argument("--rtol", type=float, default=1e-3)
    geometry_parser.add_argument("--cuda-allocator-limit-gib", type=float, default=6.0)

    fid_parser = subparsers.add_parser("fid")
    fid_parser.add_argument("--output-root", type=Path, required=True)
    fid_parser.add_argument("--gpus", type=parse_gpus, default=(3,))
    fid_parser.add_argument(
        "--residualizations",
        type=parse_floats,
        default=(0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
    )
    add_sampling_args(fid_parser)

    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--repo", type=Path, required=True)
    worker_parser.add_argument("--data", type=Path, required=True)
    worker_parser.add_argument("--adm-python", type=Path, required=True)
    worker_parser.add_argument("--output-dir", type=Path, required=True)
    worker_parser.add_argument("--residualization", type=float, required=True)
    add_sampling_args(worker_parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("sample count must be positive")
    if hasattr(args, "batch_size") and args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.command == "geometry":
        geometry(args)
    elif args.command == "fid":
        fid(args)
    else:
        fid_worker(args)


if __name__ == "__main__":
    main()
