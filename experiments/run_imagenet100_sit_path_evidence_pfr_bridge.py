#!/usr/bin/env python3
"""Test whether finite-horizon path evidence explains the PFR revision.

This runner keeps the established ImageNet-100 SiT-v800/depth-4 IG protocol
fixed.  It compares the deployed PFR revision with a deterministic proxy for
the future Feynman--Kac evidence gradient, first geometrically and then with
paired ADM FID-1K.

The evidence proxy is intentionally named as such.  It follows one nominal IG
Euler step and differentiates a trapezoidal path-cost quadrature; it is not the
exact stochastic Doob value.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.path_evidence_pfr_bridge import (  # noqa: E402
    finite_horizon_nominal_evidence_gradient,
    match_sample_rms,
    pfr_revision,
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


EXPECTED_NOISE = "ab8419c7fdfd5b15dacbf4d37a3d567158e4332f25fd94580d3df73bac87e2c2"
EXPECTED_LABEL = "7c3ae6894e7ebab5c9b6524606f03b6a56b38dccbe472ff40edde26e48654fe6"
HORIZON = 1.0 / 32.0
INTERVENTION_TIME = 0.5
MODES = (
    "pfr_evidence_parallel",
    "pfr_evidence_orthogonal",
    "evidence_rms_matched",
    "evidence_rms_matched_negative",
)
ALL_MODES = ("ordinary_ig", "pfr_full", *MODES)


def gamma_at(time_value: float) -> float:
    if time_value < 0.25:
        return 0.6
    if time_value < INTERVENTION_TIME:
        return 0.7
    return 0.0


def parse_modes(value: str) -> tuple[str, ...]:
    modes = tuple(item.strip() for item in value.split(",") if item.strip())
    if not modes or len(set(modes)) != len(modes) or not set(modes) <= set(ALL_MODES):
        raise argparse.ArgumentTypeError(
            "modes must be unique values from " + ",".join(ALL_MODES)
        )
    return modes


def parse_times(value: str) -> tuple[float, ...]:
    try:
        times = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("times must be comma-separated floats") from error
    if not times or any(not 0.0 < item < INTERVENTION_TIME for item in times):
        raise argparse.ArgumentTypeError("times must lie in (0,0.5)")
    if tuple(sorted(set(times))) != times:
        raise argparse.ArgumentTypeError("times must be unique and increasing")
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
        raise ValueError("cannot write empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mode_formula(mode: str) -> str:
    formulas = {
        "ordinary_ig": "G",
        "pfr_full": "G+r_pfr",
        "pfr_evidence_parallel": "G+Proj_uPE(r_pfr)",
        "pfr_evidence_orthogonal": "G+r_pfr-Proj_uPE(r_pfr)",
        "evidence_rms_matched": "G+RMS_match(u_PE,r_pfr)",
        "evidence_rms_matched_negative": "G-RMS_match(u_PE,r_pfr)",
    }
    return formulas[mode]


@dataclass
class Runtime:
    modules: dict[str, Any]
    strong: Any
    head: Any
    strong_metadata: dict[str, Any]
    paths: dict[str, Path]
    device: torch.device

    def evaluate_pair(
        self, time_value: torch.Tensor, state: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        times = time_value.expand(len(state))
        full, trained, _ = self.modules["evaluate_source_with_heads"](
            self.strong,
            state,
            times,
            labels,
            heads={"depth4_v": self.head},
        )
        return full, trained["depth4_v"]

    def evaluate_weak(
        self, time_value: torch.Tensor, state: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        from experiments.imagenet100_sit_multiscale_models import (
            evaluate_internal_head_only,
        )

        return evaluate_internal_head_only(
            self.strong,
            state,
            time_value.expand(len(state)),
            labels,
            spec=self.head,
        )


def load_runtime(
    *,
    repo: Path,
    data: Path,
    adm_python: Path,
    device: torch.device,
    allocator_limit_gib: float,
) -> tuple[Runtime, dict[str, Any]]:
    paths = runtime_paths(repo, data, adm_python)
    modules = load_repo_modules(repo)
    if allocator_limit_gib > 0.0:
        allocator = modules["configure_cuda_allocator"](
            device, limit_gib=allocator_limit_gib
        )
    else:
        torch.cuda.reset_peak_memory_stats(device)
        allocator = {
            "allocator_limit_gib": None,
            "allocator_fraction": None,
            "device_total_memory_bytes": int(
                torch.cuda.get_device_properties(device).total_memory
            ),
        }
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    sit_module, source_metadata = modules["load_official_sit_module"](
        Path(modules["DEFAULT_OFFICIAL_SIT_REPO"]).expanduser().resolve(),
        verify_source=True,
    )
    strong, semantics, strong_metadata = modules["load_sit_field_model"](
        checkpoint_path=paths["strong"],
        weights="ema",
        sit_module=sit_module,
        source_metadata=source_metadata,
        device=device,
    )
    if semantics.prediction_target != "velocity":
        raise ValueError("the bridge requires the native-velocity v800 source")
    head = modules["load_internal_head_for_source"](
        checkpoint_path=paths["depth4"],
        name="depth4_v",
        head_weights="ema",
        model=strong,
        sit_module=sit_module,
        source_checkpoint_path=paths["strong"],
        source_metadata=source_metadata,
        device=device,
    )
    return (
        Runtime(
            modules=modules,
            strong=strong,
            head=head,
            strong_metadata=strong_metadata,
            paths=paths,
            device=device,
        ),
        allocator,
    )


@dataclass(frozen=True)
class BridgeParts:
    strong: torch.Tensor
    weak: torch.Tensor
    guided: torch.Tensor
    pfr_revision: torch.Tensor
    evidence_direction: torch.Tensor
    parallel: torch.Tensor
    orthogonal: torch.Tensor
    matched: torch.Tensor
    query_alpha: torch.Tensor
    evidence_value: torch.Tensor
    forward_max_abs_error: float


def compute_bridge_parts(
    runtime: Runtime,
    time_value: torch.Tensor,
    state: torch.Tensor,
    labels: torch.Tensor,
) -> BridgeParts:
    scalar_time = float(time_value.detach().float().item())
    gamma = gamma_at(scalar_time)
    with torch.inference_mode():
        strong, weak = runtime.evaluate_pair(time_value, state, labels)
        guided = strong + gamma * (strong - weak)
        revision, _, _, query_alpha = pfr_revision(
            state,
            time_value,
            strong=strong,
            weak=weak,
            guided=guided,
            gamma=gamma,
            horizon=HORIZON,
            intervention_time=INTERVENTION_TIME,
            evaluate_weak=lambda query_time, query_state: runtime.evaluate_weak(
                query_time, query_state, labels
            ),
        )

    if gamma == 0.0:
        zeros = torch.zeros_like(state)
        return BridgeParts(
            strong=strong,
            weak=weak,
            guided=guided,
            pfr_revision=zeros,
            evidence_direction=zeros,
            parallel=zeros,
            orthogonal=zeros,
            matched=zeros,
            query_alpha=query_alpha,
            evidence_value=torch.zeros(len(state), device=state.device),
            forward_max_abs_error=0.0,
        )

    with torch.inference_mode(False), torch.enable_grad():
        differentiable_state = state.detach().clone().requires_grad_(True)
        differentiable_time = time_value.detach().clone()
        evidence = finite_horizon_nominal_evidence_gradient(
            differentiable_state,
            differentiable_time,
            horizon=HORIZON,
            intervention_time=INTERVENTION_TIME,
            evaluate_pair=lambda query_time, query_state: runtime.evaluate_pair(
                query_time, query_state, labels
            ),
            gamma_at=gamma_at,
        )
        evidence_direction = evidence.gradient.detach()
        evidence_value = evidence.value.detach()
        forward_error = max(
            float((evidence.strong.detach() - strong).abs().max().cpu()),
            float((evidence.weak.detach() - weak).abs().max().cpu()),
        )
    projection = project_per_sample(revision, evidence_direction)
    matched = match_sample_rms(evidence_direction, revision)
    return BridgeParts(
        strong=strong,
        weak=weak,
        guided=guided,
        pfr_revision=revision,
        evidence_direction=evidence_direction,
        parallel=projection.parallel,
        orthogonal=projection.orthogonal,
        matched=matched,
        query_alpha=query_alpha,
        evidence_value=evidence_value,
        forward_max_abs_error=forward_error,
    )


def field_value(mode: str, parts: BridgeParts) -> torch.Tensor:
    if mode == "ordinary_ig":
        return parts.guided
    if mode == "pfr_full":
        return parts.guided + parts.pfr_revision
    if mode == "pfr_evidence_parallel":
        return parts.guided + parts.parallel
    if mode == "pfr_evidence_orthogonal":
        return parts.guided + parts.orthogonal
    if mode == "evidence_rms_matched":
        return parts.guided + parts.matched
    if mode == "evidence_rms_matched_negative":
        return parts.guided - parts.matched
    raise ValueError(mode)


class BridgeField:
    def __init__(self, runtime: Runtime, labels: torch.Tensor, mode: str) -> None:
        self.runtime = runtime
        self.labels = labels
        self.mode = mode
        self.nfe = 0
        self.diagnostics: dict[str, list[float]] = {}

    def _record(self, name: str, values: torch.Tensor) -> None:
        self.diagnostics.setdefault(name, []).extend(
            values.detach().float().cpu().flatten().tolist()
        )

    def __call__(self, time_value: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        self.nfe += 1
        scalar_time = float(time_value.detach().float().item())
        if scalar_time >= INTERVENTION_TIME:
            with torch.inference_mode():
                return self.runtime.evaluate_pair(time_value, state, self.labels)[0]
        parts = compute_bridge_parts(
            self.runtime, time_value, state, self.labels
        )
        revision_rms = sample_rms(parts.pfr_revision)
        parallel_rms = sample_rms(parts.parallel)
        orthogonal_rms = sample_rms(parts.orthogonal)
        denominator = revision_rms.square().clamp_min(torch.finfo(torch.float32).tiny)
        self._record(
            "pfr_evidence_cosine",
            sample_cosine(parts.pfr_revision, parts.evidence_direction),
        )
        self._record(
            "parallel_energy_fraction", parallel_rms.square() / denominator
        )
        self._record("pfr_revision_rms", revision_rms)
        self._record("evidence_direction_rms", sample_rms(parts.evidence_direction))
        self._record("parallel_rms", parallel_rms)
        self._record("orthogonal_rms", orthogonal_rms)
        self._record("query_alpha", parts.query_alpha)
        self._record("evidence_value", parts.evidence_value)
        self.diagnostics.setdefault("autograd_forward_max_abs_error", []).append(
            parts.forward_max_abs_error
        )
        return field_value(self.mode, parts)


class OrdinaryIGField:
    def __init__(self, runtime: Runtime, labels: torch.Tensor) -> None:
        self.runtime = runtime
        self.labels = labels
        self.nfe = 0

    def __call__(self, time_value: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        self.nfe += 1
        with torch.inference_mode():
            strong, weak = self.runtime.evaluate_pair(time_value, state, self.labels)
            gamma = gamma_at(float(time_value.detach().float().item()))
            return strong + gamma * (strong - weak)


def integrate(
    field: Any,
    state: torch.Tensor,
    start: float,
    end: float,
    *,
    atol: float,
    rtol: float,
) -> torch.Tensor:
    from torchdiffeq import odeint

    return odeint(
        field,
        state,
        torch.tensor([start, end], device=state.device),
        method="dopri5",
        atol=atol,
        rtol=rtol,
    )[-1]


def normalize_impulse(direction: torch.Tensor, rms: float) -> torch.Tensor:
    target = torch.full_like(direction, float(rms))
    return match_sample_rms(direction, target)


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
    local_rows: list[dict[str, Any]] = []
    transport_rows: list[dict[str, Any]] = []
    started = torch.cuda.Event(enable_timing=True)
    finished = torch.cuda.Event(enable_timing=True)
    started.record()

    for time_scalar in args.times:
        ordinary = OrdinaryIGField(runtime, labels)
        state = integrate(
            ordinary,
            noise,
            0.0,
            time_scalar,
            atol=args.atol,
            rtol=args.rtol,
        )
        time_value = torch.tensor(time_scalar, device=device)
        parts = compute_bridge_parts(runtime, time_value, state, labels)
        projection = project_per_sample(
            parts.pfr_revision, parts.evidence_direction
        )
        revision_rms = sample_rms(parts.pfr_revision)
        evidence_rms = sample_rms(parts.evidence_direction)
        parallel_rms = sample_rms(projection.parallel)
        orthogonal_rms = sample_rms(projection.orthogonal)
        cosine = sample_cosine(parts.pfr_revision, parts.evidence_direction)
        for index in range(args.num_samples):
            denominator = max(float(revision_rms[index].square()), 1e-30)
            local_rows.append(
                {
                    "sample": index,
                    "time": time_scalar,
                    "pfr_evidence_cosine": float(cosine[index]),
                    "pfr_revision_rms": float(revision_rms[index]),
                    "evidence_direction_rms": float(evidence_rms[index]),
                    "parallel_rms": float(parallel_rms[index]),
                    "orthogonal_rms": float(orthogonal_rms[index]),
                    "parallel_energy_fraction": float(
                        parallel_rms[index].square() / denominator
                    ),
                    "query_alpha": float(parts.query_alpha[index]),
                    "evidence_value": float(parts.evidence_value[index]),
                    "autograd_forward_max_abs_error": parts.forward_max_abs_error,
                }
            )

        base_field = OrdinaryIGField(runtime, labels)
        base_endpoint = integrate(
            base_field,
            state,
            time_scalar,
            1.0,
            atol=args.atol,
            rtol=args.rtol,
        )
        directions = {
            "pfr_full": parts.pfr_revision,
            "evidence": parts.evidence_direction,
            "pfr_parallel": projection.parallel,
            "pfr_orthogonal": projection.orthogonal,
        }
        endpoint_shifts: dict[str, torch.Tensor] = {}
        for name, direction in directions.items():
            impulse = normalize_impulse(direction, args.impulse_rms)
            field = OrdinaryIGField(runtime, labels)
            endpoint = integrate(
                field,
                state + impulse,
                time_scalar,
                1.0,
                atol=args.atol,
                rtol=args.rtol,
            )
            endpoint_shifts[name] = endpoint - base_endpoint

        pfr_shift = endpoint_shifts["pfr_full"]
        for index in range(args.num_samples):
            transport_rows.append(
                {
                    "sample": index,
                    "time": time_scalar,
                    "impulse_rms": args.impulse_rms,
                    "terminal_pfr_rms": float(sample_rms(pfr_shift)[index]),
                    "terminal_evidence_rms": float(
                        sample_rms(endpoint_shifts["evidence"])[index]
                    ),
                    "terminal_parallel_rms": float(
                        sample_rms(endpoint_shifts["pfr_parallel"])[index]
                    ),
                    "terminal_orthogonal_rms": float(
                        sample_rms(endpoint_shifts["pfr_orthogonal"])[index]
                    ),
                    "terminal_pfr_evidence_cosine": float(
                        sample_cosine(pfr_shift, endpoint_shifts["evidence"])[index]
                    ),
                    "terminal_pfr_parallel_cosine": float(
                        sample_cosine(pfr_shift, endpoint_shifts["pfr_parallel"])[index]
                    ),
                    "terminal_pfr_orthogonal_cosine": float(
                        sample_cosine(pfr_shift, endpoint_shifts["pfr_orthogonal"])[index]
                    ),
                }
            )
        print(
            json.dumps(
                {
                    "event": "geometry_time_complete",
                    "time": time_scalar,
                    "mean_local_cosine": float(cosine.mean()),
                    "mean_projection_energy": float(
                        (parallel_rms.square() / revision_rms.square().clamp_min(1e-30)).mean()
                    ),
                    "mean_terminal_cosine": float(
                        sample_cosine(pfr_shift, endpoint_shifts["evidence"]).mean()
                    ),
                }
            ),
            flush=True,
        )

    finished.record()
    torch.cuda.synchronize(device)
    write_csv(output / "local_geometry.csv", local_rows)
    write_csv(output / "transport_geometry.csv", transport_rows)
    summary = {
        "format": "eqvae_path_evidence_pfr_bridge_geometry_v1",
        "protocol": {
            "strong": str(runtime.paths["strong"]),
            "weak": str(runtime.paths["depth4"]),
            "weights": "ema",
            "times": list(args.times),
            "horizon": HORIZON,
            "intervention_time": INTERVENTION_TIME,
            "num_samples": args.num_samples,
            "seed": args.seed,
            "impulse_rms": args.impulse_rms,
            "atol": args.atol,
            "rtol": args.rtol,
            "evidence_proxy": (
                "gradient of one-Euler-step trapezoidal nominal path-evidence cost"
            ),
        },
        "local": {
            key: summarize([float(row[key]) for row in local_rows])
            for key in local_rows[0]
            if key not in {"sample", "time"}
        },
        "transport": {
            key: summarize([float(row[key]) for row in transport_rows])
            for key in transport_rows[0]
            if key not in {"sample", "time"}
        },
        "elapsed_seconds": started.elapsed_time(finished) / 1000.0,
        "allocator": allocator,
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    atomic_json(output / "summary.json", summary)


def result_reusable(path: Path, mode: str, args: argparse.Namespace) -> bool:
    if not path.is_file():
        return False
    try:
        result = read_json(path)
        manifest = result["sampling_manifest"]
        metrics = result["metrics"]
        return (
            result["mode"] == mode
            and int(manifest["sampling"]["num_samples"]) == args.num_samples
            and int(manifest["sampling"]["batch_size"]) == args.batch_size
            and int(manifest["sampling"]["seed"]) == args.seed
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

    repo = Path(args.repo).resolve()
    data = Path(args.data).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "condition_result.json"
    if result_reusable(result_path, args.mode, args):
        print(json.dumps({"event": "reuse", "mode": args.mode}), flush=True)
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
            field = BridgeField(runtime, labels, args.mode)
            endpoint = integrate(
                field,
                noise.float(),
                0.0,
                1.0,
                atol=args.atol,
                rtol=args.rtol,
            )
            if not torch.isfinite(endpoint).all():
                raise FloatingPointError(args.mode)
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
            for key, values in field.diagnostics.items():
                diagnostics.setdefault(key, []).extend(values)
            cursor = stop
            if cursor == current_batch or cursor == args.num_samples or cursor % 128 == 0:
                print(
                    json.dumps(
                        {
                            "mode": args.mode,
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
    save_image(preview, output / "preview.png", nrow=4, normalize=True, value_range=(-1, 1))
    sampling_manifest = {
        "format": "eqvae_path_evidence_pfr_bridge_samples_v1",
        "mode": args.mode,
        "formula": mode_formula(args.mode),
        "sampling": {
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "integrator": "dopri5",
            "atol": args.atol,
            "rtol": args.rtol,
        },
        "bridge": {
            "horizon": HORIZON,
            "intervention_time": INTERVENTION_TIME,
            "evidence_proxy": (
                "gradient of one-Euler-step trapezoidal nominal path-evidence cost"
            ),
            "rate": "beta*(beta-1)*t/(1-t)*mean((S-W)^2)",
            "warning": "deterministic finite-horizon proxy, not exact stochastic Doob value",
        },
        "strong": runtime.strong_metadata,
        "weak_checkpoint": str(runtime.paths["depth4"]),
        "noise_sha256": noise_hash.hexdigest(),
        "label_sha256": label_hash.hexdigest(),
        "total_nfe": total_nfe,
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
        "format": "eqvae_path_evidence_pfr_bridge_result_v1",
        "mode": args.mode,
        "formula": mode_formula(args.mode),
        "sampling_manifest": sampling_manifest,
        "metrics": metrics,
        "sample_retained": bool(args.keep_samples),
    }
    atomic_json(result_path, result)
    if not args.keep_samples:
        sample_path.unlink(missing_ok=True)
    print(
        json.dumps({"event": "complete", "mode": args.mode, "fid": metrics["fid"]}),
        flush=True,
    )


def run_one_mode(
    args: argparse.Namespace,
    *,
    gpu: int,
    mode: str,
    repo: Path,
    data: Path,
    adm_python: Path,
) -> dict[str, Any]:
    output = args.output_root / mode
    result_path = output / "condition_result.json"
    if result_reusable(result_path, mode, args):
        result = read_json(result_path)
        print(f"[reuse] {mode}: FID={float(result['metrics']['fid']):.4f}", flush=True)
        return result
    output.mkdir(parents=True, exist_ok=True)
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
        "--mode",
        mode,
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
        tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-80:])
        raise RuntimeError(f"{mode} failed on GPU {gpu}\n{tail}")
    result = read_json(result_path)
    print(f"[GPU {gpu}] {mode}: FID={float(result['metrics']['fid']):.4f}", flush=True)
    return result


def locate_anchor(data: Path, kind: str) -> dict[str, Any]:
    if kind == "ordinary_ig":
        path = (
            data
            / "internal_guidance_path_endpoint_v1/fid1k_coarse/"
            "ig_depth4_best_global/condition_result.json"
        )
    elif kind == "pfr_full":
        path = (
            data
            / "calibration_split_ig_v1/canonical_ablation_fid1k/"
            "fmd_decomposition_weak_calibration_projected_h0p03125/"
            "condition_result.json"
        )
    else:
        raise ValueError(kind)
    result = read_json(path)
    return {
        "mode": kind,
        "source": str(path),
        "formula": mode_formula(kind),
        "metrics": result["metrics"],
        "noise_sha256": result["sampling_manifest"]["noise_sha256"],
        "label_sha256": result["sampling_manifest"]["label_sha256"],
    }


def fid(args: argparse.Namespace) -> None:
    repo = detect_repo()
    data = detect_data()
    adm_python = detect_adm_python()
    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    lanes: list[list[str]] = [[] for _ in args.gpus]
    for index, mode in enumerate(args.modes):
        lanes[index % len(args.gpus)].append(mode)

    def lane(gpu: int, modes: list[str]) -> list[dict[str, Any]]:
        return [
            run_one_mode(
                args,
                gpu=gpu,
                mode=mode,
                repo=repo,
                data=data,
                adm_python=adm_python,
            )
            for mode in modes
        ]

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [
            pool.submit(lane, gpu, modes)
            for gpu, modes in zip(args.gpus, lanes)
            if modes
        ]
        for future in as_completed(futures):
            results.extend(future.result())

    rows: list[dict[str, Any]] = []
    anchors = [locate_anchor(data, "ordinary_ig"), locate_anchor(data, "pfr_full")]
    for anchor in anchors:
        rows.append(
            {
                "mode": anchor["mode"],
                "formula": anchor["formula"],
                "fid": float(anchor["metrics"]["fid"]),
                "sfid": float(anchor["metrics"]["sfid"]),
                "inception_score": float(anchor["metrics"]["inception_score"]),
                "noise_sha256": anchor["noise_sha256"],
                "label_sha256": anchor["label_sha256"],
                "source": anchor["source"],
            }
        )
    for result in results:
        manifest = result["sampling_manifest"]
        rows.append(
            {
                "mode": result["mode"],
                "formula": result["formula"],
                "fid": float(result["metrics"]["fid"]),
                "sfid": float(result["metrics"]["sfid"]),
                "inception_score": float(result["metrics"]["inception_score"]),
                "noise_sha256": manifest["noise_sha256"],
                "label_sha256": manifest["label_sha256"],
                "source": str(args.output_root / result["mode"] / "condition_result.json"),
            }
        )
    if {row["noise_sha256"] for row in rows} != {EXPECTED_NOISE}:
        raise RuntimeError("noise bank is not the historical paired FID-1K bank")
    if {row["label_sha256"] for row in rows} != {EXPECTED_LABEL}:
        raise RuntimeError("label bank is not the historical paired FID-1K bank")
    rows.sort(key=lambda row: float(row["fid"]))
    write_csv(args.output_root / "summary.csv", rows)
    summary = {
        "format": "eqvae_path_evidence_pfr_bridge_summary_v1",
        "protocol": {
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "atol": args.atol,
            "rtol": args.rtol,
            "paired": True,
            "horizon": HORIZON,
        },
        "warning": "FID-1K is a screening metric; the evidence field is a deterministic proxy.",
        "rows": rows,
        "best": rows[0],
    }
    atomic_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def add_common_sampling_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--vae-decode-batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--cuda-allocator-limit-gib", type=float, default=20.0)
    parser.add_argument("--fid-batch-size", type=int, default=16)
    parser.add_argument("--fid-gpu-memory-fraction", type=float, default=0.25)
    parser.add_argument("--keep-samples", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    geometry_parser = subparsers.add_parser("geometry")
    geometry_parser.add_argument("--output-dir", type=Path, required=True)
    geometry_parser.add_argument("--device", default="cuda:3")
    geometry_parser.add_argument("--num-samples", type=int, default=8)
    geometry_parser.add_argument(
        "--times", type=parse_times, default=parse_times("0.0625,0.1875,0.3125,0.4375")
    )
    geometry_parser.add_argument("--impulse-rms", type=float, default=1e-3)
    geometry_parser.add_argument("--seed", type=int, default=0)
    geometry_parser.add_argument("--atol", type=float, default=1e-6)
    geometry_parser.add_argument("--rtol", type=float, default=1e-3)
    geometry_parser.add_argument("--cuda-allocator-limit-gib", type=float, default=20.0)

    fid_parser = subparsers.add_parser("fid")
    fid_parser.add_argument("--output-root", type=Path, required=True)
    fid_parser.add_argument("--gpus", type=parse_gpus, default=(3,))
    fid_parser.add_argument("--modes", type=parse_modes, default=MODES)
    add_common_sampling_args(fid_parser)

    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--repo", type=Path, required=True)
    worker_parser.add_argument("--data", type=Path, required=True)
    worker_parser.add_argument("--adm-python", type=Path, required=True)
    worker_parser.add_argument("--output-dir", type=Path, required=True)
    worker_parser.add_argument("--mode", choices=ALL_MODES, required=True)
    add_common_sampling_args(worker_parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "geometry":
        geometry(args)
    elif args.command == "fid":
        fid(args)
    elif args.command == "worker":
        fid_worker(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
