#!/usr/bin/env python3
"""Collect paired trajectories for the SiT terminal-distribution control audit."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torchdiffeq import odeint
from torchvision.utils import save_image

try:
    from experiments.imagenet100_sit_terminal_distribution import (
        DEFAULT_AUDIT_CONDITIONS,
        AuditCondition,
        closed_terms,
        factorized_terms,
        parse_condition,
        sample_mean_product,
        sample_mean_square,
        validate_conditions,
    )
    from experiments.run_imagenet100_sit_nominal_transfer_geometry import _load_pair
    from experiments.sample_imagenet100_sit_fid import (
        configure_cuda_allocator,
        decode_latents_in_chunks,
        official_pixel_quantization,
        official_rank_seed,
    )
    from experiments.train_imagenet100_sit_flow import (
        DEFAULT_OFFICIAL_SIT_REPO,
        LATENT_SHAPE,
        NUM_CLASSES,
        SD_VAE_SCALING_FACTOR,
        atomic_json_dump,
    )
except ModuleNotFoundError:
    from imagenet100_sit_terminal_distribution import (
        DEFAULT_AUDIT_CONDITIONS,
        AuditCondition,
        closed_terms,
        factorized_terms,
        parse_condition,
        sample_mean_product,
        sample_mean_square,
        validate_conditions,
    )
    from run_imagenet100_sit_nominal_transfer_geometry import _load_pair
    from sample_imagenet100_sit_fid import (
        configure_cuda_allocator,
        decode_latents_in_chunks,
        official_pixel_quantization,
        official_rank_seed,
    )
    from train_imagenet100_sit_flow import (
        DEFAULT_OFFICIAL_SIT_REPO,
        LATENT_SHAPE,
        NUM_CLASSES,
        SD_VAE_SCALING_FACTOR,
        atomic_json_dump,
    )


BASE = Path("/home/zhoushunyu/data/eqvae/imagenet_sit_flow")
DEFAULT_ANCHOR = BASE / "runs/sit-s-2_seed0/checkpoints/step_00800000.pt"
DEFAULT_OTHER = BASE / "runs/sit-s-2_seed0/checkpoints/step_00500000.pt"
DEFAULT_OUTPUT = BASE / "terminal_distribution_audit_800k_v1/seed0"
DEFAULT_TIMES = (
    0.0,
    0.02,
    0.05,
    0.1,
    0.15,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.85,
    0.9,
    0.93,
    0.95,
    0.97,
    0.98,
    0.99,
    1.0,
)
DIAGNOSTIC_METRICS = (
    "state_shift_mse",
    "energy_per_dimension",
    "nominal_gap_mse",
    "state_response_mse",
    "forcing_mse",
    "response_control_mse",
    "forcing_response_cross",
    "control_mse",
    "drift_mse",
    "baseline_drift_mse",
    "response_work",
    "forcing_work",
    "incremental_rate_q",
)


def _parse_times(value: str) -> tuple[float, ...]:
    times = tuple(float(item) for item in value.split(",") if item.strip())
    if len(times) < 2 or not all(np.isfinite(times)):
        raise argparse.ArgumentTypeError("times must contain at least two finite values")
    if times[0] != 0.0 or times[-1] != 1.0 or any(
        right <= left for left, right in zip(times, times[1:])
    ):
        raise argparse.ArgumentTypeError("times must increase strictly from 0 to 1")
    return times


def _sha256_update(digest: "hashlib._Hash", value: torch.Tensor) -> None:
    digest.update(value.detach().cpu().contiguous().numpy().tobytes())


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _summarize(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return {name: float("nan") for name in ("mean", "std", "q05", "median", "q95")}
    return {
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
        "q05": float(np.quantile(finite, 0.05)),
        "median": float(np.quantile(finite, 0.5)),
        "q95": float(np.quantile(finite, 0.95)),
    }


def _condition_terms(
    condition: AuditCondition,
    *,
    anchor_baseline: torch.Tensor,
    anchor_current: torch.Tensor,
    other_baseline: torch.Tensor,
    other_current: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    if condition.mode == "closed":
        if other_current is None:
            raise ValueError("closed conditions require the weak field at the current state")
        return closed_terms(
            anchor_current,
            other_current,
            gamma=condition.gamma,
        )
    return factorized_terms(
        anchor_baseline,
        anchor_current,
        other_baseline,
        gamma=condition.gamma,
        response_scale=condition.response_scale,
    )


def _evaluate_fields(
    fields,
    states: torch.Tensor,
    time_value: torch.Tensor,
    labels: torch.Tensor,
    conditions: tuple[AuditCondition, ...],
) -> tuple[torch.Tensor, torch.Tensor, dict[int, torch.Tensor]]:
    branch_count, batch_size = states.shape[:2]
    if branch_count != 1 + len(conditions):
        raise ValueError("trajectory branch count does not match conditions")
    flat_states = states.reshape(branch_count * batch_size, *states.shape[2:])
    anchor = fields.anchor(flat_states, time_value, labels.repeat(branch_count))
    anchor = anchor.reshape(branch_count, batch_size, *states.shape[2:])
    closed_indices = [
        index for index, condition in enumerate(conditions, start=1)
        if condition.mode == "closed"
    ]
    weak_states = torch.cat([states[0], *(states[index] for index in closed_indices)])
    weak = fields.other(
        weak_states,
        time_value,
        labels.repeat(1 + len(closed_indices)),
    ).reshape(1 + len(closed_indices), batch_size, *states.shape[2:])
    weak_current = {
        condition_index: weak[weak_index]
        for weak_index, condition_index in enumerate(closed_indices, start=1)
    }
    return anchor, weak[0], weak_current


def _integrate_conditions(
    fields,
    noise: torch.Tensor,
    labels: torch.Tensor,
    conditions: tuple[AuditCondition, ...],
    output_times: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> tuple[torch.Tensor, int]:
    batch_size = len(noise)
    branch_count = 1 + len(conditions)
    initial = noise.repeat(branch_count, 1, 1, 1)
    nfe = 0

    def derivative(time_value: torch.Tensor, combined: torch.Tensor) -> torch.Tensor:
        nonlocal nfe
        nfe += 1
        states = combined.reshape(branch_count, batch_size, *noise.shape[1:])
        anchor, other_baseline, weak_current = _evaluate_fields(
            fields,
            states,
            time_value,
            labels,
            conditions,
        )
        derivatives = [anchor[0]]
        for condition_index, condition in enumerate(conditions, start=1):
            terms = _condition_terms(
                condition,
                anchor_baseline=anchor[0],
                anchor_current=anchor[condition_index],
                other_baseline=other_baseline,
                other_current=weak_current.get(condition_index),
            )
            derivatives.append(terms["drift"])
        return torch.cat(derivatives)

    trajectory = odeint(
        derivative,
        initial.float(),
        output_times,
        method="dopri5",
        atol=float(atol),
        rtol=float(rtol),
    )
    return trajectory.reshape(
        len(output_times),
        branch_count,
        batch_size,
        *noise.shape[1:],
    ), nfe


def _integrate_partitioned_conditions(
    fields,
    noise: torch.Tensor,
    labels: torch.Tensor,
    conditions: tuple[AuditCondition, ...],
    output_times: torch.Tensor,
    *,
    atol: float,
    rtol: float,
    closed_atol: float,
    closed_rtol: float,
) -> tuple[torch.Tensor, int]:
    """Share one baseline across factorized controls and isolate closed controls.

    A high-gamma closed loop can be substantially stiffer than the factorized
    controls. Giving it an independent adaptive solver prevents that branch from
    changing the accepted steps, and therefore the numerical solution, of every
    other condition in the audit.
    """

    factorized = tuple(
        (index, condition)
        for index, condition in enumerate(conditions)
        if condition.mode == "factorized"
    )
    closed = tuple(
        (index, condition)
        for index, condition in enumerate(conditions)
        if condition.mode == "closed"
    )
    trajectories: dict[int, torch.Tensor] = {}
    baseline: torch.Tensor | None = None
    total_nfe = 0

    if factorized:
        grouped, nfe = _integrate_conditions(
            fields,
            noise,
            labels,
            tuple(condition for _, condition in factorized),
            output_times,
            atol=atol,
            rtol=rtol,
        )
        total_nfe += nfe
        baseline = grouped[:, 0]
        for local_index, (condition_index, _) in enumerate(factorized, start=1):
            trajectories[condition_index] = grouped[:, local_index]

    for condition_index, condition in closed:
        isolated, nfe = _integrate_conditions(
            fields,
            noise,
            labels,
            (condition,),
            output_times,
            atol=closed_atol,
            rtol=closed_rtol,
        )
        total_nfe += nfe
        if baseline is None:
            baseline = isolated[:, 0]
        trajectories[condition_index] = isolated[:, 1]

    if baseline is None or len(trajectories) != len(conditions):
        raise AssertionError("partitioned integration did not produce every branch")
    ordered = [baseline, *(trajectories[index] for index in range(len(conditions)))]
    return torch.stack(ordered, dim=1), total_nfe


def _diagnose_snapshot(
    fields,
    states: torch.Tensor,
    time_value: torch.Tensor,
    labels: torch.Tensor,
    conditions: tuple[AuditCondition, ...],
) -> dict[str, torch.Tensor]:
    anchor, other_baseline, weak_current = _evaluate_fields(
        fields,
        states,
        time_value,
        labels,
        conditions,
    )
    baseline = states[0]
    baseline_drift_mse = sample_mean_square(anchor[0])
    output = {
        name: torch.empty(
            len(conditions),
            len(labels),
            device=states.device,
            dtype=torch.float32,
        )
        for name in DIAGNOSTIC_METRICS
    }
    tiny = torch.finfo(torch.float32).tiny
    for condition_index, condition in enumerate(conditions, start=1):
        current = states[condition_index]
        delta = current - baseline
        state_response = anchor[condition_index] - anchor[0]
        terms = _condition_terms(
            condition,
            anchor_baseline=anchor[0],
            anchor_current=anchor[condition_index],
            other_baseline=other_baseline,
            other_current=weak_current.get(condition_index),
        )
        direct_error = sample_mean_square(terms["drift"] - terms["direct_drift"])
        if direct_error.max().item() > 1e-10:
            raise AssertionError("factorized drift decompositions are not equivalent")
        delta_mse = sample_mean_square(delta)
        output["state_shift_mse"][condition_index - 1] = delta_mse
        output["energy_per_dimension"][condition_index - 1] = 0.5 * delta_mse
        output["nominal_gap_mse"][condition_index - 1] = sample_mean_square(
            terms["nominal_gap"]
        )
        output["state_response_mse"][condition_index - 1] = sample_mean_square(
            state_response
        )
        output["forcing_mse"][condition_index - 1] = sample_mean_square(
            terms["forcing"]
        )
        output["response_control_mse"][condition_index - 1] = sample_mean_square(
            terms["response_control"]
        )
        output["forcing_response_cross"][condition_index - 1] = sample_mean_product(
            terms["forcing"],
            terms["response_control"],
        )
        output["control_mse"][condition_index - 1] = sample_mean_square(
            terms["control"]
        )
        output["drift_mse"][condition_index - 1] = sample_mean_square(terms["drift"])
        output["baseline_drift_mse"][condition_index - 1] = baseline_drift_mse
        output["response_work"][condition_index - 1] = sample_mean_product(
            delta,
            state_response,
        )
        output["forcing_work"][condition_index - 1] = sample_mean_product(
            delta,
            terms["forcing"],
        )
        output["incremental_rate_q"][condition_index - 1] = (
            sample_mean_product(delta, state_response) / delta_mse.clamp_min(tiny)
        )
        output["incremental_rate_q"][condition_index - 1] = torch.where(
            delta_mse > 1e-20,
            output["incremental_rate_q"][condition_index - 1],
            torch.full_like(delta_mse, float("nan")),
        )
    return output


def _diagnostic_summary_rows(
    diagnostics: dict[str, np.ndarray],
    conditions: tuple[AuditCondition, ...],
    times: tuple[float, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for condition_index, condition in enumerate(conditions):
        for time_index, time_value in enumerate(times):
            row: dict[str, object] = {
                "condition": condition.name,
                "mode": condition.mode,
                "gamma": condition.gamma,
                "response_scale": condition.response_scale,
                "time": time_value,
            }
            for metric, values in diagnostics.items():
                for statistic, value in _summarize(
                    values[condition_index, time_index]
                ).items():
                    row[f"{metric}_{statistic}"] = value
            rows.append(row)
    return rows


def _integrated_summary_rows(
    diagnostics: dict[str, np.ndarray],
    conditions: tuple[AuditCondition, ...],
    times: tuple[float, ...],
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    time_array = np.asarray(times, dtype=np.float64)
    integrated = {
        "control_action": 0.5 * np.trapezoid(
            diagnostics["control_mse"], time_array, axis=1
        ),
        "forcing_action": 0.5 * np.trapezoid(
            diagnostics["forcing_mse"], time_array, axis=1
        ),
        "response_control_action": 0.5 * np.trapezoid(
            diagnostics["response_control_mse"], time_array, axis=1
        ),
        "forcing_response_cross_action": np.trapezoid(
            diagnostics["forcing_response_cross"], time_array, axis=1
        ),
        "drift_energy": 0.5 * np.trapezoid(
            diagnostics["drift_mse"], time_array, axis=1
        ),
    }
    rows: list[dict[str, object]] = []
    for condition_index, condition in enumerate(conditions):
        row: dict[str, object] = {
            "condition": condition.name,
            "mode": condition.mode,
            "gamma": condition.gamma,
            "response_scale": condition.response_scale,
        }
        for metric, values in integrated.items():
            for statistic, value in _summarize(values[condition_index]).items():
                row[f"{metric}_{statistic}"] = value
        rows.append(row)
    return rows, integrated


def _decode_endpoints(
    endpoint_path: Path,
    branch_names: tuple[str, ...],
    *,
    device: torch.device,
    batch_size: int,
    output_dir: Path,
) -> dict[str, str]:
    from diffusers.models import AutoencoderKL

    endpoints = np.load(endpoint_path, mmap_mode="r")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        local_files_only=True,
    )
    vae.to(device).eval().requires_grad_(False)
    sample_paths: dict[str, str] = {}
    for branch_index, name in enumerate(branch_names):
        images = np.empty((endpoints.shape[1], 256, 256, 3), dtype=np.uint8)
        preview = None
        for start in range(0, endpoints.shape[1], batch_size):
            stop = min(start + batch_size, endpoints.shape[1])
            latent = torch.from_numpy(
                np.array(endpoints[branch_index, start:stop], copy=True)
            ).to(device)
            decoded = decode_latents_in_chunks(
                vae,
                latent,
                scaling_factor=SD_VAE_SCALING_FACTOR,
                chunk_size=batch_size,
            )
            images[start:stop] = official_pixel_quantization(decoded)
            if preview is None:
                preview = decoded.detach().cpu()
        sample_path = output_dir / f"samples_{name}_n{endpoints.shape[1]}.npz"
        np.savez(sample_path, images)
        sample_paths[name] = str(sample_path)
        if preview is not None:
            save_image(
                (preview[: min(16, len(preview))] + 1.0) / 2.0,
                output_dir / f"preview_{name}.png",
                nrow=4,
            )
        print(f"[decode] {name}: {sample_path}", flush=True)
    return sample_paths


def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.num_samples <= 0 or args.batch_size <= 0:
        raise ValueError("num-samples and batch-size must be positive")
    if args.batch_pause_seconds < 0:
        raise ValueError("batch-pause-seconds cannot be negative")
    conditions = tuple(args.condition or DEFAULT_AUDIT_CONDITIONS)
    validate_conditions(conditions)
    times = tuple(args.times)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    allocator = configure_cuda_allocator(device, limit_gib=args.cuda_allocator_limit_gib)
    rank_seed = official_rank_seed(args.global_seed, 1, 0)
    torch.manual_seed(rank_seed)
    torch.cuda.manual_seed(rank_seed)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
    torch.set_float32_matmul_precision("high" if args.allow_tf32 else "highest")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fields, metadata = _load_pair(args, device)
    output_times = torch.tensor(times, device=device, dtype=torch.float32)
    branch_names = ("baseline", *(condition.name for condition in conditions))
    branch_count = len(branch_names)
    trajectory_shape = (
        branch_count,
        len(times),
        args.num_samples,
        *LATENT_SHAPE,
    )
    trajectories = np.lib.format.open_memmap(
        output_dir / "trajectory_snapshots_fp16.npy",
        mode="w+",
        dtype=np.float16,
        shape=trajectory_shape,
    )
    endpoint_latents = np.lib.format.open_memmap(
        output_dir / "endpoint_latents_fp32.npy",
        mode="w+",
        dtype=np.float32,
        shape=(branch_count, args.num_samples, *LATENT_SHAPE),
    )
    labels_array = np.empty(args.num_samples, dtype=np.int16)
    diagnostics = {
        name: np.empty(
            (len(conditions), len(times), args.num_samples),
            dtype=np.float32,
        )
        for name in DIAGNOSTIC_METRICS
    }
    noise_digest = hashlib.sha256()
    label_digest = hashlib.sha256()
    total_nfe = 0
    equivalence_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, args.num_samples, args.batch_size):
            stop = min(start + args.batch_size, args.num_samples)
            current_batch_size = stop - start
            noise = torch.randn(current_batch_size, *LATENT_SHAPE, device=device)
            labels = torch.randint(0, NUM_CLASSES, (current_batch_size,), device=device)
            _sha256_update(noise_digest, noise)
            _sha256_update(label_digest, labels)
            labels_array[start:stop] = labels.cpu().numpy().astype(np.int16, copy=False)
            trajectory, nfe = _integrate_partitioned_conditions(
                fields,
                noise,
                labels,
                conditions,
                output_times,
                atol=args.atol,
                rtol=args.rtol,
                closed_atol=args.closed_atol,
                closed_rtol=args.closed_rtol,
            )
            total_nfe += nfe
            if not torch.isfinite(trajectory).all():
                raise FloatingPointError("non-finite terminal-distribution trajectory")
            if start == 0 and args.verify_first_batch_individual:
                for condition_index, condition in enumerate(conditions, start=1):
                    if condition.mode == "closed":
                        continue
                    individual, _ = _integrate_conditions(
                        fields,
                        noise,
                        labels,
                        (condition,),
                        output_times,
                        atol=args.atol,
                        rtol=args.rtol,
                    )
                    for grouped_branch, individual_branch, branch_role in (
                        (0, 0, "baseline"),
                        (condition_index, 1, "condition"),
                    ):
                        difference = (
                            trajectory[:, grouped_branch] - individual[:, individual_branch]
                        ).float()
                        row = {
                            "condition": condition.name,
                            "branch_role": branch_role,
                            "trajectory_rms_difference": float(
                                torch.sqrt(difference.square().mean()).item()
                            ),
                            "endpoint_rms_difference": float(
                                torch.sqrt(difference[-1].square().mean()).item()
                            ),
                            "endpoint_max_abs_difference": float(
                                difference[-1].abs().max().item()
                            ),
                        }
                        equivalence_rows.append(row)
                if equivalence_rows:
                    maximum_rms = max(
                        float(row["endpoint_rms_difference"])
                        for row in equivalence_rows
                    )
                    if maximum_rms > args.equivalence_rms_tolerance:
                        raise AssertionError(
                            "grouped and individual real-model integrations differ by "
                            f"RMS={maximum_rms:.6g}, above "
                            f"{args.equivalence_rms_tolerance:.6g}"
                        )
            cpu_trajectory = trajectory.detach().cpu().numpy()
            trajectories[:, :, start:stop] = np.transpose(cpu_trajectory, (1, 0, 2, 3, 4, 5))
            endpoint_latents[:, start:stop] = cpu_trajectory[-1]
            for time_index, time_value in enumerate(output_times):
                snapshot = _diagnose_snapshot(
                    fields,
                    trajectory[time_index],
                    time_value,
                    labels,
                    conditions,
                )
                for metric, values in snapshot.items():
                    diagnostics[metric][:, time_index, start:stop] = (
                        values.detach().cpu().numpy()
                    )
            print(
                f"[{stop:04d}/{args.num_samples}] nfe={nfe} "
                f"elapsed={time.perf_counter()-started:.1f}s",
                flush=True,
            )
            if stop < args.num_samples and args.batch_pause_seconds > 0:
                time.sleep(args.batch_pause_seconds)

    trajectories.flush()
    endpoint_latents.flush()
    np.save(output_dir / "labels.npy", labels_array)
    if equivalence_rows:
        _write_csv(equivalence_rows, output_dir / "grouped_individual_equivalence.csv")
    np.savez_compressed(output_dir / "trajectory_diagnostics.npz", **diagnostics)
    _write_csv(
        _diagnostic_summary_rows(diagnostics, conditions, times),
        output_dir / "diagnostics_by_time.csv",
    )
    integrated_rows, integrated = _integrated_summary_rows(
        diagnostics,
        conditions,
        times,
    )
    np.savez_compressed(output_dir / "integrated_actions.npz", **integrated)
    _write_csv(integrated_rows, output_dir / "integrated_action_summary.csv")
    field_counts = {
        "anchor_forwards": fields.anchor_forwards,
        "other_forwards": fields.other_forwards,
        "anchor_examples": fields.anchor_examples,
        "other_examples": fields.other_examples,
    }
    del trajectories, endpoint_latents, fields
    gc.collect()
    torch.cuda.empty_cache()
    sample_paths: dict[str, str] = {}
    if not args.skip_decode:
        sample_paths = _decode_endpoints(
            output_dir / "endpoint_latents_fp32.npy",
            branch_names,
            device=device,
            batch_size=args.vae_decode_batch_size,
            output_dir=output_dir,
        )
    manifest = {
        "format": "eqvae_imagenet100_sit_terminal_distribution_audit_v1",
        "baseline_formula": "b'=S(b,t)",
        "conditions": [condition.as_dict() for condition in conditions],
        "branches": list(branch_names),
        "anchor_checkpoint": str(args.anchor_checkpoint.expanduser().resolve()),
        "other_checkpoint": str(args.other_checkpoint.expanduser().resolve()),
        "allow_step_mismatch": bool(args.allow_step_mismatch),
        "weights": args.weights,
        "global_seed": int(args.global_seed),
        "rank_seed": int(rank_seed),
        "num_samples": int(args.num_samples),
        "batch_size": int(args.batch_size),
        "vae_decode_batch_size": int(args.vae_decode_batch_size),
        "times": list(times),
        "sampler": {
            "method": "dopri5",
            "atol": float(args.atol),
            "rtol": float(args.rtol),
            "closed_atol": float(args.closed_atol),
            "closed_rtol": float(args.closed_rtol),
            "shared_adaptive_steps_across_factorized_branches": True,
            "closed_conditions_integrated_individually": True,
        },
        "noise_sha256": noise_digest.hexdigest(),
        "label_sha256": label_digest.hexdigest(),
        "label_histogram": np.bincount(labels_array, minlength=NUM_CLASSES).tolist(),
        "trajectory_snapshots": str(output_dir / "trajectory_snapshots_fp16.npy"),
        "trajectory_shape": list(trajectory_shape),
        "endpoint_latents": str(output_dir / "endpoint_latents_fp32.npy"),
        "samples": sample_paths,
        "total_nfe": int(total_nfe),
        "elapsed_seconds": time.perf_counter() - started,
        "batch_pause_seconds": float(args.batch_pause_seconds),
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "field_counts": field_counts,
        "grouped_individual_equivalence": equivalence_rows,
        "metadata": metadata,
        **allocator,
    }
    atomic_json_dump(manifest, output_dir / "manifest.json")
    (output_dir / "SAMPLING_COMPLETE").touch()
    print(json.dumps(manifest, indent=2, default=str), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-checkpoint", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--other-checkpoint", type=Path, default=DEFAULT_OTHER)
    parser.add_argument(
        "--allow-step-mismatch",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument(
        "--condition",
        type=parse_condition,
        action="append",
        default=None,
        help="Repeat name:mode:gamma:response_scale; defaults to the preregistered set.",
    )
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vae-decode-batch-size", type=int, default=2)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--batch-pause-seconds", type=float, default=0.0)
    parser.add_argument(
        "--times",
        type=_parse_times,
        default=DEFAULT_TIMES,
        help="Strictly increasing comma-separated output times from 0 to 1.",
    )
    parser.add_argument("--atol", type=float, default=1e-7)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--closed-atol", type=float, default=1e-8)
    parser.add_argument("--closed-rtol", type=float, default=1e-5)
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-decode", action="store_true")
    parser.add_argument("--verify-first-batch-individual", action="store_true")
    parser.add_argument("--equivalence-rms-tolerance", type=float, default=5e-3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cuda-allocator-limit-gib", type=float, default=8.0)
    parser.add_argument("--official-sit-repo", type=Path, default=DEFAULT_OFFICIAL_SIT_REPO)
    parser.add_argument(
        "--verify-sit-source",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
