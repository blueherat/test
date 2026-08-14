#!/usr/bin/env python3
"""Run paired finite-strength guidance diagnostics on SiT 400K fields."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torchdiffeq import odeint

try:
    from experiments.finite_guidance_dynamics import (
        central_difference_metrics,
        integrate_baseline_tangent,
        integrate_frozen_closed_sweep,
        integrate_guidance_sweep,
        linearity_metrics,
        sample_cosine,
        sample_rms,
    )
    from experiments.sample_imagenet100_sit_static_pair_fid import (
        _load_field_model,
        validate_pair_compatibility,
    )
    from experiments.imagenet100_sit_static_pair import output_to_field_velocity
    from experiments.train_imagenet100_sit_flow import (
        DEFAULT_OFFICIAL_SIT_REPO,
        LATENT_SHAPE,
        NUM_CLASSES,
        atomic_json_dump,
        load_official_sit_module,
    )
except ModuleNotFoundError:
    from finite_guidance_dynamics import (
        central_difference_metrics,
        integrate_baseline_tangent,
        integrate_frozen_closed_sweep,
        integrate_guidance_sweep,
        linearity_metrics,
        sample_cosine,
        sample_rms,
    )
    from sample_imagenet100_sit_static_pair_fid import (
        _load_field_model,
        validate_pair_compatibility,
    )
    from imagenet100_sit_static_pair import output_to_field_velocity
    from train_imagenet100_sit_flow import (
        DEFAULT_OFFICIAL_SIT_REPO,
        LATENT_SHAPE,
        NUM_CLASSES,
        atomic_json_dump,
        load_official_sit_module,
    )


DEFAULT_ANCHOR = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/"
    "sit-s-2_seed0/checkpoints/step_00400000.pt"
)
DEFAULT_X400 = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/"
    "sit-s-2_x-velocity-loss-floor0p05_seed0/checkpoints/step_00400000.pt"
)
DEFAULT_V270 = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/"
    "sit-s-2_seed0/checkpoints/step_00270000.pt"
)
DEFAULT_X800 = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/"
    "sit-s-2_x-velocity-loss-floor0p05_seed0/checkpoints/step_00800000.pt"
)
DEFAULT_V500 = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/"
    "sit-s-2_seed0/checkpoints/step_00500000.pt"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/"
    "finite_guidance_400k_mechanism"
)


def _parse_floats(value: str) -> list[float]:
    parsed = [float(item) for item in value.split(",") if item.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one comma-separated float")
    if not all(math.isfinite(item) for item in parsed):
        raise argparse.ArgumentTypeError("all values must be finite")
    return parsed


def _parse_ints(value: str) -> list[int]:
    parsed = [int(item) for item in value.split(",") if item.strip()]
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return parsed


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _input_bank(num_samples: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn(num_samples, *LATENT_SHAPE, generator=generator)
    labels = torch.randint(NUM_CLASSES, (num_samples,), generator=generator)
    return noise, labels


def _summary(values: torch.Tensor) -> dict[str, float]:
    array = values.detach().float().cpu().numpy().astype(np.float64, copy=False)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "q05": float(np.quantile(array, 0.05)),
        "median": float(np.quantile(array, 0.50)),
        "q95": float(np.quantile(array, 0.95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _gamma_index(values: list[float], target: float, *, atol: float = 1e-6) -> int:
    distances = [abs(value - float(target)) for value in values]
    index = int(np.argmin(distances))
    if distances[index] > atol:
        raise ValueError(f"gamma {target} is missing from {values}")
    return index


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class PairedFields:
    """Evaluate one anchor and one weak field with condition-folded labels."""

    def __init__(
        self,
        anchor_model: torch.nn.Module,
        other_model: torch.nn.Module,
        anchor_semantics,
        other_semantics,
        labels: torch.Tensor,
        *,
        math_attention: bool,
    ) -> None:
        self.anchor_model = anchor_model
        self.other_model = other_model
        self.anchor_semantics = anchor_semantics
        self.other_semantics = other_semantics
        self.labels = labels
        self.math_attention = bool(math_attention)
        self.anchor_forwards = 0
        self.other_forwards = 0

    def _labels_for(self, state: torch.Tensor) -> torch.Tensor:
        if len(state) % len(self.labels) != 0:
            raise ValueError("condition-folded state batch is not divisible by base labels")
        return self.labels.repeat(len(state) // len(self.labels))

    @staticmethod
    def _times_for(time_value: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return time_value.expand(len(state))

    def anchor(self, time_value: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        self.anchor_forwards += 1
        times = self._times_for(time_value, state)
        context = sdpa_kernel(SDPBackend.MATH) if self.math_attention else nullcontext()
        with context:
            output = self.anchor_model(state, times, self._labels_for(state))
        return output_to_field_velocity(
            output,
            state=state,
            time_value=times,
            semantics=self.anchor_semantics,
        )

    def direction(
        self,
        time_value: torch.Tensor,
        state: torch.Tensor,
        anchor_velocity: torch.Tensor,
    ) -> torch.Tensor:
        self.other_forwards += 1
        times = self._times_for(time_value, state)
        context = sdpa_kernel(SDPBackend.MATH) if self.math_attention else nullcontext()
        with context:
            output = self.other_model(state, times, self._labels_for(state))
        other_velocity = output_to_field_velocity(
            output,
            state=state,
            time_value=times,
            semantics=self.other_semantics,
        )
        return anchor_velocity - other_velocity


def _load_pair(args: argparse.Namespace, labels: torch.Tensor, device: torch.device):
    sit_module, source_metadata = load_official_sit_module(
        args.official_sit_repo.expanduser().resolve(),
        verify_source=args.verify_sit_source,
    )
    anchor_model, anchor_semantics, anchor_meta, anchor_checkpoint = _load_field_model(
        checkpoint_path=args.anchor_checkpoint.expanduser().resolve(),
        requested_field="auto",
        weights=args.weights,
        sit_module=sit_module,
        source_metadata=source_metadata,
        device=device,
    )
    checkpoint_by_direction = {
        "x400": args.x400_checkpoint,
        "v270": args.v270_checkpoint,
        "x800": args.x800_checkpoint,
        "v500": args.v500_checkpoint,
    }
    other_checkpoint_path = checkpoint_by_direction[args.direction]
    other_model, other_semantics, other_meta, other_checkpoint = _load_field_model(
        checkpoint_path=other_checkpoint_path.expanduser().resolve(),
        requested_field="auto",
        weights=args.weights,
        sit_module=sit_module,
        source_metadata=source_metadata,
        device=device,
    )
    assert anchor_model is not None and other_model is not None
    validate_pair_compatibility(
        anchor_checkpoint,
        other_checkpoint,
        anchor_meta,
        other_meta,
        allow_step_mismatch=args.direction in {"v270", "v500"},
    )
    if anchor_semantics.prediction_target != "velocity":
        raise ValueError("the anchor checkpoint must provide a native velocity field")
    expected_other = "x" if args.direction.startswith("x") else "velocity"
    if other_semantics.prediction_target != expected_other:
        raise ValueError(
            f"{args.direction} must provide {expected_other!r}, got "
            f"{other_semantics.prediction_target!r}"
        )
    fields = PairedFields(
        anchor_model,
        other_model,
        anchor_semantics,
        other_semantics,
        labels,
        math_attention=args.math_attention,
    )
    metadata = {
        "official_sit": source_metadata,
        "anchor": anchor_meta,
        "other": other_meta,
        "direction": args.direction,
        "direction_formula": "u(z,t) = anchor(z,t) - weak(z,t)",
        "score_gap_formula": "Delta score = t / (1 - t) * Delta velocity",
    }
    return fields, metadata


def _manifest(args: argparse.Namespace, metadata: dict, noise: torch.Tensor, labels: torch.Tensor) -> dict:
    return {
        "format": (
            "eqvae_sit_finite_guidance_mechanism_v2"
            if args.study == "tangent_frozen" or args.direction in {"x800", "v500"}
            else "eqvae_sit400_finite_guidance_mechanism_v1"
        ),
        "study": args.study,
        "direction": args.direction,
        "num_samples": int(args.num_samples),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "noise_sha256": _tensor_sha256(noise),
        "labels_sha256": _tensor_sha256(labels),
        "weights": args.weights,
        "precision": "fp32",
        "allow_tf32": bool(args.allow_tf32),
        "math_attention": bool(args.math_attention),
        "heun_steps": int(args.heun_steps),
        "gammas": [float(value) for value in args.gammas],
        "central_delta": float(args.central_delta),
        "solver_step_counts": [int(value) for value in args.solver_step_counts],
        "adaptive_atol": float(args.adaptive_atol),
        "adaptive_rtol": float(args.adaptive_rtol),
        "integration_interval": [0.0, 1.0],
        "linearity_thresholds": {"cosine": 0.95, "relative_residual": 0.20},
        "central_difference_thresholds": {"cosine": 0.99, "relative_residual": 0.05},
        "feedback_retention_threshold": 0.70,
        **metadata,
    }


def _shard_path(output_dir: Path, batch_index: int) -> Path:
    return output_dir / "trajectory_shards" / f"batch_{batch_index:05d}.pt"


def _run_linearity_batch(
    fields: PairedFields,
    noise: torch.Tensor,
    time_grid: torch.Tensor,
    gammas: list[float],
    central_delta: float,
) -> dict[str, object]:
    all_gammas = sorted(set([-central_delta, 0.0, central_delta, *gammas]))
    gamma_tensor = torch.tensor(all_gammas, device=noise.device, dtype=torch.float32)
    baseline, tangent = integrate_baseline_tangent(
        fields.anchor,
        fields.direction,
        noise,
        time_grid,
    )
    with torch.no_grad():
        endpoints = integrate_guidance_sweep(
            fields.anchor,
            fields.direction,
            noise,
            time_grid,
            gamma_tensor,
        )
    return {
        "gammas": torch.tensor(all_gammas),
        "baseline": baseline.detach().cpu(),
        "tangent": tangent.detach().cpu(),
        "endpoints": endpoints.detach().cpu(),
    }


def _run_feedback_batch(
    fields: PairedFields,
    noise: torch.Tensor,
    time_grid: torch.Tensor,
    gammas: list[float],
) -> dict[str, object]:
    all_gammas = sorted(set([0.0, *gammas]))
    gamma_tensor = torch.tensor(all_gammas, device=noise.device, dtype=torch.float32)
    with torch.no_grad():
        baseline, frozen, closed = integrate_frozen_closed_sweep(
            fields.anchor,
            fields.direction,
            noise,
            time_grid,
            gamma_tensor,
        )
    return {
        "gammas": torch.tensor(all_gammas),
        "baseline": baseline.detach().cpu(),
        "frozen": frozen.detach().cpu(),
        "closed": closed.detach().cpu(),
    }


def _run_tangent_frozen_batch(
    fields: PairedFields,
    noise: torch.Tensor,
    time_grid: torch.Tensor,
    gammas: list[float],
    central_delta: float,
) -> dict[str, object]:
    """Pair the gamma-zero tangent with exact frozen and closed endpoints."""

    all_gammas = sorted(set([-central_delta, 0.0, central_delta, *gammas]))
    gamma_tensor = torch.tensor(all_gammas, device=noise.device, dtype=torch.float32)
    baseline, tangent = integrate_baseline_tangent(
        fields.anchor,
        fields.direction,
        noise,
        time_grid,
    )
    with torch.no_grad():
        feedback_baseline, frozen, closed = integrate_frozen_closed_sweep(
            fields.anchor,
            fields.direction,
            noise,
            time_grid,
            gamma_tensor,
        )
    return {
        "gammas": torch.tensor(all_gammas),
        "baseline": baseline.detach().cpu(),
        "feedback_baseline": feedback_baseline.detach().cpu(),
        "tangent": tangent.detach().cpu(),
        "frozen": frozen.detach().cpu(),
        "closed": closed.detach().cpu(),
    }


def _adaptive_sweep(
    fields: PairedFields,
    noise: torch.Tensor,
    gammas: list[float],
    *,
    atol: float,
    rtol: float,
) -> torch.Tensor:
    gamma_tensor = torch.tensor(gammas, device=noise.device, dtype=torch.float32)
    condition_count = len(gamma_tensor)
    batch_size = len(noise)
    state = noise.unsqueeze(0).expand(condition_count, *noise.shape).reshape(
        condition_count * batch_size, *noise.shape[1:]
    )
    scales = gamma_tensor.repeat_interleave(batch_size).reshape(
        -1, *([1] * (state.ndim - 1))
    )

    def guided(time_value: torch.Tensor, current_state: torch.Tensor) -> torch.Tensor:
        anchor = fields.anchor(time_value, current_state)
        direction = fields.direction(time_value, current_state, anchor)
        return anchor + scales * direction

    endpoints = odeint(
        guided,
        state.float(),
        torch.tensor([0.0, 1.0], device=state.device),
        method="dopri5",
        atol=float(atol),
        rtol=float(rtol),
    )[-1]
    return endpoints.reshape(condition_count, batch_size, *noise.shape[1:])


def _run_solver_batch(
    fields: PairedFields,
    noise: torch.Tensor,
    gammas: list[float],
    solver_step_counts: list[int],
    *,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    all_gammas = sorted(set([0.0, *gammas]))
    with torch.no_grad():
        reference = _adaptive_sweep(
            fields,
            noise,
            all_gammas,
            atol=atol,
            rtol=rtol,
        )
        fixed: dict[int, torch.Tensor] = {}
        for step_count in solver_step_counts:
            time_grid = torch.linspace(
                0.0,
                1.0,
                step_count + 1,
                device=noise.device,
            )
            fixed[step_count] = integrate_guidance_sweep(
                fields.anchor,
                fields.direction,
                noise,
                time_grid,
                torch.tensor(all_gammas, device=noise.device),
            ).cpu()
    return {
        "gammas": torch.tensor(all_gammas),
        "adaptive": reference.cpu(),
        "fixed": fixed,
    }


def _aggregate_linearity(shards: list[dict[str, object]], delta: float) -> tuple[list[dict], dict]:
    gammas = shards[0]["gammas"]
    assert isinstance(gammas, torch.Tensor)
    baseline = torch.cat([item["baseline"] for item in shards])
    tangent = torch.cat([item["tangent"] for item in shards])
    endpoints = torch.cat([item["endpoints"] for item in shards], dim=1)
    gamma_values = [float(value) for value in gammas.tolist()]
    zero_index = _gamma_index(gamma_values, 0.0)
    minus_index = _gamma_index(gamma_values, -float(delta))
    plus_index = _gamma_index(gamma_values, float(delta))
    sweep_baseline = endpoints[zero_index]
    baseline_difference = sample_rms(sweep_baseline - baseline)
    central = central_difference_metrics(
        endpoints[minus_index],
        endpoints[plus_index],
        tangent,
        delta=delta,
    )
    rows: list[dict[str, object]] = []
    for gamma, endpoint in zip(gamma_values, endpoints, strict=True):
        if gamma == 0.0:
            continue
        metrics = linearity_metrics(baseline, endpoint, tangent, gamma=gamma)
        row: dict[str, object] = {"gamma": gamma}
        for name, values in metrics.items():
            for statistic, value in _summary(values).items():
                row[f"{name}_{statistic}"] = value
        rows.append(row)
    central_summary = {
        name: _summary(values) for name, values in central.items()
    }
    positive_rows = [row for row in rows if float(row["gamma"]) > 0]
    linear_gammas = [
        float(row["gamma"])
        for row in positive_rows
        if float(row["cosine_mean"]) >= 0.95
        and float(row["relative_residual_mean"]) <= 0.20
    ]
    summary = {
        "baseline_sweep_difference": _summary(baseline_difference),
        "central_difference": central_summary,
        "central_difference_pass": bool(
            central_summary["cosine"]["mean"] >= 0.99
            and central_summary["relative_residual"]["mean"] <= 0.05
        ),
        "largest_passing_positive_gamma": max(linear_gammas, default=None),
        "linearity_at_gamma_one": next(
            (row for row in positive_rows if abs(float(row["gamma"]) - 1.0) <= 1e-6),
            None,
        ),
    }
    return rows, summary


def _aggregate_feedback(shards: list[dict[str, object]]) -> tuple[list[dict], dict]:
    gammas = shards[0]["gammas"]
    assert isinstance(gammas, torch.Tensor)
    baseline = torch.cat([item["baseline"] for item in shards])
    frozen = torch.cat([item["frozen"] for item in shards], dim=1)
    closed = torch.cat([item["closed"] for item in shards], dim=1)
    rows: list[dict[str, object]] = []
    for gamma, frozen_endpoint, closed_endpoint in zip(
        [float(value) for value in gammas.tolist()],
        frozen,
        closed,
        strict=True,
    ):
        if gamma == 0.0:
            continue
        frozen_response = frozen_endpoint - baseline
        closed_response = closed_endpoint - baseline
        feedback = closed_endpoint - frozen_endpoint
        metrics = {
            "response_cosine": sample_cosine(frozen_response, closed_response),
            "frozen_over_closed_rms": sample_rms(frozen_response)
            / sample_rms(closed_response).clamp_min(torch.finfo(torch.float32).tiny),
            "feedback_over_closed_rms": sample_rms(feedback)
            / sample_rms(closed_response).clamp_min(torch.finfo(torch.float32).tiny),
            "frozen_response_rms": sample_rms(frozen_response),
            "closed_response_rms": sample_rms(closed_response),
        }
        row: dict[str, object] = {"gamma": gamma}
        for name, values in metrics.items():
            for statistic, value in _summary(values).items():
                row[f"{name}_{statistic}"] = value
        rows.append(row)
    gamma_one = next(
        (row for row in rows if abs(float(row["gamma"]) - 1.0) <= 1e-6),
        None,
    )
    summary = {
        "feedback_at_gamma_one": gamma_one,
        "gamma_one_frozen_response_retained": (
            gamma_one is not None
            and float(gamma_one["frozen_over_closed_rms_mean"]) >= 0.70
            and float(gamma_one["response_cosine_mean"]) >= 0.90
        ),
    }
    return rows, summary


def _aggregate_tangent_frozen(
    shards: list[dict[str, object]],
    delta: float,
) -> tuple[list[dict], dict]:
    """Summarize tangent accuracy for exact frozen and closed responses."""

    gammas = shards[0]["gammas"]
    assert isinstance(gammas, torch.Tensor)
    baseline = torch.cat([item["baseline"] for item in shards])
    feedback_baseline = torch.cat([item["feedback_baseline"] for item in shards])
    tangent = torch.cat([item["tangent"] for item in shards])
    frozen = torch.cat([item["frozen"] for item in shards], dim=1)
    closed = torch.cat([item["closed"] for item in shards], dim=1)
    gamma_values = [float(value) for value in gammas.tolist()]
    zero_index = _gamma_index(gamma_values, 0.0)
    minus_index = _gamma_index(gamma_values, -float(delta))
    plus_index = _gamma_index(gamma_values, float(delta))

    central = central_difference_metrics(
        frozen[minus_index],
        frozen[plus_index],
        tangent,
        delta=delta,
    )
    rows: list[dict[str, object]] = []
    for gamma, frozen_endpoint, closed_endpoint in zip(
        gamma_values,
        frozen,
        closed,
        strict=True,
    ):
        if gamma == 0.0:
            continue
        for response_name, endpoint in (
            ("frozen", frozen_endpoint),
            ("closed", closed_endpoint),
        ):
            metrics = linearity_metrics(baseline, endpoint, tangent, gamma=gamma)
            row: dict[str, object] = {
                "gamma": gamma,
                "response": response_name,
            }
            for name, values in metrics.items():
                for statistic, value in _summary(values).items():
                    row[f"{name}_{statistic}"] = value
            rows.append(row)

    central_summary = {name: _summary(values) for name, values in central.items()}
    positive_frozen = [
        row
        for row in rows
        if row["response"] == "frozen" and float(row["gamma"]) > 0
    ]
    linear_gammas = [
        float(row["gamma"])
        for row in positive_frozen
        if float(row["cosine_mean"]) >= 0.95
        and float(row["relative_residual_mean"]) <= 0.20
    ]

    def gamma_one(response: str) -> dict[str, object] | None:
        return next(
            (
                row
                for row in rows
                if row["response"] == response
                and abs(float(row["gamma"]) - 1.0) <= 1e-6
            ),
            None,
        )

    summary = {
        "baseline_pair_difference": _summary(
            sample_rms(feedback_baseline - baseline)
        ),
        "zero_gamma_frozen_difference": _summary(
            sample_rms(frozen[zero_index] - baseline)
        ),
        "zero_gamma_closed_difference": _summary(
            sample_rms(closed[zero_index] - baseline)
        ),
        "central_difference": central_summary,
        "central_difference_pass": bool(
            central_summary["cosine"]["mean"] >= 0.99
            and central_summary["relative_residual"]["mean"] <= 0.05
        ),
        "largest_passing_positive_frozen_gamma": max(linear_gammas, default=None),
        "frozen_at_gamma_one": gamma_one("frozen"),
        "closed_at_gamma_one": gamma_one("closed"),
    }
    return rows, summary


def _aggregate_solver(shards: list[dict[str, object]]) -> tuple[list[dict], dict]:
    rows: list[dict[str, object]] = []
    gammas = shards[0]["gammas"]
    assert isinstance(gammas, torch.Tensor)
    adaptive = torch.cat([item["adaptive"] for item in shards], dim=1)
    step_counts = sorted(shards[0]["fixed"])
    for step_count in step_counts:
        fixed = torch.cat([item["fixed"][step_count] for item in shards], dim=1)
        for gamma, fixed_endpoint, reference_endpoint in zip(
            [float(value) for value in gammas.tolist()],
            fixed,
            adaptive,
            strict=True,
        ):
            difference = fixed_endpoint - reference_endpoint
            relative = sample_rms(difference) / sample_rms(reference_endpoint).clamp_min(
                torch.finfo(torch.float32).tiny
            )
            row: dict[str, object] = {"heun_steps": step_count, "gamma": gamma}
            for name, values in {
                "endpoint_difference_rms": sample_rms(difference),
                "relative_endpoint_difference": relative,
            }.items():
                for statistic, value in _summary(values).items():
                    row[f"{name}_{statistic}"] = value
            rows.append(row)
    best_steps = max(step_counts)
    best_rows = [row for row in rows if int(row["heun_steps"]) == best_steps]
    return rows, {"highest_step_count": best_steps, "highest_step_rows": best_rows}


def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.batch_size <= 0 or args.num_samples <= 0:
        raise ValueError("batch-size and num-samples must be positive")
    if args.num_samples % args.batch_size != 0:
        raise ValueError("num-samples must be divisible by batch-size")
    if args.central_delta <= 0:
        raise ValueError("central-delta must be positive")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
    torch.set_float32_matmul_precision("high" if args.allow_tf32 else "highest")

    noise_bank, label_bank = _input_bank(args.num_samples, args.seed)
    output_dir = (
        args.output_root.expanduser().resolve()
        / args.study
        / args.direction
        / f"n{args.num_samples}_seed{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    time_grid = torch.linspace(0.0, 1.0, args.heun_steps + 1, device=device)
    started = time.perf_counter()
    initial_labels = label_bank[: args.batch_size].to(device)
    fields, metadata = _load_pair(args, initial_labels, device)
    manifest = _manifest(args, metadata, noise_bank, label_bank)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest != manifest:
            raise ValueError(
                "existing output manifest differs from this invocation; use a new "
                "output root or --overwrite"
            )
    else:
        atomic_json_dump(manifest, manifest_path)

    for batch_index, start in enumerate(range(0, args.num_samples, args.batch_size)):
        shard_path = _shard_path(output_dir, batch_index)
        if shard_path.exists() and not args.overwrite:
            print(json.dumps({"event": "skip", "shard": str(shard_path)}), flush=True)
            continue
        stop = start + args.batch_size
        noise = noise_bank[start:stop].to(device)
        labels = label_bank[start:stop].to(device)
        fields.labels = labels
        forwards_before = (fields.anchor_forwards, fields.other_forwards)
        if args.study == "linearity":
            payload = _run_linearity_batch(
                fields,
                noise,
                time_grid,
                args.gammas,
                args.central_delta,
            )
        elif args.study == "feedback":
            payload = _run_feedback_batch(
                fields,
                noise,
                time_grid,
                args.gammas,
            )
        elif args.study == "tangent_frozen":
            payload = _run_tangent_frozen_batch(
                fields,
                noise,
                time_grid,
                args.gammas,
                args.central_delta,
            )
        elif args.study == "solver":
            payload = _run_solver_batch(
                fields,
                noise,
                args.gammas,
                args.solver_step_counts,
                atol=args.adaptive_atol,
                rtol=args.adaptive_rtol,
            )
        else:  # pragma: no cover
            raise AssertionError(args.study)
        payload.update(
            {
                "sample_start": start,
                "sample_stop": stop,
                "noise": noise_bank[start:stop],
                "labels": label_bank[start:stop],
                "anchor_forwards": fields.anchor_forwards - forwards_before[0],
                "other_forwards": fields.other_forwards - forwards_before[1],
            }
        )
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = shard_path.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(shard_path)
        print(
            json.dumps(
                {
                    "event": "batch_complete",
                    "batch": batch_index,
                    "samples": [start, stop],
                    "elapsed_seconds": time.perf_counter() - started,
                    "shard": str(shard_path),
                }
            ),
            flush=True,
        )

    shard_paths = [
        _shard_path(output_dir, index)
        for index in range(args.num_samples // args.batch_size)
    ]
    missing = [str(path) for path in shard_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing trajectory shards: {missing[:3]}")
    shards = [torch.load(path, map_location="cpu", weights_only=False) for path in shard_paths]
    if args.study == "linearity":
        rows, summary = _aggregate_linearity(shards, args.central_delta)
    elif args.study == "feedback":
        rows, summary = _aggregate_feedback(shards)
    elif args.study == "tangent_frozen":
        rows, summary = _aggregate_tangent_frozen(shards, args.central_delta)
    else:
        rows, summary = _aggregate_solver(shards)
    _write_csv(rows, output_dir / "metrics.csv")
    summary.update(
        {
            "study": args.study,
            "direction": args.direction,
            "num_samples": args.num_samples,
            "elapsed_seconds": time.perf_counter() - started,
            "trajectory_shards": [str(path) for path in shard_paths],
        }
    )
    atomic_json_dump(summary, output_dir / "summary.json")
    print(json.dumps({"event": "complete", **summary}, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study",
        choices=("solver", "linearity", "feedback", "tangent_frozen"),
        required=True,
    )
    parser.add_argument(
        "--direction",
        choices=("x400", "v270", "x800", "v500"),
        required=True,
    )
    parser.add_argument("--anchor-checkpoint", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--x400-checkpoint", type=Path, default=DEFAULT_X400)
    parser.add_argument("--v270-checkpoint", type=Path, default=DEFAULT_V270)
    parser.add_argument("--x800-checkpoint", type=Path, default=DEFAULT_X800)
    parser.add_argument("--v500-checkpoint", type=Path, default=DEFAULT_V500)
    parser.add_argument("--official-sit-repo", type=Path, default=DEFAULT_OFFICIAL_SIT_REPO)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--heun-steps", type=int, default=50)
    parser.add_argument(
        "--gammas",
        type=_parse_floats,
        default=_parse_floats("0.02,0.05,0.1,0.2,0.5,0.75,1.0"),
    )
    parser.add_argument("--central-delta", type=float, default=0.01)
    parser.add_argument(
        "--solver-step-counts",
        type=_parse_ints,
        default=_parse_ints("25,50,100,200"),
    )
    parser.add_argument("--adaptive-atol", type=float, default=1e-7)
    parser.add_argument("--adaptive-rtol", type=float, default=1e-6)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--math-attention",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force the SDPA math kernel; defaults on for linearity/JVP only.",
    )
    parser.add_argument("--verify-sit-source", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    if parsed.math_attention is None:
        parsed.math_attention = parsed.study in {"linearity", "tangent_frozen"}
    main(parsed)
