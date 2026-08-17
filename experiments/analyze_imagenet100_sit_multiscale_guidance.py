#!/usr/bin/env python3
"""Build the successful/failed latent spectrum and calibration atlas."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from torchdiffeq import odeint

try:
    from experiments.imagenet100_sit_multiscale_guidance import (
        BAND_NAMES,
        TIME_NAMES,
        band_time_component,
        frequency_statistics,
        observation_time_grid,
        per_sample_mean_square,
        per_sample_rms,
        project_frequency_band,
    )
    from experiments.imagenet100_sit_multiscale_models import (
        evaluate_sit_field,
        evaluate_source_with_heads,
        load_internal_head_for_source,
        load_sit_field_model,
    )
    from experiments.imagenet100_sit_vx_dual_head import (
        clean_prediction_to_velocity,
        split_velocity_clean_output,
    )
    from experiments.sample_imagenet100_sit_frozen_v_clean_head_fid import (
        load_frozen_velocity_clean_model,
    )
    from experiments.train_imagenet100_sit_flow import (
        DEFAULT_OFFICIAL_SIT_REPO,
        LATENT_SHAPE,
        NUM_CLASSES,
        atomic_json_dump,
        load_official_sit_module,
    )
except ModuleNotFoundError:
    from imagenet100_sit_multiscale_guidance import (
        BAND_NAMES,
        TIME_NAMES,
        band_time_component,
        frequency_statistics,
        observation_time_grid,
        per_sample_mean_square,
        per_sample_rms,
        project_frequency_band,
    )
    from imagenet100_sit_multiscale_models import (
        evaluate_sit_field,
        evaluate_source_with_heads,
        load_internal_head_for_source,
        load_sit_field_model,
    )
    from imagenet100_sit_vx_dual_head import (
        clean_prediction_to_velocity,
        split_velocity_clean_output,
    )
    from sample_imagenet100_sit_frozen_v_clean_head_fid import (
        load_frozen_velocity_clean_model,
    )
    from train_imagenet100_sit_flow import (
        DEFAULT_OFFICIAL_SIT_REPO,
        LATENT_SHAPE,
        NUM_CLASSES,
        atomic_json_dump,
        load_official_sit_module,
    )


DEFAULT_ROOT = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/multiscale_guidance_study_v1"
)
DEFAULT_V800 = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/"
    "sit-s-2_seed0/checkpoints/step_00800000.pt"
)
DEFAULT_V500 = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/"
    "sit-s-2_seed0/checkpoints/step_00500000.pt"
)
DEFAULT_DEPTH8_V = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/"
    "sit-s-2_v800-ema_frozen-internal-v-depth8_seed0/checkpoints/step_00050000.pt"
)
DEFAULT_DEPTH8_X = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/"
    "sit-s-2_v800-ema_frozen-internal-x-depth8_seed0/checkpoints/step_00050000.pt"
)
DEFAULT_DEPTH8_EPS = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/"
    "sit-s-2_v800-ema_frozen-internal-eps-depth8_seed0/checkpoints/step_00050000.pt"
)
DEFAULT_DEPTH12_X = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/"
    "sit-s-2_v800-ema_frozen-final-x-fullhead-depth12_seed0/"
    "checkpoints/step_00050000.pt"
)
DEFAULT_FINAL_LINEAR_X = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/"
    "sit-s-2_v800-ema_frozen-clean-head_seed0/checkpoints/step_00050000.pt"
)


PROVIDER_LABELS = {
    "depth8_v": ("useful", 12.4539),
    "depth8_x": ("useful", 5.6779),
    "depth8_epsilon": ("useful", 7.1910),
    "depth12_x": ("failed", 0.0458),
    "final_linear_x": ("failed", -0.3325),
    "raw_final_h8": ("failed", -0.0390),
    "external_v500": ("useful", 6.6),
}
ACTION_PROVIDERS = ("depth8_v", "depth12_x", "external_v500")


def parse_head_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("head must use NAME=PATH")
    name, path = value.split("=", maxsplit=1)
    if not name or not path:
        raise argparse.ArgumentTypeError("head must use non-empty NAME=PATH")
    return name, Path(path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean_stats(field: torch.Tensor) -> dict[str, float]:
    return {
        name: float(value.mean().item())
        for name, value in frequency_statistics(field).items()
    }


def _symmetric_nmse(left: torch.Tensor, right: torch.Tensor) -> float:
    numerator = (left.float() - right.float()).square().mean()
    denominator = 0.5 * (
        left.float().square().mean() + right.float().square().mean()
    )
    return float((numerator / denominator.clamp_min(1e-12)).item())


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_flat = left.float().flatten(1)
    right_flat = right.float().flatten(1)
    value = torch.nn.functional.cosine_similarity(left_flat, right_flat, dim=1)
    return float(value.mean().item())


def fit_band_delays(
    *,
    times: list[float],
    strong_clean: list[torch.Tensor],
    weak_clean: list[torch.Tensor],
    max_lag_steps: int,
) -> tuple[dict[str, float], dict[str, int], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    fitted_time: dict[str, float] = {}
    fitted_steps: dict[str, int] = {}
    median_step = float(np.median(np.diff(np.asarray(times, dtype=np.float64))))
    for band in BAND_NAMES:
        band_rows: list[dict[str, object]] = []
        for lag in range(max_lag_steps + 1):
            requested_lag_time = lag * median_step
            nmse_values: list[float] = []
            cosine_values: list[float] = []
            realized_lag_times: list[float] = []
            for index in range(len(times)):
                if not 0.1 <= times[index] <= 0.9:
                    continue
                target_time = times[index] - requested_lag_time
                if target_time < times[0]:
                    continue
                source_index = min(
                    range(index + 1),
                    key=lambda candidate: abs(times[candidate] - target_time),
                )
                weak_band = project_frequency_band(weak_clean[index], band)
                strong_band = project_frequency_band(strong_clean[source_index], band)
                nmse_values.append(_symmetric_nmse(weak_band, strong_band))
                cosine_values.append(_cosine(weak_band, strong_band))
                realized_lag_times.append(times[index] - times[source_index])
            row = {
                "band": band,
                "atlas_lag_steps": lag,
                "requested_lag_time": requested_lag_time,
                "realized_lag_time_mean": float(np.mean(realized_lag_times)),
                "symmetric_nmse": float(np.mean(nmse_values)),
                "cosine": float(np.mean(cosine_values)),
                "comparison_count": len(nmse_values),
            }
            rows.append(row)
            band_rows.append(row)
        best = min(band_rows, key=lambda row: row["symmetric_nmse"])
        fitted_steps[band] = int(best["atlas_lag_steps"])
        fitted_time[band] = float(best["realized_lag_time_mean"])
    return fitted_time, fitted_steps, rows


def integrated_actions(
    times: list[float],
    action_series: dict[str, dict[str, list[float]]],
    *,
    max_equal_action_scale: float,
) -> dict[str, object]:
    payload: dict[str, object] = {}
    x = np.asarray(times, dtype=np.float64)
    for provider, cells in action_series.items():
        actions = {
            cell: float(np.trapezoid(np.asarray(values, dtype=np.float64), x=x))
            for cell, values in cells.items()
        }
        positive = [value for value in actions.values() if value > 0.0]
        reference = float(np.median(positive))
        scales = {
            cell: min(
                float(max_equal_action_scale),
                math.sqrt(reference / max(value, np.finfo(np.float64).tiny)),
            )
            for cell, value in actions.items()
        }
        payload[provider] = {
            "cell_actions": actions,
            "reference_action": reference,
            "equal_action_scales": scales,
            "max_scale": float(max_equal_action_scale),
            "capped_cells": [
                cell
                for cell, scale in scales.items()
                if math.isclose(scale, float(max_equal_action_scale))
            ],
        }
    return payload


@torch.inference_mode()
def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sit_module, source_metadata = load_official_sit_module(
        args.official_sit_repo.expanduser().resolve(),
        verify_source=args.verify_sit_source,
    )
    strong, strong_semantics, strong_metadata = load_sit_field_model(
        checkpoint_path=args.strong_checkpoint,
        weights="ema",
        sit_module=sit_module,
        source_metadata=source_metadata,
        device=device,
    )
    if strong_semantics.prediction_target != "velocity":
        raise ValueError("the strong source must be a native velocity model")
    external, external_semantics, external_metadata = load_sit_field_model(
        checkpoint_path=args.external_weak_checkpoint,
        weights="ema",
        sit_module=sit_module,
        source_metadata=source_metadata,
        device=device,
    )
    if external_semantics.prediction_target != "velocity":
        raise ValueError("the external weak model must be a native velocity model")

    head_paths = dict(args.head)
    required = {
        "depth8_v": args.depth8_v_checkpoint,
        "depth8_x": args.depth8_x_checkpoint,
        "depth8_epsilon": args.depth8_epsilon_checkpoint,
        "depth12_x": args.depth12_x_checkpoint,
    }
    for name, path in required.items():
        if name in head_paths and head_paths[name].resolve() != path.resolve():
            raise ValueError(f"duplicate conflicting path for {name}")
        head_paths[name] = path
    heads = {
        name: load_internal_head_for_source(
            checkpoint_path=path,
            name=name,
            head_weights="ema",
            model=strong,
            sit_module=sit_module,
            source_checkpoint_path=args.strong_checkpoint,
            source_metadata=source_metadata,
            device=device,
        )
        for name, path in head_paths.items()
    }
    depth_head_names = {
        spec.depth: name
        for name, spec in heads.items()
        if spec.prediction_target == "velocity" and name.startswith("depth")
    }
    if 8 not in depth_head_names:
        raise ValueError("a depth-8 velocity head is required")

    final_linear_model, final_linear_metadata = load_frozen_velocity_clean_model(
        head_checkpoint_path=args.final_linear_x_checkpoint,
        head_weights="ema",
        sit_module=sit_module,
        source_metadata=source_metadata,
        device=device,
    )

    generator = torch.Generator(device=device).manual_seed(args.seed)
    noise = torch.randn(
        args.samples,
        *LATENT_SHAPE,
        generator=generator,
        device=device,
    )
    labels = torch.randint(
        0,
        NUM_CLASSES,
        (args.samples,),
        generator=generator,
        device=device,
    )
    times = observation_time_grid(
        args.time_min,
        args.time_max,
        args.time_points,
        anchors=(0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95),
    )
    time_tensor = torch.as_tensor([0.0, *times], device=device, dtype=torch.float32)
    nfe = 0

    def strong_field(time_value: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        nonlocal nfe
        nfe += 1
        expanded = time_value.expand(len(state))
        return evaluate_sit_field(
            strong,
            strong_semantics,
            state,
            expanded,
            labels,
        )

    trajectory = odeint(
        strong_field,
        noise.float(),
        time_tensor,
        method="dopri5",
        atol=args.atol,
        rtol=args.rtol,
    )[1:]

    rows: list[dict[str, object]] = []
    action_series: dict[str, dict[str, list[float]]] = {
        provider: {
            f"{interval}_{band}": []
            for interval in TIME_NAMES
            for band in BAND_NAMES
        }
        for provider in ACTION_PROVIDERS
    }
    rms_tables: dict[str, list[float]] = {}
    strong_clean: list[torch.Tensor] = []
    weak_depth8_clean: list[torch.Tensor] = []
    raw_depths = tuple(sorted(set(args.raw_depths)))

    for time_value, state in zip(times, trajectory, strict=True):
        expanded = torch.full(
            (len(state),),
            float(time_value),
            device=device,
            dtype=torch.float32,
        )
        full, trained, raw = evaluate_source_with_heads(
            strong,
            state,
            expanded,
            labels,
            heads=heads,
            raw_depths=raw_depths,
        )
        external_velocity = evaluate_sit_field(
            external,
            external_semantics,
            state,
            expanded,
            labels,
        )
        final_output = final_linear_model(state, expanded, labels)
        final_velocity, final_clean = split_velocity_clean_output(
            final_output,
            latent_channels=LATENT_SHAPE[0],
        )
        if not torch.allclose(full, final_velocity.float(), atol=2e-5, rtol=2e-5):
            raise ValueError("final-linear probe changed the frozen strong velocity")
        final_clean_velocity = clean_prediction_to_velocity(
            final_clean,
            state=state,
            time_value=expanded,
            denominator_floor=float(final_linear_metadata["denominator_floor"]),
        )
        gaps: dict[str, torch.Tensor] = {
            name: full - value for name, value in trained.items()
        }
        gaps["external_v500"] = full - external_velocity
        gaps["final_linear_x"] = full - final_clean_velocity
        for depth, value in raw.items():
            gaps[f"raw_final_h{depth}"] = full - value

        for provider, gap in gaps.items():
            stats = _mean_stats(gap)
            label, known_gain = PROVIDER_LABELS.get(provider, ("depth_probe", math.nan))
            rows.append(
                {
                    "provider": provider,
                    "evidence_label": label,
                    "known_best_fid1k_gain": known_gain,
                    "time": time_value,
                    **stats,
                }
            )
            rms_tables.setdefault(provider, []).append(stats["rms"])

        for provider in ACTION_PROVIDERS:
            gap = gaps[provider]
            for interval in TIME_NAMES:
                for band in BAND_NAMES:
                    cell = f"{interval}_{band}"
                    component = band_time_component(
                        gap,
                        expanded,
                        band=band,
                        interval=interval,
                        transition_width=args.transition_width,
                    )
                    action_series[provider][cell].append(
                        float(per_sample_mean_square(component).mean().item())
                    )

        strong_clean.append(
            (state.float() + (1.0 - float(time_value)) * full.float()).cpu()
        )
        weak_depth8 = trained[depth_head_names[8]]
        weak_depth8_clean.append(
            (state.float() + (1.0 - float(time_value)) * weak_depth8.float()).cpu()
        )

    fitted_lag_time, fitted_lag_steps, delay_rows = fit_band_delays(
        times=times,
        strong_clean=strong_clean,
        weak_clean=weak_depth8_clean,
        max_lag_steps=args.max_delay_steps,
    )
    action_payload = integrated_actions(
        times,
        action_series,
        max_equal_action_scale=args.max_equal_action_scale,
    )
    reference_rms = np.asarray(rms_tables[depth_head_names[8]], dtype=np.float64)
    rms_calibration = {}
    for provider, values in rms_tables.items():
        denominator = np.maximum(np.asarray(values, dtype=np.float64), 1e-12)
        scale = np.minimum(args.max_rms_match_scale, reference_rms / denominator)
        rms_calibration[provider] = {
            "times": times,
            "rms": np.asarray(values, dtype=np.float64).tolist(),
            "scale_to_depth8_v": scale.tolist(),
            "capped_count": int(np.isclose(scale, args.max_rms_match_scale).sum()),
        }

    write_csv(args.output_dir / "latent_spectrum_atlas.csv", rows)
    write_csv(args.output_dir / "band_delay_fit.csv", delay_rows)
    payload = {
        "format": "eqvae_imagenet100_sit_multiscale_atlas_v1",
        "scope": (
            "paired latent-space successful/failed gap spectra, equal-action "
            "calibration, and band-wise clean-prediction delay fitting"
        ),
        "frequency_semantics": {
            "space": "SD-VAE latent spatial grid",
            "channel_handling": "independent per-channel FFT; energy summed across channels",
            "dc_removed": False,
            "bands_cycles_per_latent_pixel": {
                "low": [0.0, 0.125],
                "mid": [0.125, 0.25],
                "high": [0.25, math.sqrt(0.5)],
            },
        },
        "strong": strong_metadata,
        "external_weak": external_metadata,
        "final_linear_head": final_linear_metadata,
        "heads": {
            name: {
                "depth": spec.depth,
                "prediction_target": spec.prediction_target,
                "checkpoint": spec.checkpoint,
                "checkpoint_sha256": spec.checkpoint_sha256,
            }
            for name, spec in heads.items()
        },
        "samples": args.samples,
        "seed": args.seed,
        "times": times,
        "baseline_nfe": nfe,
        "trajectory_shape": list(trajectory.shape),
        "action_calibration": action_payload,
        "rms_calibration": rms_calibration,
        "delay_fit": {
            "definition": (
                "band-wise symmetric NMSE between weak clean estimate at t and "
                "strong clean estimate at an earlier baseline-trajectory time; "
                "the fitted physical time delay is converted to sampler steps"
            ),
            "fitted_lag_time": fitted_lag_time,
            "atlas_lag_steps": fitted_lag_steps,
            "max_lag_steps": args.max_delay_steps,
        },
        "files": {
            "atlas": str(args.output_dir / "latent_spectrum_atlas.csv"),
            "delay": str(args.output_dir / "band_delay_fit.csv"),
        },
    }
    atomic_json_dump(payload, args.output_dir / "atlas_summary.json")
    print(json.dumps({"status": "complete", "output": str(args.output_dir)}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "atlas")
    parser.add_argument("--official-sit-repo", type=Path, default=DEFAULT_OFFICIAL_SIT_REPO)
    parser.add_argument("--strong-checkpoint", type=Path, default=DEFAULT_V800)
    parser.add_argument("--external-weak-checkpoint", type=Path, default=DEFAULT_V500)
    parser.add_argument("--depth8-v-checkpoint", type=Path, default=DEFAULT_DEPTH8_V)
    parser.add_argument("--depth8-x-checkpoint", type=Path, default=DEFAULT_DEPTH8_X)
    parser.add_argument(
        "--depth8-epsilon-checkpoint", type=Path, default=DEFAULT_DEPTH8_EPS
    )
    parser.add_argument("--depth12-x-checkpoint", type=Path, default=DEFAULT_DEPTH12_X)
    parser.add_argument(
        "--final-linear-x-checkpoint", type=Path, default=DEFAULT_FINAL_LINEAR_X
    )
    parser.add_argument(
        "--head",
        action="append",
        type=parse_head_argument,
        default=[],
        metavar="NAME=PATH",
    )
    parser.add_argument("--raw-depths", type=int, nargs="+", default=(4, 6, 8, 10))
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--time-min", type=float, default=0.02)
    parser.add_argument("--time-max", type=float, default=0.98)
    parser.add_argument("--time-points", type=int, default=49)
    parser.add_argument("--max-delay-steps", type=int, default=8)
    parser.add_argument("--transition-width", type=float, default=0.04)
    parser.add_argument("--max-equal-action-scale", type=float, default=8.0)
    parser.add_argument("--max-rms-match-scale", type=float, default=8.0)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--verify-sit-source", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
