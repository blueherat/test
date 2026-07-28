"""No-training RAE probe that switches vector fields by time and DCT band.

The paired baseline and spectral EMA models are combined only at inference.
This tests whether the high-noise rollout damage comes from the coarse radial
band whose teacher MSE was sacrificed by the spectral objective.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
RAE_SRC = ROOT / "external" / "RAE" / "src"
for import_path in (ROOT, RAE_SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from experiments.rae_spectral_direction_loss import DCTDirectionLoss  # noqa: E402
from experiments.rae_spectral_gradient_audit import (  # noqa: E402
    RAEAuditConfig,
    load_cached_latents,
    load_validation_labels,
)
from experiments.rae_teacher_rollout_gap import (  # noqa: E402
    configure_fp32,
    fixed_gaussian_matrix,
    latent_band_energy,
    latent_summary,
    official_time_grid,
    sliced_wasserstein,
)
from experiments.rae_vector_field_switch_probe import load_stage2  # noqa: E402


DEFAULT_SEED = 161803
FREQUENCY_SCHEDULES = (
    "baseline",
    "partial_high_all",
    "partial_high_band0",
    "partial_high_nonzero",
    "partial_mid_all",
    "partial_mid_band0",
    "partial_mid_nonzero",
)


def schedule_spec(
    schedule: str,
    time: float,
    band_count: int,
) -> tuple[bool, tuple[int, ...]]:
    """Return whether a switch is active and which partial output bands it uses."""

    if schedule == "baseline":
        return False, ()
    if schedule.startswith("partial_high_"):
        active = float(time) >= 0.85
        suffix = schedule.removeprefix("partial_high_")
    elif schedule.startswith("partial_mid_"):
        active = 0.30 <= float(time) < 0.85
        suffix = schedule.removeprefix("partial_mid_")
    else:
        raise ValueError(f"unknown frequency schedule: {schedule}")
    if suffix == "all":
        bands = tuple(range(int(band_count)))
    elif suffix.startswith("band") and suffix.removeprefix("band").isdigit():
        band = int(suffix.removeprefix("band"))
        if not 0 <= band < int(band_count):
            raise ValueError(f"invalid frequency schedule band: {band}")
        bands = (band,)
    elif suffix == "nonzero":
        bands = tuple(range(1, int(band_count)))
    else:
        raise ValueError(f"unknown frequency schedule suffix: {suffix}")
    return active, bands


@torch.no_grad()
def blend_velocity_bands(
    baseline_velocity: torch.Tensor,
    partial_velocity: torch.Tensor,
    partial_bands: Sequence[int],
    analyzer: DCTDirectionLoss,
) -> torch.Tensor:
    if baseline_velocity.shape != partial_velocity.shape:
        raise ValueError("paired velocities must have equal shapes")
    selected = torch.zeros(analyzer.band_count, device=baseline_velocity.device, dtype=torch.bool)
    for band in partial_bands:
        if not 0 <= int(band) < analyzer.band_count:
            raise ValueError(f"invalid band {band}")
        selected[int(band)] = True
    mask = selected[analyzer.band_index.to(baseline_velocity.device)]
    baseline_coefficients = analyzer.transform(baseline_velocity)
    partial_coefficients = analyzer.transform(partial_velocity)
    coefficients = torch.where(
        mask[None, None], partial_coefficients, baseline_coefficients
    )
    matrix = analyzer.dct.to(device=coefficients.device, dtype=coefficients.dtype)
    return torch.matmul(torch.matmul(matrix.T, coefficients), matrix)


@torch.no_grad()
def frequency_switched_endpoint(
    baseline: torch.nn.Module,
    partial: torch.nn.Module,
    noise: torch.Tensor,
    labels: torch.Tensor,
    times: torch.Tensor,
    analyzer: DCTDirectionLoss,
    schedule: str,
) -> torch.Tensor:
    state = noise
    times = times.to(device=state.device, dtype=state.dtype)
    for current, following in zip(times[:-1], times[1:]):
        active, bands = schedule_spec(schedule, float(current), analyzer.band_count)
        batch_time = torch.full(
            (len(state),), float(current), device=state.device, dtype=state.dtype
        )
        if not active:
            velocity = baseline(state, batch_time, y=labels)
        elif len(bands) == analyzer.band_count:
            velocity = partial(state, batch_time, y=labels)
        else:
            baseline_velocity = baseline(state, batch_time, y=labels)
            partial_velocity = partial(state, batch_time, y=labels)
            velocity = blend_velocity_bands(
                baseline_velocity, partial_velocity, bands, analyzer
            )
        state = state + (following - current) * velocity
    return state


@torch.no_grad()
def run_probe(
    baseline_branch: Path,
    partial_branch: Path,
    *,
    device: str,
    count: int,
    batch_size: int,
    evaluation_seed: int,
    schedules: Sequence[str] = FREQUENCY_SCHEDULES,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if int(count) < 2:
        raise ValueError("count must be at least two")
    configure_fp32(evaluation_seed)
    torch_device = torch.device(device)
    baseline_branch = baseline_branch.expanduser().resolve()
    partial_branch = partial_branch.expanduser().resolve()
    baseline, config, baseline_manifest = load_stage2(baseline_branch, torch_device)
    partial, partial_config, partial_manifest = load_stage2(partial_branch, torch_device)
    if int(baseline_manifest["global_seed"]) != int(partial_manifest["global_seed"]):
        raise ValueError("paired branches must share a training seed")
    if float(baseline_manifest["gamma"]) != 0.0 or float(partial_manifest["gamma"]) == 0.0:
        raise ValueError("expected gamma=0 baseline and nonzero partial branch")
    if OmegaConf.to_container(config.misc, resolve=True) != OmegaConf.to_container(
        partial_config.misc, resolve=True
    ):
        raise ValueError("paired branches disagree on latent/sampling configuration")

    audit = RAEAuditConfig(train_count=1, validation_count=int(count))
    payload = load_cached_latents(audit)
    clean_all = payload["validation"][:count].float()
    labels_all = load_validation_labels(
        audit.dataset_path, payload["validation_indices"][:count]
    )
    generator = torch.Generator(device="cpu").manual_seed(int(evaluation_seed))
    noise_all = torch.randn(clean_all.shape, generator=generator, dtype=torch.float32)

    spectral = config.training.spectral_direction_loss
    analyzer = DCTDirectionLoss(
        int(spectral.spatial_size),
        list(spectral.second_moments),
        gamma=0.0,
        damping=float(spectral.damping),
        min_weight=float(spectral.min_weight),
        max_weight=float(spectral.max_weight),
    ).to(torch_device)
    shift = math.sqrt(
        float(config.misc.time_dist_shift_dim) / float(config.misc.time_dist_shift_base)
    )
    times = official_time_grid(50, time_shift=shift).to(torch_device)
    projection = fixed_gaussian_matrix(clean_all.shape[1], 32, evaluation_seed + 17).to(
        torch_device
    )
    directions = fixed_gaussian_matrix(32 + analyzer.band_count, 64, evaluation_seed + 29)

    schedules = tuple(schedules)
    if len(set(schedules)) != len(schedules) or "baseline" not in schedules:
        raise ValueError("schedules must be unique and include baseline")
    for schedule in schedules:
        schedule_spec(schedule, 0.9, analyzer.band_count)
    endpoint_chunks: dict[str, list[torch.Tensor]] = {name: [] for name in schedules}
    clean_summary_chunks = []
    clean_band_chunks = []
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        clean = clean_all[start:end].to(torch_device)
        noise = noise_all[start:end].to(torch_device)
        labels = labels_all[start:end].to(torch_device)
        clean_summary_chunks.append(latent_summary(clean, analyzer, projection).cpu())
        clean_band_chunks.append(latent_band_energy(clean, analyzer).cpu())
        for schedule in schedules:
            endpoint = frequency_switched_endpoint(
                baseline, partial, noise, labels, times, analyzer, schedule
            )
            endpoint_chunks[schedule].append(endpoint.cpu())
        print(f"seed {baseline_manifest['global_seed']}: {end}/{count}", flush=True)

    clean_summary = torch.cat(clean_summary_chunks)
    clean_energy = torch.cat(clean_band_chunks).mean(0)
    endpoints = {name: torch.cat(chunks) for name, chunks in endpoint_chunks.items()}
    baseline_endpoint = endpoints["baseline"]
    baseline_summary = latent_summary(
        baseline_endpoint.to(torch_device), analyzer, projection
    ).cpu()
    identity = {
        "seed": int(baseline_manifest["global_seed"]),
        "sample_count": int(count),
    }
    rows = []
    band_rows = []
    for schedule, endpoint_cpu in endpoints.items():
        endpoint = endpoint_cpu.to(torch_device)
        summary = latent_summary(endpoint, analyzer, projection).cpu()
        energy = latent_band_energy(endpoint, analyzer).mean(0).cpu()
        log_ratio = (energy / clean_energy.clamp_min(1e-12)).log()
        relative_rms = (
            (endpoint_cpu - baseline_endpoint).square().flatten(1).mean(1).sqrt()
            / baseline_endpoint.square().flatten(1).mean(1).sqrt().clamp_min(1e-12)
        )
        values = {
            "summary_swd_to_validation": float(
                sliced_wasserstein(clean_summary, summary, directions)
            ),
            "summary_swd_to_baseline_endpoint": float(
                sliced_wasserstein(baseline_summary, summary, directions)
            ),
            "energy_abs_log_gap_to_validation": float(log_ratio.abs().mean()),
            "paired_endpoint_relative_rms_to_baseline": float(relative_rms.mean()),
        }
        rows.extend(
            {**identity, "schedule": schedule, "metric": metric, "value": value}
            for metric, value in values.items()
        )
        band_rows.extend(
            {
                **identity,
                "schedule": schedule,
                "metric": "energy_log_ratio_to_validation",
                "band": int(band),
                "value": float(value),
            }
            for band, value in enumerate(log_ratio)
        )

    metadata = {
        **identity,
        "baseline_branch": str(baseline_branch),
        "partial_branch": str(partial_branch),
        "evaluation_seed": int(evaluation_seed),
        "validation_indices": [int(v) for v in payload["validation_indices"][:count]],
        "precision": "fp32",
        "tf32": False,
        "times": [float(value) for value in times.cpu()],
        "schedules": list(schedules),
        "scope": "same-noise time-and-output-band vector-field switch; latent proxy, not FID",
    }
    return pd.DataFrame(rows), pd.DataFrame(band_rows), metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--evaluation-seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    metrics, bands, metadata = run_probe(
        args.baseline,
        args.partial,
        device=args.device,
        count=args.count,
        batch_size=args.batch_size,
        evaluation_seed=args.evaluation_seed,
    )
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "metrics.csv", index=False)
    bands.to_csv(output / "bands.csv", index=False)
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(metrics.to_string(index=False), flush=True)
    print(output, flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "FREQUENCY_SCHEDULES",
    "blend_velocity_bands",
    "frequency_switched_endpoint",
    "run_probe",
    "schedule_spec",
]
