"""Localize the teacher-to-rollout gap by switching vector fields in time.

The baseline and spectral branches share a source checkpoint and are paired by
training seed.  This no-training probe integrates the same initial noise with
hard switches between their EMA vector fields.  It tests whether degradation
is introduced mainly at high, middle, or low noise; it is not a generation
quality metric.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
RAE_SRC = ROOT / "external" / "RAE" / "src"
for import_path in (ROOT, RAE_SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from experiments.rae_spectral_direction_loss import DCTDirectionLoss
from experiments.rae_spectral_gradient_audit import (
    RAEAuditConfig,
    load_cached_latents,
    load_validation_labels,
)
from experiments.rae_teacher_rollout_gap import (
    configure_fp32,
    fixed_gaussian_matrix,
    latent_band_energy,
    latent_summary,
    official_time_grid,
    sliced_wasserstein,
)
from utils.model_utils import instantiate_from_config


DEFAULT_SEED = 161803
SCHEDULES = (
    "baseline",
    "partial",
    "partial_high_ge_085",
    "partial_mid_030_085",
    "partial_low_lt_030",
    "baseline_high_partial_below_085",
)


def uses_partial(schedule: str, time: float) -> bool:
    """Return which paired vector field a hard-switch schedule uses."""

    value = float(time)
    if schedule == "baseline":
        return False
    if schedule == "partial":
        return True
    if schedule == "partial_high_ge_085":
        return value >= 0.85
    if schedule == "partial_mid_030_085":
        return 0.30 <= value < 0.85
    if schedule == "partial_low_lt_030":
        return value < 0.30
    if schedule == "baseline_high_partial_below_085":
        return value < 0.85
    raise ValueError(f"unknown switch schedule: {schedule}")


def load_stage2(branch: Path, device: torch.device) -> tuple[torch.nn.Module, OmegaConf, dict]:
    config = OmegaConf.load(branch / "config.yaml")
    model = instantiate_from_config(config.stage_2).to(device=device, dtype=torch.float32)
    checkpoint = branch / "generation" / "ema_step-0010000.pt"
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
    )
    model.requires_grad_(False).eval()
    manifest = json.loads((branch / "manifest.json").read_text(encoding="utf-8"))
    return model, config, manifest


@torch.no_grad()
def switched_endpoint(
    baseline: torch.nn.Module,
    partial: torch.nn.Module,
    noise: torch.Tensor,
    labels: torch.Tensor,
    times: torch.Tensor,
    schedule: str,
) -> torch.Tensor:
    state = noise
    times = times.to(device=state.device, dtype=state.dtype)
    for current, following in zip(times[:-1], times[1:]):
        model = partial if uses_partial(schedule, float(current)) else baseline
        batch_time = torch.full(
            (len(state),), float(current), device=state.device, dtype=state.dtype
        )
        velocity = model(state, batch_time, y=labels)
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
        raise ValueError("baseline and partial branches must share a training seed")
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

    endpoint_chunks: dict[str, list[torch.Tensor]] = {name: [] for name in SCHEDULES}
    clean_summary_chunks: list[torch.Tensor] = []
    clean_band_chunks: list[torch.Tensor] = []
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        clean = clean_all[start:end].to(torch_device)
        noise = noise_all[start:end].to(torch_device)
        labels = labels_all[start:end].to(torch_device)
        clean_summary_chunks.append(latent_summary(clean, analyzer, projection).cpu())
        clean_band_chunks.append(latent_band_energy(clean, analyzer).cpu())
        for schedule in SCHEDULES:
            endpoint = switched_endpoint(
                baseline, partial, noise, labels, times, schedule
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

    rows: list[dict[str, object]] = []
    band_rows: list[dict[str, object]] = []
    identity = {
        "seed": int(baseline_manifest["global_seed"]),
        "sample_count": int(count),
    }
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
        for metric, value in values.items():
            rows.append({**identity, "schedule": schedule, "metric": metric, "value": value})
        for band, value in enumerate(log_ratio):
            band_rows.append(
                {
                    **identity,
                    "schedule": schedule,
                    "metric": "energy_log_ratio_to_validation",
                    "band": int(band),
                    "value": float(value),
                }
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
        "schedules": list(SCHEDULES),
        "scope": "same-noise hard vector-field switch; distribution proxy, not FID",
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


__all__ = ["SCHEDULES", "run_probe", "switched_endpoint", "uses_partial"]
