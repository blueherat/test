"""No-training probe for faster RAE Euler time grids.

The 50-step official trajectory is the paired numerical reference.  Reduced
grids use the same EMA model, initial noise, and labels, so endpoint errors are
valid per-sample solver errors.  This is a screening proxy, not a replacement
for 5k/50k FID.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import pandas as pd
import torch


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
    compare_decoder_features,
    decode_features_batched,
    fixed_gaussian_matrix,
    latent_band_energy,
    latent_summary,
    load_inception,
    load_models,
    official_time_grid,
    sliced_wasserstein,
)


DEFAULT_SEED = 130363


def unique_indices(count: int, total: int) -> torch.Tensor:
    indices = torch.linspace(0, int(total) - 1, int(count)).round().long().unique(sorted=True)
    if len(indices) != int(count):
        raise RuntimeError("rounded schedule indices were not unique")
    return indices


def candidate_time_grids(
    official: torch.Tensor,
    *,
    time_shift: float = math.sqrt(196608.0 / 4096.0),
) -> dict[str, torch.Tensor]:
    terminal = float(official[-1])
    hybrid20 = torch.cat(
        [official[:5].cpu(), torch.linspace(float(official[5]), terminal, 15)]
    )
    hybrid16 = torch.cat(
        [official[:4].cpu(), torch.linspace(float(official[4]), terminal, 12)]
    )
    return {
        "official_50": official.cpu(),
        "official_numsteps_25": official_time_grid(25, time_shift=time_shift),
        "official_numsteps_16": official_time_grid(16, time_shift=time_shift),
        "shifted_subsample_25": official[unique_indices(25, len(official))].cpu(),
        "shifted_subsample_16": official[unique_indices(16, len(official))].cpu(),
        "uniform_actual_t_20": torch.linspace(float(official[0]), terminal, 20),
        "hybrid_early5_uniformlate_20": hybrid20,
        "hybrid_early4_uniformlate_16": hybrid16,
    }


@torch.no_grad()
def rollout_endpoint(
    model: torch.nn.Module,
    noise: torch.Tensor,
    labels: torch.Tensor,
    times: torch.Tensor,
) -> torch.Tensor:
    state = noise
    times = times.to(device=state.device, dtype=state.dtype)
    for current, following in zip(times[:-1], times[1:]):
        batch_time = torch.full(
            (len(state),), float(current), device=state.device, dtype=state.dtype
        )
        velocity = model(state, batch_time, y=labels)
        state = state + (following - current) * velocity
    return state


@torch.no_grad()
def run_probe(
    branch: Path,
    *,
    device: str,
    count: int,
    batch_size: int,
    perceptual_count: int,
    perceptual_batch_size: int,
    evaluation_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    configure_fp32(evaluation_seed)
    torch_device = torch.device(device)
    branch = branch.expanduser().resolve()
    manifest = json.loads((branch / "manifest.json").read_text(encoding="utf-8"))
    model, rae, config = load_models(branch, torch_device)
    inception = load_inception(torch_device) if perceptual_count > 0 else None

    audit_config = RAEAuditConfig(train_count=1, validation_count=int(count))
    payload = load_cached_latents(audit_config)
    clean_all = payload["validation"][:count].float()
    labels_all = load_validation_labels(
        audit_config.dataset_path, payload["validation_indices"][:count]
    )
    generator = torch.Generator(device="cpu").manual_seed(int(evaluation_seed))
    noise_all = torch.randn(clean_all.shape, generator=generator)

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
    official = official_time_grid(50, time_shift=shift)
    grids = candidate_time_grids(official, time_shift=shift)
    projection = fixed_gaussian_matrix(clean_all.shape[1], 32, evaluation_seed + 5).to(torch_device)
    directions = fixed_gaussian_matrix(32 + analyzer.band_count, 64, evaluation_seed + 7)

    endpoint_chunks: dict[str, list[torch.Tensor]] = {name: [] for name in grids}
    elapsed: dict[str, float] = {name: 0.0 for name in grids}
    paired_values: dict[tuple[str, str], list[torch.Tensor]] = {}
    clean_summaries: list[torch.Tensor] = []
    clean_bands: list[torch.Tensor] = []
    schedule_summaries: dict[str, list[torch.Tensor]] = {name: [] for name in grids}
    schedule_bands: dict[str, list[torch.Tensor]] = {name: [] for name in grids}

    for batch_index, start in enumerate(range(0, count, batch_size)):
        end = min(start + batch_size, count)
        clean = clean_all[start:end].to(torch_device)
        noise = noise_all[start:end].to(torch_device)
        labels = labels_all[start:end].to(torch_device)
        if batch_index == 0:
            warm_time = torch.ones(len(noise), device=torch_device)
            _ = model(noise, warm_time, y=labels)
            torch.cuda.synchronize(torch_device)

        order = list(grids)
        order = order[batch_index % len(order) :] + order[: batch_index % len(order)]
        endpoints: dict[str, torch.Tensor] = {}
        for name in order:
            torch.cuda.synchronize(torch_device)
            started = time.perf_counter()
            endpoint = rollout_endpoint(model, noise, labels, grids[name])
            torch.cuda.synchronize(torch_device)
            elapsed[name] += time.perf_counter() - started
            endpoints[name] = endpoint

        reference = endpoints["official_50"]
        perceptual_local = max(0, min(end, perceptual_count) - start)
        if perceptual_local > 0 and inception is not None:
            reference_features = decode_features_batched(
                rae, inception, reference[:perceptual_local], perceptual_batch_size
            )
        else:
            reference_features = None

        clean_summaries.append(latent_summary(clean, analyzer, projection).cpu())
        clean_bands.append(latent_band_energy(clean, analyzer).cpu())
        for name, endpoint in endpoints.items():
            endpoint_chunks[name].append(endpoint.cpu())
            schedule_summaries[name].append(latent_summary(endpoint, analyzer, projection).cpu())
            schedule_bands[name].append(latent_band_energy(endpoint, analyzer).cpu())
            error = endpoint - reference
            relative_rms = error.square().flatten(1).mean(1).sqrt() / reference.square().flatten(1).mean(1).sqrt().clamp_min(1e-12)
            paired_values.setdefault((name, "endpoint_latent_relative_rms"), []).append(relative_rms.cpu())
            band_error = analyzer.band_mse(error)
            for band in range(analyzer.band_count):
                paired_values.setdefault((name, f"endpoint_band_mse_{band}"), []).append(
                    band_error[:, band].cpu()
                )
            if reference_features is not None:
                candidate_features = decode_features_batched(
                    rae, inception, endpoint[:perceptual_local], perceptual_batch_size
                )
                for metric, values in compare_decoder_features(
                    candidate_features, reference_features
                ).items():
                    paired_values.setdefault((name, f"endpoint_{metric}"), []).append(values.cpu())
        print(f"{branch.name}: {end}/{count}", flush=True)

    identity = {
        "branch": branch.name,
        "seed": int(manifest["global_seed"]),
        "treatment": "baseline" if float(manifest["gamma"]) == 0 else "partial",
    }
    rows: list[dict[str, object]] = []
    for (schedule, metric), chunks in sorted(paired_values.items()):
        values = torch.cat(chunks)
        rows.append(
            {
                **identity,
                "schedule": schedule,
                "points": int(len(grids[schedule])),
                "model_evaluations": int(len(grids[schedule]) - 1),
                "theoretical_speedup": 49.0 / float(len(grids[schedule]) - 1),
                "metric": metric,
                "value": float(values.mean()),
                "sample_count": int(len(values)),
            }
        )

    clean_summary = torch.cat(clean_summaries)
    clean_energy = torch.cat(clean_bands).mean(0)
    distribution_rows: list[dict[str, object]] = []
    for schedule in grids:
        summary = torch.cat(schedule_summaries[schedule])
        energy = torch.cat(schedule_bands[schedule]).mean(0)
        distribution_rows.append(
            {
                **identity,
                "schedule": schedule,
                "points": int(len(grids[schedule])),
                "metric": "endpoint_summary_swd_to_validation",
                "band": -1,
                "value": float(sliced_wasserstein(clean_summary, summary, directions)),
            }
        )
        log_ratio = (energy / clean_energy.clamp_min(1e-12)).log()
        for band, value in enumerate(log_ratio):
            distribution_rows.append(
                {
                    **identity,
                    "schedule": schedule,
                    "points": int(len(grids[schedule])),
                    "metric": "endpoint_energy_log_ratio_to_validation",
                    "band": int(band),
                    "value": float(value),
                }
            )

    timing = {
        name: {
            "seconds": elapsed[name],
            "images": int(count),
            "seconds_per_image": elapsed[name] / float(count),
        }
        for name in grids
    }
    metadata = {
        **identity,
        "count": int(count),
        "perceptual_count": int(perceptual_count),
        "evaluation_seed": int(evaluation_seed),
        "reference": "same-model official shifted 50-point Euler grid",
        "scope": "numerical endpoint fidelity proxy; not generation FID",
        "grids": {name: [float(value) for value in grid] for name, grid in grids.items()},
        "timing": timing,
    }
    return pd.DataFrame(rows), pd.DataFrame(distribution_rows), metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--perceptual-count", type=int, default=12)
    parser.add_argument("--perceptual-batch-size", type=int, default=2)
    parser.add_argument("--evaluation-seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    metrics, distribution, metadata = run_probe(
        args.branch,
        device=args.device,
        count=args.count,
        batch_size=args.batch_size,
        perceptual_count=args.perceptual_count,
        perceptual_batch_size=args.perceptual_batch_size,
        evaluation_seed=args.evaluation_seed,
    )
    output = args.branch.expanduser().resolve() / "step_schedule_probe"
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "metrics.csv", index=False)
    distribution.to_csv(output / "distribution.csv", index=False)
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(metrics.to_string(index=False), flush=True)
    print(output, flush=True)


if __name__ == "__main__":
    main()


__all__ = ["candidate_time_grids", "rollout_endpoint", "run_probe", "unique_indices"]
