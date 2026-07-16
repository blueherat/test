"""Audit RAE radial-band marginal drift on shared teacher and rollout states.

For each paired baseline/spectral seed, both vector fields are evaluated on:

* exact validation interpolation states,
* baseline-generated rollout states, and
* spectral-generated rollout states.

The key quantity is ``d E[z_b^2] / dt = 2 E[z_b v_b]``.  Shared states
separate an on-path marginal-transport error from an off-path response.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

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
    DEFAULT_TARGET_TIMES,
    configure_fp32,
    official_time_grid,
    select_time_indices,
)
from experiments.rae_vector_field_switch_probe import load_stage2  # noqa: E402


DEFAULT_SEED = 161803


def band_cross_mean(
    first: torch.Tensor,
    second: torch.Tensor,
    analyzer: DCTDirectionLoss,
) -> torch.Tensor:
    if first.shape != second.shape:
        raise ValueError("band cross inputs must have equal shapes")
    product = (analyzer.transform(first) * analyzer.transform(second)).mean(dim=1).flatten(1)
    index = analyzer.band_index.flatten().to(first.device)
    sums = torch.zeros(
        (len(first), analyzer.band_count), device=first.device, dtype=first.dtype
    )
    sums.scatter_add_(1, index[None].expand(len(first), -1), product)
    counts = analyzer.band_counts.to(device=first.device, dtype=first.dtype)
    return sums / counts[None]


@torch.no_grad()
def selected_rollout_states(
    model: torch.nn.Module,
    noise: torch.Tensor,
    labels: torch.Tensor,
    times: torch.Tensor,
    selected_indices: set[int],
) -> dict[int, torch.Tensor]:
    state = noise
    selected = {}
    for index, (current, following) in enumerate(zip(times[:-1], times[1:])):
        if index in selected_indices:
            selected[index] = state
        batch_time = torch.full(
            (len(state),), float(current), device=state.device, dtype=state.dtype
        )
        velocity = model(state, batch_time, y=labels)
        state = state + (following - current) * velocity
    final_index = len(times) - 1
    if final_index in selected_indices:
        selected[final_index] = state
    return selected


@torch.no_grad()
def run_probe(
    baseline_branch: Path,
    partial_branch: Path,
    *,
    device: str,
    count: int,
    batch_size: int,
    evaluation_seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
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
    selected = select_time_indices(times, DEFAULT_TARGET_TIMES)
    selected_set = set(selected)
    fields = {"baseline": baseline, "partial": partial}

    clean_energy_sum = torch.zeros(analyzer.band_count, dtype=torch.float64)
    state_energy_sums: dict[tuple[str, int], torch.Tensor] = defaultdict(
        lambda: torch.zeros(analyzer.band_count, dtype=torch.float64)
    )
    drift_sums: dict[tuple[str, str, int], torch.Tensor] = defaultdict(
        lambda: torch.zeros(analyzer.band_count, dtype=torch.float64)
    )
    microscopic_sums: dict[int, torch.Tensor] = defaultdict(
        lambda: torch.zeros(analyzer.band_count, dtype=torch.float64)
    )
    seen = 0
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        clean = clean_all[start:end].to(torch_device)
        noise = noise_all[start:end].to(torch_device)
        labels = labels_all[start:end].to(torch_device)
        clean_energy_sum += analyzer.band_mse(clean).double().sum(dim=0).cpu()
        baseline_states = selected_rollout_states(
            baseline, noise, labels, times, selected_set
        )
        partial_states = selected_rollout_states(
            partial, noise, labels, times, selected_set
        )
        target = noise - clean
        for index in selected:
            time = float(times[index])
            teacher_state = (1.0 - time) * clean + time * noise
            contexts = {
                "teacher": teacher_state,
                "baseline_rollout": baseline_states[index],
                "partial_rollout": partial_states[index],
            }
            microscopic_sums[index] += (
                2.0 * band_cross_mean(teacher_state, target, analyzer).double().sum(dim=0).cpu()
            )
            for context, state in contexts.items():
                state_energy_sums[(context, index)] += (
                    analyzer.band_mse(state).double().sum(dim=0).cpu()
                )
                batch_time = torch.full(
                    (len(state),), time, device=state.device, dtype=state.dtype
                )
                for field, model in fields.items():
                    prediction = model(state, batch_time, y=labels)
                    drift_sums[(context, field, index)] += (
                        2.0
                        * band_cross_mean(state, prediction, analyzer)
                        .double()
                        .sum(dim=0)
                        .cpu()
                    )
        seen += len(clean)
        print(f"seed {baseline_manifest['global_seed']}: {end}/{count}", flush=True)

    clean_energy = clean_energy_sum / seen
    rows = []
    for index in selected:
        time = float(times[index])
        reference_energy = (1.0 - time) ** 2 * clean_energy + time**2
        reference_drift = -2.0 * (1.0 - time) * clean_energy + 2.0 * time
        microscopic_drift = microscopic_sums[index] / seen
        for context in ("teacher", "baseline_rollout", "partial_rollout"):
            state_energy = state_energy_sums[(context, index)] / seen
            for field in ("baseline", "partial"):
                predicted_drift = drift_sums[(context, field, index)] / seen
                for band in range(analyzer.band_count):
                    reference = reference_energy[band].clamp_min(1e-12)
                    rows.append(
                        {
                            "seed": int(baseline_manifest["global_seed"]),
                            "sample_count": int(count),
                            "context": context,
                            "field": field,
                            "time_index": int(index),
                            "time": time,
                            "band": band,
                            "state_log_energy_ratio": float(
                                (state_energy[band].clamp_min(1e-12) / reference).log()
                            ),
                            "predicted_energy_drift": float(predicted_drift[band]),
                            "reference_energy_drift": float(reference_drift[band]),
                            "microscopic_teacher_energy_drift": float(
                                microscopic_drift[band]
                            ),
                            "log_energy_drift_error": float(
                                (predicted_drift[band] - reference_drift[band]) / reference
                            ),
                        }
                    )

    metadata = {
        "seed": int(baseline_manifest["global_seed"]),
        "sample_count": int(count),
        "baseline_branch": str(baseline_branch),
        "partial_branch": str(partial_branch),
        "evaluation_seed": int(evaluation_seed),
        "validation_indices": [int(v) for v in payload["validation_indices"][:count]],
        "precision": "fp32",
        "tf32": False,
        "times": [float(times[index]) for index in selected],
        "contexts": ["teacher", "baseline_rollout", "partial_rollout"],
        "scope": "shared-state radial-band second-moment drift; no decoder and no FID",
    }
    return pd.DataFrame(rows), metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--evaluation-seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    metrics, metadata = run_probe(
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
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(metrics.groupby(["context", "field", "time"]).log_energy_drift_error.apply(
        lambda values: float(((values**2).mean()) ** 0.5)
    ).to_string())
    print(output, flush=True)


if __name__ == "__main__":
    main()


__all__ = ["band_cross_mean", "run_probe", "selected_rollout_states"]
