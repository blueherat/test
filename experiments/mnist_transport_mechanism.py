"""Mechanism audit for the MNIST teacher-forcing/rollout gap.

This module tests four falsifiable explanations for the gap:

1. Euler discretization: compare converged endpoint metrics across step counts.
2. Marginal transport: measure the exact band-energy drift ``2 <z_b, v_b>``.
3. Local volume change: estimate vector-field divergence on teacher and rollout states.
4. Mediation: intervene with train-only radial-band energy calibration during sampling.

The intervention is an oracle diagnostic, not a proposed generative method.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mnist_spectral_rollout_toy import (  # noqa: E402
    MNISTToyConfig,
    TinyVelocityUNet,
    _random_directions,
    configure_fp32,
    descending_time_grid,
    frechet_distance,
    load_mnist_tensors,
    sliced_wasserstein,
    train_feature_classifier,
)
from experiments.rae_spectral_direction_loss import DCTDirectionLoss  # noqa: E402


@dataclass(frozen=True)
class MechanismConfig:
    run_dir: Path
    device: str = "cuda:0"
    audit_count: int = 256
    divergence_count: int = 16
    divergence_probes: int = 4
    sample_count: int = 1024
    batch_size: int = 128
    ode_steps: tuple[int, ...] = (25, 50, 100, 200)
    audit_ode_steps: int = 50
    switch_ode_steps: int = 200
    target_times: tuple[float, ...] = (0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05)
    save: bool = True


@dataclass
class MechanismResult:
    teacher_transport: pd.DataFrame
    rollout_transport: pd.DataFrame
    divergence: pd.DataFrame
    step_convergence: pd.DataFrame
    intervention: pd.DataFrame
    intervention_bands: pd.DataFrame
    switch_intervention: pd.DataFrame
    frequency_switch_intervention: pd.DataFrame
    result_dir: Path | None


def _toy_config_from_json(path: Path, device: str) -> MNISTToyConfig:
    values = json.loads(path.read_text(encoding="utf-8"))
    values["data_root"] = Path(values["data_root"])
    values["output_root"] = Path(values["output_root"])
    values["eval_times"] = tuple(values["eval_times"])
    values["device"] = device
    return MNISTToyConfig(**values)


def load_saved_experiment(
    run_dir: str | Path,
    device: str,
) -> tuple[
    MNISTToyConfig,
    dict[str, TinyVelocityUNet],
    DCTDirectionLoss,
    dict[str, torch.Tensor | dict[str, float]],
]:
    run_dir = Path(run_dir).expanduser().resolve()
    toy_config = _toy_config_from_json(run_dir / "config.json", device)
    torch_device = torch.device(device)
    loaded = load_mnist_tensors(
        toy_config.data_root,
        toy_config.train_size,
        toy_config.test_size,
        toy_config.seed,
    )
    for name in ("train", "test", "train_labels", "test_labels"):
        loaded[name] = loaded[name].to(torch_device)
    state = torch.load(run_dir / "state.pt", map_location=torch_device, weights_only=True)
    moments = state["second_moments"].float().cpu()
    analyzer = DCTDirectionLoss(
        28,
        moments.tolist(),
        gamma=toy_config.gamma,
        damping=1e-4,
        min_weight=0.2,
        max_weight=2.0,
    ).to(torch_device)
    models: dict[str, TinyVelocityUNet] = {}
    for name, model_state in state["models"].items():
        model = TinyVelocityUNet(toy_config.width, toy_config.depth).to(torch_device)
        model.load_state_dict(model_state)
        model.eval()
        models[name] = model
    return toy_config, models, analyzer, loaded


def band_cross_mean(
    first: torch.Tensor,
    second: torch.Tensor,
    analyzer: DCTDirectionLoss,
) -> torch.Tensor:
    """Per-sample mean DCT coefficient product in each radial band."""

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


def reference_band_energy(
    clean_energy: torch.Tensor,
    time: torch.Tensor | float,
) -> torch.Tensor:
    time_tensor = torch.as_tensor(time, device=clean_energy.device, dtype=clean_energy.dtype)
    return (1.0 - time_tensor).square()[..., None] * clean_energy + time_tensor.square()[..., None]


def reference_band_drift(
    clean_energy: torch.Tensor,
    time: torch.Tensor | float,
) -> torch.Tensor:
    time_tensor = torch.as_tensor(time, device=clean_energy.device, dtype=clean_energy.dtype)
    return -2.0 * (1.0 - time_tensor)[..., None] * clean_energy + 2.0 * time_tensor[..., None]


@torch.no_grad()
def _predict_batched(
    model: torch.nn.Module,
    state: torch.Tensor,
    time: float,
    batch_size: int,
) -> torch.Tensor:
    outputs = []
    for batch in state.split(int(batch_size)):
        times = torch.full((len(batch),), float(time), device=batch.device, dtype=batch.dtype)
        outputs.append(model(batch, times))
    return torch.cat(outputs)


@torch.no_grad()
def teacher_transport_audit(
    models: Mapping[str, torch.nn.Module],
    clean: torch.Tensor,
    analyzer: DCTDirectionLoss,
    target_times: Sequence[float],
    *,
    seed: int,
    batch_size: int,
) -> pd.DataFrame:
    generator = torch.Generator(device=clean.device).manual_seed(int(seed) + 601)
    noise = torch.randn(clean.shape, device=clean.device, generator=generator)
    clean_energy = analyzer.band_mse(clean).mean(dim=0)
    rows: list[dict[str, float | int | str]] = []
    for time in target_times:
        state = (1.0 - float(time)) * clean + float(time) * noise
        target = noise - clean
        state_energy = analyzer.band_mse(state).mean(dim=0)
        expected_energy = reference_band_energy(clean_energy, float(time))
        expected_drift = reference_band_drift(clean_energy, float(time))
        microscopic_drift = 2.0 * band_cross_mean(state, target, analyzer).mean(dim=0)
        for name, model in models.items():
            prediction = _predict_batched(model, state, float(time), batch_size)
            prediction_drift = 2.0 * band_cross_mean(state, prediction, analyzer).mean(dim=0)
            velocity_mse = analyzer.band_mse(prediction - target).mean(dim=0)
            for band in range(analyzer.band_count):
                energy = state_energy[band].clamp_min(1e-12)
                reference = expected_energy[band].clamp_min(1e-12)
                rows.append(
                    {
                        "path": "teacher",
                        "variant": name,
                        "time": float(time),
                        "band": band,
                        "state_log_energy_ratio": float((energy / reference).log()),
                        "velocity_mse": float(velocity_mse[band]),
                        "predicted_energy_drift": float(prediction_drift[band]),
                        "microscopic_energy_drift": float(microscopic_drift[band]),
                        "reference_energy_drift": float(expected_drift[band]),
                        "log_energy_drift_error": float(
                            (prediction_drift[band] - expected_drift[band]) / reference
                        ),
                    }
                )
    return pd.DataFrame(rows)


@torch.no_grad()
def rollout_states(
    model: torch.nn.Module,
    initial: torch.Tensor,
    times: torch.Tensor,
    batch_size: int,
    *,
    analyzer: DCTDirectionLoss | None = None,
    clean_moments: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    state = initial
    states = [state]
    for current, following in zip(times[:-1], times[1:]):
        velocity = _predict_batched(model, state, float(current), batch_size)
        state = state + (following - current) * velocity
        if analyzer is not None:
            if clean_moments is None:
                raise ValueError("clean moments are required for calibrated rollout")
            target = reference_band_energy(clean_moments.to(state), following)
            state = calibrate_band_energy(state, target, analyzer)
        states.append(state)
    return states


def _nearest_time_indices(times: torch.Tensor, targets: Sequence[float]) -> list[int]:
    return [int(torch.argmin((times - float(target)).abs())) for target in targets]


@torch.no_grad()
def rollout_transport_audit(
    models: Mapping[str, torch.nn.Module],
    initial: torch.Tensor,
    reference_clean: torch.Tensor,
    analyzer: DCTDirectionLoss,
    times: torch.Tensor,
    target_times: Sequence[float],
    batch_size: int,
) -> tuple[pd.DataFrame, dict[str, list[torch.Tensor]]]:
    clean_energy = analyzer.band_mse(reference_clean).mean(dim=0)
    selected = _nearest_time_indices(times, target_times)
    rows: list[dict[str, float | int | str]] = []
    trajectories: dict[str, list[torch.Tensor]] = {}
    for name, model in models.items():
        states = rollout_states(model, initial, times, batch_size)
        trajectories[name] = states
        for index in selected:
            time = float(times[index])
            state = states[index]
            velocity = _predict_batched(model, state, time, batch_size)
            energy = analyzer.band_mse(state).mean(dim=0)
            expected_energy = reference_band_energy(clean_energy, time)
            expected_drift = reference_band_drift(clean_energy, time)
            predicted_drift = 2.0 * band_cross_mean(state, velocity, analyzer).mean(dim=0)
            if index + 1 < len(states):
                following_energy = analyzer.band_mse(states[index + 1]).mean(dim=0)
                actual_drift = (following_energy - energy) / float(times[index + 1] - times[index])
            else:
                actual_drift = torch.full_like(energy, torch.nan)
            for band in range(analyzer.band_count):
                reference = expected_energy[band].clamp_min(1e-12)
                rows.append(
                    {
                        "path": "rollout",
                        "variant": name,
                        "time": time,
                        "band": band,
                        "state_log_energy_ratio": float(
                            (energy[band].clamp_min(1e-12) / reference).log()
                        ),
                        "predicted_energy_drift": float(predicted_drift[band]),
                        "actual_discrete_energy_drift": float(actual_drift[band]),
                        "reference_energy_drift": float(expected_drift[band]),
                        "log_energy_drift_error": float(
                            (predicted_drift[band] - expected_drift[band]) / reference
                        ),
                    }
                )
    return pd.DataFrame(rows), trajectories


def hutchinson_divergence(
    model: torch.nn.Module,
    state: torch.Tensor,
    time: float,
    *,
    probes: int,
    seed: int,
) -> torch.Tensor:
    """Estimate trace(dv/dz) per dimension for each sample."""

    generator = torch.Generator(device=state.device).manual_seed(int(seed))
    estimates = []
    dimension = state[0].numel()
    for _ in range(int(probes)):
        value = state.detach().requires_grad_(True)
        probe = torch.randint(
            0, 2, value.shape, device=value.device, generator=generator, dtype=torch.int64
        ).to(value.dtype)
        probe = probe * 2.0 - 1.0
        times = torch.full((len(value),), float(time), device=value.device, dtype=value.dtype)
        prediction = model(value, times)
        gradient = torch.autograd.grad((prediction * probe).sum(), value)[0]
        estimates.append((gradient * probe).flatten(1).sum(dim=1) / dimension)
    return torch.stack(estimates).mean(dim=0).detach()


def divergence_audit(
    models: Mapping[str, torch.nn.Module],
    clean: torch.Tensor,
    teacher_noise: torch.Tensor,
    trajectories: Mapping[str, list[torch.Tensor]],
    rollout_times: torch.Tensor,
    target_times: Sequence[float],
    *,
    probes: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    selected = _nearest_time_indices(rollout_times, target_times)
    for name, model in models.items():
        for time_index, (target_time, rollout_index) in enumerate(zip(target_times, selected)):
            teacher_state = (1.0 - float(target_time)) * clean + float(target_time) * teacher_noise
            rollout_state = trajectories[name][rollout_index][: len(clean)]
            for path_index, (path, state, time) in enumerate(
                (
                    ("teacher", teacher_state, float(target_time)),
                    ("rollout", rollout_state, float(rollout_times[rollout_index])),
                )
            ):
                estimate = hutchinson_divergence(
                    model,
                    state,
                    time,
                    probes=probes,
                    # Shared probes make baseline/weighted differences paired.
                    seed=seed + 100 * time_index + path_index,
                )
                rows.append(
                    {
                        "variant": name,
                        "path": path,
                        "time": time,
                        "divergence_per_dimension_mean": float(estimate.mean()),
                        "divergence_per_dimension_std": float(estimate.std(unbiased=False)),
                    }
                )
    return pd.DataFrame(rows)


@torch.no_grad()
def calibrate_band_energy(
    state: torch.Tensor,
    target_energy: torch.Tensor,
    analyzer: DCTDirectionLoss,
) -> torch.Tensor:
    """Match aggregate radial-band second moments while preserving directions."""

    if target_energy.shape != (analyzer.band_count,):
        raise ValueError(f"expected target energy shape {(analyzer.band_count,)}")
    coefficients = analyzer.transform(state)
    current = analyzer.band_mse(state).mean(dim=0).clamp_min(1e-12)
    scales = torch.sqrt(target_energy.to(current).clamp_min(1e-12) / current)
    scale_grid = scales[analyzer.band_index.to(state.device)]
    calibrated = coefficients * scale_grid[None, None]
    matrix = analyzer.dct.to(device=state.device, dtype=state.dtype)
    return torch.matmul(torch.matmul(matrix.T, calibrated), matrix)


@torch.no_grad()
def _score_generated(
    generated: torch.Tensor,
    reference: torch.Tensor,
    classifier: torch.nn.Module,
    analyzer: DCTDirectionLoss,
    normalization: Mapping[str, float],
    *,
    seed: int,
) -> tuple[dict[str, float], list[dict[str, float | int]]]:
    if len(generated) != len(reference):
        raise ValueError("generated and reference counts must match")
    mean = float(normalization["mean"])
    std = float(normalization["std"])
    generated_pixels = (generated * std + mean).clamp(0.0, 1.0)
    reference_pixels = (reference * std + mean).clamp(0.0, 1.0)
    generated_decoded = (generated_pixels - mean) / std
    reference_decoded = (reference_pixels - mean) / std
    generated_logits, generated_features = classifier(generated_decoded, return_features=True)
    _, reference_features = classifier(reference_decoded, return_features=True)
    latent_directions = _random_directions(28 * 28, 64, seed + 701, generated.device)
    pixel_directions = _random_directions(28 * 28, 64, seed + 709, generated.device)
    feature_directions = _random_directions(64, 64, seed + 719, generated.device)
    probabilities = generated_logits.softmax(dim=1)
    mean_probability = probabilities.mean(dim=0)
    generated_energy = analyzer.band_mse(generated).mean(dim=0)
    reference_energy = analyzer.band_mse(reference).mean(dim=0)
    log_ratio = (generated_energy / reference_energy.clamp_min(1e-12)).clamp_min(1e-12).log()
    summary = {
        "latent_swd": sliced_wasserstein(
            reference.flatten(1), generated.flatten(1), latent_directions
        ),
        "decoded_pixel_swd": sliced_wasserstein(
            reference_pixels.flatten(1), generated_pixels.flatten(1), pixel_directions
        ),
        "feature_swd": sliced_wasserstein(
            reference_features, generated_features, feature_directions
        ),
        "feature_fid": frechet_distance(reference_features, generated_features),
        "classifier_confidence": float(probabilities.max(dim=1).values.mean()),
        "class_entropy": float(
            -(mean_probability * mean_probability.clamp_min(1e-12).log()).sum()
        ),
        "band_log_energy_rmse": float(log_ratio.square().mean().sqrt()),
        "latent_mean": float(generated.mean()),
        "latent_std": float(generated.std(unbiased=False)),
    }
    bands = [
        {
            "band": band,
            "reference_energy": float(reference_energy[band]),
            "generated_energy": float(generated_energy[band]),
            "log_energy_ratio": float(log_ratio[band]),
        }
        for band in range(analyzer.band_count)
    ]
    return summary, bands


@torch.no_grad()
def step_convergence_audit(
    models: Mapping[str, torch.nn.Module],
    initial: torch.Tensor,
    reference: torch.Tensor,
    classifier: torch.nn.Module,
    analyzer: DCTDirectionLoss,
    normalization: Mapping[str, float],
    step_counts: Sequence[int],
    *,
    time_shift: float,
    batch_size: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for steps in step_counts:
        times = descending_time_grid(int(steps), time_shift, device=initial.device)
        for name, model in models.items():
            generated = rollout_states(model, initial, times, batch_size)[-1]
            summary, _ = _score_generated(
                generated, reference, classifier, analyzer, normalization, seed=seed
            )
            rows.append({"variant": name, "ode_steps": int(steps), **summary})
    return pd.DataFrame(rows)


@torch.no_grad()
def calibration_intervention_audit(
    models: Mapping[str, torch.nn.Module],
    initial: torch.Tensor,
    reference: torch.Tensor,
    classifier: torch.nn.Module,
    analyzer: DCTDirectionLoss,
    train_moments: torch.Tensor,
    normalization: Mapping[str, float],
    *,
    steps: int,
    time_shift: float,
    batch_size: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    times = descending_time_grid(int(steps), time_shift, device=initial.device)
    summary_rows = []
    band_rows = []
    for name, model in models.items():
        raw = rollout_states(model, initial, times, batch_size)[-1]
        path_calibrated = rollout_states(
            model,
            initial,
            times,
            batch_size,
            analyzer=analyzer,
            clean_moments=train_moments,
        )[-1]
        candidates = {f"{name}_raw": raw, f"{name}_path_calibrated": path_calibrated}
        if name == "weighted":
            candidates["weighted_endpoint_calibrated"] = calibrate_band_energy(
                raw, train_moments.to(raw), analyzer
            )
        for intervention, generated in candidates.items():
            summary, bands = _score_generated(
                generated, reference, classifier, analyzer, normalization, seed=seed
            )
            summary_rows.append({"intervention": intervention, **summary})
            band_rows.extend({"intervention": intervention, **row} for row in bands)
    return pd.DataFrame(summary_rows), pd.DataFrame(band_rows)


@torch.no_grad()
def hybrid_rollout_state(
    baseline: torch.nn.Module,
    weighted: torch.nn.Module,
    initial: torch.Tensor,
    times: torch.Tensor,
    batch_size: int,
    use_weighted,
) -> torch.Tensor:
    state = initial
    for current, following in zip(times[:-1], times[1:]):
        model = weighted if bool(use_weighted(float(current))) else baseline
        velocity = _predict_batched(model, state, float(current), batch_size)
        state = state + (following - current) * velocity
    return state


@torch.no_grad()
def switch_intervention_audit(
    models: Mapping[str, torch.nn.Module],
    initial: torch.Tensor,
    reference: torch.Tensor,
    classifier: torch.nn.Module,
    analyzer: DCTDirectionLoss,
    normalization: Mapping[str, float],
    *,
    steps: int,
    time_shift: float,
    batch_size: int,
    seed: int,
) -> pd.DataFrame:
    """Causally localize treatment damage with hard time-window field switches."""

    baseline = models["baseline"]
    weighted = models["weighted"]
    times = descending_time_grid(int(steps), time_shift, device=initial.device)
    selectors = {
        "baseline_all": lambda time: False,
        "weighted_all": lambda time: True,
        "weighted_high_only": lambda time: time >= 0.7,
        "weighted_mid_only": lambda time: 0.3 <= time < 0.7,
        "weighted_low_only": lambda time: time < 0.3,
        "weighted_without_low": lambda time: time >= 0.3,
        "weighted_without_high": lambda time: time < 0.7,
    }
    rows = []
    for intervention, selector in selectors.items():
        generated = hybrid_rollout_state(
            baseline,
            weighted,
            initial,
            times,
            batch_size,
            selector,
        )
        summary, _ = _score_generated(
            generated, reference, classifier, analyzer, normalization, seed=seed
        )
        rows.append({"intervention": intervention, "ode_steps": int(steps), **summary})
    return pd.DataFrame(rows)


@torch.no_grad()
def blend_velocity_bands(
    baseline_velocity: torch.Tensor,
    weighted_velocity: torch.Tensor,
    weighted_bands: Sequence[int],
    analyzer: DCTDirectionLoss,
) -> torch.Tensor:
    """Use weighted velocity only in selected radial DCT output bands."""

    if baseline_velocity.shape != weighted_velocity.shape:
        raise ValueError("velocity fields must have equal shapes")
    selected = torch.zeros(analyzer.band_count, device=baseline_velocity.device, dtype=torch.bool)
    for band in weighted_bands:
        if not 0 <= int(band) < analyzer.band_count:
            raise ValueError(f"invalid band {band}")
        selected[int(band)] = True
    mask = selected[analyzer.band_index.to(baseline_velocity.device)]
    baseline_coefficients = analyzer.transform(baseline_velocity)
    weighted_coefficients = analyzer.transform(weighted_velocity)
    blended = torch.where(
        mask[None, None], weighted_coefficients, baseline_coefficients
    )
    matrix = analyzer.dct.to(device=blended.device, dtype=blended.dtype)
    return torch.matmul(torch.matmul(matrix.T, blended), matrix)


@torch.no_grad()
def frequency_switch_intervention_audit(
    models: Mapping[str, torch.nn.Module],
    initial: torch.Tensor,
    reference: torch.Tensor,
    classifier: torch.nn.Module,
    analyzer: DCTDirectionLoss,
    normalization: Mapping[str, float],
    *,
    steps: int,
    time_shift: float,
    batch_size: int,
    seed: int,
    high_time_threshold: float = 0.7,
) -> pd.DataFrame:
    """Causally localize high-noise damage to coarse or fine output bands."""

    baseline = models["baseline"]
    weighted = models["weighted"]
    times = descending_time_grid(int(steps), time_shift, device=initial.device)
    band_sets = {
        "baseline_all": (),
        "weighted_high_all_bands": tuple(range(analyzer.band_count)),
        "weighted_high_band0_only": (0,),
        "weighted_high_bands0_1_only": (0, 1),
        "weighted_high_bands1_7_only": tuple(range(1, analyzer.band_count)),
        "weighted_high_bands2_7_only": tuple(range(2, analyzer.band_count)),
        "weighted_high_band7_only": (analyzer.band_count - 1,),
    }
    rows = []
    for intervention, selected_bands in band_sets.items():
        state = initial
        for current, following in zip(times[:-1], times[1:]):
            baseline_velocity = _predict_batched(baseline, state, float(current), batch_size)
            if float(current) >= float(high_time_threshold) and selected_bands:
                weighted_velocity = _predict_batched(weighted, state, float(current), batch_size)
                velocity = blend_velocity_bands(
                    baseline_velocity,
                    weighted_velocity,
                    selected_bands,
                    analyzer,
                )
            else:
                velocity = baseline_velocity
            state = state + (following - current) * velocity
        summary, _ = _score_generated(
            state, reference, classifier, analyzer, normalization, seed=seed
        )
        rows.append(
            {
                "intervention": intervention,
                "ode_steps": int(steps),
                "high_time_threshold": float(high_time_threshold),
                "weighted_bands": ",".join(str(band) for band in selected_bands),
                **summary,
            }
        )
    return pd.DataFrame(rows)


def _save_tables(result: MechanismResult, config: MechanismConfig) -> Path:
    result_dir = config.run_dir.expanduser().resolve() / "mechanism_v1"
    result_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "teacher_transport.csv": result.teacher_transport,
        "rollout_transport.csv": result.rollout_transport,
        "divergence.csv": result.divergence,
        "step_convergence.csv": result.step_convergence,
        "intervention.csv": result.intervention,
        "intervention_bands.csv": result.intervention_bands,
        "switch_intervention.csv": result.switch_intervention,
        "frequency_switch_intervention.csv": result.frequency_switch_intervention,
    }
    for filename, frame in tables.items():
        frame.to_csv(result_dir / filename, index=False)
    metadata = {
        "run_dir": str(config.run_dir.expanduser().resolve()),
        "audit_count": config.audit_count,
        "divergence_count": config.divergence_count,
        "divergence_probes": config.divergence_probes,
        "sample_count": config.sample_count,
        "ode_steps": list(config.ode_steps),
        "audit_ode_steps": config.audit_ode_steps,
        "switch_ode_steps": config.switch_ode_steps,
        "target_times": list(config.target_times),
        "contracts": {
            "calibration_statistics": "train split only",
            "evaluation_reference": "official disjoint MNIST test split",
            "paired_initial_noise": True,
            "calibration_role": "oracle diagnostic, not proposed method",
        },
    }
    (result_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return result_dir


def run_mechanism_study(config: MechanismConfig) -> MechanismResult:
    toy_config, models, analyzer, loaded = load_saved_experiment(config.run_dir, config.device)
    configure_fp32(toy_config.seed + 809)
    train = loaded["train"]
    test = loaded["test"]
    train_labels = loaded["train_labels"]
    test_labels = loaded["test_labels"]
    normalization = loaded["normalization"]
    audit_clean = test[: min(config.audit_count, len(test))]
    sample_count = min(config.sample_count, len(test))
    reference = test[:sample_count]
    generator = torch.Generator(device=test.device).manual_seed(toy_config.seed + 811)
    initial = torch.randn(reference.shape, device=test.device, generator=generator)
    train_moments = analyzer.band_mse(train).mean(dim=0)

    teacher = teacher_transport_audit(
        models,
        audit_clean,
        analyzer,
        config.target_times,
        seed=toy_config.seed,
        batch_size=config.batch_size,
    )
    audit_times = descending_time_grid(
        config.audit_ode_steps, toy_config.time_shift, device=test.device
    )
    rollout, trajectories = rollout_transport_audit(
        models,
        initial[: len(audit_clean)],
        audit_clean,
        analyzer,
        audit_times,
        config.target_times,
        config.batch_size,
    )
    divergence_count = min(config.divergence_count, len(audit_clean))
    divergence_clean = audit_clean[:divergence_count]
    divergence_generator = torch.Generator(device=test.device).manual_seed(toy_config.seed + 821)
    teacher_noise = torch.randn(
        divergence_clean.shape, device=test.device, generator=divergence_generator
    )
    divergence = divergence_audit(
        models,
        divergence_clean,
        teacher_noise,
        trajectories,
        audit_times,
        config.target_times,
        probes=config.divergence_probes,
        seed=toy_config.seed + 823,
    )
    classifier, classifier_accuracy = train_feature_classifier(
        train,
        train_labels,
        test,
        test_labels,
        epochs=toy_config.classifier_epochs,
        batch_size=toy_config.classifier_batch_size,
        seed=toy_config.seed,
    )
    step_convergence = step_convergence_audit(
        models,
        initial,
        reference,
        classifier,
        analyzer,
        normalization,
        config.ode_steps,
        time_shift=toy_config.time_shift,
        batch_size=config.batch_size,
        seed=toy_config.seed,
    )
    step_convergence["classifier_accuracy"] = classifier_accuracy
    intervention, intervention_bands = calibration_intervention_audit(
        models,
        initial,
        reference,
        classifier,
        analyzer,
        train_moments,
        normalization,
        steps=config.audit_ode_steps,
        time_shift=toy_config.time_shift,
        batch_size=config.batch_size,
        seed=toy_config.seed,
    )
    intervention["classifier_accuracy"] = classifier_accuracy
    switch_intervention = switch_intervention_audit(
        models,
        initial,
        reference,
        classifier,
        analyzer,
        normalization,
        steps=config.switch_ode_steps,
        time_shift=toy_config.time_shift,
        batch_size=config.batch_size,
        seed=toy_config.seed,
    )
    switch_intervention["classifier_accuracy"] = classifier_accuracy
    frequency_switch_intervention = frequency_switch_intervention_audit(
        models,
        initial,
        reference,
        classifier,
        analyzer,
        normalization,
        steps=config.switch_ode_steps,
        time_shift=toy_config.time_shift,
        batch_size=config.batch_size,
        seed=toy_config.seed,
    )
    frequency_switch_intervention["classifier_accuracy"] = classifier_accuracy
    result = MechanismResult(
        teacher_transport=teacher,
        rollout_transport=rollout,
        divergence=divergence,
        step_convergence=step_convergence,
        intervention=intervention,
        intervention_bands=intervention_bands,
        switch_intervention=switch_intervention,
        frequency_switch_intervention=frequency_switch_intervention,
        result_dir=None,
    )
    if config.save:
        result.result_dir = _save_tables(result, config)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--audit-count", type=int, default=256)
    parser.add_argument("--divergence-count", type=int, default=16)
    parser.add_argument("--divergence-probes", type=int, default=4)
    parser.add_argument("--sample-count", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_mechanism_study(
        MechanismConfig(
            run_dir=args.run_dir,
            device=args.device,
            audit_count=args.audit_count,
            divergence_count=args.divergence_count,
            divergence_probes=args.divergence_probes,
            sample_count=args.sample_count,
            batch_size=args.batch_size,
            save=not args.no_save,
        )
    )
    print("\nstep convergence")
    print(result.step_convergence.to_string(index=False))
    print("\ncalibration intervention")
    print(result.intervention.to_string(index=False))
    print("\ndivergence")
    print(result.divergence.to_string(index=False))
    print("\ntime-switch intervention")
    print(result.switch_intervention.to_string(index=False))
    print("\nhigh-noise frequency-switch intervention")
    print(result.frequency_switch_intervention.to_string(index=False))
    if result.result_dir is not None:
        print(f"\nsaved to: {result.result_dir}")


if __name__ == "__main__":
    main()
