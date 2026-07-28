"""Estimate signed endpoint leverage before retraining a weighted flow.

For a frozen baseline checkpoint, the probe compares the local update induced by
the proposed weighted objective with the ordinary MSE update. It then measures
their difference against a train-only differentiable-rollout moment objective.
No weighted checkpoint or test endpoint metric is used to construct the score.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mnist_spectral_rollout_toy import (  # noqa: E402
    configure_fp32,
    descending_time_grid,
    shifted_uniform,
)
from experiments.small_image_basis_mechanism import (  # noqa: E402
    _load_run,
    _load_study_config,
)
from experiments.small_image_basis_transport import (  # noqa: E402
    dct_pixel_basis,
    load_small_image_tensors,
    radial_band_index,
)


@dataclass(frozen=True)
class SignedLeverageConfig:
    study_dir: Path
    output_root: Path = Path.home() / "data/eqvae/experiments/small_image_signed_leverage"
    bases: tuple[str, ...] = ("dct", "pca", "random")
    training_seeds: tuple[int, ...] = (3, 4)
    probe_seeds: tuple[int, ...] = (2901, 2902, 2903)
    devices: tuple[str, ...] = ("cuda:0", "cuda:1", "cuda:2", "cuda:3")
    update_count: int = 128
    endpoint_count: int = 32
    ode_steps: int = 30
    band_count: int = 8
    finite_difference_relative_step: float = 1e-4
    save: bool = True


def differentiable_euler_sample(
    model: torch.nn.Module,
    initial: torch.Tensor,
    *,
    ode_steps: int,
) -> torch.Tensor:
    if int(ode_steps) < 1:
        raise ValueError("ode_steps must be positive")
    state = initial
    times = descending_time_grid(int(ode_steps), device=initial.device)
    for current, following in zip(times[:-1], times[1:]):
        time = torch.full(
            (len(state),), float(current), device=state.device, dtype=state.dtype
        )
        state = state + (following - current) * model(state, time)
    return state


def canonical_band_energies(
    images: torch.Tensor,
    basis: torch.Tensor,
    group_index: torch.Tensor,
    band_count: int,
) -> torch.Tensor:
    coefficients = images.flatten(1) @ basis.to(images.device, images.dtype)
    component_energy = coefficients.square().mean(dim=0)
    sums = torch.zeros(int(band_count), device=images.device, dtype=images.dtype)
    counts = torch.zeros_like(sums)
    groups = group_index.to(images.device)
    sums.scatter_add_(0, groups, component_energy)
    counts.scatter_add_(0, groups, torch.ones_like(component_energy))
    return sums / counts.clamp_min(1.0)


def endpoint_moment_loss(
    generated: torch.Tensor,
    reference_energy: torch.Tensor,
    basis: torch.Tensor,
    group_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    generated_energy = canonical_band_energies(
        generated,
        basis,
        group_index,
        len(reference_energy),
    )
    log_gap = (
        generated_energy.clamp_min(1e-12).log()
        - reference_energy.to(generated.device).clamp_min(1e-12).log()
    )
    return log_gap.square().mean(), log_gap


def _flatten(values: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([value.reshape(-1) for value in values])


def clip_gradient_tuple(
    gradients: Sequence[torch.Tensor], max_norm: float = 1.0
) -> tuple[torch.Tensor, ...]:
    total_norm = torch.sqrt(sum(gradient.square().sum() for gradient in gradients))
    scale = min(1.0, float(max_norm) / max(float(total_norm), 1e-12))
    return tuple(gradient * scale for gradient in gradients)


def block_normalized_update(
    gradients: Sequence[torch.Tensor], epsilon: float = 1e-8
) -> tuple[torch.Tensor, ...]:
    return tuple(
        -gradient / gradient.square().mean().sqrt().clamp_min(float(epsilon))
        for gradient in gradients
    )


def candidate_update_difference(
    baseline_gradients: Sequence[torch.Tensor],
    weighted_gradients: Sequence[torch.Tensor],
    *,
    mode: str,
) -> tuple[torch.Tensor, ...]:
    if len(baseline_gradients) != len(weighted_gradients):
        raise ValueError("gradient tuples must have equal lengths")
    if mode == "raw":
        baseline = tuple(-value for value in clip_gradient_tuple(baseline_gradients))
        weighted = tuple(-value for value in clip_gradient_tuple(weighted_gradients))
    elif mode == "block":
        baseline = block_normalized_update(baseline_gradients)
        weighted = block_normalized_update(weighted_gradients)
    else:
        raise ValueError(f"unknown update mode: {mode}")
    return tuple(
        weighted_value - baseline_value
        for baseline_value, weighted_value in zip(baseline, weighted)
    )


def update_alignment(
    endpoint_gradients: Sequence[torch.Tensor],
    baseline_gradients: Sequence[torch.Tensor],
    weighted_gradients: Sequence[torch.Tensor],
) -> dict[str, float]:
    if not (len(endpoint_gradients) == len(baseline_gradients) == len(weighted_gradients)):
        raise ValueError("gradient tuples must have equal lengths")
    raw_difference = candidate_update_difference(
        baseline_gradients, weighted_gradients, mode="raw"
    )
    block_difference = candidate_update_difference(
        baseline_gradients, weighted_gradients, mode="block"
    )
    endpoint = _flatten(endpoint_gradients)

    def metrics(prefix: str, difference: Sequence[torch.Tensor]) -> dict[str, float]:
        vector = _flatten(difference)
        endpoint_norm = torch.linalg.vector_norm(endpoint)
        vector_norm = torch.linalg.vector_norm(vector)
        dot = torch.dot(endpoint, vector)
        cosine = dot / (endpoint_norm * vector_norm).clamp_min(1e-20)
        return {
            f"{prefix}_directional_derivative": float(dot),
            f"{prefix}_cosine": float(cosine),
            f"{prefix}_update_difference_norm": float(vector_norm),
        }

    return {
        **metrics("raw", raw_difference),
        **metrics("block", block_difference),
        "endpoint_gradient_norm": float(torch.linalg.vector_norm(endpoint)),
    }


def finite_difference_directional_derivative(
    model: torch.nn.Module,
    direction: Sequence[torch.Tensor],
    objective,
    *,
    relative_step: float,
) -> float:
    if float(relative_step) <= 0:
        raise ValueError("relative_step must be positive")
    parameters = tuple(model.parameters())
    if len(parameters) != len(direction):
        raise ValueError("direction must match model parameters")
    parameter_norm = torch.sqrt(
        sum(parameter.detach().square().sum() for parameter in parameters)
    )
    direction_norm = torch.sqrt(sum(value.square().sum() for value in direction))
    alpha = float(relative_step) * float(parameter_norm) / max(float(direction_norm), 1e-12)
    with torch.no_grad():
        for parameter, value in zip(parameters, direction):
            parameter.add_(value, alpha=alpha)
        plus = float(objective())
        for parameter, value in zip(parameters, direction):
            parameter.add_(value, alpha=-2.0 * alpha)
        minus = float(objective())
        for parameter, value in zip(parameters, direction):
            parameter.add_(value, alpha=alpha)
    return (plus - minus) / (2.0 * alpha)


def _training_gradients(
    model: torch.nn.Module,
    analyzer: torch.nn.Module,
    clean: torch.Tensor,
    *,
    seed: int,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    generator = torch.Generator(device=clean.device).manual_seed(int(seed))
    noise = torch.randn(clean.shape, generator=generator, device=clean.device)
    time = shifted_uniform(len(clean), 1.0, device=clean.device, generator=generator)
    expanded = time[:, None, None, None]
    state = (1.0 - expanded) * clean + expanded * noise
    target = noise - clean
    prediction = model(state, time)
    baseline_loss = F.mse_loss(prediction, target)
    weighted_loss = analyzer(prediction, target, time)[0].mean()
    parameters = tuple(model.parameters())
    baseline = torch.autograd.grad(baseline_loss, parameters, retain_graph=True)
    weighted = torch.autograd.grad(weighted_loss, parameters)
    return (
        tuple(value.detach() for value in baseline),
        tuple(value.detach() for value in weighted),
    )


def _endpoint_gradients(
    model: torch.nn.Module,
    reference: torch.Tensor,
    *,
    seed: int,
    ode_steps: int,
    basis: torch.Tensor,
    group_index: torch.Tensor,
    band_count: int,
) -> tuple[
    tuple[torch.Tensor, ...],
    float,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    reference_energy = canonical_band_energies(
        reference.detach(), basis, group_index, int(band_count)
    ).detach()
    generator = torch.Generator(device=reference.device).manual_seed(int(seed))
    initial = torch.randn(reference.shape, generator=generator, device=reference.device)
    generated = differentiable_euler_sample(model, initial, ode_steps=int(ode_steps))
    loss, log_gap = endpoint_moment_loss(
        generated, reference_energy, basis, group_index
    )
    gradients = torch.autograd.grad(loss, tuple(model.parameters()))
    return (
        tuple(value.detach() for value in gradients),
        float(loss.detach()),
        log_gap.detach(),
        reference_energy,
        initial,
    )


def _run_training_seed(
    config: SignedLeverageConfig,
    training_seed: int,
    device_name: str,
) -> pd.DataFrame:
    study_dir = config.study_dir.expanduser().resolve()
    study_config = _load_study_config(study_dir)
    configure_fp32(int(training_seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device(
        device_name if torch.cuda.is_available() or not device_name.startswith("cuda") else "cpu"
    )
    required = int(config.update_count) + int(config.endpoint_count)
    loaded = load_small_image_tensors(
        study_config.dataset,
        study_config.data_root,
        max(study_config.train_size, required),
        study_config.test_size,
        int(training_seed),
        download=False,
    )
    train = loaded["train"].to(device)
    update_clean = train[: int(config.update_count)]
    endpoint_reference = train[
        int(config.update_count) : int(config.update_count) + int(config.endpoint_count)
    ]
    if len(endpoint_reference) != int(config.endpoint_count):
        raise ValueError("not enough disjoint training images for endpoint probe")
    size = int(train.shape[-1])
    canonical_basis = dct_pixel_basis(size).to(device)
    canonical_groups = radial_band_index(size, int(config.band_count)).flatten().to(device)
    target_rows = pd.read_csv(study_dir / "study_summary.csv")
    rows: list[dict[str, float | int | str]] = []

    for basis_name in config.bases:
        run_dir = study_dir / f"{basis_name}_seed{training_seed}"
        models, analyzer, _ = _load_run(run_dir, study_config, device)
        model = models["baseline"]
        baseline_gradients, weighted_gradients = _training_gradients(
            model,
            analyzer,
            update_clean,
            seed=int(training_seed) + 31_001,
        )
        target = target_rows[
            target_rows["basis"].eq(basis_name)
            & target_rows["seed"].eq(int(training_seed))
        ]
        if len(target) != 1:
            raise ValueError(f"missing endpoint target for {basis_name} seed {training_seed}")
        endpoint_ratio = float(target.iloc[0]["rollout_feature_fid_ratio"])
        for probe_seed in config.probe_seeds:
            (
                endpoint_gradients,
                endpoint_loss,
                log_gap,
                reference_energy,
                initial,
            ) = _endpoint_gradients(
                model,
                endpoint_reference,
                seed=int(probe_seed),
                ode_steps=int(config.ode_steps),
                basis=canonical_basis,
                group_index=canonical_groups,
                band_count=int(config.band_count),
            )
            raw_difference = candidate_update_difference(
                baseline_gradients, weighted_gradients, mode="raw"
            )

            def objective() -> float:
                generated = differentiable_euler_sample(
                    model, initial, ode_steps=int(config.ode_steps)
                )
                loss, _ = endpoint_moment_loss(
                    generated,
                    reference_energy,
                    canonical_basis,
                    canonical_groups,
                )
                return float(loss)

            raw_finite_difference = finite_difference_directional_derivative(
                model,
                raw_difference,
                objective,
                relative_step=float(config.finite_difference_relative_step),
            )
            alignment = update_alignment(
                endpoint_gradients, baseline_gradients, weighted_gradients
            )
            rows.append(
                {
                    "dataset": study_config.dataset,
                    "basis": basis_name,
                    "training_seed": int(training_seed),
                    "probe_seed": int(probe_seed),
                    "endpoint_moment_loss": endpoint_loss,
                    "endpoint_band0_log_gap": float(log_gap[0]),
                    **alignment,
                    "raw_finite_difference_derivative": raw_finite_difference,
                    "raw_finite_difference_sign_match": (
                        raw_finite_difference
                        * alignment["raw_directional_derivative"]
                        > 0
                    ),
                    # Joined only after every prospective feature above is computed.
                    "observed_endpoint_fid_ratio": endpoint_ratio,
                    "observed_endpoint_log_damage": math.log(max(endpoint_ratio, 1e-12)),
                }
            )
        for loaded_model in models.values():
            loaded_model.cpu()
        analyzer.cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def summarize_signed_leverage(metrics: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [
        "endpoint_moment_loss",
        "endpoint_band0_log_gap",
        "raw_directional_derivative",
        "raw_cosine",
        "block_directional_derivative",
        "block_cosine",
        "raw_finite_difference_derivative",
    ]
    grouped = metrics.groupby(
        ["dataset", "basis", "training_seed"], as_index=False
    ).agg(
        probe_seeds=("probe_seed", "nunique"),
        **{column: (column, "mean") for column in feature_columns},
        observed_endpoint_fid_ratio=("observed_endpoint_fid_ratio", "first"),
        observed_endpoint_log_damage=("observed_endpoint_log_damage", "first"),
    )
    grouped["raw_sign_correct"] = grouped["raw_directional_derivative"].gt(0).eq(
        grouped["observed_endpoint_log_damage"].gt(0)
    )
    grouped["block_sign_correct"] = grouped["block_directional_derivative"].gt(0).eq(
        grouped["observed_endpoint_log_damage"].gt(0)
    )
    return grouped.sort_values(["dataset", "training_seed", "basis"]).reset_index(drop=True)


def run_signed_leverage_study(
    config: SignedLeverageConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, Path | None]:
    devices = config.devices or ("cpu",)
    tasks = [
        (config, int(seed), devices[index % len(devices)])
        for index, seed in enumerate(config.training_seeds)
    ]
    if len(tasks) == 1:
        frames = [_run_training_seed(*tasks[0])]
    else:
        context = get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(len(tasks), len(devices)), mp_context=context
        ) as executor:
            futures = [executor.submit(_run_training_seed, *task) for task in tasks]
            frames = [future.result() for future in as_completed(futures)]
    metrics = pd.concat(frames, ignore_index=True).sort_values(
        ["training_seed", "basis", "probe_seed"]
    )
    expected = len(config.training_seeds) * len(config.bases) * len(config.probe_seeds)
    if len(metrics) != expected:
        raise RuntimeError(f"expected {expected} rows, received {len(metrics)}")
    summary = summarize_signed_leverage(metrics)
    result_dir = None
    if config.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = config.output_root.expanduser() / f"pilot_{timestamp}"
        result_dir.mkdir(parents=True, exist_ok=False)
        serialized = asdict(config)
        for key in ("study_dir", "output_root"):
            serialized[key] = str(serialized[key])
        (result_dir / "config.json").write_text(
            json.dumps(serialized, indent=2), encoding="utf-8"
        )
        metrics.to_csv(result_dir / "metrics.csv", index=False)
        summary.to_csv(result_dir / "summary.csv", index=False)
    return metrics, summary, result_dir


def _integers(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())


def _strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--bases", default="dct,pca,random")
    parser.add_argument("--training-seeds", default="3,4")
    parser.add_argument("--probe-seeds", default="2901,2902,2903")
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3")
    parser.add_argument("--update-count", type=int, default=128)
    parser.add_argument("--endpoint-count", type=int, default=32)
    parser.add_argument("--ode-steps", type=int, default=30)
    parser.add_argument("--band-count", type=int, default=8)
    parser.add_argument("--finite-difference-relative-step", type=float, default=1e-4)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    config = SignedLeverageConfig(
        study_dir=args.study_dir,
        output_root=args.output_root or SignedLeverageConfig.output_root,
        bases=_strings(args.bases),
        training_seeds=_integers(args.training_seeds),
        probe_seeds=_integers(args.probe_seeds),
        devices=_strings(args.devices) or ("cpu",),
        update_count=args.update_count,
        endpoint_count=args.endpoint_count,
        ode_steps=args.ode_steps,
        band_count=args.band_count,
        finite_difference_relative_step=args.finite_difference_relative_step,
        save=not args.no_save,
    )
    _, summary, result_dir = run_signed_leverage_study(config)
    print(summary.round(5).to_string(index=False))
    print(f"result_dir={result_dir}")


if __name__ == "__main__":
    main()
