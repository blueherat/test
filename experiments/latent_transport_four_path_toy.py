"""Paired four-path causal toy for nonlinear latent coordinates.

All transformed branches are inverted back to the same analytic 2D mixture
before endpoint evaluation.  The primary question is whether a strict
pushforward path recovers a degradation caused by Gaussian-straight training.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.dinov2_token_diagnostics import configure_fp32  # noqa: E402
from experiments.latent_transport_paths import (  # noqa: E402
    conditional_path_sample,
    jvp_relative_error,
    relative_l2_per_sample,
)


BRANCHES = ("base", "gaussian_straight", "matched_chord", "pushforward")


@dataclass(frozen=True)
class RingMixtureConfig:
    modes: int = 8
    radius: float = 2.0
    component_std: float = 0.15


@dataclass(frozen=True)
class FourPathToyConfig:
    output_root: Path = Path.home() / "data/eqvae/experiments/latent_transport_four_path_toy"
    strengths: tuple[float, ...] = (0.5, 1.0)
    primary_strength: float = 1.0
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    devices: tuple[str, ...] = ("cuda:0", "cuda:1", "cuda:2", "cuda:3")
    hidden_size: int = 64
    depth: int = 3
    batch_size: int = 512
    steps: int = 4000
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    eval_every: int = 500
    eval_count: int = 8192
    sample_count: int = 8192
    sliced_directions: int = 256
    ode_steps: int = 100
    solver_steps: int = 200
    solver_count: int = 2048
    grad_clip: float = 5.0
    save: bool = True


class QuadraticShear(nn.Module):
    """Triangular coupling f(x1,x2)=(x1,x2+a*x1^2)."""

    is_linear = False

    def __init__(self, strength: float):
        super().__init__()
        self.strength = float(strength)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] != 2:
            raise ValueError("QuadraticShear expects a final dimension of two")
        first, second = value.unbind(dim=-1)
        return torch.stack(
            (first, second + self.strength * first.square()),
            dim=-1,
        )

    def inverse(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] != 2:
            raise ValueError("QuadraticShear expects a final dimension of two")
        first, second = value.unbind(dim=-1)
        return torch.stack(
            (first, second - self.strength * first.square()),
            dim=-1,
        )


class ResidualTimeMLP(nn.Module):
    def __init__(self, hidden_size: int, depth: int):
        super().__init__()
        hidden_size = int(hidden_size)
        self.input = nn.Linear(6, hidden_size)
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size * 2),
                nn.SiLU(),
                nn.Linear(hidden_size * 2, hidden_size),
            )
            for _ in range(int(depth))
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 2),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    @staticmethod
    def features(state: torch.Tensor, time_value: torch.Tensor) -> torch.Tensor:
        if time_value.ndim == 1:
            time_value = time_value[:, None]
        phase = 2.0 * math.pi * time_value
        return torch.cat(
            (state, time_value, time_value.square(), torch.sin(phase), torch.cos(phase)),
            dim=1,
        )

    def forward(self, state: torch.Tensor, time_value: torch.Tensor) -> torch.Tensor:
        hidden = self.input(self.features(state, time_value))
        for block in self.blocks:
            hidden = hidden + block(hidden) / math.sqrt(len(self.blocks))
        return self.output(hidden)


class ConjugatedVelocityField(nn.Module):
    """Exact coordinate pushforward of a field defined in base coordinates."""

    def __init__(self, base_field: nn.Module, transform: QuadraticShear):
        super().__init__()
        self.base_field = base_field
        self.transform = transform

    def forward(self, transformed_state: torch.Tensor, time_value: torch.Tensor) -> torch.Tensor:
        base_state = self.transform.inverse(transformed_state)
        base_velocity = self.base_field(base_state, time_value)
        _, transformed_velocity = torch.func.jvp(
            self.transform,
            (base_state,),
            (base_velocity,),
        )
        return transformed_velocity


def configure_seed(seed: int) -> None:
    configure_fp32()
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def mixture_centers(config: RingMixtureConfig, device: torch.device | str = "cpu") -> torch.Tensor:
    angle = 2.0 * math.pi * torch.arange(config.modes, device=device) / config.modes
    return config.radius * torch.stack((torch.cos(angle), torch.sin(angle)), dim=1)


def sample_ring_mixture(
    config: RingMixtureConfig,
    count: int,
    *,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = torch.randint(
        0,
        config.modes,
        (int(count),),
        generator=generator,
        device=device,
    )
    centers = mixture_centers(config, device)
    noise = torch.randn(
        (int(count), 2),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    return centers[labels] + config.component_std * noise, labels


def exact_mixture_nll(value: torch.Tensor, config: RingMixtureConfig) -> float:
    value = value.double()
    centers = mixture_centers(config).double()
    variance = float(config.component_std) ** 2
    log_component = (
        -0.5 * (value[:, None] - centers[None]).square().sum(dim=2) / variance
        - math.log(2.0 * math.pi * variance)
        - math.log(config.modes)
    )
    return float(-torch.logsumexp(log_component, dim=1).mean())


def sliced_wasserstein_1(
    generated: torch.Tensor,
    reference: torch.Tensor,
    *,
    directions: int,
    seed: int,
) -> float:
    if generated.shape != reference.shape:
        raise ValueError("generated and reference must have equal shapes")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    vectors = torch.randn((2, int(directions)), generator=generator, dtype=torch.float64)
    vectors = F.normalize(vectors, dim=0)
    left = torch.sort(generated.double().cpu() @ vectors, dim=0).values
    right = torch.sort(reference.double().cpu() @ vectors, dim=0).values
    return float((left - right).abs().mean())


def distribution_metrics(
    generated: torch.Tensor,
    reference: torch.Tensor,
    config: RingMixtureConfig,
    *,
    directions: int,
    seed: int,
) -> dict[str, float | int]:
    generated = generated.float().cpu()
    reference = reference.float().cpu()
    generated_mean = generated.mean(dim=0)
    reference_mean = reference.mean(dim=0)
    generated_centered = generated - generated_mean
    reference_centered = reference - reference_mean
    generated_covariance = generated_centered.T @ generated_centered / (len(generated) - 1)
    reference_covariance = reference_centered.T @ reference_centered / (len(reference) - 1)
    centers = mixture_centers(config)
    assignment = torch.cdist(generated, centers).argmin(dim=1)
    proportions = torch.bincount(assignment, minlength=config.modes).float() / len(generated)
    uniform = torch.full_like(proportions, 1.0 / config.modes)
    coordinate_w1 = torch.mean(
        torch.abs(
            torch.sort(generated, dim=0).values
            - torch.sort(reference, dim=0).values
        )
    )
    return {
        "sliced_w1": sliced_wasserstein_1(
            generated,
            reference,
            directions=directions,
            seed=seed,
        ),
        "coordinate_w1": float(coordinate_w1),
        "exact_mixture_nll": exact_mixture_nll(generated, config),
        "mean_l2": float(torch.linalg.vector_norm(generated_mean - reference_mean)),
        "covariance_relative_frobenius": float(
            torch.linalg.matrix_norm(generated_covariance - reference_covariance)
            / torch.linalg.matrix_norm(reference_covariance).clamp_min(1e-12)
        ),
        "mode_coverage": int((proportions >= 0.01).sum()),
        "mode_total_variation": float(0.5 * torch.abs(proportions - uniform).sum()),
    }


@torch.no_grad()
def sample_model(
    model: nn.Module,
    branch: str,
    transform: QuadraticShear,
    epsilon: torch.Tensor,
    *,
    ode_steps: int,
) -> torch.Tensor:
    if branch not in BRANCHES:
        raise ValueError(f"unknown branch: {branch}")
    model.eval()
    state = epsilon if branch in {"base", "gaussian_straight"} else transform(epsilon)
    times = torch.linspace(1.0, 0.0, int(ode_steps) + 1, device=epsilon.device)
    for current, following in zip(times[:-1], times[1:]):
        current_batch = torch.full((len(state),), float(current), device=state.device)
        first = model(state, current_batch)
        proposal = state + (following - current) * first
        following_batch = torch.full((len(state),), float(following), device=state.device)
        second = model(proposal, following_batch)
        state = state + 0.5 * (following - current) * (first + second)
    return state if branch == "base" else transform.inverse(state)


@torch.no_grad()
def heldout_teacher_metrics(
    models: Mapping[str, nn.Module],
    transform: QuadraticShear,
    mixture: RingMixtureConfig,
    config: FourPathToyConfig,
    *,
    seed: int,
    device: torch.device,
) -> list[dict[str, float | str]]:
    generator = torch.Generator(device=device).manual_seed(300_001 + int(seed))
    data, _ = sample_ring_mixture(
        mixture,
        config.eval_count,
        generator=generator,
        device=device,
    )
    epsilon = torch.randn(
        data.shape,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    time_value = torch.rand(
        (len(data),),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    rows = []
    for branch, model in models.items():
        path = conditional_path_sample(
            data,
            epsilon,
            time_value,
            branch=branch,
            transform=transform,
        )
        prediction = model(path.state, time_value)
        error = prediction - path.velocity
        rows.append(
            {
                "branch": branch,
                "microscopic_velocity_mse": float(error.square().mean()),
                "relative_velocity_l2": float(
                    relative_l2_per_sample(prediction, path.velocity).mean()
                ),
                "target_velocity_rms": float(path.velocity.square().mean().sqrt()),
            }
        )
    return rows


def train_one(
    mixture: RingMixtureConfig,
    config: FourPathToyConfig,
    *,
    strength: float,
    seed: int,
    device_name: str,
) -> dict[str, object]:
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    configure_seed(seed)
    transform = QuadraticShear(strength).to(device)
    template = ResidualTimeMLP(config.hidden_size, config.depth).to(device)
    models = {branch: copy.deepcopy(template).to(device) for branch in BRANCHES}
    initial_state = copy.deepcopy(template.state_dict())
    initial_max_gap = max(
        float((models[branch].state_dict()[key] - value).abs().max())
        for branch in BRANCHES
        for key, value in initial_state.items()
    )
    optimizers = {
        branch: torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        for branch, model in models.items()
    }
    schedulers = {
        branch: torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.steps,
            eta_min=config.learning_rate * 0.05,
        )
        for branch, optimizer in optimizers.items()
    }
    generator = torch.Generator(device=device).manual_seed(100_003 + int(seed))
    history = []
    for step in range(1, config.steps + 1):
        data, _ = sample_ring_mixture(
            mixture,
            config.batch_size,
            generator=generator,
            device=device,
        )
        epsilon = torch.randn(
            data.shape,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        time_value = torch.rand(
            (len(data),),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        for branch, model in models.items():
            path = conditional_path_sample(
                data,
                epsilon,
                time_value,
                branch=branch,
                transform=transform,
            )
            prediction = model(path.state, time_value)
            loss = (prediction - path.velocity).square().mean()
            optimizer = optimizers[branch]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            schedulers[branch].step()
            if step == 1 or step % config.eval_every == 0 or step == config.steps:
                history.append(
                    {
                        "step": int(step),
                        "branch": branch,
                        "train_loss": float(loss.detach()),
                        "learning_rate": float(schedulers[branch].get_last_lr()[0]),
                    }
                )

    teacher_rows = heldout_teacher_metrics(
        models,
        transform,
        mixture,
        config,
        seed=seed,
        device=device,
    )
    sample_generator = torch.Generator(device=device).manual_seed(400_009 + int(seed))
    epsilon = torch.randn(
        (config.sample_count, 2),
        generator=sample_generator,
        device=device,
        dtype=torch.float32,
    )
    reference, _ = sample_ring_mixture(
        mixture,
        config.sample_count,
        generator=sample_generator,
        device=device,
    )
    second_reference, _ = sample_ring_mixture(
        mixture,
        config.sample_count,
        generator=sample_generator,
        device=device,
    )
    generated = {
        branch: sample_model(
            model,
            branch,
            transform,
            epsilon,
            ode_steps=config.ode_steps,
        ).cpu()
        for branch, model in models.items()
    }
    endpoint_rows = []
    for branch, samples in generated.items():
        endpoint_rows.append(
            {
                "branch": branch,
                **distribution_metrics(
                    samples,
                    reference.cpu(),
                    mixture,
                    directions=config.sliced_directions,
                    seed=500_009 + int(seed),
                ),
            }
        )
    reference_floor = distribution_metrics(
        second_reference.cpu(),
        reference.cpu(),
        mixture,
        directions=config.sliced_directions,
        seed=500_009 + int(seed),
    )

    solver_epsilon = epsilon[: config.solver_count]
    solver_rows = []
    for branch, model in models.items():
        coarse = sample_model(
            model,
            branch,
            transform,
            solver_epsilon,
            ode_steps=config.ode_steps,
        )
        fine = sample_model(
            model,
            branch,
            transform,
            solver_epsilon,
            ode_steps=config.solver_steps,
        )
        solver_rows.append(
            {
                "branch": branch,
                "endpoint_relative_l2": float(relative_l2_per_sample(coarse, fine).mean()),
                "endpoint_max_l2": float(torch.linalg.vector_norm(coarse - fine, dim=1).max()),
            }
        )

    cycle_generator = torch.Generator(device=device).manual_seed(600_011 + int(seed))
    cycle_value = torch.randn((4096, 2), generator=cycle_generator, device=device)
    cycle_max = float(relative_l2_per_sample(transform.inverse(transform(cycle_value)), cycle_value).max())
    jvp_error = float(
        jvp_relative_error(
            transform,
            cycle_value[:256],
            torch.randn(
                (256, 2), generator=cycle_generator, device=device, dtype=torch.float32
            ),
            # Central differences are exact for this quadratic map in exact
            # arithmetic; a larger step suppresses fp32 cancellation.
            step=1e-1,
        ).max()
    )
    states = {
        branch: {key: value.detach().cpu() for key, value in model.state_dict().items()}
        for branch, model in models.items()
    }
    for model in models.values():
        model.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "strength": float(strength),
        "seed": int(seed),
        "device": device_name,
        "initial_max_parameter_gap": initial_max_gap,
        "cycle_relative_l2_max": cycle_max,
        "jvp_finite_difference_relative_l2_max": jvp_error,
        "reference_floor": reference_floor,
        "history": history,
        "teacher": teacher_rows,
        "endpoint": endpoint_rows,
        "solver": solver_rows,
        "samples": generated if seed == min(config.seeds) else {},
        "reference": reference.cpu() if seed == min(config.seeds) else None,
        "states": states,
    }


def _run_group(tasks: Sequence[tuple]) -> list[dict[str, object]]:
    completed = []
    for mixture, config, strength, seed, device in tasks:
        result = train_one(
            mixture,
            config,
            strength=strength,
            seed=seed,
            device_name=device,
        )
        endpoint = {row["branch"]: row["sliced_w1"] for row in result["endpoint"]}
        print(
            f"a={strength:g} seed={seed}: "
            + ", ".join(f"{key}={value:.4f}" for key, value in endpoint.items()),
            flush=True,
        )
        completed.append(result)
    return completed


def _pair_summary(endpoint: pd.DataFrame, primary_strength: float) -> pd.DataFrame:
    primary = endpoint[endpoint.strength.eq(float(primary_strength))]
    rows = []
    for seed, frame in primary.groupby("seed"):
        values = frame.set_index("branch")["sliced_w1"]
        base = float(values["base"])
        gaussian = float(values["gaussian_straight"])
        matched = float(values["matched_chord"])
        pushforward = float(values["pushforward"])
        denominator = gaussian - base
        recovery = (gaussian - pushforward) / denominator if denominator > 0 else float("nan")
        rows.append(
            {
                "seed": int(seed),
                "base_sliced_w1": base,
                "gaussian_sliced_w1": gaussian,
                "matched_sliced_w1": matched,
                "pushforward_sliced_w1": pushforward,
                "gaussian_over_base": gaussian / max(base, 1e-12),
                "matched_recovery": (gaussian - matched) / denominator
                if denominator > 0
                else float("nan"),
                "pushforward_recovery": recovery,
                "has_coordinate_gap": gaussian >= 1.10 * base,
                "pushforward_recovers_half": bool(
                    gaussian >= 1.10 * base and recovery >= 0.50
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("seed")


def acceptance_report(
    endpoint: pd.DataFrame,
    solver: pd.DataFrame,
    run_rows: pd.DataFrame,
    pair_summary: pd.DataFrame,
    config: FourPathToyConfig,
) -> dict[str, object]:
    primary_endpoint = endpoint[endpoint.strength.eq(config.primary_strength)]
    base = primary_endpoint[primary_endpoint.branch.eq("base")]
    base_valid_seeds = int(((base.mode_coverage == 8) & (base.sliced_w1 < 0.20)).sum())
    solver_max = float(solver.endpoint_relative_l2.max())
    gap_seeds = int(pair_summary.has_coordinate_gap.sum())
    recovery_seeds = int(pair_summary.pushforward_recovers_half.sum())
    push_ratio = float(
        primary_endpoint[primary_endpoint.branch.eq("pushforward")].sliced_w1.mean()
        / base.sliced_w1.mean()
    )
    checks = {
        "paired_initialization": {
            "value": float(run_rows.initial_max_parameter_gap.max()),
            "threshold": 0.0,
            "passed": float(run_rows.initial_max_parameter_gap.max()) == 0.0,
        },
        "transform_cycle": {
            "value": float(run_rows.cycle_relative_l2_max.max()),
            "threshold": 1e-6,
            "passed": float(run_rows.cycle_relative_l2_max.max()) <= 1e-6,
        },
        "jvp_finite_difference": {
            "value": float(run_rows.jvp_finite_difference_relative_l2_max.max()),
            "threshold": 1e-4,
            "passed": float(run_rows.jvp_finite_difference_relative_l2_max.max()) <= 1e-4,
        },
        "solver_stability": {
            "value": solver_max,
            "threshold": 0.02,
            "passed": solver_max < 0.02,
        },
        "base_quality_seed_count": {
            "value": base_valid_seeds,
            "threshold": 4,
            "passed": base_valid_seeds >= 4,
        },
        "coordinate_gap_seed_count": {
            "value": gap_seeds,
            "threshold": 4,
            "passed": gap_seeds >= 4,
        },
        "pushforward_half_recovery_seed_count": {
            "value": recovery_seeds,
            "threshold": 4,
            "passed": recovery_seeds >= 4,
        },
        "pushforward_mean_over_base": {
            "value": push_ratio,
            "threshold": 1.10,
            "passed": push_ratio <= 1.10,
        },
    }
    validity_names = (
        "paired_initialization",
        "transform_cycle",
        "jvp_finite_difference",
        "solver_stability",
        "base_quality_seed_count",
    )
    validity = all(checks[name]["passed"] for name in validity_names)
    if not validity:
        decision = "invalid_or_underfit_toy_fix_before_interpretation"
    elif gap_seeds < 4:
        decision = "no_stable_coordinate_gap_stop_h4_h5"
    elif recovery_seeds < 4 or push_ratio > 1.10:
        decision = "pushforward_does_not_recover_stop_h4_h5"
    else:
        decision = "phase3a_pass_small_image_authorized"
    return {
        "checks": checks,
        "validity_passed": validity,
        "method_passed": decision == "phase3a_pass_small_image_authorized",
        "decision": decision,
    }


def _plot_samples(results: Sequence[dict[str, object]], config: FourPathToyConfig, path: Path) -> None:
    selected = [result for result in results if result["samples"]]
    figure, axes = plt.subplots(
        len(selected),
        5,
        figsize=(16, 3.2 * len(selected)),
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, result in enumerate(sorted(selected, key=lambda item: item["strength"])):
        panels = [("reference", result["reference"])] + [
            (branch, result["samples"][branch]) for branch in BRANCHES
        ]
        for axis, (name, values) in zip(axes[row_index], panels):
            values = values[:3000]
            axis.scatter(values[:, 0], values[:, 1], s=2, alpha=0.35)
            axis.set_xlim(-3.5, 3.5)
            axis.set_ylim(-3.5, 3.5)
            axis.set_aspect("equal")
            axis.set_title(f"a={result['strength']:g} {name}")
            axis.grid(alpha=0.15)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_study(
    config: FourPathToyConfig = FourPathToyConfig(),
    mixture: RingMixtureConfig = RingMixtureConfig(),
) -> tuple[dict[str, object], Path | None]:
    if config.primary_strength not in config.strengths:
        raise ValueError("primary_strength must appear in strengths")
    tasks = []
    pairs = [(strength, seed) for strength in config.strengths for seed in config.seeds]
    devices = config.devices or ("cpu",)
    for index, (strength, seed) in enumerate(pairs):
        tasks.append((mixture, config, strength, seed, devices[index % len(devices)]))
    groups = [[] for _ in devices]
    for index, task in enumerate(tasks):
        groups[index % len(devices)].append(task)
    results = []
    if len(devices) == 1:
        results = _run_group(groups[0])
    else:
        context = get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(devices), mp_context=context) as executor:
            futures = [executor.submit(_run_group, group) for group in groups if group]
            for future in as_completed(futures):
                results.extend(future.result())
    results.sort(key=lambda item: (item["strength"], item["seed"]))

    run_rows = pd.DataFrame(
        [
            {
                "strength": result["strength"],
                "seed": result["seed"],
                "device": result["device"],
                "initial_max_parameter_gap": result["initial_max_parameter_gap"],
                "cycle_relative_l2_max": result["cycle_relative_l2_max"],
                "jvp_finite_difference_relative_l2_max": result[
                    "jvp_finite_difference_relative_l2_max"
                ],
                **{
                    f"reference_floor_{key}": value
                    for key, value in result["reference_floor"].items()
                },
            }
            for result in results
        ]
    )
    history = pd.DataFrame(
        [
            {"strength": result["strength"], "seed": result["seed"], **row}
            for result in results
            for row in result["history"]
        ]
    )
    teacher = pd.DataFrame(
        [
            {"strength": result["strength"], "seed": result["seed"], **row}
            for result in results
            for row in result["teacher"]
        ]
    )
    endpoint = pd.DataFrame(
        [
            {"strength": result["strength"], "seed": result["seed"], **row}
            for result in results
            for row in result["endpoint"]
        ]
    )
    solver = pd.DataFrame(
        [
            {"strength": result["strength"], "seed": result["seed"], **row}
            for result in results
            for row in result["solver"]
        ]
    )
    pair = _pair_summary(endpoint, config.primary_strength)
    acceptance = acceptance_report(endpoint, solver, run_rows, pair, config)

    result_dir = None
    if config.save:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = config.output_root.expanduser() / f"preregistered_v1_{stamp}"
        result_dir.mkdir(parents=True, exist_ok=False)
        serialized = asdict(config)
        serialized["output_root"] = str(config.output_root.expanduser())
        (result_dir / "config.json").write_text(
            json.dumps(serialized, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (result_dir / "mixture.json").write_text(
            json.dumps(asdict(mixture), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for name, frame in (
            ("run_metrics", run_rows),
            ("training_history", history),
            ("teacher_metrics", teacher),
            ("endpoint_metrics", endpoint),
            ("solver_metrics", solver),
            ("pair_summary", pair),
        ):
            frame.to_csv(result_dir / f"{name}.csv", index=False)
        (result_dir / "acceptance.json").write_text(
            json.dumps(acceptance, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        torch.save(
            {
                f"a{result['strength']}_seed{result['seed']}": result["states"]
                for result in results
            },
            result_dir / "models.pt",
        )
        _plot_samples(results, config, result_dir / "endpoint_samples.png")
        summary = {
            "protocol_version": 1,
            "timestamp": datetime.now().astimezone().isoformat(),
            "result_dir": str(result_dir),
            "config": serialized,
            "mixture": asdict(mixture),
            "acceptance": acceptance,
        }
        (result_dir / "result.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return {
        "run_metrics": run_rows,
        "training_history": history,
        "teacher_metrics": teacher,
        "endpoint_metrics": endpoint,
        "solver_metrics": solver,
        "pair_summary": pair,
        "acceptance": acceptance,
    }, result_dir


def parse_args() -> tuple[FourPathToyConfig, RingMixtureConfig]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    devices = tuple(value.strip() for value in args.devices.split(",") if value.strip())
    config = FourPathToyConfig(devices=devices or ("cpu",), save=not args.no_save)
    if args.quick:
        config = replace(
            config,
            strengths=(1.0,),
            seeds=(0,),
            devices=(devices[0],) if devices else ("cpu",),
            batch_size=32,
            steps=4,
            eval_every=2,
            eval_count=64,
            sample_count=64,
            sliced_directions=16,
            ode_steps=4,
            solver_steps=8,
            solver_count=32,
        )
    return config, RingMixtureConfig()


if __name__ == "__main__":
    started = time.time()
    parsed_config, parsed_mixture = parse_args()
    outputs, directory = run_study(parsed_config, parsed_mixture)
    print(json.dumps(outputs["acceptance"], ensure_ascii=False, indent=2), flush=True)
    print(f"result_dir={directory} elapsed={time.time() - started:.1f}s", flush=True)
