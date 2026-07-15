"""Nonlinear stochastic flow-matching toy with analytic conditional velocity.

Each latent coordinate follows an independent symmetric two-Gaussian mixture.
The linear FM path, best linear skip, total residual covariance, and nonlinear
conditional velocity are all available in closed form.  Small neural models
can therefore be trained on microscopic velocity targets and evaluated against
the noise-free population optimum.
"""

from __future__ import annotations

import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from multiprocessing import get_context
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


DEFAULT_VARIANCE = (0.05, 0.10, 0.20, 0.50, 1.0, 2.0, 5.0, 10.0)
DEFAULT_BIMODAL_FRACTION = (0.0, 0.0, 0.20, 0.80, 0.95, 0.99, 0.999, 0.9999)
DEFAULT_DECODER_GAIN = (8.0, 8.0, 4.0, 2.0, 1.0, 1.0, 1.0, 1.0)


@dataclass(frozen=True)
class MixtureFMConfig:
    variance: Tuple[float, ...] = DEFAULT_VARIANCE
    bimodal_fraction: Tuple[float, ...] = DEFAULT_BIMODAL_FRACTION
    decoder_gain: Tuple[float, ...] = DEFAULT_DECODER_GAIN
    t_min: float = 0.0
    t_max: float = 1.0
    input_whiten: bool = True

    @property
    def dimension(self) -> int:
        return len(self.variance)


@dataclass(frozen=True)
class NeuralTrainConfig:
    architecture: str = "mlp"
    gamma: float = 0.5
    batch_size: int = 128
    steps: int = 800
    learning_rate: float = 2e-3
    min_learning_rate_ratio: float = 0.05
    weight_decay: float = 1e-4
    hidden_size: int = 96
    depth: int = 3
    num_heads: int = 4
    eval_every: int = 40
    eval_count: int = 4096
    gradient_clip: float = 1.0
    damping: float = 1e-4
    seed: int = 0
    device: str = "cuda:0"


@dataclass
class NeuralRun:
    config: NeuralTrainConfig
    problem: MixtureFMConfig
    model: nn.Module
    history: pd.DataFrame
    summary: Dict[str, float]


def _validate_problem(problem: MixtureFMConfig) -> None:
    variance = np.asarray(problem.variance, dtype=np.float64)
    fraction = np.asarray(problem.bimodal_fraction, dtype=np.float64)
    gain = np.asarray(problem.decoder_gain, dtype=np.float64)
    if not (variance.ndim == fraction.ndim == gain.ndim == 1):
        raise ValueError("problem vectors must be one-dimensional")
    if not (len(variance) == len(fraction) == len(gain) and len(variance) > 0):
        raise ValueError("variance, bimodal_fraction, and decoder_gain must match")
    if np.any(variance <= 0.0):
        raise ValueError("variance must be strictly positive")
    if np.any((fraction < 0.0) | (fraction >= 1.0)):
        raise ValueError("bimodal_fraction must lie in [0, 1)")
    if np.any(gain <= 0.0):
        raise ValueError("decoder_gain must be strictly positive")
    if not 0.0 <= problem.t_min < problem.t_max <= 1.0:
        raise ValueError("require 0 <= t_min < t_max <= 1")


def problem_tensors(
    problem: MixtureFMConfig,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    _validate_problem(problem)
    variance = torch.tensor(problem.variance, device=device, dtype=dtype)
    fraction = torch.tensor(problem.bimodal_fraction, device=device, dtype=dtype)
    mean = torch.sqrt(variance * fraction)
    component_variance = variance * (1.0 - fraction)
    component_std = torch.sqrt(component_variance)
    decoder_gain = torch.tensor(problem.decoder_gain, device=device, dtype=dtype)
    return {
        "variance": variance,
        "mean": mean,
        "component_variance": component_variance,
        "component_std": component_std,
        "decoder_gain": decoder_gain,
    }


def configure_torch(seed: int = 0) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def resolve_device(device: str) -> torch.device:
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return requested


def fm_statistics(
    x: torch.Tensor,
    t: torch.Tensor,
    problem: MixtureFMConfig,
) -> Dict[str, torch.Tensor]:
    """Compute exact linear skip and conditional velocity for the mixture path."""

    tensors = problem_tensors(problem, x.device, x.dtype)
    variance = tensors["variance"].view(1, -1)
    mixture_mean = tensors["mean"].view(1, -1)
    component_variance = tensors["component_variance"].view(1, -1)
    if t.ndim == 1:
        t = t[:, None]
    a = 1.0 - t
    b = t

    input_variance = a.square() * variance + b.square()
    linear_skip = (b - a * variance) / input_variance
    residual_variance = variance / input_variance

    component_input_variance = a.square() * component_variance + b.square()
    component_input_variance = component_input_variance.clamp_min(1e-8)
    sign_mean = torch.tanh(a * mixture_mean * x / component_input_variance)
    conditional_velocity = (
        (b - a * component_variance) * x / component_input_variance
        - b * mixture_mean * sign_mean / component_input_variance
    )
    conditional_residual = conditional_velocity - linear_skip * x
    network_input = x / torch.sqrt(input_variance) if problem.input_whiten else x
    return {
        "input_variance": input_variance,
        "linear_skip": linear_skip,
        "residual_variance": residual_variance,
        "conditional_velocity": conditional_velocity,
        "conditional_residual": conditional_residual,
        "network_input": network_input,
    }


def sample_fm_batch(
    problem: MixtureFMConfig,
    batch_size: int,
    device: torch.device | str,
    generator: torch.Generator,
) -> Dict[str, torch.Tensor]:
    """Sample paired data/noise endpoints and the microscopic FM target."""

    device = torch.device(device)
    tensors = problem_tensors(problem, device)
    shape = (int(batch_size), problem.dimension)
    signs = torch.randint(0, 2, shape, device=device, generator=generator).float() * 2.0 - 1.0
    latent = (
        signs * tensors["mean"]
        + torch.randn(shape, device=device, generator=generator) * tensors["component_std"]
    )
    noise = torch.randn(shape, device=device, generator=generator)
    t = (
        problem.t_min
        + (problem.t_max - problem.t_min)
        * torch.rand((int(batch_size), 1), device=device, generator=generator)
    )
    a = 1.0 - t
    b = t
    x = a * latent + b * noise
    velocity = noise - latent
    statistics = fm_statistics(x, t, problem)
    residual_target = velocity - statistics["linear_skip"] * x
    return {
        "latent": latent,
        "noise": noise,
        "t": t,
        "x": x,
        "velocity": velocity,
        "residual_target": residual_target,
        **statistics,
    }


def residual_weight_normalizer(
    problem: MixtureFMConfig,
    gamma: float,
    damping: float,
    *,
    grid_size: int = 4097,
) -> float:
    """Normalize the expected scalar loss weight to one over time/directions."""

    if not 0.0 <= float(gamma) <= 1.0:
        raise ValueError("gamma must lie in [0, 1]")
    if damping < 0.0:
        raise ValueError("damping must be non-negative")
    t = torch.linspace(problem.t_min, problem.t_max, int(grid_size), dtype=torch.float64)[:, None]
    variance = torch.tensor(problem.variance, dtype=torch.float64)[None]
    input_variance = (1.0 - t).square() * variance + t.square()
    residual_variance = variance / input_variance
    return float(torch.mean(torch.pow(residual_variance + damping, -float(gamma))))


def weighted_residual_loss(
    prediction: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    *,
    gamma: float,
    damping: float,
    normalizer: float,
) -> torch.Tensor:
    weight = torch.pow(batch["residual_variance"] + damping, -float(gamma)) / normalizer
    return torch.mean(weight * torch.square(prediction - batch["residual_target"]))


def time_features(t: torch.Tensor, frequency_count: int = 8) -> torch.Tensor:
    if t.ndim == 2:
        t = t[:, 0]
    frequencies = torch.pow(
        t.new_tensor(2.0), torch.arange(frequency_count, device=t.device, dtype=t.dtype)
    )
    angles = 2.0 * math.pi * t[:, None] * frequencies[None]
    return torch.cat([t[:, None], torch.sin(angles), torch.cos(angles)], dim=1)


class ResidualMLP(nn.Module):
    def __init__(self, dimension: int, hidden_size: int = 96, depth: int = 3):
        super().__init__()
        time_dim = 17
        layers = [nn.Linear(3 * dimension + time_dim, hidden_size), nn.SiLU()]
        for _ in range(max(1, int(depth)) - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.SiLU()])
        self.backbone = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_size, dimension)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # The best linear skip makes the learnable residual orthogonal to x.
        # Explicit low-order nonlinear features prevent a random-feature
        # warm-up from dominating this mechanism experiment.
        features = torch.cat([x, x.square(), x.pow(3), time_features(t)], dim=1)
        return self.output(self.backbone(features))


class MiniDiT(nn.Module):
    """Small shared token transformer with time and direction embeddings."""

    def __init__(
        self,
        dimension: int,
        hidden_size: int = 96,
        depth: int = 3,
        num_heads: int = 4,
    ):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.dimension = int(dimension)
        self.input = nn.Linear(1, hidden_size)
        self.position = nn.Parameter(torch.zeros(1, dimension, hidden_size))
        self.time = nn.Sequential(
            nn.Linear(17, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=num_heads,
                    dim_feedforward=4 * hidden_size,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(int(depth))
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.output = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.position, std=0.02)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        tokens = self.input(x[:, :, None]) + self.position + self.time(time_features(t))[:, None]
        for block in self.blocks:
            tokens = block(tokens)
        return self.output(self.norm(tokens)).squeeze(-1)


def build_model(problem: MixtureFMConfig, config: NeuralTrainConfig) -> nn.Module:
    architecture = config.architecture.strip().lower()
    if architecture == "mlp":
        return ResidualMLP(problem.dimension, config.hidden_size, config.depth)
    if architecture in {"mini_dit", "dit", "transformer"}:
        return MiniDiT(problem.dimension, config.hidden_size, config.depth, config.num_heads)
    raise ValueError(f"unknown architecture: {config.architecture}")


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    batch: Mapping[str, torch.Tensor],
    problem: MixtureFMConfig,
) -> Dict[str, object]:
    model.eval()
    prediction = model(batch["network_input"], batch["t"])
    error = prediction - batch["conditional_residual"]
    target_error = prediction - batch["residual_target"]
    gain = problem_tensors(problem, prediction.device)["decoder_gain"][None]
    direction_mse = torch.mean(error.square(), dim=0)
    return {
        "excess_mse": float(torch.mean(error.square()).item()),
        "decoder_weighted_mse": float(
            (torch.sum(error.square() * gain.square()) / torch.sum(gain.square()) / len(error)).item()
        ),
        "target_mse": float(torch.mean(target_error.square()).item()),
        "mean_abs_prediction": float(torch.mean(torch.abs(prediction)).item()),
        "direction_mse": direction_mse.detach().cpu().numpy(),
    }


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(int(seed))


def train_neural_model(
    problem: MixtureFMConfig,
    config: NeuralTrainConfig,
    *,
    verbose: bool = False,
) -> NeuralRun:
    """Train one paired neural run and retain the final model on CPU."""

    _validate_problem(problem)
    if not 0.0 <= config.gamma <= 1.0:
        raise ValueError("gamma must lie in [0, 1]")
    if config.steps < 1 or config.batch_size < 1 or config.eval_count < 1:
        raise ValueError("steps, batch_size, and eval_count must be positive")
    device = resolve_device(config.device)
    configure_torch(config.seed)
    model = build_model(problem, config).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.steps,
        eta_min=config.learning_rate * config.min_learning_rate_ratio,
    )
    normalizer = residual_weight_normalizer(problem, config.gamma, config.damping)
    train_generator = _make_generator(device, 100_003 + config.seed)
    eval_generator = _make_generator(device, 200_003 + config.seed)
    eval_batch = sample_fm_batch(problem, config.eval_count, device, eval_generator)

    rows = []
    start_time = time.perf_counter()

    def record(step: int, train_loss: float, gradient_norm: float) -> None:
        metrics = evaluate_model(model, eval_batch, problem)
        row = {
            "step": int(step),
            "train_loss": float(train_loss),
            "gradient_norm": float(gradient_norm),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "architecture": config.architecture,
            "gamma": float(config.gamma),
            "batch_size": int(config.batch_size),
            "seed": int(config.seed),
            "excess_mse": metrics["excess_mse"],
            "decoder_weighted_mse": metrics["decoder_weighted_mse"],
            "target_mse": metrics["target_mse"],
            "mean_abs_prediction": metrics["mean_abs_prediction"],
        }
        for index, value in enumerate(metrics["direction_mse"]):
            row[f"direction_mse_{index}"] = float(value)
        rows.append(row)

    record(0, float("nan"), float("nan"))
    last_loss = float("nan")
    last_gradient_norm = float("nan")
    for step in range(1, config.steps + 1):
        model.train()
        batch = sample_fm_batch(problem, config.batch_size, device, train_generator)
        prediction = model(batch["network_input"], batch["t"])
        loss = weighted_residual_loss(
            prediction,
            batch,
            gamma=config.gamma,
            damping=config.damping,
            normalizer=normalizer,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if config.gradient_clip > 0.0:
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            last_gradient_norm = float(gradient_norm.item())
        else:
            square_sum = sum(
                float(parameter.grad.detach().square().sum().item())
                for parameter in model.parameters()
                if parameter.grad is not None
            )
            last_gradient_norm = math.sqrt(square_sum)
        optimizer.step()
        scheduler.step()
        last_loss = float(loss.item())
        if step % config.eval_every == 0 or step == config.steps:
            record(step, last_loss, last_gradient_norm)
            if verbose:
                print(
                    f"{config.architecture} gamma={config.gamma:g} B={config.batch_size} "
                    f"seed={config.seed} step={step}/{config.steps} "
                    f"excess={rows[-1]['excess_mse']:.5g}"
                )

    history = pd.DataFrame(rows)
    final = history.iloc[-1]
    summary = {
        "architecture": config.architecture,
        "gamma": float(config.gamma),
        "batch_size": int(config.batch_size),
        "seed": int(config.seed),
        "steps": int(config.steps),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "final_excess_mse": float(final["excess_mse"]),
        "best_excess_mse": float(history["excess_mse"].min()),
        "final_decoder_weighted_mse": float(final["decoder_weighted_mse"]),
        "best_decoder_weighted_mse": float(history["decoder_weighted_mse"].min()),
        "final_target_mse": float(final["target_mse"]),
        "runtime_seconds": float(time.perf_counter() - start_time),
        "weight_normalizer": float(normalizer),
    }
    model = model.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return NeuralRun(config=config, problem=problem, model=model, history=history, summary=summary)


def run_training_grid(
    problem: MixtureFMConfig,
    base_config: NeuralTrainConfig,
    *,
    architectures: Iterable[str],
    gammas: Iterable[float],
    batch_sizes: Iterable[int],
    seeds: Iterable[int],
    verbose: bool = True,
) -> tuple[Dict[tuple, NeuralRun], pd.DataFrame, pd.DataFrame]:
    runs: Dict[tuple, NeuralRun] = {}
    histories = []
    summaries = []
    for architecture in architectures:
        for batch_size in batch_sizes:
            for gamma in gammas:
                for seed in seeds:
                    config = replace(
                        base_config,
                        architecture=str(architecture),
                        gamma=float(gamma),
                        batch_size=int(batch_size),
                        seed=int(seed),
                    )
                    run = train_neural_model(problem, config, verbose=False)
                    key = (str(architecture), int(batch_size), float(gamma), int(seed))
                    runs[key] = run
                    histories.append(run.history)
                    summaries.append(run.summary)
                    if verbose:
                        print(
                            f"done {architecture} B={batch_size} gamma={gamma:g} seed={seed}: "
                            f"excess={run.summary['final_excess_mse']:.5g}, "
                            f"{run.summary['runtime_seconds']:.1f}s"
                        )
    return runs, pd.concat(histories, ignore_index=True), pd.DataFrame(summaries)


def _train_device_group(
    problem: MixtureFMConfig,
    tasks: Sequence[tuple],
) -> list[tuple]:
    completed = []
    for key, config in tasks:
        run = train_neural_model(problem, config, verbose=False)
        completed.append((key, run))
    return completed


def run_training_grid_parallel(
    problem: MixtureFMConfig,
    base_config: NeuralTrainConfig,
    *,
    architectures: Iterable[str],
    gammas: Iterable[float],
    batch_sizes: Iterable[int],
    seeds: Iterable[int],
    devices: Sequence[str],
    verbose: bool = True,
) -> tuple[Dict[tuple, NeuralRun], pd.DataFrame, pd.DataFrame]:
    """Run independent configurations concurrently, with one process per GPU."""

    devices = tuple(str(device) for device in devices)
    if len(devices) <= 1:
        config = replace(base_config, device=devices[0] if devices else base_config.device)
        return run_training_grid(
            problem,
            config,
            architectures=architectures,
            gammas=gammas,
            batch_sizes=batch_sizes,
            seeds=seeds,
            verbose=verbose,
        )

    task_rows = []
    for architecture in architectures:
        for batch_size in batch_sizes:
            for gamma in gammas:
                for seed in seeds:
                    key = (str(architecture), int(batch_size), float(gamma), int(seed))
                    task_rows.append((key, architecture, batch_size, gamma, seed))
    grouped_tasks = [[] for _ in devices]
    for index, (key, architecture, batch_size, gamma, seed) in enumerate(task_rows):
        device = devices[index % len(devices)]
        config = replace(
            base_config,
            architecture=str(architecture),
            gamma=float(gamma),
            batch_size=int(batch_size),
            seed=int(seed),
            device=device,
        )
        grouped_tasks[index % len(devices)].append((key, config))

    runs: Dict[tuple, NeuralRun] = {}
    context = get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(devices), mp_context=context) as executor:
        futures = [
            executor.submit(_train_device_group, problem, group)
            for group in grouped_tasks
            if group
        ]
        for future in as_completed(futures):
            for key, run in future.result():
                runs[key] = run
                if verbose:
                    print(
                        f"done {key[0]} B={key[1]} gamma={key[2]:g} seed={key[3]}: "
                        f"excess={run.summary['final_excess_mse']:.5g}, "
                        f"{run.summary['runtime_seconds']:.1f}s"
                    )
    ordered_runs = {key: runs[key] for key, *_ in task_rows}
    histories = pd.concat([run.history for run in ordered_runs.values()], ignore_index=True)
    summaries = pd.DataFrame([run.summary for run in ordered_runs.values()])
    return ordered_runs, histories, summaries


@torch.no_grad()
def velocity_field(
    model: Optional[nn.Module],
    x: torch.Tensor,
    t: torch.Tensor,
    problem: MixtureFMConfig,
    *,
    oracle: bool = False,
) -> torch.Tensor:
    statistics = fm_statistics(x, t, problem)
    if oracle:
        return statistics["conditional_velocity"]
    if model is None:
        raise ValueError("model is required unless oracle=True")
    residual = model(statistics["network_input"], t)
    return statistics["linear_skip"] * x + residual


@torch.no_grad()
def reverse_ode_samples(
    problem: MixtureFMConfig,
    *,
    model: Optional[nn.Module] = None,
    sample_count: int = 4096,
    ode_steps: int = 80,
    seed: int = 0,
    device: str = "cuda:0",
    oracle: bool = False,
) -> torch.Tensor:
    """Integrate the probability-flow ODE from the noise endpoint to data."""

    actual_device = resolve_device(device)
    generator = _make_generator(actual_device, 700_001 + int(seed))
    x = torch.randn(
        (int(sample_count), problem.dimension),
        generator=generator,
        device=actual_device,
    )
    if model is not None:
        model = model.to(actual_device).eval()
    times = torch.linspace(1.0, 0.0, int(ode_steps) + 1, device=actual_device)
    for index in range(int(ode_steps)):
        current = times[index]
        following = times[index + 1]
        dt = following - current
        t_current = torch.full((len(x), 1), current, device=actual_device)
        first = velocity_field(model, x, t_current, problem, oracle=oracle)
        proposal = x + dt * first
        t_following = torch.full((len(x), 1), following, device=actual_device)
        second = velocity_field(model, proposal, t_following, problem, oracle=oracle)
        x = x + 0.5 * dt * (first + second)
    result = x.float().cpu()
    if model is not None:
        model.cpu()
    if actual_device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def sample_latent_reference(
    problem: MixtureFMConfig,
    sample_count: int,
    seed: int = 0,
) -> torch.Tensor:
    device = torch.device("cpu")
    tensors = problem_tensors(problem, device)
    generator = _make_generator(device, 800_001 + int(seed))
    shape = (int(sample_count), problem.dimension)
    signs = torch.randint(0, 2, shape, generator=generator).float() * 2.0 - 1.0
    return signs * tensors["mean"] + torch.randn(shape, generator=generator) * tensors["component_std"]


def distribution_metrics(
    generated: torch.Tensor,
    reference: torch.Tensor,
) -> Dict[str, float]:
    generated = generated.double()
    reference = reference.double()
    if generated.shape != reference.shape:
        raise ValueError("generated and reference samples must have identical shape")
    generated_mean = generated.mean(dim=0)
    reference_mean = reference.mean(dim=0)
    generated_covariance = torch.cov(generated.T)
    reference_covariance = torch.cov(reference.T)
    sorted_generated = torch.sort(generated, dim=0).values
    sorted_reference = torch.sort(reference, dim=0).values
    return {
        "mean_l2": float(torch.linalg.vector_norm(generated_mean - reference_mean).item()),
        "covariance_rel_fro": float(
            (
                torch.linalg.matrix_norm(generated_covariance - reference_covariance)
                / torch.linalg.matrix_norm(reference_covariance).clamp_min(1e-12)
            ).item()
        ),
        "mean_coordinate_w1": float(torch.mean(torch.abs(sorted_generated - sorted_reference)).item()),
        "max_coordinate_w1": float(torch.max(torch.mean(torch.abs(sorted_generated - sorted_reference), dim=0)).item()),
        "sign_balance_error": float(torch.mean(torch.abs((generated > 0).double().mean(dim=0) - 0.5)).item()),
    }


def estimate_predictability(
    problem: MixtureFMConfig,
    *,
    sample_count: int = 131072,
    seed: int = 0,
    device: str = "cuda:0",
) -> pd.DataFrame:
    """Estimate S/R using the analytic conditional residual on fresh samples."""

    actual_device = resolve_device(device)
    generator = _make_generator(actual_device, 900_001 + int(seed))
    batch = sample_fm_batch(problem, int(sample_count), actual_device, generator)
    predictable = torch.mean(batch["conditional_residual"].square(), dim=0)
    total = torch.mean(batch["residual_target"].square(), dim=0)
    irreducible = torch.mean(
        torch.square(batch["residual_target"] - batch["conditional_residual"]), dim=0
    )
    table = pd.DataFrame(
        {
            "direction": np.arange(problem.dimension),
            "latent_variance": np.asarray(problem.variance),
            "bimodal_fraction": np.asarray(problem.bimodal_fraction),
            "predictable_residual_S": predictable.cpu().numpy(),
            "irreducible_residual_N": irreducible.cpu().numpy(),
            "total_residual_R": total.cpu().numpy(),
            "explained_fraction_rho": (predictable / total.clamp_min(1e-12)).cpu().numpy(),
            "decomposition_rel_error": (
                torch.abs(total - predictable - irreducible) / total.clamp_min(1e-12)
            ).cpu().numpy(),
        }
    )
    return table


__all__ = [
    "DEFAULT_BIMODAL_FRACTION",
    "DEFAULT_DECODER_GAIN",
    "DEFAULT_VARIANCE",
    "MiniDiT",
    "MixtureFMConfig",
    "NeuralRun",
    "NeuralTrainConfig",
    "ResidualMLP",
    "build_model",
    "configure_torch",
    "distribution_metrics",
    "estimate_predictability",
    "evaluate_model",
    "fm_statistics",
    "problem_tensors",
    "residual_weight_normalizer",
    "reverse_ode_samples",
    "run_training_grid",
    "run_training_grid_parallel",
    "sample_fm_batch",
    "sample_latent_reference",
    "train_neural_model",
    "velocity_field",
    "weighted_residual_loss",
]
