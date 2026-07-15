"""Causal gate for separating temporal and directional FM loss weighting.

The raw inverse-scale weight q_i(t) = R_i(t)^(-gamma) is factorized into a
per-time directional term and a scalar temporal term.  This module reuses the
analytic nonlinear FM mixture problem while keeping optimizer, data stream,
and model initialization paired across treatments.
"""

from __future__ import annotations

import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from multiprocessing import get_context
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from experiments.nonlinear_fm_whitening_toy import (
    MixtureFMConfig,
    NeuralTrainConfig,
    build_model,
    configure_torch,
    evaluate_model,
    residual_weight_normalizer,
    resolve_device,
    sample_fm_batch,
)


WEIGHT_MODES = ("baseline", "direction", "time", "full")


@dataclass(frozen=True)
class GateTreatment:
    mode: str
    gamma: float

    @property
    def name(self) -> str:
        return f"{self.mode}:gamma={self.gamma:g}"


@dataclass
class GateRun:
    treatment: GateTreatment
    config: NeuralTrainConfig
    model: torch.nn.Module
    history: pd.DataFrame
    summary: Dict[str, float | int | str]


def validate_treatment(treatment: GateTreatment) -> None:
    if treatment.mode not in WEIGHT_MODES:
        raise ValueError(f"mode must be one of {WEIGHT_MODES}, got {treatment.mode!r}")
    if not 0.0 <= float(treatment.gamma) <= 1.0:
        raise ValueError("gamma must lie in [0, 1]")
    if treatment.mode == "baseline" and treatment.gamma != 0.0:
        raise ValueError("baseline treatment requires gamma=0")


def gate_weight(
    residual_variance: torch.Tensor,
    treatment: GateTreatment,
    *,
    global_normalizer: float,
    damping: float,
) -> torch.Tensor:
    """Return a weight with an explicit temporal/directional interpretation."""

    validate_treatment(treatment)
    if treatment.mode == "baseline" or treatment.gamma == 0.0:
        return torch.ones_like(residual_variance)
    raw = torch.pow(residual_variance + damping, -float(treatment.gamma))
    temporal = raw.mean(dim=1, keepdim=True)
    if treatment.mode == "direction":
        return raw / temporal.clamp_min(1e-12)
    if treatment.mode == "time":
        return (temporal / float(global_normalizer)).expand_as(raw)
    return raw / float(global_normalizer)


def treatment_weight_diagnostics(
    problem: MixtureFMConfig,
    treatments: Sequence[GateTreatment],
    *,
    damping: float = 1e-4,
    grid_size: int = 4097,
) -> pd.DataFrame:
    t = torch.linspace(problem.t_min, problem.t_max, int(grid_size), dtype=torch.float64)[:, None]
    variance = torch.tensor(problem.variance, dtype=torch.float64)[None]
    residual_variance = variance / ((1.0 - t).square() * variance + t.square())
    rows = []
    for treatment in treatments:
        normalizer = residual_weight_normalizer(problem, treatment.gamma, damping)
        weight = gate_weight(
            residual_variance,
            treatment,
            global_normalizer=normalizer,
            damping=damping,
        )
        per_time_mean = weight.mean(dim=1)
        within_time_std = weight.std(dim=1, unbiased=False)
        rows.append(
            {
                "treatment": treatment.name,
                "mode": treatment.mode,
                "gamma": treatment.gamma,
                "mean_weight": float(weight.mean()),
                "max_weight": float(weight.max()),
                "per_time_mean_cv": float(per_time_mean.std(unbiased=False) / per_time_mean.mean()),
                "mean_within_time_cv": float(
                    torch.mean(within_time_std / per_time_mean.clamp_min(1e-12))
                ),
            }
        )
    return pd.DataFrame(rows)


def gate_loss(
    prediction: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    treatment: GateTreatment,
    *,
    global_normalizer: float,
    damping: float,
) -> torch.Tensor:
    weight = gate_weight(
        batch["residual_variance"],
        treatment,
        global_normalizer=global_normalizer,
        damping=damping,
    )
    return torch.mean(weight * torch.square(prediction - batch["residual_target"]))


def _parameter_norm(model: torch.nn.Module) -> float:
    return math.sqrt(sum(float(parameter.detach().square().sum()) for parameter in model.parameters()))


def train_gate_model(
    problem: MixtureFMConfig,
    config: NeuralTrainConfig,
    treatment: GateTreatment,
) -> GateRun:
    validate_treatment(treatment)
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
    global_normalizer = residual_weight_normalizer(problem, treatment.gamma, config.damping)
    train_generator = torch.Generator(device=device).manual_seed(100_003 + config.seed)
    eval_generator = torch.Generator(device=device).manual_seed(200_003 + config.seed)
    eval_batch = sample_fm_batch(problem, config.eval_count, device, eval_generator)
    rows = []
    clipped_steps = 0
    update_ratios = []
    start_time = time.perf_counter()

    def record(step: int, train_loss: float, gradient_norm: float) -> None:
        metrics = evaluate_model(model, eval_batch, problem)
        row = {
            "step": int(step),
            "train_loss": float(train_loss),
            "gradient_norm": float(gradient_norm),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "architecture": config.architecture,
            "batch_size": config.batch_size,
            "seed": config.seed,
            "treatment": treatment.name,
            "mode": treatment.mode,
            "gamma": treatment.gamma,
            "excess_mse": metrics["excess_mse"],
            "decoder_weighted_mse": metrics["decoder_weighted_mse"],
            "target_mse": metrics["target_mse"],
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
        loss = gate_loss(
            prediction,
            batch,
            treatment,
            global_normalizer=global_normalizer,
            damping=config.damping,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if config.gradient_clip > 0.0:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            last_gradient_norm = float(norm)
            clipped_steps += int(last_gradient_norm > config.gradient_clip)
        else:
            last_gradient_norm = math.sqrt(
                sum(
                    float(parameter.grad.detach().square().sum())
                    for parameter in model.parameters()
                    if parameter.grad is not None
                )
            )
        before = [parameter.detach().clone() for parameter in model.parameters()]
        parameter_norm = _parameter_norm(model)
        optimizer.step()
        update_norm = math.sqrt(
            sum(
                float((parameter.detach() - previous).square().sum())
                for parameter, previous in zip(model.parameters(), before)
            )
        )
        update_ratios.append(update_norm / max(parameter_norm, 1e-12))
        scheduler.step()
        last_loss = float(loss.detach())
        if step % config.eval_every == 0 or step == config.steps:
            record(step, last_loss, last_gradient_norm)

    history = pd.DataFrame(rows)
    final = history.iloc[-1]
    summary: Dict[str, float | int | str] = {
        "architecture": config.architecture,
        "batch_size": config.batch_size,
        "seed": config.seed,
        "treatment": treatment.name,
        "mode": treatment.mode,
        "gamma": treatment.gamma,
        "steps": config.steps,
        "final_excess_mse": float(final["excess_mse"]),
        "best_excess_mse": float(history["excess_mse"].min()),
        "excess_auc": float(np.trapz(history["excess_mse"], history["step"]) / config.steps),
        "final_decoder_weighted_mse": float(final["decoder_weighted_mse"]),
        "final_target_mse": float(final["target_mse"]),
        "clip_rate": clipped_steps / config.steps,
        "mean_update_to_weight": float(np.mean(update_ratios)),
        "runtime_seconds": time.perf_counter() - start_time,
    }
    for column in (name for name in history.columns if name.startswith("direction_mse_")):
        summary[f"final_{column}"] = float(final[column])
    model = model.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return GateRun(treatment=treatment, config=config, model=model, history=history, summary=summary)


def _run_device_tasks(problem: MixtureFMConfig, tasks: Sequence[tuple]) -> list[tuple]:
    completed = []
    for key, config, treatment in tasks:
        completed.append((key, train_gate_model(problem, config, treatment)))
    return completed


def run_gate_grid_parallel(
    problem: MixtureFMConfig,
    setups: Sequence[Tuple[str, NeuralTrainConfig]],
    treatments: Sequence[GateTreatment],
    seeds: Iterable[int],
    devices: Sequence[str],
    *,
    verbose: bool = True,
) -> tuple[Dict[tuple, GateRun], pd.DataFrame, pd.DataFrame]:
    """Run paired treatments with one long-lived worker per device."""

    devices = tuple(devices) or ("cpu",)
    task_rows = []
    for setup_name, base_config in setups:
        for treatment in treatments:
            for seed in seeds:
                key = (setup_name, treatment.mode, float(treatment.gamma), int(seed))
                task_rows.append((key, setup_name, base_config, treatment, int(seed)))
    groups = [[] for _ in devices]
    for index, (key, _, base_config, treatment, seed) in enumerate(task_rows):
        device = devices[index % len(devices)]
        config = replace(base_config, seed=seed, device=device)
        groups[index % len(devices)].append((key, config, treatment))

    runs: Dict[tuple, GateRun] = {}
    if len(devices) == 1:
        completed_groups = [_run_device_tasks(problem, groups[0])]
    else:
        context = get_context("spawn")
        completed_groups = []
        with ProcessPoolExecutor(max_workers=len(devices), mp_context=context) as executor:
            futures = [executor.submit(_run_device_tasks, problem, group) for group in groups if group]
            for future in as_completed(futures):
                completed_groups.append(future.result())
    for completed in completed_groups:
        for key, run in completed:
            runs[key] = run
            if verbose:
                print(
                    f"done {key}: excess={run.summary['final_excess_mse']:.5g}, "
                    f"clip={run.summary['clip_rate']:.3f}"
                )
    ordered = {row[0]: runs[row[0]] for row in task_rows}
    history = pd.concat([run.history for run in ordered.values()], ignore_index=True)
    summary = pd.DataFrame([run.summary for run in ordered.values()])
    return ordered, history, summary


def paired_treatment_effects(
    summary: pd.DataFrame,
    *,
    baseline: str = "baseline:gamma=0",
    metrics: Sequence[str] = ("final_excess_mse", "excess_auc", "final_decoder_weighted_mse"),
) -> pd.DataFrame:
    rows = []
    for (architecture, batch_size), group in summary.groupby(["architecture", "batch_size"]):
        for metric in metrics:
            pivot = group.pivot(index="seed", columns="treatment", values=metric)
            if baseline not in pivot:
                continue
            for treatment in pivot.columns:
                if treatment == baseline:
                    continue
                valid = pivot[[baseline, treatment]].dropna()
                gain = (valid[baseline] - valid[treatment]) / valid[baseline]
                rows.append(
                    {
                        "architecture": architecture,
                        "batch_size": batch_size,
                        "metric": metric,
                        "treatment": treatment,
                        "relative_gain_mean": gain.mean(),
                        "relative_gain_std": gain.std(ddof=1),
                        "wins": int((gain > 0).sum()),
                        "seed_count": len(gain),
                    }
                )
    return pd.DataFrame(rows)


__all__ = [
    "GateRun",
    "GateTreatment",
    "WEIGHT_MODES",
    "gate_loss",
    "gate_weight",
    "paired_treatment_effects",
    "run_gate_grid_parallel",
    "train_gate_model",
    "treatment_weight_diagnostics",
]
