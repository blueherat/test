"""Analytic mixture transport audit without the oracle linear skip."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.nonlinear_fm_whitening_toy import (  # noqa: E402
    MixtureFMConfig,
    ResidualMLP,
    configure_torch,
    distribution_metrics,
    fm_statistics,
    residual_weight_normalizer,
    sample_fm_batch,
    sample_latent_reference,
)
from experiments.nonlinear_transport_gap import _sliced_wasserstein  # noqa: E402


@dataclass(frozen=True)
class RawVelocityConfig:
    output_root: Path = Path.home() / "data/eqvae/experiments/nonlinear_raw_velocity_gap"
    hidden_sizes: tuple[int, ...] = (24, 96)
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    devices: tuple[str, ...] = ("cuda:0", "cuda:1", "cuda:2", "cuda:3")
    gamma: float = 0.5
    damping: float = 1e-4
    batch_size: int = 128
    steps: int = 800
    learning_rate: float = 2e-3
    depth: int = 2
    eval_count: int = 4096
    sample_count: int = 4096
    ode_steps: int = 80
    save: bool = True


def _device(name: str) -> torch.device:
    requested = torch.device(name)
    if requested.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return requested


def _weighted_raw_loss(
    prediction: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    *,
    gamma: float,
    damping: float,
    normalizer: float,
) -> torch.Tensor:
    weights = (batch["residual_variance"] + float(damping)).pow(-float(gamma)) / float(
        normalizer
    )
    return torch.mean(weights * (prediction - batch["velocity"]).square())


@torch.no_grad()
def evaluate_raw_models(
    models: Mapping[str, torch.nn.Module],
    batch: Mapping[str, torch.Tensor],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    direction_rows = []
    for variant, model in models.items():
        prediction = model(batch["network_input"], batch["t"])
        error = prediction - batch["conditional_velocity"]
        direction = error.square().mean(dim=0)
        rows.append(
            {
                "variant": variant,
                "exact_field_mse": float(direction.mean()),
                "microscopic_target_mse": float(
                    (prediction - batch["velocity"]).square().mean()
                ),
            }
        )
        for index, value in enumerate(direction.tolist()):
            direction_rows.append(
                {
                    "variant": variant,
                    "direction": int(index),
                    "exact_field_mse": float(value),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(direction_rows)


@torch.no_grad()
def raw_rollout(
    model: torch.nn.Module,
    problem: MixtureFMConfig,
    *,
    sample_count: int,
    ode_steps: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(700_001 + int(seed))
    state = torch.randn((int(sample_count), problem.dimension), generator=generator, device=device)
    model = model.to(device).eval()
    times = torch.linspace(1.0, 0.0, int(ode_steps) + 1, device=device)
    for current, following in zip(times[:-1], times[1:]):
        current_time = torch.full((len(state), 1), float(current), device=device)
        first_batch = fm_statistics(state, current_time, problem)
        first = model(first_batch["network_input"], current_time)
        proposal = state + (following - current) * first
        following_time = torch.full((len(state), 1), float(following), device=device)
        second_batch = fm_statistics(proposal, following_time, problem)
        second = model(second_batch["network_input"], following_time)
        state = state + 0.5 * (following - current) * (first + second)
    return state.detach().cpu()


def train_pair(
    problem: MixtureFMConfig,
    config: RawVelocityConfig,
    *,
    hidden_size: int,
    seed: int,
    device_name: str,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, dict[str, dict[str, torch.Tensor]]]:
    device = _device(device_name)
    configure_torch(int(seed))
    baseline = ResidualMLP(problem.dimension, int(hidden_size), config.depth).to(device)
    weighted = copy.deepcopy(baseline)
    models = {"baseline": baseline, "weighted": weighted}
    optimizers = {
        name: torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
        for name, model in models.items()
    }
    schedulers = {
        name: torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.steps,
            eta_min=config.learning_rate * 0.05,
        )
        for name, optimizer in optimizers.items()
    }
    normalizer = residual_weight_normalizer(problem, config.gamma, config.damping)
    train_generator = torch.Generator(device=device).manual_seed(100_003 + int(seed))
    eval_generator = torch.Generator(device=device).manual_seed(200_003 + int(seed))
    eval_batch = sample_fm_batch(problem, config.eval_count, device, eval_generator)
    for step in range(1, int(config.steps) + 1):
        batch = sample_fm_batch(problem, config.batch_size, device, train_generator)
        for variant, model in models.items():
            prediction = model(batch["network_input"], batch["t"])
            if variant == "baseline":
                loss = (prediction - batch["velocity"]).square().mean()
            else:
                loss = _weighted_raw_loss(
                    prediction,
                    batch,
                    gamma=config.gamma,
                    damping=config.damping,
                    normalizer=normalizer,
                )
            optimizers[variant].zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizers[variant].step()
            schedulers[variant].step()

    evaluation, direction = evaluate_raw_models(models, eval_batch)
    reference = sample_latent_reference(problem, config.sample_count, int(seed) + 801)
    endpoint_rows = []
    for variant, model in models.items():
        generated = raw_rollout(
            model,
            problem,
            sample_count=config.sample_count,
            ode_steps=config.ode_steps,
            seed=int(seed),
            device=device,
        )
        metrics = distribution_metrics(generated, reference)
        metrics["sliced_w1"] = _sliced_wasserstein(
            generated, reference, seed=int(seed) + 809
        )
        for metric, value in metrics.items():
            endpoint_rows.append(
                {"variant": variant, "metric": metric, "value": float(value)}
            )
    endpoint = pd.DataFrame(endpoint_rows)
    exact = evaluation.set_index("variant")["exact_field_mse"]
    endpoint_pivot = endpoint.pivot(index="metric", columns="variant", values="value")
    high_directions = direction[direction["direction"].ge(problem.dimension - 2)].pivot(
        index="direction", columns="variant", values="exact_field_mse"
    )
    summary: dict[str, object] = {
        "hidden_size": int(hidden_size),
        "seed": int(seed),
        "teacher_exact_mse_ratio": float(exact["weighted"] / exact["baseline"]),
        "high_variance_direction_mse_ratio": float(
            high_directions["weighted"].mean() / high_directions["baseline"].mean()
        ),
    }
    for metric in ("mean_coordinate_w1", "covariance_rel_fro", "sliced_w1"):
        summary[f"endpoint_{metric}_ratio"] = float(
            endpoint_pivot.loc[metric, "weighted"]
            / max(endpoint_pivot.loc[metric, "baseline"], 1e-12)
        )
    states = {
        variant: {name: value.detach().cpu() for name, value in model.state_dict().items()}
        for variant, model in models.items()
    }
    evaluation.insert(0, "seed", int(seed))
    evaluation.insert(0, "hidden_size", int(hidden_size))
    direction.insert(0, "seed", int(seed))
    direction.insert(0, "hidden_size", int(hidden_size))
    endpoint.insert(0, "seed", int(seed))
    endpoint.insert(0, "hidden_size", int(hidden_size))
    detail = pd.concat(
        [
            evaluation.assign(table="teacher"),
            endpoint.assign(table="endpoint"),
        ],
        ignore_index=True,
        sort=False,
    )
    for model in models.values():
        model.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary, detail, direction, states


def _run_group(
    tasks: Sequence[tuple[MixtureFMConfig, RawVelocityConfig, int, int, str]],
) -> list[tuple]:
    completed = []
    for problem, config, hidden_size, seed, device in tasks:
        result = train_pair(
            problem,
            config,
            hidden_size=hidden_size,
            seed=seed,
            device_name=device,
        )
        print(
            f"raw h={hidden_size} seed={seed}: teacher={result[0]['teacher_exact_mse_ratio']:.3f}, "
            f"endpoint_w1={result[0]['endpoint_mean_coordinate_w1_ratio']:.3f}"
        )
        completed.append((hidden_size, seed, *result))
    return completed


def run_raw_velocity_study(
    config: RawVelocityConfig = RawVelocityConfig(),
    problem: MixtureFMConfig = MixtureFMConfig(),
) -> tuple[pd.DataFrame, Path | None]:
    devices = config.devices or ("cpu",)
    tasks = []
    for index, (hidden_size, seed) in enumerate(
        (hidden_size, seed) for hidden_size in config.hidden_sizes for seed in config.seeds
    ):
        tasks.append((problem, config, int(hidden_size), int(seed), devices[index % len(devices)]))
    grouped = [[] for _ in devices]
    for index, task in enumerate(tasks):
        grouped[index % len(devices)].append(task)
    completed = []
    if len(devices) == 1:
        completed = _run_group(grouped[0])
    else:
        context = get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(devices), mp_context=context) as executor:
            futures = [executor.submit(_run_group, group) for group in grouped if group]
            for future in as_completed(futures):
                completed.extend(future.result())
    completed.sort(key=lambda value: (value[0], value[1]))
    summary = pd.DataFrame([value[2] for value in completed])
    result_dir = None
    if config.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = config.output_root.expanduser() / f"preregistered_v1_{timestamp}"
        result_dir.mkdir(parents=True, exist_ok=False)
        serialized = asdict(config)
        serialized["output_root"] = str(config.output_root.expanduser())
        (result_dir / "config.json").write_text(
            json.dumps(serialized, indent=2), encoding="utf-8"
        )
        (result_dir / "problem.json").write_text(
            json.dumps(asdict(problem), indent=2), encoding="utf-8"
        )
        summary.to_csv(result_dir / "summary.csv", index=False)
        pd.concat([value[3] for value in completed], ignore_index=True).to_csv(
            result_dir / "detail.csv", index=False
        )
        pd.concat([value[4] for value in completed], ignore_index=True).to_csv(
            result_dir / "direction_metrics.csv", index=False
        )
        torch.save(
            {f"h{value[0]}_seed{value[1]}": value[5] for value in completed},
            result_dir / "models.pt",
        )
    return summary, result_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    devices = tuple(value.strip() for value in args.devices.split(",") if value.strip())
    config = RawVelocityConfig(devices=devices or ("cpu",), save=not args.no_save)
    if args.quick:
        config = RawVelocityConfig(
            hidden_sizes=(24,),
            seeds=(0,),
            devices=(devices[0],) if devices else ("cpu",),
            steps=4,
            eval_count=32,
            sample_count=32,
            ode_steps=10,
            save=not args.no_save,
        )
    summary, result_dir = run_raw_velocity_study(config)
    print(f"result_dir={result_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
