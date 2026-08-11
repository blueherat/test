#!/usr/bin/env python3
"""Reproduce AutoGuidance and Internal Guidance on the official 2-D toy.

This runner deliberately keeps the study independent of RAE/RAEv2.  It uses
the fractal Gaussian mixture and exact score-matching protocol released with
AutoGuidance.  AutoGuidance uses two separately trained models for the same
target; Internal Guidance uses a shared trunk with intermediate and final
heads trained against the same exact score.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
EDM2_ROOT = ROOT / "research_repos" / "internal_guidance_study" / "edm2"
if str(EDM2_ROOT) not in sys.path:
    sys.path.insert(0, str(EDM2_ROOT))

import dnnlib  # noqa: E402
import toy_example as ag_toy  # noqa: E402
import training.phema  # noqa: E402


EPS = 1e-12


@dataclass(frozen=True)
class ExperimentConfig:
    train_iters: int = 4096
    batch_size: int = 4096
    hidden_dim: int = 64
    num_layers: int = 4
    intermediate_after: int = 1
    intermediate_weight: float = 0.5
    seed: int = 0
    p_mean: float = -2.3
    p_std: float = 1.5
    sigma_data: float = 0.5
    lr_ref: float = 1e-2
    lr_iter: int = 512
    ema_std: float = 0.010


class InternalToyModel(nn.Module):
    """EDM2 toy potential with an early and a final density head.

    ``intermediate_after=1`` means that the auxiliary head reads the first
    hidden block after the input projection, matching Appendix F of the IG
    paper.  Each head has its own zero-initialized scalar gain while all MLP
    layers up to the intermediate point are shared.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        num_layers: int = 4,
        intermediate_after: int = 1,
        sigma_data: float = 0.5,
    ) -> None:
        super().__init__()
        if not 1 <= intermediate_after <= num_layers:
            raise ValueError("intermediate_after must be in [1, num_layers]")
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.intermediate_after = int(intermediate_after)
        self.sigma_data = float(sigma_data)
        self.input_layer = ag_toy.MPLinear(4, hidden_dim)
        self.activations = nn.ModuleList(
            ag_toy.MPSiLU() for _ in range(num_layers)
        )
        self.hidden_layers = nn.ModuleList(
            ag_toy.MPLinear(hidden_dim, hidden_dim) for _ in range(num_layers)
        )
        self.intermediate_gain = nn.Parameter(torch.zeros([]))
        self.final_gain = nn.Parameter(torch.zeros([]))

    @staticmethod
    def _potential(
        features: torch.Tensor,
        normalized_x: torch.Tensor,
        sigma: torch.Tensor,
        gain: torch.Tensor,
    ) -> torch.Tensor:
        return (
            features.square().mean(dim=-1) * gain / sigma.squeeze(-1)
            - 0.5 * normalized_x.square().sum(dim=-1)
        )

    def potentials(
        self, x: torch.Tensor, sigma: torch.Tensor | float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sigma_tensor = torch.as_tensor(
            sigma, dtype=torch.float32, device=x.device
        ).broadcast_to(x.shape[:-1])
        sigma_column = sigma_tensor.unsqueeze(-1)
        normalized_x = x / torch.sqrt(
            self.sigma_data**2 + sigma_column.square()
        )
        network_input = torch.cat(
            [
                normalized_x,
                sigma_column.log() / 4.0,
                torch.ones_like(sigma_column),
            ],
            dim=-1,
        )
        features = self.input_layer(network_input)
        intermediate = None
        for index, (activation, layer) in enumerate(
            zip(self.activations, self.hidden_layers), start=1
        ):
            features = layer(activation(features))
            if index == self.intermediate_after:
                intermediate = features
        if intermediate is None:
            raise RuntimeError("intermediate features were not produced")
        return (
            self._potential(
                intermediate,
                normalized_x,
                sigma_column,
                self.intermediate_gain,
            ),
            self._potential(
                features,
                normalized_x,
                sigma_column,
                self.final_gain,
            ),
        )

    def scores(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor | float,
        *,
        graph: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        differentiable_x = x.detach().requires_grad_(True)
        intermediate_logp, final_logp = self.potentials(
            differentiable_x, sigma
        )
        intermediate_score = torch.autograd.grad(
            intermediate_logp.sum(),
            differentiable_x,
            create_graph=graph,
            retain_graph=True,
        )[0]
        final_score = torch.autograd.grad(
            final_logp.sum(),
            differentiable_x,
            create_graph=graph,
        )[0]
        return intermediate_score, final_score


class InternalScoreView:
    """Expose one IG score or their affine extrapolation to the EDM sampler."""

    def __init__(self, model: InternalToyModel, weight: float) -> None:
        self.model = model
        self.weight = float(weight)

    def score(
        self, x: torch.Tensor, sigma: torch.Tensor | float
    ) -> torch.Tensor:
        intermediate, final = self.model.scores(x, sigma)
        return torch.lerp(intermediate, final, self.weight)


def load_pickle(path_or_url: str, device: torch.device) -> nn.Module:
    with dnnlib.util.open_url(path_or_url) as handle:
        return pickle.load(handle).to(device).eval().requires_grad_(False)


def save_checkpoint(
    path: Path,
    *,
    model: InternalToyModel,
    config: ExperimentConfig,
    step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(config),
            "step": int(step),
        },
        temporary,
    )
    temporary.replace(path)


def load_internal_checkpoint(
    path: Path, device: torch.device
) -> tuple[InternalToyModel, ExperimentConfig, int]:
    payload = torch.load(path, map_location=device, weights_only=False)
    config = ExperimentConfig(**payload["config"])
    model = InternalToyModel(
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        intermediate_after=config.intermediate_after,
        sigma_data=config.sigma_data,
    ).to(device)
    model.load_state_dict(payload["model"])
    return model.eval().requires_grad_(False), config, int(payload["step"])


def train_internal_model(
    *,
    config: ExperimentConfig,
    device: torch.device,
    checkpoint_dir: Path,
    checkpoint_every: int,
) -> InternalToyModel:
    torch.manual_seed(config.seed)
    model = InternalToyModel(
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        intermediate_after=config.intermediate_after,
        sigma_data=config.sigma_data,
    ).to(device).train().requires_grad_(True)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.99))
    distribution = ag_toy.gt("A", device)
    progress = tqdm(range(config.train_iters), desc="Training IG toy")
    for index in progress:
        optimizer.param_groups[0]["lr"] = config.lr_ref / math.sqrt(
            max(index / config.lr_iter, 1.0)
        )
        optimizer.zero_grad(set_to_none=True)
        sigma = (
            torch.randn(config.batch_size, device=device) * config.p_std
            + config.p_mean
        ).exp()
        samples = distribution.sample(config.batch_size, sigma)
        target_score = distribution.score(samples, sigma)
        intermediate_score, final_score = model.scores(
            samples, sigma, graph=True
        )
        intermediate_loss = sigma.square() * (
            intermediate_score - target_score
        ).square().mean(dim=-1)
        final_loss = sigma.square() * (
            final_score - target_score
        ).square().mean(dim=-1)
        loss = final_loss.mean() + config.intermediate_weight * intermediate_loss.mean()
        loss.backward()
        optimizer.step()

        beta = training.phema.power_function_beta(
            std=config.ema_std, t_next=index + 1, t_delta=1
        )
        for source, target in zip(model.parameters(), ema.parameters()):
            target.lerp_(source.detach(), 1.0 - beta)
        if (index + 1) % 64 == 0:
            progress.set_postfix(
                final=f"{float(final_loss.mean().detach()):.4f}",
                inter=f"{float(intermediate_loss.mean().detach()):.4f}",
            )
        if checkpoint_every > 0 and (index + 1) % checkpoint_every == 0:
            save_checkpoint(
                checkpoint_dir / f"ig_step{index + 1:04d}.pt",
                model=ema,
                config=config,
                step=index + 1,
            )
    ema.eval().requires_grad_(False)
    return ema


def parse_weights(value: str) -> list[float]:
    weights = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not weights or any(not math.isfinite(item) for item in weights):
        raise ValueError("weights must be a non-empty list of finite values")
    return weights


def sample_in_batches(
    *,
    model: object,
    initial: torch.Tensor,
    batch_size: int,
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
) -> np.ndarray:
    outputs = []
    for batch in initial.split(batch_size):
        trajectory = ag_toy.do_sample(
            model,
            batch,
            num_steps=num_steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
        outputs.append(trajectory[-1].detach().float().cpu().numpy())
    result = np.concatenate(outputs, axis=0)
    if not np.isfinite(result).all():
        invalid = int(result.size - np.isfinite(result).sum())
        raise FloatingPointError(
            f"EDM trajectory produced {invalid} non-finite values; "
            "use the paper's 32 Heun steps or a less aggressive sigma range"
        )
    return result


def projection_directions(count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    directions = rng.standard_normal((2, count))
    directions /= np.linalg.norm(directions, axis=0, keepdims=True).clip(EPS)
    return directions


def projected_swd(
    reference: np.ndarray, sample: np.ndarray, directions: np.ndarray
) -> float:
    left = np.sort(reference.astype(np.float64) @ directions, axis=0)
    right = np.sort(sample.astype(np.float64) @ directions, axis=0)
    return float(np.abs(left - right).mean())


def rff_parameters(
    reference: np.ndarray, *, features: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    subset = reference[
        rng.choice(len(reference), size=min(1024, len(reference)), replace=False)
    ]
    first, second = np.array_split(subset, 2)
    bandwidth = float(
        np.median(np.linalg.norm(first[: len(second)] - second, axis=1))
    )
    bandwidth = max(bandwidth, 1e-4)
    weight = rng.standard_normal((2, features)) / bandwidth
    phase = rng.uniform(0.0, 2.0 * math.pi, size=features)
    return weight, phase


def rff_mean(
    samples: np.ndarray,
    weight: np.ndarray,
    phase: np.ndarray,
    batch_size: int = 2048,
) -> np.ndarray:
    total = np.zeros(len(phase), dtype=np.float64)
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size].astype(np.float64)
        total += np.cos(batch @ weight + phase).sum(axis=0)
    return math.sqrt(2.0 / len(phase)) * total / len(samples)


def jensen_shannon(left: np.ndarray, right: np.ndarray) -> float:
    left = left.astype(np.float64) / max(float(left.sum()), EPS)
    right = right.astype(np.float64) / max(float(right.sum()), EPS)
    middle = 0.5 * (left + right)

    def kl(first: np.ndarray, second: np.ndarray) -> float:
        mask = first > 0
        return float(
            np.sum(first[mask] * np.log(first[mask] / second[mask].clip(EPS)))
        )

    return 0.5 * kl(left, middle) + 0.5 * kl(right, middle)


def component_log_probabilities(
    distribution: ag_toy.GaussianMixture,
    samples: np.ndarray,
    batch_size: int = 2048,
) -> list[torch.Tensor]:
    """Stable per-component log densities at sigma=0."""
    outputs = []
    tensor = torch.from_numpy(samples).to(distribution.mu.device)
    dimension = distribution.mu.shape[-1]
    constant = float(dimension * math.log(2.0 * math.pi))
    log_weight = distribution.phi.clamp_min(EPS).log()
    log_determinant = distribution._L.clamp_min(EPS).log().sum(dim=-1)
    for batch in tensor.split(batch_size):
        residual = batch[:, None, :] - distribution.mu[None, :, :]
        eigen_coordinates = torch.einsum(
            "kij,bki->bkj", distribution._Q, residual
        )
        mahalanobis = (
            eigen_coordinates.square() / distribution._L.clamp_min(EPS)[None]
        ).sum(dim=-1)
        outputs.append(
            log_weight[None]
            - 0.5 * (constant + log_determinant[None] + mahalanobis)
        )
    return outputs


def component_histogram(
    distribution: ag_toy.GaussianMixture,
    samples: np.ndarray,
    batch_size: int = 2048,
) -> np.ndarray:
    histogram = torch.zeros(
        len(distribution.phi), dtype=torch.float64, device=distribution.mu.device
    )
    for log_components in component_log_probabilities(
        distribution, samples, batch_size=batch_size
    ):
        assignment = log_components.argmax(dim=-1)
        histogram += torch.bincount(
            assignment, minlength=len(histogram)
        ).double()
    return histogram.cpu().numpy()


def log_probabilities(
    distribution: ag_toy.GaussianMixture,
    samples: np.ndarray,
    batch_size: int = 2048,
) -> np.ndarray:
    values = [
        torch.logsumexp(log_components, dim=-1).detach().cpu()
        for log_components in component_log_probabilities(
            distribution, samples, batch_size=batch_size
        )
    ]
    return torch.cat(values).numpy()


def evaluate_samples(
    *,
    samples: dict[str, np.ndarray],
    reference_name: str,
    distribution: ag_toy.GaussianMixture,
    seed: int,
) -> list[dict[str, float | str]]:
    reference = samples[reference_name]
    for name, values in samples.items():
        if values.shape != reference.shape:
            raise ValueError(
                f"sample shape mismatch: {name}={values.shape}, "
                f"reference={reference.shape}"
            )
        if not np.isfinite(values).all():
            raise FloatingPointError(f"condition {name} contains NaN or Inf")
    directions = projection_directions(256, seed + 101)
    rff_weight, rff_phase = rff_parameters(
        reference, features=1024, seed=seed + 103
    )
    reference_rff = rff_mean(reference, rff_weight, rff_phase)
    reference_logp = log_probabilities(distribution, reference)
    contour_99 = float(np.quantile(reference_logp, 0.01))
    reference_histogram = component_histogram(distribution, reference)
    rows: list[dict[str, float | str]] = []
    for name, values in samples.items():
        logp = log_probabilities(distribution, values)
        rff = rff_mean(values, rff_weight, rff_phase)
        histogram = component_histogram(distribution, values)
        rows.append(
            {
                "condition": name,
                "swd": projected_swd(reference, values, directions),
                "rff_mmd2": float(np.square(reference_rff - rff).sum()),
                "mean_nll": float(-logp.mean()),
                "median_nll": float(-np.median(logp)),
                "outlier_rate_99": float(np.mean(logp < contour_99)),
                "component_jsd": jensen_shannon(
                    reference_histogram, histogram
                ),
                "occupied_components": float(np.count_nonzero(histogram)),
            }
        )
    return rows


def save_rows(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_samples(
    *,
    path: Path,
    samples: dict[str, np.ndarray],
    ordered_names: list[str],
    titles: dict[str, str],
    reference_name: str,
    limit: int = 8192,
) -> None:
    columns = 4
    rows = math.ceil(len(ordered_names) / columns)
    figure, axes = plt.subplots(
        rows, columns, figsize=(5.0 * columns, 5.0 * rows), squeeze=False
    )
    reference = samples[reference_name][:limit]
    x_min, y_min = np.quantile(reference, 0.002, axis=0)
    x_max, y_max = np.quantile(reference, 0.998, axis=0)
    padding = 0.08 * max(x_max - x_min, y_max - y_min)
    for axis, name in zip(axes.flat, ordered_names):
        values = samples[name][:limit]
        axis.scatter(
            reference[:, 0],
            reference[:, 1],
            s=2,
            c="#b8b8b8",
            alpha=0.12,
            linewidths=0,
            rasterized=True,
        )
        axis.scatter(
            values[:, 0],
            values[:, 1],
            s=2,
            c="#111111" if name == reference_name else "#e66b1a",
            alpha=0.30,
            linewidths=0,
            rasterized=True,
        )
        axis.set_title(titles.get(name, name), fontsize=12)
        axis.set_xlim(x_min - padding, x_max + padding)
        axis.set_ylim(y_min - padding, y_max + padding)
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in axes.flat[len(ordered_names) :]:
        axis.set_visible(False)
    figure.suptitle(
        "Official fractal toy: same initial noise for every condition",
        fontsize=16,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-iters", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--sample-count", type=int, default=16384)
    parser.add_argument("--sample-batch-size", type=int, default=2048)
    parser.add_argument("--sample-steps", type=int, default=32)
    parser.add_argument("--checkpoint-every", type=int, default=512)
    parser.add_argument("--ig-weights", default="1,1.5,2,3")
    parser.add_argument("--ag-weights", default="1,1.5,2,3")
    parser.add_argument("--ig-checkpoint", type=Path)
    parser.add_argument(
        "--ag-strong",
        default=(
            "https://nvlabs-fi-cdn.nvidia.com/edm2/toy-example/"
            "clsA-layers04-dim64/iter4096.pkl"
        ),
    )
    parser.add_argument(
        "--ag-weak",
        default=(
            "https://nvlabs-fi-cdn.nvidia.com/edm2/toy-example/"
            "clsA-layers04-dim32/iter0512.pkl"
        ),
    )
    args = parser.parse_args()

    if args.train_iters <= 0 or args.batch_size <= 0:
        raise ValueError("training sizes must be positive")
    if args.sample_count <= 0 or args.sample_batch_size <= 0:
        raise ValueError("sample sizes must be positive")
    if args.sample_steps < 8:
        raise ValueError(
            "sample_steps must be at least 8; the official setting is 32 and "
            "very coarse Heun grids can make even the exact-score path diverge"
        )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    config = ExperimentConfig(
        train_iters=args.train_iters,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    if args.ig_checkpoint is not None:
        internal, loaded_config, loaded_step = load_internal_checkpoint(
            args.ig_checkpoint, device
        )
        config = loaded_config
    else:
        internal = train_internal_model(
            config=config,
            device=device,
            checkpoint_dir=output_dir / "checkpoints",
            checkpoint_every=args.checkpoint_every,
        )
        loaded_step = config.train_iters
        save_checkpoint(
            output_dir / "ig_final.pt",
            model=internal,
            config=config,
            step=loaded_step,
        )

    strong = load_pickle(args.ag_strong, device)
    weak = load_pickle(args.ag_weak, device)
    distribution = ag_toy.gt("A", device)
    generator = torch.Generator(device=device.type).manual_seed(args.seed + 1009)
    initial = distribution.sample(
        args.sample_count, sigma=5.0, generator=generator
    )

    conditions: list[tuple[str, object]] = [
        ("reference", distribution),
        ("ag_strong", strong),
        ("ag_weak", weak),
        ("ig_final", InternalScoreView(internal, 1.0)),
        ("ig_intermediate", InternalScoreView(internal, 0.0)),
    ]
    for weight in parse_weights(args.ag_weights):
        if weight != 1.0:
            name = f"ag_w{weight:g}"

            class AutoGuidanceView:
                def score(
                    self, x: torch.Tensor, sigma: torch.Tensor | float,
                    *, _weight: float = weight,
                ) -> torch.Tensor:
                    weak_score = weak.score(x, sigma)
                    strong_score = strong.score(x, sigma)
                    return torch.lerp(weak_score, strong_score, _weight)

            conditions.append((name, AutoGuidanceView()))
    for weight in parse_weights(args.ig_weights):
        if weight != 1.0:
            conditions.append(
                (f"ig_w{weight:g}", InternalScoreView(internal, weight))
            )

    samples: dict[str, np.ndarray] = {}
    for name, model in tqdm(conditions, desc="Sampling conditions"):
        samples[name] = sample_in_batches(
            model=model,
            initial=initial,
            batch_size=args.sample_batch_size,
            num_steps=args.sample_steps,
            sigma_min=0.002,
            sigma_max=5.0,
        )
    np.savez_compressed(output_dir / "samples.npz", **samples)
    metrics = evaluate_samples(
        samples=samples,
        reference_name="reference",
        distribution=distribution,
        seed=args.seed,
    )
    save_rows(output_dir / "metrics.csv", metrics)
    ordered_names = [name for name, _ in conditions]
    titles = {
        "reference": "Exact ground truth",
        "ag_strong": "AG strong, w=1",
        "ag_weak": "AG weak",
        "ig_final": "IG final, w=1",
        "ig_intermediate": "IG intermediate",
    }
    for name in ordered_names:
        if name.startswith("ag_w") and name != "ag_weak":
            titles[name] = f"AutoGuidance w={name[4:]}"
        if name.startswith("ig_w"):
            titles[name] = f"Internal Guidance w={name[4:]}"
    plot_samples(
        path=output_dir / "comparison.png",
        samples=samples,
        ordered_names=ordered_names,
        titles=titles,
        reference_name="reference",
    )
    manifest = {
        "protocol": "official_fractal_autoguidance_internal_guidance_v1",
        "config": asdict(config),
        "loaded_ig_step": loaded_step,
        "conditions": ordered_names,
        "same_initial_noise": True,
        "sampler": {
            "type": "EDM Heun",
            "steps": args.sample_steps,
            "sigma_min": 0.002,
            "sigma_max": 5.0,
            "rho": 7,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({row["condition"]: row for row in metrics}, indent=2))


if __name__ == "__main__":
    main()
