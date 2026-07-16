"""Low-cost MNIST diagnostic for teacher-forced versus rollout behavior.

The experiment keeps the ingredients needed to test the RAE spectral-loss
mechanism while removing the expensive tokenizer and decoder:

* an exactly invertible image latent (normalized MNIST pixels),
* radial DCT directions and the same mean-one weighting used by the RAE study,
* a finite-capacity shared convolutional velocity field,
* paired baseline/weighted training, and
* separate held-out teacher-path and self-generated rollout measurements.

MNIST is a mechanism gate, not evidence about ImageNet generation quality.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.rae_spectral_direction_loss import DCTDirectionLoss


DEFAULT_DATA_ROOT = Path("/data/shared/mnist")
DEFAULT_OUTPUT_ROOT = Path.home() / "data/eqvae/experiments/mnist_spectral_rollout_toy"


@dataclass(frozen=True)
class MNISTToyConfig:
    data_root: Path = DEFAULT_DATA_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    train_size: int = 10_000
    test_size: int = 2_000
    sample_count: int = 2_000
    batch_size: int = 128
    steps: int = 2_000
    learning_rate: float = 2e-4
    width: int = 32
    depth: int = 2
    gamma: float = 0.5
    band_count: int = 8
    time_shift: float = 1.0
    ode_steps: int = 50
    classifier_epochs: int = 3
    classifier_batch_size: int = 256
    eval_times: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)
    seed: int = 0
    device: str = "cuda:0"
    save: bool = True


@dataclass
class MNISTToyResult:
    config: MNISTToyConfig
    history: pd.DataFrame
    teacher_summary: pd.DataFrame
    teacher_bands: pd.DataFrame
    rollout_summary: pd.DataFrame
    rollout_bands: pd.DataFrame
    samples: dict[str, torch.Tensor]
    classifier_accuracy: float
    second_moments: torch.Tensor
    normalization: dict[str, float]
    result_dir: Path | None


def configure_fp32(seed: int) -> None:
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


def _fixed_subset(length: int, count: int, seed: int) -> torch.Tensor:
    if not 1 <= int(count) <= int(length):
        raise ValueError(f"subset count must lie in [1, {length}]")
    return torch.randperm(int(length), generator=torch.Generator().manual_seed(int(seed)))[:count]


def load_mnist_tensors(
    data_root: str | Path,
    train_size: int,
    test_size: int,
    seed: int,
) -> dict[str, torch.Tensor | dict[str, float]]:
    """Load fixed disjoint official splits and normalize from train only."""

    train_set = MNIST(root=str(data_root), train=True, download=True)
    test_set = MNIST(root=str(data_root), train=False, download=True)
    train_index = _fixed_subset(len(train_set), train_size, seed)
    test_index = _fixed_subset(len(test_set), test_size, seed + 1)
    train_images = train_set.data[train_index].float().unsqueeze(1) / 255.0
    test_images = test_set.data[test_index].float().unsqueeze(1) / 255.0
    train_labels = train_set.targets[train_index].long()
    test_labels = test_set.targets[test_index].long()
    mean = float(train_images.mean())
    std = float(train_images.std(unbiased=False).clamp_min(1e-6))
    return {
        "train": (train_images - mean) / std,
        "test": (test_images - mean) / std,
        "train_labels": train_labels,
        "test_labels": test_labels,
        "normalization": {"mean": mean, "std": std},
        "train_indices": train_index,
        "test_indices": test_index,
    }


def shifted_uniform(
    count: int,
    shift: float,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    if float(shift) <= 0:
        raise ValueError("time shift must be positive")
    raw = torch.rand((int(count),), device=device, generator=generator)
    return float(shift) * raw / (1.0 + (float(shift) - 1.0) * raw)


def descending_time_grid(
    steps: int,
    shift: float = 1.0,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return ``steps`` Euler intervals from noise t=1 to data t=0."""

    if int(steps) < 1:
        raise ValueError("steps must be positive")
    if float(shift) <= 0:
        raise ValueError("time shift must be positive")
    raw = torch.linspace(1.0, 0.0, int(steps) + 1, device=device)
    return float(shift) * raw / (1.0 + (float(shift) - 1.0) * raw)


def estimate_band_second_moments(
    clean: torch.Tensor,
    band_count: int,
    batch_size: int = 512,
) -> torch.Tensor:
    analyzer = DCTDirectionLoss(28, torch.ones(int(band_count)), gamma=0.0).to(clean.device)
    total = torch.zeros(int(band_count), device=clean.device, dtype=torch.float64)
    count = 0
    with torch.no_grad():
        for batch in clean.split(int(batch_size)):
            total += analyzer.band_mse(batch).double().sum(dim=0)
            count += len(batch)
    return (total / max(count, 1)).float().clamp_min(1e-6).cpu()


def sinusoidal_time_embedding(time: torch.Tensor, dimension: int) -> torch.Tensor:
    half = int(dimension) // 2
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=time.device, dtype=time.dtype)
        / max(half - 1, 1)
    )
    angles = time[:, None] * frequencies[None] * 1_000.0
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if embedding.shape[1] < int(dimension):
        embedding = F.pad(embedding, (0, int(dimension) - embedding.shape[1]))
    return embedding


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if int(channels) % groups == 0:
            return groups
    return 1


class TimeResBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(input_channels), input_channels)
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.time = nn.Linear(time_dim, 2 * output_channels)
        self.norm2 = nn.GroupNorm(_group_count(output_channels), output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.skip = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv2d(input_channels, output_channels, 1)
        )

    def forward(self, value: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(value)))
        scale, bias = self.time(F.silu(time_embedding)).chunk(2, dim=1)
        hidden = self.norm2(hidden) * (1.0 + scale[:, :, None, None]) + bias[:, :, None, None]
        hidden = self.conv2(F.silu(hidden))
        return self.skip(value) + hidden


class TinyVelocityUNet(nn.Module):
    """Small shared velocity field; width controls the capacity bottleneck."""

    def __init__(self, width: int = 32, depth: int = 2):
        super().__init__()
        width = int(width)
        time_dim = 4 * width
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )
        self.input = nn.Conv2d(1, width, 3, padding=1)
        self.high = TimeResBlock(width, width, time_dim)
        self.down1 = nn.Conv2d(width, 2 * width, 3, stride=2, padding=1)
        self.middle_in = TimeResBlock(2 * width, 2 * width, time_dim)
        self.middle = nn.ModuleList(
            [TimeResBlock(2 * width, 2 * width, time_dim) for _ in range(int(depth))]
        )
        self.up1 = nn.Conv2d(2 * width, width, 3, padding=1)
        self.output_block = TimeResBlock(2 * width, width, time_dim)
        self.output_norm = nn.GroupNorm(_group_count(width), width)
        self.output = nn.Conv2d(width, 1, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, value: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        embedding = self.time_mlp(sinusoidal_time_embedding(time, self.time_dim))
        high = self.high(self.input(value), embedding)
        hidden = self.middle_in(self.down1(high), embedding)
        for block in self.middle:
            hidden = block(hidden, embedding)
        hidden = F.interpolate(hidden, size=high.shape[-2:], mode="nearest")
        hidden = self.up1(hidden)
        hidden = self.output_block(torch.cat([hidden, high], dim=1), embedding)
        return self.output(F.silu(self.output_norm(hidden)))


class MNISTFeatureNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.project = nn.Linear(64 * 7 * 7, 64)
        self.classifier = nn.Linear(64, 10)

    def forward(
        self, value: torch.Tensor, *, return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        hidden = self.features(value).flatten(1)
        features = F.relu(self.project(hidden))
        logits = self.classifier(features)
        return (logits, features) if return_features else logits


def train_feature_classifier(
    train: torch.Tensor,
    labels: torch.Tensor,
    test: torch.Tensor,
    test_labels: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
) -> tuple[MNISTFeatureNet, float]:
    device = train.device
    torch.manual_seed(int(seed) + 59)
    if device.type == "cuda":
        torch.cuda.manual_seed(int(seed) + 59)
    model = MNISTFeatureNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    generator = torch.Generator(device=device).manual_seed(int(seed) + 73)
    model.train()
    for _ in range(int(epochs)):
        permutation = torch.randperm(len(train), device=device, generator=generator)
        for indices in permutation.split(int(batch_size)):
            loss = F.cross_entropy(model(train[indices]), labels[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, targets in zip(test.split(batch_size), test_labels.split(batch_size)):
            correct += int((model(images).argmax(dim=1) == targets).sum())
    return model, correct / len(test)


def train_paired_velocity_fields(
    clean: torch.Tensor,
    config: MNISTToyConfig,
    analyzer: DCTDirectionLoss,
) -> tuple[dict[str, TinyVelocityUNet], pd.DataFrame]:
    device = clean.device
    baseline = TinyVelocityUNet(config.width, config.depth).to(device)
    weighted = copy.deepcopy(baseline)
    models = {"baseline": baseline, "weighted": weighted}
    optimizers = {
        name: torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
        for name, model in models.items()
    }
    generator = torch.Generator(device=device).manual_seed(int(config.seed) + 101)
    rows: list[dict[str, float | int | str]] = []
    log_every = max(1, int(config.steps) // 40)
    for step in range(1, int(config.steps) + 1):
        indices = torch.randint(
            len(clean), (int(config.batch_size),), device=device, generator=generator
        )
        data = clean[indices]
        noise = torch.randn(data.shape, device=device, generator=generator)
        time = shifted_uniform(
            len(data), config.time_shift, device=device, generator=generator
        )
        expanded = time[:, None, None, None]
        state = (1.0 - expanded) * data + expanded * noise
        target = noise - data
        for name, model in models.items():
            prediction = model(state, time)
            if name == "baseline":
                loss = F.mse_loss(prediction, target)
            else:
                loss = analyzer(prediction, target, time)[0].mean()
            optimizers[name].zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizers[name].step()
            if step == 1 or step % log_every == 0 or step == config.steps:
                rows.append(
                    {
                        "step": step,
                        "variant": name,
                        "training_objective": float(loss.detach()),
                        "raw_mse": float(F.mse_loss(prediction.detach(), target)),
                    }
                )
    return models, pd.DataFrame(rows)


@torch.no_grad()
def evaluate_teacher_path(
    models: Mapping[str, nn.Module],
    clean: torch.Tensor,
    analyzer: DCTDirectionLoss,
    eval_times: Sequence[float],
    seed: int,
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    generator = torch.Generator(device=clean.device).manual_seed(int(seed) + 211)
    noise = torch.randn(clean.shape, device=clean.device, generator=generator)
    summary_rows: list[dict[str, float | str]] = []
    band_rows: list[dict[str, float | str | int]] = []
    for time_value in eval_times:
        time = torch.full((len(clean),), float(time_value), device=clean.device)
        expanded = time[:, None, None, None]
        state = (1.0 - expanded) * clean + expanded * noise
        target = noise - clean
        for name, model in models.items():
            mse_sum = 0.0
            clean_mse_sum = 0.0
            band_sum = torch.zeros(analyzer.band_count, device=clean.device)
            seen = 0
            for indices in torch.arange(len(clean), device=clean.device).split(batch_size):
                prediction = model(state[indices], time[indices])
                error = prediction - target[indices]
                count = len(indices)
                mse_sum += float(error.square().mean()) * count
                estimate = state[indices] - expanded[indices] * prediction
                clean_mse_sum += float(F.mse_loss(estimate, clean[indices])) * count
                band_sum += analyzer.band_mse(error).sum(dim=0)
                seen += count
            summary_rows.append(
                {
                    "variant": name,
                    "time": float(time_value),
                    "velocity_mse": mse_sum / seen,
                    "clean_estimate_mse": clean_mse_sum / seen,
                }
            )
            for band, value in enumerate((band_sum / seen).tolist()):
                band_rows.append(
                    {
                        "variant": name,
                        "time": float(time_value),
                        "band": band,
                        "velocity_mse": value,
                    }
                )
    return pd.DataFrame(summary_rows), pd.DataFrame(band_rows)


@torch.no_grad()
def euler_sample(
    model: nn.Module,
    initial: torch.Tensor,
    times: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    outputs = []
    for batch in initial.split(int(batch_size)):
        state = batch
        for current, following in zip(times[:-1], times[1:]):
            time = torch.full((len(state),), float(current), device=state.device)
            state = state + (following - current) * model(state, time)
        outputs.append(state)
    return torch.cat(outputs)


def _random_directions(dimension: int, count: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(int(seed))
    directions = torch.randn((dimension, count), device=device, generator=generator)
    return directions / directions.square().sum(dim=0, keepdim=True).sqrt().clamp_min(1e-12)


def sliced_wasserstein(reference: torch.Tensor, candidate: torch.Tensor, directions: torch.Tensor) -> float:
    if len(reference) != len(candidate):
        raise ValueError("sliced Wasserstein requires equal sample counts")
    reference_projection = torch.sort(reference @ directions, dim=0).values
    candidate_projection = torch.sort(candidate @ directions, dim=0).values
    return float((reference_projection - candidate_projection).square().mean().sqrt())


def frechet_distance(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    first = reference.detach().double().cpu().numpy()
    second = candidate.detach().double().cpu().numpy()
    mean_difference = first.mean(axis=0) - second.mean(axis=0)
    covariance_first = np.cov(first, rowvar=False)
    covariance_second = np.cov(second, rowvar=False)
    covariance_first = 0.5 * (covariance_first + covariance_first.T)
    covariance_second = 0.5 * (covariance_second + covariance_second.T)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance_first)
    first_root = (eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))) @ eigenvectors.T
    middle = first_root @ covariance_second @ first_root
    middle = 0.5 * (middle + middle.T)
    middle_eigenvalues = np.linalg.eigvalsh(middle)
    covariance_root_trace = float(np.sqrt(np.clip(middle_eigenvalues, 0.0, None)).sum())
    value = (
        mean_difference @ mean_difference
        + np.trace(covariance_first)
        + np.trace(covariance_second)
        - 2.0 * covariance_root_trace
    )
    return float(max(value, 0.0))


@torch.no_grad()
def evaluate_rollouts(
    models: Mapping[str, nn.Module],
    reference: torch.Tensor,
    classifier: MNISTFeatureNet,
    analyzer: DCTDirectionLoss,
    config: MNISTToyConfig,
    normalization: Mapping[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, torch.Tensor]]:
    count = min(int(config.sample_count), len(reference))
    reference = reference[:count]
    generator = torch.Generator(device=reference.device).manual_seed(int(config.seed) + 307)
    initial = torch.randn(reference.shape, device=reference.device, generator=generator)
    times = descending_time_grid(config.ode_steps, config.time_shift, device=reference.device)
    latent_directions = _random_directions(28 * 28, 64, config.seed + 401, reference.device)
    pixel_directions = _random_directions(28 * 28, 64, config.seed + 403, reference.device)
    mean = float(normalization["mean"])
    std = float(normalization["std"])
    reference_pixels = (reference * std + mean).clamp(0.0, 1.0)
    reference_decoded = (reference_pixels - mean) / std
    _, reference_features = classifier(reference_decoded, return_features=True)
    reference_bands = analyzer.band_mse(reference).mean(dim=0)
    samples: dict[str, torch.Tensor] = {"reference": reference[:64].cpu()}
    summary_rows: list[dict[str, float | str]] = []
    band_rows: list[dict[str, float | str | int]] = []
    for name, model in models.items():
        generated = euler_sample(model, initial, times, config.batch_size)
        generated_pixels = (generated * std + mean).clamp(0.0, 1.0)
        generated_decoded = (generated_pixels - mean) / std
        logits, features = classifier(generated_decoded, return_features=True)
        probabilities = logits.softmax(dim=1)
        mean_probability = probabilities.mean(dim=0)
        class_entropy = float(-(mean_probability * mean_probability.clamp_min(1e-12).log()).sum())
        summary_rows.append(
            {
                "variant": name,
                "latent_swd": sliced_wasserstein(
                    reference.flatten(1), generated.flatten(1), latent_directions
                ),
                "decoded_pixel_swd": sliced_wasserstein(
                    reference_pixels.flatten(1), generated_pixels.flatten(1), pixel_directions
                ),
                "feature_swd": sliced_wasserstein(
                    reference_features,
                    features,
                    _random_directions(64, 64, config.seed + 409, reference.device),
                ),
                "feature_fid": frechet_distance(reference_features, features),
                "classifier_confidence": float(probabilities.max(dim=1).values.mean()),
                "class_entropy": class_entropy,
                "latent_mean": float(generated.mean()),
                "latent_std": float(generated.std(unbiased=False)),
            }
        )
        generated_bands = analyzer.band_mse(generated).mean(dim=0)
        for band in range(analyzer.band_count):
            band_rows.append(
                {
                    "variant": name,
                    "band": band,
                    "reference_energy": float(reference_bands[band]),
                    "generated_energy": float(generated_bands[band]),
                    "log_energy_ratio": float(
                        (generated_bands[band] / reference_bands[band].clamp_min(1e-12)).log()
                    ),
                }
            )
        samples[name] = generated[:64].cpu()
    return pd.DataFrame(summary_rows), pd.DataFrame(band_rows), samples


def _save_result(result: MNISTToyResult, models: Mapping[str, nn.Module]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"seed{result.config.seed}_w{result.config.width}_"
        f"steps{result.config.steps}_{timestamp}"
    )
    result_dir = result.config.output_root.expanduser() / run_name
    result_dir.mkdir(parents=True, exist_ok=False)
    config = asdict(result.config)
    config["data_root"] = str(config["data_root"])
    config["output_root"] = str(config["output_root"])
    (result_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    result.history.to_csv(result_dir / "history.csv", index=False)
    result.teacher_summary.to_csv(result_dir / "teacher_summary.csv", index=False)
    result.teacher_bands.to_csv(result_dir / "teacher_bands.csv", index=False)
    result.rollout_summary.to_csv(result_dir / "rollout_summary.csv", index=False)
    result.rollout_bands.to_csv(result_dir / "rollout_bands.csv", index=False)
    torch.save(
        {
            "models": {name: model.state_dict() for name, model in models.items()},
            "samples": result.samples,
            "second_moments": result.second_moments,
            "normalization": result.normalization,
            "classifier_accuracy": result.classifier_accuracy,
        },
        result_dir / "state.pt",
    )
    return result_dir


def run_mnist_spectral_toy(config: MNISTToyConfig) -> MNISTToyResult:
    configure_fp32(config.seed)
    device = resolve_device(config.device)
    loaded = load_mnist_tensors(
        config.data_root, config.train_size, config.test_size, config.seed
    )
    train = loaded["train"].to(device)
    test = loaded["test"].to(device)
    train_labels = loaded["train_labels"].to(device)
    test_labels = loaded["test_labels"].to(device)
    moments = estimate_band_second_moments(train, config.band_count)
    analyzer = DCTDirectionLoss(
        28,
        moments.tolist(),
        gamma=config.gamma,
        damping=1e-4,
        min_weight=0.2,
        max_weight=2.0,
    ).to(device)
    models, history = train_paired_velocity_fields(train, config, analyzer)
    for model in models.values():
        model.eval()
    teacher_summary, teacher_bands = evaluate_teacher_path(
        models,
        test,
        analyzer,
        config.eval_times,
        config.seed,
        config.batch_size,
    )
    classifier, classifier_accuracy = train_feature_classifier(
        train,
        train_labels,
        test,
        test_labels,
        epochs=config.classifier_epochs,
        batch_size=config.classifier_batch_size,
        seed=config.seed,
    )
    rollout_summary, rollout_bands, samples = evaluate_rollouts(
        models, test, classifier, analyzer, config, loaded["normalization"]
    )
    result = MNISTToyResult(
        config=config,
        history=history,
        teacher_summary=teacher_summary,
        teacher_bands=teacher_bands,
        rollout_summary=rollout_summary,
        rollout_bands=rollout_bands,
        samples=samples,
        classifier_accuracy=classifier_accuracy,
        second_moments=moments,
        normalization=loaded["normalization"],
        result_dir=None,
    )
    if config.save:
        result.result_dir = _save_result(result, models)
    return result


def comparison_tables(result: MNISTToyResult) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return weighted/baseline ratios; below one is better for error metrics."""

    teacher = result.teacher_summary.pivot(index="time", columns="variant")
    teacher_rows = []
    for time in teacher.index:
        teacher_rows.append(
            {
                "time": float(time),
                "velocity_mse_ratio": float(
                    teacher.loc[time, ("velocity_mse", "weighted")]
                    / teacher.loc[time, ("velocity_mse", "baseline")]
                ),
                "clean_mse_ratio": float(
                    teacher.loc[time, ("clean_estimate_mse", "weighted")]
                    / teacher.loc[time, ("clean_estimate_mse", "baseline")]
                ),
            }
        )
    rollout = result.rollout_summary.set_index("variant")
    lower_is_better = ("latent_swd", "decoded_pixel_swd", "feature_swd", "feature_fid")
    rollout_comparison = pd.DataFrame(
        [
            {
                "metric": metric,
                "baseline": float(rollout.loc["baseline", metric]),
                "weighted": float(rollout.loc["weighted", metric]),
                "weighted_over_baseline": float(
                    rollout.loc["weighted", metric] / rollout.loc["baseline", metric]
                ),
            }
            for metric in lower_is_better
        ]
    )
    return pd.DataFrame(teacher_rows), rollout_comparison


def plot_result(result: MNISTToyResult):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)
    for variant, frame in result.history.groupby("variant"):
        axes[0, 0].plot(frame["step"], frame["raw_mse"], label=variant)
    axes[0, 0].set(title="Training raw MSE", xlabel="step", ylabel="MSE")
    for variant, frame in result.teacher_summary.groupby("variant"):
        axes[0, 1].plot(frame["time"], frame["velocity_mse"], marker="o", label=variant)
    axes[0, 1].set(title="Held-out teacher velocity", xlabel="t", ylabel="MSE")
    for variant, frame in result.rollout_bands.groupby("variant"):
        axes[0, 2].plot(frame["band"], frame["log_energy_ratio"], marker="o", label=variant)
    axes[0, 2].axhline(0.0, color="black", linewidth=1)
    axes[0, 2].set(title="Rollout band energy", xlabel="DCT band", ylabel="log(gen / data)")
    _, rollout_comparison = comparison_tables(result)
    axes[1, 0].bar(
        rollout_comparison["metric"], rollout_comparison["weighted_over_baseline"]
    )
    axes[1, 0].axhline(1.0, color="black", linewidth=1)
    axes[1, 0].set(
        title="Weighted / baseline rollout error",
        ylabel="ratio (below 1 is better)",
    )
    mean = result.normalization["mean"]
    std = result.normalization["std"]
    baseline_images = (result.samples["baseline"][:25, 0] * std + mean).clamp(0.0, 1.0)
    weighted_images = (result.samples["weighted"][:25, 0] * std + mean).clamp(0.0, 1.0)
    axes[1, 1].imshow(
        baseline_images.reshape(5, 5, 28, 28).permute(0, 2, 1, 3).reshape(140, 140),
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )
    axes[1, 1].set_title("Baseline samples")
    axes[1, 2].imshow(
        weighted_images.reshape(5, 5, 28, 28).permute(0, 2, 1, 3).reshape(140, 140),
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )
    axes[1, 2].set_title("Weighted samples")
    for axis in axes.flat:
        if axis in (axes[1, 1], axes[1, 2]):
            axis.axis("off")
        else:
            if axis is not axes[1, 0]:
                axis.legend()
            axis.grid(alpha=0.2)
    plt.close(figure)
    return figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-size", type=int, default=10_000)
    parser.add_argument("--test-size", type=int, default=2_000)
    parser.add_argument("--sample-count", type=int, default=2_000)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--time-shift", type=float, default=1.0)
    parser.add_argument("--ode-steps", type=int, default=50)
    parser.add_argument("--classifier-epochs", type=int, default=3)
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MNISTToyConfig(
        train_size=args.train_size,
        test_size=args.test_size,
        sample_count=args.sample_count,
        batch_size=args.batch_size,
        steps=args.steps,
        width=args.width,
        depth=args.depth,
        gamma=args.gamma,
        time_shift=args.time_shift,
        ode_steps=args.ode_steps,
        classifier_epochs=args.classifier_epochs,
        seed=args.seed,
        device=args.device,
        save=not args.no_save,
    )
    result = run_mnist_spectral_toy(config)
    print(f"classifier accuracy: {result.classifier_accuracy:.4f}")
    print("\nteacher summary")
    print(result.teacher_summary.to_string(index=False))
    print("\nrollout summary")
    print(result.rollout_summary.to_string(index=False))
    if result.result_dir is not None:
        print(f"\nsaved to: {result.result_dir}")


if __name__ == "__main__":
    main()
