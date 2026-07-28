"""Fit equal-budget latent flow priors for frozen generation-time bottlenecks."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mnist_generation_time_bottleneck import (  # noqa: E402
    BottleneckConfig,
    CompactConditionEncoder,
    ConditionalVelocityUNet,
    encode_batched,
    euler_sample_conditioned,
)
from experiments.mnist_spectral_rollout_toy import (  # noqa: E402
    _random_directions,
    configure_fp32,
    descending_time_grid,
    frechet_distance,
    load_mnist_tensors,
    sinusoidal_time_embedding,
    sliced_wasserstein,
    train_feature_classifier,
)


DEFAULT_OUTPUT = Path.home() / "data/eqvae/generation_time_prior"


@dataclass(frozen=True)
class PriorConfig:
    stage2_run: Path
    output_root: Path = DEFAULT_OUTPUT
    steps: int = 6_000
    batch_size: int = 256
    hidden_size: int = 128
    depth: int = 3
    learning_rate: float = 2e-4
    sample_count: int = 2_000
    prior_ode_steps: int = 50
    pixel_ode_steps: int = 50
    classifier_epochs: int = 20
    device: str = "cuda:0"
    save: bool = True


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.network = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.SiLU(),
            nn.Linear(2 * hidden_size, hidden_size),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.network(self.norm(value))


class LatentVelocityMLP(nn.Module):
    def __init__(self, latent_dim: int, hidden_size: int = 128, depth: int = 3):
        super().__init__()
        self.time_dim = int(hidden_size)
        self.input = nn.Linear(latent_dim, hidden_size)
        self.time = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(hidden_size) for _ in range(int(depth))]
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.output = nn.Linear(hidden_size, latent_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, value: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        hidden = self.input(value) + self.time(
            sinusoidal_time_embedding(time, self.time_dim)
        )
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(self.norm(hidden))


def _stage2_config(path: Path) -> BottleneckConfig:
    values = json.loads((path / "config.json").read_text())
    values["data_root"] = Path(values["data_root"])
    values["output_root"] = Path(values["output_root"])
    values["eval_times"] = tuple(values["eval_times"])
    return BottleneckConfig(**values)


def load_stage2(
    run: Path,
    device: torch.device,
) -> tuple[BottleneckConfig, CompactConditionEncoder, ConditionalVelocityUNet]:
    config = _stage2_config(run)
    if config.mode not in ("all_time", "high_noise"):
        raise ValueError("latent prior comparison only supports all_time/high_noise")
    state = torch.load(run / "state.pt", map_location=device, weights_only=True)
    encoder = CompactConditionEncoder(config.latent_dim, config.encoder_width).to(device)
    model = ConditionalVelocityUNet(
        config.latent_dim,
        config.model_width,
        config.model_depth,
        mode=config.mode,
        gate_low=config.gate_low,
        gate_high=config.gate_high,
    ).to(device)
    encoder.load_state_dict(state["encoder"])
    model.load_state_dict(state["model"])
    encoder.eval()
    model.eval()
    for module in (encoder, model):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return config, encoder, model


def train_prior(
    latent: torch.Tensor,
    config: PriorConfig,
) -> tuple[LatentVelocityMLP, pd.DataFrame]:
    stage2 = _stage2_config(config.stage2_run)
    torch.manual_seed(int(stage2.seed) + 701)
    if latent.device.type == "cuda":
        torch.cuda.manual_seed(int(stage2.seed) + 701)
    model = LatentVelocityMLP(
        stage2.latent_dim, config.hidden_size, config.depth
    ).to(latent.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=1e-4
    )
    generator = torch.Generator(device=latent.device).manual_seed(int(stage2.seed) + 709)
    rows: list[dict[str, float | int]] = []
    log_every = max(1, int(config.steps) // 40)
    for step in range(1, int(config.steps) + 1):
        indices = torch.randint(
            len(latent), (int(config.batch_size),), device=latent.device, generator=generator
        )
        clean = latent[indices]
        noise = torch.randn(clean.shape, device=latent.device, generator=generator)
        time = torch.rand((len(clean),), device=latent.device, generator=generator)
        state = (1.0 - time[:, None]) * clean + time[:, None] * noise
        target = noise - clean
        prediction = model(state, time)
        loss = F.mse_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % log_every == 0 or step == config.steps:
            rows.append({"step": step, "loss": float(loss.detach())})
    model.eval()
    return model, pd.DataFrame(rows)


@torch.no_grad()
def sample_prior(
    model: LatentVelocityMLP,
    count: int,
    latent_dim: int,
    steps: int,
    *,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(int(seed))
    state = torch.randn((int(count), int(latent_dim)), device=device, generator=generator)
    times = descending_time_grid(steps, device=device)
    for current, following in zip(times[:-1], times[1:]):
        time = torch.full((len(state),), float(current), device=device)
        state = state + (following - current) * model(state, time)
    return state


def train_linear_probe(
    latent: torch.Tensor,
    labels: torch.Tensor,
    *,
    seed: int,
    steps: int = 800,
) -> nn.Linear:
    torch.manual_seed(int(seed) + 811)
    probe = nn.Linear(latent.shape[1], 10).to(latent.device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=0.03)
    generator = torch.Generator(device=latent.device).manual_seed(int(seed) + 821)
    for _ in range(int(steps)):
        indices = torch.randint(
            len(latent), (min(256, len(latent)),), device=latent.device, generator=generator
        )
        loss = F.cross_entropy(probe(latent[indices]), labels[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    probe.eval()
    return probe


def categorical_metrics(logits: torch.Tensor) -> dict[str, float]:
    probabilities = logits.softmax(dim=1)
    mean_probability = probabilities.mean(dim=0)
    entropy = -(mean_probability * mean_probability.clamp_min(1e-12).log()).sum()
    return {
        "class_entropy": float(entropy),
        "effective_classes": float(entropy.exp()),
        "classifier_confidence": float(probabilities.max(dim=1).values.mean()),
    }


@torch.no_grad()
def nearest_train_distance(
    generated: torch.Tensor,
    train: torch.Tensor,
    batch_size: int = 256,
) -> float:
    distances = []
    for batch in generated.split(int(batch_size)):
        distances.append(torch.cdist(batch, train).min(dim=1).values)
    return float(torch.cat(distances).mean())


def evaluate(
    prior: LatentVelocityMLP,
    encoder: CompactConditionEncoder,
    pixel_model: ConditionalVelocityUNet,
    train: torch.Tensor,
    test: torch.Tensor,
    train_labels: torch.Tensor,
    test_labels: torch.Tensor,
    classifier: nn.Module,
    stage2: BottleneckConfig,
    config: PriorConfig,
    normalization: Mapping[str, float],
) -> tuple[dict[str, float | int | str], torch.Tensor, torch.Tensor]:
    train_latent = encode_batched(
        encoder, train, stage2.batch_size, stage2.latent_dim, enabled=True
    )
    test_latent = encode_batched(
        encoder, test, stage2.batch_size, stage2.latent_dim, enabled=True
    )
    count = min(int(config.sample_count), len(test_latent))
    generated_latent = sample_prior(
        prior,
        count,
        stage2.latent_dim,
        config.prior_ode_steps,
        seed=stage2.seed + 907,
        device=train.device,
    )
    directions = _random_directions(stage2.latent_dim, 128, stage2.seed + 911, train.device)
    probe = train_linear_probe(train_latent, train_labels, seed=stage2.seed)
    with torch.no_grad():
        latent_logits = probe(generated_latent)
        latent_metrics = categorical_metrics(latent_logits)
        generator = torch.Generator(device=train.device).manual_seed(stage2.seed + 919)
        initial = torch.randn(
            (count, 1, 28, 28), device=train.device, generator=generator
        )
        times = descending_time_grid(config.pixel_ode_steps, device=train.device)
        generated_images = euler_sample_conditioned(
            pixel_model, initial, generated_latent, times, stage2.batch_size
        )
        mean, std = float(normalization["mean"]), float(normalization["std"])
        pixels = (generated_images * std + mean).clamp(0.0, 1.0)
        normalized = (pixels - mean) / std
        image_logits, image_features = classifier(normalized, return_features=True)
        _, reference_features = classifier(test[:count], return_features=True)
        image_metrics = categorical_metrics(image_logits)
    test_subset = test_latent[:count]
    summary: dict[str, float | int | str] = {
        "mode": stage2.mode,
        "seed": stage2.seed,
        "prior_parameters": sum(parameter.numel() for parameter in prior.parameters()),
        "latent_swd": sliced_wasserstein(test_subset, generated_latent, directions),
        "latent_frechet": frechet_distance(test_subset, generated_latent),
        "latent_nearest_train_distance": nearest_train_distance(generated_latent, train_latent),
        "latent_class_entropy": latent_metrics["class_entropy"],
        "latent_effective_classes": latent_metrics["effective_classes"],
        "latent_classifier_confidence": latent_metrics["classifier_confidence"],
        "image_feature_fid": frechet_distance(reference_features, image_features),
        "image_class_entropy": image_metrics["class_entropy"],
        "image_effective_classes": image_metrics["effective_classes"],
        "image_classifier_confidence": image_metrics["classifier_confidence"],
    }
    return summary, generated_latent.detach(), generated_images[:64].detach().cpu()


def run(config: PriorConfig) -> Path | None:
    config = PriorConfig(**{**asdict(config), "stage2_run": config.stage2_run.expanduser().resolve()})
    stage2 = _stage2_config(config.stage2_run)
    configure_fp32(stage2.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    stage2, encoder, pixel_model = load_stage2(config.stage2_run, device)
    data = load_mnist_tensors(
        stage2.data_root, stage2.train_size, stage2.test_size, stage2.seed
    )
    train = data["train"].to(device)
    test = data["test"].to(device)
    train_labels = data["train_labels"].to(device)
    test_labels = data["test_labels"].to(device)
    train_latent = encode_batched(
        encoder, train, stage2.batch_size, stage2.latent_dim, enabled=True
    )
    prior, history = train_prior(train_latent, config)
    classifier, classifier_accuracy = train_feature_classifier(
        train,
        train_labels,
        test,
        test_labels,
        epochs=config.classifier_epochs,
        batch_size=256,
        seed=stage2.seed,
    )
    summary, generated_latent, samples = evaluate(
        prior,
        encoder,
        pixel_model,
        train,
        test,
        train_labels,
        test_labels,
        classifier,
        stage2,
        config,
        data["normalization"],
    )
    summary["evaluation_classifier_accuracy"] = classifier_accuracy
    output = None
    if config.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = config.output_root.expanduser() / f"{stage2.mode}_seed{stage2.seed}_{timestamp}"
        output.mkdir(parents=True, exist_ok=False)
        values = asdict(config)
        values["stage2_run"] = str(values["stage2_run"])
        values["output_root"] = str(values["output_root"])
        (output / "config.json").write_text(json.dumps(values, indent=2))
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
        history.to_csv(output / "history.csv", index=False)
        torch.save(
            {
                "prior": prior.state_dict(),
                "generated_latent": generated_latent.cpu(),
                "samples": samples,
            },
            output / "state.pt",
        )
    print(json.dumps(summary, indent=2))
    if output is not None:
        print(f"Results: {output}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage2-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=6_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--sample-count", type=int, default=2_000)
    parser.add_argument("--prior-ode-steps", type=int, default=50)
    parser.add_argument("--pixel-ode-steps", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        PriorConfig(
            stage2_run=args.stage2_run,
            output_root=args.output_root,
            steps=args.steps,
            batch_size=args.batch_size,
            hidden_size=args.hidden_size,
            depth=args.depth,
            sample_count=args.sample_count,
            prior_ode_steps=args.prior_ode_steps,
            pixel_ode_steps=args.pixel_ode_steps,
            device=args.device,
            save=not args.no_save,
        )
    )


if __name__ == "__main__":
    main()
