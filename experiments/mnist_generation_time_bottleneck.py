"""Controlled MNIST study of when a compact condition may influence generation."""

from __future__ import annotations

import argparse
import json
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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mnist_spectral_rollout_toy import (
    TimeResBlock,
    _group_count,
    configure_fp32,
    descending_time_grid,
    frechet_distance,
    load_mnist_tensors,
    sinusoidal_time_embedding,
    train_feature_classifier,
)
from experiments.noise_responsibility_profile import (
    ResponsibilityBatch,
    aggregate_profile,
    derangement,
    identity_control_error,
    responsibility_rows,
)


DEFAULT_OUTPUT = Path.home() / "data/eqvae/generation_time_bottleneck"
MODES = ("all_time", "high_noise", "low_noise", "none")


@dataclass(frozen=True)
class BottleneckConfig:
    mode: str = "high_noise"
    data_root: Path = Path("/data/shared/mnist")
    output_root: Path = DEFAULT_OUTPUT
    train_size: int = 10_000
    test_size: int = 2_000
    eval_count: int = 1_000
    latent_dim: int = 8
    encoder_width: int = 32
    model_width: int = 32
    model_depth: int = 2
    batch_size: int = 128
    steps: int = 4_000
    learning_rate: float = 2e-4
    latent_noise: float = 0.05
    covariance_weight: float = 0.01
    gate_low: float = 0.55
    gate_high: float = 0.75
    ode_steps: int = 50
    classifier_epochs: int = 20
    eval_times: tuple[float, ...] = (0.1, 0.3, 0.55, 0.7, 0.85, 0.99)
    seed: int = 0
    device: str = "cuda:0"
    save: bool = True


class CompactConditionEncoder(nn.Module):
    def __init__(self, latent_dim: int, width: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, width, 3, stride=2, padding=1),
            nn.GroupNorm(_group_count(width), width),
            nn.SiLU(),
            nn.Conv2d(width, 2 * width, 3, stride=2, padding=1),
            nn.GroupNorm(_group_count(2 * width), 2 * width),
            nn.SiLU(),
        )
        self.project = nn.Linear(2 * width * 7 * 7, latent_dim)
        self.normalize = nn.LayerNorm(latent_dim, elementwise_affine=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.normalize(self.project(self.features(image).flatten(1)))


def condition_gate(
    time: torch.Tensor,
    mode: str,
    *,
    low: float = 0.55,
    high: float = 0.75,
) -> torch.Tensor:
    """Smooth condition schedule from clean ``t=0`` to noise ``t=1``."""

    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    if not 0.0 <= float(low) < float(high) <= 1.0:
        raise ValueError("require 0 <= low < high <= 1")
    if mode == "all_time":
        return torch.ones_like(time)
    if mode == "none":
        return torch.zeros_like(time)
    position = ((time - float(low)) / (float(high) - float(low))).clamp(0.0, 1.0)
    high_gate = position.square() * (3.0 - 2.0 * position)
    return high_gate if mode == "high_noise" else 1.0 - high_gate


class ConditionalVelocityUNet(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        width: int = 32,
        depth: int = 2,
        *,
        mode: str = "high_noise",
        gate_low: float = 0.55,
        gate_high: float = 0.75,
    ):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
        self.mode = mode
        self.gate_low = float(gate_low)
        self.gate_high = float(gate_high)
        width = int(width)
        time_dim = 4 * width
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )
        self.condition_mlp = nn.Sequential(
            nn.Linear(latent_dim, time_dim, bias=False),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim, bias=False),
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

    def forward(
        self,
        value: torch.Tensor,
        time: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        gate = condition_gate(
            time, self.mode, low=self.gate_low, high=self.gate_high
        )[:, None]
        embedding = self.time_mlp(sinusoidal_time_embedding(time, self.time_dim))
        embedding = embedding + gate * self.condition_mlp(condition)
        high = self.high(self.input(value), embedding)
        hidden = self.middle_in(self.down1(high), embedding)
        for block in self.middle:
            hidden = block(hidden, embedding)
        hidden = F.interpolate(hidden, size=high.shape[-2:], mode="nearest")
        hidden = self.up1(hidden)
        hidden = self.output_block(torch.cat([hidden, high], dim=1), embedding)
        return self.output(F.silu(self.output_norm(hidden)))


def covariance_regularizer(latent: torch.Tensor) -> torch.Tensor:
    """Keep code scale non-degenerate without using labels or reconstruction."""

    if latent.ndim != 2 or len(latent) < 2:
        raise ValueError("covariance regularizer expects at least two latent rows")
    centered = latent - latent.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / (len(latent) - 1)
    diagonal = torch.diagonal(covariance)
    off_diagonal = covariance - torch.diag_embed(diagonal)
    return (
        latent.mean(dim=0).square().mean()
        + (diagonal - 1.0).square().mean()
        + off_diagonal.square().mean()
    )


def _sample_training_batch(
    clean: torch.Tensor,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    indices = torch.randint(len(clean), (int(batch_size),), device=clean.device, generator=generator)
    data = clean[indices]
    noise = torch.randn(data.shape, device=clean.device, generator=generator)
    time = torch.rand((len(data),), device=clean.device, generator=generator)
    expanded = time[:, None, None, None]
    state = (1.0 - expanded) * data + expanded * noise
    return data, time, state, noise - data


def train_model(
    clean: torch.Tensor,
    config: BottleneckConfig,
) -> tuple[CompactConditionEncoder, ConditionalVelocityUNet, pd.DataFrame]:
    torch.manual_seed(int(config.seed) + 101)
    if clean.device.type == "cuda":
        torch.cuda.manual_seed(int(config.seed) + 101)
    encoder = CompactConditionEncoder(config.latent_dim, config.encoder_width).to(clean.device)
    model = ConditionalVelocityUNet(
        config.latent_dim,
        config.model_width,
        config.model_depth,
        mode=config.mode,
        gate_low=config.gate_low,
        gate_high=config.gate_high,
    ).to(clean.device)
    parameters = list(model.parameters())
    if config.mode != "none":
        parameters += list(encoder.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=1e-4)
    data_generator = torch.Generator(device=clean.device).manual_seed(int(config.seed) + 211)
    latent_generator = torch.Generator(device=clean.device).manual_seed(int(config.seed) + 223)
    rows: list[dict[str, float | int | str]] = []
    log_every = max(1, int(config.steps) // 40)
    for step in range(1, int(config.steps) + 1):
        data, time, state, target = _sample_training_batch(
            clean, config.batch_size, data_generator
        )
        if config.mode == "none":
            latent = torch.zeros(
                (len(data), config.latent_dim), device=clean.device, dtype=clean.dtype
            )
            regularizer = state.new_zeros(())
        else:
            latent = encoder(data)
            regularizer = covariance_regularizer(latent)
            if config.latent_noise > 0.0:
                latent = latent + float(config.latent_noise) * torch.randn(
                    latent.shape, device=latent.device, generator=latent_generator
                )
        prediction = model(state, time, latent)
        velocity_loss = F.mse_loss(prediction, target)
        loss = velocity_loss + float(config.covariance_weight) * regularizer
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if step == 1 or step % log_every == 0 or step == config.steps:
            gate = condition_gate(
                time, config.mode, low=config.gate_low, high=config.gate_high
            )
            rows.append(
                {
                    "step": step,
                    "mode": config.mode,
                    "loss": float(loss.detach()),
                    "velocity_loss": float(velocity_loss.detach()),
                    "covariance_regularizer": float(regularizer.detach()),
                    "mean_gate": float(gate.mean()),
                }
            )
    encoder.eval()
    model.eval()
    return encoder, model, pd.DataFrame(rows)


@torch.no_grad()
def encode_batched(
    encoder: nn.Module,
    images: torch.Tensor,
    batch_size: int,
    latent_dim: int,
    *,
    enabled: bool,
) -> torch.Tensor:
    if not enabled:
        return torch.zeros((len(images), latent_dim), device=images.device, dtype=images.dtype)
    return torch.cat([encoder(batch) for batch in images.split(int(batch_size))])


@torch.no_grad()
def evaluate_teacher_responsibility(
    encoder: nn.Module,
    model: ConditionalVelocityUNet,
    clean: torch.Tensor,
    config: BottleneckConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    count = min(int(config.eval_count), len(clean))
    clean = clean[:count]
    latent = encode_batched(
        encoder,
        clean,
        config.batch_size,
        config.latent_dim,
        enabled=config.mode != "none",
    )
    permutation = derangement(count, seed=config.seed + 307, device=clean.device)
    generator = torch.Generator(device=clean.device).manual_seed(int(config.seed) + 311)
    noise = torch.randn(clean.shape, device=clean.device, generator=generator)
    all_rows: list[pd.DataFrame] = []
    controls: list[dict[str, float | str]] = []
    for time_value in config.eval_times:
        time = torch.full((count,), float(time_value), device=clean.device)
        expanded = time[:, None, None, None]
        state = (1.0 - expanded) * clean + expanded * noise
        target = noise - clean
        branches: dict[str, list[torch.Tensor]] = {name: [] for name in ("real", "null", "shuffle")}
        repeat_differences: list[dict[str, float]] = []
        for indices in torch.arange(count, device=clean.device).split(config.batch_size):
            real = model(state[indices], time[indices], latent[indices])
            repeated = model(state[indices], time[indices], latent[indices])
            repeat_differences.append(identity_control_error(real, repeated))
            branches["real"].append(real)
            branches["null"].append(model(state[indices], time[indices], torch.zeros_like(latent[indices])))
            branches["shuffle"].append(model(state[indices], time[indices], latent[permutation[indices]]))
        predictions = {name: torch.cat(parts) for name, parts in branches.items()}
        rows = responsibility_rows(
            ResponsibilityBatch(
                timestep=time,
                target=target,
                predictions=predictions,
                sample_index=torch.arange(count, device=clean.device),
            )
        )
        rows.insert(0, "mode", config.mode)
        rows["shuffle_index"] = permutation.cpu().numpy()
        all_rows.append(rows)
        controls.append(
            {
                "mode": config.mode,
                "timestep": float(time_value),
                "absolute_rms_max": max(item["absolute_rms_max"] for item in repeat_differences),
                "relative_rms_max": max(item["relative_rms_max"] for item in repeat_differences),
            }
        )
    paired = pd.concat(all_rows, ignore_index=True)
    profile = aggregate_profile(paired).sort_values("timestep", ascending=False)
    profile.insert(0, "mode", config.mode)
    return paired, profile, pd.DataFrame(controls)


@torch.no_grad()
def euler_sample_conditioned(
    model: ConditionalVelocityUNet,
    initial: torch.Tensor,
    condition: torch.Tensor,
    times: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    outputs = []
    for state, latent in zip(initial.split(batch_size), condition.split(batch_size)):
        for current, following in zip(times[:-1], times[1:]):
            time = torch.full((len(state),), float(current), device=state.device)
            state = state + (following - current) * model(state, time, latent)
        outputs.append(state)
    return torch.cat(outputs)


@torch.no_grad()
def evaluate_rollout(
    encoder: nn.Module,
    model: ConditionalVelocityUNet,
    clean: torch.Tensor,
    labels: torch.Tensor,
    classifier: nn.Module,
    config: BottleneckConfig,
    normalization: Mapping[str, float],
) -> tuple[pd.DataFrame, dict[str, torch.Tensor]]:
    count = min(int(config.eval_count), len(clean))
    clean = clean[:count]
    labels = labels[:count]
    latent = encode_batched(
        encoder,
        clean,
        config.batch_size,
        config.latent_dim,
        enabled=config.mode != "none",
    )
    permutation = derangement(count, seed=config.seed + 401, device=clean.device)
    conditions = {
        "real": latent,
        "null": torch.zeros_like(latent),
        "shuffle": latent[permutation],
    }
    generator = torch.Generator(device=clean.device).manual_seed(int(config.seed) + 409)
    initial = torch.randn(clean.shape, device=clean.device, generator=generator)
    times = descending_time_grid(config.ode_steps, device=clean.device)
    mean, std = float(normalization["mean"]), float(normalization["std"])
    reference_pixels = (clean * std + mean).clamp(0.0, 1.0)
    _, reference_features = classifier(clean, return_features=True)
    rows: list[dict[str, float | str]] = []
    samples: dict[str, torch.Tensor] = {}
    for branch, condition in conditions.items():
        generated = euler_sample_conditioned(
            model, initial, condition, times, config.batch_size
        )
        pixels = (generated * std + mean).clamp(0.0, 1.0)
        normalized = (pixels - mean) / std
        logits, features = classifier(normalized, return_features=True)
        prediction = logits.argmax(dim=1)
        rows.append(
            {
                "mode": config.mode,
                "branch": branch,
                "source_class_match": float((prediction == labels).float().mean()),
                "shuffled_class_match": float((prediction == labels[permutation]).float().mean()),
                "classifier_confidence": float(logits.softmax(dim=1).max(dim=1).values.mean()),
                "feature_fid": frechet_distance(reference_features, features),
                "pixel_mse_to_source": float(F.mse_loss(pixels, reference_pixels)),
            }
        )
        samples[branch] = generated[:64].cpu()
    return pd.DataFrame(rows), samples


def linear_probe_accuracy(
    train_latent: torch.Tensor,
    train_labels: torch.Tensor,
    test_latent: torch.Tensor,
    test_labels: torch.Tensor,
    *,
    seed: int,
    steps: int = 400,
) -> float:
    torch.manual_seed(int(seed) + 503)
    probe = nn.Linear(train_latent.shape[1], 10).to(train_latent.device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=0.03)
    generator = torch.Generator(device=train_latent.device).manual_seed(int(seed) + 509)
    probe.train()
    for _ in range(int(steps)):
        indices = torch.randint(
            len(train_latent), (min(256, len(train_latent)),),
            device=train_latent.device, generator=generator
        )
        loss = F.cross_entropy(probe(train_latent[indices]), train_labels[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    probe.eval()
    with torch.no_grad():
        return float((probe(test_latent).argmax(dim=1) == test_labels).float().mean())


def _save(
    config: BottleneckConfig,
    encoder: nn.Module,
    model: nn.Module,
    history: pd.DataFrame,
    paired: pd.DataFrame,
    profile: pd.DataFrame,
    controls: pd.DataFrame,
    rollout: pd.DataFrame,
    samples: Mapping[str, torch.Tensor],
    summary: Mapping[str, float | str],
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = config.output_root.expanduser() / f"{config.mode}_seed{config.seed}_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    values = asdict(config)
    values["data_root"] = str(values["data_root"])
    values["output_root"] = str(values["output_root"])
    values["eval_times"] = list(values["eval_times"])
    (output / "config.json").write_text(json.dumps(values, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(dict(summary), indent=2), encoding="utf-8")
    history.to_csv(output / "history.csv", index=False)
    paired.to_csv(output / "teacher_paired.csv", index=False)
    profile.to_csv(output / "teacher_profile.csv", index=False)
    controls.to_csv(output / "identity_controls.csv", index=False)
    rollout.to_csv(output / "rollout.csv", index=False)
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "model": model.state_dict(),
            "samples": dict(samples),
        },
        output / "state.pt",
    )
    return output


def run(config: BottleneckConfig) -> Path | None:
    if config.mode not in MODES:
        raise ValueError(f"unknown mode {config.mode!r}")
    configure_fp32(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    loaded = load_mnist_tensors(config.data_root, config.train_size, config.test_size, config.seed)
    train = loaded["train"].to(device)
    test = loaded["test"].to(device)
    train_labels = loaded["train_labels"].to(device)
    test_labels = loaded["test_labels"].to(device)
    encoder, model, history = train_model(train, config)
    paired, profile, controls = evaluate_teacher_responsibility(
        encoder, model, test, config
    )
    classifier, classifier_accuracy = train_feature_classifier(
        train,
        train_labels,
        test,
        test_labels,
        epochs=config.classifier_epochs,
        batch_size=256,
        seed=config.seed,
    )
    rollout, samples = evaluate_rollout(
        encoder,
        model,
        test,
        test_labels,
        classifier,
        config,
        loaded["normalization"],
    )
    train_latent = encode_batched(
        encoder, train, config.batch_size, config.latent_dim, enabled=config.mode != "none"
    )
    test_latent = encode_batched(
        encoder, test, config.batch_size, config.latent_dim, enabled=config.mode != "none"
    )
    probe_accuracy = (
        0.1
        if config.mode == "none"
        else linear_probe_accuracy(
            train_latent, train_labels, test_latent, test_labels, seed=config.seed
        )
    )
    summary: dict[str, float | str] = {
        "mode": config.mode,
        "classifier_accuracy": classifier_accuracy,
        "latent_linear_probe_accuracy": probe_accuracy,
        "identity_absolute_max": float(controls["absolute_rms_max"].max()),
        "identity_relative_max": float(controls["relative_rms_max"].max()),
        "train_latent_mean_abs": float(train_latent.mean(dim=0).abs().mean()),
        "train_latent_std_mean": float(train_latent.std(dim=0, unbiased=False).mean()),
    }
    output = None
    if config.save:
        output = _save(
            config, encoder, model, history, paired, profile, controls, rollout, samples, summary
        )
    print("\nTeacher responsibility")
    print(profile[["mode", "timestep", "delta_shuffle_mean", "delta_shuffle_positive_rate"]].to_string(index=False))
    print("\nRollout")
    print(rollout.to_string(index=False))
    print("\nSummary")
    print(json.dumps(summary, indent=2))
    if output is not None:
        print(f"\nResults: {output}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-size", type=int, default=10_000)
    parser.add_argument("--test-size", type=int, default=2_000)
    parser.add_argument("--eval-count", type=int, default=1_000)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--steps", type=int, default=4_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--model-width", type=int, default=32)
    parser.add_argument("--model-depth", type=int, default=2)
    parser.add_argument("--ode-steps", type=int, default=50)
    parser.add_argument("--classifier-epochs", type=int, default=20)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        BottleneckConfig(
            mode=args.mode,
            output_root=args.output_root,
            train_size=args.train_size,
            test_size=args.test_size,
            eval_count=args.eval_count,
            latent_dim=args.latent_dim,
            steps=args.steps,
            batch_size=args.batch_size,
            model_width=args.model_width,
            model_depth=args.model_depth,
            ode_steps=args.ode_steps,
            classifier_epochs=args.classifier_epochs,
            seed=args.seed,
            device=args.device,
            save=not args.no_save,
        )
    )


if __name__ == "__main__":
    main()
