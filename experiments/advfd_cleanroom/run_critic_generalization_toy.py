#!/usr/bin/env python3
"""Stress-test adaptive Fréchet critics without a generator update.

The null setting uses independent real/fake samples from the same distribution;
any persistent train-only discrepancy is critic overfitting. The shift setting
contains a genuine population discrepancy and checks whether calibration erases
useful power. This is a paper-only mechanism test, not an AdvFD reproduction.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn

from experiments.advfd_cleanroom.core import (
    CalibrationMode,
    batch_moments,
    fit_calibration,
    frechet_from_features,
)


@dataclass(frozen=True)
class ToyConfig:
    train_samples: int = 256
    heldout_samples: int = 4096
    modes: int = 8
    radius: float = 2.0
    noise_std: float = 0.10
    hidden_dim: int = 128
    depth: int = 4
    feature_dim: int = 12
    steps: int = 1000
    eval_every: int = 25
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    whitening_epsilon: float = 1e-3


class FeatureCritic(nn.Module):
    def __init__(self, hidden_dim: int, depth: int, feature_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(2, hidden_dim), nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, feature_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_ring(
    count: int,
    *,
    seed: int,
    modes: int,
    radius: float,
    noise_std: float,
    angular_shift: float = 0.0,
    radial_scale: float = 1.0,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(modes, (count,), generator=generator)
    angles = indices.to(torch.float32) * (2.0 * math.pi / modes) + angular_shift
    centers = radial_scale * radius * torch.stack(
        [angles.cos(), angles.sin()], dim=1
    )
    noise = noise_std * torch.randn(count, 2, generator=generator)
    return centers + noise


def make_datasets(
    config: ToyConfig, *, seed: int, regime: str
) -> dict[str, torch.Tensor]:
    if regime not in {"matched", "shift"}:
        raise ValueError(f"Unknown regime: {regime}")
    shift = 0.0 if regime == "matched" else math.pi / (4.0 * config.modes)
    scale = 1.0 if regime == "matched" else 1.08
    return {
        "real_train": sample_ring(
            config.train_samples,
            seed=seed + 11,
            modes=config.modes,
            radius=config.radius,
            noise_std=config.noise_std,
        ),
        "fake_train": sample_ring(
            config.train_samples,
            seed=seed + 29,
            modes=config.modes,
            radius=config.radius,
            noise_std=config.noise_std,
            angular_shift=shift,
            radial_scale=scale,
        ),
        "real_heldout": sample_ring(
            config.heldout_samples,
            seed=seed + 101,
            modes=config.modes,
            radius=config.radius,
            noise_std=config.noise_std,
        ),
        "fake_heldout": sample_ring(
            config.heldout_samples,
            seed=seed + 211,
            modes=config.modes,
            radius=config.radius,
            noise_std=config.noise_std,
            angular_shift=shift,
            radial_scale=scale,
        ),
    }


def covariance_effective_rank(features: torch.Tensor) -> float:
    covariance = batch_moments(features).covariance
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    total = eigenvalues.sum()
    if total <= 0:
        return 0.0
    probabilities = eigenvalues / total
    entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum()
    return float(entropy.exp())


@torch.no_grad()
def evaluate(
    critic: nn.Module,
    datasets: dict[str, torch.Tensor],
    *,
    mode: CalibrationMode,
    epsilon: float,
) -> dict[str, float]:
    features = {name: critic(values) for name, values in datasets.items()}
    calibration = fit_calibration(
        features["real_train"],
        features["fake_train"],
        mode=mode,
        epsilon=epsilon,
    )
    train_real = calibration.apply(features["real_train"])
    train_fake = calibration.apply(features["fake_train"])
    heldout_real = calibration.apply(features["real_heldout"])
    heldout_fake = calibration.apply(features["fake_heldout"])
    train = frechet_from_features(train_real, train_fake)
    heldout = frechet_from_features(heldout_real, heldout_fake)

    heldout_calibration = fit_calibration(
        features["real_heldout"],
        features["fake_heldout"],
        mode=mode,
        epsilon=epsilon,
    )
    heldout_refit = frechet_from_features(
        heldout_calibration.apply(features["real_heldout"]),
        heldout_calibration.apply(features["fake_heldout"]),
    )
    parameter_norm = math.sqrt(
        sum(float(parameter.detach().square().sum()) for parameter in critic.parameters())
    )
    return {
        "train_fd": float(train.total),
        "train_mean_fd": float(train.mean),
        "train_covariance_fd": float(train.covariance),
        "heldout_fd": float(heldout.total),
        "heldout_mean_fd": float(heldout.mean),
        "heldout_covariance_fd": float(heldout.covariance),
        "heldout_refit_fd": float(heldout_refit.total),
        "raw_real_rms": float(features["real_train"].square().mean().sqrt()),
        "raw_fake_rms": float(features["fake_train"].square().mean().sqrt()),
        "calibrated_real_rms": float(train_real.square().mean().sqrt()),
        "calibrated_fake_rms": float(train_fake.square().mean().sqrt()),
        "feature_effective_rank": covariance_effective_rank(train_real),
        "parameter_norm": parameter_norm,
    }


def run_condition(
    config: ToyConfig,
    *,
    seed: int,
    regime: str,
    mode: CalibrationMode,
    device: torch.device,
) -> list[dict[str, float | int | str]]:
    seed_everything(seed)
    datasets = {
        name: values.to(device) for name, values in make_datasets(
            config, seed=seed, regime=regime
        ).items()
    }
    critic = FeatureCritic(
        config.hidden_dim, config.depth, config.feature_dim
    ).to(device)
    initial_state = copy.deepcopy(critic.state_dict())
    critic.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW(
        critic.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    rows: list[dict[str, float | int | str]] = []
    for step in range(config.steps + 1):
        if step % config.eval_every == 0 or step == config.steps:
            metrics = evaluate(
                critic,
                datasets,
                mode=mode,
                epsilon=config.whitening_epsilon,
            )
            rows.append(
                {"seed": seed, "regime": regime, "mode": mode, "step": step, **metrics}
            )
            if not all(math.isfinite(value) for value in metrics.values()):
                break
        if step == config.steps:
            break

        critic.train()
        optimizer.zero_grad(set_to_none=True)
        # AdvFD's paper protocol treats the real reference as detached in the
        # adaptive update. The shared critic still changes its real outputs when
        # parameters are updated through generated features.
        real_features = critic(datasets["real_train"]).detach()
        fake_features = critic(datasets["fake_train"])
        calibration = fit_calibration(
            real_features,
            fake_features,
            mode=mode,
            epsilon=config.whitening_epsilon,
            detach_statistics=True,
        )
        distance = frechet_from_features(
            calibration.apply(real_features),
            calibration.apply(fake_features),
        ).total
        (-distance).backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), config.gradient_clip)
        optimizer.step()

    return rows


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    final = frame.sort_values("step").groupby(
        ["regime", "mode", "seed"], as_index=False
    ).tail(1)
    final = final.copy()
    final["generalization_ratio"] = final["heldout_fd"] / final[
        "train_fd"
    ].clip(lower=1e-12)
    columns = [
        "regime",
        "mode",
        "seed",
        "step",
        "train_fd",
        "heldout_fd",
        "heldout_refit_fd",
        "generalization_ratio",
        "raw_real_rms",
        "raw_fake_rms",
        "calibrated_fake_rms",
        "feature_effective_rank",
        "parameter_norm",
    ]
    return final[columns].sort_values(["regime", "mode", "seed"])


def plot_curves(frame: pd.DataFrame, output: Path) -> None:
    regimes = sorted(frame["regime"].unique())
    figure, axes = plt.subplots(
        len(regimes), 2, figsize=(11, 4.2 * len(regimes)), squeeze=False
    )
    colors = {"none": "#8c2d2d", "real": "#2463a5", "pooled": "#2d7d46"}
    for row, regime in enumerate(regimes):
        selected = frame[frame["regime"] == regime]
        for mode in ("none", "real", "pooled"):
            mode_frame = selected[selected["mode"] == mode]
            grouped = mode_frame.groupby("step")
            steps = sorted(mode_frame["step"].unique())
            for metric, linestyle in (("train_fd", "-"), ("heldout_fd", "--")):
                means = grouped[metric].mean().reindex(steps)
                axes[row, 0].plot(
                    steps,
                    means,
                    color=colors[mode],
                    linestyle=linestyle,
                    label=f"{mode} {metric.replace('_fd', '')}",
                )
            rms = grouped["raw_fake_rms"].mean().reindex(steps)
            axes[row, 1].plot(
                steps, rms, color=colors[mode], label=mode
            )
        axes[row, 0].set_yscale("symlog", linthresh=1e-3)
        axes[row, 0].set_title(f"{regime}: adaptive FD")
        axes[row, 0].set_xlabel("critic step")
        axes[row, 0].set_ylabel("FD")
        axes[row, 0].legend(fontsize=8, ncol=2)
        axes[row, 1].set_yscale("log")
        axes[row, 1].set_title(f"{regime}: raw fake feature RMS")
        axes[row, 1].set_xlabel("critic step")
        axes[row, 1].set_ylabel("RMS")
        axes[row, 1].legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/data/users/zhoushunyu/eqvae/experiments/advfd_cleanroom_critic_toy"
        ),
    )
    parser.add_argument("--seeds", default="8101,8102,8103")
    parser.add_argument("--regimes", default="matched,shift")
    parser.add_argument("--modes", default="none,real,pooled")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--heldout-samples", type=int, default=4096)
    parser.add_argument("--device", default="cuda:3")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ToyConfig(
        train_samples=args.train_samples,
        heldout_samples=args.heldout_samples,
        steps=args.steps,
        eval_every=args.eval_every,
    )
    seeds = [int(item) for item in args.seeds.split(",") if item]
    regimes = [item for item in args.regimes.split(",") if item]
    modes: list[CalibrationMode] = [
        item for item in args.modes.split(",") if item  # type: ignore[misc]
    ]
    device = torch.device(args.device)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "config.json").write_text(
        json.dumps(
            {
                "config": asdict(config),
                "seeds": seeds,
                "regimes": regimes,
                "modes": modes,
                "device": str(device),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    rows: list[dict[str, float | int | str]] = []
    for regime in regimes:
        for seed in seeds:
            for mode in modes:
                print(f"running regime={regime} seed={seed} mode={mode}", flush=True)
                rows.extend(
                    run_condition(
                        config,
                        seed=seed,
                        regime=regime,
                        mode=mode,
                        device=device,
                    )
                )
                pd.DataFrame(rows).to_csv(args.output_root / "metrics.csv", index=False)

    frame = pd.DataFrame(rows)
    summary = summarize(frame)
    summary.to_csv(args.output_root / "summary.csv", index=False)
    plot_curves(frame, args.output_root / "critic_generalization.png")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
