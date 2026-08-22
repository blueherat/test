#!/usr/bin/env python3
"""Audit matrix-free posterior-response projectors on trained SiT-S/2.

No manifold labels or oracle representation are used.  The audit compares the
response operator P_theta = t J_m of an EMA clean-prediction model and an EMA
velocity model converted to clean estimates on paired ImageNet-100 latents.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from experiments.posterior_response_projector import (
    posterior_response_action,
    posterior_response_basis,
)
from experiments.run_prediction_target_extrapolation_toy_v4 import parse_float_list
from experiments.run_prediction_target_rank_symmetry_toy import save_csv
from experiments.train_imagenet100_sit_flow import (
    LATENT_SHAPE,
    NUM_CLASSES,
    SD_VAE_SCALING_FACTOR,
    load_official_sit_module,
)


@dataclass
class SiTEstimator:
    name: str
    kind: str
    model: torch.nn.Module
    labels: torch.Tensor

    def _labels_for(self, batch: int) -> torch.Tensor:
        if batch == len(self.labels):
            return self.labels
        if batch % len(self.labels):
            raise ValueError("expanded estimator batch is not a multiple of labels")
        return self.labels.repeat_interleave(batch // len(self.labels))

    def clean(self, state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        prediction = self.model(state, time, self._labels_for(len(state))).float()
        if self.kind == "x":
            return prediction
        if self.kind == "v":
            remaining = (1.0 - time).reshape(-1, 1, 1, 1)
            return state.float() + remaining * prediction
        raise ValueError(self.kind)


def load_model(
    checkpoint_path: Path,
    *,
    official_sit_repo: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    sit_module, source_metadata = load_official_sit_module(
        official_sit_repo, verify_source=True
    )
    if checkpoint.get("official_sit") != source_metadata:
        raise ValueError("checkpoint and local official SiT source differ")
    model = sit_module.SiT_models[config["model_name"]](
        input_size=LATENT_SHAPE[-1],
        num_classes=NUM_CLASSES,
        class_dropout_prob=float(config["cfg_dropout"]),
    )
    model.load_state_dict(checkpoint["ema"], strict=True)
    model.to(device).eval().requires_grad_(False)
    return model, config


def load_validation_latents(
    *,
    cache_dir: Path,
    samples: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    moments = np.load(cache_dir / "validation_moments.npy", mmap_mode="r")
    labels = np.load(cache_dir / "validation_labels.npy", mmap_mode="r")
    if samples > len(moments):
        raise ValueError("requested more validation samples than available")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(moments), samples, replace=False)
    selected_moments = torch.from_numpy(np.asarray(moments[indices]).copy()).to(device)
    selected_labels = torch.from_numpy(np.asarray(labels[indices]).copy()).long().to(device)
    generator = torch.Generator(device=device.type).manual_seed(seed + 1)
    mean, std = selected_moments.chunk(2, dim=1)
    posterior_noise = torch.randn(mean.shape, device=device, generator=generator)
    data = (mean + std * posterior_noise) * SD_VAE_SCALING_FACTOR
    flow_noise = torch.randn(data.shape, device=device, generator=generator)
    return data, flow_noise, selected_labels


def flatten_norm(value: torch.Tensor) -> torch.Tensor:
    return value.float().flatten(1).norm(dim=1)


def row_cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(left.double().flatten(1), right.double().flatten(1), dim=1).float()


def summarize(value: torch.Tensor, prefix: str) -> dict[str, float]:
    value = value.detach().float().cpu().flatten()
    return {
        f"{prefix}_mean": float(value.mean()),
        f"{prefix}_median": float(value.median()),
        f"{prefix}_q10": float(torch.quantile(value, 0.1)),
        f"{prefix}_q90": float(torch.quantile(value, 0.9)),
    }


@torch.no_grad()
def source_metrics(
    *,
    source: SiTEstimator,
    state: torch.Tensor,
    time: torch.Tensor,
    direction: torch.Tensor,
    args: argparse.Namespace,
    generator: torch.Generator,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    alpha = time
    basis, singular_values = posterior_response_basis(
        source.clean,
        state=state,
        time=time,
        alpha=alpha,
        probes=args.probes,
        rank=min(args.report_rank, args.probes),
        relative_step=args.relative_step,
        generator=generator,
    )
    del basis
    squared = singular_values.square()
    normalized = squared / squared.sum(dim=1, keepdim=True).clamp_min(1e-12)
    effective_rank = normalized.square().sum(dim=1).reciprocal()
    response_gap_1_2 = singular_values[:, 0] / singular_values[:, 1].clamp_min(1e-12)
    response_gap_mid = (
        singular_values[:, args.report_rank - 1]
        / singular_values[:, args.report_rank].clamp_min(1e-12)
        if args.report_rank < args.probes
        else torch.full_like(response_gap_1_2, float("nan"))
    )
    action = posterior_response_action(
        source.clean,
        state=state,
        time=time,
        direction=direction,
        alpha=alpha,
        relative_step=args.relative_step,
    )
    reduce_dims = tuple(range(1, direction.ndim))
    direction_energy = direction.float().square().sum(
        dim=reduce_dims, keepdim=True
    ).clamp_min(1e-12)
    scalar_coefficient = (action.float() * direction.float()).sum(
        dim=reduce_dims, keepdim=True
    ) / direction_energy
    parallel_action = scalar_coefficient * direction.float()
    nonparallel_action = action.float() - parallel_action
    action_twice = posterior_response_action(
        source.clean,
        state=state,
        time=time,
        direction=action,
        alpha=alpha,
        relative_step=args.relative_step,
    )
    idempotence = flatten_norm(action_twice - action) / flatten_norm(action).clamp_min(1e-8)

    probe_a = torch.randn(direction.shape, device=state.device, generator=generator)
    probe_b = torch.randn(direction.shape, device=state.device, generator=generator)
    action_a = posterior_response_action(
        source.clean,
        state=state,
        time=time,
        direction=probe_a,
        alpha=alpha,
        relative_step=args.relative_step,
    )
    action_b = posterior_response_action(
        source.clean,
        state=state,
        time=time,
        direction=probe_b,
        alpha=alpha,
        relative_step=args.relative_step,
    )
    bilinear_ab = (probe_a.double() * action_b.double()).flatten(1).sum(dim=1)
    bilinear_ba = (action_a.double() * probe_b.double()).flatten(1).sum(dim=1)
    symmetry_scale = 0.5 * (
        flatten_norm(probe_a).double() * flatten_norm(action_b).double()
        + flatten_norm(probe_b).double() * flatten_norm(action_a).double()
    ).clamp_min(1e-8)
    symmetry = ((bilinear_ab - bilinear_ba).abs() / symmetry_scale).float()

    half_action = posterior_response_action(
        source.clean,
        state=state,
        time=time,
        direction=direction,
        alpha=alpha,
        relative_step=args.relative_step / 2.0,
    )
    double_action = posterior_response_action(
        source.clean,
        state=state,
        time=time,
        direction=direction,
        alpha=alpha,
        relative_step=args.relative_step * 2.0,
    )
    metrics = {
        "response_effective_rank": effective_rank,
        "response_gap_1_2": response_gap_1_2,
        "response_gap_report": response_gap_mid,
        "action_norm_ratio": flatten_norm(action) / flatten_norm(direction).clamp_min(1e-8),
        "action_input_cosine": row_cosine(action, direction),
        "action_scalar_coefficient": scalar_coefficient.flatten(1).mean(dim=1),
        "action_nonparallel_fraction": nonparallel_action.square().flatten(1).sum(dim=1)
        / action.float().square().flatten(1).sum(dim=1).clamp_min(1e-12),
        "idempotence_relative_error": idempotence,
        "symmetry_relative_error": symmetry,
        "step_half_cosine": row_cosine(action, half_action),
        "step_double_cosine": row_cosine(action, double_action),
        "step_half_relative_error": flatten_norm(action - half_action) / flatten_norm(action).clamp_min(1e-8),
        "step_double_relative_error": flatten_norm(action - double_action) / flatten_norm(action).clamp_min(1e-8),
    }
    for index in range(args.probes):
        metrics[f"response_singular_{index + 1}"] = singular_values[:, index]
    return metrics, action


def plot_metric(rows: list[dict], path: Path, metric: str, ylabel: str) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    for source in sorted({row["source"] for row in rows}):
        selected = sorted((row for row in rows if row["source"] == source), key=lambda row: row["time"])
        axis.plot([row["time"] for row in selected], [row[metric] for row in selected], marker="o", label=source)
    axis.set_xlabel("SiT interpolation time t (0=noise, 1=data)")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--v-checkpoint", type=Path, default=Path("/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/sit-s-2_seed0/checkpoints/step_00800000.pt"))
    parser.add_argument("--x-checkpoint", type=Path, default=Path("/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/sit-s-2_x-velocity-loss-floor0p05_seed0/checkpoints/step_00800000.pt"))
    parser.add_argument("--official-sit-repo", type=Path, default=Path("/home/zhoushunyu/data/research_repos/SiT"))
    parser.add_argument("--cache-dir", type=Path, default=Path("/home/zhoushunyu/data/eqvae/imagenet_sit_flow/imagenet100_cmc_sdvae"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--times", type=parse_float_list, default=parse_float_list("0.1,0.3,0.5,0.7,0.9,0.95"))
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--probes", type=int, default=8)
    parser.add_argument("--report-rank", type=int, default=4)
    parser.add_argument("--relative-step", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--device", default="cuda:3")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for field in ("v_checkpoint", "x_checkpoint", "official_sit_repo", "cache_dir", "output_root"):
        setattr(args, field, getattr(args, field).expanduser().resolve())
    args.output_root.mkdir(parents=True, exist_ok=True)
    if not (1 <= args.report_rank <= args.probes):
        raise ValueError("report_rank must lie in [1,probes]")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    v_model, v_config = load_model(args.v_checkpoint, official_sit_repo=args.official_sit_repo, device=device)
    x_model, x_config = load_model(args.x_checkpoint, official_sit_repo=args.official_sit_repo, device=device)
    if str(v_config.get("prediction_target", "velocity")) != "velocity":
        raise ValueError("v checkpoint is not a native velocity model")
    if str(x_config.get("prediction_target")) != "x":
        raise ValueError("x checkpoint is not a clean-prediction model")
    data, flow_noise, labels = load_validation_latents(
        cache_dir=args.cache_dir,
        samples=args.samples,
        seed=args.seed,
        device=device,
    )
    rows: list[dict] = []
    for time_index, time_value in enumerate(args.times):
        per_source: dict[str, list[dict[str, torch.Tensor]]] = {"x800": [], "v800_to_x": []}
        cross_cosines: list[torch.Tensor] = []
        for start in range(0, args.samples, args.batch_size):
            stop = min(start + args.batch_size, args.samples)
            batch_data = data[start:stop]
            batch_noise = flow_noise[start:stop]
            batch_labels = labels[start:stop]
            time = torch.full((len(batch_data),), float(time_value), device=device)
            time_image = time.reshape(-1, 1, 1, 1)
            state = (1.0 - time_image) * batch_noise + time_image * batch_data
            v_output = v_model(state, time, batch_labels).float()
            x_output = x_model(state, time, batch_labels).float()
            x_velocity = (x_output - state) / (1.0 - time_image).clamp_min(
                float(x_config.get("denominator_floor", 0.05))
            )
            direction = v_output - x_velocity
            actions: dict[str, torch.Tensor] = {}
            for source in (
                SiTEstimator("x800", "x", x_model, batch_labels),
                SiTEstimator("v800_to_x", "v", v_model, batch_labels),
            ):
                generator = torch.Generator(device=device.type).manual_seed(
                    args.seed + 100000 * time_index + 1000 * start
                )
                metrics, action = source_metrics(
                    source=source,
                    state=state,
                    time=time,
                    direction=direction,
                    args=args,
                    generator=generator,
                )
                per_source[source.name].append(metrics)
                actions[source.name] = action
            cross_cosines.append(row_cosine(actions["x800"], actions["v800_to_x"]))

        cross_summary = summarize(torch.cat(cross_cosines), "x_v_action_cosine")
        for source_name, batches in per_source.items():
            row = {
                "time": float(time_value),
                "source": source_name,
                "samples": args.samples,
                "probes": args.probes,
                "relative_step": args.relative_step,
                **cross_summary,
            }
            for key in batches[0]:
                row.update(summarize(torch.cat([batch[key] for batch in batches]), key))
            rows.append(row)
            print(
                f"[SiT t={time_value:g} {source_name}] "
                f"erank={row['response_effective_rank_mean']:.3f} "
                f"idem={row['idempotence_relative_error_median']:.3f} "
                f"stepcos={row['step_half_cosine_mean']:.4f} "
                f"cross={row['x_v_action_cosine_mean']:.3f}",
                flush=True,
            )
    save_csv(args.output_root / "sit_posterior_response_metrics.csv", rows)
    plot_metric(rows, args.output_root / "effective_rank.png", "response_effective_rank_mean", "probe-space effective rank")
    plot_metric(rows, args.output_root / "idempotence.png", "idempotence_relative_error_median", "median idempotence relative error")
    plot_metric(rows, args.output_root / "action_gain.png", "action_norm_ratio_median", "median ||P a|| / ||a||")
    manifest = {
        "definition": "P_theta a=t J_clean(z,t) a on SiT path z=(1-t)noise+t data",
        "weights": "EMA",
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "v_config": v_config,
        "x_config": x_config,
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[done] {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
