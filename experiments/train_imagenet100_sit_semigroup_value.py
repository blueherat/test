#!/usr/bin/env python3
"""Train a soft-Bellman potential on frozen v800 + depth-4 Internal Guidance."""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from experiments import train_imagenet100_sit_flow as base
    from experiments.semigroup_guidance_value import (
        TokenPotentialHead,
        bellman_log_value_target,
        final_features,
        flow_time_from_heat_variance,
        flow_velocity_to_heat_score,
        source_weak_and_final_features,
        velocity_gap_to_heat_score_gap,
    )
    from experiments.train_imagenet100_sit_frozen_internal_v_head import (
        create_frozen_internal_probe,
    )
except ModuleNotFoundError:
    import train_imagenet100_sit_flow as base
    from semigroup_guidance_value import (
        TokenPotentialHead,
        bellman_log_value_target,
        final_features,
        flow_time_from_heat_variance,
        flow_velocity_to_heat_score,
        source_weak_and_final_features,
        velocity_gap_to_heat_score_gap,
    )
    from train_imagenet100_sit_frozen_internal_v_head import (
        create_frozen_internal_probe,
    )


PROTOCOL = "imagenet100_sit_semigroup_value_v1"
DEFAULT_SOURCE = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/"
    "sit-s-2_seed0/checkpoints/step_00800000.pt"
)
DEFAULT_WEAK = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/"
    "multiscale_guidance_study_v1/runs/depth4_v/checkpoints/step_00050000.pt"
)
DEFAULT_OUTPUT = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/"
    "semigroup_value_depth4_beta1p6_v1"
)


@dataclass(frozen=True)
class TrainConfig:
    cache_dir: str
    output_dir: str
    official_sit_repo: str
    source_checkpoint: str
    source_sha256: str
    weak_checkpoint: str
    weak_sha256: str
    model_name: str
    internal_depth: int
    beta: float
    intervention_time: float
    minimum_time: float
    heat_levels: int
    bellman_particles: int
    curriculum_steps_per_level: int
    global_batch_size: int
    max_steps: int
    learning_rate: float
    weight_decay: float
    target_decay: float
    precision: str
    allow_tf32: bool
    num_workers: int
    prefetch_factor: int
    log_every: int
    save_every: int
    seed: int


def broadcast_string(value: str, context: base.DistributedContext) -> str:
    if context.world_size == 1:
        return value
    values: list[str | None] = [value if context.is_main else None]
    dist.broadcast_object_list(values, src=0)
    if not isinstance(values[0], str):
        raise RuntimeError("failed to broadcast string")
    return values[0]


def geometric_heat_levels(config: TrainConfig, device: torch.device) -> torch.Tensor:
    start = ((1.0 - config.intervention_time) / config.intervention_time) ** 2
    end = ((1.0 - config.minimum_time) / config.minimum_time) ** 2
    return torch.logspace(
        math.log10(start),
        math.log10(end),
        config.heat_levels + 1,
        device=device,
        dtype=torch.float32,
    )


def load_models(
    *,
    config: TrainConfig,
    source_payload: dict,
    weak_payload: dict,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module, TokenPotentialHead, dict]:
    sit_module, source_metadata = base.load_official_sit_module(
        Path(config.official_sit_repo),
        verify_source=True,
    )
    if source_payload.get("official_sit") != source_metadata:
        raise ValueError("source checkpoint official SiT metadata mismatch")
    source, weak, architecture = create_frozen_internal_probe(
        sit_module,
        model_name=config.model_name,
        cfg_dropout=float(source_payload.get("config", {}).get("cfg_dropout", 0.1)),
        source_state=source_payload["ema"],
        internal_depth=config.internal_depth,
    )
    weak_config = weak_payload.get("config", {})
    if int(weak_config.get("internal_depth", -1)) != config.internal_depth:
        raise ValueError("weak checkpoint depth mismatch")
    if str(weak_config.get("prediction_target")) != "velocity":
        raise ValueError("semigroup value currently requires a velocity weak head")
    if str(weak_config.get("source_checkpoint_sha256")) != config.source_sha256:
        raise ValueError("weak head was not trained on the selected source checkpoint")
    weak.load_state_dict(weak_payload["internal_head_ema"], strict=True)
    for parameter in weak.parameters():
        parameter.requires_grad_(False)
    weak.eval()
    hidden_size = int(source.pos_embed.shape[-1])
    value = TokenPotentialHead(
        hidden_size,
        intervention_time=config.intervention_time,
    )
    architecture.update(
        {
            "potential_parameter_count": sum(
                parameter.numel() for parameter in value.parameters()
            ),
            "potential_hidden_size": hidden_size,
        }
    )
    return source.to(device).eval(), weak.to(device).eval(), value.to(device), architecture


def train(args: argparse.Namespace) -> None:
    context = base.initialize_distributed(args.device)
    try:
        if args.global_batch_size % context.world_size:
            raise ValueError("global batch size must be divisible by world size")
        source_path = args.source_checkpoint.expanduser().resolve()
        weak_path = args.weak_checkpoint.expanduser().resolve()
        for path in (source_path, weak_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        source_hash = broadcast_string(
            base.sha256_file(source_path) if context.is_main else "",
            context,
        )
        weak_hash = broadcast_string(
            base.sha256_file(weak_path) if context.is_main else "",
            context,
        )
        source_payload = torch.load(
            source_path, map_location="cpu", weights_only=False, mmap=True
        )
        weak_payload = torch.load(
            weak_path, map_location="cpu", weights_only=False, mmap=True
        )
        model_name = str(source_payload.get("config", {}).get("model_name", "SiT-S/2"))
        config = TrainConfig(
            cache_dir=str(args.cache_dir.expanduser().resolve()),
            output_dir=str(args.output_dir.expanduser().resolve()),
            official_sit_repo=str(args.official_sit_repo.expanduser().resolve()),
            source_checkpoint=str(source_path),
            source_sha256=source_hash,
            weak_checkpoint=str(weak_path),
            weak_sha256=weak_hash,
            model_name=model_name,
            internal_depth=int(args.internal_depth),
            beta=float(args.beta),
            intervention_time=float(args.intervention_time),
            minimum_time=float(args.minimum_time),
            heat_levels=int(args.heat_levels),
            bellman_particles=int(args.bellman_particles),
            curriculum_steps_per_level=int(args.curriculum_steps_per_level),
            global_batch_size=int(args.global_batch_size),
            max_steps=int(args.max_steps),
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            target_decay=float(args.target_decay),
            precision=str(args.precision),
            allow_tf32=bool(args.allow_tf32),
            num_workers=int(args.num_workers),
            prefetch_factor=int(args.prefetch_factor),
            log_every=int(args.log_every),
            save_every=int(args.save_every),
            seed=int(args.seed),
        )
        if not (
            config.beta > 1.0
            and 0.0 < config.minimum_time < config.intervention_time < 1.0
            and config.heat_levels > 1
            and config.bellman_particles > 0
            and 0.0 <= config.target_decay < 1.0
        ):
            raise ValueError("invalid semigroup value configuration")
        base.configure_runtime(config.seed, context.rank, config.allow_tf32)
        local_batch = config.global_batch_size // context.world_size
        output_dir = Path(config.output_dir)
        cache_dir = Path(config.cache_dir)
        manifest_path = cache_dir / "manifest.json"
        cache_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_hash = base.sha256_file(manifest_path)

        source, weak, value, architecture = load_models(
            config=config,
            source_payload=source_payload,
            weak_payload=weak_payload,
            device=context.device,
        )
        del source_payload, weak_payload
        gc.collect()
        target = base.ModelEMA(value)
        optimizer = torch.optim.AdamW(
            value.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=config.weight_decay,
            fused=True,
        )
        train_value: torch.nn.Module = value
        if context.world_size > 1:
            train_value = DDP(
                value,
                device_ids=[context.local_rank],
                output_device=context.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
                static_graph=True,
            )

        loader, sampler = base.create_loader(
            cache_dir=cache_dir,
            split="train",
            local_batch_size=local_batch,
            context=context,
            seed=config.seed,
            shuffle=True,
            num_workers=config.num_workers,
            prefetch_factor=config.prefetch_factor,
            drop_last=True,
        )
        batches = base.infinite_train_batches(loader, sampler, 0)
        levels = geometric_heat_levels(config, context.device)

        if context.is_main:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "checkpoints").mkdir(exist_ok=True)
            base.atomic_json_dump(
                {
                    "protocol": PROTOCOL,
                    "config": asdict(config),
                    "world_size": context.world_size,
                    "architecture": architecture,
                    "data_manifest_sha256": manifest_hash,
                    "objective": {
                        "intervention_density": "p_tstar^beta*q_tstar^(1-beta)",
                        "fixed_point": (
                            "delta(tau+dt,y)=log E exp[c*dt+delta(tau,Y')]"
                        ),
                        "running_cost": (
                            "0.5*beta*(beta-1)*||score_strong-score_weak||^2"
                        ),
                        "base_guidance_gamma": config.beta - 1.0,
                        "extra_inference_gain": 0,
                    },
                    "heat_levels": levels.cpu().tolist(),
                },
                output_dir / "run_config.json",
            )
            print(
                json.dumps(
                    {
                        "event": "start",
                        "world_size": context.world_size,
                        "global_batch": config.global_batch_size,
                        "local_batch": local_batch,
                        "potential_parameters": architecture[
                            "potential_parameter_count"
                        ],
                        "heat_levels": config.heat_levels,
                        "beta": config.beta,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        base.barrier(context)

        metrics_path = output_dir / "train_metrics.jsonl"
        running = torch.zeros(8, device=context.device, dtype=torch.float64)
        running_steps = 0
        interval_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        latent_dim = math.prod(base.LATENT_SHAPE)

        for step in range(1, config.max_steps + 1):
            moments, labels = next(batches)
            moments = moments.to(context.device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(context.device, dtype=torch.long, non_blocking=True)
            clean = base.sample_sdvae_posterior(
                moments,
                torch.randn((len(moments), *base.LATENT_SHAPE), device=context.device),
            )
            max_level = min(
                config.heat_levels,
                1 + (step - 1) // config.curriculum_steps_per_level,
            )
            level_index = torch.randint(
                1,
                max_level + 1,
                (len(clean),),
                device=context.device,
            )
            tau_noisy = levels[level_index]
            tau_clean = levels[level_index - 1]
            heat_step = tau_noisy - tau_clean
            time_noisy = flow_time_from_heat_variance(tau_noisy)
            time_clean = flow_time_from_heat_variance(tau_clean)
            endpoint_noise = torch.randn_like(clean)
            y_state = clean + _batch_scale(tau_noisy.sqrt(), clean) * endpoint_noise
            lhs_state = _batch_scale(time_noisy, clean) * y_state
            query_state = _batch_scale(time_clean, clean) * y_state

            with torch.no_grad(), base.autocast_context(config.precision):
                strong, weak_output, _, _ = source_weak_and_final_features(
                    source,
                    weak,
                    query_state,
                    time_clean,
                    labels,
                    internal_depth=config.internal_depth,
                    latent_channels=base.LATENT_SHAPE[0],
                )
            strong = strong.float()
            weak_output = weak_output.float()
            velocity_gap = strong - weak_output
            static_velocity = weak_output + config.beta * velocity_gap
            static_score = flow_velocity_to_heat_score(
                static_velocity,
                state=query_state,
                time_value=time_clean,
            )
            score_gap = velocity_gap_to_heat_score_gap(
                velocity_gap,
                time_value=time_clean,
            )
            running_cost = (
                0.5
                * config.beta
                * (config.beta - 1.0)
                * score_gap.flatten(1).square().sum(dim=1)
            )

            particle_noise = torch.randn(
                (
                    config.bellman_particles,
                    len(clean),
                    *base.LATENT_SHAPE,
                ),
                device=context.device,
            )
            rhs_y = (
                y_state.unsqueeze(0)
                + _batch_scale(heat_step, clean).unsqueeze(0)
                * static_score.unsqueeze(0)
                + _batch_scale(heat_step.sqrt(), clean).unsqueeze(0)
                * particle_noise
            )
            rhs_state = _batch_scale(time_clean, clean).unsqueeze(0) * rhs_y
            feature_states = torch.cat(
                (lhs_state, rhs_state.flatten(0, 1)),
                dim=0,
            )
            feature_times = torch.cat(
                (
                    time_noisy,
                    time_clean.repeat(config.bellman_particles),
                )
            )
            feature_labels = torch.cat(
                (labels, labels.repeat(config.bellman_particles))
            )
            with torch.no_grad(), base.autocast_context(config.precision):
                tokens, conditioning = final_features(
                    source,
                    feature_states,
                    feature_times,
                    feature_labels,
                )
            lhs_tokens = tokens[: len(clean)].detach()
            lhs_conditioning = conditioning[: len(clean)].detach()
            rhs_tokens = tokens[len(clean) :].detach()
            rhs_conditioning = conditioning[len(clean) :].detach()
            with base.autocast_context(config.precision):
                prediction = train_value(
                    lhs_tokens,
                    lhs_conditioning,
                    time_noisy,
                ).float()
                with torch.no_grad():
                    next_values = target.module(
                        rhs_tokens,
                        rhs_conditioning,
                        time_clean.repeat(config.bellman_particles),
                    ).float().reshape(config.bellman_particles, len(clean))
                    bellman_target = bellman_log_value_target(
                        next_values,
                        running_cost=running_cost,
                        heat_step=heat_step,
                    )
            loss = F.mse_loss(prediction, bellman_target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite Bellman loss at step {step}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(value.parameters(), 10.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            target.update(config.target_decay)

            with torch.no_grad():
                running += torch.tensor(
                    [
                        float(loss.item()),
                        float(prediction.mean().item()),
                        float(prediction.std().item()),
                        float(bellman_target.mean().item()),
                        float(bellman_target.std().item()),
                        float(running_cost.mean().item()),
                        float(heat_step.mean().item()),
                        float(grad_norm.item()),
                    ],
                    device=context.device,
                    dtype=torch.float64,
                )
                running_steps += 1

            if step % config.log_every == 0 or step == config.max_steps:
                torch.cuda.synchronize(context.device)
                elapsed = time.perf_counter() - interval_start
                values = torch.cat(
                    (
                        running,
                        torch.tensor(
                            [running_steps],
                            device=context.device,
                            dtype=torch.float64,
                        ),
                    )
                )
                base.reduce_sum(values, context)
                count = values[-1]
                row = {
                    "step": step,
                    "max_curriculum_level": max_level,
                    "bellman_loss": float(values[0].item() / count.item()),
                    "prediction_mean": float(values[1].item() / count.item()),
                    "prediction_std": float(values[2].item() / count.item()),
                    "target_mean": float(values[3].item() / count.item()),
                    "target_std": float(values[4].item() / count.item()),
                    "running_cost_mean": float(values[5].item() / count.item()),
                    "heat_step_mean": float(values[6].item() / count.item()),
                    "gradient_norm": float(values[7].item() / count.item()),
                    "steps_per_second": running_steps / elapsed,
                    "images_per_second": running_steps
                    * config.global_batch_size
                    / elapsed,
                    "max_allocated_gb": torch.cuda.max_memory_allocated(
                        context.device
                    )
                    / 2**30,
                    "latent_dim": latent_dim,
                }
                if context.is_main:
                    with metrics_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
                    print(json.dumps(row, sort_keys=True), flush=True)
                running.zero_()
                running_steps = 0
                interval_start = time.perf_counter()
                torch.cuda.reset_peak_memory_stats(context.device)

            if step % config.save_every == 0 or step == config.max_steps:
                rng_states = base.gather_rng_states(context)
                if context.is_main:
                    checkpoint = output_dir / "checkpoints" / f"step_{step:08d}.pt"
                    base.atomic_torch_save(
                        {
                            "protocol": PROTOCOL,
                            "step": step,
                            "potential": value.state_dict(),
                            "potential_ema": target.module.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "rng_states": rng_states,
                            "config": {**asdict(config), "world_size": context.world_size},
                            "data_manifest_sha256": manifest_hash,
                        },
                        checkpoint,
                    )
                    print(
                        json.dumps({"event": "checkpoint", "path": str(checkpoint)}),
                        flush=True,
                    )
                base.barrier(context)
    finally:
        base.cleanup_distributed(context)


def _batch_scale(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    while value.ndim < target.ndim:
        value = value.unsqueeze(-1)
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=base.DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--official-sit-repo", type=Path, default=base.DEFAULT_OFFICIAL_SIT_REPO)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--weak-checkpoint", type=Path, default=DEFAULT_WEAK)
    parser.add_argument("--internal-depth", type=int, default=4)
    parser.add_argument("--beta", type=float, default=1.6)
    parser.add_argument("--intervention-time", type=float, default=0.5)
    parser.add_argument("--minimum-time", type=float, default=0.02)
    parser.add_argument("--heat-levels", type=int, default=128)
    parser.add_argument("--bellman-particles", type=int, default=2)
    parser.add_argument("--curriculum-steps-per-level", type=int, default=20)
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=5_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--target-decay", type=float, default=0.95)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
