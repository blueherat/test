#!/usr/bin/env python3
"""Train a diagonal posterior-response head on frozen ImageNet-100 SiTs."""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from experiments import train_imagenet100_sit_flow as base
from experiments.imagenet100_sit_internal_v_head import (
    FrozenPrefix,
    extract_internal_features,
    freeze_source_model,
    full_velocity_from_features,
    validate_internal_depth,
)
from experiments.imagenet100_sit_posterior_response_head import (
    create_diagonal_response_head,
    diagonal_response_action,
    diagonal_response_gain,
    finite_difference_clean_response_action,
    rademacher_probe_like,
)


PROTOCOL = "imagenet100_sit_diagonal_posterior_response_head_v1"
DEFAULT_V_CHECKPOINT = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/"
    "sit-s-2_seed0/checkpoints/step_00800000.pt"
)
DEFAULT_X_CHECKPOINT = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/"
    "sit-s-2_x-velocity-loss-floor0p05_seed0/checkpoints/step_00800000.pt"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/"
    "sit-s-2_v800-x800_diagonal-posterior-response-depth8_seed0"
)


@dataclass(frozen=True)
class TrainConfig:
    cache_dir: str
    output_dir: str
    official_sit_repo: str
    v_checkpoint: str
    x_checkpoint: str
    v_checkpoint_sha256: str
    x_checkpoint_sha256: str
    model_name: str
    cfg_dropout: float
    internal_depth: int
    global_batch_size: int
    max_steps: int
    learning_rate: float
    weight_decay: float
    ema_decay: float
    relative_step: float
    time_min: float
    time_max: float
    precision: str
    allow_tf32: bool
    num_workers: int
    prefetch_factor: int
    log_every: int
    validation_every: int
    validation_batches: int
    save_every: int
    seed: int


def _broadcast_string(value: str, context: base.DistributedContext) -> str:
    if context.world_size == 1:
        return value
    values: list[str | None] = [value if context.is_main else None]
    dist.broadcast_object_list(values, src=0)
    if not isinstance(values[0], str):
        raise RuntimeError("failed to broadcast checkpoint hash")
    return values[0]


def _load_model(
    sit_module,
    *,
    payload: dict,
    model_name: str,
    cfg_dropout: float,
    state_key: str,
) -> torch.nn.Module:
    model = sit_module.SiT_models[model_name](
        input_size=base.LATENT_SHAPE[-1],
        num_classes=base.NUM_CLASSES,
        class_dropout_prob=cfg_dropout,
    )
    model.load_state_dict(payload[state_key], strict=True)
    return freeze_source_model(model)


def _row_cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(
        left.double().flatten(1), right.double().flatten(1), dim=1
    ).float()


def _relative_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    numerator = (prediction.float() - target.float()).flatten(1).norm(dim=1)
    denominator = target.float().flatten(1).norm(dim=1).clamp_min(1e-8)
    return numerator / denominator


@torch.inference_mode()
def validate(
    *,
    source: torch.nn.Module,
    clean_teacher: torch.nn.Module,
    head: torch.nn.Module,
    loader,
    context: base.DistributedContext,
    config: TrainConfig,
    batches: int,
    seed: int,
    x_denominator_floor: float,
) -> dict[str, float]:
    names = (
        "random_mse",
        "random_cosine",
        "random_relative_error",
        "task_mse",
        "task_cosine",
        "task_relative_error",
        "gain_mean",
        "gain_std",
    )
    totals = torch.zeros(len(names) + 1, device=context.device, dtype=torch.float64)
    generator = torch.Generator(device=context.device).manual_seed(seed)
    source.eval()
    clean_teacher.eval()
    head.eval()
    for batch_index, (moments, labels) in enumerate(loader):
        if batch_index >= batches:
            break
        moments = moments.to(context.device, dtype=torch.float32, non_blocking=True)
        labels = labels.to(context.device, dtype=torch.long, non_blocking=True)
        posterior_noise = torch.randn(
            (len(moments), *base.LATENT_SHAPE),
            generator=generator,
            device=context.device,
        )
        clean = base.sample_sdvae_posterior(moments, posterior_noise)
        noise = torch.randn(clean.shape, generator=generator, device=context.device)
        time_value = torch.empty(len(clean), device=context.device).uniform_(
            config.time_min,
            config.time_max,
            generator=generator,
        )
        state, _target_velocity = base.linear_flow_state_target(
            clean, noise, time_value
        )
        with base.autocast_context(config.precision):
            features, conditioning = extract_internal_features(
                source,
                state,
                time_value,
                labels,
                internal_depth=config.internal_depth,
            )
            full_velocity = full_velocity_from_features(
                source,
                features,
                conditioning,
                internal_depth=config.internal_depth,
                latent_channels=base.LATENT_SHAPE[0],
            ).float()
            gain = diagonal_response_gain(
                source,
                head,
                features,
                conditioning,
                latent_channels=base.LATENT_SHAPE[0],
            )

        probe = torch.empty_like(state).bernoulli_(0.5, generator=generator)
        probe.mul_(2.0).sub_(1.0)
        random_teacher = finite_difference_clean_response_action(
            clean_teacher,
            state=state,
            time_value=time_value,
            labels=labels,
            direction=probe,
            alpha=time_value,
            relative_step=config.relative_step,
        )
        random_prediction = diagonal_response_action(gain, probe)

        x_prediction = clean_teacher(state, time_value, labels).float()
        remaining = (1.0 - time_value).reshape(-1, 1, 1, 1)
        x_velocity = (x_prediction - state) / remaining.clamp_min(
            x_denominator_floor
        )
        task_direction = full_velocity - x_velocity
        task_teacher = finite_difference_clean_response_action(
            clean_teacher,
            state=state,
            time_value=time_value,
            labels=labels,
            direction=task_direction,
            alpha=time_value,
            relative_step=config.relative_step,
        )
        task_prediction = diagonal_response_action(gain, task_direction)

        values = torch.stack(
            (
                (random_prediction - random_teacher).square().flatten(1).mean(1),
                _row_cosine(random_prediction, random_teacher),
                _relative_error(random_prediction, random_teacher),
                (task_prediction - task_teacher).square().flatten(1).mean(1),
                _row_cosine(task_prediction, task_teacher),
                _relative_error(task_prediction, task_teacher),
                gain.flatten(1).mean(1),
                gain.flatten(1).std(1),
            ),
            dim=1,
        ).double()
        totals[:-1] += values.sum(dim=0)
        totals[-1] += len(values)

    base.reduce_sum(totals, context)
    if totals[-1].item() == 0:
        raise RuntimeError("validation loader produced no samples")
    means = totals[:-1] / totals[-1]
    return {
        name: float(value.item())
        for name, value in zip(names, means, strict=True)
    }


def train(args: argparse.Namespace) -> None:
    context = base.initialize_distributed(args.device)
    try:
        if args.global_batch_size % context.world_size:
            raise ValueError("global batch size must be divisible by world size")
        if not (0.0 < args.time_min < args.time_max < 1.0):
            raise ValueError("require 0 < time_min < time_max < 1")
        if args.relative_step <= 0:
            raise ValueError("relative step must be positive")
        base.configure_runtime(args.seed, context.rank, args.allow_tf32)

        v_path = args.v_checkpoint.expanduser().resolve()
        x_path = args.x_checkpoint.expanduser().resolve()
        for path in (v_path, x_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        v_hash = _broadcast_string(
            base.sha256_file(v_path) if context.is_main else "", context
        )
        x_hash = _broadcast_string(
            base.sha256_file(x_path) if context.is_main else "", context
        )
        v_payload = torch.load(v_path, map_location="cpu", weights_only=False, mmap=True)
        x_payload = torch.load(x_path, map_location="cpu", weights_only=False, mmap=True)
        v_config = v_payload["config"]
        x_config = x_payload["config"]
        model_name = str(v_config["model_name"])
        cfg_dropout = float(v_config["cfg_dropout"])
        if model_name != str(x_config["model_name"]):
            raise ValueError("v and x checkpoints use different architectures")
        if str(v_config.get("prediction_target", "velocity")) != "velocity":
            raise ValueError("v checkpoint is not a velocity model")
        if str(x_config.get("prediction_target")) != "x":
            raise ValueError("x checkpoint is not a clean-prediction model")
        for key in ("data_manifest_sha256", "official_sit"):
            if v_payload.get(key) != x_payload.get(key):
                raise ValueError(f"v and x checkpoint {key} differ")

        output_dir = args.output_dir.expanduser().resolve()
        cache_dir = args.cache_dir.expanduser().resolve()
        config = TrainConfig(
            cache_dir=str(cache_dir),
            output_dir=str(output_dir),
            official_sit_repo=str(args.official_sit_repo.expanduser().resolve()),
            v_checkpoint=str(v_path),
            x_checkpoint=str(x_path),
            v_checkpoint_sha256=v_hash,
            x_checkpoint_sha256=x_hash,
            model_name=model_name,
            cfg_dropout=cfg_dropout,
            internal_depth=int(args.internal_depth),
            global_batch_size=int(args.global_batch_size),
            max_steps=int(args.max_steps),
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            ema_decay=float(args.ema_decay),
            relative_step=float(args.relative_step),
            time_min=float(args.time_min),
            time_max=float(args.time_max),
            precision=str(args.precision),
            allow_tf32=bool(args.allow_tf32),
            num_workers=int(args.num_workers),
            prefetch_factor=int(args.prefetch_factor),
            log_every=int(args.log_every),
            validation_every=int(args.validation_every),
            validation_batches=int(args.validation_batches),
            save_every=int(args.save_every),
            seed=int(args.seed),
        )
        if min(
            config.global_batch_size,
            config.max_steps,
            config.learning_rate,
            config.log_every,
            config.save_every,
        ) <= 0:
            raise ValueError("batch, steps, intervals, and learning rate must be positive")

        sit_module, source_metadata = base.load_official_sit_module(
            Path(config.official_sit_repo), verify_source=args.verify_sit_source
        )
        if v_payload.get("official_sit") != source_metadata:
            raise ValueError("checkpoint and official SiT source differ")
        source = _load_model(
            sit_module,
            payload=v_payload,
            model_name=model_name,
            cfg_dropout=cfg_dropout,
            state_key="ema",
        )
        clean_teacher = _load_model(
            sit_module,
            payload=x_payload,
            model_name=model_name,
            cfg_dropout=cfg_dropout,
            state_key="ema",
        )
        depth = validate_internal_depth(source, config.internal_depth)
        head = create_diagonal_response_head(
            sit_module,
            source,
            latent_channels=base.LATENT_SHAPE[0],
        )
        del v_payload, x_payload
        gc.collect()
        source = source.to(context.device).eval()
        clean_teacher = clean_teacher.to(context.device).eval()
        head = head.to(context.device)
        ema = base.ModelEMA(head)
        optimizer = torch.optim.AdamW(
            head.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            fused=True,
        )
        train_head: torch.nn.Module = head
        if context.world_size > 1:
            train_head = DDP(
                head,
                device_ids=[context.local_rank],
                output_device=context.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
                static_graph=True,
            )
        ema.module.load_state_dict(head.state_dict())
        prefix = FrozenPrefix(source, depth).eval()

        local_batch_size = config.global_batch_size // context.world_size
        train_loader, train_sampler = base.create_loader(
            cache_dir=cache_dir,
            split="train",
            local_batch_size=local_batch_size,
            context=context,
            seed=config.seed,
            shuffle=True,
            num_workers=config.num_workers,
            prefetch_factor=config.prefetch_factor,
            drop_last=True,
        )
        validation_loader, validation_sampler = base.create_loader(
            cache_dir=cache_dir,
            split="validation",
            local_batch_size=local_batch_size,
            context=context,
            seed=config.seed + 1,
            shuffle=False,
            num_workers=max(0, min(2, config.num_workers)),
            prefetch_factor=config.prefetch_factor,
            drop_last=False,
        )
        validation_sampler.set_epoch(0)
        batches = base.infinite_train_batches(train_loader, train_sampler, 0)
        x_denominator_floor = float(x_config.get("denominator_floor", 0.05))

        if context.is_main:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            cache_manifest_path = cache_dir / "manifest.json"
            base.atomic_json_dump(
                {
                    "protocol": PROTOCOL,
                    "config": asdict(config),
                    "world_size": context.world_size,
                    "official_sit": source_metadata,
                    "data_manifest_sha256": base.sha256_file(cache_manifest_path),
                    "trainable_parameters": sum(p.numel() for p in head.parameters()),
                    "objective": {
                        "teacher": "t * J_x800(state,t,label) * random_rademacher_probe",
                        "student": "sigmoid(diagonal_head(v800_depth8_features)) * probe",
                        "loss": "mean squared action error",
                        "source_and_teacher_frozen": True,
                    },
                },
                output_dir / "run_config.json",
            )
            print(
                json.dumps(
                    {
                        "event": "start",
                        "world_size": context.world_size,
                        "global_batch": config.global_batch_size,
                        "local_batch": local_batch_size,
                        "internal_depth": depth,
                        "trainable_parameters": sum(
                            p.numel() for p in head.parameters()
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        base.barrier(context)

        metrics_path = output_dir / "train_metrics.jsonl"
        if config.validation_every > 0:
            initial_metrics = validate(
                source=source,
                clean_teacher=clean_teacher,
                head=head,
                loader=validation_loader,
                context=context,
                config=config,
                batches=config.validation_batches,
                seed=config.seed + 900_000,
                x_denominator_floor=x_denominator_floor,
            )
            if context.is_main:
                row = {
                    "step": 0,
                    "raw_validation": initial_metrics,
                    "ema_validation": initial_metrics,
                }
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                print(json.dumps(row, sort_keys=True), flush=True)
            head.train()

        running_loss = torch.zeros(1, device=context.device, dtype=torch.float64)
        running_steps = 0
        interval_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)

        for step in range(1, config.max_steps + 1):
            moments, labels = next(batches)
            moments = moments.to(context.device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(context.device, dtype=torch.long, non_blocking=True)
            posterior_noise = torch.randn(
                (len(moments), *base.LATENT_SHAPE), device=context.device
            )
            clean = base.sample_sdvae_posterior(moments, posterior_noise)
            noise = torch.randn_like(clean)
            time_value = torch.empty(len(clean), device=context.device).uniform_(
                config.time_min, config.time_max
            )
            state, _target_velocity = base.linear_flow_state_target(
                clean, noise, time_value
            )
            probe = rademacher_probe_like(state)

            with torch.no_grad():
                teacher_action = finite_difference_clean_response_action(
                    clean_teacher,
                    state=state,
                    time_value=time_value,
                    labels=labels,
                    direction=probe,
                    alpha=time_value,
                    relative_step=config.relative_step,
                )
                with base.autocast_context(config.precision):
                    features, conditioning = prefix(state, time_value, labels)
            with base.autocast_context(config.precision):
                gain = diagonal_response_gain(
                    source,
                    train_head,
                    features.detach(),
                    conditioning.detach(),
                    latent_channels=base.LATENT_SHAPE[0],
                )
                prediction = diagonal_response_action(gain, probe)
                loss = F.mse_loss(prediction, teacher_action.float())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            ema.update(config.ema_decay)
            running_loss += loss.detach().double()
            running_steps += 1

            if step % config.log_every == 0 or step == config.max_steps:
                torch.cuda.synchronize(context.device)
                elapsed = time.perf_counter() - interval_started
                values = torch.cat(
                    (
                        running_loss,
                        torch.tensor(
                            [running_steps],
                            device=context.device,
                            dtype=torch.float64,
                        ),
                    )
                )
                base.reduce_sum(values, context)
                memory = torch.tensor(
                    torch.cuda.max_memory_allocated(context.device) / 2**30,
                    device=context.device,
                )
                base.reduce_max(memory, context)
                row = {
                    "step": step,
                    "train_action_mse": float((values[0] / values[1]).item()),
                    "steps_per_second": float(running_steps / elapsed),
                    "images_per_second": float(
                        running_steps * config.global_batch_size / elapsed
                    ),
                    "max_allocated_gb": float(memory.item()),
                }
                if context.is_main:
                    with metrics_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
                    print(json.dumps(row, sort_keys=True), flush=True)
                running_loss.zero_()
                running_steps = 0
                interval_started = time.perf_counter()
                torch.cuda.reset_peak_memory_stats(context.device)

            should_validate = config.validation_every > 0 and (
                step % config.validation_every == 0 or step == config.max_steps
            )
            if should_validate:
                raw_metrics = validate(
                    source=source,
                    clean_teacher=clean_teacher,
                    head=head,
                    loader=validation_loader,
                    context=context,
                    config=config,
                    batches=config.validation_batches,
                    seed=config.seed + 900_000,
                    x_denominator_floor=x_denominator_floor,
                )
                ema_metrics = validate(
                    source=source,
                    clean_teacher=clean_teacher,
                    head=ema.module,
                    loader=validation_loader,
                    context=context,
                    config=config,
                    batches=config.validation_batches,
                    seed=config.seed + 900_000,
                    x_denominator_floor=x_denominator_floor,
                )
                if context.is_main:
                    row = {
                        "step": step,
                        "raw_validation": raw_metrics,
                        "ema_validation": ema_metrics,
                    }
                    with metrics_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
                    print(json.dumps(row, sort_keys=True), flush=True)
                head.train()

            if step % config.save_every == 0 or step == config.max_steps:
                rng_states = base.gather_rng_states(context)
                if context.is_main:
                    checkpoint_path = (
                        output_dir / "checkpoints" / f"step_{step:08d}.pt"
                    )
                    base.atomic_torch_save(
                        {
                            "protocol": PROTOCOL,
                            "step": step,
                            "diagonal_head": head.state_dict(),
                            "diagonal_head_ema": ema.module.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "rng_states": rng_states,
                            "config": {**asdict(config), "world_size": context.world_size},
                            "official_sit": source_metadata,
                        },
                        checkpoint_path,
                    )
                    print(
                        json.dumps({"event": "checkpoint", "path": str(checkpoint_path)}),
                        flush=True,
                    )
                base.barrier(context)
    finally:
        base.cleanup_distributed(context)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=base.DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--official-sit-repo", type=Path, default=base.DEFAULT_OFFICIAL_SIT_REPO)
    parser.add_argument("--v-checkpoint", type=Path, default=DEFAULT_V_CHECKPOINT)
    parser.add_argument("--x-checkpoint", type=Path, default=DEFAULT_X_CHECKPOINT)
    parser.add_argument("--internal-depth", type=int, default=8)
    parser.add_argument("--global-batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=1_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--relative-step", type=float, default=0.01)
    parser.add_argument("--time-min", type=float, default=0.02)
    parser.add_argument("--time-max", type=float, default=0.98)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verify-sit-source", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--validation-every", type=int, default=200)
    parser.add_argument("--validation-batches", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
