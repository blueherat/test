#!/usr/bin/env python3
"""Continue v270 with a learned full-rank spectral x/velocity selector."""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from experiments import train_imagenet100_sit_flow as base
from experiments.imagenet100_sit_spectral_target_selector import (
    SpectralTargetSelector,
)


PROTOCOL = "imagenet100_sit_spectral_target_selector_v1"
DEFAULT_SOURCE = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/"
    "sit-s-2_seed0/checkpoints/step_00270000.pt"
)
DEFAULT_OUTPUT = Path("/tmp/eqvae_sit_spectral_target_from_v270_seed0")


class SpectralTargetSiT(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        selector: SpectralTargetSelector,
        *,
        denominator_floor: float,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.selector = selector
        self.denominator_floor = float(denominator_floor)

    def forward(
        self,
        state: torch.Tensor,
        time_value: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        native_output = self.backbone(state, time_value, labels)
        eigenvalues = self.selector.eigenvalues(time_value)
        velocity = self.selector.output_to_velocity(
            native_output,
            state=state,
            time_value=time_value,
            eigenvalues=eigenvalues,
            denominator_floor=self.denominator_floor,
        )
        return native_output, velocity, eigenvalues


@dataclass(frozen=True)
class TrainConfig:
    cache_dir: str
    output_dir: str
    official_sit_repo: str
    source_checkpoint: str
    source_checkpoint_sha256: str
    model_name: str
    source_step: int
    global_batch_size: int
    max_steps: int
    learning_rate: float
    selector_learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    cfg_dropout: float
    ema_decay: float
    denominator_floor: float
    time_terms: int
    initial_x_fraction: float
    maximum_x_fraction: float
    precision: str
    compile: bool
    compile_mode: str
    allow_tf32: bool
    num_workers: int
    prefetch_factor: int
    log_every: int
    validation_every: int
    validation_batches: int
    save_every: int
    seed: int
    restore_source_rng: bool


def _broadcast_string(value: str, context: base.DistributedContext) -> str:
    if context.world_size == 1:
        return value
    values: list[str | None] = [value if context.is_main else None]
    dist.broadcast_object_list(values, src=0)
    if not isinstance(values[0], str):
        raise RuntimeError("failed to broadcast source checkpoint hash")
    return values[0]


def _validate_source(checkpoint: dict, source_metadata: dict) -> None:
    config = checkpoint["config"]
    expectations = {
        "model_name": "SiT-S/2",
        "global_batch_size": 256,
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
        "cfg_dropout": 0.1,
        "precision": "bf16",
        "seed": 0,
    }
    legacy = {
        "prediction_target": "velocity",
        "loss_space": "velocity",
        "time_sampler": "uniform",
    }
    problems = []
    for key, expected in expectations.items():
        if config.get(key) != expected:
            problems.append(f"{key}: expected {expected!r}, found {config.get(key)!r}")
    for key, expected in legacy.items():
        if config.get(key, expected) != expected:
            problems.append(f"{key}: expected {expected!r}, found {config.get(key)!r}")
    if checkpoint.get("official_sit") != source_metadata:
        problems.append("official SiT source metadata differs")
    if int(checkpoint.get("step", -1)) != 270_000:
        problems.append(f"source step must be 270000, found {checkpoint.get('step')}")
    if problems:
        raise ValueError("invalid v270 source checkpoint:\n  " + "\n  ".join(problems))


def _load_backbone(sit_module, config: dict, state: dict) -> nn.Module:
    model = sit_module.SiT_models[str(config["model_name"])](
        input_size=base.LATENT_SHAPE[-1],
        num_classes=base.NUM_CLASSES,
        class_dropout_prob=float(config["cfg_dropout"]),
    )
    model.load_state_dict(state, strict=True)
    return model


def _loss_rows(
    *,
    model: SpectralTargetSiT,
    state: torch.Tensor,
    data: torch.Tensor,
    velocity_target: torch.Tensor,
    time_value: torch.Tensor,
    labels: torch.Tensor,
    precision: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    with base.autocast_context(precision):
        native_output, velocity_prediction, eigenvalues = model(
            state, time_value, labels
        )
    native_target = model.selector.native_target(
        data=data,
        velocity=velocity_target,
        eigenvalues=eigenvalues,
    )
    velocity_losses = (
        (velocity_prediction.float() - velocity_target.float())
        .square()
        .flatten(1)
        .mean(1)
    )
    native_losses = (
        (native_output.float() - native_target.float()).square().flatten(1).mean(1)
    )
    output_as_velocity_losses = (
        (native_output.float() - velocity_target.float()).square().flatten(1).mean(1)
    )
    conversion_delta = (
        (velocity_prediction.float() - native_output.float())
        .square()
        .flatten(1)
        .mean(1)
        .sqrt()
    )
    values = {
        "velocity": velocity_losses,
        "native": native_losses,
        "output_as_velocity": output_as_velocity_losses,
        "conversion_delta_rms": conversion_delta,
        "p_mean": eigenvalues.float().flatten(1).mean(1),
        "p_std": eigenvalues.float().flatten(1).std(1),
    }
    return velocity_losses.mean(), values


@torch.inference_mode()
def validate(
    *,
    model: SpectralTargetSiT,
    loader,
    context: base.DistributedContext,
    config: TrainConfig,
    batches: int,
    seed: int,
) -> dict[str, float]:
    names = (
        "velocity_mse",
        "native_mse",
        "output_as_velocity_mse",
        "conversion_delta_rms",
        "p_mean",
        "p_std",
        "x_fraction_early",
        "x_fraction_middle",
        "x_fraction_late",
    )
    totals = torch.zeros(len(names) + 4, device=context.device, dtype=torch.float64)
    generator = torch.Generator(device=context.device).manual_seed(int(seed))
    model.eval()
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
        data = base.sample_sdvae_posterior(moments, posterior_noise)
        noise = torch.randn(data.shape, generator=generator, device=context.device)
        time_value = torch.rand(
            len(data), generator=generator, device=context.device
        )
        state, velocity_target = base.linear_flow_state_target(
            data, noise, time_value
        )
        _loss, values = _loss_rows(
            model=model,
            state=state,
            data=data,
            velocity_target=velocity_target,
            time_value=time_value,
            labels=labels,
            precision=config.precision,
        )
        columns = torch.stack(
            (
                values["velocity"],
                values["native"],
                values["output_as_velocity"],
                values["conversion_delta_rms"],
                values["p_mean"],
                values["p_std"],
            ),
            dim=1,
        ).double()
        totals[:6] += columns.sum(0)
        totals[9] += len(data)
        x_fraction = 1.0 - values["p_mean"].double()
        for region_index, mask in enumerate(
            (time_value < 1 / 3, (time_value >= 1 / 3) & (time_value < 2 / 3), time_value >= 2 / 3)
        ):
            if mask.any():
                totals[6 + region_index] += x_fraction[mask].sum()
                totals[10 + region_index] += mask.sum()
    base.reduce_sum(totals, context)
    if totals[9].item() == 0:
        raise RuntimeError("validation produced no samples")
    means = totals[:6] / totals[9]
    region_means = [
        totals[6 + index] / totals[10 + index].clamp_min(1.0)
        for index in range(3)
    ]
    result_values = [*means.tolist(), *(value.item() for value in region_means)]
    return dict(zip(names, (float(value) for value in result_values), strict=True))


def train(args: argparse.Namespace) -> None:
    context = base.initialize_distributed(args.device)
    try:
        if args.global_batch_size % context.world_size:
            raise ValueError("global batch size must be divisible by world size")
        base.configure_runtime(args.seed, context.rank, args.allow_tf32)
        source_path = args.source_checkpoint.expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        source_hash = _broadcast_string(
            base.sha256_file(source_path) if context.is_main else "", context
        )
        sit_module, source_metadata = base.load_official_sit_module(
            args.official_sit_repo.expanduser().resolve(),
            verify_source=args.verify_sit_source,
        )
        source = torch.load(
            source_path, map_location=context.device, weights_only=False
        )
        _validate_source(source, source_metadata)
        source_config = source["config"]
        cache_dir = args.cache_dir.expanduser().resolve()
        output_dir = args.output_dir.expanduser().resolve()
        cache_manifest_path = cache_dir / "manifest.json"
        cache_manifest_hash = base.sha256_file(cache_manifest_path)
        if source.get("data_manifest_sha256") != cache_manifest_hash:
            raise ValueError("source checkpoint and cache manifest differ")
        config = TrainConfig(
            cache_dir=str(cache_dir),
            output_dir=str(output_dir),
            official_sit_repo=str(args.official_sit_repo.expanduser().resolve()),
            source_checkpoint=str(source_path),
            source_checkpoint_sha256=source_hash,
            model_name=str(source_config["model_name"]),
            source_step=int(source["step"]),
            global_batch_size=int(args.global_batch_size),
            max_steps=int(args.max_steps),
            learning_rate=float(args.learning_rate),
            selector_learning_rate=float(args.selector_learning_rate),
            weight_decay=float(args.weight_decay),
            beta1=float(args.beta1),
            beta2=float(args.beta2),
            cfg_dropout=float(source_config["cfg_dropout"]),
            ema_decay=float(args.ema_decay),
            denominator_floor=float(args.denominator_floor),
            time_terms=int(args.time_terms),
            initial_x_fraction=float(args.initial_x_fraction),
            maximum_x_fraction=float(args.maximum_x_fraction),
            precision=str(args.precision),
            compile=bool(args.compile),
            compile_mode=str(args.compile_mode),
            allow_tf32=bool(args.allow_tf32),
            num_workers=int(args.num_workers),
            prefetch_factor=int(args.prefetch_factor),
            log_every=int(args.log_every),
            validation_every=int(args.validation_every),
            validation_batches=int(args.validation_batches),
            save_every=int(args.save_every),
            seed=int(args.seed),
            restore_source_rng=bool(args.restore_source_rng),
        )
        if config.max_steps <= config.source_step:
            raise ValueError("max_steps must exceed the 270K source step")

        backbone = _load_backbone(
            sit_module, source_config, source["model"]
        ).to(context.device)
        selector = SpectralTargetSelector(
            channels=base.LATENT_SHAPE[0],
            side=base.LATENT_SHAPE[-1],
            time_terms=config.time_terms,
            initial_x_fraction=config.initial_x_fraction,
            maximum_x_fraction=config.maximum_x_fraction,
        ).to(context.device)
        raw_model = SpectralTargetSiT(
            backbone,
            selector,
            denominator_floor=config.denominator_floor,
        ).to(context.device)
        ema = base.ModelEMA(raw_model)
        ema.module.backbone.load_state_dict(source["ema"], strict=True)

        optimizer = torch.optim.AdamW(
            raw_model.backbone.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
            fused=True,
        )
        optimizer.load_state_dict(source["optimizer"])
        optimizer.add_param_group(
            {
                "params": list(raw_model.selector.parameters()),
                "lr": config.selector_learning_rate,
                "weight_decay": 0.0,
            }
        )
        start_step = config.source_step
        restored_rng = None
        if args.resume.lower() not in {"none", "false", "no"}:
            resume_path = (
                base.latest_checkpoint(output_dir)
                if args.resume.lower() == "auto"
                else Path(args.resume).expanduser().resolve()
            )
            if resume_path is not None:
                resumed = torch.load(
                    resume_path, map_location=context.device, weights_only=False
                )
                if resumed.get("protocol") != PROTOCOL:
                    raise ValueError("resume checkpoint uses another protocol")
                raw_model.load_state_dict(resumed["model"], strict=True)
                ema.load_state_dict(resumed["ema"])
                optimizer.load_state_dict(resumed["optimizer"])
                start_step = int(resumed["step"])
                states = resumed.get("rng_states")
                if not isinstance(states, list) or len(states) != context.world_size:
                    raise ValueError("resume checkpoint has incompatible RNG states")
                restored_rng = states[context.rank]
        if restored_rng is None and config.restore_source_rng:
            states = source.get("rng_states")
            if not isinstance(states, list) or len(states) != context.world_size:
                raise ValueError(
                    "exact source RNG restore requires the original four-rank world size"
                )
            restored_rng = states[context.rank]
        del source
        gc.collect()

        local_batch = config.global_batch_size // context.world_size
        train_loader, train_sampler = base.create_loader(
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
        validation_loader, validation_sampler = base.create_loader(
            cache_dir=cache_dir,
            split="validation",
            local_batch_size=local_batch,
            context=context,
            seed=config.seed + 1,
            shuffle=False,
            num_workers=max(0, min(2, config.num_workers)),
            prefetch_factor=config.prefetch_factor,
            drop_last=False,
        )
        validation_sampler.set_epoch(0)

        train_model: nn.Module = raw_model
        if config.compile:
            train_model = torch.compile(
                raw_model,
                mode=config.compile_mode,
                fullgraph=True,
                dynamic=False,
            )
        if context.world_size > 1:
            train_model = DDP(
                train_model,
                device_ids=[context.local_rank],
                output_device=context.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
                static_graph=True,
            )
        if restored_rng is not None:
            base.restore_rng_state(restored_rng, context.device)
        raw_model.train()
        ema.module.eval()
        batches = base.infinite_train_batches(train_loader, train_sampler, start_step)

        if context.is_main:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            base.atomic_json_dump(
                {
                    "protocol": PROTOCOL,
                    "config": {**asdict(config), "world_size": context.world_size},
                    "official_sit": source_metadata,
                    "data_manifest_sha256": cache_manifest_hash,
                    "parameter_count": sum(p.numel() for p in raw_model.parameters()),
                    "selector_parameter_count": sum(
                        p.numel() for p in raw_model.selector.parameters()
                    ),
                    "objective": {
                        "path": "state=(1-t)*noise+t*data",
                        "native_target": "y=P(t)*velocity+(I-P(t))*data",
                        "conversion": "v=[(1-t)I+tP]^-1[y-(I-P)state]",
                        "optimized_loss": "velocity MSE",
                        "v300_access_during_training": False,
                    },
                },
                output_dir / "run_config.json",
            )
            print(
                json.dumps(
                    {
                        "event": "start",
                        "step": start_step,
                        "max_steps": config.max_steps,
                        "world_size": context.world_size,
                        "global_batch": config.global_batch_size,
                        "selector_parameters": sum(
                            p.numel() for p in raw_model.selector.parameters()
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        base.barrier(context)

        metrics_path = output_dir / "train_metrics.jsonl"
        initial_raw = validate(
            model=raw_model,
            loader=validation_loader,
            context=context,
            config=config,
            batches=config.validation_batches,
            seed=config.seed + 700_000,
        )
        initial_ema = validate(
            model=ema.module,
            loader=validation_loader,
            context=context,
            config=config,
            batches=config.validation_batches,
            seed=config.seed + 700_000,
        )
        raw_model.train()
        if context.is_main:
            row = {
                "step": start_step,
                "raw_validation": initial_raw,
                "ema_validation": initial_ema,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps(row, sort_keys=True), flush=True)

        running = torch.zeros(4, device=context.device, dtype=torch.float64)
        running_steps = 0
        interval_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for step in range(start_step + 1, config.max_steps + 1):
            moments, labels = next(batches)
            moments = moments.to(context.device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(context.device, dtype=torch.long, non_blocking=True)
            posterior_noise = torch.randn(
                (len(moments), *base.LATENT_SHAPE), device=context.device
            )
            data = base.sample_sdvae_posterior(moments, posterior_noise)
            noise = torch.randn_like(data)
            time_value = torch.rand(len(data), device=context.device)
            state, velocity_target = base.linear_flow_state_target(
                data, noise, time_value
            )
            with base.autocast_context(config.precision):
                native_output, velocity_prediction, eigenvalues = train_model(
                    state, time_value, labels
                )
            velocity_loss = F.mse_loss(
                velocity_prediction.float(), velocity_target.float()
            )
            if not torch.isfinite(velocity_loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            velocity_loss.backward()
            optimizer.step()
            raw_model.selector.project_parameters_()
            optimizer.zero_grad(set_to_none=True)
            ema.update(config.ema_decay)
            with torch.no_grad():
                running += torch.stack(
                    (
                        velocity_loss.detach().double(),
                        eigenvalues.float().mean().double(),
                        eigenvalues.float().std().double(),
                        torch.tensor(1.0, device=context.device, dtype=torch.float64),
                    )
                )
            running_steps += 1

            if step % config.log_every == 0 or step == config.max_steps:
                torch.cuda.synchronize(context.device)
                elapsed = torch.tensor(
                    time.perf_counter() - interval_started, device=context.device
                )
                base.reduce_max(elapsed, context)
                values = running.clone()
                base.reduce_sum(values, context)
                memory = torch.tensor(
                    torch.cuda.max_memory_allocated(context.device) / 2**30,
                    device=context.device,
                )
                base.reduce_max(memory, context)
                denominator = values[3].clamp_min(1.0)
                row = {
                    "step": step,
                    "train_velocity_mse": float((values[0] / denominator).item()),
                    "p_mean": float((values[1] / denominator).item()),
                    "p_std": float((values[2] / denominator).item()),
                    "steps_per_second": float(running_steps / elapsed.item()),
                    "images_per_second": float(
                        running_steps * config.global_batch_size / elapsed.item()
                    ),
                    "max_allocated_gb": float(memory.item()),
                }
                if context.is_main:
                    with metrics_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
                    print(json.dumps(row, sort_keys=True), flush=True)
                running.zero_()
                running_steps = 0
                interval_started = time.perf_counter()
                torch.cuda.reset_peak_memory_stats(context.device)

            should_validate = config.validation_every > 0 and (
                step % config.validation_every == 0 or step == config.max_steps
            )
            if should_validate:
                paused = time.perf_counter()
                raw_metrics = validate(
                    model=raw_model,
                    loader=validation_loader,
                    context=context,
                    config=config,
                    batches=config.validation_batches,
                    seed=config.seed + 700_000,
                )
                ema_metrics = validate(
                    model=ema.module,
                    loader=validation_loader,
                    context=context,
                    config=config,
                    batches=config.validation_batches,
                    seed=config.seed + 700_000,
                )
                raw_model.train()
                if context.is_main:
                    row = {
                        "step": step,
                        "raw_validation": raw_metrics,
                        "ema_validation": ema_metrics,
                    }
                    with metrics_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
                    print(json.dumps(row, sort_keys=True), flush=True)
                interval_started += time.perf_counter() - paused

            if step % config.save_every == 0 or step == config.max_steps:
                paused = time.perf_counter()
                rng_states = base.gather_rng_states(context)
                if context.is_main:
                    checkpoint_path = (
                        output_dir / "checkpoints" / f"step_{step:08d}.pt"
                    )
                    base.atomic_torch_save(
                        {
                            "protocol": PROTOCOL,
                            "step": step,
                            "model": raw_model.state_dict(),
                            "ema": ema.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "rng_states": rng_states,
                            "config": {**asdict(config), "world_size": context.world_size},
                            "official_sit": source_metadata,
                            "data_manifest_sha256": cache_manifest_hash,
                        },
                        checkpoint_path,
                    )
                    print(
                        json.dumps({"event": "checkpoint", "path": str(checkpoint_path)}),
                        flush=True,
                    )
                base.barrier(context)
                interval_started += time.perf_counter() - paused
    finally:
        base.cleanup_distributed(context)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=base.DEFAULT_CACHE_DIR)
    parser.add_argument("--official-sit-repo", type=Path, default=base.DEFAULT_OFFICIAL_SIT_REPO)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=300_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--selector-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--denominator-floor", type=float, default=0.05)
    parser.add_argument("--time-terms", type=int, default=8)
    parser.add_argument("--initial-x-fraction", type=float, default=1e-3)
    parser.add_argument("--maximum-x-fraction", type=float, default=0.999)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verify-sit-source", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--validation-every", type=int, default=5_000)
    parser.add_argument("--validation-batches", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--restore-source-rng", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", default="none")
    parser.add_argument("--device", default="cuda:0")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
