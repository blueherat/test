"""Train only a clean-x output head on a frozen ImageNet-100 v800 SiT.

The source SiT checkpoint, including its final AdaLN and native velocity
projection, remains frozen in sampling/eval mode. The official unused sigma
half of the final projection is exposed as a separate clean-latent linear and
is the only trainable module.
"""

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
from torch.utils.data import DataLoader

try:
    from experiments import train_imagenet100_sit_flow as base
    from experiments.imagenet100_sit_vx_dual_head import (
        VelocityCleanProjection,
        clean_prediction_to_velocity,
        freeze_except_clean_head,
        retrofit_velocity_clean_heads,
        split_velocity_clean_output,
    )
    from experiments.official_imagenet100_sit_s2 import source_step
except ModuleNotFoundError:
    import train_imagenet100_sit_flow as base
    from imagenet100_sit_vx_dual_head import (
        VelocityCleanProjection,
        clean_prediction_to_velocity,
        freeze_except_clean_head,
        retrofit_velocity_clean_heads,
        split_velocity_clean_output,
    )
    from official_imagenet100_sit_s2 import source_step


PROTOCOL = "imagenet100_sit_frozen_v_clean_linear_probe_v1"
DEFAULT_SOURCE_CHECKPOINT = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/"
    "sit-s-2_seed0/checkpoints/step_00800000.pt"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/"
    "runs/sit-s-2_v800-ema_frozen-clean-head_seed0"
)
TIME_BIN_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


@dataclass(frozen=True)
class FrozenHeadTrainConfig:
    cache_dir: str
    output_dir: str
    official_sit_repo: str
    source_checkpoint: str
    source_checkpoint_sha256: str
    source_state_key: str
    source_step: int
    model_name: str
    cfg_dropout: float
    global_batch_size: int
    max_steps: int
    learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    ema_decay: float
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
    clean_velocity_denominator_floor: float


def broadcast_string(value: str, context: base.DistributedContext) -> str:
    if context.world_size == 1:
        return value
    values: list[str | None] = [value if context.is_main else None]
    dist.broadcast_object_list(values, src=0)
    if not isinstance(values[0], str):
        raise RuntimeError("failed to broadcast source checkpoint hash")
    return values[0]


def source_unused_projection_max_abs(model: nn.Module) -> float:
    linear = model.final_layer.linear
    patch_size = int(model.x_embedder.patch_size[0])
    channels = base.LATENT_SHAPE[0]
    weight = linear.weight.detach().reshape(
        patch_size,
        patch_size,
        2 * channels,
        linear.in_features,
    )
    maximum = weight[:, :, channels:].abs().max()
    if linear.bias is not None:
        bias = linear.bias.detach().reshape(patch_size, patch_size, 2 * channels)
        maximum = torch.maximum(maximum, bias[:, :, channels:].abs().max())
    return float(maximum.item())


def create_frozen_clean_probe(
    sit_module,
    *,
    model_name: str,
    cfg_dropout: float,
    source_state: dict[str, torch.Tensor],
) -> tuple[nn.Module, dict[str, float | int]]:
    """Load the source velocity model and expose its unused half as x head."""

    model = sit_module.SiT_models[model_name](
        input_size=base.LATENT_SHAPE[-1],
        num_classes=base.NUM_CLASSES,
        class_dropout_prob=cfg_dropout,
    )
    model.load_state_dict(source_state, strict=True)
    parameter_count_before = sum(parameter.numel() for parameter in model.parameters())
    unused_max_abs = source_unused_projection_max_abs(model)
    if unused_max_abs != 0.0:
        raise ValueError(
            "the source checkpoint's unused output channels are not exactly zero: "
            f"max_abs={unused_max_abs}"
        )
    retrofit_velocity_clean_heads(model, latent_channels=base.LATENT_SHAPE[0])
    projection = model.final_layer.linear
    if not isinstance(projection, VelocityCleanProjection):
        raise AssertionError("velocity/clean projection retrofit failed")
    nn.init.zeros_(projection.clean_head.weight)
    if projection.clean_head.bias is not None:
        nn.init.zeros_(projection.clean_head.bias)
    freeze_except_clean_head(model)
    parameter_count_after = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if parameter_count_after != parameter_count_before:
        raise AssertionError("dual-head retrofit changed total parameter count")
    return model, {
        "parameter_count": parameter_count_after,
        "trainable_parameter_count": trainable_count,
        "source_unused_projection_max_abs": unused_max_abs,
    }


def clean_head(model: nn.Module) -> nn.Linear:
    projection = model.final_layer.linear
    if not isinstance(projection, VelocityCleanProjection):
        raise TypeError("model does not expose velocity/clean heads")
    return projection.clean_head


def validate_resume(
    stored: dict,
    current: FrozenHeadTrainConfig,
    world_size: int,
) -> None:
    immutable = (
        "cache_dir",
        "official_sit_repo",
        "source_checkpoint",
        "source_checkpoint_sha256",
        "source_state_key",
        "source_step",
        "model_name",
        "cfg_dropout",
        "global_batch_size",
        "learning_rate",
        "weight_decay",
        "beta1",
        "beta2",
        "ema_decay",
        "precision",
        "compile",
        "compile_mode",
        "allow_tf32",
        "seed",
        "clean_velocity_denominator_floor",
    )
    current_values = asdict(current)
    mismatches = [
        f"{key}: checkpoint={stored.get(key)!r}, current={current_values[key]!r}"
        for key in immutable
        if stored.get(key) != current_values[key]
    ]
    if int(stored.get("world_size", world_size)) != world_size:
        mismatches.append(
            f"world_size: checkpoint={stored.get('world_size')!r}, current={world_size}"
        )
    if mismatches:
        raise ValueError("incompatible resume configuration:\n  " + "\n  ".join(mismatches))


@torch.inference_mode()
def validation_metrics(
    *,
    model: nn.Module,
    loader: DataLoader,
    context: base.DistributedContext,
    precision: str,
    batches: int,
    seed: int,
    denominator_floor: float,
) -> dict[str, object]:
    """Track x regression while verifying the frozen v field remains stable."""

    generator = torch.Generator(device=context.device).manual_seed(int(seed))
    metric_names = (
        "clean_mse",
        "clean_derived_velocity_mse",
        "frozen_velocity_mse",
    )
    totals = torch.zeros(len(metric_names) + 1, device=context.device, dtype=torch.float64)
    bin_totals = torch.zeros(
        len(TIME_BIN_EDGES) - 1,
        len(metric_names) + 1,
        device=context.device,
        dtype=torch.float64,
    )
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
        clean = base.sample_sdvae_posterior(moments, posterior_noise)
        noise = torch.randn(clean.shape, generator=generator, device=context.device)
        time_value = torch.rand(
            (len(clean),),
            generator=generator,
            device=context.device,
        )
        state, target_velocity = base.linear_flow_state_target(clean, noise, time_value)
        with base.autocast_context(precision):
            output = model(state, time_value, labels)
        velocity_prediction, clean_prediction = split_velocity_clean_output(
            output,
            latent_channels=base.LATENT_SHAPE[0],
        )
        clean_velocity = clean_prediction_to_velocity(
            clean_prediction,
            state=state,
            time_value=time_value,
            denominator_floor=denominator_floor,
        )
        values = torch.stack(
            (
                (clean_prediction.float() - clean.float()).square().flatten(1).mean(1),
                (clean_velocity - target_velocity.float()).square().flatten(1).mean(1),
                (velocity_prediction.float() - target_velocity.float())
                .square()
                .flatten(1)
                .mean(1),
            ),
            dim=1,
        ).double()
        totals[:-1] += values.sum(dim=0)
        totals[-1] += len(values)
        for index, (lower, upper) in enumerate(
            zip(TIME_BIN_EDGES[:-1], TIME_BIN_EDGES[1:], strict=True)
        ):
            mask = (time_value >= lower) & (
                time_value < upper if upper < 1.0 else time_value <= upper
            )
            if mask.any():
                bin_totals[index, :-1] += values[mask].sum(dim=0)
                bin_totals[index, -1] += mask.sum()

    base.reduce_sum(totals, context)
    base.reduce_sum(bin_totals, context)
    if totals[-1].item() == 0:
        raise RuntimeError("validation loader produced no samples")
    means = totals[:-1] / totals[-1]
    result: dict[str, object] = {
        name: float(value.item())
        for name, value in zip(metric_names, means, strict=True)
    }
    bins: list[dict[str, float | int]] = []
    for index, (lower, upper) in enumerate(
        zip(TIME_BIN_EDGES[:-1], TIME_BIN_EDGES[1:], strict=True)
    ):
        count = bin_totals[index, -1]
        if count.item() == 0:
            continue
        means = bin_totals[index, :-1] / count
        row: dict[str, float | int] = {
            "t_min": lower,
            "t_max": upper,
            "count": int(count.item()),
        }
        row.update(
            {
                name: float(value.item())
                for name, value in zip(metric_names, means, strict=True)
            }
        )
        bins.append(row)
    result["time_bins"] = bins
    return result


def build_metadata(
    *,
    config: FrozenHeadTrainConfig,
    context: base.DistributedContext,
    source_metadata: dict,
    architecture_stats: dict[str, float | int],
    cache_manifest: dict,
) -> dict:
    import timm

    return {
        "protocol": PROTOCOL,
        "config": asdict(config),
        "world_size": context.world_size,
        **architecture_stats,
        "official_sit": source_metadata,
        "objective": {
            "path": "x_t=(1-t)*noise+t*clean",
            "frozen_native_head": "velocity=clean-noise",
            "trainable_head": "direct clean SD-VAE latent prediction",
            "loss": "MSE(clean_prediction, clean)",
            "training_time_distribution": {"name": "uniform", "interval": "[0,1)"},
            "source_backbone_mode": "eval; class dropout disabled",
            "clean_velocity_denominator_floor": config.clean_velocity_denominator_floor,
        },
        "fairness": {
            "source_velocity_parameters_updated": False,
            "optimizer_contains_only_clean_head": True,
            "velocity_output_preserved_by_interleaved_projection_split": True,
            "clean_head_initialization": "exact zeros from unused official sigma channels",
            "total_parameter_count_unchanged": True,
        },
        "latent": {
            "posterior": "mean+std*N(0,I)",
            "scaling_factor": base.SD_VAE_SCALING_FACTOR,
            "shape": list(base.LATENT_SHAPE),
        },
        "data_manifest": cache_manifest,
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "timm": timm.__version__,
            "gpus": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
            "repo_git_commit": base.git_value(base.REPO_ROOT, "rev-parse", "HEAD"),
        },
    }


def train(args: argparse.Namespace) -> None:
    context = base.initialize_distributed(args.device)
    try:
        if args.global_batch_size % context.world_size:
            raise ValueError("--global-batch-size must be divisible by world size")
        if args.source_state_key not in {"model", "ema"}:
            raise ValueError("--source-state-key must be model or ema")
        source_path = args.source_checkpoint.expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        source_hash = base.sha256_file(source_path) if context.is_main else ""
        source_hash = broadcast_string(source_hash, context)
        source_payload = torch.load(
            source_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        if args.source_state_key not in source_payload:
            raise KeyError(f"source checkpoint lacks {args.source_state_key!r}")
        source_config = source_payload.get("config", {})
        model_name = str(source_config.get("model_name", "SiT-S/2"))
        cfg_dropout = float(source_config.get("cfg_dropout", 0.1))
        source_checkpoint_step = source_step(source_payload)

        config = FrozenHeadTrainConfig(
            cache_dir=str(args.cache_dir.expanduser().resolve()),
            output_dir=str(args.output_dir.expanduser().resolve()),
            official_sit_repo=str(args.official_sit_repo.expanduser().resolve()),
            source_checkpoint=str(source_path),
            source_checkpoint_sha256=source_hash,
            source_state_key=args.source_state_key,
            source_step=source_checkpoint_step,
            model_name=model_name,
            cfg_dropout=cfg_dropout,
            global_batch_size=int(args.global_batch_size),
            max_steps=int(args.max_steps),
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            beta1=float(args.beta1),
            beta2=float(args.beta2),
            ema_decay=float(args.ema_decay),
            precision=args.precision,
            compile=bool(args.compile),
            compile_mode=args.compile_mode,
            allow_tf32=bool(args.allow_tf32),
            num_workers=int(args.num_workers),
            prefetch_factor=int(args.prefetch_factor),
            log_every=int(args.log_every),
            validation_every=int(args.validation_every),
            validation_batches=int(args.validation_batches),
            save_every=int(args.save_every),
            seed=int(args.seed),
            clean_velocity_denominator_floor=float(
                args.clean_velocity_denominator_floor
            ),
        )
        if min(
            config.global_batch_size,
            config.max_steps,
            config.log_every,
            config.save_every,
            config.learning_rate,
            config.clean_velocity_denominator_floor,
        ) <= 0:
            raise ValueError("batch, steps, intervals, learning rate, and floor must be positive")
        if config.clean_velocity_denominator_floor >= 0.5:
            raise ValueError("--clean-velocity-denominator-floor must be below 0.5")
        local_batch_size = config.global_batch_size // context.world_size
        base.configure_runtime(config.seed, context.rank, config.allow_tf32)

        cache_dir = Path(config.cache_dir)
        output_dir = Path(config.output_dir)
        cache_manifest_path = cache_dir / "manifest.json"
        cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
        cache_manifest_sha256 = base.sha256_file(cache_manifest_path)
        if cache_manifest.get("format") != "eqvae_imagenet100_cmc_sdvae_moments_v1":
            raise ValueError(f"unsupported data manifest: {cache_manifest_path}")

        sit_module, source_metadata = base.load_official_sit_module(
            Path(config.official_sit_repo),
            verify_source=args.verify_sit_source,
        )
        if source_payload.get("official_sit") != source_metadata:
            raise ValueError("source checkpoint official SiT metadata does not match")
        raw_model, architecture_stats = create_frozen_clean_probe(
            sit_module,
            model_name=config.model_name,
            cfg_dropout=config.cfg_dropout,
            source_state=source_payload[config.source_state_key],
        )
        del source_payload
        gc.collect()
        raw_model = raw_model.to(context.device)
        ema = base.ModelEMA(raw_model)
        trainable_parameters = [
            parameter for parameter in raw_model.parameters() if parameter.requires_grad
        ]
        if sum(parameter.numel() for parameter in trainable_parameters) != int(
            architecture_stats["trainable_parameter_count"]
        ):
            raise AssertionError("trainable parameter audit changed after CUDA transfer")
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
            fused=True,
        )

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

        resume_path = base.resolve_resume(args.resume, output_dir)
        start_step = 0
        restored_rng: dict | None = None
        if resume_path is not None:
            checkpoint = torch.load(
                resume_path,
                map_location=context.device,
                weights_only=False,
            )
            if checkpoint.get("protocol") != PROTOCOL:
                raise ValueError(f"unexpected checkpoint protocol: {checkpoint.get('protocol')!r}")
            validate_resume(checkpoint["config"], config, context.world_size)
            if checkpoint.get("data_manifest_sha256") != cache_manifest_sha256:
                raise ValueError("checkpoint data manifest does not match current cache")
            clean_head(raw_model).load_state_dict(checkpoint["clean_head"], strict=True)
            clean_head(ema.module).load_state_dict(
                checkpoint["clean_head_ema"],
                strict=True,
            )
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_step = int(checkpoint["step"])
            rng_states = checkpoint.get("rng_states")
            if not isinstance(rng_states, list) or len(rng_states) != context.world_size:
                raise ValueError("checkpoint is missing per-rank RNG states")
            restored_rng = rng_states[context.rank]

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
        if resume_path is None:
            ema.load_state_dict(raw_model.state_dict())
        if restored_rng is not None:
            base.restore_rng_state(restored_rng, context.device)
        raw_model.eval()
        ema.module.eval()
        batches = base.infinite_train_batches(train_loader, train_sampler, start_step)

        if context.is_main:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            metadata = build_metadata(
                config=config,
                context=context,
                source_metadata=source_metadata,
                architecture_stats=architecture_stats,
                cache_manifest=cache_manifest,
            )
            base.atomic_json_dump(metadata, output_dir / "run_config.json")
            print(
                json.dumps(
                    {
                        "event": "start",
                        "step": start_step,
                        "source_step": config.source_step,
                        "source_state_key": config.source_state_key,
                        "model": config.model_name,
                        "parameters": architecture_stats["parameter_count"],
                        "trainable_parameters": architecture_stats[
                            "trainable_parameter_count"
                        ],
                        "world_size": context.world_size,
                        "global_batch": config.global_batch_size,
                        "local_batch": local_batch_size,
                        "precision": config.precision,
                        "compile": config.compile,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        base.barrier(context)

        metrics_path = output_dir / "train_metrics.jsonl"
        running_loss = torch.zeros(1, device=context.device, dtype=torch.float64)
        running_steps = 0
        interval_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)

        for step in range(start_step + 1, config.max_steps + 1):
            moments, labels = next(batches)
            moments = moments.to(context.device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(context.device, dtype=torch.long, non_blocking=True)
            posterior_noise = torch.randn(
                (len(moments), *base.LATENT_SHAPE),
                device=context.device,
            )
            clean = base.sample_sdvae_posterior(moments, posterior_noise)
            noise = torch.randn_like(clean)
            time_value = torch.rand((len(clean),), device=context.device)
            state, _ = base.linear_flow_state_target(clean, noise, time_value)

            with base.autocast_context(config.precision):
                output = train_model(state, time_value, labels)
            _, clean_prediction = split_velocity_clean_output(
                output,
                latent_channels=base.LATENT_SHAPE[0],
            )
            loss = F.mse_loss(clean_prediction.float(), clean.float())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at step {step}")
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
                elapsed_tensor = torch.tensor(elapsed, device=context.device)
                base.reduce_max(elapsed_tensor, context)
                memory_tensor = torch.tensor(
                    torch.cuda.max_memory_allocated(context.device) / 2**30,
                    device=context.device,
                )
                base.reduce_max(memory_tensor, context)
                row = {
                    "step": step,
                    "train_clean_loss": float(values[0].item() / values[1].item()),
                    "steps_per_second": float(running_steps / elapsed_tensor.item()),
                    "images_per_second": float(
                        running_steps * config.global_batch_size / elapsed_tensor.item()
                    ),
                    "max_allocated_gb": float(memory_tensor.item()),
                    "learning_rate": optimizer.param_groups[0]["lr"],
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
                pause_started = time.perf_counter()
                raw_metrics = validation_metrics(
                    model=raw_model,
                    loader=validation_loader,
                    context=context,
                    precision=config.precision,
                    batches=config.validation_batches,
                    seed=config.seed + 700_000,
                    denominator_floor=config.clean_velocity_denominator_floor,
                )
                ema_metrics = validation_metrics(
                    model=ema.module,
                    loader=validation_loader,
                    context=context,
                    precision=config.precision,
                    batches=config.validation_batches,
                    seed=config.seed + 700_000,
                    denominator_floor=config.clean_velocity_denominator_floor,
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
                interval_started += time.perf_counter() - pause_started

            should_save = step % config.save_every == 0 or step == config.max_steps
            if should_save:
                pause_started = time.perf_counter()
                rng_states = base.gather_rng_states(context)
                if context.is_main:
                    checkpoint_path = output_dir / "checkpoints" / f"step_{step:08d}.pt"
                    base.atomic_torch_save(
                        {
                            "protocol": PROTOCOL,
                            "step": step,
                            "clean_head": clean_head(raw_model).state_dict(),
                            "clean_head_ema": clean_head(ema.module).state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "rng_states": rng_states,
                            "config": {**asdict(config), "world_size": context.world_size},
                            "official_sit": source_metadata,
                            "data_manifest_sha256": cache_manifest_sha256,
                        },
                        checkpoint_path,
                    )
                    print(
                        json.dumps({"event": "checkpoint", "path": str(checkpoint_path)}),
                        flush=True,
                    )
                base.barrier(context)
                interval_started += time.perf_counter() - pause_started
    finally:
        base.cleanup_distributed(context)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=base.DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--official-sit-repo", type=Path, default=base.DEFAULT_OFFICIAL_SIT_REPO)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE_CHECKPOINT)
    parser.add_argument("--source-state-key", choices=("ema", "model"), default="ema")
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=200_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
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
    parser.add_argument("--save-every", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clean-velocity-denominator-floor", type=float, default=0.05)
    parser.add_argument(
        "--resume",
        default="auto",
        help="auto, none, or an explicit clean-head checkpoint path",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
