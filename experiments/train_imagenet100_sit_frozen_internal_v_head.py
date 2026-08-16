#!/usr/bin/env python3
"""Train an Internal-Guidance velocity head on a frozen ImageNet-100 v800 SiT."""

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
    from experiments.imagenet100_sit_internal_v_head import (
        FrozenPrefix,
        create_internal_velocity_head,
        freeze_source_model,
        full_and_internal_velocity,
        unpatchify_channels,
        validate_internal_depth,
    )
except ModuleNotFoundError:
    import train_imagenet100_sit_flow as base
    from imagenet100_sit_internal_v_head import (
        FrozenPrefix,
        create_internal_velocity_head,
        freeze_source_model,
        full_and_internal_velocity,
        unpatchify_channels,
        validate_internal_depth,
    )


PROTOCOL = "imagenet100_sit_frozen_v_internal_velocity_head_v1"
DEFAULT_SOURCE_CHECKPOINT = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/"
    "sit-s-2_seed0/checkpoints/step_00800000.pt"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/"
    "sit-s-2_v800-ema_frozen-internal-v-depth8_seed0"
)
DEFAULT_IG_REPO = (
    base.REPO_ROOT
    / "research_repos/internal_guidance_study/Internal-Guidance/SiT"
)
TIME_BIN_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


@dataclass(frozen=True)
class FrozenInternalTrainConfig:
    cache_dir: str
    output_dir: str
    official_sit_repo: str
    official_ig_repo: str
    source_checkpoint: str
    source_checkpoint_sha256: str
    source_state_key: str
    source_step: int
    model_name: str
    cfg_dropout: float
    internal_depth: int
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


def broadcast_string(value: str, context: base.DistributedContext) -> str:
    if context.world_size == 1:
        return value
    values: list[str | None] = [value if context.is_main else None]
    dist.broadcast_object_list(values, src=0)
    if not isinstance(values[0], str):
        raise RuntimeError("failed to broadcast source checkpoint hash")
    return values[0]


def create_frozen_internal_probe(
    sit_module,
    *,
    model_name: str,
    cfg_dropout: float,
    source_state: dict[str, torch.Tensor],
    internal_depth: int,
) -> tuple[nn.Module, nn.Module, dict[str, int]]:
    source = sit_module.SiT_models[model_name](
        input_size=base.LATENT_SHAPE[-1],
        num_classes=base.NUM_CLASSES,
        class_dropout_prob=cfg_dropout,
    )
    source.load_state_dict(source_state, strict=True)
    freeze_source_model(source)
    depth = validate_internal_depth(source, internal_depth)
    head = create_internal_velocity_head(
        sit_module,
        source,
        latent_channels=base.LATENT_SHAPE[0],
    )
    source_parameters = sum(parameter.numel() for parameter in source.parameters())
    head_parameters = sum(parameter.numel() for parameter in head.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in (*source.parameters(), *head.parameters())
        if parameter.requires_grad
    )
    if trainable_parameters != head_parameters:
        raise AssertionError("parameters outside the internal head are trainable")
    return source, head, {
        "source_parameter_count": source_parameters,
        "internal_head_parameter_count": head_parameters,
        "total_parameter_count": source_parameters + head_parameters,
        "trainable_parameter_count": trainable_parameters,
        "source_block_count": len(source.blocks),
        "internal_depth": depth,
    }


def validate_resume(
    stored: dict,
    current: FrozenInternalTrainConfig,
    world_size: int,
) -> None:
    immutable = (
        "cache_dir",
        "official_sit_repo",
        "official_ig_repo",
        "source_checkpoint",
        "source_checkpoint_sha256",
        "source_state_key",
        "source_step",
        "model_name",
        "cfg_dropout",
        "internal_depth",
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


def per_sample_metrics(
    full: torch.Tensor,
    internal: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    full_flat = full.float().flatten(1)
    internal_flat = internal.float().flatten(1)
    target_flat = target.float().flatten(1)
    direction = full_flat - internal_flat
    residual = target_flat - full_flat
    direction_norm = direction.square().mean(dim=1).sqrt()
    residual_norm = residual.square().mean(dim=1).sqrt()
    alignment = (direction * residual).mean(dim=1)
    cosine = alignment / (direction_norm * residual_norm).clamp_min(1e-12)
    return torch.stack(
        (
            (internal_flat - target_flat).square().mean(dim=1),
            (full_flat - target_flat).square().mean(dim=1),
            direction_norm,
            cosine,
            alignment.gt(0).float(),
        ),
        dim=1,
    )


@torch.inference_mode()
def validation_metrics(
    *,
    source: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    context: base.DistributedContext,
    precision: str,
    batches: int,
    seed: int,
    internal_depth: int,
) -> dict[str, object]:
    generator = torch.Generator(device=context.device).manual_seed(int(seed))
    metric_names = (
        "internal_velocity_mse",
        "frozen_velocity_mse",
        "full_internal_gap_rms",
        "direction_residual_cosine",
        "positive_alignment_fraction",
    )
    totals = torch.zeros(len(metric_names) + 1, device=context.device, dtype=torch.float64)
    bin_totals = torch.zeros(
        len(TIME_BIN_EDGES) - 1,
        len(metric_names) + 1,
        device=context.device,
        dtype=torch.float64,
    )
    source.eval()
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
        time_value = torch.rand(
            (len(clean),),
            generator=generator,
            device=context.device,
        )
        state, target_velocity = base.linear_flow_state_target(clean, noise, time_value)
        with base.autocast_context(precision):
            full, internal = full_and_internal_velocity(
                source,
                head,
                state,
                time_value,
                labels,
                internal_depth=internal_depth,
                latent_channels=base.LATENT_SHAPE[0],
            )
        values = per_sample_metrics(full, internal, target_velocity).double()
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
    config: FrozenInternalTrainConfig,
    context: base.DistributedContext,
    source_metadata: dict,
    architecture_stats: dict[str, int],
    cache_manifest: dict,
) -> dict:
    ig_repo = Path(config.official_ig_repo)
    ig_model_path = ig_repo / "models/sit.py"
    return {
        "protocol": PROTOCOL,
        "config": asdict(config),
        "world_size": context.world_size,
        **architecture_stats,
        "official_sit": source_metadata,
        "official_internal_guidance_reference": {
            "repo": str(ig_repo),
            "git_commit": base.git_value(ig_repo.parent, "rev-parse", "HEAD"),
            "model_sha256": base.sha256_file(ig_model_path),
            "reference_structure": "independent FinalLayer after encoder_depth=8",
            "reference_auxiliary_loss_weight": 0.5,
            "reference_training": "joint full-model training",
        },
        "objective": {
            "path": "x_t=(1-t)*noise+t*clean",
            "target": "velocity=clean-noise",
            "loss": "MSE(internal_velocity, target_velocity)",
            "training_time_distribution": {"name": "uniform", "interval": "[0,1)"},
            "source_backbone_mode": "eval; class dropout disabled",
            "head_input": f"hidden state after block {config.internal_depth}",
            "head_architecture": "official SiT FinalLayer (AdaLN + linear projection)",
        },
        "fairness": {
            "source_parameters_updated": False,
            "optimizer_contains_only_internal_head": True,
            "source_state_key": config.source_state_key,
            "source_output_is_recomputed_without_modification": True,
            "same_data_path_and_velocity_target_as_source_sit": True,
            "head_zero_initialized_like_official_internal_guidance": True,
        },
        "scope_boundary": (
            "frozen-probe Internal Guidance; unlike the paper, the full model and "
            "intermediate head are not jointly trained"
        ),
        "latent": {
            "posterior": "mean+std*N(0,I)",
            "scaling_factor": base.SD_VAE_SCALING_FACTOR,
            "shape": list(base.LATENT_SHAPE),
        },
        "data_manifest": cache_manifest,
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
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
        source_path = args.source_checkpoint.expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        ig_repo = args.official_ig_repo.expanduser().resolve()
        if not (ig_repo / "models/sit.py").is_file():
            raise FileNotFoundError(f"missing official IG SiT source under {ig_repo}")
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
        source_step = int(source_payload.get("step", -1))
        if source_step < 1:
            raise ValueError("source checkpoint has no valid training step")

        config = FrozenInternalTrainConfig(
            cache_dir=str(args.cache_dir.expanduser().resolve()),
            output_dir=str(args.output_dir.expanduser().resolve()),
            official_sit_repo=str(args.official_sit_repo.expanduser().resolve()),
            official_ig_repo=str(ig_repo),
            source_checkpoint=str(source_path),
            source_checkpoint_sha256=source_hash,
            source_state_key=args.source_state_key,
            source_step=source_step,
            model_name=model_name,
            cfg_dropout=cfg_dropout,
            internal_depth=int(args.internal_depth),
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
        )
        if min(
            config.internal_depth,
            config.global_batch_size,
            config.max_steps,
            config.log_every,
            config.save_every,
            config.learning_rate,
        ) <= 0:
            raise ValueError("depth, batch, steps, intervals, and learning rate must be positive")
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
        source, head, architecture_stats = create_frozen_internal_probe(
            sit_module,
            model_name=config.model_name,
            cfg_dropout=config.cfg_dropout,
            source_state=source_payload[config.source_state_key],
            internal_depth=config.internal_depth,
        )
        del source_payload
        gc.collect()
        source = source.to(context.device).eval()
        head = head.to(context.device)
        ema = base.ModelEMA(head)
        optimizer = torch.optim.AdamW(
            head.parameters(),
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
            head.load_state_dict(checkpoint["internal_head"], strict=True)
            ema.module.load_state_dict(checkpoint["internal_head_ema"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_step = int(checkpoint["step"])
            rng_states = checkpoint.get("rng_states")
            if not isinstance(rng_states, list) or len(rng_states) != context.world_size:
                raise ValueError("checkpoint is missing per-rank RNG states")
            restored_rng = rng_states[context.rank]

        prefix: nn.Module = FrozenPrefix(source, config.internal_depth).eval()
        train_head: nn.Module = head
        if config.compile:
            prefix = torch.compile(
                prefix,
                mode=config.compile_mode,
                fullgraph=True,
                dynamic=False,
            )
            train_head = torch.compile(
                head,
                mode=config.compile_mode,
                fullgraph=True,
                dynamic=False,
            )
        if context.world_size > 1:
            train_head = DDP(
                train_head,
                device_ids=[context.local_rank],
                output_device=context.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
                static_graph=True,
            )
        if resume_path is None:
            ema.module.load_state_dict(head.state_dict())
        if restored_rng is not None:
            base.restore_rng_state(restored_rng, context.device)
        source.eval()
        head.eval()
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
                        "internal_depth": config.internal_depth,
                        "source_blocks": architecture_stats["source_block_count"],
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
            state, target_velocity = base.linear_flow_state_target(clean, noise, time_value)

            with torch.no_grad(), base.autocast_context(config.precision):
                features, conditioning = prefix(state, time_value, labels)
            with base.autocast_context(config.precision):
                projected = train_head(features.detach(), conditioning.detach())
                internal = unpatchify_channels(
                    source,
                    projected,
                    channels=base.LATENT_SHAPE[0],
                )
            loss = F.mse_loss(internal.float(), target_velocity.float())
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
                    "train_internal_velocity_loss": float(
                        values[0].item() / values[1].item()
                    ),
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
                    source=source,
                    head=head,
                    loader=validation_loader,
                    context=context,
                    precision=config.precision,
                    batches=config.validation_batches,
                    seed=config.seed + 800_000,
                    internal_depth=config.internal_depth,
                )
                ema_metrics = validation_metrics(
                    source=source,
                    head=ema.module,
                    loader=validation_loader,
                    context=context,
                    precision=config.precision,
                    batches=config.validation_batches,
                    seed=config.seed + 800_000,
                    internal_depth=config.internal_depth,
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
                            "internal_head": head.state_dict(),
                            "internal_head_ema": ema.module.state_dict(),
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=base.DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--official-sit-repo", type=Path, default=base.DEFAULT_OFFICIAL_SIT_REPO)
    parser.add_argument("--official-ig-repo", type=Path, default=DEFAULT_IG_REPO)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE_CHECKPOINT)
    parser.add_argument("--source-state-key", choices=("ema", "model"), default="ema")
    parser.add_argument("--internal-depth", type=int, default=8)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=50_000)
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
    parser.add_argument("--save-every", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--resume",
        default="auto",
        help="auto, none, or an explicit internal-head checkpoint path",
    )
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
