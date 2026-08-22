"""Train a strict two-stage additive SiT on cached ImageNet-100 latents.

Stage ``weak`` trains the input embedding, the first ``split_depth`` blocks,
and a shallow velocity head.  Stage ``innovation`` starts from the stage-1
EMA checkpoint, freezes that complete predictor, and trains only the remaining
blocks to predict its velocity residual.  No final-model gradient can rewrite
the weak predictor.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

try:
    from experiments import train_imagenet100_sit_flow as base
    from experiments.imagenet100_sit_progressive_innovation import (
        PHASES,
        InnovationStage,
        ProgressiveInnovationSiT,
        WeakStage,
        configure_training_phase,
        create_progressive_innovation_sit,
        innovation_losses,
        trainable_parameter_names,
    )
except ModuleNotFoundError:
    import train_imagenet100_sit_flow as base
    from imagenet100_sit_progressive_innovation import (
        PHASES,
        InnovationStage,
        ProgressiveInnovationSiT,
        WeakStage,
        configure_training_phase,
        create_progressive_innovation_sit,
        innovation_losses,
        trainable_parameter_names,
    )


PROTOCOL = "imagenet100_sit_progressive_innovation_v1"
DEFAULT_OUTPUT_ROOT = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/"
    "runs/sit-s-2_progressive-innovation-split6_seed0"
)
TIME_BIN_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


@dataclass(frozen=True)
class InnovationTrainConfig:
    cache_dir: str
    output_dir: str
    official_sit_repo: str
    model_name: str
    phase: str
    split_depth: int
    stage1_checkpoint: str
    stage1_checkpoint_sha256: str
    global_step_offset: int
    global_batch_size: int
    max_steps: int
    learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    cfg_dropout: float
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


def create_model(
    sit_module,
    config: InnovationTrainConfig,
) -> ProgressiveInnovationSiT:
    return create_progressive_innovation_sit(
        sit_module,
        model_name=config.model_name,
        num_classes=base.NUM_CLASSES,
        input_size=base.LATENT_SHAPE[-1],
        cfg_dropout=config.cfg_dropout,
        split_depth=config.split_depth,
        latent_channels=base.LATENT_SHAPE[0],
    )


def validate_stage1_checkpoint(
    payload: dict,
    *,
    model_name: str,
    split_depth: int,
    cache_manifest_sha256: str,
    source_metadata: dict,
    world_size: int,
) -> None:
    if payload.get("protocol") != PROTOCOL or payload.get("phase") != "weak":
        raise ValueError("stage-1 source is not a progressive weak checkpoint")
    config = payload.get("config", {})
    mismatches = []
    if config.get("model_name") != model_name:
        mismatches.append("model_name")
    if int(config.get("split_depth", -1)) != int(split_depth):
        mismatches.append("split_depth")
    if int(config.get("world_size", world_size)) != int(world_size):
        mismatches.append("world_size")
    if payload.get("data_manifest_sha256") != cache_manifest_sha256:
        mismatches.append("data_manifest_sha256")
    if payload.get("official_sit") != source_metadata:
        mismatches.append("official_sit")
    if mismatches:
        raise ValueError("incompatible stage-1 checkpoint: " + ", ".join(mismatches))


def validate_resume(
    stored: dict,
    current: InnovationTrainConfig,
    world_size: int,
) -> None:
    immutable = (
        "cache_dir",
        "official_sit_repo",
        "model_name",
        "phase",
        "split_depth",
        "stage1_checkpoint",
        "stage1_checkpoint_sha256",
        "global_step_offset",
        "global_batch_size",
        "learning_rate",
        "weight_decay",
        "beta1",
        "beta2",
        "cfg_dropout",
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


def stage_view(
    model: ProgressiveInnovationSiT,
    phase: str,
) -> nn.Module:
    if phase == "weak":
        return WeakStage(model)
    if phase == "innovation":
        return InnovationStage(model)
    raise ValueError(f"unsupported innovation phase: {phase!r}")


@torch.inference_mode()
def validation_metrics(
    *,
    model: ProgressiveInnovationSiT,
    loader: DataLoader,
    context: base.DistributedContext,
    precision: str,
    batches: int,
    seed: int,
) -> dict[str, object]:
    generator = torch.Generator(device=context.device).manual_seed(int(seed))
    names = (
        "weak_mse",
        "cumulative_mse",
        "innovation_rms",
        "residual_rms",
        "innovation_residual_cosine",
    )
    totals = torch.zeros(len(names) + 1, device=context.device, dtype=torch.float64)
    bins = torch.zeros(
        len(TIME_BIN_EDGES) - 1,
        len(names) + 1,
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
        data = base.sample_sdvae_posterior(moments, posterior_noise)
        noise = torch.randn(data.shape, generator=generator, device=context.device)
        time_value = torch.rand(
            (len(data),), generator=generator, device=context.device
        )
        state, target = base.linear_flow_state_target(data, noise, time_value)
        with base.autocast_context(precision):
            weak, innovation, cumulative = model.forward_components(
                state, time_value, labels
            )
        weak_float = weak.float()
        innovation_float = innovation.float()
        target_float = target.float()
        residual = target_float - weak_float
        weak_mse = (weak_float - target_float).square().flatten(1).mean(1)
        cumulative_mse = (
            (cumulative.float() - target_float).square().flatten(1).mean(1)
        )
        innovation_rms = innovation_float.square().flatten(1).mean(1).sqrt()
        residual_rms = residual.square().flatten(1).mean(1).sqrt()
        dot = (innovation_float * residual).flatten(1).sum(1)
        cosine = dot / (
            innovation_float.flatten(1).norm(dim=1)
            * residual.flatten(1).norm(dim=1)
        ).clamp_min(1e-12)
        values = torch.stack(
            (weak_mse, cumulative_mse, innovation_rms, residual_rms, cosine),
            dim=1,
        ).double()
        totals[:-1] += values.sum(0)
        totals[-1] += len(values)
        for index, (lower, upper) in enumerate(
            zip(TIME_BIN_EDGES[:-1], TIME_BIN_EDGES[1:])
        ):
            mask = (time_value >= lower) & (
                time_value < upper if upper < 1.0 else time_value <= upper
            )
            if mask.any():
                bins[index, :-1] += values[mask].sum(0)
                bins[index, -1] += mask.sum()
    base.reduce_sum(totals, context)
    base.reduce_sum(bins, context)
    if totals[-1].item() == 0:
        raise RuntimeError("validation loader produced no samples")
    overall = {
        name: float((totals[index] / totals[-1]).item())
        for index, name in enumerate(names)
    }
    overall["cumulative_gain_over_weak"] = (
        overall["weak_mse"] - overall["cumulative_mse"]
    )
    by_time = []
    for index, (lower, upper) in enumerate(
        zip(TIME_BIN_EDGES[:-1], TIME_BIN_EDGES[1:])
    ):
        count = bins[index, -1]
        row: dict[str, object] = {
            "lower": lower,
            "upper": upper,
            "count": int(count.item()),
        }
        if count.item() > 0:
            row.update(
                {
                    name: float((bins[index, offset] / count).item())
                    for offset, name in enumerate(names)
                }
            )
        by_time.append(row)
    return {"overall": overall, "time_bins": by_time}


def build_metadata(
    *,
    config: InnovationTrainConfig,
    context: base.DistributedContext,
    source_metadata: dict,
    model: ProgressiveInnovationSiT,
    cache_manifest: dict,
) -> dict[str, object]:
    trainable = trainable_parameter_names(model)
    return {
        "protocol": PROTOCOL,
        "config": {**asdict(config), "world_size": context.world_size},
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "trainable_parameter_names": list(trainable),
        "official_sit": source_metadata,
        "data_manifest": cache_manifest,
        "objective": {
            "path": "x_t=(1-t)*noise+t*data",
            "target": "velocity=data-noise",
            "weak_stage": "MSE(V0, velocity)",
            "innovation_stage": "MSE(Delta1, stopgrad(velocity-V0))",
            "sampling_field": "V0+Delta1",
            "gradient_isolation": "stage-1 EMA frozen before stage 2",
        },
    }


def train(args: argparse.Namespace) -> None:
    context = base.initialize_distributed(args.device)
    try:
        if args.global_batch_size % context.world_size:
            raise ValueError("--global-batch-size must be divisible by world size")
        if args.phase not in PHASES:
            raise ValueError(f"unsupported phase: {args.phase!r}")
        local_batch_size = args.global_batch_size // context.world_size
        base.configure_runtime(args.seed, context.rank, args.allow_tf32)

        cache_dir = args.cache_dir.expanduser().resolve()
        output_dir = args.output_dir.expanduser().resolve()
        cache_manifest_path = cache_dir / "manifest.json"
        cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
        cache_manifest_sha256 = base.sha256_file(cache_manifest_path)
        if cache_manifest.get("format") != "eqvae_imagenet100_cmc_sdvae_moments_v1":
            raise ValueError(f"unsupported data manifest: {cache_manifest_path}")

        sit_module, source_metadata = base.load_official_sit_module(
            args.official_sit_repo.expanduser().resolve(),
            verify_source=args.verify_sit_source,
        )
        stage1_payload: dict | None = None
        stage1_path = ""
        stage1_hash = ""
        global_step_offset = 0
        initial_rng_states: list[dict] | None = None
        if args.phase == "innovation":
            if args.stage1_checkpoint is None:
                raise ValueError("innovation phase requires --stage1-checkpoint")
            resolved = args.stage1_checkpoint.expanduser().resolve()
            stage1_path = str(resolved)
            stage1_hash = base.sha256_file(resolved)
            stage1_payload = torch.load(
                resolved, map_location="cpu", weights_only=False
            )
            validate_stage1_checkpoint(
                stage1_payload,
                model_name=args.model,
                split_depth=args.split_depth,
                cache_manifest_sha256=cache_manifest_sha256,
                source_metadata=source_metadata,
                world_size=context.world_size,
            )
            global_step_offset = int(stage1_payload["total_step"])
            initial_rng_states = stage1_payload.get("rng_states")
            if not isinstance(initial_rng_states, list) or len(initial_rng_states) != context.world_size:
                raise ValueError("stage-1 checkpoint lacks matching per-rank RNG states")

        config = InnovationTrainConfig(
            cache_dir=str(cache_dir),
            output_dir=str(output_dir),
            official_sit_repo=str(args.official_sit_repo.expanduser().resolve()),
            model_name=args.model,
            phase=args.phase,
            split_depth=int(args.split_depth),
            stage1_checkpoint=stage1_path,
            stage1_checkpoint_sha256=stage1_hash,
            global_step_offset=global_step_offset,
            global_batch_size=int(args.global_batch_size),
            max_steps=int(args.max_steps),
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            beta1=float(args.beta1),
            beta2=float(args.beta2),
            cfg_dropout=float(args.cfg_dropout),
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
            config.global_batch_size,
            config.max_steps,
            config.log_every,
            config.save_every,
        ) < 1:
            raise ValueError("batch, step and logging values must be positive")
        if not 0 <= config.cfg_dropout < 1:
            raise ValueError("--cfg-dropout must be in [0,1)")

        raw_model = create_model(sit_module, config)
        if stage1_payload is not None:
            raw_model.load_state_dict(stage1_payload["ema"], strict=True)
        configure_training_phase(raw_model, config.phase)
        raw_model.to(context.device)
        ema = base.ModelEMA(raw_model)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in raw_model.parameters() if parameter.requires_grad],
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
        phase_step = 0
        restored_rng: dict | None = None
        if resume_path is not None:
            checkpoint = torch.load(
                resume_path, map_location=context.device, weights_only=False
            )
            if checkpoint.get("protocol") != PROTOCOL:
                raise ValueError(f"unexpected checkpoint protocol: {checkpoint.get('protocol')!r}")
            validate_resume(checkpoint["config"], config, context.world_size)
            if checkpoint.get("data_manifest_sha256") != cache_manifest_sha256:
                raise ValueError("checkpoint data manifest does not match")
            if checkpoint.get("official_sit") != source_metadata:
                raise ValueError("checkpoint official SiT source does not match")
            raw_model.load_state_dict(checkpoint["model"], strict=True)
            ema.load_state_dict(checkpoint["ema"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            phase_step = int(checkpoint["phase_step"])
            rng_states = checkpoint.get("rng_states")
            if not isinstance(rng_states, list) or len(rng_states) != context.world_size:
                raise ValueError("checkpoint lacks matching per-rank RNG states")
            restored_rng = rng_states[context.rank]
        elif initial_rng_states is not None:
            restored_rng = initial_rng_states[context.rank]

        view: nn.Module = stage_view(raw_model, config.phase)
        if config.compile:
            view = torch.compile(
                view,
                mode=config.compile_mode,
                fullgraph=True,
                dynamic=False,
            )
        if context.world_size > 1:
            view = DDP(
                view,
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

        raw_model.train()
        ema.module.eval()
        total_start_step = config.global_step_offset + phase_step
        batches = base.infinite_train_batches(
            train_loader, train_sampler, total_start_step
        )

        if context.is_main:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            metadata = build_metadata(
                config=config,
                context=context,
                source_metadata=source_metadata,
                model=raw_model,
                cache_manifest=cache_manifest,
            )
            base.atomic_json_dump(metadata, output_dir / "run_config.json")
            print(
                json.dumps(
                    {
                        "event": "start",
                        "phase": config.phase,
                        "phase_step": phase_step,
                        "total_step": total_start_step,
                        "split_depth": config.split_depth,
                        "parameters": metadata["parameter_count"],
                        "trainable_parameters": metadata["trainable_parameter_count"],
                        "world_size": context.world_size,
                        "global_batch": config.global_batch_size,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        base.barrier(context)

        metrics_path = output_dir / "train_metrics.jsonl"
        running = torch.zeros(5, device=context.device, dtype=torch.float64)
        running_steps = 0
        interval_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)

        for next_phase_step in range(phase_step + 1, config.max_steps + 1):
            moments, labels = next(batches)
            moments = moments.to(context.device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(context.device, dtype=torch.long, non_blocking=True)
            posterior_noise = torch.randn(
                (len(moments), *base.LATENT_SHAPE), device=context.device
            )
            data = base.sample_sdvae_posterior(moments, posterior_noise)
            noise = torch.randn_like(data)
            time_value = torch.rand((len(data),), device=context.device)
            state, target = base.linear_flow_state_target(data, noise, time_value)

            with base.autocast_context(config.precision):
                output = view(state, time_value, labels)
            if config.phase == "weak":
                weak = output
                loss = (weak.float() - target.float()).square().mean()
                values = torch.stack(
                    (
                        loss.detach(),
                        loss.detach(),
                        torch.zeros_like(loss),
                        (target.float() - weak.detach().float()).square().mean().sqrt(),
                        loss.detach(),
                    )
                )
            else:
                weak, innovation, _ = output
                losses = innovation_losses(weak, innovation, target)
                loss = losses["optimized"]
                values = torch.stack(
                    (
                        losses["optimized"].detach(),
                        losses["weak"].detach(),
                        losses["innovation_rms"].detach(),
                        losses["residual_rms"].detach(),
                        losses["cumulative"].detach(),
                    )
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite {config.phase} loss at phase step {next_phase_step}"
                )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            ema.update(config.ema_decay)
            running += values.double()
            running_steps += 1

            total_step = config.global_step_offset + next_phase_step
            if next_phase_step % config.log_every == 0 or next_phase_step == config.max_steps:
                torch.cuda.synchronize(context.device)
                elapsed = time.perf_counter() - interval_started
                packed = torch.cat(
                    (
                        running,
                        torch.tensor([running_steps], device=context.device, dtype=torch.float64),
                    )
                )
                base.reduce_sum(packed, context)
                elapsed_tensor = torch.tensor(elapsed, device=context.device)
                base.reduce_max(elapsed_tensor, context)
                memory = torch.tensor(
                    torch.cuda.max_memory_allocated(context.device) / 2**30,
                    device=context.device,
                )
                base.reduce_max(memory, context)
                means = packed[:5] / packed[5]
                row = {
                    "phase": config.phase,
                    "phase_step": next_phase_step,
                    "total_step": total_step,
                    "optimized_loss": float(means[0].item()),
                    "weak_loss": float(means[1].item()),
                    "innovation_rms": float(means[2].item()),
                    "residual_rms": float(means[3].item()),
                    "cumulative_loss": float(means[4].item()),
                    "steps_per_second": float(running_steps / elapsed_tensor.item()),
                    "images_per_second": float(
                        running_steps * config.global_batch_size / elapsed_tensor.item()
                    ),
                    "max_allocated_gb": float(memory.item()),
                    "learning_rate": optimizer.param_groups[0]["lr"],
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
                next_phase_step % config.validation_every == 0
                or next_phase_step == config.max_steps
            )
            if should_validate:
                paused = time.perf_counter()
                raw_metrics = validation_metrics(
                    model=raw_model,
                    loader=validation_loader,
                    context=context,
                    precision=config.precision,
                    batches=config.validation_batches,
                    seed=config.seed + 810_000,
                )
                ema_metrics = validation_metrics(
                    model=ema.module,
                    loader=validation_loader,
                    context=context,
                    precision=config.precision,
                    batches=config.validation_batches,
                    seed=config.seed + 810_000,
                )
                raw_model.train()
                if context.is_main:
                    row = {
                        "phase": config.phase,
                        "phase_step": next_phase_step,
                        "total_step": total_step,
                        "raw_validation": raw_metrics,
                        "ema_validation": ema_metrics,
                    }
                    with metrics_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
                    print(json.dumps(row, sort_keys=True), flush=True)
                interval_started += time.perf_counter() - paused

            should_save = (
                next_phase_step % config.save_every == 0
                or next_phase_step == config.max_steps
            )
            if should_save:
                paused = time.perf_counter()
                rng_states = base.gather_rng_states(context)
                if context.is_main:
                    checkpoint_path = (
                        output_dir / "checkpoints" / f"step_{total_step:08d}.pt"
                    )
                    base.atomic_torch_save(
                        {
                            "protocol": PROTOCOL,
                            "phase": config.phase,
                            "phase_step": next_phase_step,
                            "total_step": total_step,
                            "model": raw_model.state_dict(),
                            "ema": ema.state_dict(),
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
                interval_started += time.perf_counter() - paused
    finally:
        base.cleanup_distributed(context)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=base.DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--official-sit-repo", type=Path, default=base.DEFAULT_OFFICIAL_SIT_REPO
    )
    parser.add_argument("--model", default="SiT-S/2")
    parser.add_argument("--split-depth", type=int, default=6)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--cfg-dropout", type=float, default=0.1)
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
    parser.add_argument("--resume", default="auto")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
