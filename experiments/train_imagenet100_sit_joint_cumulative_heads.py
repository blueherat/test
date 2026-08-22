#!/usr/bin/env python3
"""Jointly train cumulative velocity readouts on a frozen v800 SiT."""

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
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from experiments import train_imagenet100_sit_flow as base
    from experiments.imagenet100_sit_joint_cumulative_heads import (
        DEFAULT_DEPTHS,
        CumulativeReadoutStack,
        FrozenMultiDepthPrefix,
        create_joint_cumulative_parts,
        sequence_losses,
        source_velocity_from_final_features,
    )
except ModuleNotFoundError:
    import train_imagenet100_sit_flow as base
    from imagenet100_sit_joint_cumulative_heads import (
        DEFAULT_DEPTHS,
        CumulativeReadoutStack,
        FrozenMultiDepthPrefix,
        create_joint_cumulative_parts,
        sequence_losses,
        source_velocity_from_final_features,
    )


PROTOCOL = "imagenet100_sit_frozen_joint_cumulative_heads_v1"
DEFAULT_SOURCE = (
    Path("/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/")
    / "sit-s-2_seed0/checkpoints/step_00800000.pt"
)
DEFAULT_OUTPUT = (
    Path("/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/")
    / "sit-s-2_v800-frozen-joint-cumulative-heads_seed0"
)
TIME_BIN_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def parse_depths(value: str) -> tuple[int, ...]:
    depths = tuple(int(item) for item in value.split(",") if item.strip())
    if not depths or tuple(sorted(set(depths))) != depths:
        raise argparse.ArgumentTypeError("depths must be unique and increasing")
    return depths


@dataclass(frozen=True)
class JointTrainConfig:
    cache_dir: str
    output_dir: str
    official_sit_repo: str
    source_checkpoint: str
    source_checkpoint_sha256: str
    source_state_key: str
    source_step: int
    model_name: str
    cfg_dropout: float
    depths: tuple[int, ...]
    global_batch_size: int
    max_steps: int
    learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    ema_decay: float
    monotonic_weight: float
    contraction_ratio: float
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


def validate_resume(stored: dict, current: JointTrainConfig, world_size: int) -> None:
    current_values = asdict(current)
    mismatches = [
        f"{key}: checkpoint={stored.get(key)!r}, current={value!r}"
        for key, value in current_values.items()
        if key not in {"output_dir", "max_steps", "log_every", "validation_every", "save_every"}
        and stored.get(key) != value
    ]
    if int(stored.get("world_size", world_size)) != world_size:
        mismatches.append("world_size")
    if mismatches:
        raise ValueError("incompatible resume configuration:\n  " + "\n  ".join(mismatches))


def consecutive_cosines(innovations: tuple[torch.Tensor, ...]) -> torch.Tensor:
    if len(innovations) < 3:
        return innovations[0].new_empty((len(innovations[0]), 0))
    rows = []
    # The first value is the base prediction; only later values are innovations.
    correction = innovations[1:]
    for previous, following in zip(correction[:-1], correction[1:], strict=True):
        left = previous.float().flatten(1)
        right = following.float().flatten(1)
        cosine = (left * right).sum(1) / (
            left.norm(dim=1) * right.norm(dim=1)
        ).clamp_min(1e-12)
        rows.append(cosine)
    return torch.stack(rows, dim=1)


@torch.inference_mode()
def validation_metrics(
    *,
    source: nn.Module,
    prefix: nn.Module,
    readouts: nn.Module,
    depths: tuple[int, ...],
    loader,
    context: base.DistributedContext,
    precision: str,
    batches: int,
    seed: int,
    monotonic_weight: float,
    contraction_ratio: float,
) -> dict[str, object]:
    generator = torch.Generator(device=context.device).manual_seed(int(seed))
    stage_count = len(depths)
    transition_count = max(0, stage_count - 2)
    # stage MSEs, strong MSE, optimized/supervised/monotonic, strict fraction,
    # and consecutive innovation cosines.
    width = stage_count + 5 + transition_count
    totals = torch.zeros(width + 1, device=context.device, dtype=torch.float64)
    prefix.eval()
    readouts.eval()
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
            (len(clean),), generator=generator, device=context.device
        )
        state, target = base.linear_flow_state_target(clean, noise, time_value)
        with base.autocast_context(precision):
            features, conditioning = prefix(state, time_value, labels)
            outputs, innovations = readouts(features, conditioning)
            strong = source_velocity_from_final_features(
                source,
                features[-1],
                conditioning,
                latent_channels=base.LATENT_SHAPE[0],
            )
        losses = sequence_losses(
            outputs,
            target,
            monotonic_weight=monotonic_weight,
            contraction_ratio=contraction_ratio,
        )
        stage_mse = losses["per_sample_mse"].double().mean(0)
        strong_mse = (strong.float() - target.float()).square().flatten(1).mean(1).double().mean()
        cosines = consecutive_cosines(innovations).double()
        row = torch.cat(
            (
                stage_mse,
                strong_mse.reshape(1),
                losses["optimized"].double().reshape(1),
                losses["supervised"].double().reshape(1),
                losses["monotonic"].double().reshape(1),
                losses["strict_monotonic_fraction"].double().reshape(1),
                cosines.mean(0) if transition_count else totals.new_empty(0),
            )
        )
        totals[:-1] += row * len(moments)
        totals[-1] += len(moments)
    base.reduce_sum(totals, context)
    if totals[-1].item() == 0:
        raise RuntimeError("validation loader produced no samples")
    means = totals[:-1] / totals[-1]
    offset = 0
    result: dict[str, object] = {
        "stage_mse": {
            f"d{depth}": float(means[index]) for index, depth in enumerate(depths)
        }
    }
    offset += stage_count
    for name in ("strong_mse", "optimized", "supervised", "monotonic", "strict_monotonic_fraction"):
        result[name] = float(means[offset])
        offset += 1
    result["consecutive_innovation_cosine"] = {
        f"d{left}_to_d{middle}_vs_d{middle}_to_d{right}": float(means[offset + index])
        for index, (left, middle, right) in enumerate(
            zip(depths[:-2], depths[1:-1], depths[2:], strict=True)
        )
    }
    return result


def train(args: argparse.Namespace) -> None:
    context = base.initialize_distributed(args.device)
    try:
        if args.global_batch_size % context.world_size:
            raise ValueError("--global-batch-size must be divisible by world size")
        base.configure_runtime(args.seed, context.rank, args.allow_tf32)
        source_path = args.source_checkpoint.expanduser().resolve()
        source_payload = torch.load(
            source_path, map_location="cpu", weights_only=False, mmap=True
        )
        sit_module, source_metadata = base.load_official_sit_module(
            args.official_sit_repo.expanduser().resolve(),
            verify_source=args.verify_sit_source,
        )
        if source_payload.get("official_sit") != source_metadata:
            raise ValueError("source checkpoint official SiT metadata does not match")
        source_config = source_payload["config"]
        config = JointTrainConfig(
            cache_dir=str(args.cache_dir.expanduser().resolve()),
            output_dir=str(args.output_dir.expanduser().resolve()),
            official_sit_repo=str(args.official_sit_repo.expanduser().resolve()),
            source_checkpoint=str(source_path),
            source_checkpoint_sha256=base.sha256_file(source_path),
            source_state_key=args.source_state_key,
            source_step=int(source_payload["step"]),
            model_name=str(source_config["model_name"]),
            cfg_dropout=float(source_config["cfg_dropout"]),
            depths=tuple(args.depths),
            global_batch_size=int(args.global_batch_size),
            max_steps=int(args.max_steps),
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            beta1=float(args.beta1),
            beta2=float(args.beta2),
            ema_decay=float(args.ema_decay),
            monotonic_weight=float(args.monotonic_weight),
            contraction_ratio=float(args.contraction_ratio),
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
        source = sit_module.SiT_models[config.model_name](
            input_size=base.LATENT_SHAPE[-1],
            num_classes=base.NUM_CLASSES,
            class_dropout_prob=config.cfg_dropout,
        )
        source.load_state_dict(source_payload[config.source_state_key], strict=True)
        del source_payload
        gc.collect()
        prefix, readouts = create_joint_cumulative_parts(
            sit_module,
            source,
            depths=config.depths,
            latent_channels=base.LATENT_SHAPE[0],
        )
        prefix.to(context.device).eval()
        readouts.to(context.device).train()
        ema = base.ModelEMA(readouts)
        optimizer = torch.optim.AdamW(
            readouts.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
            fused=True,
        )
        cache_dir = Path(config.cache_dir)
        output_dir = Path(config.output_dir)
        manifest_path = cache_dir / "manifest.json"
        cache_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_sha = base.sha256_file(manifest_path)
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

        resume_path = base.resolve_resume(args.resume, output_dir)
        start_step = 0
        restored_rng = None
        if resume_path is not None:
            checkpoint = torch.load(resume_path, map_location=context.device, weights_only=False)
            if checkpoint.get("protocol") != PROTOCOL:
                raise ValueError("unexpected checkpoint protocol")
            validate_resume(checkpoint["config"], config, context.world_size)
            if checkpoint.get("data_manifest_sha256") != manifest_sha:
                raise ValueError("data manifest mismatch")
            readouts.load_state_dict(checkpoint["readouts"], strict=True)
            ema.module.load_state_dict(checkpoint["readouts_ema"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_step = int(checkpoint["step"])
            rng_states = checkpoint.get("rng_states")
            if not isinstance(rng_states, list) or len(rng_states) != context.world_size:
                raise ValueError("checkpoint lacks matching RNG states")
            restored_rng = rng_states[context.rank]
        else:
            ema.module.load_state_dict(readouts.state_dict())
        if restored_rng is not None:
            base.restore_rng_state(restored_rng, context.device)

        train_prefix: nn.Module = prefix
        train_readouts: nn.Module = readouts
        if config.compile:
            train_prefix = torch.compile(
                prefix, mode=config.compile_mode, fullgraph=True, dynamic=False
            )
            train_readouts = torch.compile(
                readouts, mode=config.compile_mode, fullgraph=True, dynamic=False
            )
        if context.world_size > 1:
            train_readouts = DDP(
                train_readouts,
                device_ids=[context.local_rank],
                output_device=context.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
                static_graph=True,
            )
        batches = base.infinite_train_batches(train_loader, train_sampler, start_step)
        if context.is_main:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            metadata = {
                "protocol": PROTOCOL,
                "config": {**asdict(config), "world_size": context.world_size},
                "official_sit": source_metadata,
                "data_manifest": cache_manifest,
                "trainable_parameter_count": sum(p.numel() for p in readouts.parameters()),
                "source_parameter_count": sum(p.numel() for p in source.parameters()),
                "objective": {
                    "cumulative": "V_dk=sum_{j<=k} innovation_j",
                    "supervision": "mean_k MSE(V_dk, clean-noise)",
                    "monotonic_penalty": (
                        "mean relu(MSE_next-rho^2*MSE_previous)"
                    ),
                    "source_frozen": True,
                },
            }
            base.atomic_json_dump(metadata, output_dir / "run_config.json")
            print(json.dumps({"event": "start", **metadata["config"]}, sort_keys=True), flush=True)
        base.barrier(context)

        metrics_path = output_dir / "train_metrics.jsonl"
        running = torch.zeros(5, device=context.device, dtype=torch.float64)
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
            clean = base.sample_sdvae_posterior(moments, posterior_noise)
            noise = torch.randn_like(clean)
            time_value = torch.rand((len(clean),), device=context.device)
            state, target = base.linear_flow_state_target(clean, noise, time_value)
            with torch.no_grad(), base.autocast_context(config.precision):
                features, conditioning = train_prefix(state, time_value, labels)
            detached_features = tuple(feature.detach() for feature in features)
            with base.autocast_context(config.precision):
                outputs, innovations = train_readouts(
                    detached_features, conditioning.detach()
                )
            losses = sequence_losses(
                outputs,
                target,
                monotonic_weight=config.monotonic_weight,
                contraction_ratio=config.contraction_ratio,
            )
            loss = losses["optimized"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            ema.update(config.ema_decay)
            cosine = consecutive_cosines(innovations)
            running += torch.stack(
                (
                    loss.detach().double(),
                    losses["supervised"].detach().double(),
                    losses["monotonic"].detach().double(),
                    losses["strict_monotonic_fraction"].detach().double(),
                    cosine.detach().double().mean() if cosine.numel() else loss.new_zeros(()).double(),
                )
            )
            running_steps += 1
            if step % config.log_every == 0 or step == config.max_steps:
                torch.cuda.synchronize(context.device)
                elapsed = time.perf_counter() - interval_started
                packed = torch.cat((running, running.new_tensor([running_steps])))
                base.reduce_sum(packed, context)
                elapsed_tensor = torch.tensor(elapsed, device=context.device)
                base.reduce_max(elapsed_tensor, context)
                means = packed[:5] / packed[5]
                row = {
                    "step": step,
                    "optimized_loss": float(means[0]),
                    "supervised_loss": float(means[1]),
                    "monotonic_loss": float(means[2]),
                    "strict_monotonic_fraction": float(means[3]),
                    "consecutive_innovation_cosine": float(means[4]),
                    "steps_per_second": running_steps / float(elapsed_tensor),
                    "images_per_second": running_steps * config.global_batch_size / float(elapsed_tensor),
                    "max_allocated_gb": torch.cuda.max_memory_allocated(context.device) / 2**30,
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
                raw_metrics = validation_metrics(
                    source=source,
                    prefix=prefix,
                    readouts=readouts,
                    depths=config.depths,
                    loader=validation_loader,
                    context=context,
                    precision=config.precision,
                    batches=config.validation_batches,
                    seed=config.seed + 820_000,
                    monotonic_weight=config.monotonic_weight,
                    contraction_ratio=config.contraction_ratio,
                )
                ema_metrics = validation_metrics(
                    source=source,
                    prefix=prefix,
                    readouts=ema.module,
                    depths=config.depths,
                    loader=validation_loader,
                    context=context,
                    precision=config.precision,
                    batches=config.validation_batches,
                    seed=config.seed + 820_000,
                    monotonic_weight=config.monotonic_weight,
                    contraction_ratio=config.contraction_ratio,
                )
                readouts.train()
                if context.is_main:
                    row = {"step": step, "raw_validation": raw_metrics, "ema_validation": ema_metrics}
                    with metrics_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
                    print(json.dumps(row, sort_keys=True), flush=True)
                interval_started += time.perf_counter() - paused

            should_save = step % config.save_every == 0 or step == config.max_steps
            if should_save:
                paused = time.perf_counter()
                rng_states = base.gather_rng_states(context)
                if context.is_main:
                    checkpoint_path = output_dir / "checkpoints" / f"step_{step:08d}.pt"
                    base.atomic_torch_save(
                        {
                            "protocol": PROTOCOL,
                            "step": step,
                            "readouts": readouts.state_dict(),
                            "readouts_ema": ema.module.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "rng_states": rng_states,
                            "config": {**asdict(config), "world_size": context.world_size},
                            "official_sit": source_metadata,
                            "data_manifest_sha256": manifest_sha,
                        },
                        checkpoint_path,
                    )
                    print(json.dumps({"event": "checkpoint", "path": str(checkpoint_path)}), flush=True)
                base.barrier(context)
                interval_started += time.perf_counter() - paused
    finally:
        base.cleanup_distributed(context)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=base.DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--official-sit-repo", type=Path, default=base.DEFAULT_OFFICIAL_SIT_REPO)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--source-state-key", choices=("ema", "model"), default="ema")
    parser.add_argument("--depths", type=parse_depths, default=DEFAULT_DEPTHS)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--monotonic-weight", type=float, default=0.0)
    parser.add_argument("--contraction-ratio", type=float, default=0.98)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verify-sit-source", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--validation-every", type=int, default=2_000)
    parser.add_argument("--validation-batches", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", default="auto")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
