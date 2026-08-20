#!/usr/bin/env python3
"""Train a frozen-generator CAFM tangent critic on ImageNet-100 SiT-S/2."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from experiments.imagenet100_sit_cafm_tangent import (
    CAFM_REPOSITORY,
    CAFM_REVISION,
    TangentJVP,
    critic_from_sit_state,
    lsgan_tangent_losses,
)
from experiments.train_imagenet100_sit_flow import (
    DEFAULT_CACHE_DIR,
    DEFAULT_OFFICIAL_SIT_REPO,
    LATENT_SHAPE,
    NUM_CLASSES,
    NpyMomentsDataset,
    linear_flow_state_target,
    load_official_sit_module,
    sample_sdvae_posterior,
    sha256_file,
)


DEFAULT_STRONG = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/"
    "sit-s-2_seed0/checkpoints/step_00800000.pt"
)
DEFAULT_OUTPUT = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/"
    "cafm_tangent_predictivity_v1/critics/seed0"
)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def autocast_context(precision: str):
    if precision == "fp32":
        return nullcontext()
    if precision == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    raise ValueError(f"unsupported precision: {precision}")


def make_loader(
    cache_dir: Path,
    split: str,
    *,
    batch_size: int,
    workers: int,
    seed: int,
    shuffle: bool,
    drop_last: bool,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    dataset = NpyMomentsDataset(cache_dir, split)
    generator = torch.Generator().manual_seed(int(seed))
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=int(world_size),
            rank=int(rank),
            shuffle=bool(shuffle),
            seed=int(seed),
            drop_last=bool(drop_last),
        )
    kwargs = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle) if sampler is None else False,
        "sampler": sampler,
        "num_workers": int(workers),
        "pin_memory": True,
        "drop_last": bool(drop_last),
        "persistent_workers": int(workers) > 0,
        "generator": generator,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = 4
    return DataLoader(**kwargs)


def next_batch(iterator, loader, epoch: int):
    try:
        return next(iterator), iterator, epoch
    except StopIteration:
        epoch += 1
        if isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(epoch)
        iterator = iter(loader)
        return next(iterator), iterator, epoch


def load_models(args, device: torch.device):
    checkpoint = torch.load(args.strong_checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    sit_module, source_metadata = load_official_sit_module(
        args.official_sit_repo, verify_source=True
    )
    if checkpoint.get("official_sit") != source_metadata:
        raise ValueError("strong checkpoint does not match the audited SiT source")
    if int(checkpoint["step"]) != 800_000:
        raise ValueError("the preregistered predictivity screen requires v800 strong")
    if str(config.get("prediction_target", "velocity")) != "velocity":
        raise ValueError("strong checkpoint must be a native velocity model")

    strong = sit_module.SiT_models[str(config["model_name"])](
        input_size=LATENT_SHAPE[-1],
        num_classes=NUM_CLASSES,
        class_dropout_prob=float(config["cfg_dropout"]),
    )
    strong.load_state_dict(checkpoint["ema"], strict=True)
    strong.to(device).eval().requires_grad_(False)

    critic = critic_from_sit_state(
        sit_module=sit_module,
        model_name=str(config["model_name"]),
        state_dict=checkpoint["ema"],
        input_size=LATENT_SHAPE[-1],
        num_classes=NUM_CLASSES,
        class_dropout_prob=float(config["cfg_dropout"]),
    )
    critic.to(device)
    metadata = {
        "strong_checkpoint": str(args.strong_checkpoint),
        "strong_checkpoint_sha256": sha256_file(args.strong_checkpoint),
        "strong_step": int(checkpoint["step"]),
        "strong_state_key": "ema",
        "strong_protocol": checkpoint.get("protocol"),
        "model_name": str(config["model_name"]),
        "official_sit": source_metadata,
        "cafm_repository": CAFM_REPOSITORY,
        "cafm_revision": CAFM_REVISION,
        "time_orientation": "native_noise_t0_to_data_t1",
    }
    del checkpoint
    return strong, critic, metadata


def prepare_teacher_batch(
    moments: torch.Tensor,
    labels: torch.Tensor,
    *,
    device: torch.device,
    generator: torch.Generator | None = None,
    class_dropout_probability: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    moments = moments.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    if not 0.0 <= class_dropout_probability <= 1.0:
        raise ValueError("class dropout probability must lie in [0, 1]")
    if class_dropout_probability > 0.0:
        keep = torch.rand(
            labels.shape,
            device=device,
            generator=generator,
        ) >= float(class_dropout_probability)
        labels = torch.where(keep, labels, torch.full_like(labels, NUM_CLASSES))
    posterior_noise = torch.randn(
        (moments.shape[0], *LATENT_SHAPE),
        device=device,
        generator=generator,
    )
    data = sample_sdvae_posterior(moments, posterior_noise)
    noise = torch.randn(data.shape, device=device, generator=generator)
    time_value = torch.rand(data.shape[0], device=device, generator=generator)
    state, real_velocity = linear_flow_state_target(data, noise, time_value)
    return state, time_value, labels, real_velocity


@torch.no_grad()
def evaluate(
    *,
    critic,
    strong,
    loader: DataLoader,
    device: torch.device,
    precision: str,
    batches: int,
    seed: int,
    centering_scale: float,
    class_dropout_probability: float,
) -> dict[str, float]:
    critic.eval()
    strong.eval()
    wrapper = TangentJVP(critic)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    sums = torch.zeros(10, device=device, dtype=torch.float64)
    count = 0
    for batch_index, (moments, labels) in enumerate(loader):
        if batch_index >= batches:
            break
        state, time_value, labels, real_velocity = prepare_teacher_batch(
            moments,
            labels,
            device=device,
            generator=generator,
            class_dropout_probability=class_dropout_probability,
        )
        with autocast_context(precision):
            fake_velocity = strong(state, time_value, labels)
        # Forward-over-reverse through math SDPA is kept in FP32.  PyTorch 2.9
        # currently produces a mixed-gradient dtype error for BF16 JVP
        # backward; the frozen strong forward remains BF16 when requested.
        velocities = torch.stack((real_velocity, fake_velocity.float()))
        time_velocities = torch.ones(
            (2, state.shape[0]), device=device, dtype=time_value.dtype
        )
        values, logits = wrapper(
            state.float(), time_value.float(), labels, velocities, time_velocities
        )
        losses = lsgan_tangent_losses(
            values[0], logits[0], logits[1], centering_scale=centering_scale
        )
        real = logits[0].float()
        fake = logits[1].float()
        batch_count = state.shape[0]
        metrics = torch.stack(
            (
                losses["total"].float(),
                losses["real"].float(),
                losses["fake"].float(),
                losses["centering"].float(),
                real.mean(),
                fake.mean(),
                (real - fake).mean(),
                (real > 0).float().mean(),
                (fake < 0).float().mean(),
            )
        ).double()
        sums[:9] += metrics * batch_count
        sums[9] += batch_count
        count += batch_count
    if dist.is_initialized():
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
    count = int(sums[9].item())
    if count == 0:
        raise RuntimeError("validation loader produced no batches")
    means = (sums[:9] / sums[9]).cpu().tolist()
    keys = (
        "loss",
        "real_loss",
        "fake_loss",
        "centering",
        "real_logit",
        "fake_logit",
        "margin",
        "real_sign_accuracy",
        "fake_sign_accuracy",
    )
    return {**dict(zip(keys, means)), "samples": int(count)}


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strong-checkpoint", type=Path, default=DEFAULT_STRONG)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--official-sit-repo", type=Path, default=DEFAULT_OFFICIAL_SIT_REPO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--centering-scale", type=float, default=1e-3)
    parser.add_argument("--class-dropout-probability", type=float, default=0.1)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--validation-batches", type=int, default=32)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(
        args.steps,
        args.batch_size,
        args.accumulation_steps,
        args.log_every,
        args.validate_every,
        args.validation_batches,
        args.save_every,
    ) < 1:
        raise ValueError("step, batch, logging, validation and save values must be positive")
    if not 0.0 <= args.class_dropout_probability <= 1.0:
        raise ValueError("class dropout probability must lie in [0, 1]")
    args.strong_checkpoint = args.strong_checkpoint.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.official_sit_repo = args.official_sit_repo.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "checkpoints").mkdir(exist_ok=True)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
        )
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device)
    is_main = rank == 0
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CAFM tangent training requires CUDA")
    torch.cuda.set_device(device)
    process_seed = int(args.seed) + rank * 100_003
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    torch.cuda.manual_seed(process_seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # PyTorch 2.9 flash/memory-efficient SDPA does not implement forward AD.
    # CAFM differentiates the critic with torch.func.jvp, so force the exact
    # math SDPA backend instead of silently replacing JVP with another loss.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    torch.set_float32_matmul_precision("high")

    strong, critic, model_metadata = load_models(args, device)
    optimizer = torch.optim.AdamW(
        critic.parameters(),
        lr=args.learning_rate,
        betas=(0.0, 0.95),
        weight_decay=0.0,
        fused=True,
    )
    start_step = 0
    best_loss = float("inf")
    if args.resume:
        resume = torch.load(args.resume, map_location=device, weights_only=False)
        critic.load_state_dict(resume["critic"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        start_step = int(resume["step"])
        best_loss = float(resume.get("best_validation_loss", best_loss))
    wrapper = TangentJVP(critic)
    if world_size > 1:
        # This is the ordering used by the official CAFM ImageNet code:
        # DDP owns the JVP wrapper so gradient synchronization covers the
        # forward-mode discriminator computation itself.
        wrapper = DistributedDataParallel(
            wrapper,
            device_ids=[device.index],
            output_device=device.index,
            broadcast_buffers=False,
        )

    train_loader = make_loader(
        args.cache_dir,
        "train",
        batch_size=args.batch_size,
        workers=args.workers,
        seed=args.seed + 10,
        shuffle=True,
        drop_last=True,
        rank=rank,
        world_size=world_size,
    )
    validation_loader = make_loader(
        args.cache_dir,
        "validation",
        batch_size=args.batch_size,
        workers=min(args.workers, 2),
        seed=args.seed + 20,
        shuffle=False,
        drop_last=False,
        rank=rank,
        world_size=world_size,
    )
    train_epoch = (
        start_step * args.batch_size * args.accumulation_steps * world_size
    ) // len(train_loader.dataset)
    if isinstance(train_loader.sampler, DistributedSampler):
        train_loader.sampler.set_epoch(train_epoch)
    iterator = iter(train_loader)
    config = {
        **vars(args),
        "strong_checkpoint": str(args.strong_checkpoint),
        "cache_dir": str(args.cache_dir),
        "official_sit_repo": str(args.official_sit_repo),
        "output_dir": str(args.output_dir),
        "resume": str(args.resume) if args.resume else None,
        "optimizer": "AdamW",
        "betas": [0.0, 0.95],
        "weight_decay": 0.0,
        "local_batch_size": args.batch_size,
        "world_size": world_size,
        "effective_batch_size": (
            args.batch_size * args.accumulation_steps * world_size
        ),
        "actual_device": str(device),
        "critic_compute_precision": "fp32",
        "generator_update": False,
        "real_label": 1.0,
        "fake_label": -1.0,
    }
    if is_main:
        atomic_json(
            {"config": config, "model": model_metadata}, args.output_dir / "run.json"
        )
        print(json.dumps({"config": config, "model": model_metadata}, indent=2), flush=True)

    train_csv = args.output_dir / "train_log.csv"
    started = time.monotonic()
    critic.train()
    for step in range(start_step + 1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        accumulated = torch.zeros(7, device=device, dtype=torch.float64)
        for accumulation_index in range(args.accumulation_steps):
            sync_context = nullcontext()
            if world_size > 1 and accumulation_index + 1 < args.accumulation_steps:
                sync_context = wrapper.no_sync()
            with sync_context:
                (moments, labels), iterator, train_epoch = next_batch(
                    iterator, train_loader, train_epoch
                )
                state, time_value, labels, real_velocity = prepare_teacher_batch(
                    moments,
                    labels,
                    device=device,
                    class_dropout_probability=args.class_dropout_probability,
                )
                with torch.no_grad(), autocast_context(args.precision):
                    fake_velocity = strong(state, time_value, labels)
                values, logits = wrapper(
                    state.float(),
                    time_value.float(),
                    labels,
                    torch.stack((real_velocity, fake_velocity.float())),
                    torch.ones(
                        (2, state.shape[0]), device=device, dtype=time_value.dtype
                    ),
                )
                losses = lsgan_tangent_losses(
                    values[0],
                    logits[0],
                    logits[1],
                    centering_scale=args.centering_scale,
                )
                (losses["total"] / args.accumulation_steps).backward()
                accumulated += torch.tensor(
                    [
                        losses["total"].detach(),
                        losses["real"].detach(),
                        losses["fake"].detach(),
                        losses["centering"].detach(),
                        logits[0].detach().mean(),
                        logits[1].detach().mean(),
                        (logits[0] - logits[1]).detach().mean(),
                    ],
                    device=device,
                    dtype=torch.float64,
                )
        grad_norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), 10_000.0)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite critic gradient at step {step}")
        optimizer.step()

        if step % args.log_every == 0 or step == 1:
            if world_size > 1:
                dist.all_reduce(accumulated, op=dist.ReduceOp.SUM)
            values = (
                accumulated / (args.accumulation_steps * world_size)
            ).cpu().tolist()
            elapsed = time.monotonic() - started
            row = {
                "step": step,
                "loss": values[0],
                "real_loss": values[1],
                "fake_loss": values[2],
                "centering": values[3],
                "real_logit": values[4],
                "fake_logit": values[5],
                "margin": values[6],
                "grad_norm": float(grad_norm),
                "elapsed_seconds": elapsed,
                "steps_per_second": (step - start_step) / max(elapsed, 1e-9),
                "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            }
            if is_main:
                append_csv(train_csv, row)
                print(json.dumps(row), flush=True)

        validation = None
        if step % args.validate_every == 0 or step == args.steps:
            validation = evaluate(
                critic=critic,
                strong=strong,
                loader=validation_loader,
                device=device,
                precision=args.precision,
                batches=args.validation_batches,
                seed=args.seed + 100_000 + rank * 100_003,
                centering_scale=args.centering_scale,
                class_dropout_probability=args.class_dropout_probability,
            )
            validation["step"] = step
            if is_main:
                atomic_json(validation, args.output_dir / f"validation_{step:06d}.json")
                print("validation " + json.dumps(validation), flush=True)
            critic.train()
            if is_main and validation["loss"] < best_loss:
                best_loss = float(validation["loss"])
                atomic_save(
                    {
                        "format": "eqvae_cafm_tangent_critic_v1",
                        "step": step,
                        "critic": critic.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_validation_loss": best_loss,
                        "config": config,
                        "model": model_metadata,
                        "validation": validation,
                    },
                    args.output_dir / "checkpoints" / "best.pt",
                )

        if is_main and (step % args.save_every == 0 or step == args.steps):
            atomic_save(
                {
                    "format": "eqvae_cafm_tangent_critic_v1",
                    "step": step,
                    "critic": critic.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_validation_loss": best_loss,
                    "config": config,
                    "model": model_metadata,
                    "validation": validation,
                },
                args.output_dir / "checkpoints" / f"step_{step:06d}.pt",
            )
        if world_size > 1 and (
            step % args.validate_every == 0
            or step % args.save_every == 0
            or step == args.steps
        ):
            dist.barrier()

    if is_main:
        atomic_json(
            {
                "status": "complete",
                "steps": args.steps,
                "best_validation_loss": best_loss,
                "elapsed_seconds": time.monotonic() - started,
                "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20,
                "world_size": world_size,
            },
            args.output_dir / "complete.json",
        )
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
