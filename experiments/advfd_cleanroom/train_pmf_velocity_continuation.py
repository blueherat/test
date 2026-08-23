"""Train the released pMF-B checkpoint with only local velocity MSE."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler


EQVAE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OFFICIAL_ROOT = Path("/data/users/zhoushunyu/research_repos/AdvFD")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pure local-velocity continuation control for released pMF-B"
    )
    parser.add_argument("--official-root", type=Path, default=DEFAULT_OFFICIAL_ROOT)
    parser.add_argument("--packed-data", type=Path, required=True)
    parser.add_argument("--load-from", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--batch-size", type=int, default=18, help="per GPU")
    parser.add_argument("--total-steps", type=int, default=10_000)
    parser.add_argument("--warmup-steps", type=int, default=6_250)
    parser.add_argument("--save-steps", type=int, nargs="+", default=[5_000, 10_000])
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--print-freq", type=int, default=20)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--keep-n-checkpoints", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def setup_distributed(seed: int) -> tuple[int, int, int, torch.device]:
    if "RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device("cuda", local_rank),
        )
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = local_rank = 0
        world_size = 1
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    rank_seed = int(seed) + rank
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    torch.cuda.manual_seed(rank_seed)
    return rank, world_size, local_rank, device


def cosine_lr(step: int, *, lr: float, min_lr: float, warmup: int, total: int) -> float:
    if warmup > 0 and step < warmup:
        return lr * step / warmup
    if total <= warmup:
        return min_lr
    progress = min(max((step - warmup) / (total - warmup), 0.0), 1.0)
    return min_lr + (lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    value = tensor.detach().float().clone()
    if dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value.div_(dist.get_world_size())
    return value


def latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    paths = sorted(checkpoint_dir.glob("step_*.pth"))
    return paths[-1] if paths else None


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    ema_model,
    optimizer: torch.optim.Optimizer,
    step: int,
    samples_seen: int,
    elapsed_seconds: float,
    protocol: dict[str, object],
) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    checkpoint = {
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "model_ema": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        # Match the public trainer: ``step`` is the last completed zero-based
        # iteration, while ``current_step`` is the number of completed steps.
        "step": int(step) - 1,
        "current_step": int(step),
        "samples_seen": int(samples_seen),
        "last_elapsed_time": float(elapsed_seconds),
        "velocity_control_protocol": protocol,
    }
    torch.save(checkpoint, tmp)
    os.replace(tmp, path)


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank, device = setup_distributed(args.seed)
    is_main = rank == 0

    official_root = args.official_root.expanduser().resolve()
    packed_root = args.packed_data.expanduser().resolve()
    load_from = args.load_from.expanduser().resolve()
    run_dir = args.output_dir.expanduser().resolve() / args.exp_name
    checkpoint_dir = run_dir / "checkpoints"
    if is_main:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    sys.path.insert(0, str(EQVAE_ROOT))
    sys.path.insert(0, str(official_root))
    from experiments.advfd_cleanroom.pmf_velocity_control import (  # noqa: PLC0415
        PMFVelocityObjective,
        VelocityControlProtocol,
    )
    from experiments.raev2_training_core import (  # noqa: PLC0415
        DeterministicImageNetPacked,
    )
    from models.denoiser_pmf import (  # noqa: PLC0415
        convert_pmf_checkpoint,
        pMFDenoiser_models,
    )
    from utils.ema_util import EMAModel  # noqa: PLC0415

    dataset = DeterministicImageNetPacked(
        packed_root,
        split="train",
        image_size=256,
        augmentation_seed=args.seed,
        horizontal_flip=False,
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=0,
        drop_last=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    loader_iter = iter(loader)
    loader_epoch = 0

    model = pMFDenoiser_models["pMF_B"](
        img_size=256,
        patch_size=16,
        in_channels=3,
        tokenizer_patch_size=1,
        num_classes=1000,
        label_drop_prob=0.1,
        P_mean=0.8,
        P_std=0.8,
        ratio_r_neq_t=0.5,
        cfg_beta=1.0,
        cfg_omega_max=7.0,
        aux_head_depth=8,
        class_tokens=8,
        time_tokens=4,
        guidance_tokens=4,
        interval_tokens=2,
        rope_2d=True,
        learned_pe=True,
        disable_v_head=True,
        grad_checkpointing=args.grad_checkpointing,
        t_eps=0.05,
        noise_scale=1.0,
    ).to(device)

    loaded = torch.load(load_from, map_location="cpu", weights_only=False)
    state = loaded["model"] if isinstance(loaded, dict) and "model" in loaded else loaded
    incompatible = model.load_state_dict(convert_pmf_checkpoint(state), strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Released pMF checkpoint did not load exactly: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    del loaded, state
    if dist.is_initialized():
        for parameter in model.parameters():
            dist.broadcast(parameter.data, src=0)

    objective = PMFVelocityObjective(model)
    ddp = DistributedDataParallel(
        objective,
        device_ids=[local_rank],
        broadcast_buffers=False,
        bucket_cap_mb=25,
        static_graph=True,
    ) if world_size > 1 else objective

    named = list(model.named_parameters())
    no_decay = lambda name, parameter: (
        parameter.ndim < 2
        or any(token in name for token in (
            "ln", "bias", "embedding", "norm", "gamma", "embed", "token", "diffloss"
        ))
    )
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for n, p in named if no_decay(n, p) and p.requires_grad],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in named if not no_decay(n, p) and p.requires_grad],
                "weight_decay": args.weight_decay,
            },
        ],
        lr=args.lr,
        betas=(args.beta1, args.beta2),
    )
    ema_model = EMAModel(
        model,
        ema_type="edm",
        values=[250, 500, 1000, 2000],
        batch_size=args.batch_size * world_size,
    )

    protocol = VelocityControlProtocol().to_dict()
    start_step = 0
    samples_seen = 0
    elapsed_before = 0.0
    resume_path = args.resume_from
    if resume_path is None and args.auto_resume:
        resume_path = latest_checkpoint(checkpoint_dir)
    if resume_path is not None:
        resume = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(resume["model"], strict=True)
        ema_model.load_state_dict(resume["model_ema"])
        ema_model.to(device)
        optimizer.load_state_dict(resume["optimizer"])
        start_step = int(resume.get("current_step", resume["step"]))
        samples_seen = int(resume.get("samples_seen", start_step * args.batch_size * world_size))
        elapsed_before = float(resume.get("last_elapsed_time", 0.0))
        del resume

    manifest = {
        "protocol_name": "pmf_b_local_velocity_mse_continuation_v1",
        "protocol": protocol,
        "scientific_scope": (
            "Local h=0 flow-matching continuation control; not the original full "
            "pMF mean-flow objective because the released checkpoint has no v head."
        ),
        "official_root": str(official_root),
        "official_commit": git_head(official_root),
        "eqvae_root": str(EQVAE_ROOT),
        "eqvae_commit_at_launch": git_head(EQVAE_ROOT),
        "trainer_sha256": sha256_file(Path(__file__).resolve()),
        "load_from": str(load_from),
        "load_from_sha256": sha256_file(load_from),
        "packed_data": str(packed_root),
        "horizontal_flip": False,
        "crop": "ADM center crop",
        "world_size": world_size,
        "local_batch_size": args.batch_size,
        "global_batch_size": args.batch_size * world_size,
        "optimizer": "AdamW",
        "lr": args.lr,
        "min_lr": args.min_lr,
        "warmup_steps": args.warmup_steps,
        "total_steps": args.total_steps,
        "save_steps": sorted(set(args.save_steps)),
        "dtype": args.dtype,
        "grad_checkpointing": args.grad_checkpointing,
        "seed": args.seed,
        "resume_from": str(resume_path) if resume_path is not None else None,
        "start_step": start_step,
        "smoke": args.smoke,
    }
    metrics_path = run_dir / "training_metrics.jsonl"
    if is_main:
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)

    amp_context = (
        lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.dtype == "bf16"
        else contextlib.nullcontext
    )
    save_steps = set(args.save_steps)
    if args.total_steps not in save_steps:
        save_steps.add(args.total_steps)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    session_start = time.time()
    last_time = time.perf_counter()
    for step in range(start_step, args.total_steps):
        try:
            images, labels, _ = next(loader_iter)
        except StopIteration:
            loader_epoch += 1
            sampler.set_epoch(loader_epoch)
            loader_iter = iter(loader)
            images, labels, _ = next(loader_iter)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        lr = cosine_lr(
            step,
            lr=args.lr,
            min_lr=args.min_lr,
            warmup=args.warmup_steps,
            total=args.total_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        with amp_context():
            loss, metrics = ddp(images, labels)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_clip if args.grad_clip > 0 else float("inf")
        )
        finite = torch.isfinite(grad_norm)
        if dist.is_initialized():
            finite_int = finite.to(torch.int32)
            dist.all_reduce(finite_int, op=dist.ReduceOp.MIN)
            finite = finite_int.bool()
        if bool(finite.item()):
            optimizer.step()
            ema_model.step(model)
        optimizer.zero_grad(set_to_none=True)

        current_step = step + 1
        samples_seen += args.batch_size * world_size
        now = time.perf_counter()
        step_time = now - last_time
        last_time = now
        reduced = {key: reduce_mean(value) for key, value in metrics.items()}
        reduced_grad = reduce_mean(grad_norm)
        if is_main and (step == start_step or current_step % args.print_freq == 0):
            row = {
                "step": current_step,
                "loss": float(reduced["velocity_mse"]),
                **{key: float(value) for key, value in reduced.items()},
                "grad_norm": float(reduced_grad),
                "lr": lr,
                "step_time_seconds": step_time,
                "samples_per_second": args.batch_size * world_size / max(step_time, 1e-12),
                "samples_seen": samples_seen,
                "peak_reserved_gib_per_gpu": torch.cuda.max_memory_reserved(device) / 2**30,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps(row, sort_keys=True), flush=True)

        if current_step in save_steps:
            if dist.is_initialized():
                dist.barrier()
            if is_main:
                elapsed = elapsed_before + time.time() - session_start
                save_checkpoint(
                    checkpoint_dir / f"step_{current_step:07d}.pth",
                    model=model,
                    ema_model=ema_model,
                    optimizer=optimizer,
                    step=current_step,
                    samples_seen=samples_seen,
                    elapsed_seconds=elapsed,
                    protocol=protocol,
                )
                checkpoints = sorted(checkpoint_dir.glob("step_*.pth"))
                for stale in checkpoints[:-args.keep_n_checkpoints]:
                    stale.unlink()
                print(f"saved checkpoint at step {current_step}", flush=True)
            if dist.is_initialized():
                dist.barrier()

    if is_main:
        elapsed = elapsed_before + time.time() - session_start
        (run_dir / "complete.json").write_text(
            json.dumps(
                {
                    "complete": True,
                    "total_steps": args.total_steps,
                    "samples_seen": samples_seen,
                    "elapsed_seconds": elapsed,
                },
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
