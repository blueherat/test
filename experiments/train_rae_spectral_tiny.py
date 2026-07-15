"""Strict paired microtraining for direction-only spectral weighting in RAE.

This tracked entry point intentionally reuses the official frozen RAE encoder,
DiTDH-S model, transport path, optimizer and scheduler while owning the small
amount of experiment-specific logic: fixed DCT weighting, branch checkpointing,
structured logs and deterministic paired random streams.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
RAE_ROOT = ROOT / "external" / "RAE"
RAE_SRC = RAE_ROOT / "src"
for path in (ROOT, RAE_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.rae_spectral_direction_loss import DCTDirectionLoss
from stage1 import RAE
from stage2.models import Stage2ModelProtocol
from stage2.transport import ModelType, create_transport
from utils.model_utils import instantiate_from_config
from utils.optim_utils import build_optimizer, build_scheduler
from utils.train_utils import ParquetImageNetDataset, center_crop_arr, update_ema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--global-seed", type=int, required=True)
    parser.add_argument("--spectral-gamma", type=float, required=True)
    parser.add_argument("--max-train-steps", type=int, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, torch.device]:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    return dist.get_rank(), dist.get_world_size(), device


def make_logger(experiment_dir: Path, rank: int) -> logging.Logger:
    logger = logging.getLogger(f"rae-spectral-rank-{rank}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if rank == 0:
        file_handler = logging.FileHandler(experiment_dir / "train.log", mode="a")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def tensor_fingerprint(value: torch.Tensor) -> str:
    array = value.detach().to(device="cpu").contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.stem.split("-")[-1])
    except ValueError:
        return -1


def latest_branch_checkpoint(checkpoint_dir: Path) -> Path | None:
    candidates = sorted(checkpoint_dir.glob("step-*.pt"), key=checkpoint_step)
    return candidates[-1] if candidates else None


def save_checkpoint(
    path: Path,
    *,
    model: DDP,
    ema: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    global_step: int,
    branch_start_step: int,
    epoch: int,
) -> None:
    state = {
        "step": int(global_step),
        "branch_start_step": int(branch_start_step),
        "epoch": int(epoch),
        "model": model.module.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(
    path: Path,
    *,
    model: DDP,
    ema: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    restore_rng: bool,
) -> tuple[int, int, int]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("model", "ema", "optimizer", "scheduler", "step"):
        if key not in state:
            raise KeyError(f"checkpoint {path} lacks {key!r}")
    model.module.load_state_dict(state["model"], strict=True)
    ema.load_state_dict(state["ema"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state["scheduler"] is not None:
        scheduler.load_state_dict(state["scheduler"])
    if restore_rng and "rng_cpu" in state and "rng_cuda" in state:
        torch.set_rng_state(state["rng_cpu"])
        torch.cuda.set_rng_state_all(state["rng_cuda"])
    step = int(state["step"])
    branch_start = int(state.get("branch_start_step", step))
    epoch = int(state.get("epoch", 0))
    return step, branch_start, epoch


def all_reduce_mean(values: torch.Tensor, world_size: int) -> torch.Tensor:
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return values / float(world_size)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    rank, world_size, device = setup_distributed()
    experiment_dir = args.results_dir.expanduser() / args.experiment_name
    checkpoint_dir = experiment_dir / "checkpoints"
    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    logger = make_logger(experiment_dir, rank)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=True)
    seed = int(args.global_seed) * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    config = OmegaConf.load(args.config)
    training = OmegaConf.to_container(config.training, resolve=True)
    misc = OmegaConf.to_container(config.misc, resolve=True)
    transport_params = OmegaConf.to_container(config.transport.params, resolve=True)
    spectral = dict(training["spectral_direction_loss"])
    global_batch = int(training["global_batch_size"])
    grad_accum = int(training["grad_accum_steps"])
    if global_batch % (world_size * grad_accum) != 0:
        raise ValueError("global batch must divide world_size * grad_accum_steps")
    micro_batch = global_batch // (world_size * grad_accum)

    rae: RAE = instantiate_from_config(config.stage_1).to(device=device, dtype=torch.float32)
    rae.requires_grad_(False).eval()
    model: Stage2ModelProtocol = instantiate_from_config(config.stage_2).to(
        device=device, dtype=torch.float32
    )
    ema = deepcopy(model).to(device=device, dtype=torch.float32)
    ema.requires_grad_(False).eval()
    model.requires_grad_(True).train()
    ddp_model = DDP(model, device_ids=[device.index], broadcast_buffers=False)

    transform = transforms.Compose(
        [
            transforms.Lambda(lambda image: center_crop_arr(image, int(args.image_size))),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    dataset = ParquetImageNetDataset(args.data_path, split="train", transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=int(args.global_seed),
        drop_last=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=micro_batch,
        sampler=sampler,
        num_workers=int(training["num_workers"]),
        pin_memory=True,
        drop_last=True,
        persistent_workers=int(training["num_workers"]) > 0,
    )
    steps_per_epoch = len(loader) // grad_accum
    if steps_per_epoch < 1:
        raise RuntimeError("no optimizer steps per epoch")

    optimizer, optimizer_message = build_optimizer(ddp_model.parameters(), training)
    scheduler, scheduler_message = build_scheduler(optimizer, steps_per_epoch, training)
    latent_size = tuple(int(value) for value in misc["latent_size"])
    time_shift = math.sqrt(float(misc["time_dist_shift_dim"]) / float(misc["time_dist_shift_base"]))
    transport = create_transport(**dict(transport_params), time_dist_shift=time_shift)
    if transport.model_type != ModelType.VELOCITY or str(transport_params["path_type"]) != "Linear":
        raise ValueError("this experiment requires linear velocity flow matching")
    loss_module = DCTDirectionLoss(
        spatial_size=int(spectral["spatial_size"]),
        second_moments=spectral["second_moments"],
        gamma=float(args.spectral_gamma),
        damping=float(spectral["damping"]),
        min_weight=float(spectral["min_weight"]),
        max_weight=float(spectral["max_weight"]),
    ).to(device)
    if latent_size[-2:] != (loss_module.spatial_size, loss_module.spatial_size):
        raise ValueError("latent size and DCT loss size disagree")

    local_checkpoint = latest_branch_checkpoint(checkpoint_dir)
    load_path = local_checkpoint or args.ckpt.expanduser()
    global_step, branch_start_step, start_epoch = load_checkpoint(
        load_path,
        model=ddp_model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        restore_rng=local_checkpoint is not None,
    )
    if local_checkpoint is None:
        branch_start_step = global_step
        start_epoch = 0
    if args.max_train_steps <= global_step:
        logger.info("Already at step %d >= endpoint %d; nothing to do.", global_step, args.max_train_steps)
        dist.destroy_process_group()
        return

    representative_times = torch.tensor([0.55, 0.70, 0.85, 0.95], device=device)
    weights = loss_module.weights(representative_times).detach().cpu()
    coefficient_means = (
        weights * loss_module.band_counts.cpu()[None]
    ).sum(1) / loss_module.band_counts.sum().cpu()
    if not torch.allclose(coefficient_means, torch.ones_like(coefficient_means), atol=2e-6, rtol=0):
        raise AssertionError("spectral weights are not coefficient-mean one")

    if rank == 0:
        actual_config = OmegaConf.to_container(config, resolve=True)
        actual_config["training"]["spectral_direction_loss"]["gamma"] = float(args.spectral_gamma)
        OmegaConf.save(OmegaConf.create(actual_config), experiment_dir / "config.yaml")
        shutil.copy2(Path(__file__), experiment_dir / Path(__file__).name)
        shutil.copy2(ROOT / "experiments/rae_spectral_direction_loss.py", experiment_dir)
        manifest = {
            "experiment_name": args.experiment_name,
            "source_checkpoint": str(args.ckpt.expanduser()),
            "loaded_checkpoint": str(load_path),
            "branch_start_step": branch_start_step,
            "endpoint_step": int(args.max_train_steps),
            "global_seed": int(args.global_seed),
            "world_size": world_size,
            "global_batch_size": global_batch,
            "micro_batch_size": micro_batch,
            "grad_accum_steps": grad_accum,
            "precision": "fp32",
            "tf32": False,
            "gamma": float(args.spectral_gamma),
            "weight_times": representative_times.cpu().tolist(),
            "weights": weights.tolist(),
            "coefficient_weight_means": coefficient_means.tolist(),
            "stats_source": spectral.get("stats_source"),
            "pairing_scope": "fresh deterministic stream from a shared full-state checkpoint",
        }
        (experiment_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        logger.info("Loaded %s at step %d; branch starts at %d.", load_path, global_step, branch_start_step)
        logger.info("%s | %s", optimizer_message, scheduler_message)
        logger.info(
            "gamma=%.2f, world=%d, micro=%d, accum=%d, global_batch=%d, endpoint=%d",
            args.spectral_gamma,
            world_size,
            micro_batch,
            grad_accum,
            global_batch,
            args.max_train_steps,
        )
        logger.info("weights=%s", weights.tolist())

    log_interval = int(training["log_interval"])
    clip_grad = float(training["clip_grad"])
    ema_decay = float(training["ema_decay"])
    save_offsets = {500, 1000, 2000, 5000}
    metrics_path = experiment_dir / "metrics.jsonl"
    window = torch.zeros(4 + 2 * loss_module.band_count, device=device, dtype=torch.float64)
    # [weighted loss, raw MSE, grad norm, clip hit, band MSE..., band weight...]
    window_microbatches = 0
    window_optimizer_steps = 0
    optimizer.zero_grad(set_to_none=True)
    training_start = perf_counter()

    epoch = start_epoch
    while global_step < args.max_train_steps:
        sampler.set_epoch(epoch)
        for images, labels in loader:
            if global_step >= args.max_train_steps:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.no_grad():
                latent = rae.encode(images)
            time, noise, clean = transport.sample(latent)
            time, noisy, target = transport.path_sampler.plan(time, noise, clean)
            prediction = ddp_model(noisy, time, y=labels)
            per_sample_loss, details = loss_module(prediction, target, time)
            loss = per_sample_loss.mean()
            if rank == 0 and global_step == branch_start_step and window_microbatches == 0:
                fingerprint = {
                    "step": global_step,
                    "images_sha256": tensor_fingerprint(images),
                    "labels_sha256": tensor_fingerprint(labels),
                    "time_sha256": tensor_fingerprint(time),
                    "noise_sha256": tensor_fingerprint(noise),
                    "noisy_latent_sha256": tensor_fingerprint(noisy),
                    "target_sha256": tensor_fingerprint(target),
                    "prediction_sha256": tensor_fingerprint(prediction),
                    "labels": labels.detach().cpu().tolist(),
                    "time": time.detach().cpu().tolist(),
                    "raw_mse": float(details["raw_mse"].mean().detach()),
                    "band_mse": details["band_mse"].mean(0).detach().cpu().tolist(),
                }
                (experiment_dir / "pair_fingerprint.json").write_text(
                    json.dumps(fingerprint, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
            (loss / grad_accum).backward()

            window[0] += float(loss.detach())
            window[1] += float(details["raw_mse"].mean().detach())
            window[4 : 4 + loss_module.band_count] += (
                details["band_mse"].mean(0).detach().double()
            )
            window[4 + loss_module.band_count :] += (
                details["band_weights"].mean(0).detach().double()
            )
            window_microbatches += 1

            if window_microbatches % grad_accum != 0:
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), clip_grad)
            optimizer.step()
            scheduler.step()
            update_ema(ema, ddp_model.module, decay=ema_decay)
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            window[2] += float(grad_norm)
            window[3] += float(grad_norm > clip_grad)
            window_optimizer_steps += 1

            if global_step % log_interval == 0:
                reduced = all_reduce_mean(window.clone(), world_size)
                if rank == 0:
                    micro_denominator = max(window_microbatches, 1)
                    step_denominator = max(window_optimizer_steps, 1)
                    row: dict[str, Any] = {
                        "step": global_step,
                        "branch_update": global_step - branch_start_step,
                        "weighted_loss": float(reduced[0] / micro_denominator),
                        "raw_mse": float(reduced[1] / micro_denominator),
                        "grad_norm": float(reduced[2] / step_denominator),
                        "clip_rate": float(reduced[3] / step_denominator),
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        "elapsed_seconds": perf_counter() - training_start,
                    }
                    for band in range(loss_module.band_count):
                        row[f"band_mse_{band}"] = float(
                            reduced[4 + band] / micro_denominator
                        )
                        row[f"band_weight_{band}"] = float(
                            reduced[4 + loss_module.band_count + band] / micro_denominator
                        )
                    append_jsonl(metrics_path, row)
                    logger.info(
                        "step=%d update=%d weighted=%.5f raw=%.5f grad=%.3f clip=%.2f lr=%.6g",
                        global_step,
                        global_step - branch_start_step,
                        row["weighted_loss"],
                        row["raw_mse"],
                        row["grad_norm"],
                        row["clip_rate"],
                        row["lr"],
                    )
                window.zero_()
                window_microbatches = 0
                window_optimizer_steps = 0

            branch_update = global_step - branch_start_step
            if branch_update in save_offsets or global_step == args.max_train_steps:
                dist.barrier()
                if rank == 0:
                    path = checkpoint_dir / f"step-{global_step:07d}.pt"
                    save_checkpoint(
                        path,
                        model=ddp_model,
                        ema=ema,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        global_step=global_step,
                        branch_start_step=branch_start_step,
                        epoch=epoch,
                    )
                    logger.info("Saved %s", path)
                dist.barrier()
        epoch += 1

    if window_microbatches > 0:
        reduced = all_reduce_mean(window.clone(), world_size)
        if rank == 0:
            row = {
                "step": global_step,
                "branch_update": global_step - branch_start_step,
                "weighted_loss": float(reduced[0] / window_microbatches),
                "raw_mse": float(reduced[1] / window_microbatches),
                "grad_norm": float(reduced[2] / max(window_optimizer_steps, 1)),
                "clip_rate": float(reduced[3] / max(window_optimizer_steps, 1)),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "elapsed_seconds": perf_counter() - training_start,
                "partial_window": True,
            }
            for band in range(loss_module.band_count):
                row[f"band_mse_{band}"] = float(reduced[4 + band] / window_microbatches)
                row[f"band_weight_{band}"] = float(
                    reduced[4 + loss_module.band_count + band] / window_microbatches
                )
            append_jsonl(metrics_path, row)
    if rank == 0:
        logger.info("Completed endpoint step %d in %.1f seconds.", global_step, perf_counter() - training_start)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
