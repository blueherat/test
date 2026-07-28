"""Strict paired continuation for the official RAEv2 ImageNet checkpoint.

The source model predicts clean latents (``transport.prediction == "x"``).
Flow and LPL branches restore the same model, EMA, GMuon, and scheduler states,
then consume an index-deterministic ImageNet stream.  The unavailable original
dataloader cursor is never claimed to be restored.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import logging
import math
import os
import random
import sys
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler


ROOT = Path(__file__).resolve().parents[1]
RAEV2_ROOT = ROOT / "external" / "RAEv2"
RAEV2_SRC = RAEV2_ROOT / "src"
for path in (RAEV2_SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.rae_strict_lpl import (  # noqa: E402
    decoder_feature_pyramid,
    decoder_hidden_indices,
    lpl_time_gate,
    strict_lpl_per_sample,
)
from experiments.raev2_training_core import (  # noqa: E402
    DeterministicImageNetParquet,
    append_jsonl,
    branch_epoch,
    file_sha256,
    infer_source_steps_per_epoch,
    official_flow_loss_map,
    predicted_clean_latent,
    synchronize_loaded_gmuon_param_groups,
    tensor_fingerprint,
    validate_full_stage2_checkpoint,
)
from configs.stage2 import Stage2Config  # noqa: E402
from stage2.transport import create_transport  # noqa: E402
from stage2.utils import apply_cfg_dropout, validate_stage2_config  # noqa: E402
from utils.model_utils import instantiate_from_config  # noqa: E402
from utils.optim_utils import build_optimizer, build_scheduler  # noqa: E402
from utils.train_utils import update_ema  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--objective", choices=("flow", "lpl"), required=True)
    parser.add_argument("--max-updates", type=int, required=True)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument(
        "--ema-device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="CPU keeps the official fp32 EMA update while preserving shared GPU headroom.",
    )
    parser.add_argument("--source-steps-per-epoch", type=int)
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--lpl-weight", type=float, default=1.0)
    parser.add_argument("--lpl-noise-threshold", type=float, default=3.0)
    parser.add_argument("--lpl-max-samples-per-rank", type=int, default=1)
    parser.add_argument("--calibration-batches", type=int, default=0)
    parser.add_argument("--calibration-target-ratio", type=float, default=0.20)
    parser.add_argument("--skip-checkpoint-save", action="store_true")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--min-free-gib", type=float, default=2.5)
    parser.add_argument(
        "--dino-ckpt-dir",
        type=Path,
        default=Path("/home/zhoushunyu/data/eqvae/models/RAEv2/encoders/dinov3"),
    )
    parser.add_argument("--dino-repo-dir", type=Path)
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, torch.device]:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    return dist.get_rank(), dist.get_world_size(), torch.device("cuda", local_rank)


def make_logger(directory: Path, rank: int) -> logging.Logger:
    logger = logging.getLogger(f"raev2-strict-lpl-{rank}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if rank == 0:
        file_handler = logging.FileHandler(directory / "train.log", mode="a")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


class OffsetSampler(Sampler[int]):
    """Skip already-consumed microbatches without decoding their images."""

    def __init__(self, sampler: Sampler[int], offset_samples: int) -> None:
        self.sampler = sampler
        self.offset_samples = int(offset_samples)
        if self.offset_samples < 0:
            raise ValueError("offset_samples must be non-negative")

    def __iter__(self) -> Iterator[int]:
        return itertools.islice(iter(self.sampler), self.offset_samples, None)

    def __len__(self) -> int:
        return max(len(self.sampler) - self.offset_samples, 0)


def load_config(path: Path) -> Stage2Config:
    config = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(Stage2Config), OmegaConf.load(path))
    )
    config.post_process()
    validate_stage2_config(config)
    if config.transport.prediction != "x":
        raise ValueError(
            "the official dinov3l-k7 RAEv2 checkpoint is x-prediction; "
            f"got {config.transport.prediction!r}"
        )
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def autocast_context(precision: str):
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def optimizer_device_audit(
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
) -> dict[str, int]:
    model_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter.numel() > 0
    }
    if optimizer_ids != model_ids:
        raise RuntimeError(
            "optimizer boundary mismatch: "
            f"missing={len(model_ids - optimizer_ids)}, extra={len(optimizer_ids - model_ids)}"
        )
    return {
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "optimizer_parameter_tensors": len(optimizer_ids),
    }


def collect_rank_rng(rank: int, world_size: int) -> list[dict[str, Any]] | None:
    payload = {
        "rank": rank,
        "torch_cpu": torch.get_rng_state().cpu(),
        "torch_cuda": torch.cuda.get_rng_state().cpu(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    gathered = [None] * world_size if rank == 0 else None
    dist.gather_object(payload, gathered, dst=0)
    return gathered


def restore_rank_rng(checkpoint: dict[str, Any], rank: int) -> None:
    metadata = checkpoint.get("raev2_lpl", {})
    states = metadata.get("rank_rng_states")
    if states is None:
        raise ValueError("branch checkpoint has no per-rank RNG state")
    matches = [state for state in states if int(state["rank"]) == rank]
    if len(matches) != 1:
        raise ValueError(f"expected one RNG state for rank {rank}, found {len(matches)}")
    state = matches[0]
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["torch_cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def free_memory_gib(device: torch.device) -> float:
    free_bytes, _ = torch.cuda.mem_get_info(device)
    return free_bytes / (1024**3)


def require_memory_reserve(device: torch.device, minimum_gib: float, phase: str) -> None:
    free_gib = free_memory_gib(device)
    if free_gib < minimum_gib:
        raise RuntimeError(
            f"GPU memory reserve below limit after {phase}: "
            f"{free_gib:.2f} GiB free < {minimum_gib:.2f} GiB"
        )


@torch.no_grad()
def update_ema_cpu(
    ema_model: torch.nn.Module,
    model: torch.nn.Module,
    *,
    decay: float,
) -> None:
    """Apply the official parameter EMA formula with rank-0 CPU storage."""

    ema_parameters = dict(ema_model.named_parameters())
    model_parameters = dict(model.named_parameters())
    if ema_parameters.keys() != model_parameters.keys():
        raise RuntimeError("EMA and model parameter names differ")
    for name, parameter in model_parameters.items():
        target = ema_parameters[name]
        target.mul_(float(decay)).add_(
            parameter.detach().to(device="cpu", dtype=target.dtype),
            alpha=1.0 - float(decay),
        )


def save_checkpoint(
    path: Path,
    *,
    ddp_model: DDP,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    source_step: int,
    source_epoch: int,
    source_sha256: str,
    source_steps_per_epoch: int,
    branch_update_value: int,
    objective: str,
    config_sha256: str,
    data_indices_sha256: str,
    rank_rng_states: list[dict[str, Any]],
) -> None:
    checkpoint = {
        "step": int(source_step + branch_update_value),
        "epoch": branch_epoch(
            source_epoch, branch_update_value, source_steps_per_epoch
        ),
        "model": ddp_model.module.state_dict(),
        "ema": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "raev2_lpl": {
            "format_version": 1,
            "source_step": int(source_step),
            "source_epoch": int(source_epoch),
            "source_sha256": source_sha256,
            "source_steps_per_epoch": int(source_steps_per_epoch),
            "branch_update": int(branch_update_value),
            "objective": objective,
            "config_sha256": config_sha256,
            "data_indices_sha256": data_indices_sha256,
            "rank_rng_states": rank_rng_states,
            "original_dataloader_cursor_restored": False,
            "paired_branch_stream_is_exact": True,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def main() -> None:
    args = parse_args()
    if args.max_updates <= 0:
        raise ValueError("--max-updates must be positive")
    if args.save_every <= 0:
        raise ValueError("--save-every must be positive")
    if args.calibration_batches < 0:
        raise ValueError("--calibration-batches must be non-negative")
    if args.objective == "flow" and args.calibration_batches:
        raise ValueError("LPL calibration requires --objective lpl")

    os.environ["DINOV3_CKPT_DIR"] = str(args.dino_ckpt_dir.expanduser().resolve())
    if args.dino_repo_dir is not None:
        os.environ["DINOV3_REPO_DIR"] = str(args.dino_repo_dir.expanduser().resolve())

    rank, world_size, device = setup_distributed()
    experiment_dir = args.results_dir.expanduser() / args.experiment_name
    if rank == 0:
        experiment_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    logger = make_logger(experiment_dir, rank)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    config = load_config(args.config)
    global_batch_size = int(config.training.global_batch_size)
    grad_accum_steps = int(config.training.grad_accum_steps)
    if global_batch_size != 1024:
        raise ValueError(
            "strict RAEv2 ImageNet continuation requires the official global "
            f"batch size 1024; got {global_batch_size}"
        )
    if global_batch_size % (world_size * grad_accum_steps):
        raise ValueError("global batch must be divisible by world_size * grad_accum_steps")
    micro_batch_size = global_batch_size // (world_size * grad_accum_steps)
    if micro_batch_size < 1:
        raise ValueError("micro batch size must be positive")
    if micro_batch_size != 1 and rank == 0:
        logger.warning("Pilot was designed for micro_batch_size=1; got %d", micro_batch_size)

    rank_seed = int(args.global_seed) * world_size + rank
    set_seed(rank_seed)
    config_sha256 = file_sha256(args.config)
    source_sha256 = file_sha256(args.source_checkpoint) if rank == 0 else ""
    broadcast_hash = [source_sha256]
    dist.broadcast_object_list(broadcast_hash, src=0)
    source_sha256 = broadcast_hash[0]

    load_path = args.resume if args.resume is not None else args.source_checkpoint
    checkpoint = torch.load(
        load_path.expanduser(), map_location="cpu", weights_only=False, mmap=True
    )
    validate_full_stage2_checkpoint(checkpoint)
    if args.resume is not None:
        metadata = checkpoint.get("raev2_lpl")
        if not metadata:
            raise ValueError("--resume must point to a branch checkpoint")
        if metadata["source_sha256"] != source_sha256:
            raise ValueError("resume checkpoint and source checkpoint hashes disagree")
        if metadata["objective"] != args.objective:
            raise ValueError("cannot resume a branch with a different objective")
        branch_update_value = int(metadata["branch_update"])
        source_step = int(metadata["source_step"])
        source_epoch = int(metadata["source_epoch"])
        source_steps_per_epoch = int(metadata["source_steps_per_epoch"])
    else:
        branch_update_value = 0
        source_step = int(checkpoint["step"])
        source_epoch = int(checkpoint["epoch"])
        source_steps_per_epoch = (
            int(args.source_steps_per_epoch)
            if args.source_steps_per_epoch is not None
            else infer_source_steps_per_epoch(source_step, source_epoch)
        )
    if branch_update_value >= args.max_updates:
        raise ValueError("resume checkpoint is already at or beyond --max-updates")

    if rank == 0:
        logger.info(
            "Source step=%d epoch=%d source_steps_per_epoch=%d branch_update=%d",
            source_step,
            source_epoch,
            source_steps_per_epoch,
            branch_update_value,
        )
        logger.info(
            "Continuation batch: global=%d micro=%d accumulation=%d world=%d "
            "(official global batch preserved)",
            global_batch_size,
            micro_batch_size,
            grad_accum_steps,
            world_size,
        )

    dataset = DeterministicImageNetParquet(
        args.data_path,
        split="train",
        image_size=int(config.training.image_size),
        augmentation_seed=int(args.global_seed),
        horizontal_flip=True,
    )
    distributed_sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=int(args.global_seed),
        drop_last=True,
    )
    distributed_sampler.set_epoch(source_epoch)
    consumed_samples_per_rank = branch_update_value * grad_accum_steps * micro_batch_size
    sampler = OffsetSampler(distributed_sampler, consumed_samples_per_rank)
    loader_generator = torch.Generator().manual_seed(rank_seed + 100_000)
    num_workers = (
        int(args.num_workers)
        if args.num_workers is not None
        else int(config.training.num_workers)
    )
    loader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        multiprocessing_context="spawn" if num_workers > 0 else None,
        generator=loader_generator,
    )

    # Stage 1 is frozen.  Flow does not need its decoder after construction.
    rae = instantiate_from_config(config.stage_1).to(device)
    rae.eval()
    rae.requires_grad_(False)
    if args.objective == "flow":
        del rae.decoder
        torch.cuda.empty_cache()

    config.prepare_model_params()
    model = instantiate_from_config(config.stage_2).to(device)
    model.requires_grad_(True)
    if args.ema_device == "cuda":
        ema_model = deepcopy(model).to(device)
    elif rank == 0:
        ema_model = instantiate_from_config(config.stage_2).cpu()
    else:
        ema_model = None
    if ema_model is not None:
        ema_model.requires_grad_(False)
        ema_model.eval()
    ddp_model = DDP(
        model, device_ids=[device.index], broadcast_buffers=False, find_unused_parameters=False
    )
    model = ddp_model.module

    optimizer, optimizer_message = build_optimizer(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        config.training.optimizer,
    )
    scheduler, scheduler_message = build_scheduler(
        optimizer, source_steps_per_epoch, config.training.scheduler
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    if ema_model is not None:
        ema_model.load_state_dict(checkpoint["ema"], strict=True)
    loaded_optimizer_state = checkpoint["optimizer"]
    optimizer.load_state_dict(loaded_optimizer_state)
    optimizer_restore_audit = synchronize_loaded_gmuon_param_groups(
        optimizer, loaded_optimizer_state
    )
    if checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    optimizer_audit = optimizer_device_audit(optimizer, model)

    if args.resume is not None:
        restore_rank_rng(checkpoint, rank)
    else:
        # Source checkpoints do not carry RNG state.  Re-seed after all model
        # construction so independently launched branches start identically.
        set_seed(rank_seed + 1_000_000)
    del checkpoint

    latent_size = tuple(config.misc.latent_size)
    time_dist_shift = math.sqrt(
        (config.misc.time_dist_shift_dim or math.prod(latent_size))
        / config.misc.time_dist_shift_base
    )
    transport = create_transport(config=config.transport, time_dist_shift=time_dist_shift)
    null_context = torch.full(
        (micro_batch_size,),
        int(config.misc.num_classes),
        dtype=torch.long,
        device=device,
    )
    layer_indices = None
    if args.objective == "lpl":
        layer_indices = decoder_hidden_indices(len(rae.decoder.decoder_layers))

    require_memory_reserve(device, args.min_free_gib, "model and optimizer load")
    if rank == 0:
        manifest = {
            "format_version": 1,
            "config": str(args.config.resolve()),
            "config_sha256": config_sha256,
            "source_checkpoint": str(args.source_checkpoint.expanduser().resolve()),
            "source_sha256": source_sha256,
            "source_step": source_step,
            "source_epoch": source_epoch,
            "source_steps_per_epoch": source_steps_per_epoch,
            "objective": args.objective,
            "max_updates": args.max_updates,
            "save_every": args.save_every,
            "precision": args.precision,
            "ema_device": args.ema_device,
            "ema_update_rank": 0 if args.ema_device == "cpu" else "all",
            "world_size": world_size,
            "global_batch_size": global_batch_size,
            "micro_batch_size": micro_batch_size,
            "grad_accum_steps": grad_accum_steps,
            "dataset": str(args.data_path.expanduser().resolve()),
            "dataset_split": dataset.split,
            "dataset_shards": len(dataset.files),
            "dataset_samples": len(dataset),
            "deterministic_index_flip": True,
            "original_dataloader_cursor_restored": False,
            "paired_branch_stream_is_exact": True,
            "optimizer": optimizer_message,
            "scheduler": scheduler_message,
            "scheduler_last_epoch": scheduler.last_epoch,
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
            "optimizer_audit": optimizer_audit,
            "optimizer_restore_audit": optimizer_restore_audit,
            "lpl_weight": args.lpl_weight if args.objective == "lpl" else 0.0,
            "lpl_noise_threshold": args.lpl_noise_threshold,
            "lpl_layer_indices": layer_indices,
        }
        (experiment_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Optimizer: %s", optimizer_message)
        logger.info("Scheduler: %s; restored last_epoch=%d", scheduler_message, scheduler.last_epoch)
        logger.info("Initial learning rates: %s", [group["lr"] for group in optimizer.param_groups])

    ddp_model.train()
    optimizer.zero_grad(set_to_none=True)
    data_index_digest = __import__("hashlib").sha256()
    metrics_path = experiment_dir / "train_metrics.jsonl"
    first_batch_written = False
    calibration_flow_sum = torch.zeros((), device=device)
    calibration_lpl_sum = torch.zeros((), device=device)
    calibration_lpl_count = torch.zeros((), device=device)
    calibration_flow_count = torch.zeros((), device=device)
    micro_since_boundary = 0
    last_time = perf_counter()

    for images, labels, indices in loader:
        if branch_update_value >= args.max_updates:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        indices = indices.to(device)
        for value in indices.detach().cpu().tolist():
            data_index_digest.update(int(value).to_bytes(8, "little", signed=False))

        # Match the official trainer: Stage 1 encoding stays in fp32 even when
        # the Stage 2 forward uses bf16 autocast.
        with torch.no_grad():
            clean_latent = rae.encode(images)
        model_kwargs = {"context": labels, "attn_mask": None}
        null_kwargs = {"context": null_context, "attn_mask": None}

        if args.calibration_batches:
            is_boundary = False
            sync_context = nullcontext()
        else:
            micro_since_boundary += 1
            is_boundary = micro_since_boundary == grad_accum_steps
            sync_context = nullcontext() if is_boundary else ddp_model.no_sync()

        # DDP requires no_sync() to cover both forward and backward.  Wrapping
        # backward alone still synchronizes every accumulation microbatch.
        with sync_context:
            with autocast_context(args.precision):
                dropped_kwargs, cfg_mask = apply_cfg_dropout(
                    model_kwargs,
                    null_kwargs,
                    float(config.conditioning.cfg_dropout_prob),
                )
                time, noise, clean_latent = transport.sample(clean_latent)
                scale = time.reshape(
                    (time.shape[0],) + (1,) * (clean_latent.ndim - 1)
                )
                noisy_latent = (1.0 - scale) * clean_latent + scale * noise
                target_velocity = (noisy_latent - clean_latent) / scale.clamp_min(
                    float(config.transport.t_eps)
                )
                model_output = ddp_model(noisy_latent, time, **dropped_kwargs)
                flow_map, _ = official_flow_loss_map(
                    transport,
                    model_output,
                    target_velocity=target_velocity,
                    noisy_latent=noisy_latent,
                    time=time,
                    base_model_coeff=float(config.internal_guidance.base_model_coeff),
                )
                flow_loss = flow_map.mean()
                total_loss = flow_loss
                lpl_loss = torch.zeros((), device=device)
                gate_count = 0

                if args.objective == "lpl":
                    primary_output = (
                        model_output[0] if isinstance(model_output, tuple) else model_output
                    )
                    clean_prediction = predicted_clean_latent(
                        primary_output,
                        prediction=config.transport.prediction,
                        noisy_latent=noisy_latent,
                        time=time,
                    )
                    gate = lpl_time_gate(time, float(args.lpl_noise_threshold))
                    selected = torch.nonzero(gate, as_tuple=False).flatten()
                    selected = selected[: int(args.lpl_max_samples_per_rank)]
                    gate_count = int(selected.numel())
                    if gate_count:
                        with torch.no_grad():
                            target_features = tuple(
                                feature.float()
                                for feature in decoder_feature_pyramid(
                                    rae,
                                    clean_latent.index_select(0, selected),
                                    layer_indices=layer_indices,
                                )
                            )
                        predicted_features = tuple(
                            feature.float()
                            for feature in decoder_feature_pyramid(
                                rae,
                                clean_prediction.index_select(0, selected),
                                layer_indices=layer_indices,
                            )
                        )
                        lpl_per_sample, _ = strict_lpl_per_sample(
                            target_features, predicted_features
                        )
                        lpl_loss = lpl_per_sample.mean()
                        total_loss = total_loss + float(args.lpl_weight) * lpl_loss

            if not args.calibration_batches:
                (total_loss / grad_accum_steps).backward()

        if not first_batch_written:
            row = {
                "rank": rank,
                "indices": indices.detach().cpu().tolist(),
                "image_sha256": tensor_fingerprint(images),
                "label_sha256": tensor_fingerprint(labels),
                "latent_sha256": tensor_fingerprint(clean_latent),
                "noise_sha256": tensor_fingerprint(noise),
                "time_sha256": tensor_fingerprint(time),
                "cfg_mask_sha256": tensor_fingerprint(cfg_mask),
            }
            gathered = [None] * world_size if rank == 0 else None
            dist.gather_object(row, gathered, dst=0)
            if rank == 0:
                (experiment_dir / "first_batch_audit.json").write_text(
                    json.dumps(gathered, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            first_batch_written = True

        if args.calibration_batches:
            calibration_flow_sum += flow_loss.detach()
            calibration_flow_count += 1
            if gate_count:
                calibration_lpl_sum += lpl_loss.detach()
                calibration_lpl_count += 1
            if int(calibration_flow_count.item()) >= args.calibration_batches:
                break
            continue

        if not is_boundary:
            continue

        micro_since_boundary = 0
        if config.training.clip_grad is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                ddp_model.parameters(), float(config.training.clip_grad)
            )
        else:
            grad_norm = torch.tensor(float("nan"), device=device)
        if branch_update_value == 0:
            # Release construction/forward fragments before GMuon creates its
            # persistent momentum buffers on the first optimizer step.
            torch.cuda.empty_cache()
        require_memory_reserve(device, args.min_free_gib, "backward")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        if ema_model is not None:
            if args.ema_device == "cpu":
                update_ema_cpu(
                    ema_model, model, decay=float(config.training.ema_decay)
                )
            else:
                update_ema(
                    ema_model, model, decay=float(config.training.ema_decay)
                )
        require_memory_reserve(device, args.min_free_gib, "optimizer and EMA update")
        branch_update_value += 1

        if branch_update_value % int(config.training.log_interval) == 0:
            reduced = torch.tensor(
                [
                    float(flow_loss.detach()),
                    float(lpl_loss.detach()),
                    float(total_loss.detach()),
                    float(grad_norm.detach()),
                    float(gate_count),
                ],
                device=device,
            )
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
            reduced /= world_size
            if rank == 0:
                now = perf_counter()
                row = {
                    "branch_update": branch_update_value,
                    "global_step": source_step + branch_update_value,
                    "flow_loss": float(reduced[0]),
                    "lpl_loss": float(reduced[1]),
                    "total_loss": float(reduced[2]),
                    "grad_norm": float(reduced[3]),
                    "mean_lpl_gate_count": float(reduced[4]),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "seconds_per_log_interval": now - last_time,
                    "free_gpu_gib_rank0": free_memory_gib(device),
                }
                append_jsonl(metrics_path, row)
                logger.info(
                    "update=%d global=%d flow=%.6f lpl=%.6f total=%.6f "
                    "grad=%.4f lr=%.3e free=%.2fGiB",
                    branch_update_value,
                    source_step + branch_update_value,
                    row["flow_loss"],
                    row["lpl_loss"],
                    row["total_loss"],
                    row["grad_norm"],
                    row["lr"],
                    row["free_gpu_gib_rank0"],
                )
                last_time = now

        should_save = (
            branch_update_value % args.save_every == 0
            or branch_update_value == args.max_updates
        )
        if should_save and not args.skip_checkpoint_save:
            rank_rng_states = collect_rank_rng(rank, world_size)
            if rank == 0:
                if ema_model is None:
                    raise RuntimeError("rank 0 must own the EMA model")
                checkpoint_path = (
                    experiment_dir
                    / "checkpoints"
                    / f"branch-{branch_update_value:07d}-global-{source_step + branch_update_value:07d}.pt"
                )
                save_checkpoint(
                    checkpoint_path,
                    ddp_model=ddp_model,
                    ema_model=ema_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    source_step=source_step,
                    source_epoch=source_epoch,
                    source_sha256=source_sha256,
                    source_steps_per_epoch=source_steps_per_epoch,
                    branch_update_value=branch_update_value,
                    objective=args.objective,
                    config_sha256=config_sha256,
                    data_indices_sha256=data_index_digest.hexdigest(),
                    rank_rng_states=rank_rng_states,
                )
                logger.info("Saved %s", checkpoint_path)
            dist.barrier()

    if args.calibration_batches:
        totals = torch.stack(
            [
                calibration_flow_sum,
                calibration_flow_count,
                calibration_lpl_sum,
                calibration_lpl_count,
            ]
        )
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        if rank == 0:
            flow_mean = float(totals[0] / totals[1].clamp_min(1))
            lpl_mean = float(totals[2] / totals[3].clamp_min(1))
            recommended = (
                float(args.calibration_target_ratio) * flow_mean / lpl_mean
                if lpl_mean > 0
                else float("nan")
            )
            result = {
                "flow_mean": flow_mean,
                "lpl_mean_on_gated_batches": lpl_mean,
                "flow_batches": int(totals[1]),
                "gated_batches": int(totals[3]),
                "target_lpl_over_flow": float(args.calibration_target_ratio),
                "recommended_lpl_weight": recommended,
            }
            (experiment_dir / "lpl_calibration.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            logger.info("Calibration: %s", result)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
