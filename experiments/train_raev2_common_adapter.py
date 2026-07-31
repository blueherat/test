"""Train a small common residual adapter on a frozen RAEv2 Stage-2 model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


ROOT = Path(__file__).resolve().parents[1]
RAEV2_SRC = ROOT / "external" / "RAEv2" / "src"
for path in (RAEV2_SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from configs.stage2 import Stage2Config  # noqa: E402
from experiments.rae_lpl_detach_audit import (  # noqa: E402
    decoder_feature_objective_per_sample,
)
from experiments.rae_strict_lpl import (  # noqa: E402
    decoder_feature_pyramid,
    decoder_hidden_indices,
    lpl_time_gate,
)
from experiments.raev2_common_adapter import (  # noqa: E402
    COMMON_ADAPTER_FORMAT,
    CommonResidualAdapter,
    internal_guidance_prediction,
)
from experiments.raev2_stage1_compat import (  # noqa: E402
    install_raev2_decoder_config_compat,
)
from experiments.raev2_training_core import (  # noqa: E402
    DeterministicImageNetPacked,
    DeterministicImageNetParquet,
    append_jsonl,
    file_sha256,
    official_flow_loss_map,
    tensor_fingerprint,
    validate_full_stage2_checkpoint,
)
from experiments.train_raev2_strict_lpl import (  # noqa: E402
    OffsetSampler,
    autocast_context,
    collect_rank_rng,
    free_memory_gib,
    make_logger,
    require_memory_reserve,
    restore_rank_rng,
    set_seed,
    setup_distributed,
)
from stage2.transport import create_transport  # noqa: E402
from stage2.utils import apply_cfg_dropout, validate_stage2_config  # noqa: E402
from utils.model_utils import instantiate_from_config  # noqa: E402


LPL_LAYER_FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)
LPL_VARIANTS = (
    "prediction_full",
    "prediction_detach",
    "target_normalized",
    "symmetric",
    "raw",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--packed-data-path", type=Path)
    parser.add_argument("--index-map", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-state-key", choices=("model", "ema"), default="model")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--objective", choices=("flow", "lpl"), required=True)
    parser.add_argument("--max-updates", type=int, required=True)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--global-batch-size", type=int, default=1024)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--adapter-eps", type=float, default=1e-6)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--ema-decay", type=float, default=0.9995)
    parser.add_argument("--lpl-weight", type=float, default=1.0)
    parser.add_argument("--lpl-variant", choices=LPL_VARIANTS, default="raw")
    parser.add_argument(
        "--lpl-prediction-target",
        choices=("guided", "full"),
        default="guided",
    )
    parser.add_argument("--lpl-noise-threshold", type=float, default=3.0)
    parser.add_argument("--lpl-max-samples-per-rank", type=int, default=1)
    parser.add_argument(
        "--gradient-audit-component",
        choices=("flow", "lpl"),
        help="Accumulate a gradient probe and exit without an optimizer step.",
    )
    parser.add_argument("--gradient-audit-microbatches", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--min-free-gib", type=float, default=0.5)
    parser.add_argument(
        "--dino-ckpt-dir",
        type=Path,
        default=Path("/home/zhoushunyu/data/eqvae/models/RAEv2/encoders/dinov3"),
    )
    parser.add_argument("--dino-repo-dir", type=Path)
    return parser.parse_args()


def load_config(path: Path) -> Stage2Config:
    config = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(Stage2Config), OmegaConf.load(path))
    )
    config.post_process()
    validate_stage2_config(config)
    if config.transport.prediction != "x":
        raise ValueError("common-adapter study requires RAEv2 x-prediction")
    config.prepare_model_params()
    return config


def _validate_args(args: argparse.Namespace, world_size: int) -> int:
    positive = {
        "max_updates": args.max_updates,
        "save_every": args.save_every,
        "global_batch_size": args.global_batch_size,
        "micro_batch_size": args.micro_batch_size,
        "hidden_channels": args.hidden_channels,
        "learning_rate": args.learning_rate,
        "gradient_audit_microbatches": args.gradient_audit_microbatches,
        "lpl_max_samples_per_rank": args.lpl_max_samples_per_rank,
    }
    invalid = [name for name, value in positive.items() if float(value) <= 0]
    if invalid:
        raise ValueError(f"arguments must be positive: {invalid}")
    if args.adapter_eps <= 0:
        raise ValueError("--adapter-eps must be positive")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay must be non-negative")
    if not 0 <= args.ema_decay < 1:
        raise ValueError("--ema-decay must lie in [0, 1)")
    if args.objective == "flow" and args.gradient_audit_component == "lpl":
        raise ValueError("an LPL gradient audit requires --objective lpl")
    denominator = world_size * int(args.micro_batch_size)
    if int(args.global_batch_size) % denominator:
        raise ValueError(
            "global batch size must be divisible by world size * micro batch size"
        )
    return int(args.global_batch_size) // denominator


def _optimizer_boundary(
    optimizer: torch.optim.Optimizer,
    adapter: torch.nn.Module,
    source_model: torch.nn.Module,
) -> dict[str, int]:
    adapter_ids = {id(parameter) for parameter in adapter.parameters()}
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if optimizer_ids != adapter_ids:
        raise RuntimeError("optimizer does not contain exactly the adapter parameters")
    if any(parameter.requires_grad for parameter in source_model.parameters()):
        raise RuntimeError("source Stage-2 model is not fully frozen")
    return {
        "trainable_parameters": sum(parameter.numel() for parameter in adapter.parameters()),
        "optimizer_parameter_tensors": len(optimizer_ids),
        "frozen_source_parameters": sum(
            parameter.numel() for parameter in source_model.parameters()
        ),
    }


def _restore_resume_rng(checkpoint: dict[str, Any], rank: int) -> None:
    shadow = {
        "raev2_lpl": {
            "rank_rng_states": checkpoint["common_adapter"]["rank_rng_states"]
        }
    }
    restore_rank_rng(shadow, rank)


def _save_checkpoint(
    path: Path,
    *,
    adapter: CommonResidualAdapter,
    adapter_ema: CommonResidualAdapter,
    optimizer: torch.optim.Optimizer,
    source_checkpoint: Path,
    source_sha256: str,
    source_state_key: str,
    objective: str,
    lpl_variant: str,
    lpl_prediction_target: str,
    branch_update: int,
    data_indices_sha256: str,
    rank_rng_states: list[dict[str, Any]],
) -> None:
    payload = {
        "format": COMMON_ADAPTER_FORMAT,
        "adapter_config": adapter.config_dict(),
        "adapter": adapter.state_dict(),
        "adapter_ema": adapter_ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "common_adapter": {
            "source_checkpoint": str(source_checkpoint.expanduser().resolve()),
            "source_sha256": source_sha256,
            "source_state_key": source_state_key,
            "objective": objective,
            "lpl_variant": lpl_variant if objective == "lpl" else None,
            "lpl_prediction_target": (
                lpl_prediction_target if objective == "lpl" else None
            ),
            "branch_update": int(branch_update),
            "data_indices_sha256": data_indices_sha256,
            "rank_rng_states": rank_rng_states,
            "source_model_frozen": True,
            "contrast_function_preserved_by_parameterization": True,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


@torch.no_grad()
def _update_ema(
    target: torch.nn.Module,
    source: torch.nn.Module,
    *,
    decay: float,
) -> None:
    target_parameters = dict(target.named_parameters())
    source_parameters = dict(source.named_parameters())
    if target_parameters.keys() != source_parameters.keys():
        raise RuntimeError("adapter and adapter EMA parameter names differ")
    for name, parameter in source_parameters.items():
        target_parameters[name].mul_(float(decay)).add_(
            parameter.detach(),
            alpha=1.0 - float(decay),
        )
    target_buffers = dict(target.named_buffers())
    source_buffers = dict(source.named_buffers())
    for name, buffer in source_buffers.items():
        target_buffers[name].copy_(buffer)


def main() -> None:
    install_raev2_decoder_config_compat()
    args = parse_args()
    os.environ["DINOV3_CKPT_DIR"] = str(args.dino_ckpt_dir.expanduser().resolve())
    if args.dino_repo_dir is not None:
        os.environ["DINOV3_REPO_DIR"] = str(args.dino_repo_dir.expanduser().resolve())

    rank, world_size, device = setup_distributed()
    grad_accum_steps = _validate_args(args, world_size)
    experiment_dir = args.results_dir.expanduser() / args.experiment_name
    if rank == 0:
        experiment_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    logger = make_logger(experiment_dir, rank)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    config = load_config(args.config)
    rank_seed = int(args.global_seed) * world_size + rank
    set_seed(rank_seed)

    source_checkpoint_path = args.source_checkpoint.expanduser().resolve()
    source_sha256 = file_sha256(source_checkpoint_path) if rank == 0 else ""
    hashes = [source_sha256]
    dist.broadcast_object_list(hashes, src=0)
    source_sha256 = hashes[0]
    source_checkpoint = torch.load(
        source_checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    validate_full_stage2_checkpoint(source_checkpoint)
    source_model = instantiate_from_config(config.stage_2).to(device)
    source_model.load_state_dict(
        source_checkpoint[args.source_state_key],
        strict=True,
    )
    source_model.eval()
    source_model.requires_grad_(False)
    source_step = int(source_checkpoint["step"])
    source_epoch = int(source_checkpoint["epoch"])
    del source_checkpoint

    adapter = CommonResidualAdapter(
        int(source_model.in_channels),
        hidden_channels=int(args.hidden_channels),
        eps=float(args.adapter_eps),
    ).to(device)
    adapter_ema = deepcopy(adapter).to(device)
    adapter_ema.eval()
    adapter_ema.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    branch_update = 0
    resume_rank_rng_states = None

    if args.resume is not None:
        resume_checkpoint = torch.load(
            args.resume.expanduser(),
            map_location="cpu",
            weights_only=False,
        )
        if resume_checkpoint.get("format") != COMMON_ADAPTER_FORMAT:
            raise ValueError("resume checkpoint is not a common-adapter checkpoint")
        metadata = resume_checkpoint["common_adapter"]
        checks = {
            "source_sha256": source_sha256,
            "source_state_key": args.source_state_key,
            "objective": args.objective,
        }
        for key, expected in checks.items():
            if metadata.get(key) != expected:
                raise ValueError(
                    f"resume {key} mismatch: {metadata.get(key)!r} != {expected!r}"
                )
        stored_variant = metadata.get("lpl_variant")
        expected_variant = args.lpl_variant if args.objective == "lpl" else None
        if stored_variant != expected_variant:
            raise ValueError("resume LPL variant mismatch")
        stored_prediction_target = metadata.get("lpl_prediction_target")
        expected_prediction_target = (
            args.lpl_prediction_target if args.objective == "lpl" else None
        )
        if stored_prediction_target != expected_prediction_target:
            raise ValueError("resume LPL prediction target mismatch")
        if resume_checkpoint.get("adapter_config") != adapter.config_dict():
            raise ValueError("resume adapter config mismatch")
        adapter.load_state_dict(resume_checkpoint["adapter"], strict=True)
        adapter_ema.load_state_dict(resume_checkpoint["adapter_ema"], strict=True)
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        branch_update = int(metadata["branch_update"])
        resume_rank_rng_states = metadata.get("rank_rng_states")
        if resume_rank_rng_states is None:
            raise ValueError("resume checkpoint has no per-rank RNG state")
        del resume_checkpoint
    else:
        # Independent Flow/LPL launches use identical adapter initialization and
        # all subsequent training randomness.
        set_seed(rank_seed + 1_000_000)

    if branch_update >= args.max_updates:
        raise ValueError("resume checkpoint is already at or beyond max updates")
    start_branch_update = branch_update
    ddp_adapter = DDP(
        adapter,
        device_ids=[device.index],
        broadcast_buffers=False,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
    )
    adapter = ddp_adapter.module
    optimizer_audit = _optimizer_boundary(optimizer, adapter, source_model)

    dataset_parameters = {
        "split": "train",
        "image_size": int(config.training.image_size),
        "augmentation_seed": int(args.global_seed),
        "horizontal_flip": False,
        "index_map_path": args.index_map,
    }
    if args.packed_data_path is not None:
        dataset = DeterministicImageNetPacked(
            args.packed_data_path,
            **dataset_parameters,
        )
        dataset_backend = "packed_random_access"
        active_data_path = args.packed_data_path
    else:
        dataset = DeterministicImageNetParquet(
            args.data_path,
            **dataset_parameters,
        )
        dataset_backend = "parquet_row_group"
        active_data_path = args.data_path
    distributed_sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=int(args.global_seed),
        drop_last=True,
    )
    distributed_sampler.set_epoch(source_epoch)
    consumed_samples_per_rank = (
        branch_update * grad_accum_steps * int(args.micro_batch_size)
    )
    sampler = OffsetSampler(distributed_sampler, consumed_samples_per_rank)
    loader_generator = torch.Generator().manual_seed(rank_seed + 100_000)
    loader = DataLoader(
        dataset,
        batch_size=int(args.micro_batch_size),
        sampler=sampler,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=True,
        persistent_workers=int(args.num_workers) > 0,
        multiprocessing_context="spawn" if int(args.num_workers) > 0 else None,
        generator=loader_generator,
    )

    rae = instantiate_from_config(config.stage_1).to(device)
    rae.eval()
    rae.requires_grad_(False)
    if args.objective == "flow":
        del rae.decoder
        torch.cuda.empty_cache()

    latent_size = tuple(config.misc.latent_size)
    time_dist_shift = math.sqrt(
        (config.misc.time_dist_shift_dim or math.prod(latent_size))
        / config.misc.time_dist_shift_base
    )
    transport = create_transport(config=config.transport, time_dist_shift=time_dist_shift)
    null_context = torch.full(
        (int(args.micro_batch_size),),
        int(config.misc.num_classes),
        dtype=torch.long,
        device=device,
    )
    layer_indices = None
    layer_weights = None
    if args.objective == "lpl":
        layer_indices = decoder_hidden_indices(
            len(rae.decoder.decoder_layers),
            fractions=LPL_LAYER_FRACTIONS,
        )
        layer_weights = (1.0,) * len(layer_indices)

    manifest = {
        "format": COMMON_ADAPTER_FORMAT,
        "config": str(args.config.resolve()),
        "source_checkpoint": str(source_checkpoint_path),
        "source_sha256": source_sha256,
        "source_state_key": args.source_state_key,
        "source_step": source_step,
        "source_epoch": source_epoch,
        "objective": args.objective,
        "lpl_variant": args.lpl_variant if args.objective == "lpl" else None,
        "lpl_prediction_target": (
            args.lpl_prediction_target if args.objective == "lpl" else None
        ),
        "lpl_weight": float(args.lpl_weight) if args.objective == "lpl" else 0.0,
        "sampling_guidance": {
            "scale": float(config.guidance.ig.scale),
            "interval": [
                float(config.guidance.ig.t_min),
                float(config.guidance.ig.t_max),
            ],
        },
        "adapter_config": adapter.config_dict(),
        "optimizer": {
            "type": "AdamW",
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
        },
        "ema_decay": float(args.ema_decay),
        "world_size": world_size,
        "global_batch_size": int(args.global_batch_size),
        "micro_batch_size": int(args.micro_batch_size),
        "grad_accum_steps": grad_accum_steps,
        "global_seed": int(args.global_seed),
        "dataset": str(active_data_path.expanduser().resolve()),
        "dataset_backend": dataset_backend,
        "dataset_index_map": str(args.index_map.expanduser().resolve()),
        "dataset_index_map_sha256": file_sha256(args.index_map),
        "source_model_frozen": True,
        "optimizer_boundary": optimizer_audit,
        "contrast_function_preserved_by_parameterization": True,
        "validation_or_reference_used_for_training": False,
    }
    if rank == 0:
        (experiment_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Common adapter config: %s", adapter.config_dict())
        logger.info(
            "Batch global=%d micro=%d accumulation=%d world=%d",
            args.global_batch_size,
            args.micro_batch_size,
            grad_accum_steps,
            world_size,
        )

    require_memory_reserve(device, args.min_free_gib, "model load")
    if resume_rank_rng_states is not None:
        _restore_resume_rng(
            {"common_adapter": {"rank_rng_states": resume_rank_rng_states}},
            rank,
        )
    ddp_adapter.train()
    optimizer.zero_grad(set_to_none=True)
    metrics_path = experiment_dir / "train_metrics.jsonl"
    data_index_digest = hashlib.sha256()
    first_batch_written = False
    accumulation_boundary = (
        int(args.gradient_audit_microbatches)
        if args.gradient_audit_component
        else grad_accum_steps
    )
    micro_since_boundary = 0
    update_flow_sum = torch.zeros((), device=device)
    update_lpl_sum = torch.zeros((), device=device)
    update_total_sum = torch.zeros((), device=device)
    update_gate_count = torch.zeros((), device=device)
    update_contrast_error = torch.zeros((), device=device)
    update_correction_rms_sum = torch.zeros((), device=device)
    update_source_guided_rms_sum = torch.zeros((), device=device)
    last_time = perf_counter()

    for images, labels, indices in loader:
        if branch_update >= args.max_updates:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        indices = indices.to(device)
        for value in indices.detach().cpu().tolist():
            data_index_digest.update(int(value).to_bytes(8, "little", signed=False))

        with torch.no_grad():
            clean_latent = rae.encode(images)
        model_kwargs = {"context": labels, "attn_mask": None}
        null_kwargs = {"context": null_context, "attn_mask": None}
        micro_since_boundary += 1
        is_boundary = micro_since_boundary == accumulation_boundary
        sync_context = nullcontext() if is_boundary else ddp_adapter.no_sync()

        with sync_context:
            with autocast_context(args.precision):
                dropped_kwargs, cfg_mask = apply_cfg_dropout(
                    model_kwargs,
                    null_kwargs,
                    float(config.conditioning.cfg_dropout_prob),
                )
                time, noise, clean_latent = transport.sample(clean_latent)
                time_scale = time.reshape(
                    (time.shape[0],) + (1,) * (clean_latent.ndim - 1)
                )
                noisy_latent = (
                    (1.0 - time_scale) * clean_latent + time_scale * noise
                )
                target_velocity = (
                    (noisy_latent - clean_latent)
                    / time_scale.clamp_min(float(config.transport.t_eps))
                )

                selected = torch.empty(0, device=device, dtype=torch.long)
                target_features = None
                if args.objective == "lpl":
                    gate = lpl_time_gate(time, float(args.lpl_noise_threshold))
                    selected = torch.nonzero(gate, as_tuple=False).flatten()
                    selected = selected[: int(args.lpl_max_samples_per_rank)]
                    if selected.numel():
                        with torch.no_grad():
                            target_features = tuple(
                                feature.float()
                                for feature in decoder_feature_pyramid(
                                    rae,
                                    clean_latent.index_select(0, selected),
                                    layer_indices=layer_indices,
                                )
                            )

                with torch.no_grad():
                    source_full, source_base = source_model(
                        noisy_latent,
                        time,
                        **dropped_kwargs,
                    )
                correction = ddp_adapter(
                    noisy_latent,
                    time,
                    source_full,
                    source_base,
                ).float()
                corrected_full = source_full.float() + correction
                corrected_base = source_base.float() + correction
                corrected_output = (corrected_full, corrected_base)
                source_guided = internal_guidance_prediction(
                    source_full,
                    source_base,
                    time,
                    scale=float(config.guidance.ig.scale),
                    interval=(
                        float(config.guidance.ig.t_min),
                        float(config.guidance.ig.t_max),
                    ),
                )
                corrected_guided = source_guided.float() + correction
                contrast_error = (
                    (corrected_full - corrected_base)
                    - (source_full.float() - source_base.float())
                ).float().abs().max()

                flow_map, _ = official_flow_loss_map(
                    transport,
                    corrected_output,
                    target_velocity=target_velocity,
                    noisy_latent=noisy_latent,
                    time=time,
                    base_model_coeff=float(config.internal_guidance.base_model_coeff),
                )
                flow_loss = flow_map.mean()
                lpl_loss = torch.zeros((), device=device)
                total_loss = flow_loss
                if args.objective == "lpl" and selected.numel():
                    if target_features is None:
                        raise RuntimeError("active LPL gate has no target features")
                    predicted_features = tuple(
                        feature.float()
                        for feature in decoder_feature_pyramid(
                            rae,
                            (
                                corrected_guided
                                if args.lpl_prediction_target == "guided"
                                else corrected_full
                            ).index_select(0, selected),
                            layer_indices=layer_indices,
                        )
                    )
                    lpl_per_sample, _ = decoder_feature_objective_per_sample(
                        args.lpl_variant,
                        target_features,
                        predicted_features,
                        layer_weights=layer_weights,
                    )
                    lpl_loss = lpl_per_sample.mean()
                    total_loss = total_loss + float(args.lpl_weight) * lpl_loss

            if args.gradient_audit_component == "flow":
                backward_loss = flow_loss
            elif args.gradient_audit_component == "lpl":
                backward_loss = flow_loss * 0.0 + float(args.lpl_weight) * lpl_loss
            else:
                backward_loss = total_loss
            (backward_loss / accumulation_boundary).backward()

        if not first_batch_written:
            audit = {
                "rank": rank,
                "indices": indices.detach().cpu().tolist(),
                "image_sha256": tensor_fingerprint(images),
                "label_sha256": tensor_fingerprint(labels),
                "latent_sha256": tensor_fingerprint(clean_latent),
                "noise_sha256": tensor_fingerprint(noise),
                "time_sha256": tensor_fingerprint(time),
                "cfg_mask_sha256": tensor_fingerprint(cfg_mask),
                "initial_correction_max_abs": float(correction.detach().abs().max()),
                "contrast_error_max_abs": float(contrast_error.detach()),
            }
            gathered = [None] * world_size if rank == 0 else None
            dist.gather_object(audit, gathered, dst=0)
            if rank == 0:
                audit_name = (
                    "first_batch_audit.json"
                    if start_branch_update == 0
                    else f"resume_from_{start_branch_update:07d}_batch_audit.json"
                )
                (experiment_dir / audit_name).write_text(
                    json.dumps(gathered, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            first_batch_written = True

        update_flow_sum += flow_loss.detach()
        update_lpl_sum += lpl_loss.detach()
        update_total_sum += total_loss.detach()
        update_gate_count += int(selected.numel())
        update_contrast_error.copy_(
            torch.maximum(update_contrast_error, contrast_error.detach())
        )
        update_correction_rms_sum += correction.detach().square().mean().sqrt()
        update_source_guided_rms_sum += (
            source_guided.detach().float().square().mean().sqrt()
        )
        if not is_boundary:
            continue

        micro_since_boundary = 0
        grad_norm = torch.nn.utils.clip_grad_norm_(
            adapter.parameters(),
            float(config.training.clip_grad)
            if config.training.clip_grad is not None
            else float("inf"),
        )
        if args.gradient_audit_component:
            reduced = torch.stack(
                (
                    update_flow_sum,
                    update_lpl_sum,
                    update_gate_count,
                    grad_norm.detach(),
                    update_contrast_error,
                    update_correction_rms_sum,
                    update_source_guided_rms_sum,
                )
            )
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
            reduced[:3] /= world_size * accumulation_boundary
            reduced[3] /= world_size
            reduced[5:] /= world_size * accumulation_boundary
            if rank == 0:
                gradient_vector = torch.cat(
                    tuple(
                        parameter.grad.detach().float().reshape(-1).cpu()
                        for parameter in adapter.parameters()
                        if parameter.grad is not None
                    )
                )
                gradient_norm_after_clip = gradient_vector.norm()
                if not torch.isfinite(gradient_norm_after_clip) or not (
                    gradient_norm_after_clip > 0
                ):
                    raise RuntimeError("gradient audit produced an invalid gradient")
                gradient_path = experiment_dir / "gradient_unit.pt"
                torch.save(
                    {
                        "component": args.gradient_audit_component,
                        "unit_gradient": gradient_vector / gradient_norm_after_clip,
                        "parameter_names": tuple(
                            name
                            for name, parameter in adapter.named_parameters()
                            if parameter.grad is not None
                        ),
                    },
                    gradient_path,
                )
                result = {
                    "format": COMMON_ADAPTER_FORMAT,
                    "component": args.gradient_audit_component,
                    "objective": args.objective,
                    "lpl_variant": (
                        args.lpl_variant if args.objective == "lpl" else None
                    ),
                    "lpl_prediction_target": (
                        args.lpl_prediction_target
                        if args.objective == "lpl"
                        else None
                    ),
                    "lpl_weight": (
                        float(args.lpl_weight) if args.objective == "lpl" else 0.0
                    ),
                    "global_samples": (
                        world_size
                        * accumulation_boundary
                        * int(args.micro_batch_size)
                    ),
                    "flow_loss": float(reduced[0]),
                    "lpl_loss": float(reduced[1]),
                    "mean_lpl_gate_count": float(reduced[2]),
                    "parameter_gradient_norm": float(reduced[3]),
                    "gradient_norm_after_clip": float(gradient_norm_after_clip),
                    "gradient_unit_path": str(gradient_path.resolve()),
                    "gradient_unit_sha256": file_sha256(gradient_path),
                    "contrast_error_rank_sum_upper_bound": float(reduced[4]),
                    "mean_correction_rms": float(reduced[5]),
                    "mean_source_guided_rms": float(reduced[6]),
                    "correction_over_source_guided": float(
                        reduced[5] / reduced[6].clamp_min(1e-12)
                    ),
                    "data_indices_sha256": data_index_digest.hexdigest(),
                }
                (experiment_dir / "gradient_audit.json").write_text(
                    json.dumps(result, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                logger.info("Gradient audit: %s", result)
            optimizer.zero_grad(set_to_none=True)
            break

        require_memory_reserve(device, args.min_free_gib, "adapter backward")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        _update_ema(adapter_ema, adapter, decay=float(args.ema_decay))
        branch_update += 1

        reduced = torch.stack(
            (
                update_flow_sum,
                update_lpl_sum,
                update_total_sum,
                grad_norm.detach(),
                update_gate_count,
                update_contrast_error,
                update_correction_rms_sum,
                update_source_guided_rms_sum,
            )
        )
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        reduced[:3] /= world_size * grad_accum_steps
        reduced[3] /= world_size
        reduced[4] /= world_size * grad_accum_steps
        reduced[6:] /= world_size * grad_accum_steps
        if rank == 0:
            now = perf_counter()
            row = {
                "branch_update": branch_update,
                "flow_loss": float(reduced[0]),
                "lpl_loss": float(reduced[1]),
                "total_loss": float(reduced[2]),
                "grad_norm": float(reduced[3]),
                "mean_lpl_gate_count": float(reduced[4]),
                "contrast_error_rank_sum_upper_bound": float(reduced[5]),
                "mean_correction_rms": float(reduced[6]),
                "mean_source_guided_rms": float(reduced[7]),
                "correction_over_source_guided": float(
                    reduced[6] / reduced[7].clamp_min(1e-12)
                ),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "seconds_per_update": now - last_time,
                "free_gpu_gib_rank0": free_memory_gib(device),
            }
            append_jsonl(metrics_path, row)
            logger.info(
                "update=%d flow=%.6f lpl=%.6f total=%.6f grad=%.4f "
                "contrast<=%.3g correction/source=%.3g free=%.2fGiB",
                branch_update,
                row["flow_loss"],
                row["lpl_loss"],
                row["total_loss"],
                row["grad_norm"],
                row["contrast_error_rank_sum_upper_bound"],
                row["correction_over_source_guided"],
                row["free_gpu_gib_rank0"],
            )
            last_time = now

        update_flow_sum.zero_()
        update_lpl_sum.zero_()
        update_total_sum.zero_()
        update_gate_count.zero_()
        update_contrast_error.zero_()
        update_correction_rms_sum.zero_()
        update_source_guided_rms_sum.zero_()

        should_save = (
            branch_update % int(args.save_every) == 0
            or branch_update == int(args.max_updates)
        )
        if should_save:
            rank_rng_states = collect_rank_rng(rank, world_size)
            if rank == 0:
                checkpoint_path = (
                    experiment_dir
                    / "checkpoints"
                    / f"adapter-{branch_update:07d}.pt"
                )
                _save_checkpoint(
                    checkpoint_path,
                    adapter=adapter,
                    adapter_ema=adapter_ema,
                    optimizer=optimizer,
                    source_checkpoint=source_checkpoint_path,
                    source_sha256=source_sha256,
                    source_state_key=args.source_state_key,
                    objective=args.objective,
                    lpl_variant=args.lpl_variant,
                    lpl_prediction_target=args.lpl_prediction_target,
                    branch_update=branch_update,
                    data_indices_sha256=data_index_digest.hexdigest(),
                    rank_rng_states=rank_rng_states,
                )
                logger.info("Saved %s", checkpoint_path)
            dist.barrier()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
