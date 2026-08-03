"""Train an invertible latent reparameterizer around a frozen RAEv2 Stage-2.

Only ``A`` is optimized.  The RAE encoder, decoder, and Stage-2 model remain
frozen.  Clean latents are transformed as ``u=A(z)``; frozen Stage-2 endpoint
predictions are mapped back through ``A^{-1}`` before decoder-feature LPL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


ROOT = Path(__file__).resolve().parents[1]
RAEV2_SRC = ROOT / "external" / "RAEv2" / "src"
for path in (RAEV2_SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from configs.stage2 import Stage2Config  # noqa: E402
from experiments.latent_equiv_adapter import InvertibleLatentAdapter  # noqa: E402
from experiments.rae_lpl_detach_audit import (  # noqa: E402
    decoder_feature_objective_per_sample,
)
from experiments.rae_strict_lpl import (  # noqa: E402
    decoder_feature_pyramid,
    decoder_hidden_indices,
    lpl_time_gate,
)
from experiments.raev2_common_adapter import (  # noqa: E402
    internal_guidance_prediction,
)
from experiments.raev2_invertible_latent_lpl import (  # noqa: E402
    INVERTIBLE_LATENT_LPL_FORMAT,
    adapter_config,
    all_reduce_adapter_gradients,
    cycle_metrics,
    inverse_prediction,
    make_reparameterized_path,
    normalized_mse,
    trainable_parameter_boundary,
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
    make_logger,
    require_memory_reserve,
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
    parser.add_argument("--init-adapter-checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--adapter-state-key",
        choices=("adapter", "adapter_ema"),
        default="adapter",
    )
    parser.add_argument("--objective", choices=("flow", "lpl"), required=True)
    parser.add_argument("--max-updates", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--ema-decay", type=float, default=0.9995)
    parser.add_argument("--lpl-weight", type=float, default=0.0)
    parser.add_argument("--lpl-variant", choices=LPL_VARIANTS, default="prediction_full")
    parser.add_argument("--lpl-prediction-target", choices=("full", "guided"), default="full")
    parser.add_argument("--lpl-noise-threshold", type=float, default=3.0)
    parser.add_argument("--lpl-max-samples-per-rank", type=int, default=1)
    parser.add_argument("--data-identity-weight", type=float, default=1e-3)
    parser.add_argument("--noise-identity-weight", type=float, default=1e-3)
    parser.add_argument("--gradient-audit-only", action="store_true")
    parser.add_argument("--gradient-audit-microbatches", type=int, default=8)
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
        raise ValueError("invertible latent LPL currently requires RAEv2 x-prediction")
    config.prepare_model_params()
    return config


def validate_args(args: argparse.Namespace, world_size: int) -> int:
    positive = {
        "max_updates": args.max_updates,
        "save_every": args.save_every,
        "global_batch_size": args.global_batch_size,
        "micro_batch_size": args.micro_batch_size,
        "blocks": args.blocks,
        "hidden_channels": args.hidden_channels,
        "learning_rate": args.learning_rate,
        "gradient_audit_microbatches": args.gradient_audit_microbatches,
        "lpl_max_samples_per_rank": args.lpl_max_samples_per_rank,
    }
    invalid = [name for name, value in positive.items() if float(value) <= 0]
    if invalid:
        raise ValueError(f"arguments must be positive: {invalid}")
    for name in (
        "weight_decay",
        "lpl_weight",
        "data_identity_weight",
        "noise_identity_weight",
        "min_free_gib",
    ):
        if float(getattr(args, name)) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if args.objective == "lpl" and args.lpl_weight <= 0 and not args.gradient_audit_only:
        raise ValueError("LPL training requires a positive --lpl-weight")
    if args.init_adapter_checkpoint is not None and args.resume is not None:
        raise ValueError("--init-adapter-checkpoint and --resume are mutually exclusive")
    if not 0 <= args.ema_decay < 1:
        raise ValueError("--ema-decay must lie in [0, 1)")
    denominator = world_size * int(args.micro_batch_size)
    if int(args.global_batch_size) % denominator:
        raise ValueError(
            "global batch size must be divisible by world size * micro batch size"
        )
    return int(args.global_batch_size) // denominator


def _gradient_vector(
    loss: torch.Tensor,
    parameters: tuple[torch.nn.Parameter, ...],
    *,
    retain_graph: bool,
) -> torch.Tensor:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    return torch.cat(
        tuple(
            torch.zeros_like(parameter).reshape(-1)
            if gradient is None
            else gradient.detach().float().reshape(-1)
            for parameter, gradient in zip(parameters, gradients, strict=True)
        )
    )


def _cosine(left: torch.Tensor, right: torch.Tensor, eps: float = 1e-12) -> float:
    denominator = left.norm() * right.norm()
    return float((left @ right / denominator.clamp_min(float(eps))).item())


def _save_checkpoint(
    path: Path,
    *,
    adapter: InvertibleLatentAdapter,
    adapter_ema: InvertibleLatentAdapter,
    optimizer: torch.optim.Optimizer,
    metadata: dict[str, Any],
    rank_rng_states: list[dict[str, Any]],
) -> None:
    payload = {
        "format": INVERTIBLE_LATENT_LPL_FORMAT,
        "adapter_config": adapter_config(adapter),
        "adapter": adapter.state_dict(),
        "adapter_ema": adapter_ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "invertible_latent_lpl": {
            **metadata,
            "rank_rng_states": rank_rng_states,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _update_ema(
    target: torch.nn.Module,
    source: torch.nn.Module,
    decay: float,
) -> None:
    with torch.no_grad():
        for target_parameter, source_parameter in zip(
            target.parameters(), source.parameters(), strict=True
        ):
            target_parameter.mul_(float(decay)).add_(
                source_parameter, alpha=1.0 - float(decay)
            )


def _restore_rank_rng(states: list[dict[str, Any]], rank: int) -> None:
    matches = [state for state in states if int(state["rank"]) == rank]
    if len(matches) != 1:
        raise ValueError(f"expected one RNG state for rank {rank}, found {len(matches)}")
    state = matches[0]
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["torch_cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def main() -> None:
    args = parse_args()
    install_raev2_decoder_config_compat()
    os.environ["DINO_CKPT_DIR"] = str(args.dino_ckpt_dir.expanduser().resolve())
    if args.dino_repo_dir is not None:
        os.environ["DINOV3_REPO_DIR"] = str(args.dino_repo_dir.expanduser().resolve())

    rank, world_size, device = setup_distributed()
    grad_accum_steps = validate_args(args, world_size)
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

    source_path = args.source_checkpoint.expanduser().resolve()
    source_sha256 = file_sha256(source_path) if rank == 0 else ""
    hashes = [source_sha256]
    dist.broadcast_object_list(hashes, src=0)
    source_sha256 = hashes[0]
    source_checkpoint = torch.load(
        source_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    validate_full_stage2_checkpoint(source_checkpoint)
    source_model = instantiate_from_config(config.stage_2).to(device)
    source_model.load_state_dict(source_checkpoint[args.source_state_key], strict=True)
    source_model.eval().requires_grad_(False)
    source_step = int(source_checkpoint["step"])
    source_epoch = int(source_checkpoint["epoch"])
    del source_checkpoint

    rae = instantiate_from_config(config.stage_1).to(device)
    rae.eval().requires_grad_(False)

    # Every rank must start from the exact same random coupling weights.
    set_seed(int(args.global_seed) + 2_000_000)
    adapter = InvertibleLatentAdapter(
        channels=int(source_model.in_channels),
        hidden_channels=int(args.hidden_channels),
        blocks=int(args.blocks),
    ).to(device)
    init_adapter_path = None
    init_adapter_sha256 = None
    resume_path = None
    resume_sha256 = None
    resume_checkpoint = None
    resume_metadata: dict[str, Any] = {}
    resume_rank_rng_states = None
    branch_update = 0
    if args.init_adapter_checkpoint is not None:
        init_adapter_path = args.init_adapter_checkpoint.expanduser().resolve()
        init_adapter_sha256 = file_sha256(init_adapter_path) if rank == 0 else ""
        adapter_hashes = [init_adapter_sha256]
        dist.broadcast_object_list(adapter_hashes, src=0)
        init_adapter_sha256 = adapter_hashes[0]
        adapter_checkpoint = torch.load(
            init_adapter_path,
            map_location="cpu",
            weights_only=False,
        )
        if adapter_checkpoint.get("format") != INVERTIBLE_LATENT_LPL_FORMAT:
            raise ValueError("init adapter checkpoint has an unexpected format")
        if adapter_checkpoint.get("adapter_config") != adapter_config(adapter):
            raise ValueError("init adapter checkpoint architecture mismatch")
        adapter_metadata = adapter_checkpoint.get("invertible_latent_lpl", {})
        if adapter_metadata.get("source_sha256") != source_sha256:
            raise ValueError("init adapter checkpoint source-model hash mismatch")
        if adapter_metadata.get("source_state_key") != args.source_state_key:
            raise ValueError("init adapter checkpoint source-state mismatch")
        adapter.load_state_dict(
            adapter_checkpoint[args.adapter_state_key],
            strict=True,
        )
        del adapter_checkpoint
    elif args.resume is not None:
        resume_path = args.resume.expanduser().resolve()
        resume_sha256 = file_sha256(resume_path) if rank == 0 else ""
        resume_hashes = [resume_sha256]
        dist.broadcast_object_list(resume_hashes, src=0)
        resume_sha256 = resume_hashes[0]
        resume_checkpoint = torch.load(
            resume_path,
            map_location="cpu",
            weights_only=False,
        )
        if resume_checkpoint.get("format") != INVERTIBLE_LATENT_LPL_FORMAT:
            raise ValueError("resume checkpoint has an unexpected format")
        if resume_checkpoint.get("adapter_config") != adapter_config(adapter):
            raise ValueError("resume adapter checkpoint architecture mismatch")
        resume_metadata = resume_checkpoint.get("invertible_latent_lpl", {})
        checks = {
            "source_sha256": source_sha256,
            "source_state_key": args.source_state_key,
            "objective": args.objective,
            "lpl_variant": args.lpl_variant,
            "lpl_prediction_target": args.lpl_prediction_target,
            "lpl_weight": float(args.lpl_weight),
        }
        for key, expected in checks.items():
            actual = resume_metadata.get(key)
            if actual != expected:
                raise ValueError(f"resume {key} mismatch: {actual!r} != {expected!r}")
        adapter.load_state_dict(resume_checkpoint["adapter"], strict=True)
        branch_update = int(resume_metadata["branch_update"])
        resume_rank_rng_states = resume_metadata.get("rank_rng_states")
        if resume_rank_rng_states is None:
            raise ValueError("resume checkpoint has no per-rank RNG states")
    adapter_ema = deepcopy(adapter).to(device).eval().requires_grad_(False)
    if resume_checkpoint is not None:
        adapter_ema.load_state_dict(resume_checkpoint["adapter_ema"], strict=True)
    set_seed(rank_seed + 1_000_000)
    for parameter in adapter.parameters():
        dist.broadcast(parameter.data, src=0)

    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        del resume_checkpoint
    if branch_update >= int(args.max_updates):
        raise ValueError("resume checkpoint is already at or beyond --max-updates")
    start_branch_update = branch_update
    optimizer_audit = trainable_parameter_boundary(
        adapter,
        (source_model, rae),
        optimizer,
    )

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
    loader = DataLoader(
        dataset,
        batch_size=int(args.micro_batch_size),
        sampler=sampler,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=True,
        persistent_workers=int(args.num_workers) > 0,
        multiprocessing_context="spawn" if int(args.num_workers) > 0 else None,
        generator=torch.Generator().manual_seed(rank_seed + 100_000),
    )

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
    layer_indices = decoder_hidden_indices(
        len(rae.decoder.decoder_layers),
        fractions=LPL_LAYER_FRACTIONS,
    )
    layer_weights = (1.0,) * len(layer_indices)

    manifest = {
        "format": INVERTIBLE_LATENT_LPL_FORMAT,
        "config": str(args.config.resolve()),
        "source_checkpoint": str(source_path),
        "source_sha256": source_sha256,
        "source_state_key": args.source_state_key,
        "source_step": source_step,
        "source_epoch": source_epoch,
        "objective": args.objective,
        "lpl_variant": args.lpl_variant,
        "lpl_prediction_target": args.lpl_prediction_target,
        "lpl_weight": float(args.lpl_weight),
        "adapter_config": adapter_config(adapter),
        "init_adapter_checkpoint": (
            str(init_adapter_path) if init_adapter_path is not None else None
        ),
        "init_adapter_sha256": init_adapter_sha256,
        "adapter_state_key": (
            args.adapter_state_key if init_adapter_path is not None else None
        ),
        "resume_checkpoint": str(resume_path) if resume_path is not None else None,
        "resume_sha256": resume_sha256,
        "start_branch_update": start_branch_update,
        "optimizer": {
            "type": "AdamW",
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
        },
        "regularization": {
            "data_identity_weight": float(args.data_identity_weight),
            "noise_identity_weight": float(args.noise_identity_weight),
        },
        "sampling_guidance": {
            "scale": float(config.guidance.ig.scale),
            "interval": [
                float(config.guidance.ig.t_min),
                float(config.guidance.ig.t_max),
            ],
        },
        "world_size": world_size,
        "global_batch_size": int(args.global_batch_size),
        "micro_batch_size": int(args.micro_batch_size),
        "grad_accum_steps": grad_accum_steps,
        "global_seed": int(args.global_seed),
        "dataset": str(active_data_path.expanduser().resolve()),
        "dataset_backend": dataset_backend,
        "dataset_index_map": str(args.index_map.expanduser().resolve()),
        "dataset_index_map_sha256": file_sha256(args.index_map),
        "stage2_frozen": True,
        "stage1_encoder_frozen": True,
        "stage1_decoder_frozen": True,
        "optimizer_boundary": optimizer_audit,
        "clean_autoencoder_cycle_exact_by_construction": True,
        "validation_or_reference_used_for_training": False,
        "gradient_audit_only": bool(args.gradient_audit_only),
    }
    if rank == 0:
        (experiment_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Invertible adapter config: %s", adapter_config(adapter))
        logger.info("Optimizer boundary: %s", optimizer_audit)
    require_memory_reserve(device, args.min_free_gib, "model load")

    parameters = tuple(adapter.parameters())
    flow_gradient_sum = torch.zeros(
        sum(parameter.numel() for parameter in parameters),
        device=device,
        dtype=torch.float32,
    ) if args.gradient_audit_only else None
    lpl_gradient_sum = torch.zeros_like(flow_gradient_sum) if args.gradient_audit_only else None
    audit_batches = 0
    audit_active_samples = 0
    audit_flow_loss_sum = 0.0
    audit_lpl_loss_sum = 0.0
    micro_since_boundary = 0
    first_batch_written = False
    data_index_digest = hashlib.sha256()
    optimizer.zero_grad(set_to_none=True)
    metrics_path = experiment_dir / "train_metrics.jsonl"
    update_sums = torch.zeros(8, device=device, dtype=torch.float64)
    last_time = perf_counter()

    if resume_rank_rng_states is not None:
        _restore_rank_rng(resume_rank_rng_states, rank)

    for images, labels, indices in loader:
        if args.gradient_audit_only:
            if audit_batches >= int(args.gradient_audit_microbatches):
                break
        elif branch_update >= int(args.max_updates):
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

        with autocast_context(args.precision):
            dropped_kwargs, cfg_mask = apply_cfg_dropout(
                model_kwargs,
                null_kwargs,
                float(config.conditioning.cfg_dropout_prob),
            )
            time, noise, clean_latent = transport.sample(clean_latent)
            path = make_reparameterized_path(
                adapter,
                clean_latent,
                noise,
                time,
                t_eps=float(config.transport.t_eps),
            )

            source_output = source_model(
                path.noisy_transformed,
                time,
                **dropped_kwargs,
            )
            if not (isinstance(source_output, tuple) and len(source_output) == 2):
                raise RuntimeError("RAEv2 source model must return (full, base)")
            source_full, source_base = source_output
            flow_map, _ = official_flow_loss_map(
                transport,
                (source_full, source_base),
                target_velocity=path.target_velocity,
                noisy_latent=path.noisy_transformed,
                time=time,
                base_model_coeff=float(config.internal_guidance.base_model_coeff),
            )
            flow_loss = flow_map.mean()

            gate = lpl_time_gate(time, float(args.lpl_noise_threshold))
            selected = torch.nonzero(gate, as_tuple=False).flatten()
            selected = selected[: int(args.lpl_max_samples_per_rank)]
            lpl_loss = flow_loss * 0.0
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
                if args.lpl_prediction_target == "guided":
                    predicted_transformed = internal_guidance_prediction(
                        source_full,
                        source_base,
                        time,
                        scale=float(config.guidance.ig.scale),
                        interval=(
                            float(config.guidance.ig.t_min),
                            float(config.guidance.ig.t_max),
                        ),
                    )
                else:
                    predicted_transformed = source_full
                predicted_original = inverse_prediction(
                    adapter,
                    predicted_transformed.index_select(0, selected),
                )
                predicted_features = tuple(
                    feature.float()
                    for feature in decoder_feature_pyramid(
                        rae,
                        predicted_original,
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

            data_identity = normalized_mse(path.transformed_clean, clean_latent)
            transformed_noise = adapter(noise)
            noise_identity = normalized_mse(transformed_noise, noise)
            total_loss = (
                flow_loss
                + float(args.data_identity_weight) * data_identity
                + float(args.noise_identity_weight) * noise_identity
            )
            if args.objective == "lpl":
                total_loss = total_loss + float(args.lpl_weight) * lpl_loss

        if not first_batch_written:
            with torch.no_grad():
                initial_cycle = cycle_metrics(adapter, clean_latent)
            first_batch = {
                "rank": rank,
                "indices": indices.detach().cpu().tolist(),
                "image_sha256": tensor_fingerprint(images),
                "label_sha256": tensor_fingerprint(labels),
                "latent_sha256": tensor_fingerprint(clean_latent),
                "noise_sha256": tensor_fingerprint(noise),
                "time_sha256": tensor_fingerprint(time),
                "cfg_mask_sha256": tensor_fingerprint(cfg_mask),
                "initial_cycle_max_abs": float(initial_cycle["cycle_max_abs"]),
                "initial_forward_relative_mse": float(
                    initial_cycle["forward_relative_mse"]
                ),
            }
            gathered = [None] * world_size if rank == 0 else None
            dist.gather_object(first_batch, gathered, dst=0)
            if rank == 0:
                (experiment_dir / "first_batch_audit.json").write_text(
                    json.dumps(gathered, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            first_batch_written = True

        if args.gradient_audit_only:
            flow_gradient_sum += _gradient_vector(
                flow_loss,
                parameters,
                retain_graph=True,
            )
            lpl_gradient_sum += _gradient_vector(
                lpl_loss,
                parameters,
                retain_graph=False,
            )
            audit_batches += 1
            audit_active_samples += int(selected.numel())
            audit_flow_loss_sum += float(flow_loss.detach())
            audit_lpl_loss_sum += float(lpl_loss.detach())
            continue

        (total_loss / grad_accum_steps).backward()
        micro_since_boundary += 1
        update_sums += torch.tensor(
            [
                float(flow_loss.detach()),
                float(lpl_loss.detach()),
                float(total_loss.detach()),
                float(data_identity.detach()),
                float(noise_identity.detach()),
                float(selected.numel()),
                float(path.transformed_clean.detach().float().square().mean().sqrt()),
                float(clean_latent.detach().float().square().mean().sqrt()),
            ],
            device=device,
            dtype=torch.float64,
        )
        if micro_since_boundary < grad_accum_steps:
            continue

        micro_since_boundary = 0
        all_reduce_adapter_gradients(adapter, world_size=world_size)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            adapter.parameters(),
            float(config.training.clip_grad)
            if config.training.clip_grad is not None
            else float("inf"),
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        _update_ema(adapter_ema, adapter, float(args.ema_decay))
        branch_update += 1

        reduced = update_sums.clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        reduced /= float(world_size * grad_accum_steps)
        update_sums.zero_()
        elapsed = perf_counter() - last_time
        last_time = perf_counter()
        row = {
            "branch_update": branch_update,
            "flow_loss": float(reduced[0]),
            "lpl_loss": float(reduced[1]),
            "total_loss": float(reduced[2]),
            "data_identity": float(reduced[3]),
            "noise_identity": float(reduced[4]),
            "selected_lpl_per_microbatch": float(reduced[5]),
            "transformed_clean_rms": float(reduced[6]),
            "clean_latent_rms": float(reduced[7]),
            "grad_norm": float(grad_norm.detach()),
            "seconds": elapsed,
        }
        if rank == 0:
            append_jsonl(metrics_path, row)
            logger.info(
                "update=%d flow=%.6f lpl=%.3f data_id=%.3e noise_id=%.3e "
                "grad=%.4f sec=%.1f",
                branch_update,
                row["flow_loss"],
                row["lpl_loss"],
                row["data_identity"],
                row["noise_identity"],
                row["grad_norm"],
                row["seconds"],
            )

        if branch_update % int(args.save_every) == 0 or branch_update == int(args.max_updates):
            rank_rng_states = collect_rank_rng(rank, world_size)
            if rank == 0:
                checkpoint_metadata = {
                    "source_sha256": source_sha256,
                    "source_state_key": args.source_state_key,
                    "objective": args.objective,
                    "lpl_variant": args.lpl_variant,
                    "lpl_prediction_target": args.lpl_prediction_target,
                    "lpl_weight": float(args.lpl_weight),
                    "branch_update": branch_update,
                    "data_indices_sha256": data_index_digest.hexdigest(),
                    "parent_checkpoint_sha256": resume_sha256,
                    "parent_data_indices_sha256": resume_metadata.get(
                        "data_indices_sha256"
                    ),
                }
                _save_checkpoint(
                    experiment_dir / "checkpoints" / f"update_{branch_update:07d}.pt",
                    adapter=adapter,
                    adapter_ema=adapter_ema,
                    optimizer=optimizer,
                    metadata=checkpoint_metadata,
                    rank_rng_states=rank_rng_states,
                )
            dist.barrier()

    if args.gradient_audit_only:
        dist.all_reduce(flow_gradient_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(lpl_gradient_sum, op=dist.ReduceOp.SUM)
        audit_scalars = torch.tensor(
            [
                float(audit_batches),
                float(audit_active_samples),
                audit_flow_loss_sum,
                audit_lpl_loss_sum,
            ],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(audit_scalars, op=dist.ReduceOp.SUM)
        denominator = float(world_size * max(audit_batches, 1))
        flow_gradient_sum /= denominator
        lpl_gradient_sum /= denominator
        flow_norm = float(flow_gradient_sum.norm())
        lpl_norm = float(lpl_gradient_sum.norm())
        active_samples = int(audit_scalars[1].item())
        audited_samples = float(audit_scalars[0].item())
        audit_denominator = max(audited_samples, 1.0)
        audit = {
            "microbatches_per_rank": audit_batches,
            "world_size": world_size,
            "active_lpl_samples": active_samples,
            "active_lpl_fraction": active_samples / audit_denominator,
            "mean_flow_loss": float(audit_scalars[2].item() / audit_denominator),
            "mean_zero_outside_gate_lpl": float(
                audit_scalars[3].item() / audit_denominator
            ),
            "flow_gradient_norm": flow_norm,
            "lpl_gradient_norm": lpl_norm,
            "gradient_cosine": _cosine(flow_gradient_sum, lpl_gradient_sum),
            "suggested_lpl_weight_for_0p25_gradient_ratio": (
                None if active_samples == 0 else 0.25 * flow_norm / max(lpl_norm, 1e-30)
            ),
            "suggested_lpl_weight_for_1p00_gradient_ratio": (
                None if active_samples == 0 else flow_norm / max(lpl_norm, 1e-30)
            ),
        }
        if rank == 0:
            (experiment_dir / "gradient_audit.json").write_text(
                json.dumps(audit, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info("Gradient audit: %s", audit)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
