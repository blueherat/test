"""Strict fp32 DDP training for time-dependent layerwise RAE paths."""

from __future__ import annotations

import argparse
import hashlib
import json
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
import torch.nn.functional as F
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

from experiments.rae_layerwise_path import (  # noqa: E402
    DetailSubspace,
    plan_layerwise_path,
    random_detail_basis,
    split_semantic_detail,
)
from experiments.rae_latent_cache import (  # noqa: E402
    CachedRAELatentDataset,
    load_cache_manifest,
)
from experiments.train_rae_spectral_tiny import (  # noqa: E402
    all_reduce_mean,
    append_jsonl,
    latest_branch_checkpoint,
    load_checkpoint,
    make_logger,
    save_checkpoint,
    setup_distributed,
    tensor_fingerprint,
)
from stage1 import RAE  # noqa: E402
from stage2.models import Stage2ModelProtocol  # noqa: E402
from stage2.transport import ModelType, create_transport  # noqa: E402
from utils.model_utils import instantiate_from_config  # noqa: E402
from utils.optim_utils import build_optimizer, build_scheduler  # noqa: E402
from utils.train_utils import (  # noqa: E402
    ParquetImageNetDataset,
    center_crop_arr,
    update_ema,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument(
        "--ckpt",
        type=Path,
        help="Optional source checkpoint. Omit to use a fresh seeded initialization.",
    )
    parser.add_argument("--subspaces", type=Path, required=True)
    parser.add_argument("--latent-cache", type=Path)
    parser.add_argument("--subspace-rank", type=int, default=16)
    parser.add_argument(
        "--path-mode",
        choices=("static", "annealed", "reverse"),
        required=True,
    )
    parser.add_argument("--path-power", type=float, default=2.0)
    parser.add_argument("--path-family", choices=("power", "rational"), default="power")
    parser.add_argument("--path-floor", type=float, default=0.0)
    parser.add_argument("--path-alpha", type=float, default=1.0)
    parser.add_argument("--detail-scale", type=float, default=1.0)
    parser.add_argument("--path-switch-step", type=int)
    parser.add_argument(
        "--path-mode-after-switch",
        choices=("static", "annealed", "reverse"),
    )
    parser.add_argument(
        "--ema-reset-step",
        type=int,
        help="Reset EMA to the online model immediately after this optimizer step.",
    )
    parser.add_argument(
        "--cache-order-seed",
        type=int,
        help="Deterministically permute the cached latent stream without duplicating it.",
    )
    parser.add_argument(
        "--save-steps",
        default="500,1000,2000,5000,10000",
        help="Comma-separated optimizer-step offsets to checkpoint.",
    )
    parser.add_argument(
        "--clean-component", choices=("final", "semantic"), default="final"
    )
    parser.add_argument("--random-subspace", action="store_true")
    parser.add_argument("--random-subspace-seed", type=int, default=202_607_18)
    parser.add_argument("--global-seed", type=int, required=True)
    parser.add_argument("--max-train-steps", type=int, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument(
        "--fork-full-state",
        action="store_true",
        help=(
            "Treat --ckpt as an exact training fork: restore RNG/epoch and preserve "
            "the checkpoint's branch start so cached data continues at the same offset."
        ),
    )
    parser.add_argument(
        "--isolate-loader-rng",
        action="store_true",
        help="Use a private DataLoader generator so iterator creation cannot advance training RNG.",
    )
    return parser.parse_args()


def basis_fingerprint(basis: torch.Tensor) -> str:
    return hashlib.sha256(basis.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def load_subspace(args: argparse.Namespace) -> DetailSubspace:
    payload = torch.load(args.subspaces.expanduser(), map_location="cpu", weights_only=False)
    entries = payload.get("subspaces", {})
    entry = entries.get(int(args.subspace_rank), entries.get(str(args.subspace_rank)))
    if entry is None:
        raise KeyError(
            f"rank {args.subspace_rank} is absent from {args.subspaces}; "
            f"available={list(entries)}"
        )
    basis = entry["basis"].float().contiguous()
    if args.random_subspace:
        basis = random_detail_basis(
            basis.shape[0],
            basis.shape[1],
            seed=int(args.random_subspace_seed),
        )
    return DetailSubspace(
        basis=basis,
        explained_predictable_fraction=float(entry["explained_predictable_fraction"]),
        explained_final_fraction=float(entry["explained_final_fraction"]),
        ridge_scale=float(entry["ridge_scale"]),
        token_count=int(entry["token_count"]),
    )


def configure_determinism(seed: int) -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed(int(seed))


def resolve_stage1_paths(config: Any, rae_root: Path = RAE_ROOT) -> None:
    """Resolve official RAE-relative stage-1 files independent of cwd."""

    params = config.stage_1.params
    for name in (
        "decoder_config_path",
        "pretrained_decoder_path",
        "normalization_stat_path",
    ):
        value = params.get(name)
        if value is None:
            continue
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = rae_root / path
        params[name] = str(path.resolve())


def verify_restored_optimizer_config(
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    training: dict[str, Any],
    *,
    resumed: bool = False,
) -> dict[str, Any]:
    """Reject silent optimizer overrides from a full-state source checkpoint."""

    optimizer_config = dict(training["optimizer"])
    scheduler_config = dict(training["scheduler"])
    group = optimizer.param_groups[0]
    expected = {
        "lr": float(optimizer_config["lr"]),
        "betas": tuple(float(value) for value in optimizer_config["betas"]),
        "weight_decay": float(optimizer_config["weight_decay"]),
        "scheduler_base_lr": float(scheduler_config["base_lr"]),
    }
    actual = {
        "lr": float(group["lr"]),
        "betas": tuple(float(value) for value in group["betas"]),
        "weight_decay": float(group["weight_decay"]),
        "scheduler_base_lr": float(scheduler.base_lrs[0]),
    }
    for name in ("weight_decay", "scheduler_base_lr"):
        if not math.isclose(actual[name], expected[name], rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"restored {name}={actual[name]} disagrees with config {expected[name]}"
            )
    expected_current_lr = (
        float(scheduler.get_last_lr()[0]) if resumed else expected["lr"]
    )
    if not math.isclose(
        actual["lr"], expected_current_lr, rel_tol=0.0, abs_tol=1e-12
    ):
        source = "scheduler last lr" if resumed else "config"
        raise ValueError(
            f"restored lr={actual['lr']} disagrees with {source} {expected_current_lr}"
        )
    if actual["betas"] != expected["betas"]:
        raise ValueError(
            f"restored betas={actual['betas']} disagree with config {expected['betas']}"
        )
    return actual


def exact_resume_requested(
    local_checkpoint: Path | None, fork_full_state: bool
) -> bool:
    """Return whether all stochastic and stream state must be restored."""

    return local_checkpoint is not None or bool(fork_full_state)


def active_path_mode(
    step: int,
    initial_mode: str,
    switch_step: int | None,
    mode_after_switch: str | None,
) -> str:
    """Resolve the path used to produce the next optimizer update."""

    if switch_step is None:
        if mode_after_switch is not None:
            raise ValueError("path mode after switch requires path-switch-step")
        return initial_mode
    if switch_step <= 0:
        raise ValueError("path-switch-step must be positive")
    if mode_after_switch is None:
        raise ValueError("path-switch-step requires path-mode-after-switch")
    return mode_after_switch if int(step) >= int(switch_step) else initial_mode


def parse_save_steps(value: str) -> set[int]:
    steps = {int(item.strip()) for item in value.split(",") if item.strip()}
    if any(step <= 0 for step in steps):
        raise ValueError("save steps must be positive")
    return steps


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if float(args.detail_scale) <= 0:
        raise ValueError("detail_scale must be positive")
    if not 0.0 <= float(args.path_floor) < 1.0:
        raise ValueError("path_floor must lie in [0, 1)")
    if args.path_family == "rational" and float(args.path_alpha) <= 0.0:
        raise ValueError("path_alpha must be positive")
    if args.clean_component == "semantic" and args.path_mode != "static":
        raise ValueError("semantic endpoint training requires the static path")
    active_path_mode(
        0,
        args.path_mode,
        args.path_switch_step,
        args.path_mode_after_switch,
    )
    if args.path_switch_step is not None and args.path_switch_step >= args.max_train_steps:
        raise ValueError("path-switch-step must be smaller than max-train-steps")
    if args.ema_reset_step is not None and not 0 < args.ema_reset_step < args.max_train_steps:
        raise ValueError("ema-reset-step must lie inside the training interval")
    save_offsets = parse_save_steps(args.save_steps)
    rank, world_size, device = setup_distributed()
    configure_determinism(int(args.global_seed) * world_size + rank)
    experiment_dir = args.results_dir.expanduser() / args.experiment_name
    checkpoint_dir = experiment_dir / "checkpoints"
    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    logger = make_logger(experiment_dir, rank)

    config = OmegaConf.load(args.config)
    resolve_stage1_paths(config)
    training = OmegaConf.to_container(config.training, resolve=True)
    misc = OmegaConf.to_container(config.misc, resolve=True)
    transport_params = OmegaConf.to_container(config.transport.params, resolve=True)
    global_batch = int(training["global_batch_size"])
    grad_accum = int(training["grad_accum_steps"])
    if global_batch % (world_size * grad_accum) != 0:
        raise ValueError("global batch must divide world_size * grad_accum_steps")
    micro_batch = global_batch // (world_size * grad_accum)

    cache_manifest = None
    rae: RAE | None = None
    if args.latent_cache is None:
        rae = instantiate_from_config(config.stage_1).to(device=device, dtype=torch.float32)
        rae.requires_grad_(False).eval()
    else:
        cache_manifest = load_cache_manifest(args.latent_cache)
    model: Stage2ModelProtocol = instantiate_from_config(config.stage_2).to(
        device=device, dtype=torch.float32
    )
    ema = deepcopy(model).to(device=device, dtype=torch.float32)
    ema.requires_grad_(False).eval()
    model.requires_grad_(True).train()
    ddp_model = DDP(model, device_ids=[device.index], broadcast_buffers=False)

    if args.latent_cache is None:
        transform = transforms.Compose(
            [
                transforms.Lambda(lambda image: center_crop_arr(image, int(args.image_size))),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        dataset = ParquetImageNetDataset(args.data_path, split="train", transform=transform)
    else:
        dataset = CachedRAELatentDataset(
            args.latent_cache, order_seed=args.cache_order_seed
        )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=args.latent_cache is None,
        seed=int(args.global_seed),
        drop_last=False,
    )
    loader_generator = None
    if args.isolate_loader_rng:
        loader_generator = torch.Generator()
        loader_generator.manual_seed(int(args.global_seed) * world_size + rank + 99173)
    loader = DataLoader(
        dataset,
        batch_size=micro_batch,
        sampler=sampler,
        num_workers=int(training["num_workers"]),
        pin_memory=True,
        drop_last=True,
        persistent_workers=int(training["num_workers"]) > 0,
        generator=loader_generator,
    )
    steps_per_epoch = len(loader) // grad_accum
    if steps_per_epoch < 1:
        raise RuntimeError("no optimizer steps per epoch")

    optimizer, optimizer_message = build_optimizer(ddp_model.parameters(), training)
    scheduler, scheduler_message = build_scheduler(optimizer, steps_per_epoch, training)
    time_shift = math.sqrt(
        float(misc["time_dist_shift_dim"]) / float(misc["time_dist_shift_base"])
    )
    transport = create_transport(**dict(transport_params), time_dist_shift=time_shift)
    if transport.model_type != ModelType.VELOCITY or str(transport_params["path_type"]) != "Linear":
        raise ValueError("this experiment requires linear velocity flow matching")

    subspace = load_subspace(args)
    if subspace.channels != int(misc["latent_size"][0]):
        raise ValueError("subspace channels disagree with configured latent channels")
    basis = subspace.basis.to(device=device, dtype=torch.float32)

    local_checkpoint = latest_branch_checkpoint(checkpoint_dir)
    source_checkpoint = args.ckpt.expanduser() if args.ckpt is not None else None
    load_path = local_checkpoint or source_checkpoint
    exact_resume = exact_resume_requested(local_checkpoint, args.fork_full_state)
    if load_path is None:
        global_step = 0
        branch_start_step = 0
        start_epoch = 0
    else:
        global_step, branch_start_step, start_epoch = load_checkpoint(
            load_path,
            model=ddp_model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            restore_rng=exact_resume,
        )
        if not exact_resume:
            branch_start_step = global_step
            start_epoch = 0
    restored_optimizer = verify_restored_optimizer_config(
        optimizer,
        scheduler,
        training,
        resumed=exact_resume,
    )
    if args.latent_cache is not None:
        consumed = max(int(global_step - branch_start_step), 0) * global_batch
        required = max(int(args.max_train_steps - global_step), 0) * global_batch
        available = int(cache_manifest["sample_count"]) - consumed
        if available < required:
            raise ValueError(
                f"latent cache has {available} unconsumed samples but training needs {required}"
            )
        if consumed:
            dataset = CachedRAELatentDataset(
                args.latent_cache,
                start=consumed,
                order_seed=args.cache_order_seed,
            )
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
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
                generator=loader_generator,
            )
    if args.max_train_steps <= global_step:
        logger.info("Already at step %d >= endpoint %d; nothing to do.", global_step, args.max_train_steps)
        dist.destroy_process_group()
        return

    if rank == 0:
        OmegaConf.save(config, experiment_dir / "config.yaml")
        shutil.copy2(Path(__file__), experiment_dir / Path(__file__).name)
        shutil.copy2(ROOT / "experiments/rae_layerwise_path.py", experiment_dir)
        manifest = {
            "experiment_name": args.experiment_name,
            "source_checkpoint": (
                str(source_checkpoint) if source_checkpoint is not None else None
            ),
            "loaded_checkpoint": str(load_path) if load_path is not None else None,
            "fresh_initialization": load_path is None,
            "fork_full_state": bool(args.fork_full_state),
            "exact_resume": bool(exact_resume),
            "isolate_loader_rng": bool(args.isolate_loader_rng),
            "branch_start_step": int(branch_start_step),
            "endpoint_step": int(args.max_train_steps),
            "global_seed": int(args.global_seed),
            "world_size": int(world_size),
            "global_batch_size": global_batch,
            "micro_batch_size": micro_batch,
            "grad_accum_steps": grad_accum,
            "precision": "fp32",
            "tf32": False,
            "path_mode": args.path_mode,
            "path_power": float(args.path_power),
            "path_family": args.path_family,
            "path_floor": float(args.path_floor),
            "path_switch_step": args.path_switch_step,
            "path_mode_after_switch": args.path_mode_after_switch,
            "ema_reset_step": args.ema_reset_step,
            "path_alpha": float(args.path_alpha),
            "detail_scale": float(args.detail_scale),
            "clean_component": args.clean_component,
            "subspace_path": str(args.subspaces.expanduser()),
            "subspace_rank": int(subspace.rank),
            "subspace_kind": (
                "random_energy_matched" if args.random_subspace else "middle_guided"
            ),
            "random_subspace_seed": int(args.random_subspace_seed),
            "basis_sha256": basis_fingerprint(subspace.basis),
            "endpoint_invariant": "semantic + detail equals original final RAE latent",
            "pairing_scope": "same init, data sampler, time, noise, optimizer, and endpoint",
            "restored_optimizer": restored_optimizer,
            "input_kind": "cached_final_rae_latent" if args.latent_cache else "image",
            "latent_cache": str(args.latent_cache.expanduser()) if args.latent_cache else None,
            "cache_order_seed": args.cache_order_seed,
            "latent_cache_manifest": cache_manifest,
        }
        (experiment_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        logger.info(
            "Loaded %s at step %d; branch starts at %d.",
            load_path if load_path is not None else "fresh initialization",
            global_step,
            branch_start_step,
        )
        logger.info("%s | %s", optimizer_message, scheduler_message)
        logger.info(
            "mode=%s family=%s power=%.2f floor=%.3f alpha=%.3f detail_scale=%.4f rank=%d kind=%s world=%d global_batch=%d endpoint=%d",
            args.path_mode,
            args.path_family,
            args.path_power,
            args.path_floor,
            args.path_alpha,
            args.detail_scale,
            subspace.rank,
            "random" if args.random_subspace else "middle_guided",
            world_size,
            global_batch,
            args.max_train_steps,
        )

    log_interval = int(training["log_interval"])
    clip_grad = float(training["clip_grad"])
    ema_decay = float(training["ema_decay"])
    metrics_path = experiment_dir / "metrics.jsonl"
    # loss, target energy, prediction energy, semantic energy, detail energy,
    # state energy, time, grad norm, clip hit
    window = torch.zeros(9, device=device, dtype=torch.float64)
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
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.no_grad():
                latent = images if rae is None else rae.encode(images)
                if args.clean_component == "semantic":
                    latent, _ = split_semantic_detail(latent, basis)
            time_values, noise, clean = transport.sample(latent)
            current_path_mode = active_path_mode(
                global_step,
                args.path_mode,
                args.path_switch_step,
                args.path_mode_after_switch,
            )
            plan = plan_layerwise_path(
                clean,
                noise,
                time_values,
                basis,
                mode=current_path_mode,
                power=float(args.path_power),
                family=args.path_family,
                floor=float(args.path_floor),
                alpha=float(args.path_alpha),
                detail_scale=float(args.detail_scale),
            )
            prediction = ddp_model(plan.state, time_values, y=labels)
            per_sample_loss = F.mse_loss(
                prediction,
                plan.target,
                reduction="none",
            ).flatten(1).mean(1)
            loss = per_sample_loss.mean()
            if rank == 0 and global_step == branch_start_step and window_microbatches == 0:
                fingerprint = {
                    "step": int(global_step),
                    "images_sha256": tensor_fingerprint(images),
                    "labels_sha256": tensor_fingerprint(labels),
                    "time_sha256": tensor_fingerprint(time_values),
                    "noise_sha256": tensor_fingerprint(noise),
                    "clean_sha256": tensor_fingerprint(clean),
                    "state_sha256": tensor_fingerprint(plan.state),
                    "target_sha256": tensor_fingerprint(plan.target),
                    "prediction_sha256": tensor_fingerprint(prediction),
                    "labels": labels.detach().cpu().tolist(),
                    "time": time_values.detach().cpu().tolist(),
                    "loss": float(loss.detach()),
                }
                (experiment_dir / "pair_fingerprint.json").write_text(
                    json.dumps(fingerprint, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
            (loss / grad_accum).backward()

            window[0] += float(loss.detach())
            window[1] += float(plan.target.square().mean().detach())
            window[2] += float(prediction.square().mean().detach())
            window[3] += float(plan.semantic.square().mean().detach())
            window[4] += float(plan.detail.square().mean().detach())
            window[5] += float(plan.state.square().mean().detach())
            window[6] += float(time_values.mean().detach())
            window_microbatches += 1

            if window_microbatches % grad_accum != 0:
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), clip_grad)
            optimizer.step()
            scheduler.step()
            update_ema(ema, ddp_model.module, decay=ema_decay)
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if args.ema_reset_step is not None and global_step == args.ema_reset_step:
                ema.load_state_dict(ddp_model.module.state_dict())
                if rank == 0:
                    logger.info("Reset EMA to online model at step %d.", global_step)
            window[7] += float(grad_norm)
            window[8] += float(grad_norm > clip_grad)
            window_optimizer_steps += 1

            if global_step % log_interval == 0:
                reduced = all_reduce_mean(window.clone(), world_size)
                if rank == 0:
                    micro_denominator = max(window_microbatches, 1)
                    step_denominator = max(window_optimizer_steps, 1)
                    row: dict[str, Any] = {
                        "step": int(global_step),
                        "branch_update": int(global_step - branch_start_step),
                        "loss": float(reduced[0] / micro_denominator),
                        "target_energy": float(reduced[1] / micro_denominator),
                        "prediction_energy": float(reduced[2] / micro_denominator),
                        "semantic_energy": float(reduced[3] / micro_denominator),
                        "detail_energy": float(reduced[4] / micro_denominator),
                        "state_energy": float(reduced[5] / micro_denominator),
                        "mean_time": float(reduced[6] / micro_denominator),
                        "grad_norm": float(reduced[7] / step_denominator),
                        "clip_rate": float(reduced[8] / step_denominator),
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        "path_mode": current_path_mode,
                        "elapsed_seconds": perf_counter() - training_start,
                    }
                    append_jsonl(metrics_path, row)
                    logger.info(
                        "step=%d loss=%.5f target=%.4f grad=%.3f clip=%.2f lr=%.6g",
                        global_step,
                        row["loss"],
                        row["target_energy"],
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
                    checkpoint_path = checkpoint_dir / f"step-{global_step:07d}.pt"
                    save_checkpoint(
                        checkpoint_path,
                        model=ddp_model,
                        ema=ema,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        global_step=global_step,
                        branch_start_step=branch_start_step,
                        epoch=epoch,
                    )
                    logger.info("Saved %s", checkpoint_path)
                dist.barrier()
        epoch += 1

    if rank == 0:
        logger.info(
            "Completed endpoint step %d in %.1f seconds.",
            global_step,
            perf_counter() - training_start,
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
