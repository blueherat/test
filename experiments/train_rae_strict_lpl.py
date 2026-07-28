"""Paired continuation of RAE DiT with controlled decoder-feature objectives.

The encoder and decoder are frozen. Only the latent velocity model is updated.
For the linear OT path used by RAE, the predicted clean latent is
``z_hat = z_t - t * v_theta(z_t, t)``.
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
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
RAE_ROOT = ROOT / "external" / "RAE"
RAE_SRC = RAE_ROOT / "src"
for path in (ROOT, RAE_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.rae_strict_lpl import (
    decoder_feature_pyramid,
    decoder_hidden_indices,
    flow_clean_estimate,
    lpl_time_gate,
)
from experiments.rae_lpl_detach_audit import decoder_feature_objective_per_sample
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
    parser.add_argument(
        "--ckpt",
        type=Path,
        help="Full-state source checkpoint. Omit only to train a shared source prior from scratch.",
    )
    parser.add_argument(
        "--model-ckpt",
        type=Path,
        help=(
            "Model-only source checkpoint, such as an official RAE stage2_model.pt. "
            "The model and EMA start from these weights while optimizer/scheduler state "
            "is initialized identically for both paired branches."
        ),
    )
    parser.add_argument(
        "--objective",
        choices=("flow", "raw", "detach", "full", "lpl"),
        required=True,
        help="lpl is retained as a backward-compatible alias for full.",
    )
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--max-train-steps", type=int, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--lpl-weight", type=float, default=3.0)
    parser.add_argument("--lpl-noise-threshold", type=float, default=3.0)
    parser.add_argument("--lpl-max-samples-per-rank", type=int, default=1)
    parser.add_argument(
        "--calibration-batches",
        type=int,
        default=0,
        help="Measure source-checkpoint loss scales without updating parameters.",
    )
    parser.add_argument(
        "--calibration-target-lpl-over-flow",
        type=float,
        default=0.25,
        help="Target weighted-LPL/flow ratio used for the reported transfer weight.",
    )
    parser.add_argument(
        "--calibration-mode",
        choices=("mean_contribution", "variance"),
        default="mean_contribution",
        help=(
            "Select the reported calibration. mean_contribution matches the main "
            "paper's roughly one-fifth total-loss contribution; variance reproduces "
            "the supplement A.4 fair-comparison rule."
        ),
    )
    parser.add_argument(
        "--calibration-target-variance-ratio",
        type=float,
        default=0.1,
        help="Target Var(weight * LPL) / Var(flow) for the A.4 sensitivity rule.",
    )
    parser.add_argument(
        "--skip-checkpoint-save",
        action="store_true",
        help="Run training without writing checkpoints; intended only for memory smoke tests.",
    )
    parser.add_argument(
        "--allow-nonexact-resume",
        action="store_true",
        help=(
            "Allow continuation from a branch-local checkpoint even though the legacy "
            "checkpoint does not store the dataloader cursor or per-rank RNG states."
        ),
    )
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, torch.device]:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    return dist.get_rank(), dist.get_world_size(), device


def make_logger(experiment_dir: Path, rank: int) -> logging.Logger:
    logger = logging.getLogger(f"rae-strict-lpl-{rank}")
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


def file_sha256(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def text_sequence_sha256(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


class IndexedDataset(Dataset):
    """Attach the source row index so paired data streams can be audited."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        image, label = self.dataset[index]
        return image, label, int(index)


def moments_from_totals(count: float, total: float, total_sq: float) -> dict[str, float]:
    if count <= 0:
        return {"count": 0, "mean": float("nan"), "variance": float("nan")}
    mean = total / count
    variance = max(total_sq / count - mean * mean, 0.0)
    return {"count": int(count), "mean": mean, "variance": variance}


def variance_matched_weight(
    flow_variance: float,
    feature_variance: float,
    *,
    target_ratio: float,
) -> float:
    if target_ratio <= 0:
        raise ValueError("target variance ratio must be positive")
    if flow_variance < 0 or feature_variance < 0:
        raise ValueError("loss variances must be non-negative")
    if feature_variance == 0:
        return float("nan")
    return math.sqrt(target_ratio * flow_variance / feature_variance)


def assert_optimizer_boundary(
    optimizer: torch.optim.Optimizer,
    *,
    trainable_model: torch.nn.Module,
    frozen_modules: tuple[torch.nn.Module, ...],
) -> None:
    optimized = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    expected = {
        id(parameter)
        for parameter in trainable_model.parameters()
        if parameter.requires_grad
    }
    if optimized != expected:
        raise RuntimeError(
            "optimizer parameters do not exactly match the trainable stage-2 model"
        )
    frozen = {
        id(parameter)
        for module in frozen_modules
        for parameter in module.parameters()
    }
    if optimized & frozen:
        raise RuntimeError("optimizer contains parameters from a frozen module")


def assert_frozen_modules_have_no_grad(
    modules: tuple[torch.nn.Module, ...],
) -> None:
    offenders = [
        name
        for module_index, module in enumerate(modules)
        for name, parameter in module.named_parameters()
        if parameter.grad is not None
        for name in (f"module_{module_index}.{name}",)
    ]
    if offenders:
        raise RuntimeError(
            "frozen parameters received gradients: " + ", ".join(offenders[:8])
        )


def module_state_versions(module: torch.nn.Module) -> tuple[tuple[str, int], ...]:
    values = list(module.named_parameters()) + list(module.named_buffers())
    return tuple((name, value._version) for name, value in values)


def update_stream_fingerprint(
    digest: Any,
    *,
    indices: torch.Tensor,
    labels: torch.Tensor,
    images: torch.Tensor,
    time: torch.Tensor,
    noise: torch.Tensor,
) -> None:
    for value in (
        indices,
        labels,
        images[..., ::32, ::32],
        time,
        noise[:, : min(noise.shape[1], 8), ::4, ::4],
    ):
        array = value.detach().to(device="cpu").contiguous().numpy()
        digest.update(array.tobytes())


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.stem.split("-")[-1])
    except ValueError:
        return -1


def latest_branch_checkpoint(checkpoint_dir: Path) -> Path | None:
    candidates = sorted(checkpoint_dir.glob("step-*.pt"), key=checkpoint_step)
    return candidates[-1] if candidates else None


def validate_resume_policy(
    local_checkpoint: Path | None,
    *,
    endpoint_step: int,
    allow_nonexact_resume: bool,
) -> None:
    """Prevent a legacy checkpoint resume from being mistaken for exact pairing."""

    if local_checkpoint is None or checkpoint_step(local_checkpoint) >= int(endpoint_step):
        return
    if not allow_nonexact_resume:
        raise RuntimeError(
            f"{local_checkpoint} is an incomplete branch checkpoint. Exact continuation "
            "is unavailable because legacy checkpoints omit the dataloader cursor and "
            "per-rank RNG states. Restart in an empty experiment directory, or pass "
            "--allow-nonexact-resume only for a deliberately non-paired run."
        )


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
    if restore_rng:
        for key in ("rng_cpu", "rng_cuda", "branch_start_step", "epoch"):
            if key not in state:
                raise KeyError(f"checkpoint {path} lacks exact-resume state {key!r}")
        torch.set_rng_state(state["rng_cpu"])
        torch.cuda.set_rng_state_all(state["rng_cuda"])
    step = int(state["step"])
    return step, int(state.get("branch_start_step", step)), int(state.get("epoch", 0))


def model_state_from_checkpoint(state: Any) -> dict[str, torch.Tensor]:
    """Extract a model state dict from official and common wrapped checkpoints."""
    if not isinstance(state, dict):
        raise TypeError(f"model checkpoint must be a mapping, got {type(state)!r}")
    for key in ("ema", "model", "state_dict"):
        candidate = state.get(key)
        if isinstance(candidate, dict) and candidate:
            state = candidate
            break
    if not state or not all(isinstance(key, str) for key in state):
        raise ValueError("model checkpoint does not contain a non-empty state dict")
    if all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    if not all(torch.is_tensor(value) for value in state.values()):
        raise ValueError("model state dict contains non-tensor values")
    return state


def load_model_checkpoint(
    path: Path,
    *,
    model: DDP,
    ema: torch.nn.Module,
) -> None:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model_state = model_state_from_checkpoint(state)
    model.module.load_state_dict(model_state, strict=True)
    ema.load_state_dict(model_state, strict=True)


def resolve_rae_paths(config: Any) -> None:
    params = config.stage_1.params
    for name in (
        "decoder_config_path",
        "pretrained_decoder_path",
        "normalization_stat_path",
    ):
        value = params.get(name)
        if value is not None and not Path(str(value)).is_absolute():
            params[name] = str(RAE_ROOT / str(value))


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.lpl_weight < 0:
        raise ValueError("lpl-weight must be non-negative")
    if args.lpl_max_samples_per_rank < 1:
        raise ValueError("lpl-max-samples-per-rank must be positive")
    if args.calibration_batches < 0:
        raise ValueError("calibration-batches must be non-negative")
    if args.calibration_target_lpl_over_flow <= 0:
        raise ValueError("calibration-target-lpl-over-flow must be positive")
    if args.calibration_target_variance_ratio <= 0:
        raise ValueError("calibration-target-variance-ratio must be positive")
    if args.ckpt is not None and args.model_ckpt is not None:
        raise ValueError("--ckpt and --model-ckpt are mutually exclusive")
    if args.calibration_batches > 0 and args.objective == "flow":
        raise ValueError("feature-loss calibration requires a non-flow objective")

    rank, world_size, device = setup_distributed()
    experiment_dir = args.results_dir.expanduser().resolve() / args.experiment_name
    checkpoint_dir = experiment_dir / "checkpoints"
    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    logger = make_logger(experiment_dir, rank)
    torch.cuda.reset_peak_memory_stats(device)

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

    config = OmegaConf.load(args.config.expanduser().resolve())
    resolve_rae_paths(config)
    training = OmegaConf.to_container(config.training, resolve=True)
    misc = OmegaConf.to_container(config.misc, resolve=True)
    transport_params = OmegaConf.to_container(config.transport.params, resolve=True)
    lpl_config = dict(training["strict_lpl"])
    global_batch = int(training["global_batch_size"])
    grad_accum = int(training["grad_accum_steps"])
    if global_batch % (world_size * grad_accum) != 0:
        raise ValueError("global batch must divide world_size * grad_accum_steps")
    micro_batch = global_batch // (world_size * grad_accum)

    rae: RAE = instantiate_from_config(config.stage_1).to(device=device, dtype=torch.float32)
    rae.requires_grad_(False).eval()
    if float(getattr(rae, "noise_tau", 0.0)) != 0.0:
        raise ValueError("strict deterministic-decoder LPL requires noise_tau=0")
    model: Stage2ModelProtocol = instantiate_from_config(config.stage_2).to(
        device=device, dtype=torch.float32
    )
    ema = deepcopy(model).to(device=device, dtype=torch.float32)
    ema.requires_grad_(False).eval()
    model.requires_grad_(True).train()
    ddp_model = DDP(
        model,
        device_ids=[device.index],
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
    )

    transform = transforms.Compose(
        [
            transforms.Lambda(lambda image: center_crop_arr(image, int(args.image_size))),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    data_path = args.data_path.expanduser().resolve()
    base_dataset = ParquetImageNetDataset(data_path, split="train", transform=transform)
    non_train_shards = [
        path.name for path in base_dataset.files if not path.name.startswith("train-")
    ]
    if non_train_shards:
        raise RuntimeError(f"non-train shards entered the trainer: {non_train_shards[:4]}")
    dataset = IndexedDataset(base_dataset)
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
    assert_optimizer_boundary(
        optimizer,
        trainable_model=ddp_model.module,
        frozen_modules=(rae, ema),
    )
    frozen_rae_versions = module_state_versions(rae)
    time_shift = math.sqrt(float(misc["time_dist_shift_dim"]) / float(misc["time_dist_shift_base"]))
    transport = create_transport(**dict(transport_params), time_dist_shift=time_shift)
    if transport.model_type != ModelType.VELOCITY or str(transport_params["path_type"]) != "Linear":
        raise ValueError("strict RAE LPL implementation requires linear velocity flow matching")

    decoder_depth = len(rae.decoder.decoder_layers)
    layer_indices = decoder_hidden_indices(
        decoder_depth, tuple(float(value) for value in lpl_config["layer_fractions"])
    )
    layer_weights = [1.0] * len(layer_indices)

    local_checkpoint = latest_branch_checkpoint(checkpoint_dir)
    validate_resume_policy(
        local_checkpoint,
        endpoint_step=int(args.max_train_steps),
        allow_nonexact_resume=bool(args.allow_nonexact_resume),
    )
    source_checkpoint = args.ckpt.expanduser().resolve() if args.ckpt is not None else None
    source_model_checkpoint = (
        args.model_ckpt.expanduser().resolve() if args.model_ckpt is not None else None
    )
    load_path = local_checkpoint or source_checkpoint
    if local_checkpoint is not None or source_checkpoint is not None:
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
    elif source_model_checkpoint is not None:
        load_model_checkpoint(source_model_checkpoint, model=ddp_model, ema=ema)
        global_step = 0
        branch_start_step = 0
        start_epoch = 0
    else:
        global_step = 0
        branch_start_step = 0
        start_epoch = 0

    source_model_versions = module_state_versions(ddp_model.module)
    frozen_ema_versions = module_state_versions(ema)
    if args.calibration_batches > 0:
        if source_checkpoint is None and source_model_checkpoint is None:
            raise ValueError("LPL calibration requires --ckpt or --model-ckpt")
        model.eval()
        sampler.set_epoch(0)
        # Per-sample moments: flow count/sum/sumsq, gated LPL count/sum/sumsq,
        # conditional LPL count/sum/sumsq, eligible, examples, batches.
        calibration = torch.zeros(12, device=device, dtype=torch.float64)
        with torch.no_grad():
            for batch_index, (images, labels, _indices) in enumerate(loader):
                if batch_index >= int(args.calibration_batches):
                    break
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                clean = rae.encode(images)
                time, noise, clean = transport.sample(clean)
                time, noisy, target_velocity = transport.path_sampler.plan(time, noise, clean)
                prediction = ddp_model(noisy, time, y=labels)
                flow_values = (
                    (prediction - target_velocity).square().flatten(1).mean(1)
                )
                gate = lpl_time_gate(time, float(args.lpl_noise_threshold))
                eligible_indices = torch.nonzero(gate, as_tuple=False).flatten()
                gated_lpl_values = prediction.new_zeros(images.shape[0])
                if eligible_indices.numel() > 0:
                    predicted_clean = flow_clean_estimate(noisy, prediction, time)
                    for selected_indices in eligible_indices.split(
                        int(args.lpl_max_samples_per_rank)
                    ):
                        target_features = decoder_feature_pyramid(
                            rae,
                            clean.index_select(0, selected_indices),
                            layer_indices=layer_indices,
                        )
                        predicted_features = decoder_feature_pyramid(
                            rae,
                            predicted_clean.index_select(0, selected_indices),
                            layer_indices=layer_indices,
                        )
                        lpl_values, _ = decoder_feature_objective_per_sample(
                            args.objective,
                            target_features,
                            predicted_features,
                            layer_weights=layer_weights,
                            outlier_quantile=float(lpl_config["outlier_quantile"]),
                            outlier_opening=int(lpl_config["outlier_opening"]),
                            outlier_closing=int(lpl_config["outlier_closing"]),
                            eps=float(lpl_config["normalization_eps"]),
                        )
                        gated_lpl_values.index_copy_(
                            0, selected_indices, lpl_values
                        )
                conditional_lpl_values = gated_lpl_values.index_select(
                    0, eligible_indices
                )
                calibration[0] += float(flow_values.numel())
                calibration[1] += float(flow_values.sum())
                calibration[2] += float(flow_values.double().square().sum())
                calibration[3] += float(gated_lpl_values.numel())
                calibration[4] += float(gated_lpl_values.sum())
                calibration[5] += float(gated_lpl_values.double().square().sum())
                calibration[6] += float(conditional_lpl_values.numel())
                calibration[7] += float(conditional_lpl_values.sum())
                calibration[8] += float(
                    conditional_lpl_values.double().square().sum()
                )
                calibration[9] += float(eligible_indices.numel())
                calibration[10] += float(images.shape[0])
                calibration[11] += 1.0
        dist.all_reduce(calibration, op=dist.ReduceOp.SUM)
        if rank == 0:
            flow_moments = moments_from_totals(
                *[float(value) for value in calibration[0:3]]
            )
            gated_lpl_moments = moments_from_totals(
                *[float(value) for value in calibration[3:6]]
            )
            conditional_lpl_moments = moments_from_totals(
                *[float(value) for value in calibration[6:9]]
            )
            mean_weight = (
                float(args.calibration_target_lpl_over_flow)
                * flow_moments["mean"]
                / gated_lpl_moments["mean"]
                if gated_lpl_moments["mean"] > 0
                else float("nan")
            )
            gated_variance_weight = variance_matched_weight(
                flow_moments["variance"],
                gated_lpl_moments["variance"],
                target_ratio=float(args.calibration_target_variance_ratio),
            )
            conditional_variance_weight = variance_matched_weight(
                flow_moments["variance"],
                conditional_lpl_moments["variance"],
                target_ratio=float(args.calibration_target_variance_ratio),
            )
            recommended_weight = (
                mean_weight
                if args.calibration_mode == "mean_contribution"
                else gated_variance_weight
            )
            result = {
                "source_checkpoint": str(source_checkpoint or source_model_checkpoint),
                "source_checkpoint_sha256": file_sha256(
                    source_checkpoint or source_model_checkpoint
                ),
                "objective": args.objective,
                "source_checkpoint_type": (
                    "full_state" if source_checkpoint is not None else "model_only"
                ),
                "global_seed": int(args.global_seed),
                "world_size": world_size,
                "config_path": str(args.config.expanduser().resolve()),
                "config_sha256": file_sha256(args.config.expanduser().resolve()),
                "data_path": str(data_path),
                "dataset_split": "train",
                "dataset_examples": len(dataset),
                "dataset_parquet_shards": len(base_dataset.files),
                "dataset_shard_names_sha256": text_sequence_sha256(
                    [path.name for path in base_dataset.files]
                ),
                "evaluation_reference_loaded_by_trainer": False,
                "rae_decoder_path": str(
                    Path(str(config.stage_1.params.pretrained_decoder_path))
                ),
                "rae_decoder_sha256": file_sha256(
                    Path(str(config.stage_1.params.pretrained_decoder_path))
                ),
                "rae_statistics_path": str(
                    Path(str(config.stage_1.params.normalization_stat_path))
                ),
                "rae_statistics_sha256": file_sha256(
                    Path(str(config.stage_1.params.normalization_stat_path))
                ),
                "encoder_frozen": True,
                "decoder_frozen": True,
                "optimizer_exactly_stage2_parameters": True,
                "local_batches": int(args.calibration_batches),
                "global_examples": int(calibration[10].item()),
                "flow_per_sample": flow_moments,
                "gated_lpl_per_sample": gated_lpl_moments,
                "conditional_lpl_per_sample": conditional_lpl_moments,
                "mean_flow_loss": flow_moments["mean"],
                "mean_raw_lpl_batch_contribution": gated_lpl_moments["mean"],
                "mean_unweighted_feature_batch_contribution": (
                    gated_lpl_moments["mean"]
                ),
                "eligible_rate": float(
                    calibration[9] / max(float(calibration[10]), 1.0)
                ),
                "selected_examples": int(calibration[9].item()),
                "target_weighted_lpl_over_flow": float(
                    args.calibration_target_lpl_over_flow
                ),
                "target_lpl_to_flow_variance_ratio": float(
                    args.calibration_target_variance_ratio
                ),
                "mean_contribution_weight": mean_weight,
                "gated_variance_weight": gated_variance_weight,
                "conditional_variance_weight_sensitivity": (
                    conditional_variance_weight
                ),
                "calibration_mode": args.calibration_mode,
                "recommended_lpl_weight": recommended_weight,
                "variance_definition": (
                    "population variance across samples; primary LPL variable is "
                    "zero outside the t/(1-t) gate"
                ),
                "selection_rule": (
                    "fixed once at the source checkpoint before paired continuation"
                ),
            }
            output = experiment_dir / "lpl_calibration.json"
            output.write_text(
                json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            logger.info("Calibration: %s", json.dumps(result, ensure_ascii=False))
        if module_state_versions(rae) != frozen_rae_versions:
            raise RuntimeError(
                "frozen RAE parameters or buffers changed during calibration"
            )
        if module_state_versions(ddp_model.module) != source_model_versions:
            raise RuntimeError("stage-2 model changed during no-update calibration")
        if module_state_versions(ema) != frozen_ema_versions:
            raise RuntimeError("EMA changed during no-update calibration")
        dist.barrier()
        dist.destroy_process_group()
        return

    if args.max_train_steps <= global_step:
        logger.info("Already at step %d >= endpoint %d; nothing to do.", global_step, args.max_train_steps)
        dist.destroy_process_group()
        return

    if rank == 0:
        OmegaConf.save(config, experiment_dir / "config.yaml")
        shutil.copy2(Path(__file__), experiment_dir / Path(__file__).name)
        shutil.copy2(ROOT / "experiments/rae_strict_lpl.py", experiment_dir)
        shutil.copy2(
            ROOT / "experiments/rae_lpl_detach_audit.py",
            experiment_dir,
        )
        normalization = {
            "flow": "none",
            "raw": "none; prediction-derived outlier mask only",
            "detach": "prediction variance with denominator detached",
            "full": "differentiable prediction variance",
            "lpl": "differentiable prediction variance",
        }[args.objective]
        resolved_source = source_checkpoint or source_model_checkpoint
        decoder_path = Path(str(config.stage_1.params.pretrained_decoder_path))
        statistics_path = Path(str(config.stage_1.params.normalization_stat_path))
        config_path = args.config.expanduser().resolve()
        manifest = {
            "experiment_name": args.experiment_name,
            "objective": args.objective,
            "source_checkpoint": (
                str(resolved_source) if resolved_source is not None else None
            ),
            "source_checkpoint_sha256": (
                file_sha256(resolved_source) if resolved_source is not None else None
            ),
            "source_checkpoint_type": (
                "full_state"
                if source_checkpoint is not None
                else "model_only"
                if source_model_checkpoint is not None
                else None
            ),
            "optimizer_state_at_branch_start": (
                "restored"
                if source_checkpoint is not None
                else "fresh_shared"
                if source_model_checkpoint is not None
                else "fresh_random_model"
            ),
            "loaded_checkpoint": (
                str(local_checkpoint or source_checkpoint or source_model_checkpoint)
                if local_checkpoint is not None
                or source_checkpoint is not None
                or source_model_checkpoint is not None
                else None
            ),
            "fresh_initialization": (
                source_checkpoint is None and source_model_checkpoint is None
            ),
            "branch_start_step": branch_start_step,
            "endpoint_step": int(args.max_train_steps),
            "global_seed": int(args.global_seed),
            "world_size": world_size,
            "global_batch_size": global_batch,
            "micro_batch_size": micro_batch,
            "grad_accum_steps": grad_accum,
            "precision": "fp32",
            "tf32": False,
            "config_path": str(config_path),
            "config_sha256": file_sha256(config_path),
            "data_path": str(data_path),
            "dataset_split": "train",
            "dataset_examples": len(dataset),
            "dataset_parquet_shards": len(base_dataset.files),
            "dataset_shard_names_sha256": text_sequence_sha256(
                [path.name for path in base_dataset.files]
            ),
            "dataset_files_asserted_train_only": True,
            "evaluation_reference_loaded_by_trainer": False,
            "encoder_frozen": True,
            "decoder_frozen": True,
            "frozen_boundary_runtime_assertions": True,
            "optimizer_exactly_stage2_parameters": True,
            "decoder_deterministic": True,
            "rae_decoder_path": str(decoder_path),
            "rae_decoder_sha256": file_sha256(decoder_path),
            "rae_statistics_path": str(statistics_path),
            "rae_statistics_sha256": file_sha256(statistics_path),
            "clean_estimate": "z_hat = z_t - t * velocity",
            "time_gate": "t / (1 - t) <= threshold",
            "lpl_weight": float(args.lpl_weight),
            "lpl_noise_to_signal_threshold": float(args.lpl_noise_threshold),
            "lpl_max_samples_per_rank": int(args.lpl_max_samples_per_rank),
            "decoder_depth": decoder_depth,
            "decoder_hidden_indices": list(layer_indices),
            "decoder_layer_weights": layer_weights,
            "outlier_quantile": float(lpl_config["outlier_quantile"]),
            "outlier_opening": int(lpl_config["outlier_opening"]),
            "outlier_closing": int(lpl_config["outlier_closing"]),
            "feature_normalization": normalization,
            "cross_normalization": normalization,
            "method_identity": (
                "RAE-adapted LPL: original cross-normalized decoder-feature "
                "objective applied to frozen ViT decoder hidden states"
            ),
            "paper_code_available": False,
            "pairing_scope": "fresh deterministic stream from one shared full-state checkpoint",
            "resumed_from_branch_checkpoint": local_checkpoint is not None,
            "resume_is_exact": local_checkpoint is None,
        }
        if source_model_checkpoint is not None:
            manifest["pairing_scope"] = (
                "fresh deterministic stream from shared official model weights "
                "and identically initialized optimizer/scheduler state"
            )
        if local_checkpoint is not None:
            manifest["pairing_scope"] = (
                "explicit nonexact continuation: legacy checkpoint lacks the "
                "dataloader cursor and per-rank RNG states"
            )
        (experiment_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        logger.info(
            "Loaded %s at step %d; branch starts at %d.",
            load_path if load_path is not None else "fresh initialization",
            global_step,
            branch_start_step,
        )
        logger.info("%s | %s", optimizer_message, scheduler_message)
        logger.info(
            "objective=%s world=%d micro=%d accum=%d global_batch=%d endpoint=%d",
            args.objective,
            world_size,
            micro_batch,
            grad_accum,
            global_batch,
            args.max_train_steps,
        )
        logger.info(
            "LPL weight=%.3f threshold=%.3f layers=%s",
            args.lpl_weight,
            args.lpl_noise_threshold,
            layer_indices,
        )

    log_interval = int(training["log_interval"])
    clip_grad = float(training["clip_grad"])
    ema_decay = float(training["ema_decay"])
    save_offsets = {
        int(value)
        for value in training.get(
            "checkpoint_offsets", (10, 50, 100, 250, 500, 1000, 2000, 5000)
        )
    }
    metrics_path = experiment_dir / "metrics.jsonl"
    # total, flow, batch-LPL, conditional-LPL sum, selected, eligible, examples,
    # mask keep sum, grad norm, clip hit, optimizer steps, microbatches
    window = torch.zeros(12, device=device, dtype=torch.float64)
    optimizer.zero_grad(set_to_none=True)
    training_start = perf_counter()
    stream_digest = hashlib.sha256()
    stream_microbatches = 0

    epoch = start_epoch
    while global_step < args.max_train_steps:
        sampler.set_epoch(epoch)
        for images, labels, indices in loader:
            if global_step >= args.max_train_steps:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.no_grad():
                clean = rae.encode(images)
            time, noise, clean = transport.sample(clean)
            time, noisy, target_velocity = transport.path_sampler.plan(time, noise, clean)
            update_stream_fingerprint(
                stream_digest,
                indices=indices,
                labels=labels,
                images=images,
                time=time,
                noise=noise,
            )
            stream_microbatches += 1
            prediction = ddp_model(noisy, time, y=labels)
            flow_per_sample = (prediction - target_velocity).square().flatten(1).mean(1)
            flow_loss = flow_per_sample.mean()
            total_loss = flow_loss

            gate = lpl_time_gate(time, float(args.lpl_noise_threshold))
            eligible_indices = torch.nonzero(gate, as_tuple=False).flatten()
            selected_indices = eligible_indices[: int(args.lpl_max_samples_per_rank)]
            lpl_batch_contribution = prediction.new_zeros(())
            conditional_lpl_sum = prediction.new_zeros(())
            mask_keep_sum = prediction.new_zeros(())
            if args.objective != "flow" and selected_indices.numel() > 0:
                predicted_clean = flow_clean_estimate(noisy, prediction, time)
                with torch.no_grad():
                    target_features = decoder_feature_pyramid(
                        rae, clean.index_select(0, selected_indices), layer_indices=layer_indices
                    )
                predicted_features = decoder_feature_pyramid(
                    rae,
                    predicted_clean.index_select(0, selected_indices),
                    layer_indices=layer_indices,
                )
                lpl_values, lpl_details = decoder_feature_objective_per_sample(
                    args.objective,
                    target_features,
                    predicted_features,
                    layer_weights=layer_weights,
                    outlier_quantile=float(lpl_config["outlier_quantile"]),
                    outlier_opening=int(lpl_config["outlier_opening"]),
                    outlier_closing=int(lpl_config["outlier_closing"]),
                    eps=float(lpl_config["normalization_eps"]),
                )
                conditional_lpl_sum = lpl_values.sum()
                # Preserve E[1_gate * LPL] if an eligible local batch is capped.
                cap_correction = eligible_indices.numel() / selected_indices.numel()
                lpl_batch_contribution = (
                    conditional_lpl_sum * cap_correction / float(images.shape[0])
                )
                mask_keep_sum = lpl_details["mask_keep_fraction"].mean(1).sum()
                total_loss = total_loss + float(args.lpl_weight) * lpl_batch_contribution

            if rank == 0 and global_step == branch_start_step and window[11] == 0:
                fingerprint = {
                    "step": global_step,
                    "objective": args.objective,
                    "indices_sha256": tensor_fingerprint(indices),
                    "images_sha256": tensor_fingerprint(images),
                    "labels_sha256": tensor_fingerprint(labels),
                    "time_sha256": tensor_fingerprint(time),
                    "noise_sha256": tensor_fingerprint(noise),
                    "noisy_latent_sha256": tensor_fingerprint(noisy),
                    "target_velocity_sha256": tensor_fingerprint(target_velocity),
                    "prediction_sha256": tensor_fingerprint(prediction),
                    "labels": labels.detach().cpu().tolist(),
                    "time": time.detach().cpu().tolist(),
                    "flow_loss": float(flow_loss.detach()),
                    "eligible": gate.detach().cpu().tolist(),
                    "lpl_batch_contribution": float(lpl_batch_contribution.detach()),
                }
                (experiment_dir / "pair_fingerprint.json").write_text(
                    json.dumps(fingerprint, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )

            (total_loss / grad_accum).backward()
            if global_step == branch_start_step and int(window[11].item()) == 0:
                assert_frozen_modules_have_no_grad((rae, ema))
            window[0] += float(total_loss.detach())
            window[1] += float(flow_loss.detach())
            window[2] += float(lpl_batch_contribution.detach())
            window[3] += float(conditional_lpl_sum.detach())
            window[4] += float(selected_indices.numel())
            window[5] += float(eligible_indices.numel())
            window[6] += float(images.shape[0])
            window[7] += float(mask_keep_sum.detach())
            window[11] += 1.0

            if int(window[11].item()) % grad_accum != 0:
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), clip_grad)
            optimizer.step()
            scheduler.step()
            update_ema(ema, ddp_model.module, decay=ema_decay)
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            window[8] += float(grad_norm)
            window[9] += float(grad_norm > clip_grad)
            window[10] += 1.0

            if global_step % log_interval == 0:
                reduced = window.clone()
                dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
                if rank == 0:
                    batch_denominator = max(float(reduced[11]), 1.0)
                    selected_denominator = max(float(reduced[4]), 1.0)
                    step_denominator = max(float(reduced[10]), 1.0)
                    row = {
                        "step": global_step,
                        "branch_update": global_step - branch_start_step,
                        "total_loss": float(reduced[0] / batch_denominator),
                        "flow_loss": float(reduced[1] / batch_denominator),
                        "lpl_batch_contribution": float(reduced[2] / batch_denominator),
                        "lpl_conditional": float(reduced[3] / selected_denominator),
                        "eligible_rate": float(reduced[5] / max(float(reduced[6]), 1.0)),
                        "selected_examples": int(reduced[4].item()),
                        "mask_keep_fraction": float(reduced[7] / selected_denominator),
                        "grad_norm": float(reduced[8] / step_denominator),
                        "clip_rate": float(reduced[9] / step_denominator),
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        "elapsed_seconds": perf_counter() - training_start,
                    }
                    append_jsonl(metrics_path, row)
                    logger.info(
                        "step=%d update=%d total=%.5f flow=%.5f lpl=%.5f eligible=%.3f grad=%.3f",
                        global_step,
                        global_step - branch_start_step,
                        row["total_loss"],
                        row["flow_loss"],
                        row["lpl_batch_contribution"],
                        row["eligible_rate"],
                        row["grad_norm"],
                    )
                window.zero_()

            branch_update = global_step - branch_start_step
            should_save = branch_update in save_offsets or global_step == args.max_train_steps
            if should_save and not args.skip_checkpoint_save:
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

    if module_state_versions(rae) != frozen_rae_versions:
        raise RuntimeError("frozen RAE parameters or buffers changed during training")
    free_memory, total_memory = torch.cuda.mem_get_info(device)
    stream_audit = {
        "rank": rank,
        "objective": args.objective,
        "global_seed": int(args.global_seed),
        "microbatches": stream_microbatches,
        "sha256": stream_digest.hexdigest(),
        "fields": [
            "dataset_index",
            "label",
            "augmented_image_stride32",
            "time",
            "noise_channels8_stride4",
        ],
        "gpu_memory": {
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / (1024**2),
            "physical_free_at_end_mib": free_memory / (1024**2),
            "physical_total_mib": total_memory / (1024**2),
        },
    }
    (experiment_dir / f"stream_audit_rank{rank}.json").write_text(
        json.dumps(stream_audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if rank == 0:
        logger.info("Completed endpoint step %d in %.1f seconds.", global_step, perf_counter() - training_start)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
