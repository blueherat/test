#!/usr/bin/env python3
"""Paired RAEv2 sampling with the normalized semigroup HJB value correction."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torchvision.utils import save_image


ROOT = Path(__file__).resolve().parents[1]
RAEV2_SRC = ROOT / "external" / "RAEv2" / "src"
for path in (RAEV2_SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.raev2_pfr_retiming import clean_to_velocity  # noqa: E402
from experiments.raev2_semigroup_value import (  # noqa: E402
    RAEv2NormalizedOUValue,
    semigroup_value_guided_clean,
    state_to_ou,
)
from experiments.raev2_stage1_compat import (  # noqa: E402
    install_raev2_decoder_config_compat,
)
from experiments.raev2_training_core import file_sha256  # noqa: E402
from experiments.sample_raev2_depth_condition_guidance import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    evaluate_four_corners,
)
from experiments.sample_raev2_pfr_retiming import (  # noqa: E402
    generator_sha256,
    load_config,
    shifted_time_grid,
    tensor_sha256,
)
from utils.model_utils import instantiate_from_config  # noqa: E402


PROTOCOL = "raev2_normalized_ou_hjb_value_sampling_v1"
VALUE_PROTOCOL = "raev2_normalized_ou_hjb_value_v1"


def sample_rms(value: torch.Tensor) -> torch.Tensor:
    return value.float().flatten(1).square().mean(1).sqrt()


def load_value(
    path: Path,
    *,
    weights: str,
    latent_channels: int,
    num_classes: int,
    device: torch.device,
) -> tuple[RAEv2NormalizedOUValue, dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("protocol") != VALUE_PROTOCOL:
        raise ValueError("unexpected semigroup-value checkpoint protocol")
    request = payload["request"]
    value = RAEv2NormalizedOUValue(
        latent_channels,
        num_classes,
        width=int(request["width"]),
        depth=int(request["depth"]),
        switch_time=float(request["switch_time"]),
    )
    state_key = "value_ema" if weights == "ema" else "value"
    value.load_state_dict(payload[state_key], strict=True)
    value.to(device).eval().requires_grad_(False)
    metadata = {
        "path": str(path),
        "sha256": file_sha256(path),
        "step": int(payload["step"]),
        "weights": weights,
        "request": request,
    }
    return value, metadata


def sample_local_shard(
    *,
    model: torch.nn.Module,
    decoder: torch.nn.Module,
    value: RAEv2NormalizedOUValue,
    beta: float,
    switch_time: float,
    config,
    time_grid: torch.Tensor,
    global_ids: np.ndarray,
    per_rank_batch: int,
    sampling_seed: int,
    precision: str,
    output_dir: Path,
    rank: int,
    world_size: int,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device=time_grid.device)
    generator.manual_seed(int(sampling_seed) * world_size + rank)
    initial_rng = generator_sha256(generator)
    images_local: list[np.ndarray] = []
    preview = None
    first_noise_sha256 = None
    first_label_sha256 = None
    correction_rms_sum = 0.0
    ordinary_rms_sum = 0.0
    correction_ratio_sum = 0.0
    correction_count = 0
    source_calls = 0
    value_gradient_calls = 0
    extrapolated_time_calls = 0
    started = time.perf_counter()
    t_floor = float(config.transport.t_eps)
    null_label = int(config.misc.num_classes)
    maximum_training_time = float(value_training_maximum(value))
    source_cast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if precision == "bf16"
        else nullcontext()
    )

    for start in range(0, len(global_ids), per_rank_batch):
        ids = global_ids[start : start + per_rank_batch]
        batch_size = len(ids)
        state = torch.randn(
            batch_size,
            *config.misc.latent_size,
            generator=generator,
            device=time_grid.device,
            dtype=torch.float32,
        )
        labels = torch.from_numpy(ids % null_label).to(
            device=time_grid.device, dtype=torch.long
        )
        if first_noise_sha256 is None:
            first_noise_sha256 = tensor_sha256(state)
            first_label_sha256 = tensor_sha256(labels)

        for index in range(len(time_grid) - 1):
            current = float(time_grid[index].item())
            following = float(time_grid[index + 1].item())
            times = torch.full(
                (batch_size,), current, device=state.device, dtype=torch.float32
            )
            with torch.inference_mode(), source_cast:
                full_clean, base_clean, _, _ = evaluate_four_corners(
                    model,
                    state,
                    times,
                    labels,
                    null_label=null_label,
                )
            source_calls += 1
            if switch_time < current < 1.0:
                ou_state = state_to_ou(state.detach(), times).requires_grad_(True)
                with torch.enable_grad():
                    normalized_value = value(ou_state, times, labels)
                    normalized_gradient = torch.autograd.grad(
                        normalized_value.sum(),
                        ou_state,
                        create_graph=False,
                        retain_graph=False,
                    )[0]
                clean, correction = semigroup_value_guided_clean(
                    full_clean.float(),
                    base_clean.float(),
                    normalized_gradient.float(),
                    noise_time=times,
                    beta=beta,
                )
                ordinary = base_clean.float() + beta * (
                    full_clean.float() - base_clean.float()
                )
                correction_rms = sample_rms(correction)
                ordinary_rms = sample_rms(ordinary)
                correction_rms_sum += float(correction_rms.sum().item())
                ordinary_rms_sum += float(ordinary_rms.sum().item())
                correction_ratio_sum += float(
                    (correction_rms / ordinary_rms.clamp_min(1e-8)).sum().item()
                )
                correction_count += batch_size
                value_gradient_calls += 1
                extrapolated_time_calls += int(current > maximum_training_time)
            elif current > switch_time:
                clean = base_clean.float() + beta * (
                    full_clean.float() - base_clean.float()
                )
            else:
                clean = full_clean.float()
            if not torch.isfinite(clean).all():
                raise FloatingPointError(
                    f"non-finite guided clean prediction at t={current:.8f}"
                )
            drift = clean_to_velocity(
                clean,
                state,
                times,
                denominator_floor=t_floor,
            )
            state = state - (current - following) * drift.float()

        with torch.inference_mode():
            decoded = decoder.decode(state).clamp(0, 1)
        if preview is None:
            preview = decoded[: min(16, len(decoded))].float().cpu()
        images_local.append(
            decoded.mul(255)
            .permute(0, 2, 3, 1)
            .to(device="cpu", dtype=torch.uint8)
            .numpy()
        )

    images = np.concatenate(images_local, axis=0)
    np.save(output_dir / f"images-rank{rank:02d}.npy", images)
    np.save(output_dir / f"ids-rank{rank:02d}.npy", global_ids)
    if preview is not None:
        save_image(preview, output_dir / f"preview-rank{rank:02d}.png", nrow=4)
    count = max(1, correction_count)
    audit = {
        "protocol": PROTOCOL,
        "rank": rank,
        "world_size": world_size,
        "sampling_seed": sampling_seed,
        "sample_count": int(len(global_ids)),
        "per_rank_batch": per_rank_batch,
        "precision": precision,
        "initial_generator_sha256": initial_rng,
        "final_generator_sha256": generator_sha256(generator),
        "first_noise_sha256": first_noise_sha256,
        "first_label_sha256": first_label_sha256,
        "source_doubled_calls": source_calls,
        "value_gradient_calls": value_gradient_calls,
        "calls_above_value_training_time": extrapolated_time_calls,
        "correction_count": correction_count,
        "correction_rms_mean": correction_rms_sum / count,
        "ordinary_clean_rms_mean": ordinary_rms_sum / count,
        "correction_to_ordinary_rms_ratio_mean": correction_ratio_sum / count,
        "elapsed_seconds": time.perf_counter() - started,
        "max_memory_allocated_bytes": int(
            torch.cuda.max_memory_allocated(time_grid.device)
        ),
    }
    (output_dir / f"sampling_audit_rank{rank}.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return audit


def value_training_maximum(value: RAEv2NormalizedOUValue) -> float:
    maximum = getattr(value, "training_maximum_noise_time", None)
    if maximum is None:
        raise RuntimeError("value network is missing training-domain metadata")
    return float(maximum)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--value-checkpoint", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=1000)
    parser.add_argument("--per-rank-batch", type=int, default=4)
    parser.add_argument("--sampling-seed", type=int, default=20260903)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--state-key", choices=("ema", "model"), default="ema")
    parser.add_argument("--value-weights", choices=("ema", "model"), default="ema")
    parser.add_argument(
        "--dino-ckpt-dir",
        type=Path,
        default=Path("/home/zhoushunyu/data/eqvae/models/RAEv2/encoders/dinov3"),
    )
    parser.add_argument("--dino-repo-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_count <= 0 or args.per_rank_batch <= 0:
        raise ValueError("sample count and per-rank batch must be positive")
    install_raev2_decoder_config_compat()
    config_path = args.config.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    value_path = args.value_checkpoint.expanduser().resolve()
    results_dir = args.results_dir.expanduser().resolve()
    for path in (config_path, checkpoint_path, value_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    os.environ["DINOV3_CKPT_DIR"] = str(args.dino_ckpt_dir.expanduser().resolve())
    if args.dino_repo_dir is not None:
        os.environ["DINOV3_REPO_DIR"] = str(args.dino_repo_dir.expanduser().resolve())

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    allow_tf32 = args.precision != "fp32"
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.cuda.reset_peak_memory_stats(device)

    config = load_config(config_path)
    decoder = instantiate_from_config(config.stage_1).to(device).eval().requires_grad_(False)
    del decoder.encoder
    torch.cuda.empty_cache()
    model = instantiate_from_config(config.stage_2).to(device).eval().requires_grad_(False)
    payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    model.load_state_dict(payload[args.state_key], strict=True)
    source_step = int(payload.get("step", 0))
    del payload
    value, value_metadata = load_value(
        value_path,
        weights=args.value_weights,
        latent_channels=int(config.misc.latent_size[0]),
        num_classes=int(config.misc.num_classes),
        device=device,
    )
    value_request = value_metadata["request"]
    assert isinstance(value_request, dict)
    if value_request["source_checkpoint_sha256"] != file_sha256(checkpoint_path):
        raise ValueError("value checkpoint was trained against a different source")
    if int(value_request.get("bank_classes_covered", -1)) != int(
        config.misc.num_classes
    ):
        raise ValueError("value checkpoint does not cover every conditioned class")
    value.training_maximum_noise_time = float(value_request["maximum_noise_time"])
    beta = float(value_request["beta"])
    switch_time = float(value_request["switch_time"])

    shift = math.sqrt(
        (config.misc.time_dist_shift_dim or math.prod(config.misc.latent_size))
        / config.misc.time_dist_shift_base
    )
    time_grid = shifted_time_grid(config.sampler.num_steps, shift, device)
    global_ids = np.arange(rank, args.sample_count, world_size, dtype=np.int64)
    output = results_dir / "semigroup_value"
    results_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        request = {
            "protocol": PROTOCOL,
            "config": str(config_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": value_request["source_checkpoint_sha256"],
            "checkpoint_step": source_step,
            "state_key": args.state_key,
            "value": value_metadata,
            "beta": beta,
            "switch_time": switch_time,
            "sample_count": args.sample_count,
            "per_rank_batch": args.per_rank_batch,
            "sampling_seed": args.sampling_seed,
            "precision": args.precision,
            "world_size": world_size,
            "sampler_steps": int(config.sampler.num_steps),
            "time_shift": shift,
            "inference_correction_scale": 1.0,
            "free_scientific_parameters": 0,
        }
        (results_dir / "request.json").write_text(
            json.dumps(request, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    dist.barrier()

    sample_local_shard(
        model=model,
        decoder=decoder,
        value=value,
        beta=beta,
        switch_time=switch_time,
        config=config,
        time_grid=time_grid,
        global_ids=global_ids,
        per_rank_batch=args.per_rank_batch,
        sampling_seed=args.sampling_seed,
        precision=args.precision,
        output_dir=output,
        rank=rank,
        world_size=world_size,
    )
    dist.barrier()
    if rank == 0:
        all_ids = []
        all_images = []
        audits = []
        for shard in range(world_size):
            all_ids.append(np.load(output / f"ids-rank{shard:02d}.npy"))
            all_images.append(np.load(output / f"images-rank{shard:02d}.npy"))
            audits.append(
                json.loads(
                    (output / f"sampling_audit_rank{shard}.json").read_text(
                        encoding="utf-8"
                    )
                )
            )
        ids = np.concatenate(all_ids)
        images = np.concatenate(all_images)
        order = np.argsort(ids)
        ids = ids[order]
        images = images[order]
        if not np.array_equal(ids, np.arange(args.sample_count)):
            raise RuntimeError("distributed sample IDs are incomplete or duplicated")
        archive = output / "samples.npz"
        np.savez(archive, images)
        summary = {
            "protocol": PROTOCOL,
            "samples": int(len(images)),
            "archive": str(archive),
            "archive_sha256": file_sha256(archive),
            "checkpoint_step": source_step,
            "value_step": int(value_metadata["step"]),
            "max_rank_elapsed_seconds": max(
                float(audit["elapsed_seconds"]) for audit in audits
            ),
            "max_memory_allocated_bytes": max(
                int(audit["max_memory_allocated_bytes"]) for audit in audits
            ),
            "correction_rms_mean": sum(
                float(audit["correction_rms_mean"]) * int(audit["correction_count"])
                for audit in audits
            )
            / max(1, sum(int(audit["correction_count"]) for audit in audits)),
        }
        (output / "sampling_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        for shard in range(world_size):
            (output / f"ids-rank{shard:02d}.npy").unlink()
            (output / f"images-rank{shard:02d}.npy").unlink()
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
