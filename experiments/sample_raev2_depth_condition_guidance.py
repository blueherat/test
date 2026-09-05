#!/usr/bin/env python3
"""Paired RAEv2 sampling for depth-by-condition guidance components.

Every model call evaluates the same latent twice, once with its ImageNet label
and once with the checkpoint's learned null label.  The full/base outputs form
the four corners used by :mod:`experiments.raev2_depth_condition_guidance`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
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

from configs.stage2 import Stage2Config  # noqa: E402
from experiments.raev2_characteristic_guidance import (  # noqa: E402
    evaluate_first_characteristic_clean,
)
from experiments.raev2_depth_condition_guidance import (  # noqa: E402
    guided_clean_prediction,
)
from experiments.raev2_pfr_retiming import clean_to_velocity  # noqa: E402
from experiments.raev2_stage1_compat import (  # noqa: E402
    install_raev2_decoder_config_compat,
)
from experiments.raev2_training_core import file_sha256  # noqa: E402
from experiments.sample_raev2_pfr_retiming import (  # noqa: E402
    generator_sha256,
    load_config,
    shifted_time_grid,
    tensor_sha256,
)
from stage2.utils import validate_stage2_config  # noqa: E402
from utils.model_utils import instantiate_from_config  # noqa: E402


PROTOCOL = "raev2_depth_condition_guidance_v1"
DEFAULT_CONFIG = ROOT / "experiments/configs/raev2_strict_lpl_dinov3l_k7.yaml"
DEFAULT_CHECKPOINT = Path(
    "/home/zhoushunyu/data/eqvae/models/RAEv2/stage2/imagenet/"
    "dinov3l-k7/checkpoint.pt"
)


@dataclass(frozen=True)
class SamplingCondition:
    name: str
    mode: str
    guidance_scale: float = 1.78
    guidance_min_time: float = 0.5
    guidance_max_time: float = 1.0

    def validate(self) -> None:
        if not self.name or any(char in self.name for char in "/=,:"):
            raise ValueError("condition name must be a safe path component")
        if self.mode not in {
            "full_conditional",
            "conditional_depth",
            "marginal_depth",
            "interaction",
            "conditional_marginal_midpoint",
            "conditional_marginal_consensus",
            "conditional_marginal_orthogonal_positive",
            "conditional_marginal_orthogonal_negative",
            "conditional_marginal_orthogonal_donor",
            "characteristic_one_step",
        }:
            raise ValueError(f"unknown guidance mode: {self.mode}")
        if not math.isfinite(self.guidance_scale) or self.guidance_scale < 0.0:
            raise ValueError("guidance scale must be finite and non-negative")
        if not (
            math.isfinite(self.guidance_min_time)
            and math.isfinite(self.guidance_max_time)
            and 0.0 <= self.guidance_min_time <= self.guidance_max_time <= 1.0
        ):
            raise ValueError("guidance interval must satisfy 0 <= min <= max <= 1")

    def active_at(self, noise_time: float) -> bool:
        return (
            self.mode != "full_conditional"
            and self.guidance_scale != 1.0
            and self.guidance_min_time < noise_time <= self.guidance_max_time
        )


DEFAULT_CONDITIONS = (
    SamplingCondition("full_conditional", "full_conditional", 1.0),
    SamplingCondition("conditional_ig_s1p78_t0p5_1p0", "conditional_depth"),
    SamplingCondition("marginal_ig_s1p78_t0p5_1p0", "marginal_depth"),
    SamplingCondition("interaction_ig_s1p78_t0p5_1p0", "interaction"),
)


def parse_condition(text: str) -> SamplingCondition:
    parts = text.split(",")
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(
            "condition must be NAME,MODE,SCALE,MIN_TIME,MAX_TIME"
        )
    try:
        condition = SamplingCondition(
            name=parts[0],
            mode=parts[1],
            guidance_scale=float(parts[2]),
            guidance_min_time=float(parts[3]),
            guidance_max_time=float(parts[4]),
        )
        condition.validate()
        return condition
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def evaluate_four_corners(
    model: torch.nn.Module,
    state: torch.Tensor,
    times: torch.Tensor,
    labels: torch.Tensor,
    *,
    null_label: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate and return ``full_c, base_c, full_u, base_u``."""

    batch_size = state.shape[0]
    if times.shape != (batch_size,) or labels.shape != (batch_size,):
        raise ValueError("times and labels must match the state batch")
    doubled_state = torch.cat((state, state), dim=0)
    doubled_times = torch.cat((times, times), dim=0)
    null_labels = torch.full_like(labels, int(null_label))
    contexts = torch.cat((labels, null_labels), dim=0)
    output = model(
        doubled_state,
        doubled_times,
        context=contexts,
        attn_mask=None,
    )
    if not isinstance(output, tuple) or len(output) != 2:
        raise TypeError("RAEv2 internal-guidance model must return (full, base)")
    full, base = output
    if full.shape[0] != 2 * batch_size or base.shape != full.shape:
        raise ValueError("model outputs do not preserve the doubled batch")
    full_conditional, full_unconditional = full.chunk(2, dim=0)
    base_conditional, base_unconditional = base.chunk(2, dim=0)
    return (
        full_conditional,
        base_conditional,
        full_unconditional,
        base_unconditional,
    )


def sample_condition(
    *,
    model: torch.nn.Module,
    decoder: torch.nn.Module,
    condition: SamplingCondition,
    config: Stage2Config,
    time_grid: torch.Tensor,
    global_ids: np.ndarray,
    per_rank_batch: int,
    sampling_seed: int,
    precision: str,
    output_dir: Path,
    rank: int,
    world_size: int,
) -> dict[str, object]:
    condition.validate()
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device=time_grid.device)
    generator.manual_seed(int(sampling_seed) * world_size + rank)
    initial_rng = generator_sha256(generator)
    images_local: list[np.ndarray] = []
    preview = None
    first_noise_sha256 = None
    first_label_sha256 = None
    model_calls = 0
    started = time.perf_counter()
    t_floor = float(config.transport.t_eps)
    null_label = int(config.misc.num_classes)

    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if precision == "bf16"
        else nullcontext()
    )
    with torch.inference_mode(), autocast:
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
                step = current - following
                times = torch.full(
                    (batch_size,), current, device=state.device, dtype=torch.float32
                )
                full_c, base_c, full_u, base_u = evaluate_four_corners(
                    model,
                    state,
                    times,
                    labels,
                    null_label=null_label,
                )
                model_calls += 1
                if condition.active_at(current):
                    if condition.mode == "characteristic_one_step":
                        clean, _ = evaluate_first_characteristic_clean(
                            model,
                            state,
                            times,
                            labels,
                            full_c,
                            base_c,
                            guidance_scale=condition.guidance_scale,
                        )
                        model_calls += 1
                    else:
                        clean = guided_clean_prediction(
                            full_conditional=full_c,
                            base_conditional=base_c,
                            full_unconditional=full_u,
                            base_unconditional=base_u,
                            guidance_scale=condition.guidance_scale,
                            mode=condition.mode,
                        )
                else:
                    clean = full_c
                drift = clean_to_velocity(
                    clean,
                    state,
                    times,
                    denominator_floor=t_floor,
                )
                state = state - step * drift

            decoded = decoder.decode(state).clamp(0, 1)
            if preview is None:
                preview = decoded[: min(16, len(decoded))].float().cpu()
            images_local.append(
                decoded.mul(255)
                .permute(0, 2, 3, 1)
                .to(device="cpu", dtype=torch.uint8)
                .numpy()
            )

    elapsed = time.perf_counter() - started
    images = np.concatenate(images_local, axis=0)
    np.save(output_dir / f"images-rank{rank:02d}.npy", images)
    np.save(output_dir / f"ids-rank{rank:02d}.npy", global_ids)
    if preview is not None:
        save_image(preview, output_dir / f"preview-rank{rank:02d}.png", nrow=4)
    audit = {
        "protocol": PROTOCOL,
        "condition": asdict(condition),
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
        "doubled_model_calls": model_calls,
        "elapsed_seconds": elapsed,
        "max_memory_allocated_bytes": int(
            torch.cuda.max_memory_allocated(time_grid.device)
        ),
        "max_memory_reserved_bytes": int(
            torch.cuda.max_memory_reserved(time_grid.device)
        ),
    }
    (output_dir / f"sampling_audit_rank{rank}.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=1000)
    parser.add_argument("--per-rank-batch", type=int, default=1)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--sampling-seed", type=int, default=20260903)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--state-key", choices=("ema", "model"), default="ema")
    parser.add_argument("--condition", action="append", type=parse_condition)
    parser.add_argument(
        "--dino-ckpt-dir",
        type=Path,
        default=Path("/home/zhoushunyu/data/eqvae/models/RAEv2/encoders/dinov3"),
    )
    parser.add_argument("--dino-repo-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    install_raev2_decoder_config_compat()
    args = parse_args()
    conditions = tuple(args.condition or DEFAULT_CONDITIONS)
    if len({condition.name for condition in conditions}) != len(conditions):
        raise ValueError("condition names must be unique")
    if args.sample_count <= 0 or args.per_rank_batch <= 0:
        raise ValueError("sample count and per-rank batch must be positive")
    if args.num_steps is not None and args.num_steps <= 0:
        raise ValueError("number of sampling steps must be positive")
    args.config = args.config.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.results_dir = args.results_dir.expanduser().resolve()
    if not args.config.is_file() or not args.checkpoint.is_file():
        raise FileNotFoundError("RAEv2 config or checkpoint is missing")
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

    config = load_config(args.config)
    decoder = instantiate_from_config(config.stage_1).to(device).eval()
    decoder.requires_grad_(False)
    del decoder.encoder
    torch.cuda.empty_cache()
    model = instantiate_from_config(config.stage_2).to(device).eval()
    model.requires_grad_(False)
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    model.load_state_dict(checkpoint[args.state_key], strict=True)
    checkpoint_step = int(checkpoint.get("step", 0))
    del checkpoint

    shift = math.sqrt(
        (config.misc.time_dist_shift_dim or math.prod(config.misc.latent_size))
        / config.misc.time_dist_shift_base
    )
    num_steps = int(
        config.sampler.num_steps if args.num_steps is None else args.num_steps
    )
    time_grid = shifted_time_grid(num_steps, shift, device)
    global_ids = np.arange(rank, args.sample_count, world_size, dtype=np.int64)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        request = {
            "protocol": PROTOCOL,
            "config": str(args.config),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "checkpoint_step": checkpoint_step,
            "state_key": args.state_key,
            "conditions": [asdict(condition) for condition in conditions],
            "sample_count": args.sample_count,
            "per_rank_batch": args.per_rank_batch,
            "sampling_seed": args.sampling_seed,
            "precision": args.precision,
            "allow_tf32": allow_tf32,
            "world_size": world_size,
            "sampler_steps": num_steps,
            "time_shift": shift,
            "transport_prediction": str(config.transport.prediction),
            "transport_t_eps": float(config.transport.t_eps),
            "null_label": int(config.misc.num_classes),
        }
        (args.results_dir / "request.json").write_text(
            json.dumps(request, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    dist.barrier()

    for condition in conditions:
        output = args.results_dir / condition.name
        sample_condition(
            model=model,
            decoder=decoder,
            condition=condition,
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
                "condition": asdict(condition),
                "samples": int(len(images)),
                "archive": str(archive),
                "archive_sha256": file_sha256(archive),
                "checkpoint_step": checkpoint_step,
                "state_key": args.state_key,
                "total_doubled_model_calls": sum(
                    int(audit["doubled_model_calls"]) for audit in audits
                ),
                "max_rank_elapsed_seconds": max(
                    float(audit["elapsed_seconds"]) for audit in audits
                ),
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
