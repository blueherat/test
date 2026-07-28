"""Strict same-noise sampling for official, Flow, and LPL RAEv2 checkpoints."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
RAEV2_SRC = ROOT / "external" / "RAEv2" / "src"
for path in (RAEV2_SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from configs.stage2 import Stage2Config  # noqa: E402
from experiments.raev2_training_core import file_sha256, tensor_fingerprint  # noqa: E402
from stage2.transport import create_sampler, create_transport  # noqa: E402
from stage2.utils import validate_stage2_config  # noqa: E402
from utils.guidance_utils import get_model_forward_fn  # noqa: E402
from utils.model_utils import instantiate_from_config  # noqa: E402


def parse_branch(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("branch must be NAME=CHECKPOINT")
    name, path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("branch name cannot be empty")
    return name, Path(path).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--branch", action="append", type=parse_branch, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=5000)
    parser.add_argument("--per-rank-batch", type=int, default=4)
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--state-key", choices=("ema", "model"), default="ema")
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
    config.prepare_model_params()
    return config


def generator_fingerprint(generator: torch.Generator) -> str:
    return hashlib.sha256(generator.get_state().cpu().numpy().tobytes()).hexdigest()


def main() -> None:
    args = parse_args()
    if args.sample_count <= 0 or args.sample_count % 1000:
        raise ValueError("--sample-count must be a positive multiple of 1000")
    if args.per_rank_batch <= 0:
        raise ValueError("--per-rank-batch must be positive")
    names = [name for name, _ in args.branch]
    if len(names) != len(set(names)):
        raise ValueError("branch names must be unique")

    os.environ["DINOV3_CKPT_DIR"] = str(args.dino_ckpt_dir.expanduser().resolve())
    if args.dino_repo_dir is not None:
        os.environ["DINOV3_REPO_DIR"] = str(args.dino_repo_dir.expanduser().resolve())

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    config = load_config(args.config)
    latent_size = tuple(config.misc.latent_size)
    rae = instantiate_from_config(config.stage_1).to(device)
    rae.eval()
    rae.requires_grad_(False)
    # Sampling only needs the deterministic decoder.
    del rae.encoder
    torch.cuda.empty_cache()

    model = instantiate_from_config(config.stage_2).to(device)
    model.eval()
    model.requires_grad_(False)
    model_fn, model_kwargs_base = get_model_forward_fn(model, config.guidance)
    use_guidance = config.guidance.any_guidance_active
    time_dist_shift = math.sqrt(
        (config.misc.time_dist_shift_dim or math.prod(latent_size))
        / config.misc.time_dist_shift_base
    )
    transport = create_transport(config=config.transport, time_dist_shift=time_dist_shift)
    sampler = create_sampler(transport, guidance_config=config.guidance)
    sample_fn = sampler.sample_ode(**dataclasses.asdict(config.sampler))

    global_ids = np.arange(rank, args.sample_count, world_size, dtype=np.int64)
    args.results_dir.expanduser().mkdir(parents=True, exist_ok=True)

    for branch_name, checkpoint_path in args.branch:
        checkpoint_path = checkpoint_path.resolve()
        output_dir = args.results_dir.expanduser() / branch_name
        if rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
        dist.barrier()

        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False, mmap=True
        )
        if args.state_key not in checkpoint:
            raise KeyError(f"{checkpoint_path} has no {args.state_key!r} state")
        model.load_state_dict(checkpoint[args.state_key], strict=True)
        checkpoint_step = int(checkpoint.get("step", 0))
        del checkpoint

        generator = torch.Generator(device=device)
        generator.manual_seed(int(args.sampling_seed) * world_size + rank)
        initial_generator_sha256 = generator_fingerprint(generator)
        images_local: list[np.ndarray] = []
        first_noise_sha256 = None
        first_label_sha256 = None

        with torch.inference_mode():
            for start in range(0, len(global_ids), args.per_rank_batch):
                ids = global_ids[start : start + args.per_rank_batch]
                batch_size = len(ids)
                noise = torch.randn(
                    batch_size,
                    *latent_size,
                    generator=generator,
                    device=device,
                    dtype=torch.float32,
                )
                labels = torch.from_numpy(ids % 1000).to(device=device, dtype=torch.long)
                if first_noise_sha256 is None:
                    first_noise_sha256 = tensor_fingerprint(noise)
                    first_label_sha256 = tensor_fingerprint(labels)

                sample_noise = noise
                context = labels
                if use_guidance:
                    sample_noise = torch.cat([sample_noise, sample_noise], dim=0)
                    null = torch.full(
                        (batch_size,),
                        int(config.misc.num_classes),
                        device=device,
                        dtype=torch.long,
                    )
                    context = torch.cat([context, null], dim=0)
                model_kwargs = dict(model_kwargs_base)
                model_kwargs.update(context=context, attn_mask=None)
                autocast = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if args.precision == "bf16"
                    else __import__("contextlib").nullcontext()
                )
                with autocast:
                    latent = sample_fn(sample_noise, model_fn, **model_kwargs)[-1]
                    if use_guidance:
                        latent = latent.chunk(2, dim=0)[0]
                    decoded = rae.decode(latent).clamp(0, 1)
                images_local.append(
                    decoded.mul(255)
                    .permute(0, 2, 3, 1)
                    .to(device="cpu", dtype=torch.uint8)
                    .numpy()
                )

        images_array = np.concatenate(images_local, axis=0)
        np.save(output_dir / f"images-rank{rank:02d}.npy", images_array)
        np.save(output_dir / f"ids-rank{rank:02d}.npy", global_ids)
        audit = {
            "protocol": "raev2_same_noise_v1",
            "branch": branch_name,
            "rank": rank,
            "world_size": world_size,
            "checkpoint": str(checkpoint_path),
            "checkpoint_step": checkpoint_step,
            "state_key": args.state_key,
            "sampling_seed": args.sampling_seed,
            "sample_count": args.sample_count,
            "per_rank_batch": args.per_rank_batch,
            "sampler_steps": int(config.sampler.num_steps),
            "guidance_cfg_scale": float(config.guidance.cfg.scale),
            "guidance_ig_scale": float(config.guidance.ig.scale),
            "guidance_ig_t_min": float(config.guidance.ig.t_min),
            "initial_generator_sha256": initial_generator_sha256,
            "first_noise_sha256": first_noise_sha256,
            "first_label_sha256": first_label_sha256,
            "first_labels": (global_ids[: args.per_rank_batch] % 1000).tolist(),
            "final_generator_sha256": generator_fingerprint(generator),
        }
        (output_dir / f"sampling_audit_rank{rank}.json").write_text(
            json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        dist.barrier()

        if rank == 0:
            all_ids = []
            all_images = []
            for shard_rank in range(world_size):
                all_ids.append(np.load(output_dir / f"ids-rank{shard_rank:02d}.npy"))
                all_images.append(np.load(output_dir / f"images-rank{shard_rank:02d}.npy"))
            ids = np.concatenate(all_ids)
            images = np.concatenate(all_images)
            order = np.argsort(ids)
            ids = ids[order]
            images = images[order]
            expected = np.arange(args.sample_count, dtype=np.int64)
            if not np.array_equal(ids, expected):
                raise RuntimeError("distributed sample IDs are incomplete or duplicated")
            archive = output_dir / "samples.npz"
            np.savez(archive, images)
            summary = {
                "branch": branch_name,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": file_sha256(checkpoint_path),
                "checkpoint_step": checkpoint_step,
                "state_key": args.state_key,
                "samples": int(images.shape[0]),
                "shape": list(images.shape),
                "archive": str(archive),
                "archive_sha256": file_sha256(archive),
            }
            (output_dir / "sampling_summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(json.dumps(summary, ensure_ascii=False))
        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
