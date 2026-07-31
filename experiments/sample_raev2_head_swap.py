"""Same-noise RAEv2 sampling with independently sourced full/contrast fields."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.raev2_head_swap import forward_with_head_swap
from experiments.raev2_training_core import file_sha256, tensor_fingerprint
from experiments.sample_raev2_threeway import (
    generator_fingerprint,
    load_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--flow-checkpoint", type=Path, required=True)
    parser.add_argument("--treatment-checkpoint", type=Path, required=True)
    parser.add_argument("--treatment-name", default="lpl")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=1000)
    parser.add_argument("--per-rank-batch", type=int, default=4)
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--state-key", choices=("ema", "model"), default="model")
    parser.add_argument("--ig-scale", type=float, default=1.78)
    parser.add_argument(
        "--dino-ckpt-dir",
        type=Path,
        default=Path("/home/zhoushunyu/data/eqvae/models/RAEv2/encoders/dinov3"),
    )
    parser.add_argument("--dino-repo-dir", type=Path)
    return parser.parse_args()


def load_checkpoint_model(
    model: torch.nn.Module,
    path: Path,
    *,
    state_key: str,
) -> dict[str, object]:
    path = path.expanduser().resolve()
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if state_key not in checkpoint:
        raise KeyError(f"{path} has no {state_key!r} state")
    model.load_state_dict(checkpoint[state_key], strict=True)
    metadata = {
        "path": str(path),
        "sha256": file_sha256(path),
        "step": int(checkpoint.get("step", 0)),
        "state_key": state_key,
    }
    del checkpoint
    return metadata


def main() -> None:
    args = parse_args()
    if args.sample_count <= 0 or args.sample_count % 1000:
        raise ValueError("--sample-count must be a positive multiple of 1000")
    if args.per_rank_batch <= 0:
        raise ValueError("--per-rank-batch must be positive")
    if args.ig_scale < 0:
        raise ValueError("--ig-scale must be non-negative")
    if not args.treatment_name or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for character in args.treatment_name
    ):
        raise ValueError("--treatment-name must be a non-empty safe path component")

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
    guidance_interval = (
        float(config.guidance.ig.t_min),
        float(config.guidance.ig.t_max),
    )

    from experiments.raev2_stage1_compat import install_raev2_decoder_config_compat
    from stage2.transport import create_sampler, create_transport
    from utils.model_utils import instantiate_from_config

    install_raev2_decoder_config_compat()
    rae = instantiate_from_config(config.stage_1).to(device)
    rae.eval()
    rae.requires_grad_(False)
    del rae.encoder
    torch.cuda.empty_cache()

    flow_model = instantiate_from_config(config.stage_2).to(device)
    treatment_model = instantiate_from_config(config.stage_2).to(device)
    for model in (flow_model, treatment_model):
        model.eval()
        model.requires_grad_(False)

    flow_metadata = load_checkpoint_model(
        flow_model,
        args.flow_checkpoint,
        state_key=args.state_key,
    )
    treatment_metadata = load_checkpoint_model(
        treatment_model,
        args.treatment_checkpoint,
        state_key=args.state_key,
    )

    time_dist_shift = math.sqrt(
        (config.misc.time_dist_shift_dim or math.prod(latent_size))
        / config.misc.time_dist_shift_base
    )
    transport = create_transport(config=config.transport, time_dist_shift=time_dist_shift)
    sampler = create_sampler(transport, guidance_config=config.guidance)
    sample_fn = sampler.sample_ode(**dataclasses.asdict(config.sampler))
    global_ids = np.arange(rank, args.sample_count, world_size, dtype=np.int64)
    args.results_dir.expanduser().mkdir(parents=True, exist_ok=True)

    treatment = args.treatment_name
    branches = (
        ("flowF_flowD", flow_model, flow_model, "flow", "flow"),
        (
            f"{treatment}F_flowD",
            treatment_model,
            flow_model,
            treatment,
            "flow",
        ),
        (
            f"flowF_{treatment}D",
            flow_model,
            treatment_model,
            "flow",
            treatment,
        ),
        (
            f"{treatment}F_{treatment}D",
            treatment_model,
            treatment_model,
            treatment,
            treatment,
        ),
    )

    for (
        branch_name,
        full_model,
        contrast_model,
        full_source,
        contrast_source,
    ) in branches:
        output_dir = args.results_dir.expanduser() / branch_name
        if rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
        dist.barrier()

        generator = torch.Generator(device=device)
        generator.manual_seed(int(args.sampling_seed) * world_size + rank)
        initial_generator_sha256 = generator_fingerprint(generator)
        images_local: list[np.ndarray] = []
        first_noise_sha256 = None
        first_label_sha256 = None
        model_fn = partial(
            forward_with_head_swap,
            full_model,
            contrast_model,
            guidance_scale=float(args.ig_scale),
            guidance_interval=guidance_interval,
        )

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

                null = torch.full(
                    (batch_size,),
                    int(config.misc.num_classes),
                    device=device,
                    dtype=torch.long,
                )
                sample_noise = torch.cat([noise, noise], dim=0)
                context = torch.cat([labels, null], dim=0)
                autocast = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if args.precision == "bf16"
                    else __import__("contextlib").nullcontext()
                )
                with autocast:
                    latent = sample_fn(
                        sample_noise,
                        model_fn,
                        context=context,
                        attn_mask=None,
                    )[-1]
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
            "protocol": "raev2_head_swap_same_noise_v1",
            "branch": branch_name,
            "rank": rank,
            "world_size": world_size,
            "full_source": full_source,
            "contrast_source": contrast_source,
            "flow_checkpoint": flow_metadata,
            "treatment_checkpoint": treatment_metadata,
            "sampling_seed": int(args.sampling_seed),
            "sample_count": int(args.sample_count),
            "per_rank_batch": int(args.per_rank_batch),
            "sampler_steps": int(config.sampler.num_steps),
            "guidance_ig_scale": float(args.ig_scale),
            "guidance_ig_interval": list(guidance_interval),
            "initial_generator_sha256": initial_generator_sha256,
            "first_noise_sha256": first_noise_sha256,
            "first_label_sha256": first_label_sha256,
            "final_generator_sha256": generator_fingerprint(generator),
        }
        (output_dir / f"sampling_audit_rank{rank}.json").write_text(
            json.dumps(audit, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        dist.barrier()

        if rank == 0:
            all_ids = []
            all_images = []
            for shard_rank in range(world_size):
                all_ids.append(np.load(output_dir / f"ids-rank{shard_rank:02d}.npy"))
                all_images.append(
                    np.load(output_dir / f"images-rank{shard_rank:02d}.npy")
                )
            ids = np.concatenate(all_ids)
            images = np.concatenate(all_images)
            order = np.argsort(ids)
            ids = ids[order]
            images = images[order]
            if not np.array_equal(
                ids,
                np.arange(args.sample_count, dtype=np.int64),
            ):
                raise RuntimeError("distributed sample IDs are incomplete or duplicated")
            archive = output_dir / "samples.npz"
            np.savez(archive, images)
            summary = {
                "protocol": "raev2_head_swap_same_noise_v1",
                "branch": branch_name,
                "full_source": full_source,
                "contrast_source": contrast_source,
                "flow_checkpoint": flow_metadata,
                "treatment_checkpoint": treatment_metadata,
                "samples": int(images.shape[0]),
                "shape": list(images.shape),
                "archive": str(archive),
                "archive_sha256": file_sha256(archive),
            }
            (output_dir / "sampling_summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            for shard_rank in range(world_size):
                (output_dir / f"images-rank{shard_rank:02d}.npy").unlink()
                (output_dir / f"ids-rank{shard_rank:02d}.npy").unlink()
            print(json.dumps(summary, ensure_ascii=False))
        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
