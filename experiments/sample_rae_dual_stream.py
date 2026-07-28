"""Sample semantic-only, paired-detail, and shuffled-detail RAE controls."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
RAE_ROOT = ROOT / "external/RAE"
RAE_SRC = RAE_ROOT / "src"
for path in (ROOT, RAE_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.rae_dual_stream import (  # noqa: E402
    SemanticConditionedDetailDDT,
    fuse_semantic_coefficients,
    split_semantic_coefficients,
)
from experiments.train_rae_layerwise_path import (  # noqa: E402
    configure_determinism,
    resolve_stage1_paths,
)
from sample_ddp import count_contiguous_png_prefix, create_npz_from_sample_folder  # noqa: E402
from stage1 import RAE  # noqa: E402
from stage2.transport import Sampler, create_transport  # noqa: E402
from utils.model_utils import instantiate_from_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--semantic-checkpoint", type=Path, required=True)
    parser.add_argument("--paired-detail-checkpoint", type=Path, required=True)
    parser.add_argument("--shuffled-detail-checkpoint", type=Path, required=True)
    parser.add_argument("--subspaces", type=Path, required=True)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--per-process-batch", type=int, default=4)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--global-seed", type=int, default=20_260_718)
    return parser.parse_args()


def load_detail(path: Path, device: torch.device) -> SemanticConditionedDetailDDT:
    model = SemanticConditionedDetailDDT().to(device=device, dtype=torch.float32)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    return model.requires_grad_(False).eval()


def decode_uint8(rae: RAE, latent: torch.Tensor) -> np.ndarray:
    images = rae.decode(latent).clamp(0, 1)
    return images.mul(255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()


def main() -> None:
    args = parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    configure_determinism(int(args.global_seed) * world_size + rank)

    config = OmegaConf.load(args.config)
    resolve_stage1_paths(config)
    rae: RAE = instantiate_from_config(config.stage_1).to(device=device, dtype=torch.float32)
    semantic_model = instantiate_from_config(config.stage_2).to(
        device=device, dtype=torch.float32
    )
    semantic_checkpoint = torch.load(
        args.semantic_checkpoint, map_location="cpu", weights_only=False
    )
    semantic_model.load_state_dict(semantic_checkpoint["ema"])
    semantic_model.requires_grad_(False).eval()
    rae.requires_grad_(False).eval()
    paired_model = load_detail(args.paired_detail_checkpoint, device)
    shuffled_model = load_detail(args.shuffled_detail_checkpoint, device)
    subspaces = torch.load(args.subspaces, map_location="cpu", weights_only=False)
    entry = subspaces["subspaces"].get(16, subspaces["subspaces"].get("16"))
    basis = entry["basis"].to(device=device, dtype=torch.float32)

    transport_params = OmegaConf.to_container(config.transport.params, resolve=True)
    shift = math.sqrt(
        float(config.misc.time_dist_shift_dim) / float(config.misc.time_dist_shift_base)
    )
    transport = create_transport(**dict(transport_params), time_dist_shift=shift)
    sample_fn = Sampler(transport).sample_ode(
        sampling_method="euler", num_steps=int(args.steps), atol=1e-6, rtol=1e-3
    )

    folders = {
        "semantic_only": args.sample_dir / "semantic_only",
        "paired_detail": args.sample_dir / "paired_detail",
        "shuffled_detail": args.sample_dir / "shuffled_detail",
    }
    if rank == 0:
        for folder in folders.values():
            folder.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    batch = int(args.per_process_batch)
    global_batch = batch * world_size
    total_samples = int(math.ceil(int(args.num_samples) / global_batch) * global_batch)
    iterations = total_samples // global_batch
    if rank == 0:
        completed = min(
            count_contiguous_png_prefix(str(folder))[0] // global_batch
            for folder in folders.values()
        )
    else:
        completed = 0
    completed_tensor = torch.tensor([completed], device=device, dtype=torch.long)
    dist.broadcast(completed_tensor, src=0)
    completed = int(completed_tensor.item())

    semantic_generator = torch.Generator(device=device).manual_seed(
        int(args.global_seed) * world_size + rank
    )
    detail_generator = torch.Generator(device=device).manual_seed(
        (int(args.global_seed) + 1_000_003) * world_size + rank
    )
    for _ in range(completed):
        torch.randn(
            (batch, 768, 16, 16), device=device, dtype=torch.float32,
            generator=semantic_generator,
        )
        torch.randn(
            (batch, 16, 16, 16), device=device, dtype=torch.float32,
            generator=detail_generator,
        )

    iterator = range(completed, iterations)
    progress = tqdm(iterator) if rank == 0 else iterator
    with torch.inference_mode():
        for step_index in progress:
            semantic_noise = torch.randn(
                (batch, 768, 16, 16),
                device=device,
                dtype=torch.float32,
                generator=semantic_generator,
            )
            detail_noise = torch.randn(
                (batch, 16, 16, 16),
                device=device,
                dtype=torch.float32,
                generator=detail_generator,
            )
            global_indices = (
                step_index * global_batch
                + torch.arange(batch, device=device, dtype=torch.long) * world_size
                + rank
            )
            labels = global_indices.remainder(1000)
            semantic = sample_fn(
                semantic_noise, semantic_model.forward, y=labels
            )[-1]
            semantic, _ = split_semantic_coefficients(semantic, basis)

            def paired_forward(value, time, y):
                return paired_model(value, time, y, semantic)

            def shuffled_forward(value, time, y):
                return shuffled_model(value, time, y, semantic)

            paired_coefficients = sample_fn(
                detail_noise.clone(), paired_forward, y=labels
            )[-1]
            shuffled_coefficients = sample_fn(
                detail_noise.clone(), shuffled_forward, y=labels
            )[-1]
            outputs = {
                "semantic_only": semantic,
                "paired_detail": fuse_semantic_coefficients(
                    semantic, paired_coefficients, basis
                ),
                "shuffled_detail": fuse_semantic_coefficients(
                    semantic, shuffled_coefficients, basis
                ),
            }
            for name, latent in outputs.items():
                images = decode_uint8(rae, latent)
                for local_index, image in enumerate(images):
                    index = step_index * global_batch + local_index * world_size + rank
                    if index < int(args.num_samples):
                        Image.fromarray(image).save(folders[name] / f"{index:06d}.png")
            dist.barrier()
    dist.barrier()
    if rank == 0:
        for folder in folders.values():
            create_npz_from_sample_folder(str(folder), int(args.num_samples))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
