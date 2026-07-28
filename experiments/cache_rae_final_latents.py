"""Cache a fixed augmented ImageNet stream as normalized fp32 RAE latents."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF


ROOT = Path(__file__).resolve().parents[1]
RAE_SRC = ROOT / "external/RAE/src"
for path in (ROOT, RAE_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.rae_latent_cache import cache_directory, split_range  # noqa: E402
from experiments.train_rae_layerwise_path import (  # noqa: E402
    configure_determinism,
    resolve_stage1_paths,
)
from experiments.train_rae_spectral_tiny import setup_distributed  # noqa: E402
from stage1 import RAE  # noqa: E402
from utils.model_utils import instantiate_from_config  # noqa: E402
from utils.train_utils import ParquetImageNetDataset, center_crop_arr  # noqa: E402


class FixedAugmentedSubset(Dataset):
    def __init__(
        self,
        base: Dataset,
        indices: np.ndarray,
        flips: np.ndarray,
        image_size: int,
    ) -> None:
        if len(indices) != len(flips):
            raise ValueError("indices and flips must have equal length")
        self.base = base
        self.indices = indices
        self.flips = flips
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, int]:
        image, label = self.base[int(self.indices[int(position)])]
        image = center_crop_arr(image, self.image_size)
        if bool(self.flips[int(position)]):
            image = TF.hflip(image)
        return TF.to_tensor(image), int(label)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sample-count", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--flush-every", type=int, default=50)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def prepare_selection(
    output: Path,
    *,
    dataset_size: int,
    sample_count: int,
    seed: int,
) -> None:
    if sample_count > dataset_size:
        raise ValueError("cache stream must not repeat ImageNet samples")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    logical_indices = torch.randperm(dataset_size, generator=generator)[:sample_count].numpy()
    indices = np.sort(logical_indices)
    stream_order = np.searchsorted(indices, logical_indices)
    if not np.array_equal(indices[stream_order], logical_indices):
        raise RuntimeError("failed to construct logical-to-physical cache order")
    flip_generator = torch.Generator(device="cpu").manual_seed(int(seed) + 1_000_003)
    logical_flips = (torch.rand(sample_count, generator=flip_generator) < 0.5).numpy()
    flips = np.empty(sample_count, dtype=np.bool_)
    flips[stream_order] = logical_flips
    np.save(output / "indices.npy", indices.astype(np.int64, copy=False))
    np.save(output / "flip_h.npy", flips.astype(np.bool_, copy=False))
    np.save(output / "stream_order.npy", stream_order.astype(np.int64, copy=False))


def main() -> None:
    args = parse_args()
    rank, world_size, device = setup_distributed()
    configure_determinism(int(args.seed) * world_size + rank)
    output = cache_directory(args.cache_root, args.seed, args.sample_count)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    complete_manifest = output / "manifest.json"
    if complete_manifest.exists():
        manifest = json.loads(complete_manifest.read_text(encoding="utf-8"))
        if bool(manifest.get("complete")):
            dist.destroy_process_group()
            return

    base = ParquetImageNetDataset(args.data_path, split="train", transform=None)
    selection_files = (output / "indices.npy", output / "flip_h.npy", output / "stream_order.npy")
    if rank == 0 and not all(path.exists() for path in selection_files):
        prepare_selection(
            output,
            dataset_size=len(base),
            sample_count=int(args.sample_count),
            seed=int(args.seed),
        )
    dist.barrier()
    all_indices = np.load(output / "indices.npy", mmap_mode="r")
    all_flips = np.load(output / "flip_h.npy", mmap_mode="r")
    start, end = split_range(int(args.sample_count), rank, world_size)

    config = OmegaConf.load(args.config)
    resolve_stage1_paths(config)
    rae: RAE = instantiate_from_config(config.stage_1)
    rae.decoder = torch.nn.Identity()
    rae = rae.to(device=device, dtype=torch.float32).requires_grad_(False).eval()

    latent_path = output / f"latents-rank{rank:02d}.npy"
    label_path = output / f"labels-rank{rank:02d}.npy"
    progress_path = output / f"progress-rank{rank:02d}.json"
    local_count = end - start
    completed = 0
    if progress_path.exists() and latent_path.exists() and label_path.exists():
        completed = int(json.loads(progress_path.read_text(encoding="utf-8"))["completed"])
        latent_array = np.lib.format.open_memmap(latent_path, mode="r+")
        label_array = np.lib.format.open_memmap(label_path, mode="r+")
    else:
        latent_array = np.lib.format.open_memmap(
            latent_path,
            mode="w+",
            dtype=np.float32,
            shape=(local_count, 768, 16, 16),
        )
        label_array = np.lib.format.open_memmap(
            label_path, mode="w+", dtype=np.int64, shape=(local_count,)
        )
    if not 0 <= completed <= local_count:
        raise ValueError(f"invalid rank-{rank} cache progress: {completed}")

    subset = FixedAugmentedSubset(
        base,
        all_indices[start + completed : end],
        all_flips[start + completed : end],
        int(args.image_size),
    )
    loader = DataLoader(
        subset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=False,
        persistent_workers=int(args.num_workers) > 0,
    )
    cursor = completed
    started = perf_counter()
    with torch.inference_mode():
        for batch_index, (images, labels) in enumerate(loader, start=1):
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            latents = rae.encode(images)
            if latents.dtype != torch.float32 or tuple(latents.shape[1:]) != (768, 16, 16):
                raise ValueError(f"unexpected RAE latent {latents.dtype} {tuple(latents.shape)}")
            count = len(latents)
            latent_array[cursor : cursor + count] = latents.cpu().numpy()
            label_array[cursor : cursor + count] = labels.numpy()
            cursor += count
            if batch_index % int(args.flush_every) == 0:
                latent_array.flush()
                label_array.flush()
                atomic_json(progress_path, {"completed": cursor, "count": local_count})
                if rank == 0:
                    processed = max(cursor - completed, 1)
                    rate = processed / max(perf_counter() - started, 1e-6)
                    print(
                        f"cache rank0 {cursor}/{local_count} ({cursor / local_count:.1%}) "
                        f"at {rate:.1f} images/s",
                        flush=True,
                    )
    latent_array.flush()
    label_array.flush()
    atomic_json(progress_path, {"completed": cursor, "count": local_count})
    if cursor != local_count:
        raise RuntimeError(f"rank {rank} cached {cursor}, expected {local_count}")
    dist.barrier()

    if rank == 0:
        shards = []
        for shard_rank in range(world_size):
            shard_start, shard_end = split_range(int(args.sample_count), shard_rank, world_size)
            progress = json.loads(
                (output / f"progress-rank{shard_rank:02d}.json").read_text(encoding="utf-8")
            )
            count = shard_end - shard_start
            if int(progress["completed"]) != count:
                raise RuntimeError(f"rank {shard_rank} cache is incomplete")
            shards.append(
                {
                    "rank": shard_rank,
                    "start": shard_start,
                    "count": count,
                    "latents": f"latents-rank{shard_rank:02d}.npy",
                    "labels": f"labels-rank{shard_rank:02d}.npy",
                }
            )
        atomic_json(
            complete_manifest,
            {
                "complete": True,
                "seed": int(args.seed),
                "sample_count": int(args.sample_count),
                "dataset_size": len(base),
                "data_path": str(args.data_path.expanduser()),
                "dtype": "float32",
                "latent_shape": [768, 16, 16],
                "normalization": "official RAE DINOv2 ImageNet-1k statistics",
                "augmentation": "center_crop_256_then_fixed_seed_horizontal_flip",
                "physical_order": "ascending ImageNet index for sequential parquet reads",
                "logical_order": "seeded random subset order restored by stream_order.npy",
                "stream_order": "stream_order.npy",
                "source_config": str(args.config.resolve()),
                "world_size": world_size,
                "shards": shards,
            },
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
