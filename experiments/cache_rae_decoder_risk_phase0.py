"""Build leakage-controlled ImageNet latents for the decoder-risk Phase-0 audit.

Calibration images come from ImageNet train but exclude every image used by the
10k static-path latent stream.  The held-out test split comes from ImageNet
validation, which the stage-2 continuation never consumed.
"""

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

from experiments.rae_latent_cache import split_range  # noqa: E402
from experiments.train_rae_layerwise_path import (  # noqa: E402
    configure_determinism,
    resolve_stage1_paths,
)
from experiments.train_rae_spectral_tiny import setup_distributed  # noqa: E402
from utils.model_utils import instantiate_from_config  # noqa: E402
from utils.train_utils import ParquetImageNetDataset, center_crop_arr  # noqa: E402


DEFAULT_SOURCE_CACHE = (
    Path.home()
    / "data/eqvae/cache/rae_layerwise_path_streams/seed3407_n160000_fp32"
)
DEFAULT_OUTPUT = (
    Path.home()
    / "data/eqvae/cache/rae_decoder_risk_phase0/seed20260718_cal1024_test2048_fp32"
)
DEFAULT_CONFIG = (
    Path.home()
    / "data/eqvae/experiments/rae_layerwise_path_train"
    / "seed3407_static_rank16_s0_to_10000/config.yaml"
)


class Phase0ImageDataset(Dataset):
    def __init__(
        self,
        train: Dataset,
        validation: Dataset,
        calibration_indices: np.ndarray,
        test_indices: np.ndarray,
        *,
        start: int,
        stop: int,
        image_size: int,
    ) -> None:
        self.train = train
        self.validation = validation
        self.calibration_indices = calibration_indices
        self.test_indices = test_indices
        self.start = int(start)
        self.stop = int(stop)
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return self.stop - self.start

    def __getitem__(self, local_index: int) -> tuple[torch.Tensor, int]:
        position = self.start + int(local_index)
        calibration_count = len(self.calibration_indices)
        if position < calibration_count:
            image, label = self.train[int(self.calibration_indices[position])]
        else:
            test_position = position - calibration_count
            image, label = self.validation[int(self.test_indices[test_position])]
        image = center_crop_arr(image, self.image_size)
        return TF.to_tensor(image), int(label)


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def prepare_indices(
    output: Path,
    *,
    train_size: int,
    validation_size: int,
    source_cache: Path,
    calibration_count: int,
    test_count: int,
    seed: int,
) -> None:
    used = np.load(source_cache / "indices.npy", mmap_mode="r")
    excluded = np.zeros(int(train_size), dtype=np.bool_)
    excluded[np.asarray(used, dtype=np.int64)] = True
    generator = np.random.default_rng(int(seed))
    train_order = generator.permutation(int(train_size))
    calibration = train_order[~excluded[train_order]][: int(calibration_count)]
    if len(calibration) != int(calibration_count):
        raise RuntimeError("not enough unused ImageNet training images")
    test = generator.permutation(int(validation_size))[: int(test_count)]
    if len(test) != int(test_count):
        raise RuntimeError("not enough ImageNet validation images")
    np.save(output / "calibration_indices.npy", calibration.astype(np.int64))
    np.save(output / "test_indices.npy", test.astype(np.int64))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-path", type=Path, default=Path("/data/shared/imagenet-1k"))
    parser.add_argument("--source-cache", type=Path, default=DEFAULT_SOURCE_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--calibration-count", type=int, default=1024)
    parser.add_argument("--test-count", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20_260_718)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank, world_size, device = setup_distributed()
    configure_determinism(int(args.seed) * world_size + rank)
    output = args.output.expanduser().resolve()
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if bool(manifest.get("complete")):
            if rank == 0:
                print(f"Phase-0 latent cache already complete: {output}", flush=True)
            dist.destroy_process_group()
            return

    train = ParquetImageNetDataset(args.data_path, split="train", transform=None)
    validation = ParquetImageNetDataset(args.data_path, split="validation", transform=None)
    selection_paths = (output / "calibration_indices.npy", output / "test_indices.npy")
    if rank == 0 and not all(path.exists() for path in selection_paths):
        prepare_indices(
            output,
            train_size=len(train),
            validation_size=len(validation),
            source_cache=args.source_cache.expanduser(),
            calibration_count=args.calibration_count,
            test_count=args.test_count,
            seed=args.seed,
        )
    dist.barrier()
    calibration_indices = np.load(selection_paths[0], mmap_mode="r")
    test_indices = np.load(selection_paths[1], mmap_mode="r")
    total = len(calibration_indices) + len(test_indices)
    start, stop = split_range(total, rank, world_size)
    dataset = Phase0ImageDataset(
        train,
        validation,
        calibration_indices,
        test_indices,
        start=start,
        stop=stop,
        image_size=args.image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    config = OmegaConf.load(args.config.expanduser())
    resolve_stage1_paths(config)
    latent_shape = tuple(int(value) for value in config.misc.latent_size)
    if len(latent_shape) != 3:
        raise ValueError(f"expected CHW latent shape, got {latent_shape}")
    rae = instantiate_from_config(config.stage_1)
    rae.decoder = torch.nn.Identity()
    rae = rae.to(device=device, dtype=torch.float32).requires_grad_(False).eval()

    local_count = stop - start
    latent_path = output / f"latents-rank{rank:02d}.npy"
    label_path = output / f"labels-rank{rank:02d}.npy"
    latents = np.lib.format.open_memmap(
        latent_path, mode="w+", dtype=np.float32, shape=(local_count, *latent_shape)
    )
    labels = np.lib.format.open_memmap(
        label_path, mode="w+", dtype=np.int64, shape=(local_count,)
    )
    cursor = 0
    started = perf_counter()
    with torch.inference_mode():
        for images, batch_labels in loader:
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            latent = rae.encode(images)
            if latent.dtype != torch.float32 or tuple(latent.shape[1:]) != latent_shape:
                raise ValueError(f"unexpected RAE latent {latent.dtype} {tuple(latent.shape)}")
            count = len(latent)
            latents[cursor : cursor + count] = latent.cpu().numpy()
            labels[cursor : cursor + count] = batch_labels.numpy()
            cursor += count
            if rank == 0 and cursor % max(args.batch_size * 8, 1) == 0:
                rate = cursor / max(perf_counter() - started, 1e-6)
                print(f"phase0 cache rank0 {cursor}/{local_count} at {rate:.1f} img/s", flush=True)
    latents.flush()
    labels.flush()
    if cursor != local_count:
        raise RuntimeError(f"rank {rank} wrote {cursor}, expected {local_count}")
    dist.barrier()

    if rank == 0:
        shards = []
        for shard_rank in range(world_size):
            shard_start, shard_stop = split_range(total, shard_rank, world_size)
            shards.append(
                {
                    "rank": shard_rank,
                    "start": shard_start,
                    "count": shard_stop - shard_start,
                    "latents": f"latents-rank{shard_rank:02d}.npy",
                    "labels": f"labels-rank{shard_rank:02d}.npy",
                }
            )
        atomic_json(
            manifest_path,
            {
                "complete": True,
                "seed": int(args.seed),
                "sample_count": int(total),
                "dataset_size": int(total),
                "data_path": str(args.data_path.expanduser()),
                "dtype": "float32",
                "latent_shape": list(latent_shape),
                "normalization": (
                    "official RAE ImageNet-1k statistics for "
                    f"{config.stage_1.params.encoder_config_path}"
                ),
                "augmentation": "center_crop_256_no_random_augmentation",
                "calibration_count": int(len(calibration_indices)),
                "test_count": int(len(test_indices)),
                "calibration_source": "ImageNet train excluding the 160k continuation cache",
                "test_source": "ImageNet validation",
                "source_cache_exclusion": str(args.source_cache.expanduser()),
                "source_config": str(args.config.expanduser()),
                "world_size": int(world_size),
                "shards": shards,
            },
        )
        print(f"Wrote leakage-controlled Phase-0 cache: {output}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
