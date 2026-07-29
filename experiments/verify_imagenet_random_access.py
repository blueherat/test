"""Verify exact parity and benchmark ImageNet random-access data loading."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.raev2_training_core import (  # noqa: E402
    DeterministicImageNetPacked,
    DeterministicImageNetParquet,
    tensor_fingerprint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--packed", type=Path, required=True)
    parser.add_argument("--index-map", type=Path, required=True)
    parser.add_argument("--verify-samples", type=int, default=128)
    parser.add_argument("--benchmark-samples", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def deterministic_indices(length: int, count: int, seed: int) -> list[int]:
    if count <= 0:
        raise ValueError("sample count must be positive")
    rng = np.random.default_rng(seed)
    anchors = [0, length - 1, length // 2]
    random_count = max(count - len(anchors), 0)
    random_indices = rng.choice(
        length,
        size=min(random_count, length),
        replace=False,
    ).tolist()
    return (anchors + random_indices)[:count]


def benchmark(dataset, indices: list[int], workers: int) -> dict[str, float | int]:
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        multiprocessing_context="spawn" if workers > 0 else None,
    )
    started = perf_counter()
    count = 0
    for images, labels, sample_indices in loader:
        if images.shape[0] != labels.shape[0] or labels.shape[0] != sample_indices.shape[0]:
            raise RuntimeError("benchmark loader returned inconsistent batch shapes")
        count += int(images.shape[0])
    elapsed = perf_counter() - started
    return {
        "samples": count,
        "seconds": elapsed,
        "samples_per_second": count / elapsed,
    }


def main() -> None:
    args = parse_args()
    common = {
        "split": "train",
        "image_size": 256,
        "augmentation_seed": 42,
        "horizontal_flip": False,
        "index_map_path": args.index_map,
    }
    parquet = DeterministicImageNetParquet(args.parquet, **common)
    packed = DeterministicImageNetPacked(args.packed, **common)
    if len(parquet) != len(packed):
        raise ValueError(f"dataset lengths differ: {len(parquet)} != {len(packed)}")

    verification_indices = deterministic_indices(
        len(parquet),
        args.verify_samples,
        args.seed,
    )
    digest = __import__("hashlib").sha256()
    for position, index in enumerate(verification_indices, start=1):
        parquet_image, parquet_label, parquet_index = parquet[index]
        packed_image, packed_label, packed_index = packed[index]
        if parquet_index != packed_index or parquet_index != index:
            raise AssertionError(f"index mismatch at {index}")
        if parquet_label != packed_label:
            raise AssertionError(
                f"label mismatch at {index}: {parquet_label} != {packed_label}"
            )
        if not torch.equal(parquet_image, packed_image):
            difference = float((parquet_image - packed_image).abs().max())
            raise AssertionError(
                f"image tensor mismatch at {index}; max_abs_difference={difference}"
            )
        digest.update(index.to_bytes(8, "little"))
        digest.update(bytes.fromhex(tensor_fingerprint(packed_image)))
        if position % 16 == 0:
            print(f"verified {position}/{len(verification_indices)}", flush=True)

    benchmark_indices = deterministic_indices(
        len(packed),
        args.benchmark_samples,
        args.seed + 1,
    )
    result = {
        "verified_samples": len(verification_indices),
        "verification_sha256": digest.hexdigest(),
        "packed_benchmark": benchmark(packed, benchmark_indices, args.workers),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
