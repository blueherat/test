"""Disk-backed fp32 RAE latent streams for paired generation experiments."""

from __future__ import annotations

import bisect
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def cache_directory(root: Path, seed: int, sample_count: int) -> Path:
    return Path(root).expanduser() / f"seed{int(seed)}_n{int(sample_count)}_fp32"


def split_range(total: int, rank: int, world_size: int) -> tuple[int, int]:
    if total < 0 or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("invalid split arguments")
    start = total * rank // world_size
    end = total * (rank + 1) // world_size
    return start, end


def load_cache_manifest(path: Path) -> dict[str, object]:
    manifest_path = Path(path).expanduser() / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not bool(manifest.get("complete")):
        raise RuntimeError(f"latent cache is incomplete: {manifest_path}")
    if manifest.get("dtype") != "float32":
        raise ValueError(f"latent cache must be float32: {manifest_path}")
    return manifest


class CachedRAELatentDataset(Dataset):
    """Concatenate memory-mapped cache shards without loading them into RAM."""

    def __init__(
        self,
        path: Path,
        *,
        start: int = 0,
        stop: int | None = None,
        order_seed: int | None = None,
    ) -> None:
        self.path = Path(path).expanduser()
        self.manifest = load_cache_manifest(self.path)
        self.shards = list(self.manifest["shards"])
        counts = [int(shard["count"]) for shard in self.shards]
        self.offsets = [0]
        for count in counts:
            self.offsets.append(self.offsets[-1] + count)
        declared = int(self.manifest["sample_count"])
        if self.offsets[-1] != declared:
            raise ValueError(
                f"shard count {self.offsets[-1]} disagrees with manifest {declared}"
            )
        self.start = int(start)
        self.stop = declared if stop is None else int(stop)
        if not 0 <= self.start <= self.stop <= declared:
            raise ValueError(f"invalid cache slice [{self.start}, {self.stop})")
        self._latents: dict[int, np.ndarray] = {}
        self._labels: dict[int, np.ndarray] = {}
        stream_name = self.manifest.get("stream_order")
        self._stream_order_path = self.path / str(stream_name) if stream_name else None
        self._stream_order: np.ndarray | None = None
        self.order_seed = None if order_seed is None else int(order_seed)
        self._order: np.ndarray | None = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_latents"] = {}
        state["_labels"] = {}
        state["_stream_order"] = None
        state["_order"] = None
        return state

    def __len__(self) -> int:
        return self.stop - self.start

    def _arrays(self, shard_index: int) -> tuple[np.ndarray, np.ndarray]:
        if shard_index not in self._latents:
            shard = self.shards[shard_index]
            self._latents[shard_index] = np.load(
                self.path / str(shard["latents"]), mmap_mode="r"
            )
            self._labels[shard_index] = np.load(
                self.path / str(shard["labels"]), mmap_mode="r"
            )
        return self._latents[shard_index], self._labels[shard_index]

    def _physical_index(self, logical_index: int) -> int:
        if self.order_seed is not None:
            if self._order is None:
                declared = int(self.manifest["sample_count"])
                self._order = np.random.default_rng(self.order_seed).permutation(
                    declared
                )
            logical_index = int(self._order[logical_index])
        if self._stream_order_path is None:
            return logical_index
        if self._stream_order is None:
            self._stream_order = np.load(self._stream_order_path, mmap_mode="r")
            if len(self._stream_order) != int(self.manifest["sample_count"]):
                raise ValueError("stream order length disagrees with manifest")
        return int(self._stream_order[logical_index])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        absolute = self._physical_index(self.start + index)
        shard_index = bisect.bisect_right(self.offsets, absolute) - 1
        local = absolute - self.offsets[shard_index]
        latents, labels = self._arrays(shard_index)
        latent = torch.from_numpy(np.array(latents[local], dtype=np.float32, copy=True))
        return latent, int(labels[local])
