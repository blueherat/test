from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from experiments.rae_latent_cache import CachedRAELatentDataset


def _cache(root: Path) -> Path:
    root.mkdir()
    latents = np.arange(8, dtype=np.float32).reshape(8, 1, 1, 1)
    labels = np.arange(8, dtype=np.int64)
    np.save(root / "latents.npy", latents)
    np.save(root / "labels.npy", labels)
    np.save(root / "stream.npy", np.array([7, 5, 3, 1, 6, 4, 2, 0]))
    manifest = {
        "complete": True,
        "dtype": "float32",
        "sample_count": 8,
        "stream_order": "stream.npy",
        "shards": [{"count": 8, "latents": "latents.npy", "labels": "labels.npy"}],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_seeded_cache_order_is_deterministic_and_slice_consistent(tmp_path: Path) -> None:
    path = _cache(tmp_path / "cache")
    first = CachedRAELatentDataset(path, order_seed=17)
    repeat = CachedRAELatentDataset(path, order_seed=17)
    other = CachedRAELatentDataset(path, order_seed=19)
    first_labels = [first[index][1] for index in range(len(first))]
    assert first_labels == [repeat[index][1] for index in range(len(repeat))]
    assert first_labels != [other[index][1] for index in range(len(other))]

    suffix = CachedRAELatentDataset(path, start=3, order_seed=17)
    assert [suffix[index][1] for index in range(len(suffix))] == first_labels[3:]
    assert sorted(first_labels) == list(range(8))
    assert torch.equal(first[0][0], repeat[0][0])
