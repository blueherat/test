from __future__ import annotations

import json

import numpy as np
import torch

from experiments.rae_latent_cache import CachedRAELatentDataset, split_range


def test_split_range_is_contiguous_and_complete() -> None:
    ranges = [split_range(11, rank, 4) for rank in range(4)]
    assert ranges == [(0, 2), (2, 5), (5, 8), (8, 11)]


def test_cached_latent_dataset_concatenates_and_slices_shards(tmp_path) -> None:
    values = np.arange(5 * 2 * 2 * 2, dtype=np.float32).reshape(5, 2, 2, 2)
    labels = np.arange(5, dtype=np.int64) + 10
    shards = []
    for rank, (start, end) in enumerate(((0, 2), (2, 5))):
        latent_name = f"latents-rank{rank:02d}.npy"
        label_name = f"labels-rank{rank:02d}.npy"
        np.save(tmp_path / latent_name, values[start:end])
        np.save(tmp_path / label_name, labels[start:end])
        shards.append(
            {
                "rank": rank,
                "start": start,
                "count": end - start,
                "latents": latent_name,
                "labels": label_name,
            }
        )
    manifest = {
        "complete": True,
        "dtype": "float32",
        "sample_count": 5,
        "shards": shards,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    dataset = CachedRAELatentDataset(tmp_path, start=1, stop=5)
    assert len(dataset) == 4
    latent, label = dataset[2]
    torch.testing.assert_close(latent, torch.from_numpy(values[3]))
    assert label == 13


def test_cached_latent_dataset_restores_logical_stream_order(tmp_path) -> None:
    values = np.arange(5, dtype=np.float32).reshape(5, 1, 1, 1)
    labels = np.arange(5, dtype=np.int64)
    np.save(tmp_path / "latents.npy", values)
    np.save(tmp_path / "labels.npy", labels)
    np.save(tmp_path / "order.npy", np.array([4, 0, 3, 1, 2], dtype=np.int64))
    manifest = {
        "complete": True,
        "dtype": "float32",
        "sample_count": 5,
        "stream_order": "order.npy",
        "shards": [
            {
                "rank": 0,
                "start": 0,
                "count": 5,
                "latents": "latents.npy",
                "labels": "labels.npy",
            }
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    dataset = CachedRAELatentDataset(tmp_path)
    assert [dataset[index][1] for index in range(5)] == [4, 0, 3, 1, 2]
