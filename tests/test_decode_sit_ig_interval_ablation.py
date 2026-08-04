from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.decode_sit_ig_interval_ablation import (
    decode_latents,
    merge_shards,
    parse_condition_names,
)


class IdentityVAE:
    def decode(self, latent):
        return SimpleNamespace(sample=latent[:, :3])


def test_condition_selection_defaults_and_validates():
    assert parse_condition_names("", ["a", "b"]) == ["a", "b"]
    assert parse_condition_names("b,a", ["a", "b"]) == ["b", "a"]
    with pytest.raises(ValueError, match="unknown"):
        parse_condition_names("c", ["a", "b"])


def test_decode_latents_applies_official_scale_and_uint8_mapping():
    latent = torch.zeros((1, 4, 2, 2), dtype=torch.float32)
    image = decode_latents(IdentityVAE(), latent)
    assert image.dtype == np.uint8
    assert image.shape == (1, 2, 2, 3)
    assert np.all(image == 127)


def test_merge_shards_restores_global_interleaving(tmp_path):
    samples, world_size = 4, 2
    for rank, values in enumerate(((10, 30), (20, 40))):
        shard = np.full((2, 256, 256, 3), 0, dtype=np.uint8)
        shard[0] = values[0]
        shard[1] = values[1]
        np.save(tmp_path / f".test_rank{rank:02d}.npy", shard)
        np.save(
            tmp_path / f".test_progress_rank{rank:02d}.npy",
            np.ones(2, dtype=np.bool_),
        )
    result = merge_shards(
        output_dir=tmp_path,
        condition_name="test",
        samples=samples,
        world_size=world_size,
    )
    with np.load(result) as data:
        images = data["arr_0"]
    assert images[:, 0, 0, 0].tolist() == [10, 20, 30, 40]
    assert not list(tmp_path.glob(".test_rank*.npy"))
