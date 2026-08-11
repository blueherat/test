from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from evaluate_imagenet100_sit_fid import center_crop_arr  # noqa: E402
from sample_imagenet100_sit_fid import (  # noqa: E402
    GIB,
    cuda_allocator_fraction,
    decode_latents_in_chunks,
    official_pixel_quantization,
    official_rank_seed,
    official_total_samples,
)


class FakeVAE(torch.nn.Module):
    def decode(self, latents: torch.Tensor):
        return type("Decoded", (), {"sample": latents * 2.0})()


def test_official_rank_seed_matches_sit_formula() -> None:
    assert [official_rank_seed(3, 4, rank) for rank in range(4)] == [12, 13, 14, 15]


def test_official_total_samples_rounds_to_global_batch() -> None:
    assert official_total_samples(5_000, 64, 4) == 5_120


def test_cuda_allocator_fraction_enforces_absolute_budget() -> None:
    assert cuda_allocator_fraction(7.5, 24 * GIB) == 7.5 / 24
    for limit, total in ((0.0, 24 * GIB), (24.0, 24 * GIB)):
        try:
            cuda_allocator_fraction(limit, total)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid CUDA allocator budget must be rejected")


def test_official_pixel_quantization_matches_formula() -> None:
    images = torch.tensor([[[[-1.0, 0.0, 1.0]]]]).expand(1, 3, 1, 3)
    result = official_pixel_quantization(images)
    assert result.shape == (1, 1, 3, 3)
    assert result[0, 0, 0].tolist() == [0, 0, 0]
    assert result[0, 0, 1].tolist() == [128, 128, 128]
    assert result[0, 0, 2].tolist() == [255, 255, 255]


def test_chunked_vae_decode_preserves_order_and_values() -> None:
    latents = torch.arange(7 * 4, dtype=torch.float32).reshape(7, 1, 2, 2)
    decoded = decode_latents_in_chunks(
        FakeVAE(), latents, scaling_factor=2.0, chunk_size=3
    )
    assert torch.equal(decoded, latents)


def test_chunked_vae_decode_rejects_nonpositive_chunk() -> None:
    latents = torch.zeros(1, 1, 1, 1)
    try:
        decode_latents_in_chunks(
            FakeVAE(), latents, scaling_factor=1.0, chunk_size=0
        )
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("nonpositive VAE decode chunks must be rejected")


def test_center_crop_arr_matches_requested_shape() -> None:
    image = Image.fromarray(np.zeros((600, 800, 3), dtype=np.uint8))
    cropped = center_crop_arr(image, 256)
    assert cropped.shape == (256, 256, 3)
    assert cropped.dtype == np.uint8
