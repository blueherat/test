from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.evaluate_imagenette_sdvae_samples import (
    GeneratedTensorDataset,
    reference_cache_name,
)


def test_generated_samples_are_converted_from_minus_one_one_to_uint8(tmp_path: Path) -> None:
    images = torch.tensor([-1.0, 0.0, 1.0]).view(1, 3, 1, 1)
    path = tmp_path / "samples.pt"
    torch.save({"images": images, "labels": torch.tensor([0])}, path)
    dataset = GeneratedTensorDataset(path)
    converted = dataset[0]
    assert converted.dtype == torch.uint8
    assert converted[:, 0, 0].tolist() == [0, 128, 255]


def test_reference_cache_is_resolution_specific(tmp_path: Path) -> None:
    assert reference_cache_name(tmp_path, 128) != reference_cache_name(tmp_path, 256)
