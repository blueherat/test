"""Evaluate Imagenette sample tensors with a matched real-image pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.imagenette_sdvae_latent_diffusion import (
    DEFAULT_DATA_ROOT,
    IMAGE_SIZE,
    atomic_json_dump,
    imagenette_transforms,
)


class GeneratedTensorDataset(Dataset):
    """Expose saved ``[-1, 1]`` float samples as torch-fidelity RGB tensors."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        payload = torch.load(self.path, map_location="cpu", weights_only=False)
        images = payload["images"] if isinstance(payload, dict) else payload
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"expected NCHW RGB samples, got {tuple(images.shape)}")
        self.images = images.float().contiguous()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = self.images[int(index)]
        return image.add(1.0).mul(127.5).round().clamp(0, 255).to(torch.uint8)


class ImagenetteValidationDataset(Dataset):
    """Apply the exact validation crop used by the latent-diffusion trainer."""

    def __init__(self, data_root: str | Path, image_size: int = IMAGE_SIZE) -> None:
        _, validation_transform = imagenette_transforms(image_size)
        root = Path(data_root).expanduser().resolve()
        self.dataset = ImageFolder(str(root / "val"), transform=validation_transform)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> torch.Tensor:
        image, _ = self.dataset[int(index)]
        return image.add(1.0).mul(127.5).round().clamp(0, 255).to(torch.uint8)


def reference_cache_name(data_root: str | Path, image_size: int = IMAGE_SIZE) -> str:
    root = str(Path(data_root).expanduser().resolve())
    digest = hashlib.sha256(root.encode("utf-8")).hexdigest()[:12]
    return f"imagenette_val_center_crop_{int(image_size)}_v1_{digest}"


def calculate_distribution_metrics(
    samples: Dataset,
    reference: Dataset,
    *,
    batch_size: int,
    seed: int,
    cache_name: str,
) -> dict[str, float]:
    from torch_fidelity import calculate_metrics

    if len(samples) < 2 or len(reference) < 2:
        raise ValueError("FID requires at least two generated and two reference images")
    kid_subset_size = min(1_000, len(samples), len(reference))
    use_kid = kid_subset_size >= 100
    metrics = calculate_metrics(
        input1=samples,
        input2=reference,
        cuda=torch.cuda.is_available(),
        batch_size=int(batch_size),
        isc=True,
        fid=True,
        kid=use_kid,
        kid_subsets=100 if use_kid else 1,
        kid_subset_size=kid_subset_size,
        rng_seed=int(seed),
        input2_cache_name=cache_name,
        cache=True,
        verbose=True,
    )
    return {str(key): float(value) for key, value in metrics.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    samples = GeneratedTensorDataset(args.samples)
    sample_size = int(samples.images.shape[-1])
    if samples.images.shape[-2] != sample_size:
        raise ValueError(f"expected square samples, got {tuple(samples.images.shape[-2:])}")
    image_size = sample_size if args.image_size is None else int(args.image_size)
    if image_size != sample_size:
        raise ValueError(
            f"requested image size {image_size} does not match generated samples {sample_size}"
        )
    reference = ImagenetteValidationDataset(args.data_root, image_size)
    cache_name = reference_cache_name(args.data_root, image_size)
    metrics = calculate_distribution_metrics(
        samples,
        reference,
        batch_size=args.batch_size,
        seed=args.seed,
        cache_name=cache_name,
    )
    result = {
        "protocol": "imagenette_sdvae_ldm_torch_fidelity_v2",
        "samples": str(Path(args.samples).expanduser().resolve()),
        "sample_count": len(samples),
        "data_root": str(Path(args.data_root).expanduser().resolve()),
        "reference_count": len(reference),
        "image_size": image_size,
        "reference_preprocessing": (
            f"resize_{round(image_size * 1.125)}_center_crop_{image_size}_uint8"
        ),
        "reference_cache_name": cache_name,
        "torch_fidelity": metrics,
    }
    output = Path(args.output).expanduser().resolve()
    atomic_json_dump(result, output)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
