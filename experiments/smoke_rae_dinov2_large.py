"""Validate the converted official DINOv2-L RAE on real ImageNet images."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torchvision.transforms import functional as TF
from torchvision.utils import save_image


ROOT = Path(__file__).resolve().parents[1]
RAE_SRC = ROOT / "external" / "RAE" / "src"
for path in (ROOT, RAE_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.rae_teacher_rollout_gap import load_frozen_decoder  # noqa: E402
from experiments.train_rae_layerwise_path import (  # noqa: E402
    configure_determinism,
    resolve_stage1_paths,
)
from utils.model_utils import instantiate_from_config  # noqa: E402
from utils.train_utils import ParquetImageNetDataset, center_crop_arr  # noqa: E402


DEFAULT_CONFIG = ROOT / "experiments/configs/rae_strict_lpl_ditdh_s_dinov2_large.yaml"
DEFAULT_OUTPUT = (
    Path.home() / "data/eqvae/experiments/rae_dinov2_large/smoke_validation"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-path", type=Path, default=Path("/data/shared/imagenet-1k"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count < 1:
        raise ValueError("count must be positive")
    configure_determinism(args.seed)
    device = torch.device(args.device)

    config = OmegaConf.load(args.config.expanduser())
    resolve_stage1_paths(config)
    stage_1 = OmegaConf.create(OmegaConf.to_container(config.stage_1, resolve=True))
    rae = instantiate_from_config(stage_1)
    rae = rae.to(device=device, dtype=torch.float32).requires_grad_(False).eval()

    dataset = ParquetImageNetDataset(args.data_path.expanduser(), split="validation")
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(dataset), generator=generator)[: args.count].tolist()
    images = []
    labels = []
    for index in indices:
        image, label = dataset[index]
        images.append(TF.to_tensor(center_crop_arr(image, 256)))
        labels.append(int(label))
    x = torch.stack(images).to(device=device, dtype=torch.float32)

    with torch.inference_mode():
        z = rae.encode(x)
        reconstruction = rae.decode(z)
    if tuple(z.shape) != (args.count, 1024, 16, 16):
        raise RuntimeError(f"unexpected latent shape {tuple(z.shape)}")
    if z.dtype != torch.float32 or not torch.isfinite(z).all():
        raise RuntimeError("large RAE produced an invalid latent")
    if tuple(reconstruction.shape) != (args.count, 3, 256, 256):
        raise RuntimeError(f"unexpected reconstruction shape {tuple(reconstruction.shape)}")
    if reconstruction.dtype != torch.float32 or not torch.isfinite(reconstruction).all():
        raise RuntimeError("large RAE produced an invalid reconstruction")

    decoder_only = load_frozen_decoder(stage_1)
    decoder_only = (
        decoder_only.to(device=device, dtype=torch.float32).requires_grad_(False).eval()
    )
    with torch.inference_mode():
        decoder_only_reconstruction = decoder_only(z)
    decoder_max_abs = float(
        (decoder_only_reconstruction - reconstruction).abs().max().item()
    )
    if decoder_max_abs > 2e-6:
        raise RuntimeError(
            "full RAE and decoder-only paths disagree: "
            f"max_abs={decoder_max_abs:.8g}"
        )

    clipped = reconstruction.clamp(0.0, 1.0)
    per_sample_mse = F.mse_loss(clipped, x, reduction="none").flatten(1).mean(1)
    per_sample_psnr = -10.0 * torch.log10(per_sample_mse.clamp_min(1e-12))
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    save_image(
        torch.stack((x.cpu(), clipped.cpu()), dim=1).flatten(0, 1),
        output / "input_reconstruction.png",
        nrow=2,
    )
    summary = {
        "config": str(args.config.expanduser().resolve()),
        "data_path": str(args.data_path.expanduser().resolve()),
        "indices": indices,
        "labels": labels,
        "device": str(device),
        "dtype": str(z.dtype),
        "latent_shape": list(z.shape),
        "latent_mean": float(z.mean()),
        "latent_std": float(z.std(unbiased=False)),
        "reconstruction_shape": list(reconstruction.shape),
        "decoder_only_max_abs": decoder_max_abs,
        "mse": [float(value) for value in per_sample_mse],
        "psnr_db": [float(value) for value in per_sample_psnr],
        "mean_mse": float(per_sample_mse.mean()),
        "mean_psnr_db": float(per_sample_psnr.mean()),
        "finite": bool(
            torch.isfinite(z).all()
            and torch.isfinite(reconstruction).all()
            and math.isfinite(decoder_max_abs)
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(output / "input_reconstruction.png")


if __name__ == "__main__":
    main()
