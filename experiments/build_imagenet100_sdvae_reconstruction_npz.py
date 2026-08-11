"""Decode ImageNet-100 validation posterior means for an ADM rFID floor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import save_image

from sample_imagenet100_sit_fid import DEFAULT_OUTPUT_DIR, official_pixel_quantization
from train_imagenet100_sit_flow import atomic_json_dump, sha256_file


DEFAULT_MOMENTS = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/"
    "imagenet100_cmc_sdvae/validation_moments.npy"
)


@torch.inference_mode()
def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    moments_path = Path(args.moments).expanduser().resolve()
    moments = np.load(moments_path, mmap_mode="r", allow_pickle=False)
    if moments.shape != (5_000, 8, 32, 32) or moments.dtype != np.float32:
        raise ValueError(f"unexpected moments: {moments.shape}/{moments.dtype}")

    from diffusers.models import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse", local_files_only=True
    )
    vae.to(device).eval().requires_grad_(False)
    images = np.empty((5_000, 256, 256, 3), dtype=np.uint8)
    preview: torch.Tensor | None = None
    for start in range(0, len(moments), args.batch_size):
        stop = min(start + args.batch_size, len(moments))
        posterior_mean = torch.from_numpy(
            np.asarray(moments[start:stop, :4]).copy()
        ).to(device)
        decoded = vae.decode(posterior_mean).sample
        images[start:stop] = official_pixel_quantization(decoded)
        if preview is None:
            preview = decoded[:16].detach().cpu()
        if stop == len(moments) or stop % args.log_every == 0:
            print(json.dumps({"event": "decoded", "count": stop}), flush=True)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "reference_sdvae_posterior_mean_n5000.npz"
    np.savez(output_path, arr_0=images)
    assert preview is not None
    save_image(
        preview,
        output_dir / "reference_sdvae_posterior_mean_preview.png",
        nrow=4,
        normalize=True,
        value_range=(-1, 1),
    )
    manifest = {
        "format": "eqvae_imagenet100_sdvae_reconstruction_npz_v1",
        "scope": "diagnostic tokenizer reconstruction floor, not generation FID",
        "moments": str(moments_path),
        "moments_sha256": sha256_file(moments_path),
        "posterior_statistic": "mean (channels 0:4)",
        "vae": "stabilityai/sd-vae-ft-mse",
        "latent_scaling": "none; cached posterior moments are in raw VAE coordinates",
        "pixel_quantization": "official SiT clamp(127.5*x + 128, 0, 255)",
        "count": 5_000,
        "output": str(output_path),
    }
    atomic_json_dump(manifest, output_dir / "sdvae_reconstruction_manifest.json")
    print(json.dumps({"event": "complete", "output": str(output_path)}), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moments", type=Path, default=DEFAULT_MOMENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
