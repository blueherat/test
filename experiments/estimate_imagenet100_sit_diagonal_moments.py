"""Estimate SD-VAE posterior marginal moments from the training cache only."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

try:
    from experiments.imagenet100_sit_moment_residual import MOMENT_STATS_FORMAT
    from experiments.train_imagenet100_sit_flow import (
        DEFAULT_CACHE_DIR,
        LATENT_SHAPE,
        MOMENT_SHAPE,
        SD_VAE_SCALING_FACTOR,
        sha256_file,
    )
except ModuleNotFoundError:
    from imagenet100_sit_moment_residual import MOMENT_STATS_FORMAT
    from train_imagenet100_sit_flow import (
        DEFAULT_CACHE_DIR,
        LATENT_SHAPE,
        MOMENT_SHAPE,
        SD_VAE_SCALING_FACTOR,
        sha256_file,
    )


DEFAULT_OUTPUT = DEFAULT_CACHE_DIR / "train_diagonal_moments.pt"


def estimate_diagonal_moments(
    moments_path: Path,
    *,
    batch_size: int,
    scaling_factor: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    moments = np.load(moments_path, mmap_mode="r", allow_pickle=False)
    if moments.dtype != np.float32 or tuple(moments.shape[1:]) != MOMENT_SHAPE:
        raise ValueError(f"unexpected moments array: {moments.shape}/{moments.dtype}")
    total = np.zeros(LATENT_SHAPE, dtype=np.float64)
    second_total = np.zeros(LATENT_SHAPE, dtype=np.float64)
    scale = float(scaling_factor)
    scale_squared = scale * scale
    for start in range(0, len(moments), batch_size):
        batch = np.asarray(moments[start : start + batch_size], dtype=np.float32)
        posterior_mean = batch[:, : LATENT_SHAPE[0]]
        posterior_std = batch[:, LATENT_SHAPE[0] :]
        total += posterior_mean.sum(axis=0, dtype=np.float64) * scale
        second_total += (
            np.square(posterior_mean) + np.square(posterior_std)
        ).sum(axis=0, dtype=np.float64) * scale_squared
    mean = total / len(moments)
    variance = second_total / len(moments) - np.square(mean)
    if not np.isfinite(mean).all() or not np.isfinite(variance).all():
        raise FloatingPointError("non-finite training moments")
    if (variance <= 0).any():
        raise FloatingPointError("non-positive training variance")
    return mean, variance, len(moments)


def main(args: argparse.Namespace) -> None:
    cache_dir = args.cache_dir.expanduser().resolve()
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "eqvae_imagenet100_cmc_sdvae_moments_v1":
        raise ValueError("unsupported cache manifest")
    moments_path = cache_dir / "train_moments.npy"
    mean, variance, count = estimate_diagonal_moments(
        moments_path,
        batch_size=args.batch_size,
        scaling_factor=args.scaling_factor,
    )
    payload = {
        "format": MOMENT_STATS_FORMAT,
        "mean": torch.from_numpy(mean.astype(np.float32)),
        "variance": torch.from_numpy(variance.astype(np.float32)),
        "count": count,
        "split": "train",
        "estimator": "E[x]=E[posterior_mean], E[x^2]=E[mean^2+std^2]",
        "cache_manifest_sha256": sha256_file(manifest_path),
        "scaling_factor": float(args.scaling_factor),
        "source_path": str(moments_path),
        "source_sha256": str(manifest["splits"]["train"]["moments_sha256"]),
        "diagnostics": {
            "mean_min": float(mean.min()),
            "mean_max": float(mean.max()),
            "variance_min": float(variance.min()),
            "variance_max": float(variance.max()),
            "variance_mean": float(variance.mean()),
        },
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)
    print(json.dumps({"output": str(output), **payload["diagnostics"]}, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--scaling-factor", type=float, default=SD_VAE_SCALING_FACTOR)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
