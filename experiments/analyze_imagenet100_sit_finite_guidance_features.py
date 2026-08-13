#!/usr/bin/env python3
"""Decode finite-guidance endpoints and measure paired feature responses.

This is a mechanism diagnostic, not an FID evaluator.  It keeps decoded pixels
continuous and uses ImageNet Inception-v3 pre-logit features so the small
central difference used by the linearity audit is not dominated by uint8
quantization.  Selected generation conditions must still be evaluated with the
repository's ADM FID-5K pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import Inception_V3_Weights, inception_v3
from torchvision.utils import make_grid, save_image

try:
    from experiments.finite_guidance_dynamics import sample_cosine, sample_rms
    from experiments.sample_imagenet100_sit_fid import decode_latents_in_chunks
    from experiments.train_imagenet100_sit_flow import (
        SD_VAE_SCALING_FACTOR,
        atomic_json_dump,
    )
except ModuleNotFoundError:
    from finite_guidance_dynamics import sample_cosine, sample_rms
    from sample_imagenet100_sit_fid import decode_latents_in_chunks
    from train_imagenet100_sit_flow import SD_VAE_SCALING_FACTOR, atomic_json_dump


def _summary(values: torch.Tensor) -> dict[str, float]:
    array = values.detach().float().cpu().numpy().astype(np.float64, copy=False)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "q05": float(np.quantile(array, 0.05)),
        "median": float(np.quantile(array, 0.50)),
        "q95": float(np.quantile(array, 0.95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _gamma_index(gammas: torch.Tensor, target: float, *, atol: float = 1e-6) -> int:
    distances = (gammas.float() - float(target)).abs()
    index = int(distances.argmin().item())
    if float(distances[index]) > atol:
        raise ValueError(f"gamma {target} is missing from {gammas.tolist()}")
    return index


def _gamma_key(gamma: float) -> str:
    sign = "m" if gamma < 0 else "p"
    magnitude = f"{abs(float(gamma)):.8f}".rstrip("0").rstrip(".")
    return f"g{sign}{magnitude.replace('.', 'p')}"


def load_trajectory_shards(run_dir: Path) -> tuple[dict[str, torch.Tensor], dict]:
    """Load and validate one completed trajectory run in sample order."""

    run_dir = run_dir.expanduser().resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    paths = sorted((run_dir / "trajectory_shards").glob("batch_*.pt"))
    if not paths:
        raise FileNotFoundError(f"no trajectory shards under {run_dir}")
    shards = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    cursor = 0
    reference_gammas = shards[0]["gammas"].float()
    for path, shard in zip(paths, shards, strict=True):
        if int(shard["sample_start"]) != cursor:
            raise ValueError(f"non-contiguous sample range in {path}")
        cursor = int(shard["sample_stop"])
        if not torch.equal(shard["gammas"].float(), reference_gammas):
            raise ValueError(f"gamma grid changed in {path}")
    if cursor != int(manifest["num_samples"]):
        raise ValueError(f"loaded {cursor} samples, expected {manifest['num_samples']}")

    payload: dict[str, torch.Tensor] = {
        "gammas": reference_gammas,
        "labels": torch.cat([shard["labels"] for shard in shards]),
        "baseline": torch.cat([shard["baseline"] for shard in shards]),
    }
    study = str(manifest["study"])
    if study == "linearity":
        payload["tangent"] = torch.cat([shard["tangent"] for shard in shards])
        payload["endpoints"] = torch.cat(
            [shard["endpoints"] for shard in shards], dim=1
        )
    elif study == "feedback":
        payload["frozen"] = torch.cat([shard["frozen"] for shard in shards], dim=1)
        payload["closed"] = torch.cat([shard["closed"] for shard in shards], dim=1)
    else:
        raise ValueError(f"feature analysis does not support study={study!r}")
    return payload, manifest


def latent_conditions(
    payload: dict[str, torch.Tensor], manifest: dict
) -> dict[str, torch.Tensor]:
    """Give every decoded condition a stable, self-describing name."""

    conditions = {"baseline": payload["baseline"]}
    gammas = payload["gammas"].tolist()
    if manifest["study"] == "linearity":
        for gamma, endpoint in zip(gammas, payload["endpoints"], strict=True):
            conditions[f"closed_{_gamma_key(float(gamma))}"] = endpoint
    else:
        for gamma, frozen, closed in zip(
            gammas, payload["frozen"], payload["closed"], strict=True
        ):
            key = _gamma_key(float(gamma))
            conditions[f"frozen_{key}"] = frozen
            conditions[f"closed_{key}"] = closed
    return conditions


class ContinuousInceptionProbe:
    """ImageNet Inception-v3 pre-logit features for continuous [0, 1] images."""

    def __init__(self, device: torch.device) -> None:
        self.model = (
            inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
            .to(device=device, dtype=torch.float32)
            .requires_grad_(False)
            .eval()
        )
        self.device = device

    @torch.inference_mode()
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        resized = F.interpolate(
            images.float(), size=(299, 299), mode="bilinear", align_corners=False
        )
        mean = resized.new_tensor((0.485, 0.456, 0.406)).reshape(1, 3, 1, 1)
        std = resized.new_tensor((0.229, 0.224, 0.225)).reshape(1, 3, 1, 1)
        captured: list[torch.Tensor] = []

        def capture(_module, _inputs, output):
            captured.append(output.flatten(1))

        handle = self.model.avgpool.register_forward_hook(capture)
        try:
            self.model((resized - mean) / std)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError(f"expected one feature tensor, got {len(captured)}")
        return captured[0].float()


@torch.inference_mode()
def extract_condition_features(
    conditions: dict[str, torch.Tensor],
    *,
    device: torch.device,
    decode_batch: int,
    preview_count: int,
    preview_path: Path,
) -> dict[str, torch.Tensor]:
    """Decode all conditions and extract paired continuous-image features."""

    from diffusers.models import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse", local_files_only=True
    )
    vae = vae.to(device=device, dtype=torch.float32).requires_grad_(False).eval()
    probe = ContinuousInceptionProbe(device)
    features: dict[str, torch.Tensor] = {}
    preview_rows: list[torch.Tensor] = []
    for name, latents in conditions.items():
        parts: list[torch.Tensor] = []
        first_images: torch.Tensor | None = None
        for start in range(0, len(latents), decode_batch):
            batch = latents[start : start + decode_batch].to(device=device, dtype=torch.float32)
            decoded = decode_latents_in_chunks(
                vae,
                batch,
                scaling_factor=SD_VAE_SCALING_FACTOR,
                chunk_size=decode_batch,
            )
            images = decoded.add(1).mul(0.5).clamp(0, 1)
            if first_images is None:
                first_images = images[:preview_count].cpu()
            parts.append(probe(images).cpu())
        features[name] = torch.cat(parts)
        if first_images is not None:
            preview_rows.append(first_images)
        print(json.dumps({"event": "features", "condition": name}), flush=True)

    if preview_rows and preview_count > 0:
        preview = torch.cat(preview_rows)
        grid = make_grid(preview, nrow=preview_count, padding=2)
        save_image(grid, preview_path)
    return features


def response_metrics(
    actual: torch.Tensor, predicted: torch.Tensor
) -> dict[str, torch.Tensor]:
    residual = actual - predicted
    tiny = torch.finfo(actual.dtype).tiny
    return {
        "actual_rms": sample_rms(actual),
        "predicted_rms": sample_rms(predicted),
        "residual_rms": sample_rms(residual),
        "relative_residual": sample_rms(residual) / sample_rms(actual).clamp_min(tiny),
        "cosine": sample_cosine(actual, predicted),
        "magnitude_ratio": sample_rms(actual) / sample_rms(predicted).clamp_min(tiny),
    }


def aggregate_feature_metrics(
    features: dict[str, torch.Tensor], manifest: dict
) -> tuple[list[dict[str, object]], dict]:
    """Aggregate feature-space linearity or feedback response metrics."""

    gammas = torch.tensor(manifest["gammas"], dtype=torch.float32)
    # Linearity runs save +/- central_delta in addition to manifest gammas.
    if manifest["study"] == "linearity":
        delta = float(manifest["central_delta"])
        minus = features[f"closed_{_gamma_key(-delta)}"]
        plus = features[f"closed_{_gamma_key(delta)}"]
        zero = features[f"closed_{_gamma_key(0.0)}"]
        tangent = (plus - minus) / (2.0 * delta)
        rows: list[dict[str, object]] = []
        for gamma in gammas.tolist():
            endpoint = features[f"closed_{_gamma_key(float(gamma))}"]
            metrics = response_metrics(endpoint - zero, float(gamma) * tangent)
            row: dict[str, object] = {"gamma": float(gamma)}
            for name, values in metrics.items():
                for statistic, value in _summary(values).items():
                    row[f"feature_{name}_{statistic}"] = value
            rows.append(row)
        gamma_one = next(
            (row for row in rows if math.isclose(float(row["gamma"]), 1.0)), None
        )
        return rows, {
            "feature_linearity_at_gamma_one": gamma_one,
            "feature_tangent_definition": "central difference at +/- central_delta",
        }

    baseline = features["baseline"]
    rows = []
    for gamma in gammas.tolist():
        if math.isclose(float(gamma), 0.0):
            continue
        key = _gamma_key(float(gamma))
        frozen_response = features[f"frozen_{key}"] - baseline
        closed_response = features[f"closed_{key}"] - baseline
        feedback = closed_response - frozen_response
        tiny = torch.finfo(closed_response.dtype).tiny
        metrics = {
            "response_cosine": sample_cosine(frozen_response, closed_response),
            "frozen_over_closed_rms": sample_rms(frozen_response)
            / sample_rms(closed_response).clamp_min(tiny),
            "feedback_over_closed_rms": sample_rms(feedback)
            / sample_rms(closed_response).clamp_min(tiny),
            "frozen_response_rms": sample_rms(frozen_response),
            "closed_response_rms": sample_rms(closed_response),
        }
        row = {"gamma": float(gamma)}
        for name, values in metrics.items():
            for statistic, value in _summary(values).items():
                row[f"feature_{name}_{statistic}"] = value
        rows.append(row)
    gamma_one = next(
        (row for row in rows if math.isclose(float(row["gamma"]), 1.0)), None
    )
    return rows, {"feature_feedback_at_gamma_one": gamma_one}


def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.decode_batch <= 0 or args.preview_count < 0:
        raise ValueError("invalid decode batch or preview count")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    payload, manifest = load_trajectory_shards(args.run_dir)
    conditions = latent_conditions(payload, manifest)
    output_dir = args.run_dir.expanduser().resolve() / "decoded_feature_response"
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    features = extract_condition_features(
        conditions,
        device=device,
        decode_batch=args.decode_batch,
        preview_count=args.preview_count,
        preview_path=output_dir / "paired_previews.png",
    )
    names = list(features)
    np.savez_compressed(
        output_dir / "continuous_inception_features.npz",
        names=np.asarray(names),
        features=np.stack([features[name].numpy() for name in names]),
        labels=payload["labels"].numpy(),
    )
    rows, summary = aggregate_feature_metrics(features, manifest)
    _write_csv(rows, output_dir / "metrics.csv")
    summary.update(
        {
            "format": "eqvae_sit400_finite_guidance_feature_response_v1",
            "source_run": str(args.run_dir.expanduser().resolve()),
            "study": manifest["study"],
            "direction": manifest["direction"],
            "num_samples": int(manifest["num_samples"]),
            "feature_backend": "torchvision Inception-v3 ImageNet pre-logit, continuous pixels",
            "is_fid": False,
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    atomic_json_dump(summary, output_dir / "summary.json")
    print(json.dumps({"event": "complete", **summary}, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--decode-batch", type=int, default=4)
    parser.add_argument("--preview-count", type=int, default=4)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
