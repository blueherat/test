"""Fixed held-out velocity diagnostics for tiny RAE spectral branches."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
RAE_ROOT = ROOT / "external" / "RAE"
RAE_SRC = RAE_ROOT / "src"
for path in (ROOT, RAE_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.rae_spectral_direction_loss import DCTDirectionLoss
from experiments.rae_spectral_gradient_audit import (
    RAEAuditConfig,
    load_cached_latents,
    load_validation_labels,
    sample_shifted_logit_normal,
)
from utils.model_utils import instantiate_from_config


DEFAULT_RESULTS = Path.home() / "data/eqvae/experiments/rae_spectral_tiny"
DEFAULT_DECODER_SENSITIVITY = (
    Path.home()
    / "data/eqvae/experiments/rae_spectral_gradient_audit_full/decoder_sensitivity.csv"
)


def branch_directories(results: Path) -> list[Path]:
    return sorted(
        path
        for path in results.glob("seed*_*_from_s5000")
        if (path / "manifest.json").exists()
    )


def checkpoint_paths(branch: Path, endpoint_only: bool) -> list[Path]:
    paths = sorted((branch / "checkpoints").glob("step-*.pt"))
    if not paths:
        return []
    return [paths[-1]] if endpoint_only else paths


def load_decoder_weights(path: Path, band_count: int) -> torch.Tensor:
    table = pd.read_csv(path).sort_values("spatial_band")
    if len(table) != band_count:
        raise ValueError(f"expected {band_count} decoder bands, got {len(table)}")
    values = torch.tensor(table["pixel_sensitivity_mean"].to_numpy(), dtype=torch.float64)
    return values / values.mean().clamp_min(1e-12)


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint: Path,
    manifest: dict,
    config: OmegaConf,
    latents: torch.Tensor,
    labels: torch.Tensor,
    loss_module: DCTDirectionLoss,
    decoder_weights: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    evaluation_seed: int,
) -> dict[str, float | int | str]:
    model = instantiate_from_config(config.stage_2).to(device=device, dtype=torch.float32)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["ema"], strict=True)
    model.requires_grad_(False).eval()

    generator = torch.Generator(device=device).manual_seed(int(evaluation_seed))
    raw_sum = torch.zeros((), device=device, dtype=torch.float64)
    band_sum = torch.zeros(loss_module.band_count, device=device, dtype=torch.float64)
    count = 0
    time_shift = math.sqrt(
        float(config.misc.time_dist_shift_dim) / float(config.misc.time_dist_shift_base)
    )
    for start in range(0, len(latents), batch_size):
        clean = latents[start : start + batch_size].to(device)
        batch_labels = labels[start : start + batch_size].to(device)
        noise = torch.randn(clean.shape, device=device, dtype=clean.dtype, generator=generator)
        time = sample_shifted_logit_normal(
            len(clean), time_shift, device=device, generator=generator
        ).to(clean)
        expanded_time = time.reshape(-1, 1, 1, 1)
        noisy = (1.0 - expanded_time) * clean + expanded_time * noise
        target = noise - clean
        prediction = model(noisy, time, y=batch_labels)
        error = prediction - target
        band = loss_module.band_mse(error)
        raw_sum += error.square().flatten(1).mean(1).double().sum()
        band_sum += band.double().sum(0)
        count += len(clean)

    band_mean = band_sum / max(count, 1)
    raw_mse = raw_sum / max(count, 1)
    decoder_weighted = (band_mean * decoder_weights.to(device)).mean()
    high_frequency = band_mean[len(band_mean) // 2 :].mean()
    low_frequency = band_mean[: len(band_mean) // 2].mean()
    row: dict[str, float | int | str] = {
        "branch": str(manifest["experiment_name"]),
        "seed": int(manifest["global_seed"]),
        "treatment": "baseline" if float(manifest["gamma"]) == 0 else "partial",
        "gamma": float(manifest["gamma"]),
        "checkpoint": str(checkpoint),
        "step": int(state["step"]),
        "branch_update": int(state["step"]) - int(manifest["branch_start_step"]),
        "sample_count": int(count),
        "evaluation_seed": int(evaluation_seed),
        "raw_mse": float(raw_mse),
        "decoder_weighted_mse": float(decoder_weighted),
        "low_frequency_mse": float(low_frequency),
        "high_frequency_mse": float(high_frequency),
    }
    for band, value in enumerate(band_mean):
        row[f"band_mse_{band}"] = float(value)
    del model, state
    torch.cuda.empty_cache()
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--evaluation-seed", type=int, default=99173)
    parser.add_argument("--endpoint-only", action="store_true")
    parser.add_argument("--decoder-sensitivity", type=Path, default=DEFAULT_DECODER_SENSITIVITY)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.manual_seed(args.evaluation_seed)
    np.random.seed(args.evaluation_seed)
    device = torch.device(args.device)

    audit_config = RAEAuditConfig(validation_count=int(args.count))
    payload = load_cached_latents(audit_config)
    latents = payload["validation"][: args.count].float()
    labels = load_validation_labels(
        audit_config.dataset_path,
        payload["validation_indices"][: args.count],
    )

    rows = []
    for branch in branch_directories(args.results):
        manifest = json.loads((branch / "manifest.json").read_text(encoding="utf-8"))
        config = OmegaConf.load(branch / "config.yaml")
        spectral = config.training.spectral_direction_loss
        loss_module = DCTDirectionLoss(
            int(spectral.spatial_size),
            list(spectral.second_moments),
            gamma=0.0,
            damping=float(spectral.damping),
            min_weight=float(spectral.min_weight),
            max_weight=float(spectral.max_weight),
        ).to(device)
        decoder_weights = load_decoder_weights(
            args.decoder_sensitivity, loss_module.band_count
        )
        for checkpoint in checkpoint_paths(branch, args.endpoint_only):
            print(f"evaluating {checkpoint}", flush=True)
            rows.append(
                evaluate_checkpoint(
                    checkpoint,
                    manifest,
                    config,
                    latents,
                    labels,
                    loss_module,
                    decoder_weights,
                    device=device,
                    batch_size=args.batch_size,
                    evaluation_seed=args.evaluation_seed,
                )
            )

    if not rows:
        raise RuntimeError("no completed branch checkpoints found")
    table = pd.DataFrame(rows).sort_values(["seed", "treatment", "step"])
    output = args.results / "fixed_validation_metrics.csv"
    table.to_csv(output, index=False)
    metadata = {
        "output": str(output),
        "cache": audit_config.cache_path,
        "split": "cached ImageNet validation",
        "count": int(args.count),
        "evaluation_seed": int(args.evaluation_seed),
        "decoder_sensitivity": str(args.decoder_sensitivity),
        "checkpoint_weights": "ema",
    }
    (args.results / "fixed_validation_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(table.to_string(index=False))
    print(output)


if __name__ == "__main__":
    main()
