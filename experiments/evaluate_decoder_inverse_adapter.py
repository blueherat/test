from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.dinov2_token_diagnostics import configure_fp32, load_named_dataset  # noqa: E402
from baselines.visual_adapters import load_rae_adapter  # noqa: E402
from experiments.train_decoder_inverse_adapter import (  # noqa: E402
    DEFAULT_ADAPTER,
    DecoderAdapterImageDataset,
    decode_m11,
    encode_adapted_y,
    load_flow_checkpoint,
    pick_indices,
    psnr_from_mse,
    relative_l2,
)


@dataclass
class DecoderInverseEvalConfig:
    dataset_name: str = "imagenet_parquet"
    data_root: str = "/data/shared"
    dataset_path: str = "/data/shared/imagenet-1k"
    split: str = "validation"
    image_size: int = 256
    count: int = 2048
    sequential: bool = False
    model_key: str = "rae_dinov2"
    rae_repo_path: str = "external/RAE"
    adapter_checkpoint: str = str(DEFAULT_ADAPTER)
    decoder_adapter_checkpoint: str = (
        "artifacts/decoder_inverse_adapter/dinov2_e3_decoder_inverse_imagenet_e1/decoder_adapter.pt"
    )
    output_dir: str = "artifacts/decoder_inverse_adapter_eval"
    run_name: str = ""
    device: str = "cuda:0"
    seed: int = 0
    noise_seed: int = 12345
    noise_tau: float = 0.8
    batch_size: int = 32
    num_workers: int = 4


def make_noisy(y: torch.Tensor, noise_tau: float, generator: torch.Generator) -> torch.Tensor:
    if noise_tau <= 0:
        return y
    sigma = noise_tau * torch.rand(
        (y.shape[0],) + (1,) * (y.ndim - 1),
        device=y.device,
        dtype=y.dtype,
        generator=generator,
    )
    eps = torch.randn(y.shape, device=y.device, dtype=y.dtype, generator=generator)
    return y + sigma * eps


def tensor_metrics(prefix: str, recon: torch.Tensor, x: torch.Tensor, z_pred: torch.Tensor, z: torch.Tensor) -> Dict[str, float]:
    mse = float(F.mse_loss(recon.clamp(-1.0, 1.0), x).cpu())
    return {
        f"{prefix}_l1": float(F.l1_loss(recon, x).cpu()),
        f"{prefix}_mse": mse,
        f"{prefix}_psnr": psnr_from_mse(mse),
        f"{prefix}_latent_rel": float(relative_l2(z_pred, z).cpu()),
    }


@torch.no_grad()
def run(cfg: DecoderInverseEvalConfig) -> dict:
    configure_fp32()
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    dataset = load_named_dataset(
        cfg.dataset_name,
        cfg.data_root,
        split=cfg.split,
        dataset_path=cfg.dataset_path,
    )
    indices = pick_indices(len(dataset), cfg.count, cfg.seed, sequential=cfg.sequential)
    loader = DataLoader(
        DecoderAdapterImageDataset(dataset, indices, cfg.image_size, random_crop=False),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    rae = load_rae_adapter(cfg.model_key, repo_path=cfg.rae_repo_path, device=device, dtype=torch.float32)
    rae.model.eval().requires_grad_(False)
    encoder_flow, _ = load_flow_checkpoint(cfg.adapter_checkpoint, device)
    initial_flow, _ = load_flow_checkpoint(cfg.adapter_checkpoint, device)
    trained_flow, _ = load_flow_checkpoint(cfg.decoder_adapter_checkpoint, device)
    encoder_flow.eval().requires_grad_(False)
    initial_flow.eval().requires_grad_(False)
    trained_flow.eval().requires_grad_(False)

    generator = torch.Generator(device=device)
    generator.manual_seed(int(cfg.noise_seed))
    totals: Dict[str, float] = {
        "base_l1": 0.0,
        "initial_clean_l1": 0.0,
        "initial_clean_mse": 0.0,
        "initial_clean_psnr": 0.0,
        "initial_clean_latent_rel": 0.0,
        "initial_noisy_l1": 0.0,
        "initial_noisy_mse": 0.0,
        "initial_noisy_psnr": 0.0,
        "initial_noisy_latent_rel": 0.0,
        "trained_clean_l1": 0.0,
        "trained_clean_mse": 0.0,
        "trained_clean_psnr": 0.0,
        "trained_clean_latent_rel": 0.0,
        "trained_noisy_l1": 0.0,
        "trained_noisy_mse": 0.0,
        "trained_noisy_psnr": 0.0,
        "trained_noisy_latent_rel": 0.0,
    }
    batches = 0
    for x_cpu, _ in loader:
        x = x_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
        z, y = encode_adapted_y(rae, encoder_flow, x)
        y_noisy = make_noisy(y, cfg.noise_tau, generator)
        base = decode_m11(rae, z)
        totals["base_l1"] += float(F.l1_loss(base, x).cpu())

        for name, flow in (("initial", initial_flow), ("trained", trained_flow)):
            z_clean = flow.inverse(y)
            z_noisy = flow.inverse(y_noisy)
            clean = decode_m11(rae, z_clean)
            noisy = decode_m11(rae, z_noisy)
            for key, value in tensor_metrics(f"{name}_clean", clean, x, z_clean, z).items():
                totals[key] += value
            for key, value in tensor_metrics(f"{name}_noisy", noisy, x, z_noisy, z).items():
                totals[key] += value
        batches += 1
        if batches % 20 == 0 or batches == len(loader):
            print(f"evaluated {batches}/{len(loader)} batches", flush=True)

    metrics = {key: value / max(1, batches) for key, value in totals.items()}
    metrics["delta_noisy_latent_rel"] = metrics["trained_noisy_latent_rel"] - metrics["initial_noisy_latent_rel"]
    metrics["delta_noisy_l1"] = metrics["trained_noisy_l1"] - metrics["initial_noisy_l1"]
    metrics["delta_clean_latent_rel"] = metrics["trained_clean_latent_rel"] - metrics["initial_clean_latent_rel"]
    result = {
        "config": asdict(cfg),
        "metrics": metrics,
        "artifacts": {},
    }
    name = cfg.run_name.strip()
    if not name:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        name = f"decoder_inverse_eval_{Path(cfg.decoder_adapter_checkpoint).parent.name}_{stamp}"
    run_dir = Path(cfg.output_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    result["artifacts"]["run_dir"] = str(run_dir)
    output = run_dir / "metrics.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "metrics": metrics}, ensure_ascii=False, indent=2), flush=True)
    return result


def parse_args() -> DecoderInverseEvalConfig:
    parser = argparse.ArgumentParser(description="Fair fixed-noise evaluation for decoder-side inverse adapters.")
    parser.add_argument("--dataset-name", default=DecoderInverseEvalConfig.dataset_name)
    parser.add_argument("--data-root", default=DecoderInverseEvalConfig.data_root)
    parser.add_argument("--dataset-path", default=DecoderInverseEvalConfig.dataset_path)
    parser.add_argument("--split", default=DecoderInverseEvalConfig.split)
    parser.add_argument("--image-size", type=int, default=DecoderInverseEvalConfig.image_size)
    parser.add_argument("--count", type=int, default=DecoderInverseEvalConfig.count)
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--model-key", default=DecoderInverseEvalConfig.model_key)
    parser.add_argument("--rae-repo-path", default=DecoderInverseEvalConfig.rae_repo_path)
    parser.add_argument("--adapter-checkpoint", default=DecoderInverseEvalConfig.adapter_checkpoint)
    parser.add_argument("--decoder-adapter-checkpoint", default=DecoderInverseEvalConfig.decoder_adapter_checkpoint)
    parser.add_argument("--output-dir", default=DecoderInverseEvalConfig.output_dir)
    parser.add_argument("--run-name", default=DecoderInverseEvalConfig.run_name)
    parser.add_argument("--device", default=DecoderInverseEvalConfig.device)
    parser.add_argument("--seed", type=int, default=DecoderInverseEvalConfig.seed)
    parser.add_argument("--noise-seed", type=int, default=DecoderInverseEvalConfig.noise_seed)
    parser.add_argument("--noise-tau", type=float, default=DecoderInverseEvalConfig.noise_tau)
    parser.add_argument("--batch-size", type=int, default=DecoderInverseEvalConfig.batch_size)
    parser.add_argument("--num-workers", type=int, default=DecoderInverseEvalConfig.num_workers)
    args = parser.parse_args()
    return DecoderInverseEvalConfig(
        dataset_name=args.dataset_name,
        data_root=args.data_root,
        dataset_path=args.dataset_path,
        split=args.split,
        image_size=args.image_size,
        count=args.count,
        sequential=args.sequential,
        model_key=args.model_key,
        rae_repo_path=args.rae_repo_path,
        adapter_checkpoint=args.adapter_checkpoint,
        decoder_adapter_checkpoint=args.decoder_adapter_checkpoint,
        output_dir=args.output_dir,
        run_name=args.run_name,
        device=args.device,
        seed=args.seed,
        noise_seed=args.noise_seed,
        noise_tau=args.noise_tau,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    run(parse_args())
