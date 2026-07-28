"""Fixed validation evaluation for deterministic-decoder RAE LPL branches."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
RAE_ROOT = ROOT / "external" / "RAE"
RAE_SRC = RAE_ROOT / "src"
for path in (ROOT, RAE_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.rae_strict_lpl import (
    decoder_feature_pyramid,
    decoder_hidden_indices,
    flow_clean_estimate,
    strict_lpl_per_sample,
)
from stage1 import RAE
from utils.model_utils import instantiate_from_config
from utils.train_utils import ParquetImageNetDataset, center_crop_arr


def parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must be NAME=PATH")
    name, path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("checkpoint name cannot be empty")
    return name, Path(path).expanduser().resolve()


def resolve_rae_paths(config) -> None:
    params = config.stage_1.params
    for name in (
        "decoder_config_path",
        "pretrained_decoder_path",
        "normalization_stat_path",
    ):
        value = params.get(name)
        if value is not None and not Path(str(value)).is_absolute():
            params[name] = str(RAE_ROOT / str(value))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", type=parse_checkpoint, required=True)
    parser.add_argument("--state-key", choices=("model", "ema"), default="ema")
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--noise-ratios", type=float, nargs="+", default=(0.5, 1.0, 2.0, 3.0))
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if args.num_samples % world_size != 0:
        raise ValueError("num-samples must be divisible by world size")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=True)

    config = OmegaConf.load(args.config.expanduser().resolve())
    resolve_rae_paths(config)
    lpl_config = OmegaConf.to_container(config.training.strict_lpl, resolve=True)
    rae: RAE = instantiate_from_config(config.stage_1).to(device=device, dtype=torch.float32)
    rae.requires_grad_(False).eval()
    model = instantiate_from_config(config.stage_2).to(device=device, dtype=torch.float32)
    model.requires_grad_(False).eval()
    layer_indices = decoder_hidden_indices(
        len(rae.decoder.decoder_layers),
        tuple(float(value) for value in lpl_config["layer_fractions"]),
    )

    transform = transforms.Compose(
        [
            transforms.Lambda(lambda image: center_crop_arr(image, 256)),
            transforms.ToTensor(),
        ]
    )
    full_dataset = ParquetImageNetDataset(
        args.data_path.expanduser().resolve(), split="validation", transform=transform
    )
    dataset = Subset(full_dataset, range(int(args.num_samples)))
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )

    rows = []
    for checkpoint_name, checkpoint_path in args.checkpoint:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if args.state_key not in state:
            raise KeyError(f"{checkpoint_path} lacks {args.state_key!r}")
        model.load_state_dict(state[args.state_key], strict=True)
        model.eval()
        generator = torch.Generator(device=device)
        generator.manual_seed(int(args.seed) + rank)
        # Per ratio: flow sum, clean-latent MSE sum, LPL sum, mask sum, count.
        totals = torch.zeros(
            len(args.noise_ratios), 5, device=device, dtype=torch.float64
        )
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                clean = rae.encode(images)
                noise = torch.randn(
                    clean.shape,
                    generator=generator,
                    device=device,
                    dtype=clean.dtype,
                )
                target_velocity = noise - clean
                target_features = decoder_feature_pyramid(
                    rae, clean, layer_indices=layer_indices
                )
                for ratio_index, ratio in enumerate(args.noise_ratios):
                    time_value = float(ratio) / (1.0 + float(ratio))
                    time = torch.full(
                        (images.shape[0],), time_value, device=device, dtype=clean.dtype
                    )
                    noisy = (1.0 - time_value) * clean + time_value * noise
                    prediction = model(noisy, time, y=labels)
                    predicted_clean = flow_clean_estimate(noisy, prediction, time)
                    predicted_features = decoder_feature_pyramid(
                        rae, predicted_clean, layer_indices=layer_indices
                    )
                    lpl, details = strict_lpl_per_sample(
                        target_features,
                        predicted_features,
                        layer_weights=[1.0] * len(layer_indices),
                        outlier_quantile=float(lpl_config["outlier_quantile"]),
                        outlier_opening=int(lpl_config["outlier_opening"]),
                        outlier_closing=int(lpl_config["outlier_closing"]),
                        eps=float(lpl_config["normalization_eps"]),
                    )
                    flow = (prediction - target_velocity).square().flatten(1).mean(1)
                    latent_mse = (predicted_clean - clean).square().flatten(1).mean(1)
                    totals[ratio_index, 0] += flow.double().sum()
                    totals[ratio_index, 1] += latent_mse.double().sum()
                    totals[ratio_index, 2] += lpl.double().sum()
                    totals[ratio_index, 3] += (
                        details["mask_keep_fraction"].mean(1).double().sum()
                    )
                    totals[ratio_index, 4] += images.shape[0]
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        if rank == 0:
            for ratio_index, ratio in enumerate(args.noise_ratios):
                count = float(totals[ratio_index, 4])
                rows.append(
                    {
                        "checkpoint": checkpoint_name,
                        "state_key": args.state_key,
                        "noise_to_signal_ratio": float(ratio),
                        "time": float(ratio) / (1.0 + float(ratio)),
                        "samples": int(count),
                        "flow_loss": float(totals[ratio_index, 0] / count),
                        "clean_latent_mse": float(totals[ratio_index, 1] / count),
                        "decoder_lpl": float(totals[ratio_index, 2] / count),
                        "mask_keep_fraction": float(totals[ratio_index, 3] / count),
                    }
                )

    if rank == 0:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset": "ImageNet-1k validation",
            "num_samples": int(args.num_samples),
            "seed": int(args.seed),
            "precision": "fp32",
            "tf32": False,
            "decoder_deterministic": True,
            "decoder_hidden_indices": list(layer_indices),
            "rows": rows,
        }
        output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        table = pd.DataFrame(rows)
        table.to_csv(output.with_suffix(".csv"), index=False)
        print(table.to_string(index=False))
        print(output)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
