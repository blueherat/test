from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.dinov2_token_diagnostics import configure_fp32, load_named_dataset  # noqa: E402
from baselines.visual_adapters import load_rae_adapter  # noqa: E402
from experiments.latent_equiv_adapter import InvertibleLatentAdapter, relative_l2  # noqa: E402


DEFAULT_ADAPTER = (
    ROOT
    / "artifacts/latent_adapter/"
    "dinov2_adapter_imagenet_full_e3_light_r1_strict_id5_lr3e6_from_e2/adapter.pt"
)


@dataclass
class DecoderInverseAdapterConfig:
    dataset_name: str = "imagenet_parquet"
    data_root: str = "/data/shared"
    dataset_path: str = "/data/shared/imagenet-1k"
    train_split: str = "train"
    val_split: str = "validation"
    image_size: int = 256
    model_key: str = "rae_dinov2"
    rae_repo_path: str = "external/RAE"
    rae_auto_clone: bool = False
    rae_auto_download: bool = False
    adapter_checkpoint: str = str(DEFAULT_ADAPTER)
    init_decoder_adapter_checkpoint: str = ""
    output_dir: str = "artifacts/decoder_inverse_adapter"
    run_name: str = ""
    device: str = "cuda:0"
    seed: int = 0
    train_count: int = 0
    val_count: int = 2048
    sequential_train: bool = False
    sequential_val: bool = False
    batch_size: int = 8
    eval_batch_size: int = 8
    num_workers: int = 4
    epochs: int = 1
    lr: float = 1e-5
    weight_decay: float = 0.0
    noise_tau: float = 0.8
    eval_noise_tau: float = 0.8
    noisy_recon_weight: float = 0.0
    clean_recon_weight: float = 0.0
    noisy_latent_weight: float = 1.0
    latent_weight: float = 0.1
    grad_clip: float = 1.0
    progress_interval: int = 100
    train_base_decoder: bool = False


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.Resampling.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.Resampling.BICUBIC)

    arr = np.array(pil_image.convert("RGB"))
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


def tensor01_to_m11(x: torch.Tensor) -> torch.Tensor:
    return x * 2.0 - 1.0


class DecoderAdapterImageDataset(Dataset):
    def __init__(
        self,
        dataset,
        indices: Sequence[int],
        image_size: int,
        *,
        random_crop: bool,
    ):
        self.dataset = dataset
        self.indices = [int(i) for i in indices]
        self.image_size = int(image_size)
        self.random_crop = bool(random_crop)
        first_crop_size = 384 if self.image_size == 256 else int(self.image_size * 1.5)
        self.train_transform = transforms.Compose(
            [
                transforms.Resize(first_crop_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomCrop(self.image_size),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, int]:
        index = self.indices[item]
        sample = self.dataset[index]
        image = sample[0] if isinstance(sample, (tuple, list)) else sample
        image = image.convert("RGB")
        if self.random_crop:
            x01 = self.train_transform(image)
        else:
            cropped = center_crop_arr(image, self.image_size)
            x01 = transforms.functional.to_tensor(cropped)
        return tensor01_to_m11(x01), index


def pick_indices(total: int, count: int, seed: int, sequential: bool = False) -> List[int]:
    if count <= 0:
        return list(range(total))
    if total < count:
        raise ValueError(f"dataset has {total} images, less than requested count={count}")
    if sequential:
        return list(range(int(count)))
    rng = np.random.default_rng(seed)
    return [int(i) for i in rng.permutation(total)[:count]]


def make_loader(
    dataset,
    indices: Sequence[int],
    cfg: DecoderInverseAdapterConfig,
    *,
    random_crop: bool,
    batch_size: int,
    sampler=None,
) -> DataLoader:
    return DataLoader(
        DecoderAdapterImageDataset(dataset, indices, cfg.image_size, random_crop=random_crop),
        batch_size=batch_size,
        shuffle=(random_crop and sampler is None),
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def load_flow_checkpoint(path: str | Path, device: torch.device) -> tuple[InvertibleLatentAdapter, dict]:
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = dict(checkpoint.get("config", {}))
    state_dict = checkpoint.get("state_dict", checkpoint)
    channels, hidden_channels, blocks = infer_adapter_arch(state_dict)
    flow = InvertibleLatentAdapter(
        channels=int(checkpoint.get("channels", cfg.get("channels", channels))),
        hidden_channels=int(checkpoint.get("hidden_channels", cfg.get("hidden_channels", hidden_channels))),
        blocks=int(checkpoint.get("blocks", cfg.get("blocks", blocks))),
    )
    flow.load_state_dict(state_dict, strict=True)
    return flow.to(device=device, dtype=torch.float32), checkpoint


def infer_adapter_arch(state_dict: dict[str, torch.Tensor]) -> tuple[int, int, int]:
    first_weight = state_dict.get("blocks.0.net.net.0.weight")
    if first_weight is None:
        return 768, 128, 4
    hidden_channels = int(first_weight.shape[0])
    channels = int(first_weight.shape[1]) * 2
    block_ids = []
    for key in state_dict:
        parts = key.split(".")
        if len(parts) > 1 and parts[0] == "blocks" and parts[1].isdigit():
            block_ids.append(int(parts[1]))
    blocks = max(block_ids) + 1 if block_ids else 4
    return channels, hidden_channels, blocks


def module_inverse(module: nn.Module, y: torch.Tensor) -> torch.Tensor:
    target = module.module if isinstance(module, DDP) else module
    return target.inverse(y)


def add_rae_style_noise(y: torch.Tensor, noise_tau: float) -> torch.Tensor:
    if noise_tau <= 0:
        return y
    sigma = float(noise_tau) * torch.rand((y.shape[0],) + (1,) * (y.ndim - 1), device=y.device, dtype=y.dtype)
    return y + sigma * torch.randn_like(y)


def decode_m11(rae_adapter, z: torch.Tensor) -> torch.Tensor:
    x01 = rae_adapter.model.decode(z.contiguous())
    return x01 * 2.0 - 1.0


@torch.no_grad()
def encode_base_z(rae_adapter, x_m11: torch.Tensor) -> torch.Tensor:
    return rae_adapter.encode(x_m11)


@torch.no_grad()
def encode_adapted_y(rae_adapter, encoder_flow: nn.Module, x_m11: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    z = encode_base_z(rae_adapter, x_m11)
    y = encoder_flow(z)
    return z, y


def psnr_from_mse(mse: float) -> float:
    return -10.0 * math.log10(max(mse, 1e-12))


def reduce_mean_metrics(metrics: Dict[str, float], device: torch.device) -> Dict[str, float]:
    if not dist.is_available() or not dist.is_initialized():
        return metrics
    keys = sorted(metrics)
    values = torch.tensor([float(metrics[key]) for key in keys], device=device, dtype=torch.float64)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= dist.get_world_size()
    return {key: float(value.item()) for key, value in zip(keys, values)}


def train_one_epoch(
    decoder_flow: nn.Module,
    encoder_flow: nn.Module,
    rae_adapter,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: DecoderInverseAdapterConfig,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    decoder_flow.train()
    uses_decoder_loss = cfg.noisy_recon_weight > 0 or cfg.clean_recon_weight > 0
    if cfg.train_base_decoder:
        rae_adapter.model.decoder.train()
    else:
        rae_adapter.model.decoder.eval()
    totals = {
        "loss": 0.0,
        "noisy_recon": 0.0,
        "clean_recon": 0.0,
        "noisy_latent": 0.0,
        "clean_latent": 0.0,
    }
    batches = 0
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    started = time.time()
    for batch_index, (x_cpu, _) in enumerate(loader):
        x = x_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
        with torch.no_grad():
            z, y = encode_adapted_y(rae_adapter, encoder_flow, x)

        y_noisy = add_rae_style_noise(y, cfg.noise_tau)
        z_noisy = module_inverse(decoder_flow, y_noisy)
        noisy_latent = relative_l2(z_noisy, z)

        clean_recon = x.new_zeros(())
        noisy_recon = x.new_zeros(())
        z_clean = module_inverse(decoder_flow, y)
        clean_latent = relative_l2(z_clean, z)
        if uses_decoder_loss:
            if cfg.noisy_recon_weight > 0:
                recon_noisy = decode_m11(rae_adapter, z_noisy)
                noisy_recon = F.l1_loss(recon_noisy, x)
            if cfg.clean_recon_weight > 0:
                recon_clean = decode_m11(rae_adapter, z_clean)
                clean_recon = F.l1_loss(recon_clean, x)

        loss = (
            cfg.noisy_recon_weight * noisy_recon
            + cfg.clean_recon_weight * clean_recon
            + cfg.noisy_latent_weight * noisy_latent
            + cfg.latent_weight * clean_latent
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            params = [p for group in optimizer.param_groups for p in group["params"]]
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        optimizer.step()

        for name, value in (
            ("loss", loss),
            ("noisy_recon", noisy_recon),
            ("clean_recon", clean_recon),
            ("noisy_latent", noisy_latent),
            ("clean_latent", clean_latent),
        ):
            totals[name] += float(value.detach().cpu())
        batches += 1
        if rank == 0 and cfg.progress_interval > 0 and (
            batches % cfg.progress_interval == 0 or batches == len(loader)
        ):
            seen = batches * int(cfg.batch_size) * int(world_size)
            elapsed = (time.time() - started) / 60.0
            print(
                f"epoch {epoch + 1}/{cfg.epochs} batch={batches}/{len(loader)} "
                f"images~={seen} loss={totals['loss'] / batches:.5f} "
                f"noisy_latent={totals['noisy_latent'] / batches:.5f} "
                f"clean_latent={totals['clean_latent'] / batches:.5f} elapsed_min={elapsed:.1f}",
                flush=True,
            )
    return {name: total / max(1, batches) for name, total in totals.items()}


@torch.no_grad()
def evaluate(
    decoder_flow: InvertibleLatentAdapter,
    encoder_flow: InvertibleLatentAdapter,
    rae_adapter,
    loader: DataLoader,
    cfg: DecoderInverseAdapterConfig,
    device: torch.device,
) -> Dict[str, float]:
    decoder_flow.eval()
    encoder_flow.eval()
    rae_adapter.model.eval()
    totals = {
        "base_l1": 0.0,
        "clean_l1": 0.0,
        "noisy_l1": 0.0,
        "clean_mse": 0.0,
        "noisy_mse": 0.0,
        "clean_latent_rel": 0.0,
        "noisy_latent_rel": 0.0,
    }
    batches = 0
    for x_cpu, _ in loader:
        x = x_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
        z, y = encode_adapted_y(rae_adapter, encoder_flow, x)
        base = decode_m11(rae_adapter, z)
        z_clean = decoder_flow.inverse(y)
        y_noisy = add_rae_style_noise(y, cfg.eval_noise_tau)
        z_noisy = decoder_flow.inverse(y_noisy)
        clean = decode_m11(rae_adapter, z_clean)
        noisy = decode_m11(rae_adapter, z_noisy)
        totals["base_l1"] += float(F.l1_loss(base, x).cpu())
        totals["clean_l1"] += float(F.l1_loss(clean, x).cpu())
        totals["noisy_l1"] += float(F.l1_loss(noisy, x).cpu())
        totals["clean_mse"] += float(F.mse_loss(clean.clamp(-1.0, 1.0), x).cpu())
        totals["noisy_mse"] += float(F.mse_loss(noisy.clamp(-1.0, 1.0), x).cpu())
        totals["clean_latent_rel"] += float(relative_l2(z_clean, z).cpu())
        totals["noisy_latent_rel"] += float(relative_l2(z_noisy, z).cpu())
        batches += 1
    metrics = {name: total / max(1, batches) for name, total in totals.items()}
    metrics["clean_psnr"] = psnr_from_mse(metrics["clean_mse"])
    metrics["noisy_psnr"] = psnr_from_mse(metrics["noisy_mse"])
    return metrics


def write_metrics(run_dir: Path, metrics: Dict[str, object]) -> None:
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    rows = []
    for section, values in metrics.items():
        if isinstance(values, dict):
            for key, value in values.items():
                if isinstance(value, (int, float, str)):
                    rows.append((section, key, value))
    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "metric", "value"])
        writer.writerows(rows)


def save_checkpoint(
    run_dir: Path,
    filename: str,
    decoder_flow: InvertibleLatentAdapter,
    optimizer: torch.optim.Optimizer,
    cfg: DecoderInverseAdapterConfig,
    history: List[Dict[str, float]],
    shape: Sequence[int],
) -> str:
    path = run_dir / filename
    state_dict = decoder_flow.state_dict()
    channels, hidden_channels, blocks = infer_adapter_arch(state_dict)
    torch.save(
        {
            "config": asdict(cfg),
            "state_dict": state_dict,
            "optimizer": optimizer.state_dict(),
            "history": history,
            "channels": int(channels),
            "hidden_channels": int(hidden_channels),
            "blocks": int(blocks),
            "height": int(shape[2]),
            "width": int(shape[3]),
            "decoder_side_adapter": True,
        },
        path,
    )
    return str(path)


def build_run_dir(cfg: DecoderInverseAdapterConfig) -> Path:
    name = cfg.run_name.strip()
    if not name:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        name = f"{cfg.model_key}_decoder_inverse_noise{cfg.noise_tau:g}_{stamp}"
    run_dir = Path(cfg.output_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run(cfg: DecoderInverseAdapterConfig) -> Dict[str, object]:
    configure_fp32()
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        if cfg.train_base_decoder:
            raise ValueError("DDP path currently trains only the decoder-side inverse adapter, not the full decoder.")
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA.")
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        rank = 0
        world_size = 1
        device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    set_seed(cfg.seed + rank)
    run_dir = build_run_dir(cfg)

    train_dataset = load_named_dataset(
        cfg.dataset_name,
        cfg.data_root,
        split=cfg.train_split,
        dataset_path=cfg.dataset_path,
    )
    val_dataset = load_named_dataset(
        cfg.dataset_name,
        cfg.data_root,
        split=cfg.val_split,
        dataset_path=cfg.dataset_path,
    )
    train_indices = pick_indices(len(train_dataset), cfg.train_count, cfg.seed, sequential=cfg.sequential_train)
    val_indices = pick_indices(len(val_dataset), cfg.val_count, cfg.seed + 17, sequential=cfg.sequential_val)
    train_tensor_dataset = DecoderAdapterImageDataset(
        train_dataset,
        train_indices,
        cfg.image_size,
        random_crop=True,
    )
    train_sampler = (
        DistributedSampler(
            train_tensor_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=not cfg.sequential_train,
            seed=cfg.seed,
            drop_last=False,
        )
        if distributed
        else None
    )
    train_loader = DataLoader(
        train_tensor_dataset,
        batch_size=cfg.batch_size,
        shuffle=(train_sampler is None and not cfg.sequential_train),
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = make_loader(
        val_dataset,
        val_indices,
        cfg,
        random_crop=False,
        batch_size=cfg.eval_batch_size,
    )

    rae = load_rae_adapter(
        cfg.model_key,
        repo_path=cfg.rae_repo_path,
        device=device,
        dtype=torch.float32,
        auto_clone=cfg.rae_auto_clone,
        auto_download=cfg.rae_auto_download,
    )
    rae.model.encoder.requires_grad_(False)
    rae.model.decoder.requires_grad_(bool(cfg.train_base_decoder))
    if cfg.train_base_decoder:
        rae.model.decoder.train()
    else:
        rae.model.decoder.eval()

    encoder_flow, encoder_checkpoint = load_flow_checkpoint(cfg.adapter_checkpoint, device)
    encoder_flow.eval().requires_grad_(False)
    decoder_init = cfg.init_decoder_adapter_checkpoint.strip() or cfg.adapter_checkpoint
    decoder_flow, _ = load_flow_checkpoint(decoder_init, device)
    decoder_flow.train().requires_grad_(True)

    params = list(decoder_flow.parameters())
    if cfg.train_base_decoder:
        params += [p for p in rae.model.decoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_module: nn.Module = (
        DDP(decoder_flow, device_ids=[device.index], broadcast_buffers=False, find_unused_parameters=False)
        if distributed
        else decoder_flow
    )

    with torch.no_grad():
        first_x, _ = next(iter(train_loader))
        first_z = encode_base_z(rae, first_x.to(device=device, dtype=torch.float32))

    if rank == 0:
        config_path = run_dir / "config.json"
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)
        initial_val = evaluate(decoder_flow, encoder_flow, rae, val_loader, cfg, device)
        print(json.dumps({"initial_val": initial_val}, ensure_ascii=False, indent=2), flush=True)
    else:
        initial_val = {}

    history: List[Dict[str, float]] = []
    for epoch in range(cfg.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_metrics = reduce_mean_metrics(
            train_one_epoch(train_module, encoder_flow, rae, train_loader, optimizer, cfg, device, epoch),
            device,
        )
        if distributed:
            dist.barrier()
        if rank == 0:
            val_metrics = evaluate(decoder_flow, encoder_flow, rae, val_loader, cfg, device)
            row = {
                "epoch": epoch + 1,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            history.append(row)
            latest_path = save_checkpoint(
                run_dir,
                "decoder_adapter_latest.pt",
                decoder_flow,
                optimizer,
                cfg,
                history,
                first_z.shape,
            )
            print(
                f"epoch {epoch + 1}/{cfg.epochs} "
                f"loss={train_metrics['loss']:.5f} "
                f"val_clean_l1={val_metrics['clean_l1']:.5f} "
                f"val_noisy_l1={val_metrics['noisy_l1']:.5f} "
                f"val_noisy_latent={val_metrics['noisy_latent_rel']:.5f} "
                f"ckpt={latest_path}",
                flush=True,
            )

    if distributed:
        dist.barrier()
    if rank != 0:
        if distributed:
            dist.destroy_process_group()
        return {"rank": rank, "run_dir": str(run_dir)}

    final_val = evaluate(decoder_flow, encoder_flow, rae, val_loader, cfg, device)
    checkpoint_path = save_checkpoint(
        run_dir,
        "decoder_adapter.pt",
        decoder_flow,
        optimizer,
        cfg,
        history,
        first_z.shape,
    )
    metrics: Dict[str, object] = {
        "config": asdict(cfg),
        "split_sources": {
            "train": {
                "dataset_name": cfg.dataset_name,
                "split": cfg.train_split,
                "dataset_path": cfg.dataset_path,
                "count": len(train_indices),
                "sequential": bool(cfg.sequential_train),
            },
            "val": {
                "dataset_name": cfg.dataset_name,
                "split": cfg.val_split,
                "dataset_path": cfg.dataset_path,
                "count": len(val_indices),
                "sequential": bool(cfg.sequential_val),
            },
        },
        "initial_val": initial_val,
        "final_val": final_val,
        "history": history,
        "artifacts": {
            "run_dir": str(run_dir),
            "checkpoint": checkpoint_path,
            "checkpoint_latest": str(run_dir / "decoder_adapter_latest.pt"),
            "encoder_adapter_checkpoint": str(Path(cfg.adapter_checkpoint).expanduser()),
            "encoder_adapter_history": encoder_checkpoint.get("history", []),
        },
    }
    write_metrics(run_dir, metrics)
    print(json.dumps({"run_dir": str(run_dir), "final_val": final_val}, ensure_ascii=False, indent=2), flush=True)
    if distributed:
        dist.destroy_process_group()
    return metrics


def parse_args() -> DecoderInverseAdapterConfig:
    parser = argparse.ArgumentParser(description="Fine-tune a decoder-side inverse adapter for adapted RAE latents.")
    parser.add_argument("--dataset-name", default=DecoderInverseAdapterConfig.dataset_name)
    parser.add_argument("--data-root", default=DecoderInverseAdapterConfig.data_root)
    parser.add_argument("--dataset-path", default=DecoderInverseAdapterConfig.dataset_path)
    parser.add_argument("--train-split", default=DecoderInverseAdapterConfig.train_split)
    parser.add_argument("--val-split", default=DecoderInverseAdapterConfig.val_split)
    parser.add_argument("--image-size", type=int, default=DecoderInverseAdapterConfig.image_size)
    parser.add_argument("--model-key", default=DecoderInverseAdapterConfig.model_key)
    parser.add_argument("--rae-repo-path", default=DecoderInverseAdapterConfig.rae_repo_path)
    parser.add_argument("--rae-auto-clone", action="store_true")
    parser.add_argument("--rae-auto-download", action="store_true")
    parser.add_argument("--adapter-checkpoint", default=DecoderInverseAdapterConfig.adapter_checkpoint)
    parser.add_argument("--init-decoder-adapter-checkpoint", default=DecoderInverseAdapterConfig.init_decoder_adapter_checkpoint)
    parser.add_argument("--output-dir", default=DecoderInverseAdapterConfig.output_dir)
    parser.add_argument("--run-name", default=DecoderInverseAdapterConfig.run_name)
    parser.add_argument("--device", default=DecoderInverseAdapterConfig.device)
    parser.add_argument("--seed", type=int, default=DecoderInverseAdapterConfig.seed)
    parser.add_argument("--train-count", type=int, default=DecoderInverseAdapterConfig.train_count)
    parser.add_argument("--val-count", type=int, default=DecoderInverseAdapterConfig.val_count)
    parser.add_argument("--sequential-train", action="store_true")
    parser.add_argument("--sequential-val", action="store_true")
    parser.add_argument("--batch-size", type=int, default=DecoderInverseAdapterConfig.batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=DecoderInverseAdapterConfig.eval_batch_size)
    parser.add_argument("--num-workers", type=int, default=DecoderInverseAdapterConfig.num_workers)
    parser.add_argument("--epochs", type=int, default=DecoderInverseAdapterConfig.epochs)
    parser.add_argument("--lr", type=float, default=DecoderInverseAdapterConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=DecoderInverseAdapterConfig.weight_decay)
    parser.add_argument("--noise-tau", type=float, default=DecoderInverseAdapterConfig.noise_tau)
    parser.add_argument("--eval-noise-tau", type=float, default=DecoderInverseAdapterConfig.eval_noise_tau)
    parser.add_argument("--noisy-recon-weight", type=float, default=DecoderInverseAdapterConfig.noisy_recon_weight)
    parser.add_argument("--clean-recon-weight", type=float, default=DecoderInverseAdapterConfig.clean_recon_weight)
    parser.add_argument("--noisy-latent-weight", type=float, default=DecoderInverseAdapterConfig.noisy_latent_weight)
    parser.add_argument("--latent-weight", type=float, default=DecoderInverseAdapterConfig.latent_weight)
    parser.add_argument("--grad-clip", type=float, default=DecoderInverseAdapterConfig.grad_clip)
    parser.add_argument("--progress-interval", type=int, default=DecoderInverseAdapterConfig.progress_interval)
    parser.add_argument("--train-base-decoder", action="store_true")
    args = parser.parse_args()
    return DecoderInverseAdapterConfig(
        dataset_name=args.dataset_name,
        data_root=args.data_root,
        dataset_path=args.dataset_path,
        train_split=args.train_split,
        val_split=args.val_split,
        image_size=args.image_size,
        model_key=args.model_key,
        rae_repo_path=args.rae_repo_path,
        rae_auto_clone=args.rae_auto_clone,
        rae_auto_download=args.rae_auto_download,
        adapter_checkpoint=args.adapter_checkpoint,
        init_decoder_adapter_checkpoint=args.init_decoder_adapter_checkpoint,
        output_dir=args.output_dir,
        run_name=args.run_name,
        device=args.device,
        seed=args.seed,
        train_count=args.train_count,
        val_count=args.val_count,
        sequential_train=args.sequential_train,
        sequential_val=args.sequential_val,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        noise_tau=args.noise_tau,
        eval_noise_tau=args.eval_noise_tau,
        noisy_recon_weight=args.noisy_recon_weight,
        clean_recon_weight=args.clean_recon_weight,
        noisy_latent_weight=args.noisy_latent_weight,
        latent_weight=args.latent_weight,
        grad_clip=args.grad_clip,
        progress_interval=args.progress_interval,
        train_base_decoder=args.train_base_decoder,
    )


if __name__ == "__main__":
    run(parse_args())
