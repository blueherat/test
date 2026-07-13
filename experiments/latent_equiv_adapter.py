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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.dinov2_token_diagnostics import (  # noqa: E402
    P,
    center_crop_resize,
    configure_fp32,
    load_named_dataset,
    pil_to_tensor_m11,
    split_train_val_test_indices,
)
from baselines.visual_adapters import load_rae_adapter  # noqa: E402
from evaluation.calculate_fid import calculate_fid_given_paths  # noqa: E402


FID_CACHE_PATH = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "pt_inception-2015-12-05-6726825d.pth"


@dataclass
class AdapterExperimentConfig:
    data_root: str = "/data/shared"
    dataset_name: str = "caltech101"
    dataset_split: str = "train"
    dataset_path: str = ""
    eval_dataset_name: str = ""
    eval_dataset_split: str = ""
    eval_dataset_path: str = ""
    download_dataset: bool = False
    image_size: int = 256
    model_key: str = "rae_dinov2"
    rae_repo_path: str = "external/RAE"
    rae_auto_clone: bool = False
    rae_auto_download: bool = False
    device: str = "cuda:0"
    seed: int = 0
    train_count: int = 128
    val_count: int = 32
    test_count: int = 32
    sequential_split: bool = False
    batch_size: int = 4
    num_workers: int = 2
    epochs: int = 3
    lr: float = 1e-4
    weight_decay: float = 0.0
    blocks: int = 4
    hidden_channels: int = 128
    train_transforms: Tuple[str, ...] = ("flip_h", "flip_v", "rot180")
    eval_transform: str = "rot90"
    equiv_weight: float = 1.0
    inverse_weight: float = 1.0
    identity_weight: float = 0.02
    decoder_weight: float = 0.0
    grad_clip: float = 1.0
    init_checkpoint: str = ""
    output_dir: str = "artifacts/latent_adapter"
    run_name: str = ""
    viz_count: int = 8
    fid_count: int = 32
    fid_dims: int = 2048
    fid_batch_size: int = 16
    fid_num_workers: int = 2
    skip_fid: bool = False
    progress_interval: int = 250


class ImageTensorDataset(Dataset):
    def __init__(self, dataset, indices: Sequence[int], image_size: int):
        self.dataset = dataset
        self.indices = [int(i) for i in indices]
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, int]:
        index = self.indices[item]
        sample = self.dataset[index]
        image = sample[0] if isinstance(sample, (tuple, list)) else sample
        image = center_crop_resize(image.convert("RGB"), self.image_size)
        return pil_to_tensor_m11(image), index


class CouplingNet(nn.Module):
    def __init__(self, channels: int, hidden_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AdditiveCouplingBlock(nn.Module):
    def __init__(self, channels: int, hidden_channels: int, swap: bool = False):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError(f"AdditiveCouplingBlock 需要偶数通道数，收到 {channels}")
        self.channels = int(channels)
        self.half = channels // 2
        self.swap = bool(swap)
        self.net = CouplingNet(self.half, hidden_channels)

    def _split(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.swap:
            x = torch.cat((x[:, self.half :], x[:, : self.half]), dim=1)
        return x[:, : self.half], x[:, self.half :]

    def _merge(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        y = torch.cat((a, b), dim=1)
        if self.swap:
            y = torch.cat((y[:, self.half :], y[:, : self.half]), dim=1)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self._split(x)
        return self._merge(a, b + self.net(a))

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        a, b = self._split(y)
        return self._merge(a, b - self.net(a))


class InvertibleLatentAdapter(nn.Module):
    def __init__(self, channels: int = 768, hidden_channels: int = 128, blocks: int = 4):
        super().__init__()
        self.blocks = nn.ModuleList(
            AdditiveCouplingBlock(channels, hidden_channels, swap=bool(i % 2))
            for i in range(int(blocks))
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        y = z
        for block in self.blocks:
            y = block(y)
        return y

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        z = y
        for block in reversed(self.blocks):
            z = block.inverse(z)
        return z


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_tensor_dataset(dataset, indices: Sequence[int], cfg: AdapterExperimentConfig) -> ImageTensorDataset:
    return ImageTensorDataset(dataset, indices, cfg.image_size)


def make_loader_from_tensor_dataset(
    tensor_dataset: ImageTensorDataset,
    cfg: AdapterExperimentConfig,
    shuffle: bool,
    sampler=None,
) -> DataLoader:
    return DataLoader(
        tensor_dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def make_loader(dataset, indices: Sequence[int], cfg: AdapterExperimentConfig, shuffle: bool) -> DataLoader:
    return make_loader_from_tensor_dataset(make_tensor_dataset(dataset, indices, cfg), cfg, shuffle=shuffle)


def flow_inverse(flow: nn.Module, y: torch.Tensor) -> torch.Tensor:
    module = flow.module if isinstance(flow, DDP) else flow
    return module.inverse(y)


def relative_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    num = (pred - target).flatten(1).pow(2).sum(dim=1)
    den = target.flatten(1).pow(2).sum(dim=1).clamp_min(eps)
    return torch.sqrt(num / den).mean()


@torch.no_grad()
def encode_images(rae_adapter, x: torch.Tensor) -> torch.Tensor:
    return rae_adapter.encode(x)


def decode_for_loss(rae_adapter, z: torch.Tensor) -> torch.Tensor:
    x = rae_adapter.model.decode(z.contiguous()).clamp(0.0, 1.0)
    return x * 2.0 - 1.0


@torch.no_grad()
def decode_images(rae_adapter, z: torch.Tensor) -> torch.Tensor:
    return rae_adapter.decode(z)


def train_one_epoch(
    flow: InvertibleLatentAdapter,
    rae_adapter,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: AdapterExperimentConfig,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    flow.train()
    totals = {"loss": 0.0, "equiv": 0.0, "inverse": 0.0, "identity": 0.0, "decoder": 0.0}
    batches = 0
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    epoch_start = time.time()
    log_every = int(cfg.progress_interval)
    total_batches = len(loader)
    for batch_index, (x_cpu, _) in enumerate(loader):
        x = x_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
        transform = cfg.train_transforms[(batch_index + epoch) % len(cfg.train_transforms)]
        xg = P(x, transform)
        with torch.no_grad():
            z = encode_images(rae_adapter, x)
            zg = encode_images(rae_adapter, xg)
        y = flow(z)
        yg = flow(zg)
        py = P(y, transform)
        z_pred = flow_inverse(flow, py)

        equiv = relative_l2(yg, py)
        inverse = relative_l2(z_pred, zg)
        identity = F.mse_loss(y, z)
        decoder = z.new_tensor(0.0)
        if cfg.decoder_weight > 0:
            decoder = F.l1_loss(decode_for_loss(rae_adapter, z_pred), xg)
        loss = (
            cfg.equiv_weight * equiv
            + cfg.inverse_weight * inverse
            + cfg.identity_weight * identity
            + cfg.decoder_weight * decoder
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(flow.parameters(), cfg.grad_clip)
        optimizer.step()

        for name, value in (
            ("loss", loss),
            ("equiv", equiv),
            ("inverse", inverse),
            ("identity", identity),
            ("decoder", decoder),
        ):
            totals[name] += float(value.detach().cpu())
        batches += 1
        if rank == 0 and log_every > 0 and (batches % log_every == 0 or batches == total_batches):
            elapsed_min = (time.time() - epoch_start) / 60.0
            approx_images = batches * int(cfg.batch_size) * int(world_size)
            print(
                f"epoch {epoch + 1}/{cfg.epochs} "
                f"batch={batches}/{total_batches} "
                f"images~={approx_images} "
                f"loss={totals['loss'] / max(1, batches):.4f} "
                f"equiv={totals['equiv'] / max(1, batches):.4f} "
                f"inverse={totals['inverse'] / max(1, batches):.4f} "
                f"identity={totals['identity'] / max(1, batches):.6f} "
                f"elapsed_min={elapsed_min:.1f}",
                flush=True,
            )
    return {name: total / max(1, batches) for name, total in totals.items()}


def reduce_mean_metrics(metrics: Dict[str, float], device: torch.device) -> Dict[str, float]:
    if not dist.is_available() or not dist.is_initialized():
        return metrics
    keys = sorted(metrics)
    values = torch.tensor([float(metrics[key]) for key in keys], device=device, dtype=torch.float64)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= dist.get_world_size()
    return {key: float(value.item()) for key, value in zip(keys, values)}


@torch.no_grad()
def evaluate_latent(
    flow: InvertibleLatentAdapter,
    rae_adapter,
    loader: DataLoader,
    transform: str,
    device: torch.device,
) -> Dict[str, float]:
    flow.eval()
    totals = {"baseline_latent": 0.0, "adapted_y": 0.0, "adapted_z": 0.0, "identity_rel": 0.0, "cycle": 0.0}
    batches = 0
    for x_cpu, _ in loader:
        x = x_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
        xg = P(x, transform)
        z = encode_images(rae_adapter, x)
        zg = encode_images(rae_adapter, xg)
        y = flow(z)
        yg = flow(zg)
        py = P(y, transform)
        z_pred = flow.inverse(py)
        z_cycle = flow.inverse(y)
        totals["baseline_latent"] += float(relative_l2(P(z, transform), zg).cpu())
        totals["adapted_y"] += float(relative_l2(py, yg).cpu())
        totals["adapted_z"] += float(relative_l2(z_pred, zg).cpu())
        totals["identity_rel"] += float(relative_l2(y, z).cpu())
        totals["cycle"] += float(relative_l2(z_cycle, z).cpu())
        batches += 1
    return {name: total / max(1, batches) for name, total in totals.items()}


@torch.no_grad()
def evaluate_group_metrics(
    flow: InvertibleLatentAdapter,
    rae_adapter,
    loader: DataLoader,
    device: torch.device,
    transforms: Sequence[str] = ("flip_h", "flip_v", "rot180"),
) -> Dict[str, float]:
    flow.eval()
    totals: Dict[str, float] = {"identity_rel": 0.0, "cycle": 0.0}
    for transform in transforms:
        totals[f"{transform}_baseline_latent"] = 0.0
        totals[f"{transform}_adapted_y"] = 0.0
        totals[f"{transform}_adapted_z"] = 0.0
    totals["compose_h_then_v_y"] = 0.0
    totals["compose_v_then_h_y"] = 0.0
    batches = 0
    for x_cpu, _ in loader:
        x = x_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
        z = encode_images(rae_adapter, x)
        y = flow(z)
        totals["identity_rel"] += float(relative_l2(y, z).cpu())
        totals["cycle"] += float(relative_l2(flow.inverse(y), z).cpu())

        y_by_transform: Dict[str, torch.Tensor] = {}
        z_by_transform: Dict[str, torch.Tensor] = {}
        for transform in transforms:
            zg = encode_images(rae_adapter, P(x, transform))
            yg = flow(zg)
            py = P(y, transform)
            z_pred = flow.inverse(py)
            z_by_transform[transform] = zg
            y_by_transform[transform] = yg
            totals[f"{transform}_baseline_latent"] += float(relative_l2(P(z, transform), zg).cpu())
            totals[f"{transform}_adapted_y"] += float(relative_l2(py, yg).cpu())
            totals[f"{transform}_adapted_z"] += float(relative_l2(z_pred, zg).cpu())

        if {"flip_h", "flip_v", "rot180"}.issubset(y_by_transform):
            y_h = y_by_transform["flip_h"]
            y_v = y_by_transform["flip_v"]
            y_180 = y_by_transform["rot180"]
            totals["compose_h_then_v_y"] += float(relative_l2(P(y_h, "flip_v"), y_180).cpu())
            totals["compose_v_then_h_y"] += float(relative_l2(P(y_v, "flip_h"), y_180).cpu())
        batches += 1
    return {name: total / max(1, batches) for name, total in totals.items()}


def tensor_to_uint8_image(x: torch.Tensor) -> Image.Image:
    image = x.detach().float().cpu()
    if image.ndim == 4:
        image = image[0]
    image = image.clamp(-1.0, 1.0)
    arr = ((image + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    arr = arr.permute(1, 2, 0).numpy()
    return Image.fromarray(arr)


def save_tensor_images(batch: torch.Tensor, directory: Path, prefix: str, start_index: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for i, image in enumerate(batch):
        tensor_to_uint8_image(image).save(directory / f"{prefix}_{start_index + i:06d}.png")


def make_visual_grid(rows: List[List[torch.Tensor]], labels: Sequence[str], path: Path, cell: int = 192) -> None:
    margin_top = 28
    label_height = 22
    gap = 6
    columns = len(labels)
    height = margin_top + len(rows) * (cell + gap) + label_height
    width = columns * (cell + gap) - gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for col, label in enumerate(labels):
        draw.text((col * (cell + gap) + 4, 4), label, fill=(0, 0, 0))
    for row_idx, row in enumerate(rows):
        top = margin_top + row_idx * (cell + gap)
        for col_idx, tensor in enumerate(row):
            image = tensor_to_uint8_image(tensor).resize((cell, cell), Image.Resampling.BICUBIC)
            canvas.paste(image, (col_idx * (cell + gap), top))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


@torch.no_grad()
def save_eval_artifacts(
    flow: InvertibleLatentAdapter,
    rae_adapter,
    loader: DataLoader,
    cfg: AdapterExperimentConfig,
    run_dir: Path,
    device: torch.device,
) -> Dict[str, str]:
    flow.eval()
    transform = cfg.eval_transform
    image_dirs = {
        "source": run_dir / "images" / "source",
        "target": run_dir / "images" / f"target_{transform}",
        "baseline": run_dir / "images" / f"baseline_Pz_{transform}",
        "adapted": run_dir / "images" / f"adapted_FinvPFz_{transform}",
        "recon": run_dir / "images" / "recon_identity",
    }
    counters = {name: 0 for name in image_dirs}
    visual_rows: List[List[torch.Tensor]] = []
    processed = 0
    for x_cpu, _ in loader:
        if processed >= cfg.fid_count:
            break
        x = x_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
        if processed + x.shape[0] > cfg.fid_count:
            x = x[: cfg.fid_count - processed]
        xg = P(x, transform)
        z = encode_images(rae_adapter, x)
        y = flow(z)
        baseline = decode_images(rae_adapter, P(z, transform))
        adapted = decode_images(rae_adapter, flow.inverse(P(y, transform)))
        recon = decode_images(rae_adapter, flow.inverse(y))
        batches = {"source": x, "target": xg, "baseline": baseline, "adapted": adapted, "recon": recon}
        for name, images in batches.items():
            save_tensor_images(images, image_dirs[name], name, counters[name])
            counters[name] += images.shape[0]
        remaining_viz = max(0, cfg.viz_count - len(visual_rows))
        for i in range(min(remaining_viz, x.shape[0])):
            visual_rows.append([x[i], xg[i], baseline[i], adapted[i], recon[i]])
        processed += x.shape[0]
    make_visual_grid(
        visual_rows,
        labels=("x", f"{transform}(x)", "D(P z)", "D(F^-1 P F z)", "D(F^-1 F z)"),
        path=run_dir / f"visual_{transform}.png",
    )
    return {name: str(path) for name, path in image_dirs.items()}


def compute_rfid(image_dirs: Dict[str, str], cfg: AdapterExperimentConfig, device: torch.device) -> Dict[str, float]:
    if cfg.skip_fid:
        return {}
    target = image_dirs["target"]
    source = image_dirs["source"]
    results = {}
    for name, reference, candidate in (
        ("rfid_baseline_transform", target, image_dirs["baseline"]),
        ("rfid_adapted_transform", target, image_dirs["adapted"]),
        ("rfid_reconstruction_identity", source, image_dirs["recon"]),
    ):
        for attempt in range(2):
            try:
                results[name] = float(
                    calculate_fid_given_paths(
                        [reference, candidate],
                        batch_size=cfg.fid_batch_size,
                        dims=cfg.fid_dims,
                        device=device,
                        num_workers=cfg.fid_num_workers,
                    )
                )
                break
            except RuntimeError as exc:
                message = str(exc).lower()
                if attempt == 0 and ("unexpected eof" in message or "corrupt" in message):
                    if FID_CACHE_PATH.exists():
                        FID_CACHE_PATH.unlink()
                    print("FID Inception 权重缓存损坏，已删除并重试下载。", flush=True)
                    continue
                raise
    return results


def write_metrics(run_dir: Path, metrics: Dict[str, object]) -> None:
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    flat_rows = []
    for section, values in metrics.items():
        if isinstance(values, dict):
            for key, value in values.items():
                if isinstance(value, (int, float, str)):
                    flat_rows.append((section, key, value))
    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "metric", "value"])
        writer.writerows(flat_rows)


def write_partial_state(
    run_dir: Path,
    cfg: AdapterExperimentConfig,
    split: Dict[str, List[int]],
    split_sources: Dict[str, Dict[str, str]],
    flow: InvertibleLatentAdapter,
    optimizer: torch.optim.Optimizer,
    initial_val: Dict[str, float],
    history: List[Dict[str, float]],
    latest_val: Dict[str, float],
    latent_shape: Sequence[int],
) -> None:
    torch.save(
        {
            "config": asdict(cfg),
            "state_dict": flow.state_dict(),
            "optimizer": optimizer.state_dict(),
            "history": history,
            "channels": int(latent_shape[1]),
            "height": int(latent_shape[2]),
            "width": int(latent_shape[3]),
        },
        run_dir / "adapter_latest.pt",
    )
    partial = {
        "config": asdict(cfg),
        "split_indices": split,
        "split_sources": split_sources,
        "initial_val": initial_val,
        "latest_val": latest_val,
        "history": history,
        "artifacts": {
            "run_dir": str(run_dir),
            "checkpoint_latest": str(run_dir / "adapter_latest.pt"),
        },
    }
    with (run_dir / "metrics_partial.json").open("w", encoding="utf-8") as f:
        json.dump(partial, f, ensure_ascii=False, indent=2)


def build_run_dir(cfg: AdapterExperimentConfig) -> Path:
    name = cfg.run_name.strip()
    if not name:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        name = f"{cfg.model_key}_{cfg.dataset_name}_{cfg.eval_transform}_{stamp}"
    run_dir = Path(cfg.output_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_experiment(cfg: AdapterExperimentConfig) -> Dict[str, object]:
    configure_fp32()
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed adapter training requires CUDA.")
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

    dataset = load_named_dataset(
        cfg.dataset_name,
        cfg.data_root,
        cfg.dataset_split,
        download=cfg.download_dataset,
        dataset_path=cfg.dataset_path,
    )
    use_external_eval = bool(cfg.eval_dataset_split.strip())
    if use_external_eval:
        needed = int(cfg.train_count) + int(cfg.val_count)
        if cfg.train_count <= 0 or cfg.val_count < 0:
            raise ValueError("train_count 必须大于 0，val_count 不能为负。")
        if len(dataset) < needed:
            raise ValueError(f"train dataset has {len(dataset)} images, less than requested train+val={needed}")
        if cfg.sequential_split:
            perm = list(range(needed))
        else:
            rng = np.random.default_rng(cfg.seed)
            perm = [int(i) for i in rng.permutation(len(dataset))]
        split = {
            "train": perm[: cfg.train_count],
            "val": perm[cfg.train_count : cfg.train_count + cfg.val_count],
            "test": [],
        }
    else:
        if cfg.sequential_split:
            needed = int(cfg.train_count) + int(cfg.val_count) + int(cfg.test_count)
            if len(dataset) < needed:
                raise ValueError(f"dataset has {len(dataset)} images, less than requested total={needed}")
            split = {
                "train": list(range(cfg.train_count)),
                "val": list(range(cfg.train_count, cfg.train_count + cfg.val_count)),
                "test": list(range(cfg.train_count + cfg.val_count, needed)),
            }
        else:
            split = split_train_val_test_indices(len(dataset), cfg.train_count, cfg.val_count, cfg.test_count, cfg.seed)
    train_tensor_dataset = make_tensor_dataset(dataset, split["train"], cfg)
    train_sampler = (
        DistributedSampler(
            train_tensor_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=not cfg.sequential_split,
            seed=cfg.seed,
            drop_last=False,
        )
        if distributed
        else None
    )
    train_loader = make_loader_from_tensor_dataset(
        train_tensor_dataset,
        cfg,
        shuffle=not cfg.sequential_split,
        sampler=train_sampler,
    )
    val_loader = make_loader(dataset, split["val"], cfg, shuffle=False)
    if use_external_eval:
        eval_dataset = load_named_dataset(
            cfg.eval_dataset_name or cfg.dataset_name,
            cfg.data_root,
            cfg.eval_dataset_split,
            download=cfg.download_dataset,
            dataset_path=cfg.eval_dataset_path or cfg.dataset_path,
        )
        if len(eval_dataset) < cfg.test_count:
            raise ValueError(
                f"eval dataset split {cfg.eval_dataset_split!r} has {len(eval_dataset)} images, "
                f"less than requested test_count={cfg.test_count}"
            )
        split["test"] = [int(i) for i in range(cfg.test_count)]
        test_loader = make_loader(eval_dataset, split["test"], cfg, shuffle=False)
    else:
        test_loader = make_loader(dataset, split["test"], cfg, shuffle=False)
    split_sources = {
        "train": {"dataset_name": cfg.dataset_name, "split": cfg.dataset_split, "dataset_path": cfg.dataset_path},
        "val": {"dataset_name": cfg.dataset_name, "split": cfg.dataset_split, "dataset_path": cfg.dataset_path},
        "test": {
            "dataset_name": cfg.eval_dataset_name or cfg.dataset_name,
            "split": cfg.eval_dataset_split or cfg.dataset_split,
            "dataset_path": cfg.eval_dataset_path or cfg.dataset_path,
        },
    }

    rae = load_rae_adapter(
        cfg.model_key,
        repo_path=cfg.rae_repo_path,
        device=device,
        dtype=torch.float32,
        auto_clone=cfg.rae_auto_clone,
        auto_download=cfg.rae_auto_download,
    )
    for param in rae.model.parameters():
        param.requires_grad_(False)
    rae.model.eval()

    with torch.no_grad():
        first_x, _ = next(iter(train_loader))
        first_z = encode_images(rae, first_x.to(device=device, dtype=torch.float32))
    flow = InvertibleLatentAdapter(
        channels=first_z.shape[1],
        hidden_channels=cfg.hidden_channels,
        blocks=cfg.blocks,
    ).to(device=device, dtype=torch.float32)
    if cfg.init_checkpoint.strip():
        init_path = Path(cfg.init_checkpoint).expanduser()
        checkpoint = torch.load(init_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        flow.load_state_dict(state_dict, strict=True)
        print(f"loaded initial adapter checkpoint: {init_path}", flush=True)
    optimizer = torch.optim.AdamW(flow.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_flow: nn.Module = (
        DDP(flow, device_ids=[device.index], broadcast_buffers=False, find_unused_parameters=False)
        if distributed
        else flow
    )

    initial_val = evaluate_latent(flow, rae, val_loader, cfg.eval_transform, device) if rank == 0 else {}
    initial_group_val = evaluate_group_metrics(flow, rae, val_loader, device) if rank == 0 else {}
    history = []
    for epoch in range(cfg.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_metrics = reduce_mean_metrics(train_one_epoch(train_flow, rae, train_loader, optimizer, cfg, device, epoch), device)
        if rank == 0:
            val_metrics = evaluate_latent(flow, rae, val_loader, cfg.eval_transform, device)
            row = {"epoch": epoch + 1, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
            history.append(row)
            write_partial_state(
                run_dir,
                cfg,
                split,
                split_sources,
                flow,
                optimizer,
                initial_val,
                history,
                val_metrics,
                first_z.shape,
            )
            print(
                f"epoch {epoch + 1}/{cfg.epochs} "
                f"loss={train_metrics['loss']:.4f} "
                f"val_baseline={val_metrics['baseline_latent']:.4f} "
                f"val_adapted_z={val_metrics['adapted_z']:.4f}",
                flush=True,
            )

    if distributed:
        dist.barrier()
    if rank != 0:
        if distributed:
            dist.destroy_process_group()
        return {"rank": rank, "run_dir": str(run_dir)}

    final_val = evaluate_latent(flow, rae, val_loader, cfg.eval_transform, device)
    final_test = evaluate_latent(flow, rae, test_loader, cfg.eval_transform, device)
    final_group_val = evaluate_group_metrics(flow, rae, val_loader, device)
    final_group_test = evaluate_group_metrics(flow, rae, test_loader, device)
    image_dirs = save_eval_artifacts(flow, rae, test_loader, cfg, run_dir, device)
    rfid = compute_rfid(image_dirs, cfg, device)

    checkpoint = {
        "config": asdict(cfg),
        "state_dict": flow.state_dict(),
        "channels": int(first_z.shape[1]),
        "height": int(first_z.shape[2]),
        "width": int(first_z.shape[3]),
    }
    torch.save(checkpoint, run_dir / "adapter.pt")
    metrics: Dict[str, object] = {
        "config": asdict(cfg),
        "split_indices": split,
        "split_sources": split_sources,
        "initial_val": initial_val,
        "initial_group_val": initial_group_val,
        "final_val": final_val,
        "final_test": final_test,
        "final_group_val": final_group_val,
        "final_group_test": final_group_test,
        "rfid": rfid,
        "history": history,
        "artifacts": {
            "run_dir": str(run_dir),
            "checkpoint": str(run_dir / "adapter.pt"),
            "visual": str(run_dir / f"visual_{cfg.eval_transform}.png"),
            "image_dirs": image_dirs,
        },
    }
    write_metrics(run_dir, metrics)
    print(
        json.dumps(
            {"run_dir": str(run_dir), "final_test": final_test, "final_group_test": final_group_test, "rfid": rfid},
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if distributed:
        dist.destroy_process_group()
    return metrics


def parse_args() -> AdapterExperimentConfig:
    parser = argparse.ArgumentParser(description="Train a small invertible latent adapter for RAE equivariance.")
    parser.add_argument("--data-root", default=AdapterExperimentConfig.data_root)
    parser.add_argument("--dataset-name", default=AdapterExperimentConfig.dataset_name)
    parser.add_argument("--dataset-split", default=AdapterExperimentConfig.dataset_split)
    parser.add_argument("--dataset-path", default=AdapterExperimentConfig.dataset_path)
    parser.add_argument("--eval-dataset-name", default=AdapterExperimentConfig.eval_dataset_name)
    parser.add_argument("--eval-dataset-split", default=AdapterExperimentConfig.eval_dataset_split)
    parser.add_argument("--eval-dataset-path", default=AdapterExperimentConfig.eval_dataset_path)
    parser.add_argument("--download-dataset", action="store_true")
    parser.add_argument("--image-size", type=int, default=AdapterExperimentConfig.image_size)
    parser.add_argument("--model-key", default=AdapterExperimentConfig.model_key)
    parser.add_argument("--rae-repo-path", default=AdapterExperimentConfig.rae_repo_path)
    parser.add_argument("--rae-auto-clone", action="store_true")
    parser.add_argument("--rae-auto-download", action="store_true")
    parser.add_argument("--device", default=AdapterExperimentConfig.device)
    parser.add_argument("--seed", type=int, default=AdapterExperimentConfig.seed)
    parser.add_argument("--train-count", type=int, default=AdapterExperimentConfig.train_count)
    parser.add_argument("--val-count", type=int, default=AdapterExperimentConfig.val_count)
    parser.add_argument("--test-count", type=int, default=AdapterExperimentConfig.test_count)
    parser.add_argument("--sequential-split", action="store_true")
    parser.add_argument("--batch-size", type=int, default=AdapterExperimentConfig.batch_size)
    parser.add_argument("--num-workers", type=int, default=AdapterExperimentConfig.num_workers)
    parser.add_argument("--epochs", type=int, default=AdapterExperimentConfig.epochs)
    parser.add_argument("--lr", type=float, default=AdapterExperimentConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=AdapterExperimentConfig.weight_decay)
    parser.add_argument("--blocks", type=int, default=AdapterExperimentConfig.blocks)
    parser.add_argument("--hidden-channels", type=int, default=AdapterExperimentConfig.hidden_channels)
    parser.add_argument("--train-transforms", nargs="+", default=list(AdapterExperimentConfig.train_transforms))
    parser.add_argument("--eval-transform", default=AdapterExperimentConfig.eval_transform)
    parser.add_argument("--equiv-weight", type=float, default=AdapterExperimentConfig.equiv_weight)
    parser.add_argument("--inverse-weight", type=float, default=AdapterExperimentConfig.inverse_weight)
    parser.add_argument("--identity-weight", type=float, default=AdapterExperimentConfig.identity_weight)
    parser.add_argument("--decoder-weight", type=float, default=AdapterExperimentConfig.decoder_weight)
    parser.add_argument("--grad-clip", type=float, default=AdapterExperimentConfig.grad_clip)
    parser.add_argument("--init-checkpoint", default=AdapterExperimentConfig.init_checkpoint)
    parser.add_argument("--output-dir", default=AdapterExperimentConfig.output_dir)
    parser.add_argument("--run-name", default=AdapterExperimentConfig.run_name)
    parser.add_argument("--viz-count", type=int, default=AdapterExperimentConfig.viz_count)
    parser.add_argument("--fid-count", type=int, default=AdapterExperimentConfig.fid_count)
    parser.add_argument("--fid-dims", type=int, default=AdapterExperimentConfig.fid_dims)
    parser.add_argument("--fid-batch-size", type=int, default=AdapterExperimentConfig.fid_batch_size)
    parser.add_argument("--fid-num-workers", type=int, default=AdapterExperimentConfig.fid_num_workers)
    parser.add_argument("--skip-fid", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=AdapterExperimentConfig.progress_interval)
    args = parser.parse_args()
    return AdapterExperimentConfig(
        data_root=args.data_root,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        dataset_path=args.dataset_path,
        eval_dataset_name=args.eval_dataset_name,
        eval_dataset_split=args.eval_dataset_split,
        eval_dataset_path=args.eval_dataset_path,
        download_dataset=args.download_dataset,
        image_size=args.image_size,
        model_key=args.model_key,
        rae_repo_path=args.rae_repo_path,
        rae_auto_clone=args.rae_auto_clone,
        rae_auto_download=args.rae_auto_download,
        device=args.device,
        seed=args.seed,
        train_count=args.train_count,
        val_count=args.val_count,
        test_count=args.test_count,
        sequential_split=args.sequential_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        blocks=args.blocks,
        hidden_channels=args.hidden_channels,
        train_transforms=tuple(args.train_transforms),
        eval_transform=args.eval_transform,
        equiv_weight=args.equiv_weight,
        inverse_weight=args.inverse_weight,
        identity_weight=args.identity_weight,
        decoder_weight=args.decoder_weight,
        grad_clip=args.grad_clip,
        init_checkpoint=args.init_checkpoint,
        output_dir=args.output_dir,
        run_name=args.run_name,
        viz_count=args.viz_count,
        fid_count=args.fid_count,
        fid_dims=args.fid_dims,
        fid_batch_size=args.fid_batch_size,
        fid_num_workers=args.fid_num_workers,
        skip_fid=args.skip_fid,
        progress_interval=args.progress_interval,
    )


if __name__ == "__main__":
    run_experiment(parse_args())
