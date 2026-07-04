from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset, Sampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "train_eqvae") not in sys.path:
    sys.path.insert(0, str(ROOT / "train_eqvae"))

from baselines.visual_adapters import RAE_SPECS, load_rae_adapter


TRANSFORMS: Tuple[str, ...] = ("identity", "rot90", "rot180", "rot270", "flip_h", "flip_v")
NON_IDENTITY_TRANSFORMS: Tuple[str, ...] = tuple(g for g in TRANSFORMS if g != "identity")
DEFAULT_LAMBDAS: Tuple[float, ...] = (1e-4, 1e-3, 1e-2)

HF_VAE_SPECS: Dict[str, dict] = {
    "eqvae": {"repo_id": "zelaki/eq-vae", "scaling_factor": None},
    "eqvae_ema": {"repo_id": "zelaki/eq-vae-ema", "scaling_factor": None},
    "sdvae": {"repo_id": "stabilityai/sd-vae-ft-mse", "scaling_factor": 0.18215},
    "sdvae_ft_mse": {"repo_id": "stabilityai/sd-vae-ft-mse", "scaling_factor": 0.18215},
    "sdvae_ft_ema": {"repo_id": "stabilityai/sd-vae-ft-ema", "scaling_factor": 0.18215},
}


@dataclass
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    distributed: bool

    @property
    def is_main(self) -> bool:
        return self.rank == 0


@dataclass
class Phase1Config:
    data_root: str = "/data/shared"
    dataset_name: str = "caltech101"
    dataset_split: str = "train"
    dataset_path: str = ""
    download_dataset: bool = False
    image_size: int = 256
    seed: int = 0
    train_count: int = 512
    val_count: int = 256
    test_count: int = 256
    batch_size: int = 16
    num_workers: int = 4
    models: Tuple[str, ...] = ("rae_dinov2", "rae_mae", "rae_siglip2", "eqvae", "sdvae")
    transforms: Tuple[str, ...] = TRANSFORMS
    lambdas: Tuple[float, ...] = DEFAULT_LAMBDAS
    rae_repo_path: str = "external/RAE"
    rae_auto_clone: bool = False
    rae_auto_download: bool = False
    artifact_dir: str = "artifacts/group_action_phase1"
    posterior: str = "mode"
    fit_generators: bool = False
    generator_models: Tuple[str, ...] = ()
    generator_steps: int = 300
    generator_lr: float = 1e-3
    generator_law_weights: Tuple[float, ...] = (0.01, 0.1, 1.0)
    generator_anchor_weight: float = 0.01
    generator_score_law_weight: float = 0.5
    generator_grad_clip: float = 10.0


@dataclass
class HFVAEAdapter:
    key: str
    model: torch.nn.Module
    device: torch.device
    dtype: torch.dtype
    scaling_factor: float

    @torch.no_grad()
    def encode(self, x: torch.Tensor, posterior: Optional[str] = "mode") -> torch.Tensor:
        self.model.eval()
        x = x.to(device=self.device, dtype=self.dtype)
        latent_dist = self.model.encode(x).latent_dist
        z = latent_dist.mode() if posterior == "mode" else latent_dist.sample()
        return z * self.scaling_factor

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        z = z.to(device=self.device, dtype=self.dtype) / self.scaling_factor
        return self.model.decode(z).sample.clamp(-1.0, 1.0)


class ImageTensorDataset(Dataset):
    def __init__(self, base_dataset, indices: Sequence[int], image_size: int):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, int]:
        index = int(self.indices[item])
        sample = self.base_dataset[index]
        img = sample[0] if isinstance(sample, (tuple, list)) else sample
        img = center_crop_resize(img.convert("RGB"), self.image_size)
        return pil_to_tensor_m11(img), index


class DistributedShardSampler(Sampler[int]):
    """Shard indices across ranks without padding or dropping samples."""

    def __init__(self, dataset: Dataset, rank: int, world_size: int):
        self.dataset = dataset
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        total = len(self.dataset)
        if total <= self.rank:
            return 0
        return (total - 1 - self.rank) // self.world_size + 1


def configure_fp32() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")


def setup_distributed() -> DistributedContext:
    configure_fp32()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if torch.cuda.is_available():
        if distributed:
            torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}" if distributed else "cuda")
    else:
        device = torch.device("cpu")

    if distributed and not dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        if backend == "nccl":
            try:
                dist.init_process_group(backend=backend, device_id=device)
            except TypeError:
                dist.init_process_group(backend=backend)
        else:
            dist.init_process_group(backend=backend)

    return DistributedContext(rank=rank, local_rank=local_rank, world_size=world_size, device=device, distributed=distributed)


def distributed_barrier(ctx: DistributedContext) -> None:
    if not (ctx.distributed and dist.is_initialized()):
        return
    if ctx.device.type == "cuda":
        try:
            dist.barrier(device_ids=[ctx.local_rank])
            return
        except TypeError:
            pass
    dist.barrier()


def cleanup_distributed(ctx: DistributedContext) -> None:
    if ctx.distributed and dist.is_initialized():
        distributed_barrier(ctx)
        dist.destroy_process_group()


def log_main(ctx: DistributedContext, message: str) -> None:
    if ctx.is_main:
        print(message, flush=True)


def normalize_dataset_name(name: str) -> str:
    return (name or "caltech101").strip().lower().replace("-", "_")


def load_named_dataset(name: str, root: str, split: str, download: bool = False, dataset_path: str = ""):
    from torchvision.datasets import CIFAR10, CIFAR100, STL10, Caltech101, Flowers102, ImageFolder, OxfordIIITPet

    name = normalize_dataset_name(name)
    split = (split or "train").strip().lower()
    root = str(root)

    if name == "image_folder":
        folder = Path(dataset_path).expanduser() if dataset_path else Path(root)
        if not folder.exists():
            raise FileNotFoundError(f"image_folder 路径不存在：{folder}")
        return ImageFolder(str(folder))
    if name == "cifar10":
        return CIFAR10(root=root, train=split != "test", download=download)
    if name == "cifar100":
        return CIFAR100(root=root, train=split != "test", download=download)
    if name == "stl10":
        stl_split = split if split in {"train", "test", "unlabeled", "train+unlabeled"} else "train"
        return STL10(root=root, split=stl_split, download=download)
    if name == "caltech101":
        return Caltech101(root=root, download=download)
    if name in {"flowers102", "flowers"}:
        flower_split = "val" if split in {"val", "valid", "validation"} else split
        if flower_split not in {"train", "val", "test"}:
            flower_split = "train"
        return Flowers102(root=root, split=flower_split, download=download)
    if name in {"oxford_iiit_pet", "oxford_pet", "pets"}:
        pet_split = "test" if split == "test" else "trainval"
        return OxfordIIITPet(root=root, split=pet_split, download=download)

    raise ValueError(
        "不支持的数据集："
        f"{name}。可选：cifar10, cifar100, stl10, caltech101, flowers102, oxford_iiit_pet, image_folder"
    )


def split_indices(total: int, train_count: int, val_count: int, test_count: int, seed: int) -> Dict[str, List[int]]:
    needed = int(train_count) + int(val_count) + int(test_count)
    if total < needed:
        raise ValueError(f"数据集只有 {total} 张，少于请求的 {needed} 张。请调小 train/val/test count。")
    rng = np.random.default_rng(seed)
    perm = [int(i) for i in rng.permutation(total)]
    train_end = train_count
    val_end = train_count + val_count
    return {
        "train": perm[:train_end],
        "val": perm[train_end:val_end],
        "test": perm[val_end:val_end + test_count],
    }


def validate_config(config: Phase1Config, ctx: DistributedContext) -> None:
    known_models = set(RAE_SPECS) | set(HF_VAE_SPECS)
    unknown_models = [name for name in config.models if name not in known_models]
    if unknown_models:
        raise ValueError(f"未知模型：{unknown_models}；可选 {sorted(known_models)}")

    required_transforms = set(TRANSFORMS)
    missing_transforms = sorted(required_transforms - set(config.transforms))
    if missing_transforms:
        raise ValueError(f"第一阶段群律检查需要这些 transform：{sorted(required_transforms)}；当前缺少 {missing_transforms}")
    if len(set(config.transforms)) != len(config.transforms):
        raise ValueError(f"transforms 里有重复项：{config.transforms}")
    if not config.lambdas:
        raise ValueError("至少需要一个 ridge lambda。")
    if any(float(value) < 0 for value in config.lambdas):
        raise ValueError(f"ridge lambda 必须非负：{config.lambdas}")
    if any(float(value) < 0 for value in config.generator_law_weights):
        raise ValueError(f"generator law weight 必须非负：{config.generator_law_weights}")
    if min(config.train_count, config.val_count, config.test_count) <= 0:
        raise ValueError("train/val/test count 都必须大于 0。")
    if ctx.distributed and config.train_count < ctx.world_size:
        raise ValueError(
            f"DDP 下 train_count={config.train_count} 小于 world_size={ctx.world_size}，"
            "会导致部分 rank 无法初始化 ridge 统计量。请增大 train_count 或减少 GPU 数。"
        )


def should_fit_generators(config: Phase1Config, model_name: str) -> bool:
    if not config.fit_generators:
        return False
    return not config.generator_models or model_name in set(config.generator_models)


def make_loaders(config: Phase1Config, ctx: DistributedContext) -> Dict[str, DataLoader]:
    base = load_named_dataset(
        config.dataset_name,
        config.data_root,
        config.dataset_split,
        download=config.download_dataset,
        dataset_path=config.dataset_path,
    )
    splits = split_indices(len(base), config.train_count, config.val_count, config.test_count, config.seed)
    loaders: Dict[str, DataLoader] = {}
    for split, indices in splits.items():
        dataset = ImageTensorDataset(base, indices, config.image_size)
        sampler = None
        if ctx.distributed:
            sampler = DistributedShardSampler(dataset, rank=ctx.rank, world_size=ctx.world_size)
        loaders[split] = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=config.num_workers,
            pin_memory=ctx.device.type == "cuda",
            drop_last=False,
        )
    return loaders


def center_crop_resize(img: Image.Image, size: int) -> Image.Image:
    width, height = img.size
    scale = size / min(width, height)
    resized = img.resize((round(width * scale), round(height * scale)), Image.Resampling.BICUBIC)
    left = (resized.width - size) // 2
    top = (resized.height - size) // 2
    return resized.crop((left, top, left + size, top + size))


def pil_to_tensor_m11(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1) * 2.0 - 1.0


def tensor_to_pils_m11(x: torch.Tensor) -> List[Image.Image]:
    x = x.detach().float().cpu()
    if x.ndim == 3:
        x = x.unsqueeze(0)
    x = x.clamp(-1.0, 1.0)
    x = ((x + 1.0) * 127.5).round().byte()
    return [Image.fromarray(x[i].permute(1, 2, 0).numpy(), mode="RGB") for i in range(x.shape[0])]


def make_labeled_grid(items: Sequence[Tuple[str, Image.Image]], columns: int = 5, cell_size: int = 192, label_h: int = 28) -> Image.Image:
    if not items:
        raise ValueError("没有可视化内容。")
    columns = max(1, int(columns))
    rows = (len(items) + columns - 1) // columns
    grid = Image.new("RGB", (columns * cell_size, rows * (cell_size + label_h)), (245, 245, 245))
    draw = ImageDraw.Draw(grid)
    for idx, (name, img) in enumerate(items):
        row = idx // columns
        col = idx % columns
        x0 = col * cell_size
        y0 = row * (cell_size + label_h)
        draw.rectangle((x0, y0, x0 + cell_size, y0 + label_h), fill=(32, 32, 32))
        draw.text((x0 + 6, y0 + 7), name[:42], fill=(255, 255, 255))
        grid.paste(img.resize((cell_size, cell_size), Image.Resampling.BILINEAR), (x0, y0 + label_h))
    return grid


def apply_spatial_operator(tensor: torch.Tensor, transform: str) -> torch.Tensor:
    if transform == "identity":
        return tensor
    if transform == "rot90":
        return torch.rot90(tensor, k=1, dims=(-2, -1))
    if transform == "rot180":
        return torch.rot90(tensor, k=2, dims=(-2, -1))
    if transform == "rot270":
        return torch.rot90(tensor, k=3, dims=(-2, -1))
    if transform == "flip_h":
        return torch.flip(tensor, dims=(-1,))
    if transform == "flip_v":
        return torch.flip(tensor, dims=(-2,))
    raise ValueError(f"未知变换：{transform}")


P = apply_spatial_operator


def ensure_square_latent_grid(z: torch.Tensor, context: str = "latent") -> None:
    if z.ndim != 4:
        raise ValueError(f"{context} 必须是 [B,C,H,W]，当前 shape={tuple(z.shape)}")
    if z.shape[-2] != z.shape[-1]:
        raise ValueError(
            f"{context} 的 token grid 必须是方形才能做第一阶段 D4 群作用实验，"
            f"当前 H={z.shape[-2]}, W={z.shape[-1]}。"
        )


def flatten_tokens(z: torch.Tensor) -> torch.Tensor:
    if z.ndim != 4:
        raise ValueError(f"latent 必须是 [B,C,H,W]，当前 shape={tuple(z.shape)}")
    return z.permute(0, 2, 3, 1).reshape(-1, z.shape[1]).float()


def unflatten_tokens(rows: torch.Tensor, template: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = template.shape
    return rows.reshape(batch, height, width, channels).permute(0, 3, 1, 2).contiguous()


def apply_channel_map(z: torch.Tensor, channel_map: torch.Tensor) -> torch.Tensor:
    rows = flatten_tokens(z)
    channel_map = channel_map.to(device=rows.device, dtype=rows.dtype)
    return unflatten_tokens(rows @ channel_map, z)


def lambda_key(value: float) -> str:
    return f"{float(value):.0e}"


def freeze_module(module: torch.nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)


def default_hf_scaling_factor(model_name: str, config_scaling, fallback: Optional[float]) -> float:
    if fallback is not None:
        return float(fallback)
    if config_scaling is not None:
        return float(config_scaling)
    if model_name in {"stabilityai/sd-vae-ft-mse", "stabilityai/sd-vae-ft-ema"}:
        return 0.18215
    return 1.0


def load_model_adapter(
    model_name: str,
    device: torch.device,
    config: Phase1Config,
    dtype: torch.dtype = torch.float32,
):
    if model_name in RAE_SPECS:
        adapter = load_rae_adapter(
            model_name,
            repo_path=ROOT / config.rae_repo_path,
            device=device,
            dtype=dtype,
            auto_clone=config.rae_auto_clone,
            auto_download=config.rae_auto_download,
        )
        freeze_module(adapter.model)
        return adapter

    if model_name in HF_VAE_SPECS:
        from diffusers.models import AutoencoderKL

        spec = HF_VAE_SPECS[model_name]
        vae = AutoencoderKL.from_pretrained(spec["repo_id"]).to(device=device, dtype=dtype).eval()
        freeze_module(vae)
        scaling = default_hf_scaling_factor(spec["repo_id"], getattr(vae.config, "scaling_factor", None), spec["scaling_factor"])
        return HFVAEAdapter(model_name, vae, device, dtype, scaling)

    raise KeyError(f"未知模型 {model_name}；可选 RAE={list(RAE_SPECS)} HF-VAE={list(HF_VAE_SPECS)}")


@torch.no_grad()
def encode_adapter(adapter, x: torch.Tensor, posterior: str) -> torch.Tensor:
    return adapter.encode(x, posterior=posterior).float()


def init_ridge_stats(channels: int, transforms: Iterable[str], device: torch.device) -> Dict[str, Dict[str, torch.Tensor]]:
    stats = {}
    for transform in transforms:
        stats[transform] = {
            "xtx": torch.zeros((channels, channels), device=device, dtype=torch.float32),
            "xty": torch.zeros((channels, channels), device=device, dtype=torch.float32),
            "yty": torch.zeros((), device=device, dtype=torch.float32),
            "n_tokens": torch.zeros((), device=device, dtype=torch.float32),
        }
    return stats


@torch.no_grad()
def accumulate_ridge_stats(adapter, loader: DataLoader, transforms: Sequence[str], device: torch.device, posterior: str):
    stats = None
    for x, _ in loader:
        x = x.to(device=device, dtype=torch.float32, non_blocking=True)
        z = encode_adapter(adapter, x, posterior)
        ensure_square_latent_grid(z, "encoder latent")
        if stats is None:
            stats = init_ridge_stats(z.shape[1], transforms, device)

        for transform in transforms:
            y = z if transform == "identity" else encode_adapter(adapter, apply_spatial_operator(x, transform), posterior)
            pz = apply_spatial_operator(z, transform)
            x_rows = flatten_tokens(pz)
            y_rows = flatten_tokens(y)
            stats[transform]["xtx"].add_(x_rows.T @ x_rows)
            stats[transform]["xty"].add_(x_rows.T @ y_rows)
            stats[transform]["yty"].add_(y_rows.pow(2).sum())
            stats[transform]["n_tokens"].add_(float(x_rows.shape[0]))

    if stats is None:
        raise ValueError("loader 为空，无法累计 ridge 统计量。")
    return stats


def all_reduce_stats(stats: Mapping[str, Mapping[str, torch.Tensor]], ctx: DistributedContext) -> None:
    if not ctx.distributed:
        return
    for transform_stats in stats.values():
        for tensor in transform_stats.values():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)


def fit_ridge_maps(stats: Mapping[str, Mapping[str, torch.Tensor]], lambdas: Sequence[float]) -> Dict[str, Dict[str, torch.Tensor]]:
    maps_by_lambda: Dict[str, Dict[str, torch.Tensor]] = {}
    for ridge_lambda in lambdas:
        key = lambda_key(ridge_lambda)
        maps_by_lambda[key] = {}
        for transform, transform_stats in stats.items():
            xtx = transform_stats["xtx"]
            xty = transform_stats["xty"]
            channels = xtx.shape[0]
            identity = torch.eye(channels, device=xtx.device, dtype=xtx.dtype)
            if transform == "identity":
                maps_by_lambda[key][transform] = identity
                continue
            scale = (torch.trace(xtx) / max(1, channels)).clamp_min(1e-12)
            lhs = xtx + float(ridge_lambda) * scale * identity
            maps_by_lambda[key][transform] = torch.linalg.solve(lhs, xty)
    return maps_by_lambda


def matrix_reconstruction_loss(transform_stats: Mapping[str, torch.Tensor], channel_map: torch.Tensor) -> torch.Tensor:
    xtx = transform_stats["xtx"]
    xty = transform_stats["xty"]
    yty = transform_stats["yty"].clamp_min(1e-12)
    xtx_map = xtx @ channel_map
    numerator = torch.trace(channel_map.T @ xtx_map) - 2.0 * torch.trace(channel_map.T @ xty) + transform_stats["yty"]
    return numerator.clamp_min(0.0) / yty


def generator_channel_maps(rot90: torch.Tensor, flip_h: torch.Tensor) -> Dict[str, torch.Tensor]:
    channels = rot90.shape[0]
    identity = torch.eye(channels, device=rot90.device, dtype=rot90.dtype)
    rot180 = rot90 @ rot90
    rot270 = rot180 @ rot90
    return {
        "identity": identity,
        "rot90": rot90,
        "rot180": rot180,
        "rot270": rot270,
        "flip_h": flip_h,
        "flip_v": flip_h @ rot180,
    }


def squared_relative_frobenius(target: torch.Tensor, estimate: torch.Tensor) -> torch.Tensor:
    numerator = torch.linalg.norm(target - estimate).pow(2)
    denominator = torch.linalg.norm(target).pow(2).clamp_min(1e-12)
    return numerator / denominator


def generator_law_loss(rot90: torch.Tensor, flip_h: torch.Tensor) -> torch.Tensor:
    channels = rot90.shape[0]
    identity = torch.eye(channels, device=rot90.device, dtype=rot90.dtype)
    rot180 = rot90 @ rot90
    rot270 = rot180 @ rot90
    rot360 = rot270 @ rot90
    flip_v = flip_h @ rot180
    terms = [
        squared_relative_frobenius(identity, rot360),
        squared_relative_frobenius(identity, flip_h @ flip_h),
        squared_relative_frobenius(identity, flip_v @ flip_v),
        squared_relative_frobenius(rot270, flip_h @ rot90 @ flip_h),
    ]
    return torch.stack(terms).mean()


def generator_anchor_loss(
    maps: Mapping[str, torch.Tensor],
    anchors: Mapping[str, torch.Tensor],
    transforms: Sequence[str],
) -> torch.Tensor:
    terms = []
    for transform in transforms:
        if transform == "identity":
            continue
        anchor = anchors[transform].detach()
        terms.append(squared_relative_frobenius(anchor, maps[transform]))
    return torch.stack(terms).mean() if terms else torch.zeros((), device=next(iter(maps.values())).device)


def generator_task_loss(
    stats: Mapping[str, Mapping[str, torch.Tensor]],
    maps: Mapping[str, torch.Tensor],
    transforms: Sequence[str],
) -> torch.Tensor:
    terms = []
    for transform in transforms:
        if transform == "identity":
            continue
        terms.append(matrix_reconstruction_loss(stats[transform], maps[transform]))
    return torch.stack(terms).mean()


def fit_generator_maps(
    stats: Mapping[str, Mapping[str, torch.Tensor]],
    init_maps: Mapping[str, torch.Tensor],
    transforms: Sequence[str],
    law_weight: float,
    anchor_weight: float,
    steps: int,
    lr: float,
    grad_clip: float,
    ctx: DistributedContext,
    model_name: str,
) -> Dict[str, torch.Tensor]:
    rot90 = init_maps["rot90"].detach().clone().requires_grad_(True)
    flip_h = init_maps["flip_h"].detach().clone().requires_grad_(True)
    optimizer = torch.optim.AdamW([rot90, flip_h], lr=lr, weight_decay=0.0)
    log_every = max(1, steps // 5)

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        maps = generator_channel_maps(rot90, flip_h)
        task = generator_task_loss(stats, maps, transforms)
        law = generator_law_loss(rot90, flip_h)
        anchor = generator_anchor_loss(maps, init_maps, transforms)
        loss = task + float(law_weight) * law + float(anchor_weight) * anchor
        if not torch.isfinite(loss):
            raise RuntimeError(f"{model_name} generator loss is not finite at step {step}: {float(loss.detach().cpu())}")
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([rot90, flip_h], grad_clip)
        optimizer.step()

        if ctx.is_main and (step == 0 or (step + 1) % log_every == 0 or step + 1 == steps):
            print(
                f"{model_name}: generator law_weight={law_weight:g} "
                f"step={step + 1}/{steps} loss={float(loss.detach().cpu()):.6f} "
                f"task={float(task.detach().cpu()):.6f} law={float(law.detach().cpu()):.6f} "
                f"anchor={float(anchor.detach().cpu()):.6f}",
                flush=True,
            )

    return {name: value.detach() for name, value in generator_channel_maps(rot90, flip_h).items()}


def init_metric_sums(
    transforms: Sequence[str],
    map_keys: Sequence[str],
    device: torch.device,
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, Dict[str, Dict[str, torch.Tensor]]]]:
    base = {
        transform: {
            "err_p_num": torch.zeros((), device=device, dtype=torch.float32),
            "den": torch.zeros((), device=device, dtype=torch.float32),
        }
        for transform in transforms
    }
    mapped = {
        map_key: {
            transform: {"err_pc_num": torch.zeros((), device=device, dtype=torch.float32)}
            for transform in transforms
        }
        for map_key in map_keys
    }
    return base, mapped


@torch.no_grad()
def evaluate_maps(
    adapter,
    loader: DataLoader,
    transforms: Sequence[str],
    maps_by_key: Mapping[str, Mapping[str, torch.Tensor]],
    device: torch.device,
    posterior: str,
):
    base_sums, mapped_sums = init_metric_sums(transforms, list(maps_by_key), device)
    for x, _ in loader:
        x = x.to(device=device, dtype=torch.float32, non_blocking=True)
        z = encode_adapter(adapter, x, posterior)
        ensure_square_latent_grid(z, "encoder latent")
        for transform in transforms:
            y = z if transform == "identity" else encode_adapter(adapter, apply_spatial_operator(x, transform), posterior)
            pz = apply_spatial_operator(z, transform)
            base_sums[transform]["err_p_num"].add_((y - pz).pow(2).sum())
            base_sums[transform]["den"].add_(y.pow(2).sum())
            for map_key, maps in maps_by_key.items():
                pred = apply_channel_map(pz, maps[transform])
                mapped_sums[map_key][transform]["err_pc_num"].add_((y - pred).pow(2).sum())

    if dist.is_initialized():
        for transform in transforms:
            dist.all_reduce(base_sums[transform]["err_p_num"], op=dist.ReduceOp.SUM)
            dist.all_reduce(base_sums[transform]["den"], op=dist.ReduceOp.SUM)
        for map_key in maps_by_key:
            for transform in transforms:
                dist.all_reduce(mapped_sums[map_key][transform]["err_pc_num"], op=dist.ReduceOp.SUM)

    result: Dict[str, Dict[str, dict]] = {}
    for map_key in maps_by_key:
        result[map_key] = {}
        for transform in transforms:
            den = base_sums[transform]["den"].clamp_min(1e-12)
            err_p = torch.sqrt(base_sums[transform]["err_p_num"] / den)
            err_pc = torch.sqrt(mapped_sums[map_key][transform]["err_pc_num"] / den)
            ratio = err_pc / err_p if err_p.item() > 1e-8 else None
            result[map_key][transform] = {
                "err_p": float(err_p.detach().cpu()),
                "err_pc": float(err_pc.detach().cpu()),
                "ratio_pc_over_p": None if ratio is None else float(ratio.detach().cpu()),
                "gain": None if ratio is None else float((1.0 - ratio).detach().cpu()),
            }
    return result


def select_lambdas(val_metrics: Mapping[str, Mapping[str, dict]], transforms: Sequence[str]) -> Dict[str, str]:
    selected: Dict[str, str] = {}
    for transform in transforms:
        best_key = min(val_metrics.keys(), key=lambda key: val_metrics[key][transform]["err_pc"])
        selected[transform] = best_key
    return selected


def select_maps(maps_by_lambda: Mapping[str, Mapping[str, torch.Tensor]], selected_lambdas: Mapping[str, str]) -> Dict[str, torch.Tensor]:
    return {
        transform: maps_by_lambda[lambda_key][transform]
        for transform, lambda_key in selected_lambdas.items()
    }


def frobenius_relative_error(target: torch.Tensor, estimate: torch.Tensor) -> float:
    numerator = torch.linalg.norm(target - estimate)
    denominator = torch.linalg.norm(target).clamp_min(1e-12)
    return float((numerator / denominator).detach().cpu())


def group_law_metrics(channel_maps: Mapping[str, torch.Tensor]) -> Dict[str, float]:
    first_map = next(iter(channel_maps.values()))
    identity = torch.eye(first_map.shape[0], device=first_map.device, dtype=first_map.dtype)
    metrics = {
        "rot180_vs_rot90_rot90": frobenius_relative_error(channel_maps["rot180"], channel_maps["rot90"] @ channel_maps["rot90"]),
        "rot270_vs_rot90_rot180": frobenius_relative_error(channel_maps["rot270"], channel_maps["rot90"] @ channel_maps["rot180"]),
        "rot90_cycle4_vs_identity": frobenius_relative_error(identity, channel_maps["rot90"] @ channel_maps["rot90"] @ channel_maps["rot90"] @ channel_maps["rot90"]),
        "flip_h_square_vs_identity": frobenius_relative_error(identity, channel_maps["flip_h"] @ channel_maps["flip_h"]),
        "flip_v_square_vs_identity": frobenius_relative_error(identity, channel_maps["flip_v"] @ channel_maps["flip_v"]),
    }
    metrics["mean_composition_error"] = float(np.mean(list(metrics.values())))
    return metrics


def assess_success(train_metrics: Mapping[str, dict], test_metrics: Mapping[str, dict], group_metrics: Mapping[str, float]) -> Dict[str, object]:
    def effective_ratio(metrics: Mapping[str, float | None]) -> float:
        ratio = metrics["ratio_pc_over_p"]
        if ratio is not None:
            return float(ratio)
        return 0.0 if metrics["err_pc"] <= 1e-6 else float("inf")

    ratios = [effective_ratio(test_metrics[g]) for g in NON_IDENTITY_TRANSFORMS if g in test_metrics]
    gains = [1.0 - ratio for ratio in ratios]
    train_err = float(np.mean([train_metrics[g]["err_pc"] for g in NON_IDENTITY_TRANSFORMS if g in train_metrics]))
    test_err = float(np.mean([test_metrics[g]["err_pc"] for g in NON_IDENTITY_TRANSFORMS if g in test_metrics]))
    train_test_gap = abs(test_err - train_err) / max(train_err, 1e-12)
    checks = {
        "mean_ratio_le_0_70": float(np.mean(ratios)) <= 0.70,
        "max_ratio_le_0_85": float(np.max(ratios)) <= 0.85,
        "four_gains_ge_0_25": int(sum(gain >= 0.25 for gain in gains)) >= 4,
        "train_test_gap_le_0_15": train_test_gap <= 0.15,
        "composition_mean_le_0_25": group_metrics["mean_composition_error"] <= 0.25,
    }
    return {
        "passes_thresholds": bool(all(checks.values())),
        "checks": checks,
        "mean_ratio_pc_over_p": float(np.mean(ratios)),
        "max_ratio_pc_over_p": float(np.max(ratios)),
        "num_gains_ge_0_25": int(sum(gain >= 0.25 for gain in gains)),
        "train_test_err_pc_gap": float(train_test_gap),
    }


def cpu_maps(channel_maps: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {transform: channel_map.detach().cpu() for transform, channel_map in channel_maps.items()}


def save_model_artifact(
    artifact_dir: Path,
    model_name: str,
    channel_maps: Mapping[str, torch.Tensor],
    selected_lambdas: Mapping[str, str],
    model_summary: Mapping[str, object],
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": model_name,
            "map_orientation": "row_map: flattened_latent @ C_tilde approximates E(gx); this is C_g^T in the plan notation",
            "channel_maps": cpu_maps(channel_maps),
            "selected_lambdas": dict(selected_lambdas),
            "summary": dict(model_summary),
        },
        artifact_dir / f"{model_name}_maps.pt",
    )


def save_generator_artifact(
    artifact_dir: Path,
    model_name: str,
    channel_maps: Mapping[str, torch.Tensor],
    generator_summary: Mapping[str, object],
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": model_name,
            "map_orientation": "row_map: flattened_latent @ C_tilde approximates E(gx); this is C_g^T in the plan notation",
            "fit_type": "generator_constrained",
            "channel_maps": cpu_maps(channel_maps),
            "summary": dict(generator_summary),
        },
        artifact_dir / f"{model_name}_generator_maps.pt",
    )


def write_summary(artifact_dir: Path, summary: Mapping[str, object]) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with (artifact_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def mean_non_identity(metrics: Mapping[str, Mapping[str, object]], key: str) -> float:
    values = [metrics[g][key] for g in NON_IDENTITY_TRANSFORMS if g in metrics and metrics[g][key] is not None]
    return float(np.mean(values)) if values else float("nan")


def generator_selection_score(
    val_metrics: Mapping[str, Mapping[str, object]],
    laws: Mapping[str, float],
    law_score_weight: float,
) -> float:
    return mean_non_identity(val_metrics, "ratio_pc_over_p") + float(law_score_weight) * float(laws["mean_composition_error"])


def fit_and_evaluate_generators(
    model_name: str,
    stats: Mapping[str, Mapping[str, torch.Tensor]],
    init_maps: Mapping[str, torch.Tensor],
    adapter,
    loaders: Mapping[str, DataLoader],
    config: Phase1Config,
    ctx: DistributedContext,
) -> Dict[str, object]:
    candidates = {}
    best_key = None
    best_score = float("inf")
    best_maps = None

    for law_weight in config.generator_law_weights:
        candidate_key = f"law_{law_weight:g}"
        log_main(ctx, f"{model_name}: 训练生成元约束 C_g ({candidate_key})")
        maps = fit_generator_maps(
            stats=stats,
            init_maps=init_maps,
            transforms=config.transforms,
            law_weight=float(law_weight),
            anchor_weight=config.generator_anchor_weight,
            steps=config.generator_steps,
            lr=config.generator_lr,
            grad_clip=config.generator_grad_clip,
            ctx=ctx,
            model_name=model_name,
        )
        val_metrics = evaluate_maps(adapter, loaders["val"], config.transforms, {candidate_key: maps}, ctx.device, config.posterior)[candidate_key]
        laws = group_law_metrics(maps)
        score = generator_selection_score(val_metrics, laws, config.generator_score_law_weight)
        candidates[candidate_key] = {
            "law_weight": float(law_weight),
            "selection_score": float(score),
            "val_mean_ratio": mean_non_identity(val_metrics, "ratio_pc_over_p"),
            "val_mean_group_law": float(laws["mean_composition_error"]),
            "val_metrics": val_metrics,
            "group_law": laws,
        }
        if score < best_score:
            best_score = score
            best_key = candidate_key
            best_maps = maps

    if best_key is None or best_maps is None:
        raise RuntimeError("没有成功训练任何 generator candidate。")

    selected_key = "generator_selected"
    train_metrics = evaluate_maps(adapter, loaders["train"], config.transforms, {selected_key: best_maps}, ctx.device, config.posterior)[selected_key]
    val_metrics = evaluate_maps(adapter, loaders["val"], config.transforms, {selected_key: best_maps}, ctx.device, config.posterior)[selected_key]
    test_metrics = evaluate_maps(adapter, loaders["test"], config.transforms, {selected_key: best_maps}, ctx.device, config.posterior)[selected_key]
    laws = group_law_metrics(best_maps)
    success = assess_success(train_metrics, test_metrics, laws)

    return {
        "selected_candidate": best_key,
        "selection_score": float(best_score),
        "maps": best_maps,
        "summary": {
            "selected_candidate": best_key,
            "selection_score": float(best_score),
            "candidates": candidates,
            "metrics": {
                "train": train_metrics,
                "val": val_metrics,
                "test": test_metrics,
            },
            "group_law": laws,
            "success": success,
        },
    }


def run_one_model(model_name: str, config: Phase1Config, loaders: Mapping[str, DataLoader], ctx: DistributedContext) -> Dict[str, object]:
    log_main(ctx, f"\n==> {model_name}: 加载 frozen adapter")
    adapter = load_model_adapter(model_name, ctx.device, config, dtype=torch.float32)

    log_main(ctx, f"{model_name}: 累计 train split 的 X^T X / X^T Y")
    stats = accumulate_ridge_stats(adapter, loaders["train"], config.transforms, ctx.device, config.posterior)
    all_reduce_stats(stats, ctx)

    log_main(ctx, f"{model_name}: 拟合 ridge C_g 并用 val split 选择 lambda")
    maps_by_lambda = fit_ridge_maps(stats, config.lambdas)
    val_by_lambda = evaluate_maps(adapter, loaders["val"], config.transforms, maps_by_lambda, ctx.device, config.posterior)
    selected_lambdas = select_lambdas(val_by_lambda, config.transforms)
    selected_maps = select_maps(maps_by_lambda, selected_lambdas)
    selected_key = "selected"
    selected_maps_by_key = {selected_key: selected_maps}

    train_metrics = evaluate_maps(adapter, loaders["train"], config.transforms, selected_maps_by_key, ctx.device, config.posterior)[selected_key]
    val_metrics = evaluate_maps(adapter, loaders["val"], config.transforms, selected_maps_by_key, ctx.device, config.posterior)[selected_key]
    test_metrics = evaluate_maps(adapter, loaders["test"], config.transforms, selected_maps_by_key, ctx.device, config.posterior)[selected_key]
    laws = group_law_metrics(selected_maps)
    success = assess_success(train_metrics, test_metrics, laws)

    model_summary = {
        "fit_type": "independent_ridge",
        "selected_lambdas": selected_lambdas,
        "metrics": {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
            "val_all_lambdas": val_by_lambda,
        },
        "group_law": laws,
        "success": success,
    }

    if ctx.is_main:
        artifact_dir = ROOT / config.artifact_dir
        save_model_artifact(artifact_dir, model_name, selected_maps, selected_lambdas, model_summary)
        log_main(ctx, f"{model_name}: test mean ratio={success['mean_ratio_pc_over_p']:.4f}, pass={success['passes_thresholds']}")

    if should_fit_generators(config, model_name):
        generator_payload = fit_and_evaluate_generators(
            model_name=model_name,
            stats=stats,
            init_maps=selected_maps,
            adapter=adapter,
            loaders=loaders,
            config=config,
            ctx=ctx,
        )
        model_summary["generator_constrained"] = generator_payload["summary"]
        if ctx.is_main:
            artifact_dir = ROOT / config.artifact_dir
            save_generator_artifact(
                artifact_dir,
                model_name,
                generator_payload["maps"],
                generator_payload["summary"],
            )
            gen_success = generator_payload["summary"]["success"]
            gen_law = generator_payload["summary"]["group_law"]["mean_composition_error"]
            log_main(
                ctx,
                f"{model_name}: generator selected={generator_payload['selected_candidate']} "
                f"test mean ratio={gen_success['mean_ratio_pc_over_p']:.4f}, "
                f"mean law={gen_law:.4f}, pass={gen_success['passes_thresholds']}",
            )

    del adapter
    if ctx.device.type == "cuda":
        torch.cuda.empty_cache()
    return model_summary


def run_experiment(config: Phase1Config) -> Dict[str, object]:
    ctx = setup_distributed()
    try:
        validate_config(config, ctx)
        if ctx.is_main:
            (ROOT / config.artifact_dir).mkdir(parents=True, exist_ok=True)
        distributed_barrier(ctx)

        log_main(ctx, f"设备: {ctx.device}, world_size={ctx.world_size}, fp32/no-TF32")
        loaders = make_loaders(config, ctx)
        summary = {
            "config": asdict(config),
            "world_size": ctx.world_size,
            "map_orientation": "row_map: flattened_latent @ C_tilde; C_tilde is C_g^T in the plan notation",
            "models": {},
        }
        for model_name in config.models:
            model_summary = run_one_model(model_name, config, loaders, ctx)
            if ctx.is_main:
                summary["models"][model_name] = model_summary
                write_summary(ROOT / config.artifact_dir, summary)

        if ctx.is_main:
            write_summary(ROOT / config.artifact_dir, summary)
            log_main(ctx, f"\n完成。结果已写入：{ROOT / config.artifact_dir}")
        return summary
    finally:
        cleanup_distributed(ctx)


def load_phase1_maps(
    artifact_dir: str | Path,
    model_name: str,
    device: Optional[torch.device] = None,
    fit_type: str = "independent",
) -> Dict[str, torch.Tensor]:
    path = Path(artifact_dir)
    if not path.is_absolute():
        path = ROOT / path
    suffix = "_generator_maps.pt" if fit_type in {"generator", "generator_constrained"} else "_maps.pt"
    payload = torch.load(path / f"{model_name}{suffix}", map_location="cpu")
    maps = payload["channel_maps"]
    if device is not None:
        maps = {name: value.to(device=device, dtype=torch.float32) for name, value in maps.items()}
    return maps


class Phase1NotebookSession:
    def __init__(
        self,
        artifact_dir: str | Path = "artifacts/group_action_phase1",
        rae_repo_path: str | Path = "external/RAE",
        device: str | torch.device = "cuda:0",
        image_size: int = 256,
        posterior: str = "mode",
        fit_type: str = "independent",
    ):
        configure_fp32()
        self.device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
        self.image_size = int(image_size)
        self.posterior = posterior
        self.fit_type = fit_type
        self.config = Phase1Config(artifact_dir=str(artifact_dir), rae_repo_path=str(rae_repo_path), image_size=image_size)
        self.adapters = {}
        self.map_cache = {}

    def adapter(self, model_name: str):
        if model_name not in self.adapters:
            self.adapters[model_name] = load_model_adapter(model_name, self.device, self.config, dtype=torch.float32)
        return self.adapters[model_name]

    def E(self, model_name: str, x: torch.Tensor) -> torch.Tensor:
        return encode_adapter(self.adapter(model_name), x.to(self.device, dtype=torch.float32), self.posterior)

    def D(self, model_name: str, z: torch.Tensor) -> torch.Tensor:
        return self.adapter(model_name).decode(z.to(self.device, dtype=torch.float32)).float()

    def maps(self, model_name: str) -> Dict[str, torch.Tensor]:
        cache_key = (model_name, self.fit_type)
        if cache_key not in self.map_cache:
            self.map_cache[cache_key] = load_phase1_maps(self.config.artifact_dir, model_name, self.device, fit_type=self.fit_type)
        return self.map_cache[cache_key]

    def C(self, z: torch.Tensor, transform: str, model_name: str = "rae_dinov2") -> torch.Tensor:
        return apply_channel_map(z, self.maps(model_name)[transform])

    def V_group(self, x: torch.Tensor, transform: str = "rot90", model_name: str = "rae_dinov2", cell_size: int = 192) -> Image.Image:
        adapter = self.adapter(model_name)
        maps = self.maps(model_name)
        return visualize_group_action(adapter, x.to(self.device, dtype=torch.float32), transform, maps, self.posterior, cell_size=cell_size)


@torch.no_grad()
def visualize_group_action(
    adapter,
    x: torch.Tensor,
    transform: str,
    channel_maps: Mapping[str, torch.Tensor],
    posterior: str = "mode",
    cell_size: int = 192,
) -> Image.Image:
    x = x.to(adapter.device, dtype=torch.float32)
    x_g = apply_spatial_operator(x, transform)
    z = encode_adapter(adapter, x, posterior)
    ensure_square_latent_grid(z, "encoder latent")
    y = encode_adapter(adapter, x_g, posterior)
    pz = apply_spatial_operator(z, transform)
    pcz = apply_channel_map(pz, channel_maps[transform])
    decoded_p = adapter.decode(pz).float()
    decoded_pc = adapter.decode(pcz).float()
    decoded_y = adapter.decode(y).float()

    images = {
        "x": x,
        f"g(x): {transform}": x_g,
        "D(P_g z)": decoded_p,
        "D(P_g z C_g^T)": decoded_pc,
        "D(E(gx))": decoded_y,
    }
    items: List[Tuple[str, Image.Image]] = []
    batch = x.shape[0]
    for sample_idx in range(batch):
        for label, tensor in images.items():
            items.append((f"{sample_idx}:{label}" if batch > 1 else label, tensor_to_pils_m11(tensor[sample_idx:sample_idx + 1])[0]))
    return make_labeled_grid(items, columns=5, cell_size=cell_size)


def parse_tuple(value: str, cast=str) -> Tuple:
    if not value:
        return tuple()
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1: validate fixed-P_g latent group action for RAE/VAE baselines.")
    parser.add_argument("--data-root", default="/data/shared")
    parser.add_argument("--dataset-name", default="caltech101")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--download-dataset", action="store_true")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-count", type=int, default=512)
    parser.add_argument("--val-count", type=int, default=256)
    parser.add_argument("--test-count", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--models", default="rae_dinov2,rae_mae,rae_siglip2,eqvae,sdvae")
    parser.add_argument("--transforms", default=",".join(TRANSFORMS))
    parser.add_argument("--lambdas", default=",".join(str(v) for v in DEFAULT_LAMBDAS))
    parser.add_argument("--rae-repo-path", default="external/RAE")
    parser.add_argument("--rae-auto-clone", action="store_true")
    parser.add_argument("--rae-auto-download", action="store_true")
    parser.add_argument("--artifact-dir", default="artifacts/group_action_phase1")
    parser.add_argument("--posterior", choices=("mode", "sample"), default="mode")
    parser.add_argument("--fit-generators", action="store_true")
    parser.add_argument("--generator-models", default="")
    parser.add_argument("--generator-steps", type=int, default=300)
    parser.add_argument("--generator-lr", type=float, default=1e-3)
    parser.add_argument("--generator-law-weights", default="0.01,0.1,1.0")
    parser.add_argument("--generator-anchor-weight", type=float, default=0.01)
    parser.add_argument("--generator-score-law-weight", type=float, default=0.5)
    parser.add_argument("--generator-grad-clip", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Phase1Config(
        data_root=args.data_root,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        dataset_path=args.dataset_path,
        download_dataset=args.download_dataset,
        image_size=args.image_size,
        seed=args.seed,
        train_count=args.train_count,
        val_count=args.val_count,
        test_count=args.test_count,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        models=parse_tuple(args.models, str),
        transforms=parse_tuple(args.transforms, str),
        lambdas=parse_tuple(args.lambdas, float),
        rae_repo_path=args.rae_repo_path,
        rae_auto_clone=args.rae_auto_clone,
        rae_auto_download=args.rae_auto_download,
        artifact_dir=args.artifact_dir,
        posterior=args.posterior,
        fit_generators=args.fit_generators,
        generator_models=parse_tuple(args.generator_models, str),
        generator_steps=args.generator_steps,
        generator_lr=args.generator_lr,
        generator_law_weights=parse_tuple(args.generator_law_weights, float),
        generator_anchor_weight=args.generator_anchor_weight,
        generator_score_law_weight=args.generator_score_law_weight,
        generator_grad_clip=args.generator_grad_clip,
    )
    run_experiment(config)


if __name__ == "__main__":
    main()
