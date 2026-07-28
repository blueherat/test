from __future__ import annotations

import bisect
import io
import json
import math
from collections import OrderedDict
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from baselines.visual_adapters import RAE_SPECS, load_rae_adapter


TRANSFORMS: Tuple[str, ...] = (
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "flip_h",
    "flip_v",
    "translate_right",
    "translate_left",
    "translate_up",
    "translate_down",
    "zoom_in",
    "zoom_out",
)
NON_IDENTITY_TRANSFORMS: Tuple[str, ...] = tuple(g for g in TRANSFORMS if g != "identity")
C4_TRANSFORMS: Tuple[str, ...] = ("identity", "rot90", "rot180", "rot270")

HF_VAE_SPECS: Dict[str, dict] = {
    "eqvae": {"repo_id": "zelaki/eq-vae", "scaling_factor": None},
    "eqvae_ema": {"repo_id": "zelaki/eq-vae-ema", "scaling_factor": None},
    "sdvae": {"repo_id": "stabilityai/sd-vae-ft-mse", "scaling_factor": 0.18215},
    "sdvae_ft_mse": {"repo_id": "stabilityai/sd-vae-ft-mse", "scaling_factor": 0.18215},
    "sdvae_ft_ema": {"repo_id": "stabilityai/sd-vae-ft-ema", "scaling_factor": 0.18215},
}


@dataclass
class HFVAEAdapter:
    key: str
    model: torch.nn.Module
    device: torch.device
    dtype: torch.dtype
    scaling_factor: float
    posterior: str = "mode"

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        x = x.to(device=self.device, dtype=self.dtype)
        latent_dist = self.model.encode(x).latent_dist
        z = latent_dist.mode() if self.posterior == "mode" else latent_dist.sample()
        return z * self.scaling_factor

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        z = z.to(device=self.device, dtype=self.dtype)
        return self.model.decode(z / self.scaling_factor).sample.clamp(-1.0, 1.0)


def configure_fp32() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")


def _translate_fractional(tensor: torch.Tensor, dx_frac: float = 0.0, dy_frac: float = 0.0) -> torch.Tensor:
    height, width = tensor.shape[-2:]
    dx = int(round(width * dx_frac))
    dy = int(round(height * dy_frac))
    if dx_frac and dx == 0:
        dx = 1 if dx_frac > 0 else -1
    if dy_frac and dy == 0:
        dy = 1 if dy_frac > 0 else -1
    return torch.roll(tensor, shifts=(dy, dx), dims=(-2, -1))


def _center_scale(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    if scale <= 0:
        raise ValueError(f"scale 必须为正数，收到：{scale}")
    squeeze = tensor.ndim == 3
    if tensor.ndim not in (3, 4):
        raise ValueError(f"缩放算子只支持 CHW/BCHW tensor，收到维度：{tensor.ndim}")
    x = tensor.unsqueeze(0) if squeeze else tensor
    height, width = x.shape[-2:]
    new_height = max(1, int(round(height * scale)))
    new_width = max(1, int(round(width * scale)))
    scaled = F.interpolate(x.float(), size=(new_height, new_width), mode="bilinear", align_corners=False)
    if new_height >= height:
        top = (new_height - height) // 2
        scaled = scaled[..., top:top + height, :]
    else:
        pad_top = (height - new_height) // 2
        pad_bottom = height - new_height - pad_top
        scaled = F.pad(scaled, (0, 0, pad_top, pad_bottom), value=0.0)
    if new_width >= width:
        left = (new_width - width) // 2
        scaled = scaled[..., left:left + width]
    else:
        pad_left = (width - new_width) // 2
        pad_right = width - new_width - pad_left
        scaled = F.pad(scaled, (pad_left, pad_right, 0, 0), value=0.0)
    scaled = scaled.to(dtype=tensor.dtype)
    return scaled.squeeze(0) if squeeze else scaled


def apply_d4(tensor: torch.Tensor, transform: str) -> torch.Tensor:
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
    if transform in {"translate_right", "shift_right"}:
        return _translate_fractional(tensor, dx_frac=1.0 / 8.0)
    if transform in {"translate_left", "shift_left"}:
        return _translate_fractional(tensor, dx_frac=-1.0 / 8.0)
    if transform in {"translate_up", "shift_up"}:
        return _translate_fractional(tensor, dy_frac=-1.0 / 8.0)
    if transform in {"translate_down", "shift_down"}:
        return _translate_fractional(tensor, dy_frac=1.0 / 8.0)
    if transform in {"zoom_in", "scale_up"}:
        return _center_scale(tensor, 1.25)
    if transform in {"zoom_out", "scale_down"}:
        return _center_scale(tensor, 0.80)
    raise ValueError(f"未知变换：{transform}")


P = apply_d4


def inverse_transform(transform: str) -> str:
    inverse = {
        "identity": "identity",
        "rot90": "rot270",
        "rot180": "rot180",
        "rot270": "rot90",
        "flip_h": "flip_h",
        "flip_v": "flip_v",
        "translate_right": "translate_left",
        "translate_left": "translate_right",
        "translate_up": "translate_down",
        "translate_down": "translate_up",
        "shift_right": "shift_left",
        "shift_left": "shift_right",
        "shift_up": "shift_down",
        "shift_down": "shift_up",
        "zoom_in": "zoom_out",
        "zoom_out": "zoom_in",
        "scale_up": "scale_down",
        "scale_down": "scale_up",
    }
    if transform not in inverse:
        raise ValueError(f"未知变换：{transform}")
    return inverse[transform]


def rotation_power(transform: str) -> int:
    powers = {"identity": 0, "rot90": 1, "rot180": 2, "rot270": 3}
    if transform not in powers:
        raise ValueError(f"orbit consistency matrix 当前只支持 C4 旋转，收到：{transform}")
    return powers[transform]


def rotation_from_power(power: int) -> str:
    return C4_TRANSFORMS[int(power) % 4]


def relative_c4_transform(target: str, source: str) -> str:
    return rotation_from_power(rotation_power(target) - rotation_power(source))


def center_crop_resize(img: Image.Image, size: int) -> Image.Image:
    width, height = img.size
    scale = size / min(width, height)
    resized = img.resize((round(width * scale), round(height * scale)), Image.Resampling.BICUBIC)
    left = (resized.width - size) // 2
    top = (resized.height - size) // 2
    return resized.crop((left, top, left + size, top + size))


def pil_to_tensor_m11(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1) * 2.0 - 1.0


def tensor_to_image01(x: torch.Tensor) -> np.ndarray:
    x = x.detach().float().cpu()
    if x.ndim == 4:
        x = x[0]
    x = x.clamp(-1.0, 1.0)
    x = ((x + 1.0) * 0.5).permute(1, 2, 0).numpy()
    return x.clip(0.0, 1.0)


def normalize_dataset_name(name: str) -> str:
    return (name or "caltech101").strip().lower().replace("-", "_")


class ParquetImageDataset:
    """Minimal random-access image dataset for HF parquet shards."""

    def __init__(self, root: str | Path, split: str = "test"):
        self.root = Path(root).expanduser()
        self.data_dir = self.root / "data" if (self.root / "data").exists() else self.root
        split_alias = {"val": "validation", "valid": "validation"}
        self.split = split_alias.get(split, split)
        self.files = sorted(self.data_dir.glob(f"{self.split}-*.parquet"))
        if not self.files:
            raise FileNotFoundError(
                f"没有找到 {self.split} parquet 分片：{self.data_dir}/{self.split}-*.parquet"
            )
        import pyarrow.parquet as pq

        self._pq = pq
        self._metadata_rows = []
        self._offsets = [0]
        for path in self.files:
            rows = int(pq.ParquetFile(path).metadata.num_rows)
            self._metadata_rows.append(rows)
            self._offsets.append(self._offsets[-1] + rows)
        self._pf_cache = {}
        self._row_group_offsets = {}
        self._row_group_cache = OrderedDict()
        self._row_group_cache_size = 4

    def __len__(self) -> int:
        return self._offsets[-1]

    def _parquet_file(self, file_index: int):
        path = self.files[file_index]
        if path not in self._pf_cache:
            self._pf_cache[path] = self._pq.ParquetFile(path)
        return self._pf_cache[path]

    def _row_group_for(self, file_index: int, local_index: int) -> Tuple[int, int]:
        if file_index not in self._row_group_offsets:
            pf = self._parquet_file(file_index)
            offsets = [0]
            for row_group in range(pf.num_row_groups):
                offsets.append(offsets[-1] + pf.metadata.row_group(row_group).num_rows)
            self._row_group_offsets[file_index] = offsets
        offsets = self._row_group_offsets[file_index]
        row_group = bisect.bisect_right(offsets, local_index) - 1
        return row_group, local_index - offsets[row_group]

    def _row_group_table(self, file_index: int, row_group: int):
        key = (int(file_index), int(row_group))
        if key in self._row_group_cache:
            self._row_group_cache.move_to_end(key)
            return self._row_group_cache[key]
        table = self._parquet_file(file_index).read_row_group(row_group, columns=["image", "label"])
        self._row_group_cache[key] = table
        self._row_group_cache.move_to_end(key)
        while len(self._row_group_cache) > self._row_group_cache_size:
            self._row_group_cache.popitem(last=False)
        return table

    def __getitem__(self, index: int):
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        file_index = bisect.bisect_right(self._offsets, index) - 1
        local_index = index - self._offsets[file_index]
        row_group, row_in_group = self._row_group_for(file_index, local_index)
        table = self._row_group_table(file_index, row_group).slice(row_in_group, 1)
        row = table.to_pydict()
        image_info = row["image"][0]
        image_bytes = image_info.get("bytes")
        if image_bytes is None:
            image_path = image_info.get("path")
            if image_path is None:
                raise ValueError(f"parquet image row 缺少 bytes/path：{self.files[file_index]}")
            image = Image.open(self.data_dir / image_path).convert("RGB")
        else:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return image, int(row["label"][0])


def load_named_dataset(name: str, root: str, split: str = "train", download: bool = False, dataset_path: str = ""):
    from torchvision.datasets import CIFAR10, CIFAR100, STL10, Caltech101, Flowers102, ImageFolder, OxfordIIITPet

    name = normalize_dataset_name(name)
    split = (split or "train").strip().lower()
    if name in {"imagenet_parquet", "imagenet1k_parquet", "imagenet_1k_parquet"}:
        path = Path(dataset_path).expanduser() if dataset_path else Path(root).expanduser() / "imagenet-1k"
        return ParquetImageDataset(path, split=split)
    if name == "image_folder":
        folder = Path(dataset_path).expanduser() if dataset_path else Path(root).expanduser()
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
        flower_split = flower_split if flower_split in {"train", "val", "test"} else "train"
        return Flowers102(root=root, split=flower_split, download=download)
    if name in {"oxford_iiit_pet", "oxford_pet", "pets"}:
        return OxfordIIITPet(root=root, split="test" if split == "test" else "trainval", download=download)
    raise ValueError(
        "不支持的数据集："
        f"{name}。可选：caltech101, stl10, flowers102, oxford_iiit_pet, "
        "cifar10, cifar100, image_folder, imagenet_parquet"
    )


def split_indices(total: int, count: int, seed: int) -> List[int]:
    if count <= 0:
        raise ValueError("count 必须大于 0。")
    if total < count:
        raise ValueError(f"数据集只有 {total} 张，少于请求的 {count} 张。")
    rng = np.random.default_rng(seed)
    return [int(i) for i in rng.permutation(total)[:count]]


def split_train_val_test_indices(
    total: int,
    train_count: int,
    val_count: int,
    test_count: int,
    seed: int,
) -> Dict[str, List[int]]:
    needed = int(train_count) + int(val_count) + int(test_count)
    if min(train_count, val_count, test_count) < 0:
        raise ValueError("train/val/test count 不能为负。")
    if train_count <= 0 or test_count <= 0:
        raise ValueError("train_count 和 test_count 必须大于 0。")
    if total < needed:
        raise ValueError(f"数据集只有 {total} 张，少于请求的 {needed} 张。")
    rng = np.random.default_rng(seed)
    perm = [int(i) for i in rng.permutation(total)]
    train_end = train_count
    val_end = train_count + val_count
    return {
        "train": perm[:train_end],
        "val": perm[train_end:val_end],
        "test": perm[val_end:val_end + test_count],
    }


def pick_dataset_images(
    dataset,
    count: int = 4,
    seed: int = 0,
    indices: Optional[Sequence[int]] = None,
    image_size: int = 256,
) -> Tuple[torch.Tensor, List[int]]:
    chosen = [int(i) for i in indices] if indices is not None else split_indices(len(dataset), count, seed)
    images = []
    for index in chosen:
        sample = dataset[index]
        img = sample[0] if isinstance(sample, (tuple, list)) else sample
        images.append(pil_to_tensor_m11(center_crop_resize(img, image_size)))
    return torch.stack(images), chosen


def default_hf_scaling_factor(repo_id: str, config_scaling, fallback: Optional[float]) -> float:
    if fallback is not None:
        return float(fallback)
    if config_scaling is not None:
        return float(config_scaling)
    if repo_id in {"stabilityai/sd-vae-ft-mse", "stabilityai/sd-vae-ft-ema"}:
        return 0.18215
    return 1.0


def load_dinov2_adapter(
    device: str | torch.device = "cuda:0",
    rae_repo_path: str | Path = "external/RAE",
    auto_clone: bool = False,
    auto_download: bool = False,
):
    configure_fp32()
    device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
    return load_rae_adapter(
        "rae_dinov2",
        repo_path=rae_repo_path,
        device=device,
        dtype=torch.float32,
        auto_clone=auto_clone,
        auto_download=auto_download,
    )


def load_baseline_adapter(
    key: str,
    device: str | torch.device = "cuda:0",
    rae_repo_path: str | Path = "external/RAE",
    rae_auto_clone: bool = False,
    rae_auto_download: bool = False,
    posterior: str = "mode",
):
    configure_fp32()
    device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
    if key in RAE_SPECS:
        return load_rae_adapter(
            key,
            repo_path=rae_repo_path,
            device=device,
            dtype=torch.float32,
            auto_clone=rae_auto_clone,
            auto_download=rae_auto_download,
        )
    if key in HF_VAE_SPECS:
        from diffusers.models import AutoencoderKL

        spec = HF_VAE_SPECS[key]
        model = AutoencoderKL.from_pretrained(spec["repo_id"]).to(device=device, dtype=torch.float32).eval()
        scaling = default_hf_scaling_factor(
            spec["repo_id"],
            getattr(model.config, "scaling_factor", None),
            spec["scaling_factor"],
        )
        return HFVAEAdapter(key=key, model=model, device=device, dtype=torch.float32, scaling_factor=scaling, posterior=posterior)
    raise KeyError(f"未知 baseline: {key}; 可选 {sorted(set(RAE_SPECS) | set(HF_VAE_SPECS))}")


def load_baseline_adapters(
    keys: Sequence[str],
    device: str | torch.device = "cuda:0",
    rae_repo_path: str | Path = "external/RAE",
    rae_auto_clone: bool = False,
    rae_auto_download: bool = False,
    posterior: str = "mode",
) -> Dict[str, object]:
    adapters = {}
    for key in keys:
        adapters[key] = load_baseline_adapter(
            key,
            device=device,
            rae_repo_path=rae_repo_path,
            rae_auto_clone=rae_auto_clone,
            rae_auto_download=rae_auto_download,
            posterior=posterior,
        )
    return adapters


@torch.no_grad()
def run_baseline_diagnostics(
    keys: Sequence[str],
    x: torch.Tensor,
    device: str | torch.device = "cuda:0",
    rae_repo_path: str | Path = "external/RAE",
    rae_auto_clone: bool = False,
    rae_auto_download: bool = False,
    posterior: str = "mode",
    transforms: Sequence[str] = NON_IDENTITY_TRANSFORMS,
    orbit_transforms: Sequence[str] = C4_TRANSFORMS,
    center: str = "sample",
) -> Tuple[List[Dict[str, float | str]], List[Dict[str, float | str]], List[Dict[str, float | str]]]:
    metric_rows: List[Dict[str, float | str]] = []
    procrustes_rows: List[Dict[str, float | str]] = []
    orbit_rows: List[Dict[str, float | str]] = []
    for key in keys:
        adapter = load_baseline_adapter(
            key,
            device=device,
            rae_repo_path=rae_repo_path,
            rae_auto_clone=rae_auto_clone,
            rae_auto_download=rae_auto_download,
            posterior=posterior,
        )
        metric_rows.extend(compare_models_table({key: adapter}, x, transforms=transforms, center=center))
        procrustes_rows.extend(compare_procrustes_table({key: adapter}, x, transforms=transforms))
        orbit_rows.extend(orbit_consistency_table({key: adapter}, x, orbit_transforms=orbit_transforms, center=center))
        del adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return metric_rows, procrustes_rows, orbit_rows


@torch.no_grad()
def run_train_test_procrustes(
    keys: Sequence[str],
    dataset,
    device: str | torch.device = "cuda:0",
    rae_repo_path: str | Path = "external/RAE",
    rae_auto_clone: bool = False,
    rae_auto_download: bool = False,
    posterior: str = "mode",
    transforms: Sequence[str] = NON_IDENTITY_TRANSFORMS,
    centers: Sequence[str] = ("none", "sample"),
    image_size: int = 256,
    train_count: int = 32,
    val_count: int = 16,
    test_count: int = 16,
    seed: int = 0,
    batch_size: int = 16,
    save: bool = False,
    save_json_path: Optional[str | Path] = None,
) -> Tuple[List[Dict[str, float | str | None]], List[Dict[str, float | str]], Dict[str, List[int]]]:
    split_map = split_train_val_test_indices(len(dataset), train_count, val_count, test_count, seed)
    split_tensors = {}
    for split, indices in split_map.items():
        if not indices:
            continue
        split_tensors[split], _ = pick_dataset_images(dataset, indices=indices, image_size=image_size)

    procrustes_rows: List[Dict[str, float | str | None]] = []
    law_rows: List[Dict[str, float | str]] = []
    for key in keys:
        adapter = load_baseline_adapter(
            key,
            device=device,
            rae_repo_path=rae_repo_path,
            rae_auto_clone=rae_auto_clone,
            rae_auto_download=rae_auto_download,
            posterior=posterior,
        )
        for center in centers:
            maps = fit_orthogonal_maps(
                adapter,
                split_tensors["train"],
                transforms=transforms,
                center=center,
                batch_size=batch_size,
            )
            for split, x_split in split_tensors.items():
                for row in evaluate_channel_maps(
                    adapter,
                    x_split,
                    maps,
                    transforms=transforms,
                    center=center,
                    batch_size=batch_size,
                ):
                    procrustes_rows.append({"model": key, "center": center, "split": split, **row})
            law_rows.append({"model": key, "center": center, **group_law_metrics(maps)})
        del adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if save:
        if save_json_path is None:
            raise ValueError("save=True 时必须显式提供 save_json_path。")
        path = Path(save_json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(
                {"procrustes_rows": procrustes_rows, "law_rows": law_rows, "split_indices": split_map},
                f,
                ensure_ascii=False,
                indent=2,
            )

    return procrustes_rows, law_rows, split_map


@torch.no_grad()
def E(adapter, x: torch.Tensor) -> torch.Tensor:
    return adapter.encode(x.to(adapter.device, dtype=torch.float32)).float()


def ensure_square_grid(z: torch.Tensor) -> None:
    if z.ndim != 4:
        raise ValueError(f"latent 必须是 [B,C,H,W]，当前 shape={tuple(z.shape)}")
    if z.shape[-2] != z.shape[-1]:
        raise ValueError(f"当前 token grid 不是方形：H={z.shape[-2]}, W={z.shape[-1]}")


def token_rows(z: torch.Tensor, center: str = "none") -> torch.Tensor:
    ensure_square_grid(z)
    rows = z.permute(0, 2, 3, 1).reshape(z.shape[0], z.shape[2] * z.shape[3], z.shape[1]).float()
    if center == "none":
        return rows
    if center == "sample":
        return rows - rows.mean(dim=1, keepdim=True)
    raise ValueError(f"未知 center 模式：{center}；可选 none/sample")


def direct_error(z_g: torch.Tensor, pz: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((z_g - pz).pow(2).flatten(1).sum(dim=1) / z_g.pow(2).flatten(1).sum(dim=1).clamp_min(1e-12))


def relative_token_error(z_g: torch.Tensor, pz: torch.Tensor, center: str = "none") -> torch.Tensor:
    y = token_rows(z_g, center=center)
    x = token_rows(pz, center=center)
    numerator = (y - x).pow(2).flatten(1).sum(dim=1)
    denominator = y.pow(2).flatten(1).sum(dim=1).clamp_min(1e-12)
    return torch.sqrt(numerator / denominator)


def token_similarity(pz: torch.Tensor, z_g: torch.Tensor, center: str = "sample") -> torch.Tensor:
    x = F.normalize(token_rows(pz, center=center), dim=-1)
    y = F.normalize(token_rows(z_g, center=center), dim=-1)
    return torch.bmm(x, y.transpose(1, 2))


def grid_coords(height: int, width: int, device: torch.device) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=-1)


def correspondence_metrics(z: torch.Tensor, z_g: torch.Tensor, transform: str, center: str = "sample") -> Dict[str, float]:
    pz = P(z, transform)
    sim = token_similarity(pz, z_g, center=center)
    batch, tokens, _ = sim.shape
    side = int(math.sqrt(tokens))
    coords = grid_coords(side, side, sim.device)
    argmax = sim.argmax(dim=-1)
    expected = torch.arange(tokens, device=sim.device).view(1, tokens).expand(batch, tokens)
    displacement = torch.linalg.norm(coords[argmax] - coords[expected], dim=-1)
    return {
        "direct_error": float(relative_token_error(z_g, pz, center=center).mean().detach().cpu()),
        "mean_diag_cosine": float(sim.diagonal(dim1=-2, dim2=-1).mean().detach().cpu()),
        "mean_best_cosine": float(sim.max(dim=-1).values.mean().detach().cpu()),
        "mean_best_match_displacement": float(displacement.mean().detach().cpu()),
        "exact_match_rate": float((argmax == expected).float().mean().detach().cpu()),
        "within_1_token_rate": float((displacement <= 1.0).float().mean().detach().cpu()),
    }


@torch.no_grad()
def diagnostic_table(
    adapter,
    x: torch.Tensor,
    transforms: Sequence[str] = NON_IDENTITY_TRANSFORMS,
    center: str = "sample",
) -> List[Dict[str, float | str]]:
    x = x.to(adapter.device, dtype=torch.float32)
    z = E(adapter, x)
    rows = []
    for transform in transforms:
        z_g = E(adapter, P(x, transform))
        rows.append({"transform": transform, **correspondence_metrics(z, z_g, transform, center=center)})
    return rows


def orthogonal_procrustes_error(z: torch.Tensor, z_g: torch.Tensor, transform: str, center: bool = True) -> Dict[str, float]:
    x = token_rows(P(z, transform), center="none").reshape(-1, z.shape[1])
    y = token_rows(z_g, center="none").reshape(-1, z.shape[1])
    if center:
        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)
    cross = x.T @ y
    u, _, vh = torch.linalg.svd(cross, full_matrices=False)
    q = u @ vh
    pred = x @ q
    err = torch.linalg.norm(y - pred) / torch.linalg.norm(y).clamp_min(1e-12)
    direct = torch.linalg.norm(y - x) / torch.linalg.norm(y).clamp_min(1e-12)
    if direct.item() <= 1e-6:
        relative_gain = torch.zeros((), device=direct.device, dtype=direct.dtype) if err.item() <= 1e-6 else torch.tensor(float("-inf"))
    else:
        relative_gain = 1.0 - err / direct
    return {
        "direct_centered_error": float(direct.detach().cpu()),
        "orthogonal_procrustes_error": float(err.detach().cpu()),
        "relative_gain": float(relative_gain.detach().cpu()),
    }


def _iter_image_batches(x: torch.Tensor, batch_size: int):
    batch_size = max(1, int(batch_size))
    for start in range(0, x.shape[0], batch_size):
        yield x[start:start + batch_size]


def _rows_for_channel_fit(z: torch.Tensor, center: str) -> torch.Tensor:
    return token_rows(z, center=center).reshape(-1, z.shape[1])


def _tokens_to_grid(tokens: torch.Tensor) -> torch.Tensor:
    batch, token_count, channels = tokens.shape
    side = int(math.sqrt(token_count))
    if side * side != token_count:
        raise ValueError(f"token 数不是平方数：{token_count}")
    return tokens.transpose(1, 2).reshape(batch, channels, side, side).contiguous()


def _grid_to_tokens(grid: torch.Tensor) -> torch.Tensor:
    return grid.flatten(2).transpose(1, 2).contiguous()


def _preprocess_rae_encoder_input(adapter, x: torch.Tensor) -> torch.Tensor:
    if not hasattr(adapter, "model") or not hasattr(adapter.model, "encoder_input_size"):
        raise TypeError("ViT layerwise diagnostic 只支持 RAE adapter。")
    model = adapter.model
    x = ((x + 1.0) / 2.0).clamp(0.0, 1.0).to(adapter.device, dtype=torch.float32)
    size = int(model.encoder_input_size)
    if x.shape[-2:] != (size, size):
        x = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False)
    return (x - model.encoder_mean.to(x.device)) / model.encoder_std.to(x.device)


def _adapter_key(adapter) -> str:
    return str(getattr(adapter, "key", ""))


def _dinov2_patch_pos(adapter, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_in = _preprocess_rae_encoder_input(adapter, x)
    inner = adapter.model.encoder.encoder
    embeddings = inner.embeddings
    patch = embeddings.patch_embeddings(
        x_in.to(dtype=embeddings.patch_embeddings.projection.weight.dtype)
    ).float()
    batch, _, height, width = x_in.shape
    cls_tokens = embeddings.cls_token.expand(batch, -1, -1).float()
    pos = embeddings.interpolate_pos_encoding(torch.cat((cls_tokens, patch), dim=1), height, width).float()
    return x_in, patch, pos


def _mae_patch_pos(adapter, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_in = _preprocess_rae_encoder_input(adapter, x)
    inner = adapter.model.encoder.model
    embeddings = inner.embeddings
    patch = embeddings.patch_embeddings(x_in, interpolate_pos_encoding=True).float()
    _, _, height, width = x_in.shape
    pos = embeddings.interpolate_pos_encoding(patch, height, width).float()
    return x_in, patch, pos


def _siglip2_patch_pos(adapter, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_in = _preprocess_rae_encoder_input(adapter, x)
    inner = adapter.model.encoder.model
    embeddings = inner.embeddings
    batch, _, height, width = x_in.shape
    patch_map = embeddings.patch_embedding(
        x_in.to(dtype=embeddings.patch_embedding.weight.dtype)
    ).float()
    patch = patch_map.flatten(2).transpose(1, 2).contiguous()
    pos = embeddings.interpolate_pos_encoding(patch, height, width).float()
    if pos.ndim == 2:
        pos = pos.unsqueeze(0)
    if pos.shape[0] == 1:
        pos = pos.expand(batch, -1, -1)
    return x_in, patch, pos


@torch.no_grad()
def extract_vit_stage_latents(
    adapter,
    x: torch.Tensor,
    hidden_indices: Sequence[int] = (0, 1, 3, 6, 9, 12),
    include_rae_normalized: bool = True,
) -> Dict[str, torch.Tensor]:
    key = _adapter_key(adapter)
    if key == "rae_dinov2":
        x_in, patch, pos = _dinov2_patch_pos(adapter, x)
        inner = adapter.model.encoder.encoder
        out = inner(x_in, output_hidden_states=True)
        hidden_states = out.hidden_states or ()
        post_pos = patch + pos[:, 1:]
        stages = {
            "patch_pre_pos": _tokens_to_grid(patch),
            "pos_only": _tokens_to_grid(pos[:, 1:].expand(patch.shape[0], -1, -1)),
            "post_pos": _tokens_to_grid(post_pos),
            "patch_plus_pos": _tokens_to_grid(post_pos),
            "final_raw": _tokens_to_grid(out.last_hidden_state[:, 5:].float()),
        }
        if include_rae_normalized:
            stages["rae_normalized"] = E(adapter, x)
        for index in hidden_indices:
            if index < len(hidden_states):
                stages[f"hidden_{index}"] = _tokens_to_grid(hidden_states[index][:, 5:].float())
        return stages
    if key == "rae_mae":
        x_in, patch, pos = _mae_patch_pos(adapter, x)
        inner = adapter.model.encoder.model
        noise = torch.arange(patch.shape[1], device=x_in.device, dtype=x_in.dtype).unsqueeze(0).expand(x_in.shape[0], -1)
        out = inner(x_in, noise=noise, interpolate_pos_encoding=True, output_hidden_states=True)
        hidden_states = out.hidden_states or ()
        post_pos = patch + pos[:, 1:]
        stages = {
            "patch_pre_pos": _tokens_to_grid(patch),
            "pos_only": _tokens_to_grid(pos[:, 1:].expand(patch.shape[0], -1, -1)),
            "post_pos": _tokens_to_grid(post_pos),
            "patch_plus_pos": _tokens_to_grid(post_pos),
            "final_raw": _tokens_to_grid(out.last_hidden_state[:, 1:].float()),
        }
        if include_rae_normalized:
            stages["rae_normalized"] = E(adapter, x)
        for index in hidden_indices:
            if index < len(hidden_states):
                stages[f"hidden_{index}"] = _tokens_to_grid(hidden_states[index][:, 1:].float())
        return stages
    if key == "rae_siglip2":
        x_in, patch, pos = _siglip2_patch_pos(adapter, x)
        inner = adapter.model.encoder.model
        out = inner(x_in, output_hidden_states=True, interpolate_pos_encoding=True)
        hidden_states = getattr(out, "hidden_states", None) or ()
        post_pos = patch + pos
        stages = {
            "patch_pre_pos": _tokens_to_grid(patch),
            "pos_only": _tokens_to_grid(pos),
            "post_pos": _tokens_to_grid(post_pos),
            "patch_plus_pos": _tokens_to_grid(post_pos),
            "final_raw": _tokens_to_grid(out.last_hidden_state.float()),
        }
        if include_rae_normalized:
            stages["rae_normalized"] = E(adapter, x)
        for index in hidden_indices:
            if index < len(hidden_states):
                stages[f"hidden_{index}"] = _tokens_to_grid(hidden_states[index].float())
        return stages
    raise ValueError("extract_vit_stage_latents 当前只支持 rae_dinov2、rae_mae 和 rae_siglip2。")


@torch.no_grad()
def vit_position_patch_diagnostic_table(
    adapter,
    x: torch.Tensor,
    transforms: Sequence[str] = ("rot90", "rot180", "flip_h", "flip_v"),
    center: str = "sample",
    hidden_indices: Sequence[int] = (0, 1, 3, 6, 9, 12),
) -> List[Dict[str, float | str]]:
    x = x.to(adapter.device, dtype=torch.float32)
    base_stages = extract_vit_stage_latents(adapter, x, hidden_indices=hidden_indices)
    rows: List[Dict[str, float | str]] = []
    for transform in transforms:
        transformed_stages = extract_vit_stage_latents(adapter, P(x, transform), hidden_indices=hidden_indices)
        for stage, z in base_stages.items():
            err = relative_token_error(transformed_stages[stage], P(z, transform), center=center).mean()
            rows.append(
                {
                    "model": _adapter_key(adapter),
                    "transform": transform,
                    "stage": stage,
                    "error": float(err.detach().cpu()),
                }
            )
        pos = base_stages["pos_only"][:1]
        rows.append(
            {
                "model": _adapter_key(adapter),
                "transform": transform,
                "stage": "pos_symmetry_none",
                "error": float(relative_token_error(pos, P(pos, transform), center="none").mean().detach().cpu()),
            }
        )
    return rows


@torch.no_grad()
def _dinov2_custom_pos_forward(adapter, x: torch.Tensor, pos_mode: str, transform_for_pos: str) -> torch.Tensor:
    x_in, patch, pos = _dinov2_patch_pos(adapter, x)
    inner = adapter.model.encoder.encoder
    embeddings = inner.embeddings
    batch = x_in.shape[0]
    cls_tokens = embeddings.cls_token.expand(batch, -1, -1).float()
    if pos_mode == "zero":
        cls_with_pos = cls_tokens
        patch_with_pos = patch
    elif pos_mode == "normal":
        cls_with_pos = cls_tokens + pos[:, :1]
        patch_with_pos = patch + pos[:, 1:]
    elif pos_mode == "rotated":
        cls_with_pos = cls_tokens + pos[:, :1]
        pos_grid = _tokens_to_grid(pos[:, 1:].expand(batch, -1, -1))
        patch_with_pos = patch + _grid_to_tokens(P(pos_grid, transform_for_pos))
    else:
        raise ValueError(f"未知 pos_mode：{pos_mode}")
    sequence = torch.cat((cls_with_pos, patch_with_pos), dim=1)
    sequence = torch.cat(
        (sequence[:, :1], embeddings.register_tokens.expand(batch, -1, -1).float(), sequence[:, 1:]),
        dim=1,
    )
    out = inner.encoder(sequence).last_hidden_state
    out = inner.layernorm(out)
    return _tokens_to_grid(out[:, 5:].float())


@torch.no_grad()
def _mae_custom_pos_forward(adapter, x: torch.Tensor, pos_mode: str, transform_for_pos: str) -> torch.Tensor:
    x_in, patch, pos = _mae_patch_pos(adapter, x)
    inner = adapter.model.encoder.model
    embeddings = inner.embeddings
    batch = x_in.shape[0]
    if pos_mode == "zero":
        cls_tokens = embeddings.cls_token.expand(batch, -1, -1).float()
        patch_with_pos = patch
    elif pos_mode == "normal":
        cls_tokens = (embeddings.cls_token + pos[:, :1]).expand(batch, -1, -1).float()
        patch_with_pos = patch + pos[:, 1:]
    elif pos_mode == "rotated":
        cls_tokens = (embeddings.cls_token + pos[:, :1]).expand(batch, -1, -1).float()
        pos_grid = _tokens_to_grid(pos[:, 1:].expand(batch, -1, -1))
        patch_with_pos = patch + _grid_to_tokens(P(pos_grid, transform_for_pos))
    else:
        raise ValueError(f"未知 pos_mode：{pos_mode}")
    sequence = torch.cat((cls_tokens, patch_with_pos), dim=1)
    out = inner.encoder(sequence).last_hidden_state
    out = inner.layernorm(out)
    return _tokens_to_grid(out[:, 1:].float())


@torch.no_grad()
def _siglip2_custom_pos_forward(adapter, x: torch.Tensor, pos_mode: str, transform_for_pos: str) -> torch.Tensor:
    x_in, patch, pos = _siglip2_patch_pos(adapter, x)
    inner = adapter.model.encoder.model
    if pos_mode == "zero":
        sequence = patch
    elif pos_mode == "normal":
        sequence = patch + pos
    elif pos_mode == "rotated":
        pos_grid = _tokens_to_grid(pos)
        sequence = patch + _grid_to_tokens(P(pos_grid, transform_for_pos))
    else:
        raise ValueError(f"未知 pos_mode：{pos_mode}")
    out = inner.encoder(inputs_embeds=sequence).last_hidden_state
    out = inner.post_layernorm(out)
    return _tokens_to_grid(out.float())


@torch.no_grad()
def vit_pos_intervention_table(
    adapter,
    x: torch.Tensor,
    transforms: Sequence[str] = ("rot90", "rot180", "flip_h", "flip_v"),
    center: str = "sample",
) -> List[Dict[str, float | str]]:
    key = _adapter_key(adapter)
    if key == "rae_dinov2":
        custom_forward = _dinov2_custom_pos_forward
    elif key == "rae_mae":
        custom_forward = _mae_custom_pos_forward
    elif key == "rae_siglip2":
        custom_forward = _siglip2_custom_pos_forward
    else:
        raise ValueError("vit_pos_intervention_table 当前只支持 rae_dinov2、rae_mae 和 rae_siglip2。")

    x = x.to(adapter.device, dtype=torch.float32)
    rows: List[Dict[str, float | str]] = []
    for transform in transforms:
        normal = custom_forward(adapter, x, "normal", "identity")
        normal_g = custom_forward(adapter, P(x, transform), "normal", "identity")
        rotated_pos_g = custom_forward(adapter, P(x, transform), "rotated", transform)
        zero = custom_forward(adapter, x, "zero", "identity")
        zero_g = custom_forward(adapter, P(x, transform), "zero", "identity")
        for mode, source, target in (
            ("normal_pos", normal, normal_g),
            ("rotated_pos_for_gx", normal, rotated_pos_g),
            ("zero_pos_both", zero, zero_g),
        ):
            err = relative_token_error(target, P(source, transform), center=center).mean()
            rows.append(
                {
                    "model": key,
                    "transform": transform,
                    "mode": mode,
                    "error": float(err.detach().cpu()),
                }
            )
    return rows


@torch.no_grad()
def run_vit_position_embedding_study(
    dataset,
    keys: Sequence[str] = ("rae_dinov2", "rae_mae"),
    device: str | torch.device = "cuda:0",
    rae_repo_path: str | Path = "external/RAE",
    rae_auto_clone: bool = False,
    rae_auto_download: bool = False,
    transforms: Sequence[str] = ("rot90", "rot180", "flip_h", "flip_v"),
    image_size: int = 256,
    count: int = 4,
    seed: int = 0,
    center: str = "sample",
    hidden_indices: Sequence[int] = (0, 1, 3, 6, 9, 12),
) -> Dict[str, object]:
    x, indices = pick_dataset_images(dataset, count=count, seed=seed, image_size=image_size)
    stage_rows: List[Dict[str, float | str]] = []
    intervention_rows: List[Dict[str, float | str]] = []
    for key in keys:
        adapter = load_baseline_adapter(
            key,
            device=device,
            rae_repo_path=rae_repo_path,
            rae_auto_clone=rae_auto_clone,
            rae_auto_download=rae_auto_download,
            posterior="mode",
        )
        stage_rows.extend(
            vit_position_patch_diagnostic_table(
                adapter,
                x,
                transforms=transforms,
                center=center,
                hidden_indices=hidden_indices,
            )
        )
        intervention_rows.extend(
            vit_pos_intervention_table(
                adapter,
                x,
                transforms=transforms,
                center=center,
            )
        )
        del adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        "stage_rows": stage_rows,
        "intervention_rows": intervention_rows,
        "indices": indices,
    }


def _orthogonal_map_from_cross(cross: torch.Tensor) -> torch.Tensor:
    u, _, vh = torch.linalg.svd(cross, full_matrices=False)
    return u @ vh


@torch.no_grad()
def fit_orthogonal_maps(
    adapter,
    train_x: torch.Tensor,
    transforms: Sequence[str] = NON_IDENTITY_TRANSFORMS,
    center: str = "sample",
    batch_size: int = 16,
) -> Dict[str, torch.Tensor]:
    train_x = train_x.to(adapter.device, dtype=torch.float32)
    maps: Dict[str, torch.Tensor] = {}
    cross_by_transform: Dict[str, torch.Tensor] = {}
    channels = None
    for x_batch in _iter_image_batches(train_x, batch_size):
        z = E(adapter, x_batch)
        if channels is None:
            channels = z.shape[1]
            for transform in transforms:
                cross_by_transform[transform] = torch.zeros((channels, channels), device=adapter.device, dtype=torch.float32)
        for transform in transforms:
            z_g = z if transform == "identity" else E(adapter, P(x_batch, transform))
            x_rows = _rows_for_channel_fit(P(z, transform), center=center)
            y_rows = _rows_for_channel_fit(z_g, center=center)
            cross_by_transform[transform].add_(x_rows.T @ y_rows)

    if channels is None:
        raise ValueError("train_x 为空，无法拟合 Procrustes map。")

    identity = torch.eye(channels, device=adapter.device, dtype=torch.float32)
    for transform in transforms:
        maps[transform] = identity if transform == "identity" else _orthogonal_map_from_cross(cross_by_transform[transform])
    return maps


@torch.no_grad()
def evaluate_channel_maps(
    adapter,
    eval_x: torch.Tensor,
    maps: Mapping[str, torch.Tensor],
    transforms: Sequence[str] = NON_IDENTITY_TRANSFORMS,
    center: str = "sample",
    batch_size: int = 16,
) -> List[Dict[str, float | str | None]]:
    eval_x = eval_x.to(adapter.device, dtype=torch.float32)
    sums = {
        transform: {
            "err_p_num": torch.zeros((), device=adapter.device, dtype=torch.float32),
            "err_pc_num": torch.zeros((), device=adapter.device, dtype=torch.float32),
            "den": torch.zeros((), device=adapter.device, dtype=torch.float32),
        }
        for transform in transforms
    }
    for x_batch in _iter_image_batches(eval_x, batch_size):
        z = E(adapter, x_batch)
        for transform in transforms:
            z_g = z if transform == "identity" else E(adapter, P(x_batch, transform))
            x_rows = _rows_for_channel_fit(P(z, transform), center=center)
            y_rows = _rows_for_channel_fit(z_g, center=center)
            channel_map = maps[transform].to(device=adapter.device, dtype=torch.float32)
            pred_rows = x_rows @ channel_map
            sums[transform]["err_p_num"].add_((y_rows - x_rows).pow(2).sum())
            sums[transform]["err_pc_num"].add_((y_rows - pred_rows).pow(2).sum())
            sums[transform]["den"].add_(y_rows.pow(2).sum())

    rows: List[Dict[str, float | str | None]] = []
    for transform in transforms:
        den = sums[transform]["den"].clamp_min(1e-12)
        err_p = torch.sqrt(sums[transform]["err_p_num"] / den)
        err_pc = torch.sqrt(sums[transform]["err_pc_num"] / den)
        ratio = None if err_p.item() <= 1e-8 else err_pc / err_p
        rows.append(
            {
                "transform": transform,
                "err_p": float(err_p.detach().cpu()),
                "err_pc": float(err_pc.detach().cpu()),
                "ratio_pc_over_p": None if ratio is None else float(ratio.detach().cpu()),
                "gain": None if ratio is None else float((1.0 - ratio).detach().cpu()),
            }
        )
    return rows


def frobenius_relative_error(target: torch.Tensor, estimate: torch.Tensor) -> float:
    numerator = torch.linalg.norm(target - estimate)
    denominator = torch.linalg.norm(target).clamp_min(1e-12)
    return float((numerator / denominator).detach().cpu())


def group_law_metrics(maps: Mapping[str, torch.Tensor]) -> Dict[str, float]:
    first_map = next(iter(maps.values()))
    identity = torch.eye(first_map.shape[0], device=first_map.device, dtype=first_map.dtype)
    metrics: Dict[str, float] = {}
    if {"rot90", "rot180"}.issubset(maps):
        metrics["rot180_vs_rot90_rot90"] = frobenius_relative_error(maps["rot180"], maps["rot90"] @ maps["rot90"])
    if {"rot90", "rot180", "rot270"}.issubset(maps):
        metrics["rot270_vs_rot90_rot180"] = frobenius_relative_error(maps["rot270"], maps["rot90"] @ maps["rot180"])
    if "rot90" in maps:
        metrics["rot90_cycle4_vs_identity"] = frobenius_relative_error(
            identity,
            maps["rot90"] @ maps["rot90"] @ maps["rot90"] @ maps["rot90"],
        )
    if "flip_h" in maps:
        metrics["flip_h_square_vs_identity"] = frobenius_relative_error(identity, maps["flip_h"] @ maps["flip_h"])
    if "flip_v" in maps:
        metrics["flip_v_square_vs_identity"] = frobenius_relative_error(identity, maps["flip_v"] @ maps["flip_v"])
    law_values = list(metrics.values())
    metrics["mean_group_law_error"] = float(np.mean(law_values)) if law_values else float("nan")
    return metrics


def _compose_map_for_law(maps: Mapping[str, torch.Tensor], law_name: str) -> Tuple[str, torch.Tensor]:
    if law_name == "rot180_from_rot90_rot90":
        return "rot180", maps["rot90"] @ maps["rot90"]
    if law_name == "rot270_from_rot90_rot180":
        return "rot270", maps["rot90"] @ maps["rot180"]
    if law_name == "rot90_cycle4":
        return "identity", maps["rot90"] @ maps["rot90"] @ maps["rot90"] @ maps["rot90"]
    if law_name == "flip_h_square":
        return "identity", maps["flip_h"] @ maps["flip_h"]
    if law_name == "flip_v_square":
        return "identity", maps["flip_v"] @ maps["flip_v"]
    raise ValueError(f"未知 group law: {law_name}")


def _available_functional_laws(maps: Mapping[str, torch.Tensor], transforms: Sequence[str]) -> List[str]:
    transform_set = set(transforms)
    laws = []
    if {"rot90", "rot180"}.issubset(maps) and "rot180" in transform_set:
        laws.append("rot180_from_rot90_rot90")
    if {"rot90", "rot180", "rot270"}.issubset(maps) and "rot270" in transform_set:
        laws.append("rot270_from_rot90_rot180")
    if "rot90" in maps:
        laws.append("rot90_cycle4")
    if "flip_h" in maps:
        laws.append("flip_h_square")
    if "flip_v" in maps:
        laws.append("flip_v_square")
    return laws


@torch.no_grad()
def _evaluate_channel_maps_with_encoder(
    encode_fn,
    eval_x: torch.Tensor,
    maps: Mapping[str, torch.Tensor],
    transforms: Sequence[str],
    center: str,
    batch_size: int,
    device: torch.device,
) -> List[Dict[str, float | str | None]]:
    eval_x = eval_x.to(device=device, dtype=torch.float32)
    sums = {
        transform: {
            "err_p_num": torch.zeros((), device=device, dtype=torch.float32),
            "err_pc_num": torch.zeros((), device=device, dtype=torch.float32),
            "den": torch.zeros((), device=device, dtype=torch.float32),
        }
        for transform in transforms
    }
    for x_batch in _iter_image_batches(eval_x, batch_size):
        z = encode_fn(x_batch)
        for transform in transforms:
            z_g = z if transform == "identity" else encode_fn(P(x_batch, transform))
            x_rows = _rows_for_channel_fit(P(z, transform), center=center)
            y_rows = _rows_for_channel_fit(z_g, center=center)
            channel_map = maps[transform].to(device=device, dtype=torch.float32)
            pred_rows = x_rows @ channel_map
            sums[transform]["err_p_num"].add_((y_rows - x_rows).pow(2).sum())
            sums[transform]["err_pc_num"].add_((y_rows - pred_rows).pow(2).sum())
            sums[transform]["den"].add_(y_rows.pow(2).sum())

    rows: List[Dict[str, float | str | None]] = []
    for transform in transforms:
        den = sums[transform]["den"].clamp_min(1e-12)
        err_p = torch.sqrt(sums[transform]["err_p_num"] / den)
        err_pc = torch.sqrt(sums[transform]["err_pc_num"] / den)
        ratio = None if err_p.item() <= 1e-8 else err_pc / err_p
        rows.append(
            {
                "transform": transform,
                "err_p": float(err_p.detach().cpu()),
                "err_pc": float(err_pc.detach().cpu()),
                "ratio_pc_over_p": None if ratio is None else float(ratio.detach().cpu()),
                "gain": None if ratio is None else float((1.0 - ratio).detach().cpu()),
            }
        )
    return rows


@torch.no_grad()
def _fit_orthogonal_maps_with_encoder(
    encode_fn,
    train_x: torch.Tensor,
    transforms: Sequence[str],
    center: str,
    batch_size: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    train_x = train_x.to(device=device, dtype=torch.float32)
    maps: Dict[str, torch.Tensor] = {}
    cross_by_transform: Dict[str, torch.Tensor] = {}
    channels = None
    for x_batch in _iter_image_batches(train_x, batch_size):
        z = encode_fn(x_batch)
        if channels is None:
            channels = z.shape[1]
            for transform in transforms:
                cross_by_transform[transform] = torch.zeros((channels, channels), device=device, dtype=torch.float32)
        for transform in transforms:
            z_g = z if transform == "identity" else encode_fn(P(x_batch, transform))
            x_rows = _rows_for_channel_fit(P(z, transform), center=center)
            y_rows = _rows_for_channel_fit(z_g, center=center)
            cross_by_transform[transform].add_(x_rows.T @ y_rows)

    if channels is None:
        raise ValueError("train_x 为空，无法拟合 Procrustes map。")

    identity = torch.eye(channels, device=device, dtype=torch.float32)
    for transform in transforms:
        maps[transform] = identity if transform == "identity" else _orthogonal_map_from_cross(cross_by_transform[transform])
    return maps


def _unique_transforms(transforms: Sequence[str]) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(transforms))


def _matrix_power(matrix: torch.Tensor, power: int) -> torch.Tensor:
    power = int(power)
    if power < 0:
        raise ValueError(f"matrix power 必须非负，收到 {power}")
    identity = torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
    if power == 0:
        return identity
    result = identity
    for _ in range(power):
        result = result @ matrix
    return result


def _identity_like_map(maps: Mapping[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    first_map = next(iter(maps.values()))
    return torch.eye(first_map.shape[0], device=device, dtype=torch.float32)


def _rotation_power_map(generator_maps: Mapping[str, torch.Tensor], transform: str, device: torch.device) -> torch.Tensor:
    if transform not in C4_TRANSFORMS:
        raise ValueError(f"power group diagnostic 只支持 C4 旋转，收到：{transform}")
    if transform == "identity":
        return _identity_like_map(generator_maps, device)
    if "rot90" not in generator_maps:
        raise KeyError("generator_maps 必须包含 rot90 才能计算 C4 power map。")
    return _matrix_power(generator_maps["rot90"].to(device=device, dtype=torch.float32), rotation_power(transform))


def _ratio_or_none(numerator: torch.Tensor, denominator: torch.Tensor) -> Optional[torch.Tensor]:
    return None if denominator.item() <= 1e-8 else numerator / denominator


@torch.no_grad()
def fit_generator_maps(
    adapter,
    train_x: torch.Tensor,
    generator_transforms: Sequence[str] = ("rot90", "flip_h"),
    center: str = "sample",
    batch_size: int = 16,
) -> Dict[str, torch.Tensor]:
    encode_fn = lambda x_batch: E(adapter, x_batch)
    return _fit_generator_maps_with_encoder(
        encode_fn=encode_fn,
        train_x=train_x,
        generator_transforms=generator_transforms,
        center=center,
        batch_size=batch_size,
        device=adapter.device,
    )


@torch.no_grad()
def _fit_generator_maps_with_encoder(
    encode_fn,
    train_x: torch.Tensor,
    generator_transforms: Sequence[str],
    center: str,
    batch_size: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    transforms = _unique_transforms(("identity", *tuple(generator_transforms)))
    return _fit_orthogonal_maps_with_encoder(
        encode_fn,
        train_x,
        transforms=transforms,
        center=center,
        batch_size=batch_size,
        device=device,
    )


@torch.no_grad()
def evaluate_power_group_maps(
    adapter,
    eval_x: torch.Tensor,
    generator_maps: Mapping[str, torch.Tensor],
    independent_maps: Optional[Mapping[str, torch.Tensor]] = None,
    transforms: Sequence[str] = C4_TRANSFORMS,
    center: str = "sample",
    batch_size: int = 16,
) -> List[Dict[str, float | str | int | None]]:
    encode_fn = lambda x_batch: E(adapter, x_batch)
    return _evaluate_power_group_maps_with_encoder(
        encode_fn=encode_fn,
        eval_x=eval_x,
        generator_maps=generator_maps,
        independent_maps=independent_maps,
        transforms=transforms,
        center=center,
        batch_size=batch_size,
        device=adapter.device,
    )


@torch.no_grad()
def _evaluate_power_group_maps_with_encoder(
    encode_fn,
    eval_x: torch.Tensor,
    generator_maps: Mapping[str, torch.Tensor],
    independent_maps: Optional[Mapping[str, torch.Tensor]],
    transforms: Sequence[str],
    center: str,
    batch_size: int,
    device: torch.device,
) -> List[Dict[str, float | str | int | None]]:
    eval_x = eval_x.to(device=device, dtype=torch.float32)
    transforms = _unique_transforms(transforms)
    sums = {
        transform: {
            "err_p_num": torch.zeros((), device=device, dtype=torch.float32),
            "err_ind_num": torch.zeros((), device=device, dtype=torch.float32),
            "err_power_num": torch.zeros((), device=device, dtype=torch.float32),
            "den": torch.zeros((), device=device, dtype=torch.float32),
            "has_ind": False,
        }
        for transform in transforms
    }
    for x_batch in _iter_image_batches(eval_x, batch_size):
        z = encode_fn(x_batch)
        for transform in transforms:
            z_g = z if transform == "identity" else encode_fn(P(x_batch, transform))
            x_rows = _rows_for_channel_fit(P(z, transform), center=center)
            y_rows = _rows_for_channel_fit(z_g, center=center)
            power_map = _rotation_power_map(generator_maps, transform, device=device)
            power_pred = x_rows @ power_map

            sums[transform]["err_p_num"].add_((y_rows - x_rows).pow(2).sum())
            sums[transform]["err_power_num"].add_((y_rows - power_pred).pow(2).sum())
            sums[transform]["den"].add_(y_rows.pow(2).sum())

            if independent_maps is not None and (transform in independent_maps or transform == "identity"):
                if transform == "identity" and transform not in independent_maps:
                    ind_map = torch.eye(x_rows.shape[1], device=device, dtype=torch.float32)
                else:
                    ind_map = independent_maps[transform].to(device=device, dtype=torch.float32)
                ind_pred = x_rows @ ind_map
                sums[transform]["err_ind_num"].add_((y_rows - ind_pred).pow(2).sum())
                sums[transform]["has_ind"] = True

    rows: List[Dict[str, float | str | int | None]] = []
    for transform in transforms:
        den = sums[transform]["den"].clamp_min(1e-12)
        err_p = torch.sqrt(sums[transform]["err_p_num"] / den)
        err_power = torch.sqrt(sums[transform]["err_power_num"] / den)
        err_ind = torch.sqrt(sums[transform]["err_ind_num"] / den) if sums[transform]["has_ind"] else None
        ind_over_p = None if err_ind is None else _ratio_or_none(err_ind, err_p)
        power_over_p = _ratio_or_none(err_power, err_p)
        power_over_ind = None if err_ind is None else _ratio_or_none(err_power, err_ind)
        rows.append(
            {
                "transform": transform,
                "rotation_power": rotation_power(transform),
                "err_p": float(err_p.detach().cpu()),
                "err_ind": None if err_ind is None else float(err_ind.detach().cpu()),
                "err_power": float(err_power.detach().cpu()),
                "ind_over_p": None if ind_over_p is None else float(ind_over_p.detach().cpu()),
                "power_over_p": None if power_over_p is None else float(power_over_p.detach().cpu()),
                "power_over_ind": None if power_over_ind is None else float(power_over_ind.detach().cpu()),
                "ind_gain": None if ind_over_p is None else float((1.0 - ind_over_p).detach().cpu()),
                "power_gain": None if power_over_p is None else float((1.0 - power_over_p).detach().cpu()),
            }
        )
    return rows


def _d4_relation_specs(generator_maps: Mapping[str, torch.Tensor], device: torch.device) -> List[Tuple[str, str, torch.Tensor]]:
    specs: List[Tuple[str, str, torch.Tensor]] = []
    if "rot90" in generator_maps:
        r = generator_maps["rot90"].to(device=device, dtype=torch.float32)
        specs.append(("rot90_cycle4", "identity", _matrix_power(r, 4)))
    if "flip_h" in generator_maps:
        s = generator_maps["flip_h"].to(device=device, dtype=torch.float32)
        specs.append(("flip_h_square", "identity", s @ s))
    if {"rot90", "flip_h"}.issubset(generator_maps):
        r = generator_maps["rot90"].to(device=device, dtype=torch.float32)
        s = generator_maps["flip_h"].to(device=device, dtype=torch.float32)
        specs.append(("flip_h_rot90_flip_h_vs_rot270", "rot270", s @ r @ s))
    return specs


@torch.no_grad()
def evaluate_d4_relation_maps(
    adapter,
    eval_x: torch.Tensor,
    generator_maps: Mapping[str, torch.Tensor],
    independent_maps: Optional[Mapping[str, torch.Tensor]] = None,
    center: str = "sample",
    batch_size: int = 16,
) -> List[Dict[str, float | str | None]]:
    encode_fn = lambda x_batch: E(adapter, x_batch)
    return _evaluate_d4_relation_maps_with_encoder(
        encode_fn=encode_fn,
        eval_x=eval_x,
        generator_maps=generator_maps,
        independent_maps=independent_maps,
        center=center,
        batch_size=batch_size,
        device=adapter.device,
    )


@torch.no_grad()
def _evaluate_d4_relation_maps_with_encoder(
    encode_fn,
    eval_x: torch.Tensor,
    generator_maps: Mapping[str, torch.Tensor],
    independent_maps: Optional[Mapping[str, torch.Tensor]],
    center: str,
    batch_size: int,
    device: torch.device,
) -> List[Dict[str, float | str | None]]:
    eval_x = eval_x.to(device=device, dtype=torch.float32)
    specs = _d4_relation_specs(generator_maps, device=device)
    sums = {
        relation: {
            "err_p_num": torch.zeros((), device=device, dtype=torch.float32),
            "err_ind_num": torch.zeros((), device=device, dtype=torch.float32),
            "err_relation_num": torch.zeros((), device=device, dtype=torch.float32),
            "relation_vs_ind_num": torch.zeros((), device=device, dtype=torch.float32),
            "den": torch.zeros((), device=device, dtype=torch.float32),
            "ind_den": torch.zeros((), device=device, dtype=torch.float32),
            "has_ind": False,
        }
        for relation, _, _ in specs
    }
    for x_batch in _iter_image_batches(eval_x, batch_size):
        z = encode_fn(x_batch)
        for relation, target_transform, relation_map in specs:
            z_target = z if target_transform == "identity" else encode_fn(P(x_batch, target_transform))
            x_rows = _rows_for_channel_fit(P(z, target_transform), center=center)
            y_rows = _rows_for_channel_fit(z_target, center=center)
            relation_pred = x_rows @ relation_map

            sums[relation]["err_p_num"].add_((y_rows - x_rows).pow(2).sum())
            sums[relation]["err_relation_num"].add_((y_rows - relation_pred).pow(2).sum())
            sums[relation]["den"].add_(y_rows.pow(2).sum())

            if independent_maps is not None and (target_transform in independent_maps or target_transform == "identity"):
                if target_transform == "identity" and target_transform not in independent_maps:
                    ind_map = torch.eye(x_rows.shape[1], device=device, dtype=torch.float32)
                else:
                    ind_map = independent_maps[target_transform].to(device=device, dtype=torch.float32)
                ind_pred = x_rows @ ind_map
                sums[relation]["err_ind_num"].add_((y_rows - ind_pred).pow(2).sum())
                sums[relation]["relation_vs_ind_num"].add_((ind_pred - relation_pred).pow(2).sum())
                sums[relation]["ind_den"].add_(ind_pred.pow(2).sum())
                sums[relation]["has_ind"] = True

    rows: List[Dict[str, float | str | None]] = []
    for relation, target_transform, _ in specs:
        den = sums[relation]["den"].clamp_min(1e-12)
        err_p = torch.sqrt(sums[relation]["err_p_num"] / den)
        err_relation = torch.sqrt(sums[relation]["err_relation_num"] / den)
        if sums[relation]["has_ind"]:
            err_ind = torch.sqrt(sums[relation]["err_ind_num"] / den)
            ind_den = sums[relation]["ind_den"].clamp_min(1e-12)
            relation_vs_ind = torch.sqrt(sums[relation]["relation_vs_ind_num"] / ind_den)
        else:
            err_ind = None
            relation_vs_ind = None
        relation_over_p = _ratio_or_none(err_relation, err_p)
        relation_over_ind = None if err_ind is None else _ratio_or_none(err_relation, err_ind)
        rows.append(
            {
                "relation": relation,
                "target_transform": target_transform,
                "err_p": float(err_p.detach().cpu()),
                "err_ind": None if err_ind is None else float(err_ind.detach().cpu()),
                "err_relation": float(err_relation.detach().cpu()),
                "relation_over_p": None if relation_over_p is None else float(relation_over_p.detach().cpu()),
                "relation_over_ind": None if relation_over_ind is None else float(relation_over_ind.detach().cpu()),
                "relation_vs_independent": None if relation_vs_ind is None else float(relation_vs_ind.detach().cpu()),
            }
        )
    return rows


@torch.no_grad()
def rotation_orbit_closure_table(
    adapter,
    eval_x: torch.Tensor,
    generator_maps: Mapping[str, torch.Tensor],
    orbit_transforms: Sequence[str] = C4_TRANSFORMS,
    center: str = "sample",
    batch_size: int = 16,
) -> List[Dict[str, float | str | int]]:
    encode_fn = lambda x_batch: E(adapter, x_batch)
    return _rotation_orbit_closure_table_with_encoder(
        encode_fn=encode_fn,
        eval_x=eval_x,
        generator_maps=generator_maps,
        orbit_transforms=orbit_transforms,
        center=center,
        batch_size=batch_size,
        device=adapter.device,
    )


@torch.no_grad()
def _rotation_orbit_closure_table_with_encoder(
    encode_fn,
    eval_x: torch.Tensor,
    generator_maps: Mapping[str, torch.Tensor],
    orbit_transforms: Sequence[str],
    center: str,
    batch_size: int,
    device: torch.device,
) -> List[Dict[str, float | str | int]]:
    if "rot90" not in generator_maps:
        raise KeyError("generator_maps 必须包含 rot90 才能计算 C4 orbit closure。")
    eval_x = eval_x.to(device=device, dtype=torch.float32)
    orbit_transforms = _unique_transforms(orbit_transforms)
    sums: Dict[Tuple[str, str], Dict[str, torch.Tensor]] = {}
    for source in orbit_transforms:
        rotation_power(source)
        for target in orbit_transforms:
            rotation_power(target)
            sums[(source, target)] = {
                "num": torch.zeros((), device=device, dtype=torch.float32),
                "den": torch.zeros((), device=device, dtype=torch.float32),
            }

    for x_batch in _iter_image_batches(eval_x, batch_size):
        z_by_transform = {
            transform: (encode_fn(x_batch) if transform == "identity" else encode_fn(P(x_batch, transform)))
            for transform in orbit_transforms
        }
        for source in orbit_transforms:
            z_source = z_by_transform[source]
            for target in orbit_transforms:
                relative = relative_c4_transform(target, source)
                z_target = z_by_transform[target]
                x_rows = _rows_for_channel_fit(P(z_source, relative), center=center)
                y_rows = _rows_for_channel_fit(z_target, center=center)
                power_map = _rotation_power_map(generator_maps, relative, device=device)
                pred_rows = x_rows @ power_map
                sums[(source, target)]["num"].add_((y_rows - pred_rows).pow(2).sum())
                sums[(source, target)]["den"].add_(y_rows.pow(2).sum())

    rows: List[Dict[str, float | str | int]] = []
    for source in orbit_transforms:
        for target in orbit_transforms:
            relative = relative_c4_transform(target, source)
            den = sums[(source, target)]["den"].clamp_min(1e-12)
            err = torch.sqrt(sums[(source, target)]["num"] / den)
            rows.append(
                {
                    "source_transform": source,
                    "target_transform": target,
                    "relative_transform": relative,
                    "relative_power": rotation_power(relative),
                    "closure_error": float(err.detach().cpu()),
                }
            )
    return rows


DEFAULT_GEOMETRY_STAGES: Tuple[str, ...] = (
    "patch_pre_pos",
    "post_pos",
    "hidden_1",
    "hidden_3",
    "hidden_6",
    "hidden_9",
    "hidden_12",
    "final_raw",
    "rae_normalized",
)


def _resolve_stage_names(
    adapter,
    x: torch.Tensor,
    stage_names: Sequence[str],
    hidden_indices: Sequence[int],
) -> Tuple[str, ...]:
    stages = extract_vit_stage_latents(adapter, x[:1].to(adapter.device, dtype=torch.float32), hidden_indices=hidden_indices)
    resolved = tuple(stage for stage in stage_names if stage in stages)
    missing = [stage for stage in stage_names if stage not in stages]
    if missing:
        print(f"跳过当前 encoder 不存在的 stage: {missing}")
    if not resolved:
        raise ValueError("没有可用的 layer stage。")
    return resolved


@torch.no_grad()
def _fit_stage_orthogonal_maps(
    adapter,
    train_x: torch.Tensor,
    stage_names: Sequence[str],
    transforms: Sequence[str],
    center: str,
    batch_size: int,
    hidden_indices: Sequence[int],
) -> Dict[str, Dict[str, torch.Tensor]]:
    train_x = train_x.to(adapter.device, dtype=torch.float32)
    transforms = _unique_transforms(transforms)
    cross: Dict[str, Dict[str, torch.Tensor]] = {}
    channels_by_stage: Dict[str, int] = {}
    for x_batch in _iter_image_batches(train_x, batch_size):
        base_stages = extract_vit_stage_latents(adapter, x_batch, hidden_indices=hidden_indices)
        if not cross:
            for stage in stage_names:
                channels = base_stages[stage].shape[1]
                channels_by_stage[stage] = channels
                cross[stage] = {
                    transform: torch.zeros((channels, channels), device=adapter.device, dtype=torch.float32)
                    for transform in transforms
                }
        transformed_cache = {"identity": base_stages}
        for transform in transforms:
            transformed_stages = transformed_cache.get(transform)
            if transformed_stages is None:
                transformed_stages = extract_vit_stage_latents(adapter, P(x_batch, transform), hidden_indices=hidden_indices)
                transformed_cache[transform] = transformed_stages
            for stage in stage_names:
                x_rows = _rows_for_channel_fit(P(base_stages[stage], transform), center=center)
                y_rows = _rows_for_channel_fit(transformed_stages[stage], center=center)
                cross[stage][transform].add_(x_rows.T @ y_rows)

    if not cross:
        raise ValueError("train_x 为空，无法拟合 layerwise Procrustes map。")

    maps: Dict[str, Dict[str, torch.Tensor]] = {}
    for stage in stage_names:
        identity = torch.eye(channels_by_stage[stage], device=adapter.device, dtype=torch.float32)
        maps[stage] = {
            transform: identity if transform == "identity" else _orthogonal_map_from_cross(cross[stage][transform])
            for transform in transforms
        }
    return maps


def _stage_generator_maps(
    maps_by_stage: Mapping[str, Mapping[str, torch.Tensor]],
    generator_transforms: Sequence[str],
) -> Dict[str, Dict[str, torch.Tensor]]:
    wanted = _unique_transforms(("identity", *tuple(generator_transforms)))
    return {
        stage: {transform: stage_maps[transform] for transform in wanted if transform in stage_maps}
        for stage, stage_maps in maps_by_stage.items()
    }


@torch.no_grad()
def _evaluate_stage_channel_maps(
    adapter,
    eval_x: torch.Tensor,
    maps_by_stage: Mapping[str, Mapping[str, torch.Tensor]],
    stage_names: Sequence[str],
    transforms: Sequence[str],
    center: str,
    batch_size: int,
    hidden_indices: Sequence[int],
) -> List[Dict[str, float | str | None]]:
    eval_x = eval_x.to(adapter.device, dtype=torch.float32)
    transforms = _unique_transforms(transforms)
    sums = {
        stage: {
            transform: {
                "err_p_num": torch.zeros((), device=adapter.device, dtype=torch.float32),
                "err_pc_num": torch.zeros((), device=adapter.device, dtype=torch.float32),
                "den": torch.zeros((), device=adapter.device, dtype=torch.float32),
            }
            for transform in transforms
        }
        for stage in stage_names
    }
    for x_batch in _iter_image_batches(eval_x, batch_size):
        base_stages = extract_vit_stage_latents(adapter, x_batch, hidden_indices=hidden_indices)
        transformed_cache = {"identity": base_stages}
        for transform in transforms:
            transformed_stages = transformed_cache.get(transform)
            if transformed_stages is None:
                transformed_stages = extract_vit_stage_latents(adapter, P(x_batch, transform), hidden_indices=hidden_indices)
                transformed_cache[transform] = transformed_stages
            for stage in stage_names:
                x_rows = _rows_for_channel_fit(P(base_stages[stage], transform), center=center)
                y_rows = _rows_for_channel_fit(transformed_stages[stage], center=center)
                channel_map = maps_by_stage[stage][transform].to(device=adapter.device, dtype=torch.float32)
                pred_rows = x_rows @ channel_map
                sums[stage][transform]["err_p_num"].add_((y_rows - x_rows).pow(2).sum())
                sums[stage][transform]["err_pc_num"].add_((y_rows - pred_rows).pow(2).sum())
                sums[stage][transform]["den"].add_(y_rows.pow(2).sum())

    rows: List[Dict[str, float | str | None]] = []
    for stage in stage_names:
        for transform in transforms:
            den = sums[stage][transform]["den"].clamp_min(1e-12)
            err_p = torch.sqrt(sums[stage][transform]["err_p_num"] / den)
            err_pc = torch.sqrt(sums[stage][transform]["err_pc_num"] / den)
            ratio = None if err_p.item() <= 1e-8 else err_pc / err_p
            rows.append(
                {
                    "stage": stage,
                    "transform": transform,
                    "err_p": float(err_p.detach().cpu()),
                    "err_pc": float(err_pc.detach().cpu()),
                    "ratio_pc_over_p": None if ratio is None else float(ratio.detach().cpu()),
                    "gain": None if ratio is None else float((1.0 - ratio).detach().cpu()),
                }
            )
    return rows


@torch.no_grad()
def _evaluate_stage_power_maps(
    adapter,
    eval_x: torch.Tensor,
    generator_maps_by_stage: Mapping[str, Mapping[str, torch.Tensor]],
    independent_maps_by_stage: Mapping[str, Mapping[str, torch.Tensor]],
    stage_names: Sequence[str],
    transforms: Sequence[str],
    center: str,
    batch_size: int,
    hidden_indices: Sequence[int],
) -> List[Dict[str, float | str | int | None]]:
    eval_x = eval_x.to(adapter.device, dtype=torch.float32)
    transforms = _unique_transforms(transforms)
    sums = {
        stage: {
            transform: {
                "err_p_num": torch.zeros((), device=adapter.device, dtype=torch.float32),
                "err_ind_num": torch.zeros((), device=adapter.device, dtype=torch.float32),
                "err_power_num": torch.zeros((), device=adapter.device, dtype=torch.float32),
                "den": torch.zeros((), device=adapter.device, dtype=torch.float32),
                "has_ind": False,
            }
            for transform in transforms
        }
        for stage in stage_names
    }
    for x_batch in _iter_image_batches(eval_x, batch_size):
        base_stages = extract_vit_stage_latents(adapter, x_batch, hidden_indices=hidden_indices)
        transformed_cache = {"identity": base_stages}
        for transform in transforms:
            transformed_stages = transformed_cache.get(transform)
            if transformed_stages is None:
                transformed_stages = extract_vit_stage_latents(adapter, P(x_batch, transform), hidden_indices=hidden_indices)
                transformed_cache[transform] = transformed_stages
            for stage in stage_names:
                x_rows = _rows_for_channel_fit(P(base_stages[stage], transform), center=center)
                y_rows = _rows_for_channel_fit(transformed_stages[stage], center=center)
                power_map = _rotation_power_map(generator_maps_by_stage[stage], transform, device=adapter.device)
                power_pred = x_rows @ power_map
                sums[stage][transform]["err_p_num"].add_((y_rows - x_rows).pow(2).sum())
                sums[stage][transform]["err_power_num"].add_((y_rows - power_pred).pow(2).sum())
                sums[stage][transform]["den"].add_(y_rows.pow(2).sum())
                if transform in independent_maps_by_stage[stage] or transform == "identity":
                    if transform == "identity" and transform not in independent_maps_by_stage[stage]:
                        ind_map = torch.eye(x_rows.shape[1], device=adapter.device, dtype=torch.float32)
                    else:
                        ind_map = independent_maps_by_stage[stage][transform].to(device=adapter.device, dtype=torch.float32)
                    ind_pred = x_rows @ ind_map
                    sums[stage][transform]["err_ind_num"].add_((y_rows - ind_pred).pow(2).sum())
                    sums[stage][transform]["has_ind"] = True

    rows: List[Dict[str, float | str | int | None]] = []
    for stage in stage_names:
        for transform in transforms:
            den = sums[stage][transform]["den"].clamp_min(1e-12)
            err_p = torch.sqrt(sums[stage][transform]["err_p_num"] / den)
            err_power = torch.sqrt(sums[stage][transform]["err_power_num"] / den)
            err_ind = torch.sqrt(sums[stage][transform]["err_ind_num"] / den) if sums[stage][transform]["has_ind"] else None
            ind_over_p = None if err_ind is None else _ratio_or_none(err_ind, err_p)
            power_over_p = _ratio_or_none(err_power, err_p)
            power_over_ind = None if err_ind is None else _ratio_or_none(err_power, err_ind)
            rows.append(
                {
                    "stage": stage,
                    "transform": transform,
                    "rotation_power": rotation_power(transform),
                    "err_p": float(err_p.detach().cpu()),
                    "err_ind": None if err_ind is None else float(err_ind.detach().cpu()),
                    "err_power": float(err_power.detach().cpu()),
                    "ind_over_p": None if ind_over_p is None else float(ind_over_p.detach().cpu()),
                    "power_over_p": None if power_over_p is None else float(power_over_p.detach().cpu()),
                    "power_over_ind": None if power_over_ind is None else float(power_over_ind.detach().cpu()),
                    "ind_gain": None if ind_over_p is None else float((1.0 - ind_over_p).detach().cpu()),
                    "power_gain": None if power_over_p is None else float((1.0 - power_over_p).detach().cpu()),
                }
            )
    return rows


@torch.no_grad()
def run_layerwise_geometry_study(
    dataset,
    keys: Sequence[str] = ("rae_dinov2", "rae_mae"),
    device: str | torch.device = "cuda:0",
    rae_repo_path: str | Path = "external/RAE",
    rae_auto_clone: bool = False,
    rae_auto_download: bool = False,
    posterior: str = "mode",
    stage_names: Sequence[str] = DEFAULT_GEOMETRY_STAGES,
    hidden_indices: Sequence[int] = (0, 1, 3, 6, 9, 12),
    analysis_transforms: Sequence[str] = ("rot90", "flip_h"),
    independent_transforms: Sequence[str] = ("identity", "rot90", "rot180", "rot270", "flip_h"),
    generator_transforms: Sequence[str] = ("rot90", "flip_h"),
    power_transforms: Sequence[str] = C4_TRANSFORMS,
    centers: Sequence[str] = ("sample",),
    image_size: int = 256,
    train_count: int = 128,
    test_count: int = 64,
    seed: int = 0,
    batch_size: int = 16,
) -> Dict[str, object]:
    split_map = split_train_val_test_indices(len(dataset), train_count, 0, test_count, seed)
    train_x, _ = pick_dataset_images(dataset, indices=split_map["train"], image_size=image_size)
    test_x, _ = pick_dataset_images(dataset, indices=split_map["test"], image_size=image_size)
    split_tensors = {"train": train_x, "test": test_x}
    layer_direct_rows: List[Dict[str, float | str | None]] = []
    layer_procrustes_rows: List[Dict[str, float | str | None]] = []
    layer_power_rows: List[Dict[str, float | str | int | None]] = []

    for key in keys:
        adapter = load_baseline_adapter(
            key,
            device=device,
            rae_repo_path=rae_repo_path,
            rae_auto_clone=rae_auto_clone,
            rae_auto_download=rae_auto_download,
            posterior=posterior,
        )
        resolved_stages = _resolve_stage_names(adapter, train_x, stage_names, hidden_indices)
        for center in centers:
            independent_maps = _fit_stage_orthogonal_maps(
                adapter,
                train_x,
                stage_names=resolved_stages,
                transforms=independent_transforms,
                center=center,
                batch_size=batch_size,
                hidden_indices=hidden_indices,
            )
            generator_maps = _stage_generator_maps(independent_maps, generator_transforms)
            for split, x_split in split_tensors.items():
                procrustes_rows = _evaluate_stage_channel_maps(
                    adapter,
                    x_split,
                    maps_by_stage=independent_maps,
                    stage_names=resolved_stages,
                    transforms=analysis_transforms,
                    center=center,
                    batch_size=batch_size,
                    hidden_indices=hidden_indices,
                )
                for row in procrustes_rows:
                    common = {"model": key, "center": center, "split": split, **row}
                    layer_procrustes_rows.append(common)
                    layer_direct_rows.append(
                        {
                            "model": key,
                            "center": center,
                            "split": split,
                            "stage": row["stage"],
                            "transform": row["transform"],
                            "err_p": row["err_p"],
                        }
                    )
                for row in _evaluate_stage_power_maps(
                    adapter,
                    x_split,
                    generator_maps_by_stage=generator_maps,
                    independent_maps_by_stage=independent_maps,
                    stage_names=resolved_stages,
                    transforms=power_transforms,
                    center=center,
                    batch_size=batch_size,
                    hidden_indices=hidden_indices,
                ):
                    layer_power_rows.append({"model": key, "center": center, "split": split, **row})
        del adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "layer_direct_rows": layer_direct_rows,
        "layer_procrustes_rows": layer_procrustes_rows,
        "layer_power_rows": layer_power_rows,
        "split_indices": split_map,
    }


def _final_subspace_encode_fn(adapter, subspace: str):
    def encode_fn(x_batch: torch.Tensor) -> torch.Tensor:
        z = E(adapter, x_batch)
        mean = z.mean(dim=(-2, -1), keepdim=True)
        if subspace == "token_mean":
            return mean
        if subspace == "spatial_residual":
            return z - mean
        if subspace == "full":
            return z
        raise ValueError(f"未知 subspace：{subspace}")

    return encode_fn


@torch.no_grad()
def run_mean_residual_geometry_study(
    dataset,
    keys: Sequence[str] = ("rae_dinov2", "rae_mae"),
    device: str | torch.device = "cuda:0",
    rae_repo_path: str | Path = "external/RAE",
    rae_auto_clone: bool = False,
    rae_auto_download: bool = False,
    posterior: str = "mode",
    subspaces: Sequence[str] = ("token_mean", "spatial_residual"),
    analysis_transforms: Sequence[str] = ("rot90", "flip_h"),
    independent_transforms: Sequence[str] = ("identity", "rot90", "rot180", "rot270", "flip_h"),
    generator_transforms: Sequence[str] = ("rot90", "flip_h"),
    power_transforms: Sequence[str] = C4_TRANSFORMS,
    centers: Sequence[str] = ("none",),
    image_size: int = 256,
    train_count: int = 128,
    test_count: int = 64,
    seed: int = 0,
    batch_size: int = 16,
) -> Dict[str, object]:
    split_map = split_train_val_test_indices(len(dataset), train_count, 0, test_count, seed)
    train_x, _ = pick_dataset_images(dataset, indices=split_map["train"], image_size=image_size)
    test_x, _ = pick_dataset_images(dataset, indices=split_map["test"], image_size=image_size)
    split_tensors = {"train": train_x, "test": test_x}
    rows: List[Dict[str, float | str | int | None]] = []

    for key in keys:
        adapter = load_baseline_adapter(
            key,
            device=device,
            rae_repo_path=rae_repo_path,
            rae_auto_clone=rae_auto_clone,
            rae_auto_download=rae_auto_download,
            posterior=posterior,
        )
        for subspace in subspaces:
            encode_fn = _final_subspace_encode_fn(adapter, subspace)
            for center in centers:
                independent_maps = _fit_orthogonal_maps_with_encoder(
                    encode_fn,
                    train_x,
                    transforms=independent_transforms,
                    center=center,
                    batch_size=batch_size,
                    device=adapter.device,
                )
                generator_maps = {
                    transform: independent_maps[transform]
                    for transform in _unique_transforms(("identity", *tuple(generator_transforms)))
                    if transform in independent_maps
                }
                for split, x_split in split_tensors.items():
                    for row in _evaluate_channel_maps_with_encoder(
                        encode_fn,
                        x_split,
                        maps=independent_maps,
                        transforms=analysis_transforms,
                        center=center,
                        batch_size=batch_size,
                        device=adapter.device,
                    ):
                        rows.append(
                            {
                                "model": key,
                                "subspace": subspace,
                                "center": center,
                                "split": split,
                                "diagnostic": "linear_alignability",
                                **row,
                            }
                        )
                    for row in _evaluate_power_group_maps_with_encoder(
                        encode_fn,
                        x_split,
                        generator_maps=generator_maps,
                        independent_maps=independent_maps,
                        transforms=power_transforms,
                        center=center,
                        batch_size=batch_size,
                        device=adapter.device,
                    ):
                        rows.append(
                            {
                                "model": key,
                                "subspace": subspace,
                                "center": center,
                                "split": split,
                                "diagnostic": "group_consistency",
                                **row,
                            }
                        )
        del adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {"mean_residual_rows": rows, "split_indices": split_map}


@torch.no_grad()
def functional_group_law_metrics(
    adapter,
    x: torch.Tensor,
    maps: Mapping[str, torch.Tensor],
    transforms: Sequence[str] = NON_IDENTITY_TRANSFORMS,
    center: str = "sample",
    batch_size: int = 16,
) -> List[Dict[str, float | str | None]]:
    encode_fn = lambda x_batch: E(adapter, x_batch)
    return _functional_group_law_metrics_with_encoder(
        encode_fn=encode_fn,
        x=x,
        maps=maps,
        transforms=transforms,
        center=center,
        batch_size=batch_size,
        device=adapter.device,
    )


@torch.no_grad()
def _functional_group_law_metrics_with_encoder(
    encode_fn,
    x: torch.Tensor,
    maps: Mapping[str, torch.Tensor],
    transforms: Sequence[str],
    center: str,
    batch_size: int,
    device: torch.device,
) -> List[Dict[str, float | str | None]]:
    x = x.to(device=device, dtype=torch.float32)
    laws = _available_functional_laws(maps, transforms)
    sums = {
        law: {
            "direct_num": torch.zeros((), device=device, dtype=torch.float32),
            "composed_num": torch.zeros((), device=device, dtype=torch.float32),
            "direct_vs_composed_num": torch.zeros((), device=device, dtype=torch.float32),
            "den": torch.zeros((), device=device, dtype=torch.float32),
            "direct_den": torch.zeros((), device=device, dtype=torch.float32),
        }
        for law in laws
    }
    for x_batch in _iter_image_batches(x, batch_size):
        z = encode_fn(x_batch)
        for law in laws:
            target_transform, composed_map = _compose_map_for_law(maps, law)
            z_target = z if target_transform == "identity" else encode_fn(P(x_batch, target_transform))
            x_rows = _rows_for_channel_fit(P(z, target_transform), center=center)
            y_rows = _rows_for_channel_fit(z_target, center=center)
            if target_transform == "identity":
                direct_map = torch.eye(x_rows.shape[1], device=device, dtype=torch.float32)
            else:
                direct_map = maps[target_transform].to(device=device, dtype=torch.float32)
            composed_map = composed_map.to(device=device, dtype=torch.float32)
            direct_pred = x_rows @ direct_map
            composed_pred = x_rows @ composed_map
            sums[law]["direct_num"].add_((y_rows - direct_pred).pow(2).sum())
            sums[law]["composed_num"].add_((y_rows - composed_pred).pow(2).sum())
            sums[law]["direct_vs_composed_num"].add_((direct_pred - composed_pred).pow(2).sum())
            sums[law]["den"].add_(y_rows.pow(2).sum())
            sums[law]["direct_den"].add_(direct_pred.pow(2).sum())

    rows: List[Dict[str, float | str | None]] = []
    for law in laws:
        target_transform, _ = _compose_map_for_law(maps, law)
        den = sums[law]["den"].clamp_min(1e-12)
        direct_den = sums[law]["direct_den"].clamp_min(1e-12)
        direct_err = torch.sqrt(sums[law]["direct_num"] / den)
        composed_err = torch.sqrt(sums[law]["composed_num"] / den)
        direct_vs_composed = torch.sqrt(sums[law]["direct_vs_composed_num"] / direct_den)
        ratio = None if direct_err.item() <= 1e-8 else composed_err / direct_err
        rows.append(
            {
                "law": law,
                "target_transform": target_transform,
                "direct_map_err": float(direct_err.detach().cpu()),
                "composed_map_err": float(composed_err.detach().cpu()),
                "composed_over_direct": None if ratio is None else float(ratio.detach().cpu()),
                "functional_law_error": float(direct_vs_composed.detach().cpu()),
            }
        )
    return rows


@torch.no_grad()
def fit_pca_basis(
    adapter,
    x: torch.Tensor,
    max_components: int = 256,
    center: str = "sample",
    batch_size: int = 16,
) -> Dict[str, torch.Tensor | str | int]:
    x = x.to(adapter.device, dtype=torch.float32)
    total_rows = 0
    row_sum = None
    row_cross = None
    for x_batch in _iter_image_batches(x, batch_size):
        z = E(adapter, x_batch)
        rows = _rows_for_channel_fit(z, center=center)
        if row_sum is None:
            channels = rows.shape[1]
            row_sum = torch.zeros(channels, device=adapter.device, dtype=torch.float32)
            row_cross = torch.zeros((channels, channels), device=adapter.device, dtype=torch.float32)
        row_sum.add_(rows.sum(dim=0))
        row_cross.add_(rows.T @ rows)
        total_rows += rows.shape[0]

    if row_sum is None or row_cross is None or total_rows == 0:
        raise ValueError("x 为空，无法拟合 PCA basis。")

    mean = row_sum / float(total_rows)
    covariance = row_cross / float(total_rows) - torch.outer(mean, mean)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0.0)
    eigenvectors = eigenvectors[:, order]
    keep = min(int(max_components), eigenvectors.shape[1])
    total_variance = eigenvalues.sum().clamp_min(1e-12)
    return {
        "center": center,
        "mean": mean.detach(),
        "components": eigenvectors[:, :keep].T.contiguous().detach(),
        "explained_variance": eigenvalues[:keep].detach(),
        "explained_variance_ratio": (eigenvalues[:keep] / total_variance).detach(),
        "all_eigenvalues": eigenvalues.detach(),
        "total_variance": total_variance.detach(),
        "num_rows": total_rows,
    }


def project_latent_to_pca(z: torch.Tensor, basis: Mapping[str, torch.Tensor | str | int], k: int) -> torch.Tensor:
    rows = token_rows(z, center=str(basis.get("center", "sample")))
    mean = basis["mean"].to(device=rows.device, dtype=rows.dtype)
    components = basis["components"].to(device=rows.device, dtype=rows.dtype)
    actual_k = min(int(k), components.shape[0])
    projected = (rows - mean.view(1, 1, -1)) @ components[:actual_k].T
    return projected.permute(0, 2, 1).reshape(z.shape[0], actual_k, z.shape[2], z.shape[3]).contiguous()


def _pca_curve_rows(
    model_name: str,
    center: str,
    basis: Mapping[str, torch.Tensor | str | int],
    component_counts: Sequence[int],
) -> List[Dict[str, float | int | str]]:
    ratios = basis["explained_variance_ratio"].detach().cpu()
    rows = []
    for requested_k in component_counts:
        actual_k = min(int(requested_k), int(ratios.shape[0]))
        rows.append(
            {
                "model": model_name,
                "center": center,
                "requested_k": int(requested_k),
                "actual_k": actual_k,
                "explained_variance": float(ratios[:actual_k].sum()),
            }
        )
    return rows


@torch.no_grad()
def run_pca_subspace_diagnostics(
    adapter,
    train_x: torch.Tensor,
    test_x: torch.Tensor,
    transforms: Sequence[str] = ("rot90", "rot180", "rot270", "flip_h"),
    center: str = "sample",
    component_counts: Sequence[int] = (3, 16, 64, 128, 256),
    max_components: int = 256,
    batch_size: int = 16,
    model_name: str = "model",
) -> Tuple[List[Dict[str, float | str | int | None]], List[Dict[str, float | str | int | None]], List[Dict[str, float | str | int]]]:
    basis = fit_pca_basis(adapter, train_x, max_components=max_components, center=center, batch_size=batch_size)
    pca_rows: List[Dict[str, float | str | int | None]] = []
    functional_rows: List[Dict[str, float | str | int | None]] = []
    curve_rows = _pca_curve_rows(model_name, center, basis, component_counts)

    for requested_k in component_counts:
        actual_k = min(int(requested_k), int(basis["components"].shape[0]))

        def encode_projected(x_batch, k=actual_k):
            return project_latent_to_pca(E(adapter, x_batch), basis, k)

        maps = _fit_orthogonal_maps_with_encoder(
            encode_projected,
            train_x,
            transforms=transforms,
            center="none",
            batch_size=batch_size,
            device=adapter.device,
        )
        for split, x_split in (("train", train_x), ("test", test_x)):
            for row in _evaluate_channel_maps_with_encoder(
                encode_projected,
                x_split,
                maps,
                transforms=transforms,
                center="none",
                batch_size=batch_size,
                device=adapter.device,
            ):
                pca_rows.append(
                    {
                        "model": model_name,
                        "center": center,
                        "requested_k": int(requested_k),
                        "actual_k": actual_k,
                        "split": split,
                        **row,
                    }
                )
        for row in _functional_group_law_metrics_with_encoder(
            encode_projected,
            test_x,
            maps,
            transforms=transforms,
            center="none",
            batch_size=batch_size,
            device=adapter.device,
        ):
            functional_rows.append(
                {
                    "model": model_name,
                    "center": center,
                    "requested_k": int(requested_k),
                    "actual_k": actual_k,
                    "split": "test",
                    **row,
                }
            )
    return pca_rows, functional_rows, curve_rows


def _spectrum_summary(eigenvalues: torch.Tensor, prefix: str, component_counts: Sequence[int]) -> Dict[str, float]:
    values = eigenvalues.detach().float().cpu().clamp_min(0.0)
    total = values.sum().clamp_min(1e-12)
    probs = values / total
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum()
    summary = {
        f"{prefix}_effective_rank": float(torch.exp(entropy)),
        f"{prefix}_participation_ratio": float(total.pow(2) / values.pow(2).sum().clamp_min(1e-12)),
        f"{prefix}_top1_ratio": float(probs[0]) if probs.numel() else float("nan"),
    }
    for k in component_counts:
        actual_k = min(int(k), int(probs.shape[0]))
        summary[f"{prefix}_ev_{int(k)}"] = float(probs[:actual_k].sum()) if actual_k > 0 else float("nan")
    return summary


def _covariance_eigenvalues_from_rows(rows: torch.Tensor) -> torch.Tensor:
    rows = rows.float()
    rows = rows - rows.mean(dim=0, keepdim=True)
    covariance = rows.T @ rows / max(1, rows.shape[0])
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = torch.linalg.eigvalsh(covariance).flip(0).clamp_min(0.0)
    return eigenvalues


@torch.no_grad()
def latent_distribution_metrics(
    adapter,
    x: torch.Tensor,
    center: str = "sample",
    component_counts: Sequence[int] = (3, 16, 64, 128, 256),
    batch_size: int = 16,
    nn_transform: str = "flip_h",
) -> Dict[str, float | str | int]:
    x = x.to(adapter.device, dtype=torch.float32)
    latents = []
    for x_batch in _iter_image_batches(x, batch_size):
        latents.append(E(adapter, x_batch).detach())
    z = torch.cat(latents, dim=0)
    channel_rows = _rows_for_channel_fit(z, center=center)
    channel_eigs = _covariance_eigenvalues_from_rows(channel_rows)

    token_matrix = token_rows(z, center="none").permute(0, 2, 1).reshape(-1, z.shape[2] * z.shape[3])
    token_eigs = _covariance_eigenvalues_from_rows(token_matrix)

    z_aug = align_orbit_latent(E(adapter, P(x, nn_transform)), nn_transform)
    aug_error = float(relative_token_error(z, z_aug, center=center).mean().detach().cpu())
    raw_embed = token_rows(z, center="none").mean(dim=1)
    aug_embed = token_rows(z_aug, center="none").mean(dim=1)
    if raw_embed.shape[0] > 1:
        raw_dist = torch.cdist(raw_embed.float(), raw_embed.float())
        aug_dist = torch.cdist(aug_embed.float(), aug_embed.float())
        raw_dist.fill_diagonal_(float("inf"))
        aug_dist.fill_diagonal_(float("inf"))
        nn_preservation = float((raw_dist.argmin(dim=1) == aug_dist.argmin(dim=1)).float().mean().detach().cpu())
    else:
        nn_preservation = float("nan")

    summary: Dict[str, float | str | int] = {
        "center": center,
        "num_images": int(z.shape[0]),
        "channels": int(z.shape[1]),
        "height": int(z.shape[2]),
        "width": int(z.shape[3]),
        "nn_transform": nn_transform,
        "augmentation_relative_error": aug_error,
        "nn_preservation_rate": nn_preservation,
    }
    summary.update(_spectrum_summary(channel_eigs, "channel", component_counts))
    summary.update(_spectrum_summary(token_eigs, "token", component_counts))
    return summary


@torch.no_grad()
def run_mae_dinov2_mechanism_study(
    dataset,
    keys: Sequence[str] = ("rae_dinov2", "rae_mae"),
    device: str | torch.device = "cuda:0",
    rae_repo_path: str | Path = "external/RAE",
    rae_auto_clone: bool = False,
    rae_auto_download: bool = False,
    transforms: Sequence[str] = ("rot90", "rot180", "rot270", "flip_h"),
    centers: Sequence[str] = ("sample",),
    pca_component_counts: Sequence[int] = (3, 16, 64, 128, 256),
    image_size: int = 256,
    train_count: int = 128,
    test_count: int = 64,
    seed: int = 0,
    batch_size: int = 16,
    save: bool = False,
    save_json_path: Optional[str | Path] = None,
) -> Dict[str, object]:
    split_map = split_train_val_test_indices(len(dataset), train_count, 0, test_count, seed)
    train_x, _ = pick_dataset_images(dataset, indices=split_map["train"], image_size=image_size)
    test_x, _ = pick_dataset_images(dataset, indices=split_map["test"], image_size=image_size)
    full_rows: List[Dict[str, float | str | int | None]] = []
    matrix_law_rows: List[Dict[str, float | str]] = []
    functional_law_rows: List[Dict[str, float | str | int | None]] = []
    pca_rows: List[Dict[str, float | str | int | None]] = []
    pca_functional_rows: List[Dict[str, float | str | int | None]] = []
    pca_curve_rows: List[Dict[str, float | str | int]] = []
    distribution_rows: List[Dict[str, float | str | int]] = []

    for key in keys:
        adapter = load_baseline_adapter(
            key,
            device=device,
            rae_repo_path=rae_repo_path,
            rae_auto_clone=rae_auto_clone,
            rae_auto_download=rae_auto_download,
            posterior="mode",
        )
        for center in centers:
            maps = fit_orthogonal_maps(adapter, train_x, transforms=transforms, center=center, batch_size=batch_size)
            for split, x_split in (("train", train_x), ("test", test_x)):
                for row in evaluate_channel_maps(adapter, x_split, maps, transforms=transforms, center=center, batch_size=batch_size):
                    full_rows.append({"model": key, "center": center, "split": split, "space": "full", **row})
            matrix_law_rows.append({"model": key, "center": center, "space": "full", **group_law_metrics(maps)})
            for row in functional_group_law_metrics(adapter, test_x, maps, transforms=transforms, center=center, batch_size=batch_size):
                functional_law_rows.append({"model": key, "center": center, "split": "test", "space": "full", **row})

            pca_result = run_pca_subspace_diagnostics(
                adapter,
                train_x,
                test_x,
                transforms=transforms,
                center=center,
                component_counts=pca_component_counts,
                max_components=max(pca_component_counts),
                batch_size=batch_size,
                model_name=key,
            )
            pca_rows.extend(pca_result[0])
            pca_functional_rows.extend(pca_result[1])
            pca_curve_rows.extend(pca_result[2])
            distribution = latent_distribution_metrics(
                adapter,
                train_x,
                center=center,
                component_counts=pca_component_counts,
                batch_size=batch_size,
            )
            distribution_rows.append({"model": key, "split": "train", **distribution})
        del adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result: Dict[str, object] = {
        "full_rows": full_rows,
        "matrix_law_rows": matrix_law_rows,
        "functional_law_rows": functional_law_rows,
        "pca_rows": pca_rows,
        "pca_functional_rows": pca_functional_rows,
        "pca_curve_rows": pca_curve_rows,
        "distribution_rows": distribution_rows,
        "split_indices": split_map,
    }
    if save:
        if save_json_path is None:
            raise ValueError("save=True 时必须显式提供 save_json_path。")
        path = Path(save_json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    return result


@torch.no_grad()
def run_group_structure_study(
    dataset,
    keys: Sequence[str] = ("rae_dinov2", "rae_mae"),
    device: str | torch.device = "cuda:0",
    rae_repo_path: str | Path = "external/RAE",
    rae_auto_clone: bool = False,
    rae_auto_download: bool = False,
    posterior: str = "mode",
    generator_transforms: Sequence[str] = ("rot90", "flip_h"),
    independent_transforms: Sequence[str] = ("identity", "rot90", "rot180", "rot270", "flip_h"),
    power_transforms: Sequence[str] = C4_TRANSFORMS,
    orbit_transforms: Sequence[str] = C4_TRANSFORMS,
    centers: Sequence[str] = ("sample",),
    pca_component_counts: Sequence[int] = (3, 16, 64, 128, 256),
    image_size: int = 256,
    train_count: int = 128,
    test_count: int = 64,
    seed: int = 0,
    batch_size: int = 16,
    save: bool = False,
    save_json_path: Optional[str | Path] = None,
) -> Dict[str, object]:
    split_map = split_train_val_test_indices(len(dataset), train_count, 0, test_count, seed)
    train_x, _ = pick_dataset_images(dataset, indices=split_map["train"], image_size=image_size)
    test_x, _ = pick_dataset_images(dataset, indices=split_map["test"], image_size=image_size)
    split_tensors = {"train": train_x, "test": test_x}

    power_rows: List[Dict[str, float | str | int | None]] = []
    d4_relation_rows: List[Dict[str, float | str | int | None]] = []
    orbit_closure_rows: List[Dict[str, float | str | int]] = []
    pca_power_rows: List[Dict[str, float | str | int | None]] = []
    pca_d4_relation_rows: List[Dict[str, float | str | int | None]] = []
    pca_orbit_closure_rows: List[Dict[str, float | str | int]] = []
    pca_curve_rows: List[Dict[str, float | str | int]] = []

    for key in keys:
        adapter = load_baseline_adapter(
            key,
            device=device,
            rae_repo_path=rae_repo_path,
            rae_auto_clone=rae_auto_clone,
            rae_auto_download=rae_auto_download,
            posterior=posterior,
        )
        for center in centers:
            independent_maps = fit_orthogonal_maps(
                adapter,
                train_x,
                transforms=independent_transforms,
                center=center,
                batch_size=batch_size,
            )
            generator_maps = fit_generator_maps(
                adapter,
                train_x,
                generator_transforms=generator_transforms,
                center=center,
                batch_size=batch_size,
            )
            for split, x_split in split_tensors.items():
                for row in evaluate_power_group_maps(
                    adapter,
                    x_split,
                    generator_maps=generator_maps,
                    independent_maps=independent_maps,
                    transforms=power_transforms,
                    center=center,
                    batch_size=batch_size,
                ):
                    power_rows.append({"model": key, "center": center, "split": split, "space": "full", **row})
                for row in evaluate_d4_relation_maps(
                    adapter,
                    x_split,
                    generator_maps=generator_maps,
                    independent_maps=independent_maps,
                    center=center,
                    batch_size=batch_size,
                ):
                    d4_relation_rows.append({"model": key, "center": center, "split": split, "space": "full", **row})
                for row in rotation_orbit_closure_table(
                    adapter,
                    x_split,
                    generator_maps=generator_maps,
                    orbit_transforms=orbit_transforms,
                    center=center,
                    batch_size=batch_size,
                ):
                    orbit_closure_rows.append({"model": key, "center": center, "split": split, "space": "full", **row})

            if pca_component_counts:
                basis = fit_pca_basis(
                    adapter,
                    train_x,
                    max_components=max(pca_component_counts),
                    center=center,
                    batch_size=batch_size,
                )
                pca_curve_rows.extend(_pca_curve_rows(key, center, basis, pca_component_counts))
                for requested_k in pca_component_counts:
                    actual_k = min(int(requested_k), int(basis["components"].shape[0]))

                    def encode_projected(x_batch, k=actual_k):
                        return project_latent_to_pca(E(adapter, x_batch), basis, k)

                    independent_pca_maps = _fit_orthogonal_maps_with_encoder(
                        encode_projected,
                        train_x,
                        transforms=independent_transforms,
                        center="none",
                        batch_size=batch_size,
                        device=adapter.device,
                    )
                    generator_pca_maps = _fit_generator_maps_with_encoder(
                        encode_projected,
                        train_x,
                        generator_transforms=generator_transforms,
                        center="none",
                        batch_size=batch_size,
                        device=adapter.device,
                    )
                    for split, x_split in split_tensors.items():
                        for row in _evaluate_power_group_maps_with_encoder(
                            encode_projected,
                            x_split,
                            generator_maps=generator_pca_maps,
                            independent_maps=independent_pca_maps,
                            transforms=power_transforms,
                            center="none",
                            batch_size=batch_size,
                            device=adapter.device,
                        ):
                            pca_power_rows.append(
                                {
                                    "model": key,
                                    "center": center,
                                    "requested_k": int(requested_k),
                                    "actual_k": actual_k,
                                    "split": split,
                                    "space": "pca",
                                    **row,
                                }
                            )
                        for row in _evaluate_d4_relation_maps_with_encoder(
                            encode_projected,
                            x_split,
                            generator_maps=generator_pca_maps,
                            independent_maps=independent_pca_maps,
                            center="none",
                            batch_size=batch_size,
                            device=adapter.device,
                        ):
                            pca_d4_relation_rows.append(
                                {
                                    "model": key,
                                    "center": center,
                                    "requested_k": int(requested_k),
                                    "actual_k": actual_k,
                                    "split": split,
                                    "space": "pca",
                                    **row,
                                }
                            )
                        for row in _rotation_orbit_closure_table_with_encoder(
                            encode_projected,
                            x_split,
                            generator_maps=generator_pca_maps,
                            orbit_transforms=orbit_transforms,
                            center="none",
                            batch_size=batch_size,
                            device=adapter.device,
                        ):
                            pca_orbit_closure_rows.append(
                                {
                                    "model": key,
                                    "center": center,
                                    "requested_k": int(requested_k),
                                    "actual_k": actual_k,
                                    "split": split,
                                    "space": "pca",
                                    **row,
                                }
                            )
        del adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result: Dict[str, object] = {
        "power_rows": power_rows,
        "d4_relation_rows": d4_relation_rows,
        "orbit_closure_rows": orbit_closure_rows,
        "pca_power_rows": pca_power_rows,
        "pca_d4_relation_rows": pca_d4_relation_rows,
        "pca_orbit_closure_rows": pca_orbit_closure_rows,
        "pca_curve_rows": pca_curve_rows,
        "split_indices": split_map,
    }
    if save:
        if save_json_path is None:
            raise ValueError("save=True 时必须显式提供 save_json_path。")
        path = Path(save_json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    return result


@torch.no_grad()
def run_geometry_response_atlas(
    dataset,
    keys: Sequence[str] = ("rae_dinov2", "rae_mae"),
    device: str | torch.device = "cuda:0",
    rae_repo_path: str | Path = "external/RAE",
    rae_auto_clone: bool = False,
    rae_auto_download: bool = False,
    posterior: str = "mode",
    stage_names: Sequence[str] = DEFAULT_GEOMETRY_STAGES,
    hidden_indices: Sequence[int] = (0, 1, 3, 6, 9, 12),
    analysis_transforms: Sequence[str] = ("rot90", "flip_h"),
    independent_transforms: Sequence[str] = ("identity", "rot90", "rot180", "rot270", "flip_h"),
    generator_transforms: Sequence[str] = ("rot90", "flip_h"),
    power_transforms: Sequence[str] = C4_TRANSFORMS,
    centers: Sequence[str] = ("sample",),
    mean_residual_centers: Sequence[str] = ("none",),
    pca_component_counts: Sequence[int] = (8, 16, 32, 64, 128, 256),
    image_size: int = 256,
    train_count: int = 128,
    test_count: int = 64,
    position_count: int = 4,
    seed: int = 0,
    batch_size: int = 16,
    save: bool = False,
    save_json_path: Optional[str | Path] = None,
) -> Dict[str, object]:
    layer_result = run_layerwise_geometry_study(
        dataset,
        keys=keys,
        device=device,
        rae_repo_path=rae_repo_path,
        rae_auto_clone=rae_auto_clone,
        rae_auto_download=rae_auto_download,
        posterior=posterior,
        stage_names=stage_names,
        hidden_indices=hidden_indices,
        analysis_transforms=analysis_transforms,
        independent_transforms=independent_transforms,
        generator_transforms=generator_transforms,
        power_transforms=power_transforms,
        centers=centers,
        image_size=image_size,
        train_count=train_count,
        test_count=test_count,
        seed=seed,
        batch_size=batch_size,
    )
    mean_residual_result = run_mean_residual_geometry_study(
        dataset,
        keys=keys,
        device=device,
        rae_repo_path=rae_repo_path,
        rae_auto_clone=rae_auto_clone,
        rae_auto_download=rae_auto_download,
        posterior=posterior,
        analysis_transforms=analysis_transforms,
        independent_transforms=independent_transforms,
        generator_transforms=generator_transforms,
        power_transforms=power_transforms,
        centers=mean_residual_centers,
        image_size=image_size,
        train_count=train_count,
        test_count=test_count,
        seed=seed,
        batch_size=batch_size,
    )
    group_result = run_group_structure_study(
        dataset,
        keys=keys,
        device=device,
        rae_repo_path=rae_repo_path,
        rae_auto_clone=rae_auto_clone,
        rae_auto_download=rae_auto_download,
        posterior=posterior,
        generator_transforms=generator_transforms,
        independent_transforms=independent_transforms,
        power_transforms=power_transforms,
        orbit_transforms=C4_TRANSFORMS,
        centers=centers,
        pca_component_counts=pca_component_counts,
        image_size=image_size,
        train_count=train_count,
        test_count=test_count,
        seed=seed,
        batch_size=batch_size,
    )
    position_result = run_vit_position_embedding_study(
        dataset,
        keys=keys,
        device=device,
        rae_repo_path=rae_repo_path,
        rae_auto_clone=rae_auto_clone,
        rae_auto_download=rae_auto_download,
        transforms=_unique_transforms((*tuple(analysis_transforms), "rot180", "flip_v")),
        image_size=image_size,
        count=position_count,
        seed=seed,
        center=centers[0] if centers else "sample",
        hidden_indices=hidden_indices,
    )

    result: Dict[str, object] = {
        "layer_direct_rows": layer_result["layer_direct_rows"],
        "layer_procrustes_rows": layer_result["layer_procrustes_rows"],
        "layer_power_rows": layer_result["layer_power_rows"],
        "mean_residual_rows": mean_residual_result["mean_residual_rows"],
        "pca_subspace_rows": group_result["pca_power_rows"],
        "pca_curve_rows": group_result["pca_curve_rows"],
        "position_rows": position_result["stage_rows"],
        "position_intervention_rows": position_result["intervention_rows"],
        "split_indices": layer_result["split_indices"],
        "position_indices": position_result["indices"],
    }
    if save:
        if save_json_path is None:
            raise ValueError("save=True 时必须显式提供 save_json_path。")
        path = Path(save_json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    return result


@torch.no_grad()
def procrustes_table(adapter, x: torch.Tensor, transforms: Sequence[str] = NON_IDENTITY_TRANSFORMS) -> List[Dict[str, float | str]]:
    x = x.to(adapter.device, dtype=torch.float32)
    z = E(adapter, x)
    rows = []
    for transform in transforms:
        z_g = E(adapter, P(x, transform))
        rows.append({"transform": transform, **orthogonal_procrustes_error(z, z_g, transform, center=True)})
    return rows


@torch.no_grad()
def compare_models_table(
    adapters: Mapping[str, object],
    x: torch.Tensor,
    transforms: Sequence[str] = NON_IDENTITY_TRANSFORMS,
    center: str = "sample",
) -> List[Dict[str, float | str]]:
    rows = []
    for model_name, adapter in adapters.items():
        for row in diagnostic_table(adapter, x, transforms=transforms, center=center):
            rows.append({"model": model_name, "center": center, **row})
    return rows


@torch.no_grad()
def compare_procrustes_table(
    adapters: Mapping[str, object],
    x: torch.Tensor,
    transforms: Sequence[str] = NON_IDENTITY_TRANSFORMS,
) -> List[Dict[str, float | str]]:
    rows = []
    for model_name, adapter in adapters.items():
        for row in procrustes_table(adapter, x, transforms=transforms):
            rows.append({"model": model_name, **row})
    return rows


def pca_token_maps(tensors: Sequence[torch.Tensor]) -> List[np.ndarray]:
    maps = []
    rows = []
    shapes = []
    for tensor in tensors:
        if tensor.ndim == 4:
            tensor = tensor[0]
        channels, height, width = tensor.shape
        row = tensor.permute(1, 2, 0).reshape(height * width, channels).float()
        rows.append(row)
        shapes.append((height, width))
    all_rows = torch.cat(rows, dim=0)
    all_rows = all_rows - all_rows.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(all_rows, full_matrices=False)
    projected = all_rows @ vh[:3].T
    lo = torch.quantile(projected, 0.01, dim=0)
    hi = torch.quantile(projected, 0.99, dim=0)
    projected = ((projected - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)
    offset = 0
    for height, width in shapes:
        n = height * width
        maps.append(projected[offset:offset + n].reshape(height, width, 3).detach().cpu().numpy())
        offset += n
    return maps


def channel_norm_map(tensor: torch.Tensor) -> np.ndarray:
    if tensor.ndim == 4:
        tensor = tensor[0]
    values = tensor.detach().float().pow(2).sum(dim=0).sqrt().cpu()
    lo = torch.quantile(values, 0.01)
    hi = torch.quantile(values, 0.99)
    values = ((values - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)
    return values.numpy()


@torch.no_grad()
def orbit_latents(
    adapter,
    x: torch.Tensor,
    orbit_transforms: Sequence[str] = C4_TRANSFORMS,
) -> Dict[str, torch.Tensor]:
    x = x.to(adapter.device, dtype=torch.float32)
    return {transform: E(adapter, P(x, transform)) for transform in orbit_transforms}


def align_orbit_latent(z_g: torch.Tensor, transform: str) -> torch.Tensor:
    return P(z_g, inverse_transform(transform))


@torch.no_grad()
def orbit_consistency_matrix(
    adapter,
    x: torch.Tensor,
    orbit_transforms: Sequence[str] = C4_TRANSFORMS,
    center: str = "sample",
    sample_index: Optional[int] = None,
) -> np.ndarray:
    latents = orbit_latents(adapter, x, orbit_transforms)
    names = list(orbit_transforms)
    values = torch.zeros((len(names), len(names)), device=next(iter(latents.values())).device, dtype=torch.float32)
    for row, target in enumerate(names):
        z_target = latents[target]
        if sample_index is not None:
            z_target = z_target[sample_index:sample_index + 1]
        for col, source in enumerate(names):
            z_source = latents[source]
            if sample_index is not None:
                z_source = z_source[sample_index:sample_index + 1]
            relative = relative_c4_transform(target, source)
            values[row, col] = relative_token_error(z_target, P(z_source, relative), center=center).mean()
    return values.detach().cpu().numpy()


@torch.no_grad()
def orbit_consistency_table(
    adapters: Mapping[str, object],
    x: torch.Tensor,
    orbit_transforms: Sequence[str] = C4_TRANSFORMS,
    center: str = "sample",
) -> List[Dict[str, float | str]]:
    rows = []
    for model_name, adapter in adapters.items():
        matrix = orbit_consistency_matrix(adapter, x, orbit_transforms=orbit_transforms, center=center)
        off_diag = matrix[~np.eye(matrix.shape[0], dtype=bool)]
        rows.append(
            {
                "model": model_name,
                "center": center,
                "orbit_mean_error": float(matrix.mean()),
                "orbit_offdiag_mean_error": float(off_diag.mean()),
                "orbit_offdiag_max_error": float(off_diag.max()),
            }
        )
    return rows


@torch.no_grad()
def diagnostic_figure(
    adapter,
    x: torch.Tensor,
    transform: str = "rot90",
    sample_index: int = 0,
    center: str = "sample",
    figsize: Tuple[int, int] = (18, 8),
):
    import matplotlib.pyplot as plt

    x = x.to(adapter.device, dtype=torch.float32)
    z = E(adapter, x)
    x_g = P(x, transform)
    z_g = E(adapter, x_g)
    pz = P(z, transform)
    sim = token_similarity(pz[sample_index:sample_index + 1], z_g[sample_index:sample_index + 1], center=center)[0]
    best = sim.argmax(dim=-1)
    side = int(math.sqrt(sim.shape[0]))
    coords = grid_coords(side, side, sim.device)
    expected = torch.arange(sim.shape[0], device=sim.device)
    displacement = torch.linalg.norm(coords[best] - coords[expected], dim=-1).reshape(side, side).detach().cpu().numpy()
    aligned_zg = align_orbit_latent(z_g, transform)
    error = z_g - pz
    pca_z, pca_zg, pca_pz, pca_aligned = pca_token_maps(
        [z[sample_index], z_g[sample_index], pz[sample_index], aligned_zg[sample_index]]
    )

    fig, axes = plt.subplots(2, 4, figsize=figsize)
    panels = [
        ("x", tensor_to_image01(x[sample_index])),
        (f"g(x): {transform}", tensor_to_image01(x_g[sample_index])),
        ("PCA(E(x))", pca_z),
        ("PCA(E(gx))", pca_zg),
        ("PCA(P_g E(x))", pca_pz),
        ("PCA(P_g^-1 E(gx))", pca_aligned),
        ("error norm", channel_norm_map(error[sample_index])),
        ("best-match displacement", displacement),
    ]
    for ax, (title, image) in zip(axes.reshape(-1), panels):
        ax.imshow(image, cmap="magma" if image.ndim == 2 else None)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    return fig


@torch.no_grad()
def orbit_alignment_figure(
    adapter,
    x: torch.Tensor,
    orbit_transforms: Sequence[str] = C4_TRANSFORMS,
    sample_index: int = 0,
    figsize: Tuple[int, int] = (16, 7),
):
    import matplotlib.pyplot as plt

    x = x.to(adapter.device, dtype=torch.float32)
    latents = orbit_latents(adapter, x, orbit_transforms)
    aligned = [align_orbit_latent(latents[transform], transform)[sample_index] for transform in orbit_transforms]
    pca_maps = pca_token_maps(aligned)

    fig, axes = plt.subplots(2, len(orbit_transforms), figsize=figsize)
    for col, transform in enumerate(orbit_transforms):
        axes[0, col].imshow(tensor_to_image01(P(x, transform)[sample_index]))
        axes[0, col].set_title(f"{transform}(x)")
        axes[0, col].axis("off")
        axes[1, col].imshow(pca_maps[col])
        axes[1, col].set_title(f"P^-1 E({transform} x)")
        axes[1, col].axis("off")
    fig.tight_layout()
    return fig


@torch.no_grad()
def token_correspondence_figure(
    adapter,
    x: torch.Tensor,
    transform: str = "rot90",
    sample_index: int = 0,
    center: str = "sample",
    figsize: Tuple[int, int] = (18, 5),
):
    import matplotlib.pyplot as plt

    x = x.to(adapter.device, dtype=torch.float32)
    z = E(adapter, x)
    z_g = E(adapter, P(x, transform))
    sim = token_similarity(P(z, transform)[sample_index:sample_index + 1], z_g[sample_index:sample_index + 1], center=center)[0]
    best_score, best = sim.max(dim=-1)
    side = int(math.sqrt(sim.shape[0]))
    coords = grid_coords(side, side, sim.device)
    expected = torch.arange(sim.shape[0], device=sim.device)
    displacement = torch.linalg.norm(coords[best] - coords[expected], dim=-1).reshape(side, side)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    panels = [
        ("best-match cosine", best_score.reshape(side, side).detach().cpu().numpy(), "viridis", None),
        ("best-match displacement", displacement.detach().cpu().numpy(), "magma", None),
        ("full similarity matrix", sim.detach().cpu().numpy(), "coolwarm", (-1.0, 1.0)),
    ]
    for ax, (title, image, cmap, limits) in zip(axes, panels):
        kwargs = {"cmap": cmap}
        if limits is not None:
            kwargs.update({"vmin": limits[0], "vmax": limits[1]})
        im = ax.imshow(image, **kwargs)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


@torch.no_grad()
def orbit_consistency_figure(
    adapter,
    x: torch.Tensor,
    orbit_transforms: Sequence[str] = C4_TRANSFORMS,
    center: str = "sample",
    sample_index: Optional[int] = None,
    figsize: Tuple[int, int] = (6, 5),
):
    import matplotlib.pyplot as plt

    matrix = orbit_consistency_matrix(
        adapter,
        x,
        orbit_transforms=orbit_transforms,
        center=center,
        sample_index=sample_index,
    )
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(matrix, cmap="magma")
    ax.set_xticks(range(len(orbit_transforms)), labels=orbit_transforms, rotation=45, ha="right")
    ax.set_yticks(range(len(orbit_transforms)), labels=orbit_transforms)
    ax.set_title("orbit consistency error")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


@torch.no_grad()
def similarity_figure(
    adapter,
    x: torch.Tensor,
    transform: str = "rot90",
    sample_index: int = 0,
    center: str = "sample",
    figsize: Tuple[int, int] = (7, 6),
):
    import matplotlib.pyplot as plt

    x = x.to(adapter.device, dtype=torch.float32)
    z = E(adapter, x)
    z_g = E(adapter, P(x, transform))
    sim = token_similarity(P(z, transform)[sample_index:sample_index + 1], z_g[sample_index:sample_index + 1], center=center)[0]
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(sim.detach().cpu().numpy(), vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_title(f"cosine similarity: P_{transform} E(x) vs E({transform} x)")
    ax.set_xlabel("token in E(gx)")
    ax.set_ylabel("token in P_g E(x)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig
