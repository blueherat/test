from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from baselines.visual_adapters import load_rae_adapter


TRANSFORMS: Tuple[str, ...] = ("identity", "rot90", "rot180", "rot270", "flip_h", "flip_v")
NON_IDENTITY_TRANSFORMS: Tuple[str, ...] = tuple(g for g in TRANSFORMS if g != "identity")


def configure_fp32() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")


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
    raise ValueError(f"未知变换：{transform}")


P = apply_d4


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


def load_named_dataset(name: str, root: str, split: str = "train", download: bool = False, dataset_path: str = ""):
    from torchvision.datasets import CIFAR10, CIFAR100, STL10, Caltech101, Flowers102, ImageFolder, OxfordIIITPet

    name = normalize_dataset_name(name)
    split = (split or "train").strip().lower()
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
        f"{name}。可选：caltech101, stl10, flowers102, oxford_iiit_pet, cifar10, cifar100, image_folder"
    )


def split_indices(total: int, count: int, seed: int) -> List[int]:
    if count <= 0:
        raise ValueError("count 必须大于 0。")
    if total < count:
        raise ValueError(f"数据集只有 {total} 张，少于请求的 {count} 张。")
    rng = np.random.default_rng(seed)
    return [int(i) for i in rng.permutation(total)[:count]]


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
        "direct_error": float(direct_error(z_g, pz).mean().detach().cpu()),
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
    if direct.item() <= 1e-8:
        relative_gain = torch.zeros((), device=direct.device, dtype=direct.dtype) if err.item() <= 1e-8 else torch.tensor(float("-inf"))
    else:
        relative_gain = 1.0 - err / direct
    return {
        "direct_centered_error": float(direct.detach().cpu()),
        "orthogonal_procrustes_error": float(err.detach().cpu()),
        "relative_gain": float(relative_gain.detach().cpu()),
    }


@torch.no_grad()
def procrustes_table(adapter, x: torch.Tensor, transforms: Sequence[str] = NON_IDENTITY_TRANSFORMS) -> List[Dict[str, float | str]]:
    x = x.to(adapter.device, dtype=torch.float32)
    z = E(adapter, x)
    rows = []
    for transform in transforms:
        z_g = E(adapter, P(x, transform))
        rows.append({"transform": transform, **orthogonal_procrustes_error(z, z_g, transform, center=True)})
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


@torch.no_grad()
def diagnostic_figure(
    adapter,
    x: torch.Tensor,
    transform: str = "rot90",
    sample_index: int = 0,
    center: str = "sample",
    figsize: Tuple[int, int] = (14, 8),
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
    pca_z, pca_zg, pca_pz = pca_token_maps([z[sample_index], z_g[sample_index], pz[sample_index]])

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    panels = [
        ("x", tensor_to_image01(x[sample_index])),
        (f"g(x): {transform}", tensor_to_image01(x_g[sample_index])),
        ("PCA(E(x))", pca_z),
        ("PCA(E(gx))", pca_zg),
        ("PCA(P_g E(x))", pca_pz),
        ("best-match displacement", displacement),
    ]
    for ax, (title, image) in zip(axes.reshape(-1), panels):
        ax.imshow(image, cmap="magma" if image.ndim == 2 else None)
        ax.set_title(title)
        ax.axis("off")
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
