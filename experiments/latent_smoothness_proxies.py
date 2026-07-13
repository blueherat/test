from __future__ import annotations

import argparse
import json
import math
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.dinov2_token_diagnostics import configure_fp32, load_named_dataset  # noqa: E402
from baselines.visual_adapters import load_rae_adapter  # noqa: E402


@dataclass
class SmoothnessProxyConfig:
    dataset_name: str = "imagenet_parquet"
    data_root: str = "/data/shared"
    dataset_path: str = "/data/shared/imagenet-1k"
    dataset_split: str = "validation"
    image_size: int = 256
    count: int = 128
    batch_size: int = 16
    num_workers: int = 4
    device: str = "cuda:0"
    rae_repo_path: str = "external/RAE"
    model: tuple[str, ...] = ("official=rae_dinov2",)
    noise_sigma: tuple[float, ...] = (0.01, 0.03, 0.05, 0.10, 0.20)
    decoder_noise_sigma: tuple[float, ...] = (0.01, 0.03, 0.05, 0.10)
    hfr_threshold: tuple[float, ...] = (0.35, 0.50, 0.70)
    pca_k: tuple[int, ...] = (8, 32, 64)
    local_k: int = 16
    decoder_count: int = 16
    output_dir: str = "artifacts/latent_smoothness"
    run_name: str = ""
    skip_decoder: bool = False
    overwrite: bool = False


class IndexedImageDataset(Dataset):
    def __init__(self, dataset, indices: Sequence[int], image_size: int):
        self.dataset = dataset
        self.indices = [int(i) for i in indices]
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int]:
        index = self.indices[item]
        sample = self.dataset[index]
        image = sample[0] if isinstance(sample, (tuple, list)) else sample
        image = center_crop_resize(image.convert("RGB"), self.image_size)
        return pil_to_tensor_m11(image), index


class ModelAdapter:
    def __init__(self, name: str, model: torch.nn.Module, device: torch.device, input_is_m11: bool):
        self.name = name
        self.model = model
        self.device = device
        self.input_is_m11 = bool(input_is_m11)

    @torch.no_grad()
    def encode(self, x_m11: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        x = x_m11.to(self.device, dtype=torch.float32, non_blocking=True)
        if self.input_is_m11:
            return self.model.encode(x)
        return self.model.encode(((x + 1.0) * 0.5).clamp(0.0, 1.0))

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        x01 = self.model.decode(z.to(self.device, dtype=torch.float32, non_blocking=True)).clamp(0.0, 1.0)
        return x01 * 2.0 - 1.0


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


@contextmanager
def pushd(path: Path):
    old = Path.cwd()
    try:
        path.mkdir(parents=True, exist_ok=True)
        import os

        os.chdir(path)
        yield
    finally:
        import os

        os.chdir(old)


def _maybe_abs(path_like: str | None, base: Path) -> str | None:
    if path_like is None:
        return None
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return str(path)
    if path_like.startswith(("facebook/", "google/", "openai/")):
        return path_like
    return str((base / path).resolve())


def load_stage1_from_config(config_path: str | Path, repo_path: Path, device: torch.device) -> ModelAdapter:
    import yaml

    config_path = Path(config_path).expanduser().resolve()
    src_path = str((repo_path / "src").resolve())
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    from utils.model_utils import instantiate_from_config

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    stage_1 = dict(config["stage_1"])
    params = dict(stage_1.get("params", {}))

    if "adapter_checkpoint_path" in params:
        params["adapter_checkpoint_path"] = _maybe_abs(params["adapter_checkpoint_path"], ROOT)
    base_params = dict(params.get("base_rae_params", {}))
    for key in ("decoder_config_path", "pretrained_decoder_path", "normalization_stat_path"):
        if key in base_params:
            base_params[key] = _maybe_abs(base_params[key], repo_path)
    params["base_rae_params"] = base_params

    for key in ("decoder_config_path", "pretrained_decoder_path", "normalization_stat_path"):
        if key in params:
            params[key] = _maybe_abs(params[key], repo_path)
    stage_1["params"] = params

    with pushd(repo_path):
        model = instantiate_from_config(stage_1)
    model = model.to(device=device, dtype=torch.float32).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return ModelAdapter(name=config_path.stem, model=model, device=device, input_is_m11=False)


def parse_model_spec(spec: str) -> tuple[str, str, str]:
    if "=" in spec:
        name, value = spec.split("=", 1)
    else:
        value = spec
        name = Path(value).stem if value.endswith((".yaml", ".yml")) else value
    value = value.strip()
    name = name.strip() or "model"
    if value.startswith("config:"):
        return name, "config", value[len("config:") :]
    if value.endswith((".yaml", ".yml")):
        return name, "config", value
    return name, "rae_key", value


def load_model(spec: str, cfg: SmoothnessProxyConfig, device: torch.device) -> ModelAdapter:
    name, kind, value = parse_model_spec(spec)
    repo_path = (ROOT / cfg.rae_repo_path).resolve() if not Path(cfg.rae_repo_path).is_absolute() else Path(cfg.rae_repo_path)
    if kind == "config":
        adapter = load_stage1_from_config(value, repo_path, device)
        adapter.name = name
        return adapter

    rae = load_rae_adapter(
        value,
        repo_path=repo_path,
        device=device,
        dtype=torch.float32,
        auto_clone=False,
        auto_download=False,
    )
    for param in rae.model.parameters():
        param.requires_grad_(False)
    return ModelAdapter(name=name, model=rae.model, device=device, input_is_m11=False)


def contiguous_indices(dataset_len: int, count: int) -> list[int]:
    count = dataset_len if count <= 0 else int(count)
    if count > dataset_len:
        raise ValueError(f"dataset has only {dataset_len} items, requested {count}")
    return list(range(count))


@torch.no_grad()
def collect_latents(model: ModelAdapter, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    latents = []
    images = []
    for x_cpu, _ in loader:
        x = x_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
        z = model.encode(x)
        latents.append(z.detach().cpu())
        images.append(x_cpu.detach().cpu())
        print(f"{model.name}: encoded {sum(t.shape[0] for t in latents)}/{len(loader.dataset)}", flush=True)
    return torch.cat(latents, dim=0), torch.cat(images, dim=0)


def flatten_latents(z: torch.Tensor) -> torch.Tensor:
    return z.float().reshape(z.shape[0], -1).contiguous()


def participation_ratio(values: torch.Tensor, eps: float = 1e-12) -> float:
    values = values.float().clamp_min(0)
    total = values.sum()
    if float(total) <= eps:
        return 0.0
    return float((total * total / (values.square().sum() + eps)).cpu())


def entropy_effective_rank(values: torch.Tensor, eps: float = 1e-12) -> float:
    values = values.float().clamp_min(0)
    total = values.sum()
    if float(total) <= eps:
        return 0.0
    p = values / total
    return float(torch.exp(-(p * torch.log(p + eps)).sum()).cpu())


def spectrum_metrics(flat: torch.Tensor) -> dict:
    x = flat.float()
    x = x - x.mean(dim=0, keepdim=True)
    gram = x @ x.T / max(1, x.shape[1])
    eig = torch.linalg.eigvalsh(gram).flip(0).clamp_min(0)
    total = eig.sum().clamp_min(1e-12)
    explained = eig / total
    return {
        "global_participation_rank_sample_limited": participation_ratio(eig),
        "global_entropy_rank_sample_limited": entropy_effective_rank(eig),
        "explained_top1": float(explained[:1].sum().cpu()),
        "explained_top5": float(explained[:5].sum().cpu()),
        "explained_top10": float(explained[:10].sum().cpu()),
        "explained_top32": float(explained[:32].sum().cpu()),
        "nonzero_spectrum_count": int((eig > eig.max().clamp_min(1e-12) * 1e-7).sum().cpu()),
    }


def spatial_frequency_metrics(z: torch.Tensor, thresholds: Iterable[float]) -> dict:
    if z.ndim != 4:
        return {}
    _, _, h, w = z.shape
    fft = torch.fft.fftn(z.float(), dim=(-2, -1), norm="ortho")
    power = fft.abs().square()
    fy = torch.fft.fftfreq(h, device=z.device).view(h, 1)
    fx = torch.fft.fftfreq(w, device=z.device).view(1, w)
    radius = torch.sqrt(fx.square() + fy.square())
    radius = radius / radius.max().clamp_min(1e-12)
    total = power.sum(dim=(-2, -1)).clamp_min(1e-12)
    metrics = {}
    for threshold in thresholds:
        mask = radius > float(threshold)
        ratio = power[..., mask].sum(dim=-1) / total
        metrics[f"hfr_gt_{threshold:.2f}"] = float(ratio.mean().cpu())
    low_mask = radius <= 0.35
    high_mask = radius > 0.50
    low = power[..., low_mask].sum(dim=-1).clamp_min(1e-12)
    high = power[..., high_mask].sum(dim=-1)
    metrics["high_over_low_0.50_0.35"] = float((high / low).mean().cpu())
    return metrics


def pairwise_distances(flat: torch.Tensor) -> torch.Tensor:
    return torch.cdist(flat.float(), flat.float(), p=2)


def nearest_neighbor_metrics(flat: torch.Tensor) -> dict:
    dist = pairwise_distances(flat)
    n = dist.shape[0]
    dist.fill_diagonal_(float("inf"))
    nn = dist.min(dim=1).values
    norm = flat.norm(dim=1).clamp_min(1e-12)
    return {
        "nearest_neighbor_l2_mean": float(nn.mean().cpu()),
        "nearest_neighbor_l2_median": float(nn.median().cpu()),
        "nearest_neighbor_rel_mean": float((nn / norm).mean().cpu()),
    }


def knn_denoising_metrics(flat: torch.Tensor, sigmas: Sequence[float]) -> list[dict]:
    x = flat.float()
    n = x.shape[0]
    latent_rms = x.square().mean().sqrt().clamp_min(1e-12)
    clean_dist = pairwise_distances(x)
    clean_dist.fill_diagonal_(float("inf"))
    clean_nn = clean_dist.min(dim=1).values
    generator = torch.Generator(device=x.device).manual_seed(0)
    rows = []
    for sigma in sigmas:
        noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
        noisy = x + float(sigma) * latent_rms * noise
        dist = torch.cdist(noisy, x, p=2)
        nn_dist, nn_idx = dist.min(dim=1)
        self_dist = dist[torch.arange(n), torch.arange(n)]
        recovered = nn_idx == torch.arange(n)
        rows.append(
            {
                "sigma_rel_rms": float(sigma),
                "top1_self_recovery": float(recovered.float().mean().cpu()),
                "mean_noisy_to_self": float(self_dist.mean().cpu()),
                "mean_noisy_to_nn": float(nn_dist.mean().cpu()),
                "mean_clean_nn": float(clean_nn.mean().cpu()),
                "noise_over_clean_nn": float((self_dist / clean_nn.clamp_min(1e-12)).mean().cpu()),
                "nn_distance_over_self": float((nn_dist / self_dist.clamp_min(1e-12)).mean().cpu()),
            }
        )
    return rows


def pca_knn_denoising_metrics(
    flat: torch.Tensor,
    sigmas: Sequence[float],
    pca_ks: Sequence[int],
) -> list[dict]:
    x_raw = flat.float()
    mean = x_raw.mean(dim=0, keepdim=True)
    x = x_raw - mean
    n, d = x.shape
    valid_ks = sorted({int(k) for k in pca_ks if int(k) > 0})
    if not valid_ks:
        return []
    max_k = min(max(valid_ks), n - 1, d)
    if max_k < 1:
        return []

    # Low-rank PCA keeps this usable for RAE's 196k-dimensional token map.
    _, _, v = torch.pca_lowrank(x, q=max_k, center=False, niter=4)
    latent_rms = x_raw.square().mean().sqrt().clamp_min(1e-12)
    generator = torch.Generator(device=x_raw.device).manual_seed(17)
    rows = []
    for k in valid_ks:
        k = min(k, max_k)
        basis = v[:, :k].contiguous()
        clean = x @ basis
        clean_dist = torch.cdist(clean, clean, p=2)
        clean_dist.fill_diagonal_(float("inf"))
        clean_nn = clean_dist.min(dim=1).values
        for sigma in sigmas:
            noise = torch.randn(x_raw.shape, generator=generator, device=x_raw.device, dtype=x_raw.dtype)
            noisy = (x_raw + float(sigma) * latent_rms * noise - mean) @ basis
            dist = torch.cdist(noisy, clean, p=2)
            nn_dist, nn_idx = dist.min(dim=1)
            self_dist = dist[torch.arange(n), torch.arange(n)]
            recovered = nn_idx == torch.arange(n)
            rows.append(
                {
                    "pca_k": int(k),
                    "sigma_rel_rms": float(sigma),
                    "top1_self_recovery": float(recovered.float().mean().cpu()),
                    "mean_noisy_to_self": float(self_dist.mean().cpu()),
                    "mean_noisy_to_nn": float(nn_dist.mean().cpu()),
                    "mean_clean_nn": float(clean_nn.mean().cpu()),
                    "noise_over_clean_nn": float((self_dist / clean_nn.clamp_min(1e-12)).mean().cpu()),
                    "nn_distance_over_self": float((nn_dist / self_dist.clamp_min(1e-12)).mean().cpu()),
                }
            )
    return rows


def local_intrinsic_metrics(flat: torch.Tensor, local_k: int) -> dict:
    x = flat.float()
    n = x.shape[0]
    k = min(max(2, int(local_k)), n - 1)
    dist = pairwise_distances(x)
    dist.fill_diagonal_(float("inf"))
    indices = dist.topk(k=k, largest=False).indices
    ranks_pr = []
    ranks_ent = []
    anisotropy = []
    for i in range(n):
        neighbors = x[indices[i]]
        local = neighbors - neighbors.mean(dim=0, keepdim=True)
        gram = local @ local.T / max(1, local.shape[1])
        eig = torch.linalg.eigvalsh(gram).flip(0).clamp_min(0)
        ranks_pr.append(participation_ratio(eig))
        ranks_ent.append(entropy_effective_rank(eig))
        denom = eig[1:].mean().clamp_min(1e-12) if eig.numel() > 1 else eig[0].clamp_min(1e-12)
        anisotropy.append(float((eig[0] / denom).cpu()))
    pr = torch.tensor(ranks_pr)
    ent = torch.tensor(ranks_ent)
    aniso = torch.tensor(anisotropy)
    return {
        "local_k": int(k),
        "local_participation_rank_mean": float(pr.mean()),
        "local_participation_rank_std": float(pr.std(unbiased=False)),
        "local_entropy_rank_mean": float(ent.mean()),
        "local_entropy_rank_std": float(ent.std(unbiased=False)),
        "local_anisotropy_top_over_rest_mean": float(aniso.mean()),
    }


@torch.no_grad()
def decoder_noise_metrics(
    model: ModelAdapter,
    images_m11: torch.Tensor,
    latents: torch.Tensor,
    sigmas: Sequence[float],
    count: int,
) -> list[dict]:
    if count <= 0:
        return []
    device = model.device
    z = latents[:count].to(device=device, dtype=torch.float32)
    x = images_m11[:count].to(device=device, dtype=torch.float32)
    clean = model.decode(z).detach()
    latent_rms = z.square().mean().sqrt().clamp_min(1e-12)
    rows = []
    generator = torch.Generator(device=device).manual_seed(123)
    for sigma in sigmas:
        eps = torch.randn(z.shape, generator=generator, device=device, dtype=z.dtype)
        noisy = model.decode(z + float(sigma) * latent_rms * eps).detach()
        mse_clean = F.mse_loss(noisy, clean).item()
        mse_input = F.mse_loss(noisy, x).item()
        rows.append(
            {
                "sigma_rel_rms": float(sigma),
                "mse_to_clean_recon": float(mse_clean),
                "psnr_to_clean_recon": float(-10.0 * math.log10(max(mse_clean, 1e-12))),
                "mse_to_input": float(mse_input),
                "psnr_to_input": float(-10.0 * math.log10(max(mse_input, 1e-12))),
            }
        )
    return rows


def summarize_model(
    model: ModelAdapter,
    loader: DataLoader,
    cfg: SmoothnessProxyConfig,
    device: torch.device,
) -> dict:
    z, images = collect_latents(model, loader, device)
    flat = flatten_latents(z)
    metrics = {
        "model": model.name,
        "latent_shape": list(z.shape),
        "latent_mean": float(z.mean()),
        "latent_std": float(z.std(unbiased=False)),
        "latent_abs_mean": float(z.abs().mean()),
        "latent_rms": float(z.square().mean().sqrt()),
        "spectrum": spectrum_metrics(flat),
        "spatial_frequency": spatial_frequency_metrics(z, cfg.hfr_threshold),
        "nearest_neighbor": nearest_neighbor_metrics(flat),
        "knn_denoising": knn_denoising_metrics(flat, cfg.noise_sigma),
        "pca_knn_denoising": pca_knn_denoising_metrics(flat, cfg.noise_sigma, cfg.pca_k),
        "local_intrinsic": local_intrinsic_metrics(flat, cfg.local_k),
        "decoder_noise": [],
    }
    if not cfg.skip_decoder:
        metrics["decoder_noise"] = decoder_noise_metrics(
            model,
            images,
            z,
            cfg.decoder_noise_sigma,
            min(cfg.decoder_count, z.shape[0]),
        )
    return metrics


def build_run_dir(cfg: SmoothnessProxyConfig) -> Path:
    run_name = cfg.run_name.strip()
    if not run_name:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        run_name = f"{cfg.dataset_split}_n{cfg.count}_{stamp}"
    run_dir = Path(cfg.output_dir) / run_name
    if run_dir.exists() and not cfg.overwrite:
        raise FileExistsError(f"run_dir exists: {run_dir}; pass --overwrite or choose --run-name")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run(cfg: SmoothnessProxyConfig) -> dict:
    configure_fp32()
    torch.set_grad_enabled(False)
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    dataset = load_named_dataset(
        cfg.dataset_name,
        cfg.data_root,
        split=cfg.dataset_split,
        dataset_path=cfg.dataset_path,
    )
    indices = contiguous_indices(len(dataset), cfg.count)
    image_dataset = IndexedImageDataset(dataset, indices, cfg.image_size)
    loader = DataLoader(
        image_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    run_dir = build_run_dir(cfg)
    results = {"config": asdict(cfg), "models": []}
    for spec in cfg.model:
        model = load_model(spec, cfg, device)
        print(f"== model {model.name} ==", flush=True)
        results["models"].append(summarize_model(model, loader, cfg, device))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output = run_dir / "metrics.json"
    with output.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(json.dumps({"run_dir": str(run_dir), "metrics": str(output)}, ensure_ascii=False, indent=2), flush=True)
    return results


def parse_args() -> SmoothnessProxyConfig:
    parser = argparse.ArgumentParser(description="No-training latent smoothness/generative-friendliness proxies.")
    parser.add_argument("--dataset-name", default=SmoothnessProxyConfig.dataset_name)
    parser.add_argument("--data-root", default=SmoothnessProxyConfig.data_root)
    parser.add_argument("--dataset-path", default=SmoothnessProxyConfig.dataset_path)
    parser.add_argument("--dataset-split", default=SmoothnessProxyConfig.dataset_split)
    parser.add_argument("--image-size", type=int, default=SmoothnessProxyConfig.image_size)
    parser.add_argument("--count", type=int, default=SmoothnessProxyConfig.count)
    parser.add_argument("--batch-size", type=int, default=SmoothnessProxyConfig.batch_size)
    parser.add_argument("--num-workers", type=int, default=SmoothnessProxyConfig.num_workers)
    parser.add_argument("--device", default=SmoothnessProxyConfig.device)
    parser.add_argument("--rae-repo-path", default=SmoothnessProxyConfig.rae_repo_path)
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="Repeated model spec. Examples: official=rae_dinov2, adapter_e3=config:experiments/configs/finetune...yaml",
    )
    parser.add_argument("--noise-sigma", nargs="+", type=float, default=list(SmoothnessProxyConfig.noise_sigma))
    parser.add_argument(
        "--decoder-noise-sigma",
        nargs="+",
        type=float,
        default=list(SmoothnessProxyConfig.decoder_noise_sigma),
    )
    parser.add_argument("--hfr-threshold", nargs="+", type=float, default=list(SmoothnessProxyConfig.hfr_threshold))
    parser.add_argument("--pca-k", nargs="+", type=int, default=list(SmoothnessProxyConfig.pca_k))
    parser.add_argument("--local-k", type=int, default=SmoothnessProxyConfig.local_k)
    parser.add_argument("--decoder-count", type=int, default=SmoothnessProxyConfig.decoder_count)
    parser.add_argument("--output-dir", default=SmoothnessProxyConfig.output_dir)
    parser.add_argument("--run-name", default=SmoothnessProxyConfig.run_name)
    parser.add_argument("--skip-decoder", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    return SmoothnessProxyConfig(
        dataset_name=args.dataset_name,
        data_root=args.data_root,
        dataset_path=args.dataset_path,
        dataset_split=args.dataset_split,
        image_size=args.image_size,
        count=args.count,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        rae_repo_path=args.rae_repo_path,
        model=tuple(args.model or SmoothnessProxyConfig.model),
        noise_sigma=tuple(args.noise_sigma),
        decoder_noise_sigma=tuple(args.decoder_noise_sigma),
        hfr_threshold=tuple(args.hfr_threshold),
        pca_k=tuple(args.pca_k),
        local_k=args.local_k,
        decoder_count=args.decoder_count,
        output_dir=args.output_dir,
        run_name=args.run_name,
        skip_decoder=args.skip_decoder,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    run(parse_args())
