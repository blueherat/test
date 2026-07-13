from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.dinov2_token_diagnostics import configure_fp32, load_named_dataset  # noqa: E402
from baselines.visual_adapters import load_rae_adapter  # noqa: E402


@dataclass
class RAEReconstructionRFIDConfig:
    dataset_name: str = "imagenet_parquet"
    data_root: str = "/data/shared"
    dataset_path: str = "/data/shared/imagenet-1k"
    dataset_split: str = "validation"
    image_size: int = 256
    count: int = 50000
    model_key: str = "rae_dinov2"
    rae_repo_path: str = "external/RAE"
    rae_auto_clone: bool = False
    rae_auto_download: bool = False
    device: str = "cuda:0"
    batch_size: int = 32
    num_workers: int = 4
    output_dir: str = "artifacts/rae_rfid"
    run_name: str = ""
    fid_batch_size: int = 128
    skip_reconstruction: bool = False
    skip_rfid: bool = False
    overwrite: bool = False


def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    """ADM center crop used by the RAE sampling script."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.Resampling.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.Resampling.BICUBIC)

    arr = np.array(pil_image.convert("RGB"))
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


def pil_to_m11_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1) * 2.0 - 1.0


class IndexedCroppedImageDataset(Dataset):
    def __init__(self, dataset, indices: Sequence[int], image_size: int):
        self.dataset = dataset
        self.indices = [int(i) for i in indices]
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        dataset_index = self.indices[item]
        sample = self.dataset[dataset_index]
        image = sample[0] if isinstance(sample, (tuple, list)) else sample
        cropped = center_crop_arr(image.convert("RGB"), self.image_size)
        reference = np.asarray(cropped, dtype=np.uint8)
        return pil_to_m11_tensor(cropped), torch.from_numpy(reference.copy()), int(dataset_index)


def contiguous_indices(dataset_len: int, count: int) -> List[int]:
    if count <= 0:
        count = dataset_len
    if dataset_len < count:
        raise ValueError(f"dataset has {dataset_len} images, less than requested count={count}")
    return list(range(int(count)))


def tensor_m11_to_uint8_nhwc(x: torch.Tensor) -> np.ndarray:
    x01 = ((x.detach().float().clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0)
    return x01.mul(255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()


def check_npy(path: Path, count: int, image_size: int) -> bool:
    if not path.exists():
        return False
    try:
        arr = np.load(path, mmap_mode="r")
        return arr.shape == (count, image_size, image_size, 3) and arr.dtype == np.uint8
    except Exception:
        return False


def create_or_open_npy(path: Path, shape: tuple[int, int, int, int], overwrite: bool):
    if path.exists() and overwrite:
        path.unlink()
    if path.exists():
        return np.load(path, mmap_mode="r+")
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=shape)


@torch.no_grad()
def generate_reconstructions(cfg: RAEReconstructionRFIDConfig, run_dir: Path, device: torch.device) -> dict:
    dataset = load_named_dataset(
        cfg.dataset_name,
        cfg.data_root,
        split=cfg.dataset_split,
        dataset_path=cfg.dataset_path,
    )
    indices = contiguous_indices(len(dataset), cfg.count)
    count = len(indices)
    shape = (count, cfg.image_size, cfg.image_size, 3)

    reference_path = run_dir / f"reference_{cfg.dataset_split}_{cfg.image_size}_n{count}.npy"
    recon_path = run_dir / f"reconstruction_{cfg.model_key}_{cfg.dataset_split}_{cfg.image_size}_n{count}.npy"

    reference_ready = check_npy(reference_path, count, cfg.image_size)
    recon_ready = check_npy(recon_path, count, cfg.image_size)
    if cfg.skip_reconstruction and not (reference_ready and recon_ready):
        raise FileNotFoundError("skip_reconstruction=True but reference/reconstruction arrays are incomplete.")
    if reference_ready and recon_ready and not cfg.overwrite:
        return {"reference": str(reference_path), "reconstruction": str(recon_path), "count": count}

    reference = create_or_open_npy(reference_path, shape, overwrite=cfg.overwrite)
    reconstruction = create_or_open_npy(recon_path, shape, overwrite=cfg.overwrite)

    loader = DataLoader(
        IndexedCroppedImageDataset(dataset, indices, cfg.image_size),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

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

    offset = 0
    for x_cpu, ref_cpu, _ in loader:
        x = x_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
        z = rae.encode(x)
        recon = rae.decode(z)
        batch = x.shape[0]
        reference[offset : offset + batch] = ref_cpu.numpy()
        reconstruction[offset : offset + batch] = tensor_m11_to_uint8_nhwc(recon)
        offset += batch
        if offset % max(1, cfg.batch_size * 20) == 0 or offset == count:
            print(f"{cfg.model_key}: reconstructed {offset}/{count}", flush=True)

    reference.flush()
    reconstruction.flush()
    return {"reference": str(reference_path), "reconstruction": str(recon_path), "count": count}


def compute_rfid(reference_path: Path, reconstruction_path: Path, cfg: RAEReconstructionRFIDConfig, device: torch.device) -> float:
    src_path = ROOT / "external" / "RAE" / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from eval.fid import calculate_rfid

    reference = np.load(reference_path, mmap_mode="r")
    reconstruction = np.load(reconstruction_path, mmap_mode="r")
    cuda_device = "cuda" if device.type == "cuda" else "cpu"
    return float(calculate_rfid(reference, reconstruction, bs=cfg.fid_batch_size, device=cuda_device))


def build_run_dir(cfg: RAEReconstructionRFIDConfig) -> Path:
    name = cfg.run_name.strip()
    if not name:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        name = f"{cfg.model_key}_{cfg.dataset_split}_n{cfg.count}_{stamp}"
    run_dir = Path(cfg.output_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run(cfg: RAEReconstructionRFIDConfig) -> dict:
    configure_fp32()
    torch.set_grad_enabled(False)
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    run_dir = build_run_dir(cfg)
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    arrays = generate_reconstructions(cfg, run_dir, device)
    metrics = {"config": asdict(cfg), "arrays": arrays, "rfid": None}
    if not cfg.skip_rfid:
        metrics["rfid"] = compute_rfid(Path(arrays["reference"]), Path(arrays["reconstruction"]), cfg, device)
        print(f"rFID: {metrics['rfid']:.6f}", flush=True)

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps({"run_dir": str(run_dir), "rfid": metrics["rfid"], "arrays": arrays}, ensure_ascii=False, indent=2), flush=True)
    return metrics


def parse_args() -> RAEReconstructionRFIDConfig:
    parser = argparse.ArgumentParser(description="Compute RAE reconstruction rFID on ImageNet parquet validation.")
    parser.add_argument("--dataset-name", default=RAEReconstructionRFIDConfig.dataset_name)
    parser.add_argument("--data-root", default=RAEReconstructionRFIDConfig.data_root)
    parser.add_argument("--dataset-path", default=RAEReconstructionRFIDConfig.dataset_path)
    parser.add_argument("--dataset-split", default=RAEReconstructionRFIDConfig.dataset_split)
    parser.add_argument("--image-size", type=int, default=RAEReconstructionRFIDConfig.image_size)
    parser.add_argument("--count", type=int, default=RAEReconstructionRFIDConfig.count)
    parser.add_argument("--model-key", default=RAEReconstructionRFIDConfig.model_key)
    parser.add_argument("--rae-repo-path", default=RAEReconstructionRFIDConfig.rae_repo_path)
    parser.add_argument("--rae-auto-clone", action="store_true")
    parser.add_argument("--rae-auto-download", action="store_true")
    parser.add_argument("--device", default=RAEReconstructionRFIDConfig.device)
    parser.add_argument("--batch-size", type=int, default=RAEReconstructionRFIDConfig.batch_size)
    parser.add_argument("--num-workers", type=int, default=RAEReconstructionRFIDConfig.num_workers)
    parser.add_argument("--output-dir", default=RAEReconstructionRFIDConfig.output_dir)
    parser.add_argument("--run-name", default=RAEReconstructionRFIDConfig.run_name)
    parser.add_argument("--fid-batch-size", type=int, default=RAEReconstructionRFIDConfig.fid_batch_size)
    parser.add_argument("--skip-reconstruction", action="store_true")
    parser.add_argument("--skip-rfid", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    return RAEReconstructionRFIDConfig(
        dataset_name=args.dataset_name,
        data_root=args.data_root,
        dataset_path=args.dataset_path,
        dataset_split=args.dataset_split,
        image_size=args.image_size,
        count=args.count,
        model_key=args.model_key,
        rae_repo_path=args.rae_repo_path,
        rae_auto_clone=args.rae_auto_clone,
        rae_auto_download=args.rae_auto_download,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        run_name=args.run_name,
        fid_batch_size=args.fid_batch_size,
        skip_reconstruction=args.skip_reconstruction,
        skip_rfid=args.skip_rfid,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    run(parse_args())
