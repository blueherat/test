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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.dinov2_token_diagnostics import P, configure_fp32, load_named_dataset  # noqa: E402
from baselines.visual_adapters import load_rae_adapter  # noqa: E402
from experiments.latent_equiv_adapter import InvertibleLatentAdapter  # noqa: E402
from experiments.rae_reconstruction_rfid import (  # noqa: E402
    IndexedCroppedImageDataset,
    contiguous_indices,
    tensor_m11_to_uint8_nhwc,
)


@dataclass
class AdapterTransformRFIDConfig:
    checkpoint: str
    dataset_name: str = "imagenet_parquet"
    data_root: str = "/data/shared"
    dataset_path: str = "/data/shared/imagenet-1k"
    dataset_split: str = "validation"
    image_size: int = 256
    count: int = 50000
    transform: str = "rot90"
    model_key: str = ""
    rae_repo_path: str = "external/RAE"
    rae_auto_clone: bool = False
    rae_auto_download: bool = False
    device: str = "cuda:0"
    batch_size: int = 32
    num_workers: int = 4
    output_dir: str = "artifacts/adapter_rfid"
    run_name: str = ""
    fid_batch_size: int = 128
    skip_generation: bool = False
    skip_rfid: bool = False
    overwrite: bool = False


def check_npy(path: Path, count: int, image_size: int) -> bool:
    if not path.exists():
        return False
    try:
        arr = np.load(path, mmap_mode="r")
        return arr.shape == (count, image_size, image_size, 3) and arr.dtype == np.uint8
    except Exception:
        return False


def create_or_open_npy(path: Path, count: int, image_size: int, overwrite: bool):
    if path.exists() and overwrite:
        path.unlink()
    if path.exists():
        return np.load(path, mmap_mode="r+")
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(count, image_size, image_size, 3))


def load_flow(checkpoint_path: Path, device: torch.device) -> tuple[InvertibleLatentAdapter, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = dict(checkpoint.get("config", {}))
    channels = int(checkpoint.get("channels", cfg.get("channels", 768)))
    hidden_channels = int(cfg.get("hidden_channels", 128))
    blocks = int(cfg.get("blocks", 4))
    flow = InvertibleLatentAdapter(channels=channels, hidden_channels=hidden_channels, blocks=blocks)
    flow.load_state_dict(checkpoint["state_dict"])
    flow.to(device=device, dtype=torch.float32).eval()
    return flow, cfg


@torch.no_grad()
def generate_arrays(cfg: AdapterTransformRFIDConfig, run_dir: Path, device: torch.device) -> Dict[str, str | int]:
    checkpoint_path = Path(cfg.checkpoint).expanduser().resolve()
    flow, checkpoint_cfg = load_flow(checkpoint_path, device)
    model_key = cfg.model_key or str(checkpoint_cfg.get("model_key", "rae_dinov2"))

    dataset = load_named_dataset(
        cfg.dataset_name,
        cfg.data_root,
        split=cfg.dataset_split,
        dataset_path=cfg.dataset_path,
    )
    indices = contiguous_indices(len(dataset), cfg.count)
    count = len(indices)
    suffix = f"{cfg.dataset_split}_{cfg.transform}_{cfg.image_size}_n{count}"
    paths = {
        "source": run_dir / f"source_{suffix}.npy",
        "target": run_dir / f"target_{suffix}.npy",
        "recon_identity": run_dir / f"recon_identity_{suffix}.npy",
        "baseline_transform": run_dir / f"baseline_Pz_{suffix}.npy",
        "adapted_transform": run_dir / f"adapted_FinvPFz_{suffix}.npy",
    }
    if cfg.skip_generation:
        missing = [name for name, path in paths.items() if not check_npy(path, count, cfg.image_size)]
        if missing:
            raise FileNotFoundError(f"skip_generation=True but arrays are incomplete: {missing}")
        return {name: str(path) for name, path in paths.items()} | {"count": count}

    arrays = {
        name: create_or_open_npy(path, count, cfg.image_size, cfg.overwrite)
        for name, path in paths.items()
    }
    loader = DataLoader(
        IndexedCroppedImageDataset(dataset, indices, cfg.image_size),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    rae = load_rae_adapter(
        model_key,
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
    for x_cpu, source_cpu, _ in loader:
        x = x_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
        z = rae.encode(x)
        y = flow(z)
        target = P(x, cfg.transform)
        recon_identity = rae.decode(z)
        baseline_transform = rae.decode(P(z, cfg.transform))
        adapted_transform = rae.decode(flow.inverse(P(y, cfg.transform)))
        batch = x.shape[0]
        arrays["source"][offset : offset + batch] = source_cpu.numpy()
        arrays["target"][offset : offset + batch] = tensor_m11_to_uint8_nhwc(target)
        arrays["recon_identity"][offset : offset + batch] = tensor_m11_to_uint8_nhwc(recon_identity)
        arrays["baseline_transform"][offset : offset + batch] = tensor_m11_to_uint8_nhwc(baseline_transform)
        arrays["adapted_transform"][offset : offset + batch] = tensor_m11_to_uint8_nhwc(adapted_transform)
        offset += batch
        if offset % max(1, cfg.batch_size * 20) == 0 or offset == count:
            print(f"adapter rFID arrays: {offset}/{count}", flush=True)

    for arr in arrays.values():
        arr.flush()
    return {name: str(path) for name, path in paths.items()} | {"count": count}


def calculate_pair_rfid(reference_path: str, candidate_path: str, cfg: AdapterTransformRFIDConfig, device: torch.device) -> float:
    src_path = ROOT / "external" / "RAE" / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from eval.fid import calculate_rfid

    reference = np.load(reference_path, mmap_mode="r")
    candidate = np.load(candidate_path, mmap_mode="r")
    cuda_device = "cuda" if device.type == "cuda" else "cpu"
    return float(calculate_rfid(reference, candidate, bs=cfg.fid_batch_size, device=cuda_device))


def compute_metrics(paths: Dict[str, str | int], cfg: AdapterTransformRFIDConfig, device: torch.device) -> Dict[str, float]:
    return {
        "rfid_recon_identity": calculate_pair_rfid(str(paths["source"]), str(paths["recon_identity"]), cfg, device),
        "rfid_baseline_transform": calculate_pair_rfid(str(paths["target"]), str(paths["baseline_transform"]), cfg, device),
        "rfid_adapted_transform": calculate_pair_rfid(str(paths["target"]), str(paths["adapted_transform"]), cfg, device),
    }


def build_run_dir(cfg: AdapterTransformRFIDConfig) -> Path:
    name = cfg.run_name.strip()
    if not name:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        name = f"adapter_{Path(cfg.checkpoint).stem}_{cfg.dataset_split}_{cfg.transform}_n{cfg.count}_{stamp}"
    run_dir = Path(cfg.output_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run(cfg: AdapterTransformRFIDConfig) -> dict:
    configure_fp32()
    torch.set_grad_enabled(False)
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    run_dir = build_run_dir(cfg)
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)
    paths = generate_arrays(cfg, run_dir, device)
    metrics = {} if cfg.skip_rfid else compute_metrics(paths, cfg, device)
    payload = {"config": asdict(cfg), "arrays": paths, "metrics": metrics}
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps({"run_dir": str(run_dir), "metrics": metrics}, ensure_ascii=False, indent=2), flush=True)
    return payload


def parse_args() -> AdapterTransformRFIDConfig:
    parser = argparse.ArgumentParser(description="Evaluate a trained latent adapter with strict transform rFID arrays.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-name", default=AdapterTransformRFIDConfig.dataset_name)
    parser.add_argument("--data-root", default=AdapterTransformRFIDConfig.data_root)
    parser.add_argument("--dataset-path", default=AdapterTransformRFIDConfig.dataset_path)
    parser.add_argument("--dataset-split", default=AdapterTransformRFIDConfig.dataset_split)
    parser.add_argument("--image-size", type=int, default=AdapterTransformRFIDConfig.image_size)
    parser.add_argument("--count", type=int, default=AdapterTransformRFIDConfig.count)
    parser.add_argument("--transform", default=AdapterTransformRFIDConfig.transform)
    parser.add_argument("--model-key", default=AdapterTransformRFIDConfig.model_key)
    parser.add_argument("--rae-repo-path", default=AdapterTransformRFIDConfig.rae_repo_path)
    parser.add_argument("--rae-auto-clone", action="store_true")
    parser.add_argument("--rae-auto-download", action="store_true")
    parser.add_argument("--device", default=AdapterTransformRFIDConfig.device)
    parser.add_argument("--batch-size", type=int, default=AdapterTransformRFIDConfig.batch_size)
    parser.add_argument("--num-workers", type=int, default=AdapterTransformRFIDConfig.num_workers)
    parser.add_argument("--output-dir", default=AdapterTransformRFIDConfig.output_dir)
    parser.add_argument("--run-name", default=AdapterTransformRFIDConfig.run_name)
    parser.add_argument("--fid-batch-size", type=int, default=AdapterTransformRFIDConfig.fid_batch_size)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-rfid", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    return AdapterTransformRFIDConfig(
        checkpoint=args.checkpoint,
        dataset_name=args.dataset_name,
        data_root=args.data_root,
        dataset_path=args.dataset_path,
        dataset_split=args.dataset_split,
        image_size=args.image_size,
        count=args.count,
        transform=args.transform,
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
        skip_generation=args.skip_generation,
        skip_rfid=args.skip_rfid,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    run(parse_args())
