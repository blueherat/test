#!/usr/bin/env python3
"""Rebuild the fixed class-balanced B calibration bank with audited FP32 encoding.

Uses original B IDs 0..999 and their unchanged real-train rows and labels.
Old caches are read only. No Stage-2 prediction, sampling or FID is involved.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.audit_raev2_calibration_cache import (
    DATA, DEFAULT_BANKS, DEFAULT_CONFIG, DeterministicImageNetPacked, artifact,
    install_raev2_decoder_config_compat, instantiate_from_config, load_config, tensor_hash,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-bank", type=Path, default=DEFAULT_BANKS["B"])
    parser.add_argument("--packed-data-path", type=Path, default=Path("/data/shared/imagenet-1k/random_access_v1"))
    parser.add_argument("--dino-ckpt-dir", type=Path, default=DATA / "models/RAEv2/encoders/dinov3")
    parser.add_argument("--dino-repo-dir", type=Path, default=DATA / "models/RAEv2/dinov3_repo")
    args = parser.parse_args()
    out = args.output_dir.expanduser().resolve()
    if (out / "request.json").exists() or (out / "clean_rank00.npz").exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    source_files = sorted(args.source_bank.glob("clean_rank*.npz"))
    metadata = {key: [] for key in ("ids", "rows", "labels")}
    for path in source_files:
        with np.load(path, allow_pickle=False) as payload:
            for key in metadata:
                metadata[key].append(payload[key])
    metadata = {key: np.concatenate(parts) for key, parts in metadata.items()}
    selected = np.flatnonzero(metadata["ids"] < 1000)
    selected = selected[np.argsort(metadata["ids"][selected])]
    metadata = {key: values[selected] for key, values in metadata.items()}
    if not np.array_equal(metadata["ids"], np.arange(1000)):
        raise ValueError("source bank must contain exactly IDs 0..999")
    if not np.array_equal(metadata["labels"], np.arange(1000)):
        raise ValueError("selected source IDs must cover each class exactly once")
    if len(np.unique(metadata["rows"])) != 1000:
        raise ValueError("source rows contain duplicates")
    a_rows = np.concatenate([np.load(path)["rows"] for path in sorted(DEFAULT_BANKS["A"].glob("clean_rank*.npz"))])
    if np.intersect1d(a_rows, metadata["rows"]).size:
        raise ValueError("new heldout rows overlap calibration bank A")

    os.environ["DINOV3_CKPT_DIR"] = str(args.dino_ckpt_dir.resolve())
    os.environ["DINOV3_REPO_DIR"] = str(args.dino_repo_dir.resolve())
    install_raev2_decoder_config_compat()
    config = load_config(args.config.resolve())
    source_paths = [Path(__file__), ROOT / "experiments/audit_raev2_calibration_cache.py",
                    ROOT / "experiments/raev2_training_core.py",
                    ROOT / "experiments/raev2_stage1_compat.py",
                    ROOT / "external/RAEv2/src/encoders/vision_encoder.py",
                    ROOT / "external/RAEv2/src/encoders/models/dinov3_loader.py",
                    ROOT / "external/RAEv2/src/stage1/rae.py",
                    args.dino_repo_dir / "dinov3/models/vision_transformer.py"]
    request = {
        "protocol": "raev2_heldout_clean_reencode_fp32_v1",
        "selection_rule": "Original bank B sample IDs 0..999, fixed before re-encoding",
        "sample_count": 1000, "num_classes": 1000, "source_split": "train",
        "heldout_means": "Disjoint from calibration bank A; not ImageNet validation split",
        "source_bank": str(args.source_bank.resolve()),
        "source_cache_artifacts": [artifact(path) for path in source_files],
        "config": artifact(args.config),
        "dino_weights": artifact(args.dino_ckpt_dir / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"),
        "normalization_stats": artifact(Path(config.stage_1.params["normalization_stat_path"])),
        "packed_manifest": artifact(args.packed_data_path / "manifest.json"),
        "sources": {str(path): artifact(path) for path in source_paths},
        "source_metadata_sha256": {key: tensor_hash(torch.from_numpy(values)) for key, values in metadata.items()},
        "source_metadata": {key: values.tolist() for key, values in metadata.items()},
        "source_latent_values_used": False, "old_caches_modified": False,
        "precision": "fp32_no_autocast_tf32_disabled", "encode_batch_size": 1,
        "horizontal_flip": False, "cache_dtype": "float16",
        "normalization_applied_by": "official RAE.encode; do not normalize saved latents again",
        "torch_version": torch.__version__, "sampling_performed": False, "fid_performed": False,
    }
    (out / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dataset = DeterministicImageNetPacked(args.packed_data_path, split="train",
                                         image_size=int(config.training.image_size), horizontal_flip=False)
    rae = instantiate_from_config(config.stage_1).to(device).eval().requires_grad_(False)
    del rae.decoder
    torch.cuda.empty_cache()
    latents = np.empty((1000, 1024, 16, 16), dtype=np.float16)
    image_hashes = []
    roundoff_sq = 0.0
    encoded_sq = 0.0
    started = time.perf_counter()
    with torch.inference_mode():
        for index, row in enumerate(metadata["rows"]):
            image, label, _ = dataset[int(row)]
            if int(label) != int(metadata["labels"][index]):
                raise ValueError(f"label mismatch at selected ID {index}")
            fresh = rae.encode(image.unsqueeze(0).to(device)).float().cpu()[0]
            if not torch.isfinite(fresh).all():
                raise FloatingPointError(f"nonfinite encoding for selected ID {index}")
            half = fresh.to(torch.float16)
            latents[index] = half.numpy()
            image_hashes.append(tensor_hash(image))
            roundoff_sq += float((fresh.double() - half.double()).square().sum())
            encoded_sq += float(fresh.double().square().sum())
            if index == 0 or (index + 1) % 64 == 0 or index == 999:
                progress = {"completed": index + 1, "total": 1000,
                            "elapsed_encode_seconds": time.perf_counter() - started}
                (out / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
                print(json.dumps(progress), flush=True)
    encode_elapsed = time.perf_counter() - started
    peak_gpu = torch.cuda.max_memory_allocated()
    del rae, dataset, fresh, half
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({"encoder_released": True, "encoding_complete": True,
                      "elapsed_encode_seconds": encode_elapsed}), flush=True)
    cache_path = out / "clean_rank00.npz"
    np.savez_compressed(cache_path, **metadata, latents=latents)
    count = latents.size
    summary = {
        "protocol": request["protocol"], "complete": True, "samples": 1000,
        "num_classes": 1000, "unique_source_rows": 1000,
        "overlap_with_bank_a_source_rows": 0,
        "metadata_identical_to_old_b_ids_below_1000": True,
        "cache": artifact(cache_path), "keys": ["ids", "rows", "labels", "latents"],
        "latent_shape": list(latents.shape), "latent_dtype": str(latents.dtype),
        "source_metadata_sha256": request["source_metadata_sha256"],
        "source_image_tensor_sha256": image_hashes,
        "encode_elapsed_seconds": encode_elapsed,
        "fp32_to_half_roundoff_rms": (roundoff_sq / count) ** 0.5,
        "fp32_to_half_roundoff_relative_rms": (roundoff_sq / encoded_sq) ** 0.5,
        "peak_gpu_memory_allocated_bytes": peak_gpu,
        "sampling_performed": False, "fid_performed": False, "old_caches_modified": False,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "source_image_tensor_sha256"}), flush=True)


if __name__ == "__main__":
    main()
