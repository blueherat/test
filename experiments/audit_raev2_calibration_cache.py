#!/usr/bin/env python3
"""Re-encode fixed real train examples to audit existing RAEv2 latent caches.

No Stage-2 model, sampling trajectory, generated image or FID is evaluated.
Existing caches are read only. Selected IDs are fixed before reading errors.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/home/zhoushunyu/data/eqvae")
for source in (ROOT / "external/RAEv2/src", ROOT):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from experiments.raev2_stage1_compat import install_raev2_decoder_config_compat
from experiments.raev2_training_core import DeterministicImageNetPacked, file_sha256
from experiments.sample_raev2_pfr_retiming import DEFAULT_CONFIG, load_config
from utils.model_utils import instantiate_from_config


DEFAULT_BANKS = {
    "A": DATA / "experiments/spectral_ig_mechanism_v3_formal_1k/01_clean_latents",
    "B": DATA / "experiments/frequency_axis_suite_v1_full/01_reference",
}
FIXED_IDS = (0, 127, 511, 999)


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def artifact(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size}


def read_selected(bank: Path) -> list[dict]:
    selected = {}
    for path in sorted(bank.glob("clean_rank*.npz")):
        with np.load(path, allow_pickle=False) as payload:
            ids = payload["ids"]
            matches = [(sample_id, np.flatnonzero(ids == sample_id)) for sample_id in FIXED_IDS]
            matches = [(sample_id, indices) for sample_id, indices in matches if len(indices)]
            if not matches:
                continue
            latents = payload["latents"]
            for sample_id, indices in matches:
                if len(indices) != 1 or sample_id in selected:
                    raise ValueError(f"duplicate sample ID {sample_id} in {bank}")
                index = int(indices[0])
                if latents.dtype != np.float16 or latents.shape[1:] != (1024, 16, 16):
                    raise ValueError(f"unexpected cache format: {path}")
                selected[sample_id] = {
                    "sample_id": sample_id, "source_row": int(payload["rows"][index]),
                    "label": int(payload["labels"][index]), "cache_path": str(path),
                    "cache_index": index, "latent": torch.from_numpy(latents[index].copy()),
                }
    if set(selected) != set(FIXED_IDS):
        raise ValueError(f"missing fixed IDs in {bank}")
    return [selected[sample_id] for sample_id in FIXED_IDS]


def comparison(fresh: torch.Tensor, cached: torch.Tensor) -> dict:
    fresh_half = fresh.to(torch.float16)
    original = fresh.double()
    half = fresh_half.double()
    old = cached.double()
    difference = half - old
    roundoff = half - original
    rms = lambda value: float(value.square().mean().sqrt())
    difference_rms, roundoff_rms = rms(difference), rms(roundoff)
    return {
        "cached_rms": rms(old), "fresh_fp32_rms": rms(original),
        "half_difference_rms": difference_rms,
        "half_difference_relative_rms": difference_rms / max(rms(old), 1e-30),
        "half_difference_max_abs": float(difference.abs().max()),
        "half_bitwise_equal_fraction": float((fresh_half.view(torch.int16) == cached.view(torch.int16)).double().mean()),
        "half_all_bitwise_equal": bool(torch.equal(fresh_half.view(torch.int16), cached.view(torch.int16))),
        "fresh_fp32_to_half_roundoff_rms": roundoff_rms,
        "fresh_fp32_to_half_roundoff_max_abs": float(roundoff.abs().max()),
        "half_difference_over_own_roundoff_rms": difference_rms / max(roundoff_rms, 1e-30),
        "fresh_fp32_vs_cached_rms": rms(original - old),
        "cached_tensor_sha256": tensor_hash(cached),
        "fresh_half_tensor_sha256": tensor_hash(fresh_half),
        "fresh_fp32_tensor_sha256": tensor_hash(fresh),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--packed-data-path", type=Path, default=Path("/data/shared/imagenet-1k/random_access_v1"))
    parser.add_argument("--dino-ckpt-dir", type=Path, default=DATA / "models/RAEv2/encoders/dinov3")
    parser.add_argument("--dino-repo-dir", type=Path, default=DATA / "models/RAEv2/dinov3_repo")
    args = parser.parse_args()
    out = args.output_dir.expanduser().resolve()
    if (out / "request.json").exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    os.environ["DINOV3_CKPT_DIR"] = str(args.dino_ckpt_dir.resolve())
    os.environ["DINOV3_REPO_DIR"] = str(args.dino_repo_dir.resolve())
    install_raev2_decoder_config_compat()
    config = load_config(args.config.resolve())
    selected = {name: read_selected(path) for name, path in DEFAULT_BANKS.items()}
    dino_weights = args.dino_ckpt_dir / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    source_paths = [Path(__file__), ROOT / "experiments/raev2_training_core.py",
                    ROOT / "experiments/raev2_stage1_compat.py",
                    ROOT / "external/RAEv2/src/encoders/vision_encoder.py",
                    ROOT / "external/RAEv2/src/encoders/models/dinov3_loader.py",
                    ROOT / "external/RAEv2/src/stage1/rae.py",
                    args.dino_repo_dir / "dinov3/models/vision_transformer.py"]
    request = {
        "protocol": "raev2_calibration_cache_reencode_v1", "fixed_ids": list(FIXED_IDS),
        "fixed_ids_selected_before_errors": True, "precision": "fp32_no_autocast_tf32_disabled",
        "encode_batch_size": 1, "split": "train", "horizontal_flip": False,
        "config": artifact(args.config), "dino_weights": artifact(dino_weights),
        "normalization_stats": artifact(Path(config.stage_1.params["normalization_stat_path"])),
        "packed_manifest": artifact(args.packed_data_path / "manifest.json"),
        "sources": {str(path): artifact(path) for path in source_paths},
        "cache_artifacts": {name: [artifact(path) for path in sorted(bank.glob("clean_rank*.npz"))]
                            for name, bank in DEFAULT_BANKS.items()},
        "selected_examples": {name: [{k: v for k, v in row.items() if k != "latent"} for row in rows]
                              for name, rows in selected.items()},
        "torch_version": torch.__version__, "sampling_performed": False,
        "fid_performed": False, "old_caches_modified": False,
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
    rows = []
    started = time.perf_counter()
    with torch.inference_mode():
        for bank_name, examples in selected.items():
            for item in examples:
                image, label, _ = dataset[item["source_row"]]
                if int(label) != item["label"]:
                    raise ValueError("packed/cache label mismatch")
                fresh = rae.encode(image.unsqueeze(0).to(device)).float().cpu()[0].contiguous()
                if not torch.isfinite(fresh).all():
                    raise FloatingPointError("nonfinite encoder output")
                row = {"bank": bank_name, **{k: v for k, v in item.items() if k != "latent"},
                       "source_image_tensor_sha256": tensor_hash(image),
                       **comparison(fresh, item["latent"])}
                rows.append(row)
                print(json.dumps(row), flush=True)
    peak = torch.cuda.max_memory_allocated()
    del rae, dataset
    gc.collect()
    torch.cuda.empty_cache()
    with (out / "rows.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "protocol": request["protocol"], "complete": True, "examples": len(rows),
        "elapsed_encode_seconds": time.perf_counter() - started,
        "peak_gpu_memory_allocated_bytes": peak,
        "by_bank": {name: {"examples": len(subset),
                           "mean_half_difference_rms": float(np.mean([r["half_difference_rms"] for r in subset])),
                           "max_half_difference_relative_rms": max(r["half_difference_relative_rms"] for r in subset),
                           "mean_half_bitwise_equal_fraction": float(np.mean([r["half_bitwise_equal_fraction"] for r in subset])),
                           "mean_difference_over_roundoff_rms": float(np.mean([r["half_difference_over_own_roundoff_rms"] for r in subset]))}
                    for name in selected if (subset := [row for row in rows if row["bank"] == name])},
        "rows": rows, "sampling_performed": False, "fid_performed": False,
        "boundary": "Fixed-example provenance check only; does not certify every cache row or generation quality.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
